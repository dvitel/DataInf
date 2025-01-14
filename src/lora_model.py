from collections import defaultdict
from tqdm import tqdm
import pickle
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torch.autograd.functional as F
from transformers import (
    AutoModelForSequenceClassification, AutoModel,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig,
    LlamaForCausalLM,
    LlamaTokenizer
)
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model
)
from datasets import Dataset
import evaluate

    # def __init__(self, 
    #             model_name_or_path="roberta-large",
    #             target_modules=["value"],
    #             train_dataloader=None,
    #             eval_dataloader=None,
    #             device="cuda",
    #             num_epochs=10,
    #             lr=3e-4,
    #             low_rank=2,
    #             task="mrpc"):
    #     self.model_name_or_path=model_name_or_path
    #     self.target_modules=target_modules
    #     self.train_dataloader=train_dataloader
    #     self.eval_dataloader=eval_dataloader
    #     self.device=device
    #     self.num_epochs=num_epochs
    #     self.lr=lr
    #     self.task=task
    #     self.low_rank=low_rank
        
def build_LORA_model(model_name_or_path, target_modules, low_rank):
    '''
    This function fine-tunes a model for classification tasks. 
    For text generation tasks, please see `notebooks/Influential_Data_Identification-Llama2-Math.ipynb`.
    '''
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path,
                                                                    return_dict=True)
    model.config.use_cache = False
    model.config.pad_token_id = model.config.eos_token_id
        
    peft_config = LoraConfig(task_type="SEQ_CLS",
                                inference_mode=False, 
                                target_modules=target_modules,
                                r=low_rank,
                                lora_alpha=low_rank, 
                                lora_dropout=0.05)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model

def load_pretrained_LORA_model(model_name_or_path):
    '''
    This function loads a pre-trained model.
    '''
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    model = PeftModel.from_pretrained(base_model, model_name_or_path, is_trainable=True)
    model.config.use_cache = False
    model.config.pad_token_id = model.config.eos_token_id
    model.print_trainable_parameters()
    return model

def train_LORA_model(model,
        train_dataloader=None,
        eval_dataloader=None,
        device="cuda",
        num_epochs=10,
        lr=3e-4,
        task="mrpc"):
    '''
    This function fine-tunes a model for GLUE classification tasks. 
    For text generation tasks, please see `notebooks/Influential_Data_Identification-Llama2-Math.ipynb`.
    '''
    metric = evaluate.load("glue", task)
    optimizer = AdamW(params=model.parameters(), lr=lr)

    # Instantiate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0.06*(len(train_dataloader)*num_epochs),
        num_training_steps=(len(train_dataloader)*num_epochs),
    )

    model.to(device)
    eval_metrics = []
    for epoch in range(num_epochs):
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader)):
            batch.to(device)
            model.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            loss_grads = {nm: pvalue.grad.clone() for nm, pvalue in model.named_parameters() if "lora" in nm}
            # optimizer.step()
            # lr_scheduler.step()
            optimizer.zero_grad()

            # if epoch == 3:
            outputs = model(**batch) # another forward pass

            # loss2_fn = CrossEntropyLoss(reduction='none')
            # loss2 = loss2_fn(outputs.logits, batch["labels"])

            def compute_loss_func(_params, _batch):
                _output = torch.func.functional_call(model, _params, (), kwargs=_batch)
                _loss = torch.nn.functional.cross_entropy(_output.logits, _batch['labels'], reduction='none')
                return _loss

            loss2_jac_fn = torch.func.jacrev(compute_loss_func, has_aux=False)

            loss2_grads = loss2_jac_fn({nm:pval for nm, pval in model.named_parameters() if 'lora' in nm}, batch)


            # loss2_grads = {nm: torch.zeros((len(loss2), *pvalue.shape), device = loss2.device) for nm, pvalue in model.named_parameters() if "lora" in nm}
            # for i in range(len(loss2)):
            #     filter_v = torch.zeros_like(loss2)
            #     filter_v[i] = 1
            #     model.zero_grad()
            #     loss2.backward(gradient = filter_v, retain_graph = True)
            #     for nm, pvalue in model.named_parameters():
            #         if "lora" in nm:
            #             loss2_grads[nm][i] = pvalue.grad

            for nm, loss_grad in loss_grads.items():
                loss2_grad = loss2_grads[nm]
                loss2_grad_mean = torch.mean(loss2_grad, dim=0)
                loss2_grad_sum = torch.sum(loss2_grad, dim=0)
                l1 = loss2_grad.reshape(loss2_grad.shape[0], -1)
                l2 = l1.unsqueeze(1)
                l3 = l1.unsqueeze(0)
                cos_sim = torch.nn.functional.cosine_similarity(l2, l3, dim=-1)
                zero_diag = 1 - torch.eye(cos_sim.shape[0], device=cos_sim.device)
                directions = zero_diag * (cos_sim > 0)
                print(loss_grad, loss2_grad)
            # def loss2_func(params):
            #     return loss2

            # for nm, pvalue in model.named_parameters():
            #     if "lora" in nm:
            #         loss2_layer_jacobian = jacobian(loss2_func, pvalue, vectorize=True)
            #         jacobian_mean = torch.mean(loss2_layer_jacobian, dim=0)
            #         jacobian_sum = torch.sum(loss2_layer_jacobian,  dim=0)
            #         loss_grad = loss_grads[nm]
            #         pass
            # def compute_loss2(params):
            #     loss2 = loss2_fn(outputs.logits, batch["labels"])
            #     return loss2
            # loss2_jacobian = jacobian(compute_loss2, outputs.logits)
            # print(loss2_grads)
            pass

        model.eval()
        for step, batch in enumerate(tqdm(eval_dataloader)):
            batch.to(device)
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = predictions, batch["labels"]
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()
        print(f"Epoch {(epoch+1)}:", eval_metric)
        eval_metrics.append(eval_metric)
    return eval_metrics

def compute_grads(model, dataloader, device="cuda", bring_to_cpu=False):
    ''' Builds tensor of grads, collected accross the model '''
    module_grads = {}
    num_samples = len(dataloader)
    model.to(device)
    module_filter = ['lora_A', 'lora_B', 'modules_to_save.default.out_proj.weight']
    for k, v in model.named_parameters():
        if any(f in k for f in module_filter):
            grad = torch.empty((num_samples, v.numel()), device=device)
            module_grads[k] = grad
        else:
            pass         
    # collator = DataCollatorWithPadding(tokenizer, padding="longest", return_tensors="pt")
    # dataloader = DataLoader(dataset, shuffle=False, collate_fn=collate_fn, batch_size=1)        
    for step, batch in enumerate(tqdm(dataloader)):
        model.zero_grad() # zeroing out gradient
        batch.to(device)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        
        for k, v in model.named_parameters():
            if k in module_grads:
                module_grads[k][step] = v.grad.view(-1)
            else:
                pass
    if bring_to_cpu:
        for k, v in module_grads.items():
            module_grads[k] = v.cpu()
            del v
    return module_grads
    
    # def compute_gradient_old(self, tokenized_datasets, collate_fn):
    #     train_dataloader_stochastic = DataLoader(tokenized_datasets["train"], 
    #                                               shuffle=False,
    #                                               collate_fn=collate_fn,
    #                                               batch_size=1)
    #     val_dataloader_stochastic = DataLoader(tokenized_datasets["validation"], 
    #                                               shuffle=False,
    #                                               collate_fn=collate_fn,
    #                                               batch_size=1)
    #     # Compute the gradient
    #     self.model.eval()
    #     tr_grad_dict = {}
    #     for step, batch in enumerate(tqdm(train_dataloader_stochastic)):
    #         self.model.zero_grad() # zeroing out gradient
    #         batch.to(self.device)
    #         outputs = self.model(**batch)
    #         loss = outputs.loss
    #         loss.backward()
            
    #         grad_dict={}
    #         for k, v in self.model.named_parameters():
    #             if 'lora_A' in k:
    #                 grad_dict[k]=v.grad.cpu()
    #             elif 'lora_B' in k:
    #                 # first index of shape indicates low-rank
    #                 grad_dict[k]=v.grad.cpu().T
    #             elif 'modules_to_save.default.out_proj.weight' in k:
    #                 grad_dict[k]=v.grad.cpu()
    #             else:
    #                 pass
    #         tr_grad_dict[step]=grad_dict
    #         del grad_dict
            
    #     val_grad_dict = {}
    #     for step, batch in enumerate(tqdm(val_dataloader_stochastic)):
    #         self.model.zero_grad() # zeroing out gradient
    #         batch.to(self.device)
    #         outputs = self.model(**batch)
    #         loss = outputs.loss
    #         loss.backward()
            
    #         grad_dict={}
    #         for k, v in self.model.named_parameters():
    #             if 'lora_A' in k:
    #                 grad_dict[k]=v.grad.cpu()
    #             elif 'lora_B' in k:
    #                 # first index of shape indicates low-rank
    #                 grad_dict[k]=v.grad.cpu().T
    #             elif 'modules_to_save.default.out_proj.weight' in k:
    #                 grad_dict[k]=v.grad.cpu()
    #             else:
    #                 pass
    #         val_grad_dict[step]=grad_dict    
    #         del grad_dict
            
    #     return tr_grad_dict, val_grad_dict


class LORAEngineGeneration(object):
    def __init__(self, 
                base_path,
                project_path,
                dataset_name='math_with_reason',
                device="cuda"):
        self.base_path = base_path
        self.project_path = project_path
        self.adapter_path = f"{self.project_path}/models/math_with_reason_13bf"
        self.dataset_name = dataset_name
        self.device=device
        self.load_pretrained_network()
        self.load_datasets()

    def load_pretrained_network(self):
        # setup tokenizer
        self.tokenizer = LlamaTokenizer.from_pretrained(self.base_path)
        self.tokenizer.padding_side = "right"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # load a base model
        quantization_config = BitsAndBytesConfig(load_in_8bit=True, load_in_4bit=False)
        base_model = LlamaForCausalLM.from_pretrained(
            self.base_path,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            offload_folder="offload",
            offload_state_dict=True,
        )

        # load a pre-trained model.
        self.model = PeftModel.from_pretrained(base_model, self.adapter_path, is_trainable=True)
        self.finetuned_config = LoraConfig.from_pretrained(pretrained_model_name_or_path=self.adapter_path)

    def load_datasets(self):
        self.train_dataset = Dataset.load_from_disk(f"{self.project_path}/datasets/{self.dataset_name}_train.hf")
        self.validation_dataset = Dataset.load_from_disk(f"{self.project_path}/datasets/{self.dataset_name}_test.hf")

    def create_tokenized_datasets(self):
        tokenize_func = lambda x: self.tokenizer(
            x["prompt"], truncation=True, padding=True, max_length=128, return_tensors="pt" # text should be more appropritate
        ).to(self.device)

        if 'with_reason' in self.dataset_name:
            column_list=["text", "answer", "variation", "prompt", "reason"]
        else:
            column_list=["text", "answer", "variation", "prompt"]

        tokenized_datasets=dict()
        tokenized_datasets["train"] = self.train_dataset.map(
            tokenize_func,
            batched=True,
            remove_columns=column_list,
        )
        tokenized_datasets["validation"] = self.validation_dataset.map(
            tokenize_func,
            batched=True,
            remove_columns=column_list,
        )
        collate_fn = lambda x: self.tokenizer.pad(x, padding="longest", return_tensors="pt")

        return tokenized_datasets, collate_fn

    def compute_gradient(self, tokenized_datasets, collate_fn):
        train_dataloader_stochastic = DataLoader(tokenized_datasets["train"], 
                                                  shuffle=False,
                                                  collate_fn=collate_fn,
                                                  batch_size=1)
        val_dataloader_stochastic = DataLoader(tokenized_datasets["validation"], 
                                                  shuffle=False,
                                                  collate_fn=collate_fn,
                                                  batch_size=1)
        # Compute the gradient
        self.model.eval()
        tr_grad_dict = {}
        for step, batch in enumerate(tqdm(train_dataloader_stochastic)):
            self.model.zero_grad() # zeroing out gradient
            batch['labels'] = batch['input_ids']
            batch.to(self.device)
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            
            grad_dict={}
            for k, v in self.model.named_parameters():
                if 'lora_A' in k:
                    grad_dict[k]=v.grad.cpu()
                elif 'lora_B' in k:
                    # first index of shape indicates low-rank
                    grad_dict[k]=v.grad.cpu().T
                else:
                    pass
            tr_grad_dict[step]=grad_dict
            del grad_dict
            
        val_grad_dict = {}
        for step, batch in enumerate(tqdm(val_dataloader_stochastic)):
            self.model.zero_grad() # zeroing out gradient
            batch['labels'] = batch['input_ids']
            batch.to(self.device)
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            
            grad_dict={}
            for k, v in self.model.named_parameters():
                if 'lora_A' in k:
                    grad_dict[k]=v.grad.cpu()
                elif 'lora_B' in k:
                    # first index of shape indicates low-rank
                    grad_dict[k]=v.grad.cpu().T
                else:
                    pass
            val_grad_dict[step]=grad_dict    
            del grad_dict
            
        return tr_grad_dict, val_grad_dict

