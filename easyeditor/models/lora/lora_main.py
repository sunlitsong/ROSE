import os
from copy import deepcopy
from typing import Any, Dict, List, Tuple
from .peft_ours import get_peft_model, AdaLoraConfig, TaskType, get_peft_model_state_dict, set_peft_model_state_dict, LoraConfig
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

from .lora_hparams import LoRAHyperParams
from .lora_multimodal_hparams import LoRAMultimodalHyperParams
from ...util import nethook
import deepspeed

# load from rome
from pathlib import Path
from datasets import load_dataset
from datasets import load_from_disk
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from ...util.globals import *
from ...util.nethook import Trace, set_requires_grad
from ...util.runningstats import CombinedStat, Mean, NormMean, SecondMoment, tally
from ..rome.tok_dataset import (
    TokenizedDataset,
    dict_to_,
    flatten_masked_batch,
    length_collation,
)
from ..ourmodule.our_layers import DualWeightLinear 

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
P_loaded = False
STAT_TYPES = {
    "mom2": SecondMoment,
    "mean": Mean,
    "norm_mean": NormMean,
}

def apply_lora_to_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: LoRAHyperParams,
        copy=False,
        return_orig_weights=False,
        keep_original_weight=False,
        **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    pass
def execute_lora(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: LoRAHyperParams,
        keep_original_weight=False,
        **kwargs: Any,
) -> Dict[str, Tuple[torch.Tensor]]:
    pass

def apply_lora_to_multimodal_model(
        model,
        opt,
        scheduler,
        tok: AutoProcessor,
        manager,
        requests: List[Dict],
        hparams: LoRAMultimodalHyperParams,
        copy=False,
        return_orig_weights=False,
        keep_original_weight=False,
        **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    
    weights_copy = {}
    if copy:
        model = deepcopy(model)
        
    manager.set_branch_requires_grad(state=True)
    edited_model = execute_multimodal_lora(model, opt, scheduler, tok, manager, requests, hparams, keep_original_weight)

    return edited_model, weights_copy

def execute_multimodal_lora(
        model_engine,
        opt, 
        scheduler,
        processor: AutoProcessor,
        manager,
        requests: List[Dict],
        hparams: LoRAMultimodalHyperParams,
        keep_original_weight=False,
        **kwargs: Any,
):
    """
    Executes the Branch Tuning update algorithm (formerly LoRA).
    """      
    requests = deepcopy(requests)
    device = torch.device(f'cuda:{hparams.device}')
    
    prompts = [r["prompt"] for r in requests]
    labels = [r["target"] for r in requests]
    input_images = [r['image'] for r in requests]
    loss_meter = AverageMeter()
    

    # --- Training Loop ---
    for it in range(hparams.num_steps):
        loss_meter.reset()
        file_type = 'image'

        for txt, tgt in zip(chunks(prompts, hparams.batch_size), chunks(labels, hparams.batch_size)):
            model_engine.zero_grad()
            
            if hasattr(hparams, 'use_chat_template') and hparams.use_chat_template:
                 if file_type == "multi-image":
                    num_images = len(input_images[0])
                 else:
                    num_images = 1
                 temp_prompt = [processor.apply_chat_template([
                                            {
                                                "role": "user",
                                                "content": [{"type": "image"}] * num_images + [{"type": "text", "text": p}],
                                            },
                                        ], add_generation_prompt=True, tokenize=False) + l
                                for p, l in zip(txt, tgt)]
            else:
                 temp_prompt = [f"{p} {l}" for p, l in zip(txt, tgt)]

            # Training Batch Construction
            multimodal_inputs = processor(
                images=input_images, 
                text=temp_prompt, 
                return_tensors="pt", 
                padding="longest", 
                max_length=2048, 
                truncation=True
            ).to(device)
            
            tokens = multimodal_inputs
            targets = processor(labels[0], add_special_tokens=False, return_tensors="pt", padding=True, max_length=multimodal_inputs["input_ids"].size(1))["input_ids"].to(device)
            labels_ids = torch.full_like(multimodal_inputs["input_ids"], -100).to(device)
            labels_ids[:, -targets.size(1):] = targets
            tokens["labels"] = labels_ids
            tokens = tokens.to(device)
            bs = tokens["input_ids"].shape[0]

            # Token loc for decoupling
            if manager.if_split_token is False:
                manager.set_stage('train') # Ensure hooks are enabled
                manager.get_token_loc(processor.tokenizer.convert_tokens_to_ids(processor.image_token), multimodal_inputs, processor)
                manager.if_split_token = True
                manager.attn_mask = multimodal_inputs["attention_mask"]

            pred = model_engine(**tokens)
            loss = pred.loss
            # loss = loss + manager.get_orth_loss()
            loss_meter.update(loss.item(), n=bs)

            model_engine.backward(loss)
            
            
            model_engine.step()
        if loss_meter.avg  < 1e-2:
            print("Loss {}, early stop".format(loss.item()))
            break

        if not it % 10:
            if hparams.ds_args.local_rank == 0:
                print(f"Total loss {loss_meter.avg}")
                        
    return model_engine

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    chunk = []
    for a in arr:
        chunk.append(a)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk