from copy import deepcopy
from typing import Any, Dict, List, Tuple
from collections import deque

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...util import nethook 
from .ft_hparams import FTHyperParams

def apply_ft_to_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer, 
        requests: List[Dict],
        hparams: FTHyperParams, 
        copy=False,
        return_orig_weights=False,
        keep_original_weight=False, 
        manager=None,
        **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) the weights that changed
    """
    weights_copy = {}
    if copy:
        model = deepcopy(model)
    
    deltas = execute_ft(model, tok, requests, hparams, manager=manager) 

    with torch.no_grad():
        for w_name, upd_matrix in deltas.items():
            w = nethook.get_parameter(model, w_name)
            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()
            w[...] += upd_matrix

    if not deltas and manager is not None:
         print(f"Weights successfully updated in-place via Manager for {len(requests)} requests")
    else:
         print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model, weights_copy


def execute_ft(
        model: AutoModelForCausalLM,
        processor, 
        requests: List[Dict],
        hparams: FTHyperParams, 
        manager=None,
        **kwargs: Any,
) -> Dict[str, Tuple[torch.Tensor]]: 
    """
    Executes the fine-tuning algorithm.
    """
    model.config.use_cache = False
    model.supports_gradient_checkpointing = True
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    device = torch.device(f'cuda:{hparams.device}')
    # model.to(dtype=torch.float32)

    requests = deepcopy(requests)
    
    prompts = [r["prompt"] for r in requests]
    labels = [r["target"] for r in requests]
    input_images = [r['image'] for r in requests] if 'image' in requests[0] else None
    
    if manager is not None:
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=hparams.lr,
            weight_decay=hparams.weight_decay if hasattr(hparams, 'weight_decay') else 0,
        )
    else:
        weights = {
            n: p
            for n, p in model.named_parameters()
            for layer in hparams.layers
            if hparams.rewrite_module_tmp.format(layer) in n
        }
        weights_copy = {k: v.detach().clone() for k, v in weights.items()}
        opt = torch.optim.AdamW(
            [v for _, v in weights.items()],
            lr=hparams.lr,
            weight_decay=hparams.weight_decay,
        )
        for name, w in model.named_parameters():
            w.requires_grad = name in weights

    loss_meter = AverageMeter()
    
    for it in range(hparams.num_steps):
        loss_meter.reset()
        
        for batch_idx, (txt_batch, tgt_batch) in enumerate(zip(
                chunks(prompts, hparams.batch_size), chunks(labels, hparams.batch_size)
        )):
            mask_token = -100
            opt.zero_grad()
            
            current_batch_images = None
            current_batch_videos = None 

            if hasattr(hparams, 'use_chat_template') and hparams.use_chat_template:
                start_idx = batch_idx * hparams.batch_size
                end_idx = start_idx + len(txt_batch)
                if input_images: 
                    current_batch_images = input_images[start_idx:end_idx] 

                num_images = 0
                file_type = "image"
                if current_batch_images and len(current_batch_images) > 0:
                    if file_type == "multi-image":
                        num_images = len(current_batch_images[0])
                    else:
                        num_images = 1
                
                temp_prompt = [processor.apply_chat_template([
                                        {
                                            "role": "user",
                                            "content": [{"type": "image"}] * num_images + [{"type": "text", "text": p}],
                                        },
                                    ],
                                                    add_generation_prompt=True,
                                                    tokenize=False)  + l
                                for p, l in zip(txt_batch, tgt_batch)]              
                
                full_prompt = temp_prompt
                if file_type in ["image", "single-image", "multi-image"]:
                    multimodal_inputs = processor(images=current_batch_images, text=full_prompt, return_tensors="pt", padding=True).to(device, dtype=torch.float32)
                elif file_type == "video":
                    multimodal_inputs = processor(videos=current_batch_videos, text=full_prompt, return_tensors="pt", padding=True).to(device, dtype=torch.float32)
                    
                tokens = multimodal_inputs
                            
            tokenized_labels = processor.tokenizer(tgt_batch, add_special_tokens=False,
                                                  return_tensors="pt", padding=True, 
                                                  max_length=tokens["input_ids"].size(1))["input_ids"] 
    
            labels_ids = torch.full_like(tokens["input_ids"], mask_token)
            for i, target_ids in enumerate(tokenized_labels):
                labels_ids[i, -target_ids.size(0):] = target_ids
            
            tokens["labels"] = labels_ids
            bs = tokens["input_ids"].shape[0]

            # --- Manager Hooks (Reference lora_main) ---
            if manager is not None:
                if manager.if_split_token is False:
                    manager.set_stage('train') 
                    manager.get_token_loc(multimodal_inputs, processor)
                    manager.if_split_token = True
                    manager.attn_mask = multimodal_inputs["attention_mask"]

            # Forward
            pred = model(**tokens)
            loss = pred.loss
            loss_meter.update(loss.item(), n=bs)

            loss.backward()
            
            if manager is not None:
                manager.fix_gradient()
                
            opt.step()

            if manager is None and type(hparams.norm_constraint) is float:
                eps = hparams.norm_constraint
                with torch.no_grad():
                    for k, v in weights.items():
                        v[...] = torch.clamp(
                            v, min=weights_copy[k] - eps, max=weights_copy[k] + eps
                        )

        print(f"Epoch {it} Loss: {loss_meter.avg}")
        if loss_meter.avg < 1e-2:
            print("Early stop due to low loss")
            break

    if manager is not None:
        return {}
    else:
        deltas = {k: (weights[k] - weights_copy[k]).detach() for k in weights}
        with torch.no_grad():
            for k, v in weights.items():
                v[...] = weights_copy[k]
        return deltas


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