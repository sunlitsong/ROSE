import torch
import torch.nn as nn
import sys
import os
import math
import torch.nn.functional as F
from copy import deepcopy

class OurManager:
    def __init__(self, hparams):
        self.local_rank = getattr(hparams, 'local_rank', 0) if hasattr(hparams, 'local_rank') else 0
        if hasattr(hparams, 'ds_args') and hasattr(hparams.ds_args, 'local_rank'):
             self.local_rank = hparams.ds_args.local_rank

        self.target_layers = hparams.target_modules

        self.current_U_vis = {}
        self.current_U_txt = {}
        self.proj_U_vis = {}
        self.proj_U_txt = {}
        
        self.tuned_weight_pre = None
        
        self.model_engine = None
        self.attn_mask = None
        self.mask_vis = None
        self.start_idx_t = -1
        self.end_idx_t = -1
        self.if_split_token = False
        self.edit_iter = 0

    def set_model_engine(self, model_engine):
        self.model_engine = model_engine
        for name, module in self.model_engine.named_modules():
            if hasattr(module, 'tuned_weight'):
                self.current_U_vis[name] = None
                self.current_U_txt[name] = None
                self.proj_U_vis[name] = None
                self.proj_U_txt[name] = None

    def get_token_loc(self, multimodal_inputs, processor): 
        image_token_str = getattr(processor, 'image_token', '<image>') 
        if hasattr(processor, 'image_token_id'):
            img_token_id = processor.image_token_id
        else:
            img_token_id = processor.tokenizer.convert_tokens_to_ids(image_token_str)            
        input_ids = multimodal_inputs['input_ids']
        img_ids = torch.tensor(img_token_id).to(input_ids.device)
        if input_ids.dim() == 2: 
            input_ids = input_ids.squeeze(0)                
        self.mask_vis = (img_ids == input_ids).nonzero(as_tuple=True)[0]
        if len(self.mask_vis) == 0:
            self.mask_vis = torch.tensor([], device=input_ids.device)
            return 

        start_idx = self.mask_vis[-1].item() + 1
        separator = "ASSISTANT:" 
        sep_ids = processor.tokenizer.encode(separator, add_special_tokens=False)  
        if len(sep_ids) > 0 and sep_ids[0] == 29871: sep_ids = sep_ids[1:]
        
        end_idx = -1   
        seq_len = input_ids.size(0)
        while start_idx < seq_len and input_ids[start_idx].item() in [13, 29871]:
            start_idx += 1
        seperator_len = len(sep_ids)
        seperator_tensor = torch.tensor(sep_ids, device=input_ids.device)
        
        for i in range(start_idx, seq_len - seperator_len + 1):
            if torch.equal(input_ids[i : i + seperator_len], seperator_tensor):
                end_idx = i
                break
        
        if end_idx == -1:
             end_idx = start_idx
             
        while end_idx > start_idx and input_ids[end_idx - 1].item() in  [13, 29871]:  
            end_idx -= 1
        
        self.start_idx_t = start_idx
        self.end_idx_t = end_idx
        
        for name, module in self.model_engine.named_modules():
            if hasattr(module, 'mask_vis'):
                module.mask_vis = self.mask_vis
                module.indices_txt = (self.start_idx_t, self.end_idx_t)
    
    def set_stage(self, stage):
        for name, module in self.model_engine.named_modules():
            if hasattr(module, 'if_inference'):
                if stage == 'train':
                    module.if_inference = False
                elif stage == 'inf':
                    module.if_inference = True

    def decouple_branch(self, state):
        for name, module in self.model_engine.named_modules():
            if hasattr(module, 'return_backbone'):
                module.return_backbone = state

    def set_branch_requires_grad(self, state):
        for name, module in self.model_engine.named_modules():
            if hasattr(module, 'tuned_weight'):
                module.tuned_weight.requires_grad = state
                if module.tuned_bias is not None:
                    module.tuned_bias.requires_grad = state

    def calculate_neuron_wise_sensitivity(self, weight_backbone, weight_tuned):
        with torch.no_grad():
            diff = (weight_tuned.to(torch.float32) - weight_backbone.to(torch.float32)).squeeze()
            neuron_scores = diff.abs()
            if torch.sum(neuron_scores).item() != 0.0:
                k = int(neuron_scores.numel() * 0.5)
                threshold = torch.topk(neuron_scores.view(-1), k).values[-1]
                mask_neuron = neuron_scores > threshold
                return mask_neuron
            else:
                return torch.zeros_like(diff, dtype=torch.bool)

    def flatten_masked_batch(self, data, mask):
        flat_data = data.view(-1, data.size(-1))
        attended_tokens = mask.view(-1).nonzero()[:, 0]
        return flat_data[attended_tokens]

    def update_proj_mat(self, U_dict, layer, features, type): 
        if features.size(0) == 0: 
            return
        if type == 'txt':
            Vh = features[-1, :].unsqueeze(0)
        elif type == 'vis':
            Vh = torch.mean(features, dim=0, keepdim=True)
        else:
            raise ValueError("Unknown type")
        Vh = F.normalize(Vh, p=2, dim=1)
                        
        if U_dict[layer] is None:
            U_dict[layer] = Vh.T 
        else:
            U_curr = U_dict[layer].to(Vh.device)
            U_new = Vh - ((Vh @ U_curr) @ U_curr.T)
            if torch.norm(U_new, p=2) > 1e-2:
                U_new = F.normalize(U_new, p=2, dim=1)
                U_new = U_new.T
                U_combined = torch.cat([U_dict[layer], U_new], dim=1)  
                U_dict[layer] = U_combined
            else:
                print('[WARNING] L2 norm of U_new is LOW!!!')
                pass

    def update_U_dict(self, U_dict, layer, features):
        if features.size(0) == 0: 
            return
        Vh = torch.mean(features, dim=0, keepdim=True)
        Vh = F.normalize(Vh, p=2, dim=1)
                        
        if U_dict[layer] is None:
            U_dict[layer] = Vh.T 
        else:
            U_new = Vh.T
            U_combined = torch.cat([U_dict[layer], U_new], dim=1)  
            U_dict[layer] = U_combined        
        
    def update_subspace(self):
        for name, module in self.model_engine.named_modules():
            if not hasattr(module, 'x_storage'): 
                continue
            
            device = module.tuned_weight.device 
            x_inter = self.flatten_masked_batch(module.x_storage.to(device), self.attn_mask).float()
            
            valid_vis_mask = self.mask_vis < x_inter.shape[0]
            vis_indices = self.mask_vis[valid_vis_mask]
            x_vis = x_inter[vis_indices, :]

            start_idx = self.start_idx_t
            end_idx = self.end_idx_t
            txt_indices = torch.arange(start_idx, end_idx, device=x_inter.device)
            x_txt = x_inter[txt_indices, :]

            self.update_U_dict(self.current_U_vis, name, x_vis)
            self.update_U_dict(self.current_U_txt, name, x_txt)
            self.update_proj_mat(self.proj_U_vis, name, x_vis, 'vis')
            self.update_proj_mat(self.proj_U_txt, name, x_txt, 'txt')
            
    def fix_gradient(self):
        if self.edit_iter > 0:
            for name, module in self.model_engine.named_modules():
                if not hasattr(module, 'tuned_weight'): 
                    continue
                if module.tuned_weight.grad is None:
                    continue
                out_backbone = module.x_storage @ module.weight.T
                neuron_mask = self.calculate_neuron_wise_sensitivity(module.weight, module.tuned_weight.detach().clone())
                    
                grad = module.tuned_weight.grad
                
                U_vis = self.proj_U_vis.get(name)
                U_txt = self.proj_U_txt.get(name)
                
                U_list = []
                if U_vis is not None: 
                    U_list.append(U_vis.to(grad.device))
                if U_txt is not None: 
                    U_list.append(U_txt.to(grad.device))
                
                if len(U_list) > 0:
                    U_merged = torch.cat(U_list, dim=1) 
                    Q, _ = torch.linalg.qr(U_merged, mode='reduced') 
                    Q = Q.to(dtype=grad.dtype)
                    G_Q = torch.matmul(grad, Q)
                    grad_proj = grad - torch.matmul(G_Q, Q.T)
                    grad_new = torch.zeros_like(grad)
                    grad_new[neuron_mask] = grad_proj[neuron_mask]
                    
                    module.tuned_weight.grad = grad_new
                
    def before_edit(self, edit_iter):
        self.edit_iter = edit_iter
        for name, module in self.model_engine.named_modules():
            if hasattr(module, 'tuned_weight'):
                self.tuned_weight_pre = module.tuned_weight.detach().clone()
                module.x_storage = None
                module.edit_iter = edit_iter

    def after_edit(self):
        self.update_subspace()
        for name, module in self.model_engine.named_modules():
            if hasattr(module, 'tuned_weight'):
                module.U_vis = self.current_U_vis.get(name)
                module.U_txt = self.current_U_txt.get(name)
                
        self.if_split_token = False