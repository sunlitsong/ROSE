import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DualWeightLinear(nn.Module):
    def __init__(self, original_linear: nn.Linear):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        
        self.register_buffer('weight', original_linear.weight.detach().clone())
        if original_linear.bias is not None:
            self.register_buffer('bias', original_linear.bias.detach().clone())
        else:
            self.register_buffer('bias', None)
            
        self.tuned_weight = nn.Parameter(original_linear.weight.detach().clone())
        
        if original_linear.bias is not None:
            self.tuned_bias = nn.Parameter(original_linear.bias.detach().clone())
        else:
            self.register_parameter('tuned_bias', None)

        self.x_storage = None  
        self.out_storage = None
        self.if_inference = False
        
        self.U_vis = None
        self.U_txt = None
        
        self.mask_vis = None
        self.indices_txt = None
        self.return_backbone = None
        
    def calculate_neuron_wise_mask(self, out_backbone, out_tuned):
        with torch.no_grad():
            diff = out_tuned - out_backbone
            neuron_scores = diff.abs() # [Batch, Seq, Dim]
            
            mean_score = neuron_scores.mean()
            std_score = neuron_scores.std()
            sigma = 1.0
            threshold = mean_score + sigma * std_score
            mask_neuron = (neuron_scores > threshold).to(out_backbone.dtype)
            return mask_neuron

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype
        x = x.to(self.weight.dtype)
        if self.return_backbone:
            return F.linear(x, self.weight, self.bias)

        if not self.if_inference:
            self.x_storage = x.detach().clone()
            out = F.linear(x, self.tuned_weight, self.tuned_bias)
            self.out_storage = out.detach().clone()
            return out.to(previous_dtype)

        else:
            U_txt = self.U_txt.to(dtype=x.dtype)
            U_vis = self.U_vis.to(dtype=x.dtype)
            
            x_txt = x[:, self.indices_txt[0]:self.indices_txt[1], :]
            query_txt = torch.mean(x_txt, dim=1) if x_txt.size(1) > 0 else torch.zeros(x.size(0), x.size(-1), device=x.device)
            query_txt = F.normalize(query_txt, p=2, dim=1)            
            
            x_vis = x[:, self.mask_vis, :]
            query_vis = torch.mean(x_vis, dim=1) if x_vis.size(1) > 0 else torch.zeros(x.size(0), x.size(-1), device=x.device)
            query_vis = F.normalize(query_vis, p=2, dim=1)        
            
            sim_matrix_txt = torch.matmul(query_txt, U_txt)
            max_sim_txt, _ = torch.max(sim_matrix_txt, dim=1) # [Batch]                
            sim_matrix_vis = torch.matmul(query_vis, U_vis)
            max_sim_vis, _ = torch.max(sim_matrix_vis, dim=1) # [Batch]            
            tau_min, tau, tau_gap = 0.4, 0.96, 0.01
            gap = (max_sim_vis - max_sim_txt).abs()
            meets_minimum = (
                (max_sim_vis > tau_min) & (max_sim_txt > tau_min)
            )
            high_confidence = (
                (max_sim_vis > tau) | (max_sim_txt > tau)
            )
            cross_modal_agreement = gap < tau_gap

            gate = meets_minimum & (high_confidence | cross_modal_agreement)
            mask_sample = gate.to(dtype=x.dtype).view(-1, 1, 1)

            out_backbone = F.linear(x, self.weight, self.bias)
            out_tuned = F.linear(x, self.tuned_weight, self.tuned_bias)

            if mask_sample.sum() == 0:
                out = out_backbone
            else:
                mask_neuron = self.calculate_neuron_wise_mask(out_backbone, out_tuned)
                
                final_mask = mask_sample * mask_neuron
                
                out = out_backbone + final_mask * (out_tuned - out_backbone)

            return out.to(previous_dtype)

def replace_layers(model, target_modules):
    for name, module in model.named_modules():
        if any(t in name for t in target_modules) and isinstance(module, nn.Linear):
            if isinstance(module, DualWeightLinear): 
                continue
            print(f"--> Replacing layer: {name}")
            if '.' in name:
                parent_name = name.rsplit('.', 1)[0]
                child_name = name.rsplit('.', 1)[1]
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                child_name = name
            new_layer = DualWeightLinear(module)
            new_layer.to(module.weight.device)
            setattr(parent, child_name, new_layer)
            
    return model