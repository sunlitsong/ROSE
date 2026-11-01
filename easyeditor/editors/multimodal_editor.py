from ..dataset.processor.blip_processors import BlipImageEvalProcessor
from ..dataset.processor.llavaov_processors import LLaVAOneVisionProcessor
from ..dataset.processor.qwen2vl_processors import Qwen2VLProcessor
from .editor import BaseEditor
import os.path
import copy
from typing import Optional, Union, List, Tuple, Dict
from time import time
from torch.utils.data import Dataset
from tqdm import tqdm
import json
import torch
import logging
import numpy as np
import math
import random
from PIL import Image

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import GPT2TokenizerFast, GPT2Tokenizer
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, Qwen2VLForConditionalGeneration, LlavaForConditionalGeneration

from ..util.globals import *
from .batch_editor import BatchEditor
from ..evaluate import (compute_icl_multimodal_edit_quality, 
                        compute_multimodal_edit_results,
                        compute_multimodal_edit_results_demo,
                        compute_multimodal_hf_edit_results)
from ..util import nethook
from ..util.hparams import HyperParams
from ..util.alg_dict import *
from ..models.lora.peft_ours import get_peft_model, AdaLoraConfig, TaskType, get_peft_model_state_dict, set_peft_model_state_dict, LoraConfig
from ..models.lora.our_manager import OurManager
from ..models.ourmodule.our_layers import replace_layers

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

LOG = logging.getLogger(__name__)


def make_logs():

    f_h, s_h = get_handler("logs/", log_name='run.log')
    LOG.addHandler(f_h)
    LOG.addHandler(s_h)


class MultimodalEditor:
    """Multimodal editor for all methods"""
    
    @classmethod
    def from_hparams(cls, hparams: HyperParams):

        return cls(hparams)

    def __init__(self,
                hparams: HyperParams,
                 ):

        assert hparams is not None or print('Error: hparams is None.')

        self.model_name = hparams.model_name
        self.apply_algo = ALG_MULTIMODAL_DICT[hparams.alg_name]
        self.alg_name = hparams.alg_name
        self.manager = None # Initialize manager as None
        self.model_engine = None

        make_logs()

        LOG.info("Instantiating model")

        if type(self.model_name) is str:
            if hparams.model_name == "blip2":
                from ..trainer.blip2_models import Blip2OPT
                
                self.model = Blip2OPT(
                    vit_model="eva_clip_g",
                    img_size=364,
                    use_grad_checkpoint=True,
                    vit_precision="fp32",
                    freeze_vit=True,
                    opt_model=hparams.name,
                    state_dict_file=hparams.state_dict_file,
                    qformer_name_or_path=hparams.qformer_name_or_path,
                    qformer_checkpoint=hparams.qformer_checkpoint
                )
                vis_processor = BlipImageEvalProcessor(image_size=364, mean=None, std=None)
                self.vis_tok = vis_processor
            elif hparams.model_name == "minigpt4":
                from ..trainer.blip2_models import MiniGPT4
                
                self.model = MiniGPT4(
                    vit_model="eva_clip_g",
                    qformer_checkpoint=hparams.qformer_checkpoint,
                    img_size=364,
                    use_grad_checkpoint=True,
                    vit_precision="fp32",
                    freeze_vit=True,
                    llama_model=hparams.name,
                    state_dict_file=hparams.state_dict_file,
                    qformer_name_or_path=hparams.qformer_name_or_path,
                    pretrained_ckpt=hparams.pretrained_ckpt,
                )
                vis_processor = BlipImageEvalProcessor(image_size=364, mean=None, std=None)
                self.vis_tok = vis_processor
            elif "llava-onevision" in hparams.model_name.lower():   
                self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                    hparams.model_name,
                    torch_dtype=torch.float32,
                    # attn_implementation="flash_attention_2" 
                )
                self.vis_tok = LLaVAOneVisionProcessor()
                self.tok = AutoProcessor.from_pretrained(hparams.model_name)
                self.model_name = "llava-onevision"
            elif "llava" in hparams.model_name.lower(): # Standard LLaVA
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    hparams.name, 
                    torch_dtype=torch.bfloat16, 
                    low_cpu_mem_usage=True, 
                ) # .to(device=hparams.device) will be called later if not lora
                vis_processor = transformers.CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
                self.vis_tok = lambda image: vis_processor(image, return_tensors='pt')['pixel_values'].to(dtype=torch.bfloat16)
            elif "qwen2-vl" in hparams.model_name.lower():
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    hparams.model_name, 
                    torch_dtype=torch.float32,
                    # attn_implementation="flash_attention_2"
                )
                self.vis_tok = Qwen2VLProcessor()
                self.tok = AutoProcessor.from_pretrained(hparams.model_name)
                self.model_name = "qwen2-vl"
            elif "qwen-vl" in hparams.model_name.lower(): # Old Qwen-VL
                tokenizer = AutoTokenizer.from_pretrained(hparams.name, trust_remote_code=True, pad_token='<|endoftext|>')
                self.model = AutoModelForCausalLM.from_pretrained(
                    hparams.name, trust_remote_code=True
                )
                vis_processor = BlipImageEvalProcessor(image_size=448, mean=None, std=None)
                self.vis_tok = vis_processor
            elif "owl-2" in hparams.model_name.lower():
                from ..trainer.mPLUG_Owl2.mplug_owl2.model.builder import load_pretrained_model
                from ..trainer.mPLUG_Owl2.mplug_owl2.model.modeling_mplug_owl2 import replace_llama_modality_adaptive
                replace_llama_modality_adaptive()
                tokenizer , self.model, _, _ = load_pretrained_model(hparams.name, None, 'mplug_owl2', load_8bit=False, load_4bit=False, device=f"cuda:{hparams.device}")
                for param in self.model.parameters():
                    param.requires_grad = True
                from transformers.models.clip.image_processing_clip import CLIPImageProcessor
                vis_processor = CLIPImageProcessor.from_pretrained(hparams.name, trust_remote_code=True)
                self.vis_tok = vis_processor
            else:
                raise NotImplementedError(f"Model {hparams.model_name} not supported")

            # Tokenizer handling if not already set by specific models (like LLaVA-OV/Qwen2)
            if not hasattr(self, 'tok') or self.tok is None:
                if (hparams is not None and hasattr(hparams, 'tokenizer_name')):
                    tok_name = (
                        hparams.tokenizer_name
                        if hparams.tokenizer_name is not None
                        else hparams.name
                    )
                    # Try AutoProcessor first, then fallback to AutoTokenizer or specified class
                    try:
                         tokenizer = AutoProcessor.from_pretrained(tok_name)
                    except:
                        if hasattr(hparams, 'tokenizer_class'):
                             tokenizer = getattr(transformers, hparams.tokenizer_class).from_pretrained(tok_name)
                        else:
                             tokenizer = AutoTokenizer.from_pretrained(tok_name)
                    
                    if hasattr(tokenizer, 'pad_token') and (tokenizer.pad_token == None or tokenizer.pad_token == ''):
                        tokenizer.pad_token = tokenizer.eos_token
                    self.tok = tokenizer
                else:
                    raise NotImplementedError('missing tokenizer_name in hparams')                   
        else:
            self.model, self.tok = self.model_name

        self.hparams = hparams
        self.vis_root = hparams.coco_image
        self.rephrase_root = hparams.rephrase_image

            
        LOG.info("Replacing target modules with DualWeightLinear layers...")
        self.model = replace_layers(self.model, hparams.target_modules)
        self.model.to(f'cuda:{hparams.device}')

        for n, p in self.model.named_parameters():
            if 'tuned_weight' in n or 'tuned_bias' in n:
                p.requires_grad = True
            else:
                p.requires_grad = False

        self.manager = OurManager(hparams)
        
        self.model_engine = self.model 
        self.manager.set_model_engine(self.model_engine)  

    def edit(self,
            prompts: Union[str, List[str]],
            targets: Union[str, List[str]],
            image: Union[str, List[str]],
            file_type: Union[str,List[str]] = None,
            rephrase_prompts: Optional[Union[str, List[str]]] = None,
            rephrase_image: Optional[Union[str, List[str]]] = None,
            locality_inputs: Optional[dict] = None,
            keep_original_weight=False,
            verbose=True,
            sequential_edit=False,
            **kwargs
            ):
        """
        `prompts`: list or str
            the prompts to edit
        `targets`: str
            the expected outputs
        `image`: dict
            for multimodal
        """
        if isinstance(prompts, List):
            assert len(prompts) == len(targets) == len(image)
        else:
            prompts, targets, image = [prompts,], [targets,], [image,]
        
        if file_type is None:
            file_type = [None for _ in range(len(prompts))]

        if hasattr(self.hparams, 'batch_size'):  # For Singleton Editing, bs=1
            self.hparams.batch_size = 1

        requests = self._prepare_requests(prompts, targets, image, rephrase_prompts, rephrase_image, locality_inputs, file_type, **kwargs)

        if hasattr(self.hparams, 'batch_size') :
               assert self.hparams.batch_size == 1

        all_metrics = []

        # --- Single / Batch Editing Logic ---
        for i, request in enumerate(requests):
            start = time()

            if self.alg_name == 'IKE':
                pass 
            else:
                edited_model, weights_copy = self.apply_algo(
                    self.model,
                    self.tok,
                    [request],
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=keep_original_weight,
                    manager=self.manager,
                    train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None
                )
            
            exec_time = time() - start
            LOG.info(f"Execution {i} editing took {exec_time}")

            start = time()
            if self.model_name in ['minigpt4', 'blip2'] and self.alg_name != 'IKE':
                eval_func = compute_multimodal_edit_results
            else:
                eval_func = compute_multimodal_hf_edit_results
            
            metrics = {
                'case_id': i,
                "time": exec_time,
                "post": eval_func(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device),
            }
            metrics["pre"] = eval_func(self.model, self.model_name, self.hparams, self.tok, request, self.hparams.device)

            all_metrics.append(metrics)

        return all_metrics, edited_model, weights_copy

    def edit_dataset(self,
                        ds: Dataset,
                        keep_original_weight=False,
                        verbose=True,
                        cur_time='',
                        **kwargs
                        ):
            from torch.utils.data import DataLoader
            import gc 
            
            batch_size = getattr(self.hparams, 'batch_size', 1)
            dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x, num_workers=4)
            all_metrics = []
            port_result = []
            history_records = []
            
            current_ds_idx = 0
            
            if self.manager is not None:
                self.manager.set_stage(stage='train')
                self.manager.set_branch_requires_grad(state=True)
            
            disable_tqdm = False 
            if self.manager is None:
                self.model_copy = copy.deepcopy(self.model_engine)
            
            for i, batch_requests in tqdm(enumerate(dataloader), desc='Editing dataset', total=len(dataloader), ncols=120, disable=disable_tqdm):
                start_edit_time = time()
                
                if self.manager is not None:
                    self.manager.before_edit(edit_iter=i) 
                    self.manager.decouple_branch(state=False)
                
                edited_model, _ = self.apply_algo(
                    self.model_engine,
                    self.tok,
                    batch_requests,
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=keep_original_weight,
                    manager=self.manager, 
                    train_ds=None
                )
                
                if self.manager is not None:
                    self.manager.after_edit()

                batch_exec_time = time() - start_edit_time
                for _ in batch_requests:
                    global_idx = len(history_records) 
                    current_record = {
                        "case_id": global_idx,
                        "dataset_idx": current_ds_idx,
                        "edit_time": batch_exec_time 
                    }
                    history_records.append(current_record)
                    current_ds_idx += 1
                
                if ((i + 1) % self.hparams.test_gap == 0) or ((i + 1) == len(dataloader)) or (i == 0) or ((i + 1) == 100):
                    if self.manager is not None:
                        self.manager.set_stage(stage='inf')
                        self.manager.set_branch_requires_grad(state=False)
                    metrics_eval = []
                    for hist_i, record in tqdm(enumerate(history_records), desc='Evaluating iter{}'.format(i+1), total=len(history_records), ncols=120, disable=disable_tqdm):
                        hist_req = ds[record['dataset_idx']]

                        if self.manager is not None:
                            self.manager.decouple_branch(state=True)
                        
                        with torch.no_grad():
                            pre_res = compute_multimodal_hf_edit_results(self.model_engine, self.model_name, self.hparams, self.tok, hist_req, self.hparams.device, pre_edit=True, manager=self.manager)
                        
                        if self.manager is not None:
                            self.manager.decouple_branch(state=False)
                        
                        start_inf_time = time()
                        with torch.no_grad():
                            post = compute_multimodal_hf_edit_results(self.model_engine if self.manager is not None else edited_model, 
                                                                    self.model_name, self.hparams, self.tok, hist_req, self.hparams.device, pre_edit=False, manager=self.manager)
                        end_inf_time = time()
                        inference_time = end_inf_time - start_inf_time
                        metrics = {
                            'case_id': hist_i,
                            "post": post,
                            "pre": pre_res,
                            "edit_time": record['edit_time'],  
                            "inference_time": inference_time  
                        }
                        # Metric calc logic
                        if 'locality_output' in metrics['post'].keys():
                            assert len(metrics['post']['locality_output']) == \
                                    len(metrics['pre']['locality_output'])
                            base_logits = metrics['pre']['locality_output'].to(torch.float32)
                            post_logits = metrics['post']['locality_output'].to(torch.float32)
                            if post_logits.shape[1] > base_logits.shape[1]:
                                post_logits = post_logits[:, -base_logits.shape[1]:, :]
                            else:
                                base_logits = base_logits[:, -post_logits.shape[1]:, :]
                            base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits, dim=-1), k=1, dim=-1).indices
                            post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_logits, dim=-1), k=1, dim=-1).indices
                            metrics['post']['locality_acc'] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
                            metrics['post'].pop('locality_output')
                            metrics['pre'].pop('locality_output')
                            
                        if 'multimodal_locality_output' in metrics['post'].keys():
                            assert len(metrics['post']['multimodal_locality_output']) == \
                                    len(metrics['pre']['multimodal_locality_output'])
                            base_image_logits = metrics['pre']['multimodal_locality_output'].to(torch.float32)
                            post_image_logits = metrics['post']['multimodal_locality_output'].to(torch.float32)
                            if post_image_logits.shape[1] > base_image_logits.shape[1]:
                                post_image_logits = post_image_logits[:, -base_image_logits.shape[1]:, :]
                            else:
                                base_image_logits = base_image_logits[:, -post_image_logits.shape[1]:, :]

                            base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_image_logits, dim=-1), k=10, dim=-1).indices
                            post_image_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_logits, dim=-1), k=10, dim=-1).indices
                            metrics['post']['multimodal_locality_acc'] = sum(post_image_base_logits_softmax_top_k.view(-1) == base_image_logits_softmax_top_k.view(-1))/post_image_base_logits_softmax_top_k.view(-1).shape[0]
                            metrics['post'].pop('multimodal_locality_output')
                            metrics['pre'].pop('multimodal_locality_output')

                        if metrics['post'].get('portability_acc') is not None:
                            port = {
                                'edit_input': ' '.join([hist_req["prompt"], hist_req["target"]]),
                                'port_input': ' '.join([hist_req['portability_prompt'][0], hist_req['portability_ground_truth'][0]]) if len(hist_req['portability_prompt']) > 0 else "",
                                'port_acc': metrics['post']['portability_acc'],
                            }
                            port_result.append(port)
                        
                        if (i + 1) == len(dataloader):  
                            all_metrics.append(metrics)
                        metrics_eval.append(metrics)                        
                        del hist_req, pre_res, post, metrics

                    if self.manager is not None:
                        self.manager.decouple_branch(state=False) 
                        self.manager.set_stage(stage='train')
                        self.manager.set_branch_requires_grad(state=True)
                    from multimodal_edit import print_result
                    print_result(metrics_eval)                        
                    
                    gc.collect()
                    torch.cuda.empty_cache()
            return all_metrics, edited_model, _

    def _chunks(self, arr, n):
        """Yield successive n-sized chunks from arr."""
        for i in range(0, len(arr), n):
            yield arr[i: i + n]
                    
    def _init_ds(self, ds: Dataset):
        """Init ds to inputs format."""
        data = {
            'prompts': [],
            'targets': [],
            'image': [],
            'rephrase_prompts': [],
            'rephrase_image': [],
            'locality_inputs': {'text': {'prompt': [], 'ground_truth': []}, 'vision': {'image': [], 'prompt': [], 'ground_truth': []}}
        }
        
        for record in ds:
            data['prompts'].append(record['src'])
            data['targets'].append(record['alt'])
            data['image'].append(record['image'])
            data['rephrase_prompts'].append(record['rephrase'])
            data['rephrase_image'].append(record['image_rephrase'])
            data['locality_inputs']['text']['prompt'].append(record['loc'])
            data['locality_inputs']['text']['ground_truth'].append(record['loc_ans'])
            data['locality_inputs']['vision']['image'].append(record['m_loc'])
            data['locality_inputs']['vision']['prompt'].append(record['m_loc_q'])
            data['locality_inputs']['vision']['ground_truth'].append(record['m_loc_a'])
            
        return data
    
    def _prepare_requests(self,
                          prompts: Union[str, List[str]],
                          targets: Union[str, List[str]],
                          image: Union[str, List[str]],
                          rephrase_prompts: Optional[Union[str, List[str]]] = None,
                          rephrase_image: Optional[Union[str, List[str]]] = None,
                          locality_inputs: Optional[dict] = None,
                          **kwargs
                          ):
        if isinstance(image, str):
            image = [image, ]
        image_path = [os.path.join(self.vis_root, image_) for image_ in image]
        image = [Image.open(ip).convert("RGB") for ip in image_path]
        image = [self.vis_tok(i).to(self.hparams.device) for i in image]
        
        requests = [{
            'prompt': prompt,
            'target': target,
            'image': image_,
        }        
        for prompt, target, image_ in zip(prompts, targets, image)
        ]
        
        if "text" in locality_inputs.keys():
            locality_prompts = locality_inputs['text']['prompt']
            locality_ground_truth = locality_inputs['text']['ground_truth']
            if isinstance(locality_prompts, str):
                locality_prompts = [locality_prompts, ]
            if isinstance(locality_ground_truth, str):
                locality_ground_truth = [locality_ground_truth, ]
            assert len(locality_inputs['text']['prompt']) == len(locality_inputs['text']['ground_truth']) \
                == len(requests) or print('One Edit instance needs one locality input.....')
        if "vision" in locality_inputs.keys():
            multimodal_locality_prompts = locality_inputs['vision']['prompt']
            multimodal_locality_ground_truth = locality_inputs['vision']['ground_truth']
            multimodal_locality_image = locality_inputs['vision']['image']
            if isinstance(multimodal_locality_prompts, str):
                multimodal_locality_prompts = [multimodal_locality_prompts, ]
            if isinstance(multimodal_locality_ground_truth, str):
                multimodal_locality_ground_truth = [multimodal_locality_ground_truth, ]
            if isinstance(multimodal_locality_image, str):
                multimodal_locality_image = [multimodal_locality_image, ]
            assert len(locality_inputs['vision']['prompt']) == len(locality_inputs['vision']['ground_truth']) \
                == len(locality_inputs['vision']['image']) == len(requests) or print('One Edit instance needs one locality input.....')

        if rephrase_prompts is not None:
            if isinstance(rephrase_prompts, str):
                rephrase_prompts = [rephrase_prompts,]

            for i, request in enumerate(requests):
                request.update(
                    {
                        'rephrase_prompt': rephrase_prompts[i],
                    }
                )
        if rephrase_image is not None:
            if isinstance(rephrase_image, str):
                rephrase_image = [rephrase_image, ]
            rephrase_image_path = [os.path.join(self.rephrase_root, rephrase_image_) for rephrase_image_ in rephrase_image]
            rephrase_image = [Image.open(ip).convert("RGB") for ip in rephrase_image_path]
            rephrase_image = [self.vis_tok(i).to(self.hparams.device) for i in rephrase_image]
            
            for i, request in enumerate(requests):
                request.update(
                    {
                        'image_rephrase': rephrase_image[i],
                    }
                )
        
        if "text" in locality_inputs.keys():
            
            for i, request in enumerate(requests):
                request.update(
                    {
                        'locality_prompt': locality_prompts[i],
                        'locality_ground_truth': locality_ground_truth[i]
                    }
                )
        
        if "vision" in locality_inputs.keys():
            
            locality_image_path = [os.path.join(self.vis_root, multimodal_locality_image_) for multimodal_locality_image_ in multimodal_locality_image]
            locality_image = [Image.open(ip).convert("RGB") for ip in locality_image_path]
            locality_image = [self.vis_tok(i).to(self.hparams.device) for i in locality_image]
             
            for i, request in enumerate(requests):
                request.update(
                    {
                        'multimodal_locality_image': locality_image[i],
                        'multimodal_locality_prompt': multimodal_locality_prompts[i],
                        'multimodal_locality_ground_truth': multimodal_locality_ground_truth[i],
                    }
                )
            
        return requests


# if __name__ == "__main__":
#
#     editor = BaseEditor(alg_name='KN', model_name='/nature/peng/serac/hugging_cache/t5-3b-finetuned-counterfact-10000', hparams_fname='t5-3b.json')
#
#     editor.edit(
#         prompts='What university did Watts Humphrey attend?',
#         ground_truth='Illinois Institute of Technology',
#         target_new='University of Michigan'
#     )
#
#     metrics, edited_model, _ = editor.edit(prompts='What university did Watts Humphrey attend?', ground_truth='Illinois Institute of Technology', target_new='University of Michigan')


