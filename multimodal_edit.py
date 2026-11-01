import torch
import types
from statistics import mean

from easyeditor import BaseEditor, MultimodalTrainer, MultimodalEditor
from easyeditor import CaptionDataset, VQADataset, ComprehendEditDataset
from easyeditor import MENDMultimodalTrainingHparams, SERACMultimodalTrainingHparams, IKEMultimodalHyperParams, MENDMultimodalHparams \
    , SERACMultimodalHparams, WISEMultimodalHyperParams, GraceHyperParams, LoRAMultimodalHyperParams, FTHyperParams
from easyeditor import encode_ike_facts_multimodal
from sentence_transformers import SentenceTransformer
import random
import os
import argparse 
import deepspeed 
import torch.distributed as dist
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def print_result(metrics, result_pth=None):
    local_clean_metrics = []
    for m in metrics:
        clean_m = {
            'rewrite_acc': m['post']['rewrite_acc'].item() if torch.is_tensor(m['post']['rewrite_acc']) else m['post']['rewrite_acc'],
            'rephrase_acc': m['post']['rephrase_acc'].item() if torch.is_tensor(m['post']['rephrase_acc']) else m['post']['rephrase_acc'],
            'image_rephrase_acc': m['post']['image_rephrase_acc'].item() if torch.is_tensor(m['post']['image_rephrase_acc']) else m['post']['image_rephrase_acc'],
            'locality_acc': m['post']['locality_acc'].item() if torch.is_tensor(m['post']['locality_acc']) else m['post']['locality_acc'],
            'multimodal_locality_acc': m['post']['multimodal_locality_acc'].item() if torch.is_tensor(m['post']['multimodal_locality_acc']) else m['post']['multimodal_locality_acc'],
        }
        
        if 'edit_time' in m:
            val = m['edit_time']
            clean_m['edit_time'] = val.item() if torch.is_tensor(val) else val
            
        if 'inference_time' in m:
            val = m['inference_time']
            clean_m['inference_time'] = val.item() if torch.is_tensor(val) else val

        if 'portability_acc' in m['post']:
            val = m['post']['portability_acc']
            clean_m['portability_acc'] = val.item() if torch.is_tensor(val) else val        
        local_clean_metrics.append(clean_m)

    world_size = 1
    rank = 0
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    all_metrics = []
    
    if world_size > 1:
        gathered_data = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_data, local_clean_metrics)
        for sublist in gathered_data:
            all_metrics.extend(sublist)
    else:
        all_metrics = local_clean_metrics
    if rank == 0:
        total_samples = len(all_metrics)
        print(f"------------------------------------------------------")
        print(f"Evaluated on {total_samples} samples across {world_size} GPUs")
        rewrite_acc = mean([m['rewrite_acc'] for m in all_metrics])
        rephrase_acc = mean([m['rephrase_acc'] for m in all_metrics])
        rephrase_image_acc = mean([m['image_rephrase_acc'] for m in all_metrics])
        locality_acc = mean([m['locality_acc'] for m in all_metrics])
        locality_image_acc = mean([m['multimodal_locality_acc'] for m in all_metrics])

        avg_edit_time = None
        if all_metrics and 'edit_time' in all_metrics[0]:
            avg_edit_time = mean([m['edit_time'] for m in all_metrics])

        avg_inference_time = None
        if all_metrics and 'inference_time' in all_metrics[0]:
            avg_inference_time = mean([m['inference_time'] for m in all_metrics])

        print(f'rewrite_acc: {rewrite_acc:.4f}')
        print(f'rephrase_acc: {rephrase_acc:.4f}')
        print(f'rephrase_image_acc: {rephrase_image_acc:.4f}')
        print(f'locality_acc: {locality_acc:.4f}')
        print(f'multimodal_locality_acc: {locality_image_acc:.4f}')
        portability_acc = None
        if all_metrics and 'portability_acc' in all_metrics[0]:
            portability_acc = mean([m['portability_acc'] for m in all_metrics])
            print(f'portability_acc: {portability_acc:.4f}')        
        if avg_edit_time is not None:
            print(f'edit_time: {avg_edit_time:.4f} s')
        if avg_inference_time is not None:
            print(f'inference_time: {avg_inference_time:.4f} s')
        print(f"------------------------------------------------------")
        if result_pth is not None:
            os.makedirs(os.path.dirname(result_pth), exist_ok=True)
            with open(result_pth, 'w') as f:
                f.write(f'rewrite_acc: {rewrite_acc}\n')
                f.write(f'rephrase_acc: {rephrase_acc}\n')
                f.write(f'rephrase_image_acc: {rephrase_image_acc}\n')
                f.write(f'locality_acc: {locality_acc}\n')
                f.write(f'locality_image_acc: {locality_image_acc}\n')
                
                if avg_edit_time is not None:
                    f.write(f'edit_time: {avg_edit_time}\n')
                if avg_inference_time is not None:
                    f.write(f'inference_time: {avg_inference_time}\n')
                
                if portability_acc is not None:
                    f.write(f'portability_acc: {portability_acc}\n')  

def train_test_ROSE_LLaVA_VLKEB(args):
    hparams = FTHyperParams.from_hparams('./hparams/ROSE/llavaov-7b.yaml')
    hparams.ds_args = args
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        hparams.device = args.local_rank
        print(f"[Rank {args.local_rank}] Device set to {device}")  
    if hparams.device == 0:
        print('hparams:{}'.format(hparams))
    editor = MultimodalEditor.from_hparams(hparams)
    # train_ds = CaptionDataset('VLKEB/train.json', config=hparams)
    eval_ds = CaptionDataset('VLKEB/eval_edit_onehop.json', config=hparams, size=1000, hop=1) 
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        # train_ds=eval_ds,
        keep_original_weight=True        
    )
    print_result(metrics, result_pth='./result.txt')

def train_test_ROSE_LLaVA_MMEdit_caption(args):
    hparams = FTHyperParams.from_hparams('./hparams/ROSE/llavaov-7b.yaml')
    hparams.ds_args = args
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        hparams.device = args.local_rank
        print(f"[Rank {args.local_rank}] Device set to {device}")  
    if hparams.device == 0:
        print('hparams:{}'.format(hparams))
    editor = MultimodalEditor.from_hparams(hparams)
    eval_ds = CaptionDataset('MMEdit/MMEdit-eic/caption_eval_edit.json', config=hparams, size=1000) 
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        # train_ds=eval_ds,
        keep_original_weight=True        
    )
    print_result(metrics, result_pth='./result.txt')


def train_test_ROSE_LLaVA_MMEdit_vqa(args):
    hparams = FTHyperParams.from_hparams('./hparams/ROSE/llavaov-7b.yaml')
    hparams.ds_args = args
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        hparams.device = args.local_rank
        print(f"[Rank {args.local_rank}] Device set to {device}")  
    if hparams.device == 0:
        print('hparams:{}'.format(hparams))
    editor = MultimodalEditor.from_hparams(hparams)
    eval_ds = CaptionDataset('MMEdit/MMEdit-evqa/vqa_eval.json', config=hparams, size=1000) 
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        # train_ds=eval_ds,
        keep_original_weight=True        
    )
    print_result(metrics, result_pth='./result.txt')

def train_test_FT_LLaVA_VLKEB(args):
    hparams = FTHyperParams.from_hparams('./hparams/FT/llavaov-7b.yaml')
    hparams.ds_args = args
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        hparams.device = args.local_rank
        print(f"[Rank {args.local_rank}] Device set to {device}")  
    if hparams.device == 0:
        print('hparams:{}'.format(hparams))
    editor = MultimodalEditor.from_hparams(hparams)
    # train_ds = CaptionDataset('VLKEB/train.json', config=hparams)
    eval_ds = CaptionDataset('VLKEB/eval_edit_onehop.json', config=hparams, size=1000, hop=1) 
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        # train_ds=eval_ds,
        keep_original_weight=True        
    )
    print_result(metrics, result_pth='./result.txt')


def train_test_ROSE_LLaVA_MMKE(args):
    hparams = FTHyperParams.from_hparams('./hparams/ROSE/llavaov-7b.yaml')
    hparams.ds_args = args
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        hparams.device = args.local_rank
        print(f"[Rank {args.local_rank}] Device set to {device}")  
    if hparams.device == 0:
        print('hparams:{}'.format(hparams))
    editor = MultimodalEditor.from_hparams(hparams)
    eval_ds = CaptionDataset('MMKE/data_json/visual_eval.json', config=hparams, size=1000, hop=1) 
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        # train_ds=eval_ds,
        keep_original_weight=True        
    )
    print_result(metrics, result_pth='./result.txt')
    
def train_test_WISE_LLaVA_VLKEB(args):
    hparams = WISEMultimodalHyperParams.from_hparams('./hparams/WISE/llavaov-7b.yaml')
    hparams.ds_args = args
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        hparams.device = args.local_rank
        print(f"[Rank {args.local_rank}] Device set to {device}")  
    if hparams.device == 0:
        print('hparams:{}'.format(hparams))    
    editor = MultimodalEditor.from_hparams(hparams)
    # train_ds = CaptionDataset('VLKEB/train.json', config=hparams)
    eval_ds = CaptionDataset('VLKEB/eval_edit_onehop.json', config=hparams, hop=1) 
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        # train_ds=eval_ds,
        keep_original_weight=True        
    )
    print_result(metrics)    

def train_test_WISE_LLaVA_MMKE(args):
    hparams = WISEMultimodalHyperParams.from_hparams('./hparams/WISE/llavaov-7b.yaml')
    hparams.ds_args = args
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        hparams.device = args.local_rank
        print(f"[Rank {args.local_rank}] Device set to {device}")  
    if hparams.device == 0:
        print('hparams:{}'.format(hparams))
    editor = MultimodalEditor.from_hparams(hparams)
    eval_ds = CaptionDataset('MMKE/data_json/user_eval.json', config=hparams, hop=1) 
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        # train_ds=eval_ds,
        keep_original_weight=True        
    )
    print_result(metrics, result_pth='./result.txt')
    
def train_test_GRACE_LLaVA_VLKEB(args):
    hparams = GraceHyperParams.from_hparams('./hparams/GRACE/llavaov-7b.yaml')
    hparams.ds_args = args
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        hparams.device = args.local_rank
        print(f"[Rank {args.local_rank}] Device set to {device}")  
    if hparams.device == 0:
        print('hparams:{}'.format(hparams))    
    editor = MultimodalEditor.from_hparams(hparams)
    # train_ds = CaptionDataset('VLKEB/train.json', config=hparams)
    eval_ds = CaptionDataset('VLKEB/eval_edit_onehop.json', config=hparams, hop=1) 
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        # train_ds=eval_ds,
        keep_original_weight=True        
    )
    print_result(metrics)   

def test_IKE_LLaVA_VLKEB():
    hparams = IKEMultimodalHyperParams.from_hparams('hparams/IKE/llavaov_7b.yaml')
    hparams.task_name = 'caption'
    hparams.ds_args = args
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        hparams.device = args.local_rank
        print(f"[Rank {args.local_rank}] Device set to {device}")  
    if hparams.device == 0:
        print('hparams:{}'.format(hparams))    
    train_ds = CaptionDataset('VLKEB/train.json', config=hparams)
    sentence_model = SentenceTransformer(hparams.sentence_model_name).to(f'cuda:{hparams.device}')
    encode_ike_facts_multimodal(sentence_model, train_ds, hparams)    
    editor = MultimodalEditor.from_hparams(hparams)
    # train_ds = CaptionDataset('VLKEB/train.json', config=hparams)
    eval_ds = CaptionDataset('VLKEB/eval_edit_onehop.json', config=hparams, size=1000, hop=1) 
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=eval_ds,
        keep_original_weight=True
    )
    print_result(metrics)   

def train_MEND_MiniGPT4_Caption():
    hparams = MENDMultimodalTrainingHparams.from_hparams('hparams/TRAINING/MEND/minigpt4.yaml')
    train_ds = CaptionDataset('data/caption_train_edit.json', config=hparams)
    eval_ds = CaptionDataset('data/caption_eval_edit.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    
    trainer.run()    


def train_MEND_MiniGPT4_VQA():
    hparams = MENDMultimodalTrainingHparams.from_hparams('hparams/TRAINING/MEND/minigpt4.yaml')
    train_ds = VQADataset('data/vqa_train.json', config=hparams)
    eval_ds = VQADataset('data/vqa_eval.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    
    trainer.run() 

def train_MEND_MiniGPT4_VQA_debug():
    hparams = MENDMultimodalTrainingHparams.from_hparams('hparams/TRAINING/MEND/minigpt4.yaml')
    train_ds = VQADataset('data/vqa_train.json', config=hparams, size=100)
    eval_ds = VQADataset('data/vqa_eval.json', config=hparams, size=100)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    
    trainer.run() 
  
       
def train_MEND_Blip2OPT_Caption(debug=False):
    hparams = MENDMultimodalTrainingHparams.from_hparams('hparams/TRAINING/MEND/blip2.yaml')
    if debug:
        train_ds = CaptionDataset('data/caption_train_edit.json', config=hparams, size=20)
        eval_ds = CaptionDataset('data/caption_eval_edit.json', config=hparams, size=20)
    else:
        train_ds = CaptionDataset('data/caption_train_edit.json', config=hparams)
        eval_ds = CaptionDataset('data/caption_eval_edit.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    
    trainer.run()
        
def train_MEND_Blip2OPT_VQA():
    hparams = MENDMultimodalTrainingHparams.from_hparams('hparams/TRAINING/MEND/blip2.yaml')
    train_ds = VQADataset('data/vqa_train.json', config=hparams)
    eval_ds = VQADataset('data/vqa_eval.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    
    trainer.run()   
     
def train_MEND_Blip2OPT_VQA_debug():
    hparams = MENDMultimodalTrainingHparams.from_hparams('hparams/TRAINING/MEND/blip2.yaml')
    train_ds = VQADataset('data/vqa_train.json', config=hparams, size=20)
    eval_ds = VQADataset('data/vqa_eval.json', config=hparams, size=20)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    
    trainer.run()    
    
def train_MEND_Blip2OPT_VQA_Vision_debug():
    hparams = MENDMultimodalTrainingHparams.from_hparams('hparams/TRAINING/MEND/blip2-vision.yaml')
    train_ds = VQADataset('data/vqa_train.json', config=hparams, size=20)
    eval_ds = VQADataset('data/vqa_eval.json', config=hparams, size=20)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    
    trainer.run()  
      
def train_MEND_Blip2OPT_VQA_Vision():
    hparams = MENDMultimodalTrainingHparams.from_hparams('hparams/TRAINING/MEND/blip2-vision.yaml')
    train_ds = VQADataset('data/vqa_train.json', config=hparams)
    eval_ds = VQADataset('data/vqa_eval.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    
    trainer.run()    
    
def test_MEND_MiniGPT4_VQA():
    hparams = MENDMultimodalHparams.from_hparams('hparams/MEND/minigpt4.yaml')
    # train_ds = VQADataset('data/vqa_train.json', config=hparams)
    eval_ds = VQADataset('data/vqa_eval.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    
    trainer.run()  

def test_MEND_MiniGPT4_Caption():
    hparams = MENDMultimodalHparams.from_hparams('hparams/MEND/minigpt4.yaml')
    # train_ds = CaptionDataset('data/caption_train_edit.json', config=hparams)
    eval_ds = CaptionDataset('data/caption_eval_edit.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    
    trainer.run()  

def test_MEND_Blip2OPT_VQA():
    hparams = MENDMultimodalHparams.from_hparams('hparams/MEND/blip2.yaml')
    # train_ds = VQADataset('data/vqa_train.json', config=hparams)
    eval_ds = VQADataset('data/vqa_eval.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    
    trainer.run()  

def test_MEND_Blip2OPT_Caption():
    hparams = MENDMultimodalHparams.from_hparams('hparams/MEND/blip2.yaml')
    # train_ds = CaptionDataset('data/caption_train_edit.json', config=hparams)
    eval_ds = CaptionDataset('data/caption_eval_edit.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    
    trainer.run()  
  
def test_SERAC_MiniGPT4_VQA():
    hparams = SERACMultimodalHparams.from_hparams('hparams/SERAC/minigpt4.yaml')
    # train_ds = VQADataset('data/vqa_train.json', config=hparams)
    eval_ds = VQADataset('data/vqa_eval.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    
    trainer.run()    

def test_SERAC_Blip2OPT_VQA():
    hparams = SERACMultimodalHparams.from_hparams('hparams/SERAC/blip2.yaml')
    # train_ds = VQADataset('data/vqa_train.json', config=hparams)
    eval_ds = VQADataset('data/vqa_eval.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    
    trainer.run()    

def test_SERAC_Blip2OPT_Caption():
    hparams = SERACMultimodalHparams.from_hparams('hparams/SERAC/blip2.yaml')
    # train_ds = CaptionDataset('data/caption_train_edit.json', config=hparams)
    eval_ds = CaptionDataset('data/caption_eval_edit.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    
    trainer.run()   

def test_SERAC_MiniGPT4_Caption():
    hparams = SERACMultimodalHparams.from_hparams('hparams/SERAC/minigpt4.yaml')
    # train_ds = CaptionDataset('data/caption_train_edit.json', config=hparams)
    eval_ds = CaptionDataset('data/caption_eval_edit.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    
    trainer.run()    


def train_SERAC_MiniGPT4_Caption():
    hparams = SERACMultimodalTrainingHparams.from_hparams('hparams/TRAINING/SERAC/minigpt4.yaml')
    train_ds = CaptionDataset('data/caption_train_edit.json', config=hparams)
    eval_ds = CaptionDataset('data/caption_eval_edit.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    
    trainer.run()
    
def train_SERAC_MiniGPT4_Caption_debug():
    hparams = SERACMultimodalTrainingHparams.from_hparams('hparams/TRAINING/SERAC/minigpt4.yaml')
    train_ds = CaptionDataset('data/caption_train_edit.json', config=hparams, size=5)
    eval_ds = CaptionDataset('data/caption_eval_edit.json', config=hparams, size=5)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    
    trainer.run()
    
def train_SERAC_MiniGPT4_VQA():
    hparams = SERACMultimodalTrainingHparams.from_hparams('hparams/TRAINING/SERAC/minigpt4.yaml')
    train_ds = VQADataset('data/vqa_train.json', config=hparams)
    eval_ds = VQADataset('data/vqa_eval.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    
    trainer.run()
  
    
def train_SERAC_Blip2OPT_Caption():
    hparams = SERACMultimodalTrainingHparams.from_hparams('hparams/TRAINING/SERAC/blip2.yaml')
    train_ds = CaptionDataset('data/caption_train_edit.json', config=hparams)
    eval_ds = CaptionDataset('data/caption_eval_edit.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    
    trainer.run()
    
def train_SERAC_Blip2OPT_VQA():
    hparams = SERACMultimodalTrainingHparams.from_hparams('hparams/TRAINING/SERAC/blip2.yaml')
    train_ds = VQADataset('data/vqa_train.json', config=hparams)
    eval_ds = VQADataset('data/vqa_eval.json', config=hparams)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    
    trainer.run()

def train_SERAC_Blip2OPT_Caption_debug():
    hparams = SERACMultimodalTrainingHparams.from_hparams('hparams/TRAINING/SERAC/blip2.yaml')
    # train_ds = CaptionDataset('data/caption_train_edit.json', config=hparams, size=20)
    eval_ds = CaptionDataset('data/caption_eval_edit.json', config=hparams, size=20)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    
    trainer.run()

def edit_IKE_MiniGPT4_VQA():
    prompts = [
        "How many tennis balls are in the picture?",
        "What is the red food?"
    ]
    targets = [
        "2",
        "tomatoes",
    ]
    image = [
        "val2014/COCO_val2014_000000451435.jpg",
        "val2014/COCO_val2014_000000189446.jpg"
    ]
    rephrase_prompts = [
        "What is the number of tennis balls depicted in the image?",
        "What is the name of the food that is red in color?"
    ]
    rephrase_image = [
        "val2014_image_rephrase/451435003_COCO_val2014_000000451435.png",
        "val2014_image_rephrase/189446003_COCO_val2014_000000189446.png"
    ]
    locality_inputs = {
        'text': {
            'prompt': ["nq question: what purpose did seasonal monsoon winds have on trade", "nq question: what purpose did seasonal monsoon winds have on trade",],
            'ground_truth': ["enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans", "enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans"]
        },
        'vision': {
            'prompt': ["What sport can you use this for?", "What sport can you use this for?"],
            'ground_truth': ["riding", "riding",],
            'image': ["val2014/COCO_val2014_000000297147.jpg", "val2014/COCO_val2014_000000297147.jpg"],
        }
    }
    
    hparams = IKEMultimodalHyperParams.from_hparams('hparams/IKE/minigpt4.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    train_ds = VQADataset('data/vqa_train.json', config=hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        targets=targets,
        image=image,
        rephrase_prompts=rephrase_prompts,
        rephrase_image=rephrase_image,
        locality_inputs=locality_inputs,
        train_ds=train_ds,
        keep_original_weight=True        
    )


def edit_IKE_MiniGPT4_Caption():
    prompts = [
        "a photo of",
        "a photo of"
    ]
    targets = [
        "A selection of wooden kitchen tools on a counter.",
        "Bicyclists on a city street, most not using the bike lane",
    ]
    image = [
        "val2014/COCO_val2014_000000386164.jpg",
        "val2014/COCO_val2014_000000462565.jpg"
    ]
    rephrase_prompts = [
        "provide a brief overview of the image content,",
        "describe the image content,"
    ]
    rephrase_image = [
        "val2014_image_rephrase/COCO_val2014_000000386164.png",
        "val2014_image_rephrase/COCO_val2014_000000462565.png"
    ]
    locality_inputs = {
        'text': {
            'prompt': ["nq question: what purpose did seasonal monsoon winds have on trade", "nq question: what purpose did seasonal monsoon winds have on trade",],
            'ground_truth': ["enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans", "enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans"]
        },
        'vision': {
            'prompt': ["What sport can you use this for?", "What sport can you use this for?"],
            'ground_truth': ["riding", "riding",],
            'image': ["val2014/COCO_val2014_000000297147.jpg", "val2014/COCO_val2014_000000297147.jpg"],
        }
    }
    
    hparams = IKEMultimodalHyperParams.from_hparams('hparams/IKE/minigpt4.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    train_ds = VQADataset('data/vqa_train.json', config=hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        targets=targets,
        image=image,
        rephrase_prompts=rephrase_prompts,
        rephrase_image=rephrase_image,
        locality_inputs=locality_inputs,
        train_ds=train_ds,
        keep_original_weight=True        
    )


def edit_IKE_Blip2OPT_VQA():
    prompts = [
        "How many tennis balls are in the picture?",
        "What is the red food?"
    ]
    targets = [
        "2",
        "tomatoes",
    ]
    image = [
        "val2014/COCO_val2014_000000451435.jpg",
        "val2014/COCO_val2014_000000189446.jpg"
    ]
    rephrase_prompts = [
        "What is the number of tennis balls depicted in the image?",
        "What is the name of the food that is red in color?"
    ]
    rephrase_image = [
        "val2014_image_rephrase/451435003_COCO_val2014_000000451435.png",
        "val2014_image_rephrase/189446003_COCO_val2014_000000189446.png"
    ]
    locality_inputs = {
        'text': {
            'prompt': ["nq question: what purpose did seasonal monsoon winds have on trade", "nq question: what purpose did seasonal monsoon winds have on trade",],
            'ground_truth': ["enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans", "enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans"]
        },
        'vision': {
            'prompt': ["What sport can you use this for?", "What sport can you use this for?"],
            'ground_truth': ["riding", "riding",],
            'image': ["val2014/COCO_val2014_000000297147.jpg", "val2014/COCO_val2014_000000297147.jpg"],
        }
    }
    
    hparams = IKEMultimodalHyperParams.from_hparams('hparams/IKE/blip2.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    train_ds = VQADataset('data/vqa_train.json', config=hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        targets=targets,
        image=image,
        rephrase_prompts=rephrase_prompts,
        rephrase_image=rephrase_image,
        locality_inputs=locality_inputs,
        train_ds=train_ds,
        keep_original_weight=True        
    )


def edit_MEND_Blip2OPT_VQA_debug():
    prompts = [
        "Question: How many tennis balls are in the picture? Short answer:",
        "Question: What is the red food? Short answer:"
    ]
    targets = [
        "2",
        "tomatoes",
    ]
    image = [
        "val2014/COCO_val2014_000000451435.jpg",
        "val2014/COCO_val2014_000000189446.jpg"
    ]
    rephrase_prompts = [
        "What is the number of tennis balls depicted in the image?",
        "What is the name of the food that is red in color?"
    ]
    rephrase_image = [
        "val2014_image_rephrase/451435003_COCO_val2014_000000451435.png",
        "val2014_image_rephrase/189446003_COCO_val2014_000000189446.png"
    ]
    locality_inputs = {
        'text': {
            'prompt': ["nq question: what purpose did seasonal monsoon winds have on trade", "nq question: what purpose did seasonal monsoon winds have on trade",],
            'ground_truth': ["enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans", "enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans"]
        },
        'vision': {
            'prompt': ["What sport can you use this for?", "What sport can you use this for?"],
            'ground_truth': ["riding", "riding",],
            'image': ["val2014/COCO_val2014_000000297147.jpg", "val2014/COCO_val2014_000000297147.jpg"],
        }
    }
    
    hparams = MENDMultimodalHparams.from_hparams('hparams/MEND/blip2.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        targets=targets,
        image=image,
        rephrase_prompts=rephrase_prompts,
        rephrase_image=rephrase_image,
        locality_inputs=locality_inputs,
        keep_original_weight=True        
    )

    print_result(metrics)


def edit_SERAC_Blip2OPT_VQA_debug():
    prompts = [
        "Question: How many tennis balls are in the picture? Short answer:",
        "Question: What is the red food? Short answer:"
    ]
    targets = [
        "2",
        "tomatoes",
    ]
    image = [
        "val2014/COCO_val2014_000000451435.jpg",
        "val2014/COCO_val2014_000000189446.jpg"
    ]
    rephrase_prompts = [
        "What is the number of tennis balls depicted in the image?",
        "What is the name of the food that is red in color?"
    ]
    rephrase_image = [
        "val2014_image_rephrase/451435003_COCO_val2014_000000451435.png",
        "val2014_image_rephrase/189446003_COCO_val2014_000000189446.png"
    ]
    locality_inputs = {
        'text': {
            'prompt': ["nq question: what purpose did seasonal monsoon winds have on trade", "nq question: what purpose did seasonal monsoon winds have on trade",],
            'ground_truth': ["enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans", "enabled European empire expansion into the Americas and trade routes to become established across the Atlantic and Pacific oceans"]
        },
        'vision': {
            'prompt': ["What sport can you use this for?", "What sport can you use this for?"],
            'ground_truth': ["riding", "riding",],
            'image': ["val2014/COCO_val2014_000000297147.jpg", "val2014/COCO_val2014_000000297147.jpg"],
        }
    }
    
    hparams = SERACMultimodalHparams.from_hparams('hparams/SERAC/blip2.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        targets=targets,
        image=image,
        rephrase_prompts=rephrase_prompts,
        rephrase_image=rephrase_image,
        locality_inputs=locality_inputs,
        keep_original_weight=True        
    )

    print_result(metrics)


def edit_SERAC_Blip2OPT_VQA():
    
    hparams = SERACMultimodalHparams.from_hparams('hparams/SERAC/blip2.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    eval_ds = VQADataset('data/vqa_eval.json', config=hparams)
    template = "Question: {} Short answer:"
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        keep_original_weight=True,
        template=template       
    )

    print_result(metrics)


def edit_MEND_Blip2OPT_VQA():
    
    hparams = MENDMultimodalHparams.from_hparams('hparams/MEND/blip2.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    eval_ds = VQADataset('data/vqa_eval.json', config=hparams, size=100)
    template = "Question: {} Short answer:"
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        keep_original_weight=True,
        template=template       
    )

    print_result(metrics)


def test_IKE_Blip2OPT_Caption():
    
    hparams = IKEMultimodalHyperParams.from_hparams('hparams/IKE/blip2.yaml')
    hparams.task_name = 'caption'
    editor = MultimodalEditor.from_hparams(hparams)
    train_ds = CaptionDataset('data/caption_train_edit.json', config=hparams)
    eval_ds = CaptionDataset('data/caption_eval_edit.json', config=hparams)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=train_ds,
        keep_original_weight=True        
    )
    
    print_result(metrics)

def test_IKE_Blip2OPT_VQA():
    
    hparams = IKEMultimodalHyperParams.from_hparams('hparams/IKE/blip2.yaml')
    hparams.task_name = 'vqa'
    editor = MultimodalEditor.from_hparams(hparams)
    train_ds = VQADataset('data/vqa_train.json', config=hparams)
    eval_ds = VQADataset('data/vqa_eval.json', config=hparams)
    template = "Question: {} Short answer:"
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=train_ds,
        keep_original_weight=True,
        template=template        
    )
    
    print_result(metrics)
    
def Generate_Embedding_for_IKE():
    
    hparams = IKEMultimodalHyperParams.from_hparams('hparams/IKE/blip2.yaml')
    train_ds = VQADataset('data/vqa_train.json', config=hparams)
    ## Generate embedding files for IKE
    sentence_model = SentenceTransformer(hparams.sentence_model_name).to(f'cuda:{hparams.device}')
    encode_ike_facts_multimodal(sentence_model, train_ds, hparams)
    
def test_IKE_MiniGPT4_Caption():
    
    hparams = IKEMultimodalHyperParams.from_hparams('hparams/IKE/minigpt4.yaml')
    hparams.task_name = 'caption'
    editor = MultimodalEditor.from_hparams(hparams)
    eval_ds = CaptionDataset('data/caption_eval_edit.json', config=hparams)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=eval_ds,
        keep_original_weight=True        
    )
    
    print_result(metrics)
    
def test_IKE_MiniGPT4_VQA_debug():
    
    hparams = IKEMultimodalHyperParams.from_hparams('hparams/IKE/minigpt4.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    train_ds = VQADataset('data/vqa_train.json', config=hparams, size=20)
    eval_ds = VQADataset('data/vqa_eval.json', config=hparams, size=20)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=train_ds,
        keep_original_weight=True        
    )
    
    print_result(metrics)
    
def test_IKE_MiniGPT4_VQA():
    
    hparams = IKEMultimodalHyperParams.from_hparams('hparams/IKE/minigpt4.yaml')
    hparams.task_name = 'vqa'
    editor = MultimodalEditor.from_hparams(hparams)
    # train_ds = VQADataset('data/vqa_train.json', config=hparams, size=5)
    eval_ds = VQADataset('data/vqa_eval.json', config=hparams)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=eval_ds,
        keep_original_weight=True        
    )
    
    print_result(metrics)
    
def test_IKE_Blip2OPT_VQA_debug():
    
    hparams = IKEMultimodalHyperParams.from_hparams('hparams/IKE/blip2.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    train_ds = VQADataset('data/vqa_train.json', config=hparams, size=100)
    eval_ds = VQADataset('data/vqa_eval.json', config=hparams, size=100)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds=train_ds,
        keep_original_weight=True        
    )
    
    print_result(metrics)


DATASET_DICT = {
    'vqa': VQADataset,
    'comprehendedit': ComprehendEditDataset,
}
data_dir = {
    'vqa': './MMEdit',
    'comprehendedit': './ComprehendEdit',
}
def train_SERAC(model='blip2', dataset='vqa', train=True, topk=5):
    hparams = SERACMultimodalTrainingHparams.from_hparams('EasyEdit/hparams/TRAINING/SERAC/{}.yaml'.format(model))
    Dataset = DATASET_DICT[dataset.lower()]
    print(f'Using datasets: {dataset.lower()}')

    eval_ds = Dataset(data_dir[dataset.lower()], config=hparams, size=10, mode='test', topk=topk)
    train_ds = Dataset(data_dir[dataset.lower()], config=hparams, size=10, mode='train', topk=-1) if train else eval_ds
    
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )

    if train:
        print('Start Training...')
        trainer.run()
    else:
        print('Start Testing...')
        # make sure hparams.archive is not null
        # assert hparams.archive is not None
        val_steps = len(eval_ds._data)+1
        val_info = trainer.validate(log=True)
        trainer.echo(val_steps, val_info, pretty=True)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1, 
                        help='local rank passed from distributed launcher')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()    
    
    # train_test_ROSE_LLaVA_MMEdit_caption(args)
    # train_test_ROSE_LLaVA_MMEdit_vqa(args)
    train_test_ROSE_LLaVA_VLKEB(args)