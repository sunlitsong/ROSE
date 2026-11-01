"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict

from .processor.base_dataset import BaseDataset
from .processor.blip_processors import BlipImageEvalProcessor
from ..trainer.utils import dict_to
from PIL import Image
import random
import typing
import torch
import transformers
from transformers import AutoTokenizer
from tqdm import tqdm
from copy import deepcopy

from ..trainer.mPLUG_Owl2.mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from ..trainer.mPLUG_Owl2.mplug_owl2.mm_utils import tokenizer_image_token, process_images
import torch.distributed as dist

def random_sample(data, ratio):
    n = len(data)
    k = round(n * ratio)
    indexes = list(range(n))
    selected_indexes = random.sample(indexes, k)
    selected_indexes.sort()
    return [data[i] for i in selected_indexes]

class CaptionDataset(BaseDataset):
    def __init__(self, data_dir: str, size:  typing.Optional[int] = None, config=None, no_image=False, hop=None, *args, **kwargs):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        random.seed(42) 
        # get tokenizer and vis_processor
        if config.model_class == "Blip2OPT":
            vis_processor = BlipImageEvalProcessor(image_size=364, mean=None, std=None)
        elif config.model_class == "LLaVA":
            vis_processor = transformers.CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        elif config.model_class ==  "qwen-vl":
            vis_processor = BlipImageEvalProcessor(image_size=448, mean=None, std=None)
        elif "owl-2" in config.model_name.lower():
            from transformers.models.clip.image_processing_clip import CLIPImageProcessor
            vis_processor = CLIPImageProcessor.from_pretrained(config.name, trust_remote_code=True)
        else:
            raise NotImplementedError("unknown model class")

        if (config is not None and hasattr(config, 'tokenizer_name')):
            tok_name = config.tokenizer_name if config.tokenizer_name is not None else config.name
            if config.tokenizer_class == "QWenTokenizer":
                tokenizer = AutoTokenizer.from_pretrained(config.name, trust_remote_code=True, pad_token='<|endoftext|>')
            elif config.model_name == "owl-2":
                tokenizer = AutoTokenizer.from_pretrained(config.name, use_fast=False, trust_remote_code=True)
            else:
                tokenizer = getattr(transformers, config.tokenizer_class).from_pretrained(tok_name, trust_remote_code=True)            
            if tokenizer.pad_token is None or tokenizer.pad_token == '':
                tokenizer.pad_token = tokenizer.eos_token  
                
        vis_root = config.coco_image
        rephrase_root = config.rephrase_image
        super().__init__(vis_processor, vis_root, rephrase_root, [data_dir])

        self.config = config
        self.tok = tokenizer
        self.max_length = 32

        # Hop logic
        port_type = None
        data = []
        if size is not None:
            self.annotation = self.annotation[:size]
        if hop:
            self.hop = hop
            assert int(hop) in [1, 2, 3, 4], "hop should be 1, 2, 3, or 4"
            port_types_list = ['', '1-hop', '2-hop', '3-hop', '4-hop']
            port_type = port_types_list[int(hop)]
        # self.annotation = random_sample(self.annotation, 0.5) # Uncomment if needed
        random.shuffle(self.annotation)
        disable_tqdm = (config.ds_args.local_rank != -1) and (config.ds_args.local_rank != 0)
        for record in tqdm(self.annotation, ncols=120, desc='Loading Data', disable=disable_tqdm):
            
            if record['alt'] == "":
                continue
            if hop and 'port_new' not in record.keys():
                continue
            
            image_path = os.path.join(self.vis_root, record["image"])
            rephrase_image_path = os.path.join(self.rephrase_root, record["image_rephrase"])
            locality_image_path = os.path.join(self.vis_root, record['m_loc'])
            raw_k_type = record.get('knowledge_type', 'entity_level')
            item = {
                'prompt': record['src'],
                'pred': record['pred'],
                'target': record['alt'],
                'rephrase_prompt': record['rephrase'],
                'image': image_path,
                'image_rephrase': rephrase_image_path,
                'cond': "{} >> {} || {}".format(record['pred'], record['alt'], record['src']),
                'locality_prompt': record['loc'],
                'locality_ground_truth': record['loc_ans'],
                'multimodal_locality_image': locality_image_path,
                'multimodal_locality_prompt': record['m_loc_q'],
                'multimodal_locality_ground_truth': record['m_loc_a']
            }

            if raw_k_type == 'visual_knowledge':
                item['knowledge_type'] = 2
                item['one_hop_img'] = os.path.join(self.vis_root, record['one_hop_img'])
                item['image_rephrase_question'] = record.get('image_rephrase_question', '')
                item['rel_prompt'] = record.get('rel')
                item['rel_ground_truth'] = record.get('rel_ans')
                item['m_rel_prompt'] = record.get('m_rel')
                item['m_rel_ground_truth'] = record.get('m_rel_ans')
            elif raw_k_type == 'user_specific':
                item['knowledge_type'] = 1
                item['one_hop_img'] = os.path.join(self.vis_root, record['one_hop_img'])
                item['rel_prompt_1'] = record.get('rel_1')
                item['rel_ground_truth_1'] = record.get('rel_ans_1', record.get('rel_1_ans'))
                item['rel_prompt_2'] = record.get('rel_2')
                item['rel_ground_truth_2'] = record.get('rel_ans_2', record.get('rel_2_ans'))
                item['m_rel_prompt_1'] = record.get('m_rel_1')
                item['m_rel_ground_truth_1'] = record.get('m_rel_ans_1', record.get('m_rel_1_ans'))
                item['m_rel_prompt_2'] = record.get('m_rel_2')
                item['m_rel_ground_truth_2'] = record.get('m_rel_ans_2', record.get('m_rel_2_ans'))
            else: # entity_level
                item['knowledge_type'] = 0
                if 'rel_1' in record or 'rel' in record:
                    item['rel_prompt_1'] = record.get('rel_1')
                    item['rel_ground_truth_1'] = record.get('rel_ans_1', record.get('rel_1_ans'))
                    item['rel_prompt_2'] = record.get('rel_2')
                    item['rel_ground_truth_2'] = record.get('rel_ans_2', record.get('rel_2_ans'))
                    item['m_rel_prompt_1'] = record.get('m_rel_1')
                    item['m_rel_ground_truth_1'] = record.get('m_rel_ans_1', record.get('m_rel_1_ans'))
                    item['m_rel_prompt_2'] = record.get('m_rel_2')
                    item['m_rel_ground_truth_2'] = record.get('m_rel_ans_2', record.get('m_rel_2_ans'))

            if hop and 'port_new' in record.keys():
                item['portability_prompt'] = []
                item['portability_ground_truth'] = []
                find_hop = False
                for ports in record['port_new']:
                    current_port_type = ports.get('port_type')
                    if current_port_type is not None and current_port_type != port_type:
                        continue
                    q = ports.get('Q&A', {}).get('Question', ports.get('port_q'))
                    a = ports.get('Q&A', {}).get('Answer', ports.get('port_a'))
                    if q and a:
                        find_hop = True
                        item['portability_prompt'].append(q)
                        item['portability_ground_truth'].append(a)
                        if current_port_type is not None: break 
                if not find_hop: continue

            data.append(item)
            
        self._data = data
        self.no_image = no_image
        self.global_len = len(self._data)
        rank = config.ds_args.local_rank
        world_size = torch.cuda.device_count()
        if world_size > 1:
            divisible_len = (self.global_len // world_size) * world_size
            self._data = self._data[:divisible_len]
            self._data = self._data[rank::world_size]
            print(f"Dataset Sharded: Total {self.global_len} -> Each GPU handles approx {len(self._data)}")        

    def __getitem__(self, index):
        if self.no_image:
            return self._data[index]

        data = deepcopy(self._data[index])        
        image_path = data['image']
        rephrase_image_path = data['image_rephrase']
        locality_image_path = data['multimodal_locality_image']
        
        image = Image.open(image_path).convert("RGB")
        rephrase_image = Image.open(rephrase_image_path).convert("RGB")
        locality_image = Image.open(locality_image_path).convert("RGB")
        one_hop_img = None
        if 'one_hop_img' in data:
            one_hop_img = Image.open(data['one_hop_img']).convert("RGB")
        
        # Qwen needs paths, others keep PIL
        if self.config.model_class == "qwen-vl":
            image = image_path
            rephrase_image = rephrase_image_path
            locality_image = locality_image_path
            if one_hop_img: one_hop_img = data['one_hop_img']

        data['image'] = image
        data['image_rephrase'] = rephrase_image
        data['multimodal_locality_image'] = locality_image
        if one_hop_img is not None:
            data['one_hop_img'] = one_hop_img

        return data
    
    def __len__(self):
        return len(self._data)