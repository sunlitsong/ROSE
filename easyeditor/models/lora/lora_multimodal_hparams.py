from dataclasses import dataclass
from typing import List, Any
from ...util.hparams import HyperParams
import yaml


@dataclass
class LoRAMultimodalHyperParams(HyperParams):
    # Method
    lora_type: str
    layers: List[int]
    num_steps: int
    lr: float
    sh_lr: float
    weight_decay: float
    kl_factor: float
    norm_constraint: float
    target_modules: List[str]
    rank: int
    lora_alpha: float
    lora_dropout: float
    # Module templates

    device: int
    alg_name: str
    model_name: str
    
    coco_image: str
    rephrase_image: str    
    tokenizer_class: str
    tokenizer_name: str
    model_class: str
    name: str
    dropout: float
    inner_params: List[str]
    no_grad_layers: Any
    alg: str
    results_dir: str

    mom2_dataset: str
    mom2_n_samples: int
    P_loc: str
    mom2_dtype: str
    stats_dir: str
    nullspace_threshold: float
    test_gap: int

    # Defaults
    pretrained_lora_pth: str = ''
    load_pretrained_lora: bool = False
    batch_size: int = 128
    max_length: int = 40
    model_parallel: bool = False
    use_chat_template: bool = True
    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'LoRA') or print(
            f'LoRAHyperParams can not load from {hparams_name_or_path}, '
            f'alg_name is {config["alg_name"]} ')
        return cls(**config)
