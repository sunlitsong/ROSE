# ROSE: Risk-Aware Orthogonal Subspace Navigation for Lifelong Knowledge Editing in Multimodal Large Language Models

## Abstract

Multimodal Large Language Models (MLLMs) require continuous factual updates to stay current, yet lifelong knowledge editing remains a daunting challenge. The primary obstacle is the severe Locality Erosion and Catastrophic Forgetting, where sequential edits inevitably interfere with pre-trained capabilities and prior updates. In this paper, we propose **ROSE**, a **R**isk-aware **O**rthogonal **S**ubspace navigation framework for lifelong knowledge **E**diting. ROSE addresses forgetting by aiming to restrict parameter updates to the orthogonal complement of previously edited subspaces. Crucially, it introduces a risk-guided mechanism that utilizes a dynamically computed risk mask to safeguard specific factual parameter subspaces essential for model stability. During inference, ROSE employs a multi-granularity knowledge integration strategy, featuring prototype-anchored semantic gating and perturbation-driven subspace fusion to precisely activate edited knowledge for relevant queries while limiting interference on unrelated inputs. Comprehensive experiments on two prominent multimodal benchmarks show that ROSE is competitive with state-of-the-art approaches across five key metrics, while maintaining outstanding stability over $1,000$ sequential updates.


## Installation

The following commands use Bash and a CUDA-enabled PyTorch environment.

```bash
git clone https://github.com/sunlitsong/ROSE.git
cd ROSE
python -m pip install -r requirements.txt
```

## Download Datasets

- **VLKEB:** download annotations and images from the [official project](https://github.com/VLKEB/VLKEB) or its [Hugging Face dataset](https://huggingface.co/datasets/HymanH/VLKEB-data).
- **MMEdit:** download the E-IC and E-VQA annotations and the image collections linked in the [official MMEdit instructions](https://github.com/zjunlp/EasyEdit/blob/main/examples/MMEdit.md).

Place the evaluation annotations at the paths used by the corresponding entries:

```text
VLKEB/eval_edit_onehop.json
MMEdit/MMEdit-eic/caption_eval_edit.json
MMEdit/MMEdit-evqa/vqa_eval.json
```

Update the model and image paths in [`hparams/ROSE/llavaov-7b.yaml`](hparams/ROSE/llavaov-7b.yaml). For the LLaVA/VLKEB example:

```yaml
name: hugging_cache/llava-1.5-7b-hf
tokenizer_name: hugging_cache/llava-1.5-7b-hf
coco_image: VLKEB/VLKEB_images/mmkb_images
rephrase_image: VLKEB/VLKEB_images/mmkb_images
```

Set `coco_image` and `rephrase_image` to the actual image roots for the selected benchmark, so the annotation filenames resolve correctly. Archive extraction directories may differ from the example.

## Run

Select one experiment call in the `__main__` block of [`multimodal_edit.py`](multimodal_edit.py). The existing LLaVA entries are:

```python
train_test_ROSE_LLaVA_VLKEB(args)
# train_test_ROSE_LLaVA_MMEdit_caption(args)
# train_test_ROSE_LLaVA_MMEdit_vqa(args)
```

Keep only the selected call active and update the YAML paths for that dataset. The entries above read `hparams/ROSE/llavaov-7b.yaml`.

Launch from the repository root with the supplied single-GPU script:

```bash
bash run.sh
```

Equivalently:

```bash
deepspeed --num_gpus=1 multimodal_edit.py
```

This project is built on [EasyEdit](https://github.com/zjunlp/EasyEdit). Thanks for their contributions to the community!
