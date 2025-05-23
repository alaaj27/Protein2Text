
# [NAACL 2025] Protein2Text: Resampling Mechanism to Translate Protein Sequences into Human-Interpretable Text
## **📌 Introduction**

<div align=center><img src="figures/Archi.png" width="100%" height="100%" /></div>

Proteins are fundamental to biological processes, yet **99.7% of the 227 million known protein sequences remain uncharacterized** due to experimental limitations. **Protein2Text** is a **multimodal large language model** designed to bridge this gap by **interpreting protein sequences** and **generating informative text-based answers** to open-ended scientific questions.  

Built on an **adapted LLaVA framework** with an advanced **resampling mechanism**, Protein2Text effectively translates protein sequences into a language-compatible space, allowing it to **handle diverse and complex biological queries**.  






## **🚀 Features**  
✔ **Multimodal Capability**: Interprets protein sequences and generates textual insights.  
✔ **LLaVA-Based Architecture**: Integrates a resampling mechanism for improved protein-to-text mapping.  
✔ **Curated Training Data**: Derived from **PubMed articles** to enhance domain-specific knowledge.  
✔ **Comprehensive Benchmarking**: Evaluated on **six rigorous benchmarks**, including in-domain, cross-domain, zero-shot, and classification benchmarks.  
✔ **Open-Ended Q&A**: Outperforms existing models in generating informative and relevant protein function insights.  
✔ **Publicly Available**: Model weights, evaluation datasets, and evaluation scripts are **fully open-source**.  

## **📊 Performance Highlights**  
Protein2Text has been **rigorously evaluated** using multiple benchmarks and has demonstrated **superior performance** in **open-ended question-answering** compared to existing models.  

🔹 **Outperforms several baselines** in biological Q&A tasks.  
🔹 **Addresses limitations** in existing evaluation metrics for template-based methods.  
🔹 **Proposes improved assessment strategies** to reduce bias in evaluating protein-related NLP models. 

## Release

- [03/7/25] The evaluation datasets of Protein2Text are released [here](https://github.com/alaaj27/Protein2Text.git), evaluation scripts are released [here](https://github.com/alaaj27/Protein2Text.git), and LoRA-based model weights 🤗 are released [here](https://huggingface.co/tumorailab/protein2text-llama3.1-8B-instruct-esm2-650M).

- [02/12/25] [Protein2Text](https://github.com/alaaj27/Protein2Text.git) is accepted by NAACL 2025 -Industry Track.


[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)

**Usage and License Notices**: This project utilizes certain datasets and checkpoints that are subject to their respective original licenses. Users must comply with all terms and conditions of these original licenses, including but not limited to [LLaVA](https://github.com/haotian-liu/LLaVA/blob/main/LICENSE) and the specific licenses for base language models for checkpoints trained using the dataset (e.g. [Llama community license](https://ai.meta.com/llama/license/) for LLaMA-3.1, LLaMA-2, and Vicuna-v1.5). This project does not impose any additional constraints beyond those stipulated in the original licenses. Furthermore, users are reminded to ensure that their use of the dataset and checkpoints is in compliance with all applicable laws and regulations.


## Contents
- [Install](#install)
- [Dataset Collection](#dataset-collection)
- [Train](#train)
- [Hyperparameters](#hyperparameters)
- [Evaluation](#evaluation)
- [Citation](#citation)

## Install

1. Clone this repository and navigate to the Protein2Text folder:
      ```bash
    git clone https://github.com/alaaj27/Protein2Text.git
    cd Protein2Text
      ```

2. Download model weights:
    ```bash
    mkdir checkpoints
    cd checkpoints
    git git clone https://huggingface.co/tumorailab/protein2text-llama3.1-8B-instruct-esm2-650M
    cd ..
    ```

3. Install packages:

    3.1. Using conda 
      ```bash
      conda create -n protein2text_env python=3.10 -y
      conda activate protein2text_env
      pip install --upgrade pip  
      pip install -e .
      ```

    3.2. Using python venv: `[Note: This will create a new directory called protein2text_env]`  
      ```bash
      python -m venv protein2text_env
      source protein2text_env/bin/activate
      pip install --upgrade pip  
      pip install -e .
      ```

4. Download checkpoint:
    ```bash
    mkdir checkpoints
    cd checkpoints
    git clone https://huggingface.co/tumorailab/protein2text-llama3.1-8B-instruct-esm2-650M
    ```

5. Inference a batch:
    ```bash
    cd .. 
    python evaluation/inference_model.py
      --input_file data/[FILE_NAME].json 
      --model_path checkpoints/protein2text-llama3.1-8B-instruct-esm2-650M/
      --model_base meta-llama/Meta-Llama-3.1-8B-Instruct
    ```

<!-- 3. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
``` -->

<!-- ### Upgrade to latest code base

```Shell
git pull
pip install -e . -->
<!-- 
# if you see some import errors when you upgrade,
# please try running the command below (without #)
# pip install flash-attn --no-build-isolation --no-cache-dir
``` -->
<!-- 
### Quick Start With HuggingFace

<details>
<summary>Example Code</summary>

```Python
from llava.model.builder import load_pretrained_model
from evaluation.inference_model import eval_model, get_model_name_from_path, protein_sequence_tokenizer


model_path = "liuhaotian/llava-v1.5-7b"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)
```

Check out the details wth the `load_pretrained_model` function in `llava/model/builder.py`.

You can also use the `eval_model` function in `llava/eval/run_llava.py` to get the output easily. By doing so, you can use this code on Colab directly after downloading this repository.

``` python
model_path = "liuhaotian/llava-v1.5-7b"
prompt = "What are the things I should be cautious about when I visit here?"
image_file = "https://llava-vl.github.io/static/images/view.jpg"

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

eval_model(args)
```
</details>

## LLaVA Weights
Please check out our [Model Zoo](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md) for all public LLaVA checkpoints, and the instructions of how to use the weights. -->


## Train

Ptotein2Text training consists of two stages:  
1. Projector and Resampler Alignment Stage (Pretraining): We collected a dataset that spans 394,000 protein amino acid sequences and function descriptions collected from UniProt.  This dataset is entirely used to train the resampler and the projector during the pretraining stage to connect a *frozen pretrained protein encoder* to a *frozen pretrained LLM*.  

2. Sequence Instruction Tuning stage(Fine-tuning): we generate a comprehensive question and answering dataset ( **Protein2Text-QA**) to fine-tune the model parameters. The dataset spans approximately 210,000 pairs of QA. We utilize research carried out on proteins from published articles in the PubMed Central (PMC) database to create questions and answers. Articles mentioning the protein names are located and fed to the LLaMA3.1 model to generate a series of QA pairs, such that they focus only on the protein name given. 


Model training and inferencing were mainly performed on 2 NVIDIA H100 PCIe GPUs of 80GB VRAM. The estimated training time is roughly dependent on the number of parameters, the batch sizes, and other configurations such as gradient checkpointing, LoRA parameters, and resampler configurations. However, the estimated training time for the pretraining stage varies from 8 to 13 hours while the fine-tuning stage varies from 12-20 hours.


## Hyperparameters
We tried to align the hyperparameters of Protein2Text with the set of hyperparameters provided by LLaVA. Both hyperparameters used in pretraining and fine-tuning are provided below.

### Training Configuration  

| Phase              | Global Batch Size | Learning Rate | Epochs | Max Length | Weight Decay | Precision | Optimizer | Gradient Accumulation Steps | Warmup Ratio |
|--------------------|------------------:|--------------:|-------:|-----------:|-------------:|-----------|------------|-----------------------------:|-------------:|
| **Pretraining**   | 256               | 2 × 10⁻³      | 1      | 2048       | 0            | bf16 (Mixed Precision) | AdamW | 1 step | 0.03 |
| **Fine-tuning**   | 128               | 8 × 10⁻⁶      | 5      | 2048       | 0            | bf16 (Mixed Precision) | AdamW | 1 step | 0.03 |


#### Additional Model Components  

- **Protein Encoder:**  
  - Model: ESM2-650M  
  - Output Tokens: All (i.e., no truncation)  
  - Feature Layer: -2 (second to last)  

- **Language Model:**  
  - Model: LLaMA-3.1-8B-Instruct  
  - LoRA Rank: 64  
  - LoRA Alpha: 16  
  - Context Length: 2048  

- **Projector:**  
  - Number of Layers: 2  
  - Activation: GELU  
  - Hidden Dimensions: 4096  

- **Perceiver Resampler:**  
  - Number of Attention Layers: 4096  
  - Attention Heads: 8  
  - Dimension of Attention Heads: 4  
  - Multiplication Factor of Hidden State: 2  
  - Number of Latent Tokens: 128  


## Dataset Collection

<div align=center><img src="figures/DataCollectionPipelineH.png" width="100%" height="100%" /></div>



The **Protein2Text-QA** dataset was created in two main steps: retrieving relevant abstracts and generating question-answer (QA) pairs using **LLaMA3**.  

### 1. Abstract Retrieval  
Protein-related abstracts were collected using systematic queries in the **PubMed Central (PMC)** database via the **Entrez** library. Abstracts containing specific protein keywords were fetched using their **PMC IDs**, ensuring relevance by including only those explicitly mentioning the queried proteins. The abstracts were preprocessed to remove redundant text and formatting inconsistencies before being passed to the QA generation pipeline.  

### 2. QA Generation with LLaMA3  
The cleaned abstracts, along with protein names and roles, were input into **LLaMA3.1-8B-Instruct** to generate conversation-style QAs. The model was prompted to create QAs focusing solely on the protein’s function and attributes while ignoring unrelated information. Each abstract produced up to **10 QA pairs**, which were further filtered to remove irrelevant or uninformative responses. The dataset was refined to ensure questions remained general and protein-specific rather than abstract-specific.  

The final dataset includes structured **QA pairs**, covering unique proteins and their associated attributes. The complete data pipeline, including extraction and preprocessing, is outlined in the repository, along with dataset statistics such as the number of QA pairs, unique proteins, and sequence lengths.



## Download LLaMA 3.1 and ESM 2 checkpoints:

Please refer to the following pages, [LLaMA 3.1 8B](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) and [ESM-2 3B](https://huggingface.co/facebook/esm2_t36_3B_UR50D), for instructions and requirements on how to download the model weights for the base models.

<!-- ## Evaluation -->

<!-- 
## Evaluation

In LLaVA-1.5, we evaluate models on a diverse set of 12 benchmarks. To ensure reproducibility, we evaluate the models with greedy decoding. We do not evaluate using beam search to make the inference process consistent with the chat demo of real-time outputs.

See [Evaluation.md](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md). -->


## Citation

If you find Protein2Text useful for your research and applications, please cite using this BibTeX:
```bibtex
@inproceedings{jararweh2025protein2text,
  title={Protein2Text: Resampling Mechanism to Translate Protein Sequences into Human-Interpretable Text},
  author={Jararweh, Ala and Macaulay, Oladimeji and Arredondo, David and Hu, Yue and Tafoya, Luis E and Virupakshappa, Kushal and Sahu, Avinash},
  booktitle={Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 3: Industry Track)},
  pages={918--937},
  year={2025}
}

``` 
<!-- 
## Acknowledgement

- [LLaVA](https://github.com/lm-sys/FastChat): the codebase we built upon, and our base model Vicuna-13B that has the amazing language capabilities!

## Related Projects

- [Instruction Tuning with GPT-4](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)
- [LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day](https://github.com/microsoft/LLaVA-Med)
- [Otter: In-Context Multi-Modal Instruction Tuning](https://github.com/Luodian/Otter)

For future project ideas, please check out:
- [SEEM: Segment Everything Everywhere All at Once](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once)
- [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) to detect, segment, and generate anything by marrying [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) and [Segment-Anything](https://github.com/facebookresearch/segment-anything). -->
