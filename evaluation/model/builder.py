import os
import warnings
import json
from transformers import AutoTokenizer
import torch
from evaluation.model import *
from evaluation.constants import DEFAULT_PROT_START_TOKEN, DEFAULT_PROT_END_TOKEN, PROTEIN_SEQUENCE_TOKEN_INDEX
from .gated_cross_attention.GCA import build_gca_components
from llava.model.language_model.llava_llama import LlavaConfig

class CustomConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

def load_pretrained_model(model_path, model_base, model_name,
                           load_8bit=False, load_4bit=False,
                             device_map="auto", device="cuda",
                               use_flash_attn=False, **kwargs):

    kwargs['device_map'] = {"": device}

    # Handle 8-bit and 4-bit loading without conflict
    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['bnb_4bit_compute_dtype'] = torch.float16
        kwargs['bnb_4bit_use_double_quant'] = True
        kwargs['bnb_4bit_quant_type'] = 'nf4'
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    if 'protein2text' in model_name.lower():
        
        if model_base is None:
            raise ValueError('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument.')
        
        else:
            
            lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features

            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                raise ValueError(f"Error: File not found at {model_path}")


            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}


            # Load the json file model_args.json from model_path
            model_args = os.path.join(model_path, 'model_args.json')
            if os.path.exists(model_args):
                with open(model_args, 'r') as f:
                    model_args = json.load(f)
                    model_args = CustomConfig(**model_args)

            model.get_model().config.mm_use_resampler_gca = model_args.mm_use_resampler_gca

            #supporting older versions:
            if model_args.mm_use_resampler_gca == True and "resampler" in model_path:
                model_args.mm_use_resampler_gca = "resampler"
                model.get_model().config.mm_use_resampler_gca = "resampler"

            elif model_args.mm_use_resampler_gca == True and "gca" in model_path:
                model_args.mm_use_resampler_gca = "gca"
                model.get_model().config.mm_use_resampler_gca = "gca"

            elif model_args.mm_use_resampler_gca == True and "projector" in model_path:
                model_args.mm_use_resampler_gca = "projector"
                model.get_model().config.mm_use_resampler_gca = "projector"


            if model_args.mm_use_resampler_gca == "gca" or model_args.pretrain_mm_gated_cross_attention is not None:
                mm_resampler, mm_gated_cross_attention = build_gca_components(model_args)
                model.get_model().mm_resampler = mm_resampler.to(device=device, dtype=torch.float16)
                model.get_model().mm_gated_cross_attention = mm_gated_cross_attention.to(device=device, dtype=torch.float16)



                model.load_state_dict(non_lora_trainables, strict=False)

            elif model_args.mm_use_resampler_gca == "resampler" or model_args.pretrain_mm_resampler is not None:
                mm_resampler, _ = build_gca_components(model_args)
                model.get_model().mm_resampler = mm_resampler.to(device=device, dtype=torch.float16)
                model.load_state_dict(non_lora_trainables, strict=False)
            else:
                mm_resampler = None
                mm_gated_cross_attention = None

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')

    else:
        raise ValueError(f"Error: Model {model_name} not found.")


    if 'protein2text' in model_name.lower():
        mm_use_protein_start_end = getattr(model.config, "mm_use_protein_start_end", False)
        if mm_use_protein_start_end:
            tokenizer.add_tokens([DEFAULT_PROT_START_TOKEN, DEFAULT_PROT_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        protein_encoder = model.get_protein_encoder()  
        if not protein_encoder.is_loaded:  
            protein_encoder.load_model(device_map=device_map)  
        if device_map != 'auto':
            protein_encoder.to(device=device_map, dtype=torch.float16)  

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, context_len  
