import os
import json
import argparse
import torch
from tqdm import tqdm
from evaluation.model.builder import (
    load_pretrained_model,
    # get_model_name_from_path,
    # protein_sequence_tokenizer
)

# from evaluation.mm_utils import get_model_name_from_path
# from evaluation.train.train_protein import protein_sequence_tokenizer
from evaluation.conversation import conv_templates
from evaluation.constants import DEFAULT_PROTEIN_SEQUENCE_TOKEN, PROTEIN_SEQUENCE_TOKEN_INDEX
import warnings
from transformers import logging as transformers_logging

# Configure logging and warnings
transformers_logging.set_verbosity_warning()
warnings.filterwarnings("ignore", category=UserWarning)



def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]
    

def protein_sequence_tokenizer(prompt, tokenizer, protein_sequence_token_index=PROTEIN_SEQUENCE_TOKEN_INDEX, return_tensors=None):
    
    # Split the prompt into chunks by the protein sequence token
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<protein_sequence>')]

    def insert_separator(X, sep):
        # Insert the separator token between each chunk
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0

    # Check if the first token is the beginning of sequence token (e.g., <s> or <bos>)
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    # Insert the protein sequence token index between chunks
    for x in insert_separator(prompt_chunks, [protein_sequence_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids





def eval_model(args, model, tokenizer):
    qs = DEFAULT_PROTEIN_SEQUENCE_TOKEN + "\n" + args.query
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    input_ids = (
        protein_sequence_tokenizer(prompt, tokenizer, PROTEIN_SEQUENCE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(args.device)
    )
    with torch.inference_mode():
        model = model.bfloat16()
        output_ids = model.generate(
            input_ids,
            args.amino_seq,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            num_return_sequences=args.num_return_sequences,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs

def predict_data(input_file, output_file, model_path, model_base, device, temperature, top_p, num_beams, max_new_tokens):
    print(f"Processing input file: {input_file} on device {device}.")
    model_name = get_model_name_from_path(model_path)

    print("#" * 50)
    print(f"Loading model from {model_path} with base model {model_base} and name {model_name}.")
    print("#" * 50)

    tokenizer, model, _ = load_pretrained_model(model_path, model_base, model_name, device=device)
    
    with open(input_file, "r") as f:
        dataset = json.load(f)
    
    with open(output_file, "w") as output_f, tqdm(total=len(dataset), desc=f"Processing", unit="sample") as pbar:
        for sample in dataset:
            long_format_id = sample.get("long_format_id", sample.get("id", "Unknown_ID"))
            if long_format_id == "Unknown_ID":
                raise ValueError("ID not found in sample.")
            
            prompt = sample["conversations"][0]["value"].replace("<protein_sequence>\n", "")
            ground_truth = sample["conversations"][1]["value"]
            amino_seq = sample["amino_seq"]
            
            # Add user prompt template to the conversation for better Llama performance.
            # https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/
            new_prompt = f"""
            <|start_header_id|>user<|end_header_id|>
            {prompt}
            <|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
            """
            
            args = type("Args", (), {
                "query": new_prompt,
                "conv_mode": "plain",
                "amino_seq": amino_seq,
                "temperature": temperature,
                "top_p": top_p,
                "num_beams": num_beams,
                "num_return_sequences": 1,
                "max_new_tokens": max_new_tokens,
                "device": device
            })()
            
            try:
                predicted = eval_model(args, model, tokenizer)
                result = {"long_format_id": long_format_id, "Prompt": prompt, "Ground Truth": ground_truth, "Predicted": predicted}
                output_f.write(json.dumps(result) + "\n")
            except Exception as e:
                print(f"Error during prediction for ID {long_format_id}: {e}")
                raise e
            pbar.update(1)
    print(f"Processing completed. Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prediction script.")
    parser.add_argument("--input_file", type=str, help="Path to the input JSON file.")
    parser.add_argument("--output_file", type=str, nargs="?", help="File to save the output predictions.", default=None)
    parser.add_argument("--model_path", type=str, help="Path to the model checkpoint.")
    parser.add_argument("--model_base", type=str, help="Base model name.", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (e.g., 'cuda' or 'cpu').")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p for nucleus sampling.")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for beam search.")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of new tokens to generate.")
    args = parser.parse_args()
    
    # Set output file if not provided
    if args.output_file is None:
        input_filename = os.path.splitext(os.path.basename(args.input_file))[0]
        args.output_file = os.path.join("results", f"{input_filename}_results.jsonl")
        os.makedirs("results", exist_ok=True)
    
    predict_data(
        args.input_file, 
        args.output_file, 
        args.model_path, 
        args.model_base, 
        args.device, 
        args.temperature, 
        args.top_p, 
        args.num_beams, 
        args.max_new_tokens
    )
