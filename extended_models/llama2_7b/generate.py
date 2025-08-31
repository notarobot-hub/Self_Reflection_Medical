import argparse
import os
import sys
import torch
import transformers
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from tqdm import tqdm
import jsonlines

def generate_step(prompt, model, tokenizer, args):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(args.device)
    
    generation_config = GenerationConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_beams=args.num_beams,
        early_stopping=True,
        max_new_tokens=args.max_new_tokens,
    )

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
        )
    
    s = generation_output.sequences[0][len(input_ids[0]):]
    output = tokenizer.decode(s, skip_special_tokens=True)
    return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--load_8bit", action="store_true")
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    print(f"Loading model from {args.model_path}")
    tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = LlamaForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto" if args.device == "cuda" else None,
        load_in_8bit=args.load_8bit,
    )
    
    if args.device != "cuda":
        model = model.to(args.device)
    
    model.eval()
    
    # Process input file
    with jsonlines.open(args.input_file) as reader, jsonlines.open(args.out_file, mode='w') as writer:
        for line in tqdm(reader):
            question = line['question']
            
            # Create prompt for general QA
            prompt = f"Question: {question}\nAnswer:"
            
            response = generate_step(prompt, model, tokenizer, args)
            
            line.update({'generated_answer': response})
            writer.write(line)

if __name__ == "__main__":
    main()

