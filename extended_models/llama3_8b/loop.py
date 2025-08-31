#!/usr/bin/env python
# coding: utf-8
import argparse
import os
import sys
import torch
import transformers
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from tqdm import tqdm
import jsonlines
import numpy as np

# Add evaluation modules
sys.path.append('../evaluate')
from sent_similarity import Sent_Similar
from CTRLEval.ctrleval import CTRLEval
from loop_eval_utils import evaluate_response, evaluate_knowledge

# Add main loop utility
sys.path.append('..')
from loop_utils import main_loop

def generate_step(args, model, tokenizer, prompt):
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

def knowledge_loop(args, model, tokenizer, question, knowledge_loop_list=[]):
    print("knowledge_loop")
    THRESHOLD_FACTUAL = args.threshold_fact
    MAX_KNOWLEDGE_LOOP = args.max_knowledge_loop
    candidates = []
    history = []
    
    # Create prompt for knowledge generation using Llama 3.1 format
    instruction = "Provide background knowledge to answer the following question."
    prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}\n\nQuestion: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    
    if knowledge_loop_list:
        knowledge = knowledge_loop_list[0]
    else:
        knowledge = generate_step(args, model, tokenizer, prompt)
    
    loop_i = 0
    if MAX_KNOWLEDGE_LOOP > 1:
        if args.gptscore_model == 'gpt3':
            factuality_score = evaluate_knowledge('gpt3', args.demo_num, question, knowledge)
        else:
            factuality_score = evaluate_knowledge(model, args.demo_num, question, knowledge, tokenizer)
        candidates.append([factuality_score, knowledge])
        history.append([loop_i, knowledge, factuality_score])
    
    # Refine knowledge
    loop_i += 1
    while (loop_i < MAX_KNOWLEDGE_LOOP) and factuality_score < THRESHOLD_FACTUAL:
        if args.no_aspect:
            instruction = "Please refine the knowledge."
        elif args.no_number:
            instruction = "The knowledge is not strongly supported by empirical evidence. Please refine the knowledge to improve its factuality."
        else:
            instruction = f"The factuality score for the knowledge is {factuality_score} less than {THRESHOLD_FACTUAL}, which means the knowledge is not strongly supported by empirical evidence. Please refine the knowledge to improve its factuality."
        
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}\n\nPrevious Knowledge: {knowledge}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        knowledge = generate_step(args, model, tokenizer, prompt)
        
        if args.gptscore_model == 'gpt3':
            factuality_score = evaluate_knowledge('gpt3', args.demo_num, question, knowledge)
        else:
            factuality_score = evaluate_knowledge(model, args.demo_num, question, knowledge, tokenizer)
            
        candidates.append([factuality_score, knowledge])
        history.append([loop_i, knowledge, factuality_score])
        loop_i += 1
        
    if (MAX_KNOWLEDGE_LOOP > 1) and factuality_score < THRESHOLD_FACTUAL:
        # Still not satisfied, return highest score
        candidates.sort()
        return candidates[-1][-1], history
    else:
        return knowledge, history

def response_loop(args, model, tokenizer, question, final_knowledge):
    print("response_loop")
    THRESHOLD_CONS = args.threshold_consistency
    MAX_RESPONSE_LOOP = args.max_response_loop
    candidates = []
    entailment_score_question_list = []
    history = []
    
    instruction = f'''Refer to the knowledge: "{final_knowledge}" and answer the question: "{question}" with one paragraph.'''
    prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    
    response = generate_step(args, model, tokenizer, prompt)
    
    loop_i = 0
    if MAX_RESPONSE_LOOP > 1:
        entailment_score_question, cons_score_knowledge = evaluate_response(entailment_scorer, ctrleval_scorer, question, response, final_knowledge)
        candidates.append([(entailment_score_question + cons_score_knowledge) / 2, response])
        entailment_score_question_list.append(entailment_score_question)
        history.append([loop_i, response, entailment_score_question, cons_score_knowledge])
    
    loop_i += 1
    while loop_i < MAX_RESPONSE_LOOP and cons_score_knowledge < THRESHOLD_CONS:
        if args.no_aspect:
            instruction = "Please refine the response."
        elif args.no_number:
            instruction = "The alignment and consistency between response and knowledge are low. Please refine the response to improve its consistency."
        else:
            instruction = f"The consistency score for the knowledge is {cons_score_knowledge} less than {THRESHOLD_CONS}, which means the alignment and consistency between response and knowledge are low. Please refine the response to improve its consistency."
        
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}\n\nPrevious Response: {response}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        response = generate_step(args, model, tokenizer, prompt)
        
        entailment_score_question, cons_score_knowledge = evaluate_response(entailment_scorer, ctrleval_scorer, question, response, final_knowledge)
        candidates.append([(entailment_score_question + cons_score_knowledge) / 2, response])
        entailment_score_question_list.append(entailment_score_question)
        history.append([loop_i, response, entailment_score_question, cons_score_knowledge])
        
        loop_i += 1
        
    if MAX_RESPONSE_LOOP > 1 and cons_score_knowledge < THRESHOLD_CONS:
        # Still not satisfied, return highest score
        merge = zip(candidates, entailment_score_question_list)
        merge = sorted(merge)
        candidates, entailment_score_question_list = zip(*merge)
        return candidates[-1][-1], history, entailment_score_question_list[-1]
    else:
        return response, history, entailment_score_question

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--continue-generate", action="store_true")
    parser.add_argument("--no-number", action="store_true")
    parser.add_argument("--no-aspect", action="store_true")
    
    parser.add_argument("--out-dir", type=str, default="llama3_8b_loop")
    parser.add_argument('--sources', nargs='+', required=True)
    parser.add_argument("--max-loop", type=int, default=1)
    parser.add_argument("--max-knowledge-loop", type=int, default=1)
    parser.add_argument("--max-response-loop", type=int, default=1)
    parser.add_argument("--gptscore-model", type=str, default="llama3")
    parser.add_argument("--demo-num", type=int, default=0)
    
    parser.add_argument("--threshold-entailment", type=float, default=0.8)
    parser.add_argument("--threshold-fact", type=float, default=-1)
    parser.add_argument("--threshold-consistency", type=float, default=-5)
    
    parser.add_argument("--max-sample", type=int, default=3000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model-path", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--load-8bit", action="store_true")
    
    args = parser.parse_args()
    
    # Initialize scorers
    if args.max_response_loop > 1:
        ctrleval_scorer = CTRLEval(device=args.device)
    entailment_scorer = Sent_Similar()
    
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
    
    # Create output directory
    out_dir = f"{args.out_dir}_MaxL{args.max_loop}_MaxKL{args.max_knowledge_loop}MaxRL{args.max_response_loop}_ThE{args.threshold_entailment}ThF{args.threshold_fact}ThC{args.threshold_consistency}_{args.gptscore_model}_Demo{args.demo_num}"
    os.makedirs(out_dir, exist_ok=True)
    
    # Process each source
    for source in args.sources:
        print(f"Processing source: {source}")
        input_file = args.input_file.format(source=source)
        
        if args.no_aspect:
            out_file = f'{out_dir}/{source}_T{args.temperature}_no_aspect.jsonl'
        elif args.no_number:
            out_file = f'{out_dir}/{source}_T{args.temperature}_no_number.jsonl'
        else:
            out_file = f'{out_dir}/{source}_T{args.temperature}.jsonl'
        
        if args.continue_generate and os.path.exists(out_file):
            print("Continuing generation from existing file")
            with jsonlines.open(out_file) as reader:
                old_lines = list(reader)
            with jsonlines.open(input_file) as reader:
                reader = list(reader)
                for i, line in tqdm(enumerate(reader), total=len(reader)):
                    if i < len(old_lines):
                        continue
                    if i > args.max_sample:
                        break
                    
                    final_knowledge, final_response, all_history_knowledge, all_history_response = main_loop(
                        args, line, model, tokenizer, knowledge_loop, response_loop
                    )
                    
                    line.update({'history_knowledge': all_history_knowledge})
                    line.update({'history_response': all_history_response})
                    line.update({'generated_knowledge': final_knowledge})
                    line.update({'generated_answer': final_response})
                    
                    with jsonlines.open(out_file, mode='a') as writer:
                        writer.write(line)
        else:
            with jsonlines.open(input_file) as reader:
                reader = list(reader)
                for i, line in tqdm(enumerate(reader), total=len(reader)):
                    if i > args.max_sample:
                        break
                    
                    final_knowledge, final_response, all_history_knowledge, all_history_response = main_loop(
                        args, line, model, tokenizer, knowledge_loop, response_loop
                    )
                    
                    line.update({'history_knowledge': all_history_knowledge})
                    line.update({'history_response': all_history_response})
                    line.update({'generated_knowledge': final_knowledge})
                    line.update({'generated_answer': final_response})
                    
                    with jsonlines.open(out_file, mode='a') as writer:
                        writer.write(line)

if __name__ == "__main__":
    main()

