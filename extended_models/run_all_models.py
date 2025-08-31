#!/usr/bin/env python
# coding: utf-8
"""
Script to run all extended models on a given dataset
"""

import argparse
import subprocess
import os
import sys

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Success!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run all extended models on a dataset')
    parser.add_argument('--dataset', type=str, required=True, 
                       choices=['truthfulqa', 'triviaqa', 'tydiqa', 'coqa'],
                       help='Dataset to run models on')
    parser.add_argument('--input-file', type=str, required=True,
                       help='Path to input dataset file')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory for results')
    parser.add_argument('--max-sample', type=int, default=100,
                       help='Maximum number of samples to process')
    parser.add_argument('--run-baseline', action='store_true',
                       help='Run baseline generation (no self-reflection)')
    parser.add_argument('--run-loop', action='store_true',
                       help='Run self-reflection loop generation')
    parser.add_argument('--max-loop', type=int, default=3,
                       help='Maximum number of loops for self-reflection')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    models = [
        ('llama2_7b', 'meta-llama/Llama-2-7b-hf'),
        ('llama3_8b', 'meta-llama/Llama-3.1-8B-Instruct'),
        ('vicuna_7b', 'lmsys/vicuna-7b-v1.5')
    ]
    
    if args.run_baseline:
        print("Running baseline generation...")
        for model_name, model_path in models:
            output_file = f"{args.output_dir}/{model_name}_{args.dataset}_baseline.jsonl"
            
            cmd = [
                'python', f'extended_models/{model_name}/generate.py',
                '--input_file', args.input_file,
                '--out_file', output_file,
                '--model_path', model_path,
                '--device', args.device,
                '--max_sample', str(args.max_sample)
            ]
            
            success = run_command(cmd, f"Baseline generation with {model_name}")
            if not success:
                print(f"Failed to run baseline generation for {model_name}")
    
    if args.run_loop:
        print("Running self-reflection loop generation...")
        for model_name, model_path in models:
            output_dir = f"{args.output_dir}/{model_name}_loop"
            
            cmd = [
                'python', f'extended_models/{model_name}/loop.py',
                '--input-file', args.input_file,
                '--sources', args.dataset,
                '--out-dir', output_dir,
                '--max-loop', str(args.max_loop),
                '--max-knowledge-loop', str(args.max_loop),
                '--max-response-loop', str(args.max_loop),
                '--gptscore-model', model_name,
                '--demo-num', '0',
                '--threshold-entailment', '0.8',
                '--threshold-fact', '-1.0',
                '--threshold-consistency', '-5',
                '--max-sample', str(args.max_sample),
                '--device', args.device,
                '--model-path', model_path
            ]
            
            success = run_command(cmd, f"Self-reflection loop generation with {model_name}")
            if not success:
                print(f"Failed to run self-reflection loop for {model_name}")
    
    print("\nAll models completed!")

if __name__ == "__main__":
    main()


