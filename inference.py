import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import json
from datetime import datetime

# Setup simple argument parser with only essential parameters
parser = argparse.ArgumentParser(description='Run inference with a language model on JSON data')
parser.add_argument('--model_name', type=str, required=True, 
                    help='The model to use for inference')
parser.add_argument('--json_file', type=str, required=True,
                    help='Path to a JSON file containing examples')
parser.add_argument('--MiP', action='store_true', default=False,
                    help='Use "insufficient_question" field instead of "question"')
parser.add_argument('--output_file', type=str, default=None,
                    help='Path to save the results as JSONL (default: auto-generated filename)')
parser.add_argument('--cache_dir', type=str,
                    help='Directory to cache the model')

args = parser.parse_args()

# If output file is not specified, generate a default name
if args.output_file is None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_file = f"results_{args.model_name.replace('/', '_')}_{timestamp}.jsonl"

# Create directory if it doesn't exist
output_dir = os.path.dirname(os.path.abspath(args.output_file))
if output_dir:
    os.makedirs(output_dir, exist_ok=True)


# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    torch_dtype="auto",
    device_map="auto",
    cache_dir=args.cache_dir
)
tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)

# Run inference on a prompt
def run_inference(prompt):
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Run the model
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# Main process
with open(args.json_file, 'r', encoding='utf-8') as f:
    examples = json.load(f)

# Set the field to use for prompts
prompt_field = "insufficient_question" if args.MiP else "question"

# Process examples and write results immediately to JSONL file
from tqdm import tqdm

# Process all examples first
results = []
for i, example in tqdm(enumerate(examples), total=len(examples), desc="Processing examples"):
    # Check if prompt field exists
    if prompt_field not in example:
        continue
    prompt = example[prompt_field]
    answer = run_inference(prompt)
    # Create result
    result = {
        "model_answer": answer
    }
    # Add reference answer and original question if available
    if 'answer' in example:
        result["answer"] = example['answer']
    if 'question' in example:
        result["question"] = example['question']
    if args.MiP:
        result["insufficient_question"] = example['insufficient_question']
    
    # Add to results list
    results.append(result)

# Save all results at once as a JSON array
with open(args.output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"Results have been saved to {args.output_file}")