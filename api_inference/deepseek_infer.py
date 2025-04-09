import os
import json
import time
import pathlib
import argparse
from tqdm import tqdm
from openai import OpenAI

def process_gsm8k_with_deepseek(args, input_file, output_file, api_key, prompt_key='insufficient_question', model="deepseek-reasoner"):
    """
    Reads GSM8K data from a JSON file, performs inference using Deepseek model
    with the specified field, and saves the results to a JSONL file.
    Each item is saved immediately after processing and only unprocessed items are processed.
    
    Args:
        input_file (str): Path to the input JSON file
        output_file (str): Path where the output JSONL file will be saved
        api_key (str): Your OpenAI API key
        prompt_key (str): Key in the JSON to use as the prompt for Deepseek
        model (str): OpenAI model to use for inference
    """
    # Initialize the OpenAI client
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    
    # Read the input JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Define the output key that will be used to identify processed items
    output_key = 'model_answer'
    
    # Load existing processed items if output file exists
    processed_questions = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    item = json.loads(line)
                    # Use the "question" field as the identifier
                    if "question" in item:
                        processed_questions.add(item["question"])
        print(f"Loaded {len(processed_questions)} previously processed questions from {output_file}")
    
    # Process each item in the JSON
    with open(output_file, 'a') as outfile:
        # Process each item in the list if data is a list
        if isinstance(data, list):
            for item in tqdm(data):
                # Skip if the item doesn't have the expected structure
                if not isinstance(item, dict) or prompt_key not in item:
                    print(f"Warning: Item missing '{prompt_key}' field: {item}")
                    continue
                
                # Skip if the question has already been processed
                if "question" in item and item["question"] in processed_questions:
                    print(f"Skipping already processed question: {item['question'][:50]}...")
                    continue
                
                # Use the specified key as the prompt without any prefix
                prompt = item[prompt_key]
                if args.premise_prompt:
                    prompt += 'You can tell me if some premise is missing.'
                response = process_item(client, model, prompt, item)
                
                print(f"Processed item with prompt:\n{prompt}")
                print(f"Response:\n{response}")
                
                # Add the response to the item
                item[output_key] = response
                
                # Write the item to the JSONL file
                outfile.write(json.dumps(item) + '\n')
                outfile.flush()  # Ensure the data is written immediately
                
                # Update the processed questions set
                if "question" in item:
                    processed_questions.add(item["question"])

                pass
    
    print(f"Processing complete. Results saved to {output_file}")
    
    # Read all items from the JSONL file to return
    all_items = []
    with open(output_file, 'r') as f:
        for line in f:
            if line.strip():
                all_items.append(json.loads(line))
    
    return all_items

def process_item(client, model, prompt, original_item, max_retries=6):
    """
    Process a single item with OpenAI API, with retry logic
    
    Args:
        client: The OpenAI client
        model: The model name
        prompt: The prompt to send
        original_item: The original item data
        max_retries: Maximum number of retry attempts
    
    Returns:
        The Deepseek response text
    """
    retry_count = 0
    
    while retry_count <= max_retries:
        try:
            # Make the API call
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    # {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=8192,
                stream=False,
            )
            thinking_tokens = completion.choices[0].message.reasoning_content
            response_tokens = completion.choices[0].message.content
            
            return thinking_tokens + '</think>' + response_tokens
            
        except Exception as e:
            retry_count += 1
            if retry_count <= max_retries:
                error_msg = f"Error (attempt {retry_count}/{max_retries}): {str(e)}. Retrying in 3 seconds..."
                print(error_msg)
                time.sleep(3)  # Wait for 3 seconds before retrying
            else:
                # All retries exhausted
                print(f"Error after {max_retries} attempts: {str(e)}. Giving up.")
                return f"Error after {max_retries} attempts: {str(e)}"

# Example usage
if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Process JSON data with Deepseek model')
    parser.add_argument('--input', type=str, 
                        help='Path to the input JSON file')
    parser.add_argument('--output', type=str,
                        help='Path where the output JSONL file will be saved')
    parser.add_argument('--MiP', action='store_true', default=False,
                        help='Use "insufficient_question" field instead of "question"')
    parser.add_argument('--model', type=str, default="deepseek-reasoner",
                        help='OpenAI model to use (default: deepseek-reasoner)')
    parser.add_argument('--api-key', type=str, default="",
                        help='OpenAI API key')
    parser.add_argument("--premise_prompt", action='store_true')
    
    args = parser.parse_args()
    
    # Ensure API key is provided
    if not args.api_key:
        parser.error("Please provide an API key with --api-key")
    
    # Ensure output file has the correct extension
    if not args.output.endswith('.jsonl'):
        args.output = args.output + '.jsonl'
    
    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if args.MiP:
        key = 'insufficient_question'
    else:
        key = 'question'
    # Process the JSON file
    process_gsm8k_with_deepseek(args, args.input, args.output, args.api_key, key, args.model)