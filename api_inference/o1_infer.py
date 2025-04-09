import os
import json
import time
import pathlib
import argparse
from tqdm import tqdm
from openai import OpenAI

def process_gsm8k_with_gpt4o(input_file, output_file, api_key, prompt_key='insufficient_question', model="gpt-4o"):
    """
    Reads GSM8K data from a JSON file, performs inference using GPT-4o model
    with the specified field, and saves the results to a new JSON file.
    
    Args:
        input_file (str): Path to the input JSON file
        output_file (str): Path where the output JSON file will be saved
        api_key (str): Your OpenAI API key
        prompt_key (str): Key in the JSON to use as the prompt for GPT-4o
        model (str): OpenAI model to use for inference
    """
    # Initialize the OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Read the input JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Process each item in the JSON
    results = []
    
    # Handle the data structure
    if isinstance(data, list):
        # Process each item in the list
        for item in tqdm(data):
            if isinstance(item, dict) and prompt_key in item:
                # Use the specified key as the prompt without any prefix
                prompt = item[prompt_key]
                response, all_token_count, thinking_token_count = process_item(client, model, prompt, item)
                print(f"Processed item with prompt:\n{prompt}")
                print(f"Response:\n{response}")
                key_new = 'model_answer'
                key_new_all_token_count = 'model_answer_all_token_count'
                key_new_thinking_token_count = 'model_answer_thinking_token_count'
                item[key_new] = response
                item[key_new_all_token_count] = all_token_count
                item[key_new_thinking_token_count] = thinking_token_count
                results.append(item)
            else:
                # If the item doesn't have the expected structure, include it unchanged
                results.append(item)
                print(f"Warning: Item missing '{prompt_key}' field: {item}")

            pass
    
    # Save the results to the output file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Processing complete. Results saved to {output_file}")
    return results

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
        The GPT-4o response text
    """
    retry_count = 0
    
    while retry_count <= max_retries:
        try:
            # Make the API call
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            thinking_token_count = completion.usage.completion_tokens_details.reasoning_tokens
            all_token_count = completion.usage.completion_tokens
            return completion.choices[0].message.content, all_token_count, thinking_token_count
            
        except Exception as e:
            retry_count += 1
            if retry_count <= max_retries:
                error_msg = f"Error (attempt {retry_count}/{max_retries}): {str(e)}. Retrying in 3 seconds..."
                print(error_msg)
                time.sleep(3)  # Wait for 3 seconds before retrying
            else:
                # All retries exhausted
                print(f"Error after {max_retries} attempts: {str(e)}. Giving up.")
                return f"Error after {max_retries} attempts: {str(e)}", 0, 0

# Example usage
if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Process JSON data with GPT-4o model')
    parser.add_argument('--input', type=str, 
                        help='Path to the input JSON file')
    parser.add_argument('--output', type=str, 
                        help='Path where the output JSON file will be saved')
    parser.add_argument('--MiP', action='store_true', default=False,
                        help='Use "insufficient_question" field instead of "question"')
    parser.add_argument('--model', type=str, default="o1-mini",
                        help='OpenAI model to use (default: o1-mini)')
    parser.add_argument('--api-key', type=str, default="",
                        help='OpenAI API key')
    
    args = parser.parse_args()
    
    # Ensure API key is provided
    if not args.api_key:
        parser.error("Please provide an API key with --api-key")
    
    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if args.MiP:
        key = 'insufficient_question'
    else:
        key = 'question'
    # Process the JSON file
    process_gsm8k_with_gpt4o(args.input, args.output, args.api_key, key, args.model)