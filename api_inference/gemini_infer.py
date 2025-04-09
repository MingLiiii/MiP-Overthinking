import os
import time
import json
import pathlib
import argparse
from tqdm import tqdm
from google import genai

def process_gsm8k_with_gemini(input_file, output_file, api_key, prompt_key='insufficient_question', model="gemini-2.0-flash"):
    """
    Reads GSM8K data from a JSON file, performs inference using specified Gemini model
    with the specified field, and saves the results to a new JSON file.
    
    Args:
        input_file (str): Path to the input JSON file
        output_file (str): Path where the output JSON file will be saved
        api_key (str): Your Gemini API key
        prompt_key (str): Key in the JSON to use as the prompt for Gemini
        model (str): Gemini model to use for inference
    """
    # Initialize the Gemini client
    client = genai.Client(api_key=api_key)
    
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
                response = process_item(client, model, prompt, item)
                print(f"Processed item with prompt:\n{prompt}")
                print(f"Response:\n{response}")
                key_new = 'model_answer'
                item[key_new] = response
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
    Process a single item with Gemini API, with retry logic
    
    Args:
        client: The Gemini client
        model: The model name
        prompt: The prompt to send
        original_item: The original item data
        max_retries: Maximum number of retry attempts
    
    Returns:
        A dictionary with the original data and Gemini's response
    """
    retry_count = 0
    
    while retry_count <= max_retries:
        try:
            # Make the API call
            response = client.models.generate_content(
                model=model, 
                contents=prompt
            )
            
            return response.text
            
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
    parser = argparse.ArgumentParser(description='Process JSON data with Gemini models')
    parser.add_argument('--input', type=str, help='Path to the input JSON file')
    parser.add_argument('--output', type=str, help='Path where the output JSON file will be saved')
    parser.add_argument('--MiP', action='store_true', default=False,
                        help='Use "insufficient_question" field instead of "question"')
    parser.add_argument('--model', type=str, default="gemini-2.0-flash",
                        help='Gemini model to use (default: gemini-2.0-flash)')
    parser.add_argument('--api-key', type=str, default='',
                        help='Gemini API key (defaults to GEMINI_API_KEY environment variable)')
    
    args = parser.parse_args()
    
    # Ensure API key is provided
    if not args.api_key:
        parser.error("Please provide an API key with --api-key or set the GEMINI_API_KEY environment variable")
    
    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if args.MiP:
        key = 'insufficient_question'
    else:
        key = 'question'
    # Process the JSON file
    process_gsm8k_with_gemini(args.input, args.output, args.api_key, key, args.model)