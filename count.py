import os
import json
import argparse
from utils import get_answer_key

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str)
parser.add_argument("--data_root", type=str, default="gsm8k")
parser.add_argument("--version", type=str)
parser.add_argument("--google_api_key", type=str, default=None)
args = parser.parse_args()

word_list = ['alternatively', 'wait', 'check', 'but', 'maybe', 'might', 'perhaps', '\n\n']
model_name = args.model_name
data_root = args.data_root
version = args.version
print(f"Analyzing {model_name} on {data_root}, {version}")

short_model_mapping = {
    "QwQ": "Qwen/QwQ-32B",
    "s1.1": "simplescaling/s1.1-32B",
    "DSQ": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "Gemma227": "google/gemma-2-27b-it",
    "Limo": "GAIR/LIMO",
    "gpt3_5": "gpt3_5",
    "gpt4o": "gpt4o",
    "o1mini_medium": "o1mini_medium",
    "o3mini_medium": "o3mini_medium",
    "o1": "o1",
    "gemini1_5": "gemini1_5",
    "gemini2": "gemini2",
    "deepseekR1": "deepseek-ai/DeepSeek-R1",
    "sft-qwen-1": "Qwen/Qwen2.5-7B-Instruct",
    "sft-qwen-3": "Qwen/Qwen2.5-7B-Instruct",
    "phi3": "microsoft/Phi-3-medium-128k-instruct",
}

model_type = {
    "Qwen/QwQ-32B": "non-api",
    "simplescaling/s1.1-32B": "non-api",
    "Qwen/Qwen2.5-32B": "non-api",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": "non-api",
    "google/gemma-2-27b-it": "non-api",
    "GAIR/LIMO": "non-api",
    "gpt3_5": "openai",
    "gpt4o": "openai",
    "o1mini_medium": "openai",
    "o3mini_medium": "openai",
    "o1": "openai",
    "gemini1_5": "gemini",
    "gemini2": "gemini",
    "deepseek-ai/DeepSeek-R1": "non-api",
    "Qwen/Qwen2.5-7B-Instruct": "non-api",
    "meta-llama/Llama-3.1-8B-Instruct": "non-api",
    "Qwen/Qwen2.5-32B-Instruct": "non-api",
    "microsoft/Phi-3-medium-128k-instruct": "non-api",
    "Qwen/Qwen2.5-7B-Instruct": "non-api"
}

long_model_name = short_model_mapping[model_name]

# Select the tokenizer or token counter based on the model type.
if model_type[long_model_name] == "non-api":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(long_model_name)
elif model_type[long_model_name] == "openai":
    import tiktoken
    tokenizer = tiktoken.get_encoding("cl100k_base")
elif model_type[long_model_name] == "gemini":
    import google.generativeai as genai
    genai.configure(api_key=args.google_api_key)
    # Choose the appropriate Gemini endpoint based on the model name.
    if model_name == "gemini1_5":
        gemini_model = genai.GenerativeModel("models/gemini-1.5-flash")
    if model_name == "gemini2":
        gemini_model = genai.GenerativeModel("models/gemini-2.0-flash")
    
    # Create a simple tokenizer wrapper for Gemini models.
    class GeminiTokenizer:
        def __init__(self, model):
            self.model = model
        def encode(self, text, add_special_tokens=False):
            # Use the Gemini model's count_tokens method.
            token_count = self.model.count_tokens(text).total_tokens
            # Return a dummy list with length equal to the token count.
            return [None] * token_count

    tokenizer = GeminiTokenizer(gemini_model)

result_dir = f"{data_root}/{version}/{model_name}"

data = []
for filename in os.listdir(result_dir):
    if filename.endswith('.json') and 'analysis_info' not in filename:
        file_path = os.path.join(result_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                file_data = json.load(f)
                if isinstance(file_data, list):
                    data.extend(file_data)
                else:
                    data.append(file_data)
            except json.JSONDecodeError:
                # Handle line-by-line JSON format
                f.seek(0)  # Reset file pointer to beginning
                for line in f:
                    if line.strip():  # Skip empty lines
                        data.append(json.loads(line))

ans_key_tag = 'insufficient_question' if version == 'result' or version == 'prompt_re' else 'question'
model_answer_key = 'model_answer'


def get_answer_length(data, tokenizer, is_openai=False, model_name='', ans_key_tag=''):
    answer_lengths = []
    for item in data:
        if 'o1mini_' in  model_name or 'o3mini_' in  model_name or model_name == 'o1':
            if 'o1mini_' in  model_name:
                item_len = item['o1-mini ({}) all_token_count'.format(ans_key_tag)]
            if 'o3mini_' in  model_name:
                item_len = item['o3-mini ({}) all_token_count'.format(ans_key_tag)]
            if model_name == 'o1':
                item_len = item['o1 ({}) all_token_count'.format(ans_key_tag)]
            answer_lengths.append(item_len)
            pass
        else:
            answer = item[model_answer_key]
            if is_openai:
                # For OpenAI responses, use the tokenizer.encode method without extra arguments
                answer_tokens = tokenizer.encode(answer)
                answer_tokens_len = len(answer_tokens)
            else:
                answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
                answer_tokens_len = len(answer_tokens)
            answer_lengths.append(answer_tokens_len)
    return answer_lengths

def get_word_count(data, word_list, sum_token_length):
    word_counts = {}
    word_freq = {}
    for word in word_list:
        count = 0
        for item in data:
            count += item[model_answer_key].lower().count(word.lower())
        word_counts[word] = count
        word_freq[word] = count / sum_token_length
    return word_counts, word_freq

info = {}
is_openai = model_type[long_model_name] == "openai"
answer_lengths = get_answer_length(data, tokenizer, is_openai=is_openai, model_name=model_name, ans_key_tag=ans_key_tag)
info['mean_answer_length'] = sum(answer_lengths) / len(answer_lengths)

sum_token_length = sum(answer_lengths)
word_counts, word_freq = get_word_count(data, word_list, sum_token_length)
info['word_counts'] = word_counts
info['word_freq'] = word_freq

# Save the analysis information to a JSON file
output_file = f"{result_dir}/analysis_info.json"
if os.path.exists(output_file):
    with open(output_file, 'r', encoding='utf-8') as f:
        existing_data = json.load(f)
        # Update the existing data with new information
        existing_data.update(info)
        info = existing_data

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(info, f, indent=4)
