import os
import json
import re
import argparse
import numpy as np
import openai
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="QwQ")
parser.add_argument("--data_root", type=str, default="gsm8k")
parser.add_argument("--version", type=str)
parser.add_argument("--cache_dir", type=str)
args = parser.parse_args()
word_list = ['alternatively', 'wait', 'check', 'but', 'maybe', 'might', 'perhaps', '\n\n']

model_name = args.model_name
result_dir = f"{args.data_root}/{args.version}/{args.model_name}"
data = []
for filename in os.listdir(result_dir):
    if filename.endswith('.json') and 'analysis_info' not in filename:
        file_path = os.path.join(result_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            file_data = json.load(f)
            if isinstance(file_data, list):
                data.extend(file_data)
            else:
                data.append(file_data)

model_answer_key = 'model_answer'

def split_into_sentences(text):
    sentences = re.split('\n\n', text)
    return sentences

def verify_insufficient(sentence):
    prompt = 'You are a helpful assistant. You will be given a paragraph which is part of the answer to a question. You need to identify if the paragraph doubt that the answer depends on some other unspecified condition. '
    prompt += f'Paragraph: {sentence}\n'
    prompt += 'Answer in one word, yes or no.'
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip().lower() == 'yes'


eval_prompt = f"""
    You are a helpful assistant that evaluates the quality of a model's answer. You will be given a question and a model's answer. You need to evaluate the correctness of the model's answer. If the model output says that the condition of the question is insufficient, you should return 0. Otherwise, if the model give a clear answer and matches the reference answer, you should return 1. If the model's answer does not match the reference answer, you should return 2. Please only return the number, no other text.
    """

def eval_sample(item):
    model_answer = item[model_answer_key]
    reference_answer = item['answer']
    short_ref_answer = reference_answer.split('####')[-1].strip()
    paragraphs = model_answer.split('\n\n')
    if len(paragraphs) >= 2:
        short_model_answer = '\n\n'.join(paragraphs[-2:]).strip()
    else:
        short_model_answer = model_answer.strip()
    # print('short_model_answer', short_model_answer)
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": eval_prompt},
            {"role": "user", "content": f"Model Answer: {short_model_answer}\nReference Answer: {short_ref_answer}"}
        ]
    )
    return response.choices[0].message.content.strip()

first_identified_paragraph_idx = []
total_paragraphs = []
first_identified_proportion = []
ever_identified = []
eval_results = []
word_counts_after_identification = {}
word_freq_after_identification = {}
num_trials = 3

for data_idx, item in tqdm(enumerate(data), total=len(data)):

    sentences = split_into_sentences(item[model_answer_key])
    trial_answer = []
    for trial_idx in range(num_trials):
        for idx, sentence in enumerate(sentences):
            if verify_insufficient(sentence):
                trial_answer.append(idx)
                break
        else:  
            trial_answer.append(len(sentences))
    
    if len(set(trial_answer)) == 1:  
        final_answer = trial_answer[0]
    elif len(set(trial_answer)) == 2:  
        final_answer = max(set(trial_answer), key=trial_answer.count)
    else:  
        final_answer = sorted(trial_answer)[1]
    
    ever_identified.append(final_answer != len(sentences))  
    total_paragraphs.append(len(sentences))
    first_identified_paragraph_idx.append(final_answer + 1)

    try:
        eval_result = eval_sample(item)
    except Exception as e:
        print(f"Error: {e}, retrying in 10 seconds...")
        import time
        time.sleep(10)
        eval_result = eval_sample(item)
    if eval_result == '0' or eval_result == '1' or eval_result == '2':
        result_code = int(eval_result)
    else:
        print('format error', eval_result)
        result_code = 3
    eval_results.append(result_code)


identified_and_insufficient = 0
total_samples = len(eval_results)

for i in range(len(eval_results)):
    if eval_results[i] == 0 and ever_identified[i]:
        identified_and_insufficient += 1

identified_and_clear = sum(ever_identified) - identified_and_insufficient
unidentified_and_insufficient = eval_results.count(0) - identified_and_insufficient
unidentified_and_clear = total_samples - identified_and_insufficient - unidentified_and_insufficient - identified_and_clear

result_counts = {
    0: eval_results.count(0),  # Insufficient condition
    1: eval_results.count(1),  # Correct answer
    2: eval_results.count(2),  # Incorrect answer
    3: eval_results.count(3)   # Format error
}
print(result_counts)
evaluation_results ={
    "correct_answer": result_counts[1] / (total_samples - result_counts[3]),
    "incorrect_answer": result_counts[2] / (total_samples - result_counts[3]),
    "insufficient_condition": result_counts[0] / (total_samples - result_counts[3]),
}

first_identified_info ={
    "avg_first_identified_paragraph_idx": np.mean(first_identified_paragraph_idx),
    "identified_and_insufficient": identified_and_insufficient,
    "identified_and_clear": identified_and_clear,
    "unidentified_and_insufficient": unidentified_and_insufficient,
    "unidentified_and_clear": unidentified_and_clear,
}

# Update the analysis_info.json file with evaluation results
analysis_info_path = f"{result_dir}/analysis_info.json"
if os.path.exists(analysis_info_path):
    with open(analysis_info_path, 'r', encoding='utf-8') as f:
        analysis_info = json.load(f)
else:
    analysis_info = {}

# Add evaluation results to the analysis info
analysis_info.update({
    "first_identified_info": first_identified_info,
    "evaluation_results": evaluation_results
})

# Save the updated analysis info
with open(analysis_info_path, 'w', encoding='utf-8') as f:
    json.dump(analysis_info, f, indent=4)

print(f"\nResults added to {analysis_info_path}")
