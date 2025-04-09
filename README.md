# MiP-Overthinking

## Installation

```bash
pip install -r requirements.txt
```

## Non-api model inference

For well-defined question inference, use the following command:
```bash
python inference.py --model_name <model_name> --json_file <json_file> --output_file <output_file> --cache_dir <cache_dir>
```

For MiP question inference, use the following command:
```bash
python inference.py --model_name <model_name> --json_file <json_file> --MiP --output_file <output_file> --cache_dir <cache_dir>
```

## API model inference

For well-defined question inference, use the following command:
```bash
python api_inference/deepseek_infer.py --input <input_file> --output <output_file> --model <model> --api_key <api_key> 
```

For MiP question inference, use the following command:
```bash
python api_inference/deepseek_infer.py --input <input_file> --output <output_file> --MiP --model <model> --api_key <api_key>
```

## Evaluation

To get the token count and word count, use the following command:
```bash
python count.py --model_name <model_name> --data_root <data_root> --version <version> --google_api_key <google_api_key>
```

To get the evaluation results for accuracy, abstain rate, and information about the suspicion of MiP, use the following command:
```bash
python eval.py --model_name <model_name> --data_root <data_root> --version <version> --google_api_key <google_api_key>
```


