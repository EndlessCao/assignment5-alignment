import json
import pathlib
DATA_PATH = pathlib.Path(__file__).parent

metric = []
with open(DATA_PATH / 'gsm8k_lora.jsonl','r') as f:
    for line in f:
        data = json.loads(line)
        metric.append(data)


total = len(metric)
format_accuracy = sum((m['format_reward'] for m in metric)) / total * 100
answer_accuracy = sum((m['answer_reward'] for m in metric)) / total * 100
total_accuracy = sum((m['reward'] for m in metric)) / total * 100

print(f"format_accuracy: {format_accuracy} %")
print(f"answer_accuracy: {answer_accuracy} %")
print(f"total_accuracy: {total_accuracy} %")