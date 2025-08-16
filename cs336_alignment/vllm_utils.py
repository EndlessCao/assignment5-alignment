from unittest.mock import patch

from torch._dynamo import eval_frame

from transformers import AutoModelForCausalLM, AutoTokenizer,PreTrainedModel
import tempfile
import shutil
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from drgrpo_grader import r1_zero_reward_fn
from pprint import pprint
from vllm import LLM, SamplingParams
from typing import Callable, List
import pathlib
import json
import os
from tqdm import tqdm
from vllm.model_executor import set_random_seed as vllm_set_random_seed
import dotenv
from peft import PeftModel

dotenv.load_dotenv()
# r1_zero_reward_fn(response, ground_truth, fast=True)
DATA_PATH = pathlib.Path(__file__).parent.parent / "data" / "gsm8k" / "sft_test.jsonl"
SAVE_PATH = pathlib.Path(__file__).parent / "output"
OUTPUT_PATH = SAVE_PATH / "gsm8k_lora.jsonl"
MODEL_PATH = os.environ["MODEL_PATH"]
os.makedirs(SAVE_PATH, exist_ok=True)

ZERO_SHOT_PROMPT = """
A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>
"""


def read_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    answers: list[str],
    eval_sampling_params: SamplingParams,
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    metics = []
    outputs = vllm_model.generate(prompts, eval_sampling_params, use_tqdm=True)
    for j, (output, answer) in enumerate(zip(outputs, answers)):
        resp = output.outputs[0].text
        eval_metric = reward_fn(resp, answer)
        eval_metric["response"] = resp
        eval_metric["answer"] = answer
        eval_metric["prompt"] = prompts[j]
        metics.append(eval_metric)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for metric in metics:
            f.write(json.dumps(metric, ensure_ascii=False) + "\n")
    print("Evaluation done! Output saved to:", OUTPUT_PATH)


def init_vllm(
    model_id: str, device: str, seed: int = 114514, gpu_memory_utilization: float = 0.85
):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )

    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

    


    