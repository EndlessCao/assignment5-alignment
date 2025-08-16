from sft import SftTrainer, SFTArgs
import pathlib
from transformers import AutoTokenizer, AutoModelForCausalLM
from drgrpo_grader import r1_zero_reward_fn
import dotenv
import os
from datasets import Dataset
import torch
from peft import LoraConfig, get_peft_model
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
env = dotenv.find_dotenv()
dotenv.load_dotenv(env)

DATA_PATH = pathlib.Path(__file__).parent.parent / "data"
MODEL_PATH = "/data/home/Caowei/models/Qwen/Qwen3-0___6B"

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH,
                                             torch_dtype=torch.bfloat16,
                                             ).to("cuda:1")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

LORA_R = 128 # 512
LORA_ALPHA = LORA_R * 2# 1024
LORA_DROPOUT = 0.05
# Define LoRA Config
lora_config = LoraConfig(
    r = LORA_R, # the dimension of the low-rank matrices
    lora_alpha = LORA_ALPHA, # scaling factor for the weight matrices
    lora_dropout = LORA_DROPOUT, # dropout probability of the LoRA layers
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

# Prepare int-8 model for training - utility function that prepares a PyTorch model for int8 quantization training. <https://huggingface.co/docs/peft/task_guides/int8-asr>
# model = prepare_model_for_int8_training(model)
# initialize the model with the LoRA framework
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

train_data = Dataset.load_from_disk(DATA_PATH / "gsm8k/train_dataset_sft")
val_data = Dataset.load_from_disk(DATA_PATH / "gsm8k/test_dataset_sft")
args = SFTArgs(
    num_epoch=1,
    batch_size=64,
    micro_batch_size = 2,
    gradient_accumulation_steps=2,
    group_size = 4,
    normalize_constant=1,
    eval_interval=10,
    eval_batch_size = 512,
    lr=2e-4,
    use_wandb=True,
    wandb_project="cs336_alignment",
    wandb_name="sft_test3",
    reward_fn=r1_zero_reward_fn,
    vllm_kwargs={
        "model_id": MODEL_PATH,
        "device": torch.device("cuda:0"),
        "gpu_memory_utilization": 0.5,
    },
)
if __name__ == '__main__':
    trainer = SftTrainer(model, tokenizer, train_data, val_data, args)
    trainer.train()
