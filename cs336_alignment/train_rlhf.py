from rlhf import GRPOTrainer, GRPOTrainArgs
import pathlib
from transformers import AutoTokenizer, AutoModelForCausalLM
import dotenv
import os
from datasets import Dataset
import torch
import random
from peft import LoraConfig, get_peft_model
import time
env = dotenv.find_dotenv()
dotenv.load_dotenv(env)

seed = 114514
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
DATA_PATH = pathlib.Path(__file__).parent.parent / "data"
MODEL_PATH ="/data/home/Caowei/models/Qwen/Qwen2___5-Math-1___5B"

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH,
                                             #load_in_8bit=True,
                                             torch_dtype=torch.bfloat16,
                                             device_map="cuda:2",
                                             )
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side='left')

LORA_R = 128 # 512
LORA_ALPHA = LORA_R * 2 # 1024
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

# from peft import prepare_model_for_kbit_training
# model = prepare_model_for_kbit_training(model)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

train_data = Dataset.load_from_disk(DATA_PATH / "gsm8k/train_dataset_rlhf")
val_data = Dataset.load_from_disk(DATA_PATH / "gsm8k/test_dataset_rlhf")
args = GRPOTrainArgs(
    n_grpo_steps= 200,
    train_batch_size= 256 , # On-policy
    gradient_accumulation_steps= 128, # microbatch size is 2, will fit on H100
    reward_fn = "r1_zero_reward",
    eval_batch_size= 512,
    eval_interval= 10,
    advantage_eps= 1e-6,
    learning_rate= 2.5e-5,
    rollout_batch_size= 256,
    group_size= 8,
    sampling_temperature= 1.0,
    sampling_min_tokens= 4 ,
    sampling_max_tokens= 1024,
    epochs_per_rollout_batch= 1 , 
    loss_type= "grpo_clip",
    importance_sample_level = "token",
    use_std_normalization = True,
    cliprange= 0.2,  
    max_grad_norm= 1.0, 
    output_dir= "train_grpo_" + time.strftime("%Y%m%d%H%M%S"),
    vllm_kwargs={
        "gpu_memory_utilization": 0.5,
        "device": "cuda:3",
    },
    use_8bit_adamw= True,
    use_swanlab= True,
)

if __name__ == "__main__":
    trainer = GRPOTrainer(
        policy=model,
        tokenizer=tokenizer,
        train_data=train_data,
        val_data=val_data,
        train_args=args,
    )
    trainer.train()

    #model.save_pretrained("grop_lora_model")