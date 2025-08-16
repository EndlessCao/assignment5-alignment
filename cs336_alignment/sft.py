import os
from typing import Optional, Callable, Union, Literal
import gc
from torch.utils.data import Dataset, DataLoader
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from transformers import PreTrainedModel, PreTrainedTokenizer
from tools import tokenize_prompt_and_output
from tqdm import tqdm
from dataclasses import dataclass, field
import vllm
from pprint import pprint
from vllm.lora.request import LoRARequest
from vllm_utils import init_vllm, load_policy_into_vllm_instance
import uuid
from optimizer import AdamW, CosineScheduler, gradient_clipping
import pathlib
import wandb

def tokenize_prompt_and_output(prompts, outputs, tokenizer: PreTrainedTokenizer):
    prompt_ids = []
    output_ids = []
    resp_mask = []
    for prompt, output in zip(prompts, outputs):
        prompt_id = tokenizer(prompt, padding=True, truncation=True)
        output_id = tokenizer(output, padding=True, truncation=True)
        output_ids.append(prompt_id['input_ids'] + output_id['input_ids'])
        resp_mask.append([0] * len(prompt_id['input_ids']) + [1] * len(output_id['input_ids']))
    batch_size = len(output_ids)
    max_seq_len = max(len(output_id) for output_id in output_ids)
    ids_tensor = torch.full((batch_size, max_seq_len), tokenizer.pad_token_id, dtype=torch.long)
    resp_mask_tensor = torch.full((batch_size, max_seq_len), 0, dtype=torch.long)
    for i, (output_id, resp_mask) in enumerate(zip(output_ids, resp_mask)):
        ids_tensor[i, :len(output_id)] = torch.tensor(output_id, dtype=torch.long)
        resp_mask_tensor[i, :len(resp_mask)] = torch.tensor(resp_mask, dtype=torch.long)
    return {
        "input_ids": ids_tensor[:, :-1],
        "labels": ids_tensor[:, 1:],
        "response_mask": resp_mask_tensor[:, 1:]
    }

def compute_entropy(logits: Tensor):
    # -sigma p(x)log(p(x))
    p = F.log_softmax(logits, dim=-1)
    return -torch.sum(torch.exp(p) * p, dim=-1)


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    logits = model(input_ids).logits
    log_probs = F.log_softmax(logits, dim=-1) # (batch_size, seq_len, vocab_size)
    res = {}
    res["log_probs"] = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1) # (batch_size, seq_len)
    if return_token_entropy:
        res["token_entropy"] = compute_entropy(logits) # (batch_size, seq_len)
    return res

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    masked_tensor = tensor * mask
    if dim is None:
        return masked_tensor.sum() / normalize_constant
    return masked_tensor.sum(dim=dim) /  normalize_constant

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch_size, seq_len = policy_log_probs.shape
    loss_sum = -masked_normalize(policy_log_probs, response_mask, normalize_constant)
    loss = loss_sum / batch_size / gradient_accumulation_steps
    loss.backward()
    metadata = {
        
    }
    return loss.detach(), metadata

@dataclass
class SFTArgs:
    lr: float
    num_epoch: int
    batch_size: int
    micro_batch_size: int
    eval_interval: int
    eval_batch_size: int
    max_grad_norm: float = field(default=1.0)
    normalize_constant: float = field(default=1.0)
    gradient_accumulation_steps: int = field(default=1)
    lr_max: float = field(default=5e-4)
    lr_min: float = field(default=1e-6)
    warm_up: int = field(default=10)
    scheduler: Callable | None = field(default=None)
    save_path: str| os.PathLike = field(default=None)
    use_wandb: bool = field(default=False)
    wandb_project: Optional[str] = field(default=None)
    wandb_name: Optional[str] = field(default=None)
    # 显存优化参数
    empty_cache_steps: int = field(default=100)  # 每N步清理缓存
    # for expert iteration
    group_size :int = field(default=8)
    reward_fn: Callable = field(default=None)
    vllm_kwargs: dict | None = None
    save_strategy: Literal["epoch", "steps", "eval"] = field(default="epoch")
    save_steps: int = field(default=100)
class SftTrainer:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, train_data, val_data, args:SFTArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = AdamW(self.model.parameters(), lr=args.lr)
        self.scheduler = args.scheduler if args.scheduler else CosineScheduler(args.lr_max, args.lr_min, args.warm_up, 10000)
        self.train_data = train_data
        self.val_data = val_data
        self.args = args
        self.gradient_accumulation_steps = args.gradient_accumulation_steps if args.gradient_accumulation_steps else 1
        self.normalize_constant = args.normalize_constant if args.normalize_constant else 1.0
        self.eval_interval = args.eval_interval if args.eval_interval else 1000
        self.batch_size = args.batch_size if args.batch_size else 4
        assert self.batch_size % self.gradient_accumulation_steps == 0
        self.micro_batch_size = self.args.micro_batch_size if self.args.micro_batch_size else 1
        self.save_path = pathlib.Path(args.save_path).resolve().absolute() if args.save_path else pathlib.Path.cwd()
        self.device = self.model.device
        self.enable_lora = hasattr(self.model, 'peft_config')
        self.reward_fn = args.reward_fn
        self.vllm_kwargs = self.args.vllm_kwargs
        if self.vllm_kwargs is not None:
            self.vllm_model = self._init_vllm()
        if self.reward_fn is None:
            raise ValueError("reward_fn must be set.")
        if self.args.use_wandb and (self.args.wandb_project is None or self.args.wandb_name is None):
            raise ValueError("wandb_project or wandb_name must be set.")
        self.eval_batch_size = self.args.eval_batch_size
        self.eval_dataloader = DataLoader(
            self.val_data,
            batch_size=self.eval_batch_size,
            shuffle=False,
        )
    
    def _init_vllm(self) -> vllm.LLM:
        llm = init_vllm(self.model.config.name_or_path, 
                                    device = self.vllm_kwargs['device'], 
                                    gpu_memory_utilization = self.vllm_kwargs['gpu_memory_utilization'])
        print("Load vLLM model successfully.")
        return llm
    
    def train(self):
        print("Training on device: ", self.model.device)
        self.model.train()
        if self.args.use_wandb:
            wandb.init(project=self.args.wandb_project, name=self.args.wandb_name)
            wandb.config.learning_rate = self.args.lr
            wandb.config.num_epoch = self.args.num_epoch
            wandb.config.batch_size = self.args.batch_size
            wandb.config.gradient_accumulation_steps = self.args.gradient_accumulation_steps
            wandb.config.normalize_constant = self.args.normalize_constant
        num_epoch = self.args.num_epoch

        total_batch = len(self.train_data) // self.batch_size
        for epoch in range(num_epoch):
            process_bar = tqdm(range(total_batch), total = total_batch, desc=f'Training epoch {epoch + 1}/{num_epoch}')
            global_step = 0
            for i in process_bar:
                batch = self.train_data[i * self.batch_size: (i + 1) * self.batch_size]
                prompt, label = batch["prompt"], batch["label"]
                lossess = []
                for j in range(0, self.batch_size, self.micro_batch_size):
                    micro_batch_prompt, micro_batch_label = prompt[j: j + self.micro_batch_size], label[j: j + self.micro_batch_size]
                    micro_batch_data = tokenize_prompt_and_output(micro_batch_prompt, micro_batch_label, self.tokenizer)
                    micro_input_ids, micro_labels, micro_response_mask = micro_batch_data["input_ids"].to(self.device), micro_batch_data["labels"].to(self.device), micro_batch_data["response_mask"].to(self.device)
                    log_probs = get_response_log_probs(self.model, micro_input_ids, micro_labels)["log_probs"]
                    loss, _ = sft_microbatch_train_step(
                        log_probs,
                        micro_response_mask,
                        gradient_accumulation_steps=self.gradient_accumulation_steps,
                        normalize_constant=self.normalize_constant,
                    )
                    lossess.append(loss.item())
                    micro_step = (j // self.micro_batch_size) + 1
                    if (micro_step) % self.gradient_accumulation_steps == 0:
                        gradient_clipping(self.model.parameters(), self.args.max_grad_norm)
                        update_step = global_step // self.gradient_accumulation_steps
                        lr = self.scheduler(update_step)
                        self.optimizer.set_lr(lr)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    global_step += 1
                self._log_loss(process_bar, sum(lossess) / len(lossess), i)
                if (i + 1) % self.eval_interval == 0:
                    eval_metrics = self.eval()
                    pprint(eval_metrics)
                    if self.args.save_strategy == "eval": 
                        self.model.save_pretrained(self.save_path / f"checkpoint_{epoch}_{i + 1}")
                if self.args.save_strategy == "steps" and (i + 1) % self.args.save_steps == 0:
                    self.model.save_pretrained(self.save_path / f"checkpoint_{epoch}_{i + 1}")
            if self.args.save_strategy == "epoch":
                self.model.save_pretrained(self.save_path / f"checkpoint_{epoch}")
        self.model.save_pretrained(self.save_path / "checkpoint_final")

    def eval(self):
        if self.vllm_model is not None:
            self._load_policy_to_vllm()
        eval_metrics = []
        for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
            responses = self._generate_responses(batch['prompt'])
            for resp, gt in zip(responses, batch['answer']):
                metadata = self.reward_fn(resp, gt)
                eval_metrics.append(metadata)
        avg_reward = sum(i['reward'] for i in eval_metrics) / len(eval_metrics)
        avg_format_reward = sum(i['format_reward'] for i in eval_metrics) / len(eval_metrics)
        accuracy = sum(i['answer_reward'] for i in eval_metrics) / len(eval_metrics)
        print(f"Evaluation completed - Avg Reward: {avg_reward:.4f}, Avg Format Reward: {avg_format_reward:.4f}, Accuracy: {accuracy:.4f}")
        return {
            'avg_reward': avg_reward,
            'avg_format_reward': avg_format_reward,
            'accuracy': accuracy
        }

    def _log_loss(self, process_bar, loss, step):
        process_bar.set_postfix({
            "loss": loss,
        })
        if self.args.use_wandb:
            wandb.log({"epoch": step, "loss": loss})
    def _load_policy_to_vllm(self):
        if self.enable_lora:
            self.model.merge_adapter()
            for name, param in self.model.named_parameters():
                name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                if self.model.prefix in name or "original_module" in name:
                    continue
                def _fix_param_name_to_vllm(name, extra_prefixes: list[str] | None= None):
                    extra_prefixes = extra_prefixes or []
                    prefixes = ["_checkpoint_wrapped_module."] + extra_prefixes
                    for prefix in prefixes:
                        name = name.replace(prefix, "")
                    return name
                name = _fix_param_name_to_vllm(name, extra_prefixes=["modules_to_save.default."])
                llm_model = self.vllm_model.llm_engine.model_executor.driver_worker.model_runner.model
                llm_model.load_weights([(name, param.data)])
            self.model.unmerge_adapter()
        else:
            load_policy_into_vllm_instance(self.model, self.vllm_model)

    def _generate_responses(self, prompts: list[str], n_samples = 1) -> list[str]:
        sampling_params = vllm.SamplingParams(
            temperature=1.0,
            min_tokens=4,
            max_tokens=1024,
            top_p=1.0,
            n = n_samples,
            stop=[r"</answer>","<|im_end|>"],
            include_stop_str_in_output=True,
        )
        output_lists = self.vllm_model.generate(prompts,sampling_params,use_tqdm=False)
        responses = []
        for output_list in output_lists:
            temp = []
            for output in output_list.outputs:
                temp.append(output.text)
            responses.extend(temp)
        return responses
    def train_expert(self):
        print("Training on device: ", self.device)
        if self.args.use_wandb:
            wandb.init(project=self.args.wandb_project, name=self.args.wandb_name)
            wandb.config.learning_rate = self.args.lr
            wandb.config.num_epoch = self.args.num_epoch
            wandb.config.batch_size = self.args.batch_size
            wandb.config.gradient_accumulation_steps = self.args.gradient_accumulation_steps
            wandb.config.normalize_constant = self.args.normalize_constant
        num_epoch = self.args.num_epoch

        total_batch = len(self.train_data) // self.batch_size

        for epoch in range(num_epoch):
            self.model.train()
            process_bar = tqdm(range(total_batch), total = total_batch, desc=f'Training epoch {epoch + 1}/{num_epoch}')
            for i in process_bar:
                batch = self.train_data[i * self.batch_size: (i + 1) * self.batch_size]
                questions, groundtruths = batch["prompt"], batch["answer"]
                responses = self._generate_responses(questions, self.args.group_size)
                labels = []
                for candidates, ground_truth in zip(responses, groundtruths):
                    best_candidate = candidates[0]
                    best_reward = -float("inf")
                    for candidate in candidates:
                        reward = self.reward_fn(candidate, ground_truth)['reward']
                        if reward > best_reward:
                            best_reward = reward
                            best_candidate = candidate
                    labels.append(best_candidate)
                for j in range(0, self.batch_size, self.micro_batch_size):
                    micro_batch_prompt, micro_batch_label = questions[j: j + self.micro_batch_size], labels[j: j + self.micro_batch_size]
                    data = tokenize_prompt_and_output(micro_batch_prompt, micro_batch_label, self.tokenizer)
                    micro_input_ids, micro_labels, micro_response_mask = data["input_ids"].to(self.device), data["labels"].to(self.device), data["response_mask"].to(self.device)
                    log_probs = get_response_log_probs(self.model, micro_input_ids, micro_labels)["log_probs"]
                    loss, metadata = sft_microbatch_train_step(
                        log_probs,
                        micro_response_mask,
                        gradient_accumulation_steps=self.gradient_accumulation_steps,
                        normalize_constant=self.normalize_constant,
                    )
                    if ((j + 1) // self.micro_batch_size) % self.gradient_accumulation_steps == 0:
                        gradient_clipping(self.model.parameters(), self.args.max_grad_norm)
                        lr = self.scheduler(i)
                        self.optimizer.set_lr(lr)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        torch.cuda.empty_cache()
                    self._log_loss(process_bar, loss.item(), i)
                if (i + 1) % self.eval_interval == 0:
                    eval_metrics = self.eval()
                    pprint(eval_metrics)
                    self.model.save_pretrained(self.save_path / f"checkpoint_{i + 1}")
                
        self.model.save_pretrained(self.save_path / "checkpoint_final")
        
        