import pathlib
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Callable, Tuple, Literal
from dataclasses import dataclass, field, asdict
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
import torch.nn.functional as F
import vllm
from vllm_utils import init_vllm, load_policy_into_vllm_instance
from tqdm import tqdm
from drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn
import bitsandbytes as bnb
import numpy as np
import pandas as pd
import swanlab

def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> Tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    rollout_batch_size = len(repeated_ground_truths)
    assert rollout_batch_size % group_size == 0
    advantages = torch.zeros(rollout_batch_size, dtype=torch.float32)
    rewards = torch.zeros(rollout_batch_size, dtype=torch.float32)
    for i in range(0, rollout_batch_size, group_size):
        batch_resp = rollout_responses[i : i + group_size]
        batch_gt = repeated_ground_truths[i : i + group_size]
        batch_advantages = []
        for resp, gt in zip(batch_resp, batch_gt):
            reward_dict = reward_fn(resp, gt)
            reward = reward_dict.get("reward", 0)
            batch_advantages.append(reward)
        rewards[i : i + group_size] = torch.tensor(batch_advantages, dtype=torch.float32)
        denom = advantage_eps + rewards[i : i + group_size].std() if normalize_by_std else 1
        advantages[i : i + group_size] = (rewards[i : i + group_size] - rewards[i : i + group_size].mean()) / denom

    metadata = {
        "reward_mean": rewards.mean().item(),
        "reward_std": rewards.std().item(),
        "advantage_mean": advantages.mean().item(),
        "advantage_std": advantages.std().item(),
        "response_length": sum([len(resp) for resp in rollout_responses]) // len(rollout_responses),
    }
    return advantages, rewards.detach(), metadata


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    return -raw_rewards_or_advantages * policy_log_probs


def compute_grpo_clip_loss(
    advantages: torch.Tensor,  # (batch_size, 1)
    policy_log_probs: torch.Tensor,  # (batch_size, sequence_length)
    old_log_probs: torch.Tensor,  # (batch_size, sequence_length)
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ratio = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - cliprange, 1 + cliprange)
    loss = -torch.min(ratio, clipped_ratio) * advantages
    return loss, {"ratio": ratio.detach(), "clipped_ratio": clipped_ratio.detach()}


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if loss_type == "no_baseline":
        assert raw_rewards is not None
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        return loss, {}
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        return loss, {}
    elif loss_type == "grpo_clip":
        assert advantages is not None
        assert old_log_probs is not None
        assert cliprange is not None
        loss, metadata = compute_grpo_clip_loss(advantages,policy_log_probs,old_log_probs,cliprange,)
        return loss, metadata


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    masked_tensor = tensor * mask
    mask_sum = mask.sum(dim=dim)
    # 防止除零错误导致NaN
    mask_sum = torch.clamp(mask_sum, min=1e-8)
    return masked_tensor.sum(dim=dim) / mask_sum


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:

    loss, metadata = compute_policy_gradient_loss(
        policy_log_probs,
        loss_type,
        raw_rewards,
        advantages,
        old_log_probs,
        cliprange,
    ) #(B, S)

    loss = masked_mean(loss, response_mask, dim=-1).mean()
    loss = loss / gradient_accumulation_steps  # (1)
    loss.backward()
    return loss.detach(), metadata


@dataclass
class GRPOTrainArgs:
    reward_fn: str = "r1_zero_reward"
    n_grpo_steps: int = 200
    learning_rate: float = 1e-5
    advantage_eps: float = 1e-6
    rollout_batch_size: int = 4
    group_size: int = 4
    sampling_temperature: float = 1.0
    sampling_min_tokens: int = 4  # As in Expiter, disallow empty string responses
    sampling_max_tokens: int = 1024
    epochs_per_rollout_batch: int = 1  # On-policy
    train_batch_size: int = 4  # On-policy
    gradient_accumulation_steps: int = 1  # microbatch size is 2, will fit on H100
    eval_batch_size: int = 8
    eval_interval: int = 1000
    loss_type: Literal[
        "no_baseline",
        "reinforce_with_baseline",
        "grpo_clip",
    ] = "reinforce_with_baseline"
    importance_sample_level: Literal["token", "sentence"] = "sentence"
    use_std_normalization: bool = True
    cliprange: float = 0.2  
    max_grad_norm: float = 1.0  
    output_dir: str = "outputs"
    vllm_kwargs: dict | None = None 
    use_8bit_adamw: bool = False
    use_swanlab: bool = False
    
class GRPOTrainer:
    def __init__(
        self,
        policy: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_data: Dataset,
        val_data: Dataset,
        train_args: GRPOTrainArgs,
    ):
        self.policy = policy
        self.tokenizer = tokenizer
        self.args = train_args
        self.train_data = train_data
        self.val_data = val_data
        self.reward_fn = (
            r1_zero_reward_fn
            if self.args.reward_fn == "r1_zero_reward"
            else question_only_reward_fn
        )
        self.trainable_parameters = [param for param in self.policy.parameters() if param.requires_grad]
        #self.backup_params = [param.detach().cpu() for param in self.trainable_parameters]
        if self.args.use_8bit_adamw:
            self.optimizer = bnb.optim.AdamW8bit(
                self.trainable_parameters,
                lr=self.args.learning_rate,
                weight_decay=0.0,
                betas=(0.9, 0.95),
            )
        else:
            self.optimizer = torch.optim.AdamW(
                self.trainable_parameters,
                lr=self.args.learning_rate,
                weight_decay=0.0,
                betas=(0.9, 0.95),
            )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.args.n_grpo_steps
        )
        self.use_peft = hasattr(self.policy, "peft_config")
        self.vllm_kwargs = self.args.vllm_kwargs
        self.vllm_model = None
        if self.vllm_kwargs is not None:
            self.vllm_model = self._init_vllm()
        
        self.output_dir = pathlib.Path(self.args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.output_dir / "train.log"
        self.log_file.touch(exist_ok=True)
        # Save training args to json file
        args_dict = asdict(self.args)
        args_dict["model_name"] = self.policy.config.name_or_path
        args_file = self.output_dir / "train_args.json"
        import json

        with open(args_file, "w") as f:
            json.dump(args_dict, f, indent=4)
        self.use_swanlab = self.args.use_swanlab
        if self.use_swanlab:
            import time
            swanlab.init(
                project="GRPO_Train",
                name=time.strftime("%Y%m%d%H%M%S"),
                config=args_dict,
            )
        assert (
            self.args.train_batch_size % self.args.gradient_accumulation_steps == 0
        ), "train_batch_size must be divisible by gradient_accumulation_steps"
        self.micro_train_batch_size = (
            self.args.train_batch_size // self.args.gradient_accumulation_steps
        )

        assert (
            self.args.rollout_batch_size % self.args.group_size == 0
        ), "rollout_batch_size must be divisible by group_size"
        self.n_prompts_per_rollout_batch = (
            self.args.rollout_batch_size // self.args.group_size
        )

        assert (
            self.args.train_batch_size >= self.args.group_size
        ), "train_batch_size must be greater than or equal to group_size"
        self.n_microbatches_per_rollout_batch = (
            self.args.rollout_batch_size // self.micro_train_batch_size
        )

        # 初始化数据加载器
        self.train_dataloader = DataLoader(
            self.train_data,
            batch_size=self.n_prompts_per_rollout_batch,
            shuffle=True,
            collate_fn=self._collate_fn,
        )
        self.eval_batch_size = self.args.eval_batch_size
        self.eval_dataloader = DataLoader(
            self.val_data,
            batch_size=self.eval_batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
        )

    def _collate_fn(self, batch):
        """数据批处理函数"""
        prompts = [item["prompt"] for item in batch]
        ground_truths = [item.get("ground_truth", "") for item in batch]
        return {"prompts": prompts, "ground_truths": ground_truths}

    def _init_vllm(self) -> vllm.LLM:
        llm = init_vllm(
            self.policy.config.name_or_path,
            device=self.vllm_kwargs["device"],
            gpu_memory_utilization=self.vllm_kwargs["gpu_memory_utilization"],
        )
        print("Load vLLM model successfully.")
        return llm

    def _load_policy_to_vllm(self):
        assert self.vllm_model is not None
        # 加载策略到vLLM模型
        if self.use_peft:
            self.policy.merge_adapter()
            for name, param in self.policy.named_parameters():
                name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                if self.policy.prefix in name or "original_module" in name:
                    continue
                def _fix_param_name_to_vllm(name, extra_prefixes: list[str] | None = None):
                    extra_prefixes = extra_prefixes or []
                    prefixes = ["_checkpoint_wrapped_module."] + extra_prefixes
                    for prefix in prefixes:
                        name = name.replace(prefix, "")
                    return name

                name = _fix_param_name_to_vllm(
                    name, extra_prefixes=["modules_to_save.default."]
                )
                llm = self.vllm_model.llm_engine.model_executor.driver_worker.model_runner.model
                llm.load_weights([(name, param.data)])
            self.policy.unmerge_adapter()
        else:
            load_policy_into_vllm_instance(self.policy, self.vllm_model)

    def _generate_responses(self, prompts: list[str], n_samples=1) -> list[str]:
        assert self.vllm_model is not None
        # 生成响应
        sampling_params = vllm.SamplingParams(
            temperature=self.args.sampling_temperature,
            min_tokens=self.args.sampling_min_tokens,
            max_tokens=self.args.sampling_max_tokens,
            top_p=1.0,
            n=n_samples,
            stop=[r"</answer>", "<|im_end|>"],
            include_stop_str_in_output=True,
        )
        output_lists = self.vllm_model.generate(
            prompts, sampling_params, use_tqdm=False
        )
        responses = []
        for output_list in output_lists:
            for output in output_list.outputs:
                responses.append(output.text)
        return responses

    def evaluate(self, output_dir: pathlib.Path | None = None):
        if self.vllm_model is None:
            self.vllm_model = self._init_vllm()
        self._load_policy_to_vllm()
        eval_metrics = []
        data_list = []
        for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
            responses = self._generate_responses(batch["prompts"])
            for prompt, resp, gt in zip(batch["prompts"], responses, batch["ground_truths"]):
                metadata = self.reward_fn(resp, gt)
                eval_metrics.append(metadata)
                data_list.append({"prompt": prompt, "response": resp, "ground_truth": gt, **metadata})
        if output_dir:
            df = pd.DataFrame(data_list)
            output_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_dir / f"eval_results.csv", index=False)
        avg_reward = sum(i["reward"] for i in eval_metrics) / len(eval_metrics)
        avg_format_reward = sum(i["format_reward"] for i in eval_metrics) / len(eval_metrics)
        accuracy = sum(i["answer_reward"] for i in eval_metrics) / len(eval_metrics)
        print(f"Evaluation completed - Avg Reward: {avg_reward:.4f}, Avg Format Reward: {avg_format_reward:.4f}, Accuracy: {accuracy:.4f}")
        return {
            "avg_reward": avg_reward,
            "avg_format_reward": avg_format_reward,
            "accuracy": accuracy,
        }

    def _log(self, data: dict):
        import json
        with open(self.log_file, "a") as f:
            f.write(json.dumps(data) + "\n")
        if self.use_swanlab:
            swanlab.log(data)
    
    
    def _get_log_probs(self, prompts: list[str], responses: list[str], loss_level: str = "token"):
        full_texts = [p + r for p, r in zip(prompts, responses)]
        inputs = self.tokenizer(full_texts,return_tensors="pt",padding=True,padding_side="left",truncation=True,max_length=self.args.sampling_max_tokens,
        ).to(self.policy.device)
        prompt_tokens = self.tokenizer(prompts,return_tensors="pt",padding=True,padding_side="left",truncation=True,max_length=self.args.sampling_max_tokens,
        ).input_ids
        
        logits = self.policy(**inputs).logits # (batch_size, seq_len, vocab_size)
        logits = logits[:, :-1, :] # (batch_size, seq_len-1, vocab_size)
        log_probs = F.log_softmax(logits, dim=-1) # (batch_size, seq_len-1, vocab_size)
        log_probs = torch.gather(log_probs, 2, inputs.input_ids[:, 1:].unsqueeze(-1)).squeeze(-1) # (batch_size, seq_len-1)

        response_mask = torch.ones_like(log_probs)
        for j in range(len(prompts)):
            prompt_len = torch.sum(prompt_tokens[j] != self.tokenizer.pad_token_id)
            pad_len = torch.sum(inputs.input_ids[j] == self.tokenizer.pad_token_id)
            response_mask[j, : pad_len + prompt_len - 1] = 0
        if loss_level == "sentence":
            log_probs = masked_mean(log_probs, response_mask, dim=1).unsqueeze(-1) # (batch_size, 1)
        return log_probs, response_mask

    def train(self):
        print("Start training.")
        global_step = 0
        process_bar = tqdm(self.train_dataloader, total=min(len(self.train_dataloader), self.args.n_grpo_steps), desc="Training")
        for grpo_step, batch in enumerate(process_bar):
            if grpo_step >= self.args.n_grpo_steps:
                break
            self.policy.eval()
            if self.vllm_model is not None:
                self._load_policy_to_vllm()

            prompts = batch["prompts"]
            ground_truths = batch["ground_truths"]

            responses = self._generate_responses(prompts, n_samples=self.args.group_size)

            rollout_prompts = [p for p in prompts for _ in range(self.args.group_size)]
            repeated_ground_truths = [gt for gt in ground_truths for _ in range(self.args.group_size)]

            advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
                self.reward_fn,
                responses,
                repeated_ground_truths,
                self.args.group_size,
                self.args.advantage_eps,
                self.args.use_std_normalization,
            )
            advantages = advantages.unsqueeze(-1)
            raw_rewards = raw_rewards.unsqueeze(-1)
            self._log(reward_metadata)
            if reward_metadata["reward_mean"] <= 0:
                print("Reward mean is zero, stop train.")
                break
            with torch.no_grad():
                if self.args.loss_type == "grpo_clip":
                    old_log_probs = []
                    for i in range(0, len(responses), self.micro_train_batch_size):
                        micro_batch_prompts = rollout_prompts[i : i + self.micro_train_batch_size]
                        micro_batch_responses = responses[i : i + self.micro_train_batch_size]
                        masked_log_probs, _ = self._get_log_probs(micro_batch_prompts, micro_batch_responses, self.args.importance_sample_level)
                        old_log_probs.append(masked_log_probs.detach())
                else:
                    old_log_probs = None

            # === Training Phase ===
            self.policy.train()
            
            for _ in range(self.args.epochs_per_rollout_batch):
                
                losses = []
                for i in range(0, len(responses), self.micro_train_batch_size):
                    global_step += 1
                    micro_batch_slice = slice(i, i + self.micro_train_batch_size)
                    batch_prompts = rollout_prompts[micro_batch_slice]
                    batch_responses = responses[micro_batch_slice]
                    batch_advantages = advantages[micro_batch_slice].to(self.policy.device)
                    batch_raw_rewards = raw_rewards[micro_batch_slice].to(self.policy.device)
                    batch_old_log_probs = old_log_probs[i // self.micro_train_batch_size].to(self.policy.device) if old_log_probs  else None

                    policy_log_probs, response_mask = self._get_log_probs(batch_prompts, batch_responses, self.args.importance_sample_level)

                    loss, metadata = grpo_microbatch_train_step(
                        policy_log_probs=policy_log_probs,
                        response_mask=response_mask,
                        gradient_accumulation_steps=self.args.gradient_accumulation_steps,
                        loss_type=self.args.loss_type,
                        raw_rewards=batch_raw_rewards,
                        advantages=batch_advantages,
                        old_log_probs=batch_old_log_probs,
                        cliprange=self.args.cliprange,
                    )

                    log_data = {
                        "loss": loss.item(),
                    }
                    losses.append(loss.item())
                    if metadata:
                        log_data.update(
                            {k: v.mean().item() for k, v in metadata.items()}
                        )

                    if ((i // self.micro_train_batch_size) + 1) % self.args.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.trainable_parameters, self.args.max_grad_norm)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                self._log({"loss": np.mean(losses)})
            self.scheduler.step()
            if (grpo_step + 1) % self.args.eval_interval == 0:
                self.policy.eval()
                eval_metrics = self.evaluate(output_dir=self.output_dir / f"checkpoint-step-{grpo_step + 1}")
                self._log(eval_metrics) 
                self.policy.save_pretrained(self.output_dir / f"checkpoint-step-{grpo_step + 1}")
                self.tokenizer.save_pretrained(self.output_dir / f"checkpoint-step-{grpo_step + 1}")
        self.policy.save_pretrained(self.output_dir / f"checkpoint-final")
        self.tokenizer.save_pretrained(self.output_dir / f"checkpoint-final")