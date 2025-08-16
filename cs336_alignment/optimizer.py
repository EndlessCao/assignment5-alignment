import torch
from typing import Callable, Optional, Iterable
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas = (0.9, 0.999), weight_decay = 0.01, eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        beta1, beta2 = betas
        if beta1 < 0 or beta1 >= 1:
            raise ValueError(f"Invalid beta1: {beta1}")
        if beta2 < 0 or beta2 >= 1:
            raise ValueError(f"Invalid beta2: {beta2}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon: {eps}")
            
        defaults = {
            "lr": lr,
            "beta1": beta1,
            "beta2": beta2,
            "weight_decay": weight_decay,
            "eps": eps,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                state = self.state[p]
                
                # 初始化状态
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)  # 一阶动量
                    state["exp_avg_sq"] = torch.zeros_like(p.data)  # 二阶动量
                state["step"] += 1
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                
                
                # 更新一阶动量和二阶动量
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1) # m = beta1 * m + (1 - beta1) * grad
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2) # v = beta2 * v + (1 - beta2) * grad * grad
                
                # 计算偏差修正
                bias_correction1 = 1 - beta1 ** state["step"] # (1 - beta1 ** t)
                bias_correction2 = 1 - beta2 ** state["step"] # (1 - beta2 ** t)
                
                # 更新参数
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1 # lr * sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                denom = exp_avg_sq.sqrt().add_(eps)  # sqrt(v) + eps
                p.data.addcdiv_(exp_avg, denom, value=-step_size) # p = p - step_size * m / (sqrt(v) + eps ) / sqrt(1 - beta2 ** t)
                
                # AdamW的权重衰减
                p.data.add_(p.data, alpha=-lr * weight_decay)
                
                
        return loss
    def set_lr(self, lr):
        for group in self.param_groups:
            group['lr'] = lr

    
class CosineScheduler:
    def __init__(self, amax, amin, Tw, Tc):
        self.amax = amax
        self.amin = amin
        self.Tw = Tw
        self.Tc = Tc
    def __call__(self, t):
        if t < self.Tw:
            return self.amax * t / self.Tw
        if t <= self.Tc:
            return self.amin + 0.5 * (1 + math.cos((t - self.Tw) / (self.Tc - self.Tw) * math.pi)) * (self.amax - self.amin)
        return self.amin
    


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps=1e-6):
    # 计算所有参数的梯度的L2范数，不使用stack以避免形状不一致的问题
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.detach().norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    # 如果总范数大于最大允许范数,则进行截断
    clip_coef = max_l2_norm / (total_norm + eps)
    clip_coef = min(clip_coef, 1.0)
    
    # 对所有参数的梯度进行截断
    for p in parameters:
        if p.grad is not None:
            p.grad.detach().mul_(clip_coef)