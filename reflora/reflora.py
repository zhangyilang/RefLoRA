import math
import warnings
from collections import defaultdict
from typing import Optional

import torch

from torch import nn

from peft import PeftModel
from peft.tuners.lora.model import LoraModel
from peft.tuners.lora.layer import LoraLayer, Embedding


# Optimizer-state entries that RefLoRA-S rescales after refactoring the weights in place
_FIRST_MOMENT_KEYS = ("exp_avg", "momentum_buffer")
_SECOND_MOMENT_KEYS = ("exp_avg_sq", "max_exp_avg_sq")


def _moments(optimizer: torch.optim.Optimizer, param: torch.Tensor) -> list:
    '''(tensor, order) for each first (order 1) and second (order 2) moment of `param` in the optimizer state.'''
    moments = []
    for key, value in optimizer.state.get(param, {}).items():
        if key == "step" or not torch.is_tensor(value):
            continue
        if key in _FIRST_MOMENT_KEYS:
            moments.append((value, 1))
        elif key in _SECOND_MOMENT_KEYS:
            moments.append((value, 2))
        else:
            raise RuntimeError(f"RefLoRA-S cannot rescale optimizer state '{key}' of {type(optimizer).__name__}; "
                               "supported optimizers are Adam/AdamW (including AMSGrad) and SGD")
    return moments


def _compute_dtype(dtype: torch.dtype) -> torch.dtype:
    # torch.linalg.eigh does not support half precision; fp32/fp64 are kept as is
    return torch.promote_types(dtype, torch.float32)


class Refactorer:
    '''Optimal refactoring of LoRA factors (RefLoRA, or RefLoRA-S with `use_scalar=True`).

    Args:
        model: a `PeftModel`/`LoraModel`, or any module containing PEFT LoRA layers.
        warmup_steps: optimizer steps run as vanilla LoRA before refactoring starts. PEFT initializes
            `lora_B = 0`, which violates the full-rank assumption, so a short warmup (e.g., 100) or
            `re_init=True` is recommended.
        re_init: randomly re-initialize both LoRA factors and subtract their product from the frozen
            base weight, so the model is unchanged but both factors are full rank.
        interval: refactor every `interval` optimizer steps after warmup.
        use_scalar: use the lightweight RefLoRA-S (scalar refactoring).
        batched: batch the refactoring of all layers (eigendecompositions for RefLoRA, norms and
            rescaling for RefLoRA-S), which is much faster on GPUs than the original per-layer
            computation (`batched=False`) and gives the same results up to floating-point round-off.
    '''

    def __init__(self,
                 model: nn.Module,
                 warmup_steps: int = 0,
                 re_init: bool = False,
                 interval: int = 1,
                 use_scalar: bool = False,
                 batched: bool = True
                 ) -> None:
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")
        if interval < 1:
            raise ValueError(f"interval must be a positive integer, got {interval}")

        self.warmup_steps = warmup_steps
        self.use_scalar = use_scalar
        self.interval = interval
        self.batched = batched
        self.skip_steps = warmup_steps
        self._hook_handle = None
        self._batches = None
        self._batch_signature = None

        self.lora_weights = []
        for module in self.get_lora_model(model).modules():
            if not isinstance(module, LoraLayer):
                continue

            for adapter in module.active_adapters:
                if isinstance(module, Embedding):
                    if adapter not in module.lora_embedding_A:
                        continue
                    lora_A = module.lora_embedding_A[adapter]
                    lora_B = module.lora_embedding_B[adapter]
                else:   # Linear and its quantized variants (e.g., bitsandbytes)
                    if adapter not in module.lora_A:
                        continue
                    lora_A = module.lora_A[adapter].weight
                    lora_B = module.lora_B[adapter].weight
                if lora_A.ndim != 2 or lora_B.ndim != 2:
                    raise ValueError(f"Unsupported LoraLayer type: {type(module)}; "
                                     "only Linear-like and Embedding layers are supported")
                self.lora_weights.append((lora_A, lora_B))  # (A: r x m, B: n x r)

                if re_init:
                    self._re_init(module, adapter, lora_A, lora_B)

        if not self.lora_weights:
            warnings.warn("No LoRA layers found in the model; Refactorer will have no effect.")
        elif warmup_steps == 0 and any(not A.any() or not B.any() for A, B in self.lora_weights):
            warnings.warn("Some LoRA factors are all zeros (PEFT initializes lora_B = 0). Refactoring is "
                          "skipped for those layers until both factors are nonzero; consider "
                          "warmup_steps > 0 (e.g., 100) or re_init=True.")

    @staticmethod
    @torch.no_grad()
    def _re_init(module: LoraLayer, adapter: str, lora_A: torch.Tensor, lora_B: torch.Tensor) -> None:
        base_weight = module.get_base_layer().weight
        if not base_weight.dtype.is_floating_point:
            raise ValueError("re_init=True requires a non-quantized base layer")

        nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(lora_B, a=math.sqrt(5))

        # keep W_pt + scaling * B @ A unchanged
        weight = base_weight.data.to(torch.float32)
        weight -= module.get_delta_weight(adapter).to(torch.float32)
        base_weight.data = weight.to(base_weight.dtype)

    def _skip(self) -> bool:
        '''Advance the warmup/interval schedule by one step; return True if this step is skipped.'''
        if self.skip_steps > 0:
            self.skip_steps -= 1
            return True
        self.skip_steps = self.interval - 1
        return False

    @staticmethod
    def _safe_ratio(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
        '''num / den, or 1 (no refactoring) if either is zero. Avoids a host sync.'''
        ratio = num / den
        valid = (num > 0) & (den > 0)
        return torch.where(valid, ratio, torch.ones_like(ratio))

    @staticmethod
    def _preconditioners(lora_A: torch.Tensor, lora_B: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        '''S = (A^T A)^{-1} # (B^T B) in (9) and its inverse, via two r x r eigendecompositions.'''
        dtype = _compute_dtype(lora_A.dtype)
        A, B = lora_A.to(dtype), lora_B.to(dtype)
        eps = torch.finfo(dtype).eps

        # clamp round-off negative eigenvalues, which would turn into NaN under sqrt
        sigmaA_sq, VA = torch.linalg.eigh(A @ A.t())
        sigmaA = torch.sqrt(sigmaA_sq.clamp_min(0))

        # S_A V_A^T B^T B V_A S_A = V_M S_M^2 V_M^T
        M_right = B @ (VA * sigmaA)   # broadcast multiplication for diagonal matrix
        sigmaM_sq, VM = torch.linalg.eigh(M_right.t() @ M_right)
        sigmaM = torch.sqrt(sigmaM_sq.clamp_min(0))

        # S = V_A S_A^{-1} V_M S_M V_M^T S_A^{-1} V_A^T
        S_left = (VA * (1 / (sigmaA + eps))) @ VM
        S = S_left * sigmaM @ S_left.t()

        # S^{-1} = V_A S_A V_M S_M^{-1} V_M^T S_A V_A^T
        Sinv_left = (VA * sigmaA) @ VM
        Sinv = (Sinv_left * (1 / (sigmaM + eps))) @ Sinv_left.t()

        # a zero factor (e.g., lora_B = 0 at init) gives S = 0; fall back to vanilla LoRA (S = I)
        valid = sigmaM_sq[-1] > 0
        eye = torch.eye(S.shape[0], dtype=dtype, device=S.device)
        return torch.where(valid, S, eye), torch.where(valid, Sinv, eye)

    def _get_batches(self) -> list:
        '''Group layers sharing rank, device, and dtypes; within a group, index the layers by factor shape.

        Cached, and rebuilt only if a device or dtype changes (e.g., the model is moved after creating
        the Refactorer). Index tensors live on the device to avoid host syncs.
        '''
        signature = [(A.device, A.dtype, B.dtype) for A, B in self.lora_weights]
        if signature == self._batch_signature:
            return self._batches

        groups = defaultdict(list)
        for i, (A, B) in enumerate(self.lora_weights):
            groups[(A.shape[0], A.device, A.dtype, B.dtype)].append(i)

        def by_shape(pairs, factor, device):
            shapes = defaultdict(list)
            for pos, pair in enumerate(pairs):
                shapes[tuple(pair[factor].shape)].append(pos)
            return [(positions, torch.tensor(positions, device=device)) for positions in shapes.values()]

        self._batches = []
        for (_, device, _, _), layers in groups.items():
            pairs = [self.lora_weights[i] for i in layers]
            self._batches.append((pairs, by_shape(pairs, 0, device), by_shape(pairs, 1, device)))
        self._batch_signature = signature
        return self._batches

    @staticmethod
    def _batched_preconditioners(pairs: list, by_A: list, by_B: list) -> tuple[torch.Tensor, torch.Tensor]:
        '''Same as `_preconditioners` for a group of layers, with one batched eigh per stage.'''
        A0 = pairs[0][0]
        L, r, device = len(pairs), A0.shape[0], A0.device
        dtype = _compute_dtype(A0.dtype)
        eps = torch.finfo(dtype).eps

        G = torch.empty(L, r, r, dtype=dtype, device=device)
        for positions, index in by_A:
            X = torch.stack([pairs[p][0].data for p in positions]).to(dtype)
            G.index_copy_(0, index, X @ X.mT)
        sigmaA_sq, VA = torch.linalg.eigh(G)
        sigmaA = torch.sqrt(sigmaA_sq.clamp_min(0))
        VAS = VA * sigmaA.unsqueeze(-2)

        H = torch.empty(L, r, r, dtype=dtype, device=device)
        for positions, index in by_B:
            M = torch.stack([pairs[p][1].data for p in positions]).to(dtype) @ VAS.index_select(0, index)
            H.index_copy_(0, index, M.mT @ M)
        sigmaM_sq, VM = torch.linalg.eigh(H)
        sigmaM = torch.sqrt(sigmaM_sq.clamp_min(0))

        S_left = (VA * (1 / (sigmaA + eps)).unsqueeze(-2)) @ VM
        S = (S_left * sigmaM.unsqueeze(-2)) @ S_left.mT
        Sinv_left = VAS @ VM
        Sinv = (Sinv_left * (1 / (sigmaM + eps)).unsqueeze(-2)) @ Sinv_left.mT

        valid = (sigmaM_sq[:, -1] > 0)[:, None, None]
        eye = torch.eye(r, dtype=dtype, device=device)
        return torch.where(valid, S, eye), torch.where(valid, Sinv, eye)

    @staticmethod
    def _apply_batched(pairs: list, groups: list, factor: int, precond: torch.Tensor, active: list) -> None:
        '''Precondition the gradients of one factor: Sinv @ grad_A (factor 0) or grad_B @ S (factor 1).'''
        def apply(P, G):
            return P @ G if factor == 0 else G @ P

        for positions, index in groups:
            P = precond.index_select(0, index)
            grads = [pairs[p][factor].grad for p in positions]
            if all(active[p] for p in positions):
                new = apply(P, torch.stack(grads).to(P.dtype))
                for j, (p, grad) in enumerate(zip(positions, grads)):
                    pairs[p][factor].grad = new[j].to(grad.dtype)
            else:
                for j, (p, grad) in enumerate(zip(positions, grads)):
                    if active[p]:
                        pairs[p][factor].grad = apply(P[j], grad.to(P.dtype)).to(grad.dtype)

    def _batched_dummy_step(self) -> None:
        for pairs, by_A, by_B in self._get_batches():
            active = [A.grad is not None and B.grad is not None for A, B in pairs]
            if not any(active):
                continue
            S, Sinv = self._batched_preconditioners(pairs, by_A, by_B)
            self._apply_batched(pairs, by_A, 0, Sinv, active)    # A is r x m
            self._apply_batched(pairs, by_B, 1, S, active)

    @staticmethod
    def _refactor_matrices(lora_A: torch.Tensor, lora_B: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        '''P^T and P^{-T} with P P^T = S in (9).'''
        dtype = _compute_dtype(lora_A.dtype)
        A, B = lora_A.to(dtype), lora_B.to(dtype)
        eps = torch.finfo(dtype).eps

        sigmaA_sq, VA = torch.linalg.eigh(A @ A.t())
        sigmaA = torch.sqrt(sigmaA_sq.clamp_min(0))

        # S_A V_A^T B^T B V_A S_A = V_M S_M^2 V_M^T
        M_right = B @ (VA * sigmaA)   # broadcast multiplication for diagonal matrix
        sigmaM_sq, VM = torch.linalg.eigh(M_right.t() @ M_right)
        sigmaM_sqrt = torch.pow(sigmaM_sq.clamp_min(0), 0.25)

        # P = V_A S_A^{-1} V_M S_M^{1/2}
        Pt = ((VA * (1 / (sigmaA + eps))) @ (VM * sigmaM_sqrt)).t()

        # P^{-T} = V_A S_A V_M S_M^{-1/2}
        Ptinv = (VA * sigmaA) @ (VM * (1 / (sigmaM_sqrt + eps)))

        valid = sigmaM_sq[-1] > 0
        eye = torch.eye(Pt.shape[0], dtype=dtype, device=Pt.device)
        return torch.where(valid, Pt, eye), torch.where(valid, Ptinv, eye)

    @staticmethod
    def _rescale_state(optimizer: torch.optim.Optimizer,
                       param: torch.Tensor,
                       first: torch.Tensor,
                       second: torch.Tensor
                       ) -> None:
        for value, order in _moments(optimizer, param):
            value *= first if order == 1 else second

    @staticmethod
    def _rescale_states(optimizer: torch.optim.Optimizer, params: list, first: list, second: list) -> None:
        '''Batched `_rescale_state` with per-parameter Python scalars, using fused multi-tensor kernels.'''
        tensors, scalars = ([], []), ([], [])
        for param, f, s in zip(params, first, second):
            for value, order in _moments(optimizer, param):
                tensors[order - 1].append(value)
                scalars[order - 1].append(f if order == 1 else s)
        for ts, ss in zip(tensors, scalars):
            if ts:
                torch._foreach_mul_(ts, ss)

    def _batched_scalar_step(self, optimizer: Optional[torch.optim.Optimizer]) -> None:
        for pairs, _, _ in self._get_batches():
            As, Bs = [A.data for A, _ in pairs], [B.data for _, B in pairs]
            S = self._safe_ratio(torch.stack(torch._foreach_norm(Bs)), torch.stack(torch._foreach_norm(As)))
            P = torch.sqrt(S)
            # one host sync, so that all layers are rescaled by a few fused kernels
            P_list, Pinv_list, S_list, Sinv_list = torch.stack([P, 1 / P, S, 1 / S]).tolist()

            torch._foreach_mul_(As, P_list)
            torch._foreach_mul_(Bs, Pinv_list)
            if optimizer is not None:
                self._rescale_states(optimizer, [A for A, _ in pairs], first=Pinv_list, second=Sinv_list)
                self._rescale_states(optimizer, [B for _, B in pairs], first=P_list, second=S_list)

    @torch.no_grad()
    def dummy_step(self) -> None:
        '''Refactoring via preconditioning (11); runs before each optimizer step.'''
        if self._skip():
            return

        if self.batched and not self.use_scalar:
            self._batched_dummy_step()
            return

        for lora_A, lora_B in self.lora_weights:
            if lora_A.grad is None or lora_B.grad is None:
                continue

            if self.use_scalar:
                S = self._safe_ratio(torch.linalg.norm(lora_B.data), torch.linalg.norm(lora_A.data))
                Sinv = 1 / S

                lora_A.grad *= Sinv
                lora_B.grad *= S
            else:
                S, Sinv = self._preconditioners(lora_A.data, lora_B.data)
                grad_A, grad_B = lora_A.grad, lora_B.grad
                lora_A.grad = (Sinv @ grad_A.to(Sinv.dtype)).to(grad_A.dtype)    # A is r x m
                lora_B.grad = (grad_B.to(S.dtype) @ S).to(grad_B.dtype)

    @torch.no_grad()
    def step(self, optimizer: Optional[torch.optim.Optimizer] = None) -> None:
        '''Real refactoring of the LoRA weights; runs after each optimizer step.

        RefLoRA-S also rescales the optimizer moments accordingly. Full RefLoRA cannot transform
        Adam's second moment; with stateful optimizers, use `integrate_into_optimizer`, which
        preconditions the gradients (`dummy_step`) instead.
        '''
        if self._skip():
            return

        if not self.use_scalar and optimizer is not None and optimizer.state:
            warnings.warn("Full RefLoRA refactors the weights without transforming the optimizer state; "
                          "with stateful optimizers (e.g., Adam), use integrate_into_optimizer instead.")

        if self.batched and self.use_scalar:
            self._batched_scalar_step(optimizer)
            return

        for lora_A, lora_B in self.lora_weights:
            if self.use_scalar:
                S = self._safe_ratio(torch.linalg.norm(lora_B.data), torch.linalg.norm(lora_A.data))
                P = torch.sqrt(S)

                lora_A.data *= P
                lora_B.data *= 1 / P

                if optimizer is not None:
                    self._rescale_state(optimizer, lora_A, first=1 / P, second=1 / S)
                    self._rescale_state(optimizer, lora_B, first=P, second=S)
            else:
                Pt, Ptinv = self._refactor_matrices(lora_A.data, lora_B.data)
                lora_A.data = (Pt @ lora_A.data.to(Pt.dtype)).to(lora_A.dtype)    # A is r x m
                lora_B.data = (lora_B.data.to(Ptinv.dtype) @ Ptinv).to(lora_B.dtype)

    def fast_forward(self, num_steps: int) -> None:
        '''Advance the warmup/interval schedule by `num_steps` optimizer steps without refactoring.

        Used when resuming from a checkpoint. Optimizer steps skipped by an fp16 grad scaler do not
        trigger the hooks, so the schedule may drift slightly from the original run during warmup.
        '''
        if num_steps <= self.skip_steps:
            self.skip_steps -= num_steps
        else:
            remaining = num_steps - self.skip_steps - 1
            self.skip_steps = (self.interval - 1) - remaining % self.interval

    @staticmethod
    def get_lora_model(model: nn.Module) -> LoraModel:
        # PeftModel.base_model is a BaseTuner (e.g., LoraModel)
        if isinstance(model, LoraModel):
            lora_model = model
        elif isinstance(model, PeftModel):
            lora_model = model.base_model
        else:
            return model
            # raise ValueError(f"Expected model to be PeftModel or LoraModel, but got {type(model)} instead")

        return lora_model

    def integrate_into_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        '''Register refactoring as an optimizer hook; calling it again replaces the previous hook.'''
        if self._hook_handle is not None:
            self._hook_handle.remove()
        self.skip_steps = self.warmup_steps

        if self.use_scalar:
            def refactor_hook(optimizer, args, kwargs) -> None:
                self.step(optimizer)
            self._hook_handle = optimizer.register_step_post_hook(refactor_hook)
            self.step(optimizer)     # need one initial call as we use post hook
        else:
            def refactor_hook(optimizer, args, kwargs) -> None:
                self.dummy_step()
            self._hook_handle = optimizer.register_step_pre_hook(refactor_hook)
