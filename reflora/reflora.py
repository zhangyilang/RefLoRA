import math
import torch

from torch import nn

from peft import PeftModel
from peft.tuners.lora.model import LoraModel
from peft.tuners.lora.layer import LoraLayer, Linear, Embedding
from peft.utils.other import transpose


class Refactorer:
    def __init__(self, 
                 model: nn.Module, 
                 warmup_steps: int = 0, 
                 re_init = False, 
                 interval: int = 1, 
                 use_scalar: bool = False
                 ) -> None:
        self.warmup_steps = warmup_steps
        self.use_scalar = use_scalar
        self.interval = interval

        self.lora_weights = []
        for module in self.get_lora_model(model).modules():
            if not isinstance(module, LoraLayer):
                continue
            if not isinstance(module, (Linear, Embedding)):
                raise ValueError(f"Unsupported LoraLayer type: {type(module)}")
            
            for adapter in module.active_adapters:
                if isinstance(module, Linear):
                    lora_A = module.lora_A[adapter].weight
                    lora_B = module.lora_B[adapter].weight
                elif isinstance(module, Embedding):
                    lora_A = module.self.lora_embedding_A[adapter]
                    lora_B = module.self.lora_embedding_B[adapter]
                self.lora_weights.append((lora_A, lora_B))  # (A: r x m, B: n x r)

                if re_init:
                    nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
                    nn.init.kaiming_uniform_(lora_B, a=math.sqrt(5))
                    dtype = module.get_base_layer().weight.dtype
                    weight = transpose(module.get_base_layer().weight.data.to(torch.float32), module.fan_in_fan_out)
                    weight -= module.scaling[adapter] * lora_B @ lora_A
                    module.get_base_layer().weight.data = transpose(weight.to(dtype), module.fan_in_fan_out)

    @torch.no_grad()
    def dummy_step(self) -> None:
        '''Refactoring via preconditioning'''
        if self.skip_steps > 0:
            self.skip_steps -= 1
            return
        else:
            self.skip_steps = self.interval - 1
        
        for lora_A, lora_B in self.lora_weights:
            if self.use_scalar:
                S = torch.linalg.norm(lora_B.data) / torch.linalg.norm(lora_A.data)
                Sinv = 1 / S
                
                lora_A.grad *= Sinv
                lora_B.grad *= S
            else:
                eps = torch.finfo(lora_A.dtype).eps
                sigmaA_sq, VA = torch.linalg.eigh(lora_A.data @ lora_A.data.t())
                sigmaA = torch.sqrt(sigmaA_sq)

                # S_A V_A^T B^T B V_A S_A = V_M S_M^2 V_M^T
                M_right = lora_B.data @ (VA * sigmaA)   # broadcast multiplication for diagonal matrix
                sigmaM_sq, VM = torch.linalg.eigh(M_right.t() @ M_right)
                sigmaM = torch.sqrt(sigmaM_sq)

                # S = V_A S_A^{-1} V_M S_M V_M^T S_A^{-1} V_A^T
                S_left = (VA * (1 / (sigmaA + eps))) @ VM
                S = S_left * sigmaM @ S_left.t()

                # S^{-1} = V_A S_A V_M S_M^{-1} V_M^T S_A V_A^T
                Sinv_left = (VA * sigmaA) @ VM
                Sinv = (Sinv_left * (1 / (sigmaM + eps))) @ Sinv_left.t()

                lora_A.grad = Sinv @ lora_A.grad    # A is r x m
                lora_B.grad @= S
    
    @torch.no_grad()
    def step(self, optimizer) -> None:
        '''Real refactoring'''
        if self.skip_steps > 0:
            self.skip_steps -= 1
            return
        else:
            self.skip_steps = self.interval - 1

        for lora_A, lora_B in self.lora_weights:
            if self.use_scalar:
                S = torch.linalg.norm(lora_B.data) / torch.linalg.norm(lora_A.data)
                P = torch.sqrt(S)
                
                lora_A.data *= P
                lora_B.data *= 1 / P

                if optimizer.state:
                    optimizer.state[lora_A]['exp_avg'] *= 1 / P
                    optimizer.state[lora_A]['exp_avg_sq'] *= 1 / S
                    optimizer.state[lora_B]['exp_avg'] *= P
                    optimizer.state[lora_B]['exp_avg_sq'] *= S
            else:
                eps = torch.finfo(lora_A.dtype).eps
                sigmaA_sq, VA = torch.linalg.eigh(lora_A.data @ lora_A.data.t())
                sigmaA = torch.sqrt(sigmaA_sq)

                # S_A V_A^T B^T B V_A S_A = V_M S_M^2 V_M^T
                M_right = lora_B.data @ (VA * sigmaA)   # broadcast multiplication for diagonal matrix
                sigmaM_sq, VM = torch.linalg.eigh(M_right.t() @ M_right)
                sigmaM_sqrt = torch.pow(sigmaM_sq, 0.25)

                # P = V_A S_A^{-1} V_M S_M^{1/2}
                Pt = ((VA * (1 / (sigmaA + eps))) @ (VM * sigmaM_sqrt)).t()

                # P^{-T} = V_A S_A V_M S_M^{-1/2}
                Ptinv = (VA * sigmaA) @ (VM * (1 / sigmaM_sqrt))

                lora_A.data = Pt @ lora_A.data    # A is r x m
                lora_B.data @= Ptinv

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
    
    def integrate_into_optimizer(self, optimizer) -> None:
        self.skip_steps = self.warmup_steps

        if self.use_scalar:
            def refactor_hook(optimizer, args, kwargs) -> None:
                self.step(optimizer)
            optimizer.register_step_post_hook(refactor_hook)
            self.step(optimizer)     # need one initial call as we use post hook
        else:
            def refactor_hook(optimizer, args, kwargs) -> None:
                self.dummy_step()
            optimizer.register_step_pre_hook(refactor_hook)
