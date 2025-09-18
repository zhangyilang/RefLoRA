from typing import Optional, Union
from dataclasses import dataclass, field


@dataclass
class PEFTArguments:
    """
    Arguments pertaining to PEFT.
    """

    lora_r: Optional[int] = field(default=8, metadata={"help": "Lora attention dimension"})
    lora_alpha: Optional[int] = field(default=16, metadata={"help": "Lora alpha"})
    lora_dropout: Optional[float] = field(default=0.0, metadata={"help": "Lora dropout"})
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with LoRA."
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'."
                "This can also be a wildcard 'all-linear' which matches all linear/Conv1D "
                "(if the model is a PreTrainedModel, the output layer excluded)."
                "If not specified, modules will be chosen according to the model architecture, If the architecture is "
                "not known, an error will be raised -- in this case, you should specify the target modules manually."
            ),
        },
    )
    init_lora_weights: Optional[str] = field(
        default=True,
        metadata={
            "help": (
                "How to initialize the weights of the LoRA layers. "
                "Passing True (default) results in the default initialization from the reference implementation from "
                "Microsoft, with the LoRA B weight being set to 0. This means that without further training, the LoRA "
                "adapter will be a no-op. "
                "Setting the initialization to False leads to random initialization of LoRA A and B, meaning that LoRA "
                "is not a no-op before training; this setting is intended for debugging purposes. "
                "Passing `'gaussian'` results in Gaussian initialization scaled by the LoRA rank for linear and layers. "
                "Passing `'eva'` results in a data-driven initialization of Explained Variance Adaptation. "
                "Passing `'olora'` results in OLoRA initialization. "
                "Passing `'pissa'` results in PiSSA initialization. "
                "Passing `'pissa_niter_[number of iters]'` initiates Fast-SVD-based PiSSA initialization, where "
                "[number of iters] indicates the number of subspace iterations to perform fsvd, and must be a "
                "nonnegative integer. "
                "Passing `'corda'` results in CorDA initialization. "
                "Pass `'loftq'` to use LoftQ initialization."
            ),
        },
    )
    refactor: Optional[bool] = field(default=False, metadata={"help": "Whether to use RefLoRA or not"})
    use_scalar: Optional[bool] = field(default=False, metadata={"help": "Whether to use RefLoRA-S or not"})
    reflora_warmup: Optional[int] = field(default=100, metadata={"help": "Number of warmup steps for RefLoRA"})
    reflora_interval: Optional[int] = field(default=1, metadata={"help": "Interval of Refactoring; default to 1."})
