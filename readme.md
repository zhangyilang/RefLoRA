# [NeurIPS 2025] RefLoRA
[![NeurIPS](https://img.shields.io/badge/NeurIPS-openreview-8c1b13)](https://openreview.net/forum?id=zefDc9oi5T) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

This repository provides codes for reproducing the results in our NeurIPS 2025 paper [RefLoRA: Refactored Low-Rank Adaptation for Efficient Fine-Tuning of Large Models](https://openreview.net/pdf?id=zefDc9oi5T). 

## Overview

This paper deals with the non-unique factorization challenge in low-rank adaptation (LoRA), which leads to inconsistent updates, unbalanced weights, and slow convergence. Specifically, for equivalent low-rank factorizations $\mathbf{A} \mathbf{B}^\top = \tilde{\mathbf{A}}\tilde{\mathbf{B}}^\top$, the resultant weight increment can differ remarkably; see analysis in our paper. 

**Key idea:** RefLoRA identifies the *optimal factorization* that minimizes the loss upper bound. We prove that this solution admits a closed-form expression, resulting in flatter loss landscape that facilitates stable and efficient optimization. 

With the optimal refactoring, RefLoRA guarantees consistent and balanced weight updates, as well as faster empirical convergence. A simplified variant termed RefLoRA-S is developed to further reduce the overhead.

<p float="left">
    <img src="assets/MF-loss.png" alt="matrix factorization" height=300 />
    <img src="assets/loss-epoch.png" alt="glue" height=300 />
    <img src="assets/overhead.png" alt="overheads" height=350 />
</p>

## Experiments
### Setup

Our codes are tested with python 3.12, and packages speficied in `requirements.txt`, where our RefLoRA package is installed in the editable mode. 

```bash
pip install -r requirements.txt
```

Next, download the commonsense reasoning datasets from [LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters), and prepare them as
```bash
commonsense_reasoning/
├── dataset/
│   ├── ARC-Challenge/
│   ├── ARC-Easy/
│   ├── boolq/
│   ├── hellaswag/
│   ├── openbookqa/
│   ├── piqa/
│   ├── social_i_qa/
│   └── winogrande/
└── ft-training_set/
    └── commonsense_170k.json

```

### General usage
RefLoRA combines seamlessly with Hugging Face’s `Trainer` class via just three lines of modification, replacing `Trainer` with `RefTrainer`. Set `use_scalar=True` to use RefLoRA-S. 
```python
from reflora import Refactorer, RefTrainer

refactorer = Refactorer(model, use_scalar=False, warmup_steps=100)
trainer = RefTrainer(*args, **kwargs, refactorer=refactorer)
```

Alternatively, it can be directly integrated into a `torch.optim.Optimizer` via

```python
from reflora import Refactorer

refactorer = Refactorer(model, use_scalar=False, warmup_steps=100)
refactorer.integrate_into_optimizer(optimizer)
```



### Natural language understanding

![GLUE](assets/tab-glue.png)


The following commands are used to reproduce the results for the GLUE benchmark. Hyperparameters are offered in Table 6 of Appendix D.6. 

```bash
cd glue
bash debertaV3-base.sh
```

### Commonsense reasoning

![GLUE](assets/tab-common.png)

To reproduce results on commonsense reasoning datasets:

```bash
cd commonsense_reasoning
bash scripts/llama_7B.sh
bash scripts/llama2_7B.sh
bash scripts/llama3_8B.sh
```

## Credits
Our implementation builds upon the following repositories:

- https://github.com/AGI-Edgerunners/LLM-Adapters
- https://github.com/NVlabs/DoRA

## Citation
If you find this work useful, please consider citing:
> Y. Zhang, B. Li, and G. B. Giannakis, “RefLoRA: Refactored Low-Rank Adaptation for Efficient Fine-Tuning of Large Models,” in *Proceedings of Advances in Neural Information Processing Systems (NeurIPS)*, 2025. 

```tex
@inproceedings{RefLoRA, 
  title={RefLo{RA}: Refactored Low-Rank Adaptation for Efficient Fine-Tuning of Large Models},
  author={Yilang Zhang and Bingcong Li and Georgios B. Giannakis},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=zefDc9oi5T}
}

```
