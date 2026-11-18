"""Tests for RefLoRA; run with `pytest` (CPU is enough)."""
import pytest
import torch
from torch import nn
from peft import LoraConfig, get_peft_model

from reflora import Refactorer, RefTrainer
from reference_reflora import Refactorer as ReferenceRefactorer


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(50, 16)
        self.fc1 = nn.Linear(16, 24)
        self.fc2 = nn.Linear(24, 12)

    def forward(self, x, labels=None):
        out = self.fc2(torch.tanh(self.fc1(self.emb(x))))
        return out if labels is None else {"loss": ((out - labels) ** 2).mean(), "logits": out}


def make_model(targets=("fc1", "fc2"), seed=0):
    torch.manual_seed(seed)
    return get_peft_model(Net(), LoraConfig(r=4, lora_alpha=4, target_modules=list(targets)))


def make_data(n, seed=1):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randint(0, 50, (32,), generator=g), torch.randn(32, 12, generator=g)) for _ in range(n)]


def train(model, refactorer, make_optimizer=None, steps=10):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = make_optimizer(params) if make_optimizer else torch.optim.AdamW(params, lr=1e-2)
    refactorer.integrate_into_optimizer(optimizer)
    for x, y in make_data(steps):
        optimizer.zero_grad()
        ((model(x).float() - y) ** 2).mean().backward()
        optimizer.step()
    return optimizer


def lora_params(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if "lora" in n}


def all_finite(model):
    return all(torch.isfinite(p).all() for p in lora_params(model).values())


# ---------- identical to the paper implementation ----------

@pytest.mark.parametrize("use_scalar", [False, True])
@pytest.mark.parametrize("warmup", [2, 5])
@pytest.mark.parametrize("interval", [1, 3])
def test_matches_reference(use_scalar, warmup, interval):
    kwargs = dict(use_scalar=use_scalar, warmup_steps=warmup, interval=interval)
    ref_model, model = make_model(), make_model()
    train(ref_model, ReferenceRefactorer(ref_model, **kwargs))
    train(model, Refactorer(model, batched=False, **kwargs))
    expected, actual = lora_params(ref_model), lora_params(model)
    assert all_finite(ref_model)
    assert all(torch.equal(expected[k], actual[k]) for k in expected)


def test_re_init_matches_reference():
    ref_model, model = make_model(), make_model()
    torch.manual_seed(5)
    ReferenceRefactorer(ref_model, re_init=True)
    torch.manual_seed(5)
    Refactorer(model, re_init=True)
    expected, actual = dict(ref_model.named_parameters()), dict(model.named_parameters())
    assert all(torch.equal(expected[k], actual[k]) for k in expected)


# ---------- math ----------

@pytest.fixture
def factors():
    torch.manual_seed(0)
    # code layout: lora_A is r x m (A^T in the paper), lora_B is n x r
    return torch.randn(4, 30, dtype=torch.float64), torch.randn(20, 4, dtype=torch.float64)


def test_preconditioner_is_geometric_mean(factors):
    A, B = factors
    S, Sinv = Refactorer._preconditioners(A, B)
    eye = torch.eye(4, dtype=torch.float64)
    assert torch.allclose(S, S.t())
    assert torch.linalg.eigvalsh(S).min() > 0
    assert torch.allclose(S @ Sinv, eye, atol=1e-8)
    assert torch.allclose(S @ A @ A.t() @ S, B.t() @ B, atol=1e-8)    # S A^T A S = B^T B, cf. (21)


def test_refactoring_balances_and_keeps_product(factors):
    A, B = factors
    S, _ = Refactorer._preconditioners(A, B)
    Pt, Ptinv = Refactorer._refactor_matrices(A, B)
    A_new, B_new = Pt @ A, B @ Ptinv
    assert torch.allclose(Pt.t() @ Pt, S, atol=1e-8)
    assert torch.allclose(A_new @ A_new.t(), B_new.t() @ B_new, atol=1e-8)
    assert torch.allclose(B_new @ A_new, B @ A, atol=1e-8)


# ---------- batched eigendecompositions ----------

def test_batched_matches_per_layer():
    # layers of different shapes (Embedding + Linear) and ranks, so that several batches are formed
    torch.manual_seed(0)
    config = LoraConfig(r=4, lora_alpha=4, target_modules=["emb", "fc1", "fc2"], rank_pattern={"fc2": 2})
    model = get_peft_model(Net(), config)
    train(model, Refactorer(model, warmup_steps=100), steps=3)     # vanilla steps so that lora_B != 0
    x, y = make_data(1, seed=7)[0]
    model.zero_grad()
    ((model(x) - y) ** 2).mean().backward()

    per_layer, batched = Refactorer(model, batched=False), Refactorer(model, batched=True)
    saved = [(A.grad.clone(), B.grad.clone()) for A, B in per_layer.lora_weights]
    per_layer.dummy_step()
    expected = [(A.grad.clone(), B.grad.clone()) for A, B in per_layer.lora_weights]
    for (A, B), (gA, gB) in zip(batched.lora_weights, saved):
        A.grad, B.grad = gA.clone(), gB.clone()
    batched.dummy_step()

    assert len(batched._get_batches()) == 2
    for (A, B), (eA, eB) in zip(batched.lora_weights, expected):
        assert torch.allclose(A.grad, eA, rtol=1e-4, atol=1e-6)
        assert torch.allclose(B.grad, eB, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("make_optimizer", [None, lambda p: torch.optim.AdamW(p, lr=1e-2, amsgrad=True),
                                            lambda p: torch.optim.SGD(p, lr=1e-2, momentum=0.9)])
def test_scalar_batched_matches_per_layer(make_optimizer):
    models = {}
    for batched in (False, True):
        torch.manual_seed(0)
        config = LoraConfig(r=4, lora_alpha=4, target_modules=["emb", "fc1", "fc2"], rank_pattern={"fc2": 2})
        models[batched] = get_peft_model(Net(), config)
        train(models[batched], Refactorer(models[batched], use_scalar=True, warmup_steps=2, batched=batched),
              make_optimizer)
    expected, actual = lora_params(models[False]), lora_params(models[True])
    assert all(torch.allclose(expected[k], actual[k], rtol=1e-4, atol=1e-6) for k in expected)


def test_batched_skips_layers_without_grad():
    model = make_model()
    for name, param in model.named_parameters():
        if "lora" in name and "fc2" in name:
            param.requires_grad_(False)
    frozen = {k: v for k, v in lora_params(model).items() if "fc2" in k}
    train(model, Refactorer(model, warmup_steps=2, batched=True))
    after = lora_params(model)
    assert all_finite(model)
    assert all(torch.equal(frozen[k], after[k]) for k in frozen)


# ---------- layers, precision, and initialization ----------

@pytest.mark.parametrize("use_scalar", [False, True])
def test_embedding_layers(use_scalar):
    model = make_model(targets=("emb", "fc1"))
    refactorer = Refactorer(model, use_scalar=use_scalar, warmup_steps=2)
    train(model, refactorer)
    assert len(refactorer.lora_weights) == 2
    assert all_finite(model)
    if use_scalar:
        A, B = refactorer.lora_weights[0]
        assert torch.allclose(A.norm(), B.norm(), rtol=1e-5)


@pytest.mark.parametrize("batched", [False, True])
def test_bf16_adapters(batched):
    model = make_model().to(torch.bfloat16)
    train(model, Refactorer(model, warmup_steps=2, batched=batched), steps=5)
    assert all_finite(model)


@pytest.mark.parametrize("use_scalar,batched", [(False, False), (False, True), (True, False), (True, True)])
def test_zero_init_without_warmup(use_scalar, batched):
    model = make_model()
    with pytest.warns(UserWarning, match="all zeros"):
        refactorer = Refactorer(model, use_scalar=use_scalar, warmup_steps=0, batched=batched)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-2)
    refactorer.integrate_into_optimizer(optimizer)
    lora_B = refactorer.lora_weights[0][1]
    snapshots = []
    for x, y in make_data(6):
        optimizer.zero_grad()
        ((model(x) - y) ** 2).mean().backward()
        optimizer.step()
        snapshots.append(lora_B.detach().clone())
    assert all_finite(model)
    assert snapshots[0].abs().sum() > 0
    assert all(not torch.equal(a, b) for a, b in zip(snapshots, snapshots[1:]))    # B is not frozen


def test_re_init_keeps_output():
    model = make_model(targets=("emb", "fc1", "fc2"))
    x = torch.randint(0, 50, (32,))
    with torch.no_grad():
        before = model(x)
    Refactorer(model, re_init=True)
    with torch.no_grad():
        after = model(x)
    assert torch.allclose(before, after, atol=1e-5)


# ---------- optimizers ----------

@pytest.mark.parametrize("batched", [False, True])
def test_scalar_amsgrad(batched):
    model = make_model()
    refactorer = Refactorer(model, use_scalar=True, warmup_steps=2, batched=batched)
    optimizer = train(model, refactorer, lambda p: torch.optim.AdamW(p, lr=1e-2, amsgrad=True))
    assert all_finite(model)
    for param in (p for pair in refactorer.lora_weights for p in pair):
        state = optimizer.state[param]
        assert (state["max_exp_avg_sq"] >= state["exp_avg_sq"]).all()


@pytest.mark.parametrize("batched", [False, True])
def test_scalar_sgd_momentum(batched):
    model = make_model()
    train(model, Refactorer(model, use_scalar=True, warmup_steps=2, batched=batched), lambda p: torch.optim.SGD(p, lr=1e-2, momentum=0.9))
    assert all_finite(model)


@pytest.mark.parametrize("batched", [False, True])
def test_scalar_unsupported_optimizer_raises(batched):
    model = make_model()
    with pytest.raises(RuntimeError, match="cannot rescale"):
        train(model, Refactorer(model, use_scalar=True, warmup_steps=2, batched=batched), lambda p: torch.optim.Adagrad(p, lr=1e-2))


@pytest.mark.parametrize("use_scalar", [False, True])
def test_integrate_twice_registers_one_hook(use_scalar):
    once, twice = make_model(), make_model()
    train(once, Refactorer(once, use_scalar=use_scalar, warmup_steps=2))
    refactorer = Refactorer(twice, use_scalar=use_scalar, warmup_steps=2)
    params = [p for p in twice.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-2)
    refactorer.integrate_into_optimizer(optimizer)
    train(twice, refactorer, lambda p: optimizer)
    expected, actual = lora_params(once), lora_params(twice)
    assert all(torch.equal(expected[k], actual[k]) for k in expected)


# ---------- schedule and resuming ----------

@pytest.mark.parametrize("warmup", range(6))
@pytest.mark.parametrize("interval", range(1, 5))
def test_fast_forward_matches_stepping(warmup, interval):
    for num_steps in range(16):
        stepped, forwarded = Refactorer.__new__(Refactorer), Refactorer.__new__(Refactorer)
        for r in (stepped, forwarded):
            r.interval, r.skip_steps = interval, warmup
        for _ in range(num_steps):
            stepped._skip()
        forwarded.fast_forward(num_steps)
        assert stepped.skip_steps == forwarded.skip_steps


@pytest.mark.parametrize("use_scalar", [False, True])
def test_trainer_resume_matches_uninterrupted(use_scalar, tmp_path):
    pytest.importorskip("accelerate")
    from transformers import TrainingArguments

    samples = [(x[i:i + 8], y[i:i + 8]) for x, y in make_data(8, seed=3) for i in range(0, 32, 8)]
    dataset = [{"x": x, "labels": y} for x, y in samples]

    def run(resume=None):
        model = make_model()
        refactorer = Refactorer(model, use_scalar=use_scalar, warmup_steps=6)
        args = TrainingArguments(output_dir=str(tmp_path), max_steps=10, save_steps=4, per_device_train_batch_size=1,
                                 learning_rate=1e-2, lr_scheduler_type="constant", report_to="none", use_cpu=True,
                                 logging_steps=100, seed=0, save_safetensors=False, disable_tqdm=True)
        trainer = RefTrainer(model=model, args=args, train_dataset=dataset, refactorer=refactorer,
                             data_collator=lambda batch: batch[0])
        trainer.train(resume_from_checkpoint=resume)
        return lora_params(model), refactorer.skip_steps

    # warmup ends after the checkpoint at step 4, so a restarted warmup would skip refactoring entirely
    expected, expected_skip = run()
    actual, actual_skip = run(resume=str(tmp_path / "checkpoint-4"))
    assert expected_skip == actual_skip
    assert all(torch.allclose(expected[k], actual[k], atol=1e-6) for k in expected)
