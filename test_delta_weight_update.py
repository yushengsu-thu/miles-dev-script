#!/usr/bin/env python3
"""
Test script for Tensor-level Delta Weight Update.

Validates the DeltaWeightFilter module that powers --delta-weight-update
across all three Miles weight transfer modes (colocated, broadcast, P2P).

Run:  python test_delta_weight_update.py
Deps: torch  (no GPU required)
"""

import sys
import time

import torch

sys.path.insert(0, "/home/ubuntu/yushengsu/miles")
from miles.backends.megatron_utils.update_weight.delta_filter import DeltaWeightFilter


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_model_state(num_layers: int = 32, hidden: int = 3584) -> dict[str, torch.Tensor]:
    """Mimic HF-format named tensors for a Qwen-class model."""
    state: dict[str, torch.Tensor] = {}
    state["model.embed_tokens.weight"] = torch.randn(151936, hidden, dtype=torch.bfloat16)
    for i in range(num_layers):
        pfx = f"model.layers.{i}"
        state[f"{pfx}.self_attn.q_proj.weight"] = torch.randn(hidden, hidden, dtype=torch.bfloat16)
        state[f"{pfx}.self_attn.k_proj.weight"] = torch.randn(hidden // 7, hidden, dtype=torch.bfloat16)
        state[f"{pfx}.self_attn.v_proj.weight"] = torch.randn(hidden // 7, hidden, dtype=torch.bfloat16)
        state[f"{pfx}.self_attn.o_proj.weight"] = torch.randn(hidden, hidden, dtype=torch.bfloat16)
        state[f"{pfx}.mlp.gate_proj.weight"] = torch.randn(hidden * 4, hidden, dtype=torch.bfloat16)
        state[f"{pfx}.mlp.up_proj.weight"] = torch.randn(hidden * 4, hidden, dtype=torch.bfloat16)
        state[f"{pfx}.mlp.down_proj.weight"] = torch.randn(hidden, hidden * 4, dtype=torch.bfloat16)
        state[f"{pfx}.input_layernorm.weight"] = torch.randn(hidden, dtype=torch.bfloat16)
        state[f"{pfx}.post_attention_layernorm.weight"] = torch.randn(hidden, dtype=torch.bfloat16)
    state["model.norm.weight"] = torch.randn(hidden, dtype=torch.bfloat16)
    state["lm_head.weight"] = torch.randn(151936, hidden, dtype=torch.bfloat16)
    return state


def simulate_rl_step(
    state: dict[str, torch.Tensor], change_ratio: float = 0.02, lr: float = 1e-5, seed: int = 0
) -> dict[str, torch.Tensor]:
    """Simulate one RL step: small fp32 perturbation cast back to bf16."""
    torch.manual_seed(seed)
    new_state: dict[str, torch.Tensor] = {}
    for name, param in state.items():
        fp32 = param.float()
        grad = torch.randn(fp32.shape, dtype=torch.float32) * lr
        if torch.rand(1).item() > change_ratio:
            grad *= 1e-10
        new_state[name] = (fp32 + grad).bfloat16()
    return new_state


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_disabled_filter_passes_everything():
    print("TEST 1: Disabled filter passes everything through ... ", end="", flush=True)
    f = DeltaWeightFilter(enabled=False)
    tensors = [("a", torch.randn(4, 4, dtype=torch.bfloat16)),
               ("b", torch.randn(4, 4, dtype=torch.bfloat16))]
    out = f.filter(tensors)
    assert len(out) == 2
    print("PASSED")


def test_first_step_sends_all():
    print("TEST 2: First step sends all tensors (no cache yet) ... ", end="", flush=True)
    f = DeltaWeightFilter(enabled=True)
    tensors = [("a", torch.randn(4, 4, dtype=torch.bfloat16)),
               ("b", torch.randn(4, 4, dtype=torch.bfloat16))]
    out = f.filter(tensors)
    assert len(out) == 2
    f.step_done()
    print("PASSED")


def test_unchanged_tensors_skipped():
    print("TEST 3: Unchanged tensors are skipped on second call ... ", end="", flush=True)
    f = DeltaWeightFilter(enabled=True)
    t_a = torch.randn(8, 8, dtype=torch.bfloat16)
    t_b = torch.randn(8, 8, dtype=torch.bfloat16)
    f.filter([("a", t_a), ("b", t_b)])
    f.step_done()

    out = f.filter([("a", t_a.clone()), ("b", t_b.clone())])
    assert len(out) == 0, f"Expected 0 changed tensors, got {len(out)}"
    f.step_done()
    print("PASSED")


def test_changed_tensor_detected():
    print("TEST 4: Changed tensor is detected and sent ... ", end="", flush=True)
    f = DeltaWeightFilter(enabled=True)
    t_a = torch.randn(8, 8, dtype=torch.bfloat16)
    t_b = torch.randn(8, 8, dtype=torch.bfloat16)
    f.filter([("a", t_a), ("b", t_b)])
    f.step_done()

    t_b_new = t_b.clone()
    t_b_new[0, 0] += 1.0
    out = f.filter([("a", t_a.clone()), ("b", t_b_new)])
    assert len(out) == 1
    assert out[0][0] == "b"
    f.step_done()
    print("PASSED")


def test_bitwise_correctness_over_multiple_steps():
    print("TEST 5: Bitwise correctness over 5 simulated RL steps ... ", end="", flush=True)
    num_layers = 4
    state = make_model_state(num_layers=num_layers, hidden=256)

    f = DeltaWeightFilter(enabled=True)
    rollout_weights = {}

    for step in range(5):
        new_state = simulate_rl_step(state, change_ratio=0.02, lr=1e-5, seed=step)
        named_list = list(new_state.items())

        delta = f.filter(named_list)
        for name, tensor in delta:
            rollout_weights[name] = tensor.clone()

        for name in new_state:
            if name not in rollout_weights:
                rollout_weights[name] = new_state[name].clone()
                continue
            assert torch.equal(rollout_weights[name], new_state[name]), (
                f"Mismatch at step {step}, param {name}"
            )

        state = {k: v.clone() for k, v in new_state.items()}
        f.step_done()

    print("PASSED")


def test_sparsity_direct_control():
    """Directly control which tensors change to verify filter accuracy."""
    print("TEST 6: Sparsity with directly controlled changes ... ", end="", flush=True)
    num_tensors = 100
    change_count = 3

    tensors_v1 = [(f"p.{i}", torch.randn(512, 512, dtype=torch.bfloat16)) for i in range(num_tensors)]
    tensors_v2 = [(name, t.clone()) for name, t in tensors_v1]
    for i in range(change_count):
        name, t = tensors_v2[i]
        t[0, 0] += 1.0
        tensors_v2[i] = (name, t)

    f = DeltaWeightFilter(enabled=True)
    f.filter(tensors_v1)
    f.step_done()

    delta = f.filter(tensors_v2)
    f.step_done()

    assert len(delta) == change_count, f"Expected {change_count}, got {len(delta)}"
    sparsity = 1.0 - len(delta) / num_tensors
    print(f"({len(delta)}/{num_tensors} changed, sparsity={sparsity:.0%}) ", end="")
    print("PASSED")


def test_performance_overhead():
    print("TEST 7: torch.equal overhead on ~1 GB model ... ", end="", flush=True)
    state = {f"p{i}": torch.randn(2048, 2048, dtype=torch.bfloat16) for i in range(64)}

    f = DeltaWeightFilter(enabled=True)
    f.filter(list(state.items()))
    f.step_done()

    start = time.perf_counter()
    for _ in range(5):
        f.filter(list(state.items()))
        f.step_done()
    elapsed = (time.perf_counter() - start) / 5

    total_gb = sum(t.nelement() * t.element_size() for t in state.values()) / 1e9
    print(f"({total_gb:.1f} GB in {elapsed*1000:.0f} ms) ", end="")
    print("PASSED")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print(" Delta Weight Update — Test Suite")
    print("=" * 60)

    test_disabled_filter_passes_everything()
    test_first_step_sends_all()
    test_unchanged_tensors_skipped()
    test_changed_tensor_detected()
    test_bitwise_correctness_over_multiple_steps()
    test_sparsity_direct_control()
    test_performance_overhead()

    print("=" * 60)
    print(" ALL TESTS PASSED")
    print("=" * 60)
