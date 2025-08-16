"""Benchmark for CPU Paged Attention in SwiftLLM."""

import csv
import dataclasses
import time
import timeit
from typing import Dict, List, Literal

import cpuinfo
import torch

from swiftllm.worker.kernels.paged_attn import paged_attention


@dataclasses.dataclass
class MockModelConfig:
    """Mock configuration for the model."""

    num_hidden_layers = 32
    num_q_heads = 32
    num_kv_heads = 32
    head_dim = 128
    hidden_size = num_q_heads * head_dim  # 4096


@dataclasses.dataclass
class MockEngineConfig:
    """Mock configuration for the engine."""

    block_size = 0  # to be set later
    max_blocks_per_seq = 0  # to be set later


@dataclasses.dataclass
class MockParameters:
    """Mock parameters for paged attention."""

    number_of_executions: int
    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    block_table: torch.Tensor
    seq_block_size: int
    num_seq_blocks: int
    softmax_scale: float
    seq_ids: torch.Tensor
    seq_lens: torch.Tensor
    o: torch.Tensor
    model_config: MockModelConfig
    engine_config: MockEngineConfig


def get_mock_parameters(
    number_of_executions=20,
    batch_size: int = 64,
    block_size: int = 16,
    seq_len: int = 2048,
    device: Literal["cpu", "cuda"] = "cuda",
):
    """Generate mock parameters for paged attention."""
    allocation_device = "cuda" if torch.cuda.is_available() else "cpu"
    allocation_dtype = torch.float16 if device == "cuda" else torch.float32

    model_config = MockModelConfig()
    engine_config = MockEngineConfig()
    engine_config.block_size = block_size
    engine_config.max_blocks_per_seq = (
        seq_len + engine_config.block_size - 1
    ) // engine_config.block_size

    num_blocks = batch_size * engine_config.max_blocks_per_seq
    q = (
        torch.randn(
            (
                batch_size,
                model_config.num_q_heads,
                model_config.head_dim,
            ),
            dtype=allocation_dtype,
            device=allocation_device,
        )
        .to(device)
        .to(torch.float16)
    )
    k_cache = (
        torch.randn(
            (
                num_blocks,
                model_config.num_kv_heads,
                engine_config.block_size,
                model_config.head_dim,
            ),
            dtype=allocation_dtype,
            device=allocation_device,
        )
        .to(device)
        .to(torch.float16)
    )
    v_cache = (
        torch.randn_like(
            k_cache, dtype=allocation_dtype, device=allocation_device
        )
        .to(device)
        .to(torch.float16)
    )
    block_table = torch.randint(
        0,
        num_blocks,
        (
            batch_size,
            engine_config.max_blocks_per_seq,
        ),
        dtype=torch.int32,
        device=device,
    )
    seq_ids = torch.arange(0, batch_size, dtype=torch.int32, device=device)
    seq_lens = torch.empty((batch_size,), dtype=torch.int32, device=device)
    seq_lens.fill_(engine_config.max_blocks_per_seq * engine_config.block_size)
    softmax_scale = 0.125
    o = (
        torch.zeros(
            (batch_size, model_config.hidden_size),
            dtype=allocation_dtype,
            device=allocation_device,
        )
        .to(device)
        .to(torch.float16)
    )
    seq_block_size = max(32, block_size)
    num_seq_blocks = (
        seq_lens.max().item() + seq_block_size - 1
    ) // seq_block_size

    return MockParameters(
        number_of_executions=number_of_executions,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=block_table,
        seq_block_size=seq_block_size,
        num_seq_blocks=num_seq_blocks,
        softmax_scale=softmax_scale,
        seq_ids=seq_ids,
        seq_lens=seq_lens,
        o=o,
        model_config=model_config,
        engine_config=engine_config,
    )


def benchmark_paged_attention(parameters: MockParameters):
    """Benchmark the paged attention operation."""

    number_of_executions = parameters.number_of_executions
    q = parameters.q
    k_cache = parameters.k_cache
    v_cache = parameters.v_cache
    block_table = parameters.block_table
    seq_block_size = parameters.seq_block_size
    num_seq_blocks = parameters.num_seq_blocks
    softmax_scale = parameters.softmax_scale
    seq_ids = parameters.seq_ids
    seq_lens = parameters.seq_lens
    o = parameters.o
    model_config = parameters.model_config
    engine_config = parameters.engine_config

    time_per_execution = 0.0
    for _ in range(number_of_executions):
        start_time = timeit.default_timer()
        paged_attention(
            q,
            k_cache,
            v_cache,
            block_table,
            seq_block_size,
            num_seq_blocks,
            softmax_scale,
            seq_ids,
            seq_lens,
            model_config,
            engine_config,
            o,
        )
        if q.device.type == "cuda":
            torch.cuda.synchronize()
        time_per_execution += timeit.default_timer() - start_time
    time_per_execution /= number_of_executions
    return time_per_execution


def main():
    """Main function to run the benchmark."""

    results: List[Dict[str, float]] = []

    device_name_map = {
        "cuda": torch.cuda.get_device_name(),
        "cpu": cpuinfo.get_cpu_info()["brand_raw"],
    }

    # Warm up the GPU
    benchmark_paged_attention(
        get_mock_parameters(
            number_of_executions=20,
            batch_size=16,
            block_size=16,
            seq_len=1024,
            device="cuda",
        )
    )
    # Warm up the CPU
    benchmark_paged_attention(
        get_mock_parameters(
            number_of_executions=20,
            batch_size=16,
            block_size=16,
            seq_len=1024,
            device="cpu",
        )
    )

    # Collect configurations to benchmark
    configurations = []
    for device in ["cuda", "cpu"]:
        for batch_size in [1, 16]:
            for block_size in [4, 16, 64]:
                for seq_len in [2**i for i in range(5, 15)]:
                    if block_size > seq_len:
                        # Skip invalid configurations
                        continue
                    configurations.append(
                        {
                            "device": device,
                            "batch_size": batch_size,
                            "block_size": block_size,
                            "seq_len": seq_len,
                        }
                    )
    # Run benchmarks
    print(
        "device, batch_size, block_size, seq_len, time_per_execution (seconds)"
    )
    for config in configurations:
        device = config["device"]
        batch_size = config["batch_size"]
        block_size = config["block_size"]
        seq_len = config["seq_len"]
        parameters = get_mock_parameters(
            number_of_executions=20,
            batch_size=batch_size,
            block_size=block_size,
            seq_len=seq_len,
            device=device,
        )
        time_per_execution = benchmark_paged_attention(parameters)
        del parameters
        print(
            f"{device_name_map[device]}, {batch_size:10d}, {block_size:10d}"
            f", {seq_len:7d}, {time_per_execution:.6f}"
        )
        results.append(
            {
                "device": device_name_map[device],
                "batch_size": batch_size,
                "block_size": block_size,
                "seq_len": seq_len,
                "time_per_execution": time_per_execution,
            }
        )
        time.sleep(1)

    # Save results to CSV
    with open(
        "paged_attn_benchmark_results.csv",
        mode="w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "device",
                "batch_size",
                "block_size",
                "seq_len",
                "time_per_execution",
            ],
        )
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
