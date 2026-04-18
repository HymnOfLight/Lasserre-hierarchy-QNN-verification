#!/usr/bin/env python3
"""
Example: Verify adversarial robustness of a quantised LLM (Qwen2).

Verifies that the next-token prediction is stable under L_inf
embedding-space perturbation, using Jacobian-based bounding with
optional Lasserre hierarchy SDP refinement (SCS solver).

Models are downloaded from ModelScope (国内镜像, 默认) to ./models/ 目录.

Usage:
    # 默认使用 ModelScope 下载 Qwen2-0.5B 到 ./models/
    python verify_llm.py

    # Qwen2-7B (GPU fp16, ~16GB 显存)
    python verify_llm.py --model Qwen/Qwen2-7B --device cuda

    # Qwen2-7B int8 量化 (~8GB 显存)
    python verify_llm.py --model Qwen/Qwen2-7B --quantization int8

    # Qwen2-7B int4 量化 (~4GB 显存)
    python verify_llm.py --model Qwen/Qwen2-7B --quantization int4

    # 使用 HuggingFace 镜像 (海外)
    python verify_llm.py --mirror huggingface

    # 指定本地已下载的模型目录
    python verify_llm.py --model ./models/qwen/Qwen2-0.5B

    # 自定义 prompt 和 epsilon
    python verify_llm.py --prompt "The capital of France is" --epsilon 0.001

    # 搜索最大可认证扰动半径
    python verify_llm.py --find-radius --prompt "2 + 2 ="
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from qnn_verifier.llm import load_llm, LLMRobustnessVerifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


DEFAULT_PROMPTS = [
    "The capital of France is",
    "2 + 2 =",
    "The largest planet in the solar system is",
    "def fibonacci(n):",
    "In machine learning, overfitting means",
]

DEFAULT_EPSILONS = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]


def main():
    parser = argparse.ArgumentParser(description="LLM robustness verification")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-0.5B",
                        help="Model name or local path (e.g. Qwen/Qwen2-7B)")
    parser.add_argument("--mirror", type=str, default="modelscope",
                        choices=["modelscope", "huggingface"],
                        help="Download source: modelscope (国内默认) / huggingface")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Model cache directory (default: ./models/)")
    parser.add_argument("--quantization", type=str, default=None,
                        choices=[None, "int8", "int4", "fp16", "fp32"],
                        help="Quantisation mode")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cuda / cpu / auto")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Custom prompt to verify")
    parser.add_argument("--epsilon", type=float, default=None,
                        help="Single epsilon value")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Top-k tokens to consider")
    parser.add_argument("--find-radius", action="store_true",
                        help="Binary-search for max certifiable radius")
    parser.add_argument("--use-sdp", action="store_true",
                        help="Enable Lasserre SDP refinement")
    args = parser.parse_args()

    # ---- Load model ----
    print("=" * 64)
    print(f"  Model       : {args.model}")
    print(f"  Mirror      : {args.mirror}")
    print(f"  Cache dir   : {args.cache_dir or './models/ (default)'}")
    print(f"  Quantisation: {args.quantization or 'auto'}")
    print("=" * 64)

    llm = load_llm(
        model_name=args.model,
        quantization=args.quantization,
        device=args.device,
        mirror=args.mirror,
        cache_dir=args.cache_dir,
    )

    print(f"  Model       : {llm.model_name}")
    print(f"  Parameters  : {sum(p.numel() for p in llm.model.parameters()) / 1e9:.2f}B")
    print(f"  Hidden dim  : {llm.hidden_size}")
    print(f"  Vocab size  : {llm.vocab_size}")
    print(f"  Layers      : {llm.num_layers}")
    print(f"  Device      : {llm.device}")

    verifier = LLMRobustnessVerifier(
        llm, poly_degree=4, max_lasserre_order=2, verbose=False
    )

    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS
    epsilons = [args.epsilon] if args.epsilon else DEFAULT_EPSILONS

    # ---- Verify ----
    if args.find_radius:
        print("\n" + "=" * 64)
        print("  CERTIFIED RADIUS SEARCH")
        print("=" * 64)
        for prompt in prompts:
            radius = verifier.find_certified_radius(
                prompt, eps_min=1e-6, eps_max=0.1, n_steps=15
            )
            preds = llm.predict_next_token(prompt, top_k=3)
            print(f"  \"{prompt}\"")
            print(f"    Prediction: \"{preds[0]['token']}\"")
            print(f"    Certified radius: {radius:.6f}")
            print()
    else:
        print("\n" + "=" * 64)
        print("  MULTI-PROMPT / MULTI-EPSILON VERIFICATION")
        print("=" * 64)

        for prompt in prompts:
            print(f"\n--- Prompt: \"{prompt}\" ---")

            preds = llm.predict_next_token(prompt, top_k=5)
            print(f"  Top-1: \"{preds[0]['token']}\"  (p={preds[0]['probability']:.4f})")
            if len(preds) > 1:
                print(f"  Top-2: \"{preds[1]['token']}\"  (p={preds[1]['probability']:.4f})")
            print(f"  Nominal margin: {preds[0]['logit'] - preds[1]['logit']:.4f}")

            for eps in epsilons:
                result = verifier.verify_next_token(
                    prompt, epsilon=eps, top_k=args.top_k,
                    use_sdp_refinement=args.use_sdp,
                )
                tag = "CERTIFIED" if result.verified else "FAIL     "
                print(
                    f"    eps={eps:.1e}: [{tag}]  margin_lb={result.margin:+.6f}  "
                    f"time={result.computation_time:.2f}s"
                )

    print("\nDone.")


if __name__ == "__main__":
    main()
