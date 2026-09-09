#!/usr/bin/env python3
"""CLI for mgflow inverse design generator.

Usage:
    python -m mgflow.generator.run --target BMG --elements Zr,Cu,Al,Ni --generations 200
    python -m mgflow.generator.run --target Tg_high --elements Zr,Ti,Cu,Ni,Al,Nb,Ag,Hf --generations 500 --batch-size 200 --save-top 100
"""

import argparse
import sys

from .config import ALL_TARGETS, CANDIDATE_ELEMENT_SETS, DEFAULT_ELEMENTS, LARGE_ELEMENT_POOL
from .loop import run_inverse_design


def main():
    parser = argparse.ArgumentParser(
        description="MGflow inverse design: generate alloy compositions with target properties.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --target BMG --elements Zr,Cu,Al,Ni
  %(prog)s --target supercooled_range --elementset ZrCuAlNiTiNbCoAg --generations 1000
  %(prog)s --target Tg_low --elements Zr,Ti,Cu,Ni --batch-size 50
  %(prog)s --mode auto --target BMG
  %(prog)s --mode auto --target Tl_high --elements Zr,Cu,Ni,Nb,Ta,Hf
  %(prog)s --target TgTx_high --elements Zr,Cu,Al,Ni      # 多目标
  %(prog)s --target strong_glass --elementset ZrCuAlNiTi  # 多目标
        """,
    )

    parser.add_argument(
        "--mode", type=str, default=None,
        choices=["fixed", "auto"],
        help="'fixed' — manual element selection (default); "
             "'auto' — large 30+ element pool, generator auto-selects.",
    )
    parser.add_argument(
        "--target", type=str, default="BMG",
        choices=list(ALL_TARGETS.keys()),
        help=f"Target property (单目标/多目标). Options: {list(ALL_TARGETS.keys())}. Default: BMG",
    )
    parser.add_argument(
        "--elements", type=str, default=None,
        help="Comma-separated element symbols, e.g. 'Zr,Cu,Al,Ni'. "
             "Overrides --elementset. In 'auto' mode, restricts the pool.",
    )
    parser.add_argument(
        "--elementset", type=str, default=None,
        choices=list(CANDIDATE_ELEMENT_SETS.keys()),
        help=f"Named element set for 'fixed' mode. Options: {list(CANDIDATE_ELEMENT_SETS.keys())}. "
             f"Default uses the 8-element ZrTiCuNiAlNbAgHf set.",
    )
    parser.add_argument(
        "--generations", "-g", type=int, default=None,
        help=f"Number of generations. Default: {500}",
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=None,
        help=f"Candidates per generation. Default: {100}",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate for generator. Default: 1e-4",
    )
    parser.add_argument(
        "--keep-top", type=int, default=None,
        help=f"Top-k candidates retained per generation. Default: {10}",
    )
    parser.add_argument(
        "--save-top", type=int, default=None,
        help=f"Save top N across all generations. Default: {50}",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.01,
        help="Minimum fraction (0-1) to keep an element. Default: 0.01",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output directory. Default: mgflow/generator/output/",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress per-generation progress.",
    )

    args = parser.parse_args()

    # Resolve mode & element set
    mode = args.mode if args.mode is not None else "fixed"

    if mode == "auto":
        if args.elements is not None:
            elements = [s.strip() for s in args.elements.split(",")]
        else:
            elements = None  # loop.py will use LARGE_ELEMENT_POOL
    else:  # fixed
        if args.elements is not None:
            elements = [s.strip() for s in args.elements.split(",")]
        elif args.elementset is not None:
            elements = list(CANDIDATE_ELEMENT_SETS[args.elementset])
        else:
            elements = list(DEFAULT_ELEMENTS)

    # Resolve numeric defaults
    from .config import (
        DEFAULT_N_GENERATIONS, DEFAULT_BATCH_SIZE,
        DEFAULT_KEEP_TOP_K, DEFAULT_SAVE_TOP_N,
    )

    n_generations = args.generations if args.generations is not None else DEFAULT_N_GENERATIONS
    batch_size = args.batch_size if args.batch_size is not None else DEFAULT_BATCH_SIZE
    keep_top_k = args.keep_top if args.keep_top is not None else DEFAULT_KEEP_TOP_K
    save_top_n = args.save_top if args.save_top is not None else DEFAULT_SAVE_TOP_N

    # Print config
    print("=" * 60)
    print("MGflow Inverse Design")
    print("=" * 60)
    print(f"  Mode:            {mode}")
    print(f"  Target:          {args.target}")
    print(f"  Elements:        {elements if elements else 'large pool (auto-select)'}")
    if elements:
        print(f"  Elements count:  {len(elements)}")
    print(f"  Generations:     {n_generations}")
    print(f"  Batch size:      {batch_size}")
    print(f"  Learning rate:   {args.lr}")
    print(f"  Keep top-K/gen:  {keep_top_k}")
    print(f"  Save top-N:      {save_top_n}")
    print(f"  Threshold:       {args.threshold * 100:.0f}%")
    print(f"  Output:          {args.output or '(default)'}")
    print("-" * 60)

    # Run
    result = run_inverse_design(
        target=args.target,
        elements=elements,
        mode=mode,
        n_generations=n_generations,
        batch_size=batch_size,
        lr=args.lr,
        keep_top_k=keep_top_k,
        threshold=args.threshold,
        save_top_n=save_top_n,
        verbose=not args.quiet,
        output_dir=args.output,
    )

    # Summary
    best = result["best"]
    if best:
        print("\n" + "=" * 60)
        print("BEST CANDIDATE")
        print("=" * 60)
        comp_str = best["composition_str"]
        props = best.get("properties", {})
        print(f"  Composition:  {comp_str}")
        for k, v in props.items():
            print(f"  {k}:           {v}")
        print(f"  Fitness:      {best['fitness']:.4f}")
        print(f"  Generation:   {best['generation']}")
    else:
        print("\nNo valid candidates found.")

    return result


if __name__ == "__main__":
    main()
