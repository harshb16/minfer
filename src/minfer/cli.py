"""Streaming command-line interface; no server or external generation loop."""

import argparse
import sys

from .config import MODEL_ID, SamplingParams
from .engine import LLMEngine


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="minfer", description="Educational cached LLM inference")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="Generate and stream text")
    generate.add_argument("--model", default=MODEL_ID)
    generate.add_argument(
        "--prompt", action="append", required=True, help="Repeat to submit multiple prompts"
    )
    generate.add_argument("--device", default="auto")
    generate.add_argument(
        "--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"]
    )
    generate.add_argument("--max-active-requests", type=int, default=4)
    generate.add_argument("--max-batched-tokens", type=int)
    generate.add_argument("--max-kv-tokens", type=int)
    generate.add_argument("--max-new-tokens", type=int, default=64)
    generate.add_argument("--temperature", type=float, default=0)
    generate.add_argument("--top-k", type=int, default=0)
    generate.add_argument("--top-p", type=float, default=1)
    generate.add_argument("--seed", type=int)
    generate.add_argument(
        "--raw-prompt", action="store_true", help="Disable instruction chat template"
    )
    args = parser.parse_args(argv)
    try:
        params = SamplingParams(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed,
        )
        engine = LLMEngine(
            model=args.model,
            device=args.device,
            dtype=args.dtype,
            max_active_requests=args.max_active_requests,
            max_batched_tokens=args.max_batched_tokens,
            max_kv_tokens=args.max_kv_tokens,
            chat_template=not args.raw_prompt,
        )
        ids = [engine.add_request(prompt, params) for prompt in args.prompt]
        labels = {request_id: index + 1 for index, request_id in enumerate(ids)}
        while engine.has_unfinished_requests():
            for event in engine.step():
                if len(ids) == 1:
                    print(event.token_text, end="", flush=True)
                elif event.token_text:
                    print(f"[{labels[event.request_id]}] {event.token_text}", flush=True)
        if len(ids) == 1:
            print()
    except (ValueError, OSError) as exc:
        print(f"minfer: {exc}", file=sys.stderr)
        return 1
    return 0
