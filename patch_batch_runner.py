"""Patches phase2_batch_runner.py to add --assistant_model flag."""

path = "/home/vasudev_majhi_2021/multi_turn_cot/lost_in_conversation/phase2_batch_runner.py"
with open(path) as f:
    src = f.read()

# 1. Add --assistant_model argument after --max_turns
old_arg = '''\
    ap.add_argument("--max_turns", type=int, default=None,
                    help="Hard cap on assistant turns per conversation (prevents runaway long convs).")
    ap.add_argument("--exclude_dirs", nargs="*", default=[],'''

new_arg = '''\
    ap.add_argument("--max_turns", type=int, default=None,
                    help="Hard cap on assistant turns per conversation (prevents runaway long convs).")
    ap.add_argument("--assistant_model", type=str, default="deepseek-r1-distill-qwen-7b",
                    help="Model name for the assistant under test (must be in MODEL_REGISTRY).")
    ap.add_argument("--exclude_dirs", nargs="*", default=[],'''

assert old_arg in src, "--max_turns arg block not found"
src = src.replace(old_arg, new_arg)

# 2. Use args.assistant_model when loading the assistant
old_load = '''\
    # --- Pre-load BOTH models before any conversation starts ---
    # This fails fast if VRAM is insufficient, avoids OOM mid-run.
    print("\\nLoading R1-Distill-7B...", flush=True)
    r1_model, r1_tok = _load_model(_normalize_model_name("deepseek-r1-distill-qwen-7b"))
    print("Loading Qwen2.5-7B-Instruct...", flush=True)
    _load_model(_normalize_model_name("qwen2.5-7b-instruct"))
    task_obj = get_task(args.task)
    models = {
        "assistant": "deepseek-r1-distill-qwen-7b",
        "system": "qwen2.5-7b-instruct",
        "user": "qwen2.5-7b-instruct",
        "assistant_temp": args.assistant_temp,
        "user_temp": args.user_temp,
    }'''

new_load = '''\
    # --- Pre-load BOTH models before any conversation starts ---
    # This fails fast if VRAM is insufficient, avoids OOM mid-run.
    asst_model_key = _normalize_model_name(args.assistant_model)
    print(f"\\nLoading assistant model: {asst_model_key}...", flush=True)
    r1_model, r1_tok = _load_model(asst_model_key)
    print("Loading Qwen2.5-7B-Instruct (user simulator)...", flush=True)
    _load_model(_normalize_model_name("qwen2.5-7b-instruct"))
    task_obj = get_task(args.task)
    models = {
        "assistant": asst_model_key,
        "system": "qwen2.5-7b-instruct",
        "user": "qwen2.5-7b-instruct",
        "assistant_temp": args.assistant_temp,
        "user_temp": args.user_temp,
    }'''

assert old_load in src, "Pre-load block not found"
src = src.replace(old_load, new_load)

with open(path, "w") as f:
    f.write(src)

print("phase2_batch_runner.py patched successfully.")
print("Changes: --assistant_model flag added; model loading uses args.assistant_model.")
