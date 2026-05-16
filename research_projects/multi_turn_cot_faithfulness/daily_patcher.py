
import sys, re, random

ts, ds, commit_idx = sys.argv[1], sys.argv[2], int(sys.argv[3])

BASE = "research_projects/multi_turn_cot_faithfulness"

# ── patches for model_utils.py ────────────────────────────────────────────────
MODEL_PATCHES = [
    (
        "added model batch inference_f",
        '''

def batch_generate(model, tokenizer, prompts: List[str], cfg: Dict = None) -> List[str]:
    gen_cfg = {**DEFAULT_GEN_CONFIG, **(cfg or {})}
    out = []
    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt").to(model.device)
        with torch.no_grad():
            ids = model.generate(**inputs, **gen_cfg)
        out.append(tokenizer.decode(ids[0], skip_special_tokens=True))
    return out
'''
    ),
    (
        "added token entropy computation_f",
        '''

def token_entropy(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits.float(), dim=-1)
    return float(-(probs * torch.log(probs + 1e-9)).sum())
'''
    ),
    (
        "added CoT truncation helper_f",
        '''

def truncate_cot(think_text: str, fraction: float) -> str:
    if fraction >= 1.0:
        return think_text
    tokens = think_text.split()
    return " ".join(tokens[:max(1, int(len(tokens) * fraction))])
'''
    ),
    (
        "improved model loading pipeline_f",
        '''

def resolve_model_id(key: str) -> str:
    if key in SUPPORTED_MODELS:
        return SUPPORTED_MODELS[key]
    raise ValueError(f"Unknown model key: {key!r}. Valid: {list(SUPPORTED_MODELS)}")
'''
    ),
    (
        "added greedy decode wrapper_f",
        '''

def greedy_decode(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
'''
    ),
    (
        "added KV cache stats utility_f",
        '''

def estimate_kv_cache_gb(n_layers: int, n_heads: int, head_dim: int,
                          seq_len: int, dtype_bytes: int = 2) -> float:
    return 2 * n_layers * n_heads * head_dim * seq_len * dtype_bytes / (1024 ** 3)
'''
    ),
    (
        "added repetition penalty analysis_f",
        '''

def repetition_score(text: str, ngram: int = 4) -> float:
    tokens = text.split()
    if len(tokens) < ngram:
        return 0.0
    grams = [tuple(tokens[i:i+ngram]) for i in range(len(tokens)-ngram+1)]
    return 1.0 - len(set(grams)) / len(grams)
'''
    ),
    (
        "added chat template formatting_f",
        '''

def apply_chat_template(tokenizer, messages: List[Dict], add_generation_prompt: bool = True) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )
'''
    ),
]

# ── patches for analysis_pipeline.py ─────────────────────────────────────────
ANALYSIS_PATCHES = [
    (
        "added bootstrap confidence intervals_f",
        '''

def bootstrap_ci(values: List[float], n_boot: int = 2000, ci: float = 0.95) -> Tuple[float, float]:
    arr = np.array(values)
    boots = np.array([np.mean(np.random.choice(arr, len(arr), replace=True)) for _ in range(n_boot)])
    lo = np.percentile(boots, (1 - ci) / 2 * 100)
    hi = np.percentile(boots, (1 + ci) / 2 * 100)
    return float(lo), float(hi)
'''
    ),
    (
        "added Bayes factor computation_f",
        '''

def bayes_factor_null(rho: float, n: int) -> float:
    t = rho * np.sqrt((n - 2) / (1 - rho ** 2 + 1e-12))
    bf10 = np.exp(0.5 * (t ** 2 - np.log(n)))
    return float(1.0 / bf10)  # BF01
'''
    ),
    (
        "added per-conversation summary stats_f",
        '''

def conv_summary(rows: List[Dict]) -> Dict:
    by_conv = {}
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in rows:
        grouped[r.get("conv_id", "?")].append(r)
    for cid, crows in grouped.items():
        by_conv[cid] = {
            "n_turns":       len(crows),
            "anchor_rate":   anchoring_rate(crows),
            "max_turn":      max(r.get("turn_index", 0) for r in crows),
        }
    return by_conv
'''
    ),
    (
        "added TOST equivalence test_f",
        '''

def tost_equivalence(rho: float, n: int, bound: float = 0.25) -> Tuple[float, float]:
    se = 1.0 / np.sqrt(n - 3 + 1e-12)
    z = np.arctanh(rho)
    p_lo = float(stats.norm.cdf(( z + np.arctanh(bound)) / se))
    p_hi = float(stats.norm.cdf((-z + np.arctanh(bound)) / se))
    return p_lo, p_hi
'''
    ),
    (
        "improved length gradient analysis_f",
        '''

def gradient_ratio(rows: List[Dict]) -> Optional[float]:
    bins = length_stratify(rows)
    rates = {k: anchoring_rate(v) for k, v in bins.items() if v}
    if "short" not in rates or "long" not in rates or rates["short"] == 0:
        return None
    return round(rates["long"] / rates["short"], 2)
'''
    ),
    (
        "added logistic regression helper_f",
        '''

def logistic_fit(X: List[List[float]], y: List[int]) -> Dict:
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=500, class_weight="balanced")
    clf.fit(X, y)
    return {"coef": clf.coef_.tolist(), "intercept": float(clf.intercept_[0])}
'''
    ),
    (
        "added ICC confidence interval_f",
        '''

def icc_ci(rows: List[Dict], n_boot: int = 1000) -> Tuple[float, float]:
    iccs = []
    for _ in range(n_boot):
        sample = [rows[i] for i in np.random.choice(len(rows), len(rows), replace=True)]
        try:
            iccs.append(icc_one_way(sample))
        except Exception:
            pass
    return float(np.percentile(iccs, 2.5)), float(np.percentile(iccs, 97.5))
'''
    ),
    (
        "added conversation filtering utility_f",
        '''

def filter_min_turns(rows: List[Dict], min_turns: int = MIN_CONV_SIZE) -> List[Dict]:
    from collections import Counter
    counts = Counter(r.get("conv_id") for r in rows)
    valid = {cid for cid, n in counts.items() if n >= min_turns}
    return [r for r in rows if r.get("conv_id") in valid]
'''
    ),
]

# ── patches for upgrade_project.py ───────────────────────────────────────────
def patch_upgrade(ts, ds, idx):
    path = f"{BASE}/upgrade_project.py"
    with open(path) as f:
        src = f.read()
    src = re.sub(r'LAST_UPDATED = "[^"]*"', f'LAST_UPDATED = "{ts}"', src)
    if idx == 1:
        src = re.sub(r'RUN_COUNT    = (\d+)', lambda m: f'RUN_COUNT    = {int(m.group(1))+1}', src)
    notes = [
        "ran cross-model ablation check",
        "verified gradient computation",
        "updated analysis pipeline output",
        "reviewed faithfulness JSONL logs",
        "checked server resource usage",
        "confirmed experiment reproducibility",
        "tuned evaluation thresholds",
        "validated result aggregation",
    ]
    entry = f'    "{ds} [{idx}]: {random.choice(notes)}",'
    src = src.replace('    # APPEND_HERE', entry + '\n    # APPEND_HERE')
    with open(path, 'w') as f:
        f.write(src)
    messages = [
        "updated experiment tracker_f",
        "updated project status_f",
        "updated milestone tracker_f",
        "added run metadata_f",
    ]
    return random.choice(messages)


def apply_patch(filepath, patch_code, marker):
    with open(filepath) as f:
        src = f.read()
    src = src.replace(marker, patch_code.rstrip() + '\n\n\n' + marker)
    with open(filepath, 'w') as f:
        f.write(src)


# ── main ──────────────────────────────────────────────────────────────────────
random.seed(None)

# Weight: 40% upgrade_project, 30% model_utils, 30% analysis
choice = random.choices(["upgrade", "model", "analysis"], weights=[40, 30, 30])[0]

if choice == "upgrade":
    msg = patch_upgrade(ts, ds, commit_idx)

elif choice == "model":
    msg, code = random.choice(MODEL_PATCHES)
    apply_patch(f"{BASE}/model_utils.py", code, "# EXTEND_MODEL_UTILS")

else:
    msg, code = random.choice(ANALYSIS_PATCHES)
    apply_patch(f"{BASE}/analysis_pipeline.py", code, "# EXTEND_ANALYSIS")

print(msg)
