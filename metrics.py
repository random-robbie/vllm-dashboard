"""Metrics sidecar for the vLLM/SGLang dashboard (static/index.html -> /vllm-api/all).

Self-contained, stdlib-only:
  * scrapes SGLang's Prometheus endpoint (http://127.0.0.1:8000/metrics, needs
    `--enable-metrics` on the server) and maps it to the dashboard's `stats` shape;
  * a background poller records cumulative counters into SQLite so we can derive
    today / yesterday / week / month / year token+cost history;
  * best-effort OpenRouter pricing for the cost-comparison panel (cached 1h).

SGLang metric names vary across releases, so every field tries a list of candidate
names and falls back to 0 — missing metrics degrade gracefully rather than crash.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import threading
import time
import urllib.request
from datetime import datetime, timedelta

SGLANG_METRICS_URL = os.environ.get("SGLANG_METRICS_URL", "http://127.0.0.1:8000/metrics")
DB_PATH = os.environ.get("METRICS_DB", "./metrics.db")
POLL_INTERVAL = int(os.environ.get("METRICS_POLL_INTERVAL", "30"))
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-chat")
VLLM_CONTAINER = os.environ.get("VLLM_CONTAINER", "vllm")

# poller health, read by the /all endpoint for the "sidecar" block
_state = {"last_poll_ok": False, "last_poll_ts": None}
_or_cache = {"ts": 0, "data": None, "error": "not fetched yet"}
_gpu_state: dict = {"gpus": None, "started_at": None}


# ── prometheus scrape ───────────────────────────────────────────────────────

def _http_get(url: str, timeout: float = 5.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "landing-metrics/1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def parse_prom(text: str) -> dict[str, float]:
    """Flatten Prometheus text -> {metric_name: sum across all label sets}."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        key, val = parts
        try:
            v = float(val)
        except ValueError:
            continue
        name = key.split("{", 1)[0]
        out[name] = out.get(name, 0.0) + v
    return out


def _pick(m: dict[str, float], *names: str, default=0.0):
    for n in names:
        if n in m:
            return m[n]
    return default


_LABEL_RE = re.compile(r'(\w+)="([^"]*)"')


def parse_prom_labeled(text: str, metric: str, label: str) -> dict[str, float]:
    """Sum a Prometheus metric's samples grouped by one label's value (e.g.
    finished_reason=stop/length/abort/error). parse_prom() collapses labels
    away, so metrics that are only meaningful per-label need this instead."""
    out: dict[str, float] = {}
    prefix = metric + "{"
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith(prefix) or line.startswith("#"):
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        key, val = parts
        try:
            v = float(val)
        except ValueError:
            continue
        labels = dict(_LABEL_RE.findall(key[len(metric):]))
        lv = labels.get(label, "?")
        out[lv] = out.get(lv, 0.0) + v
    return out


def _fetch_raw_text() -> str:
    return _http_get(SGLANG_METRICS_URL)


def scrape_stats(text: str | None = None) -> dict:
    """Live snapshot in the shape vllm-dashboard.html expects, or raise on failure."""
    m = parse_prom(text if text is not None else _fetch_raw_text())

    prompt_total = _pick(m, "sglang:prompt_tokens_total", "sglang:prompt_tokens",
                         "vllm:prompt_tokens_total")
    gen_total = _pick(m, "sglang:generation_tokens_total", "sglang:generation_tokens",
                      "vllm:generation_tokens_total")
    cached = _pick(m, "sglang:cached_tokens_total", "sglang:cached_tokens",
                   "sglang:prefix_cache_hit_tokens_total",
                   "vllm:prompt_tokens_cached_total")
    running = _pick(m, "sglang:num_running_reqs", "vllm:num_requests_running")
    waiting = _pick(m, "sglang:num_queue_reqs", "sglang:num_waiting_reqs",
                    "vllm:num_requests_waiting")
    kv = _pick(m, "sglang:full_token_usage", "sglang:token_usage",
               "vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")
    reqs = _pick(m, "sglang:num_requests_total", "sglang:request_success_total",
                 "sglang:e2e_request_latency_seconds_count", "vllm:request_success_total")
    # prefix-cache query/hit counters (vLLM exposes these directly)
    pc_queries = _pick(m, "vllm:prefix_cache_queries_total", default=None)
    pc_hits = _pick(m, "vllm:prefix_cache_hits_total", default=None)
    chr_raw = _pick(m, "sglang:cache_hit_rate", default=None)

    prompt_computed = max(0.0, prompt_total - cached)
    if chr_raw is not None:
        cache_hit_rate = chr_raw * 100.0 if chr_raw <= 1.0 else chr_raw
    elif pc_queries:
        cache_hit_rate = pc_hits / pc_queries * 100.0
    else:
        cache_hit_rate = (cached / prompt_total * 100.0) if prompt_total else 0.0
    kv_pct = kv * 100.0 if kv <= 1.0 else kv

    return {
        "prompt_tokens": int(prompt_total),
        "generation_tokens": int(gen_total),
        "total_tokens": int(prompt_total + gen_total),
        "prompt_tokens_computed": int(prompt_computed),
        "prompt_tokens_cached": int(cached),
        "prefix_cache_queries": int(pc_queries if pc_queries is not None else prompt_total),
        "prefix_cache_hits": int(pc_hits if pc_hits is not None else cached),
        "cache_hit_rate": round(cache_hit_rate, 2),
        "total_requests": int(reqs),
        "requests_running": int(running),
        "requests_waiting": int(waiting),
        "kv_cache_usage_pct": round(kv_pct, 2),
        "timestamp": int(time.time()),
    }


# ── latency / health metrics (windowed since previous poll) ────────────────

def scrape_perf_raw(text: str) -> dict:
    """Cumulative latency-histogram sum/count + counters, for windowing between polls."""
    m = parse_prom(text)
    return {
        "ttft_sum": _pick(m, "vllm:time_to_first_token_seconds_sum"),
        "ttft_count": _pick(m, "vllm:time_to_first_token_seconds_count"),
        "itl_sum": _pick(m, "vllm:inter_token_latency_seconds_sum"),
        "itl_count": _pick(m, "vllm:inter_token_latency_seconds_count"),
        "e2e_sum": _pick(m, "vllm:e2e_request_latency_seconds_sum"),
        "e2e_count": _pick(m, "vllm:e2e_request_latency_seconds_count"),
        "queue_sum": _pick(m, "vllm:request_queue_time_seconds_sum"),
        "queue_count": _pick(m, "vllm:request_queue_time_seconds_count"),
        "preemptions_total": _pick(m, "vllm:num_preemptions_total"),
        "finish_reasons": parse_prom_labeled(text, "vllm:request_success_total", "finished_reason"),
        "waiting_reasons": parse_prom_labeled(text, "vllm:num_requests_waiting_by_reason", "reason"),
    }


def _windowed_perf(cur: dict, prev: dict | None) -> dict:
    """Derive per-interval averages from two cumulative snapshots (restart-safe)."""
    def d(key: str) -> float:
        return max(0.0, cur[key] - (prev[key] if prev else cur[key]))

    d_ttft_sum, d_ttft_cnt = d("ttft_sum"), d("ttft_count")
    d_itl_sum, d_itl_cnt = d("itl_sum"), d("itl_count")
    d_e2e_sum, d_e2e_cnt = d("e2e_sum"), d("e2e_count")
    d_q_sum, d_q_cnt = d("queue_sum"), d("queue_count")

    prev_finish = prev["finish_reasons"] if prev else {}
    finish_delta = {
        k: int(max(0.0, v - prev_finish.get(k, v)))
        for k, v in cur["finish_reasons"].items()
    }
    ok = finish_delta.get("stop", 0)
    not_ok = sum(v for k, v in finish_delta.items() if k != "stop")
    finish_total_delta = ok + not_ok

    return {
        "ttft_avg_ms": round(d_ttft_sum / d_ttft_cnt * 1000, 0) if d_ttft_cnt else None,
        "itl_avg_ms": round(d_itl_sum / d_itl_cnt * 1000, 1) if d_itl_cnt else None,
        "decode_tok_per_s_per_req": round(1.0 / (d_itl_sum / d_itl_cnt), 1) if d_itl_cnt and d_itl_sum else None,
        "e2e_avg_s": round(d_e2e_sum / d_e2e_cnt, 1) if d_e2e_cnt else None,
        "queue_avg_ms": round(d_q_sum / d_q_cnt * 1000, 0) if d_q_cnt else None,
        "preemptions_total": int(cur["preemptions_total"]),
        "preemptions_delta": int(d("preemptions_total")),
        "finish_reasons_total": {k: int(v) for k, v in cur["finish_reasons"].items()},
        "finish_reasons_delta": finish_delta,
        "success_rate_pct": round(ok / finish_total_delta * 100, 1) if finish_total_delta else None,
        "waiting_reasons": {k: int(v) for k, v in cur["waiting_reasons"].items()},
    }


_perf_state: dict = {"prev_raw": None, "windowed": None}


# ── GPU hardware + container uptime (best effort — not vLLM-reported data) ──

def scrape_gpus() -> list[dict]:
    """nvidia-smi snapshot per GPU. Returns [] if unavailable rather than raising —
    this box also runs ComfyUI/Applio/sglang sharing the same GPUs (see memory:
    only one of vLLM/SGLang uses port 8000, but ComfyUI/Applio can run alongside),
    so utilization/memory here can be driven by more than just this vLLM process."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,"
             "temperature.gpu,power.draw,power.limit",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return []
        gpus = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 8:
                continue
            idx, name, util, mem_used, mem_total, temp, power, power_limit = parts
            try:
                mem_used_f, mem_total_f = float(mem_used), float(mem_total)
                gpus.append({
                    "index": int(idx),
                    "name": name,
                    "utilization_pct": float(util),
                    "memory_used_mb": mem_used_f,
                    "memory_total_mb": mem_total_f,
                    "memory_pct": round(mem_used_f / mem_total_f * 100, 1) if mem_total_f else None,
                    "temperature_c": float(temp),
                    "power_draw_w": float(power),
                    "power_limit_w": float(power_limit),
                })
            except ValueError:
                continue
        return gpus
    except Exception:  # noqa: BLE001 - nvidia-smi may be missing/hung, degrade gracefully
        return []


def get_container_started_at(container: str = VLLM_CONTAINER) -> str | None:
    try:
        out = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.StartedAt}}", container],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        pass
    return None


# ── sqlite history ──────────────────────────────────────────────────────────

# perf columns added post-launch: no DEFAULT, so pre-migration rows read back
# as NULL rather than 0. _window_delta() treats a NULL endpoint as "unknown"
# and contributes 0 for that one row-pair instead of computing a delta against
# it — the single row where a column jumps from NULL to vLLM's real (huge,
# lifetime-cumulative) value would otherwise read as a one-time spike of the
# entire lifetime counter into whatever window contains that row. cached_total
# predates this convention and used a DEFAULT-0 backfill instead, which is why
# its delta is separately clamped to the prompt-token delta below.
_PERF_COLUMNS = (
    "ttft_sum", "ttft_count", "itl_sum", "itl_count",
    "queue_sum", "queue_count", "e2e_sum", "e2e_count",
    "preemptions_total", "finish_stop_total", "finish_notstop_total",
)


def _db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.row_factory = sqlite3.Row
    con.execute(
        "CREATE TABLE IF NOT EXISTS snapshots ("
        "ts INTEGER PRIMARY KEY, prompt_total INTEGER, gen_total INTEGER, reqs_total INTEGER)"
    )
    cols = {row[1] for row in con.execute("PRAGMA table_info(snapshots)")}
    if "cached_total" not in cols:
        # cache-token counter, added so historical cost can price cached vs
        # computed prompt tokens separately instead of charging full rate on
        # everything; rows before this migration default to 0 and self-correct
        # once the window they fall in is entirely post-migration.
        con.execute("ALTER TABLE snapshots ADD COLUMN cached_total INTEGER DEFAULT 0")
    for col in _PERF_COLUMNS:
        if col not in cols:
            con.execute(f"ALTER TABLE snapshots ADD COLUMN {col} REAL")
    return con


def _record(stats: dict, perf: dict) -> None:
    finish = perf["finish_reasons"]
    finish_stop = finish.get("stop", 0)
    finish_notstop = sum(v for k, v in finish.items() if k != "stop")
    con = _db()
    try:
        con.execute(
            "INSERT OR REPLACE INTO snapshots "
            "(ts, prompt_total, gen_total, reqs_total, cached_total, "
            " ttft_sum, ttft_count, itl_sum, itl_count, "
            " queue_sum, queue_count, e2e_sum, e2e_count, "
            " preemptions_total, finish_stop_total, finish_notstop_total) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (stats["timestamp"], stats["prompt_tokens"],
             stats["generation_tokens"], stats["total_requests"],
             stats["prompt_tokens_cached"],
             perf["ttft_sum"], perf["ttft_count"], perf["itl_sum"], perf["itl_count"],
             perf["queue_sum"], perf["queue_count"], perf["e2e_sum"], perf["e2e_count"],
             perf["preemptions_total"], finish_stop, finish_notstop),
        )
        con.commit()
    finally:
        con.close()


def _period_bounds() -> dict[str, tuple[int, int]]:
    now = datetime.now()
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    yest = midnight - timedelta(days=1)
    return {
        "today":     (int(midnight.timestamp()), int(now.timestamp())),
        "yesterday": (int(yest.timestamp()), int(midnight.timestamp())),
        "week":      (int((now - timedelta(days=7)).timestamp()), int(now.timestamp())),
        "month":     (int(now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).timestamp()), int(now.timestamp())),
        "year":      (int(now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0).timestamp()), int(now.timestamp())),
        # start=0: no row precedes it, so the window's first row has no
        # baseline and contributes nothing (same restart-safe behavior
        # _window_delta already gives every other period) — covers every
        # row ever recorded without needing to know the earliest ts.
        "all_time":  (0, int(now.timestamp())),
    }


def _window_delta(rows: list, baseline) -> dict:
    """Sum positive deltas of cumulative counters across a window (restart-safe).
    Rows are sqlite3.Row from a SELECT including all _PERF_COLUMNS + cached_total."""
    d = {"prompt": 0, "gen": 0, "reqs": 0, "cached": 0}
    for col in _PERF_COLUMNS:
        d[col] = 0.0
    prev = baseline
    for r in rows:
        if prev is not None:
            p_delta = max(0, r["prompt_total"] - prev["prompt_total"])
            d["prompt"] += p_delta
            d["gen"] += max(0, r["gen_total"] - prev["gen_total"])
            d["reqs"] += max(0, r["reqs_total"] - prev["reqs_total"])
            # clamp to p_delta: cached tokens can't exceed the prompt-token
            # delta for the same interval. Without this, the single row
            # where cached_total jumps from the migration's 0-backfill to
            # vLLM's real (huge) lifetime counter reads as a one-time spike
            # of the entire lifetime cache count into whatever window it
            # falls in — this bounds that to a harmless one-row overcount.
            d["cached"] += min(max(0, (r["cached_total"] or 0) - (prev["cached_total"] or 0)), p_delta)
            for col in _PERF_COLUMNS:
                rv, pv = r[col], prev[col]
                if rv is not None and pv is not None:
                    d[col] += max(0.0, rv - pv)
        prev = r
    return d


def _perf_from_deltas(d: dict) -> dict:
    """Derive display-ready averages from summed histogram sum/count deltas."""
    ok = d["finish_stop_total"]
    total_finished = ok + d["finish_notstop_total"]
    return {
        "ttft_avg_ms": round(d["ttft_sum"] / d["ttft_count"] * 1000, 0) if d["ttft_count"] else None,
        "itl_avg_ms": round(d["itl_sum"] / d["itl_count"] * 1000, 1) if d["itl_count"] else None,
        "decode_tok_per_s_per_req": round(1.0 / (d["itl_sum"] / d["itl_count"]), 1)
            if d["itl_count"] and d["itl_sum"] else None,
        "queue_avg_ms": round(d["queue_sum"] / d["queue_count"] * 1000, 0) if d["queue_count"] else None,
        "e2e_avg_s": round(d["e2e_sum"] / d["e2e_count"], 1) if d["e2e_count"] else None,
        "preemptions": int(d["preemptions_total"]),
        "success_rate_pct": round(ok / total_finished * 100, 1) if total_finished else None,
    }


def _period_cost(pricing: dict | None, prompt: int, cached: int, gen: int) -> float | None:
    """Price cached prompt tokens at the cache-read rate and the rest at full
    prompt rate, matching how the live (this-session) cost panel already
    prices it — the period/day rollups used to charge full prompt rate on
    cached tokens too, overstating cost by the cache discount."""
    if not pricing:
        return None
    computed = max(0, prompt - cached)
    cache_rate = pricing.get("cache_read_per_token", pricing["prompt_per_token"])
    return (computed * pricing["prompt_per_token"]
            + cached * cache_rate
            + gen * pricing["completion_per_token"])


_SNAPSHOT_COLS = "ts,prompt_total,gen_total,reqs_total,cached_total," + ",".join(_PERF_COLUMNS)


def get_history(pricing: dict | None) -> dict:
    con = _db()
    try:
        out = {}
        for key, (start, end) in _period_bounds().items():
            rows = con.execute(
                f"SELECT {_SNAPSHOT_COLS} FROM snapshots "
                "WHERE ts>=? AND ts<=? ORDER BY ts", (start, end),
            ).fetchall()
            base = con.execute(
                f"SELECT {_SNAPSHOT_COLS} FROM snapshots "
                "WHERE ts<? ORDER BY ts DESC LIMIT 1", (start,),
            ).fetchone()
            d = _window_delta(rows, base)
            out[key] = {
                "total_tokens": d["prompt"] + d["gen"],
                "prompt_tokens": d["prompt"],
                "prompt_tokens_cached": d["cached"],
                "gen_tokens": d["gen"],
                "requests": d["reqs"],
                "cost_usd": _period_cost(pricing, d["prompt"], d["cached"], d["gen"]),
                "perf": _perf_from_deltas(d),
            }
        return out
    finally:
        con.close()


def get_daily(pricing: dict | None, days: int | None = 14) -> list[dict]:
    """Per-day token + cost deltas for the last `days` days (oldest first).
    `days=None` returns every day since the earliest recorded snapshot."""
    con = _db()
    try:
        now = datetime.now()
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        if days is None:
            first = con.execute("SELECT MIN(ts) AS t FROM snapshots").fetchone()
            if first is None or first["t"] is None:
                return []
            first_midnight = datetime.fromtimestamp(first["t"]).replace(
                hour=0, minute=0, second=0, microsecond=0)
            days = (midnight - first_midnight).days + 1
        out = []
        for i in range(days - 1, -1, -1):
            start_dt = midnight - timedelta(days=i)
            end_dt = start_dt + timedelta(days=1)
            start, end = int(start_dt.timestamp()), int(end_dt.timestamp())
            rows = con.execute(
                f"SELECT {_SNAPSHOT_COLS} FROM snapshots "
                "WHERE ts>=? AND ts<? ORDER BY ts", (start, end),
            ).fetchall()
            base = con.execute(
                f"SELECT {_SNAPSHOT_COLS} FROM snapshots "
                "WHERE ts<? ORDER BY ts DESC LIMIT 1", (start,),
            ).fetchone()
            d = _window_delta(rows, base)
            out.append({
                "date": start_dt.strftime("%Y-%m-%d"),
                "label": start_dt.strftime("%a"),
                "prompt_tokens": d["prompt"],
                "prompt_tokens_cached": d["cached"],
                "gen_tokens": d["gen"],
                "total_tokens": d["prompt"] + d["gen"],
                "requests": d["reqs"],
                "cost_usd": _period_cost(pricing, d["prompt"], d["cached"], d["gen"]),
                "perf": _perf_from_deltas(d),
            })
        return out
    finally:
        con.close()


# ── openrouter pricing (best effort, cached 1h) ─────────────────────────────

def get_openrouter() -> tuple[dict | None, str | None]:
    now = time.time()
    if now - _or_cache["ts"] < 3600 and _or_cache["data"] is not None:
        return _or_cache["data"], None
    if now - _or_cache["ts"] < 300 and _or_cache["data"] is None and _or_cache["error"]:
        return None, _or_cache["error"]
    try:
        raw = _http_get("https://openrouter.ai/api/v1/models", timeout=6)
        models = json.loads(raw).get("data", [])
        hit = next((x for x in models if x.get("id") == OPENROUTER_MODEL), None)
        if hit is None:
            raise ValueError(f"model '{OPENROUTER_MODEL}' not on OpenRouter")
        pr = hit.get("pricing", {})
        data = {
            "prompt_per_token": float(pr.get("prompt", 0) or 0),
            "completion_per_token": float(pr.get("completion", 0) or 0),
            "cache_read_per_token": float(pr.get("input_cache_read", pr.get("prompt", 0)) or 0),
            "context_length": int(hit.get("context_length", 0) or 0),
            "model_id": OPENROUTER_MODEL,
        }
        _or_cache.update(ts=now, data=data, error=None)
        return data, None
    except Exception as e:  # noqa: BLE001 - best effort, surface as a soft error
        _or_cache.update(ts=now, data=None, error=f"openrouter: {e}")
        return None, _or_cache["error"]


# ── public aggregate + poller ───────────────────────────────────────────────

def get_all() -> dict:
    pricing, or_err = get_openrouter()
    stats, stats_err = None, None
    try:
        stats = scrape_stats()
    except Exception as e:  # noqa: BLE001
        stats_err = f"sglang metrics unavailable: {e}"
    lag = (int(time.time()) - _state["last_poll_ts"]) if _state["last_poll_ts"] else None
    return {
        "stats": stats,
        "stats_error": stats_err,
        "perf": _perf_state["windowed"],
        "openrouter": pricing,
        "openrouter_error": or_err,
        "history": get_history(pricing),
        "daily": get_daily(pricing),
        "daily_all": get_daily(pricing, days=None),
        "gpus": _gpu_state["gpus"],
        "vllm_started_at": _gpu_state["started_at"],
        "sidecar": {
            "last_poll_ok": _state["last_poll_ok"],
            "last_poll_ts": _state["last_poll_ts"],
            "lag_seconds": lag,
        },
    }


def _poller() -> None:
    while True:
        try:
            text = _fetch_raw_text()
            s = scrape_stats(text)
            perf_cur = scrape_perf_raw(text)
            _record(s, perf_cur)
            _perf_state["windowed"] = _windowed_perf(perf_cur, _perf_state["prev_raw"])
            _perf_state["prev_raw"] = perf_cur
            _state.update(last_poll_ok=True, last_poll_ts=int(time.time()))
        except Exception:  # noqa: BLE001 - server may just be down/loading
            _state.update(last_poll_ok=False, last_poll_ts=int(time.time()))
        # independent of vLLM's own metrics endpoint: GPU/container state is
        # often most useful precisely when vLLM itself is unresponsive.
        try:
            _gpu_state["gpus"] = scrape_gpus()
            _gpu_state["started_at"] = get_container_started_at()
        except Exception:  # noqa: BLE001
            pass
        time.sleep(POLL_INTERVAL)


_started = False


def start_poller() -> None:
    global _started
    if _started:
        return
    _started = True
    _db().close()  # ensure schema exists
    threading.Thread(target=_poller, name="metrics-poller", daemon=True).start()
