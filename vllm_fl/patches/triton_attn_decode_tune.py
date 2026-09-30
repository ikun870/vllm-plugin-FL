# Copyright (c) 2025 BAAI. All rights reserved.
"""TRITON_ATTN 的纯 decode 路径选择 + launch 特化 —— vLLM monkey-patch。

问题
----
vLLM ``triton_attn`` 后端的 2D/3D 分界 ``seq_threshold_3D = MIN_LAUNCH_GRID_SIZE_2D(128)
// num_kv_heads``。MiniCPM5-2B 只有 2 个 KV head ⇒ 阈值 64 ⇒ 官方 64 并发的纯 decode 步
（``num_seqs <= 64``）全部走 **3D split-KV**：16 段并行 softmax + 额外一次 ``reduce_segments``。

这个阈值的本意是「2D grid 不足 128 个 CTA 时用 3D 补并行度」。但在
decode（q_len=1、BLOCK_Q=2）下，2D grid 的有效 CTA 数 = num_seqs × num_kv_heads，
n=64 时就有 128 个（>114 SM），KV 流量已足以打满 DRAM —— 3D 的 16 段 + reduce 纯属开销。

分档（只在 max_seqlen_q==1 & npk==8 & head_size==128 & 无滑窗 & 非 batch-invariant 时生效）
----
  num_seqs > 32       : 强制 2D，TILE_SIZE=64，num_warps=4，num_stages=2          （level ≥1）
  16 < num_seqs <= 32 : 保持 3D，分段 16→4，num_warps=2，num_stages=2             （level ≥2）
  num_seqs <= 16      : 不动（生产默认）
分界 32 是 cudagraph capture size ⇒ 每张图在捕获期就定死一条路径，与上游阈值对齐 capture
size 的做法一致（``triton_attn.py`` 里 ``min(capture_sizes, key=...)``）。

实测依据（scripts/e61_decode3d_sweep.py，4090D 卡1，生产原版 kernel + CUDA Graph，
8 轮交错，双基线臂噪声底 ≤0.1%；attention + reduce_segments 一起计时）
--------
  kernel 级 vs 生产 3D(SEG16/T16)：
    nseq        24     32     40     48     56     64
    4k  2D_T64  -5.8   -0.7   +5.7   +5.5   +7.5   +6.7   (%)
    4k  S4W2S2  +1.5   +2.1   +5.1   +3.0   +5.1   +3.9
    16k 2D_T64  -5.9   -0.7   +2.7   +2.0   +3.3   +2.7
    16k S4W2S2  +1.5   +1.8   +2.9   +1.7   +2.5   +1.6
  误差（对 fp32 真值）与基线同量级；与基线逐位相同 ~51%（改了 fp32 归约的分段方式）
  ⇒ 不是 bit-exact，精度须走交错评测，不能照搬 P4b 的「逐位相同」论证。

做法
----
与 P4b（``triton_attn_prefill_tune``）同一套：取 ``unified_attention`` **当前**源码
（若 P4b 已装，取的就是 P4b 版本）→ 注入一处锚点 → exec → 替换两个模块里的符号 → linecache。
锚点只有 ``TILE_SIZE_DECODE = _get_tile_size(...)`` 一处，P4b 不碰它。
锚点不匹配时安全回退原实现，并落文件留痕。

注意
----
* 必须在 P4b **之后**安装（``register_model()`` 里紧跟 P4b）。
* ⚠️ 参数来源卡：NVIDIA RTX 4090 D（AD102, 114 SM）。分界 32/16 与 SM 数强相关，
  换卡必须重扫（e61 phase M），与 P4b 相同的交付约束。
* CUDA Graph 回放不走 Python ⇒ 留痕记录的是**捕获期 / eager 期**各 num_seqs 走了哪一档。
"""
from __future__ import annotations

import inspect
import logging
import os

logger = logging.getLogger(__name__)

_TRACE_PATH = os.environ.get(
    "FLAGOS_DECODE_TUNE_TRACE", "/tmp/flagos_decode_tune_applied.log")


def _trace(level: int, note: str, desc: str = "") -> None:
    """文件留痕（worker 子进程里 logger 可能没有 handler）。level=0 也留痕。"""
    try:
        with open(_TRACE_PATH, "a", encoding="utf-8") as fh:
            fh.write(
                "[FL] decode tune %-8s pid=%d  level=%d%s\n"
                % (note, os.getpid(), level, ("  " + desc) if desc else "")
            )
    except Exception:  # pragma: no cover
        pass


_SEEN: set = set()


def _fl_dt_note(num_seqs: int, band: str) -> None:
    """每个 (num_seqs, band) 只落一行：证明捕获期各图走了哪一档。"""
    key = (int(num_seqs), band)
    if key in _SEEN:
        return
    _SEEN.add(key)
    _trace(FLAGOS_DECODE_TUNE_LEVEL, "band", "num_seqs=%d band=%s" % key)


# ---------------------------------------------------------------------------
# 源码字面量开关（A/B 时改这一行；多进程下环境变量读取点不可靠，同 P4b）
#   0 = 关闭（等价基线）   1 = num_seqs>32 强制 2D/T64/W4/S2   2 = 再 + 16<n<=32 走 3D SEG4/W2/S2
# ---------------------------------------------------------------------------
FLAGOS_DECODE_TUNE_LEVEL = 2

_ANCHOR = """    TILE_SIZE_DECODE = _get_tile_size(
        head_size, sliding_window_val, q.element_size(), is_prefill=False
    )"""

_NEW = _ANCHOR + """
    if (
        FLAGOS_DECODE_TUNE_LEVEL >= 1
        and max_seqlen_q == 1
        and num_queries_per_kv == 8
        and head_size == 128
        and sliding_window_val == 0
        and not is_batch_invariant
        and not use_td
        and seq_threshold_3D is not None
    ):
        if num_seqs > 32:
            seq_threshold_3D = 0
            TILE_SIZE_PREFILL = 64
            launch_num_warps = 4
            launch_num_stages = 2
            _fl_dt_note(num_seqs, "2D_T64_W4_S2")
        elif (
            FLAGOS_DECODE_TUNE_LEVEL >= 2
            and num_seqs > 16
            and num_par_softmax_segments is not None
            and num_par_softmax_segments >= 4
        ):
            num_par_softmax_segments = 4
            launch_num_warps = 2
            launch_num_stages = 2
            _fl_dt_note(num_seqs, "3D_S4_W2_S2")"""


def apply_triton_attn_decode_tune() -> bool:
    """安装 decode 路径特化。返回是否已生效（失败则保持原实现）。"""
    _trace(FLAGOS_DECODE_TUNE_LEVEL, "entry")
    if not FLAGOS_DECODE_TUNE_LEVEL:
        return False

    try:
        from vllm.v1.attention.backends import triton_attn as _tattn
        from vllm.v1.attention.ops import triton_unified_attention as _tua
    except Exception as exc:  # pragma: no cover
        logger.debug("[FL] decode tune: vLLM attention 模块不可用: %s", exc)
        return False

    cur = _tattn.unified_attention
    if getattr(cur, "_fl_dt_tuned", False):
        return True

    try:
        src = inspect.getsource(cur)   # P4b 已装时取到的是 P4b 版本（linecache 已注册）
    except Exception as exc:  # pragma: no cover
        logger.warning("[FL] decode tune: 取 unified_attention 源码失败: %s", exc)
        _trace(FLAGOS_DECODE_TUNE_LEVEL, "SKIP", "getsource failed")
        return False

    if src.count(_ANCHOR) != 1:
        logger.warning("[FL] decode tune: 锚点不匹配（vLLM 版本可能已变更），保持原实现")
        _trace(FLAGOS_DECODE_TUNE_LEVEL, "SKIP", "anchor hits=%d" % src.count(_ANCHOR))
        return False

    new_src = src.replace(_ANCHOR, _NEW, 1)
    src_name = "<vllm_fl.decode_tune>"
    # 以当前函数的 globals 为底：P4b 的 FLAGOS_PREFILL_TUNE_LEVEL 等注入项随之保留
    ns = dict(getattr(cur, "__globals__", _tua.__dict__))
    ns["FLAGOS_DECODE_TUNE_LEVEL"] = FLAGOS_DECODE_TUNE_LEVEL
    ns["_fl_dt_note"] = _fl_dt_note
    try:
        exec(compile(new_src, src_name, "exec"), ns)  # noqa: S102
    except Exception as exc:  # pragma: no cover
        logger.warning("[FL] decode tune: 编译特化版本失败: %s", exc)
        _trace(FLAGOS_DECODE_TUNE_LEVEL, "SKIP", "compile failed: %s" % exc)
        return False
    try:
        import linecache

        linecache.cache[src_name] = (len(new_src), None, new_src.splitlines(True), src_name)
    except Exception:  # pragma: no cover
        pass

    tuned = ns.get("unified_attention")
    if not callable(tuned):  # pragma: no cover
        logger.warning("[FL] decode tune: 未得到特化函数，跳过")
        return False

    # 继承 P4b 的幂等标记，防止 register_model 再次调用时 P4b 在本版本之上重装
    for attr in ("_fl_pf_tuned",):
        if getattr(cur, attr, False):
            setattr(tuned, attr, True)
    tuned._fl_dt_tuned = True  # type: ignore[attr-defined]
    _tattn.unified_attention = tuned
    _tua.unified_attention = tuned

    desc = {
        1: "n>32: 2D/T64/W4/S2",
        2: "n>32: 2D/T64/W4/S2 ; 16<n<=32: 3D/SEG4/W2/S2",
    }.get(FLAGOS_DECODE_TUNE_LEVEL, "?")
    logger.info("[FL] TRITON_ATTN decode 特化已启用（level=%d）：%s",
                FLAGOS_DECODE_TUNE_LEVEL, desc)
    _trace(FLAGOS_DECODE_TUNE_LEVEL, "applied",
           desc + "  on_top_of_pf=%d" % int(bool(getattr(cur, "_fl_pf_tuned", False))))
    return True


__all__ = ["apply_triton_attn_decode_tune", "FLAGOS_DECODE_TUNE_LEVEL"]
