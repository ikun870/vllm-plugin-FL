# Copyright (c) 2025 BAAI. All rights reserved.
"""TRITON_ATTN 的 prefill launch 特化 —— vLLM monkey-patch。

问题
----
MiniCPM5-2B 是 GQA（16 Q heads / 2 KV heads）+ head_size=128，prefill 时
``num_queries_per_kv = 8``，于是 vLLM ``v1/attention/ops/triton_unified_attention.py``
里 ``unified_attention`` 取默认

    BLOCK_M    = 16   (→ BLOCK_Q = BLOCK_M // 8 = 2)
    TILE_SIZE  = 32
    num_warps  = None （Triton 默认）
    num_stages = None （Triton 默认 = 3）

这是 **decode 导向**的保守配置。对 prefill 形状而言 M 维只有 16 行，K/V tile
被 Q 行「摊薄」的倍数不足，且 3 级软流水把 shared memory 占满、压低驻留 CTA 数。
实测该 kernel 在 4k 官方负载的混合步上只有 ~40% MFU。

两级特化
--------
  level 1：BLOCK_M 16→32（BLOCK_Q 2→4）+ num_warps=4
  level 2：在 level 1 之上再 num_stages=1（关掉软流水 → shared memory 让位给驻留 CTA）

实测依据
--------
（a）真机混合步张量离线扫参（scripts/attn_triton_launch_sweep.py /
     attn_triton_refine.py）。形状：n_seq=64（62 decode + 2 prefill chunk）/
     total_q=2048 / max_q=1842 / max_seqlen_k=4222 / block_size=16 /
     真机 grid=(1088, 2) / use_3d=False；KV cache 268.9 MB（> L2 72 MB，避免 L2 假象）

     实测（ms；S 行之上的两行来自同一次扫参运行，下面 8 行是同进程内 3 轮重复测的 min）：
        BM16 /T32/S- (真机默认)  0.8952   1.000x   ← 3 轮重复，抖动 0.5%
        BM32 /T32/W4/S-          0.6743   1.327x   ← level 1（同一次扫参运行内测得）
        BM32 /T32/W4/S1          0.6309   1.419x   ← level 2，3 轮 min
        BM32 /T32/W4/S2          0.6284   1.424x   3 轮 min（与 S1 基本同分）
        BM32 /T32/W8/S1          1.0485   0.854x   （回落）
        BM64 /T32/W4/S1          0.6863   1.304x   （回落：grid 1088→320）
        BM32 /T128/W4/S1         0.9947   0.900x   （回落：TILE 过大）
     全部配置数值 allclose 一致；同进程 3 轮重复测量抖动 ≤0.8%。
     ⚠️ 跨运行同一配置能漂 ~3% ⇒ 上面「S1 vs S-」的 6.8% 需要在端到端 A/B 里再确认一次。

（b）整步耗时拆解（scripts/attn_triton_parts_roofline.py）：
       只 prefill 2 条 ：0.827 → 0.513 ms   ← 特化主要作用在这里
       只 decode 62 条 ：0.280 → 0.283 ms   ← 几乎不变（且 0.28 ms ≈ DRAM roofline 0.275 ms）
       所以纯 decode 步（max_seqlen_q==1，走 3D split-KV）完全不受影响。

（c）端到端（官方 b4k，源码字面量开关 + 起服后回读留痕）
        T  （level 0）: 10,220.7 tok/s   TTFT 2,153.6 ms   Median ITL 20.84  P99 ITL 110.5
        P4 （level 1）: 10,444.7 tok/s   TTFT 2,044.0 ms   Median ITL 20.84  P99 ITL 103.1
     ⇒ 吞吐 +2.19%、TTFT −5.1%、P99 ITL −6.7%；Median ITL 不变，证明只影响 prefill。

做法
----
不改 vLLM 源码：取 ``unified_attention`` 的源码，注入两处锚点，``exec`` 成新函数后
替换 ``triton_attn`` / ``triton_unified_attention`` 模块里的同名符号，并用 linecache
注册虚拟源文件（exec 出来的函数默认没有真实源文件，`inspect.getsource` 会抛 OSError）。
锚点不匹配（vLLM 升级）时安全回退到原实现，只打警告，不影响正确性。

注意
----
* 只作用于 ``max_seqlen_q > 1``（prefill / 混合步）；纯 decode 步不受影响。
* launch 参数不改变数值/算法，精度与基线一致（离线全部配置 allclose）。
* 本模块必须在 attention backend 被使用前安装；由 ``vllm_fl.register_model()``
  在 driver 与 spawn 出来的 worker 进程里分别调用（幂等）。
* ⚠️⚠️ **参数来源卡：NVIDIA RTX 4090 D（AD102, 114 SM, 48 GiB）**。
  ``BLOCK_M`` / ``num_warps`` / ``num_stages`` 的最优值是**卡相关**的（上游 vLLM 的
  ``tuned_large_head`` 就用 ``is_device_capability_family(100)`` 把作用域钉在特定架构上）。
  ⇒ **交付前必须在每一张目标卡（天数 / 沐曦）上各跑一遍三臂 A/B**，再用
  ``FLAGOS_PREFILL_TUNE_LEVEL`` 决定常驻哪一级；若两卡结论不一致，按设备名/SM 数条件化。
  这就是把它做成整数级数开关而不是 bool 的原因：现场一行即可降级。
"""
from __future__ import annotations

import inspect
import logging
import os

logger = logging.getLogger(__name__)

# 生效留痕：logger 在 vLLM 的 worker 子进程里可能没有 handler（踩过这个坑），
# 所以额外落一行文件，便于"起服务后回读留痕"。可用环境变量覆盖路径。
_TRACE_PATH = os.environ.get(
    "FLAGOS_PREFILL_TUNE_TRACE", "/tmp/flagos_prefill_tune_applied.log")


def _trace(level: int, note: str, desc: str = "") -> None:
    """写一行文件留痕（logger 在 worker 子进程可能没有 handler，必须落文件）。

    ⚠️ **level=0 也必须留痕**：对照臂拿不到留痕 ⇒ 「关」这个状态没有进程级证据 ⇒
    无法区分「开关关了」和「补丁根本没跑」。留痕格式统一，便于 grep 某个 pid。
    """
    try:
        with open(_TRACE_PATH, "a", encoding="utf-8") as fh:
            fh.write(
                "[FL] prefill tune %-8s pid=%d  level=%d%s\n"
                % (note, os.getpid(), level, ("  " + desc) if desc else "")
            )
    except Exception:  # pragma: no cover
        pass

# ---------------------------------------------------------------------------
# 源码字面量开关。A/B 时直接改这一行（不用环境变量 —— vLLM 多进程下
# os.environ 的读取点可能在 EngineCore/worker 子进程，不可靠）。
#   0 = 关闭（等价基线）   1 = BLOCK_M 32 + num_warps 4   2 = 再 + num_stages 1
# ---------------------------------------------------------------------------
FLAGOS_PREFILL_TUNE_LEVEL = 2

# ---- 锚点 1：BLOCK_M / BLOCK_Q 决策 ----
_ANCHOR_BLOCK_M = """    BLOCK_M = (
        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    )
    BLOCK_Q = BLOCK_M // num_queries_per_kv"""

_NEW_BLOCK_M = """    _fl_pf_level = (
        FLAGOS_PREFILL_TUNE_LEVEL
        if (
            max_seqlen_q > 1
            and num_queries_per_kv == 8
            and head_size == 128
        )
        else 0
    )
    BLOCK_M = (
        32
        if _fl_pf_level >= 1
        else (
            16
            if num_queries_per_kv <= 16
            else triton.next_power_of_2(num_queries_per_kv)
        )
    )
    BLOCK_Q = BLOCK_M // num_queries_per_kv"""

# ---- 锚点 2：launch num_warps / num_stages ----
_ANCHOR_WARPS = """    launch_num_warps: int | None = None
    launch_num_stages: int | None = None"""

_NEW_WARPS = """    launch_num_warps: int | None = None
    launch_num_stages: int | None = None
    if _fl_pf_level >= 1:
        launch_num_warps = 4
    if _fl_pf_level >= 2:
        launch_num_stages = 1"""


def apply_triton_attn_prefill_tune() -> bool:
    """安装 prefill launch 特化。返回是否已生效（失败则保持原实现）。"""
    # ⚠️ 留痕必须**先于** level 判定：level=0（对照臂）拿不到留痕的话，
    # 就等于「关」这个状态没有 worker 进程级证据 —— 与铁律④「开关要可靠地关」冲突。
    # 之前 level=0 直接 return，导致 e27 的 V 臂 pf 留痕里根本没有该 pid。
    _trace(FLAGOS_PREFILL_TUNE_LEVEL, "entry")
    if not FLAGOS_PREFILL_TUNE_LEVEL:
        return False

    try:
        from vllm.v1.attention.backends import triton_attn as _tattn
        from vllm.v1.attention.ops import triton_unified_attention as _tua
    except Exception as exc:  # pragma: no cover
        logger.debug("[FL] prefill tune: vLLM attention 模块不可用: %s", exc)
        return False

    # 幂等：worker 进程里 register_model 可能被调用多次
    if getattr(_tattn.unified_attention, "_fl_pf_tuned", False):
        return True

    try:
        src = inspect.getsource(_tua.unified_attention)
    except Exception as exc:  # pragma: no cover
        logger.warning("[FL] prefill tune: 取 unified_attention 源码失败: %s", exc)
        return False

    if src.count(_ANCHOR_BLOCK_M) != 1 or src.count(_ANCHOR_WARPS) != 1:
        logger.warning(
            "[FL] prefill tune: 锚点不匹配（VLLM 版本可能已变更），"
            "跳过特化并保持原实现"
        )
        # ⚠️ 这是「上官方卡后发现版本不同」时唯一的现场证据，必须落文件
        _trace(
            FLAGOS_PREFILL_TUNE_LEVEL,
            "SKIP",
            "anchor mismatch bm=%d warps=%d"
            % (src.count(_ANCHOR_BLOCK_M), src.count(_ANCHOR_WARPS)),
        )
        return False

    new_src = src.replace(_ANCHOR_BLOCK_M, _NEW_BLOCK_M, 1)
    new_src = new_src.replace(_ANCHOR_WARPS, _NEW_WARPS, 1)

    # 用虚拟文件名 + linecache 注册：exec 出来的函数默认没有源文件，
    # 某些框架/调试路径调用 inspect.getsource 会抛 OSError，这里补上。
    src_name = "<vllm_fl.prefill_tune>"
    ns = dict(_tua.__dict__)
    ns["FLAGOS_PREFILL_TUNE_LEVEL"] = FLAGOS_PREFILL_TUNE_LEVEL
    try:
        exec(compile(new_src, src_name, "exec"), ns)  # noqa: S102
    except Exception as exc:  # pragma: no cover
        logger.warning("[FL] prefill tune: 编译特化版本失败: %s", exc)
        return False
    try:
        import linecache

        linecache.cache[src_name] = (
            len(new_src),
            None,
            new_src.splitlines(True),
            src_name,
        )
    except Exception:  # pragma: no cover
        pass

    tuned = ns.get("unified_attention")
    if not callable(tuned):  # pragma: no cover
        logger.warning("[FL] prefill tune: 未得到特化函数，跳过")
        return False

    tuned._fl_pf_tuned = True  # type: ignore[attr-defined]
    # 同时替换源模块与 backend 模块里的符号（backend 是模块级引用）
    _tattn.unified_attention = tuned
    _tua.unified_attention = tuned

    desc = {
        1: "BLOCK_M=32, num_warps=4",
        2: "BLOCK_M=32, num_warps=4, num_stages=1",
    }.get(FLAGOS_PREFILL_TUNE_LEVEL, "?")
    logger.info(
        "[FL] TRITON_ATTN prefill launch 特化已启用（level=%d）："
        "npk==8 & head_size==128 & max_seqlen_q>1 → %s",
        FLAGOS_PREFILL_TUNE_LEVEL,
        desc,
    )
    _trace(FLAGOS_PREFILL_TUNE_LEVEL, "applied", desc)
    return True


__all__ = ["apply_triton_attn_prefill_tune", "FLAGOS_PREFILL_TUNE_LEVEL"]
