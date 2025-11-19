"""
策略选择缓存模块

用于缓存每只股票在自适应评估后选出的最优组合：
- 卖出策略: original / gradual
- 执行顺序: high_frequency / high_quality

缓存位置:
    data/strategy_cache.json

注意:
- 仅用于加速回测或固定策略模式，不参与交易逻辑本身
- 如果配置或算法有重大调整，建议手动清空该文件
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import threading

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_cache_loaded = False
_strategy_cache: Dict[str, Dict[str, Any]] = {}

# data 目录位于 src 的上一级
_ROOT = Path(__file__).parent.parent
_CACHE_PATH = _ROOT / "data" / "strategy_cache.json"


def _load_cache() -> None:
    """惰性加载策略缓存到内存"""
    global _cache_loaded, _strategy_cache
    if _cache_loaded:
        return

    with _lock:
        if _cache_loaded:
            return

        if _CACHE_PATH.exists():
            try:
                with _CACHE_PATH.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        _strategy_cache = data
                    else:
                        logger.warning("策略缓存文件格式异常，忽略并重建")
                        _strategy_cache = {}
            except Exception as e:
                logger.warning(f"读取策略缓存失败 { _CACHE_PATH }: {e}，将忽略旧缓存")
                _strategy_cache = {}
        else:
            _strategy_cache = {}

        _cache_loaded = True


def _save_cache() -> None:
    """将内存中的策略缓存写回文件"""
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _CACHE_PATH.with_suffix(".json.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(_strategy_cache, f, ensure_ascii=False, indent=2)
        tmp_path.replace(_CACHE_PATH)
    except Exception as e:
        logger.warning(f"写入策略缓存失败 { _CACHE_PATH }: {e}")


def get_strategy_choice(code: str) -> Optional[Dict[str, Any]]:
    """
    获取某只股票的策略选择

    Returns:
        {'sell_strategy': 'original'|'gradual', 'order': 'high_frequency'|'high_quality', ...}
        或 None 表示无缓存
    """
    if not code:
        return None

    _load_cache()
    with _lock:
        entry = _strategy_cache.get(str(code))
        # 返回浅拷贝，避免调用方修改内部结构
        return dict(entry) if isinstance(entry, dict) else None


def save_strategy_choice(code: str, sell_strategy: str, order: str) -> None:
    """
    保存/更新某只股票的策略选择

    仅在自适应评估完成后调用。
    """
    if not code:
        return

    _load_cache()
    key = str(code)
    with _lock:
        old = _strategy_cache.get(key)
        if (
            isinstance(old, dict)
            and old.get("sell_strategy") == sell_strategy
            and old.get("order") == order
        ):
            # 未发生变化，避免无谓的写入
            return

        _strategy_cache[key] = {
            "sell_strategy": sell_strategy,
            "order": order,
        }
        _save_cache()

