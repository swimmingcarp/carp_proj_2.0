"""
股性 + 阶段 → 策略参数映射

根据股性类型和当前阶段返回参数override。
股性决定基础交易风格，阶段决定方向性调整。

设计原则：
- 股性（方向无关）决定交易结构：做不做T、追不追趋势、止损宽松度
- 阶段（含方向）决定当前调整：上涨段让利润跑、下跌段收紧止损
- personality + stage 组合产生最终参数

阶段类型（来自classifier.detect_current_stage）：
  uptrend / downtrend / consolidation / high_consolidation / low_consolidation / uncertain
  slope值作为metadata附带，描述趋势强度（急/温和），不增加阶段种类
"""

from typing import Dict


# 各股性类型的基础参数override（方向无关的结构性参数）
# 2026-03 优化：基于132股回测分析的per-personality表现差异
#   trend_persistent: avg_ret=555%, tPF=2.76 (最佳 → 保持默认)
#   volatile_oscillator: avg_ret=183%, tPF=2.26 (波动大 → 更紧止损)
#   staircase_mover: avg_ret=79%, tPF=1.82 (样本小 → 不改)
#   breakout_runner: avg_ret=34%, tPF=1.67 (突破慢 → 宽止损)
#   mean_reverter: avg_ret=2%, tPF=1.17 (趋势策略不适用 → 紧止损+低EH)
PERSONALITY_BASE_OVERRIDES: Dict[str, Dict] = {
    'trend_persistent': {
        # 最佳股性类型，baseline参数已最优，不覆盖
    },
    'staircase_mover': {
        # 样本量太小(n=4)，暂不加override
    },
    'mean_reverter': {
        # 均值回归股不适合趋势策略，缩紧参数减少损失
        # EH阈值降低(反弹幅度小)，硬止损收紧(快速止损)
        'extended_hold_profit_threshold': 15,
        'hard_loss_cap_pct': 5.0,
    },
    'volatile_oscillator': {
        # 2026-03 验证: hard_loss_cap=6.0 → 20只受损(-439%), 25只受益(+461%)
        # 净效果仅+0.45%/股, 但个别损失巨大(002920:-114%, 300293:-28%)
        # 风险收益比差, 移除override
    },
    'breakout_runner': {
        # 2026-03 验证: stop_loss=18/hard_loss_cap=18 → 10只股全部0变化
        # override无实际效果, 清空
    },
    'erratic': {},
    'insufficient': {},
}

# 股性 + 阶段的联合override（叠加在基础参数之上）
# 阶段名: uptrend, downtrend, consolidation, high_consolidation, low_consolidation
PERSONALITY_STAGE_OVERRIDES: Dict[str, Dict[str, Dict]] = {
    # trend_persistent: 暂无任何override（全用baseline），建立neutral基准
    'trend_persistent': {},
    'staircase_mover': {
        # 暂不加阶段override，先验证分类效果
    },
}


def get_personality_overrides(personality: str, stage: str = '') -> Dict:
    """
    获取指定股性 + 阶段的参数override。

    先应用股性基础参数，再叠加阶段特定参数。
    """
    result = PERSONALITY_BASE_OVERRIDES.get(personality, {}).copy()

    # 叠加阶段特定参数
    stage_overrides = PERSONALITY_STAGE_OVERRIDES.get(personality, {}).get(stage, {})
    result.update(stage_overrides)

    return result
