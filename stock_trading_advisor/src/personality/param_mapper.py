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
PERSONALITY_BASE_OVERRIDES: Dict[str, Dict] = {
    'trend_persistent': {
        # sweep验证：baseline 15%止损和30% EH阈值已是最优，不覆盖
    },
    'staircase_mover': {
        # 样本量太小(n=4)，暂不加override
    },
    'mean_reverter': {
        # sweep: EH阈值20%最优(+0.015 vs 30%基线), n=6
        # 均值回归股反弹幅度较小，更早激活EH能多抓一些利润
        'extended_hold_profit_threshold': 20,
    },
    'volatile_oscillator': {
        # 暂不加override：40%在group-sweep有效，但PE跨类型交叉污染削弱效果
    },
    'breakout_runner': {
        # sweep: 止损18%比15%好(+0.024), n=10
        # 601328的37/241个entry bar被分为mean_reverter→EH=20影响，其余124个breakout_runner entry
        # 可获益于宽止损，组合测试确认净效果
        'stop_loss': 18.0,
        'hard_loss_cap_pct': 18.0,
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
