"""
数据获取配置文件
只保留缓存机制和基础配置
"""

# 数据源配置
DATA_SOURCE = 'akshare'  # 可选: 'akshare', 'tushare', 'yfinance'
DEFAULT_ADJUST = 'qfq'   # 普通分析/实盘信号默认使用前复权

# ============ 缓存配置 ============
CACHE_ENABLED = True        # 是否启用本地缓存
CACHE_MAX_AGE_DAYS = 1      # 缓存有效期（天）

# 说明：
# - 缓存位置: data/cache/
# - 缓存格式: {股票代码}_{哈希}.csv
# - 缓存作用: 避免重复请求，提高速度
# - 清空缓存: rm data/cache/*.csv

# ============ 基础配置 ============
MAX_RETRIES = 3             # 最大重试次数
RETRY_DELAY = 1.0           # 重试延迟（秒）
VALIDATE_DATA = True        # 是否启用数据验证

# ============ 其他配置 ============
# Tushare 配置（如果使用 Tushare 数据源）
TUSHARE_TOKEN = None

# 日志配置
LOG_LEVEL = 'INFO'  # 可选: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
