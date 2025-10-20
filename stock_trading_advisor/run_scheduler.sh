#!/bin/bash
# 股票交易提醒系统 - 快速启动脚本

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "  股票交易定时提醒系统"
echo "========================================"
echo ""

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到 python3"
    exit 1
fi

# 检查依赖
if ! python3 -c "import schedule" 2>/dev/null; then
    echo "⚠️  检测到缺少依赖，正在安装..."
    pip3 install --break-system-packages -q schedule || {
        echo "❌ 安装失败，请手动运行: pip3 install -r requirements.txt"
        exit 1
    }
    echo "✓ 依赖安装完成"
    echo ""
fi

# 显示菜单
echo "请选择操作:"
echo "  1) 立即运行一次分析"
echo "  2) 启动定时调度器"
echo "  3) 立即运行并启动调度器"
echo "  4) 查看最新报告"
echo "  5) 编辑监控列表"
echo "  6) 编辑配置文件"
echo "  0) 退出"
echo ""
read -p "请输入选项 [0-6]: " choice

case $choice in
    1)
        echo ""
        echo "🚀 正在执行分析..."
        python3 scheduler.py --run-now
        ;;
    2)
        echo ""
        echo "⏰ 启动定时调度器..."
        python3 scheduler.py
        ;;
    3)
        echo ""
        echo "🚀 立即运行并启动调度器..."
        python3 scheduler.py --run-now --start
        ;;
    4)
        echo ""
        if [ -f "reports/trading_alerts.txt" ]; then
            echo "📊 最新报告:"
            echo "========================================"
            tail -50 reports/trading_alerts.txt
        else
            echo "❌ 报告文件不存在"
        fi
        ;;
    5)
        ${EDITOR:-nano} config/watch_list.txt
        ;;
    6)
        ${EDITOR:-nano} config/scheduler_config.yaml
        ;;
    0)
        echo "👋 再见！"
        exit 0
        ;;
    *)
        echo "❌ 无效选项"
        exit 1
        ;;
esac
