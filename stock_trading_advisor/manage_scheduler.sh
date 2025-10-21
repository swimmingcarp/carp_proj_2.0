#!/bin/bash
# Stock Trading Scheduler 管理脚本

SERVICE_NAME="stock-scheduler.service"

case "$1" in
    start)
        echo "🚀 启动调度器服务..."
        sudo systemctl start $SERVICE_NAME
        sleep 1
        sudo systemctl status $SERVICE_NAME --no-pager
        ;;
    stop)
        echo "🛑 停止调度器服务..."
        sudo systemctl stop $SERVICE_NAME
        echo "✓ 服务已停止"
        ;;
    restart)
        echo "🔄 重启调度器服务..."
        sudo systemctl restart $SERVICE_NAME
        sleep 1
        sudo systemctl status $SERVICE_NAME --no-pager
        ;;
    status)
        echo "📊 调度器服务状态："
        sudo systemctl status $SERVICE_NAME --no-pager
        echo ""
        echo "📋 进程信息："
        ps aux | grep "scheduler.py" | grep -v grep
        ;;
    logs)
        echo "📜 查看最近日志："
        echo ""
        echo "=== 服务输出日志 ==="
        tail -50 /home/ubuntu/workspace/carp_proj_2.0/stock_trading_advisor/logs/scheduler_service.log
        echo ""
        echo "=== 调度器内部日志 ==="
        tail -50 /home/ubuntu/workspace/carp_proj_2.0/stock_trading_advisor/logs/scheduler.log
        ;;
    enable)
        echo "✅ 设置开机自启动..."
        sudo systemctl enable $SERVICE_NAME
        ;;
    disable)
        echo "❌ 禁用开机自启动..."
        sudo systemctl disable $SERVICE_NAME
        ;;
    *)
        echo "Stock Trading Scheduler 管理工具"
        echo ""
        echo "用法: $0 {start|stop|restart|status|logs|enable|disable}"
        echo ""
        echo "命令说明："
        echo "  start    - 启动调度器服务"
        echo "  stop     - 停止调度器服务"
        echo "  restart  - 重启调度器服务"
        echo "  status   - 查看服务状态"
        echo "  logs     - 查看最近日志"
        echo "  enable   - 设置开机自启动"
        echo "  disable  - 禁用开机自启动"
        echo ""
        echo "示例："
        echo "  $0 status   # 查看运行状态"
        echo "  $0 logs     # 查看日志"
        echo "  $0 restart  # 重启服务"
        exit 1
        ;;
esac
