# Stock Trading Scheduler - Systemd 服务管理

## 📋 概述

调度器现在已配置为 systemd 服务，具有以下特性：
- ✅ **开机自启动** - 系统重启后自动运行
- ✅ **崩溃自动重启** - 进程异常退出后10秒自动重启
- ✅ **日志记录** - 完整的服务日志和运行日志
- ✅ **进程监控** - systemd 自动监控进程状态

## 🚀 快速开始

### 使用管理脚本（推荐）

```bash
cd /home/ubuntu/workspace/carp_proj_2.0/stock_trading_advisor

# 查看服务状态
./manage_scheduler.sh status

# 查看日志
./manage_scheduler.sh logs

# 重启服务
./manage_scheduler.sh restart

# 停止服务
./manage_scheduler.sh stop

# 启动服务
./manage_scheduler.sh start
```

### 直接使用 systemctl 命令

```bash
# 查看服务状态
sudo systemctl status stock-scheduler.service

# 启动服务
sudo systemctl start stock-scheduler.service

# 停止服务
sudo systemctl stop stock-scheduler.service

# 重启服务
sudo systemctl restart stock-scheduler.service

# 查看实时日志
sudo journalctl -u stock-scheduler.service -f

# 查看最近100行日志
sudo journalctl -u stock-scheduler.service -n 100
```

## 📂 日志文件位置

1. **服务标准输出日志**
   ```
   /home/ubuntu/workspace/carp_proj_2.0/stock_trading_advisor/logs/scheduler_service.log
   ```

2. **服务错误日志**
   ```
   /home/ubuntu/workspace/carp_proj_2.0/stock_trading_advisor/logs/scheduler_service_error.log
   ```

3. **调度器内部日志**
   ```
   /home/ubuntu/workspace/carp_proj_2.0/stock_trading_advisor/logs/scheduler.log
   ```

4. **systemd 系统日志**
   ```bash
   sudo journalctl -u stock-scheduler.service
   ```

## 🔧 配置文件

### 调度器配置
```
/home/ubuntu/workspace/carp_proj_2.0/stock_trading_advisor/config/scheduler_config.yaml
```

**修改配置后需要重启服务：**
```bash
./manage_scheduler.sh restart
```

### systemd 服务配置
```
/etc/systemd/system/stock-scheduler.service
```

**修改服务配置后需要：**
```bash
sudo systemctl daemon-reload
sudo systemctl restart stock-scheduler.service
```

## 📊 当前配置

- **运行时间**: 每天 14:57
- **监控股票**: 6只（见 config/watch_list.txt）
- **微信通知**: 已启用（企业微信机器人）
- **交易日检查**: 已启用（只在工作日运行）

## ⚙️ 重启策略

服务配置了自动重启策略：
- **Restart=always** - 无论何种原因退出都会重启
- **RestartSec=10** - 退出后等待10秒重启

这确保了即使进程崩溃，也会在10秒内自动恢复。

## 🔍 故障排查

### 服务无法启动

```bash
# 查看详细错误信息
sudo systemctl status stock-scheduler.service -l

# 查看完整日志
sudo journalctl -u stock-scheduler.service --no-pager
```

### 任务未执行

```bash
# 检查服务是否运行
./manage_scheduler.sh status

# 查看最近日志
./manage_scheduler.sh logs

# 手动运行一次测试
cd /home/ubuntu/workspace/carp_proj_2.0/stock_trading_advisor
python3 scheduler.py --run-now
```

### 微信消息未发送

1. 检查微信配置：`config/scheduler_config.yaml`
2. 检查webhook URL是否正确
3. 查看错误日志：`logs/scheduler_service_error.log`

## 📝 维护建议

1. **定期查看日志**
   ```bash
   ./manage_scheduler.sh logs
   ```

2. **监控服务状态**
   ```bash
   ./manage_scheduler.sh status
   ```

3. **更新配置后重启**
   ```bash
   ./manage_scheduler.sh restart
   ```

4. **清理旧日志**（可选）
   ```bash
   # 清空日志文件
   > logs/scheduler_service.log
   > logs/scheduler.log
   ```

## ⚠️ 注意事项

1. **不要手动运行 scheduler.py** - 会与 systemd 服务冲突
2. **修改配置后记得重启服务**
3. **开机后无需手动启动** - 服务会自动运行
4. **进程崩溃会自动重启** - 无需担心进程异常退出

## 📞 常见问题

**Q: 如何查看今天是否发送了微信消息？**
```bash
tail -50 logs/scheduler.log | grep "微信通知"
```

**Q: 如何临时禁用调度器？**
```bash
./manage_scheduler.sh stop
```

**Q: 如何永久禁用开机自启动？**
```bash
./manage_scheduler.sh disable
```

**Q: 如何立即执行一次任务（不等待定时）？**
```bash
# 先停止服务
./manage_scheduler.sh stop

# 手动运行一次
python3 scheduler.py --run-now

# 重新启动服务
./manage_scheduler.sh start
```
