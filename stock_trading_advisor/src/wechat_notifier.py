#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
微信通知模块
支持企业微信机器人推送
"""

import requests
import json
import logging
from typing import Dict, Optional
from datetime import datetime


class WorkWeChatRobot:
    """企业微信机器人通知"""

    def __init__(self, webhook_url: str):
        """
        初始化企业微信机器人

        Args:
            webhook_url: 机器人webhook地址
        """
        self.webhook_url = webhook_url
        self.logger = logging.getLogger(__name__)

    def send(self, title: str, content: str, msg_type: str = 'text') -> bool:
        """
        发送通知到企业微信群

        Args:
            title: 通知标题
            content: 通知内容
            msg_type: 消息类型（text/markdown）

        Returns:
            是否发送成功
        """
        try:
            if msg_type == 'markdown':
                # Markdown格式：使用更美观的排版
                data = {
                    "msgtype": "markdown",
                    "markdown": {
                        "content": content  # Markdown格式不需要单独的标题
                    }
                }
            else:
                data = {
                    "msgtype": "text",
                    "text": {
                        "content": f"{title}\n\n{content}"
                    }
                }

            response = requests.post(
                self.webhook_url,
                json=data,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )

            result = response.json()

            if result.get('errcode') == 0:
                self.logger.info("企业微信通知发送成功")
                return True
            else:
                self.logger.error(f"企业微信通知发送失败: {result.get('errmsg')}")
                return False

        except Exception as e:
            self.logger.error(f"发送企业微信通知异常: {e}")
            return False


class WeChatNotificationManager:
    """微信通知管理器"""

    def __init__(self, config: Dict):
        """
        初始化通知管理器

        Args:
            config: 通知配置
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.notifiers = []

        # 初始化企业微信通知器
        self._init_notifiers()

    def _init_notifiers(self):
        """初始化通知器"""
        wechat_config = self.config.get('wechat', {})

        # 企业微信机器人
        if wechat_config.get('work_wechat', {}).get('enabled'):
            webhook_url = wechat_config['work_wechat'].get('webhook_url')
            if webhook_url:
                self.notifiers.append({
                    'name': '企业微信',
                    'notifier': WorkWeChatRobot(webhook_url),
                    'msg_type': wechat_config['work_wechat'].get('msg_type', 'text')
                })
                self.logger.info("已启用企业微信通知")

    def send_alert(self, alert_text: str) -> bool:
        """
        发送交易提醒

        Args:
            alert_text: 提醒文本

        Returns:
            是否发送成功
        """
        if not self.notifiers:
            self.logger.warning("未配置企业微信通知")
            return False

        # 发送到企业微信
        for notifier_info in self.notifiers:
            name = notifier_info['name']
            notifier = notifier_info['notifier']
            msg_type = notifier_info.get('msg_type', 'text')

            try:
                # 根据消息类型格式化内容
                if msg_type == 'markdown':
                    content = self._format_markdown(alert_text)
                else:
                    title = f"📊 股票交易提醒 - {datetime.now().strftime('%m月%d日 %H:%M')}"
                    content = alert_text

                if notifier.send("", content, msg_type):
                    self.logger.info(f"{name} 发送成功")
                    return True
                else:
                    self.logger.error(f"{name} 发送失败")
                    return False
            except Exception as e:
                self.logger.error(f"{name} 发送失败: {e}")
                return False

        return False

    def _format_markdown(self, alert_text: str) -> str:
        """
        将纯文本格式转换为Markdown格式

        Args:
            alert_text: 原始提醒文本

        Returns:
            Markdown格式的文本
        """
        lines = alert_text.split('\n')
        markdown_lines = []

        for line in lines:
            # 标题行
            if '股票交易提醒' in line:
                # 提取时间
                import re
                match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', line)
                if match:
                    time_str = match.group()
                    # 转换为更友好的格式
                    from datetime import datetime
                    dt = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
                    friendly_time = dt.strftime('%m月%d日 %H:%M')
                    markdown_lines.append(f"# 📊 股票交易提醒")
                    markdown_lines.append(f"> {friendly_time}")
                else:
                    markdown_lines.append(f"# {line.strip()}")
            # 分隔线跳过
            elif '=' in line and len(set(line.strip())) <= 1:
                continue
            # 统计信息行
            elif '监控股票总数:' in line:
                markdown_lines.append(f"**{line.strip()}**")
            elif '买入:' in line and '卖出:' in line:
                markdown_lines.append(f"**{line.strip()}**")
            # 买入/卖出信号标题
            elif '买入信号' in line or '卖出信号' in line:
                markdown_lines.append(f"\n## {line.strip()}")
            # 持有/观望标题
            elif '持有中' in line or '观望' in line or '全部观望' in line:
                markdown_lines.append(f"\n## {line.strip()}")
            # 股票代码行 【000001】格式
            elif '【' in line and '】' in line:
                markdown_lines.append(f"\n### {line.strip()}")
            # 操作行、理由行等
            elif '操作:' in line or '理由:' in line or 'K值:' in line or '价格:' in line:
                # 添加代码块样式
                markdown_lines.append(f"`{line.strip()}`")
            # 股票列表行 (数字开头)
            elif line.strip() and line.strip()[0].isdigit():
                markdown_lines.append(f"- {line.strip()}")
            # 其他行
            elif line.strip():
                markdown_lines.append(line)
            # 空行
            else:
                if markdown_lines and markdown_lines[-1] != '':
                    markdown_lines.append('')

        return '\n'.join(markdown_lines)

    def test_notification(self) -> Dict[str, bool]:
        """
        测试企业微信通知器

        Returns:
            测试结果
        """
        results = {}
        test_title = "测试通知"
        test_content = "这是一条测试消息，如果你收到了这条消息，说明企业微信通知配置成功！"

        for notifier_info in self.notifiers:
            name = notifier_info['name']
            notifier = notifier_info['notifier']
            msg_type = notifier_info.get('msg_type', 'text')

            try:
                results[name] = notifier.send(test_title, test_content, msg_type)
            except Exception as e:
                self.logger.error(f"{name} 测试失败: {e}")
                results[name] = False

        return results


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    # 测试配置
    test_config = {
        'wechat': {
            'work_wechat': {
                'enabled': False,  # 改为True并填入webhook_url
                'webhook_url': 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=YOUR_KEY',
                'msg_type': 'text'
            }
        }
    }

    manager = WeChatNotificationManager(test_config)

    if manager.notifiers:
        print("开始测试企业微信通知...")
        results = manager.test_notification()
        print("\n测试结果：")
        for name, success in results.items():
            status = "✓ 成功" if success else "✗ 失败"
            print(f"  {name}: {status}")
    else:
        print("未配置企业微信通知，请修改配置后重试")
