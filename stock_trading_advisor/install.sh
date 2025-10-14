#!/bin/bash
# 安装脚本 - 使用虚拟环境

echo "创建虚拟环境..."
python3 -m venv venv --without-pip

echo "激活虚拟环境..."
source venv/bin/activate

echo "安装 pip..."
curl https://bootstrap.pypa.io/get-pip.py | python

echo "安装依赖..."
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "安装完成！"
echo ""
echo "使用方法："
echo "  source venv/bin/activate"
echo "  python main.py -s 000001"
