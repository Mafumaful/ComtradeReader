#!/bin/bash
# 波形可视化程序运行脚本

cd "$(dirname "$0")"

# 检查虚拟环境是否存在
if [ ! -d "venv" ]; then
    echo "创建Python虚拟环境..."
    python3 -m venv venv
    source venv/bin/activate
    echo "安装依赖包..."
    pip install plotly numpy
else
    source venv/bin/activate
fi

echo "运行波形可视化程序..."
python waveform_viewer.py

echo ""
echo "完成！可视化结果已保存为HTML文件。"
echo "您可以在浏览器中打开这些HTML文件查看波形。"
