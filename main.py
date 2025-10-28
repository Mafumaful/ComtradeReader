#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMTRADE波形可视化工具 - 主启动脚本

使用方法:
    python main.py              # 使用交互式菜单（支持方向键）
    python main.py --simple     # 使用简单数字菜单
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from waveform_viewer.app import WaveformViewerApp


def main():
    """主函数"""
    # 解析命令行参数
    use_simple_menu = '--simple' in sys.argv or '-s' in sys.argv

    # 波形文件夹路径
    waves_dir = Path(__file__).parent / "waves"

    if not waves_dir.exists():
        print(f"错误: 波形文件夹不存在: {waves_dir}")
        print("请确保 'waves' 文件夹存在并包含波形数据")
        print("\n建议的目录结构:")
        print("  波形记录/")
        print("  ├── main.py")
        print("  ├── waves/")
        print("  │   ├── 机励磁波形2025.1013/")
        print("  │   │   ├── *.cfg")
        print("  │   │   └── *.dat")
        print("  │   └── 其他波形文件夹/")
        print("  └── waveform_viewer/")
        sys.exit(1)

    # 创建并运行应用
    print("正在启动COMTRADE波形可视化工具...")
    app = WaveformViewerApp(str(waves_dir), use_simple_menu=use_simple_menu)
    app.run()


if __name__ == '__main__':
    main()
