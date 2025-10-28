#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基本功能测试脚本

测试重构后的代码是否能够正常工作
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径（test文件夹的父目录）
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """测试模块导入"""
    print("测试1: 模块导入...")
    try:
        from waveform_viewer.core.reader import ComtradeReader, Channel
        from waveform_viewer.core.channel_selector import (
            ImportantPatternSelector,
            FirstNChannelsSelector,
            AllChannelsSelector
        )
        from waveform_viewer.visualizers.plotly_viz import PlotlyVisualizer
        from waveform_viewer.plugins.manager import PluginManager
        from waveform_viewer.ui.menu import SimpleMenu
        from waveform_viewer.utils.file_finder import WaveformFileFinder
        from waveform_viewer.app import WaveformViewerApp
        print("  ✓ 所有模块导入成功")
        return True
    except Exception as e:
        print(f"  ✗ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_file_finder():
    """测试文件查找器"""
    print("\n测试2: 文件查找器...")
    try:
        from waveform_viewer.utils.file_finder import WaveformFileFinder

        # 使用绝对路径：项目根目录下的 waves 文件夹
        waves_path = Path(__file__).parent.parent / "waves"
        finder = WaveformFileFinder(str(waves_path))
        folders = finder.find_waveform_folders()

        print(f"  找到 {len(folders)} 个波形文件夹")
        for folder in folders:
            cfg_files = finder.find_cfg_files(folder)
            print(f"    - {folder.name}: {len(cfg_files)} 个文件")

        print("  ✓ 文件查找器工作正常")
        return True
    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
        return False


def test_channel_selector():
    """测试通道选择器"""
    print("\n测试3: 通道选择器...")
    try:
        from waveform_viewer.core.reader import Channel
        from waveform_viewer.core.channel_selector import ImportantPatternSelector

        # 创建测试通道
        channels = [
            Channel(1, "A相电压", "", "kV", 1.0, 0.0),
            Channel(2, "B相电压", "", "kV", 1.0, 0.0),
            Channel(3, "A相电流", "", "A", 1.0, 0.0),
            Channel(4, "有功功率", "", "MW", 1.0, 0.0),
            Channel(5, "频率", "", "Hz", 1.0, 0.0),
            Channel(6, "其他参数", "", "", 1.0, 0.0),
        ]

        selector = ImportantPatternSelector()
        selected = selector.select_channels(channels, max_channels=10)

        print(f"  从 {len(channels)} 个通道中选择了 {len(selected)} 个")
        for ch in selected:
            print(f"    - {ch.name}")

        print("  ✓ 通道选择器工作正常")
        return True
    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
        return False


def test_plugin_manager():
    """测试插件管理器"""
    print("\n测试4: 插件管理器...")
    try:
        from waveform_viewer.plugins.manager import PluginManager

        pm = PluginManager()
        pm.add_plugin_directory("waveform_viewer/plugins")
        pm.discover_plugins()

        plugins = pm.get_all_plugins()
        print(f"  加载了 {len(plugins)} 个插件")
        for name, plugin in plugins.items():
            print(f"    - {name}: {plugin.description}")

        print("  ✓ 插件管理器工作正常")
        return True
    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reader():
    """测试COMTRADE读取器"""
    print("\n测试5: COMTRADE读取器...")
    try:
        from waveform_viewer.core.reader import ComtradeReader
        from pathlib import Path

        # 查找第一个.cfg文件
        waves_dir = Path(__file__).parent.parent / "waves"
        cfg_files = list(waves_dir.rglob("*.cfg"))

        if not cfg_files:
            print("  ⚠ 未找到测试文件，跳过测试")
            return True

        cfg_file = cfg_files[0]
        print(f"  测试文件: {cfg_file.name}")

        reader = ComtradeReader(str(cfg_file))

        print(f"  - 站点: {reader.station_name}")
        print(f"  - 采样率: {reader.sample_rate} Hz")
        print(f"  - 样本数: {reader.num_samples}")
        print(f"  - 模拟通道数: {len(reader.analog_channels)}")
        print(f"  - 数字通道数: {len(reader.digital_channels)}")
        print(f"  - 时间范围: {reader.time_values[0]:.6f}s ~ {reader.time_values[-1]:.6f}s")

        print("  ✓ COMTRADE读取器工作正常")
        return True
    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("=" * 70)
    print("COMTRADE波形可视化工具 - 基本功能测试")
    print("=" * 70)

    tests = [
        test_imports,
        test_file_finder,
        test_channel_selector,
        test_plugin_manager,
        test_reader,
    ]

    results = []
    for test in tests:
        results.append(test())

    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)

    passed = sum(results)
    total = len(results)

    print(f"通过: {passed}/{total}")

    if passed == total:
        print("\n✓ 所有测试通过！代码工作正常。")
        return 0
    else:
        print(f"\n✗ {total - passed} 个测试失败。请检查错误信息。")
        return 1


if __name__ == '__main__':
    sys.exit(main())
