#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能测试脚本

测试不同可视化器的性能和效果
"""

import sys
import time
from pathlib import Path

# 添加项目根目录到Python路径（test文件夹的父目录）
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_visualizers():
    """测试不同可视化器的性能"""
    print("=" * 70)
    print("性能测试：大量通道可视化")
    print("=" * 70)

    try:
        from waveform_viewer.core.reader import ComtradeReader
        from waveform_viewer.visualizers.plotly_viz import PlotlyVisualizer
        from waveform_viewer.visualizers.optimized_plotly_viz import (
            OptimizedPlotlyVisualizer,
            FastPlotlyVisualizer
        )
        from waveform_viewer.visualizers.base import VisualizationConfig

        # 查找测试文件
        waves_dir = Path(__file__).parent.parent / "waves"
        cfg_files = list(waves_dir.rglob("*.cfg"))

        if not cfg_files:
            print("⚠ 未找到测试文件")
            return 1

        cfg_file = cfg_files[0]
        print(f"\n测试文件: {cfg_file.name}")

        # 读取数据
        print("\n读取数据...")
        reader = ComtradeReader(str(cfg_file))
        print(f"通道数: {len(reader.analog_channels)}")
        print(f"数据点: {len(reader.time_values)}")
        print(f"总数据量: {len(reader.analog_channels) * len(reader.time_values):,} 个点")

        # 使用所有通道进行测试
        channels = reader.analog_channels

        # 配置（禁用自动打开以便测试）
        config = VisualizationConfig(auto_open=False)

        visualizers = [
            ("标准模式（Scatter）", PlotlyVisualizer(config), "test_standard.html"),
            ("优化模式（WebGL）", OptimizedPlotlyVisualizer(config), "test_optimized.html"),
            ("快速模式", FastPlotlyVisualizer(config), "test_fast.html"),
        ]

        results = []

        for name, visualizer, output_file in visualizers:
            print("\n" + "-" * 70)
            print(f"测试: {name}")
            print("-" * 70)

            output_path = Path(output_file)

            # 计时
            start_time = time.time()

            try:
                visualizer.visualize(reader, channels, str(output_path))
                elapsed = time.time() - start_time

                # 获取文件大小
                if output_path.exists():
                    file_size = output_path.stat().st_size / (1024 * 1024)  # MB
                    print(f"✓ 完成")
                    print(f"  耗时: {elapsed:.2f} 秒")
                    print(f"  文件大小: {file_size:.2f} MB")

                    results.append({
                        'name': name,
                        'time': elapsed,
                        'size': file_size,
                        'success': True
                    })

                    # 清理测试文件
                    output_path.unlink()
                else:
                    print(f"✗ 生成失败")
                    results.append({
                        'name': name,
                        'success': False
                    })

            except Exception as e:
                print(f"✗ 错误: {e}")
                results.append({
                    'name': name,
                    'success': False,
                    'error': str(e)
                })

        # 输出比较结果
        print("\n" + "=" * 70)
        print("性能比较")
        print("=" * 70)
        print(f"{'模式':<25} {'耗时':<12} {'文件大小':<12} {'状态'}")
        print("-" * 70)

        for result in results:
            if result['success']:
                print(f"{result['name']:<25} {result['time']:<10.2f}s {result['size']:<10.2f}MB ✓")
            else:
                print(f"{result['name']:<25} {'N/A':<12} {'N/A':<12} ✗")

        # 计算改进
        if len([r for r in results if r['success']]) >= 2:
            standard_time = next((r['time'] for r in results if r['success'] and '标准' in r['name']), None)
            optimized_time = next((r['time'] for r in results if r['success'] and '优化' in r['name']), None)

            if standard_time and optimized_time:
                improvement = ((standard_time - optimized_time) / standard_time) * 100
                print(f"\n性能提升: {improvement:.1f}%")

        print("\n" + "=" * 70)
        print("建议:")
        print("  • 大量通道（>20）: 使用优化模式或快速模式")
        print("  • 少量通道（<10）: 使用标准模式")
        print("  • 极大量通道（>40）: 优先使用快速模式")
        print("=" * 70)

        return 0

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(test_visualizers())
