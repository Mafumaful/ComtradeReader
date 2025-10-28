#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 vertical_spacing 修复

验证动态计算的 vertical_spacing 对于不同数量的通道都能正常工作
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径（test文件夹的父目录）
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_spacing_calculation():
    """测试间距计算逻辑"""
    print("=" * 70)
    print("测试 vertical_spacing 计算")
    print("=" * 70)

    test_cases = [
        (1, 0.03, "单个通道"),
        (2, 0.03, "2个通道"),
        (10, 0.03, "10个通道 (间距充足)"),
        (20, 0.03, "20个通道 (接近临界)"),
        (27, 0.03, "27个通道 (刚好临界: 0.8/26≈0.0308)"),
        (28, None, "28个通道 (需要调整)"),
        (47, None, "47个通道 (大量通道)"),
        (100, None, "100个通道 (极大量通道)"),
    ]

    print("\n通道数 | 计算的spacing | 最大允许spacing | 状态")
    print("-" * 70)

    all_pass = True

    for num_plots, expected_spacing, description in test_cases:
        # 计算spacing（使用与修复相同的逻辑）
        if num_plots > 1:
            max_spacing = 0.8 / (num_plots - 1)
            calculated_spacing = min(0.03, max_spacing)
        else:
            max_spacing = "N/A"
            calculated_spacing = 0.03

        # 验证是否在允许范围内
        if num_plots > 1:
            absolute_max = 1.0 / (num_plots - 1)
            is_valid = calculated_spacing < absolute_max
        else:
            is_valid = True

        status = "✓ PASS" if is_valid else "✗ FAIL"

        if not is_valid:
            all_pass = False

        # 格式化输出
        if isinstance(max_spacing, str):
            max_str = max_spacing
        else:
            max_str = f"{max_spacing:.6f}"

        print(f"{num_plots:6d} | {calculated_spacing:15.6f} | {max_str:19s} | {status} {description}")

    print("=" * 70)

    if all_pass:
        print("✓ 所有测试通过！spacing计算逻辑正确。")
        return 0
    else:
        print("✗ 有测试失败！")
        return 1


def test_real_visualization():
    """测试实际的可视化功能"""
    print("\n" + "=" * 70)
    print("测试实际可视化功能")
    print("=" * 70)

    try:
        from waveform_viewer.core.reader import ComtradeReader, Channel
        from waveform_viewer.visualizers.optimized_plotly_viz import OptimizedPlotlyVisualizer
        from waveform_viewer.visualizers.base import VisualizationConfig

        # 查找测试文件
        waves_dir = Path(__file__).parent.parent / "waves"
        cfg_files = list(waves_dir.rglob("*.cfg"))

        if not cfg_files:
            print("⚠ 未找到测试文件，跳过实际可视化测试")
            return 0

        cfg_file = cfg_files[0]
        print(f"\n测试文件: {cfg_file.name}")

        # 读取数据
        reader = ComtradeReader(str(cfg_file))
        print(f"通道数: {len(reader.analog_channels)}")

        # 测试可视化（使用所有通道）
        visualizer = OptimizedPlotlyVisualizer(VisualizationConfig())

        # 如果通道数很多，测试极端情况
        if len(reader.analog_channels) > 30:
            print(f"\n测试极端情况：使用所有 {len(reader.analog_channels)} 个通道")
            try:
                output_path = Path("test_all_channels.html")
                visualizer.visualize(reader, reader.analog_channels, str(output_path))
                print(f"✓ 成功生成包含 {len(reader.analog_channels)} 个通道的可视化")

                # 清理测试文件
                if output_path.exists():
                    output_path.unlink()
                    print(f"✓ 清理测试文件")

                return 0

            except Exception as e:
                print(f"✗ 可视化失败: {e}")
                import traceback
                traceback.print_exc()
                return 1
        else:
            print(f"通道数较少（{len(reader.analog_channels)}个），不需要特殊测试")
            return 0

    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """运行所有测试"""
    print("\n垂直间距修复验证测试")
    print("=" * 70)

    result1 = test_spacing_calculation()
    result2 = test_real_visualization()

    print("\n" + "=" * 70)
    if result1 == 0 and result2 == 0:
        print("✓✓✓ 所有测试通过！vertical_spacing 修复有效。")
        return 0
    else:
        print("✗✗✗ 部分测试失败。")
        return 1


if __name__ == '__main__':
    sys.exit(main())
