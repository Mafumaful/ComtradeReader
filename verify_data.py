#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMTRADE数据验证脚本
输出数据的统计信息来验证读取是否正确

v2.0 - 更新以使用新的模块化架构
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

# 支持新旧两种导入方式
try:
    from waveform_viewer.core.reader import ComtradeReader
    print("使用新版模块化架构")
except ImportError:
    from waveform_viewer import ComtradeReader
    print("使用旧版模块")

import numpy as np


def verify_comtrade_data(cfg_file):
    """验证COMTRADE数据"""
    print(f"\n{'='*70}")
    print(f"验证文件: {cfg_file}")
    print('='*70)

    reader = ComtradeReader(cfg_file)

    print(f"\n基本信息:")
    print(f"  采样率: {reader.sample_rate} Hz")
    print(f"  样本数: {reader.num_samples}")
    print(f"  实际读取: {len(reader.time_values)} 个样本")

    if len(reader.time_values) > 0:
        duration = reader.time_values[-1] - reader.time_values[0]
        print(f"  时间跨度: {duration:.6f} 秒")
        print(f"  理论时间跨度: {reader.num_samples / reader.sample_rate:.6f} 秒")

        # 计算采样间隔
        if len(reader.time_values) > 1:
            intervals = np.diff(reader.time_values)
            avg_interval = np.mean(intervals)
            print(f"  平均采样间隔: {avg_interval*1000:.3f} ms (理论: {1000/reader.sample_rate:.3f} ms)")

    print(f"\n通道数据统计 (前10个重要通道):")
    print(f"{'序号':<6} {'通道名':<25} {'单位':<8} {'最小值':<12} {'最大值':<12} {'均值':<12}")
    print('-'*85)

    # 选择一些重要通道
    important_indices = [1, 2, 3, 4, 7, 9, 10, 11, 13, 14]  # 电压、电流、功率、频率等

    for idx in important_indices[:10]:
        time_vals, data_vals = reader.get_analog_data(idx)
        if data_vals and len(data_vals) > 0:
            ch = reader.analog_channels[idx-1]
            data_array = np.array(data_vals)
            min_val = np.min(data_array)
            max_val = np.max(data_array)
            mean_val = np.mean(data_array)

            print(f"{idx:<6} {ch['name']:<25} {ch['unit']:<8} {min_val:<12.4f} {max_val:<12.4f} {mean_val:<12.4f}")

    # 检查是否有异常值
    print(f"\n数据质量检查:")
    channels_with_issues = 0
    for idx, ch in enumerate(reader.analog_channels[:20], 1):
        time_vals, data_vals = reader.get_analog_data(idx)
        if data_vals and len(data_vals) > 0:
            data_array = np.array(data_vals)

            # 检查NaN或Inf
            if np.any(np.isnan(data_array)) or np.any(np.isinf(data_array)):
                print(f"  警告: 通道 {idx} ({ch['name']}) 包含NaN或Inf值")
                channels_with_issues += 1

            # 检查是否所有值都相同（可能表示通道未连接）
            if np.std(data_array) < 1e-10:
                print(f"  提示: 通道 {idx} ({ch['name']}) 数据无变化 (std={np.std(data_array):.2e})")

    if channels_with_issues == 0:
        print(f"  ✓ 未发现明显的数据质量问题")

    # 显示一些采样点的原始数据
    print(f"\n采样点示例 (前5个时间点):")
    print(f"{'样本':<8} {'时间(s)':<12} {'通道1':<12} {'通道4(AVR)':<12} {'通道13(频率)':<12}")
    print('-'*60)
    for i in range(min(5, len(reader.time_values))):
        t = reader.time_values[i]
        v1 = reader.analog_data[0][i] if len(reader.analog_data) > 0 else 0
        v4 = reader.analog_data[3][i] if len(reader.analog_data) > 3 else 0
        v13 = reader.analog_data[12][i] if len(reader.analog_data) > 12 else 0
        print(f"{i+1:<8} {t:<12.6f} {v1:<12.4f} {v4:<12.4f} {v13:<12.4f}")


if __name__ == '__main__':
    # 验证所有cfg文件
    from pathlib import Path

    # 查找waves目录下的所有.cfg文件
    waves_dir = Path(__file__).parent / 'waves'

    if not waves_dir.exists():
        print(f"错误: waves目录不存在: {waves_dir}")
        print("请确保波形文件放在 waves/ 目录下")
        sys.exit(1)

    cfg_files = list(waves_dir.rglob('*.cfg'))

    if not cfg_files:
        print(f"未在 {waves_dir} 中找到.cfg文件")
        print("请确保波形文件（.cfg和.dat）位于waves目录的子文件夹中")
        sys.exit(1)

    print(f"找到 {len(cfg_files)} 个COMTRADE文件\n")

    for cfg_file in cfg_files:
        verify_comtrade_data(str(cfg_file))

    print(f"\n{'='*70}")
    print("验证完成！")
    print('='*70)
