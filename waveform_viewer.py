#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMTRADE波形文件可视化工具
使用plotly创建交互式波形图表
"""

import os
import struct
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path


class ComtradeReader:
    """COMTRADE文件读取器"""

    def __init__(self, cfg_file):
        """
        初始化COMTRADE读取器

        Args:
            cfg_file: .cfg配置文件路径
        """
        self.cfg_file = cfg_file
        self.base_name = cfg_file.rsplit('.', 1)[0]
        self.dat_file = self.base_name + '.dat'

        self.analog_channels = []
        self.digital_channels = []
        self.analog_data = []
        self.digital_data = []
        self.time_values = []
        self.sample_rate = 0
        self.num_samples = 0

        self._parse_cfg()
        self._read_dat()

    def _parse_cfg(self):
        """解析.cfg配置文件"""
        # COMTRADE配置文件通常使用GB2312/GBK编码（中国电力系统）
        # 按优先级尝试不同编码
        encodings = ['gbk', 'gb2312', 'gb18030', 'utf-8']
        lines = []

        for encoding in encodings:
            try:
                with open(self.cfg_file, 'r', encoding=encoding) as f:
                    lines = [line.strip() for line in f.readlines()]
                # 如果成功读取，跳出循环
                break
            except (UnicodeDecodeError, LookupError):
                continue

        if not lines:
            # 如果所有编码都失败，使用gbk并忽略错误
            with open(self.cfg_file, 'r', encoding='gbk', errors='replace') as f:
                lines = [line.strip() for line in f.readlines()]

        # 第0行：站点名称等信息
        self.station_name = lines[0] if lines else ""

        # 第1行：通道数量
        if len(lines) > 1:
            parts = lines[1].split(',')
            total_channels = int(parts[0])
            num_analog = int(parts[1].rstrip('A'))
            num_digital = int(parts[2].rstrip('D'))
        else:
            return

        # 解析模拟量通道
        line_idx = 2
        for i in range(num_analog):
            if line_idx >= len(lines):
                break
            parts = lines[line_idx].split(',')
            if len(parts) >= 6:
                channel = {
                    'index': int(parts[0]),
                    'name': parts[1].strip(),
                    'phase': parts[2].strip() if len(parts) > 2 else '',
                    'unit': parts[4].strip() if len(parts) > 4 else '',
                    'a': float(parts[5]) if len(parts) > 5 and parts[5] else 1.0,
                    'b': float(parts[6]) if len(parts) > 6 and parts[6] else 0.0,
                }
                self.analog_channels.append(channel)
            line_idx += 1

        # 解析数字量通道
        for i in range(num_digital):
            if line_idx >= len(lines):
                break
            parts = lines[line_idx].split(',')
            if len(parts) >= 2:
                channel = {
                    'index': int(parts[0]),
                    'name': parts[1].strip(),
                }
                self.digital_channels.append(channel)
            line_idx += 1

        # 查找采样率信息
        # 找到纯数字行（频率）
        for i in range(line_idx, len(lines)):
            line = lines[i].strip()
            try:
                self.frequency = float(line)
                line_idx = i + 1
                break
            except ValueError:
                continue

        # 下一行应该是采样率组数
        if line_idx < len(lines):
            line_idx += 1  # 跳过组数

        # 采样点数和采样率
        if line_idx < len(lines):
            parts = lines[line_idx].split(',')
            if len(parts) >= 1:
                self.sample_rate = int(parts[0])  # 采样率 (Hz)
                if len(parts) >= 2:
                    self.num_samples = int(parts[1])  # 样本数

        print(f"  配置信息: {num_analog}个模拟通道, {num_digital}个数字通道")
        print(f"  采样率: {self.sample_rate} Hz, 样本数: {self.num_samples}")

    def _read_dat(self):
        """读取.dat二进制数据文件"""
        with open(self.dat_file, 'rb') as f:
            data = f.read()

        # 计算每个采样点的字节数
        analog_bytes = len(self.analog_channels) * 2  # 每个模拟量2字节
        digital_bytes = (len(self.digital_channels) + 7) // 8  # 数字量按字节对齐

        # 每个样本：4字节样本号 + 4字节时间戳 + 模拟量 + 数字量
        # 可能有填充字节，使用实际文件大小计算
        calculated_bytes = 8 + analog_bytes + digital_bytes
        actual_bytes = len(data) // self.num_samples if self.num_samples > 0 else calculated_bytes

        bytes_per_sample = actual_bytes
        print(f"  每样本字节数: {bytes_per_sample} (计算值: {calculated_bytes}, 模拟: {analog_bytes}, 数字: {digital_bytes})")

        # 初始化数据数组
        self.analog_data = [[] for _ in range(len(self.analog_channels))]
        self.time_values = []

        # 解析每个样本
        offset = 0
        samples_read = 0

        while offset + bytes_per_sample <= len(data) and samples_read < self.num_samples:
            try:
                # 读取样本号和时间戳
                sample_num = struct.unpack('<I', data[offset:offset+4])[0]
                timestamp_us = struct.unpack('<I', data[offset+4:offset+8])[0]

                # 时间戳单位是微秒，转换为秒
                self.time_values.append(timestamp_us / 1000000.0)

                offset += 8

                # 读取模拟量数据
                for ch_idx in range(len(self.analog_channels)):
                    if offset + 2 > len(data):
                        break
                    raw_value = struct.unpack('<h', data[offset:offset+2])[0]

                    # 应用转换公式：实际值 = a * 原始值 + b
                    channel = self.analog_channels[ch_idx]
                    actual_value = channel['a'] * raw_value + channel['b']
                    self.analog_data[ch_idx].append(actual_value)
                    offset += 2

                # 跳过数字量和填充字节
                padding = bytes_per_sample - 8 - analog_bytes
                offset += padding

                samples_read += 1

            except Exception as e:
                print(f"  警告: 读取样本 {samples_read} 时出错: {e}")
                break

        print(f"  成功读取 {samples_read} 个样本")
        if samples_read > 0:
            print(f"  时间范围: {self.time_values[0]:.6f}s ~ {self.time_values[-1]:.6f}s")

    def get_channel_names(self):
        """获取所有模拟量通道名称"""
        return [(ch['index'], ch['name'], ch['unit']) for ch in self.analog_channels]

    def get_analog_data(self, channel_index):
        """获取指定通道的模拟量数据"""
        for idx, ch in enumerate(self.analog_channels):
            if ch['index'] == channel_index:
                return self.time_values, self.analog_data[idx]
        return None, None


def visualize_waveforms(comtrade_folder):
    """
    可视化COMTRADE波形文件

    Args:
        comtrade_folder: 包含COMTRADE文件的文件夹路径
    """
    # 查找所有.cfg文件
    cfg_files = list(Path(comtrade_folder).glob('*.cfg'))

    if not cfg_files:
        print(f"在 {comtrade_folder} 中未找到.cfg文件")
        return

    print(f"找到 {len(cfg_files)} 个COMTRADE文件组\n")

    # 为每个文件创建可视化
    for cfg_file in cfg_files:
        print(f"处理文件: {cfg_file.name}")

        try:
            # 读取COMTRADE数据
            reader = ComtradeReader(str(cfg_file))

            # 获取通道列表
            channels = reader.get_channel_names()

            if not channels:
                print("  未找到模拟量通道")
                continue

            # 显示所有可用通道
            print(f"\n  所有通道列表:")
            for idx, (ch_idx, ch_name, ch_unit) in enumerate(channels[:20]):  # 只显示前20个
                print(f"    {ch_idx}. {ch_name} ({ch_unit})")
            if len(channels) > 20:
                print(f"    ... 还有 {len(channels) - 20} 个通道")

            # 选择要可视化的通道
            # 优先选择电力系统中常见的重要参数
            important_patterns = [
                ('电压', '有功', '无功', '电流'),  # 组1：电气量
                ('频率', '角度', '功率'),  # 组2：系统参数
                ('AVR', 'PSS'),  # 组3：控制器
                ('励磁',),  # 组4：励磁系统
            ]

            selected_channels = []

            # 按优先级选择通道
            for patterns in important_patterns:
                for ch_idx, ch_name, ch_unit in channels:
                    if any(pattern in ch_name for pattern in patterns):
                        if (ch_idx, ch_name, ch_unit) not in selected_channels:
                            selected_channels.append((ch_idx, ch_name, ch_unit))

            # 如果选中的通道太少，补充前面的通道
            if len(selected_channels) < 8:
                for ch in channels:
                    if ch not in selected_channels:
                        selected_channels.append(ch)
                    if len(selected_channels) >= 12:
                        break

            # 限制最多显示12个通道
            selected_channels = selected_channels[:12]

            print(f"\n  选择以下 {len(selected_channels)} 个通道进行可视化:")
            for ch_idx, ch_name, ch_unit in selected_channels:
                print(f"    {ch_idx}. {ch_name} ({ch_unit})")

            # 创建子图
            num_plots = len(selected_channels)

            # 动态计算vertical_spacing，避免超出Plotly的限制
            # 最大允许值为 1 / (rows - 1)，我们使用 0.8 倍以留出余地
            if num_plots > 1:
                max_spacing = 0.8 / (num_plots - 1)
                vertical_spacing = min(0.03, max_spacing)
            else:
                vertical_spacing = 0.03

            fig = make_subplots(
                rows=num_plots,
                cols=1,
                subplot_titles=[f"{name} ({unit})" if unit else f"{name}"
                               for _, name, unit in selected_channels],
                vertical_spacing=vertical_spacing,
                shared_xaxes=True
            )

            # 为每个选中的通道添加数据
            for plot_idx, (ch_idx, ch_name, ch_unit) in enumerate(selected_channels, 1):
                time_vals, data_vals = reader.get_analog_data(ch_idx)

                if time_vals and data_vals and len(time_vals) > 0:
                    # 使用散点图模式显示
                    fig.add_trace(
                        go.Scatter(
                            x=time_vals,
                            y=data_vals,
                            mode='markers',  # 改用散点模式
                            name=ch_name,
                            marker=dict(
                                size=3,  # 点的大小
                                opacity=0.7,  # 透明度
                            ),
                            showlegend=False
                        ),
                        row=plot_idx,
                        col=1
                    )

                    # 设置y轴标签
                    y_title = f"{ch_unit}" if ch_unit else "值"
                    fig.update_yaxes(title_text=y_title, row=plot_idx, col=1)

            # 更新布局
            fig.update_layout(
                height=250 * num_plots,  # 每个子图250像素高
                title_text=f"波形数据可视化: {cfg_file.stem}",
                hovermode='x unified',
                showlegend=False,
                font=dict(size=10)
            )

            # 更新x轴标签（只在最后一个子图显示）
            fig.update_xaxes(title_text="时间 (秒)", row=num_plots, col=1)

            # 保存为HTML文件
            output_file = cfg_file.stem + '_visualization.html'
            output_path = Path(comtrade_folder).parent / output_file
            fig.write_html(str(output_path))
            print(f"\n  ✓ 可视化结果已保存到: {output_path.name}\n")
            print("=" * 70)

        except Exception as e:
            print(f"\n  ✗ 处理 {cfg_file.name} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            print("=" * 70)


def main():
    """主函数"""
    # 获取当前脚本所在目录
    current_dir = Path(__file__).parent

    # 查找波形文件夹（排除隐藏文件夹和虚拟环境）
    exclude_dirs = {'venv', '.venv', 'env', '.env', '__pycache__', '.git', '.claude'}
    waveform_folders = [
        d for d in current_dir.iterdir()
        if d.is_dir() and d.name not in exclude_dirs and not d.name.startswith('.')
    ]

    if not waveform_folders:
        print("未找到波形数据文件夹")
        return

    print("=" * 70)
    print("COMTRADE波形可视化工具")
    print("=" * 70)
    print(f"\n找到以下波形数据文件夹:")
    for idx, folder in enumerate(waveform_folders, 1):
        print(f"  {idx}. {folder.name}")
    print()

    # 处理所有文件夹
    for folder in waveform_folders:
        print("=" * 70)
        print(f"处理文件夹: {folder.name}")
        print("=" * 70)
        visualize_waveforms(str(folder))

    print("\n" + "=" * 70)
    print("所有波形文件处理完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()
