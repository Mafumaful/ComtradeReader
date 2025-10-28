#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMTRADE文件读取器核心模块
"""

import struct
from pathlib import Path
from typing import List, Tuple, Dict, Optional


class Channel:
    """通道数据类"""

    def __init__(self, index: int, name: str, phase: str = '',
                 unit: str = '', a: float = 1.0, b: float = 0.0):
        self.index = index
        self.name = name
        self.phase = phase
        self.unit = unit
        self.a = a  # 转换系数
        self.b = b  # 转换偏移

    def __repr__(self):
        return f"Channel({self.index}, {self.name}, {self.unit})"


class ComtradeReader:
    """
    COMTRADE文件读取器

    负责读取和解析COMTRADE格式的波形文件
    """

    def __init__(self, cfg_file: str):
        """
        初始化COMTRADE读取器

        Args:
            cfg_file: .cfg配置文件路径
        """
        self.cfg_file = cfg_file
        self.base_name = cfg_file.rsplit('.', 1)[0]
        self.dat_file = self.base_name + '.dat'

        self.station_name = ""
        self.frequency = 50.0
        self.analog_channels: List[Channel] = []
        self.digital_channels: List[Dict] = []
        self.analog_data: List[List[float]] = []
        self.digital_data: List[List[int]] = []
        self.time_values: List[float] = []
        self.sample_rate = 0
        self.num_samples = 0

        self._parse_cfg()
        self._read_dat()

    def _parse_cfg(self):
        """解析.cfg配置文件"""
        encodings = ['gbk', 'gb2312', 'gb18030', 'utf-8']
        lines = []

        for encoding in encodings:
            try:
                with open(self.cfg_file, 'r', encoding=encoding) as f:
                    lines = [line.strip() for line in f.readlines()]
                break
            except (UnicodeDecodeError, LookupError):
                continue

        if not lines:
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
                channel = Channel(
                    index=int(parts[0]),
                    name=parts[1].strip(),
                    phase=parts[2].strip() if len(parts) > 2 else '',
                    unit=parts[4].strip() if len(parts) > 4 else '',
                    a=float(parts[5]) if len(parts) > 5 and parts[5] else 1.0,
                    b=float(parts[6]) if len(parts) > 6 and parts[6] else 0.0,
                )
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

        # 查找频率信息
        for i in range(line_idx, len(lines)):
            line = lines[i].strip()
            try:
                self.frequency = float(line)
                line_idx = i + 1
                break
            except ValueError:
                continue

        # 跳过采样率组数
        if line_idx < len(lines):
            line_idx += 1

        # 采样点数和采样率
        if line_idx < len(lines):
            parts = lines[line_idx].split(',')
            if len(parts) >= 1:
                self.sample_rate = int(parts[0])
                if len(parts) >= 2:
                    self.num_samples = int(parts[1])

        print(f"  配置信息: {num_analog}个模拟通道, {num_digital}个数字通道")
        print(f"  采样率: {self.sample_rate} Hz, 样本数: {self.num_samples}")

    def _read_dat(self):
        """读取.dat二进制数据文件"""
        with open(self.dat_file, 'rb') as f:
            data = f.read()

        # 计算每个采样点的字节数
        analog_bytes = len(self.analog_channels) * 2
        digital_bytes = (len(self.digital_channels) + 7) // 8

        calculated_bytes = 8 + analog_bytes + digital_bytes
        actual_bytes = len(data) // self.num_samples if self.num_samples > 0 else calculated_bytes

        bytes_per_sample = actual_bytes
        print(f"  每样本字节数: {bytes_per_sample}")

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

                self.time_values.append(timestamp_us / 1000000.0)
                offset += 8

                # 读取模拟量数据
                for ch_idx in range(len(self.analog_channels)):
                    if offset + 2 > len(data):
                        break
                    raw_value = struct.unpack('<h', data[offset:offset+2])[0]

                    channel = self.analog_channels[ch_idx]
                    actual_value = channel.a * raw_value + channel.b
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

    def get_channel_names(self) -> List[Tuple[int, str, str]]:
        """获取所有模拟量通道名称"""
        return [(ch.index, ch.name, ch.unit) for ch in self.analog_channels]

    def get_analog_data(self, channel_index: int) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        """获取指定通道的模拟量数据"""
        for idx, ch in enumerate(self.analog_channels):
            if ch.index == channel_index:
                return self.time_values, self.analog_data[idx]
        return None, None

    def get_channel_by_name(self, name_pattern: str) -> Optional[Channel]:
        """根据名称模式查找通道"""
        for ch in self.analog_channels:
            if name_pattern in ch.name:
                return ch
        return None
