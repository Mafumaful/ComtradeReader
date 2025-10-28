#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例插件

演示如何创建自定义插件
"""

import csv
import json
from typing import Dict, Any
from .base import Plugin, DataProcessorPlugin, ExportPlugin
from ..core.reader import ComtradeReader


class StatisticsPlugin(Plugin):
    """
    统计信息插件

    计算并显示波形数据的统计信息
    """

    def __init__(self):
        super().__init__()
        self.description = "计算波形数据的统计信息（最小值、最大值、平均值等）"

    def execute(self, reader: ComtradeReader, context: Dict[str, Any]) -> Dict:
        """计算统计信息"""
        import numpy as np

        stats = {}

        for idx, channel in enumerate(reader.analog_channels):
            if idx >= len(reader.analog_data):
                continue

            data = np.array(reader.analog_data[idx])

            stats[channel.name] = {
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'unit': channel.unit
            }

        return stats


class CSVExportPlugin(ExportPlugin):
    """
    CSV导出插件

    将波形数据导出为CSV格式
    """

    def __init__(self):
        super().__init__()
        self.description = "将波形数据导出为CSV格式"

    def execute(self, reader: ComtradeReader, context: Dict[str, Any]) -> str:
        """执行导出"""
        output_path = context.get('output_path', 'output.csv')
        return self.export(reader, output_path)

    def export(self, reader: ComtradeReader, output_path: str, **kwargs) -> str:
        """导出为CSV"""
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)

            # 写入表头
            header = ['时间(秒)'] + [ch.name for ch in reader.analog_channels]
            writer.writerow(header)

            # 写入数据
            for i, time_val in enumerate(reader.time_values):
                row = [time_val]
                for ch_data in reader.analog_data:
                    if i < len(ch_data):
                        row.append(ch_data[i])
                    else:
                        row.append('')
                writer.writerow(row)

        return output_path

    def get_file_extension(self) -> str:
        return '.csv'


class JSONExportPlugin(ExportPlugin):
    """
    JSON导出插件

    将波形数据导出为JSON格式
    """

    def __init__(self):
        super().__init__()
        self.description = "将波形数据导出为JSON格式"

    def execute(self, reader: ComtradeReader, context: Dict[str, Any]) -> str:
        """执行导出"""
        output_path = context.get('output_path', 'output.json')
        return self.export(reader, output_path)

    def export(self, reader: ComtradeReader, output_path: str, **kwargs) -> str:
        """导出为JSON"""
        data = {
            'metadata': {
                'station_name': reader.station_name,
                'sample_rate': reader.sample_rate,
                'num_samples': reader.num_samples,
                'frequency': reader.frequency,
            },
            'channels': [],
            'time': reader.time_values,
        }

        for idx, channel in enumerate(reader.analog_channels):
            channel_data = {
                'index': channel.index,
                'name': channel.name,
                'unit': channel.unit,
                'data': reader.analog_data[idx] if idx < len(reader.analog_data) else []
            }
            data['channels'].append(channel_data)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return output_path

    def get_file_extension(self) -> str:
        return '.json'


class DataFilterPlugin(DataProcessorPlugin):
    """
    数据滤波插件

    对波形数据进行简单的移动平均滤波
    """

    def __init__(self):
        super().__init__()
        self.description = "对波形数据进行移动平均滤波"

    def execute(self, reader: ComtradeReader, context: Dict[str, Any]) -> ComtradeReader:
        """执行滤波"""
        window_size = context.get('window_size', 5)

        for idx in range(len(reader.analog_data)):
            time_vals, data_vals = reader.time_values, reader.analog_data[idx]
            filtered_time, filtered_data = self.process_data(
                time_vals, data_vals, window_size=window_size
            )
            reader.analog_data[idx] = filtered_data

        return reader

    def process_data(self, time_values, data_values, **kwargs):
        """移动平均滤波"""
        import numpy as np

        window_size = kwargs.get('window_size', 5)
        data_array = np.array(data_values)

        # 移动平均
        filtered = np.convolve(data_array, np.ones(window_size)/window_size, mode='same')

        return time_values, filtered.tolist()
