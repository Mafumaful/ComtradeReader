#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的Plotly可视化器

针对大量通道和数据点进行性能优化
"""

from typing import List
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import numpy as np

from .base import BaseVisualizer, VisualizationConfig
from ..core.reader import ComtradeReader, Channel
from ..utils.file_opener import open_file_in_browser


class OptimizedPlotlyVisualizer(BaseVisualizer):
    """
    优化的Plotly可视化器

    主要优化：
    1. 使用 Scattergl (WebGL渲染) 代替 Scatter，大幅提升性能
    2. 数据降采样，减少数据点
    3. 优化布局，减小空白间隙
    4. 精简UI元素
    """

    def __init__(self, config: VisualizationConfig = None):
        super().__init__(config)
        self.use_webgl = True  # 启用WebGL渲染
        self.downsample = True  # 启用数据降采样

    def visualize(self,
                  reader: ComtradeReader,
                  channels: List[Channel],
                  output_path: str) -> str:
        """创建优化的可视化"""
        if not self._validate_channels(channels):
            raise ValueError("通道列表为空或无效")

        num_plots = len(channels)

        # 动态计算vertical_spacing - 进一步优化
        if num_plots > 1:
            # 对于大量通道，使用更激进的间距
            if num_plots > 20:
                max_spacing = 0.5 / (num_plots - 1)  # 更小的间距
            else:
                max_spacing = 0.8 / (num_plots - 1)
            vertical_spacing = min(0.02, max_spacing)  # 默认值从0.03降到0.02
        else:
            vertical_spacing = 0.02

        # 创建子图 - 优化布局
        fig = make_subplots(
            rows=num_plots,
            cols=1,
            subplot_titles=[f"{ch.name} ({ch.unit})" if ch.unit else f"{ch.name}"
                           for ch in channels],
            vertical_spacing=vertical_spacing,
            shared_xaxes=True
        )

        # 为每个通道添加数据
        for plot_idx, channel in enumerate(channels, 1):
            time_vals, data_vals = reader.get_analog_data(channel.index)

            if time_vals and data_vals and len(time_vals) > 0:
                # 数据降采样（如果启用且数据点很多）
                if self.downsample and len(time_vals) > 5000:
                    time_vals, data_vals = self._downsample_data(
                        time_vals, data_vals, max_points=5000
                    )

                # 使用 Scattergl (WebGL) 代替 Scatter，性能大幅提升
                trace_type = go.Scattergl if self.use_webgl else go.Scatter

                fig.add_trace(
                    trace_type(
                        x=time_vals,
                        y=data_vals,
                        mode='lines',  # 对于密集数据，线图比散点图性能更好
                        name=channel.name,
                        line=dict(width=0.8),  # 更细的线
                        showlegend=False,
                        hovertemplate='%{y:.4f}<extra></extra>'  # 简化悬停信息
                    ),
                    row=plot_idx,
                    col=1
                )

                # 设置y轴标签 - 更小的字体
                y_title = f"{channel.unit}" if channel.unit else ""
                fig.update_yaxes(
                    title_text=y_title,
                    title_font=dict(size=8),  # 更小的字体
                    tickfont=dict(size=7),
                    row=plot_idx,
                    col=1
                )

        # 优化布局 - 减小空白
        title = self.config.title or f"波形数据: {Path(reader.cfg_file).stem}"

        # 动态调整每个子图的高度
        if num_plots > 30:
            height_per_plot = 150  # 大量通道时减小高度
        elif num_plots > 20:
            height_per_plot = 180
        else:
            height_per_plot = self.config.height_per_plot

        fig.update_layout(
            height=height_per_plot * num_plots,
            width=self.config.width,
            title_text=title,
            title_font=dict(size=14),
            hovermode='x unified',
            showlegend=False,
            font=dict(size=8),  # 全局更小的字体
            margin=dict(l=60, r=30, t=50, b=40),  # 优化边距
        )

        # 更新x轴 - 只在最后一个子图显示标签
        for i in range(1, num_plots + 1):
            if i == num_plots:
                # 最后一个子图显示x轴标签
                fig.update_xaxes(
                    title_text="时间 (秒)",
                    title_font=dict(size=9),
                    tickfont=dict(size=7),
                    row=i,
                    col=1
                )
            else:
                # 其他子图隐藏x轴标签以节省空间
                fig.update_xaxes(
                    tickfont=dict(size=7),
                    showticklabels=True,  # 保留刻度但字体更小
                    row=i,
                    col=1
                )

        # 优化子图标题
        for annotation in fig['layout']['annotations']:
            annotation['font'] = dict(size=9)  # 更小的标题字体
            annotation['yshift'] = -5  # 减小标题与图的距离

        # 保存为HTML - 使用优化配置
        config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': Path(reader.cfg_file).stem,
                'height': height_per_plot * num_plots,
                'width': self.config.width,
                'scale': 1
            }
        }

        fig.write_html(
            str(output_path),
            config=config,
            include_plotlyjs='cdn'  # 使用CDN加载Plotly.js，减小文件大小
        )

        # 自动打开文件
        if self.config.auto_open:
            abs_path = Path(output_path).resolve()
            print(f"  正在打开文件: {abs_path.name}")

            if open_file_in_browser(str(abs_path)):
                print(f"  ✓ 已在浏览器中打开")
            else:
                print(f"  提示: 无法自动打开文件")
                print(f"  请手动打开: {abs_path}")

        return str(output_path)

    def _downsample_data(self, time_vals, data_vals, max_points=5000):
        """
        数据降采样，保持关键特征

        使用快速均匀采样 + 关键点保留
        """
        if len(time_vals) <= max_points:
            return time_vals, data_vals

        # 转换为numpy数组以提高性能
        time_array = np.array(time_vals)
        data_array = np.array(data_vals)

        # 计算采样间隔
        interval = len(time_vals) // max_points

        # 快速均匀降采样
        indices = np.arange(0, len(time_vals), interval)

        # 始终包含最后一个点
        if indices[-1] != len(time_vals) - 1:
            indices = np.append(indices, len(time_vals) - 1)

        # 取样
        downsampled_time = time_array[indices].tolist()
        downsampled_data = data_array[indices].tolist()

        reduction_ratio = (1 - len(indices) / len(time_vals)) * 100

        return downsampled_time, downsampled_data

    def get_supported_formats(self) -> List[str]:
        return ['html']


class FastPlotlyVisualizer(OptimizedPlotlyVisualizer):
    """
    快速可视化器

    牺牲一些细节，最大化性能
    """

    def __init__(self, config: VisualizationConfig = None):
        super().__init__(config)
        self.use_webgl = True
        self.downsample = True

    def visualize(self,
                  reader: ComtradeReader,
                  channels: List[Channel],
                  output_path: str) -> str:
        """超快速可视化 - 更激进的优化"""
        if not self._validate_channels(channels):
            raise ValueError("通道列表为空或无效")

        num_plots = len(channels)

        # 更激进的降采样
        max_points_per_channel = 2000 if num_plots > 30 else 3000

        # 极小的间距
        if num_plots > 1:
            vertical_spacing = 0.3 / (num_plots - 1)
            vertical_spacing = min(0.01, vertical_spacing)
        else:
            vertical_spacing = 0.01

        fig = make_subplots(
            rows=num_plots,
            cols=1,
            subplot_titles=[f"{ch.name}" for ch in channels],  # 简化标题
            vertical_spacing=vertical_spacing,
            shared_xaxes=True
        )

        for plot_idx, channel in enumerate(channels, 1):
            time_vals, data_vals = reader.get_analog_data(channel.index)

            if time_vals and data_vals and len(time_vals) > 0:
                # 激进降采样
                if len(time_vals) > max_points_per_channel:
                    time_vals, data_vals = self._downsample_data(
                        time_vals, data_vals, max_points=max_points_per_channel
                    )

                fig.add_trace(
                    go.Scattergl(
                        x=time_vals,
                        y=data_vals,
                        mode='lines',
                        name=channel.name,
                        line=dict(width=0.5),  # 极细的线
                        showlegend=False,
                        hovertemplate='%{y:.3f}<extra></extra>'
                    ),
                    row=plot_idx,
                    col=1
                )

                fig.update_yaxes(
                    title_text=channel.unit if channel.unit else "",
                    title_font=dict(size=7),
                    tickfont=dict(size=6),
                    row=plot_idx,
                    col=1
                )

        # 极简布局
        height_per_plot = 120 if num_plots > 30 else 150

        fig.update_layout(
            height=height_per_plot * num_plots,
            width=self.config.width,
            title_text=f"{Path(reader.cfg_file).stem}",
            title_font=dict(size=12),
            hovermode='closest',  # 更快的悬停模式
            showlegend=False,
            font=dict(size=7),
            margin=dict(l=50, r=20, t=40, b=30),
        )

        fig.update_xaxes(title_text="时间 (s)", title_font=dict(size=8), row=num_plots, col=1)

        # 精简注释
        for annotation in fig['layout']['annotations']:
            annotation['font'] = dict(size=8)
            annotation['yshift'] = -3

        # 保存 - 最小配置
        fig.write_html(str(output_path), include_plotlyjs='cdn')

        if self.config.auto_open:
            abs_path = Path(output_path).resolve()
            print(f"  正在打开文件: {abs_path.name}")
            if open_file_in_browser(str(abs_path)):
                print(f"  ✓ 已在浏览器中打开")

        return str(output_path)
