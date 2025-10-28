#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotly可视化器实现

基于Plotly创建交互式波形图表
"""

from typing import List
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

from .base import BaseVisualizer, VisualizationConfig
from ..core.reader import ComtradeReader, Channel
from ..utils.file_opener import open_file_in_browser


class PlotlyVisualizer(BaseVisualizer):
    """
    Plotly可视化器

    使用Plotly创建交互式HTML波形图表
    """

    def __init__(self, config: VisualizationConfig = None):
        super().__init__(config)

    def visualize(self,
                  reader: ComtradeReader,
                  channels: List[Channel],
                  output_path: str) -> str:
        """
        创建Plotly交互式图表

        Args:
            reader: COMTRADE读取器
            channels: 要可视化的通道列表
            output_path: 输出文件路径

        Returns:
            生成的HTML文件路径
        """
        if not self._validate_channels(channels):
            raise ValueError("通道列表为空或无效")

        num_plots = len(channels)

        # 动态计算vertical_spacing，避免超出Plotly的限制
        # 最大允许值为 1 / (rows - 1)，我们使用 0.8 倍以留出余地
        if num_plots > 1:
            max_spacing = 0.8 / (num_plots - 1)
            vertical_spacing = min(0.03, max_spacing)
        else:
            vertical_spacing = 0.03

        # 创建子图
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
                # 使用散点图模式显示
                fig.add_trace(
                    go.Scatter(
                        x=time_vals,
                        y=data_vals,
                        mode='markers',
                        name=channel.name,
                        marker=dict(
                            size=3,
                            opacity=0.7,
                        ),
                        showlegend=False
                    ),
                    row=plot_idx,
                    col=1
                )

                # 设置y轴标签
                y_title = f"{channel.unit}" if channel.unit else "值"
                fig.update_yaxes(title_text=y_title, row=plot_idx, col=1)

        # 更新布局
        title = self.config.title or f"波形数据可视化: {Path(reader.cfg_file).stem}"
        fig.update_layout(
            height=self.config.height_per_plot * num_plots,
            width=self.config.width,
            title_text=title,
            hovermode='x unified',
            showlegend=self.config.show_legend,
            font=dict(size=10)
        )

        # 更新x轴标签（只在最后一个子图显示）
        fig.update_xaxes(title_text="时间 (秒)", row=num_plots, col=1)

        # 保存为HTML文件
        fig.write_html(str(output_path))

        # 自动打开文件（如果配置了）
        if self.config.auto_open:
            abs_path = Path(output_path).resolve()
            print(f"  正在打开文件: {abs_path.name}")

            if open_file_in_browser(str(abs_path)):
                print(f"  ✓ 已在浏览器中打开")
            else:
                print(f"  提示: 无法自动打开文件")
                print(f"  请手动打开: {abs_path}")

        return str(output_path)

    def get_supported_formats(self) -> List[str]:
        """获取支持的输出格式"""
        return ['html']


class PlotlyLineVisualizer(PlotlyVisualizer):
    """
    Plotly线图可视化器

    使用线条而不是散点显示波形
    """

    def visualize(self,
                  reader: ComtradeReader,
                  channels: List[Channel],
                  output_path: str) -> str:
        """创建线图"""
        if not self._validate_channels(channels):
            raise ValueError("通道列表为空或无效")

        num_plots = len(channels)

        # 动态计算vertical_spacing，避免超出Plotly的限制
        # 最大允许值为 1 / (rows - 1)，我们使用 0.8 倍以留出余地
        if num_plots > 1:
            max_spacing = 0.8 / (num_plots - 1)
            vertical_spacing = min(0.03, max_spacing)
        else:
            vertical_spacing = 0.03

        # 创建子图
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
                # 使用线图模式显示
                fig.add_trace(
                    go.Scatter(
                        x=time_vals,
                        y=data_vals,
                        mode='lines',
                        name=channel.name,
                        line=dict(width=1),
                        showlegend=False
                    ),
                    row=plot_idx,
                    col=1
                )

                # 设置y轴标签
                y_title = f"{channel.unit}" if channel.unit else "值"
                fig.update_yaxes(title_text=y_title, row=plot_idx, col=1)

        # 更新布局
        title = self.config.title or f"波形数据可视化: {Path(reader.cfg_file).stem}"
        fig.update_layout(
            height=self.config.height_per_plot * num_plots,
            width=self.config.width,
            title_text=title,
            hovermode='x unified',
            showlegend=self.config.show_legend,
            font=dict(size=10)
        )

        # 更新x轴标签
        fig.update_xaxes(title_text="时间 (秒)", row=num_plots, col=1)

        # 保存为HTML文件
        fig.write_html(str(output_path))
        print(f"  已保存文件: {output_path}")

        # 自动打开文件（如果配置了）
        if self.config.auto_open:
            abs_path = Path(output_path).resolve()
            print(f"  正在打开文件: {abs_path.name}")

            if open_file_in_browser(str(abs_path)):
                print(f"  ✓ 已在浏览器中打开")
            else:
                print(f"  提示: 无法自动打开文件")
                print(f"  请手动打开: {abs_path}")

        return str(output_path)
