#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HDR可视化器实现

基于Plotly创建HDR文件数据的交互式图表
"""

from typing import List
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

from .base import VisualizationConfig
from ..core.hdr_reader import HdrReader
from ..utils.file_opener import open_file_in_browser


class HdrVisualizer:
    """
    HDR可视化器

    使用Plotly创建HDR文件数据的交互式HTML图表
    """

    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()

    def visualize(self,
                  hdr_reader: HdrReader,
                  output_path: str) -> str:
        """
        创建HDR数据的Plotly交互式图表

        Args:
            hdr_reader: HDR读取器
            output_path: 输出文件路径

        Returns:
            生成的HTML文件路径
        """
        # 计算需要的子图数量
        num_plots = 0
        has_trip_info = len(hdr_reader.trip_infos) > 0
        has_digital_events = len(hdr_reader.digital_events) > 0
        has_active_statuses = len(hdr_reader.get_active_digital_statuses()) > 0
        has_settings = len(hdr_reader.setting_values) > 0

        subplot_titles = []

        if has_trip_info:
            num_plots += 1
            subplot_titles.append("跳闸信息 (Trip Info)")

        if has_digital_events:
            num_plots += 1
            subplot_titles.append("数字事件 (Digital Events)")

        if has_active_statuses:
            num_plots += 1
            subplot_titles.append("激活的数字状态 (Active Digital Status)")

        if has_settings:
            num_plots += 1
            subplot_titles.append("关键设置值 (Key Setting Values)")

        if num_plots == 0:
            print("  警告: HDR文件中没有可视化的数据")
            return ""

        # 动态计算vertical_spacing
        if num_plots > 1:
            max_spacing = 2.0 / (num_plots - 1)
            vertical_spacing = min(0.05, max_spacing)
        else:
            vertical_spacing = 0.05

        # 创建子图
        fig = make_subplots(
            rows=num_plots,
            cols=1,
            subplot_titles=subplot_titles,
            vertical_spacing=vertical_spacing,
            specs=[[{"type": "scatter"}] if i < num_plots - (1 if has_settings else 0)
                   else [{"type": "table"}] for i in range(num_plots)]
        )

        current_row = 1

        # 1. 绘制跳闸信息时间线
        if has_trip_info:
            trip_times = []
            trip_names = []
            trip_values = []

            for trip in hdr_reader.trip_infos:
                # 将时间字符串转换为可显示的格式
                trip_times.append(trip.time)
                label = f"{trip.name}"
                if trip.phase:
                    label += f" ({trip.phase})"
                trip_names.append(label)
                trip_values.append(trip.value)

            # 使用散点图显示跳闸事件
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(hdr_reader.trip_infos))),
                    y=trip_values,
                    mode='markers+text',
                    marker=dict(size=12, color='red'),
                    text=trip_names,
                    textposition='top center',
                    name='跳闸事件',
                    hovertext=[f"时间: {t}<br>名称: {n}<br>值: {v}"
                              for t, n, v in zip(trip_times, trip_names, trip_values)],
                    hoverinfo='text'
                ),
                row=current_row,
                col=1
            )

            fig.update_xaxes(title_text="事件序号", row=current_row, col=1)
            fig.update_yaxes(title_text="状态值", row=current_row, col=1)
            current_row += 1

        # 2. 绘制数字事件时间线
        if has_digital_events:
            event_times = [event.time for event in hdr_reader.digital_events]
            event_names = [event.name for event in hdr_reader.digital_events]
            event_values = [event.value for event in hdr_reader.digital_events]

            fig.add_trace(
                go.Scatter(
                    x=event_times,
                    y=event_values,
                    mode='markers+lines',
                    marker=dict(size=10, color='blue'),
                    name='数字事件',
                    hovertext=[f"时间: {t}ms<br>名称: {n}<br>值: {v}"
                              for t, n, v in zip(event_times, event_names, event_values)],
                    hoverinfo='text'
                ),
                row=current_row,
                col=1
            )

            fig.update_xaxes(title_text="时间 (ms)", row=current_row, col=1)
            fig.update_yaxes(title_text="事件值", row=current_row, col=1)
            current_row += 1

        # 3. 绘制激活的数字状态（条形图）
        if has_active_statuses:
            active_statuses = hdr_reader.get_active_digital_statuses()
            status_names = [status.name for status in active_statuses[:20]]  # 限制显示前20个
            status_values = [status.value for status in active_statuses[:20]]

            fig.add_trace(
                go.Bar(
                    x=status_values,
                    y=status_names,
                    orientation='h',
                    marker=dict(color='green'),
                    name='激活状态',
                    hovertext=[f"名称: {n}<br>值: {v}"
                              for n, v in zip(status_names, status_values)],
                    hoverinfo='text'
                ),
                row=current_row,
                col=1
            )

            fig.update_xaxes(title_text="状态值", row=current_row, col=1)
            fig.update_yaxes(title_text="状态名称", row=current_row, col=1)
            current_row += 1

        # 4. 显示关键设置值（表格）
        if has_settings:
            # 选择前50个设置值显示
            settings_to_show = hdr_reader.setting_values[:50]

            header_values = ["设置名称", "值", "单位"]
            cell_values = [
                [s.name for s in settings_to_show],
                [s.value for s in settings_to_show],
                [s.unit for s in settings_to_show]
            ]

            fig.add_trace(
                go.Table(
                    header=dict(
                        values=header_values,
                        fill_color='paleturquoise',
                        align='left',
                        font=dict(size=12)
                    ),
                    cells=dict(
                        values=cell_values,
                        fill_color='lavender',
                        align='left',
                        font=dict(size=10)
                    )
                ),
                row=current_row,
                col=1
            )
            current_row += 1

        # 更新布局
        title = self.config.title or f"HDR故障报告可视化"
        if hdr_reader.fault_start_time:
            title += f" - {hdr_reader.fault_start_time}"

        fig.update_layout(
            height=self.config.height_per_plot * num_plots,
            width=self.config.width,
            title_text=title,
            showlegend=False,
            font=dict(size=10)
        )

        # 保存为HTML文件
        fig.write_html(str(output_path))
        print(f"  已保存HDR可视化文件: {Path(output_path).name}")

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
