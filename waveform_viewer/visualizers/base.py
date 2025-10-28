#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化器基类模块 (Strategy Pattern)

定义可视化策略的基类接口
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional
from ..core.reader import ComtradeReader, Channel


class VisualizationConfig:
    """可视化配置类"""

    def __init__(self,
                 output_dir: Optional[str] = None,
                 width: int = 1200,
                 height_per_plot: int = 250,
                 title: Optional[str] = None,
                 show_legend: bool = False,
                 auto_open: bool = True):
        self.output_dir = output_dir or '.'
        self.width = width
        self.height_per_plot = height_per_plot
        self.title = title
        self.show_legend = show_legend
        self.auto_open = auto_open  # 是否自动打开生成的文件


class BaseVisualizer(ABC):
    """
    可视化器基类

    定义可视化策略的接口，所有具体的可视化器都需要实现这个接口
    """

    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        初始化可视化器

        Args:
            config: 可视化配置
        """
        self.config = config or VisualizationConfig()

    @abstractmethod
    def visualize(self,
                  reader: ComtradeReader,
                  channels: List[Channel],
                  output_path: str) -> str:
        """
        创建可视化图表

        Args:
            reader: COMTRADE读取器
            channels: 要可视化的通道列表
            output_path: 输出文件路径

        Returns:
            生成的文件路径
        """
        pass

    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """
        获取支持的输出格式

        Returns:
            支持的格式列表，如 ['html', 'png', 'pdf']
        """
        pass

    def _validate_channels(self, channels: List[Channel]) -> bool:
        """
        验证通道列表

        Args:
            channels: 通道列表

        Returns:
            是否有效
        """
        return channels is not None and len(channels) > 0
