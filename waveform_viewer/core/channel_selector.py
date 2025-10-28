#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通道选择策略模块 (Strategy Pattern)

提供不同的通道选择策略，用于从大量通道中选择需要可视化的通道
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
from .reader import Channel


class ChannelSelectionStrategy(ABC):
    """通道选择策略基类"""

    @abstractmethod
    def select_channels(self, channels: List[Channel],
                       max_channels: int = 12) -> List[Channel]:
        """
        从通道列表中选择需要可视化的通道

        Args:
            channels: 所有可用通道列表
            max_channels: 最大通道数

        Returns:
            选中的通道列表
        """
        pass


class ImportantPatternSelector(ChannelSelectionStrategy):
    """
    基于重要性模式的通道选择器

    优先选择电力系统中常见的重要参数
    """

    def __init__(self):
        # 按优先级定义重要参数模式
        self.important_patterns = [
            ('电压', '有功', '无功', '电流'),  # 组1：电气量
            ('频率', '角度', '功率'),          # 组2：系统参数
            ('AVR', 'PSS'),                   # 组3：控制器
            ('励磁',),                        # 组4：励磁系统
        ]

    def select_channels(self, channels: List[Channel],
                       max_channels: int = 12) -> List[Channel]:
        """选择重要通道"""
        selected_channels = []

        # 按优先级选择通道
        for patterns in self.important_patterns:
            for ch in channels:
                if any(pattern in ch.name for pattern in patterns):
                    if ch not in selected_channels:
                        selected_channels.append(ch)

        # 如果选中的通道太少，补充前面的通道
        if len(selected_channels) < 8:
            for ch in channels:
                if ch not in selected_channels:
                    selected_channels.append(ch)
                if len(selected_channels) >= max_channels:
                    break

        # 限制最大通道数
        return selected_channels[:max_channels]


class FirstNChannelsSelector(ChannelSelectionStrategy):
    """简单选择前N个通道"""

    def select_channels(self, channels: List[Channel],
                       max_channels: int = 12) -> List[Channel]:
        """选择前N个通道"""
        return channels[:max_channels]


class PatternMatchSelector(ChannelSelectionStrategy):
    """基于模式匹配的通道选择器"""

    def __init__(self, patterns: List[str]):
        """
        Args:
            patterns: 需要匹配的模式列表
        """
        self.patterns = patterns

    def select_channels(self, channels: List[Channel],
                       max_channels: int = 12) -> List[Channel]:
        """选择匹配模式的通道"""
        selected = []
        for ch in channels:
            if any(pattern in ch.name for pattern in self.patterns):
                selected.append(ch)
                if len(selected) >= max_channels:
                    break
        return selected


class AllChannelsSelector(ChannelSelectionStrategy):
    """选择所有通道"""

    def select_channels(self, channels: List[Channel],
                       max_channels: int = None) -> List[Channel]:
        """选择所有通道"""
        if max_channels is None:
            return channels
        return channels[:max_channels]
