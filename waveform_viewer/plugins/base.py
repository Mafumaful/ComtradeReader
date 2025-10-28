#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
插件基类模块

定义插件系统的基础接口
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from ..core.reader import ComtradeReader


class Plugin(ABC):
    """
    插件基类

    所有插件都需要继承这个类并实现相应的方法
    """

    def __init__(self):
        self.name = self.__class__.__name__
        self.version = "1.0.0"
        self.description = ""
        self.enabled = True

    @abstractmethod
    def execute(self, reader: ComtradeReader, context: Dict[str, Any]) -> Any:
        """
        执行插件功能

        Args:
            reader: COMTRADE读取器
            context: 上下文信息，包含配置等

        Returns:
            插件执行结果
        """
        pass

    def on_load(self):
        """插件加载时调用"""
        pass

    def on_unload(self):
        """插件卸载时调用"""
        pass

    def get_info(self) -> Dict[str, str]:
        """获取插件信息"""
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'enabled': str(self.enabled)
        }


class DataProcessorPlugin(Plugin):
    """
    数据处理插件基类

    用于处理和转换波形数据
    """

    @abstractmethod
    def process_data(self, time_values, data_values, **kwargs) -> tuple:
        """
        处理数据

        Args:
            time_values: 时间值列表
            data_values: 数据值列表
            **kwargs: 其他参数

        Returns:
            处理后的 (time_values, data_values)
        """
        pass


class ExportPlugin(Plugin):
    """
    导出插件基类

    用于将数据导出到不同格式
    """

    @abstractmethod
    def export(self, reader: ComtradeReader, output_path: str, **kwargs) -> str:
        """
        导出数据

        Args:
            reader: COMTRADE读取器
            output_path: 输出路径
            **kwargs: 其他参数

        Returns:
            导出的文件路径
        """
        pass

    @abstractmethod
    def get_file_extension(self) -> str:
        """获取文件扩展名"""
        pass
