#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
插件管理器

负责插件的加载、管理和执行
"""

import os
import importlib.util
from pathlib import Path
from typing import Dict, List, Type, Optional
from .base import Plugin


class PluginManager:
    """
    插件管理器

    使用单例模式，负责管理所有插件的生命周期
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._plugins: Dict[str, Plugin] = {}
        self._plugin_dirs: List[str] = []
        self._initialized = True

    def add_plugin_directory(self, directory: str):
        """
        添加插件目录

        Args:
            directory: 插件目录路径
        """
        if os.path.isdir(directory):
            self._plugin_dirs.append(directory)

    def discover_plugins(self):
        """
        发现并加载所有插件

        从指定的插件目录中自动发现Python插件
        """
        for plugin_dir in self._plugin_dirs:
            self._load_plugins_from_dir(plugin_dir)

    def _load_plugins_from_dir(self, directory: str):
        """从目录加载插件"""
        plugin_path = Path(directory)

        if not plugin_path.exists():
            return

        for py_file in plugin_path.glob('*.py'):
            if py_file.name.startswith('_'):
                continue

            try:
                # 动态导入模块
                spec = importlib.util.spec_from_file_location(
                    py_file.stem, py_file
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # 查找Plugin子类
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and
                        issubclass(attr, Plugin) and
                        attr != Plugin):

                        plugin_instance = attr()
                        self.register_plugin(plugin_instance)

            except Exception as e:
                print(f"加载插件 {py_file.name} 时出错: {e}")

    def register_plugin(self, plugin: Plugin):
        """
        注册插件

        Args:
            plugin: 插件实例
        """
        plugin.on_load()
        self._plugins[plugin.name] = plugin
        print(f"已注册插件: {plugin.name} v{plugin.version}")

    def unregister_plugin(self, plugin_name: str):
        """
        注销插件

        Args:
            plugin_name: 插件名称
        """
        if plugin_name in self._plugins:
            plugin = self._plugins[plugin_name]
            plugin.on_unload()
            del self._plugins[plugin_name]
            print(f"已注销插件: {plugin_name}")

    def get_plugin(self, plugin_name: str) -> Optional[Plugin]:
        """
        获取插件实例

        Args:
            plugin_name: 插件名称

        Returns:
            插件实例，如果不存在返回None
        """
        return self._plugins.get(plugin_name)

    def get_all_plugins(self) -> Dict[str, Plugin]:
        """获取所有插件"""
        return self._plugins.copy()

    def get_enabled_plugins(self) -> Dict[str, Plugin]:
        """获取所有启用的插件"""
        return {name: plugin for name, plugin in self._plugins.items()
                if plugin.enabled}

    def execute_plugin(self, plugin_name: str, *args, **kwargs):
        """
        执行插件

        Args:
            plugin_name: 插件名称
            *args, **kwargs: 传递给插件的参数

        Returns:
            插件执行结果
        """
        plugin = self.get_plugin(plugin_name)
        if plugin and plugin.enabled:
            return plugin.execute(*args, **kwargs)
        return None

    def list_plugins(self):
        """列出所有插件信息"""
        print("\n已加载的插件:")
        print("-" * 70)
        for name, plugin in self._plugins.items():
            info = plugin.get_info()
            print(f"  {info['name']} v{info['version']}")
            print(f"    {info['description']}")
            print(f"    状态: {'启用' if plugin.enabled else '禁用'}")
