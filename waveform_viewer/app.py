#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMTRADE波形可视化工具主应用程序

这是整个应用的入口点，整合了所有功能模块
"""

import sys
from pathlib import Path
from typing import List, Optional

from .core.reader import ComtradeReader
from .core.channel_selector import (
    ChannelSelectionStrategy,
    ImportantPatternSelector,
    FirstNChannelsSelector,
    AllChannelsSelector
)
from .visualizers.base import VisualizationConfig
from .visualizers.plotly_viz import PlotlyVisualizer, PlotlyLineVisualizer
from .plugins.manager import PluginManager
from .utils.file_finder import WaveformFileFinder
from .ui.menu import InteractiveMenu, SimpleMenu


class WaveformViewerApp:
    """
    波形查看器主应用程序

    整合所有功能模块，提供统一的应用接口
    """

    def __init__(self, base_dir: str, use_simple_menu: bool = False):
        """
        初始化应用程序

        Args:
            base_dir: 基础目录路径
            use_simple_menu: 是否使用简单菜单（不支持方向键的环境）
        """
        self.base_dir = Path(base_dir)
        self.file_finder = WaveformFileFinder(str(self.base_dir))

        # 初始化菜单
        if use_simple_menu:
            self.menu = SimpleMenu()
        else:
            try:
                self.menu = InteractiveMenu()
            except Exception:
                # 如果交互式菜单不可用，回退到简单菜单
                self.menu = SimpleMenu()
                print("注意: 使用简单数字选择菜单")

        # 初始化插件管理器
        self.plugin_manager = PluginManager()
        self._setup_plugins()

        # 默认配置
        self.channel_selector: ChannelSelectionStrategy = ImportantPatternSelector()
        self.visualizer = PlotlyVisualizer()
        self.viz_config = VisualizationConfig()

    def _setup_plugins(self):
        """设置插件系统"""
        # 添加插件目录
        plugin_dir = Path(__file__).parent / 'plugins'
        self.plugin_manager.add_plugin_directory(str(plugin_dir))

        # 发现并加载插件
        self.plugin_manager.discover_plugins()

    def run(self):
        """运行主应用程序"""
        try:
            print("\n" + "=" * 70)
            print("COMTRADE波形可视化工具")
            print("=" * 70)

            # 查找波形文件夹
            waveform_groups = self.file_finder.group_waveform_files()

            if not waveform_groups:
                print("\n未找到波形数据文件夹")
                print("请确保波形文件（.cfg和.dat）位于子文件夹中")
                return

            # 1. 选择要处理的文件夹
            folders = self._select_folders(list(waveform_groups.keys()))

            if not folders:
                print("\n未选择任何文件夹，退出程序")
                return

            # 2. 为每个文件夹选择要处理的文件
            all_selected_files = []
            for folder_name in folders:
                cfg_files = waveform_groups[folder_name]
                selected_files = self._select_files(folder_name, cfg_files)
                all_selected_files.extend(selected_files)

            if not all_selected_files:
                print("\n未选择任何文件，退出程序")
                return

            # 3. 选择可视化选项
            self._configure_visualization()

            # 4. 处理文件
            print("\n" + "=" * 70)
            print("开始处理波形文件...")
            print("=" * 70)

            for cfg_file in all_selected_files:
                self._process_file(cfg_file)

            print("\n" + "=" * 70)
            print("所有波形文件处理完成！")
            print("=" * 70)

        except KeyboardInterrupt:
            print("\n\n程序已取消")
            sys.exit(0)
        except Exception as e:
            print(f"\n错误: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    def _select_folders(self, folder_names: List[str]) -> List[str]:
        """选择要处理的文件夹"""
        if len(folder_names) == 0:
            return []

        if len(folder_names) == 1:
            # 只有一个文件夹，直接返回
            print(f"\n找到波形文件夹: {folder_names[0]}")
            return folder_names

        # 多个文件夹，让用户选择
        self.menu.title = "选择要处理的波形文件夹（可多选）:"

        try:
            indices, selected = self.menu.multi_choice(
                folder_names,
                default_indices=list(range(len(folder_names)))  # 默认全选
            )
            return selected
        except AttributeError:
            # 简单菜单
            indices, selected = self.menu.multi_choice(folder_names)
            return selected

    def _select_files(self, folder_name: str, cfg_files: List[Path]) -> List[Path]:
        """选择要处理的文件"""
        if len(cfg_files) == 0:
            return []

        print(f"\n文件夹: {folder_name}")
        print(f"找到 {len(cfg_files)} 个波形文件")

        if len(cfg_files) == 1:
            # 只有一个文件，直接返回
            print(f"  → {cfg_files[0].stem}")
            return cfg_files

        # 创建选项列表
        options = [f"{f.stem}" for f in cfg_files]

        self.menu.title = f"[{folder_name}] 选择要可视化的文件（可多选）:"

        try:
            indices, _ = self.menu.multi_choice(
                options,
                default_indices=list(range(len(cfg_files)))  # 默认全选
            )
        except AttributeError:
            indices, _ = self.menu.multi_choice(options)

        return [cfg_files[i] for i in indices]

    def _configure_visualization(self):
        """配置可视化选项"""
        print("\n" + "=" * 70)
        print("配置可视化选项")
        print("=" * 70)

        # 1. 选择通道选择策略
        strategies = [
            "智能选择（优先显示重要通道）",
            "显示前12个通道",
            "显示所有通道"
        ]

        self.menu.title = "选择通道选择策略:"
        idx, _ = self.menu.single_choice(strategies, default_index=0)

        if idx == 0:
            self.channel_selector = ImportantPatternSelector()
        elif idx == 1:
            self.channel_selector = FirstNChannelsSelector()
        else:
            self.channel_selector = AllChannelsSelector()

        # 2. 选择可视化样式
        styles = [
            "散点图（推荐用于密集采样）",
            "线图（推荐用于稀疏采样）"
        ]

        self.menu.title = "选择可视化样式:"
        idx, _ = self.menu.single_choice(styles, default_index=0)

        if idx == 0:
            self.visualizer = PlotlyVisualizer(self.viz_config)
        else:
            self.visualizer = PlotlyLineVisualizer(self.viz_config)

        print("\n配置完成！")

    def _process_file(self, cfg_file: Path):
        """处理单个波形文件"""
        print(f"\n处理文件: {cfg_file.stem}")
        print("-" * 70)

        try:
            # 1. 读取COMTRADE数据
            reader = ComtradeReader(str(cfg_file))

            # 2. 选择通道
            channels = self.channel_selector.select_channels(
                reader.analog_channels,
                max_channels=12 if not isinstance(self.channel_selector, AllChannelsSelector) else None
            )

            if not channels:
                print("  警告: 未找到可用通道")
                return

            print(f"\n  选择以下 {len(channels)} 个通道进行可视化:")
            for ch in channels[:10]:  # 只显示前10个
                print(f"    {ch.index}. {ch.name} ({ch.unit})")
            if len(channels) > 10:
                print(f"    ... 还有 {len(channels) - 10} 个通道")

            # 3. 生成可视化
            output_path = cfg_file.parent.parent / f"{cfg_file.stem}_visualization.html"
            result_path = self.visualizer.visualize(reader, channels, str(output_path))

            print(f"\n  ✓ 可视化结果已保存到: {Path(result_path).name}")

        except Exception as e:
            print(f"\n  ✗ 处理文件时出错: {e}")
            import traceback
            traceback.print_exc()

    def set_channel_selector(self, selector: ChannelSelectionStrategy):
        """设置通道选择器"""
        self.channel_selector = selector

    def set_visualizer(self, visualizer, config: Optional[VisualizationConfig] = None):
        """设置可视化器"""
        self.visualizer = visualizer
        if config:
            self.viz_config = config

    def execute_plugin(self, plugin_name: str, reader: ComtradeReader, **kwargs):
        """执行插件"""
        return self.plugin_manager.execute_plugin(plugin_name, reader, kwargs)


def main():
    """主函数"""
    # 获取当前脚本所在目录
    current_dir = Path(__file__).parent.parent

    # 波形文件夹路径
    waves_dir = current_dir / "waves"

    if not waves_dir.exists():
        print(f"错误: 波形文件夹不存在: {waves_dir}")
        print("请确保 'waves' 文件夹存在并包含波形数据")
        sys.exit(1)

    # 创建并运行应用
    app = WaveformViewerApp(str(waves_dir), use_simple_menu=False)
    app.run()


if __name__ == '__main__':
    main()
