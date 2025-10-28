#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互式菜单模块

提供基于方向键的交互式菜单选择功能
"""

import sys
import tty
import termios
from typing import List, Tuple, Optional, Callable


class InteractiveMenu:
    """
    交互式菜单

    支持使用方向键（上下）和空格键进行多选
    """

    def __init__(self, title: str = "请选择:"):
        """
        初始化交互式菜单

        Args:
            title: 菜单标题
        """
        self.title = title

    def _get_key(self) -> str:
        """获取用户按键"""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            key = sys.stdin.read(1)
            # 处理方向键（ESC序列）
            if key == '\x1b':
                key += sys.stdin.read(2)
            return key
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def single_choice(self,
                     options: List[str],
                     default_index: int = 0) -> Tuple[int, str]:
        """
        单选菜单

        Args:
            options: 选项列表
            default_index: 默认选中的索引

        Returns:
            (选中的索引, 选中的选项)
        """
        current = default_index

        while True:
            # 清屏并显示菜单
            self._display_menu(options, current, [])

            # 获取按键
            key = self._get_key()

            if key == '\x1b[A':  # 上箭头
                current = (current - 1) % len(options)
            elif key == '\x1b[B':  # 下箭头
                current = (current + 1) % len(options)
            elif key == '\r' or key == '\n':  # 回车确认
                self._clear_screen()
                return current, options[current]
            elif key == '\x03':  # Ctrl+C
                self._clear_screen()
                raise KeyboardInterrupt()
            elif key == 'q':  # q退出
                self._clear_screen()
                raise KeyboardInterrupt()

    def multi_choice(self,
                    options: List[str],
                    default_indices: Optional[List[int]] = None) -> Tuple[List[int], List[str]]:
        """
        多选菜单

        Args:
            options: 选项列表
            default_indices: 默认选中的索引列表

        Returns:
            (选中的索引列表, 选中的选项列表)
        """
        current = 0
        selected = set(default_indices or [])

        while True:
            # 清屏并显示菜单
            self._display_menu(options, current, selected)

            # 获取按键
            key = self._get_key()

            if key == '\x1b[A':  # 上箭头
                current = (current - 1) % len(options)
            elif key == '\x1b[B':  # 下箭头
                current = (current + 1) % len(options)
            elif key == ' ':  # 空格切换选中状态
                if current in selected:
                    selected.remove(current)
                else:
                    selected.add(current)
            elif key == '\r' or key == '\n':  # 回车确认
                self._clear_screen()
                selected_list = sorted(list(selected))
                return selected_list, [options[i] for i in selected_list]
            elif key == '\x03':  # Ctrl+C
                self._clear_screen()
                raise KeyboardInterrupt()
            elif key == 'q':  # q退出
                self._clear_screen()
                raise KeyboardInterrupt()
            elif key == 'a':  # a全选
                selected = set(range(len(options)))
            elif key == 'n':  # n取消全选
                selected = set()

    def _display_menu(self, options: List[str], current: int, selected: set):
        """显示菜单"""
        self._clear_screen()

        print(f"\n{self.title}")
        print("=" * 70)

        for idx, option in enumerate(options):
            # 选中标记
            cursor = "→ " if idx == current else "  "

            # 多选标记
            checkbox = ""
            if len(selected) > 0 or isinstance(selected, set):
                checkbox = "[✓] " if idx in selected else "[ ] "

            print(f"{cursor}{checkbox}{option}")

        print("\n" + "=" * 70)
        if len(selected) > 0 or isinstance(selected, set):
            print("提示: ↑↓ 移动 | 空格 选择 | 回车 确认 | a 全选 | n 取消全选 | q 退出")
        else:
            print("提示: ↑↓ 移动 | 回车 确认 | q 退出")

    def _clear_screen(self):
        """清屏"""
        print('\033[2J\033[H', end='')

    @staticmethod
    def confirm(message: str, default: bool = True) -> bool:
        """
        确认对话框

        Args:
            message: 确认消息
            default: 默认值

        Returns:
            用户确认结果
        """
        default_hint = "[Y/n]" if default else "[y/N]"
        response = input(f"{message} {default_hint}: ").strip().lower()

        if not response:
            return default

        return response in ['y', 'yes', '是']


class SimpleMenu:
    """
    简单菜单（不需要termios，使用数字选择）

    适用于不支持方向键的环境
    """

    def __init__(self, title: str = "请选择:"):
        self.title = title

    def single_choice(self, options: List[str], default_index: int = 0) -> Tuple[int, str]:
        """数字选择菜单"""
        print(f"\n{self.title}")
        print("=" * 70)

        for idx, option in enumerate(options):
            marker = "*" if idx == default_index else " "
            print(f"{marker} {idx + 1}. {option}")

        print("=" * 70)

        while True:
            try:
                choice = input(f"请输入序号 (1-{len(options)}, 默认 {default_index + 1}): ").strip()

                if not choice:
                    return default_index, options[default_index]

                choice_idx = int(choice) - 1

                if 0 <= choice_idx < len(options):
                    return choice_idx, options[choice_idx]
                else:
                    print(f"请输入 1 到 {len(options)} 之间的数字")

            except ValueError:
                print("请输入有效的数字")
            except KeyboardInterrupt:
                raise

    def multi_choice(self, options: List[str],
                    default_indices: Optional[List[int]] = None) -> Tuple[List[int], List[str]]:
        """多选数字菜单"""
        print(f"\n{self.title}")
        print("=" * 70)

        for idx, option in enumerate(options):
            marker = "✓" if default_indices and idx in default_indices else " "
            print(f"[{marker}] {idx + 1}. {option}")

        print("=" * 70)
        print("提示: 输入序号，用逗号或空格分隔，如: 1,3,5 或 1 3 5")
        print("      输入 'all' 选择全部")

        while True:
            try:
                choice = input("请输入选择: ").strip()

                if not choice and default_indices:
                    return default_indices, [options[i] for i in default_indices]

                if choice.lower() == 'all':
                    all_indices = list(range(len(options)))
                    return all_indices, options

                # 解析输入
                choice = choice.replace(',', ' ')
                indices = []

                for num_str in choice.split():
                    try:
                        idx = int(num_str) - 1
                        if 0 <= idx < len(options):
                            if idx not in indices:
                                indices.append(idx)
                    except ValueError:
                        continue

                if indices:
                    indices.sort()
                    return indices, [options[i] for i in indices]
                else:
                    print("请输入有效的序号")

            except KeyboardInterrupt:
                raise
