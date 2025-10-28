#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件打开工具

提供跨平台的文件打开功能
"""

import sys
import subprocess
import os
from pathlib import Path


def open_file_in_browser(file_path: str) -> bool:
    """
    在默认浏览器中打开文件

    Args:
        file_path: 文件路径

    Returns:
        是否成功打开
    """
    try:
        abs_path = Path(file_path).resolve()

        if not abs_path.exists():
            print(f"  错误: 文件不存在 - {abs_path}")
            return False

        # 根据操作系统使用不同的方法
        system = sys.platform

        if system == 'darwin':  # macOS
            # 使用 open 命令，这是 macOS 最可靠的方法
            subprocess.run(['open', str(abs_path)], check=True)
            return True

        elif system == 'win32':  # Windows
            # Windows 使用 os.startfile
            os.startfile(str(abs_path))
            return True

        elif system.startswith('linux'):  # Linux
            # Linux 使用 xdg-open
            subprocess.run(['xdg-open', str(abs_path)], check=True)
            return True

        else:
            # 其他系统，尝试使用 webbrowser
            import webbrowser
            webbrowser.open(f'file://{abs_path}')
            return True

    except FileNotFoundError as e:
        print(f"  错误: 找不到打开命令 - {e}")
        return False
    except subprocess.CalledProcessError as e:
        print(f"  错误: 打开文件失败 - {e}")
        return False
    except Exception as e:
        print(f"  错误: 无法打开文件 - {e}")
        return False


def open_file_with_default_app(file_path: str) -> bool:
    """
    使用系统默认应用程序打开文件（更通用的方法）

    Args:
        file_path: 文件路径

    Returns:
        是否成功打开
    """
    return open_file_in_browser(file_path)
