#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件查找工具

用于查找和管理COMTRADE波形文件
"""

from pathlib import Path
from typing import List, Dict, Tuple


class WaveformFileFinder:
    """波形文件查找器"""

    def __init__(self, base_dir: str):
        """
        初始化文件查找器

        Args:
            base_dir: 基础目录路径
        """
        self.base_dir = Path(base_dir)

    def find_waveform_folders(self, exclude_dirs: set = None) -> List[Path]:
        """
        查找所有包含波形文件的文件夹

        Args:
            exclude_dirs: 要排除的目录集合

        Returns:
            文件夹路径列表
        """
        if exclude_dirs is None:
            exclude_dirs = {'venv', '.venv', 'env', '.env',
                          '__pycache__', '.git', '.claude', 'waveform_viewer'}

        folders = []

        if not self.base_dir.exists():
            return folders

        for item in self.base_dir.iterdir():
            if item.is_dir() and item.name not in exclude_dirs and not item.name.startswith('.'):
                # 检查是否包含.cfg文件
                if list(item.glob('*.cfg')):
                    folders.append(item)

        return folders

    def find_cfg_files(self, folder: Path = None) -> List[Path]:
        """
        查找指定文件夹中的所有.cfg文件

        Args:
            folder: 文件夹路径，如果为None则使用base_dir

        Returns:
            .cfg文件路径列表
        """
        search_path = folder or self.base_dir
        return sorted(search_path.glob('*.cfg'))

    def group_waveform_files(self) -> Dict[str, List[Path]]:
        """
        按文件夹分组所有波形文件

        Returns:
            {文件夹名: [.cfg文件列表]}
        """
        result = {}

        folders = self.find_waveform_folders()

        for folder in folders:
            cfg_files = self.find_cfg_files(folder)
            if cfg_files:
                result[folder.name] = cfg_files

        return result

    def get_waveform_info(self, cfg_file: Path) -> Dict[str, str]:
        """
        获取波形文件基本信息

        Args:
            cfg_file: .cfg文件路径

        Returns:
            文件信息字典
        """
        info = {
            'name': cfg_file.stem,
            'folder': cfg_file.parent.name,
            'path': str(cfg_file),
            'size': self._get_file_size(cfg_file)
        }

        # 检查对应的.dat文件是否存在
        dat_file = cfg_file.with_suffix('.dat')
        info['dat_exists'] = dat_file.exists()
        if info['dat_exists']:
            info['dat_size'] = self._get_file_size(dat_file)

        return info

    @staticmethod
    def _get_file_size(file_path: Path) -> str:
        """获取文件大小（人类可读格式）"""
        if not file_path.exists():
            return "N/A"

        size_bytes = file_path.stat().st_size

        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024

        return f"{size_bytes:.1f} TB"

    def validate_waveform_file(self, cfg_file: Path) -> Tuple[bool, str]:
        """
        验证波形文件是否完整

        Args:
            cfg_file: .cfg文件路径

        Returns:
            (是否有效, 错误消息)
        """
        if not cfg_file.exists():
            return False, f"配置文件不存在: {cfg_file}"

        # 检查.dat文件
        dat_file = cfg_file.with_suffix('.dat')
        if not dat_file.exists():
            return False, f"数据文件不存在: {dat_file}"

        # 检查文件是否为空
        if cfg_file.stat().st_size == 0:
            return False, "配置文件为空"

        if dat_file.stat().st_size == 0:
            return False, "数据文件为空"

        return True, "文件有效"
