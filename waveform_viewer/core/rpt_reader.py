#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RPT文件读取器模块

负责读取和解析COMTRADE格式的.rpt报告文件（专有二进制格式）
"""

import struct
from pathlib import Path
from typing import List, Dict, Optional, Any


class RptReader:
    """
    RPT文件读取器

    负责读取COMTRADE格式的.rpt报告文件
    注意: .rpt文件是专有二进制格式，此实现提供基础框架
    """

    def __init__(self, rpt_file: str):
        """
        初始化RPT读取器

        Args:
            rpt_file: .rpt文件路径
        """
        self.rpt_file = rpt_file
        self.file_size: int = 0
        self.raw_data: bytes = b''
        self.header_info: Dict[str, Any] = {}
        self.is_valid: bool = False

        self._read_rpt()

    def _read_rpt(self):
        """读取.rpt二进制文件"""
        try:
            file_path = Path(self.rpt_file)

            if not file_path.exists():
                print(f"  警告: RPT文件不存在: {self.rpt_file}")
                return

            self.file_size = file_path.stat().st_size
            print(f"  RPT文件信息:")
            print(f"    文件大小: {self.file_size} 字节")

            # 读取文件内容
            with open(self.rpt_file, 'rb') as f:
                self.raw_data = f.read()

            # 尝试解析基本信息
            self._parse_basic_info()

        except Exception as e:
            print(f"  警告: 读取RPT文件时出错: {e}")

    def _parse_basic_info(self):
        """
        解析RPT文件的基本信息

        注意: 由于.rpt文件是专有二进制格式，这里仅提供基础框架
        具体格式需要根据设备厂商的文档进行解析
        """
        if len(self.raw_data) < 16:
            print(f"    警告: 文件太小，可能不是有效的RPT文件")
            return

        try:
            # 尝试读取前几个字节作为示例
            # 这里仅为演示，实际格式需要根据厂商文档确定

            # 检查是否有可识别的文本标识
            text_markers = []
            offset = 0
            while offset < min(len(self.raw_data), 1024):
                # 查找可打印的ASCII字符串
                if self.raw_data[offset:offset+1].isalpha():
                    # 尝试提取文本
                    end = offset
                    while end < len(self.raw_data) and self.raw_data[end:end+1].isalnum():
                        end += 1
                    if end - offset > 3:  # 至少4个字符
                        try:
                            text = self.raw_data[offset:end].decode('ascii', errors='ignore')
                            if len(text) > 3:
                                text_markers.append((offset, text))
                        except:
                            pass
                offset += 1

            if text_markers:
                print(f"    发现文本标识: {len(text_markers)} 个")
                for offset, text in text_markers[:5]:  # 显示前5个
                    print(f"      偏移 {offset}: {text}")

            # 尝试识别一些数值字段（假设前16字节可能是头部）
            if len(self.raw_data) >= 16:
                # 这只是示例，实际格式需要文档支持
                header_bytes = self.raw_data[:16]
                self.header_info['first_4_bytes'] = ' '.join(f'{b:02X}' for b in header_bytes[:4])
                self.header_info['next_4_bytes'] = ' '.join(f'{b:02X}' for b in header_bytes[4:8])

                print(f"    头部信息 (hex):")
                print(f"      字节 0-3:  {self.header_info['first_4_bytes']}")
                print(f"      字节 4-7:  {self.header_info['next_4_bytes']}")

            self.is_valid = True

        except Exception as e:
            print(f"    警告: 解析RPT基本信息时出错: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """
        获取RPT文件摘要信息

        Returns:
            包含摘要信息的字典
        """
        return {
            'file_path': self.rpt_file,
            'file_size': self.file_size,
            'is_valid': self.is_valid,
            'header_info': self.header_info
        }

    def export_raw_data(self, output_file: str, max_bytes: int = 1024) -> str:
        """
        导出原始数据的十六进制表示（用于分析）

        Args:
            output_file: 输出文件路径
            max_bytes: 最多导出的字节数

        Returns:
            输出文件路径
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"RPT文件分析报告\n")
                f.write(f"=" * 70 + "\n")
                f.write(f"文件路径: {self.rpt_file}\n")
                f.write(f"文件大小: {self.file_size} 字节\n")
                f.write(f"\n")

                f.write(f"十六进制转储 (前 {max_bytes} 字节):\n")
                f.write(f"-" * 70 + "\n")

                bytes_to_export = min(max_bytes, len(self.raw_data))
                for i in range(0, bytes_to_export, 16):
                    # 地址
                    f.write(f"{i:08X}  ")

                    # 十六进制
                    hex_part = ' '.join(f'{b:02X}' for b in self.raw_data[i:i+16])
                    f.write(f"{hex_part:<48}  ")

                    # ASCII表示
                    ascii_part = ''.join(
                        chr(b) if 32 <= b < 127 else '.'
                        for b in self.raw_data[i:i+16]
                    )
                    f.write(f"{ascii_part}\n")

            print(f"    已导出RPT原始数据分析到: {output_file}")
            return output_file

        except Exception as e:
            print(f"    警告: 导出RPT数据时出错: {e}")
            return ""

    def has_valid_data(self) -> bool:
        """检查是否有有效数据"""
        return self.is_valid and len(self.raw_data) > 0
