#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HDR文件读取器模块

负责读取和解析COMTRADE格式的.hdr头文件（XML格式）
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime


class TripInfo:
    """跳闸信息数据类"""

    def __init__(self, time: str, name: str, phase: str, value: int):
        self.time = time
        self.name = name
        self.phase = phase
        self.value = value

    def __repr__(self):
        return f"TripInfo(time={self.time}, name={self.name}, value={self.value})"


class DigitalStatus:
    """数字状态数据类"""

    def __init__(self, name: str, value: int):
        self.name = name
        self.value = value

    def __repr__(self):
        return f"DigitalStatus(name={self.name}, value={self.value})"


class DigitalEvent:
    """数字事件数据类"""

    def __init__(self, time: int, name: str, value: int):
        self.time = time
        self.name = name
        self.value = value

    def __repr__(self):
        return f"DigitalEvent(time={self.time}ms, name={self.name}, value={self.value})"


class SettingValue:
    """设置值数据类"""

    def __init__(self, name: str, value: str, unit: str):
        self.name = name
        self.value = value
        self.unit = unit

    def __repr__(self):
        return f"SettingValue(name={self.name}, value={self.value}, unit={self.unit})"


class HdrReader:
    """
    HDR文件读取器

    负责读取和解析COMTRADE格式的.hdr头文件
    """

    def __init__(self, hdr_file: str):
        """
        初始化HDR读取器

        Args:
            hdr_file: .hdr文件路径
        """
        self.hdr_file = hdr_file
        self.fault_start_time: Optional[str] = None
        self.data_file_size: Optional[int] = None
        self.fault_keeping_time: Optional[str] = None

        self.trip_infos: List[TripInfo] = []
        self.digital_statuses: List[DigitalStatus] = []
        self.digital_events: List[DigitalEvent] = []
        self.setting_values: List[SettingValue] = []

        self._parse_hdr()

    def _parse_hdr(self):
        """解析.hdr XML文件"""
        try:
            tree = ET.parse(self.hdr_file)
            root = tree.getroot()

            # 解析故障开始时间
            fault_start_time_elem = root.find('FaultStartTime')
            if fault_start_time_elem is not None:
                self.fault_start_time = fault_start_time_elem.text

            # 解析数据文件大小
            data_file_size_elem = root.find('DataFileSize')
            if data_file_size_elem is not None and data_file_size_elem.text:
                self.data_file_size = int(data_file_size_elem.text)

            # 解析故障保持时间
            fault_keeping_time_elem = root.find('FaultKeepingTime')
            if fault_keeping_time_elem is not None:
                self.fault_keeping_time = fault_keeping_time_elem.text

            # 解析跳闸信息
            for trip_info_elem in root.findall('TripInfo'):
                time = trip_info_elem.find('time').text.strip() if trip_info_elem.find('time') is not None else ""
                name = trip_info_elem.find('name').text.strip() if trip_info_elem.find('name') is not None else ""
                phase = trip_info_elem.find('phase').text.strip() if trip_info_elem.find('phase') is not None else ""
                value = int(trip_info_elem.find('value').text) if trip_info_elem.find('value') is not None else 0

                self.trip_infos.append(TripInfo(time, name, phase, value))

            # 解析数字状态
            for digital_status_elem in root.findall('DigitalStatus'):
                name = digital_status_elem.find('name').text.strip() if digital_status_elem.find('name') is not None else ""
                value = int(digital_status_elem.find('value').text) if digital_status_elem.find('value') is not None else 0

                self.digital_statuses.append(DigitalStatus(name, value))

            # 解析数字事件
            for digital_event_elem in root.findall('DigitalEvent'):
                time = int(digital_event_elem.find('time').text) if digital_event_elem.find('time') is not None else 0
                name = digital_event_elem.find('name').text.strip() if digital_event_elem.find('name') is not None else ""
                value = int(digital_event_elem.find('value').text) if digital_event_elem.find('value') is not None else 0

                self.digital_events.append(DigitalEvent(time, name, value))

            # 解析设置值
            for setting_value_elem in root.findall('SettingValue'):
                name = setting_value_elem.find('name').text.strip() if setting_value_elem.find('name') is not None else ""
                value = setting_value_elem.find('value').text.strip() if setting_value_elem.find('value') is not None else ""
                unit = setting_value_elem.find('unit').text.strip() if setting_value_elem.find('unit') is not None else ""

                self.setting_values.append(SettingValue(name, value, unit))

            print(f"  HDR文件信息:")
            print(f"    故障开始时间: {self.fault_start_time}")
            print(f"    跳闸信息数量: {len(self.trip_infos)}")
            print(f"    数字状态数量: {len(self.digital_statuses)}")
            print(f"    数字事件数量: {len(self.digital_events)}")
            print(f"    设置值数量: {len(self.setting_values)}")

        except Exception as e:
            print(f"  警告: 解析HDR文件时出错: {e}")

    def get_trip_infos(self) -> List[TripInfo]:
        """获取跳闸信息列表"""
        return self.trip_infos

    def get_digital_statuses(self) -> List[DigitalStatus]:
        """获取数字状态列表"""
        return self.digital_statuses

    def get_digital_events(self) -> List[DigitalEvent]:
        """获取数字事件列表"""
        return self.digital_events

    def get_setting_values(self) -> List[SettingValue]:
        """获取设置值列表"""
        return self.setting_values

    def get_active_digital_statuses(self) -> List[DigitalStatus]:
        """获取激活状态（值为1）的数字状态"""
        return [status for status in self.digital_statuses if status.value == 1]

    def get_summary(self) -> Dict[str, any]:
        """
        获取HDR文件摘要信息

        Returns:
            包含摘要信息的字典
        """
        return {
            'fault_start_time': self.fault_start_time,
            'data_file_size': self.data_file_size,
            'fault_keeping_time': self.fault_keeping_time,
            'trip_info_count': len(self.trip_infos),
            'digital_status_count': len(self.digital_statuses),
            'digital_event_count': len(self.digital_events),
            'setting_value_count': len(self.setting_values),
            'active_status_count': len(self.get_active_digital_statuses())
        }
