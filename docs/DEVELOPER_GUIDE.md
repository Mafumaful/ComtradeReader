# COMTRADE波形可视化工具 - 开发者指南

本文档为后续开发者提供详细的开发指南，包括架构说明、扩展方法和最佳实践。

## 目录

1. [架构概览](#架构概览)
2. [核心模块说明](#核心模块说明)
3. [如何添加新的通道选择策略](#如何添加新的通道选择策略)
4. [如何添加新的可视化器](#如何添加新的可视化器)
5. [如何开发插件](#如何开发插件)
6. [如何扩展UI](#如何扩展ui)
7. [测试指南](#测试指南)
8. [最佳实践](#最佳实践)

---

## 架构概览

### 设计模式

本项目采用了多种设计模式，确保代码的可扩展性和可维护性：

#### 1. Strategy Pattern (策略模式)

用于：
- **通道选择策略** (`channel_selector.py`)
- **可视化策略** (`visualizers/`)

好处：
- 可以轻松添加新的策略而不修改现有代码
- 策略可以在运行时动态切换

#### 2. Plugin Architecture (插件架构)

用于：
- **数据处理插件** (`plugins/`)
- **导出插件** (`plugins/`)

好处：
- 第三方可以开发自己的插件
- 插件可以独立开发和测试
- 不影响核心功能

#### 3. Singleton Pattern (单例模式)

用于：
- **PluginManager** - 全局唯一的插件管理器

好处：
- 确保插件只被加载一次
- 全局统一的插件管理

#### 4. Factory Pattern (工厂模式)

用于：
- 创建不同类型的选择器和可视化器

### 模块依赖关系

```
main.py
  └─> app.py (WaveformViewerApp)
       ├─> core/
       │   ├─> reader.py (ComtradeReader)
       │   └─> channel_selector.py (策略类)
       ├─> visualizers/
       │   ├─> base.py (BaseVisualizer)
       │   └─> plotly_viz.py (具体实现)
       ├─> plugins/
       │   ├─> base.py (Plugin基类)
       │   ├─> manager.py (PluginManager)
       │   └─> example_plugins.py (示例插件)
       ├─> ui/
       │   └─> menu.py (交互式菜单)
       └─> utils/
           └─> file_finder.py (文件查找)
```

---

## 核心模块说明

### 1. core/reader.py - COMTRADE读取器

**核心类**:
- `Channel`: 通道数据类，存储通道信息
- `ComtradeReader`: COMTRADE文件读取器

**主要功能**:
- 解析.cfg配置文件（支持多种中文编码）
- 读取.dat二进制数据文件
- 数据转换（原始值→实际值）

**扩展点**:
```python
# 如果需要支持其他格式，可以创建新的Reader类
class CustomFormatReader(ComtradeReader):
    def _parse_cfg(self):
        # 自定义配置文件解析逻辑
        pass

    def _read_dat(self):
        # 自定义数据文件读取逻辑
        pass
```

### 2. core/channel_selector.py - 通道选择策略

**基类**: `ChannelSelectionStrategy`

**内置策略**:
- `ImportantPatternSelector`: 智能选择重要通道
- `FirstNChannelsSelector`: 选择前N个通道
- `AllChannelsSelector`: 选择所有通道
- `PatternMatchSelector`: 基于模式匹配选择

### 3. visualizers/ - 可视化模块

**基类**: `BaseVisualizer`

**内置可视化器**:
- `PlotlyVisualizer`: 散点图可视化
- `PlotlyLineVisualizer`: 线图可视化

### 4. plugins/ - 插件系统

**基类**:
- `Plugin`: 所有插件的基类
- `DataProcessorPlugin`: 数据处理插件基类
- `ExportPlugin`: 导出插件基类

**插件管理器**: `PluginManager` (单例)

---

## 如何添加新的通道选择策略

### 步骤1: 创建新的策略类

在 `waveform_viewer/core/channel_selector.py` 中添加：

```python
class CustomChannelSelector(ChannelSelectionStrategy):
    """自定义通道选择器"""

    def __init__(self, custom_patterns: List[str]):
        """
        Args:
            custom_patterns: 自定义的匹配模式列表
        """
        self.custom_patterns = custom_patterns

    def select_channels(self, channels: List[Channel],
                       max_channels: int = 12) -> List[Channel]:
        """
        选择通道的具体实现

        Args:
            channels: 所有可用通道列表
            max_channels: 最大通道数

        Returns:
            选中的通道列表
        """
        selected = []

        # 你的选择逻辑
        for ch in channels:
            if any(pattern in ch.name for pattern in self.custom_patterns):
                selected.append(ch)
                if len(selected) >= max_channels:
                    break

        return selected
```

### 步骤2: 在应用中使用

```python
from waveform_viewer.app import WaveformViewerApp
from waveform_viewer.core.channel_selector import CustomChannelSelector

app = WaveformViewerApp("waves")

# 使用自定义选择器
custom_selector = CustomChannelSelector(['电压', '电流', '功率'])
app.set_channel_selector(custom_selector)

app.run()
```

### 示例：按单位选择通道

```python
class UnitBasedSelector(ChannelSelectionStrategy):
    """基于单位选择通道"""

    def __init__(self, units: List[str]):
        self.units = units

    def select_channels(self, channels: List[Channel],
                       max_channels: int = 12) -> List[Channel]:
        selected = []
        for ch in channels:
            if ch.unit in self.units:
                selected.append(ch)
        return selected[:max_channels]

# 使用：只选择电压和电流通道
selector = UnitBasedSelector(['V', 'kV', 'A', 'kA'])
```

---

## 如何添加新的可视化器

### 步骤1: 创建新的可视化器类

在 `waveform_viewer/visualizers/` 目录下创建新文件（例如 `matplotlib_viz.py`）：

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Matplotlib可视化器实现
"""

from typing import List
import matplotlib.pyplot as plt
from pathlib import Path

from .base import BaseVisualizer, VisualizationConfig
from ..core.reader import ComtradeReader, Channel


class MatplotlibVisualizer(BaseVisualizer):
    """
    Matplotlib可视化器

    使用Matplotlib创建静态图像
    """

    def __init__(self, config: VisualizationConfig = None):
        super().__init__(config)

    def visualize(self,
                  reader: ComtradeReader,
                  channels: List[Channel],
                  output_path: str) -> str:
        """创建Matplotlib图表"""

        if not self._validate_channels(channels):
            raise ValueError("通道列表为空或无效")

        num_plots = len(channels)
        fig, axes = plt.subplots(num_plots, 1, figsize=(12, num_plots * 3))

        if num_plots == 1:
            axes = [axes]

        for idx, (ax, channel) in enumerate(zip(axes, channels)):
            time_vals, data_vals = reader.get_analog_data(channel.index)

            if time_vals and data_vals:
                ax.plot(time_vals, data_vals, 'b-', linewidth=0.5)
                ax.set_ylabel(f"{channel.unit}" if channel.unit else "值")
                ax.set_title(f"{channel.name}")
                ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("时间 (秒)")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        return str(output_path)

    def get_supported_formats(self) -> List[str]:
        """支持的输出格式"""
        return ['png', 'pdf', 'svg', 'jpg']
```

### 步骤2: 在应用中使用

```python
from waveform_viewer.app import WaveformViewerApp
from waveform_viewer.visualizers.matplotlib_viz import MatplotlibVisualizer
from waveform_viewer.visualizers.base import VisualizationConfig

app = WaveformViewerApp("waves")

# 使用自定义可视化器
config = VisualizationConfig(height_per_plot=400)
visualizer = MatplotlibVisualizer(config)
app.set_visualizer(visualizer, config)

app.run()
```

---

## 如何开发插件

插件系统是本项目最强大的扩展机制。你可以开发各种类型的插件。

### 插件类型

1. **通用插件** (`Plugin`)
2. **数据处理插件** (`DataProcessorPlugin`)
3. **导出插件** (`ExportPlugin`)

### 示例1: 创建统计分析插件

在 `waveform_viewer/plugins/` 目录下创建 `my_plugins.py`：

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自定义插件示例
"""

from typing import Dict, Any
from .base import Plugin
from ..core.reader import ComtradeReader


class AdvancedStatisticsPlugin(Plugin):
    """
    高级统计分析插件

    计算更多统计指标：峰峰值、RMS、THD等
    """

    def __init__(self):
        super().__init__()
        self.description = "高级统计分析（峰峰值、RMS、THD等）"

    def execute(self, reader: ComtradeReader, context: Dict[str, Any]) -> Dict:
        """执行统计分析"""
        import numpy as np

        stats = {}

        for idx, channel in enumerate(reader.analog_channels):
            if idx >= len(reader.analog_data):
                continue

            data = np.array(reader.analog_data[idx])

            # 计算各种统计指标
            stats[channel.name] = {
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'rms': float(np.sqrt(np.mean(data**2))),
                'peak_to_peak': float(np.ptp(data)),
                'median': float(np.median(data)),
                'unit': channel.unit
            }

        return stats
```

### 示例2: 创建Excel导出插件

```python
class ExcelExportPlugin(ExportPlugin):
    """
    Excel导出插件

    将波形数据导出为Excel格式
    """

    def __init__(self):
        super().__init__()
        self.description = "将波形数据导出为Excel格式（.xlsx）"

    def execute(self, reader: ComtradeReader, context: Dict[str, Any]) -> str:
        """执行导出"""
        output_path = context.get('output_path', 'output.xlsx')
        return self.export(reader, output_path)

    def export(self, reader: ComtradeReader, output_path: str, **kwargs) -> str:
        """导出为Excel"""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("需要安装pandas和openpyxl: pip install pandas openpyxl")

        # 创建DataFrame
        data = {'时间(秒)': reader.time_values}

        for idx, channel in enumerate(reader.analog_channels):
            if idx < len(reader.analog_data):
                col_name = f"{channel.name} ({channel.unit})" if channel.unit else channel.name
                data[col_name] = reader.analog_data[idx]

        df = pd.DataFrame(data)

        # 导出到Excel
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='波形数据', index=False)

            # 添加元数据工作表
            metadata = pd.DataFrame({
                '项目': ['站点名称', '采样率', '样本数', '频率'],
                '值': [reader.station_name, reader.sample_rate,
                      reader.num_samples, reader.frequency]
            })
            metadata.to_excel(writer, sheet_name='元数据', index=False)

        return output_path

    def get_file_extension(self) -> str:
        return '.xlsx'
```

### 示例3: 创建FFT分析插件

```python
class FFTAnalysisPlugin(DataProcessorPlugin):
    """
    FFT频谱分析插件
    """

    def __init__(self):
        super().__init__()
        self.description = "对波形数据进行FFT频谱分析"

    def execute(self, reader: ComtradeReader, context: Dict[str, Any]) -> Dict:
        """执行FFT分析"""
        import numpy as np
        from scipy import signal

        results = {}

        for idx, channel in enumerate(reader.analog_channels[:10]):  # 只分析前10个通道
            if idx >= len(reader.analog_data):
                continue

            data = np.array(reader.analog_data[idx])

            # 执行FFT
            fft_result = np.fft.fft(data)
            freq = np.fft.fftfreq(len(data), 1.0 / reader.sample_rate)

            # 只保留正频率部分
            positive_freq_idx = freq > 0
            freq = freq[positive_freq_idx]
            magnitude = np.abs(fft_result[positive_freq_idx])

            # 找到主要频率分量
            top_indices = np.argsort(magnitude)[-5:][::-1]  # 前5个峰值

            results[channel.name] = {
                'frequencies': freq.tolist(),
                'magnitude': magnitude.tolist(),
                'dominant_frequencies': freq[top_indices].tolist(),
                'dominant_magnitudes': magnitude[top_indices].tolist(),
            }

        return results

    def process_data(self, time_values, data_values, **kwargs):
        """处理数据（接口要求）"""
        # 这里可以实现数据滤波等预处理
        return time_values, data_values
```

### 使用插件

```python
from waveform_viewer.app import WaveformViewerApp
from waveform_viewer.core.reader import ComtradeReader

# 方式1: 通过应用程序使用插件
app = WaveformViewerApp("waves")
reader = ComtradeReader("waves/某文件夹/waveform.cfg")

# 执行统计插件
stats = app.execute_plugin("AdvancedStatisticsPlugin", reader, context={})
print(stats)

# 导出Excel
app.execute_plugin("ExcelExportPlugin", reader,
                  context={'output_path': 'data.xlsx'})

# 方式2: 直接使用插件管理器
from waveform_viewer.plugins.manager import PluginManager

pm = PluginManager()
pm.add_plugin_directory("waveform_viewer/plugins")
pm.discover_plugins()

# 列出所有插件
pm.list_plugins()

# 执行插件
result = pm.execute_plugin("FFTAnalysisPlugin", reader, {})
```

---

## 如何扩展UI

### 添加新的菜单选项

修改 `waveform_viewer/app.py` 中的 `_configure_visualization` 方法：

```python
def _configure_visualization(self):
    """配置可视化选项"""
    print("\n" + "=" * 70)
    print("配置可视化选项")
    print("=" * 70)

    # 1. 选择通道选择策略
    strategies = [
        "智能选择（优先显示重要通道）",
        "显示前12个通道",
        "显示所有通道",
        "自定义模式选择",  # 新增选项
    ]

    self.menu.title = "选择通道选择策略:"
    idx, _ = self.menu.single_choice(strategies, default_index=0)

    if idx == 0:
        self.channel_selector = ImportantPatternSelector()
    elif idx == 1:
        self.channel_selector = FirstNChannelsSelector()
    elif idx == 2:
        self.channel_selector = AllChannelsSelector()
    elif idx == 3:
        # 自定义模式
        patterns = input("请输入要匹配的关键词（逗号分隔）: ").split(',')
        patterns = [p.strip() for p in patterns]
        self.channel_selector = PatternMatchSelector(patterns)

    # ... 其他配置 ...
```

### 创建自定义交互流程

```python
class CustomWaveformApp(WaveformViewerApp):
    """自定义应用程序"""

    def run(self):
        """自定义运行流程"""
        print("欢迎使用自定义波形分析工具！")

        # 1. 选择分析模式
        modes = [
            "标准可视化",
            "频谱分析",
            "统计分析",
            "导出数据"
        ]

        idx, mode = self.menu.single_choice(modes)

        if idx == 0:
            super().run()  # 调用父类的标准流程
        elif idx == 1:
            self._run_fft_analysis()
        elif idx == 2:
            self._run_statistics()
        elif idx == 3:
            self._run_export()

    def _run_fft_analysis(self):
        """执行FFT分析"""
        # 你的自定义逻辑
        pass
```

---

## 测试指南

### 单元测试示例

创建 `tests/test_channel_selector.py`:

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from waveform_viewer.core.reader import Channel
from waveform_viewer.core.channel_selector import (
    ImportantPatternSelector,
    FirstNChannelsSelector
)


class TestChannelSelector(unittest.TestCase):
    """通道选择器测试"""

    def setUp(self):
        """准备测试数据"""
        self.channels = [
            Channel(1, "A相电压", "", "kV", 1.0, 0.0),
            Channel(2, "B相电压", "", "kV", 1.0, 0.0),
            Channel(3, "A相电流", "", "A", 1.0, 0.0),
            Channel(4, "有功功率", "", "MW", 1.0, 0.0),
            Channel(5, "频率", "", "Hz", 1.0, 0.0),
            Channel(6, "其他参数", "", "", 1.0, 0.0),
        ]

    def test_important_pattern_selector(self):
        """测试智能选择器"""
        selector = ImportantPatternSelector()
        selected = selector.select_channels(self.channels, max_channels=10)

        # 验证选择了重要通道
        names = [ch.name for ch in selected]
        self.assertIn("A相电压", names)
        self.assertIn("有功功率", names)
        self.assertIn("频率", names)

    def test_first_n_selector(self):
        """测试前N个选择器"""
        selector = FirstNChannelsSelector()
        selected = selector.select_channels(self.channels, max_channels=3)

        self.assertEqual(len(selected), 3)
        self.assertEqual(selected[0].name, "A相电压")


if __name__ == '__main__':
    unittest.main()
```

### 集成测试示例

创建 `tests/test_integration.py`:

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from pathlib import Path
from waveform_viewer.core.reader import ComtradeReader
from waveform_viewer.core.channel_selector import ImportantPatternSelector
from waveform_viewer.visualizers.plotly_viz import PlotlyVisualizer


class TestIntegration(unittest.TestCase):
    """集成测试"""

    def test_full_workflow(self):
        """测试完整工作流程"""
        # 假设有测试数据
        cfg_file = "test_data/test.cfg"

        if not Path(cfg_file).exists():
            self.skipTest("测试数据不存在")

        # 1. 读取文件
        reader = ComtradeReader(cfg_file)
        self.assertGreater(len(reader.analog_channels), 0)

        # 2. 选择通道
        selector = ImportantPatternSelector()
        channels = selector.select_channels(reader.analog_channels)
        self.assertGreater(len(channels), 0)

        # 3. 生成可视化
        visualizer = PlotlyVisualizer()
        output = visualizer.visualize(reader, channels, "test_output.html")

        self.assertTrue(Path(output).exists())

        # 清理
        Path(output).unlink()


if __name__ == '__main__':
    unittest.main()
```

---

## 最佳实践

### 1. 代码风格

- 遵循PEP 8编码规范
- 使用类型提示（Type Hints）
- 编写清晰的docstring

```python
def process_data(self, time_values: List[float],
                data_values: List[float],
                **kwargs) -> Tuple[List[float], List[float]]:
    """
    处理数据

    Args:
        time_values: 时间值列表
        data_values: 数据值列表
        **kwargs: 其他参数

    Returns:
        处理后的 (time_values, data_values)

    Raises:
        ValueError: 如果输入数据无效
    """
    pass
```

### 2. 错误处理

```python
try:
    reader = ComtradeReader(cfg_file)
except FileNotFoundError:
    print(f"错误: 文件不存在 - {cfg_file}")
    return
except Exception as e:
    print(f"错误: 读取文件时出错 - {e}")
    import traceback
    traceback.print_exc()
    return
```

### 3. 性能优化

- 对于大文件，考虑使用生成器
- 缓存计算结果
- 使用NumPy进行批量计算

```python
# 好的做法：使用NumPy
import numpy as np
data_array = np.array(data_values)
result = np.mean(data_array)

# 避免：使用Python循环
result = sum(data_values) / len(data_values)
```

### 4. 文档

- 每个模块都应有模块级docstring
- 所有公共类和函数都应有文档
- 复杂逻辑应添加注释

### 5. 版本控制

使用Git进行版本控制：

```bash
# 提交新功能
git add waveform_viewer/plugins/my_plugin.py
git commit -m "添加新的FFT分析插件"

# 创建功能分支
git checkout -b feature/new-visualizer
```

---

## 扩展示例：完整的自定义可视化器

下面是一个完整的示例，展示如何创建一个基于Matplotlib的3D可视化器：

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D可视化器示例
"""

from typing import List
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from waveform_viewer.visualizers.base import BaseVisualizer, VisualizationConfig
from waveform_viewer.core.reader import ComtradeReader, Channel


class WaveformVisualizer3D(BaseVisualizer):
    """
    3D波形可视化器

    将多个通道以3D形式展示（时间、通道、幅值）
    """

    def __init__(self, config: VisualizationConfig = None):
        super().__init__(config)

    def visualize(self,
                  reader: ComtradeReader,
                  channels: List[Channel],
                  output_path: str) -> str:
        """创建3D可视化"""

        if not self._validate_channels(channels):
            raise ValueError("通道列表为空或无效")

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 为每个通道绘制3D线条
        for ch_idx, channel in enumerate(channels[:10]):  # 最多10个通道
            time_vals, data_vals = reader.get_analog_data(channel.index)

            if time_vals and data_vals:
                time_array = np.array(time_vals)
                data_array = np.array(data_vals)
                channel_array = np.full_like(time_array, ch_idx)

                ax.plot(time_array, channel_array, data_array,
                       label=channel.name, linewidth=0.5)

        ax.set_xlabel('时间 (秒)')
        ax.set_ylabel('通道')
        ax.set_zlabel('幅值')
        ax.set_title('波形数据 3D 可视化')
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        return str(output_path)

    def get_supported_formats(self) -> List[str]:
        return ['png', 'pdf']
```

---

## 常见问题解答

**Q: 如何调试插件？**

A: 在插件的 `execute` 方法中添加打印语句或使用Python调试器：

```python
def execute(self, reader, context):
    import pdb; pdb.set_trace()  # 设置断点
    # 你的代码
```

**Q: 如何让插件支持配置？**

A: 通过 `context` 参数传递配置：

```python
def execute(self, reader, context):
    window_size = context.get('window_size', 5)
    threshold = context.get('threshold', 0.1)
    # 使用配置
```

**Q: 如何处理大文件？**

A: 考虑分块读取或使用生成器：

```python
def read_large_file(self, file_path, chunk_size=1000):
    """分块读取大文件"""
    for chunk_start in range(0, self.num_samples, chunk_size):
        chunk_end = min(chunk_start + chunk_size, self.num_samples)
        yield self._read_chunk(chunk_start, chunk_end)
```

---

## 联系与支持

如有问题或建议，请联系我微信: SoberPin。

Happy Coding! 🚀
