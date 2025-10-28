# COMTRADE波形可视化工具 v2.0.3

一个功能强大、可扩展的COMTRADE波形文件可视化工具，采用现代设计模式构建，支持插件化开发。

**v2.0.3 新功能**: 性能优化，解决大量通道时的卡顿和空白问题！

## 功能特性

- 📊 **交互式可视化**: 基于Plotly创建交互式HTML波形图表
- 🎯 **智能通道选择**: 自动识别并优先显示重要通道（电压、电流、功率、频率、角度、AVR、PSS等）
- 🔌 **插件系统**: 支持自定义插件扩展功能（CSV导出、JSON导出、数据滤波等）
- 🎨 **多种可视化样式**: 支持散点图、线图等多种显示方式
- ⌨️ **交互式菜单**: 支持方向键导航和多选（或数字输入模式）
- 📁 **批量处理**: 支持同时处理多个波形文件
- 🌐 **自动打开**: 生成后自动在浏览器中打开，无需手动查找文件（可选）
- 🏗️ **设计模式**: 采用Strategy、Plugin、Factory、Singleton等设计模式

## 快速开始

### 1. 安装依赖

```bash
pip install plotly numpy
```

或使用：
```bash
pip install -r requirements.txt
```

### 2. 准备波形数据

将COMTRADE波形文件（.cfg和.dat文件）放入 `waves/` 目录下的子文件夹中：

```
waves/
├── 机励磁波形2025.1013/
│   ├── waveform1.cfg
│   ├── waveform1.dat
│   ├── waveform2.cfg
│   └── waveform2.dat
└── 其他波形数据/
    └── ...
```

### 3. 运行程序

**交互式菜单模式（推荐）**：
```bash
python main.py
```

使用方向键（↑↓）选择，空格键多选，回车确认。

**简单菜单模式**：
```bash
python main.py --simple
```

使用数字输入选择，适用于不支持方向键的终端环境。

**传统方式（兼容旧版本）**：
```bash
# 使用运行脚本
./run_viewer.sh

# 或直接运行旧版脚本
python waveform_viewer.py
```

### 4. 使用流程

1. **选择文件夹**: 从可用的波形文件夹中选择要处理的文件夹
2. **选择文件**: 从每个文件夹中选择要可视化的波形文件
3. **配置可视化**:
   - 选择通道选择策略（智能选择/前N个/全部）
   - 选择可视化样式（散点图/线图）
4. **查看结果**: 程序会生成HTML文件，保存在项目根目录

## 文件说明

### 主要文件
- `main.py` - **新版主启动脚本**（推荐使用）
- `waveform_viewer.py` - 旧版脚本（仍可使用，保持兼容）
- `verify_data.py` - 数据验证脚本
- `README.md` - 本文件
- `DEVELOPER_GUIDE.md` - **开发者指南**（如何扩展功能、开发插件）
- `requirements.txt` - Python依赖包列表

### 旧版文档（参考）
- `analysis.md` - COMTRADE文件格式详细分析文档
- `ENCODING_FIX.md` - 编码问题修复说明文档
- `run_viewer.sh` - 运行脚本（Mac/Linux）

### 数据验证

使用验证脚本检查波形数据：

```bash
python verify_data.py
```

验证脚本会输出：
- 采样率和样本数信息
- 时间跨度和采样间隔
- 各通道的最小值、最大值、平均值
- 数据质量检查（NaN、Inf、常数值等）
- 采样点示例数据

## COMTRADE文件格式说明

COMTRADE（Common format for Transient Data Exchange）是电力系统中常用的波形记录格式，通常包含：

- `.cfg` - **配置文件**（必需）：定义通道信息和采样参数（文本格式，GBK编码）
- `.dat` - **数据文件**（必需）：包含实际的波形数据（二进制格式）
- `.hdr` - **头文件**（可选）：包含故障和事件信息（XML格式，UTF-8编码）
- `.rpt` - **报告文件**（可选）：包含设备专有的详细报告（专有二进制格式）

### 📚 深入了解

想要深入了解每种文件的详细格式、数据结构和解析方法，请阅读：
**👉 [analysis.md - COMTRADE文件格式详细分析](./docs/analysis.md)**

该文档包含：
- 每种文件的详细结构说明
- 二进制数据的字节级解析
- 数据转换公式和示例
- 文件关系图和使用流程
- 常见问题与解决方案

## 输出文件

程序会为每个COMTRADE文件组生成一个HTML可视化文件，命名格式为：
```
原文件名_visualization.html
```

这些HTML文件可以直接在浏览器中打开，无需安装任何额外软件。

## 项目结构

```
波形记录/
├── main.py                     # 新版主启动脚本
├── waveform_viewer.py          # 旧版脚本（兼容）
├── verify_data.py              # 数据验证脚本
├── README.md                   # 使用文档
├── DEVELOPER_GUIDE.md         # 开发者指南
├── requirements.txt            # 依赖包列表
├── waves/                      # 波形文件存放目录
│   └── 机励磁波形2025.1013/   # 波形数据文件夹
│       ├── *.cfg              # COMTRADE配置文件
│       └── *.dat              # COMTRADE数据文件
└── waveform_viewer/           # 主应用模块
    ├── app.py                 # 主应用程序
    ├── core/                  # 核心功能模块
    │   ├── reader.py          # COMTRADE读取器
    │   └── channel_selector.py # 通道选择策略
    ├── visualizers/           # 可视化模块
    │   ├── base.py           # 可视化基类
    │   └── plotly_viz.py     # Plotly可视化实现
    ├── plugins/               # 插件系统
    │   ├── base.py           # 插件基类
    │   ├── manager.py        # 插件管理器
    │   └── example_plugins.py # 示例插件
    ├── ui/                    # 用户界面模块
    │   └── menu.py           # 交互式菜单
    └── utils/                 # 工具模块
        └── file_finder.py    # 文件查找器
```

## 编程式使用

除了交互式界面，你也可以在代码中使用这个库：

```python
from pathlib import Path
from waveform_viewer.core.reader import ComtradeReader
from waveform_viewer.core.channel_selector import ImportantPatternSelector
from waveform_viewer.visualizers.plotly_viz import PlotlyVisualizer
from waveform_viewer.visualizers.base import VisualizationConfig

# 读取COMTRADE文件
reader = ComtradeReader("waves/某文件夹/waveform.cfg")

# 选择通道
selector = ImportantPatternSelector()
channels = selector.select_channels(reader.analog_channels, max_channels=12)

# 创建可视化
config = VisualizationConfig(height_per_plot=300)
visualizer = PlotlyVisualizer(config)
output_path = visualizer.visualize(reader, channels, "output.html")

print(f"可视化已保存到: {output_path}")
```

## 系统要求

- Python 3.6+
- plotly库
- numpy库

## 技术细节

### 设计模式

本项目采用了多种设计模式，使代码易于扩展和维护：

1. **Strategy Pattern (策略模式)** - 通道选择和可视化策略
2. **Plugin Architecture (插件架构)** - 可扩展的插件系统
3. **Factory Pattern (工厂模式)** - 对象创建
4. **Singleton Pattern (单例模式)** - 插件管理器

### 时间戳解析
- COMTRADE二进制文件中的时间戳以**微秒**为单位存储
- 程序自动转换为秒用于显示

### 数据转换
- 原始16位整数值通过线性公式转换：`实际值 = a × 原始值 + b`
- 转换系数存储在.cfg配置文件中

### 采样信息
- 标准采样率：300 Hz (每3.33ms一个样本)
- 每个文件通常包含6000个样本点（20秒数据）

## 中文编码支持

本工具已正确处理中文编码问题：

- ✅ 自动识别GBK/GB2312/GB18030编码的COMTRADE文件
- ✅ 正确显示中文通道名称和单位
- ✅ HTML文件中的中文完全可读
- ✅ 终端输出的中文正常显示

**支持的编码格式**（按优先级）：
1. GBK - Windows简体中文（推荐）
2. GB2312 - 简体中文基本字符集
3. GB18030 - 中国国家标准（最全）
4. UTF-8 - 国际标准

## 常见问题

**Q: 如何添加自定义通道选择模式？**
A: 继承 `ChannelSelectionStrategy` 类并实现 `select_channels` 方法。详见 DEVELOPER_GUIDE.md

**Q: 如何开发自定义插件？**
A: 继承 `Plugin` 基类并实现 `execute` 方法。详见 DEVELOPER_GUIDE.md

**Q: 生成的HTML文件在哪里？**
A: 默认保存在项目根目录，文件名格式为 `{原文件名}_visualization.html`

**Q: 交互式菜单不工作怎么办？**
A: 使用 `python main.py --simple` 切换到数字输入模式

**Q: 如何导出其他格式（如PNG、PDF）？**
A: 可以通过开发新的可视化插件或使用 Plotly 的导出功能

## 故障排除

如果遇到问题：

1. 确保文件夹中包含完整的COMTRADE文件（.cfg和.dat文件）
2. **中文乱码**: 程序已自动处理GBK编码，如果仍有问题请查看 `ENCODING_FIX.md`
3. 查看终端输出的错误信息和调试信息
4. 确保 `waves/` 目录存在并包含波形数据子文件夹

## 版本历史

### v2.0.0 (2025-01)
- ✨ 完全重构代码，采用设计模式
- ✨ 添加插件系统
- ✨ 添加交互式菜单
- ✨ 支持批量处理
- ✨ 改进通道选择逻辑
- 📦 模块化代码结构

### v1.0.0 (2025-01)
- 初始版本
- 基本的COMTRADE读取和可视化功能

## 贡献与开发

想要扩展功能或开发插件？请查看 **DEVELOPER_GUIDE.md** 获取详细的开发指南。

## 许可证

本项目仅供内部使用。
