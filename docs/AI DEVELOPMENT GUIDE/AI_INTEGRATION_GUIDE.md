# AI 专家系统集成指南

## 目录
- [概述](#概述)
- [系统架构](#系统架构)
- [快速开始](#快速开始)
- [工具函数开发](#工具函数开发)
- [LLM 集成](#llm-集成)
- [最佳实践](#最佳实践)
- [常见问题](#常见问题)

---

## 概述

本文档介绍如何在 ComtradeReader 项目中集成大模型驱动的电力波形分析专家系统。该系统允许用户通过自然语言查询波形数据，AI 模型会自动调用相应的分析工具并给出专业解释。

### 核心特性

- **自然语言接口**：用户可以用自然语言提问，无需了解技术细节
- **工具调用能力**：AI 模型可以自动选择和调用各种波形分析工具
- **专家级解释**：不仅提供分析结果，还给出专业的工程解释
- **可扩展架构**：基于插件系统，易于添加新的分析工具

### 适用场景

- 故障录波快速诊断
- 批量波形自动分析
- 保护动作正确性评估
- 电能质量评估
- 教学和培训辅助

---

## 系统架构

### 三层架构设计

```
┌─────────────────────────────────────────────────────────┐
│              Layer 3: AI Expert Interface               │
│                                                         │
│  ┌─────────────┐         ┌──────────────┐             │
│  │WaveformExpert│ ◄───── │ LLM Client   │             │
│  │  (专家系统)  │         │ (GPT/Claude) │             │
│  └─────────────┘         └──────────────┘             │
│         │                                               │
│         │ 调用                                          │
│         ▼                                               │
└─────────────────────────────────────────────────────────┘
         │
         │ Function Calling
         ▼
┌─────────────────────────────────────────────────────────┐
│           Layer 2: Analysis Tool Registry               │
│                                                         │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐       │
│  │Data Access │  │Basic       │  │Advanced    │       │
│  │Tools       │  │Analysis    │  │Analysis    │       │
│  └────────────┘  └────────────┘  └────────────┘       │
│                                                         │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐       │
│  │Event       │  │Comparison  │  │Reporting   │       │
│  │Analysis    │  │Tools       │  │Tools       │       │
│  └────────────┘  └────────────┘  └────────────┘       │
│                                                         │
│              ToolRegistry (工具注册中心)                 │
└─────────────────────────────────────────────────────────┘
         │
         │ 调用现有 API
         ▼
┌─────────────────────────────────────────────────────────┐
│         Layer 1: Existing Core Components               │
│                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │ComtradeReader│  │HdrReader    │  │RptReader    │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
│                                                         │
│  ┌─────────────┐  ┌─────────────┐                      │
│  │Visualizers  │  │Plugins      │                      │
│  └─────────────┘  └─────────────┘                      │
│                                                         │
│              (现有代码，保持不变)                         │
└─────────────────────────────────────────────────────────┘
```

### 目录结构

```
waveform_viewer/
├── ai/                              # AI 专家系统模块（新增）
│   ├── __init__.py
│   ├── expert.py                    # WaveformExpert 主类
│   ├── tool_registry.py             # 工具注册表
│   ├── function_schemas.py          # LLM Function Calling Schemas
│   ├── config.py                    # AI 配置管理
│   └── tools/                       # 工具函数库
│       ├── __init__.py
│       ├── data_access.py           # 数据访问工具
│       ├── basic_analysis.py        # 基础分析工具
│       ├── advanced_analysis.py     # 高级分析工具
│       ├── event_analysis.py        # 事件分析工具
│       ├── comparison.py            # 对比分析工具
│       └── reporting.py             # 报告生成工具
├── core/                            # 核心功能（现有）
├── visualizers/                     # 可视化模块（现有）
├── plugins/                         # 插件系统（现有）
└── ...
```

---

## 快速开始

### 1. 安装依赖

```bash
# 安装 AI 相关依赖
pip install openai anthropic scipy scikit-learn pydantic

# 或使用 requirements 文件
pip install -r requirements-ai.txt
```

### 2. 配置 API Key

创建 `.env` 文件（不要提交到 Git）：

```bash
# OpenAI API（如果使用 GPT）
OPENAI_API_KEY=sk-...

# Anthropic API（如果使用 Claude）
ANTHROPIC_API_KEY=sk-ant-...

# 选择使用的模型
AI_MODEL_PROVIDER=openai  # 或 anthropic
AI_MODEL_NAME=gpt-4       # 或 claude-3-5-sonnet-20241022
```

### 3. 最小示例

```python
from waveform_viewer.ai.expert import WaveformExpert
from waveform_viewer.ai.tool_registry import get_default_registry
import openai

# 初始化 LLM 客户端
openai.api_key = "your-api-key"

# 创建专家系统
registry = get_default_registry()
expert = WaveformExpert(openai, registry)

# 分析查询
result = expert.analyze(
    query="这个波形有电压跌落吗？",
    cfg_path="waves/20241030_115240/20241030_115240.cfg"
)

print(result)
```

### 4. 命令行使用

```bash
# 启动 AI 模式
python main.py --ai

# 或指定文件直接分析
python main.py --ai --file waves/20241030_115240/20241030_115240.cfg
```

---

## 工具函数开发

### 工具函数规范

每个工具函数应该遵循以下规范：

#### 1. 函数签名

```python
def tool_name(cfg_path: str, param1: type, param2: type = default) -> Dict[str, Any]:
    """
    工具函数的简短描述（会被 LLM 看到）

    Args:
        cfg_path: COMTRADE 配置文件路径
        param1: 参数1的描述
        param2: 参数2的描述（可选）

    Returns:
        包含分析结果的字典，必须可以 JSON 序列化

    Raises:
        ValueError: 参数无效时
        FileNotFoundError: 文件不存在时
    """
    pass
```

#### 2. 返回格式

工具函数**必须**返回可 JSON 序列化的 Python 字典：

```python
# ✅ 正确示例
return {
    "status": "success",
    "data": {
        "min": 10.5,
        "max": 100.2,
        "mean": 55.3
    },
    "metadata": {
        "channel_name": "A相电压",
        "unit": "%"
    }
}

# ❌ 错误示例
return np.array([1, 2, 3])  # NumPy 数组不能直接序列化
return pd.DataFrame(...)     # DataFrame 不能直接序列化
```

#### 3. 错误处理

所有工具函数都应该捕获异常并返回错误信息：

```python
def my_tool(cfg_path: str) -> Dict[str, Any]:
    try:
        reader = ComtradeReader(cfg_path)
        # ... 分析逻辑
        return {"status": "success", "data": result}

    except FileNotFoundError:
        return {
            "status": "error",
            "error_type": "FileNotFoundError",
            "message": f"文件不存在: {cfg_path}"
        }

    except Exception as e:
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "message": str(e)
        }
```

### 工具函数示例

#### 数据访问工具

```python
# waveform_viewer/ai/tools/data_access.py

from typing import Dict, Any, List
from waveform_viewer.core.reader import ComtradeReader

def list_available_channels(cfg_path: str) -> Dict[str, Any]:
    """
    列出 COMTRADE 文件中所有可用的通道信息

    Args:
        cfg_path: COMTRADE 配置文件路径

    Returns:
        包含站点信息和通道列表的字典
    """
    try:
        reader = ComtradeReader(cfg_path)

        return {
            "status": "success",
            "station_name": reader.station_name,
            "frequency": reader.frequency,
            "sample_rate": reader.sample_rate,
            "num_samples": reader.num_samples,
            "duration": round(reader.time_values[-1], 3),
            "analog_channels": [
                {
                    "index": ch.index,
                    "name": ch.name,
                    "phase": ch.phase,
                    "unit": ch.unit,
                    "circuit_component": ch.circuit_component
                }
                for ch in reader.analog_channels
            ],
            "digital_channels": [
                {
                    "index": ch["index"],
                    "name": ch["name"]
                }
                for ch in reader.digital_channels
            ]
        }

    except Exception as e:
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "message": str(e)
        }


def get_channel_data(cfg_path: str, channel_name: str) -> Dict[str, Any]:
    """
    获取指定通道的时序数据

    Args:
        cfg_path: COMTRADE 配置文件路径
        channel_name: 通道名称或匹配模式（如 "A相电压"）

    Returns:
        包含时间序列和数据值的字典
    """
    try:
        reader = ComtradeReader(cfg_path)
        channel = reader.get_channel_by_name(channel_name)

        if not channel:
            return {
                "status": "error",
                "message": f"未找到通道: {channel_name}"
            }

        data = reader.get_analog_data(channel.index)

        # 限制返回的数据点数量（避免 token 过多）
        max_points = 1000
        step = max(1, len(data) // max_points)

        return {
            "status": "success",
            "channel": {
                "name": channel.name,
                "unit": channel.unit,
                "phase": channel.phase
            },
            "time": reader.time_values[::step],
            "values": data[::step],
            "sample_count": len(data),
            "downsampled": step > 1
        }

    except Exception as e:
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "message": str(e)
        }
```

#### 基础分析工具

```python
# waveform_viewer/ai/tools/basic_analysis.py

import numpy as np
from typing import Dict, Any, List
from waveform_viewer.core.reader import ComtradeReader

def calculate_statistics(cfg_path: str, channel_name: str) -> Dict[str, Any]:
    """
    计算指定通道的统计特征

    Args:
        cfg_path: COMTRADE 配置文件路径
        channel_name: 通道名称

    Returns:
        包含统计指标的字典（最小值、最大值、均值、标准差、RMS等）
    """
    try:
        reader = ComtradeReader(cfg_path)
        channel = reader.get_channel_by_name(channel_name)

        if not channel:
            return {
                "status": "error",
                "message": f"未找到通道: {channel_name}"
            }

        data = np.array(reader.get_analog_data(channel.index))

        return {
            "status": "success",
            "channel": {
                "name": channel.name,
                "unit": channel.unit
            },
            "statistics": {
                "min": float(np.min(data)),
                "max": float(np.max(data)),
                "mean": float(np.mean(data)),
                "median": float(np.median(data)),
                "std": float(np.std(data)),
                "rms": float(np.sqrt(np.mean(data**2))),
                "peak_to_peak": float(np.ptp(data)),
                "variance": float(np.var(data))
            }
        }

    except Exception as e:
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "message": str(e)
        }


def detect_voltage_sags(cfg_path: str, threshold_percent: float = 90.0) -> Dict[str, Any]:
    """
    检测电压跌落事件

    Args:
        cfg_path: COMTRADE 配置文件路径
        threshold_percent: 电压跌落阈值（百分比），默认90%

    Returns:
        包含检测到的电压跌落事件列表
    """
    try:
        reader = ComtradeReader(cfg_path)

        # 查找所有电压通道
        voltage_channels = [
            ch for ch in reader.analog_channels
            if '电压' in ch.name and ch.unit == '%'
        ]

        if not voltage_channels:
            return {
                "status": "error",
                "message": "未找到电压通道"
            }

        events = []

        for ch in voltage_channels:
            data = np.array(reader.get_analog_data(ch.index))
            time = np.array(reader.time_values)

            # 找到低于阈值的点
            sag_mask = data < threshold_percent

            if np.any(sag_mask):
                # 找到连续的跌落区间
                sag_indices = np.where(sag_mask)[0]

                # 简单处理：找到第一个跌落点
                start_idx = sag_indices[0]
                end_idx = sag_indices[-1]

                events.append({
                    "channel": ch.name,
                    "phase": ch.phase,
                    "start_time": float(time[start_idx]),
                    "end_time": float(time[end_idx]),
                    "duration": float(time[end_idx] - time[start_idx]),
                    "min_value": float(np.min(data[sag_indices])),
                    "threshold": threshold_percent
                })

        return {
            "status": "success",
            "events": events,
            "event_count": len(events)
        }

    except Exception as e:
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "message": str(e)
        }


def find_peaks(cfg_path: str, channel_name: str,
               prominence: float = None) -> Dict[str, Any]:
    """
    检测通道数据中的峰值点

    Args:
        cfg_path: COMTRADE 配置文件路径
        channel_name: 通道名称
        prominence: 峰值显著性阈值（可选）

    Returns:
        包含峰值点信息的字典
    """
    try:
        from scipy.signal import find_peaks as scipy_find_peaks

        reader = ComtradeReader(cfg_path)
        channel = reader.get_channel_by_name(channel_name)

        if not channel:
            return {
                "status": "error",
                "message": f"未找到通道: {channel_name}"
            }

        data = np.array(reader.get_analog_data(channel.index))
        time = np.array(reader.time_values)

        # 自动计算 prominence
        if prominence is None:
            prominence = (np.max(data) - np.min(data)) * 0.1

        peaks, properties = scipy_find_peaks(data, prominence=prominence)

        return {
            "status": "success",
            "channel": {
                "name": channel.name,
                "unit": channel.unit
            },
            "peaks": [
                {
                    "time": float(time[idx]),
                    "value": float(data[idx]),
                    "prominence": float(properties["prominences"][i])
                }
                for i, idx in enumerate(peaks[:20])  # 限制返回前20个峰值
            ],
            "total_peaks": len(peaks)
        }

    except ImportError:
        return {
            "status": "error",
            "message": "需要安装 scipy 库: pip install scipy"
        }
    except Exception as e:
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "message": str(e)
        }
```

### 注册工具函数

```python
# waveform_viewer/ai/tool_registry.py

from typing import Dict, Callable, Any, List
from .tools import data_access, basic_analysis

class ToolRegistry:
    """工具函数注册中心"""

    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._schemas: Dict[str, Dict] = {}

    def register(self, func: Callable, schema: Dict):
        """
        注册工具函数

        Args:
            func: 工具函数
            schema: OpenAI Function Calling 格式的 schema
        """
        self._tools[func.__name__] = func
        self._schemas[func.__name__] = schema

    def get_tool(self, name: str) -> Callable:
        """获取工具函数"""
        return self._tools.get(name)

    def get_all_schemas(self) -> List[Dict]:
        """获取所有工具的 schemas（用于 LLM Function Calling）"""
        return list(self._schemas.values())

    def execute(self, tool_name: str, **kwargs) -> Any:
        """
        执行工具函数

        Args:
            tool_name: 工具名称
            **kwargs: 工具参数

        Returns:
            工具执行结果
        """
        tool = self.get_tool(tool_name)
        if not tool:
            return {
                "status": "error",
                "message": f"工具 {tool_name} 不存在"
            }

        try:
            return tool(**kwargs)
        except Exception as e:
            return {
                "status": "error",
                "error_type": type(e).__name__,
                "message": str(e)
            }


def get_default_registry() -> ToolRegistry:
    """获取预注册了所有内置工具的注册表"""
    from .function_schemas import TOOL_SCHEMAS

    registry = ToolRegistry()

    # 注册所有工具
    tool_modules = [data_access, basic_analysis]

    for schema in TOOL_SCHEMAS:
        tool_name = schema["name"]

        # 从模块中查找对应的函数
        for module in tool_modules:
            if hasattr(module, tool_name):
                func = getattr(module, tool_name)
                registry.register(func, schema)
                break

    return registry
```

---

## LLM 集成

### OpenAI GPT 集成

```python
# waveform_viewer/ai/expert.py

import json
from typing import List, Dict, Any
import openai

class WaveformExpert:
    """电力波形分析专家系统"""

    def __init__(self, openai_client, tool_registry, model="gpt-4"):
        """
        初始化专家系统

        Args:
            openai_client: OpenAI 客户端（已配置 API key）
            tool_registry: ToolRegistry 实例
            model: 使用的模型名称
        """
        self.openai = openai_client
        self.registry = tool_registry
        self.model = model
        self.conversation_history = []

    def analyze(self, query: str, cfg_path: str, max_iterations: int = 10) -> str:
        """
        分析用户查询

        Args:
            query: 用户的自然语言问题
            cfg_path: COMTRADE 文件路径
            max_iterations: 最大工具调用迭代次数

        Returns:
            专家系统的分析结果
        """
        # 系统提示词
        system_prompt = self._build_system_prompt(cfg_path)

        # 构建消息
        messages = [
            {"role": "system", "content": system_prompt},
            *self.conversation_history,
            {"role": "user", "content": query}
        ]

        iteration = 0

        while iteration < max_iterations:
            # 调用 LLM
            response = self.openai.chat.completions.create(
                model=self.model,
                messages=messages,
                functions=self.registry.get_all_schemas(),
                function_call="auto",
                temperature=0.1  # 降低温度以获得更确定的结果
            )

            message = response.choices[0].message

            # 如果不需要调用函数，返回最终答案
            if not message.function_call:
                answer = message.content

                # 更新对话历史
                self.conversation_history.append({
                    "role": "user",
                    "content": query
                })
                self.conversation_history.append({
                    "role": "assistant",
                    "content": answer
                })

                return answer

            # 调用函数
            func_call = message.function_call
            func_name = func_call.name
            func_args = json.loads(func_call.arguments)

            # 自动注入 cfg_path（如果需要）
            if "cfg_path" in func_args and not func_args.get("cfg_path"):
                func_args["cfg_path"] = cfg_path

            # 执行工具
            print(f"  → 调用工具: {func_name}({func_args})")
            result = self.registry.execute(func_name, **func_args)

            # 将函数调用和结果添加到消息历史
            messages.append({
                "role": "assistant",
                "content": None,
                "function_call": {
                    "name": func_name,
                    "arguments": json.dumps(func_args, ensure_ascii=False)
                }
            })
            messages.append({
                "role": "function",
                "name": func_name,
                "content": json.dumps(result, ensure_ascii=False, indent=2)
            })

            iteration += 1

        return "抱歉，分析过程超时（达到最大迭代次数）"

    def _build_system_prompt(self, cfg_path: str) -> str:
        """构建系统提示词"""
        return f"""你是一个专业的电力系统暂态波形分析专家。你擅长分析 COMTRADE 格式的故障录波数据，能够：

1. 识别故障类型（单相接地、两相短路、三相短路等）
2. 分析电压跌落、电流突变、频率波动等异常现象
3. 评估保护装置动作的正确性和及时性
4. 进行频域分析（谐波、FFT）
5. 提供专业的工程解释和建议

当前分析的文件：{cfg_path}

请根据用户的问题，选择合适的工具进行分析。你可以多次调用不同的工具来获取完整信息。

注意事项：
- 如果用户的问题不够明确，先调用 list_available_channels 了解文件包含哪些通道
- 分析结果要结合电力系统的工程实际，给出专业解释
- 如果检测到异常，要说明可能的原因和影响
- 使用专业术语，但要确保表达清晰易懂
"""

    def reset_conversation(self):
        """重置对话历史"""
        self.conversation_history = []
```

### Anthropic Claude 集成

```python
# waveform_viewer/ai/expert_claude.py

import json
from typing import List, Dict, Any
import anthropic

class ClaudeWaveformExpert:
    """基于 Claude 的波形分析专家"""

    def __init__(self, anthropic_client, tool_registry,
                 model="claude-3-5-sonnet-20241022"):
        self.anthropic = anthropic_client
        self.registry = tool_registry
        self.model = model
        self.conversation_history = []

    def analyze(self, query: str, cfg_path: str, max_iterations: int = 10) -> str:
        """使用 Claude 进行分析"""
        system_prompt = self._build_system_prompt(cfg_path)

        messages = [
            *self.conversation_history,
            {"role": "user", "content": query}
        ]

        iteration = 0

        while iteration < max_iterations:
            response = self.anthropic.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system_prompt,
                messages=messages,
                tools=self._convert_schemas_to_claude_format(),
                temperature=0.1
            )

            # 检查是否需要调用工具
            if response.stop_reason == "end_turn":
                # 提取文本回答
                answer = ""
                for block in response.content:
                    if block.type == "text":
                        answer += block.text

                # 更新对话历史
                self.conversation_history.append({
                    "role": "user",
                    "content": query
                })
                self.conversation_history.append({
                    "role": "assistant",
                    "content": answer
                })

                return answer

            elif response.stop_reason == "tool_use":
                # 执行工具调用
                for block in response.content:
                    if block.type == "tool_use":
                        func_name = block.name
                        func_args = block.input

                        # 注入 cfg_path
                        if "cfg_path" not in func_args or not func_args["cfg_path"]:
                            func_args["cfg_path"] = cfg_path

                        print(f"  → 调用工具: {func_name}")
                        result = self.registry.execute(func_name, **func_args)

                        # 添加工具调用结果到消息
                        messages.append({
                            "role": "assistant",
                            "content": response.content
                        })
                        messages.append({
                            "role": "user",
                            "content": [{
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": json.dumps(result, ensure_ascii=False)
                            }]
                        })

            iteration += 1

        return "抱歉，分析过程超时"

    def _convert_schemas_to_claude_format(self) -> List[Dict]:
        """将 OpenAI 格式的 schema 转换为 Claude 格式"""
        claude_tools = []

        for schema in self.registry.get_all_schemas():
            claude_tools.append({
                "name": schema["name"],
                "description": schema["description"],
                "input_schema": schema["parameters"]
            })

        return claude_tools

    def _build_system_prompt(self, cfg_path: str) -> str:
        # 与 OpenAI 版本相同
        return f"""你是一个专业的电力系统暂态波形分析专家..."""

    def reset_conversation(self):
        self.conversation_history = []
```

---

## 最佳实践

### 1. 工具设计原则

#### 原子性
每个工具只做一件事：

```python
# ✅ 好的设计
def calculate_statistics(cfg_path, channel_name)
def detect_voltage_sags(cfg_path, threshold)

# ❌ 不好的设计
def analyze_everything(cfg_path)  # 功能过于宽泛
```

#### 无状态
工具函数应该是纯函数：

```python
# ✅ 好的设计
def get_channel_data(cfg_path: str, channel_name: str):
    reader = ComtradeReader(cfg_path)  # 每次都创建新实例
    return reader.get_analog_data(...)

# ❌ 不好的设计
class Analyzer:
    def __init__(self):
        self.cache = {}  # 有状态
```

#### 明确的错误处理

```python
def my_tool(cfg_path: str):
    try:
        # 处理逻辑
        return {"status": "success", "data": ...}
    except SpecificException as e:
        return {"status": "error", "message": str(e)}
```

### 2. Schema 设计技巧

```python
# waveform_viewer/ai/function_schemas.py

TOOL_SCHEMAS = [
    {
        "name": "detect_voltage_sags",
        "description": (
            "检测电压跌落事件。当用户询问'有没有电压跌落'、"
            "'电压是否正常'、'电压质量如何'时使用此工具。"
        ),  # 详细描述何时使用
        "parameters": {
            "type": "object",
            "properties": {
                "cfg_path": {
                    "type": "string",
                    "description": "COMTRADE 配置文件路径"
                },
                "threshold_percent": {
                    "type": "number",
                    "description": (
                        "电压跌落阈值（百分比），默认90。"
                        "对于高压系统，可以设置为85-90；"
                        "对于低压系统，可以设置为70-80。"
                    ),  # 给出参考值
                    "default": 90
                }
            },
            "required": ["cfg_path"]  # 只标记必需参数
        }
    }
]
```

### 3. 提示词优化

```python
def _build_system_prompt(self, cfg_path: str) -> str:
    return f"""你是电力系统暂态波形分析专家。

## 你的能力
- 故障类型识别（单相接地、两相短路、三相短路等）
- 电能质量分析（电压跌落、谐波、频率波动）
- 保护装置评估（动作正确性、时序分析）
- 频域分析（FFT、谐波、THD）

## 分析流程
1. 如果用户问题不明确，先用 list_available_channels 了解数据
2. 根据问题类型选择合适的分析工具
3. 可以多次调用工具，逐步深入分析
4. 给出专业解释，说明异常原因、影响、建议

## 当前文件
{cfg_path}

## 输出格式
使用清晰的结构化格式：
- **分析结果**：简洁总结
- **详细数据**：关键数值
- **专业解释**：工程原理
- **建议**：改进措施（如果需要）

开始分析吧！"""
```

### 4. 性能优化

#### 数据降采样

```python
def get_channel_data(cfg_path: str, channel_name: str):
    reader = ComtradeReader(cfg_path)
    data = reader.get_analog_data(channel.index)

    # 如果数据点太多，进行降采样
    if len(data) > 1000:
        step = len(data) // 1000
        data = data[::step]
        time = reader.time_values[::step]

    return {"time": time, "values": data}
```

#### 缓存机制

```python
from functools import lru_cache

@lru_cache(maxsize=10)
def _load_comtrade_reader(cfg_path: str) -> ComtradeReader:
    """缓存 ComtradeReader 实例"""
    return ComtradeReader(cfg_path)

def my_tool(cfg_path: str):
    reader = _load_comtrade_reader(cfg_path)  # 使用缓存
    # ...
```

### 5. 安全性考虑

#### 文件路径验证

```python
import os
from pathlib import Path

def validate_cfg_path(cfg_path: str) -> bool:
    """验证文件路径的安全性"""
    path = Path(cfg_path)

    # 检查文件存在
    if not path.exists():
        return False

    # 检查文件扩展名
    if path.suffix.lower() != '.cfg':
        return False

    # 防止路径遍历攻击
    if '..' in cfg_path:
        return False

    return True
```

#### API Key 管理

```python
# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class AIConfig:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

    @classmethod
    def validate(cls):
        if not cls.OPENAI_API_KEY and not cls.ANTHROPIC_API_KEY:
            raise ValueError("至少需要配置一个 API Key")
```

---

## 常见问题

### Q1: 工具调用失败，如何调试？

```python
# 在 expert.py 中添加日志
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def analyze(self, query: str, cfg_path: str):
    # ...
    logger.debug(f"调用工具: {func_name}")
    logger.debug(f"参数: {func_args}")
    result = self.registry.execute(func_name, **func_args)
    logger.debug(f"结果: {result}")
```

### Q2: LLM 没有调用正确的工具？

可能原因：
1. Schema 描述不够清晰
2. 工具名称不够直观
3. 提示词不够具体

解决方案：
```python
# 改进 schema 描述
{
    "name": "detect_voltage_sags",  # 使用动词开头
    "description": (
        "检测电压跌落事件。"
        "使用场景：用户询问电压是否正常、有无跌落、电压质量等。"
        "输入：文件路径和阈值。"
        "输出：跌落事件列表（时间、持续时间、最低值）。"
    )
}
```

### Q3: 如何限制 Token 使用？

```python
def get_channel_data(cfg_path: str, channel_name: str, max_points: int = 100):
    """限制返回的数据点数"""
    data = reader.get_analog_data(channel.index)

    if len(data) > max_points:
        indices = np.linspace(0, len(data)-1, max_points, dtype=int)
        data = [data[i] for i in indices]
        time = [reader.time_values[i] for i in indices]

    return {"time": time, "values": data}
```

### Q4: 支持流式输出吗？

可以，但需要修改 expert.py：

```python
def analyze_stream(self, query: str, cfg_path: str):
    """流式返回分析结果"""
    # OpenAI
    response = self.openai.chat.completions.create(
        model=self.model,
        messages=messages,
        stream=True  # 启用流式
    )

    for chunk in response:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
```

---

## 下一步

- [API 参考文档](./AI_API_REFERENCE.md) - 详细的 API 文档
- [工具函数库参考](./AI_TOOLS_REFERENCE.md) - 所有内置工具的说明
- [高级功能开发](./AI_ADVANCED_FEATURES.md) - 向量数据库、RAG 等高级功能
- [部署指南](./AI_DEPLOYMENT.md) - 生产环境部署方案

---

**文档版本**: 1.0
**最后更新**: 2024-11-10
**维护者**: ComtradeReader Team
