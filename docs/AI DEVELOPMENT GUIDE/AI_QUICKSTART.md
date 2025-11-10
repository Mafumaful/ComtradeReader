# AI 专家系统快速开始指南

这是一个 15 分钟快速入门指南，帮助你快速搭建和运行 ComtradeReader AI 专家系统。

---

## 第一步：环境准备（5分钟）

### 1. 安装依赖

```bash
# 基础依赖（如果还没安装）
pip install numpy plotly

# AI 相关依赖
pip install openai anthropic scipy pydantic python-dotenv

# 或使用 requirements 文件
pip install -r requirements-ai.txt
```

### 2. 配置 API Key

创建 `.env` 文件在项目根目录：

```bash
# .env
OPENAI_API_KEY=sk-your-openai-api-key-here
# 或者使用 Claude
# ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here

# 选择使用的模型
AI_MODEL_PROVIDER=openai
AI_MODEL_NAME=gpt-4
```

**获取 API Key：**
- OpenAI: https://platform.openai.com/api-keys
- Anthropic: https://console.anthropic.com/

---

## 第二步：创建最小实现（5分钟）

### 1. 创建工具函数

创建文件 `waveform_viewer/ai/tools/basic_analysis.py`：

```python
from waveform_viewer.core.reader import ComtradeReader
import numpy as np
from typing import Dict, Any

def calculate_statistics(cfg_path: str, channel_name: str) -> Dict[str, Any]:
    """计算通道统计特征"""
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
            "channel": {"name": channel.name, "unit": channel.unit},
            "statistics": {
                "min": float(np.min(data)),
                "max": float(np.max(data)),
                "mean": float(np.mean(data)),
                "std": float(np.std(data)),
                "rms": float(np.sqrt(np.mean(data**2)))
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def detect_voltage_sags(cfg_path: str, threshold_percent: float = 90.0) -> Dict[str, Any]:
    """检测电压跌落"""
    try:
        reader = ComtradeReader(cfg_path)
        voltage_channels = [ch for ch in reader.analog_channels if '电压' in ch.name]

        if not voltage_channels:
            return {"status": "error", "message": "未找到电压通道"}

        events = []
        for ch in voltage_channels:
            data = np.array(reader.get_analog_data(ch.index))
            time = np.array(reader.time_values)

            sag_mask = data < threshold_percent
            if np.any(sag_mask):
                sag_indices = np.where(sag_mask)[0]
                events.append({
                    "channel": ch.name,
                    "start_time": float(time[sag_indices[0]]),
                    "min_value": float(np.min(data[sag_indices])),
                    "threshold": threshold_percent
                })

        return {"status": "success", "events": events, "event_count": len(events)}
    except Exception as e:
        return {"status": "error", "message": str(e)}
```

### 2. 定义 Function Schemas

创建文件 `waveform_viewer/ai/function_schemas.py`：

```python
TOOL_SCHEMAS = [
    {
        "name": "calculate_statistics",
        "description": "计算指定通道的统计特征（最小值、最大值、均值、标准差、RMS）",
        "parameters": {
            "type": "object",
            "properties": {
                "cfg_path": {
                    "type": "string",
                    "description": "COMTRADE 配置文件路径"
                },
                "channel_name": {
                    "type": "string",
                    "description": "通道名称或匹配模式，如 'A相电压'"
                }
            },
            "required": ["cfg_path", "channel_name"]
        }
    },
    {
        "name": "detect_voltage_sags",
        "description": "检测电压跌落事件。当用户询问电压是否正常、有无跌落时使用",
        "parameters": {
            "type": "object",
            "properties": {
                "cfg_path": {
                    "type": "string",
                    "description": "COMTRADE 配置文件路径"
                },
                "threshold_percent": {
                    "type": "number",
                    "description": "电压跌落阈值（百分比），默认90%",
                    "default": 90.0
                }
            },
            "required": ["cfg_path"]
        }
    }
]
```

### 3. 创建工具注册表

创建文件 `waveform_viewer/ai/tool_registry.py`：

```python
from typing import Dict, Callable, Any, List

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._schemas: Dict[str, Dict] = {}

    def register(self, func: Callable, schema: Dict):
        self._tools[func.__name__] = func
        self._schemas[func.__name__] = schema

    def get_tool(self, name: str) -> Callable:
        return self._tools.get(name)

    def get_all_schemas(self) -> List[Dict]:
        return list(self._schemas.values())

    def execute(self, tool_name: str, **kwargs) -> Any:
        tool = self.get_tool(tool_name)
        if not tool:
            return {"status": "error", "message": f"工具 {tool_name} 不存在"}

        try:
            return tool(**kwargs)
        except Exception as e:
            return {"status": "error", "message": str(e)}


def get_default_registry() -> ToolRegistry:
    """获取预注册的工具注册表"""
    from .tools import basic_analysis
    from .function_schemas import TOOL_SCHEMAS

    registry = ToolRegistry()

    # 注册工具
    for schema in TOOL_SCHEMAS:
        tool_name = schema["name"]
        if hasattr(basic_analysis, tool_name):
            func = getattr(basic_analysis, tool_name)
            registry.register(func, schema)

    return registry
```

### 4. 创建专家系统

创建文件 `waveform_viewer/ai/expert.py`：

```python
import json
from typing import List, Dict
import openai

class WaveformExpert:
    def __init__(self, openai_client, tool_registry, model="gpt-4"):
        self.openai = openai_client
        self.registry = tool_registry
        self.model = model
        self.conversation_history = []

    def analyze(self, query: str, cfg_path: str, max_iterations: int = 10) -> str:
        system_prompt = f"""你是电力系统暂态波形分析专家。

当前分析的文件：{cfg_path}

你可以调用分析工具来回答用户问题。请根据问题选择合适的工具，并给出专业的解释。"""

        messages = [
            {"role": "system", "content": system_prompt},
            *self.conversation_history,
            {"role": "user", "content": query}
        ]

        iteration = 0

        while iteration < max_iterations:
            response = self.openai.chat.completions.create(
                model=self.model,
                messages=messages,
                functions=self.registry.get_all_schemas(),
                function_call="auto",
                temperature=0.1
            )

            message = response.choices[0].message

            # 如果不需要调用函数，返回答案
            if not message.function_call:
                answer = message.content
                self.conversation_history.append({"role": "user", "content": query})
                self.conversation_history.append({"role": "assistant", "content": answer})
                return answer

            # 调用函数
            func_call = message.function_call
            func_name = func_call.name
            func_args = json.loads(func_call.arguments)

            # 注入 cfg_path
            if "cfg_path" in func_args and not func_args.get("cfg_path"):
                func_args["cfg_path"] = cfg_path

            print(f"  → 调用工具: {func_name}")
            result = self.registry.execute(func_name, **func_args)

            # 添加到消息历史
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

        return "抱歉，分析超时"

    def reset_conversation(self):
        self.conversation_history = []
```

---

## 第三步：测试运行（5分钟）

### 创建测试脚本

创建文件 `test_ai_expert.py`：

```python
import os
from dotenv import load_dotenv
import openai

# 加载环境变量
load_dotenv()

# 导入我们的模块
from waveform_viewer.ai.expert import WaveformExpert
from waveform_viewer.ai.tool_registry import get_default_registry

# 配置 OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# 创建专家系统
registry = get_default_registry()
expert = WaveformExpert(openai, registry)

# 测试查询
cfg_path = "waves/20241030_115240/20241030_115240.cfg"  # 替换为你的文件路径

print("=== AI 专家系统测试 ===\n")

# 测试 1：询问统计信息
print("问题 1: A相电压的统计特征是什么？")
result = expert.analyze("A相电压的统计特征是什么？", cfg_path)
print(f"回答：\n{result}\n")
print("-" * 60)

# 测试 2：检测电压跌落
print("\n问题 2: 这个波形有电压跌落吗？")
result = expert.analyze("这个波形有电压跌落吗？", cfg_path)
print(f"回答：\n{result}\n")
print("-" * 60)

print("\n✅ 测试完成！")
```

### 运行测试

```bash
python test_ai_expert.py
```

**预期输出：**

```
=== AI 专家系统测试 ===

问题 1: A相电压的统计特征是什么？
  → 调用工具: calculate_statistics
回答：
A相电压的统计特征如下：

- **最小值**: 15.2%
- **最大值**: 105.8%
- **平均值**: 98.5%
- **标准差**: 12.3%
- **有效值 (RMS)**: 99.1%

从数据可以看出，A相电压在大部分时间保持在正常范围，但出现了显著的最小值（15.2%），
这表明可能发生了电压跌落事件。建议进一步检查电压跌落的时刻和持续时间。

------------------------------------------------------------

问题 2: 这个波形有电压跌落吗？
  → 调用工具: detect_voltage_sags
回答：
检测到电压跌落事件：

**A相电压**：
- 跌落开始时间：10.123 秒
- 最低值：15.2%
- 阈值：90%

**分析**：
这是一次严重的电压跌落事件，电压跌落至15.2%，远低于正常运行水平。
这种程度的电压跌落通常由短路故障引起，特别是单相接地故障。

**建议**：
1. 检查该时刻的故障录波数据，确认故障类型
2. 检查保护装置的动作情况
3. 分析故障原因，必要时进行设备检修

------------------------------------------------------------

✅ 测试完成！
```

---

## 第四步：集成到主程序（可选）

### 修改 `main.py`

```python
import argparse
from waveform_viewer.app import WaveformViewerApp

def main():
    parser = argparse.ArgumentParser(description='COMTRADE Waveform Viewer')
    parser.add_argument('--simple', action='store_true', help='使用简化菜单模式')
    parser.add_argument('--ai', action='store_true', help='启用 AI 专家分析')
    parser.add_argument('--file', type=str, help='直接分析指定文件')
    args = parser.parse_args()

    if args.ai:
        # AI 模式
        from waveform_viewer.ai.expert import WaveformExpert
        from waveform_viewer.ai.tool_registry import get_default_registry
        import openai
        import os
        from dotenv import load_dotenv

        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")

        registry = get_default_registry()
        expert = WaveformExpert(openai, registry)

        if args.file:
            # 直接分析指定文件
            print(f"\n分析文件: {args.file}\n")
            while True:
                query = input("您的问题 (输入 'exit' 退出) > ")
                if query.lower() == 'exit':
                    break

                print("\n🤖 正在分析...\n")
                result = expert.analyze(query, args.file)
                print(f"📊 {result}\n")
        else:
            print("请使用 --file 参数指定要分析的文件")
    else:
        # 原有模式
        app = WaveformViewerApp("waves/", use_simple_menu=args.simple)
        app.run()

if __name__ == "__main__":
    main()
```

### 使用 AI 模式

```bash
# 启动 AI 交互模式
python main.py --ai --file waves/20241030_115240/20241030_115240.cfg

# 然后就可以自然语言提问了
您的问题 > 这个波形有什么异常吗？
您的问题 > A相电流的最大值是多少？
您的问题 > 分析一下电压质量
```

---

## 常见问题

### Q1: 提示 "No module named 'waveform_viewer.ai'"

**原因：** 还没创建 `waveform_viewer/ai/__init__.py`

**解决：**
```bash
mkdir -p waveform_viewer/ai/tools
touch waveform_viewer/ai/__init__.py
touch waveform_viewer/ai/tools/__init__.py
```

### Q2: OpenAI API 报错 "Incorrect API key"

**原因：** API Key 配置错误

**解决：**
1. 检查 `.env` 文件中的 API Key 是否正确
2. 确认 API Key 有足够的余额
3. 尝试在 OpenAI 网站重新生成 Key

### Q3: 工具没有被调用

**原因：** Schema 描述不够清晰

**解决：** 改进 Schema 的 description 字段：

```python
{
    "name": "detect_voltage_sags",
    "description": (
        "检测电压跌落事件。"
        "使用场景：用户询问'电压是否正常'、'有没有电压跌落'、"
        "'电压质量如何'等问题时使用此工具。"
    ),
    # ...
}
```

### Q4: 响应太慢

**原因：** 使用的模型较大或网络延迟

**解决：**
1. 改用 `gpt-3.5-turbo`（更快但能力稍弱）
2. 减少返回的数据量（降采样）
3. 使用流式输出改善用户体验

---

## 下一步

🎉 恭喜！你已经成功搭建了一个基础的 AI 专家系统。

**继续学习：**
- [完整开发指南](./AI_INTEGRATION_GUIDE.md) - 深入了解架构设计
- [工具函数参考](./AI_TOOLS_REFERENCE.md) - 添加更多分析工具
- [架构设计文档](./AI_ARCHITECTURE.md) - 理解系统架构

**建议的改进方向：**
1. 添加更多分析工具（FFT、谐波分析、故障识别）
2. 实现多轮对话和上下文管理
3. 添加可视化生成功能
4. 集成向量数据库（历史案例检索）
5. 支持批量分析和报告生成

---

**文档版本**: 1.0
**最后更新**: 2024-11-10
