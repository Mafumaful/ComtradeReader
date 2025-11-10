# AI 专家系统架构设计文档

## 文档概述

本文档详细描述 ComtradeReader AI 专家系统的架构设计、技术选型、实现细节和扩展方案。

---

## 目录

- [系统概述](#系统概述)
- [架构设计](#架构设计)
- [核心组件](#核心组件)
- [数据流](#数据流)
- [技术选型](#技术选型)
- [性能优化](#性能优化)
- [安全性设计](#安全性设计)
- [扩展性设计](#扩展性设计)

---

## 系统概述

### 设计目标

ComtradeReader AI 专家系统旨在通过大语言模型（LLM）为电力系统工程师提供自然语言交互的波形分析能力。

**核心目标：**
1. **易用性**：用自然语言提问，无需记忆复杂命令
2. **智能化**：自动选择和组合分析工具
3. **专业性**：提供符合工程实践的专业解释
4. **扩展性**：易于添加新的分析工具
5. **高性能**：快速响应，支持大文件分析

### 系统能力

- **自然语言理解**：理解用户的专业术语和意图
- **工具自动调用**：根据问题自动选择合适的分析工具
- **多步推理**：通过多次工具调用完成复杂分析
- **专家级解释**：结合电力系统知识给出工程解释
- **可视化生成**：自动生成图表和报告

---

## 架构设计

### 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  CLI Mode    │  │  API Mode    │  │  Batch Mode  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    AI Expert Layer                           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         WaveformExpert (Orchestrator)                │   │
│  │  - Conversation Management                           │   │
│  │  - Intent Recognition                                │   │
│  │  - Tool Selection & Execution                        │   │
│  │  - Response Generation                               │   │
│  └──────────────────────────────────────────────────────┘   │
│                              │                               │
│         ┌────────────────────┼────────────────────┐         │
│         ▼                    ▼                    ▼         │
│  ┌────────────┐      ┌────────────┐      ┌────────────┐    │
│  │  LLM       │      │   Tool     │      │  Context   │    │
│  │  Client    │      │  Registry  │      │  Manager   │    │
│  └────────────┘      └────────────┘      └────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Tool Function Layer                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │   Data   │  │  Basic   │  │ Advanced │  │  Event   │   │
│  │  Access  │  │ Analysis │  │ Analysis │  │ Analysis │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│  ┌──────────┐  ┌──────────┐                                │
│  │Comparison│  │ Reporting│                                │
│  └──────────┘  └──────────┘                                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Core Components Layer                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Comtrade    │  │     HDR      │  │     RPT      │      │
│  │   Reader     │  │   Reader     │  │   Reader     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│  ┌──────────────┐  ┌──────────────┐                        │
│  │ Visualizers  │  │   Plugins    │                        │
│  └──────────────┘  └──────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

### 模块职责

#### 1. User Interface Layer（用户界面层）

**职责：**
- 接收用户输入
- 显示分析结果
- 管理交互流程

**模块：**
- `CLIMode`: 命令行交互模式
- `APIMode`: 编程接口模式
- `BatchMode`: 批量处理模式

#### 2. AI Expert Layer（AI 专家层）

**职责：**
- 理解用户意图
- 编排工具调用流程
- 生成专业解释

**核心组件：**
- `WaveformExpert`: 主协调器
- `LLMClient`: LLM 通信客户端
- `ToolRegistry`: 工具注册中心
- `ContextManager`: 上下文管理

#### 3. Tool Function Layer（工具函数层）

**职责：**
- 实现具体的分析算法
- 标准化输入输出
- 错误处理

**工具分类：**
- Data Access Tools（数据访问）
- Basic Analysis Tools（基础分析）
- Advanced Analysis Tools（高级分析）
- Event Analysis Tools（事件分析）
- Comparison Tools（对比分析）
- Reporting Tools（报告生成）

#### 4. Core Components Layer（核心组件层）

**职责：**
- 文件解析
- 数据读取
- 可视化生成

**现有组件：**
- ComtradeReader
- HdrReader
- RptReader
- Visualizers
- Plugins

---

## 核心组件

### 1. WaveformExpert（专家协调器）

**作用：** 核心协调器，管理整个分析流程。

**类图：**
```
┌─────────────────────────────────────┐
│        WaveformExpert               │
├─────────────────────────────────────┤
│ - llm_client: LLMClient             │
│ - tool_registry: ToolRegistry       │
│ - context_manager: ContextManager   │
│ - conversation_history: List        │
│ - model: str                        │
├─────────────────────────────────────┤
│ + analyze(query, cfg_path)          │
│ + analyze_stream(query, cfg_path)   │
│ + batch_analyze(tasks)              │
│ + reset_conversation()              │
│ - _build_system_prompt(cfg_path)    │
│ - _execute_tool_call(func, args)    │
└─────────────────────────────────────┘
```

**关键方法：**

```python
def analyze(self, query: str, cfg_path: str, max_iterations: int = 10) -> str:
    """
    主分析方法

    流程：
    1. 构建系统提示词（包含电力系统专业知识）
    2. 调用 LLM 理解用户意图
    3. 如果需要工具，执行工具调用
    4. 将工具结果返回给 LLM
    5. 重复 3-4 直到 LLM 给出最终答案
    6. 更新对话历史
    """
```

**状态机：**

```
[用户提问] → [LLM 分析意图]
                    │
                    ├─→ [不需要工具] → [直接回答] → [结束]
                    │
                    └─→ [需要工具]
                            │
                            ▼
                    [选择工具] → [执行工具] → [获取结果]
                            │                      │
                            └──────────────────────┘
                                    │
                                    ▼
                            [LLM 解读结果]
                                    │
                                    ├─→ [需要更多工具] → [继续执行]
                                    │
                                    └─→ [信息充足] → [生成最终答案] → [结束]
```

---

### 2. ToolRegistry（工具注册中心）

**作用：** 管理所有分析工具，提供工具发现和执行能力。

**类图：**
```
┌─────────────────────────────────────┐
│         ToolRegistry                │
├─────────────────────────────────────┤
│ - _tools: Dict[str, Callable]       │
│ - _schemas: Dict[str, Dict]         │
│ - _categories: Dict[str, List]      │
├─────────────────────────────────────┤
│ + register(func, schema)            │
│ + get_tool(name)                    │
│ + get_all_schemas()                 │
│ + execute(tool_name, **kwargs)      │
│ + list_tools_by_category(category)  │
│ + search_tools(keyword)             │
└─────────────────────────────────────┘
```

**工具注册机制：**

```python
# 方式 1: 手动注册
registry = ToolRegistry()
registry.register(calculate_statistics, STATISTICS_SCHEMA)

# 方式 2: 装饰器注册（推荐）
@registry.register_tool(
    category="basic_analysis",
    schema=STATISTICS_SCHEMA
)
def calculate_statistics(cfg_path: str, channel_name: str):
    pass

# 方式 3: 批量注册
registry.register_module(
    module=waveform_viewer.ai.tools.basic_analysis,
    schemas=BASIC_ANALYSIS_SCHEMAS
)
```

**工具发现：**

```python
# 获取所有工具 schemas（供 LLM 使用）
schemas = registry.get_all_schemas()

# 按类别查找工具
basic_tools = registry.list_tools_by_category("basic_analysis")

# 关键词搜索
voltage_tools = registry.search_tools("voltage")
```

---

### 3. LLMClient（LLM 客户端抽象）

**作用：** 抽象不同 LLM 提供商的 API，提供统一接口。

**类图：**
```
┌─────────────────────────────────────┐
│        LLMClient (Abstract)         │
├─────────────────────────────────────┤
│ + chat(messages, tools, **kwargs)   │
│ + chat_stream(messages, tools)      │
└─────────────────────────────────────┘
         △                    △
         │                    │
         │                    │
┌────────┴────────┐   ┌──────┴────────┐
│ OpenAIClient    │   │ ClaudeClient  │
├─────────────────┤   ├───────────────┤
│ - api_key       │   │ - api_key     │
│ - model         │   │ - model       │
├─────────────────┤   ├───────────────┤
│ + chat()        │   │ + chat()      │
│ + chat_stream() │   │ + chat_stream()│
└─────────────────┘   └───────────────┘
```

**统一接口：**

```python
class LLMClient(ABC):
    @abstractmethod
    def chat(
        self,
        messages: List[Dict],
        tools: List[Dict],
        temperature: float = 0.1,
        max_tokens: int = 4096
    ) -> Dict:
        """
        标准聊天接口

        Returns:
            {
                "content": "回答内容",
                "tool_calls": [
                    {"name": "tool_name", "arguments": {...}},
                    ...
                ]
            }
        """
        pass
```

**适配器实现：**

```python
class OpenAIClient(LLMClient):
    def chat(self, messages, tools, **kwargs):
        response = openai.chat.completions.create(
            model=self.model,
            messages=messages,
            functions=tools,  # OpenAI 格式
            **kwargs
        )
        return self._normalize_response(response)


class ClaudeClient(LLMClient):
    def chat(self, messages, tools, **kwargs):
        response = anthropic.messages.create(
            model=self.model,
            messages=messages,
            tools=self._convert_to_claude_format(tools),  # Claude 格式
            **kwargs
        )
        return self._normalize_response(response)
```

---

### 4. ContextManager（上下文管理器）

**作用：** 管理对话上下文，优化 Token 使用。

**功能：**
- 对话历史管理
- Token 计数和限制
- 上下文压缩
- 相关信息检索

**类图：**
```
┌─────────────────────────────────────┐
│        ContextManager               │
├─────────────────────────────────────┤
│ - history: List[Message]            │
│ - max_tokens: int                   │
│ - current_cfg_path: str             │
│ - metadata_cache: Dict              │
├─────────────────────────────────────┤
│ + add_message(role, content)        │
│ + get_context(max_tokens)           │
│ + compress_history()                │
│ + clear()                           │
│ + get_relevant_context(query)       │
└─────────────────────────────────────┘
```

**上下文压缩策略：**

```python
def compress_history(self) -> List[Dict]:
    """
    上下文压缩策略：
    1. 保留系统提示词（始终）
    2. 保留最近 N 轮对话（完整）
    3. 压缩早期对话（仅保留摘要）
    """
    compressed = []

    # 1. 系统提示词
    compressed.append(self.history[0])

    # 2. 最近 3 轮对话（完整保留）
    recent = self.history[-6:]  # 3轮 = 6条消息（用户+助手）

    # 3. 早期对话（压缩为摘要）
    if len(self.history) > 7:
        old_messages = self.history[1:-6]
        summary = self._summarize_messages(old_messages)
        compressed.append({
            "role": "system",
            "content": f"[早期对话摘要]: {summary}"
        })

    compressed.extend(recent)
    return compressed
```

---

## 数据流

### 1. 标准分析流程

```
┌─────────┐
│ 用户提问 │ "这个波形有电压跌落吗？"
└────┬────┘
     │
     ▼
┌────────────────────────────────────────────┐
│ WaveformExpert                             │
│ 1. 构建提示词（注入电力系统知识）            │
│ 2. 初始化对话上下文                         │
└────┬───────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────┐
│ LLM (GPT-4 / Claude)                       │
│ 分析：用户想知道是否有电压跌落               │
│ 决策：需要调用 detect_voltage_sags 工具     │
└────┬───────────────────────────────────────┘
     │
     │ Function Call: detect_voltage_sags(cfg_path="...")
     ▼
┌────────────────────────────────────────────┐
│ ToolRegistry                               │
│ 1. 查找工具函数                             │
│ 2. 验证参数                                │
│ 3. 执行工具                                │
└────┬───────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────┐
│ detect_voltage_sags()                      │
│ 1. 读取 COMTRADE 文件                       │
│ 2. 查找电压通道                             │
│ 3. 检测低于阈值的点                         │
│ 4. 返回结构化结果                           │
└────┬───────────────────────────────────────┘
     │
     │ Result: {"status": "success", "events": [...]}
     ▼
┌────────────────────────────────────────────┐
│ LLM (再次调用)                              │
│ 输入：工具返回的结果                         │
│ 分析：A相在10.123s发生跌落，最低15%         │
│ 输出：生成专业解释                          │
└────┬───────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────┐
│ 最终回答（返回给用户）                        │
│                                             │
│ "检测到电压跌落事件：                        │
│                                             │
│ - **A相电压**：在 10.123 秒时发生严重跌落   │
│   - 最低值：15.2%                           │
│   - 持续时间：333ms                         │
│                                             │
│ **分析**：这是典型的单相接地故障特征。       │
│ A相电压跌落至15%，说明故障点接地电阻较小...  │
│                                             │
│ **建议**：检查A相接地保护定值是否合理。"     │
└─────────────────────────────────────────────┘
```

### 2. 多工具协作流程

```
用户：分析这个故障的类型和保护动作是否正确

    │
    ▼
LLM 分析：需要多个工具配合
    │
    ├─→ Tool 1: identify_fault_type()
    │      → 结果：单相接地故障（A相）
    │
    ├─→ Tool 2: extract_fault_timeline()
    │      → 结果：故障时刻 10.123s，保护 23ms 动作
    │
    ├─→ Tool 3: analyze_protection_action()
    │      → 结果：距离保护I段动作正确
    │
    └─→ 综合所有结果，生成最终答案
```

### 3. 数据结构

**消息格式：**

```python
# 用户消息
{
    "role": "user",
    "content": "这个波形有电压跌落吗？"
}

# 助手消息（文本回答）
{
    "role": "assistant",
    "content": "检测到电压跌落..."
}

# 助手消息（工具调用）
{
    "role": "assistant",
    "content": None,
    "tool_calls": [
        {
            "name": "detect_voltage_sags",
            "arguments": {"cfg_path": "waves/test.cfg"}
        }
    ]
}

# 工具结果消息
{
    "role": "function",
    "name": "detect_voltage_sags",
    "content": '{"status": "success", "events": [...]}'
}
```

**工具 Schema 格式：**

```python
{
    "name": "detect_voltage_sags",
    "description": "检测电压跌落事件...",
    "parameters": {
        "type": "object",
        "properties": {
            "cfg_path": {
                "type": "string",
                "description": "COMTRADE 文件路径"
            },
            "threshold_percent": {
                "type": "number",
                "description": "电压跌落阈值",
                "default": 90.0
            }
        },
        "required": ["cfg_path"]
    }
}
```

---

## 技术选型

### LLM 提供商对比

| 维度 | OpenAI GPT-4 | Anthropic Claude 3.5 | 本地模型 (Llama 3.1) |
|------|--------------|----------------------|---------------------|
| **Function Calling** | ⭐⭐⭐⭐⭐ 成熟稳定 | ⭐⭐⭐⭐ 较新但强大 | ⭐⭐ 需要微调 |
| **推理能力** | ⭐⭐⭐⭐ 优秀 | ⭐⭐⭐⭐⭐ 顶级 | ⭐⭐⭐ 良好 |
| **上下文窗口** | 128K tokens | 200K tokens | 8K-128K |
| **成本** | $$$ 中等 | $$$$ 较高 | $ 低（硬件成本） |
| **延迟** | 1-3秒 | 1-3秒 | 0.5-2秒（GPU） |
| **网络要求** | 需要 | 需要 | 不需要 |
| **数据隐私** | 云端处理 | 云端处理 | 本地处理 |
| **推荐场景** | 通用场景 | 复杂推理 | 私有化部署 |

### 推荐方案

**生产环境：**
- 主力模型：GPT-4 Turbo（平衡性能和成本）
- 备用模型：Claude 3.5 Sonnet（复杂分析任务）

**私有化部署：**
- 基础模型：Llama 3.1 70B（需要微调）
- 向量检索：本地 Embedding 模型

**开发测试：**
- GPT-3.5 Turbo（成本低，速度快）

---

## 性能优化

### 1. Token 优化

**策略：**

```python
# 1. 数据降采样
def get_channel_data(cfg_path, channel_name):
    data = reader.get_analog_data(...)

    # 返回时降采样到 100 个点
    if len(data) > 100:
        indices = np.linspace(0, len(data)-1, 100, dtype=int)
        data = [data[i] for i in indices]

    return data

# 2. 工具结果压缩
def _compress_tool_result(result: Dict) -> Dict:
    """压缩工具结果，减少 token"""
    if "time" in result and len(result["time"]) > 50:
        # 只保留关键点
        result["time"] = result["time"][::10]
        result["values"] = result["values"][::10]

    return result

# 3. 对话历史压缩
def compress_history(self):
    # 早期对话用摘要替代
    summary = self.llm.chat([
        {"role": "user", "content": f"请总结以下对话：\n{old_messages}"}
    ])
    return summary
```

### 2. 缓存机制

```python
from functools import lru_cache
import hashlib

# 1. ComtradeReader 缓存
@lru_cache(maxsize=10)
def _get_cached_reader(cfg_path: str) -> ComtradeReader:
    return ComtradeReader(cfg_path)

# 2. 工具结果缓存
class ToolResultCache:
    def __init__(self):
        self._cache = {}

    def get_cache_key(self, tool_name: str, **kwargs) -> str:
        # 生成缓存键
        key_str = f"{tool_name}:{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, tool_name: str, **kwargs):
        key = self.get_cache_key(tool_name, **kwargs)
        return self._cache.get(key)

    def set(self, tool_name: str, result, **kwargs):
        key = self.get_cache_key(tool_name, **kwargs)
        self._cache[key] = result

# 使用
cache = ToolResultCache()

def execute_with_cache(tool_name, **kwargs):
    cached = cache.get(tool_name, **kwargs)
    if cached:
        return cached

    result = registry.execute(tool_name, **kwargs)
    cache.set(tool_name, result, **kwargs)
    return result
```

### 3. 并行执行

```python
from concurrent.futures import ThreadPoolExecutor

def batch_analyze(self, queries: List[Dict]) -> List[str]:
    """并行处理多个查询"""
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(self.analyze, q["query"], q["cfg_path"])
            for q in queries
        ]

        results = [f.result() for f in futures]

    return results
```

### 4. 流式输出

```python
def analyze_stream(self, query: str, cfg_path: str):
    """流式返回结果，改善用户体验"""
    response = self.llm.chat_stream(messages, tools)

    buffer = ""
    for chunk in response:
        if chunk.get("content"):
            buffer += chunk["content"]
            yield chunk["content"]  # 实时输出

        if chunk.get("tool_call"):
            # 执行工具调用
            result = self._execute_tool(chunk["tool_call"])
            # 继续流式输出
```

---

## 安全性设计

### 1. 输入验证

```python
def validate_cfg_path(cfg_path: str) -> bool:
    """文件路径安全验证"""
    path = Path(cfg_path)

    # 1. 检查文件存在
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {cfg_path}")

    # 2. 检查扩展名
    if path.suffix.lower() not in ['.cfg', '.dat', '.hdr']:
        raise ValueError(f"不支持的文件类型: {path.suffix}")

    # 3. 防止路径遍历
    if ".." in str(path):
        raise ValueError("检测到路径遍历攻击")

    # 4. 检查文件大小
    if path.stat().st_size > 100 * 1024 * 1024:  # 100MB
        raise ValueError("文件过大")

    return True
```

### 2. API Key 管理

```python
# config.py
from pathlib import Path
from dotenv import load_dotenv
import os

# 加载 .env 文件
load_dotenv()

class AIConfig:
    """AI 配置管理（单例模式）"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        # 从环境变量读取
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

        # 从配置文件读取（备选）
        config_file = Path.home() / ".comtrade_ai" / "config.json"
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
                self.openai_api_key = config.get("openai_api_key", self.openai_api_key)
                # ...

    def validate(self):
        if not self.openai_api_key and not self.anthropic_api_key:
            raise ValueError("至少需要配置一个 API Key")

# 使用
config = AIConfig()
config.validate()
```

### 3. Prompt Injection 防护

```python
def sanitize_user_input(user_input: str) -> str:
    """清理用户输入，防止 Prompt Injection"""

    # 1. 移除系统提示词关键字
    dangerous_patterns = [
        r"ignore\s+previous\s+instructions",
        r"you\s+are\s+now",
        r"forget\s+everything",
        r"<\|im_start\|>",
        r"<\|im_end\|>"
    ]

    for pattern in dangerous_patterns:
        user_input = re.sub(pattern, "", user_input, flags=re.IGNORECASE)

    # 2. 限制长度
    if len(user_input) > 2000:
        user_input = user_input[:2000]

    return user_input

def analyze(self, query: str, cfg_path: str):
    # 清理输入
    query = sanitize_user_input(query)
    cfg_path = validate_cfg_path(cfg_path)

    # 继续处理...
```

### 4. 输出过滤

```python
def filter_sensitive_info(output: str) -> str:
    """过滤输出中的敏感信息"""

    # 1. 移除可能的 API Key
    output = re.sub(r'sk-[a-zA-Z0-9]{48}', '[API_KEY_REDACTED]', output)

    # 2. 移除绝对路径
    output = re.sub(r'/Users/[^/]+/', '/Users/***/', output)

    # 3. 移除 IP 地址（可选）
    output = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[IP_REDACTED]', output)

    return output
```

---

## 扩展性设计

### 1. 插件式工具扩展

```python
# waveform_viewer/ai/tools/custom_analysis.py

from waveform_viewer.ai.tool_registry import registry

@registry.register_tool(
    category="custom_analysis",
    schema={
        "name": "my_custom_analysis",
        "description": "我的自定义分析",
        "parameters": {...}
    }
)
def my_custom_analysis(cfg_path: str, param1: str) -> Dict:
    """自定义分析工具"""
    # 实现你的分析逻辑
    return {"status": "success", "data": ...}

# 自动发现和注册
registry.discover_tools("waveform_viewer.ai.tools")
```

### 2. 多模型支持

```python
# waveform_viewer/ai/llm_factory.py

class LLMFactory:
    @staticmethod
    def create(provider: str, **kwargs) -> LLMClient:
        if provider == "openai":
            return OpenAIClient(**kwargs)
        elif provider == "anthropic":
            return ClaudeClient(**kwargs)
        elif provider == "local":
            return LocalLLMClient(**kwargs)
        else:
            raise ValueError(f"不支持的提供商: {provider}")

# 使用
llm = LLMFactory.create(
    provider=config.ai_model_provider,
    api_key=config.api_key,
    model=config.model_name
)
```

### 3. 自定义提示词模板

```python
# waveform_viewer/ai/prompts/templates.py

class PromptTemplate:
    DEFAULT = """你是电力系统暂态波形分析专家..."""

    FAULT_ANALYSIS = """你是故障分析专家，专注于识别故障类型..."""

    PROTECTION_EVALUATION = """你是继电保护专家，专注于评估保护动作..."""

# 使用
expert = WaveformExpert(
    llm_client=llm,
    tool_registry=registry,
    prompt_template=PromptTemplate.FAULT_ANALYSIS
)
```

### 4. RAG（检索增强生成）集成

```python
# waveform_viewer/ai/rag/retriever.py

from sentence_transformers import SentenceTransformer
import faiss

class CaseRetriever:
    """历史案例检索器"""

    def __init__(self, embedding_model="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embedding_model)
        self.index = None
        self.cases = []

    def add_case(self, case: Dict):
        """添加历史案例"""
        embedding = self.model.encode([case["description"]])[0]
        if self.index is None:
            self.index = faiss.IndexFlatL2(embedding.shape[0])

        self.index.add(embedding.reshape(1, -1))
        self.cases.append(case)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """检索相似案例"""
        query_embedding = self.model.encode([query])[0]
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1),
            top_k
        )

        return [self.cases[idx] for idx in indices[0]]

# 集成到 Expert
class WaveformExpertWithRAG(WaveformExpert):
    def __init__(self, *args, retriever: CaseRetriever = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.retriever = retriever

    def _build_system_prompt(self, cfg_path: str) -> str:
        base_prompt = super()._build_system_prompt(cfg_path)

        # 检索相关案例
        if self.retriever:
            similar_cases = self.retriever.retrieve(cfg_path)
            cases_text = "\n".join([
                f"- {case['description']}: {case['conclusion']}"
                for case in similar_cases
            ])

            base_prompt += f"\n\n## 相关历史案例\n{cases_text}"

        return base_prompt
```

---

## 部署架构

### 开发环境

```
Developer Machine
├── Python 3.8+
├── pip packages
├── .env (API keys)
└── waves/ (test data)
```

### 生产环境

```
┌─────────────────────────────────────┐
│         Load Balancer               │
└─────────────────┬───────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
        ▼                   ▼
┌───────────────┐   ┌───────────────┐
│  App Server 1 │   │  App Server 2 │
│  (Docker)     │   │  (Docker)     │
└───────┬───────┘   └───────┬───────┘
        │                   │
        └─────────┬─────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
        ▼                   ▼
┌───────────────┐   ┌───────────────┐
│  Redis Cache  │   │  Vector DB    │
└───────────────┘   └───────────────┘
```

---

**文档版本**: 1.0
**最后更新**: 2024-11-10
**维护者**: ComtradeReader Team
