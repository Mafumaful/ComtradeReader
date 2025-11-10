# AI 专家系统 - 示例代码集合

本文档提供完整的可运行示例代码，帮助你快速理解和使用 AI 专家系统。

---

## 目录

- [基础示例](#基础示例)
- [工具函数示例](#工具函数示例)
- [高级功能示例](#高级功能示例)
- [实际应用案例](#实际应用案例)

---

## 基础示例

### 示例 1: 最简单的 AI 查询

```python
"""
最简单的 AI 专家系统使用示例
"""
import os
from dotenv import load_dotenv
import openai

from waveform_viewer.ai.expert import WaveformExpert
from waveform_viewer.ai.tool_registry import get_default_registry

# 加载环境变量
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# 初始化
registry = get_default_registry()
expert = WaveformExpert(openai, registry, model="gpt-4")

# 分析
result = expert.analyze(
    query="这个波形有电压跌落吗？",
    cfg_path="waves/20241030_115240/20241030_115240.cfg"
)

print(result)
```

### 示例 2: 交互式对话

```python
"""
交互式 AI 对话示例
支持多轮对话，保持上下文
"""
import os
from dotenv import load_dotenv
import openai

from waveform_viewer.ai.expert import WaveformExpert
from waveform_viewer.ai.tool_registry import get_default_registry

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

registry = get_default_registry()
expert = WaveformExpert(openai, registry)

cfg_path = "waves/20241030_115240/20241030_115240.cfg"

print("=== AI 专家对话模式 ===")
print("输入 'exit' 退出，输入 'reset' 重置对话\n")

while True:
    query = input("您的问题 > ")

    if query.lower() == 'exit':
        break

    if query.lower() == 'reset':
        expert.reset_conversation()
        print("对话已重置\n")
        continue

    print("\n🤖 正在分析...\n")
    result = expert.analyze(query, cfg_path)
    print(f"📊 分析结果：\n{result}\n")
    print("-" * 60 + "\n")
```

### 示例 3: 批量分析

```python
"""
批量分析多个问题
"""
from waveform_viewer.ai.expert import WaveformExpert
from waveform_viewer.ai.tool_registry import get_default_registry
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

registry = get_default_registry()
expert = WaveformExpert(openai, registry)

# 定义要分析的任务
tasks = [
    {
        "query": "这个波形有电压跌落吗？",
        "cfg_path": "waves/case1.cfg"
    },
    {
        "query": "A相电流的统计特征是什么？",
        "cfg_path": "waves/case1.cfg"
    },
    {
        "query": "保护装置动作是否正确？",
        "cfg_path": "waves/case1.cfg"
    }
]

# 批量分析
results = expert.batch_analyze(tasks)

# 输出结果
for i, (task, result) in enumerate(zip(tasks, results), 1):
    print(f"\n=== 问题 {i}: {task['query']} ===")
    print(result)
    print("-" * 60)
```

---

## 工具函数示例

### 示例 4: 创建自定义工具函数

```python
"""
创建自定义的分析工具
"""
from waveform_viewer.core.reader import ComtradeReader
import numpy as np
from typing import Dict, Any

def analyze_frequency_stability(cfg_path: str) -> Dict[str, Any]:
    """
    分析频率稳定性
    """
    try:
        reader = ComtradeReader(cfg_path)

        # 查找频率通道
        freq_channel = reader.get_channel_by_name("频率")
        if not freq_channel:
            return {
                "status": "error",
                "message": "未找到频率通道"
            }

        data = np.array(reader.get_analog_data(freq_channel.index))

        # 计算统计指标
        mean_freq = np.mean(data)
        std_freq = np.std(data)
        max_deviation = max(abs(mean_freq - 50), abs(np.min(data) - 50), abs(np.max(data) - 50))

        # 判断稳定性
        if max_deviation < 0.1:
            stability = "优秀"
        elif max_deviation < 0.3:
            stability = "良好"
        elif max_deviation < 0.5:
            stability = "一般"
        else:
            stability = "较差"

        return {
            "status": "success",
            "mean_frequency": float(mean_freq),
            "std_deviation": float(std_freq),
            "min_frequency": float(np.min(data)),
            "max_frequency": float(np.max(data)),
            "max_deviation_from_nominal": float(max_deviation),
            "stability_assessment": stability
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


# 定义对应的 Schema
FREQUENCY_STABILITY_SCHEMA = {
    "name": "analyze_frequency_stability",
    "description": "分析系统频率稳定性。用于评估频率偏差、波动情况",
    "parameters": {
        "type": "object",
        "properties": {
            "cfg_path": {
                "type": "string",
                "description": "COMTRADE 配置文件路径"
            }
        },
        "required": ["cfg_path"]
    }
}

# 注册到工具注册表
from waveform_viewer.ai.tool_registry import get_default_registry

registry = get_default_registry()
registry.register(analyze_frequency_stability, FREQUENCY_STABILITY_SCHEMA)

# 现在 AI 就可以调用这个工具了！
```

### 示例 5: 使用装饰器注册工具

```python
"""
使用装饰器简化工具注册
"""
from waveform_viewer.ai.tool_registry import ToolRegistry
from waveform_viewer.core.reader import ComtradeReader
import numpy as np

# 创建注册表实例
registry = ToolRegistry()

# 使用装饰器注册
@registry.register_tool(
    category="power_quality",
    schema={
        "name": "calculate_voltage_unbalance",
        "description": "计算三相电压不平衡度",
        "parameters": {
            "type": "object",
            "properties": {
                "cfg_path": {"type": "string", "description": "文件路径"}
            },
            "required": ["cfg_path"]
        }
    }
)
def calculate_voltage_unbalance(cfg_path: str):
    """计算三相电压不平衡度"""
    reader = ComtradeReader(cfg_path)

    # 获取三相电压
    va = reader.get_channel_by_name("A相电压")
    vb = reader.get_channel_by_name("B相电压")
    vc = reader.get_channel_by_name("C相电压")

    if not (va and vb and vc):
        return {"status": "error", "message": "未找到三相电压通道"}

    # 计算 RMS 值
    va_rms = np.sqrt(np.mean(np.array(reader.get_analog_data(va.index))**2))
    vb_rms = np.sqrt(np.mean(np.array(reader.get_analog_data(vb.index))**2))
    vc_rms = np.sqrt(np.mean(np.array(reader.get_analog_data(vc.index))**2))

    # 计算不平衡度
    v_avg = (va_rms + vb_rms + vc_rms) / 3
    max_deviation = max(abs(va_rms - v_avg), abs(vb_rms - v_avg), abs(vc_rms - v_avg))
    unbalance = (max_deviation / v_avg) * 100

    return {
        "status": "success",
        "voltage_a_rms": float(va_rms),
        "voltage_b_rms": float(vb_rms),
        "voltage_c_rms": float(vc_rms),
        "unbalance_percent": float(unbalance),
        "assessment": "正常" if unbalance < 2 else "超标"
    }
```

### 示例 6: 组合多个工具

```python
"""
创建高级分析工具，内部调用多个基础工具
"""
from waveform_viewer.ai.tool_registry import get_default_registry

def comprehensive_fault_analysis(cfg_path: str) -> Dict[str, Any]:
    """
    综合故障分析
    自动调用多个工具，给出完整的故障诊断报告
    """
    registry = get_default_registry()

    results = {}

    # 1. 检测电压跌落
    results['voltage_sags'] = registry.execute('detect_voltage_sags', cfg_path=cfg_path)

    # 2. 检测电流突变
    results['current_surges'] = registry.execute('detect_current_surges', cfg_path=cfg_path)

    # 3. 提取故障时间线
    results['fault_timeline'] = registry.execute('extract_fault_timeline', cfg_path=cfg_path)

    # 4. 识别故障类型
    results['fault_type'] = registry.execute('identify_fault_type', cfg_path=cfg_path)

    # 5. 分析保护动作
    results['protection'] = registry.execute('analyze_protection_action', cfg_path=cfg_path)

    # 综合评估
    assessment = []

    if results['fault_type']['status'] == 'success':
        assessment.append(f"故障类型: {results['fault_type']['fault_type']}")

    if results['voltage_sags']['event_count'] > 0:
        assessment.append(f"检测到 {results['voltage_sags']['event_count']} 个电压跌落事件")

    if results['protection']['status'] == 'success':
        assessment.append(f"保护动作: {results['protection']['overall_assessment']}")

    return {
        "status": "success",
        "detailed_results": results,
        "summary": assessment
    }
```

---

## 高级功能示例

### 示例 7: 流式输出

```python
"""
流式输出 AI 回答，改善用户体验
"""
from waveform_viewer.ai.expert import WaveformExpert

class StreamingExpert(WaveformExpert):
    """支持流式输出的专家系统"""

    def analyze_stream(self, query: str, cfg_path: str):
        """流式返回分析结果"""
        system_prompt = self._build_system_prompt(cfg_path)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        # 使用流式 API
        response = self.openai.chat.completions.create(
            model=self.model,
            messages=messages,
            functions=self.registry.get_all_schemas(),
            stream=True,
            temperature=0.1
        )

        buffer = ""
        for chunk in response:
            delta = chunk.choices[0].delta

            # 处理文本内容
            if delta.content:
                buffer += delta.content
                yield delta.content

            # 处理函数调用（非流式）
            if delta.function_call:
                # 执行工具调用
                func_name = delta.function_call.name
                func_args = json.loads(delta.function_call.arguments)

                result = self.registry.execute(func_name, **func_args)

                # 继续流式输出结果解释
                # （需要再次调用 LLM）
                yield f"\n\n[调用工具: {func_name}]\n"

        return buffer


# 使用示例
import sys

expert = StreamingExpert(openai, registry)

print("🤖 AI 分析中", end="", flush=True)

for chunk in expert.analyze_stream("分析这个故障", "waves/test.cfg"):
    print(chunk, end="", flush=True)

print("\n\n✅ 分析完成")
```

### 示例 8: 上下文管理和压缩

```python
"""
智能管理对话上下文，避免超出 Token 限制
"""
from typing import List, Dict
import tiktoken

class ContextManager:
    """对话上下文管理器"""

    def __init__(self, max_tokens: int = 8000):
        self.max_tokens = max_tokens
        self.history: List[Dict] = []
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")

    def add_message(self, role: str, content: str):
        """添加消息到历史"""
        self.history.append({"role": role, "content": content})

    def count_tokens(self, messages: List[Dict]) -> int:
        """计算消息的 token 数"""
        text = "\n".join([m.get("content", "") for m in messages])
        return len(self.tokenizer.encode(text))

    def get_context(self, system_prompt: str) -> List[Dict]:
        """获取适合的上下文（自动压缩）"""
        messages = [{"role": "system", "content": system_prompt}]

        # 计算系统提示词的 token
        system_tokens = self.count_tokens(messages)

        # 从最新开始添加历史消息
        remaining_tokens = self.max_tokens - system_tokens
        recent_history = []

        for msg in reversed(self.history):
            msg_tokens = self.count_tokens([msg])
            if remaining_tokens - msg_tokens > 0:
                recent_history.insert(0, msg)
                remaining_tokens -= msg_tokens
            else:
                break

        messages.extend(recent_history)
        return messages

    def compress_history(self, llm_client):
        """压缩早期对话历史"""
        if len(self.history) <= 6:  # 少于3轮对话，不压缩
            return

        # 压缩除最近3轮外的所有对话
        old_messages = self.history[:-6]
        recent_messages = self.history[-6:]

        # 使用 LLM 生成摘要
        summary_prompt = f"""请简要总结以下对话的关键信息：
{old_messages}

要求：
1. 保留重要的分析结果
2. 保留用户关注的重点
3. 控制在100字以内"""

        response = llm_client.chat.completions.create(
            model="gpt-3.5-turbo",  # 用便宜的模型做摘要
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=200
        )

        summary = response.choices[0].message.content

        # 替换历史
        self.history = [
            {"role": "system", "content": f"[早期对话摘要]: {summary}"},
            *recent_messages
        ]


# 集成到 Expert
class ContextAwareExpert(WaveformExpert):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context_manager = ContextManager(max_tokens=8000)

    def analyze(self, query: str, cfg_path: str):
        system_prompt = self._build_system_prompt(cfg_path)

        # 使用上下文管理器获取消息
        messages = self.context_manager.get_context(system_prompt)
        messages.append({"role": "user", "content": query})

        # ... 调用 LLM

        # 添加到历史
        self.context_manager.add_message("user", query)
        self.context_manager.add_message("assistant", answer)

        # 如果接近限制，压缩历史
        if self.context_manager.count_tokens(self.context_manager.history) > 6000:
            self.context_manager.compress_history(self.openai)

        return answer
```

### 示例 9: 多模型支持

```python
"""
支持多个 LLM 提供商
"""
from abc import ABC, abstractmethod

class LLMClient(ABC):
    """LLM 客户端抽象基类"""

    @abstractmethod
    def chat(self, messages, tools, **kwargs):
        pass


class OpenAIClient(LLMClient):
    """OpenAI 客户端"""

    def __init__(self, api_key: str, model: str = "gpt-4"):
        import openai
        self.openai = openai
        self.openai.api_key = api_key
        self.model = model

    def chat(self, messages, tools, **kwargs):
        response = self.openai.chat.completions.create(
            model=self.model,
            messages=messages,
            functions=tools,
            **kwargs
        )
        return self._normalize_response(response)

    def _normalize_response(self, response):
        """标准化响应格式"""
        message = response.choices[0].message

        result = {
            "content": message.content,
            "tool_calls": []
        }

        if message.function_call:
            result["tool_calls"].append({
                "name": message.function_call.name,
                "arguments": json.loads(message.function_call.arguments)
            })

        return result


class ClaudeClient(LLMClient):
    """Claude 客户端"""

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        import anthropic
        self.anthropic = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def chat(self, messages, tools, **kwargs):
        # 转换工具格式
        claude_tools = self._convert_tools(tools)

        response = self.anthropic.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=messages[1:],  # 跳过 system 消息
            system=messages[0]["content"],
            tools=claude_tools,
            **kwargs
        )

        return self._normalize_response(response)

    def _convert_tools(self, openai_tools):
        """转换 OpenAI 格式到 Claude 格式"""
        return [
            {
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool["parameters"]
            }
            for tool in openai_tools
        ]

    def _normalize_response(self, response):
        result = {
            "content": "",
            "tool_calls": []
        }

        for block in response.content:
            if block.type == "text":
                result["content"] += block.text
            elif block.type == "tool_use":
                result["tool_calls"].append({
                    "name": block.name,
                    "arguments": block.input
                })

        return result


# 工厂模式
class LLMFactory:
    @staticmethod
    def create(provider: str, **kwargs) -> LLMClient:
        if provider == "openai":
            return OpenAIClient(**kwargs)
        elif provider == "anthropic":
            return ClaudeClient(**kwargs)
        else:
            raise ValueError(f"不支持的提供商: {provider}")


# 使用
import os
from dotenv import load_dotenv

load_dotenv()

# 创建客户端
llm = LLMFactory.create(
    provider="openai",  # 或 "anthropic"
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4"
)

# 创建专家系统
from waveform_viewer.ai.expert import WaveformExpert
from waveform_viewer.ai.tool_registry import get_default_registry

registry = get_default_registry()
expert = WaveformExpert(llm, registry)

# 使用方式完全相同
result = expert.analyze("分析故障", "waves/test.cfg")
```

---

## 实际应用案例

### 案例 1: 自动故障诊断脚本

```python
"""
自动故障诊断脚本
读取指定目录下的所有录波文件，自动生成诊断报告
"""
import os
from pathlib import Path
from waveform_viewer.ai.expert import WaveformExpert
from waveform_viewer.ai.tool_registry import get_default_registry
import openai
from dotenv import load_dotenv
import json
from datetime import datetime

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

registry = get_default_registry()
expert = WaveformExpert(openai, registry)


def auto_diagnose_directory(waves_dir: str, output_dir: str):
    """自动诊断目录下所有波形文件"""

    waves_path = Path(waves_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # 查找所有 .cfg 文件
    cfg_files = list(waves_path.rglob("*.cfg"))

    print(f"找到 {len(cfg_files)} 个录波文件")

    results = []

    for i, cfg_file in enumerate(cfg_files, 1):
        print(f"\n[{i}/{len(cfg_files)}] 分析: {cfg_file.name}")

        # 重置对话（每个文件独立分析）
        expert.reset_conversation()

        try:
            # 执行诊断查询
            diagnosis = expert.analyze(
                query="""请对这个录波文件进行全面诊断，包括：
1. 是否发生故障？什么类型的故障？
2. 电压和电流有什么异常？
3. 保护装置动作是否正确？
4. 给出诊断结论和建议""",
                cfg_path=str(cfg_file)
            )

            result = {
                "file": str(cfg_file),
                "timestamp": datetime.now().isoformat(),
                "diagnosis": diagnosis,
                "status": "success"
            }

        except Exception as e:
            result = {
                "file": str(cfg_file),
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "status": "error"
            }

        results.append(result)

        # 保存单个结果
        output_file = output_path / f"{cfg_file.stem}_diagnosis.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"文件: {cfg_file}\n")
            f.write(f"时间: {result['timestamp']}\n")
            f.write(f"\n{'='*60}\n\n")
            f.write(result.get('diagnosis', result.get('error', '')))

        print(f"  ✅ 结果已保存到: {output_file}")

    # 保存汇总报告
    summary_file = output_path / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 所有分析完成！汇总报告: {summary_file}")


# 使用
if __name__ == "__main__":
    auto_diagnose_directory(
        waves_dir="waves/",
        output_dir="diagnosis_reports/"
    )
```

### 案例 2: 批量电能质量评估

```python
"""
批量评估电能质量
"""
from waveform_viewer.ai.expert import WaveformExpert
from waveform_viewer.ai.tool_registry import get_default_registry
import pandas as pd

def batch_power_quality_assessment(cfg_files: List[str]) -> pd.DataFrame:
    """
    批量评估电能质量

    Returns:
        DataFrame 包含所有文件的电能质量指标
    """
    registry = get_default_registry()

    results = []

    for cfg_file in cfg_files:
        # 使用工具直接计算（不需要 LLM）
        pq_metrics = registry.execute('calculate_power_quality_metrics', cfg_path=cfg_file)

        if pq_metrics['status'] == 'success':
            results.append({
                'file': cfg_file,
                'voltage_deviation_A': pq_metrics['voltage_deviation']['A'],
                'voltage_unbalance': pq_metrics['voltage_unbalance'],
                'thd_voltage_A': pq_metrics['thd_voltage']['A'],
                'frequency_deviation': pq_metrics['frequency_deviation'],
                'assessment': pq_metrics['overall_assessment']
            })

    df = pd.DataFrame(results)

    # 标记不合格项
    df['voltage_deviation_ok'] = df['voltage_deviation_A'].abs() < 7  # GB/T 12325
    df['voltage_unbalance_ok'] = df['voltage_unbalance'] < 2
    df['thd_ok'] = df['thd_voltage_A'] < 5
    df['frequency_ok'] = df['frequency_deviation'].abs() < 0.5

    return df


# 使用
cfg_files = [
    "waves/case1.cfg",
    "waves/case2.cfg",
    "waves/case3.cfg"
]

df = batch_power_quality_assessment(cfg_files)

# 输出到 Excel
df.to_excel("power_quality_report.xlsx", index=False)

# 打印不合格项
failed = df[~(df['voltage_deviation_ok'] & df['voltage_unbalance_ok'] &
              df['thd_ok'] & df['frequency_ok'])]

print(f"\n发现 {len(failed)} 个电能质量不合格的文件：")
print(failed[['file', 'assessment']])
```

### 案例 3: 对比分析工具

```python
"""
对比分析多个录波文件
"""
def compare_multiple_waveforms(cfg_files: List[str], reference_file: str):
    """
    将多个文件与参考文件对比
    """
    from waveform_viewer.ai.tool_registry import get_default_registry

    registry = get_default_registry()

    print(f"参考文件: {reference_file}\n")

    for cfg_file in cfg_files:
        print(f"\n对比文件: {cfg_file}")
        print("=" * 60)

        # 对比 A 相电压
        comparison = registry.execute(
            'compare_waveforms',
            cfg_paths=[reference_file, cfg_file],
            channel_name="A相电压"
        )

        if comparison['status'] == 'success':
            sim = comparison['comparison']['similarity']
            corr = comparison['comparison']['correlation']

            print(f"相似度: {sim}")
            print(f"相关系数: {corr:.3f}")

            if corr > 0.95:
                print("✅ 波形高度相似")
            elif corr > 0.8:
                print("⚠️ 波形存在差异")
            else:
                print("❌ 波形差异显著")

        # 基线偏差分析
        deviation = registry.execute(
            'baseline_deviation_analysis',
            cfg_path=cfg_file,
            baseline_cfg_path=reference_file
        )

        if deviation['status'] == 'success':
            for dev in deviation['deviations']:
                if dev['exceeds_threshold']:
                    print(f"\n⚠️ {dev['channel']} 偏差 {dev['deviation_percent']:.1f}%")


# 使用
compare_multiple_waveforms(
    cfg_files=[
        "waves/day1.cfg",
        "waves/day2.cfg",
        "waves/day3.cfg"
    ],
    reference_file="waves/baseline.cfg"
)
```

### 案例 4: 生成 HTML 报告

```python
"""
生成美观的 HTML 分析报告
"""
def generate_html_report(cfg_path: str, output_path: str):
    """生成包含可视化的 HTML 报告"""

    from waveform_viewer.ai.expert import WaveformExpert
    from waveform_viewer.ai.tool_registry import get_default_registry
    import openai
    from dotenv import load_dotenv

    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    registry = get_default_registry()
    expert = WaveformExpert(openai, registry)

    # 执行分析
    analysis = expert.analyze(
        query="请对这个录波进行全面分析，包括故障诊断、保护评价、电能质量评估",
        cfg_path=cfg_path
    )

    # 生成可视化
    viz_result = registry.execute(
        'create_visualization',
        cfg_path=cfg_path,
        channel_names=["A相电压", "A相电流"],
        output_path="temp_viz.html"
    )

    # 读取可视化 HTML
    with open("temp_viz.html", "r") as f:
        viz_html = f.read()

    # 生成报告 HTML
    html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>录波分析报告</title>
    <style>
        body {{
            font-family: "Microsoft YaHei", Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .section {{
            background: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        pre {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 COMTRADE 录波分析报告</h1>
        <p>文件: {cfg_path}</p>
        <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="section">
        <h2>🤖 AI 分析结论</h2>
        <pre>{analysis}</pre>
    </div>

    <div class="section">
        <h2>📈 波形可视化</h2>
        {viz_html}
    </div>

    <div class="section">
        <h2>ℹ️ 分析说明</h2>
        <p>本报告由 ComtradeReader AI 专家系统自动生成。</p>
        <p>使用模型: GPT-4</p>
    </div>
</body>
</html>
    """

    # 保存报告
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_template)

    print(f"✅ HTML 报告已生成: {output_path}")


# 使用
generate_html_report(
    cfg_path="waves/20241030_115240/20241030_115240.cfg",
    output_path="analysis_report.html"
)
```

---

## 调试和测试示例

### 示例 10: 调试工具调用

```python
"""
调试模式：查看工具调用的详细信息
"""
import logging

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class DebugExpert(WaveformExpert):
    """带调试功能的专家系统"""

    def analyze(self, query: str, cfg_path: str):
        print(f"\n{'='*60}")
        print(f"🔍 开始分析")
        print(f"问题: {query}")
        print(f"文件: {cfg_path}")
        print(f"{'='*60}\n")

        # ... 原有逻辑 ...

        # 在工具调用时打印详细信息
        print(f"\n🛠️  调用工具: {func_name}")
        print(f"📥 参数: {json.dumps(func_args, ensure_ascii=False, indent=2)}")

        result = self.registry.execute(func_name, **func_args)

        print(f"📤 结果: {json.dumps(result, ensure_ascii=False, indent=2)[:500]}...")

        # 继续...


# 使用
debug_expert = DebugExpert(openai, registry)
debug_expert.analyze("电压有跌落吗？", "waves/test.cfg")
```

### 示例 11: 单元测试

```python
"""
工具函数的单元测试
"""
import unittest
from waveform_viewer.ai.tools.basic_analysis import calculate_statistics, detect_voltage_sags

class TestAnalysisTools(unittest.TestCase):
    """测试分析工具"""

    def setUp(self):
        self.test_cfg = "waves/20241030_115240/20241030_115240.cfg"

    def test_calculate_statistics(self):
        """测试统计计算"""
        result = calculate_statistics(self.test_cfg, "A相电压")

        self.assertEqual(result['status'], 'success')
        self.assertIn('statistics', result)
        self.assertIn('min', result['statistics'])
        self.assertIn('max', result['statistics'])
        self.assertGreater(result['statistics']['max'], result['statistics']['min'])

    def test_detect_voltage_sags(self):
        """测试电压跌落检测"""
        result = detect_voltage_sags(self.test_cfg, threshold_percent=90)

        self.assertEqual(result['status'], 'success')
        self.assertIn('events', result)
        self.assertIsInstance(result['events'], list)

    def test_invalid_file(self):
        """测试无效文件处理"""
        result = calculate_statistics("nonexistent.cfg", "A相电压")

        self.assertEqual(result['status'], 'error')
        self.assertIn('message', result)


if __name__ == '__main__':
    unittest.main()
```

---

## 性能优化示例

### 示例 12: 缓存工具结果

```python
"""
缓存工具执行结果，避免重复计算
"""
import hashlib
import json
from functools import wraps

class ToolResultCache:
    """工具结果缓存"""

    def __init__(self):
        self._cache = {}

    def get_cache_key(self, tool_name: str, **kwargs) -> str:
        """生成缓存键"""
        key_str = f"{tool_name}:{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def cached_execute(self, tool_func):
        """缓存装饰器"""
        @wraps(tool_func)
        def wrapper(**kwargs):
            cache_key = self.get_cache_key(tool_func.__name__, **kwargs)

            # 检查缓存
            if cache_key in self._cache:
                print(f"  💾 使用缓存: {tool_func.__name__}")
                return self._cache[cache_key]

            # 执行工具
            result = tool_func(**kwargs)

            # 保存到缓存
            self._cache[cache_key] = result

            return result

        return wrapper


# 使用缓存
cache = ToolResultCache()

@cache.cached_execute
def calculate_statistics(cfg_path, channel_name):
    # 实际的工具函数
    ...
```

---

**更多示例持续更新...**

如有问题或需要更多示例，请参考：
- [完整开发指南](./AI_INTEGRATION_GUIDE.md)
- [工具函数参考](./AI_TOOLS_REFERENCE.md)
- [架构设计文档](./AI_ARCHITECTURE.md)

---

**文档版本**: 1.0
**最后更新**: 2024-11-10
