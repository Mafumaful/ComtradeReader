 关于集成大模型专家系统的架构分析报告

  一、项目现状总结

  ComtradeReader 是一个成熟的电力系统暂态波形分析工具（v2.0.4），用于读取、分析和可视化 COMTRADE 标准格式的电力故障录波数据。代码库约
  2859 行，架构清晰，模块化设计良好。

  核心能力：
  - 解析 COMTRADE 文件（.cfg/.dat/.hdr/.rpt）
  - 提取模拟量和数字量通道数据
  - 交互式 HTML 波形可视化
  - 插件系统支持功能扩展
  - 事件时间线分析（HDR 文件）
  - 统计分析和数据导出

  ---
  二、现有架构优势

  你的项目已经具备了极佳的 AI 集成基础：

  1. ✅ 插件架构：waveform_viewer/plugins/ 提供了标准化的扩展机制
  2. ✅ 策略模式：通道选择和可视化都支持自定义策略
  3. ✅ 清晰的数据模型：ComtradeReader 类封装了所有波形数据
  4. ✅ 编程式 API：不仅有 CLI，还可以作为 Python 库调用
  5. ✅ 工具函数基础：已有统计分析、CSV/JSON 导出等工具插件

  ---
  三、建议的集成架构

  我建议采用三层架构来集成大模型专家系统：

  ┌─────────────────────────────────────────────────────────┐
  │              Layer 3: AI Expert Interface               │
  │  (自然语言查询 → 意图识别 → 工具调用 → 结果解释)          │
  └─────────────────────────────────────────────────────────┘
                              ↓
  ┌─────────────────────────────────────────────────────────┐
  │           Layer 2: Analysis Tool Registry               │
  │   (标准化的工具函数，供 LLM Function Calling 使用)         │
  └─────────────────────────────────────────────────────────┘
                              ↓
  ┌─────────────────────────────────────────────────────────┐
  │         Layer 1: Existing Core Components               │
  │    (ComtradeReader, Plugins, Visualizers - 保持不变)     │
  └─────────────────────────────────────────────────────────┘

  ---
  四、具体实施方案

  阶段 1：构建工具注册表（Tool Registry）

  在 waveform_viewer/ai/ 下创建新模块：

  waveform_viewer/ai/
  ├── __init__.py
  ├── tools/                      # AI 可调用的工具函数
  │   ├── __init__.py
  │   ├── data_access.py         # 数据读取工具
  │   ├── basic_analysis.py      # 基础分析（统计、峰值检测）
  │   ├── advanced_analysis.py   # 高级分析（FFT、THD、谐波）
  │   ├── event_analysis.py      # 事件分析（故障时序、保护动作）
  │   ├── comparison.py          # 多文件对比
  │   └── reporting.py           # 报告生成
  ├── tool_registry.py           # 工具注册和元数据管理
  └── function_schemas.py        # LLM Function Calling 的 JSON Schema

  工具函数示例：

  # waveform_viewer/ai/tools/data_access.py
  def list_available_channels(cfg_path: str) -> Dict[str, Any]:
      """列出所有可用通道及其元数据"""
      reader = ComtradeReader(cfg_path)
      return {
          "station": reader.station_name,
          "analog_channels": [
              {"index": ch.index, "name": ch.name, "unit": ch.unit}
              for ch in reader.analog_channels
          ],
          "digital_channels": reader.digital_channels,
          "sample_rate": reader.sample_rate,
          "duration": reader.time_values[-1]
      }

  def get_channel_data(cfg_path: str, channel_name: str) -> Dict:
      """获取指定通道的时序数据"""
      reader = ComtradeReader(cfg_path)
      channel = reader.get_channel_by_name(channel_name)
      if not channel:
          raise ValueError(f"Channel {channel_name} not found")

      data = reader.get_analog_data(channel.index)
      return {
          "name": channel.name,
          "unit": channel.unit,
          "time": reader.time_values,
          "values": data
      }

  # waveform_viewer/ai/tools/basic_analysis.py
  def calculate_statistics(cfg_path: str, channel_name: str) -> Dict:
      """计算通道统计特征"""
      reader = ComtradeReader(cfg_path)
      channel = reader.get_channel_by_name(channel_name)
      data = np.array(reader.get_analog_data(channel.index))

      return {
          "min": float(np.min(data)),
          "max": float(np.max(data)),
          "mean": float(np.mean(data)),
          "std": float(np.std(data)),
          "rms": float(np.sqrt(np.mean(data**2))),
          "peak_to_peak": float(np.ptp(data))
      }

  def detect_voltage_sags(cfg_path: str, threshold_percent: float = 90) -> List[Dict]:
      """检测电压跌落事件"""
      reader = ComtradeReader(cfg_path)
      # 查找电压通道
      voltage_channels = [ch for ch in reader.analog_channels if '电压' in ch.name]

      events = []
      for ch in voltage_channels:
          data = np.array(reader.get_analog_data(ch.index))
          threshold = threshold_percent  # 假设单位是百分比
          sag_indices = np.where(data < threshold)[0]

          if len(sag_indices) > 0:
              events.append({
                  "channel": ch.name,
                  "start_time": reader.time_values[sag_indices[0]],
                  "duration": (sag_indices[-1] - sag_indices[0]) / reader.sample_rate,
                  "min_value": float(np.min(data[sag_indices]))
              })

      return events

  # waveform_viewer/ai/tools/advanced_analysis.py
  from scipy import signal
  from scipy.fft import fft, fftfreq

  def perform_fft_analysis(cfg_path: str, channel_name: str) -> Dict:
      """频域分析（FFT）"""
      reader = ComtradeReader(cfg_path)
      channel = reader.get_channel_by_name(channel_name)
      data = np.array(reader.get_analog_data(channel.index))

      # FFT 计算
      N = len(data)
      yf = fft(data)
      xf = fftfreq(N, 1/reader.sample_rate)[:N//2]

      # 只返回正频率部分
      magnitude = 2.0/N * np.abs(yf[:N//2])

      # 找到主要频率分量（前 5 个峰值）
      peaks, _ = signal.find_peaks(magnitude, height=np.max(magnitude)*0.1)
      top_freqs = sorted(zip(xf[peaks], magnitude[peaks]),
                        key=lambda x: x[1], reverse=True)[:5]

      return {
          "fundamental_freq": reader.frequency,
          "sample_rate": reader.sample_rate,
          "frequencies": xf.tolist()[:100],  # 限制返回数据量
          "magnitudes": magnitude.tolist()[:100],
          "dominant_frequencies": [
              {"frequency": f, "magnitude": m} for f, m in top_freqs
          ]
      }

  def calculate_thd(cfg_path: str, channel_name: str) -> float:
      """计算总谐波畸变率（THD）"""
      fft_result = perform_fft_analysis(cfg_path, channel_name)
      freqs = fft_result["dominant_frequencies"]

      if len(freqs) < 2:
          return 0.0

      fundamental = freqs[0]["magnitude"]
      harmonics = sum(f["magnitude"]**2 for f in freqs[1:])

      return 100 * np.sqrt(harmonics) / fundamental if fundamental > 0 else 0

  # waveform_viewer/ai/tools/event_analysis.py
  def extract_fault_timeline(cfg_path: str) -> Dict:
      """提取故障事件时间线（从 HDR 文件）"""
      hdr_path = cfg_path.replace('.cfg', '.hdr')
      if not os.path.exists(hdr_path):
          return {"error": "HDR file not found"}

      hdr_reader = HdrReader(hdr_path)

      return {
          "fault_start_time": hdr_reader.fault_start_time,
          "trip_info": [
              {
                  "time": trip.time,
                  "trip_signal": trip.trip_signal,
                  "relay": trip.relay_name
              }
              for trip in hdr_reader.trip_infos
          ],
          "digital_events": [
              {
                  "time": event.time,
                  "name": event.name,
                  "status": event.status
              }
              for event in hdr_reader.digital_events
          ]
      }

  ---
  阶段 2：创建工具注册表和 Function Schema

  # waveform_viewer/ai/tool_registry.py
  from typing import Dict, List, Callable
  import inspect

  class ToolRegistry:
      """AI 工具注册中心"""

      def __init__(self):
          self._tools: Dict[str, Callable] = {}
          self._schemas: Dict[str, Dict] = {}

      def register(self, func: Callable, schema: Dict):
          """注册工具函数及其 schema"""
          self._tools[func.__name__] = func
          self._schemas[func.__name__] = schema

      def get_tool(self, name: str) -> Callable:
          """获取工具函数"""
          return self._tools.get(name)

      def get_all_schemas(self) -> List[Dict]:
          """获取所有工具的 Function Calling schemas"""
          return list(self._schemas.values())

      def execute(self, tool_name: str, **kwargs) -> Any:
          """执行工具函数"""
          tool = self.get_tool(tool_name)
          if not tool:
              raise ValueError(f"Tool {tool_name} not found")
          return tool(**kwargs)

  # 全局注册表实例
  registry = ToolRegistry()

  # waveform_viewer/ai/function_schemas.py
  """LLM Function Calling 的 JSON Schema 定义"""

  TOOL_SCHEMAS = [
      {
          "name": "list_available_channels",
          "description": "列出 COMTRADE 文件中所有可用的通道信息，包括模拟量和数字量通道",
          "parameters": {
              "type": "object",
              "properties": {
                  "cfg_path": {
                      "type": "string",
                      "description": "COMTRADE 配置文件（.cfg）的路径"
                  }
              },
              "required": ["cfg_path"]
          }
      },
      {
          "name": "calculate_statistics",
          "description": "计算指定通道的统计特征，包括最小值、最大值、均值、标准差、RMS 等",
          "parameters": {
              "type": "object",
              "properties": {
                  "cfg_path": {"type": "string", "description": "CFG 文件路径"},
                  "channel_name": {"type": "string", "description": "通道名称或匹配模式"}
              },
              "required": ["cfg_path", "channel_name"]
          }
      },
      {
          "name": "detect_voltage_sags",
          "description": "检测电压跌落事件，返回跌落时刻、持续时间和最小值",
          "parameters": {
              "type": "object",
              "properties": {
                  "cfg_path": {"type": "string"},
                  "threshold_percent": {
                      "type": "number",
                      "description": "电压跌落阈值（百分比），默认 90%"
                  }
              },
              "required": ["cfg_path"]
          }
      },
      {
          "name": "perform_fft_analysis",
          "description": "对指定通道进行快速傅里叶变换（FFT），分析频域特征",
          "parameters": {
              "type": "object",
              "properties": {
                  "cfg_path": {"type": "string"},
                  "channel_name": {"type": "string"}
              },
              "required": ["cfg_path", "channel_name"]
          }
      },
      {
          "name": "extract_fault_timeline",
          "description": "从 HDR 文件中提取故障事件时间线，包括故障时刻、保护动作、跳闸信息",
          "parameters": {
              "type": "object",
              "properties": {
                  "cfg_path": {"type": "string"}
              },
              "required": ["cfg_path"]
          }
      }
      # ... 更多工具
  ]

  ---
  阶段 3：构建 AI 专家接口

  # waveform_viewer/ai/expert.py
  from typing import List, Dict, Any
  import json

  class WaveformExpert:
      """电力波形分析专家系统（LLM 驱动）"""

      def __init__(self, llm_client, tool_registry):
          """
          Args:
              llm_client: LLM 客户端（OpenAI/Claude/其他支持 Function Calling 的 API）
              tool_registry: ToolRegistry 实例
          """
          self.llm = llm_client
          self.registry = tool_registry
          self.conversation_history = []

      def analyze(self, query: str, cfg_path: str) -> str:
          """
          自然语言查询接口
          
          Args:
              query: 用户的自然语言问题，例如：
                     "这个波形有电压跌落吗？"
                     "A相电流的频谱分析显示什么？"
                     "故障是什么时候发生的？保护是否正确动作？"
              cfg_path: 要分析的 COMTRADE 文件路径
          
          Returns:
              专家系统的分析结果（自然语言）
          """
          # 构建系统提示词
          system_prompt = f"""你是一个电力系统暂态波形分析专家。
  你可以调用各种分析工具来帮助回答用户关于 COMTRADE 波形文件的问题。

  当前分析的文件：{cfg_path}

  你可以使用的工具包括：
  - 数据访问：查看通道列表、读取波形数据
  - 基础分析：统计特征、峰值检测、电压跌落检测
  - 高级分析：FFT频谱分析、谐波分析、THD 计算
  - 事件分析：故障时序、保护动作分析
  - 对比分析：多文件对比、基线偏差

  请根据用户的问题，选择合适的工具进行分析，并给出专业的解释。"""

          # 调用 LLM（支持 Function Calling）
          messages = [
              {"role": "system", "content": system_prompt},
              *self.conversation_history,
              {"role": "user", "content": query}
          ]

          response = self.llm.chat.completions.create(
              model="gpt-4",  # 或其他模型
              messages=messages,
              functions=self.registry.get_all_schemas(),
              function_call="auto"
          )

          # 处理 Function Calling
          while response.choices[0].finish_reason == "function_call":
              func_call = response.choices[0].message.function_call
              func_name = func_call.name
              func_args = json.loads(func_call.arguments)

              # 自动注入 cfg_path（如果工具需要）
              if "cfg_path" in func_args and not func_args["cfg_path"]:
                  func_args["cfg_path"] = cfg_path

              # 执行工具
              result = self.registry.execute(func_name, **func_args)

              # 将结果返回给 LLM
              messages.append({
                  "role": "function",
                  "name": func_name,
                  "content": json.dumps(result, ensure_ascii=False)
              })

              response = self.llm.chat.completions.create(
                  model="gpt-4",
                  messages=messages,
                  functions=self.registry.get_all_schemas(),
                  function_call="auto"
              )

          # 获取最终回答
          answer = response.choices[0].message.content

          # 更新对话历史
          self.conversation_history.append({"role": "user", "content": query})
          self.conversation_history.append({"role": "assistant", "content": answer})

          return answer

      def batch_analyze(self, tasks: List[Dict[str, str]]) -> List[str]:
          """批量分析多个任务"""
          return [self.analyze(task["query"], task["cfg_path"]) for task in tasks]

      def reset_conversation(self):
          """重置对话历史"""
          self.conversation_history = []

  ---
  阶段 4：集成到现有应用

  # waveform_viewer/app.py（扩展现有类）

  class WaveformViewerApp:
      # ... 现有代码 ...

      def __init__(self, waves_dir, use_simple_menu=False, enable_ai=False):
          # ... 现有初始化 ...

          # AI 专家系统（可选）
          self.ai_expert = None
          if enable_ai:
              from waveform_viewer.ai.expert import WaveformExpert
              from waveform_viewer.ai.tool_registry import registry
              # 这里需要配置 LLM 客户端
              self.ai_expert = WaveformExpert(llm_client, registry)

      def run_ai_query(self, cfg_path: str):
          """AI 查询模式"""
          if not self.ai_expert:
              print("AI 功能未启用")
              return

          print("\n=== AI 专家分析模式 ===")
          print("输入您的问题（输入 'exit' 退出）：\n")

          while True:
              query = input("您的问题 > ")
              if query.lower() == 'exit':
                  break

              print("\n🤖 专家正在分析...")
              answer = self.ai_expert.analyze(query, cfg_path)
              print(f"\n📊 分析结果：\n{answer}\n")

  # main.py（添加 AI 模式入口）

  def main():
      parser = argparse.ArgumentParser(description='COMTRADE Waveform Viewer')
      parser.add_argument('--simple', action='store_true', help='使用简化菜单模式')
      parser.add_argument('--ai', action='store_true', help='启用 AI 专家分析')
      args = parser.parse_args()

      app = WaveformViewerApp("waves/",
                             use_simple_menu=args.simple,
                             enable_ai=args.ai)
      app.run()

  ---
  五、使用场景示例

  启用 AI 模式后的用户体验：

  $ python main.py --ai

  === AI 专家分析模式 ===
  您的问题 > 这个录波文件记录的是什么类型的故障？

  🤖 专家正在分析...

  📊 分析结果：
  根据对波形数据的分析，这是一次**单相接地故障**。主要依据：

  1. **故障时刻**：11:52:40.123（从 HDR 文件提取）
  2. **电压特征**：A 相电压跌落至 15%，B/C 相电压上升至 110%（典型的单相接地特征）
  3. **电流特征**：A 相电流突增至 8.5kA，零序电流出现明显增量
  4. **保护动作**：距离保护 I 段在故障后 23ms 动作，跳闸正确
  5. **频谱分析**：A 相电流出现明显的 2 次谐波分量（故障电弧特征）

  建议进一步检查：
  - 故障点定位（根据行波或阻抗计算）
  - 接地电阻值估算
  - 保护配置是否需要优化

  您的问题 > A 相电流的谐波含量高吗？

  🤖 专家正在分析...

  📊 分析结果：
  A 相电流的谐波分析结果：

  - **总谐波畸变率（THD）**：12.3%（正常运行时应 <5%）
  - **主要谐波分量**：
    - 基波（50Hz）：8.5kA
    - 2 次谐波（100Hz）：0.85kA（10%）
    - 5 次谐波（250Hz）：0.25kA（2.9%）

  **结论**：谐波含量偏高，主要是故障电弧导致的 2 次谐波。这是典型的接地故障特征。

  您的问题 > exit

  ---
  六、技术选型建议

  LLM 选择：

  | 选项                | 优点                         | 缺点                          | 推荐场景   |
  |-------------------|----------------------------|-----------------------------|--------|
  | OpenAI GPT-4      | Function Calling 成熟、API 稳定 | 需要联网、有成本                    | 云端部署   |
  | Claude 3.5 Sonnet | 推理能力强、Token 窗口大            | Function Calling 较新         | 复杂分析任务 |
  | 本地模型（Llama 3.1）   | 无网络要求、成本低                  | 需要 GPU、Function Calling 能力弱 | 私有化部署  |
  | 混合方案              | 简单任务本地、复杂任务云端              | 架构复杂                        | 企业级应用  |

  依赖库：

  # requirements-ai.txt
  openai >= 1.0.0          # OpenAI API（如果使用 GPT）
  anthropic >= 0.7.0       # Claude API（如果使用 Claude）
  langchain >= 0.1.0       # 可选：工具链管理
  scipy >= 1.10.0          # FFT/信号处理
  scikit-learn >= 1.2.0    # 可选：机器学习特征
  pydantic >= 2.0.0        # 数据验证

  ---
  七、实施路线图

  第一阶段（2-3 周）：基础工具层
  - 创建 waveform_viewer/ai/tools/ 目录
  - 实现 10-15 个核心工具函数（数据访问、基础分析、事件分析）
  - 编写工具的单元测试
  - 创建 Function Schema 定义

  第二阶段（1-2 周）：注册表和专家系统
  - 实现 ToolRegistry 类
  - 实现 WaveformExpert 类
  - 集成 LLM API（OpenAI/Claude）
  - 测试 Function Calling 流程

  第三阶段（1 周）：UI 集成
  - 在 main.py 添加 --ai 参数
  - 实现交互式 AI 查询界面
  - 添加对话历史管理
  - 错误处理和用户体验优化

  第四阶段（持续）：高级功能
  - 添加多文件对比工具
  - 实现故障类型自动分类
  - 添加报告生成功能
  - 集成向量数据库（历史案例检索）
  - 支持流式输出（大模型 streaming）

  ---
  八、关键设计建议

  ✅ 保持架构整洁：

  - 新增的 AI 模块应该完全独立，不修改现有核心代码
  - 通过依赖注入（Dependency Injection）方式集成
  - AI 功能应该是可选的（通过 --ai 参数启用）

  ✅ 工具函数设计原则：

  - 原子性：每个工具只做一件事
  - 无状态：工具函数应该是纯函数（输入相同则输出相同）
  - 返回结构化数据：使用 Dict/List，便于 JSON 序列化
  - 异常处理：所有工具都应该捕获异常并返回错误信息

  ✅ 性能优化：

  - 大文件分析前先检查文件大小
  - FFT 等耗时操作应该有缓存机制
  - 对于大量通道，支持批量分析和并行处理

  ✅ 安全性：

  - 文件路径验证（防止路径遍历攻击）
  - LLM 输出过滤（防止 Prompt Injection）
  - API Key 管理（使用环境变量，不硬编码）

  ---
  九、示例代码片段（快速启动）

  如果你想立即开始，可以先创建一个最小可行版本：

  # waveform_viewer/ai/simple_expert.py（最简实现）

  from waveform_viewer.core.reader import ComtradeReader
  import numpy as np

  class SimpleExpert:
      """简单专家系统（基于规则，不依赖 LLM）"""

      def analyze_voltage_sag(self, cfg_path: str) -> str:
          """分析电压跌落"""
          reader = ComtradeReader(cfg_path)

          results = []
          for ch in reader.analog_channels:
              if '电压' in ch.name:
                  data = np.array(reader.get_analog_data(ch.index))
                  min_val = np.min(data)

                  if min_val < 80:  # 电压跌落阈值
                      results.append(
                          f"{ch.name} 发生电压跌落，最低值 {min_val:.2f}{ch.unit}"
                      )

          return "\n".join(results) if results else "未检测到电压跌落"

      def analyze_fault_type(self, cfg_path: str) -> str:
          """故障类型判断（简单规则）"""
          reader = ComtradeReader(cfg_path)

          # 获取三相电压
          voltages = {}
          for phase in ['A', 'B', 'C']:
              ch = reader.get_channel_by_name(f'{phase}相.*电压')
              if ch:
                  voltages[phase] = np.min(reader.get_analog_data(ch.index))

          # 简单判断逻辑
          low_voltage_phases = [p for p, v in voltages.items() if v < 80]

          if len(low_voltage_phases) == 1:
              return f"单相接地故障（{low_voltage_phases[0]}相）"
          elif len(low_voltage_phases) == 2:
              return f"两相短路故障（{','.join(low_voltage_phases)}）"
          elif len(low_voltage_phases) == 3:
              return "三相短路故障"
          else:
              return "未检测到明显故障"

  # 使用示例
  expert = SimpleExpert()
  print(expert.analyze_voltage_sag("waves/20241030_115240/20241030_115240.cfg"))
  print(expert.analyze_fault_type("waves/20241030_115240/20241030_115240.cfg"))

  这个简单版本可以先验证架构可行性，之后再逐步替换为 LLM 驱动的智能分析。

  ---
  十、总结

  你的项目非常适合集成 AI 专家系统，因为：

  1. ✅ 已有插件架构：扩展点清晰
  2. ✅ 数据模型完善：ComtradeReader 提供了良好的数据访问接口
  3. ✅ 工具基础良好：已有统计、导出等工具可以直接复用
  4. ✅ 文档完善：有详细的开发者指南

  推荐的实施路径：
  1. 先实现工具函数层（独立于 LLM）
  2. 集成简单的规则专家系统验证架构
  3. 接入 LLM Function Calling 实现智能分析
  4. 逐步添加高级分析能力（FFT、故障分类、报告生成）

  预计工作量：
  - 基础版本（10-15 个工具 + LLM 集成）：3-4 周
  - 完整版本（30+ 工具 + 高级功能）：2-3 个月