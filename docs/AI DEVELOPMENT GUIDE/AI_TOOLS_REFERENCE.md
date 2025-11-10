# AI 工具函数库参考

本文档详细描述了 ComtradeReader AI 专家系统中所有可用的分析工具函数。

## 目录

- [数据访问工具](#数据访问工具)
- [基础分析工具](#基础分析工具)
- [高级分析工具](#高级分析工具)
- [事件分析工具](#事件分析工具)
- [对比分析工具](#对比分析工具)
- [报告生成工具](#报告生成工具)

---

## 数据访问工具

### list_available_channels

列出 COMTRADE 文件中所有可用的通道信息。

**函数签名:**
```python
def list_available_channels(cfg_path: str) -> Dict[str, Any]
```

**参数:**
- `cfg_path` (str): COMTRADE 配置文件路径

**返回值:**
```python
{
    "status": "success",
    "station_name": "变电站名称",
    "frequency": 50.0,
    "sample_rate": 300,
    "num_samples": 6000,
    "duration": 20.0,
    "analog_channels": [
        {
            "index": 1,
            "name": "机端电压百分值",
            "phase": "A",
            "unit": "%",
            "circuit_component": "VT"
        },
        # ... 更多通道
    ],
    "digital_channels": [
        {
            "index": 1,
            "name": "保护跳闸"
        },
        # ...
    ]
}
```

**使用场景:**
- 用户询问"这个文件有哪些通道？"
- 在不确定通道名称时，先查看可用通道列表
- 了解文件的基本信息（采样率、时长等）

**示例:**
```python
result = list_available_channels("waves/test.cfg")
print(f"共有 {len(result['analog_channels'])} 个模拟量通道")
print(f"采样率: {result['sample_rate']} Hz")
```

---

### get_channel_data

获取指定通道的时序数据。

**函数签名:**
```python
def get_channel_data(cfg_path: str, channel_name: str) -> Dict[str, Any]
```

**参数:**
- `cfg_path` (str): COMTRADE 配置文件路径
- `channel_name` (str): 通道名称或匹配模式（如 "A相电压"）

**返回值:**
```python
{
    "status": "success",
    "channel": {
        "name": "机端电压百分值",
        "unit": "%",
        "phase": "A"
    },
    "time": [0.0, 0.00333, 0.00667, ...],  # 时间序列（秒）
    "values": [100.5, 100.3, 100.1, ...],  # 数据值
    "sample_count": 6000,
    "downsampled": True  # 是否进行了降采样
}
```

**注意事项:**
- 为避免返回过多数据，函数会自动降采样到最多 1000 个点
- 如果通道名称不完全匹配，会尝试模糊匹配

**使用场景:**
- 需要绘制波形图
- 进行自定义的数据处理
- 检查某个时刻的具体数值

**示例:**
```python
result = get_channel_data("waves/test.cfg", "A相电压")
min_voltage = min(result['values'])
print(f"最低电压: {min_voltage} {result['channel']['unit']}")
```

---

### get_channel_value_at_time

获取指定通道在特定时刻的数值。

**函数签名:**
```python
def get_channel_value_at_time(
    cfg_path: str,
    channel_name: str,
    time_seconds: float
) -> Dict[str, Any]
```

**参数:**
- `cfg_path` (str): COMTRADE 配置文件路径
- `channel_name` (str): 通道名称
- `time_seconds` (float): 时间点（秒）

**返回值:**
```python
{
    "status": "success",
    "channel": "A相电压",
    "time": 5.123,
    "value": 98.5,
    "unit": "%",
    "nearest_sample_time": 5.12333  # 实际采样点时间
}
```

**使用场景:**
- 查询故障发生时刻的电压/电流值
- 对比不同通道在同一时刻的数值
- 验证保护动作时的系统状态

---

## 基础分析工具

### calculate_statistics

计算指定通道的统计特征。

**函数签名:**
```python
def calculate_statistics(cfg_path: str, channel_name: str) -> Dict[str, Any]
```

**参数:**
- `cfg_path` (str): COMTRADE 配置文件路径
- `channel_name` (str): 通道名称

**返回值:**
```python
{
    "status": "success",
    "channel": {
        "name": "A相电压",
        "unit": "%"
    },
    "statistics": {
        "min": 15.2,          # 最小值
        "max": 105.8,         # 最大值
        "mean": 98.5,         # 平均值
        "median": 99.2,       # 中位数
        "std": 12.3,          # 标准差
        "rms": 99.1,          # 有效值（RMS）
        "peak_to_peak": 90.6, # 峰峰值
        "variance": 151.29    # 方差
    }
}
```

**使用场景:**
- 快速了解通道的数值范围
- 评估信号的稳定性（标准差）
- 计算电压/电流的有效值

**示例:**
```python
result = calculate_statistics("waves/test.cfg", "A相电流")
rms = result['statistics']['rms']
print(f"A相电流有效值: {rms} A")
```

---

### detect_voltage_sags

检测电压跌落事件。

**函数签名:**
```python
def detect_voltage_sags(
    cfg_path: str,
    threshold_percent: float = 90.0
) -> Dict[str, Any]
```

**参数:**
- `cfg_path` (str): COMTRADE 配置文件路径
- `threshold_percent` (float): 电压跌落阈值（百分比），默认 90%

**返回值:**
```python
{
    "status": "success",
    "events": [
        {
            "channel": "A相电压百分值",
            "phase": "A",
            "start_time": 10.123,     # 跌落开始时间（秒）
            "end_time": 10.456,       # 跌落结束时间（秒）
            "duration": 0.333,        # 持续时间（秒）
            "min_value": 15.2,        # 最低值
            "threshold": 90.0         # 使用的阈值
        },
        # ... 其他相的跌落事件
    ],
    "event_count": 1
}
```

**使用场景:**
- 检测电压质量问题
- 识别故障期间的电压跌落
- 评估电压暂降的严重程度

**阈值建议:**
- 高压系统：85-90%
- 低压系统：70-80%
- 严格标准：95%

**示例:**
```python
result = detect_voltage_sags("waves/test.cfg", threshold_percent=85)
if result['event_count'] > 0:
    for event in result['events']:
        print(f"{event['channel']} 在 {event['start_time']}s 发生跌落")
        print(f"  最低值: {event['min_value']}%, 持续 {event['duration']*1000:.1f}ms")
```

---

### detect_current_surges

检测电流突变/浪涌事件。

**函数签名:**
```python
def detect_current_surges(
    cfg_path: str,
    threshold_multiplier: float = 2.0
) -> Dict[str, Any]
```

**参数:**
- `cfg_path` (str): COMTRADE 配置文件路径
- `threshold_multiplier` (float): 电流突变阈值（倍数），默认 2.0（两倍于正常值）

**返回值:**
```python
{
    "status": "success",
    "events": [
        {
            "channel": "A相电流",
            "phase": "A",
            "time": 10.123,
            "peak_value": 8500.0,      # 峰值（A）
            "normal_level": 1200.0,    # 正常水平（A）
            "surge_ratio": 7.08        # 突变倍数
        }
    ],
    "event_count": 1
}
```

**使用场景:**
- 检测短路故障
- 识别启动冲击电流
- 分析过载情况

---

### find_peaks

检测通道数据中的峰值点。

**函数签名:**
```python
def find_peaks(
    cfg_path: str,
    channel_name: str,
    prominence: float = None
) -> Dict[str, Any]
```

**参数:**
- `cfg_path` (str): COMTRADE 配置文件路径
- `channel_name` (str): 通道名称
- `prominence` (float, optional): 峰值显著性阈值，默认自动计算

**返回值:**
```python
{
    "status": "success",
    "channel": {
        "name": "A相电流",
        "unit": "A"
    },
    "peaks": [
        {
            "time": 10.123,
            "value": 8500.0,
            "prominence": 7300.0  # 峰值突出程度
        },
        # ... 最多返回前 20 个峰值
    ],
    "total_peaks": 45
}
```

**使用场景:**
- 识别振荡波形的峰值
- 分析冲击电流的峰值时刻
- 检测周期性信号的周期

**注意:**
- `prominence` 值越大，检测到的峰值越少（只保留显著的峰值）
- 如果不指定 `prominence`，会自动设置为 `(max - min) * 0.1`

---

### detect_frequency_deviation

检测频率偏差。

**函数签名:**
```python
def detect_frequency_deviation(
    cfg_path: str,
    nominal_frequency: float = 50.0,
    tolerance: float = 0.5
) -> Dict[str, Any]
```

**参数:**
- `cfg_path` (str): COMTRADE 配置文件路径
- `nominal_frequency` (float): 标称频率（Hz），默认 50Hz
- `tolerance` (float): 允许偏差（Hz），默认 ±0.5Hz

**返回值:**
```python
{
    "status": "success",
    "nominal_frequency": 50.0,
    "tolerance": 0.5,
    "frequency_channels": [
        {
            "channel": "系统频率",
            "mean_frequency": 49.95,
            "min_frequency": 49.2,
            "max_frequency": 50.3,
            "deviation_detected": True,
            "out_of_range_duration": 2.5  # 超出范围的时长（秒）
        }
    ]
}
```

**使用场景:**
- 检测频率稳定性
- 识别低频减载事件
- 评估电网质量

---

## 高级分析工具

### perform_fft_analysis

对指定通道进行快速傅里叶变换（FFT）频域分析。

**函数签名:**
```python
def perform_fft_analysis(
    cfg_path: str,
    channel_name: str,
    max_frequency: float = None
) -> Dict[str, Any]
```

**参数:**
- `cfg_path` (str): COMTRADE 配置文件路径
- `channel_name` (str): 通道名称
- `max_frequency` (float, optional): 最大分析频率，默认为采样率的一半

**返回值:**
```python
{
    "status": "success",
    "channel": {
        "name": "A相电流",
        "unit": "A"
    },
    "fundamental_freq": 50.0,
    "sample_rate": 300,
    "frequencies": [0, 0.05, 0.1, ..., 150],  # 频率点（Hz）
    "magnitudes": [0.1, 0.2, 1500, ...],      # 幅值
    "dominant_frequencies": [
        {
            "frequency": 50.0,    # 基波
            "magnitude": 8500.0,
            "harmonic_number": 1
        },
        {
            "frequency": 100.0,   # 2次谐波
            "magnitude": 850.0,
            "harmonic_number": 2
        },
        {
            "frequency": 150.0,   # 3次谐波
            "magnitude": 425.0,
            "harmonic_number": 3
        }
        # ... 最多显示前 10 个主要频率分量
    ]
}
```

**使用场景:**
- 谐波分析
- 识别非工频分量
- 分析故障电弧特征（偶次谐波）
- 检测振荡频率

**示例:**
```python
result = perform_fft_analysis("waves/test.cfg", "A相电流")
for component in result['dominant_frequencies']:
    print(f"{component['frequency']}Hz: {component['magnitude']:.1f}A "
          f"(第{component['harmonic_number']}次谐波)")
```

---

### calculate_thd

计算总谐波畸变率（Total Harmonic Distortion）。

**函数签名:**
```python
def calculate_thd(
    cfg_path: str,
    channel_name: str,
    max_harmonic_order: int = 50
) -> Dict[str, Any]
```

**参数:**
- `cfg_path` (str): COMTRADE 配置文件路径
- `channel_name` (str): 通道名称
- `max_harmonic_order` (int): 最高谐波次数，默认 50

**返回值:**
```python
{
    "status": "success",
    "channel": {
        "name": "A相电压",
        "unit": "%"
    },
    "thd_percent": 5.2,           # THD 百分比
    "fundamental": {
        "frequency": 50.0,
        "magnitude": 100.0
    },
    "harmonics": [
        {
            "order": 2,           # 谐波次数
            "frequency": 100.0,
            "magnitude": 2.5,
            "percentage": 2.5    # 占基波的百分比
        },
        {
            "order": 3,
            "frequency": 150.0,
            "magnitude": 3.8,
            "percentage": 3.8
        },
        # ... 其他主要谐波
    ],
    "evaluation": "良好"  # 评价：优秀/良好/一般/较差
}
```

**THD 评价标准:**
- THD < 3%: 优秀
- 3% ≤ THD < 5%: 良好
- 5% ≤ THD < 8%: 一般
- THD ≥ 8%: 较差

**使用场景:**
- 电能质量评估
- 非线性负载影响分析
- 谐波污染检测

---

### analyze_transient_response

分析暂态响应特性（过冲、上升时间、稳定时间等）。

**函数签名:**
```python
def analyze_transient_response(
    cfg_path: str,
    channel_name: str,
    event_time: float
) -> Dict[str, Any]
```

**参数:**
- `cfg_path` (str): COMTRADE 配置文件路径
- `channel_name` (str): 通道名称
- `event_time` (float): 事件发生时刻（秒）

**返回值:**
```python
{
    "status": "success",
    "channel": "励磁电压",
    "event_time": 10.0,
    "steady_state_before": 100.0,     # 事件前稳态值
    "steady_state_after": 120.0,      # 事件后稳态值
    "peak_value": 145.0,              # 峰值
    "overshoot_percent": 20.8,        # 超调量（%）
    "rise_time": 0.05,                # 上升时间（秒）
    "settling_time": 0.8,             # 稳定时间（秒）
    "settling_tolerance": 2.0         # 稳定容差（%）
}
```

**使用场景:**
- 励磁系统响应分析
- PSS（电力系统稳定器）性能评估
- 控制系统调试

---

### calculate_power_quality_metrics

计算电能质量指标（电压偏差、不平衡度、闪变等）。

**函数签名:**
```python
def calculate_power_quality_metrics(cfg_path: str) -> Dict[str, Any]
```

**参数:**
- `cfg_path` (str): COMTRADE 配置文件路径

**返回值:**
```python
{
    "status": "success",
    "voltage_deviation": {
        "A": 0.5,   # A相电压偏差（%）
        "B": -0.3,
        "C": 0.2
    },
    "voltage_unbalance": 1.2,         # 三相不平衡度（%）
    "current_unbalance": 2.5,
    "frequency_deviation": -0.05,     # 频率偏差（Hz）
    "thd_voltage": {
        "A": 2.1,
        "B": 2.3,
        "C": 2.0
    },
    "thd_current": {
        "A": 8.5,
        "B": 7.8,
        "C": 8.2
    },
    "overall_assessment": "良好"      # 综合评价
}
```

**使用场景:**
- 电能质量综合评估
- 合规性检查（GB/T 12325）
- 电网质量监测

---

## 事件分析工具

### extract_fault_timeline

从 HDR 文件中提取故障事件时间线。

**函数签名:**
```python
def extract_fault_timeline(cfg_path: str) -> Dict[str, Any]
```

**参数:**
- `cfg_path` (str): COMTRADE 配置文件路径（会自动查找同名 .hdr 文件）

**返回值:**
```python
{
    "status": "success",
    "fault_start_time": "2024-10-30 11:52:40.123",
    "trip_info": [
        {
            "time": "11:52:40.146",
            "relative_time_ms": 23,      # 相对故障发生的时间（毫秒）
            "trip_signal": "距离保护I段",
            "relay_name": "主保护"
        }
    ],
    "digital_events": [
        {
            "time": "11:52:40.123",
            "relative_time_ms": 0,
            "name": "故障检测",
            "status": "动作"
        },
        {
            "time": "11:52:40.146",
            "relative_time_ms": 23,
            "name": "断路器跳闸",
            "status": "分"
        }
    ],
    "total_duration_ms": 500
}
```

**使用场景:**
- 故障分析报告
- 保护动作时序检查
- 事件重演

---

### analyze_protection_action

分析保护装置动作的正确性和及时性。

**函数签名:**
```python
def analyze_protection_action(
    cfg_path: str,
    expected_action_time_ms: float = 100
) -> Dict[str, Any]
```

**参数:**
- `cfg_path` (str): COMTRADE 配置文件路径
- `expected_action_time_ms` (float): 期望的保护动作时间（毫秒）

**返回值:**
```python
{
    "status": "success",
    "fault_detected": True,
    "fault_time": "11:52:40.123",
    "protection_actions": [
        {
            "name": "距离保护I段",
            "action_time": "11:52:40.146",
            "reaction_time_ms": 23,
            "expected_time_ms": 30,
            "assessment": "正确",           # 正确/延时/拒动/误动
            "severity": "正常"
        }
    ],
    "breaker_trip_time": "11:52:40.170",
    "total_clearing_time_ms": 47,          # 故障切除时间
    "overall_assessment": "保护动作正确、及时"
}
```

**动作评价标准:**
- **正确**: 在期望时间内动作
- **延时**: 超过期望时间但最终动作
- **拒动**: 应动作而未动作
- **误动**: 不应动作而动作

**使用场景:**
- 保护定值校核
- 故障分析报告
- 保护装置检修

---

### correlate_analog_digital_events

关联模拟量变化与数字量动作。

**函数签名:**
```python
def correlate_analog_digital_events(cfg_path: str) -> Dict[str, Any]
```

**参数:**
- `cfg_path` (str): COMTRADE 配置文件路径

**返回值:**
```python
{
    "status": "success",
    "correlations": [
        {
            "time": 10.123,
            "analog_changes": [
                {
                    "channel": "A相电压",
                    "change_type": "突降",
                    "before": 100.0,
                    "after": 15.0,
                    "change_percent": -85.0
                }
            ],
            "digital_actions": [
                {
                    "channel": "故障检测",
                    "status": "动作"
                }
            ],
            "interpretation": "检测到A相接地故障，故障检测元件正确动作"
        }
    ]
}
```

**使用场景:**
- 理解保护动作逻辑
- 验证保护定值合理性
- 培训和教学

---

### identify_fault_type

自动识别故障类型。

**函数签名:**
```python
def identify_fault_type(cfg_path: str) -> Dict[str, Any]
```

**参数:**
- `cfg_path` (str): COMTRADE 配置文件路径

**返回值:**
```python
{
    "status": "success",
    "fault_type": "单相接地",              # 故障类型
    "fault_phases": ["A"],                # 故障相
    "confidence": 0.95,                   # 置信度（0-1）
    "evidence": [
        "A相电压跌落至15%（严重跌落）",
        "B/C相电压上升至110%（健全相过电压）",
        "零序电流出现明显增量（2.5A）",
        "A相电流突增至8.5kA"
    ],
    "fault_characteristics": {
        "voltage_sag_percent": 85.0,
        "fault_current_ka": 8.5,
        "zero_sequence_current_a": 2.5
    },
    "alternative_types": [              # 其他可能的故障类型
        {
            "type": "单相接地经电阻",
            "confidence": 0.15
        }
    ]
}
```

**支持的故障类型:**
- 单相接地（A、B、C）
- 两相短路（AB、BC、CA）
- 两相短路接地
- 三相短路
- 三相短路接地

**识别依据:**
- 三相电压变化特征
- 三相电流变化特征
- 零序/负序分量
- 故障阻抗特性

**使用场景:**
- 快速故障诊断
- 自动化分析报告
- 统计分析

---

## 对比分析工具

### compare_waveforms

对比多个 COMTRADE 文件的波形。

**函数签名:**
```python
def compare_waveforms(
    cfg_paths: List[str],
    channel_name: str
) -> Dict[str, Any]
```

**参数:**
- `cfg_paths` (List[str]): 多个 COMTRADE 文件路径
- `channel_name` (str): 要对比的通道名称

**返回值:**
```python
{
    "status": "success",
    "channel_name": "A相电压",
    "files": [
        {
            "path": "waves/case1.cfg",
            "label": "case1",
            "statistics": {
                "min": 15.0,
                "max": 105.0,
                "mean": 98.5
            }
        },
        {
            "path": "waves/case2.cfg",
            "label": "case2",
            "statistics": {
                "min": 20.0,
                "max": 103.0,
                "mean": 97.8
            }
        }
    ],
    "comparison": {
        "max_difference": 5.0,
        "correlation": 0.95,        # 相关系数
        "similarity": "高度相似"
    }
}
```

**使用场景:**
- 对比不同时间的录波
- 验证仿真模型
- 对比不同设备的响应

---

### baseline_deviation_analysis

将当前波形与基线（正常状态）对比。

**函数签名:**
```python
def baseline_deviation_analysis(
    cfg_path: str,
    baseline_cfg_path: str,
    threshold_percent: float = 10.0
) -> Dict[str, Any]
```

**参数:**
- `cfg_path` (str): 待分析的文件
- `baseline_cfg_path` (str): 基线文件（正常状态）
- `threshold_percent` (float): 偏差阈值（%），默认 10%

**返回值:**
```python
{
    "status": "success",
    "deviations": [
        {
            "channel": "A相电压",
            "baseline_mean": 100.0,
            "current_mean": 85.0,
            "deviation_percent": -15.0,
            "exceeds_threshold": True,
            "assessment": "异常偏离"
        }
    ],
    "overall_deviation_score": 25.0,
    "assessment": "存在明显异常"
}
```

**使用场景:**
- 趋势分析
- 异常检测
- 设备健康度评估

---

## 报告生成工具

### generate_analysis_report

生成完整的波形分析报告（Markdown 格式）。

**函数签名:**
```python
def generate_analysis_report(
    cfg_path: str,
    include_sections: List[str] = None
) -> Dict[str, Any]
```

**参数:**
- `cfg_path` (str): COMTRADE 配置文件路径
- `include_sections` (List[str], optional): 要包含的章节，默认全部

**可用章节:**
- `basic_info`: 基本信息
- `fault_analysis`: 故障分析
- `protection_evaluation`: 保护评价
- `power_quality`: 电能质量
- `frequency_analysis`: 频域分析

**返回值:**
```python
{
    "status": "success",
    "report_markdown": "# 故障录波分析报告\n\n## 基本信息\n...",
    "report_html": "<html>...",  # 可选
    "summary": {
        "fault_type": "单相接地",
        "fault_time": "11:52:40.123",
        "protection_action": "正确",
        "key_findings": [
            "A相电压跌落85%",
            "距离保护I段23ms动作",
            "故障切除时间47ms"
        ]
    }
}
```

**示例:**
```python
result = generate_analysis_report(
    "waves/test.cfg",
    include_sections=["basic_info", "fault_analysis"]
)

# 保存报告
with open("report.md", "w", encoding="utf-8") as f:
    f.write(result['report_markdown'])
```

---

### export_to_json

导出分析结果为 JSON 格式。

**函数签名:**
```python
def export_to_json(
    cfg_path: str,
    output_path: str = None,
    include_raw_data: bool = False
) -> Dict[str, Any]
```

**参数:**
- `cfg_path` (str): COMTRADE 配置文件路径
- `output_path` (str, optional): 输出 JSON 文件路径
- `include_raw_data` (bool): 是否包含原始数据，默认 False

**返回值:**
```python
{
    "status": "success",
    "output_path": "waves/test_analysis.json",
    "data": {
        "metadata": {...},
        "statistics": {...},
        "events": {...},
        "raw_data": {...}  # 仅当 include_raw_data=True
    }
}
```

---

### create_visualization

创建自定义可视化图表。

**函数签名:**
```python
def create_visualization(
    cfg_path: str,
    channel_names: List[str],
    visualization_type: str = "time_series",
    output_path: str = None
) -> Dict[str, Any]
```

**参数:**
- `cfg_path` (str): COMTRADE 配置文件路径
- `channel_names` (List[str]): 要可视化的通道列表
- `visualization_type` (str): 可视化类型
  - `time_series`: 时间序列图（默认）
  - `frequency_spectrum`: 频谱图
  - `phasor`: 相量图
  - `correlation`: 相关性矩阵
- `output_path` (str, optional): 输出 HTML 文件路径

**返回值:**
```python
{
    "status": "success",
    "output_path": "visualization.html",
    "visualization_type": "time_series",
    "channels": ["A相电压", "B相电压", "C相电压"]
}
```

**使用场景:**
- 自定义报告图表
- 演示和展示
- 深度分析

---

## 工具组合使用示例

### 示例 1: 完整的故障诊断流程

```python
# 1. 先了解文件信息
channels = list_available_channels("waves/fault.cfg")

# 2. 提取故障时间线
timeline = extract_fault_timeline("waves/fault.cfg")
fault_time = timeline['fault_start_time']

# 3. 识别故障类型
fault_type = identify_fault_type("waves/fault.cfg")

# 4. 检测电压跌落
voltage_sags = detect_voltage_sags("waves/fault.cfg")

# 5. 分析保护动作
protection = analyze_protection_action("waves/fault.cfg")

# 6. 生成报告
report = generate_analysis_report("waves/fault.cfg")
```

### 示例 2: 电能质量评估

```python
# 1. 电能质量指标
pq_metrics = calculate_power_quality_metrics("waves/test.cfg")

# 2. 谐波分析
thd_voltage = calculate_thd("waves/test.cfg", "A相电压")
thd_current = calculate_thd("waves/test.cfg", "A相电流")

# 3. 频率稳定性
freq_dev = detect_frequency_deviation("waves/test.cfg")

# 4. 生成电能质量报告
report = generate_analysis_report(
    "waves/test.cfg",
    include_sections=["power_quality", "frequency_analysis"]
)
```

### 示例 3: 多文件对比分析

```python
# 1. 对比故障前后的录波
comparison = compare_waveforms(
    ["waves/before.cfg", "waves/after.cfg"],
    "A相电压"
)

# 2. 基线偏差分析
deviation = baseline_deviation_analysis(
    cfg_path="waves/current.cfg",
    baseline_cfg_path="waves/normal.cfg"
)
```

---

## 工具函数速查表

| 工具名称 | 类别 | 用途 | 关键参数 |
|---------|------|------|---------|
| `list_available_channels` | 数据访问 | 列出所有通道 | cfg_path |
| `get_channel_data` | 数据访问 | 获取时序数据 | cfg_path, channel_name |
| `calculate_statistics` | 基础分析 | 统计特征 | cfg_path, channel_name |
| `detect_voltage_sags` | 基础分析 | 电压跌落检测 | cfg_path, threshold_percent |
| `detect_current_surges` | 基础分析 | 电流突变检测 | cfg_path, threshold_multiplier |
| `find_peaks` | 基础分析 | 峰值检测 | cfg_path, channel_name |
| `perform_fft_analysis` | 高级分析 | FFT频域分析 | cfg_path, channel_name |
| `calculate_thd` | 高级分析 | 谐波畸变率 | cfg_path, channel_name |
| `analyze_transient_response` | 高级分析 | 暂态响应分析 | cfg_path, channel_name, event_time |
| `calculate_power_quality_metrics` | 高级分析 | 电能质量指标 | cfg_path |
| `extract_fault_timeline` | 事件分析 | 故障时间线 | cfg_path |
| `analyze_protection_action` | 事件分析 | 保护动作分析 | cfg_path |
| `identify_fault_type` | 事件分析 | 故障类型识别 | cfg_path |
| `compare_waveforms` | 对比分析 | 多文件对比 | cfg_paths, channel_name |
| `baseline_deviation_analysis` | 对比分析 | 基线偏差分析 | cfg_path, baseline_cfg_path |
| `generate_analysis_report` | 报告生成 | 生成分析报告 | cfg_path, include_sections |

---

**文档版本**: 1.0
**最后更新**: 2024-11-10
