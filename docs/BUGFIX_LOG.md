# Bug修复日志

## Bug #1: Plotly vertical_spacing 错误 (2025-01)

### 问题描述

当选择显示所有通道（例如47个通道）时，Plotly的`make_subplots`函数抛出错误：

```
ValueError: Vertical spacing cannot be greater than (1 / (rows - 1)) = 0.021739.
The resulting plot would have 47 rows (rows=47).
```

### 根本原因

在 `waveform_viewer/visualizers/plotly_viz.py` 和 `waveform_viewer.py` 中，`vertical_spacing` 被硬编码为 `0.03`。

当子图行数很多时，Plotly对 `vertical_spacing` 有限制：
- 最大允许值 = `1 / (rows - 1)`
- 对于47行：最大值 = `1 / 46 ≈ 0.021739`
- 而代码中使用的 `0.03 > 0.021739`，因此报错

### 解决方案

动态计算 `vertical_spacing`，确保不超过Plotly的限制：

```python
# 动态计算vertical_spacing，避免超出Plotly的限制
# 最大允许值为 1 / (rows - 1)，我们使用 0.8 倍以留出余地
if num_plots > 1:
    max_spacing = 0.8 / (num_plots - 1)
    vertical_spacing = min(0.03, max_spacing)
else:
    vertical_spacing = 0.03
```

公式说明：
- 使用 `0.8 / (num_plots - 1)` 而不是 `1 / (num_plots - 1)`，留出20%的安全余地
- 使用 `min(0.03, max_spacing)` 确保在通道数少时仍使用理想的0.03间距

### 修改的文件

1. `waveform_viewer/visualizers/plotly_viz.py`
   - 修改 `PlotlyVisualizer.visualize()` 方法
   - 修改 `PlotlyLineVisualizer.visualize()` 方法

2. `waveform_viewer.py` (旧版文件，保持兼容性)
   - 修改 `visualize_waveforms()` 函数

### 测试

测试用例：
- ✅ 1个通道：spacing = 0.03
- ✅ 10个通道：spacing = 0.03 (max = 0.089)
- ✅ 20个通道：spacing = 0.03 (max = 0.042)
- ✅ 47个通道：spacing ≈ 0.0174 (max = 0.0174, 动态调整)
- ✅ 100个通道：spacing ≈ 0.0081 (max = 0.0081, 动态调整)

### 影响范围

- 影响所有使用Plotly可视化的功能
- 向后兼容，不影响现有功能
- 自动适应任意数量的通道

### 预防措施

在未来开发新的可视化器时：
1. 不要硬编码布局参数
2. 根据数据量动态计算参数
3. 考虑极端情况（如大量通道、大量数据点）
4. 参考 `visualizers/plotly_viz.py` 的实现

### 相关问题

可能的改进：
- 当通道数超过某个阈值（如30个）时，考虑分页显示
- 添加"选择性显示"功能，让用户选择最关心的通道
- 实现通道分组功能

### 参考资料

- Plotly文档：https://plotly.com/python/subplots/
- Plotly源码：`plotly/_subplots.py` 中的 `_check_hv_spacing()` 函数

---

## Bug #2: [待记录]

...

---

**维护说明**：
- 每次修复bug后，请在此文件中记录
- 包含问题描述、原因分析、解决方案、测试结果
- 这将帮助未来的开发者避免类似问题
