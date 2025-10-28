# 自动打开功能说明

## 功能概述

v2.0.1 版本新增了**自动打开**功能。生成HTML可视化文件后，程序会自动在默认浏览器中打开文件，无需手动查找和打开。

## 使用方法

### 新版程序 (main.py)

运行程序时，会在配置阶段询问是否自动打开：

```bash
python main.py
```

在配置过程中，会看到以下选项：

```
生成后自动打开文件？
→ 是（推荐）- 自动在浏览器中打开
  否 - 仅保存文件
```

- 选择"是"：生成后自动在浏览器中打开
- 选择"否"：仅保存文件，不自动打开

### 旧版程序 (waveform_viewer.py)

旧版程序默认启用自动打开，无需额外配置：

```bash
python waveform_viewer.py
```

生成文件后会自动打开。

### 编程式使用

在代码中可以通过 `VisualizationConfig` 控制：

```python
from waveform_viewer.visualizers.base import VisualizationConfig
from waveform_viewer.visualizers.plotly_viz import PlotlyVisualizer

# 启用自动打开（默认）
config = VisualizationConfig(auto_open=True)
visualizer = PlotlyVisualizer(config)

# 或者禁用自动打开
config = VisualizationConfig(auto_open=False)
visualizer = PlotlyVisualizer(config)
```

## 技术实现

### 实现原理

**v2.0.2 改进**: 使用系统原生命令，更加可靠：

```python
import subprocess
import sys
from pathlib import Path

def open_file_in_browser(file_path):
    """在默认浏览器中打开文件"""
    abs_path = Path(file_path).resolve()
    system = sys.platform

    if system == 'darwin':  # macOS
        subprocess.run(['open', str(abs_path)], check=True)
    elif system == 'win32':  # Windows
        os.startfile(str(abs_path))
    elif system.startswith('linux'):  # Linux
        subprocess.run(['xdg-open', str(abs_path)], check=True)
```

### 跨平台支持

使用系统原生命令：

| 平台 | 命令/方法 | 说明 |
|------|----------|------|
| macOS | `open` | macOS系统命令，最可靠 |
| Windows | `os.startfile()` | Windows标准API |
| Linux | `xdg-open` | XDG标准工具 |

**优势**:
- ✅ 比 `webbrowser` 更可靠
- ✅ 使用系统默认应用
- ✅ 不受浏览器配置影响

### 错误处理

如果自动打开失败（如没有默认浏览器），程序会：
1. 打印警告信息
2. 显示文件路径
3. 继续正常运行（不影响主流程）

示例输出：

```
✓ 可视化结果已保存到: waveform_visualization.html
提示: 无法自动打开文件 - [错误信息]
请手动打开: /path/to/waveform_visualization.html
```

## 配置选项

### VisualizationConfig 参数

```python
class VisualizationConfig:
    def __init__(self,
                 output_dir: Optional[str] = None,
                 width: int = 1200,
                 height_per_plot: int = 250,
                 title: Optional[str] = None,
                 show_legend: bool = False,
                 auto_open: bool = True):  # 新增参数
        ...
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `auto_open` | bool | True | 是否自动在浏览器中打开生成的文件 |

## 使用场景

### 适合自动打开的场景

- ✅ 交互式使用（手动运行程序）
- ✅ 查看单个文件的可视化
- ✅ 快速检查结果
- ✅ 演示和展示

### 不适合自动打开的场景

- ❌ 批量处理大量文件（会打开很多浏览器标签）
- ❌ 自动化脚本（无人值守运行）
- ❌ 服务器环境（没有图形界面）
- ❌ 仅需要文件，不需要查看

对于批量处理，建议选择"否 - 仅保存文件"选项。

## 测试

运行测试脚本验证功能：

```bash
python test_auto_open.py
```

测试会：
1. 生成测试可视化（启用自动打开）
2. 生成测试可视化（禁用自动打开）
3. 清理测试文件
4. 报告结果

## 常见问题

**Q: 文件没有自动打开？**

A: 检查以下几点：
1. 是否选择了"是 - 自动打开"选项
2. 系统是否有默认浏览器
3. 查看终端输出的错误信息

**Q: 打开了错误的浏览器？**

A: `webbrowser` 使用系统默认浏览器。修改系统默认浏览器设置即可。

**Q: 批量处理时打开太多标签？**

A: 在配置阶段选择"否 - 仅保存文件"，或在代码中设置 `auto_open=False`。

**Q: 服务器环境报错？**

A: 程序会捕获错误并继续运行，不影响文件生成。

**Q: 如何指定使用特定浏览器？**

A: 可以在代码中指定：

```python
import webbrowser

# 指定Chrome
chrome = webbrowser.get('chrome')
chrome.open(f'file://{abs_path}')

# 指定Firefox
firefox = webbrowser.get('firefox')
firefox.open(f'file://{abs_path}')
```

## 修改历史

- v2.0.1 (2025-01): 新增自动打开功能
  - 添加 `VisualizationConfig.auto_open` 参数
  - 新版程序中添加用户选择选项
  - 旧版程序默认启用
  - 添加错误处理和跨平台支持

## 相关文件

- `waveform_viewer/visualizers/base.py` - 配置类定义
- `waveform_viewer/visualizers/plotly_viz.py` - 自动打开实现
- `waveform_viewer/app.py` - 用户选择界面
- `waveform_viewer.py` - 旧版支持
- `test_auto_open.py` - 测试脚本

---

**提示**: 此功能旨在提高用户体验，使可视化结果能够立即查看，无需手动查找文件。
