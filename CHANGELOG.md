# 更新日志

所有重要的项目变更都将记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)。

---

## [v2.0.2] - 2025-10-28

### 修复 🐛
- **修复自动打开功能**: 改用系统原生命令，更加可靠
  - macOS: 使用 `open` 命令
  - Windows: 使用 `os.startfile()`
  - Linux: 使用 `xdg-open` 命令
  - 不再依赖 `webbrowser` 模块
  - 相关文档: `BUGFIX_AUTO_OPEN.md`

### 新增 ✨
- 添加专用的文件打开工具 `waveform_viewer/utils/file_opener.py`
- 改进错误处理和用户反馈

---

## [v2.0.1] - 2025-10-28

### 新增 ✨
- **自动打开功能**: 生成HTML可视化后自动在浏览器中打开
  - 新版程序中添加用户选择选项
  - 旧版程序默认启用
  - 跨平台支持
  - 可通过 `VisualizationConfig.auto_open` 配置

### 修复 🐛
- **修复 vertical_spacing 错误**: 当通道数量较多时（≥28个）无法生成可视化的问题
  - 动态计算 `vertical_spacing`，适应任意数量的通道
  - 对于 ≤26 个通道，保持原有显示效果
  - 对于 >26 个通道，自动调整间距
  - 相关文档: `BUGFIX_SUMMARY.md`, `BUGFIX_LOG.md`

### 改进 🔧
- 完善错误处理，自动打开失败不影响主流程
- 添加更详细的用户提示信息

### 文档 📚
- 添加 `FEATURE_AUTO_OPEN.md` - 自动打开功能详细说明
- 添加 `BUGFIX_SUMMARY.md` - Bug修复总结报告
- 添加 `BUGFIX_LOG.md` - 详细的Bug修复记录
- 添加 `test_auto_open.py` - 自动打开功能测试
- 添加 `test_vertical_spacing_fix.py` - vertical_spacing修复验证
- 更新 `README.md` - 添加自动打开功能说明

---

## [v2.0.0] - 2025-10-28

### 新增 ✨
- **完全重构代码架构**，采用现代设计模式
  - Strategy Pattern: 通道选择策略、可视化策略
  - Plugin Architecture: 可扩展的插件系统
  - Singleton Pattern: 插件管理器
  - Factory Pattern: 对象创建

- **模块化代码结构**
  - `waveform_viewer/core/` - 核心功能模块
  - `waveform_viewer/visualizers/` - 可视化模块
  - `waveform_viewer/plugins/` - 插件系统
  - `waveform_viewer/ui/` - 用户界面模块
  - `waveform_viewer/utils/` - 工具模块

- **交互式菜单系统**
  - 支持方向键导航（↑↓）
  - 支持空格键多选
  - 支持简单数字输入模式（兼容性）
  - 降级处理，适应不同终端环境

- **插件系统**
  - StatisticsPlugin - 统计信息计算
  - CSVExportPlugin - CSV格式导出
  - JSONExportPlugin - JSON格式导出
  - DataFilterPlugin - 数据滤波
  - 支持第三方插件开发

- **批量处理功能**
  - 可选择多个文件夹
  - 可选择多个波形文件
  - 统一处理流程

- **灵活的通道选择**
  - ImportantPatternSelector - 智能选择重要通道
  - FirstNChannelsSelector - 选择前N个通道
  - AllChannelsSelector - 选择所有通道
  - PatternMatchSelector - 基于模式匹配选择

- **多种可视化方式**
  - PlotlyVisualizer - 散点图可视化
  - PlotlyLineVisualizer - 线图可视化
  - 易于扩展更多可视化器

- **新的主程序**
  - `main.py` - 新版主入口
  - 完整的应用程序类 `WaveformViewerApp`
  - 保留旧版 `waveform_viewer.py` 的兼容性

### 改进 🔧
- 更好的中文编码支持（GBK/GB2312/GB18030/UTF-8）
- 更清晰的用户界面和提示信息
- 更健壮的错误处理
- 性能优化

### 文档 📚
- **README.md** - 完整的用户使用指南
- **DEVELOPER_GUIDE.md** - 详细的开发者指南
  - 架构说明
  - 如何添加新功能
  - 如何开发插件
  - 最佳实践
  - 测试指南
- **REFACTORING_SUMMARY.md** - 重构总结和架构说明
- **QUICK_START.md** - 快速开始指南
- **requirements.txt** - Python依赖列表
- **test_basic.py** - 基本功能测试脚本

### 向后兼容 ✅
- 保留旧版 `waveform_viewer.py`，完全向后兼容
- `verify_data.py` 支持新旧两种导入方式
- 旧的使用方式仍然可用

---

## [v1.0.0] - 2025-10-28 (初始版本)

### 功能
- 基本的COMTRADE文件读取和解析
- Plotly交互式可视化
- 自动扫描波形文件夹
- 智能通道选择
- 散点图显示
- 中文编码支持
- 数据验证脚本

---

## 版本号说明

版本号格式：`主版本.次版本.修订号`

- **主版本号**: 不兼容的API修改
- **次版本号**: 向下兼容的功能性新增
- **修订号**: 向下兼容的问题修复

---

## 计划中的功能 🚀

### v2.1.0 (计划中)
- [ ] Web界面（Flask/Django）
- [ ] Excel导出插件
- [ ] FFT频谱分析插件
- [ ] 数据对比功能
- [ ] 更多可视化样式（3D、热图等）

### v2.2.0 (计划中)
- [ ] 数据库支持
- [ ] 报告生成功能
- [ ] 批量对比分析
- [ ] 配置文件支持

### 长期计划
- [ ] 实时数据流处理
- [ ] 机器学习分析
- [ ] 云端部署
- [ ] RESTful API

---

## 如何贡献

1. 查看 `DEVELOPER_GUIDE.md` 了解开发规范
2. 创建功能分支
3. 提交清晰的commit消息
4. 更新相关文档
5. 添加测试用例

---

**维护者**: 开发团队
**许可证**: 内部使用
**项目地址**: /Users/miakho/Documents/NARI/波形记录
