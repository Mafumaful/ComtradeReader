# 快速开始指南

## 立即开始使用

### 1. 安装依赖

```bash
pip install plotly numpy
```

### 2. 准备数据

确保波形文件放在 `waves/` 目录下：

```
waves/
└── 机励磁波形2025.1013/
    ├── *.cfg
    └── *.dat
```

### 3. 运行程序

```bash
# 推荐：使用新版交互式菜单
python main.py

# 或者：使用旧版（兼容）
python waveform_viewer.py
```

## 新功能

### ✨ 交互式菜单
- 使用方向键（↑↓）导航
- 空格键多选
- 回车确认
- 支持批量处理

### 🔌 插件系统
- CSV导出
- JSON导出
- 统计分析
- 数据滤波

### 🎨 灵活配置
- 智能通道选择
- 多种可视化样式
- 可自定义扩展

## 常用命令

```bash
# 运行主程序
python main.py

# 使用简单菜单（不支持方向键的环境）
python main.py --simple

# 验证数据
python verify_data.py

# 运行测试
python test_basic.py
```

## 下一步

- 阅读 `README.md` 了解详细功能
- 阅读 `DEVELOPER_GUIDE.md` 学习如何开发插件
- 查看 `REFACTORING_SUMMARY.md` 了解架构设计

## 需要帮助？

1. 查看 `README.md` 的常见问题部分
2. 查看 `DEVELOPER_GUIDE.md` 的故障排除部分
3. 联系开发团队

---

**开始创建你的第一个可视化吧！** 🚀
