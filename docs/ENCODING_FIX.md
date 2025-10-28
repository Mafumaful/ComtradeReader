# 编码问题修复说明

## 问题分析

### 原始问题
COMTRADE配置文件（.cfg）在终端和HTML中显示为乱码，例如：
- `���˵�ѹ�ٷ�ֵ` 而不是 `机端电压百分值`
- `PSS_Pͨ` 而不是 `PSS_P通道输出`

### 根本原因
1. **文件编码**: COMTRADE配置文件使用 **GBK/GB2312** 编码（中国电力系统标准）
2. **读取错误**: 原代码优先使用UTF-8读取，然后使用`errors='ignore'`忽略解码错误
3. **字符丢失**: `errors='ignore'`参数会导致无法解码的字符被静默删除

## 修复方案

### 代码修改

**修改前:**
```python
def _parse_cfg(self):
    try:
        with open(self.cfg_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [line.strip() for line in f.readlines()]
    except:
        with open(self.cfg_file, 'r', encoding='gbk', errors='ignore') as f:
            lines = [line.strip() for line in f.readlines()]
```

**修改后:**
```python
def _parse_cfg(self):
    # COMTRADE配置文件通常使用GB2312/GBK编码（中国电力系统）
    # 按优先级尝试不同编码
    encodings = ['gbk', 'gb2312', 'gb18030', 'utf-8']
    lines = []

    for encoding in encodings:
        try:
            with open(self.cfg_file, 'r', encoding=encoding) as f:
                lines = [line.strip() for line in f.readlines()]
            # 如果成功读取，跳出循环
            break
        except (UnicodeDecodeError, LookupError):
            continue

    if not lines:
        # 如果所有编码都失败，使用gbk并替换无法解码的字符
        with open(self.cfg_file, 'r', encoding='gbk', errors='replace') as f:
            lines = [line.strip() for line in f.readlines()]
```

### 关键改进

1. **优先级调整**: 优先使用GBK/GB2312（中国电力系统常用）
2. **移除errors='ignore'**: 使用严格解码模式，确保编码正确
3. **多编码尝试**: 支持GB2312、GBK、GB18030、UTF-8
4. **降级处理**: 只有在所有编码都失败时才使用`errors='replace'`

## 验证结果

### 修复后的正确显示

**通道名称:**
- ✅ 机端电压百分值
- ✅ 机端电压2百分值
- ✅ 机端电压一次值
- ✅ AVR给定
- ✅ 转子电流给定百分值
- ✅ 转子电流百分值
- ✅ 转子电流一次值
- ✅ 触发角度
- ✅ 有功实际值
- ✅ 无功实际值
- ✅ 定子电流一次值
- ✅ 定子电流百分值
- ✅ 同步频率
- ✅ 机端频率
- ✅ 励磁电压一次值

**HTML可视化:**
- ✅ 图表标题正确显示中文
- ✅ 子图标题显示正确的通道名
- ✅ 悬停提示显示正确的中文标签

## 技术说明

### 常见中文编码

| 编码 | 全称 | 特点 | 适用场景 |
|------|------|------|----------|
| GB2312 | 简体中文字符集 | 6763个汉字 | 早期简体中文系统 |
| GBK | 汉字内码扩展规范 | 21003个汉字 | Windows简体中文 |
| GB18030 | 信息技术中文编码字符集 | 最全 | 中国国家标准 |
| UTF-8 | Unicode | 全球通用 | 现代系统推荐 |

### 电力系统COMTRADE文件
- **标准**: IEC 60255-24 / IEEE C37.111
- **中国实现**: 通常使用GBK编码保存中文设备名和参数名
- **国际实现**: 通常使用ASCII或UTF-8

## 最佳实践

### 处理COMTRADE中文文件
1. 优先尝试GBK/GB2312编码
2. 不要使用`errors='ignore'`，会丢失字符
3. 确保HTML输出使用UTF-8编码
4. Plotly会自动处理Unicode字符

### 测试编码
```bash
# 使用file命令检测文件
file yourfile.cfg

# 使用Python测试编码
python3 << 'EOF'
encodings = ['gbk', 'gb2312', 'utf-8']
for enc in encodings:
    try:
        with open('yourfile.cfg', 'r', encoding=enc) as f:
            print(f"{enc}: OK - {f.readline()[:50]}")
    except:
        print(f"{enc}: Failed")
EOF
```

## 相关文件
- `waveform_viewer.py` - 主程序（已修复）
- `verify_data.py` - 验证脚本（自动继承修复）
- 所有生成的HTML文件都能正确显示中文
