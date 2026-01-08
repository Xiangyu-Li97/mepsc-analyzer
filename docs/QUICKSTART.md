# mEPSC 分析工具 - 快速入门指南

## 5 分钟上手

### Python 版本

#### 第 1 步：安装依赖

```bash
pip install pyabf numpy pandas matplotlib scipy
```

#### 第 2 步：运行分析

```bash
python3.11 mepsc_analyzer.py your_data.abf
```

就这么简单！工具会自动：
- 检测所有事件
- 提取特征参数
- 生成分析报告图
- 保存结果为 CSV

#### 第 3 步：查看结果

在与输入文件相同的目录下，您会找到：
- `*_events_*.csv` - 每个事件的详细参数
- `*_summary_*.csv` - 汇总统计信息
- `*_analysis_*.png` - 可视化分析报告

---

### MATLAB 版本

#### 第 1 步：安装 abfload

从 [MATLAB File Exchange](https://www.mathworks.com/matlabcentral/fileexchange/6190-abfload) 下载并安装 `abfload` 函数。

#### 第 2 步：运行分析

```matlab
results = mepsc_analyzer('your_data.abf');
```

#### 第 3 步：查看结果

```matlab
% 查看汇总统计
results.summary

% 查看事件特征
results.event_features

% 绘制特定事件
figure;
plot(results.time_trace, results.data_trace);
```

---

## 常用参数调整

### 如果漏检了小事件

```bash
# Python
python3.11 mepsc_analyzer.py data.abf --threshold 2.5

# MATLAB
results = mepsc_analyzer('data.abf', 'threshold', 2.5);
```

### 如果误检了太多噪声

```bash
# Python
python3.11 mepsc_analyzer.py data.abf --threshold 4.5 --min-amp 15

# MATLAB
results = mepsc_analyzer('data.abf', 'threshold', 4.5, 'min_amp', 15);
```

### 批量处理多个文件

**Python**:
```python
import glob
from mepsc_analyzer import mEPSCAnalyzer

for file in glob.glob('*.abf'):
    analyzer = mEPSCAnalyzer(file)
    analyzer.estimate_baseline()
    analyzer.estimate_noise()
    analyzer.detect_events()
    analyzer.analyze_all_events()
    analyzer.save_results()
```

**MATLAB**:
```matlab
files = dir('*.abf');
for i = 1:length(files)
    mepsc_analyzer(files(i).name);
end
```

---

## 示例输出解读

### 汇总统计示例

```
文件名: 1-example.abf
事件总数: 172
事件频率: 3.44 Hz
平均振幅: 29.9 ± 13.1 pA
平均上升时间: 0.82 ms
平均衰减 tau: 1.8 ms
平均面积: 282.3 pA*ms
```

**解读**：
- **事件频率 3.44 Hz**：平均每秒发生 3-4 个突触事件，这是典型的自发 mEPSC 频率
- **平均振幅 29.9 pA**：单个突触囊泡释放产生的平均电流
- **平均衰减 tau 1.8 ms**：AMPA 受体通道的典型关闭时间常数

### 分析报告图解读

1. **完整轨迹图**（第一行）
   - 黑色曲线：原始记录
   - 红点：检测到的事件
   - 蓝色虚线：基线
   - 检查红点是否准确标记在每个明显事件上

2. **振幅分布图**（第二行左）
   - 应该呈现右偏分布
   - 如果有双峰，可能表示两种不同类型的突触输入

3. **上升时间分布图**（第二行中）
   - 通常集中在 0.5-2 ms
   - 反映了突触后膜的电容特性

4. **衰减时间分布图**（第二行右）
   - AMPA 型 mEPSC：通常 1-5 ms
   - NMDA 型 mEPSC：通常 50-200 ms
   - GABA 型 mIPSC：通常 10-50 ms

5. **示例事件波形**（第四行）
   - 检查事件形状是否符合预期
   - 应该是快速下降 + 指数衰减

---

## 故障排除

### 问题：检测到的事件太少

**可能原因**：
1. 阈值设置过高
2. 数据信噪比太低
3. 基线不稳定

**解决方案**：
```bash
# 降低阈值
python3.11 mepsc_analyzer.py data.abf --threshold 2.5
```

### 问题：检测到很多噪声

**可能原因**：
1. 阈值设置过低
2. 数据质量差

**解决方案**：
```bash
# 提高阈值并设置最小振幅
python3.11 mepsc_analyzer.py data.abf --threshold 4.0 --min-amp 15
```

### 问题：衰减 tau 显示为 NaN

**可能原因**：
1. 事件太小，无法拟合
2. 事件形状不规则
3. 事件发生在记录末尾

**解决方案**：
- 这是正常现象，不影响其他参数的计算
- 可以在后续分析中过滤掉这些事件

---

## 下一步

- 阅读完整的 [README_mEPSC_Analyzer.md](README_mEPSC_Analyzer.md) 了解详细功能
- 查看 `example_output/` 目录中的示例输出
- 尝试调整参数以优化您的数据分析

**祝您分析顺利！**
