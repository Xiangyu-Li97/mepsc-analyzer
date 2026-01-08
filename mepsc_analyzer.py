#!/usr/bin/env python3.11
"""
mEPSC/mIPSC 自动分析工具

功能:
- 自动检测 mEPSC/mIPSC 事件
- 提取事件特征参数（振幅、上升时间、衰减时间、面积等）
- 生成详细的分析报告和可视化图表
- 导出结果为 CSV 和 Excel 格式

作者: Manus AI
日期: 2026-01-08
"""

import pyabf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, optimize
import argparse
import os
from datetime import datetime

class mEPSCAnalyzer:
    """mEPSC/mIPSC 事件检测和分析类"""
    
    def __init__(self, abf_path, channel=0):
        """
        初始化分析器
        
        参数:
            abf_path: ABF 文件路径
            channel: 要分析的通道编号（默认为 0）
        """
        self.abf_path = abf_path
        self.abf = pyabf.ABF(abf_path)
        self.abf.setSweep(sweepNumber=0, channel=channel)
        
        self.data = self.abf.sweepY
        self.time = self.abf.sweepX
        self.sampling_rate = self.abf.dataRate
        self.channel = channel
        
        self.baseline = None
        self.noise_std = None
        self.event_indices = None
        self.event_features = None
        
        print(f"已加载文件: {os.path.basename(abf_path)}")
        print(f"采样率: {self.sampling_rate} Hz")
        print(f"记录时长: {self.abf.dataLengthSec:.2f} 秒")
        print(f"通道: {self.abf.sweepLabelY} ({self.abf.sweepUnitsY})")
    
    def estimate_baseline(self, percentile=50, method='percentile'):
        """
        估计基线电流
        
        参数:
            percentile: 用于估计基线的百分位数
            method: 'percentile' 或 'mode'
        """
        if method == 'percentile':
            self.baseline = np.percentile(self.data, percentile)
        elif method == 'mode':
            # 使用直方图模式
            hist, bins = np.histogram(self.data, bins=200)
            self.baseline = bins[np.argmax(hist)]
        
        print(f"基线估计: {self.baseline:.2f} {self.abf.sweepUnitsY}")
        return self.baseline
    
    def estimate_noise(self, threshold_factor=2):
        """
        估计噪声水平（标准差）
        
        参数:
            threshold_factor: 用于选择接近基线数据的阈值因子
        """
        if self.baseline is None:
            self.estimate_baseline()
        
        # 选择接近基线的数据点来估计噪声
        near_baseline = self.data[np.abs(self.data - self.baseline) < 
                                   threshold_factor * np.std(self.data)]
        self.noise_std = np.std(near_baseline)
        
        print(f"噪声标准差: {self.noise_std:.2f} {self.abf.sweepUnitsY}")
        return self.noise_std
    
    def detect_events(self, threshold_factor=3.5, min_interval_ms=5, 
                     min_amplitude=None, max_amplitude=None):
        """
        检测 mEPSC/mIPSC 事件
        
        参数:
            threshold_factor: 检测阈值（倍数的噪声标准差）
            min_interval_ms: 最小事件间隔（毫秒）
            min_amplitude: 最小振幅过滤（pA）
            max_amplitude: 最大振幅过滤（pA）
        
        返回:
            event_indices: 事件峰值索引数组
        """
        if self.baseline is None:
            self.estimate_baseline()
        if self.noise_std is None:
            self.estimate_noise()
        
        # 计算检测阈值
        threshold = self.baseline - threshold_factor * self.noise_std
        
        # 找到所有低于阈值的点
        below_threshold = self.data < threshold
        
        # 找到阈值穿越点（从上到下）
        crossings = np.diff(below_threshold.astype(int))
        event_starts = np.where(crossings == 1)[0]
        
        # 在每个事件中找到峰值（最小值）
        min_interval_samples = int(min_interval_ms * self.sampling_rate / 1000)
        event_peaks = []
        
        for start in event_starts:
            # 定义搜索窗口
            search_window = int(2 * self.sampling_rate / 1000)  # 2 ms
            end = min(start + search_window, len(self.data))
            
            # 在窗口内找到最小值
            window_data = self.data[start:end]
            if len(window_data) > 0:
                peak_idx = start + np.argmin(window_data)
                peak_amplitude = abs(self.data[peak_idx] - self.baseline)
                
                # 振幅过滤
                if min_amplitude is not None and peak_amplitude < min_amplitude:
                    continue
                if max_amplitude is not None and peak_amplitude > max_amplitude:
                    continue
                
                # 检查是否与之前的事件太近
                if len(event_peaks) == 0 or (peak_idx - event_peaks[-1]) > min_interval_samples:
                    event_peaks.append(peak_idx)
        
        self.event_indices = np.array(event_peaks)
        
        print(f"\n检测到 {len(self.event_indices)} 个事件")
        print(f"事件频率: {len(self.event_indices) / self.abf.dataLengthSec:.2f} Hz")
        
        return self.event_indices
    
    def extract_features(self, event_idx):
        """
        提取单个事件的特征参数
        
        返回:
            features: 包含所有特征参数的字典
        """
        features = {}
        
        # 定义事件窗口
        pre_samples = int(2 * self.sampling_rate / 1000)  # 2 ms before peak
        post_samples = int(20 * self.sampling_rate / 1000)  # 20 ms after peak
        
        start_idx = max(0, event_idx - pre_samples)
        end_idx = min(len(self.data), event_idx + post_samples)
        
        event_data = self.data[start_idx:end_idx]
        event_time = self.time[start_idx:end_idx]
        
        # 基本特征
        peak_value = self.data[event_idx]
        features['peak_time'] = self.time[event_idx]
        features['peak_index'] = event_idx
        features['amplitude'] = abs(peak_value - self.baseline)
        
        # 找到事件起始点
        pre_event = self.data[start_idx:event_idx]
        if len(pre_event) > 5:
            # 从峰值向前找，找到第一个接近基线的点（10% 振幅处）
            onset_threshold = self.baseline - 0.1 * features['amplitude']
            onset_candidates = np.where(pre_event > onset_threshold)[0]
            if len(onset_candidates) > 0:
                onset_idx = start_idx + onset_candidates[-1]
            else:
                onset_idx = start_idx
        else:
            onset_idx = start_idx
        
        features['onset_time'] = self.time[onset_idx]
        
        # 上升时间 (10%-90%)
        if onset_idx < event_idx:
            amp_10 = self.baseline - 0.1 * features['amplitude']
            amp_90 = self.baseline - 0.9 * features['amplitude']
            
            rise_data = self.data[onset_idx:event_idx+1]
            idx_10 = np.where(rise_data < amp_10)[0]
            idx_90 = np.where(rise_data < amp_90)[0]
            
            if len(idx_10) > 0 and len(idx_90) > 0:
                t_10 = self.time[onset_idx + idx_10[0]]
                t_90 = self.time[onset_idx + idx_90[0]]
                features['rise_time_10_90'] = (t_90 - t_10) * 1000  # ms
            else:
                features['rise_time_10_90'] = np.nan
        else:
            features['rise_time_10_90'] = np.nan
        
        # 衰减时间 - 指数拟合
        decay_data = self.data[event_idx:end_idx]
        decay_time_axis = (self.time[event_idx:end_idx] - self.time[event_idx]) * 1000  # ms
        
        if len(decay_data) > 20:
            try:
                # 单指数拟合: y = A * exp(-t/tau) + C
                def exp_decay(t, A, tau, C):
                    return A * np.exp(-t / tau) + C
                
                # 初始猜测
                p0 = [-(peak_value - self.baseline), 5, self.baseline]
                
                # 只拟合前 10 ms 的数据
                fit_samples = min(len(decay_data), int(10 * self.sampling_rate / 1000))
                
                popt, pcov = optimize.curve_fit(
                    exp_decay, 
                    decay_time_axis[:fit_samples], 
                    decay_data[:fit_samples], 
                    p0=p0, 
                    maxfev=2000,
                    bounds=([-np.inf, 0.1, -np.inf], [0, 50, np.inf])
                )
                
                features['decay_tau'] = abs(popt[1])
                features['decay_fit_quality'] = np.sqrt(np.diag(pcov))[1] / abs(popt[1])  # 相对误差
            except Exception as e:
                features['decay_tau'] = np.nan
                features['decay_fit_quality'] = np.nan
            
            # 衰减时间 90%-10% (从峰值到基线的恢复)
            amp_90_decay = self.baseline - 0.9 * features['amplitude']
            amp_10_decay = self.baseline - 0.1 * features['amplitude']
            
            idx_90_decay = np.where(decay_data < amp_90_decay)[0]
            idx_10_decay = np.where(decay_data > amp_10_decay)[0]
            
            if len(idx_90_decay) > 0 and len(idx_10_decay) > 0:
                t_90 = decay_time_axis[0]  # 峰值时间
                t_10_candidates = decay_time_axis[idx_10_decay]
                t_10_valid = t_10_candidates[t_10_candidates > t_90]
                if len(t_10_valid) > 0:
                    features['decay_time_90_10'] = t_10_valid[0]
                else:
                    features['decay_time_90_10'] = np.nan
            else:
                features['decay_time_90_10'] = np.nan
        else:
            features['decay_tau'] = np.nan
            features['decay_fit_quality'] = np.nan
            features['decay_time_90_10'] = np.nan
        
        # 事件面积（使用梯形积分）
        if onset_idx < end_idx:
            area_data = self.data[onset_idx:end_idx] - self.baseline
            area_time = self.time[onset_idx:end_idx]
            # 只计算低于基线的部分
            area_data = np.minimum(area_data, 0)
            features['area'] = abs(np.trapezoid(area_data, area_time) * 1000)  # pA*ms
        else:
            features['area'] = np.nan
        
        # 半高宽
        half_amp = self.baseline - 0.5 * features['amplitude']
        half_width_points = np.where(event_data < half_amp)[0]
        if len(half_width_points) > 1:
            features['half_width'] = (event_time[half_width_points[-1]] - 
                                     event_time[half_width_points[0]]) * 1000
        else:
            features['half_width'] = np.nan
        
        return features
    
    def analyze_all_events(self):
        """分析所有检测到的事件"""
        if self.event_indices is None or len(self.event_indices) == 0:
            print("错误: 请先运行 detect_events()")
            return None
        
        print(f"\n正在提取 {len(self.event_indices)} 个事件的特征...")
        
        features_list = []
        for i, idx in enumerate(self.event_indices):
            if (i + 1) % 50 == 0:
                print(f"  进度: {i+1}/{len(self.event_indices)}")
            
            features = self.extract_features(idx)
            features['event_number'] = i + 1
            features_list.append(features)
        
        # 转换为 DataFrame
        self.event_features = pd.DataFrame(features_list)
        
        # 重新排列列顺序
        column_order = ['event_number', 'peak_time', 'onset_time', 'amplitude', 
                       'rise_time_10_90', 'decay_tau', 'decay_time_90_10', 
                       'half_width', 'area', 'decay_fit_quality', 'peak_index']
        self.event_features = self.event_features[column_order]
        
        print("特征提取完成!")
        return self.event_features
    
    def get_summary_statistics(self):
        """计算汇总统计信息"""
        if self.event_features is None:
            print("错误: 请先运行 analyze_all_events()")
            return None
        
        summary = {
            '文件名': os.path.basename(self.abf_path),
            '记录时长 (s)': self.abf.dataLengthSec,
            '事件总数': len(self.event_features),
            '事件频率 (Hz)': len(self.event_features) / self.abf.dataLengthSec,
            '基线 (pA)': self.baseline,
            '噪声标准差 (pA)': self.noise_std,
            '平均振幅 (pA)': self.event_features['amplitude'].mean(),
            '振幅标准差 (pA)': self.event_features['amplitude'].std(),
            '振幅中位数 (pA)': self.event_features['amplitude'].median(),
            '平均上升时间 (ms)': self.event_features['rise_time_10_90'].mean(),
            '平均衰减 tau (ms)': self.event_features['decay_tau'].mean(),
            '平均面积 (pA*ms)': self.event_features['area'].mean(),
            '平均半高宽 (ms)': self.event_features['half_width'].mean(),
        }
        
        return pd.Series(summary)
    
    def save_results(self, output_dir=None):
        """
        保存分析结果
        
        参数:
            output_dir: 输出目录（默认为当前目录）
        """
        if self.event_features is None:
            print("错误: 请先运行 analyze_all_events()")
            return
        
        if output_dir is None:
            output_dir = os.path.dirname(self.abf_path)
        
        # 创建输出文件名
        base_name = os.path.splitext(os.path.basename(self.abf_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存详细结果为 CSV
        csv_path = os.path.join(output_dir, f"{base_name}_events_{timestamp}.csv")
        self.event_features.to_csv(csv_path, index=False)
        print(f"已保存事件详情: {csv_path}")
        
        # 保存汇总统计
        summary = self.get_summary_statistics()
        summary_path = os.path.join(output_dir, f"{base_name}_summary_{timestamp}.csv")
        summary.to_csv(summary_path, header=['Value'])
        print(f"已保存汇总统计: {summary_path}")
        
        return csv_path, summary_path
    
    def plot_overview(self, save_path=None):
        """绘制分析概览图"""
        if self.event_features is None:
            print("错误: 请先运行 analyze_all_events()")
            return
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # 1. 完整轨迹 + 检测到的事件
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(self.time, self.data, 'k-', linewidth=0.3, alpha=0.7)
        ax1.plot(self.time[self.event_indices], self.data[self.event_indices], 
                'r.', markersize=3, label=f'检测到的事件 (n={len(self.event_indices)})')
        ax1.axhline(self.baseline, color='b', linestyle='--', alpha=0.5, label='基线')
        ax1.set_xlabel('时间 (s)')
        ax1.set_ylabel(f'{self.abf.sweepLabelY} ({self.abf.sweepUnitsY})')
        ax1.set_title('完整记录轨迹与检测到的事件')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 振幅分布直方图
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.hist(self.event_features['amplitude'], bins=30, color='steelblue', edgecolor='black')
        ax2.axvline(self.event_features['amplitude'].mean(), color='r', 
                   linestyle='--', label=f'均值: {self.event_features["amplitude"].mean():.1f} pA')
        ax2.set_xlabel('振幅 (pA)')
        ax2.set_ylabel('事件数')
        ax2.set_title('振幅分布')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 上升时间分布
        ax3 = fig.add_subplot(gs[1, 1])
        valid_rise = self.event_features['rise_time_10_90'].dropna()
        ax3.hist(valid_rise, bins=30, color='green', edgecolor='black')
        ax3.axvline(valid_rise.mean(), color='r', 
                   linestyle='--', label=f'均值: {valid_rise.mean():.2f} ms')
        ax3.set_xlabel('上升时间 10-90% (ms)')
        ax3.set_ylabel('事件数')
        ax3.set_title('上升时间分布')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 衰减时间常数分布
        ax4 = fig.add_subplot(gs[1, 2])
        valid_decay = self.event_features['decay_tau'].dropna()
        valid_decay = valid_decay[valid_decay < 20]  # 过滤异常值
        ax4.hist(valid_decay, bins=30, color='orange', edgecolor='black')
        ax4.axvline(valid_decay.mean(), color='r', 
                   linestyle='--', label=f'均值: {valid_decay.mean():.2f} ms')
        ax4.set_xlabel('衰减 τ (ms)')
        ax4.set_ylabel('事件数')
        ax4.set_title('衰减时间常数分布')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. 事件面积分布
        ax5 = fig.add_subplot(gs[2, 0])
        valid_area = self.event_features['area'].dropna()
        ax5.hist(valid_area, bins=30, color='purple', edgecolor='black')
        ax5.axvline(valid_area.mean(), color='r', 
                   linestyle='--', label=f'均值: {valid_area.mean():.1f} pA*ms')
        ax5.set_xlabel('面积 (pA*ms)')
        ax5.set_ylabel('事件数')
        ax5.set_title('事件面积分布')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. 振幅 vs 衰减时间散点图
        ax6 = fig.add_subplot(gs[2, 1])
        valid_data = self.event_features[['amplitude', 'decay_tau']].dropna()
        valid_data = valid_data[valid_data['decay_tau'] < 20]
        ax6.scatter(valid_data['amplitude'], valid_data['decay_tau'], 
                   alpha=0.5, s=20, color='steelblue')
        ax6.set_xlabel('振幅 (pA)')
        ax6.set_ylabel('衰减 τ (ms)')
        ax6.set_title('振幅 vs 衰减时间')
        ax6.grid(True, alpha=0.3)
        
        # 7. 事件频率随时间变化
        ax7 = fig.add_subplot(gs[2, 2])
        bin_size = 5  # 5 秒的时间窗
        time_bins = np.arange(0, self.abf.dataLengthSec, bin_size)
        event_times = self.event_features['peak_time'].values
        hist, _ = np.histogram(event_times, bins=time_bins)
        freq = hist / bin_size
        ax7.plot(time_bins[:-1], freq, 'o-', color='darkgreen')
        ax7.axhline(freq.mean(), color='r', linestyle='--', 
                   label=f'平均: {freq.mean():.2f} Hz')
        ax7.set_xlabel('时间 (s)')
        ax7.set_ylabel('事件频率 (Hz)')
        ax7.set_title(f'事件频率随时间变化 (窗口: {bin_size}s)')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8-10. 示例事件波形
        for i in range(3):
            ax = fig.add_subplot(gs[3, i])
            if i < len(self.event_indices):
                idx = self.event_indices[i * (len(self.event_indices) // 3)]
                
                window_ms = 15
                window_samples = int(window_ms * self.sampling_rate / 1000)
                start = max(0, idx - window_samples // 4)
                end = min(len(self.data), idx + 3 * window_samples // 4)
                
                plot_time = (self.time[start:end] - self.time[idx]) * 1000
                plot_data = self.data[start:end]
                
                ax.plot(plot_time, plot_data, 'k-', linewidth=1.5)
                ax.axhline(self.baseline, color='r', linestyle='--', alpha=0.5)
                ax.axvline(0, color='b', linestyle='--', alpha=0.5)
                
                event_feat = self.event_features.iloc[i * (len(self.event_indices) // 3)]
                ax.set_title(f'示例事件 {i+1}\n振幅: {event_feat["amplitude"]:.1f} pA, '
                           f'τ: {event_feat["decay_tau"]:.1f} ms')
                ax.set_xlabel('时间 (ms)')
                ax.set_ylabel('电流 (pA)')
                ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'mEPSC 分析报告 - {os.path.basename(self.abf_path)}', 
                    fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"已保存概览图: {save_path}")
        
        return fig


def main():
    """主函数 - 命令行接口"""
    parser = argparse.ArgumentParser(description='mEPSC/mIPSC 自动分析工具')
    parser.add_argument('input_file', help='输入的 ABF 文件路径')
    parser.add_argument('-c', '--channel', type=int, default=0, 
                       help='要分析的通道编号 (默认: 0)')
    parser.add_argument('-t', '--threshold', type=float, default=3.5,
                       help='检测阈值因子 (默认: 3.5)')
    parser.add_argument('-m', '--min-interval', type=float, default=5,
                       help='最小事件间隔 (ms) (默认: 5)')
    parser.add_argument('-o', '--output-dir', default=None,
                       help='输出目录 (默认: 与输入文件相同)')
    parser.add_argument('--min-amp', type=float, default=None,
                       help='最小振幅过滤 (pA)')
    parser.add_argument('--max-amp', type=float, default=None,
                       help='最大振幅过滤 (pA)')
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = mEPSCAnalyzer(args.input_file, channel=args.channel)
    
    # 估计基线和噪声
    analyzer.estimate_baseline()
    analyzer.estimate_noise()
    
    # 检测事件
    analyzer.detect_events(
        threshold_factor=args.threshold,
        min_interval_ms=args.min_interval,
        min_amplitude=args.min_amp,
        max_amplitude=args.max_amp
    )
    
    # 分析所有事件
    analyzer.analyze_all_events()
    
    # 显示汇总统计
    print("\n" + "="*60)
    print("汇总统计")
    print("="*60)
    summary = analyzer.get_summary_statistics()
    print(summary.to_string())
    
    # 保存结果
    output_dir = args.output_dir if args.output_dir else os.path.dirname(args.input_file)
    csv_path, summary_path = analyzer.save_results(output_dir)
    
    # 绘制并保存概览图
    base_name = os.path.splitext(os.path.basename(args.input_file))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(output_dir, f"{base_name}_analysis_{timestamp}.png")
    analyzer.plot_overview(save_path=plot_path)
    
    print("\n分析完成!")


if __name__ == "__main__":
    main()
