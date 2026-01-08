#!/usr/bin/env python3
"""
批量分析多个 ABF 文件的示例脚本

用法:
    python batch_analysis.py /path/to/data/folder
"""

import os
import sys
import glob
import pandas as pd
from pathlib import Path

# 添加父目录到路径以导入 mepsc_analyzer
sys.path.insert(0, str(Path(__file__).parent.parent))
from mepsc_analyzer import mEPSCAnalyzer


def batch_analyze(data_folder, output_folder=None, **kwargs):
    """
    批量分析文件夹中的所有 ABF 文件
    
    参数:
        data_folder: 包含 ABF 文件的文件夹路径
        output_folder: 输出文件夹（默认为 data_folder/results）
        **kwargs: 传递给 detect_events 的参数
    """
    # 设置输出文件夹
    if output_folder is None:
        output_folder = os.path.join(data_folder, 'results')
    os.makedirs(output_folder, exist_ok=True)
    
    # 找到所有 ABF 文件
    abf_files = glob.glob(os.path.join(data_folder, '*.abf'))
    
    if len(abf_files) == 0:
        print(f"错误: 在 {data_folder} 中未找到 ABF 文件")
        return
    
    print(f"找到 {len(abf_files)} 个 ABF 文件")
    print(f"输出目录: {output_folder}\n")
    
    # 存储所有汇总结果
    all_summaries = []
    
    # 分析每个文件
    for i, abf_file in enumerate(abf_files, 1):
        print(f"[{i}/{len(abf_files)}] 处理: {os.path.basename(abf_file)}")
        
        try:
            # 创建分析器
            analyzer = mEPSCAnalyzer(abf_file)
            
            # 估计基线和噪声
            analyzer.estimate_baseline()
            analyzer.estimate_noise()
            
            # 检测事件
            analyzer.detect_events(**kwargs)
            
            # 分析所有事件
            analyzer.analyze_all_events()
            
            # 获取汇总统计
            summary = analyzer.get_summary_statistics()
            all_summaries.append(summary)
            
            # 保存结果
            analyzer.save_results(output_folder)
            
            # 生成并保存分析图
            base_name = os.path.splitext(os.path.basename(abf_file))[0]
            plot_path = os.path.join(output_folder, f"{base_name}_analysis.png")
            analyzer.plot_overview(save_path=plot_path)
            
            print(f"  ✓ 检测到 {len(analyzer.event_indices)} 个事件")
            print(f"  ✓ 平均振幅: {summary['平均振幅 (pA)']:.2f} pA\n")
            
        except Exception as e:
            print(f"  ✗ 错误: {str(e)}\n")
            continue
    
    # 合并所有汇总结果
    if all_summaries:
        combined_summary = pd.DataFrame(all_summaries)
        summary_path = os.path.join(output_folder, 'all_files_summary.csv')
        combined_summary.to_csv(summary_path, index=False)
        print(f"\n所有文件处理完成!")
        print(f"合并的汇总结果已保存到: {summary_path}")
        
        # 显示统计信息
        print("\n=== 整体统计 ===")
        print(f"成功分析的文件数: {len(all_summaries)}")
        print(f"平均事件频率: {combined_summary['事件频率 (Hz)'].mean():.2f} Hz")
        print(f"平均振幅: {combined_summary['平均振幅 (pA)'].mean():.2f} pA")
    else:
        print("\n警告: 没有成功分析任何文件")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python batch_analysis.py <data_folder> [output_folder]")
        print("\n示例:")
        print("  python batch_analysis.py ./data")
        print("  python batch_analysis.py ./data ./results")
        sys.exit(1)
    
    data_folder = sys.argv[1]
    output_folder = sys.argv[2] if len(sys.argv) > 2 else None
    
    # 运行批量分析
    batch_analyze(
        data_folder,
        output_folder,
        threshold_factor=3.5,
        min_interval_ms=5
    )
