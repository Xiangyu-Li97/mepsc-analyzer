function batch_analysis(data_folder, output_folder, varargin)
% BATCH_ANALYSIS - 批量分析多个 ABF 文件
%
% 用法:
%   batch_analysis(data_folder)
%   batch_analysis(data_folder, output_folder)
%   batch_analysis(data_folder, output_folder, 'threshold', 3.5)
%
% 输入:
%   data_folder    - 包含 ABF 文件的文件夹路径
%   output_folder  - 输出文件夹（可选，默认为 data_folder/results）
%   varargin       - 传递给 mepsc_analyzer 的参数
%
% 示例:
%   batch_analysis('./data')
%   batch_analysis('./data', './results', 'threshold', 4.0)

% 设置输出文件夹
if nargin < 2 || isempty(output_folder)
    output_folder = fullfile(data_folder, 'results');
end

if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% 找到所有 ABF 文件
files = dir(fullfile(data_folder, '*.abf'));

if isempty(files)
    error('在 %s 中未找到 ABF 文件', data_folder);
end

fprintf('找到 %d 个 ABF 文件\n', length(files));
fprintf('输出目录: %s\n\n', output_folder);

% 存储所有汇总结果
all_summaries = struct();
success_count = 0;

% 分析每个文件
for i = 1:length(files)
    abf_file = fullfile(files(i).folder, files(i).name);
    fprintf('[%d/%d] 处理: %s\n', i, length(files), files(i).name);
    
    try
        % 添加父目录到路径
        addpath(fileparts(fileparts(mfilename('fullpath'))));
        
        % 运行分析
        results = mepsc_analyzer(abf_file, varargin{:});
        
        % 保存结果
        success_count = success_count + 1;
        all_summaries(success_count) = results.summary;
        
        fprintf('  ✓ 检测到 %d 个事件\n', results.summary.num_events);
        fprintf('  ✓ 平均振幅: %.2f pA\n\n', results.summary.mean_amplitude_pA);
        
    catch ME
        fprintf('  ✗ 错误: %s\n\n', ME.message);
        continue;
    end
end

% 保存合并的汇总结果
if success_count > 0
    summary_table = struct2table(all_summaries);
    summary_path = fullfile(output_folder, 'all_files_summary.csv');
    writetable(summary_table, summary_path);
    
    fprintf('\n所有文件处理完成!\n');
    fprintf('合并的汇总结果已保存到: %s\n', summary_path);
    
    % 显示统计信息
    fprintf('\n=== 整体统计 ===\n');
    fprintf('成功分析的文件数: %d\n', success_count);
    fprintf('平均事件频率: %.2f Hz\n', mean([all_summaries.event_frequency_Hz]));
    fprintf('平均振幅: %.2f pA\n', mean([all_summaries.mean_amplitude_pA]));
else
    fprintf('\n警告: 没有成功分析任何文件\n');
end

end
