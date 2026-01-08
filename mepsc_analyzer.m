function results = mepsc_analyzer(abf_file, varargin)
% MEPSC_ANALYZER - mEPSC/mIPSC 自动检测和分析工具 (MATLAB版本)
%
% 用法:
%   results = mepsc_analyzer(abf_file)
%   results = mepsc_analyzer(abf_file, 'threshold', 3.5, 'min_interval', 5)
%
% 输入参数:
%   abf_file       - ABF 文件路径
%   'threshold'    - 检测阈值因子 (默认: 3.5)
%   'min_interval' - 最小事件间隔 (ms) (默认: 5)
%   'min_amp'      - 最小振幅过滤 (pA) (可选)
%   'max_amp'      - 最大振幅过滤 (pA) (可选)
%   'channel'      - 通道编号 (默认: 1)
%
% 输出:
%   results        - 包含所有分析结果的结构体
%       .event_features  - 事件特征表
%       .summary         - 汇总统计
%       .baseline        - 基线电流
%       .noise_std       - 噪声标准差
%
% 示例:
%   results = mepsc_analyzer('data.abf', 'threshold', 4.0);
%
% 注意: 需要安装 abfload 函数来读取 ABF 文件
%       可从 https://www.mathworks.com/matlabcentral/fileexchange/6190-abfload
%       或使用 Molecular Devices 提供的工具
%
% 作者: Manus AI
% 日期: 2026-01-08

%% 解析输入参数
p = inputParser;
addRequired(p, 'abf_file', @ischar);
addParameter(p, 'threshold', 3.5, @isnumeric);
addParameter(p, 'min_interval', 5, @isnumeric);
addParameter(p, 'min_amp', [], @isnumeric);
addParameter(p, 'max_amp', [], @isnumeric);
addParameter(p, 'channel', 1, @isnumeric);
parse(p, abf_file, varargin{:});

params = p.Results;

%% 读取 ABF 文件
fprintf('正在读取 ABF 文件...\n');

% 检查是否有 abfload 函数
if ~exist('abfload', 'file')
    error('未找到 abfload 函数。请从以下地址下载:\nhttps://www.mathworks.com/matlabcentral/fileexchange/6190-abfload');
end

try
    [data, si, h] = abfload(abf_file);
catch
    error('无法读取 ABF 文件。请确保文件格式正确。');
end

% 提取数据
sampling_interval = si;  % 微秒
sampling_rate = 1e6 / sampling_interval;  % Hz
data_trace = data(:, params.channel);
time_trace = (0:length(data_trace)-1)' / sampling_rate;
record_length = length(data_trace) / sampling_rate;

fprintf('文件: %s\n', abf_file);
fprintf('采样率: %.0f Hz\n', sampling_rate);
fprintf('记录时长: %.2f 秒\n', record_length);
fprintf('数据点数: %d\n', length(data_trace));

%% 估计基线和噪声
fprintf('\n估计基线和噪声...\n');
baseline = prctile(data_trace, 50);
near_baseline = data_trace(abs(data_trace - baseline) < 2 * std(data_trace));
noise_std = std(near_baseline);

fprintf('基线: %.2f pA\n', baseline);
fprintf('噪声标准差: %.2f pA\n', noise_std);

%% 检测事件
fprintf('\n检测事件...\n');
threshold = baseline - params.threshold * noise_std;
min_interval_samples = round(params.min_interval * sampling_rate / 1000);

% 找到低于阈值的点
below_threshold = data_trace < threshold;

% 找到阈值穿越点
crossings = diff([0; below_threshold]);
event_starts = find(crossings == 1);

% 在每个事件中找到峰值
event_peaks = [];
search_window = round(2 * sampling_rate / 1000);  % 2 ms

for i = 1:length(event_starts)
    start_idx = event_starts(i);
    end_idx = min(start_idx + search_window, length(data_trace));
    
    [peak_value, local_idx] = min(data_trace(start_idx:end_idx));
    peak_idx = start_idx + local_idx - 1;
    peak_amplitude = abs(peak_value - baseline);
    
    % 振幅过滤
    if ~isempty(params.min_amp) && peak_amplitude < params.min_amp
        continue;
    end
    if ~isempty(params.max_amp) && peak_amplitude > params.max_amp
        continue;
    end
    
    % 检查事件间隔
    if isempty(event_peaks) || (peak_idx - event_peaks(end)) > min_interval_samples
        event_peaks(end+1) = peak_idx;
    end
end

event_peaks = event_peaks(:);
num_events = length(event_peaks);
event_frequency = num_events / record_length;

fprintf('检测到 %d 个事件\n', num_events);
fprintf('事件频率: %.2f Hz\n', event_frequency);

%% 提取事件特征
fprintf('\n提取事件特征...\n');

% 预分配数组
event_features = struct();
event_features.event_number = (1:num_events)';
event_features.peak_time = zeros(num_events, 1);
event_features.onset_time = zeros(num_events, 1);
event_features.amplitude = zeros(num_events, 1);
event_features.rise_time_10_90 = zeros(num_events, 1);
event_features.decay_tau = zeros(num_events, 1);
event_features.decay_time_90_10 = zeros(num_events, 1);
event_features.half_width = zeros(num_events, 1);
event_features.area = zeros(num_events, 1);

pre_samples = round(2 * sampling_rate / 1000);   % 2 ms
post_samples = round(20 * sampling_rate / 1000);  % 20 ms

for i = 1:num_events
    if mod(i, 50) == 0
        fprintf('  进度: %d/%d\n', i, num_events);
    end
    
    peak_idx = event_peaks(i);
    start_idx = max(1, peak_idx - pre_samples);
    end_idx = min(length(data_trace), peak_idx + post_samples);
    
    event_data = data_trace(start_idx:end_idx);
    event_time = time_trace(start_idx:end_idx);
    
    % 基本特征
    peak_value = data_trace(peak_idx);
    event_features.peak_time(i) = time_trace(peak_idx);
    event_features.amplitude(i) = abs(peak_value - baseline);
    
    % 找到起始点
    pre_event = data_trace(start_idx:peak_idx);
    onset_threshold = baseline - 0.1 * event_features.amplitude(i);
    onset_candidates = find(pre_event > onset_threshold);
    if ~isempty(onset_candidates)
        onset_idx = start_idx + onset_candidates(end) - 1;
    else
        onset_idx = start_idx;
    end
    event_features.onset_time(i) = time_trace(onset_idx);
    
    % 上升时间 10%-90%
    if onset_idx < peak_idx
        amp_10 = baseline - 0.1 * event_features.amplitude(i);
        amp_90 = baseline - 0.9 * event_features.amplitude(i);
        
        rise_data = data_trace(onset_idx:peak_idx);
        idx_10 = find(rise_data < amp_10, 1, 'first');
        idx_90 = find(rise_data < amp_90, 1, 'first');
        
        if ~isempty(idx_10) && ~isempty(idx_90)
            t_10 = time_trace(onset_idx + idx_10 - 1);
            t_90 = time_trace(onset_idx + idx_90 - 1);
            event_features.rise_time_10_90(i) = (t_90 - t_10) * 1000;  % ms
        else
            event_features.rise_time_10_90(i) = NaN;
        end
    else
        event_features.rise_time_10_90(i) = NaN;
    end
    
    % 衰减时间 - 指数拟合
    decay_data = data_trace(peak_idx:end_idx);
    decay_time = (time_trace(peak_idx:end_idx) - time_trace(peak_idx)) * 1000;  % ms
    
    if length(decay_data) > 20
        try
            % 单指数拟合
            fit_samples = min(length(decay_data), round(10 * sampling_rate / 1000));
            ft = fittype('A*exp(-x/tau) + C', 'independent', 'x');
            opts = fitoptions('Method', 'NonlinearLeastSquares');
            opts.StartPoint = [-(peak_value - baseline), 5, baseline];
            opts.Lower = [-Inf, 0.1, -Inf];
            opts.Upper = [0, 50, Inf];
            
            [fitted_curve, gof] = fit(decay_time(1:fit_samples), ...
                                      decay_data(1:fit_samples), ft, opts);
            
            event_features.decay_tau(i) = abs(fitted_curve.tau);
        catch
            event_features.decay_tau(i) = NaN;
        end
        
        % 衰减时间 90%-10%
        amp_90_decay = baseline - 0.9 * event_features.amplitude(i);
        amp_10_decay = baseline - 0.1 * event_features.amplitude(i);
        
        idx_90 = find(decay_data < amp_90_decay, 1, 'first');
        idx_10 = find(decay_data > amp_10_decay, 1, 'first');
        
        if ~isempty(idx_90) && ~isempty(idx_10) && idx_10 > idx_90
            event_features.decay_time_90_10(i) = decay_time(idx_10);
        else
            event_features.decay_time_90_10(i) = NaN;
        end
    else
        event_features.decay_tau(i) = NaN;
        event_features.decay_time_90_10(i) = NaN;
    end
    
    % 事件面积
    if onset_idx < end_idx
        area_data = data_trace(onset_idx:end_idx) - baseline;
        area_time = time_trace(onset_idx:end_idx);
        area_data = min(area_data, 0);  % 只计算低于基线的部分
        event_features.area(i) = abs(trapz(area_time, area_data)) * 1000;  % pA*ms
    else
        event_features.area(i) = NaN;
    end
    
    % 半高宽
    half_amp = baseline - 0.5 * event_features.amplitude(i);
    half_width_points = find(event_data < half_amp);
    if length(half_width_points) > 1
        event_features.half_width(i) = ...
            (event_time(half_width_points(end)) - event_time(half_width_points(1))) * 1000;
    else
        event_features.half_width(i) = NaN;
    end
end

fprintf('特征提取完成!\n');

%% 计算汇总统计
summary = struct();
summary.file_name = abf_file;
summary.record_length_sec = record_length;
summary.num_events = num_events;
summary.event_frequency_Hz = event_frequency;
summary.baseline_pA = baseline;
summary.noise_std_pA = noise_std;
summary.mean_amplitude_pA = nanmean(event_features.amplitude);
summary.std_amplitude_pA = nanstd(event_features.amplitude);
summary.median_amplitude_pA = nanmedian(event_features.amplitude);
summary.mean_rise_time_ms = nanmean(event_features.rise_time_10_90);
summary.mean_decay_tau_ms = nanmean(event_features.decay_tau);
summary.mean_area_pA_ms = nanmean(event_features.area);
summary.mean_half_width_ms = nanmean(event_features.half_width);

fprintf('\n========================================\n');
fprintf('汇总统计\n');
fprintf('========================================\n');
fprintf('文件: %s\n', summary.file_name);
fprintf('事件总数: %d\n', summary.num_events);
fprintf('事件频率: %.2f Hz\n', summary.event_frequency_Hz);
fprintf('平均振幅: %.2f ± %.2f pA\n', summary.mean_amplitude_pA, summary.std_amplitude_pA);
fprintf('平均上升时间: %.2f ms\n', summary.mean_rise_time_ms);
fprintf('平均衰减 tau: %.2f ms\n', summary.mean_decay_tau_ms);
fprintf('平均面积: %.2f pA*ms\n', summary.mean_area_pA_ms);

%% 保存结果
[filepath, name, ~] = fileparts(abf_file);
timestamp = datestr(now, 'yyyymmdd_HHMMSS');

% 保存为 MAT 文件
results_file = fullfile(filepath, sprintf('%s_results_%s.mat', name, timestamp));
save(results_file, 'event_features', 'summary', 'baseline', 'noise_std', ...
     'data_trace', 'time_trace', 'event_peaks');
fprintf('\n已保存结果: %s\n', results_file);

% 保存为 CSV 文件
csv_file = fullfile(filepath, sprintf('%s_events_%s.csv', name, timestamp));
features_table = struct2table(event_features);
writetable(features_table, csv_file);
fprintf('已保存 CSV: %s\n', csv_file);

%% 绘制分析图
fprintf('\n生成分析图表...\n');
fig = figure('Position', [100, 100, 1400, 1000]);

% 1. 完整轨迹
subplot(4, 3, [1, 2, 3]);
plot(time_trace, data_trace, 'k-', 'LineWidth', 0.3);
hold on;
plot(time_trace(event_peaks), data_trace(event_peaks), 'r.', 'MarkerSize', 5);
yline(baseline, 'b--', 'LineWidth', 1.5);
xlabel('时间 (s)');
ylabel('电流 (pA)');
title(sprintf('完整记录轨迹 (检测到 %d 个事件)', num_events));
legend('数据', '检测到的事件', '基线', 'Location', 'best');
grid on;

% 2. 振幅分布
subplot(4, 3, 4);
histogram(event_features.amplitude, 30, 'FaceColor', [0.2, 0.4, 0.8]);
hold on;
xline(summary.mean_amplitude_pA, 'r--', 'LineWidth', 2);
xlabel('振幅 (pA)');
ylabel('事件数');
title(sprintf('振幅分布 (均值: %.1f pA)', summary.mean_amplitude_pA));
grid on;

% 3. 上升时间分布
subplot(4, 3, 5);
valid_rise = event_features.rise_time_10_90(~isnan(event_features.rise_time_10_90));
histogram(valid_rise, 30, 'FaceColor', [0.2, 0.8, 0.4]);
hold on;
xline(nanmean(valid_rise), 'r--', 'LineWidth', 2);
xlabel('上升时间 10-90% (ms)');
ylabel('事件数');
title(sprintf('上升时间分布 (均值: %.2f ms)', nanmean(valid_rise)));
grid on;

% 4. 衰减时间分布
subplot(4, 3, 6);
valid_decay = event_features.decay_tau(~isnan(event_features.decay_tau));
valid_decay = valid_decay(valid_decay < 20);
histogram(valid_decay, 30, 'FaceColor', [0.8, 0.4, 0.2]);
hold on;
xline(nanmean(valid_decay), 'r--', 'LineWidth', 2);
xlabel('衰减 τ (ms)');
ylabel('事件数');
title(sprintf('衰减时间分布 (均值: %.2f ms)', nanmean(valid_decay)));
grid on;

% 5. 面积分布
subplot(4, 3, 7);
histogram(event_features.area, 30, 'FaceColor', [0.6, 0.2, 0.8]);
hold on;
xline(summary.mean_area_pA_ms, 'r--', 'LineWidth', 2);
xlabel('面积 (pA*ms)');
ylabel('事件数');
title(sprintf('事件面积分布 (均值: %.1f pA*ms)', summary.mean_area_pA_ms));
grid on;

% 6. 振幅 vs 衰减时间
subplot(4, 3, 8);
valid_idx = ~isnan(event_features.decay_tau) & event_features.decay_tau < 20;
scatter(event_features.amplitude(valid_idx), event_features.decay_tau(valid_idx), ...
        20, 'filled', 'MarkerFaceAlpha', 0.5);
xlabel('振幅 (pA)');
ylabel('衰减 τ (ms)');
title('振幅 vs 衰减时间');
grid on;

% 7. 事件频率随时间变化
subplot(4, 3, 9);
bin_size = 5;  % 5秒窗口
time_bins = 0:bin_size:record_length;
[counts, ~] = histcounts(event_features.peak_time, time_bins);
freq_over_time = counts / bin_size;
plot(time_bins(1:end-1), freq_over_time, 'o-', 'LineWidth', 1.5, 'MarkerSize', 6);
hold on;
yline(event_frequency, 'r--', 'LineWidth', 2);
xlabel('时间 (s)');
ylabel('事件频率 (Hz)');
title(sprintf('事件频率随时间变化 (窗口: %ds)', bin_size));
grid on;

% 8-10. 示例事件
for i = 1:3
    subplot(4, 3, 9 + i);
    if i <= num_events
        event_idx = round(i * num_events / 4);
        peak_idx = event_peaks(event_idx);
        
        window_ms = 15;
        window_samples = round(window_ms * sampling_rate / 1000);
        start_idx = max(1, peak_idx - round(window_samples / 4));
        end_idx = min(length(data_trace), peak_idx + round(3 * window_samples / 4));
        
        plot_time = (time_trace(start_idx:end_idx) - time_trace(peak_idx)) * 1000;
        plot_data = data_trace(start_idx:end_idx);
        
        plot(plot_time, plot_data, 'k-', 'LineWidth', 1.5);
        hold on;
        yline(baseline, 'r--');
        xline(0, 'b--');
        
        xlabel('时间 (ms)');
        ylabel('电流 (pA)');
        title(sprintf('示例事件 %d\n振幅: %.1f pA, τ: %.1f ms', ...
                     i, event_features.amplitude(event_idx), ...
                     event_features.decay_tau(event_idx)));
        grid on;
    end
end

sgtitle(sprintf('mEPSC 分析报告 - %s', name), 'FontSize', 14, 'FontWeight', 'bold');

% 保存图表
fig_file = fullfile(filepath, sprintf('%s_analysis_%s.png', name, timestamp));
saveas(fig, fig_file);
fprintf('已保存分析图: %s\n', fig_file);

%% 输出结果
results = struct();
results.event_features = event_features;
results.summary = summary;
results.baseline = baseline;
results.noise_std = noise_std;
results.data_trace = data_trace;
results.time_trace = time_trace;
results.event_peaks = event_peaks;

fprintf('\n分析完成!\n');

end
