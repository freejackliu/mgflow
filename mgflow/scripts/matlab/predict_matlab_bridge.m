%% MATLAB Bridge: Predict properties from pre-computed features
% Called by Python's Predictor(mode="matlab")
%
% Reads features_in.csv (CSV, one row = one sample),
% outputs predictions.json
%
% Usage (standalone):
%   matlab -batch "predict_matlab_bridge('features_in.csv', 'predictions.json')"
%   matlab -batch "predict_matlab_bridge"

function predict_matlab_bridge(features_file, output_file)
    if nargin < 1
        features_file = 'features_in.csv';
    end
    if nargin < 2
        output_file = 'predictions.json';
    end

    % Add paths to model directory
    base = fileparts(mfilename('fullpath'));
    model_dir = fullfile(base, '..', '..', 'Matlab 程序文件夹', ...
                         'TgTxTl_E_H', 'TgTxTl_E_H');
    addpath(model_dir);

    % Load models
    load(fullfile(model_dir, 'RQGPR_Tg_r.mat'));
    load(fullfile(model_dir, 'RQGPR_Tx_r.mat'));
    load(fullfile(model_dir, 'RQGPR_Tl_r8_new.mat'));
    load(fullfile(model_dir, 'RQGPR_E_r10.mat'));
    load(fullfile(model_dir, 'H_8_reduced_byziqing.mat'));

    % Load reduction matrices
    load(fullfile(model_dir, 'Tg_reduction.mat'));
    load(fullfile(model_dir, 'Tx_reduction.mat'));
    load(fullfile(model_dir, 'Tl_reduced_r_new_0.mat'));
    load(fullfile(model_dir, 'E_reduction.mat'));
    load(fullfile(model_dir, 'H_reduce_new_byziqing.mat'));

    % Read features
    xx = csvread(features_file);

    xx_Tg = xx; xx_Tx = xx; xx_Tl = xx; xx_E = xx; xx_H = xx;

    % Apply reductions
    load(fullfile(model_dir, 'E_reduction.mat'));
    for k = 1:find(RMSE_mean_min == min(min(RMSE_mean_min)))
        xx_E(:, i_min(k)) = [];
    end

    load(fullfile(model_dir, 'H_reduce_new_byziqing.mat'));
    for k = 1:find(RMSE_mean_min == min(min(RMSE_mean_min)))
        xx_H(:, i_min(k)) = [];
    end

    load(fullfile(model_dir, 'Tg_reduction.mat'));
    for k = 1:find(RMSE_mean_min == min(min(RMSE_mean_min)))
        xx_Tg(:, i_min(k)) = [];
    end

    load(fullfile(model_dir, 'Tx_reduction.mat'));
    for k = 1:find(RMSE_mean_min == min(min(RMSE_mean_min)))
        xx_Tx(:, i_min(k)) = [];
    end

    load(fullfile(model_dir, 'Tl_reduced_r_new_0.mat'));
    for k = 1:find(RMSE_mean_min == min(min(RMSE_mean_min)))
        xx_Tl(:, i_min(k)) = [];
    end

    Tg = RQGPR_Tg_r.predictFcn(xx_Tg);
    Tx = RQGPR_Tx_r.predictFcn(xx_Tx);
    Tl = RQGPR_Tl_r8_new.predictFcn(xx_Tl);
    E  = RQGPR_E_r10.predictFcn(xx_E);
    H  = H_8.predictFcn(xx_H);

    % Output as JSON
    result = struct();
    result.Tg = Tg;
    result.Tx = Tx;
    result.Tl = Tl;
    result.E  = E;
    result.H  = H;

    fid = fopen(output_file, 'w');
    fprintf(fid, '%s', jsonencode(result));
    fclose(fid);
    fprintf('Predictions saved to %s\n', output_file);
end
