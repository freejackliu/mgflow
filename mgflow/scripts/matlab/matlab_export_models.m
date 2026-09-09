%% MATLAB script: Export RQGPR model parameters for Python
% Run this ONCE in MATLAB to save model parameters as simple .mat files
% that Python can read with scipy.io.loadmat.
%
% Usage: Run this script from the TgTxTl_E_H directory in MATLAB.

models = {
    'RQGPR_Tg_r',    'Tg';
    'RQGPR_Tx_r',    'Tx';
    'RQGPR_Tl_r8_new.mat', 'Tl';
    'RQGPR_E_r10',   'E';
    'H_8_reduced_byziqing', 'H';
};

reduction_files = {
    'Tg_reduction.mat',         'Tg';
    'Tx_reduction.mat',         'Tx';
    'Tl_reduced_r_new_0.mat',   'Tl';
    'E_reduction.mat',          'E';
    'H_reduce_new_byziqing.mat','H';
};

output_dir = 'exported_models';
mkdir(output_dir);

% Export reduction matrices
fprintf('=== Exporting reduction matrices ===\n');
for i = 1:size(reduction_files, 1)
    fname = reduction_files{i,1};
    prop  = reduction_files{i,2};
    if ~isfile(fname)
        fprintf('  SKIP %s (not found)\n', fname);
        continue;
    end
    load(fname, 'i_min', 'RMSE_mean_min');
    
    save(fullfile(output_dir, sprintf('reduction_%s.mat', prop)), ...
         'i_min', 'RMSE_mean_min');
    fprintf('  reduction_%s.mat saved (i_min: %d vars)\n', prop, length(i_min));
end

% Export GPR model parameters
fprintf('\n=== Exporting GPR model parameters ===\n');
for i = 1:size(models, 1)
    mname = models{i,1};
    prop  = models{i,2};
    if ~isfile(mname)
        % Try without .mat
        if ~isfile([mname, '.mat'])
            fprintf('  SKIP %s (not found)\n', mname);
            continue;
        end
        mname = [mname, '.mat'];
    end
    
    S = load(mname);
    fnames = fieldnames(S);
    gp_struct = S.(fnames{1});
    
    if ~isfield(gp_struct, 'RegressionGP')
        fprintf('  SKIP %s (no RegressionGP field)\n', mname);
        continue;
    end
    
    gp = gp_struct.RegressionGP;
    
    % Extract parameters
    Alpha = gp.Alpha;
    ActiveSetVectors = gp.ActiveSetVectors;
    KernelInfo = gp.KernelInformation;
    Sigma = gp.Sigma;
    Beta = gp.Beta;
    
    % Standardization info
    if isprop(gp, 'StandardizeMu') && ~isempty(gp.StandardizeMu)
        StandardizeMu = gp.StandardizeMu;
        StandardizeSigma = gp.StandardizeSigma;
    else
        StandardizeMu = [];
        StandardizeSigma = [];
    end
    
    % Predictor location/scale
    if isprop(gp, 'PredictorLocation')
        PredictorLocation = gp.PredictorLocation;
        PredictorScale = gp.PredictorScale;
    else
        PredictorLocation = [];
        PredictorScale = [];
    end
    
    save(fullfile(output_dir, sprintf('gpr_%s.mat', prop)), ...
         'Alpha', 'ActiveSetVectors', 'KernelInfo', ...
         'Sigma', 'Beta', ...
         'StandardizeMu', 'StandardizeSigma', ...
         'PredictorLocation', 'PredictorScale');
    fprintf('  gpr_%s.mat saved (ActiveSet: %d pts, kernel: %s)\n', ...
        prop, size(ActiveSetVectors, 1), KernelInfo.KernelFunction);
end

fprintf('\nDone! Models exported to: %s\n', fullfile(pwd, output_dir));
