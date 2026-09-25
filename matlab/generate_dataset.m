function data = generate_dataset(n, seed, out_file, opts)
%GENERATE_DATASET  Build a balanced labelled S11 dataset and save it as .mat.
%
%   data = generate_dataset()                          12000 samples, default seed
%   data = generate_dataset(n, seed, out_file)
%   data = generate_dataset(n, seed, out_file, opts)   opts.contrast (default 1),
%                                                      opts.benign_lesion_prob (0.5)
%
%   Runs validate_model() first and refuses to continue if it fails.
%   S11 is stored CLEAN; apply_measurement_noise() corrupts it afterwards, so the
%   same physics can be re-corrupted at any noise level without regenerating.
%
%   Output fields: S (n x 602 complex), y (1 = malignant), f, band_idx, meta (table).

if nargin < 1 || isempty(n), n = 12000; end
if nargin < 2 || isempty(seed), seed = 20260801; end
if nargin < 3 || isempty(out_file), out_file = fullfile(fileparts(mfilename('fullpath')), '..', 'data', 'dataset_matlab.mat'); end
if nargin < 4, opts = struct(); end
if ~isfield(opts, 'contrast'), opts.contrast = 1.0; end
if ~isfield(opts, 'benign_lesion_prob'), opts.benign_lesion_prob = 0.5; end

if ~validate_model()
    error('generate_dataset:validation', 'Forward model failed validation; not generating.');
end

tp = tissue_params(opts.contrast);
f = [tp.bands(1).f tp.bands(2).f];
n1 = numel(tp.bands(1).f);
band_idx = {1:n1, n1+1:numel(f)};

rng(seed);
y = [ones(1, floor(n/2)) zeros(1, n - floor(n/2))];
y = y(randperm(n));                      % shuffled so order carries nothing

S = complex(zeros(n, numel(f)));
metas = cell(n, 1);
t0 = tic;
for k = 1:n
    [S(k,:), metas{k}] = sample_stack(tp, f, y(k), opts.benign_lesion_prob);
    if mod(k, 1000) == 0
        fprintf('  %d/%d  (%.0f samples/s)\n', k, n, k/toc(t0));
    end
end

data.S = S; data.y = y(:); data.f = f; data.band_idx = band_idx;
data.meta = struct2table([metas{:}]);
data.config = struct('n', n, 'seed', seed, 'contrast', opts.contrast, ...
    'benign_lesion_prob', opts.benign_lesion_prob, ...
    'model', '1D layered, plane wave, normal incidence');

out_dir = fileparts(out_file);
if ~isempty(out_dir) && ~exist(out_dir, 'dir'), mkdir(out_dir); end
save(out_file, '-struct', 'data', '-v7');
fprintf('wrote %s  (%d x %d complex, %d malignant / %d benign)\n', ...
    out_file, n, numel(f), sum(y), sum(1 - y));
end
