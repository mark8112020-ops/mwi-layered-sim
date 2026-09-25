%% DEMO_PHYSICS  Layered-skin S11 model: the physics half of the experiment in MATLAB.
%
%  Open this file and press Run. It needs base MATLAB only -- no toolboxes.
%  Each section can also be run on its own with Ctrl+Enter (Cmd+Enter on Mac).
%
%  The machine-learning half (random forest, cross-validation, the phase-error
%  sweep) lives in ../python. This file reproduces the physics so you can explore
%  it interactively, and exports a dataset the same way the Python does.

clear; close all; clc;
here = fileparts(mfilename('fullpath'));
addpath(here);

%% 1. Is the forward model trustworthy?  (must say ALL CHECKS PASSED)
assert(validate_model(), 'Validation failed -- stop here.');

%% 2. The tissues: how permittivity and loss change with frequency
tp = tissue_params();
f = linspace(1e9, 30e9, 400);
mats = {tp.m.stratum_corneum, tp.m.epidermis, tp.m.dermis, tp.m.hypodermis, tp.m.malignant, tp.m.benign};
figure('Name', 'Tissue dielectrics');
for k = 1:numel(mats)
    e = eps_c(mats{k}, f);
    subplot(1,2,1); plot(f/1e9, real(e), 'LineWidth', 1.3); hold on;
    subplot(1,2,2); plot(f/1e9, -imag(e), 'LineWidth', 1.3); hold on;
end
names = cellfun(@(m) strrep(m.name, '_', ' '), mats, 'UniformOutput', false);
subplot(1,2,1); xlabel('GHz'); ylabel('\epsilon'''); title('How polar (real part)'); grid on;
subplot(1,2,2); xlabel('GHz'); ylabel('\epsilon'''''); title('How lossy (imag part)'); grid on;
legend(names, 'Location', 'northeast');

%% 3. One patient: build a stack by hand and look at its S11
fb = [tp.bands(1).f tp.bands(2).f];
band_idx = {1:301, 302:602};
rng(1);
[S_mal, meta_mal, layers] = sample_stack(tp, fb, 1);
fprintf('\nOne malignant sample, layers top to bottom:\n');
for k = 1:numel(layers)-1
    fprintf('  %-16s %8.1f um\n', layers(k).name, layers(k).t*1e6);
end
fprintf('  %-16s  semi-infinite\n', layers(end).name);

%% 4. Benign vs malignant, averaged over many patients
rng(2);
N = 400;
Sb = complex(zeros(N, numel(fb))); Sm = Sb;
for k = 1:N
    Sb(k,:) = sample_stack(tp, fb, 0);
    Sm(k,:) = sample_stack(tp, fb, 1);
end
figure('Name', 'Class means');
for b = 1:2
    idx = band_idx{b};
    subplot(1,2,b); hold on;
    hb = shade(fb(idx)/1e9, 20*log10(abs(Sb(:,idx))), [0.2 0.4 0.8]);
    hm = shade(fb(idx)/1e9, 20*log10(abs(Sm(:,idx))), [0.85 0.3 0.25]);
    xlabel('GHz'); ylabel('|S11|, dB'); grid on;
    title(sprintf('%s band: mean \\pm 1 sd', tp.bands(b).name));
end
legend([hb hm], {'benign', 'malignant'});
fprintf(['\nNote how small the class difference is next to the spread between patients.\n' ...
         'That is deliberate: hydration, body site and age move these numbers more than\n' ...
         'the tumour does, and a model without that spread produces a fake 99%%.\n']);

%% 5. Why phase error hurts Re/Im but not |S11|  (the arrow picture)
Sclean = Sm(1:50, :);
opts = struct('snr_db', Inf, 'mag_gain_std', 0, 'mag_drift_std', 0);
opts.phase_err_deg = 0;  Y0  = apply_measurement_noise(Sclean, fb, band_idx, opts);
opts.phase_err_deg = 20; Y20 = apply_measurement_noise(Sclean, fb, band_idx, opts);
ch_mag = max(abs(abs(Y20) - abs(Y0)), [], 'all');
ch_re  = max(abs(real(Y20) - real(Y0)), [], 'all');
fprintf('\n20 deg of phase error changed |S11| by at most %.2e and Re(S11) by up to %.3f.\n', ch_mag, ch_re);
fprintf('Rotating the arrow moves its shadows (Re, Im) but never its length (|S11|).\n');

figure('Name', 'Phase error rotates the arrow');
k = 150;
plot(real(Y0(:,k)), imag(Y0(:,k)), 'o', 'MarkerFaceColor', [0.2 0.4 0.8]); hold on;
plot(real(Y20(:,k)), imag(Y20(:,k)), 'x', 'Color', [0.85 0.3 0.25], 'LineWidth', 1.2);
th = linspace(0, 2*pi, 200); plot(cos(th), sin(th), 'k:');
axis equal; grid on; xlabel('Re(S11)'); ylabel('Im(S11)');
title(sprintf('50 patients at %.1f GHz: 0 vs 20 deg phase error', fb(k)/1e9));
legend({'no phase error', '20 deg rms', '|S11| = 1'});

%% 6. Export a full dataset (optional, ~15-60 s)
% data = generate_dataset(12000);
% noisy = apply_measurement_noise(data.S, data.f, data.band_idx);   % nominal: 5 deg, 30 dB

function h = shade(x, Ydb, c)
mu = mean(Ydb, 1); sd = std(Ydb, 0, 1);
fill([x fliplr(x)], [mu+sd fliplr(mu-sd)], c, 'FaceAlpha', 0.15, 'EdgeColor', 'none');
h = plot(x, mu, 'Color', c, 'LineWidth', 1.5);
end
