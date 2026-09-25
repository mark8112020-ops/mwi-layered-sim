function Y = apply_measurement_noise(S, f, band_idx, opts)
%APPLY_MEASUREMENT_NOISE  Corrupt clean S11 the way a real front end would.
%
%   Y = apply_measurement_noise(S, f, band_idx, opts)
%
%   S         n_samples x n_freq complex, clean
%   f         1 x n_freq, Hz
%   band_idx  cell array of column-index vectors, one per band (phase is unwrapped
%             and smooth errors are drawn per band)
%   opts      struct, any field optional:
%     .snr_db          30     additive receiver noise, vs mean |S11|^2 in band
%     .mag_gain_std    0.02   per-sweep scalar gain error
%     .mag_drift_std   0.01   smooth in-band magnitude ripple
%     .phase_err_deg   5      TOTAL rms phase error (the swept parameter)
%     .phase_noise_frac 0.40  iid point-to-point share
%     .phase_drift_frac 0.917 smooth share (0.4^2 + 0.917^2 = 1)
%     .delay_err_frac  0.70   share of the smooth part that is residual cable delay
%
%   Why this exists: a noiseless comparison always favours complex S11, because
%   complex data is a superset of magnitude data. Phase is the fragile quantity in a
%   cheap front end -- cable flex and temperature act mostly as a residual electrical
%   delay (phase linear in frequency) -- and that is modelled explicitly here.
%   The smooth phase error is normalised so each sweep has exactly phase_err_deg rms.

d = struct('snr_db', 30, 'mag_gain_std', 0.02, 'mag_drift_std', 0.01, ...
           'phase_err_deg', 5, 'phase_noise_frac', 0.40, 'phase_drift_frac', 0.917, ...
           'delay_err_frac', 0.70);
if nargin < 4, opts = struct(); end
fn = fieldnames(d);
for k = 1:numel(fn)
    if ~isfield(opts, fn{k}), opts.(fn{k}) = d.(fn{k}); end
end
if nargin < 3 || isempty(band_idx), band_idx = {1:size(S,2)}; end

[n, nf] = size(S);
mag = abs(S); pha = angle(S);

% --- magnitude: per-sweep gain + smooth 3-harmonic ripple -----------------------------
gain = 1 + opts.mag_gain_std*randn(n, 1);
drift = zeros(n, nf);
for b = 1:numel(band_idx)
    idx = band_idx{b};
    u = linspace(0, 1, numel(idx));
    comp = zeros(n, numel(idx));
    for k = 1:3
        comp = comp + randn(n,1) .* cos(2*pi*k*u + 2*pi*rand(n,1));
    end
    drift(:, idx) = comp / sqrt(1.5);            % unit rms
end
mag = mag .* gain .* (1 + opts.mag_drift_std*drift);

% --- phase: iid noise + smooth drift (mostly residual delay) ---------------------------
ph = opts.phase_err_deg*pi/180;
if ph > 0
    pha = pha + ph*opts.phase_noise_frac*randn(n, nf);
    smooth = zeros(n, nf);
    for b = 1:numel(band_idx)
        idx = band_idx{b};
        fb = f(idx);
        u = (fb - fb(1)) / (fb(end) - fb(1));
        shape = u - mean(u); shape = shape / sqrt(mean(shape.^2));
        delay = randn(n,1) .* shape;                 % linear-in-frequency delay error
        ripple = zeros(n, numel(idx));
        for k = 1:2
            ripple = ripple + randn(n,1) .* cos(2*pi*k*u + 2*pi*rand(n,1));
        end
        sm = sqrt(opts.delay_err_frac)*delay + sqrt(1 - opts.delay_err_frac)*ripple;
        rms = sqrt(mean(sm.^2, 2)); rms(rms == 0) = 1;
        smooth(:, idx) = sm ./ rms;                  % exactly unit rms per sweep
    end
    pha = pha + ph*opts.phase_drift_frac*smooth;
end

Y = mag .* exp(1i*pha);

% --- additive receiver noise -----------------------------------------------------------
if isfinite(opts.snr_db)
    for b = 1:numel(band_idx)
        idx = band_idx{b};
        p = mean(abs(Y(:, idx)).^2, 'all');
        sig = sqrt(p / 10^(opts.snr_db/10) / 2);
        Y(:, idx) = Y(:, idx) + sig*(randn(n, numel(idx)) + 1i*randn(n, numel(idx)));
    end
end
end
