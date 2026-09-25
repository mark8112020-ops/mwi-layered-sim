function tp = tissue_params(contrast_scale)
%TISSUE_PARAMS  Single source of truth for every dielectric number (MATLAB port).
%
%   tp = tissue_params()          nominal tumour contrast
%   tp = tissue_params(0.5)       tumour contrast scaled (0 = null experiment)
%
%   Mirrors python/tissue_params.py exactly. Each material is a struct:
%       .name, .eps_inf, .poles (N x 3: [d_eps, tau_s, alpha]), .sigma (S/m),
%       .source, .provisional
%
%   Dispersion model (N-pole Cole-Cole, time convention exp(+jwt)):
%       eps_c(w) = eps_inf + sum d_eps/(1+(j*w*tau)^(1-alpha)) - j*sigma/(w*eps0)
%   alpha = 0 is single-pole Debye.
%
%   Anything marked provisional = true still needs a real source (TODO_SOURCE).
%   See docs/DETAILS.md for the full sourcing notes.

if nargin < 1, contrast_scale = 1.0; end

ps = 1e-12; ns = 1e-9; us = 1e-6; ms = 1e-3; um = 1e-6; mm = 1e-3;

tp.eps0 = 8.8541878128e-12;
tp.c0   = 299792458.0;
tp.eta0 = 376.730313668;

GABRIEL = 'Gabriel, Gabriel & Corthout, Phys. Med. Biol. 41(11), 2271-2293, 1996';
MIRBEIK = 'Mirbeik-Sabzevari et al., IEEE TBME 65(5), 1320-1329, 2018';

% --- reference / coupling media ----------------------------------------------
m.duroid = mat('rt_duroid_5880', 2.2, zeros(0,3), ...
    2.2*0.0009*2*pi*10e9*tp.eps0, 'Rogers RT/duroid 5880 datasheet', false);
m.air = mat('air', 1.00059, zeros(0,3), 0, 'standard atmosphere', false);

% --- Gabriel 4-Cole-Cole tissues -------------------------------------------------
m.skin_dry = mat('skin_dry', 4.0, ...
    [32.0 7.23*ps 0.00; 1100 32.48*ns 0.20; 0 159.15*us 0.20; 0 15.915*ms 0.20], ...
    0.0002, [GABRIEL ' Skin (Dry)'], false);
m.skin_wet = mat('skin_wet', 4.0, ...
    [39.0 7.96*ps 0.10; 280 79.58*ns 0.00; 3.0e4 1.59*us 0.16; 3.0e4 1.59*ms 0.20], ...
    0.0004, [GABRIEL ' Skin (Wet)'], false);
m.fat = mat('fat_not_infiltrated', 2.5, ...
    [3.0 7.96*ps 0.20; 15 15.92*ns 0.10; 3.3e4 159.15*us 0.05; 1.0e7 15.915*ms 0.01], ...
    0.01, [GABRIEL ' Fat (Not Infiltrated)'], false);

% --- layer materials ---------------------------------------------------------------
m.stratum_corneum = mat('stratum_corneum', 2.0, [2.0 8.0*ps 0.0], 0.002, ...
    'TODO_SOURCE: our low-water Debye fit', true);
m.epidermis  = m.skin_dry;  m.epidermis.name  = 'epidermis';  m.epidermis.provisional  = true;
m.dermis     = m.skin_wet;  m.dermis.name     = 'dermis';     m.dermis.provisional     = true;
m.hypodermis = m.fat;       m.hypodermis.name = 'hypodermis';

% --- lesions: explicit, auditable contrast on top of dermis ---------------------------
% Tumour = dermis with first pole d_eps x1.16, tau x0.97, sigma +0.35 S/m.
% contrast_scale multiplies the departure from dermis. Benign lesion sits halfway.
TUMOR_D_EPS = 1.16; TUMOR_TAU = 0.97; TUMOR_SIGMA_ADD = 0.35; BENIGN_FRAC = 0.5;
m.malignant = lesion(m.dermis, 'malignant_lesion', contrast_scale, ...
    TUMOR_D_EPS, TUMOR_TAU, TUMOR_SIGMA_ADD, ['TODO_SOURCE: fitted to trend in ' MIRBEIK]);
m.benign = lesion(m.dermis, 'benign_lesion', contrast_scale*BENIGN_FRAC, ...
    TUMOR_D_EPS, TUMOR_TAU, TUMOR_SIGMA_ADD, 'TODO_SOURCE: intermediate, no measurement');
tp.m = m;
tp.reference = m.duroid;

% --- layer stack, top to bottom (last layer semi-infinite) ----------------------------
tp.stack = struct( ...
    'name',  {'coupling_gap', 'stratum_corneum', 'epidermis', 'dermis', 'hypodermis'}, ...
    'mat',   {m.air, m.stratum_corneum, m.epidermis, m.dermis, m.hypodermis}, ...
    't_min', {0,         10*um, 50*um,  1.0*mm, 4.0*mm}, ...
    't_max', {150*um,    20*um, 150*um, 3.0*mm, 10.0*mm}, ...
    'is_tissue', {false, true, true, true, true});

tp.lesion_top_depth = [0.05 1.50]*mm;   % measured from the skin surface
tp.lesion_thickness = [0.30 2.50]*mm;

% --- physiological variation (label-independent) ----------------------------------------
tp.jitter_std    = 0.10;   % per-layer multiplicative, every Cole-Cole parameter
tp.hydration_std = 0.08;   % shared across all skin layers per sample

% --- instrument bands: the two existing spirals ------------------------------------------
tp.bands(1) = struct('name', 'low', 'f', linspace(2e9, 12e9, 301));
tp.bands(2) = struct('name', 'mid', 'f', linspace(8e9, 26e9, 301));

% --- published anchors (IFAC/Gabriel tool, read 2026-08-01) ------------------------------
% {material field, f_Hz, eps', sigma S/m, known mismatch note}
tp.anchors = { ...
    'skin_dry', 2.45e9, 38.007, 1.4640,  '';
    'skin_dry', 10.0e9, 31.290, 8.0138,  '';
    'skin_wet', 2.45e9, 42.853, 1.5919,  '';
    'skin_wet', 10.0e9, 33.528, 8.9510,  '';
    'fat',      2.45e9, 5.2801, 0.10452, 'known ~6% low on sigma; TODO_SOURCE re-transcribe Gabriel Table 1'};
end

% -----------------------------------------------------------------------------------------
function s = mat(name, eps_inf, poles, sigma, source, provisional)
s = struct('name', name, 'eps_inf', eps_inf, 'poles', poles, 'sigma', sigma, ...
           'source', source, 'provisional', provisional);
end

function s = lesion(base, name, cs, d_eps_scale, tau_scale, sigma_add, source)
s = base;
s.name = name;
s.poles(1,1) = base.poles(1,1) * (1 + (d_eps_scale - 1)*cs);
s.poles(1,2) = base.poles(1,2) * (1 + (tau_scale - 1)*cs);
s.sigma      = base.sigma + sigma_add*cs;
s.source = source;
s.provisional = true;
end
