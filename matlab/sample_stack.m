function [S, meta, layers] = sample_stack(tp, f, label, benign_lesion_prob)
%SAMPLE_STACK  Draw one physiological realisation and compute its clean S11.
%
%   [S, meta, layers] = sample_stack(tp, f, label, benign_lesion_prob)
%
%   label: 1 = malignant, 0 = benign. Per sample this randomises:
%     - layer thicknesses within the table ranges
%     - a shared hydration factor across all skin layers (body site, age, hydration)
%     - independent 10% log-normal jitter on every Cole-Cole parameter
%     - the coupling gap (probe standoff / contact-pressure proxy)
%     - lesion presence, depth and thickness
%   A benign_lesion_prob fraction of BENIGN samples also get an inclusion (with
%   intermediate properties), so the classifier cannot win by detecting "any lump".
%   Uses the global random stream: call rng(seed) first for reproducibility.

if nargin < 4, benign_lesion_prob = 0.5; end

hyd = lognorm1(tp.hydration_std);
meta = struct('label', label, 'hydration', hyd);

layers = struct('name', {}, 'eps', {}, 't', {});
for k = 1:numel(tp.stack)
    sp = tp.stack(k);
    if sp.is_tissue
        m = jitter(sp.mat, tp.jitter_std, hyd);
    else
        m = sp.mat;                      % the air gap gets no physiological jitter
    end
    t = sp.t_min + (sp.t_max - sp.t_min)*rand;
    layers(end+1) = struct('name', sp.name, 'eps', eps_c(m, f), 't', t); %#ok<AGROW>
    meta.(['t_' sp.name]) = t;
end
t_before = sum([layers(1:end-1).t]);

has_inc = (label == 1) || (rand < benign_lesion_prob);
meta.has_inclusion = double(has_inc);
meta.lesion_top_depth = 0; meta.lesion_thickness = 0;

if has_inc
    % Same geometry distribution for both classes, so geometry carries no label.
    skin = meta.t_stratum_corneum + meta.t_epidermis + meta.t_dermis;
    r = tp.lesion_top_depth;  d_top = r(1) + (r(2)-r(1))*rand;
    d_top = min(d_top, max(0, skin - 0.2e-3));
    r = tp.lesion_thickness;  t_les = r(1) + (r(2)-r(1))*rand;
    t_les = min(t_les, skin - d_top);            % keep the lesion inside the skin

    if label == 1, m = tp.m.malignant; else, m = tp.m.benign; end
    m = jitter(m, tp.jitter_std, hyd);           % lesion shares the hydration factor
    layers = carve_inclusion(layers, d_top, t_les, 'lesion', eps_c(m, f), 2);
    meta.lesion_top_depth = d_top; meta.lesion_thickness = t_les;
end

% Leakage tripwire: carving must never change total stack thickness.
t_after = sum([layers(1:end-1).t]);
assert(abs(t_after - t_before) < 1e-12, 'carve_inclusion changed total thickness: label leak');
meta.t_total = t_after;

S = layered_s11(layers, f, eps_c(tp.reference, f));
end

% -----------------------------------------------------------------------------------------
function m = jitter(m, sd, common)
% Multiplicative log-normal jitter (cannot go negative), plus the shared hydration
% factor on the water-driven quantities (d_eps and sigma).
m.eps_inf = m.eps_inf * lognorm1(sd);
for k = 1:size(m.poles, 1)
    m.poles(k,1) = m.poles(k,1) * lognorm1(sd) * common;
    m.poles(k,2) = m.poles(k,2) * lognorm1(sd);
end
m.sigma = m.sigma * lognorm1(sd) * common;
end

function x = lognorm1(sd)
% Mean-one log-normal draw with the given relative std (base MATLAB, no toolbox).
if sd <= 0, x = 1; return; end
s = sqrt(log(1 + sd^2));
x = exp(-0.5*s^2 + s*randn);
end
