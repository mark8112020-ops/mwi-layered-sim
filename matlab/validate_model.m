function ok = validate_model()
%VALIDATE_MODEL  Forward-model checks. If any fail, STOP -- do not use the model.
%
%   ok = validate_model()
%
%   0. Dielectric tables reproduce the published IFAC/Gabriel anchor values.
%   1. Lossless half-space matches the closed-form Fresnel coefficient.
%   2. Lossy tissue attenuates: the field decays rather than grows.
%   3. A quarter-wave matching layer nulls S11 at its design frequency (and 3x it).
%   4. Thin-layer limit: as a layer's thickness -> 0, S11 matches the stack without it.
%   5. Passivity: |S11| never exceeds 1 for random physiological stacks.
%
%   The same checks, with the same tolerances, as python/validate.py.

tp = tissue_params();
f = linspace(2e9, 26e9, 241);
res = false(1, 6);
fprintf('%s\nforward-model validation\n%s\n', repmat('=',1,70), repmat('=',1,70));

% 0 -------------------------------------------------------------------------------------
fprintf('%-20s %6s %10s %9s %10s %9s  %s\n', 'tissue', 'f GHz', 'eps model', 'eps pub', 'sig model', 'sig pub', 'status');
bad = 0;
for k = 1:size(tp.anchors, 1)
    [key, fa, e_pub, s_pub, note] = tp.anchors{k, :};
    mt = tp.m.(key);
    e_m = real(eps_c(mt, fa)); s_m = sigma_eff(mt, fa);
    good = abs(e_m - e_pub)/e_pub <= 0.01 && abs(s_m - s_pub)/s_pub <= 0.02;
    if good, st = 'OK'; elseif ~isempty(note), st = 'WARN (documented)'; else, st = 'FAIL'; bad = bad + 1; end
    fprintf('%-20s %6.2f %10.3f %9.3f %10.4f %9.4f  %s\n', key, fa/1e9, e_m, e_pub, s_m, s_pub, st);
end
res(1) = bad == 0;
report(0, 'dielectric anchors vs IFAC/Gabriel', res(1), sprintf('%d unexpected failures', bad));

% 1 -------------------------------------------------------------------------------------
worst = 0; e_ref = 2.2*ones(size(f));
for er = [1 4 12 40]
    e_t = er*ones(size(f));
    got = layered_s11(struct('name', 'half', 'eps', e_t, 't', 0), f, e_ref);
    eta1 = 1./sqrt(e_ref); eta2 = 1./sqrt(e_t);
    want = (eta2 - eta1)./(eta2 + eta1);
    worst = max(worst, max(abs(got - want)));
end
res(2) = worst < 1e-12;
report(1, 'lossless half-space vs Fresnel', res(2), sprintf('max error %.3e (tol 1e-12)', worst));

% 2 -------------------------------------------------------------------------------------
good = true;
for mt = {tp.m.dermis, tp.m.malignant, tp.m.hypodermis, tp.m.skin_dry}
    e = eps_c(mt{1}, f);
    g = 1i*(2*pi*f/tp.c0).*sqrt(e);
    good = good && all(imag(e) <= 0) && all(real(g) > 0) && all(real(1./sqrt(e)) > 0);
end
d = 1/real(1i*(2*pi*2e9/tp.c0)*sqrt(eps_c(tp.m.dermis, 2e9)));
res(3) = good;
report(2, 'lossy tissue decays', res(3), sprintf('dermis penetration depth %.2f mm at 2 GHz', d*1e3));

% 3 -------------------------------------------------------------------------------------
e1 = 2.2; e2 = 12; em = sqrt(e1*e2); f0 = 10e9;
dq = tp.c0/(f0*sqrt(em))/4;
fq = linspace(2e9, 34e9, 6401);
st = struct('name', {'match', 'half'}, 'eps', {em*ones(size(fq)), e2*ones(size(fq))}, 't', {dq, 0});
r = abs(layered_s11(st, fq, e1*ones(size(fq))));
[~, i0] = min(abs(fq - f0)); [~, i3] = min(abs(fq - 3*f0)); [~, imin] = min(r);
res(4) = r(i0) < 1e-9 && r(i3) < 1e-9 && abs(fq(imin) - f0)/f0 < 1e-3;
report(3, 'quarter-wave null at design f', res(4), sprintf('|S11| at 10 GHz %.2e, at 30 GHz %.2e', r(i0), r(i3)));

% 4 -------------------------------------------------------------------------------------
e_ref = eps_c(tp.reference, f);
base = struct('name', {'epi', 'derm', 'hypo'}, ...
    'eps', {eps_c(tp.m.epidermis, f), eps_c(tp.m.dermis, f), eps_c(tp.m.hypodermis, f)}, ...
    't', {100e-6, 2e-3, 0});
ref = layered_s11(base, f, e_ref);
errs = zeros(1, 6); ds = [1e-3 1e-4 1e-5 1e-6 1e-8 0];
for k = 1:numel(ds)
    thin = struct('name', 'thin', 'eps', eps_c(tp.m.malignant, f), 't', ds(k));
    errs(k) = max(abs(layered_s11([thin base], f, e_ref) - ref));
end
res(5) = all(diff(errs) <= 1e-15 + 0.5*errs(1:end-1)) && errs(end) < 1e-14;
report(4, 'thin-layer limit', res(5), sprintf('error %.2e at 1 mm -> %.2e at 0', errs(1), errs(end)));

% 5 -------------------------------------------------------------------------------------
rng(0);
fb = [tp.bands(1).f tp.bands(2).f];
worst = 0;
for k = 1:400
    S = sample_stack(tp, fb, randi([0 1]), 0.5);
    worst = max(worst, max(abs(S)));
end
res(6) = worst <= 1 + 1e-12;
report(5, 'passivity |S11| <= 1', res(6), sprintf('max |S11| over 400 stacks = %.6f', worst));

ok = all(res);
fprintf('%s\n', repmat('-',1,70));
if ok, fprintf('ALL CHECKS PASSED\n');
else,  fprintf('VALIDATION FAILED -- do not use this model\n'); end
end

function report(n, name, good, msg)
if good, s = 'PASS'; else, s = 'FAIL'; end
fprintf('[%s] %d. %s\n       %s\n', s, n, name, msg);
end
