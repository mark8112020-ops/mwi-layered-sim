function S = layered_s11(layers, f, eps_ref)
%LAYERED_S11  Complex S11 at the probe face for a planar layered stack.
%
%   S = layered_s11(layers, f, eps_ref)
%
%   layers   struct array with fields .name, .eps (complex row vector over f),
%            .t (thickness, m). The LAST layer is semi-infinite; its .t is ignored.
%   f        frequency vector, Hz
%   eps_ref  complex permittivity of the reference medium (sets eta_ref)
%
%   Plane wave at normal incidence, transmission-line recursion from the bottom up:
%       eta_i   = eta0 ./ sqrt(eps_i)
%       gamma_i = 1j*(w/c) .* sqrt(eps_i)
%       Z_N     = eta_N
%       Z_i     = eta_i .* (Z_{i+1} + eta_i.*tanh(g_i*d_i)) ./ (eta_i + Z_{i+1}.*tanh(g_i*d_i))
%       S11     = (Z_1 - eta_ref) ./ (Z_1 + eta_ref)
%
%   Branch check: with eps = eps' - j*eps'', MATLAB's principal sqrt gives
%   Re(gamma) > 0, so the field decays into tissue. This function errors if it ever
%   does not -- a wrong branch silently ruins every result downstream.

eta0 = 376.730313668; c0 = 299792458.0;
w = 2*pi*f;

eta_b = eta0 ./ sqrt(layers(end).eps);
check_decay(1i*(w/c0).*sqrt(layers(end).eps), layers(end).name);
Z = eta_b;

for i = numel(layers)-1:-1:1
    L = layers(i);
    if L.t <= 0, continue; end            % degenerate layer passes Z through
    eta   = eta0 ./ sqrt(L.eps);
    gamma = 1i*(w/c0) .* sqrt(L.eps);
    check_decay(gamma, L.name);
    x = gamma * L.t;
    x = complex(min(max(real(x), -30), 30), imag(x));   % tanh(30)==1 to 1e-26
    t = tanh(x);
    Z = eta .* (Z + eta.*t) ./ (eta + Z.*t);
end

eta_ref = eta0 ./ sqrt(eps_ref);
S = (Z - eta_ref) ./ (Z + eta_ref);
end

function check_decay(gamma, name)
if any(real(gamma) < -1e-9)
    error('layered_s11:branch', ...
        '%s: Re(gamma) < 0, field grows into the medium. Branch or sign of eps'''' is wrong.', name);
end
end
