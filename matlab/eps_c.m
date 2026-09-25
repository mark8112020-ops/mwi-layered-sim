function e = eps_c(material, f)
%EPS_C  Complex relative permittivity eps' - j*eps'' of a material at frequencies f (Hz).
%
%   e = eps_c(material, f)   material is a struct from tissue_params()
%
%   N-pole Cole-Cole plus static conductivity, time convention exp(+jwt):
%       eps_c = eps_inf + sum d_eps./(1+(j*w*tau).^(1-alpha)) - j*sigma./(w*eps0)

eps0 = 8.8541878128e-12;
w = 2*pi*f;
e = material.eps_inf * ones(size(f));
for k = 1:size(material.poles, 1)
    d_eps = material.poles(k,1); tau = material.poles(k,2); alpha = material.poles(k,3);
    if d_eps ~= 0
        e = e + d_eps ./ (1 + (1i*w*tau).^(1 - alpha));
    end
end
e = e - 1i*material.sigma ./ (w*eps0);
end
