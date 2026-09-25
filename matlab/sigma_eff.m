function s = sigma_eff(material, f)
%SIGMA_EFF  Effective conductivity (S/m) = eps'' * w * eps0, as papers usually tabulate it.
eps0 = 8.8541878128e-12;
s = -imag(eps_c(material, f)) .* (2*pi*f) * eps0;
end
