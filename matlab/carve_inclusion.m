function out = carve_inclusion(layers, top_depth, thickness, inc_name, inc_eps, origin)
%CARVE_INCLUSION  Replace the material in a depth window with an inclusion.
%
%   out = carve_inclusion(layers, top_depth, thickness, inc_name, inc_eps, origin)
%
%   The tumour is carved OUT of the existing stack rather than appended to it, so
%   every interface below it keeps its position and total stack thickness is
%   unchanged. That is the anti-leakage guarantee: the classifier cannot reach the
%   label through geometry, only through the dielectric contrast.
%
%   top_depth is measured from the top of layer ORIGIN (the skin surface = 2, i.e.
%   the layer just below the coupling gap). The last layer is semi-infinite and is
%   never carved into.

if thickness <= 0, out = layers; return; end

out = layers(1:origin-1);
z = 0; z0 = top_depth; z1 = top_depth + thickness;
inserted = false;
body = layers(origin:end);

for i = 1:numel(body)
    L = body(i);
    if i == numel(body)                 % semi-infinite bottom: always terminates
        out(end+1) = L; %#ok<AGROW>
        break
    end
    a = z; b = z + L.t; z = b;
    if b <= z0 || a >= z1               % entirely outside the window
        out(end+1) = L; %#ok<AGROW>
        continue
    end
    if a < z0                           % part above the window keeps its material
        out(end+1) = mk(L.name, L.eps, z0 - a); %#ok<AGROW>
    end
    lo = max(a, z0); hi = min(b, z1);
    if hi > lo                          % overlap becomes inclusion (merged)
        if inserted && strcmp(out(end).name, inc_name)
            out(end).t = out(end).t + (hi - lo);
        else
            out(end+1) = mk(inc_name, inc_eps, hi - lo); %#ok<AGROW>
            inserted = true;
        end
    end
    if b > z1                           % part below the window keeps its material
        out(end+1) = mk(L.name, L.eps, b - z1); %#ok<AGROW>
    end
end
end

function L = mk(name, e, t)
L = struct('name', name, 'eps', e, 't', t);
end
