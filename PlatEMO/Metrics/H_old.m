function score = H_old(Population,optimum)
% <min> <multi/many> <real> <large/none> <constrained/none>
% H_old - Entropy-inspired HKKT indicator with fixed saturation (1/e)
%
% Faithful to the 2017 definition:
%   alpha*(x) in argmin_{alpha>=0, 1^T alpha = 1} ||G(x) alpha||^2
%   s(x) = ||G(x) alpha*(x)||^2
%   H_old(X) = -(1/N) sum_i t_i log(t_i),  t_i = min(1/e, s_i), 0log0:=0
%
% PlatEMO calling convention:
%   PROBLEM/CalMetric calls: score = feval(metName,Population,obj.optimum)
%   Therefore, the PROBLEM instance is available in the *caller* workspace as 'obj'.

    Problem = [];
    % 1) Caller workspace (PlatEMO PROBLEM/CalMetric)
    try
        Problem = evalin('caller','obj');
    catch
        Problem = [];
    end
    % 2) Some pipelines may pass PROBLEM as optimum
    if isempty(Problem)
        try
            if isobject(optimum) && ismethod(optimum,'CalObj')
                Problem = optimum;
            end
        catch
        end
    end
    % 3) Script fallback (base workspace)
    if isempty(Problem)
        try
            Problem = evalin('base','Problem');
        catch
            Problem = [];
        end
    end

    if isempty(Problem) || isempty(Population) || ~isobject(Problem) || ~ismethod(Problem,'CalObj')
        score = NaN;
        return;
    end

    s = localResiduals(Population,Problem);
    if any(~isfinite(s))
        score = NaN;
        return;
    end

    sat  = 1/exp(1);
    t    = min(sat, max(0,s(:)));
    term = t .* log(t);
    term(t==0) = 0;
    score = -mean(term);
end

% ======================= helpers =======================

function s = localResiduals(Population,Problem)
    N = numel(Population);
    s = NaN(N,1);
    for i = 1:N
        x = Population(i).dec(:);
        G = localJacobianSafe(Problem,x);   % n x m
        if isempty(G) || any(~isfinite(G(:)))
            continue;
        end
        q = localQdir(G);
        s(i) = sum(q.^2);
    end
end

function G = localJacobianSafe(Problem,x)
    % Returns n x m, columns are gradients of objectives at x.
    G = [];

    % (1) Prefer CalGrad if present
    try
        if ismethod(Problem,'CalGrad')
            Grad = double(Problem.CalGrad(x'));  % may be m x n or n x m
            m = [];
            try, m = Problem.M; catch, end
            if isempty(m)
                tmp0 = Problem.CalObj(x');
                m = numel(double(tmp0(:)));
            end
            if size(Grad,1)==m
                G = Grad.';      % n x m
            elseif size(Grad,2)==m
                G = Grad;        % n x m
            else
                G = [];
            end
        end
    catch
        G = [];
    end
    if ~isempty(G), return; end

    % (2) Central finite differences with bound clipping
    try
        tmp0 = Problem.CalObj(x');
        f0   = double(tmp0(:));
        m    = numel(f0);
        n    = numel(x);
        G    = zeros(n,m);

        [lb,ub] = localBounds(Problem,n);

        range = ub - lb;
        range(~isfinite(range) | range<=0) = 1;
        hvec  = 1e-6 * range;

        for k = 1:n
            h  = hvec(k);
            xp = x; xm = x;
            xp(k) = min(ub(k), x(k) + h);
            xm(k) = max(lb(k), x(k) - h);

            if xp(k) == xm(k)
                G(k,:) = 0; continue;
            end

            tmpP = Problem.CalObj(xp');
            tmpM = Problem.CalObj(xm');
            fp   = double(tmpP(:));
            fm   = double(tmpM(:));

            if any(~isfinite(fp)) || any(~isfinite(fm))
                % one-sided within bounds
                xq = x; xq(k) = xp(k);
                tmpQ = Problem.CalObj(xq');
                fq   = double(tmpQ(:));
                denom = xq(k) - x(k);
                if denom == 0 || any(~isfinite(fq))
                    G(k,:) = 0;
                else
                    G(k,:) = ((fq - f0)/denom).';
                end
            else
                G(k,:) = ((fp - fm) / (xp(k)-xm(k))).';
            end
        end
    catch
        G = [];
    end
end

function [lb,ub] = localBounds(Problem,n)
    lb = -inf(n,1); ub = inf(n,1);
    try
        if isprop(Problem,'lower') && ~isempty(Problem.lower)
            L = Problem.lower(:);
            if numel(L)==n, lb = L; end
        end
    catch
    end
    try
        if isprop(Problem,'upper') && ~isempty(Problem.upper)
            U = Problem.upper(:);
            if numel(U)==n, ub = U; end
        end
    catch
    end
end

function q = localQdir(G)
    [~,m] = size(G);
    H = 2*(G.'*G);
    f = zeros(m,1);
    A = -eye(m); b = zeros(m,1);
    Aeq = ones(1,m); beq = 1;
    lb = zeros(m,1); ub = ones(m,1);

    alpha = [];
    try
        options = optimoptions('quadprog','Display','off');
        alpha = quadprog(H,f,A,b,Aeq,beq,lb,ub,[],options);
    catch
        alpha = [];
    end
    if isempty(alpha) || any(~isfinite(alpha))
        try
            options = optimoptions('lsqlin','Algorithm','interior-point','Display','off');
            d = zeros(size(G,1),1);
            alpha = lsqlin(G,d,A,b,Aeq,beq,lb,ub,[],options);
        catch
            alpha = [];
        end
    end
    if isempty(alpha) || any(~isfinite(alpha)) || any(alpha < -1e-10)
        alpha = ones(m,1)/m;
    end
    q = G*alpha;
end
