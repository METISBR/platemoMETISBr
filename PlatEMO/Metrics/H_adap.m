function score = H_adap(Population, optimum)
% <min> <multi/many> <real> <large/none> <constrained/none>
% H_adap - Adaptive HKKT indicator via quantile winsorization
%
% Ajustes feitos:
% (A) Corrige "Invalid array indexing" removendo padrao: CalObj(...)(:)
% (B) Evita score=NaN se apenas parte da populacao falhar (filtra s finitos)
% (C) Captura Problem (obj) no nivel correto para PlatEMO (caller)

    % ================================================================
    % 1) Recuperar PROBLEM no nivel certo (PlatEMO PROBLEM/CalMetric)
    % ================================================================
    Problem = [];
    try
        Problem = evalin('caller','obj');     % PlatEMO costuma ter "obj"
    catch
        Problem = [];
    end

    % Fallback: as vezes optimum pode ser um PROBLEM
    if isempty(Problem)
        try
            if isobject(optimum) && ismethod(optimum,'CalObj')
                Problem = optimum;
            end
        catch
        end
    end

    % Fallback: base workspace (execucao fora do PlatEMO)
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

    % ================================================================
    % 2) Residuos HKKT (espinha dorsal do H_old)
    % ================================================================
    s = localResiduals(Population, Problem);

    % >>> Correção crítica (B): filtrar invalidos antes de agregar <<<
    s = s(isfinite(s));
    if isempty(s)
        score = NaN;   % ninguem foi computavel
        return;
    end

    % ================================================================
    % 3) Núcleo adaptativo (winsorization por quantis)
    % ================================================================
    alpha = 0.10;
    beta  = 0.90;
    try alpha = evalin('base','HAlpha'); catch; end
    try beta  = evalin('base','HBeta');  catch; end
    alpha = max(min(alpha,0.49),0.01);
    beta  = max(min(beta,0.99),alpha+0.01);

    if exist('quantile','file') == 2
        ql = quantile(s, alpha);
        qu = quantile(s, beta);
    else
        ql = prctile(s, 100*alpha);
        qu = prctile(s, 100*beta);
    end

    denom = qu - ql;
    if ~(denom > 0) || ~isfinite(denom)
        score = 0;
        return;
    end

    shat = min(max(s, ql), qu);
    eps0 = 1e-12;
    z = (shat - ql) ./ (denom + eps0);
    z = max(0, min(1, z));

    term = z .* log(z + eps0);
    term(z == 0) = 0;
    score = -mean(term);
end

% ======================= helpers =======================

function s = localResiduals(Population, Problem)
    N = numel(Population);
    s = NaN(N,1);

    for i = 1:N
        x = Population(i).dec(:);

        G = localJacobianSafe(Problem, x);
        if isempty(G) || any(~isfinite(G(:)))
            continue; % fica NaN, será filtrado depois
        end

        q = localQdir(G);
        if isempty(q) || any(~isfinite(q))
            continue;
        end

        s(i) = sum(q.^2);
    end
end

function G = localJacobianSafe(Problem, x)
    % Retorna n x m (colunas = gradientes das funcoes objetivo)
    G = [];

    % ---- (1) Se existir CalGrad, usa ----
    try
        if ismethod(Problem,'CalGrad')
            Grad = double(Problem.CalGrad(x'));  % pode vir m x n ou n x m

            m = [];
            try, m = Problem.M; catch, end
            if isempty(m)
                tmp0 = Problem.CalObj(x');
                f0   = double(tmp0(:));
                m    = numel(f0);
            end

            if size(Grad,1) == m
                G = Grad.';       % n x m
            elseif size(Grad,2) == m
                G = Grad;         % n x m
            else
                G = [];
            end
        end
    catch
        G = [];
    end
    if ~isempty(G), return; end

    % ---- (2) Diferencas finitas centrais (robusto) ----
    try
        % (A) Aqui é onde estava o erro: NAO faça CalObj(...)(:)
        tmp0 = Problem.CalObj(x');
        f0   = double(tmp0(:));

        m = numel(f0);
        n = numel(x);
        G = zeros(n,m);

        [lb, ub] = localBounds(Problem, n);

        range = ub - lb;
        range(~isfinite(range) | range <= 0) = 1;
        hvec = 1e-6 * range;

        for k = 1:n
            h  = hvec(k);
            xp = x; xm = x;

            xp(k) = min(ub(k), x(k) + h);
            xm(k) = max(lb(k), x(k) - h);

            if xp(k) == xm(k)
                G(k,:) = 0;
                continue;
            end

            % (A) Corrigido: separar chamada e indexação
            tmpP = Problem.CalObj(xp');
            tmpM = Problem.CalObj(xm');
            fp   = double(tmpP(:));
            fm   = double(tmpM(:));

            if any(~isfinite(fp)) || any(~isfinite(fm))
                G(k,:) = 0;
            else
                G(k,:) = ((fp - fm) / (xp(k) - xm(k))).';
            end
        end

    catch
        G = [];
    end
end

function [lb, ub] = localBounds(Problem, n)
    lb = -inf(n,1);
    ub =  inf(n,1);

    try
        if isprop(Problem,'lower') && ~isempty(Problem.lower)
            L = Problem.lower(:);
            if numel(L) == n, lb = L; end
        end
    catch
    end

    try
        if isprop(Problem,'upper') && ~isempty(Problem.upper)
            U = Problem.upper(:);
            if numel(U) == n, ub = U; end
        end
    catch
    end
end

function q = localQdir(G)
    % Resolve min ||G*alpha||^2 s.a. alpha>=0, sum(alpha)=1 (simplex)
    [~, m] = size(G);

    H   = 2*(G.'*G);
    f   = zeros(m,1);
    A   = -eye(m);
    b   = zeros(m,1);
    Aeq = ones(1,m);
    beq = 1;
    lb  = zeros(m,1);
    ub  = ones(m,1);

    alpha = [];

    % tenta quadprog
    try
        options = optimoptions('quadprog','Display','off');
        alpha = quadprog(H, f, A, b, Aeq, beq, lb, ub, [], options);
    catch
        alpha = [];
    end

    % fallback simples e estável
    if isempty(alpha) || any(~isfinite(alpha)) || any(alpha < -1e-10)
        alpha = ones(m,1) / m;
    end

    q = G * alpha;
end
