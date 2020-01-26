function [z, history] = basis_pursuit_CRM_smart(A, b, rho, alpha)
% basis_pursuit  Solve basis pursuit via ADMM
%
% [x, history] = basis_pursuit(A, b, rho, alpha)
% 
% Solves the following problem via ADMM:
% 
%   minimize     ||x||_1
%   subject to   Ax = b
%
% The solution is returned in the vector x.
%
% history is a structure that contains the objective value, the primal and 
% dual residual norms, and the tolerances for the primal and dual residual 
% norms at each iteration.
% 
% rho is the augmented Lagrangian parameter. 
%
% alpha is the over-relaxation parameter (typical values for alpha are 
% between 1.0 and 1.8).
%
% More information can be found in the paper linked at:
% http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
%

t_start = tic;

%% Global constants and defaults

QUIET    = 0;
MAX_ITER = 1000;
ABSTOL   = 1e-8;
RELTOL   = 1e-8;

%% Data preprocessing

[m n] = size(A);

%% ADMM solver

x = zeros(n,1);
z = zeros(n,1);
u = zeros(n,1);
Rach = zeros(n,1);

if ~QUIET
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
      'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
end

% precompute static variables for x-update (projection on to Ax=b)
AAt = A*A';
P = eye(n) - A' * (AAt \ A);
q = A' * (AAt \ b);

%We start with regular ADMM

for k = 1:500
    % x-update
    x = P*(z - u) + q;

    % z-update with relaxation
    zold = z;
    x_hat = alpha*x + (1 - alpha)*zold;
    z = shrinkage(x_hat + u, 1/rho);

    uold=u;
    u = u + (x_hat - z);
    
    %I save the previous DR iterate
    Rachold = Rach;
    %I compute the next DR iterate
    Rach = u + rho*z; 
    
    

    % diagnostics, reporting, termination checks
    history.objval(k)  = objective(A, b, x);
    
    history.u_diff(k)  = norm(uold-u);

    history.r_norm(k)  = norm(x - z);
    history.s_norm(k)  = norm(-rho*(z - zold));
    
    %I record the DR iterate subsequent differences
    history.Rach_diff(k) = norm(Rachold-Rach);
    
    history.eps_pri(k) = sqrt(n)*ABSTOL + RELTOL*max(norm(x), norm(-z));
    history.eps_dual(k)= sqrt(n)*ABSTOL + RELTOL*norm(rho*u);
    history.eps_u(k)= sqrt(n)*ABSTOL + RELTOL*norm(u);

    if ~QUIET
        fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
            history.r_norm(k), history.eps_pri(k), ...
            history.s_norm(k), history.eps_dual(k), history.objval(k));
    end

    if (history.r_norm(k) < history.eps_pri(k) && ...
       history.s_norm(k) < history.eps_dual(k))
         break;
    end
end

%Now we do the accelerated version
uCT = u;
uREG = u;

for k = 500:MAX_ITER
    
    % x-update
    xCT = P*(z - uCT) + q;
    xREG = P*(z - uREG) + q;
    if objective(A, b, xCT) > objective(A,b,xREG)
        x = xREG;
        u = uREG;
    else
        x = xCT;
        u = uCT;
    end
    
    %having updated x one extra time, I can update all my reflections and governing dual sequence
    
    %y_{k-1} update
    DRy = u+rho*z;
    %R_cd2 y_{k-1} update
    Ry = u-rho*z;
    %R_cd1 R_cd2 y_{k-1} update
    RRy = u -rho*z+2*rho*x; 
    

    % z-update with relaxation
    zold = z;
    x_hat = alpha*x + (1 - alpha)*zold;
    z = shrinkage(x_hat + u, 1/rho);
    
    %update the multiplier
    uold=u;
    if colinear_check(DRy,Ry,RRy,10^(-3)) == 1 %I check for colinearity
		center = circumcenter(Ry,RRy,DRy); %I take the circumcenter of the dual
        uCT = proj_box(center,1); %I compute its shadow
    else
        uCT = u + (x_hat - z); % If colinear, do the regular ADMM update
    end
    uREG = u + (x_hat - z); %I compute the regular ADMM multiplier update for comparison
    
    
    
    
 
    

    % diagnostics, reporting, termination checks
	
	
    Rachold = Rach; %I save the previous DR iterate
    Rach = u + rho*z; %I compute the next DR iterate
	
    history.objval(k)  = objective(A, b, x);
    
    history.u_diff(k)  = norm(uold-u);

    history.r_norm(k)  = norm(x - z);
    history.s_norm(k)  = norm(-rho*(z - zold));
    history.DR_diff(k) = norm(DRy-center);
    
    %I record the DR iterate subsequent differences
    history.Rach_diff(k) = norm(Rachold-Rach);
    
    history.eps_pri(k) = sqrt(n)*ABSTOL + RELTOL*max(norm(x), norm(-z));
    history.eps_dual(k)= sqrt(n)*ABSTOL + RELTOL*norm(rho*u);
    history.eps_u(k)= sqrt(n)*ABSTOL + RELTOL*norm(u);

    if ~QUIET
        fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
            history.r_norm(k), history.eps_pri(k), ...
            history.s_norm(k), history.eps_dual(k), history.objval(k));
    end

    if (history.r_norm(k) < history.eps_pri(k) && ...
       history.s_norm(k) < history.eps_dual(k))
         break;
    end
end









if ~QUIET
    toc(t_start);
end

end






function obj = objective(A, b, x)
    obj = norm(x,1);
end

function y = shrinkage(a, kappa)
    y = max(0, a-kappa) - max(0, -a-kappa);
end
