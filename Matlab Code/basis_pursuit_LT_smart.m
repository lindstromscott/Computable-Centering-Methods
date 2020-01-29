function [z, history] = basis_pursuit_LT_smart(A, b, rho, alpha)
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
MAX_ITER = 50000;
ABSTOL   = 1e-8;
RELTOL   = 1e-8;

%% Data preprocessing

[m n] = size(A);

%% ADMM solver

x = zeros(n,1);
z = zeros(n,1);
u = zeros(n,1);
y1 = zeros(n,1);
y2 = zeros(n,1);
y3 = zeros(n,1);

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

for k = 1:2
    
    % x-update
    x = P*(z - u) + q;

    % z-update with relaxation
    zold = z;
    x_hat = alpha*x + (1 - alpha)*zold;
    z = shrinkage(x_hat + u, 1/rho);

    uold = u;
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
uREG = u;
uLT = u;
zLT = z;


for k = 2:MAX_ITER
    
    uold = u;
    
    if mod(k,3) == 0 %if it's time to check the centering step, enter
        % x-update
       xREG = P*(z - uREG) + q;
        xLT = P*(zLT - uLT) + q;
       if objective(A, b, xREG) > objective(A,b,xLT)
            x = xLT;
            u = uLT;
            
       else
           x = xREG;
           u = uREG; 
           
       end
    else %if it isn't time to check the centering step, skip the objective function check and update normally
        x = P*(z - u) + q;
    end

    % z-update with relaxation
    zold = z;
    x_hat = alpha*x + (1 - alpha)*zold;
    z = shrinkage(x_hat + u, 1/rho);
	
	%update the multiplier
	u = u + (x_hat - z);
    
    %y_{k-1} update
    if mod(k,3) == 0
        y1 = u+rho*z;
    elseif mod(k,3) == 1
        y2 = u+rho*z;
    else
        y3 = u+rho*z;
    end
    
    %compute the alternative multiplier candidate based on centering
    if mod(k,3) == 2 && colinear_check(y1,2*y2-y1,piT(y1,y2,y3),10^(-20)) == 1

        %on third updates, I circumcenter the dual if not colinear.
        %I then set shadow u to be prox_cd2 of the dual governing sequence
		yLT=circumcenter(y1,2*y2-y1,piT(y1,y2,y3));
        
        uLT = proj_box(yLT,1); 
        zLT = yLT-uLT;
    else
        % otherwise, I do the regular ADMM update
        uLT = u; 
    end
    uREG = u;
    zREG = z;
    
    

    
    

    % diagnostics, reporting, termination checks
	
	    
    Rachold = Rach; %I save the previous DR iterate
    Rach = u + rho*z; %I compute the next DR iterate
	
	
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
