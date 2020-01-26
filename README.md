# Computable-Centering-Methods
Code for the article Computable Centering Methods for Spiraling Algorithms and their Duals, with Motivations from the Theory of Lyapunov Functions


The following code generates the data and prints the example from the article.

rand('seed', 0);
randn('seed', 0);

n = 30;
m = 10;
A = randn(m,n);

x = sprandn(n, 1, 0.1*n);
b = A*x;

xtrue = x;

[LTx LThistory] = basis_pursuit_LT_smart(A, b, 1.0, 1.0);
[x history] = basis_pursuit(A, b, 1.0, 1.0);

K = length(history.objval);
LTK = length(LThistory.objval);

P3= history.Rach_diff;

semilogy(1:K,P3,'DisplayName','ADMM dual')
title('Regular ADMM vs LT centering')

hold on

P1=history.r_norm;
plot(1:K,P1,'DisplayName','ADMM |x-z|')

LTP3= LThistory.Rach_diff;
plot(1:LTK,LTP3,'DisplayName','LT dual')

hold on

LTP1=LThistory.r_norm;
plot(1:LTK,LTP1,'DisplayName','LT |x-z|')

hold on 

%P2 = history.s_norm;
%LTP2 = LThistory.u_diff;
%plot(1:K,P2)
%plot(1:LTK,LTP2)

hold off

legend
