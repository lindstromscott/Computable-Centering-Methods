# Computable-Centering-Methods
Code for the article "Computable Centering Methods for Spiraling Algorithms and their Duals, with Motivations from the Theory of Lyapunov Functions"

The Cinderella scripts were used to generate the images in the paper. They are self-explanatory.

The basis pursuit code that I have adapted is from S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein. It is located here: https://web.stanford.edu/~boyd/papers/admm/

____________________________________________________________________________________________________
File description
----------------------

FILE: colinear_check
DESCRIPTION: This function takes as its input three points y1,y2,y3 and a nonnegative number r. It returns the normalized inner product two normalized distance vectors, which should have absolute value 1 if y1,y2,y3 are distinct and colinear. The nonnegative number is a numerical threshold, so that if the absolute value computed is greater than 1-r, we decide the three vectors are colinear and return the value 0. Otherwise, we return 1.

FILE: circumcenter
DESCRIPTION: returns the circumcenter of three points, assuming that the three points are not all distinct and colinear. If the three points are distinct and colinear, it will throw an error.

FILE: basis_pursuit
DESCRIPTION: This function is the original ADMM code.

FILE: basis_pursuit_CRM
DESCRIPTION: This function computes regular ADMM and then changes to computing a primal-dual method based on CRM. It does not solve the basis pursuit problem, for reasons explained in the article.

FILE: basis_pursuit_CRM_smart
DESCRIPTION: This function computes regular ADMM and then changes to a "smart" version of the primal-dual method based on CRM, where we query the objective function values to decide whether to accept an update based on centering or to reject it in favor of a normal update. It also does not solve the basis pursuit problem.

FILE: piT
DESCRIPTION: takes as inputs y,y+, and y++, and returns pi_T(y).

FILE: proj_box
DESCRIPTION: computes the projection onto the unit box, which is the unit ball in the max norm.

FILE: basis_pursuit_LT
DESCRIPTION: This function computes regular ADMM and then changes to computing the primal-dual method based on LT.

FILE: basis_pursuit_LT_smart
DESCRIPTION: This function computes regular ADMM for a chosen number of iterations and then changes to a "smart" version of the primal-dual method based on LT, where we query the objective function values to decide whether to accept an update based on centering or to reject it in favor of a normal update.

FILE: basis_pursuit_experiments
DESCRIPTION: runs experiments and generates comparison data

____________________________________________________________________________________________________
Pasting the following code into the console generates the data and prints the example from the article.
----------------------


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


____________________________________________________________________________________________________
Copying and pasting the following code into the console produces the descriptive statistics.
----------------------


%Run my experiments

[statisticsLT statisticsADMM] = basis_pursuit_experiments(A, b, 1000)


%should be near zero, to verify both ADMM and LT had the same solutions

max(statisticsLT.LTvsADMM) 

%view the wins of LT

LTwins = sum(statisticsLT.wins)

%view the failures of LT

LTfails = sum(statisticsLT.fails)

%view the wins of ADMM

ADMMwins = sum(statisticsADMM.wins)

%view the failures of ADMM

ADMMfails = sum(statisticsADMM.fails)

%view quantiles

pp = 0:0.25:1;

LTdata = quantile(statisticsLT.iterates,pp)

ADMMdata = quantile(statisticsADMM.iterates,pp)


