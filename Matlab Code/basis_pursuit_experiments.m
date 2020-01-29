function [statisticsLT, statisticsADMM] = basis_pursuit_experiments(A, b, num_experiments)


for k = 1:num_experiments

	rand('seed', k);
	randn('seed', k);

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

	%semilogy(1:K,P3,'DisplayName','ADMM dual')
	%title('Regular ADMM vs LT centering')

	%hold on

	%P1=history.r_norm;
	%plot(1:K,P1,'DisplayName','ADMM |x-z|')

	%LTP3= LThistory.Rach_diff;
	%plot(1:LTK,LTP3,'DisplayName','LT dual')

	%hold on

	%LTP1=LThistory.r_norm;
	%plot(1:LTK,LTP1,'DisplayName','LT |x-z|')

	%hold on 

	%P2 = history.s_norm;
	%LTP2 = LThistory.u_diff;
	%plot(1:K,P2)
	%plot(1:LTK,LTP2)

	%hold off

	%legend
	
	%update statistics
	
	statisticsLT.objval(k) = LThistory.objval(length(LThistory.objval)); %objective function values
	statisticsADMM.objval(k) = history.objval(length(history.objval));  %objective function values
	
	statisticsLT.r_norm(k) = LThistory.r_norm(length(LThistory.r_norm));
	statisticsADMM.r_norm(k) = history.r_norm(length(history.r_norm));
	
	statisticsLT.iterates(k) = length(LThistory.objval);    %number of iterates it took LT to solve
	statisticsADMM.iterates(k) = length(history.objval);    %number of iterates it took ADMM to solve
    
    if length(LThistory.objval) < length(history.objval)
        statisticsLT.wins(k) = 1;
        statisticsADMM.wins(k) = 0;
    elseif length(LThistory.objval) > length(history.objval)
        statisticsADMM.wins(k) = 1;
        statisticsLT.wins(k) = 0;
    else
        statisticsLT.wins(k) = 0;
        statisticsADMM.wins(k) = 0;
    end
    
    if length(history.objval) == 50000
        statisticsADMM.fails(k) = 1;
        statisticsLT.LTvsADMM(k) = 0; %I record the difference as a zero if ADMM fails to solve
    else
        statisticsADMM.fails(k) = 0;
        statisticsLT.LTvsADMM(k) = norm(x-LTx); %I record the difference between LT solutions and ADMM solutions when they both solved the problem
    end
    
    if length(LThistory.objval) == 50000
        statisticsLT.fails(k) = 1;
    else
        statisticsLT.fails(k) = 0;
    end
    
	
end	
	
	

