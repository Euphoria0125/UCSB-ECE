clear; clc;

A = [5 3 0 0 7 0 0 0 0;
     6 0 0 1 9 5 0 0 0;
     0 9 8 0 0 0 0 6 0;
     8 0 0 0 6 0 0 0 3;
     4 0 0 8 0 3 0 0 1;
     7 0 0 0 2 0 0 0 6;
     0 6 0 0 0 0 2 8 0;
     0 0 0 4 1 9 0 0 5;
     0 0 0 0 8 0 0 7 9];

[m,n] = size(A);

f = [zeros(m,1); -1];

Aineq = [-A', ones(n,1)];
bineq = zeros(n,1);

Aeq = [ones(1,m), 0];
beq = 1;

lb = [zeros(m,1); -inf];

options = optimoptions('linprog','Display','none');

[x, fval] = linprog(f, Aineq, bineq, Aeq, beq, lb, [], options);

p = x(1:m);
v = x(end);

fprintf('Mixed security value:\n');
disp(v)

fprintf('Mixed strategy:\n');
disp(p)

fprintf('Nonzero probabilities:\n');
disp(find(p > 1e-6))
