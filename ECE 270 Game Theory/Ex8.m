A = [ 2  6 -2 10;
     -6 -4 -3 -6;
      0  4 -3 -8 ];

B = [ 0  1  2  3;
      1  0  1  2;
      0  1  0  1;
     -1  0  1  0 ];

games = {A, B};
names = {'Game A', 'Game B'};

options = optimoptions('linprog', 'Display', 'off');

for k = 1:length(games)
    M = games{k};
    [m, n] = size(M);

    % LP for Player 1 (minimizer):
    % min v  s.t.  M'*p <= v*1,  sum(p)=1,  p >= 0
    f = [zeros(m, 1); 1];
    Aineq = [M', -ones(n, 1)];
    bineq = zeros(n, 1);
    Aeq = [ones(1, m), 0];
    beq = 1;
    lb = [zeros(m, 1); -Inf];

    x = linprog(f, Aineq, bineq, Aeq, beq, lb, [], [], options);

    p_star = x(1:m);
    v_star = x(end);

    % Display results
    fprintf('\n=== %s ===\n', names{k});
    fprintf('Mixed security value V: %.4f\n', v_star);
    fprintf('Mixed security policy p*:\n');
    for i = 1:m
        fprintf('  Row %d: %.4f\n', i, p_star(i));
    end
end