%% Exercise 10 - Part 2: Matrix Form Representation and Mixed Equilibrium
% This MATLAB script computes the strategic form of the multi-stage game
% and finds the pure security levels and mixed saddle-point equilibrium

clear; clc;

fprintf('================================================================================\n');
fprintf('EXERCISE 10 - PART 2: MATRIX FORM AND MIXED EQUILIBRIUM\n');
fprintf('================================================================================\n\n');

%% Step 1: Generate all pure strategies for both players

% P1 has 5 decision nodes: {α1, α2, α3, α4, α5}
% α1: T=0, B=1
% α2, α3, α4, α5: L=0, R=1

% P2 has 6 information sets: {β1, β2, β3, β4, β5, β6}
% β1: T=0, B=1
% β2, β3, β4, β5, β6: L=0, R=1

n_p1_nodes = 5;
n_p2_nodes = 6;

n_p1_strategies = 2^n_p1_nodes;  % 32 strategies
n_p2_strategies = 2^n_p2_nodes;  % 64 strategies

fprintf('Pure Strategy Spaces:\n');
fprintf('  P1 has %d decision nodes, giving 2^%d = %d pure strategies\n', ...
        n_p1_nodes, n_p1_nodes, n_p1_strategies);
fprintf('  P2 has %d information sets, giving 2^%d = %d pure strategies\n', ...
        n_p2_nodes, n_p2_nodes, n_p2_strategies);
fprintf('  Strategic form matrix: %d x %d\n\n', n_p1_strategies, n_p2_strategies);

% Generate all pure strategies (as binary vectors)
p1_strategies = zeros(n_p1_strategies, n_p1_nodes);
p2_strategies = zeros(n_p2_strategies, n_p2_nodes);

for i = 1:n_p1_strategies
    binary = dec2bin(i-1, n_p1_nodes);
    for j = 1:n_p1_nodes
        p1_strategies(i,j) = str2double(binary(j));
    end
end

for i = 1:n_p2_strategies
    binary = dec2bin(i-1, n_p2_nodes);
    for j = 1:n_p2_nodes
        p2_strategies(i,j) = str2double(binary(j));
    end
end

%% Step 2: Compute the payoff matrix

fprintf('Computing payoff matrix...\n');

A = zeros(n_p1_strategies, n_p2_strategies);

for i = 1:n_p1_strategies
    for j = 1:n_p2_strategies
        A(i,j) = compute_payoff(p1_strategies(i,:), p2_strategies(j,:));
    end
    if mod(i, 8) == 0
        fprintf('  Progress: %d/%d rows completed\n', i, n_p1_strategies);
    end
end

fprintf('Payoff matrix computed: %d x %d\n\n', size(A,1), size(A,2));

%% Step 3: Compute pure security levels

fprintf('================================================================================\n');
fprintf('PURE SECURITY LEVELS\n');
fprintf('================================================================================\n\n');

% P1 (minimizer) security level
p1_row_mins = min(A, [], 2);
[p1_security_value, p1_security_idx] = max(p1_row_mins);

fprintf('P1''s Pure Security Level:\n');
fprintf('  Value: %.4f\n', p1_security_value);
fprintf('  Strategy %d: ', p1_security_idx);
print_strategy(p1_strategies(p1_security_idx,:), 'P1');
fprintf('\n\n');

% P2 (maximizer) security level
p2_col_maxs = max(A, [], 1);
[p2_security_value, p2_security_idx] = min(p2_col_maxs);

fprintf('P2''s Pure Security Level:\n');
fprintf('  Value: %.4f\n', p2_security_value);
fprintf('  Strategy %d: ', p2_security_idx);
print_strategy(p2_strategies(p2_security_idx,:), 'P2');
fprintf('\n\n');

% Check for pure strategy saddle points
if abs(p1_security_value - p2_security_value) < 1e-6
    fprintf('✓ Pure strategy saddle points exist!\n');
    fprintf('  Common value = %.4f\n\n', p1_security_value);
    
    % Count saddle points
    saddle_count = 0;
    saddle_list = [];
    
    for i = 1:n_p1_strategies
        for j = 1:n_p2_strategies
            is_row_min = (A(i,j) <= min(A(i,:)) + 1e-9);
            is_col_max = (A(i,j) >= max(A(:,j)) - 1e-9);
            
            if is_row_min && is_col_max
                saddle_count = saddle_count + 1;
                saddle_list = [saddle_list; i, j, A(i,j)];
            end
        end
    end
    
    fprintf('  Total saddle points found: %d\n', saddle_count);
    
    if saddle_count <= 10
        fprintf('  All saddle points:\n');
        for k = 1:saddle_count
            fprintf('    P1=%d, P2=%d, value=%.2f\n', ...
                    saddle_list(k,1), saddle_list(k,2), saddle_list(k,3));
        end
    else
        fprintf('  First 5 saddle points:\n');
        for k = 1:5
            i = saddle_list(k,1);
            j = saddle_list(k,2);
            fprintf('    P1=%d ', i);
            print_strategy(p1_strategies(i,:), 'P1');
            fprintf(', P2=%d ', j);
            print_strategy(p2_strategies(j,:), 'P2');
            fprintf('\n');
        end
    end
    fprintf('\n');
end

%% Step 4: Solve for mixed equilibrium

fprintf('================================================================================\n');
fprintf('MIXED SADDLE-POINT EQUILIBRIUM\n');
fprintf('================================================================================\n\n');

m = size(A, 1);
n = size(A, 2);

% Solve for P1 (minimizer)
fprintf('Solving for P1''s mixed strategy...\n');

f = [zeros(m, 1); 1];
A_ineq = [A', -ones(n, 1)];
b_ineq = zeros(n, 1);
A_eq = [ones(1, m), 0];
b_eq = 1;
lb = [zeros(m, 1); -inf];
ub = [inf(m, 1); inf];

options = optimoptions('linprog', 'Display', 'off', 'Algorithm', 'dual-simplex');
[sol_p1, ~, exitflag_p1] = linprog(f, A_ineq, b_ineq, A_eq, b_eq, lb, ub, options);

if exitflag_p1 > 0
    x_p1 = sol_p1(1:m);
    v_p1 = sol_p1(m+1);
    
    fprintf('  Game value: %.6f\n', v_p1);
    
    support_p1 = find(x_p1 > 1e-6);
    fprintf('  Support size: %d\n', length(support_p1));
    
    if length(support_p1) <= 5
        fprintf('  Strategies in support:\n');
        for k = 1:length(support_p1)
            idx = support_p1(k);
            fprintf('    %d (%.4f): ', idx, x_p1(idx));
            print_strategy(p1_strategies(idx,:), 'P1');
            fprintf('\n');
        end
    end
else
    fprintf('  ERROR: Solver failed\n');
    v_p1 = NaN;
end

fprintf('\n');

% Solve for P2 (maximizer)
fprintf('Solving for P2''s mixed strategy...\n');

f = [zeros(n, 1); -1];
A_ineq = [-A, ones(m, 1)];
b_ineq = zeros(m, 1);
A_eq = [ones(1, n), 0];
b_eq = 1;
lb = [zeros(n, 1); -inf];
ub = [inf(n, 1); inf];

[sol_p2, ~, exitflag_p2] = linprog(f, A_ineq, b_ineq, A_eq, b_eq, lb, ub, options);

if exitflag_p2 > 0
    y_p2 = sol_p2(1:n);
    v_p2 = sol_p2(n+1);
    
    fprintf('  Game value: %.6f\n', v_p2);
    
    support_p2 = find(y_p2 > 1e-6);
    fprintf('  Support size: %d\n', length(support_p2));
    
    if length(support_p2) <= 5
        fprintf('  Strategies in support:\n');
        for k = 1:length(support_p2)
            idx = support_p2(k);
            fprintf('    %d (%.4f): ', idx, y_p2(idx));
            print_strategy(p2_strategies(idx,:), 'P2');
            fprintf('\n');
        end
    end
else
    fprintf('  ERROR: Solver failed\n');
    v_p2 = NaN;
end

fprintf('\n');

%% Summary

fprintf('================================================================================\n');
fprintf('SUMMARY\n');
fprintf('================================================================================\n\n');

fprintf('Pure Security Levels: %.4f (both players)\n', p1_security_value);
fprintf('Mixed Equilibrium Value: %.6f\n', v_p1);
fprintf('\n');

if abs(v_p1 - v_p2) < 1e-4
    fprintf('✓ P1 and P2 solutions match\n');
end

behavioral_value = -1;
if abs(behavioral_value - v_p1) < 1e-4
    fprintf('✓ Consistent with behavioral equilibrium value = %.4f\n', behavioral_value);
end

fprintf('\n');
fprintf('================================================================================\n');

%% Helper Functions

function payoff = compute_payoff(p1_strat, p2_strat)
    if p1_strat(1) == 0  % α1 = T
        if p2_strat(1) == 0  % β1 = T
            if p1_strat(2) == 0  % α2 = L
                payoff = (p2_strat(2) == 0) * 1 + (p2_strat(2) == 1) * (-1);
            else  % α2 = R
                payoff = (p2_strat(3) == 0) * 1 + (p2_strat(3) == 1) * (-1);
            end
        else  % β1 = B
            if p1_strat(3) == 0  % α3 = L
                payoff = (p2_strat(4) == 0) * (-2) + (p2_strat(4) == 1) * (-1);
            else  % α3 = R
                payoff = (p2_strat(4) == 0) * 1 + (p2_strat(4) == 1) * (-1);
            end
        end
    else  % α1 = B
        if p2_strat(1) == 0  % β1 = T
            if p1_strat(4) == 0  % α4 = L
                payoff = (p2_strat(5) == 0) * (-1) + (p2_strat(5) == 1) * (-3);
            else  % α4 = R
                payoff = (p2_strat(5) == 0) * (-3) + (p2_strat(5) == 1) * (-1);
            end
        else  % β1 = B
            if p1_strat(5) == 0  % α5 = L
                payoff = (p2_strat(6) == 0) * (-1) + (p2_strat(6) == 1) * 2;
            else  % α5 = R
                payoff = (p2_strat(6) == 0) * (-1) + (p2_strat(6) == 1) * 2;
            end
        end
    end
end

function print_strategy(strat, player)
    if strcmp(player, 'P1')
        choices = {'T', 'B'; 'L', 'R'; 'L', 'R'; 'L', 'R'; 'L', 'R'};
    else
        choices = {'T', 'B'; 'L', 'R'; 'L', 'R'; 'L', 'R'; 'L', 'R'; 'L', 'R'};
    end
    
    fprintf('(');
    for i = 1:length(strat)
        fprintf('%s', choices{i, strat(i)+1});
        if i < length(strat)
            fprintf(',');
        end
    end
    fprintf(')');
end
