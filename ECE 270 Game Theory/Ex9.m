% Exercise 9: Resistor Design Game - Complete Solution

clc; clear; close all;

%% 1. Game Matrix Construction
R_nom = 0.70:0.05:1.20;      % 11 values, 0.05¦¸ spacing
delta = -0.10:0.01:0.10;     % 21 values, 1% spacing (CORRECTED from 0.02 to 0.01)

m = length(R_nom);
n = length(delta);

A = zeros(m, n);
for i = 1:m
    for j = 1:n
        R_actual = R_nom(i) * (1 + delta(j));
        A(i, j) = abs(1 - 1/R_actual);
    end
end

fprintf('=== Resistor Design Game ===\n');
fprintf('Matrix size: %d (R_nom choices) ¡Á %d (delta choices)\n\n', m, n);

%% 2. Pure Security Analysis (from friend's code, adapted)
% P1 (Designer, minimizer): min max
max_per_row = max(A, [], 2);
[V_pure, best_row_idx] = min(max_per_row);

% P2 (Opponent, maximizer): max min  
min_per_col = min(A, [], 1);
[maximin_val, best_col_idx] = max(min_per_col);

fprintf('--- Pure Strategy Analysis ---\n');
fprintf('Designer (P1, minimizer):\n');
fprintf('  Security level (minimax): %.6f\n', V_pure);
fprintf('  Optimal pure strategy: R_nom = %.2f ¦¸\n', R_nom(best_row_idx));

fprintf('\nOpponent (P2, maximizer):\n');
fprintf('  Security level (maximin): %.6f\n', maximin_val);
fprintf('  Optimal pure strategy: delta = %.0f%%\n', delta(best_col_idx)*100);

%% 3. Check for Pure Saddle Point (from friend's code)
tolerance = 1e-6;
if abs(V_pure - maximin_val) < tolerance
    fprintf('\n=== RESULT: Pure Saddle Point Exists ===\n');
    fprintf('Game value V = %.6f\n', V_pure);
    fprintf('Optimal strategies are pure (no randomness needed).\n');
else
    fprintf('\n=== RESULT: No Pure Saddle Point ===\n');
    fprintf('Proceeding to mixed strategy calculation...\n\n');
    
    %% 4. Mixed Security via LP (YOUR corrected formulation)
    % P1's problem: min v, s.t. A'*p <= v*1, sum(p)=1, p>=0
    f = [zeros(m, 1); 1];            % minimize v
    Aineq = [A', -ones(n, 1)];       % A'*p - v <= 0
    bineq = zeros(n, 1);
    Aeq = [ones(1, m), 0];          % sum(p) = 1
    beq = 1;
    lb = [zeros(m, 1); -Inf];       % p >= 0, v unrestricted
    
    options = optimoptions('linprog', 'Display', 'off');
    x = linprog(f, Aineq, bineq, Aeq, beq, lb, [], [], options);
    
    p_star = x(1:m);      % Optimal mixed strategy for P1
    v_star = x(end);      % Mixed security value
    
    %% 5. Results Display
    fprintf('--- Mixed Strategy Analysis ---\n');
    fprintf('Mixed security value: %.6f\n', v_star);
    
    % Display significant probabilities
    tol = 1e-4;
    nonzero_idx = find(p_star > tol);
    
    if isempty(nonzero_idx)
        fprintf('Optimal mixed strategy is pure.\n');
    else
        fprintf('\nOptimal mixed strategy for Designer (nonzero probabilities):\n');
        fprintf('R_nom (¦¸)   Probability\n');
        fprintf('--------   -----------\n');
        for k = 1:length(nonzero_idx)
            i = nonzero_idx(k);
            fprintf('   %.2f       %.4f\n', R_nom(i), p_star(i));
        end
    end
    
    %% 6. Verification (optional but good practice)
    fprintf('\n--- Verification ---\n');
    expected_payoffs = A' * p_star;
    fprintf('Expected payoff against each delta:\n');
    for j = 1:min(5, n)  % Show first 5 for brevity
        fprintf('  delta = %4.1f%%: %.6f\n', delta(j)*100, expected_payoffs(j));
    end
    if n > 5
        fprintf('  ... (and %d more)\n', n-5);
    end
    
    max_expected = max(expected_payoffs);
    fprintf('\nMaximum expected payoff (worst-case): %.6f\n', max_expected);
    fprintf('This should equal mixed security value: %.6f\n', v_star);
    
    %% 7. Improvement Analysis
    improvement = V_pure - v_star;
    improvement_percent = (improvement / V_pure) * 100;
    
    fprintf('\n--- Improvement Analysis ---\n');
    fprintf('Pure strategy value:    %.6f\n', V_pure);
    fprintf('Mixed strategy value:   %.6f\n', v_star);
    fprintf('Absolute improvement:   %.6f\n', improvement);
    fprintf('Relative improvement:   %.2f%%\n', improvement_percent);
end

%% Helper function (from friend's code, but using correct formulation)
function [p_opt, v_opt] = solve_mixed_strategy(A)
    % Correct LP formulation for minimizer (P1)
    % min v, s.t. A'*p <= v*1, sum(p)=1, p>=0
    [m, n] = size(A);
    
    f = [zeros(m, 1); 1];
    Aineq = [A', -ones(n, 1)];
    bineq = zeros(n, 1);
    Aeq = [ones(1, m), 0];
    beq = 1;
    lb = [zeros(m, 1); -Inf];
    
    options = optimoptions('linprog', 'Display', 'off');
    x = linprog(f, Aineq, bineq, Aeq, beq, lb, [], [], options);
    
    p_opt = x(1:m);
    v_opt = x(end);
end