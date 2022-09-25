function index = SPCA_AMGL(X, lambda1, lambda2, lambda3, d)

[n, b] = size(X);
n_neighbors = 6;

%% Call the generate_graphs function in Python
Si = pyrunfile("generate_graphs.py", "Si", a = X, b = n_neighbors);
% Convert Python data type back to Matlab type
Si = double(Si);

K = size(Si, 3);

% Initial W
W = orth(rand(b, d));

% Initial S
S = sum(Si, 3);

% Initial multigraph parameters
alpha = ones([1, K]) / K;

iteration = 0;
iter_max = 6;
while iteration < iter_max
    % update W
    W = optimize_W(X, W, S, lambda1, lambda2);
    % update S
    S = optimize_S(X, W, Si, lambda2, lambda3, alpha);
    % update alpha
    alpha = optimize_alpha(S, Si);
    iteration = iteration + 1;
end

%% Sort the bands, return the index of top d bands
score = sum(W.*W,2);
[~,index] = sort(score);
index = index(end-d+1:end);
end
