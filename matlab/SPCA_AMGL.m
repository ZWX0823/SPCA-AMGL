function index = SPCA_AMGL(X, lambda1, lambda2, lambda3, d)

[n, b] = size(X);
n_neighbors = 6;

%% Call the generate_graphs function in Python
Si = py.generate_graphs.generate_graphs(X, n_neighbors);

K = size(Si, 3);

% Initial W
W = orth(rand(b, d));

% Initial S
S = sum(Si, 3);

% Initial multigraph parameters
alpha_k = ones([1, K]) / K;

iteration = 0;
iter_max = 6;
while iteration < iter_max
    % update W
    W = Optimize_W(X, W, S, lambda1, lambda2);
    % update S
    S = Optimize_S(X, W, Si, lambda2, lambda3, alpha_k);
    % update alpha_k
    alpha_k = Optimize_alpha_k(S, Si);
end

%% Sort the bands, return the index of top d bands
score = sum(W.*W,2);
[~,index] = sort(score);
index = index(end-d+1:end);
end
