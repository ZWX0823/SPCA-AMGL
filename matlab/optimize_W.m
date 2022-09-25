function W = optimize_W(X, W, S, lambda1, lambda2)

[n, b] = size(X);
D = 0.5 * (diag(sum(S,1)) + diag(sum(S,2)));
L = D - 0.5 * (S + S');
Q = zeros(b,b);
for i = 1:b
    Q(i,i) = 1/norm(W(i,:));
end
A = -(X' * X) + lambda1 * Q + lambda2 * (X' * L * X);

ww = rand(b, 1);
for i = 1:10
    m1 = A * ww;
    q = m1 / norm(m1);
    ww = q;
end
lambda1_A = abs(ww' * A * ww);

t = 0;
t_max = 15;
A_til = lambda1_A.*eye(b) - A;
while t < t_max
    M = A_til * W;
    [U, ~, V] = svd(M, 'econ');
    W = U * V';
    t=t+1;
end
