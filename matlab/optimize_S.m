function S = optimize_S(X, W, Si, lambda2, lambda3, alpha)

[n,b] = size(X);
K = length(alpha);

% Calcuate matrix B
B = zeros(n,n);
for i = 1:n
    for j = 1:n
        if i>j
            B(i,j) = norm((X(i,:) - X(j,:)) * W);
            B(j,i) = B(i,j);
        end
    end
end
% Multiply factor in advance
B = (lambda2/2).*B;

% Calcuate matrix C
C = zeros(n,n);
for k = 1:K
    C = C + alpha(k)*alpha(k)*Si(:,:,k);
end
% Multiply factor in advance
C = lambda3.*C;

S = zeros(n,n);
for i = 1:n
    index_nonzero = find(C(i,:));
    try
        lambda = fzero(@(x)fun(x, B(i,index_nonzero), C(i, index_nonzero)), [0+eps,10000]);
    catch
        lambda = fzero(@(x)fun(x, B(i,index_nonzero), C(i, index_nonzero)), [0+0.01*eps,10000]);
    end
    S(i,index_nonzero) = C(i,index_nonzero)./(B(i,index_nonzero)+lambda);
end
end

