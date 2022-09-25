function alpha = optimize_alpha(S, Si)
    K = size(Si,3);
    c = zeros(K);
    for k=1:K
        SS = Si(:,:,k).*log(max((Si(:,:,k)./max(S,eps)),eps));
        c(k) = max(sum(sum(SS)),eps);
    end
    c = c.^(-1);
    alpha = c./sum(c);
end