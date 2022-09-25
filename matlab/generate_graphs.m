function generate_graphs(data, n_neighbors)

% the number of similarity graphs
K = 3;
[b, n] = size(data);
% permute to (n_sample, n_feature)
data = permute(data, [2,1]);

S1 = cosineSimilarity(data)
S2 = 

end