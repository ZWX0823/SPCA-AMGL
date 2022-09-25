close all
clear
load Indian_pines_corrected.mat;
data = indian_pines_corrected;
clear indian_pines_corrected;

load Indian_pines_gt.mat;
label = indian_pines_gt;
clear indian_pines_gt;

%% preprocess
w = 20;
h = 20;
[X, y] = preprocess(data, label, w, h);
index = SPCA_AMGL(X, 1, 1, 1, 15);
