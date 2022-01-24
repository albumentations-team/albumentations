function [meanRG, deltaRG, meanYB, deltaYB, uicm] = UICM(img)
R = double(img(:,:,1));
G = double(img(:,:,2));
B = double(img(:,:,3));
RG = R - G;
YB = (R + G) / 2 - B;

K = size(R,1) * size(R,2);

% for R-G channel
RG1 = reshape(RG, 1, K);
RG1 = sort(RG1);
alphaL = 0.1;
alphaR = 0.1;
RG1 = RG1(1, int32(alphaL*K+1) : int32(K*(1-alphaR)));
N = K * (1 - alphaL - alphaR);
meanRG = sum(RG1) / N;
deltaRG = sqrt(sum((RG1 - meanRG).^2) / N);

% for Y-B channel
YB1 = reshape(YB, 1, K);
YB1 = sort(YB1);
alphaL = 0.1;
alphaR = 0.1;
YB1 = YB1(1, int32(alphaL*K+1) : int32(K*(1-alphaR)));
N = K * (1 - alphaL - alphaR);
meanYB = sum(YB1) / N;
deltaYB = sqrt(sum((YB1 - meanYB).^2) / N);

% UICM
uicm = -0.0268 * sqrt(meanRG^2 + meanYB^2) + ...
    0.1586* sqrt(deltaRG^2 + deltaYB^2);