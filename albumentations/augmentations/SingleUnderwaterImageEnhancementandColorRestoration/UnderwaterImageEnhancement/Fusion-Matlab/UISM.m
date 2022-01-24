function uism = UISM(img)
Ir = double(img(:,:,1));
Ig = double(img(:,:,2));
Ib = double(img(:,:,3));

hx=[1 2 1; 0 0 0 ; -1 -2 -1];%生产sobel垂直梯度模板  
hy=[-1 0 1; -2 0 2; -1 0 1];    

SobelR = abs(imfilter(Ir, hx, 'replicate', 'same', 'conv') + ...
            imfilter(Ir, hy, 'replicate', 'same', 'conv'));
SobelG = abs(imfilter(Ig, hx, 'replicate', 'same', 'conv') + ...
            imfilter(Ig, hy, 'replicate', 'same', 'conv'));
SobelB = abs(imfilter(Ib, hx, 'replicate', 'same', 'conv') + ...
            imfilter(Ib, hy, 'replicate', 'same', 'conv'));

patchsz = 5;
[m, n] = size(Ir);
% resize the input image to match the patch size
if mod(m, patchsz) ~= 0 || mod(n, patchsz) ~= 0
    SobelR = imresize(SobelR, [m - mod(m, patchsz) + patchsz, ...
        n - mod(n, patchsz) + patchsz]);
    SobelG = imresize(SobelG, [m - mod(m, patchsz) + patchsz, ...
        n - mod(n, patchsz) + patchsz]);
    SobelB = imresize(SobelB, [m - mod(m, patchsz) + patchsz, ...
        n - mod(n, patchsz) + patchsz]);
end
[m, n] = size(Ir);
k1 = m / patchsz;
k2 = n / patchsz;
% calculate the EME value
EMER = 0;
for i = 1 : patchsz : m
    for j = 1 : patchsz : n
        sz = patchsz - 1;
        im = SobelR(i:i+sz,j:j+sz);
        if (max(max(im)) ~= 0 && min(min(im)) ~= 0)
            EMER = EMER + log(max(max(im)) / min(min(im))); 
        end
    end
end
EMER = 2 / (k1 * k2) * abs(EMER);
EMEG = 0;
for i = 1 : patchsz : m
    for j = 1 : patchsz : n
        sz = patchsz - 1;
        im = SobelG(i:i+sz,j:j+sz);
        if (max(max(im)) ~= 0 && min(min(im)) ~= 0)
            EMEG = EMEG + log(max(max(im)) / min(min(im))); 
        end
    end
end
EMEG = 2 / (k1 * k2) * abs(EMEG);
EMEB = 0;
for i = 1 : patchsz : m
    for j = 1 : patchsz : n
        sz = patchsz - 1;
        im = SobelB(i:i+sz,j:j+sz);
        if (max(max(im)) ~= 0 && min(min(im)) ~= 0)
            EMEB = EMEB + log(max(max(im)) / min(min(im))); 
        end
    end
end
EMEB = 2 / (k1 * k2) * abs(EMEB);
lambdaR = 0.299;
lambdaG = 0.587;
lambdaB = 0.114;
uism = lambdaR * EMER + lambdaG * EMEG + lambdaB * EMEB;