function uiconm = UIConM(img)
R = double(img(:,:,1));
G = double(img(:,:,2));
B = double(img(:,:,3));

patchsz = 5;
[m, n] = size(R);
% resize the input image to match the patch size
if mod(m, patchsz) ~= 0 || mod(n, patchsz) ~= 0
    R = imresize(R, [m - mod(m, patchsz) + patchsz, ...
        n - mod(n, patchsz) + patchsz]);
    G = imresize(G, [m - mod(m, patchsz) + patchsz, ...
        n - mod(n, patchsz) + patchsz]);
    B = imresize(B, [m - mod(m, patchsz) + patchsz, ...
        n - mod(n, patchsz) + patchsz]);
end
[m, n] = size(R);
k1 = m / patchsz;
k2 = n / patchsz;

AMEER = 0;
for i = 1 : patchsz : m
    for j = 1 : patchsz : n
        sz = patchsz - 1;
        im = R(i:i+sz,j:j+sz);
        Max = max(max(im));
        Min = min(min(im));
        if ( (Max ~= 0 || Min ~= 0) && Max ~= Min )
            AMEER = AMEER + ...
                log( (Max - Min) / (Max + Min) ) * ...
                ( (Max - Min) / (Max + Min) ); 
        end
    end
end
AMEER = 1 / (k1 * k2) * abs(AMEER);
AMEEG = 0;
for i = 1 : patchsz : m
    for j = 1 : patchsz : n
        sz = patchsz - 1;
        im = G(i:i+sz,j:j+sz);
        Max = max(max(im));
        Min = min(min(im));
        if ( (Max ~= 0 || Min ~= 0) && Max ~= Min )
            AMEEG = AMEEG + ...
                log( (Max - Min) / (Max + Min) ) * ...
                ( (Max - Min) / (Max + Min) ); 
        end
    end
end
AMEEG = 1 / (k1 * k2) * abs(AMEEG);
AMEEB = 0;
for i = 1 : patchsz : m
    for j = 1 : patchsz : n
        sz = patchsz - 1;
        im = B(i:i+sz,j:j+sz);
        Max = max(max(im));
        Min = min(min(im));
        if ( (Max ~= 0 || Min ~= 0) && Max ~= Min )
            AMEEB = AMEEB + ...
                log( (Max - Min) / (Max + Min) ) * ...
                ( (Max - Min) / (Max + Min) ); 
        end
    end
end
AMEEB = 1 / (k1 * k2) * abs(AMEEB);
uiconm = AMEER + AMEEG + AMEEB;