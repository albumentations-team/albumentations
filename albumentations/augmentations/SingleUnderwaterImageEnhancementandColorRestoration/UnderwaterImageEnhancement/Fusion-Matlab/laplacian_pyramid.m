function out = laplacian_pyramid(img, level)
h = 1/16* [1, 4, 6, 4, 1];
%filt = h'*h;
out{1} = img;
temp_img = img;
for i = 2 : level
    temp_img = temp_img(1 : 2 : end, 1 : 2 : end);
    %out{i} = imfilter(temp_img, filt, 'replicate', 'conv');
    out{i} = temp_img;
end
% calculate the DoG
for i = 1 : level - 1
    [m, n] = size(out{i});
    newOUt = imresize(out{i+1}, [m, n]);
    out{i} = out{i} - imresize(out{i+1}, [m, n]);
end