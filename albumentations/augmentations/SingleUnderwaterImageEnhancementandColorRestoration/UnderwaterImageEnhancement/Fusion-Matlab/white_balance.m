%my own white-balance function, created by Qu Jingwei
function new_image = white_balance3(src_image)
[height,width,dim] = size(src_image);
temp = zeros(height,width);
%transform the RGB color space to YCbCr color space 
ycbcr_image = rgb2ycbcr(src_image);
Y = ycbcr_image(:,:,1);
Cb = ycbcr_image(:,:,2);
Cr = ycbcr_image(:,:,3);
%calculate the average value of Cb,Cr
Cb_ave = mean(mean(Cb));
Cr_ave = mean(mean(Cr));
%calculate the mean square error of Cb, Cr
Db = sum(sum(abs(Cb-Cb_ave))) / (height*width);
Dr = sum(sum(abs(Cr-Cr_ave))) / (height*width);
%find the candidate reference white point
%if meeting the following requriments
%then the point is a candidate reference white point
temp1 = abs(Cb - (Cb_ave + Db * sign(Cb_ave)));
temp2 = abs(Cb - (1.5 * Cr_ave + Dr * sign(Cr_ave)));
idx_1 = find(temp1<1.5*Db);
idx_2 = find(temp2<1.5*Dr);
idx = intersect(idx_1,idx_2);
point = Y(idx);
temp(idx) = Y(idx);
count = length(point);
count = count - 1;
%sort the candidate reference white point set with descend value of Y
temp_point = sort(point,'descend');
%get the 10% points of the candidate reference white point set, which is
%closer to the white region, as the reference white point set
n = round(count/10);
white_point(1:n) = temp_point(1:n);
temp_min = min(white_point);
idx0 = find(temp<temp_min);
temp(idx0) = 0;
idx1 = find(temp>=temp_min);
temp(idx1) = 1;
%get the reference white points' R,G,B
white_R = double(src_image(:,:,1)).*temp;
white_G = double(src_image(:,:,2)).*temp;
white_B = double(src_image(:,:,3)).*temp;
%get the averange value of the reference white points' R,G,B
white_R_ave = mean(mean(white_R));
white_G_ave = mean(mean(white_G));
white_B_ave = mean(mean(white_B));
%the maximum Y value of the source image
Ymax = double(max(max(Y))) / 15;
%calculate the white-balance gain
R_gain = Ymax / white_R_ave;
G_gain = Ymax / white_G_ave;
B_gain = Ymax / white_B_ave;
%white-balance correction
new_image(:,:,1) = R_gain * src_image(:,:,1);
new_image(:,:,2) = G_gain * src_image(:,:,2);
new_image(:,:,3) = B_gain * src_image(:,:,3);
new_image = uint8(new_image);
end