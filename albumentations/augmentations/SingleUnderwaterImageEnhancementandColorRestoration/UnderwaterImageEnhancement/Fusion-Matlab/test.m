num = 8; %1~13
img = load_image('8');

% white balance
img1 = SimplestColorBalance(img);
lab1 = rgb_to_lab(img1);
%figure,imshow(img1)

% CLAHE
lab2 = lab1;
lab2(:, :, 1) = adapthisteq(lab2(:, :, 1));
lab2(:, :, 1) = uint8(bilateralFilter(double(lab2(:, :, 1))));
img2 = lab_to_rgb(lab2);
%figure,imshow(img2);

% input1
R1 = double(lab1(:, :, 1)) / 255;
% calculate laplacian contrast weight
WC1 = sqrt(((double(img1(:,:,1))/255 - double(R1)).^2 + ...
            (double(img1(:,:,2))/255 - double(R1)).^2 + ...
            (double(img1(:,:,3))/255 - double(R1)).^2) / 3);
% calculate the saliency weight
WS1 = saliency_detection(img1);
% calculate the exposedness weight
sigma = 0.25;
aver = 0.5;
WE1 = exp(-(R1 - aver).^2 / (2*sigma^2));
figure,imshow(WE1, [])

% input2
R2 = double(lab2(:, :, 1)) / 255;
% calculate laplacian contrast weight
WC2 = sqrt(((double(img2(:,:,1))/255 - double(R2)).^2 + ...
            (double(img2(:,:,2))/255 - double(R2)).^2 + ...
            (double(img2(:,:,3))/255 - double(R2)).^2) / 3);
%figure,imshow(WC2, [])
% calculate the saliency weight
WS2 = saliency_detection(img2);
%figure,imshow(WS2, [])
% calculate the exposedness weight
sigma = 0.3;
aver = 0.5;
WE2 = exp(-(R2 - aver).^2 / (2*sigma^2));
figure,imshow(WE2, [])


% calculate the normalized weight
W1 = (WC1 + WS1 + WE1) ./ ...
     (WC1 + WS1 + WE1 + WC2 + WS2 + WE2);
W2 = (WC2 + WS2 + WE2) ./ ...
     (WC1 + WS1 + WE1 + WC2 + WS2 + WE2);
figure,imshow([W1,W2])