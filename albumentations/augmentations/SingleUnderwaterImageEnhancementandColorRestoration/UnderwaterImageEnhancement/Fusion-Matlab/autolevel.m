function imDst = autolevel(varargin)
[I,lowCut,highCut] =parse_inputs(varargin{:});
[hei,wid,~] = size(I);

PixelAmount = wid * hei;
if size(I,3)==3

   [HistRed,~]  = imhist(I(:,:,1));
   [HistGreen,~] = imhist(I(:,:,2));
   [HistBlue,~] = imhist(I(:,:,3));

   CumRed = cumsum(HistRed);
   CumGreen = cumsum(HistGreen);
   CumBlue = cumsum(HistBlue);

   minR =find(CumRed>=PixelAmount*lowCut,1,'first');
   minG = find(CumGreen>=PixelAmount*lowCut,1,'first');
   minB =find(CumBlue>=PixelAmount*lowCut,1,'first');

   maxR =find(CumRed>=PixelAmount*(1-highCut),1,'first');
   maxG =find(CumGreen>=PixelAmount*(1-highCut),1,'first');
   maxB = find(CumBlue>=PixelAmount*(1-highCut),1,'first');

   RedMap = linearmap(minR,maxR);
   GreenMap = linearmap(minG,maxG);
   BlueMap = linearmap(minB,maxB);

   imDst = zeros(hei,wid,3,'uint8');
   imDst(:,:,1) = RedMap (I(:,:,1)+1);
   imDst(:,:,2) = GreenMap(I(:,:,2)+1);
   imDst(:,:,3) = BlueMap(I(:,:,3)+1);

else
   HistGray = imhist(I(:,:));
   CumGray = cumsum(HistRed);
   minGray =find(CumGray>=PixelAmount*lowCut,1,'first');
   maxGray =find(CumGray>=PixelAmount*(1-highCut),1,'first');
   GrayMap = linearmap(minGray,maxGray);

   imDst = zeros(hei,wid,'uint8');
   imDst(:,:) = GrayMap (I(:,:)+1); 
end

%--------------------------------------------------------------------
function map = linearmap(low,high)
map = [0:1:255];
for i=0:255
   if(i<low)
       map(i+1) = 0;
   elseif (i>high)
       map(i+1) = 255;
   else
       map(i+1) =uint8((i-low)/(high-low)*255);
   end
end


%-------------------------------------------------------------------
function [I,lowCut,highCut] = parse_inputs(varargin)
narginchk(1,3)
I = varargin{1};
validateattributes(I,{'double','logical','uint8','uint16','int16','single'},{},...
             mfilename,'Image',1);

if nargin == 1
   lowCut = 0.005;
   highCut = 0.005;
elseif nargin == 3
   lowCut = varargin{2};
   highCut = varargin{3};
else
   error(message('images:im2double:invalidIndexedImage','single, or logical.'));
end