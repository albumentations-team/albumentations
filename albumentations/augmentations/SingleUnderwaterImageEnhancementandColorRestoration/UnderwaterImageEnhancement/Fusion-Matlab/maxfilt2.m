function Y = maxfilt2(X,varargin)
%  MAXFILT2    Two-dimensional max filter
%
%     Y = MAXFILT2(X,[M N]) performs two-dimensional maximum
%     filtering on the image X using an M-by-N window. The result
%     Y contains the maximun value in the M-by-N neighborhood around
%     each pixel in the original image. 
%     This function uses the van Herk algorithm for max filters.
%
%     Y = MAXFILT2(X,M) is the same as Y = MAXFILT2(X,[M M])
%
%     Y = MAXFILT2(X) uses a 3-by-3 neighborhood.
%
%     Y = MAXFILT2(..., 'shape') returns a subsection of the 2D
%     filtering specified by 'shape' :
%        'full'  - Returns the full filtering result,
%        'same'  - (default) Returns the central filter area that is the
%                   same size as X,
%        'valid' - Returns only the area where no filter elements are outside
%                  the image.
%
%     See also : MINFILT2, VANHERK
%

% Initialization
[S, shape] = parse_inputs(varargin{:});

% filtering
Y = vanherk(X,S(1),'max',shape);
Y = vanherk(Y,S(2),'max','col',shape);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [S, shape] = parse_inputs(varargin)
shape = 'same';
flag = [0 0]; % size shape

for i = 1 : nargin
   t = varargin{i};
   if strcmp(t,'full') & flag(2) == 0
      shape = 'full';
      flag(2) = 1;
   elseif strcmp(t,'same') & flag(2) == 0
      shape = 'same';
      flag(2) = 1;
   elseif strcmp(t,'valid') & flag(2) == 0
      shape = 'valid';
      flag(2) = 1;
   elseif flag(1) == 0
      S = t;
      flag(1) = 1;
   else
      error(['Too many / Unkown parameter : ' t ])
   end
end

if flag(1) == 0
   S = [3 3];
end
if length(S) == 1;
   S(2) = S(1);
end
if length(S) ~= 2
   error('Wrong window size parameter.')
end

