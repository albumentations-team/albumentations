%
% output = bilateralFilter( data, edge, ...
%                          edgeMin, edgeMax, ...
%                          sigmaSpatial, sigmaRange, ...
%                          samplingSpatial, samplingRange )
%
% Bilateral and Cross-Bilateral Filter using the Bilateral Grid.
%
% Bilaterally filters the image 'data' using the edges in the image 'edge'.
% If 'data' == 'edge', then it the standard bilateral filter.
% Otherwise, it is the 'cross' or 'joint' bilateral filter.
% For convenience, you can also pass in [] for 'edge' for the normal
% bilateral filter.
%
% Note that for the cross bilateral filter, data does not need to be
% defined everywhere.  Undefined values can be set to 'NaN'.  However, edge
% *does* need to be defined everywhere.
%
% data and edge should be of the greyscale, double-precision floating point
% matrices of the same size (i.e. they should be [ height x width ])
%
% data is the only required argument
%
% edgeMin and edgeMax specifies the min and max values of 'edge' (or 'data'
% for the normal bilateral filter) and is useful when the input is in a
% range that's not between 0 and 1.  For instance, if you are filtering the
% L channel of an image that ranges between 0 and 100, set edgeMin to 0 and
% edgeMax to 100.
% 
% edgeMin defaults to min( edge( : ) ) and edgeMax defaults to max( edge(:)).
% This is probably *not* what you want, since the input may not span the
% entire range.
%
% sigmaSpatial and sigmaRange specifies the standard deviation of the space
% and range gaussians, respectively.
% sigmaSpatial defaults to min( width, height ) / 16
% sigmaRange defaults to ( edgeMax - edgeMin ) / 10.
%
% samplingSpatial and samplingRange specifies the amount of downsampling
% used for the approximation.  Higher values use less memory but are also
% less accurate.  The default and recommended values are:
% 
% samplingSpatial = sigmaSpatial
% samplingRange = sigmaRange
%

function output = bilateralFilter( data, edge, edgeMin, edgeMax,...
    sigmaSpatial, sigmaRange, samplingSpatial, samplingRange )

if( ndims( data ) > 2 ),
    error( 'data must be a greyscale image with size [ height, width ]' );
end

if( ~isa( data, 'double' ) ),
    error( 'data must be of class "double"' );
end

if ~exist( 'edge', 'var' ),
    edge = data;
elseif isempty( edge ),
    edge = data;
end

if( ndims( edge ) > 2 ),
    error( 'edge must be a greyscale image with size [ height, width ]' );
end

if( ~isa( edge, 'double' ) ),
    error( 'edge must be of class "double"' );
end

inputHeight = size( data, 1 );
inputWidth = size( data, 2 );

if ~exist( 'edgeMin', 'var' ),
    edgeMin = min( edge( : ) );
    %warning( 'edgeMin not set!  Defaulting to: %f\n', edgeMin );
end

if ~exist( 'edgeMax', 'var' ),
    edgeMax = max( edge( : ) );
    %warning( 'edgeMax not set!  Defaulting to: %f\n', edgeMax );
end

edgeDelta = edgeMax - edgeMin;

if ~exist( 'sigmaSpatial', 'var' ),
    sigmaSpatial = min( inputWidth, inputHeight ) / 16;
    %fprintf( 'Using default sigmaSpatial of: %f\n', sigmaSpatial );
end

if ~exist( 'sigmaRange', 'var' ),
    sigmaRange = 0.1 * edgeDelta;
    %fprintf( 'Using default sigmaRange of: %f\n', sigmaRange );
end

if ~exist( 'samplingSpatial', 'var' ),
    samplingSpatial = sigmaSpatial;
end

if ~exist( 'samplingRange', 'var' ),
    samplingRange = sigmaRange;
end

if size( data ) ~= size( edge ),
    error( 'data and edge must be of the same size' );
end

% parameters
derivedSigmaSpatial = sigmaSpatial / samplingSpatial;
derivedSigmaRange = sigmaRange / samplingRange;

paddingXY = floor( 2 * derivedSigmaSpatial ) + 1;
paddingZ = floor( 2 * derivedSigmaRange ) + 1;

% allocate 3D grid
downsampledWidth = floor( ( inputWidth - 1 ) / samplingSpatial )...
    + 1 + 2 * paddingXY;
downsampledHeight = floor( ( inputHeight - 1 ) / samplingSpatial )...
    + 1 + 2 * paddingXY;
downsampledDepth = floor( edgeDelta / samplingRange ) + 1 + 2 * paddingZ;

gridData = zeros( downsampledHeight, downsampledWidth, downsampledDepth );
gridWeights = zeros( downsampledHeight, downsampledWidth, downsampledDepth );

% compute downsampled indices
[ jj, ii ] = meshgrid( 0 : inputWidth - 1, 0 : inputHeight - 1 );

% ii =
% 0 0 0 0 0
% 1 1 1 1 1
% 2 2 2 2 2

% jj =
% 0 1 2 3 4
% 0 1 2 3 4
% 0 1 2 3 4

% so when iterating over ii( k ), jj( k )
% get: ( 0, 0 ), ( 1, 0 ), ( 2, 0 ), ... (down columns first)

di = round( ii / samplingSpatial ) + paddingXY + 1;
dj = round( jj / samplingSpatial ) + paddingXY + 1;
dz = round( ( edge - edgeMin ) / samplingRange ) + paddingZ + 1;

% perform scatter (there's probably a faster way than this)
% normally would do downsampledWeights( di, dj, dk ) = 1, but we have to
% perform a summation to do box downsampling
for k = 1 : numel( dz ),
       
    dataZ = data( k ); % traverses the image column wise, same as di( k )
    if ~isnan( dataZ  ),
        
        dik = di( k );
        djk = dj( k );
        dzk = dz( k );

        gridData( dik, djk, dzk ) = gridData( dik, djk, dzk ) + dataZ;
        gridWeights( dik, djk, dzk ) = gridWeights( dik, djk, dzk ) + 1;
        
    end
end

% make gaussian kernel
kernelWidth = 2 * derivedSigmaSpatial + 1;
kernelHeight = kernelWidth;
kernelDepth = 2 * derivedSigmaRange + 1;

halfKernelWidth = floor( kernelWidth / 2 );
halfKernelHeight = floor( kernelHeight / 2 );
halfKernelDepth = floor( kernelDepth / 2 );

[gridX, gridY, gridZ] = meshgrid( 0 : kernelWidth - 1,...
    0 : kernelHeight - 1, 0 : kernelDepth - 1 );
gridX = gridX - halfKernelWidth;
gridY = gridY - halfKernelHeight;
gridZ = gridZ - halfKernelDepth;
gridRSquared = ( gridX .* gridX + gridY .* gridY ) /...
    ( derivedSigmaSpatial * derivedSigmaSpatial ) +...
    ( gridZ .* gridZ ) / ( derivedSigmaRange * derivedSigmaRange );
kernel = exp( -0.5 * gridRSquared );

% convolve
blurredGridData = convn( gridData, kernel, 'same' );
blurredGridWeights = convn( gridWeights, kernel, 'same' );

% divide
% avoid divide by 0, won't read there anyway
blurredGridWeights( blurredGridWeights == 0 ) = -2; 
normalizedBlurredGrid = blurredGridData ./ blurredGridWeights;
% put 0s where it's undefined
normalizedBlurredGrid( blurredGridWeights < -1 ) = 0; 

% for debugging
% blurredGridWeights( blurredGridWeights < -1 ) = 0; % put zeros back

% upsample
% meshgrid does x, then y, so output arguments need to be reversed
[ jj, ii ] = meshgrid( 0 : inputWidth - 1, 0 : inputHeight - 1 ); 
% no rounding
di = ( ii / samplingSpatial ) + paddingXY + 1;
dj = ( jj / samplingSpatial ) + paddingXY + 1;
dz = ( edge - edgeMin ) / samplingRange + paddingZ + 1;

% interpn takes rows, then cols, etc
% i.e. size(v,1), then size(v,2), ...
output = interpn( normalizedBlurredGrid, di, dj, dz );