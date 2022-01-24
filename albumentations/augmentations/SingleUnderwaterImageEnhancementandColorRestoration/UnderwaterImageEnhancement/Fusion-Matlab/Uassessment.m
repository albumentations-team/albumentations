%setDir = fullfile('I:\outPut_restoration');
setDir = fullfile('F:\PaperExperiments\PhysicalModel\InitialResults\OutputImages');
imds = imageDatastore(setDir,'FileExtensions',{'.jpg'});
FileNames =  imds.Files;
numFiles =  size(FileNames,1);
Fianl = [];
for fileIndex = 1:numFiles
     filename = cell2mat(FileNames(fileIndex));
     img = imread(filename);
     uiqm = UIQM(img);
     Fianl(fileIndex) = uiqm;
end
MeanUiqmInitial = mean(Fianl);