close all;

%Parameters
slice = 30;
dimension = 3;
transparency = 0.35;
edgeTransparency = 0.8;
rescaleFactor = 4;
threshold = 0.3;

  
% fullIndexes = {'1','2','4','6','7','8','9','10','12'};
fullIndexes = [1,2,4,5,6,7,8,9,10,11,12,13,14];

startingDir = pwd;

addpath('/media/data1/artin/code/general_Research/overlay/');
addpath('/media/data1/artin/code/general_Research/overlay/NIfTI_20140122/')
dataDir = '/media/data1/artin/overlay/';
cd(dataDir);
predSt = ['tm' , '.nii.gz'];
% [f,p] = uigetfile('*.nii.gz','Multiselect','on');
niiWMN = load_untouch_nii(fullfile(dataDir,'maskwmn.nii.gz'));
addpath('/media/data1/artin/overlay/');

nii = cell(1,max(length(fullIndexes)));
for i = fullIndexes
    name = NucleiSelection(i);
    nii{i} = load_untouch_nii(fullfile([dataDir,'prediction/'],[name,'.nii.gz']));
end
%%

if (dimension == 1)
    im = rot90(fliplr(squeeze(niiWMN.img(slice,:,:))),3);
    imT = rot90(fliplr(squeeze(nii{1}.img(slice,:,:))),3);
    plane = 'sagittal';
elseif (dimension == 2)
    im = rot90(fliplr(squeeze(niiWMN.img(:,slice,:))),3);
    imT = rot90(fliplr(squeeze(nii{1}.img(:,slice,:))),3);
    plane = 'coronal';
else
    im = rot90(fliplr(squeeze(niiWMN.img(:,:,slice))),3);
    imT = rot90(fliplr(squeeze(nii{1}.img(:,:,slice))),3);
    plane = 'axial';
end

% nClusters = numel(f);
nClusters = max(fullIndexes);
cMap = jet(nClusters);

originalSize = size(im);
im = [im,im];
im = imresize(im,rescaleFactor,'bicubic');
imT = imresize(imT,originalSize,'nearest');
imT = imresize(imT,rescaleFactor,'bicubic');
imT(imT >= threshold) = 1;
imT(imT < threshold) = 0;
imT = [0*imT,imT];

figure;
hold on;
imagesc(im);
axis equal;
colormap(gray);

niiMask = cell(1,max(fullIndexes));
for i = fullIndexes
    name = NucleiSelection(i);
    niiMask{i} = load_untouch_nii(fullfile([dataDir,'manual/'],[name,'_PProcessed.nii.gz']));
end

for i = fullIndexes
    
%     niiMask = load_untouch_nii(fullfile(p,f{i}));
    
    if (dimension == 1)
        mask = rot90(fliplr(squeeze(niiMask{i}.img(slice,:,:))),3);
    elseif (dimension == 2)
        mask = rot90(fliplr(squeeze(niiMask{i}.img(:,slice,:))),3);
    else
        mask = rot90(fliplr(squeeze(niiMask{i}.img(:,:,slice))),3);
    end
    
    mask = imresize(mask,rescaleFactor,'bicubic');
    mask(mask >= threshold) = 1;
    mask(mask < threshold) = 0;
    mask = [0*mask,mask];
    indices = find(mask > 0);
    
    color = cat(3,cMap(i,1)*ones(size(im)),cMap(i,2)*ones(size(im)),cMap(i,3)*ones(size(im)));
    colorMask = zeros(size(im));
    colorMask(indices) = transparency;
    hColor = imshow(color);
    set(hColor,'AlphaData',colorMask);
    
end

edgeColor = cat(3,ones(size(im)),ones(size(im)),zeros(size(im)));%Change this - loop over all overlays

% MLabelSt = ['st','.nii.gz'];

for i = fullIndexes
    
%     niiEdge = load_untouch_nii([stOverlays{i},MLabelSt]);
    niiEdge = nii{i};
    
    if (dimension == 1)
        maskEdge = rot90(fliplr(squeeze(niiEdge.img(slice,:,:))),3);
    elseif (dimension == 2)
        maskEdge = rot90(fliplr(squeeze(niiEdge.img(:,slice,:))),3);
    else
        maskEdge = rot90(fliplr(squeeze(niiEdge.img(:,:,slice))),3);
    end
    
    maskEdge = imresize(maskEdge,rescaleFactor,'bicubic');
    maskEdge(maskEdge >= threshold) = 1;
    maskEdge(maskEdge < threshold) = 0;
    maskEdge = [0*maskEdge,maskEdge];    
    imEdge = edgeTransparency*edge(maskEdge,'canny');
    indices = find(imEdge > 0);
    colorMask = zeros(size(im));
    colorMask(indices) = edgeTransparency;
    hColor = imshow(edgeColor);
    set(hColor,'AlphaData',colorMask);
end
saveas(gcf, [dataDir , plane, num2str(slice) , '.png'] )
title(['Segmentation comparison: ',plane,' plane, slice ',num2str(slice)]);
% axis tight;

cd(startingDir);