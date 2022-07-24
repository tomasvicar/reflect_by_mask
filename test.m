clc;clear all; close all;

img = double(imread('Gacr_01_001_01_L.JPG')) / 255 ;
%%% 'Gacr_01_001_01_L.JPG' is 2469/45 = 55  pix/degree

mask0 = img(:,:,1)  > (5/255);
mask = imerode(mask0,strel('disk',1));

dt = bwdist(mask==1);


[sx,sy] = imgradientxy(dt,'sobel');

magnitude = sqrt(sx.^2 + sy.^2);


[xx, yy] = meshgrid(1:size(dt,2), 1:size(dt,1));

xxx = xx - sx/4 .* dt;
yyy = yy - sy/4 .* dt;


img_interp(:,:,1) = interp2(img(:,:,1),xxx,yyy,'linear',0);
img_interp(:,:,2) = interp2(img(:,:,2),xxx,yyy,'linear',0);
img_interp(:,:,3) = interp2(img(:,:,3),xxx,yyy,'linear',0);

G = imgaussfilt(img_interp, 80 / (55/25));
img_interp = (img_interp - G) ./ G + 0.5;
img_interp(img_interp < 0) = 0; 
img_interp(img_interp > 1) = 1;

img_interp = rgb2hsv(img_interp);

img_interp(:,:,3) = adapthisteq(img_interp(:,:,3), 'NumTiles',[round(size(img_interp,1) / (250 / (55/25))) round(size(img_interp,2) / (250 / (55/25)))],'ClipLimit',0.005);

img_interp = hsv2rgb(img_interp);

img_interp(repmat(mask0 == 0,[1,1,3])) = 0;

imshow(img_interp,[])