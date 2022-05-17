import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import sobel
from scipy.interpolate import interp2d
from scipy.interpolate import RectBivariateSpline
from skimage.transform import rescale
import cv2
from scipy.ndimage import binary_erosion
from skimage.morphology import disk
from skimage.exposure import equalize_adapthist
from skimage.filters import gaussian

img = imread('Gacr_01_001_01_L.JPG').astype(np.float32) / 255 
# img = rescale(img, 0.1)

mask0 = img[:,:,1]  > (5/255)

mask = binary_erosion(mask0,disk(1))


dt = distance_transform_edt(mask==0)


sx = sobel(dt,axis=0)/4
sy = sobel(dt,axis=1)/4
magnitude = np.hypot(sx,sy)


xx, yy = np.meshgrid(np.arange(dt.shape[1]), np.arange(dt.shape[0]))

xxx = xx - sy * dt
yyy = yy - sx * dt
                      

        
img_interp = cv2.remap(img, xxx.astype(np.float32), yyy.astype(np.float32), cv2.INTER_LINEAR)


G = gaussian(img_interp,80)
img_interp = (img_interp - G) / G + 0.5
img_interp[img_interp < 0] = 0 
img_interp[img_interp > 1] = 1

img_interp = equalize_adapthist(img_interp, kernel_size = 250, clip_limit=0.005 )

img_interp[mask0 == 0] = 0

plt.imshow(img_interp)
