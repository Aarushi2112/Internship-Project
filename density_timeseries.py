#Animates the electron density files(in a directory) for a given width

import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from elmtheory import *
#define path
path = './CME/electron_density/'
pa_i = 180
pa_f = 260
degree = 360
center = (511,511)

def polar(img):
    pimg = cv2.warpPolar(img,(511,degree),center,511,cv2.INTER_NEAREST)
    
    pimg = np.delete(pimg,slice(95),1)
    return pimg



#Call Electron density function

# filenames = np.sort(listdir('./CME/K_corona'))
# 
# for i in range(len(filenames)):
#     edensity(filenames[i],i)
    
    

# Run animation
filenames = listdir(path)
filenames = np.sort(filenames)

combined_data = np.zeros((len(filenames), pa_f - pa_i, 416))

for i, fn in enumerate(filenames):
   x = np.load('./CME/electron_density/'+fn)
   p = polar(x)
   combined_data[i, :, :] = p[pa_i:pa_f, :]
   

from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
i=0
im = plt.imshow(combined_data[0,:,:],cmap='gray')
def update(i):
    im_normed = combined_data[i,:,:]
    ax.imshow(im_normed,cmap='gray')
    ax.set_axis_off()

anim = FuncAnimation(fig, update, frames=11, interval=200)

plt.show()
