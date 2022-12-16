import cv2
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import os

S = 1367

def normalize(j):
    j = ((j - np.min(j))/(np.max(j)-np.min(j)))*255
    return j

            
def masking(x):
    mask = np.zeros((1024,1024))
    mask = cv2.circle(mask,(511,511),511,(255,255,255),-1)
    mask = cv2.circle(mask, (515,510), 95, (0,0,0),-1)
    mask = np.true_divide(mask,255)
    x_m = np.multiply(x,mask)
    return x_m


m = (2.0/3.0)*np.array([[1, 1, 1],[-1, 2, -1],[-np.sqrt(3), 0, np.sqrt(3)]])

filenames = np.sort(os.listdir('./CME/post_IDL'))

for i in range(len(filenames[::3])):
    data = fits.open('F:/Study/Solar_Physics/F&K_pB_images/code/CME/post_IDL/'+filenames[3*i])
    x0 = masking(np.array(data[0].data))
    data = fits.open('F:/Study/Solar_Physics/F&K_pB_images/code/CME/post_IDL/'+filenames[3*i+1])
    x120 = masking(np.array(data[0].data))
    data = fits.open('F:/Study/Solar_Physics/F&K_pB_images/code/CME/post_IDL/'+filenames[3*i+2])
    x240 = masking(np.array(data[0].data))


    # plt.subplot(1, 3, 1)
    # plt.imshow(x0, cmap='gray')
    # plt.title("0 degree")
    # plt.subplot(1, 3, 2)
    # plt.imshow(x120, cmap='gray')
    # plt.title("120 degree")
    # plt.subplot(1, 3, 3)
    # plt.imshow(x240, cmap='gray')
    # plt.title("240 degree")

    I = x120*m[0,0] + x0*m[0,1] + x240*m[0,2]
    Q = x120*m[1,0] + x0*m[1,1] + x240*m[1,2]
    U = x120*m[2,0] + x0*m[2,1] + x240*m[2,2]

    Ip_real = np.sqrt(np.square(Q) + np.square(U))
    Iu_real = I - Ip_real

    Ip_real , Iu_real = Ip_real*S, Iu_real*S

#     I = np.log(normalize(masking(I))+1)
#     Ip = np.log(normalize(masking(Ip_real))+1)
#     Iu = np.log(normalize(masking(Iu_real))+1)
#     #Iu = np.clip(Iu,0.8,np.max(Iu))
#     for i in range(np.shape(Iu)[0]):
#         for j in range(np.shape(Iu)[1]):
#             if Iu[i][j] < 1:
#                 Iu[i][j] = 1
                
    np.save('./CME/K_corona/'+str(i),Ip_real)

# print(np.mean(Ip_real))
# plt.subplot(1, 3, 1)
# plt.imshow(I, cmap='gray')
# plt.title("Total Brightness")
# plt.colorbar(fraction=0.046, pad=0.04)
# plt.subplot(1, 3, 2)
# plt.imshow(Ip, cmap='gray')
# plt.title("K corona")
# plt.colorbar(fraction=0.046, pad=0.04)
# plt.subplot(1, 3, 3)
# plt.imshow(Iu, cmap='gray')
# plt.title("F corona")
# plt.colorbar(fraction=0.046, pad=0.04)
# 
# # plt.hist(Iu.ravel(), 50, (np.min(Iu), np.max(Iu)))
# np.save("polarized", Ip_real)
# plt.show()

