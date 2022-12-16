import math
import numpy as np
import matplotlib.pyplot as plt
import cv2


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


def eltheory(Rin,t):
    if Rin <95 or Rin > 500:
        return 0
    const = 1.24878 * pow(10,-25)
    u = 0.63
    const = const/(1-u/3.00)
    
    
    theta = min(t,math.pi/2.00)
    R = Rin/np.cos(theta)
    sinchi2 = (Rin/R)**2     #angle b/w sun center, electron and observer
    s = min(float(1/R), 0.9999999)  #sin(omega)
    c = math.sqrt(1-s**2)    #cos(omega)
    g = (1-s**2)*(math.log((1 + s)/c))/s
    
    #van de Hulst coefficients

    ael = c*(s**2)
    cel = (4-c*(2 + c**2))/3
    bel = -(1 - 3*(s**2) - g*(1 + 3*(s**2)))/8
    d_el = (5 + s**2 - g*(5 - s**2))/8
    
    Bt = const*(cel + u*(d_el - cel))
    pB = const*sinchi2*((ael + u*(bel-ael)))

    B = 2*Bt - pB
    
    return B


def edensity(file,i):

    
    center = (511,511)
    
    img = np.load('./CME/K_corona/'+file)
    
    ed = np.zeros(np.shape(img))

    for i in range(np.shape(img)[0]):
        for j in range(np.shape(img)[1]):
            Rin = math.dist(center, (i,j))
            t = 0
            ed[i,j] = eltheory(Rin,t)
            if ed[i,j] == 0:
                ed[i,j] = np.inf
            
    edf = np.divide(img,ed)
    np.save('./CME/electron_density/'+file,edf)

    

# print(np.min(edf),np.max(edf))
# 
# plt.figure(1)
# plt.subplot(1,2,1)
# plt.title("K corona")
# plt.imshow(img,cmap = 'gray')
# plt.subplot(1,2,2)
# plt.title("Electron density")
# plt.imshow(edf,cmap = 'gray')
# 
# plt.figure(2)
# plt.subplot(1,2,1)
# plt.title("K corona")
# plt.imshow(np.log(normalize(masking(img))+1),cmap = 'gray')
# plt.subplot(1,2,2)
# plt.title("Electron density")
# plt.imshow(np.log(normalize(masking(edf))+1),cmap = 'gray')
# plt.show()
    
