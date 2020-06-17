#! /usr/bin/env python
#
# Version 0.9  (HS 09/03/2020)
#
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# a function that plots polygons A and B to visualisie the 2D region
def plotPolygs(A_x,A_y,B_x,B_y):
    plt.plot(A_x,A_y, 'ro')
    plt.plot(B_x,B_y,'bo')
    plt.axis([-5,5,-5,5])
    plt.xticks(np.arange(-5, 5, step=0.5))
    plt.yticks(np.arange(-5, 5, step=0.5))
    x = np.linspace(-5,5,100)
    line1 = 1.1788883222845488*x+1.4109455798062212
    line2 = -0.48209768832257976*x+3.7946432134886177
    line3 = 1.1936472630608137*x+ 0.15090339120056706
    line4 = -0.5291232972406565*x+4.532012292280824
    plt.plot(x, line1, '-r', label='line 1') #thick green, left
    plt.plot(x, line2, '-r', label='line 2') #thick blue, bottom
    plt.plot(x, line3, '-r', label='line 3') #right
    plt.plot(x, line4, '-r', label='line 4') #upper
    plt.show()

# a function that determines weights from the edges of the polygon
def findWeights(x1,x2,y1,y2):
    weights = []
    gradient = (y2-y1)/(x2-x1)
    c = y1-gradient*x1
    w1=1 #x
    w2=-1/gradient #y
    w0=-c*w2 #bias
    weights.append(w0) #bias
    weights.append(w1) #x 
    weights.append(w2) #y
    weights = np.asarray(weights)
    weights = weights/np.amax(weights)
    return weights


if __name__ == "__main__":
    A_x=[1.82731, 1.43511, 2.1744, 2.54306]
    A_y=[3.56514, 3.10278, 2.74637, 3.18642]
    B_x = [1.53615, 7.2419, 1.96605, -1.00973]
    B_y = [5.51733, 1.17359, 1.80886, 0.680014]
    plotPolygs(A_x,A_y,B_x,B_y)
    
    # find weights for polygon A
    Aweights1 = findWeights(1.82731, 1.43511,3.56514,3.10278)
    Aweights2 = findWeights(1.43511, 2.1744, 3.10278,2.74637)
    Aweights3 = -1*findWeights(2.1744, 2.54306,2.74637,3.18642)
    Aweights4 = -1*findWeights(2.54306,1.82731,3.18642,3.56514)
    
    # find weights for polygon B
    Bweights5 = -1*findWeights(1.53615, 7.2419, 5.51733, 1.17359) #2
    Bweights6 = findWeights(7.2419, 1.96605, 1.17359, 1.80886)
    Bweights7 = -1*findWeights(1.96605, -1.00973,1.80886, 0.680014)
    Bweights8 = findWeights(-1.00973,1.53615,0.680014,5.51733) #2



    print(Aweights1)
    print(Aweights2)
    print(Aweights3)
    print(Aweights4)
    
   # print(Bweights5)
   # print(Bweights6)
   # print(Bweights7)
   # print(Bweights8)
    
    point_inside = np.asarray([1,1.8,3.1])
    point_outside = np.asarray([1,1.5,2.7])
    print(np.dot(Aweights1,point_outside))
    print(np.dot(Aweights2,point_outside))
    print(np.dot(Aweights3,point_outside))
    print(np.dot(Aweights4,point_outside))
    
 
    
  


   

    # Polygon_A:  1.82731 3.56514 1.43511 3.10278 2.1744 2.74637 2.54306 3.18642
    # Polygon_B:  1.53615 5.51733 7.2419 1.17359 1.96605 1.80886 -1.00973 0.680014
