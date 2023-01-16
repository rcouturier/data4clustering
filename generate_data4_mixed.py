import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import numpy as np
import sys
from sklearn.cluster import KMeans
from PIL import Image
from sklearn.datasets import make_blobs
import random
import math
import os


np.random.seed(12)

def getPointsofCluster (Allpointsclusters, clust,numberofcluster):
    ind = (np.where(clust == numberofcluster))
    ClusterPoints=np.asarray(Allpointsclusters[ind]) 
    return ClusterPoints


def generate_data_with_blobs(nb_clust):
    centers=GenerateCenters(nb_clust)
    nb_samples=np.random.randint(20000,50000,size=nb_clust)
    X,y=make_blobs(n_samples=nb_samples, centers=centers, n_features=2, cluster_std=0.15+(nb_clust-2)*0.3)    
    return X,y,centers

def GenerateCenters(nb_clust):
    x= (np.random.permutation(nb_clust)*6)-3*nb_clust
    y= (np.random.permutation(nb_clust)*6)-3*nb_clust
    x=x[:nb_clust]
    y=y[:nb_clust]
    randomval=np.random.normal(0,1,size=(nb_clust,2))
    x=(x+randomval[:,0]).reshape(-1,1)
    y=(y+randomval[:,1]).reshape(-1,1)
    centers=np.concatenate((x, y), axis=1)
    return centers


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


size_im=640



def choose_distant_points(npoints=2,mindist=3,space=15):
    def genpt():
        return ((random.random()*2-1)*space, (random.random()*2-1)*space)

    def distance(p1,p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    restart=True
    sample = []
    while restart==True:
        restart=False

        l=0
        while len(sample) < npoints and l<100:
            newp = genpt()
            for p in sample:
                if distance(newp,p) < mindist: break
            else:
                sample.append(newp)
            l=l+1
        print(l)
        if l==100:
            restart=True
            sample = []

            
    return sample


for iter in range(1000):


    nb_clust=np.random.randint(2,10)
    L=[]
    Pos=[]
    space=20
    center=choose_distant_points(npoints=nb_clust,mindist=0,space=space)
    center=np.array(center)

    for i in range(nb_clust):
        cov=(np.random.rand(nb_clust,2)+0.1)*5
        nb=np.random.randint(20000,50000,size=nb_clust)
        values=np.random.multivariate_normal(mean=center[i,:], cov=[[cov[i,0], 0], [0, cov[i,1]]], size=nb[i])
        angle=np.random.rand()*math.pi*2
        values2=[rotate(center[i,:], point, angle) for point in values]
        values=np.array(values2)
        xmin=values[:,0].min()
        xmax=values[:,0].max()
        ymin=values[:,1].min()
        ymax=values[:,1].max()

        L.append(values)
        Pos.append((xmin,xmax,ymin,ymax))

    df = np.concatenate(L)


    xmin=ymin=-space-5
    xmax=ymax=space+5



    H, xedges, yedges=np.histogram2d(df[:,0],df[:,1],
                                 bins=(size_im,size_im),
                                 density=True,
                                 range=[[xmin, xmax], [ymin, ymax]])
    H = H.T


    H=H/H.max()

    H=1.-H

    plt.imshow(H, interpolation='nearest', origin='lower')

    centers=[]
    c2=[]
    for i in range(nb_clust):
        xminl,xmaxl,yminl,ymaxl=Pos[i]

        x=(xmaxl+xminl)/2
        y=(ymaxl+yminl)/2
        w=xmaxl-xminl
        h=ymaxl-yminl
        x=(x-xmin)/(xmax-xmin)
        y=((y-ymin)/(ymax-ymin)) 
        w=w/(xmax-xmin)
        h=h/(ymax-ymin)


        centers.append((0,x,y,w,h))








    im = Image.fromarray(np.uint8(H*255.)).convert('RGB')

    path = "train_data4_mixed/images/data"

    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    path = "train_data4_mixed/labels/data"
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    
    im.save("train_data4_mixed/images/data/img"+str(iter)+".png")
    np.savetxt('train_data4_mixed/labels/data/img'+str(iter)+'.txt', centers, delimiter=' ')

