import numpy as np
import matplotlib.pyplot as plt


def cloud_in_the_corner_numpy(N,sigma,S) :
    arr=np.random.randn(S*N).reshape(S,N)
    return arr*sigma+.8

def cloud_in_the_corner(C,sigma,S) :
    return tf.convert_to_tensor(cloud_in_the_corner_numpy(C,sigma,S),dtype="float32")

def mean_squared_dist_cloud_to_center(cloud) :
    return np.mean((np.sum(np.square(cloud-.8),axis=1)))

def expected_mean_squared_dist_cloud_to_center(N,sigma,S):
    return sigma*sigma*N

def verif_mean_squared_dist_cloud_to_center(N,sigma,S) :
    mean_squared_dists=[]
    for i in range(10000) :
        cloud=cloud_in_the_corner_numpy(N,sigma,S)
        mean_squared_dists.append(mean_squared_dist_cloud_to_center(cloud))
    print(f"""Among 10000 clouds of {S} points, the distribution of the mean squared distance 
          from the points of the cloud to the center has the following shape :""")
    a=plt.hist(mean_squared_dists,bins=50,density=True)
    expected = expected_mean_squared_dist_cloud_to_center(N,sigma,S)
    plt.plot([expected,expected],[0,np.max(a[0])])
    plt.text(expected, np.max(a[0])*1.1, r'$N\sigma^2$', dict(size=10))
    plt.ylim(0,np.max(a[0])*1.2)
    plt.show()
        
np.random.seed(0)  
verif_mean_squared_dist_cloud_to_center(10,.1,100)

def squared_dist_barycenter_to_center(cloud) :
    S=cloud.shape[0]
    N=cloud.shape[1]
    barycenter = 1/S*np.sum(cloud,axis=0)
    diff= barycenter - .8
    return np.sum(np.square(diff))

def expected_squared_dist_barycenter_to_center(N,sigma,S) :
    return sigma*sigma*N/S

def verif_squared_dist_barycenter_to_center(N,sigma,S) :
    squared_dists=[]
    for i in range(10000) :
        cloud=cloud_in_the_corner_numpy(N,sigma,S)
        squared_dists.append(squared_dist_barycenter_to_center(cloud))
    print(f"""Among 10000 clouds of {S} points, the distribution of the squared distance 
          from the barycenter of the cloud to the center has the following shape :""")
    a=plt.hist(squared_dists,bins=50,density=True)
    expected = expected_squared_dist_barycenter_to_center(N,sigma,S)
    plt.plot([expected,expected],[0,np.max(a[0])])
    plt.text(expected, np.max(a[0])*1.1, r'$\frac{N\sigma^2}{S}$', dict(size=10))
    plt.ylim(0,np.max(a[0])*1.2)
    plt.show()

np.random.seed(0) 
verif_squared_dist_barycenter_to_center(10,.1,100) 