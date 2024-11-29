import numpy as np
import matplotlib.pyplot as plt

# Define array_of_pictures for demonstration purposes
array_of_pictures = np.random.rand(10000, 256)

flat=array_of_pictures.reshape(10000,256)
from sklearn.decomposition import PCA
pca = PCA(n_components=49)
pca.fit(flat)
X=pca.transform(flat)
X/=np.std(X,axis=0)

from scipy.stats import norm
x=np.linspace(-5,5,50)
y=norm.pdf(x)

fig,axes=plt.subplots(7,7,figsize=(21,11))
for i in range(49) :
    axes[i//7,i%7].hist(X[:,i],density=True, bins=40)
    axes[i//7,i%7].plot(x,y, color="magenta")
plt.show()