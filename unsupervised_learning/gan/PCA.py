import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_100(images, title):
    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    fig.suptitle(title)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='gray')
        ax.axis('off')
    plt.show()

class PCA_generator:
    def __init__(self, array_of_pictures, depth):
        self.arr = array_of_pictures
        self.depth = depth

        self.n_ims = self.arr.shape[0]
        self.im_dims = self.arr.shape[1:]
        self.prod_im_dims = np.prod(self.im_dims)

        self.mean_face = self.arr.mean(axis=0)
        self.centered = self.arr - self.mean_face
        self.flat = self.centered.reshape(self.n_ims, self.prod_im_dims)
        self.pca = PCA(n_components=self.depth)
        self.transformed = self.pca.fit_transform(self.flat)
        self.stds = np.std(self.transformed, axis=0)

    def get_centered_sample_from_coords(self, coords):
        return self.pca.inverse_transform(coords).reshape([coords.shape[0]] + list(self.im_dims))

    def get_fake_sample(self, k):
        return self.get_centered_sample_from_coords(np.random.randn(k, self.depth) * self.stds) + self.mean_face

    def plot_100_fake(self):
        plot_100(self.get_fake_sample(100), f"PCA fake faces (depth={self.depth})")

    def plot_100_reconstructed_real(self):
        H = self.get_centered_sample_from_coords(self.transformed[:100]) + self.mean_face
        plot_100(H, f"PCA reconstructed real faces (depth={self.depth})")

array_of_pictures = np.load("small_res_faces_10000.npy")
array_of_pictures = array_of_pictures.astype("float32") / 255

G = PCA_generator(array_of_pictures, 128)
G.plot_100_fake()
G.plot_100_reconstructed_real()

H = G.get_fake_sample(10000)
flat = H.reshape(10000, -1)  # Flatten the array correctly
pca = PCA(n_components=49)
pca.fit(flat)
X = pca.transform(flat)
X = X / np.std(X, axis=0)

plt.hist(np.sum(np.square(X), axis=1), bins=100, density=True)
from scipy.stats import chi2
x = np.linspace(0, 200, 100)
plt.plot(x, chi2.pdf(x, 49))
plt.savefig("not_chi2.png")
plt.show()


H = G.get_fake_sample(10000)
flat = H.reshape(10000, -1)  # Flatten the array correctly
pca = PCA(n_components=49)
pca.fit(flat)
X = pca.transform(flat)
X = X / np.std(X, axis=0)

plt.hist(np.sum(np.square(X), axis=1), bins=100, density=True)
from scipy.stats import chi2
x = np.linspace(0, 200, 100)
plt.plot(x, chi2.pdf(x, 49))
plt.savefig("not_chi2.png")
plt.show()
