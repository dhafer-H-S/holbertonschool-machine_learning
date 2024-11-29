import matplotlib.pyplot as plt
import numpy as np

# Load and preprocess the images
array_of_pictures = np.load("small_res_faces_10000.npy")
array_of_pictures = array_of_pictures.astype("float32") / 255

# Display the real faces
fig, axes = plt.subplots(10, 10, figsize=(10, 10))
fig.suptitle("Real Faces")
for i in range(100):
    axes[i // 10, i % 10].imshow(array_of_pictures[i, :, :])
    axes[i // 10, i % 10].axis("off")
plt.show()

# Calculate the mean face and normalize the images
mean_face = array_of_pictures.mean(axis=0)
plt.imshow(mean_face)
centered_array = array_of_pictures - mean_face
multiplier = np.max(np.abs(array_of_pictures), axis=0)
normalized_array = centered_array / multiplier

def recover(normalized):
    return normalized * multiplier + mean_face

# Define the generator and discriminator models
def convolutional_GenDiscr():
    def get_generator():
        from keras.models import Sequential
        from keras.layers import Dense, Reshape, Conv2DTranspose, BatchNormalization, Activation

        model = Sequential()
        model.add(Dense(7 * 7 * 256, input_dim=100))
        model.add(Reshape((7, 7, 256)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2DTranspose(128, kernel_size=5, strides=2, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2DTranspose(64, kernel_size=5, strides=2, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2DTranspose(1, kernel_size=5, strides=1, padding='same', activation='sigmoid'))

        return model

    def get_discriminator():
        from keras.models import Sequential
        from keras.layers import Conv2D, Flatten, Dense, Dropout, Activation

        model = Sequential()
        model.add(Conv2D(64, kernel_size=5, strides=2, padding='same', input_shape=(28, 28, 1)))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Conv2D(128, kernel_size=5, strides=2, padding='same'))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        return model

    return get_generator(), get_discriminator()

# Create the generator and discriminator models
gen, discr = convolutional_GenDiscr()
print(gen.summary(line_length=100))
print(discr.summary(line_length=100))