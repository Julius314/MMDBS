import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import umap
import random


def performSanityCheck(autoencoder, encoder, x_test, y_test):
    #reconstruction(autoencoder, x_test)
    #distribution(encoder, x_test)
    projection(encoder, x_test, y_test)
    plt.show()

# visualize the reconstruction results
def reconstruction(autoencoder, x_test):
    print("doing reconstruction")

    reconstructed_imgs = autoencoder.predict(x_test)
    img_shape = (32, 32, 3)

    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(img_shape))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed_imgs[i].reshape(img_shape))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

def distribution(encoder, x_test):
    print("doing distribution")
    #distribution analysis
#    sampling = random.choices(x_test, k=50)
    sampling = x_test[:50]
    dist = encoder.predict(sampling)
    df = pd.DataFrame(dist)
    sns.pairplot(df)

def projection(encoder, x_test, y_test):
    print("doing projection")
    project = encoder.predict(x_test)
    df = pd.DataFrame(project)
    um = umap.UMAP()
    p = um.fit_transform(df)
    x, y = p.T

    labels = []
    for el in y_test:
        labels.append(el[0])

    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, c=labels, cmap='Paired', s=1.5)



