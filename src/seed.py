import argparse
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures

def sliceChannels(width, height, R, G, B, n=2):
    p = height

    U = []
    V = []
    W = []
    for k_i in range(n):
        for k_j in range(n):
            i_min = np.int((p/n)*k_i)
            i_max = np.int((p/n)*(k_i + 1))
            j_min = np.int((p/n)*k_j)
            j_max = np.int((p/n)*(k_j + 1))

            _M = R[i_min:i_max, j_min:j_max]
            U.append(_M)

            _M = G[i_min:i_max, j_min:j_max]
            V.append(_M)

            _M = B[i_min:i_max, j_min:j_max]
            W.append(_M)
            
    return (U, V, W)

def main(img_path):
    img = Image.open(img_path)
    # img.show()

    (width, height) = img.size

    n = 2

    R = np.asarray(img.getdata(band=0), dtype=np.uint8)
    R = R.reshape(height, width)

    G = np.asarray(img.getdata(band=1), dtype=np.uint8)
    G = G.reshape(height, width)

    B = np.asarray(img.getdata(band=2), dtype=np.uint8)
    B = B.reshape(height, width)

    X = np.linspace(0, 1, int(width/n)).reshape(1, int(width/n))
    Y = np.linspace(0, 1, int(height/n)).reshape(int(width/n), 1)
    (XX, YY) = np.meshgrid(X, Y, sparse=False, indexing='ij')

    # print(X.flatten())
    # print(Y.flatten())

    # print(X)
    
    # features = {}
    # features['X^0*Y^0'] = np.matmul(X**0, Y**0).flatten()
    # print(features['X^0*Y^0'])
    # features['X*Y'] = np.matmul(X, Y).flatten()
    # features['X*Y^2'] = np.matmul(X, Y**2).flatten()
    # features['X^2*Y^0'] = np.matmul(X**2, Y**0).flatten()
    # features['X^2*Y'] = np.matmul(X**2, Y).flatten()
    # features['X^3*Y^2'] = np.matmul(X**3, Y**2).flatten()
    # features['X^3*Y'] = np.matmul(X**3, Y).flatten()
    # features['X^0*Y^3'] = np.matmul(X**0, Y**3).flatten()
    # dataset = pd.DataFrame(features)

    # print(len(X))
    # print(len(Y))
    # print(len(R[0:int((height/n)), 0:int((width/n))]))
    
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.plot_surface(X=X, Y=Y, Z=R[0:int((height/n)), 0:int((width/n))])
    # plt.show()

    (U, V, W) = sliceChannels(width, height, R, G, B, n)

    # print(dataset.values.shape)
    # print(U[0].flatten().shape)

    # print(dataset)

    print(X.flatten())

    # poly_features = PolynomialFeatures(degree=3)
    # print(poly_features)
    # X_poly = poly_features.fit_transform(np.array([X.flatten(), Y.flatten()]))
    # print(X_poly[2])

    # print(reg)

    # print(len(R[0:int((height/n)), 0:int((width/n))]))

    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.plot_surface(X=_X, Y=_Y, Z=V[0])
    # plt.show()

    # DEBUG: Check slicing
    # img_U0 = Image.fromarray(U[0], mode='L')
    # img_U0.show()
    # print(U[0].shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('img_path', type=str, help='image path')
    args = parser.parse_args()
    main(args.img_path)