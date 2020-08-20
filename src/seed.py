import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler

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

    # print(len(R[0:int((height/n)), 0:int((width/n))]))
    
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.plot_surface(X=X, Y=Y, Z=R[0:int((height/n)), 0:int((width/n))])
    # plt.show()

    (U, V, W) = sliceChannels(width, height, R, G, B, n)

    print("len(U)", len(U))

    NU0 = MinMaxScaler().fit_transform(U[0].reshape(-1, 1)).reshape(U[0].shape[0], U[0].shape[1])

    X, Y = np.meshgrid(np.linspace(start=0, stop=1, num=NU0.shape[1], endpoint=True), 
                    np.linspace(start=0, stop=1, num=NU0.shape[0], endpoint=True), 
                    sparse=False, indexing='xy')

    C = np.c_[X.flatten(), Y.flatten()]

    M = PolynomialFeatures(degree=3).fit_transform(C)

    CNU0 = np.linalg.lstsq(M, NU0.flatten(), rcond=None)[0]
    CNU0 = np.asarray(CNU0).reshape(M.shape[1], 1)

    print(CNU0)

    D = np.matmul(M, CNU0).reshape(NU0.shape[0], NU0.shape[1])

    # Todo optimise: If coeff < 10e-9 then coeff = 0

    # Test
    # D = np.matmul()

    # print(U[0].flatten().shape)

    # print(dataset)

    # poly_features = PolynomialFeatures(degree=3)
    # print(poly_features)
    # X_poly = poly_features.fit_transform(np.array([X.flatten(), Y.flatten()]))
    # print(X_poly[2])

    # print(reg)

    # print(len(R[0:int((height/n)), 0:int((width/n))]))

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X=X, Y=Y, Z=NU0)
    ax.plot_surface(X=X, Y=Y, Z=D) # Debug
    plt.show()

    # DEBUG: Check slicing
    # img_U0 = Image.fromarray(U[0], mode='L')
    # img_U0.show()
    # print(U[0].shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('img_path', type=str, help='image path')
    args = parser.parse_args()
    main(args.img_path)