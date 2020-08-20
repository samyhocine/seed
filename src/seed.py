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

# Polynomial regression
def getPolyCoeff(M):

    N = MinMaxScaler().fit_transform(M.reshape(-1, 1)).reshape(M.shape[0], M.shape[1]) # Min-Max normalisation of M

    # Get meshgrids along 2 axis from 0 to 1 on each axis
    X, Y = np.meshgrid(np.linspace(start=0, stop=1, num=M.shape[1], endpoint=True), 
                       np.linspace(start=0, stop=1, num=M.shape[0], endpoint=True), 
                       sparse=False, indexing='xy')

    # Concatenation
    C = np.c_[X.flatten(), Y.flatten()]

    # A = [1 X Y X**2 2*X*Y Y**2 ...]
    A = PolynomialFeatures(degree=3, interaction_only=False, include_bias=True).fit_transform(C)

    # K is such as N = A*k
    K = np.linalg.lstsq(A, N.flatten(), rcond=None)[0]
    K = np.asarray(K).reshape(A.shape[1], 1)

    # Debug
    # D = np.matmul(A, K).reshape(M.shape[0], M.shape[1])
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.plot_surface(X=X, Y=Y, Z=N)
    # ax.plot_surface(X=X, Y=Y, Z=D)
    # plt.show()

    return K

def grow(width, height, n, KU, KV, KW):

    h = int(height/n)
    w = int(width/n)

    X, Y = np.meshgrid(np.linspace(start=0, stop=1, num=w, endpoint=True), 
                       np.linspace(start=0, stop=1, num=h, endpoint=True), 
                       sparse=False, indexing='xy')

    C = np.c_[X.flatten(), Y.flatten()]

    # A = [1 X Y X**2 2*X*Y Y**2 ...]
    A = PolynomialFeatures(degree=3, interaction_only=False, include_bias=True).fit_transform(C)

    DU = []
    DV = []
    DW = []
    for i in range(n*n):
        DU.append(255*np.matmul(A, KU[i]).reshape(h, w))
        DV.append(255*np.matmul(A, KV[i]).reshape(h, w))
        DW.append(255*np.matmul(A, KW[i]).reshape(h, w))
        
        DU[i] = DU[i].clip(0, 255)
        DV[i] = DV[i].clip(0, 255)
        DW[i] = DW[i].clip(0, 255)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X=X, Y=Y, Z=DU[0])
    
    plt.show()

    print(DU[0].dtype)
    print(DU[1].dtype)

    A = np.concatenate([DU[0], DU[1]], axis=1)
    B = np.concatenate([DU[2], DU[3]], axis=1)

    R = np.concatenate((A, B), axis=0)

    A = np.concatenate([DV[0], DV[1]], axis=1)
    B = np.concatenate([DV[2], DV[3]], axis=1)

    G = np.concatenate((A, B), axis=0)

    A = np.concatenate([DW[0], DW[1]], axis=1)
    B = np.concatenate([DW[2], DW[3]], axis=1)

    B = np.concatenate((A, B), axis=0)

    RGB = np.zeros([height, width, 3], dtype=np.uint8)
    RGB[:,:,0] = R
    RGB[:,:,1] = G
    RGB[:,:,2] = B

    img = Image.fromarray(RGB)
    img.save('testrgb.png')

    # print(DU)

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

    KU = []
    KV = []
    KW = []
    for i in range(n*n):
        KU.append(getPolyCoeff(U[i])*1000)
        KV.append(getPolyCoeff(V[i])*1000)
        KW.append(getPolyCoeff(W[i])*1000)

    KU = np.round(KU)
    KV = np.round(KV)
    KW = np.round(KW)

    # Done

    # Test

    KU = KU[:]/1000
    KV = KV[:]/1000
    KW = KW[:]/1000

    grow(width, height, n, KU, KV, KW)
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



    # DEBUG: Check slicing
    # img_U0 = Image.fromarray(U[0], mode='L')
    # img_U0.show()
    # print(U[0].shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('img_path', type=str, help='image path')
    args = parser.parse_args()
    main(args.img_path)