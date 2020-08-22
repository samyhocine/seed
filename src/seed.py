import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler

def sliceChannels(width, height, R, G, B, n):
    p = height # hypothesis: width=height (for now)

    # Debug
    RGB = np.zeros([height, width, 3], dtype=np.uint8)
    RGB[:,:,0] = R
    RGB[:,:,1] = G
    RGB[:,:,2] = B
    img = Image.fromarray(RGB)
    img.save('../data/dbg/in.png')

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

            # Debug
            RGB = np.zeros([int(height/n), int(width/n), 3], dtype=np.uint8)
            RGB[:,:,0] = R[i_min:i_max, j_min:j_max]
            RGB[:,:,1] = 0
            RGB[:,:,2] = 0
            img = Image.fromarray(RGB)
            img.save('../data/dbg/'+str(k_i)+'_'+str(k_j)+'_in.png')
            
    return (U, V, W)

# Polynomial regression
def getPolyCoeff(M, dbg=False):

    # N = MinMaxScaler().fit_transform(M.reshape(-1, 1)).reshape(M.shape[0], M.shape[1]) # Min-Max normalisation of M
    N = M # No normalisation

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
    if dbg:
        D = np.matmul(A, K).reshape(M.shape[0], M.shape[1])
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(X=X, Y=Y, Z=N)
        ax.plot_surface(X=X, Y=Y, Z=D)
        plt.show()

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
        # With normalisation ON
        # DU.append(255*np.matmul(A, KU[i]).reshape(h, w))
        # DV.append(255*np.matmul(A, KV[i]).reshape(h, w))
        # DW.append(255*np.matmul(A, KW[i]).reshape(h, w))

        # Without normalisation     
        DU.append(np.matmul(A, KU[i]).reshape(h, w))
        DV.append(np.matmul(A, KV[i]).reshape(h, w))
        DW.append(np.matmul(A, KW[i]).reshape(h, w))
        
        DU[i] = DU[i].clip(0, 255)
        DV[i] = DV[i].clip(0, 255)
        DW[i] = DW[i].clip(0, 255)

    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.plot_surface(X=X, Y=Y, Z=DU[0])
    # plt.show()

    RC = []
    GC = []
    BC = []
    for i in range(n):
        A = np.concatenate([DU[i*n], DU[i*n + 1]], axis=1)
        B = np.concatenate([DV[i*n], DV[i*n + 1]], axis=1)
        C = np.concatenate([DW[i*n], DW[i*n + 1]], axis=1)
        for j in range(2, n):
            idx = i*n + j
            A = np.concatenate([A, DU[idx]], axis=1)
            B = np.concatenate([B, DV[idx]], axis=1)
            C = np.concatenate([C, DW[idx]], axis=1)
        RC.append(A)
        GC.append(B)
        BC.append(C)

    R = np.concatenate([RC[0], RC[1]], axis=0)
    G = np.concatenate([GC[0], GC[1]], axis=0)
    B = np.concatenate([BC[0], BC[1]], axis=0)
    for i in range(2, n):
        R = np.concatenate([R, RC[i]], axis=0)
        G = np.concatenate([G, GC[i]], axis=0)
        B = np.concatenate([B, BC[i]], axis=0)

    print("R.shape=", R.shape)

    RGB = np.zeros([height, width, 3], dtype=np.uint8)
    RGB[:,:,0] = R
    RGB[:,:,1] = G
    RGB[:,:,2] = B

    img = Image.fromarray(RGB)
    img.save('../data/dbg/out.png')
   
    # Debug
    RGB = np.zeros([int(height/n), int(width/n), 3], dtype=np.uint8)
    RGB[:,:,1] = 0
    RGB[:,:,2] = 0
    for i in range(n):
        for j in range(n):
            idx = i*n + j
            RGB[:,:,0] = DU[idx]
            img = Image.fromarray(RGB)
            img.save('../data/dbg/'+str(i)+'_'+str(j)+'_out.png')

    print("Growing process done")

def main(img_path):

    img = Image.open(img_path)

    (width, height) = img.size
    n = 4

    R = np.asarray(img.getdata(band=0), dtype=np.uint8).reshape(height, width)
    G = np.asarray(img.getdata(band=1), dtype=np.uint8).reshape(height, width)
    B = np.asarray(img.getdata(band=2), dtype=np.uint8).reshape(height, width)

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

    # # Debug
    # k_i = 2
    # k_j = 0
    # idx = k_i*n + k_j
    # getPolyCoeff(U[idx], True)

    # Test
    KU = KU[:]/1000
    KV = KV[:]/1000
    KW = KW[:]/1000

    grow(width, height, n, KU, KV, KW)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('img_path', type=str, help='image path')
    args = parser.parse_args()
    main(args.img_path)