from sklearn import decomposition
import pandas as pd
import numpy as np

def MinMaxScaleClip(x, xmin, xmax):
    scaled = (x - xmin) / ((xmax - xmin) + 1e-8)
    return scaled
def StandardScaler( x, xmean, xstd):
    return (x-xmean) / (xstd + 1e-8) 


def PCA_extract (df_original,pc_require):
    pca = decomposition.PCA(n_components=pc_require,random_state = 8)
    pca_res = pd.DataFrame(pca.fit_transform(df_original))
    
    return pca_res

def PCA_transform (df_original,pc_require):
    pca_res = PCA_extract(df_original,pc_require)
    pca_res = pca_res.values
    pca_res= ((pca_res[..., None])+(pca_res[:, None, :]))*0.5
    pca_res = np.triu(pca_res).reshape(len(pca_res),-1)
    df_pair = pd.DataFrame(pca_res, index=df_original.index)
    df_pair = df_pair.loc[:,(df_pair!=0).any(axis=0)].fillna(0)
    df_pair = StandardScaler(df_pair,df_pair.mean(),df_pair.std())
    
    return df_pair