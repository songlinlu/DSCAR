from tqdm import tqdm
from copy import copy
from aggmap.utils.matrixopt import conv2
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.preprocessing import StandardScaler
import gc
import tensorflow.keras.backend as K
import pandas as pd
import numpy as np
def islice(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]
def GlobalIMP(clf, mp, X, Y, task_type = 'classification', 
              binary_task = False,
              sigmoidy = False, 
              apply_logrithm = False,
              apply_smoothing = False, 
              kernel_size = 5, 
              sigma = 1.6):

    if task_type == 'classification':
        f = log_loss
    else:
        f = mean_squared_error
        
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    scaler = StandardScaler()
    df_grid = mp.df_grid_reshape
    backgroud = mp.transform_mpX_to_df(X).min().values #min value in the training set

    dfY = pd.DataFrame(Y)
    Y_true = Y
    Y_prob = clf.predict(X)
    N, W, H, C = X.shape
    T = len(df_grid)
    nX = 20 

    if (sigmoidy) & (task_type == 'classification'):
        Y_prob = sigmoid(Y_prob)
    

    final_res_just_log = {}
    final_res_just_smooth = {}
    final_res_both = {}
    final_res_original = {}
    for k, col in enumerate(dfY.columns):
        if (task_type == 'classification') & (binary_task):
            if k == 0:
                continue
        print('calculating feature importance for column %s ...' % col)
        results = []
        loss = f(Y_true[:, k].tolist(), Y_prob[:, k].tolist())
        
        tmp_X = []
        flag = 0
        for i in tqdm(range(T), ascii= True):
            ts = df_grid.iloc[i]
            y = ts.y
            x = ts.x
            
            ## step 1: make permutaions
            X1 = np.array(X)

            vmin = backgroud[i]
            X1[:, y, x,:] = np.full(X1[:, y, x,:].shape, fill_value = vmin)
            tmp_X.append(X1)
            
            if (flag == nX) | (i == T-1):
                X2p = np.concatenate(tmp_X)

                ## step 2: make predictions
                Y_pred_prob = clf.predict(X2p) #predict ont by one is not efficiency
                gc.collect()
                K.clear_session()
                if (sigmoidy) & (task_type == 'classification'):
                    Y_pred_prob = sigmoid(Y_pred_prob)    

                ## step 3: calculate changes
                for Y_pred in islice(Y_pred_prob, N):
                    mut_loss = f(Y_true[:, k].tolist(), Y_pred[:, k].tolist()) 
                    res =  mut_loss - loss 
                    results.append(res)

                flag = 0
                del tmp_X
                tmp_X = []

            flag += 1

        ## step 4:apply scaling or smothing 
        s_just_log = pd.DataFrame(results).values
        s_just_smooth = pd.DataFrame(results).values
        
        if apply_logrithm:
            s_just_log = np.log(s_just_log)


        smin = np.nanmin(s_just_log[s_just_log != -np.inf])
        smax = np.nanmax(s_just_log[s_just_log != np.inf])

        s_just_log = np.nan_to_num(s_just_log, nan=smin, posinf=smax, neginf=smin) #fillna with smin
        s_just_smooth = np.nan_to_num(s_just_smooth, nan=smin, posinf=smax, neginf=smin) #fillna with smin


        a_just_log = scaler.fit_transform(s_just_log).reshape(*mp._S.fmap_shape)
        a_just_smooth = scaler.fit_transform(s_just_smooth).reshape(*mp._S.fmap_shape)


        covda_just_smooth = conv2(a_just_smooth, kernel_size=kernel_size, sigma=sigma)
        results_just_smooth = covda_just_smooth.reshape(-1,).tolist()

        results_just_log = a_just_log.reshape(-1,).tolist()


        final_res_just_log.update({col:results_just_log})
        final_res_just_smooth.update({col:results_just_smooth})
        
    df_just_log = pd.DataFrame(final_res_just_log)
    df_just_log.columns = df_just_log.columns.astype(str)
    df_just_log.columns = 'col_' + df_just_log.columns + '_importance'
    df_just_log = df_grid.join(df_just_log)

    df_just_smooth = pd.DataFrame(final_res_just_smooth)
    df_just_smooth.columns = df_just_smooth.columns.astype(str)
    df_just_smooth.columns = 'col_' + df_just_smooth.columns + '_importance'
    df_just_smooth = df_grid.join(df_just_smooth)
  
    del clf
    
    gc.collect()
    K.clear_session()
    return df_just_log,df_just_smooth