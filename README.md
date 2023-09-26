<img src="picture/scarvover.png" align="right" height="250" width="480" >

# DSCAR & DSCARNet
<br />

### A new tool for Raman spectra 2D representation and modeling


<br />

## DSCAR: 2D representation of Raman spectra
![image](picture/2d.png)

## DSCARNet: Deep learning model for 2D-SARs and 2D-CARs
![image](picture/dscar_net.png)

---
- The performance of DSCARNet was evaluated on 12 datasets from 8 references listed here.
```
1.Gala de Pablo J, Armistead F J, Peyman S A, et al. Biochemical fingerprint of colorectal cancer cell lines using label‐free live single‐cell Raman spectroscopy[J]. Journal of Raman Spectroscopy, 2018, 49(8): 1323-1332.
2.Baria E, Cicchi R, Malentacchi F, et al. Supervised learning methods for the recognition of melanoma cell lines through the analysis of their Raman spectra[J]. Journal of Biophotonics, 2021, 14(3): 202000365.
3.Akagi Y, Mori N, Kawamura T, et al. Non-invasive cell classification using the Paint Raman Express Spectroscopy System (PRESS)[J]. Scientific reports, 2021, 11(1): 1-15.
4.Hsu C C, Xu J, Brinkhof B, et al. A single-cell Raman-based platform to identify developmental stages of human pluripotent stem cell-derived neurons[J]. Proceedings of the National Academy of Sciences, 2020, 117(31): 18412-18423.
5.García‐Timermans C, Rubbens P, Heyse J, et al. Discriminating bacterial phenotypes at the population and single‐cell level: a comparison of flow cytometry and Raman spectroscopy fingerprinting[J]. Cytometry Part A, 2020, 97(7): 713-726.
6.Pavillon N, Hobro A J, Akira S, et al. Noninvasive detection of macrophage activation with single-cell resolution through machine learning[J]. Proceedings of the National Academy of Sciences, 2018, 115(12): E2676-E2685.
7.Du J, Su Y, Qian C, et al. Raman-guided subcellular pharmaco-metabolomics for metastatic melanoma cells[J]. Nature communications, 2020, 11(1): 1-16.
8.Ho C S, Jean N, Hogan C A, et al. Rapid identification of pathogenic bacteria using Raman spectroscopy and deep learning[J]. Nature communications, 2019, 10(1): 4927.
```

## Usage
1. You should install the AggMap tool first. Follow the installation instruction of AggMap.For more information, please visit https://github.com/shenwanxiang/bidd-aggmap
2. Preprocess your data, make it into .csv format. The feature should be the wavenumber, and the feature value should be the intensity of the wavenumber.
3. Follow the steps in 1_model_training.ipynb.
```python
from aggmap import AggMap
import tensorflow as tf
import os
import pandas as pd
import numpy as np

from utils.dscarnet import dual_dscarnet
from utils.PCA_tool import PCA_transform
#load data
df = pd.read_csv('your_path_X.csv')
df_pca = PCA_transform(df,how_many_PCs_you_need)
#2D transformation
mp_sar = AggMap(df,metric = 'euclidean')
mp_sar = mp_sar.fit(cluster_channels = 9, verbose = 0)
mp_car = AggMap(df_pca,metric = 'euclidean')
mp_car = mp_car.fit(cluster_channels = 9, verbose = 0)
X1 = mp_sar.batch_transform(df.values,scale_method = 'minmax')
X2 = mp_car.batch_transform(df_pca.values,scale_method = 'minmax')
#model training
Y = pd.read_csv('your_path_Y.csv')
Y = pd.get_dummies(Y).values
#5-FCV example
outer = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 8)
outer_idx = list(outer.split(X1,df['label']))
for i, idx in enumerate(outer_idx):
    
    train_idx, valid_idx = idx

    validY = Y[valid_idx]
    validX = X1[valid_idx],X2[valid_idx]

    trainY = Y[train_idx]
    trainX = X1[train_idx],X2[train_idx]
    
    model = dual_dscarnet(X1.shape[1:], X2.shape[1:])
    opt = tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) #
    model.compile(optimizer = opt, loss = 'categorical_crossentropy',metrics=['accuracy'])


    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10,#for example
                                                      restore_best_weights=True)

    model.fit(trainX, trainY,
                              batch_size=128, 
                              epochs= 10,#for example 
                                verbose= 1, shuffle = True, 
                              validation_data = (validX, validY), 
                               callbacks=[early_stopping_cb],)
    break#for example
```