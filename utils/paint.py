import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def get_color_dict(mp):
    df = mp._S.df
    return df.set_index('Subtypes')['colors'].to_dict()


def show_fmap(mp, X, fax,get_line = True,theme='dark', color_custom_dic=None,show_legend=False):
    if color_custom_dic is None:
        mp_colors = get_color_dict(mp)
    else:
        mp_colors = color_custom_dic

    channels = [i for i in mp.colormaps.keys() if i in mp._S.channels ]
             
    
    for i, j  in enumerate(channels):

        data = X[:,:,mp._S.channels.index(j)]
        min_v = data.max()/3
        color = mp_colors[j]
        if theme == 'dark':
            cmap = sns.dark_palette(color, n_colors =  100, reverse=False)
        else:
            cmap = sns.light_palette(color, n_colors =  100, reverse=False)
        if get_line == True:
            sns.heatmap(
                        np.where(data !=0, data, np.nan), 
                        cmap = cmap, 
                        yticklabels=False, xticklabels=False, cbar=False, 
                        linewidths=0.005,
                        linecolor = '0.9',
                        ax = fax)
        else:
            sns.heatmap(
                        np.where(data !=0, data, np.nan), 
                        cmap = cmap, 
                        yticklabels=False, xticklabels=False, cbar=False, 
                           ax = fax)

    fax.axhline(y=0, color='grey',lw=2, ls =  '--')
    fax.axvline(x=data.shape[1], color='grey',lw=2, ls =  '--')
    fax.axhline(y=data.shape[0], color='grey',lw=2, ls =  '--')
    fax.axvline(x=0, color='grey',lw=2, ls =  '--')
    mp_colors['NaN'] = '#000000'
    patches = [ plt.plot([],[], marker="s", ms=8, ls="", mec=None, color=j, 
                label=i)[0]  for i,j in mp_colors.items() if i in channels]
   
    l = 1.45

    if show_legend ==True:
        fax.legend(handles=patches, 
                   bbox_to_anchor=(l,1.01), 
                   fontsize = 20,
                   loc='upper right', 
                   ncol=1, facecolor="w", 
                   numpoints=1 
                   
                  )    
    

