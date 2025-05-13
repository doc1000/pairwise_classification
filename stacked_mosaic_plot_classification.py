#!/usr/bin/env python
# coding: utf-8

# # Visualize classification in a mosaic plot
# 
# **Note:** This is not my idea, it is inspired by:
# 
# *Jakob Raymaekers, Peter J. Rousseeuw, Mia Hubert. Visualizing classification results. arXiv:2007.14495 [stat.ML]*
# 
# and
# 
# *Friendly, Michael. "Mosaic Displays for Multi-Way Contingency Tables." Journal of the American Statistical Association, vol. 89, no. 425, 1994, pp. 190–200. JSTOR, www.jstor.org/stable/2291215. Accessed 13 Aug. 2020.*
# 
# **In short:**
# This is an approach of visualizing the results of a multiclass estimator by taking the class distribution into account. The predicted classes are plotted along the y-axis and the class size on the x-axis. That way, one can immediatly see the class distribution in the data set. That makes it easier to judge the performance of the estimator.

# ## Imports

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
from statsmodels.graphics.mosaicplot import mosaic
from matplotlib.patches import Patch
import itertools
from collections import deque
#import matplotlib.colors as mcolors
from mycolorpy import colorlist as mcp


# In[2]:


def nclass_classification_mosaic_plot(n_classes, results,class_labels='',cmap="seismic",ax=None):
    """
    build a mosaic plot from the results of a classification
    
    parameters:
    n_classes: number of classes
    results: results of the prediction in form of an array of arrays
    
    In case of 3 classes the prdiction could look like
    [[10, 2, 4],
     [1, 12, 3],
     [2, 2, 9]
    ]
    where there is one array for each class and each array holds the
    predictions for each class [class 1, class 2, class 3].
    
    This is just a prototype including colors for 6 classes.
    """
    class_lists = [range(n_classes)]*2
    mosaic_tuples = tuple(itertools.product(*class_lists))
    
    res_list = results[0]
    for i, l in enumerate(results):
        if i == 0:
            pass
        else:
            tmp = deque(l)
            tmp.rotate(-i)
            res_list.extend(tmp)
    data = {t:res_list[i] for i,t in enumerate(mosaic_tuples)}

    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 12))
    plt.rcParams.update({'font.size': 16})

    font_color = '#2c3e50'

    color1=np.array(mcp.gen_color(cmap=cmap,n=n_classes))
    #pallet = color1[np.random.choice(np.arange(len(color1)),n_classes)]
    pallet=color1[:n_classes]
    colors = deque(pallet[:n_classes])
    all_colors = []
    for i in range(n_classes):
        if i > 0:
            colors.rotate(-1)
        all_colors.extend(colors)
     
    props = {(str(a), str(b)):{'color':all_colors[i]} for i,(a, b) in enumerate(mosaic_tuples)}

    labelizer = lambda k: ''

    p = mosaic(data, labelizer=labelizer, properties=props, ax=ax)

    title_font_dict = {
        'fontsize': 20,
        'color' : font_color,
    }
    axis_label_font_dict = {
        'fontsize': 16,
        'color' : font_color,
    }

    ax.tick_params(axis = "x", which = "both", bottom = False, top = False)
    ax.axes.yaxis.set_ticks([])
    ax.tick_params(axis='x', which='major', labelsize=14)

    ax.set_title('Classification Mosaic', fontdict=title_font_dict, pad=25)
    ax.set_xlabel('Text Class', fontdict=axis_label_font_dict, labelpad=10)
    ax.set_ylabel('Image Class', fontdict=axis_label_font_dict, labelpad=35)
    if class_labels=='':
        class_labels = range(n_classes)
    legend_elements = [Patch(facecolor=all_colors[i], label='{}'.format(c)) for i,c in enumerate(class_labels)]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1,1.018), fontsize='small')

    #plt.tight_layout()
    #plt.show()


# In[3]:


n_classes = 4 # number of classes
results = [
    [50, 4, 1, 2], # predictions for class 1
    [1, 40, 4, 3], # predictions for class 2
    [3, 2, 30, 1], # predictions for class 3
    [1, 3, 1, 60], # predictions for class 4
]


# In[4]:

if __name__ == "__main__":
    nclass_classification_mosaic_plot(n_classes, results)

