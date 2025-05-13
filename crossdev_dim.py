

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from mycolorpy import colorlist as mcp

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from stacked_mosaic_plot_classification import nclass_classification_mosaic_plot

from PIL import Image
import requests
from transformers import CLIPModel, CLIPProcessor, AutoTokenizer
#from numba import njit, prange
import gc

#gc.collect()
plt.style.use('fivethirtyeight')

# default variables to be used... will figure out best way to handle
#clip_image_features=np.empty([0,0])
#clip_text_features=np.empty([0,0])
#product_ids=[]
#new_index=...
#new_image_class=np.empty(0)
#new_text_class=np.empty(0)
#new_cat_img_features = np.empty([0,0])
#new_cat_text_features = np.empty([0,0])

cr_df = pd.DataFrame()
# ## Load data

# In[2]:

def load_data():
    fileloc = 'shopping-queries-image-dataset\sqid\product_features.parquet'
    product_embeddings = pd.read_parquet(fileloc)
    product_embeddings = product_embeddings.dropna()
    clip_image_features = np.array(product_embeddings['clip_image_features'].to_list())
    clip_text_features = np.array(product_embeddings['clip_text_features'].to_list())
    product_ids = product_embeddings.iloc[:,0].to_list()
    #del product_embeddings
    img_db = pd.read_csv('shopping-queries-image-dataset/sqid/product_image_urls.csv')
    if False: #this is a bit of memory hog, not used for calc, just reference so have line by line loader
        product_descriptions = pd.read_parquet("esci-data/shopping_queries_dataset/shopping_queries_dataset_products.parquet")
        product_descriptions = product_descriptions[product_descriptions['product_id'].isin(product_ids)]

    _ = gc.collect()
    return clip_image_features, clip_text_features, img_db, product_ids


def get_product_description(product_id):
    fileloc= "esci-data/shopping_queries_dataset/shopping_queries_dataset_products.parquet"
    if type(product_id) != list:
        product_id=[product_id]
    sel = [[("product_id", "==", _id)] for _id in product_id]
    prod = pd.read_parquet(fileloc,engine='pyarrow',filters=sel)
    return prod



# ## Initialize CLIP model and tokenizer
def load_model(clip_model = "openai/clip-vit-large-patch14", token_model="openai/clip-vit-large-patch14"):
    model = CLIPModel.from_pretrained(clip_model)
    tokenizer = AutoTokenizer.from_pretrained(token_model)
    return model, tokenizer



# ## crossvar and other methods


def calc_category_features(image_category_prompts,text_category_prompts,tokenizer, model):
    features = []
    for prompt in [image_category_prompts,text_category_prompts]:
        inputs = tokenizer(prompt, padding=True, return_tensors="pt")
        features.append(model.get_text_features(**inputs).detach().numpy())
    return features

def add_undecided(cosims, zero_threshold=0.0,min_cosim=0.0):
    #undecided = ((np.abs(cosims[0]-cosims[1])<zero_threshold)*1)
    cosim_sorted = np.sort(cosims,axis=0)
    zero_threshold_mask=(np.abs(cosim_sorted[-1,:]-cosim_sorted[-2,:])<zero_threshold)
    min_cosim_mask = (cosims[:2,:].max(axis=0)<min_cosim)
    undecided = (zero_threshold_mask|min_cosim_mask)*1
    return np.vstack([cosims,undecided])

#@njit
def cross_variance(arr_a, arr_b, c_type='ratio'):
    """c_type: {var: variance, std: standard deviation, ratio: sqrt(cross_var/pearson product)
    I'm using root ratio of cross dev (swapped dev) to pearson denominator (sigma_a*sigma_b)

    calc_method: which method for calculating cross var... keeping for progression, but 'ms_sm' is unequivically better
    ms_sm: mean of squares - square of means.  no matrices, based on crossvar proof (analagous to variance proof)
    einsum: np.einsum version... still rectangular matrix calc, but faster than np_broadcast
    np_broadcast: numpy broadcasting.  faster than iterating, but still rectangular matrix calc
    """
    ret_dict = {}
    calc_method = 'ms_sm'     
    if calc_method == 'ms_sm':
        a_mean, b_mean = arr_a.mean(axis=0), arr_b.mean(axis=0)
        a_sqr_mean, b_sqr_mean = np.power(arr_a,2).mean(axis=0) , np.power(arr_b,2).mean(axis=0) 
        cv = (a_sqr_mean+b_sqr_mean)/2 - a_mean*b_mean
        ret_dict['var']=cv
    if calc_method == 'einsum':
        # the following is a 50% faster way to calc crossvar, but harder to understand... see the commented code below
        difference = (arr_a[:] - arr_b[:,None])
        ret_dict['var'] = np.einsum('ij...,ij...->...', difference, difference) /(len(arr_a)*len(arr_b)*2)
    if calc_method == 'np_broadcast':
        # the code below is more straight forward to understand... broadcast, take difference, square, take mean, div 2
        ret_dict['var'] = np.power((arr_a[:] - arr_b[:,None]),2).mean(axis=(0,1))/2
    
    ret_dict['std'] = np.sqrt(ret_dict['var'])
    ret_dict['ratio'] = np.sqrt( ret_dict['var']/(arr_a.std(axis=0)*arr_b.std(axis=0)) )
    return ret_dict[c_type]

import contextlib

@contextlib.contextmanager
def clear_memory():
    try:
        yield
    finally:
        gc.collect()

    # your code here
    
def ordered_sampling(arr_a,bins = 1000):
    """pulls sample from an ordered set of values per feature.  More likely to get a representative distribution
    """
    if bins>=len(arr_a):
        return arr_a
    binsize = int(np.round(n)/bins)
    padding = min(binsize-n%binsize,binsize-1)
    sort_a = np.sort(np.pad(arr_a,[(0,padding),(0,0)],mode='constant'),axis=0)
    binsize = int((len(sort_a)/bins))
    bins = int(len(sort_a)/binsize)
    sample_ix = (np.random.choice(np.arange(binsize),bins)+np.arange(0,n,binsize))
    sample = sort_a[sample_ix]
    print(sample.shape)
    return sample


def cross_dev_ratio(
feature_array
, class_assignment = ''#image_class
,a_targ = 1 #subcat_dict['shoes, handbags & sunglasses, specifically a shoes']
,b_targ = 2 #subcat_dict['shoes, handbags & sunglasses, specifically a  sunglasses']
,num_samples = 1500
,pairwise=False
,plot=True
):
    """feature_array = clip_image_features"""
    if pairwise:
        a_features = feature_array[class_assignment == a_targ]
        b_features = feature_array[class_assignment == b_targ]
    else: #one vs rest of distribution
        a_features = feature_array[class_assignment == a_targ]
        b_features = feature_array[class_assignment != a_targ]

    #simple sampling
    small_a =a_features[np.random.choice(np.arange(a_features.shape[0]),min(num_samples,a_features.shape[0])),:]
    small_b =b_features[np.random.choice(np.arange(b_features.shape[0]),min(num_samples,b_features.shape[0])),:]
    
    # may not need sampling if using ms_sm method for crossvar
    #small_a = ordered_sampling(a_features,bins = num_samples)
    #small_b = ordered_sampling(b_features,bins = num_samples)
    with clear_memory():
        dev_comp = cross_variance(small_a, small_b, c_type='ratio')
    if plot:
        _ = plt.hist(dev_comp);
        print(sum(dev_comp>1.25))
    return dev_comp


# ### Class Assignment Methods


def create_prompts(category_list):
    cat_str_prompts = [f"{c.strip().lower()}" for c in category_list]
    cat_photo_prompts = [f"a photo of {c.strip().lower()}" for c in category_list]
    cat_dict = {}
    for i, c in enumerate(category_list):
        cat_dict[c]=i
        cat_dict[i]=c
    return cat_photo_prompts, cat_str_prompts, cat_dict

    
def get_base_classes(category_list,clip_image_features, clip_text_features,zero_threshold=0.02, min_cosim=0.0,tokenizer=None, model=None):
    cat_photo_prompt, cat_text_prompt, cat_dict = create_prompts(category_list)
    cat_length = len(category_list)
    cat_dict[cat_length]='undecided'
    cat_dict['undecided']=cat_length
    cat_img_features,cat_text_features = calc_category_features(cat_photo_prompt,cat_text_prompt,tokenizer=tokenizer, model=model)
    base_image_cosim = cosine_similarity(cat_img_features,clip_image_features)
    base_text_cosim = cosine_similarity(cat_text_features,clip_text_features)
    #cosim_sorted = np.sort(base_image_cosim,axis=0)
    #undecided = ((np.abs(cosim_sorted[-1,:]-cosim_sorted[-2,:])<zero_threshold)*1)
    #base_image_cosim = np.vstack([base_image_cosim,undecided])
    base_image_cosim = add_undecided(base_image_cosim, zero_threshold,min_cosim)
    #cosim_sorted = np.sort(base_text_cosim,axis=0)
    #undecided = ((np.abs(cosim_sorted[-1,:]-cosim_sorted[-2,:])<zero_threshold)*1)
    #base_text_cosim = np.vstack([base_text_cosim,undecided])
    base_text_cosim = add_undecided(base_text_cosim, zero_threshold,min_cosim)
    image_class = base_image_cosim.argmax(axis=0)
    text_class = base_text_cosim.argmax(axis=0)
    _ = [print(f"{avg_type} f1 score:\t {f1_score(text_class, image_class,average=avg_type):.2f}") for avg_type in ['micro','macro','weighted']]

    #print(f"f1: {f1_score(text_class==0,image_class==0):.2f} \nimage_mean: {image_class.mean():.2f} text_mean: {text_class.mean():.2f}")
    print(f"undecided pct:\t {(image_class==cat_length).mean():.2f}\t{(text_class==cat_length).mean():.2f}" )
    return base_image_cosim, base_text_cosim, image_class, text_class, cat_img_features,cat_text_features ,cat_dict

def assign_crd_class(dev_comp,dev_comp_text,cat_img_features, clip_image_features,cat_text_features,clip_text_features,
                     img_cutoff= 1.10,txt_cutoff=1.10,zero_threshold=0.00,min_cosim=0.0):
    img_features = np.where(dev_comp>img_cutoff)[0]
    #tar_emb = clip_image_features[:,img_features]
    txt_features = np.where(dev_comp_text>txt_cutoff)[0]
    
    crd_image_cosim = cosine_similarity(cat_img_features[:,img_features],clip_image_features[:,img_features])
    crd_text_cosim = cosine_similarity(cat_text_features[:,txt_features],clip_text_features[:,txt_features])
    crd_image_cosim = add_undecided(crd_image_cosim, zero_threshold,min_cosim)
    crd_text_cosim = add_undecided(crd_text_cosim, zero_threshold,min_cosim)
    
    crd_image_class = crd_image_cosim.argmax(axis=0)
    crd_text_class = crd_text_cosim.argmax(axis=0)
    print(f"f1: {f1_score(crd_text_class==0,crd_image_class==0):.2f}")
    #print pct undecided
    return crd_image_class, crd_text_class, crd_image_cosim, crd_text_cosim

def calc_crossratio_gain(a_targ,b_targ,ix,dev_comp,clip_image_features,
             cat_img_features,
             base_image_cosim, assigned_class):
    """ix=... default"""
    if ix==...:
        ix = (assigned_class==a_targ)|(assigned_class==b_targ)
    A = []
    B = []
    feat_count = []
    arange = np.arange(1,1.4,0.01)
    for cutoff in arange:
        img_features = np.where(dev_comp>cutoff)[0]
        #print(len(img_features))
        
        if len(img_features)==0:
            break
        feat_count.append(len(img_features))
        #cosine_similarity(cat_photo_features[subcat_dict[sb],img_features].reshape(1,-1),clip_image_features[:,img_features])
        tar_emb = clip_image_features[ix][:,img_features]
        if len(tar_emb.shape)==1: tar_emb = tar_emb.reshape(1,-1)
        
        #print(f"A: {cosine_similarity(clothing_img_features[a_targ,img_features].reshape(1,-1),tar_emb)[0,0]:.2f}")
        #print(f"B: {cosine_similarity(clothing_img_features[b_targ,img_features].reshape(1,-1),tar_emb)[0,0]:.2f}")
        A.append(cosine_similarity(cat_img_features[a_targ,img_features].reshape(1,-1),tar_emb).mean())
        B.append(cosine_similarity(cat_img_features[b_targ,img_features].reshape(1,-1),tar_emb).mean())
    arange=arange[:len(A)]
    return A, B, feat_count, arange


# ### Plotting methods

def plot_crossratio(A,B,feat_count, arange):
    feat_df = pd.DataFrame(data=np.array(A),columns=['class_A'])
    feat_df['class_B'] = np.array(B)
    feat_df['feature_count'] = np.array(feat_count)
    feat_df.set_index('feature_count',inplace=True)
    feat_df.head()
    
    fig, [ax1,ax2] = plt.subplots(1,2,figsize=[10,5],sharey=True)
    ax1.plot(arange,np.array([A,B]).T,label=["class_A","class_B"])
    ax1.set_title("relative to crossratio")
    ax1.set_xlabel("CrossRatio",fontsize='medium')
    ax1.set_ylabel("Cosine Similarity",fontsize='medium')
    ax1.legend()
    #ax1.grid()
    
    feat_df[(feat_df.index<100)&(feat_df.index>0)].plot(title="relative to feature count",ax=ax2)
    plt.gca().invert_xaxis()
    fig.suptitle("Within-class Avg Cosim",fontsize='medium')
    plt.tight_layout()


def plot_gain(A,B,feat_count, arange):
    a_targ,b_targ = 0,1
    fig, [ax1,ax2] = plt.subplots(2,1,figsize=[10,7],sharex=True,gridspec_kw={'height_ratios':[3,1]})
    ax1.plot(arange,np.array([A,B]).T,label=["A","B"],alpha=0.5)
    ax1.plot(arange,np.array([np.array(A)-np.array(B)]).T,label=["gain"])
    
    #ax1.hlines(base_image_cosim[a_targ][:].mean()-base_image_cosim[b_targ][:].mean(),xmin=min(arange),xmax=max(arange),label="initial gain")
    ax1.legend()
    ax1.set_ylabel("Cosine Similarity",fontsize='medium')
    ax1.grid(True)
    #plt.show()
    ax2.plot(arange,feat_count,label='feature count');
    ax2.set_ylim(0,150)
    ax2.grid(True)
    ax2.set_ylabel("# Features",fontsize='medium')
    ax2.set_xlabel("CrossRatio",fontsize='medium')
    ax2.legend()
    fig.suptitle("Within-class Avg Cosim\nvs feature crossratio threshold",fontsize='medium')
    plt.tight_layout()

def plot_highlight_features(dev_comp,cutoff=1.1,category='clothing'):
    mask1 = dev_comp >cutoff
    xaxis = np.arange(len(dev_comp))
    fig, ax = plt.subplots(figsize=(10,2))
    ax.bar(np.arange(len(dev_comp))[~mask1],dev_comp[~mask1],color='blue')
    ax.bar(np.arange(len(dev_comp))[mask1],dev_comp[mask1],color='red')
    ax = plt.gca()
    ax.set_ylim([1,dev_comp.max()-0.1]);
    ax.set_title(f"Cross Ratio of Clip Embeddings\nsub-features for '{category}'",fontsize='medium')
    

def add_violin_label(violin, label,labels=[]):
    color = violin["bodies"][0].get_facecolor().flatten()
    labels.append((mpatches.Patch(color=color), label))
    return labels

def plot_violin_crossdev_features(clip_image_features, image_class, dev_comp):
    impt_0 = clip_image_features[image_class==0,np.argmax(dev_comp)][:10000]
    impt_1 = clip_image_features[image_class==1,np.argmax(dev_comp)][:10000]
    
    unimpt_0 = clip_image_features[image_class==0,np.argmin(dev_comp)][:10000]
    unimpt_1 = clip_image_features[image_class==1,np.argmin(dev_comp)][:10000]
    
    labels=[]    
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(10,5))
    ax.set_title("Distribution of features\n by class",fontsize='medium')
    ax.set_ylabel('feature values',fontsize='medium')
    violin_parts = ax.violinplot([impt_0,unimpt_0]);
    for pc in violin_parts['bodies']:
        pc.set_facecolor('blue')
    labels = add_violin_label(violin_parts,'Clothing',labels)
    
    violin_parts = ax.violinplot([impt_1,unimpt_1]);
    for pc in violin_parts['bodies']:
        pc.set_facecolor('red')
    labels = add_violin_label(violin_parts,'Other items',labels)
    xlabels = ['High Crossgain\nFeature','Low Crossgain\nFeature']
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=xlabels);
    plt.legend(*zip(*labels),loc=9)
    plt.tight_layout()

def plot_class_split(base_image_cosim, image_class,class_ix=[0,1],zero_threshold=0.01,ax=None,title=None,cmap='viridis'):
    subset = np.random.choice(np.arange(len(base_image_cosim[0])),10000)
    class_subset = image_class[subset]
    cosim_subset = base_image_cosim[:,subset]
    mask_diff = np.abs(cosim_subset[class_ix[0],:]-cosim_subset[class_ix[1],:])>zero_threshold
    mask = np.where(class_subset,cosim_subset[class_ix[0],:]<np.median(cosim_subset[class_ix[0],:]),
                    cosim_subset[class_ix[0],:]>np.median(cosim_subset[class_ix[0],:]))
    #mask1 = mask_diff*(cosim_subset[0,:]<np.median(cosim_subset[0,:]))*(class_subset==1)
    #mask0 = mask_diff*(cosim_subset[0,:]>np.median(cosim_subset[0,:]))*(class_subset==0)
    mask1 = mask_diff*(class_subset==1)
    mask0 = mask_diff*(class_subset==0)
    #subset = np.random.choice(np.arange(np.sum(mask==True)),10000)
    color1=np.array(mcp.gen_color(cmap=cmap,n=4))
    #pallet = color1[np.random.choice(np.arange(len(color1)),n_classes)]
    pallet=color1[:4]

    if ax is None:
        fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(cosim_subset[:,mask1][class_ix[0]],cosim_subset[:,mask1][class_ix[1]],alpha=0.8,label='Non-clothing Class',color=pallet[0]);
    ax.scatter(cosim_subset[:,mask0][class_ix[0]],cosim_subset[:,mask0][class_ix[1]],alpha=0.8,label='Clothing Class',color=pallet[1]);
    
    #subset = np.random.choice(np.arange(np.sum(mask==False)),10000)
    ax.scatter(cosim_subset[:,~mask_diff][class_ix[0]],cosim_subset[:,~mask_diff][class_ix[1]],alpha=0.8,color=pallet[2],label='Excluded: cosim diff');
    ax.scatter(cosim_subset[:,(~mask0*~mask1*mask_diff)][0],cosim_subset[:,(~mask0*~mask1*mask_diff)][1],alpha=0.8,color=pallet[3],label='Excluded: cosim limit');
    if title is None:
        ax.set_title("Clothing Class Assignment:\nisolating class characteristics",fontsize='medium')
    else:
        ax.set_title(title,fontsize='medium')
    ax.set_xlabel('Clothing Cosim',fontsize='medium');
    ax.set_ylabel('Non Clothing Cosim',fontsize='medium')
    #ax.legend(ncol=3,loc='upper center')#bbox_to_anchor=(1.1, 1.05));
    ax.legend(fontsize='medium')
    return ax
    #plt.tight_layout();

def plot_class_splits_both(base_image_cosim, image_class,base_text_cosim, text_class,class_ix=[0,1],zero_threshold=0.01,cmap='viridis'):
    fig, [ax1,ax2] = plt.subplots(1,2,figsize = (10,6))
    plot_class_split(base_image_cosim, image_class,class_ix=class_ix,zero_threshold=zero_threshold,ax=ax1,title='image embeddings',cmap=cmap)
    plot_class_split(base_text_cosim, text_class,class_ix=class_ix,zero_threshold=zero_threshold,ax=ax2,title ='text embeddings',cmap=cmap);
    fig.suptitle("Class Assignment:\nisolating class characteristics",fontsize='medium')
    plt.tight_layout()


# ### Determine which items are likely to clothing via zero shot classification

# ### Calculate crossratio for image and text base class (clothing vs not clothing)


def show_item_details(ix=None,holder={},product_ids=[],return_details=True):
    """holder = {}
    holder['Base Model']=[image_class, text_class,cat_dict]
    holder['CRD Model']=[crd_image_class, crd_text_class, cat_dict]
    """
    plt.style.use('ggplot')
    #for ix in np.random.choice(np.arange(len(product_ids)),1):
    if ix is None:
        ix = np.random.choice(np.arange(len(product_ids)),1)[0]
    product_id = product_ids[ix]
    url = img_db[img_db['product_id']==product_id]['image_url'].values[0]
    image = Image.open(requests.get(url, stream=True).raw)
    
    _img, _txt = [],[]
    keys = list(holder.keys())
    for key in keys:
        image_class, text_class,cat_dict = holder[key]
        
        _img.append(cat_dict[image_class[ix]])
        _txt.append(cat_dict[text_class[ix]])
    
    cols = ['Image Class','Text Class']
    
    fig, axs = plt.subplots(1,2,figsize=(12,4))
    axs[0].imshow(image)
    axs[1].table(cellText=[_img,_txt], colLabels=cols,rowLabels=keys ,loc='best')
    for ax in axs:
        ax.grid(False)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    _ = gc.collect()
    if return_details:
        prod = get_product_description(product_id)
        prod = prod.T
        pd.set_option('display.max_colwidth',150)
        dfStyler = prod.style.set_properties(**{'text-align': 'left'})
        plt.style.use('fivethirtyeight')
        return dfStyler.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])
    else:
        return

def get_class_labels(new_cat_dict):
    return [new_cat_dict[k].split(" ",1)[0] for k in np.sort(np.array([k for k in new_cat_dict.keys() if type(k)==int]))]

def plot_confusion_matrix(new_text_class, new_image_class, new_cat_dict, cmap='coolwarm',title='', ax=None):
    txt_class = [new_cat_dict[i].split(" ",1)[0] for i in new_text_class]
    img_class = [new_cat_dict[i].split(" ",1)[0] for i in new_image_class]
    if ax is None:
        fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(txt_class, img_class,include_values=False,normalize='true',xticks_rotation=45,ax=ax,cmap=cmap);
    #(confusion_matrix, *, display_labels=None)
    ax.set_title(f"Text vs Image Confusion Matrix\n{title}",fontsize='medium');
    ax.set_xlabel("Text Class",fontsize='medium')
    ax.set_ylabel("Image Class",fontsize='medium')
    #multilabel_confusion_matrix(new_text_class, new_image_class)
    return ax

def plot_classification_mosaic(new_image_class, new_text_class, new_cat_dict,cmap='Blues',title='', ax=None):
    n_classes = len(np.unique(new_image_class)) # number of classes
    holder = []
    for k in np.arange(n_classes):
        sub_holder = []
        text_mask = new_text_class==k
        for j in np.arange(n_classes):
            sub_holder.append(int(np.sum(new_image_class[text_mask]==j)))
        holder.append(sub_holder)
    
    #np.arange(np.array([k for k in new_cat_dict.keys() if type(k)==int]).max())
    class_labels = [new_cat_dict[k].split(" ",1)[0] for k in np.sort(np.array([k for k in new_cat_dict.keys() if type(k)==int]))]
    if ax is None:
        fig, ax = plt.subplots()
    nclass_classification_mosaic_plot(n_classes, holder,class_labels=class_labels ,cmap=cmap,ax=ax)
    ax.set_title(f"Classification Mosaic\n{title}",fontsize='medium')
    return ax

def plot_class_hist(image_classes, text_classes, category_dict,title='',ax=None):
    class_labels = get_class_labels(category_dict)
    #plt.hist(pat_classes['img'],bins=len(class_labels))
    bin_size = [int(np.sum(image_classes==c)) for c in np.arange(len(class_labels))]
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,5))

    ax.bar(class_labels, bin_size,label='Image Classes',alpha=0.5,color='red')
    ax.tick_params(axis='x', labelrotation=45)
    bin_size = [int(np.sum(text_classes==c)) for c in np.arange(len(class_labels))]
    ax.bar(class_labels, bin_size,label='Text Classes',alpha=0.5)
    ax.legend()
    ax.set_title(f"Class counts\n{title}",fontsize='medium')
    return ax

def plot_class_count_comp(new_image_class, new_text_class,pat_img_classes,pat_txt_classes,new_cat_dict):
    """pat_classes['img'],pat_classes['txt'] """

    fig, [ax1,ax2] = plt.subplots(1,2,figsize=(12,5))
    
    plot_class_hist(new_image_class, new_text_class, category_dict=new_cat_dict,title='Base Model',ax=ax1)
    plot_class_hist(pat_img_classes, pat_txt_classes, category_dict=new_cat_dict,title='Pairwise Model',ax=ax2);
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax2.tick_params(axis='both', which='major', labelsize=10)

def plot_confusion_comp(new_image_class, new_text_class,pat_img_classes,pat_txt_classes,new_cat_dict):
    """pat_classes['img'],pat_classes['txt'] """
    
    fig, [ax1,ax2] = plt.subplots(1,2,figsize=(12,5),sharey=False)
    
    plot_confusion_matrix(new_text_class, new_image_class, new_cat_dict,cmap='coolwarm',title='Base Model',ax=ax1)
    plot_confusion_matrix(pat_txt_classes,pat_img_classes, new_cat_dict,cmap='coolwarm',title='Pairwise Model',ax=ax2)
    ax2.set_ylabel('')
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax2.tick_params(axis='both', which='major', labelsize=10)

def plot_mosaics_comp(new_image_class, new_text_class,pat_img_classes,pat_txt_classes,new_cat_dict):
    """pat_classes['img'],pat_classes['txt'] """
    fig, [ax1,ax2] = plt.subplots(1,2,figsize=(12,6),sharey=False)
    
    plot_classification_mosaic(new_image_class=new_image_class, new_text_class=new_text_class, new_cat_dict=new_cat_dict,cmap='Blues',ax=ax1,title='Base Model')
    plot_classification_mosaic(new_image_class=pat_img_classes, new_text_class=pat_txt_classes, new_cat_dict=new_cat_dict,cmap='Blues',ax=ax2,title='Pairwise Model')
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax2.tick_params(axis='both', which='major', labelsize=10)
    ax2.set_ylabel('')
    ax1.get_legend().remove();
    plt.tight_layout()


def create_crossratio_df(category_list,
    img_feature_array, img_class_assignment,
    txt_feature_array, txt_class_assignment,
    num_samples=1500):
    """ create dataframe to hold crossratio that is easily referencable by class and image/text
        img_feature_array = clip_image_features[new_index,:], img_class_assignment = new_image_class,
        txt_feature_array = clip_text_features[new_index,:], txt_class_assignment = new_text_class,

    """
    subcat_crossratio = {}
    for i in np.arange(len(category_list)):
        subcat_crossratio[int(i)]={}
        for j in np.arange(i+1,len(category_list)):
            subcat_crossratio[int(i)][int(j)]={}
            a_targ = i #cat_dict['pants']
            b_targ = j #cat_dict['boots']
            dev_comp = cross_dev_ratio(feature_array = img_feature_array, class_assignment = img_class_assignment,a_targ = a_targ,b_targ = b_targ,num_samples = num_samples,pairwise=True,plot=False)
            dev_comp_text = cross_dev_ratio(feature_array = txt_feature_array, class_assignment = txt_class_assignment,a_targ = a_targ,b_targ = b_targ,num_samples = num_samples,pairwise=True,plot=False)
    
            subcat_crossratio[int(i)][int(j)]['img']=dev_comp
            subcat_crossratio[int(i)][int(j)]['txt']=dev_comp_text
    
    cr_list=[]
    #sb_dict = {}
    for i in list(subcat_crossratio.keys()):
        if len(subcat_crossratio[i].keys())>0:
            for j in list(subcat_crossratio[int(i)].keys()):
                for t in ['img','txt']:
                    cr_list.append([i,j,t,subcat_crossratio[i][j][t]])
                    cr_list.append([j,i,t,subcat_crossratio[i][j][t]])
                    #sb_dict[int(i),int(j),t]=subcat_crossratio[i][j][t]
                    #sb_dict[int(j),int(i),t]=subcat_crossratio[i][j][t]
    cr_df = pd.DataFrame(cr_list,columns=['ix_a','ix_b','feat_type','crossratio']).set_index(['ix_a','ix_b','feat_type'])
    return cr_df



# pairwise tourney methods... in reverse order of use... prob combine into a class... prob been done before but this works for my purposes...
# most of what I saw was for individual tests, not arrays.

def create_matchup(a,b):
    """take two arrays of classes, create  the pair-wise 'matchups' in order to compare the sections of the arrays"""
    bu = np.array(list(set(b))) #unique set of classes
    au = np.array(list(set(a)))  
    return np.dstack(np.meshgrid(au, bu)).reshape(-1,2)

def random_winner(matching_ix,matchup):
    winner_binary = np.random.rand(len(matching_ix))>0.5
    winners = np.where(winner_binary,matchup[0],matchup[1])
    return winners


def pairwise_subfeature_test(matching_ix, matchup,
        item_features,
        cat_features,
        cr_df,
        feat_type,
        cutoff):
    """determines winner between two sub arrays based on class assignment
    feeds into pairwise tourney structure
    will get matching_ix, matchup from the tourney parent
    update pairwise_subfeature_inputs (dictionary) before running tourney

            item_features=pairwise_subfeature_inputs['item_features'],
        cat_features=pairwise_subfeature_inputs['cat_features'],
        cr_df=pairwise_subfeature_inputs['cr_df'],
        feat_type=pairwise_subfeature_inputs['feat_type'],
        cutoff=pairwise_subfeature_inputs['cutoff']
    """
    sub_features = cr_df.loc[matchup[0],matchup[1],feat_type]['crossratio']
    sub_features = np.where(sub_features>cutoff)[0]
    #print(len(sub_features))
    _cat = cat_features[matchup][:,sub_features]
    _item = item_features[matching_ix][:,sub_features]
    cosim = cosine_similarity(_item,_cat)
    winners = cosim.argmax(axis=1)
    winners = np.where(winners==True,matchup[1],matchup[0])# logic: where winner==1 (vs 0), then matchup1
    return winners

def determine_round_winner(a,b,test_function=random_winner):
    """take two 1-D arrays of class and create sub arrays of pairwise matchups for all possible matchups
    this is intended to be a round in a tourney style process, but will work as matrix comparison of all vs all
    pass a test function that takes the indices in the paired arrays that match the matchup (matching_ix) and the matchup list, ie [1,4]
    """
    matchups = create_matchup(a,b)
    next_round = np.ones(a.shape,dtype=np.int8)*-1
    for matchup in matchups:
        # find the indices where the classes match the current matchup
        matching_ix = np.where(1*(b==matchup[1])*(a==matchup[0])==1)[0]
        criteria = (len(matching_ix)>0)&(matchup[0]!=matchup[1])
        if criteria:
            # this winner should provide the winning values for the subarray, replace as necessary
            winner = test_function(matching_ix, matchup)#mnp.random.rand(len(matching_ix))>0.5
            next_round[matching_ix]=winner
        # this is where I run the pairwise compare... I'll have to get locations where class matches... then assign that location to the winner
    return next_round


def play_round(current_round, test_function=random_winner):
    next_round = []
    while len(current_round)>=2:
        team_a = current_round.pop()
        team_b = current_round.pop()
        next_round.append(determine_round_winner(team_a,team_b,test_function=test_function))
    if len(current_round)==1: #advance odd number... will work out eventually
        next_round.append(current_round[0])
    return next_round

def pairwise_array_tourney(classes = np.arange(9),num_rows = 15, test_function=random_winner):
    """this will create a log_2 num_rows pairwise tourney to determine winning class/team for each row in an array
    use for simplifying pairwise comparisons of large array
    # pass the unique classes and the length of the target array (number of items/rows)
    """
    current_round = [np.repeat(i,num_rows) for i in np.random.choice(classes,len(classes),replace=False)]
    while len(current_round)>1:
        current_round = play_round(current_round,test_function=test_function)
    return np.array(current_round).squeeze()

from functools import partial


def generate_pairwise_classes(psi_image,psi_text):
    pat_classes = {}
    pairwise_subfeature_inputs = psi_image.copy()
    classes = np.arange(len(pairwise_subfeature_inputs['cat_features']))
    
    pairtest = partial(pairwise_subfeature_test,
        item_features=pairwise_subfeature_inputs['item_features'],
        cat_features=pairwise_subfeature_inputs['cat_features'],
        cr_df=pairwise_subfeature_inputs['cr_df'],
        feat_type=pairwise_subfeature_inputs['feat_type'],
        cutoff=pairwise_subfeature_inputs['cutoff'])
    
    #get image classes
    num_rows=len(pairwise_subfeature_inputs['item_features'])
    pat_classes['img'] = pairwise_array_tourney(classes = classes,num_rows = num_rows, test_function=pairtest)
    #get text classes
    pairwise_subfeature_inputs = psi_text.copy()
    pat_classes['txt'] = pairwise_array_tourney(classes = classes,num_rows = num_rows, test_function=pairtest)
    return pat_classes


# pass the unique classes and the length of the target array (number of items/rows)

#pairwise_array_tourney()
_ = """
psi_image ={
    'item_features':clip_image_features[new_index,:],
    'cat_features':new_cat_img_features,
    'cr_df':cr_df,
    'feat_type':'img',
    'cutoff':1.10}

psi_text={
    'item_features':clip_text_features[new_index,:],
    'cat_features':new_cat_text_features,
    'cr_df':cr_df,
    'feat_type':'txt',
    'cutoff':1.10}
"""

pipeline_description = """
# Pipeline Description
### Load Data & Initialize Model

## Binary classification
### Determine 'clothing' category subset
    Assign category class based on full feature size
    Homogenize classes using higher uncertainty
    Calculate crossratios for image and text classes
    Plot Distribution Characteristics (optional)
    Assign category class based on subfeatures
## Multi-class classification
### Determine subcategory classes for 'clothing' category
    Assign class based on full feature size
    Homogenize classes using higher uncertainty
    Calculate pairwise crossratios
    Assign class based on pairwise tourney
### Describe results
    F1 Chart
    Plots
"""
verbose=False
show_plots=False
category_list  = ['people wearing clothing','general household items']
subcategory_list = ['dress or skirt','footwear such as boots, shoes, slippers',
                        'pants, leggings or jeans','shirt or blouse','jewelry such as earrings, rings, necklaces',
                        'eyewear such as glasses or sunglasses','coat, wrap or shawl','shorts','hat',
                        'bag or purse','phone','belt','socks']

def run_pipeline(category_list=category_list, subcategory_list=subcategory_list, return_payload=False,verbose=False
        ,show_plots=False):
    data_payload = load_data()
    model_payload = load_model(clip_model = "openai/clip-vit-large-patch14", token_model="openai/clip-vit-large-patch14")

    binary_base_model_results, binary_crd_model_results, new_clip_image_features, new_clip_text_features = get_binary_classification(
        category_list=category_list
        ,base_zero_threshold=0.01
        ,base_min_cosim=0.1
        ,crd_zero_threshold=0.005
        ,crd_min_cosim = 0.05
        ,verbose=verbose
        ,show_plots=show_plots
        , data_payload=data_payload
        , model_payload=model_payload)

    pat_classes, cr_df, subclass_base_model_results, subclass_training_model_results = get_multiclass_classification(
        category_list = subcategory_list
        ,base_zero_threshold=0.01
        ,base_min_cosim=0.10
        ,crossratio_img_cutoff=1.10
        ,crossratio_txt_cutoff=1.10
        , clip_image_features=new_clip_image_features
        , clip_text_features=new_clip_text_features
        #,new_index=new_index
        #,binary_base_model_results=binary_base_model_results
        #,binary_crd_model_results=binary_crd_model_results
        )    
    
    show_results(subclass_base_model_results, pat_classes)
    if return_payload:
        pipeline_payload = [data_payload, model_payload,binary_base_model_results, binary_crd_model_results, new_clip_image_features, new_clip_text_features, pat_classes, cr_df, subclass_base_model_results, subclass_training_model_results]
        print("""data_payload, model_payload,binary_base_model_results, binary_crd_model_results, new_clip_image_features, new_clip_text_features, pat_classes, cr_df, subclass_base_model_results, subclass_training_model_results=pipeline_payload""")
        return pipeline_payload 


## Binary classification
def get_binary_classification(
    category_list=['people wearing clothing','general household items']
    ,base_zero_threshold=0.01
    ,base_min_cosim=0.1
    ,crd_zero_threshold=0.005
    ,crd_min_cosim = 0.05
    ,verbose=False
    ,show_plots=False
    , data_payload=None
    , model_payload=None
):
    #unpack data and model payload
    if data_payload is None:
        clip_image_features, clip_text_features, img_db, product_ids = load_data()
    else:
        clip_image_features, clip_text_features, img_db, product_ids = data_payload

    if model_payload is None:
        model, tokenizer = load_model(clip_model = "openai/clip-vit-large-patch14", token_model="openai/clip-vit-large-patch14")
    else:
        model, tokenizer = model_payload
    #"""Assign category class based on full feature size"""
    base_image_cosim, base_text_cosim, image_class, text_class,clothing_img_features,clothing_text_features, cat_dict = get_base_classes(
        category_list,clip_image_features, clip_text_features,zero_threshold=base_zero_threshold, min_cosim=base_min_cosim,tokenizer=tokenizer, model=model) 
    binary_base_model_results =[base_image_cosim, base_text_cosim, image_class, text_class,clothing_img_features,clothing_text_features, cat_dict]

    # Calculate crossratios for image and text classes
    dev_comp = cross_dev_ratio(feature_array = clip_image_features, class_assignment = image_class
    ,a_targ = 0,b_targ = 1,num_samples = 160000,pairwise=True,plot=False)
    dev_comp_text = cross_dev_ratio(feature_array = clip_text_features, class_assignment = text_class
    ,a_targ = 0,b_targ = 1,num_samples = 160000,pairwise=True,plot=False)

    # Assign category class based on subfeatures
    crd_image_class, crd_text_class, crd_image_cosim, crd_text_cosim = assign_crd_class(dev_comp,dev_comp_text,clothing_img_features,
                        clip_image_features,clothing_text_features,clip_text_features,
                        img_cutoff= 1.10,txt_cutoff=1.05,zero_threshold=crd_zero_threshold,min_cosim=crd_min_cosim)
    new_index = (crd_image_class==crd_text_class)&(crd_image_class==0) #assign new classes based on model agreement on class 0
    new_clip_image_features= clip_image_features[new_index,:]
    new_clip_text_features= clip_text_features[new_index,:]

    binary_crd_model_results =[crd_image_cosim, crd_text_cosim, crd_image_class, crd_text_class]

    #All the plots of printouts together
    if verbose: print(f"number of clothing images: {np.sum(image_class==0)}\nagreed clothing items: {np.sum((image_class==text_class)&(image_class==0))}")
    if show_plots: plot_class_splits_both(base_image_cosim, image_class,base_text_cosim, text_class,class_ix=[0,1],zero_threshold=base_zero_threshold,cmap='coolwarm')

    if show_plots: plot_highlight_features(dev_comp,cutoff=1.1,category='clothing image')
    if show_plots: plot_violin_crossdev_features(clip_image_features, image_class, dev_comp)
    if show_plots:
        A, B, feat_count, arange = calc_crossratio_gain(a_targ=0,b_targ=1,ix=...,dev_comp=dev_comp,clip_image_features=clip_image_features,
                    cat_img_features=clothing_img_features,
                    base_image_cosim=base_image_cosim,assigned_class=image_class)

        plot_gain(A,B,feat_count, arange)
        plot_crossratio(A,B,feat_count, arange)

    if verbose: print(clip_image_features[new_index,:].shape)
    if show_plots: plot_class_splits_both(crd_image_cosim, crd_image_class,crd_text_cosim, crd_text_class,class_ix=[0,1],zero_threshold=crd_zero_threshold,cmap='coolwarm')

    return binary_base_model_results, binary_crd_model_results, new_clip_image_features, new_clip_text_features




def get_multiclass_classification(
    category_list = ['dress or skirt','footwear such as boots, shoes, slippers',
                    'pants, leggings or jeans','shirt or blouse','jewelry such as earrings, rings, necklaces',
                    'eyewear such as glasses or sunglasses','coat, wrap or shawl','shorts','hat',
                    'bag or purse','phone','belt','socks']
    ,base_zero_threshold=0.01
    ,base_min_cosim=0.10
    ,crossratio_img_cutoff=1.10
    ,crossratio_txt_cutoff=1.10
    ,clip_image_features=None
    ,clip_text_features=None
    , model_payload=None
    #,new_index = None
    #,binary_base_model_results=None
    #,binary_crd_model_results=None
    ):
    ## Multi-class classification
    if model_payload is None:
        model, tokenizer = load_model(clip_model = "openai/clip-vit-large-patch14", token_model="openai/clip-vit-large-patch14")
    else:
        model, tokenizer = model_payload

    ### Determine subcategory classes for 'clothing' category
    # unpack binary_classification results
    #base_image_cosim, base_text_cosim, image_class, text_class,clothing_img_features,clothing_text_features, cat_dict = binary_base_model_results
    #crd_image_cosim, crd_text_cosim, crd_image_class, crd_text_class = binary_crd_model_results
    
    #    Assign class based on full feature size
    new_image_cosim, new_text_cosim, new_image_class, new_text_class,new_cat_img_features,new_cat_text_features, new_cat_dict = get_base_classes(
        category_list,clip_image_features, clip_text_features,zero_threshold=0,min_cosim=0, tokenizer=tokenizer, model=model) 
    subclass_base_model_results =[new_image_cosim, new_text_cosim, new_image_class, new_text_class,new_cat_img_features,new_cat_text_features, new_cat_dict]

    #Homogenize classes using higher uncertainty
    new_image_cosim, new_text_cosim, new_image_class, new_text_class,new_cat_img_features,new_cat_text_features, new_cat_dict = get_base_classes(
        category_list,clip_image_features, clip_text_features,zero_threshold=base_zero_threshold,min_cosim=base_min_cosim, tokenizer=tokenizer, model=model) 
    subclass_training_model_results =[new_image_cosim, new_text_cosim, new_image_class, new_text_class,new_cat_img_features,new_cat_text_features, new_cat_dict]
    #Calculate pairwise crossratios
    cr_df = create_crossratio_df(category_list,img_feature_array = clip_image_features,
                                img_class_assignment = new_image_class,
                                txt_feature_array = clip_text_features,
                                txt_class_assignment = new_text_class)
    #Assign class based on pairwise tourney
    psi_image ={
        'item_features':clip_image_features,
        'cat_features':new_cat_img_features,
        'cr_df':cr_df,
        'feat_type':'img',
        'cutoff':crossratio_img_cutoff}

    psi_text={
        'item_features':clip_text_features,
        'cat_features':new_cat_text_features,
        'cr_df':cr_df,
        'feat_type':'txt',
        'cutoff':crossratio_txt_cutoff}

    pat_classes = generate_pairwise_classes(psi_image,psi_text)
    print("""unpack commands:
           new_image_cosim, new_text_cosim, new_image_class, new_text_class,new_cat_img_features,new_cat_text_features, new_cat_dict = subclass_base_model_results
          new_image_cosim, new_text_cosim, new_image_class, new_text_class,new_cat_img_features,new_cat_text_features, new_cat_dict = subclass_training_model_results
          """)
    return pat_classes, cr_df, subclass_base_model_results, subclass_training_model_results


def show_results(subclass_base_model_results, pat_classes):
    new_image_cosim, new_text_cosim, new_image_class, new_text_class,new_cat_img_features,new_cat_text_features, new_cat_dict=subclass_base_model_results
    plot_class_count_comp(new_image_class, new_text_class,pat_classes['img'],pat_classes['txt'],new_cat_dict)
    plot_confusion_comp(new_image_class, new_text_class,pat_classes['img'],pat_classes['txt'],new_cat_dict)
    plot_mosaics_comp(new_image_class, new_text_class,pat_classes['img'],pat_classes['txt'],new_cat_dict)




_ = """
    # Long version
    ## Binary classification

    #Assign category class based on full feature size
    category_list=['people wearing clothing','general household items']
    zero_threshold=0.01
    min_cosim=0.1
    base_image_cosim, base_text_cosim, image_class, text_class,clothing_img_features,clothing_text_features, cat_dict = get_base_classes(
        category_list,clip_image_features, clip_text_features,zero_threshold=zero_threshold, min_cosim=min_cosim) 
    if verbose: print(f"number of clothing images: {np.sum(image_class==0)}\nagreed clothing items: {np.sum((image_class==text_class)&(image_class==0))}")
    if show_plots: plot_class_splits_both(base_image_cosim, image_class,base_text_cosim, text_class,class_ix=[0,1],zero_threshold=zero_threshold,cmap='coolwarm')

    # Calculate crossratios for image and text classes
    dev_comp = cross_dev_ratio(feature_array = clip_image_features, class_assignment = image_class
    ,a_targ = 0,b_targ = 1,num_samples = 160000,pairwise=True,plot=False)
    if show_plots: plot_highlight_features(dev_comp,cutoff=1.1,category='clothing image')
    if show_plots: plot_violin_crossdev_features(clip_image_features, image_class, dev_comp)
    if show_plots:
        A, B, feat_count, arange = calc_crossratio_gain(a_targ=0,b_targ=1,ix=...,dev_comp=dev_comp,clip_image_features=clip_image_features,
                    cat_img_features=clothing_img_features,
                    base_image_cosim=base_image_cosim,assigned_class=image_class)

        plot_gain(A,B,feat_count, arange)
        plot_crossratio(A,B,feat_count, arange)

    dev_comp_text = cross_dev_ratio(feature_array = clip_text_features, class_assignment = text_class
    ,a_targ = 0,b_targ = 1,num_samples = 160000,pairwise=True,plot=False)

    if show_plots: plot_highlight_features(dev_comp_text,cutoff=1.1,category='clothing')
    if show_plots: plot_violin_crossdev_features(clip_text_features, text_class, dev_comp_text)
    if show_plots:
        A, B, feat_count, arange = calc_crossratio_gain(a_targ=0,b_targ=0,ix=...,dev_comp=dev_comp_text,clip_image_features=clip_text_features,
                cat_img_features=clothing_img_features,
                base_image_cosim=base_text_cosim,assigned_class=text_class)

        plot_gain(A,B,feat_count, arange)
        plot_crossratio(A,B,feat_count, arange)

    # Assign category class based on subfeatures
    zero_threshold=0.005
    min_cosim = 0.05
    crd_image_class, crd_text_class, crd_image_cosim, crd_text_cosim = assign_crd_class(dev_comp,dev_comp_text,clothing_img_features,
                        clip_image_features,clothing_text_features,clip_text_features,
                        img_cutoff= 1.10,txt_cutoff=1.05,zero_threshold=zero_threshold,min_cosim=min_cosim)

    new_index = (crd_image_class==crd_text_class)&(crd_image_class==0)
    if verbose: print(clip_image_features[new_index,:].shape)
    if show_plots: plot_class_splits_both(crd_image_cosim, crd_image_class,crd_text_cosim, crd_text_class,class_ix=[0,1],zero_threshold=zero_threshold,cmap='coolwarm')
"""


