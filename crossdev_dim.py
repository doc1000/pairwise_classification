#!/usr/bin/env python
# coding: utf-8

# In[246]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score

from PIL import Image
import requests
from transformers import CLIPModel, CLIPProcessor, AutoTokenizer
from numba import njit, prange


# In[256]:


fileloc = 'shopping-queries-image-dataset\sqid\product_features.parquet'
product_embeddings = pd.read_parquet(fileloc)
product_embeddings = product_embeddings.dropna()
clip_image_features = np.array(product_embeddings['clip_image_features'].to_list())
clip_text_features = np.array(product_embeddings['clip_text_features'].to_list())


# In[257]:


img_db = pd.read_csv('shopping-queries-image-dataset/sqid/product_image_urls.csv')
img_db.head()


# In[258]:


model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")

tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")


# In[336]:


def calc_category_features(image_category_prompts,text_category_prompts,tokenizer=tokenizer, model=model):
    features = []
    for prompt in [image_category_prompts,text_category_prompts]:
        inputs = tokenizer(prompt, padding=True, return_tensors="pt")
        features.append(model.get_text_features(**inputs).detach().numpy())
    return features

def add_undecided(cosims, zero_threshold):
    undecided = ((np.abs(cosims[0]-cosims[1])<zero_threshold)*1)
    return np.vstack([cosims,undecided])

#@njit
def cross_variance(arr_a, arr_b, c_type='ratio'):
    """c_type: {var: variance, std: standard deviation, ratio: sqrt(cross_var/pearson product)
    I'm using root ratio of cross dev (swapped dev) to pearson denominator (sigma_a*sigma_b)
    """
    ret_dict = {}
    # the following is a 50% faster way to calc crossvar, but harder to understand... see the commented code below
    difference = (arr_a[:] - arr_b[:,None])
    ret_dict['var'] = np.einsum('ij...,ij...->...', difference, difference) /(len(arr_a)*len(arr_b)*2)
    # the code below is more straight forward to understand... broadcast, take difference, square, take mean, div 2
    #ret_dict['var'] = np.power((arr_a[:] - arr_b[:,None]),2).mean(axis=(0,1))/2
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


def cross_dev_ratio(
feature_array = clip_image_features
, class_assignment = ''#image_class
,a_targ = 1 #subcat_dict['shoes, handbags & sunglasses, specifically a shoes']
,b_targ = 2 #subcat_dict['shoes, handbags & sunglasses, specifically a  sunglasses']
,num_samples = 1500
,pairwise=False
,plot=True
):
    if pairwise:
        a_features = feature_array[class_assignment == a_targ]
        b_features = feature_array[class_assignment == b_targ]
    else: #one vs rest of distribution
        a_features = feature_array[class_assignment == a_targ]
        b_features = feature_array[class_assignment != a_targ]
        
    #(a_features-b_features[:,np.newaxis]).shape
    
    small_a =a_features[np.random.choice(np.arange(a_features.shape[0]),min(num_samples,a_features.shape[0])),:]
    small_b =b_features[np.random.choice(np.arange(b_features.shape[0]),min(num_samples,b_features.shape[0])),:]
    
    #small_a = a_features[:2000,:]
    #small_b = b_features[:2000,:]
    #(a_features[:10,:5]-a_features[np.newaxis,:11,:5]).shape
    #broadcast to get cross dev values... shape should be number of dimensions
    #cross_var = cross_variance(small_a, small_b, c_type='var')
    #dev_comp = np.sqrt(cross_var/(small_a.std(axis=0)*small_b.std(axis=0)))
    with clear_memory():
        dev_comp = cross_variance(small_a, small_b, c_type='ratio')
    if plot:
        _ = plt.hist(dev_comp);
        print(sum(dev_comp>1.25))
    return dev_comp


# In[313]:


_ = """
arr_a = clip_image_features[:1000]
arr_b = clip_image_features[1000:2000]
arr_a_ix = np.arange(len(arr_a))
arr_b_ix = np.arange(len(arr_b))
#broadcasting
#np.power((arr_a[:] - arr_b[:,None]),2).mean(axis=(0,1))/2

def cartesian_product_calc(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    score = np.empty([len(a) for a in arrays] + [la], dtype=dtype)

    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
        arr[...,i] = a
    return arr.reshape(-1, la)

def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

#%timeit cross_variance(a_features, b_features, c_type='var').shape
cp = cartesian_product(*(arr_a_ix,arr_b_ix))

#%timeit np.power(arr_a[cp.T[0],:] - arr_b[cp.T[1],:],2).mean(axis=0)/2
%timeit np.power((arr_a[:] - arr_b[:,None]),2).mean(axis=(0,1))/2
#%timeit f(arr_a, arr_b,cp).mean(axis=0)/2
#%timeit k(arr_a, arr_b).mean(axis=0)/2
%timeit tile_cross(arr_a,arr_b)

from numba import njit, prange

@njit
def f(arr_a, arr_b,cp):
    return np.power(arr_a[cp.T[0],:] - arr_b[cp.T[1],:],2)#.sum(axis=0)[:10]/2

@njit
def k(arr_a, arr_b):
    return np.power((arr_a[:] - arr_b[:,None]),2)

test = f(arr_a, arr_b,cp).mean(axis=0)/2
test.shape
"""



# In[261]:


# need default variables... could pull them out on refactor
#dev_comp=''
#clip_image_features=''
#cat_img_features='',
#base_image_cosim=''#base_image_cosim
#assigned_class=''#image_class
def plot_gain(a_targ,b_targ,ix,dev_comp,clip_image_features,
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
    fig, [ax1,ax2] = plt.subplots(2,1,figsize=[10,7],sharex=True)
    ax1.plot(arange,np.array([A,B]).T,label=["A","B"],alpha=0.5)
    ax1.plot(arange,np.array([np.array(A)-np.array(B)]).T,label=["gain"])
    
    ax1.hlines(base_image_cosim[a_targ][ix].mean()-base_image_cosim[b_targ][ix].mean(),xmin=min(arange),xmax=max(arange),label="initial gain")
    ax1.legend()
    ax1.grid()
    #plt.show()
    ax2.plot(arange,feat_count,label='feature count');
    ax2.set_ylim(0,100)
    ax2.grid()
    ax2.legend()
    plt.tight_layout()


# In[292]:


#category_list=['a photo of clothing']
subcat_list = ['dress or skirt','footwear such as boots, shoes, slippers',
                   'pants, leggings or jeans','shirt or blouse','jewelry such as earrings, rings, necklaces',
                   'eyewear such as glasses or sunglasses','coat, wrap or shawl','shorts','hat']
def create_prompts(category_list):
    cat_str_prompts = [f"{c.strip().lower()}" for c in category_list]
    cat_photo_prompts = [f"a photo of {c.strip().lower()}" for c in category_list]
    cat_dict = {}
    for i, c in enumerate(category_list):
        cat_dict[c]=i
        cat_dict[i]=c
    return cat_photo_prompts, cat_str_prompts, cat_dict


# In[293]:


#category_list=['clothing, shoes or jewelry','not clothing, not shoes, not jewelry']
category_list=['people wearing clothing','general household items']
cat_photo_prompt, cat_text_prompt, cat_dict = create_prompts(category_list)
cat_dict[2]='undecided'
cat_dict['undecided']=2
clothing_img_features,clothing_text_features = calc_category_features(cat_photo_prompt,cat_text_prompt,tokenizer=tokenizer, model=model)
base_image_cosim = cosine_similarity(clothing_img_features,clip_image_features)
base_text_cosim = cosine_similarity(clothing_text_features,clip_text_features)
zero_threshold=0.03
undecided = ((np.abs(base_image_cosim[0]-base_image_cosim[1])<zero_threshold)*1)
base_image_cosim = np.vstack([base_image_cosim,undecided])
undecided = ((np.abs(base_text_cosim[0]-base_text_cosim[1])<zero_threshold)*1)
base_text_cosim = np.vstack([base_text_cosim,undecided])

#base_image_cosim[0]-=zero_threshold
#base_text_cosim[0]-=zero_threshold
image_class = base_image_cosim.argmax(axis=0)
text_class = base_text_cosim.argmax(axis=0)
print(f"f1: {f1_score(text_class==0,image_class==0):.2f} \nimage_mean: {image_class.mean():.2f} text_mean: {text_class.mean():.2f}")
(image_class==2).mean(),(text_class==2).mean() 


# In[309]:


a_targ = 0 #cat_dict['pants']
b_targ = 1 #cat_dict['boots']
dev_comp = cross_dev_ratio(feature_array = clip_image_features, class_assignment = image_class
,a_targ = a_targ,b_targ = b_targ,num_samples = 2000,pairwise=True,plot=False)

dev_comp_text = cross_dev_ratio(feature_array = clip_text_features, class_assignment = text_class
,a_targ = a_targ,b_targ = b_targ,num_samples = 2000,pairwise=True,plot=False)

_ = """
plot_gain(a_targ,b_targ,ix=...,dev_comp=dev_comp,clip_image_features=clip_image_features,
             cat_img_features=clothing_img_features,
             base_image_cosim=base_image_cosim,assigned_class=image_class)

plot_gain(a_targ,b_targ,ix=...,dev_comp=dev_comp_text,clip_image_features=clip_text_features,
             cat_img_features=clothing_img_features,
             base_image_cosim=base_text_cosim,assigned_class=text_class)
"""


# In[310]:


def assign_crd_class(dev_comp,dev_comp_text,cat_img_features, clip_image_features,cat_text_features,clip_text_features,
                     img_cutoff= 1.10,txt_cutoff=1.10,zero_threshold=0.00):
    img_features = np.where(dev_comp>img_cutoff)[0]
    #tar_emb = clip_image_features[:,img_features]
    txt_features = np.where(dev_comp_text>txt_cutoff)[0]
    
    crd_image_cosim = cosine_similarity(cat_img_features[:,img_features],clip_image_features[:,img_features])
    crd_text_cosim = cosine_similarity(cat_text_features[:,txt_features],clip_text_features[:,txt_features])

    # I could use the one below to make switching easier/harder
    #crd_image_cosim[0]-=zero_threshold
    #crd_text_cosim[0]-=zero_threshold
    crd_image_cosim = add_undecided(crd_image_cosim, zero_threshold)
    crd_text_cosim = add_undecided(crd_text_cosim, zero_threshold)
    
    crd_image_class = crd_image_cosim.argmax(axis=0)
    crd_text_class = crd_text_cosim.argmax(axis=0)
    print(f"img/txt agree:\t crd: {(crd_image_class==crd_text_class).mean():.2f}\tbase: {(image_class==text_class).mean():.2f}")
    print(f"image:\t\t crd: {crd_image_class.mean():.2f}\tbase: {image_class.mean():.2f}")
    print(f"text:\t\t crd: {crd_text_class.mean():.2f}\tbase: {text_class.mean():.2f}")
    print(f"f1: {f1_score(crd_text_class==0,crd_image_class==0):.2f}")
    #print pct undecided
    return crd_image_class, crd_text_class, crd_image_cosim, crd_text_cosim

crd_image_class, crd_text_class, crd_image_cosim, crd_text_cosim = assign_crd_class(dev_comp,dev_comp_text,clothing_img_features,
                    clip_image_features,clothing_text_features,clip_text_features,
                     img_cutoff= 1.10,txt_cutoff=1.10,zero_threshold=0.00)


# In[311]:


new_index = (crd_image_class==crd_text_class)&(crd_image_class==0)
clip_image_features[new_index,:].shape


# In[312]:


import gc
gc.collect()


# ### narrow inputs to just clothing
# find the items where text and images agree that they are clothing

# In[315]:


ix = 5000
for ix in np.random.choice(np.arange(len(product_embeddings)),3):
    product_id = product_embeddings.iloc[ix,0]
    url = img_db[img_db['product_id']==product_id]['image_url'].values[0]
    
    image = Image.open(requests.get(url, stream=True).raw)
    plt.imshow(image);
    print(f"image cdev:\t {crd_image_cosim[crd_image_class[ix],ix]:.2f}\t {cat_dict[crd_image_cosim.argmax(axis=0)[ix]]}")
    print(f"text cdev:\t {crd_text_cosim[crd_text_class[ix],ix]:.2f}\t {cat_dict[crd_text_cosim.argmax(axis=0)[ix]]}")
    print(f"image full:\t {base_image_cosim[image_class[ix],ix]:.2f}\t {cat_dict[image_class[ix]]}")
    print(f"text full:\t {base_text_cosim[text_class[ix],ix]:.2f}\t {cat_dict[text_class[ix]]}")
    
    print(crd_image_cosim[0,ix],crd_image_cosim[1,ix])
    print(crd_text_cosim[0,ix],crd_text_cosim[1,ix])
    plt.show()
#products[products['product_id']==product_id]


# In[319]:


category_list=['people wearing clothing','general household items'] #add more variaty here
category_list = ['dress or skirt','footwear such as boots, shoes, slippers',
                   'pants, leggings or jeans','shirt or blouse','jewelry such as earrings, rings, necklaces',
                   'eyewear such as glasses or sunglasses','coat, wrap or shawl','shorts','hat',
                'bag or purse','phone','belt','socks']
def get_base_classes(category_list,clip_image_features, clip_text_features,zero_threshold=0.02):
    cat_photo_prompt, cat_text_prompt, cat_dict = create_prompts(category_list)
    cat_length = len(category_list)
    cat_dict[cat_length]='undecided'
    cat_dict['undecided']=cat_length
    cat_img_features,cat_text_features = calc_category_features(cat_photo_prompt,cat_text_prompt,tokenizer=tokenizer, model=model)
    base_image_cosim = cosine_similarity(cat_img_features,clip_image_features)
    base_text_cosim = cosine_similarity(cat_text_features,clip_text_features)
    
    undecided = ((np.abs(base_image_cosim[0]-base_image_cosim[1])<zero_threshold)*1)
    base_image_cosim = np.vstack([base_image_cosim,undecided])
    undecided = ((np.abs(base_text_cosim[0]-base_text_cosim[1])<zero_threshold)*1)
    base_text_cosim = np.vstack([base_text_cosim,undecided])
    
    #base_image_cosim[0]-=zero_threshold
    #base_text_cosim[0]-=zero_threshold
    image_class = base_image_cosim.argmax(axis=0)
    text_class = base_text_cosim.argmax(axis=0)
    _ = [print(f"{avg_type} f1 score:\t {f1_score(new_text_class, new_image_class,average=avg_type):.2f}") for avg_type in ['micro','macro']]

    #print(f"f1: {f1_score(text_class==0,image_class==0):.2f} \nimage_mean: {image_class.mean():.2f} text_mean: {text_class.mean():.2f}")
    print(f"undecided pct:\t {(image_class==cat_length).mean():.2f}\t{(text_class==cat_length).mean():.2f}" )
    return base_image_cosim, base_text_cosim, image_class, text_class, cat_img_features,cat_text_features ,cat_dict


new_image_cosim, new_text_cosim, new_image_class, new_text_class,new_cat_img_features,new_cat_text_features, new_cat_dict = get_base_classes(
    category_list,clip_image_features[new_index,:], clip_text_features[new_index,:],zero_threshold=0.02) 


# In[320]:


a_targ = 0 #cat_dict['pants']
b_targ = 1 #cat_dict['boots']
num_samples=1500
print(f"{new_cat_dict[a_targ]}\t {new_cat_dict[b_targ]}")
dev_comp = cross_dev_ratio(feature_array = clip_image_features[new_index,:], class_assignment = new_image_class,a_targ = a_targ,b_targ = b_targ,num_samples = num_samples,pairwise=True,plot=False)
dev_comp_text = cross_dev_ratio(feature_array = clip_text_features[new_index,:], class_assignment = new_text_class,a_targ = a_targ,b_targ = b_targ,num_samples = num_samples,pairwise=True,plot=False)


# In[836]:


plot_gain(a_targ,b_targ,ix=...,dev_comp=dev_comp,clip_image_features=clip_image_features[new_index,:],
             cat_img_features=new_cat_img_features,
             base_image_cosim=new_image_cosim,assigned_class=new_image_class)


# In[321]:


new_crd_image_class, new_crd_text_class, new_crd_image_cosim, new_crd_text_cosim = assign_crd_class(dev_comp,dev_comp_text,new_cat_img_features,
                    clip_image_features[new_index,:],new_cat_text_features,clip_text_features[new_index,:],
                     img_cutoff= 1.1,txt_cutoff=1.1,zero_threshold=0.00)


# In[325]:


f1_score(new_text_class,new_image_class,average='weighted')


# In[359]:


for ix in np.random.choice(np.arange(len(product_embeddings[new_index])),3):
    product_id = product_embeddings[new_index].iloc[ix,0]
    url = img_db[img_db['product_id']==product_id]['image_url'].values[0]
    
    image = Image.open(requests.get(url, stream=True).raw)
    plt.imshow(image);
    print(f"image cdev:\t {new_crd_image_cosim[new_crd_image_class[ix],ix]:.2f}\t {new_cat_dict[new_crd_image_cosim.argmax(axis=0)[ix]]}")
    print(f"text cdev:\t {new_crd_text_cosim[new_crd_text_class[ix],ix]:.2f}\t {new_cat_dict[new_crd_text_cosim.argmax(axis=0)[ix]]}")
    print(f"image full:\t {new_image_cosim[new_image_class[ix],ix]:.2f}\t {new_cat_dict[new_image_class[ix]]}")
    print(f"text full:\t {new_text_cosim[new_text_class[ix],ix]:.2f}\t {new_cat_dict[new_text_class[ix]]}")
    
    print(new_crd_image_cosim[0,ix],new_crd_image_cosim[1,ix])
    print(new_crd_text_cosim[0,ix],new_crd_text_cosim[1,ix])

    print(f"pst text: {new_cat_dict[pat_classes['txt'][ix]]}\npst image: {new_cat_dict[pat_classes['img'][ix]]}")
    plt.show()
#products[products['product_id']==product_id]


# In[1178]:





# In[334]:


def create_crossratio_df(category_list,
    img_feature_array = clip_image_features[new_index,:], img_class_assignment = new_image_class,
    txt_feature_array = clip_text_features[new_index,:], txt_class_assignment = new_text_class,
    num_samples=1500):
    """ create dataframe to hold crossratio that is easily referencable by class and image/text
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


# In[335]:


cr_df = create_crossratio_df(category_list,img_feature_array = clip_image_features[new_index,:], img_class_assignment = new_image_class,txt_feature_array = clip_text_features[new_index,:], txt_class_assignment = new_text_class)


# In[339]:


# pairwise tourney methods... in reverse order of use... prob combine into a class... prob been done before but this works for my purposes...
# most of what I saw was for individual tests, not arrays.
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

pairwise_subfeature_inputs = psi_image.copy()

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
        item_features=pairwise_subfeature_inputs['item_features'],
        cat_features=pairwise_subfeature_inputs['cat_features'],
        cr_df=pairwise_subfeature_inputs['cr_df'],
        feat_type=pairwise_subfeature_inputs['feat_type'],
        cutoff=pairwise_subfeature_inputs['cutoff']):
    """determines winner between two sub arrays based on class assignment
    feeds into pairwise tourney structure
    will get matching_ix, matchup from the tourney parent
    update pairwise_subfeature_inputs (dictionary) before running tourney
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

def generate_pairwise_classes(psi_image,psi_text):
    pat_classes = {}
    pairwise_subfeature_inputs = psi_image.copy()
    classes = np.arange(len(pairwise_subfeature_inputs['cat_features']))
    #get image classes
    num_rows=len(pairwise_subfeature_inputs['item_features'])
    pat_classes['img'] = pairwise_array_tourney(classes = classes,num_rows = num_rows, test_function=pairwise_subfeature_test)
    #get text classes
    pairwise_subfeature_inputs = psi_text.copy()
    pat_classes['txt'] = pairwise_array_tourney(classes = classes,num_rows = num_rows, test_function=pairwise_subfeature_test)
    return pat_classes


# In[346]:


# pass the unique classes and the length of the target array (number of items/rows)

#pairwise_array_tourney()

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


pat_classes = generate_pairwise_classes(psi_image,psi_text)


# In[347]:


print(f1_score(pat_classes['img'],pat_classes['txt'],average='weighted'))
print(f1_score(new_crd_text_class, new_crd_image_class,average='weighted'))
print(f1_score(new_text_class, new_image_class,average='weighted'))

print(f1_score(pat_classes['img'], new_image_class,average='weighted'))
print(f1_score(pat_classes['txt'], new_text_class,average='weighted'))


z = plt.hist(new_text_class)
z = plt.hist(pat_classes['txt'])



# In[ ]:


#save indices of features that matter
cutoff = 1.05
max_features = 100
feature_dict = {}

for feat, class_asn,arr_type in [[clip_image_features,image_class,'image'],[clip_text_features,text_class,'text']]:
    feature_dict[arr_type] = {}
    for sb in subcat_str_list:
        dev_comp = cross_dev_ratio(feat, class_asn,
            a_targ = subcat_dict[sb]
            #,b_targ = subcat_dict['shoes, handbags & sunglasses, specifically a  sunglasses']
            ,num_samples = 1500,pairwise=False,plot=False
            )
        imp_features = np.where(dev_comp>cutoff)[0]
        imp_features = imp_features[dev_comp[imp_features].argsort()][-10:]
        feature_dict[arr_type][sb] = imp_features


# In[ ]:


[(k,v) for k,v in feature_dict['image'].items()]


# In[388]:


img_cosim_holder = []
text_cosim_holder = []

sb = 'clothing & accessories, specifically a  sweaters'
for sb in subcat_str_list:
    img_features = feature_dict['image'][sb]
    text_features = feature_dict['text'][sb]
    
    image_cosim = cosine_similarity(cat_photo_features[subcat_dict[sb],img_features].reshape(1,-1),clip_image_features[:,img_features])#.argmax(axis=0)
    text_cosim = cosine_similarity(cat_text_features[subcat_dict[sb],text_features].reshape(1,-1),clip_text_features[:,text_features])#.argmax(axis=0)
    
    img_cosim_holder.append(image_cosim.squeeze())
    text_cosim_holder.append(text_cosim.squeeze())

#text_class = cosine_similarity(cat_text_features[:,imp_features],clip_text_features[:,imp_features])#.argmax(axis=0)

np.array(img_cosim_holder).shape


# In[389]:


icm = np.array(img_cosim_holder)
tcm = np.array(text_cosim_holder)
np.mean(icm.argmax(axis=0)==tcm.argmax(axis=0) )


# In[337]:


tcm.T[:3,:10]
#tcm.argmax(axis=0).shape


# In[476]:


ix = 1205
product_id = product_embeddings.iloc[ix,0]
url = img_db[img_db['product_id']==product_id]['image_url'].values[0]

image = Image.open(requests.get(url, stream=True).raw)
plt.imshow(image);
print(f"image cdev:\t {icm[image_class[ix],ix]:.2f}\t {subcat_dict[icm.argmax(axis=0)[ix]]}")
print(f"text cdev:\t {tcm[text_class[ix],ix]:.2f}\t {subcat_dict[tcm.argmax(axis=0)[ix]]}")
print(f"image full:\t {base_image_cosim[image_class[ix],ix]:.2f}\t {subcat_dict[image_class[ix]]}")
print(f"text full:\t {base_text_cosim[text_class[ix],ix]:.2f}\t {subcat_dict[text_class[ix]]}")


products[products['product_id']==product_id]


# In[468]:


a_targ = subcat_dict['pants']
b_targ = subcat_dict['boots']

dev_comp = cross_dev_ratio(
feature_array = clip_image_features
, class_assignment = image_class
,a_targ = a_targ
,b_targ = b_targ
,num_samples = 1500
,pairwise=True
,plot=True
)


# In[475]:


img_features = np.where(dev_comp>1.10)[0]
print(len(img_features))
#cosine_similarity(cat_photo_features[subcat_dict[sb],img_features].reshape(1,-1),clip_image_features[:,img_features])
tar_emb = clip_image_features[ix,img_features].reshape(1,-1)
print(f"A: {cosine_similarity(cat_photo_features[a_targ,img_features].reshape(1,-1),tar_emb)[0,0]:.2f}")
print(f"B: {cosine_similarity(cat_photo_features[b_targ,img_features].reshape(1,-1),tar_emb)[0,0]:.2f}")
base_image_cosim[[a_targ,b_targ],ix] 


# In[456]:


product_embeddings.iloc[ix]


# In[443]:


clip_image_features[ix,:10]


# make the above sections into executables and put them in a loop to run against all the categories... pairwise=False
# I can test to see if the matches improve... will this work across the text and image categories to determine if one is better than the other.  something for future.  just get this demod.  overall improvement in matching or in cosine similarity might make sense... we'll see
# 
# so run it, re-classify based on important features.  then see how text/image matching looks.  again, this would be way better with annotated data.  Write up what I have, include some images of classified items.  just a few... nothing exhaustive.
# 
# 

# In[357]:


fileloc = 'esci-data/shopping_queries_dataset/shopping_queries_dataset_products.parquet'
products = pd.read_parquet(fileloc)
products = products[products.product_locale=='us']


# In[358]:


#ls esci-data/shopping_queries_dataset/shopping_queries_dataset_products.parquet
products = products[products.product_locale=='us']

products.iloc[24:33,:]


# In[ ]:


fileloc = 'esci-data/shopping_queries_dataset/shopping_queries_dataset_examples.parquet'
queries = pd.read_parquet(fileloc)


# In[ ]:


queries.iloc[104:112,:]


# In[ ]:


_ = """
cat_file = pd.read_csv('categories_utf.csv')
cat_file.head()
cat_file['Category'].unique()
cat_list = ['Shoes, Handbags & Sunglasses','Clothing & Accessories','Jewelry']
cat_dict = {}

subcat_list = []
for cat in cat_list:
    #cat_file[cat_file['Category'].isin(cat_list)].head()
    sub_list = cat_file[cat_file['Category'].isin([cat])]['Subcategory'].tolist()[0].split(',')
    cat_dict[cat]=sub_list
    #sub_list = [f"{cat}, specifically a {c}" for c in sub_list]
    
    subcat_list.extend(sub_list)
subcat_str_list = [f"{c.strip().lower()}" for c in subcat_list]
subcat_photo_list = [f"a photo of a {c.strip().lower()}" for c in subcat_list]
subcat_str_list"""

