#!/usr/bin/env python
# coding: utf-8

# In[1]:


import anndata
import pandas as pd
import numpy as np
import matplotlib as plt


# In[2]:


import scvelo as scv


# In[3]:


sample_one = anndata.read_loom("/home/henryq/scratch/RNAvelocity/NewFeb4/velocyto/possorted_genome_bam_9B8M4.loom")
sample_two = anndata.read_loom("/home/henryq/scratch/RNAvelocity/Feb3/velocyto/possorted_genome_bam_9OTR3.loom")
sample_three = anndata.read_loom("/home/henryq/scratch/RNAvelocity/ALI/velocyto/possorted_genome_bam_F6SB1.loom")

sample_one.var_names_make_unique()
sample_two.var_names_make_unique()
sample_three.var_names_make_unique()

sample_obs1 = pd.read_csv("/home/henryq/cellID_obssample1.csv")
sample_obs2 = pd.read_csv("/home/henryq/cellID_obssample2.csv")
sample_obs3 = pd.read_csv("/home/henryq/cellID_obssample3.csv")

umap_cord = pd.read_csv("/home/henryq/cell_embeddingsmergenew.csv")
partition_clusters = pd.read_csv("/home/henryq/partitions_cell_embeddings.csv")

sample_one = sample_one[np.isin(sample_one.obs.index,sample_obs1["z"])]
sample_two = sample_two[np.isin(sample_two.obs.index,sample_obs2["z"])]
sample_three = sample_three[np.isin(sample_three.obs.index,sample_obs3["z"])]


sample_one = sample_one.concatenate(sample_two, sample_three)


# In[4]:


sample_one_index= pd.read_csv("/home/henryq/scratch/RNAvelocity/output3new.csv")

umap = umap_cord.rename(columns = {'Unnamed: 0':'Cell ID'})
umap.drop_duplicates(subset='Cell ID',inplace=True)

umap_ordered = sample_one_index.merge(umap, on = "Cell ID")

umap_ordered = umap_ordered.iloc[:,1:]
umap_ordered = umap_ordered.iloc[:,1:]
sample_one.obsm['X_umap'] = umap_ordered.values


# In[5]:


cell_clusters = pd.read_csv("/home/henryq/clusters.csv")
cell_clusters = cell_clusters.rename(columns = {'Unnamed: 0':'Cell ID'})
cell_clusters_ordered = sample_one_index.merge(cell_clusters, on = "Cell ID")
cell_clusters_ordered = cell_clusters_ordered.iloc[:,1:]
cell_clusters_ordered = cell_clusters_ordered.iloc[:,1:]

sample_one.obs['Clusters'] = np.ravel(cell_clusters_ordered.values)


# In[6]:


partition_clusters = partition_clusters.rename(columns = {'Unnamed: 0':'Cell ID'})
partition_clusters_ordered = sample_one_index.merge(partition_clusters, on = "Cell ID")
partition_clusters_ordered = partition_clusters_ordered.iloc[:,1:]
partition_clusters_ordered = partition_clusters_ordered.iloc[:,1:]

sample_one.obs['Partitions'] = np.ravel(partition_clusters_ordered.values)


# In[7]:


sample_one_part1 = sample_one[sample_one.obs['Partitions'] == 1].copy()
sample_one_part2 = sample_one[sample_one.obs['Partitions'] == 2].copy()
sample_one_part3 = sample_one[sample_one.obs['Partitions'] == 3].copy()
sample_one_part4 = sample_one[sample_one.obs['Partitions'] == 4].copy()
sample_one_part5 = sample_one[sample_one.obs['Partitions'] == 5].copy()


# In[8]:


scv.pp.filter_and_normalize(sample_one)
scv.pp.moments(sample_one)
scv.tl.velocity(sample_one, mode = "stochastic")
scv.tl.velocity_graph(sample_one)


# In[8]:


scv.pl.velocity_embedding_stream(sample_one, basis = 'umap')


# In[10]:


scv.pl.velocity_embedding_stream(sample_one, basis = 'umap', color = 'Clusters', figsize=(20,15), legend_loc="right margin", size = 1000,palette = {"NKX2-1hi/SOX9hi/SOX2lo lung progenitors (FL)":"#AA0DFE","Ciliated precursor (FL)":"#3283FE","NKX2-1hi/SOX9lo/SOX2lo lung progenitors (FL)": "#325A9B","Ciliated cells (FL)"  : "#16FF32","Cycling ciliated cells (FL)": "#1CFFCE","NKX2-1hi/SOX9lo/SOX2hi lung progenitors (FL)": "#F8A19F","PDGFRBhi mesenchymal (FL)": "#C4451C","PLAThi mesenchymal (FL)": "#1C8356","Basal cells (FL)":"#E4E1E3", "Secretory precursor (FL)":"#BC8F8F","PNEC (FL)" : "#85660D","Myofibroblasts (ALI)": "#DEA0FD", "Ciliated cells (ALI)":"#FE00FA","EMT (ALI)": "#90AD1C","Brush/PNEC (ALI)": "#CCCC99","Cycling basal cells (ALI)": "#FEAF16","Secretory precursor (ALI)": "#1CBE4F","POSTN2hi mesenchymal (ALI)" : "#B10DA1", "Goblet cells (ALI)":"#2ED9FF","Secretory club cells (ALI)": "#FBE426","Basal cells (ALI)": "#5A5156","Unknown (ALI)" : "#F5DEB3","Cycling ciliated cells (ALI)": "#B00068"}, linewidth = 5, dpi = 300, save = "scVeloS3aALIMerge.pdf")


# In[8]:


scv.pp.filter_and_normalize(sample_one_part1)
scv.pp.moments(sample_one_part1)
scv.tl.velocity(sample_one_part1, mode = "stochastic")
scv.tl.velocity_graph(sample_one_part1)


# In[9]:


scv.pp.filter_and_normalize(sample_one_part2)
scv.pp.moments(sample_one_part2)
scv.tl.velocity(sample_one_part2, mode = "stochastic")
scv.tl.velocity_graph(sample_one_part2)


# In[10]:


scv.pp.filter_and_normalize(sample_one_part4)
scv.pp.moments(sample_one_part4)
scv.tl.velocity(sample_one_part4, mode = "stochastic")
scv.tl.velocity_graph(sample_one_part4)


# In[11]:


scv.pp.filter_and_normalize(sample_one_part3)
scv.pp.moments(sample_one_part3)
scv.tl.velocity(sample_one_part3, mode = "stochastic")
scv.tl.velocity_graph(sample_one_part3)


# In[12]:


scv.pp.filter_and_normalize(sample_one_part5)
scv.pp.moments(sample_one_part5)
scv.tl.velocity(sample_one_part5, mode = "stochastic")
scv.tl.velocity_graph(sample_one_part5)


# In[87]:


scv.pl.velocity_embedding_stream(sample_one_part1, basis = 'umap', color = 'Clusters', figsize=(20,15), legend_loc="right margin", size = 1500,palette = {"NKX2-1hi/SOX9hi/SOX2lo lung progenitors (FL)":"#AA0DFE","Ciliated precursor (FL)":"#3283FE","NKX2-1hi/SOX9lo/SOX2lo lung progenitors (FL)": "#325A9B","Ciliated cells (FL)"  : "#16FF32","Cycling ciliated cells (FL)": "#1CFFCE","NKX2-1hi/SOX9lo/SOX2hi lung progenitors (FL)": "#F8A19F","PDGFRBhi mesenchymal (FL)": "#C4451C","PLAThi mesenchymal (FL)": "#1C8356","Basal cells (FL)":"#E4E1E3", "Secretory precursor (FL)":"#BC8F8F","PNEC (FL)" : "#85660D","Myofibroblasts (ALI)": "#DEA0FD", "Ciliated cells (ALI)":"#FE00FA","EMT (ALI)": "#90AD1C","Brush/PNEC (ALI)": "#CCCC99","Cycling basal cells (ALI)": "#FEAF16","Secretory precursor (ALI)": "#1CBE4F","POSTN2hi mesenchymal (ALI)" : "#B10DA1", "Goblet cells (ALI)":"#2ED9FF","Secretory club cells (ALI)": "#FBE426","Basal cells (ALI)": "#5A5156","Unknown (ALI)" : "#F5DEB3","Cycling ciliated cells (ALI)": "#B00068"}, linewidth = 2, dpi = 300, save = "Partition1scVelo.pdf")


# In[88]:


scv.pl.velocity_embedding_stream(sample_one_part2, basis = 'umap', color = 'Clusters', figsize=(20,15), legend_loc="right margin", size = 1500,palette = {"NKX2-1hi/SOX9hi/SOX2lo lung progenitors (FL)":"#AA0DFE","Ciliated precursor (FL)":"#3283FE","NKX2-1hi/SOX9lo/SOX2lo lung progenitors (FL)": "#325A9B","Ciliated cells (FL)"  : "#16FF32","Cycling ciliated cells (FL)": "#1CFFCE","NKX2-1hi/SOX9lo/SOX2hi lung progenitors (FL)": "#F8A19F","PDGFRBhi mesenchymal (FL)": "#C4451C","PLAThi mesenchymal (FL)": "#1C8356","Basal cells (FL)":"#E4E1E3", "Secretory precursor (FL)":"#BC8F8F","PNEC (FL)" : "#85660D","Myofibroblasts (ALI)": "#DEA0FD", "Ciliated cells (ALI)":"#FE00FA","EMT (ALI)": "#90AD1C","Brush/PNEC (ALI)": "#CCCC99","Cycling basal cells (ALI)": "#FEAF16","Secretory precursor (ALI)": "#1CBE4F","POSTN2hi mesenchymal (ALI)" : "#B10DA1", "Goblet cells (ALI)":"#2ED9FF","Secretory club cells (ALI)": "#FBE426","Basal cells (ALI)": "#5A5156","Unknown (ALI)" : "#F5DEB3","Cycling ciliated cells (ALI)": "#B00068"}, linewidth = 2, dpi = 300, save = "Partition2scVelo.pdf")


# In[89]:


scv.pl.velocity_embedding_stream(sample_one_part3, basis = 'umap', color = 'Clusters', figsize=(20,15), legend_loc="right margin", size = 1500,palette = {"NKX2-1hi/SOX9hi/SOX2lo lung progenitors (FL)":"#AA0DFE","Ciliated precursor (FL)":"#3283FE","NKX2-1hi/SOX9lo/SOX2lo lung progenitors (FL)": "#325A9B","Ciliated cells (FL)"  : "#16FF32","Cycling ciliated cells (FL)": "#1CFFCE","NKX2-1hi/SOX9lo/SOX2hi lung progenitors (FL)": "#F8A19F","PDGFRBhi mesenchymal (FL)": "#C4451C","PLAThi mesenchymal (FL)": "#1C8356","Basal cells (FL)":"#E4E1E3", "Secretory precursor (FL)":"#BC8F8F","PNEC (FL)" : "#85660D","Myofibroblasts (ALI)": "#DEA0FD", "Ciliated cells (ALI)":"#FE00FA","EMT (ALI)": "#90AD1C","Brush/PNEC (ALI)": "#CCCC99","Cycling basal cells (ALI)": "#FEAF16","Secretory precursor (ALI)": "#1CBE4F","POSTN2hi mesenchymal (ALI)" : "#B10DA1", "Goblet cells (ALI)":"#2ED9FF","Secretory club cells (ALI)": "#FBE426","Basal cells (ALI)": "#5A5156","Unknown (ALI)" : "#F5DEB3","Cycling ciliated cells (ALI)": "#B00068"}, linewidth = 2, dpi = 300, save = "Partition3scVelo.pdf")


# In[90]:


scv.pl.velocity_embedding_stream(sample_one_part4, basis = 'umap', color = 'Clusters', figsize=(20,15), legend_loc="right margin", size = 1500,palette = {"NKX2-1hi/SOX9hi/SOX2lo lung progenitors (FL)":"#AA0DFE","Ciliated precursor (FL)":"#3283FE","NKX2-1hi/SOX9lo/SOX2lo lung progenitors (FL)": "#325A9B","Ciliated cells (FL)"  : "#16FF32","Cycling ciliated cells (FL)": "#1CFFCE","NKX2-1hi/SOX9lo/SOX2hi lung progenitors (FL)": "#F8A19F","PDGFRBhi mesenchymal (FL)": "#C4451C","PLAThi mesenchymal (FL)": "#1C8356","Basal cells (FL)":"#E4E1E3", "Secretory precursor (FL)":"#BC8F8F","PNEC (FL)" : "#85660D","Myofibroblasts (ALI)": "#DEA0FD", "Ciliated cells (ALI)":"#FE00FA","EMT (ALI)": "#90AD1C","Brush/PNEC (ALI)": "#CCCC99","Cycling basal cells (ALI)": "#FEAF16","Secretory precursor (ALI)": "#1CBE4F","POSTN2hi mesenchymal (ALI)" : "#B10DA1", "Goblet cells (ALI)":"#2ED9FF","Secretory club cells (ALI)": "#FBE426","Basal cells (ALI)": "#5A5156","Unknown (ALI)" : "#F5DEB3","Cycling ciliated cells (ALI)": "#B00068"}, linewidth = 2, dpi = 300, save = "Partition4scVelo.pdf")

