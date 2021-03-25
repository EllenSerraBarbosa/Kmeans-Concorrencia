#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd 
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go 
import numpy  as np
import seaborn as sns
from matplotlib import pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


pd.options.display.float_format = '{:,.2f}'.format


# In[3]:


df_concorrencia = pd.read_excel('base_concorrência_sp_sprm.xlsx')


# In[ ]:


#Dados do Geoimóvel - Produtos de SP com preço médio de acima de 90K e menor que 300K


# In[5]:


df_concorrencia.head()


# In[8]:


df_concorrencia['chave_prd'] = df_concorrencia.EMPRESA +"- " + df_concorrencia.EMPREENDIMENTO


# In[10]:


df_concorrencia.EMPRESA.drop_duplicates().values


# In[11]:


df_concorrencia.EMPRESA.nunique()


# In[12]:


df_concorrencia.EMPREENDIMENTO.nunique()


# In[ ]:


#Verificando distribuição de Preço e VSO


# In[14]:


sns.boxplot(df_concorrencia.PM)


# In[15]:


sns.boxplot(df_concorrencia.VSO)


# In[ ]:


#Filtrando produtos com mais de 10 unidades em estoque


# In[16]:


mask_estoque = df_concorrencia.ESTOQUE > 10 


# In[76]:


df_concorrencia_filtro = df_concorrencia.loc[mask_estoque].groupby(['EMPRESA','chave_prd','ANO_PESQUISA'],as_index=False).agg({'PM':'mean',
                                                                                                             'VSO':'sum',
                                                                                                             'UM_DISP':'sum',                
                                                                                                             'ESTOQUE':'sum'})


# In[77]:


x = df_concorrencia_filtro[['PM']]


# In[ ]:


#Defininco nº de Clusters


# In[81]:


#definir o número de grupos de agentes ideal
get_ipython().run_line_magic('matplotlib', 'notebook')
wcss = []

for i in range(1, 11):
    kmeans2 = KMeans(n_clusters = i, init = 'random')
    kmeans2.fit(x)
    print (i,kmeans2.inertia_)
    wcss.append(kmeans2.inertia_)
    
plt.plot(range(1, 11), wcss)
plt.title('O Metodo Elbow')
plt.xlabel('Numero de Clusters')
plt.ylabel('WSS') #within cluster sum of squares
plt.show()


# In[79]:


kmeans = KMeans(n_clusters = 3, init = 'random', random_state=28)
kmeans.fit(x)
distance = kmeans.fit_transform(x)


# In[80]:


get_ipython().run_line_magic('matplotlib', 'notebook')
l = ['Cluster 1','Cluster 2', 'cluster 3']
plt.barh(l,distance[0])
plt.xlabel('Distância')
plt.title('Distância por Clusters ')
plt.show()


# In[82]:


x['Cluster'] = kmeans.labels_


# In[83]:


kmeans.labels_


# In[84]:


x.groupby('Cluster')['PM'].describe()


# In[ ]:


#Renomeando Cluster


# In[85]:


nomes = {0:'Mass',
         1: 'Masstige',
         2: 'Prestige'}


# In[86]:


x['nome_cluster'] = x.Cluster.map(nomes)


# In[87]:


df_concorrencia_filtro['nome_cluster'] = x['nome_cluster']


# In[88]:


df_concorrencia_filtro.head()


# In[89]:


df_concorrencia_filtro.groupby('nome_cluster')['PM'].describe()


# In[90]:


pd.pivot_table(df_concorrencia_filtro, values=['chave_prd', 'PM','VSO','UM_DISP','ESTOQUE'], index=['EMPRESA','ANO_PESQUISA','nome_cluster'],
                    aggfunc={
                             'PM': np.mean,
                             'UM_DISP': np.sum,
                             'VSO': np.sum,
                             'ESTOQUE': np.sum, 
                             'chave_prd': np.size}).round(2)


# In[91]:


df_concorrencia_filtro


# In[92]:


df_param_cluster = df_concorrencia_filtro.groupby(['EMPRESA','ANO_PESQUISA', 'nome_cluster'],as_index= False).agg({'chave_prd':'count',
                                                                                'PM': 'mean',
                                                                                'VSO': 'sum',
                                                                                 'UM_DISP':'sum',
                                                                                'ESTOQUE': 'sum'}).round(2)


# In[ ]:


#Base Final Clusterizada


# In[93]:


df_param_cluster


# In[61]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[67]:


plt.subplots(1,figsize=(10,8))
sns.barplot(data = df_concorrencia_filtro.groupby('nome_cluster',as_index= False).PM.mean(),
           x = 'nome_cluster', y = 'PM')
plt.title('Preço médio por Cluster');


# In[66]:


plt.subplots(1,figsize=(10,8))
sns.barplot(data = df_param_cluster, x = df_param_cluster['EMPRESA'],
            y = 'chave_prd', hue = 'nome_cluster')
plt.title('Quantidade de produtos por Cluster');


# In[94]:


df_param_cluster.to_csv('clustertenda_concorrencia_v3.csv',sep=';',decimal=',')

