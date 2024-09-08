#!/usr/bin/env python
# coding: utf-8

# In[7]:


# https://files.grouplens.org/datasets/movielens/ml-25m.zip


# In[17]:


import pandas as pd
movies = pd.read_csv("movies.csv")


# In[19]:


import re

def clean_title(title):
    title = re.sub("[^a-zA-Z0-9 ]", "", title) #cleaning and removing all titles with special characters don't really understand
    return title

movies["clean_title"] = movies["title"].apply(clean_title) #mapping each title in data frame to clean title function

movies


# In[21]:


#turned each thing title into a vector by using vectorizer

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1,2)) #gives the language model more context by using bigrams for language processing

tfidf = vectorizer.fit_transform(movies["clean_title"])


# In[37]:


#now, turning this into a function

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def search(title):
    title = clean_title(title) #first cleaning up the title for special characters
    query_vec = vectorizer.transform([title]) #transform into a vector
    similarity = cosine_similarity(query_vec, tfidf).flatten() #comparing similarity between the title with all the titles
    indices = np.argpartition(similarity, -5)[-5:] #picking top 5 
    search_results = movies.iloc[indices].iloc[::-1]
    
    return search_results



# In[54]:


#Visualize the search results by creating a widget

import ipywidgets as widgets
from IPython.display import display

movie_input = widgets.Text(
    value="",
    description = "Movie Title",
    disabled = False
    

)

#calls the search function
movie_list = widgets.Output()

def on_type(data):
     with movie_list:
            movie_list.clear_output();
            title = data["new"]
            if len(title)>= 5:
                display(search(title)); 

            
movie_input.observe(on_type, names='value')


display(movie_input, movie_list)
            


# In[6]:


import pandas as pd

ratings = pd.read_csv("ratings.csv")


# In[12]:


ratings.head(5)


# In[48]:


#find users who have liked the same movie as what is searched 

movieId = 0

similar_users = ratings[(ratings["movieId"] == movieId) & (ratings["rating"] > 4)]["userId"].unique()


# In[47]:


# finding movie recs that more then 10% or more users similar to us also liked

similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)]["movieId"]
similar_user_recs #generated all the similar user recomendations


# In[23]:


#movie recs that users similar to us also liked (# of time liked liked)
similar_user_recs.value_counts()


# In[37]:


#movie recs that 50% or more users similar to us also liked

similar_user_recs = similar_user_recs.value_counts() / len(similar_users) #number of times liked/total number of similar users

similar_user_recs = similar_user_recs[similar_user_recs > .5]


# In[45]:


similar_user_recs


# In[ ]:





# In[ ]:




