#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[4]:


af = pd.read_csv("movies.csv") 
bf = pd.read_csv("ratings.csv") 
cf = pd.read_csv("links.csv") 
df = pd.read_csv("tags.csv") 


# In[5]:


print(af.shape)
print(bf.shape)


# In[6]:


import csv
csv_reader = csv.reader(bf, delimiter=',')
uniqueIds = set()

for row in csv_reader:
    uniqueIds.add(row[0])

print(len(uniqueIds))


# In[7]:


bf['userId'].unique()


# In[8]:


ratings = bf["rating"].value_counts()
numbers = ratings.index
quantity = ratings.values
import plotly.express as px
fig = px.pie(bf, values=quantity, names=numbers)
fig.show()


# In[9]:


max_ratings_movie = bf.loc[bf['rating'].idxmax()]['movieId']

print(f"The movie with the maximum number of user ratings is: {max_ratings_movie}")


# In[10]:


bf.groupby('movieId')['rating'].mean()


# In[138]:


rating_columns = ['user_id', 'movie_id', 'rating','timestamp']
ratings = pd.read_csv("ratings.csv")
links_columns=['movie_id','imdbld','tmdbld']
links = pd.read_csv("links.csv")
movie_columns = ['movie_id', 'title','genres']
movies = pd.read_csv("movies.csv")
tag_columns = ['user_id', 'movie_id', 'tag','timestamp']
tags = pd.read_csv("tags.csv")


# In[152]:



movie_ratings = pd.merge(movies, ratings)

movie_data = pd.merge(movies,tags)
links_data=pd.merge(movies,links)


# In[54]:


movie_ratings.head()


# In[55]:


movie_data.head()


# In[57]:


ratings = movie_ratings['rating'].value_counts()
ratings


# In[58]:



# Specify the film for which you want the average rating
specific_film = 'Terminator 2: Judgment Day (1991)'

# Filter the DataFrame for the specific film and calculate its average rating
avg_rating = movie_ratings[movie_ratings['title'] == specific_film]['rating'].mean()
print(f"The average rating for {specific_film} is: {avg_rating}")


# In[59]:





# Use value_counts to count the occurrences of each movie and find the one with the maximum count
max_rated_movie = movie_ratings['title'].value_counts().idxmax()
max_ratings_count = movie_ratings['title'].value_counts().max()

print(f"The movie with the maximum number of user ratings is '{max_rated_movie}' with {max_ratings_count} ratings.")


# In[60]:




# Specify the movie you're interested in
movie_of_interest = 'Matrix, The (1999)'

# Filter the DataFrame for the specified movie's tags
tags_for_movie = movie_data[movie_data['title'] == movie_of_interest]['tag'].unique()
print(f"The correct tags for '{movie_of_interest}' submitted by users are: {list(tags_for_movie)}")


# In[65]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have a DataFrame named 'ratings' with columns 'Movie' and 'Rating'
# Replace this with your actual dataset


# Filter ratings for the "Fight Club (1999)" movie
fight_club_ratings = movie_ratings[movie_ratings['title'] == 'Fight Club (1999)']

# Plotting the data distribution using a histogram
plt.figure(figsize=(8, 6))
plt.hist(fight_club_ratings['rating'], bins=5, edgecolor='black')
plt.xlabel('rating')
plt.ylabel('Frequency')
plt.title('User Ratings Distribution for "Fight Club (1999)"')
plt.grid(True)
plt.show()


# In[113]:




# Grouping by 'movie_id' and applying count and mean aggregation functions
#grouped_ratings = movie_ratings.groupby('title').agg({'rating': ['count', 'mean']})
# Grouping by 'title' and applying count and mean aggregation functions
grouped_ratings = movie_ratings.groupby('title').agg({'rating': ['count', 'mean']})

# Assuming you already have the grouped_ratings DataFrame
# (created using the code provided in the previous response)



# Sorting the DataFrame in descending order based on mean rating
sorted_ratings = grouped_ratings.sort_values(by=('rating', 'count'), ascending=False)

# Displaying the sorted DataFrame
print(sorted_ratings)


# In[114]:


# Assuming you have the original 'movie_ratings' DataFrame
# You can use the following code to find the top 5 popular movies based on the number of user ratings

# Grouping by 'title' and applying count aggregation function
movie_popularity = movie_ratings.groupby('title').agg({'rating': 'count'})

# Sorting the DataFrame in descending order based on the number of user ratings
sorted_popularity = movie_popularity.sort_values(by='rating', ascending=False)

# Selecting the top 5 popular movies
top_5_popular_movies = sorted_popularity.head(5)

# Displaying the top 5 popular movies
print("Top 5 Popular Movies based on Number of User Ratings:")
print(top_5_popular_movies)


# In[122]:


# Assuming you have the original 'movie_ratings' DataFrame
# You can use the following code to find the third most popular Sci-Fi movie based on the number of user ratings

# Filter the DataFrame to include only Sci-Fi movies
scifi_movies = movie_ratings[movie_ratings['genres'] == 'Sci-Fi']

# Grouping Sci-Fi movies by 'title' and applying count aggregation function
scifi_popularity = scifi_movies.groupby('title').agg({'rating': 'count'})

# Sorting the DataFrame in descending order based on the number of user ratings
sorted_scifi_popularity = scifi_popularity.sort_values(by='rating')

# Selecting the third most popular Sci-Fi movie
third_most_popular_scifi_movie = sorted_scifi_popularity.index[2]

# Displaying the result
print("Third Most Popular Sci-Fi Movie based on Number of User Ratings:")
print(third_most_popular_scifi_movie)


# In[120]:


# Assuming you have the original 'movie_ratings' DataFrame
# You can use the following code to filter movies with more than 50 user ratings

# Grouping by 'title' and applying count aggregation function
movie_popularity = movie_ratings.groupby('title').agg({'rating': 'count'})

# Filtering movies with more than 50 user ratings
popular_movies = movie_popularity[movie_popularity['rating'] > 50]

# Displaying the result
print("Movies with more than 50 User Ratings:")
print(popular_movies)


# In[112]:




# Assuming you have loaded movies data into movies_df and created grouped_df

# Perform inner join on 'movieId' column
merged_df = pd.merge(af,movie_ratings, on='movieId', how='inner')
merged_df.tail(100)


# In[92]:


# Assuming you have a DataFrame 'movie-ratings' with columns 'title' and 'rating'
# Replace 'movies_data' and column names with your actual data

# Filter movies with more than 50 user ratings
filtered_movies =movie_ratings[movie_ratings['rating']>50]
print(filtered_movies)


# In[134]:


data = pd.merge(movie_ratings,movie_data)
data


# In[137]:


import pandas as pd

# Sample Data

movie_ratings = pd.DataFrame(movie_ratings)

# Filter Sci-Fi movies
scifi_movies = movie_ratings[movie_ratings['genre'] == 'Sci-Fi']

# Grouping Sci-Fi movies by 'title' and applying count aggregation function
scifi_popularity = scifi_movies.groupby('title').agg({'rating': 'count'})

# Sorting the Sci-Fi movies DataFrame in descending order based on the number of user ratings
sorted_scifi_popularity = scifi_popularity.sort_values(by='rating', ascending=False)

# Selecting the third most popular Sci-Fi movie
third_most_popular_scifi_movie = sorted_scifi_popularity.index[2]

# Displaying the result
print("Third Most Popular Sci-Fi Movie based on Number of User Ratings:")
print(third_most_popular_scifi_movie)


# In[136]:


import pandas as pd

def find_third_most_popular_scifi_movie(df):
    # Filter Sci-Fi movies
    scifi_movies = df[df['genre'] == 'Sci-Fi']

    # Grouping Sci-Fi movies by 'title' and applying count aggregation function
    scifi_popularity = scifi_movies.groupby('title').agg({'rating': 'count'})

    # Sorting the Sci-Fi movies DataFrame in descending order based on the number of user ratings
    sorted_scifi_popularity = scifi_popularity.sort_values(by='rating', ascending=False)

    # Check if there are at least 3 Sci-Fi movies
    if len(sorted_scifi_popularity) >= 3:
        # Selecting the third most popular Sci-Fi movie
        third_most_popular_scifi_movie = sorted_scifi_popularity.index[2]
        return third_most_popular_scifi_movie
    else:
        return "Not enough Sci-Fi movies available."

# Sample Data
data = pd.DataFrame(movie_data)
movie_ratings = pd.DataFrame(movie_ratings)

# Using the function to find the third most popular Sci-Fi movie
result = find_third_most_popular_scifi_movie(movie_ratings)

# Displaying the result
print("Third Most Popular Sci-Fi Movie based on Number of User Ratings:")
print(result)


# In[157]:


movies.head()


# In[148]:


import pandas as pd

# Sample Data



# Finding the movieId of the movie with the highest IMDb rating
highest_rated_movie_id = links.loc[links['imdbId'].idxmax(), 'movieId']

# Displaying the result
print("MovieId of the Movie with the Highest IMDb Rating:")
print(highest_rated_movie_id)


# In[161]:


import pandas as pd

# Sample Data


# Filtering Sci-Fi movies
scifi_movies = movies[movies['genres'] == 'Sci-Fi']

# Finding the movieId of the Sci-Fi movie with the highest IMDb rating
highest_rated_scifi_movie_id = links.loc[links['imdbld'].idxmax(), 'movieId']

# Displaying the result
print("MovieId of the Highest Rated Sci-Fi Movie:")
print(highest_rated_scifi_movie_id)


# In[ ]:




