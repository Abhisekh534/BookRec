import numpy as np
import pandas as pd

books = pd.read_csv("C:\\Users\\abhis\\Downloads\\bookRec\\Books.csv", low_memory=False)
ratings = pd.read_csv("C:\\Users\\abhis\\Downloads\\bookRec\\Ratings.csv", low_memory=False)
users = pd.read_csv("C:\\Users\\abhis\\Downloads\\bookRec\\Users.csv", low_memory=False)


ratiing_with_name = ratings.merge(books, on='ISBN')
ratiing_with_name['Book-Rating'] = pd.to_numeric(ratiing_with_name['Book-Rating'], errors='coerce')
num_rating_df = ratiing_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={'Book-Rating' : 'num Ratings'}, inplace=True)

average_rating_df = ratiing_with_name.groupby('Book-Title')['Book-Rating'].mean().reset_index()
average_rating_df.rename(columns={'Book-Rating' : 'Avg Ratings'}, inplace=True)

popularity_df = num_rating_df.merge(average_rating_df, on='Book-Title')

popularity_df = popularity_df[popularity_df['num Ratings']>=250].sort_values('Avg Ratings', ascending=False).head(50)



popularity_df = popularity_df.merge(books, on='Book-Title').drop_duplicates('Book-Title')[['Book-Title', 'Book-Author', 'Image-URL-M', 'num Ratings', 'Avg Ratings']]

#Collaborative Filtering

x = ratiing_with_name.groupby('User-ID').count()['Book-Rating']>=200
good_users = x[x].index 
filtered_rating = ratiing_with_name[ratiing_with_name['User-ID'].isin(good_users)]

y = filtered_rating.groupby('Book-Title').count()['Book-Rating']>=50
famous_books = y[y].index
final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]

#pivot table
pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
pt.fillna(0, inplace=True)

#every book is now a vector of around 800 user ratings
#the closer this vector will be for two books it means it gave more similar experience to same users

from sklearn.metrics.pairwise import cosine_similarity

#707 books ka 707 books ke saath euclidean distance
similarity_score = cosine_similarity(pt)

def recommend(book_name):
    # index fetch
    index = np.where(pt.index==book_name)[0][0]
    similar_items = sorted(list(enumerate(similarity_score[index])),key=lambda x:x[1],reverse=True)[1:5]
    
    data = []
    for i in similar_items:
        item = []
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
        
        data.append(item)
    
    return data

import pickle
pickle.dump(popularity_df, open('popular.pkl', 'wb'))

books.drop_duplicates('Book-Title')

pickle.dump(pt,open('pt.pkl','wb'))
pickle.dump(books,open('books.pkl','wb'))
pickle.dump(similarity_score,open('similarity_scores.pkl','wb'))
