import pandas as pd

rating_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('./data/u.data', sep='\t', names=rating_cols, usecols=range(3), encoding="ISO-8859-1")

movie_cols = ['movie_id', 'title']
movies = pd.read_csv('./data/u.item', sep='|', names=movie_cols, usecols=range(2), encoding="ISO-8859-1")

ratings = pd.merge(movies, ratings)
user_ratings = ratings.pivot_table(index=['user_id'], columns=['title'], values='rating')

# Expect mnimum of 100 ratings
corr_matrix = user_ratings.corr(method='pearson', min_periods=100)

# Take user id 0 as sample user
my_ratings = user_ratings.loc[0].dropna()

# Go through each rated movie and build list of recomendations
candidates_list = pd.Series()
for i in range(0, len(my_ratings.index)):
    print("Recomending movies related to " + my_ratings.index[i] )
    candidates = corr_matrix[my_ratings.index[i]].dropna()
    # Scale to how well user recomended movie
    candidates = candidates.map(lambda x: x * my_ratings[i])
    candidates_list = candidates_list.append(candidates)

candidates_list = candidates_list.groupby(candidates_list.index).sum()
candidates_list.sort_values(inplace=True, ascending=False)
# Filter out already rated by user
filtered_list = candidates_list.drop(my_ratings.index)
print("--- Recomendations ---")
print(filtered_list.head(20))