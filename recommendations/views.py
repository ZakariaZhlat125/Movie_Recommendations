from django.shortcuts import render
import pandas as pd
from itertools import combinations, chain
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from rapidfuzz import process
from mlxtend.frequent_patterns import apriori, association_rules


def load_data():
    movies = pd.read_csv('ml-latest-small/movies.csv')
    ratings = pd.read_csv('ml-latest-small/ratings.csv')
    links = pd.read_csv('ml-latest-small/links.csv')
    tags = pd.read_csv('ml-latest-small/tags.csv')
    return movies, ratings, links, tags


def train_knn_model():
    movies, ratings,_,_ = load_data()
    movie_ratings = pd.merge(ratings, movies, on='movieId')
    pivot_table = movie_ratings.pivot_table(
        index='userId', columns='title', values='rating').fillna(0)
    matrix = csr_matrix(pivot_table.values)
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(matrix)
    return knn, pivot_table


def train_content_based_model():
    movies, _,_,_ = load_data()
    movies['genres'] = movies['genres'].str.split('|')
    movies = movies.explode('genres')
    genres_one_hot = pd.get_dummies(movies[['movieId', 'genres']], columns=[
                                    'genres']).groupby('movieId').sum()
    genre_similarity = cosine_similarity(genres_one_hot)
    return genre_similarity, movies


knn_model, pivot_table = train_knn_model()
genre_similarity, movies = train_content_based_model()


def get_movie_details(movie_name):
    movies, _, links, tags = load_data()
    movie_info = movies[movies['title'] == movie_name]
    if not movie_info.empty:
        movie_id = movie_info.iloc[0]['movieId']
        link = links[links['movieId'] == movie_id]['imdbId'].values
        tag = tags[tags['movieId'] == movie_id]['tag'].unique()
        imdb_link = f"https://www.imdb.com/title/tt{int(link[0]):07d}/" if link.size > 0 else None
        return imdb_link, tag
    return None, []

def knn_recommendations(movie_name):
    movie_list = pivot_table.columns
    if movie_name in movie_list:
        movie_index = movie_list.get_loc(movie_name)
        input_vector = pd.DataFrame(
            [0] * len(pivot_table.columns), index=pivot_table.columns).T
        input_vector[movie_name] = pivot_table[movie_name].mean()
        distances, indices = knn_model.kneighbors(
            input_vector.values, n_neighbors=6)
        recommendations = []
        for i in range(1, len(distances.flatten())):
            rec_movie = movie_list[indices.flatten()[i]]
            imdb_link, tags = get_movie_details(rec_movie)
            recommendations.append({
                'title': rec_movie,
                'imdb_link': imdb_link,
                'tags': tags
            })
        return recommendations
    return []

def content_based_recommendations(movie_name):
    movie_idx = movies[movies['title'] == movie_name].index
    if not movie_idx.empty:
        movie_idx = movie_idx[0]
        similarity_scores = list(enumerate(genre_similarity[movie_idx]))
        similarity_scores = sorted(
            similarity_scores, key=lambda x: x[1], reverse=True)
        movie_indices = [i[0] for i in similarity_scores[1:6]]
        recommendations = []
        for idx in movie_indices:
            rec_movie = movies.iloc[idx]['title']
            imdb_link, tags = get_movie_details(rec_movie)
            recommendations.append({
                'title': rec_movie,
                'imdb_link': imdb_link,
                'tags': tags
            })
        return recommendations
    return []


def combine_recommendations(knn_recs, content_recs):
    combined_recs = knn_recs + [rec for rec in content_recs if rec['title'] not in {movie['title'] for movie in knn_recs}]
    return combined_recs



def HKCBMR(movie_name):
    knn_recs = knn_recommendations(movie_name)
    content_recs = content_based_recommendations(movie_name)
    combined_recommendations = combine_recommendations(knn_recs, content_recs)
    return combined_recommendations



def knn_recommend_view(request):
    movie_query = request.GET.get('movie')
    if not movie_query:
        return render(request, 'recommendations/knn.html', {'error': 'Please enter a movie name.'})

    movie_list = pivot_table.columns
    closest_match = process.extractOne(movie_query, movie_list)

    if closest_match:
        match_name, score, *_ = closest_match
        if score < 60:
            return render(request, 'recommendations/knn.html', {'error': 'No close match found for the movie name entered.'})

        knn_recs = knn_recommendations(match_name)

        return render(request, 'recommendations/knn.html', {
            'movie': match_name,
            'knn_recommendations': knn_recs,
        })

    return render(request, 'recommendations/knn.html', {'error': 'No match found for the movie name entered.'})


def cb_recommend_view(request):
    movie_query = request.GET.get('movie')
    if not movie_query:
        return render(request, 'recommendations/contentBased.html', {'error': 'Please enter a movie name.'})

    movie_list = pivot_table.columns
    closest_match = process.extractOne(movie_query, movie_list)

    if closest_match:
        match_name, score, *_ = closest_match
        if score < 60:
            return render(request, 'recommendations/contentBased.html', {'error': 'No close match found for the movie name entered.'})
        content_recs = content_based_recommendations(match_name)

        return render(request, 'recommendations/contentBased.html', {
            'movie': match_name,
            'content_based_recommendations': content_recs,
        })

    return render(request, 'recommendations/contentBased.html', {'error': 'No match found for the movie name entered.'})


def recommend_movie(request):
    movie_query = request.GET.get('movie')
    if not movie_query:
        return render(request, 'recommendations/index.html', {'error': 'Please enter a movie name.'})

    movie_list = pivot_table.columns
    closest_match = process.extractOne(movie_query, movie_list)

    if closest_match:
        match_name, score, *_ = closest_match
        if score < 60:
            return render(request, 'recommendations/index.html', {'error': 'No close match found for the movie name entered.'})

        # Get recommendations using HKCBMR
        recommendations = HKCBMR(match_name)

        return render(request, 'recommendations/index.html', {
            'movie': match_name,
            'recommendations': recommendations,
        })

    return render(request, 'recommendations/index.html', {'error': 'No match found for the movie name entered.'})


def about(request):
    return render(request, 'recommendations/about.html')




def apriori_analysis(request):

    movies, ratings = load_data()
    # Step 3: Data Transformation
    # user_movie_list = ratings.groupby('userId')['movieId'].apply(list)
    # movie_dict = movies.set_index('movieId')['title'].to_dict()
    # user_movie_list = user_movie_list.apply(lambda x: [movie_dict[movie_id] for movie_id in x])
    # transactions = user_movie_list.tolist()

    # # Define minimum support and confidence
    # min_support = 0.02
    # min_confidence = 0.3

    # # Step 4: Compute item support
    # def compute_support(itemsets, transactions):
    #     itemset_support = {}
    #     total_transactions = len(transactions)

    #     for itemset in itemsets:
    #         count = sum(1 for transaction in transactions if set(itemset).issubset(transaction))
    #         support = count / total_transactions
    #         itemset_support[frozenset(itemset)] = support

    #     return itemset_support

    # # Generate initial candidate itemsets (single items)
    # itemsets = [{item} for item in chain(*transactions)]

    # # Compute support for single items
    # single_item_support = compute_support(itemsets, transactions)

    # # Filter itemsets based on minimum support
    # frequent_itemsets = {itemset: support for itemset, support in single_item_support.items() if support >= min_support}

    # # Function to generate new candidate itemsets from previous frequent itemsets
    # def generate_candidates(frequent_itemsets, k):
    #     return [a.union(b) for a in frequent_itemsets for b in frequent_itemsets if len(a.union(b)) == k]

    # # Step 5: Generate frequent itemsets of length k
    # k = 2
    # while True:
    #     candidates = generate_candidates(frequent_itemsets, k)
    #     candidate_support = compute_support(candidates, transactions)
    #     new_frequent_itemsets = {itemset: support for itemset, support in candidate_support.items() if support >= min_support}

    #     if not new_frequent_itemsets:
    #         break

    #     frequent_itemsets.update(new_frequent_itemsets)
    #     k += 1

    # # Step 6: Generate association rules
    # rules = []
    # for itemset, support in frequent_itemsets.items():
    #     if len(itemset) > 1:
    #         for consequent in combinations(itemset, 1):
    #             antecedent = itemset - set(consequent)
    #             antecedent_support = frequent_itemsets[frozenset(antecedent)]
    #             confidence = support / antecedent_support

    #             if confidence >= min_confidence:
    #                 rules.append({
    #                     'Base': ', '.join(antecedent),
    #                     'Add': ', '.join(consequent),
    #                     'Support': support,
    #                     'Confidence': confidence,
    #                     'Lift': confidence / (frequent_itemsets[frozenset(consequent)] / len(transactions))
    #                 })

    # # Convert rules to DataFrame for sorting and presentation
    # rules_df = pd.DataFrame(rules).sort_values(by=['Lift', 'Confidence'], ascending=False)

    # # Step 7: Render the template with the top 10 rules
    # return render(request, 'recommendations/rules.html', {'rules': rules_df.head(10).to_dict(orient='records')})

    # Step 3: Data Transformation
    # Limit to a subset of users to improve performance
   # Step 3: Data Transformation
    # Limit to a subset of users and movies to improve performance
    # user_subset = ratings['userId'].unique()[:200]  # Use only the first 200 users
    # movie_subset = ratings['movieId'].unique()[:1000]  # Use only the first 1000 movies

    # ratings_subset = ratings[ratings['userId'].isin(user_subset) & ratings['movieId'].isin(movie_subset)]

    # # Create a pivot table where each row is a user and each column is a movie
    # user_movie_matrix = ratings_subset.pivot(index='userId', columns='movieId', values='rating').fillna(0)

    # # Convert to binary values
    # user_movie_matrix = user_movie_matrix.applymap(lambda x: 1 if x > 0 else 0)

    # # Convert column names to strings to avoid issues with integer column names
    # user_movie_matrix.columns = user_movie_matrix.columns.astype(str)

    # # Convert DataFrame to a sparse matrix for memory efficiency
    # sparse_user_movie_matrix = csr_matrix(user_movie_matrix.values)

    # # Convert back to DataFrame with Boolean type for efficient processing
    # user_movie_matrix_bool = pd.DataFrame.sparse.from_spmatrix(sparse_user_movie_matrix, columns=user_movie_matrix.columns)

    # # Step 4: Apply the Apriori Algorithm
    # frequent_itemsets = apriori(user_movie_matrix_bool, min_support=0.02, use_colnames=True)

    # # Step 5: Generate Association Rules
    # rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)

    # # Step 6: Convert rules to a more readable format
    # rules = rules.sort_values(by=['lift', 'confidence'], ascending=False).head(10)

    # # Map movie IDs to movie titles
    # movie_dict = movies.set_index('movieId')['title'].to_dict()

    # rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join([movie_dict[int(item)] for item in x]))
    # rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join([movie_dict[int(item)] for item in x]))

    # # Render the template with the top rules
    # return render(request, 'recommendations/rules.html', {
    #     'rules': rules.to_dict(orient='records')
    # })

