import pandas as pd
import numpy as np

models = ('Course Similarity',
          'User Profile',
          'Clustering',
          'Clustering with PCA',
          'KNN',
          'NMF',
          'Neural Network',
          'Regression with Embedding Features',
          'Classification with Embedding Features')


def load_ratings():
    return pd.read_csv('ratings.csv')


def load_course_sims():
    return pd.read_csv('sim.csv')


def load_courses():
    df = pd.read_csv('course_processed.csv')
    df['TITLE'] = df['TITLE'].str.title()
    return df


def load_bow():
    return pd.read_csv('courses_bows.csv')


def load_profile():
    return pd.read_csv('user_profile.csv')


def load_genres():
    return pd.read_csv('course_genre.csv')


def add_new_ratings(new_courses):
    res_dict = {}
    if len(new_courses) > 0:
        # Create a new user id, max id + 1
        ratings_df = load_ratings()
        new_id = ratings_df['user'].max() + 1
        users = [new_id] * len(new_courses)
        ratings = [3.0] * len(new_courses)
        res_dict['user'] = users
        res_dict['item'] = new_courses
        res_dict['rating'] = ratings
        new_df = pd.DataFrame(res_dict)
        updated_ratings = pd.concat([ratings_df, new_df])
        updated_ratings.to_csv('ratings.csv', index=False)
        return new_id


# Create course id to index and index to id mappings
def get_doc_dicts():
    bow_df = load_bow()
    grouped_df = bow_df.groupby(['doc_index', 'doc_id']).max().reset_index(drop=False)
    idx_id_dict = grouped_df[['doc_id']].to_dict()['doc_id']
    id_idx_dict = {v: k for k, v in idx_id_dict.items()}
    del grouped_df
    return idx_id_dict, id_idx_dict


def course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix):
    all_courses = set(idx_id_dict.values())
    unselected_course_ids = all_courses.difference(enrolled_course_ids)
    # Create a dictionary to store your recommendation results
    res = {}
    # First find all enrolled courses for user
    for enrolled_course in enrolled_course_ids:
        for unselect_course in unselected_course_ids:
            if enrolled_course in id_idx_dict and unselect_course in id_idx_dict:
                idx1 = id_idx_dict[enrolled_course]
                idx2 = id_idx_dict[unselect_course]
                sim = sim_matrix[idx1][idx2]
                if unselect_course not in res:
                    res[unselect_course] = sim
                else:
                    if sim >= res[unselect_course]:
                        res[unselect_course] = sim
    res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}
    return res


# Model training
def train(model_name, params):
    # TODO: Add model training code here
    if model_name == models[0]:
        pass
    elif model_name == models[1]:
        pass
    pass


# Prediction
def predict(model_name, user_ids, params):
    sim_threshold = 0.6
    score_threshold = 10
    top_courses = 10
    if 'sim_threshold' in params:
        sim_threshold = params['sim_threshold']
    if 'score_threshold' in params:
        score_threshold = params['score_threshold']
    if 'top_courses' in params:
        top_courses = params['top_courses']
    idx_id_dict, id_idx_dict = get_doc_dicts()
    sim_matrix = load_course_sims().to_numpy()
    users = []
    courses = []
    scores = []
    res_dict = {}

    for user_id in user_ids:
        # Course Similarity model
        if model_name == models[0]:
            ratings_df = load_ratings()  # ratings.csv
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_course_ids = user_ratings['item'].to_list()
            res = course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix)
            for key, score in res.items():
                if score >= sim_threshold:
                    users.append(user_id)
                    courses.append(key)
                    scores.append(score)
        # TODO: Add prediction model code here
        elif model_name == models[1]:
            ratings_df = load_ratings()  # ratings.csv
            profile_df = load_profile()  # user_profile.csv
            genres_df = load_genres()  # course_genre.csv
            user_ratings = profile_df[profile_df['user'] == user_id]
            user_vector = user_ratings.iloc[0, 1:].values
            enrolled_courses_ids = ratings_df[ratings_df['user'] == user_id]['item'].to_list()
            all_courses = set(idx_id_dict.values())
            unselected_course_ids = all_courses.difference(enrolled_course_ids)
            unselected_course_genres = genres_df[genres_df['COURSE_ID'].isin(unselected_course_ids)]
            unselected_course_ids = unselected_course_genres['COURSE_ID'].values
            recommendation_scores = np.dot(unselected_course_genres.iloc[:, 2:].values, user_vector)
            for i in range(0, len(unselected_course_ids)):
                score = recommendation_scores[i]
                if score >= score_threshold:
                    users.append(user_id)
                    courses.append(unselected_course_ids[i])
                    scores.append(recommendation_scores[i])
                    
    res_dict['USER'] = users
    res_dict['COURSE_ID'] = courses
    res_dict['SCORE'] = scores
    res_df = pd.DataFrame(res_dict, columns=['USER', 'COURSE_ID', 'SCORE'])
    res_df_ = res_df.sort_values(by='SCORE', ascending=False) 
    return res_df_.head(top_courses)
