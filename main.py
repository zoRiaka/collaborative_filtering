# %matplotlib inline
import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy import stats
# from ast import literal_eval
# from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
# from nltk.stem.snowball import SnowballStemmer
# from nltk.stem.wordnet import WordNetLemmatizer
# from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
#split and evaluate functions are removed by surprise.
#And replaced with cross_validate.

import warnings; warnings.simplefilter('ignore')

# Collaborative filtering

md = pd.read_csv('data/movies_metadata.csv') #Check path
ratings = pd.read_csv('data/ratings_small.csv')

reader = Reader()
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
# data.split(n_folds=5)

svd = SVD()
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

trainset = data.build_full_trainset()
svd.train(trainset)

film_ids= [302,]
for film_id in film_ids:
    print(svd.predict(1, film_id, 3))

# Реалізувати SVD + train (line 33)
