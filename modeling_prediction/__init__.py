from feature_engineering import multilabel_binarizer, xtrain, xval, ytrain, yval, xtrain_tfidf, xval_tfidf, tfidf_vectorizer
from data_preprocess import preprocessed_df
from data_preprocess import stemmer_stopwords_cleaner
from sklearn.linear_model import LogisticRegression
import pickle
# Binary Relevance
from sklearn.multiclass import OneVsRestClassifier

# Performance metric
from sklearn.metrics import f1_score

lr = LogisticRegression()
clf = OneVsRestClassifier(lr)
# fit model on train data
clf.fit(xtrain_tfidf, ytrain)

# make predictions for validation set
y_pred = clf.predict(xval_tfidf)

# evaluate performance
print('f1_score is : {}'.format(f1_score(yval, y_pred, average="micro")))

# We get a decent F1 score of 0.315. These predictions were made based on a threshold value of 0.5, which means that the
# probabilities greater than or equal to 0.5 were converted to 1’s and the rest to 0’s
# predict probabilities
y_pred_prob = clf.predict_proba(xval_tfidf)
t = 0.3 # threshold value
y_pred_new = (y_pred_prob >= t).astype(int)

# evaluate performance
print('f1_score is : {}'.format(f1_score(yval, y_pred_new, average="micro")))

def infer_genres(plot_summary):
    q = stemmer_stopwords_cleaner(plot_summary)
    q_vec = tfidf_vectorizer.transform([q])
    q_pred = clf.predict(q_vec)
    return multilabel_binarizer.inverse_transform(q_pred)

for i in range(5):
  k = xval.sample(1).index[0]
  print("Movie: ", preprocessed_df['movie_name'][k], "\nPredicted genre: ", infer_genres(xval[k])), print("Actual genre: ",preprocessed_df['genre_new'][k], "\n")
