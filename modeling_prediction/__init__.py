from feature_engineering import multilabel_binarizer, xtrain, xval, ytrain, yval, xtrain_tfidf, xval_tfidf, tfidf_vectorizer
from data_preprocess import preprocessed_df
from data_preprocess import stemmer_stopwords_cleaner
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.classification import log_loss


def get_sgd_lr_model_cross_grid_search():
    ###logistic regression with hyper parameter tuning
    alpha = [10 ** x for x in range(-5, 6)]  # hyperparam for SGD classifier.
    f1_score_error_array = []
    for i in alpha:
        clf = SGDClassifier(alpha=i, loss='log', penalty='l2', random_state=42)
        lr_clf = OneVsRestClassifier(clf)
        lr_clf.fit(xtrain_tfidf, ytrain)
        y_pred_new = lr_clf.predict(xval_tfidf)
        f1_score_error_array.append(f1_score(yval, y_pred_new, average="micro"))
        print('For values of alpha = ', i, "The f1_score is:",
              f1_score(yval, y_pred_new, average="micro"))
        print('For values of alpha = ', i, "The log-loss is:",
              log_loss(yval, y_pred_new, eps=1e-15))

    fig, ax = plt.subplots()
    ax.plot(alpha, f1_score_error_array, c='g')
    for i, txt in enumerate(np.round(f1_score_error_array, 3)):
        ax.annotate((alpha[i], np.round(txt, 3)), (alpha[i], f1_score_error_array[i]))
    plt.grid()
    plt.title("Cross Validation Error for each alpha")
    plt.xlabel("Alpha i's")
    plt.ylabel("Error measure")
    plt.show()

    best_alpha = int(np.argmax(f1_score_error_array))
    clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)
    lr_clf = OneVsRestClassifier(clf)
    lr_clf.fit(xtrain_tfidf, ytrain)
    y_pred_new = lr_clf.predict(xval_tfidf)
    final_f1_score = f1_score_error_array.append(f1_score(yval, y_pred_new, average="micro"))
    print('f1_score from SGDClassifier model after CV grid search is : {}'.format(final_f1_score))
    return lr_clf, final_f1_score


sgd_clf, sgd_f1_score = get_sgd_lr_model_cross_grid_search()


def get_lr_model():
    lr = LogisticRegression()
    lr_clf = OneVsRestClassifier(lr)
    # fit model on train data
    lr_clf.fit(xtrain_tfidf, ytrain)

    # make predictions for validation set
    y_pred_prob = lr_clf.predict(xval_tfidf)
    t = 0.3
    y_pred_new = (y_pred_prob >= t).astype(int)
    lr_f1_score = f1_score(yval, y_pred_new, average="micro")
    return lr_clf, lr_f1_score


lr_clf, lr_f1_score = get_lr_model()


def get_xg_boost_model():
    xg_clf = OneVsRestClassifier(XGBClassifier(n_jobs=-1, max_depth=4))
    xg_clf.fit(xtrain_tfidf, ytrain)
    y_pred_prob = xg_clf.predict_proba(xval_tfidf)
    t = 0.3
    y_pred_new = (y_pred_prob >= t).astype(int)
    xg_f1_score = f1_score(yval, y_pred_new, average="micro")
    return  xg_clf, xg_f1_score


xg_clf, xg_f1_score = get_xg_boost_model()

params = {
    'SGD': [sgd_clf, sgd_f1_score],
    'LR' : [lr_clf, lr_f1_score],
    'Xgboost': [xg_clf, xg_f1_score]
}


clf = xg_clf
f1_score = xg_f1_score
for key, val in params.items():
    if val[1] > f1_score:
        f1_score = val[1]
        clf = val[0]

print('best f1_score is : {}'.format(f1_score))


def infer_genres(plot_summary):
    q = stemmer_stopwords_cleaner(plot_summary)
    q_vec = tfidf_vectorizer.transform([q])
    q_pred = clf.predict(q_vec)
    return multilabel_binarizer.inverse_transform(q_pred)


for i in range(5):
  k = xval.sample(1).index[0]
  print("Movie: ", preprocessed_df['movie_name'][k], "\nPredicted genre: ", infer_genres(xval[k])), print("Actual genre: ",preprocessed_df['genre_new'][k], "\n")


