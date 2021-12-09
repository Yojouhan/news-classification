import csv
import pandas as pd
import logging
import numpy as np
import nltk
import itertools
from joblib import dump
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import LinearSVC

# sklearn CPU utilization parameter, and basic logging/formatting options
nltk.download('punkt')
np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
n_jobs = -1
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
labels_dict = {'Football': 0, 'Business': 1, 'Politics': 2, 'Film': 3, 'Technology': 4}
labels_dict_inverse = {number: label for label, number in labels_dict.items()}

# Custom ROC AUC scoring for multiclass predictions
def multiclass_roc_auc(truth, pred):
    lb = LabelBinarizer()
    lb.fit(truth)

    truth = lb.transform(truth)
    pred = lb.transform(pred)

    return roc_auc_score(truth, pred, average="macro")


# Evaluate using multiple measures
# Macro averaging is used for a better estimation
scoring = {'Accuracy': 'accuracy', 'Precision': 'precision_macro', 'Recall': 'recall_macro',
           'F-Measure': 'f1_macro',
           'AUC': make_scorer(multiclass_roc_auc)}


def load_dataset(subset):
    path = ''
    if subset == 'train':
        path = 'data/train_set.csv'
        df = pd.read_csv(path, delimiter='\t', usecols=['Content', 'Category'])
        return df['Content'], df['Category']
    elif subset == 'test':
        path = 'data/test_set.csv'
        df = pd.read_csv(path, delimiter='\t', usecols=['Id', 'Content'])
        return df['Id'], df['Content']


def vectorize(corpus, method):
    if method == 'bow':
        print('Vectorizing train corpus with BOW model...')
        bow_vectorizer = CountVectorizer()
        bow_corpus = bow_vectorizer.fit_transform(corpus)
        return bow_vectorizer, bow_corpus
    elif method == 'svd':
        # Vectorizer which ignores words that occur in less than 600 documents (around ~5% of documents)
        # This is to avoid memory errors when transforming the matrix with SVD
        # The desired variance is achieved around 500 components.
        svd_vectorizer = CountVectorizer(stop_words='english', min_df=600)
        print('Vectorizing train corpus with min_df = 600 ...')
        min_df_corpus = svd_vectorizer.fit_transform(corpus)
        print('Original matrix shape: ', min_df_corpus.shape)
        # SVD, for text classification the optimal value for the n_components attribute is 100 according to sklearn doc
        svd = TruncatedSVD(n_components=100, n_iter=5)
        print('Performing SVD on train corpus...')
        svd_corpus = svd.fit_transform(min_df_corpus)
        print('Corpus shape after SVD: ', svd_corpus.shape)
        print('Explained variance ratio is: ', svd.explained_variance_ratio_.sum())
        return svd, svd_corpus
    elif method == 'tfidf':
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_corpus = tfidf_vectorizer.fit_transform(corpus)
        # Save to disk, this was the most accurate vectorizer when used with the Ridgeclassifier
        dump(tfidf_vectorizer, 'vectorizer.joblib')
        return tfidf_vectorizer, tfidf_corpus
    elif method == 'w2v':
        # W2V MODEL
        # Tokenize sentences
        print('Tokenizing the corpus and training w2v model...')
        tokenized_corpus = [word_tokenize(article) for article in corpus]
        # Learn word vectors from the corpus, dimension is 100
        model = Word2Vec(tokenized_corpus, vector_size=100, window=5, min_count=5, workers=4)
        model.train(tokenized_corpus, total_examples=len(tokenized_corpus), epochs=5)
        # Transform the articles in the corpus to the corresponding average vectors
        X = []
        print('Converting documents to vectors...')
        article_counter = 0
        for article in tokenized_corpus:
            if len(article) > 0:
                doc = [word for word in article if word in model.wv]
            else:
                doc = ['empty']
            article_counter += 1
            # Average of each vector
            w2v_article = np.mean(model.wv[doc], axis=0)
            X.append(w2v_article)

        # Sanity check and conversion to numpy array
        print('Processed this number of articles: ', len(X))
        w2v_corpus = np.array(X)
        print('Train corpus shape after word2vec conversion', w2v_corpus.shape)
        return model, w2v_corpus


def train_evaluate_classifier(corpus, labels, clf):
    if clf == 'svm':
        # Train SVM and evaluate with 10fold
        # Dual = False helps speed up the process
        print('Training SVM classifier...')
        svm_clf = LinearSVC(dual=False)
        svm_score = cross_validate(svm_clf, corpus, labels, cv=10, scoring=scoring, n_jobs=n_jobs)
        return svm_score
    elif clf == 'random_forest':
        print('Training Random Forest Classifier...')
        forest_clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=n_jobs)
        print('Predicting with Random Forest...')
        forest_score = cross_validate(forest_clf, corpus, labels, cv=10, scoring=scoring, n_jobs=n_jobs)
        return forest_score
    elif clf == 'ridge':
        # Custom method using ridge classifier
        # After multiple tests, this turned out to be the most successful one metrics-wise
        # Some preprocessing is also done here, by using stop words to remove irrelevant words from the vocabulary
        print('Training custom ridge classifier (for benchmarking against the other ones)')
        ridge_clf = RidgeClassifier()
        ridge_clf.fit(corpus, labels)
        ridge_clf_score = cross_validate(ridge_clf, corpus, labels, cv=10, scoring=scoring,
                                         n_jobs=n_jobs,
                                         verbose=10)
        dump(ridge_clf, 'ridge_classifier.joblib')
        return ridge_clf, ridge_clf_score


def format_results(score_list):
    results = []
    for clf_score in score_list:
        clf_results = {'Accuracy': float("{0:.4f}".format(np.mean(clf_score['test_Accuracy']))),
                       'Precision': float("{0:.4f}".format(np.mean(clf_score['test_Precision']))),
                       'Recall': float("{0:.4f}".format(np.mean(clf_score['test_Recall']))),
                       'F-Measure': float("{0:.4f}".format(np.mean(clf_score['test_F-Measure']))),
                       'AUC': float("{0:.4f}".format(np.mean(clf_score['test_AUC'])))}
        results.append(clf_results)
    return results


def evaluation_file(results):
    # Create EvaluationMetric_10fold csv file
    with open('EvaluationMetric_10fold.csv', 'w', encoding='utf8', newline='') as outfile:
        writer = csv.writer(outfile, delimiter='\t')
        accuracy_line = ['Accuracy']
        precision_line = ['Precision']
        recall_line = ['Recall']
        fmeasure_line = ['F-Measure']
        auc_line = ['AUC']
        for result_dict in results:
            accuracy_line.append(result_dict['Accuracy'])
            precision_line.append(result_dict['Precision'])
            recall_line.append(result_dict['Recall'])
            fmeasure_line.append(result_dict['F-Measure'])
            auc_line.append(result_dict['AUC'])
        # Header
        writer.writerow(
            ['Statistic Measure', 'SVM(BoW)', 'Random Forest(BoW)', 'SVM(SVD)', 'Random Forest(SVD)', 'SVM(W2V)',
             'Random Forest(W2V)', 'My Method'])
        # One line for each metric
        for row in [accuracy_line, precision_line, recall_line,fmeasure_line, auc_line]:
            writer.writerow(row)


def predict(corpus, clf, vectorizer):
    print('Transforming test corpus...')
    test_corpus = vectorizer.transform(corpus)
    print('Predicting on test set...')
    predictions = clf.predict(test_corpus)
    return predictions


def testset_categories_file(test_ids, predictions):
    # Create testSet_categories csv
    # Mapping is as in the train set
    with open('testSet_categories.csv', 'w', encoding='utf8', newline='') as outfile:
        writer = csv.writer(outfile, delimiter='\t')
        # Header
        writer.writerow(['Test_Document_ID', 'Predicted_Category'])
        for doc_id, prediction in zip(test_ids, predictions):
            writer.writerow([doc_id, labels_dict_inverse[prediction]])


def main():
    # Load train data
    corpus, labels = load_dataset(subset='train')
    labels.replace(labels_dict, inplace=True)
    print(corpus, labels)
    print('Number of documents: ', len(labels))
    # Construct dicts for all the vectorizers and classifiers that will be tested
    corpus_dict = {'bow': [], 'svd': [], 'w2v': []}
    vect_dict = {'bow': None, 'svd': None, 'w2v': None}
    classifiers = ['svm', 'random_forest']
    for vectorizer in corpus_dict:
        model, vect_corpus = vectorize(corpus, method=vectorizer)
        vect_dict[vectorizer] = model
        corpus_dict[vectorizer].append(vect_corpus)

    # Train, evaluate classifiers and format results properly
    scores = []
    combinations = list(itertools.product(corpus_dict.values(), classifiers))
    for current_corpus, classifier in combinations:
        scores.append(train_evaluate_classifier(current_corpus[0], labels, classifier))



    # Use ridge classifier as custom method for beating the benchmark
    tfidf_vect, tfidf_corpus = vectorize(corpus, method='tfidf')
    ridge_clf, ridge_score = train_evaluate_classifier(tfidf_corpus, labels, clf='ridge')
    scores.append(ridge_score)
    formatted_scores = format_results(score_list=scores)

    # Predict on test set and generate evaluation csvs
    print(formatted_scores)
    evaluation_file(formatted_scores)
    test_ids, test_corpus = load_dataset(subset='test')
    predictions = predict(test_corpus, ridge_clf, tfidf_vect)
    testset_categories_file(test_ids, predictions)


main()
