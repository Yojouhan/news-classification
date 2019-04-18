import csv
import logging
import numpy as np
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

# Lists to hold corpus and respective class for each document in the corpus, train and test
# Also keep ids from the test set to format the csv file later on
corpus = []
categories = []
test_corpus = []
test_ids = []
# sklearn CPU utilization parameter, and basic logging/formatting options
np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
n_jobs = -1
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# Custom ROC AUC scoring for multiclass predictions
def multiclass_ROC_AUC(truth, pred):
    lb = LabelBinarizer()
    lb.fit(truth)

    truth = lb.transform(truth)
    pred = lb.transform(pred)

    return roc_auc_score(truth, pred, average="macro")


# Read train set file and skip first row
print('Reading train_set.csv...')
with open('data/train_set.csv', 'r', encoding="utf8") as trainfile:
    reader = csv.reader(trainfile, delimiter='\t')
    headers = next(reader)
    for row in reader:
        # Append document to corpus list
        corpus.append(row[3])
        # Append categories to categories list
        # Business is 0, Film is 1, Football is 2, Politics is 3, Technology is 4
        if row[4] == 'Business':
            categories.append(0)
        elif row[4] == 'Film':
            categories.append(1)
        elif row[4] == 'Football':
            categories.append(2)
        elif row[4] == 'Politics':
            categories.append(3)
        elif row[4] == 'Technology':
            categories.append(4)

print('Number of documents: ', len(categories))
# Create bag of words model
bow_vectorizer = CountVectorizer()
print('Vectorizing train corpus with BOW model...')
bow_corpus = bow_vectorizer.fit_transform(corpus)

# Train SVM and evaluate with 10fold
# Dual = False helps speed up the process
svm_clf = LinearSVC(dual=False)
print('Training SVM classifier...')
# Evaluate using multiple measures
# Macro averaging is used for a better estimation
# BOW SVM
scoring = {'Accuracy': 'accuracy', 'Precision': 'precision_macro', 'Recall': 'recall_macro', 'F-Measure': 'f1_macro',
           'AUC': make_scorer(multiclass_ROC_AUC)}
bow_SVM_score = cross_validate(svm_clf, bow_corpus, categories, cv=10, scoring=scoring, n_jobs=n_jobs)

# Random Forest BOW classifier
print('Training Random Forest Classifier...')
forest_clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=n_jobs)
print('Predicting with Random Forest...')
bow_forest_score = cross_validate(forest_clf, bow_corpus, categories, cv=10, scoring=scoring, n_jobs=n_jobs)

# Vectorizer which ignores words that occur in less than 600 documents (around ~5% of documents)
# This is to avoid memory errors when transforming the matrix with SVD
# The desired variance is achieved around 500 components.
svd_vectorizer = CountVectorizer(stop_words='english', min_df=600)
print('Vectorizing train corpus with min_df = 600 ...')
min_df_corpus = svd_vectorizer.fit_transform(corpus)
print('Original matrix shape: ', min_df_corpus.shape)
# SVD, for text classification the optimal value for the n_components attribute is 100 according to sklearn doc
svd = TruncatedSVD(n_components=500, n_iter=5)
print('Performing SVD on train corpus...')
svd_corpus = svd.fit_transform(min_df_corpus)
print('Corpus shape after SVD: ', svd_corpus.shape)
print('Explained variance ratio is: ', svd.explained_variance_ratio_.sum())
# Train and evaluate classifiers on SVD corpus
# SVM SVD
print('Evaluating SVM with SVD features...')
svd_SVM_score = cross_validate(svm_clf, svd_corpus, categories, cv=10, scoring=scoring, n_jobs=n_jobs)

# Random Forest SVD
print('Evaluating Random Forest with SVD features...')
svd_forest_score = cross_validate(forest_clf, svd_corpus, categories, cv=10, scoring=scoring, n_jobs=n_jobs)

# W2V MODEL
# Tokenize sentences
print('Tokenizing the corpus and training w2v model...')
tokenized_corpus = [word_tokenize(article) for article in corpus]
# Learn word vectors from the corpus, dimensionality is 100
model = Word2Vec(tokenized_corpus, size=100, window=5, min_count=5, workers=4)
model.train(tokenized_corpus, total_examples=len(tokenized_corpus), epochs=5)
# Transform the articles in the corpus to the corresponding average vectors
X = []
print('Converting documents to vectors...')
article_counter = 0
for article in tokenized_corpus:
    if len(article) > 0:
        doc = [word for word in article if word in model.wv.vocab]
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

print('Calculating SVM results for w2v...')
# Evaluate w2v model using multiple measures
# Macro averaging is used for a better estimation
w2v_SVM_score = cross_validate(svm_clf, w2v_corpus, categories, cv=10, scoring=scoring, n_jobs=n_jobs)
print('Calculating Random Forest results for w2v...')
w2v_forest_score = cross_validate(forest_clf, w2v_corpus, categories, cv=10, scoring=scoring, n_jobs=n_jobs)

# Custom method using ridge classifier
# After multiple tests, this turned out to be the most successful one metrics-wise
# Some preprocessing is also done here, by using stop words to remove irrelevant words from the vocabulary
print('Training custom ridge classifier (for benchmarking against the other ones)')
ridge_clf = RidgeClassifier()
ridge_vectorizer = TfidfVectorizer(stop_words='english')
ridge_corpus = ridge_vectorizer.fit_transform(corpus)
ridge_clf.fit(ridge_corpus, categories)
tfidf_ridgeclf_score = cross_validate(ridge_clf, ridge_corpus, categories, cv=10, scoring=scoring, n_jobs=n_jobs,
                                      verbose=10)

# Summarise results, while also formatting them to 4 digit precision
# One dictionary for each method
w2v_SVM_results = {'Accuracy': float("{0:.4f}".format(np.mean(w2v_SVM_score['test_Accuracy']))),
                   'Precision': float("{0:.4f}".format(np.mean(w2v_SVM_score['test_Precision']))),
                   'Recall': float("{0:.4f}".format(np.mean(w2v_SVM_score['test_Recall']))),
                   'F-Measure': float("{0:.4f}".format(np.mean(w2v_SVM_score['test_F-Measure']))),
                   'AUC': float("{0:.4f}".format(np.mean(w2v_SVM_score['test_AUC'])))}

svd_SVM_results = {'Accuracy': float("{0:.4f}".format(np.mean(svd_SVM_score['test_Accuracy']))),
                   'Precision': float("{0:.4f}".format(np.mean(svd_SVM_score['test_Precision']))),
                   'Recall': float("{0:.4f}".format(np.mean(svd_SVM_score['test_Recall']))),
                   'F-Measure': float("{0:.4f}".format(np.mean(svd_SVM_score['test_F-Measure']))),
                   'AUC': float("{0:.4f}".format(np.mean(svd_SVM_score['test_AUC'])))
                   }

w2v_forest_results = {'Accuracy': float("{0:.4f}".format(np.mean(w2v_forest_score['test_Accuracy']))),
                      'Precision': float("{0:.4f}".format(np.mean(w2v_forest_score['test_Precision']))),
                      'Recall': float("{0:.4f}".format(np.mean(w2v_forest_score['test_Recall']))),
                      'F-Measure': float("{0:.4f}".format(np.mean(w2v_forest_score['test_F-Measure']))),
                      'AUC': float("{0:.4f}".format(np.mean(w2v_forest_score['test_AUC'])))}
svd_forest_results = {'Accuracy': float("{0:.4f}".format(np.mean(svd_forest_score['test_Accuracy']))),
                      'Precision': float("{0:.4f}".format(np.mean(svd_forest_score['test_Precision']))),
                      'Recall': float("{0:.4f}".format(np.mean(svd_forest_score['test_Recall']))),
                      'F-Measure': float("{0:.4f}".format(np.mean(svd_forest_score['test_F-Measure']))),
                      'AUC': float("{0:.4f}".format(np.mean(svd_forest_score['test_AUC'])))}
bow_forest_results = {'Accuracy': float("{0:.4f}".format(np.mean(bow_forest_score['test_Accuracy']))),
                      'Precision': float("{0:.4f}".format(np.mean(bow_forest_score['test_Precision']))),
                      'Recall': float("{0:.4f}".format(np.mean(bow_forest_score['test_Recall']))),
                      'F-Measure': float("{0:.4f}".format(np.mean(bow_forest_score['test_F-Measure']))),
                      'AUC': float("{0:.4f}".format(np.mean(bow_forest_score['test_AUC'])))}
bow_SVM_results = {'Accuracy': float("{0:.4f}".format(np.mean(bow_SVM_score['test_Accuracy']))),
                   'Precision': float("{0:.4f}".format(np.mean(bow_SVM_score['test_Precision']))),
                   'Recall': float("{0:.4f}".format(np.mean(bow_SVM_score['test_Recall']))),
                   'F-Measure': float("{0:.4f}".format(np.mean(bow_SVM_score['test_F-Measure']))),
                   'AUC': float("{0:.4f}".format(np.mean(bow_SVM_score['test_AUC'])))}

tfidf_ridge_results = {'Accuracy': float("{0:.4f}".format(np.mean(tfidf_ridgeclf_score['test_Accuracy']))),
                       'Precision': float("{0:.4f}".format(np.mean(tfidf_ridgeclf_score['test_Precision']))),
                       'Recall': float("{0:.4f}".format(np.mean(tfidf_ridgeclf_score['test_Recall']))),
                       'F-Measure': float("{0:.4f}".format(np.mean(tfidf_ridgeclf_score['test_F-Measure']))),
                       'AUC': float("{0:.4f}".format(np.mean(tfidf_ridgeclf_score['test_AUC'])))}

# Print results from all methods
print('Random Forest w2v classifier', w2v_forest_results)
print('SVM w2v classifier', w2v_SVM_results)
print('Random Forest SVD results', svd_forest_results)
print('SVD SVM results', svd_SVM_results)
print('Random Forest BOW results', bow_forest_results)
print('SVM BOW Results', bow_SVM_results)
print('Ridge classifier results', tfidf_ridge_results)

# Create EvaluationMetric_10fold csv file
with open('EvaluationMetric_10fold.csv', 'w', encoding='utf8', newline='') as outfile:
    writer = csv.writer(outfile, delimiter='\t')
    # Header
    writer.writerow(
        ['Statistic Measure', 'SVM(BoW)', 'Random Forest(BoW)', 'SVM(SVD)', 'Random Forest(SVD)', 'SVM(W2V)',
         'Random Forest(W2V)', 'My Method'])
    # One line for each metric
    writer.writerow(
        ['Accuracy', bow_SVM_results['Accuracy'], bow_forest_results['Accuracy'], svd_SVM_results['Accuracy'],
         svd_forest_results['Accuracy'], w2v_SVM_results['Accuracy'], w2v_forest_results['Accuracy'],
         tfidf_ridge_results['Accuracy']])
    writer.writerow(
        ['Precision', bow_SVM_results['Precision'], bow_forest_results['Precision'], svd_SVM_results['Precision'],
         svd_forest_results['Precision'], w2v_SVM_results['Precision'], w2v_forest_results['Precision'],
         tfidf_ridge_results['Precision']])
    writer.writerow(
        ['Recall', bow_SVM_results['Recall'], bow_forest_results['Recall'], svd_SVM_results['Recall'],
         svd_forest_results['Recall'], w2v_SVM_results['Recall'], w2v_forest_results['Recall'],
         tfidf_ridge_results['Recall']])
    writer.writerow(
        ['F-Measure', bow_SVM_results['F-Measure'], bow_forest_results['F-Measure'], svd_SVM_results['F-Measure'],
         svd_forest_results['F-Measure'], w2v_SVM_results['F-Measure'], w2v_forest_results['F-Measure'],
         tfidf_ridge_results['F-Measure']])
    writer.writerow(
        ['AUC', bow_SVM_results['AUC'], bow_forest_results['AUC'], svd_SVM_results['AUC'],
         svd_forest_results['AUC'], w2v_SVM_results['AUC'], w2v_forest_results['AUC'],
         tfidf_ridge_results['AUC']])

# Create list of predictions and ids from the test set
print('Creating test corpus...')
with open('data/test_set.csv', 'r', encoding='utf8') as testfile:
    reader = csv.reader(testfile, delimiter='\t')
    headers = next(reader)
    for row in reader:
        test_ids.append(row[1])
        test_corpus.append(row[3])

print('Transforming test corpus...')
# Transform test corpus with tfidf vectorizer (english stop words) and make predictions
test_tfidf_corpus = ridge_vectorizer.transform(test_corpus)
print('Predicting on test set...')
predictions = ridge_clf.predict(test_tfidf_corpus)

# Create testSet_categories csv
# Mapping is as in the train set
# Business is 0, Film is 1, Football is 2, Politics is 3, Technology is 4
with open('testSet_categories.csv', 'w', encoding='utf8', newline='') as outfile:
    writer = csv.writer(outfile, delimiter='\t')
    # Header
    writer.writerow(['Test_Document_ID', 'Predicted_Category'])
    for docID, prediction in zip(test_ids, predictions):
        if prediction == 0:
            writer.writerow([docID, 'Business'])
        elif prediction == 1:
            writer.writerow([docID, 'Film'])
        elif prediction == 2:
            writer.writerow([docID, 'Football'])
        elif prediction == 3:
            writer.writerow([docID, 'Politics'])
        elif prediction == 4:
            writer.writerow([docID, 'Technology'])
