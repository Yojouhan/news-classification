import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from wordcloud import WordCloud

# Format numpy decimals
float_formatter = lambda x: "%.4f" % x
np.set_printoptions(formatter={'float_kind': float_formatter})
# Threshold for calculating similar documents, default is 0.7 => the one the project requires
cosine_threshold = 0.7
labels_dict = {'Football': 0, 'Business': 1, 'Politics': 2, 'Film': 3, 'Technology': 4}

# Create one list for each category: will be used to create the 5 wordclouds
corpus_dict = {'Business': [], 'Film': [], 'Football': [], 'Politics': [], 'Technology': []}
corpus_list = []  # For cosine similarity use later on
df = pd.read_csv('data/train_set.csv', delimiter='\t', usecols=['Id', 'Content', 'Category'])
train_dict = dict(zip(df['Id'], df['Content']))  # Dictionary that contains id-article (key,value) pair. will be
# converted to ordered.
train_list = []

# Append to corpus dict
for article, category in zip(df['Content'], df['Category']):
    corpus_dict[category].append(article)

# Create ordered dictionary
ordered_data = OrderedDict(sorted(train_dict.items()))

# Release some memory
# Might be useful in systems with low RAM
del train_dict

# Create training list with ordered corpus
for value in ordered_data.values():
    train_list.append(value)

# Plot all wordclouds
print('Generating wordclouds')
fig, axes = plt.subplots(2, 3)
for category, articles, indexes in zip(corpus_dict.keys(), corpus_dict.values(),
                                       [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]):
    cloud = WordCloud().generate(''.join(article for article in articles))
    axes[indexes[0], indexes[1]].set_title(category + ' category')
    axes[indexes[0], indexes[1]].imshow(cloud, interpolation='bilinear')
    axes[indexes[0], indexes[1]].set_axis_off()

fig.delaxes(axes[1, 2])
plt.show()

# Transform raw data to tf idf matrix
del corpus_dict
tfidf_vectorizer = TfidfVectorizer()
print('Vectorizing data...')
X_train = tfidf_vectorizer.fit_transform(train_list)

# Calculate cosine similarities using the linear_kernel sklearn function
print('Calculating cosine similarities...')
cosine_similarities = linear_kernel(X_train, X_train)

print(cosine_similarities)
# Map array indexes to article ids
id_dict = {}
for index, doc_id in zip(range(0, 12266), ordered_data.keys()):
    id_dict[index] = doc_id

# Get ids of similar articles after converting and write to csv
# newline parameter is needed because Windows leaves a blank row for each record while writing in the csv file otherwise
with open('duplicatePairs.csv', 'w', encoding='utf8', newline='') as outfile:
    writer = csv.writer(outfile, delimiter='\t')
    # Header
    writer.writerow(['Document_ID1', 'Document_ID2', 'Similarity'])
    for index, cos_sim in np.ndenumerate(cosine_similarities):
        if cos_sim > 0.7:
            index = list(index)
            # Avoid duplicates, which means row index must be less than column, since cosine similarity is symmetrical
            if index[0] < index[1]:
                # Map the indexes to article ids
                index[0] = id_dict[index[0]]
                index[1] = id_dict[index[1]]
                # Sanity check and output
                print(index, cos_sim)
                # Write results to csv file, float precision is 4 digits
                writer.writerow([index[0], index[1], "%.4f" % cos_sim])

# Implementation with threshold
for index, cos_sim in np.ndenumerate(cosine_similarities):
    if cos_sim > cosine_threshold:
        index = list(index)
        # Avoid duplicates, which means row index must be less than column, since cosine similarity is symmetrical
        if index[0] < index[1]:
            # Map the indexes to article ids
            index[0] = id_dict[index[0]]
            index[1] = id_dict[index[1]]
            # Sanity check and output
            print(index, cos_sim)
