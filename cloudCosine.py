import csv
from collections import OrderedDict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from wordcloud import WordCloud
import numpy as np
import matplotlib.pyplot as plt

# Format numpy decimals
float_formatter = lambda x: "%.4f" % x
np.set_printoptions(formatter={'float_kind': float_formatter})
# Threshold for calculating similar documents, default is 0.7 => the one the project requires
cosine_threshold = 0.7

# List that contains ordered corpus
# Also create one list for each category: will be used to create the 5 wordclouds
train_list = []
business_list = []
film_list = []
foot_list = []
politics_list = []
tech_list = []
# Dictionary that contains id-article (key,value) pair. will be converted to ordered.
train_dict = {}
# Read file and skip first row
print('Reading train_set.csv...')
with open('data/train_set.csv', 'r', encoding='utf8') as rf:
    reader = csv.reader(rf, delimiter='\t')
    headers = next(reader)
    for row in reader:
        train_dict[int(row[1])] = row[3]
        # Append categories to categories list
        # Business is 0, Film is 1, Football is 2, Politics is 3, Technology is 4
        if row[4] == 'Business':
            business_list.append(row[3])
        elif row[4] == 'Film':
            film_list.append(row[3])
        elif row[4] == 'Football':
            foot_list.append(row[3])
        elif row[4] == 'Politics':
            politics_list.append(row[3])
        elif row[4] == 'Technology':
            tech_list.append(row[3])
# Create ordered dictionary
ordered_data = OrderedDict(sorted(train_dict.items()))
print('Generating wordclouds...')
# Release some memory
# Might be useful in systems with low RAM
del train_dict

# Create training list with ordered corpus
for value in ordered_data.values():
    train_list.append(value)

# We need one corpus variable for each category to generate the 5 wordclouds
business_corpus = " ".join(article for article in business_list)
film_corpus = " ".join(article for article in film_list)
foot_corpus = " ".join(article for article in foot_list)
politics_corpus = " ".join(article for article in politics_list)
tech_corpus = " ".join(article for article in tech_list)

# Generate wordcloud from text for each category. This requires joining all the articles in each list to a single
# string, according to the API.
business_cloud = WordCloud(background_color="white").generate(business_corpus)
film_cloud = WordCloud(background_color="white").generate(film_corpus)
foot_cloud = WordCloud(background_color="white").generate(foot_corpus)
politics_cloud = WordCloud(background_color="white").generate(politics_corpus)
tech_cloud = WordCloud(background_color="white").generate(tech_corpus)

# Plot all wordclouds
plt.title("Business category")
plt.imshow(business_cloud, interpolation='bilinear')
plt.axis("off")
plt.show()
plt.title("Film category")
plt.imshow(film_cloud, interpolation='bilinear')
plt.axis("off")
plt.show()
plt.title("Football category")
plt.imshow(foot_cloud, interpolation='bilinear')
plt.axis("off")
plt.show()
plt.title("Politics category")
plt.imshow(politics_cloud, interpolation='bilinear')
plt.axis("off")
plt.show()
plt.title("Technology category")
plt.imshow(tech_cloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Transform raw data to tf idf matrix
tfidf_vectorizer = TfidfVectorizer()
print('Vectorizing data...')
X_train = tfidf_vectorizer.fit_transform(train_list)

# Calculate cosine similarities using the linear_kernel sklearn function
print('Calculating cosine similarities...')
cosine_similarities = linear_kernel(X_train, X_train)

print(cosine_similarities)
# Map array indexes to article ids
id_dict = {}
for index, docID in zip(range(0, 12266), ordered_data.keys()):
    id_dict[index] = docID

# Get ids of similar articles after converting and write to csv
# newline parameter is needed because Windows leaves a blank row for each record while writing in the csv file otherwise
with open('duplicatePairs.csv', 'w', encoding='utf8', newline='') as outfile:
    writer = csv.writer(outfile, delimiter='\t')
    # Header
    writer.writerow(['Document_ID1', 'Document_ID2', 'Similarity'])
    for index, cosSim in np.ndenumerate(cosine_similarities):
        if cosSim > 0.7:
            index = list(index)
            # Avoid duplicates, which means row index must be less than column, since cosine similarity is symmetrical
            if index[0] < index[1]:
                # Map the indexes to article ids
                index[0] = id_dict[index[0]]
                index[1] = id_dict[index[1]]
                # Sanity check and output
                print(index, cosSim)
                # Write results to csv file, float precision is 3 digits
                writer.writerow([index[0], index[1], "%.4f" % cosSim])

# Implementation with threshold
for index, cosSim in np.ndenumerate(cosine_similarities):
    if cosSim > cosine_threshold:
        index = list(index)
        # Avoid duplicates, which means row index must be less than column, since cosine similarity is symmetrical
        if index[0] < index[1]:
            # Map the indexes to article ids
            index[0] = id_dict[index[0]]
            index[1] = id_dict[index[1]]
            # Sanity check and output
            print(index, cosSim)
