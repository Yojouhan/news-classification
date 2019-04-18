# news-classification
A news classification system I developed in my Data Mining course in university. There are 5 possible categories (Business, Film, Football,
Politics and Technology), and each article belongs in one category, making the problem multiclass. A variety of approaches and classifiers
is used, including random forests and SVMs. Peak accuracy observed is around 97%, with the use of sklearn's RidgeClassifier.

The dataset was provided by the instructor, and I'm not sure if it can be found online. It looks like it contains 
scraped articles from a website. For the moment, I'll not be uploading it but I can 
describe its format. It's a csv with 5 columns : RowNum, Id, Title, Content, Category. The columns' contents are self-explanatory. Should
I obtain further information regarding the dataset, I'll go ahead and revise this section and upload it. 

Aside from the training set, there exists a file called test_set.csv which contains test articles without the category. The original task was 
to predict their category.

Additionally, the file called cloudCosine.py generates a wordcloud and computes the cosine similarity between documents.

This project was purely of academic nature. 
