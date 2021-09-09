# Sentiment-Analysis-with-Insights
This project show the sentiment analysis of text data using NLP and Dash. I used Amazon reviews dataset to train the model and further scrap the reviews from Etsy.com in order to test my model.

Model- http://ec2-3-109-153-96.ap-south-1.compute.amazonaws.com:8050\

#
## How this project was made?
- This project has been built using Python3 to help predict the sentiments with the help of Machine Learning and an interactive dashboard to test reviews.
- To start, I downloaded the dataset and extracted the JSON file.
- Next, I took out a portion of 7,92,000 reviews equally distributed into chunks of 24000 reviews using pandas.
- The chunks were then combined into a single CSV file called balanced_reviews.csv.
- This balanced_reviews.csv served as the base for training my model which was filtered on the basis of review greater than 3 and less than 3.
- Further, this filtered data was vectorized using TF_IDF vectorizer.
- After training the model to a 90% accuracy, the reviews were scrapped from Etsy.com in order to test our model.
- Finally, I built a dashboard in which we can check the sentiments based on input given by the user or can check the sentiments of reviews scrapped from the website.

#
## What is CountVectorizer?
CountVectorizer is a great tool provided by the scikit-learn library in Python. It is used to transform a given text into a vector on the basis of the frequency (count) of each word that occurs in the entire text. This is helpful when we have multiple such texts, and we wish to convert each word in each text into vectors (for using in further text analysis).

CountVectorizer creates a matrix in which each unique word is represented by a column of the matrix, and each text sample from the document is a row in the matrix. The value of each cell is nothing but the count of the word in that particular text sample.

## What is TF-IDF Vectorizer?
TF-IDF stands for Term Frequency - Inverse Document Frequency and is a statistic that aims to better define how important a word is for a document, while also taking into account the relation to other documents from the same corpus.

This is performed by looking at how many times a word appears into a document while also paying attention to how many times the same word appears in other documents in the corpus.

The rationale behind this is the following:
- a word that frequently appears in a document has more relevancy for that document, meaning that there is higher probability that the document is about or in relation to that specific word
- a word that frequently appears in more documents may prevent us from finding the right document in a collection; the word is relevant either for all documents or for none. Either way, it will not help us filter out a single document or a small subset of documents from the whole set.

So then TF-IDF is a score which is applied to every word in every document in our dataset. And for every word, the TF-IDF value increases with every appearance of the word in a document, but is gradually decreased with every appearance in other documents.

## What is Plotly Dash?
Dash is a productive Python framework for building web analytic applications.

Written on top of Flask, Plotly.js, and React.js, Dash is ideal for building data visualization apps with highly custom user interfaces in pure Python. It's particularly suited for anyone who works with data in Python.

Dash apps are rendered in the web browser. You can deploy your apps to servers and then share them through URLs. Since Dash apps are viewed in the web browser, Dash is inherently cross-platform and mobile ready.

Dash is an open source library, released under the permissive MIT license. Plotly develops Dash and offers a platform for managing Dash apps in an enterprise environment.

## What is Web Scrapping?
Web scraping is a term used to describe the use of a program or algorithm to extract and process large amounts of data from the web.

![Screenshot 2021-06-28 233601](https://user-images.githubusercontent.com/56514396/126277385-29c2e98f-86fa-44eb-8e03-9bf002c2dd3d.png)
![Screenshot 2021-06-28 233655](https://user-images.githubusercontent.com/56514396/126277423-188612a1-958a-4b4a-bb35-b7bdae55ce0c.png)
![Screenshot 2021-06-28 233729](https://user-images.githubusercontent.com/56514396/126277257-010dc8fd-4b00-499b-b24f-278385d6b574.png)
#
## References
**[Forsk Coding School](http://forskcodingschool.com)** Internship conducted by **[Mr. Yogendra Singh](https://in.linkedin.com/in/yogendrasinsinwar)** and **[Dr. Sylvester Fernandes](https://in.linkedin.com/in/drsylvester)**
