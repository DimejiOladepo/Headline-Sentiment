# News-Headiline-Classification-using-Embeddings-and-TF-IDF

Four models were developed to classify a corpus of news headlines with labels attached as either "Positive", "Negative", "Even". 

The news corpus was pre-processed and cleaned to remove special characters, double spaces, single characters using regular expressions.

In the first model, embeddings were learned from the corpus with each sentence tokenized, sequenced and used as a hidden layer for a neural network
to determine the cross-validation and test scores.

Gensim was used create word embeddings for the subsequent word2vec model with learned embedding saved locally. The word2vec embeddings 
were imported and mapped to each word in the vocabulary and a matrix of words was created. This matrix was used as an embedding matrix for the 
neural network created. This model did better than the first but there was still room for improvement.

The third was implemented using TF-IDF using n-grams approach. Four separate classifiers were also used and compared. OVR classifier did better with
a cross-validation and test accuracy better than the Word2Vec.

The last was implemented also using TF-IDF but using max_features approach and this did better than the others with a test accuracy of about
53% using naive bayes classifier.

Important to note:
- The news headlines was a small dataset with 1000 entries 
- It was also imbalanced with the even class double the positive class.
- Some headlines could have been mislabelled.

Key Take-aways:
- Balancing the classes with a dataset this small doesn't largely improve accuracy (Discovered by setting OVR to balanced class)
- A dataset can be trained on to improve accuracy.
- Using a large pre-trained embedding on News could have made a better classifier.
