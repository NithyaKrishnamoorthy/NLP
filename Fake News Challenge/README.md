## Fake News Stance Detection: A Machine Learning and Deep Learning Approach

### Introduction and background
Fake news, defined by the New York Times as “a made-up story with an intention to deceive”, often for a secondary gain, is arguably one of the most serious challenges facing the news industry today. 
In a December Pew Research poll, 64% of US adults said that “made-up news” has caused a “great deal of confusion” about the facts of current events.

The combat over fake news evolved into a worldwide effort. In 2017, a Fake News Challenge4 was organized as a grassroots effort of over 100 volunteers lead by Dean Pomerleau from the Carnegie Mellon University. According to Dean Pomerleau, he was inspired by the simple yet elegant AI solution for spam, non-spam classification. The original research question behind the scene was, ‘can we solve fake news detection problem the same as spam vs. non-spam classification’?

The challenge proposed to focus on Stance Detection as a good first step towards fake news detection. The difference is, stance detection does not attempt to directly label the target as genuine or fake but classify the relative agreement between two pieces of text towards a claim or a topic, on whether they agree, disagree, discuss the claim, or they are unrelated. 
We think resolving stance detection would provide a solid foundation to fake news detection, because stance detection aligns with the thought process of a human fact-checker. 
According to the Challenge, the organizing team has interviewed human fact-checkers specializing in news opinions to obtain first-hand experience. 

### Dataset
The dataset comprises of two CSV files. 
•	Train_bodies.csv - This file contains the body text of articles (the articleBody column) with corresponding IDs (Body ID)
•	Train_stances.csv - It contains close to 50,000 stance instances, where each instance consists of headline, ID, stance.
•	Headline – A headline which is to be compared against the article mentioned in the body to determine its stance toward the article. Word counts for the headlines range from 2 to approximately 40, with an average length of ~11. 
•	ID - ID of an article, this was used as the mapping key to construct the headline / article body pair
•	Stance – Ground Truth stance of the headline with respect to the article. It is divided into 4 classes: unrelated, discuss, agree, disagree
 
According to the Challenge4, it is challenging to obtain good quality training data from accredited news resources, mainly due to copyright issue; hence the dataset being used was prepared in the following manner:
The dataset was first prepared by taking a total of 1,683 unique pair of headlines and articles of which the stance has been labelled. Most headlines were then randomly matched against many different articles, to construct varying stances, eventually close to 50,000 instances was generated. 

### Features
Bin_early_count 	Number of unigrams in the headline that appears in the early part of the article body
Bin_count	Number of unigrams in the headline that appears in the entire article body
Bin_early_stop_count	Number of unigram (excluding stop words) in the headline that appears in the early part of the article body
Bin_stop_count	Number of unigram (excluding stop words) in the headline that appears in the entire article body
Bigram_early_count 	Number of bigrams in the headline that appears in the early part of the article body
Bigram _count	Number of bigrams in the headline that appears in the entire article body
Trigram_early_count 	Number of Trigram in the headline that appears in the early part of the article body
Trigram _count	Number of Trigram in the headline that appears in the entire article body
Headline_sentiment  	Compound sentiment score of the headline, obtained using VADER
Articlebody_sentiment	Compound sentiment score of the article body, obtained using VADER
TFIDF vector with Cosine Similarity	- The value of the cosine similarity is the feature value. Higher the value, higher the degree of similarity and vice versa.

### Deep Learning

•	Bag of Words
For a simple baseline, a bag of embedded words model was constructed where the embedding vectors of the headline and the body were averaged separately, concatenated, and used as the input to a feed to a feed-forward neural network with a softmax output layer.

•	Long Short-Term Memory Network (LSTM)
A standard LSTM model that processes a concatenation of tokens in the headline and the article body to produce a classification of the stance. 
