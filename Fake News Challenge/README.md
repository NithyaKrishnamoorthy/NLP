## Fake News Stance Detection: A Machine Learning Approach
http://www.fakenewschallenge.org/

### Introduction and background
Fake news, defined by the New York Times as “a made-up story with an intention to deceive”, often for a secondary gain, is arguably one of the most serious challenges facing the news industry today. 
In a December Pew Research poll, 64% of US adults said that “made-up news” has caused a “great deal of confusion” about the facts of current events.

The combat over fake news evolved into a worldwide effort. In 2017, a Fake News Challenge4 was organized as a grassroots effort of over 100 volunteers lead by Dean Pomerleau from the Carnegie Mellon University. According to Dean Pomerleau, he was inspired by the simple yet elegant AI solution for spam, non-spam classification. The original research question behind the scene was, ‘can we solve fake news detection problem the same as spam vs. non-spam classification’?

The challenge proposed to focus on Stance Detection as a good first step towards fake news detection. The difference is, stance detection does not attempt to directly label the target as genuine or fake but classify the relative agreement between two pieces of text towards a claim or a topic, on whether they agree, disagree, discuss the claim, or they are unrelated. 
We think resolving stance detection would provide a solid foundation to fake news detection, because stance detection aligns with the thought process of a human fact-checker. 
According to the Challenge, the organizing team has interviewed human fact-checkers specializing in news opinions to obtain first-hand experience. 

### Dataset
The dataset comprises of two CSV files. 
<ul>
<li>Train_bodies.csv - This file contains the body text of articles (the articleBody column) with corresponding IDs (Body ID)</li>
<li>Train_stances.csv - It contains close to 50,000 stance instances, where each instance consists of headline, ID, stance.</li>
<li>Headline – A headline which is to be compared against the article mentioned in the body to determine its stance toward the article. Word counts for the headlines range from 2 to approximately 40, with an average length of ~11. </li>
<li>ID - ID of an article, this was used as the mapping key to construct the headline / article body pair</li>
<li>Stance – Ground Truth stance of the headline with respect to the article. It is divided into 4 classes: unrelated, discuss, agree, disagree</li>
</ul>
 
According to the wesbite, it is challenging to obtain good quality training data from accredited news resources, mainly due to copyright issue; hence the dataset being used was prepared in the following manner:
The dataset was first prepared by taking a total of 1,683 unique pair of headlines and articles of which the stance has been labelled. Most headlines were then randomly matched against many different articles, to construct varying stances, eventually close to 50,000 instances was generated. 

### Features
<table>
<tr>
<td>Bin_early_count</td> 	<td>Number of unigrams in the headline that appears in the early part of the article body</td>
</tr>
<tr>
<td>Bin_count</td> 	<td>	Number of unigrams in the headline that appears in the entire article body</td></tr>
<tr><td>Bin_early_stop_count</td> 	<td>	Number of unigram (excluding stop words) in the headline that appears in the early part of the article body</td></tr>
<tr><td>Bin_stop_count</td> 	<td>	Number of unigram (excluding stop words) in the headline that appears in the entire article body</td></tr>
<tr><td>Bigram_early_count</td> 	<td> 	Number of bigrams in the headline that appears in the early part of the article body</td></tr>
<tr><td>Bigram _count</td> 	<td>	Number of bigrams in the headline that appears in the entire article body</td></tr>
<tr><td>Trigram_early_count</td> 	<td> 	Number of Trigram in the headline that appears in the early part of the article body</td></tr>
<tr><td>Trigram _count</td> 	<td>	Number of Trigram in the headline that appears in the entire article body</td></tr>
<tr><td>Headline_sentiment </td> 	<td> 	Compound sentiment score of the headline, obtained using VADER</td></tr>
<tr><td>Articlebody_sentiment</td> 	<td>	Compound sentiment score of the article body, obtained using VADER</td></tr>
<tr><td>TFIDF vector with Cosine Similarity	</td> 	<td> The value of the cosine similarity is the feature value. Higher the value, higher the degree of similarity and vice versa.</td>
</tr>
</table>

### Model
SVM and Logistic regression classifiers are used to perform the classification.

### Evaluation Metric

The official metric of the Fake News Challenge is a score based on how many predictions were correct. 
The different stance classes were weighted. If the model correctly identifies the related/unrelated stance for a headline-body pair, it is awarded 0.25 points and if it correctly predicts any of the other three classes, it is awarded 0.75 points. 
The reason for this is that the task of determining unrelated/related is trivial compared to determining other classes.

The following metrics were used to evaluate:
<ul>
<li>Classification Accuracy to get the number of correct predictions from all predictions made.</li>
<li>F1 Score is the harmonic average of Precision and Recall. Takes false positives and false negatives into account.</li>
<li>Confusion Matrix to describe the performance of a classification model on a set of test data for which the true values are known. Classification accuracy alone can be misleading as we have more than 2 classes in the dataset and have an unequal number of observations in each class.</li>
</ul>

### Results
SVM model gave a score of 7026.5 out of 8985, while the logistic regression model gave a score of 7261.5 out of 8985
Even though SVM model has a higher score, the logistic regression model has higher correct prediction on disagree/agree/discuss classes. 
Thus, it is suggested to use different machine learning models for different goals.
