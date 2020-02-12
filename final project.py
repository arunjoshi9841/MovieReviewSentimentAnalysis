
# coding: utf-8

# In[1]:


import csv
from collections import Counter
import re
from sklearn import metrics


# In[2]:


with open("train.csv", 'r') as file:
    train = list(csv.reader(file))

with open("test.csv", 'r') as file:
    test = list(csv.reader(file))


# In[3]:


def get_H_count(score):
    # Compute the count of each classification occurring in the data
    return len([r for r in test if r[1] == str(score)])


# In[4]:


# We'll use these counts for smoothing when computing the prediction
positive_review_count = get_H_count(1)
negative_review_count = get_H_count(-1)


# In[5]:


# These are the prior probabilities (we saw them in the formula as P(H))
prob_positive = positive_review_count / len(train)
prob_negative = negative_review_count / len(train)


# In[6]:


def get_text(train, score):
    # Join together the text in the reviews for a particular tone
    # Lowercase the text so that the algorithm doesn't see "Not" and "not" as different words, for example
    return " ".join([r[0].lower() for r in train if r[1] == str(score)])


# In[7]:


def count_text(text):
    # Split text into words based on whitespace -- simple but effective
    words = re.split("\s+", text)
    # Count up the occurrence of each word
    return Counter(words)


# In[8]:


negative_text = get_text(train, -1)
positive_text = get_text(train, 1)


# In[9]:


# Generate word counts(WC) dictionary for negative tone
negative_WC_dict = count_text(negative_text)


# In[10]:


# Generate word counts(WC) dictionary for positive tone
positive_WC_dict = count_text(positive_text)


# In[11]:


# H = positive review or negative review
def make_class_prediction(text, H_WC_dict, H_prob, H_count):
    prediction = 1
    text_WC_dict = count_text(text)
    
    for word in text_WC_dict:       
        prediction *=  text_WC_dict.get(word,0) * ((H_WC_dict.get(word, 0) + 1) / (sum(H_WC_dict.values()) + H_count))

        # Now we multiply by the probability of the class existing in the documents
    return prediction * H_prob


# In[12]:


# Now we can generate probabilities for the classes our reviews belong to
# The probabilities themselves aren't very useful -- we make our classification decision based on which value is greater
def make_decision(text):
    
    # Compute the negative and positive probabilities
    negative_prediction = make_class_prediction(text, negative_WC_dict, prob_negative, negative_review_count)
    positive_prediction = make_class_prediction(text, positive_WC_dict, prob_positive, positive_review_count)

    # We assign a classification based on which probability is greater
    if negative_prediction > positive_prediction:
        return -1
    return 1


# In[13]:


def make_decision1(text):
    
    # Compute the negative and positive probabilities
    negative_prediction = make_class_prediction(text, negative_WC_dict, prob_negative, negative_review_count)
    positive_prediction = make_class_prediction(text, positive_WC_dict, prob_positive, positive_review_count)

    # We assign a classification based on which probability is greater
    if negative_prediction > positive_prediction:
        return -1
    return 1


# In[14]:


predictions = [make_decision(r[0]) for r in test]
actual = [int(r[1]) for r in test]
#print(predictions)
#print(actual)


# In[15]:


# Generate the ROC curve using scikits-learn
fpr, tpr, thresholds = metrics.roc_curve(actual, predictions, pos_label=1)

# Measure the area under the curve
# The closer to 1 it is, the "better" the predictions
print("AUC of the predictions: {0}".format(metrics.auc(fpr, tpr)))


# In[19]:


name = input("Enter a review to classify")
make_decision1(name)

