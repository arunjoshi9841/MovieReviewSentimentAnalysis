# MovieReviewSentimentAnalysis

##Application workflow

###Step 1: load the dataset
###Step 2 : Split the sentence with respect to whitespace. Treat every single word as independent. 
###Step3: Calculate probability of a review being positive/ negative
###Step 3: Count the presence of every word in every review and determine the probability of a word being of either negative connotation or positive connotation.

Now to test if a review is positive or negative we use
	Bayesian formula = p(A|B) = p(B|A) * p(A) / p(B)
		Now, Probability of review B being Positive =  (summation of Probability of finding every word in positive review* probability of any review being positive)
We exclude the denominator as its only purpose is to scale the numerator.
we’re multiplying many probabilities together. we end up with really small numbers. To prevent this, we’re going to look at the log probability by taking the log of each side. 
