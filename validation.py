import sys
import os
import baseline
import hmm
import re
import random
from preprocess import *
from constants import *


# // <--------------------- Calculate precision, recall, f-score ---------------------> //

""" Returns: The resulting quotient after deviding numerator by denominator. This function will
	handle safe division of count values and precision/recall values. The behavior is dependent
	upon the argument for calculation.

	Precondition:	calculation = 	either 'precision', 'recall', or 'fscore'
					numerator = 	an int or float representing the numerator
					denominator = 	an int or float representing the denominator """
def divide_safely(calculation, numerator, denominator):

	quotient = 0

	# precision: if denominator == 0, numerator must be 0 too, so precision = 1
	if calculation == 'precision':
		if denominator == 0:
			quotient = 1
		else:
			quotient = float(numerator) / float(denominator)

	# recall: if denom == 0, recall = 1 only if numerator == 0, else recall = 0
	elif calculation == 'recall':
		if denominator == 0 and numerator == 0:
			quotient = 1
		elif denominator == 0 and numerator != 0:
			quotient = 0
		else:
			quotient = float(numerator) / float(denominator)

	# fscore: if precision AND recall == 0, fscore = 0
	elif calculation == 'fscore':
		if denominator == 0:
			quotient = 0
		else:
			quotient = float(numerator) / float(denominator)

	return quotient


""" Returns: Precision values for B, I, and O tags. A number between 0 and 1. Computed by getting 
	the number of correct tags and dividng by the total number of tags written by our system.

	Precondition:	output_tags = 	list of 3-tuples (token, POS tag, BIO tag) that result from 
									our output text files
					correct_tags =	list of 3-tuples (token, POS tag, BIO tag) from the original 
									text files, will be used to check correctness of our output
					len(output_tags) == len(correct_tags) """
def calculate_precision(output_tags, correct_tags):

	# initialize counters for each of the 3 BIO tags
	correct_b = correct_i = correct_o = 0
	b_count = i_count = o_count = 0 # these counts will be from the output sequence

	# loop through output tags list, update counts
	for i in xrange( len(output_tags) ):

		if output_tags[i] == B_CUE_TAG:
			if output_tags[i] == correct_tags[i]:
				correct_b += 1
			b_count += 1

		elif output_tags[i] == I_CUE_TAG:
			if output_tags[i] == correct_tags[i]:
				correct_i += 1
			i_count += 1

		elif output_tags[i] == O_TAG:
			if output_tags[i] == correct_tags[i]:
				correct_o += 1
			o_count += 1

	# return tuple of average of the 3 precisions
	b_precision = divide_safely('precision', correct_b, b_count)
	i_precision = divide_safely('precision', correct_i, i_count)
	o_precision = divide_safely('precision', correct_o, o_count)

	return (b_precision + i_precision + o_precision) / 3


""" Returns: Recall values for B, I, and O tags. A number between 0 and 1. Computed by getting 
	the number of correct tags and dividng by the number of tags in the 'answer key'.

	Precondition:	output_tags = 	list of 3-tuples (token, POS tag, BIO tag) that 
									result from our output text files
					correct_tags =	list of 3-tuples (token, POS tag, BIO tag) from 
									the original text files, will be used to check 
									correctness of our output
					len(output_tags) == len(correct_tags) """
def calculate_recall(output_tags, correct_tags):

	# initialize counters for each of the 3 BIO tags
	correct_b = correct_i = correct_o = 0
	b_count = i_count = o_count = 0 # these counts will be from answer key

	# loop through correct tags list, update counts
	for i in xrange( len(correct_tags) ):

		if correct_tags[i] == B_CUE_TAG:
			if output_tags[i] == correct_tags[i]:
				correct_b += 1
			b_count += 1

		elif correct_tags[i] == I_CUE_TAG:
			if output_tags[i] == correct_tags[i]:
				correct_i += 1
			i_count += 1

		elif correct_tags[i] == O_TAG:
			if output_tags[i] == correct_tags[i]:
				correct_o += 1
			o_count += 1

	# return tuple of average of the 3 recalls
	b_recall = divide_safely('recall', correct_b, b_count)
	i_recall = divide_safely('recall', correct_i, i_count)
	o_recall = divide_safely('recall', correct_o, o_count)

	return (b_recall + i_recall + o_recall) / 3	


""" Returns: f-score calculated using the harmonic mean of precision and recall: 
	(2 * P * R) / (P + R)

	Precondition: 	precision = 	precision value, a float between 0 and 1
					recall =	 	recall value, a float between 0 and 1 """
def calculate_fscore(precision, recall):
	
	return divide_safely('fscore', (2 * precision * recall), (precision + recall))


# // <------------------------------ Cross Validation Helpers ------------------------------> //

""" Returns: A 2D list of size k in which each element is a randomized subset of training doc 
	indices, each of which will be used as a validation set once (the other 9 will be used as
	training sets).

	Precondition: 	directory = 	path to diretory containing training docs
					k = 			number of folds we will use to validate """
def breakup_training(directory, k):

	# determine size of subsets using k and number of docs in directory
	subset_size = len(os.listdir(directory)) / k

	# list of all indices based on how many training docs are available
	doc_list = [i for i in xrange( len(os.listdir(directory)) )]
	random.shuffle(doc_list)

	# break the doc_list into k amount of subsets
	subset_list = []
	for i in xrange(k):
		if i == k - 1:
			subset_list.append( doc_list )
		else:
			subset_list.append( doc_list[:subset_size] )
			doc_list = doc_list[subset_size:]

	return subset_list


""" Returns: A tuple (precision, recall, f-score) for the current fold of cross-validation
	************************************ FOR HMM ONLY ************************************

	Precondition: 	directory = 		path to directory containing training docs
					hmm = 				a valid hidden markov model for the current fold
					validation_set = 	a set of indices of train docs to validate """
def cross_validate_hmm(directory, hmm, validation_set):

	sorted_files = sorted( os.listdir(directory), key=lambda x: ( int( re.sub('\D', '', x) ), x) )
	triplet_list = [] # will contain the 'answers', in order of sorted files
	viterbi_seq = [] # will contain the BIO tag sequence from Viterbi, in order of sorted files

	# loop through sorted files
	for i in xrange( len(sorted_files) ):

		# extract only the files in validation_set
		file_name = sorted_files[i]
		if file_name.endswith('.txt') and i in validation_set:
			file_path = os.path.join(directory, file_name)

			# first populate the triplet_list so we have all the info from our validation set
			tags_list = preprocess(file_path)
			triplet_list += tags_list

			if tags_list:
				# next isolate only the tokens and run Viterbi, store the accumulating sequence 
				tokens_list = [token for (token, pos_tag, bio_tag) in tags_list]
				viterbi_seq += hmm.viterbi_decode(tokens_list)

	# finally, obtain results metrics: calculate precision, recall, f-score
	correct_tag_seq = [bio_tag for (token, pos_tag, bio_tag) in triplet_list]
	precision = calculate_precision(viterbi_seq, correct_tag_seq)
	recall = calculate_recall(viterbi_seq, correct_tag_seq)
	fscore = calculate_fscore(precision, recall)

	return (precision, recall, fscore)


""" Returns: A tuple (precision, recall, f-score) for the current fold of cross-validation
	 ********************************* FOR BASELINE ONLY *********************************

	Precondition: 	directory = 		path to directory containing training docs
					baseline = 			baseline model of the specified training set
					validation_set = 	a set of indices of train docs to validate """
def cross_validate_baseline(directory, baseline, validation_set):

	sorted_files = sorted( os.listdir(directory), key=lambda x: ( int( re.sub('\D', '', x) ), x) )
	triplet_list = [] # will contain the 'answers', in order of sorted files
	baseline_seq = [] # will contain the BIO tag sequence from our Baseline, in order of sorted files

	# loop through sorted files
	for i in xrange( len(sorted_files) ):

		# extract only the files in validation_set
		file_name = sorted_files[i]
		if file_name.endswith('.txt') and i in validation_set:
			file_path = os.path.join(directory, file_name)

			# first populate the triplet_list so we have all the info from our validation set
			tags_list = preprocess(file_path)
			triplet_list += tags_list

			# next isolate only the tokens and run our Baseline, store the accumulating sequence 
			tokens_list = [token for (token, pos_tag, bio_tag) in tags_list]

			was_last_cue = False
			for t in tokens_list:
				if t in baseline.getBaseline():
					most_freq_tag = baseline.getBaseline()[t]
					if most_freq_tag == O_TAG:
						baseline_seq.append(O_TAG)
						was_last_cue = False
					elif was_last_cue:
						baseline_seq.append(I_CUE_TAG)
					else:
						baseline_seq.append(B_CUE_TAG)
						was_last_cue = True
				else:
					baseline_seq.append(O_TAG)
					was_last_cue = False

	# finally, obtain results metrics: calculate precision, recall, f-score
	correct_tag_seq = [bio_tag for (token, pos_tag, bio_tag) in triplet_list]
	precision = calculate_precision(baseline_seq, correct_tag_seq)
	recall = calculate_recall(baseline_seq, correct_tag_seq)
	fscore = calculate_fscore(precision, recall)

	return (precision, recall, fscore)


# // <------------------------------ k-fold Cross Validation ------------------------------> //

""" Returns: A list of (precision, recall, f-score) tuples for each of the 10 different models.

	k-fold cross-validation will be implemented as such: break the training docs list to k segments, 
	run k different iterations of training + validation in which each of the k segments will be 
	chosen as a validation set while the other (k - 1) will be the training set. 

	We will record the precision, recall, and f-scores for each fold and return the average values 
	for each of the 10 models across k iterations of valdiation. 

	Precondition: 	directory = 	path to directory containing training docs.
					k = 			number of folds we will use to validate """
def kfold_cross_validate(directory, k):

	print 'Beginning k-fold cross validation...'

	subset_list = breakup_training(directory, k)
	results = [[] for i in xrange(10)] # outer array = each model, inner array = results per iteration

	# loop through each subset list, run training + validation
	for i in xrange( len(subset_list) ):

		# split the training docs into training + validation
		validation_set = set( subset_list[i] )
		remaining = subset_list[:i] + subset_list[i + 1:]
		train_set = set( [index for subset in remaining for index in subset] )

		# no resampling 
		hmm_model_0 = hmm.HMM(directory, train_set, smooth_trans=True, smooth_emiss=True, resample=False) # smooth both
		hmm_model_1 = hmm.HMM(directory, train_set, smooth_trans=False, smooth_emiss=True, resample=False) # smooth emission only
		hmm_model_2 = hmm.HMM(directory, train_set, smooth_trans=True, smooth_emiss=False, resample=False) # smooth transition only
		hmm_model_3 = hmm.HMM(directory, train_set, smooth_trans=False, smooth_emiss=False, resample=False) # no smoothing

		results[0].append( cross_validate_hmm(directory, hmm_model_0, validation_set) )
		results[1].append( cross_validate_hmm(directory, hmm_model_1, validation_set) )
		results[2].append( cross_validate_hmm(directory, hmm_model_2, validation_set) )
		results[3].append( cross_validate_hmm(directory, hmm_model_3, validation_set) )

		# with resampling
		hmm_model_4 = hmm.HMM(directory, train_set, smooth_trans=True, smooth_emiss=True, resample=True) # smooth both
		hmm_model_5 = hmm.HMM(directory, train_set, smooth_trans=False, smooth_emiss=True, resample=True) # smooth emission only
		hmm_model_6 = hmm.HMM(directory, train_set, smooth_trans=True, smooth_emiss=False, resample=True) # smooth transition only
		hmm_model_7 = hmm.HMM(directory, train_set, smooth_trans=False, smooth_emiss=False, resample=True) # no smoothing

		results[4].append( cross_validate_hmm(directory, hmm_model_4, validation_set) )
		results[5].append( cross_validate_hmm(directory, hmm_model_5, validation_set) )
		results[6].append( cross_validate_hmm(directory, hmm_model_6, validation_set) )
		results[7].append( cross_validate_hmm(directory, hmm_model_7, validation_set) )

		# baseline with and without resampling
		baseline_1 = baseline.Baseline(directory, train_set, resample=False)
		baseline_2 = baseline.Baseline(directory, train_set, resample=True)

		results[8].append( cross_validate_baseline(directory, baseline_1, validation_set) )
		results[9].append( cross_validate_baseline(directory, baseline_2, validation_set) )

		# status update
		print str((float(i + 1) / k) * 100) + '% complete'

	# return the avg results tuple for each model that we train/test across all k-fold cross-validation rounds
	return [get_avg_results(model_results, k) for model_results in results]


# // <------------------------------ Results & Analysis ------------------------------> //

""" Returns: A tuple containing (avg precision, avg recall, avg f-score) over all k rounds of 
	validation.

	Precondition:	results =	a 2D list where the outer layer represents results for each model,
								while the inner layer holds the k rounds of results for each model
					k = 		number of folds we will use to validate """
def get_avg_results(results, k):

	# sum up all the values, divide each by k
	(precision, recall, fscore) = reduce((lambda (a,b,c), (d,e,f): (a+d, b+e, c+f)), results)
	return (precision / k, recall / k, fscore / k)


""" Procedure: Prints analysis information to the terminal, tell user the results of the
	kfold cross-validation

	Precondition: 	results_by_model = 	a list of 3-tuples containing the (avg precision,
										avg recall, and avg f-score) across all k rounds of
										validation for each of 10 models """
def analyze_results(results_by_model):

	best_model = 0
	for i in xrange( len(results_by_model) ):

		(precision_best, recall_best, fscore_best) = results_by_model[best_model]
		(precision, recall, fscore) = results_by_model[i]

		if fscore > fscore_best:
			best_model = i

		print 'Model ' + `(i + 1)` + ' has precision: ' + `precision` + ', recall: ' + \
		 `recall` + ', and f-score: ' + `fscore`

	(precision, recall, fscore) = results_by_model[best_model]
	if best_model == 0:
		print 'Model 1 (Hidden Markov Model with smoothed transition and emission probabilities and no resampling) performed the best with f-score ' + `fscore`
	elif best_model == 1:
		print 'Model 2 (Hidden Markov Model with smoothed emission probabilities and no resampling) performed the best with f-score ' + `fscore`
	elif best_model == 2:
		print 'Model 3 (Hidden Markov Model with smoothed transition probabilities and no resampling) performed the best with f-score ' + `fscore`
	elif best_model == 3:
		print 'Model 4 (Hidden Markov Model with no smoothing and no resampling) performed the best with f-score ' + `fscore`
	elif best_model == 4:
		print 'Model 5 (Hidden Markov Model with smoothed transition and emission probabilities and resampling) performed the best with f-score ' + `fscore`
	elif best_model == 5:
		print 'Model 6 (Hidden Markov Model with smoothed emission probabilities and resampling) performed the best with f-score ' + `fscore`
	elif best_model == 6:
		print 'Model 7 (Hidden Markov Model with smoothed transition probabilities and resampling) performed the best with f-score ' + `fscore`
	elif best_model == 7:
		print 'Model 8 (Hidden Markov Model with no smoothing and resampling) performed the best with f-score ' + `fscore`
	elif best_model == 8:
		print 'Model 9 (Baseline model with no resampling) performed the best with f-score ' + `fscore`
	elif best_model == 9:
		print 'Model 10 (Baseline model with resampling) performed the best with f-score ' + `fscore`


""" Main Function: This method will run the k-fold cross validation for our 10 different models,
	retrieve the relevant precision, recall, and f-scores for each different model over 10 iterations,
	and display all the results on the command line.

	Precondition: 	sys.argv[1] = path to directory containing training docs """
def main():

	results_by_model = kfold_cross_validate( sys.argv[1], 10 )
	analyze_results(results_by_model)


if __name__ == '__main__':
	main()

