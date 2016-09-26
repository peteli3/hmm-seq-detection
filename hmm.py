import sys
import os
import re
import math
from preprocess import *
from constants import *


# // <------------------------------ HMM Definition ------------------------------> //

class HMM(object):

	""" Initializer: Initializes an instance of Hidden Markov Model using 
		specified training docs in the specified directory.

		Precondition: 	directory = path to directory containing training docs
						train_set = set of ints (smallest being 0) indicating 
									which files we choose to use in training our model
						smooth_trans = either true or false, determines if we 
									   smooth transition probabilities
						smooth_emiss = either true or false, determines if we 
									   smooth emission probabilities 
						resample = true or false indicating whether we upsample
							   	   sentences with uncertainty and downsample sentences
							   	   without to balance the dataset """
	def __init__(self, directory, train_set, smooth_trans=True, smooth_emiss=False, resample=True):
		self.smooth_trans = smooth_trans
		self.smooth_emiss = smooth_emiss
		self.resample = resample
		self.triplet_list = self.get_triplet_list(directory, train_set)
		self.vocab = self.get_vocab() # set of vocabulary of words
		self.states = [B_CUE_TAG, I_CUE_TAG, O_TAG] # BIO tags
		self.start_table = self.build_start_table() # frequency of BIO tags
		self.transition_table = self.build_transition_table() # table of P(tag | previous tag)
		self.emission_table = self.build_emission_table() # table of P(token | tag)


	##################### Internal Functions for Training #####################


	""" Returns: Triplet list from the directory containing specified training 
		text data, with all unknowns marked as such.

		Precondition:	directory = path to directory containing training docs
						train_set = set of ints (smallest being 0) indicating 
									which files we choose to use in training our model """
	def get_triplet_list(self, directory, train_set):

		triplet_list = []

		# sort files in increasing alphanumeric order
		sorted_files = sorted( os.listdir(directory), key=lambda x: ( int( re.sub('\D', '', x) ), x) )

		for i in xrange( len(sorted_files) ):

			file_name = sorted_files[i]

			if file_name.endswith('.txt') and i in train_set:
				file_path = os.path.join(directory, file_name)
				tags_list = preprocess(file_path, self.resample)
				triplet_list += self.handle_unknown(tags_list)

		return triplet_list


	""" Returns: triplet list where every tenth token of count 1 is replaced with unknown tag.

		Precondition:	triplet_list = 	list of triplets in the form:
										(token, POS tag, BIO tag) """
	def handle_unknown(self, triplet_list):

		count_by_word = {} # indexed by token, value = count

		# build counting dictionary
		for (token, pos_tag, bio_tag) in triplet_list:

			if token not in count_by_word:
				count_by_word[token] = 1
			else:
				count_by_word[token] += 1

		# replace words with unknowns where necessary
		counter = 0
		for i in range( len(triplet_list) ):

			(token, pos_tag, bio_tag) = triplet_list[i]

			if count_by_word[token] == 1:

				if counter % 10 == 0:
					token = UNKNOWN_TAG

				counter += 1

			triplet_list[i] = (token, pos_tag, bio_tag)

		return triplet_list


	""" Returns: the set of vocabulary built from the triplet list. """
	def get_vocab(self):

		vocab = set()

		for (token, pos_tag, bio_tag) in self.triplet_list:

			if token not in vocab:
				vocab.add(token)

		return vocab


	""" Returns: Table containing the probabilities derived from relative 
		frequencies of B-CUE, I-CUE, or O tags. Since a sequence of words
		cannot start with a I-CUE tag, we place its starting weight on
		B-CUE instead. """
	def build_start_table(self):

		probs = { B_CUE_TAG : 0, I_CUE_TAG : 0, O_TAG : 0 }
		total_count = 0

		for (token, pos_tag, bio_tag) in self.triplet_list:

			if bio_tag == B_CUE_TAG or bio_tag == I_CUE_TAG:
				probs[B_CUE_TAG] += 1
			elif bio_tag == O_TAG:
				probs[O_TAG] += 1
			total_count += 1

		probs[B_CUE_TAG] = float( probs[B_CUE_TAG] ) / total_count
		probs[I_CUE_TAG] = 0.0
		probs[O_TAG] = float( probs[O_TAG] ) / total_count

		return probs


	""" Returns: The bigram probabilities table generated from the explicit 
		(or implicit without smoothing) count table. 

		Precondition:	bigrams = explicit (or implicit without smoothing) table 
								  containing all possible bigram counts """
 	def compute_prob_table(self, bigrams):

 		for t1 in bigrams:

 			total = 0

 			for t2 in bigrams[t1]:
 				total += bigrams[t1][t2]

 			for t2 in bigrams[t1]:
 				bigrams[t1][t2] = float(bigrams[t1][t2]) / total

 		return bigrams


	""" Returns: Explicit table containing all the bigrams transition probabilities 
		from one state (one of B_CUE_TAG, I_CUE_TAG, or O_TAG) to another. """
	def build_transition_table(self):

		# initialize the count dictionary
		count = self.dfs_dict_setup(2)

		for i in xrange( len(self.triplet_list) - 1 ):

			(token1, pos_tag1, bio_tag1) = self.triplet_list[i]
			(token2, pos_tag2, bio_tag2) = self.triplet_list[i+1]
			count[bio_tag1][bio_tag2] += 1

		if (self.smooth_trans):
			prob_dict = self.smooth_laplacian_trans(count) # Add-1
		else:
			prob_dict = self.compute_prob_table(count) # MLE

		return prob_dict


	""" Returns: Explicit table containing all possible n-gram counts of states
		(B_CUE_TAG, I_CUE_TAG, or O_TAG). 

		Precondition:	n = the degree of n-grams """
	def dfs_dict_setup(self, n):

		if n == 0:
			return 0

		else:
			next_level = {}

			for i in self.states:
				next_level[i] = self.dfs_dict_setup(n - 1)

		return next_level


	""" Returns: The laplacian (add-1) smoothed bigram transition probabilities 
		table. Note that since P(I | O) must be equal to 0, we do not smooth
		this count. 

		Precondition:	bigrams = explicit table containing all possible bigram
							 	  counts of states """
	def smooth_laplacian_trans(self, bigrams):

 		for tag1 in bigrams:

 			for tag2 in bigrams[tag1]:

 				if (tag1 != O_TAG or tag2 != I_CUE_TAG):
 					bigrams[tag1][tag2] += 1

 		return self.compute_prob_table(bigrams)


	""" Returns: Implicit table containing the "bigrams" emission probabilities 
		of seeing a token given a tag. """
 	def build_emission_table(self):

 		count = {}

 		for (token, pos_tag, bio_tag) in self.triplet_list:

 			if bio_tag not in count:
 				count[bio_tag] = {token : 1}
 			
 			else:
 				if token not in count[bio_tag]:
 					count[bio_tag][token] = 1
 				else:
 					count[bio_tag][token] += 1

 		if (self.smooth_emiss):
			prob_dict = self.smooth_laplacian_emiss(count) # Add-1
		else:
			prob_dict = self.compute_prob_table(count) # MLE

		return prob_dict


	""" Returns: The laplacian (add-1) smoothed bigram emission probabilities 
		table.

		Precondition:	bigrams = implict table containing the "bigram" counts 
								  of token given tag """
	def smooth_laplacian_emiss(self, bigrams):

 		for tag in bigrams:

 			total = 0

 			for token in bigrams[tag]:
 				total += bigrams[tag][token]

 			for token in bigrams[tag]:
 				bigrams[tag][token] = float( bigrams[tag][token] + 1 ) / ( total + len(self.vocab) )

 		return bigrams


	################### External Functions for Utilization ####################


	""" Returns: the conditional probability of a emission "bigram" unseen in
		training, given the tag.
		
		Precondition:	tag = one of B_CUE_TAG, I_CUE_TAG, or O_TAG """
	def get_unseen_emiss_prob(self, tag):

		seen_prob = 0.0
		num_unseen = 0

		for token in self.emission_table[tag]:
			seen_prob += self.emission_table[tag][token]

		# number of unseen bigrams starting with this tag
		num_unseen = len(self.vocab) - len(self.emission_table[tag])
		unseen_prob = float(1 - seen_prob) / num_unseen

		return unseen_prob


	""" Returns: The log of start probability of the input BIO tag in this HMM. 

		Precondition:	tag =  one of B_CUE_TAG, I_CUE_TAG, or O_TAG """
	def get_log_start_prob(self, tag):

		if (self.start_table[tag] == 0.0):
			return math.log(1e-8) # log of 0 is undefined, so choose very small number

		return math.log( self.start_table[tag] )


	""" Returns: The log of transition probability P(tag2 | tag1) in this HMM.

		Precondition:	tag1 =  one of B_CUE_TAG, I_CUE_TAG, or O_TAG
						tag2 =  one of B_CUE_TAG, I_CUE_TAG, or O_TAG """
	def get_log_trans_prob(self, tag1, tag2):

		if (self.transition_table[tag1][tag2] == 0.0):
			return math.log(1e-8) # log of 0 is undefined, so choose very small number

		return math.log( self.transition_table[tag1][tag2] )


	""" Returns: The log of emission probability P(token | tag) in this HMM.

		Precondition:	tag =  one of B_CUE_TAG, I_CUE_TAG, or O_TAG
						token =  a string (word) """
	def get_log_emiss_prob(self, tag, token):

		if token not in self.vocab: # word not encountered in training

			if (UNKNOWN_TAG not in self.emission_table[tag]): # unknown is unseen for this tag

				if (self.smooth_emiss): # if we had smoothed, unseen bigram has some weight
					return math.log( self.get_unseen_emiss_prob(tag) )
				
				else: # otherwise, probability of unseen bigram is just zero
					return math.log(1e-8) # log of 0 is undefined, so choose very small number
			
			else:
				return math.log( self.emission_table[tag][UNKNOWN_TAG] )

		else: # word encountered in training
			
			if token not in self.emission_table[tag]: # unseen bigram
				
				if (self.smooth_emiss): # if we had smoothed, unseen bigram has some weight
					return math.log( self.get_unseen_emiss_prob(tag) )
				
				else: # otherwise, probability of unseen bigram is just zero
					return math.log(1e-8) # log of 0 is undefined, so choose very small number			
			
			else:
				return math.log( self.emission_table[tag][token] )


 	""" Returns: the list representing the most likely state-sequence corresponding
 		to the input list representing the observation-sequence. We use the
 		Viterbi Algorithm to predict the state-sequence. Note that we use the 
 		addition of log probabilities in our calculations to optimize runtime 
 		and avoid underflow.
		
		Precondition:	obs_list = non-empty list of strings (words) representing observations """
 	def viterbi_decode(self, obs_list):
 		
 		# Initialize DP table
 		V = [{}]

 		# Base case
 		for state in self.states:
 			V[0][state] = {"prob": self.get_log_start_prob(state) + self.get_log_emiss_prob(state, obs_list[0]), "prev": None}

		# Iteratively build up DP table
		for i in xrange(1, len(obs_list)):

			V.append({})

			for state in self.states:
				max_trans_prob = max( ( V[i-1][prev_state]["prob"] + self.get_log_trans_prob(prev_state, state) ) for prev_state in self.states )
				
				for prev_state in self.states:

					if V[i-1][prev_state]["prob"] + self.get_log_trans_prob(prev_state, state) == max_trans_prob:
						max_prob = max_trans_prob + self.get_log_emiss_prob(state, obs_list[i])
						V[i][state] = {"prob": max_prob, "prev": prev_state}
						break

		# Initialize optimal (most-likely) list of states
		opt_sequence = []

		# Determine maximum probability value at the end of the table
		max_prob = max( value["prob"] for value in V[-1].values() )

		# Determine the most likely last state
		previous = None
		for state, info in V[-1].items():
			if info["prob"] == max_prob:
				opt_sequence.append(state) # this is the last state in the sequence
				previous = state
				break

		# Backtrack and prepend to the optimal list of states until the first observation
		for i in xrange(len(V) - 2, -1, -1):
			opt_sequence.insert(0, V[i + 1][previous]["prev"])
			previous = V[i + 1][previous]["prev"]

		return opt_sequence


# end of class definition


# // <------------------------------ Write to file ------------------------------> //


""" Procedure: Uses our Hidden Markov Model to write in uncertainty tags for 
	test docs in in_directory, and writes the tagged version into out_directory
	WITHOUT any changes in original file name. 

	Precondition:	hmm_model = our trained HMM instance
					in_directory = valid directory path containing test files 
					out_directory = valid directory path (could be new) where
									we will write the completed test files """
def write_completed_testfiles(hmm_model, in_directory, out_directory):

	for file_name in os.listdir(in_directory):

		if file_name.endswith('.txt'):
			read_path = os.path.join(in_directory, file_name)
			if not os.path.exists(out_directory):
				os.makedirs(out_directory)
			write_path = os.path.join(out_directory, file_name)

			with open(read_path, 'r') as read_file:
				with open(write_path, 'w') as write_file:
					write_new_tags(hmm_model, read_file, write_file)


""" Procedure: Helper function for writing tags. This writes in all the tags
	for each line of a test doc.

	Precondition:	hmm_model = our trained HMM instance
					read_file = valid read file path
					write_file = valid write file path (could be new) """
def write_new_tags(hmm_model, read_file, write_file):

	obs_list = []
	read_file_list = list(read_file)

	# First, get the observation list (words list)
	for line in read_file_list:

		line = line.strip()
		line_split = line.split()

		if len(line_split) > 0:
			obs_list.append( line_split[0].lower() )

	# Then, use our HMM model to get the corresponding tags list
	tags_list = hmm_model.viterbi_decode(obs_list)

	# Finally, write the predicted tags onto each test file one by one
	i = 0
	for line in read_file_list:

		line = line.strip()
		line_split = line.split()

		if len(line_split) > 0:
			write_file.write( line + '                 ' + tags_list[i] + '\n' )
			i += 1

		else:
			write_file.write('\n')


# // <------------------------------ Main Function ------------------------------> //


""" Main Function: This method will train the Hidden Markov model on all train 
	docs in the specified directory, then execute write_completed_testfiles().

	Precondition: 	sys.argv[1] = path to directory containing training docs
					sys.argv[2] = path to directory containing test files
					sys.argv[3] = path to directory where completed test files will be written
	"""
def main():
	
	directory = sys.argv[1]
	train_set = set( [ x for x in xrange( len(os.listdir(directory)) ) ] )
	hmm_model = HMM(directory, train_set)
	write_completed_testfiles(hmm_model, sys.argv[2], sys.argv[3])

if __name__ == '__main__':
	main()


