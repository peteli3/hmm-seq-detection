import sys
import os
import csv
import re
from constants import *


""" Returns: the string representation of predicted spans for Kaggle submission
	using our completed test files (from our baseline system). 

	Precondition:	directory = valid path to directory containing completed test docs. """
def uncertain_phrase_detection(directory):

	sorted_files = sorted(os.listdir(directory), key=lambda x: (int(re.sub('\D', '', x)), x))

	index = 0
	most_recent_b = -1
	range_dict = {}

	for file_name in sorted_files:
		if file_name.endswith('.txt'):
			file_path = os.path.join(directory, file_name)

			with open(file_path, 'r') as f:
				for line in f:
					line_split = line.strip().split()

					if len(line_split) > 0:
						tag = line_split[2]

						if tag == B_CUE_TAG:
							range_dict[index] = index
							most_recent_b = index
						elif tag == I_CUE_TAG:
							range_dict[most_recent_b] += 1
						index += 1

	sorted_keys = range_dict.keys()
	sorted_keys.sort()
	acc_string = ''
	for start_index in sorted_keys:
		to_add = str(start_index) + '-' + str( range_dict[start_index] ) + ' '
		acc_string += to_add

	return acc_string


""" Returns: the string representation of sentence ids for Kaggle submission
	using our completed test files (from our baseline system). 

	Precondition:	directory = valid path to directory containing completed test docs."""
def uncertain_sentence_detection(directory):
	
	sorted_files = sorted(os.listdir(directory), key=lambda x: (int(re.sub('\D', '', x)), x))

	index = 0
	previous_was_token = False
	sentence_id_list = []

	for file_name in sorted_files:
		if file_name.endswith('.txt'):
			file_path = os.path.join(directory, file_name)

			with open(file_path, 'r') as f:
				for line in f:
					line_split = line.strip().split()

					if len(line_split) > 0:
						if line_split[2] == B_CUE_TAG or line_split[2] == I_CUE_TAG:
							if index not in sentence_id_list:
								sentence_id_list.append(index)
						previous_was_token = True
					elif previous_was_token:
						index += 1
						previous_was_token = False

	sentence_id_list.sort()
	acc_string = ''
	for i in sentence_id_list:
		to_add = str(i) + ' '
		acc_string += to_add

	return acc_string


""" Main Function: This method will execute write_completed_testfiles() and
	and generate our Kaggle submission files in csv format.

	Precondition:	sys.argv[1] = path to directory containing completed private test files
					sys.argv[2] = path to directory containing completed public test files
	 """
def main():

	with open('kaggle_submission_phrases.csv', 'w') as csvfile:
		wr = csv.writer(csvfile)
		wr.writerow( ['Type', 'Spans'] )
		wr.writerow( ['CUE-public', uncertain_phrase_detection(sys.argv[2])] ) # PUBLIC FIRST
		wr.writerow( ['CUE-private', uncertain_phrase_detection(sys.argv[1])] ) # PRIVATE

	with open('kaggle_submission_sentences.csv', 'w') as csvfile:
		wr = csv.writer(csvfile)
		wr.writerow( ['Type', 'Indices'] )
		wr.writerow( ['SENTENCE-public', uncertain_sentence_detection(sys.argv[2])] ) # PUBLIC FIRST
		wr.writerow( ['SENTENCE-private', uncertain_sentence_detection(sys.argv[1])] ) # PRIVATE


if __name__ == '__main__':
	main()


