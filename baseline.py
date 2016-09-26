import sys
import os
import re
from preprocess import *
from constants import *


# // <------------------------------ Baseline Definition ------------------------------> //

class Baseline(object):

	""" Initializer: Initializes an instance of baseline using training docs in the 
		specified directory.

		Precondition: 	directory = path to directory containing training docs.
						train_set = set of ints (smallest being 0) indicating 
									which files we choose to use in training our model 
						resample = true or false indicating whether we upsample
							   	   sentences with uncertainty and downsample sentences
							   	   without to balance the dataset """
	def __init__(self, directory, train_set, resample=False):
		self.directory = directory
		self.train_set = train_set
		self.resample = resample
		self.baseline_dict = self.build_baseline()


	""" Returns: The baseline dictionary """
	def getBaseline(self):
		return self.baseline_dict


	""" Returns: 2-layer dictionary that covers tokens across all texts. 
		Top level indexed by token, low level keeps counter for tag 
		occurrences. """
	def build_tag_counter(self):

		tag_counter = {}

		# sort files in increasing alphanumeric order
		sorted_files = sorted( os.listdir(self.directory), key=lambda x: ( int( re.sub('\D', '', x) ), x) )

		for i in xrange( len(sorted_files) ):

			file_name = sorted_files[i]

			if file_name.endswith('.txt') and i in self.train_set:
				file_path = os.path.join(self.directory, file_name)
				tags_list = preprocess(file_path, self.resample)

				for (token, pos, bio) in tags_list:
					if token not in tag_counter:
						tag_counter[token] = {bio : 1}
					else:
						if bio not in tag_counter[token]:
							tag_counter[token][bio] = 1
						else:
							tag_counter[token][bio] += 1

		return tag_counter


	""" Returns: The baseline dictionary using global_tag_counter. baseline
		will map each token to its most frequently associated tag. """
	def build_baseline(self):
		
		tag_counter = self.build_tag_counter()
		baseline = {}

		for token in tag_counter:

			tags = tag_counter[token]
			tags_sorted = sorted(tags, key=lambda key: tags[key], reverse=True)
			most_freq_tag = tags_sorted[0]
			baseline[token] = most_freq_tag

		return baseline


# end of class definition


# // <------------------------------ Write to file ------------------------------> //

""" Procedure: Uses our baseline model to write in uncertainty tags for 
	test docs in in_directory, and writes the tagged version into out_directory
	WITHOUT any changes in original file name. 

	Precondition:	baseline = our baseline dictionary 
					in_directory = valid directory path containing test files 
					out_directory = valid directory path (could be new) where
									we will write the completed test files """
def write_completed_testfiles(baseline, in_directory, out_directory):

	for file_name in os.listdir(in_directory):

		if file_name.endswith('.txt'):
			read_path = os.path.join(in_directory, file_name)
			if not os.path.exists(out_directory):
				os.makedirs(out_directory)
			write_path = os.path.join(out_directory, file_name)

			with open(read_path, 'r') as read_file:
				with open(write_path, 'w') as write_file:
					write_new_tags(baseline, read_file, write_file)


""" Procedure: Helper function for writing tags. This writes in all the tags
	for each line of a test doc. 
	Precondition:	baseline = our baseline dictionary 
					read_file = valid read file path
					write_file = valid write file path (could be new) """
def write_new_tags(baseline, read_file, write_file):

	was_last_cue = False
	for line in read_file:

		line = line.strip()
		line_split = line.split()

		if len(line_split) > 0:

			if line_split[0].lower() in baseline:
				most_freq_tag = baseline[ line_split[0].lower() ]

				if most_freq_tag == O_TAG:
					write_file.write( line + '                   ' + O_TAG + '\n' )
					was_last_cue = False
				elif was_last_cue:
					write_file.write( line + '                   ' + I_CUE_TAG + '\n' )
				else:
					write_file.write( line + '                   ' + B_CUE_TAG + '\n' )
					was_last_cue = True
			else:
				write_file.write(line + '                   ' + O_TAG + '\n') # token unseen in training
				was_last_cue = False
		else:
			write_file.write('\n')
			was_last_cue = False


# // <------------------------------ Main Function ------------------------------> //


""" Main Function: This method will first create baseline, then 
	execute write_completed_testfiles()

	Precondition: 	sys.argv[1] = path to directory containing training docs
					sys.argv[2] = path to directory containing test files
					sys.argv[3] = path to directory where completed test files will be written
	"""
def main():
	directory = sys.argv[1]
	train_set = train_set = set( [ x for x in xrange( len(os.listdir(directory)) ) ] )
	base = Baseline(directory, train_set)
	write_completed_testfiles(base.getBaseline(), sys.argv[2], sys.argv[3])

if __name__ == '__main__':
	main()


