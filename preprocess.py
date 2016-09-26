from constants import *


# // <------------------------------ Preprocessing ------------------------------> //


""" Returns: The input text as a list of 3-tuples (token, POS tag, BIO tag). For 
	simplicity and consistency, we always lowercase the token (word) in our data.
	If we choose to resample the sentences in the input text (we may only use
	this option for training purposes), we repeat (once) the sentences with 
	uncertainty cues and ignore every other sentence without uncertainty cues.

	Precondition:	file_path = string representing valid path to a text file
					resample = true or false indicating whether we upsample
							   sentences with uncertainty and downsample sentences
							   without to balance individaul text data """
def preprocess(file_path, resample=False):

	if not resample:

		triplet_list = []

		with open(file_path, 'r') as f:

			seen_cues = set()

			for line in f:
				line_split = line.strip().split()

				if len(line_split) > 0:

					bio_tag = O_TAG

					if line_split[2] == '_':
						bio_tag = O_TAG

					elif 'CUE-' in line_split[2]:

						if int( line_split[2][4:] ) not in seen_cues:
							seen_cues.add( int(line_split[2][4:]) )
							bio_tag = B_CUE_TAG
						else:
							bio_tag = I_CUE_TAG

					word_with_tags = (line_split[0].lower(), line_split[1], bio_tag)
					triplet_list.append(word_with_tags)

				else:
					# empty line -> reset at the end of every sentence
					seen_cues = set() 

		return triplet_list

	else:

		sentence_list = [] # list of lists that will be flattened later

		with open(file_path, 'r') as f:

			triplet_list = []
			seen_cues = set()
			sentence_has_cue = False
			counter = 0

			for line in f:
				line_split = line.strip().split()

				if len(line_split) > 0:

					bio_tag = O_TAG

					if line_split[2] == '_':
						bio_tag = O_TAG

					elif 'CUE-' in line_split[2]:

						sentence_has_cue = True

						if int( line_split[2][4:] ) not in seen_cues:
							seen_cues.add( int(line_split[2][4:]) )
							bio_tag = B_CUE_TAG
						else:
							bio_tag = I_CUE_TAG

					word_with_tags = (line_split[0].lower(), line_split[1], bio_tag)
					triplet_list.append(word_with_tags)

				else: # empty line (end of sentence)

					if (sentence_has_cue): # double count the uncertain sentences
						sentence_list.append(triplet_list)
						sentence_list.append(triplet_list)

					else:

						if (counter % 2 == 1): # only add every other certain sentences
							sentence_list.append(triplet_list)
							counter += 1

					# reset at the end of every sentence
					triplet_list = []
					seen_cues = set()
					sentence_has_cue = False

		return [item for sublist in sentence_list for item in sublist]


