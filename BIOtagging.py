from nltk.tokenize import word_tokenize
import itertools
import re
import pandas as pd
import numpy as np
import string
from tqdm import tqdm

def PUUtext_to_tagReadyDF(input, isCSV=True, more_stopwords=None):
	"""
	This function is used to convert raw text of PUU (either CSV file or pandas Series) into tag-ready dataframe.

	Args:
	- input (pd.Series variable): either CSV file (enter its file location) or pandas Series. If you want to use pandas Series, set 'isCSV' arg to False.
	- isCSV (Boolean): if True, CSV input used. If False, pd.Series input used.
	- more_stopwords (list): add more stopwords if you'd like.

	Return:
	- result dataframe
	"""
    # in case Sastrawi not detected
	try:
		from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
		stopwords = StopWordRemoverFactory().get_stop_words()
	except ModuleNotFoundError:
		print("No module named 'Sastrawi' in your machine. Bypassing 'Sastrawi' dependency, but the number of stopwords will decrease.")
		stopwords = []

	# check input
	if isCSV:
		# add csv file that contains raw PUU texts
		text = pd.read_csv(input, delimiter='|')
		if text.iloc[:,0].dtype == 'O':
			text = text.iloc[:,0]
		else:
			raise ValueError("As 'isCSV' set to True, the 1st column of your CSV file should be the texts you'd like to process.")
			
	else:
		# if pd.Series expected
		if isinstance(input, pd.Series):		# check if data type is suitable
			text = input
		else:
			raise TypeError("As 'isCSV' set to False, 'input' should be a pandas Series.")

	# define punctuation
	punctAndSpace = string.punctuation + ' '
	# kita memerlukan karakter '(', ')', dan '.',
	# karena karakter tsb muncul di ayat dan angka
	punctAndSpace = punctAndSpace.replace('(','')
	punctAndSpace = punctAndSpace.replace(')','')
	punctAndSpace = punctAndSpace.replace('.','')

	# tambah stopwords dari argument variable
	if more_stopwords != None:	
		assert isinstance(more_stopwords, list), "'more_stopwords' arg should be list type."
		stopwords += more_stopwords

	stopwords = sorted(set(stopwords))

	# ubah Raw teks PUU menjadi tokens ke sebuah kolom df
	# lalu beri tagging 'O' secara otomatis pada tokens yang tidak masuk interest annotations
	dfList = []
	for idx, t in tqdm(enumerate(text)):
	    # tokenization
	    tokens = [[word_tokenize(w), ' '] for w in t.split()]
	    tokens = list(itertools.chain(*list(itertools.chain(*tokens))))
	    tokens = tokens[:-1]
	    
	    split_res = []
	    for t in tokens:
	        # if-else di bawah ini untuk mengcover token berbentuk seperti ini,
	        # 'Jakarta-Bogor-Ciawi'
	        if re.match(r'\w+\-\w+.*', t):
	            line = t.split('-')
	            for i,j in enumerate(line):
	                split_res.append(j)
	                if i < len(line)-1:
	                    split_res.append('-')
	        else:
	            split_res.append(t)
	    
	    # membuat tagging 'O' untuk token yang kita anggap tidak masuk list annotations
	    blank = ['' if i.lower() not in list(punctAndSpace) + stopwords else 'O' for i in split_res]
	    
	    # buat menjadi df
	    dfTemp = pd.DataFrame([split_res, blank]).T
	    # beri nama kolom sesuai dengan index looping
	    dfTemp.columns = ['token_' + str(idx),'BIO_tag_' + str(idx)]
	    dfList.append(dfTemp)
	# concat semua df
	df = pd.concat(dfList, axis=1)

	# # save ke file csv yang siap ditag manual
	# df.to_csv(output_loc)
	# print('CSV output file successfully written.')

	return df


def convert_to_spaCyformat(df, listOfEntities):
    """
    This function is used to convert the BIO-tagged-DF to spaCy format annotations.
    
    Args:
    - df (pandas.DataFrame) > BIO-tagged dataframe consisting of two columns, i.e. token and BIO_tag
    - listOfEntities (list) > list of entities/annotations used
    
    Return:
    - [text, enti] > a list consisting of the text (combined from the tokens) and the interested entities as accordance with spaCy format
    """
    # check if NaN exists
    assert not (df.iloc[:,0].isnull().any() or df.iloc[:,1].isnull().any()), 'The dataset contains nan value.'
    
    # create a dictionary to save the columns of 'token' and 'BIO_tag', and we also define the index of tokens in order
    dictTemp = {}
    dictTemp['token'] = np.array(df.iloc[:,0])
    dictTemp['BIO_tag'] = np.array(df.iloc[:,1].str.lower())
    dictTemp['indices'] = np.array([len(i) for i in dictTemp['token']])

    # first, we need to get the index of the first token
    total_idx = [dictTemp['indices'][0]] 
    temp = dictTemp['indices'][0]
    
    # then we use for loop to count index for each token in cumulative
    for i in range(len(dictTemp['indices'])):
        if i > 0:
            temp += dictTemp['indices'][i]
            total_idx.append(temp)

    # create variable for the start index of each token
    dictTemp['start_idx'] = np.array([total_idx[i-1] if i > 0 else 0 for i in range(len(total_idx))])

    # create variable for the last index of each token
    dictTemp['end_idx'] = np.array(total_idx)
    del dictTemp['indices'] # we no longer need variable indices. then remove it.

    enti = {}
    entities = []
    text = ''.join(dictTemp['token'])
    
    # combine each of listOfEntities with prefix 'b-', 'i-', and 'e-', and add 'o' annotation
    listOfEntities = ['b-'+i.lower() for i in listOfEntities] + \
                     ['i-'+i.lower() for i in listOfEntities] + \
                     ['e-'+i.lower() for i in listOfEntities] + ['o']
    
    # check if each BIO-tag is in listOfEntities
    error_tag = []
    error_boolean = []
    for i in np.unique(dictTemp['BIO_tag']):
        if i in listOfEntities:
            error_boolean.append(True)
        else:
            error_boolean.append(False)
            error_tag.append(i)
    assert all(error_boolean), "Some BIO-tag not listed in listOfEntities arg. {}".format(error_tag)
    
    # fill in entities list with all non 'O' annotations
    for row in range(len(dictTemp['token'])):
        if dictTemp['BIO_tag'][row] != 'o':
            entities.append((dictTemp['start_idx'][row], 
                             dictTemp['end_idx'][row], 
                             dictTemp['BIO_tag'][row]))

    
    start = []
    end = []
    BIO = []
    i = 0
    while i < len(entities):
        try:
            if entities[i][2][2:] == entities[i+1][2][2:]:
                if entities[i][2][0] is 'b':
#                     print('start1', entities[i][0])
                    start.append(entities[i][0])
                    i += 1
                    if entities[i][2][0] is 'e':
#                         print('end1a', entities[i][1])
                        end.append(entities[i][1])
                        BIO.append(entities[i][2][2:])
                        i += 1
                        continue
                    elif entities[i][2][0] is 'i':
                        for j in range(i, len(entities)):
                            if entities[j][2][0] is not 'e' and j < len(entities)-1:
#                                 print('sana', entities[j])
                                continue
                            elif entities[j][2][0] is 'e':
#                                 print('end1b', entities[j][1])
                                end.append(entities[j][1])
                                BIO.append(entities[j][2][2:])
                                i = j+1
                                break
                            else:
                                assert 1 == 0, \
                                    "Something error in the BIO-tag you wrote. Error BIO tag: '{}'" \
                                    .format(entities[j][2])
                    elif entities[i][2][0] is 'b':
#                         print('end1b', entities[i-1][1])
                        end.append(entities[i-1][1])
                        BIO.append(entities[i-1][2][2:])
                        continue
                        
#                         print('ss',i,j)
            else:
#                 print('start2a', entities[i][0], i)
                start.append(entities[i][0])
#                 print('end2a', entities[i][1], i)
                end.append(entities[i][1])
                BIO.append(entities[i][2][2:])
                i += 1
        except IndexError:
#             print('start2b', entities[i][0], i)
            start.append(entities[i][0])
#             print('end2b', entities[i][1], i)
            end.append(entities[i][1])
            BIO.append(entities[i][2][2:])
            i += 1

    enti['entities'] = [(i,j,k) for i,j,k in zip(start, end, BIO)]
    return [text, enti]