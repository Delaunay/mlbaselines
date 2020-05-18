import numpy as np
import pandas as pd
import urllib.request
import zipfile
import os
import pdb

def is_protein(aminoacids, sequence):
    for j in sequence:
        if not j in aminoacids:
            return False
    return True


def encode_to_sparse(sequence,aminoacids, max_length):
    seq = np.array([0])
    while len(sequence)<max_length:
        sequence+='X'
    for i in range(len(sequence)):
        aa = np.ones(20)*0.05
        if not sequence[i]=='X':
            aa[aminoacids.index(sequence[i])] = 0.9
        seq = np.hstack((seq,aa))
    return seq[1:]

def get_alleles_pMHC(folder = "NetMHC"):


	return alleles

def load_pMHC_dataset(folder = "NetMHC", alleles_only = False):
	"""
	This function prepares the dataset for the second task: predicting pan-allele peptide
	binding specificities.
	This function performs the encoding of the MHC and peptide sequences to sparse format
	"""

	### we use the following list for amino acid letter codes
	aminoacids = ["G","P","A","V","L","I","M","C","F","Y","W","H","K","R","Q","N","E","D","S","T"]

	###TODO: find a way to download from Mendeley and unzip here (see below)
    #urllib.request.urlretrieve('https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/8pz43nvvxh-3.zip',f'{folder}/temp.zip')
	###TODO: find a way to unzip the file from python
	###TODO: perform the check for the dataset and download only if needed


	### loading the dataset from the specified file and renaming the columns
	print ('Loading files...')
	data = pd.read_csv(f'{folder}/curated_training_data.no_mass_spec.csv')

	### We will only keep the human alleles
	data = data[['HLA' in i for i in data['allele']]]
	### Transforming the allele annotation so it matches the reference:
	data['allele'] = [''.join(i.split('*')) for i in data['allele']]

	### loading the allele data from specified folder
	alleles = pd.read_csv(f'{folder}/MHC_pseudo.dat',header=None,sep=' ')
	alleles = alleles[['HLA' in i for i in alleles[0]]]
	alleles.columns = ['HLA_allele', 'allele_seq']

	### The overlap is only the alleles for which we have a sequence!
	overlapping_alleles = set(data['allele'])&set(alleles['HLA_allele'])
	data = data[[i in overlapping_alleles for i in data['allele']]]

	if alleles_only:
		return alleles
	else:
		return data, alleles, overlapping_alleles


def get_panallele_dataset(folder = 'NetMHCpan_data')
	"""
	This function prepares the dataset for the second task: predicting pan-allele peptide
	binding specificities.
	This function performs the encoding of the MHC and peptide sequences to sparse format
	"""


	### getting the dataset
	data, alleles, overlapping_alleles = load_pMHC_dataset(folder)

	max_length_allele = int(np.max(alleles['allele_seq'].str.len()))
	allele_seq = []
	print ('Encoding alleles to sparse...')
	### encoding the hla alleles into sparse format, as described in original paper
	for i in list(alleles['allele_seq']):
        allele_seq.append(encode_to_sparse(i, aminoacids,max_length_allele).reshape(len(aminoacids)*max_length_allele+1,))
	alleles['allele_seq_sparse'] = allele_seq

	### merging the allele sequences with the data
	data = data.merge(alleles, left_on='allele', right_on = 'HLA_allele')


	### getting the sparse encodings for the peptides 
	### TODO: optimize this part of the code
	print ('Encoding peptides to sparse...')
	peptide_sparse = []
	where = 0
	max_length_peptides = int(np.max(train_data['peptide'].str.len()))
	for i in train_data['peptide']:
        if where%1000==0:
            print (where)
        peptide_sparse.append(encode_to_sparse(i, aminoacids).reshape(len(aminoacids)*,))
    train_data['peptide_sparse'] = peptide_sparse


    ### mergin these encodings into the dataset
    where= 0
    print ('Stacking train set...')
	input_data = np.zeros((data.shape[0], (max_length_allele*len(aminoacids)+max_length_peptides*len(aminoacids))))
	for i in range(train_data.shape[0]):
        if where%10000==0:
            print (where)
        temp = np.hstack((np.array(data['peptide_sparse'][i]),np.array(data['allele_seq_sparse'][i])))
        input_data[where,:] = temp
        where+=1

	print ('Done!')
	return data, np.array(data['']), input_test, np.array(test_data['label'])




def get_singleallele_dataset(allele='HLA-A02:01', folder='NetMHC'):
	"""
	This function prepares the dataset for the first task: predicting peptide
	binding specificity for a specific chosen allele.
	This function performs the encoding of the peptide sequences to sparse format
	"""

	### getting the dataset

	data, alleles, overlapping_alleles = load_pMHC_dataset(folder)

	### we use the following list for amino acid letter codes
	aminoacids = ["G","P","A","V","L","I","M","C","F","Y","W","H","K","R","Q","N","E","D","S","T"]

	data = data[data['allele'] == allele]

	print ('Encoding peptides to sparse...')
	inputs = list(data['peptide'])
	encoded_inputs = np.zeros((len(inputs)+1,max_length_peptides*len(aminoacids)))

	where = 0
	for i in inputs:
        if where%1000==0:
            print (where)
        encoded_inputs[where,:] = encode_to_sparse(i, aminoacids).reshape(220,)
        where+=1
	encoded_inputs = encoded_inputs[:-1,:]

	print ('Done!')
	return np.vstack((encoded_inputs, labels))


# NOTE: No builders for these datasets. We load them as matrices for sklearn like models.
