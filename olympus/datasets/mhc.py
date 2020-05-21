import numpy as np
import pandas as pd
import urllib.request
import zipfile
import os
import pdb
import pickle

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

def get_sparse_alleles(alleles, aminoacids, max_length_allele):
	### encoding the hla alleles into sparse format, as described in original paper
	allele_seq = []
	for i in list(alleles['allele_seq']):
		allele_seq.append(encode_to_sparse(i, aminoacids,max_length_allele).reshape(len(aminoacids)*max_length_allele,))

	return allele_seq

def get_sparse_peptides(peplist, aminoacids, max_length_peptides):
	### encoding the hla alleles into sparse format, as described in original paper
	peptide_sparse = []
	total_pep = len(peplist)
	where = 0
	for i in peplist:
		if where%10000==0:
			print (f'processed {where/total_pep*100:2f}% peptides')
		peptide_sparse.append(encode_to_sparse(i, aminoacids, max_length_peptides).reshape(len(aminoacids)*max_length_peptides,))
		where+=1

	return peptide_sparse

def merge_dataset(data, peptide_columns, allele_columns, aminoacids, max_length_allele, max_length_peptides):
	where= 0
	print ('Stacking train set...')
	total_pep = len(data.shape)
	input_data = np.zeros((data.shape[0], (max_length_allele*len(aminoacids)+max_length_peptides*len(aminoacids))))
	for i in range(data.shape[0]):
		if where%100000==0:
			print (f'stacked {where/total_pep*100}% peptides')
		temp = np.hstack((np.array(data[peptide_columns][i]),np.array(data[allele_columns][i])))
		input_data[where,:] = temp
		where+=1
	return input_data


def load_pMHC_dataset(data_path, folder="NetMHCpan_data", alleles_only = False):
	"""
	This function prepares the dataset for the second task: predicting pan-allele peptide
	binding specificities.
	This function performs the encoding of the MHC and peptide sequences to sparse format
	"""

	
	###TODO: find a way to download from Mendeley and unzip here (see below)
	#urllib.request.urlretrieve('https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/8pz43nvvxh-3.zip',f'{folder}/temp.zip')
	###TODO: find a way to unzip the file from python
	###TODO: perform the check for the dataset and download only if needed

	### loading the dataset from the specified file and renaming the columns
	print ('Loading files...')
	data = pd.read_csv(f'{data_path}/{folder}/curated_training_data.no_mass_spec.csv')

	### We will only keep the human alleles
	data = data[['HLA' in i for i in data['allele']]]
	### We will only keep peptides of sizes 8-11
	data = data[[len(i)<12 and len(i)>7 for i in data['peptide']]]
	### Transforming the allele annotation so it matches the reference:
	data['allele'] = [''.join(i.split('*')) for i in data['allele']]

	### loading the allele data from specified folder
	alleles = pd.read_csv(f'{data_path}/{folder}/MHC_pseudo.dat',header=None,sep=' ')
	alleles = alleles[['HLA' in i for i in alleles[0]]]
	alleles.columns = ['HLA_allele', 'allele_seq']

	### The overlap is only the alleles for which we have a sequence!
	overlapping_alleles = set(data['allele'])&set(alleles['HLA_allele'])
	data = data[[i in overlapping_alleles for i in data['allele']]]

	if alleles_only:
		return alleles
	else:
		return data, alleles, overlapping_alleles


def get_valid_dataset(folder = "NetMHC"):
	aminoacids = ["G","P","A","V","L","I","M","C","F","Y","W","H","K","R","Q","N","E","D","S","T"]
	print ('Loading files...')
	data = pd.read_csv(f'{folder}/Pearson_dataset.tsv',sep='\t')
	data = data[[len(i)<12 and len(i)>7 for i in data['Peptide Sequence']]]

	### loading the allele data from specified folder
	alleles = pd.read_csv(f'{folder}/MHC_pseudo.dat',header=None,sep=' ')
	alleles = alleles[['HLA' in i for i in alleles[0]]]
	alleles.columns = ['HLA_allele', 'allele_seq']


	### The overlap is only the alleles for which we have a sequence!
	overlapping_alleles = set(data['Allele'])&set(alleles['HLA_allele'])
	data = data[[i in overlapping_alleles for i in data['Allele']]]

	max_length_allele = int(np.max(alleles['allele_seq'].str.len()))

	print ('Encoding alleles to sparse...')
	allele_seq = get_sparse_alleles(alleles, aminoacids, max_length_allele)
	alleles['allele_seq_sparse'] = allele_seq

	### merging the allele sequences with the data
	data = data.merge(alleles, left_on='Allele', right_on = 'HLA_allele')

	max_length_peptides = int(np.max(data['Peptide Sequence'].str.len()))
	### getting the sparse encodings for the peptides 
	print ('Encoding peptides to sparse...')
	peptide_sparse = get_sparse_peptides(data['Peptide Sequence'], aminoacids, max_length_peptides)
	data['peptide_sparse'] = peptide_sparse

	### merging these encodings into the dataset
	
	input_data = merge_dataset(data, 'peptide_sparse', 'allele_seq_sparse', aminoacids, max_length_allele, max_length_peptides)

	### processing the targets
	targets = data['Binding Affinity']

	### values are capped at 50k according to MHCflurry 
	targets = np.array([min(50000,i) for i in targets ])
	targets = np.array([max(1,i) for i in targets ])
	### transforming according to MHCflurry formula to range 0-1
	targets = 1-(np.log(targets)/np.log(50000))

	input_data = np.hstack((input_data, targets.reshape(targets.shape[0],1)))

	return input_data


def get_test_dataset(folder = "NetMHC"):
	aminoacids = ["G","P","A","V","L","I","M","C","F","Y","W","H","K","R","Q","N","E","D","S","T"]
	print ('Loading files...')
	data = pd.read_csv(f'{folder}/hpv_predictions.csv')
	print ('Loading files...')
	data = data[[len(i)<12 and len(i)>7 for i in data['peptide']]]
	data['allele'] = [''.join(i.split('*')) for i in data['allele']]

	### loading the allele data from specified folder
	alleles = pd.read_csv(f'{folder}/MHC_pseudo.dat',header=None,sep=' ')
	alleles = alleles[['HLA' in i for i in alleles[0]]]
	alleles.columns = ['HLA_allele', 'allele_seq']


	### The overlap is only the alleles for which we have a sequence!
	overlapping_alleles = set(data['allele'])&set(alleles['HLA_allele'])
	data = data[[i in overlapping_alleles for i in data['allele']]]

	max_length_allele = int(np.max(alleles['allele_seq'].str.len()))

	print ('Encoding alleles to sparse...')
	allele_seq = get_sparse_alleles(alleles, aminoacids, max_length_allele)
	alleles['allele_seq_sparse'] = allele_seq

	### merging the allele sequences with the data
	data = data.merge(alleles, left_on='allele', right_on = 'HLA_allele')

	max_length_peptides = int(np.max(data['peptide'].str.len()))
	### getting the sparse encodings for the peptides 
	print ('Encoding peptides to sparse...')
	peptide_sparse = get_sparse_peptides(data['peptide'], aminoacids, max_length_peptides)
	data['peptide_sparse'] = peptide_sparse


	### merging these encodings into the dataset
	
	input_data = merge_dataset(data, 'peptide_sparse', 'allele_seq_sparse', aminoacids, max_length_allele, max_length_peptides)

	### processing the targets
	targets = data['Affinity (uM)']
	### bringing this to the same scale as the others 
	###TODO: find a better way?
	targets*=5
	targets = np.array([max(1,i) for i in targets ])
	### values are capped at 50k according to MHCflurry 
	targets = np.array([min(50000,i) for i in targets])

	### transforming according to MHCflurry formula to range 0-1
	targets = 1-(np.log(targets)/np.log(50000))

	input_data = np.hstack((input_data, targets.reshape(targets.shape[0],1)))

	return input_data



def get_train_dataset(folder = 'NetMHCpan_data', task='single_allele', min_nb_examples=1000):
	"""
	This function prepares the dataset for the second task: predicting pan-allele peptide
	binding specificities.
	This function performs the encoding of the MHC and peptide sequences to sparse format
	"""
    if not os.path.exists(f'{folder}/cached_train_dataset_{task}.npy'):
        print ('hi')
        ### we use the following list for amino acid letter codes
        aminoacids = ["G","P","A","V","L","I","M","C","F","Y","W","H","K","R","Q","N","E","D","S","T"]
        ### getting the dataset
        data, alleles, overlapping_alleles = load_pMHC_dataset(folder)
        if task == 'pan_allele':
            print ('Encoding alleles to sparse...')
            max_length_allele = int(np.max(alleles['allele_seq'].str.len()))
            allele_seq = get_sparse_alleles(alleles, aminoacids, max_length_allele)
            alleles['allele_seq_sparse'] = allele_seq

            ### merging the allele sequences with the data
            data = data.merge(alleles, left_on='allele', right_on = 'HLA_allele')
        ### getting the sparse encodings for the peptides 
        print ('Encoding peptides to sparse...')
        max_length_peptides = int(np.max(data['peptide'].str.len()))
        peptide_sparse = get_sparse_peptides(data['peptide'], aminoacids, max_length_peptides)
        data['peptide_sparse'] = peptide_sparse

        if task=='pan_allele':
            ### mergin these encodings into the dataset
            input_data = merge_dataset(data, 'peptide_sparse', 'allele_seq_sparse', aminoacids, max_length_allele, max_length_peptides)
        else:
            input_data = data

        print ('organizing targets')
        targets = data['measurement_value']

        ### values are capped at 50k according to MHCflurry 
        targets = np.array([min(50000,i) for i in targets ])
        targets = np.array([max(1,i) for i in targets ])
        ### transforming according to MHCflurry formula to range 0-1
        targets = 1-(np.log(targets)/np.log(50000))

        input_data = np.hstack((input_data, targets.reshape(targets.shape[0],1)))

        example_per_allele = data.groupby('allele').count()['peptide']
        keep_alleles = list(example_per_allele[example_per_allele>=min_nb_examples].index)

        data_dict = {}
        for allele in keep_alleles:
            temp_ix = np.arange(data.shape[0])[data['allele']==allele]
            temp = input_data[temp_ix,:]
            assert temp.shape[0]==example_per_allele.iloc[keep_alleles.index(allele)]
            data_dict[allele] = temp

        pickle.dump(data_dict, open(f'{folder}/cached_train_dataset_{task}.npy','wb'))


    else:
        print ('Pre-processed data exists')
        print ('Loading cached data...')
        input_data = pickle.load(open(f'{folder}/cached_train_dataset_{task}.npy','rb'))

	print ('Done!')
	return input_data
