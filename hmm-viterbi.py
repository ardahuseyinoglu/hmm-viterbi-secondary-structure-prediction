# Arda Huseyinoglu

import numpy as np
import sys

#import seaborn as sn
#import pandas as pd
#import matplotlib.pyplot as plt



# read training file

with open('BBM411_Assignment2_Q3_TrainingDataset.txt') as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]
    


# storing headers, aminoacid sequences and secondary structure sequences in different arrays

headers = []
aa_sequences = []
ss_sequences = []

for i,line in enumerate(lines):
    if i % 3 == 0:
        headers.append(line)
    elif i % 3 == 1:
        aa_sequences.append(line)
    elif i % 3 == 2:
        ss_sequences.append(line)
        


# cutting the regions including elements designated by "_" symbol out
# from both amino acid sequences and from SS elements sequences

empty_sequences_indices = []
for sample_index in range(len(headers)):
    deleted_indices = np.where(np.array(list(ss_sequences[sample_index])) == '_')[0]
    updated_ss_seq = np.delete(np.array(list(ss_sequences[sample_index])), deleted_indices)
    updated_aa_seq = np.delete(np.array(list(aa_sequences[sample_index])), deleted_indices)
    ss_sequences[sample_index] = updated_ss_seq
    aa_sequences[sample_index] = updated_aa_seq
    if len(updated_ss_seq) == 0:
        empty_sequences_indices.append(sample_index)
    
    
    
# delete samples having empty sequences

ss_sequences[:] = [ss_seq for i, ss_seq in enumerate(ss_sequences) if i not in empty_sequences_indices]
aa_sequences[:] = [aa_seq for i, aa_seq in enumerate(aa_sequences) if i not in empty_sequences_indices]
headers[:] = [header for i, header in enumerate(headers) if i not in empty_sequences_indices]



# simplify eight states into three states

states_mapping_dict = {'G':'H', 'H':'H', 'I':'H', 
                       'B':'E', 'E':'E', 
                       'T':'T', 'S':'T', 'L':'T'}

for sample_index in range(len(ss_sequences)):
    ss_sequences[sample_index] = np.vectorize(states_mapping_dict.get)(ss_sequences[sample_index])
    


# capitalize lower case letters in aminoacid sequences 

for i in range(len(aa_sequences)):
    aa_sequences[i] = np.char.capitalize(aa_sequences[i])
    


	
# TRAINING (LEARNING PARAMETERS OF HMM)

# initializing corresponding matrices

possible_aa_letters = list({i for aa_seq in aa_sequences for i in set(aa_seq)})
possible_ss_letters = list({i for ss_seq in ss_sequences for i in set(ss_seq)})
possible_aa_letters.sort()
possible_ss_letters.sort()

# 3 (E, H, T) for rows and columns
transition_matrix = np.zeros(shape=(len(possible_ss_letters),len(possible_ss_letters)))
# 3 (E, H, T) for rows and 25 aminoacids for columns
emission_matrix = np.zeros(shape=(len(possible_ss_letters),len(possible_aa_letters)))
# 3 (E, H, T) for rows
initial_state_matrix = np.zeros(shape=(len(possible_ss_letters),1))  



# learning transition_matrix

for sample in ss_sequences:
    for i in range(len(sample)-1):
        transition_matrix[possible_ss_letters.index(sample[i])][possible_ss_letters.index(sample[i+1])] += 1

transition_matrix = transition_matrix / np.sum(transition_matrix, axis=1)[:,None]
transition_matrix = np.log2(transition_matrix)
print("training for transition matrix finished")


# learning emission_matrix

for sample_i in range(len(ss_sequences)):
    ss = ss_sequences[sample_i]
    aa = aa_sequences[sample_i]
    for i in range(len(ss)):
        emission_matrix[possible_ss_letters.index(ss[i])][possible_aa_letters.index(aa[i])] += 1

emission_matrix = (emission_matrix + 1) / (np.sum(emission_matrix, axis=1) + len(possible_aa_letters))[:,None]
emission_matrix = np.log2(emission_matrix)
print("training for emission matrix finished")


# learning initial_state_matrix

initial_letters_of_ss = [ss_seq[0] for ss_seq in ss_sequences]
unique, counts = np.unique(np.array(initial_letters_of_ss), return_counts=True)
probs = counts / counts.sum()
for letter,prob in zip(unique, probs):
    initial_state_matrix[possible_ss_letters.index(letter)] = prob
    
initial_state_matrix = np.log2(initial_state_matrix)
print("training for initial state matrix finished")




transition_matrix[0][0] = transition_matrix[0][0] * 4.8
transition_matrix[1][1] = transition_matrix[1][1] * 10
transition_matrix[2][2] = transition_matrix[2][2] * 2.8


# handle command line args
measure_flag = 0
if len(sys.argv) == 2:
	input_file_name = sys.argv[1]
if len(sys.argv) == 3:
	input_file_name = sys.argv[1]
	gt_ss_file_name = sys.argv[2]
	measure_flag = 1


# get amino acid sequence of corresponding protein
with open(input_file_name) as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]

protein_header = lines[0]
test_seq = lines[1]


# VITERBI

def viterbi(state, i):
     
    ep = emission_matrix[possible_ss_letters.index(state)][possible_aa_letters.index(test_seq[i])]
    max_term_list = [partial_scores_table[0][i-1] + transition_matrix[0][possible_ss_letters.index(state)], 
                     partial_scores_table[1][i-1] + transition_matrix[1][possible_ss_letters.index(state)],
                     partial_scores_table[2][i-1] + transition_matrix[2][possible_ss_letters.index(state)]]
    
    prob = ep + max(max_term_list)
    backtrace_from = np.argmax(np.array(max_term_list)) 
    
    return prob, possible_ss_letters[backtrace_from]



# viterbi initialization

partial_scores_table = np.zeros(shape=(len(possible_ss_letters),len(test_seq)))
backtrace_table = np.zeros(shape=(len(possible_ss_letters),len(test_seq)), dtype='object')

for state in possible_ss_letters:
    isp = initial_state_matrix[possible_ss_letters.index(state)]
    ep = emission_matrix[possible_ss_letters.index(state)][possible_aa_letters.index(test_seq[0])]
    partial_scores_table[possible_ss_letters.index(state)][0] = isp + ep 
    


# fill partial score table

for i in range(1, len(test_seq)):
    for state in possible_ss_letters:
        prob, backtrace_from = viterbi(state, i)
        partial_scores_table[possible_ss_letters.index(state)][i] = prob
        backtrace_table[possible_ss_letters.index(state)][i] = backtrace_from



# backtrace to get ss prediction

last_col_max_index = np.argmax(partial_scores_table[:,-1])
hidden_seq = possible_ss_letters[last_col_max_index]
path_prob_log = partial_scores_table[last_col_max_index][-1]

for i in range(backtrace_table.shape[1]-1,0,-1):
    letter = backtrace_table[last_col_max_index][i]
    hidden_seq += letter
    last_col_max_index = possible_ss_letters.index(letter)
    
pred_ss_sequence = hidden_seq[::-1]




# write to output file

with open("output.txt", "w") as text_file:
    text_file.write(protein_header)
    text_file.write('\n')
    text_file.write(test_seq)
    text_file.write('\n')
    text_file.write(''.join(pred_ss_sequence))
    text_file.write('\n')
    text_file.write(f'probability of the path: 2^({path_prob_log})')

print("\n*Most possible path and path probability were written in output.txt\n")





#--------------------------------------------------------------------------------
# MEASURE PERFORMANCE

# read ground truth secondary structure sequence of corresponding protein

if measure_flag == 1:
	with open(gt_ss_file_name) as file:
		lines = file.readlines()
		lines = [line.rstrip() for line in lines]

	gt_regions = []
	for line in lines:
		splitted_line = line.split()
		
		element = splitted_line[0][0]
		if element == 'S':
			element = 'E'
		
		gt_regions.append([element, [int(splitted_line[1]) - 1, int(splitted_line[2]) - 1]])

	gt_ss_sequence = ['_'] * len(test_seq)
	for a_region in gt_regions:
		for pos in range(a_region[1][0], a_region[1][1] + 1):
			gt_ss_sequence[pos] = a_region[0]

			

	def conf_matrix_row(gt_element):
		count_pred_as_H = 0
		count_pred_as_E = 0
		count_pred_as_T = 0

		for i in range(len(test_seq)):
			if gt_ss_sequence[i] == gt_element:
				if pred_ss_sequence[i] == 'H':
					count_pred_as_H += 1
				if pred_ss_sequence[i] == 'E':
					count_pred_as_E += 1
				if pred_ss_sequence[i] == 'T':
					count_pred_as_T += 1
					
		return count_pred_as_H, count_pred_as_E, count_pred_as_T

	hh, he, ht = conf_matrix_row('H')
	eh, ee, et = conf_matrix_row('E')
	th, te, tt = conf_matrix_row('T')


	tp_h = hh
	tp_e = ee
	tp_t = tt

	tn_h = ee+et+te+tt
	tn_e = hh+ht+th+tt
	tn_t = hh+he+eh+ee

	fp_h = eh+th
	fp_e = he+te
	fp_t = ht+et

	fn_h = he+ht
	fn_e = eh+et
	fn_t = th+te

	prec_h = tp_h/(tp_h+fp_h)
	prec_e = tp_e/(tp_e+fp_e)
	prec_t = tp_t/(tp_t+fp_t)

	recall_h = tp_h/(tp_h+fn_h)
	recall_e = tp_e/(tp_e+fn_e)
	recall_t = tp_t/(tp_t+fn_t)

	f1_h = (2*prec_h*recall_h)/(prec_h+recall_h)
	f1_e = (2*prec_e*recall_e)/(prec_e+recall_e)
	f1_t = (2*prec_t*recall_t)/(prec_t+recall_t)

	acc_h = (tp_h+tn_h)/(tp_h+tn_h+fp_h+fn_h)
	acc_e = (tp_e+tn_e)/(tp_e+tn_e+fp_e+fn_e)
	acc_t = (tp_t+tn_t)/(tp_t+tn_t+fp_t+fn_t)


	print('\t\t\t    Predicted')
	print('\t\t\tH\tE\tT')
	print(f'\t\tH\t{hh}\t{he}\t{ht}')
	print(f'Ground Truth\tE\t{eh}\t{ee}\t{et}')
	print(f'\t\tT\t{th}\t{te}\t{tt}')

	print("\n")
	print("\tPrec\t\tRecall\t\tF1\t\tAcc")
	print(f"H\t{round(prec_h,4)}\t\t{round(recall_h,4)}\t\t{round(f1_h,4)}\t\t{round(acc_h,4)}")
	print(f"E\t{round(prec_e,4)}\t\t{round(recall_e,4)}\t\t{round(f1_e,4)}\t\t{round(acc_e,4)}")
	print(f"T\t{round(prec_t,4)}\t\t{round(recall_t,4)}\t\t{round(f1_t,4)}\t\t{round(acc_t,4)}")

	print(f'\nOverall Accuracy: {(hh+ee+tt)/(hh+he+ht+eh+ee+et+th+te+tt)}')

"""
df_cm = pd.DataFrame([[hh,he,ht], [eh, ee, et], [th, te, tt]], index = ['H', 'E', 'T'], columns = ['H', 'E', 'T'])
ax = sn.heatmap(df_cm, annot=True)
plt.title("Confusion Matrix\n", fontsize =15)
plt.xlabel('Predicted', fontsize = 12)
plt.ylabel('Ground Truth', fontsize = 12)
plt.show()
"""
