import numpy as np
import math

def sign_mismatches(predicts, gold):
	count = 0
	sum = 0.0	
	merge = np.sign(predicts) + np.sign(gold)
	for i in range(len(merge)):	
		if merge[i] == 0:
			count += 1
			sum += np.abs(predicts[i])
	return (count, sum)

def kullback_leibler(ground_prob_dist, target_prob_dist):
	sum = 0.0
	for i in range(len(ground_prob_dist)):
		sum += ground_prob_dist[i] * math.log(ground_prob_dist[i] / target_prob_dist[i])
	return sum

def cosine(vec1, vec2):
	return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def covariance_matrix(first, second):
	if first.shape[0] != second.shape[0]:
		raise ValueError("Input matrices must have the same number of row vectors (same first dimensions)")
	
	cov_mat = np.zeros((first.shape[1], second.shape[1]))
	for i in range(first.shape[0]):
		cov_mat = np.add(cov_mat, np.matmul(np.transpose([first[i]]), np.array([second[i]])))
	cov_mat = np.multiply(1.0 / float(first.shape[0]), cov_mat)
	return cov_mat	