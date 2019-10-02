import numpy as np
import scipy
from sklearn import cross_decomposition
import simple_stats

class CCA(object):
	"""Cannonical correlation analysis"""
	def __init__(self, first_view, second_view, num_latent_dims, reg_factor = 0):
		self.first_view_matrix = first_view
		self.second_view_matrix = second_view
		self.k = num_latent_dims
		self.reg_factor = reg_factor

	def correlate(self, sklearn = True):
		self.sklearn = sklearn
		if sklearn: 
			self.correlate_sklearn()
		else:
			self.correlate_raw()

	def correlate_raw(self):
		cov_mat_first = simple_stats.covariance_matrix(self.first_view_matrix, self.first_view_matrix) + np.multiply(self.reg_factor, np.identity(self.first_view_matrix.shape[1]))
		cov_mat_second = simple_stats.covariance_matrix(self.second_view_matrix, self.second_view_matrix) + np.multiply(self.reg_factor, np.identity(self.second_view_matrix.shape[1]))
		cov_mat_inter = simple_stats.covariance_matrix(self.first_view_matrix, self.second_view_matrix)
		
		#first_inv_sqrt = np.linalg.inv(scipy.linalg.sqrtm(cov_mat_first))
		#second_inv_sqrt = np.linalg.inv(scipy.linalg.sqrtm(cov_mat_second))
		first_inv_sqrt = scipy.linalg.sqrtm(np.linalg.inv(cov_mat_first))
		second_inv_sqrt = scipy.linalg.sqrtm(np.linalg.inv(cov_mat_second))

		prod = np.matmul(np.matmul(first_inv_sqrt, cov_mat_inter), second_inv_sqrt)
		U, s, V = np.linalg.svd(prod, full_matrices = False)
		
		first_projector = U[:, : self.k]
		second_projector = (np.transpose(V))[:, : self.k]
		self.model = [first_projector,  second_projector]

	def correlate_sklearn(self):
		print("CCA training...")
		skcca = cross_decomposition.CCA(n_components = self.k, max_iter = 1000)
		skcca.fit(self.first_view_matrix, self.second_view_matrix)
		self.model = skcca

	def transform(self, first_view, second_view):
		if self.sklearn:
			print("CCA transofrming...")
			proj_first, proj_second = self.model.transform(first_view, second_view)
		else:
			proj_first = np.matmul(first_view, self.model[0])
			proj_second = np.matmul(second_view, self.model[1])
		return proj_first, proj_second
		
		
		
		
		
		
		
		
		
		

	

