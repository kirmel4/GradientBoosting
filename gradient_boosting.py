import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import roc_auc_score

class GradientBoosting():
	def __init__ (self, n_estimators = 100, learning_rate = 0.01, max_depth = 3):
		self.n_estimators = n_estimators
		self.learning_rate = learning_rate
		self.max_depth = max_depth
		self.trees = []
		self.y_mean = None
	def fit(self, X, y):
		self.y_mean = y.mean()
		y_pred = np.array([self.y_mean for i in range(len(X))])
		for i in range(self.n_estimators):
			residual = y - y_pred
			tree = DecisionTreeRegressor(max_depth = self.max_depth)
			tree.fit(X, residual)
			self.trees.append(tree)
			y_pred += self.learning_rate * tree.predict(X)
	def predict(self, X):
		y_pred = np.array([self.y_mean for i in range(len(X))])
		for tree in self.trees: y_pred += self.learning_rate * tree.predict(X)
		return y_pred
	def get_params(self):
		self.params = {'n_estimators' : self.n_estimators,
				'learning_rate' : self.learning_rate,
				 'max_depth': self.max_depth}
		return self.params