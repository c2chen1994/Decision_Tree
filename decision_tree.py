import numpy as np
from typing import List
from classifier import Classifier

class DecisionTree(Classifier):
	def __init__(self):
		self.clf_name = "DecisionTree"
		self.root_node = None

	def train(self, features: List[List[float]], labels: List[int]):
		# init.
		assert(len(features) > 0)
		self.feautre_dim = len(features[0])
		num_cls = np.max(labels)+1
		num_avail_lbs = [n for n in range(self.feautre_dim)]

		# build the tree
		self.root_node = TreeNode(features, labels, num_cls, num_avail_lbs)
		if self.root_node.splittable:
			self.root_node.split()

		return
		
	def predict(self, features: List[List[float]]) -> List[int]:
		y_pred = []
		for feature in features:
			y_pred.append(self.root_node.predict(feature))
		return y_pred

	def print_tree(self, node=None, name='node 0', indent=''):
		if node is None:
			node = self.root_node
		print(name + '{')
		
		string = ''
		for idx_cls in range(0, node.num_cls):
			string += str(node.labels.count(idx_cls)) + ' '
		print(indent + ' num of sample / cls: ' + string)

		if node.splittable:
			print(indent + '  split by dim {:d}'.format(node.dim_split))
			for idx_child, child in enumerate(node.children):
				self.print_tree(node=child, name= '  '+name+'/'+str(idx_child), indent=indent+'  ')
		else:
			print(indent + '  cls', node.cls_max)
		print(indent+'}')


class TreeNode(object):
	def __init__(self, features: List[List[float]], labels: List[int], num_cls: int, num_avail_lbs: List[int]):
		self.features = features
		self.labels = labels
		self.children = []
		self.num_cls = num_cls
		self.num_avail_lbs = num_avail_lbs

		count_max = 0
		for label in np.unique(labels):
			if self.labels.count(label) > count_max:
				count_max = labels.count(label)
				self.cls_max = label # majority of current node
		if (len(np.unique(labels)) < 2 or len(self.features) < 1 or len(self.features[0]) < 1 or len(self.num_avail_lbs) < 1):
			self.splittable = False
		else:
			self.splittable = True

		self.dim_split = None # the index of the feature to be split

		self.feature_uniq_split = None # the possible unique values of the feature to be split


	def split(self):
		def conditional_entropy(branches: List[List[int]]) -> float:
			'''
			branches: C x B array, 
					  C is the number of classes,
					  B is the number of branches
					  it stores the number of 
					  corresponding training samples 
					  e.g.
					              ○ ○ ○ ○
					              ● ● ● ●
					            ┏━━━━┻━━━━┓
				               ○ ○       ○ ○
				               ● ● ● ●
				               
				      branches = [[2,2], [4,0]]
			'''
			########################################################
			# TODO: compute the conditional entropy
			########################################################
			branches = np.array(branches)
			bsum = np.sum(branches, axis = 0)
			bpro = branches / bsum
			wholePro = bsum / np.sum(bsum)
			#bpro_ = bpro + np.floor(np.exp(-bpro))
			bpro[bpro == 0] = 1
			etp = bpro * np.log2(bpro)
			etpSum = np.abs(np.sum(etp, axis = 0))
			etpFinal = np.sum(etpSum * wholePro)
			return etpFinal

			
		bestId = 0
		e = 10000000
		featuresArray = np.array(self.features)
		N = featuresArray.shape[0]
		lb = np.array(self.labels)
		lbU = np.unique(lb)
		for idx_dim in range(len(self.features[0])):
		############################################################
		# TODO: compare each split using conditional entropy
		#       find the best split
		############################################################
			if idx_dim not in self.num_avail_lbs:
				continue
			cl = featuresArray[:, idx_dim]
			clU = np.unique(cl)
			branches = []
			for j in range(0, len(lbU)):
				branch = []
				for i in range(0, len(clU)):
					branch.append(np.hstack((cl.reshape(N, 1), lb.reshape(N, 1))).tolist().count([clU[i], lbU[j]]))
				branches.append(branch)
			etpCur = conditional_entropy(branches)
			if etpCur < e:
				bestId = idx_dim
				e = etpCur
				self.dim_split = bestId # the index of the feature to be split
				self.feature_uniq_split = clU.tolist()




		############################################################
		# TODO: split the node, add child nodes
		############################################################
		num_avail_lbs = []
		for i in self.num_avail_lbs:
			num_avail_lbs.append(i)
		num_avail_lbs.remove(bestId)
		cl = featuresArray[:, bestId]
		clU = np.unique(cl)
		iList = [n for n in range(len(featuresArray[0]))]
		for i in range(0, len(self.feature_uniq_split)):
			newFeatures = featuresArray[featuresArray[:,bestId] == clU[i]].tolist()
			cur = np.hstack((cl.reshape(N, 1), lb.reshape(N, 1)))
			nL = cur[cur[:,0] == clU[i]][:, 1]
			newLabels = nL.tolist()
			child = TreeNode(newFeatures, newLabels, len(np.unique(nL)), num_avail_lbs)
			self.children.append(child)

		# split the child nodes
		for child in self.children:
			if child.splittable:
				child.split()

		return

	def predict(self, feature: List[int]) -> int:
		if self.splittable:
			idx_child = self.feature_uniq_split.index(feature[self.dim_split])
			return self.children[idx_child].predict(feature)
		else:
			return self.cls_max



