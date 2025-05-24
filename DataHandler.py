import pickle
import numpy as np
from scipy.sparse import coo_matrix
from Params import args
import scipy.sparse as sp
import torch
import torch.utils.data as data
import torch.utils.data as dataloader
from collections import defaultdict
from tqdm import tqdm
import random

class DataHandler:
	def __init__(self):
		if args.data == 'baby':
			predir = './Datasets/baby/'
		elif args.data == 'sports':
			predir = './Datasets/sports/'
		elif args.data == 'tiktok':
			predir = './Datasets/tiktok/'
		elif args.data == 'baby1':
			predir = './Datasets/baby1/'
		elif args.data == 'sports1':
			predir = './Datasets/sports1/'
		elif args.data == 'TMALL_CLIP':
			predir = './Datasets/TMALL_CLIP/'
		elif args.data == 'TMALL_imgaebind':
			predir = './Datasets/TMALL_imgaebind/'
		elif args.data == 'Microlens_CLIP':
			predir = './Datasets/Microlens_CLIP/'
		elif args.data == 'HM_CLIP':
			predir = './Datasets/HM_CLIP/'
		elif args.data == 'netflix':
			predir = './Datasets/netflix/'
		elif args.data == 'Microlens_imagebind':
			predir = './Datasets/Microlens_imagebind/'

		self.predir = predir
		self.trnfile = predir + 'trnMat.pkl'
		self.tstfile = predir + 'tstMat.pkl'
		self.valfile = predir + 'valMat.pkl'

		self.imagefile = predir + 'image_feat.npy'
		self.textfile = predir + 'text_feat.npy'
		if args.data == 'tiktok':
			self.audiofile = predir + 'audio_feat.npy'

	def loadOneFile(self, filename):
		with open(filename, 'rb') as fs:
			ret = (pickle.load(fs) != 0).astype(np.float32)
			# ret = pickle.load(fs)
		if type(ret) != coo_matrix:
			ret = sp.coo_matrix(ret)
		return ret

	def normalizeAdj(self, mat): 
		degree = np.array(mat.sum(axis=-1))
		degree[degree == 0] = 1e-7
		dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
		dInvSqrt[np.isinf(dInvSqrt)] = 0.0
		dInvSqrtMat = sp.diags(dInvSqrt)
		return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

	def makeTorchAdj(self, mat):
		# make ui adj
		a = sp.csr_matrix((args.user, args.user))
		b = sp.csr_matrix((args.item, args.item))
		mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
		mat = (mat != 0) * 1.0
		mat = (mat + sp.eye(mat.shape[0])) * 1.0
		mat = self.normalizeAdj(mat)

		# make cuda tensor
		idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = torch.from_numpy(mat.data.astype(np.float32))
		shape = torch.Size(mat.shape)

		# Convert self.trnMat to PyTorch sparse tensor
		trn_idxs = torch.from_numpy(np.vstack([self.trnMat.row, self.trnMat.col]).astype(np.int64))
		trn_vals = torch.from_numpy(self.trnMat.data.astype(np.float32))
		trn_shape = torch.Size(self.trnMat.shape)
		trnMat_torch = torch.sparse.FloatTensor(trn_idxs, trn_vals, trn_shape).cuda()

		return torch.sparse.FloatTensor(idxs, vals, shape).cuda(), trnMat_torch

	def loadFeatures(self, filename):
		feats = np.load(filename)
		return torch.tensor(feats).float().cuda(), np.shape(feats)[1]

	def LoadData(self):
		trnMat = self.loadOneFile(self.trnfile)
		tstMat = self.loadOneFile(self.tstfile)
		valMat = self.loadOneFile(self.valfile)
		self.trnMat = trnMat

		num_edges = len(self.trnMat.row)
		# print("self.trnMat")
		# print(f"Number of edges: {num_edges}")

		args.user, args.item = trnMat.shape
		self.torchBiAdj, self.torchTrnMat = self.makeTorchAdj(trnMat)
		self.num_inters = self.calculate_sumArr(self.trnMat)
		self.num_inters = torch.FloatTensor(1.0 / (self.num_inters + 1e-7)).cuda()

		trnData = TrnData(trnMat)
		self.trnLoader = dataloader.DataLoader(trnData, batch_size=args.batch, shuffle=True, num_workers=0)
		tstData = TstData(tstMat, trnMat)
		self.tstLoader = dataloader.DataLoader(tstData, batch_size=args.tstBat, shuffle=False, num_workers=0)
		valData = ValData(valMat, trnMat)
		self.valLoader = dataloader.DataLoader(valData, batch_size=args.valBat, shuffle=False, num_workers=0)

		self.image_feats, args.image_feat_dim = self.loadFeatures(self.imagefile)
		self.text_feats, args.text_feat_dim = self.loadFeatures(self.textfile)
		if args.data == 'tiktok':
			self.audio_feats, args.audio_feat_dim = self.loadFeatures(self.audiofile)

		print(f"Training interactions: {self.trnMat.nnz}")
		print(f"Validation interactions: {valMat.nnz}")
		print(f"Test interactions: {tstMat.nnz}")
		print(f"Image feature shape: {self.image_feats.shape}")
		print(f"Text feature shape: {self.text_feats.shape}")
		if args.data == 'tiktok':
			print(f"Audio feature shape: {self.audio_feats.shape}")

		# self.diffusionData = DiffusionData(torch.FloatTensor(self.trnMat.toarray()))
		# self.diffusionLoader = dataloader.DataLoader(self.diffusionData, batch_size=args.batch, shuffle=True, num_workers=0)

		self.A_v, self.R_v, self.A_t, self.R_t = self.torchBiAdj, self.torchTrnMat, self.torchBiAdj, self.torchTrnMat
		if args.data == 'tiktok':
			self.A_a, self.R_a = self.torchBiAdj, self.torchTrnMat

	# new added
	def calculate_sumArr(self, trnMat):
		user_degrees = np.array(trnMat.sum(axis=1)).flatten()  # Shape: (n_users,)
		item_degrees = np.array(trnMat.sum(axis=0)).flatten()  # Shape: (n_items,)

		sumArr = np.concatenate([user_degrees, item_degrees])  # Shape: (n_users + n_items,)
		return sumArr

	def compute_interaction_score(self, user_output, item_output):
		scores = torch.matmul(user_output, item_output.T) # e_u^T e~_i
		return scores

	def update_weights(self, trnMat, scores, tau):
		trnMat_coo = trnMat.tocoo()
		rows = trnMat_coo.row
		cols = trnMat_coo.col

		updated_data = 1 + tau * scores[rows, cols].detach().cpu().numpy()  # 关键修改

		updated_trnMat = sp.coo_matrix((updated_data, (rows, cols)), shape=trnMat.shape)
		return updated_trnMat.tocsr()

	def process_graphs(self, tau, rho, v_user_output, v_item_output,
	                   t_user_output, t_item_output, a_user_output=None, a_item_output=None):

		v_scores = self.compute_interaction_score(v_user_output, v_item_output)
		updated_v_trnMat = self.update_weights(self.trnMat, v_scores, tau)
		updated_v_trnMat = self.compute_edge_keep_prob(updated_v_trnMat, rho)
		self.A_v, self.R_v = self.makeTorchAdj(updated_v_trnMat)

		t_scores = self.compute_interaction_score(t_user_output, t_item_output)
		updated_t_trnMat = self.update_weights(self.trnMat, t_scores, tau)
		updated_t_trnMat = self.compute_edge_keep_prob(updated_t_trnMat, rho)
		self.A_t, self.R_t = self.makeTorchAdj(updated_t_trnMat)

		if args.data == 'tiktok':
			a_scores = self.compute_interaction_score(a_user_output, a_item_output)
			updated_a_trnMat = self.update_weights(self.trnMat, a_scores, tau)
			updated_a_trnMat = self.compute_edge_keep_prob(updated_a_trnMat, rho)
			self.A_a, self.R_a = self.makeTorchAdj(updated_a_trnMat)

	def compute_edge_keep_prob(self, updated_trnMat, edge_keeprate):
	    coo = updated_trnMat.tocoo()
	    rows, cols, weights = coo.row, coo.col, coo.data

	    row_sums = np.array(updated_trnMat.sum(axis=1)).flatten() + 1e-7
	    col_sums = np.array(updated_trnMat.sum(axis=0)).flatten() + 1e-7

	    valid_mask = (row_sums[rows] > 0) & (col_sums[cols] > 0)
	    rows, cols, weights = rows[valid_mask], cols[valid_mask], weights[valid_mask]

	    keep_probs = 1 / np.sqrt(row_sums[rows] * col_sums[cols])
	    keep_probs += 1e-8
	    keep_probs /= keep_probs.sum() + 1e-7

	    total_edges = len(weights)
	    num_kept_edges = int(total_edges * edge_keeprate)

	    if np.sum(keep_probs > 0) < num_kept_edges:
	        num_kept_edges = np.sum(keep_probs > 0)

	    kept_indices = np.random.choice(total_edges, size=num_kept_edges, replace=False, p=keep_probs)

	    kept_rows = rows[kept_indices]
	    kept_cols = cols[kept_indices]
	    kept_weights = weights[kept_indices]

	    print(f"Total edges: {total_edges}")
	    print(f"Non-zero keep_probs: {np.sum(keep_probs > 0)}")
	    print(f"Edges to keep: {num_kept_edges}")

	    sampled_matrix = sp.csr_matrix((kept_weights, (kept_rows, kept_cols)), shape=updated_trnMat.shape)

	    return sampled_matrix

	def compute_edge_keep_prob0(self, updated_trnMat, edge_keeprate):

		coo = updated_trnMat.tocoo()
		rows, cols, weights = coo.row, coo.col, coo.data

		row_sums = np.array(updated_trnMat.sum(axis=1)).flatten() + 1e-7
		col_sums = np.array(updated_trnMat.sum(axis=0)).flatten() + 1e-7

		valid_mask = (row_sums[rows] > 0) & (col_sums[cols] > 0)
		rows, cols, weights = rows[valid_mask], cols[valid_mask], weights[valid_mask]

		keep_probs = 1 / np.sqrt(row_sums[rows] * col_sums[cols])
		keep_probs /= keep_probs.sum() + 1e-7

		alpha = keep_probs # * 10
		keep_probs = np.random.dirichlet(alpha)

		total_edges = len(weights)
		num_kept_edges = int(total_edges * edge_keeprate)

		kept_indices = np.random.choice(total_edges, size=num_kept_edges, replace=False, p=keep_probs)

		kept_rows = rows[kept_indices]
		kept_cols = cols[kept_indices]
		kept_weights = weights[kept_indices]

		sampled_matrix = sp.csr_matrix((kept_weights, (kept_rows, kept_cols)), shape=updated_trnMat.shape)

		return sampled_matrix

class TrnData(data.Dataset):
	def __init__(self, coomat):
		self.rows = coomat.row
		self.cols = coomat.col
		self.dokmat = coomat.todok()
		self.negs = np.zeros(len(self.rows)).astype(np.int32)

	def negSampling(self):
		for i in range(len(self.rows)):
			u = self.rows[i]
			while True:
				iNeg = np.random.randint(args.item)
				if (u, iNeg) not in self.dokmat:
					break
			self.negs[i] = iNeg

	def __len__(self):
		return len(self.rows)

	def __getitem__(self, idx):
		return self.rows[idx], self.cols[idx], self.negs[idx]

class TstData(data.Dataset):
	def __init__(self, coomat, trnMat):
		self.csrmat = (trnMat.tocsr() != 0) * 1.0

		tstLocs = [None] * coomat.shape[0]
		tstUsrs = set()
		for i in range(len(coomat.data)):
			row = coomat.row[i]
			col = coomat.col[i]
			if tstLocs[row] is None:
				tstLocs[row] = list()
			tstLocs[row].append(col)
			tstUsrs.add(row)
		tstUsrs = np.array(list(tstUsrs))
		self.tstUsrs = tstUsrs
		self.tstLocs = tstLocs

	def __len__(self):
		return len(self.tstUsrs)

	def __getitem__(self, idx):
		return self.tstUsrs[idx], np.reshape(self.csrmat[self.tstUsrs[idx]].toarray(), [-1])

class ValData(data.Dataset):
	def __init__(self, coomat, trnMat):
		self.csrmat = (trnMat.tocsr() != 0) * 1.0

		valLocs = [None] * coomat.shape[0]
		valUsrs = set()
		for i in range(len(coomat.data)):
			row = coomat.row[i]
			col = coomat.col[i]
			if valLocs[row] is None:
				valLocs[row] = list()
			valLocs[row].append(col)
			valUsrs.add(row)
		valUsrs = np.array(list(valUsrs))
		self.valUsrs = valUsrs
		self.valLocs = valLocs

	def __len__(self):
		return len(self.valUsrs)

	def __getitem__(self, idx):
		return self.valUsrs[idx], np.reshape(self.csrmat[self.valUsrs[idx]].toarray(), [-1])
	
class DiffusionData(data.Dataset):
	def __init__(self, data):
		self.data = data

	def __getitem__(self, index):
		item = self.data[index]
		return item, index
	
	def __len__(self):
		return len(self.data)