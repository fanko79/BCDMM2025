import torch
from torch import nn
import torch.nn.functional as F
from Params import args
import numpy as np
import random
import math
from Utils.Utils import *
from Utils.TimeLogger import log
import os

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class Model(nn.Module):
	def __init__(self, image_embedding, text_embedding, audio_embedding=None):
		super(Model, self).__init__()

		self.uEmbeds = nn.Parameter(init(torch.empty(args.user, args.latdim)))
		self.iEmbeds = nn.Parameter(init(torch.empty(args.item, args.latdim)))
		self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gnn_layer)])

		self.edgeDropper = SpAdjDropEdge(args.keepRate)

		if args.trans == 1:
			self.image_trans = nn.Linear(args.image_feat_dim, args.latdim)
			self.text_trans = nn.Linear(args.text_feat_dim, args.latdim)
		elif args.trans == 0:
			self.image_trans = nn.Parameter(init(torch.empty(size=(args.image_feat_dim, args.latdim))))
			self.text_trans = nn.Parameter(init(torch.empty(size=(args.text_feat_dim, args.latdim))))
		else:
			self.image_trans = nn.Parameter(init(torch.empty(size=(args.image_feat_dim, args.latdim))))
			self.text_trans = nn.Linear(args.text_feat_dim, args.latdim)
		if audio_embedding != None:
			if args.trans == 1:
				self.audio_trans = nn.Linear(args.audio_feat_dim, args.latdim)
			else:
				self.audio_trans = nn.Parameter(init(torch.empty(size=(args.audio_feat_dim, args.latdim))))

		self.image_embedding = image_embedding
		self.text_embedding = text_embedding
		if audio_embedding != None:
			self.audio_embedding = audio_embedding
		else:
			self.audio_embedding = None

		if audio_embedding != None:
			self.modal_weight = nn.Parameter(torch.Tensor([0.3333, 0.3333, 0.3333]))
			self.modal_weight1 = nn.Parameter(torch.Tensor([0.3333, 0.3333, 0.3333]))
		else:
			self.modal_weight = nn.Parameter(torch.Tensor([0.5, 0.5]))
			self.modal_weight1 = nn.Parameter(torch.Tensor([0.5, 0.5]))
		self.modal_softmax = nn.Softmax(dim=0)
		self.modal_softmax1 = nn.Softmax(dim=0)

		self.dropout1 = nn.Dropout(p=args.dropout_rate2)
		self.dropout2 = nn.Dropout(p=args.dropout_rate1)

		self.leakyrelu = nn.LeakyReLU(0.2)

		# new_added
		out_dims = eval(args.dims) + [args.latdim]  # [1000] + [64]
		in_dims = out_dims[::-1]  # [64, 1000]

		# self.user_reverse_model = Denoise(in_dims, out_dims, args.d_emb_size, norm=args.norm).cuda()
		# self.item_reverse_model = Denoise(in_dims, out_dims, args.d_emb_size, norm=args.norm).cuda()
		self.image_reverse_model = Denoise(in_dims, out_dims, args.d_emb_size, norm=args.norm, dropout=args.denoise_drop).cuda()
		self.text_reverse_model = Denoise(in_dims, out_dims, args.d_emb_size, norm=args.norm, dropout=args.denoise_drop).cuda()
		if audio_embedding != None:
			self.audio_reverse_model = Denoise(in_dims, out_dims, args.d_emb_size, norm=args.norm).cuda()
		self.diffusion_model = GaussianDiffusion(args.noise_scale, args.noise_min, args.noise_max, args.steps).cuda()

		self.gate_v = nn.Sequential(
			nn.Linear(args.latdim, args.latdim),
			nn.Sigmoid()
		)
		self.gate_t = nn.Sequential(
			nn.Linear(args.latdim, args.latdim),
			nn.Sigmoid()
		)
		self.gate_f = nn.Sequential(
			nn.Linear(args.latdim, args.latdim),
			nn.Sigmoid()
		)

		dataset_path = os.path.abspath(args.data_path + args.data)
		image_adj_file = os.path.join(dataset_path, 'image_adj_{}_{}.pt'.format(args.image_knn_k, True))
		text_adj_file = os.path.join(dataset_path, 'text_adj_{}_{}.pt'.format(args.text_knn_k, True))
		# self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
		if os.path.exists(image_adj_file):
			image_adj = torch.load(image_adj_file)
		else:
			image_adj = build_sim(self.image_embedding.detach())
			image_adj = build_knn_normalized_graph(image_adj, topk=args.image_knn_k, is_sparse=True,
			                                       norm_type='sym')
			torch.save(image_adj, image_adj_file)
		self.image_original_adj = image_adj.cuda()

		# self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
		if os.path.exists(text_adj_file):
			text_adj = torch.load(text_adj_file)
		else:
			text_adj = build_sim(self.text_embedding.detach())
			text_adj = build_knn_normalized_graph(text_adj, topk=args.text_knn_k, is_sparse=True,
			                                      norm_type='sym')
			torch.save(text_adj, text_adj_file)
		self.text_original_adj = text_adj.cuda()

		if audio_embedding != None:
			audio_adj_file = os.path.join(dataset_path, 'audio_adj_{}_{}.pt'.format(args.audio_knn_k, True))
			if os.path.exists(audio_adj_file):
				audio_adj = torch.load(audio_adj_file)
			else:
				audio_adj = build_sim(self.audio_embedding.detach())
				audio_adj = build_knn_normalized_graph(audio_adj, topk=args.audio_knn_k, is_sparse=True,
				                                      norm_type='sym')
				torch.save(audio_adj, audio_adj_file)
			self.audio_original_adj = audio_adj.cuda()

		self.fusion_adj = self.harmonic_pool_fusion()

		self.softmax = nn.Softmax(dim=-1)

		self.query_v = nn.Sequential(
			nn.Linear(args.latdim, args.latdim),
			nn.Tanh(),
			nn.Linear(args.latdim, args.latdim, bias=False)
		)
		self.query_t = nn.Sequential(
			nn.Linear(args.latdim, args.latdim),
			nn.Tanh(),
			nn.Linear(args.latdim, args.latdim, bias=False)
		)
		if audio_embedding != None:
			self.query_a = nn.Sequential(
				nn.Linear(args.latdim, args.latdim),
				nn.Tanh(),
				nn.Linear(args.latdim, args.latdim, bias=False)
			)

		self.gate_v = nn.Sequential(
			nn.Linear(args.latdim, args.latdim),
			nn.Sigmoid()
		)
		self.gate_t = nn.Sequential(
			nn.Linear(args.latdim, args.latdim),
			nn.Sigmoid()
		)
		self.gate_f = nn.Sequential(
			nn.Linear(args.latdim, args.latdim),
			nn.Sigmoid()
		)
		if audio_embedding != None:
			self.gate_a = nn.Sequential(
				nn.Linear(args.latdim, args.latdim),
				nn.Sigmoid()
			)

		self.gate_image_prefer = nn.Sequential(
			nn.Linear(args.latdim, args.latdim),
			nn.Sigmoid()
		)
		self.gate_text_prefer = nn.Sequential(
			nn.Linear(args.latdim, args.latdim),
			nn.Sigmoid()
		)
		self.gate_fusion_prefer = nn.Sequential(
			nn.Linear(args.latdim, args.latdim),
			nn.Sigmoid()
		)
		if audio_embedding != None:
			self.gate_audio_prefer = nn.Sequential(
				nn.Linear(args.latdim, args.latdim),
				nn.Sigmoid()
			)

		self.modality_diff_weight = nn.Parameter(torch.ones(args.user+args.item))
		self.sigma_diff = nn.Parameter(torch.tensor(1.0))

	def getItemEmbeds(self):
		return self.iEmbeds

	def getUserEmbeds(self):
		return self.uEmbeds

	def getImageFeats(self):
		if args.trans == 0 or args.trans == 2:
			image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
			return image_feats
		else:
			image_feats = self.image_trans(self.image_embedding)
			return self.dropout2(image_feats)

	def getTextFeats(self):
		if args.trans == 0:
			text_feats = self.leakyrelu(torch.mm(self.text_embedding, self.text_trans))
			return text_feats
		else:
			text_feats = self.text_trans(self.text_embedding)
			return self.dropout2(text_feats)

	def getAudioFeats(self):
		if self.audio_embedding == None:
			return None
		else:
			if args.trans == 0:
				audio_feats = self.leakyrelu(torch.mm(self.audio_embedding, self.audio_trans))
			else:
				audio_feats = self.audio_trans(self.audio_embedding)
		return self.dropout2(audio_feats)

	def compute_modality_difference_loss(self, visual_emb, text_emb):
		diff = torch.abs(visual_emb - text_emb)
		loss = torch.norm(diff * self.modality_diff_weight.unsqueeze(1), p=2)
		return loss

	def compute_modality_difference_loss1(self, visual_emb, text_emb, visual_p, text_p):
		diff = torch.abs(visual_emb - text_emb)
		modality_preference_diff = torch.abs(visual_p - text_p)
		# loss = self.sigma_diff * torch.norm(diff * modality_preference_diff.unsqueeze(1), p=2)
		loss = torch.norm(diff * modality_preference_diff.unsqueeze(1), p=2)
		return loss

	def compute_modality_difference_loss2(self, feat1, feat2, weight=None):
		diff = torch.abs(feat1 - feat2)
		loss = torch.norm(diff, p=2)
		if weight is not None:
			loss *= weight
		return loss

	def harmonic_pool_fusion(self):
		def process_adj(adj):
			adj = adj.coalesce()
			indices = adj.indices().t()  # [num_edges, 2]
			values = adj.values()  # [num_edges]
			return indices, values

		modality_adjs = []
		for adj in [self.image_original_adj, self.text_original_adj]:
			indices, values = process_adj(adj)
			modality_adjs.append((indices, values))

		all_indices = torch.cat([m[0] for m in modality_adjs], dim=0)  # [total_edges, 2]
		all_values = torch.cat([m[1] for m in modality_adjs], dim=0)  # [total_edges]

		index_tuples = [f"{i.item()}_{j.item()}" for i, j in all_indices]
		from collections import defaultdict
		grouped = defaultdict(list)
		for idx, key in enumerate(index_tuples):
			grouped[key].append(all_values[idx].item())

		final_indices = []
		final_values = []
		for key, vals in grouped.items():
			i, j = map(int, key.split('_'))
			vals_tensor = torch.tensor(vals, device=all_values.device)
			# Harmonic mean
			geo_mean = len(vals_tensor) / torch.sum(1.0 / (vals_tensor + 1e-8))
			final_indices.append([i, j])
			final_values.append(geo_mean)

		final_indices = torch.tensor(final_indices, dtype=torch.long, device=all_values.device).t()  # [2, nnz]
		final_values = torch.stack(final_values)  # [nnz]

		fusion_adj = torch.sparse_coo_tensor(final_indices, final_values,
											 self.image_original_adj.size()).coalesce()
		return fusion_adj


	def forward1(self, handler):
		# modal_feats n*64
		image_feats = F.normalize(self.getImageFeats()) # (n, d)
		text_feats = F.normalize(self.getTextFeats()) # (n, d)

		image_user_feats = torch.sparse.mm(handler.torchTrnMat, image_feats) * handler.num_inters[:args.user].unsqueeze(1) # (m, d)
		text_user_feats = torch.sparse.mm(handler.torchTrnMat, text_feats) * handler.num_inters[:args.user].unsqueeze(1) # (m, d)

		image_embeds = torch.concat([image_user_feats, image_feats], dim=0) # (m+n, d)
		text_embeds = torch.concat([text_user_feats, text_feats], dim=0) # (m+n, d)

		if args.data == 'tiktok':
			audio_feats = F.normalize(self.getAudioFeats())
			audio_user_feats = torch.sparse.mm(handler.torchTrnMat, audio_feats) * handler.num_inters[
																				   :args.user].unsqueeze(1)
			audio_embeds = torch.concat([audio_user_feats, audio_feats], dim=0)
		"""
		weight = self.modal_softmax1(self.modal_weight1)
		if args.data == 'tiktok':
			fusion_embeds = weight[0] * image_embeds + weight[1] * text_embeds + weight[2] * audio_embeds  # (m+n, d)
			side_embeds = torch.mean(torch.stack([image_embeds, text_embeds, audio_embeds, fusion_embeds]), dim=0)  # (m+n, d)
		else:
			fusion_embeds = weight[0] * image_embeds + weight[1] * text_embeds # (m+n, d)
			side_embeds = torch.mean(torch.stack([image_embeds, text_embeds, fusion_embeds]), dim=0) # (m+n, d)
		side_embeds_users, side_embeds_items = torch.split(side_embeds, [args.user, args.item], dim=0)
		"""

		iEmbeds = F.normalize(self.getItemEmbeds())
		uEmbeds = F.normalize(self.getUserEmbeds())

		#  User-Item (Behavioral) View
		item_embeds = iEmbeds
		user_embeds = uEmbeds
		ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
		all_embeddings = [ego_embeddings]
		# res_embeddings = all_embeddings[0]

		for i in range(args.behavior_layers):
			side_embeddings = torch.sparse.mm(handler.torchBiAdj, ego_embeddings)
			ego_embeddings = side_embeddings
			all_embeddings += [ego_embeddings]
		all_embeddings = torch.stack(all_embeddings, dim=1)
		all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
		content_embeds = all_embeddings # + res_embeddings

		# content_embeds_users, content_embeds_items = torch.split(content_embeds, [args.user, args.item], dim=0)

		if args.data == 'tiktok':
			return content_embeds, image_embeds, text_embeds, audio_embeds
		else:
			return content_embeds, image_embeds, text_embeds

	def forward2(self, handler, content_embeds, diff_imageEmbeds, diff_textEmbeds, diff_audioEmbeds=None):
		image_user_conv, image_conv = torch.split(diff_imageEmbeds, [args.user, args.item], dim=0)
		text_user_conv, text_conv = torch.split(diff_textEmbeds, [args.user, args.item], dim=0)

		#   Item-Item Modality Specific and Fusion views
		weight = self.modal_softmax(self.modal_weight)

		iEmbeds = F.normalize(self.getItemEmbeds())
		uEmbeds = F.normalize(self.getUserEmbeds())

		if args.data == 'tiktok':
			audio_user_conv, audio_conv = torch.split(diff_audioEmbeds, [args.user, args.item], dim=0)
			fusion_conv = weight[0] * image_conv + weight[1] * text_conv + weight[2] * audio_conv
			audio_item_embeds = iEmbeds * self.gate_t(audio_conv)
			image_item_embeds = iEmbeds * self.gate_v(image_conv)
			text_item_embeds = iEmbeds * self.gate_t(text_conv)
			fusion_item_embeds = iEmbeds * self.gate_f(fusion_conv)
		else:
			fusion_conv = weight[0] * image_conv + weight[1] * text_conv
			image_item_embeds = iEmbeds * self.gate_v(image_conv)
			text_item_embeds = iEmbeds * self.gate_t(text_conv)
			fusion_item_embeds = iEmbeds * self.gate_f(fusion_conv)

		#   Image-view
		for i in range(args.mm_layers):
			image_item_embeds = torch.sparse.mm(self.image_original_adj, image_item_embeds)
		image_user_embeds = torch.sparse.mm(handler.torchTrnMat, image_item_embeds)
		image_embeds = torch.cat([image_user_embeds, image_item_embeds], dim=0)

		#   Text-view
		for i in range(args.mm_layers):
			text_item_embeds = torch.sparse.mm(self.text_original_adj, text_item_embeds)
		text_user_embeds = torch.sparse.mm(handler.torchTrnMat, text_item_embeds)
		text_embeds = torch.cat([text_user_embeds, text_item_embeds], dim=0)

		if args.data == 'tiktok':
			for i in range(args.mm_layers):
				audio_item_embeds = torch.sparse.mm(self.audio_original_adj, audio_item_embeds)
			audio_user_embeds = torch.sparse.mm(handler.torchTrnMat, audio_item_embeds)
			audio_embeds = torch.cat([audio_user_embeds, audio_item_embeds], dim=0)

		for i in range(args.mm_layers):
			fusion_item_embeds = torch.sparse.mm(self.fusion_adj, fusion_item_embeds)
		fusion_user_embeds = torch.sparse.mm(handler.torchTrnMat, fusion_item_embeds)
		fusion_embeds = torch.cat([fusion_user_embeds, fusion_item_embeds], dim=0)

		#   Modality-aware Preference Module
		fusion_att_v, fusion_att_t = self.query_v(image_embeds), self.query_t(text_embeds)
		fusion_soft_v = self.softmax(fusion_att_v)
		agg_image_embeds = fusion_soft_v * image_embeds

		fusion_soft_t = self.softmax(fusion_att_t)
		agg_text_embeds = fusion_soft_t * text_embeds

		if args.data == 'tiktok':
			fusion_att_a = self.query_a(fusion_embeds)
			fusion_soft_a = self.softmax(fusion_att_a)
			agg_audio_embeds = fusion_soft_a * audio_embeds

		image_prefer = self.gate_image_prefer(content_embeds)
		text_prefer = self.gate_text_prefer(content_embeds)
		fusion_prefer = self.gate_fusion_prefer(content_embeds)
		image_prefer, text_prefer, fusion_prefer = self.dropout1(image_prefer), self.dropout1(
			text_prefer), self.dropout1(
			fusion_prefer)

		if args.data == 'tiktok':
			audio_prefer = self.gate_audio_prefer(content_embeds)
			audio_prefer = self.dropout1(audio_prefer)

		if args.data == 'tiktok':
			differ_loss = self.compute_modality_difference_loss(agg_image_embeds, agg_text_embeds) + \
						  self.compute_modality_difference_loss(agg_image_embeds, agg_audio_embeds) + \
						  self.compute_modality_difference_loss(agg_text_embeds, agg_audio_embeds)
		else:
			differ_loss = self.compute_modality_difference_loss(agg_image_embeds, agg_text_embeds)

		agg_image_embeds = torch.multiply(image_prefer, agg_image_embeds)
		agg_text_embeds = torch.multiply(text_prefer, agg_text_embeds)
		fusion_embeds = torch.multiply(fusion_prefer, fusion_embeds)

		if args.data == 'tiktok':
			agg_audio_embeds = torch.multiply(audio_prefer, agg_audio_embeds)
			side_embeds = torch.mean(torch.stack([agg_image_embeds, agg_text_embeds, agg_audio_embeds, fusion_embeds]),
									 dim=0)

			# differ_loss = self.compute_modality_difference_loss(agg_image_embeds, agg_text_embeds) + \
			# 			  self.compute_modality_difference_loss(agg_image_embeds, agg_audio_embeds) + \
			# 			  self.compute_modality_difference_loss(agg_text_embeds, agg_audio_embeds)
		else:
			side_embeds = torch.mean(torch.stack([agg_image_embeds, agg_text_embeds, fusion_embeds]), dim=0)

			# differ_loss = self.compute_modality_difference_loss(agg_image_embeds, agg_text_embeds)

		all_embeds = args.alpha*content_embeds + side_embeds
		# all_embeddings_users, all_embeddings_items = torch.split(all_embeds, [args.user, args.item], dim=0)

		# content_embeds_users, content_embeds_items = torch.split(content_embeds, [args.user, args.item], dim=0)
		# side_embeds_users, side_embeds_items = torch.split(side_embeds, [args.user, args.item], dim=0)

		return all_embeds, differ_loss

	def cal_loss(self, handler, ancs, poss):
		if args.data == 'tiktok':
			content_embeds, image_embeds, text_embeds, audio_embeds = self.forward1(handler)

			v_diff_loss, diff_imageEmbeds = self.diffusion_model.training_losses2(self.image_reverse_model,
			                                                                    image_embeds, content_embeds, ancs, poss)
			t_diff_loss, diff_textEmbeds = self.diffusion_model.training_losses2(self.text_reverse_model,
			                                                                     text_embeds, content_embeds, ancs, poss)
			a_diff_loss, diff_audioEmbeds = self.diffusion_model.training_losses2(self.audio_reverse_model,
																				  audio_embeds, content_embeds, ancs, poss)

			diff_loss = (v_diff_loss.mean() + t_diff_loss.mean() + a_diff_loss.mean()) / 3.0

			all_embeds, differ_loss = self.forward2(handler, content_embeds, diff_imageEmbeds, diff_textEmbeds, diff_audioEmbeds)

			differ_loss = differ_loss.mean()
			all_embeds = all_embeds + args.denoise_embeds * (F.normalize(diff_imageEmbeds) + F.normalize(diff_textEmbeds) + F.normalize(diff_audioEmbeds))
			# usrEmbeds = all_embeddings_users + args.denoise_embeds * F.normalize(diff_usrEmbeds)
			# itmEmbeds = all_embeddings_items + args.denoise_embeds * F.normalize(diff_itemEmbeds)
			return all_embeds, diff_loss, differ_loss, v_diff_loss, t_diff_loss, a_diff_loss
		else:
			content_embeds, image_embeds, text_embeds = self.forward1(handler)

			v_diff_loss, diff_imageEmbeds = self.diffusion_model.training_losses2(self.image_reverse_model,
																				  image_embeds, content_embeds, ancs,
																				  poss)
			t_diff_loss, diff_textEmbeds = self.diffusion_model.training_losses2(self.text_reverse_model,
																				 text_embeds, content_embeds, ancs,
																				 poss)

			diff_loss = (v_diff_loss.mean() + t_diff_loss.mean()) / 2.0

			all_embeds, differ_loss = self.forward2(handler, content_embeds, diff_imageEmbeds, diff_textEmbeds)

			differ_loss = differ_loss.mean()
			all_embeds = all_embeds + args.denoise_embeds * (F.normalize(diff_imageEmbeds) + F.normalize(diff_textEmbeds))
			return all_embeds, diff_loss, differ_loss, v_diff_loss, t_diff_loss, diff_imageEmbeds, diff_textEmbeds

	def forward_cl_MM(self, handler):
		adj = handler.torchBiAdj

		if args.data == 'tiktok':
			audio_adj = handler.torchBiAdj
		image_adj = handler.torchBiAdj
		text_adj = handler.torchBiAdj

		if args.trans == 0:
			image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
			text_feats = self.leakyrelu(torch.mm(self.text_embedding, self.text_trans))
		elif args.trans == 1:
			image_feats = self.image_trans(self.image_embedding)
			text_feats = self.text_trans(self.text_embedding)
		else:
			image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
			text_feats = self.text_trans(self.text_embedding)

		if args.data == 'tiktok':
			if args.trans == 0:
				audio_feats = self.leakyrelu(torch.mm(self.audio_embedding, self.audio_trans))
			else:
				audio_feats = self.audio_trans(self.audio_embedding)

		embedsImage = torch.concat([self.uEmbeds, F.normalize(image_feats)])
		embedsImage = torch.spmm(image_adj, embedsImage)

		embedsText = torch.concat([self.uEmbeds, F.normalize(text_feats)])
		embedsText = torch.spmm(text_adj, embedsText)

		if args.data == 'tiktok':
			embedsAudio = torch.concat([self.uEmbeds, F.normalize(audio_feats)])
			embedsAudio = torch.spmm(audio_adj, embedsAudio)

		embeds1 = embedsImage
		embedsLst1 = [embeds1]
		for gcn in self.gcnLayers:
			embeds1 = gcn(adj, embedsLst1[-1])
			embedsLst1.append(embeds1)
		embeds1 = sum(embedsLst1)

		embeds2 = embedsText
		embedsLst2 = [embeds2]
		for gcn in self.gcnLayers:
			embeds2 = gcn(adj, embedsLst2[-1])
			embedsLst2.append(embeds2)
		embeds2 = sum(embedsLst2)

		if args.data == 'tiktok':
			embeds3 = embedsAudio
			embedsLst3 = [embeds3]
			for gcn in self.gcnLayers:
				embeds3 = gcn(adj, embedsLst3[-1])
				embedsLst3.append(embeds3)
			embeds3 = sum(embedsLst3)

		if args.data == 'tiktok':
			return embeds1[:args.user], embeds1[args.user:], embeds2[:args.user], embeds2[args.user:], embeds3[:args.user], embeds3[args.user:]
		else:
			return embeds1[:args.user], embeds1[args.user:], embeds2[:args.user], embeds2[args.user:]

	def predict(self, handler):
		if args.data == 'tiktok':
			content_embeds, image_embeds, text_embeds, audio_embeds = self.forward1(handler)

			denoised_v = self.diffusion_model.p_sample(self.image_reverse_model, image_embeds, content_embeds,
			                                           args.sampling_steps, args.sampling_noise)
			denoised_t = self.diffusion_model.p_sample(self.text_reverse_model, text_embeds, content_embeds,
			                                             args.sampling_steps, args.sampling_noise)
			denoised_a = self.diffusion_model.p_sample(self.audio_reverse_model, audio_embeds, content_embeds,
													   args.sampling_steps, args.sampling_noise)
		else:
			content_embeds, image_embeds, text_embeds = self.forward1(handler)

			denoised_v = self.diffusion_model.p_sample(self.image_reverse_model, image_embeds, content_embeds,
													   args.sampling_steps, args.sampling_noise)
			denoised_t = self.diffusion_model.p_sample(self.text_reverse_model, text_embeds, content_embeds,
													   args.sampling_steps, args.sampling_noise)

		if args.data == 'tiktok':
			all_embeds, differ_loss = self.forward2(handler, content_embeds, denoised_v, denoised_t, denoised_a)
			all_embeds = all_embeds + args.denoise_embeds * (denoised_v + denoised_t + denoised_a)
			usrEmbeds, itmEmbeds = torch.split(all_embeds, [args.user, args.item], dim=0)
		else:
			all_embeds, differ_loss = self.forward2(handler, content_embeds, denoised_v, denoised_t)
			all_embeds = all_embeds + args.denoise_embeds * (denoised_v + denoised_t)
			usrEmbeds, itmEmbeds = torch.split(all_embeds, [args.user, args.item], dim=0)

		return usrEmbeds, itmEmbeds

	def reg_loss(self):
		ret = 0
		ret += self.uEmbeds.norm(2).square()
		ret += self.iEmbeds.norm(2).square()
		return ret

class GCNLayer(nn.Module):
	def __init__(self):
		super(GCNLayer, self).__init__()

	def forward(self, adj, embeds):
		return torch.spmm(adj, embeds)

class SpAdjDropEdge(nn.Module):
	def __init__(self, keepRate):
		super(SpAdjDropEdge, self).__init__()
		self.keepRate = keepRate

	def forward(self, adj):
		vals = adj._values()
		idxs = adj._indices()
		edgeNum = vals.size()
		mask = ((torch.rand(edgeNum) + self.keepRate).floor()).type(torch.bool)

		newVals = vals[mask] / self.keepRate
		newIdxs = idxs[:, mask]

		return torch.sparse.FloatTensor(newIdxs, newVals, adj.shape)

class Denoise(nn.Module):
	def __init__(self, in_dims, out_dims, emb_size, norm=False, dropout=0.5):
		super(Denoise, self).__init__()
		self.in_dims = in_dims
		self.out_dims = out_dims
		self.time_emb_dim = emb_size
		self.norm = norm

		self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

		in_dims_temp = [self.in_dims[0] * 2 + self.time_emb_dim] + self.in_dims[1:]
		# in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
		out_dims_temp = self.out_dims
		#
		self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out)
		                                for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
		self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out)
		                                 for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])

		self.drop = nn.Dropout(dropout)
		self.init_weights()

	def init_weights(self):
		for layer in self.in_layers:
			size = layer.weight.size()
			std = np.sqrt(2.0 / (size[0] + size[1]))
			layer.weight.data.normal_(0.0, std)
			layer.bias.data.normal_(0.0, 0.001)

		for layer in self.out_layers:
			size = layer.weight.size()
			std = np.sqrt(2.0 / (size[0] + size[1]))
			layer.weight.data.normal_(0.0, std)
			layer.bias.data.normal_(0.0, 0.001)

		size = self.emb_layer.weight.size()
		std = np.sqrt(2.0 / (size[0] + size[1]))
		self.emb_layer.weight.data.normal_(0.0, std)
		self.emb_layer.bias.data.normal_(0.0, 0.001)

	def forward(self, x, gcnembs, timesteps, mess_dropout=True):
		freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.time_emb_dim//2, dtype=torch.float32) / (self.time_emb_dim//2)).cuda()
		temp = timesteps[:, None].float() * freqs[None]
		time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)
		if self.time_emb_dim % 2:
			time_emb = torch.cat([time_emb, torch.zeros_like(time_emb[:, :1])], dim=-1)
		emb = self.emb_layer(time_emb)

		if self.norm:
			x = F.normalize(x)
			gcnembs = F.normalize(gcnembs)
		if mess_dropout:
			x = self.drop(x)

		h = torch.cat([x, emb, gcnembs], dim=-1)
		# h = torch.cat([x, emb], dim=-1)
		for i, layer in enumerate(self.in_layers):
			h = layer(h)
			h = torch.tanh(h)
		for i, layer in enumerate(self.out_layers):
			h = layer(h)
			if i != len(self.out_layers) - 1:
				h = torch.tanh(h)

		return h

class GaussianDiffusion(nn.Module):
	def __init__(self, noise_scale, noise_min, noise_max, steps, beta_fixed=True):
		super(GaussianDiffusion, self).__init__()

		self.noise_scale = noise_scale # noise_scale=0.1
		self.noise_min = noise_min # noise_min=0.0001
		self.noise_max = noise_max # noise_max=0.2
		self.steps = steps # steps=5

		if noise_scale != 0:
			self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).cuda()
			if beta_fixed:
				self.betas[0] = 0.0001

			self.calculate_for_diffusion()

	def get_betas(self):
		start = self.noise_scale * self.noise_min
		end = self.noise_scale * self.noise_max
		variance = np.linspace(start, end, self.steps, dtype=np.float64)
		alpha_bar = 1 - variance
		betas = []
		betas.append(1 - alpha_bar[0])
		for i in range(1, self.steps):
			betas.append(min(1 - alpha_bar[i] / alpha_bar[i-1], 0.999))
		return np.array(betas)

	def calculate_for_diffusion(self):
		alphas = 1.0 - self.betas
		self.alphas_cumprod = torch.cumprod(alphas, axis=0).cuda()
		self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).cuda(), self.alphas_cumprod[:-1]]).cuda()
		self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]).cuda()]).cuda()

		self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
		self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
		self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
		self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
		self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

		self.posterior_variance = (
			self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
		)
		self.posterior_log_variance_clipped = torch.log(torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]]))
		self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
		self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod))

	def p_sample(self, model, x_start, m_gcn_embs, steps=0, sampling_noise=False):
		if steps == 0:
			x_t = x_start
		else:
			t = torch.tensor([steps - 1] * x_start.shape[0]).cuda()
			x_t = self.q_sample(x_start, t)

		indices = list(range(self.steps))[::-1]

		for i in indices:
			t = torch.tensor([i] * x_t.shape[0]).cuda()
			model_mean, model_log_variance = self.p_mean_variance(model, x_t, m_gcn_embs, t)
			if sampling_noise:
				noise = torch.randn_like(x_t)
				nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
				x_t = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
			else:
				x_t = model_mean
		return x_t

	def q_sample(self, x_start, t, noise=None):
		if noise is None:
			noise = torch.randn_like(x_start)
		return self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

	def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
		arr = arr.cuda()
		res = arr[timesteps].float()
		while len(res.shape) < len(broadcast_shape):
			res = res[..., None]
		return res.expand(broadcast_shape)

	def p_mean_variance(self, model, x, m_gcn_embs, t):
		model_output = model(x, m_gcn_embs, t, False)

		model_variance = self.posterior_variance
		model_log_variance = self.posterior_log_variance_clipped

		model_variance = self._extract_into_tensor(model_variance, t, x.shape)
		model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)

		model_mean = (self._extract_into_tensor(self.posterior_mean_coef1, t,
		                                        x.shape) * model_output + self._extract_into_tensor(
			self.posterior_mean_coef2, t, x.shape) * x)

		return model_mean, model_log_variance

	def process_user_embeddings(self, embeddings, bev_users, model, noise_scale, steps):
		user_size = embeddings.size(0)
		ts = torch.randint(0, steps, (user_size,)).long().cuda()
		noise = torch.randn_like(embeddings)
		x_t = self.q_sample(embeddings, ts, noise) if noise_scale != 0 else embeddings
		return model(x_t, bev_users, ts)

	def compute_loss(self, original, output, loss_type, alpha):
		if loss_type == 'mse':
			return self.mean_flat((original - output) ** alpha)
		else:
			return self.mean_flat(self.sce_criterion(original, output, alpha))

	def training_losses2(self, model, x_start, targetEmbeds, ancs, poss):
		batch_size = x_start.size(0)
		device = x_start.device
		# ts, pt = self.sample_timesteps(batch_size, device,'importance')
		ts = torch.randint(0, self.steps, (batch_size,)).long().to(device)  # time_step
		noise = torch.randn_like(x_start)
		if self.noise_scale != 0:
			x_t = self.q_sample(x_start, ts, noise)
		else:
			x_t = x_start

		model_output = model(x_t, targetEmbeds, ts) #
		mse = self.mean_flat((targetEmbeds - model_output) ** 2)
		# mse = self.compute_loss(targetEmbeds, model_output, args.alpha_l)
		# mse = self.compute_loss(x_start, model_output, args.mse, args.alpha_l)
		# mse = cal_infonce_loss(targetEmbeds,model_output,args.temp)
		weight = self.SNR(ts - 1) - self.SNR(ts)
		weight = torch.where((ts == 0), 1.0, weight)
		diff_loss = weight * mse
		# diff_loss = diff_loss[batch]
		diff_loss_ancs = diff_loss[ancs]
		diff_loss_poss = diff_loss[poss]
		diff_loss_selected = torch.cat([diff_loss_ancs, diff_loss_poss], dim=0)  # 形状 (2 * batch_size,)
		return diff_loss, model_output

	def mean_flat(self, tensor):
		return tensor.mean(dim=list(range(1, len(tensor.shape))))

	def sce_criterion(self, x, y, alpha=1):
		x = F.normalize(x, p=2, dim=-1)
		y = F.normalize(y, p=2, dim=-1)

		loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
		# loss = loss.mean()

		return loss

	def SNR(self, t):
		self.alphas_cumprod = self.alphas_cumprod.cuda()
		return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])

class HGNNLayer(nn.Module):
	def __init__(self, n_hyper_layer):
		super(HGNNLayer, self).__init__()

		self.h_layer = n_hyper_layer

	def forward(self, i_hyper, u_hyper, embeds):
		i_ret = embeds
		for _ in range(self.h_layer):
			lat = torch.mm(i_hyper.T, i_ret)  # (n*h).T * n*d
			i_ret = torch.mm(i_hyper, lat)  # n*h * h*d
			u_ret = torch.mm(u_hyper, lat)  # m*h * h*d
		return u_ret, i_ret
