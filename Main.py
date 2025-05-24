import torch
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Params import args
from Model import Model, GaussianDiffusion, Denoise
from DataHandler import DataHandler
import numpy as np
from Utils.Utils import *
import os
import scipy.sparse as sp
import random
import setproctitle
from scipy.sparse import coo_matrix

class Coach:
	def __init__(self, handler):
		self.handler = handler

		print('USER', args.user, 'ITEM', args.item)
		print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
		self.metrics = dict()
		mets = ['Loss', 'preLoss', 'Recall', 'NDCG']
		for met in mets:
			self.metrics['Train' + met] = list()
			self.metrics['Test' + met] = list()
			self.metrics['Eval' + met] = list()

	def makePrint(self, name, ep, reses, save):
		ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
		for metric in reses:
			val = reses[metric]
			ret += '%s = %.4f, ' % (metric, val)
			tem = name + metric
			if save and tem in self.metrics:
				self.metrics[tem].append(val)
		ret = ret[:-2] + '  '
		return ret

	def run(self):
		self.prepareModel()
		log('Model Prepared')

		recallMax = {k: 0 for k in args.topk}
		ndcgMax = {k: 0 for k in args.topk}
		precisionMax = {k: 0 for k in args.topk}
		tstRecallMax = {k: 0 for k in args.topk}
		tstNDCGMax = {k: 0 for k in args.topk}
		tstPrecisionMax = {k: 0 for k in args.topk}
		bestEpoch = 0
		patience = 0

		log('Model Initialized')

		for ep in range(0, args.epoch):
			tstFlag = (ep % args.tstEpoch == 0)
			reses = self.trainEpoch(ep)
			log(self.makePrint('Train', ep, reses, tstFlag))
			if tstFlag:
				resesVal = self.evalEpoch()
				if (resesVal['Recall@20'] >= recallMax[20] - 1e-6):
					patience = 0
					log(self.makePrint('Eval', ep, resesVal, tstFlag))

					resesTst = self.testEpoch()
					for k in args.topk:
						recallMax[k] = resesVal[f"Recall@{k}"]
						ndcgMax[k] = resesVal[f"NDCG@{k}"]
						precisionMax[k] = resesVal[f"Precision@{k}"]
						tstRecallMax[k] = resesTst[f"Recall@{k}"]
						tstNDCGMax[k] = resesTst[f"NDCG@{k}"]
						tstPrecisionMax[k] = resesTst[f"Precision@{k}"]
					bestEpoch = ep
					log(self.makePrint('Test', ep, resesTst, tstFlag))
				else:
					patience += 1
					log(self.makePrint('Eval', ep, resesVal, tstFlag))

					resesTst = self.testEpoch()
					log(self.makePrint('Test', ep, resesTst, tstFlag))

				if patience >= args.earlystopping:
					# log(f"Best epoch : {bestEpoch}, Recall : {tstRecallMax}, NDCG : {tstNDCGMax}, Precision : {tstPrecisionMax}")
					break
			print()

		print(f"Best epoch : {bestEpoch} | " + " | ".join(
			[f"Recall@{k}: {tstRecallMax[k]:.4f}, NDCG@{k}: {tstNDCGMax[k]:.4f}, Precision@{k}: {tstPrecisionMax[k]:.4f}" for k
			 in args.topk]
		))

	def prepareModel(self):
		if args.data == 'tiktok':
			self.model = Model(self.handler.image_feats, self.handler.text_feats, self.handler.audio_feats).cuda()
		else:
			self.model = Model(self.handler.image_feats, self.handler.text_feats).cuda()
		self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

	def normalizeAdj(self, mat): 
		degree = np.array(mat.sum(axis=-1))
		dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
		dInvSqrt[np.isinf(dInvSqrt)] = 0.0
		dInvSqrtMat = sp.diags(dInvSqrt)
		return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

	def buildUIMatrix(self, u_list, i_list, edge_list):
		mat = coo_matrix((edge_list, (u_list, i_list)), shape=(args.user, args.item), dtype=np.float32)

		a = sp.csr_matrix((args.user, args.user))
		b = sp.csr_matrix((args.item, args.item))
		mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
		mat = (mat != 0) * 1.0
		mat = (mat + sp.eye(mat.shape[0])) * 1.0
		mat = self.normalizeAdj(mat)

		idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = torch.from_numpy(mat.data.astype(np.float32))
		shape = torch.Size(mat.shape)

		return torch.sparse.FloatTensor(idxs, vals, shape).cuda()

	def trainEpoch(self,ep):
		trnLoader = self.handler.trnLoader
		trnLoader.dataset.negSampling()
		epLoss, epRecLoss, epClLoss, epDiffLoss, epDifferLoss, epRegLoss = 0, 0, 0, 0, 0, 0
		epImageLoss , epTextLoss, epAudioLoss = 0, 0, 0
		epDiLoss = 0.0
		epUser_mse, epItem_mse = 0.0, 0.0
		epDiLoss_image, epDiLoss_text = 0, 0
		loss = 0.0
		if args.data == 'tiktok':
			epDiLoss_audio = 0
		steps = trnLoader.dataset.__len__() // args.batch

		# for i, batch in enumerate(diffusionLoader):
		for i, tem in enumerate(trnLoader):
			ancs, poss, negs = tem
			ancs = ancs.long().cuda()
			poss = poss.long().cuda()
			negs = negs.long().cuda()

			self.opt.zero_grad()
			# def training_losses(self, user_model, item_model, uEmbeds, itmEmbeds, batch_index, R, A, v_item_feats)

			if args.data == 'tiktok':
				all_embeds, diff_loss, differ_loss, v_diff_loss, t_diff_loss, a_diff_loss = self.model.cal_loss(self.handler, ancs, poss)
				a_diff_loss = a_diff_loss.mean()
				epAudioLoss += a_diff_loss.item()
			else:
				all_embeds, diff_loss, differ_loss, v_diff_loss, t_diff_loss, diff_imageEmbeds, diff_textEmbeds = self.model.cal_loss(self.handler, ancs, poss)
			v_diff_loss = v_diff_loss.mean()
			t_diff_loss = t_diff_loss.mean()
			epImageLoss += v_diff_loss.item()
			epTextLoss += t_diff_loss.item()

			usrEmbeds, itmEmbeds = torch.split(all_embeds, [args.user, args.item], dim=0)
			ancEmbeds = usrEmbeds[ancs]
			posEmbeds = itmEmbeds[poss]
			negEmbeds = itmEmbeds[negs]

			scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
			bprLoss = - (scoreDiff).sigmoid().log().sum() / args.batch
			regLoss = ((torch.norm(ancEmbeds) ** 2 + torch.norm(posEmbeds) ** 2 + torch.norm(
				negEmbeds) ** 2) * args.reg) / args.batch

			# loss_image = diff_loss_image.mean() + gc_loss_image.mean() * args.e_loss
			# loss = diff_loss.mean() # + gc_loss_text.mean() * args.e_loss

			# user_mse = user_mse.mean()
			# item_mse = item_mse.mean()
			# epUser_mse += user_mse.item()
			# epItem_mse += item_mse.item()

			if args.data == 'tiktok':
				usrEmbeds1, itmEmbeds1, usrEmbeds2, itmEmbeds2, usrEmbeds3, itmEmbeds3 = self.model.forward_cl_MM(self.handler)
			else:
				usrEmbeds1, itmEmbeds1, usrEmbeds2, itmEmbeds2 = self.model.forward_cl_MM(self.handler)
				# usrEmbeds1, itmEmbeds1 = torch.split(diff_imageEmbeds, [args.user, args.item], dim=0)
				# usrEmbeds2, itmEmbeds2 = torch.split(diff_textEmbeds, [args.user, args.item], dim=0)

			if args.data == 'tiktok':
				clLoss = (contrastLoss(usrEmbeds1, usrEmbeds2, ancs, args.temp) + contrastLoss(itmEmbeds1, itmEmbeds2, poss, args.temp)) * args.ssl_reg
				clLoss += (contrastLoss(usrEmbeds1, usrEmbeds3, ancs, args.temp) + contrastLoss(itmEmbeds1, itmEmbeds3, poss, args.temp)) * args.ssl_reg
				clLoss += (contrastLoss(usrEmbeds2, usrEmbeds3, ancs, args.temp) + contrastLoss(itmEmbeds2, itmEmbeds3, poss, args.temp)) * args.ssl_reg
			else:
				clLoss = (contrastLoss(usrEmbeds1, usrEmbeds2, ancs, args.temp) + contrastLoss(itmEmbeds1, itmEmbeds2, poss, args.temp)) * args.ssl_reg

			clLoss1 = (contrastLoss(usrEmbeds, usrEmbeds1, ancs, args.temp) + contrastLoss(itmEmbeds, itmEmbeds1, poss, args.temp)) * args.ssl_reg
			clLoss2 = (contrastLoss(usrEmbeds, usrEmbeds2, ancs, args.temp) + contrastLoss(itmEmbeds, itmEmbeds2, poss, args.temp)) * args.ssl_reg
			if args.data == 'tiktok':
				clLoss3 = (contrastLoss(usrEmbeds, usrEmbeds3, ancs, args.temp) + contrastLoss(itmEmbeds, itmEmbeds3, poss, args.temp)) * args.ssl_reg
				clLoss_ = clLoss1 + clLoss2 + clLoss3
			else:
				clLoss_ = clLoss1 + clLoss2

			if args.cl_method == 1:
				clLoss = clLoss_

			#loss = diff_loss.mean() + bprLoss + regLoss + clLoss + differ_loss.mean()
			#loss = diff_loss+ bprLoss + regLoss + clLoss + differ_loss

			lambda_diff = args.lambda_diff
			# lambda_differ = args.lambda_differ if ep < 5 else args.lambda_differ*5
			lambda_differ =  args.lambda_differ # args.ssl_reg # args.lambda_differ

			loss = lambda_diff * diff_loss.mean() + bprLoss + regLoss + clLoss + lambda_differ * differ_loss.mean()

			torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)

			epRecLoss += bprLoss.item()
			epLoss += loss.item()
			epClLoss += clLoss.item()
			epDiffLoss += diff_loss.mean().item()
			epDifferLoss += differ_loss.mean().item()
			epRegLoss += regLoss.item()
			loss.backward()
			self.opt.step()
			"""
			log('Step %d/%d: bpr : %.3f ; diff : %.3f ; differ : %.3f ; reg : %.3f ; cl : %.3f ' % (
				i,
				steps,
				bprLoss.item(),
				diff_loss.item(),
				differ_loss.item(),
				regLoss.item(),
				clLoss.item()
			), save=False, oneline=True)
			"""

		ret = dict()
		ret['Loss'] = epLoss / steps
		ret['BPR Loss'] = epRecLoss / steps
		ret['Diff Loss'] = epDiffLoss / steps
		ret['Image DiLoss'] = epImageLoss / steps
		ret['Text DiLoss'] = epTextLoss / steps
		if args.data == 'tiktok':
			ret['Audio DiLoss'] = epAudioLoss / steps
		ret['Differ Loss'] = epDifferLoss / steps
		ret['CL loss'] = epClLoss / steps
		ret['REG loss'] = epRegLoss / steps
		return ret

	def testEpoch(self):
		tstLoader = self.handler.tstLoader
		# epRecall, epNdcg, epPrecision = [0] * 3
		epRecall = {k: 0 for k in args.topk}
		epNdcg = {k: 0 for k in args.topk}
		epPrecision = {k: 0 for k in args.topk}
		i = 0
		num = tstLoader.dataset.__len__()
		steps = num // args.tstBat
		self.model.eval()

		with torch.no_grad():
			usrEmbeds, itmEmbeds = self.model.predict(self.handler)

		for usr, trnMask in tstLoader:
			i += 1
			usr = usr.long().cuda()
			trnMask = trnMask.cuda()
			allPreds = torch.mm(usrEmbeds[usr], torch.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
			_, topLocs = torch.topk(allPreds, max(args.topk))
			recallDict, ndcgDict, precisionDict = self.calcRes(topLocs.cpu().numpy(),
			                                                   self.handler.tstLoader.dataset.tstLocs, usr)
			for k in args.topk:
				epRecall[k] += recallDict[k]
				epNdcg[k] += ndcgDict[k]
				epPrecision[k] += precisionDict[k]

		ret = {f"Recall@{k}": epRecall[k] / num for k in args.topk}
		ret.update({f"NDCG@{k}": epNdcg[k] / num for k in args.topk})
		ret.update({f"Precision@{k}": epPrecision[k] / num for k in args.topk})
		return ret

	def evalEpoch(self):
		valLoader = self.handler.valLoader
		# epRecall, epNdcg, epPrecision = [0] * 3
		epRecall = {k: 0 for k in args.topk}
		epNdcg = {k: 0 for k in args.topk}
		epPrecision = {k: 0 for k in args.topk}
		i = 0
		num = valLoader.dataset.__len__()
		steps = num // args.valBat
		self.model.eval()

		with torch.no_grad():
			usrEmbeds, itmEmbeds = self.model.predict(self.handler)

		for usr, trnMask in valLoader:
			i += 1
			usr = usr.long().cuda()
			trnMask = trnMask.cuda()
			allPreds = torch.mm(usrEmbeds[usr], torch.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
			_, topLocs = torch.topk(allPreds, max(args.topk))
			recallDict, ndcgDict, precisionDict = self.calcRes(topLocs.cpu().numpy(),
			                                                   self.handler.valLoader.dataset.valLocs, usr)
			for k in args.topk:
				epRecall[k] += recallDict[k]
				epNdcg[k] += ndcgDict[k]
				epPrecision[k] += precisionDict[k]

		ret = {f"Recall@{k}": epRecall[k] / num for k in args.topk}
		ret.update({f"NDCG@{k}": epNdcg[k] / num for k in args.topk})
		ret.update({f"Precision@{k}": epPrecision[k] / num for k in args.topk})
		return ret

	def calcRes(self, topLocs, tstLocs, batIds):
		assert topLocs.shape[0] == len(batIds)

		allRecall = {k: 0 for k in args.topk}
		allNdcg = {k: 0 for k in args.topk}
		allPrecision = {k: 0 for k in args.topk}

		for i in range(len(batIds)):
			temTopLocs = list(topLocs[i])
			temTstLocs = tstLocs[batIds[i]]
			tstNum = len(temTstLocs)

			for k in args.topk:
				maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, k))])

				recall = dcg = precision = 0
				for val in temTstLocs:
					if val in temTopLocs[:k]:
						recall += 1
						dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))  # 计算 DCG
						precision += 1

				allRecall[k] += recall / tstNum
				allNdcg[k] += dcg / maxDcg
				allPrecision[k] += precision / k

		return allRecall, allNdcg, allPrecision


def seed_it(seed):
	random.seed(seed)
	os.environ["PYTHONSEED"] = str(seed)
	np.random.seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True 
	torch.backends.cudnn.enabled = True
	torch.manual_seed(seed)

if __name__ == '__main__':
	seed_it(args.seed)
	log('args:')
	log(args)

	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	logger.saveDefault = True
	
	log('Start')
	handler = DataHandler()
	handler.LoadData()
	log('Load Data')

	coach = Coach(handler)
	coach.run()