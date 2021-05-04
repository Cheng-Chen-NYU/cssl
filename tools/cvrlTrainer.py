import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import pandas as pd
import math

class simclrTrainer():
	def __init__(self, log_dir, model, train_loader, memory_loader, test_loader, optimizer, temperature, k):
		self.log_dir = log_dir
		self.model = model
		self.train_loader = train_loader
		self.memory_loader = memory_loader
		self.test_loader = test_loader
		self.optimizer = optimizer
		self.temperature = temperature
		self.k = k

	def train(self, resume, epoch_start, epochs):
		results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}
		model = nn.DataParallel(self.model, device_ids=[0, 1, 2, 3]).cuda()

		batch_size = self.train_loader.batch_size
		c = len(self.memory_loader.dataset.classes)

		best_acc = 0.0

		if resume != '':
			checkpoint = torch.load(resume)
			model.load_state_dict(checkpoint['state_dict'])
			self.optimizer.load_state_dict(checkpoint['optimizer'])
			epoch_start = checkpoint['epoch'] + 1
			print('Loaded from: {}'.format(resume))

		for epoch in range(epoch_start, epochs + 1):
			
			model.train()

			total_loss, total_num, train_bar = 0.0, 0, tqdm(self.train_loader)
			for pos_1, pos_2, target in train_bar:
				
				pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
				feature_1, out_1 = model(pos_1)
				feature_2, out_2 = model(pos_2)
				# [2*B, D]
				out = torch.cat([out_1, out_2], dim=0)
				# [2*B, 2*B]
				sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
				mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
				# [2*B, 2*B-1]
				sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

				# compute loss
				pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
				# [2*B]
				pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
				loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
				
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

				total_num += batch_size
				total_loss += loss.item() * batch_size
				train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

			results['train_loss'].append(total_loss / total_num)

			model.eval()

			total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
			with torch.no_grad():
				# generate feature bank
				for data, target in tqdm(self.memory_loader, desc='Feature extracting'):
					feature, out = model(data.cuda(non_blocking=True))
					feature_bank.append(feature)
				
				# [D, N]	
				feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
				# [N]
				feature_labels = torch.tensor(self.memory_loader.dataset.targets, device=feature_bank.device)
				
				# loop test data to predict the label by weighted knn search
				test_bar = tqdm(self.test_loader)
				for data, target in test_bar:
					data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
					feature, out = model(data)

					total_num += data.size(0)
					# compute cos similarity between each feature vector and feature bank ---> [B, N]
					sim_matrix = torch.mm(feature, feature_bank)
					# [B, K]
					sim_weight, sim_indices = sim_matrix.topk(k=self.k, dim=-1)
					# [B, K]
					sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
					sim_weight = (sim_weight / self.temperature).exp()

					# counts for each class
					one_hot_label = torch.zeros(data.size(0) * self.k, c, device=sim_labels.device)
					# [B*K, C]
					one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
					# weighted score ---> [B, C]
					pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

					pred_labels = pred_scores.argsort(dim=-1, descending=True)
					total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
					total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
					test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
												.format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

			test_acc_1 = total_top1 / total_num * 100
			test_acc_5 = total_top5 / total_num * 100
			results['test_acc@1'].append(test_acc_1)
			results['test_acc@5'].append(test_acc_5)

			data_frame = pd.DataFrame(data=results, index=range(epoch_start, epoch + 1))
			data_frame.to_csv(self.log_dir + '/log.csv', index_label='epoch')
			
			if test_acc_1 > best_acc:
				best_acc = test_acc_1
				torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': self.optimizer.state_dict()}, self.log_dir + '/model_last.pth')

class mocoTrainer():
	def __init__(self, log_dir, model, train_loader, memory_loader, test_loader, optimizer, temperature, k, lr, cos):
		self.log_dir = log_dir
		self.model = model
		self.train_loader = train_loader
		self.memory_loader = memory_loader
		self.test_loader = test_loader
		self.optimizer = optimizer
		self.temperature = temperature
		self.k = k
		self.lr = lr
		self.cos = cos

	def adjust_learning_rate(self, optimizer, epoch, epochs, lr, cos):
		"""Decay the learning rate based on schedule"""
		if cos:  # cosine lr schedule
			lr *= 0.5 * (1. + math.cos(math.pi * epoch / epochs))
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr

	def train(self, resume, epoch_start, epochs):
		results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}
		model = nn.DataParallel(self.model, device_ids=[0, 1, 2, 3]).cuda()

		batch_size = self.train_loader.batch_size
		c = len(self.memory_loader.dataset.classes)

		best_acc = 0.0

		if resume != '':
			checkpoint = torch.load(resume)
			model.load_state_dict(checkpoint['state_dict'])
			self.optimizer.load_state_dict(checkpoint['optimizer'])
			epoch_start = checkpoint['epoch'] + 1
			print('Loaded from: {}'.format(resume))

		for epoch in range(epoch_start, epochs + 1):

			model.train()
			self.adjust_learning_rate(self.optimizer, epoch, epochs, self.lr, self.cos)

			total_loss, total_num, train_bar = 0.0, 0, tqdm(self.train_loader)
			for im_1, im_2, _ in train_bar:
				im_1, im_2 = im_1.cuda(non_blocking=True), im_2.cuda(non_blocking=True)

				loss = model(im_1, im_2).mean()

				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

				total_num += batch_size
				total_loss += loss.item() * batch_size
				train_bar.set_description('Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}'.format(epoch, epochs, self.optimizer.param_groups[0]['lr'], total_loss / total_num))

			results['train_loss'].append(total_loss / total_num)

			model.eval()
			total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
			with torch.no_grad():
				# generate feature bank
				for data, target in tqdm(self.memory_loader, desc='Feature extracting'):
					feature = model.encoder_q(data.cuda(non_blocking=True))
					feature_bank.append(feature)

				# [D, N]
				feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
				# [N]
				feature_labels = torch.tensor(self.memory_loader.dataset.targets, device=feature_bank.device)

				# loop test data to predict the label by weighted knn search
				test_bar = tqdm(self.test_loader)
				for data, target in test_bar:
					data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
					feature = model.encoder_q(data)

					total_num += data.size(0)
					# compute cos similarity between each feature vector and feature bank ---> [B, N]
					sim_matrix = torch.mm(feature, feature_bank)
					# [B, K]
					sim_weight, sim_indices = sim_matrix.topk(k=self.k, dim=-1)
					# [B, K]
					sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
					sim_weight = (sim_weight / self.temperature).exp()

					# counts for each class
					one_hot_label = torch.zeros(data.size(0) * self.k, c, device=sim_labels.device)
					# [B*K, C]
					one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
					# weighted score ---> [B, C]
					pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

					pred_labels = pred_scores.argsort(dim=-1, descending=True)
					total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
					total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
					test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
												.format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

			test_acc_1 = total_top1 / total_num * 100
			test_acc_5 = total_top5 / total_num * 100
			results['test_acc@1'].append(test_acc_1)
			results['test_acc@5'].append(test_acc_5)

			data_frame = pd.DataFrame(data=results, index=range(epoch_start, epoch + 1))
			data_frame.to_csv(self.log_dir + '/log.csv', index_label='epoch')

			if test_acc_1 > best_acc:
				best_acc = test_acc_1
				torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': self.optimizer.state_dict()}, self.log_dir + '/model_last.pth')


