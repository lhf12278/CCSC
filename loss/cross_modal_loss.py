import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLabelSmooth(nn.Module):
	"""Cross entropy loss with label smoothing regularizer.

	Reference:
	Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
	Equation: y = (1 - epsilon) * y + epsilon / K.

	Args:
		num_classes (int): number of classes.
		epsilon (float): weight.
	"""

	def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
		super(CrossEntropyLabelSmooth, self).__init__()
		self.num_classes = num_classes
		self.epsilon = epsilon
		self.use_gpu = use_gpu
		self.logsoftmax = nn.LogSoftmax(dim=1)

	def forward(self, inputs, targets):
		"""
		Args:
			inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
			targets: ground truth labels with shape (num_classes)
		"""
		log_probs = self.logsoftmax(inputs)
		targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
		if self.use_gpu: targets = targets.to(torch.device('cuda'))
		targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
		loss = (- targets * log_probs).mean(0).sum()
		return loss


class RankingLoss:

	def __init__(self):
		pass

	def _label2similarity(sekf, label1, label2):
		'''
		compute similarity matrix of label1 and label2
		:param label1: torch.Tensor, [m]
		:param label2: torch.Tensor, [n]
		:return: torch.Tensor, [m, n], {0, 1}
		'''
		m, n = len(label1), len(label2)
		l1 = label1.view(m, 1).expand([m, n])
		l2 = label2.view(n, 1).expand([n, m]).t()
		similarity = l1 == l2
		return similarity

	def _batch_hard(self, mat_distance, mat_similarity, more_similar):

		if more_similar is 'smaller':
			sorted_mat_distance, _ = torch.sort(mat_distance + (-9999999.) * (1 - mat_similarity), dim=1,descending=True)
			hard_p = sorted_mat_distance[:, 0]
			sorted_mat_distance, _ = torch.sort(mat_distance + (9999999.) * (mat_similarity), dim=1, descending=False)
			hard_n = sorted_mat_distance[:, 0]
			return hard_p, hard_n

		elif more_similar is 'larger':
			sorted_mat_distance, _ = torch.sort(mat_distance + (9999999.) * (1 - mat_similarity), dim=1, descending=False)
			hard_p = sorted_mat_distance[:, 0]
			sorted_mat_distance, _ = torch.sort(mat_distance + (-9999999.) * (mat_similarity), dim=1, descending=True)
			hard_n = sorted_mat_distance[:, 0]
			return hard_p, hard_n


class TripletLoss(RankingLoss):
	'''
	Compute Triplet loss augmented with Batch Hard
	Details can be seen in 'In defense of the Triplet Loss for Person Re-Identification'
	'''

	def __init__(self, margin, metric):
		'''
		:param margin: float or 'soft', for MarginRankingLoss with margin and soft margin
		:param bh: batch hard
		:param metric: l2 distance or cosine distance
		'''
		self.margin = margin
		self.margin_loss = nn.MarginRankingLoss(margin=margin)
		self.metric = metric

	def __call__(self, emb1, emb2, emb3, label1, label2, label3):
		'''

		:param emb1: torch.Tensor, [m, dim]
		:param emb2: torch.Tensor, [n, dim]
		:param label1: torch.Tensor, [m]
		:param label2: torch.Tensor, [b]
		:return:
		'''

		if self.metric == 'cosine':
			mat_dist = cosine_dist(emb1, emb2)
			mat_sim = self._label2similarity(label1, label2)
			hard_p, _ = self._batch_hard(mat_dist, mat_sim.float(), more_similar='larger')

			mat_dist = cosine_dist(emb1, emb3)
			mat_sim = self._label2similarity(label1, label3)
			_, hard_n = self._batch_hard(mat_dist, mat_sim.float(), more_similar='larger')

			margin_label = -torch.ones_like(hard_p)

		elif self.metric == 'euclidean':
			mat_dist = euclidean_dist(emb1, emb2)
			mat_sim = self._label2similarity(label1, label2)
			hard_p, _ = self._batch_hard(mat_dist, mat_sim.float(), more_similar='smaller')

			mat_dist = euclidean_dist(emb1, emb3)
			mat_sim = self._label2similarity(label1, label3)
			_, hard_n = self._batch_hard(mat_dist, mat_sim.float(), more_similar='smaller')

			margin_label = torch.ones_like(hard_p)

		return self.margin_loss(hard_n, hard_p, margin_label)




def cosine_dist(x, y):
	'''
	:param x: torch.tensor, 2d
	:param y: torch.tensor, 2d
	:return:
	'''

	bs1 = x.size()[0]
	bs2 = y.size()[0]

	frac_up = torch.matmul(x, y.transpose(0, 1))
	frac_down = (torch.sqrt(torch.sum(torch.pow(x, 2), 1))).view(bs1, 1).repeat(1, bs2) * \
	            (torch.sqrt(torch.sum(torch.pow(y, 2), 1))).view(1, bs2).repeat(bs1, 1)
	cosine = frac_up / frac_down

	return cosine


def euclidean_dist(x, y):
	"""
	Args:
	  x: pytorch Variable, with shape [m, d]
	  y: pytorch Variable, with shape [n, d]
	Returns:
	  dist: pytorch Variable, with shape [m, n]
	"""
	m, n = x.size(0), y.size(0)
	xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
	yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
	dist = xx + yy
	dist.addmm_(1, -2, x, y.t())
	dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
	return dist



def kl_loss(score1, score2, mini=1e-8):

	# prob1 = torch.zeros(score2.size()).scatter_(1, score1.unsqueeze(1).data.cpu(), 1)
	# prob1 = prob1.to(torch.device('cuda'))

	prob1 = F.softmax(score1, dim=1)
	prob2 = F.softmax(score2, dim=1)
	loss = torch.sum(prob1 * torch.log(mini + prob1 / (prob2 + mini)), 1)
	return loss.mean()



class CrossEntropyLabel(nn.Module):
	"""Cross entropy loss with label smoothing regularizer.

	Reference:
	Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
	Equation: y = (1 - epsilon) * y + epsilon / K.

	Args:
		num_classes (int): number of classes.
		epsilon (float): weight.
	"""

	def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
		super(CrossEntropyLabel, self).__init__()
		self.num_classes = num_classes
		self.epsilon = epsilon
		self.use_gpu = use_gpu
		self.logsoftmax = nn.LogSoftmax(dim=1)

	def forward(self, inputs, targets):
		"""
		Args:
			inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
			targets: ground truth labels with shape (num_classes)
		"""
		device = "cuda"
		target1 = targets.to(device)
		log_probs = self.logsoftmax(inputs)
		targets = torch.zeros(log_probs.size()).scatter_(1, target1.unsqueeze(1).data.cpu(), 1)
		if self.use_gpu: targets = targets.to(torch.device('cuda'))
		targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
		loss = (- targets * log_probs).mean(0).sum()
		return loss

