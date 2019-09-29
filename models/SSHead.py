from torch import nn
import math
import copy

class ViewFlatten(nn.Module):
	def __init__(self):
		super(ViewFlatten, self).__init__()

	def forward(self, x):
		return x.view(x.size(0), -1)

class ExtractorHead(nn.Module):
	def __init__(self, ext, head):
		super(ExtractorHead, self).__init__()
		self.ext = ext
		self.head = head

	def forward(self, x):
		return self.head(self.ext(x))

def extractor_from_layer3(net):
	layers = [net.conv1, net.layer1, net.layer2, net.layer3, net.bn, net.relu, net.avgpool, ViewFlatten()]
	return nn.Sequential(*layers)

def extractor_from_layer2(net):
	layers = [net.conv1, net.layer1, net.layer2]
	return nn.Sequential(*layers)

def head_on_layer2(net, width, classes):
	head = copy.deepcopy([net.layer3, net.bn, net.relu, net.avgpool])
	head.append(ViewFlatten())
	head.append(nn.Linear(64 * width, classes))
	return nn.Sequential(*head)
