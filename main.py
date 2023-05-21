import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sklearn.manifold as manifold
import plotly.graph_objects as go

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset = torchvision.datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor())


class NeuralNet(nn.Module):
	def __init__(self, input_size, hidden_size,  num_classes):
		super(NeuralNet, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden_size)
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(hidden_size, num_classes)

	def forward(self, x):
		out = self.fc1(x)
		out = self.relu(out)
		out = self.fc2(out)
		return out

	def get_hidden(self, x):
		out = self.fc1(x)
		out = self.relu(out)
		return out


criterion = nn.CrossEntropyLoss()


def task_1():
	train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
	test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)
	train_errors = []
	test_errors = []
	model = NeuralNet(784, 500, 10).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
	for epoch in range(5):
		train_model_epoch(train_loader, model, optimizer)
		train_errors.append(calc_error(train_loader, model))
		test_errors.append(calc_error(test_loader, model))
	plot_task_1(train_errors, test_errors)
	plot_misclassified_images(test_loader, model)


def task_2():
	fig, axs = plt.subplots(1, 5, figsize=(20, 10))
	final_errors = []
	for i in range(5):
		test_errors = []
		torch.manual_seed(i)
		train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
		test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)
		model = NeuralNet(784, 500, 10).to(device)
		optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
		for epoch in range(5):
			train_model_epoch(train_loader, model, optimizer)
			test_errors.append(calc_error(test_loader, model))
		final_errors.append(test_errors[-1])
		axs = plot_task_2(test_errors, axs, i)
	print(f"the mean of the final test errors is: {np.mean(final_errors)}")
	print(f"the standard deviation of the final test errors is: {np.std(final_errors)}")

	plt.show()


def task_3():
	for i in range(5):
		torch.manual_seed(i)
		train_with_validation(100, 0.01, 500, i)


def task_4():
	for batch_size in [100, 1000]:
		for learning_rate in [0.001, 0.01, 0.1]:
			for hidden_size in [50, 500]:
				train_with_validation(batch_size, learning_rate, hidden_size)


def task_5():
	train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=False, sampler=torch.utils.data.SubsetRandomSampler(range(6000)))
	model = NeuralNet(784, 500, 10).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
	hidden_features = []
	regular_features = []
	regular_labels = []
	hidden_labels = []
	for i in range(5):
		train_model_epoch(train_loader, model, optimizer)
	model.eval()
	with torch.no_grad():
		for images, batch_labels in train_loader:
			images = images.to(device)
			batch_hidden = model.get_hidden(images.reshape(-1, 784))
			hidden_features.append(batch_hidden.detach().numpy())
			hidden_labels.append(batch_labels.numpy())
		for images, labels in train_loader:
			images = images.to(device)
			batch_regular = images.reshape(-1, 784)
			regular_features.append(batch_regular.detach().numpy())
			regular_labels.append(labels.numpy())
	hidden_features = np.concatenate(hidden_features, axis=0)
	hidden_labels = np.concatenate(hidden_labels, axis=0)
	regular_features = np.concatenate(regular_features, axis=0)
	regular_labels = np.concatenate(regular_labels, axis=0)
	plot_tsne(regular_labels, regular_features, "xi")
	plot_tsne(hidden_labels, hidden_features, "zi")


def plot_tsne(labels, features, title):
	tsne = manifold.TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000)
	embedded_features = tsne.fit_transform(features)
	plt.figure(figsize=(8, 8))
	unique_labels = np.unique(labels)
	colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
	for i, label in enumerate(unique_labels):
		mask = labels == label
		plt.scatter(embedded_features[mask, 0], embedded_features[mask, 1], color=colors[i], label=label)
	plt.legend(title="Digit")
	plt.xlabel("t-SNE Dimension 1")
	plt.ylabel("t-SNE Dimension 2")
	plt.title(f"2D Embedding of Hidden Features ({title})")
	plt.show()


def train_with_validation(batch_size, learning_rate, hidden_size, i=None):
	if i is None:
		train_subset, val_subset = torch.utils.data.random_split(train_dataset, [50000, 10000], generator=torch.Generator().manual_seed(1))
	else:
		train_subset, val_subset = torch.utils.data.random_split(train_dataset, [50000, 10000], generator=torch.Generator().manual_seed(i))
	train_loader = torch.utils.data.DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True)
	test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
	validation_loader = torch.utils.data.DataLoader(dataset=val_subset, batch_size=batch_size, shuffle=True)
	model = NeuralNet(784, hidden_size, 10).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	test_errors = []
	validation_errors = []
	for epoch in range(5):
		train_model_epoch(train_loader, model, optimizer)
		test_errors.append(calc_error(test_loader, model))
		validation_errors.append(calc_error(validation_loader, model))
	if i is not None:
		print(f"the best validation error achieved for seed {i} is: {min(validation_errors)}")
		print(f"the test error achieved is: {test_errors[np.argmin(validation_errors)]}\n")
	else:
		print(f"for batch size = {batch_size}, learning rate = {learning_rate} and hidden size = {hidden_size} the best validation error achieved is: {min(validation_errors)}")
		print(f"the test error achieved is: {test_errors[np.argmin(validation_errors)]}\n")


def train_model_epoch(loader, model, optimizer):
	for i, (images, labels) in enumerate(loader):
		images = images.reshape(-1, 784).to(device)
		labels = labels.to(device)

		outputs = model(images)
		loss = criterion(outputs, labels)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()


def calc_error(loader, model):
	with torch.no_grad():
		correct = 0
		total = 0
		for images, labels in loader:
			images = images.reshape(-1, 784).to(device)
			labels = labels.to(device)
			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted != labels).sum().item()
		return 100 * correct / total


def plot_task_1(train_errors, test_errors):
	plt.xlabel('Epoch')
	plt.ylabel('Error')
	plt.title('Error vs Epoch')
	plt.plot(range(1, 6), train_errors, label='Train errors')
	plt.plot(range(1, 6), test_errors, label='Test errors')
	plt.legend()
	plt.show()


def plot_task_2(test_errors, axs, i):
	axs[i].set_xlabel('Epoch')
	axs[i].set_ylabel('Error')
	axs[i].set_title('Error vs Epoch for seed number: ' + str(i))
	axs[i].plot(range(1, 6), test_errors, label='Test errors')
	axs[i].legend()
	return axs


def plot_misclassified_images(loader, model):
	model.eval()
	fig = plt.figure(figsize=(20, 10))
	misclassified_count = 0
	for i, (images, labels) in enumerate(loader):
		images = images.reshape(-1, 784).to(device)
		labels = labels.to(device)
		outputs = model(images)
		_, predicted = torch.max(outputs.data, 1)
		for j in range(len(predicted)):
			if predicted[j] != labels[j]:
				misclassified_count += 1
				fig.add_subplot(2, 5, misclassified_count)
				plt.imshow(images[j].cpu().numpy().reshape(28, 28), cmap='gray')
				plt.title('Predicted: ' + str(predicted[j].cpu().numpy()) + ' Actual: ' + str(labels[j].cpu().numpy()))
				if misclassified_count == 10:
					break
		if misclassified_count == 10:
			break
	plt.show()


def main():
	task_5()


if __name__ == '__main__':
	main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
