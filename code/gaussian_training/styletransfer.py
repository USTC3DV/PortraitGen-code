import copy
# import utils
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from PIL import Image
import torchvision.transforms as T
# from settings import DEVICE, EPOCHS, STYLE_PATH, CONTENT_PATH, OUTPUT_PATH, STYLE_WEIGHT, CONTENT_WEIGHT

DEVICE = torch.device('cuda')
SIZE = 512
EPOCHS = 300

STYLE_WEIGHT = 1000000

CONTENT_WEIGHT = 1


class ContentLoss(nn.Module):
	def __init__(self, target):
		super(ContentLoss, self).__init__()
		self.target = target.detach()

	def forward(self, input):
		self.loss = F.mse_loss(input, self.target)
		return input

# Style Loss Layer
class StyleLoss(nn.Module):
	def __init__(self, target_feature):
		super(StyleLoss, self).__init__()
		self.target = gram_matrix(target_feature).detach()

	def forward(self, input):
		G = gram_matrix(input)
		self.loss = F.mse_loss(G, self.target)
		return input

def gram_matrix(input):
	a, b, c, d = input.size()
	features = input.view(a* b, c*d)
	G = torch.mm(features, features.t())
	return G.div(a*b*c*d)

# Normalization Layer to transform input images
class Normalization(nn.Module):
	def __init__(self, mean, std):
		super(Normalization, self).__init__()
		self.mean = torch.tensor(mean).view(-1, 1, 1)
		self.std = torch.tensor(std).view(-1, 1, 1)

	def forward(self, image):
		return (image - self.mean) / self.std

# Create our model with our loss layers
def style_cnn(cnn, device, normalization_mean, normalization_std, style_image, content_image):
	# Insert loss layers after these desired layers
	style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
	content_layers = ['conv_4']
	
	# Copy network to work on
	cnn = copy.deepcopy(cnn)

	# Keep track of our losses
	style_losses = []
	content_losses = []

	# Start by normalizing our image
	normalization = Normalization(normalization_mean, normalization_std).to(device)
	model = nn.Sequential(normalization)

	# Keep track of convolutional layers
	i = 0

	# Loop through vgg layers
	for layer in cnn.children():
		if isinstance(layer, nn.Conv2d):
			i += 1
			name = 'conv_{}'.format(i)
		elif isinstance(layer, nn.ReLU):
			name = 'relu_{}'.format(i)
			layer = nn.ReLU(inplace=False)
		elif isinstance(layer, nn.MaxPool2d):
			name = 'pool_{}'.format(i)
		elif isinstance(layer, nn.BatchNorm2d):
			name = 'bn_{}'.format(i)

		# Add layer to our model
		model.add_module(name, layer)

		# Insert style loss layer
		if name in style_layers:
			target_feature = model(style_image).detach()
			style_loss = StyleLoss(target_feature)
			model.add_module('style_loss_{}'.format(i), style_loss)
			style_losses.append(style_loss)

		# Insert content loss layer
		if name in content_layers:
			target = model(content_image).detach()
			content_loss = ContentLoss(target)
			model.add_module('content_loss_{}'.format(i), content_loss)
			content_losses.append(content_loss)

	# Get rid of unneeded layers after our final losses
	for i in range(len(model) - 1, -1, -1):
		if isinstance(model[i], StyleLoss) or isinstance(model[i], ContentLoss):
			break

	model = model[:(i + 1)]

	return model, style_losses, content_losses

# Image Transforms


def load_image(path, size, device, loader):

	image = loader(Image.open(path)).unsqueeze(0)
	return image.to(device, torch.float)

def save_image(tensor, path):
    unloader = T.ToPILImage()
    image = unloader(tensor.cpu().clone().squeeze(0))
    image.save(path)

class Style_transfer():
    def __init__(self, device, size, style_image_path, train_epoch=200):
        self.loader = T.Compose([
                            T.Resize(size),
                            T.CenterCrop(size),
                            T.ToTensor()
                            ])
        self.device = device
        self.size = size
        self.epoch = train_epoch
        self.style_image = load_image(style_image_path, self.size, self.device, self.loader)
        print(self.style_image.shape)

        self.cnn = models.vgg19(pretrained=True).features.to(self.device).eval()
        self.cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)

    def transfer(self, content_image_path):
        content_image = load_image(content_image_path, self.size, self.device, self.loader)

        target_image = content_image.clone().to(self.device).contiguous()

        # Build the model
        model, style_losses, content_losses = style_cnn(self.cnn, self.device, 
            self.cnn_normalization_mean, self.cnn_normalization_std, 
            self.style_image, content_image)

        # Optimization algorithm
        optimizer_new = optim.LBFGS([target_image.requires_grad_()])

        # Run style transfer
        run = [0]
        while run[0] < self.epoch:
            # Closure function is needed for LBFGS algorithm
            def closure():
                # Keep target values between 0 and 1
                target_image.data.clamp_(0, 1)

                optimizer_new.zero_grad()
                model(target_image)
                style_score = 0
                content_score = 0

                for s1 in style_losses:
                    style_score += s1.loss
                for c1 in content_losses:
                    content_score += c1.loss

                style_score *= 1000000
                content_score *= 1

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                # if run[0] == self.epoch-5:
                # if run[0] % 10 == 0:
                #     print('Run: {}'.format(run))
                #     print('Style Loss: {:4f} Content Loss: {:4f}'.format(style_score.item(), content_score.item()))
                #     print()


                return style_score + content_score

            optimizer_new.step(closure)

        target_image.data.clamp_(0, 1)

        return target_image.detach().clone().to(content_image.device)
