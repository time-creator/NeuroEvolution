import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageCNN(nn.Module):

    def __init__(self):
        super(ImageCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=10, out_channels=15, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=15, out_channels=25, kernel_size=3)
        self.fc1 = nn.Linear(in_features=25 * 6 * 6, out_features=400)
        self.fc2 = nn.Linear(in_features=400, out_features=100)
        self.fc3 = nn.Linear(in_features=100, out_features=3)

    def get_layers(self):
        layers = []
        layers.append(self.conv1)
        layers.append(self.conv2)
        layers.append(self.conv3)
        layers.append(self.conv4)
        layers.append(self.fc1)
        layers.append(self.fc2)
        layers.append(self.fc3)
        return layers

    def forward(self, x):
        # First Conv2D Layer
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2)

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2)

        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, kernel_size=2)

        x = x.reshape(-1, 25 * 6 * 6)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = F.softmax(self.fc3(x), 1)

        return x

class NIMACNN(nn.Module):

    def __init__(self):
        super(NIMACNN, self).__init__()

        # input size 224 x 224
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=10, out_channels=15, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=15, out_channels=20, kernel_size=3)
        self.conv5 = nn.Conv2d(in_channels=20, out_channels=25, kernel_size=3)
        self.fc1 = nn.Linear(in_features=25 * 5 * 5, out_features=300)
        self.fc2 = nn.Linear(in_features=300, out_features=100)
        self.fc3 = nn.Linear(in_features=100, out_features=3)

    def get_layers(self):
        layers = []
        layers.append(self.conv1)
        layers.append(self.conv2)
        layers.append(self.conv3)
        layers.append(self.conv4)
        layers.append(self.conv5)
        layers.append(self.fc1)
        layers.append(self.fc2)
        layers.append(self.fc3)
        return layers

    def forward(self, x):
        # First Conv2D Layer
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2)

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2)

        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, kernel_size=2)

        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, kernel_size=2)

        x = x.reshape(-1, 25 * 5 * 5)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = F.softmax(self.fc3(x), 1)

        return x

class ImageAutoencoder(nn.Module):

    def __init__(self):
        super(ImageAutoencoderTwo, self).__init__()

        self.encoder_conv1 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=5)
        self.encoder_conv2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3)
        self.encoder_conv3 = nn.Conv2d(in_channels=10, out_channels=15, kernel_size=3)
        self.encoder_conv4 = nn.Conv2d(in_channels=15, out_channels=25, kernel_size=3)
        self.encoder_fc1 = nn.Linear(in_features=25 * 6 * 6, out_features=400)
        self.encoder_fc2 = nn.Linear(in_features=400, out_features=100)

        self.decoder_fc1 = nn.Linear(in_features=100, out_features=400)
        self.decoder_fc2 = nn.Linear(in_features=400, out_features=25 * 6 * 6)
        self.decoder_deconv1 = nn.ConvTranspose2d(in_channels=25 ,out_channels=15 ,kernel_size=3)
        self.decoder_deconv2 = nn.ConvTranspose2d(in_channels=15 ,out_channels=10 ,kernel_size=3)
        self.decoder_deconv3 = nn.ConvTranspose2d(in_channels=10 ,out_channels=5 ,kernel_size=3)
        self.decoder_deconv4 = nn.ConvTranspose2d(in_channels=5 ,out_channels=3 ,kernel_size=5)

    def get_layers(self):
        layers = []
        layers.append(self.encoder_conv1)
        layers.append(self.encoder_conv2)
        layers.append(self.encoder_conv3)
        layers.append(self.encoder_conv4)
        layers.append(self.encoder_fc1)
        layers.append(self.encoder_fc2)

        layers.append(self.decoder_fc1)
        layers.append(self.decoder_fc2)
        layers.append(self.decoder_deconv1)
        layers.append(self.decoder_deconv2)
        layers.append(self.decoder_deconv3)
        layers.append(self.decoder_deconv4)
        return layers

    def forward(self, x):
        # Start Encoder
        x = F.relu(self.encoder_conv1(x))
        x = F.max_pool2d(x, kernel_size=2)

        x = F.relu(self.encoder_conv2(x))
        x = F.max_pool2d(x, kernel_size=2)

        x = F.relu(self.encoder_conv3(x))
        x = F.max_pool2d(x, kernel_size=2)

        x = F.relu(self.encoder_conv4(x))
        x = F.max_pool2d(x, kernel_size=2)

        x = x.reshape(-1, 25 * 6 * 6)

        x = F.relu(self.encoder_fc1(x))

        x = torch.sigmoid(self.encoder_fc2(x))
        # End Encoder and start Decoder

        x = F.relu(self.decoder_fc1(x))

        x = torch.sigmoid(self.decoder_fc2(x))

        x = x.reshape(25, 6, 6)
        x = x.unsqueeze_(0)

        x = F.interpolate(x, scale_factor=2)
        x = F.relu(self.decoder_deconv1(x))

        x = F.interpolate(x, scale_factor=2)
        x = F.relu(self.decoder_deconv2(x))

        x = F.interpolate(x, scale_factor=2)
        x = F.relu(self.decoder_deconv3(x))

        x = F.interpolate(x, scale_factor=2)
        x = torch.sigmoid(self.decoder_deconv4(x))

        return x
