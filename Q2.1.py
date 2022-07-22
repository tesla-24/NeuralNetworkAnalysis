if __name__ ==  '__main__':


    import matplotlib.pyplot as plt
    from PIL import Image
    import torch, os
    import torchvision
    import torchvision.transforms as transforms
    from tqdm import tqdm
    import numpy as np
    from torch.autograd import Variable
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.models as models

    # Defining the transforms for images
    loader = transforms.Compose([transforms.ToTensor()])

    # function to mask an image with NxN window and (i,j) coordinate
    def mask(image,N,i,j):
        """Masks the given image using an NxN window starting from i,j coordinate"""
        mask = Image.fromarray(np.zeros((N,N,3),np.uint8))
        image.paste(mask,(i,j))
        
    def image_loader(image_name,N,i,j):
        """load image, returns cuda tensor of the image that has been masked """
        image = Image.open(image_name)
        mask(image,N,i,j)
        image = loader(image).float()
        image = Variable(image, requires_grad=True)
        image = image.unsqueeze(0) 
        return image.cuda()  #assumes that you're using GPU

    def save_heatmap(image_address, N, save_as_name):
        """Saves the heat map of the image(with patch NxN) with the address 'image_address' with the name 'save_as_name' """
        HeatMap = np.zeros((84-N+1,84-N+1))
        for i in range(0,84-N+1):
            for j in range(0,84-N+1):
                image = image_loader(image_address,N,i,j)
                HeatMap[i][j] = float(net(image)[0][15])
        plt.imshow(HeatMap.transpose(), cmap='hot', interpolation='nearest')
        plt.savefig(save_as_name)
        print("Figure saved in question2_images folder")
        plt.close()        
        



    class Net(nn.Module):
        # Defining the neural network
        def __init__(self):
            super(Net, self).__init__()
            kernel_s = 3
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=kernel_s)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=kernel_s)
            self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=kernel_s)
            self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=kernel_s)
            self.fc1 = nn.Linear(in_features=512, out_features=256)
            self.fc2 = nn.Linear(in_features=256, out_features=128)
            self.fc3 = nn.Linear(in_features=128, out_features=33)      

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = self.pool(F.relu(self.conv4(x)))
            x = F.avg_pool2d(x, kernel_size=x.shape[2:])
            x = x.view(x.shape[0], -1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    print('Performing Occlusion sensitivity experiment') 
    # initialising the network   
    net = Net()

    # transfer the model to GPU
    if torch.cuda.is_available():
        net = net.cuda()
    
    # Loading the saved model with the best parameters    
    net.load_state_dict(torch.load('models/model-49.pth'))
    net.eval()

    #Performing the Occlusion sensitivity experiment
    save_heatmap('images_for_ocular_experiment/green_mamba original.jpg', 5,'question2.1/green_mambat.jpg' )
    save_heatmap('images_for_ocular_experiment/green_mamba2 original.jpg', 5,'question2.1/green_mamba2t.jpg' )
    save_heatmap('images_for_ocular_experiment/orange2 original.jpg', 5,'question2.1/orange2t.jpg' )
    save_heatmap('images_for_ocular_experiment/orange3 original.jpg', 5, 'question2.1/orange3t.jpg')
    save_heatmap('images_for_ocular_experiment/consomme (1) original.jpg', 5,'question2.1/consomme (1)t.jpg' )
    save_heatmap('images_for_ocular_experiment/consomme (2) original.jpg', 5,'question2.1/consomme (2)t.jpg' )
    save_heatmap('images_for_ocular_experiment/toucan (4) original.jpg', 5,'question2.1/toucan (4)t.jpg' )
    save_heatmap('images_for_ocular_experiment/ladybug (3) original.jpg', 5, 'question2.1/ladybug (3)t.jpg')
    save_heatmap('images_for_ocular_experiment/poodle (2) original.jpg', 5,'question2.1/poodle (2)t.jpg' )
    save_heatmap('images_for_ocular_experiment/goose (2) original.jpg', 5,'question2.1/goose (2)t.jpg' )
    save_heatmap('images_for_ocular_experiment/vase (7) original.jpg', 5, 'question2.1/vase (7)t.jpg')