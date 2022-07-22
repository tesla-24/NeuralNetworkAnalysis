if __name__ ==  '__main__':

    import cv2
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
    import matplotlib.pyplot as plt

    # Defining the neutal network
    class Net(nn.Module):
        def __init__(self):
            kernel_s = 3
            self.image_batch = []
            self.layer1 = [] # initialing listis to store the output of each layer
            self.layer2 = []
            self.layer3 = []
            self.layer4 = []
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=kernel_s)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=kernel_s)
            self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=kernel_s)
            self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=kernel_s)
            self.fc1 = nn.Linear(in_features=512, out_features=256)
            self.fc2 = nn.Linear(in_features=256, out_features=128)
            self.fc3 = nn.Linear(in_features=128, out_features=33)      # change out_features according to number of classes

        def forward(self, x):
            # appending the output of each layer
            self.image_batch.append(  np.array(x.cpu()) )
            x = F.relu(self.conv1(x))
            self.layer1.append(  np.array(x[:,:2].cpu()) )
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            self.layer2.append(  np.array(x[:,70:72].cpu()) )
            x = self.pool(x)
            x = F.relu(self.conv3(x))
            self.layer3.append(  np.array(x[:,100:102].cpu()) )
            x = self.pool(x)
            x = F.relu(self.conv4(x))
            self.layer4.append(  np.array(x[:,50:52].cpu()) )
            x = self.pool(x)
            x = F.avg_pool2d(x, kernel_size=x.shape[2:])
            x = x.view(x.shape[0], -1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            self.layer7 = x[:,:2]
            return x



    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5])])

    train_data_dir = 'train' # put path of training dataset
    val_data_dir = 'val' # put path of validation dataset
    test_data_dir = 'test' # put path of test dataset



    testset = torchvision.datasets.ImageFolder(root= test_data_dir, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size = 100,
                                             shuffle=False, num_workers=2)


    def test(testloader, net):
        """prints the accuracy of the model with data as testloader"""
        correct = 0
        total = 0
        with torch.no_grad():
            for data in tqdm(testloader):
                images, labels = data
                if torch.cuda.is_available():
                    images, labels = images.cuda(), labels.cuda()        
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (
                                        100 * correct / total))

    def find_classes(dir):
        """get details of classes and class to index mapping in a directory"""
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx


    def classwise_test(testloader, net):
        """returns the list of correctly classified images """
        correctly_classified = []
        classes, _ = find_classes(train_data_dir)
        n_class = len(classes) # number of classes

        class_correct = list(0. for i in range(n_class))
        class_total = list(0. for i in range(n_class))
        with torch.no_grad():
            for data in tqdm(testloader):
                images, labels = data
                if torch.cuda.is_available():
                    images, labels = images.cuda(), labels.cuda()
                
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                correctly_classified.append(np.where(np.array(predicted.cpu())==np.array(labels.cpu())[0]))
                c = (predicted == labels).squeeze()
                for i in range(4):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        
        return correctly_classified

    class UnNormalize(object):
        """Returns an unnormalized tensor of the given image """
        def __init__(self, mean, std):
            self.mean = mean
            self.std = std

        def __call__(self, tensor):
            for t, m, s in zip(tensor, self.mean, self.std):
                t.mul_(s).add_(m)
            return tensor
        
    def top_N_coord(array,N,kernel):
        """
        returns the top N coordinates of the given 
        array(No_of_images * no_of_kernels * height * width) 
        for the given kernel
        """
        array = array[:,kernel].copy()
        total = 0
        top_coord = []
        while(total <= N):
            indices = np.where(array == np.max(array))
            ind = list(zip(indices[0],indices[1],indices[2]))
            total = total + int(len(ind)/2)
            for i in range(0,len(ind),2):
                top_coord.append(ind[i])
            for i in ind:
                array[i] = -1000
        return top_coord[:5]

    def print_image_layer1(img_array, coord, kernel_size, kernel_number ):
        """
        Saves the patch of the image which had the highest response for 
        the selected kernel_number in layer 1 and given kernel_size
        """
        unorm = UnNormalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5])
        image= unorm(torch.tensor(img_array[coord[0]]))
        img = np.array(image).transpose((1,2,0))
        plt.imshow(img[coord[1]: coord[1]+kernel_size , coord[2] : coord[2]+kernel_size ])
        plt.savefig('question2.2/Q2.2.1/layer1_k'+str(kernel_number)+'_'+str(coord)+'.jpg')
        plt.close()
        
    def print_image_layer2(img_array, coord, kernel_size, kernel_number ):
        """
        Saves the patch of the image which had the highest response for 
        the selected kernel_number in layer 2 and given kernel_size
        """
        unorm = UnNormalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5])
        image= unorm(torch.tensor(img_array[coord[0]]))
        img = np.array(image).transpose((1,2,0))
        plt.imshow(img[coord[1] : coord[1]+kernel_size+1 , coord[2] : coord[2]+kernel_size+1 ])
        plt.savefig('question2.2/Q2.2.1/layer2_k'+str(kernel_number)+'_'+str(coord)+'.jpg')
        plt.close()
        
    def print_image_layer3(img_array, coord, kernel_size, kernel_number ):
        """
        Saves the patch of the image which had the highest response for 
        the selected kernel_number in layer 3 and given kernel_size
        """
        unorm = UnNormalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5])
        image= unorm(torch.tensor(img_array[coord[0]]))
        img = np.array(image).transpose((1,2,0))
        plt.imshow(img[coord[1] : coord[1]+kernel_size+2 , coord[2] : coord[2]+kernel_size+2 ])
        plt.savefig('question2.2/Q2.2.1/layer3_k'+str(kernel_number)+'_'+str(coord)+'.jpg')
        plt.close()
        
    def print_image_layer4(img_array, coord, kernel_size, kernel_number ):
        """
        Saves the patch of the image which had the highest response for 
        the selected kernel_number in layer 4 and given kernel_size
        """
        unorm = UnNormalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5])
        image= unorm(torch.tensor(img_array[coord[0]]))
        img = np.array(image).transpose((1,2,0))
        plt.imshow(img[coord[1] : coord[1]+kernel_size+3 , coord[2] : coord[2]+kernel_size+3 ])
        plt.savefig('question2.2/Q2.2.1/layer4_k'+str(kernel_number)+'_'+str(coord)+'.jpg')
        plt.close()

    #Initializing the model
    net = Net()
    # transfer the model to GPU
    if torch.cuda.is_available():
        net = net.cuda()
    net.load_state_dict(torch.load('models/model-49.pth'))
    net.eval()

    print('Passing all test images through the model ...')
    test(testloader, net)
    # storing the correctly classified images in correct array
    correct = classwise_test(testloader, net)

    # defining the variables for the outputs of the filter responses in the particular layers
    images = np.array(net.image_batch)
    layer1 = np.array(net.layer1)
    layer2 = np.array(net.layer2)
    layer3 = np.array(net.layer3)
    layer4 = np.array(net.layer4)


    # Reshaping the arrays as the testloader had loaded images in batches.
    images = images.reshape((np.shape(images)[0]*np.shape(images)[1], np.shape(images)[2], np.shape(images)[3], np.shape(images)[4]))
    layer1 = layer1.reshape((np.shape(layer1)[0]*np.shape(layer1)[1], np.shape(layer1)[2], np.shape(layer1)[3], np.shape(layer1)[4]))
    layer2 = layer2.reshape((np.shape(layer2)[0]*np.shape(layer2)[1], np.shape(layer2)[2], np.shape(layer2)[3], np.shape(layer2)[4]))
    layer3 = layer3.reshape((np.shape(layer3)[0]*np.shape(layer3)[1], np.shape(layer3)[2], np.shape(layer3)[3], np.shape(layer3)[4]))
    layer4 = layer4.reshape((np.shape(layer4)[0]*np.shape(layer4)[1], np.shape(layer4)[2], np.shape(layer4)[3], np.shape(layer4)[4]))

    # Finding the top 5 responses for 2 kernels in each layer
    layer1_k1  = top_N_coord(layer1,5,0)
    layer1_k2  = top_N_coord(layer1,5,1)
    layer2_k1  = top_N_coord(layer2,5,0)
    layer2_k2  = top_N_coord(layer2,5,1)
    layer3_k1  = top_N_coord(layer3,5,0)
    layer3_k2  = top_N_coord(layer3,5,1)
    layer4_k1  = top_N_coord(layer4,5,0)
    layer4_k2  = top_N_coord(layer4,5,1)

    # saving the patch corresponding to maximum response in each kernel in the predifined directory
    for i in layer1_k1:
        print_image_layer1(images, i, 3,1)

    for i in layer1_k2:
        print_image_layer1(images, i, 3,2)
        
    for i in layer2_k1:
        print_image_layer2(images, i, 3,1)

    for i in layer2_k2:
        print_image_layer2(images, i, 3,2)
        
    for i in layer3_k1:
        print_image_layer3(images, i, 3,1)
        
    for i in layer3_k2:
        print_image_layer3(images, i, 3,2)
        
    for i in layer4_k1:
        print_image_layer4(images, i, 3,1)
        
    for i in layer4_k2:
        print_image_layer4(images, i, 3,2)
    print("All the image of filter response are saved in question3_images/Q2.2.1 folder")

    # setting the following filter to zero for filter modification experiment
    filters_to_be_set_zero = {'conv1':[0,1],'conv2':[70,71],'conv3':[100, 101],'conv4':[50,51]}

    # Resetting the filters to zero 
    for index, item in enumerate(net.named_children()):
        with torch.no_grad():
            if(isinstance(item[1],nn.Conv2d)):
                wt_list = filters_to_be_set_zero[item[0]]
                for i in wt_list:
                    item[1].weight[i].copy_(torch.zeros_like(item[1].weight[i]))
                    item[1].bias[i].copy_(torch.zeros_like(item[1].bias[i]))
                    
    print('Setting the values of selected filters to zero ...')

    def wrong_after_filter_zero(testloader, model, correct_array):
        """
        prints the images that were wrongly classified after the selected filters were set to zero
        """
        i=0
        image_no = 1
        results = []
        classes, _ = find_classes(train_data_dir)
        n_class = len(classes) # number of classes

        class_correct = list(0. for i in range(n_class))
        class_total = list(0. for i in range(n_class))
        with torch.no_grad():
            for data in tqdm(testloader):
                images, labels = data
                if torch.cuda.is_available():
                    images, labels = images.cuda(), labels.cuda()
                
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                wrongly_classified = np.where(np.array(predicted.cpu())!=np.array(labels.cpu())[0])
                common  = set(list(wrongly_classified)[0]) & set(list(correct_array[i])[0])
                results.append((str(classes[i])+' : '+str(len(common))))
                if(len(common)!=0):
                    for z in list(common):
                        unorm = UnNormalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5])
                        image = unorm(torch.tensor(images[z].cpu()))
                        image = np.array(image).transpose((1,2,0))
                        plt.imshow(image)
                        plt.savefig('question2.2/Q2.2.2/misclassified_'+str(classes[i])+' '+str(image_no)+'.jpg')
                        image_no = image_no + 1
                        plt.close()
                i=i+1
        print('The misclassified images after filter modification are :-')
        for x in results:
            print(x)
    wrong_after_filter_zero(testloader, net, correct)