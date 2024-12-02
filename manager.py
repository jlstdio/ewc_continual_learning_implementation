import json
import torch
from torch import optim
import time
from dataset.cifar10.cifar10DataLoader import cifar10Dataloader
from dataset.cifar100.cifar100DataLoader import cifar100Dataloader
from model.testModel_w_softmax import testNN_w_Softmax
from prepare_dataset.clustered_iid import create_cluster_loaders
from tester import test
from trainer import train


class manager:
    def __init__(self, configPath):
        #########################
        # DATASET TYPE SETTINGS #
        #########################
        with open(configPath, 'r') as file:
            config = json.load(file)

        self.basic_info = config['basic_info']
        self.train_info = config['train_info']
        # ##################################################################

        #####################
        # PARAMETER SETTING #
        #####################
        self.batchSize = self.train_info['batch_size']
        self.epoch = self.train_info['epoch']
        self.numClusters = self.basic_info['numClusters']

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'{self.device} is available')
        # ##################################################################

        #########################
        # DATASET TYPE SETTINGS #
        #########################
        dataloader = None
        if self.basic_info['dataset'] == 'cifar-10':
            dataloader = cifar10Dataloader('./dataset/cifar10')
        elif self.basic_info['dataset'] == 'cifar-100':
            dataloader = cifar100Dataloader('./dataset/cifar100')
        self.train_set, self.test_set = dataloader.load_data() # (x_train, y_train), (x_test, y_test)

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        # ##################################################################

        ###############
        # MODEL SETUP #
        ###############
        self.model = testNN_w_Softmax().to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=4e-5)
        # ##################################################################

    def run(self):
        self.train_loader, self.val_loader = create_cluster_loaders(train_set=self.train_set,
                                                                    test_set=self.test_set,
                                                                    numClusters=self.numClusters,
                                                                    costFunc=self.train_info['costFunc'],
                                                                    batchSize=self.train_info['batch_size'])

        lossHistory = []
        for cluster in range(self.numClusters):
            lossHistory[cluster] = train(self.model, self.epoch, self.device, self.train_loader, self.val_loader, self.optimizer, self.criterion)


if __name__ == "__main__":
    configPath = ''
    manager = manager(configPath)
    manager.run()
