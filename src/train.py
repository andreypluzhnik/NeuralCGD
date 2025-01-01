
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy.sparse

from vector_dataset import *
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch import nn

def init_zeros(m):
    if isinstance(m, nn.Conv3d):
        m.weight.data.fill_(0)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)

class ResidualBlock(nn.Module):
    def __init__(self, depth):
        super(ResidualBlock, self).__init__()
        self.x_layer = nn.Conv3d(in_channels = depth, out_channels = depth, kernel_size = (3,3,3), padding = "same", bias = False)   # no change
        self.y_layer = nn.Conv3d(in_channels = depth, out_channels = depth, kernel_size = (3,3,3), padding = "same", bias = False)   # no change
        self.relu = nn.ReLU(inplace = True)
        
        # self.x_layer.apply(init_zeros)
        # self.y_layer.apply(init_zeros)
    
    def forward(self,x):
        residual = x.clone()
        x1 = self.relu(self.x_layer(x))
        x2 = self.y_layer(x1)
        out = self.relu(x2) + residual
        return out




class SimpleModel(nn.Module):
   
    def __init__(self, dim_x, dim_y, dim_z):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        super(SimpleModel, self).__init__()
        depth = 2
        self.conv1 = nn.Sequential(
           nn.Conv3d(in_channels = 1, out_channels = depth, kernel_size = (3,3,3), padding = "same"),
           nn.Conv3d(in_channels = depth, out_channels = depth, kernel_size = (3,3,3), padding = "same"),
           nn.ReLU(),
           nn.Conv3d(in_channels = depth, out_channels = depth, kernel_size = (3,3,3), padding = "same"),
           nn.ReLU(),
           nn.Conv3d(in_channels = depth, out_channels = depth, kernel_size = (3,3,3), padding = "same"),
           nn.ReLU(),
           nn.Conv3d(in_channels = depth, out_channels = depth, kernel_size = (3,3,3), padding = "same"),
           nn.ReLU(),
           nn.Conv3d(in_channels = depth, out_channels = depth, kernel_size = (3,3,3), padding = "same"),
           nn.ReLU(),
           nn.Conv3d(in_channels = depth, out_channels = depth, kernel_size = (3,3,3), padding = "same"),
           nn.ReLU(),
           nn.Conv3d(in_channels = depth, out_channels = depth, kernel_size = (3,3,3), padding = "same"),
           nn.ReLU())
        

    

        self.fc_layer = nn.Linear(in_features = 4, out_features = 1)
        self.out_layer = nn.Conv3d(in_channels = 4, out_channels = 1, kernel_size = (3,3,3), padding = "same")

        self.conv1.apply(init_weights)
        self.fc_layer.apply(init_weights)
        
    # feed through the convolutional layers
    def forward(self, x):
        x = self.conv1(x)
        x = x.permute(0, 4, 2, 3, 1) 
        x = self.fc_layer(x)
        x = x.permute(0, 4, 2, 3, 1) 
        x = x.reshape(x.size(0), 1, x.size(2) * x.size(3) * x.size(4))
        return x


class CNN(nn.Module):

    

    def __init__(self, dim_x, dim_y, dim_z):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        super(CNN, self).__init__()
        self.depth = 16
        self.conv1 = nn.Sequential(
           nn.Conv3d(in_channels = 1, out_channels = self.depth, kernel_size = (3,3,3), padding = "same"),
           ResidualBlock(self.depth),
           ResidualBlock(self.depth))
        

        self.pool_layer = nn.AvgPool3d(kernel_size = (2,2,2), stride=(2,2,2))


        self.conv2 = nn.Sequential(
           ResidualBlock(self.depth),
           ResidualBlock(self.depth),
           ResidualBlock(self.depth),
           ResidualBlock(self.depth),
           ResidualBlock(self.depth))


        self.upsample_layer = nn.Upsample(scale_factor = (2,2,2))
        
        self.conv3 = nn.Sequential(
           ResidualBlock(self.depth),
           ResidualBlock(self.depth),
           ResidualBlock(self.depth))

        self.fc_layer = nn.Linear(in_features = self.depth, out_features = 1)
        self.out_layer = nn.Conv3d(in_channels = self.depth, out_channels = 1, kernel_size = (1, 1, 1), padding = "same")

            
    # feed through the convolutional layers
    def forward(self, x):
        xr = self.conv1(x).clone()
        x = self.pool_layer(xr.clone())
        x = self.conv2(x)
        x = self.upsample_layer(x) + xr
        x = self.conv3(x)
        x = self.out_layer(x)
    
        return x


def cgd_loss_fn(pred, y):
    error = 0
    eps = 1e-24
    dataset_size = len(pred)
    for i in range(dataset_size):
        p = pred[i].squeeze()
        p = p.flatten()
        y_ = y[i].squeeze()
        Af = torch.mv(A, p)
        fAf = torch.dot(p, Af)
        bf = torch.dot(p, y_)

        normed_Af = Af/torch.norm(Af)
        normed_y_ = y_/torch.norm(y_)
        # e = torch.linalg.norm(
        #     torch.sub(input = y_, alpha = 1, other = p), ord = 2)

        # error = error + e
        # alpha = torch.div(bf, fAf).item()
        # error = error + 1 - torch.square(torch.dot(normed_y_, normed_Af)) 
        error = error + torch.sum(torch.square(
            torch.sub(input = y_, alpha = 1, other = Af * bf/(fAf + eps)))) # = ||r - alpha (A*f)||
    return error / dataset_size
      
    

def train_loop(dataloader, model, loss_fn, optimizer):
    """
    trains model for an epoch
    returns an array of loss values over the training epoch
    """
    loss_array = [] # track losses from each batch
    model.train() # train mode

    for batch, X in enumerate(dataloader):
      # move images to device
      X = X.to(DEVICE, dtype=torch.float64)
      Y = X
      if(X.shape[0] == BATCH_SIZE):
        X = X.view((BATCH_SIZE, 1, dim_x, dim_y, dim_z))
      else:
        X = X.view((X.shape[0], 1, dim_x, dim_y, dim_z))
     
      # zero gradients from previous step
      optimizer.zero_grad()

      # compute prediction and loss
      pred = model(X)
      loss = loss_fn(pred, Y)

      # backpropogation
      loss.backward()
      optimizer.step()

      loss = loss.item()
      if(batch % 5 == 0):
        print(f"loss: {loss:>7f}")

      loss_array.append(loss)


    return loss_array


def test_loop(dataloader, model, loss_fn):
    """
    tests your model on the test set
    returns average loss
    """
    test_loss = 0
    model.eval() # evaluate mode

    size = len(dataloader.dataset)

    with torch.no_grad():
      for X in dataloader:
        # move images to GPU
        X = X.to(DEVICE, dtype=torch.float64)
        Y = X.clone()
        X = X.view((1, 1, dim_x, dim_y, dim_z))

        # compute prediction and loss
        pred = model(X)
        test_loss += loss_fn(pred, Y).item()


    test_loss /= size

    print(f"Test Error:\n Avg loss: {test_loss:>8f} \n")

    return test_loss



if __name__ == "__main__":
    # parse --demo flag, if not there FLAGS.demo == False
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--existing", action="store_true", default = False)
    parser.add_argument("-b", "--batch", type = int, default = 4)
    parser.add_argument("-c", "--epochs", type = int, default = 5)
    parser.add_argument("-l", "--lr", type = float, default = 1e-4)

    FLAGS, unparsed = parser.parse_known_args()

    

    dim_x, dim_y, dim_z = 8, 8, 8
    
    # make models directory
    if not os.path.exists("../models/"):
        os.makedirs("../models/")
    
    # define loss function
    loss_fn = cgd_loss_fn  # use paper defined loss
    # use double precision
    torch.set_default_dtype(torch.float64)

    # tweak these constants as you see fit, or get them through 'argparse'
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = FLAGS.batch
    EPOCHS = FLAGS.epochs
    LR = FLAGS.lr

    domain_name = f"A_matrix_{dim_x}_{dim_y}_{dim_z}_"

    dataset = VectorDataset()
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2]) 
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle = True)
    test_dataloader =  DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle = True)
    print("Dataset Loaded")

    # load A matrix, its stored as COO
    A = scipy.sparse.load_npz(f"{domain_name}.npz").toarray()
    # convert matrix to pytorch sparse format
    A = torch.tensor(A).to_sparse_csr()
    # A = torch.sparse_coo_tensor(np.array([A.row, A.col]), A.data).to_sparse_csr()
    A = A.to(DEVICE, dtype=torch.float64)

    torch.autograd.set_detect_anomaly(True)

    model = CNN(dim_x, dim_y, dim_z).to(DEVICE)

    if(FLAGS.existing):
        models_dir = f"../models/"
        model_fn = f"{dim_x}_{dim_y}_{dim_z}_grid_state_16_mod.pth"
        model.load_state_dict(torch.load(models_dir + model_fn))
    

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, amsgrad = True, eps=1e-10)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1)
    test_loss, train_loss = [], []




    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        print("Optimizer Learning Rate: ", optimizer.param_groups[-1]['lr'])

        epoch_train_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
        epoch_test_loss = test_loop(test_dataloader, model, loss_fn)

        train_loss = train_loss + epoch_train_loss
        test_loss = test_loss + [epoch_test_loss]
        scheduler.step()


    # save the model
    torch.save(model.state_dict(), f"../models/{dim_x}_{dim_y}_{dim_z}_grid_state_16_mod.pth")
    fig, axs = plt.subplots(1, 2, figsize=(15, 15))
    axs[0].plot(np.arange(1, len(train_loss) + 1), train_loss)
    axs[0].set_title("Train Loss")
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("Loss")

    axs[1].plot(np.arange(1, len(test_loss) + 1),  test_loss)
    axs[1].set_title("Validation Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Average Loss")
    plt.show()
    fig.savefig("../plots/test_train_loss.jpg")