
import torch
import os
import sys
from data.utils import load_tensor_data
import torch.utils.data as data_utils
#from models.DeepLearning.FFN import FFN # needed if FFN class was not here
import torchvision
from torchvision import transforms
#from logger import Logger
import copy


MODEL = None



def get_device():
    return torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def get_data(device):
    x_train, y_train, _, _, _, _ = load_tensor_data()

    # For data loading
    if device == 'cpu':
        train_tensor = data_utils.TensorDataset(x_train, y_train)
    else:
        train_tensor = data_utils.TensorDataset(x_train.cuda(device=device), y_train.cuda(device=device))

    train_data = data_utils.DataLoader(train_tensor, 
                                        batch_size=batch_size, 
                                        shuffle=True)
    return train_data, x_train.shape[1]


class FFN(torch.nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, num_classes):
        super(FFN, self).__init__()
        self.h1 = torch.nn.Linear(input_size, hidden_size_1)
        self.h2 = torch.nn.Linear(hidden_size_1, hidden_size_2)
        self.h3 = torch.nn.Linear(hidden_size_2, num_classes)

    def forward(self, x):
        out = self.h1(x)
        out = torch.nn.functional.leaky_relu(out)
        out = self.h2(out)
        out = torch.nn.functional.leaky_relu(out)
        out = self.h3(out)
        out = torch.sigmoid(out)
        return out

def tell_me_more(currentEpoch, totalEpochs, currentBatch, totalBatches, currentLoss, labels, outputs):
    correct = 0
    correctBAD = 0
    totalBAD = 0
    total = 0
    #print('labels: ',labels)
    #print('outputs.data: ',outputs.data)
    printOut = [(x.item(), out[1].item(), abs(x-out[1]).item()) for x, out in zip(labels, outputs.data)]
    print(('label, output[1], abs(label-output[1])'))
    for e in printOut:
        total += 1
        if e[0] == 0:
            print(e)
            totalBAD += 1
        if e[2] < 0.1:
            if e[0] == 0:
                correctBAD += 1
            correct += 1
    print('correct / unsuccessful: %d / %d' % (correctBAD, totalBAD))
    print('\n')
    for e in printOut:
        if e[0] == 1:
            print(e)
    print('correct / successful: %d / %d' % (max(0,correct-correctBAD), total-totalBAD))
    #cross enthropy
    #_, predicted = torch.max(outputs.data, 1)
    #mse
    #predicted = copy.deepcopy(outputs.data)
    #predicted = predicted.view(predicted.numel())

    #print('predicted: ',predicted)
    
    #cross entropy
    #correct += (torch.autograd.Variable(predicted.long()) == torch.autograd.Variable(labels.long())).sum().item()
    #mse
    #error = (abs(predicted - labels))
    #for elem in error:
    #    if elem <= 0.1:
    #        correct += 1
    #print('error',error)
    #print('total: ',total)
    #print('correct: ',correct)
    print('correct / total: %d / %d' % (correct, total))
    print('[Epoch %d/%d],(Iteration %d / %d) acc: %f, loss: %f' % (currentEpoch, totalEpochs, currentBatch, totalBatches, (correct / total), currentLoss))
    

def get_model(input_size, hidden_size_1, hidden_size_2, newModel=True):
    if newModel:
        model = FFN(input_size=input_size, hidden_size_1=hidden_size_1, hidden_size_2=hidden_size_2, num_classes=2)

        #mse 
        #model = FFN(input_size=input_size, hidden_size_1=hidden_size_1, hidden_size_2=hidden_size_2, num_classes=1)
    else:
        model = torch.load('models/trained_model')
    return model

def main(optim, optim_args, hidden_size_1, hidden_size_2, loss_func, num_epochs, log_nth, initialExecution=True):
    global MODEL
    device = get_device()

    train_data, input_size = get_data(device)

    if initialExecution:
        MODEL = get_model(input_size, hidden_size_1, hidden_size_2)
    else:
        MODEL = get_model(input_size, hidden_size_1, hidden_size_2, newModel=False)

    MODEL.to(device)


    current_optim = optim(MODEL.parameters(), **optim_args)
    #print('these are the lerning rates:')
    #for g in current_optim.param_groups:
    #    print(g['lr'])
    
    iterations_per_epoch = len(train_data)

    print("iter per epoch: ", iterations_per_epoch)
    print('START TRAIN.')
    # num_iterations = iterations_per_epoch * num_epochs

    for epoch in range(num_epochs):
        running_loss = 0.0
        print("epoch: ", epoch)
        print('####################################################TRAIN##################################')        
        for i, data in enumerate(train_data, 0):

            inputs, labels = data
            #forward pass
            outputs = MODEL.forward(inputs)
            #cross entropy
            loss = loss_func(outputs, torch.autograd.Variable(labels.long()))
            #mse
            #loss = loss_func(outputs, torch.autograd.Variable(labels))

            # backwards and optimize
            current_optim.zero_grad()
            loss.backward()
            current_optim.step()
            # basically done here

            running_loss = loss.item()
            """
            if epoch < num_epochs*0.3:
                for g in current_optim.param_groups:
                    g['lr'] = 1e-5
            elif epoch < num_epochs*0.6:
                for g in current_optim.param_groups:
                    g['lr'] = 1e-6
            else:
                for g in current_optim.param_groups:
                    g['lr'] = 1e-7
            """
            # self.train_loss_history.append(running_loss)

            if (i % log_nth == 0):
                tell_me_more(currentEpoch=(epoch+1), 
                            totalEpochs=num_epochs, 
                            currentBatch=(i + 1),
                            totalBatches=iterations_per_epoch,
                            currentLoss=running_loss,
                            labels=labels,
                            outputs=outputs)
            #print(iterations_per_epoch, i)
            if i % iterations_per_epoch == iterations_per_epoch - 2:
                torch.save(MODEL, 'models/trained_model')
                tell_me_more(currentEpoch=(epoch+1), 
                            totalEpochs=num_epochs, 
                            currentBatch=(i + 1),
                            totalBatches=iterations_per_epoch,
                            currentLoss=running_loss,
                            labels=labels,
                            outputs=outputs)
                

        running_loss = 0.0
    ########################################################################
    #                             END OF YOUR CODE                         #
    ########################################################################

    torch.save(MODEL, 'models/trained_model')
    print('FINISH.')

if __name__=='__main__':
    try:
        initialize = True
        try:
            initialize = not (sys.argv[1] == 'c')
        except IndexError as e:
            pass
        hidden_size_1 = 50
        hidden_size_2 = 50

        batch_size = 30
        optim = torch.optim.Adam
        """
            lr (float, optional): learning rate (default: 1e-3)
            betas (Tuple[float, float], optional): coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
            eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
            weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        """
        optim_args = {
            "lr": 1e-5,
            "betas": ( 0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0.0
            }

        loss_func=torch.nn.CrossEntropyLoss()
        #loss_func=torch.nn.MSELoss()
        num_epochs=1000
        log_nth=1000
        main(optim=optim, optim_args=optim_args, hidden_size_1=hidden_size_1, hidden_size_2=hidden_size_2, loss_func=loss_func, num_epochs=num_epochs, log_nth=log_nth, initialExecution=initialize)
    except KeyboardInterrupt:
        print('\nInterrupted')
        torch.save(MODEL, 'models/trained_model_interrupted')
        sys.exit(0)