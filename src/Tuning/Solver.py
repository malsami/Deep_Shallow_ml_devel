import torch
from torch.autograd import Variable
from tqdm import tqdm

import numpy as np
# Adapted from deep learning class I2DL exercise assignments

# This is a large hyperparameter tuning.

def train(epoch, data_loader, net, criterion, optimizer, cuda):

    progress_bar = tqdm(data_loader)
    moving_loss = 0
    acc = list()

    net.train()
    for x, y in progress_bar:
        x, y = Variable(x), Variable(y)

        if cuda:
            x, y = Variable(x).cuda(), Variable(y).cuda()

        output = net(x)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if moving_loss == 0:
            moving_loss = loss.item()
        else:
            moving_loss = moving_loss * 0.9 + loss.item() * 0.1

        progress_bar.set_description(
            'Epoch: {}; Loss: {:.5f}; Avg: {:.5f}'.format(epoch + 1, loss.item(), moving_loss))

        acc.append((y == torch.argmax(F.softmax(output, -1), dim=-1)).sum().float() / y.size()[0])
    acc = np.mean(acc)
    print('train accuracy: {}'.format(acc))

    return moving_loss, acc


def valid(data_loader, net, cuda, criterion):
    net.eval()

    acc = list()
    total_loss = list()
    for x, y in tqdm(data_loader):
        x, y = Variable(x), Variable(y)

        if cuda:
            x, y = Variable(x).cuda(), Variable(y).cuda()

        output = net(x)
        acc.append((y == torch.argmax(F.softmax(output, -1), dim=-1)).sum().float() / y.size()[0])
        loss = total_loss.append(criterion(output, y).item())
    acc = np.mean(acc)
    print('Validation accuracy: {}'.format(acc))

    return np.mean(total_loss), acc

class Solver(object):
    default_adam_args = {"lr": 1e-3,
                         "betas": (.75, 0.9),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def _train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.
        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=3, gamma=0.95)

        self._reset_histories()
        iter_per_epoch = len(train_loader)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        criterion = self.loss_func

        print('START TRAIN.')
        ########################################################################
        # TODO:                                                                #
        # Write your own personal training method for our solver. In each      #
        # epoch iter_per_epoch shuffled training batches are processed. The    #
        # loss for each batch is stored in self.train_loss_history. Every      #
        # log_nth iteration the loss is logged. After one epoch the training   #
        # accuracy of the last mini batch is logged and stored in              #
        # self.train_acc_history. We validate at the end of each epoch, log    #
        # the result and store the accuracy of the entire validation set in    #
        # self.val_acc_history.                                                #
        #                                                                      #
        # Your logging could like something like:                              #
        #   ...                                                                #
        #   [Iteration 700/4800] TRAIN loss: 1.452                             #
        #   [Iteration 800/4800] TRAIN loss: 1.409                             #
        #   [Iteration 900/4800] TRAIN loss: 1.374                             #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                            #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                            #
        #   ...                                                                #
        ########################################################################

        for e in range(num_epochs):
            scheduler.step(e)
            loss, acc = train(e, train_loader, model, criterion, optim, torch.cuda.is_available())
            self.train_loss_history.append(loss)
            self.train_acc_history.append(acc)
            loss, acc = valid(val_loader, model, torch.cuda.is_available(), criterion)
            self.val_loss_history.append(loss)
            self.val_acc_history.append(acc)

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        print('FINISH.')

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print("iter per epoch: ", iter_per_epoch)
        print('START TRAIN.')
        ########################################################################
        # TODO:                                                                #
        # Write your own personal training method for our solver. In each      #
        # epoch iter_per_epoch shuffled training batches are processed. The    #
        # loss for each batch is stored in self.train_loss_history. Every      #
        # log_nth iteration the loss is logged. After one epoch the training   #
        # accuracy of the last mini batch is logged and stored in              #
        # self.train_acc_history. We validate at the end of each epoch, log    #
        # the result and store the accuracy of the entire validation set in    #
        # self.val_acc_history.                                                #
        #                                                                      #
        # Your logging could like something like:                              #
        #   ...                                                                #
        #   [Iteration 700/4800] TRAIN loss: 1.452                             #
        #   [Iteration 800/4800] TRAIN loss: 1.409                             #
        #   [Iteration 900/4800] TRAIN loss: 1.374                             #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                            #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                            #
        #   ...                                                                #
        ########################################################################

        num_iterations = iter_per_epoch * num_epochs  # Total number of iterations

        """for t in range(num_iterations):

            running_loss = 0.0

            for i ,  data in enumerate(train_loader, 0):

                inputs, labels = data 

                #Step function
                optim.zero_grad()
                outputs = model.forward(inputs)
                loss = self.loss_func(outputs, Variable(labels.long())
                optim.step()
                self.train_loss_history.append(running_loss)

                running_loss = loss.item()


                self."""

        for epoch in range(num_epochs):
            running_loss = 0.0
            print("epoch: ", epoch)
            for i, data in enumerate(train_loader, 0):

                inputs, labels = data

                optim.zero_grad()

                outputs = model.forward(inputs)
                print("Input: ", inputs)
                print(inputs.dim())
                print(outputs)
                print(outputs.dim())
                assert(1 == 0)
                loss = self.loss_func(outputs, Variable(labels.long()))
                loss.backward()
                optim.step()

                running_loss = loss.item()

                self.train_loss_history.append(running_loss)

               # if (i % log_nth == 0):
               #     print('(Iteration %d / %d) loss: %f' % (i + 1, iter_per_epoch, running_loss))

                # first_it = (i == 0)
                # last_it = (i == num_iterations + 1)

                if i % iter_per_epoch == iter_per_epoch - 1:  # print every 2000 mini-batches
                    print('[Epoch %d, Iteration %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss))

                    correct = 0
                    total = 0
                    val_correct = 0
                    val_total = 0
                    """total = 0
                    with torch.no_grad():
                        for data in testloader:
                            images, labels = data
                            outputs = net(images)"""
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (Variable(predicted.long()) == Variable(labels.long())).sum().item()

                    for _, val_data in enumerate(val_loader, 0):
                        val_images, val_labels = val_data
                        val_outputs = model.forward(val_images)
                        _, val_predicted = torch.max(val_outputs.data, 1)
                        val_total += val_labels.size(0)
                        val_correct += (Variable(val_predicted.long()) == Variable(val_labels.long())).sum().item()

                        val_loss = self.loss_func(val_outputs, Variable(val_labels.long()))

                    self.val_acc_history.append(val_correct / val_total)
                    self.val_loss_history.append(val_loss)

                    print('Val acc/loss: %3f/%3f' % (self.val_acc_history[-1], self.val_loss_history[-1]))

                    self.train_acc_history.append(correct / total)

                    print('Training Acc/loss: %3f/%3f ' % (
                        self.train_acc_history[-1], self.train_loss_history[-1]))

                    running_loss = 0.0
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        print('FINISH.')
