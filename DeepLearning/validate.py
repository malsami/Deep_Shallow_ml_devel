import torch
from data.utils import load_tensor_data
import torch.utils.data as data_utils
import copy


hidden_size_1 = 100
hidden_size_2 = 100

batch_size = 32


loss_func=torch.nn.CrossEntropyLoss()
#loss_func=torch.nn.MSELoss()
num_epochs=100
log_nth=1000

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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

_, _, x_val, y_val, _, _ = load_tensor_data()

# For data loading
if device == 'cpu':
    val_tensor = data_utils.TensorDataset(x_val, y_val)
else:
    val_tensor = data_utils.TensorDataset(x_val.cuda(), y_val.cuda())

val_data = data_utils.DataLoader(val_tensor, 
                                    batch_size=len(y_val), 
                                    shuffle=True)

model = torch.load('models/trained_model')
model.to(device)

def tell_me_more(currentEpoch, totalEpochs, currentBatch, totalBatches, currentLoss, labels, outputs):
    correct = 0
    correctBAD = 0
    totalBAD = 0
    total = 0
    allowedDaviation = 0.1
    splitIntoParts = 10
    splitList = [0]*splitIntoParts 
    resultSplitList = [0]*splitIntoParts 
    printOut = [(x.item(), out[1].item(), abs(x-out[1]).item()) for x, out in zip(labels, outputs.data)]
    #print(('label, output[1], abs(label-output[1])'))
    for e in printOut:
        total += 1
        if e[0] == 0:
            #print(e)
            totalBAD += 1
        if e[2] < allowedDaviation:
            if e[0] == 0:
                correctBAD += 1
            correct += 1
        for index, threshold in [(x-1,x/splitIntoParts)  for x in range(1, splitIntoParts + 1)]:
            if e[2] < threshold:
                splitList[index] += 1
                break
        for index, threshold in [(x-1,x/splitIntoParts)  for x in range(1, splitIntoParts + 1)]:
            if e[1] < threshold:
                resultSplitList[index] += 1
                break

    
    print('correct / unsuccessful: %d / %d' % (correctBAD, totalBAD))
    for e in printOut:
        if e[0] == 1:
            #print(e)
            pass
    print('correct / successful: %d / %d' % (correct-correctBAD, total-totalBAD))
    
    print('correct / total: %d / %d' % (correct, total))
    
    print('[Epoch %d/%d],(Iteration %d / %d) acc: %f, loss: %f' % (currentEpoch, totalEpochs, currentBatch, totalBatches, (correct / total), currentLoss))
    print('at',allowedDaviation, 'allowed deviation', '\n', splitList, '\n', resultSplitList, '\n')
    

print('####################################################VALIDATION##################################')        
for i, validation_data in enumerate(val_data, 0):
    val_inputs, val_labels = validation_data
    val_outputs = model.forward(val_inputs)
    #print(val_outputs, torch.autograd.Variable(val_labels.long()))
    #cross enthropy
    val_loss = loss_func(val_outputs, torch.autograd.Variable(val_labels.long()))
    
    if (i % log_nth == 0):
        
        tell_me_more(currentEpoch=0, 
                    totalEpochs=0, 
                    currentBatch=(i + 1),
                    totalBatches=1,
                    currentLoss=val_loss,
                    labels=val_labels,
                    outputs=val_outputs)
        
        
    #val_loss = loss_func(val_outputs, torch.autograd.Variable(val_labels.long()))
    #mse
    #val_loss = loss_func(val_outputs, torch.autograd.Variable(val_labels))
    #if (i % log_nth == 0):
    #    print('Val acc/loss: %3f/%3f' % ((val_correct / val_total), val_loss))