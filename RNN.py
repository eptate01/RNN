import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import time

BATCH_SIZE = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Pytorch CUDA Version is ", torch.version.cuda)
def one_hot(sequence, dictionary_size):
    encoding = np.zeros((BATCH_SIZE,len(sequence), dictionary_size), dtype=np.float32)
    for i in range(len(sequence)):
        encoding[0 ,i, sequence[i]] = 1
    return encoding

class RNNModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_size, output_size)
        

    def forward(self, x):
        hidden_state = self.init_hidden()
        output,hidden_state = self.rnn(x, hidden_state)
        #output = output.contiguous().view(-1, self.hidden_size)
        output = self.fc(output)
        return output, hidden_state
        
    def init_hidden(self):
        hidden = torch.zeros(self.num_layers, BATCH_SIZE, self.hidden_size).to(device)
        return hidden

def predict(model, character):
    
    characterInput = np.array([charInt[c] for c in character])
    #print(characterInput)
    characterInput = one_hot(characterInput, dictionary_size)
    #print(characterInput)
    characterInput = torch.from_numpy(characterInput).to(device)
    #print(character)
    out, hidden = model(characterInput)
    
    #Get output probabilities
    
    prob = nn.functional.softmax(out[-1], dim=0).data
    #print(prob)
    character_index = torch.max(prob, dim=0)[1].item()
    
    return intChar[character_index], hidden
    
def sample(model, out_len, start='MENENIUS'):
    characters = [ch for ch in start]
    currentSize = out_len - len(characters)
    for i in range(currentSize):
        character, hidden_state = predict(model, characters)
        characters.append(character)
        
    return ''.join(characters)


with open('tiny-shakespeare.txt', 'r') as file:
    text = file.read()
sentences = [text[i:i+100] for i in range(0, len(text), 100)]


#Step 1, extract all characters
characters = set(''.join(sentences))
#print(characters)

#create dictionaries of characters
intChar = dict(enumerate(characters))
#print(intChar)
charInt = {character: index for index, character in intChar.items()}

#offset input and output sentences
input_sequence = []
target_sequence = []
for i in range(len(sentences)):
    #Remove the last character from the input sequence
    input_sequence.append(sentences[i][:-1])
    #Remove the first element from target sequences
    target_sequence.append(sentences[i][1:])

max_length = len(max(input_sequence, key = len))
for i in range(0,len(input_sequence)):
    input_sequence[i] = input_sequence[i].zfill(max_length)
for i in range(0,len(target_sequence)):
    target_sequence[i] = target_sequence[i].zfill(max_length)
input_sequence = [i.replace('0', intChar[0]) for i in input_sequence]
target_sequence = [i.replace('0', intChar[0]) for i in target_sequence]


#make one hot keys
for i in range(len(sentences)):
    input_sequence[i] = [charInt[character] for character in input_sequence[i]]
    target_sequence[i] = [charInt[character] for character in target_sequence[i]]


# Main Code
dictionary_size = len(charInt)
one_hot(input_sequence[0:BATCH_SIZE-1], dictionary_size)


model = RNNModel(dictionary_size, dictionary_size, 100, 1).to(device)

#Define Loss
loss = nn.CrossEntropyLoss()

#Use Adam again
optimizer = torch.optim.Adam(model.parameters())
start = time.time()
for epoch in range(3):
    for i in range(0,len(input_sequence),BATCH_SIZE):
        optimizer.zero_grad()
        x = torch.from_numpy(one_hot(input_sequence[i], dictionary_size)).to(device)
        y = torch.Tensor(target_sequence[i:i+BATCH_SIZE]).to(device)
        if (len(target_sequence[i:i+BATCH_SIZE]) == BATCH_SIZE):
            hidden = model.init_hidden()
            output, hidden = model(x)
            #output = output.view(BATCH_SIZE,max_length,65)
            print(output.size())
            print(y.size())
            y =y.view(BATCH_SIZE, -1).long()
            print(y.size())
            #print(y.long())
            lossValue = loss(output, y).mean(dim=0)
            #Calculates gradient
            lossValue.backward()
            #Updates weights
            optimizer.step()
            if i%1 == 0:
                print("Loss: {:.4f}".format(lossValue.item()))

end = time.time()
print(end - start)
    
print(sample(model, 50))