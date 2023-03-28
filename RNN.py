import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np

sentences =[]

with open('tiny-shakespeare.txt', "r") as file:
        data = file.readlines()
        currLine = []
        for line in data:
            line.replace("\n", '')
            if line:
                currLine.append(line)
            else:
                sentences.append(' '.join(currLine))
                currLine = []
        sentences.append(' '.join(currLine))


with open('tiny-shakespeare.txt', "r") as file:
        data = file.readlines()
        for i in range(0,len(data)):
            data[i] = data[i].rstrip("\n")
sentences = [x for x in data if x != '']
#sentences = sentences[:8000]


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

def one_hot(sequence, dictionary_size):
    encoding = np.zeros((1,len(sequence), dictionary_size), dtype=np.float32)
    for i in range(len(sequence)):
        encoding[0 ,i, sequence[i]] = 1
    return encoding

class RNNModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(RNNModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        #Define the network!
		#Batch first defines where the batch parameter is in the tensor
        #self.embedding = nn.embedding(...)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_size, output_size)
        

    def forward(self, x):
        hidden_state = self.init_hidden()
        output,hidden_state = self.rnn(x, hidden_state)
		#Shouldn't need to resize if using batches, this eliminates the first dimension
        output = output.contiguous().view(-1, self.hidden_size)
        output = self.fc(output)
        return output, hidden_state
        
    def init_hidden(self):
        #Hey,this is our hidden state. Hopefully if we don't have a batch it won't yell at us
        #Also a note, pytorch, by default, wants the batch index to be the middle dimension here. 
        #So it looks like (row, BATCH, column)
        hidden = torch.zeros(self.num_layers, 1, self.hidden_size)
        return hidden


# Main Code
dictionary_size = len(charInt)
one_hot(input_sequence[0], dictionary_size)

model = RNNModel(dictionary_size, dictionary_size, 100, 1)

#Define Loss
loss = nn.CrossEntropyLoss()

#Use Adam again
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(1):
    for i in range(len(input_sequence)):
        optimizer.zero_grad()
        x = torch.from_numpy(one_hot(input_sequence[i], dictionary_size))
        #print(x)
        y = torch.Tensor(target_sequence[i])
        #print(y)
        output, hidden = model(x)
        
        #print(output)
        #print(hidden)
        #print(output.size())
        #print(x.view(-1).long().size())
        lossValue = loss(output, y.view(-1).long())
		#Calculates gradient
        lossValue.backward()
		#Updates weights
        optimizer.step()
        
        #print("Loss: {:.4f}".format(lossValue.item()))

def predict(model, character):
    
    characterInput = np.array([charInt[c] for c in character])
    #print(characterInput)
    characterInput = one_hot(characterInput, dictionary_size)
    #print(characterInput)
    characterInput = torch.from_numpy(characterInput)
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
    
print(sample(model, 50))