import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import os
import re

os.environ['KMP_DUPLICATE_LIB_OK']='True'

sequenceLength = 100
step = 1
batchSize = 32

def getData(fileName = "tiny-shakespeare.txt"):
    with open(fileName, 'r') as file:
        text = file.read()
    # split text into lines of 100 characters
    lines = [text[i:i+100] for i in range(0, len(text), 100)]
    return lines

def extractData(sentences):
    characters = set(''.join(sentences))
    return characters

def intToCharacter(characters):
    intCharacters= dict(enumerate(characters))
    return intCharacters

def characterToInt(intCharacters):
    characterInt = {character: index for index, character in intCharacters.items()}
    return characterInt

def maxLength(sequence):
    max = 0
    for i in sequence:
        if len(i) > max:
            max = len(i)
    return max

def padding(sequence, intChar):
    max = maxLength(sequence)
    sequence = sequence.zfill(max)
    sequence = sequence.replace("0", intChar[0])
    return (sequence)

def offsetInputAndOutput(sentences, intChar):
    inputSequence = []
    targetSequence = []
    for i in range(len(sentences)):
        inputSequence.append(sentences[i][:-1])
        targetSequence.append(sentences[i][1:])
    return inputSequence, targetSequence

def replaceCharacterWithInt(sentences, inputSequence, targetSequence, characterInt):
    for i in range(len(sentences)):
        inputSequence[i] = [characterInt[character] for character in inputSequence[i]]
        targetSequence[i] = [characterInt[character] for character in targetSequence[i]]

    vocabularySize = len(characterInt)
    return inputSequence, targetSequence, vocabularySize

def createOneHot(sequence, vocabularySize):
    encoding = np.zeros((len(sequence), vocabularySize), dtype=np.float32)
    for i in range(len(sequence)):
        encoding[i, sequence[i]] = 1
    return encoding

class RNNModel(nn.Module):
    def __init__(self, inputSize, outputSize, hiddenSize, numberOfLayers):
        super(RNNModel, self).__init__()
        self.hiddenSize = hiddenSize
        self.numberOfLayers = numberOfLayers
        self.rnn = nn.RNN(inputSize, hiddenSize, numberOfLayers, batch_first=True)
        self.fc = nn.Linear(hiddenSize, outputSize)

    def forward(self, x, hidden):
        output, hidden = self.rnn(x, hidden)
        output = output.contiguous().view(-1, self.hiddenSize)
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, batchSize):
        hidden = torch.zeros(self.numberOfLayers, batchSize, self.hiddenSize)
        return hidden

def calculateLoss(inputSequence, targetSequence, vocabularySize, model, optimizer):
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    for epoch in range(10):
        for i in range(0, len(inputSequence), batchSize):
            optimizer.zero_grad()
            batchInput = inputSequence[i:i+batchSize]
            batchTarget = targetSequence[i:i+batchSize]

            x = torch.from_numpy(np.array([createOneHot(seq, vocabularySize) for seq in batchInput]))
            y = torch.LongTensor([seq for seq in batchTarget]).view(-1)
            hidden = model.init_hidden(batchSize)

            # Compute model output
            logits, hidden = model(x, hidden)
            # Calculate loss
            loss = loss_fn(logits, y)
            total_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()

            print("Loss: ", (loss.item()))
        
sentences = getData()
#max = maxLength(sentences)
#padding(sentences, max)
characters = extractData(sentences)
intCharacters = intToCharacter(characters)
charactersInt = characterToInt(intCharacters)
inputSequence, targetSequence = offsetInputAndOutput(sentences, intCharacters)
inputSequence, targetSequence, vocabularySize = replaceCharacterWithInt(sentences, inputSequence, targetSequence, charactersInt)
#createOneHot(inputSequence[0], vocabularySize)
model = RNNModel(vocabularySize, vocabularySize, 100, 1)
#optimizer = torch.optim.Adam(model.parameters())
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
calculateLoss(inputSequence, targetSequence, vocabularySize, model, optimizer)
