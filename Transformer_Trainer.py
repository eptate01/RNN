import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_batch(split):
    if split == 'train':
        data = train_data
    else:
        data = test_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) #get a random offset into the data to start a batch of data at (for each batch)
    x = torch.stack([data[i:i+block_size]for i in ix]) #get the input data for each batch
    y = torch.stack([data[i+1:i+block_size+1]for i in ix]) #get the target data
    return x,y




with open('tiny-shakespeare.txt', 'r') as f:
    text = f.read()

chars = sorted(list(set(text))) #create a set of all the unique characters
vocab_size = len(chars) # get the amount of unique characters

#We wil stick to a small tokenizer (character level tokenizer) for this project. 
    #This means our dictionary of is from each unique integer to number instead of each unique combo of two integers two one number and so on

#create two dictionaries that map from each character to a unique number and vice versa
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s] #encoder: for a given string, output the integers
decoder = lambda l: ''.join([itos[i] for i in l]) #decoder: for a list of integers, output the string

#create a tensor of numbers to represent the shakespeare document (using teh encoder)
data = torch.tensor(encode(text), dtype=torch.long).to(device)

#create a training a testing difference in data sets by training on ~90% and testing on ~10%
n = int(0.9*len(data))
train_data = data[:n]
test_data = data[n:]

#crate batch and block size for how many sequences we run and how much context each sequence has
batch_size = 4
block_size = 8


xb, yb = get_batch('train')