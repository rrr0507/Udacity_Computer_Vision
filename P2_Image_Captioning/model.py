import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.num_layers = num_layers
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.LSTM = nn.LSTM(self.embed_size, self.hidden_size, self.num_layers, batch_first=True)
        self.Linear = nn.Linear(self.hidden_size, self.vocab_size)
    def forward(self, features, captions):
        captions_ = captions[:, :-1]
        embedding = self.embed(captions_)
        embedding = torch.cat((features.unsqueeze(dim = 1), embedding), dim = 1)
        out, hidden = self.LSTM(embedding)
        return self.Linear(out)

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sampled_list = []
        states = (torch.zeros((self.num_layers, 1, self.hidden_size), device=inputs.device),
                torch.zeros((self.num_layers, 1, self.hidden_size), device=inputs.device))
        for i in range(max_len):                                    
            out, states = self.LSTM(inputs, states)        # (batch_size, 1, hidden_size),
            output = self.Linear(out)          # (batch_size, vocab_size)
            output = output.squeeze(1)
            pred_index = output.argmax(dim=1)
            sampled_list.append(pred_index.item())
            inputs = self.embed(pred_index)
            inputs = inputs.unsqueeze(1)                       # (batch_size, 1, embed_size)
        
        return sampled_list