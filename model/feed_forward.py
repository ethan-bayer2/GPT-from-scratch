import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, activation_fn=nn.ReLU):
        super(FeedForward, self).__init__()

        # First linear layer (embedding_dim -> hidden_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)

        # Activation function (using ReLU but you can change this in the def)
        self.activation = activation_fn()

        # Second linear layer (hidden_dim -> embedding_dim)
        self.fc2 = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x):
        # Apply first linear transform
        x = self.fc1(x)

        # ReLU (or GeLU)
        x = self.activation(x)

        # Linear transform back down
        x = self.fc2(x)

        return x
