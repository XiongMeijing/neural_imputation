from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, input_size, code_size, hidden_sizes=[], activation=nn.ReLU()):
        super(Encoder, self).__init__()

        encoder_layers = []
        previous_layer_size = input_size

        layer_sizes = list(hidden_sizes) + [code_size]

        for layer_size in layer_sizes:
            encoder_layers.append(nn.Linear(previous_layer_size, layer_size))
            encoder_layers.append(activation)
            previous_layer_size = layer_size

        self.layers = nn.Sequential(*encoder_layers)

    def forward(self, inputs):
        return self.layers(inputs)


class Decoder(nn.Module):

    def __init__(self, code_size, output_size, hidden_sizes=[], activation=nn.ReLU()):
        super(Decoder, self).__init__()

        previous_layer_size = code_size

        layer_sizes = list(hidden_sizes) + [output_size]
        decoder_layers = []
        for layer_size in layer_sizes:
            decoder_layers.append(nn.Linear(previous_layer_size, layer_size))
            decoder_layers.append(activation)
            previous_layer_size = layer_size

        self.layers = nn.Sequential(*decoder_layers)

    def forward(self, code, training=False):
        return self.layers(code)
