import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from model import SingleLayerNet


model = SingleLayerNet(1, 2, 1)

# Cargar el estado del modelo guardado
model.load_state_dict(torch.load('model.pth'))

# Función para imprimir pesos y sesgos
def print_weights_and_biases(model):
    hidden_weights = model.hidden_layer.weight.data.numpy()
    hidden_biases = model.hidden_layer.bias.data.numpy()
    output_weights = model.output_layer.weight.data.numpy()
    output_biases = model.output_layer.bias.data.numpy()
    
    print("Pesos de la capa oculta:")
    print(hidden_weights)
    print("Sesgos de la capa oculta:")
    print(hidden_biases)
    
    print("Pesos de la capa de salida:")
    print(output_weights)
    print("Sesgos de la capa de salida:")
    print(output_biases)

# Imprimir los pesos y sesgos
print_weights_and_biases(model)