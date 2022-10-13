import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as pl
from IPython import display

X1 = np.array([[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
X2 = np.flip(X1, axis=1).copy()
z = np.array(([[8.07131, 1730.63, 233.426], [7.43155, 1554.679, 240.337]]))
T = 20
p_water = 10 ** (z[0, 0] - z[0, 1] / (T + z[0, 2]))
p_dioxane = 10 ** (z[1, 0] - z[1, 1] / (T + z[1, 2]))
U = np.array([[28.1, 34.4, 36.7, 36.9, 36.8, 36.7, 36.5, 35.4, 32.9, 27.7, 17.5]])
U = torch.tensor(U, requires_grad=False, dtype=torch.float32)

Z = Variable(torch.tensor([1.0, 1.0]), requires_grad=True)

a1 = torch.tensor(X1, requires_grad=False, dtype=torch.float32)
a2 = torch.tensor(X2, requires_grad=False, dtype=torch.float32)

z = 0.001

for i in range(100):
    P_predicted = a1 * torch.exp(Z[0] * (Z[1] * a2 / (Z[0] * a1 + Z[1] * a2)) ** 2) * p_water + \
        a2 * torch.exp(Z[1] * (Z[0] * a1 / (Z[0] * a1 + Z[1] * a2)) ** 2) * p_dioxane

    loss = (P_predicted - U) ** 2
    loss = loss.sum()

    loss.backward()

    with torch.no_grad():
        Z -= z * Z.grad

        Z.grad.zero_()

print('estimation A12 and A21 is:',Z)
print('final loss is:',loss.data.numpy())

import matplotlib.pyplot as pl
P_predicted = P_predicted.detach().numpy()[0]
U = U.detach().numpy()[0]
a1 = a1.detach().numpy()[0]

pl.plot(a1, P_predicted, label='Predicted Pressure')
pl.plot(a1, U, label='Actual Pressure')
pl.xlabel('a1')
pl.ylabel('Pressure')
pl.legend()
pl.title('Comparison between predicted pressure and actual pressure')
pl.show()
