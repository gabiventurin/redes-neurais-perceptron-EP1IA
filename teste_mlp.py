from Mlp import Mlp
import numpy as np

rede = Mlp(
    n_inputs=63,
    n_hidden=10,
    n_outputs=7
)

x = np.random.rand(63)

y = rede.forward(x)

print(y)
print(y.shape)