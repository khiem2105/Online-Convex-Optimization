import numpy as np
from commons import *
from gd import *
from mirror_descent import *
from acceleration import *
from exploration import *
from prox_svrg import *

import matplotlib.pyplot as plt
plt.style.use("ggplot")

x, accs = gd()
x1, accs1 = gd(z=100, projected=True)
x4, accs4 = sgd(projected=True)
x6, accs6 = smd()
x7, accs7 = seg()
x8, accs8 = ada_grad(z=100)
x9, accs9 = ons()
x10, accs10 = sreg()
x11, accs11 = sbeg()
x12, accs12 = prox_svrg(lr=0.01)

plt.figure(figsize=(12, 8))
plt.plot(range(len(accs12)), accs12, label="Prox-SVRG")
plt.plot(range(len(accs8)), accs8, label="AdaGrad", linestyle="dashed")
plt.plot(range(len(accs1)), accs1, label="GDproj", linestyle="dashed")
plt.plot(range(len(accs4)), accs4, label="SGDproj", linestyle="dashed")
plt.plot(range(len(accs6)), accs6, label="SMD", linestyle="dashed")
plt.plot(range(len(accs7)), accs7, label="SEG +/-", linestyle="dashed")
plt.plot(range(len(accs)), accs, label="GD")
plt.plot(range(len(accs9)), accs9, label="ONS")
plt.plot(range(len(accs10)), accs10, label="SREG +/-")
plt.plot(range(len(accs11)), accs11, label="SBEG +/-")

plt.xlabel("Number of evaluation")
plt.ylabel("Accuracy")
plt.legend()
plt.show()