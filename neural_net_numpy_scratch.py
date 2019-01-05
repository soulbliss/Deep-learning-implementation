import numpy as np

N, D_in, H, D_out = 64, 1000, 100, 10

x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

w1 = np.random.randn(D_in, H)  # (1000, 100)
w2 = np.random.randn(H, D_out) # (100, 10) 

learning_rate = 1e-6

for t in range(500):
    
    z = x.dot(w1)
    activation = np.maximum(z, 0)
    y_pred = activation.dot(w2)
    
    loss = np.square(y_pred - y).sum()
    print(t, loss)
    
    cost_activation_deri = 2.0 * (y_pred - y)
    deri_w2 = activation.T.dot(cost_activation_deri) 
    deri_activation = cost_activation_deri.dot(w2.T) 
    deri_z = deri_activation.copy()
    deri_z[z < 0] = 0
    deri_w1 = x.T.dot(deri_z)
    
    w1 -= learning_rate * deri_w1
    w2 -= learning_rate * deri_w2
