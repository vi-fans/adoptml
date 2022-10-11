import numpy as np

def form_input_output_pairs(x,input_steps,output_steps):
    train_x=[]
    train_y=[]
    for i in range(len(x)-input_steps-output_steps+1):
        train_x.append(x[i:i+input_steps,:])
        train_y.append(x[i+input_steps:i+input_steps+output_steps,0])
    train_x=np.array(train_x)
    train_y=np.array(train_y)
    return np.array(train_x),np.array(train_y)

