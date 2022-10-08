import time
import torch
import numpy as np

import preprocessing
import performance_measures

torch.random.manual_seed(0)

class lstm(torch.nn.Module):
    def __init__(self,num_features,forecast_steps,hidden_units):
        super().__init__()
        self.hidden_units=hidden_units
        self.lstm=torch.nn.LSTM(input_size=num_features,hidden_size=self.hidden_units,batch_first=True)
        self.out=torch.nn.Linear(in_features=self.hidden_units,out_features=forecast_steps)
    def forward(self,x):
        batch_size=x.shape[0]
        ho=torch.zeros(1,batch_size,self.hidden_units)
        co=torch.zeros(1,batch_size,self.hidden_units)
        ho=ho.to('cuda')
        co=co.to('cuda')
        x_,(h_,c_)=self.lstm(x,(ho,co))
        h_=h_.view(-1,self.hidden_units)
        out=self.out(h_)
        return out

def train_lstm(x,input_steps,output_steps,hidden_units,epochs=100):
    train_x,train_y=preprocessing.form_input_output_pairs(x,input_steps,output_steps)
    num_features=np.shape(train_x)[2]
    train_x=torch.autograd.Variable(torch.Tensor(train_x))
    train_y=torch.autograd.Variable(torch.Tensor(train_y))
    train_x=train_x.to('cuda')
    train_y=train_y.to('cuda')
    model=lstm(num_features,output_steps,hidden_units)
    model.to('cuda')
    optimiser=torch.optim.Adam(model.parameters(),lr=0.01)
    criterion=torch.nn.MSELoss()
    st=time.time()
    for epoch in range(epochs):
        optimiser.zero_grad()
        predictions=model(train_x)
        current_loss=criterion(predictions,train_y)
        if epoch%100==0:
            print('epoch',epoch,':',current_loss)
        current_loss.backward()
        optimiser.step()
    en=time.time()
    print('time taken for training:',en-st)
    train_y=train_y.cpu().numpy().flatten()
    model.eval()
    with torch.no_grad():
        train_predictions=model(train_x).cpu().numpy().flatten()
        print('training mape:',performance_measures.mape(train_predictions,train_y))
        test_x=torch.Tensor(np.array([np.reshape(x[-1*input_steps:],(input_steps,1))])).to('cuda')
        prediction=model(test_x).cpu().numpy().flatten()
    validate_lstm_use(model,train_x)
    return model,prediction

def validate_lstm_use(model,train_x):
    model.eval()
    with torch.no_grad():
        batch_train_predictions=model(train_x).cpu().numpy()
        for i in range(len(train_x)):
            input_x=torch.Tensor(np.array([train_x[i].cpu().numpy()])).to('cuda')
            individual_train_prediction=model(input_x).cpu().numpy()
            if np.average(np.abs(individual_train_prediction-batch_train_predictions[i]))>0.1:
                print('error in use validation, please check.')
                print('sample',i,'prediction individually:',individual_train_prediction,'prediction in batch:',batch_train_predictions[i])
                exit()
    return 1

