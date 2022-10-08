import lstm
import torch
import numpy as np
import performance_measures

if __name__=='__main__':

    #******replace this part with the actual time series******
    #generate a series for forecasting
    x=np.arange(0,np.pi*6,0.1)
    y=np.abs(np.sin(x))*100
    input_length=9
    output_length=3
    input_time_series=np.reshape(y[:-1*output_length],(len(y[:-1*output_length]),1))

    #automated modelling
    hidden_units=64
    model,prediction=lstm.train_lstm(input_time_series,input_length,output_length,hidden_units,epochs=10000)

    #study the performance
    print('prediction:',prediction)
    print('actual:',y[-1*output_length:])
    print('mape from forecasting the last few points of the series:',performance_measures.mape(prediction,y[-1*output_length:]))

