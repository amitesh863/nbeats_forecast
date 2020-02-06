import os
import matplotlib.pyplot as plt
import torch
from torch import optim
from torch.nn import functional as F
import numpy as np
from nbeats_pytorch.model import NBeatsNet 





class NBeats:   #UNIVARIATE DATA TO BE PASSED AS NUMPY ARRAY
    def __init__(self,period_to_forecast,data=None, backcast_length=None,save_checkpoint=False,path='',checkpoint_file_name='nbeats-training-checkpoint.th',mode='cpu',batch_size=None,thetas_dims=[7, 8],nb_blocks_per_stack=3,share_weights_in_stack=False,train_percent=0.8,hidden_layer_units=128,stack=None):
        if (data is None):
            print('For Prediction as no data passed')
            batch_size=np.nan
        else:
            self.data=data
            
            if(len(self.data.shape)!=2):
                raise Exception('Numpy array should be of nx1 shape')
            if self.data.shape[1]!=1:
                raise Exception('Numpy array should be of nx1 shape')
        self.forecast_length=period_to_forecast
        if backcast_length==None:
            self.backcast_length = 3 * self.forecast_length
        else:
            self.backcast_length=backcast_length
        if batch_size==None:
            self.batch_size=int(self.data.shape[0]/15)
        else:
            self.batch_size=batch_size
        self.hidden_layer_units=hidden_layer_units  
        
        if stack==None:
            self.stacks= [NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK]
        else:
            Dict = {1: NBeatsNet.GENERIC_BLOCK, 2: NBeatsNet.TREND_BLOCK , 3: NBeatsNet.SEASONALITY_BLOCK} 
            self.stacks = [Dict.get(item, item) for item in stack]

        

        self.CHECKPOINT_NAME= path+checkpoint_file_name
        self.device = torch.device(mode)       #change to gpu if gpu present for better performance
        self.thetas_dims=thetas_dims
        self.nb_blocks_per_stack=nb_blocks_per_stack
        self.train_size=train_percent
        self.share_weights_in_stack=share_weights_in_stack

        self.net = NBeatsNet(stack_types=self.stacks,
                        forecast_length=self.forecast_length,
                        thetas_dims=self.thetas_dims,
                        nb_blocks_per_stack=self.nb_blocks_per_stack,
                        backcast_length=self.backcast_length,
                        hidden_layer_units=self.hidden_layer_units,
                        share_weights_in_stack=self.share_weights_in_stack,
                        device=self.device)
        self.parameters=self.net.parameters()
        self.global_step_cl=0
        self.check_save=save_checkpoint
        self.loaded=False
        self.saved=True
        
        
        
    def plot_scatter(self,*args, **kwargs):
        plt.plot(*args, **kwargs)
        plt.scatter(*args, **kwargs)
    
    
    def data_generator(self,x_full, y_full, bs):
        def split(arr, size):
            arrays = []
            while len(arr) > size:
                slice_ = arr[:size]
                arrays.append(slice_)
                arr = arr[size:]
            arrays.append(arr)
            return arrays
    
        while True:
            for rr in split((x_full, y_full), bs):
                yield rr    
                
    def loader(self,model, optimiser):
        if os.path.exists(self.CHECKPOINT_NAME):
            checkpoint = torch.load(self.CHECKPOINT_NAME)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
            grad_step = checkpoint['grad_step']
            if self.loaded:
                self.norm_constant=checkpoint['norm_constant']
            return grad_step
        return 0
    
    def saver(self,model, optimiser, grad_step):
        if self.saved:
                 torch.save({
                'norm_constant':self.norm_constant, 
                'grad_step': grad_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimiser.state_dict(),
            }, self.CHECKPOINT_NAME) 
        else:            
            torch.save({
                'grad_step': grad_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimiser.state_dict(),
            }, self.CHECKPOINT_NAME) 
    
    
    def load(self,file=None,optimiser=None):
         if file==None:
            raise Exception('Empty File Name')
         elif file=='':
            raise Exception('Empty File Name') 
         else:
           if optimiser==None:
            self.optimiser = optim.Adam(self.net.parameters())
           else:
               self.optimiser = optimiser
           self.CHECKPOINT_NAME=file
           self.loaded=True
           self.global_step_cl=self.loader(self.net,self.optimiser)
           self.CHECKPOINT_NAME='nbeats-training-checkpoint.th'
        
            
        
    def save(self,file=None):
        if file==None:
            raise Exception('Empty File Name')
        elif file=='':
            raise Exception('Empty File Name')            
        else:
            self.CHECKPOINT_NAME=file
            self.saver(self.net,self.optimiser,self.global_step_cl)
            self.saved=True

           
    def train_100_grad_steps(self,data, test_losses):
        if not self.loaded:
            global_step = self.loader(self.net, self.optimiser)
            self.loaded=False
            self.global_step_cl=global_step
        for x_train_batch, y_train_batch in data:
            self.global_step_cl += 1
            self.optimiser.zero_grad()
            self.net.train()
            _, forecast = self.net(torch.tensor(x_train_batch, dtype=torch.float).to(self.device))
            loss = F.mse_loss(forecast, torch.tensor(y_train_batch, dtype=torch.float).to(self.device))
            loss.backward()
            self.optimiser.step()
            if self.verbose:
                if self.global_step_cl % 30 == 0:
                    print(f'grad_step = {str(self.global_step_cl).zfill(6)}, tr_loss = {loss.item():.6f}, te_loss = {test_losses[-1]:.6f}')
            if self.global_step_cl > 0 and self.global_step_cl % 100 == 0:
                with torch.no_grad():
                   self.saver(self.net, self.optimiser, self.global_step_cl)
                break
        return forecast   
    
    def eval_test(self, test_losses, x_test, y_test):
        self.net.eval()
        _, forecast = self.net(torch.tensor(x_test, dtype=torch.float))
        test_losses.append(F.mse_loss(forecast, torch.tensor(y_test, dtype=torch.float)).item())
        p = forecast.detach().numpy()
        if self.plot:
            subplots = [221, 222, 223, 224]
            plt.figure(1)
            for plot_id, i in enumerate(np.random.choice(range(len(x_test)), size=4, replace=False)):
                ff, xx, yy = p[i] * self.norm_constant, x_test[i] * self.norm_constant, y_test[i] * self.norm_constant
                plt.subplot(subplots[plot_id])
                plt.grid()
                self.plot_scatter(range(0, self.backcast_length), xx, color='b')
                self.plot_scatter(range(self.backcast_length, self.backcast_length + self.forecast_length), yy, color='g')
                self.plot_scatter(range(self.backcast_length, self.backcast_length + self.forecast_length), ff, color='r')
            plt.show()
        
        
    def fit(self,epoch=25,optimiser=None,plot=False,verbose=True):
        self.plot=plot
        self.verbose=verbose

        self.epoch=epoch
        self.norm_constant = np.max(self.data)
        self.data = self.data / self.norm_constant 
        x_train_batch, y,self.x_forecast = [], [] ,[]
        for i in range(self.backcast_length, len(self.data) - self.forecast_length):
            x_train_batch.append(self.data[i - self.backcast_length:i])
            y.append(self.data[i:i + self.forecast_length])
        i+=self.forecast_length
        self.x_forecast.append(self.data[i+1 - self.backcast_length:i+1])
        x_train_batch = np.array(x_train_batch)[..., 0]
        self.x_forecast = np.array(self.x_forecast)[..., 0]

        y = np.array(y)[..., 0]
        i+=self.forecast_length

        if optimiser==None:
            self.optimiser = optim.Adam(self.net.parameters())
        else:
            self.optimiser = optimiser
        
        c = int(len(x_train_batch) * self.train_size)
        x_train, y_train = x_train_batch[:c], y[:c]
        x_test, y_test = x_train_batch[c:], y[c:]

        train_data = self.data_generator(x_train, y_train, self.batch_size)
        test_losses = []
        for i in range(self.epoch):
            self.eval_test(test_losses, x_test, y_test)
            self.train_100_grad_steps(train_data, test_losses)
        if self.check_save:
            pass
        else:
            if os.path.exists(self.CHECKPOINT_NAME):
                os.remove(self.CHECKPOINT_NAME)
            
    def predict(self,predict_data=None):        
        if (predict_data is None):
            _, forecasted_values = self.net(torch.tensor(self.x_forecast, dtype=torch.float))
            forecasted_values= forecasted_values.detach().numpy()
            forecasted_values = forecasted_values * self.norm_constant
        else:
            if (predict_data.shape[0]!=self.backcast_length):
                raise Exception('Numpy array for prediction input should be of backcast_length: {} x 1 shape'.format(self.backcast_length))
            else:
                predict_data=predict_data/self.norm_constant
                predict_data= np.reshape(predict_data, (self.backcast_length, 1)) 
                
                _, forecasted_values = self.net(torch.tensor(predict_data.T, dtype=torch.float))
                forecasted_values= forecasted_values.detach().numpy()
                forecasted_values = forecasted_values * self.norm_constant
                
        return forecasted_values.T
