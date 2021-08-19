import torch.optim as optim
from model import *
import util

class trainer():
    def __init__(self, scaler, args, adj, global_train_steps):
        lr_new = util.lr_new(args, global_train_steps)
        self.model = stsgcn(args, adj)
        self.model.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda =lambda num_update:(lr_new.update(num_update)/args.learning_rate))
        self.loss = util.huber_loss
        self.scaler = scaler


    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input)
        real = torch.unsqueeze(real_val,dim=3)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, rho=1, null_val=0.0)
        loss.backward()
        self.optimizer.step()
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mae,mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        output = self.model(input)
        real = torch.unsqueeze(real_val,dim=3)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, rho=1, null_val=0.0)
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mae,mape,rmse
