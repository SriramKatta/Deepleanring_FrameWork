import torch as t
from sklearn.metrics import f1_score


class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):
        self.model_val = model
        self.crit_val = crit
        self.optimezer_val = optim
        self.train_dataset = train_dl
        self.validation_dataset = val_test_dl
        self.cuda_val = cuda

        self.early_stopping_patience_val = early_stopping_patience

        if cuda:
            self.model_val = model.cuda()
            self.crit_val = crit.cuda()
            
    def save_checkpoint(self, epoch):
        t.save({'state_dict': self.model_val.state_dict()}, 'checkpoints/checkpoint.ckp')
    
    def restore_checkpoint(self, epoch_n, path='checkpoints'):
        ckp = t.load(f'{path}/checkpoint.ckp', 'cuda' if self.cuda_val else None)
        
    def save_onnx(self, fn):
        m = self.model_val.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self.model_val(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})
        if self.cuda_val:
            m = self.model_val.cuda()
            
    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients. By default, PyTorch accumulates (sums up) gradients when backward() is called. This behavior is not required here, so you need to ensure that all the gradients are zero before calling the backward.
        # -propagate through the network
        # -calculate the loss
        # -compute gradient by backward propagation
        # -update weights
        # -return the loss
        #TODO
        self.optimezer_val.zero_grad()
        out = self.model_val(x)
        loss = self.crit_val(out, y.float())
        loss.backward()
        self.optimezer_val.step()
        return loss.item()
        
        
    
    def val_test_step(self, x, y):
        # predict
        # propagate through the network and calculate the loss and predictions
        # return the loss and the predictions
        #TODO
        out = self.model_val(x)
        loss = self.crit_val(out, y.float())
        out = out.detach().cpu().numpy()
        pred_0 = np.array(out[:, 0] > 0.5).astype(int)
        pred_1 = np.array(out[:, 1] > 0.5).astype(int)
        pred = np.stack([pred_0, pred_1], axis=1)
        return loss.item(), pred
        
    def train_epoch(self):
        # set training mode
        # iterate through the training set
        # transfer the batch to "cuda()" -> the gpu if a gpu is given
        # perform a training step
        # calculate the average loss for the epoch and return it
        #TODO
        self.model_val = self.model_val.train()
        avg_loss = 0
        for x, y in self.train_dataset:
            if self.cuda_val:
                x = x.cuda()
                y = y.cuda()
            loss = self.train_step(x, y)
            avg_loss += loss / len(self.train_dataset)
        return avg_loss

    
    def val_test(self):
        # set eval mode. Some layers have different behaviors during training and testing (for example: Dropout, BatchNorm, etc.). To handle those properly, you'd want to call model.eval()
        # disable gradient computation. Since you don't need to update the weights during testing, gradients aren't required anymore. 
        # iterate through the validation set
        # transfer the batch to the gpu if given
        # perform a validation step
        # save the predictions and the labels for each batch
        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        # return the loss and print the calculated metrics
        #TODO
        self.model_val = self.model_val.eval()
        with t.no_grad():
            avg_loss = 0
            preds = []
            labels = []
            for x, y in self.validation_dataset:
                if self.cuda_val:
                    x = x.cuda()
                    y = y.cuda()
                loss, pred = self.val_test_step(x, y)
                avg_loss += loss / len(self.validation_dataset)
                if self.cuda_val:
                    y = y.cpu()
                pred = pred
                preds.extend(pred)
                labels.extend(y)
            score = f1_score(labels, preds, average='micro')
        return avg_loss, score
        
    
    def fit(self, epochs=-1):
        assert self.early_stopping_patience_val > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch 
        #TODO
        # stop by epoch number
        # train for a epoch and then calculate the loss and metrics on the validation set
        # append the losses to the respective lists
        # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
        # check whether early stopping should be performed using the early stopping criterion and stop if so
        # return the losses for both training and validation
        #TODO
        train_losses = []
        val_losses = []
        epoch = 0
        while True:
            if epoch == epochs:
                break
            print('Epoch: ',(epoch+1))
            train_loss = self.train_epoch()
            val_loss, _ = self.val_test()
            
            if len(val_losses) != 0 and val_loss < min(val_losses):
                self.save_checkpoint(epoch)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if self.early_stopping_patience_val > 0:
                if len(val_losses) > self.early_stopping_patience_val:
                    if val_losses[-1] > val_losses[-self.early_stopping_patience_val-1]:
                        break
            epoch += 1
        return train_losses, val_losses
