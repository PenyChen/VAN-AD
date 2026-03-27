import torch

from data_factory.data_loader import get_train_dataloader,get_test_dataloader, get_val_dataloader
import numpy as np
from vanad.util import TimeLogger
import time
import os
from tqdm import tqdm
from torch.nn.utils import clip_grad_value_
from vanad.vanad_model import VANAD

def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.10f} --> {val_loss:.10f}).  Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


class VANAD_sovler(object):
    def __init__(self, config):
        self.config = config
        if config.use_ddp:
            self.device = torch.device(f"cuda:{config.local_rank}")
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.log_fre = 10    
        self.cri = torch.nn.MSELoss()
        self.model_save_root = "./model_save"
        self.eval_save_root = "./eval_save"
        self.logger = config.logger
        
        if not os.path.exists(self.model_save_root):
            os.makedirs(self.model_save_root)
        if not os.path.exists(self.eval_save_root):
            os.makedirs(self.eval_save_root)
        
        prefix = config.prefix
        weight_file = f"{prefix}_weight.pth"
        eval_file = f"{prefix}_eval.txt"
        self.weight_file = os.path.join(self.model_save_root, weight_file)
        self.eval_file = os.path.join(self.eval_save_root, eval_file)
        
        self.train_loader = get_train_dataloader(self.config)     
        self.vali_loader = get_val_dataloader(self.config)
        self.test_loader = get_test_dataloader(self.config)
        if config.num_vars is None:
            self.config.num_vars = self.train_loader.dataset.data.shape[1]
        self.build_model(config)                
    def build_model(self, config):
            
        self.model = VANAD(config).to(self.device)
        
        self.nf_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
            
    def test(self):
        print("======================TEST MODE======================")
        cri = torch.nn.MSELoss(reduction="none")
        
        print("======================Zero-shot Testing======================") 
            
        self.model.eval()
        self.model.to(self.device)
        input_list = []
        output_list = []
        labels_list = []
        loss_list = []
        
        with torch.no_grad():
            with tqdm(total=len(self.test_loader)*self.config.batch_size,unit='it') as t:
                for i, (inputs, labels) in enumerate(self.test_loader):
                    inputs = inputs.float().to(self.device)
                    outputs, _ = self.model(inputs)
                    loss = torch.mean(cri(outputs, inputs),dim=-1)      # [batch_size, seq_len]
                    input_list.append(inputs.detach().cpu().numpy())
                    output_list.append(outputs.detach().cpu().numpy())

                    loss_list.append(loss.detach().cpu().numpy())
                    labels_list.append(labels.cpu().numpy())
                    t.update(1*self.config.batch_size)
        loss = np.concatenate(loss_list,axis=0).reshape(-1)
        gt = np.concatenate(labels_list,axis=0).reshape(-1)
        print("[labels shape]:",gt.shape)
        print("[loss shape]:",loss.shape)
        thresh = np.percentile(loss, 100 - self.config.anormly_ratio)     
        pred = loss > thresh 
        self.evaluate(gt,pred,loss,thresh)
        return    


    def nf_vali(self):
        # time record
        print("======================VALI MODE======================")
        self.model.eval()
        total_loss = []
        timelog = TimeLogger(self.config, self.vali_loader)
        with torch.no_grad():
            with tqdm(total=len(self.vali_loader)*self.config.batch_size,unit='it') as t:
                for i, (inputs, labels) in enumerate(self.vali_loader):
                    inputs = inputs.float().to(self.device)
                    y_dec, log_prob = self.model(inputs) 
                    loss_rec = self.cri(inputs, y_dec)
                    loss_nf = -log_prob.mean()
                    loss = loss_nf
                    total_loss.append(loss.item())
                    t.update(1*self.config.batch_size)
        avg_loss = np.average(total_loss)
        return avg_loss
    
    def nf_train(self):
        early_stopping = EarlyStopping(patience=5, verbose=True, dataset_name=self.config.data)
        torch.autograd.set_detect_anomaly(True)

        for epoch in range(self.config.epochs):
            if self.config.use_ddp:
                self.train_loader.sampler.set_epoch(epoch)
            print("======================TRAIN MODE======================")
            iter_count = 0
            epoch_time = time.time()
            self.model.train()
            self.model.to(self.device)
            sum_loss = []
            with tqdm(total=len(self.train_loader)*self.config.batch_size,unit='it') as t:
                for i, (inputs, labels) in enumerate(self.train_loader):
                    iter_count += 1
                    inputs = inputs.float().to(self.device)
                    y_dec, log_prob = self.model(inputs)     # [batch_size, seq_len, n_sensor]
                    if self.config.use_nf:
                        loss_rec = self.cri(inputs, y_dec)
                        loss_nf = -log_prob.mean()
                        loss = self.config.lambda_rec*loss_rec + self.config.lambda_nf*loss_nf
                    sum_loss.append(loss.item())
                    self.nf_optimizer.zero_grad()
                    loss.backward()
                    clip_grad_value_(self.model.parameters(), 1)
                    self.nf_optimizer.step()
                    t.update(1*self.config.batch_size)

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(sum_loss)
            print("Epoch: {0}, | Train Loss: {1:.10f} ".format(epoch + 1, train_loss))
            adjust_learning_rate(self.nf_optimizer, epoch + 1, self.config.lr)
            
        # maybe last epoch isn't best, so save here
        torch.save(self.model.state_dict(), self.weight_file.format(epochs=self.config.epochs)) 
    
    def nf_test(self):
        cri = torch.nn.MSELoss(reduction="none")
        print("======================TEST MODE======================")
        
        self.model.load_weights(self.weight_file.format(epochs=self.config.epochs)) 
            
        self.model.eval()
        self.model.to(self.device)
        input_list = []
        output_list = []
        loss_list = []
        rec_score_list = []
        nf_score_list = []
        labels_list = []
        
        start = time.time()
        with torch.no_grad():
            with tqdm(total=len(self.test_loader)*self.config.batch_size,unit='it') as t:
                for i, (inputs, labels) in enumerate(self.test_loader):
                    inputs = inputs.float().to(self.device)

                    y_dec, log_prob = self.model(inputs, flag='test')     # [batch_size, seq_len, n_sensor]
                    if self.config.use_nf:
                        loss_rec = torch.mean(cri(y_dec, inputs),dim=-1)
                        loss_nf = -log_prob
                        loss = self.config.lambda_rec*loss_rec + self.config.lambda_nf*loss_nf
                        rec_score = self.config.lambda_rec*loss_rec
                        nf_score = self.config.lambda_nf*loss_nf
                    
                    loss_list.append(loss.cpu().numpy())
                    rec_score_list.append(rec_score.cpu().numpy())
                    nf_score_list.append(nf_score.cpu().numpy())
                    labels_list.append(labels.cpu().numpy())
                    t.update(1*self.config.batch_size)
        end = time.time()
        print(f"[Test time]: {end - start:.2f} seconds")
        loss = np.concatenate(loss_list,axis=0).reshape(-1)
        gt = np.concatenate(labels_list,axis=0).reshape(-1)
        print("[labels shape]:",gt.shape)
        print("[loss shape]:",loss.shape)
        thresh = np.percentile(loss, 100 - self.config.anormly_ratio)     
        pred = loss > thresh 
        self.evaluate(gt,pred,loss,thresh)
        return    
       
    def evaluate(self,gt,pred,loss,thresh):
        from utils.evaluate import eval_method,evaluate,compute_vus
        vus_roc, vus_pr = compute_vus(gt, loss)
        
        accuracy,precision,recall,f_score,auc_roc,auc_pr = eval_method(gt,pred,loss)
        res = evaluate(init_score=pred,
                        test_score=loss,
                        test_label=gt,
                        threshold=thresh)       
        accuracy  = f"[Accuracy]: {accuracy:.4f}"
        precision,recall,f_score = res["precision"],res["recall"],res["f1_score"]
        # affiliation_res = f"[Affiliation]: Precision: {precision:.4f}, Recall: {recall:.4f}, F-score: {f_score:.4f}"
        auc  = f"[AUC-ROC]: {auc_roc:.4f}, [AUC-PR]: {auc_pr:.4f}"
        vus = f"[VUS-ROC]: {vus_roc:.4f}, [VUS-PR]: {vus_pr:.4f}"
        self.logger.info(accuracy)
        # self.logger.info(affiliation_res)
        self.logger.info(auc)
        self.logger.info(vus)
        with open(self.eval_file.format(epochs=self.config.epochs), "w+") as f:
            f.write(f"{accuracy}\n")
            # f.write(f"{affiliation_res}\n")
            f.write(f"{auc}\n")
            f.write(f"{vus}\n\n")
            for arg in vars(self.config):
                f.write(f"{arg}: {getattr(self.config, arg)}\n") 

                