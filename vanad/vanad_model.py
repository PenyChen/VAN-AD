
import torch
from vanad.rec_model import VANAD_Rec
from vanad.NormalizingFlow import NormalizingFlow
import os

class VANAD(torch.nn.Module):
    def __init__(self, config):
        super(VANAD, self).__init__()
        self.config = config
        self.use_nf = config.use_nf
        self.vanad = VANAD_Rec(
            config.arch, 
            config.finetune_type, 
            config.ckpt_dir, 
            config.load_ckpt, 
            config.image_method,
            config.image_mask
            )
        
        if self.use_nf:
            self.nf = NormalizingFlow(n_blocks=self.config.n_blocks, n_sensor=self.config.n_sensor, input_size=self.config.input_size, 
                                  hidden_size=self.config.hidden_size, n_hidden=self.config.n_hidden, cond_label_size=self.config.cond_label_size,
                                  batch_norm=self.config.batch_norm, activation=self.config.activation)
        
        self.vanad.update_config(
            seq_len=config.seq_len, 
            periodicity=config.periodicity, 
            norm_const=0.4, 
            interpolation='bilinear',
            num_vars = config.num_vars
            )
        
    def forward(self, x, flag='train'):
        y_dec, y = self.vanad(x,denormalize=self.config.denormalize)
        if flag == 'test':
            y=x
        if self.use_nf:
            full_shape = y.shape
            nf_input = y.reshape(-1, full_shape[2])   # [batch_size*seq_len, n_sensor]
            log_prob = self.nf.log_prob(nf_input,full_shape[1], None).reshape([full_shape[0],full_shape[1],-1])  # [batch_size, seq_len, n_sensor]
            log_prob = log_prob.mean(dim=-1)  # [batch_size, seq_len]
            return y_dec, log_prob
        
        return y_dec, y

   
    def load_weights(self,file):
        if os.path.exists(file):
            print(f"load wights from {file}")
            self.load_state_dict(torch.load(file))
        else:
            # if weight can't be found, test will not be conducted
            raise FileNotFoundError
    
        
        
               