from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
def get_dataloader(config,mode,seq_len=None,pre_len=None,step=None,shuffle=None,drop_last=None,data_ratio=1.0):
    dataset = None
    if config.data == 'msl':
        from data_factory.msl_dataset import MSLDatasetNF
        dataset = MSLDatasetNF(
            config=config,
            mode=mode,
            seq_len=seq_len,
            pre_len=pre_len,
            step=step,
            data_ratio=data_ratio
        )
    elif config.data == 'cicids':
        from data_factory.cicids_dataset import CICIDSDatasetNF
        dataset = CICIDSDatasetNF(
            config=config,
            mode=mode,
            seq_len=seq_len,
            pre_len=pre_len,
            step=step,
            data_ratio=data_ratio
        )
    elif config.data == 'psm':
        from data_factory.psm_dataset import PSMDatasetNF
        dataset = PSMDatasetNF(
            config=config,
            mode=mode,
            seq_len=seq_len,
            pre_len=pre_len,
            step=step,
            data_ratio=data_ratio
        )
    elif config.data == 'smd':
        from data_factory.smd_dataset import SMDDatasetNF
        dataset = SMDDatasetNF(
            config=config,
            mode=mode,
            seq_len=seq_len,
            pre_len=pre_len,
            step=step,
            data_ratio=data_ratio
        )
    elif config.data == 'gecco':
        from data_factory.gecco_dataset import GECCODatasetNF
        dataset = GECCODatasetNF(
            config=config,
            mode=mode,
            seq_len=seq_len,
            pre_len=pre_len,
            step=step,
            data_ratio=data_ratio
        )
    elif config.data == 'tab':
        from data_factory.TAB_dataset import TABDatasetNF
        dataset = TABDatasetNF(
            config=config,
            mode=mode,
            seq_len=seq_len,
            pre_len=pre_len,
            step=step,
            data_ratio=data_ratio
        )
    else:
        raise NotImplementedError(f"Dataset {config['data']} is not implemented.")
    

    data_loader = DataLoader(dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=shuffle, drop_last=drop_last)
    return data_loader
    
    
def get_train_dataloader(config):
    return  get_dataloader(
            config, 
            mode='train',
            seq_len=config.seq_len, 
            pre_len=0, 
            step=10,
            shuffle=False, 
            drop_last=False,
            data_ratio=getattr(config, 'data_ratio', 1.0)
        )
    
def get_test_dataloader(config):
    return  get_dataloader(
            config, 
            mode='test',
            seq_len=config.seq_len, 
            pre_len=0, 
            step=config.seq_len,
            shuffle=False, 
            drop_last=False,
            data_ratio=1.0
        )
    
def get_val_dataloader(config):
    return  get_dataloader(
            config, 
            mode='val',
            seq_len=config.seq_len, 
            pre_len=0, 
            step=10,
            shuffle=False, 
            drop_last=False,
            data_ratio=1.0
        )