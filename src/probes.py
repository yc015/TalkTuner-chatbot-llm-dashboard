import torch
import torch.nn.functional as F
from torch import nn

class ProbeClassification(nn.Module):
    def __init__(self, device, probe_class, input_dim=512, hidden_neurons=128):  # from 0 to 15
        super().__init__()
        self.input_dim = input_dim
        self.probe_class = probe_class
        self.proj = nn.Sequential(
            nn.Linear(self.input_dim, hidden_neurons),
            nn.ReLU(True),
            nn.Linear(hidden_neurons, self.probe_class),
        )
        self.apply(self._init_weights)
        # logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))
        self.to(device)
    def forward(self, act, y=None):
        # [B, f], [B]
        logits = self.proj(act)#.reshape(-1, self.probe_number, self.probe_class)  # [B, C]
        if y is None:
            return logits, None
        else:
            targets = y.to(torch.long)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
            return logits, loss
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                if pn.endswith('bias'):
                    # biases of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        # no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )
        print("Decayed:", decay)
        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.Adam(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=0)
        return optimizer, scheduler
    

class LinearProbeClassification(nn.Module):
    def __init__(self, device, probe_class, input_dim=512, logistic=False, Relu=False, TanH=False):  # from 0 to 15
        super().__init__()
        self.input_dim = input_dim
        self.probe_class = probe_class
        if logistic:
            self.proj = nn.Sequential(
                nn.Linear(self.input_dim, self.probe_class),
                nn.Sigmoid()
            )
        elif Relu:
            self.proj = nn.Sequential(
                nn.Linear(self.input_dim, self.probe_class),
                nn.ReLU(True)
            )
        elif TanH:
            self.proj = nn.Sequential(
                nn.Linear(self.input_dim, self.probe_class),
                # nn.Hardtanh(inplace=True, min_val=0.001, max_val=0.999)
                nn.Hardsigmoid(inplace=True)
            )
        else:
            
            self.proj = nn.Sequential(
                nn.Linear(self.input_dim, self.probe_class),
            )
        
        self.apply(self._init_weights)
        # logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))
        self.to(device)
    def forward(self, act, y=None):
        # [B, f], [B]
        logits = self.proj(act)#.reshape(-1, self.probe_number, self.probe_class)  # [B, C]
        if y is None:
            return logits, None
        else:
            targets = y.to(torch.long)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
            return logits, loss
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                if pn.endswith('bias'):
                    # biases of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        # no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )
        print("Decayed:", decay)
        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.Adam(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=0)
        return optimizer, scheduler

    
class TwoLayerLinearProbeClassification(nn.Module):
    def __init__(self, device, probe_class, input_dim=512, logistic=False):  # from 0 to 15
        super().__init__()
        self.input_dim = input_dim
        self.probe_class = probe_class
        if not logistic:
            self.proj = nn.Sequential(
                nn.Linear(self.input_dim, self.input_dim),
                nn.Linear(self.input_dim, self.probe_class),
            )
        else:
            self.proj = nn.Sequential(
                nn.Linear(self.input_dim, self.input_dim),
                nn.Linear(self.input_dim, self.probe_class),
                nn.Sigmoid()
            )
        
        self.apply(self._init_weights)
        # logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))
        self.to(device)
    def forward(self, act, y=None):
        # [B, f], [B]
        logits = self.proj(act)#.reshape(-1, self.probe_number, self.probe_class)  # [B, C]
        if y is None:
            return logits, None
        else:
            targets = y.to(torch.long)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
            return logits, loss
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                if pn.endswith('bias'):
                    # biases of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        # no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )
        print("Decayed:", decay)
        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.Adam(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=0)
        return optimizer, scheduler 
    

class ProbeClassificationMixScaler(nn.Module):
    def __init__(self, device, probe_class, input_dim=512, num_layers=41, soft_weight_lr_rate=1e-1,
                 hidden_neurons=128):  # from 0 to 15
        super().__init__()
        self.input_dim = input_dim
        self.probe_class = probe_class
        self.num_layers = num_layers
        # self.mix_weights = torch.nn.Parameter(1 / num_layers * torch.ones(num_layers), requires_grad=True)
        self.mix_weights = nn.Linear(num_layers, 1, bias=False)
        torch.nn.init.constant_(self.mix_weights.weight, 1 / num_layers)
        self.soft_weight_lr_rate=soft_weight_lr_rate
        self.proj = nn.Sequential(
            nn.Linear(self.input_dim, hidden_neurons),
            nn.ReLU(True),
            nn.Linear(hidden_neurons, self.probe_class),
        )
        self.apply(self._init_weights)
        # logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))
        self.to(device)
    def forward(self, act, y=None):
        # [B, f], [B]
        softmaxed_weights = torch.nn.functional.softmax(self.mix_weights.weight, dim=1)
        act = act.permute([0, 2, 1]) 
        act = (act @ softmaxed_weights.T)[..., 0]
        logits = self.proj(act)#.reshape(-1, self.probe_number, self.probe_class)  # [B, C]
        if y is None:
            return logits, None
        else:
            targets = y.to(torch.long)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
            return logits, loss
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                if pn.endswith('bias'):
                    # biases of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and (not "mix" in fpn) and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        # no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        # assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    # % (str(param_dict.keys() - union_params), )
        print("Decayed:", decay)
        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
            {'params': self.mix_weights.weight, "lr": self.soft_weight_lr_rate, "weight_decay": train_config.weight_decay},
        ]
        optimizer = torch.optim.Adam(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=0)
        return optimizer, scheduler
    

class LinearProbeClassificationMixScaler(nn.Module):
    def __init__(self, device, probe_class, input_dim=512, num_layers=41, soft_weight_lr_rate=1e-1,
                 logistic=False):  # from 0 to 15
        super().__init__()
        self.input_dim = input_dim
        self.probe_class = probe_class
        self.num_layers = num_layers
        # self.mix_weights = torch.nn.Parameter(1 / num_layers * torch.ones(num_layers), requires_grad=True)
        self.mix_weights = nn.Linear(num_layers, 1, bias=False)
        torch.nn.init.constant_(self.mix_weights.weight, 1 / num_layers)
        self.soft_weight_lr_rate=soft_weight_lr_rate
        if not logistic:
            self.proj = nn.Sequential(
                nn.Linear(self.input_dim, self.probe_class),
            )
        else:
            self.proj = nn.Sequential(
                nn.Linear(self.input_dim, self.probe_class),
                nn.Sigmoid()
            )
        self.apply(self._init_weights)
        # logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))
        self.to(device)
    def forward(self, act, y=None):
        # [B, f], [B]
        softmaxed_weights = torch.nn.functional.softmax(self.mix_weights.weight, dim=1)
        act = act.permute([0, 2, 1]) 
        act = (act @ softmaxed_weights.T)[..., 0]
        logits = self.proj(act)#.reshape(-1, self.probe_number, self.probe_class)  # [B, C]
        if y is None:
            return logits, None
        else:
            targets = y.to(torch.long)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
            return logits, loss
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                if pn.endswith('bias'):
                    # biases of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and (not "mix" in fpn) and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        # no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        # assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    # % (str(param_dict.keys() - union_params), )
        print("Decayed:", decay)
        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
            {'params': self.mix_weights.weight, "lr": self.soft_weight_lr_rate, "weight_decay": train_config.weight_decay},
        ]
        optimizer = torch.optim.Adam(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=0)
        return optimizer, scheduler  
    
    
class TwoLayerLinearProbeClassificationMixScaler(nn.Module):
    def __init__(self, device, probe_class, input_dim=512, num_layers=41, soft_weight_lr_rate=1e-1,
                 logistic=False):  # from 0 to 15
        super().__init__()
        self.input_dim = input_dim
        self.probe_class = probe_class
        self.num_layers = num_layers
        # self.mix_weights = torch.nn.Parameter(1 / num_layers * torch.ones(num_layers), requires_grad=True)
        self.mix_weights = nn.Linear(num_layers, 1, bias=False)
        torch.nn.init.constant_(self.mix_weights.weight, 1 / num_layers)
        self.soft_weight_lr_rate=soft_weight_lr_rate
        self.rotates = nn.ModuleList([nn.Linear(5120, 5120) for _ in range(41)]),
        if not logistic:
            self.proj = nn.Sequential(
                nn.Linear(self.input_dim, self.probe_class),
            )
        else:
            self.proj = nn.Sequential(
                nn.Linear(self.input_dim, self.probe_class),
                nn.Sigmoid()
            )
        self.apply(self._init_weights)
        # logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))
        self.to(device)
    def forward(self, act, y=None):
        # [B, f], [B]
        outputs = []
        for i in range(num_vectors):
            output_i = self.rotates[i](act[:, i, :])  # shape: (batch_size, 5120)
            outputs.append(output_i)
        
        # Stack the outputs back together
        act = torch.stack(outputs, dim=1) 
        softmaxed_weights = torch.nn.functional.softmax(self.mix_weights.weight, dim=1)
        act = act.permute([0, 2, 1]) 
        act = (act @ softmaxed_weights.T)[..., 0]
        logits = self.proj(act)#.reshape(-1, self.probe_number, self.probe_class)  # [B, C]
        if y is None:
            return logits, None
        else:
            targets = y.to(torch.long)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
            return logits, loss
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                if pn.endswith('bias'):
                    # biases of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and (not "mix" in fpn) and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        # no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        # assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    # % (str(param_dict.keys() - union_params), )
        print("Decayed:", decay)
        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
            {'params': self.mix_weights.weight, "lr": self.soft_weight_lr_rate, "weight_decay": train_config.weight_decay},
        ]
        optimizer = torch.optim.Adam(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=0)
        return optimizer, scheduler
    
    
class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)