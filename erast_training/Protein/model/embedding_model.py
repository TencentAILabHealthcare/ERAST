# from einops import repeat
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict
# from einops.layers.torch import Rearrange
# import einops
from functools import partial
from transformers.modeling_outputs import SequenceClassifierOutput
def gelu(x):
    """
    This is the gelu implementation from the original ESM repo. Using F.gelu yields subtly wrong results.
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class EsmAttn(nn.Module):
    """
    TransformerEncoderLayer with preset parameters followed by global pooling and dropout
    """
    def __init__(self,cfg):
        super().__init__()
        num_layers = cfg["num_blocks"]

        encoder_layer = nn.TransformerEncoderLayer(batch_first=True, d_model=1024, dim_feedforward = 2048,
    nhead= cfg["num_heads"])
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(cfg["dropout"])
        # self.mlp = nn.Linear(1024, 512)

        # self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        # self.l1_loss = nn.L1Loss(reduction='mean')

    def forward(self, x, src_mask, src_key_padding_mask):
        x = self.encoder(x, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        
        lens = torch.logical_not(src_key_padding_mask).sum(dim=1).float()
        x = x.sum(dim=1) / lens.unsqueeze(1)
        x = self.dropout(x)
        # x = self.mlp(x)
        return x
class ScoreHead(nn.Module):
    """ESM Head for masked language modeling."""
    
    def __init__(self,config):
        super().__init__()
        self.dense = nn.Linear(config.d_model,config.d_model)
        self.layer_norm = nn.LayerNorm(config.d_model, eps=float(config.layer_norm_eps))

        self.decoder = nn.Linear(config.d_model, config.num_classes, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.num_classes))
        
        self.apply(self.init_weights)
    def init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x) + self.bias
        return x
class Scorer(nn.Module):
    
    def __init__(self,encoder,config):
        super().__init__()
        
        self.encoder = encoder
        # self.device=esm.device
        
        self.score_head=ScoreHead(config.ScoreHead)
        # self.score_head.to(self.device)
        
    
    def forward(self,input_ids1,attention_mask1,input_ids2,attention_mask2,labels):
        out1=self.encoder(input_ids1,attention_mask1)
        out2=self.encoder(input_ids2,attention_mask2)
        rep1=torch.mean(out1.last_hidden_state[:,0:-1,:],dim=1) # [b,d]
        rep2=torch.mean(out2.last_hidden_state[:,0:-1,:],dim=1) # [b,d]
        
        # rep1 = F.normalize(rep1, p=2, dim=1)
        # rep2 = F.normalize(rep2, p=2, dim=1)
        
        distance =abs(rep1-rep2)
        # distance =torch.concat((rep1,rep2),dim=-1)
        logits=self.score_head(distance) 
        
        loss = F.cross_entropy(logits, labels)
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
        )
class SScorer(nn.Module):
    
    def __init__(self,config):
        super().__init__()
        
        self.score_head=ScoreHead(config.ScoreHead)
        # self.score_head.to(self.device)
        
    
    def forward(self,embedding1,embedding2,labels):
     
        
        distance =abs(embedding1-embedding2)
        # distance =torch.concat((rep1,rep2),dim=-1)
        logits=self.score_head(distance) 
        
        loss = F.cross_entropy(logits, labels)
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
        )
class MOEmb(nn.Module):
    def __init__(self,config):
        super().__init__()
        ### para config
        self.dropout_prob = config.dropout_prob
        ### esm2
        self.net1_l1 = torch.nn.Linear(config.emb1_dim, config.emb1_dim)
        self.net1_l1.weight.detach().normal_(0.0, 0.1)
        self.net1_l1.bias.detach().zero_()
        
        self.net1_l2 = torch.nn.Linear(config.emb1_dim, config.l2_dim)
        self.net1_l2.weight.detach().normal_(0.0, 0.1)
        self.net1_l2.bias.detach().zero_()
        ### prot_T5
        self.net2_l1 = torch.nn.Linear(config.emb2_dim, config.emb2_dim)
        self.net2_l1.weight.detach().normal_(0.0, 0.1)
        self.net2_l1.bias.detach().zero_()
        
        self.net2_l2 = torch.nn.Linear(config.emb2_dim, config.l2_dim)
        self.net2_l2.weight.detach().normal_(0.0, 0.1)
        self.net2_l2.bias.detach().zero_()
        ### one hot
        self.net3_l1 = torch.nn.Linear(config.emb3_dim, config.emb3_dim)
        self.net3_l1.weight.detach().normal_(0.0, 0.1)
        self.net3_l1.bias.detach().zero_()
        
        self.net3_l2 = torch.nn.Linear(config.emb3_dim, config.l2_dim)
        self.net3_l2.weight.detach().normal_(0.0, 0.1)
        self.net3_l2.bias.detach().zero_()
        ### final emb
        # self.final_l1 = torch.nn.Linear((3*config.l2_dim), config.final_dim)
        # self.final_l1.weight.detach().normal_(0.0, 0.1)
        # self.final_l1.bias.detach().zero_()
        
        self.score_head=ScoreHead(config.ScoreHead) ### TODO:change the dim
        
    def _encoder(self, emb1,emb2,emb3):
        out1 = self.net1_l1(emb1)
        out1 = F.relu(out1)
        out1 = F.dropout(out1, p=self.dropout_prob, training=self.training)
        out1 = self.net1_l2(out1)
        
        out2 = self.net2_l1(emb2)
        out2 = F.relu(out2)
        out2 = F.dropout(out2, p=self.dropout_prob, training=self.training)
        out2 = self.net2_l2(out2)
        
        out3 = self.net3_l1(emb3)
        out3 = F.relu(out3)
        out3 = F.dropout(out3, p=self.dropout_prob, training=self.training)
        out3 = self.net3_l2(out3)

        concat_out=torch.concat([out1,out2,out3],dim=-1)
        # concat_out=self.final_l1(concat_out)
        # concat_out = F.relu(concat_out)
      
        return concat_out
    def forward(self, batch):
        rep1=self._encoder(batch["a_emb1"],batch["a_emb2"],batch["a_emb3"])
        rep2=self._encoder(batch["b_emb1"],batch["b_emb2"],batch["b_emb3"])
        
        rep1 = F.normalize(rep1, p=2, dim=1)
        rep2 = F.normalize(rep2, p=2, dim=1)
        
        distance =abs(rep1-rep2)
        logits=self.score_head(distance) 
        
        return logits

        