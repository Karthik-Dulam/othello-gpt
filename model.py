# we want to train a model to predict the next move given the current board state
# we will use a transformer model with a positional encoding
# following the paper We trained an 8-layer GPT model (Radford et al., 2018; 2019; Brown et al., 2020) with an 8-
# head attention mechanism and a 512-dimensional hidden space. The training was performed in an
# autoregressive fashion.

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import logging
import math
logger = logging.getLogger(__name__)

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1  # only applied on matmul weights
    # learning rate decay params
    lr_decay = False
    warmup_tokens = 375e6
    final_tokens = 260e9
    # checkpoint settings
    ckpt_path = None
    num_workers = 0

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class GPTConfig:
    embed_dropout = 0.1
    residual_dropout = 0.1
    attention_dropout = 0.1
    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)

# we will now define a vanilla multihead maxked self attention layer
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.embed_dim % config.n_heads == 0

        self.key = nn.Linear(config.embed_dim, config.embed_dim)
        self.query = nn.Linear(config.embed_dim, config.embed_dim)
        self.value = nn.Linear(config.embed_dim, config.embed_dim)
        # mask to prevent the model from attending to future tokens
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size)
        ) # 1 x 1 x block_size x block_size lower triangular matrix of 1s
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)

        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.residual_dropout = nn.Dropout(config.residual_dropout)
        self.n_heads = config.n_heads

    def forward(self, x, only_last = -1):
        B, T, C = x.size() # batch size, sequence length (number of tokens in context), embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in parallel
        k = (self.key(x) # gets keys for all heads in parallel B x T x C
             .view(B, T, self.n_heads, C // self.n_heads) # reshape to B x T x n_heads x head_dim separated by head
             .transpose(1, 2) # B x n_heads x T x head_dim 
        )
        q = (self.query(x) # gets keys for all heads in parallel B x T x C
             .view(B, T, self.n_heads, C // self.n_heads) # reshape to B x T x n_heads x head_dim separated by head
             .transpose(1, 2) # B x n_heads x T x head_dim 
        )
        v = (self.value(x) # gets keys for all heads in parallel B x T x C
             .view(B, T, self.n_heads, C // self.n_heads) # reshape to B x T x n_heads x head_dim separated by head
             .transpose(1, 2) # B x n_heads x T x head_dim 
        )

        # compute attention scores
        att = (q # B x n_heads x T x head_dim
               @ k.transpose(-2, -1) # B x n_heads x head_dim x T
               * (1.0 / math.sqrt(k.size(-1))) # k.size(-1) is the head dimension
        ) # B x n_heads x T x T
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, # 1 x 1 x T x T restrict the size from block_size x block_size to T x T
                             float('-inf')) # fill the upper triangular part with -inf
        # att encodes how much each token attends to each other token hence T x T
        if only_last != -1:
            att[:,:,-only_last:,:-only_last] = -float('inf') # dont let the last `only_last` tokens to attend to any other tokens
        att = F.softmax(att, dim=-1) # B x n_heads x T x T in each batch, for each head, for the nth token, we want a probability distribution over all tokens
        att = self.attention_dropout(att) # apply dropout
        y = att @ v # B x n_heads x T x T @ B x n_heads x T x head_dim = B x n_heads x T x head_dim
        y = y.transpose(1, 2).contiguous().view(B, T, C) # B x T x C we reassemble the heads into a single vector
        y = self.proj(y) # B x T x C
        y = self.residual_dropout(y) # apply dropout
        return y, att 

# a transformer block is a layer normalization followed by a self attention layer
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, 4 * config.embed_dim),
            nn.GELU(),
            nn.Linear(4 * config.embed_dim, config.embed_dim),
            nn.Dropout(config.residual_dropout),
        )

    def forward(self, x, return_attn = False, only_last = -1):
        updt, attn = self.attn(self.ln1(x), only_last)
        x = x + updt
        x = x + self.mlp(self.ln2(x))
        if return_attn:
            return x, attn
        else:
            return x

class GPT(nn.Module):
    '''
    GPT model: a stack of transformer blocks
    '''
    def __init__(self, config):
        super().__init__()
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.embed_dim) # take each token and embed it into a vector of dimensionality n_embd
        self.position_embedding_table = nn.Parameter(torch.zeros(1, config.block_size, config.embed_dim)) # position embedding table
        self.embed_dropout = nn.Dropout(config.embed_dropout)

        # transformer 
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layers)])
        self.n_layers = config.n_layers
        # decoder head
        self.ln_f = nn.LayerNorm(config.embed_dim) # final layer norm
        self.head = nn.Linear(config.embed_dim, config.vocab_size) # maps the final hidden state to a logits for each token in the vocabulary

        self.block_size = config.block_size
        self.apply(self._init_weights) # initialize the weights

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

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
        for mn, m in self.named_modules(): # mn is the name of the module, m is the module itself
            for pn, p in m.named_parameters(): # pn is the name of the parameter, p is the parameter itself
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('position_embedding_table')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, 
                                    lr=train_config.learning_rate, 
                                    betas=train_config.betas)
        return optimizer
    
    def forward(self, idx, targets = None):
        b, t = idx.size() # batch size, sequence length (number of tokens in context)
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"

        token_embeddings = self.token_embedding_table(idx) # (b, t, n_embd)
        position_embeddings = self.position_embedding_table[:, :t, :] # (1, t, n_embd) maps the position of the token in the sequence to a learnable vector
        x = self.embed_dropout(token_embeddings + position_embeddings) # (b, t, n_embd)

        x = self.blocks(x) # (b, t, n_embd)
        x = self.ln_f(x) # (b, t, n_embd)
        logits = self.head(x) # (b, t, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index = 0) 
            # to explain: we are flattening the logits and targets to 1D vectors and then computing the cross entropy loss
            # ignore_index = 0 is used to ignore the padding token
        return logits, loss

class GPTforProbing(GPT):
    def __init__(self, config, probe_layer=-1, ln=False):
        super().__init__(config)
        self.probe_layer = self.n_layers if probe_layer == -1 else probe_layer
        assert self.probe_layer <= self.n_layers and self.probe_layer >= 0, f"Probe layer {self.probe_layer} is out of bounds for {self.n_layers} layers"
        self.ln = ln

    def forward(self, idx, return_attn = False):
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"

        token_embeddings = self.token_embedding_table(idx) # (b, t, n_embd)
        position_embeddings = self.position_embedding_table[:, :t, :] # (1, t, n_embd)
        x = self.embed_dropout(token_embeddings + position_embeddings) # (b, t, n_embd)

        for block in self.blocks[:self.probe_layer]:
            if return_attn:
                x, attn = block(x, return_attn = True)
            else:
                x = block(x)

        if self.ln:
            x = self.ln_f(x)

        if return_attn:
            return x, attn
        else:
            return x

class GPTforIntervention(GPT):
    def __init__(self, config, probe_layer=-1):
        super().__init__(config)
        self.probe_layer = self.n_layers if probe_layer == -1 else probe_layer
        assert self.probe_layer <= self.n_layers and self.probe_layer >= 0, f"Probe layer {self.probe_layer} is out of bounds for {self.n_layers} layers"

    def forward_1st_stage(self, idx, targets = None):
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"

        token_embeddings = self.token_embedding_table(idx) # (b, t, n_embd)
        position_embeddings = self.position_embedding_table[:, :t, :] # (1, t, n_embd)
        x = self.embed_dropout(token_embeddings + position_embeddings) # (b, t, n_embd)

        for block in self.blocks[:self.probe_layer]:
            x = block(x)

        return x

    def forward_2nd_stage(self, x, targets = None, only_last = -1):
        for block in self.blocks[self.probe_layer:]:
            x = block(x, only_last = only_last)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index = 0)
        return logits, loss

class GPTforProbeIA(GPT):
    def __init__(self, config, probe_layer=-1):
        super().__init__(config)
        self.probe_layer = self.n_layers if probe_layer == -1 else probe_layer
        assert self.probe_layer <= self.n_layers and self.probe_layer >= 0, f"Probe layer {self.probe_layer} is out of bounds for {self.n_layers} layers"

    def forward_1st_stage(self, idx, targets = None):
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"

        token_embeddings = self.token_embedding_table(idx) # (b, t, n_embd)
        position_embeddings = self.position_embedding_table[:, :t, :] # (1, t, n_embd)
        x = self.embed_dropout(token_embeddings + position_embeddings) # (b, t, n_embd)

        for block in self.blocks[:self.probe_layer]:
            x = block(x)

        return x

    def forward_2nd_stage(self, x, start_layer, end_layer=-1):
        tbr = [] # list of tensors to return
        if end_layer == -1:
            end_layer = self.n_layers + 1

        for block in self.blocks[start_layer:end_layer]:
            x = block(x)
            tbr.append(x)
        return tbr

    def predict(self, x, targets=None):
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index = 0)
        return logits, loss

class GPT_Lit(pl.LightningModule):
    def __init__(self, config, learning_rate=3e-4, weight_decay=0.1, betas=(0.9, 0.95)):
        super().__init__()
        self.save_hyperparameters()
        self.model = GPT(config)
        
    def configure_optimizers(self):
        return self.model.configure_optimizers(self.hparams)

    def training_step(self, batch, batch_idx):
        x = batch['input_ids']
        y = batch['input_ids']  # Targets are same as inputs for language modeling
        _, loss = self.model(x, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch['input_ids']
        y = batch['input_ids']
        _, loss = self.model(x, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss
