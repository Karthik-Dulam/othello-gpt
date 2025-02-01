import argparse
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from datasets import load_dataset
from pytorch_lightning.loggers import TensorBoardLogger

from data import Tokenizer
from model import GPTConfig, GPT_Lit

class GoDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, max_seq_length=70, num_proc=32):
        super().__init__()

        self.save_hyperparameters()
        self.tokenizer = Tokenizer(max_length=max_seq_length)
        os.makedirs(".cache", exist_ok=True)

    def setup(self, stage=None):
        if stage in (None, "fit"):
            train_path = os.path.join(self.hparams.data_dir, 'train')
            self.train_ds = self._load_and_tokenize(train_path)
            
            val_path = os.path.join(self.hparams.data_dir, 'val')
            self.val_ds = self._load_and_tokenize(val_path)

    def _load_and_tokenize(self, path):
        cache_file = os.path.join(".cache", f"{os.path.basename(path)}.cache")
        dataset = load_dataset('text', data_dir=path)['train']
        return dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=["text"],
            num_proc=self.hparams.num_proc,
            cache_file_name=cache_file
        ).with_format("torch")

    def tokenize_function(self, examples):
        return {
            "input_ids": [torch.tensor(self.tokenizer.encode(text)) for text in examples["text"]],
            "attention_mask": [torch.tensor([1]*len(text.split()) + [0]*(self.tokenizer.max_length - len(text.split()))) 
                             for text in examples["text"]]
        }

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_proc,
            shuffle=True,
            collate_fn=lambda batch: {
                'input_ids': torch.stack([x['input_ids'] for x in batch]),
                'attention_mask': torch.stack([x['attention_mask'] for x in batch])
            }
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_proc,
            collate_fn=lambda batch: {
                'input_ids': torch.stack([x['input_ids'] for x in batch]),
                'attention_mask': torch.stack([x['attention_mask'] for x in batch])
            }
        )

def main(args):
    # Set matrix multiplication precision
    torch.set_float32_matmul_precision('high')
    
    dm = GoDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        num_proc=args.num_proc
    )
    
    config = GPTConfig(
        vocab_size=62,
        block_size=args.max_seq_length,
        embed_dim=512,
        n_heads=8,
        n_layers=8,
        attention_dropout=0.1,
        residual_dropout=0.1,
        embed_dropout=0.1
    )

    model = GPT_Lit(
        config,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2)
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        monitor='val_loss',
        save_top_k=3,
        mode='min'
    )

    trainer = pl.Trainer(
        logger=TensorBoardLogger(save_dir=args.log_dir),
        callbacks=[
            checkpoint_callback,
            EarlyStopping(monitor='val_loss', patience=5)
        ],
        default_root_dir=args.log_dir,
        max_epochs=args.max_epochs,
        devices=args.gpus,
        precision=args.precision,
        accelerator=args.accelerator
    )

    trainer.fit(model, dm)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--max_seq_length', type=int, default=70)
    parser.add_argument('--num_proc', type=int, default=32,
                       help='Number of workers for both dataset processing and data loading')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--checkpoint_dir', default='checkpoints')
    parser.add_argument('--log_dir', default='logs')
    
    # Add Lightning trainer args manually
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--gpus', type=int, default=1 if torch.cuda.is_available() else 0)
    parser.add_argument('--precision', type=str, default="16-mixed" if torch.cuda.is_available() else 32)
    parser.add_argument('--accelerator', type=str, default='gpu' if torch.cuda.is_available() else 'cpu')
    
    # Add batch_size argument
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    
    args = parser.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    print(args)
    
    main(args) 