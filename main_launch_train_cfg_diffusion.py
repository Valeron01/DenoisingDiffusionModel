from torch.utils.data import DataLoader
import pytorch_lightning as pl
import pytorch_lightning.loggers
import os
from dataset_builder import build_dataset
from modules.lit_modules.cfg_diffusion import CFGDiffusion
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--lightning_folder", required=False, default="./lightning")
    parser.add_argument("--checkpoints_folder", required=False, default="./lightning/checkpoints")
    parser.add_argument("--max_epochs", required=False, default=500)
    parser.add_argument("--batch_size", required=False, default=64)
    parser.add_argument("--num_workers", required=False, default=4)
    args = parser.parse_args()

    train_dataset = build_dataset(args.dataset_path, (64, 64))

    train_loader = DataLoader(
        train_dataset, args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers
    )

    model = CFGDiffusion(len(train_dataset.classes))

    logger = pl.loggers.TensorBoardLogger(args.lightning_folder)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        os.path.join(args.checkpoints_folder, f"run_{logger.version}")
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback],
        logger=logger,
        log_every_n_steps=20
    )

    trainer.fit(model, train_loader)


if __name__ == '__main__':
    main()
