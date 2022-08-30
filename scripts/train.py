import os
import argparse
import tarfile
import logging

import autogluon.core as ag
from autogluon.vision import (
    ImageDataset,
    ImagePredictor,
)

# Use logging.WARNING since logging.INFO is ignored by AutoGluon DLC
logging.basicConfig(level=logging.WARNING)  


def _parse_args():
    parser = argparse.ArgumentParser()
    
    # /opt/ml/input/data/training
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    # /opt/ml/model
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--frac', type=float, default=1.)
    parser.add_argument('--random-state', type=int, default=42)
    
    return parser.parse_known_args()

    
def train_with_args(args):
    with tarfile.open(os.path.join(args.train, 'train.tar.gz')) as f:
        f.extractall(args.train)
    train_data = ImageDataset.from_folder(args.train)
    logging.debug(args.train)
    logging.debug(train_data)
    train_data_sampled = train_data.sample(frac=args.frac, random_state=args.random_state)
    predictor = ImagePredictor(path=args.model_dir)
    predictor.fit(train_data=train_data_sampled, hyperparameters={
        'epochs': args.epochs,
        'batch_size': args.batch_size,
    })
    predictor.save(os.path.join(args.model_dir, 'image_predictor.ag'))


if __name__ == '__main__':
    args, unknown = _parse_args()
    train_with_args(args)
