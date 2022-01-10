# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
import errno
import torch
from torchvision import transforms
from os import walk
import numpy as np

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())

def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    if output_filepath[-1] != '/':
        output_filepath.append('/')
    if input_filepath[-1] != '/':
        input_filepath.append('/')

    if os.path.isdir(input_filepath) and (len(os.listdir(input_filepath)) != 0):
        filenames = next(walk(input_filepath), (None, None, []))[2]  # [] if no file
        train_paths=[input_filepath+i for i in filenames if 'train' in i]
        test_paths=[input_filepath+i for i in filenames if 'test' in i]
        
        train_images=np.concatenate([np.load(i)['images'] for i in train_paths],axis=0)
        train_labels=np.concatenate([np.load(i)['labels'] for i in train_paths],axis=0)

        test_images=np.concatenate([np.load(i)['images'] for i in test_paths],axis=0)
        test_labels=np.concatenate([np.load(i)['labels'] for i in test_paths],axis=0)
        print(train_images.shape)
        if not os.path.exists(output_filepath):
            os.makedirs(output_filepath)

        # torch.save(torch.from_numpy(train_images).float(),output_filepath+'train_images.pt')
        # torch.save(torch.from_numpy(train_labels).float(),output_filepath+'train_labels.pt')

        # torch.save(torch.from_numpy(test_images).float(),output_filepath+'test_images.pt')
        # torch.save(torch.from_numpy(test_labels).float(),output_filepath+'test_labels.pt')

        transform=transforms.Compose([
        transforms.Normalize(mean=0,std=1)
        ])
        
        torch.save(transform(torch.from_numpy(train_images)).float(),output_filepath+'train_images.pt')
        torch.save(torch.from_numpy(train_labels),output_filepath+'train_labels.pt')

        torch.save(transform(torch.from_numpy(test_images)).float(),output_filepath+'test_images.pt')
        torch.save(torch.from_numpy(test_labels),output_filepath+'test_labels.pt')

    else:
        raise FileNotFoundError(
        errno.ENOENT, os.strerror(errno.ENOENT), input_filepath)
    

    


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
