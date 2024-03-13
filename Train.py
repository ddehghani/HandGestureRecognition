import os
import argparse
import cv2 as cv
from Util import bgremove
from tqdm import trange
import shutil
from random import random
import torch
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from Util import *
from Model import *
import pickle


SUPPORTED_VIDEO_FORMATS = ['.mov', '.avi', '.mp4', '.wmv']

def main(args: argparse.Namespace):
    if not os.path.isdir(args.source):
        print('Bad source')
        return
    
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    temp_dir = './temp/'
    train_dir, test_dir = generateDataSamples(args.source, temp_dir, train_ratio = 0.8, sample_count=args.sample_count)

    # train the model on the video
    train_transform=v2.Compose([
        v2.Resize(54),              # resize shortest side
        v2.CenterCrop(96),          # crop longest side
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_ds = ImageFolder(train_dir, transform=train_transform)
    test_ds = ImageFolder(test_dir, transform=train_transform)

    print(f'\n{"-" *20}\nTraining the model:\nTraining data: {len(train_ds)}, Test data: {len(test_ds)}\n{"-" *20}\n')
    with open(os.path.join(args.output, 'classes.dict'), 'wb') as f:
        pickle.dump(train_ds.class_to_idx, f)
    
    batch_size = 64
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size*2, num_workers=4, pin_memory=True)

    device = get_default_device()
    train_dl = DeviceDataLoader(train_loader, device)
    test_dl = DeviceDataLoader(test_loader, device)
    model = to_device(CnnModel(len(train_ds.class_to_idx)), device)

    history = [evaluate(model, test_loader)]
    history += fit(epochs=args.epoch_count, lr = args.learning_rate, model = model, train_loader=train_dl, val_loader=test_dl, opt_func=torch.optim.Adam)
    plot_accuracies(history, os.path.join(args.output, 'accuracies.jpg'))
    plot_losses(history, os.path.join(args.output, 'losses.jpg'))
    torch.save(model, os.path.join(args.output, 'model.pth'))
    
    # delete the samples
    if not args.keep_samples:
        shutil.rmtree(temp_dir)

def generateDataSamples(source: os.PathLike, temp_dir: os.PathLike, train_ratio: float, sample_count: int) -> tuple[os.PathLike, os.PathLike]:
    """
    Create sample images from the video files in source 
    divide them into train and test sub directories
    and store them in the temp_dir according to the source dir structure
    return train and test directories
    """
    shutil.rmtree(temp_dir)                     # delete any temporary data from previous training
    os.mkdir(temp_dir)   
    train_dir = os.path.join(temp_dir, './train')
    test_dir = os.path.join(temp_dir, 'test') 
    os.mkdir(train_dir)
    os.mkdir(test_dir)
    for subdir in next(os.walk(source))[1]:
        os.mkdir(os.path.join(train_dir, subdir))
        os.mkdir(os.path.join(test_dir, subdir))
        os.mkdir(os.path.join(temp_dir, subdir))
        for video in next(os.walk(os.path.join(source, subdir)))[2]:
            if (os.path.splitext(video)[1] in SUPPORTED_VIDEO_FORMATS):
                generateDataSampleFromVideo(os.path.join(source, subdir, video), os.path.join(temp_dir, subdir), sample_count)
        for file in next(os.walk(os.path.join(temp_dir, subdir)))[2]:
            os.rename(os.path.join(temp_dir, subdir, file), 
                      os.path.join(train_dir if random() <= train_ratio else test_dir, subdir, file))
        shutil.rmtree(os.path.join(temp_dir, subdir))
    return (train_dir, test_dir)

def generateDataSampleFromVideo(source: os.PathLike, output: os.PathLike, sample_count: int):
    print(f'Processing video {source}:')
    cap = cv.VideoCapture(source)
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    sample_ratio = frame_count // sample_count

    for frame_index in trange(frame_count):
        _ , frame = cap.read()
        if frame_index % sample_ratio != 0:
            continue
        frame = bgremove(frame)
        cv.imwrite(os.path.join(output, f'{os.path.basename(source).split(".")[0]}{frame_index//sample_ratio}.jpg'), frame)
    
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                prog = 'HGR Trainer',
                description = 'Given data in the form of video, trains a CNN on hand gesture recognition task')
    
    parser.add_argument('source')
    parser.add_argument('-o', '--output', default = './output')
    parser.add_argument('-s', '--sample-count', default=1000, type = int)
    parser.add_argument('-e', '--epoch-count', default=2, type = int)
    parser.add_argument('-lr', '--learning-rate', default=0.001, type = float)
    parser.add_argument('-k', '--keep-samples', action='store_true')

    main(parser.parse_args())