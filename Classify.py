import argparse
import cv2 as cv
import os
from tqdm import tqdm
import torch
from Util import VideoStreamDataset, augmentImage
from torch.utils.data.dataloader import DataLoader
import pickle


def main(args: argparse.Namespace):
    with open(args.class_file, 'rb') as f:
        classes = pickle.load(f)
    # reverse key / value pairs
    classes = {value: key for key, value in classes.items()}
    model = torch.load(args.model)

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    for videoname in next(os.walk(args.source))[2]:
        video = os.path.join(args.source, videoname)
        source_video = cv.VideoCapture(video)
        loader = DataLoader(VideoStreamDataset(video), batch_size=64)
        result = cv.VideoWriter(os.path.join(args.output, videoname),
                                cv.VideoWriter_fourcc(*'XVID'),
                                fps=source_video.get(cv.CAP_PROP_FPS),
                                frameSize=(int(source_video.get(3)),
                                           int(source_video.get(4))))

        label_gen = classifyImage(loader, model)

        print(f'Processing video {video}:')
        pbar = tqdm(total=source_video.get(cv.CAP_PROP_FRAME_COUNT))
        while (source_video.isOpened()):
            ret, frame = source_video.read()
            if not ret:
                break
            augmentImage(frame, classes.get(next(label_gen)))
            result.write(frame)
            pbar.update(1)

        source_video.release()
        result.release()
        cv.destroyAllWindows()


def classifyImage(loader, model):
    for _, frame in enumerate(loader):
        with torch.no_grad():
            output = model(frame)
            _, preds = torch.max(output, dim=1)
            for pred in preds.tolist():
                yield pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                prog='HGR Classifier',
                description='Evaluates hand gestures in a batch of videos')

    parser.add_argument('source')
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-cf', '--class-file', required=True)
    parser.add_argument('-o', '--output', default='./results')

    main(parser.parse_args())
