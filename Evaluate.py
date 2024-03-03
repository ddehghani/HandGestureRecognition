import cv2 as cv
import torch
import numpy as np
from Util import VideoStreamDataset, augmentImage
from torch.utils.data.dataloader import DataLoader

PATH_TO_VIDEO = r'test2.mov'
PATH_TO_MODEL = r'v3.pt'
PATH_TO_RESULT = r'result2.avi'

def main():
    classes = ['1(bot)', '0(top)', '1(top)', '0(bot)', '2(top)', '2(bot)']
    revised_classes = ['Cls:1', 'Cls:0', 'Cls:1', 'Cls:0', 'Cls:2', 'Cls:2']
    model = torch.load(PATH_TO_MODEL)
    loader = DataLoader(VideoStreamDataset(PATH_TO_VIDEO), batch_size=5)

    labels = []
    for _,frame in enumerate(loader):
        with torch.no_grad():
            output = model(frame)
            _, preds = torch.max(output, dim=1)
            labels += preds.tolist()
    
    cap = cv.VideoCapture(PATH_TO_VIDEO)
    frame_width = int(cap.get(3)) 
    frame_height = int(cap.get(4)) 
    size = (frame_width, frame_height) 
    result = cv.VideoWriter(PATH_TO_RESULT, cv.VideoWriter_fourcc(*'MJPG'), 30, size) 
    j = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break 
        augmentImage(frame, revised_classes[labels[j]])
        j += 1
        cv.imshow('Frame', frame)
        result.write(frame)
        if cv.waitKey(25) & 0xFF == ord('q'): 
            break
    
    cap.release() 
    result.release() 
    cv.destroyAllWindows() 

if __name__ == '__main__':
    main()