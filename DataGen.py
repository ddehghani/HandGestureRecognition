import os
import cv2 as cv
from Util import bgremove
from tqdm import trange

VIDEO_DIR = r'/Users/ddehghani/Desktop/6240 ML/Final Project/video_data/2(top).MOV'
TOTAL_SAMPLES = 1000 # number of samples taken out of video
TARGET_DIR = r'./2(top)/'

def main():
    if not os.path.isdir(TARGET_DIR):
        os.mkdir(TARGET_DIR)
    os.chdir(TARGET_DIR)

    cap = cv.VideoCapture(VIDEO_DIR)

    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    sample_ratio = frame_count // TOTAL_SAMPLES

    for frame_index in trange(frame_count):
        _ , frame = cap.read()
        if frame_index % sample_ratio != 0:
            continue
        frame = bgremove(frame)
        cv.imwrite(f'{frame_index//sample_ratio}.jpg', frame)
    
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()