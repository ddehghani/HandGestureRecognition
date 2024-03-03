import os
import cv2 as cv
from tqdm import trange

VIDEO_DIR = r'/Users/ddehghani/Desktop/6240 ML/Final Project/video_data/1(bot).MOV'
TOTAL_SAMPLES = 500
TARGET_DIR = r'1(bot)/'

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

def bgremove(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) 
    # apply binary thresholding
    _ , thresh = cv.threshold(gray, 90, 255, cv.THRESH_BINARY)
    return thresh

if __name__ == '__main__':
    main()