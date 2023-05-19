import cv2 as cv
import MJPEG
from utils import *

video_name = 'video'
video_format = ('.mp4', '.avi', '.mjpeg')
compression_video_name = 'compressed_video'

def compress(frame):
    Ycr = rgb2ycbcr(frame)
    quant_size = 5
    block_size = 8
    quants = [quant_size]
    blocks = [(block_size,block_size)]  
    obj = MJPEG.jpeg(Ycr, quants)
    for bx, by in blocks:
        return obj.intiate(bx,by)
    

def main():
    video_input = cv.VideoCapture(video_name + video_format[0])
    
    fps = int(video_input.get(cv.CAP_PROP_FPS))
    width = int(video_input.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video_input.get(cv.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    
    video_output = cv.VideoWriter(f"{compression_video_name+video_format[0]}", fourcc, fps, (width,  height))
    
    if(video_input.isOpened() is False):
        print('Video is not opened')
        return

    while(video_input.isOpened() is True):
        ret, frame = video_input.read()
        if(ret is False):
            break
        
        compressed_img = compress(frame)
        video_output.write(compressed_img)
    
    video_output.release()
    video_input.release()
    
    
if __name__ == '__main__':
    main()