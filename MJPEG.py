import numpy as np
import scipy.fftpack as fftpack
import zlib
import cv2

from utils import *

class jpeg:
    
    def __init__(self, im,quants):
        self.image = im
        self.quants = quants
        super().__init__()

    def encode_quant(self,quant):
        return (self.enc / quant).astype(np.int8)

    def decode_quant(self,quant):
        return (self.encq * quant).astype(float)

    def encode_dct(self,bx, by):
        new_shape = (
            self.image.shape[0] // bx * bx,
            self.image.shape[1] // by * by,
            3)
        new = self.image[
            :new_shape[0],
            :new_shape[1]
        ].reshape((
            new_shape[0] // bx, bx,
            new_shape[1] // by, by,
            3))
        return fftpack.dctn(new, axes=[1, 3], norm='ortho')


    def decode_dct(self, bx, by):
        return fftpack.idctn(self.decq, axes=[1, 3], norm='ortho').reshape((
                                self.decq.shape[0]*bx,
                                self.decq.shape[2]*by,
                                3))
    
    def encode_zip(self):
        return zlib.compress(self.encq.astype(np.int8).tobytes())


    def decode_zip(self):
        return np.frombuffer(zlib.decompress(self.encz), dtype=np.int8).astype(float).reshape(self.encq.shape)
    

    def intiate(self,bx,by):
        QUANTIZATION_MAT = (np.array([[16,11,10,16,24,40,51,61],
                                     [12,12,14,19,26,58,60,55],
                                     [14,13,16,24,40,57,69,56],
                                     [14,17,22,29,51,87,80,62],
                                     [18,22,37,56,68,109,103,77],
                                     [24,35,55,64,81,104,113,92],
                                     [49,64,78,87,103,121,120,101],
                                     [72,92,95,98,112,100,103,99]]).reshape((1, bx, 1, by, 1)))
        
        self.enc = self.encode_dct(bx, by)
        self.encq = self.encode_quant(QUANTIZATION_MAT)
        self.encz = self.encode_zip()
        self.decz = self.decode_zip()
        self.decq = self.decode_quant(QUANTIZATION_MAT)
        self.dec = self.decode_dct(bx, by)
        img_bgr = ycbcr2rgb(self.dec)
        return img_bgr.astype(np.uint8)