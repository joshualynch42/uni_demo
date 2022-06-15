import cv2
import numpy as np


class SensorReplay:
    def __init__(self, size, **kwargs):
        self._size = size
        pass

    def __enter__(self): 
        return self
        
    def __exit__(self, typ, value, tb): 
        pass    
        
    def process(self, num_frames, outfile, start_frame=0):
        frame = np.empty((num_frames, self._size[0], self._size[1], 1))
        for i in range(0, num_frames):
            new_frame = cv2.imread(outfile[:-4] + '_0' + outfile[-4:])
            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
            frame[i,] = new_frame[..., np.newaxis]
        return frame
    

def main():
    None

if __name__ == '__main__':
    main()
