class SensorDummy:
    def __init__(self, **kwargs):
        pass

    def __enter__(self): 
        return self
        
    def __exit__(self, typ, value, tb): 
        pass    
        
    def process(self, num_frames, outfile, start_frame=0):
        return 0

    def async_process(self, num_frames=1, outfile=''):
        pass    

    def async_cancel(self):
        pass
    
    def async_result(self):
        return None

def main():
    None

if __name__ == '__main__':
    main()
