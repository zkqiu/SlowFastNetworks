import os
from pathlib import Path
import time
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset


class VideoDataset(Dataset):

    def __init__(self, directory, mode='train', clip_len=8, frame_sample_rate=1):
        folder = Path(directory)/mode/'rgb'  # get the directory of the specified split
        self.clip_len = clip_len

        self.short_side = [256, 320]
        self.crop_size = 224
        self.frame_sample_rate = frame_sample_rate
        self.mode = mode


        self.fnames, labels = [], []
        for label in sorted(os.listdir(folder)):#[0:20]
            for fname in os.listdir(os.path.join(folder, label)):
                self.fnames.append(os.path.join(folder, label, fname))
                labels.append(label)
        # prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label:index for index, label in enumerate(sorted(set(labels)))} 
        # convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

        label_file = str(len(os.listdir(folder)))+'class_labels.txt'
        with open(label_file, 'w') as f:
            for id, label in enumerate(sorted(self.label2index)):
                f.writelines(str(id + 1) + ' ' + label + '\n')

    def __getitem__(self, index):
        # loading and preprocessing. TODO move them to transform classes
        start = time.time()
        buffer = self.loadvideo_frames(self.fnames[index])
        #time1 = time.time()-start
        #_tmp = self.get64imgs(self.fnames[index])
        #print (_tmp[0][0][0][0])
        #time2 = time.time()-start-time1
        #_tmp = self._get64imgs(self.fnames[index])


        #print(time.time() - start)
        while buffer.shape[0]<self.clip_len :
            #print (buffer.shape)
            index = np.random.randint(self.__len__())
            buffer = self.loadvideo_frames(self.fnames[index])
        time1 = time.time() - start
        if self.mode == 'train' or self.mode == 'training':
            buffer = self.randomflip(buffer)
        time2 = time.time() - start - time1
        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        time3 = time.time() - start - time1 - time2
        buffer = self.normalize(buffer)
        time4 = time.time() - start - time1 - time2 - time3
        buffer = buffer.astype(np.float)
        buffer = self.to_tensor(buffer)

        #print(time1, time2, time3,time4)
        return buffer, self.label_array[index]

    def to_tensor(self, buffer):
        # convert from [D, H, W, C] format to [C, D, H, W] (what PyTorch uses)
        # D = Depth (in this case, time), H = Height, W = Width, C = Channels
        return buffer.transpose((3, 0, 1, 2))

    def loadvideo(self, fname):
        remainder = np.random.randint(self.frame_sample_rate)
        # initialize a VideoCapture object to read video data into a numpy array
        capture = cv2.VideoCapture(fname)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if frame_height < frame_width:
            resize_height = np.random.randint(self.short_side[0], self.short_side[1] + 1)
            try:
                resize_width = int(float(resize_height) / frame_height * frame_width)
            except:
                print (fname)
        else:
            resize_width = np.random.randint(self.short_side[0], self.short_side[1] + 1)
            try:
                resize_height = int(float(resize_width) / frame_width * frame_height)
            except:
                print (fname)

        # create a buffer. Must have dtype float, so it gets converted to a FloatTensor by Pytorch later
        start_idx = 0
        end_idx = frame_count-1
        frame_count_sample = frame_count // self.frame_sample_rate - 1
        if frame_count>300:
            end_idx = np.random.randint(300, frame_count)
            start_idx = end_idx - 300
            frame_count_sample = 301 // self.frame_sample_rate - 1
        buffer = np.empty((frame_count_sample, resize_height, resize_width, 3), np.dtype('float32'))

        count = 0
        retaining = True
        sample_count = 0

        # read in each frame, one at a time into the numpy buffer array
        while (count <= end_idx and retaining):
            retaining, frame = capture.read()
            if count < start_idx:
                count += 1
                continue
            if retaining is False or count>end_idx:
                break
            if count%self.frame_sample_rate == remainder and sample_count < frame_count_sample:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # will resize frames if not already final size

                if (frame_height != resize_height) or (frame_width != resize_width):
                    frame = cv2.resize(frame, (resize_width, resize_height))
                buffer[sample_count] = frame
                sample_count = sample_count + 1
            count += 1
        capture.release()
        return buffer

    def loadvideo_frames(self, fname):
        start = time.time()
        remainder = np.random.randint(self.frame_sample_rate)
        # initialize a VideoCapture object to read video data into a numpy array
        frame_names = sorted(os.listdir(fname))
        first_frame = cv2.imread(os.path.join(fname,frame_names[0]))
        frame_count = len(frame_names)
        frame_width = first_frame.shape[1]
        frame_height = first_frame.shape[0]

        if frame_height < frame_width:
            resize_height = np.random.randint(self.short_side[0], self.short_side[1] + 1)
            resize_width = int(float(resize_height) / frame_height * frame_width)
        else:
            resize_width = np.random.randint(self.short_side[0], self.short_side[1] + 1)
            resize_height = int(float(resize_width) / frame_width * frame_height)

        time1 = time.time()-start
        # create a buffer. Must have dtype float, so it gets converted to a FloatTensor by Pytorch later
        start_idx = 0
        end_idx = frame_count-1
        frame_count_sample = frame_count // self.frame_sample_rate - 1

        if frame_count_sample<self.clip_len:
            return np.empty((frame_count_sample, resize_height, resize_width, 3), np.dtype('float32'))

        if frame_count>300:
            end_idx = np.random.randint(300, frame_count)
            start_idx = end_idx - 300
            frame_count_sample = 301 // self.frame_sample_rate - 1
        #resize_height, resize_width = (224,224)
        #resize_flag = False
        #if frame_width<224 or frame_height<224:
            #resize_height, resize_width = (224,224)
            #resize_flag = True
        #else:
            #resize_height, resize_width = (frame_height, frame_width)
        #buffer = np.empty((frame_count_sample, resize_height, resize_width, 3), np.dtype('float32'))
        buffer = np.empty((self.clip_len, resize_height, resize_width, 3), np.dtype('float32'))
        #buffer = []
        count = 0
        sample_count = 0
        time2 = time.time()-start-time1
        # read in each frame, one at a time into the numpy buffer array
        index = list(range(start_idx+remainder,end_idx+1,self.frame_sample_rate))
        time_index = np.random.randint(len(index)-64)
        index = index[time_index:time_index+64]

        for idx in index:
            frame = cv2.imread(os.path.join(fname, frame_names[idx]))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (resize_width, resize_height))
            buffer[sample_count] = frame
            sample_count = sample_count + 1

        '''
        while (count <= end_idx):
            if count < start_idx:
                count += 1
                continue
            if count>end_idx:
                break
            if count%self.frame_sample_rate == remainder and sample_count < frame_count_sample:
                _start = time.time()
                frame = cv2.imread(os.path.join(fname,frame_names[count]))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                _time1 = time.time() - _start
                #if (frame_height != resize_height) or (frame_width != resize_width):
                #if resize_flag:
                frame = cv2.resize(frame, (resize_width, resize_height))
                #frame = cv2.resize(frame, (224, 224))
                #shape1 = frame.shape
                #print (shape0, ' ', shape1,)
                _time2 = time.time() - _time1 - _start

                buffer[sample_count] = frame
                #buffer.append(frame)
                _time3 = time.time() - _time1 - _start - _time2
                #print (_time1, _time2, _time3)
                sample_count = sample_count + 1
            count += 1
        '''
        time3 = time.time()-time1-time2-start
        #print ('%.2f,  %.2f,  %.2f' % (time1,time2,time3))

        return buffer
        #return np.array(buffer)


    def get64imgs(self, fname):
        frame_names = sorted(os.listdir(fname))
        buffer = []
        if len(frame_names)>=64:
            for i,name in enumerate(frame_names):
                if i<64:
                    img =  cv2.imread(os.path.join(fname, name))
                    img = cv2.resize(img,(320,256))
                    buffer.append(img)
                else:
                    break
        else: return 0

        return np.array(buffer)


    def _get64imgs(self, fname):
        frame_names = sorted(os.listdir(fname))
        buffer = []
        if len(frame_names)>=64:
            for i,name in enumerate(frame_names):
                if i<64:
                    img =  cv2.imread(os.path.join(fname, name))
                    #img = cv2.resize(img,(320,256))
                    buffer.append(img)
                else:
                    break
        else: return 0

        return np.array(buffer)



    def crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering
        #time_index = np.random.randint(buffer.shape[0] - clip_len)
        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        # crop and jitter the video using indexing. The spatial crop is performed on 
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[:,#time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        return buffer                

    def normalize(self, buffer):
        # Normalize the buffer
        # buffer = (buffer - 128)/128.0
        #for i, frame in enumerate(buffer):
            #frame = (frame - np.array([[[128.0, 128.0, 128.0]]]))/128.0
            #buffer[i] = frame
        buffer = (buffer-np.array([[[[128.0, 128.0, 128.0]]]]))/128.0
        #print (buffer.shape)
        return buffer

    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""
        if np.random.random() < 0.5:
            #for i, frame in enumerate(buffer):
                #buffer[i] = cv2.flip(frame, flipCode=1)
            buffer = buffer[:,:,::-1,:]

        return buffer

    def __len__(self):
        return len(self.fnames)


if __name__ == '__main__':

    datapath = '/data1/data/Kinetics'
    train_dataloader = \
        DataLoader( VideoDataset(datapath, mode='train',clip_len=64, frame_sample_rate=2), batch_size=10, shuffle=True, num_workers=1)
    for step, (buffer, label) in enumerate(train_dataloader):
        print("label: ", label)
        print (buffer.shape)
