import cv2 as cv


class VideoCapture:
    def __init__(self, filename, down_sample=1):
        self.filename = filename
        self.cap = cv.VideoCapture(self.filename)
        self.down_sample = down_sample
        self.mspos = 0
        self.cache = {}
        self.extents()
        self.shape()

    def __iter__(self):
        return self

    def __next__(self):
        self.mspos = self.cap.get(cv.CAP_PROP_POS_MSEC)
        ret, frame = self.cap.read()
        if ret != True:
            self.cap.release()
            raise StopIteration()
        if self.down_sample != 1:
            output = cv.resize(
                frame,
                (
                    int(
                        self.cap.get(cv.CAP_PROP_FRAME_WIDTH) //
                        self.down_sample),
                    int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)) //
                    self.down_sample,
                ),
            )
        else:
            output = frame
        return output

    def extents(self):
        if 'extents' not in self.cache:
            self.cache['extents'] = (
                int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH)) // self.down_sample,
                int(
                    self.cap.get(cv.CAP_PROP_FRAME_HEIGHT) //
                    self.down_sample),
                int(self.cap.get(cv.CAP_PROP_FRAME_COUNT)),
            )
        return self.cache['extents']

    def shape(self):
        if 'shape' not in self.cache:
            self.cache['shape'] = (
                int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH)) // self.down_sample,
                int(
                    self.cap.get(cv.CAP_PROP_FRAME_HEIGHT) //
                    self.down_sample),
            )
        return self.cache['shape']

    def msec(self):
        return self.mspos
