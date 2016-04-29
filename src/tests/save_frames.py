#!/usr/bin/python

import pygiftgrab

file_path = 'python-test-file.mp4'
frame_rate = 20
recording_duration = 0.1  # min
num_frames = int(recording_duration * 60 * frame_rate)
device_type = pygiftgrab.Device.DVI2PCIeDuo_SDI
storage_type = pygiftgrab.Storage.File_H265

try:
    source_device = pygiftgrab.VideoSourceOpenCV(0)
    print source_device.get_frame_rate()
except (RuntimeError, IOError) as e:
    print e.message

try:
    source_file = pygiftgrab.VideoSourceOpenCV("/home/dzhoshkun/data/mosaic/imageUndistorted_000001.mp4")
    print source_file.get_frame_rate()
except (RuntimeError, IOError) as e:
    print e.message

frame = pygiftgrab.VideoFrame_BGRA(False)  # to avoid "thin wrappers" required for default args
try:
    source = pygiftgrab.Factory.connect(device_type)
    target = pygiftgrab.Factory.writer(storage_type)
    target.init(file_path, frame_rate)
    for i in range(1, num_frames+1):
        source.get_frame(frame)
        print 'Frame ' + str(i) + '/' + str(num_frames) + \
              ' is ' + str(frame.cols()) + ' x ' + str(frame.rows())
        target.append(frame)
        # TODO
        # sleep(inter_frame_msec)
    target.finalise()
    pygiftgrab.Factory.disconnect(pygiftgrab.Device.DVI2PCIeDuo_SDI)
except (RuntimeError, IOError) as e:
    print e.message
