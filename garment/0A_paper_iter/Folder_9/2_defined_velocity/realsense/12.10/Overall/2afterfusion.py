# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 17:03:38 2019

@author: lancerhly
"""

# -*- coding: utf-8 -*-

from uvctypes import *
import time
import cv2
import numpy as np
import pyrealsense2 as rs
import platform

i = 0
def t():
	global i
	i +=1
try:
  from queue import Queue
except ImportError:
  from Queue import Queue
  

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640,480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
align_to = rs.stream.color
align = rs.align(rs.stream.color)
depth_img = np.empty((480,640,3),dtype = np.float32)

def get3dpoints(cx,cy,fx,fy,x,y,depth):
    imx = (x-cx)/fx
    imy = (y-cy)/fy
    mx = depth*imx
    my = depth*imy
    this_point = [mx,my,depth]
    return(this_point)
fx = 616.0172119140625
fy = 615.6632690429688
cx = 311.49847412109375
cy = 229.34274291992188
x = np.arange(0,640)
y = np.arange(0,480)
x_c = np.tile(x, (480, 1))
y_c = np.tile(y,(640,1))
y_c = np.transpose(y_c)
x_real_imgplane = (x_c-cx)/fx
y_real_imgplane = (y_c-cy)/fy
pc = rs.pointcloud()
points = rs.points()

BUF_SIZE = 2
q = Queue(BUF_SIZE)
def py_frame_callback(frame, userptr):

  array_pointer = cast(frame.contents.data, POINTER(c_uint16 * (frame.contents.width * frame.contents.height)))
  data = np.frombuffer(
    array_pointer.contents, dtype=np.dtype(np.uint16)
  ).reshape(
    frame.contents.height, frame.contents.width
  ) 

  if frame.contents.data_bytes != (2 * frame.contents.width * frame.contents.height):
    return

  if not q.full():
    q.put(data)

PTR_PY_FRAME_CALLBACK = CFUNCTYPE(None, POINTER(uvc_frame), c_void_p)(py_frame_callback)

def ktof(val):
  return (1.8 * ktoc(val) + 32.0)

def ktoc(val):
  return (val - 27315) / 100.0

def raw_to_8bit(data):
  cv2.normalize(data, data, 0, 65535, cv2.NORM_MINMAX)
  np.right_shift(data, 8, data)
  return cv2.cvtColor(np.uint8(data), cv2.COLOR_GRAY2RGB)

def display_temperature(img, val_k, loc, color):
  val = ktof(val_k)
  cv2.putText(img,"{0:.1f} degF".format(val), loc, cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
  x, y = loc
  cv2.line(img, (x - 2, y), (x + 2, y), color, 1)
  cv2.line(img, (x, y - 2), (x, y + 2), color, 1)

def main():
  ctx = POINTER(uvc_context)()
  dev = POINTER(uvc_device)()
  devh = POINTER(uvc_device_handle)()
  ctrl = uvc_stream_ctrl()

  res = libuvc.uvc_init(byref(ctx), 0)
  if res < 0:
    print("uvc_init error")
    exit(1)

  try:
    res = libuvc.uvc_find_device(ctx, byref(dev), PT_USB_VID, PT_USB_PID, 0)
    if res < 0:
      print("uvc_find_device error")
      exit(1)

    try:
      res = libuvc.uvc_open(dev, byref(devh))
      if res < 0:
        print("uvc_open error")
        exit(1)

      print("device opened!")

      print_device_info(devh)
      print_device_formats(devh)

      frame_formats = uvc_get_frame_formats_by_guid(devh, VS_FMT_GUID_Y16)
      if len(frame_formats) == 0:
        print("device does not support Y16")
        exit(1)

      libuvc.uvc_get_stream_ctrl_format_size(devh, byref(ctrl), UVC_FRAME_FORMAT_Y16,
        frame_formats[0].wWidth, frame_formats[0].wHeight, int(1e7 / frame_formats[0].dwDefaultFrameInterval)
      )

      res = libuvc.uvc_start_streaming(devh, byref(ctrl), PTR_PY_FRAME_CALLBACK, None, 0)
      if res < 0:
        print("uvc_start_streaming failed: {0}".format(res))
        exit(1)

      try:
        while True:
          data = q.get(True, 500)
          if data is None:
            break
          data = cv2.resize(data[:,:], (640, 480))
          data_o = cv2.resize(data[:,:], (160, 120))
#==============================================================================
#           minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(data)
#           display_temperature(img, minVal, minLoc, (255, 0, 0))
#           display_temperature(img, maxVal, maxLoc, (0, 0, 255))
#==============================================================================
          timg = raw_to_8bit(data)
          timg_o = raw_to_8bit(data_o)
          frames = pipeline.wait_for_frames()
          aligned_frames = align.process(frames)
          aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
          color_frame = aligned_frames.get_color_frame()
          depth_image = np.asanyarray(aligned_depth_frame.get_data())
          depth_image = np.multiply(depth_scale,depth_image)
          color_image = np.asanyarray(color_frame.get_data())
          w_x = np.multiply(x_real_imgplane,depth_image)
          w_y = np.multiply(y_real_imgplane,depth_image)
          real_point = np.stack((w_x,w_y,depth_image),axis = -1)
          #grayrimg=cv2.cvtColor(rimg, cv2.COLOR_RGB2GRAY)
          graytimg=cv2.cvtColor(timg, cv2.COLOR_RGB2GRAY)
          graytimg_o=cv2.cvtColor(timg_o, cv2.COLOR_RGB2GRAY)
          final = graytimg.copy()
          final_o = graytimg_o.copy()
          '''
          corners = cv2.goodFeaturesToTrack(graytimg,36,0.001,30)
          corners = np.int0(corners)
          for k in corners:
              x,y = k.ravel()
              cv2.circle(graytimg,(x,y),3,255,-1)
          '''
          cv2.imshow('rimg',color_image)
          cv2.imshow('timg',graytimg)
          k = cv2.waitKey(1) & 0xFF
          

          if k==ord('s'):
              cv2.imwrite('thermal_%i.jpg'%i,final)
              #cv2.imwrite('o_thermal_%i.jpg'%i,final_o)
              cv2.imwrite('rgb_%i.jpg'%i,color_image)
              np.save('world_%i'%i,real_point)
              '''
              pc.map_to(color_frame)
              points = pc.calculate(aligned_depth_frame)
              points.export_to_ply("pointCloud%i.ply"%i, color_frame)
              '''
              t()
              continue
          elif k == 27:
              break


        cv2.destroyAllWindows()
        pipeline.stop()
      finally:
        libuvc.uvc_stop_streaming(devh)

      print("done")
    finally:
      libuvc.uvc_unref_device(dev)
  finally:
    libuvc.uvc_exit(ctx)

if __name__ == '__main__':
  main()
