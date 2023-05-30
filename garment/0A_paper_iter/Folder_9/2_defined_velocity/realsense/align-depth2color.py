## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.


import pyrealsense2 as rs
import numpy as np
import cv2
import os 
import time
instance_number = int(np.load('instance_number.npy')[0])

view_number = 0
folder_create_flag = 0
record_flag = 0
time_list = []

pipeline = rs.pipeline()
config = rs.config()
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
pc = rs.pointcloud()

device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))
found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
# d_intri = depth_profile.get_intrinsics()
# d_fx, d_fy, d_height, d_model, d_ppx, d_ppy, d_width = d_intri.fx, d_intri.fy, d_intri.height,\
#     d_intri.model, d_intri.ppx, d_intri.ppy, d_intri.width

# rgb_profile = rs.video_stream_profile(profile.get_stream(rs.str.2feam.color))
# r_intri = rgb_profile.get_intrinsics()
# r_fx, r_fy, r_height, r_model, r_ppx, r_ppy, r_width = r_intri.fx, r_intri.fy, r_intri.height,\
#     r_intri.model, r_intri.ppx, r_intri.ppy, r_intri.width


#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 3 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

align_to = rs.stream.color
align = rs.align(align_to)



dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_250)
parameters =  cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)


depth_profile = profile.get_stream(rs.stream.depth) # Fetch stream profile for depth stream
intr = depth_profile.as_video_stream_profile().get_intrinsics()
print(intr.fx, intr.fy, intr.ppx, intr.ppy,intr.width,intr.height,intr.coeffs)
# mtx = np.array([
#     [intr.fx, 0.0, intr.ppx],
#     [0.0, intr.fy, intr.ppy],
#     [0.0, 0.0, 1.0]])
mtx = np.load('mtx.npy')
dist = np.load('distortion.npy')
marker_legnth = 0.04
objp = np.array([[-marker_legnth / 2,  marker_legnth / 2, 0.0],
                [ marker_legnth / 2,  marker_legnth / 2, 0.0],
                [ marker_legnth / 2, -marker_legnth / 2, 0.0],
                [-marker_legnth / 2, -marker_legnth / 2, 0.0]])
axis = np.float32([[0.02,0,0], [0,0.02,0], [0,0,-0.02]]).reshape(-1,3)
#axis = np.float32([0.02, 0.0, 0.0])

def draw(img, corners, imgpts):
    #print((int(imgpts[0,0,0]),int(imgpts[0,0,1])))
    img = cv2.line(img, (int(corners[0,0,0]), int(corners[0,0,1])),  (int(imgpts[0,0,0]),int(imgpts[0,0,1])), (0,0,255), 2)
    #img = cv2.line(img, (int(corners[0,0,0]), int(corners[0,0,1])), (int(imgpts[1,0,0]),int(imgpts[1,0,1])), (0,255,0), 2)
    #img = cv2.line(img, (int(corners[0,0,0]), int(corners[0,0,1])), (int(imgpts[2,0,0]),int(imgpts[2,0,1])), (255,0,0), 2)

    return img
# Streaming loop


# v, t = points.get_vertices(), points.get_texture_coordinates()
# verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
# texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv

# Validate that both frames are valid

def rtvec_to_matrix(rvec, tvec):
 	T = np.eye(4)
 	R, jac = cv2.Rodrigues(rvec)
 	T[:3, :3] = R
 	T[:3, 3] = tvec.reshape(3,)
 	return T





try:
    while True:

        frames = pipeline.wait_for_frames()

        aligned_frames = align.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()
        
        points = pc.calculate(aligned_depth_frame)
        mapped_frame = color_frame
        pc.map_to(mapped_frame)
        v, t = points.get_vertices(), points.get_texture_coordinates()
        verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
        texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())

        color_image = np.asanyarray(color_frame.get_data())
        markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(color_image)

        #cv2.aruco.drawDetectedMarkers(color_image, markerCorners, markerIds)
        if not markerIds is None:
            for markerId in markerIds:
                if markerId == 0:
                    ret,rvecs, tvecs = cv2.solvePnP(objp, markerCorners[0], mtx, dist)
                    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
                    #img = draw(color_image,markerCorners[0],imgpts)
                    color_ori = color_image.copy()
                    cv2.drawFrameAxes(color_image, mtx, dist, rvecs, tvecs, 0.02, 3)
                    marker_x, marker_y = int(markerCorners[0][0][0][0]), int(markerCorners[0][0][0][1])
                    #print(marker_x, marker_y)
                    print(aligned_depth_frame.get_distance(marker_y,marker_y),tvecs[2] * 100)
                    

            


        # # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

        #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        #images = np.hstack((bg_removed, depth_colormap))

        cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Align Example', 640, 480)  
        cv2.imshow('Align Example', bg_removed)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
        if key & 0xFF == ord('s'):
            if not folder_create_flag:
                print('create folder')
                os.mkdir('./data/{instance:d}'.format(instance = instance_number))
                os.mkdir('./data/{instance:d}/rgb'.format(instance = instance_number))
                os.mkdir('./data/{instance:d}/depth'.format(instance = instance_number))
                os.mkdir('./data/{instance:d}/pcd'.format(instance = instance_number))
                os.mkdir('./data/{instance:d}/vert'.format(instance = instance_number))
                os.mkdir('./data/{instance:d}/uv'.format(instance = instance_number))
                folder_create_flag = 1
            record_flag =  1
            np.save(f'./data/{instance_number:d}/rvec.npy',rvecs)
            np.save(f'./data/{instance_number:d}/tvec.npy',tvecs)
            start_time = time.time()
            while view_number < 100:
                cv2.imwrite('./data/{instance:d}/rgb/rgb_img_{view:d}.jpg'.format(instance = instance_number,view = view_number), color_ori)
                #cv2.imwrite('./data/{instance:d}/depth/depth_img_{view:d}.png'.format(instance = instance_number,view = view_number), depth_image)
                np.save(f'./data/{instance_number:d}/vert/vert_{view_number:d}.npy',verts)
                np.save(f'./data/{instance_number:d}/uv/uv_{view_number:d}.npy',texcoords)
                #points.export_to_ply('./data/{instance:d}/pcd/pcd_{view:d}.ply'.format(instance = instance_number,view = view_number), mapped_frame)
                time_list.append(time.time() - start_time)
                print('instance: {instance:d} view {view:d} saved'.format(instance = instance_number,view = view_number))
                view_number += 1
            np.save('time.npy',np.array(time_list))
            if key & 0xFF == ord('p'):
                record_flag = 0
            # cv2.destroyAllWindows()
            # break
        if key & 0xFF == ord('k'):
            folder_create_flag = 0
            view_number = 0
            instance_number += 1
            A = np.array([instance_number])
            np.save('instance_number.npy',A)
            print('new instance {instance:d}'.format(instance = instance_number))
finally:
    pipeline.stop()
