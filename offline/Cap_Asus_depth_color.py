import numpy as np
import cv2
from primesense import openni2

def capture_color_and_depth_xtion():
    openni2.initialize("C:/Program Files/OpenNI2/Redist")
    dev = openni2.Device.open_any()

    #start depth and color streams
    depth_stream = dev.create_depth_stream()
    depth_stream.start()
    color_stream = dev.create_color_stream()
    color_stream.start()

    #read frames
    depth_frame = depth_stream.read_frame()
    color_frame = color_stream.read_frame()

    #convert depth frame to numpy array 
    depth_data = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16)
    d_width = depth_stream.get_video_mode().resolutionX
    d_height = depth_stream.get_video_mode().resolutionY
    depth_image = depth_data.reshape((d_height, d_width))  #depth map

    cv2.imwrite('C:/Users/engmn/Desktop/Xtion_depth_raw_no_norm.png', depth_image)

    depth_display = np.clip(depth_image, 300, 1500)
    depth_display = ((depth_display - 300) / (1500 - 300) * 255).astype(np.uint8)
    depth_colormap = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
    cv2.imshow('Depth Visualization', depth_colormap)

    #convert color frame to numpy array 
    color_data = np.frombuffer(color_frame.get_buffer_as_uint8(), dtype=np.uint8)
    c_width = color_stream.get_video_mode().resolutionX
    c_height = color_stream.get_video_mode().resolutionY
    color_image = color_data.reshape((c_height, c_width, 3))
    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

    cv2.imwrite('C:/Users/engmn/Desktop/Xtion_color_test.jpg', color_image)
    cv2.imshow('Color Image', color_image)

    print(f"Depth map saved (raw): shape={depth_image.shape}, dtype={depth_image.dtype}")
    print(f"Color image shape: {color_image.shape}")
    print(f"Valid depth pixels: {np.count_nonzero(depth_image)} / {depth_image.size}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    depth_stream.stop()
    color_stream.stop()
    openni2.unload()

if __name__ == "__main__":
    capture_color_and_depth_xtion()
