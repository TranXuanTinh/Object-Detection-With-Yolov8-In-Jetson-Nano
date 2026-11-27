# Ensure OpenCV is built with GStreamer and V4L2 support so cv::VideoCapture(..., cv::CAP_GSTREAMER) works on Jetson Nano.
PACKAGECONFIG:append = " gstreamer v4l"
