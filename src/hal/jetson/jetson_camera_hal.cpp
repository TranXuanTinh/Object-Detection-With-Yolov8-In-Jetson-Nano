#include "jetson_camera_hal.h"
#include <sstream>

namespace edge_ai { namespace hal {

JetsonCameraHAL::JetsonCameraHAL() = default;

JetsonCameraHAL::~JetsonCameraHAL() {
    release();
}

std::string JetsonCameraHAL::buildGstreamerPipeline(int sensor_id, int width, int height, int fps, int flip_method) {
    // RidgeRun / NVIDIA Jetson best practice:
    // Keep NVMM zero-copy from ISP through VIC (nvvidconv), avoid CPU videoconvert by outputting BGRx directly,
    // use a leaky queue to guarantee zero frame-accumulation latency, and disable sink clock sync.
    std::ostringstream ss;
    ss << "nvarguscamerasrc sensor-id=" << sensor_id << " ! "
       << "video/x-raw(memory:NVMM), width=(int)" << width << ", height=(int)" << height
       << ", format=(string)NV12, framerate=(fraction)" << fps << "/1 ! "
       << "queue max-size-buffers=1 leaky=downstream ! "
       << "nvvidconv flip-method=" << flip_method << " ! "
       << "video/x-raw, format=(string)BGRx ! "
       << "appsink drop=true max-buffers=1 sync=false";
    return ss.str();
}

std::string JetsonCameraHAL::buildRtspGstreamerPipeline(const std::string& uri) {
    // Hardware-accelerated RTSP decode pipeline via Tegra NVDEC (nvv4l2decoder) + VIC (nvvidconv)
    std::ostringstream ss;
    ss << "rtspsrc location=" << uri << " latency=100 ! "
       << "rtph264depay ! h264parse ! nvv4l2decoder ! "
       << "queue max-size-buffers=1 leaky=downstream ! "
       << "nvvidconv ! "
       << "video/x-raw, format=(string)BGRx ! "
       << "appsink drop=true max-buffers=1 sync=false";
    return ss.str();
}

bool JetsonCameraHAL::open(const std::string& device,
                           int width, int height, int fps,
                           int sensor_id, int flip_method) {
    release();

    format_.width = width;
    format_.height = height;
    format_.fps = fps;
    format_.pixel_fmt = PixelFormat::BGR;

    if (device.rfind("rtsp://", 0) == 0) {
        std::string gst = buildRtspGstreamerPipeline(device);
        if (cap_.open(gst, cv::CAP_GSTREAMER)) {
            format_.pixel_fmt = PixelFormat::BGRx;
            return true;
        }
    }

    if (device == "csi" || device == "/dev/video0_csi" || sensor_id >= 0) {
        std::string gst = buildGstreamerPipeline(sensor_id, width, height, fps, flip_method);
        if (cap_.open(gst, cv::CAP_GSTREAMER)) {
            format_.pixel_fmt = PixelFormat::BGRx;
            return true;
        }
    }

    // Fallback to standard V4L2 USB camera or video file
    cap_.open(device, cv::CAP_V4L2);
    if (cap_.isOpened()) {
        cap_.set(cv::CAP_PROP_FRAME_WIDTH, width);
        cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height);
        cap_.set(cv::CAP_PROP_FPS, fps);
        format_.pixel_fmt = PixelFormat::BGR;
    }

    return cap_.isOpened();
}

bool JetsonCameraHAL::readFrame(RawFrame& frame) {
    if (!cap_.isOpened()) return false;

    if (!cap_.read(frame.mat) || frame.mat.empty()) {
        return false;
    }

    frame.sequence = ++frame_seq_;
    return true;
}

FrameFormat JetsonCameraHAL::getFormat() const {
    return format_;
}

void JetsonCameraHAL::release() {
    if (cap_.isOpened()) {
        cap_.release();
    }
}

bool JetsonCameraHAL::isOpened() const {
    return cap_.isOpened();
}

}} // namespace edge_ai::hal

