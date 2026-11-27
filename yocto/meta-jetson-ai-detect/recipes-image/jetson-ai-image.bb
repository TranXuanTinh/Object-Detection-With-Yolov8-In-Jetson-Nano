SUMMARY = "Minimal Production Image for Jetson Nano AI Edge Node"
DESCRIPTION = "Target image for Jetson Nano 4GB containing NVIDIA drivers, \
               CUDA, TensorRT, and edge-ai-manager daemon."

IMAGE_FEATURES += "ssh-server-openssh"

IMAGE_INSTALL = " \
    packagegroup-core-boot \
    packagegroup-core-full-cmdline \
    \
    kernel-modules \
    tegra-libraries \
    tegra-libraries-argus \
    cuda-toolkit \
    cuda-libraries \
    cudnn \
    tensorrt-core \
    tensorrt-plugins \
    \
    opencv \
    gstreamer1.0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-tegra \
    nvargus-daemon \
    \
    jetson-ai-detect \
    \
    tegra-nvpmodel \
    tegra-tools-jetson-clocks \
    tegra-tools-tegrastats \
    systemd \
    networkmanager \
    openssh \
    htop \
    v4l-utils \
"

IMAGE_LINGUAS = ""
NO_RECOMMENDATIONS = "1"

inherit core-image

