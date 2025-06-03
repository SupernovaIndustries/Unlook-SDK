// maybe loosing frame with CV? count frames? how? try matlab?

#include "v4l2-controls.h"
#include "/home/bsa/mlx_driver/linux/include/uapi/linux/v4l2-subdev.h"  // v4l2
#include "mlx7502x.h"
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <chrono>

#define VIDEO_DEVICE "/dev/video0"
#define VIDEO_SUBDEVICE "/dev/v4l-subdev0"

static bool do_ext_ioctl(const int subdev, 
                            v4l2_ext_control *ctrl) {
    struct v4l2_ext_controls ext_ctrls;
    ext_ctrls.ctrl_class = 0;
    ext_ctrls.count = 1;
    ext_ctrls.controls = ctrl;    
    //printf("ext ctrl cid %x \n",VIDIOC_S_EXT_CTRLS);
    return ioctl(subdev, VIDIOC_S_EXT_CTRLS, &ext_ctrls);
    
} 

template <typename T>
static bool set_array(const int subdev,
                          std::vector<T>& v,
                          int cid) {
    struct v4l2_ext_control ctrl = { 0, 0, 0, 0 };

    ctrl.id = cid;
    ctrl.ptr = &v[0];
    ctrl.size = v.size() * sizeof(T);

    return do_ext_ioctl(subdev, &ctrl);
}

template <typename T>
static bool set_array_check(const int subdev,
                                const std::vector<T>& v,
                                int cid) {
    auto loc_v = v;
    return set_array(subdev, loc_v, cid) && v == loc_v;
}

static bool set_phase_sequence(const int subdev,
                                   const std::vector<uint16_t>& v_phase_sequence) {     
    printf("funct set_phase_sequence %d %d %d %d \n", v_phase_sequence[0],v_phase_sequence[1],v_phase_sequence[2],v_phase_sequence[3]);
    return set_array_check(subdev, v_phase_sequence, V4L2_CID_TOF_PHASE_SEQ);
}

static bool set_time_integration(const int subdev,
                                   const std::vector<uint16_t>& v_time_integration) {     
    printf("funct set_time_integration %d %d %d %d \n", v_time_integration[0],v_time_integration[1],v_time_integration[2],v_time_integration[3]);
    return set_array_check(subdev, v_time_integration, V4L2_CID_TOF_TIME_INTEGRATION);
}

static bool set_frame_interval(const int subdev, const uint32_t fps) {
    printf("set_frame_interval %d \n",fps);

    int ret;
    struct v4l2_subdev_frame_interval fi;
    memset(&fi, 0, sizeof(struct v4l2_subdev_frame_interval));

    fi.reserved[0] = V4L2_SUBDEV_FORMAT_ACTIVE; // TODO: linux 6.8 rename to which
    fi.interval.numerator = 1;
    fi.interval.denominator = fps;
    fi.which = V4L2_SUBDEV_FORMAT_ACTIVE;
    fi.pad = subdev;

    ret = ioctl(subdev,VIDIOC_SUBDEV_S_FRAME_INTERVAL, &fi);
    if (ret < 0) {
        printf("ioctl failed and returned errno %s \n",strerror(errno));
    }
    return ret;
}

bool do_ioctl(int m_fd, uint32_t id, void* val) {
    int result = ioctl(m_fd, id, val);
    if (result < 0) {
        //printf("subdev: Failed ioctl {:x}: {:s}.", id, strerror(errno));
        printf("subdev: Failed ioctl \n");
        return false;
    }
    return true;
}

bool set_tof_reg(int subfd, uint16_t addr, uint32_t value, int size) {
    struct v4l2_dbg_register ctrl;
    memset(&ctrl, 0, sizeof(ctrl));
    ctrl.match.type = V4L2_CHIP_MATCH_SUBDEV;
    ctrl.match.addr = 0;
    ctrl.reg = addr;
    ctrl.val = value;
    ctrl.size = size;

    return do_ioctl(subfd, VIDIOC_DBG_S_REGISTER, &ctrl);
}


cv::Mat Mat_shiftPixelValues_1chan(cv::Mat& image_1chan, int shiftValue) {  
    int rows = 480;
    int cols = 640;
  cv::Mat imageShifted = cv::Mat::zeros(rows, cols, CV_16SC1);
  for (int y = 0; y < image_1chan.rows; y++) {
    for (int x = 0; x < image_1chan.cols; x++) {
        // Access pixel value and shift it
        imageShifted.at<int16_t>(y, x) = 
            (image_1chan.at<int16_t>(y, x))<< shiftValue;
    }
  }
  return imageShifted;
}


// Function to apply sRGB gamma correction
cv::Mat gammaCorrect(const cv::Mat& img, double gamma = 2.2) {
    cv::Mat imgFloat;
    img.convertTo(imgFloat, CV_32F, 1.0 / 4095.0); // Normalize from 12-bit range to [0,1]

    // Apply sRGB gamma correction
    cv::Mat corrected;
    cv::pow(imgFloat, 1.0 / gamma, corrected);
    corrected *= 255.0; // Scale to 8-bit range

    cv::Mat img8bit;
    corrected.convertTo(img8bit, CV_8U); // Convert to 8-bit
    return img8bit;
}

cv::Mat double_mat_to_rgb(const cv::Mat& double_mat) {
    double min_val, max_val;
    cv::minMaxLoc(double_mat, &min_val, &max_val);
    cv::Scalar mean_val, std_val;
    cv::meanStdDev(double_mat, mean_val, std_val);
    
    //printf("magnitude min max mean std values %f %f %f %f\n", min_val, max_val, mean_val[0], std_val[0]);
    
    cv::Mat scaled_mat;
    double scale_factor = 255.0 / (max_val - min_val);
    double_mat.convertTo(scaled_mat, CV_8U, scale_factor, -min_val * scale_factor);
    //double_mat.convertTo(scaled_mat, CV_8U, scale_factor);

    cv::Mat rgb_mat;
    cv::cvtColor(scaled_mat, rgb_mat, cv::COLOR_GRAY2RGB);

    return rgb_mat;
}

cv::Mat double_mat_to_rgb_phase(const cv::Mat& double_mat) {
    double min_val, max_val;
    cv::minMaxLoc(double_mat, &min_val, &max_val);
    cv::Scalar mean_val, std_val;
    cv::meanStdDev(double_mat, mean_val, std_val); 
    
    //printf("phase min max mean std values %f %f %f %f %f %f \n", min_val, max_val, mean_val[0], std_val[0]);
    
    cv::Mat scaled_mat;
    double scale_factor = 255.0 / (360.0);
    double_mat.convertTo(scaled_mat, CV_8U, scale_factor, 0 * scale_factor);
    //double scale_factor = 255.0 / (2*std_val[0]);
    //double_mat.convertTo(scaled_mat, CV_8U, scale_factor, -mean_val[0] * scale_factor);

    cv::Mat rgb_mat;
    cv::cvtColor(scaled_mat, rgb_mat, cv::COLOR_GRAY2RGB);

    return rgb_mat;
}

// Function to convert Y12P grayscale to sRGB
cv::Mat convertY12PtoSRGB(cv::Mat& y12pImage) {
    // Apply gamma correction
    
    //cv::Mat y16Image;
    //y12pImage.convertTo(y16Image, 16); //sign extend
    Mat_shiftPixelValues_1chan(y12pImage,4);
    
    int16_t pixel;
    std::printf("Convert to 16-bit Pixel Array: \n");
    for (int i = 0; i < 20; ++i) {
        for (int j = 0; j < 10; ++j) {
            pixel = y12pImage.at<int16_t>(i, j); 
            std::printf("%08x = %5d,  ", pixel, pixel);
        }
        std::printf("\n");
    }      
    
    cv::Mat sRGB_Y = gammaCorrect(y12pImage);    

    // Convert grayscale to RGB
    cv::Mat sRGBImage;
    cv::cvtColor(sRGB_Y, sRGBImage, cv::COLOR_GRAY2BGR);
    
    return sRGBImage;
}

void printfMat(const cv::Mat& imageArray) {
    int16_t pixel;
    std::printf("Convert to 16-bit Pixel Array: \n");
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            pixel = imageArray.at<int16_t>(i, j); 
            std::printf("%08x = %5d,  ", pixel, pixel);
        }
        std::printf("\n");
    }
}

void printfMatf(const cv::Mat& imageArray) {
    double pixel;
    //std::printf("Convert to 16-bit Pixel Array: \n");
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 6; ++j) {
            pixel = imageArray.at<double>(i, j); 
            std::printf("%5f,  ", pixel);
        }
        std::printf("\n");
    }
}


   

////////////////////////////////////////////////////////////
///// MAIN /////////////////////////////////////////////////
////////////////////////////////////////////////////////////

int main() {
    
    std::vector<uint16_t> phase_sequence = { 0, 180, 90, 270 };
    printf("phase_sequence %d %d %d %d \n", phase_sequence[0],phase_sequence[1],phase_sequence[2],phase_sequence[3]);
    std::vector<uint16_t> time_integration = { 1000, 1000, 1000, 1000 };
    
    int fd = open(VIDEO_DEVICE, O_RDWR);
    if (fd < 0) {
        std::cerr << "Error: Could not open video device!" << std::endl;
        return 1;
    }
    else
        printf("opened %s device \n",VIDEO_DEVICE);
        
    int subfd = open(VIDEO_SUBDEVICE, O_RDWR);
    if (subfd < 0) {
        std::cerr << "Error opening subdevice" << std::endl;
        return -1;
    }        
    else
        printf("opened /dev/v4l-subdev0 subdevice \n");
    
    set_phase_sequence(subfd, phase_sequence);
    
    //set image format
    v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = 640;
    fmt.fmt.pix.height = 480;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_Y12P;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    int ioctl_return = ioctl(fd, VIDIOC_S_FMT, &fmt);
    if (ioctl_return < 0) {
        perror("VIDIOC_S_FMT failed");
        printf("error %d \n", ioctl_return);
        close(fd);
        return -1;
    }   
    
    set_time_integration(subfd, time_integration);
    
    //set output mode 
    uint16_t output_mode = 0;
    printf("Changing output_mode = %d \n", output_mode);
    struct v4l2_ext_control output_mode_ctrl ={ .id = V4L2_CID_MLX7502X_OUTPUT_MODE, .size = 0, 0, .value = output_mode};
    struct v4l2_ext_controls output_mode_ctrls;
    output_mode_ctrls.ctrl_class = 0;
    output_mode_ctrls.count = 1;
    output_mode_ctrls.controls = &output_mode_ctrl;
    printf("ext ctrl cid %x \n",V4L2_CID_MLX7502X_OUTPUT_MODE);
    ioctl_return = ioctl(subfd, VIDIOC_S_EXT_CTRLS, &output_mode_ctrls);
    if (ioctl_return < 0) {
        std::cerr << "Error setting control" << std::endl;
        printf("error %d \n", ioctl_return);
        printf("ioctl failed and returned errno %s \n",strerror(errno));
        close(subfd);
        return -1;
    }



    set_array_check(subfd, std::vector<uint32_t>(1, 10000000), V4L2_CID_TOF_FREQ_MOD);

    //set_frame_interval(fd, 12);
    //set frame_interval
    v4l2_streamparm streamparm;
    streamparm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd, VIDIOC_G_PARM, &streamparm) == -1) {
       close(fd);
       throw std::runtime_error("Failed to get stream parameters");
    }
    streamparm.parm.capture.timeperframe.numerator = 1;   // Set numerator to 1
    streamparm.parm.capture.timeperframe.denominator = 12; // Set denominator to desired FPS (e.g., 30)
    if (ioctl(fd, VIDIOC_S_PARM, &streamparm) == -1) {
        close(fd);
        throw std::runtime_error("Failed to set stream parameters");
    }

    
    // set directly using register
    // set_tof_reg(subfd, 0x21a0, 0x00, 1);
    // set_tof_reg(subfd, 0x21a1, 0x00, 1);
    set_tof_reg(fd, 0x21a0, 0x22, 0x01);
    set_tof_reg(fd, 0x21a1, 0x22, 0x01);
    
          
    ///////
    // VIDEO CAPTURE
    //////

    cv::VideoCapture cap(VIDEO_DEVICE, cv::CAP_V4L2);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open the V4L2 stream!" << std::endl;
        close(fd);
        return 1;
    }

    cap.set(cv::CAP_PROP_FPS, 5); //set frame rate via OpenCV
    cap.set(cv::CAP_PROP_CONVERT_RGB, 0);
    
    cv::Mat rawframe0;
    cv::Mat rawframe180;
    cv::Mat rawframe90;
    cv::Mat rawframe270;
    
    cv::Mat int_frame0;
    cv::Mat int_frame180;
    cv::Mat int_frame90;
    cv::Mat int_frame270;
    
    cv::Mat double_frame0;
    cv::Mat double_frame180;
    cv::Mat double_frame90;
    cv::Mat double_frame270;
    
    cv::Mat I;
    cv::Mat Q;
    cv::Mat Isqd;
    cv::Mat Qsqd;
    cv::Mat IplusQsqd;
    
    cv::Mat magnitudeImage;
    cv::Mat phaseImage;
    cv::Mat sRGBImage;
    cv::Mat sRGBPhaseImage;    
    cv::Mat sRGBPhaseImage_CM;

    
    //cv::Mat grayChannel;
    //printf("grayChannel datatype %d \n",grayChannel.type());
    //cv::extractChannel(frame, grayChannel,1);
    

    
    cv::namedWindow("3DRGB Magnitude Image", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("3DRGB Phase Image", cv::WINDOW_AUTOSIZE);
    
  //  cv::Size size = sRGBImage.size(); // size.width = 640, size.height = 480

    
    printf("size of sRGBImage arrays %d %d %d\n", sRGBImage.rows,sRGBImage.cols, sRGBImage.channels());
    int ix = 0;
    auto start = std::chrono::high_resolution_clock::now(); 
    auto end0 = std::chrono::high_resolution_clock::now(); 
    auto end90 = std::chrono::high_resolution_clock::now(); 
    auto end180 = std::chrono::high_resolution_clock::now(); 
    auto end270 = std::chrono::high_resolution_clock::now(); 
    auto duration0 = std::chrono::duration_cast<std::chrono::milliseconds>(end0 - start);
    auto duration90 = std::chrono::duration_cast<std::chrono::milliseconds>(end90 - start);
    auto duration180 = std::chrono::duration_cast<std::chrono::milliseconds>(end180 - start);
    auto duration270 = std::chrono::duration_cast<std::chrono::milliseconds>(end270 - start);

    while (1) {

        start = std::chrono::high_resolution_clock::now(); 
        while(1) {
            cap.read(rawframe0);
            end0 = std::chrono::high_resolution_clock::now();
            duration0 = std::chrono::duration_cast<std::chrono::milliseconds>(end0 - start);
            if (duration0.count() > 25) break;
        }

        start = std::chrono::high_resolution_clock::now(); 
        cap.read(rawframe180);
        end180 = std::chrono::high_resolution_clock::now();
        duration180 = std::chrono::duration_cast<std::chrono::milliseconds>(end180 - start);

        start = std::chrono::high_resolution_clock::now(); 
        cap.read(rawframe90);
        end90 = std::chrono::high_resolution_clock::now();
        duration90 = std::chrono::duration_cast<std::chrono::milliseconds>(end90 - start);

        start = std::chrono::high_resolution_clock::now(); 
        cap.read(rawframe270);
        end270 = std::chrono::high_resolution_clock::now();
        duration270 = std::chrono::duration_cast<std::chrono::milliseconds>(end270 - start);

        //printf("rawframe0 \n");
        //printfMat(rawframe0);

        int_frame0 = Mat_shiftPixelValues_1chan(rawframe0,4);
        int_frame180 = Mat_shiftPixelValues_1chan(rawframe180,4);
        int_frame90 = Mat_shiftPixelValues_1chan(rawframe90,4);
        int_frame270 = Mat_shiftPixelValues_1chan(rawframe270,4);
        
        int_frame0.convertTo(double_frame0, CV_64F);
        int_frame180.convertTo(double_frame180, CV_64F);
        int_frame90.convertTo(double_frame90, CV_64F);
        int_frame270.convertTo(double_frame270, CV_64F);
        
        //printf("intframe0 \n");
        //printfMat(int_frame0);
        
        //printf("Phases Acquired? \n");       
        cv::subtract(double_frame0,double_frame180,I);
        cv::subtract(double_frame90,double_frame270,Q);
        //printf("I \n");
        //printfMatf(I);       
        
        //printf("IQ ready \n");
        cv::multiply(I, I,Isqd);
        cv::multiply(Q, Q,Qsqd);
        
        //printf("Isqd \n");
        //printfMatf(Isqd);               
        
        //printf("Squared ready \n");
        cv::add(Isqd,Qsqd,IplusQsqd);
        //printf("IplusQsqd \n");
        //printfMatf(IplusQsqd);                       
        
        //printf("Add Squares ready \n");
        cv::sqrt(IplusQsqd,magnitudeImage);
        //printf("magnitude \n");
        //printfMatf(magnitude);      
        //printf("Mag ready \n");
               
        // Convert Y12P to sRGB        
        sRGBImage = double_mat_to_rgb(magnitudeImage);
        
       
        cv::phase(Q, I, phaseImage, true);
        sRGBPhaseImage = double_mat_to_rgb_phase(phaseImage);
        // printf("\nph0 \n");
        // printfMatf(double_frame0);
        // printf("ph180 \n");
        // printfMatf(double_frame180);
        // printf("ph90 \n");
        // printfMatf(double_frame90);
        // printf("ph270 \n");
        // printfMatf(double_frame270);                
        /*
        printf("Phase \n");
        printfMatf(phaseImage);
        printf("I \n");
        printfMatf(I);
        printf("Q \n");
        printfMatf(Q);*/
       
        
        ////if (sRGBImage.empty()) {
        ////    std::cerr << "Error: Captured empty frame!" << std::endl;
        ////    break;
        ////}
        
        cv::Mat colormapped_image;
        //cv::applyColorMap(sRGBPhaseImage, sRGBPhaseImage_CM, cv::COLORMAP_JET); // Choose your colormap
              
        std::cout<<"Frame Duration: " << duration0.count() << "ms  " << duration180.count() << "ms  " << duration90.count() << "ms  " << duration270.count() << "ms\n";
        cv::imshow("3DRGB Magnitude Image", sRGBImage);        
        cv::imshow("3DRGB Phase Image", sRGBPhaseImage);        
        //cv::imshow("3DRGB Phase Image", sRGBPhaseImage_CM);  
              
        if (cv::waitKey(1) >= 0) break;
        ix++;
    }
    

    close(fd);
    close(subfd);        
    cap.release();

    return 0;
}

