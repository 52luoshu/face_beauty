/************************************************************************************
* File: face_beauty_test.cpp
* Brief: 大眼、瘦脸测试demo
* Date: 2020.09.22
************************************************************************************/
#include "face_beauty.h"
#define BEAUTY

void demo_YUV_I420();

int main() {
	demo_YUV_I420();
	return 0;
}


void demo_YUV_I420() {
	int width = 640;
	int height = 480;
	PixelFmt pixfmt = PixelFmt_I420;

	//对象实例化，并进行参数初始化，参数依次为图像宽、高、大眼程度、瘦脸程度
	FaceBeauty faceBeauty;
	faceBeauty.Init(width, height, 5, 5);
	//初始化指针，将分别指向输入图像和输出结果
	uint8_t *pFrame = (uint8_t *)malloc(width*height * 3 / 2);
	uint8_t *pRet = (uint8_t *)malloc(width*height * 3 / 2);

	/// 调用摄像头
	cv::VideoCapture cap(0);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, width);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, height);
	cv::Mat frame, ret;
	while (cap.isOpened()) {
		///读入当前摄像头帧并显示，并将读入的RGB数据转化为YUV_I420格式备用
		cap >> frame;
		if (frame.empty()) {
			std::cout << "can't get frame." << std::endl;
			break;
		}
		//cv::resize(frame, frame, cv::Size(width, height));
		cv::imshow("input", frame);
		cv::cvtColor(frame, frame, cv::COLOR_BGR2YUV_I420);

		//pFrame指向当前YUV_I420格式图像，执行美颜操作，参数依次为：
		//指向输入图像的指针、指向输出图像的指针、图像格式
#ifdef BEAUTY
		pFrame = frame.ptr<uint8_t>(0);
		faceBeauty.Process(pFrame, pRet, pixfmt);
#endif // BEAUTY
		///结果数据可视化
		cv::Mat result(height * 3 / 2, width, CV_8UC1, pRet);
		cv::imshow("output_yuv_i420", result);
		cv::cvtColor(result, ret, cv::COLOR_YUV2BGR_I420);
		cv::imshow("output_rgb", ret);
		///单帧处理完成后等待时长66ms，ESC键退出
		if (cv::waitKey(66) == 27) {
			cv::destroyAllWindows();
			break;
		}
	}
}
