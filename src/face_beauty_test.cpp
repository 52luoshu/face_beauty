/************************************************************************************
* File: face_beauty_test.cpp
* Brief: ���ۡ���������demo
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

	//����ʵ�����������в�����ʼ������������Ϊͼ����ߡ����۳̶ȡ������̶�
	FaceBeauty faceBeauty;
	faceBeauty.Init(width, height, 5, 5);
	//��ʼ��ָ�룬���ֱ�ָ������ͼ���������
	uint8_t *pFrame = (uint8_t *)malloc(width*height * 3 / 2);
	uint8_t *pRet = (uint8_t *)malloc(width*height * 3 / 2);

	/// ��������ͷ
	cv::VideoCapture cap(0);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, width);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, height);
	cv::Mat frame, ret;
	while (cap.isOpened()) {
		///���뵱ǰ����ͷ֡����ʾ�����������RGB����ת��ΪYUV_I420��ʽ����
		cap >> frame;
		if (frame.empty()) {
			std::cout << "can't get frame." << std::endl;
			break;
		}
		//cv::resize(frame, frame, cv::Size(width, height));
		cv::imshow("input", frame);
		cv::cvtColor(frame, frame, cv::COLOR_BGR2YUV_I420);

		//pFrameָ��ǰYUV_I420��ʽͼ��ִ�����ղ�������������Ϊ��
		//ָ������ͼ���ָ�롢ָ�����ͼ���ָ�롢ͼ���ʽ
#ifdef BEAUTY
		pFrame = frame.ptr<uint8_t>(0);
		faceBeauty.Process(pFrame, pRet, pixfmt);
#endif // BEAUTY
		///������ݿ��ӻ�
		cv::Mat result(height * 3 / 2, width, CV_8UC1, pRet);
		cv::imshow("output_yuv_i420", result);
		cv::cvtColor(result, ret, cv::COLOR_YUV2BGR_I420);
		cv::imshow("output_rgb", ret);
		///��֡������ɺ�ȴ�ʱ��66ms��ESC���˳�
		if (cv::waitKey(66) == 27) {
			cv::destroyAllWindows();
			break;
		}
	}
}
