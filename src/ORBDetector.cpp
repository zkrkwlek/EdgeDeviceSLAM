#include "./EdgeDeviceSLAM/include/ORBDetector.h"

namespace EdgeDeviceSLAM {
	ORBDetector::ORBDetector(int nFeatures, float fScaleFactor, int nLevels, float fInitThFast, float fMinThFast){
		detector = new ORBextractor(nFeatures, fScaleFactor, nLevels, fInitThFast, fMinThFast);
		init_sigma_level();
	}
	ORBDetector::~ORBDetector() {}
	void ORBDetector::detectAndCompute(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors) {
		(*detector)(image, mask, keypoints, descriptors);
	}
	void ORBDetector::Compute(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors) {
		(*detector).Compute(image, mask, keypoints, descriptors);
	}
	void ORBDetector::init_sigma_level() {
		mnScaleLevels = detector->GetLevels();
		mfScaleFactor = detector->GetScaleFactor();
		mfLogScaleFactor = log(mfScaleFactor);
		mvScaleFactors = detector->GetScaleFactors();
		mvInvScaleFactors = detector->GetInverseScaleFactors();
		mvLevelSigma2 = detector->GetScaleSigmaSquares();
		mvInvLevelSigma2 = detector->GetInverseScaleSigmaSquares();
	}
	float ORBDetector::CalculateDescDistance(cv::Mat a, cv::Mat b) {
		const int *pa = a.ptr<int32_t>();
		const int *pb = b.ptr<int32_t>();

		int dist = 0;

		for (int i = 0; i<8; i++, pa++, pb++)
		{
			unsigned  int v = *pa ^ *pb;
			v = v - ((v >> 1) & 0x55555555);
			v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
			dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
		}
		return (float)dist;
	}

}