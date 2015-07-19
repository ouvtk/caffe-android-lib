#ifndef CAFFE_MOBILE_HPP_
#define CAFFE_MOBILE_HPP_

#include <string>
#include <vector>
#include "caffe/caffe.hpp"

#include <android/bitmap.h>

using std::string;

namespace caffe {

class CaffeMobile
{
public:
	CaffeMobile(string model_path, string weights_path);
	~CaffeMobile();

	int test(string img_path);

	vector<int> predict_top_k(string img_path, int k=3);

	const vector<Blob<float>*> deepDream(string img_path);

	void putImage(AndroidBitmapInfo* info, void* pixels, const vector<Blob<float>*>& resImage);


private:
	Net<float> *caffe_net;
};

} // namespace caffe

#endif
