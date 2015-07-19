#include <string>
#include "caffe_mobile.hpp"

using std::string;
using std::static_pointer_cast;
using std::clock;
using std::clock_t;

using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using caffe::shared_ptr;
using caffe::vector;
using caffe::MemoryDataLayer;

namespace caffe {

template <typename T>
vector<size_t> ordered(vector<T> const& values) {
	vector<size_t> indices(values.size());
	std::iota(begin(indices), end(indices), static_cast<size_t>(0));

	std::sort(
		begin(indices), end(indices),
		[&](size_t a, size_t b) { return values[a] > values[b]; }
	);
	return indices;
}

CaffeMobile::CaffeMobile(string model_path, string weights_path) {
	CHECK_GT(model_path.size(), 0) << "Need a model definition to score.";
	CHECK_GT(weights_path.size(), 0) << "Need model weights to score.";

	Caffe::set_mode(Caffe::CPU);

	clock_t t_start = clock();
	caffe_net = new Net<float>(model_path, caffe::TEST);
	caffe_net->CopyTrainedLayersFrom(weights_path);
	clock_t t_end = clock();
	LOG(DEBUG) << "Loading time: " << 1000.0 * (t_end - t_start) / CLOCKS_PER_SEC << " ms.";
}

CaffeMobile::~CaffeMobile() {
	free(caffe_net);
	caffe_net = NULL;
}

int CaffeMobile::test(string img_path) {
	CHECK(caffe_net != NULL);

	Datum datum;
	CHECK(ReadImageToDatum(img_path, 0, 256, 256, true, &datum));
	const shared_ptr<MemoryDataLayer<float>> memory_data_layer =
		static_pointer_cast<MemoryDataLayer<float>>(
			caffe_net->layer_by_name("data"));
	memory_data_layer->AddDatumVector(vector<Datum>({datum}));

	vector<Blob<float>* > dummy_bottom_vec;
	float loss;
	clock_t t_start = clock();
	const vector<Blob<float>*>& result = caffe_net->Forward(dummy_bottom_vec, &loss);
	clock_t t_end = clock();
	LOG(DEBUG) << "Prediction time: " << 1000.0 * (t_end - t_start) / CLOCKS_PER_SEC << " ms.";

	const float* argmaxs = result[1]->cpu_data();
	for (int i = 0; i < result[1]->num(); i++) {
		for (int j = 0; j < result[1]->height(); j++) {
			LOG(INFO) << " Image: "<< i << " class:"
			          << argmaxs[i*result[1]->height() + j];
		}
	}

	return argmaxs[0];
}

vector<int> CaffeMobile::predict_top_k(string img_path, int k) {
	CHECK(caffe_net != NULL);

	Datum datum;
	CHECK(ReadImageToDatum(img_path, 0, 256, 256, true, &datum));
	const shared_ptr<MemoryDataLayer<float>> memory_data_layer =
		static_pointer_cast<MemoryDataLayer<float>>(
			caffe_net->layer_by_name("data"));
	memory_data_layer->AddDatumVector(vector<Datum>({datum}));

	float loss;
	vector<Blob<float>* > dummy_bottom_vec;
	clock_t t_start = clock();
	const vector<Blob<float>*>& result = caffe_net->Forward(dummy_bottom_vec, &loss);
	clock_t t_end = clock();
	LOG(DEBUG) << "Prediction time: " << 1000.0 * (t_end - t_start) / CLOCKS_PER_SEC << " ms.";

	const vector<float> probs = vector<float>(result[1]->cpu_data(), result[1]->cpu_data() + result[1]->count());
	CHECK_LE(k, probs.size());
	vector<size_t> sorted_index = ordered(probs);

	return vector<int>(sorted_index.begin(), sorted_index.begin() + k);
}

const vector<Blob<float>*> CaffeMobile::deepDream(string img_path) {
	CHECK(caffe_net != NULL);

	Datum datum;
	CHECK(ReadImageToDatum(img_path, 0, 256, 256, true, &datum));
	const shared_ptr<MemoryDataLayer<float>> memory_data_layer =
		static_pointer_cast<MemoryDataLayer<float>>(
			caffe_net->layer_by_name("data"));
	memory_data_layer->AddDatumVector(vector<Datum>({datum}));
	
	float loss;
	vector<Blob<float>* > dummy_bottom_vec;
	clock_t t_start = clock();
	const vector<Blob<float>*>& forwardResult = caffe_net->Forward(dummy_bottom_vec, &loss);
	clock_t t_end = clock();
	LOG(DEBUG) << "Forward time: " << 1000.0 * (t_end - t_start) / CLOCKS_PER_SEC << " ms.";

	t_start = clock();
	caffe_net->Backward();
	t_end = clock();
	LOG(DEBUG) << "Backward time: " << 1000.0 * (t_end - t_start) / CLOCKS_PER_SEC << " ms.";

	vector<Blob<float>*> result;

	Blob<float>* dataResult = caffe_net->blob_by_name("data").get();

	result.push_back(dataResult);

	return result;
}	

float clip(float n, float lower, float upper) {
  return std::max(lower, std::min(n, upper));
}

void CaffeMobile::putImage(AndroidBitmapInfo* info, void* pixels, const vector<Blob<float>*>& resImage) {
	Blob<float> * srcBlob = *resImage.data();

	LOG(DEBUG) << "srcBlob received";

	vector<int> shape = {1, 3, (int) info->width, (int) info->height };

	LOG(DEBUG) << "shape configured";

	Blob<float>* imgBlob = new Blob<float>();
	LOG(DEBUG) << "Blob created";

	imgBlob->Reshape(shape);
	LOG(DEBUG) << "imgBlob reshaped";

	imgBlob->CopyFrom(*srcBlob, false, true);
	LOG(DEBUG) << "imgBlob copied";

	int size = imgBlob->count();
	LOG(DEBUG) << "imgBlob size is: " << size;

	/*Partially from https://github.com/ruckus/android-image-filter-ndk*/

	uint32_t* pixelRow;
	int ix, iy, red, green, blue;

	for(iy = 0; iy < (int) info->height; iy++){

		pixelRow = (uint32_t*) pixels;

		for(ix =0; ix < (int) info->width; ix++){
			red = (int) clip(imgBlob->data_at(0,0,iy,ix), 0, 255);
			green = (int) clip(imgBlob->data_at(0,1,iy,ix), 0, 255);
			blue = (int) clip(imgBlob->data_at(0,2,iy,ix), 0, 255);

			pixelRow[ix] =
					((red << 16) & 0x00FF0000) |
					((green << 8) & 0x0000FF00) |
					(blue & 0x000000FF);
		}

		pixels = (char*)pixels + info->stride;
	}

	LOG(DEBUG) << "before return putImage " << size;

	return;
}

} // namespace caffe
