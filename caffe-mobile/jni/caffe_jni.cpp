#include <string.h>
#include <jni.h>
#include <android/log.h>
#include <android/bitmap.h>
#include <string>

#include "caffe/caffe.hpp"
#include "caffe_mobile.hpp"

#define  LOG_TAG    "MiRA-CNN"
#define  LOGV(...)  __android_log_print(ANDROID_LOG_VERBOSE,LOG_TAG, __VA_ARGS__)
#define  LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG,LOG_TAG, __VA_ARGS__)
#define  LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG, __VA_ARGS__)
#define  LOGW(...)  __android_log_print(ANDROID_LOG_WARN,LOG_TAG, __VA_ARGS__)
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,LOG_TAG, __VA_ARGS__)

#ifdef __cplusplus
extern "C" {
#endif

using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using caffe::shared_ptr;
using caffe::vector;
using caffe::MemoryDataLayer;

caffe::CaffeMobile *caffe_mobile;

int getTimeSec();

static int pfd[2];
static pthread_t thr;
static const char *tag = "stderr";

static void *thread_func(void*) {
    ssize_t rdsz;
    char buf[1024];
    while ((rdsz = read(pfd[0], buf, sizeof(buf) - 1)) > 0) {
        buf[rdsz] = 0;  // add null-terminator
        __android_log_write(ANDROID_LOG_DEBUG, tag, buf);
    }
    return 0;
}

static int start_logger() {
    /* make stdout line-buffered and stderr unbuffered */
    // setvbuf(stdout, 0, _IOLBF, 0);
    setvbuf(stderr, 0, _IONBF, 0);

    /* create the pipe and redirect stdout and stderr */
    pipe(pfd);
    // dup2(pfd[1], 1);
    dup2(pfd[1], 2);

    /* spawn the logging thread */
    if (pthread_create(&thr, 0, thread_func, 0) == -1)
        return -1;
    pthread_detach(thr);
    return 0;
}

void JNIEXPORT JNICALL
Java_ru_ouvtk_deepdreamapp_CaffeMobile_enableLog(JNIEnv* env, jobject thiz, jboolean enabled)
{
    start_logger();
    caffe::LogMessage::Enable(enabled != JNI_FALSE);
}

jint JNIEXPORT JNICALL
Java_ru_ouvtk_deepdreamapp_CaffeMobile_loadModel(JNIEnv* env, jobject thiz, jstring modelPath, jstring weightsPath)
{
    const char *model_path = env->GetStringUTFChars(modelPath, 0);
    const char *weights_path = env->GetStringUTFChars(weightsPath, 0);
    caffe_mobile = new caffe::CaffeMobile(string(model_path), string(weights_path));
    env->ReleaseStringUTFChars(modelPath, model_path);
    env->ReleaseStringUTFChars(weightsPath, weights_path);
    return 0;
}

jint JNIEXPORT JNICALL
Java_ru_ouvtk_deepdreamapp_CaffeMobile_predictImage(JNIEnv* env, jobject thiz, jstring imgPath)
{
    const char *img_path = env->GetStringUTFChars(imgPath, 0);
    caffe::vector<int> top_k = caffe_mobile->predict_top_k(string(img_path), 3);
    LOGD("top-1 result: %d", top_k[0]);

    env->ReleaseStringUTFChars(imgPath, img_path);

    return top_k[0];
}

/**
 * Code is partially from https://github.com/ruckus/android-image-filter-ndk
 */

void JNIEXPORT JNICALL
Java_ru_ouvtk_deepdreamapp_CaffeMobile_dreamImage(JNIEnv* env, jobject thiz, jstring imgPath, jobject outBitmap)
{
	AndroidBitmapInfo info;
    int ret;
    void* pixels;

    const char *img_path = env->GetStringUTFChars(imgPath, 0);
	
    const vector<Blob<float>*>& resImage = caffe_mobile->deepDream(string(img_path));
	
	if ((ret = AndroidBitmap_getInfo(env, outBitmap, &info)) < 0) {
            LOGE("AndroidBitmap_getInfo() failed ! error=%d", ret);
            return;
        }
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        LOGE("Bitmap format is not RGBA_8888 !");
        return;
    }

    if ((ret = AndroidBitmap_lockPixels(env, outBitmap, &pixels)) < 0) {
        LOGE("AndroidBitmap_lockPixels() failed ! error=%d", ret);
    }
	
    LOG(DEBUG) << "putImage will start soon ";

	caffe_mobile->putImage(&info, pixels, resImage);

	LOG(DEBUG) << "putImage ended ";

    AndroidBitmap_unlockPixels(env, outBitmap);

    env->ReleaseStringUTFChars(imgPath, img_path);

    return;
}

int getTimeSec() {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (int) now.tv_sec;
}
/*
JavaVM *g_jvm = NULL;
jobject g_obj = NULL;

void JNIEXPORT JNICALL
Java_com_sh1r0_caffe_1android_1demo_MainActivity_MainActivity_setJNIEnv(JNIEnv* env, jobject obj)
{
    env->GetJavaVM(&g_jvm);
    g_obj = env->NewGlobalRef(obj);
}
*/
jint JNIEXPORT JNICALL JNI_OnLoad(JavaVM *vm, void *reserved)
{
    JNIEnv* env = NULL;
    jint result = -1;

    if (vm->GetEnv((void**)&env, JNI_VERSION_1_6) != JNI_OK) {
        LOGE("GetEnv failed!");
        return result;
    }

    return JNI_VERSION_1_6;
}

int main(int argc, char const *argv[])
{
    string usage("usage: main <model> <weights> <img>");
    if (argc < 4) {
        std::cerr << usage << std::endl;
        return 1;
    }

    caffe::LogMessage::Enable(true); // enable logging
    caffe_mobile = new caffe::CaffeMobile(string(argv[1]), string(argv[2]));
    caffe::vector<int> top_3 = caffe_mobile->predict_top_k(string(argv[3]));
    for (auto k : top_3) {
        std::cout << k << std::endl;
    }
    return 0;
}

#ifdef __cplusplus
}
#endif
