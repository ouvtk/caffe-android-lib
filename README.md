Caffe-Android-Lib 
===============
(forked from sh1r0 https://github.com/sh1r0/caffe-android-lib)
## Goal
Porting [caffe](https://github.com/BVLC/caffe) to android platform

## Build
has few problems on Windows
```
git clone --recursive https://github.com/ouvtk/caffe-android-lib.git
cd caffe-android-lib
./build.py $(NDK_PATH)
```

## Usage
- in case you want to use it for your package change JNI functions names 
  (ex. change package from com.sh1r0.caffe-android-demo to com.example.caffe)
	```
	./caffe-mobile/jni/caffe_jni.cpp 
	
	Replace middle part from
	Java_com_sh1r0_caffe_1android_1demo_CaffeMobile_loadModel to
	Java_com_example_caffe_CaffeMobile_loadModel
	```

- put required stuff into your device

	```
	./get_model.py
	adb shell mkdir -p /sdcard/caffe_mobile/
	adb push caffe-mobile/jni/caffe/models/bvlc_reference_caffenet/ /sdcard/caffe_mobile/bvlc_reference_caffenet/
	```
- copy `caffe-mobile/libs/armeabi-v7a/*.so` to your jni lib directory (ex. for recent gradle build is src/main/jniLibs/armeabi-v7a)
- in your main activity

	```java
	static {
		System.loadLibrary("caffe");
		System.loadLibrary("caffe_jni");
	}
	```
- create `CaffeMobile.java`

	```java
	package ru.ouvtk.deepdreamapp;

	public class CaffeMobile {
		public native void enableLog(boolean enabled);
		public native int loadModel(String modelPath, String weightsPath);
		public native int predictImage(String imgPath);
		public native void dreamImage(String modelPath, Bitmap outBmp);
	}
	```
- call native methods

	```java
	CaffeMobile caffeMobile = new CaffeMobile();
	caffeMobile.enableLog(true);  // optional, enable native logging
	caffeMobile.loadModel(modelPath, weightsPath);  // init once
	...
	caffeMobile.predictImage(imgPath);
	//or
	caffeMobile.dreamImage(imgPath, bmp);
	```

## Optional
`.envrc` files are for [direnv](http://direnv.net/)
> direnv is an environment variable manager for your shell. It knows how to hook into bash, zsh and fish shell to load or unload environment variables depending on your current directory. This allows to have project-specific environment variables and not clutter the "~/.profile" file.

## Dependency
* [Boost-for-Android](https://github.com/MysticTreeGames/Boost-for-Android)
* [protobuf](https://code.google.com/p/protobuf)
* [Eigen](http://eigen.tuxfamily.org)

## Credits
* [caffe-compact](https://github.com/chyh1990/caffe-compact)
