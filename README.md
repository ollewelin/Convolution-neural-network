# Convolution-neural-network C++
Warning Skunky Obsolete code.

## Please jump to newer better repo here: 
## https://github.com/ollewelin/Nerual_Netwok_CPP

Bugg: Finetune feature kernels backpropagation is wrong, need 180 deg turn around feature when backprop the features in fine tune mode. The finetune feature need to change implementation.
There is also bugs if you change feature size from 7 to 5 seg fault will occure some pointer going wild out in neverland probably. I will not take the time to fix this or investigate in this. Don't use this code. Only some insperations.
I was only a novice at C++ programming at the time 2017. 
This is my own test of Convolution neural network use OpenCV. With 2 or 3 Layer feature and Autoencoder. Deconvolution L3 don't work out probably bug i Layer 3. 
I done this just for fun for my own to try understand and test the principle of Convolution Neural Network 
with supervised learning and unsupervied learning. I was use Denoising Autoencoder here used for feature unsupervised learning. 
A good explanation by this guy: Hugo Larochelle https://www.youtube.com/watch?v=FzS3tMl4Nsc thanks for that! 
My youtube view of this: https://www.youtube.com/watch?v=Hckvt2WV3Rk

I get about 95.5% accuracy of MNIST test after loog training time, not so impressive but it work at least.

Update test now with SOFTMAX

There are 3 different mode you can run this code

`#define MNIST`

`#define IMAGE //2 categories. I was test with 48x48 images in this case but you will set image size and nr of images at start`

`#define CIFAR` 

Dependencies:

1. Install OpenCV

To install OpenCV
first install 
cmake

`$ sudo apt-get update`

`$ wget https://github.com/Kitware/CMake/releases/download/v3.15.2/cmake-3.15.2.tar.gz`

`$ tar -zxvf cmake-3.15.2.tar.gz`

`$ cd cmake-3.15.2`

`$ ./bootstrap`

`$ make`

`$ sudo make install`

`$ cmake --version`

Then Install OpenCV4
download:

https://github.com/opencv/opencv

unzip opencv-master.zip

change directory to unziped folder

`$ mkdir build`

`$ cd build`

`/build$ sudo cmake -D OPENCV_GENERATE_PKGCONFIG=ON ..` 

`/build$ sudo make`

`/build$ sudo make install`

Uninstall Note: If you have an old OpenCV installed with cmake and a /build folder then:
Uninstall old opencv (in example opencv 3.1)

`$ cd build`

`/build$ sudo make uninstall`

Then delete old opencv-master folder and install with OpenCV4 as above.

2. Auto download MNIST option new 2021-mars update ( MNIST data set download to program root folder. http://yann.lecun.com/exdb/mnist/ )

![](Auto_download_MNIST_dataset.png)

(3.) OLD (replace the "." with a "-" in all the 4 file names from MNIST "t10k-images.idx3-ubyte" to "t10k-images-idx3-ubyte" and so on
my program ask for t10k-images-idx3-ubyte file name)

Compile by use make and Makefile

`$ make`

run

`$ ./exe_CNN`

Deconvoluted L2 layer feature.
![](cloned_sq_visualize_L2toL1%20(copy).JPG)

Start fine tune L2 conv layer
![](Screenshot_3_I_command_finetune_L2.png)

