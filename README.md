# Convolution-neural-network C++
I am only a novice at C/C++ programming (at the time 2017). 
This is my own test of Convolution neural network use OpenCV. With 2 or 3 Layer feature and Autoencoder. Deconvolution L3 don't work out probably bug i Layer 3. 
I done this just for fun for my own to try understand and test the principle of Convolution Neural Network 
with supervised learning and unsupervied learning. I was use Denoising Autoencoder here used for feature unsupervised learning. 
A good explanation by this guy: Hugo Larochelle https://www.youtube.com/watch?v=FzS3tMl4Nsc thanks for that! 
My youtube view of this: https://www.youtube.com/watch?v=Hckvt2WV3Rk

I get about 95.5% accuracy of MNIST test after loog training time, not so impressive but it work at least.
One thing how missing in the rush of this test program is the Soft Max layer at the end of fully connected layer. (Soft Max should Always be used if there is more then 1 output neuron, in this case with out the softmax the training will reach a platua becuse the total some of the 10 digits will not reach a probability of 1.0)

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

Then Install OpenCV
download:

https://github.com/opencv/opencv

unzip opencv-master.zip

change directory to unziped folder

`$ mkdir build`

`$ cd build`

Update for OpenCV 4.3.0

`/build$ sudo cmake -D OPENCV_GENERATE_PKGCONFIG=ON ..` 

`/build$ sudo make`

`/build$ sudo make install`


2. MNIST data set download to program root folder. http://yann.lecun.com/exdb/mnist/

3. replace the "." with a "-" in all the 4 file names from MNIST "t10k-images.idx3-ubyte" to "t10k-images-idx3-ubyte" and so on
my program ask for t10k-images-idx3-ubyte file name

Compile by use make and Makefile

`$ make`

run

`$ ./exe_CNN`
