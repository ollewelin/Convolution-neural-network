# Convolution-neural-network C++
I am only a novice at C/C++ programming. 
This is my own test of Convolution neural network use OpenCV. With 2 or 3 Layer feature and Autoencoder
I done this just for fun for my own to try understand and test the principle of Convolution Neural Network 
with supervised learning and unsupervied learning. I was use Denoising Autoencoder here used for feature unsupervised learning. 
A good explanation by this guy: Hugo Larochelle https://www.youtube.com/watch?v=FzS3tMl4Nsc thanks for that! 
My youtube view of this: https://www.youtube.com/watch?v=Hckvt2WV3Rk

I get about 95.5% accuracy of MNIST test after loog training time, not so impressive but it work at least.

There are 3 different mode you can run this code

#define MNIST

#define IMAGE //2 categories. I was test with 48x48 images in this case but you will set image size and nr of images at start

#define CIFAR 

Dependencies
1. install cmake for use the of the CMakeLists.txt
$sudo apt-get install cmake

2. OpenCV
I was use this way to install OpenCV
http://milq.github.io/install-opencv-ubuntu-debian/

3. MNIST data set download to program root folder. http://yann.lecun.com/exdb/mnist/

4. replace the "." with a "-" in all the 4 file names from MNIST "t10k-images.idx3-ubyte" to "t10k-images-idx3-ubyte" and so on
my program ask for t10k-images-idx3-ubyte file name
