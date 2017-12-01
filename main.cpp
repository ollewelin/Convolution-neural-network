///2017-11-30 Add fine tune L2 through backpropagation from fc fully connected network. Yet only test with L2_pool_cobe connected to fc, not tested unpooled connectiotion to fc
///Now also Tested on PC with ubuntu
///Test here with MNIST dataset
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
#include <opencv2/imgproc/imgproc.hpp> // Gaussian Blur
#include <stdio.h>
//#include <raspicam/raspicam_cv.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <cstdlib>
#include <ctime>
#include <math.h>  // exp
#include <stdlib.h>// exit(0);
#include <iostream>
using namespace std;
using namespace cv;
#include "c_func.h"
#include "cpp_func.hpp"
#include "cpp_func2.hpp"
#include <time.h> ///usleep()
///Add standard deviation of the all data after each Relu and use that value as the noise amplitude to each Autoencoder
#define USE_STD_DEVIATION_FOR_NOISE_AMPLITUDE
#ifdef USE_STD_DEVIATION_FOR_NOISE_AMPLITUDE
int do_std_dev_calc_counter=0;
const int run_std_dev_at_counter_level = 20;///Only do the calculation efter this number of training turns
///#define PRINT_STD_DEV ///Print standard deviation
#endif // USE_STD_DEVIATION_FOR_NOISE_AMPLITUDE
///Fix remove pool and move Relu operation if connect_fc_to_pool_or_conv == 1 on that layer
///******** Select one of this type USE_IMAGE, USE_MNIST or USE_CIFAR of training set ********************
///#define USE_IMAGE ///Own data set of image Pick own image training set 2 category
#define USE_MNIST /// Test also with MNIST data set
///#define USE_CIFAR
///******** END Select type of training set **************************************************************
///******** Feature size and depth settings ***********************
#ifdef USE_IMAGE
string IMAGE_cat_pos = "Positive";
string IMAGE_cat_neg = "Negative";
int IMAGE_height = 0;///Set by user
int IMAGE_width  = 0;///Set by user
int FL1_srt_size = 7;///IMAGE 7, CIFAR 5, MNIST 7 Must be a odd nubmber 1.3.5... Feature1 side size 5 = 5x5 feature patch
int FL2_srt_size = 7;///IMAGE 7, CIFAR 5, MNIST 7 Must be a odd nubmber 1.3.5... Feature2 side size 5 = 5x5 feature
int FL3_srt_size = 7;///Must be a odd nubmber 1.3.5... Feature2 side size 5 = 5x5 feature
int L1_conv_depth = 150;///IMAGE 150, CIFAR 75, MNIST 50. L1_conv_depth is the depth of the L1 Convolution cube. This is also the number of Feature on L2.
int L2_conv_depth = 200;///IMAGE 100, CIFAR 300, MNIST 75 L2_conv_depth is the depth of the L2 Convolution cube. This is also the number of Feature on L3.
int L3_conv_depth = 200;///CIFAR 300, MNIST not used L3_conv_depth is the depth of the L3 Convolution cube. This is also the number of Feature on L3.
const int connect_fc_to_layer = 2;///2 or 3. L3 Convloution now also implemented. This selector connect_fc_to_layer will connect the fully connected neural network to L2 or L3 conv->pooled data
const int connect_fc_to_pool_or_conv = 0;///0= will connect (fc) fully connected network to (pool) pooling cube data. 1= will connect (fc) network to (conv) convolution cube data.
const int L1_stride   = 2;
const int L2_stride   = 1;
const int L3_stride   = 1;
const int pooling  = 4;///maxpooling from 4->1 node. This value is the pooling area
#endif // USE_IMAGE
#ifdef USE_MNIST
int FL1_srt_size = 7;///IMAGE 7, CIFAR 5, MNIST 7 Must be a odd nubmber 1.3.5... Feature1 side size 5 = 5x5 feature patch
int FL2_srt_size = 7;///IMAGE 7, CIFAR 5, MNIST 7 Must be a odd nubmber 1.3.5... Feature2 side size 5 = 5x5 feature
int FL3_srt_size = 3;///Must be a odd nubmber 1.3.5... Feature2 side size 5 = 5x5 feature
int L1_conv_depth = 50;///IMAGE 150, CIFAR 75, MNIST 50. L1_conv_depth is the depth of the L1 Convolution cube. This is also the number of Feature on L2.
int L2_conv_depth = 75;///IMAGE 100, CIFAR 300, MNIST 75 L2_conv_depth is the depth of the L2 Convolution cube. This is also the number of Feature on L3.
int L3_conv_depth = 200;///CIFAR 300, MNIST not used L3_conv_depth is the depth of the L3 Convolution cube. This is also the number of Feature on L3.
const int connect_fc_to_layer = 2;///2 or 3. L3 Convloution now also implemented. This selector connect_fc_to_layer will connect the fully connected neural network to L2 or L3 conv->pooled data
const int connect_fc_to_pool_or_conv = 0;///0= will connect (fc) fully connected network to (pool) pooling cube data. 1= will connect (fc) network to (conv) convolution cube data.
const int L1_stride   = 1;
const int L2_stride   = 1;
const int L3_stride   = 1;
const int pooling  = 4;///maxpooling from 4->1 node. This value is the pooling area
#endif // USE_MNIST
#ifdef USE_CIFAR
int FL1_srt_size = 5;///IMAGE 7, CIFAR 5, MNIST 7 Must be a odd nubmber 1.3.5... Feature1 side size 5 = 5x5 feature patch
int FL2_srt_size = 3;///IMAGE 7, CIFAR 5, MNIST 7 Must be a odd nubmber 1.3.5... Feature2 side size 5 = 5x5 feature
int FL3_srt_size = 3;///Must be a odd nubmber 1.3.5... Feature2 side size 5 = 5x5 feature
int L1_conv_depth = 150;///IMAGE 150, CIFAR 75, MNIST 50. L1_conv_depth is the depth of the L1 Convolution cube. This is also the number of Feature on L2.
int L2_conv_depth = 150;///IMAGE 100, CIFAR 300, MNIST 75 L2_conv_depth is the depth of the L2 Convolution cube. This is also the number of Feature on L3.
int L3_conv_depth = 200;///CIFAR 300, MNIST not used L3_conv_depth is the depth of the L3 Convolution cube. This is also the number of Feature on L3.
const int connect_fc_to_layer = 2;///2 or 3. L3 Convloution now also implemented. This selector connect_fc_to_layer will connect the fully connected neural network to L2 or L3 conv->pooled data
const int connect_fc_to_pool_or_conv = 0;///0= will connect (fc) fully connected network to (pool) pooling cube data. 1= will connect (fc) network to (conv) convolution cube data.
const int L1_stride   = 2;
const int L2_stride   = 2;
const int L3_stride   = 2;
const int pooling  = 4;///maxpooling from 4->1 node. This value is the pooling area
#endif // USE_CIFAR
///******** End Feature size and depth settings ***********************
///Test fist with connect Fully connected network to L2 pooling cube
///Add bias to all features
///Select MNIST training or verify set
int auto_save_ON=1;
int auto_save_counter=98;
const int ittr_before_L1_autosave=10;
int ittr_counter_L1_autosave=0;
const int auto_save_at=100;
int test_itterations=0;
const int RESET_test_ittr = 1000000;
int test_correct=0;
float correct_ratio=0.0f;
int enable_print_nodes=1;
///#define USE_BGR_NORMALIZER
#define USE_PATCH_NOISE
FILE *fp1;//Parameter file
float start_weight_noise_range = 0.15f;//+/- Weight startnoise range
float L1_autoencoder_noise_ratio =0.0f;/// = 25.0f;///25.0f
float L2_autoencoder_noise_ratio =0.0f;/// = 5.0f;///
float L3_autoencoder_noise_ratio =0.0f;/// = 5.0f;///
int nr_of_autoenc_ittr_1_image=20;///Nr of autoencoder training of a patch with ranomized location on eact input image
int L2_nr_of_autoenc_ittr_1_image=5;///
int L3_nr_of_autoenc_ittr_1_image=5;///
///float noise_amplitude = 0.5f;///1.0f
float L1_autoencoder_noise_aplitude = 0.5f;
float L2_autoencoder_noise_aplitude = 0.5f;
float L3_autoencoder_noise_aplitude = 0.5f;
float noise_offset = 0.0f;///-0.5..+0.5
float Relu_neg_gain = 0.01f;
///const float C_L1_LearningRate =0.02f;///0.002
///const float C_L1_Momentum = 0.1f;///0.0f
///float L1_LearningRate = C_L1_LearningRate;///
///float L1_Momentum = C_L1_Momentum;///

int show_patch_noise = 0;
///float run_autoencoder_ratio = 1.0f;
char filename[100];
///char filename2[100];
///char filename_dst[100];

///Now with fully connected Logistic Regression network Supervised Learning
///************* Parameters and things regarding fully connected network **************
///#define USE_MEAN_VALUE_TO_FC_NETWORK
int fully_conn_backprop =0;
const int C_fully_hidd_nodes = 300;
const int C_fully_out_nodes = 10;
const int nr_of_hot_target_nodes = 1;
int fully_hidd_nodes = C_fully_hidd_nodes;
int fully_out_nodes = C_fully_out_nodes;
int drop_out_percent = 30;/// 50% dropout percent hidden nodes during training
int verification = 0;
float Error_level=0.0f;
const float fc_start_weight_noise_range = 0.15f;//+/- Weight startnoise range

float fc_LearningRate = 0.01f;///0.025f
float fc_Momentum = 0.8;///0.96
float High_Target_value = 1.0f;
///float Low_Target_value = 0.45f;
float Low_Target_value = 0.5f - (((float)((float)nr_of_hot_target_nodes / (float)fully_out_nodes)) * 0.5f);/// ((float)nr_of_hot_target_nodes / (float)fully_out_nodes)) / 2.0f;///
int Learning_fc = 1;///ON / OFF regarding fine tune Features or Verfication test running
int Training_fc = 1;///...

const float Bias_level =1.0f;
///*************************************************************************************
///============================== Trackbar adjustments ===============================
int H_MIN = 0;
int H_MAX = 100;
int H_MAX_1000 = 1000;
const string trackbarWindowName = "Trackbars";
#ifdef USE_IMAGE
int L1_noise_int = 25;///25 = 25%
int L2_noise_int = 1;
int L3_noise_int = 1;
int tune_L1_int = 10;///20 = 0.002f
int tune_L2_int = 20;///2 = 0.0002f
int tune_L3_int = 20;///2 = 0.0002f
int tune_L1_moment_int =100;///1000 = 0.999f
int tune_L2_moment_int =10;///100 = 0.0999f
int tune_L3_moment_int =10;///100 = 0.0999f
#endif // USE_IMAGE
#ifdef USE_MNIST
int L1_noise_int = 25;///25 = 25%
int L2_noise_int = 2;
int L3_noise_int = 1;
int tune_L1_int = 20;///20 = 0.002f
int tune_L2_int = 20;///2 = 0.0002f
int tune_L3_int = 20;///2 = 0.0002f
int tune_L1_moment_int =100;///1000 = 0.999f
int tune_L2_moment_int =10;///100 = 0.0999f
int tune_L3_moment_int =10;///100 = 0.0999f
#endif // USE_MNIST
#ifdef USE_CIFAR
int L1_noise_int = 25;///25 = 25%
int L2_noise_int = 1;
int L3_noise_int = 1;
int tune_L1_int = 100;///20 = 0.002f
int tune_L2_int = 100;///2 = 0.0002f
int tune_L3_int = 80;///2 = 0.0002f
int tune_L1_moment_int =20;///1000 = 0.999f
int tune_L2_moment_int =10;///100 = 0.0999f
int tune_L3_moment_int =10;///100 = 0.0999f
#endif // USE_CIFAR

///int target_low_value_int = 5;///45 = 0.45f
float L1_LearningRate = 0.0f;///depend on the state of Lock_L1
float L1_Momentum = 0.0f;///depend on the state of Lock_L1
float L2_LearningRate = 0.0f;///
float L2_Momentum = 0.0f;///
float L3_LearningRate = 0.0f;///0.0002f
float L3_Momentum = 0.0f;///0.9
void on_trackbar( int, void* )
{
    //This function gets called whenever a
    // trackbar position is changed
    L1_autoencoder_noise_ratio = ((float) L1_noise_int);
    L2_autoencoder_noise_ratio = ((float) L2_noise_int);
    L3_autoencoder_noise_ratio = ((float) L3_noise_int);
    L1_LearningRate = 0.0001*((float)tune_L1_int);
    L2_LearningRate = 0.0001*((float)tune_L2_int);
    L3_LearningRate = 0.0001*((float)tune_L3_int);
    L1_Momentum = 0.000999*((float)tune_L1_moment_int);
    L2_Momentum = 0.000999*((float)tune_L2_moment_int);
    L3_Momentum = 0.000999*((float)tune_L3_moment_int);
///    Low_Target_value = 0.01*((float)target_low_value_int);
    printf("L1_autoencoder_noise_ratio =%f\n", L1_autoencoder_noise_ratio);
    printf("L2_autoencoder_noise_ratio =%f\n", L2_autoencoder_noise_ratio);
    printf("L3_autoencoder_noise_ratio =%f\n", L3_autoencoder_noise_ratio);
    printf("L1_LearningRate =%f\n", L1_LearningRate);
    printf("L2_LearningRate =%f\n", L2_LearningRate);
    printf("L3_LearningRate =%f\n", L3_LearningRate);
    printf("L1_Momentum =%f\n", L1_Momentum);
    printf("L2_Momentum =%f\n", L2_Momentum);
    printf("L3_Momentum =%f\n", L3_Momentum);
///    printf("Low_Target_value =%f\n", Low_Target_value);
}
void createTrackbars()
{
    namedWindow(trackbarWindowName,0);
    char TrackbarName[50];
    sprintf( TrackbarName, "Control values");
    createTrackbar( "L1 noise [%] ", trackbarWindowName, &L1_noise_int, H_MAX, on_trackbar );
    createTrackbar( "L2 noise [%] ", trackbarWindowName, &L2_noise_int, H_MAX, on_trackbar );
    createTrackbar( "L3 noise [%] ", trackbarWindowName, &L3_noise_int, H_MAX, on_trackbar );
    createTrackbar( "L1 tune learning  gain [*0.0001] ", trackbarWindowName, &tune_L1_int, 1000, on_trackbar );
    createTrackbar( "L2 tune learning  gain [*0.0001] ", trackbarWindowName, &tune_L2_int, 1000, on_trackbar );
    createTrackbar( "L3 tune learning  gain [*0.0001] ", trackbarWindowName, &tune_L3_int, 1000, on_trackbar );
    createTrackbar( "L1 tune momentum  [*0.000999] ", trackbarWindowName, &tune_L1_moment_int, 1000, on_trackbar );
    createTrackbar( "L2 tune momentum  [*0.000999] ", trackbarWindowName, &tune_L2_moment_int, 1000, on_trackbar );
    createTrackbar( "L3 tune momentum  [*0.000999] ", trackbarWindowName, &tune_L3_moment_int, 1000, on_trackbar );
///    createTrackbar( "target_low_value_int [*0.01] ", trackbarWindowName, &target_low_value_int, H_MAX, on_trackbar );
///Init
    L1_autoencoder_noise_ratio = ((float) L1_noise_int);
    L2_autoencoder_noise_ratio = ((float) L2_noise_int);
    L3_autoencoder_noise_ratio = ((float) L3_noise_int);
    L1_LearningRate = 0.0001*((float)tune_L1_int);
    L2_LearningRate = 0.0001*((float)tune_L2_int);
    L3_LearningRate = 0.0001*((float)tune_L3_int);
    L1_Momentum = 0.000999*((float)tune_L1_moment_int);
    L2_Momentum = 0.000999*((float)tune_L2_moment_int);
    L3_Momentum = 0.000999*((float)tune_L3_moment_int);
///    Low_Target_value = 0.01*((float)target_low_value_int);
    printf("L1_autoencoder_noise_ratio =%f\n", L1_autoencoder_noise_ratio);
    printf("L2_autoencoder_noise_ratio =%f\n", L2_autoencoder_noise_ratio);
    printf("L3_autoencoder_noise_ratio =%f\n", L3_autoencoder_noise_ratio);
    printf("L1_LearningRate =%f\n", L1_LearningRate);
    printf("L2_LearningRate =%f\n", L2_LearningRate);
    printf("L3_LearningRate =%f\n", L3_LearningRate);
    printf("L1_Momentum =%f\n", L1_Momentum);
    printf("L2_Momentum =%f\n", L2_Momentum);
    printf("L3_Momentum =%f\n", L3_Momentum);
}
///===========================End Trackbar adj functions and variables ===============

void randomize_dropoutHid(int *zero_ptr_dropoutHidden, int HiddenNodes, int verification)
{
    int drop_out_part = HiddenNodes * drop_out_percent/100;//
    int*ptr_dropoutHidden;

    for(int i=0; i<HiddenNodes; i++)
    {
        ptr_dropoutHidden = zero_ptr_dropoutHidden + i;
        *ptr_dropoutHidden = 0;//reset
    }
    int check_how_many_dropout = 0;
    if(verification == 0)
    {
        for(int k=0; k<HiddenNodes*2; k++) ///Itterate max HiddenNodes*2 number of times then give up to reach drop_out_part
        {
            for(int i=0; i<(drop_out_part-check_how_many_dropout); i++)
            {
                int r=0;
                r = rand() % (HiddenNodes-1);
                ptr_dropoutHidden = zero_ptr_dropoutHidden + r;
                *ptr_dropoutHidden = 1;///
            }
            check_how_many_dropout = 0;
            for(int j=0; j<HiddenNodes; j++)
            {
                ptr_dropoutHidden = zero_ptr_dropoutHidden + j;
                check_how_many_dropout += *ptr_dropoutHidden;
            }
            if(check_how_many_dropout >= drop_out_part)
            {
                break;
            }
        }
        //  printf("check_how_many_dropout =%d\n", check_how_many_dropout);
    }
}

#ifdef USE_CIFAR
const int How_Many_CIFAR_batch_in_use = 4;///I use only batch 1..4 and leave  batch 5 for evaluation later
#endif // USE_CIFAR
#ifdef USE_MNIST
/// Input data from
/// t10k-images-idx3-ubyte
/// t10k-labels-idx1-ubyte
/// train-images-idx3-ubyte
/// train-labels-idx1-ubyte
/// http://yann.lecun.com/exdb/mnist/
int use_MNIST_verify_set=0;
const int MNIST_height = 28;
const int MNIST_width  = 28;
int MNIST_nr_of_img_p_batch = 60000;
///const int Const_nr_pic = 60000;
const int MNIST_pix_size = MNIST_height*MNIST_width;
const int MNIST_RGB_pixels = MNIST_pix_size;
///char data_10k_MNIST[10000][MNIST_pix_size];
///char data_60k_MNIST[60000][MNIST_pix_size];

const int MNIST_header_offset = 16;
const int MNIST_lable_offset = 8;
/*
TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000801(2049) magic number (MSB first)
0004     32 bit integer  10000            number of items
0008     unsigned byte   ??               label
0009     unsigned byte   ??               label
........
xxxx     unsigned byte   ??               label
The labels values are 0 to 9.
TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000803(2051) magic number
0004     32 bit integer  10000            number of images
0008     32 bit integer  28               number of rows
0012     32 bit integer  28               number of columns
0016     unsigned byte   ??               pixel
0017     unsigned byte   ??               pixel
........
xxxx     unsigned byte   ??               pixel
Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
*/
int get_MNIST_file_size(void)
{
    int file_size=0;
    FILE *fp2;
    if(use_MNIST_verify_set==0)
    {
        fp2 = fopen("train-images-idx3-ubyte", "r");
    }
    else
    {
        fp2 = fopen("t10k-images-idx3-ubyte", "r");
    }
    if (fp2 == NULL)
    {
        if(use_MNIST_verify_set==0)
        {
            puts("Error while opening file train-images-idx3-ubyte");
        }
        else
        {
            puts("Error while opening file t10k-images-idx3-ubyte");
        }
        exit(0);
    }

    fseek(fp2, 0L, SEEK_END);
    file_size = ftell(fp2);
    printf("file_size %d\n", file_size);
    rewind(fp2);
    fclose(fp2);
    return file_size;
}

int get_MNIST_lable_file_size(void)
{
    int file_size=0;
    FILE *fp2;
    if(use_MNIST_verify_set==0)
    {
        fp2 = fopen("train-labels-idx1-ubyte", "r");
    }
    else
    {
        fp2 = fopen("t10k-labels-idx1-ubyte", "r");
    }

    if (fp2 == NULL)
    {
        if(use_MNIST_verify_set==0)
        {
            puts("Error while opening file train-labels-idx1-ubyte");
        }
        else
        {
            puts("Error while opening file t10k-labels-idx1-ubyte");
        }
        exit(0);
    }

    fseek(fp2, 0L, SEEK_END);
    file_size = ftell(fp2);
    printf("file_size %d\n", file_size);
    rewind(fp2);
    fclose(fp2);
    return file_size;
}
#endif // USE_MNIST

int main()
{
    createTrackbars();
    printf("auto_save_ON =%d\n", auto_save_ON);

    FILE *fp2;
///    FILE *fp3;
///    FILE *fp4;
    int print_only_100=0;
    srand (static_cast <unsigned> (time(0)));//Seed the randomizer
    float Rando=0.0f;
    int pool_sqr = sqrt(pooling);
#ifdef USE_CIFAR
    int nr_of_CIFAR_file_bytes=0;
    nr_of_CIFAR_file_bytes = get_CIFAR_file_size();
    printf("Byte size of data_batch_1.bin = %d\n", nr_of_CIFAR_file_bytes);
    ///L1
    const int CIFAR_height = 32;
    const int CIFAR_width  = 32;
    const int CIFAR_nr_of_img_p_batch = 10000;
    const int CIFAR_RGB_pixels = CIFAR_height*CIFAR_width;
#endif // USE_CIFAR
#ifdef USE_IMAGE
    High_Target_value = 1.0f;
    Low_Target_value = 0.0f;
    fully_out_nodes = 1;
    int nr_of_pos_image;
    int nr_of_neg_image;
    int pos_image_number;///Random select
    int neg_image_number;///
    int toggle_pos_neg;///1= positive, 0= negative select during training
    printf("Enter how many positive image it is in \positive\0...JPG\n");
    scanf("%d", &nr_of_pos_image);
    printf("Enter how many negative image it is in \negative\0...JPG\n");
    scanf("%d", &nr_of_neg_image);
    printf("nr_of_pos_image =%d\n", nr_of_pos_image);
    printf("nr_of_neg_image =%d\n", nr_of_neg_image);
    Training_fc =1;
    printf("Enter the pixel width of the input data image will be\n");
    scanf("%d", &IMAGE_width);
    printf("Enter the pixel height of the input data image will be\n");
    scanf("%d", &IMAGE_height);
    getchar();
    printf("IMAGE_height %d\n", IMAGE_height);
    printf("IMAGE_width %d\n", IMAGE_width);
    int IMAGE_RGB_pixels = IMAGE_height*IMAGE_width;
#endif // USE_IMAGE
    printf("High_Target_value %f\n", High_Target_value);
    printf("Low_Target_value %f\n", Low_Target_value);
    if((FL1_srt_size%2) == 1)
    {
        printf("FL1_srt_size =%d\n", FL1_srt_size);
    }
    else
    {
        printf("Error FL1_srt_size is a Even number not allowed FL1_srt_size =%d\n", FL1_srt_size);
        exit(0);
    }
    int FL1_depth = 3;///BRG
    int FL1_size = FL1_srt_size * FL1_srt_size * FL1_depth;///The size of one Feature FL1


#ifdef USE_MNIST
    int L1_conv_hight = MNIST_height - FL1_srt_size + 1;///No padding. This is the hight of the L1 Convolution cube.
    int L1_conv_width = MNIST_width  - FL1_srt_size + 1;///No padding. This is the width of the L1 Convolution cube.
#endif // USE_MNIST
#ifdef USE_CIFAR
    int L1_conv_hight = CIFAR_height - FL1_srt_size + 1;///No padding. This is the hight of the L1 Convolution cube.
    int L1_conv_width = CIFAR_width  - FL1_srt_size + 1;///No padding. This is the width of the L1 Convolution cube.
#endif // USE_CIFAR
#ifdef USE_IMAGE
    int L1_conv_hight = IMAGE_height - FL1_srt_size + 1;///No padding. This is the hight of the L1 Convolution cube.
    int L1_conv_width = IMAGE_width  - FL1_srt_size + 1;///No padding. This is the width of the L1 Convolution cube.
#endif // USE_IMAGE
    L1_conv_hight /= L1_stride;
    L1_conv_width /= L1_stride;
    ///L2
    if((FL2_srt_size%2) == 1)
    {
        printf("FL2_srt_size =%d\n", FL2_srt_size);
    }
    else
    {
        printf("Error FL2_srt_size is a Even number not allowed FL2_srt_size =%d\n", FL2_srt_size);
        exit(0);
    }
    int FL2_depth = L1_conv_depth;///This must always be same value
    int FL2_size = FL2_srt_size * FL2_srt_size * FL2_depth;///The size of one Feature FL2
    int L2_conv_hight = (L1_conv_hight/pool_sqr) - FL2_srt_size + 1;///No padding. This is the hight of the L2 Convolution cube.
    int L2_conv_width = (L1_conv_width/pool_sqr) - FL2_srt_size + 1;///No padding. This is the width of the L2 Convolution cube.
    L2_conv_hight /= L2_stride;
    L2_conv_width /= L2_stride;
    ///L3
    if((FL3_srt_size%2) == 1)
    {
        printf("FL3_srt_size =%d\n", FL3_srt_size);
    }
    else
    {
        printf("Error FL3_srt_size is a Even number not allowed FL3_srt_size =%d\n", FL3_srt_size);
        exit(0);
    }
    int FL3_depth = L2_conv_depth;///This must always be same value
    int FL3_size = FL3_srt_size * FL3_srt_size * FL3_depth;///The size of one Feature FL3
    int L3_conv_hight = (L2_conv_hight/pool_sqr) - FL3_srt_size + 1;///No padding. This is the hight of the L3 Convolution cube.
    int L3_conv_width = (L2_conv_width/pool_sqr) - FL3_srt_size + 1;///No padding. This is the width of the L3 Convolution cube.
    L3_conv_hight /= L3_stride;
    L3_conv_width /= L3_stride;
    printf("L1_stride = %d\n", L1_stride);
    printf("L2_stride = %d\n", L2_stride);
    printf("L3_stride = %d\n", L3_stride);
    printf("pooling = %d\n", pooling);
    printf("fc take data from pool=0 or conv=1 cube. connect_fc_to_pool_or_conv =%d\n", connect_fc_to_pool_or_conv);

    printf("L1_conv_hight %d\n", L1_conv_hight);
    printf("L1_conv_width %d\n", L1_conv_width);
    printf("L1_conv_hight/pool_sqr %d\n", L1_conv_hight/pool_sqr);
    printf("L1_conv_width/pool_sqr %d\n", L1_conv_width/pool_sqr);
    printf("L1_conv_depth =%d\n", L1_conv_depth);

    printf("L2_conv_hight %d\n", L2_conv_hight);
    printf("L2_conv_width %d\n", L2_conv_width);
    printf("L2_conv_hight/pool_sqr %d\n", L2_conv_hight/pool_sqr);
    printf("L2_conv_width/pool_sqr %d\n", L2_conv_width/pool_sqr);
    printf("L2_conv_depth =%d\n", L2_conv_depth);

    printf("L3_conv_hight %d\n", L3_conv_hight);
    printf("L3_conv_width %d\n", L3_conv_width);
    printf("L3_conv_hight/pool_sqr %d\n", L3_conv_hight/pool_sqr);
    printf("L3_conv_width/pool_sqr %d\n", L3_conv_width/pool_sqr);
    printf("L3_conv_depth =%d\n", L3_conv_depth);

/// Bugg   int L3toL2_feature_revers_size      = FL3_srt_size                  * pool_sqr + 1 + FL2_srt_size/pool_sqr;
    int L3toL2_feature_revers_size      = ((FL3_srt_size * pool_sqr * L2_stride)-pool_sqr-1) + FL2_srt_size - 1;
/// Bugg   int L3toL2toL1_feature_revers_size  = L3toL2_feature_revers_size    * pool_sqr + 1 + FL1_srt_size/pool_sqr;
    int L3toL2toL1_feature_revers_size  = ((L3toL2_feature_revers_size * pool_sqr * L1_stride)-pool_sqr-1) + FL1_srt_size - 1;
    printf("L3toL2_feature_revers_size = %d\n", L3toL2_feature_revers_size);
    printf("L3toL2toL1_feature_revers_size = %d\n", L3toL2toL1_feature_revers_size);
    /// Bugg  int L2toL1_feature_reverse_size     = FL2_srt_size                  * pool_sqr + 1 + FL1_srt_size/pool_sqr;
    int L2toL1_feature_reverse_size     = ((FL2_srt_size * pool_sqr * L1_stride)-pool_sqr-1) + FL1_srt_size - 1;
    printf("L2toL1_feature_reverse_size = %d\n", L2toL1_feature_reverse_size);
///printf("(FL2_srt_size*pool_sqr + (FL1_srt_size-1)/2 +1) =%d\n", (FL2_srt_size*pool_sqr + (FL1_srt_size-1)/2 +1));

    char answer_character;

    Mat RGB_image, colour, local_norm_colour, noised_input, norm_32FC3_img, L1_patch_img, L1_noise_img, L1_autoenc_reconstructed, L1_autoenc_delta;
    Mat norm_B_32FC1img, norm_G_32FC1img, norm_R_32FC1img;
#ifdef USE_MNIST
    RGB_image.create(MNIST_height, MNIST_width, CV_8UC3);
    norm_32FC3_img.create(MNIST_height, MNIST_width, CV_32FC3);
    norm_B_32FC1img.create(MNIST_height, MNIST_width, CV_32FC1);
    norm_G_32FC1img.create(MNIST_height, MNIST_width, CV_32FC1);
    norm_R_32FC1img.create(MNIST_height, MNIST_width, CV_32FC1);

    printf("Would you like to use MNIST VERIFY set <Y>/<N> \n");
    answer_character = getchar();
    if(answer_character == 'Y' || answer_character == 'y')
    {
        use_MNIST_verify_set=1;
        MNIST_nr_of_img_p_batch = 10000;
        Training_fc =0;
    }
    else
    {
        use_MNIST_verify_set=0;
        MNIST_nr_of_img_p_batch = 60000;
        Training_fc =1;
    }
    printf("use_MNIST_verify_set =%d\n", use_MNIST_verify_set);
    printf("Training_fc =%d\n", Training_fc);

#endif // USE_MNIST
#ifdef USE_CIFAR
    RGB_image.create(CIFAR_height, CIFAR_width, CV_8UC3);
    norm_32FC3_img.create(CIFAR_height, CIFAR_width, CV_32FC3);
    norm_B_32FC1img.create(CIFAR_height, CIFAR_width, CV_32FC1);
    norm_G_32FC1img.create(CIFAR_height, CIFAR_width, CV_32FC1);
    norm_R_32FC1img.create(CIFAR_height, CIFAR_width, CV_32FC1);
#endif // USE_CIFAR
#ifdef USE_IMAGE
    RGB_image.create(IMAGE_height, IMAGE_width, CV_8UC3);
    norm_32FC3_img.create(IMAGE_height, IMAGE_width, CV_32FC3);
    norm_B_32FC1img.create(IMAGE_height, IMAGE_width, CV_32FC1);
    norm_G_32FC1img.create(IMAGE_height, IMAGE_width, CV_32FC1);
    norm_R_32FC1img.create(IMAGE_height, IMAGE_width, CV_32FC1);
#endif // USE_IMAGE
    ///L1 autoencoder input, noised, recontruction and delta out
    L1_patch_img.create             (FL1_srt_size, FL1_srt_size, CV_32FC3);/// This have same size as the Feature L1 size
    L1_noise_img.create             (FL1_srt_size, FL1_srt_size, CV_32FC3);/// This have same size as the Feature L1 size
    L1_autoenc_reconstructed.create (FL1_srt_size, FL1_srt_size, CV_32FC3);
    L1_autoenc_delta.create         (FL1_srt_size, FL1_srt_size, CV_32FC3);
    float *zero_ptr_L1_patch_img  = L1_patch_img.ptr<float>(0);
    float *index_ptr_L1_patch_img = L1_patch_img.ptr<float>(0);
    float *zero_ptr_L1_noise_img  = L1_noise_img.ptr<float>(0);
    float *index_ptr_L1_noise_img = L1_noise_img.ptr<float>(0);

    float *zero_ptr_L1_autoenc_reconstructed  = L1_autoenc_reconstructed.ptr<float>(0);
    float *index_ptr_L1_autoenc_reconstructed = L1_autoenc_reconstructed.ptr<float>(0);
    float *zero_ptr_L1_autoenc_delta  = L1_autoenc_delta.ptr<float>(0);
    float *index_ptr_L1_autoenc_delta = L1_autoenc_delta.ptr<float>(0);
    const float FLx_bias_value = 1.0f;
///======================================
///L1 layer1. setup L1 tied weight's
///======================================

    float **L1_weight_matrix_M;//Pointer fo a dynamic array. This weight_matrix_M will then have a size of rows = nr_of_pixels, colums = Hidden_nodes.
    float **L1_change_weight_M;//Pointer fo a dynamic array. This weight_matrix_M will then have a size of rows = nr_of_pixels, colums = Hidden_nodes.
    L1_weight_matrix_M = new float *[FL1_size+1];///+1 is because the bias. So the nr FLx_size+1 is the bias weight.
    L1_change_weight_M = new float *[FL1_size+1];///+1 is because the bias. So the nr FLx_size+1 is the bias weight.
    for(int i=0; i < FL1_size+1; i++)///+1 is because the bias. So the nr FLx_size+1 is the bias weight.
    {
        L1_weight_matrix_M[i] = new float[L1_conv_depth];
        L1_change_weight_M[i] = new float[L1_conv_depth];
    }
    float *f_data;///File data read/write connect to tied weights
    f_data = new float[((FL1_size+1)*L1_conv_depth)];///File data is same size as all tied weights
    int ix=0;///index to f_data[ix]

///=====================================
///
    ///L2 autoencoder input, noised, recontruction and delta out
    float **L2_patch_vect;
    float **L2_noise_vect;
    float **L2_autoenc_reconstructed;
    float **L2_autoenc_delta;
    L2_patch_vect = new float *[FL2_srt_size*FL2_srt_size];
    L2_noise_vect = new float *[FL2_srt_size*FL2_srt_size];
    L2_autoenc_reconstructed    = new float *[FL2_srt_size*FL2_srt_size];
    L2_autoenc_delta            = new float *[FL2_srt_size*FL2_srt_size];
    for(int i=0; i<FL2_srt_size*FL2_srt_size; i++) ///
    {
        L2_patch_vect[i] = new float[FL2_depth];
        L2_noise_vect[i] = new float[FL2_depth];
        L2_autoenc_reconstructed[i] = new float[FL2_depth];
        L2_autoenc_delta [i] = new float[FL2_depth];
    }
///======================================
///L2 layer2. setup L2 tied weight's
///======================================

    float **L2_weight_matrix_M;//Pointer fo a dynamic array. This weight_matrix_M will then have a size of rows = nr_of_pixels, colums = Hidden_nodes.
    float **L2_change_weight_M;//Pointer fo a dynamic array. This weight_matrix_M will then have a size of rows = nr_of_pixels, colums = Hidden_nodes.
    L2_weight_matrix_M = new float *[FL2_size+1];///+1 is because the bias. So the nr FLx_size+1 is the bias weight.
    L2_change_weight_M = new float *[FL2_size+1];///+1 is because the bias. So the nr FLx_size+1 is the bias weight.
    for(int i=0; i < FL2_size+1; i++)///+1 is because the bias. So the nr FLx_size+1 is the bias weight.
    {
        L2_weight_matrix_M[i] = new float[L2_conv_depth];
        L2_change_weight_M[i] = new float[L2_conv_depth];
    }
    float *f2_data;///File data read/write connect to tied weights
    f2_data = new float[((FL2_size+1)*L2_conv_depth)];///File data is same size as all tied weights
///   int ix=0;///index to f_data[ix]

///=====================================

///=====================================
///
    ///L3 autoencoder input, noised, recontruction and delta out
    float **L3_patch_vect;
    float **L3_noise_vect;
    float **L3_autoenc_reconstructed;
    float **L3_autoenc_delta;
    L3_patch_vect = new float *[FL3_srt_size*FL3_srt_size];
    L3_noise_vect = new float *[FL3_srt_size*FL3_srt_size];
    L3_autoenc_reconstructed    = new float *[FL3_srt_size*FL3_srt_size];
    L3_autoenc_delta            = new float *[FL3_srt_size*FL3_srt_size];
    for(int i=0; i<FL3_srt_size*FL3_srt_size; i++) ///
    {
        L3_patch_vect[i] = new float[FL3_depth];
        L3_noise_vect[i] = new float[FL3_depth];
        L3_autoenc_reconstructed[i] = new float[FL3_depth];
        L3_autoenc_delta [i] = new float[FL3_depth];
    }
///======================================
///L3 layer3. setup L3 tied weight's
///======================================

    float **L3_weight_matrix_M;//Pointer fo a dynamic array. This weight_matrix_M will then have a size of rows = nr_of_pixels, colums = Hidden_nodes.
    float **L3_change_weight_M;//Pointer fo a dynamic array. This weight_matrix_M will then have a size of rows = nr_of_pixels, colums = Hidden_nodes.
    L3_weight_matrix_M = new float *[FL3_size+1];///+1 is because the bias. So the nr FLx_size+1 is the bias weight.
    L3_change_weight_M = new float *[FL3_size+1];///+1 is because the bias. So the nr FLx_size+1 is the bias weight.
    for(int i=0; i < FL3_size+1; i++)///+1 is because the bias. So the nr FLx_size+1 is the bias weight.
    {
        L3_weight_matrix_M[i] = new float[L3_conv_depth];
        L3_change_weight_M[i] = new float[L3_conv_depth];
    }
    float *f3_data;///File data read/write connect to tied weights
    f3_data = new float[((FL3_size+1)*L3_conv_depth)];///File data is same size as all tied weights
///   int ix=0;///index to f_data[ix]

///=====================================

///**********************************************************************************
///*** Regarding the Fully connected sigmoid Logistic regression neural network *****
///**********************************************************************************
    int fc_input_NODES=0;
    if(connect_fc_to_layer== 2)
    {
        ///Note Mode connect_fc_to_layer == 2 connect the fully network to L2_pool_cube (Not to L3_pool_cube how is unused)
        if(connect_fc_to_pool_or_conv == 0)
        {
            fc_input_NODES = L2_conv_depth * ((L2_conv_hight/pool_sqr) * (L2_conv_width/pool_sqr));///Use data from pool cube
        }
        else
        {
            fc_input_NODES = L2_conv_depth * (L2_conv_hight * L2_conv_width);///Use data from the unpooled data direct from convolution cube data
        }
    }
    if(connect_fc_to_layer== 3)
    {
        ///Mode connect_fc_to_layer== 3 will Connect to last conv/pool layer
        if(connect_fc_to_pool_or_conv == 0)
        {
            fc_input_NODES = L3_conv_depth * ((L3_conv_hight/pool_sqr) * (L3_conv_width/pool_sqr));///Use data from pool cube
        }
        else
        {
            fc_input_NODES = L3_conv_depth * (L3_conv_hight * L3_conv_width);///Use data from the unpooled data direct from convolution cube data
        }
    }
    printf("fc_input_NODES =%d\n", fc_input_NODES);
    float *fc_input_node;
    float *fc_hidden_node;
    float *fc_output_node;
    float *fc_target_node;
    float *fc_output_delta;
    float *fc_hidden_delta;
    float *fc_input_delta;///this is used only for backpropagation to me in2hid wehigt updates (not needed for tied weight)
    float *fc_softmax_output;
    float *fc_softmax_out_delta;
    int *dropoutHidden;///dropout table
    dropoutHidden = new int[fully_hidd_nodes];///data 0 normal fc_hidden_node. 1= dropout this fc_hidden_node this training turn.
    fc_input_node = new float[fc_input_NODES];///fc means fully connected. sigmoid of this L2_hidden_node = new float[L2_fc_input_NODES];
    fc_hidden_node = new float[fully_hidd_nodes];///
    fc_output_node = new float[fully_out_nodes];
    fc_target_node = new float[fully_out_nodes];///The target value from lable file should be put in this
    fc_output_delta = new float[fully_out_nodes];
    fc_hidden_delta = new float[fully_hidd_nodes];///
    fc_input_delta = new float[fc_input_NODES];///this is used only for backpropagation to me in2hid wehigt updates (not needed for tied weight)
    fc_softmax_output    = new float[fully_out_nodes];
    fc_softmax_out_delta = new float[fully_out_nodes];

    float **fc_hidden_weight;
    float **fc_output_weight;
    float **fc_change_hidden_weight;
    float **fc_change_output_weight;

    fc_hidden_weight = new float*[fc_input_NODES+1];///+1 for bias node connection
    fc_change_hidden_weight = new float*[fc_input_NODES+1];
    for(int i=0; i<fc_input_NODES+1; i++) ///+1 for bias node connection
    {
        fc_hidden_weight[i] = new float[fully_hidd_nodes];
        fc_change_hidden_weight[i] = new float[fully_hidd_nodes];
    }
    fc_output_weight = new float*[fully_hidd_nodes+1];///+1 for bias node connection
    fc_change_output_weight = new float*[fully_hidd_nodes+1];
    for(int i=0; i<fully_hidd_nodes+1; i++) ///+1 for bias node connection
    {
        fc_output_weight[i] = new float[fully_out_nodes];
        fc_change_output_weight[i] = new float[fully_out_nodes];
    }

    float *f_data_fc_h_w;///File data read/write connect to fc_hidden_weight weights
    f_data_fc_h_w = new float[fully_hidd_nodes * (fc_input_NODES+1)];///File data is same size as fc_hidden_weight
    float *f_data_fc_o_w;///File data read/write connect to fc_output_weight weights
    f_data_fc_o_w = new float[(fully_hidd_nodes+1) * fully_out_nodes];///File data is same size as fc_output_weight

///*****************************************************************************
///************ End of dynamic declaration *************************************
///*****************************************************************************

///L1 visualize features
    int L1_sqr_of_H_nod_plus1=0;
    L1_sqr_of_H_nod_plus1 = sqrt(L1_conv_depth);
    L1_sqr_of_H_nod_plus1 += 1;///+1 becasue sqrt() result will be round up downwards to an integer and that may result in to small square then
    printf("L1_sqr_of_H_nod_plus1 %d\n", L1_sqr_of_H_nod_plus1);
    float* L1_ptr_M_matrix;
    Mat L1_visual_all_feature;
    L1_visual_all_feature.create(FL1_srt_size * L1_sqr_of_H_nod_plus1, FL1_srt_size * L1_sqr_of_H_nod_plus1,CV_32FC3);

///L2 visualize features is trickyer then L1 because the depth now is not fit in CV_32FC3 "BGR" format now this depth is = L1_conv_depth = FL2_depth
    int L2_vis_F_Hight = FL2_srt_size * L2_conv_depth;
    int L2_vis_F_Width = FL2_srt_size * FL2_depth;
    float* L2_ptr_M_matrix;
    Mat L2_visual_all_feature;
    L2_visual_all_feature.create(L2_vis_F_Hight, L2_vis_F_Width,CV_32FC1);///This is gray because the depth is larger then "BGR"

///L3 visualize features
    int L3_vis_F_Hight = FL3_srt_size * L3_conv_depth;
    int L3_vis_F_Width = FL3_srt_size * FL3_depth;
    float* L3_ptr_M_matrix;
    Mat L3_visual_all_feature;
    L3_visual_all_feature.create(L3_vis_F_Hight, L3_vis_F_Width,CV_32FC1);///This is gray because the depth is larger then "BGR"

///visualize_L2toL1
///    Mat visualize_L2toL1;///This will show L2 feature how it looks when projected with L1 features
///    visualize_L2toL1.create((FL2_srt_size*pool_sqr + (FL1_srt_size-1)/2 +1) , (FL2_srt_size*pool_sqr + (FL1_srt_size-1)/2 +1)*L2_conv_depth, CV_32FC3);

    int sqr_L2_conv_depth_plus1=0;
    sqr_L2_conv_depth_plus1 = sqrt(L2_conv_depth);
    sqr_L2_conv_depth_plus1 +=1;///+1 becasue sqrt() result will be round up downwards to an integer and that may result in to small square then
    Mat sq_visualize_L2toL1;///This will show L2 feature how it looks when projected with L1 features
    sq_visualize_L2toL1.create(L2toL1_feature_reverse_size*sqr_L2_conv_depth_plus1, L2toL1_feature_reverse_size*sqr_L2_conv_depth_plus1, CV_32FC3);

///visualize_L3toL1
///    Mat visualize_L3toL1;///This will show L3 feature how it looks when projected with L1 features
///    visualize_L3toL1.create((FL3_srt_size*pool_sqr*pool_sqr + (FL1_srt_size-1)/2 +1) , (FL3_srt_size*pool_sqr*pool_sqr + (FL1_srt_size-1)/2 +1)*L3_conv_depth, CV_32FC3);

    int sqr_L3_conv_depth_plus1=0;
    sqr_L3_conv_depth_plus1 = sqrt(L3_conv_depth);
    sqr_L3_conv_depth_plus1 +=1;///+1 becasue sqrt() result will be round up downwards to an integer and that may result in to small square then
    Mat sq_visualize_L3toL1;///This will show L2 feature how it looks when projected with L1 features
    sq_visualize_L3toL1.create(L3toL2toL1_feature_revers_size*sqr_L3_conv_depth_plus1, L3toL2toL1_feature_revers_size*sqr_L3_conv_depth_plus1, CV_32FC3);


///L1 feature start noise
    for(int i=0; i<FL1_size+1; i++)///+1 is because the bias. So the nr FLx_size+1 is the bias weight.
    {
        L1_ptr_M_matrix = &L1_weight_matrix_M[i][0];
        for(int j=0; j<(L1_conv_depth); j++)
        {
            Rando = (float) (rand() % 65535) / 65536;//0..1.0 range
            Rando -= 0.5f;
            Rando *= start_weight_noise_range;
            *L1_ptr_M_matrix = Rando;
            L1_ptr_M_matrix++;
            L1_change_weight_M[i][j] = 0.0f;
        }
    }

///L2 feature start noise
    for(int i=0; i<FL2_size+1; i++)///+1 is because the bias. So the nr FLx_size+1 is the bias weight.
    {
        L2_ptr_M_matrix = &L2_weight_matrix_M[i][0];
        for(int j=0; j<(L2_conv_depth); j++)
        {
            Rando = (float) (rand() % 65535) / 65536;//0..1.0 range
            Rando -= 0.5f;
            Rando *= start_weight_noise_range;
            *L2_ptr_M_matrix = Rando;
            L2_ptr_M_matrix++;
            L2_change_weight_M[i][j] = 0.0f;
        }
    }
///L3 feature start noise
    for(int i=0; i<FL3_size+1; i++)///+1 is because the bias. So the nr FLx_size+1 is the bias weight.
    {
        L3_ptr_M_matrix = &L3_weight_matrix_M[i][0];
        for(int j=0; j<(L3_conv_depth); j++)
        {
            Rando = (float) (rand() % 65535) / 65536;//0..1.0 range
            Rando -= 0.5f;
            Rando *= start_weight_noise_range;
            *L3_ptr_M_matrix = Rando;
            L3_ptr_M_matrix++;
            L3_change_weight_M[i][j] = 0.0f;
        }
    }
///************* Initialize randomized noise on fc_weight *********
    for(int i=0; i<fc_input_NODES+1; i++)///+1 is because the bias. So the nr fc_input_NODES+1 is the bias weight.
    {
        for(int j=0; j<fully_hidd_nodes; j++)
        {
            Rando = (float) (rand() % 65535) / 65536;//0..1.0 range
            Rando -= 0.5f;
            Rando *= fc_start_weight_noise_range;
            fc_hidden_weight[i][j] = Rando;///Noise around 0.5f
            fc_change_hidden_weight[i][j] = 0.0f;///Initialize with zero
        }
    }

    for(int i=0; i<fully_hidd_nodes+1; i++)///+1 is because the bias.
    {
        for(int j=0; j<fully_out_nodes; j++)
        {
            Rando = (float) (rand() % 65535) / 65536;//0..1.0 range
            Rando -= 0.5f;
            Rando *= fc_start_weight_noise_range;
            fc_output_weight[i][j] = Rando;///Noise around 0.5f
            fc_change_output_weight[i][j] = 0.0f;///Initialize with zero
        }
    }
    printf("Cleared with noise the fully connected network weights\n");
///***********************************

///============================================

/// L1 Convolution and Pooling cube
    float **L1_conv_cube;
    float **L1_pool_cube;
    L1_conv_cube = new float*[L1_conv_hight * L1_conv_width];///This is One sheet of the conv code
    for(int i=0; i<(L1_conv_hight * L1_conv_width); i++)
    {
        L1_conv_cube[i] = new float[L1_conv_depth];
    }
    L1_pool_cube = new float*[(L1_conv_hight/pool_sqr) * (L1_conv_width/pool_sqr)];
    for(int i=0; i<((L1_conv_hight/pool_sqr) * (L1_conv_width/pool_sqr)); i++)
    {
        L1_pool_cube[i] = new float[L1_conv_depth];
    }
    int L1_pool_tracking=0;///for example if max pooling is arrange 4 -> 1 then 0..3 will be the value depend on which L1_conv_cube[][] node was strongest on that conv pos
///

/// L2 Convolution and Pooling cube
    float **L2_conv_cube;
    float **L2_pool_cube;
    int **L2_pool_track_cube;///Used only for finetune to track throue pooling back propagate detla L2
    L2_conv_cube = new float*[L2_conv_hight * L2_conv_width];///This is One sheet of the conv code
    for(int i=0; i<(L2_conv_hight * L2_conv_width); i++)
    {
        L2_conv_cube[i] = new float[L2_conv_depth];
    }
    L2_pool_cube = new float*[(L2_conv_hight/pool_sqr) * (L2_conv_width/pool_sqr)];
    L2_pool_track_cube = new int*[(L2_conv_hight/pool_sqr) * (L2_conv_width/pool_sqr)];///Used only for finetune L2 This is One sheet of the conv code
    for(int i=0; i<((L2_conv_hight/pool_sqr) * (L2_conv_width/pool_sqr)); i++)
    {
        L2_pool_cube[i] = new float[L2_conv_depth];
        L2_pool_track_cube[i] = new int[L2_conv_depth];
    }
    int L2_pool_tracking=0;///for example if max pooling is arrange 4 -> 1 then 0..3 will be the value depend on which L1_conv_cube[][] node was strongest on that conv pos
///
/// L3 Convolution and Pooling cube
    float **L3_conv_cube;
    float **L3_pool_cube;
    L3_conv_cube = new float*[L3_conv_hight * L3_conv_width];///This is One sheet of the conv code
    for(int i=0; i<(L3_conv_hight * L3_conv_width); i++)
    {
        L3_conv_cube[i] = new float[L3_conv_depth];
    }
    L3_pool_cube = new float*[(L3_conv_hight/pool_sqr) * (L3_conv_width/pool_sqr)];
    for(int i=0; i<((L3_conv_hight/pool_sqr) * (L3_conv_width/pool_sqr)); i++)
    {
        L3_pool_cube[i] = new float[L3_conv_depth];
    }
    int L3_pool_tracking=0;///for example if max pooling is arrange 4 -> 1 then 0..3 will be the value depend on which L1_conv_cube[][] node was strongest on that conv pos
///

    imshow("RGB_image", RGB_image);
    waitKey(1);
    //  getchar();

#ifdef USE_MNIST
    int MNIST_file_size=0;
///Read database train-images-idx3-ubyte
    MNIST_file_size = get_MNIST_file_size();
//read_10k_MNIST();
    char *MNIST_data;
    MNIST_data = new char[MNIST_file_size];
    FILE *fp;
    char c_data=0;
    if(use_MNIST_verify_set==0)
    {
        fp = fopen("train-images-idx3-ubyte","r");
    }
    else
    {
        fp = fopen("t10k-images-idx3-ubyte","r");
    }
    if(fp == NULL)
    {
        perror("Error in opening train-images-idx3-ubyte file");
        return(-1);
    }
    int MN_index=0;
    for(int i=0; i<MNIST_file_size; i++)
    {
        c_data = fgetc(fp);
        if( feof(fp) )
        {
            break;
        }
        //printf("c_data %d\n", c_data);
        MNIST_data[MN_index] = c_data;
        if((MNIST_header_offset-1)<i)
        {
            MN_index++;
        }
    }
    fclose(fp);
    printf("train.. or t10k.. ..-images-idx3-ubyte file is successfully loaded in to MNIST_data[MN_index] memory\n");
///Read lable
///Read train-labels-idx1-ubyte
    MNIST_file_size = get_MNIST_lable_file_size();
//read_10k_MNIST();
    char *MNIST_lable;
    MNIST_lable = new char[MNIST_file_size];
    // FILE *fp;
    c_data=0;
    if(use_MNIST_verify_set==0)
    {
        fp = fopen("train-labels-idx1-ubyte","r");
    }
    else
    {
        fp = fopen("t10k-labels-idx1-ubyte", "r");
    }


    if(fp == NULL)
    {
        if(use_MNIST_verify_set==0)
        {
            perror("Error in opening train-labels-idx1-ubyte file");
        }
        else
        {
            perror("Error in opening t10k-labels-idx1-ubyte file");
        }

        return(-1);
    }
    MN_index=0;
    for(int i=0; i<MNIST_file_size; i++)
    {

        c_data = fgetc(fp);
        if( feof(fp) )
        {
            break;
        }
        //printf("c_data %d\n", c_data);
        MNIST_lable[MN_index] = c_data;
        if((MNIST_lable_offset-1)<i)
        {
            MN_index++;
        }
    }
    fclose(fp);
    printf("train... or t10k...  ...-labels-idx1-ubyte file is successfully loaded in to MNIST_lable[MN_index] memory\n");

    int MNIST_nr = 0;
    Mat gray;
    gray.create(28,28,CV_8UC1);
    char *zero_ptr_gray = gray.ptr<char>(0);
    char *ptr_gray = gray.ptr<char>(0);
    getchar();
#endif // USE_MNIST
#ifdef USE_CIFAR
    fp1 = fopen("data_batch_1.bin", "r");
    if (fp1 == NULL)
    {
        puts("Error while opening file data_batch_1.bin");
        exit(0);
    }
    ///read_CIFAR_image()
    char* CIFAR_data;
    CIFAR_data = new char[nr_of_CIFAR_file_bytes];
    int MN_index=0;
    char c_data=0;
    for(int i=0; i<nr_of_CIFAR_file_bytes; i++)
    {
        c_data = fgetc(fp1);
        if( feof(fp1) )
        {
            break;
        }
        //printf("c_data %d\n", c_data);
        CIFAR_data[MN_index] = c_data;
        MN_index++;
    }
    fclose(fp1);
    printf("data_batch_1.bin data is put into CIFAR_data\n");

    int CIFAR_nr = 0;
    const int CIFAR_row_size = 3073;/// 1 byte label, 1024 RED ch, 1024 GREEN ch, 1024 BLUE ch
#endif // USE_CIFAR
    Mat read_image;
    srand (static_cast <unsigned> (time(0)));//Seed the randomizer
    char *zero_ptr_RGB_image =      RGB_image.ptr<char>(0);
    char *index_ptr_RGB_image =     RGB_image.ptr<char>(0);
    float *zero_ptr_norm_B_32FC1img  =   norm_B_32FC1img.ptr<float>(0);
    float *zero_ptr_norm_G_32FC1img  =   norm_G_32FC1img.ptr<float>(0);
    float *zero_ptr_norm_R_32FC1img  =   norm_R_32FC1img.ptr<float>(0);
    float *index_ptr_norm_B_32FC1img  =  norm_B_32FC1img.ptr<float>(0);
    float *index_ptr_norm_G_32FC1img  =  norm_G_32FC1img.ptr<float>(0);
    float *index_ptr_norm_R_32FC1img  =  norm_R_32FC1img.ptr<float>(0);
    float *zero_ptr_norm_32FC3_img =         local_norm_colour.ptr<float>(0);
    float *index_ptr_norm_32FC3_img =        local_norm_colour.ptr<float>(0);

///Test mat Conv and Pool L1
    Mat test1, test2, test3, pol_t1, pol_t2, pol_t3;
    test1.create(L1_conv_hight, L1_conv_width, CV_32FC1);
    float *test1_zero_ptr = test1.ptr<float>(0);
    float *test1_ptr = test1.ptr<float>(0);
    test2.create(L1_conv_hight, L1_conv_width, CV_32FC1);
    float *test2_zero_ptr = test2.ptr<float>(0);
    float *test2_ptr = test2.ptr<float>(0);
    test3.create(L1_conv_hight, L1_conv_width, CV_32FC1);
    float *test3_zero_ptr = test3.ptr<float>(0);
    float *test3_ptr = test3.ptr<float>(0);
    pol_t1.create(L1_conv_hight/pool_sqr, L1_conv_width/pool_sqr, CV_32FC1);
    float *pol_t1_zero_ptr = pol_t1.ptr<float>(0);
    float *pol_t1_ptr = pol_t1.ptr<float>(0);
    pol_t2.create(L1_conv_hight/pool_sqr, L1_conv_width/pool_sqr, CV_32FC1);
    float *pol_t2_zero_ptr = pol_t2.ptr<float>(0);
    float *pol_t2_ptr = pol_t2.ptr<float>(0);
    pol_t3.create(L1_conv_hight/pool_sqr, L1_conv_width/pool_sqr, CV_32FC1);
    float *pol_t3_zero_ptr = pol_t3.ptr<float>(0);
    float *pol_t3_ptr = pol_t3.ptr<float>(0);

///Test mat Conv and Pool L1
    Mat L2_test1, L2_test2, L2_test3, L2_pol_t1, L2_pol_t2, L2_pol_t3;
    L2_test1.create(L2_conv_hight, L2_conv_width, CV_32FC1);
    float *L2_test1_zero_ptr = L2_test1.ptr<float>(0);
    float *L2_test1_ptr = L2_test1.ptr<float>(0);
    L2_test2.create(L2_conv_hight, L2_conv_width, CV_32FC1);
    float *L2_test2_zero_ptr = L2_test2.ptr<float>(0);
    float *L2_test2_ptr = L2_test2.ptr<float>(0);
    L2_test3.create(L2_conv_hight, L2_conv_width, CV_32FC1);
    float *L2_test3_zero_ptr = L2_test3.ptr<float>(0);
    float *L2_test3_ptr = L2_test3.ptr<float>(0);
    L2_pol_t1.create(L2_conv_hight/pool_sqr, L2_conv_width/pool_sqr, CV_32FC1);
    float *L2_pol_t1_zero_ptr = L2_pol_t1.ptr<float>(0);
    float *L2_pol_t1_ptr = L2_pol_t1.ptr<float>(0);
    L2_pol_t2.create(L2_conv_hight/pool_sqr, L2_conv_width/pool_sqr, CV_32FC1);
    float *L2_pol_t2_zero_ptr = L2_pol_t2.ptr<float>(0);
    float *L2_pol_t2_ptr = L2_pol_t2.ptr<float>(0);
    L2_pol_t3.create(L2_conv_hight/pool_sqr, L2_conv_width/pool_sqr, CV_32FC1);
    float *L2_pol_t3_zero_ptr = L2_pol_t3.ptr<float>(0);
    float *L2_pol_t3_ptr = L2_pol_t3.ptr<float>(0);

/// cpp_func2 instanisation
    cpp_func2 comon_func_Obj1;
    comon_func_Obj1.nr_of_positions = FL1_size;///Size of the FL1_size = FL1_srt_size * FL1_srt_size * FL1_depth
    comon_func_Obj1.L2_nr_of_positions = FL2_size;
    comon_func_Obj1.L3_nr_of_positions = FL3_size;
    comon_func_Obj1.init();
    comon_func_Obj1.noise_percent = L1_autoencoder_noise_ratio;///
    comon_func_Obj1.L2_noise_percent = L2_autoencoder_noise_ratio;///
    comon_func_Obj1.L3_noise_percent = L3_autoencoder_noise_ratio;///
    comon_func_Obj1.print_help();

    Lx_attach_weight2mat L2_attach_weight2mat;
    L2_attach_weight2mat.FL_Height = FL2_srt_size;
    L2_attach_weight2mat.FL_Width  = FL2_srt_size;
    L2_attach_weight2mat.Lx_Hidden_nodes = L2_conv_depth;

    Lx_attach_weight2mat L3_attach_weight2mat;
    L3_attach_weight2mat.FL_Height = FL3_srt_size;
    L3_attach_weight2mat.FL_Width  = FL3_srt_size;
    L3_attach_weight2mat.Lx_Hidden_nodes = L3_conv_depth;

///    getchar();
    printf("Would you like to load stored Lx_weight_matrix_M.dat <Y>/<N> \n");
    answer_character = getchar();
    if(answer_character == 'Y' || answer_character == 'y')
    {
///L1 load
        sprintf(filename, "L1_weight_matrix_M.dat");
        fp2 = fopen(filename, "r");
        if (fp2 == NULL)
        {
            printf("Error while opening file L1_weight_matrix_M.dat");
            exit(0);
        }
        fread(f_data, sizeof f_data[0], ((FL1_size+1)*L1_conv_depth), fp2);///+1 is because the bias. So the nr FLx_size+1 is the bias weight.
        ix=0;
        for(int n=0; n<L1_conv_depth; n++)
        {
            for(int p=0; p<FL1_size+1; p++)///+1 is because the bias. So the nr FLx_size+1 is the bias weight.
            {
                L1_weight_matrix_M[p][n] = f_data[ix];///File data put in to tied weights
                ix++;
            }
        }
        fclose(fp2);
        printf("weights are loaded from L1_weight_matrix_M.dat file\n");
///L2 load
        sprintf(filename, "L2_weight_matrix_M.dat");
        fp2 = fopen(filename, "r");
        if (fp2 == NULL)
        {
            printf("Error while opening file L2_weight_matrix_M.dat");
            exit(0);
        }
        fread(f2_data, sizeof f2_data[0], ((FL2_size+1)*L2_conv_depth), fp2);///+1 is because the bias. So the nr FLx_size+1 is the bias weight.
        ix=0;
        for(int n=0; n<L2_conv_depth; n++)
        {
            for(int p=0; p<FL2_size+1; p++)///+1 is because the bias. So the nr FLx_size+1 is the bias weight.
            {
                L2_weight_matrix_M[p][n] = f2_data[ix];///File data put in to tied weights
                ix++;
            }
        }
        fclose(fp2);
        printf("weights are loaded from L2_weight_matrix_M.dat file\n");
///End L2 load
///L3 load
        sprintf(filename, "L3_weight_matrix_M.dat");
        fp2 = fopen(filename, "r");
        if (fp2 == NULL)
        {
            printf("Error while opening file L3_weight_matrix_M.dat");
            exit(0);
        }
        fread(f3_data, sizeof f3_data[0], ((FL3_size+1)*L3_conv_depth), fp2);///+1 is because the bias. So the nr FLx_size+1 is the bias weight.
        ix=0;
        for(int n=0; n<L3_conv_depth; n++)
        {
            for(int p=0; p<FL3_size+1; p++)///+1 is because the bias. So the nr FLx_size+1 is the bias weight.
            {
                L3_weight_matrix_M[p][n] = f3_data[ix];///File data put in to tied weights
                ix++;
            }
        }
        fclose(fp2);
        printf("weights are loaded from L3_weight_matrix_M.dat file\n");
///End L3 load

///**************** Load fully connected network weights. ********************
///********** Load fc_hidden_weight ********************
///  fc_hidden_weight[fc_input_NODES+1][fully_hidd_nodes]
        sprintf(filename, "fc_hidden_weight.dat");//Assigne a filename with index number added
        fp2 = fopen(filename, "r");
        if (fp2 == NULL)
        {
            printf("Error while opening file fc_hidden_weight.dat");
            exit(0);
        }
        fread(f_data_fc_h_w, sizeof f_data_fc_h_w[0], ((fc_input_NODES+1)*fully_hidd_nodes), fp2);
        ix=0;
        for(int n=0; n<fully_hidd_nodes; n++)
        {
            for(int p=0; p<fc_input_NODES+1; p++)///+1 is because the bias.
            {
                fc_hidden_weight[p][n] = f_data_fc_h_w[ix];
                ix++;
            }
        }
        fclose(fp2);
        printf("weights are loaded at fc_hidden_weight.dat file\n");
///********** End Load fc_hidden_weight ********************
///********** Load fc_output_weight ********************
///  fc_output_weight[fully_hidd_nodes+1][fully_out_nodes]
        sprintf(filename, "fc_output_weight.dat");//Assigne a filename with index number added
        fp2 = fopen(filename, "r");
        if (fp2 == NULL)
        {
            printf("Error while opening file fc_output_weight.dat");
            exit(0);
        }
        fread(f_data_fc_o_w, sizeof f_data_fc_o_w[0], ((fully_hidd_nodes+1)*fully_out_nodes), fp2);

        ix=0;
        for(int n=0; n<fully_out_nodes; n++)
        {
            for(int p=0; p<fully_hidd_nodes+1; p++)///+1 is because the bias.
            {
                fc_output_weight[p][n] = f_data_fc_o_w[ix];
                ix++;
            }
        }
        fclose(fp2);
        printf("weights are loaded at fc_output_weight.dat file\n");
///********** End Load fc_output_weight ********************
    }
    float noise;
#ifdef USE_IMAGE
    Mat labeling(80,350, CV_8UC3, Scalar(0,0,40));///Image show the number of highest category
    Mat rezied_img;
    int resize_WIDTH = 128;
    int resize_HEIGHT = 128;
    Size size(resize_WIDTH,resize_HEIGHT);//the dst image size,e.g.100x100
///resize(colour, rezied_img, size);

#else
    Mat labeling(80,80, CV_8UC3, Scalar(0,0,40));///Image show the number of highest category
#endif // USE_IMAGE
    float FLx_bias_reconstruction =0.0f;
    float loss=0.0f;
    float delta=0.0f;
    float bias_delta=0.0f;
    int rand_x_start_pos=0;
    int rand_y_start_pos=0;


    while(1)
    {

        ///Update
        comon_func_Obj1.noise_percent = L1_autoencoder_noise_ratio;///
        comon_func_Obj1.L2_noise_percent = L2_autoencoder_noise_ratio;///
        comon_func_Obj1.L3_noise_percent = L3_autoencoder_noise_ratio;///

#ifdef USE_MNIST
///*********************************************************************
///read data from ************* train-images-idx3-ubyte ************ file
///*********************************************************************
        MNIST_nr = (int) (rand() % MNIST_nr_of_img_p_batch);
        zero_ptr_gray = gray.ptr<char>(0);
        index_ptr_RGB_image = zero_ptr_RGB_image;
        for(int n=0; n<MNIST_pix_size; n++)
        {
            ptr_gray = zero_ptr_gray + n;
            *ptr_gray = MNIST_data[(MNIST_pix_size*MNIST_nr) +n];

            for(int i=0; i<RGB_image.channels(); i++)
            {
                *index_ptr_RGB_image = *ptr_gray;
                index_ptr_RGB_image++;
            }
        }
        ///   gray.convertTo(RGB_image, CV_8UC3);
#endif // USE_MNIST
#ifdef USE_CIFAR
        CIFAR_nr = (int) (rand() % CIFAR_nr_of_img_p_batch);
        /// which is a number in the range 0-9. The next 3072 bytes are the values of the pixels of the image.
        /// The first 1024 bytes are the red channel values, the next 1024 the green, and the final 1024 the blue.
        /// The values are stored in row-major order, so the first 32 bytes are the red channel values of the first row of the image.
        index_ptr_RGB_image = zero_ptr_RGB_image;
        for (int i=0; i<CIFAR_RGB_pixels; i++)
        {
            for(int BGR=0; BGR<3; BGR++)
            {
                *index_ptr_RGB_image = CIFAR_data[(CIFAR_nr*CIFAR_row_size) + ((2-BGR)*CIFAR_RGB_pixels) + i];///(2-BGR) will swap Blue and Red order
                index_ptr_RGB_image++;
            }
        }
#endif // USE_CIFAR
#ifdef USE_IMAGE
        int training_image =0;
        if(toggle_pos_neg == 0)
        {
            toggle_pos_neg=1;
            pos_image_number = (int) (rand() % nr_of_pos_image);// range
            training_image = pos_image_number;
            sprintf(filename, "./positive/%d.JPG", training_image);//Assigne a filename
        }
        else
        {
            toggle_pos_neg=0;
            neg_image_number = (int) (rand() % nr_of_neg_image);// range
            training_image = neg_image_number;
            sprintf(filename, "./negative/%d.JPG", training_image);//Assigne a filename
        }
        /// RGB_image = imread( filename, 1 );
        read_image = imread( filename, 1 );
        /// RGB_image.convertTo(RGB_image, COLOR_RGB2BGR);
        read_image.convertTo(read_image, COLOR_RGB2BGR);
        RGB_image = read_image.clone();
        if ( !RGB_image.data )
        {
            printf("\n");
            printf("==================================================\n");
            printf("No image data Error! Probably not find ./positive/.. or ./negative/.. %d.JPG\n", training_image);
            printf("==================================================\n");
            printf("\n");
            //return -1;
            exit(0);
        }
#endif // USE_IMAGE
        RGB_image.convertTo(colour, CV_32FC3, 1.0/255.0);
#ifdef USE_BGR_NORMALIZER
        local_norm_colour = CV_32FC3_local_normalizing(colour);
#else
        //   colour += Scalar(0.5,0.0,0.0);
        //    local_norm_colour = colour.clone();
        local_norm_colour = colour;
#endif // USE_BGR_NORMALIZER
        imshow("colour", colour);
        imshow("local_norm_colour", local_norm_colour);
        zero_ptr_norm_32FC3_img =        local_norm_colour.ptr<float>(0);
        index_ptr_norm_32FC3_img  =   zero_ptr_norm_32FC3_img;
        index_ptr_norm_B_32FC1img   =   zero_ptr_norm_B_32FC1img;
        index_ptr_norm_G_32FC1img   =   zero_ptr_norm_G_32FC1img;
        index_ptr_norm_R_32FC1img   =   zero_ptr_norm_R_32FC1img;
#ifdef USE_MNIST
        for (int i=0; i<MNIST_pix_size; i++)
#endif //USE_MNIST
#ifdef USE_CIFAR
            for (int i=0; i<CIFAR_RGB_pixels; i++)
#endif // USE_CIFAR
#ifdef USE_IMAGE



                for (int i=0; i<IMAGE_RGB_pixels; i++)
#endif // USE_IMAGE
                {
                    for(int BGR=0; BGR<3; BGR++)
                    {
                        ///Spit BGR to B, G, R separate 0..255 image
                        switch(BGR)
                        {
                        case(0):
                            *index_ptr_norm_B_32FC1img = *index_ptr_norm_32FC3_img;
                            index_ptr_norm_B_32FC1img++;
                            break;
                        case(1):
                            *index_ptr_norm_G_32FC1img = *index_ptr_norm_32FC3_img;
                            index_ptr_norm_G_32FC1img++;
                            break;
                        case(2):
                            *index_ptr_norm_R_32FC1img = *index_ptr_norm_32FC3_img;
                            index_ptr_norm_R_32FC1img++;
                            break;
                        }
                        index_ptr_norm_32FC3_img++;
                    }
                }
        imshow("norm_B_32FC1img", norm_B_32FC1img);
        imshow("norm_G_32FC1img", norm_G_32FC1img);
        imshow("norm_R_32FC1img", norm_R_32FC1img);

        ///randu(RGB_image, Scalar::all(0), Scalar::all(255));
        ///  cvtColor(RGB_image,RGB_image,CV_BGR2RGB);
        imshow("RGB_image", RGB_image);

#ifdef USE_STD_DEVIATION_FOR_NOISE_AMPLITUDE
///Make std_deviation of the data input to prepare a proper noise amplitude value
        if(do_std_dev_calc_counter<run_std_dev_at_counter_level)
        {
            do_std_dev_calc_counter++;
        }
        else
        {
            do_std_dev_calc_counter=0;
            if(comon_func_Obj1.L1_autoencoder_ON == 1)///Do Autoencoder Standard deviation calculation
            {
                rand_x_start_pos = (int) (rand() % (L1_conv_width-1));
                rand_y_start_pos = (int) (rand() % (L1_conv_hight-1));
                for(int n=0; n<FL1_size; n++)///comon_func_Obj1.nr_of_positions = FL1_size;///Size of the FL1_size = FL1_srt_size * FL1_srt_size * FL1_depth
                {
                    index_ptr_norm_32FC3_img = zero_ptr_norm_32FC3_img + norm_32FC3_img.cols * FL1_depth * (rand_y_start_pos + n/(FL1_srt_size * FL1_depth)) + rand_x_start_pos * FL1_depth + n%(FL1_srt_size * FL1_depth);
                    comon_func_Obj1.do_mean_add_std_dev(*index_ptr_norm_32FC3_img);///Insert real data from input vector input image
                }
                comon_func_Obj1.do_mean_calc_to_std_dev();///Do mean divition
                for(int n=0; n<FL1_size; n++)///comon_func_Obj1.nr_of_positions = FL1_size;///Size of the FL1_size = FL1_srt_size * FL1_srt_size * FL1_depth
                {
                    index_ptr_norm_32FC3_img = zero_ptr_norm_32FC3_img + norm_32FC3_img.cols * FL1_depth * (rand_y_start_pos + n/(FL1_srt_size * FL1_depth)) + rand_x_start_pos * FL1_depth + n%(FL1_srt_size * FL1_depth);
                    comon_func_Obj1.do_deviation_add_std_dev(*index_ptr_norm_32FC3_img);///Insert real data from input vector input image
                }
                comon_func_Obj1.do_std_deviation_calc();
                L1_autoencoder_noise_aplitude = comon_func_Obj1.std_deviation;
#ifdef PRINT_STD_DEV
                printf("L1_autoencoder_noise_aplitude %f\n", L1_autoencoder_noise_aplitude);
                printf("L1_autoencoder_ variance  %f\n", comon_func_Obj1.variance);
                printf("L1_autoencoder_ std_mean_value  %f\n", comon_func_Obj1.std_mean_value);
#endif
            }

            if(comon_func_Obj1.L2_autoencoder_ON == 1)///Do Autoencoder Standard deviation calculation
            {
                rand_x_start_pos = (int) (rand() % (L2_conv_width-1));///-1 because then it fit exactly between Feature and Convolution
                rand_y_start_pos = (int) (rand() % (L2_conv_hight-1));
                ///Get a patch vector from L1_pool_cube and run it thoue the autoencoder
                ///Insert noise on L2_patch_vect[][]
                float pool_temporary;
                //int addr_offset;
                for(int j=0; j<FL2_depth; j++)
                {
                    for(int i=0; i<(FL2_srt_size * FL2_srt_size); i++)///comon_func_Obj1.L2_nr_of_positions = FL2_size;///Size of the FL2_size = FL2_srt_size * FL2_srt_size * FL2_depth
                    {
                        //addr_offset = (L1_conv_width/pool_sqr) * (rand_y_start_pos + i/FL2_srt_size) + rand_x_start_pos + i%FL2_srt_size;
                        pool_temporary = L1_pool_cube[(L1_conv_width/pool_sqr) * (rand_y_start_pos + i/FL2_srt_size) + rand_x_start_pos + i%FL2_srt_size][j];
                        comon_func_Obj1.do_mean_add_std_dev(pool_temporary);///Insert real data from input vector input image
                    }
                }
                comon_func_Obj1.do_mean_calc_to_std_dev();///Do mean divition
                for(int j=0; j<FL2_depth; j++)
                {
                    for(int i=0; i<(FL2_srt_size * FL2_srt_size); i++)///comon_func_Obj1.L2_nr_of_positions = FL2_size;///Size of the FL2_size = FL2_srt_size * FL2_srt_size * FL2_depth
                    {
                        //addr_offset = (L1_conv_width/pool_sqr) * (rand_y_start_pos + i/FL2_srt_size) + rand_x_start_pos + i%FL2_srt_size;
                        pool_temporary = L1_pool_cube[(L1_conv_width/pool_sqr) * (rand_y_start_pos + i/FL2_srt_size) + rand_x_start_pos + i%FL2_srt_size][j];
                        comon_func_Obj1.do_deviation_add_std_dev(pool_temporary);///Insert real data from input vector input image
                    }
                }
                comon_func_Obj1.do_std_deviation_calc();
                L2_autoencoder_noise_aplitude = comon_func_Obj1.std_deviation;
#ifdef PRINT_STD_DEV
                printf("L2_autoencoder_noise_aplitude %f\n", L2_autoencoder_noise_aplitude);
                printf("L2_autoencoder_ variance  %f\n", comon_func_Obj1.variance);
                printf("L2_autoencoder_ std_mean_value  %f\n", comon_func_Obj1.std_mean_value);
#endif
            }
            if(comon_func_Obj1.L3_autoencoder_ON == 1)///Do Autoencoder Standard deviation calculation
            {
                rand_x_start_pos = (int) (rand() % (L3_conv_width-1));///-1 because then it fit exactly between Feature and Convolution
                rand_y_start_pos = (int) (rand() % (L3_conv_hight-1));
                ///Get a patch vector from L2_pool_cube and run it thoue the autoencoder
                ///Insert noise on L3_patch_vect[][]
                float pool_temporary;
                for(int j=0; j<FL3_depth; j++)
                {
                    for(int i=0; i<(FL3_srt_size * FL3_srt_size); i++)///comon_func_Obj1.L2_nr_of_positions = FL2_size;///Size of the FL2_size = FL2_srt_size * FL2_srt_size * FL2_depth
                    {
                        pool_temporary = L2_pool_cube[(L2_conv_width/pool_sqr) * (rand_y_start_pos + i/FL3_srt_size) + rand_x_start_pos + i%FL3_srt_size][j];
                        comon_func_Obj1.do_mean_add_std_dev(pool_temporary);///Insert real data from input vector input image
                    }
                }
                comon_func_Obj1.do_mean_calc_to_std_dev();///Do mean divition
                for(int j=0; j<FL3_depth; j++)
                {
                    for(int i=0; i<(FL3_srt_size * FL3_srt_size); i++)///comon_func_Obj1.L2_nr_of_positions = FL2_size;///Size of the FL2_size = FL2_srt_size * FL2_srt_size * FL2_depth
                    {
                        pool_temporary = L2_pool_cube[(L2_conv_width/pool_sqr) * (rand_y_start_pos + i/FL3_srt_size) + rand_x_start_pos + i%FL3_srt_size][j];
                        comon_func_Obj1.do_deviation_add_std_dev(pool_temporary);///Insert real data from input vector input image
                    }
                }
                comon_func_Obj1.do_std_deviation_calc();
                L3_autoencoder_noise_aplitude = comon_func_Obj1.std_deviation;
#ifdef PRINT_STD_DEV
                printf("L3_autoencoder_noise_aplitude %f\n", L3_autoencoder_noise_aplitude);
                printf("L3_autoencoder_ variance  %f\n", comon_func_Obj1.variance);
                printf("L3_autoencoder_ std_mean_value  %f\n", comon_func_Obj1.std_mean_value);
#endif
            }


        }
#endif // USE_STD_DEVIATION_FOR_NOISE_AMPLITUDE
        if(comon_func_Obj1.L1_autoencoder_ON == 1)///Do Autoencoder L1 process (Not L1 Convolution process)
        {
            ///******************************************
            ///********* Autoencoder L1 process *********
            ///******************************************
            for(int ittr=0; ittr<nr_of_autoenc_ittr_1_image; ittr++)
            {
                rand_x_start_pos = (int) (rand() % (L1_conv_width-1));
                rand_y_start_pos = (int) (rand() % (L1_conv_hight-1));
                comon_func_Obj1.rand_input_data_pos();///Make a table (comon_func_Obj1.nr_of_positions) of randomized position inside FL1 feature size
                for(int n=0; n<FL1_size; n++)///comon_func_Obj1.nr_of_positions = FL1_size;///Size of the FL1_size = FL1_srt_size * FL1_srt_size * FL1_depth
                {
                    index_ptr_L1_patch_img = zero_ptr_L1_patch_img + n;
                    index_ptr_L1_noise_img = zero_ptr_L1_noise_img + n;
                    index_ptr_norm_32FC3_img = zero_ptr_norm_32FC3_img + norm_32FC3_img.cols * FL1_depth * (rand_y_start_pos + n/(FL1_srt_size * FL1_depth)) + rand_x_start_pos * FL1_depth + n%(FL1_srt_size * FL1_depth);
                    *index_ptr_L1_patch_img = *index_ptr_norm_32FC3_img;///Insert real data from input vector input image
                    if(comon_func_Obj1.noise_pos[n] == 1 && comon_func_Obj1.started == 1)
                    {
                        ///****************************************************
                        ///********** Select noise level ******************
                        ///****************************************************
                        noise = (float) (rand() % 65535) / 65536;//0..1.0 range
                        noise -= 0.5f;
                        noise = noise * L1_autoencoder_noise_aplitude;
                        noise += 0.5f;
                        noise += noise_offset;

                        *index_ptr_L1_noise_img = noise;///Insert noise instead of real pixel value
                    }
                    else
                    {
                        ///Insert a small region of the input vector
                        *index_ptr_L1_noise_img = *index_ptr_norm_32FC3_img;///Insert real data from input vector input image
                    }
                }


                ///*********** Forward to hidden nodes ****************
                ///Make autoencoder forward of FL1 feature
                for(int j=0; j<L1_conv_depth; j++)
                {
                    ///L1_conv_cube[0][j] = 0.0f;///Clear autoencoder hidden node (hidden node = Lx_conv_cube[0][depth] ). [0] because we borrow L1_conv_cube mem this first area sheet pos in the convoution memory area now only for autencoder
                    L1_conv_cube[0][j] = FLx_bias_value * L1_weight_matrix_M[FL1_size][j];///Clear autoencoder hidden node with bias. Begin with the bias weight signal
                    index_ptr_L1_noise_img = zero_ptr_L1_noise_img;
                    for(int i=0; i<FL1_size; i++)
                    {
                        L1_conv_cube[0][j] += *index_ptr_L1_noise_img * L1_weight_matrix_M[i][j];///Make the dot product to pruduce the node.[0] because we borrow L1_conv_cube mem this first area sheet pos in the convoution memory area now only for autencoder
                        index_ptr_L1_noise_img++;
                    }
                    ///*** Relu this node ***
                    if(L1_conv_cube[0][j] < 0.0f)
                    {
                        L1_conv_cube[0][j] = L1_conv_cube[0][j] * Relu_neg_gain;///Relu function
                    }
                }
                ///********** Forward to L1 output nodes *******************
                ///Clear the L1_autoenc_reconstructed
                index_ptr_L1_autoenc_reconstructed = zero_ptr_L1_autoenc_reconstructed;
                for(int i=0; i<FL1_size; i++)
                {
                    *index_ptr_L1_autoenc_reconstructed = 0.5f;///Clear
                    index_ptr_L1_autoenc_reconstructed++;
                }
                ///Make autoencoder reconstruction
                FLx_bias_reconstruction =0.0f;///Clear
                for(int j=0; j<L1_conv_depth; j++)
                {
                    FLx_bias_reconstruction += FLx_bias_value * L1_weight_matrix_M[FL1_size][j];///Start with adding the Bias. [FL1_size][] is the bias weight
                    index_ptr_L1_autoenc_reconstructed = zero_ptr_L1_autoenc_reconstructed;
                    for(int i=0; i<FL1_size; i++)
                    {
                        *index_ptr_L1_autoenc_reconstructed += L1_conv_cube[0][j] * L1_weight_matrix_M[i][j];///Reconstruction using the autoencoder tie weight. L1_conv_cube[0][j] is the hidden node of the autoencoder
                        index_ptr_L1_autoenc_reconstructed++;
                    }
                }
                ///Make the autoencoder loss and delta calculation
                index_ptr_L1_autoenc_delta = zero_ptr_L1_autoenc_delta;
                index_ptr_L1_autoenc_reconstructed = zero_ptr_L1_autoenc_reconstructed;
                index_ptr_L1_patch_img = zero_ptr_L1_patch_img;

                bias_delta=0.0f;
                ///loss=0.0f;
                float L1_output_node;
                bias_delta = FLx_bias_value - FLx_bias_reconstruction;///Start loss calculation with the bias.
                loss += bias_delta * bias_delta;/// loss = 1/2 SUM k (input[k] - ouput[k])²
                for(int i=0; i<FL1_size; i++)
                {
                    L1_output_node = *index_ptr_L1_autoenc_reconstructed;
                    delta = *index_ptr_L1_patch_img - L1_output_node;
                    loss += delta * delta;/// loss = 1/2 SUM k (input[k] - ouput[k])²

                    ///  if(L1_output_node < 0.0f)///
                    ///  {
                    ///      delta *= Relu_neg_gain;///??????
                    ///  }
                    *index_ptr_L1_autoenc_delta = delta;
                    index_ptr_L1_autoenc_reconstructed++;
                    index_ptr_L1_autoenc_delta++;
                    index_ptr_L1_patch_img++;
                }

                index_ptr_L1_autoenc_delta = zero_ptr_L1_autoenc_delta;
                if(comon_func_Obj1.started == 1)
                {
                    ///make the backpropagation
                    ///First update the bias weights
                    for(int j=0; j<L1_conv_depth; j++)
                    {
                        /// **** update tied weight regarding delta
                        L1_change_weight_M[FL1_size][j] = L1_LearningRate * L1_conv_cube[0][j] * bias_delta + L1_Momentum * L1_change_weight_M[FL1_size][j];///hidden_node = L1_conv_cube[0][j];
                        L1_weight_matrix_M[FL1_size][j] += L1_change_weight_M[FL1_size][j];
                    }
                    for(int i=0; i<FL1_size; i++)///
                    {
                        for(int j=0; j<L1_conv_depth; j++)
                        {
                            /// **** update tied weight regarding delta
                            L1_change_weight_M[i][j] = L1_LearningRate * L1_conv_cube[0][j] * (*index_ptr_L1_autoenc_delta) + L1_Momentum * L1_change_weight_M[i][j];///hidden_node = L1_conv_cube[0][j];
                            L1_weight_matrix_M[i][j] += L1_change_weight_M[i][j];
                        }
                        index_ptr_L1_autoenc_delta++;
                    }
                }
                ///Print Loss
                static int print_loss=0;
                if(print_loss<200)
                {
                    print_loss++;
                }
                else
                {
                    loss = loss / 2.0f;/// loss = 1/2 SUM k (input[k] - ouput[k])²
                    printf("L1 autoencoder loss error = %f\n", loss);
                    print_loss=0;
                    loss=0.0f;
                    if(auto_save_ON==1)
                    {
                        ittr_counter_L1_autosave++;
                        if(ittr_counter_L1_autosave>ittr_before_L1_autosave)
                        {
                            ittr_counter_L1_autosave=0;
                            auto_save_counter++;
                        }

                    }
                }
                ///End print loss
                ///L1 visual
                L1_visual_all_feature = Scalar(0.0,0.0,0.0);
                for(int i=0; i<FL1_size; i++)
                {
                    L1_ptr_M_matrix = &L1_weight_matrix_M[i][0];
                    attach_weight_2_mat(L1_ptr_M_matrix, i, L1_visual_all_feature, L1_sqr_of_H_nod_plus1, L1_conv_depth, FL1_srt_size, FL1_srt_size);///
                }
                L1_visual_all_feature += Scalar(0.5,0.5,0.5);
                imshow("L1_visual_all_feature", L1_visual_all_feature);
                ///L1_autoenc_delta += Scalar(0.5,0.5,0.5);
                imshow("L1_patch_img", L1_patch_img);
                imshow("L1_noise_img", L1_noise_img);
                imshow("L1_autoenc_reconstructed", L1_autoenc_reconstructed);
                imshow("L1_autoenc_delta", L1_autoenc_delta);
                waitKey(1);
            }
        }
        if(comon_func_Obj1.L1_convolution_ON == 1)/// Do convloution L1 process instead of L1 Autoencoder process
        {
            if(comon_func_Obj1.started == 0)
            {
#ifdef USE_MNIST
                //        printf("MNIST_lable[MNIST_nr] %d\n", MNIST_lable[MNIST_nr]);
                //        printf("MNIST_nr =%d\n", MNIST_nr);
#endif // USE_MNIST
#ifdef USE_CIFAR
                printf("Lable nr = %d\n", CIFAR_data[(CIFAR_nr*CIFAR_row_size)]);
                printf("CIFAR_nr =%d\n", CIFAR_nr);
#endif // USE_CIFAR
#ifdef USE_IMAGE

                resize(colour, rezied_img, size);

                imshow("rezied_img", rezied_img);


                printf("Lable nr =%d\n", toggle_pos_neg);
                printf("training_image =%d\n", training_image);
#endif // USE_IMAGE
                if(comon_func_Obj1.full_conn_backprop==1)
                {
                    waitKey(10);
                }
                else
                {
                    ///             waitKey(3000);
                }

            }
            else
            {
                waitKey(10);
            }

            ///**********************************
            ///********* Convolute L1 ***********
            ///**********************************
            for(int i=0; i<(L1_conv_hight * L1_conv_width); i++)///This loop step throue (convolute) the area of the input "sheet". No padding of the slide on one sheet area of the convolution cube
            {
                for(int j=0; j<L1_conv_depth; j++)///This loop step throue the depth of the Convolution L1 cube
                {
                    ///L1_conv_cube[i][j] = 0.0f;///Clear the one node of the conv cube
                    L1_conv_cube[i][j] = L1_weight_matrix_M[FL1_size][j] * FLx_bias_value;///Clear with bias the one node of the conv cube
                    int input_ptr_offset;
                    for(int k=0; k<FL1_size; k++)///This loop step throue one Feature cube. FL1_size = (FL1_srt_size * FL1_srt_size * FL1_depth)  The size of one Feature FL1
                    {
                        input_ptr_offset = (local_norm_colour.cols * local_norm_colour.channels() * (L1_stride*(i/L1_conv_width) + (k/(FL1_srt_size * FL1_depth)))) + (L1_stride*(i%(L1_conv_width)) * FL1_depth) + (k%(FL1_srt_size * FL1_depth));
                        index_ptr_norm_32FC3_img = zero_ptr_norm_32FC3_img + input_ptr_offset;
                        ///Convolution cube row address = (local_norm_colour.cols * local_norm_colour.channels() * ((i/L1_conv_width) + (k/(FL1_srt_size * FL1_depth))))
                        ///where local_norm_colour.cols is the width if the conv

                        ///Make the dot product
                        ///ConvNode = ConvNode + weigh[][] * input[]
                        ///L1_conv_cube[i][j] += L1_weight_matrix_M[k][j] * ((1.0f/255.0) * (float) (*index_ptr_norm_BGR_img));///Make the dot product of the 0..x Feature * input image to L1 convolution cube
                        L1_conv_cube[i][j] += L1_weight_matrix_M[k][j] * (*index_ptr_norm_32FC3_img);///Make the dot product of the 0..x Feature * input image to L1 convolution cube
                    }
                }
            }
            ///**************************************
            ///********* End Convolute L1 ***********
            ///**************************************
            ///**************************************
            ///********* Pooling L1 *****************
            ///**************************************
            for(int h=0; h<(L1_conv_hight / pool_sqr); h++)///This loop togheter with i loop step throue the "sheet" area of of the L1_pool_cube[i][x]
            {
                for(int i=0; i<(L1_conv_width / pool_sqr); i++)///This loop togheter with h loop step throue the "sheet" area of of the L1_pool_cube[i][x]
                {
                    if(comon_func_Obj1.L1_autoencoder_ON == 1)
                    {
                        L1_autoenc_reconstructed = Scalar(0.0f,0.0f,0.0f);
                    }
                    for(int j=0; j<L1_conv_depth; j++)///This loop step throue the depth of the Convolution L1 cube
                    {
                        float max_node= -1000000.0f;
                        float compare_max = 0.0f;
                        int revers_pool_row=0;
                        for(int p=0; p<pooling; p++)
                        {
                            revers_pool_row = p/pool_sqr;///Get a pool row level inside the pooling area to make it poosible to read the right row position on the convolution sheet area
                            ///compare_max = L1_conv_cube[h * pool_sqr * L1_conv_width + L1_conv_width * revers_pool_row + i * pool_sqr + p % pool_sqr][j];///Prepare pooling compare value. Pick the value from the convolute (slide) position node at convolution cube
                            compare_max = L1_conv_cube[h * pool_sqr * L1_conv_width + L1_conv_width * revers_pool_row + i * pool_sqr + p%pool_sqr][j];///Prepare pooling compare value. Pick the value from the convolute (slide) position node at convolution cube
                            if(compare_max > max_node)
                            {
                                L1_pool_tracking = p;
                                max_node = compare_max;
                            }
                        }
                        ///*** Relu this node ***
                        if(max_node < 0.0f)
                        {
                            max_node = max_node * Relu_neg_gain;///Relu function
                        }
                        L1_pool_cube[h*(L1_conv_hight / pool_sqr) + i][j] = max_node;
                        ///****************************************************************************************************************
                        ///******** Do the L1 autoencoder here when the fully depth of the pooled nodes on L1_pool_cube[][] is done *******
                        ///****************************************************************************************************************
                        if(comon_func_Obj1.L1_autoencoder_ON == 1)
                        {
                            ///3-steps to make a delta node "pixel" to each feature. Unsupervised learning
                            ///1. Make the reconstruction vector from the tie feature weights from pool_cube node. If feature was 5x5 then reconstruction vector is 6x6 when pooling=4, 7x7 if pooling=9
                            ///2. Make the input (without noise) compare vector L1_patch_img. If feature was 5x5 then input compare vector is 6x6 when pooling=4, 7x7 if pooling=9
                            ///3. Make delta vector from differance input vector - reconstruction vector.
                            ///4. Update L1 features tie weight with respect to delta vector and this slide positions in the pool_cube.
                            for(int k=0; k<FL1_size; k++) ///Go throue the Feature cube FL1_size = (FL1_srt_size * FL1_srt_size * FL1_depth) of the input features so the reconstruction vector will have same depth as the input depth
                            {
                                ///Step 1. Make the reconstruction vector from the tie feature weights from pool_cube node. If feature was 5x5 then reconstruction vector is 6x6 when pooling=4, 7x7 if pooling=9
                                index_ptr_L1_autoenc_reconstructed = zero_ptr_L1_autoenc_reconstructed + (L1_autoenc_reconstructed.cols * FL1_depth * ((L1_pool_tracking / pool_sqr) + k/(FL1_srt_size * FL1_depth)) + FL1_depth * (L1_pool_tracking % pool_sqr) + k%(FL1_srt_size * FL1_depth));
                                ///********** Forward to L1 (output nodes) reconstruction node *******************
                                *index_ptr_L1_autoenc_reconstructed += max_node * L1_weight_matrix_M[k][j];///revers the dot product, reconstruct the orginal vector
                                ///End Step 1.
                            }
                        }
                    }
                    /// Pooling of this convolute (slide) position is now done throue the full depth of nodes at L1_conv_cube[][] and L1_pool_cube[h*i][j]
                    /// h, i, pooltracking gives the convolute (slide) position to prepare the input patch/part for the autoencoder compare to reconstruction
                }
            }
            ///**************************************
            ///********* END Pooling L1 *************
            ///**************************************
///show one layer of conv L1
///void show_sheet_2_mat(float* cube, int show_layer_nr, Mat dst)
            test1_ptr = test1_zero_ptr;
            test2_ptr = test2_zero_ptr;
            test3_ptr = test3_zero_ptr;

            for(int i=0; i<L1_conv_hight*L1_conv_width; i++)
            {
                *test1_ptr = L1_conv_cube[i][0];
                test1_ptr++;
                *test2_ptr = L1_conv_cube[i][1];
                test2_ptr++;
                *test3_ptr = L1_conv_cube[i][2];
                test3_ptr++;
            }
            //  printf("L1_conv_hight*L1_conv_width %d\n", L1_conv_hight*L1_conv_width);
            //  printf("test.cols*test.channels() * test.rows %d\n", test.cols*test.channels() * test.rows);
            test1 += 0.5f;///Scalar(0.5);
            test2 += 0.5f;///Scalar(0.5);
            test3 += 0.5f;///Scalar(0.5);

            imshow("test1", test1);
            imshow("test2", test2);
            imshow("test3", test3);

            pol_t1_ptr = pol_t1_zero_ptr;
            pol_t2_ptr = pol_t2_zero_ptr;
            pol_t3_ptr = pol_t3_zero_ptr;

            for(int i=0; i<((L1_conv_hight/pool_sqr)*(L1_conv_width/pool_sqr)); i++)
            {

                *pol_t1_ptr = L1_pool_cube[i][0];
                pol_t1_ptr++;
                *pol_t2_ptr = L1_pool_cube[i][1];
                pol_t2_ptr++;
                *pol_t3_ptr = L1_pool_cube[i][2];
                pol_t3_ptr++;

            }
            pol_t1 += 0.5f;
            pol_t2 += 0.5f;
            pol_t3 += 0.5f;
            imshow("pol_t1", pol_t1);
            imshow("pol_t2", pol_t2);
            imshow("pol_t3", pol_t3);

        } ///End convloution L1 process

        ///L2
        if(comon_func_Obj1.L2_autoencoder_ON == 1)///Do Autoencoder L2 process (Not L2 Convolution process)
        {
            ///******************************************
            ///********* Autoencoder L2 process *********
            ///******************************************
            for(int ittr=0; ittr<L2_nr_of_autoenc_ittr_1_image; ittr++)
            {
                rand_x_start_pos = (int) (rand() % (L2_conv_width-1));///-1 because then it fit exactly between Feature and Convolution
                rand_y_start_pos = (int) (rand() % (L2_conv_hight-1));
                ///         printf("rand_x_start_pos %d\n", rand_x_start_pos);
                comon_func_Obj1.L2_rand_input_data_pos();///Make a table (comon_func_Obj1.L2_nr_of_positions) of randomized position inside FL2 feature size

                ///Get a patch vector from L1_pool_cube and run it thoue the autoencoder
                ///Insert noise on L2_patch_vect[][]
                float pool_temporary;
                //int addr_offset;
                for(int j=0; j<FL2_depth; j++)
                {
                    for(int i=0; i<(FL2_srt_size * FL2_srt_size); i++)///comon_func_Obj1.L2_nr_of_positions = FL2_size;///Size of the FL2_size = FL2_srt_size * FL2_srt_size * FL2_depth
                    {
                        //addr_offset = (L1_conv_width/pool_sqr) * (rand_y_start_pos + i/FL2_srt_size) + rand_x_start_pos + i%FL2_srt_size;
                        pool_temporary = L1_pool_cube[(L1_conv_width/pool_sqr) * (rand_y_start_pos + i/FL2_srt_size) + rand_x_start_pos + i%FL2_srt_size][j];
                        ///L2_patch_vect[area][depth]
                        ///pool_temporary = L1_pool_cube[addr_offset][j];
                        L2_patch_vect[i][j] = pool_temporary;///Insert real data from input vector
                        if(comon_func_Obj1.L2_noise_pos[i + (FL2_srt_size * FL2_srt_size)*j] == 1)
                        {
                            ///****************************************************
                            ///********** Select noise level ******************
                            ///****************************************************
                            noise = (float) (rand() % 65535) / 65536;//0..1.0 range
                            noise -= 0.5f;
                            noise = noise * L2_autoencoder_noise_aplitude;
                            noise += 0.5f;
                            noise += noise_offset;

                            L2_noise_vect[i][j] = noise;///Insert noise instead of real value
                        }
                        else
                        {
                            ///Insert a small region of the input vector
                            L2_noise_vect[i][j] = pool_temporary;///Insert real data from input vector
                        }
                    }
                }

                ///*********** Forward to hidden nodes ****************
                ///Make autoencoder forward of FL2 feature
                for(int k=0; k<L2_conv_depth; k++)///FL3_depth = L2_conv_depth
                {
                    ///L2_conv_cube[0][k] = 0.0f;///Clear autoencoder hidden node (hidden node = Lx_conv_cube[0][depth] ). [0] because we borrow L1_conv_cube mem this first area sheet pos in the convoution memory area now only for autencoder
                    L2_conv_cube[0][k] = FLx_bias_value * L2_weight_matrix_M[FL2_size][k];///Clear autoencoder hidden node with bias. Begin with the bias weight signal
                    for(int j=0; j<FL2_depth; j++)
                    {
                        for(int i=0; i<(FL2_srt_size * FL2_srt_size); i++)
                        {
                            L2_conv_cube[0][k] += L2_noise_vect[i][j] * L2_weight_matrix_M[i + (FL2_srt_size * FL2_srt_size)*j][k];///Make the dot product to pruduce the node.[0] because we borrow L1_conv_cube mem this first area sheet pos in the convoution memory area now only for autencoder
                        }
                        ///*** Relu this node ***
                        if(L2_conv_cube[0][k] < 0.0f)
                        {
                            L2_conv_cube[0][k] = L2_conv_cube[0][k] * Relu_neg_gain;///Relu function
                        }
                    }
                }

                ///********** Forward to L2 output nodes *******************
                ///Clear the L2_autoenc_reconstructed
                for(int j=0; j<FL2_depth; j++)
                {
                    for(int i=0; i<(FL2_srt_size * FL2_srt_size); i++)
                    {
                        L2_autoenc_reconstructed[i][j] = 0.0f;///Clear
                    }
                }

                ///Make autoencoder reconstruction
                FLx_bias_reconstruction =0.0f;///Clear
                for(int k=0; k<L2_conv_depth; k++)
                {
                    FLx_bias_reconstruction += FLx_bias_value * L2_weight_matrix_M[FL2_size][k];///Start with adding the Bias. [FL1_size][] is the bias weight
                    for(int j=0; j<FL2_depth; j++)
                    {
                        for(int i=0; i<(FL2_srt_size * FL2_srt_size); i++)
                        {
                            ///Make autoencoder reconstruction
                            L2_autoenc_reconstructed[i][j] += L2_conv_cube[0][k] * L2_weight_matrix_M[i  + (FL2_srt_size * FL2_srt_size)*j][k];///Reconstruction using the autoencoder tie weight. L1_conv_cube[0][j] is the hidden node of the autoencoder
                        }
                    }
                }
                ///Make the autoencoder loss and delta calculation
                static float L2_loss=0.0f;
                float L2_delta=0.0f;
                float L2_output_node=0.0f;
                bias_delta=0.0f;
                ///L2_loss=0.0f;
                bias_delta = FLx_bias_value - FLx_bias_reconstruction;///Start loss calculation with the bias.
                L2_loss += bias_delta * bias_delta;/// loss = 1/2 SUM k (input[k] - ouput[k])²
                for(int j=0; j<FL2_depth; j++)
                {
                    for(int i=0; i<(FL2_srt_size * FL2_srt_size); i++)
                    {
                        //L2_delta = *index_ptr_L1_patch_img - *index_ptr_L1_autoenc_reconstructed;
                        L2_output_node = L2_autoenc_reconstructed[i][j];
                        L2_delta = L2_patch_vect[i][j] - L2_output_node ;
                        L2_loss += L2_delta * L2_delta;/// loss = 1/2 SUM k (input[k] - ouput[k])²
                        ///    if(L2_output_node  < 0.0f)///
                        ///    {
                        ///        L2_delta *= Relu_neg_gain;
                        ///    }
                        L2_autoenc_delta[i][j] = L2_delta;
                    }
                }
                if(comon_func_Obj1.started == 1)
                {
                    ///make the backpropagation
                    ///First update the bias weights
                    for(int k=0; k<L2_conv_depth; k++)
                    {
                        /// **** update tied weight regarding delta
                        L2_change_weight_M[FL2_size][k] = L2_LearningRate * L2_conv_cube[0][k] * bias_delta + L2_Momentum * L2_change_weight_M[FL2_size][k];///hidden_node = L1_conv_cube[0][j];
                        L2_weight_matrix_M[FL2_size][k] += L2_change_weight_M[FL2_size][k];
                    }
                    for(int k=0; k<L2_conv_depth; k++)
                    {
                        for(int j=0; j<FL2_depth; j++)
                        {
                            for(int i=0; i<(FL2_srt_size * FL2_srt_size); i++)
                            {
                                /// **** update tied weight regarding delta
                                L2_change_weight_M[i + (FL2_srt_size * FL2_srt_size)*j][k] = L2_LearningRate * L2_conv_cube[0][k] * (L2_autoenc_delta[i][j]) + L2_Momentum * L2_change_weight_M[i + (FL2_srt_size * FL2_srt_size)*j][k];///hidden_node = L1_conv_cube[0][j];
                                L2_weight_matrix_M[i + (FL2_srt_size * FL2_srt_size)*j][k] += L2_change_weight_M[i + (FL2_srt_size * FL2_srt_size)*j][k];
                            }
                        }
                    }
                }
                ///Print Loss
                static int L2_print_loss=0;
                if(L2_print_loss<200)
                {
                    L2_print_loss++;
                }
                else
                {
                    L2_loss = L2_loss / 2.0f;/// loss = 1/2 SUM k (input[k] - ouput[k])²
                    printf("L2 autoencoder L2_loss error = %f\n", L2_loss);
                    L2_print_loss=0;
                    L2_loss=0.0f;
                    if(auto_save_ON==1)
                    {
                        auto_save_counter++;
                    }

                }
                ///End print loss
            }///End ittr loop
        }
        ///**************** End L2 Autoencoder *****************

        ///********** Show L2 features ************************
        if(comon_func_Obj1.finetune==1 || comon_func_Obj1.L2_autoencoder_ON == 1)
        {
            ///L2 visual
            static int visualL2_now_counter=1000;
            if(visualL2_now_counter > 100)
            {
                visualL2_now_counter=0;
                printf("show L2\n");
                L2_attach_weight2mat.Lx_src = L2_visual_all_feature;///attach Mat pointer
                for(int j=0; j<FL2_depth; j++)
                {
                    for(int i=0; i<(FL2_srt_size * FL2_srt_size); i++)
                    {
                        L2_attach_weight2mat.Lx_ptr_M_matrix = &L2_weight_matrix_M[i + (FL2_srt_size * FL2_srt_size)*j][0];
                        /// L2_attach_weight2mat.k = k;
                        L2_attach_weight2mat.FLx_i_location_area  = i;
                        L2_attach_weight2mat.FLx_j_location_depth = j;
                        L2_attach_weight2mat.Xattach_weight2mat();
                    }
                }
                L2_visual_all_feature += 0.5f;
                imshow("L2_visual_all_feature", L2_visual_all_feature);
                waitKey(1);
            }///Visual now end
            else
            {
                visualL2_now_counter++;
            }

            ///Show L2toL1
            static int visuL2L1_now_counter=1000;
            if(visuL2L1_now_counter>50)
            {
                visuL2L1_now_counter=0;
                sq_visualize_L2toL1 = Scalar(0.5,0.5,0.5);
                float* zero_ptr_sq_visualize_L2toL1 = sq_visualize_L2toL1.ptr<float>(0);
                float* index_ptr_sq_visualize_L2toL1 = sq_visualize_L2toL1.ptr<float>(0);

                int temp_offset=0;
                int sq_temp_offset=0;

                int show_node_patch_width_C1 = L2toL1_feature_reverse_size;
                int show_node_patch_hight_C1 = L2toL1_feature_reverse_size;
                ///Show now L2toL1
                for(int k=0; k<L2_conv_depth; k++)
                {
                    for(int j2=0; j2<FL2_depth; j2++) ///L1_conv_depth = FL2_depth
                    {
                        for(int i2=0; i2<FL2_srt_size*FL2_srt_size; i2++)
                        {
                            for(int j1=0; j1<FL1_depth; j1++)
                            {
                                for(int i1=0; i1<FL1_srt_size*FL1_srt_size; i1++)
                                {
                                    sq_temp_offset = show_node_patch_width_C1*FL1_depth*(k%sqr_L2_conv_depth_plus1) + sq_visualize_L2toL1.cols*FL1_depth*show_node_patch_hight_C1*(k/sqr_L2_conv_depth_plus1);///FL1_depth is the 3 colour. Add the start left corner colum offset for the show of L2 hidden node patch
                                    sq_temp_offset += (i2%FL2_srt_size)*FL1_depth*pool_sqr*L1_stride;///horizontal inc pos regarding horiz inc on FL2 horiz pos. FL1_depth is the 3 colour.
                                    sq_temp_offset += (i1%FL1_srt_size)*FL1_depth + j1;///horizontal inc pos regarding horiz inc on FL1 horiz pos. FL1_depth is the 3 colour.
                                    sq_temp_offset += sq_visualize_L2toL1.cols*FL1_depth*(i2/FL2_srt_size)*pool_sqr*L1_stride;///Vertical inc pos regarding FL2 vertical pos. FL1_depth is the 3 colour.
                                    sq_temp_offset += sq_visualize_L2toL1.cols*FL1_depth*(i1/FL1_srt_size);///Vertical inc pos regarding FL1 vertical pos. FL1_depth is the 3 colour.
                                    index_ptr_sq_visualize_L2toL1 = zero_ptr_sq_visualize_L2toL1 + sq_temp_offset;
                                    *index_ptr_sq_visualize_L2toL1 += L1_weight_matrix_M[j1 + i1*FL1_depth][j2] * L2_weight_matrix_M[i2 + (FL2_srt_size * FL2_srt_size)*j2][k];
                                }
                            }
                        }
                    }
                }
                /// imshow("visualize_L2toL1", visualize_L2toL1);
                imshow("sq_visualize_L2toL1", sq_visualize_L2toL1);
                Mat cloned_sq_visualize_L2toL1;
                cloned_sq_visualize_L2toL1 = sq_visualize_L2toL1.clone();
                cloned_sq_visualize_L2toL1 *= 255;
                cv::imwrite("cloned_sq_visualize_L2toL1.JPG",cloned_sq_visualize_L2toL1);

            }
            else
            {
                visuL2L1_now_counter++;
            }
            ///End L2tL1 show
        }
        ///********** END Show L2 features ************************

        ///**************** Begin L2 Convloution *****************
        if(comon_func_Obj1.L2_convolution_ON == 1)/// Do convloution L2 process instead of L2 Autoencoder process
        {
            ///**********************************
            ///********* Convolute L2 ***********
            ///**********************************
            for(int i=0; i<(L2_conv_hight * L2_conv_width); i++)///This loop step throue (convolute) the area of the input "sheet". No padding of the slide on one sheet area of the convolution cube
            {
                for(int j=0; j<L2_conv_depth; j++)///This loop step throue the depth of the Convolution L1 cube
                {
                    ///L2_conv_cube[i][j] = 0.0f;///Clear the one node of the conv cube
                    L2_conv_cube[i][j] = L2_weight_matrix_M[FL2_size][j] * FLx_bias_value;///Clear with bias the one node of the conv cube
                    //int input_ptr_offset;
                    float input_node;
                    for(int k=0; k<FL2_size; k++)///This loop step throue one Feature cube. FL2_size = (FL2_srt_size * FL2_srt_size * FL2_depth)  The size of one Feature FL2
                    {
                        ///L1_pool_cube[area][depth]
                        input_node = L1_pool_cube[(L1_conv_hight/pool_sqr)*(L2_stride*(i/L2_conv_width) + (k/(FL2_srt_size * FL2_depth))) + L2_stride*(i%L2_conv_width) + (k/FL2_depth)%FL2_srt_size][k%FL2_depth];
                        L2_conv_cube[i][j] += L2_weight_matrix_M[k][j] * input_node;///Make the dot product of the 0..x Feature * input image to L1 convolution cube
                    }
                }
            }
            ///**************************************
            ///********* End Convolute L2 ***********
            ///**************************************
            ///**************************************
            ///********* Pooling L2 *****************
            ///**************************************
            if(!(connect_fc_to_pool_or_conv == 1 && connect_fc_to_layer== 2))///Skip poolining if fc connected to conv cube on this layer
            {
                for(int h=0; h<(L2_conv_hight / pool_sqr); h++)///This loop togheter with i loop step throue the "sheet" area of of the L1_pool_cube[i][x]
                {
                    for(int i=0; i<(L2_conv_width / pool_sqr); i++)///This loop togheter with h loop step throue the "sheet" area of of the L1_pool_cube[i][x]
                    {
                        for(int j=0; j<L2_conv_depth; j++)///This loop step throue the depth of the Convolution L1 cube
                        {
                            float max_node= -1000000.0f;
                            float compare_max = 0.0f;
                            int revers_pool_row=0;
                            for(int p=0; p<pooling; p++)
                            {
                                revers_pool_row = p/pool_sqr;///Get a pool row level inside the pooling area to make it poosible to read the right row position on the convolution sheet area
                                ///compare_max = L1_conv_cube[h * pool_sqr * L1_conv_width + L1_conv_width * revers_pool_row + i * pool_sqr + p % pool_sqr][j];///Prepare pooling compare value. Pick the value from the convolute (slide) position node at convolution cube
                                compare_max = L2_conv_cube[h * pool_sqr * L2_conv_width + L2_conv_width * revers_pool_row + i * pool_sqr + p%pool_sqr][j];///Prepare pooling compare value. Pick the value from the convolute (slide) position node at convolution cube
                                if(compare_max > max_node)
                                {
                                    L2_pool_tracking = p;
                                    max_node = compare_max;
                                }
                                if(comon_func_Obj1.finetune==1)
                                {
                                    ///Save L2_pool_tracking for fine tuning later in fc backprop code
                                    L2_pool_track_cube[(h*(L2_conv_width / pool_sqr)) + i][j] = L2_pool_tracking;///L2_pool_track_cube[sheet area][depth] used later in code when update L2 weight with backprop from fc
                                }
                            }
                            ///*** Relu this node ***
                            if(max_node < 0.0f)
                            {
                                max_node = max_node * Relu_neg_gain;///Relu function
                            }
                            L2_pool_cube[h*(L2_conv_hight / pool_sqr) + i][j] = max_node;
                        }
                        /// Pooling of this convolute (slide) position is now done throue the full depth of nodes at L1_conv_cube[][] and L1_pool_cube[h*i][j]
                        /// h, i, pooltracking gives the convolute (slide) position to prepare the input patch/part for the autoencoder compare to reconstruction
                    }
                }
            }
            ///**************************************
            ///********* END Pooling L2 *************
            ///**************************************
///show one layer of conv L2
///void show_sheet_2_mat(float* cube, int show_layer_nr, Mat dst)
            L2_test1_ptr = L2_test1_zero_ptr;
            L2_test2_ptr = L2_test2_zero_ptr;
            L2_test3_ptr = L2_test3_zero_ptr;

            for(int i=0; i<L2_conv_hight*L2_conv_width; i++)
            {
                *L2_test1_ptr = L2_conv_cube[i][0];
                L2_test1_ptr++;
                *L2_test2_ptr = L2_conv_cube[i][1];
                L2_test2_ptr++;
                *L2_test3_ptr = L2_conv_cube[i][2];
                L2_test3_ptr++;
            }
            //  printf("L1_conv_hight*L1_conv_width %d\n", L1_conv_hight*L1_conv_width);
            //  printf("test.cols*test.channels() * test.rows %d\n", test.cols*test.channels() * test.rows);
            L2_test1 += 0.5f;///Scalar(0.5);
            L2_test2 += 0.5f;///Scalar(0.5);
            L2_test3 += 0.5f;///Scalar(0.5);

            imshow("L2_test1", L2_test1);
            imshow("L2_test2", L2_test2);
            imshow("L2_test3", L2_test3);

            L2_pol_t1_ptr = L2_pol_t1_zero_ptr;
            L2_pol_t2_ptr = L2_pol_t2_zero_ptr;
            L2_pol_t3_ptr = L2_pol_t3_zero_ptr;

            for(int i=0; i<((L2_conv_hight/pool_sqr)*(L2_conv_width/pool_sqr)); i++)
            {

                *L2_pol_t1_ptr = L2_pool_cube[i][0];
                L2_pol_t1_ptr++;
                *L2_pol_t2_ptr = L2_pool_cube[i][1];
                L2_pol_t2_ptr++;
                *L2_pol_t3_ptr = L2_pool_cube[i][2];
                L2_pol_t3_ptr++;

            }
            L2_pol_t1 += 0.5f;
            L2_pol_t2 += 0.5f;
            L2_pol_t3 += 0.5f;
            imshow("L2_pol_t1", L2_pol_t1);
            imshow("L2_pol_t2", L2_pol_t2);
            imshow("L2_pol_t3", L2_pol_t3);

        } ///End convloution L2 process


        ///L3
        int do_noise=0;
        if(comon_func_Obj1.L3_autoencoder_ON == 1)///Do Autoencoder L3 process (Not L3 Convolution process)
        {
            ///******************************************
            ///********* Autoencoder L3 process *********
            ///******************************************
            for(int ittr=0; ittr<L3_nr_of_autoenc_ittr_1_image; ittr++)
            {
                rand_x_start_pos = (int) (rand() % (L3_conv_width-1));///-1 because then it fit exactly between Feature and Convolution
                rand_y_start_pos = (int) (rand() % (L3_conv_hight-1));
                ///Get a patch vector from L2_pool_cube and run it thoue the autoencoder
                ///Insert noise on L3_patch_vect[][]
                float pool_temporary;
                //int addr_offset;
                int test=0;

                for(int j=0; j<FL3_depth; j++)
                {
                    for(int i=0; i<(FL3_srt_size * FL3_srt_size); i++)///comon_func_Obj1.L2_nr_of_positions = FL2_size;///Size of the FL2_size = FL2_srt_size * FL2_srt_size * FL2_depth
                    {
                        pool_temporary = L2_pool_cube[(L2_conv_width/pool_sqr) * (rand_y_start_pos + i/FL3_srt_size) + rand_x_start_pos + i%FL3_srt_size][j];
                        ///L3_patch_vect[area][depth]
                        L3_patch_vect[i][j] = pool_temporary;///Insert real data from input vector
                        if(comon_func_Obj1.L3_noise_pos[i + (FL3_srt_size * FL3_srt_size)*j] == 1 && do_noise == 1)
                        {
                            ///****************************************************
                            ///********** Select noise level ******************
                            ///****************************************************
                            noise = (float) (rand() % 65535) / 65536;//0..1.0 range
                            noise -= 0.5f;
                            noise = noise * L3_autoencoder_noise_aplitude;
                            noise += 0.5f;
                            noise += noise_offset;

                            L3_noise_vect[i][j] = noise;///Insert noise instead of real value
                        }
                        else
                        {
                            ///Insert a small region of the input vector
                            L3_noise_vect[i][j] = pool_temporary;///Insert real data from input vector
                        }
                    }
                }

                ///*********** Forward to hidden nodes ****************
                ///Make autoencoder forward of FL3 feature
                for(int k=0; k<L3_conv_depth; k++)
                {
                    ///L3_conv_cube[0][k] = 0.0f;///Clear autoencoder hidden node (hidden node = Lx_conv_cube[0][depth] ). [0] because we borrow L1_conv_cube mem this first area sheet pos in the convoution memory area now only for autencoder
                    L3_conv_cube[0][k] = FLx_bias_value * L3_weight_matrix_M[FL3_size][k];///Clear autoencoder hidden node with bias. Begin with the bias weight signal
                    for(int j=0; j<FL3_depth; j++)
                    {
                        for(int i=0; i<(FL3_srt_size * FL3_srt_size); i++)
                        {
                            L3_conv_cube[0][k] += L3_noise_vect[i][j] * L3_weight_matrix_M[i + (FL3_srt_size * FL3_srt_size)*j][k];///Make the dot product to pruduce the node.[0] because we borrow L1_conv_cube mem this first area sheet pos in the convoution memory area now only for autencoder
                        }
                        ///*** Relu this node ***
                        if(L3_conv_cube[0][k] < 0.0f)
                        {
                            L3_conv_cube[0][k] = L3_conv_cube[0][k] * Relu_neg_gain;///Relu function
                        }
                    }
                }

                ///********** Forward to L3 output nodes *******************
                ///Clear the L3_autoenc_reconstructed
                for(int j=0; j<FL3_depth; j++)
                {
                    for(int i=0; i<(FL3_srt_size * FL3_srt_size); i++)
                    {
                        L3_autoenc_reconstructed[i][j] = 0.0f;///Clear
                    }
                }

                ///Make autoencoder reconstruction
                FLx_bias_reconstruction =0.0f;///Clear
                for(int k=0; k<L3_conv_depth; k++)
                {
                    FLx_bias_reconstruction += FLx_bias_value * L3_weight_matrix_M[FL3_size][k];///Start with adding the Bias. [FL1_size][] is the bias weight
                    for(int j=0; j<FL3_depth; j++)
                    {
                        for(int i=0; i<(FL3_srt_size * FL3_srt_size); i++)
                        {
                            L3_autoenc_reconstructed[i][j] += L3_conv_cube[0][k] * L3_weight_matrix_M[i + (FL3_srt_size * FL3_srt_size)*j][k];///Reconstruction using the autoencoder tie weight. L1_conv_cube[0][j] is the hidden node of the autoencoder
                        }
                    }
                }
                ///Make the autoencoder loss and delta calculation
                static float L3_loss=0.0f;
                float L3_delta=0.0f;
                float L3_output_node=0.0f;
                bias_delta=0.0f;
                ///L3_loss=0.0f;
                bias_delta = FLx_bias_value - FLx_bias_reconstruction;///Start loss calculation with the bias.
                L3_loss += bias_delta * bias_delta;/// loss = 1/2 SUM k (input[k] - ouput[k])²

                for(int j=0; j<FL3_depth; j++)
                {
                    for(int i=0; i<(FL3_srt_size * FL3_srt_size); i++)
                    {
                        //L2_delta = *index_ptr_L1_patch_img - *index_ptr_L1_autoenc_reconstructed;
                        L3_output_node = L3_autoenc_reconstructed[i][j];
                        L3_delta = L3_patch_vect[i][j] - L3_output_node ;
                        L3_loss += L3_delta * L3_delta;/// loss = 1/2 SUM k (input[k] - ouput[k])²
                        ///    if(L3_output_node  < 0.0f)///
                        ///    {
                        ///        L3_delta *= Relu_neg_gain;
                        ///    }
                        L3_autoenc_delta[i][j] = L3_delta;
                    }
                }

                if(comon_func_Obj1.started == 1)
                {
                    ///make the backpropagation
                    ///First update the bias weights
                    for(int k=0; k<L3_conv_depth; k++)
                    {
                        /// **** update tied weight regarding delta
                        L3_change_weight_M[FL3_size][k] = L3_LearningRate * L3_conv_cube[0][k] * bias_delta + L3_Momentum * L3_change_weight_M[FL3_size][k];///hidden_node = L1_conv_cube[0][j];
                        L3_weight_matrix_M[FL3_size][k] += L3_change_weight_M[FL3_size][k];
                    }
                    for(int k=0; k<L3_conv_depth; k++)
                    {
                        for(int j=0; j<FL3_depth; j++)
                        {
                            for(int i=0; i<(FL3_srt_size * FL3_srt_size); i++)
                            {

                                /// **** update tied weight regarding delta
                                L3_change_weight_M[i + (FL3_srt_size * FL3_srt_size)*j][k] = L3_LearningRate * L3_conv_cube[0][k] * (L3_autoenc_delta[i][j]) + L3_Momentum * L3_change_weight_M[i + (FL3_srt_size * FL3_srt_size)*j][k];///hidden_node = L3_conv_cube[0][j];
                                L3_weight_matrix_M[i + (FL3_srt_size * FL3_srt_size)*j][k] += L3_change_weight_M[i + (FL3_srt_size * FL3_srt_size)*j][k];
                            }
                        }
                    }
                }
                ///Print Loss
                static int L3_print_loss=0;
                if(L3_print_loss<200)
                {
                    L3_print_loss++;
                }
                else
                {
                    L3_loss = L3_loss / 2.0f;/// loss = 1/2 SUM k (input[k] - ouput[k])²
                    printf("L3 autoencoder L3_loss error = %f\n", L3_loss);
                    L3_print_loss=0;
                    L3_loss=0.0f;
                    if(auto_save_ON==1)
                    {
                        auto_save_counter++;
                    }

                }
                ///End print loss

            }///End ittr loop



            ///L3 visual
            static int visualL3_now_counter=1000;
            if(visualL3_now_counter > 100)
            {
                visualL3_now_counter=0;
                printf("show L3\n");

                L3_attach_weight2mat.Lx_src = L3_visual_all_feature;///attach Mat pointer

                for(int j=0; j<FL3_depth; j++)
                {
                    for(int i=0; i<(FL3_srt_size * FL3_srt_size); i++)
                    {
                        L3_attach_weight2mat.Lx_ptr_M_matrix = &L3_weight_matrix_M[i + (FL3_srt_size * FL3_srt_size)*j][0];
                        /// L3_attach_weight2mat.k = k;
                        L3_attach_weight2mat.FLx_i_location_area  = i;
                        L3_attach_weight2mat.FLx_j_location_depth = j;
                        L3_attach_weight2mat.Xattach_weight2mat();
                    }
                }
                L3_visual_all_feature += 0.5f;
                imshow("L3_visual_all_feature", L3_visual_all_feature);

                waitKey(1);
            }///Visual now end
            else
            {
                visualL3_now_counter++;
            }

            ///Show L3toL1
            static int visuL3L1_now_counter=100000;
            if(visuL3L1_now_counter>8000 || comon_func_Obj1.Start_Visualize_L3==1)
            {
                comon_func_Obj1.Start_Visualize_L3=0;
                visuL3L1_now_counter=0;
                /// visualize_L3toL1 = Scalar(0.5,0.5,0.5);
                sq_visualize_L3toL1 = Scalar(0.5,0.5,0.5);
                /// float* zero_ptr_visualize_L3toL1 = visualize_L3toL1.ptr<float>(0);
                /// float* index_ptr_visualize_L3toL1 = visualize_L3toL1.ptr<float>(0);
                float* zero_ptr_sq_visualize_L3toL1 = sq_visualize_L3toL1.ptr<float>(0);
                float* index_ptr_sq_visualize_L3toL1 = sq_visualize_L3toL1.ptr<float>(0);

                int temp_offset=0;
                int sq_temp_offset=0;
                int show_node_patch_width_C1 = L3toL2toL1_feature_revers_size;
                int show_node_patch_hight_C1 = L3toL2toL1_feature_revers_size;
                ///Show now L3toL1

                for(int k3=0; k3<L3_conv_depth; k3++)
                {
                    for(int k2=0; k2<L2_conv_depth; k2++)///L2_conv_depth = FL3_depth
                    {
                        for(int i3=0; i3<FL3_srt_size*FL3_srt_size; i3++)
                        {
                            for(int j2=0; j2<FL2_depth; j2++) ///L1_conv_depth = FL2_depth
                            {
                                for(int i2=0; i2<FL2_srt_size*FL2_srt_size; i2++)
                                {
                                    for(int j1=0; j1<FL1_depth; j1++)
                                    {
                                        for(int i1=0; i1<FL1_srt_size*FL1_srt_size; i1++)
                                        {
                                            ///Add the start left corner colum offset for the show of L3 hidden node patch
                                            sq_temp_offset = show_node_patch_width_C1*FL1_depth*(k3%sqr_L3_conv_depth_plus1) + sq_visualize_L3toL1.cols*FL1_depth*show_node_patch_hight_C1*(k3/sqr_L3_conv_depth_plus1);
                                            sq_temp_offset += (i3%FL3_srt_size)*FL1_depth*pool_sqr*pool_sqr*L1_stride*L2_stride;///horizontal inc pos regarding horiz inc on FL3 horiz pos
                                            sq_temp_offset += (i2%FL2_srt_size)*FL1_depth*pool_sqr*L1_stride;///horizontal inc pos regarding horiz inc on FL2 horiz pos
                                            sq_temp_offset += (i1%FL1_srt_size)*FL1_depth + j1;///horizontal inc pos regarding horiz inc on FL1 horiz pos
                                            sq_temp_offset += sq_visualize_L3toL1.cols*FL1_depth*(i3/FL3_srt_size)*pool_sqr*pool_sqr*L1_stride*L2_stride;///Vertical inc pos regarding FL3 vertical pos
                                            sq_temp_offset += sq_visualize_L3toL1.cols*FL1_depth*(i2/FL2_srt_size)*pool_sqr*L1_stride;///Vertical inc pos regarding FL2 vertical pos
                                            sq_temp_offset += sq_visualize_L3toL1.cols*FL1_depth*(i1/FL1_srt_size);///Vertical inc pos regarding FL1 vertical pos
                                            index_ptr_sq_visualize_L3toL1 = zero_ptr_sq_visualize_L3toL1 + sq_temp_offset;
                                            *index_ptr_sq_visualize_L3toL1 += L1_weight_matrix_M[i1*FL1_depth + j1][j2] * L2_weight_matrix_M[i2 + (FL2_srt_size * FL2_srt_size)*j2][k2] * L3_weight_matrix_M[i3 + (FL3_srt_size * FL3_srt_size)*k2][k3];
                                        }
                                    }
                                }
                            }
                        }
                    }
                    printf("k3=%d\n", k3);
                    imshow("sq_visualize_L3toL1", sq_visualize_L3toL1);
                    waitKey(1);
                }
                ///   imshow("visualize_L2toL1", visualize_L2toL1);
                //  imshow("sq_visualize_L3toL1", sq_visualize_L3toL1);
                Mat cloned_sq_visualize_L3toL1;
                cloned_sq_visualize_L3toL1 = sq_visualize_L3toL1.clone();
                cloned_sq_visualize_L3toL1 *= 255;
                cv::imwrite("cloned_sq_visualize_L3toL1.JPG",cloned_sq_visualize_L3toL1);

            }
            else
            {
                visuL3L1_now_counter++;
            }
///End L3tL1 show
        }
///
        ///**************** Begin L3 Convloution *****************
        if(comon_func_Obj1.L3_convolution_ON == 1)/// Do convloution L3 process instead of L3 Autoencoder process
        {
            ///**********************************
            ///********* Convolute L3 ***********
            ///**********************************
            for(int i=0; i<(L3_conv_hight * L3_conv_width); i++)///This loop step throue (convolute) the area of the input "sheet". No padding of the slide on one sheet area of the convolution cube
            {
                for(int j=0; j<L3_conv_depth; j++)///This loop step throue the depth of the Convolution L2 cube
                {
                    ///L3_conv_cube[i][j] = 0.0f;///Clear the one node of the conv cube
                    L3_conv_cube[i][j] = L3_weight_matrix_M[FL3_size][j] * FLx_bias_value;///Clear with bias the one node of the conv cube
                    //int input_ptr_offset;
                    float input_node;
                    for(int k=0; k<FL3_size; k++)///This loop step throue one Feature cube. FL3_size = (FL3_srt_size * FL3_srt_size * FL3_depth)  The size of one Feature FL3
                    {
                        ///L2_pool_cube[area][depth]
                        input_node = L2_pool_cube[(L2_conv_hight/pool_sqr)*(L3_stride*(i/L3_conv_width) + (k/(FL3_srt_size * FL3_depth))) + L3_stride*(i%L3_conv_width) + (k/FL3_depth)%FL3_srt_size][k%FL3_depth];
                        L3_conv_cube[i][j] += L3_weight_matrix_M[k][j] * input_node;///Make the dot product of the 0..x Feature * input image to L2 convolution cube
                    }
                }
            }
            ///**************************************
            ///********* End Convolute L3 ***********
            ///**************************************
            ///**************************************
            ///********* Pooling L3 *****************
            ///**************************************

            if(!(connect_fc_to_pool_or_conv == 1 && connect_fc_to_layer== 3))///Skip poolining if fc connected to conv cube on this layer
            {
                for(int h=0; h<(L3_conv_hight / pool_sqr); h++)///This loop togheter with i loop step throue the "sheet" area of of the L2_pool_cube[i][x]
                {
                    for(int i=0; i<(L3_conv_width / pool_sqr); i++)///This loop togheter with h loop step throue the "sheet" area of of the L2_pool_cube[i][x]
                    {
                        for(int j=0; j<L3_conv_depth; j++)///This loop step throue the depth of the Convolution L2 cube
                        {
                            float max_node= -1000000.0f;
                            float compare_max = 0.0f;
                            int revers_pool_row=0;
                            for(int p=0; p<pooling; p++)
                            {
                                revers_pool_row = p/pool_sqr;///Get a pool row level inside the pooling area to make it poosible to read the right row position on the convolution sheet area
                                ///compare_max = L2_conv_cube[h * pool_sqr * L2_conv_width + L2_conv_width * revers_pool_row + i * pool_sqr + p % pool_sqr][j];///Prepare pooling compare value. Pick the value from the convolute (slide) position node at convolution cube
                                compare_max = L3_conv_cube[h * pool_sqr * L3_conv_width + L3_conv_width * revers_pool_row + i * pool_sqr + p%pool_sqr][j];///Prepare pooling compare value. Pick the value from the convolute (slide) position node at convolution cube
                                if(compare_max > max_node)
                                {
                                    L3_pool_tracking = p;
                                    max_node = compare_max;
                                }
                            }
                            ///*** Relu this node ***
                            if(max_node < 0.0f)
                            {
                                max_node = max_node * Relu_neg_gain;///Relu function
                            }
                            L3_pool_cube[h*(L3_conv_hight / pool_sqr) + i][j] = max_node;
                        }
                        /// Pooling of this convolute (slide) position is now done throue the full depth of nodes at L2_conv_cube[][] and L2_pool_cube[h*i][j]
                        /// h, i, pooltracking gives the convolute (slide) position to prepare the input patch/part for the autoencoder compare to reconstruction
                    }
                }
            }

            ///**************************************
            ///********* END Pooling L3 *************
            ///**************************************
///show one layer of conv L3
///Not show L3 conv pool layer
        } ///End convloution L3 process
///-------------------
        if(fully_conn_backprop == 1) /// object... .full_conn_backprop
        {
            if(comon_func_Obj1.finetune==1)
            {
                Learning_fc = 0;///Lock fc when finetune last layer of features
                ///fine tune will be done below after all fc weight updates
            }
            else
            {
                Learning_fc = Training_fc;/// Do weight update of (fc) fully connected network if not finetune feature throue backprop or running with verify test set
            }
///***********************************************************************************************************************************
///***********************************************************************************************************************************
///********************************** Supervised Learning Logistic regression ********************************************************
///***********************************************************************************************************************************
///***********************************************************************************************************************************


///**************************************************************************************
///************* Feed forward last layer Lx hidden node to fc_input_node ****************
/// !!!!!!!!!!!!!! Note that I test first with take data from L2 not L3
///**************************************************************************************
            float Accum=0.0f;
            int Not_dropout=0;
            ///void randomize_dropoutHid(int *zero_ptr_dropoutHidden, int HiddenNodes, int verification)
            if(comon_func_Obj1.started==1 && Learning_fc == 1)
            {
                Not_dropout=0;///Do dropout
            }
            else
            {
                Not_dropout=1;///Don't do dropout
            }

            randomize_dropoutHid(&dropoutHidden[0], fully_hidd_nodes, Not_dropout);///select dropout node to the hidden node

            float node_from_last_pool_or_conv_layer=0.0f;
#ifdef USE_MEAN_VALUE_TO_FC_NETWORK
            float meanvalue=0.0f;
            meanvalue=0.0f;
            for(int i=0; i<fc_input_NODES; i++)///Connect and sigmoid the Autencoder L2 nodes to fully connected neural network
            {
                meanvalue += L2_pool_cube[i%((L2_conv_hight/pool_sqr) * (L2_conv_width/pool_sqr))][i/((L2_conv_hight/pool_sqr) * (L2_conv_width/pool_sqr))];
            }
            meanvalue = meanvalue / fc_input_NODES;
#endif // USE_MEAN_VALUE_TO_FC_NETWORK

            for(int i=0; i<fc_input_NODES; i++)///Connect and sigmoid the Autencoder L2 nodes to fully connected neural network
            {
                ///L2_pool_cube[((L2_conv_hight/pool_sqr) * (L2_conv_width /pool_sqr))][L2_conv_depth]
                if(connect_fc_to_layer== 2)
                {
                    ///Mode connect_fc_to_layer == 2 connect the fully network to L2_pool_cube (Not to L3_pool_cube how is unused)
                    if(connect_fc_to_pool_or_conv == 0)
                    {
                        ///Use data from pool cube
                        node_from_last_pool_or_conv_layer = L2_pool_cube[i%((L2_conv_hight/pool_sqr) * (L2_conv_width/pool_sqr))][i/((L2_conv_hight/pool_sqr) * (L2_conv_width/pool_sqr))];
                    }
                    else
                    {
                        ///Use data from the unpooled data direct from convolution cube data
                        node_from_last_pool_or_conv_layer = L2_conv_cube[i%((L2_conv_hight) * (L2_conv_width))][i/((L2_conv_hight) * (L2_conv_width))];
                        ///Relu this dotproduct already here because it is normaly done after pooling
                        ///*** Relu this node ***
                        if(node_from_last_pool_or_conv_layer < 0.0f)
                        {
                            node_from_last_pool_or_conv_layer = node_from_last_pool_or_conv_layer * Relu_neg_gain;///Semi Relu function
                        }
                    }
                }
                if(connect_fc_to_layer== 3)///L3 Convloution in now also implemented.
                {
                    ///Mode connect_fc_to_layer== 3 will Connect to last conv/pool layer
                    if(connect_fc_to_pool_or_conv == 0)
                    {
                        ///Use data from pool cube
                        node_from_last_pool_or_conv_layer = L3_pool_cube[i%((L3_conv_hight/pool_sqr) * (L3_conv_width/pool_sqr))][i/((L3_conv_hight/pool_sqr) * (L3_conv_width/pool_sqr))];
                    }
                    else
                    {
                        ///Use data from the unpooled data direct from convolution cube data
                        node_from_last_pool_or_conv_layer = L3_conv_cube[i%((L3_conv_hight) * (L3_conv_width))][i/((L3_conv_hight) * (L3_conv_width))];
                        ///Relu this dotproduct already here because it is normaly done after pooling
                        ///*** Relu this node ***
                        if(node_from_last_pool_or_conv_layer < 0.0f)
                        {
                            node_from_last_pool_or_conv_layer = node_from_last_pool_or_conv_layer * Relu_neg_gain;///Semi Relu function
                        }
                    }
                }
#ifdef USE_MEAN_VALUE_TO_FC_NETWORK
                node_from_last_pool_or_conv_layer -= meanvalue;
#endif // USE_MEAN_VALUE_TO_FC_NETWORK
                /// printf("node_from_last_pool_or_conv_layer =%f\n", node_from_last_pool_or_conv_layer);
                fc_input_node[i] = 1.0/(1.0 + exp(-(node_from_last_pool_or_conv_layer)));///Sigmoid function.  x = 1.0/(1.0 + exp(-(x)))
///fc_input_node[i] = node_from_last_pool_or_conv_layer;
                //   if(comon_func_Obj1.started==0)
                //   {
                //               printf("hidden_node[%d] = %f\n", i , node_from_last_pool_or_conv_layer);
                //               printf("fc_input_node[%d] = %f\n", i, fc_input_node[i]);
                //   }

            }

            for(int j=0; j<fully_hidd_nodes; j++)
            {
                Accum = fc_hidden_weight[fc_input_NODES][j] * Bias_level;///Begin with the Bias node as the start value
                for(int i=0; i<fc_input_NODES; i++)
                {
                    Accum += fc_hidden_weight[i][j] * fc_input_node[i];///Weight in each input data modified by sigmoid from autoencoder feature map into the fully connected neural network hidden nodes
                }
                if(dropoutHidden[j] == 0)
                {
                    ///Normal forward not drop out this node
                    fc_hidden_node[j] = 1.0/(1.0 + exp(-(Accum)));///Sigmoid function.  x = 1.0/(1.0 + exp(-(x)))
                }
                else
                {
                    fc_hidden_node[j] = 0.0f;
                }
                ///    printf("fc_hidden_node[%d] =%f\n", j, fc_hidden_node[j]);
            }
///***************************************************************************
///***************************************************************************
///***************************************************************************
            ///Compare the actual output value from feed forward to target
            ///Insert target value into the fc_target_node from lable database
            for(int n=0; n<fully_out_nodes; n++)
            {
#ifdef USE_MNIST
                if(n == ((int) MNIST_lable[MNIST_nr]))
#endif // USE_MNIST
#ifdef USE_CIFAR
                    if(n == ((int) CIFAR_data[(CIFAR_nr*CIFAR_row_size)]))
#endif // USE_CIFAR
#ifdef USE_IMAGE
                        if(n == toggle_pos_neg)
#endif // USE_IMAGE
                        {
                            ///fc_target_node[n] = 1.0f;///This is the node how is correspond to the lable digits
                            fc_target_node[n] = High_Target_value;///This is the node how is correspond to the lable digits
                        }
                        else
                        {
                            ///fc_target_node[n] = 0.0f;///This is NOT the node how is correspond to the lable digits
                            fc_target_node[n] = Low_Target_value;///This is the node how is correspond to the lable digits
                        }
            }
            Error_level = 0.0f;
///***************************************************************************
///*************** Feed forward fc_hidden_node[] to fc_output_node[] *********
///***************************************************************************

            for(int j=0; j<fully_out_nodes; j++)
            {
                Accum = fc_output_weight[fully_hidd_nodes][j] * Bias_level;///Begin with the Bias node as the start value
                for(int i=0; i<fully_hidd_nodes; i++)
                {
                    Accum += fc_output_weight[i][j] * fc_hidden_node[i];///
                }
                fc_output_node[j] = 1.0/(1.0 + exp(-(Accum)));///Sigmoid function.  x = 1.0/(1.0 + exp(-(x)))
                fc_output_delta[j] = (fc_target_node[j] - fc_output_node[j]) * fc_output_node[j] * (1.0f - fc_output_node[j]);

///                fc_output_delta[j] = (fc_softmax_out_delta[j]) * fc_output_node[j] * (1.0f - fc_output_node[j]);

                if(comon_func_Obj1.started == 0 && enable_print_nodes==1)
                {
                    //          printf("debug_counter %d training target %d j = %d\n", debug_counter, training_image, j);
                    printf("fc_output_node[%d] = %f\n",j, fc_output_node[j]);
                    printf("fc_target_node[%d] = %f\n", j, fc_target_node[j]);
                    //  printf("fc_output_delta[j] = %f\n", fc_output_delta[j]);
                }
                Error_level += 0.5 * (fc_target_node[j] - fc_output_node[j]) * (fc_target_node[j] - fc_output_node[j]);///
            }
            if(comon_func_Obj1.started == 0 || print_only_100==0)
            {
                ///  printf("CIFAR lable %d\n", CIFAR_data[(CIFAR_nr*CIFAR_row_size)]);

                printf("Error_level fc =%f\n", Error_level);
                if(auto_save_ON==1)
                {
                    auto_save_counter++;
                }
            }

            if(print_only_100<1)
            {
                print_only_100++;
            }
            else
            {
                print_only_100=0;

            }
///***************************************************************************
///**************** End feed forward *****************************************
///***************************************************************************

///***************************************************************************
///**************** Backpropagate fully connected network ********************
///***************************************************************************

            /******************************************************************
            * Backpropagate errors from output layer to hidden layer
            ******************************************************************/

            for(int i = 0 ; i < fully_hidd_nodes ; i++ )
            {
                Accum = 0.0 ;
                for(int j = 0 ; j < fully_out_nodes ; j++ )
                {
                    Accum += fc_output_weight[i][j] * fc_output_delta[j] ;
                }
                if(dropoutHidden[i] == 0)
                {
                    fc_hidden_delta[i] = Accum * fc_hidden_node[i] * (1.0 - fc_hidden_node[i]);///Backpropagate gradiant decent
                }
                else
                {
                    fc_hidden_delta[i] = 0.0f;/// Hidden node delta zero when drop out no change of the weight regarding this backprop
                }
            }

            if(comon_func_Obj1.finetune==1 && comon_func_Obj1.started == 1)/// Check if Update Weight L2 feature finetune from fc backpropagation should be done
            {
                ///******************************************************************
                ///* Backpropagate errors from hidden layer to fc input layer
                ///******************************************************************
                for(int i = 0 ; i < fc_input_NODES ; i++ )
                {
                    Accum = 0.0 ;
                    for(int j = 0 ; j < fully_hidd_nodes ; j++ )
                    {
                        Accum += fc_hidden_weight[i][j] * fc_hidden_delta[j] ;
                    }
                    fc_input_delta[i] = Accum * fc_input_node[i] * (1.0 - fc_input_node[i]) ;///Backpropagate gradiant decent used to L2 weight update L2_hid_node_delta[j] = fc_input_delta[j];
                }

                /******************************************************************
                * Update Weight L2 feature finetune from fc backpropagation
                ******************************************************************/
                ///TODO add update weight L2 features
                ///Only L2  Note L3 fine tune not set NOT USE L3 fine tune it won't work yet
                if(connect_fc_to_layer== 2 && pooling == 4)
                {
                    ///**************************************************************************************************************************************************************
                    ///********* Do Convolute L2 first step but not fully only to find the input data regarding fine tune delta to establish weight update of L2 fine tune***********
                    ///**************************************************************************************************************************************************************
                    ///L2_conv_cube[i][j] = 0.0f;///Clear the one node of the conv cube
                    if(connect_fc_to_pool_or_conv == 0)///0 = used L2_pool_cube, 1 = used L2_conv_cube to fc network
                    {
                        for(int h=0; h<(L2_conv_hight / pool_sqr); h++)///This loop togheter with i loop step throue the "sheet" area of of the L1_pool_cube[i][x]
                        {
                            for(int i=0; i<(L2_conv_width / pool_sqr); i++)///This loop togheter with h loop step throue the "sheet" area of of the L1_pool_cube[i][x]
                            {
                                float test_77;
                                int L2_pool_tracker=0;

                                for(int j=0; j<L2_conv_depth; j++)///This loop step throue the depth of the Convolution L1 cube
                                {
                                    L2_pool_tracker = L2_pool_track_cube[(h*(L2_conv_width / pool_sqr)) + i][j];
                                    int fc_inp_delta_index = (h*(L2_conv_width / pool_sqr)) + (i) + (j*(L2_conv_hight / pool_sqr)*(L2_conv_width / pool_sqr));
                                    ///Test validation
                                    test_77 = L2_pool_cube[h*(L2_conv_hight / pool_sqr) + i][j];
                                    test_77 = 1.0/(1.0 + exp(-(test_77)));///Sigmoid function.  x = 1.0/(1.0 + exp(-(x)))
                                    float test_88;
                                    test_88 = fc_input_node[fc_inp_delta_index];
                                    if(test_77 != test_88)
                                    {
                                        printf("Not Equal test_77 = %f test_88 = %f\n", test_77, test_88);
                                    }
                                    if(L2_pool_tracker < 0 || L2_pool_tracker > 3)
                                    {
                                        printf("Error ! L2_pool_tracker out of range = %d\n", L2_pool_tracker);
                                        exit(0);
                                    }

                                    ///end validation
                                    L2_change_weight_M[FL2_size][j] = L2_LearningRate * FLx_bias_value * fc_input_delta[fc_inp_delta_index] + L2_Momentum * L2_change_weight_M[FL2_size][j];///hidden_node = L1_conv_cube[0][j];
                                    L2_weight_matrix_M[FL2_size][j] += L2_change_weight_M[FL2_size][j];
                                    float input_node;
                                    for(int k=0; k<FL2_size; k++)///This loop step throue one Feature cube. FL2_size = (FL2_srt_size * FL2_srt_size * FL2_depth)  The size of one Feature FL2
                                    {
                                        ///L1_pool_cube[area][depth]
                                        ///input_node = L1_pool_cube[(L1_conv_hight/pool_sqr)*(L2_stride*(i/L2_conv_width) + (k/(FL2_srt_size * FL2_depth))) + L2_stride*(i%L2_conv_width) + (k/FL2_depth)%FL2_srt_size][k%FL2_depth];
                                        int FL2_col = k % (FL2_srt_size * FL2_srt_size);
                                        int FL2_row = (k / FL2_srt_size) % FL2_srt_size;
                                        int FL2_dep = k / (FL2_srt_size * FL2_srt_size);
                                        ///input_node = L1_pool_cube[(L2_pool_tracker%pool_sqr) + (FL2_col) + i*pool_sqr*L2_stride + (((L2_pool_tracker/pool_sqr) + FL2_row + h*L2_stride)*(L1_conv_width/pool_sqr))][FL2_dep];
                                        input_node = L1_pool_cube[(FL2_col) + ((L2_pool_tracker%pool_sqr) + i*pool_sqr)*L2_stride + (FL2_row + ((L2_pool_tracker/pool_sqr) + h*pool_sqr) * (L1_conv_width/pool_sqr))][FL2_dep];

                                        ///Update L2 feature weights
                                        L2_change_weight_M[k][j] = L2_LearningRate * input_node * fc_input_delta[fc_inp_delta_index] + L2_Momentum * L2_change_weight_M[k][j];///hidden_node = L1_conv_cube[0][j];
                                        L2_weight_matrix_M[k][j] += L2_change_weight_M[k][j];
                                    }

                                }
                            }
                        }
                    }
                    else
                    {
                        int index_fc_input_NODE = 0;///This should fit in same range as i in for(int i=0; i<fc_input_NODES; i++). will inc at each use_this_input_node_data = 1
                        for(int i=0; i<(L2_conv_hight * L2_conv_width); i++)///This loop step throue (convolute) the area of the input "sheet". No padding of the slide on one sheet area of the convolution cube
                        {
                            for(int j=0; j<L2_conv_depth; j++)///This loop step throue the depth of the Convolution L2 cube
                            {

                                ///Use data from conv cube to fc
                                L2_change_weight_M[FL2_size][j] = L2_LearningRate * FLx_bias_value * fc_input_delta[index_fc_input_NODE] + L2_Momentum * L2_change_weight_M[FL2_size][j];///hidden_node = L1_conv_cube[0][j];
                                L2_weight_matrix_M[FL2_size][j] += L2_change_weight_M[FL2_size][j];
                                float input_node;
                                for(int k=0; k<FL2_size; k++)///This loop step throue one Feature cube. FL2_size = (FL2_srt_size * FL2_srt_size * FL2_depth)  The size of one Feature FL2
                                {
                                    ///L1_pool_cube[area][depth]
                                    input_node = L1_pool_cube[(L1_conv_hight/pool_sqr)*(L2_stride*(i/L2_conv_width) + (k/(FL2_srt_size * FL2_depth))) + L2_stride*(i%L2_conv_width) + (k/FL2_depth)%FL2_srt_size][k%FL2_depth];
                                    ///Update L2 feature weights
                                    L2_change_weight_M[k][j] = L2_LearningRate * input_node * fc_input_delta[index_fc_input_NODE] + L2_Momentum * L2_change_weight_M[k][j];///hidden_node = L1_conv_cube[0][j];
                                    L2_weight_matrix_M[k][j] += L2_change_weight_M[k][j];
                                }
                                index_fc_input_NODE++;
                            }
                        }
                        if(index_fc_input_NODE == fc_input_NODES)
                        {
                            printf("perfect match index_fc_input_NODE == fc_input_NODES \n");
                        }

                    }
                    ///********************************************************************************
                    ///********* End Do Convolute L2 first step for fine tune weight update ***********
                    ///********************************************************************************
                    /******************************************************************
                    * End Update L2 weights fine tune Supervised learning
                    ******************************************************************/

                }
                else
                {
                    printf("Error! settings\n");
                    printf("Fine tune through pooling ONLY supported with pooling = 4. pooling is = %d\n", pooling);
                    printf("Fine tune only supported for connect_fc_to_layer == 2 connect_fc_to_layer is = %d \n", connect_fc_to_layer);
                    exit(0);
                }
            }

///**************** End backprop **************
            if(comon_func_Obj1.started == 1)
            {
                /******************************************************************
                * Update Inner-->Hidden Weights
                ******************************************************************/
                if(Learning_fc == 1)
                {
                    for(int i = 0 ; i < fully_hidd_nodes ; i++ )
                    {
                        fc_change_hidden_weight[fc_input_NODES][i] = fc_LearningRate * Bias_level * fc_hidden_delta[i] + fc_Momentum * fc_change_hidden_weight[fc_input_NODES][i] ;///Begin with update Bias change weights
                        if(dropoutHidden[i] == 0)
                        {
                            fc_hidden_weight[fc_input_NODES][i] += fc_change_hidden_weight[fc_input_NODES][i];///Begin with update Bias weights
                            for(int j = 0 ; j < fc_input_NODES ; j++ )
                            {
                                fc_change_hidden_weight[j][i] = fc_LearningRate * fc_input_node[j] * fc_hidden_delta[i] + fc_Momentum * fc_change_hidden_weight[j][i];
                                fc_hidden_weight[j][i] += fc_change_hidden_weight[j][i] ;
                            }
                        }
                    }
                }
                /******************************************************************
                * Update Hidden-->Output Weights
                ******************************************************************/

                if(Learning_fc == 1)
                {


                    for(int i = 0 ; i < fully_out_nodes ; i++ )
                    {
                        fc_change_output_weight[fully_hidd_nodes][i] = fc_LearningRate * Bias_level * fc_output_delta[i] + fc_Momentum * fc_change_output_weight[fully_hidd_nodes][i] ;///Begin with update Bias change weights
                        fc_output_weight[fully_hidd_nodes][i] += fc_change_output_weight[fully_hidd_nodes][i];///Begin with update Bias weights
                        for(int j = 0 ; j < fully_hidd_nodes ; j++ )
                        {
                            if(dropoutHidden[j] == 0)
                            {
                                fc_change_output_weight[j][i] = fc_LearningRate * fc_hidden_node[j] * fc_output_delta[i] + fc_Momentum * fc_change_output_weight[j][i];
                                fc_output_weight[j][i] += fc_change_output_weight[j][i] ;
                            }
                        }
                    }
                }
///***********
            }///End if(comon_func_Obj1.started == 1)
        }///End if(fully_conn_backprop == 1)
///***************************************************************************
///**************** END Backpropagate fully connected network ****************
///***************************************************************************

///-------------------

///******************** Visualize the highest category **********************************************
#ifdef USE_IMAGE
        labeling.setTo(cv::Scalar(50,0,0));
        if(fc_output_node[0]> 0.5f)
        {

            cv::putText(labeling, IMAGE_cat_pos, cvPoint(15,60), CV_FONT_HERSHEY_PLAIN, 4, cvScalar(0,255,0),3);
        }
        else
        {
            cv::putText(labeling, IMAGE_cat_neg, cvPoint(15,60), CV_FONT_HERSHEY_PLAIN, 4, cvScalar(0,0,255),3);
        }

#else
        char Highest_rate =0;
        string highest = "x";
        std::string::iterator It = highest.begin();
        It = highest.begin();
        float compare_outpnodes = 0.0f;
        Highest_rate =0;
        compare_outpnodes = 0.0f;
        for(int i = 0 ; i < fully_out_nodes ; i++ )
        {
            if(compare_outpnodes < fc_output_node[i])
            {
                compare_outpnodes  = fc_output_node[i];
                Highest_rate = i;
            }
        }
        waitKey(1);
        //CvPoint num_pos = (15,100);
        *It = Highest_rate+48;
        labeling.setTo(cv::Scalar(50,0,0));
        cv::putText(labeling, highest, cvPoint(15,60), CV_FONT_HERSHEY_PLAIN, 4, cvScalar(0,255,0),3);
#endif //
//void putText(Mat& img, const string& text, Point org, int fontFace, double fontScale, Scalar color, int thickness=1, int lineType=8, bool bottomLeftOrigin=false )¶
        imshow("lable", labeling);
///********************************************************

///********** Calculate the correct prediction ratio **********
        if(fully_conn_backprop == 1)
        {
            if(test_itterations<RESET_test_ittr)
            {
                test_itterations++;
            }
            else
            {
                test_itterations=0;
                test_correct=0;
            }
            int target_lable=0;
#ifdef USE_MNIST
            target_lable = ((int) MNIST_lable[MNIST_nr]);
#endif // USE_MNIST
#ifdef USE_CIFAR
            target_lable = ((int) CIFAR_data[(CIFAR_nr*CIFAR_row_size)]);
#endif // USE_CIFAR
#ifdef USE_IMAGE
            target_lable = toggle_pos_neg;
#endif // USE_IMAGE
#ifdef USE_IMAGE
            if((fc_target_node[0] == High_Target_value && fc_output_node[0]> 0.5f) || (fc_target_node[0] == Low_Target_value && fc_output_node[0]< 0.5f))
            {

                printf("Correct prediction!\n");
                test_correct++;
            }
            else
            {
                printf("Failure prediction\n");
            }
            if((test_itterations%10) == 0)
            {
                correct_ratio = ((float) test_correct) / ((float) test_itterations);
                printf("test_itterations = %d test_correct = %d\n", test_itterations, test_correct);
                printf("correct_ratio = %f\n", correct_ratio);
            }
#else
            if(Highest_rate == target_lable)
            {
                test_correct++;
            }
            else
            {
                printf("Failure to predict correct category\n");
            }

            if((test_itterations%20) == 0)
            {
                correct_ratio = ((float) test_correct) / ((float) test_itterations);
                printf("test_itterations = %d test_correct = %d\n", test_itterations, test_correct);
                printf("correct_ratio = %f\n", correct_ratio);
            }
#endif // USE_IMAGE
            if(comon_func_Obj1.started == 0 && enable_print_nodes==0)
            {
                printf("target_lable %d\n", target_lable);
            }

        }
///******* End correct prediction ratio ***********************

///*** switch CIFAR batch *********
        static int turns=0;
        static int switch_batch_counter=0;
        static int use_batch=1;
        if(comon_func_Obj1.full_conn_backprop == 1)
        {
            if(turns < 10)
            {
                turns++;
            }
            else
            {
                turns=0;
                srand (static_cast <unsigned> (time(0)));//Seed the randomizer
                if(enable_print_nodes==1)
                {
                    printf("Seed randomizer\n");
                }
            }
#ifdef USE_MNIST
            //        printf("Highest_rate =%d\n", Highest_rate);

            //        printf("MNIST_lable[MNIST_nr] %d\n", MNIST_lable[MNIST_nr]);
            //        printf("MNIST_nr =%d\n", MNIST_nr);
#endif // USE_MNIST
#ifdef USE_CIFAR
            printf("Highest_rate =%d\n", Highest_rate);
            printf("Lable nr= %d\n", CIFAR_data[(CIFAR_nr*CIFAR_row_size)]);
            printf("CIFAR_nr at batch_%d= %d\n", use_batch, CIFAR_nr);

            ///Change batch sometimes
            if(switch_batch_counter < 100)
            {
                switch_batch_counter++;
            }
            else
            {
                switch_batch_counter=0;
                if(use_batch<How_Many_CIFAR_batch_in_use)
                {
                    use_batch++;
                }
                else
                {
                    use_batch=1;
                }
                printf("use_batch=%d\n", use_batch);

                sprintf(filename, "data_batch_%d.bin", use_batch);
                fp1 = fopen(filename, "r");
                if (fp1 == NULL)
                {
                    printf("Error while opening file data_batch_%d.bin", use_batch);
                    exit(0);
                }
                ///read_CIFAR_image()
                ///   char* CIFAR_data;
                ///   CIFAR_data = new char[nr_of_CIFAR_file_bytes];
                ///   int MN_index=0;
                ///   char c_data=0;
                MN_index=0;
                for(int i=0; i<nr_of_CIFAR_file_bytes; i++)
                {
                    c_data = fgetc(fp1);
                    if( feof(fp1) )
                    {
                        break;
                    }
                    //printf("c_data %d\n", c_data);
                    CIFAR_data[MN_index] = c_data;
                    MN_index++;
                }
                fclose(fp1);
                printf("data_batch_%d.bin data is put into CIFAR_data\n", use_batch);
            }
#endif // USE_CIFAR
#ifdef USE_MNIST
            //        printf("Highest_rate =%d\n", Highest_rate);

            //        printf("MNIST_lable[MNIST_nr] %d\n", MNIST_lable[MNIST_nr]);
            //        printf("MNIST_nr =%d\n", MNIST_nr);
#endif // USE_MNIST
#ifdef USE_IMAGE
///            printf("Highest_rate =%d\n", Highest_rate);
            printf("Lable nr= %d\n", toggle_pos_neg);
            printf("training_image =%d\n", training_image);
#endif // USE_IMAGE
        }
        waitKey(1);
        comon_func_Obj1.keyboard_event();///Check keyboard event
        if(comon_func_Obj1.started==0)
        {
            waitKey(3000);
        }
        if(comon_func_Obj1.init_random_fc_weights==1)
        {
            comon_func_Obj1.init_random_fc_weights=0;

///************* Initialize randomized noise on fc_weight *********
            for(int i=0; i<fc_input_NODES+1; i++)
            {
                for(int j=0; j<fully_hidd_nodes; j++)
                {
                    Rando = (float) (rand() % 65535) / 65536;//0..1.0 range
                    Rando -= 0.5f;
                    Rando *= fc_start_weight_noise_range;
                    fc_hidden_weight[i][j] = Rando;///Noise around 0.5f
                    fc_change_hidden_weight[i][j] = 0.0f;///Initialize with zero
                }
            }

            for(int i=0; i<fully_hidd_nodes+1; i++)
            {
                for(int j=0; j<fully_out_nodes; j++)
                {
                    Rando = (float) (rand() % 65535) / 65536;//0..1.0 range
                    Rando -= 0.5f;
                    Rando *= fc_start_weight_noise_range;
                    fc_output_weight[i][j] = Rando;///Noise around 0.5f
                    fc_change_output_weight[i][j] = 0.0f;///Initialize with zero
                }
            }
            printf("Cleared with noise the fully connected network weights\n");
            waitKey(2000);
///***********************************
        }
        fully_conn_backprop = comon_func_Obj1.full_conn_backprop;
        if(comon_func_Obj1.save_L1_weights==1 || auto_save_counter>auto_save_at)
        {
            if(auto_save_counter>auto_save_at)
            {
                printf("Autosave\n");
            }
            auto_save_counter=0;

            comon_func_Obj1.save_L1_weights=0;
///L1 save weights
            //Save weights
            sprintf(filename, "L1_weight_matrix_M.dat");//Assigne a filename with index number added
            fp2 = fopen(filename, "w+");
            if (fp2 == NULL)
            {
                printf("Error while opening file L1_weight_matrix_M.dat");
                exit(0);
            }
            //size_t fread(void *ptr, size_t size_of_elements, size_t number_of_elements, FILE *a_file);
            ix=0;
            for(int n=0; n<L1_conv_depth; n++)
            {
                for(int p=0; p<FL1_size+1; p++)
                {
                    f_data[ix] = L1_weight_matrix_M[p][n];
                    ix++;
                }
            }
            fwrite(f_data, sizeof f_data[0], ((FL1_size+1)*L1_conv_depth), fp2);
            fclose(fp2);
            printf("weights are saved at L1_weight_matrix_M.dat file\n");

///End L1 save weights
///L2 save weights
            //Save weights
            sprintf(filename, "L2_weight_matrix_M.dat");//Assigne a filename with index number added
            fp2 = fopen(filename, "w+");
            if (fp2 == NULL)
            {
                printf("Error while opening file L2_weight_matrix_M.dat");
                exit(0);
            }
            //size_t fread(void *ptr, size_t size_of_elements, size_t number_of_elements, FILE *a_file);
            ix=0;
            for(int n=0; n<L2_conv_depth; n++)
            {
                for(int p=0; p<FL2_size+1; p++)
                {
                    f2_data[ix] = L2_weight_matrix_M[p][n];
                    ix++;
                }
            }
            fwrite(f2_data, sizeof f2_data[0], ((FL2_size+1)*L2_conv_depth), fp2);
            fclose(fp2);
            printf("weights are saved at L2_weight_matrix_M.dat file\n");
///End L2 save weights
///L3 save weights
            //Save weights
            sprintf(filename, "L3_weight_matrix_M.dat");//Assigne a filename with index number added
            fp2 = fopen(filename, "w+");
            if (fp2 == NULL)
            {
                printf("Error while opening file L3_weight_matrix_M.dat");
                exit(0);
            }
            //size_t fread(void *ptr, size_t size_of_elements, size_t number_of_elements, FILE *a_file);
            ix=0;
            for(int n=0; n<L3_conv_depth; n++)
            {
                for(int p=0; p<FL3_size+1; p++)
                {
                    f3_data[ix] = L3_weight_matrix_M[p][n];
                    ix++;
                }
            }
            fwrite(f3_data, sizeof f3_data[0], ((FL3_size+1)*L3_conv_depth), fp2);
            fclose(fp2);
            printf("weights are saved at L3_weight_matrix_M.dat file\n");
///End L3 save weights
///********** Save fc_hidden_weight ********************
///  fc_hidden_weight[fc_input_NODES+1][fully_hidd_nodes]
            sprintf(filename, "fc_hidden_weight.dat");//Assigne a filename with index number added
            fp2 = fopen(filename, "w+");
            if (fp2 == NULL)
            {
                printf("Error while opening file fc_hidden_weight.dat");
                exit(0);
            }
            ix=0;
            for(int n=0; n<fully_hidd_nodes; n++)
            {
                for(int p=0; p<fc_input_NODES+1; p++)
                {
                    f_data_fc_h_w[ix] = fc_hidden_weight[p][n];
                    ix++;
                }
            }
            fwrite(f_data_fc_h_w, sizeof f_data_fc_h_w[0], ((fc_input_NODES+1)*fully_hidd_nodes), fp2);
            fclose(fp2);
            printf("weights are saved at fc_hidden_weight.dat file\n");
///********** End Save fc_hidden_weight ********************
///********** Save fc_output_weight ********************
///  fc_output_weight[fully_hidd_nodes+1][fully_out_nodes]
            sprintf(filename, "fc_output_weight.dat");//Assigne a filename with index number added
            fp2 = fopen(filename, "w+");
            if (fp2 == NULL)
            {
                printf("Error while opening file fc_output_weight.dat");
                exit(0);
            }
            ix=0;
            for(int n=0; n<fully_out_nodes; n++)
            {
                for(int p=0; p<fully_hidd_nodes+1; p++)
                {
                    f_data_fc_o_w[ix] = fc_output_weight[p][n];
                    ix++;
                }
            }
            fwrite(f_data_fc_o_w, sizeof f_data_fc_o_w[0], ((fully_hidd_nodes+1)*fully_out_nodes), fp2);
            fclose(fp2);
            printf("weights are saved at fc_output_weight.dat file\n");
///********** End Save fc_output_weight ********************

        }///End save Lx
    }
    return 0;
}
