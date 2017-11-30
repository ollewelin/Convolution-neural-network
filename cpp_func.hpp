#ifndef CPP_FUNC_HPP_INCLUDED
#define CPP_FUNC_HPP_INCLUDED

#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
#include <opencv2/imgproc/imgproc.hpp> // Gaussian Blur
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)

//#include "cpp_func.cpp"
using namespace cv;

Mat gray_local_normalizing(Mat gray)
{

//#define BLUR_FLT_NUMERATOR 2
//#define BLUR_FLT_DENOMINATOR 20
#define BLUR_FLT_NUMERATOR 2
#define BLUR_FLT_DENOMINATOR 20

    Mat float_gray, blur, num, den, store_gray;
    store_gray = gray;//Initialize size
    // convert to floating-point image
    gray.convertTo(float_gray, CV_32F, 1.0/255.0);

    // numerator = img - gauss_blur(img)
    cv::GaussianBlur(float_gray, blur, Size(0,0), BLUR_FLT_NUMERATOR, BLUR_FLT_NUMERATOR);
    num = float_gray - blur;

    // denominator = sqrt(gauss_blur(img^2))
    cv::GaussianBlur(num.mul(num), blur, Size(0,0), BLUR_FLT_DENOMINATOR, BLUR_FLT_DENOMINATOR);
    cv::pow(blur, 0.5, den);

    // output = numerator / denominator
    gray = num / den;

    // normalize output into [0,1]
    cv::normalize(gray, gray, 0.0, 1.0, NORM_MINMAX, -1);

    // Display
    //namedWindow("demo", CV_WINDOW_AUTOSIZE );
    gray.convertTo(store_gray, CV_8U, 255);
    return store_gray;

    //imshow("demo", gray);
}

Mat CV_32FC3_local_normalizing(Mat input_colour)
{

//#define BLUR_FLT_NUMERATOR 2
//#define BLUR_FLT_DENOMINATOR 20
#define BLUR_FLT_NUMERATOR 2
#define BLUR_FLT_DENOMINATOR 20

    Mat float_gray, blur, num, den, colour;
    colour = input_colour.clone();
    // convert to floating-point image

    // numerator = img - gauss_blur(img)
    cv::GaussianBlur(colour, blur, Size(0,0), BLUR_FLT_NUMERATOR, BLUR_FLT_NUMERATOR);
    num = colour - blur;

    // denominator = sqrt(gauss_blur(img^2))
    cv::GaussianBlur(num.mul(num), blur, Size(0,0), BLUR_FLT_DENOMINATOR, BLUR_FLT_DENOMINATOR);
    cv::pow(blur, 0.5, den);

    // output = numerator / denominator
    colour = num / den;

    // normalize output into [0,1]
    cv::normalize(colour, colour, 0.0, 1.0, NORM_MINMAX, -1);

    // Display
    //namedWindow("demo", CV_WINDOW_AUTOSIZE );
    return colour;

    //imshow("demo", gray);
}

void sigmoid_mat(Mat image)
{
	float* ptr_src_index;
	ptr_src_index = image.ptr<float>(0);
    int nRows = image.rows;
    int nCols = image.cols;
    for(int i=0;i<nRows;i++)
    {
		for(int j=0;j<nCols;j++)
		{
			*ptr_src_index = 1.0/(1.0 + exp(-(*ptr_src_index)));//Sigmoid function
			ptr_src_index++;
		}
	}
}

///Måste byta frpn GRAY2RGB
//attach_weight_2_mat(ptr_M_matrix, i, visual_all_feature, sqr_of_H_nod_plus1, Hidden_nodes, Height, Width);
void attach_weight_2_mat(float* ptr_M_matrix, int i, Mat src, int sqr_of_H_nod_plus1, int  conv_depth, int Height, int Width)
{
    float *start_corner_offset = src.ptr<float>(0);
    int start_offset=0;
    float *src_zero_ptr = src.ptr<float>(0);
    float *src_ptr = src.ptr<float>(0);
    for(int j=0; j<conv_depth ; j++)
    {

        start_offset = (j/sqr_of_H_nod_plus1)*Height*src.cols*src.channels() + (j%sqr_of_H_nod_plus1)*(Width*src.channels());
        start_corner_offset = start_offset + src_zero_ptr;
        src_ptr = start_corner_offset + (i/(Width*src.channels()))*src.cols*src.channels() + (i%(Width*src.channels()));
        *src_ptr = *ptr_M_matrix;
        ptr_M_matrix++;
    }
}



void attach_in2hid_w_2_mat(float* ptr_bias_weights, Mat src, int sqr_of_H_nod_plus1, int  Hidden_nodes, int Height, int Width)
{
    float *start_corner_offset = src.ptr<float>(0);
    int start_offset=0;
    float *src_zero_ptr = src.ptr<float>(0);
    float *src_ptr = src.ptr<float>(0);
    int j=Hidden_nodes;///Hidden_nodes+0 is the patch position where in2hid weight should be visualized
    start_offset = (j/sqr_of_H_nod_plus1)*Height*src.cols + (j%sqr_of_H_nod_plus1)*Width;
    start_corner_offset = start_offset + src_zero_ptr;
    for(int i=0; i<Hidden_nodes; i++)
    {
        if(i>(Height*Width-1))
        {
            break;///The hidden nodes may be larger then one visualize patches then break so it not point out in neverland
        }
        src_ptr = start_corner_offset + (i/Width)*src.cols + (i%Width);
        *src_ptr = *ptr_bias_weights;
        ptr_bias_weights++;
    }
}

void attach_hid2out_w_2_mat(float* ptr_bias_weights, Mat src, int sqr_of_H_nod_plus1, int  Hidden_nodes, int Height, int Width)
{
    float *start_corner_offset = src.ptr<float>(0);
    int start_offset=0;
    float *src_zero_ptr = src.ptr<float>(0);
    float *src_ptr = src.ptr<float>(0);
    int j=Hidden_nodes+1;///Hidden_nodes+1 is the patch position where hid2out weight should be visualized
    start_offset = (j/sqr_of_H_nod_plus1)*Height*src.cols + (j%sqr_of_H_nod_plus1)*Width;
    start_corner_offset = start_offset + src_zero_ptr;
    for(int i=0; i<(Height*Width); i++)
    {
        src_ptr = start_corner_offset + (i/Width)*src.cols + (i%Width);
        *src_ptr = *ptr_bias_weights;
        ptr_bias_weights++;
    }
}


class Lx_attach_weight2mat
{
///    L2_sqr_of_H_nod_plus1 = sqrt(L2_conv_depth);
///    L2_sqr_of_IN_depth    = sqrt(FL2_depth);///FL2_depth = L1_conv_depth
///    L2_sqr_of_H_nod_plus1 += 1;///+1 becasue sqrt() result will be round up downwards to an integer and that may result in to small square then
///    L2_sqr_of_IN_depth += 1;///+1 becasue sqrt() result will be round up downwards to an integer and that may result in to small square then
///    L2_visual_all_feature.create(FL2_srt_size  * L2_sqr_of_IN_depth * L2_sqr_of_H_nod_plus1, FL2_srt_size  * L2_sqr_of_IN_depth * L2_sqr_of_H_nod_plus1,CV_32FC1);///This is gray because the depth is larger then "BGR"
///    L2_attach_weight_2_mat(L2_ptr_M_matrix, i, j, L2_visual_all_feature, L2_sqr_of_H_nod_plus1, L2_conv_depth, FL2_srt_size, FL2_srt_size);///
    public:
        Lx_attach_weight2mat();
        ~Lx_attach_weight2mat();
        void Xattach_weight2mat(void);
        float* Lx_ptr_M_matrix;///Must point to the Lx_weight_matrix_M[i*j][0] address
        int FLx_i_location_area;///Index
        int FLx_j_location_depth;///Index
        Mat Lx_src;///Visual image pointer
        int Lx_Hidden_nodes;///Lx_conv_depth. put in a constant
        int FL_Height;///Feature height. put in a constant
        int FL_Width;///Feature width. put in a constant
      ///  int FL_depth;///Feature depth. put in a constant
    private:
    protected:
};

Lx_attach_weight2mat::Lx_attach_weight2mat()
{
    printf("Lx_attach_weight2mat contructor\n");
}
void Lx_attach_weight2mat::Xattach_weight2mat(void)
{
    float* zero_ptr = Lx_src.ptr<float>(0);
    float* index_ptr = Lx_src.ptr<float>(0);
    for(int k=0;k<Lx_Hidden_nodes;k++)///Go throue the output depth of L2 features. One line of small squares patches with size FL_Height*FL_Width and nr of boxes length = FL_depth will belong to one output node
    {
        ///Fill only in one pixel per line (k nr of boxes line)
        index_ptr = zero_ptr + k*Lx_src.cols*FL_Height + Lx_src.cols* (FLx_i_location_area/FL_Width) + FL_Width*FLx_j_location_depth + FLx_i_location_area%FL_Width;
        *index_ptr = *Lx_ptr_M_matrix;
        Lx_ptr_M_matrix++;///Increas output depth Lx_weight_matrix_M[i*j][k]
    }
}

Lx_attach_weight2mat::~Lx_attach_weight2mat()
{
    printf("Lx_attach_weight2mat destructor\n");
}


#endif // CPP_FUNC_HPP_INCLUDED
