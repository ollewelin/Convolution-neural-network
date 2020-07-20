#ifndef CPP_FUNC2_HPP
#define CPP_FUNC2_HPP

#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>


class cpp_func2
{
    public:
        cpp_func2();
        void init(void);///Setting up a new 1D array noise_pos[nr_of_positions]
        void rand_input_data_pos(void);
        void L2_rand_input_data_pos(void);
        void L3_rand_input_data_pos(void);
///=========Regarding standard deviation calculation ============
        void do_mean_add_std_dev(float);///Step 1: loop thorue all data point throue this function to make mean value
        void do_mean_calc_to_std_dev(void);///Step 2: Do the mean calculation operantion
        void do_deviation_add_std_dev(float);///Step 3: loop thorue all data point throue this function to make variance value
        void do_std_deviation_calc(void);///Step 4: do the last step in the standard deviation calculation to run this function
        float std_mean_value=0.0f;
        float variance=0.0f;
        float std_deviation=0.0f;
///==============================================================
        float noise_percent;
        float L2_noise_percent;
        float L3_noise_percent;
        int nr_of_positions;
        int L2_nr_of_positions;
        int L3_nr_of_positions;
        int *noise_pos;///Pointer to make a new 1D array noise_pos[nr_of_positions]
        int *L2_noise_pos;///Pointer to make a new 1D array noise_pos[nr_of_positions]
        int *L3_noise_pos;///Pointer to make a new 1D array noise_pos[nr_of_positions]

        void print_help(void);
        void keyboard_event(void);
        int kbhit(void);
        int started=0;
        int finetune=0;///1= do finetune of features throue backprop from fc
        int save_L1_weights=0;
        int L1_autoencoder_ON=0;
        int L2_autoencoder_ON=0;
        int L3_autoencoder_ON=0;
        int L1_convolution_ON=0;
        int L2_convolution_ON=0;
        int L3_convolution_ON=0;
        int Start_Visualize_L3=0;
        int init_random_fc_weights=0;
        int full_conn_backprop=0;
        int validation=0;
        virtual ~cpp_func2();
    protected:
    private:
    int nr_of_noise_rand_ittr;
    int noise_p_counter;
    float noise_ratio;
    int rand_pix_pos;
///=========Regarding standard deviation calculation ============
        float sum_of_data=0.0f;
        float sum_of_sqr_diff=0.0f;
        int nr_of_mean_add_ittr=0;
        int nr_of_varaince_add_ittr=0;
///==============================================================
    struct termios oldt, newt;
    int ch;
    int oldf;

};

#endif // CPP_FUNC2_HPP
