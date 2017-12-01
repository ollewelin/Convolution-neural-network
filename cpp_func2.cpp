#include "cpp_func2.hpp"
#include <stdio.h>
#include <cstdlib>/// rand()
#include <math.h>  // exp() sqrt()
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>

///time delay
#include <iostream>
#include <time.h>
using namespace std;
void Sleep(float s)
{
    int sec = int(s*1000000);
    usleep(sec);
}
///

int cpp_func2::kbhit(void)
{

    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

    ch = getchar();

    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    fcntl(STDIN_FILENO, F_SETFL, oldf);

    if(ch != EOF)
    {
        ungetc(ch, stdin);
        return 1;
    }

    return 0;
}

cpp_func2::cpp_func2()
{
    //ctor
}

///=========Regarding standard deviation calculation ============
void cpp_func2::do_mean_add_std_dev(float input_data)
{
    sum_of_data += input_data;
    nr_of_mean_add_ittr++;
}

void cpp_func2::do_mean_calc_to_std_dev(void)
{
    if(nr_of_mean_add_ittr>0)
    {
        std_mean_value = sum_of_data / nr_of_mean_add_ittr;
    }
    else
    {
        std_mean_value = 0.0f;
        printf("Error! No data point add itteration before do mean calculation\n");
    }
}
void cpp_func2::do_deviation_add_std_dev(float input_data)
{
    float diff=0.0f;
    diff = input_data - std_mean_value;
    sum_of_sqr_diff += diff * diff;///Sum up the square of the differens
    nr_of_varaince_add_ittr++;
}

void cpp_func2::do_std_deviation_calc(void)
{
    if(nr_of_varaince_add_ittr>0)
    {
        if(nr_of_varaince_add_ittr == nr_of_mean_add_ittr)
        {
            variance = sum_of_sqr_diff / nr_of_varaince_add_ittr;
        }
        else
        {
            printf("Error! the itteration tunrs of mean calc data and varaince calc data is NOT equal\n");
            variance = 0.0f;
        }
        std_deviation = sqrt(variance);
    }
    else
    {
        std_mean_value = 0.0f;
        printf("Error! No data point add itteration before standarad deviation calculation\n");
    }
    sum_of_data=0.0f;///Clear this data to Prepare for next calculation operation
    sum_of_sqr_diff=0.0f;
    nr_of_mean_add_ittr=0;///Clear this data to Prepare for next calculation operation
    nr_of_varaince_add_ittr=0;///Clear this data to Prepare for next calculation operation
}
///=============End Regarding standard deviation calculation ===============

void cpp_func2::init(void)
{
    noise_pos = new int [nr_of_positions];
    L2_noise_pos = new int [L2_nr_of_positions];
}

void cpp_func2::rand_input_data_pos(void)
{
    nr_of_noise_rand_ittr=0;
    noise_p_counter=0;
    noise_ratio=0.0f;
    rand_pix_pos=0;

    for(int n=0; n<nr_of_positions; n++)
    {
        noise_pos[n] = 0;
    }

    while(noise_ratio < (noise_percent*0.01f))
    {
        rand_pix_pos = (int) (rand() % nr_of_positions);
        if(noise_pos[rand_pix_pos] == 0)
        {
            noise_p_counter++;
        }
        noise_pos[rand_pix_pos] = 1;
        noise_ratio = ((float)noise_p_counter) / ((float)nr_of_positions);
        nr_of_noise_rand_ittr++;
        if(nr_of_noise_rand_ittr > 2*nr_of_positions)
        {
            printf("give up fill random up noise this turn\n");
            printf("noise_ratio %f\n", noise_ratio);
            break;
        }
    }
}
void cpp_func2::L2_rand_input_data_pos(void)
{
    nr_of_noise_rand_ittr=0;
    noise_p_counter=0;
    noise_ratio=0.0f;
    rand_pix_pos=0;

    for(int n=0; n<L2_nr_of_positions; n++)
    {
        L2_noise_pos[n] = 0;
    }

    while(noise_ratio < (L2_noise_percent*0.01f))
    {
        rand_pix_pos = (int) (rand() % L2_nr_of_positions);
        if(L2_noise_pos[rand_pix_pos] == 0)
        {
            noise_p_counter++;
        }
        L2_noise_pos[rand_pix_pos] = 1;
        noise_ratio = ((float)noise_p_counter) / ((float)L2_nr_of_positions);
        nr_of_noise_rand_ittr++;
        if(nr_of_noise_rand_ittr > 2*L2_nr_of_positions)
        {
            printf("give up fill random up L2 noise this turn\n");
            printf("L2 noise_ratio %f\n", noise_ratio);
            break;
        }
    }
}
void cpp_func2::L3_rand_input_data_pos(void)
{
    nr_of_noise_rand_ittr=0;
    noise_p_counter=0;
    noise_ratio=0.0f;
    rand_pix_pos=0;

    for(int n=0; n<L3_nr_of_positions; n++)
    {
        L3_noise_pos[n] = 0;
    }

    while(noise_ratio < (L3_noise_percent*0.01f))
    {
        rand_pix_pos = (int) (rand() % L3_nr_of_positions);
        if(L3_noise_pos[rand_pix_pos] == 0)
        {
            noise_p_counter++;
        }
        L3_noise_pos[rand_pix_pos] = 1;
        noise_ratio = ((float)noise_p_counter) / ((float)L3_nr_of_positions);
        nr_of_noise_rand_ittr++;
        if(nr_of_noise_rand_ittr > 2*L3_nr_of_positions)
        {
            printf("give up fill random up L3 noise this turn\n");
            printf("L3 noise_ratio %f\n", noise_ratio);
            break;
        }
    }
}

void cpp_func2::print_help(void)
{
            printf("Hit <?> or <Space>  show HELP menu\n");
            printf("Hit <Space> TOGGLE start training or stop and delay for show\n");
            printf("Hit <A> to save all weights to weight_matrix_M.dat file\n");
            printf("Hit <B> TOGGLE Autoencoder L1 ON/OFF\n");
            printf("Hit <C> TOGGLE Autoencoder L2 ON/OFF\n");
            printf("Hit <D> TOGGLE Convolution L1 ON/OFF\n");
            printf("Hit <E> TOGGLE Convolution L2 ON/OFF\n");
            printf("Hit <F> TOGGLE Autoencoder L3 ON/OFF\n");
            printf("Hit <G> TOGGLE Convolution L3 ON/OFF\n");
            printf("Hit <H> TOGGLE full_conn_backprop start/stop logistic regression learning\n");
            printf("Hit <I> TOGGLE Fine tune Feature L2 from fc backprop (will lock fc weight updates) full_conn_backprop must be ON \n");
            printf("Hit <J> \n");
            printf("Hit <K> \n");
            printf("Hit <L> \n");
            printf("Hit <M> Clear fully connected weights with random data init_random_fc_weights\n");
            printf("Hit <N> Visualize L3 Patches\n");
}

void cpp_func2::keyboard_event(void)
{
       char keyboard;
        if(kbhit())
        {
            keyboard = getchar();
            if(keyboard== ' ')
            {
                print_help();
                if(started == 1)
                {
                    started = 0;
                    printf("Stop training\n");
                    printf("Training stop now only feed forward\n");
                }
                else
                {
                    started = 1;
                    printf("Start training\n");
                }
            }

            if(keyboard== '?')
            {
                print_help();
                started = 0;
                printf("Stop training\n");
                printf("Training stop now only feed forward\n");
            }
            if(keyboard== 'A' || keyboard== 'a')
            {
                save_L1_weights=1;
            }
            if(keyboard== 'B' || keyboard== 'b')
            {
                if(L1_autoencoder_ON==0)
                {
                    L1_autoencoder_ON=1;
                }
                else
                {
                    L1_autoencoder_ON=0;
                }
                printf("L1_autoencoder_ON=%d\n", L1_autoencoder_ON);
            }
            if(keyboard== 'C' || keyboard== 'c')
            {
                if(L2_autoencoder_ON==0)
                {
                    L2_autoencoder_ON=1;
                }
                else
                {
                    L2_autoencoder_ON=0;
                }
                printf("L2_autoencoder_ON=%d\n", L2_autoencoder_ON);
            }
            if(keyboard== 'D' || keyboard== 'd')
            {
                if(L1_convolution_ON==0)
                {
                    L1_convolution_ON=1;
                }
                else
                {
                    L1_convolution_ON=0;
                }
                printf("L1_convolution_ON=%d\n", L1_convolution_ON);
            }
           if(keyboard== 'E' || keyboard== 'e')
            {
                if(L2_convolution_ON==0)
                {
                    L2_convolution_ON=1;
                }
                else
                {
                    L2_convolution_ON=0;
                }
                printf("L2_convolution_ON=%d\n", L2_convolution_ON);
            }
           if(keyboard== 'F' || keyboard== 'f')
            {
                if(L3_autoencoder_ON==0)
                {
                    L3_autoencoder_ON=1;
                }
                else
                {
                    L3_autoencoder_ON=0;
                }
                printf("L3_autoencoder_ON=%d\n", L3_autoencoder_ON);
            }

           if(keyboard== 'G' || keyboard== 'g')
            {
                if(L3_convolution_ON==0)
                {
                    L3_convolution_ON=1;
                }
                else
                {
                    L3_convolution_ON=0;
                }
                printf("L3_convolution_ON=%d\n", L3_convolution_ON);
            }

           if(keyboard== 'N' || keyboard== 'n')
            {
                if(Start_Visualize_L3==0)
                {
                    Start_Visualize_L3=1;
                }
                else
                {
                    Start_Visualize_L3=0;
                }
                printf("Start_Visualize_L3=%d\n", Start_Visualize_L3);
            }
           if(keyboard== 'H' || keyboard== 'h')
            {
                if(full_conn_backprop==0)
                {
                    full_conn_backprop=1;
                }
                else
                {
                    full_conn_backprop=0;
                }
                printf("full_conn_backprop=%d\n", full_conn_backprop);
                printf("finetune=%d\n", finetune);
            }
           if(keyboard== 'I' || keyboard== 'i')
            {
                if(finetune==0)
                {
                    finetune=1;
                }
                else
                {
                    finetune=0;
                }
                printf("finetune=%d\n", finetune);
            }

           if(keyboard== 'M' || keyboard== 'm')
            {
                init_random_fc_weights = 1;

            }

            Sleep(1);///time to read

        }
}


cpp_func2::~cpp_func2()
{
    //dtor
    if(noise_pos)
        delete[] noise_pos;
}

