#include <stdio.h>
#include <stdlib.h>// exit(0);

float abs_value(float signed_value)
{
	float abs_v;
	abs_v = signed_value;
	if(abs_v < 0)
	{
		abs_v = -abs_v;
	}
	return abs_v;
}

int get_CIFAR_file_size(void)
{
    int file_size=0;
    FILE *fp2;
    fp2 = fopen("data_batch_1.bin", "r");
    if (fp2 == NULL)
    {
        puts("Error while opening file data_batch_1.bin");
        exit(0);
    }

    fseek(fp2, 0L, SEEK_END);
    file_size = ftell(fp2);
    printf("file_size %d\n", file_size);
    rewind(fp2);
    fclose(fp2);
    return file_size;
}

