CC = g++
CFLAGS = -g -Wall
SRCS = main.cpp cpp_func2.cpp c_func.c
PROG = exe_CNN

OPENCV = `pkg-config opencv --cflags --libs`
LIBS = $(OPENCV)

$(PROG):$(SRCS)
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(LIBS)
