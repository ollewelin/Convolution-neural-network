CC = g++
CFLAGS = -g -Wall
SRCS = main.cpp cpp_func2.cpp c_func.c cpp_func.hpp cpp_func2.hpp
PROG = exe_CNN

OPENCV = `pkg-config opencv --cflags --libs`
LIBS = $(OPENCV)

$(PROG):$(SRCS)
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(LIBS)
