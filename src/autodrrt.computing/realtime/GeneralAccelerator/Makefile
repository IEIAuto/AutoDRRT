# makefile 
VERSION = 1.0.0
CC = g++
CPPFLAGS = -std=c++14 
TARGET = GeneralAccelerator
DEBUG = -Wall -c
src_path = ./ ./src/
src = $(foreach dir, $(src_path), $(wildcard $(dir)*.cpp))

obj_name = $(patsubst %.cpp, %.o, $(src))

INC = -I./
INC += -I./include/ 
header = $(foreach dir, $(src_path), $(wildcard $(dir)*.hpp))


$(TARGET) : $(obj_name) 
	$(CC) $(obj_name) $(header) -o $(TARGET)	 
	@echo "\nmakefile compile version :$(VERSION), target :$(TARGET)\n"
	rm $(obj_name)

%.o : %.c
	$(CC) $(DEBUG) $(INC) $^ -o $@ 
%.o : %.cpp
	$(CC) $(DEBUG) $(INC) $^ -o $@ 
	
.PHONY :
clean :
	@echo "start clean target and objects \n"
	rm $(TARGET) $(obj_name)
