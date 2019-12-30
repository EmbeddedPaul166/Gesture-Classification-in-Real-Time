#Generic makefile for this project

SHELL := /bin/bash

SRC_PATH := src
BUILD_PATH := build
DLL_PATH := lib

CXX := g++

CXXFLAGS += -Wall -fPIC -O2 -c
LDFLAGS += -shared

EXECUTABLE := cuda_cv
DLL := $(DLL_PATH)/$(EXECUTABLE)

SRC_CPP_EXT := cpp
SOURCES := $(shell find $(SRC_PATH) -type f -name *.$(SRC_CPP_EXT))
OBJECTS := $(patsubst $(SRC_PATH)/%, $(BUILD_PATH)/%, $(SOURCES:.$(SRC_CPP_EXT)=.o))

INC_PATH := -I include -I /usr/include
LIB_PATH := -L /usr/local/lib -L /usr/lib
LIB := -lopencv_core -lopencv_cudawarping -lopencv_cudaimgproc -lopencv_cudaarithm -lopencv_cudabgsegm -lopencv_cudafilters

$(DLL): $(OBJECTS) 
		@mkdir -p $(DLL_PATH)	
		@echo "Linking $(DLL)..."
		$(CXX) $(LDFLAGS) $^ -o $(DLL).so $(LIB_PATH) $(INC_PATH) $(LIB)
		@echo "Done"
					
$(BUILD_PATH)/%.o: $(SRC_PATH)/%.$(SRC_CPP_EXT) 
	@mkdir -p $(BUILD_PATH)
	@echo "Compiling $<..."
	$(CXX) -c $(CXXFLAGS) $(INC_PATH) $< -o $@
	@echo ""
	
.PHONY: clean
	
clean:
		@echo "Cleaning project..."
			rm -r $(BUILD_PATH)/* $(DLL_PATH)/*
