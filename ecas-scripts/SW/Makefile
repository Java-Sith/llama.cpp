SRC := $(wildcard *.cpp)
EXE := $(SRC:.cpp=)
CFLAGS := -I${XILINX_XRT}/include -Wall -std=c++17 -O3 -I. -I./hls-types/include
LDFLAGS := -L${XILINX_XRT}/lib -pthread -lxrt_core -lxrt_coreutil

.PHONY: all clean

all: $(EXE)

$(EXE): $(SRC)
	$(CXX) -o $@ $< $(CFLAGS) $(LDFLAGS)

clean:
	$(RM) $(EXE)
