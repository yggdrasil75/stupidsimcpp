# Directories
BIN_DIR := ./bin
SRC_DIR := ./tests
IMGUI_DIR = ./imgui
OBJ_DIR := $(BIN_DIR)/obj
STB_DIR := ./stb

# Compiler and flags
CXX := g++
BASE_CXXFLAGS = -std=c++23 -O3 -I$(IMGUI_DIR) -I$(IMGUI_DIR)/backends -I$(STB_DIR)
BASE_CXXFLAGS += `pkg-config --cflags glfw3`
CFLAGS = $(BASE_CXXFLAGS)
LDFLAGS := -L./imgui -limgui -lGL
LINUX_GL_LIBS = -lGL -ltbb
PKG_FLAGS := $(LINUX_GL_LIBS) `pkg-config --static --cflags --libs glfw3`
BASE_CXXFLAGS += $(PKG_FLAGS)

# Test if AVX is supported (run once, store result)
AVX_SUPPORTED := $(shell echo "int main(){}" | $(CXX) -mavx -x c++ -o /dev/null - 2>/dev/null && echo "yes" || echo "no")
SSE2_SUPPORTED := $(shell echo "int main(){}" | $(CXX) -msse2 -x c++ -o /dev/null - 2>/dev/null && echo "yes" || echo "no")

# Set SIMD flags based on detection
ifeq ($(AVX_SUPPORTED),yes)
    SIMD_CXXFLAGS = -mavx2 -mfma -DAVX
    $(info Building with AVX support)
else ifeq ($(SSE2_SUPPORTED),yes)
    SIMD_CXXFLAGS = -msse2 -DSSE
    $(info Building with SSE2 support (no AVX))
else
    SIMD_CXXFLAGS = -DNO_SIMD
    $(warning No SIMD support detected, building scalar version)
endif

CXXFLAGS = $(BASE_CXXFLAGS) $(SIMD_CXXFLAGS)

# Source files
SRC := $(SRC_DIR)/g3etest.cpp
#SRC := $(SRC_DIR)/g2chromatic2.cpp
SRC += $(IMGUI_DIR)/imgui.cpp $(IMGUI_DIR)/imgui_demo.cpp $(IMGUI_DIR)/imgui_draw.cpp $(IMGUI_DIR)/imgui_tables.cpp $(IMGUI_DIR)/imgui_widgets.cpp
SRC += $(IMGUI_DIR)/backends/imgui_impl_glfw.cpp $(IMGUI_DIR)/backends/imgui_impl_opengl3.cpp
SRC += $(SRC_DIR)/stb_image.cpp
OBJS = $(addprefix $(OBJ_DIR)/, $(addsuffix .o, $(basename $(notdir $(SRC)))))
UNAME_S := $(shell uname -s)
EXE := $(BIN_DIR)/g2gradc

$(shell mkdir -p $(OBJ_DIR))

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(OBJ_DIR)/%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(OBJ_DIR)/%.o: $(IMGUI_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(OBJ_DIR)/%.o: $(IMGUI_DIR)/backends/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(OBJ_DIR)/%.o: $(STB_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

all: $(EXE)
	@echo "Build complete for $(UNAME_S)"

$(EXE): $(OBJS)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LIBS)

clean:
	rm -f $(EXE) $(OBJS)