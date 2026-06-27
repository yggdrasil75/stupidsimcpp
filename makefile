# Directories
BIN_DIR := ./bin
SRC_DIR := ./tests
IMGUI_DIR = ./imgui
OBJ_DIR := $(BIN_DIR)/obj
STB_DIR := ./stb
SHADER_DIR := ./shaders
GRID_DIR := ./util/grid

# Compiler and flags
CXX := g++
BASE_CXXFLAGS = -std=c++23 -O3 -fopenmp -march=native -I$(IMGUI_DIR) -I$(IMGUI_DIR)/backends -I$(STB_DIR) 
BASE_CXXFLAGS += -g
BASE_CXXFLAGS += `pkg-config --cflags glfw3`

LINUX_GL_LIBS = -lGL -ltbb -lvulkan
PKG_FLAGS := $(LINUX_GL_LIBS) `pkg-config --static --cflags --libs glfw3`
BASE_CXXFLAGS += $(PKG_FLAGS) -DVULKAN_SUPPORT

CFLAGS = $(BASE_CXXFLAGS)
LDFLAGS := -L./imgui -limgui -lGL

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
# SRC := $(SRC_DIR)/ptest.cpp
SRC := $(SRC_DIR)/materialtest.cpp
#SRC := $(SRC_DIR)/g2chromatic2.cpp
SRC += $(IMGUI_DIR)/imgui.cpp $(IMGUI_DIR)/imgui_demo.cpp $(IMGUI_DIR)/imgui_draw.cpp $(IMGUI_DIR)/imgui_tables.cpp $(IMGUI_DIR)/imgui_widgets.cpp
SRC += $(IMGUI_DIR)/backends/imgui_impl_glfw.cpp $(IMGUI_DIR)/backends/imgui_impl_opengl3.cpp
SRC += $(SRC_DIR)/stb_image.cpp
SRC += $(GRID_DIR)/grid3render.cpp
SRC += $(GRID_DIR)/grid3physics.cpp
OBJS = $(addprefix $(OBJ_DIR)/, $(addsuffix .o, $(basename $(notdir $(SRC)))))
UNAME_S := $(shell uname -s)
EXE := $(BIN_DIR)/g2gradc

GLSLC := glslc --target-env=vulkan1.3

SHADER_SRCS := $(SHADER_DIR)/fast_raytrace.comp $(SHADER_DIR)/pbr_raytrace.comp $(SHADER_DIR)/fast_raytrace_hw.comp $(SHADER_DIR)/pbr_raytrace_hw.comp $(SHADER_DIR)/smooth.comp $(SHADER_DIR)/blend.comp
SHADER_SRCS += $(SHADER_DIR)/sph_density.comp $(SHADER_DIR)/sph_force.comp $(SHADER_DIR)/sph_integrate.comp
SHADER_SRCS += $(SHADER_DIR)/wf_init.comp $(SHADER_DIR)/wf_args.comp $(SHADER_DIR)/wf_extend.comp $(SHADER_DIR)/wf_shade.comp $(SHADER_DIR)/wf_shadow.comp $(SHADER_DIR)/wf_finalize.comp
SHADER_SRCS += $(SHADER_DIR)/vct_mip.comp  $(SHADER_DIR)/vct_voxelize.comp 
SHADER_SPVS := $(patsubst $(SHADER_DIR)/%.comp,$(BIN_DIR)/%.spv,$(SHADER_SRCS))

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
	
$(OBJ_DIR)/%.o: $(GRID_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

all: $(EXE) $(SHADER_SPVS)
	@echo "Build complete for $(UNAME_S)"


$(BIN_DIR)/wf_init.spv $(BIN_DIR)/wf_args.spv $(BIN_DIR)/wf_extend.spv $(BIN_DIR)/wf_shade.spv $(BIN_DIR)/wf_shadow.spv $(BIN_DIR)/wf_finalize.spv: $(SHADER_DIR)/wf_common.glsl $(SHADER_DIR)/blue_sample.glsl $(SHADER_DIR)/vct_cone.glsl

$(BIN_DIR)/%.spv: $(SHADER_DIR)/%.comp
	@echo "Compiling shader $< -> $@"
	$(GLSLC) $< -o $@

$(EXE): $(OBJS)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LIBS)

clean:
	rm -f $(EXE) $(OBJS) $(SHADER_SPVS)