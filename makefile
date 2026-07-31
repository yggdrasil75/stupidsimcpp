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

CHAR_CXXFLAGS = -std=c++23 -O3 -fopenmp -march=native -g $(SIMD_CXXFLAGS)
CHAR_LDFLAGS = -ltbb

# Source files
# SRC := $(SRC_DIR)/ptest.cpp
# SRC := $(SRC_DIR)/naturetest.cpp
SRC := $(SRC_DIR)/physicsroom.cpp
# SRC := $(SRC_DIR)/materialtestv2.cpp
SUPPORT_SRC := $(IMGUI_DIR)/imgui.cpp $(IMGUI_DIR)/imgui_demo.cpp $(IMGUI_DIR)/imgui_draw.cpp $(IMGUI_DIR)/imgui_tables.cpp $(IMGUI_DIR)/imgui_widgets.cpp
SUPPORT_SRC += $(IMGUI_DIR)/backends/imgui_impl_glfw.cpp $(IMGUI_DIR)/backends/imgui_impl_opengl3.cpp
SUPPORT_SRC += $(SRC_DIR)/stb_image.cpp
SUPPORT_SRC += $(GRID_DIR)/grid3render.cpp
SUPPORT_SRC += $(GRID_DIR)/grid3physics.cpp
SUPPORT_SRC += $(GRID_DIR)/grid3edit.cpp
SRC += $(SUPPORT_SRC)
SUPPORT_OBJS = $(addprefix $(OBJ_DIR)/, $(addsuffix .o, $(basename $(notdir $(SUPPORT_SRC)))))
OBJS = $(addprefix $(OBJ_DIR)/, $(addsuffix .o, $(basename $(notdir $(SRC)))))
EDITOR_OBJS = $(OBJ_DIR)/editor.o $(SUPPORT_OBJS)
CHARACTER_OBJS = $(OBJ_DIR)/charactertest.o
UNAME_S := $(shell uname -s)
EXE := $(BIN_DIR)/g2gradc
EDITOR_EXE := $(BIN_DIR)/editor
CHARACTER_EXE := $(BIN_DIR)/charactertest

GLSLC := glslc --target-env=vulkan1.3

SHADER_SRCS := $(SHADER_DIR)/fast_raytrace_hw.comp $(SHADER_DIR)/smooth.comp $(SHADER_DIR)/blend.comp
SHADER_SRCS += $(SHADER_DIR)/wf_init.comp $(SHADER_DIR)/wf_args.comp $(SHADER_DIR)/wf_extend.comp $(SHADER_DIR)/wf_shade.comp $(SHADER_DIR)/wf_shadow.comp $(SHADER_DIR)/wf_finalize.comp
SHADER_SRCS += $(SHADER_DIR)/vct_mip.comp $(SHADER_DIR)/vct_voxelize.comp $(SHADER_DIR)/guided_coeff.comp $(SHADER_DIR)/aabb_build.comp
SHADER_SRCS += $(SHADER_DIR)/ddgi_update.comp
SHADER_SRCS += $(SHADER_DIR)/svgf_reproject.comp $(SHADER_DIR)/svgf_moments.comp
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

all: $(EXE) $(EDITOR_EXE) $(CHARACTER_EXE) $(SHADER_SPVS)
	@echo "Build complete for $(UNAME_S)"

$(BIN_DIR)/wf_init.spv $(BIN_DIR)/wf_args.spv $(BIN_DIR)/wf_extend.spv $(BIN_DIR)/wf_shade.spv $(BIN_DIR)/wf_shadow.spv $(BIN_DIR)/wf_finalize.spv $(BIN_DIR)/ddgi_update.spv: $(SHADER_DIR)/wf_common.glsl $(SHADER_DIR)/vct_cone.glsl

$(BIN_DIR)/smooth.spv $(BIN_DIR)/svgf_reproject.spv $(BIN_DIR)/svgf_moments.spv: $(SHADER_DIR)/svgf_common.glsl

$(BIN_DIR)/%.spv: $(SHADER_DIR)/%.comp
	@echo "Compiling shader $< -> $@"
	$(GLSLC) $< -o $@

$(EXE): $(OBJS)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LIBS)

$(EDITOR_EXE): $(EDITOR_OBJS)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LIBS)

editor: $(EDITOR_EXE) $(SHADER_SPVS)
	@echo "Editor build complete for $(UNAME_S)"

$(CHARACTER_EXE): $(CHARACTER_OBJS)
	$(CXX) -o $@ $^ $(CHAR_CXXFLAGS) $(CHAR_LDFLAGS)

$(OBJ_DIR)/charactertest.o: $(SRC_DIR)/charactertest.cpp
	$(CXX) $(CHAR_CXXFLAGS) -c -o $@ $<

charactertest: $(CHARACTER_EXE)
	@echo "Character generator build complete for $(UNAME_S)"

clean:
	rm -f $(EXE) $(EDITOR_EXE) $(CHARACTER_EXE) $(OBJS) $(EDITOR_OBJS) $(CHARACTER_OBJS) $(SHADER_SPVS)