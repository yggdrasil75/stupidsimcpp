# Directories
BIN_DIR := ./bin
SRC_DIR := ./tests
IMGUI_DIR = ./imgui
OBJ_DIR := $(BIN_DIR)/obj
STB_DIR := ./stb

# Compiler and flags
CXX := g++
CXXFLAGS = -std=c++23 -I$(IMGUI_DIR) -I$(IMGUI_DIR)/backends -I$(STB_DIR)
CXXFLAGS += `pkg-config --cflags glfw3`
CFLAGS = $(CXXFLAGS)
LDFLAGS := -L./imgui -limgui -lGL
LINUX_GL_LIBS = -lGL
PKG_FLAGS := $(LINUX_GL_LIBS) `pkg-config --static --cflags --libs glfw3`
CXXFLAGS += $(PKG_FLAGS)

# Source files
SRC := $(SRC_DIR)/g2temp.cpp
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
	@echo Build complete for $(ECHO_MESSAGE)

$(EXE): $(OBJS)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LIBS)

clean:
	rm -f $(EXE) $(OBJS)