# Compiler and flags
CXX := g++
CXXFLAGS := -std=c++23 -O3 -march=native -I./imgui
LDFLAGS := -L./imgui -limgui -lstb -lGL
PKG_FLAGS := $(shell pkg-config --cflags --libs glfw3)

# Directories
BIN_DIR := ./bin
SRC_DIR := ./tests

# Source files
SRC := $(SRC_DIR)/g2chromatic2.cpp
TARGET := $(BIN_DIR)/g2gradc

# Default target
all: $(TARGET)

# Create binary directory if it doesn't exist
$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

# Build target
$(TARGET): $(SRC) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS) $(PKG_FLAGS)

# Run the program
run: $(TARGET)
	./$(TARGET)

# Clean build artifacts
clean:
	rm -rf $(BIN_DIR)

# Phony targets
.PHONY: all run clean