#include "grid3eigen.hpp"

namespace Grid {

template<typename T, typename IndexType>
struct FluidMoveAction {
    std::shared_ptr<NodeData_<T, IndexType>> node;
    PointType oldPos;
    PointType newPos;
};

template<typename T, typename GasT, typename IndexType>
void Octree<T, GasT, IndexType>::stepPhysics(float dt) {
    throw std::runtime_error("NotImplementedException");
}
}