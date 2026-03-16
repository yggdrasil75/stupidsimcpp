#include "../materials/materials.hpp"
#include <memory>
#include "../basicdefines.hpp"

template<typename T, typename IndexSize = uint16_t, typename high = double, typename medium = float, typename low = Eigen::half>
class Octree;

template<typename T, typename IndexSize, typename high, typename medium, typename low>
class NodeData {
    T data;
    private:
        PointLow position;
        Eigen::Quaternion<low> orientation;
        int objectId;
        low size;

        IndexSize colorIDX;
        IndexSize materialIDX;
        uint8_t flags;
        
    public:
        NodeData(const T& data, const PointLow& pos, bool visible, IndexSize colorIDX, float size = 0.01f,
                bool active = true, int objectId = -1, IndexSize materialIdx = 0, bool staticnode = false)
                : data(data), position(pos), objectId(objectId), size(static_cast<low>(size)), 
                colorIDX(colorIDX), materialIDX(materialIdx), flags(0) {
                    setActive(active);
                    setVisible(visible);
                    setStatic(staticnode);
                    orientation.setIdentity();
                }
        
        NodeData() : objectId(-1), size(0.0), colorIDX(0), materialIDX(0), flags(0) {
            orientation.setIdentity();
        }

        inline T getData() const {
            return data;
        }

        inline void setData(const T& newData) {
            data = newData;
        }

        inline bool isActive() const {
            return flags & ACTIVE_BIT;
        }
        inline bool isVISIBLE() const {
            return flags & VISIBLE_BIT;
        }
        inline bool isStatic() const {
            return flags & STATIC_BIT;
        }
        inline bool isActiveAndVisible() const {
            return ((flags & ACTIVE_BIT) != 0) && ((flags & VISIBLE_BIT) != 0);
        }
        
        inline void setActive(bool val) {
            val ? (flags |= ACTIVE_BIT) : (flags &= ~ACTIVE_BIT);
        }
        inline void setVisible(bool val) {
            val ? (flags |= VISIBLE_BIT) : (flags &= ~VISIBLE_BIT);
        }
        inline void setStatic(bool val) {
            val ? (flags |= STATIC_BIT) : (flags &= ~STATIC_BIT);
        }

        inline IndexSize getColorIDX() const {
            return colorIDX;
        }

        inline IndexSize getMaterialIDX() const {
            return materialIDX;
        }

        inline void setColorIdx(IndexSize idx) {
            colorIDX = idx;
        }

        inline void setMaterialIDX(IndexSize idx) {
            materialIDX = idx;
        }

        inline PointLow getPosition() const { 
            return position;
        }

        inline void setPosition(const PointLow& pos) { 
            position = pos;
        }

        inline void setOrientation(const Eigen::Quaternion<low>& rot) {
            orientation = rot;
        }

        inline void setObjectId(int id) {
            objectId = id;
        }

        inline int getObjectId() const {
            return objectId;
        }

        inline void setSize(low s) {
            size = s;
        }

        PointLow getHalfSize() const {
            return PointLow(size * 0.5f, size * 0.5f, size * 0.5f);
        }
        
        OBoundingBox getCubeBounds(const PointHigh& nodeCenter) const {
            OBoundingBox obb;
            obb.center = nodeCenter.template cast<medium>() + position.template cast<medium>();
            obb.extents = getHalfSize().template cast<medium>();
            obb.orientation = orientation;
            return obb;
        }

        PointMedium center(const PointHigh& nodeCenter) const {
            return nodeCenter + position.template cast<medium>();
        }

        void serialize(std::ostream& os) const {
            os.write(reinterpret_cast<const char*>(&data), sizeof(T));
            os.write(reinterpret_cast<const char*>(&position), sizeof(position));
            os.write(reinterpret_cast<const char*>(&orientation), sizeof(orientation));
            os.write(reinterpret_cast<const char*>(&objectId), sizeof(objectId));
            os.write(reinterpret_cast<const char*>(&size), sizeof(size));
            os.write(reinterpret_cast<const char*>(&colorIDX), sizeof(colorIDX));
            os.write(reinterpret_cast<const char*>(&materialIDX), sizeof(materialIDX));
            os.write(reinterpret_cast<const char*>(&flags), sizeof(flags));
        }

        void deserialize(std::istream& is) {
            is.read(reinterpret_cast<char*>(&data), sizeof(T));
            is.read(reinterpret_cast<char*>(&position), sizeof(position));
            is.read(reinterpret_cast<char*>(&orientation), sizeof(orientation));
            is.read(reinterpret_cast<char*>(&objectId), sizeof(objectId));
            is.read(reinterpret_cast<char*>(&size), sizeof(size));
            is.read(reinterpret_cast<char*>(&colorIDX), sizeof(colorIDX));
            is.read(reinterpret_cast<char*>(&materialIDX), sizeof(materialIDX));
            is.read(reinterpret_cast<char*>(&flags), sizeof(flags));
        }
};
