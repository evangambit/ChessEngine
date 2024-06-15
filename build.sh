g++ src/game/*.cpp \
-std=c++20 \
-O3 \
-DNDEBUG \
-I src $@

# -I /opt/homebrew/Cellar/eigen/3.4.0_1/include \
# protoc protos/weights.proto --cpp_out src --python_out ./

# Production:
# -O3 -DNDEBUG
# 
# Debug:
# -rdynamic -g1
