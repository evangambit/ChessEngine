g++ src/game/*.cpp \
-std=c++20 \
-O3 \
-DNDEBUG \
-I src $@

# protoc protos/weights.proto --cpp_out src --python_out ./
