g++ src/game/*.cpp src/protos/weights.pb.cc \
-std=c++20 \
-O3 \
-DNDEBUG \
-lprotobuf \
-L /opt/homebrew/Cellar/protobuf/24.4/lib \
-L /opt/homebrew/Cellar/abseil/20230802.1/lib \
-I /opt/homebrew/Cellar/protobuf/24.4/include \
-I /opt/homebrew/Cellar/abseil/20230802.1/include \
-I src $@
