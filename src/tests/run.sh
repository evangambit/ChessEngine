g++ -std=c++20 \
-I/opt/homebrew/Cellar/googletest/1.14.0/include \
-L/opt/homebrew/Cellar/googletest/1.14.0/lib -lgtest \
-I /opt/homebrew/Cellar/eigen/3.4.0_1/include \
-o test src/game/*.cpp $@ && ./test
