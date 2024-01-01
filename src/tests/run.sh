g++ -std=c++20 \
-I/opt/homebrew/Cellar/googletest/1.14.0/include \
-L/opt/homebrew/Cellar/googletest/1.14.0/lib -lgtest \
-o test src/game/*.cpp $@ && ./test
