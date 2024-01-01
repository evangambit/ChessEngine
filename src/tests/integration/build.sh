g++ src/tests/kpvk.cpp src/game/kpvk.cpp src/game/geometry.cpp src/game/utils.cpp -std=c++20 -I/opt/homebrew/Cellar/googletest/1.14.0/include -L/opt/homebrew/Cellar/googletest/1.14.0/lib -lgtest -o test

fn="src/tests/integration/draws.cpp"

g++ src/game/*.cpp \
${fn} \
-std=c++20 \
-I/opt/homebrew/Cellar/googletest/1.14.0/include \
-L/opt/homebrew/Cellar/googletest/1.14.0/lib \
-lgtest -o test
