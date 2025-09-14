

## Building

cmake . -DCMAKE_BUILD_TYPE=Release -DNNUE_EVAL=1
make

## Training

$ python3 generate.py --engine /usr/games/stockfish --depth=6
$ sqlite3 data/de6-md2/db.sqlite3 'select * from positions' > data/de6-md2/pos.txt
$ # Shuffle with one of these commands (second command works for very large files)
$ shuf data/de6-md2/pos.txt > data/de6-md2/pos.shuf.txt
$ awk 'BEGIN{srand();} {printf "%06d %s\n", rand()*1000000, $0;}' data/de6-md2/pos.txt | sort -n | cut -c8- > data/de6-md2/pos.shuf.txt

$ python3 generate.py --engine /usr/games/stockfish --depth=7
$ sqlite3 data/de7-md2/db.sqlite3 'select * from positions' > data/de7-md2/pos.txt

$ sh build.sh src/make_tables.cpp -o make_tables --std=c++20 -DNNUE_EVAL=1 -DNO_TORCH=1 -DFEATURES=0
$ ./make_tables data/de6-md2/pos.shuf.txt data/de6-md2/x
$ ./make_tables data/de7-md2/pos.txt data/de7-md2/x

$ python3 -i nnue-train.py

## Profiling

```
brew install gperftools
go install github.com/google/pprof@latest

sh build.sh src/uci.cpp -o uci -DNNUE_EVAL=1 -DPRINT_PV_CHANGES=0 -DDEBUG_TT=0 -L$(brew --prefix gperftools)/lib -lprofiler

CPUPROFILE=/tmp/prof.out ./uci "move e2e4 c7c5 g1f3 d7d6" "go depth 8" "lazyquit"

~/go/bin/pprof -png ./uci /tmp/prof.out
```