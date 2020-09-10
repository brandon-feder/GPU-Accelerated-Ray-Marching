echo " === Compiling === ";
nvcc main.cu -o ./bin/main.out -lSDL2 --expt-extended-lambda;

echo " === Calling === ";
./bin/main.out;
rm ./bin/main.out;
