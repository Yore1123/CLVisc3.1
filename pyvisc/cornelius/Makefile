cornelius.o:cornelius.h cornelius.cpp
	g++  -O3 -static -c cornelius.cpp
hypersf.o:hypersf.h hypersf.cpp
	h5c++ -O3 -static -c hypersf.cpp
main.o:main.cpp
	h5c++ -O3 -static -c main.cpp
main:main.o hypersf.o cornelius.o
	h5c++  -O3  main.o hypersf.o cornelius.o -o main 

clean:
	rm -f *.o main 
