CC = g++
CFLAGS = -std=c++11 -O1
TARGET = ~/wca.exe

$(TARGET) : function.o main.o
	$(CC) $(CFLAGS) -o $(TARGET) function.o main.o

function.o : function.cpp
	$(CC) $(CFLAGS) -c -o function.o function.cpp

main.o : function.h main.cpp
	$(CC) $(CFLAGS) -c -o main.o main.cpp

clean :
	rm *.o

