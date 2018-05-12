CC = g++

PARAM += -g -std=c++11 -Wall -Wextra 

CFLAGS += $(shell pkg-config --cflags --libs opencv)


all: gen_test run_test

gen_test: gen_test.cpp
	$(CC) $(PARAM) gen_test.cpp -o gen_test $(CFLAGS)

run_test: run_test.cpp
	$(CC) $(PARAM) run_test.cpp -o run_test $(CFLAGS)


clean: ; rm data/left/* data/right/*
