CC      ?= gcc
CFLAGS  ?= -O2 -Wall -Wextra -pedantic -std=c11
LDFLAGS ?=

LIBS = -ltree-sitter -ltree-sitter-c

TARGET = archmap
SRC    = main.c

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LIBS)

clean:
	rm -f $(TARGET)
