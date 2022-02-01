SRCS = $(wildcard *.gnu)
PDF = $(patsubst %.gnu,%.pdf,$(SRCS))
PS = $(patsubst %.gnu,%.ps,$(SRCS))

all: $(PDF) $(PS)

%.ps: %.gnu *.dat *.gnuh
	gnuplot $<

%.pdf: %.ps
	ps2pdf $*.ps
