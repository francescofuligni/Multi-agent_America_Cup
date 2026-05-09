TEX := main.tex
PDF := Multi-agent_America_Cup_report.pdf
OUTDIR := build

.PHONY: all clean

all:
	mkdir -p $(OUTDIR)
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=$(OUTDIR) $(TEX)
	cp $(OUTDIR)/main.pdf ../$(PDF)

clean:
	rm -rf $(OUTDIR)
	rm -f ../$(PDF)
