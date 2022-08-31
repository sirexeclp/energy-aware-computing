bib_file=/home/felix/bib/energy.bib

report.pdf: report.md
	pandoc --filter pandoc-citeproc $< -o $@

.PHONY: package

clean:
	rm report.pdf

.PHONY: clean

