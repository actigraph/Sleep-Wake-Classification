dag.pdf: Snakefile
	snakemake -np --forceall --rulegraph | dot -Tpdf > dag.pdf

# Show diff wrt test results in test_results folder
test: test.sh
	./test.sh test

# Show diff wrt paper results in paper_results folder
validate: test.sh
	./test.sh paper
