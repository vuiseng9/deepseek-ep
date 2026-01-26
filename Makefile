ngpu = $(shell nvidia-smi -L | wc -l)
intranode_run = torchrun --standalone --nproc-per-node $(ngpu)



test_intranode:
	python tests/test_intranode.py --num-processes $(ngpu)