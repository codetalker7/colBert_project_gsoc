tmux -u
conda activate colbert
cd colbert_project_gsoc
CUDA_VISIBLE_DEVICES=0 python3

## move code to GPU directory
scp -J chaudhary@cmi -r ./*  chaudhary@cmigpu:/home/chaudhary/colBert_project_gsoc/