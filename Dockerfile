FROM python:3.9

RUN pip3 install git+https://github.com/EleutherAI/lm-evaluation-harness.git

RUN pip3 install 'lm_eval[wandb]'

# CMD ["lm_eval", "--model", "hf", \
#     "--model_args", "pretrained=EleutherAI/pythia-160m,revision=step100000,dtype=float", \
#     "--tasks", "truthfulqa", \
#     "--device", "cuda:0", \
#     "--batch_size", "6", \
#     "--output_path", "./results", \
#     "--log_samples", \
#     "--limit", "10"]
