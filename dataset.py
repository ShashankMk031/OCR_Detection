from datasets import load_dataset

# From hugging face , primary dataset 
ds = load_dataset("wendlerc/CaptionedSynthText")

# secondary dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("dlxjj/ICDAR2015")