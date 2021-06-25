import ast
import torch.nn.functional as F
from datasets import load_dataset

def encode_batch(batch):
  """Encodes a batch of input data using the model tokenizer."""
  return tokenizer(batch[tokenize_col], max_length=max_n_tokens, truncation=True, padding="max_length")

def encode_labels(example):
    r_tags = []
    count = 0
    token2word = []
    tokens = ast.literal_eval(example["tokens"])
    labels = ast.literal_eval(example["labels"])

    for index, token in enumerate(tokenizer.tokenize(" ".join(tokens))):

        if token.startswith("##") or (token in tokens[index - count - 1].lower() and index - count - 1 >= 0):
            # if the token is part of a larger token and not the first we need to differ 
            # if it is a B (beginning) label the next one needs to ba assigned a I (intermediate) label
            # otherwise they can be labeled the same
            r_tags.append(r_tags[-1])
            count += 1
        else:
            
            r_tags.append(labels[index - count])

        token2word.append(index - count)


    r_tags = torch.tensor(r_tags)
    labels = {}
    # Pad token to maximum length for using batches
    labels["labels"] = F.pad(r_tags, pad=(1, 100 - r_tags.shape[0]), mode='constant', value=0)
    # Truncate if the document is too long
    labels["labels"] = labels["labels"][:100]

    return labels