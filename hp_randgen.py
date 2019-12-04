#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import logging
from tqdm import trange

import torch
import torch.nn.functional as F
import numpy as np

from transformers import GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig, XLMConfig, CTRLConfig

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from transformers import XLNetLMHeadModel, XLNetTokenizer
from transformers import TransfoXLLMHeadModel, TransfoXLTokenizer
from transformers import CTRLLMHeadModel, CTRLTokenizer
from transformers import XLMWithLMHeadModel, XLMTokenizer


with open("test.txt", "r") as f:
    hp = f.read()


MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer),
    'ctrl': (CTRLLMHeadModel, CTRLTokenizer),
    'openai-gpt': (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'xlnet': (XLNetLMHeadModel, XLNetTokenizer),
    'transfo-xl': (TransfoXLLMHeadModel, TransfoXLTokenizer),
    'xlm': (XLMWithLMHeadModel, XLMTokenizer),
}


def set_seed(seed, n_gpu):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def top_k_top_p_filtering(logits):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > 0.9
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
    logits[indices_to_remove] = -float('Inf')
    return logits


def sample_sequence(model, context, device='cpu'):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(1, 1)
    generated = context
    with torch.no_grad():
        for _ in trange(100, ascii=True, ncols=100):
            inputs = {'input_ids': generated}
            outputs = model(**inputs)
            next_token_logits = outputs[0][:, -1, :]
            
            filtered_logits = top_k_top_p_filtering(next_token_logits)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)

    return generated


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#set_seed(42, torch.cuda.device_count())

model_class, tokenizer_class = MODEL_CLASSES["gpt2"]
tokenizer = tokenizer_class.from_pretrained("output2")
model = model_class.from_pretrained("output2")
model.to(device)
model.eval();


response = 'y'
while response != 'n':
    l = 1000
    rand_indx = int(np.random.choice(np.arange(len(hp)-l)))
    t = hp[rand_indx:rand_indx+l].rstrip()

    t = t[t.find(" "):-t[::-1].find(" ")].strip()

    raw_text = t[:int(len(t)/2)]
    print("________________________________PRIMING TEXT________________________________")
    print(raw_text)

    context_tokens = tokenizer.encode(raw_text, add_special_tokens=False)
    out = sample_sequence(model=model, context=context_tokens, device=device)
    out = out[:, len(context_tokens):].tolist()
    
    text = tokenizer.decode(out[0], clean_up_tokenization_spaces=True)
    print(text)

    print("___________________________________ACTUAL TEXT_______________________________")
    print(t[int(len(t)/2):])

    response = input("Continue? (y/n):")
