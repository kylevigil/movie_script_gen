#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import logging
from tqdm import trange

import torch
import torch.nn.functional as F
import numpy as np
import pickle

from transformers import GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig, XLMConfig, CTRLConfig

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from transformers import XLNetLMHeadModel, XLNetTokenizer
from transformers import TransfoXLLMHeadModel, TransfoXLTokenizer
from transformers import CTRLLMHeadModel, CTRLTokenizer
from transformers import XLMWithLMHeadModel, XLMTokenizer

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


def sample_sequence(model, context, device='cpu', length=256):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(1, 1)
    generated = context
    with torch.no_grad():
        for _ in trange(length, ascii=True, ncols=100):
            inputs = {'input_ids': generated}
            outputs = model(**inputs)
            next_token_logits = outputs[0][:, -1, :]
            
            filtered_logits = top_k_top_p_filtering(next_token_logits)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)

    return generated

# set_seed(42,1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_class, tokenizer_class = MODEL_CLASSES["gpt2"]
tokenizer = tokenizer_class.from_pretrained("genre_output")
model = model_class.from_pretrained("genre_output")
model.to(device)
model.eval();

prime = input("prime the network>> ")
print("GENRE OPTIONS: Fantasy, Musical, Biography, Western, Sci-Fi, Animation, Comedy, Action, Drama, Family, Documentary, Horror, History, Sport, Romance, Crime, Film-Noir, Thriller, Mystery, Music, Adventure, War")
genre = input("genre (space separated list of genres)>> ")

genre_tokens = tokenizer.encode(genre + "~", add_special_tokens=False)
context_tokens = genre_tokens + tokenizer.encode(prime, add_special_tokens=False)

context = torch.tensor(context_tokens, dtype=torch.long, device=device)
context = context.unsqueeze(0).repeat(1, 1)

genre_tensor = torch.tensor(genre_tokens, dtype=torch.long, device=device)
genre_tensor = genre_tensor.unsqueeze(0).repeat(1, 1)

generated = context

# response = 'y'
# while response != 'n':
#     generated_tokens = []
#     with torch.no_grad():
#         for _ in trange(1024, ascii=True, ncols=100):
#             inputs = {'input_ids': generated}
#             outputs = model(**inputs)
#             next_token_logits = outputs[0][:, -1, :]
            
#             filtered_logits = top_k_top_p_filtering(next_token_logits)
#             next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
#             generated_tokens.append(next_token.item())
            
#             generated = torch.cat((generated, next_token), dim=1)
            
#             if generated.shape[1] > 1023:
#                 generated = torch.cat((genre_tensor, generated[:,len(genre_tokens)+1:]), dim=1)
            
#     text = tokenizer.decode(generated_tokens, clean_up_tokenization_spaces=True)
#     print(text)
#     full += text
#     response = input("Continue? (y/n): ")

print(prime, end="")
with torch.no_grad():
    for _ in range(10000):
        inputs = {'input_ids': generated}
        outputs = model(**inputs)
        next_token_logits = outputs[0][:, -1, :]

        filtered_logits = top_k_top_p_filtering(next_token_logits)
        next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)

        print(tokenizer.decode(next_token.item(), clean_up_tokenization_spaces=True), end = '')

        generated = torch.cat((generated, next_token), dim=1)

        if generated.shape[1] > 1023:
            generated = torch.cat((genre_tensor, generated[:,len(genre_tokens)+1:]), dim=1)
    
print("_________________EXITING_________________")