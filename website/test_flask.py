from flask import Flask
from flask import render_template

import time
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


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig, XLMConfig, CTRLConfig)), ())

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
        for _ in trange(100):
            inputs = {'input_ids': generated}
            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet/CTRL (cached hidden-states)
            # print(outputs.shape)
            next_token_logits = outputs[0][:, -1, :]
            
            filtered_logits = top_k_top_p_filtering(next_token_logits)
            filtered_logits = next_token_logits
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)

    return generated


def main():
    raw_text = "this is a test of the generation system"
    context_tokens = tokenizer.encode(raw_text, add_special_tokens=False)
    out = sample_sequence(model=model, context=context_tokens, device=device)
    out = out[:, len(context_tokens):].tolist()
    for o in out:
        text = tokenizer.decode(o, clean_up_tokenization_spaces=True)
        print(text)
    return text

app = Flask(__name__)
    
@app.route("/")
def home():
    return render_template("index.html")
    
@app.route('/', methods=['POST'])
def return_generation():
    time.sleep(3)
    return render_template("index.html")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42, torch.cuda.device_count())

    model_class, tokenizer_class = MODEL_CLASSES["gpt2"]
    tokenizer = tokenizer_class.from_pretrained("../output")
    model = model_class.from_pretrained("../output")
    model.to(device)
    model.eval()
    app.run(debug=True)
