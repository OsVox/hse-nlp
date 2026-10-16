"""Optional digit boundaries with the original learned vocabulary and text."""
import json

from tokenizers import pre_tokenizers
from transformers import PreTrainedTokenizerFast


def load_training_tokenizer(path, *, individual_digits=False):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(path)
    if individual_digits:
        backend = tokenizer.backend_tokenizer
        spec = json.loads(backend.to_str())['pre_tokenizer']
        digit_spec = {'type': 'Digits', 'individual_digits': True}
        if spec.get('type') == 'Sequence' and spec['pretokenizers'][-1] == digit_spec:
            return tokenizer
        vocabulary = tokenizer.get_vocab()
        backend.pre_tokenizer = pre_tokenizers.Sequence([
            backend.pre_tokenizer, pre_tokenizers.Digits(individual_digits=True)])
        assert tokenizer.get_vocab() == vocabulary
    return tokenizer
