from transformers import (
    BertTokenizer,
    PreTrainedTokenizerFast
)

def get_protbert_tokenizer() -> PreTrainedTokenizerFast:
    return BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)