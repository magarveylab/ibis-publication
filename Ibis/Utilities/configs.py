from transformers import BertConfig

def get_protbert_config() -> BertConfig:
    pretrained_dir = 'Rostlab/prot_bert'
    return BertConfig.from_pretrained(pretrained_dir)