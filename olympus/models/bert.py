import random

import numpy as np
import torch

from olympus.utils.options import option
from transformers import BertForSequenceClassification, BertConfig


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class BertWrapper(BertForSequenceClassification):
    def forward(self, *args, **kwargs):
        result = super(BertWrapper, self).forward(*args, **kwargs)
        return result[0]


def build_bert(input_size, output_size, model_seed, task):
    set_seed(model_seed)
    config = BertConfig.from_pretrained(
        'bert-base-uncased',
        num_labels=2,
        finetuning_task=task,
        cache_dir=option('model.cache', '/tmp/olympus/cache'),
    )
    model = BertWrapper.from_pretrained(
        'bert-base-uncased',
        from_tf=False,
        config=config,
        cache_dir=option('model.cache', '/tmp/olympus/cache')
    )
    return model


def build_bert_sst2(input_size, output_size, model_seed):
    return build_bert(input_size, output_size, model_seed, 'sst-2')


def build_bert_cola(input_size, output_size, model_seed):
    return build_bert(input_size, output_size, model_seed, 'cola')


def build_bert_mrpc(input_size, output_size, model_seed):
    return build_bert(input_size, output_size, model_seed, 'mrpc')


def build_bert_rte(input_size, output_size, model_seed):
    return build_bert(input_size, output_size, model_seed, 'rte')


builders = {
    'bert-sst2': build_bert_sst2,
    'bert-cola': build_bert_cola,
    'bert-mrpc': build_bert_mrpc,
    'bert-rte': build_bert_rte}
