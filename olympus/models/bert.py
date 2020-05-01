import logging
import os
import random

import numpy as np
import torch

from transformers import BertForSequenceClassification, BertConfig

from olympus.utils.options import option

logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class BertWrapper(BertForSequenceClassification):
    def forward(self, *args, **kwargs):
        result = super(BertWrapper, self).forward(*args, **kwargs)
        return result[0]


def build_bert(input_size, output_size, task):
    cache_dir = option('model.cache', '/tmp/olympus/cache')
    logger.info('model cache folder: {}'.format(cache_dir))
    config = BertConfig.from_pretrained(
        'bert-base-uncased',
        num_labels=2,
        finetuning_task=task,
        cache_dir=cache_dir)
    model = BertWrapper.from_pretrained(
        'bert-base-uncased',
        from_tf=False,
        config=config,
        cache_dir=cache_dir)
    return model


def build_bert_sst2(input_size, output_size):
    return build_bert(input_size, output_size, 'sst-2')


def build_bert_cola(input_size, output_size):
    return build_bert(input_size, output_size, 'cola')


def build_bert_mrpc(input_size, output_size):
    return build_bert(input_size, output_size, 'mrpc')


def build_bert_rte(input_size, output_size):
    return build_bert(input_size, output_size, 'rte')


builders = {
    'bert-sst2': build_bert_sst2,
    'bert-cola': build_bert_cola,
    'bert-mrpc': build_bert_mrpc,
    'bert-rte': build_bert_rte}
