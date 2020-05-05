from olympus.utils import option, info
from transformers import BertForSequenceClassification, BertConfig


class BertWrapper(BertForSequenceClassification):
    def forward(self, *args, **kwargs):
        result = super(BertWrapper, self).forward(*args, **kwargs)
        return result[0]


class BertFactory():

    def __init__(self, task):
        super(BertFactory, self).__init__()

        self.task = task

    def __call__(self, input_size, output_size, attention_probs_dropout_prob, hidden_dropout_prob):

        cache_dir = option('model.cache', '/tmp/olympus/cache')
        info('model cache folder: {}'.format(cache_dir))

        config = BertConfig.from_pretrained(
            'bert-base-uncased',
            num_labels=2,
            finetuning_task=self.task,
            cache_dir=cache_dir)

        config.attention_probs_dropout_prob = attention_probs_dropout_prob
        config.hidden_dropout_prob = hidden_dropout_prob

        model = BertWrapper.from_pretrained(
            'bert-base-uncased',
            from_tf=False,
            config=config,
            cache_dir=cache_dir)

        return model

    @staticmethod
    def get_space():
        return {
            'attention_probs_dropout_prob': 'uniform(0, 0.5)',
            'hidden_dropout_prob': 'uniform(0, 0.5)'
        }


builders = {
    'bert-sst2': BertFactory('sst2'),
    'bert-cola': BertFactory('cola'),
    'bert-mrpc': BertFactory('mrpc'),
    'bert-rte': BertFactory('rte')}
