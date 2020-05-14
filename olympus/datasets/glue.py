import logging
import os

import torch
from torch.utils.data import TensorDataset
from transformers import BertTokenizer
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors

from olympus.datasets.dataset import AllDataset
from olympus.utils.options import option

logger = logging.getLogger(__name__)


class GLUE(AllDataset):
    """ the General Language Understanding Evaluation (GLUE) benchmark is a collection of
    tools for evaluating the performance of models across a diverse set of existing NLU tasks.
    More on `arxiv <https://arxiv.org/abs/1804.07461>`_.
    `Official website <https://gluebenchmark.com/tasks>`_.


    References
    ----------
    .. [1] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, Samuel R. Bowman
        GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding, 2018

    """
    def __init__(self, data_path, task_name=None, **kwargs):
        transformations = None

        if task_name is None:
            raise ValueError('do not use this class directly - instantiate a subclass')
        data_folder = os.path.join(data_path, task_name.upper() if task_name != 'cola' else 'CoLA')
        # hard-coding the model type for now..
        model_name_or_path = 'bert-base-uncased'
        model_type = 'bert'
        # and sequence size..
        max_seq_length = 128

        cache_dir = option('tokenizer.cache', '/tmp/olympus/cache_tok')
        logger.info('tokenizer cache folder: {}'.format(cache_dir))
        tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased',
            do_lower_case=True,
            cache_dir=cache_dir,
        )

        try:
            train_dataset = load_and_cache_examples(
                    task_name, tokenizer, data_folder, model_name_or_path, max_seq_length, model_type,
                    evaluate=False)
            test_dataset = load_and_cache_examples(
                    task_name, tokenizer, data_folder, model_name_or_path, max_seq_length, model_type,
                    evaluate=True)
        except FileNotFoundError:
            raise ValueError('please point the environment variable OLYMPUS_DATA_PATH '
                             'to the folder containing the GLUE data. Currently, it is '
                             'set as "{}"'.format(data_path))

        super(GLUE, self).__init__(
            torch.utils.data.ConcatDataset([train_dataset, test_dataset]),
            test_size=len(test_dataset),
            transforms=transformations
        )

    @staticmethod
    def categories():
        return {'classification'}


def glue_sst2(data_path, **kwargs):
    return GLUE(data_path=data_path, task_name='sst-2')


def glue_mrpc(data_path, **kwargs):
    return GLUE(data_path=data_path, task_name='mrpc')


def glue_cola(data_path, **kwargs):
    return GLUE(data_path=data_path, task_name='cola')


def glue_rte(data_path, **kwargs):
    return GLUE(data_path=data_path, task_name='rte')


builders = {
    'glue-sst2': glue_sst2,
    'glue-mrpc': glue_mrpc,
    'glue-cola': glue_cola,
    'glue-rte': glue_rte}


def load_and_cache_examples(task, tokenizer, data_dir, model_name_or_path, max_seq_length,
                            model_type, evaluate=False, overwrite_cache=False):

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        data_dir,
        "cached_{}_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, model_name_or_path.split("/"))).pop(),
            str(max_seq_length),
            str(task),
        ),
    )
    if os.path.exists(cached_features_file) and not overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", data_dir)
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and model_type in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = (
            processor.get_dev_examples(data_dir) if evaluate else processor.get_train_examples(data_dir)
        )
        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=max_seq_length,
            output_mode=output_mode,
            pad_on_left=bool(model_type in ["xlnet"]),  # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if model_type in ["xlnet"] else 0,
        )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    return TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
