from model.model.LSTM import LSTM
from model.model.TextCNN import TextCNN
from model.model.demo.Judge_prediction import JudgePrediction
from model.model.demo.NaiveLSTM import NaiveLSTM
from model.model.demo.DemoMultiTaskCNN import DemoMultiTaskCNN
from model.model.demo.MultiTaskBert import MultiTaskBert

from model.model.demo.bert import BertDemo

from model.model.SFKS.BasicCNN import BasicCNN
from model.model.SFKS.co_matching import CoMatching
from model.model.SFKS.SeaReader.SeaReader import SeaReader
from model.model.SFKS.co_matching import CoMatching, CoMatching2, CoMatching3
from model.model.SFKS.multi_matching import MultiMatchNet
from model.model.SFKS.conv_spatial_att import ConvSpatialAtt
from model.model.SFKS.bert import SFKSBert
from model.model.bert import Bert
from model.model.SFKS.simple import SimpleAndEffective
from model.model.SFKS.DSQA_lyk import DSQA
from model.model.DPCNN import DPCNN
from model.model.SFKS.DenoiseDSQA import DenoiseDSQA

model_list = {
    "LSTM": LSTM,
    "TextCNN": TextCNN,
    "JudgePrediction": JudgePrediction,
    "NaiveLSTM": NaiveLSTM,
    "DemoCNN": DemoMultiTaskCNN,
    "BertDemo": BertDemo,
    "MultiTaskBert": MultiTaskBert,

    "SFKSCNN": BasicCNN,
    "Comatching": CoMatching,
    "SeaReader": SeaReader,
    "Comatching2": CoMatching2,
    "Comatching3": CoMatching3,
    "MultiMatch": MultiMatchNet,
    "ConvSpatialAtt": ConvSpatialAtt,
    "Bert": Bert,
    "SFKS_bert": SFKSBert,
    "SFKSSimpleAndEffective": SimpleAndEffective,
    "DSQA": DSQA,
    "DPCNN": DPCNN,
    "DenoiseDSQA": DenoiseDSQA
}


def get_model(name, config):
    if name in model_list.keys():
        return model_list[name](config)
    else:
        raise NotImplementedError
