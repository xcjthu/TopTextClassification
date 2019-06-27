from model.model.LSTM import LSTM
from model.model.TextCNN import TextCNN
from model.model.SFKS.BasicCNN import BasicCNN
from model.model.SFKS.co_matching import CoMatching
from model.model.SFKS.SeaReader.SeaReader import SeaReader
from model.model.SFKS.co_matching import CoMatching, CoMatching2, CoMatching3, CoMatching4
from model.model.SFKS.multi_matching import MultiMatchNet
from model.model.SFKS.conv_spatial_att import ConvSpatialAtt
from model.model.SFKS.bert import SFKSBert
from model.model.bert import Bert
from model.model.SFKS.simple import SimpleAndEffective
from model.model.SFKS.DSQA_lyk import DSQA
from model.model.DPCNN import DPCNN
from model.model.SFKS.DenoiseDSQA import DenoiseDSQA
from model.model.cail.ys import YSBert
from model.model.ljp.ljp import LJPBert

model_list = {
    "LSTM": LSTM,
    "TextCNN": TextCNN,
    "SFKSCNN": BasicCNN,
    "Comatching": CoMatching,
    "SeaReader": SeaReader,
    "Comatching2": CoMatching2,
    "Comatching3": CoMatching3,
    "Comatching4": CoMatching4,
    "MultiMatch": MultiMatchNet,
    "ConvSpatialAtt": ConvSpatialAtt,
    "Bert": Bert,
    "SFKS_bert": SFKSBert,
    "SFKSSimpleAndEffective": SimpleAndEffective,
    "DSQA": DSQA,
    "DPCNN": DPCNN,
    "DenoiseDSQA": DenoiseDSQA,
    "YSBert": YSBert,
    "LJPBert": LJPBert
}


def get_model(name, config):
    if name in model_list.keys():
        return model_list[name](config)
    else:
        raise NotImplementedError
