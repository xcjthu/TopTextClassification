import time
import multiprocessing
import random

from utils.util import get_file_list
from reader.formatter.AYYC import AYPredictionFormatter
from reader.formatter.AJLX import AJLXPredictionFormatter
from reader.formatter.SFKS.SFKS_word import SFKSWordFormatter
from reader.formatter.SFKS.Comatching import ComatchingFormatter
from reader.formatter.SFKS.SeaReader import SeaReaderFormatter
from reader.formatter.race.race import RaceFormatter
from word2vec.word2vec import init_transformer


def init_formatter(config):
    global formatter
    if config.get("data", "formatter") == "AYYC":
        formatter = AYPredictionFormatter(config)
    elif config.get("data", "formatter") == "AJLX":
        formatter = AJLXPredictionFormatter(config)
    elif config.get("data", "formatter") == "SFKS_word":
        formatter = SFKSWordFormatter(config)
    elif config.get("data", "formatter") == "SFKS_comatching":
        formatter = ComatchingFormatter(config)
    elif config.get("data","formatter") == "SFKS_seareader":
        formatter = SeaReaderFormatter(config)
    elif config.get("data","formatter") == "RACE":
        formatter = RACE(config)
    else:
        raise NotImplementedError


class reader():
    def __init__(self, file_list, config, num_process, mode):
        self.file_list = file_list
        self.mode = mode

        self.temp_file = None
        self.read_cnt = 0

        self.file_queue = multiprocessing.Queue()
        self.data_queue = multiprocessing.Queue()
        self.lock = multiprocessing.Lock()

        self.init_file_list(config)

        self.read_process = []
        self.num_process = num_process

        self.none_cnt = 0

        from word2vec.word2vec import transformer
        self.transformer = transformer

        for a in range(0, num_process):
            process = multiprocessing.Process(target=self.always_read_data,
                                              args=(config, self.data_queue, self.file_queue, a,))
            self.read_process.append(process)
            self.read_process[-1].start()

    def init_file_list(self, config):
        if config.getboolean("train", "shuffle") and self.mode == "train":
            random.shuffle(self.file_list)
        for a in range(0, len(self.file_list)):
            self.file_queue.put(self.file_list[a])

    def always_read_data(self, config, data_queue, file_queue, idx):
        cnt = config.getint("reader", "max_queue_size")
        # print('read data', cnt)

        put_needed = False
        while True:
            if data_queue.qsize() < cnt:
                data = self.fetch_data_process(config, file_queue)
                if data is None:
                    if put_needed:
                        data_queue.put(data)
                        put_needed = False
                        time.sleep(5)
                else:
                    data_queue.put(data)
                    put_needed = True

    def gen_new_file(self, config, file_queue):
        if file_queue.qsize() == 0:
            self.temp_file = None
            return
        self.lock.acquire()
        try:
            p = file_queue.get(timeout=1)
            self.temp_file = open(p, "r")
            # print_info("Loading file from " + str(p))
        except Exception as e:
            self.temp_file = None
        self.lock.release()

    def fetch_data_process(self, config, file_queue):
        batch_size = config.getint("train", "batch_size")

        data_list = []

        if self.temp_file is None:
            self.gen_new_file(config, file_queue)
            if self.temp_file is None:
                return None

        while len(data_list) < batch_size:
            x = self.temp_file.readline()
            if x == "" or x is None:
                self.gen_new_file(config, file_queue)
                if self.temp_file is None:
                    return None
                continue

            x = formatter.check(x, config)
            if not (x is None):
                data_list.append(x)

        if len(data_list) < batch_size:
            return None

        return formatter.format(data_list, config, self.transformer, self.mode)

    def fetch_data(self, config):
        while True:
            # print('fetch_data_in')
            # print(self.num_process)
            data = self.data_queue.get()
            # print(data)

            if data is None:
                self.none_cnt += 1
                if self.none_cnt == self.num_process:
                    self.init_file_list(config)
                    self.none_cnt = 0
                    break
            else:
                break

        return data


def create_dataset(file_list, config, num_process, mode):
    return reader(file_list, config, num_process, mode)


def init_train_dataset(config):
    return create_dataset(get_file_list(config.get("data", "train_data_path"), config.get("data", "train_file_list")),
                          config, config.getint("reader", "train_reader_num"), "train")


def init_valid_dataset(config):
    print(config.get('data', 'valid_file_list'))
    return create_dataset(get_file_list(config.get("data", "valid_data_path"), config.get("data", "valid_file_list")),
                          config, config.getint("reader", "valid_reader_num"), "valid")


def init_dataset(config):
    init_transformer(config)
    init_formatter(config)
    train_dataset = init_train_dataset(config)
    valid_dataset = init_valid_dataset(config)

    return train_dataset, valid_dataset
