import os
import elasticsearch
import elasticsearch.exceptions

if os.path.exists(os.path.join(os.path.dirname(os.path.realpath(__file__)), "local_config.py")):
    from . import local_config as config
else:
    from . import config as config

es = elasticsearch.Elasticsearch("%s:%s" % (config.host, str(config.port)),
                                 http_auth=(config.username, config.password))


def insert_doc(index, doc_type, doc, uid):
    response = es.index(index=index, doc_type=doc_type, body=doc, id=uid, request_timeout=30)


def set_mapping(index, doc_type, body):
    response = es.indices.put_mapping(index=index, doc_type=doc_type, body=body)


def get_mapping(index, doc_type):
    return es.indices.get_mapping(index, doc_type)


def create_index(index, body):
    response = es.indices.create(index=index, body=body, ignore=400)
    print(response)


def delete_index(index):
    try:
        response = es.indices.delete(index=index)
    except elasticsearch.exceptions.NotFoundError as e:
        pass


def search(index, doc_type, body, size=10, from_=0, timeout=30, scroll="1m"):
    response = es.search(index=index, doc_type=doc_type, body=body, size=size, from_=from_, request_timeout=timeout,
                         scroll=scroll)

    return response


def scroll(scroll_id, scroll_time="1m", timeout=30):
    response = es.scroll(scroll_id=scroll_id, scroll=scroll_time, request_timeout=timeout)

    return response


def fetch_doc(index, doc_type, idx, timeout=30):
    response = es.get(index=index, doc_type=doc_type, id=idx, request_timeout=timeout)

    return response


def update_doc(index, doc_type, idx, data, timeout=30):
    es.update(index=index, doc_type=doc_type, id=idx, body={"doc": data}, request_timeout=timeout)
