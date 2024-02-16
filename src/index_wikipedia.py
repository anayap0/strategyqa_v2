#!/usr/bin/python3
import argparse
import boto3
import json
import logging

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from requests_aws4auth import AWS4Auth
from smart_open import open
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_esclient(host, port, region=None):
    if region:
        service = 'es'
        try:
            credentials = boto3.Session().get_credentials()
            awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service)
            return Elasticsearch(
                hosts=[{'host': host, 'port': 443}],
                http_auth=awsauth,
                use_ssl=True,
                verify_certs=True,
            )
        except Exception as e:
            logger.error(f"Error: {e}")
            return None
    else:
        return Elasticsearch(hosts=[{"host": host, "port": port}])
    
if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(
        description='Add paragraphs from a JSON file to an Elasticsearch index.')
    parser.add_argument('-H', '--host', help='Elastic Search hostname', required=True)
    parser.add_argument('-p', '--port', type=int, default=9200, help='Port number')
    parser.add_argument('-r', '--region', help='The region for AWS ES Host. '
                        'If set, we use AWS_PROFILE credentials to connect to the host')
    parser.add_argument('file', help='Path of file to index, e.g. /path/to/my_corpus.json')
    parser.add_argument('index', help='Name of index to create')
    
    args = parser.parse_args()

    # Get Index Name
    index_name = args.index

    # Document Type constant
    TYPE = "paragraph"

    # Get an ElasticSearch client
    es = get_esclient(args.host, args.port, args.region)

    mapping = '''
    {
      "settings": {
        "index": {
          "number_of_shards": 5
        }
      },
      "mappings": {
        "properties": {
          "docId": {
            "type": "keyword"
          },
          "secId": {
             "type": "integer"
          },
          "headerId": {
             "type": "keyword"
          },
          "paraId": {
             "type": "integer"
          },
          "title": {
            "analyzer": "snowball",
            "type": "text"
          },
          "section":{
            "analyzer": "snowball",
            "type": "text"
          },
          "header":{
            "analyzer": "snowball",
            "type": "text"
          },
          "text": {
            "analyzer": "snowball",
            "type": "text",
            "fields": {
              "raw": {
                "type": "keyword"
              }
            }
          },
          "tags": {
            "type": "keyword"
          }
        }
      }
    }'''

    # Function that constructs a json body to add each line of the file to index
    def make_documents(f):
        doc_id = 0
        for l in tqdm(f):
            para_json = json.loads(l)
            doc = {
                '_op_type': 'create',
                '_index': index_name,
                '_type': TYPE,
                '_id': doc_id,
                '_source': {
                    'docId': para_json.get("docid", ""),
                    'secId': para_json.get("secid", 0),
                    'headerId': para_json.get("headerid", ""),
                    'paraId': para_json.get("para_id", 0),
                    'title': para_json.get("title", ""),
                    'section': para_json.get("section", ""),
                    'header': para_json.get("header", ""),
                    'text': para_json.get("para", ""),
                    'tags': para_json.get("tags", [])
                }
            }
            doc_id += 1
            yield doc

    try:
        # Create an index, ignore if it exists already
        res = es.indices.create(index=index_name, ignore=400, body=mapping)

        # Bulk-insert documents into index
        with open(args.file, "r") as f:
            res = bulk(es, make_documents(f))
            doc_count = res[0]

        logger.info(f"Index {index_name} is ready. Added {doc_count} documents.")

    except Exception as e:
        logger.error(f"Error: {e}")