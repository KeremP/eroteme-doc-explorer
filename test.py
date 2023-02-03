#!/usr/bin/env python
from handler import lambda_handler
from pprint import PrettyPrinter

if __name__ == "__main__":
    EVENT = {
        "ids":["0704.0001","2104.06821"],
        "query":"What is the higgs-boson?"
    }

    pp = PrettyPrinter(indent=4)

    resp = lambda_handler(EVENT, None)
    print(resp['results'], resp['sources'])

    pp.pprint(resp['figures'])
    pp.pprint(resp['sections'])