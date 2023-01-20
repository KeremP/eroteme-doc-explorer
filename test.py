#!/usr/bin/env python
from handler import lambda_handler

if __name__ == "__main__":
    EVENT = {
        "ids":["0704.0001"],
        "query":"What is the higgs-boson?"
    }

    resp = lambda_handler(EVENT, None)
    print(resp['result'])