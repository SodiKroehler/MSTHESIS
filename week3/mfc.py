#read in json


import pandas as pd
import json


if __name__ == "__main__":
    codes = {}
    smoking = {}
    samesex = {}
    immigration = {}

    with open('./media_frames_corpus/annotations/codes.json') as f:
        data = json.load(f)

    with open('./media_frames_corpus/annotations/smoking.json') as f:
        smoking = json.load(f)

    with open('./media_frames_corpus/annotations/samesex.json') as f:
        samesex = json.load(f)

    with open('./media_frames_corpus/annotations/immigration.json') as f:
        immigration = json.load(f)

    

