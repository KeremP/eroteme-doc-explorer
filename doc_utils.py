#!/usr/bin/env python
import os
import re
import json

def extract_sections(text):
    drop_sections = re.split(r'\\section\*{.*?}', text, re.S)
    results = re.split(r'\\section{.*?}', drop_sections[0], re.S)
    return results[1:], drop_sections[0]

def load_tex(f_path: str):
    with open(f_path, 'r') as f: data = f.read()
    return data

def get_sections(tex):
    results = re.findall(r'\\section{(.*?)}', tex)
    return results

def map_sections(sections, parsed_tex):
    mapped = {sections[i]:parsed_tex[i] for i in range(len(sections))}
    return mapped

def parse_tex(path: str):
    tex = load_tex(path)
    res, raw = extract_sections(tex)
    sections = get_sections(raw)
    mapped_sections = map_sections(sections, res)
    return mapped_sections

def remove_latex(input_str:str):
    return re.sub(r'(\$)(.*?)(\$)', '', input_str)