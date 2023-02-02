#!/usr/bin/env python
import os
import re
import json
import arxiv
from extract_figures import build_fig_dict, extract_figures
from extract_sections import get_section_mapping

def extract_sections(text):
    drop_sections = re.split(r'\\section\*{.*?}', text, re.S)
    results = re.split(r'\\section{.*?}', drop_sections[0], re.S)
    return results[1:], drop_sections[0]

def load_tex(f_path: str):
    with open(f_path, 'r') as f: data = f.read()
    return data

def load_bib(f_path: str):
    return load_tex(f_path)

def get_sections(tex):
    results = re.findall(r'\\section{(.*?)}', tex)
    return results

def get_paper_title(ids):
    results = arxiv.Search(
        id_list=ids
    )
    titles = [res.title for res in results.results()]
    title_dict = dict(zip(ids,titles))
    return title_dict

def map_sections(sections, parsed_tex, title, arxiv_id):
    mapped = {
            sections[i]:
            {
                "content":remove_citations(parsed_tex[i]), "title":title, "arxiv_id": arxiv_id
            } for i in range(len(sections))
        }
    return mapped

def parse_tex(path: str, paper_title: str, arxiv_id: str):
    tex = load_tex(path)
    res, raw = extract_sections(tex)
    sections = get_sections(raw)
    mapped_sections = map_sections(sections, res, paper_title, arxiv_id)
    fig_matches = extract_figures(tex)
    fig_dict = build_fig_dict(fig_matches, arxiv_id)
    section_map = get_section_mapping(tex)
    return mapped_sections, fig_dict, section_map

def remove_latex(input_str:str):
    return re.sub(r'(\$)(.*?)(\$)', '', input_str)

def remove_citations(text: str):
    return re.sub(r'cite\{.*?\}', '', text)