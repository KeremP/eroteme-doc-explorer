#!/usr/bin/env python
import re
from typing import List, Dict

def extract_sections(text: str) -> List[str]:
    text = "".join([t.strip() for t in text.split("\n")])
    RE = r'(section\{.*?\}%?\\label\{.*?\})'
    matches = re.findall(RE, text, re.DOTALL)
    return matches

# def extract_subsections(text: str) -> List[str]:
#     text = "".join([t.strip() for t in text.split("\n")])
#     RE = r'(\\subsection\{.*?\}%?\\label\{.*?\})'
#     matches = re.findall(RE, text, re.DOTALL)
#     return matches

def build_section_map(matches: List[str]) -> Dict[str, Dict[str,str]]:
    section_map = {}
    for match in matches:
        section = re.search(r'section\{.*?\}', match)
        label = re.search(r'\\label\{.*?\}', match)
        if "fig" not in label.group().lower():
            section = section.group()
            section = re.sub(r'\\label\{.*?\}', '', section)
            section_map.update({
                label.group().strip("\\label{").strip("}"):{'section':section.strip("\\section{").strip("}")}
            })
    return section_map

def get_section_mapping(text:str) -> Dict[str, Dict[str,str]]:
    matches = extract_sections(text)
    section_map = build_section_map(matches)
    return section_map

if __name__ == "__main__":

    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    arxiv_id = "21040.6821"
    # with open("./temp/210406821/Higgs.tex") as f:
    with open("./temp/07040001/diphoton.tex") as f:
        text = f.read()
    
    matches = extract_sections(text)

    section_map = build_section_map(matches)

    # print(matches)
    pp.pprint(section_map)