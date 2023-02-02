#!/usr/bin/env python
import re
from typing import List, Dict, Tuple

reg = re.compile(r'\{.*?\}')
def extract_figures(text: str) -> List[str]:
    RE = r'(\\begin\{figure\}.*?\\end\{figure})'
    matches = re.findall(RE, text, re.DOTALL)
    return matches

def extract_figure_data(text: str) -> Tuple[List[str], List[str]]:
    text = "".join(text.split("\n"))
    RE = r'(\\includegraphics\[?.*?\]?\{.*?\})'
    matches = re.findall(RE, text, re.DOTALL)
    paths = [re.findall(reg, t)[0] for t in matches]
    L_RE = r'(\\label\{.*?\})'
    label_matches = re.findall(L_RE, text, re.DOTALL)
    labels = [re.search(reg, t).group().strip("{").strip("}") for t in label_matches]
    return paths, labels

def build_fig_dict(matches: List[str], arxiv_id: str) -> Dict[str,list]:
    fig_dict = {}
    id = "".join(arxiv_id.split("."))
    for match in matches:
        paths, labels = extract_figure_data(match)
        for label in labels:
            temp = {label:[id+"/"+p.strip("{").strip("}") for p in paths]}
            fig_dict.update(temp)
    return fig_dict

if __name__ == "__main__":

    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    arxiv_id = "210406821"
    with open("./temp/210406821/Higgs.tex") as f:
    # with open("./temp/07040001/diphoton.tex") as f:
        text = f.read()

    matches = extract_figures(text)
    print(matches[0])

    fig_dict = {}


    for match in matches:
        paths, labels = extract_figure_data(match, arxiv_id)
        for label in labels:
            temp = {label:paths}
            fig_dict.update(temp)

    pp.pprint(fig_dict)

    
    