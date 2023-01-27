# eroteme-doc-explorer
Microservice for using chained LLM calls and document embeddings to explore sets of documents.

## TODO:
- reduce latency -> build_docstore() runtime now down to < 0.5s (~.3s) w/ Redis caching
- ~~cache embeddings of split docs~~
    - ~~use as key Arxiv ID and store array of b64 str representations of embeddings~~
- ~~test with mulitple documents~~
- ~~fix occasional bug with returning "sources" w/ answer (source sometimes changes for same answer/query)~~

## example resp
```
{
    'answer': ' The Higgs boson is a particle of electroweak symmetry breaking in particle physics, with a mass of around 125 GeV, which decays into vector bosons, namely a pair of photons, W bosons, or Z bosons\n',
    
    'sources':[
        {'source': {'section': 'Introduction', 'content': 'The long-sought Higgs boson(s) $h$ of electroweak symmetry breaking\\nin particle physics may soon be observed at the CERN Large Hadron\\nCollider (LHC) through the diphoton decay mode ($h\\\\rightarrow\\\\gamma\\\\gamma$).\\nPurely hadronic standard model processes are a copious source of diphotons,\\nand a narrow Higgs boson signal at relatively low masses will appear\\nas a small peak above this considerable background. A precise theoretical\\nunderstanding of the kinematic distributions for diphoton production\\nin the standard model could provide valuable guidance in the search\\nfor the Higgs boson signal and assist in the important measurement \\nof Higgs boson coupling strengths.', 'title': 'Calculation of prompt diphoton production cross sections at Tevatron and LHC energies', 'id': '0704.0001'}}...
        ]
}
```