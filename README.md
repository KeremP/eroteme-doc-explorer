# eroteme-doc-explorer
Microservice for using chained LLM calls and document embeddings to explore sets of documents.

## TODO:
- reduce latency -> build_docstore() runtime now down to < 0.5s (~.3s)
- ~~cache embeddings of split docs~~
    - ~~use as key Arxiv ID and store array of b64 str representations of embeddings~~
- test with mulitple documents

## example resp
{'answer': ' The Higgs boson is a particle of electroweak symmetry breaking in particle physics.\n', 'sources': 'Introduction, The long-sought Higgs boson(s) $h$ of electroweak symmetry breaking\\nin particle physics may soon be observed at the CERN Large Hadron\\nCollider (LHC) through the diphoton decay mode ($h\\\\rightarrow\\\\gamma\\\\gamma$).'}

## notes:
- drag and drop UI for adding additional databases, chaining databases, etc.
- potential UX flow:
    - search query -> finds top-x papers -> presents in "select which papers to include" UI (w/ abstracts easily accessible) -> begin asking questions about papers
    - include command mappings e.g., "pull all figures", etc.
    - chain of reasoning for whether additional data is needed (triggers another doc dowload/pull into the session's prompt/latent space thru embeddings)