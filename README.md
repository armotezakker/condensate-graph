# condensate-graph

condensate-graph builds chain-level contact networks from molecular simulations of protein-RNA biomolecular condensates. It reads any trajectory format that MDAnalysis supports, identifies protein and RNA chains from user-supplied selections, and finds inter-chain contacts with minimum-image periodic distances. Network metrics such as RNA bridging between proteins, percolation of the dense-phase network and hub identification will be built on top of these contacts.

## Install

```bash
git clone https://github.com/armotezakker/condensate-graph.git
cd condensate-graph
pip install -e ".[dev]"
pytest
```

## Status

Pre-release. The current version covers periodic box handling and contact search only; the API will change.
