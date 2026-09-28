# Neural Dispersion on Graphs

This repository contains the source code for our ICML 2026 paper [Neural Dispersion on Graphs](https://openreview.net/forum?id=dtvCOsy3ie). Neural graph dispersion (NGD) generates a set of *k* graphs on *N* vertices that is maximally diverse under a chosen graph distance. An ensemble of generators is trained under a repulsive potential, so that the graphs it produces disperse along the training trajectories, and a diverse set is then selected from these graphs for each distance. NGD requires no training data, and one training run serves all distances.

For questions, please contact ryien@uchicago.edu.

## Installation

```bash
git clone https://github.com/ryienh/neural-graph-dispersion.git
cd neural-graph-dispersion
pip install -e .
```

The graphlet correlation distance (GCD) uses the [ORCA](https://github.com/thocevar/orca) graphlet counter, which is GPL-3.0 licensed and therefore not included here. To use GCD, download and compile ORCA into `ngd/metrics/orca/`:

```bash
curl -L https://raw.githubusercontent.com/thocevar/orca/e146a8b1a99a90f5e3096b7bcc2ab0ea246c3ca7/orca.cpp -o ngd/metrics/orca/orca.cpp
g++ -O2 -std=c++11 -o ngd/metrics/orca/orca ngd/metrics/orca/orca.cpp
```

## Quick Start

The supported distances are `gcd`, `netlsd_heat`, `netlsd_wave` and `portrait_div`.

```python
import ngd

# 100 graphs on 16 vertices, diverse under the NetLSD heat distance.
graphs = ngd.generate(n_vertices=16, k=100, metric="netlsd_heat", seed=0)

# Mean pairwise distance of the set.
print(ngd.diversity(graphs, "netlsd_heat"))
```

`ngd.generate` returns a list of `networkx` graphs. Any other setting in [`ngd/config.py`](ngd/config.py) can be passed as a keyword argument. The same settings can be given as a YAML config:

```bash
python generate.py --config configs/default.yaml --seed 0
```

This writes one set per distance to `results/graphs_<metric>.pkl`, and the diversity of each set to `results/summary.json`. A GPU is recommended.

## Paper Experiments

Configs for Table 1 (N=16 and N=128) are in `configs/table1/`, one per distance. `reproduce.py` runs a config over five seeds and reports the mean and standard deviation of the diversity:

```bash
for config in configs/table1/n16/*.yaml configs/table1/n128/*.yaml; do
    python reproduce.py --config $config
done
```

Example configs for larger graphs (N=256, N=512) and larger sets (k=1024) are in `configs/scaling/`:

```bash
python reproduce.py --config configs/scaling/n256.yaml
```

## Citation

```bibtex
@inproceedings{hosseini2026neural,
    title={Neural Dispersion on Graphs},
    author={Ryien Hosseini and Pouya Mahdi Gholami and Filippo Simini and Venkatram Vishwanath and Rebecca Willett and Henry Hoffmann},
    booktitle={Forty-third International Conference on Machine Learning},
    year={2026},
    url={https://openreview.net/forum?id=dtvCOsy3ie}
}
```
