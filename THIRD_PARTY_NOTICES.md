# Third-Party Notices and Source Traceability

## AHEAD

The project includes AHEAD source and separate local modifications. The original AHEAD license applies to upstream code. Original research additions and modifications are addressed separately below; upstream copyright and license notices remain in force. Original repository: https://github.com/YayaLee666/AHEAD ; compared revision b9d5a805143aa280a665a11d6bdd53d211943888. The complete original AHEAD MIT license is retained unmodified in `src/LICENSE` and copied to the repository-root `LICENSE`.

## Original research contributions

The project licenses original research code additions and modifications for which it holds the necessary rights under the MIT License terms reproduced in the repository-root `LICENSE`. Copyright in those contributions remains with the respective rights holders. This grant includes the original portions of the local AHEAD modifications, the preprocessing, training, scoring, calibration evaluation and statistics entries, and their original tests; it does not assert ownership of inherited third-party material. It does not itself license datasets, labels, historical results, checkpoints or scores. Third-party components remain subject to their own applicable licenses and retained notices.

## DyGLib: source correspondence and MIT notice

The AHEAD-version files `src/utils/DataLoader.py`, `src/models/GraphMixer.py`, `src/utils/utils.py`, and `src/evaluate_models_utils.py` contain substantial matching code blocks compared with the public DyGLib repository at revision `3aacc36b94b8d2d8293d70a74fdf6d39089b4163`: https://github.com/yule-BUAA/DyGLib/tree/3aacc36b94b8d2d8293d70a74fdf6d39089b4163 . AHEAD also references DyGLib for GraphMixer. The comparison establishes code similarity, but the complete provenance and attribution of these files remain unverified. The following MIT notice is preserved as a conservative attribution and does not assert that every file originates in DyGLib. The MIT license at the compared revision is available at https://github.com/yule-BUAA/DyGLib/blob/3aacc36b94b8d2d8293d70a74fdf6d39089b4163/LICENSE and matches the current master license by SHA-256 (24d69467ed699619c319a1de89377685ac519587c77b1d256734eb2f26b4d8d0).

```text
MIT License

Copyright (c) 2023 Yu Le

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## GraphMixer and JODIE

GraphMixer publication and original project: https://github.com/CongWeilin/GraphMixer . This repository includes a GraphMixer implementation already present in AHEAD, with substantial similarity to the DyGLib implementation. The immediately affected file is `src/models/GraphMixer.py`, included in AHEAD and substantially similar to DyGLib; dependent model code includes `src/models/PartitionedGraphMixer.py` and `src/models/AHEADGraphMixer.py`. Similarity alone does not establish a direct-copy chain to the GraphMixer author repository. Its GitHub license API returned 404 when checked; the applicable license and any direct-copy chain remain unconfirmed. DyGLib’s MIT license does not establish rights to unrelated code.

JODIE project: https://github.com/claws-lab/jodie ; JODIE's repository provides an MIT code license. JODIE is cited here as a data source and research reference; no direct copying of JODIE repository code into this repository was verified. The JODIE code license does not establish redistribution rights for its datasets. Dataset sources and unresolved redistribution conditions are described in `README.md`.

## Other source and data rights

This notice does not establish the provenance or license of all other code, data, labels, checkpoints, scores, or derived results. Historical aggregate and selection records include `src/research_runs/fixed_degree_up_exploration_20261123/report.json`, `src/research_runs/c_reference_pairwise_exploration_20261123/report.json`, the per-run provenance/selection/development JSON files, and `src/saved_results/degree_ablation/digg/*.json`. These are summaries and experiment records; their availability does not by itself establish redistribution rights. No explicit prohibition applicable to these summaries was identified in the sources reviewed, but applicable dataset and derivative-result rights remain unconfirmed. Review upstream rights and dataset terms before distribution. No raw data, checkpoints, or per-event scores are included in this repository.
