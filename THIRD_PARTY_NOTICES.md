# Third-Party Notices and Source Traceability

## AHEAD

The project includes AHEAD source and separate local modifications. The original AHEAD license covers upstream code. Rights and licensing for project-specific additions remain subject to the decisions of their respective rightsholders. Original repository: https://github.com/YayaLee666/AHEAD ; compared revision b9d5a805143aa280a665a11d6bdd53d211943888. The complete original AHEAD MIT license is retained unmodified in `src/LICENSE` and copied to the repository-root `LICENSE`.

## DyGLib: source correspondence and MIT notice

The AHEAD-version files `src/utils/DataLoader.py`, `src/models/GraphMixer.py`, `src/utils/utils.py`, and `src/evaluate_models_utils.py` contain substantial matching code blocks compared with the public DyGLib repository at revision `3aacc36b94b8d2d8293d70a74fdf6d39089b4163`: https://github.com/yule-BUAA/DyGLib/tree/3aacc36b94b8d2d8293d70a74fdf6d39089b4163 . AHEAD also references DyGLib for GraphMixer. The comparison establishes code similarity, but not the complete history or attribution of every file. The following MIT notice is preserved as a conservative attribution and does not assert that every file originates in DyGLib. The current master license is available at https://github.com/yule-BUAA/DyGLib/blob/master/LICENSE ; the license file at the compared historical commit could not be accessed; the license at that revision remains unverified.

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

GraphMixer publication and original project: https://github.com/CongWeilin/GraphMixer . This package includes a GraphMixer implementation already present in AHEAD, with substantial similarity to the DyGLib implementation. No direct-copy chain to the GraphMixer author repository was established, and its repository license remains unverified; do not infer that DyGLib's MIT license grants rights to unrelated code.

JODIE project: https://github.com/claws-lab/jodie ; JODIE's repository provides an MIT code license. JODIE is cited here as a data source and research reference; no direct copying of JODIE repository code into this package was verified. The JODIE code license does not establish redistribution rights for its datasets. Dataset sources and unresolved redistribution conditions are described in `README.md`.

## Other source and data rights

This notice does not establish the provenance or license of all other code, data, labels, checkpoints, scores, or derived results. Review applicable upstream rights and dataset terms before public distribution. No raw data, checkpoints, or per-event scores are included in this package.
