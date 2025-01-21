This repository uses [beir2.0.0](https://github.com/beir-cellar/beir/releases/tag/v2.0.0) for BM25.

Due to server limitations, package `pytrec_eval` cannot be used. Therefore, the evaluate function in the `beir.retrieval.evaluation` which depends on `pytrec_eval` but isn't needed has been commented out.