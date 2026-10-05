# Artifact Repository — Paper B (P02)

**Manuscript.** *Architecture Search in Self-Modifying Systems: The Scale-Decay Regularity of General Selection and Randomized Controlled Audit Discipline*

**Author.** Yulong Li — ORCID [0009-0006-0492-857X](https://orcid.org/0009-0006-0492-857X) — Independent Researcher, Guiyang, China

This repository hosts the auditable artifact for the manuscript above. It contains no experimental
results that are not reproducible from the released code and logs.

---

## Where to start

| Path | What it is |
|---|---|
| **`P02_果蝇B_成果包_v2/`** | The artifact package: platform code, per-run event logs, verdict JSONs, frozen criteria files, and authorization records. **Begin with its `README.md`**, which maps every quantitative claim in the manuscript to the file that supports it. |
| `P02_果蝇B_成果包_v2/MANIFEST.sha256` | SHA256 of every released file, with paths relative to the package root. |
| `P02_果蝇B_成果包_v2/zenodo.json` | The metadata used for the archival deposit, including the license split (MIT for code, CC-BY-4.0 for data and logs). |

## Verify integrity

Every file in the package is checksummed. After cloning:

```bash
cd P02_果蝇B_成果包_v2
sha256sum -c MANIFEST.sha256
```

All 191 entries should report `OK`. The manifest deliberately excludes two kinds of non-artifact
paths: tool-state directories (`.mimosa/`) and bytecode caches (`__pycache__/`). If you prefer Python:

```bash
python - <<'PY'
import hashlib, pathlib
root = pathlib.Path("P02_果蝇B_成果包_v2")
bad = [l.split("  ", 1) for l in (root/"MANIFEST.sha256").read_text(encoding="utf-8").splitlines() if l]
bad = [(f, p) for f, p in bad if hashlib.sha256((root/p).read_bytes()).hexdigest() != f]
print("mismatches:", len(bad))
PY
```

## Frozen decision rules

The three criteria files are frozen and each one's SHA256 matches the `criteria_sha256` recorded
inside the verdict JSONs it governs — so a third party can check that the decision rules were not
changed after the fact:

| Criteria file | SHA256 | Governs |
|---|---|---|
| `prereg/TJ1-预注册判据-frozen-v1.0.json` | `b2b05c0d96c6136767c984829f34ab7b971c4470c632b3055f5c5a63d9872939` | the preregistered three-arm verdict batch |
| `TJ-B1b-frozen-v1.0.json` | `2f07deeb9c27cdda2ffd8559bb4c041be3301162bd96da1de445de7f0f57074a` | the post-hoc grammar-constrained extension, step one |
| `TJ-B1C-frozen-v1.0.json` | `f83ff0b178c8d301ccc1ac29d3af0e25256d71d0fd70eeb02f922c8fb2c73bec` | the post-hoc grammar-constrained extension, step two |

## Data availability (as printed in the manuscript)

> The code, internally frozen decision rules, adjudication records, and analysis scripts that support
> the findings of this study are openly available at
> https://github.com/printing10101/p02-connectome-detectability and
> https://github.com/printing10101/p02-prereg-evidence. Upon publication the complete artifact will
> be permanently archived at Zenodo under a DOI to be assigned at deposition.

## Reproducing

All local experiments rerun on a single RTX 3080 Laptop GPU. The two language-model tiers ran
through a local `llama-server` inference chain rather than an external API. Setup, entry points, and
the small number of path adjustments needed to run the scripts directly from this package layout are
documented in `P02_果蝇B_成果包_v2/README.md`.

## License and citation

- Code: MIT. Data and logs: CC-BY-4.0. See `P02_果蝇B_成果包_v2/zenodo.json`.
- When citing, please give both the manuscript DOI and the artifact DOI, and quote the snapshot
  manifest hash.
