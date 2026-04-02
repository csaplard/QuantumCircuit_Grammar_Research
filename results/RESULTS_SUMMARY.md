# Results — összefoglaló index

**Generálva:** 2026-04-02  
**Cél:** Egy helyen összefoglalni, mely eredményfájl mit tartalmaz; **az eredeti CSV/TXT/MD/naplók változatlanok** maradnak ebben a mappában.

---

## 1. Sycamore — Grammar validáció (osztályozás)

| Mit | Fájl |
|-----|------|
| Automatikus jelentés (PASS/FAIL, sweep) | `validation_report.txt` |
| Robusztaság (3 seed) | `robustness_audit_sax7.txt` |
| 60 kombinációs sweep | `parameter_sweep_sax7.csv` |
| Shuffled / LSB kontrollok | `shuffled_control_sax7.txt`, `readout_lsb_bit_control_sax7.txt` |
| LR/RF baseline | `sycamore_sax_lr_rf_summary.csv` / `.txt` / `.json` |

**Rövid számok** (részletek a fenti fájlokban): eredeti SAX=7 átlag pontosság **~84.5%** (seed std **~1.7%**); shuffled **50%**; sweepben **max_pts=8000** körül **~92.9%** cella (epoch=10).

---

## 2. Fisher — Sycamore (28 readout, seed 0–2)

| Mit | Fájl |
|-----|------|
| Publikációs tábla (medián N*, IQR, stabil) | `fisher_threshold_median_iqr_publication.md`, `.csv` |
| Medián seed-ek között | `fisher_estimated_thresholds_median_seeds012.csv` |
| Seedenkénti becslés | `fisher_estimated_thresholds_per_readout_all_readouts_seed0.csv` … `seed2.csv` |
| Fisher trace vs N | `fisher_metric_vs_datalength_all_readouts_seed0.csv` … `seed2.csv` |
| Összefoglaló szövegek (régebbi / egy seed) | `fisher_information_analysis_all_readouts.txt`, `fisher_information_analysis_all_readouts_seed*.txt` |
| Futtatási naplók | `fisher_full_run*.log`, `fisher_queue_seeds_1_2.log` |

**Összkép:** 28 konfiguráció; **medián N*** topológia szerint és readoutonként a publikációs táblában; **46q Bulk** példa: **N* = 750** mindhárom seednél (stabil) — részletek a `.md` táblában.

---

## 3. Fisher — robustness (összesített statisztikák)

| Mit | Fájl |
|-----|------|
| Topológia szerinti N* eloszlás | `fisher_robustness_summary.txt` |

N rács és log-Fisher összefoglaló ugyanitt (rövid szöveg).

---

## 4. IBM — Fisher küszöb / sweep

| Mit | Fájl |
|-----|------|
| Backend összegzés | `ibm_fisher_threshold_backend_summary.csv` |
| Normalizált küszöbök | `ibm_fisher_thresholds_normalized.csv`, `ibm_fisher_thresholds_normalized_*_summary.csv` |
| Marrakesh / Torino részletek | `ibm_fisher_thresholds_marrakesh40960.csv`, `ibm_fisher_thresholds_torino40960.csv`, `ibm_fisher_sweep_marrakesh40960.csv`, `ibm_fisher_sweep_torino40960.csv` |
| Összehasonlító jelentés | `ibm_fisher_threshold_report_marrakesh40960.txt`, `ibm_fisher_threshold_report_torino40960.txt`, `ibm_fisher_threshold_compare_marrakesh_vs_torino.csv` |
| GHZ vs Hadamard layers | `ibm_ibm_marrakesh_ghz_vs_hadamardlayers_threshold.csv`, `ibm_ibm_torino_ghz_vs_hadamardlayers_threshold.csv` |

**Backend átlagok** (normalizált táblából összegzés; pontos számok a CSV-ben): Marrakesh / Torino átlag küszöb **~18k** tartomány (szórás nagy; min–max **1500–36000**).

---

## 5. IBM Torino — Grammar / shot összehasonlítások

| Mit | Fájl |
|-----|------|
| Páronkénti KL | `ibm_torino_grammar_pairwise_kl.csv`, `.txt` |
| Hamming margók | `ibm_torino_marginals_hamming_compare.csv`, `.txt` |
| Joint / run összehasonlítás | `ibm_torino_run_compare_distributions.csv`, `.txt` |

Nagyobb klaszter / Ward mátrixok: `ibm_torino_*_ward_*.csv`, `*_ward_report.txt` (több részminta).

---

## 6. Cross-platform (IBM vs Sycamore ujjlenyomat)

| Mit | Fájl |
|-----|------|
| Szöveges jelentés | `cross_platform_universality_report.txt` |
| Összegző CSV | `cross_platform_universality_summary.csv` |
| régió-pár Frobenius | `cross_platform_regime_pair_frobenius.csv` |

**Rövid számok** (részletek a reportban): IBM **k=3** Ward vs physics3 tisztaság **~0.68**; Sycamore topológia **~0.54**; 1D+2D részhalmaz **k=2** **~0.67**.

---

## 7. Threshold transfer (Sycamore ↔ IBM)

| Mit | Fájl |
|-----|------|
| Egyoldalas összefoglaló | `threshold_transfer_onepager.txt` |
| Modell jelentés | `threshold_transfer_model_report.txt`, `threshold_transfer_uncertainty.txt` |
| CV / predikciók | `threshold_transfer_cv_*.csv`, `threshold_transfer_predictions.csv`, `threshold_transfer_loocv.csv` |

**Példa számok** (`threshold_transfer_onepager.txt`): **Ridge** teljes mintás **R² log ~0.62** (a visszatranszformált **N** predikciók és a megfigyelt **N*** összevetése **log** térben). **Stratifikált CV** átlag RMSE log **~0.76 ± 0.13** (Ridge) vs **baseline OLS** CV **~0.78 ± 0.15** — a Ridge CV-ban *nem* rosszabb; a regularizáció célja a túlilleszkedés csökkentése kis IBM mintán.

**Ne keverjük össze a ~0.85-t:** a **bootstrap** 95% intervallum a Ridge **R²**-re (`threshold_transfer_uncertainty.txt`) **felső vége ~0.84** — ez *nem* „OLS R²”, hanem a Ridge becslés bizonytalansága. A **simultán OLS** (`log(N*) ~ intercept + logQ + régió/backend dummy`) ugyanazon az adaton teljes mintás **R² ≈ 0.63** (számítás: `fit_threshold_transfer_model.fit_full_baseline`), tehát **nem ~0.85**. A **Ridge ~0.62** és az **OLS ~0.63** *nem ugyanarra a célfüggvényre* vonatkozik: a fejlett pipeline Ridge-et **log(N√Q)** célra tanít, majd **N**-re képez vissza, és **log(N)** térben értékel; az OLS baseline **közvetlenül log(N*)**-ra illeszt lineáris modellt — a két R² összehasonlítása csak óvatosan értelmezhető.

---

## 8. Sycamore readout — ujjlenyomat / tanulás (külön építések)

| Mit | Fájl |
|-----|------|
| Teljes 28 / core | `sycamore_readout_grammar_learning_results.csv`, `sycamore_readout_grammar_learning_core3.csv` |
| Naplók | `sycamore_full28_build.log`, `sycamore_e50_build.log` |

---

## 9. Egyéb / nyers

| Mit | Fájl |
|-----|------|
| IBM protokoll / naplók | `ibm_protocol_*.txt`, `results/ibm_raw_shots/` (nyers shotok) |
| DOCX / ábrák | ha a repo könyvtárában vannak, a `results/` mellett vagy almappákban |

---

*Ha új eredményfájlok kerülnek ide, ezt az összefoglalót érdemes kézzel vagy szkripttel frissíteni.*
