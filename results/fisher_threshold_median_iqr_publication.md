# Fisher threshold N* (median across seeds 0, 1, 2)

Per-readout estimate from `fisher_information_analysis` phase-transition heuristic. **median_N_star** = median of `estimated_N_threshold` over three grammar-learner seeds; **N_star_q25 / N_star_q75** = quartiles across the same three values (IQR = q75 − q25).

| Topology | Q | median N* | q25 | q75 | IQR width | stable (3 seeds) |
|----------|---|-----------|-----|-----|-----------|------------------|
| 1D_Snake | 14q | 9500 | 9500 | 9500 | 0 | yes |
| 1D_Snake | 18q | 9500 | 9500 | 9500 | 0 | yes |
| 1D_Snake | 22q | 6500 | 6500 | 6500 | 0 | yes |
| 1D_Snake | 28q | 5500 | 4000 | 11500 | 7500 | no |
| 1D_Snake | 32q | 4500 | 3500 | 11000 | 7500 | no |
| 1D_Snake | 47q | 3500 | 3500 | 3500 | 0 | yes |
| 1D_Snake | 49q | 17500 | 13000 | 17500 | 4500 | no |
| 2D_Block | 12q | 9500 | 9500 | 9500 | 0 | yes |
| 2D_Block | 16q | 9500 | 8500 | 9500 | 1000 | no |
| 2D_Block | 20q | 9500 | 9500 | 9500 | 0 | yes |
| 2D_Block | 24q | 6500 | 3625 | 6500 | 2875 | no |
| 2D_Block | 30q | 12500 | 7500 | 15000 | 7500 | no |
| 2D_Block | 34q | 5500 | 5500 | 5500 | 0 | yes |
| 2D_Block | 39q | 3500 | 3500 | 5500 | 2000 | no |
| 2D_Block | 40q | 7500 | 4125 | 8000 | 3875 | no |
| 2D_Block | 41q | 2500 | 1625 | 3000 | 1375 | no |
| 2D_Block | 42q | 7500 | 6000 | 8000 | 2000 | no |
| 2D_Block | 43q | 12500 | 12500 | 15000 | 2500 | no |
| 2D_Block | 44q | 3500 | 3000 | 3500 | 500 | no |
| 2D_Block | 45q | 3500 | 3500 | 3500 | 0 | yes |
| 2D_Block | 50q | 17500 | 15000 | 17500 | 2500 | no |
| Bulk_Full | 26q | 6500 | 6500 | 6500 | 0 | yes |
| Bulk_Full | 36q | 3500 | 3500 | 10500 | 7000 | no |
| Bulk_Full | 38q | 5500 | 5000 | 5500 | 500 | no |
| Bulk_Full | 46q | 750 | 750 | 750 | 0 | yes |
| Bulk_Full | 48q | 9500 | 9500 | 9500 | 0 | yes |
| Bulk_Full | 51q | 8500 | 8000 | 8500 | 500 | no |
| Bulk_Full | 53q | 9500 | 5125 | 9500 | 4375 | no |

Source CSV: `fisher_estimated_thresholds_median_seeds012.csv`