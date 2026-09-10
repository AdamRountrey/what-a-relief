# Solver Comparison Tables

Generated from saved arrays; lower angular errors are better.

## Mean Angular Error (Degrees)

| Case | ls | huber10 | huber80 | cauchy | l1_unit | l1_255 | sbl_unit | sbl_255 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean_noise_8_binary | 0.350 | 0.350 | 0.350 | 0.353 | 0.426 | 0.426 | 0.476 | 0.457 |
| noisy_color_8_binary | 0.776 | 0.743 | 0.742 | 0.727 | 0.839 | 0.839 | 0.828 | 0.850 |
| noisy_color_8_headroom | 0.772 | 0.739 | 0.739 | 0.721 | 0.818 | 0.818 | 0.828 | 0.849 |
| gain_mismatch_irregular_8_binary | 1.006 | 2.629 | 2.639 | 3.030 | 2.926 | 2.926 | 5.122 | 2.940 |
| gain_mismatch_ring_8_binary | 6.685 | 6.829 | 6.829 | 6.932 | 6.873 | 6.873 | 7.108 | 6.918 |
| clean_noise_16_binary | 0.240 | 0.240 | 0.240 | 0.240 | 0.294 | 0.294 | 0.329 | 0.326 |
| noisy_color_16_binary | 0.604 | 0.560 | 0.560 | 0.568 | 0.594 | 0.594 | 0.665 | 0.664 |
| noisy_color_16_headroom | 0.606 | 0.560 | 0.560 | 0.568 | 0.591 | 0.591 | 0.666 | 0.666 |
| clean_noise_40_binary | 0.151 | 0.151 | 0.151 | 0.153 | 0.189 | 0.189 | 0.216 | 0.205 |
| noisy_color_40_binary | 0.512 | 0.485 | 0.485 | 0.503 | 0.491 | 0.491 | 0.528 | 0.534 |
| noisy_color_40_headroom | 0.514 | 0.486 | 0.486 | 0.502 | 0.488 | 0.488 | 0.528 | 0.537 |
| gain_mismatch_irregular_40_binary | 0.762 | 1.006 | 1.006 | 0.897 | 1.464 | 1.468 | 1.708 | 1.561 |
| gain_mismatch_ring_40_binary | 6.684 | 6.739 | 6.739 | 6.768 | 6.860 | 6.860 | 7.083 | 6.929 |
| clean_noise_64_binary | 0.119 | 0.119 | 0.119 | 0.119 | 0.145 | 0.145 | 0.168 | 0.153 |
| noisy_color_64_binary | 0.487 | 0.464 | 0.464 | 0.481 | 0.452 | 0.452 | 0.491 | 0.491 |
| noisy_color_64_headroom | 0.488 | 0.464 | 0.464 | 0.481 | 0.450 | 0.450 | 0.491 | 0.492 |
| broad_gloss_8_binary | 18.134 | 18.048 | 18.048 | 17.998 | 18.123 | 18.123 | 16.095 | 18.229 |
| pristine_8_binary | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| robust_v1_binary | 2.881 | 2.166 | 2.131 | 1.760 | 1.804 | 1.804 | 1.564 | 1.684 |
| robust_v1_headroom | 2.881 | 2.165 | 2.131 | 1.761 | 1.805 | 1.805 | 1.564 | 1.684 |
| textured_primitives_v1_binary | 1.846 | 1.050 | 1.055 | 0.884 | 0.994 | 0.994 | 0.686 | 0.901 |
| textured_primitives_v1_headroom | 1.845 | 1.050 | 1.055 | 0.884 | 0.994 | 0.994 | 0.686 | 0.901 |
| holdout_relief_v1_binary | 3.683 | 3.539 | 3.537 | 3.614 | 3.577 | 3.577 | 3.467 | 3.655 |
| holdout_relief_v1_headroom | 3.682 | 3.538 | 3.536 | 3.613 | 3.577 | 3.577 | 3.467 | 3.655 |

## Fish Perturbations: P99 Change (Degrees)

| Case | ls | huber10 | huber80 | cauchy | l1_unit | l1_255 | sbl_unit | sbl_255 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fish_fixed_binary | 0.728 | 1.058 | 1.049 | 1.328 | 2.023 | 2.015 | 6.313 | 8.034 |
| fish_fixed_headroom | 0.702 | 1.087 | 1.096 | 1.459 | 2.023 | 2.023 | 6.153 | 8.800 |
| fish_dynamic_binary | 9.635 | 7.625 | 7.938 | 7.823 | 8.896 | 8.896 | 10.565 | 10.286 |
| fish_dynamic_headroom | 2.737 | 2.001 | 1.979 | 2.755 | 3.945 | 3.945 | 9.977 | 9.318 |

## Coverage And Convergence

Counts below use each method's own stopping rule.

| Case | Method | Coverage % | Solved mean iterations | Solved but not converged |
| --- | --- | ---: | ---: | ---: |
| clean_noise_8_binary | l1_unit | 100.00 | 89.2 | 4 |
| clean_noise_8_binary | l1_255 | 100.00 | 130.0 | 5 |
| clean_noise_8_binary | sbl_unit | 100.00 | 144.4 | 3 |
| clean_noise_8_binary | sbl_255 | 100.00 | 993.2 | 505 |
| noisy_color_8_binary | huber10 | 100.00 | 1.5 | 5 |
| noisy_color_8_binary | sbl_unit | 100.00 | 793.9 | 285 |
| noisy_color_8_binary | sbl_255 | 100.00 | 998.1 | 511 |
| noisy_color_8_headroom | huber10 | 100.00 | 1.5 | 5 |
| noisy_color_8_headroom | l1_255 | 100.00 | 196.2 | 5 |
| noisy_color_8_headroom | sbl_unit | 100.00 | 792.7 | 282 |
| noisy_color_8_headroom | sbl_255 | 100.00 | 998.1 | 511 |
| gain_mismatch_irregular_8_binary | huber10 | 100.00 | 9.5 | 352 |
| gain_mismatch_irregular_8_binary | sbl_unit | 100.00 | 794.7 | 284 |
| gain_mismatch_irregular_8_binary | sbl_255 | 100.00 | 1000.0 | 512 |
| gain_mismatch_ring_8_binary | huber10 | 100.00 | 8.8 | 281 |
| gain_mismatch_ring_8_binary | sbl_unit | 100.00 | 892.4 | 395 |
| gain_mismatch_ring_8_binary | sbl_255 | 100.00 | 990.7 | 507 |
| clean_noise_16_binary | l1_unit | 100.00 | 118.0 | 11 |
| clean_noise_16_binary | l1_255 | 100.00 | 168.6 | 21 |
| clean_noise_16_binary | sbl_255 | 100.00 | 975.6 | 479 |
| noisy_color_16_binary | huber10 | 100.00 | 1.4 | 2 |
| noisy_color_16_binary | l1_unit | 100.00 | 128.7 | 13 |
| noisy_color_16_binary | l1_255 | 100.00 | 178.6 | 25 |
| noisy_color_16_binary | sbl_unit | 100.00 | 273.0 | 18 |
| noisy_color_16_binary | sbl_255 | 100.00 | 1000.0 | 512 |
| noisy_color_16_headroom | huber10 | 100.00 | 1.5 | 2 |
| noisy_color_16_headroom | l1_unit | 100.00 | 129.6 | 15 |
| noisy_color_16_headroom | l1_255 | 100.00 | 176.3 | 23 |
| noisy_color_16_headroom | sbl_unit | 100.00 | 273.7 | 17 |
| noisy_color_16_headroom | sbl_255 | 100.00 | 1000.0 | 512 |
| clean_noise_40_binary | l1_unit | 100.00 | 99.2 | 6 |
| clean_noise_40_binary | l1_255 | 100.00 | 150.5 | 17 |
| clean_noise_40_binary | sbl_255 | 100.00 | 899.6 | 385 |
| noisy_color_40_binary | l1_unit | 100.00 | 110.5 | 10 |
| noisy_color_40_binary | l1_255 | 100.00 | 158.1 | 17 |
| noisy_color_40_binary | sbl_unit | 100.00 | 73.6 | 1 |
| noisy_color_40_binary | sbl_255 | 100.00 | 986.2 | 492 |
| noisy_color_40_headroom | huber10 | 100.00 | 1.5 | 1 |
| noisy_color_40_headroom | l1_unit | 100.00 | 118.0 | 10 |
| noisy_color_40_headroom | l1_255 | 100.00 | 169.9 | 19 |
| noisy_color_40_headroom | sbl_unit | 100.00 | 72.5 | 1 |
| noisy_color_40_headroom | sbl_255 | 100.00 | 984.3 | 490 |
| gain_mismatch_irregular_40_binary | huber10 | 100.00 | 9.9 | 388 |
| gain_mismatch_irregular_40_binary | l1_unit | 100.00 | 110.9 | 3 |
| gain_mismatch_irregular_40_binary | l1_255 | 100.00 | 159.9 | 11 |
| gain_mismatch_irregular_40_binary | sbl_unit | 100.00 | 121.1 | 1 |
| gain_mismatch_irregular_40_binary | sbl_255 | 100.00 | 994.3 | 505 |
| gain_mismatch_ring_40_binary | huber10 | 100.00 | 8.2 | 216 |
| gain_mismatch_ring_40_binary | sbl_unit | 100.00 | 238.4 | 2 |
| gain_mismatch_ring_40_binary | sbl_255 | 100.00 | 977.1 | 499 |
| clean_noise_64_binary | l1_unit | 100.00 | 120.3 | 13 |
| clean_noise_64_binary | l1_255 | 100.00 | 172.3 | 26 |
| clean_noise_64_binary | sbl_255 | 100.00 | 835.1 | 325 |
| noisy_color_64_binary | l1_unit | 100.00 | 122.4 | 10 |
| noisy_color_64_binary | l1_255 | 100.00 | 179.3 | 18 |
| noisy_color_64_binary | sbl_255 | 100.00 | 956.3 | 461 |
| noisy_color_64_headroom | l1_unit | 100.00 | 107.6 | 9 |
| noisy_color_64_headroom | l1_255 | 100.00 | 163.7 | 16 |
| noisy_color_64_headroom | sbl_255 | 100.00 | 964.0 | 470 |
| broad_gloss_8_binary | huber10 | 100.00 | 4.4 | 63 |
| broad_gloss_8_binary | sbl_unit | 100.00 | 705.8 | 138 |
| broad_gloss_8_binary | sbl_255 | 100.00 | 998.2 | 255 |
| robust_v1_binary | ls | 98.25 | 0.0 | 0 |
| robust_v1_binary | huber10 | 99.15 | 2.3 | 152 |
| robust_v1_binary | huber80 | 99.15 | 3.4 | 1 |
| robust_v1_binary | cauchy | 99.15 | 10.6 | 1 |
| robust_v1_binary | l1_unit | 99.15 | 26.1 | 0 |
| robust_v1_binary | l1_255 | 99.15 | 36.2 | 0 |
| robust_v1_binary | sbl_unit | 99.15 | 394.1 | 294 |
| robust_v1_binary | sbl_255 | 99.30 | 969.2 | 1911 |
| robust_v1_headroom | ls | 98.25 | 0.0 | 0 |
| robust_v1_headroom | huber10 | 99.15 | 2.3 | 152 |
| robust_v1_headroom | huber80 | 99.15 | 3.4 | 1 |
| robust_v1_headroom | cauchy | 99.15 | 10.6 | 1 |
| robust_v1_headroom | l1_unit | 99.15 | 27.5 | 2 |
| robust_v1_headroom | l1_255 | 99.15 | 37.6 | 2 |
| robust_v1_headroom | sbl_unit | 99.15 | 394.5 | 296 |
| robust_v1_headroom | sbl_255 | 99.30 | 969.2 | 1911 |
| textured_primitives_v1_binary | huber10 | 100.00 | 2.8 | 97 |
| textured_primitives_v1_binary | huber80 | 100.00 | 3.3 | 2 |
| textured_primitives_v1_binary | l1_unit | 100.00 | 154.4 | 86 |
| textured_primitives_v1_binary | l1_255 | 100.00 | 207.8 | 167 |
| textured_primitives_v1_binary | sbl_unit | 100.00 | 207.2 | 71 |
| textured_primitives_v1_binary | sbl_255 | 100.00 | 996.1 | 1977 |
| textured_primitives_v1_headroom | huber10 | 100.00 | 2.8 | 97 |
| textured_primitives_v1_headroom | huber80 | 100.00 | 3.3 | 2 |
| textured_primitives_v1_headroom | l1_unit | 100.00 | 154.2 | 86 |
| textured_primitives_v1_headroom | l1_255 | 100.00 | 207.3 | 167 |
| textured_primitives_v1_headroom | sbl_unit | 100.00 | 207.2 | 71 |
| textured_primitives_v1_headroom | sbl_255 | 100.00 | 996.1 | 1977 |
| holdout_relief_v1_binary | ls | 95.75 | 0.0 | 0 |
| holdout_relief_v1_binary | huber10 | 95.75 | 2.6 | 173 |
| holdout_relief_v1_binary | huber80 | 95.75 | 3.5 | 4 |
| holdout_relief_v1_binary | cauchy | 96.05 | 14.5 | 0 |
| holdout_relief_v1_binary | l1_unit | 95.95 | 128.8 | 54 |
| holdout_relief_v1_binary | l1_255 | 95.95 | 175.9 | 85 |
| holdout_relief_v1_binary | sbl_unit | 96.15 | 438.6 | 381 |
| holdout_relief_v1_binary | sbl_255 | 96.15 | 970.5 | 1853 |
| holdout_relief_v1_headroom | ls | 95.75 | 0.0 | 0 |
| holdout_relief_v1_headroom | huber10 | 95.75 | 2.6 | 173 |
| holdout_relief_v1_headroom | huber80 | 95.75 | 3.5 | 4 |
| holdout_relief_v1_headroom | cauchy | 96.05 | 14.5 | 0 |
| holdout_relief_v1_headroom | l1_unit | 95.95 | 128.5 | 54 |
| holdout_relief_v1_headroom | l1_255 | 95.95 | 175.7 | 84 |
| holdout_relief_v1_headroom | sbl_unit | 96.15 | 438.4 | 381 |
| holdout_relief_v1_headroom | sbl_255 | 96.15 | 970.5 | 1853 |
| fish_fixed_binary | huber10 | 100.00 | 10.0 | 1708 |
| fish_fixed_binary | huber80 | 100.00 | 26.2 | 21 |
| fish_fixed_binary | l1_unit | 100.00 | 140.1 | 56 |
| fish_fixed_binary | l1_255 | 100.00 | 185.0 | 95 |
| fish_fixed_binary | sbl_unit | 100.00 | 404.5 | 178 |
| fish_fixed_binary | sbl_255 | 100.00 | 998.3 | 1723 |
| fish_fixed_headroom | huber10 | 100.00 | 10.0 | 1699 |
| fish_fixed_headroom | huber80 | 100.00 | 26.2 | 31 |
| fish_fixed_headroom | l1_unit | 100.00 | 132.2 | 36 |
| fish_fixed_headroom | l1_255 | 100.00 | 178.1 | 76 |
| fish_fixed_headroom | sbl_unit | 100.00 | 394.5 | 169 |
| fish_fixed_headroom | sbl_255 | 100.00 | 998.0 | 1721 |
| fish_dynamic_binary | huber10 | 100.00 | 10.0 | 1721 |
| fish_dynamic_binary | huber80 | 100.00 | 26.8 | 32 |
| fish_dynamic_binary | l1_unit | 100.00 | 135.5 | 42 |
| fish_dynamic_binary | l1_255 | 100.00 | 181.5 | 80 |
| fish_dynamic_binary | sbl_unit | 100.00 | 380.8 | 142 |
| fish_dynamic_binary | sbl_255 | 100.00 | 997.7 | 1722 |
| fish_dynamic_headroom | huber10 | 100.00 | 10.0 | 1708 |
| fish_dynamic_headroom | huber80 | 100.00 | 26.3 | 31 |
| fish_dynamic_headroom | l1_unit | 100.00 | 124.5 | 33 |
| fish_dynamic_headroom | l1_255 | 100.00 | 169.9 | 55 |
| fish_dynamic_headroom | sbl_unit | 100.00 | 363.6 | 125 |
| fish_dynamic_headroom | sbl_255 | 100.00 | 998.1 | 1723 |

## Rendered Per-Object Mean Error

| Scene / policy / object | ls | huber10 | huber80 | cauchy | l1_unit | l1_255 | sbl_unit | sbl_255 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| robust_v1_binary / floor | 1.576 | 0.863 | 0.829 | 0.722 | 0.765 | 0.765 | 0.528 | 0.740 |
| robust_v1_binary / matte_mound | 0.326 | 0.326 | 0.326 | 0.321 | 0.336 | 0.336 | 0.349 | 0.366 |
| robust_v1_binary / black_gloss | 26.243 | 21.361 | 21.242 | 16.882 | 16.807 | 16.807 | 16.269 | 14.929 |
| robust_v1_binary / rough_gloss | 8.102 | 5.318 | 5.313 | 4.741 | 5.487 | 5.487 | 4.422 | 5.184 |
| robust_v1_headroom / floor | 1.576 | 0.863 | 0.829 | 0.722 | 0.765 | 0.765 | 0.528 | 0.740 |
| robust_v1_headroom / matte_mound | 0.328 | 0.328 | 0.328 | 0.323 | 0.340 | 0.340 | 0.349 | 0.365 |
| robust_v1_headroom / black_gloss | 26.243 | 21.345 | 21.239 | 16.881 | 16.807 | 16.807 | 16.269 | 14.929 |
| robust_v1_headroom / rough_gloss | 8.102 | 5.318 | 5.313 | 4.741 | 5.487 | 5.487 | 4.422 | 5.184 |
| textured_primitives_v1_binary / floor | 1.934 | 0.882 | 0.891 | 0.734 | 0.780 | 0.780 | 0.511 | 0.763 |
| textured_primitives_v1_binary / ripple_patch | 0.405 | 0.214 | 0.214 | 0.198 | 0.238 | 0.238 | 0.258 | 0.257 |
| textured_primitives_v1_binary / striped_rod | 4.621 | 3.173 | 3.171 | 2.843 | 3.045 | 3.045 | 2.670 | 3.153 |
| textured_primitives_v1_binary / dark_gloss_block | 8.497 | 7.635 | 7.635 | 6.102 | 7.717 | 7.717 | 3.907 | 4.988 |
| textured_primitives_v1_binary / textured_disk | 0.455 | 0.308 | 0.308 | 0.297 | 0.365 | 0.365 | 0.419 | 0.431 |
| textured_primitives_v1_headroom / floor | 1.934 | 0.882 | 0.891 | 0.734 | 0.780 | 0.780 | 0.511 | 0.763 |
| textured_primitives_v1_headroom / ripple_patch | 0.405 | 0.214 | 0.214 | 0.198 | 0.238 | 0.238 | 0.258 | 0.257 |
| textured_primitives_v1_headroom / striped_rod | 4.611 | 3.172 | 3.170 | 2.843 | 3.045 | 3.045 | 2.670 | 3.153 |
| textured_primitives_v1_headroom / dark_gloss_block | 8.497 | 7.635 | 7.635 | 6.102 | 7.717 | 7.717 | 3.907 | 4.988 |
| textured_primitives_v1_headroom / textured_disk | 0.455 | 0.308 | 0.308 | 0.297 | 0.365 | 0.365 | 0.419 | 0.431 |
| holdout_relief_v1_binary / floor | 3.380 | 3.144 | 3.141 | 3.105 | 3.083 | 3.083 | 3.020 | 3.078 |
| holdout_relief_v1_binary / corrugated_relief | 2.463 | 2.352 | 2.351 | 2.245 | 2.294 | 2.294 | 2.021 | 2.259 |
| holdout_relief_v1_binary / dark_rod | 18.329 | 18.329 | 18.329 | 20.617 | 19.956 | 19.956 | 19.946 | 21.189 |
| holdout_relief_v1_binary / rough_block | 5.559 | 5.556 | 5.556 | 5.311 | 5.082 | 5.082 | 5.689 | 5.229 |
| holdout_relief_v1_binary / textured_medallion | 4.248 | 4.248 | 4.248 | 4.213 | 4.283 | 4.283 | 4.202 | 4.267 |
| holdout_relief_v1_headroom / floor | 3.378 | 3.141 | 3.139 | 3.104 | 3.082 | 3.082 | 3.020 | 3.077 |
| holdout_relief_v1_headroom / corrugated_relief | 2.463 | 2.352 | 2.351 | 2.245 | 2.294 | 2.294 | 2.021 | 2.259 |
| holdout_relief_v1_headroom / dark_rod | 18.329 | 18.329 | 18.329 | 20.617 | 19.956 | 19.956 | 19.946 | 21.189 |
| holdout_relief_v1_headroom / rough_block | 5.559 | 5.556 | 5.556 | 5.311 | 5.082 | 5.082 | 5.689 | 5.229 |
| holdout_relief_v1_headroom / textured_medallion | 4.248 | 4.248 | 4.248 | 4.213 | 4.283 | 4.283 | 4.202 | 4.267 |
