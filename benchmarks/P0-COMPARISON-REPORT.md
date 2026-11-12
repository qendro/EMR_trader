# P0 partially observed CartPole comparison

All four methods used the frozen five evolution seeds, 100 generations, 100 paired held-out episodes and the 500-step normalization scale. Each method retained its own genome operators and matching G9 pilot.

| Method | Held-out means by seed | Median paired gain | 95% seed bootstrap interval | Raw target seeds | Research target |
| --- | --- | ---: | ---: | ---: | --- |
| adaptive | 47.00, 57.99, 85.31, 59.10, 52.17 | 6.86% | [4.87%, 12.83%] | 0/5 | open |
| direct | 44.33, 43.84, 44.91, 42.07, 43.60 | 4.34% | [4.03%, 4.80%] | 0/5 | open |
| fixed | 37.23, 38.96, 284.39, 39.69, 35.63 | 3.20% | [2.92%, 52.65%] | 1/5 | open |
| weight | 500.00, 500.00, 462.62, 474.40, 500.00 | 95.27% | [88.30%, 96.08%] | 5/5 | pass |

The research target requires median paired normalized improvement of at least 20% and a positive lower bootstrap endpoint resampled over evolution seeds. The raw 120-step seed count is a separate diagnostic. A failed method remains in the table.

The random-action baseline and known finite-difference controller are recorded separately in the P0 quality report. This comparison supports method diagnosis; it does not replace M4 scale, stability or both-card performance gates.
