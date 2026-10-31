Proxy CPU: odkurzacz; phone = proxy x 1.44 (BlazePose single-thread vs Pixel 4)

| component | proxy ms / photo | models MB |
|---|---|---|
| person detector (BlazePose) | 7 | 3 |
| MediaPipe Pose heavy | 115 | 29 |
| NLF-S | 103 | 108 |
| NLF-L | 337 | 496 |
| MoGe-2-S | 3359 | 134 |
| MoGe-2-B | 5151 | 400 |
| MoGe-2-L | 10686 | 1263 |
| Qwen3-VL-2B Q4 @640 px, 14 questions | 38932 (image 11133 + 1986/question) | 1481 |

| configuration | photos | proxy s | est. phone s | models MB | feasible (<= 60 s, <= 2 GB) |
|---|---|---|---|---|---|
| people only: NLF-S, 1 photo | 1 | 0.1 | 0.1 | 108 | yes |
| people only: NLF-S 2D + triangulation, 5 photos | 5 | 0.5 | 0.7 | 108 | yes |
| people only: NLF-L, 1 photo | 1 | 0.3 | 0.5 | 496 | yes |
| people only: MediaPipe heavy 2D + triangulation, 5 photos | 5 | 0.6 | 0.8 | 29 | yes |
| full: NLF-S + Qwen3-VL-2B@640 + MoGe-2-S depth, 1 photo | 1 | 42.4 | 60.8 | 1723 | no |
| full: NLF-S + Qwen3-VL-2B@640 + MoGe-2-L depth, 1 photo | 1 | 49.7 | 71.4 | 2852 | no |
| full: NLF-S + Qwen3-VL-2B@640 + triangulation, 2 photos | 2 | 78.1 | 112.1 | 1589 | no |
| full: NLF-S + Qwen3-VL-2B@640 + triangulation, 5 photos | 5 | 195.2 | 280.1 | 1589 | no |