# Fixed baseline results

Fixed deterministic seed 7, 100 episodes each, completed with plasticity disabled. SAVE/GOAL are logged only and do not become modulation or reward.

| Controller | Saves | Goals | Save rate | Wall time | Brain time |
| --- | ---: | ---: | ---: | ---: | ---: |
| MaleCNS DNp20 decoder | 26 | 74 | 26% | 57.16 s | 28.0 s |
| RandomGoalkeeper | 21 | 79 | 21% | 47.67 s | 28.0 s |

The MaleCNS run made 1,400 actions: LEFT 134 (9.6%), STAY 694 (49.6%), RIGHT 572 (40.9%). It therefore showed a pronounced rightward relative bias and mostly stayed still. Random made LEFT 461, STAY 454, RIGHT 485. These single-seed results are descriptive—not evidence of skill or superiority. Exact artifacts: `workspace/outputs/flykeeper/89c18157f5c444e087cc8f9514788836/` and `workspace/outputs/flykeeper/f3b34bc148684f6582e6c523b1329cbf/`.

Biological data: MaleCNS connectivity, source body IDs, annotations, and retained R1-R6 mapping. Engineered assumptions: pixel renderer, timing, luminance display, DNp20 left/right action association, threshold rule, and goalkeeper motion. This is not an exact fly or evidence of football understanding, intelligence, or learning.
