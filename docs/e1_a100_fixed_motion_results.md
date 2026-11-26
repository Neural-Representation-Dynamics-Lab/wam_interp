# FastWAM fixed-motion book experiment

We tested whether a reach follows a depicted change in the book's location more than a comparable change in the mug's location. The initial observation, future robot trajectory, cameras, recipient instruction and action noise were paired. Each future showed a three-centimetre leftward or rightward translation of one object. The recorded effect is half the difference between the executed gripper's two final world-x coordinates after 32 controls. A positive value means the reach shifted in the depicted translation direction.

Both the source edit and unedited video encoding passed on all four book development states. We then attempted ten held-out book states twice each. Of 20 seed runs, 15 passed the visual edit checks and 15 passed the unedited encoding control. **7 of ten held-out states** contributed both seeds. Estimates weight starting states equally. The descriptive 95% intervals resample whole starting states.

| Displayed object | Signed gripper half-difference, mm | State-bootstrap 95% interval, mm |
|---|---:|---:|
| Book (target) | -3.73 | -4.35 to -3.13 |
| Mug (distractor) | -7.20 | -7.96 to -6.34 |
| Book minus mug | +3.47 | +2.63 to +4.35 |

The mean three-dimensional endpoint half-distance was 4.19 mm for the book and 9.23 mm for the mug. The signed mug interval fell entirely below the lower bound of the prespecified ±5 mm engineering margin. Signed means can conceal individual responses in opposite directions.

The alternate instruction caused first contact with the mug in 0 of 15 eligible seed runs. This left zero valid instruction-by-property comparisons. Both object edits shifted the reach against the depicted motion, and the mug produced the larger shift. The distinct appearance and geometry of the book and mug still confound object identity with task relevance. These data describe this task, this prescribed visual translation and this early approach phase. An independent reviewer can inspect blinded clips using the [study instructions](e1_a100_study.md).

## Depicted robot-motion control

We replayed a pair of native robot trajectories in each of the four book development states, varying a world-x control bias and keeping both objects stationary. The first qualifying pair in a fixed list had two to four centimetres of gripper displacement in each direction, under one millimetre of object drift, and zero detected robot–scene contact substeps. Every state passed the source and unedited encoding checks. The average depicted gripper half-difference was +34.92 mm. The executed gripper half-difference was **+27.52 mm**, with state values +28.78, +27.16, +28.11 and +26.01 mm.

The wrist camera moved with the arm in this control. The four states were development cases. Together with the held-out book and mug edits, these results support sensitivity to depicted arm motion and show that an irrelevant object's visual motion can also change the reach. They leave open which visual cues account for each effect.
