# FastWAM Optional IDM on the A100

We tested two official LIBERO-Long tasks with one named movable object. Four initial states per task each contributed an official and a physically relocated layout. The relocation moved the free joint to a position from another released state of the same task, typically about three centimetres away. Each layout ran once with IDM and once with first-frame conditioning using the same weights, seed sequence and native ten-step replanning. We used bf16, MuJoCo 3.3.2 and the released Optional IDM sampler shift.

| Task | Official, IDM | Relocated, IDM | Official, first frame | Relocated, first frame | Paired IDM contact criterion |
|---|---:|---:|---:|---:|---:|
| Black bowl into drawer, then close drawer | 4/4 | 4/4 | 4/4 | 3/4 | 2/4 states |
| Book into back compartment of caddy | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 states |

The contact criterion requires success in both layouts and a gripper displacement along the relocation axis of at least half the object's displacement at first recorded contact. We set this exploratory criterion after the first bowl state began returning results. The book qualified for the fixed-motion intervention. The bowl illustrates why task completion and tracking a small displacement deserve separate measurements. Of the 16 matched success comparisons, 15 succeeded in both modes and one succeeded only in IDM. Eight starting states across two tasks provide little evidence about a general success advantage.

Across the 16 IDM episodes, we compared all 369 forecasts to their own executed prefix. Each ten-control prefix offers observations at offsets four and eight when the episode lasts that long. In the external 224-pixel view, local gripper displacement had mean error 1.47 px versus 6.61 px for a stationary prediction, with 475/721 eligible offsets tracked. Target-object error was 0.58 px versus 1.87 px, with 473/721 tracked. Errors weight episodes equally. On the subset with four preceding controls, constant-velocity errors were 2.84 px for gripper and 1.39 px for target, versus forecast errors of 1.47 and 0.61 px on that subset. Tracking selects visible local features using forward/backward checks and agreement with projected executed motion. Occluded and difficult points can be underrepresented. Independent landmark review remains pending.

The simulator-only initial-velocity study tested six speeds in each direction on one reset per object. Only the bowl achieved the required planar displacement, height stability and rotation bounds in both directions. A physically matched target/distractor pair could not be formed with those stimuli. The prescribed-render follow-up tests smooth visual translation while replaying identical robot motion. Its interpretation concerns the displayed motion of an externally moved object.

The [study commands and protocols](e1_a100_study.md) reproduce the native test and forecast scoring. The experiment writes full JSON, saved latent forecasts and matched image frames under the local results directory.
