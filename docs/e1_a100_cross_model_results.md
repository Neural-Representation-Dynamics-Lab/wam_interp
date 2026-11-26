# A100 checks on four additional WAMs

Each released checkpoint ran a native bf16 episode with its own sampler and replanning or history protocol on the official LIBERO-Long bowl task. We tested three decisions in each episode. On every checked decision, capturing the selected future features and reinserting those same features each reproduced the uninstrumented action exactly. The captured native and replay feature tensors also agreed exactly. Replacing them with zeros changed the predicted actions. These replay checks concern the selected site, layer and denoising stage.

| Backend | Future representation tested | Bowl episode, controls | Largest normalized action change from zeroing, range over three decisions | Matched scenes | Mean target minus distractor endpoint shift, mm |
|---|---|---|---:|---:|---:|
| Faster-WAM | Sparse fused future K/V, layer 16 | Success, 224 | 0.03125–0.03516 | 2/8 | -0.130 |
| LingBot-VA | Predicted video cache, layer 15 | Success, 241 | 0.00781–0.01953 | 2/8 | +0.083 |
| Long-WAM IDM | Partially noisy future, before history concatenation | Success, 227 | 0.40039–0.56055 | 2/8 | +4.979 |
| OpenWAM | Action-facing video K/V, layer 15, first denoising call | Success, 224 | 0.00781–0.00781 | 2/8 | +0.024 |

In a separate test at the first decision, we moved either the physical target or a distractor within the range of released object positions. The donor and recipient received the same language and action seed. Only the selected donor features were transplanted into the original scene. The metric projects the change in executed gripper endpoint after ten controls along the target relocation axis. All eight candidate states are accounted for. Six failed the physical move-matching criterion. The two book states remaining passed the visual-magnitude check. These two states do not support ranking backends or a general claim of semantic selectivity.

These features carry information from the donor's current scene and its model computation. Their noise levels, history, attention layout and scales differ across backends. A layer-level feature transplant is a separate intervention from moving one object in FastWAM's decoded future with fixed robot motion. One successful episode is a smoke check for that checkpoint.

See the [reproduction guide](e1_a100_study.md) and [held-out FastWAM result](e1_a100_fixed_motion_results.md).
