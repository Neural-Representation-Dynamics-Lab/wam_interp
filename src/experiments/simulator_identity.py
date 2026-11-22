"""Render complete native object assets in exchanged slots; reproducible layouts."""
import numpy as np
from .object_identity import exchanged_visual_positions
from .simulator_materials import ObjectMaterialCapture
from .material_cues import swapped_materials, robot_pixel_audit
from .simulator_video import dilate


def reset_with_layout(env, initial_state, *, instruction, metadata, offsets):
    """Apply explicit XY offsets before the caller's ordinary warm-up.

    The returned raw observation predates the edit. Callers must execute their
    declared warm-up and use its last observation, as LiberoPilot does.
    """
    from wam_interp.evaluations.LIBERO.environment import reset_to_initial_state
    observation = reset_to_initial_state(env, initial_state, instruction=instruction, metadata=metadata)
    model, data = env.sim.model, env.sim.data
    for name, xy in offsets.items():
        xy = np.asarray(xy, dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all() or np.abs(xy).max() > .05:
            raise ValueError('Layout offsets must be finite XY translations within 5 cm')
        body = env.env.obj_body_id[name]
        joints = np.flatnonzero(model.jnt_bodyid == body)
        if len(joints) != 1 or model.jnt_type[joints[0]] != 0:
            raise ValueError('Layout edit requires one free joint per object')
        address = int(model.jnt_qposadr[joints[0]])
        data.qpos[address:address + 2] += xy
    env.sim.forward()
    return observation


class ObjectAssetCapture(ObjectMaterialCapture):
    """Native sensor-time original, full-asset swap and material-only images."""
    def _render(self, *args, **kwargs):
        image = self.render(*args, **kwargs)
        cam = kwargs.get('camera_name')
        if not self.enabled or cam not in ('agentview', 'robot0_eye_in_hand') or kwargs.get('depth') or kwargs.get('segmentation'):
            return image
        before, time = self.snapshot(), float(self.sim.data.time)
        positions = self.sim.data.geom_xpos.copy()
        orientations = self.sim.data.geom_xmat.copy()
        opts = {**kwargs, 'depth': True, 'segmentation': False}
        rgb, depth = self.render(*args, **opts)
        if not np.array_equal(image, rgb):
            raise RuntimeError('Native repeated render differs')
        groups = self.sim._render_context_offscreen.vopt.sitegroup
        saved_groups = groups.copy()
        h, w = image.shape[:2]; scale = self.coverage_scale
        segopts = {**kwargs, 'depth': False, 'segmentation': True}
        def segmentation():
            try:
                groups[:] = 0
                low = self.render(*args, **segopts)
                high = self.render(**{**segopts, 'height': h*scale, 'width': w*scale})
                return low, high
            finally:
                groups[:] = saved_groups
        def mask(high, ids):
            return ((high[..., 0] == self.geom_type) & np.isin(high[..., 1], ids)).reshape(h, scale, w, scale).any(axis=(1, 3))
        seg, high = segmentation()
        support = mask(high, self.geom_ids)
        robot_support, gripper_support = mask(high, self.robot_geoms), mask(high, self.gripper_geoms)
        centres = self.sim.data.body_xpos[self.sim.model.geom_bodyid[self.geom_ids]].copy()
        with exchanged_visual_positions(self.sim.data.geom_xpos, self.geom_ids, centres):
            edited, edited_depth = self.render(*args, **opts)
            edited_seg, edited_high = segmentation()
        support |= mask(edited_high, self.geom_ids)
        with swapped_materials(self.sim.model.geom_matid, self.geom_ids):
            material, material_depth = self.render(*args, **opts)
        if not np.array_equal(image, self.render(*args, **kwargs)):
            raise RuntimeError('Native image restoration differs')
        after = self.snapshot()
        if time != self.sim.data.time or any(not np.array_equal(v, after[k]) for k, v in before.items()):
            raise RuntimeError('Render changed simulator or camera state')
        if not np.array_equal(positions, self.sim.data.geom_xpos) or not np.array_equal(orientations, self.sim.data.geom_xmat):
            raise RuntimeError('Visual geometry restoration differs')
        robot, audit = robot_pixel_audit(rgb, edited, robot_support, support)
        outside = ~dilate(support, 2)
        difference = np.abs(rgb.astype(float) - edited.astype(float))
        diagnostics = {'outside_support_changed_pixels': int((np.any(difference > 0, axis=-1) & outside).sum()),
                       'outside_support_mae_255': float(difference[outside].mean()) if outside.any() else 0.,
                       'outside_support_depth_exact': bool(np.array_equal(depth[outside], edited_depth[outside])),
                       'robot_interior_rgb_exact': not audit['robot_interior_changed'],
                       'physics_camera_and_geometry_restored_exact': True}
        c = self.convention
        self.latest[cam] = {'rgb': rgb[::c].copy(), 'depth': depth[::c].copy(), 'mask': support[::c].copy(),
            'robot_mask': robot[::c].copy(), 'gripper_mask': (gripper_support & ~dilate(support, 2))[::c].copy(),
            'robot_support_mask': robot_support[::c].copy(), 'gripper_support_mask': gripper_support[::c].copy(),
            'segmentation': seg[::c].copy(), 'edited_segmentation': edited_seg[::c].copy(),
            'robot_pixel_audit': audit, 'asset_diagnostics': diagnostics,
            'variant_diagnostics': {'swapped': diagnostics},
            'variants': {'swapped': edited[::c].copy(), 'material_only': material[::c].copy()},
            'variant_depths': {'swapped': edited_depth[::c].copy(), 'material_only': material_depth[::c].copy()},
            'sensor_time': time, 'sensor_state': before, 'sham_exact': True}
        self.calls += 1
        return image
