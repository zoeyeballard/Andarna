"""Probe which MuJoCo GL backend can render offscreen on this headless box.
gym-aloha / LIBERO eval need to render camera frames each step."""
import os, sys

backend = sys.argv[1] if len(sys.argv) > 1 else "egl"
os.environ["MUJOCO_GL"] = backend
try:
    import mujoco
    xml = "<mujoco><worldbody><geom type='box' size='.1 .1 .1' rgba='0 .9 0 1'/></worldbody></mujoco>"
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, height=120, width=160)
    mujoco.mj_forward(m, d)
    r.update_scene(d)
    img = r.render()
    print(f"MUJOCO_GL={backend:8s} -> RENDER OK, frame shape {img.shape}, mean px {img.mean():.1f}")
except Exception as e:
    print(f"MUJOCO_GL={backend:8s} -> FAIL: {type(e).__name__}: {str(e)[:110]}")
