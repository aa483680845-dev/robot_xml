import time
import mujoco
import mujoco.viewer
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "scene.xml"

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as v:
    real_start = time.time()
    while v.is_running():
        mujoco.mj_step(model, data)
        v.sync()

        # 等待直到 wall-clock 与仿真时间对齐（实时速率）
        elapsed = time.time() - real_start
        if data.time > elapsed:
            time.sleep(data.time - elapsed)
