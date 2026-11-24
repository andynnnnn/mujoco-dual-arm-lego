"""Two-Panda tabletop simulation for the XDOF Count Legos task."""

import argparse
import base64
from collections import Counter
from copy import deepcopy
from io import BytesIO
from pathlib import Path
import json
import math
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lego_recording import RecordingDashboard


ROOT = Path(__file__).resolve().parent
TABLE = 0.32
CAMERA = np.array([0.49, 0.0, 1.05])
FOV = 43
SIZE = 768
BIN = np.array([0.625, 0.0])
BINSIZE = np.array([0.071, 0.115])
BIN_FLOOR = TABLE + .006
RELEASE_GAP = .004
MAX_RELEASE_GAP = .010
ARM_START = np.array([0., -0.75, 0., -2.10, 0., 1.56, 0.78])
COLORS = {"red": ".80 .025 .045 1", "orange": "1 .30 .008 1", "blue": ".02 .24 .78 1", "green": ".06 .60 .24 1"}
QUOTAS = {"red": 2, "orange": 3}
BRICK_HALF = np.array([.024, .012])
TABLE_CENTER = np.array([.46, 0.])
TABLE_HALF = np.array([.34, .42])
DROP_CLEARANCE = .08
GRIP_TARGET_N = 2.5  # Per finger pad, tuned in this simulated scene.
GRIP_MIN_N = .55
SUPPORT_MIN_N = .06
SUPPORT_STEPS = 4


def box(parent, name, pos, size, rgba, contype, affinity, **attributes):
    return ET.SubElement(parent, "geom", name=name, type="box",
                         pos=" ".join(map(str, pos)), size=" ".join(map(str, size)),
                         rgba=rgba, contype=str(contype), conaffinity=str(affinity), **attributes)


def rotation2(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def overhead_point(u, v, height):
    scale = 2 * (CAMERA[2] - height) * math.tan(math.radians(FOV / 2)) / SIZE
    return CAMERA[:2] + (np.array([v, u]) + .5 - SIZE / 2) * scale


def clustered_layout(rng, bin_center, bin_axes):
    # Two distractors cover red pieces, so even a correct count requires clearing at least one stack.
    bottom = ["red", "red", *rng.permutation(["red"] + ["orange"] * 4 + ["blue"])]
    centers = np.array([[.380, -.090], [.445, .090]]) + rng.uniform(-.018, .018, (2, 2))
    footprints = [(bin_center, bin_axes, BINSIZE + .006)]
    layout = []
    for i, color in enumerate(bottom):
        for _ in range(1500):
            xy = centers[i % 2] + rng.normal(0, [.040, .047])
            yaw = rng.uniform(-math.pi / 2, math.pi / 2)
            axes = rotation2(yaw)
            if not (.326 < xy[0] < .503 and abs(xy[1]) < .205):
                continue
            clear = True
            for other, basis, half in footprints:
                # Separating axes for two oriented rectangles, including a small spawn gap.
                directions = np.column_stack((axes, basis))
                distance = np.abs((xy - other) @ directions)
                extent = np.abs(directions.T @ axes) @ BRICK_HALF + np.abs(directions.T @ basis) @ half
                if not np.any(distance > extent + .003):
                    clear = False
                    break
            if clear:
                break
        else:
            raise ValueError("Could not sample an irregular source cluster")
        footprints.append((xy, axes, BRICK_HALF))
        layout.append((color, *xy, TABLE + .012, yaw))
    for i, color in enumerate(("blue", "green")):
        _, x, y, _, yaw = layout[i]
        jitter = rotation2(yaw) @ rng.uniform([-.005, -.003], [.005, .003])
        layout.append((color, x + jitter[0], y + jitter[1], TABLE + .035, yaw + rng.uniform(-.12, .12)))
    return layout


def scene(seed, clutter=False, force_feedback=False):
    rng = np.random.default_rng(seed)
    tree = ET.parse(ROOT / "generated/panda_scene.xml")
    root = tree.getroot()
    for asset in root.iter():
        path = asset.get("file")
        if path is not None:
            asset.set("file", str(ROOT / path))
    root.set("model", "Count Legos — dual Panda")
    original = root.find("worldbody")
    world = ET.Element("worldbody")
    root.remove(original)
    root.append(world)
    equality = root.find("equality")
    finger_mimic = deepcopy(equality[0])
    equality.clear()
    for side, offset in (("left", -.29), ("right", .29)):
        base = ET.SubElement(world, "body", name=side + "_base", pos=f"0 {offset} .19")
        for child in original:
            if child.tag == "body" or (child.tag == "geom" and child.get("type") == "mesh"):
                base.append(deepcopy(child))
        for item in base.iter():
            if item is not base and "name" in item.attrib:
                item.set("name", side + "_" + item.get("name"))
            if item.tag == "geom" and item.get("contype", "1") != "0":
                item.set("contype", "1")
                item.set("conaffinity", "3")
        for sign, name in ((1, "leftfinger"), (-1, "rightfinger")):
            finger = base.find(f".//body[@name='{side}_panda_{name}']")
            for geom in finger.findall("geom"):
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
            box(finger, f"{side}_{name}_pad", (0, sign * .006, .044), (.012, .006, .007),
                ".14 .18 .20 1", 8, 6, friction="2.5 .015 .002", priority="1",
                solref=".004 1", solimp=".98 .999 .001", condim="4")
        hand = base.find(f".//body[@name='{side}_panda_link7']/frame/frame")
        ET.SubElement(hand, "site", name=side + "_tcp", pos="0 0 .1024", size=".002", rgba="0 0 0 0")
        sign = -1 if side == "left" else 1
        ET.SubElement(hand, "camera", name=side + "_wrist", pos=f"0 {sign * .068} -.008",
                      xyaxes=f"1 0 0 0 -.86 {-sign * .51}", fovy="69")
        mimic = deepcopy(finger_mimic)
        for key in ("joint1", "joint2"):
            mimic.set(key, side + "_" + mimic.get(key))
        equality.append(mimic)

    box(world, "table", (*TABLE_CENTER, TABLE - .025), (*TABLE_HALF, .025), ".69 .72 .75 1", 2, 13)
    bin_center = BIN if not clutter else np.array([rng.uniform(.595, .635), rng.uniform(-.068, .068)])
    bin_yaw = 0 if not clutter else rng.choice((-1, 1)) * rng.uniform(.14, .54)
    container = ET.SubElement(world, "body", name="sorting_bin", pos=f"{bin_center[0]} {bin_center[1]} 0",
                              euler=f"0 0 {bin_yaw}")
    box(container, "bin_floor", (0, 0, TABLE + .003), (*BINSIZE, .003), ".20 .31 .39 1", 2, 13)
    for axis in (0, 1):
        for sign in (-1, 1):
            center = np.zeros(2)
            center[axis] += sign * (BINSIZE[axis] + .003)
            half = BINSIZE.copy()
            half[axis] = .003
            half[1 - axis] += .006
            box(container, f"bin_wall_{axis}_{sign}", (*center, TABLE + .022), (*half, .022), ".24 .43 .53 1", 2, 13)
    for x, y in ((0, -.29), (0, .29)):
        box(world, f"pedestal_{y}", (x, y, .09), (.095, .095, .10), ".24 .29 .34 1", 0, 0)
    ET.SubElement(world, "geom", name="floor", type="plane", size="2 2 .1", rgba=".33 .39 .46 1",
                  contype="0", conaffinity="0")
    for pos in ("0 -.5 1.8", "1 .6 1.7"):
        ET.SubElement(world, "light", pos=pos, dir="0 0 -1", diffuse=".55 .55 .55", castshadow="false")
    ET.SubElement(world, "camera", name="overhead", pos=" ".join(map(str, CAMERA)),
                  xyaxes="0 1 0 -1 0 0", fovy=str(FOV))
    ET.SubElement(world, "camera", name="overview", pos="1.06 -1.02 1.01",
                  xyaxes=".88 .48 0 -.24 .44 .86", fovy="48")

    if clutter:
        layout = clustered_layout(rng, bin_center, rotation2(bin_yaw))
    else:
        colors = rng.permutation(["red"] * 3 + ["orange"] * 4 + ["blue"] * 2 + ["green"])
        locations = [(x, y) for x in (.335, .415, .490) for y in (-.205, -.07, .07, .205)]
        layout = []
        for color, index in zip(colors, rng.permutation(len(locations))):
            x, y = np.array(locations[index]) + rng.uniform(-.006, .006, 2)
            yaw = 0 if rng.integers(2) == 0 else math.pi / 2
            layout.append((color, x, y, TABLE + .013, yaw))
    bricks = {}
    for i, (color, x, y, z, yaw) in enumerate(layout):
        name = f"brick_{i:02d}"
        bricks[name] = color
        brick = ET.SubElement(world, "body", name=name, pos=f"{x} {y} {z}", euler=f"0 0 {yaw}")
        ET.SubElement(brick, "freejoint")
        box(brick, name + "_base", (0, 0, 0), (*BRICK_HALF, .011), COLORS[color], 4, 15 if force_feedback else 14,
            mass=".018", friction=".9 .005 .001", group="2")
        for row in (-.006, .006):
            for col in (-.018, -.006, .006, .018):
                ET.SubElement(brick, "geom", type="cylinder", pos=f"{col} {row} .012",
                              size=".0038 .002", rgba=COLORS[color], contype="0", conaffinity="0", mass="0", group="2")
    return ET.tostring(root, encoding="unicode"), bricks


class Arm:
    def __init__(self, model, side):
        self.side = side
        joints = [model.joint(f"{side}_panda_joint{i}").id for i in range(1, 8)]
        finger = [model.joint(f"{side}_panda_finger_joint{i}").id for i in (1, 2)]
        self.q = model.jnt_qposadr[joints]
        self.v = model.jnt_dofadr[joints]
        self.fq = model.jnt_qposadr[finger]
        self.fv = model.jnt_dofadr[finger]
        self.limits = model.jnt_range[joints]
        self.site = model.site(side + "_tcp").id
        self.pad_ids = tuple(model.geom(f"{side}_{finger}_pad").id for finger in ("leftfinger", "rightfinger"))
        self.pads = set(self.pad_ids)
        self.target = ARM_START.copy()
        self.finger = .036
        self.retained_brick = None


class ForceStop(ValueError):
    pass


class ContactFeedback:
    guarded = {"approach", "grasp", "lift", "carry", "separate", "lower", "lower_table", "support", "inspect"}
    carrying = {"carry", "separate", "lower", "lower_table", "support"}
    holding = carrying | {"lift"}

    def __init__(self, model, arms, bricks, enabled):
        self.model = model
        self.bricks = bricks
        self.enabled = enabled
        self.pad_map = {pad: (side, finger) for side, arm in arms.items()
                        for finger, pad in enumerate(arm.pad_ids)}
        self.geom_names = [model.geom(i).name for i in range(model.ngeom)]
        self.geom_bodies = model.geom_bodyid
        bodies = [model.body(i).name for i in self.geom_bodies]
        self.geom_arms = [next((side for side in arms if name.startswith(side + "_")), None) for name in bodies]
        self.fingers = {side: [0., 0.] for side in arms}
        self.wrench = np.empty(6)
        self.begin(None, None)

    def begin(self, arm, brick):
        self.arm, self.brick = arm, brick
        self.events = []
        self.floor = None
        self.loads = np.zeros(2)
        self.peak_grip = np.zeros(2)
        self.peak_unexpected = 0.
        self.unexpected = 0.
        self.support = 0.
        self.support_at_release = None
        self.grip_steps = self.loss_steps = self.support_steps = self.contact_steps = 0
        self.latched = False

    def sample(self, data, phase):
        arm, brick = self.arm, self.brick
        side = arm.side if arm else None
        target_loads = np.zeros(2)
        unwanted = Counter()
        for loads in self.fingers.values():
            loads[:] = [0., 0.]
        self.support = 0.
        for i, contact in enumerate(data.contact):
            if contact.efc_address < 0:
                continue
            g1, g2 = contact.geom1, contact.geom2
            b1, b2 = self.geom_bodies[g1], self.geom_bodies[g2]
            pads = [(self.pad_map[g], body) for g, body in ((g1, b2), (g2, b1)) if g in self.pad_map]
            supporting = brick is not None and ((b1 == brick and g2 == self.floor) or (b2 == brick and g1 == self.floor))
            obstacle = None
            if side and phase in self.guarded:
                if self.geom_arms[g1] == side and self.geom_arms[g2] != side:
                    if not (g1 in arm.pads and b2 == brick):
                        obstacle = g2
                elif self.geom_arms[g2] == side and self.geom_arms[g1] != side:
                    if not (g2 in arm.pads and b1 == brick):
                        obstacle = g1
                elif brick is not None and phase in self.carrying and not supporting:
                    if b1 == brick:
                        obstacle = g2
                    elif b2 == brick:
                        obstacle = g1
            if not (pads or supporting or obstacle is not None):
                continue
            mj.mj_contactForce(self.model, data, i, self.wrench)
            normal = max(0., float(self.wrench[0]))
            for (pad_side, finger), other in pads:
                self.fingers[pad_side][finger] += normal
                if pad_side == side and other == brick:
                    target_loads[finger] += normal
            if supporting:
                self.support += normal
            if obstacle is not None:
                unwanted[obstacle] += normal

        self.peak_grip = np.maximum(self.peak_grip, target_loads)
        self.loads += .22 * (target_loads - self.loads)
        self.support_steps = self.support_steps + 1 if self.support >= SUPPORT_MIN_N else 0
        self.grip_steps = self.grip_steps + 1 if min(self.loads) >= GRIP_MIN_N else 0
        self.unexpected = max(unwanted.values(), default=0.)
        self.peak_unexpected = max(self.peak_unexpected, self.unexpected)
        if not (self.enabled and arm and phase in self.guarded):
            return
        for geom, normal in unwanted.most_common():
            neighbor = self.geom_bodies[geom] in self.bricks
            threshold, hard = (.75, 4.) if neighbor else (2., 8.)
            if normal >= threshold:
                self.contact_steps += 1
                if self.contact_steps >= 2 or normal >= hard:
                    self.stop("neighbor_contact" if neighbor else "obstacle_contact", phase, normal, self.geom_names[geom])
                break
        else:
            self.contact_steps = 0
        if phase in self.holding:
            self.loss_steps = self.loss_steps + 1 if min(self.loads) < GRIP_MIN_N else 0
            if self.loss_steps >= 15:
                self.stop("grip_loss", phase, min(self.loads))

    def regulate(self, data, phase):
        if not (self.enabled and self.arm and self.brick is not None and phase in self.holding | {"grasp", "retreat"}):
            return
        if not self.latched and max(self.loads) >= .25:
            self.latched = True
            self.arm.finger = max(self.arm.finger, float(np.mean(data.qpos[self.arm.fq])) - .002)
        if self.latched:
            # Change the common finger position reference based on the higher measured pad load.
            speed = np.clip(.008 * (max(self.loads) - GRIP_TARGET_N), -.018, .028)
            self.arm.finger = float(np.clip(self.arm.finger + speed * self.model.opt.timestep, .003, .037))

    def stop(self, code, phase, force=0., geom=None):
        self.events.append({"code": code, "phase": phase, "force_n": round(float(force), 2), "geom": geom})
        name = {"neighbor_contact": "Contact with a neighboring brick", "obstacle_contact": "Unexpected obstacle contact",
                "grip_loss": "One finger lost its grip", "weak_grip": "Stable two-finger grip not confirmed",
                "no_support": "Support not confirmed; the gripper stayed closed"}[code]
        raise ForceStop(f"{name} ({phase}, {force:.2f} N)" + (f": {geom}" if geom else ""))

    def telemetry(self):
        return {"finger_n": {side: [round(n, 2) for n in values] for side, values in self.fingers.items()},
                "support_n": round(self.support, 2), "unexpected_n": round(self.unexpected, 2),
                "event": self.events[-1] if self.events else None}

    def summary(self):
        return {"peak_finger_n": [round(float(n), 2) for n in self.peak_grip],
                "peak_unexpected_n": round(self.peak_unexpected, 2),
                "support_before_release_n": self.support_at_release, "events": self.events.copy()}


class Lab:
    def __init__(self, seed, output, live=False, clutter=False, force_feedback=False, pose_policy=None,
                 record_dashboard=False):
        self.output = output
        self.live = live
        self.clutter = clutter
        self.pose_policy = pose_policy
        self.pose_records = []
        self.dashboard = RecordingDashboard() if record_dashboard or (live and pose_policy == "predicted") else None
        self.record_cameras = live or self.dashboard is not None
        self.next_frame = time.monotonic()
        output.mkdir(parents=True, exist_ok=True)
        xml, names = scene(seed, clutter, force_feedback)
        self.model = mj.MjModel.from_xml_string(xml)
        if self.record_cameras:
            camera = self.model.camera("overview")
            camera.pos[:] = [1.32, -.93, 1.56]
            front = np.array([.35, -.01, .43]) - camera.pos
            front /= np.linalg.norm(front)
            right = np.cross(front, [0, 0, 1])
            right /= np.linalg.norm(right)
            up = np.cross(right, front)
            mj.mju_mat2Quat(camera.quat, np.column_stack((right, up, -front)).flatten())
            self.model.cam_fovy[camera.id] = 44
        self.model.opt.timestep = .002
        self.model.vis.global_.offwidth = self.model.vis.global_.offheight = SIZE
        self.data = mj.MjData(self.model)
        self.ik = mj.MjData(self.model)
        self.arms = {side: Arm(self.model, side) for side in ("left", "right")}
        self.bricks = {self.model.body(name).id: color for name, color in names.items()}
        self.force = ContactFeedback(self.model, self.arms, self.bricks, force_feedback) if clutter else None
        self.stage = "idle"
        self.picked = {color: set() for color in QUOTAS}
        self.placed = {color: set() for color in QUOTAS}
        self.attempts = []
        self.render = mj.Renderer(self.model, SIZE, SIZE)
        self.camera_render = mj.Renderer(self.model, 512, 512) if self.record_cameras else None
        self.camera_font = (ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
                            if self.record_cameras else None)
        width, height = self.dashboard.size if self.dashboard else ((1024, 1024) if live else (SIZE, SIZE))
        self.video = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{width}x{height}", "-r", "20", "-i", "-", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output / "episode.mp4")], stdin=subprocess.PIPE)
        for arm in self.arms.values():
            self.data.qpos[arm.q] = arm.target
            self.data.qpos[arm.fq] = arm.finger
        mj.mj_forward(self.model, self.data)
        bin_id = self.model.body("sorting_bin").id
        self.bin_center = self.data.xpos[bin_id, :2].copy()
        self.bin_axes = self.data.xmat[bin_id].reshape(3, 3)[:2, :2].copy()
        self.frame = 0
        self.step(.4)

    def pose(self, arm, xyz, yaw, start=None):
        target_rotation = np.array([[math.cos(yaw), math.sin(yaw), 0],
                                    [math.sin(yaw), -math.cos(yaw), 0], [0, 0, -1]])
        self.ik.qpos[:] = self.data.qpos
        self.ik.qpos[arm.q] = arm.target if start is None else start
        jp = np.empty((3, self.model.nv))
        jr = np.empty_like(jp)
        for _ in range(100):
            mj.mj_kinematics(self.model, self.ik)
            mj.mj_comPos(self.model, self.ik)
            rotation = self.ik.site_xmat[arm.site].reshape(3, 3)
            ep = np.asarray(xyz) - self.ik.site_xpos[arm.site]
            er = .5 * np.cross(rotation.T, target_rotation.T).sum(axis=0)
            if np.linalg.norm(ep) < .0005 and np.linalg.norm(er) < .006 and np.trace(rotation.T @ target_rotation) > 2.9:
                return self.ik.qpos[arm.q].copy()
            mj.mj_jacSite(self.model, self.ik, jp, jr, arm.site)
            jac = np.concatenate((jp[:, arm.v], jr[:, arm.v] * .28), axis=0)
            delta = jac.T @ np.linalg.solve(jac @ jac.T + .00015 * np.eye(6), np.r_[ep, .28 * er])
            self.ik.qpos[arm.q] = np.clip(self.ik.qpos[arm.q] + np.clip(delta, -.14, .14), *arm.limits.T)
        raise ValueError(f"IK did not converge for {np.round(xyz, 3)}")

    def pixels(self, frame, cam):
        if self.record_cameras and frame is None:
            self.live_frame()
            return
        self.render.update_scene(self.data, camera=cam)
        rgb = self.render.render()
        if isinstance(frame, Path):
            Image.fromarray(rgb).save(frame)
        else:
            self.video.stdin.write(rgb.tobytes())

    def live_frame(self, repeat=1, phase=None):
        image = Image.new("RGB", (1024, 1024))
        draw = ImageDraw.Draw(image)
        views = (("overview", "Overview"), ("overhead", "Overhead"),
                 ("left_wrist", "Left wrist"), ("right_wrist", "Right wrist"))
        for index, (camera, label) in enumerate(views):
            x, y = index % 2 * 512, index // 2 * 512
            self.camera_render.update_scene(self.data, camera=camera)
            image.paste(Image.fromarray(self.camera_render.render()), (x, y))
            draw.rounded_rectangle((x + 9, y + 9, x + 132, y + 39), radius=6, fill="#162632")
            draw.text((x + 15, y + 11), label, fill="white", font=self.camera_font)
            if index >= 2 and self.force and self.force.enabled:
                side = "left" if index == 2 else "right"
                left, right = self.force.fingers[side]
                draw.rounded_rectangle((x + 9, y + 474, x + 246, y + 503), radius=6, fill="#162632")
                draw.text((x + 15, y + 477), f"L {left:.2f} N   R {right:.2f} N", fill="white", font=self.camera_font)
        reading = self.force.telemetry() if self.force and self.force.enabled else {}
        recorded = self.dashboard.compose(image, phase or self.stage, reading) if self.dashboard else image
        pixels = recorded.tobytes()
        for _ in range(repeat):
            self.video.stdin.write(pixels)
        if not self.live or phase is not None:
            return
        jpeg = BytesIO()
        image.save(jpeg, format="JPEG", quality=80)
        frame = {"frame": base64.b64encode(jpeg.getvalue()).decode("ascii")}
        if self.force and self.force.enabled:
            frame["force"] = reading
        print(json.dumps(frame), flush=True)
        now = time.monotonic()
        self.next_frame = max(self.next_frame + .05, now)
        time.sleep(max(0, self.next_frame - now))

    def phase(self, name, arm):
        self.stage = name
        if self.live:
            status = {"phase": name, "arm": arm}
            if self.force and self.force.enabled:
                status["force"] = self.force.telemetry()
            print(json.dumps(status), flush=True)

    def step(self, duration, arm=None, target=None, finger=None, until_supported=False):
        start = arm.target.copy() if arm else None
        opening = arm.finger if arm else None
        steps = round(duration / self.model.opt.timestep)
        for i in range(steps):
            if arm:
                t = (i + 1) / steps
                f = t * t * (3 - 2 * t)
                if target is not None:
                    arm.target = start + f * (target - start)
                if finger is not None and not (self.force and self.force.latched and self.stage == "grasp"):
                    arm.finger = opening + f * (finger - opening)
            for a in self.arms.values():
                torque = self.data.qfrc_bias[a.v] + np.array([300, 300, 250, 220, 110, 105, 75]) * (a.target - self.data.qpos[a.q])
                torque -= np.array([35, 35, 30, 25, 15, 14, 11]) * self.data.qvel[a.v]
                self.data.qfrc_applied[a.v] = np.clip(torque, -100, 100)
                force = self.data.qfrc_bias[a.fv] + 1550 * (a.finger - self.data.qpos[a.fq]) - 20 * self.data.qvel[a.fv]
                self.data.qfrc_applied[a.fv] = np.clip(force, -20, 20)
            mj.mj_step(self.model, self.data)
            self.frame += 1
            if self.force:
                self.force.sample(self.data, self.stage)
                self.force.regulate(self.data, self.stage)
            if self.frame % 25 == 0:
                self.pixels(None, "overview")
            if until_supported and self.force.support_steps >= SUPPORT_STEPS:
                arm.target = self.data.qpos[arm.q].copy()
                break

    def move(self, arm, point, yaw, duration=.8, until_supported=False):
        start = arm.target.copy()
        goal = self.pose(arm, point, yaw)
        try:
            self.step(duration, arm, goal, until_supported=until_supported)
            if not until_supported:
                self.step(.27)
        except ForceStop:
            self.retreat(arm, start)
            raise

    def retreat(self, arm, target):
        arm.target = self.data.qpos[arm.q].copy()
        self.phase("retreat", arm.side)
        self.step(.8, arm, target=target)

    def bin_contents(self):
        return {i for i in self.bricks
                if np.all(np.abs((self.data.xpos[i, :2] - self.bin_center) @ self.bin_axes) < BINSIZE - .003)
                and TABLE + .005 < self.data.xpos[i, 2] < TABLE + .065}

    def gripping_pads(self, arm, brick):
        return {pad for c in self.data.contact
                for pad, other in ((c.geom1, c.geom2), (c.geom2, c.geom1))
                if pad in arm.pads and self.model.geom_bodyid[other] == brick}

    def brick_bottom(self, brick):
        geom = self.model.geom(self.model.body(brick).name + "_base").id
        vertical = self.data.geom_xmat[geom].reshape(3, 3)[2]
        return self.data.geom_xpos[geom, 2] - np.abs(vertical) @ self.model.geom_size[geom]

    def check_drop(self, point, brick):
        if np.any(np.abs(point - TABLE_CENTER) > TABLE_HALF - .065):
            raise ValueError("Drop point is too close to the table edge")
        if np.all(np.abs((point - self.bin_center) @ self.bin_axes) < BINSIZE + .065):
            raise ValueError("Drop point is too close to the bin")
        if any(np.linalg.norm(self.data.xpos[i, :2] - point) < DROP_CLEARANCE for i in self.bricks if i != brick):
            raise ValueError("Drop point is too close to another brick")

    def place(self, arm, side, brick, yaw, occupied, destination=None):
        separate = destination is not None
        floor = TABLE if separate else BIN_FLOOR
        if separate:
            self.check_drop(destination, brick)
        else:
            local_slots = [np.array([x, y]) for y in (-.062, 0, .062) for x in (-.032, .032)]
            slots = (self.bin_center + self.bin_axes @ point for point in local_slots)
            destination = next((slot for slot in slots if all(
                np.any(np.abs((self.data.xpos[i, :2] - slot) @ self.bin_axes) > .054) for i in occupied)), None)
            if destination is None:
                raise ValueError("No clear placement slot")
        self.phase("separate" if separate else "carry", side)
        self.move(arm, (*destination, TABLE + .167), yaw, 1.65)
        if self.gripping_pads(arm, brick) != arm.pads:
            raise ValueError("Brick slipped during carry")

        # Use the brick's measured height to lower it close to the support before opening.
        tcp = self.data.site_xpos[arm.site].copy()
        horizontal_offset = self.data.xpos[brick, :2] - tcp[:2]
        release_xy = destination - horizontal_offset
        release_z = tcp[2] + floor + RELEASE_GAP - self.brick_bottom(brick)
        clearance_target = arm.target.copy()
        if separate:
            self.check_drop(destination, brick)
        if self.force:
            self.force.floor = self.model.geom("table" if separate else "bin_floor").id
        self.phase("lower_table" if separate else "lower", side)
        self.move(arm, (*release_xy, release_z), yaw, 1.05)
        if self.force and self.force.enabled:
            self.phase("support", side)
            if self.force.support_steps < SUPPORT_STEPS:
                # Creep a few millimeters through the nominal contact height, stopping on support force.
                self.move(arm, (*release_xy, release_z - .007), yaw, .95, until_supported=True)
            if self.force.support_steps < SUPPORT_STEPS:
                support = self.force.support
                self.retreat(arm, clearance_target)
                self.force.stop("no_support", "support", support)
            self.force.support_at_release = round(self.force.support, 3)
        gap = self.brick_bottom(brick) - floor
        inside = (np.linalg.norm(self.data.xpos[brick, :2] - destination) < .025 if separate else
                  np.all(np.abs((self.data.xpos[brick, :2] - self.bin_center) @ self.bin_axes) < BINSIZE - .015))
        if (self.gripping_pads(arm, brick) != arm.pads or not -.002 <= gap <= MAX_RELEASE_GAP
                or not inside):
            if self.force and self.force.enabled:
                self.force.support_at_release = None
                self.retreat(arm, clearance_target)
            raise ValueError(f"Placement not ready: hold or clearance check failed (gap {gap * 1000:.1f} mm)")
        self.phase("release", side)
        self.step(.38, arm, finger=.022)
        self.step(.22)
        self.move(arm, (*release_xy, TABLE + .170), yaw)
        return float(gap * 1000)

    def pick_geometry(self, u, v, yaw, expected_color):
        direction = np.r_[(np.array([v, u]) + .5 - SIZE / 2) * (2 * math.tan(math.radians(FOV / 2)) / SIZE), -1.]
        geom = np.empty(1, dtype=np.int32)
        distance = mj.mj_ray(self.model, self.data, CAMERA, direction,
                             np.array([0, 0, 1, 0, 0, 0], dtype=np.uint8), True, -1, geom)
        if distance < 0 or geom[0] < 0:
            raise ValueError("No visible brick at the selected pixel; use a fresh view")
        brick = int(self.model.geom_bodyid[geom[0]])
        if brick in self.bin_contents():
            raise ValueError("Selected brick is already in the bin")
        if expected_color not in ("any", self.bricks[brick]):
            raise ValueError("The selected pixel does not match the requested color; inspect or select again")
        rotation = self.data.xmat[brick].reshape(3, 3)
        if abs(rotation[2, 2]) < .9:
            raise ValueError("Brick is tilted or on edge; try a clearer target")
        axis = math.atan2(rotation[1, 0], rotation[0, 0])
        correction = (axis - yaw + math.pi / 2) % math.pi - math.pi / 2
        if self.pose_policy is None and abs(correction) > math.radians(32):
            raise ValueError("Grip angle does not match the visible brick; check its long axis")
        return brick, self.data.xpos[brick].copy(), yaw + correction

    def inspect(self, side, u, v, index):
        if side not in self.arms or not (0 <= u < SIZE and 0 <= v < SIZE):
            raise ValueError("Invalid inspection target")
        scale = 2 * (CAMERA[2] - TABLE) * math.tan(math.radians(FOV / 2)) / SIZE
        x, y = CAMERA[:2] + (np.array([v, u]) - SIZE / 2) * scale
        if not .25 < x < .53 or abs(y) > .24:
            raise ValueError("Inspection point is outside the source area")
        arm = self.arms[side]
        if arm.retained_brick is not None and self.gripping_pads(arm, arm.retained_brick):
            raise ValueError("Arm is still holding a brick")
        self.next_frame = time.monotonic()
        if self.force:
            self.force.begin(arm, None)
        self.phase("inspect", side)
        try:
            self.move(arm, (x, y, TABLE + .15), 0)
        except ForceStop as exc:
            return {"error": str(exc), "force": self.force.summary()}
        name = f"wrist_{index:02d}.jpg"
        self.pixels(self.output / name, side + "_wrist")
        self.phase("park", side)
        self.step(.85, arm, target=ARM_START)
        self.step(.25)
        return {"detail": name, "inspected": side}

    def transfer(self, side, u, v, angle, kind="pick", color="any", drop=(None, None), grasp_height_mm=None,
                 observation=None):
        if (side not in self.arms or not -180 <= angle <= 180 or not (0 <= u < SIZE and 0 <= v < SIZE)
                or kind not in ("pick", "separate")):
            raise ValueError("Invalid arm, pixel or angle")
        scale = 2 * (CAMERA[2] - TABLE - .024) * math.tan(math.radians(FOV / 2)) / SIZE
        x, y = CAMERA[:2] + (np.array([v, u]) - SIZE / 2) * scale
        audit = None
        if self.pose_policy is not None:
            if type(grasp_height_mm) is not int or not 3 <= grasp_height_mm <= 65:
                raise ValueError("Grasp height must be 3–65 mm above the table")
            predicted_z = TABLE + grasp_height_mm / 1000
            predicted_xy = overhead_point(u, v, predicted_z + .012)
            audit = {"observation": observation, "kind": kind, "color": color, "stage": "proposed",
                     "prediction_xyz_m": [*predicted_xy, predicted_z], "prediction_yaw_deg": angle}
            self.pose_records.append(audit)
            x, y = predicted_xy
        if not (.19 if self.clutter else .30) < x < .555 or not -.247 < y < .247:
            raise ValueError("Pixel falls outside the source area")
        arm = self.arms[side]
        if arm.retained_brick is not None and self.gripping_pads(arm, arm.retained_brick):
            raise ValueError("Arm is still holding a brick after an interrupted placement")
        arm.retained_brick = None
        yaw = math.radians(angle)
        before_bin = self.bin_contents()
        before_pos = {i: self.data.xpos[i].copy() for i in self.bricks}
        before_rotation = {i: self.data.xquat[i].copy() for i in self.bricks}
        selected = None
        destination = None
        grasp_z = TABLE + .0115
        if self.clutter:
            selected, point, corrected_yaw = self.pick_geometry(u, v, yaw, color)
            reference = point + [0., 0., .001]
            if audit is not None:
                audit.update(reference_xyz_m=reference.tolist(), reference_yaw_deg=math.degrees(corrected_yaw))
            if self.pose_policy == "predicted":
                x, y = predicted_xy
                grasp_z = predicted_z
            else:
                x, y, grasp_z = reference
                yaw = corrected_yaw
        if kind == "separate":
            if not self.clutter or not all(type(pixel) is int and 0 <= pixel < SIZE for pixel in drop):
                raise ValueError("Separate requires a tabletop drop pixel from the current overhead image")
            destination = overhead_point(*drop, TABLE)
            self.check_drop(destination, selected)
            if np.linalg.norm(destination - [x, y]) < DROP_CLEARANCE:
                raise ValueError("Drop point is too close to the brick's starting position")
        if self.clutter:
            # Check equivalent gripper orientations and the requested release pose before touching the brick.
            alternatives = [yaw, yaw + (math.pi if yaw < 0 else -math.pi)]
            feasible = []
            for candidate in alternatives:
                try:
                    joints = self.pose(arm, (x, y, TABLE + .135), candidate)
                    if destination is not None:
                        carry = self.pose(arm, (*destination, TABLE + .167), candidate, start=joints)
                        self.pose(arm, (*destination, TABLE + .012 + RELEASE_GAP), candidate, start=carry)
                    feasible.append((np.linalg.norm(joints - arm.target), candidate))
                except ValueError:
                    continue
            if not feasible:
                raise ValueError("No reachable wrist orientation for the requested action")
            yaw = min(feasible)[1]
        if audit is not None:
            audit.update(stage="approach", commanded_xyz_m=[x, y, grasp_z], commanded_yaw_deg=math.degrees(yaw))
        lifted = []
        error = None
        release_gap = None
        approach_joints = None
        self.next_frame = time.monotonic()
        if self.force:
            self.force.begin(arm, selected)
        self.phase("approach", side)
        try:
            self.step(.32, arm, finger=.037)
            self.move(arm, (x, y, TABLE + .135), yaw)
            approach_joints = arm.target.copy()
            if self.force and self.force.enabled:
                self.step(.25, arm, finger=.020)
                self.move(arm, (x, y, grasp_z + .036), yaw)
            self.move(arm, (x, y, grasp_z), yaw)
            if audit is not None:
                actual_rotation = self.data.site_xmat[arm.site].reshape(3, 3)
                audit.update(stage="grasp", achieved_xyz_m=self.data.site_xpos[arm.site].tolist(),
                             achieved_yaw_deg=math.degrees(math.atan2(actual_rotation[1, 0], actual_rotation[0, 0])))
            self.phase("grasp", side)
            self.step(.75 if self.force and self.force.enabled else .55, arm, finger=.003)
            self.step(.25)
            if self.force and self.force.enabled and self.force.grip_steps < 10:
                self.force.stop("weak_grip", "grasp", min(self.force.loads))
            self.phase("lift", side)
            self.move(arm, (x, y, TABLE + .145), yaw)
            lifted = [i for i in self.bricks if self.gripping_pads(arm, i) == arm.pads
                      and self.data.xpos[i, 2] - before_pos[i][2] > .055
                      and np.linalg.norm(self.data.xpos[i, :2] - [x, y]) < .065]
            if selected is not None and lifted and selected not in lifted:
                raise ValueError("Contact moved the intended brick; select again from the new view")
            for i in lifted:
                color = self.bricks[i]
                if color in QUOTAS:
                    self.picked[color].add(i)
            if len(lifted) == 1:
                release_gap = self.place(arm, side, lifted[0], yaw, before_bin, destination)
            elif lifted:
                raise ValueError("More than one brick is held")
        except ValueError as exc:
            error = str(exc)
        finally:
            if error:
                arm.retained_brick = next((i for i in lifted if self.gripping_pads(arm, i)), None)
                if (self.force and self.force.enabled and selected is not None and self.gripping_pads(arm, selected)
                        and self.data.xpos[selected, 2] - before_pos[selected][2] > .015):
                    arm.retained_brick = selected
            if arm.retained_brick is None:
                if self.force and self.force.events and approach_joints is not None:
                    if self.force.events[-1]["phase"] in ("approach", "grasp", "lift"):
                        self.force.latched = False
                        self.force.brick = None
                        self.phase("retreat", side)
                        self.step(.18, arm, finger=.037)
                        self.retreat(arm, approach_joints)
                self.phase("park", side)
                self.step(.85, arm, target=ARM_START, finger=.036)
                self.step(.4)
        after_bin = self.bin_contents()
        occupancy = Counter(self.bricks[i] for i in before_bin)
        for i in sorted(after_bin - before_bin):
            color = self.bricks[i]
            if release_gap is not None and i in lifted and color in QUOTAS and occupancy[color] < QUOTAS[color]:
                self.placed[color].add(i)
            occupancy[color] += 1
        result = {"lifted": bool(lifted), "placed": release_gap is not None and bool(set(lifted) & after_bin)}
        if self.clutter:
            shifted = sum(1 for i in self.bricks if i != selected and (
                np.linalg.norm(self.data.xpos[i] - before_pos[i]) > .008
                or abs(self.data.xquat[i] @ before_rotation[i]) < math.cos(math.radians(5))))
            drop_error = (float(np.linalg.norm(self.data.xpos[lifted[0], :2] - destination))
                          if destination is not None and release_gap is not None else None)
            result.update(separated=drop_error is not None and drop_error < .025,
                          shifted_neighbors=shifted, lifted_color=self.bricks[lifted[0]] if len(lifted) == 1 else None)
            if destination is not None:
                result.update(drop_pixel=list(drop), drop_error_mm=None if drop_error is None else round(drop_error * 1000, 1))
        if error:
            result["error"] = error
        if self.force:
            result["force"] = self.force.summary()
            if arm.retained_brick is not None:
                result["holding"] = self.bricks[arm.retained_brick]
        self.attempts.append({"arm": side, "pixel": [u, v], "angle": angle, "kind": kind,
                              "lifted": [self.bricks[i] for i in lifted],
                              "entered_bin": [self.bricks[i] for i in sorted(after_bin - before_bin)],
                              "release_gap_mm": release_gap,
                              "separated": result.get("separated", False),
                              "shifted_neighbors": result.get("shifted_neighbors", 0),
                              "error": error})
        if destination is not None:
            self.attempts[-1].update(drop_pixel=result["drop_pixel"], drop_error_mm=result["drop_error_mm"])
        if self.force:
            self.attempts[-1]["force"] = result["force"]
        if audit is not None:
            audit.update(lifted=selected in lifted, placed=result["placed"], error=error)
        return result

    def observe(self, index, result=None):
        name = f"observation_{index:02d}.jpg"
        self.render.update_scene(self.data, camera="overhead")
        image = Image.fromarray(self.render.render())
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
        for coordinate in range(64, SIZE, 64):
            for start in range(0, SIZE, 16):
                draw.line((coordinate, start, coordinate, start + 4), fill="#647890")
                draw.line((start, coordinate, start + 4, coordinate), fill="#647890")
            draw.text((coordinate, 5), str(coordinate), font=font, fill="white", stroke_width=2, stroke_fill="#193040")
            draw.text((5, coordinate), str(coordinate), font=font, fill="white", stroke_width=2, stroke_fill="#193040")
        image.save(self.output / name, quality=94)
        if self.dashboard:
            self.dashboard.record_observation(index, self.output / name, self.score(), result or {})
            self.live_frame(repeat=5, phase="planning")
        return {"observation": name}

    def score(self):
        actual = Counter(self.bricks[i] for i in self.bin_contents())
        checkpoints = {color: {"picked": min(len(self.picked[color]), quota),
                               "placed": min(len(self.placed[color]), quota)}
                       for color, quota in QUOTAS.items()}
        return {"score": sum(sum(v.values()) for v in checkpoints.values()), "max_score": 10,
                "success": dict(actual) == QUOTAS, "counts_in_bin": dict(actual),
                "checkpoints": checkpoints}

    def finish(self):
        self.pixels(self.output / "final_overview.png", "overview")
        self.pixels(self.output / "final_overhead.png", "overhead")
        if self.dashboard:
            self.live_frame(repeat=60, phase="finished")
        self.video.stdin.close()
        if self.video.wait() != 0:
            raise RuntimeError("Could not encode episode video")
        result = {**self.score(), "actions": self.attempts, "scene": "clutter" if self.clutter else "grid",
                  "force_feedback": bool(self.force and self.force.enabled),
                  "bin_xy_m": self.bin_center.tolist(),
                  "bin_yaw_deg": math.degrees(math.atan2(self.bin_axes[1, 0], self.bin_axes[0, 0]))}
        if self.pose_policy is not None:
            result.update(pose_policy=self.pose_policy, pose_records=self.pose_records)
        (self.output / "score.json").write_text(json.dumps(result, indent=2))
        if self.camera_render:
            self.camera_render.close()
        self.render.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--clutter", action="store_true")
    parser.add_argument("--force-feedback", action="store_true")
    parser.add_argument("--pose-policy", choices=("corrected", "predicted"))
    parser.add_argument("--record-dashboard", action="store_true")
    args = parser.parse_args()
    if args.force_feedback and not args.clutter:
        parser.error("--force-feedback requires --clutter")
    if args.pose_policy and not args.force_feedback:
        parser.error("--pose-policy requires --force-feedback")
    if args.record_dashboard and args.pose_policy != "predicted":
        parser.error("--record-dashboard requires --pose-policy predicted")
    lab = Lab(args.seed, args.output, args.live, args.clutter, args.force_feedback, args.pose_policy,
              args.record_dashboard)
    print(json.dumps(lab.observe(0)), flush=True)
    try:
        for index, line in enumerate(sys.stdin, 1):
            command = json.loads(line)
            if command.get("done"):
                if lab.dashboard:
                    lab.dashboard.record_decision(command, index)
                break
            try:
                if command.get("observation", index - 1) != index - 1:
                    raise ValueError("Observation is stale; retry using the latest image")
                if lab.dashboard:
                    lab.dashboard.record_decision(command, index)
                if command.get("kind") == "inspect":
                    result = lab.inspect(command["arm"], command["u"], command["v"], index)
                else:
                    result = lab.transfer(command["arm"], command["u"], command["v"], command["angle"],
                                          command.get("kind", "pick"), command.get("color", "any"),
                                          (command.get("drop_u"), command.get("drop_v")), command.get("grasp_height_mm"),
                                          index - 1)
            except (ValueError, KeyError, TypeError) as error:
                result = {"error": str(error)}
            if args.live or args.pose_policy:
                result["progress"] = lab.score()
            print(json.dumps({**lab.observe(index, result), **result}), flush=True)
    finally:
        lab.finish()


if __name__ == "__main__":
    main()
