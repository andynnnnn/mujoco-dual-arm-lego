"""Compose the B-group recording with cameras and time-aligned experiment data."""

from PIL import Image, ImageDraw, ImageFont


PHASE_NAMES = {
    "idle": "布置场景", "planning": "观测已更新", "approach": "接近积木", "grasp": "闭合夹爪",
    "lift": "提起积木", "carry": "移向收纳箱", "separate": "移向桌面空位", "lower": "下降到箱底",
    "lower_table": "下降到桌面", "support": "确认承托", "release": "松开夹爪",
    "retreat": "受力停退", "park": "退回待机位", "inspect": "腕部近看", "finished": "实验已结束",
}
STOP_NAMES = {
    "neighbor_contact": "碰到邻砖", "obstacle_contact": "碰到障碍物", "grip_loss": "一侧指垫失力",
    "weak_grip": "双指夹持未稳定", "no_support": "未确认承托",
}
COLOR_NAMES = {"red": "红砖", "orange": "橙砖", "blue": "蓝砖", "green": "绿砖"}
INK, MUTED, ACCENT = "#edf4f8", "#adbdc8", "#94e8d3"


class RecordingDashboard:
    size = (1600, 1024)

    def __init__(self):
        fonts = "/usr/share/fonts/opentype/noto/NotoSansCJK-{}.ttc"
        self.fonts = {size: ImageFont.truetype(fonts.format(weight), size, index=2)
                      for size, weight in ((18, "Regular"), (20, "Regular"), (23, "Bold"), (52, "Bold"))}
        self.observation = None
        self.observation_index = 0
        self.score = {"score": 0, "counts_in_bin": {}, "success": False}
        self.action = None
        self.action_index = 0
        self.target = None
        self.target_observation = None
        self.model_calls = 0
        self.model_seconds = 0.
        self.last_force = {}
        self.last_stop = None

    def record_decision(self, command, index):
        timing = command.get("recording", {})
        self.model_calls = timing.get("model_calls", self.model_calls)
        self.model_seconds = timing.get("model_seconds", self.model_seconds)
        if command.get("done"):
            return
        self.action_index = index
        self.action = command
        if command.get("kind") in ("pick", "separate"):
            self.target = command
            self.target_observation = index - 1

    def record_observation(self, index, path, score, result):
        with Image.open(path) as image:
            self.observation = image.resize((368, 368), Image.Resampling.LANCZOS)
        self.observation_index = index
        self.score = score
        self.last_force = result.get("force") or {}
        if self.last_force.get("events"):
            self.last_stop = self.last_force["events"][-1]

    def compose(self, cameras, phase, force):
        frame = Image.new("RGB", self.size, "#101821")
        frame.paste(cameras, (0, 0))
        draw = ImageDraw.Draw(frame)
        for top, bottom in ((16, 226), (239, 487), (500, 1008)):
            draw.rounded_rectangle((1040, top, 1584, bottom), radius=15, fill="#1a2732", outline="#3b4d5b")

        def label(y, text, size=20, color=INK, x=1060):
            draw.text((x, y), text, font=self.fonts[size], fill=color, anchor="lt")

        counts = self.score["counts_in_bin"]
        others = sum(count for color, count in counts.items() if color not in ("red", "orange"))
        finished = phase == "finished"
        label(34, "双臂数乐高 · B 组", 23)
        label(77, f"{self.score['score']} / 10", 52)
        action = self.action or {}
        arm = {"left": "左臂", "right": "右臂"}.get(action.get("arm"), "")
        stage = PHASE_NAMES.get(phase, phase)
        label(82, "任务成功" if finished and self.score["success"] else "最终得分" if finished else stage,
              23, ACCENT, 1290)
        label(117, f"动作 {self.action_index} · {arm}" if arm and not finished else f"动作 {self.action_index}", 18, MUTED, 1290)
        label(153, f"红 {counts.get('red', 0)}/2", 20, "#ff8b99")
        label(153, f"橙 {counts.get('orange', 0)}/3", 20, "#ffab6e", 1210)
        label(153, f"其他 {others}", 20, MUTED, 1360)
        label(190, f"模型规划 {self.model_calls} 次 · 累计 {self.model_seconds:.1f} 秒", 20, MUTED)

        label(256, "指垫法向力", 23)
        for x, text in ((1060, "机械臂"), (1255, "左指（N）"), (1417, "右指（N）")):
            label(296, text, 18, MUTED, x)
        for y, side, name in ((327, "left", "左臂"), (358, "right", "右臂")):
            draw.line((1060, y - 7, 1564, y - 7), fill="#354653")
            left, right = force.get("finger_n", {}).get(side, (0., 0.))
            label(y, name)
            label(y, f"{left:.2f}", x=1255)
            label(y, f"{right:.2f}", x=1417)
        label(396, f"承托 {force.get('support_n', 0):.2f} N · 异常接触 {force.get('unexpected_n', 0):.2f} N", 18, MUTED)
        peaks = self.last_force.get("peak_finger_n")
        support = self.last_force.get("support_before_release_n")
        if peaks:
            detail = f"上一步峰值 {peaks[0]:.2f} / {peaks[1]:.2f} N"
            if support is not None:
                detail += f" · 松爪前承托 {support:.2f} N"
            label(424, detail, 18, MUTED)
        stop = force.get("event") or self.last_stop
        if stop:
            label(452, f"停退 · {STOP_NAMES.get(stop['code'], stop['code'])} · {stop['force_n']:.2f} N", 18, "#ffc18b")

        label(519, f"模型观测 · {self.observation_index:02d}", 23)
        if self.target:
            target = self.target
            arm = "左臂" if target["arm"] == "left" else "右臂"
            verb = "移开" if target["kind"] == "separate" else "抓取"
            prefix = "模型目标" if self.target_observation == self.observation_index else "最近模型目标"
            label(557, f"{prefix} · {arm}{verb}{COLOR_NAMES.get(target['color'], '')}", 20, ACCENT)
            label(585, f"点 ({target['u']}, {target['v']}) · 高 {target['grasp_height_mm']} mm · 朝向 {target['angle']}°", 18)
            if target["kind"] == "separate":
                label(610, f"桌面落点 ({target['drop_u']}, {target['drop_v']})", 18, MUTED)
        observation = self.observation or cameras.crop((512, 0, 1024, 512)).resize((368, 368))
        left, top = 1128, 632
        frame.paste(observation, (left, top))
        if self.target and self.target_observation == self.observation_index:
            x = left + (self.target["u"] + .5) * 368 / 768
            y = top + (self.target["v"] + .5) * 368 / 768
            draw.ellipse((x - 8, y - 8, x + 8, y + 8), outline="#101821", width=4)
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline=ACCENT, width=2)
        return frame
