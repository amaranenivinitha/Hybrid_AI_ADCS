import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.image as mpimg
import os
from skimage.transform import resize
import sys, time as t
from matplotlib.animation import FFMpegWriter, PillowWriter

# =====================================
# Load True Simulation Results
# =====================================
csv_path = "results/simulation_quat_log.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"{csv_path} not found — run src.sim first.")

df = pd.read_csv(csv_path)
time = df["time"].values
q = df[["q0", "q1", "q2", "q3"]].values
pd_error = df["pd_error_deg"].values if "pd_error_deg" in df.columns else np.zeros(len(time))
ai_error = df["ai_error_deg"].values if "ai_error_deg" in df.columns else np.zeros(len(time))
tau_pd = df["tau_pd"].values if "tau_pd" in df.columns else np.zeros(len(time))
tau_ai = df["tau_ai"].values if "tau_ai" in df.columns else np.zeros(len(time))

# =====================================
# Quaternion → Rotation Matrix
# =====================================
def quat_to_rot_matrix(q):
    q0, q1, q2, q3 = q
    return np.array([
        [1-2*(q2**2+q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3), 1-2*(q1**2+q3**2), 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1-2*(q1**2+q2**2)]
    ])

# =====================================
# Cube Geometry (Satellite Body)
# =====================================
cube = np.array([
    [-1, -1, -1],
    [1, -1, -1],
    [1, 1, -1],
    [-1, 1, -1],
    [-1, -1, 1],
    [1, -1, 1],
    [1, 1, 1],
    [-1, 1, 1],
]) * 0.4

faces = [
    [cube[0], cube[1], cube[2], cube[3]],
    [cube[4], cube[5], cube[6], cube[7]],
    [cube[0], cube[1], cube[5], cube[4]],
    [cube[2], cube[3], cube[7], cube[6]],
    [cube[1], cube[2], cube[6], cube[5]],
    [cube[4], cube[7], cube[3], cube[0]]
]

# =====================================
# Visualization Setup
# =====================================
fig = plt.figure(figsize=(10, 10))
ax3d = fig.add_subplot(311, projection='3d')
ax_err = fig.add_subplot(312)
ax_torque = fig.add_subplot(313)

ax3d.set_xlim(-2, 2)
ax3d.set_ylim(-2, 2)
ax3d.set_zlim(-2, 2)
ax3d.set_title("AI-Powered Hybrid ADCS — Real Satellite Attitude Simulation")
ax3d.set_axis_off()

# --- Earth Texture Mapping ---
earth_texture_path = "assets/earth_texture.jpg"
if not os.path.exists(earth_texture_path):
    raise FileNotFoundError(f"Missing Earth texture at {earth_texture_path}")

earth_img = mpimg.imread(earth_texture_path).astype(float)
if earth_img.max() > 1.0:
    earth_img /= 255.0

# Create sphere mesh
u_steps, v_steps = 180, 90
u = np.linspace(0, 2 * np.pi, u_steps)
v = np.linspace(0, np.pi, v_steps)
u, v = np.meshgrid(u, v)
x = np.cos(u) * np.sin(v)
y = np.sin(u) * np.sin(v)
z = np.cos(v)

# Resize and trim texture safely
resized_tex = resize(earth_img, (v_steps, u_steps, 3), anti_aliasing=True)
resized_tex = np.clip(resized_tex, 0, 1)

vh, uh, _ = resized_tex.shape
if vh != v_steps:
    resized_tex = resized_tex[:v_steps, :, :]
if uh != u_steps:
    resized_tex = resized_tex[:, :u_steps, :]

earth_surface = ax3d.plot_surface(
    x, y, z,
    rstride=2, cstride=2,
    facecolors=resized_tex,
    linewidth=0, antialiased=False
)

# --- Target Direction ---
target_dir = np.array([1, 0, 0])
ax3d.quiver(0, 0, 0, *target_dir, color='limegreen', length=1.2, linewidth=3, label='Target +X')
ax3d.legend(loc='upper right')

# --- Satellite Cube ---
body = Poly3DCollection(faces, alpha=0.85, edgecolors='k', facecolor='lightgray')
ax3d.add_collection3d(body)

# =====================================
# Error & Torque Plots
# =====================================
ax_err.set_xlim(time[0], time[-1])
ax_err.set_ylim(0, max(np.max(pd_error), np.max(ai_error)) * 1.2)
ax_err.set_ylabel('Pointing Error (°)')
line_pd, = ax_err.plot([], [], 'r-', lw=2, label='PD Controller')
line_ai, = ax_err.plot([], [], 'b-', lw=2, label='AI Controller')
text_box = ax_err.text(0.02, 0.9, "", transform=ax_err.transAxes, color='green', fontsize=12)
ax_err.legend(loc='upper right')

ax_torque.set_xlim(time[0], time[-1])
ax_torque.set_ylim(0, max(np.max(tau_pd), np.max(tau_ai)) * 1.2)
ax_torque.set_ylabel('Torque (Nm)')
ax_torque.set_xlabel('Time (s)')
tline_pd, = ax_torque.plot([], [], 'r-', lw=2, label='PD Torque')
tline_ai, = ax_torque.plot([], [], 'b-', lw=2, label='AI Torque')
ax_torque.legend(loc='upper right')

# =====================================
# Update Animation
# =====================================
def update(i):
    Ri = quat_to_rot_matrix(q[i])
    rotated_faces = [(np.dot(face, Ri.T)) for face in faces]
    body.set_verts(rotated_faces)

    # Rotate camera and Earth slowly for cinematic motion
    ax3d.view_init(elev=25, azim=i * 0.6)
    earth_surface.remove()
    earth_surface_new = ax3d.plot_surface(
        x, y, z,
        rstride=2, cstride=2,
        facecolors=np.roll(resized_tex, i // 2, axis=1),
        linewidth=0, antialiased=False
    )

    line_pd.set_data(time[:i], pd_error[:i])
    line_ai.set_data(time[:i], ai_error[:i])
    tline_pd.set_data(time[:i], tau_pd[:i])
    tline_ai.set_data(time[:i], tau_ai[:i])

    if i > 30:
        rms_pd = np.sqrt(np.mean(pd_error[:i]**2))
        rms_ai = np.sqrt(np.mean(ai_error[:i]**2))
        imp = 100 * (1 - rms_ai / rms_pd)
        text_box.set_text(f"AI Improvement: +{imp:.1f}%")

    return [body, line_pd, line_ai, tline_pd, tline_ai, earth_surface_new]

# =====================================
# FAST ANIMATION SETUP (Progress + MP4)
# =====================================
frame_skip = 10
frames = range(0, len(time), frame_skip)
ani = FuncAnimation(fig, update, frames=frames, interval=40, blit=False)

os.makedirs("results", exist_ok=True)
out_path_mp4 = "results/final_hybrid_adcs_earth.mp4"
out_path_gif = "results/final_hybrid_adcs_earth.gif"

print("\n🚀 Rendering started — estimated time: 1–3 minutes")
print("---------------------------------------------------")

class ProgressWriter:
    def __init__(self, writer, total):
        self.writer = writer
        self.total = total
        self.current = 0
        self.last_update = t.time()

    def grab_frame(self, **kwargs):
        self.writer.grab_frame(**kwargs)
        self.current += 1
        now = t.time()
        if now - self.last_update > 0.5 or self.current == self.total:
            percent = 100 * self.current / self.total
            sys.stdout.write(f"\r⏳ Rendering frame {self.current}/{self.total} ({percent:.1f}%)")
            sys.stdout.flush()
            self.last_update = now

    def finish(self):
        print("\n✅ Rendering complete.\n")
        self.writer.finish()

try:
    # ✅ MP4 mode (fast, high quality)
    print("\n🎞️ Saving as MP4 (fast mode)...")
    ffmpeg_writer = FFMpegWriter(fps=25, codec='libx264', bitrate=1800)
    ani.save(out_path_mp4, writer=ffmpeg_writer)
    print(f"\n🎥 Saved fast MP4 demo to: {out_path_mp4}\n")

except Exception as e:
    print(f"\n⚠️ MP4 failed ({e}) — switching to GIF.\n")
    print("🎞️ Saving as GIF (fallback mode)...")
    pillow_writer = PillowWriter(fps=15)
    ani.save(out_path_gif, writer=pillow_writer)
    print(f"✅ Saved GIF fallback to: {out_path_gif}\n")

plt.show()
