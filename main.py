import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from tkinter import ttk
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import imageio
import numpy as np
import ale_py
import time
import queue

gym.register_envs(ale_py)


class AITrainerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Trainer")

        self.env_name_var = tk.StringVar(value="CartPole-v1")
        self.timesteps_var = tk.StringVar(value="10000")
        self.gif_frames_var = tk.StringVar(value="200")  # Nouveau champ
        self.model = None
        self.env = None
        self.cancel_training = False
        self.training_start_time = None
        self.progress_queue = queue.Queue()

        self.create_widgets()
        self.poll_progress_queue()

    def create_widgets(self):
        frame = tk.Frame(self.root)
        frame.pack(padx=10, pady=10)

        tk.Label(frame, text="Env Name:").grid(row=0, column=0, sticky="w")
        tk.Entry(frame, textvariable=self.env_name_var, width=20).grid(row=0, column=1, sticky="w")

        tk.Label(frame, text="Training Frames:").grid(row=1, column=0, sticky="w")
        tk.Entry(frame, textvariable=self.timesteps_var, width=20).grid(row=1, column=1, sticky="w")

        tk.Label(frame, text="GIF Frames (-1 = until done):").grid(row=2, column=0, sticky="w")
        tk.Entry(frame, textvariable=self.gif_frames_var, width=20).grid(row=2, column=1, sticky="w")

        tk.Button(frame, text="Load Environment", command=self.load_env).grid(row=0, column=2, padx=5)
        tk.Button(frame, text="Load Model", command=self.load_model).grid(row=1, column=2, padx=5)
        tk.Button(frame, text="Save Model", command=self.save_model).grid(row=2, column=2, padx=5)
        tk.Button(frame, text="Train", command=self.start_training_thread).grid(row=3, column=2, padx=5)
        tk.Button(frame, text="Cancel", command=self.cancel_training_now).grid(row=4, column=2, padx=5)
        tk.Button(frame, text="Make GIF", command=self.make_gif).grid(row=5, column=2, padx=5)

        self.progress = ttk.Progressbar(self.root, orient='horizontal', length=500, mode='determinate')
        self.progress.pack(padx=10, pady=(5, 0))

        self.status_label = tk.Label(self.root, text="Progress: 0% | ETA: ∞s")
        self.status_label.pack()

        tk.Label(self.root, text="Logs:").pack(anchor="w", padx=10)
        self.log_text = scrolledtext.ScrolledText(self.root, height=15, width=70, state='disabled')
        self.log_text.pack(padx=10, pady=5)

    def log(self, msg):
        self.log_text.configure(state='normal')
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state='disabled')

    def load_env(self):
        env_name = self.env_name_var.get()
        try:
            self.env = DummyVecEnv([lambda: gym.make(env_name, render_mode='rgb_array')])
            self.log(f"✅ Environment '{env_name}' loaded.")
        except Exception as e:
            self.log(f"❌ Environment load error: {e}")

    def load_model(self):
        path = filedialog.askopenfilename(title="Load Model", filetypes=[("SB3 Models", "*.zip"), ("All files", "*.*")])
        if path:
            try:
                self.model = PPO.load(path, env=self.env)
                self.log(f"📂 Model '{path}' loaded.")
            except Exception as e:
                self.log(f"❌ Model load error: {e}")

    def save_model(self):
        if self.model is None:
            messagebox.showwarning("Warning", "No model loaded to save.")
            return
        path = filedialog.asksaveasfilename(title="Save Model", defaultextension=".zip",
                                            filetypes=[("SB3 Models", "*.zip"), ("All files", "*.*")])
        if path:
            try:
                self.model.save(path)
                self.log(f"💾 Model saved as '{path}'.")
            except Exception as e:
                self.log(f"❌ Model save error: {e}")

    def update_progress_ui(self, current, total, eta):
        percent = (current / total) * 100
        self.progress['value'] = current
        self.progress['maximum'] = total
        self.status_label.config(text=f"Progress: {percent:.1f}% | ETA: {eta:.1f}s")

    def poll_progress_queue(self):
        try:
            while True:
                current, total, eta, reward, loss = self.progress_queue.get_nowait()
                self.update_progress_ui(current, total, eta)
                if current % 1000 == 0 or current == total:
                    self.log(f"📈 Step: {current} | Avg Reward: {reward:.3f} | Policy Loss: {loss:.5f}")
        except queue.Empty:
            pass
        self.root.after(100, self.poll_progress_queue)

    def cancel_training_now(self):
        self.cancel_training = True
        self.log("⛔ Training cancellation requested...")

    def train(self):
        if self.env is None:
            self.log("❌ Environment not loaded.")
            return
        try:
            timesteps = int(self.timesteps_var.get())
        except ValueError:
            self.log("❌ Invalid number for training frames.")
            return

        try:
            self.log("🏋️‍♂️ Training started...")
            self.progress['value'] = 0
            self.status_label.config(text="Progress: 0% | ETA: ∞s")
            self.training_start_time = time.time()
            self.cancel_training = False

            self.model = PPO("MlpPolicy", self.env, verbose=0)

            callback = ProgressCallback(
                app=self,
                total_timesteps=timesteps,
                start_time=self.training_start_time,
                progress_queue=self.progress_queue
            )
            self.model.learn(total_timesteps=timesteps, callback=callback)

            if self.cancel_training:
                self.log("🛑 Training was cancelled.")
            else:
                self.progress['value'] = timesteps
                self.status_label.config(text="Progress: 100% | ETA: 0.0s")
                self.log("✅ Training finished.")
        except Exception as e:
            self.log(f"❌ Training error: {e}")

    def start_training_thread(self):
        train_thread = threading.Thread(target=self.train, daemon=True)
        train_thread.start()

    def make_gif(self):
        if self.model is None or self.env is None:
            self.log("❌ Model or environment not loaded.")
            return
        try:
            frame_limit = int(self.gif_frames_var.get())
            self.log(f"🎞️ Generating GIF (frames={frame_limit})...")
            frames = []

            reset_result = self.env.reset()
            obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result

            frame_count = 0
            done = False

            while (frame_limit == -1 and not done) or (frame_limit != -1 and frame_count < frame_limit):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, dones, _ = self.env.step(action)

                frame = self.env.get_images()[0]
                if isinstance(frame, np.ndarray):
                    frames.append(frame)
                else:
                    self.log("⚠️ Empty frame, possible render_mode issue.")

                frame_count += 1
                done = dones[0]
                if done:
                    break

            env_name = self.env_name_var.get()
            timesteps = self.timesteps_var.get()
            gif_path = f"ppo_{env_name}_{timesteps}.gif"
            gif_path = gif_path.replace("/", "")
            imageio.mimsave(gif_path, frames, fps=30)
            self.log(f"🎞️ GIF saved as: {gif_path}")
        except Exception as e:
            self.log(f"❌ GIF error: {e}")


class ProgressCallback(BaseCallback):
    def __init__(self, app, total_timesteps, start_time, progress_queue, verbose=0):
        super().__init__(verbose)
        self.app = app
        self.total_timesteps = total_timesteps
        self.start_time = start_time
        self.progress_queue = progress_queue

    def _on_step(self) -> bool:
        current = self.num_timesteps
        elapsed = time.time() - self.start_time
        eta = (elapsed / current) * (self.total_timesteps - current) if current > 0 else float('inf')
        reward = float(np.mean(self.locals['rewards']))
        loss = float(self.model.logger.name_to_value.get("train/policy_loss", 0.0))
        self.progress_queue.put((current, self.total_timesteps, eta, reward, loss))
        return not self.app.cancel_training


if __name__ == "__main__":
    root = tk.Tk()
    app = AITrainerApp(root)
    root.mainloop()

