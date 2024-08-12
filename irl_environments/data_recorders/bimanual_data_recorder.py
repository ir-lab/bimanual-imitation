import subprocess
from abc import abstractmethod
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from irl_environments.data_recorders.base_data_recorder import DataRecorder

def export_video(frames, video_path, compress=False):
        import cv2
        
        # import imageio
        # with imageio.get_writer(gif_filename, mode="I") as writer:
        #     for img in imgs:
        #         writer.append_data(img)

        video_path = Path(video_path)
        height, width, layers = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(str(video_path), fourcc, 30, (width, height))
        for frame in frames:
            video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        video.release()

        if compress:
            out_video_path = video_path.parent / f"out_{video_path.name}"

            # You will need use a ffmpeg built/linked with h264
            ffmpeg_bin = Path.home() / "local/ffmpeg/bin/ffmpeg"
            if not ffmpeg_bin.exists():
                ffmpeg_bin = "ffmpeg"

            command = [
                str(ffmpeg_bin),
                "-i",
                str(video_path),
                "-vcodec",
                "libx264",
                "-crf",
                "23",
                "-acodec",
                "aac",
                "-b:a",
                "192k",
                str(out_video_path),
            ]

            subprocess.run(command)

            video_path.unlink()
            out_video_path.rename(video_path)

        print(f"Video saved at {video_path}!")

class BimanualDataRecorder(DataRecorder):
    def __init__(self):
        DataRecorder.__init__(self)

    def record_path_states(self, action, constrained_action, observation, reward, time):
        self._action_hist.append(constrained_action)
        self._observation_hist.append(observation)
        self._reward_hist.append(reward)
        self._env_time = time

    @abstractmethod
    def get_mujoco_renders(self):
        raise NotImplementedError

    def export_gif_recording(self):
        if self.render_mode != "rgb_array":
            print("Skipping GIF recording... render_mode must be rgb_array to record GIFs!")
            return

        if (
            self._gif_export_dir is None
            and self._gif_export_prefix is None
            and self._gif_export_suffix is None
        ):
            raise ValueError("GIF export parameters not set")

        frames = self.get_mujoco_renders()
        
        video_path = (
            Path(self._gif_export_dir) / f"{self._gif_export_prefix}_{self._gif_export_suffix}.mp4"
        )
        
        export_video(frames, video_path)

        self._record_gif = False

    def plot_trajs(self):
        actions = np.vstack(self._action_hist)
        n_cols = 6
        n_a_rows = actions.shape[1] // n_cols + 1
        fig1, axs1 = plt.subplots(n_a_rows, n_cols, layout="tight", figsize=(12, 8))
        for a_idx in range(actions.shape[1]):
            row_idx = a_idx // n_cols
            col_idx = a_idx % n_cols
            if n_a_rows == 1:
                axs1[col_idx].plot(actions[:, a_idx])
            else:
                axs1[row_idx, col_idx].plot(actions[:, a_idx])

        observations = np.vstack(self._observation_hist)
        n_o_rows = observations.shape[1] // n_cols + 1
        fig2, axs2 = plt.subplots(n_o_rows, n_cols, layout="tight", figsize=(12, 8))
        for obs_idx in range(observations.shape[1]):
            row_idx = obs_idx // n_cols
            col_idx = obs_idx % n_cols
            if n_o_rows == 1:
                axs2[col_idx].plot(observations[:, obs_idx])
            else:
                axs2[row_idx, col_idx].plot(observations[:, obs_idx])
        plt.show()
