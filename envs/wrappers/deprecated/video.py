# TODO: consider deleting — deprecated, unimported, excluded from wheel.
# import os
# import imageio
# from typing import Optional, Dict, Any, Callable
# from pettingzoo import AECEnv
# from pettingzoo.utils import BaseWrapper
# import gymnasium as gym


# class EpisodeTrigger:
#     def __init__(self, period: int):
#         self.period = period

#     def __call__(self, episode_id: int) -> bool:
#         return episode_id % self.period == 0


# TODO: this is from the old library
# class RecordVideo(BaseWrapper):
#     def __init__(
#         self,
#         env: AECEnv,
#         video_folder: str = "videos",
#         episode_trigger: Optional[Callable[[int], bool]] = None,
#         video_length: int = 0,
#         name_prefix: str = "episode",
#         fps: int = 30,
#     ):
#         super().__init__(env)
#         self.video_folder = video_folder
#         self.episode_trigger = episode_trigger or EpisodeTrigger(1000)
#         self.video_length, self.name_prefix, self.fps = video_length, name_prefix, fps
#         os.makedirs(self.video_folder, exist_ok=True)
#         (
#             self.recording,
#             self.video_writer,
#             self.frames_recorded,
#             self.episode_id,
#             self.episode_started,
#         ) = (False, None, 0, 0, False)
#         if not hasattr(env, "render"):
#             raise ValueError("Environment must support rendering to record video")

#     def reset(
#         self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
#     ):
#         self.env.reset(seed=seed, options=options)
#         if self.episode_trigger(self.episode_id):
#             self._start_recording()
#         elif self.recording:
#             self._stop_recording()
#         if self.recording:
#             self._capture_frame()
#         self.episode_started = True

#     def step(self, action):
#         self.env.step(action)
#         if self.recording:
#             self._capture_frame()
#         done = (
#             not self.env.agents
#             or all(self.env.terminations.values())
#             or all(self.env.truncations.values())
#         )
#         if self.recording and (
#             done
#             or (self.video_length > 0 and self.frames_recorded >= self.video_length)
#         ):
#             self._stop_recording()
#         if done and self.episode_started:
#             self.episode_id += 1
#             self.episode_started = False

#     def _start_recording(self):
#         if self.recording:
#             self._stop_recording()
#         video_path = os.path.join(
#             self.video_folder, f"{self.name_prefix}_{self.episode_id:06d}.mp4"
#         )
#         try:
#             self.video_writer = imageio.get_writer(
#                 video_path, fps=self.fps, macro_block_size=1
#             )
#             self.recording, self.frames_recorded = True, 0
#         except Exception as e:
#             print(f"Warning: Could not open video writer: {e}")

#     def _stop_recording(self):
#         if self.video_writer:
#             self.video_writer.close()
#             self.video_writer = None
#         self.recording, self.frames_recorded = False, 0

#     def _get_frame(self):
#         try:
#             frame = self.env.render()
#             if frame is not None and frame.ndim == 3 and frame.shape[2] == 4:
#                 frame = frame[:, :, :3]
#             return frame
#         except Exception:
#             return None

#     def _capture_frame(self):
#         if self.recording and self.video_writer:
#             frame = self._get_frame()
#             if frame is not None:
#                 self.video_writer.append_data(frame)
#                 self.frames_recorded += 1

#     def last(self, observe: bool = True):
#         _, reward, term, trunc, info = self.env.last(observe=False)
#         obs = self.observe(self.env.agent_selection) if observe else None
#         return obs, reward, term, trunc, info

#     def close(self):
#         if self.recording:
#             self._stop_recording()
#         super().close()


# TODO: this is from the old library
# class GymRecordVideo(gym.Wrapper):
#     def __init__(
#         self,
#         env: gym.Env,
#         video_folder: str = "videos",
#         episode_trigger: Optional[Callable[[int], bool]] = None,
#         video_length: int = 0,
#         name_prefix: str = "episode",
#         fps: int = 30,
#     ):
#         super().__init__(env)
#         self.video_folder = video_folder
#         self.episode_trigger = episode_trigger or EpisodeTrigger(1000)
#         self.video_length, self.name_prefix, self.fps = video_length, name_prefix, fps
#         os.makedirs(self.video_folder, exist_ok=True)
#         self.recording, self.video_writer, self.frames_recorded, self.episode_id = (
#             False,
#             None,
#             0,
#             0,
#         )

#     def reset(self, **kwargs):
#         res = super().reset(**kwargs)
#         if self.episode_trigger(self.episode_id):
#             self._start_recording()
#         elif self.recording:
#             self._stop_recording()
#         if self.recording:
#             self._capture_frame()
#         return res

#     def step(self, action):
#         obs, reward, term, trunc, info = super().step(action)
#         done = term or trunc
#         if self.recording:
#             self._capture_frame()
#         if self.recording and (
#             done
#             or (self.video_length > 0 and self.frames_recorded >= self.video_length)
#         ):
#             self._stop_recording()
#         if done:
#             self.episode_id += 1
#         return obs, reward, term, trunc, info

#     def _start_recording(self):
#         if self.recording:
#             self._stop_recording()
#         video_path = os.path.join(
#             self.video_folder, f"{self.name_prefix}_{self.episode_id:06d}.mp4"
#         )
#         try:
#             self.video_writer = imageio.get_writer(
#                 video_path, fps=self.fps, macro_block_size=1
#             )
#             self.recording, self.frames_recorded = True, 0
#         except Exception as e:
#             print(f"Warning: Could not open video writer: {e}")

#     def _stop_recording(self):
#         if self.video_writer:
#             self.video_writer.close()
#             self.video_writer = None
#         self.recording, self.frames_recorded = False, 0

#     def _capture_frame(self):
#         if self.recording and self.video_writer:
#             frame = self.env.render()
#             if frame is not None:
#                 if frame.ndim == 3 and frame.shape[2] == 4:
#                     frame = frame[:, :, :3]
#                 self.video_writer.append_data(frame)
#                 self.frames_recorded += 1

#     def close(self):
#         if self.recording:
#             self._stop_recording()
#         super().close()


# def wrap_recording(env, **kwargs):
#     is_pz = hasattr(env, "possible_agents") and hasattr(env, "agent_selection")
#     if not is_pz and hasattr(env, "unwrapped"):
#         is_pz = hasattr(env.unwrapped, "possible_agents") and hasattr(
#             env.unwrapped, "agent_selection"
#         )
#     return RecordVideo(env, **kwargs) if is_pz else GymRecordVideo(env, **kwargs)
