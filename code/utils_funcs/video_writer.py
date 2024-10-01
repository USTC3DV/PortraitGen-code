# import ffmpeg
import numpy as np
import ipdb
import cv2

class VideoWriter():
    def __init__(self, filename, fps=25, width=None, height=None):
        self.filename = filename
        self.fps = fps
        self.writer = None
        self.width = width
        self.height = height

    def write_frame(self, frame):
        if self.writer is None:
            # Initialize VideoWriter with the specified filename, codec, fps, and frame size
            if self.width is None or self.height is None:
                self.height, self.width, _ = frame.shape

            self.writer = cv2.VideoWriter(self.filename, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.width, self.height))
        
        # Write the frame
        self.writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None

# class VideoWriter():
#     def __init__(self, filename, fps = 25):
#         self.filename = filename
#         self.fps = fps
#         self.writer = None

#     def write_frame(self, frame, crf = 23):
#         if self.writer is None:
#             self.writer = ffmpeg.input(
#                 "pipe:0",
#                 format="rawvideo",
#                 pix_fmt="rgb24",
#                 s="{}x{}".format(frame.shape[1], frame.shape[0])
#             ).output(self.filename, pix_fmt="yuv420p", vcodec='libx264', r=self.fps, crf = crf).overwrite_output().run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
#         # ipdb.set_trace()
#         self.writer.stdin.write(np.ascontiguousarray(frame).tobytes())
#         # self.writer.stdin.close()
#         # self.writer.wait()
    
#     def close(self):
#         if self.writer is not None:
#             self.writer.stdin.close()
#             self.writer.wait()
#             self.writer = None