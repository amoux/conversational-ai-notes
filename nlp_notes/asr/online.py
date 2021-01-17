import time
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import pyaudio
import torch
from nemo.collections.asr.models import EncDecCTCModel
from nemo.core.classes import IterableDataset
from nemo.core.neural_types import AudioSignal, LengthsType, NeuralType
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from wasabi import msg

from .utils import spectrogram_normalization


def speech_to_text_demo(asr: "ASROnlineAudioFrame") -> None:
    """Speech to Text (ASR) Microphone Demo.

    Interrupt the notebook's kernel to stop the app from recoring.
    """
    asr.reset()
    audio = pyaudio.PyAudio()

    offset = {"count": 0}
    columns = []
    devices = []

    for idx in range(audio.get_device_count()):
        device = audio.get_device_info_by_index(idx)
        if not device.get("maxInputChannels"):
            continue
        devices.append(idx)
        columns.append((idx, device.get("name")))

    if columns:
        msg.good("Found the following input devices!")
        msg.table(columns, header=("ID", "Devices"), divider=True)

    if devices:
        device_index = -2
        while device_index not in devices:
            msg.info("Please enter the device ID")
            device_index = int(input())

        def callback(in_data, frame_count, time_info, status):
            signal = np.frombuffer(in_data, dtype=np.int16)
            text = asr.transcribe(signal)
            if text:
                print(text, end="")
                offset["count"] = asr.params.offset
            elif offset["count"] > 0:
                offset["count"] -= 1
                if offset["count"] == 0:
                    print(" ", end="")
            return (in_data, pyaudio.paContinue)

        stream = audio.open(
            input=True,
            format=pyaudio.paInt16,
            input_device_index=device_index,
            stream_callback=callback,
            channels=asr.params.channels,
            rate=asr.params.sample_rate,
            frames_per_buffer=asr.chunk_size,
        )

        msg.loading("Listening...")
        stream.start_stream()

        try:
            while stream.is_active():
                time.sleep(0.1)
        except (KeyboardInterrupt, Exception) as e:
            stream.stop_stream()
            stream.close()
            audio.terminate()
            msg.warn("WARNING: ASR stream stopped.", e)
    else:
        msg.fail("ERROR", "No audio input device found.")


class ASROnlineAudioData(IterableDataset):
    def __init__(self, sample_rate: int) -> None:
        self._sample_rate = sample_rate
        self.has_signal: bool = False

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def types(self) -> Dict[str, NeuralType]:
        signal = NeuralType(("B", "T"), AudioSignal(freq=self._sample_rate))
        length = NeuralType(tuple("B"), LengthsType())
        return {"audio_signal": signal, "a_sig_length": length}

    def set(self, signal: np.ndarray) -> None:
        signal = signal.astype(np.float32) / 32768.0
        self.signal_shape = signal.size
        self.signal, self.has_signal = signal, True

    def __iter__(self) -> IterableDataset:
        return self

    def __next__(self) -> Tuple[torch.LongTensor, torch.Tensor]:
        if not self.has_signal:
            raise StopIteration
        self.has_signal = False
        return (
            torch.as_tensor(self.signal, dtype=torch.float32),
            torch.as_tensor(self.signal_shape, dtype=torch.int64),
        )

    def __len__(self) -> int:
        return 1


class ASRInferenceParameters(NamedTuple):
    sample_rate: int = 16000
    window_stride: float = 0.0
    encoder_blocks: List[Dict[str, Any]] = []
    vocabulary: List[str] = []
    frame_length: Union[float, int] = 2.0
    frame_overlap: Union[float, int] = 2.5
    offset: int = 10
    channels: int = 1

    @property
    def buffer_size(self) -> int:
        return 2 * self.n_frame_overlap + self.n_frame_length

    @property
    def n_frame_length(self) -> int:
        return int(self.frame_length * self.sample_rate)

    @property
    def n_frame_overlap(self) -> int:
        return int(self.frame_overlap * self.sample_rate)

    @property
    def chunk_size(self) -> int:
        return int(self.frame_length * self.sample_rate)

    def compute_overlap_timesteps(self):
        timesteps = self.window_stride
        for block in self.encoder_blocks:
            timesteps *= block["stride"][0] ** block["repeat"]
        overlap_timesteps = int(self.frame_overlap / timesteps) - 2
        return overlap_timesteps

    @staticmethod
    def from_omega(cfg: OmegaConf, **kwargs) -> "ASRInferenceParameters":
        return ASRInferenceParameters(
            window_stride=float(cfg.preprocessor.params.window_stride),
            encoder_blocks=list(cfg.encoder.params.jasper),
            vocabulary=list(cfg.decoder.params.vocabulary) + ["_"],
            **kwargs,
        )


class ASRAudioEncoderDecoder(torch.nn.Module):
    def __init__(
        self,
        model: EncDecCTCModel,
        sample_rate: int,
        batch_size: int = 1,
        device: str = "cuda",
    ) -> None:
        super(ASRAudioEncoderDecoder, self).__init__()
        self.online_audio = ASROnlineAudioData(sample_rate)
        self.data_loader = DataLoader(
            dataset=self.online_audio,
            batch_size=batch_size,
            collate_fn=self.online_audio.collate_fn,
        )
        model.eval()
        self.device = torch.device(device)
        self.model = model.to(self.device)

    def forward(self, buffer: np.ndarray) -> torch.Tensor:
        self.online_audio.set(buffer)
        audio_signal, audio_signal_length = next(iter(self.data_loader))
        output = self.model.forward(
            input_signal=audio_signal.to(self.device),
            input_signal_length=audio_signal_length.to(self.device),
        )
        logits = output[0]
        return logits


class ASROnlineAudioFrame:
    def __init__(
        self,
        encoder_decoder: EncDecCTCModel,
        batch_size: int = 1,
        dither: float = 0.0,
        pad_to: int = 0,
        device: str = "cuda",
        **kwargs
    ) -> None:
        cfg = encoder_decoder._cfg
        OmegaConf.set_struct(cfg.preprocessor, value=False)
        cfg.preprocessor.params.dither = dither
        cfg.preprocessor.params.pad_to = pad_to
        cfg.preprocessor.params.normalize = spectrogram_normalization()
        OmegaConf.set_struct(cfg.preprocessor, value=True)
        encoder_decoder.preprocessor = encoder_decoder.from_config_dict(
            cfg.preprocessor
        )
        self.params = ASRInferenceParameters.from_omega(cfg, **kwargs)
        self.overlap_timesteps = self.params.compute_overlap_timesteps()
        self.buffer = np.zeros(self.params.buffer_size, dtype=np.float32)
        sample_rate = self.params.sample_rate
        self.audio_encoder_decoder = ASRAudioEncoderDecoder(
            encoder_decoder, sample_rate, batch_size, device=device,
        )
        self.prev_char = ""
        self.reset()

    @property
    def infer(self) -> np.ndarray:
        signal = self.audio_encoder_decoder(self.buffer)
        return signal.detach().cpu().numpy()[0]

    @property
    def chunk_size(self) -> int:
        return self.params.chunk_size

    def frame_length(self) -> int:
        return self.params.n_frame_length

    def num_overlaps(self) -> int:
        return self.params.n_frame_overlap

    def num_timesteps(self) -> int:
        return self.overlap_timesteps

    def decode(self, frame: np.ndarray, offset=0):
        frame_length = self.frame_length()
        assert len(frame) == frame_length
        self.buffer[:-frame_length] = self.buffer[frame_length:]
        self.buffer[-frame_length:] = frame
        logits = self.infer[self.num_timesteps() : -self.num_timesteps()]
        decoded = self.greedy_decoder(logits, self.params.vocabulary)
        return decoded[: len(decoded) - offset]

    @torch.no_grad()
    def transcribe(self, frame: Optional[np.ndarray] = None, merge=True):
        frame_length = self.frame_length()
        if frame is None:
            frame = np.zeros(frame_length, dtype=np.float32)
        if len(frame) < frame_length:
            padding = [0, frame_length - len(frame)]
            frame = np.pad(frame, padding, mode="constant")
        unmerged = self.decode(frame, offset=self.params.offset)
        if not unmerged:
            return unmerged
        return self.greedy_merge(unmerged)

    def greedy_decoder(self, logits: np.ndarray, vocab: List[str]) -> str:
        tokens = [vocab[np.argmax(logits[i])] for i in range(logits.shape[0])]
        string = "".join(tokens)
        return string

    def greedy_merge(self, string: str) -> str:
        merged = ""
        for i in range(len(string)):
            if string[i] != self.prev_char:
                self.prev_char = string[i]
                if self.prev_char != "_":
                    merged += self.prev_char
        return merged

    def reset(self) -> None:
        self.buffer = np.zeros(self.buffer.shape, dtype=np.float32)
        self.prev_char = ""
