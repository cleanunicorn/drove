"""TTS engine backends. Each engine synthesizes one paragraph at a time and
exposes a sample rate. Integrated from cleanunicorn/cathy at
db303bae77440f3adccb7345c592a67aad3522a1; see CATHY-LICENSE.

Cathy’s chunk splitting, voice blending, and CUDA OOM recovery are retained."""

from __future__ import annotations

import re
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

ENGINES = ("kokoro", "qwen", "chatterbox", "fish")

# Top-level module each engine extra provides, to detect availability.
ENGINE_IMPORTS = {"qwen": "qwen_tts", "chatterbox": "chatterbox", "fish": "fish_speech"}


def engine_available(name: str) -> bool:
    """True if the engine can run in the current environment."""
    if name == "kokoro":
        return True
    import importlib.util

    return importlib.util.find_spec(ENGINE_IMPORTS[name]) is not None


# Curated subset of Kokoro's built-in voices. Prefix encodes language/gender:
# a=American, b=British English; f=female, m=male.
KOKORO_VOICES = {
    "af_heart": "American female (highest rated)",
    "af_bella": "American female, warm",
    "af_nicole": "American female, soft-spoken",
    "am_michael": "American male",
    "am_fenrir": "American male, deep",
    "bf_emma": "British female",
    "bm_george": "British male",
    "bm_fable": "British male, narrator",
}

KOKORO_ALIASES = {
    "male": "am_michael:2,am_fenrir:1",
    "female": "af_heart",
    "british-male": "bm_george:2,bm_fable:1",
    "british-female": "bf_emma",
}

# Non-English Kokoro presets; the prefix picks the language and its G2P runs
# through espeak-ng, which cathy already requires. (Japanese/Chinese presets
# exist upstream too, but need the misaki[ja]/misaki[zh] extras.)
KOKORO_INTL = {
    "ef_dora": "Spanish female",
    "em_alex": "Spanish male",
    "ff_siwis": "French female",
    "hf_alpha": "Hindi female",
    "hm_omega": "Hindi male",
    "if_sara": "Italian female",
    "im_nicola": "Italian male",
    "pf_dora": "Portuguese (BR) female",
    "pm_alex": "Portuguese (BR) male",
}

QWEN_SPEAKERS = {
    "Ryan": "male, dynamic",
    "Aiden": "male, casual American",
    "Eric": "male, lively Sichuan accent",
    "Dylan": "male, Beijing accent",
    "Uncle_Fu": "male, seasoned",
    "Vivian": "female, bright",
    "Serena": "female, gentle",
    "Ono_Anna": "female, Japanese",
    "Sohee": "female, Korean",
}

QWEN_ALIASES = {"male": "Ryan", "female": "Vivian"}

# Read by Kokoro to build a reference clip when a cloning engine (chatterbox,
# fish) is asked for a generic 'male'/'female' voice instead of a .wav sample.
REFERENCE_TEXT = (
    "The old lighthouse keeper climbed the spiral stairs slowly, "
    "carrying a lantern in one hand and a worn leather book in the other. "
    "Outside, the waves rolled in from the dark Atlantic, "
    "steady and patient as they had always been."
)


# CJK punctuation carries no trailing space, so the whitespace that ends a
# Western sentence has to be optional for those scripts.
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|(?<=[。！？])\s*")
_CLAUSE_END = re.compile(r"(?<=[,;:—–])\s+|(?<=[、，；：])\s*")


def _hard_split(text: str, limit: int) -> list[str]:
    """Last resort for a run of text with no usable punctuation: break it on
    whitespace, then mid-run if a single stretch is still too long — Chinese
    and Japanese put no spaces between words, so whitespace alone can't be
    relied on to get under the limit."""
    pieces, current = [], ""
    for word in text.split():
        while len(word) > limit:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(word[:limit])
            word = word[limit:]
        if not word:
            continue
        if current and len(current) + len(word) + 1 > limit:
            pieces.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        pieces.append(current)
    return pieces


def _units(text: str, limit: int) -> list[str]:
    """Break text into pieces that are each at most `limit` characters,
    preferring sentence ends, then clause punctuation, then whitespace."""
    units = []
    for sentence in _SENTENCE_END.split(text):
        if len(sentence) <= limit:
            units.append(sentence)
            continue
        for clause in _CLAUSE_END.split(sentence):
            if len(clause) <= limit:
                units.append(clause)
            else:
                units.extend(_hard_split(clause, limit))
    return [u for u in units if u]


def _pack(units: list[str], size: int) -> list[str]:
    """Join units into groups of at most `size` characters, greedily."""
    groups, current = [], ""
    for unit in units:
        if current and len(current) + len(unit) + 1 > size:
            groups.append(current)
            current = unit
        else:
            current = f"{current} {unit}".strip()
    if current:
        groups.append(current)
    return groups


def split_sentences(text: str, limit: int = 1000) -> list[str]:
    """Split text into groups of at most `limit` characters.

    Text that already fits is returned whole: engines read a full paragraph
    with better prosody than a string of fragments, and every group boundary
    costs an audible SENTENCE_PAUSE. Longer text splits into roughly equal
    groups, so no chunk is a stray tail, and the limit is a hard guarantee
    even when a single sentence exceeds it."""
    if len(text) <= limit:
        return [text]

    units = _units(text, limit)
    # Fewest groups the text can fit in, then the smallest group size that
    # still achieves it: filling each group to the brim instead would leave
    # the last one a stray fragment.
    count = -(-len(text) // limit)
    low, high = -(-len(text) // count), limit
    while low < high:
        mid = (low + high) // 2
        if len(_pack(units, mid)) <= count:
            high = mid
        else:
            low = mid + 1
    return _pack(units, low)


# Below this the chunks are too short to carry prosody, so a card that still
# OOMs here is simply too small and the error is worth surfacing.
CHUNK_FLOOR = 150


def _synthesize(engine: Any, text: str, synth: Any) -> Any:
    """Yield audio for `text` one chunk at a time, where `synth` renders one
    chunk into a list of audio arrays.

    A character cap can't guarantee the GPU fits a chunk — that depends on the
    card, on what else is using it, and on how much audio the text turns into.
    So on CUDA OOM, halve this engine's chunk limit and re-split everything
    still unsynthesized — not just the chunk that failed, which would leave
    the rest of the paragraph queued at the size that just OOMed and fail
    again a chunk later. `synth` must finish before anything is yielded, or a
    retry would duplicate audio that the caller already wrote. Recovery is
    best-effort: an OOM can leave the allocator fragmented, and past
    CHUNK_FLOOR the error propagates."""
    torch = import_module("torch")

    pending = split_sentences(text, engine.chunk_limit)
    while pending:
        group = pending.pop(0)
        try:
            yield from synth(group)
        except torch.cuda.OutOfMemoryError:
            if engine.chunk_limit <= CHUNK_FLOOR:
                raise
            engine.chunk_limit = max(CHUNK_FLOOR, engine.chunk_limit // 2)
            torch.cuda.empty_cache()
            print(
                f"\nnote: GPU out of memory; continuing with smaller chunks "
                f"(--max-chunk-chars {engine.chunk_limit})",
                file=sys.stderr,
            )
            pending = split_sentences(" ".join([group, *pending]), engine.chunk_limit)


def _quiet() -> None:
    """Silence third-party warnings and logging we cannot act on."""
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # descript-audiotools (fish) has docstrings with invalid escape sequences,
    # which warn on first import in a fresh environment
    warnings.filterwarnings("ignore", category=SyntaxWarning)

    try:
        logger = import_module("loguru").logger

        for noisy in ("kokoro", "misaki", "fish_speech"):
            logger.disable(noisy)
    except ImportError:
        pass

    try:
        hf_logging = import_module("transformers.utils.logging")

        hf_logging.set_verbosity_error()
    except ImportError:
        pass


def _hushed(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Run fn with stdout/stderr swallowed — engines emit chatter and their own
    tqdm bars that fight our progress bar. Replayed if fn actually fails."""
    import io
    from contextlib import redirect_stderr, redirect_stdout

    buffer = io.StringIO()
    try:
        with redirect_stdout(buffer), redirect_stderr(buffer):
            return fn(*args, **kwargs)
    except Exception:
        sys.stderr.write(buffer.getvalue())
        raise


class KokoroEngine:
    """Kokoro-82M: fastest by far, preset voices, best fit for whole books."""

    sr = 24_000

    def __init__(
        self,
        voice: str,
        speed: float,
        device: str,
        language: str | None = None,
        model_dir: Path | None = None,
    ) -> None:
        # Kokoro's language follows the voice prefix (see KOKORO_INTL);
        # --language is a qwen option.
        _quiet()
        KPipeline = import_module("kokoro").KPipeline

        voice = KOKORO_ALIASES.get(voice, voice)
        lang_code = voice[0]  # Kokoro convention: voice prefix picks the language
        self.model_dir = model_dir
        options: dict[str, Any] = {}
        if model_dir is not None:
            KModel = import_module("kokoro").KModel
            checkpoint = model_dir / "kokoro-v1_0.pth"
            if not checkpoint.is_file():
                candidates = sorted(model_dir.glob("*.pth"))
                if len(candidates) != 1:
                    raise ValueError("Expected a Kokoro .pth checkpoint in the downloaded model.")
                checkpoint = candidates[0]
            options["model"] = (
                KModel(
                    repo_id="hexgrad/Kokoro-82M",
                    config=str(model_dir / "config.json"),
                    model=str(checkpoint),
                )
                .to(device)
                .eval()
            )
        self.pipeline = KPipeline(
            lang_code=lang_code,
            repo_id="hexgrad/Kokoro-82M",
            device=device,
            **options,
        )
        self.voice = self._resolve(voice)
        self.speed = speed

    def _resolve(self, spec: str) -> Any:
        """Support weighted blends like 'af_heart:2,af_bella:1'."""
        if ":" not in spec and "," not in spec:
            return self._load_voice(spec) if self.model_dir is not None else spec

        torch = import_module("torch")

        names, weights = [], []
        for part in spec.split(","):
            name, _, weight = part.partition(":")
            names.append(name.strip())
            weights.append(float(weight or 1))
        packs = torch.stack([self._load_voice(n) for n in names])
        w = torch.tensor(weights, dtype=packs.dtype) / sum(weights)
        return torch.sum(packs * w.reshape(-1, *[1] * (packs.dim() - 1)), dim=0)

    def _load_voice(self, name: str) -> Any:
        if self.model_dir is not None:
            voice = self.model_dir / "voices" / f"{name}.pt"
            if not voice.is_file():
                raise ValueError(f"Voice '{name}' is absent from the downloaded model.")
            return self.pipeline.load_single_voice(str(voice))
        return self.pipeline.load_single_voice(name)

    def validate_voice(self, name: str) -> None:
        """Check the downloaded voice packs or the pipeline's repository/cache."""
        EntryNotFoundError = import_module("huggingface_hub.errors").EntryNotFoundError
        try:
            self._load_voice(name)
        except EntryNotFoundError as exc:
            raise ValueError(f"Unknown Kokoro voice '{name}'.") from exc

    def paragraph(self, text: str) -> Any:
        for _, _, audio in self.pipeline(text, voice=self.voice, speed=self.speed):
            yield audio.numpy()


class QwenEngine:
    """Qwen3-TTS-0.6B-CustomVoice: expressive preset speakers, ~real-time."""

    # Not VRAM-bound the way fish is — peak memory sits at ~2.8 GiB from 600
    # to 1400 chars. The ceiling is fidelity: at 1400 chars generation ran
    # long and produced 45% more audio than the text calls for (repetition or
    # trailing babble). Clean at 600-1000 on both test texts; 800 keeps
    # margin below where it starts to wander.
    chunk_limit = 800

    def __init__(
        self,
        voice: str,
        speed: float,
        device: str,
        language: str | None = None,
        model_dir: Path | None = None,
    ) -> None:
        self.language = language or "English"
        _quiet()
        import shutil

        if shutil.which("sox") is None:
            sys.exit(
                "error: the qwen engine needs the sox system package (e.g. sudo apt install sox)"
            )
        try:
            Qwen3TTSModel = import_module("qwen_tts").Qwen3TTSModel
        except ImportError:
            sys.exit("error: qwen engine not installed in the drove TTS runtime")

        torch = import_module("torch")

        voice = QWEN_ALIASES.get(voice, voice)
        if voice not in QWEN_SPEAKERS:
            sys.exit(f"error: unknown qwen speaker {voice!r}; options: {', '.join(QWEN_SPEAKERS)}")
        self.voice = voice
        self.model = Qwen3TTSModel.from_pretrained(
            str(model_dir) if model_dir is not None else "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            device_map=device,
            dtype=torch.bfloat16,
        )
        wavs, self.sr = self._generate("Hello.")  # warm up and learn sample rate

    def _generate(self, text: str) -> Any:
        return _hushed(
            self.model.generate_custom_voice,
            text=text,
            language=self.language,
            speaker=self.voice,
        )

    def paragraph(self, text: str) -> Any:
        def synth(group: Any) -> Any:
            wavs, _ = self._generate(group)
            return [wavs[0]]

        yield from _synthesize(self, text, synth)


class ClonedVoiceMixin:
    """Resolve 'male'/'female'/path specs into a reference clip for cloning."""

    def reference_clip(self, voice: str, device: str) -> Path:
        if voice not in ("male", "female", "british-male", "british-female"):
            path = Path(voice)
            if path.suffix.lower() != ".wav" or not path.exists():
                sys.exit(
                    f"error: {type(self).__name__} needs a voice of 'male', 'female', "
                    f"or a path to an existing reference .wav, got {voice!r}"
                )
            return path

        sf = import_module("soundfile")

        cache = Path.home() / ".cache" / "drove" / "tts"
        cache.mkdir(parents=True, exist_ok=True)
        clip = cache / f"reference-{voice}.wav"
        if not clip.exists():
            kokoro = KokoroEngine(voice, 1.0, device)
            np = import_module("numpy")

            audio = np.concatenate(list(kokoro.paragraph(REFERENCE_TEXT)))
            sf.write(clip, audio, kokoro.sr)
        return clip


class ChatterboxEngine(ClonedVoiceMixin):
    """Chatterbox-Nano: 110M cloning model, expressive, MIT-licensed."""

    # Not measured — chatterbox isn't installed on the machine fish and qwen
    # were profiled on, and both of those degraded silently (dropped or
    # invented audio, no error) above their measured ceilings. This 110M model
    # is the smallest of the three, so it stays near the 300 it was known to
    # cope with until someone runs the same profile on it.
    chunk_limit = 400

    def __init__(
        self,
        voice: str,
        speed: float,
        device: str,
        language: str | None = None,
        model_dir: Path | None = None,
    ) -> None:
        _quiet()
        try:
            ChatterboxTurboTTS = import_module("chatterbox.tts_turbo").ChatterboxTurboTTS
        except ImportError:
            sys.exit("error: chatterbox engine not installed in the drove TTS runtime")

        self.clip = str(self.reference_clip(voice, device))
        self.model = (
            ChatterboxTurboTTS.from_local(model_dir, device=device, nano=True)
            if model_dir is not None
            else ChatterboxTurboTTS.from_pretrained(device=device, nano=True)
        )
        self.sr = self.model.sr

    def paragraph(self, text: str) -> Any:
        def synth(group: Any) -> Any:
            wav = _hushed(self.model.generate, group, audio_prompt_path=self.clip)
            return [wav.squeeze(0).cpu().numpy()]

        yield from _synthesize(self, text, synth)


class FishEngine(ClonedVoiceMixin):
    """Fish Audio OpenAudio S1-mini: 0.5B cloning model, research license."""

    # Two ceilings, both measured on a 10 GB card. VRAM: the DAC decoder
    # synthesizes a request's whole audio in one pass, so peak memory tracks
    # request length — 300/600/800 chars peak at 6.0/7.2/8.7 GiB and 2000
    # chars OOMs. Fidelity, and the tighter of the two: around 1000 chars the
    # model starts ending generation early and silently drops text (one test
    # paragraph came back 56% short, reproducibly). 600 keeps ~2.5 GB of
    # headroom and stays well clear of the truncation region.
    chunk_limit = 600

    def __init__(
        self,
        voice: str,
        speed: float,
        device: str,
        language: str | None = None,
        model_dir: Path | None = None,
    ) -> None:
        _quiet()
        try:
            fish_speech = import_module("fish_speech")
        except ImportError:
            sys.exit("error: fish engine not installed in the drove TTS runtime")

        # fish-speech modules locate their repo root via a .project-root marker
        # at import time; recreate it next to the installed package.
        marker = Path(next(iter(fish_speech.__path__))).parent / ".project-root"
        if not marker.exists():
            marker.touch()

        TTSInferenceEngine = import_module("fish_speech.inference_engine").TTSInferenceEngine

        # The generation loop runs in a worker thread and draws its own tqdm
        # bar per paragraph (stream redirection can't reach it); disable it at
        # the module reference it actually uses.
        from functools import partial

        _tqdm = import_module("tqdm").tqdm

        _t2s: Any = import_module("fish_speech.models.text2semantic.inference")

        _t2s.tqdm = partial(_tqdm, disable=True)

        torch = import_module("torch")
        load_decoder_model = import_module("fish_speech.models.dac.inference").load_model
        launch_thread_safe_queue = import_module(
            "fish_speech.models.text2semantic.inference"
        ).launch_thread_safe_queue
        ServeReferenceAudio = import_module("fish_speech.utils.schema").ServeReferenceAudio
        snapshot_download = import_module("huggingface_hub").snapshot_download

        clip = self.reference_clip(voice, device)
        if voice in ("male", "female", "british-male", "british-female"):
            reference_text = REFERENCE_TEXT
        else:
            transcript = clip.with_suffix(".txt")
            if not transcript.is_file() or not (reference_text := transcript.read_text().strip()):
                raise ValueError(
                    "Fish custom reference audio requires a nonempty companion .txt transcript."
                )
        self.reference = ServeReferenceAudio(audio=clip.read_bytes(), text=reference_text)

        GatedRepoError = import_module("huggingface_hub.errors").GatedRepoError

        try:
            checkpoints = (
                model_dir if model_dir is not None else Path(snapshot_download("fishaudio/s1-mini"))
            )
        except GatedRepoError:
            set_token = (
                "setx HF_TOKEN hf_yourtoken   (then open a NEW terminal)"
                if sys.platform == "win32"
                else "export HF_TOKEN=hf_yourtoken"
            )
            sys.exit(
                "error: fishaudio/s1-mini is a gated model; downloading it "
                "needs a (free) authenticated Hugging Face account:\n"
                "  1. accept the terms at https://huggingface.co/fishaudio/s1-mini\n"
                "  2. create a token at https://huggingface.co/settings/tokens\n"
                f"  3. {set_token}\n"
                "then retry."
            )
        precision = torch.bfloat16
        llama_queue = launch_thread_safe_queue(
            checkpoint_path=checkpoints, device=device, precision=precision, compile=False
        )
        # The codec loader prints stray zero-byte tqdm bars ("Reconstruction
        # complete", "Download complete") to stderr; swallow them, but replay
        # everything if loading actually fails.
        import io
        from contextlib import redirect_stderr, redirect_stdout

        buffer = io.StringIO()
        try:
            with redirect_stdout(buffer), redirect_stderr(buffer):
                decoder = load_decoder_model(
                    config_name="modded_dac_vq",
                    checkpoint_path=checkpoints / "codec.pth",
                    device=device,
                )
        except Exception:
            sys.stderr.write(buffer.getvalue())
            raise
        self.engine = TTSInferenceEngine(
            llama_queue=llama_queue,
            decoder_model=decoder,
            precision=precision,
            compile=False,
        )
        if hasattr(decoder, "spec_transform"):
            self.sr = int(decoder.spec_transform.sample_rate)
        else:
            self.sr = int(decoder.sample_rate)

    def paragraph(self, text: str) -> Any:
        ServeTTSRequest = import_module("fish_speech.utils.schema").ServeTTSRequest

        # One request per chunk (see chunk_limit above). use_memory_cache
        # reuses the encoded reference clip across requests instead of
        # re-encoding it every time.
        def synth(group: Any) -> Any:
            request = ServeTTSRequest(
                text=group,
                references=[self.reference],
                max_new_tokens=2048,
                seed=42,  # keep the timbre stable across paragraphs
                format="wav",
                use_memory_cache="on",
            )
            segments = []
            for result in _hushed(lambda: list(self.engine.inference(request))):
                if result.code == "error":
                    raise result.error
                if result.code == "segment" and result.audio is not None:
                    segments.append(result.audio[1])
                elif result.code == "final" and result.audio is not None and not segments:
                    segments.append(result.audio[1])
            return segments

        yield from _synthesize(self, text, synth)


def build_engine(
    name: str,
    voice: str | None,
    speed: float,
    device: str,
    language: str | None = None,
    max_chunk_chars: int | None = None,
    model_dir: Path | None = None,
) -> Any:
    classes: dict[str, Any] = {
        "kokoro": KokoroEngine,
        "qwen": QwenEngine,
        "chatterbox": ChatterboxEngine,
        "fish": FishEngine,
    }
    engine: Any = classes[name](voice or "male", speed, device, language, model_dir)
    if max_chunk_chars is not None:
        # Kokoro has no chunk_limit: it hands whole paragraphs to KPipeline,
        # which chunks internally.
        engine.chunk_limit = max(CHUNK_FLOOR, max_chunk_chars)
    return engine
