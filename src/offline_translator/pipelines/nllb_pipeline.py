from functools import partial
from src.offline_translator.models import BaseModel, NLLBModels
from src.offline_translator.pipelines.languages import LanguageTuple, Languages
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch


class BaseNLLBPipeline:
    def __init__(self, model: BaseModel, device: torch.device = torch.device("cpu")):
        """Base class for NLLB pipelines

        Args:
            model (BaseModel): model to use
            device (torch.device, optional): device to use. Defaults to torch.device("cpu").

        Raises:
            ValueError: if device is not supported
        """
        self.model = model
        self._device = None
        self.device = device

        self.tokenizer: AutoTokenizer | None = None
        self.model: AutoModelForSeq2SeqLM | None = None

        self.setup_variables()

    def setup_variables(self):
        """Setup variables for pipeline

        Raises:
            RuntimeError: if model or tokenizer is not found
        """
        try:
            self.tokenizer = self.model.tokenizer_cls.from_pretrained(self.model.model_url)
            self.model = self.model.model_cls.from_pretrained(self.model.model_url)
        except RuntimeError as e:
            raise RuntimeError("RuntimeError: {}".format(e))

    @property
    def device(self):
        """Get device

        Returns:
            torch.device: device

        Raises:
            ValueError: if device is not supported
        """
        return self._device

    # noinspection PyUnresolvedReferences
    @device.setter
    def device(self, device: torch.device):
        """Set device

        Args:
            device (torch.device): device to use

        Raises:
            ValueError: if device is not supported
        """
        # check if torch.cuda.is_available() and device.type == "cuda"
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available on this device")

        self._device = device

        if self.model is not None:
            self.model.to(self._device)

    def decode_tokens(self, tokens: torch.tensor):
        """Decode tokens to string

        Args:
            tokens (torch.tensor (batch_size, seq_len)): tokens to decode

        Returns:
            str: decoded string
        """
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]

    def translate(self, text: str, language: LanguageTuple):
        """Translate text to target language

        Args:
            text (str): text to translate
            language (LanguageTuple): target language

        Returns:
            str: translated text
        """
        raise NotImplementedError


class NLLBPipeline(BaseNLLBPipeline):
    def __init__(self, model: BaseModel, device: torch.device = torch.device("cpu")):
        """NLLB 200 pipeline

        Args:
            device (torch.device, optional): device to use. Defaults to torch.device("cpu").
        """
        super().__init__(model, device)

    def translate(self, text: str, language: LanguageTuple):
        """Translate text

        Args:
            text (str): text to translate
            language (LanguageTuple): language to translate to

        Returns:
            str: translated text
        """
        # encode text
        encoded_text = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

        # generate translation
        generated_translation = self.model.generate(
            encoded_text,
            max_length=len(encoded_text[0]) * 2,
            num_beams=4,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
            use_cache=True,
            forced_bos_token_id=self.tokenizer.lang_code_to_id[language.flores_name]
        )

        # decode tokens
        return self.decode_tokens(generated_translation)


NLLB200Pipeline600M = partial(NLLBPipeline, NLLBModels.NLLB_200_600M)
NLLB200Pipeline1_3B = partial(NLLBPipeline, NLLBModels.NLLB_200_1_3B)
NLLB200Pipeline3_3B = partial(NLLBPipeline, NLLBModels.NLLB_200_3_3B)
