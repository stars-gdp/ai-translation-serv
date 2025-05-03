from translators.base import BaseTranslator
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM

class NLLBTranslator(BaseTranslator):
    def __init__(self, model_name="facebook/nllb-200-distilled-1.3B"):
        self.tokenizer = NllbTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).half().to("cuda")

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        self.tokenizer.src_lang = src_lang
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)

        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
        output_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=512,
        )

        translated = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
        return translated

    def get_languages(self) -> list[str]:
        return [
            tok.replace("<", "").replace(">", "")
            for tok in self.tokenizer.additional_special_tokens
            if "_" in tok
        ]