from app.config import Settings
from app.generation.llm import Generation, parse_generation
from app.generation.prompts import build_messages
from app.schemas import RetrievedChunk


class TransformersClient:
    def __init__(self, settings: Settings):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.s = settings
        parts = [x for x in settings.llm_model.replace("\\", "/").rstrip("/").split("/") if x and not x.isdigit()]
        self.name = f"transformers/{parts[-1] if parts else settings.llm_model}"
        cuda = torch.cuda.is_available()
        if settings.llm_device == "cuda" and not cuda:
            raise RuntimeError("AEGIS_LLM_DEVICE=cuda but no CUDA device is available")
        self.device = "cuda" if (settings.llm_device in ("auto", "cuda") and cuda) else "cpu"
        if settings.llm_dtype != "auto":
            dtype = getattr(torch, settings.llm_dtype)
        else:
            dtype = torch.float16 if self.device == "cuda" else torch.float32

        torch.manual_seed(settings.llm_seed)
        self.tokenizer = AutoTokenizer.from_pretrained(settings.llm_model, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.llm_model, dtype=dtype, local_files_only=True
        ).to(self.device).eval()
        self.torch = torch

    def generate(self, question: str, context: list[RetrievedChunk]) -> Generation:
        messages = build_messages(question, context, self.s.refusal_message)
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.s.llm_max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        raw = self.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return parse_generation(raw, self.s.refusal_message)
