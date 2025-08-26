from fastapi import FastAPI
from pydantic import BaseModel
from llmlingua import PromptCompressor

app = FastAPI()

# Ładujemy model do kompresji promptów (LLMLingua-2 jest szybszy)
compressor = PromptCompressor("microsoft/llmlingua-2")

# Definiujemy format requestu
class PromptRequest(BaseModel):
    text: str
    target_tokens: int = 200  # domyślny limit tokenów

@app.post("/compress")
def compress_prompt(req: PromptRequest):
    """Kompresja promptu do określonej liczby tokenów."""
    result = compressor.compress(req.text, target_token=req.target_tokens)
    return {"compressed": result["compressed_prompt"]}
