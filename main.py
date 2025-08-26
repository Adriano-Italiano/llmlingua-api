from fastapi import FastAPI
from pydantic import BaseModel
from llmlingua import PromptCompressor

app = FastAPI()
compressor = None  # lazy load

class PromptRequest(BaseModel):
    text: str
    target_tokens: int = 200

@app.post("/compress")
def compress_prompt(req: PromptRequest):
    global compressor
    if compressor is None:
        compressor = PromptCompressor("microsoft/llmlingua-2")  # model ładuje się dopiero przy 1. wywołaniu
    result = compressor.compress(req.text, target_token=req.target_tokens)
    return {"compressed": result["compressed_prompt"]}
