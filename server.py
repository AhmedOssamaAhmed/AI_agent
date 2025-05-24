from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama
import argparse
import uvicorn

# poetry run python server.py --model_path arcee-agent.Q4_K_M.gguf


app = FastAPI(title="Tool-Call LLM Server")



class AskModelRequest(BaseModel):
    query: str
    tools: list

class AskModelResponse(BaseModel):
    content: str

@app.post("/ask_model", response_model=AskModelResponse)
def ask_model(request: AskModelRequest):
    llm = app.state.llm
    messages = [
        {
            "role": "system",
            "content": (
                        "You are a tool‐calling agent.  For each user query you must output **only** "
                        "a JSON array of tool calls.  Each element must be an object with exactly two keys: "
                        "`name` (the tool name) and `arguments` (an object of the parameters). "
                        "Do not output any explanation, markdown fences, code syntax, or prose—"
                        "the response must parse as raw JSON."
                            )
        },
        {
            "role": "user",
            "content": f"Available tools: {request.tools}\n\nQuery: {request.query}\n\nGenerate the JSON array of tool calls."
        }
    ]
    try:
        response = llm.create_chat_completion(
            messages=messages,
        )
        return AskModelResponse(content=response['choices'][0]['message']['content'])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM server with configurable model path and context size")
    parser.add_argument("--model_path", required=True, default="arcee-agent.Q4_K_M.gguf", help="Path to the .gguf LLaMA model file")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    args = parser.parse_args()

    try:
        app.state.llm = Llama(
            model_path=args.model_path,
            n_ctx=4096,
            verbose=False
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load model at '{args.model_path}': {e}")

    uvicorn.run(app, host=args.host, port=args.port)
