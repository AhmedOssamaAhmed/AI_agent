# ellamind Coding Challenge

This project demonstrates a solution to the ellamind coding challenge, which involves serving and interacting with a local GGUF-based LLM using a simple client-server architecture.


## Setup & Execution

1. Install dependencies:
```bash
poetry install --no-root
```

2. Download the model, e.g. with:
```bash
poetry run huggingface-cli download crusoeai/Arcee-Agent-GGUF arcee-agent.Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False
```

3. run the server side (server.py) using this command 
```bash
poetry run python server.py --model_path path/to/arcee-agent.Q4_K_M.gguf
```
you can specify the port here but if not, it defaults to 8000
and you can specify the host but if not, it defaults to 0.0.0.0 which is the localhost

4. Run the main file to use the model (main.py):
```bash
poetry run python main.py --base_url http://localhost:8000
```
no need to pass the model here or any api keys, just pass the base url of the server.
you can pass the dataset folder path by --dataset but if not passed, the default value is ./dataset/ so make sure that the dataset folder is in the same folder as the script.

### Example

```bash
poetry run python server.py --model_path ./arcee-agent.Q4_K_M.gguf
poetry run python main.py --base_url http://localhost:8000 --dataset ./dataset/
```

### 🧪 Evaluation
Prompt engineering is used to guide the model to return responses in the expected format.
The evaluation compares the model's predictions against the ground truth, and in this setup, the predictions achieved 100% match.

To run the evaluation:

```bash
poetry run python evaluate_predictions.py
```
Notes:

The original dataset (including the answers column) must be located at ./dataset/

The predicted results must be located at ./my_dataset/