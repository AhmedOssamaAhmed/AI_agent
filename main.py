#!/usr/bin/env python3
import argparse
import datasets
import requests
import json
# poetry run python main.py --base_url http://localhost:8000

def call_server(query, tools, url="http://localhost:8000"):
    tools = json.loads(tools)
    url = f"{url}/ask_model"
    payload = {"query": query, "tools": tools}
    headers = {"Content-Type": "application/json"}
    resp = requests.post(url, json=payload, headers=headers)
    try:
        resp.raise_for_status()
    except requests.exceptions.HTTPError:
        print("Status:", resp.status_code)
        print("Response body:", resp.text)
        raise
    return resp.json()["content"]

def process_dataset(
    ds: datasets.Dataset, model: str, base_url: str, api_key: str
) -> datasets.Dataset:
    my_answers = []
    
    for query, tools in zip(ds["query"], ds["tools"]):
        try:
            model_answer = call_server(query, tools, base_url)
            print(f"\n model_answer {type(model_answer)} \n {model_answer}")
        except Exception as e:
            print(f"Error calling the server: {e}")
            answer = f"Error for query: {query}"

        # formatted_answer = answer
        my_answers.append(model_answer)

    # you should add the new column 'my_answers' with the expected tool calls to the dataset
    return ds.add_column("my_answers", my_answers)


def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Generate tool calls using an LLM")
    parser.add_argument("--model", required=False, help="Name of the model to use")
    parser.add_argument("--dataset", required=False,default="./dataset/", help="path of dataset folder")
    parser.add_argument(
        "--base_url",
        required=True,
        help="Base URL of the inference server, e.g. http://localhost:8000",
    )
    parser.add_argument(
        "--api_key", required=False, help="API key for the inference server"
    )
    args = parser.parse_args()

    ds = datasets.load_from_disk(args.dataset)
    assert isinstance(ds, datasets.Dataset)
    # Process the dataset and generate tool calls
    submission_ds = process_dataset(ds, args.model, args.base_url, args.api_key)
    # Save the resulting dataset
    submission_ds.save_to_disk("./my_dataset")


if __name__ == "__main__":
    main()
