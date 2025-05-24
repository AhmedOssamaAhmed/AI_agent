import datasets
import json

original_ds = datasets.load_from_disk("./dataset/")
predicted_ds = datasets.load_from_disk("./my_dataset/")

def safe_match(o, p):
    try:
        return json.loads(o) == json.loads(p)
    except json.JSONDecodeError:
        return o == p

exact_counter = 0
for original, predicted in zip(original_ds["answers"], predicted_ds["answers"]):
    if not safe_match(original, predicted):
        print("Mismatch:")
        print("Original:", original)
        print("Predicted:", predicted)
    else:
        exact_counter += 1

print(f"accuracy {(exact_counter/len(original_ds))*100}%")