import csv
from datasets import load_dataset

def load_jfleg_dataset(split="validation[:]"):
    return load_dataset("jfleg", split=split)

def generate_csv(csv_path, dataset, limit_to_one_target=False):
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["input", "target"])
        for case in dataset:
            input_text = "grammar: " + case["sentence"].strip()
            corrections = [c.strip() for c in case["corrections"] if c.strip()]
            if limit_to_one_target and corrections:
                writer.writerow([input_text, corrections[0]])
            elif not limit_to_one_target:
                for correction in corrections:
                    writer.writerow([input_text, correction])