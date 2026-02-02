from datasets import load_dataset

ds = load_dataset("princeton-nlp/SWE-bench_Lite")
subset = ds["test"].shuffle(seed=42).select(range(50))
subset.to_json("tasks_50.json")
