import os, json, pyarrow as pa, pyarrow.parquet as pq

src_dir = "/workspace/qwen-image-training/data/breast-edit-dataset/input"      # source
ref_dir = "/workspace/qwen-image-training/data/breast-edit-dataset/breast"     # reference
tgt_dir = "/workspace/qwen-image-training/data/breast-edit-dataset/output"     # target

rows = []
for name in sorted(os.listdir(src_dir)):
    base = os.path.splitext(name)[0]
    with open(os.path.join(src_dir,  f"{base}.png"), "rb") as f: src = f.read()
    with open(os.path.join(ref_dir,  f"{base}.png"), "rb") as f: ref = f.read()
    with open(os.path.join(tgt_dir,  f"{base}.png"), "rb") as f: tgt = f.read()
    rows.append({
        "input_image": src,                   # recognised as a source
        "reference_image": ref,               # any of the reference aliases works
        "target_image": tgt,                  # recognised as a target
    })

table = pa.Table.from_pylist(rows)
out_dir  = "/workspace/qwen-image-training/data/breast-edit-dataset/parquet"
os.makedirs(out_dir, exist_ok=True)
parquet_path = os.path.join(out_dir, "breast_edits.parquet")
pq.write_table(table, parquet_path)

info = {parquet_path: {"num_row_groups": pq.ParquetFile(parquet_path).num_row_groups}}
with open("/workspace/qwen-image-training/data/breast-edit-dataset/parquet_info.json", "w") as f:
    json.dump(info, f)

