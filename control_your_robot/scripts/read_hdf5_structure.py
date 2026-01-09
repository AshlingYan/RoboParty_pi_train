import h5py
import sys

def print_structure(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name}  | Shape: {obj.shape} | Type: {obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"Group:   {name}")

if len(sys.argv) < 2:
    print("Usage: python check_h5.py <file.hdf5>")
    sys.exit(1)

print(f"--- File: {sys.argv[1]} ---")
with h5py.File(sys.argv[1], 'r') as f:
    f.visititems(print_structure)