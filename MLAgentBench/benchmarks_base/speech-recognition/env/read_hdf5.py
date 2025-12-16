import h5py
import numpy as np

def read_hdf5(file_path):
    with h5py.File(file_path, 'r') as f:
        print("=== HDF5 File Structure ===")
        def print_structure(name, obj):
            obj_type = "Group" if isinstance(obj, h5py.Group) else "Dataset"
            print(f"{name} ({obj_type})")
        f.visititems(print_structure)
        for key in f.keys():
            obj = f[key]
            print(f"\n--- Details for '{key}' ---")
            print("Type:", "Group" if isinstance(obj, h5py.Group) else "Dataset")
            if hasattr(obj, 'shape'):
                print("Shape:", obj.shape)
                print("Data type:", obj.dtype)
        data = f["data"][:]   # actual recorded MEG sensor values
        times = f["times"][:] # corresponding time stamps

    print("\n--- Statistics for 'data' dataset ---")
    print("Minimum value:", np.min(data))
    print("Maximum value:", np.max(data))
    print("Mean value:", np.mean(data))
    print("Standard deviation:", np.std(data))

    print("\nAs you can see, timestamp is 0.004 seconds from the previous:")
    print(times[:10])
    print("This equals a polling rate of 250 Hz.")

    print("\nAnd these are the first first 10 time samples from the first 5 channels of the data:")
    print(data[:5, :10])

    print(times[-1])

    return data
