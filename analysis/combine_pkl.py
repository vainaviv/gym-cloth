import pickle

combined = []
with open('logs/cloth_smoothing_data_30x30_1148_iters.pkl', 'rb') as f:
    combined = pickle.load(f)
    print(len(combined))
with open('logs/cloth_smoothing_data_30x30_852_iters.pkl', 'rb') as f:
    combined.extend(pickle.load(f))
    print(len(combined))
with open('logs/cloth_smoothing_data_30x30_2000_iters.pkl', 'wb') as f:
    pickle.dump(combined, f)
