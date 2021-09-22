import pickle

combined = []
with open('logs/cloth_smoothing_data_30x30.pkl', 'rb') as f:
    combined = pickle.load(f)
with open('logs/cloth_smoothing_data_30x30_47.pkl', 'rb') as f:
    combined.extend(pickle.load(f))
with open('logs/cloth_smoothing_data_30x30_all.pkl', 'wb') as f:
    pickle.dump(combined, f)