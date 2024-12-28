def validate_data(data):
    if not isinstance(data, np.ndarray):
        raise ValueError("Data must be a numpy array.")
    if data.shape[1] != 4:
        raise ValueError("Data must have 4 columns: open, close, high, low.")
