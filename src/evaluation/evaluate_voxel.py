def evaluate_detector(detector, dataset):
    results = []
    for data in dataset:
        keypoints = detector.detect(data)
        # Compute metrics (repeatability, localization error, etc.)
        results.append(keypoints)
    return results
