from prometheus_client import Counter, Histogram

prediction_requests_total = Counter(
    "prediction_requests_total",
    "Total number of prediction requests"
)

prediction_latency_seconds = Histogram(
    "prediction_latency_seconds",
    "Prediction request latency in seconds"
)