import time
import psutil
import random
from prometheus_client import start_http_server, Gauge, Counter

# ==============================
# Definisi 10 Metriks Unik
# ==============================
CPU_USAGE = Gauge('system_cpu_usage', 'Persentase penggunaan CPU')
RAM_USAGE = Gauge('system_ram_usage', 'Persentase penggunaan RAM')
DISK_USAGE = Gauge('system_disk_usage', 'Persentase penggunaan Disk')
API_REQUESTS = Counter('api_requests_total', 'Total permintaan API')
FAILED_REQUESTS = Counter('failed_requests_total', 'Total permintaan API yang gagal')
MODEL_INFERENCE = Counter('model_inference_total', 'Total prediksi/inferensi model')
MODEL_ACCURACY = Gauge('model_accuracy', 'Akurasi model saat ini')
MODEL_F1SCORE = Gauge('model_f1score', 'F1 Score model saat ini')
NETWORK_BYTES_SENT = Counter('network_bytes_sent', 'Total data terkirim (bytes)')
NETWORK_BYTES_RECEIVED = Counter('network_bytes_received', 'Total data diterima (bytes)')

# ==============================
# Update metriks setiap 5 detik
# ==============================
def update_metrics():
    while True:
        # Metriks sistem
        CPU_USAGE.set(psutil.cpu_percent())
        RAM_USAGE.set(psutil.virtual_memory().percent)
        DISK_USAGE.set(psutil.disk_usage('/').percent)
        NETWORK_BYTES_SENT.inc(psutil.net_io_counters().bytes_sent)
        NETWORK_BYTES_RECEIVED.inc(psutil.net_io_counters().bytes_recv)

        # Metriks proses / model
        API_REQUESTS.inc(random.randint(0, 5))
        FAILED_REQUESTS.inc(random.randint(0, 2))
        MODEL_INFERENCE.inc(random.randint(1, 3))
        MODEL_ACCURACY.set(round(random.uniform(0.75, 0.85), 4))
        MODEL_F1SCORE.set(round(random.uniform(0.5, 0.7), 4))

        time.sleep(5)

# ==============================
# Main
# ==============================
if __name__ == '__main__':
    start_http_server(8000)
    print("Exporter jalan di http://localhost:8000")
    update_metrics()
