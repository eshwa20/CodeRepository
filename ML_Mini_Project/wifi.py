from pyngrok import ngrok
import subprocess
import time

# Start Streamlit in background
streamlit_process = subprocess.Popen(["streamlit", "run", "app.py"])

# Set up ngrok tunnel
public_url = ngrok.connect(8501)  # 8501 is Streamlit's default port
print(f"Public URL: {public_url}")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Shutting down...")
    ngrok.kill()
    streamlit_process.terminate()