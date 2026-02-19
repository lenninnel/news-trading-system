web: streamlit run dashboard/app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false
worker: python3 scheduler/daily_runner.py --daemon
monitor: python3 monitoring/price_monitor.py --daemon
health: uvicorn deployment.health_server:app --host 0.0.0.0 --port ${HEALTH_PORT:-8080}
