1. Pull the docker images for mini-swe-agent to save time because it is a large image
docker pull docker.io/swebench/sweb.eval.x86_64.sympy_1776_sympy-17139:latest

2. Run with Grok model
export XAI_API_KEY="your-xai-api-key"
python3 run_tasks.py --model "grok-2" --workers 1