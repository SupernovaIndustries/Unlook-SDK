@echo off
cd /d %~dp0
call .venv\Scripts\activate
python unlook/examples/scanning/static_scanning_example_fixed.py --server_ip 10.0.0.152 --auto_optimize --debug
pause