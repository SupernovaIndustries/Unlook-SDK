@echo off
echo Testing Camera Import Fix...
echo.

cd /d %~dp0
call .venv\Scripts\activate

echo Running static scanning example with debug mode...
echo.

python unlook/examples/scanning/static_scanning_example_fixed.py --server_ip 10.0.0.152 --debug

echo.
echo Test complete. Check output above for camera import errors.
pause