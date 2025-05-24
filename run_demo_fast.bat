@echo off
echo Running UnLook Gesture Demo with optimal settings...
echo.

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Run demo with optimal settings:
REM --downsample 1 = No downsampling (full resolution)
REM --presentation-mode = Optimized for demos
REM Remove --ip if you want auto-discovery

python unlook\examples\handpose\enhanced_gesture_demo.py --downsample 1 --presentation-mode

pause