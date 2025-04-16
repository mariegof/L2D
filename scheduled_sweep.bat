@echo off
echo ================================================ > C:\Users\sshuser\desktop\L2D\task_log.txt
echo Task started at %date% %time% >> C:\Users\sshuser\desktop\L2D\task_log.txt
echo ================================================ >> C:\Users\sshuser\desktop\L2D\task_log.txt

REM Create logs directory if it doesn't exist
if not exist C:\Users\sshuser\desktop\L2D\logs mkdir C:\Users\sshuser\desktop\L2D\logs
echo Created logs directory >> C:\Users\sshuser\desktop\L2D\task_log.txt

REM Change to working directory
cd /d C:\Users\sshuser\desktop\L2D
echo Changed to directory: %CD% >> C:\Users\sshuser\desktop\L2D\task_log.txt

REM Activate conda environment
echo Activating conda environment... >> C:\Users\sshuser\desktop\L2D\task_log.txt
call C:\Users\sshuser\anaconda3\Scripts\activate.bat l2d
echo Conda environment activated, PATH=%PATH% >> C:\Users\sshuser\desktop\L2D\task_log.txt

REM Run Python script (without extra arguments)
echo Running Python script... >> C:\Users\sshuser\desktop\L2D\task_log.txt
python scripts\concurrent_sweep_runner.py >> C:\Users\sshuser\desktop\L2D\task_log.txt 2>&1
echo Python script finished with exit code %ERRORLEVEL% >> C:\Users\sshuser\desktop\L2D\task_log.txt

exit /b 0