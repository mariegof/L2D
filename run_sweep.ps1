cd C:\Users\sshuser\desktop\L2D
# Activate the conda environment and run the script
& 'C:\Users\sshuser\anaconda3\Scripts\activate.bat' 'l2d'
python scripts/concurrent_sweep_runner.py > 'C:\Users\sshuser\desktop\L2D\sweep_log.txt' 2>&1
