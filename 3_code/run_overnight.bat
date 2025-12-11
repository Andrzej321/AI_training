@echo off

REM 1. Initialize conda for this shell
CALL C:\Users\ROG\miniconda\Scripts\activate.bat

REM 2. Activate your env by name
CALL conda activate pytorch200_cuda118

REM 3. (optional) verify it's correct
python -c "import sys; print(sys.executable)"

REM 4. Now run your code
python training_TCN_fast.py
python training_RNN_fast.py
