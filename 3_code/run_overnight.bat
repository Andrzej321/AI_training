@echo off
python 3_codes\training_RNN_fast.py || goto :end
python 3_codes\training_LSTM_fast.py || goto :end
python 3_codes\training_GRU_fast.py || goto :end

:end
