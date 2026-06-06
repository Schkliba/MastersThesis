python optuna_experiment.py -e lunarlander -a diff -c fit_archiving -n "diff_fit_archiving_no_repeat" &
python optuna_experiment.py -e lunarlander -a lambda -c fit_archiving -n "lambda_fit_archiving_no_repeat" &
python optuna_experiment.py -e lunarlander -a diff -c novelty_archiving -n "diff_novelty_archiving_no_repeat" &
python optuna_experiment.py -e lunarlander -a lambda -c novelty_archiving -n "lambda_novelty_archiving_no_repeat" &
wait