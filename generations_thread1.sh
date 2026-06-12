#python generation_examination.py -e cartpole -a lambda -c novelty -o "fourth_try" -R grid_search &
#python generation_examination.py -e cartpole -a diff -c novelty -o "fourth_try" -R grid_search
# python generation_examination.py -e cartpole -a lambda -c fitness -o "fourth_try" -R grid_search &
# python generation_examination.py -e cartpole -a diff -c fitness -o "fourth_try" -R grid_search
# python generation_examination.py -e cartpole -a lambda -c add_novelty -o "fourth_try" -R grid_search &
# python generation_examination.py -e cartpole -a diff -c add_novelty -o "fourth_try" -R grid_search
# python generation_examination.py -e cartpole -a lambda -c fit_archiving -o "fourth_try" -R grid_search &
# python generation_examination.py -e cartpole -a diff -c fit_archiving -o "fourth_try" -R grid_search
# python generation_examination.py -e cartpole -a lambda -c novelty_archiving -o "fourth_try" -R grid_search &
# python generation_examination.py -e cartpole -a diff -c novelty_archiving -o "fourth_try" -R grid_search
# python generation_examination.py -e lunarlander -a lambda -c novelty_archiving -o "fourth_try" -R grid_search 
# python generation_examination.py -e lunarlander -a diff -c novelty_archiving -o "fourth_try" -R grid_search
# python generation_examination.py -e lunarlander -a lambda -c fit_archiving -o "fourth_try" -R grid_search 
# python generation_examination.py -e lunarlander -a diff -c fit_archiving -o "fourth_try" -R grid_search
# python generation_examination.py -e lunarlander -a lambda -c fitness -o "fourth_try" -R grid_search 
# python generation_examination.py -e lunarlander -a diff -c fitness -o "fourth_try" -R grid_search
# python generation_examination.py -e lunarlander -a lambda -c add_novelty -o "fourth_try" -R grid_search 
# python generation_examination.py -e lunarlander -a diff -c add_novelty -o "fourth_try" -R grid_search
python generation_examination.py -e lunarlander -a lambda -c fitness -o "mean_try" -R optuna 
#python generation_examination.py -e lunarlander -a diff -c novelty -o "fourth_try" -R grid_search