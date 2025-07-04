import subprocess
import numpy as np
import cProfile
import time


def main():
    Vp = [1]
    Icut_values = [1.5]
    RF_values = [10]
    Gon_values = [100] 
    run_num = 0
  
    noise_stddev = [0]
    cur_factor = [0.001]
    for valuep in Vp:
        for value1 in Gon_values:
            for value2 in Icut_values:
                for value3 in RF_values:
                    for value4 in noise_stddev:
                        for value5 in cur_factor:
                    #print("Icut",value2,  "RF",value3)         # "Gon",value1,
                    
                            command = ["python", "main_points_duplicate_modf.py", "--Vp", str(valuep),"--Gon",str(value1),"--Icut", str(value2), "--RF", str(value3),"--noise_stddev",str(value4),"--cur_factor",str(value5)]        #"--Gon", str(value1),
                            print(f"\n➡️  Running command: {' '.join(command)}")
                            start_time = time.time()
                            try:
                                subprocess.run(command, check=True)
                                end_time = time.time()
                                run_num += 1
                                print(f"✅ Finished run {run_num} | Time: {round(end_time - start_time, 2)}s")
                            except subprocess.CalledProcessError as e:
                                print(f"\n❌ Error in run {run_num+1}: Command failed.")
                                print(f"Return code: {e.returncode}")
                                print(f"Command: {' '.join(e.cmd)}")
                                break  # or continue to skip and keep going
if __name__ == "__main__":
    cProfile.run("main()",sort = "cumulative") 

