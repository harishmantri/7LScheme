from EulerCallMonteCarlo import EulerCallMonteCarlo
import time

start_time= time.time()
callOption = EulerCallMonteCarlo(100, 50, 1, 0.5, 0.2,100000)
print(callOption.price())
print("--- %s second ---" % (time.time() - start_time))
