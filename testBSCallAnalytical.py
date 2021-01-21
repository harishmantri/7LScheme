from BSCallAnalytical import BSCallAnalytical
import time

start_time= time.time()
callOption = BSCallAnalytical(100, 50, 1, 0.5, 0.2)
assert round(callOption.price(),2) == 69.67
print(callOption.price())
print("--- %s second ---" % (time.time() - start_time))