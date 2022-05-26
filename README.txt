Please follow the tutorial. 
The PDF version is included in this repository.

Batch speed comparison:
1) DR API option as outlined in the tutorial - 2000 records = 5 seconds
2) DR PRIME option with 46 rules (see source code) - 2000 records = 3 seconds

Real-time scoring (1 record each):
Using SQLQueryStress (https://www.brentozar.com/archive/2015/05/how-to-fake-load-tests-with-sqlquerystress/) 
from a client against MS SQL Server 2019 hosted on EC2 t2.xlarge

Option 1 - DR API with one prediction server (4 cores):
1 thread: 0.89s

10 concurrent threads:
Avg per thread: 3.90s
Total elapsed time: 6.18s

20 concurrent threads:
Avg per thread: 5.73s
Total elapsed time: 11.81s

30 concurrent threads: 
Avg per thread: 8.20s
Total Elapsed: 17.76s

Option 2 - DR Prime:
1 thread: 0.81s

10 concurrent threads:
Avg per thread: 3.06s
Total elapsed time: 6.32s

20 concurrent threads:
Avg per thread: 5.06s
Total elapsed time: 11s

30 concurrent threads: 
Avg per thread: 6.38s
Total Elapsed: 16.98s

API performance could be further improved by scaling the prediction server instance.

