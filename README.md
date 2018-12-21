##Python in-database

This project show how to use Python inside SQL Server 2017 with ‘Machine Learning Services’.

To enable Python in SQL Server 2017 first you need to install ‘Machine Learning Services’.
Python is installed in a folder called PYTHON_SERVICES (typically located at C:\Program Files\Microsoft SQL Server\MSSQL14.MSSQLSERVER\) where you should find a file python.exe.

To enable external scripts execution run the following query:
'EXEC sp_configure  'external scripts enabled', 1
RECONFIGURE WITH OVERRIDE'

Now you can execute simple python code with below query:
'EXEC sp_execute_external_script  @language =N'Python',
@script=N'
OutputDataSet = InputDataSet;
',
@input_data_1 = N'SELECT 1 AS hello'
WITH RESULT SETS (([hello] int not null));
GO'

You should acquire single value: hello 1.


