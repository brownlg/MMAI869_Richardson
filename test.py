import Richardson_Logger

from Richardson_Logger import r_logger
my_logger = r_logger.R_logger("testcsvdata.csv")

my_logger.write_line("data,12,324,22\n")
my_logger.write_line("data\n")
my_logger.write_line("data\n")
my_logger.write_line("data\n")


print('Diana & Eman')