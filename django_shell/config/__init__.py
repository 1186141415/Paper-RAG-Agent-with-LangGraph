import os

if os.getenv("MYSQL_HOST"):
    import pymysql

    pymysql.install_as_MySQLdb()
