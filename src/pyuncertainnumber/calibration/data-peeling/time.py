from datetime import datetime
import time
import calendar 

if __name__ == '__main__':
    print(time.time())
    print(datetime.timestamp(datetime.now())) # Unix time
    print(time.gmtime())
    print(calendar.timegm(time.gmtime())) # Unix time, see https://en.wikipedia.org/wiki/Unix_time