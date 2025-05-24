import datetime
import os
from Params import args

logmsg = ''
timemark = dict()
saveDefault = False
log_filename = None  # 用于记录日志文件名

# 确保日志文件夹存在
if not os.path.exists('log'):
    os.makedirs('log')

def log(msg, save=None, oneline=False):
    global logmsg
    global saveDefault
    global log_filename  # 全局日志文件名变量
    
    # 获取当前时间
    time = datetime.datetime.now()
    tem = '%s: %s' % (time, msg)
    
    # 如果需要保存日志信息
    if save is not None:
        if save:
            logmsg += tem + '\n'
    elif saveDefault:
        logmsg += tem + '\n'
    
    # 打印日志到控制台
    if oneline:
        print(tem, end='\r')
    else:
        print(tem)
    
    # 如果是第一次调用 log()，初始化日志文件名
    if log_filename is None:
        log_filename = os.path.join('log', f"{args.data}_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    
    # 将日志消息追加到日志文件中
    with open(log_filename, 'a') as logfile:
        logfile.write(tem + '\n')

def marktime(marker):
    global timemark
    timemark[marker] = datetime.datetime.now()

if __name__ == '__main__':
    log('Log initialization complete', save=True)
    log('This is the second log message.', save=True)
    log('Another log message to track the process.')

