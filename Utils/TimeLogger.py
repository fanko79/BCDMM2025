import datetime
import os
from Params import args

logmsg = ''
timemark = dict()
saveDefault = False
log_filename = None
if not os.path.exists('log'):
    os.makedirs('log')

def log(msg, save=None, oneline=False):
    global logmsg
    global saveDefault
    global log_filename

    time = datetime.datetime.now()
    tem = '%s: %s' % (time, msg)

    if save is not None:
        if save:
            logmsg += tem + '\n'
    elif saveDefault:
        logmsg += tem + '\n'

    if oneline:
        print(tem, end='\r')
    else:
        print(tem)

    if log_filename is None:
        log_filename = os.path.join('log', f"{args.data}_{time.strftime('%Y%m%d_%H%M%S')}.txt")

    with open(log_filename, 'a') as logfile:
        logfile.write(tem + '\n')

def marktime(marker):
    global timemark
    timemark[marker] = datetime.datetime.now()

if __name__ == '__main__':
    log('Log initialization complete', save=True)
    log('This is the second log message.', save=True)
    log('Another log message to track the process.')

