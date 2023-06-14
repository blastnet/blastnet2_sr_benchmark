#written by W.T. Chung
#finds port address for master node
from socket import socket
import os
import platform 
import csv


if __name__ == '__main__':
    if os.environ.get('OMPI_COMM_WORLD_RANK') == '0':
        with socket() as s:
            hostname = platform.node() 
            s.bind(('',0))
            master_port  = s.getsockname()[1]
        print(hostname)
        print(master_port)
        output = [str(hostname),str(master_port)]
        job = os.environ.get('LSB_JOBID')
        with open("../"+job+".address_port.csv", 'w') as f:
            write = csv.writer(f)   
            write.writerow(output) 
