import numpy as np


def get_limit_centrality(fname="AuAu200blist.dat",cent=1):
    data = np.loadtxt(fname,dtype = float)
    data = data[data.argsort()]
    nindex = int((len(data)-1)*cent/100)
    return data[nindex] 

if __name__ == "__main__":
    clist=[]
    fin = "AuAu19blist.dat"
    fout = "AuAu19centrality.dat"
    for i in range(0,101,1):
        b=get_limit_centrality(fname=fin,cent=i)
        clist.append([i,b])
    np.savetxt(fout,np.array(clist))
    

