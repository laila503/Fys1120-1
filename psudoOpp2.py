timesWhenZeroDeg = [0.0]

for i in range(1,r[:,0].size):
    theta = arctan2(r[i,1],r[i,0])

    if theta > 0 and arctan2(r[i-1,1],r[i-1,0]) < 0:
        timesWhenZeroDeg.append(t[i])
