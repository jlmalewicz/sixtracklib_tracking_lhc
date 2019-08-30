def strat(N_strat):
    strat   = np.linspace(sigma1 ** 2 / 2, sigma2 ** 2 / 2, N_strat+1)
    strat   = np.sqrt(2 * strat)
    low_lim = strat[:-1]
    up_lim  = strat[1:]
    
    J_norm  = J1_all/epsg_x + J2_all/epsg_y
    
    masks   = []
    for i in range(N_strat):
        mask = np.logical_and(low_lim[i] ** 2 / 2 < J_norm, 
                               up_lim[i] ** 2 / 2 >= J_norm)
        masks.append(mask)
    
    
    # from 0 to sigma1
    f1 = ((1 - np.exp(-sigma1**2/2) * (1 + sigma1**2/2)) /
          (1 - np.exp(-sigma2**2/2) * (1 + sigma2**2/2)))
    
    # from sigma1 to sigma2
    Q      = np.zeros([20000,])
    Q2     = np.zeros([20000,])
    Q_tot  = np.zeros([20000,])
    errorQ = np.zeros([20000,])
    
    turns = np.array([range(0, 20000000, 1000)]).reshape(20000,)
    
    for ii in range(N_strat):
        print('* * * slice %s out of %s * * *'%(ii+1, N_strat))
        mask = masks[ii]
        t0, J1, J2 = t0_all[mask], J1_all[mask], J2_all[mask]
        for i,t in enumerate(turns):
            Q[i] = integral(t, t0, J1/epsg_x, J2/epsg_y,low_lim[ii], up_lim[ii])
            Q2[i] = integral2(t, t0, J1/epsg_x, J2/epsg_y, low_lim[ii], up_lim[ii])
            Q_tot[i] += Q[i]
            if i % 2000 == 0:
                print(i/200, '%')
        errorQ += (Q2 - Q**2)/t0.shape[0]
    errorQ = np.sqrt(errorQ)
    return Q_tot, errorQ

QQ = []
errorQQ = []

boop = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]

for i, beep in enumerate(boop):
    strat1, strat2 = strat(beep)
    QQ.append(strat1)
    errorQQ.append(strat2)

QQ_init = []
for thing in QQ:
    QQ_init.append(thing[0])
errorQQ_init = []
for thing in errorQQ:
    errorQQ_init.append(thing[0])

QQ_last = []
for thing in QQ:
    QQ_last.append(thing[-1])
errorQQ_last = []
for thing in errorQQ:
    errorQQ_last.append(thing[-1])


######### error at t = 0
plt.errorbar(boop, np.abs(QQ_init + f1 - 1), yerr=errorQQ_init, marker='s', mfc='red', ms=20)
plt.ylabel(r'$|$integral$ - 1|$ at $t = 0$')
plt.xlabel('Number of slices')
plt.title('Error intervals and deviations from 1 w.r.t stratified sampling', loc = 'right')


######### rms pull at t = 0
plt.plot(boop, np.abs(QQ_init + f1 - 1)/errorQQ_init, marker='s', mfc='red', ms=20)
plt.ylabel(r'$|$integral$ - 1|/$error at $t = 0$')
plt.xlabel('Number of slices')
plt.title('Pull and deviations from 1 w.r.t stratified sampling', loc = 'right')


######## error at t = 2e7
plt.errorbar(boop, QQ_last + f1, yerr=errorQQ_last, marker='s', mfc='red', ms=20)
plt.ylabel(r'integral at $t = 2e7$')
plt.xlabel('Number of slices')
plt.title('Error intervals and deviations from 1 w.r.t stratified sampling', loc = 'right')

######## error wrt slice at t = 0 and t = 2e7

plt.plot(boop, errorQQ_init, 'b-', label = r'at $t = 0$')
plt.plot(boop, errorQQ_last, 'g-', label = r'at $t = 2e7$')

plt.xlabel('Number of slices')
plt.ylabel('Error intervals')
plt.title('Error intervals for stratified sampling', loc = 'right')
plt.legend(loc = 'upper right')






def intensity_plot(c):
    if c is None:
        c = 'blue'

    plt.title(r'Losses, stratified sampling, $\delta_{init}$ = %s'%(mydict['init_delta']), loc = 'right')
    plt.plot(turns, Q_tot+f1, c, label = '%s slices'%N_strat)
    plt.xlabel('Number of turns')
    plt.ylabel('Intensity')

#    errorfill(turns, Q_tot+f1, yerr=errorQ, color = c)

    plt.legend(loc = 'lower left')

    plt.show()

