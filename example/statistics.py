import numpy as np

size = 1000
top  = 100

x = np.array([np.random.randint(0,top) for i in range(size)])
y = np.array([np.random.randint(0,top) for i in range(size)])
n = len(x)

xsum, var, cov = 0, 0, 0
xmean, ymean   = np.mean(x), np.mean(y)

for i in range(n):
    xsum += x[i]
    var  += (x[i] - xmean)**2
    cov  += (x[i] - xmean)*(y[i] - ymean)
    
xsum /= n
var  /= n
cov  /= (n-1)
std   = var**0.5
coxy  = cov/(np.std(x) * np.std(y))

xt = sorted(x)
if (n % 2 == 1): xmed = xt[n//2 + 1]
else:            xmed = (xt[n//2] + xt[n//2-1])/2

npcov = np.cov(x,y)[0][1]         #  np.cov(x,y) = [[(x,x) (x,y)],[(y,x) (y,y)]]
corr  = np.corrcoef(x,y)[0][1]
corrs = np.corrcoef(sorted(x),sorted(y))[0][1]

print(' Property     numpy  >< naive ')
print(' -----------------------------')
print(' Mean       = {:6.2f} >< {:6.2f}'.format(np.mean(x)  ,xsum))
print(' Median     = {:6.2f} >< {:6.2f}'.format(np.median(x),xmed))
print(' Std        = {:6.2f} >< {:6.2f}'.format(np.std(x)   ,std ))
print(' Var        = {:6.2f} >< {:6.2f}'.format(np.var(x)   ,var ))
print(' Cov_xy     = {:6.2f} >< {:6.2f}'.format(npcov       ,cov ))
print(' Cor_xy     = {:6.2f} >< {:6.2f}'.format(coxy        ,corr))
print(' Cor_sorted = {:6.2f} >< {:6.2f}'.format(corrs       ,corr))