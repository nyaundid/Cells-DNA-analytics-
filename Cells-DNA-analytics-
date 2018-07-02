
# coding: utf-8

# In[75]:


import matplotlib as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
import pandas as pd
from pylab import rcParams
import seaborn as sb
from collections import Counter
import statsmodels.formula.api as smapi
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
import statsmodels.api as sm
from sklearn import preprocessing
import matplotlib as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
import pandas as pd
from pylab import rcParams
import seaborn as sb
from collections import Counter
import statsmodels.formula.api as smapi
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
import statsmodels.api as sm
from sklearn import preprocessing
from sklearn import preprocessing
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import lasso_path, lars_path, Lasso, enet_path


# In[76]:



dataset = "Documents\\CellDNA1.csv"
DNA = pd.read_csv(dataset)
dataset = np.genfromtxt("Documents\\CellDNA1.csv", delimiter = ',')


# In[77]:


DNA.head()


# In[78]:


DNA.shape


# In[79]:


DNA.loc[:, ["222","31.18918919", "40.34234234", "35.57908668","8.883916969","0.968324558", "-80.11367302", "222.1","1","16.81247093","0.816176471", "0.578125", "78.591", "0"]]


# In[80]:


sb.pairplot(DNA)


# In[81]:


df = pd.read_csv("Documents\\CellDNA1.csv",sep=",")


# In[82]:


X_scaled = preprocessing.scale(DNA)


# In[83]:


X_scaled


# In[84]:


from scipy.stats import zscore
nz = DNA.apply(zscore)


# In[85]:


nz


# In[86]:


X1= X_train = DNA.loc[:, ["222","31.18918919", "40.34234234", "35.57908668","8.883916969","0.968324558", "-80.11367302", "222.1","1","16.81247093","0.816176471", "0.578125", "78.591"]]


# In[87]:


X1


# In[88]:


from scipy.stats import zscore
XT = X1.apply(zscore)


# In[89]:


XT


# In[90]:


X_scaled = preprocessing.scale(X_train)


# In[91]:


print(X_scaled, '\n')
print(X_scaled.mean(axis=0), X_scaled.std(axis=0))


# In[92]:


from scipy.stats import zscore
Y1 = X_train.apply(zscore)


# In[93]:


Y1


# In[94]:


Y = DNA.loc[:, ['0']]


# In[95]:


Y


# In[96]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import lasso_path, lars_path, Lasso, enet_path


# In[97]:


X = X1 
Y = Y



# In[98]:


Y = dataset[:,2];
X = dataset[:,:13];  
X[np.isnan(X)] = 0 


# In[99]:


print("Regularization path using lars_path")    


# In[100]:


alphas1, active1, coefs1= lars_path(X, Y, method='lasso', verbose=True)




# In[101]:


print("Regularization path using lars_path")  


# In[102]:


eps= 5e-6


# In[103]:


alphas2, coefs2, _= lasso_path(X, Y, eps)      


# In[104]:


print("ONE regularization using Lasso")


# In[105]:


clf = Lasso(fit_intercept=False, alpha=1.3128)


# In[106]:


clf.fit(X, Y)


# In[107]:


print(clf.intercept_, clf.coef_)


# In[108]:


clf.fit(X, Y)
print(clf.intercept_, clf.coef_)


# In[2]:



#log_alphas= -np.log10(model.alphas_)
#ax = plt.gca()
plt.plot(model.alphas_, model.coef_path_.T)
#plt.axvline(log_alphas, linestyle='--', color='k', label='alpha CV')


plt.xlabel('alpha')
plt.ylabel('Coefficients')
plt.title('LASSO Path')
plt.axis('tight')
plt.show()

fig, ax = plt.subplots(figsize=(30,25))
xx = np.sum(np.abs(coefs1.T), axis=1)
plt.plot(xx, coefs1.T)
ymin, ymax = plt.ylim()
plt.vlines(xx, ymin, ymax, linestyle='dashed',  label='alpha CV')

plt.xlabel('|coef| / max|coef|')
plt.ylabel('Coefficients')
plt.title('LARS Path')

plt.legend()





# In[71]:


#log_alphas= -np.log10(model.alphas_)
#ax = plt.gca()
plt.plot(model.alphas_, model.coef_path_.T)
#plt.axvline(log_alphas, linestyle='--', color='k', label='alpha CV')
xx = np.sum(np.abs(coefs1.T), axis=1)

plt.plot(xx, coefs1.T)
ymin, ymax = plt.ylim()
plt.vlines(xx, ymin, ymax, linestyle='dashed',  label='alpha CV')

plt.xlabel('alpha')
plt.ylabel('Coefficients')
plt.title('LASSO Path')
plt.axis('tight')
plt.show()



# In[63]:


xx


# In[70]:


#alphas,coefs, dual_gaps=model.path(X_train, y)
#coefs=coefs.reshape(11,100)
#result=pd.DataFrame(coefs.T,index=alphas,columns=X_train.columns.values.tolist())
#result.head()
xx = np.sum(np.abs( model.coef_path_.T ), axis=1)
plt.plot(xx, model.coef_path_.T)
plt.plot(xx, model.coef_path_.T)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('weights')
ymin, ymax = plt.ylim()
#plt.vlines(alphas, ymin, ymax, linestyle='dashed')
plt.xlabel('|coef| / max|coef|')
plt.ylabel('Coefficients')
plt.title('LASSO Path')
plt.axis('tight')
plt.show()


# In[65]:


from sklearn.linear_model import Lasso
for aa in np.arange(0,1.1, 0.01):
    clf = Lasso(alpha = aa)
    clf.fit(X, Y)
    print(aa, clf.coef_, clf.intercept_)


# In[66]:


from sklearn.model_selection import cross_val_score


from sklearn.cross_validation import cross_val_score



clf = linear_model.Lasso()
scores = cross_val_score(clf, X, Y, cv=10)


# In[67]:


scores


# In[68]:


scores.mean()


# In[69]:


scores.std()

