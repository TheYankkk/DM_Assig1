#print("test")
import pandas as pd
import math
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
data=pd.read_csv(r"mammographic_masses.csv",delimiter=",",header=0)
#print(data)
d=data.values.tolist()
for each in range(len(d)):
    d[each][0]="BI-RADS:"+d[each][0]
    d[each][1] = "Age:" + d[each][1]
    d[each][2] = "Shape:" + d[each][2]
    d[each][3] = "Margin:" + d[each][3]
    d[each][4] = "Density:" + d[each][4]
    d[each][-1]="Severity:"+str(d[each][-1])
#print(d)
te = TransactionEncoder()
te_ary = te.fit(d).transform(d)
#print(te.columns_)
df = pd.DataFrame(te_ary,columns=te.columns_)
#print(df)

#computing frequent itemsets and association rules
frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)

a=association_rules(frequent_itemsets, metric="confidence", min_threshold=0.9)

#visualizing association rules results
print(a[["antecedents","consequents","support","confidence"]])
dnew=[]
for each in range(len(d)):
    dnew.append(d[each][1:])
#print(dnew)
te_new = TransactionEncoder()
te_ary_new = te_new.fit(dnew).transform(dnew)
#print(te.columns_)
df_new = pd.DataFrame(te_ary_new,columns=te_new.columns_)
print(df_new)
frequent_itemsets_new = apriori(df_new, min_support=0.1, use_colnames=True)
#print(frequent_itemsets_new)
a=association_rules(frequent_itemsets_new, metric="confidence", min_threshold=0.9)
print(a[["antecedents","consequents","support","confidence"]])