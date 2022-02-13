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
    d[each][-1]=str(d[each][-1])
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