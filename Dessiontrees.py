from sklearn import tree
import pandas as pd
import pydotplus
from IPython.display import Image
golf_df = pd.DataFrame()
#add outlook
golf_df['Outlook'] = ['sunny', 'sunny', 'overcast', 'rainy', 'rainy', 'rainy', 'overcast',
'sunny','sunny', 'rainy', 'sunny', 'overcast', 'overcast', 'rainy']
#add temperature
golf_df['Temperature'] = ['hot', 'hot', 'hot', 'mild', 'cool', 'cool', 'cool', 'mild', 'cool','mild','mild', 'mild', 'hot', 'mild']
#add humidity
golf_df['Humidity'] = ['high', 'high', 'high', 'high', 'normal', 'normal', 'normal', 'high','normal','normal', 'normal', 'high', 'normal', 'high']
#add windy
golf_df['Windy'] = ['false', 'true', 'false', 'false', 'false', 'true', 'true', 'false', 'false','false','true','true', 'false', 'true']
golf_df['Play'] = ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes','yes', 'no']
print(golf_df)
one_hot_data = pd.get_dummies(golf_df[ ['Outlook', 'Temperature', 'Humidity','Windy'] ])
print(one_hot_data)
clf = tree.DecisionTreeClassifier()
clf_train = clf.fit(one_hot_data, golf_df['Play'])
print(tree.export_graphviz(clf_train, None))
dot_data = tree.export_graphviz(clf_train, out_file=None,
feature_names=list(one_hot_data.columns.values),
class_names=['Not_Play', 'Play'],
rounded=True, filled=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
