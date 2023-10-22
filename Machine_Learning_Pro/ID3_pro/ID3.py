import pandas as pd
import numpy as np

#-------------Reading the dataset----------------------------------------------------------------
train_data_m = pd.read_csv("C:/Users/Ladan_Gh/PycharmProjects/pythonProject/Machine_Learning_Pro/ID3_pro/PlayTennis(train).csv")

# print(train_data_m.head())

#-------------1.Calculating the entropy of the whole dataset----------------------------------------
def calc_total_entropy(train_data, label, class_list):
    total_row = train_data.shape[0] # sample
    total_entr = 0

    for c in class_list: #class list = {yes, no}
        total_class_count = train_data[train_data[label] == c].shape[0]
        #print(total_class_count)
        total_class_entr = - (total_class_count / total_row) * np.log2(total_class_count / total_row)  #entropy of the class
        total_entr += total_class_entr  # adding the class entropy to the total entropy of the dataset

    return total_entr

# print(calc_total_entropy(train_data_m, 'Outlook', {'Yes', 'No'}))

#--------------2.Calculating the entropy for the filtered dataset-----------------------------------
def calc_entropy(feature_value_data, label, class_list):
    class_count = feature_value_data.shape[0]
    entropy = 0

    for c in class_list:
        label_class_count = feature_value_data[feature_value_data[label] == c].shape[0]  #row count of class c
        entropy_class = 0
        if label_class_count != 0:
            probability_class = label_class_count / class_count
            entropy_class = - probability_class * np.log2(probability_class)
        entropy += entropy_class
    return entropy

#--------------3.Calculating information gain for a feature------------------------------------------
def gain(feature_name, train_data, label, class_list):
    feature_value_list = train_data[feature_name].unique()
    total_row = train_data.shape[0]
    feature_info = 0.0

    for feature_value in feature_value_list:
        feature_value_data = train_data[train_data[feature_name] == feature_value]
        feature_value_count = feature_value_data.shape[0]
        feature_value_entropy = calc_entropy(feature_value_data, label, class_list)
        feature_value_probability = feature_value_count / total_row # total prob for feature_value_data
        feature_info += feature_value_probability * feature_value_entropy

    return calc_total_entropy(train_data, label, class_list) - feature_info

#-------------4.Finding the most informative feature--------------------------------------# Max
def find_most_informative_feature(train_data, label, class_list):
    feature_list = train_data.columns.drop(label) # drop all columns of Play Tennis

    max_info_gain = -1
    max_info_feature = None

    for feature in feature_list:
        feature_info_gain = gain(feature, train_data, label, class_list)
        if max_info_gain < feature_info_gain:
            max_info_gain = feature_info_gain
            max_info_feature = feature

    return max_info_feature

#---------------5.Adding a node to the tree-------------------------------------------------------
def generate_sub_tree(feature_name, train_data, label, class_list):
    feature_value_count_dict = train_data[feature_name].value_counts(sort=False)
    tree = {}

    for feature_value, count in feature_value_count_dict.iteritems():
        feature_value_data = train_data[train_data[feature_name] == feature_value]

        assigned_to_node = False
        for c in class_list:
            class_count = feature_value_data[feature_value_data[label] == c].shape[0]  #count of class c

            if class_count == count: # Pure
                tree[feature_value] = c  #adding node
                train_data = train_data[train_data[feature_name] != feature_value]  #removing rows with feature_value
                assigned_to_node = True

        if not assigned_to_node:  # Not pure class
            tree[feature_value] = "?"  # as feature_value is not a pure class, it should be expanded further, so the branch is marking with ?

    return tree, train_data

#-----------6.generating Tree----------------------------------------------
def make_tree(root, prev_feature_value, train_data, label, class_list):
    if train_data.shape[0] != 0:  #if dataset becomes empty after updating
        max_info_feature = find_most_informative_feature(train_data, label, class_list)  #most informative feature
        tree, train_data = generate_sub_tree(max_info_feature, train_data, label, class_list)
        next_root = None

        if prev_feature_value != None:  #add to intermediate node of the tree
            root[prev_feature_value] = dict()
            root[prev_feature_value][max_info_feature] = tree
            next_root = root[prev_feature_value][max_info_feature]
        else:  #add to root of the tree
            root[max_info_feature] = tree
            next_root = root[max_info_feature]

        for node, branch in list(next_root.items()):
            if branch == "?":  #if it is expandable
                feature_value_data = train_data[train_data[max_info_feature] == node]  #using the updated dataset
                make_tree(next_root, node, feature_value_data, label, class_list)

#-------------7.Finding unique classes of the label and Starting the algorithm--------------------
def id3(train_data_m, label):
    train_data = train_data_m.copy()
    tree = {} #tree which will be updated
    class_list = train_data[label].unique()
    make_tree(tree, None, train_data, label, class_list)

    return tree

#--------------------8.Predicting from the tree---------------------------------------------------
def predict(tree, instance):
    if not isinstance(tree, dict): #if it is leaf node
        return tree #return the value
    else:
        root_node = next(iter(tree)) #getting first key/feature name of the dictionary
        feature_value = instance[root_node]
        if feature_value in tree[root_node]:
            return predict(tree[root_node][feature_value], instance) #goto next feature
        else:
            return None

#---------------Evaluating test dataset----------------------------------------------------------
def evaluate(tree, test_data_m, label):
    correct_preditct = 0
    wrong_preditct = 0
    for index, row in test_data_m.iterrows():
        result = predict(tree, test_data_m.iloc[index])
        if result == test_data_m[label].iloc[index]: #predicted value and expected value is same or not
            correct_preditct += 1
        else:
            wrong_preditct += 1
    accuracy = correct_preditct / (correct_preditct + wrong_preditct) #calculating accuracy
    return accuracy

#---------------9.Checking test dataset and Evaluating it--------------------------------------------
# tree = id3(train_data_m, 'Outlook')
tree = id3(train_data_m, 'Play Tennis')
print('Tree is :' ,tree)

test_data_m = pd.read_csv("C:/Users/Ladan_Gh/PycharmProjects/pythonProject/Machine_Learning_Pro/ID3_pro/PlayTennis(test).csv")

accuracy = evaluate(tree, test_data_m, 'Play Tennis') #evaluating the tree

print('Accuracy of the tree is :',accuracy)
