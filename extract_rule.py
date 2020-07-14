#coding:utf-8
import cPickle
import numpy as np
import warnings

warnings.filterwarnings("ignore")

#read cloth text description
def loadtxt(path):
    f = open(path,'r')
    list = []
    for line in f:
        list.append(line.strip())
    f.close()
    return list

def extract(rules=[["coat","dress",1]]):
    print 'loading train data'
    train_id = np.loadtxt(open("./id/train_ijk_shuffled_811.txt"), delimiter="\t",
                          skiprows=0)
    train_id = np.array(train_id)
    train_set_i = np.array(train_id[:, 0])
    train_set_j = np.array(train_id[:, 1])
    train_set_k = np.array(train_id[:, 2])

    print 'loading valid data'
    valid_id = np.loadtxt(open("./id/valid_ijk_shuffled_811.txt"), delimiter="\t",
                          skiprows=0)
    valid_id = np.array(valid_id)
    valid_set_i = np.array(valid_id[:, 0])
    valid_set_j = np.array(valid_id[:, 1])
    valid_set_k = np.array(valid_id[:, 2])

    print 'loading test data'
    test_id = np.loadtxt(open("./id/test_ijk_shuffled_811.txt"), delimiter="\t",
                          skiprows=0)
    test_id = np.array(test_id)
    test_set_i = np.array(test_id[:, 0])
    test_set_j = np.array(test_id[:, 1])
    test_set_k = np.array(test_id[:, 2])


    train_rules_ind = []
    valid_rules_ind = []
    test_rules_ind = []
    for rule in rules:
        print "----------extracting rules %s----------" % rule
        if rule[-1] == 1:
            train_rule_ind = extract_ind(train_set_i, train_set_j, train_set_k,rule)
            valid_rule_ind = extract_ind(valid_set_i, valid_set_j, valid_set_k,rule)
            test_rule_ind = extract_ind(test_set_i, test_set_j, test_set_k,rule)
        elif rule[-1] == 0:
            train_rule_ind = extract_ind_no(train_set_i, train_set_j, train_set_k, rule)
            valid_rule_ind = extract_ind_no(valid_set_i, valid_set_j, valid_set_k, rule)
            test_rule_ind = extract_ind_no(test_set_i, test_set_j, test_set_k, rule)
        train_rules_ind.append(train_rule_ind)
        valid_rules_ind.append(valid_rule_ind)
        test_rules_ind.append(test_rule_ind)
        print "extract rule done!"


    print 'writing...'
    write_file = open('./rule_ind/train_rules_ind.pkl', 'wb')
    cPickle.dump([train_rules_ind], write_file)
    write_file.close()

    write_file = open('./rule_ind/valid_rules_ind.pkl', 'wb')
    cPickle.dump([valid_rules_ind], write_file)
    write_file.close()

    write_file = open('./rule_ind/test_rules_ind.pkl', 'wb')
    cPickle.dump([test_rules_ind], write_file)
    write_file.close()
    print 'writing done!'


def contain(attribute,str):
    attribute_list = attribute.split("/")
    for item in attribute_list:
        if " "+item+" " in " "+str+" ":
        #if item in str:
            return True
    return False


def extract_ind(input1, input2, input3,rule):
    print 'loading top/cloth text/id....'
    top_text_list = np.loadtxt(open("./data/top_id.txt", "r"),delimiter="\n", skiprows=0)
    top_text_list = np.array(top_text_list)

    #print 'loading top/cloth text/id'
    top_text = loadtxt("./data/top_text.txt")
    top_text = np.array(top_text)
    #print(len(top_text))

    #print 'loading bottom text id'
    bottom_text_list = np.loadtxt(open("./data/bottom_id.txt", "r"), delimiter="\n", skiprows=0)
    bottom_text_list = np.array(bottom_text_list)

    #print 'loading bottom text'
    bottom_text = loadtxt("./data/bottom_text.txt")
    bottom_text = np.array(bottom_text)

    #print 'extract ind'
    #print input1.shape[0]
    ind = []
    count1 = 0
    count2 = 0
    for i in range(input1.shape[0]):
        i_id = input1[i]
        j_id = input2[i]
        k_id = input3[i]
        i_index = np.where(top_text_list == i_id)[0]
        j_index = np.where(bottom_text_list == j_id)[0]
        k_index = np.where(bottom_text_list == k_id)[0]
        #print(i_index)
        i_text = top_text[i_index]
        j_text = bottom_text[j_index]
        k_text = bottom_text[k_index]

        if contain(rule[0],i_text[0]):
            if contain(rule[1],j_text[0]):
                if contain(rule[1],k_text[0]):
                    ind.append([0,0,0])
                else:
                    ind.append([1,1,0])
                    count1 += 1
            elif contain(rule[1],k_text[0]):
                ind.append([1,0,1])
                count2 += 1
            else:
                ind.append([0,0,0])
        else:
            ind.append([0,0,0])
    ind = np.array(ind)
    print ind.shape[0]
    print 'num of 110: %i' % count1
    print 'num of 101: %i' % count2
    return ind

def extract_ind_no(input1, input2, input3,rule):
    print 'loading top/cloth text/id....'
    top_text_list = np.loadtxt(open("./data/top_id.txt", "r"),delimiter="\n", skiprows=0)
    top_text_list = np.array(top_text_list)

    #print 'loading top/cloth text/id'
    top_text = loadtxt("./data/top_text.txt")
    top_text = np.array(top_text)

    #print 'loading bottom text id'
    bottom_text_list = np.loadtxt(open("./data/bottom_id.txt", "r"), delimiter="\n", skiprows=0)
    bottom_text_list = np.array(bottom_text_list)

    #print 'loading bottom text'
    bottom_text = loadtxt("./data/bottom_text.txt")
    bottom_text = np.array(bottom_text)

    #print 'extract ind'
    #print input1.shape[0]
    ind = []
    count1 = 0
    count2 = 0
    for i in range(input1.shape[0]):
        i_id = input1[i]
        j_id = input2[i]
        k_id = input3[i]
        i_index = np.where(top_text_list == i_id)
        j_index = np.where(bottom_text_list == j_id)
        k_index = np.where(bottom_text_list == k_id)
        i_text = top_text[i_index]
        j_text = bottom_text[j_index]
        k_text = bottom_text[k_index]

        if contain(rule[0],i_text[0]):
            if contain(rule[1],j_text[0]):
                if contain(rule[1],k_text[0]):
                    ind.append([0,0,0])
                else:
                    ind.append([1,0,1])
                    count2 += 1
            elif contain(rule[1],k_text[0]):
                ind.append([1,1,0])
                count1 += 1
            else:
                ind.append([0,0,0])
        else:
            ind.append([0,0,0])
    ind = np.array(ind)
    print ind.shape[0]
    print 'num of 110: %i' % count1
    print 'num of 101: %i' % count2
    return ind

'''
rules = [["coat/coats","dress/dresses",1]]
extract(rules)
'''
