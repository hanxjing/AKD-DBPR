#coding:utf-8

import cPickle
import numpy
import random
from datetime import *

#load text description
def loadtxt(path):
    f = open(path,'r')
    list = []
    for line in f:
        list.append(line.strip())
    f.close()
    return list

#load data
print('loading top text....')
print('now():'+str( datetime.now()))
top_text = loadtxt(r'./data/top_text.txt')
top_text_r = loadtxt(r'./data/top_text_for_rule.txt')


print('loading bottom text....')
print('now():'+str( datetime.now()))
bottom_text = loadtxt(r'./data/bottom_text.txt')
bottom_text_r = loadtxt(r'./data/bottom_text_for_rule.txt')

print('top_text size: '+str(len(top_text)))
print('bottom_text size: '+str(len(bottom_text)))


print('loading top-bottom lists....')
print('now():'+str( datetime.now()))
top_list=numpy.loadtxt(open("./data/top_id.txt","r"),delimiter="\t",skiprows=0)
bottom_list = numpy.loadtxt(open("./data/bottom_id.txt","r"),delimiter="\t",skiprows=0)
top_bottom_list = numpy.loadtxt(open(r"./data/unique_newdataset_top_bottom_valid_pairs.txt","r"),delimiter="\t",skiprows=0)


'''
for i in range(5):
    print(bottom_list[i][0]+"\t"+bottom_list[i][1])
'''

#Get the triplets(top_i,bottom_j,bottom_k)
K=3;
olivettifaces1=[]
olivettifaces2=[]
olivettifaces3=[]
olivettifaces1_r=[]
olivettifaces2_r=[]
olivettifaces3_r=[]

ijk=numpy.empty((len(top_bottom_list)*K,3))

f0=open(r'./ijk_data/unique_newdataset_top_bottom_valid_pairs_shufflelist.txt','w')
f1=open(r'./ijk_data/total_ijk_shuffled_811.txt','w')
f2=open(r'./ijk_data/total_ijk_811.txt','w')
f3=open(r'./ijk_data/train_ijk_shuffled_811.txt','w')
f4=open(r'./ijk_data/valid_ijk_shuffled_811.txt','w')
f5=open(r'./ijk_data/test_ijk_shuffled_811.txt','w')

train_end=int(0.8*len(top_bottom_list))
valid_end=int(0.9*len(top_bottom_list))
y = []
print 'train_end'
print train_end
print 'valid_end'
print valid_end

index_shuf=range(len(top_bottom_list))
xxx=random.shuffle(index_shuf)
new_top_bottom_list=numpy.empty((len(top_bottom_list),2))

print 'generating new top bottom list'
count=0;
for i in index_shuf:

    new_top_bottom_list[count,0]=top_bottom_list[i,0]
    new_top_bottom_list[count,1]=top_bottom_list[i,1]
    count=count+1
    f0.write(str(top_bottom_list[i,0])+"\t"+str(top_bottom_list[i,1])+"\n");


print 'generating ijk pairs'
print('now():'+str( datetime.now() ))

for i in range(0, len(new_top_bottom_list)):
    if i%2000==0:
        print ('the '+str(i)+' top-bottom pair')

    tid=new_top_bottom_list[i,0]
    bid=new_top_bottom_list[i,1]

    top_index= numpy.where(top_list==tid)[0][0]
    bottom_index= numpy.where(bottom_list==bid)[0][0]

    #if top_list[top_index] == tid :
    #    print'yes'

    top_index_text=top_text[top_index]
    bottom_index_text=bottom_text[bottom_index]

    top_index_text_r = top_text_r[top_index]
    bottom_index_text_r = bottom_text_r[bottom_index]

    r1_list=random.sample(bottom_list,10);

    neg_num=0;
    for r1 in r1_list:
        if neg_num<K:
             check_pair= numpy.where(bottom_list==r1)[0]
             inde= numpy.where(top_bottom_list[:,1]==r1)

             inde1=numpy.where(top_bottom_list[inde[0],0]==tid)


             if len(inde1[0])>0:
                 print 'the pair'+str(tid)+",  "+str(r1)+" has been paired"

             else:
         #        print 'not find the pair'+str(tid)+",  "+str(r1)
                 if  r1<> bid:
        #             print 'get the '+str(neg_num)+' neg,,,,,'
                    neg_index= numpy.where(bottom_list==r1)[0][0]

                     # if neg_index[0]>=1000:
                     #     neg_index=999;
                    if i < train_end:
                        neg_text= bottom_text[neg_index]
                        neg_text_r = bottom_text_r[neg_index]
                        olivettifaces1.append(top_index_text)
                        olivettifaces2.append(bottom_index_text)
                        olivettifaces3.append(neg_text)

                        olivettifaces1_r.append(top_index_text_r)
                        olivettifaces2_r.append(bottom_index_text_r)
                        olivettifaces3_r.append(neg_text_r)

                        ijk[i * K + neg_num, 0] = tid;
                        ijk[i * K + neg_num, 1] = bid;
                        ijk[i * K + neg_num, 2] = r1;
                        neg_num = neg_num + 1;

                    else:
                        if i%2 == 0 :
                            neg_text = bottom_text[neg_index]
                            neg_text_r = bottom_text_r[neg_index]
                            olivettifaces1.append(top_index_text)
                            olivettifaces2.append(bottom_index_text)
                            olivettifaces3.append(neg_text)

                            olivettifaces1_r.append(top_index_text_r)
                            olivettifaces2_r.append(bottom_index_text_r)
                            olivettifaces3_r.append(neg_text_r)

                            ijk[i * K + neg_num, 0] = tid;
                            ijk[i * K + neg_num, 1] = bid;
                            ijk[i * K + neg_num, 2] = r1;
                            y.append(1)
                            neg_num=neg_num+1;
                            f2.write(str(tid)+"\t"+str(bid)+"\t"+str(r1)+"\n")
                        else :
                            neg_text = bottom_text[neg_index]
                            neg_text_r = bottom_text_r[neg_index]
                            olivettifaces1.append(top_index_text)
                            olivettifaces2.append(bottom_index_text)
                            olivettifaces3.append(neg_text)

                            olivettifaces1_r.append(top_index_text_r)
                            olivettifaces2_r.append(bottom_index_text_r)
                            olivettifaces3_r.append(neg_text_r)

                            ijk[i * K + neg_num, 0] = tid;
                            ijk[i * K + neg_num, 1] = bid;
                            ijk[i * K + neg_num, 2] = r1;
                            y.append(1)
                            neg_num = neg_num + 1;
                            f2.write(str(tid) + "\t" + str(r1) + "\t" + str(bid) + "\n")


olivettifaces1_train=olivettifaces1[0:train_end*K];
olivettifaces2_train=olivettifaces2[0:train_end*K];
olivettifaces3_train=olivettifaces3[0:train_end*K];

olivettifaces1_train_r=olivettifaces1_r[0:train_end*K];
olivettifaces2_train_r=olivettifaces2_r[0:train_end*K];
olivettifaces3_train_r=olivettifaces3_r[0:train_end*K];

ijk_train=ijk[0:train_end*K,:];

print 'olivettifaces1_train length'
print len(olivettifaces1_train)

y = numpy.array(y)
olivettifaces1_valid=olivettifaces1[train_end*K:valid_end*K];
olivettifaces2_valid=olivettifaces2[train_end*K:valid_end*K];
olivettifaces3_valid=olivettifaces3[train_end*K:valid_end*K];

olivettifaces1_valid_r=olivettifaces1_r[train_end*K:valid_end*K];
olivettifaces2_valid_r=olivettifaces2_r[train_end*K:valid_end*K];
olivettifaces3_valid_r=olivettifaces3_r[train_end*K:valid_end*K];

y_valid=y[0:(valid_end*K - train_end*K)]
ijk_valid=ijk[train_end*K:valid_end*K,:];

print 'olivettifaces1_valid length'
print len(olivettifaces1_valid)


olivettifaces1_test=olivettifaces1[valid_end*K:len(olivettifaces1)];
olivettifaces2_test=olivettifaces2[valid_end*K:len(olivettifaces1)];
olivettifaces3_test=olivettifaces3[valid_end*K:len(olivettifaces1)];

olivettifaces1_test_r=olivettifaces1_r[valid_end*K:len(olivettifaces1)];
olivettifaces2_test_r=olivettifaces2_r[valid_end*K:len(olivettifaces1)];
olivettifaces3_test_r=olivettifaces3_r[valid_end*K:len(olivettifaces1)];

y_test=y[(valid_end*K - train_end*K):]
ijk_test=ijk[valid_end*K:len(olivettifaces1),:];

print 'olivettifaces1_test length'
print len(olivettifaces1_test)


olivettifaces1_shuf_train=[]
olivettifaces2_shuf_train=[]
olivettifaces3_shuf_train=[]
olivettifaces1_shuf_train_r=[]
olivettifaces2_shuf_train_r=[]
olivettifaces3_shuf_train_r=[]

olivettifaces1_shuf_valid=[]
olivettifaces2_shuf_valid=[]
olivettifaces3_shuf_valid=[]
olivettifaces1_shuf_valid_r=[]
olivettifaces2_shuf_valid_r=[]
olivettifaces3_shuf_valid_r=[]

olivettifaces1_shuf_test=[]
olivettifaces2_shuf_test=[]
olivettifaces3_shuf_test=[]
olivettifaces1_shuf_test_r=[]
olivettifaces2_shuf_test_r=[]
olivettifaces3_shuf_test_r=[]

y_valid_shuf=[]
y_test_shuf=[]
ijk_shuf=[]
ijk_shuf_valid=[]
ijk_shuf_train=[]
ijk_shuf_test=[]


#shuffle data
print 'process training'
print   ('now():'+str( datetime.now() ))
index_shuf_train=range(len(olivettifaces1_train))
xxx=random.shuffle(index_shuf_train)
count=0;
for i in index_shuf_train:
      #  print 'writing training...'+str(count)
    olivettifaces1_shuf_train.append(olivettifaces1_train[i]);
    olivettifaces2_shuf_train.append(olivettifaces2_train[i]);
    olivettifaces3_shuf_train.append(olivettifaces3_train[i]);

    olivettifaces1_shuf_train_r.append(olivettifaces1_train_r[i]);
    olivettifaces2_shuf_train_r.append(olivettifaces2_train_r[i]);
    olivettifaces3_shuf_train_r.append(olivettifaces3_train_r[i]);

    ijk_shuf_train.append(ijk_train[i])
    f3.write(str(ijk_train[i][0])+"\t"+str(ijk_train[i][1])+"\t"+str(ijk_train[i][2])+"\n");

print 'process valid'
print   ('now():'+str( datetime.now() ))
index_shuf_valid=range(len(olivettifaces1_valid))
xxx=random.shuffle(index_shuf_valid)
count=0;
for i in index_shuf_valid:
      #  print 'writing training...'+str(count)
    olivettifaces1_shuf_valid.append(olivettifaces1_valid[i]);
    olivettifaces2_shuf_valid.append(olivettifaces2_valid[i]);
    olivettifaces3_shuf_valid.append(olivettifaces3_valid[i]);

    olivettifaces1_shuf_valid_r.append(olivettifaces1_valid_r[i]);
    olivettifaces2_shuf_valid_r.append(olivettifaces2_valid_r[i]);
    olivettifaces3_shuf_valid_r.append(olivettifaces3_valid_r[i]);

    y_valid_shuf.append(y_valid[i])
    ijk_shuf_valid.append(ijk_valid[i])
    f4.write(str(ijk_valid[i][0])+"\t"+str(ijk_valid[i][1])+"\t"+str(ijk_valid[i][2])+"\n");


print 'process testing'
print   ('now():'+str( datetime.now() ))
index_shuf_test=range(len(olivettifaces1_test))
xxx=random.shuffle(index_shuf_test)
count=0;
for i in index_shuf_test:
      #  print 'writing testing...'+str(count)
    olivettifaces1_shuf_test.append(olivettifaces1_test[i]);
    olivettifaces2_shuf_test.append(olivettifaces2_test[i]);
    olivettifaces3_shuf_test.append(olivettifaces3_test[i]);

    olivettifaces1_shuf_test_r.append(olivettifaces1_test_r[i]);
    olivettifaces2_shuf_test_r.append(olivettifaces2_test_r[i]);
    olivettifaces3_shuf_test_r.append(olivettifaces3_test_r[i]);

    y_test_shuf.append(y_test[i])
    ijk_shuf_test.append(ijk_test[i])
    f5.write(str(ijk_test[i][0])+"\t"+str(ijk_test[i][1])+"\t"+str(ijk_test[i][2])+"\n");

# print np.array(olivettifaces1_shuf_train).shape
# print np.array(olivettifaces1_shuf_valid).shape
# print np.array(olivettifaces1_shuf_test).shape

print 'writing training'
print   ('now():'+str( datetime.now() ))
write_file=open('./ijk_data/AUC_new_dataset_train_811.pkl','wb')
cPickle.dump([olivettifaces1_shuf_train,olivettifaces2_shuf_train,olivettifaces3_shuf_train],write_file)
write_file.close()

write_file=open('./ijk_data/AUC_new_dataset_train_811_for_rule.pkl','wb')
cPickle.dump([olivettifaces1_shuf_train_r,olivettifaces2_shuf_train_r,olivettifaces3_shuf_train_r],write_file)
write_file.close()



print 'writing valid'
print   ('now():'+str( datetime.now() ))
write_file=open('./ijk_data/AUC_new_dataset_valid_811.pkl','wb')
cPickle.dump([olivettifaces1_shuf_valid,olivettifaces2_shuf_valid,olivettifaces3_shuf_valid],write_file)
write_file.close()

write_file=open('./ijk_data/AUC_new_dataset_valid_811_for_rule.pkl','wb')
cPickle.dump([olivettifaces1_shuf_valid_r,olivettifaces2_shuf_valid_r,olivettifaces3_shuf_valid_r],write_file)
write_file.close()


print 'writing test'
print   ('now():'+str( datetime.now() ))
write_file=open('./ijk_data/AUC_new_dataset_test_811.pkl','wb')
cPickle.dump([olivettifaces1_shuf_test,olivettifaces2_shuf_test,olivettifaces3_shuf_test],write_file)
write_file.close()

write_file=open('./ijk_data/AUC_new_dataset_test_811_for_rule.pkl','wb')
cPickle.dump([olivettifaces1_shuf_test_r,olivettifaces2_shuf_test_r,olivettifaces3_shuf_test_r],write_file)
write_file.close()


write_file=open('./ijk_data/y_valid.pkl','wb')
cPickle.dump([y_valid_shuf],write_file)
write_file.close()
write_file=open('./ijk_data/y_test.pkl','wb')
cPickle.dump([y_test_shuf],write_file)
write_file.close()
