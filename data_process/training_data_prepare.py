# coding=utf-8
import cPickle
import os
import json
import numpy
from matplotlib import pylab
from PIL import Image
import numpy as np
import scipy.io as sio
import random
from  datetime  import  *

# dataFile = 'D:/PycharmProjects/liujinhuan/fc71.mat'
# data = sio.loadmat(dataFile)


# f=sio.loadmat('D:/PycharmProjects/liujinhuan/fc71.mat')
print ('loading top features....')
print   ('now():'+str( datetime.now() ))
my_matrix_top = numpy.loadtxt(open("/storage/songxuemeng/dataset_backup/filter_dataset/features_top/myfeatures_mean_top.csv","r"),delimiter=",",skiprows=0)

print ('loading bottom features....')
print ('now():'+str( datetime.now()))
my_matrix_bottom = numpy.loadtxt(open("/storage/songxuemeng/dataset_backup/filter_dataset/features_bottom/myfeatures_mean_bottom.csv","r"),delimiter=",",skiprows=0)
print my_matrix_bottom.shape
print my_matrix_top.shape

print ('loading top-bottom lists....')
print   ('now():'+str( datetime.now() ))
top_list=numpy.loadtxt(open("/storage/songxuemeng/dataset_backup/filter_dataset/top_list_id.txt","r"),delimiter="\t",skiprows=0)
bottom_list = numpy.loadtxt(open("/storage/songxuemeng/dataset_backup/filter_dataset/bottom_list_id.txt","r"),delimiter="\t",skiprows=0)
top_bottom_list = numpy.loadtxt(open("/storage/songxuemeng/dataset_backup/filter_dataset/unique_newdataset_top_bottom_valid_pairs.txt","r"),delimiter="\t",skiprows=0)



K=3;
olivettifaces1=numpy.empty((len(top_bottom_list)*K,4096))
olivettifaces2=numpy.empty((len(top_bottom_list)*K,4096))
olivettifaces3=numpy.empty((len(top_bottom_list)*K,4096))

ijk=numpy.empty((len(top_bottom_list)*K,3))

f1=open('/storage/songxuemeng/dataset_backup/filter_dataset/pkl/total_ijk_shuffled_811.txt','w')
f2=open('/storage/songxuemeng/dataset_backup/filter_dataset/pkl/total_ijk_811.txt','w')

f0=open('/storage/songxuemeng/dataset_backup/filter_dataset/pkl/unique_newdataset_top_bottom_valid_pairs_shufflelist.txt','w')

f3=open('/storage/songxuemeng/dataset_backup/filter_dataset/pkl/train_ijk_shuffled_811.txt','w')
f4=open('/storage/songxuemeng/dataset_backup/filter_dataset/pkl/valid_ijk_shuffled_811.txt','w')
f5=open('/storage/songxuemeng/dataset_backup/filter_dataset/pkl/test_ijk_shuffled_811.txt','w')



train_end=int(0.8*len(top_bottom_list))
valid_end=int(0.9*len(top_bottom_list))
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
print   ('now():'+str( datetime.now() ))
count=0;
for i in range (0, len(new_top_bottom_list)):
    if count%2000==0:
        print ('the '+str(i)+' top-bottom pair')

      #  print 'writing training...'+str(count)
    tid=new_top_bottom_list[i,0]
    bid=new_top_bottom_list[i,1]
    top_index= numpy.where(top_list==tid)[0]
    bottom_index= numpy.where(bottom_list==bid)[0]
    # print 'top_index'
    # print top_index
    # print 'bottom_index'
    # print bottom_index
    # if len(top_index)==0 or len(bottom_index)==0:
    #     print str(tid)+" and " +str(bid)+" is not available";
    #     continue

    # if top_index[0]>1000:
    #     top_index=999
    # if bottom_index[0]>1000:
    #     bottom_index=999
    top_feature=my_matrix_top[top_index]
    bottom_feature=my_matrix_bottom[bottom_index]

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
                     neg_index= numpy.where(bottom_list==r1)[0]

                     # if neg_index[0]>=1000:
                     #     neg_index=999;

                     neg_feature=my_matrix_bottom[neg_index]
                     olivettifaces1[count*K+neg_num]=numpy.ndarray.flatten(top_feature)
                     olivettifaces2[count*K+neg_num]=numpy.ndarray.flatten(bottom_feature)
                     olivettifaces3[count*K+neg_num]=numpy.ndarray.flatten(neg_feature)
                     ijk[count*K+neg_num,0]=tid;
                     ijk[count*K+neg_num,1]=bid;
                     ijk[count*K+neg_num,2]=r1;
                     neg_num=neg_num+1;
                     f2.write(str(tid)+"\t"+str(bid)+"\t"+str(r1)+"\n");


    count=count+1;

olivettifaces1_train=olivettifaces1[0:train_end*K,:];
olivettifaces2_train=olivettifaces2[0:train_end*K,:];
olivettifaces3_train=olivettifaces3[0:train_end*K,:];
ijk_train=ijk[0:train_end*K,:];

print 'olivettifaces1_train length'
print len(olivettifaces1_train)


olivettifaces1_valid=olivettifaces1[train_end*K:valid_end*K,:];
olivettifaces2_valid=olivettifaces2[train_end*K:valid_end*K,:];
olivettifaces3_valid=olivettifaces3[train_end*K:valid_end*K,:];
ijk_valid=ijk[train_end*K:valid_end*K,:];

print 'olivettifaces1_valid length'
print len(olivettifaces1_valid)
olivettifaces1_test=olivettifaces1[valid_end*K:len(olivettifaces1),:];
olivettifaces2_test=olivettifaces2[valid_end*K:len(olivettifaces1),:];
olivettifaces3_test=olivettifaces3[valid_end*K:len(olivettifaces1),:];
ijk_test=ijk[valid_end*K:len(olivettifaces1),:];

print 'olivettifaces1_test length'
print len(olivettifaces1_test)
olivettifaces1_shuf_train=[]
olivettifaces2_shuf_train=[]
olivettifaces3_shuf_train=[]
olivettifaces1_shuf_valid=[]
olivettifaces2_shuf_valid=[]
olivettifaces3_shuf_valid=[]
olivettifaces1_shuf_test=[]
olivettifaces2_shuf_test=[]
olivettifaces3_shuf_test=[]
ijk_shuf=[]
ijk_shuf_valid=[]
ijk_shuf_train=[]
ijk_shuf_test=[]

numpy.savetxt('train_data.csv', numpy.asarray(olivettifaces1_train[0:100,:]), fmt="%f")
numpy.savetxt('valid_data.csv', numpy.asarray(olivettifaces1_valid[0:100,:]), fmt="%f")
numpy.savetxt('test_data.csv', numpy.asarray(olivettifaces1_test[0:100,:]), fmt="%f")

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
    ijk_shuf_test.append(ijk_test[i])
    f5.write(str(ijk_test[i][0])+"\t"+str(ijk_test[i][1])+"\t"+str(ijk_test[i][2])+"\n");

# print np.array(olivettifaces1_shuf_train).shape
# print np.array(olivettifaces1_shuf_valid).shape
# print np.array(olivettifaces1_shuf_test).shape
print 'writing training'
print   ('now():'+str( datetime.now() ))
write_file=open('/storage/songxuemeng/dataset_backup/filter_dataset/pkl/AUC_new_dataset_train_811.pkl','wb')
# cPickle.dump将train数据放入pkl文件，其中train又包含三个矩阵，第一个为top的feature,第二个为和top相搭配的bottom feature，
# 第三个为随意搭配的bottom feature.
cPickle.dump([olivettifaces1_shuf_train,olivettifaces2_shuf_train,olivettifaces3_shuf_train],write_file)
write_file.close()
print 'writing valid'
print   ('now():'+str( datetime.now() ))
write_file=open('/storage/songxuemeng/dataset_backup/filter_dataset/pkl/AUC_new_dataset_test_811.pkl','wb')
# cPickle.dump将train数据放入pkl文件，其中train又包含三个矩阵，第一个为top的feature,第二个为和top相搭配的bottom feature，
# 第三个为随意搭配的bottom feature.
cPickle.dump([olivettifaces1_shuf_test,olivettifaces2_shuf_test,olivettifaces3_shuf_test],write_file)
write_file.close()
print 'writing testing'
print   ('now():'+str( datetime.now() ))
write_file=open('/storage/songxuemeng/dataset_backup/filter_dataset/pkl/AUC_new_dataset_valid_811.pkl','wb')
# cPickle.dump将train数据放入pkl文件，其中train又包含三个矩阵，第一个为top的feature,第二个为和top相搭配的bottom feature，
# 第三个为随意搭配的bottom feature.
cPickle.dump([olivettifaces1_shuf_valid,olivettifaces2_shuf_valid,olivettifaces3_shuf_valid],write_file)
write_file.close()
