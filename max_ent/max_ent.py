#-------------------------------------------------------------------------------
# Name:        模块1
# Purpose:
#
# Author:      xlm
#
# Created:     05/03/2014
# Copyright:   (c) xlm 2014
# Licence:     <your licence>
#-------------------------------------------------------------------------------

# -*- coding:utf8 -*-
import math
import sys

def load_data(fname):
        """
                从文件加载训练或测试数据
                返回得到的输入矩阵x，输出向量y，
                以及x各维特征的最大值x_max，y的最大值y_max（用于确定x,y的取值范围，它们都是离散值）
        """
        f = open(fname)
        y=[]
        x=[]
        for line in f: #类似line=f.readlines()
                line=line.strip('\n')  #trim
                str_list=line.split(' ')
                xi=[]
                for v in str_list:  #v=c5 , f20
                        if v[0] == 'c':
                                yi=v[1:]
                                y.append(int(yi))
                        if v[0] == 'f':
                                xij=v[1:]
                                xi.append(int(xij))
                x.append(xi)  #x是矩阵
        f.close()
        y_max=1
        for yval in y:
                if yval > y_max:
                        y_max=yval
        x_max=[]
        for a in x[0]:
                x_max.append(a)
        for b in x:
                for i in range(len(b)):
                        if b[i] > x_max[i]:
                                x_max[i] = b[i]
        return x,y,x_max,y_max

# 特征函数及其对应的参数

class FeatureParam:
        def __init__(self,ft_index,y_int,x_int):
                self.w=0.0
                self.ft_index=ft_index
                self.y_int=y_int
                self.x_int=x_int
        def feature_func(self,xvec,y): #特征函数，如果x=xi and y=yi则1
                if y == self.y_int and xvec[self.ft_index] == self.x_int:
                        return 1
                else:
                        return 0

def init_feature_func_and_param(x_max_vec,y_max):
        # 生成特征函数:
        # 每个不同的<特征i，特征值的值v，分类标签label的值c>三元组都确定一个特征函数f
        # f(x,y)，当向量x的第i维=v，且y=c时，特征函数值为1，否则为0.
        fvec=[]
        ft_index=-1
        for x_max in x_max_vec:
                ft_index+=1
                for xval in range(0,x_max+1):
                        for yval in range(1,y_max+1):
                                fp=FeatureParam(ft_index,yval,xval)
                                fvec.append(fp)
        return fvec

def estimated_except_feature_val(x_mat,y_vec,fparam):
        esti_efi = 0.0
        n_data=len(y_vec)
        # 将每条训练数据的特征函数值相加
        # 除以训练数据总数，即得经验特征函数期望值
        for i in range(n_data):
                x_vec = x_mat[i]
                esti_efi += fparam.feature_func(x_vec,y_vec[i])
        # perhaps there's no data match the feature function
        # to make the computation possible, let it be a small float number
        if esti_efi == 0.0:
                esti_efi = 0.0000001
        esti_efi /= 1.0*n_data
        return esti_efi

def max_ent_predict_unnormalized(x_vec,y,fvec):
        """
                未归一化的概率值
                即各特征函数值的加权和
                加权系数即对模型训练出来的参数
        """
        weighted_sum=0.0
        for fparam in fvec:
                weighted_sum += fparam.w * fparam.feature_func(x_vec,y)
        return math.exp(weighted_sum)

def max_ent_normalizer(x_vec,y_max,fvec):
        zw=0.0
        for y in range(1,y_max+1):
                zw += max_ent_predict_unnormalized(x_vec,y,fvec)
        return zw

def model_except_feature_val(x_mat,y_max,fparam,fvec,p_cached):
        data_size=len(x_mat)
        efi=0.0
        index=0
        ix=-1
        # 对各训练数据x：对所有可能的y的取值，求P(y|x)与特征函数值的乘积和
        # 并求其平均值
        # 即得模型的预测特征函数期望值
        for x_vec in x_mat:
                ix += 1
                zw=0.0
                tmp_efi=0.0
                for y in range(1,y_max+1):
                        # compute p(y|x) at current w
                        # in same iteration, p(y|x) not change, so can cache it
                        if index < len(p_cached):
                                p_y_given_x = p_cached[index]
                        else:
                                p_y_given_x = max_ent_predict_unnormalized(x_vec,y,fvec)
                                p_cached.append(p_y_given_x)
                        zw += p_y_given_x
                        tmp_efi += p_y_given_x * fparam.feature_func(x_vec,y)
                        index+=1
                tmp_efi /= zw
                efi += tmp_efi
        efi /= data_size
        if efi == 0.0:
                efi = 0.0000001
        return efi

def feature_func_sum(fvec,xvec,y):
        m = 0.0
        for f in fvec:
                m += f.feature_func(xvec,y)
        return m

def update_param(x_mat,y_vec,y_max,fvec):
        """
                更新每个特征函数对应的参数:
                x_mat: 训练数据的输入矩阵
                y_vec: 训练数据的label向量
                y_max: label的最大取值，label取值范围为[1,y_max]
                fvec: 特征函数及对应参数组成的向量
        """
        # 所有特征函数的值相加，对不同的x,y来说，它是一个常量
        m = feature_func_sum(fvec,x_mat[0],y_vec[0])
        # 是否收敛
        convergenced = True
        # 参数更新向量各维的平方和，即模的平方
        sigma_sqr_sum=0.0
        # 更新后的参数
        w_updated=[]
        # 缓存P(y|x)的结果，避免重复计算
        p_cached=[]
        # 对每一个特征函数，分别更新对应的参数
        for fparam in fvec:
                # 计算训练数据的特征函数平均值
                esti_efi = estimated_except_feature_val(x_mat,y_vec,fparam)
                # 计算当前模型的特征函数期望值
                efi = model_except_feature_val(x_mat,y_max,fparam,fvec,p_cached)
                # 计算当前参数的更新值
                # 由于m是个常数，所以可以用这种方法计算
                sigma_i= math.log(esti_efi/efi) / m
                w_updated.append(fparam.w + sigma_i)
                # 如果对参数的更新较大，则认为不收敛
                if abs(sigma_i/(fparam.w+0.000001)) >= 0.10:
                        convergenced = False
                sigma_sqr_sum += sigma_i*sigma_i
        i=0
        for fparam in fvec:
                fparam.w = w_updated[i]
                i+=1
        # 打印当前更新向量的长度
        print("sigma_len=%f"%math.sqrt(sigma_sqr_sum))
        return convergenced

def log_likelihood(x_mat,y_vec,y_max,fvec):
        """
                原理：当p有多个估计值供我们选择时，我们会选取最大的值作为估计值，这就是极大似然估计法。
                样本的联合概率密度为函数f,我们称为似然函数.如果存在一个参数x使得似然函数最大，那么这个值就是最大似然估计法
                对每条训练数据的x，求其模型预测概率P(y|x)，
                相乘并取对数，即得对数似然函数值，
                该值越大，说明模型对训练数据的拟合越准确
        """
        ix=-1
        log_likelihood = 0.0
        data_size = len(x_mat)  #返回x矩阵的行数
        for i in range(data_size):
                x_vec = x_mat[i]
                y = y_vec[i]
                log_likelihood += math.log(max_ent_predict_unnormalized(x_vec,y,fvec))
                log_likelihood -= math.log(max_ent_normalizer(x_vec,y_max,fvec))
        log_likelihood /= data_size
        return log_likelihood

def max_ent_train(x_matrix,y_vec,x_max_vec,y_max):
        """
                最大熵模型训练，使用IIS(improved iterative scaling)迭代算法
        """
        fvec = init_feature_func_and_param(x_max_vec,y_max)
        iter_time=0
        while True:
                # 更新参数,返回是否收敛
                convergenced=update_param(x_matrix,y_vec,y_max,fvec)
                # 计算对数似然值
                log_lik=log_likelihood(x_matrix,y_vec,y_max,fvec)
                print("log_likelihood=%0.12f"%log_lik)
                if convergenced:
                        break
                iter_time+=1
                if iter_time >= 100000:
                        break
        # 将训练得到的模型写到文件，一行一个参数
        fmodel=open('E:\python\python练习\max_ent\model.txt','w')
        for fparam in fvec:
                fmodel.write(str(fparam.w))
                fmodel.write('\n')
        fmodel.close()
        print("Max-ent train ok!")

def load_model():
        """
                加载已训练好的最大熵模型
        """
        x_mat,y_vec,x_max_vec,y_max=load_data("E:\python\python练习\max_ent\zoo.train")
        fvec = init_feature_func_and_param(x_max_vec,y_max)
        fmod=open('E:\python\python练习\max_ent\model.txt')
        i=-1
        for line in fmod:
                i+=1
                line=line.strip('\n')
                fvec[i].w=float(line)
        fmod.close()
        return fvec,y_max

def max_ent_test():
        fvec,y_max = load_model()
        x_mat_test,y_vec_test,x_max_vec_test,y_max_test=load_data("E:\python\python练习\max_ent\zoo.test")
        test_size=len(x_mat_test)
        ok_num=0
        for i in range(test_size):
                x_vec=x_mat_test[i]
                y=y_vec_test[i]
                most_possible_predict_y=0
                max_p=0.0
                sum_p=0.0
                for predict_y in range(1,y_max+1):
                        p = max_ent_predict_unnormalized(x_vec,predict_y,fvec)
                        sum_p += p
                        if p > max_p:
                                most_possible_predict_y = predict_y
                                max_p = p
                if y == most_possible_predict_y:
                        ok_num += 1
                p_normalized = max_p / sum_p
                print("y=%d predict_y=%d p=%f"%(y,most_possible_predict_y,p_normalized))
        print("precision-ratio=%f"%(1.0*ok_num/test_size))

if __name__=="__main__":
        flag =raw_input('train?or test?' )
        if flag=='train':
                # 最大熵模型训练
                # 训练数据zoo.train,模型输出到文件model.txt
                x_matrix,y_list,x_max_list,y_max=load_data("E:\python\python练习\max_ent\zoo.train")
                max_ent_train(x_matrix,y_list,x_max_list,y_max)

        if flag=='test':
                # 最大熵模型测试
                max_ent_test()