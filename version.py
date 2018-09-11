'''
这里进行重构。进行对于大数据集的分块操作。进行协同进化
'''

'''
代码所需要的包
'''
import tensorflow as tf
import tensorflow_probability as tfp
# from tensorflow.contrib.distributions import MultivariateNormalFullCovariance
import pandas as pd
import numpy as np
import threading
import math
import random
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import neighbors
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

'''
全局网络的构建
'''


class Global_pop(object):
    def __init__(self, name, pops, pop, sub, sub_size):
        """
        这个类是用来定义全局网络的类
        :param name: 用来表示所在种群全局网络的名称
        :param pops: 用来表示所有的线程
        :param pop: 用来表示所在的种群
        :param sub: 用来表示所在的子集
        :param sub_size: 用来表示所在种群的种群大小
        """
        with tf.variable_scope(name):
            self.name = name
            self.pops = pops
            self.pop = pop
            self.sub = sub
            self.sub_size = sub_size
            with tf.variable_scope('mean'):
                self.mean = tf.Variable(tf.truncated_normal([self.sub_size, ], stddev=0.05, mean=0.5), dtype=tf.float32,
                                        name=name + '_mean')
            with tf.variable_scope('cov'):
                self.cov = tf.Variable(1.0 * tf.eye(self.sub_size), dtype=tf.float32, name=name + '_cov')
            self.mvn = tfp.distributions.MultivariateNormalFullCovariance(loc=self.mean, covariance_matrix=abs(self.cov))
            self.mean_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name + '/mean')
            self.cov_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name + '/cov')


'''
种群网络的构建
'''


class Worker_pop(object):
    def __init__(self, name, pops, pop, sub, sub_size, global_pop, Factor=2):
        """
        这个类用来定义种群网络
        :param name: 用来表示所在种群的名称
        :param pops: 用来表示所有的线程
        :param pop: 用来表示所在的种群
        :param sub: 用来表示所在的子集
        :param sub_size: 用来表示所在种群的大小
        :param global_pop: 用来存储对应的全局网络
        :param Factor: 用来表示所选取的offset用于更新网络的个数因子
        """
        with tf.variable_scope(name):
            self.name = name
            self.pops = pops
            self.pop = pop
            self.sub = sub
            self.sub_size = sub_size
            self.N_POP_size = N_POP
            self.C_POP_size = math.floor(N_POP / Factor)
            with tf.variable_scope('mean'):
                self.mean = tf.Variable(tf.truncated_normal([self.sub_size, ], stddev=0.05, mean=0.5), dtype=tf.float32,
                                        name=name + '_mean')
            with tf.variable_scope('cov'):
                self.cov = tf.Variable(1.0 * tf.eye(self.sub_size), dtype=tf.float32, name=name + '_cov')
            self.mvn = tfp.distributions.MultivariateNormalFullCovariance(loc=self.mean, covariance_matrix=abs(self.cov))
            self.make_kid = self.mvn.sample(self.N_POP_size)
            self.tfkids_fit = tf.placeholder(tf.float32, [self.C_POP_size, ])
            self.tfkids = tf.placeholder(tf.float32, [self.C_POP_size, self.sub_size])
            self.loss = -tf.reduce_mean(self.mvn.log_prob(self.tfkids) * self.tfkids_fit)
            self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
            self.mean_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name + '/mean')
            self.cov_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name + '/cov')

            with tf.name_scope('pull'):
                self.pull_mean_op = self.mean.assign(global_pop.mean)
                self.pull_cov_op = self.cov.assign(global_pop.cov)
            with tf.name_scope('push'):
                self.push_mean_op = global_pop.mean.assign(self.mean)
                self.push_cov_op = global_pop.cov.assign(self.cov)
            with tf.name_scope('restart'):
                self.re_mean_op = self.mean.assign(
                    tf.Variable(tf.truncated_normal([self.sub_size, ], stddev=0.05, mean=0.5), dtype=tf.float32))
                self.re_cov_op = self.cov.assign(tf.Variable(1.0 * tf.eye(self.sub_size), dtype=tf.float32))

    def _update_net(self):
        sess.run([self.push_mean_op, self.push_cov_op])

    def _pull_net(self):
        sess.run([self.pull_mean_op, self.pull_cov_op])

    def _restart_net(self):
        sess.run([self.re_mean_op, self.re_cov_op])


'''
数据预处理的类
'''


class Dataset(object):
    def __init__(self, file, type):
        """
        这是一个数据预处理的类。封装了对于数据集的预处理操作
        :param filename: 数据集的名称，不包含文件名后缀
        :param type: 数据集的类型
        """
        self.file = file
        self.filename = file + '.txt'
        self.type = type
        self.DNA_size = 0
        self.Feature = []
        self.trainX = []
        self.trainy = []
        self.testX = []
        self.testy = []

    def __loadData__(self, label_loc, div_mode, test_size, select_feature_size):
        """
        这是表示对数据集加载的操作，通过传入的文件格式以及参数
        :param label_loc: 表示标签的位置，在前为0，在后为1；
        :param div_mode: 表示数据分割的方式，有','和' ';
        :param test_size: 表示数据集划分的方式，有0.3（表示随机划分），2（2折验证），10（10折验证）
        :param select_feature_size: 表示通过过滤所留下来的特征数
        :return:
        """
        self.label_loc = label_loc
        self.div_mode = div_mode
        self.test_size = test_size
        self.select_feature = select_feature_size
        self.__pretreatment__()
        if self.label_loc == 0:
            data = pd.read_table(self.filenamed, sep=' ')
            self.fea, self.lab = data.ix[:, 1:], data.ix[:, 0]
            self.DNA = len(np.array(self.fea)[0])
            if self.DNA < self.select_feature:
                self.select_feature = self.DNA
        elif self.label_loc == 1:
            data = pd.read_table(self.filenamed, sep=' ')
            f = open(self.filenamed)
            self.fea, self.lab = data.ix[:, 0:len(f.readline().split(sep=' ')) - 1], data.ix[:, -1]
            f.close()
            self.DNA = len(np.array(self.fea)[0])
            if self.DNA < self.select_feature:
                self.select_feature = self.DNA

    def __pretreatment__(self):
        """
        此方法表示为每个数据集添加一行的操作，因为pandas读取数据时会忽略第一行数据
        :return:
        """
        f = open(self.filename)
        numData = len(f.readline().split(self.div_mode))
        f.close()
        v = []
        val = []
        for i in range(numData):
            v.append(i)
        val.append(v)
        fr = open(self.filename)
        for line in fr.readlines():
            xi = []
            curline = line.strip().split(self.div_mode)
            for i in range(numData):
                xi.append((curline[i]))
            val.append(xi)
            self.filenamed = self.file + 'ed' + '.txt'
        self.saveData(self.filenamed, np.array(val))
        fr.close()

    def saveData(self, filename, dataname):
        '''
        这是用于存储数据的方法
        :param filename: 要存储的文件名
        :param dataname: 要存储的目标文件
        :return:
        '''
        with open(filename, 'w') as file_object:  # 将文件及其内容存储到变量file_object
            # 写入第一行(第一块)
            file_object.write(str(dataname[0, 0]))  # 写第一行第一列
            for j in range(1, np.size(dataname, 1)):
                file_object.write(' ' + str(dataname[0, j]))  # 写第一列后面的列

            # 写入第一行后面的行（第二块）
            for i in range(1, np.size(dataname, 0)):
                file_object.write('\n' + str(dataname[i, 0]))
                for j in range(1, np.size(dataname, 1)):
                    file_object.write(' ' + str(dataname[i, j]))

    def __getData__(self):
        '''
        这是用于过滤特征并将数据进行划分的方法
        :return:
        '''
        # self.train_selected = SelectFromModel(GradientBoostingClassifier()).fit_transform(self.fea, self.lab)
        if self.type == 'split':
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.fea, self.lab,
                                                                                    test_size=self.test_size)
            self.getMatDataBysplit(self.x_train, self.x_test, self.y_train, self.y_test)
        elif self.type == 'nfold':
            skf = StratifiedKFold(n_splits=self.test_size, shuffle=True)
            skf.get_n_splits(self.fea, self.lab)
            self.getMatDataBynfold(skf, self.fea, self.lab)

    # 将列表形式的数据转换为矩阵形式
    def getMatDataBysplit(self, x_train, x_test, y_train, y_test):
        '''
        这是用于将数据按矩阵形式存储的方法，此方法中的数据是将数据集按照随机划分得到的
        :param x_train:
        :param x_test:
        :param y_train:
        :param y_test:
        :return:
        '''
        self.trainX.append(np.array(x_train))
        self.testX.append(np.array(x_test))
        self.trainy.append(np.array(y_train))
        self.testy.append(np.array(y_test))
        self.DNA_size = len(np.array(x_train)[0])
        for i in range(self.DNA_size):
            self.Feature.append(i)  # 其中的元素是特征所在的位置

    def getMatDataBynfold(self, skf, feature, label):
        '''
        这是用于将数据按矩阵形式存储的方法，此方法中的数据是将数据集按照n_fold划分得到的
        :param skf:
        :param feature:
        :param label:
        :return:
        '''
        for train_index, test_index in skf.split(feature, label):
            print(test_index)
            trainX_ = feature[train_index]
            trainy_ = label[train_index]
            testX_ = feature[test_index]
            testy_ = label[test_index]
            self.trainX.append(trainX_)
            self.testX.append(testX_)
            self.trainy.append(trainy_)
            self.testy.append(testy_)
        self.DNA_size = len(np.array(feature)[0])
        for i in range(self.DNA_size):
            self.Feature.append(i)

    def getDNA_size(self):
        return self.DNA_size

    def getdata(self):
        return self.trainX, self.testX, self.trainy, self.testy

    def getFeature(self):
        return self.Feature

    def getReadType(self):
        return self.type

    def getTestsize(self):
        return self.test_size


'''
然后就是对每个线程工作内容的定义
'''


class Worker(object):

    def __init__(self, name, pops, pop, sub, data, classifier, factor, block):
        '''
        这里定义线程工作的类，包括各种方法
        :param name: 线程的名称
        :param pops: 所有的种群子集
        :param pop: 表示当前的线程所在的种群
        :param sub: 表示当前的线程所在的种群子集
        :param data: 表示预处理过的数据集
        :param classifier: 表示所所使用的分类器
        :param factor: 表示折扣因子，用来说明所选的offset的个数
        :param block: 表示种群子集的大小
        '''
        self.name = name
        self.pops = pops
        self.pop = pop
        self.sub = sub
        self.data = data
        self.DNA_size = data.getDNA_size()
        self.classifier = classifier
        self.factor = factor
        self.Block = block
        self.global_pop = Global_pop(name, pops, pop, sub, self.DNA_size)
        self.popnet = Worker_pop(name, pops, pop, sub, self.DNA_size, self.global_pop, self.factor)  # 每个线程在初始化中首先区初始化一个网络

    def work(self):
        for g in range(MAX_GLOBAL_EP):
            print(self.name, g + 1, '次迭代开始:')
            kids = sess.run(self.popnet.make_kid)
            print(kids)


'''
定义主过程函数，定义迭代次数，以及相关的参数
'''
if __name__ == '__main__':
    with tf.Graph().as_default(), tf.device('/cpu:0'):

        N_WORKERS = 6  # 表示运行的线程个数
        N_POP = 30  # 表示生成子代的个数
        LEARNING_RATE = 0.001  # 表示算法的学习率
        MAX_GLOBAL_EP = 301  # 表示算法的迭代次数

        # 接着就是定义一些锁来对共享资源进行锁定
        lock_kids = threading.Lock()
        lock_max_fit = threading.Lock()
        lock_dr = threading.Lock()
        lock_push = threading.Lock()
        lock_pull = threading.Lock()
        lock_Fit_val = threading.Lock()
        lock_is_choose = threading.Lock()
        lock_global1 = threading.Lock()
        lock_global2 = threading.Lock()

        # 定义分类器
        KNN = 'train_knn'
        SVM = 'svm'
        TREE = 'tree'

        # 定义数据集划分方式
        SPLIT = 'split'
        NFOLD = 'nfold'

        # 定义回话
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())  # initialize tf variables

        # 首先是将数据集特征的第一步过滤
        Factor = 3
        Block = 10
        data = Dataset('/home/zhangxin/Dataset/arcene', SPLIT)
        data.__loadData__(1, ',', 0.3, 500)
        data.__getData__()

        # 然后就是将数据及按照子集形式进行划分
        fea = data.getFeature()
        size = data.getDNA_size()
        pops = []
        step = math.ceil(size / Block)
        for p in range(N_WORKERS):
            random.shuffle(fea)
            sub = [fea[i:i + step] for i in range(0, size, step)]
            pops.append(sub)

        # 定义结果输出文件
        fh = open('/home/zhangxin/Dataset/result.txt', 'w')
        fh.write("begin:\n")
        fh.close()

        # 接着就是通过核心数来创建多线程进行工作
        with tf.device('/cpu:0'):
            workers = []
            i = 0
            for pop in range(N_WORKERS):
                for sub in range(Block):
                    with tf.device('/gpu:%d' % i):
                        print(i)
                        i += 1
            #             with tf.name_scope('sub_%d' % (sub + 1)) as scope:
            #                 i_name = 'sub_%d_of_pop_%d' % ((sub + 1), (pop + 1))
            #                 workers.append(Worker(i_name, pops, pop, sub, data, SVM, Factor, Block))
            #
            # COORD = tf.train.Coordinator()
            # sess.run(tf.global_variables_initializer())
            #
            # worker_threads = []
            # for worker in workers:
            #     job = lambda: worker.work()
            #     t = threading.Thread(target=job)
            #     t.start()
            #     worker_threads.append(t)
            # COORD.join(worker_threads)
