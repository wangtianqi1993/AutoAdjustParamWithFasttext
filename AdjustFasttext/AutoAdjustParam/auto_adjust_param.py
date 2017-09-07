# !/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import sys
import time
from datetime import datetime

import subprocess
import multiprocessing
from dir_card import Cartesian
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from Config.config import MODEL_RESULT_NAME

project_path = os.path.abspath(os.path.join(os.path.dirname("auto_adjust_param.py"), os.pardir))
data_path_raw = os.path.join(project_path, "Data")
model_path_raw = os.path.join(project_path, "Model")
model_test_result_path = os.path.join(model_path_raw, MODEL_RESULT_NAME)


class AutoAdjustParameter():

    def auto_adjust(self, train_data_name, test_data_name, **kwargs):
        """
        以多进程的方式，对fasttext自动化调参
        按照key-value的形式传入参数名与其对应的列表值
        例如：auto_adjust(lr=[1, 2, 3], epoch=[2, 3, 4])
        求所有参数对应的n个笛卡尔积，并训练对应的n个模型
        :param kwargs:
        :return:
        """
        start_time = time.time()
        param_val_store = []
        param_name_store = []
        train_data_path = os.path.join(data_path_raw, train_data_name)
        test_data_path = os.path.join(data_path_raw, test_data_name)

        for key in kwargs:
            param_name_store.append(key)
            param_val_store.append(kwargs[key])

        if len(param_name_store) == 0 or len(param_name_store)!=len(param_val_store):
            raise Exception("place input the right parameter")

        parameters = Cartesian(param_val_store).assemble()
        parameters_nums = len(param_name_store)

        test_result_data = []
        test_result_data.append("train time " + str(datetime.now())+"\n")

        # 使用最大并发进程池的方式，设置最大并发进程数,默认为当前cpu内核数
        pools = multiprocessing.Pool()
        process_result = []

        for item_index in range(len(parameters)):
            model_path = os.path.join(model_path_raw, str(item_index))
            fasttext_train_shell = "fasttext supervised -input " + train_data_path + " -output " + model_path
            fasttext_test_shell = "fasttext test " + model_path + ".bin " + test_data_path
            # fasttest_test_traindata_shell = "fasttext test " + model_path + ".bin " + train_data_path

            param_item = ""
            for index in range(parameters_nums):
                param_item += " -" + str(param_name_store[index]) + " " + str(parameters[item_index][index])
            fasttext_train_shell += param_item

            # 根据不同的参数组合训练fasttext模型
            print fasttext_train_shell
            # 创建子进程加入到进程池中
            param_item = "mode" + str(item_index) + param_item + '\n'
            process_result.append(pools.apply_async(train_fasttext, args=(fasttext_train_shell, fasttext_test_shell, param_item)))

        # 等待每个子进程执行完
        pools.close()
        pools.join()
        # 获取每个进程的输出结果
        for result_out in process_result:
            result = result_out.get()
            if result:
                test_result_data.append(result)

        with open(model_test_result_path, "a+") as f:
            f.writelines(test_result_data)
        end_time = time.time()
        print "used time", end_time - start_time


def train_fasttext(train_shell, test_shell, param_item):
    """
    训练并测试fasttext模型，返回测试的结果
    :param run_shell:
    :return:
    """
    try:
        fasttext_train = subprocess.Popen(train_shell, shell=True, executable="/bin/bash", stdout=subprocess.PIPE)
        fasttext_train.wait()
    except Exception, e:
        print "train model process failed!", e
        return

    try:
        fasttext_test = subprocess.Popen(test_shell, shell=True, executable="/bin/bash", stdout=subprocess.PIPE)
        fasttext_test.wait()
        test_out = fasttext_test.communicate()
        param_item += test_out[0]

    except Exception, e:
        print "test model process failed!", e
        return
    return param_item

instance = AutoAdjustParameter()
instance.auto_adjust("spam.train", "spam.test", lr=[0.6], epoch=[20, 30, 40], loss=["ns", "hs"])

