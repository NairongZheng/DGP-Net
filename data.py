import os
import time
import random
import skimage.io
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision as tv
from torchvision import transforms, datasets
from torchvision.datasets import CIFAR100

import cv2 as cv
from PIL import Image

class self_Dataset(Dataset):
    def __init__(self, data, label=None):
        super(self_Dataset, self).__init__()

        self.data = data
        self.label = label
    def __getitem__(self, index):
        data = self.data[index]
        # data = np.moveaxis(data, 3, 1)
        # data = data.astype(np.float32)

        if self.label is not None:
            label = self.label[index]
            # print(label)
            # label = torch.from_numpy(label)
            # label = torch.LongTensor([label])
            return data, label
        else:
            return data, 1
    def __len__(self):
        return len(self.data)

def count_data(data_dict):
    num = 0
    for key in data_dict.keys():
        num += len(data_dict[key])
    return num

class self_DataLoader(Dataset):
    def __init__(self, root, train=True, dataset='cifar100', seed=1, nway=5):
        super(self_DataLoader, self).__init__()

        self.seed = seed
        self.nway = nway
        self.num_labels = 100
        self.SAR_num_labels = 10
        self.input_channels = 3
        self.size = 32

        self.transform = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize([0.5071, 0.4866, 0.4409],
                [0.2673, 0.2564, 0.2762])
            ])
        self.transform_SAR = tv.transforms.Compose([
            tv.transforms.Grayscale(1),  # 单通道
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(
                (0.1307,), (0.3081,))
        ])

        self.full_data_dict, self.few_data_dict = self.load_data(root, train, dataset)

        print('full_data_num: %d' % count_data(self.full_data_dict))
        print('few_data_num: %d' % count_data(self.few_data_dict))

    def load_data(self, root, train, dataset):
        if dataset == 'cifar100':
            few_selected_label = random.Random(self.seed).sample(range(self.num_labels), self.nway)
            print('selected labeled', few_selected_label)

            full_data_dict = {}
            few_data_dict = {}

            d = CIFAR100(root, train=train, download=True)

            for i, (data, label) in enumerate(d):
                # print(data, label)
                # break
                data = self.transform(data)

                if label in few_selected_label:
                    data_dict = few_data_dict
                else:
                    data_dict = full_data_dict

                if label not in data_dict:
                    data_dict[label] = [data]
                else:
                    data_dict[label].append(data)
            # print(i + 1)
        elif dataset == 'MSTAR':
            # path1 = 'data/MSTAR/1'
            # filenames = os.listdir(path1)
            # for filename in filenames:
            #     img = cv.imread(os.path.join(path1, filename))
            #     transf = self.transform_SAR
            #     img_tensor = transf(Image.fromarray(img))
            #     print(1)
            #     print(filename)
            #     temp = img_tensor.squeeze(0)
            #     t = [temp[:, i] for i in range(100)]
            #     print(t[0])
            #
            # path2 = 'data/MSTAR/2'
            # filenames2 = os.listdir(path2)
            # for filename in filenames2:
            #     img = cv.imread(os.path.join(path2, filename))
            #     transf = self.transform_SAR
            #     img_tensor = transf(Image.fromarray(img))
            #     print(2)
            #     print(filename)
            #     temp = img_tensor.squeeze(0)
            #     t = [temp[:, i] for i in range(100)]
            #     print(t[0])
            #
            # path4 = 'data/MSTAR/4'
            # filenames4 = os.listdir(path4)
            # for filename in filenames4:
            #     img = cv.imread(os.path.join(path4, filename))
            #     transf = self.transform_SAR
            #     img_tensor = transf(Image.fromarray(img))
            #     print(4)
            #     print(filename)
            #     temp = img_tensor.squeeze(0)
            #     t = [temp[:, i] for i in range(100)]
            #     print(t[0])

            few_selected_label = random.Random(self.seed).sample(range(self.SAR_num_labels), self.nway)
            # few_selected_label = [0,1,9]
            print('selected labeled', few_selected_label)

            full_data_dict = {}
            few_data_dict = {}
            train_dataset = datasets.ImageFolder(root=os.path.join(root, "MSTAR"), transform=self.transform_SAR)

            train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True)
            for i, (data, label) in enumerate(train_loader):
                label = label.item()
                data = data.squeeze(0)
                if label in few_selected_label:
                    data_dict = few_data_dict
                else:
                    data_dict = full_data_dict

                if label not in data_dict:
                    data_dict[label] = [data]
                else:
                    data_dict[label].append(data)

        else:
            raise NotImplementedError

        return full_data_dict, few_data_dict

    def maml_task_sample(self, train=True, nway=3, num_shots=1):
        if train:
            data_dict = self.full_data_dict
        else:
            data_dict = self.few_data_dict

        spt_x = None
        qry_x = None
        spt_y = None
        qry_y = None
        for i in range(nway):
            if i == 0:
                spt_y = torch.tensor([i])
                qry_y = torch.tensor([i])
                for j in range(num_shots-1):
                    s_tmp = torch.tensor([i])
                    spt_y = torch.cat((spt_y, s_tmp), dim=0)
            else:
                q_tmp = torch.tensor([i])
                qry_y = torch.cat([qry_y, q_tmp], dim=0)
                for j in range(num_shots):
                    s_tmp = torch.tensor([i])
                    spt_y = torch.cat([spt_y, s_tmp], dim=0)


        sampled_classes = random.sample(data_dict.keys(), nway)

        for j, _class in enumerate(sampled_classes):
            sampled_data = random.sample(data_dict[_class], num_shots + 1)

            # cur_spt = sampled_data[0]
            # cur_qry = sampled_data[1]
            # # print("category:", i, "numbers:", j, k)

            if j == 0:
                qry_x = sampled_data[num_shots].unsqueeze(0)
                spt_x = sampled_data[num_shots-1].unsqueeze(0)
                for k in range(num_shots-1):
                    spt_x = torch.cat([spt_x, sampled_data[k].unsqueeze(0)], dim=0)
            else:
                qry_x = torch.cat([qry_x, sampled_data[num_shots].unsqueeze(0)], dim=0)
                for k in range(num_shots):
                    spt_x = torch.cat([spt_x, sampled_data[k].unsqueeze(0)], dim=0)
        # print(spt_x.shape, spt_y.shape, qry_x.shape, qry_y.shape)
        shuffle_index = torch.randperm(num_shots * nway)
        spt_x = spt_x[shuffle_index]
        spt_y = spt_y[shuffle_index]
        return spt_x, spt_y, qry_x, qry_y

    def maml_cnn_task_sample(self, train=True, nway=3, num_shots=1, classes=list()):
        if train:
            data_dict = self.full_data_dict
        else:
            data_dict = self.few_data_dict

        spt_x = None
        qry_x = None
        spt_y = None
        qry_y = None
        for i in range(nway):
            if i == 0:
                spt_y = torch.tensor([i])
                qry_y = torch.tensor([i])
                for j in range(num_shots - 1):
                    s_tmp = torch.tensor([i])
                    spt_y = torch.cat((spt_y, s_tmp), dim=0)
            else:
                q_tmp = torch.tensor([i])
                qry_y = torch.cat([qry_y, q_tmp], dim=0)
                for j in range(num_shots):
                    s_tmp = torch.tensor([i])
                    spt_y = torch.cat([spt_y, s_tmp], dim=0)

        sampled_classes = classes

        for j, _class in enumerate(sampled_classes):
            sampled_data = random.sample(data_dict[_class], num_shots + 1)

            # cur_spt = sampled_data[0]
            # cur_qry = sampled_data[1]
            # # print("category:", i, "numbers:", j, k)

            if j == 0:
                qry_x = sampled_data[num_shots].unsqueeze(0)
                spt_x = sampled_data[num_shots - 1].unsqueeze(0)
                for k in range(num_shots - 1):
                    spt_x = torch.cat([spt_x, sampled_data[k].unsqueeze(0)], dim=0)
            else:
                qry_x = torch.cat([qry_x, sampled_data[num_shots].unsqueeze(0)], dim=0)
                for k in range(num_shots):
                    spt_x = torch.cat([spt_x, sampled_data[k].unsqueeze(0)], dim=0)
        # print(spt_x.shape, spt_y.shape, qry_x.shape, qry_y.shape)
        shuffle_index = torch.randperm(num_shots * nway)
        spt_x = spt_x[shuffle_index]
        spt_y = spt_y[shuffle_index]
        return spt_x, spt_y

    def load_batch_data(self, train=True, batch_size=16, nway=5, num_shots=1):
        if train:
            data_dict = self.full_data_dict
        else:
            data_dict = self.few_data_dict

        x = []
        label_y = [] # fake label: from 0 to (nway - 1)
        one_hot_y = [] # one hot for fake label
        class_y = [] # real label

        xi = []
        label_yi = []
        one_hot_yi = []
        

        map_label2class = []

        ### the format of x, label_y, one_hot_y, class_y is 
        ### [tensor, tensor, ..., tensor] len(label_y) = batch size
        ### the first dimension of tensor = num_shots

        for i in range(batch_size):

            # sample the class to train
            sampled_classes = random.sample(data_dict.keys(), nway)

            positive_class = random.randint(0, nway - 1)

            label2class = torch.LongTensor(nway)

            single_xi = []
            single_one_hot_yi = []
            single_label_yi = []
            single_class_yi = []


            for j, _class in enumerate(sampled_classes):
                if j == positive_class:
                    ### without loss of generality, we assume the 0th 
                    ### sampled  class is the target class
                    sampled_data = random.sample(data_dict[_class], num_shots+1)
                    x.append(sampled_data[0])
                    label_y.append(torch.LongTensor([j]))

                    one_hot = torch.zeros(nway)
                    one_hot[j] = 1.0
                    one_hot_y.append(one_hot)

                    class_y.append(torch.LongTensor([_class]))

                    shots_data = sampled_data[1:]
                else:
                    shots_data = random.sample(data_dict[_class], num_shots)

                single_xi += shots_data
                single_label_yi.append(torch.LongTensor([j]).repeat(num_shots))
                one_hot = torch.zeros(nway)
                one_hot[j] = 1.0
                single_one_hot_yi.append(one_hot.repeat(num_shots, 1))

                label2class[j] = _class

            shuffle_index = torch.randperm(num_shots*nway)
            xi.append(torch.stack(single_xi, dim=0)[shuffle_index])
            label_yi.append(torch.cat(single_label_yi, dim=0)[shuffle_index])
            one_hot_yi.append(torch.cat(single_one_hot_yi, dim=0)[shuffle_index])

            map_label2class.append(label2class)

        # #x 指 test set 数据，label_y 是它对应的 label（注意，并不是在原数据集中的分类，而是在本次 task 中的分类序号），
        # #one_hot_y 是 label_y 对应的 one_hot 编码，class_y 是它对应于原数据集的分类；
        # #xi 值 support set 数据，其它带 i 的数据项，也都是指 support set 的对应部分，其解释与不带 i 的相同。
        return [torch.stack(x, 0), torch.cat(label_y, 0), torch.stack(one_hot_y, 0), \
            torch.cat(class_y, 0), torch.stack(xi, 0), torch.stack(label_yi, 0), \
            torch.stack(one_hot_yi, 0), torch.stack(map_label2class, 0)]

    # def load_batch_data(self, train=True, batch_size=16, nway=5, num_shots=1):

    #     if train:
    #         data_dict = self.full_data_dict
    #     else:
    #         data_dict = self.few_data_dict

    #     x = torch.zeros(batch_size, self.input_channels, self.size, self.size)
    #     label_y = torch.LongTensor(batch_size).zero_()
    #     one_hot_y = torch.zeros(batch_size, nway)
    #     class_y = torch.LongTensor(batch_size).zero_()
    #     xi, label_yi, one_hot_yi, class_yi = [], [], [], []

    #     for i in range(nway*num_shots):
    #         xi.append(torch.zeros(batch_size, self.input_channels, self.size, self.size))
    #         label_yi.append(torch.LongTensor(batch_size).zero_())
    #         one_hot_yi.append(torch.zeros(batch_size, nway))
    #         class_yi.append(torch.LongTensor(batch_size).zero_())

    #     # sample data

    #     for i in range(batch_size):

    #         # sample the class to train
    #         sampled_classes = random.sample(data_dict.keys(), nway)

    #         positive_class = random.randint(0, nway - 1)

    #         indexes_perm = np.random.permutation(nway * num_shots)

    #         counter = 0

    #         for j, _class in enumerate(sampled_classes):
    #             if j == positive_class:
    #                 ### without loss of generality, we assume the 0th 
    #                 ### sampled  class is the target class
    #                 sampled_data = random.sample(data_dict[_class], num_shots+1)

    #                 x[i] = sampled_data[0]
    #                 label_y[i] = j

    #                 one_hot_y[i, j] = 1.0

    #                 class_y[i] = _class

    #                 shots_data = sampled_data[1:]
    #             else:
    #                 shots_data = random.sample(data_dict[_class], num_shots)

    #             for s_i in range(0, len(shots_data)):
    #                 xi[indexes_perm[counter]][i] = shots_data[s_i]
                    
    #                 label_yi[indexes_perm[counter]][i] = j
    #                 one_hot_yi[indexes_perm[counter]][i, j] = 1.0
    #                 class_yi[indexes_perm[counter]][i] = _class

    #                 counter += 1
    #     return [x, label_y, one_hot_y, class_y, torch.stack(xi, 1), torch.stack(label_yi, 1), \
    #         torch.stack(one_hot_yi, 1), torch.stack(class_yi, 1)]

    def load_tr_batch(self, batch_size=16, nway=5, num_shots=1):
        return self.load_batch_data(True, batch_size, nway, num_shots)

    def load_te_batch(self, batch_size=16, nway=5, num_shots=1):
        return self.load_batch_data(False, batch_size, nway, num_shots)

    def get_data_list(self, data_dict):
        data_list = []
        label_list = []
        for i in data_dict.keys():
            for data in data_dict[i]:
                data_list.append(data)
                label_list.append(i)

        now_time = time.time()

        random.Random(now_time).shuffle(data_list)
        random.Random(now_time).shuffle(label_list)

        return data_list, label_list

    def get_full_data_list(self):
        return self.get_data_list(self.full_data_dict)

    def get_few_data_list(self):
        return self.get_data_list(self.few_data_dict)

if __name__ == '__main__':
    D = self_DataLoader('data', True)

    # data = D.load_tr_batch(batch_size=16, nway=5, num_shots=5)
    # print(data[7].size())
    # list_classes = [data[7][i, :] for i in range(16)]
    # for i in range(16):
    #     list_classes[i] = list_classes[i].tolist()
    # print(list_classes[0])

    [x, label_y, one_hot_y, class_y, xi, label_yi, one_hot_yi, class_yi] = \
        D.load_tr_batch(batch_size=16, nway=5, num_shots=5)
    # [spt_x, spt_y, qry_x, qry_y] = \
    #     D.maml_task_sample()
    # print(spt_x.size(), spt_y.size(), qry_x.size(), qry_y.size())
    # print(spt_y, qry_y)
    print(x.size(), label_y.size(), one_hot_y.size(), class_y.size())
    print(xi.size(), label_yi.size(), one_hot_yi.size(), class_yi.size())


    xi_s = [xi[i, :, :, :, :] for i in range(16)]
    label_yi_s = [label_yi[i, :] for i in range(16)]
    one_hot_yi_s = [one_hot_yi[i, :, :] for i in range(16)]
    for i in range(16):
        xi_s[i] = [xi_s[i][j, :, :, :] for j in range(25)]
        label_yi_s[i] = [label_yi_s[i][j] for j in range(25)]
        one_hot_yi_s[i] = [one_hot_yi_s[i][j, :] for j in range(25)]
        for k in range(5):
            one_hot = torch.zeros(5)
            one_hot[k] = 1.0
            xi_p = torch.zeros(3, 32, 32)
            label = torch.tensor(k)
            for l in range(25):
                if label_yi_s[i][l] == k:
                    xi_p = xi_p + xi_s[i][l]
                    print(xi_s[i][l])
                    print(xi_p)
            xi_s[i].append(xi_p)
            one_hot_yi_s[i].append(one_hot)
            label_yi_s[i].append(label)
        xi_s[i] = torch.stack(xi_s[i], dim=0)
        label_yi_s[i] = torch.stack(label_yi_s[i], dim=0)
        one_hot_yi_s[i] = torch.stack(one_hot_yi_s[i], dim=0)
    xi_s = torch.stack(xi_s, dim=0)
    label_yi_s = torch.stack(label_yi_s, dim=0)
    one_hot_yi_s = torch.stack(one_hot_yi_s, dim=0)

    print(xi_s.size(), label_yi_s.size(), one_hot_yi_s.size())

    # xi_sum = [xi[i, :, :, :, :] for i in range(self.batchsize)]
    # for j in range(self.batchsize):
    #     zi_s[j] = self.cnn_feature[j](xi_sum[j])
    # zi_s = torch.stack(zi_s, dim=0)
    # # print(zi_s.size())
    # # torch.Size([16, 5, 64])
    #
    # # print(label_y)
    # # print(one_hot_y)
    #
    # print(label_yi[0])
    # print(one_hot_yi[0])