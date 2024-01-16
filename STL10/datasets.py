# -*- coding: utf-8 -*-
# @Date    : 2019-07-25
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0
import numpy as np
import torch
import torch.distributed as dist
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class ImageDataset(object):
    def __init__(self, args, cur_img_size=None):
        img_size = cur_img_size if cur_img_size else args.img_size
        if args.dataset.lower() == "cifar10":
            Dt = datasets.CIFAR10
            transform = transforms.Compose(
                [
                    transforms.Resize(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            args.n_classes = 10
        elif args.dataset.lower() == "stl10":
            Dt = datasets.STL10
            transform = transforms.Compose(
                [
                    transforms.Resize(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        else:
            raise NotImplementedError("Unknown dataset: {}".format(args.dataset))

        if args.dataset.lower() == "stl10":
            self.train = torch.utils.data.DataLoader(
                Dt(
                    root=args.data_path,
                    split="train+unlabeled",
                    transform=transform,
                    download=True,
                ),
                batch_size=args.dis_batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
            )

            self.valid = torch.utils.data.DataLoader(
                Dt(root=args.data_path, split="test", transform=transform),
                batch_size=args.dis_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            self.test = self.valid

        else:
            self.train = torch.utils.data.DataLoader(
                Dt(root=args.data_path, train=True, transform=transform, download=True),
                batch_size=args.dis_batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
            )

            self.valid = torch.utils.data.DataLoader(
                Dt(root=args.data_path, train=False, transform=transform),
                batch_size=args.dis_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )

            self.test = self.valid


class NewsDataset(Dataset):
    def __init__(self, images):
        self.images = images
    
    # 读取单个样本
    def __getitem__(self, idx):
        item = self.images[idx]
        return item
    
    def __len__(self):
        return len(self.images)


class ImageDataset_bootstrap_bags(object):
    def __init__(self, args, cur_img_size=None):
        img_size = cur_img_size if cur_img_size else args.img_size
        if args.dataset.lower() == "cifar10":
            if args.Aug_Flip:
                train_CIFAR10_images = np.transpose(np.load("./data/train_CIFAR10_Aug_Flip.npy"), (0, 3, 1, 2))             # [100000, 32, 32, 3] -> [100000, 3, 32, 32], pixel_value: [-1, 1]
            else:
                train_CIFAR10_images = np.transpose(np.load("./data/train_CIFAR10.npy"), (0, 3, 1, 2))             # [50000, 32, 32, 3] -> [50000, 3, 32, 32], pixel_value: [-1, 1]
            test_CIFAR10_images = np.transpose(np.load("./data/test_CIFAR10.npy"), (0, 3, 1, 2))               # [10000, 32, 32, 3] -> [10000, 3, 32, 32], pixel_value: [-1, 1]

            args.n_classes = 10

            if args.dataset_reduce:
                if args.num_disc == 1:
                    bootstrap_bags_images = [train_CIFAR10_images]
                else:
                    # (1-1/e) = 0.64; 
                    number_sample_in_bag = int(len(train_CIFAR10_images) * 0.64)
                    stride = int(len(train_CIFAR10_images) * 0.36 / (args.num_disc-1))
                    bootstrap_bags_images = []
                    for i in range(args.num_disc):
                        # indexes = list(np.arange(i*stride, i*stride + number_sample_in_bag))
                        bootstrap_bag = train_CIFAR10_images[i*stride : i*stride + number_sample_in_bag]
                        bootstrap_bags_images.append(bootstrap_bag)
        
            else:
                if args.num_disc == 1:
                    bootstrap_bags_images = [train_CIFAR10_images]
                    
                else:
                    if args.num_disc == 2:
                        indices = []
                        r = np.random.RandomState(args.random_state)
                        section_length = int(len(train_CIFAR10_images) * 0.66)
                        indice = r.randint(0, section_length, len(train_CIFAR10_images)).tolist()
                        print("----- rank: {}, indice: {} -----".format(dist.get_rank(), indice[0:5]))
                        indices.append(indice)

                        indice = r.randint(len(train_CIFAR10_images)-section_length, len(train_CIFAR10_images), len(train_CIFAR10_images)).tolist()
                        print("----- rank: {}, indice: {} -----".format(dist.get_rank(), indice[0:5]))
                        indices.append(indice)
                        bootstrap_bags_images = train_CIFAR10_images[np.array(indices)]

                    else:
                        r = np.random.RandomState(args.random_state)
                        indices = r.randint(0, len(train_CIFAR10_images), (args.num_disc, len(train_CIFAR10_images)))
                        bootstrap_bags_images = train_CIFAR10_images[indices]
        
            self.trains = []
            for index in range(args.num_disc):
                dataset_ = NewsDataset(bootstrap_bags_images[index])
                dataset_sampler = torch.utils.data.distributed.DistributedSampler(dataset_)
                dataloader = torch.utils.data.DataLoader(
                    dataset_, 
                    batch_size=args.dis_batch_size, 
                    num_workers=args.num_workers, 
                    sampler=dataset_sampler) 

                self.trains.append(dataloader)
            self.test = []
            self.valid = self.test

        elif args.dataset.lower() == "stl10":
            train_STL10_images = np.transpose(np.load("./data/train_STL10.npy"), (0, 3, 1, 2))             # [105000, 48, 48, 3] -> [105000, 3, 48, 48], pixel_value: [-1, 1]
            
            if args.train_number is None:
                pass
            else:
                train_STL10_images = train_STL10_images[:args.train_number]

            if args.num_disc == 1:
                bootstrap_bags_images = [train_STL10_images]
            else:
                if args.Model_type == "Forest_GAN":
                    print("-----> {}; {} <-----".format(args.Model_type, len(train_STL10_images)))
                    if args.num_disc == 2:
                        indices = []
                        r = np.random.RandomState(args.random_state)
                        section_length = int(len(train_STL10_images)*0.66)
                        indice = r.randint(0, section_length, len(train_STL10_images)).tolist()

                        indice = r.randint(0, len(train_STL10_images), len(train_STL10_images)).tolist()
                        indices.append(indice)

                        all_index = set(np.arange(len(train_STL10_images)))
                        residue_indice = all_index - set(indice)
                        indice = r.randint(0, len(train_STL10_images), len(train_STL10_images)-len(residue_indice)).tolist() + list(residue_indice)
                        indices.append(indice)
                        bootstrap_bags_images = train_STL10_images[np.array(indices)]
                        
                        for index in indices:
                            print("---> length: {}; set: {}<---".format(len(index), len(list(set(index)))))
                
                    else:
                        r = np.random.RandomState(args.random_state)
                        indices = r.randint(0, len(train_STL10_images), (args.num_disc, len(train_STL10_images)))
                        bootstrap_bags_images = train_STL10_images[indices]
                        for index in indices:
                            print("---> length: {}; set: {}<---".format(len(index), len(list(set(index.tolist())))))

                elif args.Model_type == "MIX_GAN":
                    print("+++++++++++++- {} -+++++++++++".format(args.Model_type))
                    bootstrap_bags_images = []
                    for index in range(args.num_disc):
                        bootstrap_bags_images.append(train_STL10_images)
                        print("---> length: {} <---".format(len(train_STL10_images)))

                elif args.Model_type == "PAR_GAN":
                    print("+++++++++++++- {} -+++++++++++".format(args.Model_type))
                    bootstrap_bags_images = []
                    total_number = len(train_STL10_images)
                    gap_number = int(total_number / args.num_disc)
                    for index in range(args.num_disc):
                        images = train_STL10_images[index*gap_number : min(total_number, (index+1)*gap_number)]
                        bootstrap_bags_images.append(images)
                        print("---> length: {} <---".format(len(images)))
                else:
                    raise "Model_type not exist: {}".format(args.Model_type)
                
            self.trains = []
            for index in range(args.num_disc):
                dataset_ = NewsDataset(bootstrap_bags_images[index])
                dataset_sampler = torch.utils.data.distributed.DistributedSampler(dataset_)
                dataloader = torch.utils.data.DataLoader(
                    dataset_, 
                    batch_size=args.dis_batch_size, 
                    num_workers=args.num_workers, 
                    sampler=dataset_sampler
                ) 
                self.trains.append(dataloader)
            pass
        