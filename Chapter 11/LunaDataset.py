import concurrent.futures
import pandas as pd 
import tqdm
import glob
# import time
import torch 
import Ct
# import os
import accessify 
from functools import lru_cache
from torch.utils.data import Dataset
# from multi_process import select_item


@lru_cache(1, typed=True)
def getCt(series_uid):
    return Ct.Ct(series_uid)

@lru_cache
def getCtRawCandidate(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    return ct_chunk, center_irc

@lru_cache
def get_CandidateLists(file_csv):
        file_paths = pd.read_csv(file_csv)
        return [elems[1] for elems in tqdm.tqdm(file_paths.iterrows(), total=file_paths.shape[0], desc=f'Initializing Data')]

# @lru_cache
# def get_CandidateLists():
#         return glob.glob('/media/e_quitee/Data Drive/LUNASet/subset*/subset*/*.mhd')


# this will allow us to train a classifier to identify nodules
class LunaDataset(Dataset):
    def __init__(self, val_stride=0, isValSet_bool=None, series_uid=None, transform=None):
        super().__init__()
        # this attribuite contains all the file paths of our data
        # recomended that you should use pandas for this instead
        # pass in the csv containing all path in formation--candidate indolist will, will house
        # all the paths 
        # self.files = 'Candidates_All_mhd_Paths.csv'
        # self.files = 'Candidates_mhd_Path_1.csv'
        # self.files = 'Candidates_All_mhd_Paths_Ubuntu.csv'
        self.files = 'Candidates_All_mhd_positive_Ubuntu.csv'
        self.candidateInfo_list = get_CandidateLists(self.files)
        # self.candidateInfo_list = get_CandidateLists()
        self.val_stride = val_stride
        self.isValSet_bool = isValSet_bool
        self.transform = transform
        self.series_uid = series_uid

        if self.series_uid:
            self.candidateInfo_list = [x for x in self.candidateInfo_list if x.splt('/')[-1][:-4] == self.series_uid]

        if self.isValSet_bool:
            assert self.val_stride > 0, self.val_stride
            self.candidateInfo_list = self.candidateInfo_list[::self.val_stride]
            assert self.candidateInfo_list

        elif self.val_stride > 0:
            del self.candidateInfo_list[::self.val_stride]
            assert self.candidateInfo_list

    # @accessify.private
    # def get_CandidateLists(self, file_csv):
    #     file_paths = pd.read_csv(file_csv)
    #     return [elems[1] for elems in tqdm.tqdm(file_paths.iterrows(), total=file_paths.shape[0], desc=f'Initializing {self.name} Data')]
    
    # @accessify.private
    # def get_CandidateLists(self, file_csv):
    #     file_paths = pd.read_csv(file_csv)
    #     with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
    #         # select_item = lambda x: x[1]
    #         series_list = list(tqdm.tqdm(executor.map(select_item, file_paths.iterrows(), chunksize=500), desc='Initializing Data', total=file_paths.shape[0])) 
    #     # return [elems[1] for elems in tqdm.tqdm(file_paths.iterrows(), total=file_paths.shape[0], desc='Initializing Data')]
    #     return series_list

    # # @lru_cache(1, typed=True)
    # def getCt(self, series_uid):
    #     return Ct.Ct(series_uid)
    
    # # @lru_cache
    # def getCtRawCandidate(self, series_uid, center_xyz, width_irc):
    #     ct = self.getCt(series_uid)
    #     ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    #     return ct_chunk, center_irc

    def __len__(self):
        return len(self.candidateInfo_list) 

    def __getitem__(self, index):
        candidateInfo_tup = self.candidateInfo_list[index]
        width_irc = (32, 48, 48) # the size of the tensor that we will be returning

        # here we need to pass in serie_uid for a sample
        # we also need to pass in coordX, coordY and coordZ 
        # these should be contained in our dataframes
        candidate_a, center_irc = getCtRawCandidate(
            candidateInfo_tup['seriesuid'],
            tuple(candidateInfo_tup[['coordX', 'coordY', 'coordZ']]),
            width_irc
        )

        candidate_t = torch.from_numpy(candidate_a)
        candidate_t = candidate_t.to(torch.float32)
        candidate_t = candidate_t.unsqueeze(0) # here we are adding a channel dimension

        # this is our label
        pos_t = torch.tensor([not candidateInfo_tup['class'], candidateInfo_tup['class']], dtype=torch.long, requires_grad=False)

        return  (candidate_t, pos_t, candidateInfo_tup['seriesuid'], torch.tensor(center_irc))

if __name__ == '__main__':
    pass