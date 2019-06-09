
# coding: utf-8

# In[ ]:

#hangul dataset
from PIL import Image
import numpy as np
import pandas as pd

from generate_function import *


# ### 자주 쓰이는 한글 낱말

# In[ ]:

df = pd.read_csv('../data/commonly_used_hangul.csv', engine='python')
use_chars = list(set(''.join(df.낱말)))
# sort list use_char
use_chars.sort()
print(len(use_chars))


# ### 학습 데이터 만들기

# In[ ]:

# 글자의 변형
# 서체 이름
fonts = ['HANBatang', 'HBATANG', 'UNI_HSR', 'HANDotum', 'HDOTUM', 'malgun']
# 위치
location = ['c','l','r','b','t','l2','r2']
# 크기
point =  list(range(80,81))
# 회전
angles = [-30, 0, 30]
# 이미지 사이즈
size = 100


# In[ ]:

tune_inputs = [(f, l, a, p, c) 
               for f in fonts
               for l in location
               for a in angles
               for p in point
               for c in use_chars]


# In[ ]:

data_X    = np.zeros((len(tune_inputs), 1, size, size), dtype=np.uint8)
data_Y    = np.zeros((len(tune_inputs), 1), dtype=np.uint8)
data_info = np.zeros((len(tune_inputs), 3), dtype=np.uint8)
print(data_X.shape, data_Y.shape, data_info.shape)


# In[ ]:

# 데이터셋 만들기
for idx, tune_input in enumerate(tune_inputs):
    f, l, a, p, c = tune_input
    Z = face2array(f, l, a, p, c, size)
    # Z scale to 0~1
    data_X[idx,:]     = Z/np.float32(255.0)
    data_Y[idx,:]     = seri_sansserif(f)
    data_info[idx, :] = hangul2info(c)


# In[ ]:

# 데이터 저장
Dataset = {'dataX'   : data_X,
           'dataY'   : data_Y,
           'datainfo': data_info}
save_dataset(folder='../data', typ='train', data=Dataset)


# ### 테스트 데이터

# In[ ]:

# 전체 문자
cho_lst   = list(range(0,19))
jung_lst = list(range(0,21))
jong_lst  = list(range(0,28))

hanguls = [44032 + (c * 588) + (j1 * 28) + j2
          for c in cho_lst
          for j1 in jung_lst
          for j2 in jong_lst]


# In[ ]:

# 사용한 문자의 유니코드
use_hanguls = []
for c in use_chars:
    cho, jung, jong = hangul2info(c)
    use_hanguls.append(44032 + (cho * 588) + (jung * 28) + jong)


# In[ ]:

# 학습하지 않은 문자
not_use_chars = list(set(hanguls).difference(set(use_hanguls)))


# ### 테스트 1

# In[ ]:

# 글자의 변형
# 서체 이름
fonts = ['HANDotum', 'HDOTUM', 'malgun', 'HANBatang', 'HBATANG', 'UNI_HSR']
# 위치
location = ['c','l','r','b','t','l2','r2']
# 크기
point =  list(range(80,81))
# 회전
angles = [-30, 0, 30]
# 이미지 사이즈
size = 100


# In[ ]:

tune_inputs = [(f, l, a, p, c) 
               for f in fonts
               for l in location
               for a in angles
               for p in point
               for c in not_use_chars]


# In[ ]:

np.random.seed(2019)
sel_idx = np.random.choice(range(len(tune_inputs)), 30000)
tune_inputs = [tune_inputs[i] for i in sel_idx]


# In[ ]:

data_X    = np.zeros((len(tune_inputs), 1, size, size), dtype=np.uint8)
data_Y    = np.zeros((len(tune_inputs), 1), dtype=np.uint8)
data_info = np.zeros((len(tune_inputs), 3), dtype=np.uint8)
print(data_X.shape, data_Y.shape, data_info.shape)


# In[ ]:

# 데이터셋 만들기
for idx, tune_input in enumerate(tune_inputs):
    f, l, a, p, c = tune_input
    Z = face2array(f, l, a, p, c, size)
    # Z scale to 0~1
    data_X[idx,:]     = Z/np.float32(255.0)
    data_Y[idx,:]     = seri_sansserif(f)
    data_info[idx, :] = hangul2info(c)


# In[ ]:

# 데이터 저장
Dataset = {'dataX'   : data_X,
           'dataY'   : data_Y,
           'datainfo': data_info}
save_dataset(folder='../data', typ='test1', data=Dataset)


# In[ ]:




# ### 테스트 2

# In[ ]:

# 글자의 변형
# 서체 이름
fonts = ['NanumBarunGothic', 'NanumGothic', 'HMKMM', 'NanumMyeongjo']
# 위치
location = ['c','l','r','b','t','l2','r2']
# 크기
point =  list(range(80,81))
# 회전
angles = [-30, 0, 30]
# 이미지 사이즈
size = 100


# In[ ]:

tune_inputs = [(f, l, a, p, c) 
               for f in fonts
               for l in location
               for a in angles
               for p in point
               for c in not_use_chars]


# In[ ]:

np.random.seed(2019)
sel_idx = np.random.choice(range(len(tune_inputs)), 30000)
tune_inputs = [tune_inputs[i] for i in sel_idx]


# In[ ]:

data_X    = np.zeros((len(tune_inputs), 1, size, size), dtype=np.uint8)
data_Y    = np.zeros((len(tune_inputs), 1), dtype=np.uint8)
data_info = np.zeros((len(tune_inputs), 3), dtype=np.uint8)
print(data_X.shape, data_Y.shape, data_info.shape)


# In[ ]:

# 데이터셋 만들기
for idx, tune_input in enumerate(tune_inputs):
    f, l, a, p, c = tune_input
    Z = face2array(f, l, a, p, c, size)
    # Z scale to 0~1
    data_X[idx,:]     = Z/np.float32(255.0)
    data_Y[idx,:]     = seri_sansserif(f)
    data_info[idx, :] = hangul2info(c)


# In[ ]:

# 데이터 저장
Dataset = {'dataX'   : data_X,
           'dataY'   : data_Y,
           'datainfo': data_info}
save_dataset(folder='../data', typ='test2', data=Dataset)


# In[ ]:




# ### 테스트 3

# In[ ]:

# 글자의 변형
# 서체 이름
fonts = ['gulim', 'dotum', 'batang', 'HYGSRB']
# 위치
location = ['c','l','r','b','t','l2','r2']
# 크기
point =  list(range(80,81))
# 회전
angles = [-30, 0, 30]
# 이미지 사이즈
size = 100


# In[ ]:

tune_inputs = [(f, l, a, p, c) 
               for f in fonts
               for l in location
               for a in angles
               for p in point
               for c in not_use_chars]


# In[ ]:

np.random.seed(2019)
sel_idx = np.random.choice(range(len(tune_inputs)), 30000)
tune_inputs = [tune_inputs[i] for i in sel_idx]


# In[ ]:

data_X    = np.zeros((len(tune_inputs), 1, size, size), dtype=np.uint8)
data_Y    = np.zeros((len(tune_inputs), 1), dtype=np.uint8)
data_info = np.zeros((len(tune_inputs), 3), dtype=np.uint8)
print(data_X.shape, data_Y.shape, data_info.shape)


# In[ ]:

# 데이터셋 만들기
for idx, tune_input in enumerate(tune_inputs):
    f, l, a, p, c = tune_input
    Z = face2array(f, l, a, p, c, size)
    # Z scale to 0~1
    data_X[idx,:]     = Z/np.float32(255.0)
    data_Y[idx,:]     = seri_sansserif(f)
    data_info[idx, :] = hangul2info(c)


# In[ ]:

# 데이터 저장
Dataset = {'dataX'   : data_X,
           'dataY'   : data_Y,
           'datainfo': data_info}
save_dataset(folder='../data', typ='test3', data=Dataset)

