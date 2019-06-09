from PIL import Image
import numpy as np
import pandas as pd
from freetype import *
import pickle
import matplotlib.pyplot as plt

# 한글 서체를 array로 만들기
def face2array(f, l, a, p, c, size):
    if f=='gulim':
        face = Face('../font/{}.ttc'.format(f))
    else:
        face = Face('../font/{}.ttf'.format(f))
    face.set_char_size(p*64)
    
    # 회전 각도
    angle = (a/180.0)*np.pi
    matrix = FT_Matrix((int) (np.cos(angle) * 0x10000),
                       (int) (-np.sin(angle) * 0x10000),
                       (int) (np.sin(angle) * 0x10000),
                       (int) (np.cos(angle) * 0x10000),
                      )
    # matrix에 글자 나타내기
    flags = FT_LOAD_RENDER
    pen   = FT_Vector(0,0)
    FT_Set_Transform(face._FT_Face, byref(matrix), byref(pen))

    char   = face.get_char_index(c)
    face.load_glyph(char)
    slot   = face.glyph
    bitmap = slot.bitmap
    data, rows, width = bitmap.buffer, bitmap.rows, bitmap.width
    # 한글 낱자를 array형태로 만들기
    Z = np.array(data, dtype=np.uint8).reshape(rows, width)
    # padding function
    Z      = padding(l, Z, size, rows, width)
    return Z

# array에 padding해주기
def padding(l, Z, size, rows, width):
    if l =='c':
        Z = 255-np.pad(Z, (((size-rows)//2 + rows%2 ,(size-rows)//2),
                           ((size-width)//2+width%2, (size-width)//2)), 
                       'constant', constant_values=0)
    elif l =='l':
        Z = 255-np.pad(Z, ((0,(size-rows)),(0,(size-width))),
               'constant', constant_values=0)
    elif l =='r':
        Z = 255-np.pad(Z, (((size-rows),0),((size-width),0)),
               'constant', constant_values=0)
    elif l =='b':
        Z = np.delete(Z, np.s_[:20],0)
        rows, width = Z.shape[0], Z.shape[1]
        Z = 255-np.pad(Z, ((0 ,(size-rows)),
          ((size-width)//2+width%2, (size-width)//2)),
              'constant', constant_values=0)
    elif l =='t':
        Z = np.delete(Z, np.s_[50:],0)
        rows, width = Z.shape[0], Z.shape[1]
        Z = 255-np.pad(Z, (((size-rows),0), (0, (size-width))),
               'constant', constant_values=0)
    elif l =='l2':
        Z = np.delete(Z, np.s_[40:],1)
        rows, width = Z.shape[0], Z.shape[1]
        Z = 255-np.pad(Z, ((0 ,(size-rows)),((size-width),0)),
             'constant', constant_values=0)
    elif l =='r2':
        Z = np.delete(Z, np.s_[:20],1)
        rows, width = Z.shape[0], Z.shape[1]
        Z = 255-np.pad(Z, (((size-rows)//2 + rows%2 ,(size-rows)//2),
                   (0, (size-width))),
               'constant', constant_values=0)
    return(Z)



# 한글 낱자 입력 받아, 초중종으로 해체하기
def isHangeul(one_character):
    return 0xAC00 <= ord(one_character[:1]) <= 0xD7A3

def hangeulExplode(one_hangeul):
    chosung = ["ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ",
           "ㅃ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ",
           "ㅌ", "ㅍ", "ㅎ"]
    jungsung = ["ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ",
                "ㅗ", "ㅘ", "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ",
                "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ"]
    jongsung = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ",
                "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ",
                "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ",
                "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
    a = one_hangeul[:1]
    if isHangeul(a) != True:
        return False
    b    = ord(a) - 0xAC00
    cho  = b // (21*28)
    jung = b % (21*28) // 28
    jong = b % 28
    if jong == 0:
        return (chosung[cho], jungsung[jung], "")
    else:
        return (chosung[cho], jungsung[jung], jongsung[jong])
    
def pureosseugi(text):
    result = ""
    for i in text:
        if isHangeul(i) == True:
            for j in hangeulExplode(i):
                result += j
        else:
            result += i
    return result

def hangul2info(c):
    chosung = ["ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ",
           "ㅃ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ",
           "ㅌ", "ㅍ", "ㅎ"]
    jungsung = ["ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ",
                "ㅗ", "ㅘ", "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ",
                "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ"]
    jongsung = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ",
                "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ",
                "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ",
                "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
    
    cho, jung, jong = hangeulExplode(c)
    return (chosung.index(cho), jungsung.index(jung), jongsung.index(jong))

# data_Y: serif = 1 / sans_serif = 0
def seri_sansserif(f):
    serif      = ['HANBatang', 'HBATANG', 'UNI_HSR',
                 'HMKMM', 'NanumMyeongjo',
                 'batang', 'HYGSRB']
    sans_serif = ['HANDotum', 'HDOTUM', 'malgun',
                 'NanumBarunGothic', 'NanumGothic',
                 'gulim', 'dotum']

    if f in serif:
        return(1)
    else:
        return(0)
    
# Save function
def save_dataset(folder, typ, data):
    print('Save dataset to directory: ', os.path.join(folder))
    f = open(os.path.join(folder, '{}_data.pkl'.format(typ)),"wb")
    pickle.dump(data, f)
    f.close()
    
# 문자 출력
def char_show(dataset, length, idx):
    plt.subplot(1, length, idx+1)
    fig = plt.imshow(dataset[idx, 0, :,:], cmap="gray")
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
