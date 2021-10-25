# Project: LSTM_module

LSTM을 활용한 보이스피싱 탐지 모듈 개발

## 순서
1. import: 필요한 모듈 import
2. 데이터 셋 불러오기(load)
2. 전처리: 학습에 필요한 데이터 전처리 수행
3. 모델링(model): 모델 정의
4. 컴파일(compile): 모델 생성
5. 학습(fit): 모델 학습

## 모듈 import
```python
import os

import json
import pickle
import re

import pandas as pd
import numpy as np

%matplotlib inline
import matplotlib.pyplot as plt

from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from collections import Counter

from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
```
## 파일 이름 선언

필요 변수, 모델, 토큰 객체 등 파일 이름 선언
```python
file_name = "LSTM_module_ver4.0"
```
## 데이터 셋 불러오기

빈 list 생성
```python
spam_X = []
spam_temp = []
ham_X = []
ham_temp = []
```
* `ham_count`: 일반 데이터 개수 제한
* `min_data`: 텍스트 데이터 최소 길이
* `data_length`: 텍스트 데이터 일반 길이
```python
ham_count = 10000
min_data = 10
data_length = 100
```
보이스피싱 통화 데이터 가져오기
```python
path = 'fraudDataset/spamData/'
path1 = os.listdir(path)
for a in path1:
    path2 = path+a
    with open(path2, 'r', encoding='utf-8') as f:
        contents = f.read()
        contents = re.sub("사회자 :|피해자 :|\((.+?)\)|네|[^가-힣 ]", "", contents) # 특수문자 제거
        spam_temp.append(contents)
```
일반 통화 데이터 가져오기
```python
contents = ""
path = 'fraudDataset/hamData_counselor/'
path1 = os.listdir(path)
for a in path1:
    
    if len(ham_temp) > ham_count:
        break
    
    path2 = os.listdir(path+a)
    for b in path2:
        
        if len(ham_temp) > ham_count:
            break
        
        path3 = os.listdir(path+a+'/'+b)
        for c in path3:
            
            if len(ham_temp) > ham_count:
                break
                    
            path4 = os.listdir(path+a+'/'+b+'/'+c)
            for d in path4:
                
                if len(ham_temp) > ham_count:
                    break
                
                path5 = os.listdir(path+a+'/'+b+'/'+c+'/'+d)
                for e in path5:
                    
                    if len(ham_temp) > ham_count:
                        break
                    
                    path6 = os.listdir(path+a+'/'+b+'/'+c+'/'+d+'/'+e)

                    if len(contents) > min_data:
                        ham_temp.append(contents)

                    contents = ""
                    for f in path6:
                        
                        if len(ham_temp) > ham_count:
                            break
                        
                        if f.split(".")[1] == "txt":
                            path7 = path+a+'/'+b+'/'+c+'/'+d+'/'+e+'/'+f
                        else:
                            continue
                        with open(path7, "r", encoding="UTF-8") as f:
                            text = f.read()
                            text = re.sub("#@(.+?)#|네|[^가-힣 ]", "", text) # 특수문자 제거

                        contents += text+" "
                        if len(contents) > data_length:
                            ham_temp.append(contents)
                            contents = ""
                            
ham_X = ham_temp
```
피싱 데이터 1개 출력
```python
spam_temp[0]
```
일반 데이터 5개 출력
```python
ham_temp[:5]
```
## 텍스트 데이터 분리

**findIndex**: 문장이 끝나는 부분의 index값을 찾기위한 함수
```python
def findIndex(data_list, split_list, start_index):
    index_list = []
    for i in split_list:
        index = data_list.find(i, start_index)
        index_list.append(index)
        
    index_list = [i for i in index_list if i not in [-1]]
    if index_list == []:
        return -1
    
    index = min(index_list)
    
    return index     
```
* `split_list`: 문장 단위로 분리하기 위한 글자 리스트
```python
split_list = ['다 ','요 ','죠 ', '까 ']
```
`data_length` 값 이후에 `split_list` 에 있는 문자가 나오면 자르기 (스팸 데이터)
```python
for t in spam_temp:
    index = findIndex(t, split_list, data_length)
        
    i = 0
    while index != -1:
        x = t[i:index+2]
        spam_X.append(x)
        
        i = index+2
        index = findIndex(t, split_list, i+data_length)
    else:
        x = t[i:]
        if len(x) > min_data:
            spam_X.append(x) # 텍스트 마지막 부분 추가
```
## 전처리: 데이터 길이 제한

길이가 min 미만 데이터 삭제 후 중복 제거
```python
spam_df = pd.DataFrame(spam_X, columns=['text'])
spam_df.drop(spam_df[spam_df['text'].str.len() < min_data].index, inplace=True) # 길이가 min_data 미만인 데이터 삭제
spam_df.shape[0] - spam_df['text'].nunique() # 중복 개수
```
```python
ham_df = pd.DataFrame(ham_X, columns=['text'])
ham_df.drop(ham_df[ham_df['text'].str.len() < min_data].index, inplace=True) # 길이가 min_data 미만인 데이터 삭제
ham_df.shape[0] - ham_df['text'].nunique() # 중복 개수
```
```python
spam_df.drop_duplicates(subset=['text'], inplace=True) # 중복 제거
spam_df['label'] = 1
```
```python
spam_df
```
```python
ham_df.drop_duplicates(subset=['text'], inplace=True) # 중복 제거
ham_df['label'] = 0
```
```python
ham_df
```
데이터 정보 출력
```python
print("스팸 데이터 개수 : {}(read) -> {}(drop)".format(len(spam_X),len(spam_df)))
print('스팸 데이터 최소 길이',len(min(spam_df['text'], key=len)))
print('스팸 데이터 최대 길이',len(max(spam_df['text'], key=len)))
print('----------------------------------------------------')
print("일반 데이터 개수 : {}(read) -> {}(drop)".format(len(ham_X),len(ham_df)))
print('일반 데이터 최소 길이',len(min(ham_df['text'], key=len)))
print('일반 데이터 최대 길이',len(max(ham_df['text'], key=len)))
```
스팸 / 일반 데이터 합치기
```python
total_df = pd.concat([spam_df, ham_df], ignore_index=True)
```
```python
total_df
```
## Train / Validation Set 분리

0.75 : 0.25 비율로 나누기
```python
train_data, test_data = train_test_split(total_df, test_size = 0.25, random_state = 42) # 훈련용 테스트 데이터 나누기
```
훈련 / 테스트 데이터 개수 출력
```python
print('데이터 총 개수 :',len(total_df))
print('훈련용 데이터 개수 :', len(train_data))
print('테스트용 데이터 개수 :', len(test_data))
```
스팸 / 일반 데이터 개수 그래프 비교
```python
plt.subplot(121)
train_data['label'].value_counts().plot(kind = 'bar', title='Count')
print(train_data.groupby('label').size().reset_index(name = 'count'))
```
## 전처리: 빈 데이터 처리

빈 데이터 NA 처리
```python
pd.set_option('mode.chained_assignment', None) # 경고 메시지 끄기

train_data.replace('', np.nan, inplace=True)
```
```python
test_data.replace('', np.nan, inplace=True)
```
전처리 후 생긴 null 데이터 확인
```python
print(train_data.isnull().sum()) # null 데이터 확인
print()
print(test_data.isnull().sum())
```
null 값이 있는 행 제거 후 확인
```python
train_data = train_data.dropna(how='any', axis=0)
test_data = test_data.dropna(how='any', axis=0)
```
```python
print(train_data.isnull().sum()) # null 데이터 확인
print()
print(test_data.isnull().sum())
```
## 전처리: 토큰화

Okt 객체 선언
```python
okt = Okt() # 한글은 형태소 분석기 사용해야됨 KoNPLY
```
불용어 단어
```python
stopwords = [] # spam, ham 빈도 수 높은 중복단어 제거
```
어간 단위로 토큰화 및 불용어 처리
```python
X_train = []
for sentence in train_data['text']:
    temp_X = okt.morphs(sentence, stem=True) # 토큰화
    temp_X = [word for word in temp_X if not word in stopwords] # 불용어 처리
    X_train.append(temp_X)
```
```python
X_test = []
for sentence in test_data['text']:
    temp_X = okt.morphs(sentence, stem=True) # 토큰화
    temp_X = [word for word in temp_X if not word in stopwords] # 불용어 처리
    X_test.append(temp_X)
```
Dataframe 새 Column에 추가
```python
train_data['tokenized'] = X_train
test_data['tokenized'] = X_test
```
## 단어별 빈도 수 확인

스팸 단어 빈도 확인
```python
spam_words = np.hstack(train_data[train_data.label == 1]['tokenized'].values)
spam_word_count = Counter(spam_words)
print(spam_word_count.most_common(20))
```
일상 단어 빈도 확인
```python
ham_words = np.hstack(train_data[train_data.label == 0]['tokenized'].values)
ham_word_count = Counter(ham_words)
print(ham_word_count.most_common(20))
```
Dataframe 값 추출
```python
X_train = train_data['tokenized'].values
y_train = train_data['label'].values
X_test = test_data['tokenized'].values
y_test = test_data['label'].values
```
## 정수 인코딩

`vocab_size`의 값을 정하기 위한 토큰화 객체 생성
```python
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
```
최소 단어 빈도 수 제한
```python
threshold = 2 # 등장 횟수가 threshold미만인 단어는 제거
```
최적의 `vocab_size` 선정
```python
word_to_index = tokenizer.word_index
total_cnt = len(word_to_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합
```
# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
```python
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value
        
print('단어 집합(vocabulary)의 크기 :',total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)

vocab_size = total_cnt - rare_cnt + 2
print('vocab_size :',vocab_size)
```
단어 토큰화
* `num_words`: 단어 max 사이즈 지정. 빈도수가 높은 순으로 저장.
* `oov_token`: 단어 토큰에 없는 단어 표기법 지정
```python
tokenizer = Tokenizer(num_words = vocab_size, oov_token = 'OOV')
```
`fit_on_texts`로 학습할 문장에 대한 토큰화 진행
```python
tokenizer.fit_on_texts(X_train)
```
```python
for key, value in tokenizer.word_index.items():
    if len(key) < 4:
        print('{}\t======>\t {}'.format(key, value))
    else:
        print('{}======>\t {}'.format(key, value))
    if value == 25:
        break
```
토큰화된 단어 사전의 개수
```python
len(tokenizer.word_index)
```
문장을 토큰으로 치환
```python
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
```
변환된 Sequences 확인
```python
X_train[:5]
```
`vocab_size`보다 떨어지는 단어는 1로 치환 (OOV)

## 패딩
리뷰 길이
```python
plt.subplot(122)
plt.hist([len(s) for s in X_train], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

print('최대 길이 :',max(len(l) for l in X_train))
print('평균 길이 :',sum(map(len, X_train))/len(X_train))
```
3가지 옵션 입력
* `maxlen`: 최대 문장 길이 정의. 길면 잘라냄
* `truncating`: 문장 길이가 `maxlen`보다 길 때, 앞을 자를지 뒤를 자를지 정의
* `padding`: 문장 길이가 `maxlen`보다 짧을 때, 앞을 채울지 뒤를 채울지 정의
```python
max_len = 150 # 최대 길이 (그래프를 보고 판단)
trunc_type = 'post' # 잘라낼 문장의 위치
padding_type = 'post'
```
문장 길이 비율 찾기
```python
def below_threshold_len(max_len, nested_list):
    cnt = 0
    for s in nested_list:
        if(len(s) <= max_len):
            cnt = cnt + 1
    print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (cnt / len(nested_list))*100))
    
below_threshold_len(max_len, X_train)
```
```python
X_train = pad_sequences(X_train, maxlen = max_len, truncating = trunc_type, padding = padding_type)
X_test = pad_sequences(X_test, maxlen = max_len, truncating = trunc_type, padding = padding_type)
```
변화된 후 shape 확인
```python
X_train.shape
```
## 모델 정의

임베딩 차원 정의
```python
embedding_dim = 100
```
모델 구성
```python
model = Sequential([
    Embedding(vocab_size, embedding_dim),
    LSTM(128),
    Dense(1, activation='sigmoid')
])
```
## 컴파일

~~1. `optimizer`은 가장 최적화가 잘되는 알고리즘 'adam' 사용~~  
2. `loss`는 이진분류이므로, `binary_crossentropy` 사용
```python
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
```
## ModelCheckpoint

val_acc 기준으로 epoch 마다 최적의 모델을 저장하기 위한 체크포인트 설정
```python
mc = ModelCheckpoint(file_name+'.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
```
## EarlyStopping

val_loss 기준으로 4번까지 Overfitting(과적합)이 진행되면 자동으로 종료
```python
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
```
## 모델 학습
```python
history = model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=64, validation_split=0.2)
```
학습 완료 후 `load_model` 를 해주지 않으면 ModelCheckpoint를 만든 의미가 없음
```python
model = load_model(file_name+'.h5')
```
## 학습 오차 시각화
```python
epochs = range(1, len(history.history['acc']) + 1)
plt.plot(epochs, history.history['loss'])
plt.plot(epochs, history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test, y_test)[1]))
```
## 파일 저장

tokenizer 객체 저장
```python
with open(file_name+"_tokenizer.pickle", "wb") as f:
    pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
```
변수 저장
```python
d = {'min_data':min_data, 'data_length':data_length, 'split_list':split_list,'max_len':max_len, 'trunc_type':trunc_type, 'padding_type':padding_type}
json_var = json.dumps(d)
with open(file_name+'_variable.json', 'w') as f:
    f.write(json_var)
```
불용어 단어 저장
```python
json_stopwords = json.dumps(stopwords)
with open(file_name+'_stopwords.json', 'w') as f:
    f.write(json_stopwords)
```
