# %% [markdown]
# # Perceptron을 활용한 논리 게이트 및 XOR 구현

# 이 Notebook은 기본 퍼셉트론 모델을 사용하여 **AND, NAND, OR** 논리 게이트를 구현하고,  
# 이들을 조합한 2층 퍼셉트론 구조로 **XOR** 게이트를 구현하는 과정을 다룹니다.

# 또한, 신경망의 기본 개념(뉴런, 가중치, 편향, 활성화 함수)과  
# 순전파 및 역전파의 원리를 간단한 Numpy 기반 예제로 확인할 수 있습니다.
# ## 1. 단일 퍼셉트론을 이용한 논리 게이트 구현

# 퍼셉트론은 입력값에 가중치를 곱하고 편향을 더한 후,  
# 0을 기준으로 활성화 여부를 결정하는 단순 신경망 모델입니다.

# 아래는 임계값 대신 편향(bias)을 사용하여 AND, NAND, OR 게이트를 구현한 코드입니다.

# # AND Gate 구현

# %%
import numpy as np

def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7  # 편향: b = -θ
    tmp = np.sum(w * x) + b
    return 1 if tmp > 0 else 0

# AND Gate 테스트
print('AND Gate')
print(AND(0, 0))  # Expected output: 0
print(AND(0, 1))  # Expected output: 0
print(AND(1, 0))  # Expected output: 0
print(AND(1, 1))  # Expected output: 1
# %%
# %% [markdown]

# ### NAND Gate 구현

# NAND 게이트는 AND 게이트와 반대의 역할을 하며,  
# 출력이 0이 되는 경우만 AND와 반대입니다.


# %%
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7  # NAND의 경우 부호 반대로 설정
    tmp = np.sum(w * x) + b
    return 1 if tmp > 0 else 0

# NAND Gate 테스트
print('NAND Gate')
print(NAND(0, 0))  # Expected output: 1
print(NAND(0, 1))  # Expected output: 1
print(NAND(1, 0))  # Expected output: 1
print(NAND(1, 1))  # Expected output: 0
# %%
# %% [markdown]
# ### OR Gate 구현

# OR 게이트는 입력 중 하나라도 1이면 출력이 1이 되는 논리 연산입니다.

# %%
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w * x) + b
    return 1 if tmp > 0 else 0

# OR Gate 테스트
print('OR Gate')
print(OR(0, 0))  # Expected output: 0
print(OR(0, 1))  # Expected output: 1
print(OR(1, 0))  # Expected output: 1
print(OR(1, 1))  # Expected output: 1
# %%
# %% [markdown]
# ## 2. XOR Gate 구현 (2층 퍼셉트론)

# XOR 게이트는 단일 퍼셉트론으로는 구현할 수 없는 비선형 문제입니다.  
# 그러나 **NAND**, **OR**, **AND** 게이트를 계층적으로 조합하면 2층 퍼셉트론 구조로 XOR를 구현할 수 있습니다.


# %%
def XOR(x1, x2):
    # 첫 번째 층: NAND와 OR 연산 수행
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    # 두 번째 층: AND 연산을 통해 최종 출력 결정
    return AND(s1, s2)

# XOR Gate 테스트
print('XOR Gate')
print(XOR(0, 0))  # Expected output: 0
print(XOR(0, 1))  # Expected output: 1
print(XOR(1, 0))  # Expected output: 1
print(XOR(1, 1))  # Expected output: 0
# %%
# %% [markdown]
# ## 3. 간단한 2층 신경망 구현 (XOR 문제 예제)

# 아래 예제는 Numpy를 사용하여  
# 입력층(2개 뉴런), 은닉층(3개 뉴런), 출력층(1개 뉴런)으로 구성된 2층 신경망을 구현한 것입니다.

# - **순전파(Forward Propagation):** 입력부터 출력까지 계산  
# - **역전파(Backpropagation):** 손실 함수의 기울기를 계산하여 파라미터 업데이트  
# - **학습:** 여러 에포크 동안 경사 하강법을 통해 네트워크를 최적화
# %%

# 활성화 함수와 그 도함수 정의
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# 네트워크 초기화: 입력(2), 은닉(3), 출력(1)
np.random.seed(42)
W1 = np.random.randn(2, 3)
b1 = np.zeros((1, 3))
W2 = np.random.randn(3, 1)
b2 = np.zeros((1, 1))

# 순전파 함수
def forward(X):
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    cache = (Z1, A1, Z2, A2)
    return A2, cache

# 역전파 함수 (MSE 손실 기준)
def backward(X, Y, cache):
    Z1, A1, Z2, A2 = cache
    m = X.shape[0]
    
    dA2 = A2 - Y
    dZ2 = dA2 * sigmoid_derivative(Z2)
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * sigmoid_derivative(Z1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    
    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return grads

# 파라미터 업데이트 함수
def update_parameters(grads, eta=0.1):
    global W1, b1, W2, b2
    W1 -= eta * grads["dW1"]
    b1 -= eta * grads["db1"]
    W2 -= eta * grads["dW2"]
    b2 -= eta * grads["db2"]

# XOR 데이터셋
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
Y = np.array([[0],
              [1],
              [1],
              [0]])

# 학습 루프
epochs = 10000
for i in range(epochs):
    A2, cache = forward(X)
    grads = backward(X, Y, cache)
    update_parameters(grads, eta=0.5)
    if i % 1000 == 0:
        loss = np.mean((A2 - Y)**2)
        print(f"Epoch {i}, Loss: {loss:.4f}")

# 최종 예측 결과
predictions, _ = forward(X)
print("최종 예측 결과:")
print(predictions)

# %%
# %%[markdown]
# # 소프트 맥스 함수
# 소프트맥스 함수는 주어진 입력 벡터(보통 뉴런의 출력, 즉 로짓)를 확률 분포로 변환해 주는 함수입니다.
# 각 원소에 대해 지수 함수를 적용한 후 전체 합으로 나누어 계산하는데, 그 결과 모든 출력 값이 0과 1 사이의 값을 가지며 총합은 1이 됩니다.
# ## 소프트맥스 함수의 역할 및 사용 이유
# ### 1. 분류 문제에서의 역할
# **확률 분포 생성:**
# 분류 문제에서는 각 클래스에 대한 예측 확률이 필요합니다.
# 소프트맥스 함수는 로짓 값을 확률처럼 해석할 수 있게 만들어주어,
# 모델이 "이 클래스일 확률이 얼마다"라는 정보를 제공합니다.
# ### 2. 학습 시에는 사용하고 추론 시에는 생략하는 이유  
# 
# **학습 시 사용 이유**: 
# 학습 단계에서는 모델의 예측값과 실제 레이블 간의 차이를 정량적으로 계산해야 합니다.
# 소프트맥스 함수로 변환한 확률 분포와 실제 정답 사이의 차이를
# 교차 엔트로피 손실로 계산하고, 이를 미분하여 가중치를 업데이트합니다.
# 이때 소프트맥스의 미분값은 기울기 계산에 중요한 역할을 합니다.  
# <br />
# 
# **추론(예측) 시 생략하는 이유**:  
# 모델의 최종 예측은 보통 "어떤 클래스의 확률이 가장 높은가?"를 판단하는
# argmax 연산으로 결정됩니다.
# 소프트맥스 함수는 단조 증가(monotonic increasing) 함수이므로,
# 이미 배열에서 가장 큰 값을 갖는 성분은 소프트맥스를 통과시켜도 여전히 가장 큰 값을 유지합니다.
# 따라서, argmax(logits) = argmax(softmax(logits)) 성질 때문에,
# 실제 추론 단계에서는 계산 비용을 줄이기 위해 소프트맥스 계산을 생략할 수도 있습니다.
# %%
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c) 
    print(exp_a)
    sum_exp_a = np.sum(exp_a)
    print(sum_exp_a)
    y = exp_a / sum_exp_a

    return y

a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y)
print(np.sum(y))


#%%
# %%[markdown]
# # 손글씨 숫자 인식
# ## MNIST 데이터셋
# * flatten : 입력 이미지를 1차원으로 평탄화 할 지 여부를 묻습니다.
# * one-hot-label : 정답 원소만 1으로 표기하는 식으로 표기 
# * normalize : 입력 이미지의 픽셀 값을 0.0~1.0 으로 표기
#   
# (x_train, t_train) -> (훈련 이미지, 훈련 레이블),(x_test, t_test) -> (시험 이미지, 시험 레이블)
#%%
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label)  # 5

print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 형상을 원래 이미지의 크기로 변형
print(img.shape)  # (28, 28)

img_show(img)
# %%
# %%[markdown]
#
# ### init_network()
# - pickle 파일인 sample_weight.pkl 에서 학습된 가중치 매개변수를 읽습니다.
# 가중치와 편향 값이 딕셔너리 변수로 저장되어있습니다.
#   
# ### Softmax 함수  
#
# - 선형 변환 결과를 확률 분포 형식으로 바꾸어 줍니다.
# 어떤 클래스가 가장 높은 확률을 가지는가? 가 표현됩니다.
#
#   
# ### Sigmoid 함수 : 선형 결합이후 비선형 성을 높이기 위해서 사용
#   
# - **첫 번째 은닉층** 에서는 입력 데이터에 비선형 변환을 추가하여 복잡한 패턴을 학습할 수 있게 하고,
# - **두 번째 은닉층** 에서도 마찬가지로 추가적인 비선형 변환을 통해 모델의 표현력을 높입니다.
#   
# ### 신경망의 추론 처리 예제 결과
# - 정확도 평가 시 0.9352가 나옵니다.
# - normalize True로 정규화 시킴 (0.0 ~ 1.0)
# %%
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p= np.argmax(y) # 확률이 가장 높은 원소의 인덱스를 얻는다.
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

# %%
# %%[markdown]
# ### 배치 처리
# - 이미지를 지폐 다발 처럼 묶어서 한 방에 처리하여 효율을 높입니다.
# - 배치 사이즈 만큼 predict 돌리고 argmax로 axis = 1 즉, 각 배열 중 1번째 차원을 구성하는 원소의 가장 큰 값을 찾도록 설정
# - p == t[i:i+batch_size] -> 실제 정답 값과 같은지 비교함
# - GPU 사용량을 올릴 수 있습니다.
# %%

x, t = get_data()
network = init_network()

batch_size = 100 # 배치 크기
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))


# %%
# %%[markdown]
# ## 4. 결론

# 이번 Notebook에서는 다음 내용을 학습했습니다.

# - **퍼셉트론의 기본 원리:**  
#   입력, 가중치, 편향, 활성화 함수의 역할을 이해했습니다.

# - **단일 퍼셉트론으로 논리 게이트 구현:**  
#   AND, NAND, OR 게이트를 통해 선형 결정 경계를 확인했습니다.

# - **다층 퍼셉트론을 통한 XOR 문제 해결:**  
#   NAND, OR, AND 게이트를 계층적으로 조합해 XOR 게이트를 구현하는 방법을 배웠습니다.

# - **간단한 신경망 학습 구현:**  
#   순전파, 역전파 및 경사 하강법을 이용해 2층 신경망을 구현하고, XOR 문제를 학습시켰습니다.


# %%
