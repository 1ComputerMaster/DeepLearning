# Perceptron을 활용한 논리 게이트 및 XOR 구현

이 프로젝트는 기본적인 퍼셉트론을 활용하여 AND, NAND, OR 게이트를 구현하고, 이들을 조합해 2층 퍼셉트론 구조로 XOR 게이트를 구현하는 방법을 다룹니다.  
특히, 임계값(θ)을 직접 사용하는 대신 편향(bias)을 도입해 구현하는 방식을 통해, 퍼셉트론의 개념을 보다 명확히 이해할 수 있습니다.

---

## 개요

- **퍼셉트론(Perceptron)**  
  퍼셉트론은 입력에 가중치를 곱한 후 편향을 더해, 그 합이 0을 넘는지 여부로 출력을 결정하는 단순 신경망 모델입니다.  
  기존에 임계값(θ)을 사용하는 식을 다음과 같이 변환할 수 있습니다.

y = 0 if (w1x1 + w2x2 + b ≤ 0)
y = 1 if (w1x1 + w2x2 + b > 0)

- **논리 게이트 구현**  
AND, NAND, OR 게이트는 선형 결정 경계를 가지므로 단일 퍼셉트론으로 구현할 수 있습니다.

- **비선형 문제와 XOR**  
XOR 게이트는 단일 퍼셉트론으로 해결할 수 없는 비선형 문제입니다.  
하지만 NAND, OR, AND 게이트를 계층적으로 구성하면 2층 퍼셉트론 구조로 구현할 수 있습니다.

---

## 구현 코드

### 1. AND Gate

```python
import numpy as np

def AND(x1, x2):
  x = np.array([x1, x2])
  w = np.array([0.5, 0.5])
  b = -0.7  # 편향치로 임계값을 대체
  tmp = np.sum(w * x) + b

  return 1 if tmp > 0 else 0

# 테스트
print('AND Gate')
print(AND(0, 0))  # 0
print(AND(0, 1))  # 0
print(AND(1, 0))  # 0
print(AND(1, 1))  # 1

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7  # NAND의 경우 부호 반대로 설정
    tmp = np.sum(w * x) + b

    return 1 if tmp > 0 else 0

# 테스트
print('NAND Gate')
print(NAND(0, 0))  # 1
print(NAND(0, 1))  # 1
print(NAND(1, 0))  # 1
print(NAND(1, 1))  # 0

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w * x) + b

    return 1 if tmp > 0 else 0

# 테스트
print('OR Gate')
print(OR(0, 0))  # 0
print(OR(0, 1))  # 1
print(OR(1, 0))  # 1
print(OR(1, 1))  # 1

def XOR(x1, x2):
    # 첫 번째 층: NAND와 OR 연산 수행
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    # 두 번째 층: AND 연산으로 최종 출력 결정
    return AND(s1, s2)

# 테스트
print('XOR Gate')
print(XOR(0, 0))  # 0
print(XOR(0, 1))  # 1
print(XOR(1, 0))  # 1
print(XOR(1, 1))  # 0

```

### 결론
간단한 퍼셉트론 모델을 활용해 기본 논리 게이트(AND, NAND, OR)를 구현하고,
이들을 조합해 XOR와 같은 비선형 문제를 해결하는 방법을 살펴보았습니다.
실제 하드웨어나 소프트웨어에서 신경망을 구현할 때도 이러한 기본 원리가 중요한 역할을 합니다.
