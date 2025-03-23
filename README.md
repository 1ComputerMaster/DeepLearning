# 신경망 학습 프로젝트

이 가이드는 Windows 환경에서 TensorFlow 및 Jupyter Notebook을 정상적으로 구동하기 위한 초기 셋팅 과정을 단계별로 설명합니다.

---

## 1\. 시스템 요구 사항

-   **운영체제:** Windows 10 이상 (64비트)
-   **Python 버전:** 3.7 ~ 3.10 (64비트)
-   **CPU:** AVX 명령어를 지원하는 64비트 CPU
    -   오래된 CPU의 경우 TensorFlow 빌드가 동작하지 않을 수 있습니다.
-   **Visual C++ Redistributable:**  
    [공식 다운로드 페이지](https://learn.microsoft.com/ko-kr/cpp/windows/latest-supported-vc-redist?view=msvc-170)에서 설치

---

## 2\. Python 및 가상환경 설정

### 2.1 Python 설치

1.  [Python 공식 웹사이트](https://www.python.org/downloads/)에서 **Python 3.10 Windows x86-64** 설치 파일을 다운로드합니다.
2.  설치 시 **“Add Python to PATH”** 옵션을 체크합니다.

### 2.2 가상환경 생성 및 활성화

1.  **가상환경 생성 (venv)**  
---
    Windows에서는 Python Launcher를 사용하여 아래와 같이 생성합니다:
    
```python
py -3.10 -m venv tf-env
가상환경 활성화
```
    

PowerShell:
```powershell
.\\tf-env\\Scripts\\Activate.ps1  
```
실행 정책 오류 발생 시, 먼저 다음 명령어를 실행합니다:

  
```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser  
```
CMD:
 
```cmd 
tf-env\\Scripts\\activate  
```
가상환경이 활성화되면 프롬프트 앞에 (tf-env) 등이 표시됩니다.

2.  필수 라이브러리 설치  
---
    가상환경이 활성화된 상태에서 다음 명령어를 실행하세요:

```bash    
pip install --upgrade pip  
pip install tensorflow==2.8.\* # CPU 전용 환경: tensorflow-cpu==2.8.\* 로도 설치 가능  
pip install numpy matplotlib  
pip install jupyter  
```
설치 후 pip list를 통해 tensorflow, numpy, matplotlib 등이 정상적으로 설치되었는지 확인합니다.

3.  Jupyter Notebook 커널 설정
---
3.1 ipykernel 설치  
```bash  
pip install ipykernel
```
3.2 커널 등록  
```bash  
python -m ipykernel install --user --name tf-env --display-name "tf-env"
```
3.3 Jupyter Notebook 실행  
```bash  
jupyter notebook  
```
Notebook이 열리면, Kernel 메뉴에서 "tf-env" 커널을 선택합니다.  
이제 Notebook 코드 셀에서 import tensorflow as tf; print(tf.**version**)를 실행하여 TensorFlow 버전이 출력되면 정상입니다.

4. 가상환경 활성화
---
PowerShell:
```powershell
 .\\tf-env\\Scripts\\Activate.ps1 또는 CMD: tf-env\\Scripts\\activate
```
Jupyter Notebook 실행

```bash  
jupyter notebook  
```
5.1 Notebook 파일 열기

예: TwoLayerNet.ipynb 또는 DeepLearning\_Chapter4.ipynb 파일을 열고, 셀을 순서대로 실행합니다.

5.2 MNIST 데이터셋 학습

Notebook 내에서 MNIST 데이터셋을 불러와 미니배치 학습, 파라미터 업데이트, 학습 손실 및 테스트 정확도 그래프를 확인할 수 있습니다.
  

## 3\.  참고 자료  
TensorFlow Installation Guide

Python venv Documentation

Microsoft Visual C++ Redistributable

밑바닥부터 시작하는 딥러닝 1