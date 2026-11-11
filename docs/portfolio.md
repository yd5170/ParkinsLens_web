# 포트폴리오에 사용할 수 있는 프로젝트 설명

## 제목

3D-CNN·3D-ResNet 기반 뇌 MRI 분석 결과를 시각화하는 Streamlit 웹 플랫폼, ParkinsLens Web

## Overview

MRI 입력부터 전처리와 분석 결과 확인까지의 흐름을 연결한 웹 프로토타입입니다. DICOM 폴더를 NIfTI 볼륨으로 변환하고, 실행 환경에 따른 전처리·모델 추론·CNN 참고 CAM·PDF 보고서를 제공합니다.

## Technology

- Language: Python
- Tool: GitHub, Streamlit Community Cloud 배포 설정 지원
- Library: Streamlit, PyTorch, scikit-learn, pydicom, dicom2nifti, nibabel, NumPy, SciPy, Pillow, ReportLab, huggingface_hub
- Data: PPMI T2 MRI 연구 프로젝트 기반. 공개 웹 예시는 합성 DICOM·NIfTI와 가상 환자 데이터 사용

## Result

- DICOM 업로드, 전처리 결과 확인, MRI 슬라이스 탐색과 PDF 보고서 출력을 웹 화면으로 연결.
- 실행 환경에 따라 로컬 전체 전처리 또는 경량 전처리 경로를 구분하고, 모델 연결 상태와 시연 결과를 표시.
- 환자·검사 전환 시 분석 상태 격리와 입력 내용 해시 적용, 중복 추론 제거.
- 합성 예시로 56³ 경량 전처리·PDF 생성 확인, 회귀 검사 10개 및 Streamlit AppTest 경로 검사 통과.

왼쪽 이미지 4~6개와 오른쪽 Overview·Technology·Result의 한 장 구성을 사용할 수 있습니다. 이미지는 업로드 화면, 전처리 결과, 모델 흐름, 결과 뷰어, PDF, 가상 환자관리 중 선택합니다. `docs/images/`는 과거 발표자료의 화면이며 최신 수정본을 다시 캡처한 뒤 제출하는 것이 좋습니다.

위 내용은 저장소의 구현 범위 설명입니다. 개인 기여 항목에는 실제 담당 작업만 명시하고, 팀 전체 성과나 이번 공개 준비에서 수정한 내용을 과거 개인 수행으로 소급하지 않습니다. 모델 성능은 발표자료 보고값과 연결된 체크포인트의 재평가 결과를 구분해 적습니다.
