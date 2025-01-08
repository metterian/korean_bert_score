import os
import json
import zipfile
from pathlib import Path
import re  # 정규표현식을 위한 import 추가

def extract_and_load_json_data(base_path, data_type):
    """
    영-한 키워드가 있는 ZIP 파일을 추출하고 JSON 데이터를 로드하는 함수
    
    Args:
        base_path (str): 데이터가 있는 기본 경로
        data_type (str): 'training' 또는 'validation'
    
    Returns:
        list: JSON 데이터 리스트
    """
    # 라벨링 데이터 경로 설정
    labeling_path = Path(base_path) / data_type / "02.라벨링데이터"
    extract_path = labeling_path / "extracted"
    json_data = []

    # 디렉토리가 존재하는지 확인
    if not labeling_path.exists():
        print(f"경로를 찾을 수 없습니다: {labeling_path}")
        return json_data

    # 추출 디렉토리들 생성
    extract_path.mkdir(exist_ok=True)
    ht_extract_path = labeling_path / "extracted_ht"  # HT 데이터용 새로운 경로
    ht_extract_path.mkdir(exist_ok=True)

    # ZIP 파일 검색 및 압축 해제
    for zip_file in labeling_path.glob("*.zip"):
        try:
            if re.search(r'평가데이터\(MTPE\)_[^-]+-한', zip_file.name):
                # MTPE 데이터 압축 해제
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
                print(f"MTPE 데이터 압축 해제 완료: {zip_file.name}")
            elif re.search(r'VL_번역말뭉치\(HT\)_한', zip_file.name):
                # HT 데이터 압축 해제
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(ht_extract_path)
                print(f"HT 데이터 압축 해제 완료: {zip_file.name}")
        except Exception as e:
            print(f"압축 해제 중 에러 발생 ({zip_file.name}): {str(e)}")
            continue

    # 압축 해제된 JSON 파일 읽기
    for json_file in extract_path.glob("**/*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                json_data.append(data)
            print(f"JSON 파일 처리 완료: {json_file.name}")
        except Exception as e:
            print(f"JSON 파일 처리 중 에러 발생 ({json_file.name}): {str(e)}")

    return json_data

def collect_source_sentences(base_path, data_type, output_filename="source_sentences.txt"):
    """
    extracted_ht 폴더의 모든 JSON 파일에서 source_sentence를 추출하여 txt 파일로 저장하는 함수
    
    Args:
        base_path (str): 데이터가 있는 기본 경로
        data_type (str): 'training' 또는 'validation'
        output_filename (str): 출력할 txt 파일명
    """
    # 경로 설정
    ht_extract_path = Path(base_path) / data_type / "02.라벨링데이터/extracted_ht"
    output_path = Path(base_path) / data_type / "02.라벨링데이터" / output_filename

    # 디렉토리 존재 확인
    if not ht_extract_path.exists():
        print(f"경로를 찾을 수 없습니다: {ht_extract_path}")
        return

    # source_sentence 수집
    sentences = []
    for json_file in ht_extract_path.glob("**/*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict) and 'data' in data:
                    # data 리스트 내의 각 항목에서 source_sentence 추출
                    for item in data['data']:
                        if 'source_sentence' in item:
                            sentences.append(item['source_sentence'])
                print(f"JSON 파일 처리 완료: {json_file.name}")
        except Exception as e:
            print(f"JSON 파일 처리 중 에러 발생 ({json_file.name}): {str(e)}")

    # 수집된 문장들을 txt 파일로 저장
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for sentence in sentences:
                f.write(sentence + '\n')
        print(f"문장 추출 완료. 총 {len(sentences)}개의 문장이 {output_path}에 저장되었습니다.")
    except Exception as e:
        print(f"파일 저장 중 에러 발생: {str(e)}")

def main():
    # 기본 경로 설정
    base_path = "008.다국어 번역 품질 평가 데이터/3.개방데이터/1.데이터"
    
    # validation 데이터 처리
    validation_data = extract_and_load_json_data(base_path, "Validation")
    print(f"검증 데이터 개수: {len(validation_data)}")
    
    # source_sentence 수집 및 저장
    collect_source_sentences(base_path, "Validation")

if __name__ == "__main__":
    main()
