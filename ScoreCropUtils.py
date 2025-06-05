import cv2
import os

# # 이미지 불러오기
# # img_path = '애국가.jpeg'
# image = cv2.imread(img_path)

# 그레이 스케일 및 이진화 

def threshold_img(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return image

import numpy as np
def remove_noise(image):
    image = threshold_img(image)  # 이미지 이진화
    mask = np.zeros(image.shape, np.uint8)  # 보표 영역만 추출하기 위해 마스크 생성

    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(image)  # 레이블링
    for i in range(1, cnt):
        x, y, w, h, area = stats[i]
        if w > image.shape[1] * 0.5:  # 보표 영역에만
            cv2.rectangle(mask, (x, y, w, h), (255, 0, 0), -1)  # 사각형 그리기

    masked_image = cv2.bitwise_and(image, mask)  # 보표 영역 추출

    return masked_image

def remove_staves(image):
    height, width = image.shape
    staves = []  # 오선의 좌표들이 저장될 리스트

    for row in range(height):
        pixels = 0
        for col in range(width):
            pixels += (image[row][col] == 255)  # 한 행에 존재하는 흰색 픽셀의 개수를 셈
        if pixels >= width * 0.5:  # 이미지 넓이의 50% 이상이라면
            if len(staves) == 0 or abs(staves[-1][0] + staves[-1][1] - row) > 1:  # 첫 오선이거나 이전에 검출된 오선과 다른 오선
                staves.append([row, 0])  # 오선 추가 [오선의 y 좌표][오선 높이]
            else:  # 이전에 검출된 오선과 같은 오선
                staves[-1][1] += 1  # 높이 업데이트

    for staff in range(len(staves)):
        top_pixel = staves[staff][0]  # 오선의 최상단 y 좌표
        bot_pixel = staves[staff][0] + staves[staff][1]  # 오선의 최하단 y 좌표 (오선의 최상단 y 좌표 + 오선 높이)
        for col in range(width):
            if image[top_pixel - 1][col] == 0 and image[bot_pixel + 1][col] == 0:  # 오선 위, 아래로 픽셀이 있는지 탐색
                for row in range(top_pixel, bot_pixel + 1):
                    image[row][col] = 0  # 오선을 지움

    return image, [x[0] for x in staves]

def normalization(image, staves, standard):
    if len(staves) < 6:  # 최소 오선 하나(5줄)도 안 될 경우
        print("⚠️ 감지된 오선 수가 부족합니다.")
        return image, staves  # 그대로 반환

    avg_distance = 0
    lines = int(len(staves) / 5)  # 보표(5줄 기준)의 개수
    for line in range(lines):
        for staff in range(4):
            staff_above = staves[line * 5 + staff]
            staff_below = staves[line * 5 + staff + 1]
            avg_distance += abs(staff_above - staff_below)

    denominator = len(staves) - lines
    if denominator == 0:
        print("⚠️ 나눗셈 0 오류 방지를 위해 avg_distance 계산을 건너뜁니다.")
        return image, [0 for _ in staves]  # head_center와 유사한 동작을 위해 staves 모두 0으로 반환

    avg_distance /= denominator

    height, width = image.shape
    weight = standard / avg_distance
    new_width = int(width * weight)
    new_height = int(height * weight)

    image = cv2.resize(image, (new_width, new_height))
    ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    staves = [x * weight for x in staves]

    return image, staves

def weighted(value):
    standard = 10
    return int(value * (standard / 10))

def closing(image):
    kernel = np.ones((weighted(5), weighted(5)), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return image

def put_text(image, text, loc):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, str(text), loc, font, 0.6, (255, 0, 0), 2)
    
def get_center(y, h):
    return (y + y + h) / 2

def object_detection(image, staves):
    lines = int(len(staves) / 5)  # 보표의 개수
    objects = []  # 구성요소 정보가 저장될 리스트

    closing_image = closing(image)
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(closing_image)  # 모든 객체 검출하기
    for i in range(1, cnt):
        (x, y, w, h, area) = stats[i]
        if w >= weighted(5) and h >= weighted(5):  # 악보의 구성요소가 되기 위한 넓이, 높이 조건
            center = get_center(y, h)
            for line in range(lines):
                area_top = staves[line * 5] - weighted(20)  # 위치 조건 (상단)
                area_bot = staves[(line + 1) * 5 - 1] + weighted(20)  # 위치 조건 (하단)
                if area_top <= center <= area_bot:
                    objects.append([line, (x, y, w, h, area)])  # 객체 리스트에 보표 번호와 객체의 정보(위치, 크기)를 추가

    objects.sort()  # 보표 번호 → x 좌표 순으로 오름차순 정렬

    return image, objects

VERTICAL = True
HORIZONTAL = False

def get_line(image, axis, axis_value, start, end, length):
    if axis:
        points = [(i, axis_value) for i in range(start, end)]  # 수직 탐색
    else:
        points = [(axis_value, i) for i in range(start, end)]  # 수평 탐색
    pixels = 0
    for i in range(len(points)):
        (y, x) = points[i]
        pixels += (image[y][x] == 255)  # 흰색 픽셀의 개수를 셈
        next_point = image[y + 1][x] if axis else image[y][x + 1]  # 다음 탐색할 지점
        if next_point == 0 or i == len(points) - 1:  # 선이 끊기거나 마지막 탐색임
            if pixels >= weighted(length):
                break  # 찾는 길이의 직선을 찾았으므로 탐색을 중지함
            else:
                pixels = 0  # 찾는 길이에 도달하기 전에 선이 끊김 (남은 범위 다시 탐색)
    return y if axis else x, pixels

def stem_detection(image, stats, length):
    (x, y, w, h, area) = stats
    stems = []  # 기둥 정보 (x, y, w, h)
    for col in range(x, x + w):
        end, pixels = get_line(image, VERTICAL, col, y, y + h, length)
        if pixels:
            if len(stems) == 0 or abs(stems[-1][0] + stems[-1][2] - col) >= 1:
                (x, y, w, h) = col, end - pixels + 1, 1, pixels
                stems.append([x, y, w, h])
            else:
                stems[-1][2] += 1
    return stems

def object_analysis(image, objects):
    for obj in objects:
        stats = obj[1]
        stems = stem_detection(image, stats, 30)  # 객체 내의 모든 직선들을 검출함
        direction = None
        if len(stems) > 0:  # 직선이 1개 이상 존재함
            if stems[0][0] - stats[0] >= weighted(5):  # 직선이 나중에 발견되면
                direction = True  # 정 방향 음표
            else:  # 직선이 일찍 발견되면
                direction = False  # 역 방향 음표
        obj.append(stems)  # 객체 리스트에 직선 리스트를 추가
        obj.append(direction)  # 객체 리스트에 음표 방향을 추가

    return image, objects

def recognize_key(image, staves, stats):
    (x, y, w, h, area) = stats
    ts_conditions = (
        staves[0] + weighted(5) >= y >= staves[0] - weighted(5) and  # 상단 위치 조건
        staves[4] + weighted(5) >= y + h >= staves[4] - weighted(5) and  # 하단 위치 조건
        staves[2] + weighted(5) >= get_center(y, h) >= staves[2] - weighted(5) and  # 중단 위치 조건
        weighted(18) >= w >= weighted(10) and  # 넓이 조건
        weighted(45) >= h >= weighted(35)  # 높이 조건
    )
    if ts_conditions:
        return True, 0
    else:  # 조표가 있을 경우 (다장조를 제외한 모든 조)
        stems = stem_detection(image, stats, 20)
        if stems[0][0] - x >= weighted(3):  # 직선이 나중에 발견되면
            key = int(10 * len(stems) / 2)  # 샾
        else:  # 직선이 일찍 발견되면
            key = 100 * len(stems)  # 플랫

    return False, key

def count_rect_pixels(image, rect):
    x, y, w, h = rect
    pixels = 0
    for row in range(y, y + h):
        for col in range(x, x + w):
            if image[row][col] == 255:
                pixels += 1
    return pixels

def recognize_note_head(image, stem, direction):
    (x, y, w, h) = stem
    if direction:  # 정 방향 음표
        area_top = y + h - weighted(7)  # 음표 머리를 탐색할 위치 (상단)
        area_bot = y + h + weighted(7)  # 음표 머리를 탐색할 위치 (하단)
        area_left = x - weighted(14)  # 음표 머리를 탐색할 위치 (좌측)
        area_right = x  # 음표 머리를 탐색할 위치 (우측)
    else:  # 역 방향 음표
        area_top = y - weighted(7)  # 음표 머리를 탐색할 위치 (상단)
        area_bot = y + weighted(7)  # 음표 머리를 탐색할 위치 (하단)
        area_left = x + w  # 음표 머리를 탐색할 위치 (좌측)
        area_right = x + w + weighted(14)  # 음표 머리를 탐색할 위치 (우측)

    cnt = 0  # cnt = 끊기지 않고 이어져 있는 선의 개수를 셈
    cnt_max = 0  # cnt_max = cnt 중 가장 큰 값
    head_center = 0
    pixel_cnt = count_rect_pixels(image, (area_left, area_top, area_right - area_left, area_bot - area_top))

    for row in range(area_top, area_bot):
        col, pixels = get_line(image, HORIZONTAL, row, area_left, area_right, 5)
        pixels += 1
        if pixels >= weighted(5):
            cnt += 1
            cnt_max = max(cnt_max, pixels)
            head_center += row

    head_exist = (cnt >= 3 and pixel_cnt >= 50)
    head_fill = (cnt >= 8 and cnt_max >= 9 and pixel_cnt >= 80)
    
    if cnt == 0:
        return False, False, np.array([0, 0])
    
    head_center /= cnt

    return head_exist, head_fill, head_center

def count_pixels_part(image, area_top, area_bot, area_col):
    cnt = 0
    flag = False
    for row in range(area_top, area_bot):
        if not flag and image[row][area_col] == 255:
            flag = True
            cnt += 1
        elif flag and image[row][area_col] == 0:
            flag = False
    return cnt

def recognize_note_tail(image, index, stem, direction):
    (x, y, w, h) = stem
    if direction:  # 정 방향 음표
        area_top = y  # 음표 꼬리를 탐색할 위치 (상단)
        area_bot = y + h - weighted(15)  # 음표 꼬리를 탐색할 위치 (하단)
    else:  # 역 방향 음표
        area_top = y + weighted(15)  # 음표 꼬리를 탐색할 위치 (상단)
        area_bot = y + h  # 음표 꼬리를 탐색할 위치 (하단)
    if index:
        area_col = x - weighted(4)  # 음표 꼬리를 탐색할 위치 (열)
    else:
        area_col = x + w + weighted(4)  # 음표 꼬리를 탐색할 위치 (열)

    cnt = count_pixels_part(image, area_top, area_bot, area_col)

    return cnt

def recognize_note_dot(image, stem, direction, tail_cnt, stems_cnt):
    (x, y, w, h) = stem
    if direction:  # 정 방향 음표
        area_top = y + h - weighted(10)  # 음표 점을 탐색할 위치 (상단)
        area_bot = y + h + weighted(5)  # 음표 점을 탐색할 위치 (하단)
        area_left = x + w + weighted(2)  # 음표 점을 탐색할 위치 (좌측)
        area_right = x + w + weighted(12)  # 음표 점을 탐색할 위치 (우측)
    else:  # 역 방향 음표
        area_top = y - weighted(10)  # 음표 점을 탐색할 위치 (상단)
        area_bot = y + weighted(5)  # 음표 점을 탐색할 위치 (하단)
        area_left = x + w + weighted(14)  # 음표 점을 탐색할 위치 (좌측)
        area_right = x + w + weighted(24)  # 음표 점을 탐색할 위치 (우측)
    dot_rect = (
        area_left,
        area_top,
        area_right - area_left,
        area_bot - area_top
    )

    pixels = count_rect_pixels(image, dot_rect)

    threshold = (10, 15, 20, 30)
    if direction and stems_cnt == 1:
        return pixels >= weighted(threshold[tail_cnt])
    else:
        return pixels >= weighted(threshold[0])

def recognize_pitch(image, staff, head_center):
    pitch_lines = [staff[4] + weighted(30) - weighted(5) * i for i in range(21)]

    for i in range(len(pitch_lines)):
        line = pitch_lines[i]
        if line + weighted(2) >= head_center >= line - weighted(2):
            return i

def recognize_rest_dot(image, stats):
    (x, y, w, h, area) = stats
    area_top = y - weighted(10)  # 쉼표 점을 탐색할 위치 (상단)
    area_bot = y + weighted(10)  # 쉼표 점을 탐색할 위치 (하단)
    area_left = x + w  # 쉼표 점을 탐색할 위치 (좌측)
    area_right = x + w + weighted(10)  # 쉼표 점을 탐색할 위치 (우측)
    dot_rect = (
        area_left,
        area_top,
        area_right - area_left,
        area_bot - area_top
    )

    pixels = count_rect_pixels(image, dot_rect)

    return pixels >= weighted(10)

def recognition(image, staves, objects):
    import os
    from pathlib import Path

    key = 0
    time_signature = False
    beats = []  # 박자 리스트
    pitches = []  # 음이름 리스트

    # 저장 디렉토리 설정
    save_dir = Path("cropped_notes_from_recognition")
    save_dir.mkdir(parents=True, exist_ok=True)

    # binary 이미지 생성 (crop에 사용)
    binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    for i in range(1, len(objects) - 1):
        obj = objects[i]
        line = obj[0]
        stats = obj[1]
        stems = obj[2]
        direction = obj[3]
        (x, y, w, h, area) = stats
        staff = staves[line * 5: (line + 1) * 5]

        if not time_signature:
            ts, temp_key = recognize_key(image, staff, stats)
            time_signature = ts
            key += temp_key
            if time_signature:
                put_text(image, key, (x, y + h + weighted(30)))
        else:
            notes = recognize_note(image, staff, stats, stems, direction)
            if len(notes[0]):
                for beat in notes[0]:
                    beats.append(beat)
                for pitch in notes[1]:
                    pitches.append(pitch)

                # 🔥 음표 객체 crop 및 저장
                cropped = binary[y:y+h, x:x+w]
                resized = cv2.resize(cropped, (299, 299), interpolation=cv2.INTER_AREA)
                save_path = save_dir / f"note_{i:03}.png"
                cv2.imwrite(str(save_path), resized)

            else:
                rest = recognize_rest(image, staff, stats)
                if rest:
                    beats.append(rest)
                    pitches.append(-1)

                    # 🔥 쉼표도 crop해서 저장
                    cropped = binary[y:y+h, x:x+w]
                    resized = cv2.resize(cropped, (299, 299), interpolation=cv2.INTER_AREA)
                    save_path = save_dir / f"rest_{i:03}.png"
                    cv2.imwrite(str(save_path), resized)

                else:
                    whole_note, pitch = recognize_whole_note(image, staff, stats)
                    if whole_note:
                        beats.append(whole_note)
                        pitches.append(pitch)

                        # 🔥 온음표 crop 저장
                        cropped = binary[y:y+h, x:x+w]
                        resized = cv2.resize(cropped, (299, 299), interpolation=cv2.INTER_AREA)
                        save_path = save_dir / f"whole_{i:03}.png"
                        cv2.imwrite(str(save_path), resized)

        cv2.rectangle(image, (x, y, w, h), (255, 0, 0), 1)
        put_text(image, i, (x, y - weighted(20)))

    return image, key, beats, pitches



def recognize_note(image, staff, stats, stems, direction):
    (x, y, w, h, area) = stats
    notes = []
    pitches = []
    note_condition = (
            len(stems) and
            w >= weighted(10) and  # 넓이 조건
            h >= weighted(35) and  # 높이 조건
            area >= weighted(95)  # 픽셀 갯수 조건
    )
    if note_condition:
        for i in range(len(stems)):
            stem = stems[i]
            head_exist, head_fill, head_center = recognize_note_head(image, stem, direction)
            if head_exist:
                tail_cnt = recognize_note_tail(image, i, stem, direction)
                dot_exist = recognize_note_dot(image, stem, direction, len(stems), tail_cnt)
                note_classification = (
                    ((not head_fill and tail_cnt == 0 and not dot_exist), 2),
                    ((not head_fill and tail_cnt == 0 and dot_exist), -2),
                    ((head_fill and tail_cnt == 0 and not dot_exist), 4),
                    ((head_fill and tail_cnt == 0 and dot_exist), -4),
                    ((head_fill and tail_cnt == 1 and not dot_exist), 8),
                    ((head_fill and tail_cnt == 1 and dot_exist), -8),
                    ((head_fill and tail_cnt == 2 and not dot_exist), 16),
                    ((head_fill and tail_cnt == 2 and dot_exist), -16),
                    ((head_fill and tail_cnt == 3 and not dot_exist), 32),
                    ((head_fill and tail_cnt == 3 and dot_exist), -32)
                )

                for j in range(len(note_classification)):
                    if note_classification[j][0]:
                        note = note_classification[j][1]
                        pitch = recognize_pitch(image, staff, head_center)
                        notes.append(note)
                        pitches.append(pitch)
                        put_text(image, note, (stem[0] - weighted(10), stem[1] + stem[3] + weighted(30)))
                        put_text(image, pitch, (stem[0] - weighted(10), stem[1] + stem[3] + weighted(60)))
                        break

    return notes, pitches


def recognize_rest(image, staff, stats):
    (x, y, w, h, area) = stats
    rest = 0
    center = get_center(y, h)
    rest_condition = staff[3] > center > staff[1]
    if rest_condition:
        cnt = count_pixels_part(image, y, y + h, x + weighted(1))
        if weighted(35) >= h >= weighted(25):
            if cnt == 3 and weighted(11) >= w >= weighted(7):
                rest = 4
            elif cnt == 1 and weighted(14) >= w >= weighted(11):
                rest = 16
        elif weighted(22) >= h >= weighted(16):
            if weighted(15) >= w >= weighted(9):
                rest = 8
        elif weighted(8) >= h:
            if staff[1] + weighted(5) >= center >= staff[1]:
                rest = 1
            elif staff[2] >= center >= staff[1] + weighted(5):
                rest = 2
        if recognize_rest_dot(image, stats):
            rest *= -1
        if rest:
            put_text(image, rest, (x, y + h + weighted(30)))
            put_text(image, -1, (x, y + h + weighted(60)))

    return rest


def recognize_whole_note(image, staff, stats):
    whole_note = 0
    pitch = 0
    (x, y, w, h, area) = stats
    while_note_condition = (
            weighted(22) >= w >= weighted(12) >= h >= weighted(9)
    )
    if while_note_condition:
        dot_rect = (
            x + w,
            y - weighted(10),
            weighted(10),
            weighted(20)
        )
        pixels = count_rect_pixels(image, dot_rect)
        whole_note = -1 if pixels >= weighted(10) else 1
        pitch = recognize_pitch(image, staff, get_center(y, h))
        put_text(image, whole_note, (x, y + h + weighted(30)))
        put_text(image, pitch, (x, y + h + weighted(60)))

    return whole_note, pitch

def draw_staves(image, staves, color=(0, 0, 0), thickness=1):
    # === 1. 색 반전: 흑배경 → 백배경
    if len(image.shape) == 2:
        image = cv2.bitwise_not(image)                     # 반전
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)    # 컬러로 변환
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.bitwise_not(gray)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # === 2. 오선 그리기
    for line_y in staves:
        cv2.line(image, (0, int(line_y)), (image.shape[1], int(line_y)), color, thickness)

    return image

def resize_with_padding(image, target_size=299, max_scale=4.0):
    h, w = image.shape[:2]

    # 확대 비율 결정 (최대 4배까지만 확대)
    scale = min(target_size / max(h, w), max_scale)

    new_w, new_h = int(w * scale), int(h * scale)

    # 확대
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 중앙 배치 위한 여백 계산
    top = (target_size - new_h) // 2
    bottom = target_size - new_h - top
    left = (target_size - new_w) // 2
    right = target_size - new_w - left

    # 흰색 여백 추가
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255]
    )

    return padded



