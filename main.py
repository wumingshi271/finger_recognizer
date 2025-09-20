"""
优化说明：
1. 改进手指判断逻辑：通过关节向量方向判断手指是否伸直（更符合人体结构）
2. 增加置信度过滤：忽略低置信度的手部检测结果
3. 加入平滑处理：减少手指数量抖动，提高显示稳定性
4. 适配拇指特殊结构：单独优化拇指判断逻辑
"""

import cv2
import mediapipe as mp
from collections import defaultdict

# 手势对应列表（0-10根手指）
gesture = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

# 初始化MediaPipe组件
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# 定义各手指的关节点索引（拇指到小指）
# 每个手指包含4个关键点：[根部, 第一关节, 第二关节, 指尖]
finger_joints = [
    [1, 2, 3, 4],   # 拇指
    [5, 6, 7, 8],   # 食指
    [9, 10, 11, 12], # 中指
    [13, 14, 15, 16], # 无名指
    [17, 18, 19, 20]  # 小指
]

def is_finger_extended(landmarks, finger_idx):
    """判断指定手指是否伸直
    Args:
        landmarks: 手部关键点坐标集合
        finger_idx: 手指索引（0-4，对应拇指到小指）
    Returns:
        bool: 手指是否伸直
    """
    joints = finger_joints[finger_idx]
    p0 = landmarks[joints[0]]  # 根部
    p1 = landmarks[joints[1]]  # 第一关节
    p2 = landmarks[joints[2]]  # 第二关节
    p3 = landmarks[joints[3]]  # 指尖

    # 计算关节向量（方向）
    v1 = (p1.x - p0.x, p1.y - p0.y)  # 根部到第一关节
    v2 = (p2.x - p1.x, p2.y - p1.y)  # 第一到第二关节
    v3 = (p3.x - p2.x, p3.y - p2.y)  # 第二关节到指尖

    # 计算向量点积（判断方向是否一致）
    dot1 = v1[0] * v2[0] + v1[1] * v2[1]
    dot2 = v2[0] * v3[0] + v2[1] * v3[1]
    dot3 = v3[0] * v1[0] + v3[1] * v1[1]

    # 拇指结构特殊，需要额外判断与手掌的角度
    if finger_idx == 0:
        # 拇指需要考虑与食指根部的相对位置
        palm_center = landmarks[0]  # 手腕点作为参考
        thumb_tip_to_palm = (p3.x - palm_center.x, p3.y - palm_center.y)
        index_base_to_palm = (landmarks[5].x - palm_center.x, landmarks[5].y - palm_center.y)
        dot_thumb_palm = thumb_tip_to_palm[0] * index_base_to_palm[0] + thumb_tip_to_palm[1] * index_base_to_palm[1]
        return (dot3 > 0.002 and dot3 > 0.002) or (dot_thumb_palm < -0.06)  # 适配拇指外展情况
    else:
        # 其他手指通过关节向量方向判断
        return dot1 > 0.001 and dot2 > 0.001

def main():
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 初始化手部检测器（提高最小检测置信度）
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,  # 只检测一只手，减少干扰
        min_detection_confidence=0.7,  # 提高检测置信度阈值
        min_tracking_confidence=0.7) as hands:

        # 用于结果平滑的历史记录
        history = []
        history_length = 5  # 记录最近5帧结果

        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法获取帧")
                break

            # 图像预处理
            frame = cv2.flip(frame, 1)  # 水平翻转，镜像显示
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
            rgb_frame.flags.writeable = False  # 提高处理效率

            # 手部检测
            results = hands.process(rgb_frame)
            rgb_frame.flags.writeable = True

            finger_count = 0  # 初始化手指计数

            # 处理检测结果
            if results.multi_hand_landmarks and results.multi_handedness:
                # 只处理置信度最高的手
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # 过滤低置信度的手
                    if handedness.classification[0].score < 0.7:
                        continue

                    # 绘制手部关键点
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )

                    # 统计伸直的手指数量
                    for i in range(5):  # 遍历5根手指
                        if is_finger_extended(hand_landmarks.landmark, i):
                            finger_count += 1

                    # 限制最大手指数量为10（实际最多5根，这里保留原逻辑）
                    finger_count = min(finger_count, 10)

            # 结果平滑处理（取最近5帧的众数）
            history.append(finger_count)
            if len(history) > history_length:
                history.pop(0)

            # 计算历史记录中的众数作为最终结果
            count_stats = defaultdict(int)
            for count in history:
                count_stats[count] += 1
            final_count = max(count_stats, key=count_stats.get)

            # 在画面上显示结果
            cv2.putText(
                frame, gesture[final_count], (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3
            )

            # 显示画面
            cv2.imshow('手部识别', frame)

            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 释放资源
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
