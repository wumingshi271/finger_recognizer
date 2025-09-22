import cv2
import mediapipe as mp
from collections import defaultdict


class HandRecognizer:
    def __init__(self):
        self.gesture = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        # 手指关节点索引
        self.finger_joints = [
            [1, 2, 3, 4],  # 拇指
            [5, 6, 7, 8],  # 食指
            [9, 10, 11, 12],  # 中指
            [13, 14, 15, 16],  # 无名指
            [17, 18, 19, 20]  # 小指
        ]

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("无法打开摄像头")

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.history = []
        self.history_length = 5

    def is_finger_extended(self, landmarks, finger_idx):
        """判断指定手指是否伸直"""
        joints = self.finger_joints[finger_idx]
        root = landmarks[0]
        p0 = landmarks[joints[0]]
        p1 = landmarks[joints[1]]
        p2 = landmarks[joints[2]]
        p3 = landmarks[joints[3]]

        # 计算关节向量（方向）
        v1 = (p1.x - p0.x, p1.y - p0.y)  # 根部到第一关节
        v2 = (p2.x - p1.x, p2.y - p1.y)  # 第一到第二关节
        v3 = (p3.x - p2.x, p3.y - p2.y)  # 第二关节到指尖
        v4 = (root.x - p0.x, root.y - p0.y)  # 掌根到根部

        # 计算向量点积（判断方向是否一致）
        dot4 = v4[0] * v3[0] + v4[1] * v3[1]
        dot5 = v4[0] * v2[0] + v4[1] * v2[1]

        return dot4 < 0 and dot5 < 0

    def count_fingers(self, results):
        """
        手指计数的核心函数（counter函数）
        参数：results - MediaPipe的手部检测结果
        返回：当前检测到的手指数量
        """
        finger_count = 0

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # 过滤低置信度结果
                if handedness.classification[0].score < 0.7:
                    continue

                # 遍历5根手指，统计伸直的数量
                for i in range(5):
                    if self.is_finger_extended(hand_landmarks.landmark, i):
                        finger_count += 1

                # 限制最大数量为10（实际最多5根，保留扩展性）
                finger_count = min(finger_count, 10)

        return finger_count

    def process_frame(self, frame):
        """处理单帧并返回处理后的帧和最终手指数量"""
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        results = self.hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        # 调用counter函数获取当前手指数量
        current_count = self.count_fingers(results)

        # 绘制手部关键点
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

        # 结果平滑处理
        self.history.append(current_count)
        if len(self.history) > self.history_length:
            self.history.pop(0)

        count_stats = defaultdict(int)
        for count in self.history:
            count_stats[count] += 1
        final_count = max(count_stats, key=count_stats.get) if count_stats else 0

        # 绘制结果
        cv2.putText(
            frame, self.gesture[final_count], (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3
        )

        return frame, final_count

    def run(self):
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                processed_frame, _ = self.process_frame(frame)
                cv2.imshow('手部识别', processed_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

