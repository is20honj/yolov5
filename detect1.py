import sys
import cv2
import torch
import os
import asyncio
import uuid
from azure.iot.device.aio import IoTHubDeviceClient
from azure.iot.device import Message

#YOLOはYOLOでClassを作って処理を隔離
class YOLO:

    #コンストラクタでモデルを読み込み
    def __init__(self, pt_path):
        self.__model = torch.hub.load('.', 'custom', path=pt_path, source='local')

    #物体のleft, top, right, bottomを出力
    def getBoundingBox(self, frame, log=False):
        data = []
        result = self.__model(frame)
        if log:
            print('----------')
            print(result.pandas().xyxy[0])
        ndresult = result.xyxy[0].numpy()
        for v in ndresult:
            if v[5] == 0:  #今回は人だけ抽出したかったので「0=person」
                data.append([
                    int(v[0]),  #left
                    int(v[1]),  #top
                    int(v[2]),  #right
                    int(v[3]),  #bottom
                    float(v[4])  #confidence
                ])
        return data

CONFIDENCE = 40  #信頼度の閾値 [%]

# VideoCaptureのインスタンスを作成する。
# 引数でカメラを選べる。
cap = cv2.VideoCapture(0)

while True:
    yolo = YOLO('yolov5s')  #インスタンスを生成
    # VideoCaptureから1フレーム読み込む
    ret, frame = cap.read()

    # 加工処理
    edframe = frame

    results = yolo.getBoundingBox(edframe)  #物体検出

    PERSON_COUNT = 0

    for result in results:
        #信頼度が閾値を上回っていた場合は緑色の矩形を描画
        if result[4] > CONFIDENCE / 100:
            PERSON_COUNT = PERSON_COUNT + 1
            left, top, right, bottom = result[:4]
            edframe = cv2.rectangle(edframe, (left, top), (right, bottom), (0, 255, 0), 3)

    # 加工済の画像を表示する
    cv2.imshow('Edited Frame', edframe)
    
    async def main():
        # The connection string for a device should never be stored in code. For the sake of simplicity we're using an environment variable here.
        conn_str = os.getenv("IOTHUB_DEVICE_CONNECTION_STRING")

        # The client object is used to interact with your Azure IoT hub.
        device_client = IoTHubDeviceClient.create_from_connection_string(conn_str)

        # Connect the client.
        await device_client.connect()

        #メッセージとして出力
        print("sending message:")
        msg = Message(str(PERSON_COUNT)+"persons")
        msg.message_id = uuid.uuid4()
        msg.correlation_id = "correlation-1234"
        msg.custom_properties["tornado-warning"] = "yes"
        msg.content_encoding = "utf-8"
        msg.content_type = "application/json"
        await device_client.send_message(msg)
        print("done sending message:")
        
        # Finally, shut down the client
        await device_client.shutdown()
            
    if __name__ == "__main__":
        asyncio.run(main())

    # キー入力を1ms待って、k が27（ESC）だったらBreakする
    k = cv2.waitKey(1)
    if k == 27:
        break

# キャプチャをリリースして、ウィンドウをすべて閉じる
cap.release()
cv2.destroyAllWindows()