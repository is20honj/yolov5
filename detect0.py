import sys
import cv2
import torch

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



IMAGE_PATH = sys.argv[1]  #解析する画像のパス
CONFIDENCE = 40  #信頼度の閾値 [%]


if __name__ == '__main__':

    yolo = YOLO('yolov5s')  #インスタンスを生成

    image = cv2.imread(IMAGE_PATH)  #画像読み込み
    results = yolo.getBoundingBox(image)  #物体検出

    for result in results:
        #信頼度が閾値を上回っていた場合は緑色の矩形を描画
        if result[4] > CONFIDENCE / 100:
            left, top, right, bottom = result[:4]
            image = cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 3)

    #結果を表示
    cv2.imshow('', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()