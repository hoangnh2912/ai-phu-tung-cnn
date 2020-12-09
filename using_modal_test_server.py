import cv2
import numpy as np
from using_u2net import using_u2net
from time import time
from fastapi.responses import StreamingResponse

shape = (128, 128)
LEN_DATA_RES = 5
class_name = ['Cổ xả động cơ 375 VMVG1500110902S-20200714', 'Ống hơi EGR VMVG10961',
              'Giảm sóc bóng hơi cabin sau A7_VMAZ1664440068M-20110907',
              'Giảm sóc ca bin loại nhỏ VMWG1642440021S-20200919',
              'Phớt đầu trục cơ WD615 - VMVG1047010038M-20200810', 'Đài DVD A7 _ VMWG9918780001K-20110907',
              'Van hằng nhiệt WD615_VMVG1047060002S-20200919',
              'Mặt bích các đăng nối cầu_VMWG92313202005M-20200910',
              'Khớp nối bớm cao áp - VMVG1540080300S-20190619', 'Van hằng nhiệt WD615',
              'Bánh răng mặt trời cài vi sai MCY_VM810-35617-6007M-20200910', 'Lọc hơi máy VMVG1557010015E1',
              'Chân cao su chân két nước VMWG9719530275EN5', 'Bơm trợ lực lái 420- VMWG9731471225s-201912',
              'Trục lai BR trung gian D12 VMVG1246050034M-20200910',
              'Khớp nối bơm cao áp ERG 375 - VMVG1092080401S-20200804',
              'Ống dẫn dầu Turbo tăng áp_VM202V507', 'Tăng dây cu roa VMVG1246060005S-20200804',
              'Bạc càng chữ U đỡ cabin VM1642430061M 20201022', 'Cao su chân cabin A7_VMWG1664430095M-20200901',
              'Bạc trục con  lợn MCY_VM811W93021-0398M-2018010', 'Tuy uy cao áp D12',
              'Trục chữ thập vi sai cầu giữa_ VMAZ923130150S',
              'Cao su đỡ nhíp sau - 1WG9725520727S-20200919', 'Khúc khuỷu VMVG1500040105M',
              'Ly tâm cánh quạt D12 VMLT1246060030M-20200810',
              'Cụm cơ cấu đóng mở cúp bô VMWG9725542045-1M',
              'Cao su giằng cân bằng VM199100680067S-20200919', 'Xu páp xả VMG15600',
              'Cao su chân máy trước VMWG1680590095M-20200901', 'Bơm tay nhiên liệu MVG1242088004S-20200919',
              'Bi chữ thập (70x166) VMWG9370310010-1M-20200810',
              'Gioăng turbo động cơ 375 A7 VMVG10341100541M-200910']


def using(path, my_model, host_name, port):
    try:
        last_res = []
        img = using_u2net(cv2.imread(path))
        img = cv2.resize(img, shape)
        image = cv2.resize(img, dsize=shape)
        cv2.imwrite('cache/test.jpg', image)
        predict = my_model.predict(np.array([image]))
        predict = predict[0].tolist()
        predict.sort()
        predict.reverse()
        for idx, con in enumerate(predict):
            if len(last_res) <= LEN_DATA_RES:
                name = class_name[idx]
                last_res.append({'name': name, 'consider': round(float(con), 3) * 100})

        return {
            'data': {
                'predict': last_res,
                'image_url': 'http://' + host_name + ":" + str(port) + "/image?file=out.jpg"
            },
            'code': 200,
            'message': "Thành công",
            'status': 1
        }
    except ValueError:
        return {
            'data': [],
            'code': 500,
            'message': "Có lỗi xảy ra",
            'status': 0
        }
