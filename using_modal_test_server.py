import cv2
import numpy as np
from using_u2net import using_u2net
from time import time
from fastapi.responses import StreamingResponse

shape = (128, 128)
LEN_DATA_RES = 5
class_name = [' VMWG9925721009X-20190327 - Đèn vàng T5G Trái ',
              'VMWGJ664820003M-20200810- Bảng điều khiển điều hòa A7F', 'VM199012210078M-20200321 - BI T 420  ',
              'Tay mở cửa ngoài T5G bên phụ VM810W62641-6078M', 'HW21716XSTL - Xilanh (nhôm)',
              'VMVG1500050032S-202003 - Cốc đũa đẩy', 'VMVG1540080300S-20190619 - KhỚP nối cao cáp 371 ',
              'VMWG1500061203-1M-20110907 - Vỏ van hằng nhiệt WD615  (1)',
              'VMWG92313202005M-20200910- Mặt bích các đầu nối các đăng',
              'VMAZ9719570002-20110907 - Chân ga HOWO  ', 'VMWG9925720012A7S-20200919- Đèn xi nhan sườn trái A7',
              'VMAZ1630840014S-20110907 – Quạt giàn lạnh HOWO ', 'VM6313S-20200919- Vòng bi đầu trục con lợn',
              'VMWG9000360100BR5-20190301 - Bầu phanh trước ', 'Bạc tay biên WD615 – VMVG1500030077EN5-20200919',
              'VMAZ1664430103S - Giảm sóc bóng hơi cabin trước A7 ',
              'VMWG9925550703S-20201117 Báo dầu nhiên liệu A7', ' HSHW905100 Bánh răng chính chuyển tầng',
              'VMWG9731471025S-20201114- Bơm trợ lực  lái  2014',
              'VM711W61900-0050FN5-20200801 - Lọc điều hòa ', 'VMA70001M-20200810 Giảm sóc ghế hơi A7',
              'VMWG1500061203-1M-20110907 - Vỏ van hằng nhiệt WD615 ',
              'VMWG9719810011S-20110907 Đèn hậu HOWO ( 4 ốc trái)', 'VMAZ9725520238M-20180101 - Bạc ba lăng C',
              'VMWG1246030011S-20200919 Bạc biên trên D12  ', 'VM711W30715-6152S-20200919 - Tổng  côn trên T5G ',
              'VMWG1642821074S-20110907 - Giàn  nóng điều hòa HOWO T7H ',
              'AC26 – 1WG997032127S-202000 - Càng cua cài cầu', 'VMWG9000360523S-20201114 - Bộ chia hơi',
              'VMWG9112530333S-20110907-  Bình nước phụ', ' VMWG9000360170CA5-20200311 - Đầu nối kéo mooc',
              'VMWG9000360524S-20200919 - Bộ chia  hơi ', 'VMWG2203010008M-20201203 - Lọc hộp số HW10T',
              'VMWG1664110006M-20200810 - Logo Sinotruck   ', 'VMVG124060023S-20200801 - Vỏ van hằng nhiệt D12 ',
              'VMWG9719530275E1 - Chân két nước ', 'VM199100520006K-20110907 - Ắc nhíp',
              'VMWG9731471225S-20201114- Bơm trợ lực  lái  420',
              'VMWG9725593016S-20201114 Giá bắt động cơ ', 'VMWG9918780001K-20110907 - Đài DVD  A7 ']


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
