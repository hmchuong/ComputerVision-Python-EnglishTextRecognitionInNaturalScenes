# -*- coding: utf-8 -*-
from ImagePreprocessingAndTextDetection import ImageData
from data import OcrData

if __name__ == '__main__':
    # 1. IMAGE PREPROCESSING AND OBJECT DETECTION
    # Tải ảnh cần detect
    test = ImageData("test.jpg")
    # Biểu diễn việc tiền xử lý ảnh
    test.plot_preprocessed_image()
    # Detect các đối tượng trong ảnh đã tiền xử lý
    candidates = test.get_text_candidates()
    # Biển diễn các object đã detect được
    test.plot_to_check(candidates,'Total Objects Detected')

    # 2. TEXT DETECTION
        # a. Tạo mô hình dự đoán 1 đối tượng là có gồm chữ hay không (chỉ thực hiện bước này 1 lần để tạo file pickle)
            # B1. Nạp tập dữ liệu OCR từ file config
    #data = OcrData('ocr-config.py')
    #print data.labels
            # B2. Trộn với tập CIFAR, sau khi chạy bước 1 thì chỉnh sửa lại file text-config.py rồi mới chạy tiếp để tiết kiệm thời gian chạy
    #data.merge_with_cifar()
            # B3. Biễn diễn Grid Search Cross Validation để tìm mô hình tốt nhất từ tham số đưa vào
    #data.perform_grid_search_cv('linearsvc-hog')
            # B4. Huấn luyện lại thêm 1 lần nữa trên toàn bộ tập trên dựa trên mô hình tốt nhất lấy được từ bước 3 và tập dữ liệu 100 000 ảnh bước 2
    #data.generate_best_hog_model()
            # B5. Đánh giá mô hình trên tập train (sau khi hoàn thành các bước trên mới thực hiện bước này từ file pickle
    #data.evaluate('Dataset/Chars74K/linearsvc-hog-fulltrain-2016-11-24 23-33-58.042000.pickle')

    #Chọn các đối tượng có kí tự
    maybe_text = test.select_text_among_candidates('Dataset/Chars74K/linearsvc-hog-fulltrain-2016-11-24 23-33-58.042000.pickle')
    #Hiển thị kết quả sau khi xác định
    test.plot_to_check(maybe_text, 'Objects Containing Text Detected')
