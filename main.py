from ImagePreprocessingAndTextDetection import ImageData

if __name__ == '__main__':
    test = ImageData("test2.jpg")
    test.plot_preprocessed_image()
    candidates = test.get_text_candidates()
    test.plot_to_check(candidates,'Total Objects Detected')