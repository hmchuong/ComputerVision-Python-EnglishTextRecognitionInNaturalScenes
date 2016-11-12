from ImagePreprocessingAndTextDetection import ImageData
from cifarKaggle import CifarKaggle

if __name__ == '__main__':
    cifar = CifarKaggle('cifar-config.py')
    
    test = ImageData("test2.jpg")
    test.plot_preprocessed_image()
    candidates = test.get_text_candidates()
    test.plot_to_check(candidates,'Total Objects Detected')
    