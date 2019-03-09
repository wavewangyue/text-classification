"""
    Note: Before training fasttext, you should go to https://github.com/facebookresearch/fastText
install fastText as instruction in README.md.
"""
import fastText


def data_preprocess(input_file_content, input_file_labels, output_file):
    """
    Only used for convert train and test data to fasttext format.
    :return: no return
    """
    infile_content = open(input_file_content, "r")
    infile_labels = open(input_file_labels, "r")
    outfile = open(output_file, "w")
    head="__label__"
    content_list = infile_content.readlines()
    labels_list = infile_labels.readlines()
    for i, content_line in enumerate(content_list):
        new_line = head + labels_list[i][:-1] + " " + content_line
        outfile.write(new_line)
    outfile.close()


if __name__ == "__main__":
    # data_preprocess("train_contents.txt", "train_labels.txt", "fasttext_train.txt")
    # data_preprocess("test_contents.txt", "test_labels.txt", "fasttext_test.txt")

    model_save = "fasttext_model"
    train_file = "fasttext_train.txt"
    test_file = "fasttext_test.txt"
    classifier = fastText.train_supervised(input=train_file, dim=100, epoch=20, lr=0.1, wordNgrams=1, bucket=2000000)

    result = classifier.test(test_file)
    print(result)
