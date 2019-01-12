import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from tensorflow.contrib import learn
import csv
import sys

sys.path.append(r'../common/')
from txt_Word2Vec import Tibet_Word2Vec
sys.path.append(r'../cnn_classfication/')
from text_cnn import TextCNN
     
# Data Parameters
tf.flags.DEFINE_string("jsonfile", "./../../Resources/jsonfiles/ChnSentiCrop.json", "Data source for the json file.")
tf.flags.DEFINE_string("cutwordfile", "./../../Resources/CutWordPath/ChnSentiCrop_cut.txt", "Data source for the cutword save file.")
tf.flags.DEFINE_string("labelfile", "./../../Resources/labels/ChnSentiCrop_label.txt", "label save file")
tf.flags.DEFINE_string("w2v_file", "./../../Resources/Binaryfiles/sentiment.bin", "binary file")
# traincutwordfile
tf.flags.DEFINE_string("traincutwordfile", "./../../Resources/CutWordPath/sentiment_cut.txt", "Data source for the cutword save file.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
# 需要修改路径
tf.flags.DEFINE_string("checkpoint_dir", "./../cnn_sentiment/runs/1544769089/checkpoints/", "Checkpoint directory from training run")
# 需要修改为TRUE
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
# 版本更新问题
FLAGS.flag_values_dict()
print("\nParameters:")
for attr, value in sorted(FLAGS.flag_values_dict().items()):
    print("{}={}".format(attr.upper(), value))
print("")

# 选择训练集和测试集的最大句子长度中较大的那一个
def compute_maxdocument_length(filename,x_raw):
    text=list(open(filename,"r",encoding='utf-8').read().splitlines())
    actual_length=max(len(x.split(" ")) for x in text)  # 训练集的最大句子长度
    max_document_length= max([len(x.split(" ")) for x in x_raw])  # 测试集的最大句子长度
    x_raw1=[]
    if actual_length > max_document_length:
        max_document_length=actual_length
        x_raw1.extend(x_raw)
    else:      
        for x in x_raw:
            list1=x.split(" ")   
            if len(list1) >= actual_length:
                list1=list1[0:actual_length]  #截断
                x_raw1.append(" ".join(list1))
            else:
                x_raw1.append(" ".join(list1))
    return x_raw1,max_document_length


def load_data(w2v_model):
    # CHANGE THIS: Load data. Load your own data here
    if FLAGS.eval_train:
        # x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.cutwordfile,FLAGS.jsonfile,FLAGS.labelfile)
        # 第二次调用
        x_raw=open(FLAGS.cutwordfile,"r",encoding='utf-8').read().splitlines()
        y_test=np.loadtxt(FLAGS.labelfile)
        # 返回最大值所在的下标
        y_test = np.argmax(y_test, axis=1)
    else:
        x_raw = ["a masterpiece four years in the making", "everything is off."]
        y_test = [1, 0]

    # 修改新闻内容和最大文档长度
    x_raw1,max_document_length=compute_maxdocument_length(FLAGS.traincutwordfile,x_raw)

    print('正在生成词典')
    x=data_helpers.get_text_idx(x_raw1,w2v_model.model.wv.vocab,max_document_length)
    return x_raw1,x,y_test

    # # Map data into vocabulary
    # vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
    # # 创建词汇表
    # vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    # # 文本转为词ID序列
    # x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

# Evaluation
# ==================================================
def eval(w2v_model):
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            
            x_raw,x_test,y_test=load_data(w2v_model)  # 加载数据
            # Generate batches for one epoch
            batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1,shuffle=False)

            # Collect the predictions here
            all_predictions = []

            for x_test_batch in batches:
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

    # print("\nallpredictions:",all_predictions[400:500])
    # print("\ny_test:",y_test[400:500])

    if y_test is not None:
        correct_predictions = float(sum(all_predictions == y_test))
        print("Total number of test examples: {}".format(len(y_test)))
        print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))


    # Save the evaluation to a csv
    predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
    out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
    print("Saving evaluation to {0}".format(out_path))
    with open(out_path,'w',newline='',encoding='utf-8') as f:
        csv.writer(f).writerows(predictions_human_readable)
    
if __name__ == '__main__':
    w2v_wr = data_helpers.w2v_wrapper(FLAGS.w2v_file)
    eval(w2v_wr)
    

    