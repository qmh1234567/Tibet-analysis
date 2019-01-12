import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
import sys
sys.path.append(r'../common/')
from txt_Word2Vec import Tibet_Word2Vec,LoadWordList
sys.path.append(r'../fpgrowth/')
import fp_data     
# Data Parameters
tf.flags.DEFINE_string("jsonfile", "./../../Resources/jsonfiles/fudan_test.json", "Data source for the json file.")
tf.flags.DEFINE_string("cutwordfile", "./../../Resources/CutWordPath/fudan_test.txt", "Data source for the cutword save file.")
tf.flags.DEFINE_string("labelfile", "./../../Resources/labels/fudan_test_label.txt", "label save file.")
tf.flags.DEFINE_string("w2v_file", "./../../Resources/Binaryfiles/fudan_train_vector.bin", "binary file")
tf.flags.DEFINE_string("Keywordfile", "./../../Resources/CutWordPath/fudan_test_keyword.txt", "cutKeywordfile")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
# 需要修改路径
tf.flags.DEFINE_string("checkpoint_dir", "./../cnn_classfication/runs/1545199870/checkpoints/", "Checkpoint directory from training run")
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

# 生成词向量文件
# model=Tibet_Word2Vec(FLAGS.cutwordfile,FLAGS.w2v_file)

def load_data(w2v_model):
    # CHANGE THIS: Load data. Load your own data here
    if FLAGS.eval_train:
        # 第一次调用
        # x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.cutwordfile,FLAGS.jsonfile,FLAGS.labelfile)
        print("分词结束")
        # 第二次调用
        x_raw=open(FLAGS.cutwordfile,"r",encoding='utf-8').read().splitlines()
        y_test=np.loadtxt(FLAGS.labelfile)
        # print(len(x_raw),len(y_test))
         # 调用tf-idf方法提取300个关键词
        keywordlists=fp_data.TF_IDF_keyword(FLAGS.cutwordfile,300)
        fp_data.write_lists_to_file(keywordlists,FLAGS.Keywordfile)
        x_raw=LoadWordList(FLAGS.Keywordfile)
        # print(len(x_raw),len(y_test))
        # 返回最大值所在的下标
        y_test = np.argmax(y_test, axis=1)
        # print(list(y_test).count(0))
        # print(list(y_test).count(1))
        # print(list(y_test).count(2))
    else:
        x_raw = ["a masterpiece four years in the making", "everything is off."]
        y_test = [1, 0]
    max_document_length= max([len(x.split(" ")) for x in x_raw])
    x=data_helpers.get_text_idx(x_raw,w2v_model.model.wv.vocab,max_document_length)
    return x_raw,x,y_test

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

    print("\nallpredictions:",all_predictions[400:500])
    print("\ny_test:",y_test[400:500])

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
    

    