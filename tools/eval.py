import numpy as np
import pylab as plt
import sklearn.metrics as skm
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.metrics import classification_report, precision_recall_curve, accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve
import torch
import torch.nn.functional as F



def sava_auc(fpr, tpr, path):
    roc_auc = skm.auc(fpr, tpr)
    plt.figure(figsize=(6,6))
    plt.title('ROC')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(path)
    plt.close()

# output: batch_size x 3
# target: batch_size
# pred: batch_size x maxk 
# pred.t: maxk x batch_size
# target.view(1,-1): 1 x batch_size
# correct: maxk x batch_size
def accuracy(target, output, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t() 
        correct = pred.eq(target.view(1, -1).expand_as(pred))
           
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res

#label: [(batchsize x n)]x(total/batchsize) 
#pred: [(batchsize x n) x 3]x(total/batchsize) 
#@profile
def cal_binary_metric(target, output):
    #target = torch.from_numpy(np.array([0, 0, 1, 1]))
    #target = [target]
    #output = torch.from_numpy(np.random.rand(4,2))
    #output = F.softmax(output, dim=1)
    #output = [output]
    #tensor gpu
    #output = [torch.sigmoid(x).cpu().detach().numpy() for x in output]
    #target = [x.cpu().detach().numpy() for x in target]
    #numpy
    output = np.concatenate((output),axis=0)
    target = np.concatenate((target),axis=0)
    output_label = np.argmax(output, axis=1)
    output = output[:,1]
  
    n = target.shape[0]
    
    
    #TN: 预测为负，实际为负  FP: 预测为正，实际为负
    tn, fp, fn, tp = confusion_matrix(target,  output_label).reshape(-1)
    assert ((tn + fp + fn + tp) == n)
    #TPR（true positive rate，真正类率，灵敏度，Sensitivity， 召回率）    
    rec = float(tp) / (tp + fn + 1e-8) 
    #FPR（false positive rate，假正类率）
    fpr = float(fp) / (fp + tn + 1e-8) 
    #TNR（ture negative rate，真负类率，特异度，Specificity）
    spe = float(tn) / (fp + tn + 1e-8)
    #Precision（精确率）
    pre = float(tp) / (fp + tp + 1e-8) 
    #Accuracy（准确率，ACC）
    acc = float(tn + tp) / n  
    #F-Score 是精确率Precision和召回率Recall的加权调和平均值。
    #该值是为了综合衡量Precision和Recall而设定的。
    f1 = 2.0 * rec * pre / (rec + pre + 1e-8)

    #AP衡量的是模型在每个类别上的好坏, 对应Precision-Recall曲线
    ap = average_precision_score(target, output)
    #pre, recall, thresholds = precision_recall_curve(target, output)
    #print(precision_recall_curve(target, output))
    
    #AUC，对应TPR-FPR曲线
    auc = roc_auc_score(target, output)
    #fpr_, tpr_, thresholds = roc_curve(target, output)
    #sava_auc(fpr_, tpr_, 'test.jpg')
    #print(roc_curve(target, output))

    #print(tn, fp, fn, tp)
    rec_2 = recall_score(target, output_label)
    pre_2 = precision_score(target, output_label)
    acc_2 = accuracy_score(target, output_label)
    f1_2 = f1_score(target, output_label) 
     
    #print(classification_report(target, output_label, target_names=['c0', 'c1']))  
    metric = {
        'rec': rec.item(),
        'fpr': fpr.item(),
        'spe': spe.item(),
        'pre': pre.item(),
        'acc': acc,
        'f1': f1.item(),
        'auc': auc.item(),
        'ap': ap.item(),
        'rec_2' : rec_2.item(),
        'pre_2': pre_2.item(),
        'acc_2': acc_2.item(),
        'f1_2': f1_2.item()
    }
    return metric

#未修改多类
#label: [(batchsize x n)]x(total/batchsize) 
#pred: [(batchsize x n) x 3]x(total/batchsize) 
def cal_multiclass_metric(target, output):
    output = [torch.sigmoid(x).cpu().detach().numpy() for x in output]
    target = [x.cpu().detach().numpy() for x in target]
    output = np.concatenate((output),axis=0) #list -> numpy
    target = np.concatenate((target),axis=0)
    output_label = np.argmax(output, axis=1)
    #print(target)
    #print(output_label)
    #print(target.shape[0])

    #纵坐标是target,很坐标是poutput
    #cm = confusion_matrix(target,  output_label)
    acc = accuracy_score(target, output_label)
    #micro macro weighted
    rec_macro = recall_score(target, output_label, average='macro' )
    pre_macro = precision_score(target, output_label, average='macro')
    rec_micro = recall_score(target, output_label, average='micro' )
    pre_micro = precision_score(target, output_label, average='micro')
    rec_weighted = recall_score(target, output_label, average='weighted' )
    pre_weighted = precision_score(target, output_label, average='weighted')
    f1_macro = f1_score(target, output_label, average='macro') 
    f1_micro = f1_score(target, output_label, average='micro') 
    f1_weighted = f1_score(target, output_label, average='weighted') 
     
    #print(classification_report(target, output_label))  
    metric = {
        'acc': acc,
        'pre_macro': pre_macro,
        'rec_macro' : rec_macro,
        'pre_micro': pre_micro,
        'rec_micro' : rec_micro,
        'pre_weighted': pre_weighted,
        'rec_weighted' : rec_weighted,
        'f1_macro': f1_macro,
        'f1_micro' : f1_micro,
        'f1_weighted' : f1_weighted
    }
    return metric

if __name__ == '__main__':
    target = torch.from_numpy(np.array([0, 2, 1, 1]))
    target = [target]
    output = torch.from_numpy(np.random.rand(4,3))
    output = F.softmax(output, dim=1)
    output = [output]
    a = cal_binary_metric(target, output)
    a = cal_multiclass_metric(target, output)
    print(a)
