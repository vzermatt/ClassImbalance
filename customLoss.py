''''
    Implementation of Class balanced Loss and the Equalization loss 

    Reference: "Class-Balanced Loss Based on Effective Number of Samples" 
    Authors: Yin Cui and Menglin Jia and Tsung Yi Lin and Yang Song and Serge J. Belongie
    https://arxiv.org/abs/1901.05555, CVPR'19.

    Official Repository (Tensorflow) :
    https://github.com/richardaecn/class-balanced-loss
    
    Pytorch implementation of Class-Balanced-Loss:
    https://github.com/vandit15/Class-balanced-loss-pytorch

    Reference: "Equalization Loss for Long-Tailed Object Recognition"
    Authors : Tan, J., Wang, C., Li, B., Li, Q., Ouyang, W., Yin, C., & Yan, J. (2020) 
    https://openaccess.thecvf.com/content_CVPR_2020/html/Tan_Equalization_Loss_for_Long-Tailed_Object_Recognition_CVPR_2020_paper.html
    
    Official Repository:
    https://github.com/tztztztztz/eql.detectron2/issues/9

'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, pickle, glob
import util


def Compute_frequency_weights(freq,samples_per_cls=None):
    '''
        Compute class weights from the training set distribution
        input : type of class reweighting : 
                - simple: inverse frequency reweigthing
                - square: inverse squared frequency reweighting
        output : class weights with mean =1
    '''
    if samples_per_cls == None:
        samples_per_cls = [204, 1257, 301, 108, 121, 114, 61, 906, 121, 69, 75, 88, 118, 496, 1863, 447, 1321, 1200, 316, 4455,  105,  11194, 525, 172, 174, 413,  9102, 90]  

    # total number of samples : 
    tot = sum(samples_per_cls)

    # inverse frequency weights : 
    if freq =='simple':
       # print('simple inverse class frequency')
        class_weights = [tot /x  for x in samples_per_cls]   
    elif freq =='sqrt':#  inverse square root frequency weights :
       # print('square root inverse class frequency')
        class_weights = [tot /(x**0.5)  for x in samples_per_cls]
        
    # mean value =1 to avoid vanishing gradient
    mean = np.mean(class_weights)
    class_weights = [x / mean for x in class_weights] 
    
    return  torch.FloatTensor(class_weights)

def Compute_CB_weights(samples_per_cls, no_of_classes, beta):
    """
    Compute the Class Balanced weights 

    Args:
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.

    Returns:
     weights: A float tensor representing class balanced weights
    """
    device = ("cuda:0" if torch.cuda.is_available() else "cpu")

    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes
    weights = torch.Tensor(weights).float()
    return weights

def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """    
    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")
    
    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + 
            torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss

    focal_loss = torch.sum(weighted_loss)
    
    focal_loss /= torch.sum(labels)

    if False:    
        print('\n\nlogits\n\n',logits,'\nlabels\n', labels[0,:])
        print('\n\nbinary cross entropy loss: \n\n',BCLoss)
        print('\nmodulator :\n',modulator)
        print('\n bce loss * modulator :\n',loss)
        print('\nfocal loss\n\n',focal_loss,'-'*100)

    return focal_loss

def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma):
    """
    Compute the Class Balanced Loss between `logits` and the ground truth `labels`.

    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)

    where Loss is one of the standard losses used for Neural Networks.

    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.
    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    device = ("cuda:0" if torch.cuda.is_available() else "cpu")

    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes
    weights = torch.Tensor(weights).float()
    weights1 = weights.to(device)

    labels_one_hot = F.one_hot(labels, no_of_classes).float()

    weights = weights1.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1,no_of_classes)
    

    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        criterion =torch.nn.BCEWithLogitsLoss(weight= weights)
        #cb_loss = 100* criterion(logits, labels_one_hot)
        cb_loss =  F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weight = weights)
    elif loss_type == "softmax":
        #pred = logits.softmax(dim = 1)
        #cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
        # my tests : 
        #cb_loss = F.cross_entropy(input = logits, target = labels, weight = weights1)
        cb_loss = torch.nn.CrossEntropyLoss(logits, labels, weights1)

    return cb_loss

def get_eql_class_weights( lambda_):
    # select majority class: more than lambda samples (weights =1)    
    dico_path = '/home/valerie/Python/landuse/Dictionnaries/dict_NOLU46_all.p' # clean 
    set_list = '/home/valerie/Python/landuse/tile_list/train.csv'# clean train!
    samples_per_cls = util.get_samples_per_class(  set_list, dico_path)
    class_weights = np.zeros(len(samples_per_cls))
    labels = []
    idx = 0
    for idx, count in enumerate (samples_per_cls):
        class_weights[idx] = 1 if count >lambda_ else 0
    return class_weights

def replace_masked_values(tensor, mask, replace_with):
    '''
        locations in tensor with mask =1 are unchanged
        locations in tensor with mask =0 are replaced by replace_with 
    '''
    assert tensor.dim() == mask.dim(), '{} vs {}'.format(tensor.shape, mask.shape)
    one_minus_mask = 1 - mask       # select areas to replace with value 1
    values_to_add = replace_with * one_minus_mask # new values 
    return tensor * mask + values_to_add

class SoftmaxEQL(object):
    '''
        Implementation of the Equalization loss for multiclas classification with softmax cross entropy loss
        lambda_ : threshold for minority classes
        ignore_prob : beta value used to randomly maintain the gradient of minority samples
    '''
    def __init__(self, lambda_, ignore_prob):
        self.lambda_ = lambda_
        self.ignore_prob = ignore_prob  # gamma
        self.class_weights = torch.Tensor(get_eql_class_weights(self.lambda_)).cuda()

    def __call__(self, inputs, target):
        N, C = inputs.size()
        not_ignored = self.class_weights.view(1, C).repeat(N, 1).cuda() # 1 - tail ratio function
        over_prob = (torch.rand(inputs.shape).cuda() > self.ignore_prob).float() 
        is_gt = target.new_zeros((N, C)).float()
        is_gt[torch.arange(N), target] = 1  # "one hot encoding" of the ground truth
        weights = ((not_ignored + over_prob + is_gt) > 0).float() # w_tild_k = 0 
        # inputs with weights= 1 are unchanged, others become -1e7
        inputs2 = replace_masked_values(inputs, weights, -1e7) # set a very unfavorable value for selected predictions probabilities with softmax
        loss = F.cross_entropy(inputs2, target)
        return loss

class FocalEQL(object): # to be deleted
    '''
        Implementation of the Equalization loss for multiclas classification with focal loss
        lambda_ : threshold for minority classes
        ignore_prob : beta value used to randomly maintain the gradient of minority samples
        gamma : parameter for the focal loss
    '''
    def __init__(self, lambda_, ignore_prob, gamma):
        self.lambda_ = lambda_
        self.ignore_prob = ignore_prob  # gamma
        self.class_weights = torch.Tensor(get_eql_class_weights(self.lambda_)).cuda()
        self.gamma =gamma

    def __call__(self, inputs, target):
        N, C = inputs.size()
        not_ignored = self.class_weights.view(1, C).repeat(N, 1).cuda() # 1 - tail ratio function
        over_prob = (torch.rand(inputs.shape).cuda() > self.ignore_prob).float() 
        is_gt = target.new_zeros((N, C)).float()
        is_gt[torch.arange(N), target] = 1  # "one hot encoding" of the ground truth
        weights = ((not_ignored + over_prob + is_gt) > 0).float() # w_tild_k = 0 
        # inputs with weights= 1 are unchanged, others become -1e7
        inputs = replace_masked_values(inputs, weights, -1e7) # set a very unfavorable value for selected predictions probabilities with softmax
        
        labels_one_hot = F.one_hot(target, C).float().cuda()
        weights = torch.ones(N,C).cuda()
        #print('labels:\n',labels_one_hot[0,:], '\ninputs\n',inputs, '\ngamma\n',self.gamma)
        loss =  focal_loss(labels_one_hot, inputs, weights, self.gamma)

        return loss


def  MyFocalLoss(labels, logits, samples_per_cls, no_of_classes, gamma):
    """
    Compute the Focal Loss between `logits` and the ground truth `labels`.

    Focal Loss: alpha*Loss(labels, logits)
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.
    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    device = ("cuda:0" if torch.cuda.is_available() else "cpu")

    weights = Compute_frequency_weights(freq='sqrt',samples_per_cls=None)
    weights = weights.numpy()
    weights = weights / np.sum(weights) * no_of_classes
    weights = torch.Tensor(weights).float()
    weights = weights.to(device)

    labels_one_hot = F.one_hot(labels, no_of_classes).float()

    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1,no_of_classes)
    
    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels_one_hot, reduction = "none")
    
    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels_one_hot * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = weights * loss

    focal_loss = torch.sum(weighted_loss)
    
    focal_loss /= torch.sum(labels)    
    if False:    
        print('\n\nlogits\n\n',logits.size(),'\nlabels\n', labels_one_hot.size())
        print('\n\nbinary cross entropy loss: \n\n',BCLoss.size() )
        print('\nmodulator :\n',modulator)
        print('\n bce loss * modulator :\n',loss)
        print('\nfocal loss\n\n',focal_loss,'-'*100)

    return focal_loss






