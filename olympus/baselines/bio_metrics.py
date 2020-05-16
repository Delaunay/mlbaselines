from sklearn.metrics import roc_curve, auc
import numpy as np

def get_roc_auc(preds, targets):
	fpr, tpr, _  = roc_curve(targets, preds)
	auc_result = auc(fpr,tpr)

	return auc_result

def get_pcc(preds, targets):
	pcc = np.corrcoef(preds, targets)[0,1]
	return pcc

