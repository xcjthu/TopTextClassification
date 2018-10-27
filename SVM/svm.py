from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report as report
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from sklearn.externals import joblib
from read import getRunDataset

def trainTFIDF(x_train, x_test):
	tfidf = TFIDF(min_df=2, strip_accents="unicode", analyzer="word", token_pattern=r"\w{1,}", ngram_range=(1,3), max_features = 10000, use_idf=1,smooth_idf=1,sublinear_tf=1)
	tfidf.fit(x_train + x_test)
	joblib.dump(tfidf, 'tfidf_1w.model')
	return tfidf.transform(x_train), tfidf.transform(x_test)

def loadTFIDF(x_train, x_test):
	print('loading')
	tfidf = joblib.load('tfidf_1w.model')
	print('transform')
	return tfidf.transform(x_train), tfidf.transform(x_test)


def trainSVM(x_train, y_train):
	svm = LinearSVC()
	svm.fit(x_train, y_train)
	joblib.dump(svm, 'svm_1w.model')
	return svm

def predict(svm, x_test, y_test):
	y_pred = svm.predict(x_test)
	fout = open('result_1w.txt', 'w')
	print(report(y_test, y_pred), file = fout)


if __name__ == '__main__':
	print('Begin read data')
	x_train, x_test, y_train, y_test = getRunDataset()
	'''
	x_train = x_train[:1000]
	x_test = x_test[:1000]
	y_train = y_train[:1000]
	y_test = y_test[:1000]
	'''
	print('train tfidf')
	x_train_ma, x_test_ma = loadTFIDF(x_train, x_test)

	print('train SVM')
	SVM_model = trainSVM(x_train_ma, y_train)

	print('predict')
	predict(SVM_model, x_test_ma, y_test)
