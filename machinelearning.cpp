// ***********************************************************
// Machine Learning Algorithms
// CS 109 - Fall 2014
// Helen Anderson & Maggie Goulder
// ***********************************************************
// This program implements a Naive Bayesian algorithm (using 
// Maximum Likelihood Estimators and Laplace Estimators) and 
// a Logistic Regression algorithm.
// ***********************************************************


#define _USE_MATH_DEFINES
#define EPOCHS 10000
#define LEARN_RATE 0.0001

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <stack>
#include <math.h>

using namespace std;

void readDataFile(string filename, int &variableNum, int &vectorNum, vector<vector<double> > &data, vector<double> &YVals, bool laplace, bool logreg) {
	string line;
	ifstream file (filename);
	getline(file,line);
	variableNum = stoi(line);
	getline(file,line);
	vectorNum = stoi(line);

	//if using laplace, add four to vector count
	if (laplace) {
		vectorNum += 4;
	}

	for (int i = 0; i < vectorNum; i++) {
		vector<double> onedata;
		data.push_back(onedata);

		//if using logistic regression, set all first values to 1 (to account for alpha)
		if (logreg) {
			data[i].push_back(1.0);	
		}	

		getline(file,line);
		string buffer;
		stringstream ss(line);
		for (int k = 0; k < variableNum; k++) {
			ss >> buffer; 
			int value = stoi(buffer);
			data[i].push_back(value);
		}
		ss >> buffer;
		int y = stoi(buffer);
		YVals.push_back(y);
	}
	file.close();
}

vector<vector<vector<int> > > trainNaiveBayes(string filename, bool laplace){

	int variableNum;
	int vectorNum; 
	vector<double> allYs;
	vector<vector<double> > alldata; 

	readDataFile(filename, variableNum, vectorNum, alldata, allYs, laplace, false);

	//initialize all values in 3D vector
	vector<vector<vector<int> > > trainingData;
	vector<vector<int> > twodplaceholder;
	vector<int> onedplaceholder;
	
	for (int i = 0; i < variableNum; i++) {

		trainingData.push_back(twodplaceholder);		
		for (int j = 0; j < 2; j++) {
			trainingData[i].push_back(onedplaceholder);
			for (int k = 0; k < 2; k++) {
				trainingData[i][j].push_back(0);
			}
		}
	}

	//count how many of the total y values are 1's and how many are 0's
	int ycount[2] = {0,0};
	for (int i = 0; i < allYs.size(); i++) {
		int y = allYs[i];
		ycount[y]++;
	}

	//count data totals and write them to 3D vector
	for (int i = 0; i < vectorNum; i++) {
		for (int j = 0; j < variableNum; j++) {
			int x = alldata[i][j];
			int y = allYs[i];
			trainingData[j][x][y]++;
		}	
	}

	vector<vector<int> > yVector(2);
	yVector[0].push_back(ycount[0]);
	yVector[1].push_back(ycount[1]);
	trainingData.push_back(yVector);

	return trainingData;
}

void testNaiveBayes(string filename, vector<vector<vector<int> > > trainingData) {

	int variableNum;
	int vectorNum;
	vector<double> testYs;
	vector<vector<double> > testData; 

	readDataFile(filename, variableNum, vectorNum, testData, testYs, false, false);

	int classcount[2] = {0,0};
	int correctcount[2] = {0,0}; 

	int ycount[2] = {0,0};
	double yprob[2] = {0.0,0.0};

	ycount[0] = trainingData[variableNum][0][0];
	ycount[1] = trainingData[variableNum][1][0];

	yprob[0] = (double)ycount[0]/(double)vectorNum;
	yprob[1] = (double)ycount[1]/(double)vectorNum;

	//count how many of the test y values are 1's and how many are 0's
	for (int i = 0; i < testYs.size(); i++) {
		int y = testYs[i];
		classcount[y]++;
	}

	for (int i = 0; i < vectorNum; i++) {

		double prob[2] = {0.0,0.0};
		for (int j = 0; j < 2; j++) {
			prob[j] = yprob[j];
			for (int k = 0; k < variableNum; k++) {
				int value = testData[i][k];
				int count = trainingData[k][value][j];
				prob[j] *= count;
			}
			double denom = pow(ycount[j], variableNum);
			prob[j] /= (double)denom;
		}
		int yGuess = 0;
		if (prob[1] > prob[0]) {
			yGuess = 1;
		}
		if (yGuess == testYs[i]) {
			correctcount[yGuess]++;
		}
	}

	cout << "Class 0: tested " << classcount[0] << ", correctly classified " << correctcount[0] << endl;
	cout << "Class 1: tested " << classcount[1] << ", correctly classified " << correctcount[1] << endl;
	cout << "Overall: tested " << classcount[0] + classcount[1] << ", correctly classified " << correctcount[0] + correctcount[1] << endl;
	cout << "Accuracy = " <<  (double)(correctcount[0] + correctcount[1])/(double)(classcount[0] + classcount[1]) << endl;

	return;
}

vector<double> computeZVals(vector<vector<double> > data, vector<double> betas, int variableNum, int vectorNum){
	vector<double> zVals;
	for (int i = 0; i < vectorNum; i++) {
		double z = 0.0;
		zVals.push_back(z);
		for (int j = 0; j <= variableNum; j++) {
			double toAdd = betas[j]*data[i][j];
			zVals[i] += toAdd;
		}
	}
	return zVals;
}

vector<double> trainLogReg(string filename) {
	int variableNum;
	int vectorNum; 
	vector<double> allYs;
	vector<vector<double> > alldata; 

	readDataFile(filename, variableNum, vectorNum, alldata, allYs, false, true);


	vector<double> betas(variableNum + 1);
	for (int i = 0; i < betas.size(); i++) {
		betas[i] = 0.0;
	} 

	for (int e = 0; e < EPOCHS; e++) {

		vector<double> zVals = computeZVals(alldata, betas, variableNum, vectorNum);
		
		vector<double> gradients(variableNum + 1);
		for (int k = 0; k <= variableNum; k++) {
			double gradient = 0.0;
			gradients.push_back(gradient);
			for (int i = 0; i < vectorNum; i++) {
				double toAdd = alldata[i][k]*(allYs[i] - (1/(1 + pow(M_E, -zVals[i]))));
				gradients[k] += toAdd;
			}
		}
		
		for (int k = 0; k <= variableNum; k++) {
			betas[k] += (LEARN_RATE*gradients[k]);
		}
	}
	return betas;
}


void testLogReg(string filename, vector<double> trainData) {	

	int classcount[2] = {0, 0};
	int correctcount[2] = {0, 0}; 
	int variableNum;
	int vectorNum;
	vector<double> ydata;
	vector<vector<double> > testdata; 

	readDataFile(filename, variableNum, vectorNum, testdata, ydata, false, true);

	for (int i = 0; i < ydata.size(); i++) {
		int y = ydata[i];
		classcount[y]++;
	}

	vector<double> zVals = computeZVals(testdata, trainData, variableNum, vectorNum);

	for (int i = 0; i < vectorNum; i++) {
		double p = (1/(1 + pow(M_E, -zVals[i])));
		int yGuess = 0;
		if (p > 0.5) {
			yGuess = 1;
		}
		int y = ydata[i];
		if (yGuess == y) {
			correctcount[y]++;
		}
	}

	cout << "Class 0: tested " << classcount[0] << ", correctly classified " << correctcount[0] << endl;
	cout << "Class 1: tested " << classcount[1] << ", correctly classified " << correctcount[1] << endl;
	cout << "Overall: tested " << classcount[0] + classcount[1] << ", correctly classified " << correctcount[0] + correctcount[1] << endl;
	cout << "Accuracy = " <<  (double)(correctcount[0] + correctcount[1])/(double)(classcount[0] + classcount[1]) << endl;
}

void appendRowsToFile(string fileName){
	//Get number of variables from file
	string line;
	ifstream trainfile (fileName);
	getline(trainfile,line);
	int varNum = stoi(line);
	trainfile.close();

	//append four rows to file
	ofstream file;
	file.open(fileName, ios::app);
	file << "\n";
	for (int i = 0; i < varNum; i++) {
		//last digit does not have a space afterwards
		if (i == varNum -1) {
			file << "0";
		} else {
			file << "0 ";
		}
	}
	file << ": 0 \n";

	for (int i = 0; i < varNum; i++) {
		//last digit does not have a space afterwards
		if (i == varNum -1) {
			file << "0";
		} else {
			file << "0 ";
		}
	}
	file << ": 1 \n";

	for (int i = 0; i < varNum; i++) {
		//last digit does not have a space afterwards
		if (i == varNum -1) {
			file << "1";
		} else {
			file << "1 ";
		}
	}
	file << ": 0 \n";

	for (int i = 0; i < varNum; i++) {
		//last digit does not have a space afterwards
		if (i == varNum -1) {
			file << "1";
		} else {
			file << "1 ";
		}
	}
	file << ": 1";

	file.close();
}

//creates a copy of the training file
void copyFile(string fileName, string copyFileName) {
	std::ifstream in(fileName); 
    std::ofstream out(copyFileName); 
    out << in.rdbuf(); 
    in.close();
    out.close();
}

int main() {

	cout << endl << "Running Naive Bayes algorithm using Maximum Liklihood Estimators..." << endl << endl;
	cout << "...on simple data:" << endl << endl;

	vector<vector<vector<int> > > simpleVector = trainNaiveBayes("simple-train.txt", false);
	testNaiveBayes("simple-test.txt", simpleVector);

	cout << endl << "...on vote data:" << endl << endl;
	
	vector<vector<vector<int> > > voteVector = trainNaiveBayes("vote-train.txt", false);
	testNaiveBayes("vote-test.txt", voteVector);

	cout << endl << "...on heart data:" << endl << endl;

	vector<vector<vector<int> > > heartVector = trainNaiveBayes("heart-train.txt", false);
	testNaiveBayes("heart-test.txt", heartVector);

	cout << endl << "Running Naive Bayes algorithm using Laplace Estimators..." << endl << endl;

    copyFile("simple-train.txt", "simple-train-copy.txt");
    copyFile("vote-train.txt", "vote-train-copy.txt");
    copyFile("heart-train.txt", "heart-train-copy.txt");

	appendRowsToFile("simple-train-copy.txt");
	appendRowsToFile("vote-train-copy.txt");
	appendRowsToFile("heart-train-copy.txt");

	cout << "...on simple data:" << endl << endl;

	vector<vector<vector<int> > > laplaceSimpleVector = trainNaiveBayes("simple-train-copy.txt", true);
	testNaiveBayes("simple-test.txt", laplaceSimpleVector);

	cout << endl << "...on vote data:" << endl << endl;

	vector<vector<vector<int> > > laplaceVoteVector = trainNaiveBayes("vote-train-copy.txt", true);
	testNaiveBayes("vote-test.txt", laplaceVoteVector);

	cout << endl << "...on heart data:" << endl << endl;

	vector<vector<vector<int> > > laplaceHeartVector = trainNaiveBayes("heart-train-copy.txt", true);
	testNaiveBayes("heart-test.txt", laplaceHeartVector);

	cout << endl << "Running Logistic Regression algorithm..." << endl << endl;
	cout << "...on simple data:" << endl << endl;

	vector<double> logRegSimpleVector = trainLogReg("simple-train.txt");
	testLogReg("simple-test.txt", logRegSimpleVector);

	cout << endl << "...on vote data:" << endl << endl;

	vector<double> logRegVoteVector = trainLogReg("vote-train.txt");
	testLogReg("vote-test.txt", logRegVoteVector);

	cout << endl << "...on heart data:" << endl << endl;

	vector<double> logRegHeartVector = trainLogReg("heart-train.txt");
	testLogReg("heart-test.txt", logRegHeartVector);

	return 0;
}





