#include <iostream>
#include <Eigen/Dense>
#include <math.h>
#include <iomanip>

#define InputShape 10
#define Layer1_OutShape 5
#define Layer2_OutShape 1

#define DataNum 100
#define Epcho   2000

using namespace Eigen;
using namespace std;

//神经网络的整体计算为:
//Z1 = W1^t*X + B1，这样的话X每一列是一条数据，行数：10->5->1
//A1 = f1(Z1)
//Z2 = W2^t*A1 + B2
//A2 = f2(Z2)

MatrixXf Leak_ReLu(MatrixXf, float);//激活函数
float MSE(MatrixXf, MatrixXf);//loss损失函数

class NN
{
	public:
		NN(MatrixXf input, MatrixXf y_true, float alph);
		MatrixXf ForWard();
		float BackWard();
	private:
		//神经网络的输入值、输出的真实值、学习率等较为固定的值
		MatrixXf input;
		MatrixXf y_true;
		float alph;
		//神经网络待学习参数
		MatrixXf W1_T;
		MatrixXf B1;
		MatrixXf W2_T;
		MatrixXf B2;
		//神经网络中间参数
		MatrixXf Z1;       
		MatrixXf A1;
		MatrixXf Z2;
		MatrixXf A2;//==out
		//反向传播所用参数
		MatrixXf dW2;
		MatrixXf dB2;
		MatrixXf dW1;
		MatrixXf dB1;
		//对Layer2所需参数
		MatrixXf dJ_dA2;
		MatrixXf dA2_dZ2;
		MatrixXf dZ2_dW2;
		MatrixXf dZ2_dB2;
		//对Layer1所需参数
		MatrixXf dZ2_dA1;
		MatrixXf dA1_Z1;
		MatrixXf dZ1_W1;
};
NN::NN(MatrixXf input, MatrixXf y_true, float alph)//初始化权值
{
	this->input = input;
	this->y_true = y_true;
	this->alph = alph;
	this->W1_T = MatrixXf::Random(Layer1_OutShape, InputShape);
	this->W2_T = MatrixXf::Random(Layer2_OutShape, Layer1_OutShape);
	this->B1 = MatrixXf::Zero(Layer1_OutShape,this->input.cols());
	this->B2 = MatrixXf::Zero(Layer2_OutShape, this->input.cols());
}

MatrixXf NN::ForWard()
{
	this->Z1 = this->W1_T * this->input;
	this->A1 = Leak_ReLu(this->Z1, this->alph);
	this->Z2 = this->W2_T * this->A1;
	this->A2 = Leak_ReLu(this->Z2, this->alph);
	//return MSE(this->y_true, this->A2);
	return this->A2;
}

float NN::BackWard()
{
	int rows_temp, cols_temp;//临时行列变量
	//Abount Layer1 work start!!
	this->dJ_dA2 = 2 * (this->A2 - this->y_true);
	this->dA2_dZ2 = MatrixXf::Ones(this->Z2.rows(),this->Z2.cols());
	for (rows_temp = 0; rows_temp < this->Z2.rows(); ++rows_temp)
	{
		for (cols_temp = 0; cols_temp < this->Z2.cols(); ++cols_temp)
		{
			this->dA2_dZ2(rows_temp, cols_temp) = this->Z2(rows_temp, cols_temp) >= 0 ? 1.0 : this->alph;
		}
	}
	this->dZ2_dW2 = this->A1.transpose();
	this->dW2 = this->dJ_dA2.cwiseProduct(this->dA2_dZ2)*this->dZ2_dW2/DataNum;
	this->dB2 = this->dJ_dA2.cwiseProduct(this->dA2_dZ2) / DataNum;
	//Abount Layer1 work end!!

	//Abount Layer2 work start!!
	this->dZ2_dA1 = this->W2_T.transpose();
	this->dA1_Z1 = MatrixXf::Ones(this->Z1.rows(), this->Z1.cols());
	for (rows_temp = 0; rows_temp < this->Z1.rows(); ++rows_temp)
	{
		for (cols_temp = 0; cols_temp < this->Z1.cols(); ++cols_temp)
		{
			this->dA1_Z1(rows_temp, cols_temp) = this->Z1(rows_temp, cols_temp) >= 0 ? 1.0 : this->alph;
		}
	}

	this->dZ1_W1 = this->input.transpose();
	this->dW1 = this->dA1_Z1.cwiseProduct(this->dZ2_dA1 * this->dJ_dA2.cwiseProduct(this->dA2_dZ2))*this->dZ1_W1 / DataNum;
	this->dB1 = this->dA1_Z1.cwiseProduct(this->dZ2_dA1 * this->dJ_dA2.cwiseProduct(this->dA2_dZ2)) / DataNum;
	//Abount Layer2 work end!!
	//调整学习参数
	this->W2_T = this->W2_T - this->alph*this->dW2;
	this->W1_T = this->W1_T - this->alph*this->dW1;
	this->B2 = this->B2 - this->alph*this->dB2;
	this->B1 = this->B1 - this->alph*this->dB1;
	return MSE(this->y_true,this->ForWard());
}

MatrixXf Leak_ReLu(MatrixXf Z, float a)
{
	int Z_rows = Z.rows();
	int Z_cols = Z.cols();
	int i, j;
	//cout << Z_rows <<" " <<Z_cols << endl;
	MatrixXf A(Z_rows, Z_cols);
	//cout << Z_rows << " " << Z_cols << endl;
	for (i = 0; i < Z_rows; ++i)
	{
		for (j = 0; j < Z_cols; ++j)
		{
			//cout << "i=" << i << "," << "j=" << j << endl;
			A(i, j) = Z(i, j) >= 0 ? Z(i, j) : a*Z(i, j);
		}
	}

	return A;
}

float MSE(MatrixXf y_true, MatrixXf y_pred)//损失函数
{
	MatrixXf true_pred = y_true - y_pred;
	return true_pred.array().square().sum()/ true_pred.cols();
}

void main()
{
	MatrixXf input_data,y_true,y_pred;
	int rows_temp,cols_temp;//行列数临时变量
	int epch;//训练轮数临时变量
	int count_right = 0, count_error = 0;
	input_data = MatrixXf::Random(InputShape, DataNum);
	y_true = MatrixXf::Zero(Layer2_OutShape, DataNum);
	for (cols_temp = 0; cols_temp < DataNum; ++cols_temp)
	{
		y_true(0, cols_temp) = input_data.col(cols_temp).sum() > 0 ? 1.0:0.0;
	}

	NN MyNN = NN(input_data, y_true,0.1);
	for (epch = 0; epch < Epcho; ++epch)
	{
		MyNN.ForWard();
		cout << MyNN.BackWard() << endl;
	}
	y_pred = MyNN.ForWard();
	cout << setw(15) << "Predicted Value" << setw(20) << "True Value" << endl;
	for (cols_temp = 0; cols_temp < DataNum; ++cols_temp)
	{
		cout << setw(15) << y_pred(0, cols_temp) << setw(5) << "for" << setw(10) << y_true(0, cols_temp) << setw(10) << " is " ;
		if ((y_true(0, cols_temp) > 0.5&&y_pred(0, cols_temp) > 0.5) || (y_true(0, cols_temp) < 0.5&&y_pred(0, cols_temp) < 0.5))
		{
			cout << "True" << endl;
			count_right++;
		}
		else
		{
			cout << "False" << endl;
			count_error++;
		}
	}
	cout <<"预测准确率为："<< count_right/(float)DataNum << endl;

	system("pause");
}