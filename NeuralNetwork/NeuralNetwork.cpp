#include <stdafx.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

class Feedforward_Neural_Networks {
private:
	int num_input;
	int num_hidden;
	int num_output;

	double **hidden_weight;
	double **output_weight;

	void Compute_Output(double input[], double hidden[], double output[]) {
		for (int i = 0; i < num_hidden; i++) {
			double sum = 0;

			for (int j = 0; j < num_input; j++) {
				sum += input[j] * hidden_weight[i][j];
			}
			sum += hidden_weight[i][num_input];
			hidden[i] = 1 / (1 + exp(-sum));
		}
		for (int i = 0; i < num_output; i++) {
			double sum = 0;

			for (int j = 0; j < num_hidden; j++) {
				sum += hidden[j] * output_weight[i][j];
			}
			sum += output_weight[i][num_hidden];
			output[i] = 1 / (1 + exp(-sum));
		}
	}

public:
	Feedforward_Neural_Networks(int num_input, int num_hidden, int num_output) {
		this->num_input = num_input;
		this->num_hidden = num_hidden;
		this->num_output = num_output;

		output_weight = new double*[num_output];
		for (int i = 0; i < num_output; i++) {
			output_weight[i] = new double[num_hidden + 1];
		}
		hidden_weight = new double*[num_hidden];
		for (int i = 0; i < num_hidden; i++) {
			hidden_weight[i] = new double[num_input + 1];
		}
	}
	~Feedforward_Neural_Networks() {
		for (int i = 0; i < num_output; i++) {
			delete[] output_weight[i];
		}
		for (int i = 0; i < num_hidden; i++) {
			delete[] hidden_weight[i];
		}
		delete[] hidden_weight;
		delete[] output_weight;
	}

	void Test(double input[], double output[]) {
		double *hidden = new double[num_hidden];

		Compute_Output(input, hidden, output);

		delete[] hidden;
	}
	void Train(int num_train, double learning_rate, double **input, double **target_output) {
		int num_epoch = 0;
		int max_epoch = 10000;

		double *hidden = new double[num_hidden];
		double *hidden_derivative = new double[num_hidden];
		double *output = new double[num_output];
		double *output_derivative = new double[num_output];

		srand(0);
		for (int i = 0; i < num_hidden; i++) {
			for (int j = 0; j < num_input + 1; j++) {
				hidden_weight[i][j] = 0.2 * rand() / RAND_MAX - 0.1;
			}
		}
		for (int i = 0; i < num_output; i++) {
			for (int j = 0; j < num_hidden + 1; j++) {
				output_weight[i][j] = 0.2 * rand() / RAND_MAX - 0.1;
			}
		}

		do {
			double error = 0;

			for (int i = 0; i < num_train; i++) {
				Compute_Output(input[i], hidden, output);

				// 출력미분값 계산
				for (int j = 0; j < num_output; j++) {
					output_derivative[j] = learning_rate * (output[j] - target_output[i][j]) * (1 - output[j]) * output[j];
				}

				// 출력가중치 조정
				for (int j = 0; j < num_output; j++) {
					for (int k = 0; k < num_hidden; k++) {
						output_weight[j][k] -= output_derivative[j] * hidden[k];
					}
					output_weight[j][num_hidden] -= output_derivative[j];
				}

				// 은닉미분값 계산
				for (int j = 0; j < num_hidden; j++) {
					double sum = 0;

					for (int k = 0; k < num_output; k++) {
						sum += output_derivative[k] * output_weight[k][j];
					}
					hidden_derivative[j] = sum * (1 - hidden[j]) * hidden[j];
				}

				// 은닉가중치 조정
				for (int j = 0; j < num_hidden; j++) {
					for (int k = 0; k < num_input; k++) {
						hidden_weight[j][k] -= hidden_derivative[j] * input[i][k];
					}
					hidden_weight[j][num_input] -= hidden_derivative[j];
				}

				// 오차 계산
				for (int j = 0; j < num_output; j++) {
					error += 0.5 * (output[j] - target_output[i][j]) * (output[j] - target_output[i][j]);
				}
			}
			if (num_epoch % 500 == 0) {
				printf("반복횟수: %d, 오차: %lf\n", num_epoch, error);
			}
		} while (num_epoch++ < max_epoch);

		delete[] hidden;
		delete[] output;
	}
};

void main() {
	int num_input = 2;
	int num_hidden = 2;
	int num_output = 2;
	int num_train = 4;

	double learning_rate = 0.1;

	double **input = new double*[num_train];
	double **target_output = new double*[num_train];

	Feedforward_Neural_Networks *FNN = new Feedforward_Neural_Networks(num_input, num_hidden, num_output);

	for (int i = 0; i < num_train; i++) {
		input[i] = new double[num_input];
		target_output[i] = new double[num_output];
	}

	input[0][0] = 0;	input[0][1] = 0;
	input[1][0] = 0;	input[1][1] = 1;
	input[2][0] = 1;	input[2][1] = 0;
	input[3][0] = 1;	input[3][1] = 1;

	target_output[0][0] = 0;	target_output[0][1] = 0;
	target_output[1][0] = 0;	target_output[1][1] = 1;
	target_output[2][0] = 0;	target_output[2][1] = 1;
	target_output[3][0] = 1;	target_output[3][1] = 0;

	FNN->Train(num_train, learning_rate, input, target_output);

	for (int i = 0; i < num_train; i++) {
		double *output = new double[num_output];

		printf("입력: ", i);
		for (int j = 0; j < num_input; j++) {
			printf("%lf ", input[i][j]);
		}
		FNN->Test(input[i], output);

		printf("출력:");
		for (int j = 0; j < num_output; j++) {
			printf(" %lf", output[j]);
		}
		printf("\n");

		delete[] output;
	}
	for (int i = 0; i < num_train; i++) {
		delete[] input[i];
		delete[] target_output[i];
	}
	delete[] input;
	delete[] target_output;
	delete FNN;
}