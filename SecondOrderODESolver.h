#pragma once

#include <fparser.hh>
#include <vector>
#include <utility>
#include <Eigen/Dense>

class SecondOrderODESolver
{
public:
	//Constructor
	//Example:
	//ODE is y'' + p(x)·y' + q(x)·y = φ(x)
	//The boundary conditions are:
	//α1·y'(a) + β1·y(a) = γ1
	//α2·y'(b) + β2·y(b) = γ2 ,
	//where x∈[a,b]
	//functions_string_pairs is {{"p(x)", "x"}, {"q(x)", "x"}, {"φ(x)", "x"}}
	//boundary_coefficients is:
	//α1 β1 γ1
	//α2 β2 γ2
	//border is {a,b}
	//N is number of points of the partition for approximation
	SecondOrderODESolver(std::vector<std::pair<std::string, std::string>> functions_string_pairs, Eigen::Matrix<double, 2, 3> boundary_coefficients, std::pair<double, double> border, size_t N);

	//Return Y - approximation of the solution on a grid
	Eigen::VectorXd GetSolution();

	//Return condition number of matrix A (the description is in the comments to the private variable A)
	double GetConditionNumber();

	//Return matrix for solving ODE - A (the description is in the comments to the private variable A)
	Eigen::MatrixXd GetMatrixA();

private:
	//Filling the matrix A (the description is in the comments to the private variable A)
	void FillMatrixForSolving();

	//Filling the vector PHI (the description is in the comments to the private variable PHI)
	void FillVectorPHI();

	//Find Y in equation A·Y = Φ
	void Solve();

private:
	//Functions that stand as coefficients in ODE
	//Example:
	//ODE is y'' + p(x)·y' + q(x)·y = φ(x)
	//functions_coefficients[0] is p(x)
	//functions_coefficients[1] is q(x)
	//functions_coefficients[2] is φ(x)
	std::vector<FunctionParser> functions_coefficients;

	//Matrix that stores the coefficients of the boundary conditions
	//Example:
	//The boundary conditions are:
	//α1·y'(a) + β1·y(a) = γ1
	//α2·y'(b) + β2·y(b) = γ2 ,
	//where x∈[a,b]
	//boundary_coefficients is:
	//α1 β1 γ1
	//α2 β2 γ2
	Eigen::Matrix<double, 2, 3> boundary_coefficients;

	//Left border by x
	//x∈[a,b]
	double a;

	//Right border by x
	//x∈[a,b]
	double b;

	//Grid step
	double h;

	//Matrix for solving ODE
	//A·Y = Φ ,
	//where Y is solution approximation vector, Φ is approximation vector of φ(x)
	Eigen::MatrixXd A;

	//Solution approximation vector
	//A·Y = Φ ,
	//where A is matrix for solving ODE, Φ is approximation vector of φ(x)
	Eigen::VectorXd Y;

	//Approximation vector of φ(x)
	//A·Y = Φ ,
	//where A is matrix for solving ODE, Y is solution approximation vector
	Eigen::VectorXd PHI;

	//A value indicating whether the ODE has been resolved
	bool is_solved;
};

