#include "SecondOrderODESolver.h"

SecondOrderODESolver::SecondOrderODESolver(std::vector<std::pair<std::string, std::string>> functions_string_pairs, Eigen::Matrix<double, 2, 3> boundary_coefficients, std::pair<double, double> border, size_t N)
{
	for (auto function_pair : functions_string_pairs) {
		FunctionParser parser;
		if (parser.Parse(function_pair.first, function_pair.second) >= 0)
			throw std::exception(parser.ErrorMsg());
		this->functions_coefficients.push_back(parser);
	}
	if (border.first > border.second)
		throw std::exception("Invalid \'border\' values");
	if (N < 3)
		throw std::exception("Invalid \'N\' value");
	if (std::abs(boundary_coefficients(0, 0) * boundary_coefficients(0, 1)) > 0.001
		&& std::abs(boundary_coefficients(1, 0) * boundary_coefficients(1, 1)) > 0.001)
		throw std::exception("Invalid \'boundary_coefficients\' values");
	this->A = Eigen::MatrixXd::Zero(N, N);
	this->Y = Eigen::VectorXd::Zero(N);
	this->PHI = Eigen::VectorXd::Zero(N);
	this->a = border.first;
	this->b = border.second;
	this->h = (this->b - this->a) / (N - 1);
	this->boundary_coefficients = std::move(boundary_coefficients);
	this->FillMatrixForSolving();
	this->FillVectorPHI();
	this->is_solved = false;
}

Eigen::VectorXd SecondOrderODESolver::GetSolution()
{
	if(!this->is_solved)
		this->Solve();
	return this->Y;
}

double SecondOrderODESolver::GetConditionNumber()
{
	Eigen::JacobiSVD<Eigen::MatrixXd> svd(this->A);
	return svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size() - 1);
}

Eigen::MatrixXd SecondOrderODESolver::GetMatrixA()
{
	return this->A;
}

void SecondOrderODESolver::FillMatrixForSolving()
{
	//Filling 1st row of the matrix
	this->A(0, 0) = this->boundary_coefficients(0, 1) - 3 * this->boundary_coefficients(0, 0) / (2 * this->h);
	this->A(0, 1) = 2 * this->boundary_coefficients(0, 0) / this->h;
	this->A(0, 2) = -this->boundary_coefficients(0, 0) / (2 * this->h);

	//Filling the matrix from 1st to penultimate rows
	for (size_t i = 1; i < this->A.rows()-1; i++)
	{
		double x = i * this->h + this->a;
		this->A(i, i - 1) = 1 / std::pow(this->h, 2) - this->functions_coefficients[0].Eval(&x) / (2 * this->h);
		if (this->functions_coefficients[0].EvalError())
			throw std::exception("Error \'Eval\' call");
		this->A(i, i) = this->functions_coefficients[1].Eval(&x) - 2 / std::pow(this->h, 2);
		if (this->functions_coefficients[1].EvalError())
			throw std::exception("Error \'Eval\' call");
		this->A(i, i + 1) = 1 / std::pow(this->h, 2) + this->functions_coefficients[0].Eval(&x) / (2 * this->h);
		if (this->functions_coefficients[0].EvalError())
			throw std::exception("Error \'Eval\' call");
	}

	//Filling last row of the matrix
	this->A(this->A.rows() - 1, this->A.rows() - 3) = this->boundary_coefficients(1, 0) / (2 * this->h);
	this->A(this->A.rows() - 1, this->A.rows() - 2) = -2 * this->boundary_coefficients(1, 0) / this->h;
	this->A(this->A.rows() - 1, this->A.rows() - 1) = this->boundary_coefficients(1, 1) + 3 * this->boundary_coefficients(1, 0) / (2 * this->h);
}

void SecondOrderODESolver::FillVectorPHI()
{
	//Set 1st element of the vector
	this->PHI(0) = this->boundary_coefficients(0, 2);

	//Set the elements of the vector from 1st to penultimate
	for (size_t i = 1; i < this->PHI.size() - 1; i++)
	{
		double x = i * this->h + this->a;
		this->PHI(i) = this->functions_coefficients[2].Eval(&x);
		if (this->functions_coefficients[2].EvalError())
			throw std::exception("Error \'Eval\' call");
	}

	//Set last element of the vector
	this->PHI(this->PHI.size() - 1) = this->boundary_coefficients(1, 2);
}

void SecondOrderODESolver::Solve()
{
	//Solve with colPivHouseholderQr decomposition
	this->Y = this->A.colPivHouseholderQr().solve(this->PHI);
	this->is_solved = true;
}


#ifndef _DEBUG
// Needed for export to Python
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(dyns, module_handle) {
	module_handle.doc() = "Numerical Solvers library. Created in MEPhI in 2021";
	py::class_<SecondOrderODESolver>(
		module_handle, "SecondOrderODESolver"
		).def(py::init<std::vector<std::pair<std::string, std::string>>, Eigen::Matrix<double, 2, 3>, std::pair<double, double>, size_t>())
		.def("GetConditionNumber", &SecondOrderODESolver::GetConditionNumber, "Return condition number of matrix for solving ODE")
		.def("GetSolution", &SecondOrderODESolver::GetSolution, "Return Y - approximation of the solution on a grid");
}
#endif // !_DEBUG