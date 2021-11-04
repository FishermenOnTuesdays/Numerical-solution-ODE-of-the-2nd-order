#include "SecondOrderODESolver.h"
#include <iostream>

int main()
{
	//ODE is y'' + cos(x)·y' + sin(x)·y = 1 - cos(x) - sin(x)
	//The boundary conditions are:
	//-y'(0) + y(0) = 0
	//0·y'(1) + y(1) = 1.3818 ,
	//where x∈[0,1]
	FunctionParser p;
	FunctionParser q;
	FunctionParser phi;
	p.Parse("cos(x)", "x");
	q.Parse("sin(x)", "x");
	phi.Parse("1 - cos(x) - sin(x)", "x");
	std::vector<FunctionParser> functions = { p, q, phi };
	Eigen::Matrix<double, 2, 3> boundaries;
	boundaries << -1, 1, 0,
				   0, 1, 1.3818;
	SecondOrderODESolver solver(functions, boundaries, {0, 1}, 100);
	std::cout << "Condition number: " << solver.GetConditionNumber() << std::endl;
	std::cout << "Solution: " << solver.GetSolution() << std::endl;
}