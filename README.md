# Heat Equation Solver

This project was completed as part of a TU Delft course Object Oriented Scientific Programming with C++. The code approximates the solution to the heat equation $\dot{u} = \Delta u$ in 1D and 2D. The code made use of only Standard Library headers, and the Vector and Matrix classes necessary to find the solution to the partial differential equation were written from scratch as an exercise in C++.

Report.pdf describes in detail the choices taken for the different parts of the code, together with a descsription of the unit and verification tests performed to test individual code snippets as well the overall solution of the heat equation. 

For the initial condition:

$$
 u\left(\textbf{x},0\right) = \prod\limits_{k=0}^{n-1}\sin\left(\pi x_k\right)\qquad\forall \textbf{x}\in\Omega
$$

The code produced the following results, which matched the known solution to this fundamental initial condition.

In 1D, for a mesh size $m = 100$:

<img src="https://user-images.githubusercontent.com/22910604/186889985-d4354ba2-08cf-4b25-bf3f-ba46de19feb2.png" width=500/>

In 2D, for a mesh size $m = 100$ (100x100):

<img src="https://user-images.githubusercontent.com/22910604/186890804-e432fd8d-69ab-4786-9cfd-adabec4a0b4f.png" width=500/>
