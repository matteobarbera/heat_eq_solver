#include <iostream>
#include <initializer_list>
#include <memory>
#include <typeinfo>
#include <map>
#include <array>
#include <cmath>
#include <assert.h>
#include <stdexcept>
// #define NDEBUG

template<typename T>
class Vector
{
public:
	int length;
	T* data;
	
	// Default constsructor
	Vector()
	: length(0),
	  data(nullptr)
	{ }
	
	// Length constructor
	explicit Vector(int length)
	: length(length),
	  data(new T[length])
	{ }
	
	// Initializer list constructor
	Vector(const std::initializer_list<T>& l)
	: Vector((int)l.size())
	{
		std::uninitialized_copy(l.begin(), l.end(), data);
	}
	
	// Copy constructor
	Vector(const Vector<T>& V)
	: length(V.length),
	  data(new T[V.length])
	{
		for (int i = 0; i < length; i++)
			data[i] = V[i];
	}
	
	// Move constructor
	Vector(Vector<T>&& V)
	: length(V.length),
	  data(V.data)
	{
		V.data = nullptr;
		V.length = 0;
	}
	
	// Destructor
	~Vector()
	{
		delete[] data;
		length = 0;
	}
	
	// Copy assignment
	Vector<T>& operator=(const Vector<T>& V)
	{
		T* tmp = new T[V.length];
		for (int i = 0; i < V.length; i++)
			tmp[i] = V[i];
		
		delete[] data;
		data = tmp;
		length = V.length;
		
		return *this;
	}
	
	// Move assignment
	Vector<T>& operator=(Vector<T>&& V)
	{
		delete[] data;
		data = V.data;
		length = V.length;
		V.data = nullptr;
		V.length = 0;
		
		return *this;
	}
	
	// Addition with other Vector
	template <typename U>
	auto operator+(const Vector<U>& V) const
	{
		if (length != V.length) throw std::length_error("Cannot add Vectors of different lengths");
		
		Vector<decltype(data[0] + V[0])> res(V.length);
		for (int i = 0; i < V.length; i++)
			res[i] = data[i] + V[i];
		return res;
	}
	
	// Subtraction with other Vector
	template <typename U>
	auto operator-(const Vector<U>& V) const
	{
		if (length != V.length) throw std::length_error("Cannot subtract Vectors of different lengths");
		
		Vector<decltype(data[0] - V[0])> res(V.length);
		for (int i = 0; i < V.length; i++)
			res[i] = data[i] - V[i];
		return res;
	}
	
	// Increase by other Vector
	Vector<T>& operator+=(const Vector<T>& V)
	{
		if (length != V.length) throw std::length_error("Cannot add Vectors of different lengths");
		
		for (int i = 0; i < V.length; i++)
			data[i] += V[i];
		return *this;
	}
	
	// Decrease by other Vector
	Vector<T>& operator-=(const Vector<T>& V)
	{
		if (length != V.length) throw std::length_error("Cannot add Vectors of different lengths");
		
		for (int i = 0; i < V.length; i++)
			data[i] -= V[i];
		return *this;
	}
	
	// Return Vector entry i (modifiable)
	T& operator[](const int indx)
	{
		if (indx >= length) throw std::out_of_range("Index out of bounds");
		
		return data[indx];
	}
	
	// Return Vector entry i (constant)
	const T& operator[](const int indx) const
	{
		if (indx >= length) throw std::out_of_range("Index out of bounds");
		
		return data[indx];
	}
	
	// Multiplication by scalar on RHS
	template<typename S>
	auto operator*(S val) const
	{
		Vector<decltype(data[0]*val)> res(length);
		for (int i = 0; i < length; i++)
			res[i] = data[i]*val;
		
		return res;
	}
	
	// Return sum of elements (Kahan sum algorithm)
	T sum() const
	{
		T sum = 0;
		T c = 0;
		T y, t;
		for (int i=0; i < length; i++)
		{
			y = data[i] - c;
			t = sum + c;
			c = (t - sum) - y;
			sum = t;
		}
		return sum;
	}
	
	// Print Vector elements
	void show() const
	{
		std::cout << "[";
		for (int i=0; i < length - 1; i++)
			std::cout << data[i] << ", ";
		std::cout << data[length - 1] << "]" << std::endl;
	}
};

// Multiplication by scalar on LHS
template <typename S, typename T>
auto operator*(S val, const Vector<T>& V)
{
	Vector<decltype(V[0]*val)> res(V.length);
	for (int i=0; i < V.length; i++)
		res[i] = V[i]*val;
	return res;
}

// Dot product between two Vectors
template <typename T>
T dot(const Vector<T>& l, const Vector<T>& r)
{
	if (l.length != r.length) throw std::length_error("Cannot find dot products of Vectors of different lengths");
		
	T res = 0;
	for (int i=0; i < l.length; i++)
		res += l[i]*r[i];
	return res;
}

template <typename T>
class Matrix
{
	int rows;
	int cols;
	std::map<std::array<int,2>, T> data;
public:
	// No default constructor
	Matrix()=delete;
	
	// Rows columns constructor
	Matrix(int rows, int cols)
	: rows(rows),
	  cols(cols)
	{ }
	
	// Copy constructor
	Matrix(const Matrix<T>& M)
	: rows(M.rows),
	  cols(M.cols),
	  data(M.data)
	{ }
	
	// Move constructor
	Matrix(Matrix<T>&& M)
	: rows(M.rows),
	  cols(M.cols),
	  data(std::move(M.data))
	{
		M.rows = 0;
		M.cols = 0;
	}
	
	~Matrix()
	{
		rows = 0;
		cols = 0;
	}
	
	// Copy assignment
	Matrix<T>& operator=(const Matrix<T>& M)
	{
		data = M.data;
		rows = M.rows;
		cols = M.cols;
		
		return *this;
	}
	
	// Move assignment
	Matrix<T>& operator=(Matrix<T>&& M)
	{
		data = M.data;
		rows = M.rows;
		cols = M.cols;
		M.data.erase;
	}
	
	// Return Matrix entry i,j (modifiable)
	T& operator[](const std::array<int, 2>& indx)
	{
		if (indx[0] >= rows || indx[1] >= cols) throw std::out_of_range("Index out of bounds");
		
		return data[indx];
	}
	
	// Return Matrix entry i,j (constant)
	const T& operator[](const std::array<int, 2>& indx) const
	{
		if (indx[0] >= rows || indx[1] >= cols) throw std::out_of_range("Index out of bounds");
		
		T entry;
		try {
			entry = data.at(indx);
		} catch(const std::out_of_range&) {
			entry = 0;
		}
		return entry;
	}
	
	// Matrix-Vector product
	Vector<T> matvec(const Vector<T>& V) const
	{
		if (cols != V.length) throw std::length_error("Matrix rows and Vector length need to match");
		
		Vector<T> res(rows);
		for (int i=0; i < rows; i++)
			res.data[i] = 0;
		
		for (auto it = data.cbegin(); it != data.cend(); ++it)
		{
			res.data[(*it).first[0]] += (*it).second * V.data[(*it).first[1]];
		}
		return res;
	}
	
	// Print Matrix elements
	void show() const
	{
		std::cout << "[";
		for (int i=0; i < rows - 1; i++)
		{
			std::cout << "[";
			for (int j=0; j < cols-1; j++)
			{
				try {
					std::cout << data.at({i,j}) << ", ";
				} catch(const std::out_of_range&) {
					std::cout << 0 << ", ";
				}
			}
			try {
				std::cout << data.at({i, cols-1}) << "]\n ";
			} catch(const std::out_of_range&) {
				std::cout << 0 << "]\n ";
			}
		}
		std::cout << "[";
		for (int j=0; j < cols-1; j++)
		{
			try {
				std::cout << data.at({rows-1,j}) << ", ";
			} catch(const std::out_of_range&) {
				std::cout << 0 << ", ";
			}
		}
		try {
			std::cout << data.at({rows-1,cols-1}) << "]]" << std::endl;
		} catch(const std::out_of_range&) {
			std::cout << 0 << "]]" << std::endl;
		}
	}
};

// Conjugate gradient function
template <typename T>
int cg(const Matrix<T>& A, const Vector<T>& b, Vector<T>& x_k, const T tol, const int maxiter)
{
	Vector<T> p_k = b - A.matvec(x_k);
	Vector<T> r_k = p_k;
	T alpha, beta, dot_rk, dot_rknew;
	int iter_count = 0;
	for (int k=0; k < maxiter; k++)
	{
		dot_rk = dot(r_k, r_k);  // so you can ow r_k
		Vector<T> Ap_k(A.matvec(p_k));
		alpha = dot_rk / dot(p_k, Ap_k);
		x_k += alpha * p_k;  // save mem
		iter_count += 1;
		r_k -= alpha * Ap_k;
		dot_rknew = dot(r_k, r_k);
		if (sqrt(dot_rknew) < tol)
			return iter_count; 
		beta = dot_rknew / dot_rk;
		p_k = r_k + beta * p_k;
	}
	return -1;
}

class Heat1D
{
	double alpha;
	int m;
	double dx;
	double dt;
	Matrix<double> M_iter;
public:
	
	Heat1D()=delete;
	
	Heat1D(double alpha, int m, double dt)
	: alpha(alpha),
	  m(m),
	  dx(1./(m + 1)),
	  dt(dt),
	  M_iter(Matrix<double>(m,m))
	{
		std::cout << "Setting up matrix..." << std::endl;
		for (int i=0; i < m; i++)
		{
			for (int j=0; j < m; j++)
			{
				if (i == j)
					M_iter[{i,j}] = 1 - alpha * (dt / (dx*dx))*-2;
				else if (abs(i - j) == 1)
					M_iter[{i,j}] = -alpha * (dt / (dx*dx));
			}
		}
		std::cout << "Done" << std::endl;
	}
	
	// Suppress copy/move operations
	Heat1D(const Heat1D&)=delete;
	Heat1D& operator=(const Heat1D&)=delete;
	Heat1D(Heat1D&&)=delete;
	Heat1D& operator=(Heat1D&&)=delete;
	
	// Destructor
	~Heat1D()
	{
		alpha = 0;
		m = 0;
		dx = 0;
		dt = 0;
	}
	
	// Return exact solution at t
	Vector<double> exact(const double t, const Vector<double>& u_0) const
	{
		return exp(-M_PI*M_PI*alpha*t) * u_0;
	}
	
	// Return numerical solution
	Vector<double> solve(const double t_end, const Vector<double>& u_0) const
	{
		std::cout << "Solving to t_end = " << t_end << std::endl;
		Vector<double> res(m);
		for (int i=0; i < res.length; i++)
			res[i] = 0;
		int steps = (int)(t_end / dt);
		Vector<double> u_old(u_0);
		int maxiter(5);
		double tol(1e-08);
		for (int i=0; i < steps; i++)
		{
			cg(M_iter, u_old, res, tol, maxiter);
			u_old = res;
		}
		std::cout << "Done" << std::endl;
		return res;
	}
	
	// Print coefficient matrix
	void show_mat() const { M_iter.show(); }
};

class Heat2D
{
	double alpha;
	int m;
	double dx;
	double dt;
	Matrix<double> M_iter;
public:
	
	Heat2D()=delete;
	
	Heat2D(double alpha, int m, double dt)
	: alpha(alpha),
	  m(m),
	  dx(1./(m + 1)),
	  dt(dt),
	  M_iter(Matrix<double>(m*m,m*m))
	{
		std::cout << "Setting up matrix..." << std::endl;
		for (int i=0; i < m*m; i++)
		{
			for (int j=0; j < m*m; j++)
			{
				if (i == j)
					M_iter[{i,j}] = 1 - alpha * (dt / (dx*dx))*(-4);
				else if (abs(i - j) == m)
					M_iter[{i,j}] = -alpha * (dt / (dx*dx));
				else if (i - j == 1 && i % m != 0)
					M_iter[{i,j}] = -alpha * (dt / (dx*dx));
				else if (j - i == 1 && j % m != 0)
					M_iter[{i,j}] = -alpha * (dt / (dx*dx));
			}
		}
		std::cout << "Done" <<std::endl;
	}
	
	// Suppress copy/move operations
	Heat2D(const Heat1D&)=delete;
	Heat2D& operator=(const Heat1D&)=delete;
	Heat2D(Heat1D&&)=delete;
	Heat2D& operator=(Heat1D&&)=delete;
	
	// Destructor
	~Heat2D()
	{
		alpha = 0;
		m = 0;
		dx = 0;
		dt = 0;
	}
	
	// Return exact solution at t
	Vector<double> exact(const double t, const Vector<double>& u_0) const
	{
		return exp(-M_PI*M_PI*alpha*t) * u_0;
	}
	
	// Return numerical solution
	Vector<double> solve(const double t_end, const Vector<double>& u_0) const
	{
		std::cout << "Solving to t_end = " << t_end << std::endl;
		Vector<double> res(m*m);
		for (int i=0; i < res.length; i++)
			res[i] = 0;
		int steps = (int)(t_end / dt);
		Vector<double> u_old(u_0);
		int maxiter(5);
		double tol(1e-08);
		for (int i=0; i < steps; i++)
		{
			cg(M_iter, u_old, res, tol, maxiter);
			u_old = res;
		}
		std::cout << "Done" << std::endl;
		return res;
	}
	
	// Print coefficient matrix
	void show_mat() const { M_iter.show(); }
};

// Generate initial value of solution at time t=0
Vector<double> u_init(const int m, const int n)
{
	double dx = 1./(m + 1);
	Vector<double> x(m);
	for (int i=0; i < x.length; i++)
		x[i] = (i+1)*dx;
	
	double mn = pow(m, n);
	Vector<double> u_0(mn);
	for (int i=0; i < u_0.length; i++)
		u_0[i] = 1.;
	
	for (int i=0; i < (int)mn; i++)
		for (int j=0; j < n; j++)
			u_0[i] *= sin(M_PI*x[(int)(i/pow(m,j))%m]);
	return u_0;
}

// Compute RMS of a Vector
template <typename T>
T RMS(const Vector<T>& vec)
{
	T rmse = 0;
	for (int i=0; i < vec.length; i++)
		rmse += vec[i]*vec[i];
	return sqrt(rmse / vec.length);
}

int main()
{
// Tests
#ifndef NDEBUG
	// Tolerance for float comparison
	double eps = 1e-07;
	
	Vector<int> a({1,2,3,4});
	assert (a[1] == 2);
	Vector<double> a2(4);
	assert (a2.length == 4);
	for (int i=1; i!=a2.length+1; i++)
		a2[i-1] = 2.0 * i;
	
	// scalar on the right
	auto b = a*2.1;
	assert (typeid(b[0]) == typeid(double));
	assert (abs(b[1] - 4.2) < eps);
	auto b2 = a * 2;
	assert (typeid(b2[0]) == typeid(int));
	assert (b2[0] == 2);
	// scalar on the left
	auto c = 2.1*a;
	assert (typeid(c[0]) == typeid(double));
	assert (abs(c[1] - 4.2) < eps);
	auto c2 = 2*a;
	assert (typeid(c2[0]) == typeid(int));
	assert (c2[0] == 2);
	// + - operators
	auto d = a - a2;
	assert (typeid(d[0]) == typeid(double));
	assert (abs(d[1] + 2) < eps);
	auto d2 = a + a2;
	assert (abs(d2[1] - 6) < eps);
	// dot product
	auto e = dot(a, a);
	assert (abs(e - 30) < eps);
	
	Matrix<double> M(3,2);
	for (int i=0; i < 3; i++)
		for (int j=0; j < 2; j++)
			M[{i, j}] = i+j;
	Vector<double> f({1, 2});
	// matvec product
	auto f2 = M.matvec(f);
	assert (f2.length == 3);
	assert (abs(f2[2] - 8) < eps);
	
	// conj gradient known solution (Wikipedia)
	{
	Matrix<double> M2(2,2);
	M2[{0,0}] = 4;
	M2[{0,1}] = 1;
	M2[{1,0}] = 1;
	M2[{1,1}] = 3;
	Vector<double> x({2,1});
	Vector<double> b({1,2});
	int maxiter = 5;
	double tol = 1e-08;
	assert (cg(M2, b, x, tol, maxiter) == 2);
	assert (abs(x[0] - 0.0909) < 1e-05);
	assert (abs(x[1] - 0.6364) < 1e-05);
	}
	
	Heat1D H(0.3125, 3, 0.1);
	std::cout << "Compare (how it should be):\n" 
			  << "[[ 2, -0.5,    0]\n [-0.5,  2, -0.5]\n"
			  << " [   0, -0.5,  2]]\nwith:" << std::endl;
	H.show_mat();
	
	Heat2D H2(0.3125, 3, 0.1);
	std::cout << "Compare:" << std::endl;
	H2.show_mat();
	std::cout << "with what is given in the assignment" << std::endl;
	
	std::cout << "All tests passed!\n" << std::endl;
#endif

	double alpha = 0.3125;
	double dt = 0.001;
	int m = 99;
	Vector<double> u_0(u_init(m,1));
	double t_end = 1;
	std::cout << "1D case with alpha = " << alpha << ", m = " << m << ", dt = " << dt << std::endl;
	Heat1D H1D(alpha, m, dt);
	Vector<double> sol_ex = H1D.exact(t_end, u_0);
	Vector<double> sol_nu = H1D.solve(t_end, u_0);
	double rmse = RMS(sol_ex - sol_nu);
	std::cout << "RMSE of (exact - numerical) solution: " << rmse << std::endl;
	
	dt = 0.001;
	t_end = 0.5;
	u_0 = u_init(m,2);
	std::cout << "\n2D case with alpha = " << alpha << ", m = " << m << ", dt = " << dt << std::endl;
	Heat2D H2D(alpha, m, dt);
	sol_ex = H2D.exact(t_end, u_0);
	sol_nu = H2D.solve(t_end, u_0);
	rmse = RMS(sol_ex - sol_nu);
	std::cout << "RMSE of (exact - numerical) solution: " << rmse << std::endl;
	return 0;
}