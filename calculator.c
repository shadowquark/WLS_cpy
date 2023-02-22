#include<stdio.h>
#include<stdlib.h>
#include<stdbool.h>
#include<math.h>
#include<time.h>
void print(double *x, int m, int n)
{
	for (int i = 0; i < m * n; i += n)
	{
		for (int j = 0; j < n; ++ j)
			printf("%lf, ", *(x + i + j));
		printf("\n");
	}
	printf("\n");
}
double *inverse(double *x, int m)
{
	// LU decomoposition of a general matrix
	extern void dgetrf_(
		int *M, int *N, double *A,
		int *LDA, int *IPIV, int *INFO
	);
	// generate inverse of a matrix given its LU decomposition
	extern void dgetri_(
		int *N, double *A, int *lda, int *IPIV,
		double *WORK, int *lwork, int *INFO
	);
	int N = m;
	int LDA = m;
	int *IPIV = malloc(sizeof(int) * m);
	double *WORK = malloc(sizeof(double) * m * m);
	int LWORK = m * m;
	int INFO;
	dgetrf_(&N, &N, x, &LDA, IPIV, &INFO);
	dgetri_(&N, x, &N, IPIV, WORK, &LWORK, &INFO);
	free(IPIV);
	free(WORK);
	if (INFO)
		printf("Invert Fail");
	return x;
}
double *add(double *x, double *y, int m, int n)
{
	double *output = malloc(sizeof(double) * m * n);
	for (int i = 0; i < m * n; i += n)
		for (int j = 0; j < n; ++ j)
			*(output + i + j) = *(x + i + j) + *(y + i + j);
	return output;	
}
double *times(double *x, double *y, int m, int p, int n)
{
	double *output = malloc(sizeof(double) * m * n);
	for (int i = 0; i < m; ++ i )
		for (int j = 0; j < n; ++ j)
		{
			double temp = 0;
			for (int k = 0; k < p; ++ k)
				temp += *(x + i * p + k) * *(y + k * n + j);
			*(output + i * n + j) = temp;
		}
	return output;
}
double *calc_xTw(double *x, double *w, int m, int n)
{
	double *output = malloc(sizeof(double) * m * n);
	for (int i = 0; i < m; ++ i)
		for (int j = 0; j < n; ++ j)
			*(output + i * n + j) = *(x + j * m + i) * *(w + j);
	return output;
}
//double *calc_xTw(double *x, double *w, int m, int n)
//{
//	double *output = malloc(sizeof(double) * m * n);
//	for (int i = 0; i < m; ++ i)
//		for (int j = 0; j < n; ++ j)
//		{
//			double temp = 0;
//			for (int k = 0; k < n; ++ k)
//				temp += *(x + k * m + i) * *(w + k * n + j);
//			*(output + i * n + j) = temp;
//		}
//	return output;
//}
double *wls_iter(
	double *x, double *xTwx, double *xTwy,
	double *x_next, double *w_next, double *y_next,
	int n1, int n2, int m, bool update
){
	void print(double *x, int m, int n)
	{
		for (int i = 0; i < m * n; i += n)
		{
			for (int j = 0; j < n; ++ j)
				printf("%lf, ", *(x + i + j));
			printf("\n");
		}
		printf("\n");
	}
	double *inverse(double *x, int m)
	{
		// LU decomoposition of a general matrix
		extern void dgetrf_(
			int *M, int *N, double *A,
			int *LDA, int *IPIV, int *INFO
		);
		// generate inverse of a matrix given its LU decomposition
		extern void dgetri_(
			int *N, double *A, int *lda, int *IPIV,
			double *WORK, int *lwork, int *INFO
		);
		int N = m;
		int LDA = m;
		int *IPIV = malloc(sizeof(int) * m);
		double *WORK = malloc(sizeof(double) * m * m);
		int LWORK = m * m;
		int INFO;
		dgetrf_(&N, &N, x, &LDA, IPIV, &INFO);
		dgetri_(&N, x, &N, IPIV, WORK, &LWORK, &INFO);
		free(IPIV);
		free(WORK);
		if (INFO)
			printf("Invert Fail");
		return x;
	}
	double *add(double *x, double *y, int m, int n)
	{
		double *output = malloc(sizeof(double) * m * n);
		for (int i = 0; i < m * n; i += n)
			for (int j = 0; j < n; ++ j)
				*(output + i + j) = *(x + i + j) + *(y + i + j);
		return output;	
	}
	double *times(double *x, double *y, int m, int p, int n)
	{
		double *output = malloc(sizeof(double) * m * n);
		for (int i = 0; i < m; ++ i )
			for (int j = 0; j < n; ++ j)
			{
				double temp = 0;
				for (int k = 0; k < p; ++ k)
					temp += *(x + i * p + k) * *(y + k * n + j);
				*(output + i * n + j) = temp;
			}
		return output;
	}
	double *calc_xTw(double *x, double *w, int m, int n)
	{
		double *output = malloc(sizeof(double) * m * n);
		for (int i = 0; i < m; ++ i)
			for (int j = 0; j < n; ++ j)
				*(output + i * n + j) = *(x + j * m + i) * *(w + j);
		return output;
	}
	double *output = malloc(
		sizeof(double) * (n1 + 2 * n2 + m * (m + 2))
	);
	double *xTw = calc_xTw(x_next, w_next, m, n2);
	double *xTwx_iter = times(xTw, x_next, m, n2, m);
	double *xTwx_next = add(xTwx, xTwx_iter, m, m);
	for (int i = 0; i < m * m; i += m)
		for (int j = 0; j < m; ++ j)
			*(output + i + j) = *(xTwx_next + i + j);
	int pos = m * m;
	double *xTwy_next = times(xTw, y_next, m, n2, 1);
	double *xTwy_iter = add(xTwy, xTwy_next, m, 1);
	for (int i = 0; i < m; ++ i)
		*(output + i + pos) = *(xTwy_iter + i);
	pos += m;
	double *xTwx_inverse = inverse(xTwx_next, m);
	double *predict = times(xTwx_inverse, xTwy_iter, m, m, 1);
	for (int i = 0; i < m; ++ i)
		*(output + i + pos) = *(predict + i);
	pos += m;
	double *yhat_next = times(x_next, predict, n2, m, 1);
	for (int i = 0; i < n2; ++ i)
		*(output + i + pos) = *(yhat_next + i);
	pos += n2;
	if (update)
	{
		double *yhat = times(x, predict, n1, m, 1);
		for (int i = 0; i < n1; ++ i)
			*(output + i + pos) = *(yhat + i);
		pos += n1;
		for (int i = 0; i < n2; ++ i)
			*(output + i + pos) = *(yhat_next + i);
		free(yhat);
	}
	free(xTw);
	free(xTwx_next);
	free(xTwx_iter);
	free(xTwy_next);
	free(xTwy_iter);
	free(predict);
	free(yhat_next);
	return output;
}
int main()
{
	srand(time(0));
	double a[8] = {(double)rand(), (double)rand(), (double)rand(), 
					(double)rand(), (double)rand(), (double)rand(), 
					(double)rand(), (double)rand()};
	double b[9] = {(double)rand(), (double)rand(), (double)rand(), 
					(double)rand(), (double)rand(), (double)rand(), 
					(double)rand(), (double)rand(), (double)rand()};
	double c[3] = {(double)rand(), (double)rand(), (double)rand()};
	double d[6] = {(double)rand(), (double)rand(), (double)rand(), 
					(double)rand(), (double)rand(), (double)rand()};
	double e[2] = {(double)rand(), (double)rand()};
	double f[2] = {(double)rand(), (double)rand()};
	double *aa = add(a, b, 2, 3);
	double *bb = times(a, b, 2, 3, 2);
	double *cc = calc_xTw(a, c, 2, 3);
	double *output = wls_iter(a, b, c, d, e, f, 5, 2, 3, 1);
	int tot_len = 4 + 2 * 2 + 3 * (3 + 2);
	printf("%d\n", tot_len);
	for (int i = 0; i < tot_len; ++ i)
		printf("%lf\n", *(output + i));
	for (int i = 0; i < 1000000000; ++ i)
	{
		double *test = wls_iter(a, c, f, b, d, e, 0, 2, 3, 1);
		free(test);
	}
	return 0;
}

