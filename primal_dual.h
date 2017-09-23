#include <iostream>
#include <math.h>

template <class T>
void updateU (T *u, T *tau, T *div_y, T *, T *f, T *f_y, int num_vertex);

template <class T>
void updateY (T *y, T *sigma, T *grad_f_y, int num_edge);

template <class T>
void compute_dt(T *tau, T *sigma, T *w_u, T alpha, T phi, int num_vertex, int num_edge);

template <class T>
void compute_L1 (T *grad_u, T &u_norm, int num_vertex);

template <class T>
void compute_L2 (T *gap_vec, T &gap, int num_vertex);

template <class T>
void compute_scalar_product (T *u, T *f, T &uf, int num_vertex);

template <class T>
void get_max (T *div_y, T *f, T *max_vec, int num_vertex);

template <class T>
void compute_gap(T *u, T *f, T *div_y, T &gap, int num_vertex, int num_edge);


			  
