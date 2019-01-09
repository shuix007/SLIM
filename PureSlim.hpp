#ifndef PureSlim_hpp
#define PureSlim_hpp


#endif /* PureSlim_hpp */

#include <iostream>
#include <vector>
#include <fstream>
#include <unordered_map>
#include <map>
#include <random>
#include <algorithm>
#include <ctime>
#include <math.h>
#include <omp.h>

using namespace std;

class PureSlim
{
public:
  int nItem;
  int nUser;
  
  /* matrix W for item model */
  vector<unordered_map<int, double> > wtItem;
  
  PureSlim(int numItem, int numUser);
  ~PureSlim(){};
  
  void cleanUp();
  
  /* auxilary functions for training the model */
  double ip_tp(const unordered_map<int, int>& reg, const unordered_map<int, int>& tar);
  void add_hat_tp(const unordered_map<int, int>& rl, double weight, vector<double>& y_hat);
  void subtract_hat_tp(const unordered_map<int, int>& rl, double weight, vector<double>& y_hat);
  double ip_faster_tp(const unordered_map<int, int>& rl, vector<double>& y_hat);
  unordered_map<int, double> norm_tp_x(const vector<unordered_map<int, int> >& R, unordered_map<int, double>& xy_product);
  
  /* train one elastic net model (coordinate descent)*/
  void train_slim(const vector<unordered_map<int, int> >& R, vector<double>& w, int y, double l1, double l2, double tol);
  
  /* train slim model */
  void train(const vector<unordered_map<int, int> >& R, double l1, double l2, double tol, int n_threads);
  
  /* predict */
  vector<vector<int> > predict(vector<unordered_map<int, int> >& R_test, int n, int n_threads);
  vector<int> predict_list(const unordered_map<int, int>& seed, int n);
  vector<double> predict_score(const unordered_map<int, int>& R_test);
  
  void load_weight(const char* filename);
  void write_weight(const char* filenameItem);
};
