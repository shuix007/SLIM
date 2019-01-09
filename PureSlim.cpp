#include "PureSlim.hpp"

PureSlim::PureSlim(int numItem, int numUser)
{
  nItem = numItem;
  nUser = numUser;
}

void PureSlim::cleanUp()
{
  wtItem.clear();
  wtItem.shrink_to_fit();
}

double PureSlim::ip_tp(const unordered_map<int, int>& reg, const unordered_map<int, int>& tar)
{
  double ans = 0.;
  
  for (const auto &it : reg)
  {
    if (tar.find(it.first) != tar.end())
    {
      ans += 1.;
    }
  }
  
  return ans;
}

void PureSlim::add_hat_tp(const unordered_map<int, int>& rl, double weight, vector<double>& y_hat)
{
  if (weight > 0)
  {
    for (const auto &it : rl)
    {
      y_hat[it.first] += weight;
    }
  }
}


void PureSlim::subtract_hat_tp(const unordered_map<int, int>& rl, double weight, vector<double>& y_hat)
{
  if (weight > 0)
  {
    for (const auto &it : rl)
    {
      y_hat[it.first] -= weight;
    }
  }
}

double PureSlim::ip_faster_tp(const unordered_map<int, int>& rl, vector<double>& y_hat)
{
  double ans = 0.;
  
  for (const auto &it : rl)
  {
    ans += y_hat[it.first];
  }
  
  return ans;
}

unordered_map<int, double> PureSlim::norm_tp_x(const vector<unordered_map<int, int> >& R, unordered_map<int, double>& xy_product)
{
  int n = xy_product.size();
  unordered_map<int, double> norm;
  norm.reserve(n);
  
  for (const auto &it : xy_product)
  {
    norm[it.first] = (double)R[it.first].size();
  }
  
  return norm;
}


void PureSlim::train_slim(const vector<unordered_map<int, int> >& R, vector<double>& w, int y, double l1, double l2, double tol)
{
  int T = 50;
  int nCol = R.size();
  int nRow = nUser;
  
  /* precalculate the inner product of y and each x */
  unordered_map<int, double> xy_product;
  for (int i = 0; i < nCol; ++i)
  {
    if (i != y)
    {
      double ip = ip_tp(R[i], R[y]);
      if (ip > 0.)
      {
        xy_product[i] = ip;
      }
      else
      {
        w[i] = 0.;
      }
    }
  }
  
  /* declare the weight, # of weight is equal to the # of non-zero xy_product */
  if (xy_product.size() == 0) return;
  /* precalculate the estimate of y using all x (all 0s by default) */
  vector<double> y_hat(nRow, 0);
  /* precalculate the norm of x's regard of y */
  unordered_map<int, double> x_norm = norm_tp_x(R, xy_product);
  /* updating the weight */
  for (int t = 0; t < T; ++t)
  {
    double delta_weight = 0.;
    for (const auto &it : xy_product)
    {
      int i = it.first;
      double tp_weight = w[i];
      
      subtract_hat_tp(R[i], tp_weight, y_hat);
      double upper = it.second - ip_faster_tp(R[i], y_hat);
      
      /* Eq(5) in Regularization Paths for Generalized Linear Models via Coordinate Descent */
      w[i] = upper > l1 ? (upper - l1) / (x_norm[i] + l2) : 0.;
      
      add_hat_tp(R[i], w[i], y_hat);
      delta_weight += (w[i] - tp_weight) * (w[i] - tp_weight);
    }
    
    /* break if the norm of delta weight is less than tol */
    if (delta_weight < tol)
    {
      break;
    }
  }
}

void PureSlim::train(const vector<unordered_map<int, int> >& R, double l1, double l2, double tol, int n_threads)
{
  cout << "<----------ITEM MODEL---------->" << endl
  << "l1 = " << l1 << ", l2 = " << l2 << endl;
  
  int nCol = R.size();
  cout << nCol << endl;
  
  if (nCol != nItem)
  {
    cout << "Training dim not fit!" << endl;
    exit(0);
  }
  
  wtItem.clear();
  wtItem.resize(nItem);
  
  vector<unordered_map<int, double> > wt(nCol);
  
  clock_t start_train = clock();
  
  #pragma omp parallel shared (R, l1, l2, tol, nCol) num_threads(n_threads)
  {
    #pragma omp for schedule(dynamic)
    for (int i = 0; i < nCol; ++i)
    {
      vector<double> weight(nCol, 0.);
      train_slim(R, weight, i, l1, l2, tol);
      for (int j = 0; j < nCol; ++j)
      {
        if (weight[j] > 0.)
        {
          wt[i][j] = weight[j];
        }
      }
    }
  }
  
  for (int i = 0; i < nCol; ++i)
  {
    for (const auto &it : wt[i])
    {
      wtItem[it.first][i] = it.second;
    }
  }
  
  clock_t end_train = clock();
  
  cout << "Training Finished!" << endl;
  cout << "Training time = " << (double)(end_train - start_train) / CLOCKS_PER_SEC << endl;
}

vector<vector<int> > PureSlim::predict(vector<unordered_map<int, int> >& R_test, int n, int n_threads)
{
  if (wtItem.size() == 0)
  {
    cout << "Item model not exist!" << endl;
    exit(0);
  }
  cout << "<--------Normal Result!-------->" << endl;
  clock_t start_test = clock();
  
  int nTest = R_test.size();
  
  vector<vector<int> > rec_list(nTest, vector<int>(n, 0));
  #pragma omp parallel shared (R_test, rec_list) num_threads(n_threads)
  {
    #pragma omp for schedule(dynamic)
    for (int i = 0; i < nTest; ++i)
    {
      if (R_test[i].size() > 0)
      {
        vector<int> tp_rec = predict_list(R_test[i], n);
        for (int j = 0; j < n; ++j)
        {
          rec_list[i][j] = tp_rec[j];
        }
      }
      else
      {
        rec_list[i][0] = -1;
      }
    }
  }
  
  clock_t end_test = clock();
  
  cout << "Test time = " << (double)(end_test - start_test) / CLOCKS_PER_SEC << endl;
  return rec_list;
}

vector<int> PureSlim::predict_list(const unordered_map<int, int>& seed, int n)
{
  vector<double> sc = predict_score(seed);
  vector<pair<double, int> > score(nItem);
  vector<int> rec(n);
  
  for (int i = 0; i < nItem; ++i)
  {
    score[i].second = i;
    score[i].first = sc[i];
  }
  
  sort(score.begin(), score.end());
  for (int i = nItem - 1, count = 0; i >= 0; i--)
  {
    if (seed.find(score[i].second) == seed.end())
    {
      rec[count++] = score[i].second;
      
      if (count == n) break;
    }
  }
  return rec;
}

vector<double> PureSlim::predict_score(const unordered_map<int, int>& R_test)
{
  vector<double> score(nItem, 0.);
  
  for (const auto &it : R_test)
  {
    for (const auto &inner_it : wtItem[it.first])
    {
      score[inner_it.first] += inner_it.second;
    }
  }
  
  return score;
}

void PureSlim::load_weight(const char* filename)
{
  cout << "Number of Item is: " << nItem << endl;
  wtItem.clear();
  wtItem.resize(nItem);
  // read in reading list with temporal information
  ifstream infile;
  infile.open(filename);
  if (!infile)
  {
    printf("File Does Not Exist!!!");
  }
  int u_id;
  int s_id;
  double weight;
  
  // read from .csv file
  string delimiter = ",";
  string value;
  
  while (getline(infile, value))
  {
    int pos = 0;
    string token;
    
    pos = value.find(delimiter);
    token = value.substr(0, pos);
    u_id = (int)stoi(token);
    value.erase(0, pos + delimiter.length());
    
    pos = value.find(delimiter);
    token = value.substr(0, pos);
    s_id = (int)stoi(token);
    value.erase(0, pos + delimiter.length());
    
    weight = (double)stof(value);
    
    if (s_id >= nItem || u_id >= nItem)
    {
      cout << "Dim not fit!!" << endl;
      exit(0);
    }
    
    wtItem[s_id][u_id] = weight;
  }
  
  infile.close();
}

void PureSlim::write_weight(const char* filenameItem)
{
  ofstream myfile(filenameItem);
  
  for (int i = 0; i < nItem; ++i)
  {
    for (const auto &it : wtItem[i])
    {
      myfile << to_string(i) << "," << to_string(it.first) << ","
      << to_string(it.second) << "\n";
    }
  }
  myfile.close();
}
