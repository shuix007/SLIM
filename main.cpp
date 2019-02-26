#include "PureSlim.hpp"

using namespace std;

inline int rand_between(int min, int max) {
  return (int)(rand() % (max - min + 1) + min);
}

int max_ele(vector<int>& a) {
  int m = 0;
  for (const auto &it : a) {
    if (it > m) {
      m = it;
    }
  }
  return m;
}

/* load the data set */
vector<unordered_map<int, int> > load_data(const char* file_name, int& num_user, int& num_item) {
  // read in reading list with temporal information
  ifstream infile;
  infile.open(file_name);
  if (!infile) {
    cout << "File Does Not Exist!!!" << endl;
    exit(0);
  }
  int uid;
  int sid;
  int t;
  
  string user;
  string item;
  
  vector<unordered_map<int, int> > points;
  unordered_map<string, int> user2id;
  unordered_map<string, int> item2id;

  num_user = 0;
  num_item = 0;

  // read from .csv file
  string delimiter = ",";
  string value;
  while (getline(infile, value)) {
    int pos = 0;
    string token;

    pos = value.find(delimiter);
    user = value.substr(0, pos);
    value.erase(0, pos + delimiter.length());

    pos = value.find(delimiter);
    item = value.substr(0, pos);
    value.erase(0, pos + delimiter.length());

    t = (int)stoi(value);
    
    if (item2id.find(item) == item2id.end()) {
      item2id[item] = num_item++;
    }
    
    if (user2id.find(user) == user2id.end()) {
      user2id[user] = num_user++;
      unordered_map<int, int> tp_doc;
      points.push_back(tp_doc);
    }
    
    uid = user2id[user];
    sid = item2id[item];
    points[uid][sid] = t;
  }

  infile.close();
  return points;
}

template <class T>
vector<unordered_map<int, T> > transpose(vector<unordered_map<int, T> >& R, int nrows, int ncols) {
  if ((int)R.size() != nrows) {
    cout << "Dim not fit!" << endl;
    exit(0);
  }
  vector<unordered_map<int, T> > result(ncols);

  for (int i = 0; i < nrows; ++i) {
    for (const auto &it : R[i]) {
      result[it.first][i] = it.second;
    }
  }

  return result;
}

/* take out the last item as test/validation set */
vector<int> leave_one_out(vector<unordered_map<int, int> >& R, vector<unordered_map<int, int> >& R_item) {
  int num_users = R.size();
  vector<int> leave_out_list(num_users);
  
  for (int i = 0; i < num_users; ++i) {
    vector<int> latest_item_list;
    int latest_time = 0;
    
    /* find the latest time */
    for (const auto &it : R[i]) {
      if (it.second > latest_time) {
        latest_time = it.second;
      }
    }
    
    /* find the latest item */
    for (const auto &it : R[i]) {
      if (it.second == latest_time) {
        latest_item_list.push_back(it.first);
      }
    }
    
    auto random_it = std::next(begin(latest_item_list), rand_between(0, latest_item_list.size() - 1));
    leave_out_list[i] = *random_it;
    R[i].erase(*random_it);
    R_item[*random_it].erase(i);
  }
  
  return leave_out_list;
}

int main(int argc, const char * argv[]) {
  int num_user = 0;
  int num_item = 0;
  srand(1);
  
  const char * train_file_name = argv[1];
  /* hyperparameters for training model */
  double lambda_1 = (double)atof(argv[2]);
  double lambda_2 = (double)atof(argv[3]);
  
  /* top_n */
  int n = (int)atoi(argv[4]);
  
  /* number of threads */
  int p = (int)atoi(argv[5]);
  
  /* convergence criteria */
  double tol = (double)atof(argv[6]);
  
  cout << "Program Start!" << endl;
  vector<unordered_map<int, int> > R_user = load_data(train_file_name, num_user, num_item);
  vector<unordered_map<int, int> > R_item = transpose(R_user, num_user, num_item);
  cout << "Training data loaded!" << endl;
  
  vector<int> leave_out_list = leave_one_out(R_user, R_item);
  
  PureSlim m(num_item, num_user);
  
  m.train(R_item, lambda_1, lambda_2, tol, p);
  
  /* evaluate the performance of the model */
  vector<vector<int> > rec_list = m.predict(R_user, n, p);
  double HR = 0.;
  double AR = 0.;
  for (int i = 0; i < num_user; ++i) {
    for (int j = 0; j < n; ++j) {
      if (rec_list[i][j] == leave_out_list[i]) {
        HR += 1.;
        AR += 1. / (j + 1);
      }
    }
  }
  cout << "HR = " << HR / num_user << ", AR = " << AR / num_user << endl;
  
  return 0;
}
