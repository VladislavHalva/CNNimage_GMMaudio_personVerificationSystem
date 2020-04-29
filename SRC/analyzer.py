class analyzer:
  def __init__(self, cl1Data, cl2Data):
    self.cl1Data = cl1Data
    self.cl2Data = cl2Data

  def PCA_2dim(self, show=True):
    cov_tot = np.cov(np.vstack([self.cl1Data, self.cl2Data]).T, bias=True)
    # take just 2 largest eigenvalues and corresponding eigenvectors
    d, e = scipy.linalg.eigh(cov_tot, eigvals=(dim-2, dim-1))

    cl1pca = self.cl1Data.dot(e) 
    cl2pca = self.cl2Data.dot(e) 
    
    if show:
      plt.plot(cl1pca[:,1], cl1pca[:,0], 'r.', ms=1, label="třída 1")
      plt.plot(cl2pca[:,1], cl2pca[:,0], 'b.', ms=1, label="třída 1")
      plt.legend()
      plt.title("Graf PCA redukce do dvou dimenzí.")
      plt.xlabel("x")
      plt.ylabel("y")
      plt.show()
    return [cl1pca,cl2pca]

  def LDA_1dim(self, show=True):
    # LDA target train vs non-target train
    numCl1 = len(self.cl1Data)
    numCl2 = len(self.cl2Data)

    cov_total = np.cov(np.vstack([self.cl1Data, self.cl2Data]).T, bias=True)
    cov_withinClasses = (numCl1*np.cov(self.cl1Data.T, bias=True) + numCl2*np.cov(self.cl2Data.T, bias=True)) / (numCl1 + numCl2)
    cov_acrossClasses = cov_total - cov_withinClasses
    d, e = scipy.linalg.eigh(cov_acrossClasses, cov_withinClasses, eigvals=(dim-1, dim-1))
    
    if show:
      plt.figure()
      junk = plt.hist(self.cl1Data.dot(e), 40, histtype='step', color='b' , label="třída 1")
      junk = plt.hist(self.cl2Data.dot(e), 40, histtype='step', color='r', label="třída 2")
      plt.xlabel("Hodnota dimenze dle LDA")
      plt.ylabel("Počet rámců")
      plt.legend()
      plt.title("Histogram LDA redukce do 1 třídy")
      plt.show()
    return [self.cl1Data.dot(e), self.cl2Data.dot(e)]