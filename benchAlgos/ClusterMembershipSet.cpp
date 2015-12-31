#include "stdafx.h"
#include "ClusterMembershipSet.h"
#include <math.h>


std::ostream& operator<<(std::ostream& out, const ClusterMembershipSet& cds) {
	for (const auto& i : cds) {
		out << i << std::endl;
	}
	return out;
}

unsigned int ClusterMembershipSet::checkEqualMembers(const ClusterMembershipSet& truth) const {

	unsigned int totalMembers = 0;
	for (const auto& t : truth) {
		totalMembers += t.size();	//we assume that both set has same total members: n
	}
	unsigned int thisMembers = 0;
	for (const auto& h : (*this)) {
		thisMembers += h.size();	//we assume that both set has same total members: n
	}
	ASSERT_WITH_MSG(totalMembers == thisMembers, "ClusterMembershipSet::checkEqualMembers...members must be equal");

	return totalMembers;
}

void ClusterMembershipSet::compProfitMat(const ClusterMembershipSet& truth, EvalIndex::Name evalIndex, GAmatrix& profitMat) const {

	unsigned int totalMembers = checkEqualMembers(truth);

	//yjf.debug
	//if (evalIndex == EvalIndex::recall)
	//	std::cout << "compProfitMat: EvalIndex=" << EvalIndex::tostr(evalIndex) << std::endl;

	unsigned int i = 0;
	unsigned int i_size = this->size();
	unsigned int j;
	unsigned int j_size = truth.size();
	double ev;
	for (; i < i_size; ++i) {
		for (j = 0; j < j_size; ++j) {

			 ev = (*this)[i].getEvalValue(truth[j], evalIndex, totalMembers);
			 profitMat(i, j) = ev * truth[j].size() * 1.0 / totalMembers;  //weighted eval value
			
			 //yjf.debug.
			//if (evalIndex == EvalIndex::recall)
			//	std::cout << "(" << i << ", " << j << "):" << profitMat(i, j) << "\t";
		}
		//yjf.debug
		//std::cout << std::endl;
	}
}

//CA: Cluster Accuracy
double ClusterMembershipSet::compCA(const ClusterMembershipSet& truth, std::vector<long>& assignToTruth) const {

    double totalMembers = checkEqualMembers(truth);

    unsigned int i;
    unsigned int i_size = this->size();
    unsigned int j;
    unsigned int j_size = truth.size();
    GAmatrix matr(i_size, j_size);

    for (i = 0; i < i_size; ++i) {
        for (j = 0; j < j_size; ++j) {
                matr(i, j) = (*this)[i].intersect(truth[j]);
        }
    }

    double shared = matr.maxIntersectMatch(assignToTruth);
    std::cout <<"\t\t\t\t\t\t\t\t\t\t\t" << shared << " / " << totalMembers << std::endl;
    return shared / totalMembers;

}

double ClusterMembershipSet::compNMI(const ClusterMembershipSet& truth) const {

	double n = 0;		//we assume that both set has same total members: n
	for (const auto& k : (*this)) {
		n += k.size();
	}

	double size_j;
	double B = 0;
	for (const auto& j : truth) {
		size_j = j.size();
		if (size_j > Constant::nearZero)
			B += size_j * std::log(size_j / n);					//sum (size_j * ln[size_j/n] )
	}
	

	double A = 0;
	double S = 0;
	double size_i, size_s;
	for (const auto& i: (*this)) {
		size_i = i.size();
		if (size_i > Constant::nearZero)
			A += size_i * std::log(size_i / n);					//sum (size_i * ln[size_i/n] )
		else
			continue;	//size_i <= nearZero implies size_s <= nearZero by (1)
		
		for (const auto& j : truth) {
			size_j = j.size();
            size_s = i.intersect(j);		//(1)

			if (size_j > Constant::nearZero && size_s > Constant::nearZero)
				S += size_s * std::log(n * size_s / (size_i * size_j));
		}
	}
	

	if (A < 0)  A *= -1;
	if (B < 0) B *= -1;

	setLB(A);
	setLB(B);

	double d = sqrt(A * B);

	return S / d;

}
///////////////////////////////////////////////////////////////////////////////////////

void ClusterMembershipSet::compSimilarMatches(const ClusterMembershipSet& truth, GAmatrix& matches) const {

	//including the additional eval index value at the first column
	matches.resize(EvalIndex::_SIZE, truth.size() + 1);


	unsigned int i;
	unsigned int sz = EvalIndex::_SIZE;

	std::vector<long> assignToTruth;
	assignToTruth.reserve(truth.size());        //yjf.WARNING. preallocate vector's size

	for (i = 0; i < sz; ++i, assignToTruth.clear()) {
		matches(i, 0) = this->similarMatch(truth, EvalIndex::Name(i), assignToTruth);

		//copy to matches(i, 1:)
		unsigned int j = 1;
		for (const auto& a : assignToTruth) {
			matches(i, j) = a;
			++j;
		}
	}//for
}


double ClusterMembershipSet::similarMatch(const ClusterMembershipSet& truth,
	EvalIndex::Name evalIndex, std::vector<long>& assignToTruth) const {

	if (truth.size() == 0) {
		return 0;		//not cluster labels given.
	}

	if (evalIndex == EvalIndex::NMI) {

		assignToTruth.assign(truth.size(), 0);	//assign to nothing.
		double nmi = compNMI(truth);

		return nmi;
	}

	if (evalIndex == EvalIndex::CA) {
		return compCA(truth, assignToTruth);
	}


	//profitMat: columns index the true clusters
	GAmatrix profitMat(this->size(), truth.size());
	
	compProfitMat(truth, evalIndex, profitMat);
	double ave = profitMat.maxProfitMatch(assignToTruth);
	return ave;
}
////////////////////////////////////////////////////////////////////

bool ClusterMembershipSet::keepStableHistory(const ClusterMembership& newMemb) {

    unsigned int sz = this->size();
    if (sz == (Constant::cluMembQueueLen)) {
        this->erase(this->begin(), this->begin() + 1);
        sz -= 1;
    }

    if (sz > 0) {

        unsigned int pre = (*this)[sz-1].symMinus(newMemb);
        unsigned int first;
        if (sz > 1) {
            first = (*this)[0].symMinus(newMemb);
        }
        else {
            first = pre;
        }

        //yjf.debug.
        const std::string& lab = (*this)[0].getLabel();

        std::cout << "\t\t\t================> " << lab << ": size " << sz;
        std::cout << ", pre= " << pre << ", first=" << first << std::endl;



        if (pre > 2 || first > 2) {

            this->clear();
            this->push_back(new ClusterMembership(newMemb));
            return false;
        }

        this->push_back(new ClusterMembership(newMemb));
        return (sz+1) >= Constant::cluMembQueueLen;


    }
    else {

        //sz == 0
        this->push_back(new ClusterMembership(newMemb));
        return false;
    }
}

bool ClusterMembershipSet::isConvergent() const {
    return this->size() >= Constant::cluMembQueueLen;
}
