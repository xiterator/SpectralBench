#include "stdafx.h"
#include "ClusterMembership.h"

#include <vector>
#include <iostream>
#include <algorithm>

std::ostream& operator<<(std::ostream& out, const ClusterMembership& cd) {

    cd.dump(out);
    return out;
}

void ClusterMembership::dump(std::ostream& out) const {

	out << label << ":" << std::endl;

    std::vector<unsigned int> ids(this->begin(), this->end());
	std::sort(ids.begin(), ids.end());
	for (const auto& i : ids) {
		out << " " << i;
	}
}



ClusterMembership::ClusterMembership() {}

ClusterMembership::ClusterMembership(const ClusterMembership& other) :
IdSet(other), label(other.label)  { }


void ClusterMembership::setLabel(const std::string& lab) {
	label.assign(lab);
}
const std::string& ClusterMembership::getLabel() const {
	return label;
}


void ClusterMembership::addMember(unsigned int recordId) {

    if (this->find(recordId) != this->end())
		return;
    this->insert(recordId);
}


//////////////////////////////////////////////////////
//set operation
void ClusterMembership::intersect(const ClusterMembership& other, ClusterMembership& result) const {
	
    const ClusterMembership* small;
    const ClusterMembership* big;
    if (this->size() < other.size()) {
        small = this;
        big = &other;
    }
    else {
        small = &other;
        big = this;
    }

    for (const auto& i : *small) {
        if (big->find(i) != big->end())
            result.insert(i);
    }
    result.label += label;
    result.label += "^";
    result.label += other.label;
}
void ClusterMembership::minus(const ClusterMembership& other, ClusterMembership& result) const {

    for (const auto& i : *this) {
        if (other.find(i) == other.end())
           result.insert(i);				//keep it if not found
    }
    result.label += label;
    result.label += "-";
    result.label += other.label;
}


unsigned int ClusterMembership::intersect(const ClusterMembership& truth) const {

    ClusterMembership result;
    intersect(truth, result);
    return result.size();

}

unsigned int ClusterMembership::minus(const ClusterMembership& truth) const {
    ClusterMembership result;
    minus(truth, result);
    return result.size();
}
unsigned int ClusterMembership::symMinus(const ClusterMembership& truth) const {
    unsigned int mc = minus(truth);
    mc += truth.minus(*this);
    return mc;
}

///////////////////////////////////////////////////////
//evaluation

double ClusterMembership::getEvalValue(const ClusterMembership& truth, EvalIndex::Name evalIndex, unsigned int totalMembers/* = 0*/) const {

	switch (evalIndex)
	{
	case EvalIndex::accuracy:
		return getEV_accuracy(truth, totalMembers);
		
	case EvalIndex::F1_score:
		return getEV_F1_score(truth);

	case EvalIndex::G_measure:
		return getEV_G_measure(truth);

	case EvalIndex::precision:
		return getEV_precision(truth);

	case EvalIndex::recall:
		return getEV_recall(truth);
		
	default:
		ASSERT_WITH_MSG(false, "ClusterMembership::getEvalValue..unknown evalIndex=" << evalIndex);
		break;
	}
	return 0;
}

double ClusterMembership::getEV_precision(const ClusterMembership& truth) const {

	//confusion matrix:
	//              TrueP        TrueN
    //  GaussP        TP           FP       P*
    //  GaussN        FN           TN       N*
	//                P             N

	//precision = TP/P*

    double TP = this->intersect(truth);
	double Pstar = this->size();
	if (Pstar < 1) Pstar = 1;			//yjf.approximation
	return TP / Pstar;

}
double ClusterMembership::getEV_recall(const ClusterMembership& truth) const {
	//confusion matrix:
	//              TrueP        TrueN
    //  GaussP        TP           FP       P*
    //  GaussN        FN           TN       N*
	//                P             N

	//recall = TP/P
    double TP = this->intersect(truth);
	double P = truth.size();
	if (P < 1) P = 1;			//yjf.approximation
	return TP / P;
}
double ClusterMembership::getEV_F1_score(const ClusterMembership& truth) const {

	//F_beta = (1+beta^2) (precision * recall) / [ beta^2 * precision + recall ]
	//F1_score is F_beta, where beta = 1.

	double beta = 1;

    double TP = this->intersect(truth);
	double Pstar = this->size();

	if (Pstar < 1) Pstar = 1;			//yjf.approximation
	double precision = TP / Pstar;

	double P = truth.size();
	if (P < 1) P = 1;			//yjf.approximation
	double recall = TP / P;

	double d = beta * beta * precision + recall;
	if (d < Constant::nearZero) d = Constant::nearZero;

	return (1 + beta * beta) * (precision * recall) / d;

}

double ClusterMembership::getEV_accuracy(const ClusterMembership& truth, unsigned int totalMembers) const {
	//confusion matrix:
	//              TrueP        TrueN
    //  GaussP        TP           FP       P*
    //  GaussN        FN           TN       N*
	//                P             N

	//accuracy =( TP + TN ) / [ TP + (FP + FN) + TN ]
	//         = (TP + TN) / totalMembers

	//TN = GuassN ^ TrueN = (total - GuassP) ^ (total - TrueP)
	//  = total - total ^ TrueP - GuassP ^ total + GuassP ^ TrueP
	//  = total - TrueP - GuassP + TP

	//  = total - (total - truth.size) - (total-this.size) + TP  //error
	//  = truth.size + this.size + TP - total				     //error

    double TP = this->intersect(truth);
	//double TN = this->size() + truth.size() + TP - totalMembers;
	double TN = totalMembers - truth.size() - this->size() + TP;

	ASSERT_WITH_MSG(TN > 0, "ClusterMembership::getEV_accuracy.. totalMembers is wrong, TN=" << TN << ", totalN=" << totalMembers);

	return (TP + TN) / totalMembers;
}

double ClusterMembership::getEV_G_measure(const ClusterMembership& truth) const {

	//G_measure = sqrt(precision * recall)

    double TP = this->intersect(truth);
	double Pstar = this->size();
    if (Pstar < 1) Pstar = 1;			//yjf.approximation
	double precision = TP / Pstar;

	double P = truth.size();
	if (P < 1) P = 1;				//yjf.approximation
	double recall = TP / P;

	return ::sqrt(precision * recall);

}
