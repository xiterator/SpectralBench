#pragma once

#include <string>

#include <boost/unordered_set.hpp>
#include "EvalIndex.h"

namespace bun = boost::unordered;

typedef bun::unordered_set<unsigned int> IdSet;

class ClusterMembership: public IdSet
{
private:
	std::string label;

	double getEV_precision(const ClusterMembership& truth) const;
	double getEV_recall(const ClusterMembership& truth) const;
	double getEV_F1_score(const ClusterMembership& truth) const;
	double getEV_G_measure(const ClusterMembership& truth) const;

	double getEV_accuracy(const ClusterMembership& truth, unsigned int totalMembers) const;


	

public:
	ClusterMembership();
	ClusterMembership(const ClusterMembership& other); //clone

	void setLabel(const std::string& lab);
	const std::string& getLabel() const;

    void addMember(unsigned int recordId);

	//set operation
	void intersect(const ClusterMembership& truth, ClusterMembership& result) const;
    unsigned int intersect(const ClusterMembership& truth) const;
	void minus(const ClusterMembership& truth, ClusterMembership& result) const;
    unsigned int minus(const ClusterMembership& truth) const;
    unsigned int symMinus(const ClusterMembership& truth) const;

    double getEvalValue(const ClusterMembership& truth, EvalIndex::Name evalIndex,
                        unsigned int totalMembers=0) const;

	//dump
	void dump(std::ostream& out) const;
};

std::ostream& operator<<(std::ostream& out, const ClusterMembership& cd);
