#pragma once

#include "ClusterMembership.h"
#include "ClusterMembershipSet.h"
#include "DataSet.h"
#include "GAmatrix.h"

#include <vector>
#include <iostream>
#include <string>

#include <boost/multiprecision/cpp_dec_float.hpp>

namespace mp = boost::multiprecision;

//////////////////////////////////////////////////////////////////////////////

struct InitialMethod {
	enum Name {V_method1=0, V_method2, V_method3, V_method4, _SIZE};
	static std::string tostr(InitialMethod::Name n);
};

struct VupdateMethod {
	enum Name { m1_korea = 0, m2_us, m3_china, _SIZE };
	static std::string tostr(VupdateMethod::Name n);
};

class Cluster
{
public:
    typedef double value_type;


private:

	Record2 z_m;		//the center for this cluster
	GAmatrix V_txm;		//the contributions of all attributes to the group variables Gs in this cluster
	Record2 w_t;		//the weights of group variables Gs

	ClusterMembership h_members;			//the members of this cluster
    ClusterMembershipSet historyMembers;
    //std::vector<double> historyCluErrs;
        
	value_type  lambda;				//penalty parameter on the 2-norm of w_t
	value_type  betaV;				//penalty parameter for orthogonality of V
	const RecordSet2* precords;		//the dataset we will use

	InitialMethod::Name initialMethod;
	VupdateMethod::Name vUpdateMethod;

    typedef mp::number<mp::cpp_dec_float<10> > Decimal10;
    Decimal10 oldWsparseIndex;
    Decimal10 oldOrthIpIndex;


	void setCenterFromExisted(std::vector<Record2*>& existingCenters);
	void setCenterByAverage(std::vector<Record2*>& outputCenter);
    void setCenterByRandom();

	void initialCenter(std::vector<Record2*>& existingCenters);
	void initialW();

	void initialV_method1();
	void initialV_method2(unsigned int rowAllocStdev = 1);
	void initialV_method3();
	void initialV_method4();
	void initialRow(unsigned int row, Record2& occupiedColumns, int toNum, int& remainedCount);


	void normalizeV();
	void normalizeW();

	void updateCenter();
	void updateW();
	void updateV();
	void updateV2();
	void updateV3();

	void compPp_Pm(GAmatrix& Pp_mxm, GAmatrix& Pm_mxm) const;
    void compPp_Pm2(GAmatrix& Pp_mxm, GAmatrix& Pm_mxm) const;
	void compQ_txt(GAmatrix& Q_txt) const;




public:

	Cluster();
	~Cluster();

	//initial
    void init0_setting(unsigned int clusterNo, unsigned int t_groupNum, unsigned int m_attNum,
		value_type lamb, value_type beta, const RecordSet2* prds, VupdateMethod::Name n);
    void init1_random(std::vector<Record2*>& existingCenters, InitialMethod::Name n);


	//result
	const ClusterMembership& getMembers() const;

    value_type getClusteringError() const;
    value_type getObjectValue() const;
    value_type getOrthIpIndex() const;
    value_type getWsparseIndex() const;

    bool isConvergent();
                

    const Record2& getCenter() const;
	const GAmatrix& getV() const;
	const Record2& getW() const;
	value_type getLambda() const;
	value_type getBeta() const;
    unsigned int getNumGroups() const;
        
	InitialMethod::Name getInitialMethod() const;

	//updating rules
	void clearAllMembers();			//clear members in this cluster
	void update();	//updating center, V and W.	

	value_type distanceToMe(unsigned int id) const;
	void assignMember(unsigned int id);	//id: is the RecordSet records' index of the instance
	

};

