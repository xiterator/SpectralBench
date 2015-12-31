#include "stdafx.h"
#include <stdlib.h>
#include <sstream>
#include <string>

#include "ClusterMembership.h"
#include "Cluster.h"
#include "RandomDice.h"


std::string InitialMethod::tostr(InitialMethod::Name n) {
	switch (n)
	{
	case InitialMethod::V_method1:
		return "Vm1";
	case InitialMethod::V_method2:
		return "Vm2";
	
	case InitialMethod::V_method3:
		return "Vm3";

	case InitialMethod::V_method4:
		return "Vm4";

	default:
		return "default.unknown";
	}
	return "unknown";
}

std::string VupdateMethod::tostr(VupdateMethod::Name n) {

	switch (n) {
	case VupdateMethod::m1_korea:
		return "m1_korea";

	case VupdateMethod::m2_us:
		return "m2_us";

	case VupdateMethod::m3_china:
		return "m3_china";

	default:
		return "Vupdate.unknown";
	}
	return "Vupdate.unknown.ret";
}


/////////////////////////////////////////////////////////////////////////////////////

void Cluster::init0_setting(unsigned int clusterNo, unsigned int t_groupNum, unsigned int m_attNum,
	value_type lamb, value_type beta, const RecordSet2* prds, VupdateMethod::Name n) {

	betaV = beta;
	lambda = lamb;
	precords = prds;
	vUpdateMethod = n;

	std::cout << "-----> V update: " << VupdateMethod::tostr(n) << std::endl;

    historyMembers.clear();
    oldWsparseIndex=0;
    oldOrthIpIndex = 1;

	z_m.clear();
	z_m.resize(m_attNum);
	
	w_t.clear();
	w_t.resize(t_groupNum);
	
	V_txm.clear();
	V_txm.resize(t_groupNum, m_attNum);
	
	h_members.clear();

	//set a cluster label
	std::stringstream ss;
	ss << "c_" << clusterNo;
	h_members.setLabel(ss.str());
}

Cluster::Cluster() : precords(0) {}

Cluster::~Cluster()
{
}

////////////////////////////////////////////////////////////
//updating rules

Cluster::value_type Cluster::distanceToMe(unsigned int id) const {

	//|| diag(w_t) * V_txm * (x_id - z_m) || ^ 2


    Record2 diff( precords->at(id) );
    diff -= z_m;			//x_id - z_m

    Record2 vect;
    vect = (arma::mat)V_txm * (arma::vec) diff;		// V_txm * (x_id - z_m)
    vect.hadamardProd(w_t);     //diag(w_t) * V_txm * (x_id - z_m)

    double res = vect.innerProduct(vect);  //||diag(w_t) * V_txm * (x_i - z_m)||^2

	return res;
}

void Cluster::clearAllMembers() {
    h_members.clear();
}
void Cluster::assignMember(unsigned int id) {
	h_members.addMember(id);
}
void Cluster::update() {
	updateCenter();


    //DiceUniformInt ch(1, 100);

    //yjf.test
    /*
    if (this->getOrthIpIndex()< 0.015 && vUpdateMethod == VupdateMethod::m3_china) {

        std::cout << "&&&&&&&&&&&&&&&&&&&: " << VupdateMethod::tostr(VupdateMethod::m1_korea) << std::endl;
        vUpdateMethod = VupdateMethod::m1_korea;
        int which = ch.next();

        if (which > 50) {
        }
        else {
            std::cout << "&&&&&&&&&&&&&&&&&&&: " << VupdateMethod::tostr(VupdateMethod::m2_us) << std::endl;
            vUpdateMethod = VupdateMethod::m2_us;
        }

    }*/


	//updateW();
	switch(vUpdateMethod) {
	case VupdateMethod::m1_korea:
		updateV();
		break;

	case VupdateMethod::m2_us:
		updateV2();
		break;

	case VupdateMethod::m3_china:
		updateV3();
		break;

	default:
		ASSERT_WITH_MSG(false, "Cluster::update..unknown V matrix update method" << vUpdateMethod);
	}
	

	//make recall index better for batch1 dataset.
    //p: 0.652482, r : 0.639497, F1 : 1.66667e+006, G : 0.61684, ac : 0.85618, NMI : 0.641961,
	updateW();		


    //After reassigning members and
    //updating V,W,
    //we can keep the newest results
    //in history records
    //std::cout << "000000000000000000000000000 before keepStableHistory 00000000" << std::endl;
    historyMembers.keepStableHistory(h_members);

}


void Cluster::updateCenter() {

	unsigned int count = h_members.size();
	Record2 sum(z_m.dim());
    for (const auto& i : h_members) {
		sum += precords->at(i);				//sum (xi), where xi are in this cluster
	}
	sum /= count;
	z_m = sum;		//updating center
}

void Cluster::updateW() {

	// Let  pi = V_txm * (xi - z_m), where xi is in this cluster
	//      fai_j = 2 * [ lambda + sum_i(pi_j^2) ], where j=1..t
	// w_j = (1/fai_j) /[ 1/fai_1 + 1/fai_2 + ... + 1/fai_t]

	Record2 sump(w_t.dim());				//clear automatically
	Record2 diff(z_m.dim());				//xi - z_m, whose dim is same as z_m
    for (const auto& i : h_members) {
		diff = precords->at(i);				//xi, whose dim is same as z_m
		diff -= z_m;						//yjf.STRANGE effective.

		Record2 pi;
        pi = (arma::mat)V_txm * (arma::vec)diff;	//pi, whose dim is t
		
		pi.hadamardProd(pi);		//WARNING: change pi
		sump += pi;					//sum_i_cluster (pi_j^2)
	}

    //
    //yjf.Can't remove lambda, for it may lead to infinity
    //when sim_i_cluster(pi_j^2) is zero.
    //
	//fai_j = 2 * [ lambda + sum_i_cluster (pi_j^2)) ]
	Record2 vlambda(sump.dim(), lambda);
	sump += vlambda;
	//sump *= 2;		//can be removed without affecting the result of w_t

	//sump.print("fai_vet:");


	//w_j = (1/fai_j) /[ 1/fai_1 + 1/fai_2 + ... + 1/fai_t], where j=1..t
	value_type denominator = 0;
	for (const auto& d : sump) {
		denominator += (1 / d);
	}


	unsigned int j = 0;
	unsigned int t = w_t.dim();
	for (; j < t; ++j) {
		w_t[j] = (1 / sump[j]) / denominator;
	}

	//normalizeW();
}
///////////////////////////////////////////////////////////////////////////////////
// update V

void Cluster::compPp_Pm2(GAmatrix& Pp_mxm, GAmatrix& Pm_mxm) const {

    //Make sure Pp_mxm, Pm_mxm is zero filled!
    //Pp_mxm.fill(0);
    //Pm_mxm.fill(0);

    //std::cout << "1......compPp_Pm: " << h_members.size() << std::endl;

    unsigned int m = Pp_mxm.n_rows;

    //Record2 pos, neg;
    GAmatrix sum;
    sum.resize(m,m);
	sum.fill(0);
	GAmatrix tmp;
    tmp.resize(m,m);
	
    //yjf.High dimension
    DiceUniformInt ch(1, 100);

    //unsigned int many = h_members.size();

    for (const auto& id : h_members) {

        if (m > 500 && ch.next() > 30 )   //yjf.High Dimension
            continue;

        const Record2& p = precords->at(id);
        Record2 s((arma::vec)p - (arma::vec) z_m);

		tmp.assignByOutProd(s);
		
		sum += tmp;
	
        //std::cout << "L......compPp_Pm: " << id << std::endl;
    }

    sum.splitPosNeg(Pp_mxm, Pm_mxm);
	
    //double f = arma::norm((arma::mat)Pp_mxm, "fro");
    //std::cout << "\t updating Pp_mxm: " << f << std::endl;
    //f = arma::norm((arma::mat)Pm_mxm, "fro");
    //std::cout << "\t updating Pm_mxm: " << f << std::endl;

    //std::cout << "end......compPp_Pm" << std::endl;

}

void Cluster::compPp_Pm(GAmatrix& Pp_mxm, GAmatrix& Pm_mxm) const {

    //Make sure Pp_mxm, Pm_mxm is zero filled!
    Pp_mxm.fill(0);
    Pm_mxm.fill(0);

    //std::cout << "1......compPp_Pm: " << h_members.size() << std::endl;

    Record2 pos, neg;
    GAmatrix tmp;

    for (const auto& id : h_members) {
		const Record2& p = precords->at(id);
        Record2 s((arma::vec)p - (arma::vec) z_m);
        s.split(pos, neg);
                
       // std::cout << "2......compPp_Pm" << std::endl;


        //tmp = ((arma::vec)pos) * pos.t();		//xi xi^T + z z^T
        tmp.assignByOutProd(pos);

        Pp_mxm += tmp;
        //tmp = ((arma::vec) neg) * neg.t();
        tmp.assignByOutProd(neg);
        Pp_mxm += tmp;

        //std::cout << "3......compPp_Pm" << std::endl;

        //GAmatrix xz;
        //tmp = (arma::vec)pos * neg.t();		//xi z_m^T
        tmp.assignByOutProd(pos, neg);


        //std::cout << "4......compPp_Pm" << std::endl;

        //GAmatrix pm;
        //pm = (arma::mat) xz + xz.t();		//sum_i ( xi * z_m^T + z_m * xi^T )
        Pm_mxm += (arma::mat) tmp;
        Pm_mxm += tmp.t();

       // std::cout << "L......compPp_Pm: " << id << std::endl;
	}

    //double f = arma::norm((arma::mat)Pp_mxm, "fro");
    //std::cout << "\t updating Pp_mxm: " << f << std::endl;
    //f = arma::norm((arma::mat)Pm_mxm, "fro");
    //std::cout << "\t updating Pm_mxm: " << f << std::endl;

    //std::cout << "end......compPp_Pm" << std::endl;

}
void Cluster::compQ_txt(GAmatrix& Q_txt) const {

	Record2 q(w_t);
	q.hadamardProd(q);			//Q = diag(w_t)^T diag(w_t)
	Q_txt = (arma::mat) arma::diagmat((arma::vec)q);

    //double f = arma::norm((arma::mat)Q_txt, "fro");
    //std::cout << "\t\t updating Q_txt: " << f << std::endl;
}

void Cluster::updateV3() {

    // V_txm will not be updated if
    // it is very closed to orthogonal and
    // sparse in each row

    /*
    if (V_txm.avgRowInnerProd() < Constant::orthIndexSmallest
        && V_txm.minNumNonZerosPerRow() <= 2 ) {
        return;
    }*/

    std::cout << "......Cluster::update_china" << std::endl;

    
	GAmatrix Pm_mxm(z_m.dim());		//auto-clear to zero
	GAmatrix Pp_mxm(z_m.dim());		//auto-clear to zero
    //compPp_Pm(Pp_mxm, Pm_mxm);
    compPp_Pm2(Pp_mxm, Pm_mxm);


	GAmatrix Q_txt;
	compQ_txt(Q_txt);				//Q = diag(w_t)^T diag(w_t)
	
	GAmatrix QV_txm;

    //std::cout << "1......Cluster::updateV3" << std::endl;

    QV_txm = (arma::mat) Q_txt * (arma::mat)V_txm;

    //std::cout << "2......Cluster::updateV3" << std::endl;

        ////////////////////////////////////////
//        double xbeta = 1 - V_txm.rowOrthIpIndex();
//        if (xbeta>= 0.5) {
//            xbeta *= betaV * 0.1;
//        }
//        else {
//            DiceNormalDouble dice(1, 0.2);
//            xbeta = dice.next();
//        }
        
        //double xbeta = beta;
	GAmatrix U_txm;
    U_txm = (arma::mat)QV_txm * (arma::mat) Pm_mxm;
    U_txm += betaV * (arma::mat)V_txm;

    //std::cout << "3......Cluster::updateV3" << std::endl;

	GAmatrix D_txm;
    D_txm = (arma::mat)QV_txm * (arma::mat) Pp_mxm;

    //D_txm += betaV * (arma::mat) V_txm * V_txm.t() * (arma::mat) V_txm;

    GAmatrix V3_txm;
    V3_txm = betaV * (arma::mat) V_txm * V_txm.t();
    V3_txm *= (arma::mat) V_txm;

    D_txm += V3_txm;


    //std::cout << "4......Cluster::updateV3" << std::endl;

    //double f = arma::norm((arma::mat)U_txm, "fro");
    //std::cout << "------------ U_txm: " << f << std::endl;

    //f = arma::norm((arma::mat)D_txm, "fro");
    //std::cout << "------------ D_txm: " << f << std::endl;


	U_txm.hadamardDivd(D_txm);

    //std::cout << "5......Cluster::updateV3" << std::endl;

    //f = arma::norm((arma::mat)U_txm, "fro");
    //std::cout << "updating factor: " << f << std::endl;


    //
    // updating V_txm
    //
	V_txm.hadamardProd(U_txm);

   // std::cout << "6......Cluster::updateV3" << std::endl;
	

	//
	//normalize each row of V to length 1.
	//
	normalizeV();


    //std::cout << "======> Cluster::updateV3" << std::endl;

}

void Cluster::updateV2() {

    std::cout << "******* Cluster::update_usa" << std::endl;


	GAmatrix Pm_mxm(z_m.dim());		//auto-clear to zero
	GAmatrix Pp_mxm(z_m.dim());		//auto-clear to zero

    compPp_Pm2(Pp_mxm, Pm_mxm);

	GAmatrix Q_txt;
	compQ_txt(Q_txt);				//Q = diag(w_t)^T diag(w_t)

	GAmatrix QV_txm;
        QV_txm = (arma::mat)Q_txt * (arma::mat)V_txm;
	GAmatrix U_txm;
        U_txm =  (arma::mat)QV_txm * (arma::mat)Pm_mxm;

	GAmatrix S_mxm;
        S_mxm = (arma::mat)Pp_mxm + ((arma::mat)Pm_mxm - (arma::mat)Pp_mxm) * V_txm.t() * (arma::mat)V_txm;
	GAmatrix D_txm;
        D_txm = (arma::mat)QV_txm * (arma::mat) S_mxm;


	double f = arma::norm((arma::mat)U_txm, "fro");
	std::cout << "------------ U_txm: " << f << std::endl;

	f = arma::norm((arma::mat)D_txm, "fro");
	std::cout << "------------ D_txm: " << f << std::endl;



	U_txm.hadamardDivd(D_txm);

	f = arma::norm((arma::mat)U_txm, "fro");
	std::cout << "updating factor: " << f << std::endl;

	V_txm.hadamardProd(U_txm);


	//
	//normalize each row of V to length 1.
	//
	normalizeV();

}


void Cluster::updateV() {

    std::cout << "###### Cluster::update_korea" << std::endl;

	//Korea

    GAmatrix Pm_mxm(z_m.dim());		//auto-clear to zero
    GAmatrix Pp_mxm(z_m.dim());		//auto-clear to zero
    compPp_Pm2(Pp_mxm, Pm_mxm);

	GAmatrix Q_txt;
	compQ_txt(Q_txt);				//Q = diag(w_t)^T diag(w_t)
	GAmatrix QV_txm;
        QV_txm = (arma::mat) Q_txt * (arma::mat) V_txm;

	
	GAmatrix U_txm;
        U_txm = (arma::mat)QV_txm * Pm_mxm.t() + betaV* (arma::mat)V_txm * (arma::mat) Pp_mxm * V_txm.t() * (arma::mat) QV_txm;

	GAmatrix D_txm;
        D_txm = (arma::mat)QV_txm * Pp_mxm.t() + betaV* (arma::mat)V_txm * (arma::mat) Pm_mxm * V_txm.t() * (arma::mat) QV_txm;


	double f = arma::norm((arma::mat)U_txm, "fro");
	std::cout << "------------ U_txm: " << f << std::endl;

	f = arma::norm((arma::mat)D_txm, "fro");
	std::cout << "------------ D_txm: " << f << std::endl;


	U_txm.hadamardDivd(D_txm);

	f = arma::norm((arma::mat)U_txm, "fro");
	std::cout << "updating factor: " << f << std::endl;

	V_txm.hadamardProd(U_txm);
		
	
	//
	//normalize each row of V to length 1.
	//
	normalizeV();

}



////////////////////////////////////////////////////////
//results
const ClusterMembership& Cluster::getMembers() const {
	return h_members;
}

const Record2& Cluster::getCenter() const {
	return z_m;
}
const GAmatrix& Cluster::getV() const {
	return V_txm;
}
const Record2& Cluster::getW() const {
	return w_t;
}
Cluster::value_type Cluster::getLambda() const {
	return lambda;
}
Cluster::value_type Cluster::getBeta() const {
	return betaV;
}
InitialMethod::Name Cluster::getInitialMethod() const {
	return initialMethod;
}
unsigned int Cluster::getNumGroups() const {
    return w_t.size();
}


///////////////////////////////////////////////////////////////////

Cluster::value_type Cluster::getWsparseIndex() const {
    return w_t.innerProduct(w_t);
}
Cluster::value_type Cluster::getOrthIpIndex() const {
    return V_txm.avgRowInnerProd();
}

Cluster::value_type Cluster::getClusteringError() const {

    //sum_i( ||diag(w_t) * V_txm * (x_i - z_m)||^2 )
    double sum = 0;

    Record2 diff;
    double res;
    for (const auto& id : h_members) {
        diff = precords->at(id);
        diff -= z_m;			//x_id - z_m

        Record2 vect;
        vect = (arma::mat)V_txm * (arma::vec) diff;		// V_txm * (x_id - z_m)
        vect.hadamardProd(w_t);     //diag(w_t) * V_txm * (x_id - z_m)

        res = vect.innerProduct(vect);  //||diag(w_t) * V_txm * (x_i - z_m)||^2


        //checking error
        if (res > Constant::nearPosInfi) {
            std::cout << "..........objectValue for record" << id << ", is too large" << std::endl;
            std::cout << DataNormalizeMethod::tostr(precords->getUsedNormalizeMethod()) << std::endl;
            std::cout << h_members.getLabel() << "id: " << id << ", ob=" << res << std::endl;
            vect.print("vect:");
            V_txm.print("V_txm");

            char c;
            std::cin >> c;

        } else if (res < 0) {
            std::cout << this->h_members.getLabel() << " ";
            std::cout << "-------------- ov for id=" << id << " is " << res << std::endl;
            V_txm.print("V_txm");
            precords->at(id).print("record:");
            z_m.print("z_m");
            diff.print("x_id -zm");

            vect.print("before 2-norm square");
            char c;
            std::cin >> c;

        }
        else {
            sum += res;
        }

    }//for

    //yjf.debug.
    if (sum > Constant::nearPosInfi) {
        std::cout << h_members.getLabel() << ": cluErr is very big: "  << sum << std::endl;
        w_t.print("w_t:");
        V_txm.print("V_txm: ");

        sum = Constant::nearPosInfi;    //yjf.WARNING: to large
    }
    return sum;
}


Cluster::value_type Cluster::getObjectValue() const {
	
    // sum_i( ||diag(w_t) * V_txm * (x_i - z_m)||^2 ) + lambda * ||w_t||^2

    double sum = getClusteringError();
    sum += lambda * getWsparseIndex();		//lambda * ||w_t||^2
    
    double f = V_txm.avgRowInnerProd();
    double orth = (betaV/2) * (f * f);
	
    double vf = V_txm.fnorm();

    std::cout << "orthPen: " << orth << ", Vfnorm: " << vf << ", avgRowIp: " << V_txm.avgRowInnerProd() << std::endl;

    sum += orth;
	
    //check
    if (sum > Constant::nearPosInfi) {

        std::cout << "object value is too big: " << sum << std::endl;
        V_txm.print("V_txm:");
        std::cout << h_members.getLabel() << "................object value " << sum << std::endl;

        char c;
        std::cin >> c;
    }

    return sum;
}


/////////////////////////////////////////////////////////////////////////////////////////////
//initial
void Cluster::init1_random(std::vector<Record2*>& existingCenters, InitialMethod::Name n) {

	initialCenter(existingCenters);		//z_m
	this->initialMethod = n;			//keep initialMethod for gathering results.

	switch (n)
	{

	case InitialMethod::V_method1:
		initialV_method1();
		break;
	case InitialMethod::V_method2:
		initialV_method2();
		break;
	case InitialMethod::V_method3:
		initialV_method3();
		break;

	case InitialMethod::V_method4:
		initialV_method4();
		break;

	default:
        ASSERT_WITH_MSG(false, "Cluster::init1_random..unknown initial method: " << n);
		break;
	}

	initialW();							//w_t
}



void Cluster::setCenterFromExisted(std::vector<Record2*>& existingCenters) {
	
	//choose one record uniformally.
	DiceUniformInt dice(0, precords->size() - 1);
		
	value_type distFarest = 0;
	unsigned int idFarest = 0;
    unsigned int candidateNum = precords->size();
	for (unsigned int i = 0; i < candidateNum; i++) {

        unsigned int id = dice.next();		//guess one

		//compute the nearest distance to existing centers
        const Record2& test = precords->at(id);
		value_type distNearest = 1000000.0;
		for (const auto& j : existingCenters) {
			value_type d = j->squareDist(test);
			if (d < distNearest)
				distNearest = d;
		}

		//keep the (the nearest distance, id)
		if (distFarest < distNearest) {
			distFarest = distNearest;
            idFarest = i;
		}
	}

	//in candidatenum of guessed records,
	//choose the one farest from the existing centers
	//as the center z_m of this cluster
	z_m = precords->at(idFarest);
	
	//pass the pointer of the z_m
	//to next cluster for using it in choosing its center
	existingCenters.push_back(&z_m);
}

void Cluster::setCenterByAverage(std::vector<Record2*>& outputCenter) {

	//choose one record uniformally.
	DiceUniformInt dice(0, precords->size() - 1);

	Record2 sum(z_m.dim());   //auto clear to zeros
    unsigned int candidateNum ul= precords->size();
	for (unsigned int i = 0; i < candidateNum; i++) {

        unsigned int id = dice.next();		//guess one
        sum += precords->at(id);
		
	}

	sum /= candidateNum;			//average value
	z_m = sum;


	//pass the pointer of the z_m
	//to next cluster for using it in choosing its center
	outputCenter.push_back(&z_m);
}

void Cluster::setCenterByRandom() {

    static DiceUniformInt dice(0,precords->size()-1);

    unsigned int choose = dice.next();
    z_m = precords->at(choose);

}

void Cluster::initialCenter(std::vector<Record2*>& existingCenters) {

    //setCenterByRandom();

    //setCenterByAverage(existingCenters);  //performance is bad


	if (existingCenters.size() == 0){
		setCenterByAverage(existingCenters);
	}
	else {
		setCenterFromExisted(existingCenters);
    }

}

void Cluster::initialW() {

	//1'w = 1, and wi >=0 for i=1..t
    static DiceNormalDouble dice(1, 0.01);			//best variance at = 0.5
	
	double gs = dice.next();

	for (auto& i : w_t) {
		for (; gs <= Constant::nearZero; gs = dice.next());
		i = gs;
		gs = dice.next();
	}

	normalizeW();
}

////////////////////////////////////////////////////////////////////////

void Cluster::normalizeV() {

	//set lower and upper bound for very big or very small elements
	unsigned int i = 0;
	unsigned int i_size = V_txm.n_rows;
	//unsigned int j;
	//unsigned int j_size = V_txm.n_cols;
	//for (; i < i_size; ++i) {
	//	for (j = 0; j < j_size; ++j) {
	//		if (V_txm(i, j) > Constant::nearPosInfi)
	//			V_txm(i, j) = Constant::nearPosInfi;
	//		else if (V_txm(i, j) < Constant::nearZero)
	//			; // V_txm(i, j) = 0;					//do nothing
	//		else if (V_txm(i, j) < 0)  {
	//			//yjf.debug.
	//			std::string str = h_members.getLabel();
	//			str += "================ V_txm..before normalz, found negative:";
	//			V_txm.print(str);
	//		}
	//	}//for
	//}


	//for V>=0, V V^T = I
    //we make the norm-2 of each row of V equal to 1
	for (i=0; i < i_size; ++i) {
		arma::rowvec rw = V_txm.row(i);
		double length = arma::norm(rw, 2);
		V_txm.row(i) /= length;				//must be written as V_txm.row(i) /=, not rw /=
	}

}

void Cluster::normalizeW() {

	double sum = 0;
	for (auto& i : w_t) {
		sum += i;
	}
	w_t /= sum;
}

void Cluster::initialV_method4() {

    static DiceNormalDouble dice(1, 0.01);

	unsigned int rows = V_txm.n_rows;
	unsigned int columns = V_txm.n_cols;
	unsigned int i;
	unsigned int j;
	double gs = dice.next();
	for (i=0; i < rows; ++i) {
		for (j=0; j < columns; ++j) {
			for (; gs <= Constant::nearZero; gs = dice.next());		//element must be positive
			V_txm(i, j) = gs;
		}

	}//for

	normalizeV();

	//yjf.debug
	std::cout << h_members.getLabel() << ", Vmethod4" << std::endl;
	//V_txm.print("V_txm");
}

void Cluster::initialV_method3() {

	//clear V_txm
	V_txm.fill(1);


	//unsigned int rows = V_txm.n_rows;
	//unsigned int columns = V_txm.n_cols;
	//unsigned int i = 0;
	//for (; i < rows; ++i) {
	//	unsigned int j = i;
	//	for (; j < columns; j += rows) {
	//		V_txm(i, j) = 1;
	//	}

	//}//for

	normalizeV();

	//yjf.debug
	std::cout << h_members.getLabel() << ", Vmethod3" << std::endl;
	//V_txm.print("V_txm");
}


void Cluster::initialV_method1() {

	//clear V_txm
	V_txm.fill(0);


	unsigned int rows = V_txm.n_rows;
	unsigned int columns = V_txm.n_cols;
	unsigned int i = 0;
	for (; i < rows; ++i) {
		unsigned int j = i;
		for (; j < columns; j += rows) {
			V_txm(i, j) = 1;
		}

	}//for

	normalizeV();

	//yjf.debug
	//std::cout << "Vmethod1:";
	//std::cout << h_members.getLabel() << std::endl;
	//V_txm.print("V_txm");

}

void Cluster::initialRow(unsigned int row, Record2& occupiedColumns, int toNum, int& remainedCount) {

    //choose column index uniformally
	DiceUniformInt dice(0, V_txm.n_cols - 1);

	if (toNum <= 0) {
		toNum = 1;			//at least one column will be allocated for the row
	}

	int count = 0;
	for (; count < toNum; ) {
		int j = dice.next();
		if (occupiedColumns[j] == 0 || remainedCount <= 0) {
			++count;
			occupiedColumns[j] = 1;
			V_txm(row, j) = 1;
		}
	}//for all random column to choose 

	remainedCount -= toNum;			//minus already allocated '1's

}

void Cluster::initialV_method2(unsigned int rowAllocStdev/*=1*/) {
	
	//clear V_txm
	V_txm.fill(0);

	//Average number of '1' will be allocated for each row of V
	int columns = V_txm.n_cols;
	int averCols = columns / V_txm.n_rows;
	if (averCols == 0) averCols = 1;		//average is at least 1.

	//how many columns are choosed by earch row, is of normal distribution.
	DiceNormalDouble dice(averCols, rowAllocStdev);

	//allocate '1' of looping of each row
	Record2 allocated(columns);					//keep '1' already set by previous rows.
	int freeRemained = columns;					//keep how many free columns to be allocated
	
	unsigned int i = 0;
	unsigned int sz = V_txm.n_rows;
	for (; i < sz; ++i) {
		initialRow(i, allocated, (int)dice.next(), freeRemained);
	}//for each row

	normalizeV();

	//yjf.debug
	std::cout << h_members.getLabel() << ", Vmethod3" <<  std::endl;
	
}
///////////////////////////////////////////////////////////////////////////////
bool Cluster::isConvergent() {

    //yjf.TODO. add Averge Row InnerProd

    return historyMembers.isConvergent();

    /*
    Decimal10 wsparse = this->getWsparseIndex();
    if (abs(wsparse - oldWsparseIndex)/wsparse > 0.001) {
        std::cout << "///////////////////// wsparse ratio > 0.001" << std::endl;
        oldWsparseIndex = wsparse;
        return false;
    }
    oldWsparseIndex = wsparse; */

    /* always failed.
     *
    Decimal10 orth = this->getOrthIpIndex();
    if (abs(orth - oldOrthIpIndex)/orth > 0.01) {
        std::cout << "//--////--///--///-/ orthIp ratio > 0.01" << std::endl;

        oldOrthIpIndex = orth;
        return false;
    }
    oldOrthIpIndex = orth;*/

    /*
    Decimal10 orth = this->getOrthIpIndex();

    if (orth > Constant::orthIndexSmallest) {

       return false;
    }

    std::cout << "========================>" << h_members.getLabel();
    std::cout << " is convergent!" << std::endl;
    return true; */
    
    //0.0750795
    //0.00282903
}
