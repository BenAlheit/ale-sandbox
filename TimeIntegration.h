//
// Created by alhei on 2022/08/20.
//

#ifndef ALE_TIMEINTEGRATION_H
#define ALE_TIMEINTEGRATION_H

using namespace std;

class TimeIntegration {
public:
    virtual Vector<double> integrate(const Vector<double> &y_t_start,
                                     const function<Vector<double>(const double &, const Vector<double> &)> &dy) = 0;

    void set_t_start(const double &new_t_start) { t_start = new_t_start; };

    void set_t_end(const double &new_t_end) { t_end = new_t_end; };

    void set_n_steps(const unsigned int &new_n_steps) { n_steps = new_n_steps; };

protected:
    double t_start, t_end, t_c_start, t_c_end;
    unsigned int n_steps;
};

class RungeKuttaTableau {
public:
    vector<double> c;
    vector<vector<double>> a, b;
private:
};



class ThirdOrderStrongStability : public RungeKuttaTableau {
public:
    ThirdOrderStrongStability() {
        this->c = vector<double>({0., 1., 1. / 2.});

        this->b = vector<vector<double>>(
                {vector<double>({1. / 6., 1. / 6., 2. / 3.}),
                 vector<double>({1. / 6., 1. / 6., 2. / 3.})});

        this->a = vector<vector<double>>({vector<double>({0.}),
                                          vector<double>({1., 0.}),
                                          vector<double>({1. / 4., 1. / 4.})});
    };
private:
};

class RK4 : public RungeKuttaTableau {
public:
    RK4() {
        this->c = vector<double>({0., 1. / 2., 1. / 2., 1.});

        this->b = vector<vector<double>>(
                {vector<double>({1. / 6., 1. / 3., 1. / 3., 1. / 6.}),
                 vector<double>({1. / 6., 1. / 3., 1. / 3., 1. / 6.})});

        this->a = vector<vector<double>>({vector<double>({0.}),
                                          vector<double>({1. / 2, 0.}),
                                          vector<double>({0., 1. / 2., 0.}),
                                          vector<double>({0., 0., 1.})});
    };
private:
};


class R01 : public RungeKuttaTableau {
public:
    R01() {
        this->c = vector<double>({0., 1.});

        this->b = vector<vector<double>>(
                {vector<double>({0.5, 0.5}),
                 vector<double>({1., 0.})});

        this->a = vector<vector<double>>({vector<double>({0.}),
                                          vector<double>({1.})
                                         });
    };
private:
};

class RKF : public RungeKuttaTableau {
public:
    RKF() {
        this->c = vector<double>({0., 1. / 4., 3. / 8., 12. / 13., 1., 1. / 2.});

        this->b = vector<vector<double>>(
                {vector<double>({16. / 135., 0, 6656. / 12825., 28561. / 56430., -9. / 50., 2. / 55.}),
                 vector<double>({25. / 216., 0, 1408. / 2565., 2197. / 4104., -1. / 5., 0})});

        this->a = vector<vector<double>>({vector<double>({0.}),
                                          vector<double>({1. / 4.}),
                                          vector<double>({3. / 32., 9. / 32.}),
                                          vector<double>({1932. / 2197., -7200. / 2197., 7296. / 2197.}),
                                          vector<double>({439. / 216., -8., 3680. / 513., -845. / 4104.}),
                                          vector<double>({-8. / 27., 2., -3544. / 2565., 1859. / 4104., -11. / 40.})
                                         });
    };
private:
};

class RK89 : public RungeKuttaTableau {
public:
    RK89() {

        /*
         * Values below are from https://doi.org/10.1016/S0168-9274(01)00025-3
         * */

        this->c = vector<double>(16, 0.);

        this->b = vector<vector<double>>(2, vector<double>(16, 0.));

        this->a = vector<vector<double>>(16, vector<double>(16, 0.));

        this->c.at(0) = 0.;
        this->c.at(1) = 0.02173913043478260869565217391304347;
        this->c.at(2) = 0.09629581047800066670113001679819925;
        this->c.at(3) = 0.14444371571700100005169502519729888;
        this->c.at(4) = 0.52205882352941176470588235294117647;
        this->c.at(5) = 0.22842443612863469578031459099794265;
        this->c.at(6) = 0.54360353589933733219171338103002937;
        this->c.at(7) = 0.64335664335664335664335664335664335;
        this->c.at(8) = 0.48251748251748251748251748251748251;
        this->c.at(9) = 0.06818181818181818181818181818181818;
        this->c.at(10) = 0.25060827250608272506082725060827250;
        this->c.at(11) = 0.66736715965600568968278165443304378;
        this->c.at(12) = 0.85507246376811594202898550724637681;
        this->c.at(13) = 0.89795918367346938775510204081632653;
        this->c.at(14) = 1.0;
        this->c.at(15) = 1.0;

        this->b.at(0).at(0) = 0.01490902081978461022483617102382552;
        this->b.at(0).at(7) = -0.20408044692054151258349120934134791;
        this->b.at(0).at(8) = 0.22901438600570447264772469337066476;
        this->b.at(0).at(9) = 0.12800558251147375669208211573729202;
        this->b.at(0).at(10) = 0.22380626846054143649770066956485937;
        this->b.at(0).at(11) = 0.39553165293700054420552389156421651;
        this->b.at(0).at(12) = 0.05416646758806981196568364538360743;
        this->b.at(0).at(13) = 0.12691439652445903685643385312168037;
        this->b.at(0).at(14) = -0.00052539244262118876455834655383035;
        this->b.at(0).at(15) = 1.0 / 31.0;

        this->b.at(1).at(0) = 0.00653047880643482012034413441159249;
        this->b.at(1).at(7) = -2.31471038197461347517552506241529830;
        this->b.at(1).at(8) = 0.43528227238866280799530900822377013;
        this->b.at(1).at(9) = 0.14907947287101933118545845390618763;
        this->b.at(1).at(10) = 0.17905535442235532311850533252768020;
        this->b.at(1).at(11) = 2.53400872222767706921176214508820825;
        this->b.at(1).at(12) = -0.55430437423209112896721332268159015;
        this->b.at(1).at(13) = 0.56924788787870083224213506297615260;
        this->b.at(1).at(14) = -0.03644749690427461198884026816573513;
        this->b.at(1).at(15) = 1.0 / 31.0;

        this->a.at(1).at(0) = 1. / 46.;
        this->a.at(2).at(0) = -0.11698050118114486205818241524969622;
        this->a.at(2).at(1) = 0.21327631165914552875931243204789548;
        this->a.at(3).at(0) = 0.03611092892925025001292375629932472;
        this->a.at(3).at(2) = 0.10833278678775075003877126889797416;
        this->a.at(4).at(0) = 1.57329743908138605107331820072051125;
        this->a.at(4).at(2) = -5.98400943754042002888532938159655553;
        this->a.at(4).at(3) = 4.93277082198844574251789353381722074;
        this->a.at(5).at(0) = 0.05052046351120380909008334360006234;
        this->a.at(5).at(3) = 0.17686653884807108146683657390397612;
        this->a.at(5).at(4) = 0.00103743376935980522339467349390418;
        this->a.at(6).at(0) = 0.10543148021953768958529340893598138;
        this->a.at(6).at(3) = -0.16042415162569842979496486916719383;
        this->a.at(6).at(4) = 0.11643956912829316045688724281285250;
        this->a.at(6).at(5) = 0.48215663817720491194449759844838932;
        this->a.at(7).at(0) = 0.07148407148407148407148407148407148;
        this->a.at(7).at(5) = 0.32971116090443908023196389566296464;
        this->a.at(7).at(6) = 0.24216141096813279233990867620960722;
        this->a.at(8).at(0) = 0.07162368881118881118881118881118881;
        this->a.at(8).at(5) = 0.32859867301674234161492268975519694;
        this->a.at(8).at(6) = 0.11622213117906185418927311444060725;
        this->a.at(8).at(7) = -0.03392701048951048951048951048951048;
        this->a.at(9).at(0) = 0.04861540768024729180628870095388582;
        this->a.at(9).at(5) = 0.03998502200331629058445317782406268;
        this->a.at(9).at(6) = 0.10715724786209388876739304914053506;
        this->a.at(9).at(7) = -0.02177735985419485163815426357369818;
        this->a.at(9).at(8) = -0.10579849950964443770179884616296721;
        this->a.at(10).at(0) = -0.02540141041535143673515871979014924;
        this->a.at(10).at(5) = 1.0 / 30.0;
        this->a.at(10).at(6) = -0.16404854760069182073503553020238782;
        this->a.at(10).at(7) = 0.03410548898794737788891414566528526;
        this->a.at(10).at(8) = 0.15836825014108792658008718465091487;
        this->a.at(10).at(9) = 0.21425115805975734472868683695127609;
        this->a.at(11).at(0) = 0.00584833331460742801095934302256470;
        this->a.at(11).at(5) = -0.53954170547283522916525526480339109;
        this->a.at(11).at(6) = 0.20128430845560909506500331018201158;
        this->a.at(11).at(7) = 0.04347222773254789483240207937678906;
        this->a.at(11).at(8) = -0.00402998571475307250775349983910179;
        this->a.at(11).at(9) = 0.16541535721570612771420482097898952;
        this->a.at(11).at(10) = 0.79491862412512344573322086551518180;
        this->a.at(12).at(0) = -0.39964965968794892497157706711861448;
        this->a.at(12).at(5) = -3.79096577568393158554742638116249372;
        this->a.at(12).at(6) = -0.40349325653530103387515807815498044;
        this->a.at(12).at(7) = -2.82463879530435263378049668286220715;
        this->a.at(12).at(8) = 1.04226892772185985533374283289821416;
        this->a.at(12).at(9) = 1.12510956420436603974237036536924078;
        this->a.at(12).at(10) = 3.32746188718986816186934832571938138;
        this->a.at(12).at(11) = 2.77897957186355606325818219255783627;
        this->a.at(13).at(0) = 0.39545306350085237157098218205756922;
        this->a.at(13).at(5) = 5.82534730759650564865380791881446903;
        this->a.at(13).at(6) = -0.36527452339161313311889856846974452;
        this->a.at(13).at(7) = 1.18860324058346533283780076203192232;
        this->a.at(13).at(8) = 0.57970467638357921347110271762687972;
        this->a.at(13).at(9) = -0.86824862589087693262676988867897834;
        this->a.at(13).at(10) = -5.20227677296454721392873650976792184;
        this->a.at(13).at(11) = -0.79895541420753382543211121058675915;
        this->a.at(13).at(12) = 0.14360623206363792632792463778889008;
        this->a.at(14).at(0) = 8.49173149061346398013352206978380938;
        this->a.at(14).at(5) = 86.32213734729036800877634194386790750;
        this->a.at(14).at(6) = 1.02560575501091662034511526187393241;
        this->a.at(14).at(7) = 85.77427969817339941806831550695235092;
        this->a.at(14).at(8) = -13.98699305104110611795532466113248067;
        this->a.at(14).at(9) = -20.71537405501426352265946477613161883;
        this->a.at(14).at(10) = -72.16597156619946800281180102605140463;
        this->a.at(14).at(11) = -76.71211139107806345587696023064419687;
        this->a.at(14).at(12) = 4.22319427707298828839851258893735507;
        this->a.at(14).at(13) = -1.25649850482823521641825667745565428;
        this->a.at(15).at(0) = -0.42892119881959353241190195318730008;
        this->a.at(15).at(5) = -9.16865700950084689999297912545025359;
        this->a.at(15).at(6) = 1.08317616770620939241547721530003920;
        this->a.at(15).at(7) = -1.23501525358323653198215832293981810;
        this->a.at(15).at(8) = -1.21438272617593906232943856422371019;
        this->a.at(15).at(9) = 1.37226168507232166621351243731869914;
        this->a.at(15).at(10) = 9.15723239697162418155377135344394113;
        this->a.at(15).at(11) = 1.30616301842220047563298585480401671;
        this->a.at(15).at(12) = -0.25285618808937955976690569433069974;
        this->a.at(15).at(13) = 0.38099910799663987066763679926508552;
    };
private:
};

class RungeKutta : public TimeIntegration {
public:

    RungeKutta() {
        this->n_steps = 1;
//        this->tableau = new RK89();
        this->tableau = new RKF();
//        this->tableau = new ThirdOrderStrongStability();
//        this->tableau = new RK4();
    };

    Vector<double> integrate(const Vector<double> &y_t_start,
                             const function<Vector<double>(const double &, const Vector<double> &)> &dy) override;

private:

    RungeKuttaTableau *tableau;

    Vector<double> integrate_step(const Vector<double> &y_t_start,
                                  const function<Vector<double>(const double &, const Vector<double> &)> &dy);
};


Vector<double> RungeKutta::integrate(const Vector<double> &y_t_start,
                                     const function<Vector<double>(const double &, const Vector<double> &)> &dy) {
    Vector<double> y_n1(y_t_start.size()), y_n(y_t_start.size()), dy_n(y_t_start.size());
    y_n = y_t_start;
    vector<unsigned int> steps(this->n_steps);
    iota(steps.begin(), steps.end(), 0);
    double step_dt = (this->t_end - this->t_start) / this->n_steps;
    this->t_c_start = this->t_start;
    for (const auto &n: steps) {
        this->t_c_end = this->t_c_start + step_dt;
        y_n1 = integrate_step(y_n, dy);
        y_n = y_n1;
        this->t_c_start += step_dt;
    }

    return y_n1;
}

Vector<double> RungeKutta::integrate_step(const Vector<double> &y_t_start,
                                          const function<Vector<double>(const double &, const Vector<double> &)> &dy) {
    unsigned int n_dofs = y_t_start.size();
    Vector<double> out(n_dofs);
    Vector<double> error_approx(n_dofs);
    Vector<double> working(n_dofs);
    Vector<double> dy_i(n_dofs);
    unsigned int s = tableau->c.size();
    vector<unsigned int> range(s);
//    vector<unsigned int> range_b(tableau->b.at(0).size());
    vector<Vector<double>> ks(s, Vector<double>(n_dofs));
    iota(range.begin(), range.end(), 0);
    double dt = this->t_c_end - this->t_c_start;

    for (const auto &i: range) {
        working = y_t_start;
        for (unsigned int j = 0; j < i; j++) {
            working.add(dt * tableau->a.at(i).at(j), ks.at(j));
        }
        dy_i = dy(t_c_start + tableau->c.at(i) * dt, working);
        ks.at(i) = dy_i;
    }
    out = y_t_start;
    for (const auto &i: range)
        out.add(dt * tableau->b.at(0).at(i), ks.at(i));

    error_approx = 0;
    for (const auto &i: range)
        error_approx.add(dt * (tableau->b.at(0).at(i) - tableau->b.at(1).at(i)), ks.at(i));

    if (error_approx.l2_norm() / out.l2_norm() > 0.0001) {
        cout << "Large error. L2 norm ratio: " << error_approx.l2_norm() / out.l2_norm() << endl;
        cout << "Error vector: " << endl;
        error_approx.print(cout);
        cout << "Solution vector: " << endl;
        out.print(cout);
        cout << "Flow error %: " << 100 * error_approx[9] / out[9] << endl;
    }

    return out;
}


#endif //ALE_TIMEINTEGRATION_H
