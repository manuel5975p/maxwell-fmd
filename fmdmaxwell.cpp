#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <utility>
#include <tuple>
#include <any>
#include <SFML/Window.hpp>
#include <sstream>
#include <SFML/Graphics.hpp>
constexpr double c = 299792458;
constexpr double mu_0 = 4 * M_PI * 1e-7;
constexpr double eps_0 = 1.0 / (mu_0 * c * c);
constexpr size_t border_thickness = 64;
constexpr double border_intensity = 1.0 / 16.0;
template<typename _scalar>
struct yee_grid2d{
    using scalar = _scalar;
    using array =  Eigen::Array<scalar, Eigen::Dynamic, Eigen::Dynamic>;//matrix_zero_outside<scalar, Dynamic, Dynamic>;
    using vector =  Eigen::Matrix<scalar, Eigen::Dynamic, 1>;
    using vector2 = Eigen::Matrix<scalar, 2, 1>;
    array e_x;
    array e_y;
    array h_z;
    array sigma_e;
    array sigma_h;
    array mu;
    array eps;
    array j_source;
    array m_source;
    size_t m, n;
    array multmask;
    yee_grid2d(size_t _m, size_t _n) : 
    e_x(_m, _n - 1), e_y(_m - 1, _n), h_z(_m - 1, _n - 1), sigma_e(_m, _n), sigma_h(_m - 1, _n - 1), mu(_m - 1, _n - 1),
    eps(_m, _n),j_source(_m - 1, _n - 1), m_source(_m - 1, _n - 1), m(_m), n(_n), multmask(_m, _n)
    {
        assert(_m > 1);
        assert(_n > 1);
        e_x.setZero();
        e_y.setZero();
        h_z.setZero();
        mu.fill(mu_0);
        eps.fill(eps_0);
        sigma_e.setZero();
        sigma_h.fill(0);
        j_source.setZero();
        m_source.setZero();
        multmask.setConstant(std::exp(-border_intensity));
        for(size_t i = 0;i < border_thickness;i++){
            if(i == border_thickness - 1){
                multmask.block(i,i, m - 2 * i, n - 2 * i) = 1;
            }
            else{
                multmask.block(i,i, m - 2 * i, n - 2 * i) *= std::exp(border_intensity / border_thickness);
            }
        }
    }
    scalar rot_E_z(size_t i, size_t j)const{
        return (e_y(i, j + 1) - e_y(i, j)) - (e_x(i + 1, j) - e_x(i, j));
    }
    vector2 rot_H_xy(size_t i, size_t j)const{
        return vector2(h_z(i, j) - h_z(i - 1, j), -(h_z(i, j) - h_z(i, j - 1)));
    }
    size_t realm()const{
        return m - 2 * border_thickness;
    }
    size_t realn()const{
        return n - 2 * border_thickness;
    }
    yee_grid2d<scalar>& operator+=(const yee_grid2d<scalar>& o){
        e_x.noalias() += o.e_x;
        e_y.noalias() += o.e_y;
        h_z.noalias() += o.h_z;
        return *this;
    }
    yee_grid2d<scalar>& plus_alpha(const yee_grid2d<scalar>& o, scalar alpha){
        e_x.noalias() += o.e_x * alpha;
        e_y.noalias() += o.e_y * alpha;
        h_z.noalias() += o.h_z * alpha;
        return *this;
    }
    std::tuple<array, array, array> derivative()const{
        array h_z_deriv(m - 1, n - 1);
        array e_x_deriv(m    , n - 1);
        array e_y_deriv(m - 1, n    );
        scalar dx = 1.0 / (n - 1);
        scalar dy = 1.0 / (m - 1);
        h_z_deriv.setZero();
        e_x_deriv.setZero();
        e_y_deriv.setZero();
        h_z_deriv.block(0, 0, m - 1,n - 1) = 
        -(e_y.block(0,1, m - 1, n - 1) - e_y.block(0,0, m - 1, n - 1)) / dx +
        (e_x.block(1,0, m - 1, n - 1) - e_x.block(0,0, m - 1, n - 1)) / dy;
        
        array rot_H_x(m, n - 1);
        rot_H_x.setZero();
        rot_H_x.block(1, 0, m - 2, n - 1) = h_z.block(1, 0, m - 2, n - 1) - h_z.block(0, 0, m - 2, n - 1);
        #ifdef BOUNDARY_CONDITION_ZERO
        rot_H_x.row(0) = h_z.row(0);
        rot_H_x.row(m - 1) = -h_z.row(m - 2);
        #else
        rot_H_x.row(0).setZero();
        rot_H_x.row(m - 1).setZero();
        #endif

        array rot_H_y(m - 1, n);
        rot_H_y.setZero();
        rot_H_y.block(0, 1, m - 1, n - 2) = -h_z.block(0, 1, m - 1, n - 2) + h_z.block(0, 0, m - 1, n - 2);
        #ifdef BOUNDARY_CONDITION_ZERO
        rot_H_y.col(0) = -h_z.col(0);
        rot_H_y.col(n - 1) = h_z.col(n - 2);
        #else
        rot_H_y.col(0).setZero();
        rot_H_y.col(n - 1).setZero();
        #endif
        e_x_deriv += rot_H_x / dx;
        e_y_deriv += rot_H_y / dy;

        h_z_deriv += sigma_h * h_z;
        e_x_deriv -= (sigma_e.block(0, 1, m, n - 1) + sigma_e.block(0, 0, m, n - 1)) * 0.5 * e_x;
        e_y_deriv -= (sigma_e.block(1, 0, m - 1, n) + sigma_e.block(0, 0, m - 1, n)) * 0.5 * e_y;
        h_z_deriv -= m_source;
        h_z_deriv /= (mu);
        e_x_deriv /= ((eps.block(0, 1, m, n - 1) + eps.block(0, 0, m, n - 1)) * 0.5);
        e_y_deriv /= ((eps.block(1, 0, m - 1, n) + eps.block(0, 0, m - 1, n)) * 0.5);
        
        return std::make_tuple(h_z_deriv, e_x_deriv, e_y_deriv);
    }
};
unsigned char fcolor(double x){
    return std::min(256 * x, 255.0);
}
template<typename T>
auto relu(const T& x){
    return x < 0 ? 0 : x;
}
template<typename scalar>
yee_grid2d<scalar> timestep(const yee_grid2d<scalar>& grid, double h){
    yee_grid2d<scalar> retgrid(grid);

    auto kappa1 = grid.derivative();

    yee_grid2d<scalar> arg_for_kappa_2(grid);
    arg_for_kappa_2.h_z += std::get<0>(kappa1) * (h / 2.0);
    arg_for_kappa_2.e_x += std::get<1>(kappa1) * (h / 2.0);
    arg_for_kappa_2.e_y += std::get<2>(kappa1) * (h / 2.0);

    auto kappa2 = arg_for_kappa_2.derivative();

    yee_grid2d<scalar> arg_for_kappa_3(grid);
    arg_for_kappa_3.h_z -= std::get<0>(kappa1) * h;
    arg_for_kappa_3.e_x -= std::get<1>(kappa1) * h;
    arg_for_kappa_3.e_y -= std::get<2>(kappa1) * h;

    arg_for_kappa_3.h_z += std::get<0>(kappa2) * (2.0 * h);
    arg_for_kappa_3.e_x += std::get<1>(kappa2) * (2.0 * h);
    arg_for_kappa_3.e_y += std::get<2>(kappa2) * (2.0 * h);

    auto kappa3 = arg_for_kappa_3.derivative();
    retgrid.h_z += std::get<0>(kappa1) * h / 6.0;
    retgrid.e_x += std::get<1>(kappa1) * h / 6.0;
    retgrid.e_y += std::get<2>(kappa1) * h / 6.0;
    retgrid.h_z += std::get<0>(kappa2) * 2.0 * h / 3.0;
    retgrid.e_x += std::get<1>(kappa2) * 2.0 * h / 3.0;
    retgrid.e_y += std::get<2>(kappa2) * 2.0 * h / 3.0;
    retgrid.h_z += std::get<0>(kappa3) * h / 6.0;
    retgrid.e_x += std::get<1>(kappa3) * h / 6.0;
    retgrid.e_y += std::get<2>(kappa3) * h / 6.0;
    
    retgrid.h_z *= retgrid.multmask.block(0,0,retgrid.m - 1, retgrid.n - 1);

    return retgrid;
}
using scalar = float;
scalar relu(scalar x){
    return x > 0 ? x : 0;
}
using vector2 = Eigen::Matrix<scalar, 2, 1>;
using vector3 = Eigen::Matrix<scalar, 3, 1>;
using vector6 = Eigen::Matrix<scalar, 6, 1>;
vector6 dfdu(const vector3& E, const vector3& H){
    vector6 ret;
    ret.head(3) = 2 * E;
    ret.tail(3) = 2 * H;
    return ret;
}
scalar smallf(const vector3& E, const vector3& H){
    return E.dot(E) + H.dot(H);
}
std::string tu_string(double x){
    std::stringstream sstr;
    sstr << x;
    return sstr.str();
}
std::string tu_string(float x){
    std::stringstream sstr;
    sstr << x;
    return sstr.str();
}
int main(){
    using namespace Eigen;
    bool paused = true;
    constexpr size_t M = 256 + border_thickness * 2;
    constexpr size_t N = 256 + border_thickness * 2;
    yee_grid2d<scalar> grid      (M, N);
    yee_grid2d<scalar> dual_grid (M, N);
    dual_grid.eps = grid.eps;
    dual_grid.mu = grid.mu;
    dual_grid.sigma_e = grid.sigma_e;
    dual_grid.sigma_h = grid.sigma_h;
    using array = yee_grid2d<scalar>::array;
    for(int i = -25;i < 25;i++){
        for(int j = -25;j < 25;j++){
            grid.h_z(i + 128 + border_thickness, j + 50 + border_thickness) = 3 * std::exp(-(i * i + j * j) / 120.0);
        }
    }
    grid.sigma_e.fill(0);
    //grid.sigma_e.block(0, 70, grid.sigma_e.rows(), 3) = 1;
    Vector2f lenspoint1(0.57, 0.60);
    Vector2f lenspoint2(0.57, 0.90);
    array dmu(grid.m - 1, grid.n - 1);
    dmu.fill(0);
    scalar F = 0;
    scalar dFdPhi = 0;
    for(size_t i = 0;i < grid.m - 2;i++){
        for(size_t j = 0;j < grid.n - 2;j++){
            Vector2f vec(float(i - border_thickness) / grid.realm(), float(j - border_thickness) / grid.realn());
            if((vec - lenspoint1).cwiseAbs().norm() < 0.2 && (vec - lenspoint2).cwiseAbs().norm() < 0.2){
                grid.mu(i, j) *=  1.1;
                grid.eps(i, j) *= 1;
                dmu(i, j) = mu_0;
            }
            //if((vec - lenspoint1).cwiseAbs().maxCoeff() < 0.3){
            //    grid.mu(i, j) *= 2;
            //    dmu(i, j) = 1;
            //}
        }
    }
    sf::RenderWindow window(sf::VideoMode(1000, 1000), "FMD");
    using array = decltype(grid)::array;
    sf::Texture tex;
    double time = 0;
    tex.create(grid.m - 1 - 2 * border_thickness, grid.n - 1 - 2 * border_thickness);
    sf::Sprite schprit(tex);
    schprit.setScale(window.getSize().x / (grid.m - 1.0 - 2.0 * border_thickness),window.getSize().y / (grid.n - 1.0 - 2.0 * border_thickness));
    window.setVerticalSyncEnabled(0);
    window.setFramerateLimit(0);
    array elong = 
          ((grid.e_x.block(1, 0, grid.m - 1, grid.n - 1) + grid.e_x.block(0, 0, grid.m - 1, grid.n - 1)) * 0.5).square() 
        + ((grid.e_y.block(0, 1, grid.m - 1, grid.n - 1) + grid.e_y.block(0, 0, grid.m - 1, grid.n - 1)) * 0.5).square()
        + (grid.h_z).square();
    auto elongmax = elong.maxCoeff() * 10000;
    auto sigmamax = grid.sigma_e.maxCoeff() + 0.001;
    sf::Font font;
    if (!font.loadFromFile("../Arial.ttf"))
    {
        std::terminate();
    }
    while (window.isOpen()){
        //std::cout << (grid.e_y.block(0,0,grid.m - 1, grid.n - 1) / grid.h_z).abs() << "\n";
        sf::Event event;
        while (window.pollEvent(event)){
            if (event.type == sf::Event::Closed)
                window.close();
            if(event.type == sf::Event::KeyPressed){
                if(event.key.code == sf::Keyboard::Escape){
                    window.close();
                }
                if(event.key.code == sf::Keyboard::P){
                    paused = !paused;
                }
            }
        }
        window.clear();
        if(!paused)
        for(size_t i = 0;i < 1;i++){
            const decltype(grid)::scalar tstp = 0.001 / c;
            auto grid2 = timestep<decltype(grid)::scalar>(grid, tstp);
            dual_grid.m_source = dmu * ((grid2.h_z - grid.h_z) / tstp);
            //std::cout << "msource: " << (grid2.h_z - grid.h_z).maxCoeff() << "\n";
            dual_grid = timestep<decltype(grid)::scalar>(dual_grid, tstp);
            //std::cout << dual_grid.h_z.maxCoeff() << "\n";
            grid = std::move(grid2);
            time += tstp;
            vector3 E_at_point(0,0,0);
            vector3 H_at_point(0, 0, grid.h_z(grid.realm() / 2, 7 * grid.realn() / 8));
            vector3 gE_at_point(0,0,0);
            vector3 gH_at_point(0, 0, dual_grid.h_z(dual_grid.realm() / 2, 7 * dual_grid.realn() / 8));
            vector6 gEandH;
            gEandH.head(3) = gE_at_point;
            gEandH.tail(3) = gH_at_point;
            F += smallf(E_at_point, H_at_point) * tstp;
            dFdPhi += (dfdu(E_at_point, H_at_point).dot(gEandH)) * tstp;
            //std::cout << std::get<0>(deriv) << std::endl;
            
        }
        //std::terminate();
        std::vector<unsigned char> texdata((grid.m - 1 - border_thickness * 2) * (grid.n - 1 - border_thickness * 2) * 4);
        //std::cout << texdata.size() << std::endl;
        elong = 
          ((grid.e_x.block(1, 0, grid.m - 1, grid.n - 1) + grid.e_x.block(0, 0, grid.m - 1, grid.n - 1)) * 0.5).square() 
        + ((grid.e_y.block(0, 1, grid.m - 1, grid.n - 1) + grid.e_y.block(0, 0, grid.m - 1, grid.n - 1)) * 0.5).square()
        + (grid.h_z).square();
        auto blockexp_hz = grid.h_z.block(border_thickness, border_thickness, grid.m - 1 - border_thickness * 2, grid.n - 1 - border_thickness * 2);
        auto blockexp_dghz = dual_grid.h_z.block(border_thickness, border_thickness, grid.m - 1 - border_thickness * 2, grid.n - 1 - border_thickness * 2);
        auto blockexp_mu = grid.mu.block(border_thickness, border_thickness, grid.m - 1 - border_thickness * 2, grid.n - 1 - border_thickness * 2);
        for(size_t i = 0;i < grid.m - 1 - border_thickness * 2;i++){
            for(size_t j = 0;j < grid.n - 1 - border_thickness * 2;j++){
                if(i == grid.realm() / 2 && j == 7 * grid.realn() / 8){
                    texdata[(i * (grid.realn() - 1) + j) * 4] =     255;
                    texdata[(i * (grid.realn() - 1) + j) * 4 + 1] = 0;
                    texdata[(i * (grid.realn() - 1) + j) * 4 + 2] = 0;
                    texdata[(i * (grid.realn() - 1) + j) * 4 + 3] = 255;
                }
                else{
                    texdata[(i * (grid.n - 1 - 2 * border_thickness) + j) * 4] = fcolor(std::abs(blockexp_hz(i,j)));
                    texdata[(i * (grid.n - 1 - 2 * border_thickness) + j) * 4 + 1] = fcolor(relu(blockexp_dghz(i,j)));//fcolor(std::abs(grid.e_x(i,j)));
                    texdata[(i * (grid.n - 1 - 2 * border_thickness) + j) * 4 + 2] = fcolor(std::pow(blockexp_mu(i, j) / mu_0, 10) / 3);//fcolor(std::abs(grid.e_y(i,j)));;
                    texdata[(i * (grid.n - 1 - 2 * border_thickness) + j) * 4 + 3] = 255;
                }
            }
        }
        tex.update(texdata.data());
        tex.setSmooth(true);
        window.draw(schprit);
        std::stringstream sstr;
        sstr.precision(3);
        sstr << (time * 1000000000);
        std::string timestr = sstr.str();
        while(timestr.size() < 9){
            timestr += " ";
        }
        sf::Text text(timestr + " ns", font);
        sf::Text bigF("F = " + tu_string(F), font);
        sf::Text gradF("dF/dphi = " + tu_string(dFdPhi), font);
        sf::Text info_H("Red = magnetic magnitude", font);
        sf::Text info_grad("Green = positive magnetic gradient", font);
        text.setPosition(sf::Vector2f (0,0));
        bigF.setPosition(sf::Vector2f (0,50));
        gradF.setPosition(sf::Vector2f(0,100));
        text.setFillColor(sf::Color(120,180,50));
        bigF.setFillColor(sf::Color(120,180,50));
        gradF.setFillColor(sf::Color(120,180,50));
        info_H   .setFillColor(sf::Color(255,0,0));
        info_grad.setFillColor(sf::Color(255,255,0));
        info_H.setPosition(sf::Vector2f (240,0));
        info_grad.setPosition(sf::Vector2f (240,50));
        window.draw(text);
        window.draw(bigF);
        window.draw(gradF);
        window.draw(info_H);
        window.draw(info_grad);
        window.display();
    }
}
