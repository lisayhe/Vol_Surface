import numpy as np

import cmath
import math



# range for loops, from start, to finish with increment as the stepsize
def myRange(start, finish, increment):
    myZero = 1e-17
    while (start <= finish+myZero):
        yield start
        start += increment
        
        

# Periodic Linear Extension
def paramMapping(x, c, d):

    if ((x>=c) & (x<=d)):
        
        y = x

    else:
        
        range = d-c
        n = math.floor((x-c)/range)
        if (n%2 == 0):
            y = x - n*range;
        else:
            y = d + n*range - (x-c)
            
    return y

#def eValue(params, marketPrices, maturities, strikes, r, q, S0, alpha, eta, n, model):
def eValue(params, *args):
    
    marketPrices = args[0]
    maturities = args[1]
    strikes = args[2]
    r = args[3]
    q = args[4]
    S0 = args[5]
    alpha = args[6]
    eta = args[7]
    n = args[8]
    model = args[9]

    lenT = len(maturities)
    lenK = len(strikes)
    
    modelPrices = np.zeros((lenT, lenK))
    #print(marketPrices.shape)

    count = 0
    mae = 0
    for i in range(lenT):
        for j in range(lenK):
            count  = count+1
            T = maturities[i]
            K = strikes[j]
            [km, cT_km] = genericFFT(params, S0, K, r, q, T, alpha, eta, n, model)
            modelPrices[i,j] = cT_km[0]
            tmp = marketPrices[i,j]-modelPrices[i,j]
            mae += tmp**2
    
    rmse = math.sqrt(mae/count)
    return rmse

        
def generic_CF(u, params, S0, r, q, T, model):
    
    if (model == 'GBM'):
        
        sig = params[0]
        mu = np.log(S0) + (r-q-sig**2/2)*T
        a = sig*np.sqrt(T)
        phi = np.exp(1j*mu*u-(a*u)**2/2)
        
    elif(model == 'Heston'):
        
        kappa  = params[0]
        theta  = params[1]
        sigma  = params[2]
        rho    = params[3]
        v0     = params[4]
        
        kappa = paramMapping(kappa,0.1, 20)
        theta = paramMapping(theta,0.001, 0.4)
        sigma = paramMapping(sigma,0.01, 0.6)
        rho   = paramMapping(rho  ,-1.0, 1.0)
        v0    = paramMapping(v0   ,0.005, 0.25)
        
        tmp = (kappa-1j*rho*sigma*u)
        g = np.sqrt((sigma**2)*(u**2+1j*u)+tmp**2)
        
        pow1 = 2*kappa*theta/(sigma**2)
        
        numer1 = (kappa*theta*T*tmp)/(sigma**2) + 1j*u*T*r + 1j*u*math.log(S0)
        log_denum1 = pow1 * np.log(np.cosh(g*T/2)+(tmp/g)*np.sinh(g*T/2))
        tmp2 = ((u*u+1j*u)*v0)/(g/np.tanh(g*T/2)+tmp)
        log_phi = numer1 - log_denum1 - tmp2
        phi = np.exp(log_phi)
        
        #g = np.sqrt((kappa-1j*rho*sigma*u)**2+(u*u+1j*u)*sigma*sigma)
        #beta = kappa-rho*sigma*1j*u
        #tmp = g*T/2
        
        #temp1 = 1j*(np.log(S0)+(r-q)*T)*u + kappa*theta*T*beta/(sigma*sigma)
        #temp2 = -(u*u+1j*u)*v0/(g/np.tanh(tmp)+beta)
        #temp3 = (2*kappa*theta/(sigma*sigma))*np.log(np.cosh(tmp)+(beta/g)*np.sinh(tmp))
        
        #phi = np.exp(temp1+temp2-temp3);
        

    elif (model == 'VG'):
        
        sigma  = params[0];
        nu     = params[1];
        theta  = params[2];
        
        sigma = paramMapping(sigma,-10, 10)
        nu = paramMapping(nu,0.1, 10)
        theta = paramMapping(theta,-10, 10)
        
        

#        if (nu == 0):
#            mu = np.log(S0) + (r-q - theta -0.5*sigma**2)*T
#            phi  = np.exp(1j*u*mu) * np.exp((1j*theta*u-0.5*sigma**2*u**2)*T)
#        else:
#            mu  = np.log(S0) + (r-q + np.log(1-theta*nu-0.5*sigma**2*nu)/nu)*T
#            phi = np.exp(1j*u*mu)*((1-1j*nu*theta*u+0.5*nu*sigma**2*u**2)**(-T/nu))
        
        phi = (1.-1j*u*theta*nu+0.5*sigma**2*u*u*nu)**(-T/nu)
        
       
        
        #phi = 1j * u * (np.log(S0) + (r + (1/nu)*( np.log(1-theta*nu-sigma*sigma*nu/2.) )- q) * T ) - T*np.log(1 - 1j * theta * nu * u + 0.5 * sigma * sigma * u * u * nu)/nu
            
    elif (model == 'VGSA'):
        
        sigma  = params[0];
        nu     = params[1];
        theta  = params[2];
        kappa = params [3];
        eta = params[4];
        lda = params[5];
        
        sigma = paramMapping(sigma,-10, 10)
        nu = paramMapping(nu,-10, 10)
        theta = paramMapping(theta,-10, 10)
        kappa   = paramMapping(kappa  ,-10, 10)
        eta    = paramMapping(eta   ,-10, 10)
        lda = paramMapping(lda, -10, 10)
        
       
#        phiVG1 = -1./nu*cmath.log((1-1j*nu*theta*u+0.5*nu*sigma**2*u**2))
#        phiVG2 = -1./nu*cmath.log((1-1j*nu*theta*(-1j)+0.5*nu*sigma**2*(-1j)**2))
#        negiphiVG1 = -1j*phiVG1
#        negiphiVG2 = -1j * phiVG2
#        gamma1 = cmath.sqrt((kappa**2)-2.*lda**2*phiVG1)
#        gamma2 = cmath.sqrt((kappa**2)-2.*lda**2*phiVG2)
#        A1 = cmath.exp(kappa**2*eta*T/lda/lda)/((np.cosh(gamma1*T/2)+kappa/gamma1*np.sinh(gamma1*T/2))**(2*kappa*eta/lda/lda))
#        A2 = cmath.exp(kappa**2*eta*T/lda/lda)/((np.cosh(gamma2*T/2)+kappa/gamma2*np.sinh(gamma2*T/2))**(2*kappa*eta/lda/lda))
#        B1 = 2.*1j*negiphiVG1/(kappa + gamma1/np.tanh(gamma1*T/2))
#        B2 = 2.*1j*negiphiVG2/(kappa + gamma2/np.tanh(gamma2*T/2))
#        phi1 = A1* cmath.exp(B1/nu)
#        phi2 = A2* cmath.exp(B2/nu)
#        phi = cmath.exp(1j*u*(np.log(S0)+(r-q)*T))*phi1/(phi2**(1j*u))
        

        
        phi = np.exp(1j*u*(np.log(S0)+(r-q)*T))*(np.exp(kappa**2*eta*T/lda/lda)/((np.cosh((np.sqrt((kappa**2)-2.*lda**2*(-1./nu*np.log((1-1j*nu*theta*u+0.5*nu*sigma**2*u**2)))))*T/2)+kappa/(np.sqrt((kappa**2)-2.*lda**2*(-1./nu*np.log((1-1j*nu*theta*u+0.5*nu*sigma**2*u**2)))))*np.sinh((np.sqrt((kappa**2)-2.*lda**2*(-1./nu*np.log((1-1j*nu*theta*u+0.5*nu*sigma**2*u**2)))))*T/2))**(2*kappa*eta/lda/lda))* np.exp((2.*1j*(-1j*(-1./nu*np.log((1-1j*nu*theta*u+0.5*nu*sigma**2*u**2))))/(kappa + (np.sqrt((kappa**2)-2.*lda**2*(-1./nu*np.log((1-1j*nu*theta*u+0.5*nu*sigma**2*u**2)))))/np.tanh((np.sqrt((kappa**2)-2.*lda**2*(-1./nu*np.log((1-1j*nu*theta*u+0.5*nu*sigma**2*u**2)))))*T/2)))/nu))/((np.exp(kappa**2*eta*T/lda/lda)/((np.cosh((np.sqrt((kappa**2)-2.*lda**2*(-1./nu*np.log((1-1j*nu*theta*(-1j)+0.5*nu*sigma**2*(-1j)**2)))))*T/2)+kappa/(np.sqrt((kappa**2)-2.*lda**2*(-1./nu*np.log((1-1j*nu*theta*(-1j)+0.5*nu*sigma**2*(-1j)**2)))))*np.sinh((np.sqrt((kappa**2)-2.*lda**2*(-1./nu*np.log((1-1j*nu*theta*(-1j)+0.5*nu*sigma**2*(-1j)**2)))))*T/2))**(2*kappa*eta/lda/lda))* np.exp((2.*1j*(-1j * (-1./nu*np.log((1-1j*nu*theta*(-1j)+0.5*nu*sigma**2*(-1j)**2))))/(kappa + (np.sqrt((kappa**2)-2.*lda**2*(-1./nu*np.log((1-1j*nu*theta*(-1j)+0.5*nu*sigma**2*(-1j)**2)))))/np.tanh((np.sqrt((kappa**2)-2.*lda**2*(-1./nu*np.log((1-1j*nu*theta*(-1j)+0.5*nu*sigma**2*(-1j)**2)))))*T/2)))/nu))**(1j*u))
        
        
    elif (model == 'VGSSD'):
        
        sigma  = params[0]; 
        nu     = params[1]; 
        theta  = params[2]; 
        gamma  = params[3]; 
        
        sigma = paramMapping(sigma,-10, 10)
        nu = paramMapping(nu,-10, 10)
        theta = paramMapping(theta,-10, 10)
        gamma   = paramMapping(gamma  ,-10, 10)
        
        #phi1=(1.-1j*u*T**gamma*nu*theta+ 0.5*u**2*T**(2*gamma)*nu*sigma**2)**(-1/nu)
        #phi2=(1.-1j*(-1j)*T**gamma*nu*theta+ 0.5*(-1j)**2*T**(2*gamma)*nu*sigma**2)**(-1/nu)
        #phi = cmath.exp(1j*u*(np.log(S0)+(r-q)*T))*phi1/phi2
        phi = np.exp(1j*u*(np.log(S0)+(r-q)*T))*(1.-1j*u*(T**gamma)*nu*theta+ 0.5*(u**2)*(T**(2*gamma))*nu*(sigma**2))**(-1/nu)/((1.-1j*(-1j)*(T**gamma)*nu*theta+ 0.5*((-1j)**2)*(T**(2*gamma))*nu*(sigma**2))**(-1/nu))
        
    return phi

def genericFFT(params, S0, K, r, q, T, alpha, eta, n, model):
    
    N = 2**n
    
    # step-size in log strike space
    lda = (2*np.pi/N)/eta
    
    #Choice of beta
    #beta = np.log(S0)-N*lda/2
    beta = np.log(K)
    
    # forming vector x and strikes km for m=1,...,N
    km = np.zeros((N))
    xX = np.zeros((N))
    
    # discount factor
    df = math.exp(-r*T)
    
    nuJ = np.arange(N)*eta
    psi_nuJ = generic_CF(nuJ-(alpha+1)*1j, params, S0, r, q, T, model)/((alpha + 1j*nuJ)*(alpha+1+1j*nuJ))
    
    for j in range(N):  
        km[j] = beta+j*lda
        if j == 0:
            wJ = (eta/2)
        else:
            wJ = eta
        xX[j] = cmath.exp(-1j*beta*nuJ[j])*df*psi_nuJ[j]*wJ
     
    yY = np.fft.fft(xX)
    cT_km = np.zeros((N))  
    for i in range(N):
        multiplier = math.exp(-alpha*km[i])/math.pi
        cT_km[i] = multiplier*np.real(yY[i])
    
    return km, cT_km