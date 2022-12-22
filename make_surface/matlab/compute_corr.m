function corr=compute_corr(fftimg,filnew,J,L2,j1,ell1,k1,j2,ell2,k2)
    
    filid=1;
    for j=0:J-1
        for ell =0:L2-1
            if (j==j1 && ell==ell1) || (j==j2 && ell==ell2)
                fil = filnew.psi.filter{filid}.coefft{1};
                x_la = ifft2(fil.*fftimg);
                amp_la = abs(x_la);
                angle_la = angle(x_la);
                if j==j1 && ell==ell1
                    U1 = amp_la .* exp(1i*k1*angle_la);
                end
                if j==j2 && ell==ell2
                    U2 =  amp_la .* exp(1i*k2*angle_la);
                end
            end
            filid = filid + 1;
        end
    end
    assert(filid==length(filnew.psi.filter)+1)
    
    
    corr = mean(mean(U1 .* conj(U2)));
end
