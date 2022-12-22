function [loss,grad]=pwregress_maxent_2d_objfun(B,F,Y)
    % F: dxKd
    % B: Kdx1
    % Y: Kdx1
    d = size(F,1);
    Kd = size(B,1);
    assert(size(B,2)==1)
    assert(size(F,2)==Kd)
    assert(size(Y,1)==Kd)
    PB=1./(F*B); % dx1, take max with eps to make sure the power spectrum is positive
    if min(PB(:))>0
        loss = d/2*log(2*pi)+0.5*sum(log(PB(:)))+0.5*(B'*Y);
        grad = 0.5*(Y-sum(bsxfun(@times,F,PB),1)');
    else
        loss = NaN;
        grad = 0*Y;
    end
end