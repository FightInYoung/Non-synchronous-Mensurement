function[prv1]=signal_received(M,xs1,ys1,zs,Loc_M_x,Loc_M_y,Loc_M_z,Sig7,c,dt,len)
Rsm1=[];
 for q=1:M
            rsm1=sqrt((xs1-Loc_M_x(q))^2+(ys1-Loc_M_y(q))^2+(zs-Loc_M_z(q))^2);
            Rsm1=[Rsm1 rsm1];
 end
TD=Rsm1/c;
L_TD=TD/dt;
L_TD=fix(L_TD);
Srm=[];
for p=1:M
%     srm=sig7(L_TD(p):(Lsig-100+L_TD(p)))/Rsm(p)/50;
    srm=[zeros(1,L_TD(p)) Sig7]/Rsm1(p)/50;
    srm=srm(1:len);
    Srm=[Srm srm];
end
Nt=length(Srm);
dNt=Nt/M;
for i=1:M
    Mtotal1(i,:)=Srm(1+dNt*(i-1):i*dNt);
end
    

Mtotal1=Mtotal1';
prv1=Mtotal1;
Mtotal1=[];
end