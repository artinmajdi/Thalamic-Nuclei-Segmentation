function [NucleusName,FullIndexes] = NucleiSelection(ind)

if ind == 1
    NucleusName = '1-THALAMUS';
elseif ind == 2
    NucleusName = '2-AV';
elseif ind == 4567
    NucleusName = '4567-VL';
elseif ind == 4
    NucleusName = '4-VA';
elseif ind == 5
    NucleusName = '5-VLa';
elseif ind == 6
    NucleusName = '6-VLP';
elseif ind == 7
    NucleusName = '7-VPL';
elseif ind == 8
    NucleusName = '8-Pul';
elseif ind == 9
    NucleusName = '9-LGN';
elseif ind == 10
    NucleusName = '10-MGN';
elseif ind == 11
    NucleusName = '11-CM';
elseif ind == 12
    NucleusName = '12-MD-Pf';
elseif ind == 13
    NucleusName = '13-Hb';
elseif ind == 14
    NucleusName = '14-MTT';
end

FullIndexes = [1,2,4,5,6,7,8,9,10,11,12]; % ,13,14];
end