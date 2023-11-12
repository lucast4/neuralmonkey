function DATSTRUCT = datstruct_classify(DATSTRUCT)

%% ASSIGNING SU
% 1) if bipolar, then not SU.
% 2) snr>4.4 and Q<0.05
% - clear cases:
% 4.9,

% TODO:
% - compute traditional refractoriness.

% NOTES ON THRESHOLDS:
% - snr>4.9
% - Q<0.05;
% - isiviolations (frac)<0.03


[~, ~, ~, ~, THRESH_SU_SNR, THRESH_SU_ISI, ...
    THRESH_ARTIFACT_SHARP, THRESH_ARTIFACT_SHARP_LOW, THRESH_ARTIFACT_ISI, ...
    MIN_SNR, SNR_VER, THRESH_SU_SNR_LOWER, THRESH_SU_SNR_HIGHEST] = quickscript_config_load();


% EXTRACT
list_isi = [DATSTRUCT.isi_violation_pct];
list_refract = [DATSTRUCT.Q];
list_snr = [DATSTRUCT.snr_final];
list_sharpiness = [DATSTRUCT.sharpiness];

% THRESH_SU_SNR = 5;
% THRESH_SU_ISI = 0.02;
% THRESH_ARTIFACT_SHARP = 20;
% THRESH_ARTIFACT_SHARP_LOW = 10;
% THRESH_ARTIFACT_ISI = 0.12;

% 0) SU (very high snr)
inds_good_isi = list_isi<0.05;
inds_good_snr = list_snr>THRESH_SU_SNR_HIGHEST;
inds_su_0 = inds_good_isi & inds_good_snr;

% 1) SU (high snr)
% inds_good_isi = list_isi<THRESH_SU_ISI | list_refract<0.05;
inds_good_isi = list_isi<THRESH_SU_ISI | list_refract<0.2; % 11/12/23 - more lenient, to match below.
inds_good_snr = list_snr>THRESH_SU_SNR;
inds_su_1 = inds_good_isi & inds_good_snr;

% 2) SU (lower snr, but more stringent isi)
% inds_good_isi = list_isi<0.01 & list_refract<0.02;
inds_good_isi = list_isi<0.01 & list_refract<0.125; % 9/16/23 - more lenient
inds_good_snr = list_snr>THRESH_SU_SNR_LOWER;
inds_su_2 = inds_good_isi & inds_good_snr;

% combine SU
inds_su = inds_su_0 | inds_su_1 | inds_su_2;

% 2) Noise (low snr)
% MIN_SNR = 2.25;
inds_noise = list_snr<=MIN_SNR;

% 2.1) Noise (artifact)
inds_artifact1 = list_sharpiness>THRESH_ARTIFACT_SHARP;
inds_artifact2 = list_sharpiness>THRESH_ARTIFACT_SHARP_LOW & list_isi>THRESH_ARTIFACT_ISI;
inds_artifact3 = list_refract > 20 | isinf(list_refract);
% list_sharpiness>THRESH_ARTIFACT_SHARP_LOW & list_isi>THRESH_ARTIFACT_ISI;
inds_artifact = inds_artifact1 | inds_artifact2 | inds_artifact3;

% 3) mu
inds_mu = ~inds_su & ~inds_noise;

tmp = [find(inds_su) find(inds_mu) find(inds_noise)];
assert(length(unique(tmp))==length(tmp))
assert(length(unique(tmp))==length(DATSTRUCT))
assert(all(diff(sort(unique(tmp)))==1))

% distribution of su and mu across channels
for i=find(inds_su)
    DATSTRUCT(i).label_final = 'su';
    DATSTRUCT(i).label_final_int = 2;
end
for i=find(inds_mu)
    DATSTRUCT(i).label_final = 'mua';
    DATSTRUCT(i).label_final_int = 1;
end
for i=find(inds_noise)
    DATSTRUCT(i).label_final = 'noise';
    DATSTRUCT(i).label_final_int = 0;
end
for i=find(inds_artifact)
    DATSTRUCT(i).label_final = 'artifact';
    DATSTRUCT(i).label_final_int = 0;
end

