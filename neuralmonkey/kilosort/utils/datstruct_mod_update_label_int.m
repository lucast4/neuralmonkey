function DATSTRUCT = datstruct_mod_update_label_int(DATSTRUCT)
% Update label_final_int to match the string label_final, for each
% item in DATSTRUCT
for i=1:length(DATSTRUCT)
    switch DATSTRUCT(i).label_final
        case 'su'
            DATSTRUCT(i).label_final_int = 2;
        case 'mua'
            DATSTRUCT(i).label_final_int = 1;
        case 'noise'
            DATSTRUCT(i).label_final_int = 0;
        case 'artifact'
            DATSTRUCT(i).label_final_int = 0;
        otherwise
            disp(DATSTRUCT(i).label_final)
            assert(false);
    end
end
end