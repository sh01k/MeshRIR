function [pos_z, ir_z] = extract_plane(pos, ir, z)
    z_list = pos(:,3);
    pos_z_idx = find(abs(z_list-z)<eps);
    pos_z = pos(pos_z_idx,:);
    ir_z = ir(:,pos_z_idx,:);
end