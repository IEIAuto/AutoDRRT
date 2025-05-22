const int64 DecodeCAN_ID_SCU_17 = 0x22D

void AION :: CPciCanRcvSnd :: DecodeCAN_ID_SCU_17(const uint8* buf)
{
    static int cout_ManutoAuto = 0;
    uint8 data[8];
    int index = 0;
    memcpy(&data, buf, 8 * sizeof(uint8));
    //---------------------------------------------------------------------
    // 小端Motorola格式将原始字节序列倒置
    //---------------------------------------------------------------------
    for (index = 0; 8 > index; index++)
    {
        data[index] = buf[7 - index];
    }
    v_prep_msg.vcu_vehwheeltorqmax = VCU_VehWheelTorqMax_handle.RealCanDataVal(data);//驱动系统允许的最大扭矩
    v_prep_msg.vcu_gaspedalactpst = VCU_GasPedalActPst_handle.RealCanDataVal(data);//当前油门踏板实际位置
    v_prep_msg.vcu_accelececfail = VCU_AccElecECFail_handle.RealCanDataVal(data);//驱动系统故障
    v_prep_msg.vcu_accbuttinfo  = VCU_ACCButtInfo_handle.RealCanDataVal(data);
    //std::cout << "vcu_vehwheeltorqmax:" << v_prep_msg.vcu_vehwheeltorqmax <<std::endl;
    std::cout << "vcu_accbuttinfo:" << int(v_prep_msg.vcu_accbuttinfo) <<std::endl;
    if (v_prep_msg.vcu_accbuttinfo == 1) 
    {
        cout_ManutoAuto = 0;
        main_swtich = true;
    }
    else if(v_prep_msg.vcu_accbuttinfo == 3)
    {
        cout_ManutoAuto++;
        if (cout_ManutoAuto>21)
        {
            cout_ManutoAuto = 21;
        }
    }
    else if(v_prep_msg.vcu_accbuttinfo == 17)
    {
        is_engage = true;
    }
    else
    {
        main_swtich = false;
        cout_ManutoAuto = 0;
    }
    
    if (cout_ManutoAuto > 20)
    {
        ManutoAuto = true;
    }
    else
    {
        ManutoAuto = false;
    }

    std::cout << "ManutoAuto:" << int(ManutoAuto)<<std::endl;
    
}