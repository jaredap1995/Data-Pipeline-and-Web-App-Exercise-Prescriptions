from coach_center import deload, increasing_load, create_a_block, coach
from record_prescriptions import record_block
import streamlit as st
import psycopg2



def prescribe_block(conn):

    if 'block' not in st.session_state:
        st.session_state['block']=False

    #st.write('Testing Coach Center and Record Prescriptions Together')
    st.session_state['block'], name=coach(conn)
    st.write(name)
    st.write(st.session_state['block'])
    # submit_button=st.button('Record Block in Database')
    if st.session_state.block:
        record_block(conn, name, st.session_state['block'])
        st.success('Block Uploaded to Database Successfully')
        


# prescribe_block(conn)