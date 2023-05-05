# from pages.coach_center import coach
# from record_prescriptions import record_block
# import streamlit as st
# import psycopg2
# import time



# def prescribe_block(conn, name):

#     if 'block' not in st.session_state:
#         st.session_state['block']=False

#     #st.write('Testing Coach Center and Record Prescriptions Together')
#     st.session_state['block'], name, length_of_block=coach(conn, name)
#     # submit_button=st.button('Record Block in Database')
#     if st.session_state.block:
#         record_block(conn, name, st.session_state['block'], length_of_block)
#         st.success('Block Uploaded to Database Successfully. Rerunning to update state.')
#         time.sleep(4)
#         st.session_state['block']=[]
#         st.experimental_rerun()
        


# # prescribe_block(conn)