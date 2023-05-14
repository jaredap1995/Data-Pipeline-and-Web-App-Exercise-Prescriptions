import streamlit as st
import psycopg2


def name_function():
    conn = psycopg2.connect(**st.secrets.psycopg2_credentials)
    
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM client;")
    clients = [row[0] for row in cursor.fetchall()]

    if 'new_name' not in st.session_state:
        st.session_state['new_name']=False
    if 'new_submit' not in st.session_state:
        st.session_state['new_submit']=False

    # Set up the sidebar for user input
    with st.sidebar:
        st.success("Please Select your Name from the Dropdown Menu. Or if you are a new client, enter your name!")
        name = st.multiselect('Please select your name', clients, key='client_name')
        if name:
            name=name[0]
        new_client = st.button('New client?')
        if new_client or st.session_state.new_name:
            st.session_state['new_name']=True
            new_name = st.text_input('Please enter your first name:', value='')
            if st.button('Submit') or st.session_state.new_submit:
                st.session_state['new_submit']=True
                if new_name != '':
                    st.session_state['new_name'] = new_name
                    st.experimental_set_query_params(name=new_name)
                    cursor.execute("INSERT INTO client(name) VALUES (%s)", (new_name,))
                    conn.commit()
                    name=new_name
                else: 
                    st.warning('Please Enter Your Name')
        else:
            if st.button('Submit'):
                st.experimental_set_query_params(name=name)
                st.experimental_rerun()

        conn.close()
    
    if 'name' not in st.session_state or st.session_state['name'] is None:
        st.session_state['name'] = name
    else:
        name = st.session_state['name']

    return name