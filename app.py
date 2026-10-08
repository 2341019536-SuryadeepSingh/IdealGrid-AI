CSS="<style>\n@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');\nhtml,body,[class*='css']{font-family:'Inter',sans-serif}.stApp{background:#07111F;color:#E5EEF7}.block-container{max-width:1500px;padding:2rem 3rem 3rem}\nsection[data-testid='stSidebar']{background:#091725;border-right:1px solid #173149}.hero{padding:8px 0 25px;border-bottom:1px solid #173149;margin-bottom:24px}.hero h1{margin:8px 0 4px;font-size:2.7rem;font-weight:800;color:#F8FAFC}.hero p{margin:0;color:#8FA5B8}.badge{display:inline-block;padding:6px 11px;border-radius:999px;background:#0A2231;border:1px solid #14506A;color:#38BDF8;font-size:.72rem;font-weight:800;letter-spacing:.08em}.section-title{color:#F8FAFC;font-size:1.25rem;font-weight:800;margin:24px 0 10px}.section-sub{color:#8299AD;margin:-5px 0 15px}.kpi{background:linear-gradient(145deg,#0E1E2E,#0B1826);border:1px solid #1B354A;border-radius:17px;padding:18px 20px;min-height:112px}.kpi-label{color:#8299AD;font-size:.74rem;text-transform:uppercase;letter-spacing:.08em;font-weight:700}.kpi-value{color:#F8FAFC;font-size:1.75rem;font-weight:800;margin-top:6px}.kpi-note{color:#6F879B;font-size:.76rem;margin-top:4px}.region-card{background:linear-gradient(145deg,#0D1D2C,#0A1725);border:1px solid #1B354A;border-radius:18px;padding:20px;min-height:285px;margin-bottom:18px}.region-name{color:#F8FAFC;font-size:1.18rem;font-weight:800}.status{font-size:.76rem;font-weight:800;letter-spacing:.08em}.healthy{color:#10B981}.warning{color:#F59E0B}.critical{color:#EF4444}.big-number{color:#F8FAFC;font-size:1.8rem;font-weight:800;margin:11px 0 4px}.small-label{color:#7991A5;font-size:.72rem}.small-value{color:#DDEAF5;font-weight:700;font-size:.9rem}.divider{height:1px;background:#183047;margin:14px 0}.rec{color:#8FA5B8;font-size:.82rem;line-height:1.45}.info-box{background:#0A1D2A;border:1px solid #16465A;border-radius:14px;padding:14px 16px;color:#9FB4C6}div[data-testid='stMetric']{background:#0E1E2E;border:1px solid #1B354A;border-radius:15px}\n</style>"
import json
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from src.grid_engine import REGIONS, analyze_region, optimize_transfers

st.set_page_config(page_title="IdealGrid AI",page_icon="⚡",layout="wide",initial_sidebar_state="expanded")
st.markdown(CSS,unsafe_allow_html=True)

@st.cache_data
def load_grid(): return pd.read_csv("data/processed/grid_hourly.csv",parse_dates=["datetime"],index_col="datetime")
def dark(fig,title):
    fig.update_layout(template="plotly_dark",paper_bgcolor="#07111F",plot_bgcolor="#07111F",font=dict(color="#AFC0CE"),title=dict(text=title,font=dict(size=18,color="#F8FAFC")),margin=dict(l=20,r=20,t=55,b=20),legend=dict(bgcolor="rgba(0,0,0,0)")); return fig

st.sidebar.markdown("## ⚡ IdealGrid AI")
st.sidebar.caption("Regional Power Grid Intelligence")
page=st.sidebar.radio("Navigation",["Command Center","Grid Optimization","Model Performance","Data Explorer"])
st.sidebar.markdown("---")
st.sidebar.caption("Five regional XGBoost models")
st.sidebar.caption("Isolation Forest anomaly layer")
st.sidebar.markdown("**Features**")
st.sidebar.code("lag_24\nlag_168\nhour\nsin_month\ncos_month",language="text")
st.markdown('<div class="hero"><span class="badge">AI POWER GRID INTELLIGENCE</span><h1>IdealGrid AI</h1><p>Forecast • Analyze • Detect • Optimize — regional electricity intelligence in one command center.</p></div>',unsafe_allow_html=True)

if page=="Command Center":
    grid=load_grid()
    st.markdown('<div class="section-title">Analysis mode</div>',unsafe_allow_html=True)
    mode=st.radio("Mode",["Historical Replay","Future / Manual Scenario"],horizontal=True,label_visibility="collapsed")
    if mode=="Historical Replay":
        options=list(grid.index[168:]); selected=st.select_slider("Select historical hour",options=options,value=options[len(options)//2],format_func=lambda x:x.strftime("%d %b %Y • %H:%M")); timestamp=pd.Timestamp(selected)
        rows=[]
        for region in REGIONS:
            dc=f"{region} Region Hourly Demand"; sc=f"{region} Region Supply"
            rows.append({"Region":region,"Demand 24h Ago (MW)":float(grid.loc[timestamp-pd.Timedelta(hours=24),dc]),"Demand 168h Ago (MW)":float(grid.loc[timestamp-pd.Timedelta(hours=168),dc]),"Avg Generation Equivalent (MW)":float(grid.loc[timestamp,sc]),"Actual Demand (MW)":float(grid.loc[timestamp,dc])})
        inputs=pd.DataFrame(rows); st.markdown('<div class="info-box">Historical Replay automatically loads lag-24, lag-168, daily generation converted to average MW equivalent, and actual demand from your datasets. The generation figure is an analytical daily-average benchmark, not an instantaneous supply measurement.</div>',unsafe_allow_html=True); st.dataframe(inputs.style.format({c:"{:,.1f}" for c in inputs.columns if c!="Region"}),use_container_width=True,hide_index=True)
    else:
        c1,c2=st.columns(2)
        with c1: date=st.date_input("Forecast date")
        with c2: hour=st.selectbox("Forecast hour",range(24),index=18,format_func=lambda x:f"{x:02d}:00")
        timestamp=pd.Timestamp(date).replace(hour=hour); inputs=pd.DataFrame({"Region":REGIONS,"Demand 24h Ago (MW)": [0.0]*5,"Demand 168h Ago (MW)":[0.0]*5,"Avg Generation Equivalent (MW)":[0.0]*5,"Actual Demand (MW)":[None]*5})
        st.caption("Enter lag-24, lag-168 and the daily-generation average MW equivalent. Actual demand is optional for post-hoc anomaly analysis.")
        st.markdown('<div class="info-box">Use the generation-average value as a benchmark against forecast demand. It is derived from daily MU generation and should not be interpreted as real-time available capacity.</div>',unsafe_allow_html=True)
        inputs=st.data_editor(inputs,use_container_width=True,hide_index=True,column_config={"Region":st.column_config.TextColumn(disabled=True),"Demand 24h Ago (MW)":st.column_config.NumberColumn(min_value=0),"Demand 168h Ago (MW)":st.column_config.NumberColumn(min_value=0),"Avg Generation Equivalent (MW)":st.column_config.NumberColumn(min_value=0),"Actual Demand (MW)":st.column_config.NumberColumn(min_value=0)})
    if st.button("⚡ RUN GRID ANALYSIS",type="primary",use_container_width=True):
        results=[]
        for _,row in inputs.iterrows():
            if row["Demand 24h Ago (MW)"]<=0 or row["Demand 168h Ago (MW)"]<=0 or row["Avg Generation Equivalent (MW)"]<=0: st.error(f"Enter positive lag and generation-average values for {row['Region']}."); st.stop()
            actual=None if pd.isna(row["Actual Demand (MW)"]) else row["Actual Demand (MW)"]
            results.append(analyze_region(row["Region"],row["Demand 24h Ago (MW)"],row["Demand 168h Ago (MW)"],row["Avg Generation Equivalent (MW)"],timestamp,actual))
        st.session_state["results"]=results; st.session_state["mode"]=mode; st.session_state["timestamp"]=timestamp
    results=st.session_state.get("results")
    if results:
        total_d=sum(r["predicted_demand"] for r in results); total_s=sum(r["avg_generation_mw"] for r in results); gap=total_s-total_d; util=total_d/total_s*100 if total_s else 0
        h=sum(r["status"]=="HEALTHY" for r in results); w=sum(r["status"]=="WARNING" for r in results); c=sum(r["status"]=="CRITICAL" for r in results); overall="CRITICAL" if c else ("WARNING" if w else "HEALTHY")
        st.markdown('<div class="section-title">Grid command center</div>',unsafe_allow_html=True); st.markdown('<div class="info-box">Generation values are daily regional generation (MU/day) converted to an average MW equivalent. They are an analytical benchmark, not real-time dispatchable supply.</div>',unsafe_allow_html=True); k=st.columns(5)
        vals=[("Forecast demand",f"{total_d:,.0f}","MW • 5 regions"),("Avg generation equivalent",f"{total_s:,.0f}","MW • scenario"),("Net reserve",f"{gap:,.0f}","MW • generation average − demand"),("Generation coverage",f"{util:.1f}%","weighted analytical view"),("Grid health",overall,f"{h} healthy • {w} warning • {c} critical")]
        for col,(lab,val,note) in zip(k,vals): col.markdown(f'<div class="kpi"><div class="kpi-label">{lab}</div><div class="kpi-value">{val}</div><div class="kpi-note">{note}</div></div>',unsafe_allow_html=True)
        st.markdown('<div class="section-title">Regional intelligence</div>',unsafe_allow_html=True); cards=st.columns(2)
        for i,r in enumerate(results):
            diag=""
            if "anomaly" in r: diag=f'<div class="divider"></div><div class="small-label">ACTUAL-DEMAND DIAGNOSTICS</div><div style="margin-top:8px"><span class="small-label">Residual</span> <span class="small-value">{r["residual"]:,.1f} MW</span> &nbsp;&nbsp; <span class="small-label">Anomaly</span> <span class="small-value">{("DETECTED" if r["anomaly"] else "NORMAL")}</span> &nbsp;&nbsp; <span class="small-label">Inefficiency</span> <span class="small-value">{r["inefficiency_score"]:.1f}%</span></div>'
            with cards[i%2]: st.markdown(f'<div class="region-card"><div class="region-name">{r["region"]} Region</div><div class="status {r["status"].lower()}">● {r["status"]}</div><div class="divider"></div><div class="small-label">PREDICTED DEMAND</div><div class="big-number">{r["predicted_demand"]:,.1f} MW</div><div style="display:flex;gap:28px"><div><div class="small-label">AVG GENERATION</div><div class="small-value">{r["avg_generation_mw"]:,.1f} MW</div></div><div><div class="small-label">GAP</div><div class="small-value">{r["gap"]:,.1f} MW</div></div><div><div class="small-label">GENERATION COVERAGE</div><div class="small-value">{r["coverage"]:.1f}%</div></div></div>{diag}<div class="divider"></div><div class="rec">{r["recommendation"]}</div></div>',unsafe_allow_html=True)
        fig=go.Figure(); fig.add_trace(go.Bar(x=[r["region"] for r in results],y=[r["predicted_demand"] for r in results],name="Predicted Demand")); fig.add_trace(go.Bar(x=[r["region"] for r in results],y=[r["avg_generation_mw"] for r in results],name="Avg Generation Equivalent")); dark(fig,"Demand vs Generation Average Equivalent"); st.plotly_chart(fig,use_container_width=True)
        if any("actual_demand" in r for r in results):
            comp=[r for r in results if "actual_demand" in r]; fig2=go.Figure(); fig2.add_trace(go.Bar(x=[r["region"] for r in comp],y=[r["actual_demand"] for r in comp],name="Actual Demand")); fig2.add_trace(go.Bar(x=[r["region"] for r in comp],y=[r["predicted_demand"] for r in comp],name="Predicted Demand")); dark(fig2,"Actual vs Predicted Demand"); st.plotly_chart(fig2,use_container_width=True)
        if mode=="Historical Replay":
            trend=grid.loc[timestamp-pd.Timedelta(hours=24):timestamp]; fig3=go.Figure()
            for region in REGIONS: fig3.add_trace(go.Scatter(x=trend.index,y=trend[f"{region} Region Hourly Demand"],mode="lines",name=region))
            fig3.add_vline(x=timestamp,line_dash="dash"); dark(fig3,"24-hour regional demand context"); st.plotly_chart(fig3,use_container_width=True)

elif page=="Grid Optimization":
    results=st.session_state.get("results")
    if not results: st.info("Run Grid Analysis first.")
    else:
        st.markdown('<div class="section-title">Regional balancing & optimization</div>',unsafe_allow_html=True); st.markdown('<div class="section-sub">Analytical balancing recommendations based on forecast demand versus the daily-generation average equivalent. These are planning suggestions, not physical dispatch commands.</div>',unsafe_allow_html=True)
        transfers,rem_s,rem_d=optimize_transfers(results); a,b,c=st.columns(3); a.metric("Total analytical surplus",f"{sum(max(r['gap'],0) for r in results):,.0f} MW"); b.metric("Total analytical deficit",f"{sum(max(-r['gap'],0) for r in results):,.0f} MW"); c.metric("Unresolved deficit",f"{rem_d:,.0f} MW")
        if transfers:
            st.markdown("### Suggested transfers"); tdf=pd.DataFrame(transfers).rename(columns={"from":"From Region","to":"To Region","mw":"Suggested Transfer (MW)"}); st.dataframe(tdf.style.format({"Suggested Transfer (MW)":"{:,.1f}"}),use_container_width=True,hide_index=True); fig=go.Figure(go.Bar(x=[f"{x['from']} → {x['to']}" for x in transfers],y=[x["mw"] for x in transfers],name="MW")); dark(fig,"Analytical regional balancing transfers"); st.plotly_chart(fig,use_container_width=True)
        else: st.success("No regional deficit is present in the current analytical scenario.")

elif page=="Model Performance":
    metrics=json.loads(Path("models/metrics.json").read_text()); perf=pd.DataFrame([{"Region":k,**v} for k,v in metrics.items()]); st.markdown('<div class="section-title">Model performance</div>',unsafe_allow_html=True); st.markdown('<div class="section-sub">Chronological 80/20 evaluation using later observations as the test set.</div>',unsafe_allow_html=True); st.dataframe(perf.style.format({"MAE_MW":"{:,.2f}","RMSE_MW":"{:,.2f}","R2":"{:.3f}","MAPE_pct":"{:.2f}%"}),use_container_width=True,hide_index=True); fig=go.Figure(go.Bar(x=perf["Region"],y=perf["R2"],name="R²")); fig.update_yaxes(range=[0,1]); dark(fig,"R² by Region"); st.plotly_chart(fig,use_container_width=True)

elif page=="Data Explorer":
    grid=load_grid(); st.markdown('<div class="section-title">Data explorer</div>',unsafe_allow_html=True); st.markdown('<div class="info-box">Generation columns are daily generation converted to average MW equivalent and repeated hourly for analytical comparison with hourly demand.</div>',unsafe_allow_html=True); region=st.selectbox("Region",REGIONS); dc=f"{region} Region Hourly Demand"; sc=f"{region} Region Supply"; window=grid[[dc,sc]].tail(1000); fig=go.Figure(); fig.add_trace(go.Scatter(x=window.index,y=window[dc],name="Demand")); fig.add_trace(go.Scatter(x=window.index,y=window[sc],name="Avg Generation Equivalent")); dark(fig,f"{region} — demand vs generation average equivalent"); st.plotly_chart(fig,use_container_width=True); st.dataframe(window.tail(100),use_container_width=True)

st.markdown("---"); st.caption("IdealGrid AI • XGBoost Forecasting • Isolation Forest • Generation-Balance Analytics • Streamlit + Plotly")
