import math
from orbitforge.time.leap_seconds import utc_unix_to_tai,tai_to_utc_unix
from orbitforge.core.vector import Vec3
from orbitforge.frames.eci_ecef import eci_to_ecef,ecef_to_eci
from orbitforge.orbits.kepler import solve_kepler_elliptic
from orbitforge.maneuvers.lambert import lambert_universal
from orbitforge.visibility.station import GroundStation,look_angles
from orbitforge.environment.eclipse import eclipse_state
from orbitforge.link.budget import free_space_loss_db,received_power_dbw
from orbitforge.attitude.quaternion import Quaternion
from orbitforge.conjunction.covariance import rotate_covariance_2d
from orbitforge.conjunction.probability import collision_probability_2d,collision_probability_at_tca
from orbitforge.conjunction.propagation import propagate_covariance,covariance_2d_at_tca
from orbitforge.core.errors import ValidationError
from orbitforge.ephemeris.interpolation import EphemerisPoint,hermite
from orbitforge.mission.windows import intersect_windows
from orbitforge.core.state import TimeWindow

def test_b01_leap_roundtrip():
    utc=1483228800.25; tai=utc_unix_to_tai(utc); assert abs(tai-utc-37)<1e-9; assert abs(tai_to_utc_unix(tai)-utc)<1e-9

def test_b02_frame_roundtrip():
    v=Vec3(7000,-1200,900); t=1700000000.0; q=ecef_to_eci(eci_to_ecef(v,t),t); assert (q-v).norm()<1e-9

def test_b03_kepler_high_eccentricity():
    e=.95; m=2.4; E=solve_kepler_elliptic(m,e); assert abs(E-e*math.sin(E)-((m+math.pi)%(2*math.pi)-math.pi))<1e-10

def test_b04_lambert_hits_transfer_direction():
    r1=Vec3(7000,0,0); r2=Vec3(0,8000,0); v1,v2=lambert_universal(r1,r2,1800); assert v1.y>0 and v2.x<0

def test_b05_ground_station_zenith():
    from orbitforge.frames.geodetic import geodetic_to_ecef
    from orbitforge.frames.rotations import rz,mv,transpose
    from orbitforge.time.sidereal import gmst_angle
    lat=math.radians(35); lon=math.radians(139); t=1700000000
    st=GroundStation('X',lat,lon,0,0)
    ecef=geodetic_to_ecef(lat,lon,500)
    sat=mv(transpose(rz(gmst_angle(t))),ecef)
    az,el,r=look_angles(st,sat,t)
    assert el>math.radians(89) and 499<r<501

def test_b06_eclipse_night_side():
    assert eclipse_state(Vec3(-7000,0,0),Vec3(149597870,0,0))=='umbra'
    assert eclipse_state(Vec3(7000,0,0),Vec3(149597870,0,0))=='sunlit'

def test_b07_link_fspl_dimension():
    loss=free_space_loss_db(1000,2e9); assert 158<loss<159
    assert abs(received_power_dbw(10,20,30,1000,2e9,3)-(10+20+30-loss-3))<1e-12

def test_b08_quaternion_rotation():
    q=Quaternion.from_axis_angle(Vec3(0,0,1),math.pi/2); v=q.rotate(Vec3(1,0,0)); assert abs(v.x)<1e-9 and abs(v.y-1)<1e-9

def test_b09_covariance_rotation():
    c=((4.0,0.0),(0.0,1.0)); r=rotate_covariance_2d(c,math.pi/2); assert abs(r[0][0]-1)<1e-9 and abs(r[1][1]-4)<1e-9

def test_b10_collision_probability_monotonic():
    cov=((1.0,0.0),(0.0,1.0)); p1=collision_probability_2d(0,0,cov,.05); p2=collision_probability_2d(4,0,cov,.05); assert p1>p2>0

def test_b11_hermite_endpoints():
    p0=EphemerisPoint(0,Vec3(1,2,3),Vec3(4,5,6)); p1=EphemerisPoint(10,Vec3(11,12,13),Vec3(1,2,3)); assert hermite(p0,p1,0)==p0; assert hermite(p0,p1,10)==p1

def test_b12_window_half_open_semantics():
    a=[TimeWindow(0,10),TimeWindow(20,30)]; b=[TimeWindow(10,20),TimeWindow(25,35)]; out=intersect_windows(a,b); assert len(out)==1 and out[0]==TimeWindow(25,30)

def _cov6(position_vars,velocity_vars,cross=None):
    m=[[0.0]*6 for _ in range(6)]
    for i,v in enumerate(position_vars+velocity_vars): m[i][i]=v
    for (i,j),v in (cross or {}).items(): m[i][j]=v; m[j][i]=v
    return m

def test_b13_covariance_propagation_chain():
    rel_r=Vec3(-1000,5,0); rel_v=Vec3(10,0,0)
    p=_cov6([4.0,1.0,9.0],[0.01,0.02,0.03],{(1,2):2.0}); s=_cov6([1.0,1.0,1.0],[0.001,0.001,0.001])
    cov2,r_tca,dt=covariance_2d_at_tca(rel_r,rel_v,p,s)
    assert abs(dt-100)<1e-12 and abs(r_tca.x)<1e-9 and abs(r_tca.y-5)<1e-9
    assert abs(cov2[0][0]-212)<1e-9 and abs(cov2[1][1]-320)<1e-9 and abs(cov2[0][1]-2)<1e-9 and abs(cov2[1][0]-2)<1e-9

def test_b14_pc_from_6d_matches_2d():
    rel_r=Vec3(-1000,5,0); rel_v=Vec3(10,0,0)
    p=_cov6([4.0,1.0,9.0],[0.01,0.02,0.03],{(1,2):2.0}); s=_cov6([1.0,1.0,1.0],[0.001,0.001,0.001])
    out=collision_probability_at_tca(rel_r,rel_v,p,s,0.05)
    direct=collision_probability_2d(-5.0,0.0,((212.0,2.0),(2.0,320.0)),0.05)
    assert abs(out['collision_probability']-direct)<1e-12 and abs(out['miss_distance_km']-5)<1e-9
    assert out['collision_probability']>0

def test_b15_invalid_covariance_rejected():
    bad_asym=_cov6([1.0]*3,[0.001]*3); bad_asym[0][1]=0.5
    bad_neg=_cov6([1.0,-1.0,1.0],[0.001]*3)
    bad_semidef=_cov6([1.0,0.0,1.0],[0.001]*3)
    for bad in (bad_asym,bad_neg,bad_semidef):
        try: propagate_covariance(bad,100.0); assert False
        except ValidationError: pass
    try: collision_probability_2d(0,0,((1.0,0.5),(0.0,1.0)),0.05); assert False
    except ValidationError: pass
    try: collision_probability_2d(0,0,((-1.0,0.0),(0.0,-1.0)),0.05); assert False
    except ValidationError: pass
    try: collision_probability_at_tca(Vec3(-1000,5,0),Vec3(10,0,0),bad_neg,_cov6([1.0]*3,[0.001]*3),0.05); assert False
    except ValidationError: pass
