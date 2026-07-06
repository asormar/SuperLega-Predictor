import { Routes, Route, NavLink, Link } from 'react-router-dom'
import { CircleDot, Swords, Trophy, LayoutDashboard } from 'lucide-react'
import Dashboard from './pages/Dashboard.jsx'
import SimularPartido from './pages/SimularPartido.jsx'
import SimularTemporada from './pages/SimularTemporada.jsx'
import EquipoDetalle from './pages/EquipoDetalle.jsx'

function App() {
  return (
    <div className="app-layout">
      {/* Navigation */}
      <nav className="navbar">
        <div className="navbar-inner">
          <Link to="/" className="navbar-logo">
            <span className="logo-icon">
              <CircleDot size={18} />
            </span>
            SuperLega Simulator
          </Link>
          <ul className="navbar-nav">
            <li>
              <NavLink to="/" end className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
                <LayoutDashboard size={16} />
                Inicio
              </NavLink>
            </li>
            <li>
              <NavLink to="/simular-partido" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
                <Swords size={16} />
                Simular Partido
              </NavLink>
            </li>
            <li>
              <NavLink to="/simular-temporada" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
                <Trophy size={16} />
                Simular Temporada
              </NavLink>
            </li>
          </ul>
        </div>
      </nav>

      {/* Routes */}
      <main>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/simular-partido" element={<SimularPartido />} />
          <Route path="/simular-temporada" element={<SimularTemporada />} />
          <Route path="/equipo/:nombre" element={<EquipoDetalle />} />
        </Routes>
      </main>
    </div>
  )
}

export default App
