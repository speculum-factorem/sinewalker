import './style.css'
import { createWalkApp } from './walkApp'

const root = document.querySelector<HTMLDivElement>('#app')
if (!root) throw new Error('Missing #app root element')

createWalkApp(root)
