import React from 'react';
import ReactDOM from 'react-dom';
import TgHeader from './components/TgHeader';
import Train from './components/Train';
import "babel-polyfill";
import "./scss/app.scss";

const App = props => {
  return (
    <div>
      <TgHeader />
      <Train />
    </div>
  );
}
export default App;

ReactDOM.render(<App />, document.getElementById('app'));