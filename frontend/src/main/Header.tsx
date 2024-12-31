const Header: React.FC = () => {
    return (
      <header className="w-full py-1 bg-slate-950 fixed z-50">
        <div className="container mx-auto text-center">
          <span className="text-md font-mono" style={{
            color: '#ffffff',
            textShadow: '0 0 5px #32CD32, 0 0 5px #013220'
          }}>
            pipeline dream 🍄
          </span>
        </div>
      </header>
    );
};

export default Header;
