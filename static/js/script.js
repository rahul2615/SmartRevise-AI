// Custom Scripts for SmartRevise AI

const translations = {
    'en': {
        'nav_home': 'Home',
        'nav_features': 'Features',
        'nav_dashboard': 'Dashboard',
        'nav_analytics': 'Analytics',
        'nav_planner': 'Planner',
        'nav_coding': 'Coding',
        'nav_tutor': 'AI Tutor',
        'nav_login': 'Login',
        'nav_get_started': 'Get Started',
        'nav_logout': 'Logout',
        'nav_language': 'Language'
    },
    'hi': {
        'nav_home': 'होम',
        'nav_features': 'विशेषताएं',
        'nav_dashboard': 'डेशबोर्ड',
        'nav_analytics': 'विश्लेषण',
        'nav_planner': 'योजना',
        'nav_coding': 'कोडिंग',
        'nav_tutor': 'एआई शिक्षक',
        'nav_login': 'लॉग इन',
        'nav_get_started': 'शुरू करें',
        'nav_logout': 'लॉग आउट',
        'nav_language': 'भाषा'
    }
};

function changeLanguage(lang) {
    localStorage.setItem('preferredLanguage', lang);
    const elements = document.querySelectorAll('[data-i18n]');
    elements.forEach(el => {
        const key = el.getAttribute('data-i18n');
        if (translations[lang] && translations[lang][key]) {
            el.innerText = translations[lang][key];
        }
    });
}

document.addEventListener('DOMContentLoaded', function () {
    console.log('SmartRevise AI Frontend Loaded');

    // Load saved language
    const savedLang = localStorage.getItem('preferredLanguage') || 'en';
    changeLanguage(savedLang);

    // Navbar scroll effect
    const navbar = document.querySelector('.navbar');
    if (navbar) {
        window.addEventListener('scroll', () => {
            if (window.scrollY > 50) {
                navbar.classList.add('shadow-lg');
                navbar.style.background = 'rgba(255, 255, 255, 0.95)'; // Keep it light
            } else {
                navbar.classList.remove('shadow-lg');
                navbar.style.background = 'rgba(255, 255, 255, 0.9)';
            }
        });
    }
    // Feature Highlighting Logic
    if (window.location.hash === '#features') {
        highlightFeatures();
    }

    window.addEventListener('hashchange', function () {
        if (window.location.hash === '#features') {
            highlightFeatures();
        }
    });

    function highlightFeatures() {
        const cards = document.querySelectorAll('#features .clean-card');
        cards.forEach((card, index) => {
            // Remove lingering highlights first to reset if clicked again
            card.classList.remove('highlight-effect');

            // Add persistent highlight with staggered delay
            setTimeout(() => {
                card.classList.add('highlight-effect');
            }, index * 100);
        });
    }
});
