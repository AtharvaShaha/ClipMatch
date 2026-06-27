/**
 * ClipMatch Home Page - Interactive Features
 */

document.addEventListener('DOMContentLoaded', function() {
    // Mobile Menu Toggle
    const mobileMenuBtn = document.getElementById('mobileMenuBtn');
    const mobileMenu = document.getElementById('mobileMenu');

    if (mobileMenuBtn) {
        mobileMenuBtn.addEventListener('click', function() {
            mobileMenu.classList.toggle('active');
            
            // Animate hamburger icon
            const spans = mobileMenuBtn.querySelectorAll('span');
            spans[0].style.transform = mobileMenu.classList.contains('active') 
                ? 'rotate(45deg) translateY(15px)' 
                : 'rotate(0) translateY(0)';
            spans[1].style.opacity = mobileMenu.classList.contains('active') ? '0' : '1';
            spans[2].style.transform = mobileMenu.classList.contains('active') 
                ? 'rotate(-45deg) translateY(-15px)' 
                : 'rotate(0) translateY(0)';
        });

        // Close menu when link is clicked
        const mobileLinks = mobileMenu.querySelectorAll('a');
        mobileLinks.forEach(link => {
            link.addEventListener('click', function() {
                mobileMenu.classList.remove('active');
                const spans = mobileMenuBtn.querySelectorAll('span');
                spans[0].style.transform = 'rotate(0) translateY(0)';
                spans[1].style.opacity = '1';
                spans[2].style.transform = 'rotate(0) translateY(0)';
            });
        });
    }

    // Smooth scroll for anchor links
    const scrollLinks = document.querySelectorAll('.scroll-link');
    scrollLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;
            
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                targetElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });

    // Intersection Observer for animations on scroll
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const animationObserver = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.animation = entry.target.style.animation || 'slideUp 0.6s ease-out';
            }
        });
    }, observerOptions);

    // Observe feature and use-case cards
    document.querySelectorAll('.feature-card, .use-case-card, .spec-box, .process-step').forEach(card => {
        card.style.opacity = '0';
        animationObserver.observe(card);
    });

    // Scroll parallax effect
    window.addEventListener('scroll', function() {
        const scrollY = window.scrollY;
        const heroContent = document.querySelector('.hero-content');
        
        if (heroContent && scrollY < window.innerHeight) {
            heroContent.style.transform = `translateY(${scrollY * 0.5}px)`;
            heroContent.style.opacity = Math.max(0, 1 - (scrollY / window.innerHeight * 0.5));
        }
    });

    // Floating cards parallax
    window.addEventListener('mousemove', function(e) {
        const floatingCards = document.querySelectorAll('.floating-card');
        floatingCards.forEach(card => {
            const rect = card.getBoundingClientRect();
            const x = (e.clientX - rect.left) / rect.width;
            const y = (e.clientY - rect.top) / rect.height;
            
            card.style.transform = `perspective(1000px) rotateX(${(y - 0.5) * 10}deg) rotateY(${(x - 0.5) * 10}deg)`;
        });
    });

    // Stats counter animation
    const animateCounter = (element, target) => {
        let current = 0;
        const increment = target / 50;
        
        const counter = setInterval(() => {
            current += increment;
            if (current >= target) {
                element.textContent = target;
                clearInterval(counter);
            } else {
                element.textContent = Math.floor(current);
            }
        }, 30);
    };

    // Observe stats section
    const statsContainer = document.querySelector('.hero-stats');
    if (statsContainer) {
        const statsObserver = new IntersectionObserver((entries) => {
            if (entries[0].isIntersecting) {
                const statNumbers = document.querySelectorAll('.stat-number');
                statNumbers.forEach(stat => {
                    if (stat.textContent === '40%+') {
                        animateCounter(stat, 40);
                    }
                });
                statsObserver.unobserve(statsContainer);
            }
        }, { threshold: 0.5 });

        statsObserver.observe(statsContainer);
    }

    // Active nav link on scroll
    window.addEventListener('scroll', function() {
        const sections = document.querySelectorAll('section');
        const navLinks = document.querySelectorAll('.nav-link-home');
        
        let current = '';
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            if (window.pageYOffset >= sectionTop - 200) {
                current = section.getAttribute('id');
            }
        });

        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === '#' + current) {
                link.classList.add('active');
            }
        });
    });

    // Add subtle glow effect to cards on hover
    const cards = document.querySelectorAll('.feature-card, .use-case-card, .process-step, .spec-box');
    cards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.filter = 'drop-shadow(0 0 20px rgba(0, 255, 136, 0.3))';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.filter = 'drop-shadow(0 0 0px rgba(0, 255, 136, 0))';
        });
    });

    // CTA button ripple effect
    const ctaButton = document.querySelector('.cta-section .btn-primary');
    if (ctaButton) {
        ctaButton.addEventListener('click', function(e) {
            const ripple = document.createElement('span');
            const rect = this.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            const x = e.clientX - rect.left - size / 2;
            const y = e.clientY - rect.top - size / 2;

            ripple.style.width = ripple.style.height = size + 'px';
            ripple.style.left = x + 'px';
            ripple.style.top = y + 'px';
            ripple.classList.add('ripple');
            this.appendChild(ripple);

            setTimeout(() => ripple.remove(), 600);
        });
    }
});

// Prevent scroll jank with passive event listeners
window.addEventListener('scroll', () => {}, { passive: true });
window.addEventListener('mousemove', () => {}, { passive: true });
