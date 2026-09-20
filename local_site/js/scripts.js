Rot13 = {
	map: null,
	convert: function (a) {
		Rot13.init();
		var s = "";
		for (i = 0; i < a.length; i++) {
			var b = a.charAt(i);
			s += ((b >= 'A' && b <= 'Z') || (b >= 'a' && b <= 'z') ? Rot13.map[b] : b);
		}
		return s;
	},
	init: function () {
		if (Rot13.map != null) return;
		var map = new Array();
		var s = "abcdefghijklmnopqrstuvwxyz";
		for (i = 0; i < s.length; i++)
			map[s.charAt(i)] = s.charAt((i + 13) % 26);
		for (i = 0; i < s.length; i++)
			map[s.charAt(i).toUpperCase()] = s.charAt((i + 13) % 26).toUpperCase();
		Rot13.map = map;
	}
}

jQuery.fn.renderTeX = function () {
	var id = this.prop('id');
	MathJax.Hub.Queue(["Typeset", MathJax.Hub, id]);
	return this;
}

jQuery.fn.searchtabs = function () {
	// Search form tabs:
	var tab_event = function (event) {
		var tab = $(this);
		var form = $(event.data.form);
		if (tab.hasClass('ui-state-active')) {
			form.hide('fast');
			tab.removeClass('ui-tabs-selected').removeClass('ui-state-active');
		} else {
			form.show('fast');
			tab.addClass('ui-tabs-selected').addClass('ui-state-active');
		}
		tab.find('a').blur();
		return false;
	}

	// Do something about this:
	this.addClass('ui-tabs ui-widget ui-widget-content ui-corner-all');
	this.children('ul').addClass('ui-tabs-nav ui-helper-reset ui-helper-clearfix ui-widget-header ui-corner-all');
	this.children('div').addClass('ui-tabs-panel ui-widget-content ui-corner-bottom');
	this.children('div:not(.search-tab-nohide)').hide();

	/*
		var tabs = tabdiv.find("ul:first").children('li');
		tabs.each(function(i,elm){
			var partner = tabdiv.children("div:not(.search-tab-nohide)").index(i);
			alert($(this).prop('id') + ' - ' + partner.prop('id'));
	
			$(this).addClass('ui-state-default ui-corner-top');
			$(this).children('a').addClass('ui-tabs-anchor');
	
			$(this).bind('click', {form: "#"+partner}, tab_event);
		}).hover(function(){
			$(this).addClass("ui-state-hover");
		}, function(){
			$(this).removeClass("ui-state-hover");
		});;
	*/
	return this;
}

function uncrypt_mail(href) {
	href = href.replace('/contact/', '');
	href = href.replace('/', String.fromCharCode(60 + 4));
	href = href.replace(/\+/gi, '.');
	href = Rot13.convert(href);
	return 'mai' + 'lto:' + href;
}

function UnCryptMail(name, domain) {
	var i = 0;
	var n = 0;
	var NewName = "";
	var NewDomain = "";
	for (i = 0; i < name.length; i++) {
		n = name.charCodeAt(i);
		if (n >= 8364) { n = 128; }
		NewName += String.fromCharCode(n - 2);
	}
	for (i = 0; i < domain.length; i++) {
		n = domain.charCodeAt(i);
		if (n >= 8364) { n = 128; }
		NewDomain += String.fromCharCode(n - 1);
	}
	return "mai" + "lto:" + NewName + String.fromCharCode(60 + 4) + NewDomain;
}

function is_int(n) {
	return Number(n) == n && Number(n) % 1 === 0;
}

// Fix the page width to the width of the element
function fixpagewidth(selector) {
	var owidth = $(selector).outerWidth(true);
	if (owidth > $("#main").width()) {
		$("#main").width(owidth + 30);
	}
}

function dec2hms(value) {
	var hours = value / 15;
	var hours_floor = Math.floor(hours);
	var min = (hours - hours_floor) * 60;
	var min_floor = Math.floor(min);
	var sec = (min - min_floor) * 60;
	var sec_floor = Math.round(sec);

	return ((hours_floor < 10) ? "0" + hours_floor : hours_floor) + ":" + ((min_floor < 10) ? "0" + min_floor : min_floor) + ":" + ((sec_floor < 10) ? "0" + sec_floor : sec_floor);
}

function dec2dms(val, decimals = 0) {
	var sign = (parseFloat(val) < 0) ? '-' : '';
	var degree = Math.abs(parseFloat(val));
	var degree_floor = Math.floor(degree);
	var min = (degree - degree_floor) * 60;
	var min_floor = Math.floor(min);
	var sec = (min - min_floor) * 60;
	var sec_floor = (decimals == 0) ? Math.floor(sec) : sec.toFixed(decimals);

	return sign + ((degree_floor < 10) ? "0" + degree_floor : degree_floor) + ":" + ((min_floor < 10) ? "0" + min_floor : min_floor) + ":" + ((sec_floor < 10) ? "0" + sec_floor : sec_floor);
}

function dms2dec(val) {
	const regex = /([+-]?)\s*([0-9]+)\s*:\s*([0-5]?[0-9])\s*:\s*([0-5]?[0-9]([,\.][0-9]*)?)/i;
	var m = String(val).trim().match(regex);
	if (m === null) {
		return null;
	}
	var sign = (m[1] == '-') ? -1 : 1;
	var val = parseFloat(m[2]) + parseFloat(m[3]) / 60 + parseFloat(m[4].replace(',', '.')) / 3600;
	return (val == 0) ? val : sign * val;
}

$.extend({
	URLEncode: function (c) {
		var o = ''; var x = 0; c = c.toString(); var r = /(^[a-zA-Z0-9_.]*)/;
		while (x < c.length) {
			var m = r.exec(c.substr(x));
			if (m != null && m.length > 1 && m[1] != '') {
				o += m[1]; x += m[1].length;
			} else {
				if (c[x] == ' ') o += '+'; else {
					var d = c.charCodeAt(x); var h = d.toString(16);
					o += '%' + (h.length < 2 ? '0' : '') + h.toUpperCase();
				} x++;
			}
		} return o;
	},
	URLDecode: function (s) {
		var o = s; var binVal, t; var r = /(%[^%]{2})/;
		while ((m = r.exec(o)) != null && m.length > 1 && m[1] != '') {
			b = parseInt(m[1].substr(1), 16);
			t = String.fromCharCode(b); o = o.replace(m[1], t);
		} return o;
	}
});

var tab = 0;
function showsubmenu() {
	if (tab) tab.css('visibility', 'visible');
}
function hidesubmenu() {
	$("#topmenuul li ul").css('visibility', 'hidden');
}

$(document).ready(function () {
	// Top Menu:
	var show_time = 0, hide_time = 0;
	$('#topmenuul li').mouseover(function () {
		var link = $(this).children('a');
		if (!link.hasClass('topmenu-active')) link.addClass('ui-state-hover');
		if ($(this).children('ul').length > 0) {
			var pos = $(this).position();
			oldtab = tab;
			tab = $(this).children('ul').css({ 'left': pos.left, 'top': pos.top + 22 });
			show_time = window.setTimeout(showsubmenu, 500);
			if (hide_time && $(this).parent('ul').attr('id') != oldtab.attr('id')) {
				window.clearTimeout(hide_time);
			}
		}

	}).mouseout(function () {
		var link = $(this).children('a');
		if (!link.hasClass('topmenu-active')) link.removeClass('ui-state-hover');

		if ($(this).children('ul').length > 0) {
			hide_time = window.setTimeout(hidesubmenu, 500);
			if (show_time) window.clearTimeout(show_time);
		}
	});

	$("#mobile_menu_link").click(function (e) {
		$(this).parent('li').toggleClass('topmenu-active-mobile-menu');
		$("#sidebar").stop(true).toggle("slow");
		$(this).blur();
		e.preventDefault();
		return false;
	});

	$('#topmenuul li ul li a').mouseover(function () {
		$(this).addClass('ui-state-hover');
	}).mouseout(function () {
		$(this).removeClass('ui-state-hover');
	});

	// Detection of CapsLock on password fields:
	$('input[type="password"]').keypress(function (e) {
		var s = String.fromCharCode(e.which);
		var a = $('div.capsalert');
		if ((s.toUpperCase() === s && s.toLowerCase() !== s && !e.shiftKey) || (s.toUpperCase() !== s && s.toLowerCase() === s && e.shiftKey)) {
			// CapsLock is active
			if (a.length < 1) {
				var a = $('<div class="error ui-corner-all capsalert" style=\"width:120px;position:absolute;background-color:orange;\">CapsLock is on!</div>');
				$(this).after(a);
			}
			a.position({
				my: 'left center',
				at: 'right center',
				of: $(this),
				offset: '5 0'
			});
		} else {
			// CapsLock is not active
			if (a.length > 0) a.remove();
		}
	});

	// Decoration of buttons:
	$('input[type="submit"],input[type="button"],input[type="reset"],button').mouseover(function () {
		$(this).addClass("ui-state-hover");
	}).mouseout(function () {
		$(this).removeClass("ui-state-hover ui-state-active");
	}).mousedown(function () {
		$(this).addClass("ui-state-active");
	}).mouseup(function () {
		$(this).removeClass("ui-state-active");
	}).addClass("ui-button ui-state-default ui-corner-all");

	// Decoration of links:
	$('a.external').each(function () {
		$(this).prop('target', '_blank');
		var htm = $(this).html();
		$(this).html(htm + '<span class="ui-icon ui-icon-extlink" style="display:inline-block;vertical-align:middle;margin:0;padding;0;border:0;"></span>');
	});
	$('a.mail_link').each(function () {
		var href = $(this).attr('href');
		$(this).prop('href', uncrypt_mail(href));
	});
	$("#footeremail").click(function () {
		$(this).prop('href', UnCryptMail('tcuowuj', 'qizt/bv/el'));
		return true;
	});

	// Table sorter:
	$.tablesorter.addParser({
		// Set a unique id:
		id: 'filesize',
		is: function (s) {
			return s.match(new RegExp(/[0-9]+(\.[0-9]+)?\ (kB|byte|GB|MB|TB)/));
		},
		format: function (s) {
			var suf = s.match(new RegExp(/(kB|byte|GB|MB|TB)$/))[1];
			var num = parseFloat(s.match(new RegExp(/^[0-9]+(\.[0-9]+)?/))[0]);
			switch (suf) {
				case 'byte':
					return num;
				case 'kB':
					return num * 1024;
				case 'MB':
					return num * 1024 * 1024;
				case 'GB':
					return num * 1024 * 1024 * 1024;
				case 'TB':
					return num * 1024 * 1024 * 1024 * 1024;
			}
		},
		// Set type, either numeric or text:
		type: 'numeric'
	});
	$.tablesorter.addParser({
		id: 'season',
		is: function (s) {
			return s.match(new RegExp(/^(Q|C)([0-9]+)(\.[0-9]+)?$/));
		},
		format: function (s) {
			var m = s.match(new RegExp(/^(Q|C)([0-9]+)\.?([0-9]+)?$/));
			if (m[1] == 'Q') {
				return parseFloat(parseInt(m[2]) + parseInt(m[3]) / 10);
			} else {
				// Where was 17 quarters in nominal Kepler, plus K2ENG so add 19 to K2 campaigns:
				return parseFloat(19 + parseInt(m[2]) + parseInt(m[3]) / 10);
			}
		},
		type: 'numeric'
	});

	//--------------------------------------------------------------------------------------
    // Script to create the slider on HOME page
    //--------------------------------------------------------------------------------------

    let index = 0;
    const $slides = $(".slide-home");
    const totalSlides = $slides.length;
    const $wrapper = $(".slider-wrapper-home");
   
    function showSlide(i) {
        if (i >= totalSlides) index = 0;
        if (i < 0) index = totalSlides - 1;
        $wrapper.css("transform", `translateX(-${index * 100}%)`);
    }

     // Next button click event
    $("#nextBtnSlide").click(function () {
        index++;
        showSlide(index);
    });

    // Previous button click event
    $("#prevBtnSlide").click(function () {
        index--;
        showSlide(index);
    });
});
