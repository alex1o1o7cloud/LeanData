import Mathlib

namespace NUMINAMATH_GPT_meal_cost_is_correct_l2066_206607

def samosa_quantity : ℕ := 3
def samosa_price : ℝ := 2
def pakora_quantity : ℕ := 4
def pakora_price : ℝ := 3
def mango_lassi_quantity : ℕ := 1
def mango_lassi_price : ℝ := 2
def biryani_quantity : ℕ := 2
def biryani_price : ℝ := 5.5
def naan_quantity : ℕ := 1
def naan_price : ℝ := 1.5

def tip_rate : ℝ := 0.18
def sales_tax_rate : ℝ := 0.07

noncomputable def total_meal_cost : ℝ :=
  let subtotal := (samosa_quantity * samosa_price) + (pakora_quantity * pakora_price) +
                  (mango_lassi_quantity * mango_lassi_price) + (biryani_quantity * biryani_price) +
                  (naan_quantity * naan_price)
  let sales_tax := subtotal * sales_tax_rate
  let total_before_tip := subtotal + sales_tax
  let tip := total_before_tip * tip_rate
  total_before_tip + tip

theorem meal_cost_is_correct : total_meal_cost = 41.04 := by
  sorry

end NUMINAMATH_GPT_meal_cost_is_correct_l2066_206607


namespace NUMINAMATH_GPT_average_speed_remaining_l2066_206613

theorem average_speed_remaining (D : ℝ) : 
    (0.4 * D / 40 + 0.6 * D / S) = D / 50 → S = 60 :=
by 
  sorry

end NUMINAMATH_GPT_average_speed_remaining_l2066_206613


namespace NUMINAMATH_GPT_original_volume_of_ice_l2066_206660

variable (V : ℝ) 

theorem original_volume_of_ice (h1 : V * (1 / 4) * (1 / 4) = 0.25) : V = 4 :=
  sorry

end NUMINAMATH_GPT_original_volume_of_ice_l2066_206660


namespace NUMINAMATH_GPT_smallest_five_digit_congruent_to_three_mod_seventeen_l2066_206698

theorem smallest_five_digit_congruent_to_three_mod_seventeen :
  ∃ (n : ℤ), 10000 ≤ n ∧ n < 100000 ∧ n % 17 = 3 ∧ n = 10012 :=
by
  sorry

end NUMINAMATH_GPT_smallest_five_digit_congruent_to_three_mod_seventeen_l2066_206698


namespace NUMINAMATH_GPT_smallest_solution_x_squared_abs_x_eq_3x_plus_4_l2066_206629

theorem smallest_solution_x_squared_abs_x_eq_3x_plus_4 :
  ∃ x : ℝ, x^2 * |x| = 3 * x + 4 ∧ ∀ y : ℝ, (y^2 * |y| = 3 * y + 4 → y ≥ x) := 
sorry

end NUMINAMATH_GPT_smallest_solution_x_squared_abs_x_eq_3x_plus_4_l2066_206629


namespace NUMINAMATH_GPT_ratio_of_areas_of_concentric_circles_l2066_206673

theorem ratio_of_areas_of_concentric_circles
  (C1 C2 : ℝ)
  (h : (30 / 360) * C1 = (24 / 360) * C2) :
  (π * (C1 / (2 * π))^2) / (π * (C2 / (2 * π))^2) = 16 / 25 := by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_of_concentric_circles_l2066_206673


namespace NUMINAMATH_GPT_min_generic_tees_per_package_l2066_206688

def total_golf_tees_needed (n : ℕ) : ℕ := 80
def max_generic_packages_used : ℕ := 2
def tees_per_aero_flight_package : ℕ := 2
def aero_flight_packages_needed : ℕ := 28
def total_tees_from_aero_flight_packages (n : ℕ) : ℕ := aero_flight_packages_needed * tees_per_aero_flight_package

theorem min_generic_tees_per_package (G : ℕ) :
  (total_golf_tees_needed 4) - (total_tees_from_aero_flight_packages aero_flight_packages_needed) ≤ max_generic_packages_used * G → G ≥ 12 :=
by
  sorry

end NUMINAMATH_GPT_min_generic_tees_per_package_l2066_206688


namespace NUMINAMATH_GPT_equilateral_triangle_side_length_l2066_206643

theorem equilateral_triangle_side_length 
    (D A B C : ℝ × ℝ)
    (h_distances : dist D A = 2 ∧ dist D B = 3 ∧ dist D C = 5)
    (h_equilateral : dist A B = dist B C ∧ dist B C = dist C A) :
    dist A B = Real.sqrt 19 :=
by
    sorry -- Proof to be filled

end NUMINAMATH_GPT_equilateral_triangle_side_length_l2066_206643


namespace NUMINAMATH_GPT_range_of_c_l2066_206641

theorem range_of_c (a b c : ℝ) (h1 : 6 < a) (h2 : a < 10) (h3 : a / 2 ≤ b) (h4 : b ≤ 2 * a) (h5 : c = a + b) : 
  9 < c ∧ c < 30 :=
sorry

end NUMINAMATH_GPT_range_of_c_l2066_206641


namespace NUMINAMATH_GPT_solve_quadratic_eq_l2066_206670

theorem solve_quadratic_eq (x : ℝ) : x ^ 2 + 2 * x - 5 = 0 → (x = -1 + Real.sqrt 6 ∨ x = -1 - Real.sqrt 6) :=
by 
  intro h
  sorry

end NUMINAMATH_GPT_solve_quadratic_eq_l2066_206670


namespace NUMINAMATH_GPT_valid_fraction_l2066_206684

theorem valid_fraction (x: ℝ) : x^2 + 1 ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_valid_fraction_l2066_206684


namespace NUMINAMATH_GPT_grayson_fraction_l2066_206612

variable (A G O : ℕ) -- The number of boxes collected by Abigail, Grayson, and Olivia, respectively
variable (C_per_box : ℕ) -- The number of cookies per box
variable (TotalCookies : ℕ) -- The total number of cookies collected by Abigail, Grayson, and Olivia

-- Given conditions
def abigail_boxes : ℕ := 2
def olivia_boxes : ℕ := 3
def cookies_per_box : ℕ := 48
def total_cookies : ℕ := 276

-- Prove the fraction of the box that Grayson collected
theorem grayson_fraction :
  G * C_per_box = TotalCookies - (abigail_boxes + olivia_boxes) * cookies_per_box → 
  G / C_per_box = 3 / 4 := 
by
  sorry

-- Assume the variables from conditions
variable (G : ℕ := 36 / 48)
variable (TotalCookies := 276)
variable (C_per_box := 48)
variable (A := 2)
variable (O := 3)


end NUMINAMATH_GPT_grayson_fraction_l2066_206612


namespace NUMINAMATH_GPT_solve_inequality_l2066_206689

theorem solve_inequality (x : ℝ) :
  x * Real.log (x^2 + x + 1) / Real.log 10 < 0 ↔ x < -1 :=
sorry

end NUMINAMATH_GPT_solve_inequality_l2066_206689


namespace NUMINAMATH_GPT_find_x_l2066_206626

def cube_volume (s : ℝ) := s^3
def cube_surface_area (s : ℝ) := 6 * s^2

theorem find_x (x : ℝ) (s : ℝ) 
  (hv : cube_volume s = 7 * x)
  (hs : cube_surface_area s = x) : 
  x = 42 := 
by
  sorry

end NUMINAMATH_GPT_find_x_l2066_206626


namespace NUMINAMATH_GPT_santino_total_fruits_l2066_206663

theorem santino_total_fruits :
  let p := 2
  let m := 3
  let a := 4
  let o := 5
  let fp := 10
  let fm := 20
  let fa := 15
  let fo := 25
  p * fp + m * fm + a * fa + o * fo = 265 := by
  sorry

end NUMINAMATH_GPT_santino_total_fruits_l2066_206663


namespace NUMINAMATH_GPT_derivative_at_pi_over_3_l2066_206624

noncomputable def f (x : Real) : Real := 2 * Real.sin x + Real.sqrt 3 * Real.cos x

theorem derivative_at_pi_over_3 : (deriv f) (π / 3) = -1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_derivative_at_pi_over_3_l2066_206624


namespace NUMINAMATH_GPT_quadratic_properties_l2066_206664

def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_properties 
  (a b c : ℝ) (ha : a ≠ 0) (h_passes_through : quadratic_function a b c 0 = 1) (h_unique_zero : quadratic_function a b c (-1) = 0) :
  quadratic_function a b c = quadratic_function 1 2 1 ∧ 
  (∀ k, ∃ g,
    (k ≤ -2 → g = k + 3) ∧ 
    (-2 < k ∧ k ≤ 6 → g = -((k^2 - 4*k) / 4)) ∧ 
    (6 < k → g = 9 - 2*k)) :=
sorry

end NUMINAMATH_GPT_quadratic_properties_l2066_206664


namespace NUMINAMATH_GPT_solve_abs_quadratic_l2066_206699

theorem solve_abs_quadratic :
  ∃ x : ℝ, (|x - 3| + x^2 = 10) ∧ 
  (x = (-1 + Real.sqrt 53) / 2 ∨ x = (1 + Real.sqrt 29) / 2 ∨ x = (1 - Real.sqrt 29) / 2) :=
by sorry

end NUMINAMATH_GPT_solve_abs_quadratic_l2066_206699


namespace NUMINAMATH_GPT_fg_of_2_l2066_206675

-- Define the functions f and g
def f (x : ℝ) : ℝ := 5 - 4 * x
def g (x : ℝ) : ℝ := x^2 + 2

-- Prove the specific property
theorem fg_of_2 : f (g 2) = -19 := by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_fg_of_2_l2066_206675


namespace NUMINAMATH_GPT_earphone_cost_correct_l2066_206616

-- Given conditions
def mean_expenditure : ℕ := 500

def expenditure_mon : ℕ := 450
def expenditure_tue : ℕ := 600
def expenditure_wed : ℕ := 400
def expenditure_thu : ℕ := 500
def expenditure_sat : ℕ := 550
def expenditure_sun : ℕ := 300

def pen_cost : ℕ := 30
def notebook_cost : ℕ := 50

-- Goal: cost of the earphone
def total_expenditure_week : ℕ := 7 * mean_expenditure
def expenditure_6days : ℕ := expenditure_mon + expenditure_tue + expenditure_wed + expenditure_thu + expenditure_sat + expenditure_sun
def expenditure_fri : ℕ := total_expenditure_week - expenditure_6days
def expenditure_fri_items : ℕ := pen_cost + notebook_cost
def earphone_cost : ℕ := expenditure_fri - expenditure_fri_items

theorem earphone_cost_correct :
  earphone_cost = 620 :=
by
  sorry

end NUMINAMATH_GPT_earphone_cost_correct_l2066_206616


namespace NUMINAMATH_GPT_initial_percentage_reduction_l2066_206625

theorem initial_percentage_reduction (x : ℝ) :
  (1 - x / 100) * 1.17649 = 1 → x = 15 :=
by
  sorry

end NUMINAMATH_GPT_initial_percentage_reduction_l2066_206625


namespace NUMINAMATH_GPT_max_hours_wednesday_l2066_206601

theorem max_hours_wednesday (x : ℕ) 
    (h1 : ∀ (d w : ℕ), w = x → d = x → d + w + (x + 3) = 3 * 3) 
    (h2 : ∀ (a b c : ℕ), a = b → b = c → (a + b + (c + 3))/3 = 3) :
  x = 2 := 
by
  sorry

end NUMINAMATH_GPT_max_hours_wednesday_l2066_206601


namespace NUMINAMATH_GPT_train_crossing_time_l2066_206615

theorem train_crossing_time:
  ∀ (length_train : ℝ) (speed_man_kmph : ℝ) (speed_train_kmph : ℝ),
    length_train = 125 →
    speed_man_kmph = 5 →
    speed_train_kmph = 69.994 →
    (125 / ((69.994 + 5) * (1000 / 3600))) = 6.002 :=
by
  intros length_train speed_man_kmph speed_train_kmph h1 h2 h3
  sorry

end NUMINAMATH_GPT_train_crossing_time_l2066_206615


namespace NUMINAMATH_GPT_integer_solution_unique_l2066_206649

theorem integer_solution_unique
  (a b c d : ℤ)
  (h : a^2 + 5 * b^2 - 2 * c^2 - 2 * c * d - 3 * d^2 = 0) :
  a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 :=
sorry

end NUMINAMATH_GPT_integer_solution_unique_l2066_206649


namespace NUMINAMATH_GPT_solution_set_of_inequalities_l2066_206678

-- Definitions
def is_even_function (f : ℝ → ℝ) := ∀ x, f (-x) = f x

def is_periodic (f : ℝ → ℝ) (p : ℝ) := ∀ x, f (x + p) = f x

def is_strictly_decreasing (f : ℝ → ℝ) (a b : ℝ) := ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

-- Main Statement
theorem solution_set_of_inequalities
  (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_periodic : is_periodic f 2)
  (h_decreasing : is_strictly_decreasing f 0 1)
  (h_f_pi : f π = 1)
  (h_f_2pi : f (2 * π) = 2) :
  {x : ℝ | 1 ≤ x ∧ x ≤ 2 ∧ 1 ≤ f x ∧ f x ≤ 2} = {x | π - 2 ≤ x ∧ x ≤ 8 - 2 * π} :=
  sorry

end NUMINAMATH_GPT_solution_set_of_inequalities_l2066_206678


namespace NUMINAMATH_GPT_students_drawn_from_grade10_l2066_206691

-- Define the initial conditions
def total_students_grade12 : ℕ := 750
def total_students_grade11 : ℕ := 850
def total_students_grade10 : ℕ := 900
def sample_size : ℕ := 50

-- Prove the number of students drawn from grade 10 is 18
theorem students_drawn_from_grade10 : 
  total_students_grade12 = 750 ∧
  total_students_grade11 = 850 ∧
  total_students_grade10 = 900 ∧
  sample_size = 50 →
  (sample_size * total_students_grade10 / 
  (total_students_grade12 + total_students_grade11 + total_students_grade10) = 18) :=
by
  sorry

end NUMINAMATH_GPT_students_drawn_from_grade10_l2066_206691


namespace NUMINAMATH_GPT_factorize_cubic_expression_l2066_206620

theorem factorize_cubic_expression (m : ℝ) : 
  m^3 - 16 * m = m * (m + 4) * (m - 4) :=
by
  sorry

end NUMINAMATH_GPT_factorize_cubic_expression_l2066_206620


namespace NUMINAMATH_GPT_distance_AC_l2066_206650

theorem distance_AC (A B C : ℤ) (h₁ : abs (B - A) = 5) (h₂ : abs (C - B) = 3) : abs (C - A) = 2 ∨ abs (C - A) = 8 :=
sorry

end NUMINAMATH_GPT_distance_AC_l2066_206650


namespace NUMINAMATH_GPT_total_amount_paid_l2066_206686

-- Definitions of the conditions
def cost_earbuds : ℝ := 200
def tax_rate : ℝ := 0.15

-- Statement to prove
theorem total_amount_paid : (cost_earbuds + (cost_earbuds * tax_rate)) = 230 := sorry

end NUMINAMATH_GPT_total_amount_paid_l2066_206686


namespace NUMINAMATH_GPT_largest_integer_solution_of_inequality_l2066_206603

theorem largest_integer_solution_of_inequality :
  ∃ x : ℤ, x < 2 ∧ (∀ y : ℤ, y < 2 → y ≤ x) ∧ -x + 3 > 1 :=
sorry

end NUMINAMATH_GPT_largest_integer_solution_of_inequality_l2066_206603


namespace NUMINAMATH_GPT_nails_needed_l2066_206680

theorem nails_needed (nails_own nails_found nails_total_needed : ℕ) 
  (h1 : nails_own = 247) 
  (h2 : nails_found = 144) 
  (h3 : nails_total_needed = 500) : 
  nails_total_needed - (nails_own + nails_found) = 109 := 
by
  sorry

end NUMINAMATH_GPT_nails_needed_l2066_206680


namespace NUMINAMATH_GPT_find_sum_of_squares_l2066_206694

variable (x y : ℝ)

theorem find_sum_of_squares (h₁ : x * y = 8) (h₂ : x^2 * y + x * y^2 + x + y = 94) : 
  x^2 + y^2 = 7540 / 81 :=
by
  sorry

end NUMINAMATH_GPT_find_sum_of_squares_l2066_206694


namespace NUMINAMATH_GPT_find_b_l2066_206685

theorem find_b (b : ℝ) : (∃ x y : ℝ, x = 1 ∧ y = 2 ∧ y = 2 * x + b) → b = 0 := by
  sorry

end NUMINAMATH_GPT_find_b_l2066_206685


namespace NUMINAMATH_GPT_domain_implies_range_a_range_implies_range_a_l2066_206638

theorem domain_implies_range_a {a : ℝ} :
  (∀ x : ℝ, ax^2 + 2 * a * x + 1 > 0) → 0 ≤ a ∧ a < 1 :=
sorry

theorem range_implies_range_a {a : ℝ} :
  (∀ y : ℝ, ∃ x : ℝ, ax^2 + 2 * a * x + 1 = y) → 1 ≤ a :=
sorry

end NUMINAMATH_GPT_domain_implies_range_a_range_implies_range_a_l2066_206638


namespace NUMINAMATH_GPT_max_n_for_factorable_quadratic_l2066_206608

theorem max_n_for_factorable_quadratic :
  ∃ n : ℤ, (∀ x : ℤ, ∃ A B : ℤ, (3*x^2 + n*x + 108) = (3*x + A)*( x + B) ∧ A*B = 108 ∧ n = A + 3*B) ∧ n = 325 :=
by
  sorry

end NUMINAMATH_GPT_max_n_for_factorable_quadratic_l2066_206608


namespace NUMINAMATH_GPT_ticket_cost_correct_l2066_206621

def metro_sells (tickets_per_minute : ℕ) (minutes : ℕ) : ℕ :=
  tickets_per_minute * minutes

def total_earnings (tickets_sold : ℕ) (ticket_cost : ℕ) : ℕ :=
  tickets_sold * ticket_cost

theorem ticket_cost_correct (ticket_cost : ℕ) : 
  (metro_sells 5 6 = 30) ∧ (total_earnings 30 ticket_cost = 90) → ticket_cost = 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_ticket_cost_correct_l2066_206621


namespace NUMINAMATH_GPT_gcd_exponent_min_speed_for_meeting_game_probability_difference_l2066_206619

-- Problem p4
theorem gcd_exponent (a b : ℕ) (h1 : a = 6) (h2 : b = 9) (h3 : gcd a b = 3) : gcd (2^a - 1) (2^b - 1) = 7 := by
  sorry

-- Problem p5
theorem min_speed_for_meeting (v_S s : ℚ) (h : v_S = 1/2) : ∀ (s : ℚ), (s - v_S) ≥ 1 → s = 3/2 := by
  sorry

-- Problem p6
theorem game_probability_difference (N : ℕ) (p : ℚ) (h1 : N = 1) (h2 : p = 5/16) : N + p = 21/16 := by
  sorry

end NUMINAMATH_GPT_gcd_exponent_min_speed_for_meeting_game_probability_difference_l2066_206619


namespace NUMINAMATH_GPT_ratio_jordana_jennifer_10_years_l2066_206693

-- Let's define the necessary terms and conditions:
def Jennifer_future_age := 30
def Jordana_current_age := 80
def years := 10

-- Define the ratio of ages function:
noncomputable def ratio_of_ages (future_age_jen : ℕ) (current_age_jord : ℕ) (yrs : ℕ) : ℚ :=
  (current_age_jord + yrs) / future_age_jen

-- The statement we need to prove:
theorem ratio_jordana_jennifer_10_years :
  ratio_of_ages Jennifer_future_age Jordana_current_age years = 3 := by
  sorry

end NUMINAMATH_GPT_ratio_jordana_jennifer_10_years_l2066_206693


namespace NUMINAMATH_GPT_second_race_distance_remaining_l2066_206656

theorem second_race_distance_remaining
  (race_distance : ℕ)
  (A_finish_time : ℕ)
  (B_remaining_distance : ℕ)
  (A_start_behind : ℕ)
  (A_speed : ℝ)
  (B_speed : ℝ)
  (A_distance_second_race : ℕ)
  (B_distance_second_race : ℝ)
  (v_ratio : ℝ)
  (B_remaining_second_race : ℝ) :
  race_distance = 10000 →
  A_finish_time = 50 →
  B_remaining_distance = 500 →
  A_start_behind = 500 →
  A_speed = race_distance / A_finish_time →
  B_speed = (race_distance - B_remaining_distance) / A_finish_time →
  v_ratio = A_speed / B_speed →
  v_ratio = 20 / 19 →
  A_distance_second_race = race_distance + A_start_behind →
  B_distance_second_race = B_speed * (A_distance_second_race / A_speed) →
  B_remaining_second_race = race_distance - B_distance_second_race →
  B_remaining_second_race = 25 := 
by
  sorry

end NUMINAMATH_GPT_second_race_distance_remaining_l2066_206656


namespace NUMINAMATH_GPT_number_of_trees_l2066_206667

theorem number_of_trees (length_of_yard : ℕ) (distance_between_trees : ℕ) 
(h1 : length_of_yard = 273) 
(h2 : distance_between_trees = 21) : 
(length_of_yard / distance_between_trees) + 1 = 14 := by
  sorry

end NUMINAMATH_GPT_number_of_trees_l2066_206667


namespace NUMINAMATH_GPT_martha_pins_l2066_206604

theorem martha_pins (k : ℕ) :
  (2 + 9 * k > 45) ∧ (2 + 14 * k < 90) ↔ (k = 5 ∨ k = 6) :=
by
  sorry

end NUMINAMATH_GPT_martha_pins_l2066_206604


namespace NUMINAMATH_GPT_rostov_survey_min_players_l2066_206610

theorem rostov_survey_min_players :
  ∃ m : ℕ, (∀ n : ℕ, n < m → (95 + n * 1) % 100 ≠ 0) ∧ m = 11 :=
sorry

end NUMINAMATH_GPT_rostov_survey_min_players_l2066_206610


namespace NUMINAMATH_GPT_remainder_of_m_div_5_l2066_206633

theorem remainder_of_m_div_5 (m n : ℕ) (hpos : 0 < m) (hdef : m = 15 * n - 1) : m % 5 = 4 :=
sorry

end NUMINAMATH_GPT_remainder_of_m_div_5_l2066_206633


namespace NUMINAMATH_GPT_motorcyclist_average_speed_l2066_206653

theorem motorcyclist_average_speed :
  let distance_ab := 120
  let speed_ab := 45
  let distance_bc := 130
  let speed_bc := 60
  let distance_cd := 150
  let speed_cd := 50
  let time_ab := distance_ab / speed_ab
  let time_bc := distance_bc / speed_bc
  let time_cd := distance_cd / speed_cd
  (time_ab = time_bc + 2)
  → (time_cd = time_ab / 2)
  → avg_speed = (distance_ab + distance_bc + distance_cd) / (time_ab + time_bc + time_cd)
  → avg_speed = 2400 / 47 := sorry

end NUMINAMATH_GPT_motorcyclist_average_speed_l2066_206653


namespace NUMINAMATH_GPT_total_cost_l2066_206627

def copper_pipe_length := 10
def plastic_pipe_length := 15
def copper_pipe_cost_per_meter := 5
def plastic_pipe_cost_per_meter := 3

theorem total_cost (h₁ : copper_pipe_length = 10)
                   (h₂ : plastic_pipe_length = 15)
                   (h₃ : copper_pipe_cost_per_meter = 5)
                   (h₄ : plastic_pipe_cost_per_meter = 3) :
  10 * 5 + 15 * 3 = 95 :=
by sorry

end NUMINAMATH_GPT_total_cost_l2066_206627


namespace NUMINAMATH_GPT_meetings_percentage_l2066_206617

def total_minutes_in_day (hours: ℕ): ℕ := hours * 60
def first_meeting_duration: ℕ := 60
def second_meeting_duration (first_meeting_duration: ℕ): ℕ := 3 * first_meeting_duration
def total_meeting_duration (first_meeting_duration: ℕ) (second_meeting_duration: ℕ): ℕ := first_meeting_duration + second_meeting_duration
def percentage_of_workday_spent_in_meetings (total_meeting_duration: ℕ) (total_minutes_in_day: ℕ): ℚ := (total_meeting_duration / total_minutes_in_day) * 100

theorem meetings_percentage (hours: ℕ) (first_meeting_duration: ℕ) (second_meeting_duration: ℕ) (total_meeting_duration: ℕ) (total_minutes_in_day: ℕ) 
(h1: total_minutes_in_day = 600) 
(h2: first_meeting_duration = 60) 
(h3: second_meeting_duration = 180) 
(h4: total_meeting_duration = 240):
percentage_of_workday_spent_in_meetings total_meeting_duration total_minutes_in_day = 40 := by
  sorry

end NUMINAMATH_GPT_meetings_percentage_l2066_206617


namespace NUMINAMATH_GPT_coloring_probability_l2066_206637

-- Definition of the problem and its conditions
def num_cells := 16
def num_diags := 2
def chosen_diags := 7

-- Define the probability
noncomputable def prob_coloring_correct : ℚ :=
  (num_diags ^ chosen_diags : ℚ) / (num_diags ^ num_cells)

-- The Lean theorem statement
theorem coloring_probability : prob_coloring_correct = 1 / 512 := 
by 
  unfold prob_coloring_correct
  -- The proof steps would follow here (omitted)
  sorry

end NUMINAMATH_GPT_coloring_probability_l2066_206637


namespace NUMINAMATH_GPT_evaluate_expression_l2066_206692

theorem evaluate_expression :
  (3^2016 + 3^2014 + 3^2012) / (3^2016 - 3^2014 + 3^2012) = 91 / 73 := 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2066_206692


namespace NUMINAMATH_GPT_probability_all_different_digits_l2066_206655

noncomputable def total_integers := 900
noncomputable def repeating_digits_integers := 9
noncomputable def same_digit_probability : ℚ := repeating_digits_integers / total_integers
noncomputable def different_digit_probability := 1 - same_digit_probability

theorem probability_all_different_digits :
  different_digit_probability = 99 / 100 :=
by
  sorry

end NUMINAMATH_GPT_probability_all_different_digits_l2066_206655


namespace NUMINAMATH_GPT_simplify_fraction_l2066_206648

variable {F : Type*} [Field F]

theorem simplify_fraction (a b : F) (h: a ≠ -1) :
  b / (a * b + b) = 1 / (a + 1) :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l2066_206648


namespace NUMINAMATH_GPT_find_number_l2066_206657

theorem find_number (x n : ℝ) (h1 : 0.12 / x * n = 12) (h2 : x = 0.1) : n = 10 := by
  sorry

end NUMINAMATH_GPT_find_number_l2066_206657


namespace NUMINAMATH_GPT_ab_bc_ca_lt_quarter_l2066_206682

theorem ab_bc_ca_lt_quarter (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h_sum : a + b + c = 1) :
  (a * b)^(5/4) + (b * c)^(5/4) + (c * a)^(5/4) < 1/4 :=
sorry

end NUMINAMATH_GPT_ab_bc_ca_lt_quarter_l2066_206682


namespace NUMINAMATH_GPT_painting_prices_l2066_206642

theorem painting_prices (P : ℝ) (h₀ : 55000 = 3.5 * P - 500) : 
  P = 15857.14 :=
by
  -- P represents the average price of the previous three paintings.
  -- Given the condition: 55000 = 3.5 * P - 500
  -- We need to prove: P = 15857.14
  sorry

end NUMINAMATH_GPT_painting_prices_l2066_206642


namespace NUMINAMATH_GPT_prove_inequalities_l2066_206639

noncomputable def x := Real.log Real.pi
noncomputable def y := Real.logb 5 2
noncomputable def z := Real.exp (-1 / 2)

theorem prove_inequalities : y < z ∧ z < x := by
  unfold x y z
  sorry

end NUMINAMATH_GPT_prove_inequalities_l2066_206639


namespace NUMINAMATH_GPT_total_students_eq_seventeen_l2066_206671

theorem total_students_eq_seventeen 
    (N : ℕ)
    (initial_students : N - 1 = 16)
    (avg_first_day : 77 * (N - 1) = 77 * 16)
    (avg_second_day : 78 * N = 78 * N)
    : N = 17 :=
sorry

end NUMINAMATH_GPT_total_students_eq_seventeen_l2066_206671


namespace NUMINAMATH_GPT_value_of_k_l2066_206690

theorem value_of_k (k : ℤ) : 
  (∀ x : ℤ, (x + k) * (x - 4) = x^2 - 4 * x + k * x - 4 * k ∧ 
  (k - 4) * x = 0) → k = 4 := 
by 
  sorry

end NUMINAMATH_GPT_value_of_k_l2066_206690


namespace NUMINAMATH_GPT_product_of_x_y_l2066_206651

theorem product_of_x_y (x y : ℝ) :
  (54 = 5 * y^2 + 20) →
  (8 * x^2 + 2 = 38) →
  x * y = Real.sqrt (30.6) :=
by
  intros h1 h2
  -- these would be the proof steps
  sorry

end NUMINAMATH_GPT_product_of_x_y_l2066_206651


namespace NUMINAMATH_GPT_main_theorem_l2066_206644

variable {a b c : ℝ}

noncomputable def inequality_1 (a b c : ℝ) : Prop :=
  a + b + c ≤ (a^2 + b^2) / (2 * c) + (b^2 + c^2) / (2 * a) + (c^2 + a^2) / (2 * b)

noncomputable def inequality_2 (a b c : ℝ) : Prop :=
  (a^2 + b^2) / (2 * c) + (b^2 + c^2) / (2 * a) + (c^2 + a^2) / (2 * b) ≤ a^2 / (b * c) + b^2 / (c * a) + c^2 / (a * b)

theorem main_theorem (h : a < 0 ∧ b < 0 ∧ c < 0) :
  inequality_1 a b c ∧ inequality_2 a b c := by sorry

end NUMINAMATH_GPT_main_theorem_l2066_206644


namespace NUMINAMATH_GPT_general_formula_seq_arithmetic_l2066_206602

variable (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ)

-- Conditions from the problem
axiom sum_condition (n : ℕ) : (1 - q) * S n + q^n = 1
axiom nonzero_q : q * (q - 1) ≠ 0
axiom arithmetic_S : S 3 + S 9 = 2 * S 6

-- Stating the proof goals
theorem general_formula (n : ℕ) : a n = q^(n-1) :=
sorry

theorem seq_arithmetic : a 2 + a 8 = 2 * a 5 :=
sorry

end NUMINAMATH_GPT_general_formula_seq_arithmetic_l2066_206602


namespace NUMINAMATH_GPT_value_of_nested_fraction_l2066_206683

theorem value_of_nested_fraction :
  10 + 5 + (1 / 2) * (9 + 5 + (1 / 2) * (8 + 5 + (1 / 2) * (7 + 5 + (1 / 2) * (6 + 5 + (1 / 2) * (5 + 5 + (1 / 2) * (4 + 5 + (1 / 2) * (3 + 5 ))))))) = 28 + (1 / 128) :=
sorry

end NUMINAMATH_GPT_value_of_nested_fraction_l2066_206683


namespace NUMINAMATH_GPT_solve_for_x_l2066_206611

theorem solve_for_x : ∃ x : ℚ, (1/4 : ℚ) + (1/x) = 7/8 ∧ x = 8/5 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_for_x_l2066_206611


namespace NUMINAMATH_GPT_chord_length_l2066_206668

theorem chord_length {r : ℝ} (h : r = 15) : 
  ∃ (CD : ℝ), CD = 26 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_chord_length_l2066_206668


namespace NUMINAMATH_GPT_simplify_expression_l2066_206640

theorem simplify_expression (h : (Real.pi / 2) < 2 ∧ 2 < Real.pi) : 
  Real.sqrt (1 - 2 * Real.sin 2 * Real.cos 2) = Real.sin 2 - Real.cos 2 :=
sorry

end NUMINAMATH_GPT_simplify_expression_l2066_206640


namespace NUMINAMATH_GPT_triangle_is_isosceles_l2066_206635

theorem triangle_is_isosceles (A B C a b c : ℝ) (h1 : c = 2 * a * Real.cos B) : 
  A = B → a = b := 
sorry

end NUMINAMATH_GPT_triangle_is_isosceles_l2066_206635


namespace NUMINAMATH_GPT_share_of_e_l2066_206658

variable (E F : ℝ)
variable (D : ℝ := (5/3) * E)
variable (D_alt : ℝ := (1/2) * F)
variable (E_alt : ℝ := (3/2) * F)
variable (profit : ℝ := 25000)

theorem share_of_e (h1 : D = (5/3) * E) (h2 : D = (1/2) * F) (h3 : E = (3/2) * F) :
  (E / ((5/2) * F + (3/2) * F + F)) * profit = 7500 :=
by
  sorry

end NUMINAMATH_GPT_share_of_e_l2066_206658


namespace NUMINAMATH_GPT_average_difference_l2066_206669

theorem average_difference :
  let avg1 := (20 + 40 + 60) / 3
  let avg2 := (10 + 70 + 28) / 3
  avg1 - avg2 = 4 :=
by
  -- Define the averages
  let avg1 := (20 + 40 + 60) / 3
  let avg2 := (10 + 70 + 28) / 3
  sorry

end NUMINAMATH_GPT_average_difference_l2066_206669


namespace NUMINAMATH_GPT_february_first_is_friday_l2066_206645

-- Definition of conditions
def february_has_n_mondays (n : ℕ) : Prop := n = 3
def february_has_n_fridays (n : ℕ) : Prop := n = 5

-- The statement to prove
theorem february_first_is_friday (n_mondays n_fridays : ℕ) (h_mondays : february_has_n_mondays n_mondays) (h_fridays : february_has_n_fridays n_fridays) : 
  (1 : ℕ) % 7 = 5 :=
by
  sorry

end NUMINAMATH_GPT_february_first_is_friday_l2066_206645


namespace NUMINAMATH_GPT_chocolate_ticket_fraction_l2066_206681

theorem chocolate_ticket_fraction (box_cost : ℝ) (ticket_count_per_free_box : ℕ) (ticket_count_included : ℕ) :
  ticket_count_per_free_box = 10 →
  ticket_count_included = 1 →
  (1 / 9 : ℝ) * box_cost =
  box_cost / ticket_count_per_free_box + box_cost / (ticket_count_per_free_box - ticket_count_included + 1) :=
by 
  intros h1 h2 
  have h : ticket_count_per_free_box = 10 := h1 
  have h' : ticket_count_included = 1 := h2 
  sorry

end NUMINAMATH_GPT_chocolate_ticket_fraction_l2066_206681


namespace NUMINAMATH_GPT_rate_of_stream_l2066_206697

def effectiveSpeedDownstream (v : ℝ) : ℝ := 36 + v
def effectiveSpeedUpstream (v : ℝ) : ℝ := 36 - v

theorem rate_of_stream (v : ℝ) (hf1 : effectiveSpeedUpstream v = 3 * effectiveSpeedDownstream v) : v = 18 := by
  sorry

end NUMINAMATH_GPT_rate_of_stream_l2066_206697


namespace NUMINAMATH_GPT_solve_for_x_l2066_206695

theorem solve_for_x (x : ℝ) (h₁ : x ≠ -3) :
  (7 * x^2 - 3) / (x + 3) - 3 / (x + 3) = 1 / (x + 3) ↔ x = 1 ∨ x = -1 := 
sorry

end NUMINAMATH_GPT_solve_for_x_l2066_206695


namespace NUMINAMATH_GPT_sqrt_neg4_sq_eq_4_l2066_206647

theorem sqrt_neg4_sq_eq_4 : Real.sqrt ((-4 : ℝ) ^ 2) = 4 := by
  sorry

end NUMINAMATH_GPT_sqrt_neg4_sq_eq_4_l2066_206647


namespace NUMINAMATH_GPT_retailer_selling_price_l2066_206623

theorem retailer_selling_price
  (cost_price_manufacturer : ℝ)
  (manufacturer_profit_rate : ℝ)
  (wholesaler_profit_rate : ℝ)
  (retailer_profit_rate : ℝ)
  (manufacturer_selling_price : ℝ)
  (wholesaler_selling_price : ℝ)
  (retailer_selling_price : ℝ)
  (h1 : cost_price_manufacturer = 17)
  (h2 : manufacturer_profit_rate = 0.18)
  (h3 : wholesaler_profit_rate = 0.20)
  (h4 : retailer_profit_rate = 0.25)
  (h5 : manufacturer_selling_price = cost_price_manufacturer + (manufacturer_profit_rate * cost_price_manufacturer))
  (h6 : wholesaler_selling_price = manufacturer_selling_price + (wholesaler_profit_rate * manufacturer_selling_price))
  (h7 : retailer_selling_price = wholesaler_selling_price + (retailer_profit_rate * wholesaler_selling_price)) :
  retailer_selling_price = 30.09 :=
by {
  sorry
}

end NUMINAMATH_GPT_retailer_selling_price_l2066_206623


namespace NUMINAMATH_GPT_find_a_l2066_206618

variable (a b : ℝ × ℝ) 

axiom b_eq : b = (2, -1)
axiom length_eq_one : ‖a + b‖ = 1
axiom parallel_x_axis : (a + b).snd = 0

theorem find_a : a = (-1, 1) ∨ a = (-3, 1) := by
  sorry

end NUMINAMATH_GPT_find_a_l2066_206618


namespace NUMINAMATH_GPT_find_sum_l2066_206636

theorem find_sum (a b : ℝ) (ha : a^3 - 3 * a^2 + 5 * a - 17 = 0) (hb : b^3 - 3 * b^2 + 5 * b + 11 = 0) :
  a + b = 2 :=
sorry

end NUMINAMATH_GPT_find_sum_l2066_206636


namespace NUMINAMATH_GPT_g_at_5_l2066_206606

def g (x : ℝ) : ℝ := 3*x^4 - 20*x^3 + 35*x^2 - 40*x + 24

theorem g_at_5 : g 5 = 74 := 
by {
  sorry
}

end NUMINAMATH_GPT_g_at_5_l2066_206606


namespace NUMINAMATH_GPT_range_of_m_l2066_206622

def quadratic_function (m : ℝ) : Prop :=
  ∀ x : ℝ, (x^2 + m * x + 1 = 0 → false)

def ellipse_condition (m : ℝ) : Prop :=
  0 < m

theorem range_of_m (m : ℝ) :
  (quadratic_function m ∨ ellipse_condition m) ∧ ¬ (quadratic_function m ∧ ellipse_condition m) →
  m ∈ Set.Ioo (-2 : ℝ) 0 ∪ Set.Ioi 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l2066_206622


namespace NUMINAMATH_GPT_find_f_2017_l2066_206661

noncomputable def f : ℝ → ℝ :=
sorry

axiom cond1 : ∀ x : ℝ, f (1 + x) + f (1 - x) = 0
axiom cond2 : ∀ x : ℝ, f (-x) = f x
axiom cond3 : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f x = 2^x - 1

theorem find_f_2017 : f 2017 = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_f_2017_l2066_206661


namespace NUMINAMATH_GPT_weekly_earnings_correct_l2066_206634

-- Definitions based on the conditions
def hours_weekdays : Nat := 5 * 5
def hours_weekends : Nat := 3 * 2
def hourly_rate_weekday : Nat := 3
def hourly_rate_weekend : Nat := 3 * 2
def earnings_weekdays : Nat := hours_weekdays * hourly_rate_weekday
def earnings_weekends : Nat := hours_weekends * hourly_rate_weekend

-- The total weekly earnings Mitch gets
def weekly_earnings : Nat := earnings_weekdays + earnings_weekends

-- The theorem we need to prove:
theorem weekly_earnings_correct : weekly_earnings = 111 :=
by
  sorry

end NUMINAMATH_GPT_weekly_earnings_correct_l2066_206634


namespace NUMINAMATH_GPT_ratio_of_liquid_rise_l2066_206696

theorem ratio_of_liquid_rise
  (h1 h2 : ℝ) (r1 r2 rm : ℝ)
  (V1 V2 Vm : ℝ)
  (H1 : r1 = 4)
  (H2 : r2 = 9)
  (H3 : V1 = (1 / 3) * π * r1^2 * h1)
  (H4 : V2 = (1 / 3) * π * r2^2 * h2)
  (H5 : V1 = V2)
  (H6 : rm = 2)
  (H7 : Vm = (4 / 3) * π * rm^3)
  (H8 : h2 = h1 * (81 / 16))
  (h1' h2' : ℝ)
  (H9 : h1' = h1 + Vm / ((1 / 3) * π * r1^2))
  (H10 : h2' = h2 + Vm / ((1 / 3) * π * r2^2)) :
  (h1' - h1) / (h2' - h2) = 81 / 16 :=
sorry

end NUMINAMATH_GPT_ratio_of_liquid_rise_l2066_206696


namespace NUMINAMATH_GPT_oil_leakage_calculation_l2066_206654

def total_oil_leaked : ℕ := 11687
def oil_leaked_while_worked : ℕ := 5165
def oil_leaked_before_work : ℕ := 6522

theorem oil_leakage_calculation :
  oil_leaked_before_work = total_oil_leaked - oil_leaked_while_work :=
sorry

end NUMINAMATH_GPT_oil_leakage_calculation_l2066_206654


namespace NUMINAMATH_GPT_customer_wants_score_of_eggs_l2066_206630

def Score := 20
def Dozen := 12

def options (n : Nat) : Prop :=
  n = Score ∨ n = 2 * Score ∨ n = 2 * Dozen ∨ n = 3 * Score

theorem customer_wants_score_of_eggs : 
  ∃ n, options n ∧ n = Score := 
by
  exists Score
  constructor
  apply Or.inl
  rfl
  rfl

end NUMINAMATH_GPT_customer_wants_score_of_eggs_l2066_206630


namespace NUMINAMATH_GPT_ratio_PeteHand_to_TracyCartwheel_l2066_206676

noncomputable def SusanWalkingSpeed (PeteBackwardSpeed : ℕ) : ℕ :=
  PeteBackwardSpeed / 3

noncomputable def TracyCartwheelSpeed (SusanSpeed : ℕ) : ℕ :=
  SusanSpeed * 2

def PeteHandsWalkingSpeed : ℕ := 2

def PeteBackwardWalkingSpeed : ℕ := 12

theorem ratio_PeteHand_to_TracyCartwheel :
  let SusanSpeed := SusanWalkingSpeed PeteBackwardWalkingSpeed
  let TracySpeed := TracyCartwheelSpeed SusanSpeed
  (PeteHandsWalkingSpeed : ℕ) / (TracySpeed : ℕ) = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_PeteHand_to_TracyCartwheel_l2066_206676


namespace NUMINAMATH_GPT_find_integers_with_conditions_l2066_206662

theorem find_integers_with_conditions :
  ∃ a b c d : ℕ, (1 ≤ a) ∧ (1 ≤ b) ∧ (1 ≤ c) ∧ (1 ≤ d) ∧ a * b * c * d = 2002 ∧ a + b + c + d < 40 := sorry

end NUMINAMATH_GPT_find_integers_with_conditions_l2066_206662


namespace NUMINAMATH_GPT_unique_positive_real_b_l2066_206609

noncomputable def is_am_gm_satisfied (r s t : ℝ) (a : ℝ) : Prop :=
  r + s + t = 2 * a ∧ r * s * t = 2 * a ∧ (r+s+t)/3 = ((r * s * t) ^ (1/3))

noncomputable def poly_roots_real (r s t : ℝ) : Prop :=
  ∀ x : ℝ, (x = r ∨ x = s ∨ x = t)

theorem unique_positive_real_b :
  ∃ b a : ℝ, 0 < a ∧ 0 < b ∧
  (∃ r s t : ℝ, (r ≥ 0 ∧ s ≥ 0 ∧ t ≥ 0 ∧ poly_roots_real r s t) ∧
   is_am_gm_satisfied r s t a ∧
   (x^3 - 2*a*x^2 + b*x - 2*a = (x - r) * (x - s) * (x - t)) ∧
   b = 9) := sorry

end NUMINAMATH_GPT_unique_positive_real_b_l2066_206609


namespace NUMINAMATH_GPT_expression_result_zero_l2066_206659

theorem expression_result_zero (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x = y + 1) : 
  (x + 1 / x) * (y - 1 / y) = 0 := 
by sorry

end NUMINAMATH_GPT_expression_result_zero_l2066_206659


namespace NUMINAMATH_GPT_parallel_lines_k_value_l2066_206605

theorem parallel_lines_k_value (k : ℝ) : 
  (∀ x y : ℝ, y = 5 * x + 3 → y = (3 * k) * x + 1 → true) → k = 5 / 3 :=
by
  intros
  sorry

end NUMINAMATH_GPT_parallel_lines_k_value_l2066_206605


namespace NUMINAMATH_GPT_probability_fourth_roll_six_l2066_206687

noncomputable def fair_die_prob : ℚ := 1 / 6
noncomputable def biased_die_prob : ℚ := 3 / 4
noncomputable def biased_die_other_face_prob : ℚ := 1 / 20
noncomputable def prior_prob : ℚ := 1 / 2

def p := 41
def q := 67

theorem probability_fourth_roll_six (p q : ℕ) (h1 : fair_die_prob = 1 / 6) (h2 : biased_die_prob = 3 / 4) (h3 : prior_prob = 1 / 2) :
  p + q = 108 :=
sorry

end NUMINAMATH_GPT_probability_fourth_roll_six_l2066_206687


namespace NUMINAMATH_GPT_milk_cost_l2066_206677

theorem milk_cost (x : ℝ) (h1 : 4 * 2.50 + 2 * x = 17) : x = 3.50 :=
by
  sorry

end NUMINAMATH_GPT_milk_cost_l2066_206677


namespace NUMINAMATH_GPT_each_boy_makes_14_dollars_l2066_206614

noncomputable def victor_shrimp_caught := 26
noncomputable def austin_shrimp_caught := victor_shrimp_caught - 8
noncomputable def brian_shrimp_caught := (victor_shrimp_caught + austin_shrimp_caught) / 2
noncomputable def total_shrimp_caught := victor_shrimp_caught + austin_shrimp_caught + brian_shrimp_caught
noncomputable def money_made := (total_shrimp_caught / 11) * 7
noncomputable def each_boys_earnings := money_made / 3

theorem each_boy_makes_14_dollars : each_boys_earnings = 14 := by
  sorry

end NUMINAMATH_GPT_each_boy_makes_14_dollars_l2066_206614


namespace NUMINAMATH_GPT_sum_of_squares_l2066_206666

theorem sum_of_squares (x : ℤ) (h : (x + 1) ^ 2 - x ^ 2 = 199) : x ^ 2 + (x + 1) ^ 2 = 19801 :=
sorry

end NUMINAMATH_GPT_sum_of_squares_l2066_206666


namespace NUMINAMATH_GPT_smallest_root_of_polynomial_l2066_206652

theorem smallest_root_of_polynomial :
  ∃ x : ℝ, (24 * x^3 - 106 * x^2 + 116 * x - 70 = 0) ∧ x = 0.67 :=
by
  sorry

end NUMINAMATH_GPT_smallest_root_of_polynomial_l2066_206652


namespace NUMINAMATH_GPT_sacks_filled_l2066_206679

theorem sacks_filled (pieces_per_sack : ℕ) (total_pieces : ℕ) (h1 : pieces_per_sack = 20) (h2 : total_pieces = 80) : (total_pieces / pieces_per_sack) = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_sacks_filled_l2066_206679


namespace NUMINAMATH_GPT_correctPropositions_l2066_206646

-- Define the conditions and statement as Lean structures.
structure Geometry :=
  (Line : Type)
  (Plane : Type)
  (parallel : Plane → Plane → Prop)
  (parallelLine : Line → Plane → Prop)
  (perpendicular : Plane → Plane → Prop)
  (perpendicularLine : Line → Plane → Prop)
  (subsetLine : Line → Plane → Prop)

-- Main theorem to be proved in Lean 4
theorem correctPropositions (G : Geometry) :
  (∀ (α β : G.Plane) (a : G.Line), (G.parallel α β) → (G.subsetLine a α) → (G.parallelLine a β)) ∧ 
  (∀ (α β : G.Plane) (a : G.Line), (G.perpendicularLine a α) → (G.perpendicularLine a β) → (G.parallel α β)) :=
sorry -- The proof is omitted, as per instructions

end NUMINAMATH_GPT_correctPropositions_l2066_206646


namespace NUMINAMATH_GPT_derivative_f_at_zero_l2066_206672

noncomputable def f (x : ℝ) : ℝ :=
  if |x| ≤ 1 then 4 * x * (1 - |x|) else 0

theorem derivative_f_at_zero : HasDerivAt f 4 0 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_derivative_f_at_zero_l2066_206672


namespace NUMINAMATH_GPT_fourth_rectangle_area_l2066_206665

-- Define the conditions and prove the area of the fourth rectangle
theorem fourth_rectangle_area (x y z w : ℝ)
  (h_xy : x * y = 24)
  (h_xw : x * w = 35)
  (h_zw : z * w = 42)
  (h_sum : x + z = 21) :
  y * w = 33.777 := 
sorry

end NUMINAMATH_GPT_fourth_rectangle_area_l2066_206665


namespace NUMINAMATH_GPT_part1_solution_set_part2_range_of_a_l2066_206631

def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1_solution_set (x : ℝ) : 
  (f x 2 ≥ 4) ↔ (x ≤ 3 / 2 ∨ x ≥ 11 / 2) :=
sorry

theorem part2_range_of_a (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 4) → (a ≤ -1 ∨ a ≥ 3) :=
sorry

end NUMINAMATH_GPT_part1_solution_set_part2_range_of_a_l2066_206631


namespace NUMINAMATH_GPT_keira_guarantees_capture_l2066_206674

theorem keira_guarantees_capture (k : ℕ) (n : ℕ) (h_k_pos : 0 < k) (h_n_cond : n > k / 2023) :
    k ≥ 1012 :=
sorry

end NUMINAMATH_GPT_keira_guarantees_capture_l2066_206674


namespace NUMINAMATH_GPT_cranberry_parts_l2066_206628

theorem cranberry_parts (L C : ℕ) :
  L = 3 →
  L + C = 72 →
  C = L + 18 →
  C = 21 :=
by
  intros hL hSum hDiff
  sorry

end NUMINAMATH_GPT_cranberry_parts_l2066_206628


namespace NUMINAMATH_GPT_num_new_students_l2066_206600

theorem num_new_students 
  (original_avg_age : ℕ) 
  (original_num_students : ℕ) 
  (new_avg_age : ℕ) 
  (age_decrease : ℕ) 
  (total_age_orginal : ℕ := original_num_students * original_avg_age) 
  (total_new_students : ℕ := (original_avg_age - age_decrease) * (original_num_students + 12))
  (x : ℕ := total_new_students - total_age_orginal) :
  original_avg_age = 40 → 
  original_num_students = 12 →
  new_avg_age = 32 →
  age_decrease = 4 →
  x = 12 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_num_new_students_l2066_206600


namespace NUMINAMATH_GPT_wednesday_more_than_tuesday_l2066_206632

noncomputable def monday_minutes : ℕ := 450

noncomputable def tuesday_minutes : ℕ := monday_minutes / 2

noncomputable def wednesday_minutes : ℕ := 300

theorem wednesday_more_than_tuesday : wednesday_minutes - tuesday_minutes = 75 :=
by
  sorry

end NUMINAMATH_GPT_wednesday_more_than_tuesday_l2066_206632
