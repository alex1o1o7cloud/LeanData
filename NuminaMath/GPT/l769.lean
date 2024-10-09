import Mathlib

namespace abs_sum_zero_implies_diff_eq_five_l769_76945

theorem abs_sum_zero_implies_diff_eq_five (a b : ℝ) (h : |a - 2| + |b + 3| = 0) : a - b = 5 :=
  sorry

end abs_sum_zero_implies_diff_eq_five_l769_76945


namespace total_students_l769_76921

theorem total_students (T : ℝ) (h1 : 0.3 * T =  0.7 * T - 616) : T = 880 :=
by sorry

end total_students_l769_76921


namespace clock_820_angle_is_130_degrees_l769_76986

def angle_at_8_20 : ℝ :=
  let degrees_per_hour := 30.0
  let degrees_per_minute_hour_hand := 0.5
  let num_hour_sections := 4.0
  let minutes := 20.0
  let hour_angle := num_hour_sections * degrees_per_hour
  let minute_addition := minutes * degrees_per_minute_hour_hand
  hour_angle + minute_addition

theorem clock_820_angle_is_130_degrees :
  angle_at_8_20 = 130 :=
by
  sorry

end clock_820_angle_is_130_degrees_l769_76986


namespace andy_max_cookies_l769_76959

theorem andy_max_cookies (total_cookies : ℕ) (andy_cookies : ℕ) (bella_cookies : ℕ)
  (h1 : total_cookies = 30)
  (h2 : bella_cookies = 2 * andy_cookies)
  (h3 : andy_cookies + bella_cookies = total_cookies) :
  andy_cookies = 10 := by
  sorry

end andy_max_cookies_l769_76959


namespace intersection_is_as_expected_l769_76951

noncomputable def quadratic_inequality_solution : Set ℝ :=
  { x | 2 * x^2 - 3 * x - 2 ≤ 0 }

noncomputable def logarithmic_condition : Set ℝ :=
  { x | x > 0 ∧ x ≠ 1 }

noncomputable def intersection_of_sets : Set ℝ :=
  (quadratic_inequality_solution ∩ logarithmic_condition)

theorem intersection_is_as_expected :
  intersection_of_sets = { x | (0 < x ∧ x < 1) ∨ (1 < x ∧ x ≤ 2) } :=
by
  sorry

end intersection_is_as_expected_l769_76951


namespace rain_all_three_days_is_six_percent_l769_76934

-- Definitions based on conditions from step a)
def P_rain_friday : ℚ := 2 / 5
def P_rain_saturday : ℚ := 1 / 2
def P_rain_sunday : ℚ := 3 / 10

-- The probability it will rain on all three days
def P_rain_all_three_days : ℚ := P_rain_friday * P_rain_saturday * P_rain_sunday

-- The Lean 4 theorem statement
theorem rain_all_three_days_is_six_percent : P_rain_all_three_days * 100 = 6 := by
  sorry

end rain_all_three_days_is_six_percent_l769_76934


namespace line_y_intercept_l769_76932

theorem line_y_intercept (x1 y1 x2 y2 : ℝ) (h1 : (x1, y1) = (2, 9)) (h2 : (x2, y2) = (5, 21)) :
    ∃ b : ℝ, (∀ x : ℝ, y = 4 * x + b) ∧ (b = 1) :=
by
  use 1
  sorry

end line_y_intercept_l769_76932


namespace arithmetic_example_l769_76960

theorem arithmetic_example : (2468 * 629) / (1234 * 37) = 34 :=
by
  sorry

end arithmetic_example_l769_76960


namespace min_value_complex_mod_one_l769_76994

/-- Given that the modulus of the complex number \( z \) is 1, prove that the minimum value of
    \( |z - 4|^2 + |z + 3 * Complex.I|^2 \) is \( 17 \). -/
theorem min_value_complex_mod_one (z : ℂ) (h : ‖z‖ = 1) : 
  ∃ α : ℝ, (‖z - 4‖^2 + ‖z + 3 * Complex.I‖^2) = 17 :=
sorry

end min_value_complex_mod_one_l769_76994


namespace boys_in_choir_l769_76944

theorem boys_in_choir
  (h1 : 20 + 2 * 20 + 16 + b = 88)
  : b = 12 :=
by
  sorry

end boys_in_choir_l769_76944


namespace number_of_triangles_l769_76926

-- Define a structure representing a triangle with integer angles.
structure Triangle :=
  (A B C : ℕ) -- angles in integer degrees
  (angle_sum : A + B + C = 180)
  (obtuse_A : A > 90)

-- Define a structure representing point D on side BC of triangle ABC such that triangle ABD is right-angled
-- and triangle ADC is isosceles.
structure PointOnBC (ABC : Triangle) :=
  (D : ℕ) -- angle at D in triangle ABC
  (right_ABD : ABC.A = 90 ∨ ABC.B = 90 ∨ ABC.C = 90)
  (isosceles_ADC : ABC.A = ABC.B ∨ ABC.A = ABC.C ∨ ABC.B = ABC.C)

-- Problem Statement:
theorem number_of_triangles (t : Triangle) (d : PointOnBC t): ∃ n : ℕ, n = 88 :=
by
  sorry

end number_of_triangles_l769_76926


namespace cookies_guests_l769_76999

theorem cookies_guests (cc_cookies : ℕ) (oc_cookies : ℕ) (sc_cookies : ℕ) (cc_per_guest : ℚ) (oc_per_guest : ℚ) (sc_per_guest : ℕ)
    (cc_total : cc_cookies = 45) (oc_total : oc_cookies = 62) (sc_total : sc_cookies = 38) (cc_ratio : cc_per_guest = 1.5)
    (oc_ratio : oc_per_guest = 2.25) (sc_ratio : sc_per_guest = 1) :
    (cc_cookies / cc_per_guest) ≥ 0 ∧ (oc_cookies / oc_per_guest) ≥ 0 ∧ (sc_cookies / sc_per_guest) ≥ 0 → 
    Nat.floor (oc_cookies / oc_per_guest) = 27 :=
by
  sorry

end cookies_guests_l769_76999


namespace problem_statement_l769_76976

noncomputable def f1 (x : ℝ) := x + (1 / x)
noncomputable def f2 (x : ℝ) := 1 / (x ^ 2)
noncomputable def f3 (x : ℝ) := x ^ 3 - 2 * x
noncomputable def f4 (x : ℝ) := x ^ 2

theorem problem_statement : ∀ (x : ℝ), f2 (-x) = f2 x := by 
  sorry

end problem_statement_l769_76976


namespace shirley_cases_needed_l769_76979

-- Define the given conditions
def trefoils_boxes := 54
def samoas_boxes := 36
def boxes_per_case := 6

-- The statement to prove
theorem shirley_cases_needed : trefoils_boxes / boxes_per_case >= samoas_boxes / boxes_per_case ∧ 
                               samoas_boxes / boxes_per_case = 6 :=
by
  let n_cases := samoas_boxes / boxes_per_case
  have h1 : trefoils_boxes / boxes_per_case = 9 := sorry
  have h2 : samoas_boxes / boxes_per_case = 6 := sorry
  have h3 : 9 >= 6 := by linarith
  exact ⟨h3, h2⟩


end shirley_cases_needed_l769_76979


namespace smallest_number_am_median_l769_76966

theorem smallest_number_am_median :
  ∃ (a b c : ℕ), a + b + c = 90 ∧ b = 28 ∧ c = b + 6 ∧ (a ≤ b ∧ b ≤ c) ∧ a = 28 :=
by
  sorry

end smallest_number_am_median_l769_76966


namespace num_white_black_balls_prob_2_black_balls_dist_exp_black_balls_l769_76924

-- Problem 1: Number of white and black balls
theorem num_white_black_balls (n m : ℕ) (h1 : n + m = 10)
  (h2 : (10 - m) = 4) : n = 4 ∧ m = 6 :=
by sorry

-- Problem 2: Probability of drawing exactly 2 black balls with replacement
theorem prob_2_black_balls (p_black_draw : ℕ → ℕ → ℚ)
  (h1 : ∀ n m, p_black_draw n m = (6/10)^(n-m) * (4/10)^m)
  (h2 : p_black_draw 2 3 = 54/125) : p_black_draw 2 3 = 54 / 125 :=
by sorry

-- Problem 3: Distribution and Expectation of number of black balls drawn without replacement
theorem dist_exp_black_balls (prob_X : ℕ → ℚ) (expect_X : ℚ)
  (h1 : prob_X 0 = 2/15) (h2 : prob_X 1 = 8/15) (h3 : prob_X 2 = 1/3)
  (h4 : expect_X = 6 / 5) : ∀ k, prob_X k = match k with
    | 0 => 2/15
    | 1 => 8/15
    | 2 => 1/3
    | _ => 0 :=
by sorry

end num_white_black_balls_prob_2_black_balls_dist_exp_black_balls_l769_76924


namespace rectangle_ratio_l769_76950

theorem rectangle_ratio (L B : ℕ) (hL : L = 250) (hB : B = 160) : L / B = 25 / 16 := by
  sorry

end rectangle_ratio_l769_76950


namespace snooker_tournament_l769_76991

theorem snooker_tournament : 
  ∀ (V G : ℝ),
    V + G = 320 →
    40 * V + 15 * G = 7500 →
    V ≥ 80 →
    G ≥ 100 →
    G - V = 104 :=
by
  intros V G h1 h2 h3 h4
  sorry

end snooker_tournament_l769_76991


namespace profit_per_tire_l769_76961

theorem profit_per_tire
  (fixed_cost : ℝ)
  (variable_cost_per_tire : ℝ)
  (selling_price_per_tire : ℝ)
  (batch_size : ℕ)
  (total_cost : ℝ)
  (total_revenue : ℝ)
  (total_profit : ℝ)
  (profit_per_tire : ℝ)
  (h1 : fixed_cost = 22500)
  (h2 : variable_cost_per_tire = 8)
  (h3 : selling_price_per_tire = 20)
  (h4 : batch_size = 15000)
  (h5 : total_cost = fixed_cost + variable_cost_per_tire * batch_size)
  (h6 : total_revenue = selling_price_per_tire * batch_size)
  (h7 : total_profit = total_revenue - total_cost)
  (h8 : profit_per_tire = total_profit / batch_size) :
  profit_per_tire = 10.50 :=
sorry

end profit_per_tire_l769_76961


namespace lion_cub_birth_rate_l769_76920

theorem lion_cub_birth_rate :
  ∀ (x : ℕ), 100 + 12 * (x - 1) = 148 → x = 5 :=
by
  intros x h
  sorry

end lion_cub_birth_rate_l769_76920


namespace girls_ran_miles_l769_76984

def boys_laps : ℕ := 34
def extra_laps : ℕ := 20
def lap_distance : ℚ := 1 / 6
def girls_laps : ℕ := boys_laps + extra_laps

theorem girls_ran_miles : girls_laps * lap_distance = 9 := 
by 
  sorry

end girls_ran_miles_l769_76984


namespace man_swims_distance_back_l769_76954

def swimming_speed_still_water : ℝ := 8
def speed_of_water : ℝ := 4
def time_taken_against_current : ℝ := 2
def distance_swum : ℝ := 8

theorem man_swims_distance_back :
  (distance_swum = (swimming_speed_still_water - speed_of_water) * time_taken_against_current) :=
by
  -- The proof will be filled in later.
  sorry

end man_swims_distance_back_l769_76954


namespace log10_sum_diff_l769_76940

noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log10_sum_diff :
  log10 32 + log10 50 - log10 8 = 2.301 :=
by
  sorry

end log10_sum_diff_l769_76940


namespace fifteenth_entry_is_21_l769_76918

def r_9 (n : ℕ) : ℕ := n % 9

def condition (n : ℕ) : Prop := (7 * n) % 9 ≤ 5

def sequence_elements (k : ℕ) : ℕ := 
  if k = 0 then 0
  else if k = 1 then 2
  else if k = 2 then 3
  else if k = 3 then 4
  else if k = 4 then 7
  else if k = 5 then 8
  else if k = 6 then 9
  else if k = 7 then 11
  else if k = 8 then 12
  else if k = 9 then 13
  else if k = 10 then 16
  else if k = 11 then 17
  else if k = 12 then 18
  else if k = 13 then 20
  else if k = 14 then 21
  else 0 -- for the sake of ensuring completeness

theorem fifteenth_entry_is_21 : sequence_elements 14 = 21 :=
by
  -- Mathematical proof omitted.
  sorry

end fifteenth_entry_is_21_l769_76918


namespace trigonometric_identity_l769_76971

theorem trigonometric_identity :
  (Real.sqrt 3 / Real.cos (10 * Real.pi / 180) - 1 / Real.sin (170 * Real.pi / 180) = -4) :=
by
  -- Proof goes here
  sorry

end trigonometric_identity_l769_76971


namespace y_completion_days_l769_76941

theorem y_completion_days (d : ℕ) (h : (12 : ℚ) / d + 1 / 4 = 1) : d = 16 :=
by
  sorry

end y_completion_days_l769_76941


namespace baker_cakes_l769_76946

theorem baker_cakes : (62.5 + 149.25 - 144.75 = 67) :=
by
  sorry

end baker_cakes_l769_76946


namespace reflect_point_value_l769_76904

theorem reflect_point_value (mx b : ℝ) 
  (start end_ : ℝ × ℝ)
  (Hstart : start = (2, 3))
  (Hend : end_ = (10, 7))
  (Hreflection : ∃ m b: ℝ, (end_.fst, end_.snd) = 
              (2 * ((5 / 2) - (1 / 2) * 3 * m - b), 2 * ((5 / 2) + (1 / 2) * 3)) ∧ m = -2)
  : m + b = 15 :=
sorry

end reflect_point_value_l769_76904


namespace find_x_l769_76903

theorem find_x (x : ℝ) (y : ℝ) : 
  (10 * x * y - 15 * y + 3 * x - (9 / 2) = 0) ↔ x = (3 / 2) :=
by
  sorry

end find_x_l769_76903


namespace arithmetic_difference_l769_76987

variable (S : ℕ → ℤ)
variable (n : ℕ)

-- Definitions as conditions from the problem
def is_arithmetic_sum (s : ℕ → ℤ) :=
  ∀ n : ℕ, s n = 2 * n ^ 2 - 5 * n

theorem arithmetic_difference :
  is_arithmetic_sum S →
  S 10 - S 7 = 87 :=
by
  intro h
  sorry

end arithmetic_difference_l769_76987


namespace compute_fraction_product_l769_76970

theorem compute_fraction_product :
  (1 / 3)^4 * (1 / 5) = 1 / 405 :=
by
  sorry

end compute_fraction_product_l769_76970


namespace zhou_yu_age_eq_l769_76962

-- Define the conditions based on the problem statement
variable (x : ℕ)  -- x represents the tens digit of Zhou Yu's age

-- Condition: The tens digit is three less than the units digit
def units_digit := x + 3

-- Define Zhou Yu's age based on the tens and units digits
def zhou_yu_age := 10 * x + units_digit x

-- Prove the correct equation representing Zhou Yu's lifespan
theorem zhou_yu_age_eq : zhou_yu_age x = (units_digit x) ^ 2 :=
by sorry

end zhou_yu_age_eq_l769_76962


namespace total_bees_in_hive_at_end_of_7_days_l769_76905

-- Definitions of given conditions
def daily_hatch : Nat := 3000
def daily_loss : Nat := 900
def initial_bees : Nat := 12500
def days : Nat := 7
def queen_count : Nat := 1

-- Statement to prove
theorem total_bees_in_hive_at_end_of_7_days :
  initial_bees + daily_hatch * days - daily_loss * days + queen_count = 27201 := by
  sorry

end total_bees_in_hive_at_end_of_7_days_l769_76905


namespace inradius_of_triangle_l769_76968

theorem inradius_of_triangle (P A : ℝ) (hP : P = 40) (hA : A = 50) : 
  ∃ r : ℝ, r = 2.5 ∧ A = r * (P / 2) :=
by
  sorry

end inradius_of_triangle_l769_76968


namespace sum_of_fractions_l769_76915

-- Definitions of parameters and conditions
variables {x y : ℝ}
variable (hx : x ≠ 0)
variable (hy : y ≠ 0)

-- The statement of the proof problem
theorem sum_of_fractions (hx : x ≠ 0) (hy : y ≠ 0) : 
  (3 / x) + (2 / y) = (3 * y + 2 * x) / (x * y) :=
sorry

end sum_of_fractions_l769_76915


namespace sum_of_roots_l769_76919

-- Defined the equation x^2 - 7x + 2 - 16 = 0 as x^2 - 7x - 14 = 0
def equation (x : ℝ) := x^2 - 7 * x - 14 = 0 

-- State the theorem leveraging the above condition
theorem sum_of_roots : 
  (∃ x1 x2 : ℝ, equation x1 ∧ equation x2 ∧ x1 ≠ x2) →
  (∃ sum : ℝ, sum = 7) := by
  sorry

end sum_of_roots_l769_76919


namespace george_slices_l769_76942

def num_small_pizzas := 3
def num_large_pizzas := 2
def slices_per_small_pizza := 4
def slices_per_large_pizza := 8
def slices_leftover := 10
def slices_per_person := 3
def total_pizza_slices := (num_small_pizzas * slices_per_small_pizza) + (num_large_pizzas * slices_per_large_pizza)
def slices_eaten := total_pizza_slices - slices_leftover
def G := 6 -- Slices George would like to eat

theorem george_slices :
  G + (G + 1) + ((G + 1) / 2) + (3 * slices_per_person) = slices_eaten :=
by
  sorry

end george_slices_l769_76942


namespace total_shirts_correct_l769_76977

def machine_A_production_rate := 6
def machine_A_yesterday_minutes := 12
def machine_A_today_minutes := 10

def machine_B_production_rate := 8
def machine_B_yesterday_minutes := 10
def machine_B_today_minutes := 15

def machine_C_production_rate := 5
def machine_C_yesterday_minutes := 20
def machine_C_today_minutes := 0

def total_shirts_produced : Nat :=
  (machine_A_production_rate * machine_A_yesterday_minutes +
  machine_A_production_rate * machine_A_today_minutes) +
  (machine_B_production_rate * machine_B_yesterday_minutes +
  machine_B_production_rate * machine_B_today_minutes) +
  (machine_C_production_rate * machine_C_yesterday_minutes +
  machine_C_production_rate * machine_C_today_minutes)

theorem total_shirts_correct : total_shirts_produced = 432 :=
by 
  sorry 

end total_shirts_correct_l769_76977


namespace find_b_minus_a_l769_76933

theorem find_b_minus_a (a b : ℤ) (ha : 0 < a) (hb : 0 < b) (h : 2 * a - 9 * b + 18 * a * b = 2018) : b - a = 223 :=
sorry

end find_b_minus_a_l769_76933


namespace find_c_l769_76901

theorem find_c (y : ℝ) (c : ℝ) (h1 : y > 0) (h2 : (6 * y / 20) + (c * y / 10) = 0.6 * y) : c = 3 :=
by 
  -- Skipping the proof
  sorry

end find_c_l769_76901


namespace angle_trig_identity_l769_76973

theorem angle_trig_identity
  (A B C : ℝ)
  (h_sum : A + B + C = Real.pi) :
  Real.cos (A / 2) ^ 2 = Real.cos (B / 2) ^ 2 + Real.cos (C / 2) ^ 2 - 
                       2 * Real.cos (B / 2) * Real.cos (C / 2) * Real.sin (A / 2) :=
by
  sorry

end angle_trig_identity_l769_76973


namespace graph_not_through_third_quadrant_l769_76965

theorem graph_not_through_third_quadrant (k : ℝ) (h_nonzero : k ≠ 0) (h_decreasing : k < 0) : 
  ¬(∃ x y : ℝ, y = k * x - k ∧ x < 0 ∧ y < 0) :=
sorry

end graph_not_through_third_quadrant_l769_76965


namespace sum_of_squares_pentagon_greater_icosagon_l769_76990

noncomputable def compare_sum_of_squares (R : ℝ) : Prop :=
  let a_5 := 2 * R * Real.sin (Real.pi / 5)
  let a_20 := 2 * R * Real.sin (Real.pi / 20)
  4 * a_20^2 < a_5^2

theorem sum_of_squares_pentagon_greater_icosagon (R : ℝ) : 
  compare_sum_of_squares R :=
  sorry

end sum_of_squares_pentagon_greater_icosagon_l769_76990


namespace k_plus_m_eq_27_l769_76936

theorem k_plus_m_eq_27 (k m : ℝ) (a b c : ℝ) 
  (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : a > 0) (h5 : b > 0) (h6 : c > 0)
  (h7 : a + b + c = 8) 
  (h8 : k = a * b + a * c + b * c) 
  (h9 : m = a * b * c) :
  k + m = 27 :=
by
  sorry

end k_plus_m_eq_27_l769_76936


namespace tangency_point_of_parabolas_l769_76900

theorem tangency_point_of_parabolas :
  ∃ (x y : ℝ), y = x^2 + 17 * x + 40 ∧ x = y^2 + 51 * y + 650 ∧ x = -7 ∧ y = -25 :=
by
  sorry

end tangency_point_of_parabolas_l769_76900


namespace committee_selection_l769_76963

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem committee_selection :
  let seniors := 10
  let members := 30
  let non_seniors := members - seniors
  let choices := binom seniors 2 * binom non_seniors 3 +
                 binom seniors 3 * binom non_seniors 2 +
                 binom seniors 4 * binom non_seniors 1 +
                 binom seniors 5
  choices = 78552 :=
by
  sorry

end committee_selection_l769_76963


namespace interval_of_y_l769_76912

theorem interval_of_y (y : ℝ) (h : y = (1 / y) * (-y) - 5) : -6 ≤ y ∧ y ≤ -4 :=
by sorry

end interval_of_y_l769_76912


namespace f_neg_def_l769_76907

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = - f x

def f_pos_def (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → f x = x * (1 - x)

theorem f_neg_def (f : ℝ → ℝ) (h1 : is_odd_function f) (h2 : f_pos_def f) :
  ∀ x : ℝ, x < 0 → f x = x * (1 + x) :=
by
  sorry

end f_neg_def_l769_76907


namespace median_of_64_consecutive_integers_l769_76935

theorem median_of_64_consecutive_integers (n : ℕ) (S : ℕ) (h1 : n = 64) (h2 : S = 8^4) :
  S / n = 64 :=
by
  -- to skip the proof
  sorry

end median_of_64_consecutive_integers_l769_76935


namespace min_value_frac_sum_l769_76988

-- Define the main problem
theorem min_value_frac_sum (a b : ℝ) (h1 : 2 * a + 3 * b = 1) (h2 : 0 < a) (h3 : 0 < b) : 
  ∃ x : ℝ, (x = 25) ∧ ∀ y, (y = (2 / a + 3 / b)) → y ≥ x :=
sorry

end min_value_frac_sum_l769_76988


namespace minimum_small_droppers_l769_76938

/-
Given:
1. A total volume to be filled: V = 265 milliliters.
2. Small droppers can hold: s = 19 milliliters each.
3. No large droppers are used.

Prove:
The minimum number of small droppers required to fill the container completely is 14.
-/

theorem minimum_small_droppers (V s: ℕ) (hV: V = 265) (hs: s = 19) : 
  ∃ n: ℕ, n = 14 ∧ n * s ≥ V ∧ (n - 1) * s < V :=
by
  sorry  -- proof to be provided

end minimum_small_droppers_l769_76938


namespace students_suggested_tomatoes_79_l769_76989

theorem students_suggested_tomatoes_79 (T : ℕ)
  (mashed_potatoes : ℕ)
  (h1 : mashed_potatoes = 144)
  (h2 : mashed_potatoes = T + 65) :
  T = 79 :=
by {
  -- Proof steps will go here
  sorry
}

end students_suggested_tomatoes_79_l769_76989


namespace parabola_focus_directrix_distance_l769_76964

theorem parabola_focus_directrix_distance :
  ∀ (y x : ℝ), 
    y^2 = 8 * x → 
    ∃ p : ℝ, 2 * p = 8 ∧ p = 4 := by
  sorry

end parabola_focus_directrix_distance_l769_76964


namespace problem_statement_l769_76978

namespace MathProof

def p : Prop := (2 + 4 = 7)
def q : Prop := ∀ x : ℝ, x = 1 → x^2 ≠ 1

theorem problem_statement : ¬ (p ∧ q) ∧ (p ∨ q) :=
by
  -- To be filled in
  sorry

end MathProof

end problem_statement_l769_76978


namespace pine_cones_on_roof_l769_76925

theorem pine_cones_on_roof 
  (num_trees : ℕ) 
  (pine_cones_per_tree : ℕ) 
  (percent_on_roof : ℝ) 
  (weight_per_pine_cone : ℝ) 
  (h1 : num_trees = 8)
  (h2 : pine_cones_per_tree = 200)
  (h3 : percent_on_roof = 0.30)
  (h4 : weight_per_pine_cone = 4) : 
  (num_trees * pine_cones_per_tree * percent_on_roof * weight_per_pine_cone = 1920) :=
by
  sorry

end pine_cones_on_roof_l769_76925


namespace max_value_of_expression_l769_76955

theorem max_value_of_expression :
  ∃ x : ℝ, ∀ y : ℝ, -x^2 + 4*x + 10 ≤ -y^2 + 4*y + 10 ∧ -x^2 + 4*x + 10 = 14 :=
sorry

end max_value_of_expression_l769_76955


namespace find_a_of_perpendicular_tangent_and_line_l769_76931

open Real

theorem find_a_of_perpendicular_tangent_and_line :
  let e := Real.exp 1
  let slope_tangent := 1 / e
  let slope_line (a : ℝ) := a
  let tangent_perpendicular := ∀ (a : ℝ), slope_tangent * slope_line a = -1
  tangent_perpendicular -> ∃ a : ℝ, a = -e :=
by {
  sorry
}

end find_a_of_perpendicular_tangent_and_line_l769_76931


namespace ivanka_woody_total_months_l769_76916

theorem ivanka_woody_total_months
  (woody_years : ℝ)
  (months_per_year : ℝ)
  (additional_months : ℕ)
  (woody_months : ℝ)
  (ivanka_months : ℝ)
  (total_months : ℝ)
  (h1 : woody_years = 1.5)
  (h2 : months_per_year = 12)
  (h3 : additional_months = 3)
  (h4 : woody_months = woody_years * months_per_year)
  (h5 : ivanka_months = woody_months + additional_months)
  (h6 : total_months = woody_months + ivanka_months) :
  total_months = 39 := by
  sorry

end ivanka_woody_total_months_l769_76916


namespace cube_product_l769_76992

/-- A cube is a three-dimensional shape with a specific number of vertices and faces. -/
structure Cube where
  vertices : ℕ
  faces : ℕ

theorem cube_product (C : Cube) (h1: C.vertices = 8) (h2: C.faces = 6) : 
  (C.vertices * C.faces = 48) :=
by sorry

end cube_product_l769_76992


namespace isosceles_triangle_apex_angle_l769_76923

theorem isosceles_triangle_apex_angle (a b c : ℝ) (ha : a = 40) (hb : b = 40) (hc : b = c) :
  (a + b + c = 180) → (c = 100 ∨ a = 40) :=
by
-- We start the proof and provide the conditions.
  sorry  -- Lean expects the proof here.

end isosceles_triangle_apex_angle_l769_76923


namespace meadow_trees_count_l769_76917

theorem meadow_trees_count (n : ℕ) (f s m : ℕ → ℕ) :
  (f 20 = s 7) ∧ (f 7 = s 94) ∧ (s 7 > f 20) → 
  n = 100 :=
by
  sorry

end meadow_trees_count_l769_76917


namespace quadrilateral_area_l769_76957

noncomputable def AB : ℝ := 3
noncomputable def BC : ℝ := 3
noncomputable def CD : ℝ := 4
noncomputable def DA : ℝ := 8
noncomputable def angle_DAB_add_angle_ABC : ℝ := 180

theorem quadrilateral_area :
  AB = 3 ∧ BC = 3 ∧ CD = 4 ∧ DA = 8 ∧ angle_DAB_add_angle_ABC = 180 →
  ∃ area : ℝ, area = 13.2 :=
by {
  sorry
}

end quadrilateral_area_l769_76957


namespace no_real_solutions_l769_76997

open Real

theorem no_real_solutions :
  ¬(∃ x : ℝ, (3 * x^2) / (x - 2) - (x + 4) / 4 + (5 - 3 * x) / (x - 2) + 2 = 0) := by
  sorry

end no_real_solutions_l769_76997


namespace value_of_expression_l769_76906

theorem value_of_expression
  (a b c : ℝ)
  (h1 : |a - b| = 1)
  (h2 : |b - c| = 1)
  (h3 : |c - a| = 2)
  (h4 : a * b * c = 60) :
  (a / (b * c) + b / (c * a) + c / (a * b) - 1 / a - 1 / b - 1 / c) = 1 / 10 :=
sorry

end value_of_expression_l769_76906


namespace completing_square_eq_sum_l769_76982

theorem completing_square_eq_sum :
  ∃ (a b c : ℤ), a > 0 ∧ (∀ (x : ℝ), 36 * x^2 - 60 * x + 25 = (a * x + b)^2 - c) ∧ a + b + c = 26 :=
by
  sorry

end completing_square_eq_sum_l769_76982


namespace triangle_is_obtuse_l769_76937

-- Define the conditions of the problem
def angles (x : ℝ) : Prop :=
  2 * x + 3 * x + 6 * x = 180

def obtuse_angle (x : ℝ) : Prop :=
  6 * x > 90

-- State the theorem
theorem triangle_is_obtuse (x : ℝ) (hx : angles x) : obtuse_angle x :=
sorry

end triangle_is_obtuse_l769_76937


namespace midpoint_C_l769_76914

variables (A B C : ℝ × ℝ)
variables (x1 y1 x2 y2 : ℝ)
variables (AC CB : ℝ)

def segment_division (A B C : ℝ × ℝ) (m n : ℝ) : Prop :=
  C = ((m * B.1 + n * A.1) / (m + n), (m * B.2 + n * A.2) / (m + n))

theorem midpoint_C :
  A = (-2, 1) →
  B = (4, 9) →
  AC = 2 * CB →
  segment_division A B C 2 1 →
  C = (2, 19 / 3) :=
by
  sorry

end midpoint_C_l769_76914


namespace children_tickets_sold_l769_76913

theorem children_tickets_sold (A C : ℝ) (h1 : A + C = 400) (h2 : 6 * A + 4.5 * C = 2100) : C = 200 :=
sorry

end children_tickets_sold_l769_76913


namespace max_value_of_quadratic_expression_l769_76995

theorem max_value_of_quadratic_expression (s : ℝ) : ∃ x : ℝ, -3 * s^2 + 24 * s - 8 ≤ x ∧ x = 40 :=
sorry

end max_value_of_quadratic_expression_l769_76995


namespace rectangle_difference_length_width_l769_76998

theorem rectangle_difference_length_width (x y p d : ℝ) (h1 : x + y = p / 2) (h2 : x^2 + y^2 = d^2) (h3 : x > y) : 
  x - y = (Real.sqrt (8 * d^2 - p^2)) / 2 := sorry

end rectangle_difference_length_width_l769_76998


namespace cistern_problem_l769_76996

noncomputable def cistern_problem_statement : Prop :=
∀ (x : ℝ),
  (1 / 5 - 1 / x = 1 / 11.25) → x = 9

theorem cistern_problem : cistern_problem_statement :=
sorry

end cistern_problem_l769_76996


namespace math_group_question_count_l769_76922

theorem math_group_question_count (m n : ℕ) (h : m * (m - 1) + m * n + n = 51) : m = 6 ∧ n = 3 := 
sorry

end math_group_question_count_l769_76922


namespace find_f_2017_div_2_l769_76967

noncomputable def is_odd_function {X Y : Type*} [AddGroup X] [AddGroup Y] (f : X → Y) :=
  ∀ x : X, f (-x) = -f x

noncomputable def is_periodic_function {X Y : Type*} [AddGroup X] [AddGroup Y] (p : X) (f : X → Y) :=
  ∀ x : X, f (x + p) = f x

noncomputable def f : ℝ → ℝ 
| x => if -1 ≤ x ∧ x ≤ 0 then x * x + x else sorry

theorem find_f_2017_div_2 : f (2017 / 2) = 1 / 4 :=
by
  have h_odd : is_odd_function f := sorry
  have h_period : is_periodic_function 2 f := sorry
  unfold f
  sorry

end find_f_2017_div_2_l769_76967


namespace wood_length_equation_l769_76985

-- Define the conditions as hypotheses
def length_of_wood_problem (x : ℝ) :=
  (1 / 2) * (x + 4.5) = x - 1

-- Now we state the theorem we want to prove, which is equivalent to the question == answer
theorem wood_length_equation (x : ℝ) :
  (1 / 2) * (x + 4.5) = x - 1 :=
sorry

end wood_length_equation_l769_76985


namespace fish_pond_estimate_l769_76972

variable (N : ℕ)
variable (total_first_catch total_second_catch marked_in_first_catch marked_in_second_catch : ℕ)

/-- Estimate the total number of fish in the pond -/
theorem fish_pond_estimate
  (h1 : total_first_catch = 100)
  (h2 : total_second_catch = 120)
  (h3 : marked_in_first_catch = 100)
  (h4 : marked_in_second_catch = 15)
  (h5 : (marked_in_second_catch : ℚ) / total_second_catch = (marked_in_first_catch : ℚ) / N) :
  N = 800 := 
sorry

end fish_pond_estimate_l769_76972


namespace length_and_width_of_prism_l769_76983

theorem length_and_width_of_prism (w l h d : ℝ) (h_cond : h = 12) (d_cond : d = 15) (length_cond : l = 3 * w) :
  (w = 3) ∧ (l = 9) :=
by
  -- The proof is omitted as instructed in the task description.
  sorry

end length_and_width_of_prism_l769_76983


namespace correctStatement_l769_76908

def isValidInput : String → Bool
| "INPUT a, b, c;" => true
| "INPUT x=3;" => false
| _ => false

def isValidOutput : String → Bool
| "PRINT 20,3*2." => true
| "PRINT A=4;" => false
| _ => false

def isValidStatement : String → Bool
| stmt => (isValidInput stmt ∨ isValidOutput stmt)

theorem correctStatement : isValidStatement "PRINT 20,3*2." = true ∧ 
                           ¬(isValidStatement "INPUT a; b; c;" = true) ∧ 
                           ¬(isValidStatement "INPUT x=3;" = true) ∧ 
                           ¬(isValidStatement "PRINT A=4;" = true) := 
by sorry

end correctStatement_l769_76908


namespace triangle_area_ab_l769_76902

theorem triangle_area_ab (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0.5 * (12 / a) * (12 / b) = 12) : a * b = 6 :=
by
  sorry

end triangle_area_ab_l769_76902


namespace tangent_line_through_point_and_circle_l769_76943

noncomputable def tangent_line_equation : String :=
  "y - 1 = 0"

theorem tangent_line_through_point_and_circle :
  ∀ (line_eq: String), 
  (∀ (x y: ℝ), (x - 1) ^ 2 + y ^ 2 = 1 ∧ (x, y) = (1, 1) → y - 1 = 0) →
  line_eq = tangent_line_equation :=
by
  intro line_eq h
  sorry

end tangent_line_through_point_and_circle_l769_76943


namespace angle_sum_around_point_l769_76958

theorem angle_sum_around_point (y : ℕ) (h1 : 210 + 3 * y = 360) : y = 50 := 
by 
  sorry

end angle_sum_around_point_l769_76958


namespace contribution_required_l769_76952

-- Definitions corresponding to the problem statement
def total_amount : ℝ := 2000
def number_of_friends : ℝ := 7
def your_contribution_factor : ℝ := 2

-- Prove that the amount each friend needs to raise is approximately 222.22
theorem contribution_required (x : ℝ) 
  (h : 9 * x = total_amount) :
  x = 2000 / 9 := 
  by sorry

end contribution_required_l769_76952


namespace wage_increase_l769_76993

-- Definition: Regression line equation
def regression_line (x : ℝ) : ℝ := 80 * x + 50

-- Theorem: On average, when the labor productivity increases by 1000 yuan, the wage increases by 80 yuan
theorem wage_increase (x : ℝ) : regression_line (x + 1) - regression_line x = 80 :=
by
  sorry

end wage_increase_l769_76993


namespace solution_set_for_inequality_l769_76930

noncomputable def f : ℝ → ℝ := sorry

def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem solution_set_for_inequality
  (h1 : is_odd f)
  (h2 : f 2 = 0)
  (h3 : ∀ x > 0, x * deriv f x - f x < 0) :
  {x : ℝ | f x / x > 0} = {x : ℝ | -2 < x ∧ x < 0} ∪ {x : ℝ | 0 < x ∧ x < 2} :=
by sorry

end solution_set_for_inequality_l769_76930


namespace equivalent_fraction_l769_76956

theorem equivalent_fraction (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h : (4 * x + 2 * y) / (x - 4 * y) = -3) :
  (2 * x + 8 * y) / (4 * x - 2 * y) = 38 / 13 :=
by
  sorry

end equivalent_fraction_l769_76956


namespace complement_A_in_U_l769_76969

def U := {x : ℝ | -4 < x ∧ x < 4}
def A := {x : ℝ | -3 ≤ x ∧ x < 2}

theorem complement_A_in_U :
  {x : ℝ | x ∈ U ∧ x ∉ A} = {x : ℝ | (-4 < x ∧ x < -3) ∨ (2 ≤ x ∧ x < 4)} :=
by {
  sorry
}

end complement_A_in_U_l769_76969


namespace tails_and_die_1_or_2_l769_76947

noncomputable def fairCoinFlipProbability : ℚ := 1 / 2
noncomputable def fairDieRollProbability : ℚ := 1 / 6
noncomputable def combinedProbability : ℚ := fairCoinFlipProbability * (fairDieRollProbability + fairDieRollProbability)

theorem tails_and_die_1_or_2 :
  combinedProbability = 1 / 6 :=
by
  sorry

end tails_and_die_1_or_2_l769_76947


namespace abs_ineq_solution_set_l769_76910

theorem abs_ineq_solution_set {x : ℝ} : |x + 1| - |x - 3| ≥ 2 ↔ x ≥ 2 :=
by
  sorry

end abs_ineq_solution_set_l769_76910


namespace brinley_animals_count_l769_76927

theorem brinley_animals_count :
  let snakes := 100
  let arctic_foxes := 80
  let leopards := 20
  let bee_eaters := 10 * ((snakes / 2) + (2 * leopards))
  let cheetahs := 4 * (arctic_foxes - leopards)
  let alligators := 3 * (snakes * arctic_foxes * leopards)
  snakes + arctic_foxes + leopards + bee_eaters + cheetahs + alligators = 481340 := by
  sorry

end brinley_animals_count_l769_76927


namespace range_of_t_l769_76949

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem range_of_t (t : ℝ) :  
  (∀ a > 0, ∀ x₀ y₀, 
    (a - a * Real.log x₀) / x₀^2 = 1 / 2 ∧ 
    y₀ = (a * Real.log x₀) / x₀ ∧ 
    x₀ = 2 * y₀ ∧ 
    a = Real.exp 1 ∧ 
    f (f x) = t -> t = 0) :=
by
  sorry

end range_of_t_l769_76949


namespace inequality_inequality_hold_l769_76980

theorem inequality_inequality_hold (k : ℕ) (x y z : ℝ) 
  (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) 
  (h_sum : x + y + z = 1) : 
  (x ^ (k + 2) / (x ^ (k + 1) + y ^ k + z ^ k) 
  + y ^ (k + 2) / (y ^ (k + 1) + z ^ k + x ^ k) 
  + z ^ (k + 2) / (z ^ (k + 1) + x ^ k + y ^ k)) 
  ≥ (1 / 7) :=
sorry

end inequality_inequality_hold_l769_76980


namespace volume_ratio_l769_76953

namespace Geometry

variables {Point : Type} [MetricSpace Point]

noncomputable def volume_pyramid (A B1 C1 D1 : Point) : ℝ := sorry

theorem volume_ratio 
  (A B1 B2 C1 C2 D1 D2 : Point) 
  (hA_B1: dist A B1 ≠ 0) (hA_B2: dist A B2 ≠ 0)
  (hA_C1: dist A C1 ≠ 0) (hA_C2: dist A C2 ≠ 0)
  (hA_D1: dist A D1 ≠ 0) (hA_D2: dist A D2 ≠ 0) :
  (volume_pyramid A B1 C1 D1 / volume_pyramid A B2 C2 D2) = 
    (dist A B1 * dist A C1 * dist A D1) / (dist A B2 * dist A C2 * dist A D2) := 
sorry

end Geometry

end volume_ratio_l769_76953


namespace count_monomials_l769_76975

def isMonomial (expr : String) : Bool :=
  match expr with
  | "m+n" => false
  | "2x^2y" => true
  | "1/x" => true
  | "-5" => true
  | "a" => true
  | _ => false

theorem count_monomials :
  let expressions := ["m+n", "2x^2y", "1/x", "-5", "a"]
  (expressions.filter isMonomial).length = 3 :=
by { sorry }

end count_monomials_l769_76975


namespace roadRepairDays_l769_76948

-- Definitions from the conditions
def dailyRepairLength1 : ℕ := 6
def daysToFinish1 : ℕ := 8
def totalLengthOfRoad : ℕ := dailyRepairLength1 * daysToFinish1
def dailyRepairLength2 : ℕ := 8
def daysToFinish2 : ℕ := totalLengthOfRoad / dailyRepairLength2

-- Theorem to be proven
theorem roadRepairDays :
  daysToFinish2 = 6 :=
by
  sorry

end roadRepairDays_l769_76948


namespace A_is_7056_l769_76939

-- Define the variables and conditions
def D : ℕ := 4 * 3
def E : ℕ := 7 * 3
def B : ℕ := 4 * D
def C : ℕ := 7 * E
def A : ℕ := B * C

-- Prove that A = 7056 given the conditions
theorem A_is_7056 : A = 7056 := by
  -- We will skip the proof steps with 'sorry'
  sorry

end A_is_7056_l769_76939


namespace order_of_m_n_p_q_l769_76929

variable {m n p q : ℝ} -- Define the variables as real numbers

theorem order_of_m_n_p_q (h1 : m < n) 
                         (h2 : p < q) 
                         (h3 : (p - m) * (p - n) < 0) 
                         (h4 : (q - m) * (q - n) < 0) : 
    m < p ∧ p < q ∧ q < n := 
by
  sorry

end order_of_m_n_p_q_l769_76929


namespace practice_problems_total_l769_76981

theorem practice_problems_total :
  let marvin_yesterday := 40
  let marvin_today := 3 * marvin_yesterday
  let arvin_yesterday := 2 * marvin_yesterday
  let arvin_today := 2 * marvin_today
  let kevin_yesterday := 30
  let kevin_today := kevin_yesterday + 10
  let total_problems := (marvin_yesterday + marvin_today) + (arvin_yesterday + arvin_today) + (kevin_yesterday + kevin_today)
  total_problems = 550 :=
by
  sorry

end practice_problems_total_l769_76981


namespace imaginary_part_of_1_minus_2i_l769_76928

def i := Complex.I

theorem imaginary_part_of_1_minus_2i : Complex.im (1 - 2 * i) = -2 :=
by
  sorry

end imaginary_part_of_1_minus_2i_l769_76928


namespace simplify_expression_l769_76909

theorem simplify_expression : ((3 * 2 + 4 + 6) / 3 - 2 / 3) = 14 / 3 := by
  sorry

end simplify_expression_l769_76909


namespace find_s_over_r_l769_76974

-- Define the function
def f (k : ℝ) : ℝ := 9 * k ^ 2 - 6 * k + 15

-- Define constants
variables (d r s : ℝ)

-- Define the main theorem to be proved
theorem find_s_over_r : 
  (∀ k : ℝ, f k = d * (k + r) ^ 2 + s) → s / r = -42 :=
by
  sorry

end find_s_over_r_l769_76974


namespace divides_polynomial_l769_76911

theorem divides_polynomial (n : ℕ) (x : ℤ) (hn : 0 < n) :
  (x^2 + x + 1) ∣ (x^(n+2) + (x+1)^(2*n+1)) :=
sorry

end divides_polynomial_l769_76911
