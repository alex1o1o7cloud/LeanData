import Mathlib

namespace NUMINAMATH_GPT_inequality_range_l1148_114875

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, |2 * x + 1| - |2 * x - 1| < a) → a > 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_range_l1148_114875


namespace NUMINAMATH_GPT_maximum_profit_l1148_114867

def cost_price_per_unit : ℕ := 40
def initial_selling_price_per_unit : ℕ := 50
def units_sold_per_month : ℕ := 210
def price_increase_effect (x : ℕ) : ℕ := units_sold_per_month - 10 * x
def profit_function (x : ℕ) : ℕ := (price_increase_effect x) * (initial_selling_price_per_unit + x - cost_price_per_unit)

theorem maximum_profit :
  profit_function 5 = 2400 ∧ profit_function 6 = 2400 :=
by
  sorry

end NUMINAMATH_GPT_maximum_profit_l1148_114867


namespace NUMINAMATH_GPT_eval_g_inv_g_inv_14_l1148_114825

variable (g : ℝ → ℝ) (g_inv : ℝ → ℝ)

axiom g_def : ∀ x, g x = 3 * x - 4
axiom g_inv_def : ∀ y, g_inv y = (y + 4) / 3

theorem eval_g_inv_g_inv_14 : g_inv (g_inv 14) = 10 / 3 :=
by
    sorry

end NUMINAMATH_GPT_eval_g_inv_g_inv_14_l1148_114825


namespace NUMINAMATH_GPT_time_to_reach_ship_l1148_114844

-- Conditions in Lean 4
def rate : ℕ := 22
def depth : ℕ := 7260

-- The theorem that we want to prove
theorem time_to_reach_ship : depth / rate = 330 := by
  sorry

end NUMINAMATH_GPT_time_to_reach_ship_l1148_114844


namespace NUMINAMATH_GPT_intersection_M_N_l1148_114841

  open Set

  def M : Set ℝ := {x | Real.log x > 0}
  def N : Set ℝ := {x | x^2 ≤ 4}

  theorem intersection_M_N : M ∩ N = {x | 1 < x ∧ x ≤ 2} :=
  by
    sorry
  
end NUMINAMATH_GPT_intersection_M_N_l1148_114841


namespace NUMINAMATH_GPT_correct_option_b_l1148_114876

theorem correct_option_b (a : ℝ) : (-2 * a ^ 4) ^ 3 = -8 * a ^ 12 :=
sorry

end NUMINAMATH_GPT_correct_option_b_l1148_114876


namespace NUMINAMATH_GPT_line_through_point_hyperbola_l1148_114890

theorem line_through_point_hyperbola {x y k : ℝ} : 
  (∃ k : ℝ, ∃ x y : ℝ, y = k * (x - 3) ∧ x^2 / 4 - y^2 = 1 ∧ (1 - 4 * k^2) = 0) → 
  (∃! k : ℝ, (k = 1 / 2) ∨ (k = -1 / 2)) := 
sorry

end NUMINAMATH_GPT_line_through_point_hyperbola_l1148_114890


namespace NUMINAMATH_GPT_carol_rectangle_width_l1148_114863

theorem carol_rectangle_width 
  (area_jordan : ℕ) (length_jordan width_jordan : ℕ) (width_carol length_carol : ℕ)
  (h1 : length_jordan = 12)
  (h2 : width_jordan = 10)
  (h3 : width_carol = 24)
  (h4 : area_jordan = length_jordan * width_jordan)
  (h5 : area_jordan = length_carol * width_carol) :
  length_carol = 5 :=
by
  sorry

end NUMINAMATH_GPT_carol_rectangle_width_l1148_114863


namespace NUMINAMATH_GPT_correct_random_variable_l1148_114887

-- Define the given conditions
def total_white_balls := 5
def total_red_balls := 3
def total_balls := total_white_balls + total_red_balls
def balls_drawn := 3

-- Define the random variable
noncomputable def is_random_variable_correct (option : ℕ) :=
  option = 2

-- The theorem to be proved
theorem correct_random_variable: is_random_variable_correct 2 :=
by
  sorry

end NUMINAMATH_GPT_correct_random_variable_l1148_114887


namespace NUMINAMATH_GPT_union_sets_l1148_114803

def M : Set ℝ := {x | -3 < x ∧ x < 1}
def N : Set ℝ := {x | x ≤ -3}

theorem union_sets :
  M ∪ N = {x | x ≤ 1} :=
by
  sorry

end NUMINAMATH_GPT_union_sets_l1148_114803


namespace NUMINAMATH_GPT_worker_b_time_l1148_114836

theorem worker_b_time (time_A : ℝ) (time_A_B_together : ℝ) (T_B : ℝ) 
  (h1 : time_A = 8) 
  (h2 : time_A_B_together = 4.8) 
  (h3 : (1 / time_A) + (1 / T_B) = (1 / time_A_B_together)) :
  T_B = 12 :=
sorry

end NUMINAMATH_GPT_worker_b_time_l1148_114836


namespace NUMINAMATH_GPT_inscribed_circle_radius_l1148_114809

noncomputable def a : ℝ := 5
noncomputable def b : ℝ := 10
noncomputable def c : ℝ := 20

noncomputable def r : ℝ := 1 / (1 / a + 1 / b + 1 / c + 2 * Real.sqrt (1 / (a * b) + 1 / (a * c) + 1 / (b * c)))

theorem inscribed_circle_radius :
  r = 20 / (3.5 + 2 * Real.sqrt 14) :=
sorry

end NUMINAMATH_GPT_inscribed_circle_radius_l1148_114809


namespace NUMINAMATH_GPT_min_n_for_factorization_l1148_114894

theorem min_n_for_factorization (n : ℤ) :
  (∃ A B : ℤ, 6 * A * B = 60 ∧ n = 6 * B + A) → n = 66 :=
sorry

end NUMINAMATH_GPT_min_n_for_factorization_l1148_114894


namespace NUMINAMATH_GPT_shooter_mean_hits_l1148_114852

theorem shooter_mean_hits (p : ℝ) (n : ℕ) (h_prob : p = 0.9) (h_shots : n = 10) : n * p = 9 := by
  sorry

end NUMINAMATH_GPT_shooter_mean_hits_l1148_114852


namespace NUMINAMATH_GPT_order_of_values_l1148_114833

noncomputable def a : ℝ := (1 / 5) ^ 2
noncomputable def b : ℝ := 2 ^ (1 / 5)
noncomputable def c : ℝ := Real.log (1 / 5) / Real.log 2  -- change of base from log base 2 to natural log

theorem order_of_values : c < a ∧ a < b :=
by
  sorry

end NUMINAMATH_GPT_order_of_values_l1148_114833


namespace NUMINAMATH_GPT_chord_length_of_curve_by_line_l1148_114800

theorem chord_length_of_curve_by_line :
  let x (t : ℝ) := 2 + 2 * t
  let y (t : ℝ) := -t
  let curve_eq (θ : ℝ) := 4 * Real.cos θ
  ∃ a b : ℝ, (x a = 2 + 2 * a ∧ y a = -a) ∧ (x b = 2 + 2 * b ∧ y b = -b) ∧
  ((x a - x b)^2 + (y a - y b)^2 = 4^2) :=
by
  sorry

end NUMINAMATH_GPT_chord_length_of_curve_by_line_l1148_114800


namespace NUMINAMATH_GPT_no_such_function_l1148_114819

theorem no_such_function :
  ¬ ∃ f : ℝ → ℝ, ∀ x y : ℝ, f (x + f y) = x - y :=
by
  sorry

end NUMINAMATH_GPT_no_such_function_l1148_114819


namespace NUMINAMATH_GPT_equivalent_annual_rate_l1148_114831

def quarterly_to_annual_rate (quarterly_rate : ℝ) : ℝ :=
  (1 + quarterly_rate) ^ 4 - 1

def to_percentage (rate : ℝ) : ℝ :=
  rate * 100

theorem equivalent_annual_rate (quarterly_rate : ℝ) (annual_rate : ℝ) :
  quarterly_rate = 0.02 →
  annual_rate = quarterly_to_annual_rate quarterly_rate →
  to_percentage annual_rate = 8.24 :=
by
  intros
  sorry

end NUMINAMATH_GPT_equivalent_annual_rate_l1148_114831


namespace NUMINAMATH_GPT_value_of_r6_plus_s6_l1148_114812

theorem value_of_r6_plus_s6 :
  ∀ r s : ℝ, (r^2 - 2 * r + Real.sqrt 2 = 0) ∧ (s^2 - 2 * s + Real.sqrt 2 = 0) →
  (r^6 + s^6 = 904 - 640 * Real.sqrt 2) :=
by
  intros r s h
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_value_of_r6_plus_s6_l1148_114812


namespace NUMINAMATH_GPT_range_of_a_for_maximum_l1148_114884

variable {f : ℝ → ℝ}
variable {a : ℝ}

theorem range_of_a_for_maximum (h : ∀ x, deriv f x = a * (x + 1) * (x - a))
  (h_max : ∀ x, f x ≤ f a → x = a) : -1 < a ∧ a < 0 :=
sorry

end NUMINAMATH_GPT_range_of_a_for_maximum_l1148_114884


namespace NUMINAMATH_GPT_total_distance_travelled_l1148_114842

theorem total_distance_travelled (distance_to_market : ℕ) (travel_time_minutes : ℕ) (speed_mph : ℕ) 
  (h1 : distance_to_market = 30) 
  (h2 : travel_time_minutes = 30) 
  (h3 : speed_mph = 20) : 
  (distance_to_market + ((travel_time_minutes / 60) * speed_mph) = 40) :=
by
  sorry

end NUMINAMATH_GPT_total_distance_travelled_l1148_114842


namespace NUMINAMATH_GPT_min_m_plus_n_l1148_114861

theorem min_m_plus_n (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) (h3 : 32 * m = n^5) : m + n = 3 :=
  sorry

end NUMINAMATH_GPT_min_m_plus_n_l1148_114861


namespace NUMINAMATH_GPT_sum_positive_implies_at_least_one_positive_l1148_114848

variables {a b : ℝ}

theorem sum_positive_implies_at_least_one_positive (h : a + b > 0) : a > 0 ∨ b > 0 :=
sorry

end NUMINAMATH_GPT_sum_positive_implies_at_least_one_positive_l1148_114848


namespace NUMINAMATH_GPT_composite_numbers_l1148_114882

theorem composite_numbers (n : ℕ) (hn : n > 0) :
  (∃ p q, p > 1 ∧ q > 1 ∧ 2 * 2^(2^n) + 1 = p * q) ∧ 
  (∃ p q, p > 1 ∧ q > 1 ∧ 3 * 2^(2*n) + 1 = p * q) :=
sorry

end NUMINAMATH_GPT_composite_numbers_l1148_114882


namespace NUMINAMATH_GPT_prod_lcm_gcd_eq_216_l1148_114885

theorem prod_lcm_gcd_eq_216 (a b : ℕ) (h1 : a = 12) (h2 : b = 18) :
  (Nat.gcd a b) * (Nat.lcm a b) = 216 := by
  sorry

end NUMINAMATH_GPT_prod_lcm_gcd_eq_216_l1148_114885


namespace NUMINAMATH_GPT_ratio_doubled_to_original_l1148_114873

theorem ratio_doubled_to_original (x : ℝ) (h : 3 * (2 * x + 9) = 69) : (2 * x) / x = 2 :=
by
  -- We skip the proof here.
  sorry

end NUMINAMATH_GPT_ratio_doubled_to_original_l1148_114873


namespace NUMINAMATH_GPT_percentage_calculation_l1148_114877

theorem percentage_calculation :
  let total_amt := 1600
  let pct_25 := 0.25 * total_amt
  let pct_5 := 0.05 * pct_25
  pct_5 = 20 := by
sorry

end NUMINAMATH_GPT_percentage_calculation_l1148_114877


namespace NUMINAMATH_GPT_find_ab_average_l1148_114879

variable (a b c k : ℝ)

-- Conditions
def sum_condition : Prop := (4 + 6 + 8 + 12 + a + b + c) / 7 = 20
def abc_condition : Prop := a + b + c = 3 * ((4 + 6 + 8) / 3)

-- Theorem
theorem find_ab_average 
  (sum_cond : sum_condition a b c) 
  (abc_cond : abc_condition a b c) 
  (c_eq_k : c = k) : 
  (a + b) / 2 = (18 - k) / 2 :=
sorry  -- Proof is omitted


end NUMINAMATH_GPT_find_ab_average_l1148_114879


namespace NUMINAMATH_GPT_gcd_102_238_l1148_114822

theorem gcd_102_238 : Nat.gcd 102 238 = 34 :=
by
  -- Given conditions as part of proof structure
  have h1 : 238 = 102 * 2 + 34 := by rfl
  have h2 : 102 = 34 * 3 := by rfl
  sorry

end NUMINAMATH_GPT_gcd_102_238_l1148_114822


namespace NUMINAMATH_GPT_isoperimetric_inequality_l1148_114820

theorem isoperimetric_inequality (S : ℝ) (P : ℝ) : S ≤ P^2 / (4 * Real.pi) :=
sorry

end NUMINAMATH_GPT_isoperimetric_inequality_l1148_114820


namespace NUMINAMATH_GPT_train_passes_bridge_in_52_seconds_l1148_114866

def length_of_train : ℕ := 510
def speed_of_train_kmh : ℕ := 45
def length_of_bridge : ℕ := 140
def total_distance := length_of_train + length_of_bridge
def speed_of_train_ms := speed_of_train_kmh * 1000 / 3600
def time_to_pass_bridge := total_distance / speed_of_train_ms

theorem train_passes_bridge_in_52_seconds :
  time_to_pass_bridge = 52 := sorry

end NUMINAMATH_GPT_train_passes_bridge_in_52_seconds_l1148_114866


namespace NUMINAMATH_GPT_kenneth_left_with_amount_l1148_114846

theorem kenneth_left_with_amount (total_earnings : ℝ) (percentage_spent : ℝ) (amount_left : ℝ) 
    (h_total_earnings : total_earnings = 450) (h_percentage_spent : percentage_spent = 0.10) 
    (h_spent_amount : total_earnings * percentage_spent = 45) : 
    amount_left = total_earnings - total_earnings * percentage_spent :=
by sorry

end NUMINAMATH_GPT_kenneth_left_with_amount_l1148_114846


namespace NUMINAMATH_GPT_r_iterated_six_times_l1148_114891

def r (θ : ℚ) : ℚ := 1 / (1 - 2 * θ)

theorem r_iterated_six_times (θ : ℚ) : r (r (r (r (r (r θ))))) = θ :=
by sorry

example : r (r (r (r (r (r 10))))) = 10 :=
by rw [r_iterated_six_times 10]

end NUMINAMATH_GPT_r_iterated_six_times_l1148_114891


namespace NUMINAMATH_GPT_charlyn_visible_area_l1148_114874

noncomputable def visible_area (side_length vision_distance : ℝ) : ℝ :=
  let outer_rectangles_area := 4 * (side_length * vision_distance)
  let outer_squares_area := 4 * (vision_distance * vision_distance)
  let inner_square_area := 
    let inner_side_length := side_length - 2 * vision_distance
    inner_side_length * inner_side_length
  let total_walk_area := side_length * side_length
  total_walk_area - inner_square_area + outer_rectangles_area + outer_squares_area

theorem charlyn_visible_area :
  visible_area 10 2 = 160 := by
  sorry

end NUMINAMATH_GPT_charlyn_visible_area_l1148_114874


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l1148_114840

theorem arithmetic_sequence_common_difference (a_1 d : ℝ) (S : ℕ → ℝ)
  (hS2 : S 2 = 4) (hS4 : S 4 = 20)
  (hS_formula : ∀ n, S n = n / 2 * (2 * a_1 + (n - 1) * d)) : 
  d = 3 :=
by sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l1148_114840


namespace NUMINAMATH_GPT_part1_part2_l1148_114835

open Real

variable (A B C a b c : ℝ)

-- Conditions
variable (h1 : b * sin A = a * cos B)
variable (h2 : b = 3)
variable (h3 : sin C = 2 * sin A)

theorem part1 : B = π / 4 := 
  sorry

theorem part2 : ∃ a c, c = 2 * a ∧ 9 = a^2 + c^2 - 2 * a * c * cos (π / 4) := 
  sorry

end NUMINAMATH_GPT_part1_part2_l1148_114835


namespace NUMINAMATH_GPT_sin_nine_pi_over_two_plus_theta_l1148_114834

variable (θ : ℝ)

-- Conditions: Point A(4, -3) lies on the terminal side of angle θ
def terminal_point_on_angle (θ : ℝ) : Prop :=
  let x := 4
  let y := -3
  let hypotenuse := Real.sqrt ((x ^ 2) + (y ^ 2))
  hypotenuse = 5 ∧ Real.cos θ = x / hypotenuse

theorem sin_nine_pi_over_two_plus_theta (θ : ℝ) 
  (h : terminal_point_on_angle θ) : 
  Real.sin (9 * Real.pi / 2 + θ) = 4 / 5 :=
sorry

end NUMINAMATH_GPT_sin_nine_pi_over_two_plus_theta_l1148_114834


namespace NUMINAMATH_GPT_carl_max_value_l1148_114889

-- Definitions based on problem conditions.
def value_of_six_pound_rock : ℕ := 20
def weight_of_six_pound_rock : ℕ := 6
def value_of_three_pound_rock : ℕ := 9
def weight_of_three_pound_rock : ℕ := 3
def value_of_two_pound_rock : ℕ := 4
def weight_of_two_pound_rock : ℕ := 2
def max_weight_carl_can_carry : ℕ := 24

/-- Proves that Carl can carry rocks worth maximum 80 dollars given the conditions. -/
theorem carl_max_value : ∃ (n m k : ℕ),
    n * weight_of_six_pound_rock + m * weight_of_three_pound_rock + k * weight_of_two_pound_rock ≤ max_weight_carl_can_carry ∧
    n * value_of_six_pound_rock + m * value_of_three_pound_rock + k * value_of_two_pound_rock = 80 :=
by
  sorry

end NUMINAMATH_GPT_carl_max_value_l1148_114889


namespace NUMINAMATH_GPT_Parabola_vertex_form_l1148_114853

theorem Parabola_vertex_form (x : ℝ) (y : ℝ) : 
  (∃ h k : ℝ, (h = -2) ∧ (k = 1) ∧ (y = (x + h)^2 + k) ) ↔ (y = (x + 2)^2 + 1) :=
by
  sorry

end NUMINAMATH_GPT_Parabola_vertex_form_l1148_114853


namespace NUMINAMATH_GPT_two_numbers_sum_gcd_l1148_114823

theorem two_numbers_sum_gcd (x y : ℕ) (h1 : x + y = 432) (h2 : Nat.gcd x y = 36) :
  (x = 36 ∧ y = 396) ∨ (x = 180 ∧ y = 252) ∨ (x = 396 ∧ y = 36) ∨ (x = 252 ∧ y = 180) :=
by
  -- Proof TBD
  sorry

end NUMINAMATH_GPT_two_numbers_sum_gcd_l1148_114823


namespace NUMINAMATH_GPT_find_y_l1148_114892

theorem find_y (x y : ℝ) (h1 : 3 * x + 2 = 2) (h2 : y - x = 2) : y = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l1148_114892


namespace NUMINAMATH_GPT_clara_biked_more_l1148_114865

def clara_speed : ℕ := 18
def denise_speed : ℕ := 16
def race_duration : ℕ := 5

def clara_distance := clara_speed * race_duration
def denise_distance := denise_speed * race_duration
def distance_difference := clara_distance - denise_distance

theorem clara_biked_more : distance_difference = 10 := by
  sorry

end NUMINAMATH_GPT_clara_biked_more_l1148_114865


namespace NUMINAMATH_GPT_squat_percentage_loss_l1148_114872

variable (original_squat : ℕ)
variable (original_bench : ℕ)
variable (original_deadlift : ℕ)
variable (lost_deadlift : ℕ)
variable (new_total : ℕ)
variable (unchanged_bench : ℕ)

theorem squat_percentage_loss
  (h1 : original_squat = 700)
  (h2 : original_bench = 400)
  (h3 : original_deadlift = 800)
  (h4 : lost_deadlift = 200)
  (h5 : new_total = 1490)
  (h6 : unchanged_bench = 400) :
  (original_squat - (new_total - (unchanged_bench + (original_deadlift - lost_deadlift)))) * 100 / original_squat = 30 :=
by sorry

end NUMINAMATH_GPT_squat_percentage_loss_l1148_114872


namespace NUMINAMATH_GPT_boys_test_l1148_114818

-- Define the conditions
def passing_time : ℝ := 14
def test_results : List ℝ := [0.6, -1.1, 0, -0.2, 2, 0.5]

-- Define the proof problem
theorem boys_test (number_did_not_pass : ℕ) (fastest_time : ℝ) (average_score : ℝ) :
  passing_time = 14 →
  test_results = [0.6, -1.1, 0, -0.2, 2, 0.5] →
  number_did_not_pass = 3 ∧
  fastest_time = 12.9 ∧
  average_score = 14.3 :=
by
  intros
  sorry

end NUMINAMATH_GPT_boys_test_l1148_114818


namespace NUMINAMATH_GPT_extreme_value_point_of_f_l1148_114824

noncomputable def f (x : ℝ) : ℝ := sorry -- Assume the definition of f that derives this f'

def f' (x : ℝ) : ℝ := x^3 - 3 * x + 2

theorem extreme_value_point_of_f : (∃ x : ℝ, x = -2 ∧ ∀ y : ℝ, y ≠ -2 → f' y < 0) := sorry

end NUMINAMATH_GPT_extreme_value_point_of_f_l1148_114824


namespace NUMINAMATH_GPT_area_of_shaded_region_l1148_114857

def parallelogram_exists (EFGH : Type) : Prop :=
  ∃ (E F G H : EFGH) (EJ JH EH : ℝ) (height : ℝ), EJ + JH = EH ∧ EH = 12 ∧ JH = 8 ∧ height = 10

theorem area_of_shaded_region {EFGH : Type} (h : parallelogram_exists EFGH) : 
  ∃ (area_shaded : ℝ), area_shaded = 100 := 
by
  sorry

end NUMINAMATH_GPT_area_of_shaded_region_l1148_114857


namespace NUMINAMATH_GPT_find_m_l1148_114838

theorem find_m (m : ℝ) (P : Set ℝ) (Q : Set ℝ) (hP : P = {m^2 - 4, m + 1, -3})
  (hQ : Q = {m - 3, 2 * m - 1, 3 * m + 1}) (h_intersect : P ∩ Q = {-3}) :
  m = -4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l1148_114838


namespace NUMINAMATH_GPT_remaining_surface_area_correct_l1148_114854

noncomputable def remaining_surface_area (a : ℕ) (c : ℕ) : ℕ :=
  let original_surface_area := 6 * a^2
  let corner_cube_area := 3 * c^2
  let net_change := corner_cube_area - corner_cube_area
  original_surface_area + 8 * net_change 

theorem remaining_surface_area_correct :
  remaining_surface_area 4 1 = 96 := by
  sorry

end NUMINAMATH_GPT_remaining_surface_area_correct_l1148_114854


namespace NUMINAMATH_GPT_train_length_l1148_114897

theorem train_length (v_train_kmph : ℝ) (v_man_kmph : ℝ) (time_sec : ℝ) 
  (h1 : v_train_kmph = 25) 
  (h2 : v_man_kmph = 2) 
  (h3 : time_sec = 20) : 
  (150 : ℝ) = (v_train_kmph + v_man_kmph) * (1000 / 3600) * time_sec := 
by {
  -- sorry for the steps here
  sorry
}

end NUMINAMATH_GPT_train_length_l1148_114897


namespace NUMINAMATH_GPT_trigonometric_quadrant_l1148_114801

theorem trigonometric_quadrant (α : ℝ) (h1 : Real.cos α < 0) (h2 : Real.sin α > 0) : 
  (π / 2 < α) ∧ (α < π) :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_quadrant_l1148_114801


namespace NUMINAMATH_GPT_pies_per_day_l1148_114898

theorem pies_per_day (daily_pies total_pies : ℕ) (h1 : daily_pies = 8) (h2 : total_pies = 56) :
  total_pies / daily_pies = 7 :=
by sorry

end NUMINAMATH_GPT_pies_per_day_l1148_114898


namespace NUMINAMATH_GPT_fraction_identity_l1148_114808

variable (a b : ℝ)

theorem fraction_identity (h : a ≠ 0) : 
  (2 * b + a) / a + (a - 2 * b) / a = 2 := 
by
  sorry

end NUMINAMATH_GPT_fraction_identity_l1148_114808


namespace NUMINAMATH_GPT_shirt_cost_l1148_114807

theorem shirt_cost
  (J S B : ℝ)
  (h1 : 3 * J + 2 * S = 69)
  (h2 : 2 * J + 3 * S = 61)
  (h3 : 3 * J + 3 * S + 2 * B = 90) :
  S = 9 := 
by
  sorry

end NUMINAMATH_GPT_shirt_cost_l1148_114807


namespace NUMINAMATH_GPT_sphere_surface_area_l1148_114896

theorem sphere_surface_area (V : ℝ) (hV : V = 72 * Real.pi) : 
  ∃ S : ℝ, S = 36 * 2^(2/3) * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_sphere_surface_area_l1148_114896


namespace NUMINAMATH_GPT_expand_product_l1148_114828

theorem expand_product : (2 : ℝ) * (x + 2) * (x + 3) * (x + 4) = 2 * x^3 + 18 * x^2 + 52 * x + 48 :=
by
  sorry

end NUMINAMATH_GPT_expand_product_l1148_114828


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l1148_114851

theorem quadratic_inequality_solution (k : ℝ) :
  (∀ x : ℝ, x^2 - (k - 2) * x - 2 * k + 4 < 0) ↔ (-6 < k ∧ k < 2) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l1148_114851


namespace NUMINAMATH_GPT_bus_problem_l1148_114814

theorem bus_problem (x : ℕ) : 50 * x + 10 = 52 * x + 2 := 
sorry

end NUMINAMATH_GPT_bus_problem_l1148_114814


namespace NUMINAMATH_GPT_students_passed_finals_l1148_114813

def total_students := 180
def students_bombed := 1 / 4 * total_students
def remaining_students_after_bombed := total_students - students_bombed
def students_didnt_show := 1 / 3 * remaining_students_after_bombed
def students_failed_less_than_D := 20

theorem students_passed_finals : 
  total_students - students_bombed - students_didnt_show - students_failed_less_than_D = 70 := 
by 
  -- calculation to derive 70
  sorry

end NUMINAMATH_GPT_students_passed_finals_l1148_114813


namespace NUMINAMATH_GPT_calculate_product_l1148_114815

variable (EF FG GH HE : ℚ)
variable (x y : ℚ)

-- Conditions
axiom h1 : EF = 110
axiom h2 : FG = 16 * y^3
axiom h3 : GH = 6 * x + 2
axiom h4 : HE = 64
-- Parallelogram properties
axiom h5 : EF = GH
axiom h6 : FG = HE

theorem calculate_product (EF FG GH HE : ℚ) (x y : ℚ)
  (h1 : EF = 110) (h2 : FG = 16 * y ^ 3) (h3 : GH = 6 * x + 2) (h4 : HE = 64) (h5 : EF = GH) (h6 : FG = HE) :
  x * y = 18 * (4) ^ (1/3) := by
  sorry

end NUMINAMATH_GPT_calculate_product_l1148_114815


namespace NUMINAMATH_GPT_direct_proportion_m_n_l1148_114843

theorem direct_proportion_m_n (m n : ℤ) (h₁ : m - 2 = 1) (h₂ : n + 1 = 0) : m + n = 2 :=
by
  sorry

end NUMINAMATH_GPT_direct_proportion_m_n_l1148_114843


namespace NUMINAMATH_GPT_total_hair_cut_l1148_114895

-- Define the amounts cut on two consecutive days
def first_cut : ℝ := 0.375
def second_cut : ℝ := 0.5

-- Statement: Prove that the total amount cut off is 0.875 inches
theorem total_hair_cut : first_cut + second_cut = 0.875 :=
by {
  -- The exact proof would go here
  sorry
}

end NUMINAMATH_GPT_total_hair_cut_l1148_114895


namespace NUMINAMATH_GPT_slope_of_line_determined_by_any_two_solutions_l1148_114856

theorem slope_of_line_determined_by_any_two_solutions 
  (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : 4 / x₁ + 5 / y₁ = 0) 
  (h₂ : 4 / x₂ + 5 / y₂ = 0) 
  (h_distinct : x₁ ≠ x₂) : 
  (y₂ - y₁) / (x₂ - x₁) = -5 / 4 := 
sorry

end NUMINAMATH_GPT_slope_of_line_determined_by_any_two_solutions_l1148_114856


namespace NUMINAMATH_GPT_smallest_t_l1148_114826

theorem smallest_t (p q r : ℕ) (h₁ : 0 < p) (h₂ : 0 < q) (h₃ : 0 < r) (h₄ : p + q + r = 2510) 
                   (k : ℕ) (t : ℕ) (h₅ : p! * q! * r! = k * 10^t) (h₆ : ¬(10 ∣ k)) : t = 626 := 
by sorry

end NUMINAMATH_GPT_smallest_t_l1148_114826


namespace NUMINAMATH_GPT_chandra_pairings_l1148_114827

theorem chandra_pairings : 
  let bowls := 5
  let glasses := 6
  (bowls * glasses) = 30 :=
by
  sorry

end NUMINAMATH_GPT_chandra_pairings_l1148_114827


namespace NUMINAMATH_GPT_cars_meet_in_3_hours_l1148_114805

theorem cars_meet_in_3_hours
  (distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) (t : ℝ)
  (h_distance: distance = 333)
  (h_speed1: speed1 = 54)
  (h_speed2: speed2 = 57)
  (h_equation: speed1 * t + speed2 * t = distance) :
  t = 3 :=
sorry

end NUMINAMATH_GPT_cars_meet_in_3_hours_l1148_114805


namespace NUMINAMATH_GPT_find_x_satisfying_conditions_l1148_114870

theorem find_x_satisfying_conditions :
  ∃ x : ℕ, (x % 2 = 1) ∧ (x % 3 = 2) ∧ (x % 4 = 3) ∧ (x % 5 = 4) ∧ x = 59 :=
by
  sorry

end NUMINAMATH_GPT_find_x_satisfying_conditions_l1148_114870


namespace NUMINAMATH_GPT_dartboard_central_angle_l1148_114855

theorem dartboard_central_angle (A : ℝ) (x : ℝ) (P : ℝ) (h1 : P = 1 / 4) 
    (h2 : A > 0) : (x / 360 = 1 / 4) -> x = 90 :=
by
  sorry

end NUMINAMATH_GPT_dartboard_central_angle_l1148_114855


namespace NUMINAMATH_GPT_car_speed_reduction_and_increase_l1148_114899

theorem car_speed_reduction_and_increase (V x : ℝ)
  (h1 : V > 0) -- V is positive
  (h2 : V * (1 - x / 100) * (1 + 0.5 * x / 100) = V * (1 - 0.6 * x / 100)) :
  x = 20 :=
sorry

end NUMINAMATH_GPT_car_speed_reduction_and_increase_l1148_114899


namespace NUMINAMATH_GPT_carrie_hours_per_week_l1148_114871

variable (H : ℕ)

def carrie_hourly_wage : ℕ := 8
def cost_of_bike : ℕ := 400
def amount_left_over : ℕ := 720
def weeks_worked : ℕ := 4
def total_earnings : ℕ := cost_of_bike + amount_left_over

theorem carrie_hours_per_week :
  (weeks_worked * H * carrie_hourly_wage = total_earnings) →
  H = 35 := by
  sorry

end NUMINAMATH_GPT_carrie_hours_per_week_l1148_114871


namespace NUMINAMATH_GPT_father_age_is_32_l1148_114869

noncomputable def father_age (D F : ℕ) : Prop :=
  F = 4 * D ∧ (F + 5) + (D + 5) = 50

theorem father_age_is_32 (D F : ℕ) (h : father_age D F) : F = 32 :=
by
  sorry

end NUMINAMATH_GPT_father_age_is_32_l1148_114869


namespace NUMINAMATH_GPT_total_pages_in_book_l1148_114858

theorem total_pages_in_book (pages_monday pages_tuesday total_pages_read total_pages_book : ℝ)
    (h1 : pages_monday = 15.5)
    (h2 : pages_tuesday = 1.5 * pages_monday + 16)
    (h3 : total_pages_read = pages_monday + pages_tuesday)
    (h4 : total_pages_book = 2 * total_pages_read) :
    total_pages_book = 109.5 :=
by
  sorry

end NUMINAMATH_GPT_total_pages_in_book_l1148_114858


namespace NUMINAMATH_GPT_find_ab_l1148_114810

variable (a b : ℝ)

theorem find_ab (h1 : a + b = 4) (h2 : a^3 + b^3 = 136) : a * b = -6 := by
  sorry

end NUMINAMATH_GPT_find_ab_l1148_114810


namespace NUMINAMATH_GPT_find_x_l1148_114881

theorem find_x (x y : ℝ)
  (h1 : 2 * x + (x - 30) = 360)
  (h2 : y = x - 30)
  (h3 : 2 * x = 4 * y) :
  x = 130 := 
sorry

end NUMINAMATH_GPT_find_x_l1148_114881


namespace NUMINAMATH_GPT_ellipse_tangent_line_equation_l1148_114806

variable {r a b x0 y0 x y : ℝ}
variable (h_r_pos : r > 0) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_ineq : a > b)
variable (ellipse_eq : (x / a)^2 + (y / b)^2 = 1)
variable (tangent_circle_eq : x0 * x / r^2 + y0 * y / r^2 = 1)

theorem ellipse_tangent_line_equation :
  (a > b) → (a > 0) → (b > 0) → (x0 ≠ 0 ∨ y0 ≠ 0) → (x/a)^2 + (y/b)^2 = 1 →
  (x0 * x / a^2 + y0 * y / b^2 = 1) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_tangent_line_equation_l1148_114806


namespace NUMINAMATH_GPT_k_values_equation_satisfied_l1148_114837

theorem k_values_equation_satisfied : 
  {k : ℕ | k > 0 ∧ ∃ r s : ℕ, r > 0 ∧ s > 0 ∧ (k^2 - 6 * k + 11)^(r - 1) = (2 * k - 7)^s} = {2, 3, 4, 8} :=
by
  sorry

end NUMINAMATH_GPT_k_values_equation_satisfied_l1148_114837


namespace NUMINAMATH_GPT_sum_lent_is_1000_l1148_114868

theorem sum_lent_is_1000
    (P : ℝ)
    (r : ℝ)
    (t : ℝ)
    (I : ℝ)
    (h1 : r = 5)
    (h2 : t = 5)
    (h3 : I = P - 750)
    (h4 : I = P * r * t / 100) :
  P = 1000 :=
by sorry

end NUMINAMATH_GPT_sum_lent_is_1000_l1148_114868


namespace NUMINAMATH_GPT_total_cost_of_topsoil_l1148_114817

def cost_per_cubic_foot : ℝ := 8
def cubic_yards_to_cubic_feet : ℝ := 27
def volume_in_yards : ℝ := 7

theorem total_cost_of_topsoil :
  (cubic_yards_to_cubic_feet * volume_in_yards) * cost_per_cubic_foot = 1512 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_of_topsoil_l1148_114817


namespace NUMINAMATH_GPT_rational_square_root_l1148_114864

theorem rational_square_root {x y : ℚ} 
  (h : (x^2 + y^2 - 2) * (x + y)^2 + (xy + 1)^2 = 0) : 
  ∃ r : ℚ, r * r = 1 + x * y := 
sorry

end NUMINAMATH_GPT_rational_square_root_l1148_114864


namespace NUMINAMATH_GPT_subtraction_identity_l1148_114859

theorem subtraction_identity : 4444444444444 - 2222222222222 - 444444444444 = 1777777777778 :=
  by norm_num

end NUMINAMATH_GPT_subtraction_identity_l1148_114859


namespace NUMINAMATH_GPT_augmented_matrix_solution_l1148_114878

theorem augmented_matrix_solution (c₁ c₂ : ℝ) (x y : ℝ) 
  (h1 : 2 * x + 3 * y = c₁) (h2 : 3 * x + 2 * y = c₂)
  (hx : x = 2) (hy : y = 1) : c₁ - c₂ = -1 := 
by
  sorry

end NUMINAMATH_GPT_augmented_matrix_solution_l1148_114878


namespace NUMINAMATH_GPT_triangle_side_inequality_l1148_114886

theorem triangle_side_inequality (a b c : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : 1 = 1 / 2 * b * c) : b ≥ Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_triangle_side_inequality_l1148_114886


namespace NUMINAMATH_GPT_surface_area_of_parallelepiped_l1148_114811

open Real

theorem surface_area_of_parallelepiped 
  (a b c : ℝ)
  (x y z : ℝ)
  (h1: a^2 = x^2 + y^2)
  (h2: b^2 = x^2 + z^2)
  (h3: c^2 = y^2 + z^2) :
  2 * (sqrt ((x * y)) + sqrt ((x * z)) + sqrt ((y * z)))  =
  sqrt ((a^2 + b^2 - c^2) * (a^2 + c^2 - b^2)) +
  sqrt ((a^2 + b^2 - c^2) * (b^2 + c^2 - a^2)) +
  sqrt ((a^2 + c^2 - b^2) * (b^2 + c^2 - a^2)) :=
by
  sorry

end NUMINAMATH_GPT_surface_area_of_parallelepiped_l1148_114811


namespace NUMINAMATH_GPT_rectangle_width_l1148_114850

theorem rectangle_width (length_rect : ℝ) (width_rect : ℝ) (side_square : ℝ)
  (h1 : side_square * side_square = 5 * (length_rect * width_rect))
  (h2 : length_rect = 125)
  (h3 : 4 * side_square = 800) : width_rect = 64 :=
by 
  sorry

end NUMINAMATH_GPT_rectangle_width_l1148_114850


namespace NUMINAMATH_GPT_triangle_area_hypotenuse_l1148_114821

-- Definitions of the conditions
def DE : ℝ := 40
def DF : ℝ := 30
def angleD : ℝ := 90

-- Proof statement
theorem triangle_area_hypotenuse :
  let Area : ℝ := 1 / 2 * DE * DF
  let EF : ℝ := Real.sqrt (DE^2 + DF^2)
  Area = 600 ∧ EF = 50 := by
  sorry

end NUMINAMATH_GPT_triangle_area_hypotenuse_l1148_114821


namespace NUMINAMATH_GPT_arithmetic_sequence_100_l1148_114816

variable {a : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

variables (S₉ : ℝ) (a₁₀ : ℝ)

theorem arithmetic_sequence_100
  (h1: is_arithmetic_sequence a)
  (h2: S₉ = 27) 
  (h3: a₁₀ = 8): 
  a 100 = 98 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_100_l1148_114816


namespace NUMINAMATH_GPT_find_a_plus_b_l1148_114832

def smallest_two_digit_multiple_of_five : ℕ := 10
def smallest_three_digit_multiple_of_seven : ℕ := 105

theorem find_a_plus_b :
  let a := smallest_two_digit_multiple_of_five
  let b := smallest_three_digit_multiple_of_seven
  a + b = 115 := by
  sorry

end NUMINAMATH_GPT_find_a_plus_b_l1148_114832


namespace NUMINAMATH_GPT_find_sum_pqr_l1148_114830

theorem find_sum_pqr (p q r : ℕ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) 
  (h : (p + q + r)^3 - p^3 - q^3 - r^3 = 200) : 
  p + q + r = 7 :=
by 
  sorry

end NUMINAMATH_GPT_find_sum_pqr_l1148_114830


namespace NUMINAMATH_GPT_average_percentage_taller_l1148_114888

theorem average_percentage_taller 
  (h1 b1 h2 b2 h3 b3 : ℝ)
  (h1_eq : h1 = 228) (b1_eq : b1 = 200)
  (h2_eq : h2 = 120) (b2_eq : b2 = 100)
  (h3_eq : h3 = 147) (b3_eq : b3 = 140) :
  ((h1 - b1) / b1 * 100 + (h2 - b2) / b2 * 100 + (h3 - b3) / b3 * 100) / 3 = 13 := by
  rw [h1_eq, b1_eq, h2_eq, b2_eq, h3_eq, b3_eq]
  sorry

end NUMINAMATH_GPT_average_percentage_taller_l1148_114888


namespace NUMINAMATH_GPT_employees_excluding_manager_l1148_114847

theorem employees_excluding_manager (E : ℕ) (avg_salary_employee : ℕ) (manager_salary : ℕ) (new_avg_salary : ℕ) (total_employees_with_manager : ℕ) :
  avg_salary_employee = 1800 →
  manager_salary = 4200 →
  new_avg_salary = avg_salary_employee + 150 →
  total_employees_with_manager = E + 1 →
  (1800 * E + 4200) / total_employees_with_manager = new_avg_salary →
  E = 15 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_employees_excluding_manager_l1148_114847


namespace NUMINAMATH_GPT_polygon_sides_sum_l1148_114804

theorem polygon_sides_sum (n : ℕ) (x : ℝ) (hx : 0 < x ∧ x < 180) 
  (h_sum : 180 * (n - 2) - x = 2190) : n = 15 :=
sorry

end NUMINAMATH_GPT_polygon_sides_sum_l1148_114804


namespace NUMINAMATH_GPT_total_weight_proof_l1148_114883

-- Definitions of the variables and conditions given in the problem
variable (M D C : ℕ)
variable (h1 : D + C = 60)  -- Daughter and grandchild together weigh 60 kg
variable (h2 : C = 1 / 5 * M)  -- Grandchild's weight is 1/5th of grandmother's weight
variable (h3 : D = 42)  -- Daughter's weight is 42 kg

-- The goal is to prove the total weight is 150 kg
theorem total_weight_proof (M D C : ℕ) (h1 : D + C = 60) (h2 : C = 1 / 5 * M) (h3 : D = 42) :
  M + D + C = 150 :=
by
  sorry

end NUMINAMATH_GPT_total_weight_proof_l1148_114883


namespace NUMINAMATH_GPT_calculate_rate_l1148_114880

-- Definitions corresponding to the conditions in the problem
def bankers_gain (td : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  td * rate * time

-- Given values according to the problem
def BG : ℝ := 7.8
def TD : ℝ := 65
def Time : ℝ := 1
def expected_rate_percentage : ℝ := 12

-- The mathematical proof problem statement in Lean 4
theorem calculate_rate : (BG = bankers_gain TD (expected_rate_percentage / 100) Time) :=
sorry

end NUMINAMATH_GPT_calculate_rate_l1148_114880


namespace NUMINAMATH_GPT_mark_height_feet_l1148_114893

theorem mark_height_feet
  (mark_height_inches : ℕ)
  (mike_height_feet : ℕ)
  (mike_height_inches : ℕ)
  (mike_taller_than_mark : ℕ)
  (foot_in_inches : ℕ)
  (mark_height_eq : mark_height_inches = 3)
  (mike_height_eq : mike_height_feet * foot_in_inches + mike_height_inches = 73)
  (mike_taller_eq : mike_height_feet * foot_in_inches + mike_height_inches = mark_height_inches + mike_taller_than_mark)
  (foot_in_inches_eq : foot_in_inches = 12) :
  mark_height_inches = 63 ∧ mark_height_inches / foot_in_inches = 5 := by
sorry

end NUMINAMATH_GPT_mark_height_feet_l1148_114893


namespace NUMINAMATH_GPT_inequality_solution_l1148_114862

theorem inequality_solution : {x : ℝ | -2 < (x^2 - 12 * x + 20) / (x^2 - 4 * x + 8) ∧ (x^2 - 12 * x + 20) / (x^2 - 4 * x + 8) < 2} = {x : ℝ | 5 < x} := 
sorry

end NUMINAMATH_GPT_inequality_solution_l1148_114862


namespace NUMINAMATH_GPT_negation_proof_l1148_114839

theorem negation_proof (x : ℝ) : ¬ (x^2 - x + 3 > 0) ↔ (x^2 - x + 3 ≤ 0) :=
by sorry

end NUMINAMATH_GPT_negation_proof_l1148_114839


namespace NUMINAMATH_GPT_lower_bound_for_x_l1148_114845

variable {x y : ℝ}  -- declaring x and y as real numbers

theorem lower_bound_for_x 
  (h₁ : 3 < x) (h₂ : x < 6)
  (h₃ : 6 < y) (h₄ : y < 8)
  (h₅ : y - x = 4) : 
  ∃ ε > 0, 3 + ε = x := 
sorry

end NUMINAMATH_GPT_lower_bound_for_x_l1148_114845


namespace NUMINAMATH_GPT_geometric_sequence_product_l1148_114849

theorem geometric_sequence_product (a : ℕ → ℝ) (r : ℝ) (h_geom : ∀ n, a (n+1) = r * a n) (h_cond : a 7 * a 12 = 5) :
  a 8 * a 9 * a 10 * a 11 = 25 :=
by 
  sorry

end NUMINAMATH_GPT_geometric_sequence_product_l1148_114849


namespace NUMINAMATH_GPT_inverse_square_relationship_l1148_114860

theorem inverse_square_relationship (k : ℝ) (y : ℝ) (h1 : ∀ x y, x = k / y^2)
  (h2 : ∃ y, 1 = k / y^2) (h3 : 0.5625 = k / 4^2) :
  ∃ y, 1 = 9 / y^2 ∧ y = 3 :=
by
  sorry

end NUMINAMATH_GPT_inverse_square_relationship_l1148_114860


namespace NUMINAMATH_GPT_fraction_integer_solution_l1148_114829

theorem fraction_integer_solution (x y : ℝ) (h₁ : 3 < (x - y) / (x + y)) (h₂ : (x - y) / (x + y) < 8) (h₃ : ∃ t : ℤ, x = t * y) : ∃ t : ℤ, t = -1 := 
sorry

end NUMINAMATH_GPT_fraction_integer_solution_l1148_114829


namespace NUMINAMATH_GPT_sphere_radius_eq_l1148_114802

theorem sphere_radius_eq (h d : ℝ) (r_cylinder : ℝ) (r : ℝ) (pi : ℝ) 
  (h_eq : h = 14) (d_eq : d = 14) (r_cylinder_eq : r_cylinder = d / 2) :
  4 * pi * r^2 = 2 * pi * r_cylinder * h → r = 7 := by
  sorry

end NUMINAMATH_GPT_sphere_radius_eq_l1148_114802
