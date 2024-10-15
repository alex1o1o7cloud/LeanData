import Mathlib

namespace NUMINAMATH_GPT_distance_to_lake_l2373_237370

theorem distance_to_lake 
  {d : ℝ} 
  (h1 : ¬ (d ≥ 8))
  (h2 : ¬ (d ≤ 7))
  (h3 : ¬ (d ≤ 6)) : 
  (7 < d) ∧ (d < 8) :=
by
  sorry

end NUMINAMATH_GPT_distance_to_lake_l2373_237370


namespace NUMINAMATH_GPT_largest_three_digit_sum_l2373_237366

open Nat

def isDigit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9

def areDistinct (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem largest_three_digit_sum : 
  ∀ (X Y Z : ℕ), isDigit X → isDigit Y → isDigit Z → areDistinct X Y Z →
  100 ≤  (110 * X + 11 * Y + 2 * Z) → (110 * X + 11 * Y + 2 * Z) ≤ 999 → 
  110 * X + 11 * Y + 2 * Z ≤ 982 :=
by
  intros
  sorry

end NUMINAMATH_GPT_largest_three_digit_sum_l2373_237366


namespace NUMINAMATH_GPT_intersection_of_sets_l2373_237372

def A := { x : ℝ | 0 ≤ x ∧ x ≤ 2 }
def B := { x : ℝ | x^2 > 1 }
def C := { x : ℝ | 1 < x ∧ x ≤ 2 }

theorem intersection_of_sets : 
  (A ∩ B) = C := 
by sorry

end NUMINAMATH_GPT_intersection_of_sets_l2373_237372


namespace NUMINAMATH_GPT_find_a10_l2373_237392

noncomputable def a_n (a1 d : ℝ) (n : ℕ) : ℝ := a1 + (n - 1) * d

noncomputable def S_n (a1 d : ℝ) (n : ℕ) : ℝ := n * a1 + (n * (n - 1) / 2) * d

theorem find_a10 (a1 d : ℝ)
  (h1 : a_n a1 d 2 + a_n a1 d 4 = 2)
  (h2 : S_n a1 d 2 + S_n a1 d 4 = 1) :
  a_n a1 d 10 = 8 :=
sorry

end NUMINAMATH_GPT_find_a10_l2373_237392


namespace NUMINAMATH_GPT_box_white_balls_count_l2373_237310

/--
A box has exactly 100 balls, and each ball is either red, blue, or white.
Given that the box has 12 more blue balls than white balls,
and twice as many red balls as blue balls,
prove that the number of white balls is 16.
-/
theorem box_white_balls_count (W B R : ℕ) 
  (h1 : B = W + 12) 
  (h2 : R = 2 * B) 
  (h3 : W + B + R = 100) : 
  W = 16 := 
sorry

end NUMINAMATH_GPT_box_white_balls_count_l2373_237310


namespace NUMINAMATH_GPT_compare_a_b_c_compare_explicitly_defined_a_b_c_l2373_237331

theorem compare_a_b_c (a b c : ℕ) (ha : a = 81^31) (hb : b = 27^41) (hc : c = 9^61) : a > b ∧ b > c := 
by
  sorry

-- Noncomputable definitions if necessary
noncomputable def a := 81^31
noncomputable def b := 27^41
noncomputable def c := 9^61

theorem compare_explicitly_defined_a_b_c : a > b ∧ b > c := 
by
  sorry

end NUMINAMATH_GPT_compare_a_b_c_compare_explicitly_defined_a_b_c_l2373_237331


namespace NUMINAMATH_GPT_initial_welders_count_l2373_237348

theorem initial_welders_count
  (W : ℕ)
  (complete_in_5_days : W * 5 = 1)
  (leave_after_1_day : 12 ≤ W) 
  (remaining_complete_in_6_days : (W - 12) * 6 = 1) : 
  W = 72 :=
by
  -- proof steps here
  sorry

end NUMINAMATH_GPT_initial_welders_count_l2373_237348


namespace NUMINAMATH_GPT_annual_interest_rate_l2373_237320

noncomputable def compound_interest 
  (P : ℝ) (A : ℝ) (n : ℕ) (t : ℝ) (r : ℝ) : Prop :=
  A = P * (1 + r / n)^(n * t)

theorem annual_interest_rate 
  (P := 140) (A := 169.40) (n := 2) (t := 1) :
  ∃ r : ℝ, compound_interest P A n t r ∧ r = 0.2 :=
sorry

end NUMINAMATH_GPT_annual_interest_rate_l2373_237320


namespace NUMINAMATH_GPT_compute_expression_l2373_237360

theorem compute_expression : 3 * 3^4 - 27^63 / 27^61 = -486 := by
  sorry

end NUMINAMATH_GPT_compute_expression_l2373_237360


namespace NUMINAMATH_GPT_bee_fraction_remaining_l2373_237335

theorem bee_fraction_remaining (N : ℕ) (L : ℕ) (D : ℕ) (hN : N = 80000) (hL : L = 1200) (hD : D = 50) :
  (N - (L * D)) / N = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_bee_fraction_remaining_l2373_237335


namespace NUMINAMATH_GPT_molecular_weight_n2o_l2373_237365

theorem molecular_weight_n2o (w : ℕ) (n : ℕ) (h : w = 352 ∧ n = 8) : (w / n = 44) :=
sorry

end NUMINAMATH_GPT_molecular_weight_n2o_l2373_237365


namespace NUMINAMATH_GPT_intersection_value_l2373_237349

theorem intersection_value (x0 : ℝ) (h1 : -x0 = Real.tan x0) (h2 : x0 ≠ 0) :
  (x0^2 + 1) * (1 + Real.cos (2 * x0)) = 2 := 
  sorry

end NUMINAMATH_GPT_intersection_value_l2373_237349


namespace NUMINAMATH_GPT_solve_for_other_diagonal_l2373_237356

noncomputable def length_of_other_diagonal
  (area : ℝ) (d2 : ℝ) : ℝ :=
  (2 * area) / d2

theorem solve_for_other_diagonal 
  (h_area : ℝ) (h_d2 : ℝ) (h_condition : h_area = 75 ∧ h_d2 = 15) :
  length_of_other_diagonal h_area h_d2 = 10 :=
by
  -- using h_condition, prove the required theorem
  sorry

end NUMINAMATH_GPT_solve_for_other_diagonal_l2373_237356


namespace NUMINAMATH_GPT_gcd_65536_49152_l2373_237385

theorem gcd_65536_49152 : Nat.gcd 65536 49152 = 16384 :=
by
  sorry

end NUMINAMATH_GPT_gcd_65536_49152_l2373_237385


namespace NUMINAMATH_GPT_inequality_and_equality_condition_l2373_237339

variable {x y : ℝ}

theorem inequality_and_equality_condition
  (hx : 0 < x) (hy : 0 < y) :
  (x + y^2 / x ≥ 2 * y) ∧ (x + y^2 / x = 2 * y ↔ x = y) := sorry

end NUMINAMATH_GPT_inequality_and_equality_condition_l2373_237339


namespace NUMINAMATH_GPT_fitted_bowling_ball_volume_correct_l2373_237307

noncomputable def volume_of_fitted_bowling_ball : ℝ :=
  let ball_radius := 12
  let ball_volume := (4/3) * Real.pi * ball_radius^3
  let hole1_radius := 1
  let hole1_volume := Real.pi * hole1_radius^2 * 6
  let hole2_radius := 1.25
  let hole2_volume := Real.pi * hole2_radius^2 * 6
  let hole3_radius := 2
  let hole3_volume := Real.pi * hole3_radius^2 * 6
  ball_volume - (hole1_volume + hole2_volume + hole3_volume)

theorem fitted_bowling_ball_volume_correct :
  volume_of_fitted_bowling_ball = 2264.625 * Real.pi := by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_fitted_bowling_ball_volume_correct_l2373_237307


namespace NUMINAMATH_GPT_one_fourth_of_8_point_4_is_21_over_10_l2373_237351

theorem one_fourth_of_8_point_4_is_21_over_10 : (8.4 / 4 : ℚ) = 21 / 10 := 
by
  sorry

end NUMINAMATH_GPT_one_fourth_of_8_point_4_is_21_over_10_l2373_237351


namespace NUMINAMATH_GPT_parabola_point_distance_l2373_237390

theorem parabola_point_distance (x y : ℝ) (h : y^2 = 2 * x) (d : ℝ) (focus_x : ℝ) (focus_y : ℝ) :
    focus_x = 1/2 → focus_y = 0 → d = 3 →
    (x + 1/2 = d) → x = 5/2 :=
by
  intros h_focus_x h_focus_y h_d h_dist
  sorry

end NUMINAMATH_GPT_parabola_point_distance_l2373_237390


namespace NUMINAMATH_GPT_age_ratio_4_years_hence_4_years_ago_l2373_237341

-- Definitions based on the conditions
def current_age_ratio (A B : ℕ) := 5 * B = 3 * A
def age_ratio_4_years_ago_4_years_hence (A B : ℕ) := A - 4 = B + 4

-- The main theorem to prove
theorem age_ratio_4_years_hence_4_years_ago (A B : ℕ) 
  (h1 : current_age_ratio A B) 
  (h2 : age_ratio_4_years_ago_4_years_hence A B) : 
  A + 4 = 3 * (B - 4) := 
sorry

end NUMINAMATH_GPT_age_ratio_4_years_hence_4_years_ago_l2373_237341


namespace NUMINAMATH_GPT_sales_fifth_month_l2373_237361

theorem sales_fifth_month
  (a1 a2 a3 a4 a6 : ℕ)
  (h1 : a1 = 2435)
  (h2 : a2 = 2920)
  (h3 : a3 = 2855)
  (h4 : a4 = 3230)
  (h6 : a6 = 1000)
  (avg : ℕ)
  (h_avg : avg = 2500) :
  a1 + a2 + a3 + a4 + (15000 - 1000 - (a1 + a2 + a3 + a4)) + a6 = avg * 6 :=
by
  sorry

end NUMINAMATH_GPT_sales_fifth_month_l2373_237361


namespace NUMINAMATH_GPT_number_of_intersections_l2373_237322

   -- Definitions corresponding to conditions
   def C1 (x y : ℝ) : Prop := x^2 - y^2 + 4*y - 3 = 0
   def C2 (a x y : ℝ) : Prop := y = a*x^2
   def positive_real (a : ℝ) : Prop := a > 0

   -- Final statement converting the question, conditions, and correct answer into Lean code
   theorem number_of_intersections (a : ℝ) (ha : positive_real a) :
     ∃ (count : ℕ), (count = 4) ∧
     (∀ x y : ℝ, C1 x y → C2 a x y → True) := sorry
   
end NUMINAMATH_GPT_number_of_intersections_l2373_237322


namespace NUMINAMATH_GPT_convert_to_scientific_notation_l2373_237337

def original_value : ℝ := 3462.23
def scientific_notation_value : ℝ := 3.46223 * 10^3

theorem convert_to_scientific_notation : 
  original_value = scientific_notation_value :=
sorry

end NUMINAMATH_GPT_convert_to_scientific_notation_l2373_237337


namespace NUMINAMATH_GPT_age_of_eldest_boy_l2373_237380

theorem age_of_eldest_boy (x : ℕ) (h1 : (3*x + 5*x + 7*x) / 3 = 15) :
  7 * x = 21 :=
sorry

end NUMINAMATH_GPT_age_of_eldest_boy_l2373_237380


namespace NUMINAMATH_GPT_sum_of_integers_with_product_2720_l2373_237368

theorem sum_of_integers_with_product_2720 (n : ℤ) (h1 : n > 0) (h2 : n * (n + 2) = 2720) : n + (n + 2) = 104 :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_integers_with_product_2720_l2373_237368


namespace NUMINAMATH_GPT_policeman_catches_thief_l2373_237316

/-
  From a police station situated on a straight road infinite in both directions, a thief has stolen a police car.
  Its maximal speed equals 90% of the maximal speed of a police cruiser. When the theft is discovered some time
  later, a policeman starts to pursue the thief on a cruiser. However, the policeman does not know in which direction
  along the road the thief has gone, nor does he know how long ago the car has been stolen. The goal is to prove
  that it is possible for the policeman to catch the thief.
-/
theorem policeman_catches_thief (v : ℝ) (T₀ : ℝ) (o₀ : ℝ) :
  (0 < v) →
  (0 < T₀) →
  ∃ T p, T₀ ≤ T ∧ p ≤ v * T :=
sorry

end NUMINAMATH_GPT_policeman_catches_thief_l2373_237316


namespace NUMINAMATH_GPT_downstream_distance_l2373_237343

-- Define the speeds and distances as constants or variables
def speed_boat := 30 -- speed in kmph
def speed_stream := 10 -- speed in kmph
def distance_upstream := 40 -- distance in km
def time_upstream := distance_upstream / (speed_boat - speed_stream) -- time in hours

-- Define the variable for the downstream distance
variable {D : ℝ}

-- The Lean 4 statement to prove that the downstream distance is the specified value
theorem downstream_distance : 
  (time_upstream = D / (speed_boat + speed_stream)) → D = 80 :=
by
  sorry

end NUMINAMATH_GPT_downstream_distance_l2373_237343


namespace NUMINAMATH_GPT_hexagon_vertices_zero_l2373_237378

theorem hexagon_vertices_zero (n : ℕ) (a0 a1 a2 a3 a4 a5 : ℕ) 
  (h_sum : a0 + a1 + a2 + a3 + a4 + a5 = n) 
  (h_pos : 0 < n) :
  (n = 2 ∨ n % 2 = 1) → 
  ∃ (b0 b1 b2 b3 b4 b5 : ℕ), b0 = 0 ∧ b1 = 0 ∧ b2 = 0 ∧ b3 = 0 ∧ b4 = 0 ∧ b5 = 0 := sorry

end NUMINAMATH_GPT_hexagon_vertices_zero_l2373_237378


namespace NUMINAMATH_GPT_num_pos_int_solutions_2a_plus_3b_eq_15_l2373_237338

theorem num_pos_int_solutions_2a_plus_3b_eq_15 : 
  (∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ 2 * a + 3 * b = 15) ∧ 
  (∀ (a1 a2 b1 b2 : ℕ), 0 < a1 ∧ 0 < a2 ∧ 0 < b1 ∧ 0 < b2 ∧ 
  (2 * a1 + 3 * b1 = 15) ∧ (2 * a2 + 3 * b2 = 15) → 
  ((a1 = 3 ∧ b1 = 3 ∨ a1 = 6 ∧ b1 = 1) ∧ (a2 = 3 ∧ b2 = 3 ∨ a2 = 6 ∧ b2 = 1))) := 
  sorry

end NUMINAMATH_GPT_num_pos_int_solutions_2a_plus_3b_eq_15_l2373_237338


namespace NUMINAMATH_GPT_percent_profit_l2373_237327

theorem percent_profit (CP LP SP Profit : ℝ) 
  (hCP : CP = 100) 
  (hLP : LP = CP + 0.30 * CP)
  (hSP : SP = LP - 0.10 * LP) 
  (hProfit : Profit = SP - CP) : 
  (Profit / CP) * 100 = 17 :=
by
  sorry

end NUMINAMATH_GPT_percent_profit_l2373_237327


namespace NUMINAMATH_GPT_determine_a_range_l2373_237355

open Real

theorem determine_a_range (a : ℝ) :
  (∃ x : ℝ, x^2 + 2*x + a ≤ 0) → a ≤ 1 :=
sorry

end NUMINAMATH_GPT_determine_a_range_l2373_237355


namespace NUMINAMATH_GPT_ratio_of_discretionary_income_l2373_237393

theorem ratio_of_discretionary_income
  (net_monthly_salary : ℝ) 
  (vacation_fund_pct : ℝ) 
  (savings_pct : ℝ) 
  (socializing_pct : ℝ) 
  (gifts_amt : ℝ)
  (D : ℝ) 
  (ratio : ℝ)
  (salary : net_monthly_salary = 3700)
  (vacation_fund : vacation_fund_pct = 0.30)
  (savings : savings_pct = 0.20)
  (socializing : socializing_pct = 0.35)
  (gifts : gifts_amt = 111)
  (discretionary_income : D = gifts_amt / 0.15)
  (net_salary_ratio : ratio = D / net_monthly_salary) :
  ratio = 1 / 5 := sorry

end NUMINAMATH_GPT_ratio_of_discretionary_income_l2373_237393


namespace NUMINAMATH_GPT_translated_line_expression_l2373_237336

theorem translated_line_expression (x y : ℝ) (b : ℝ) :
  (∀ x y, y = 2 * x + 3 ∧ (5, 1).2 = 2 * (5, 1).1 + b) → y = 2 * x - 9 :=
by
  sorry

end NUMINAMATH_GPT_translated_line_expression_l2373_237336


namespace NUMINAMATH_GPT_simplify_fraction_product_l2373_237321

theorem simplify_fraction_product : 
  (270 / 24) * (7 / 210) * (6 / 4) = 4.5 :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_product_l2373_237321


namespace NUMINAMATH_GPT_red_window_exchange_l2373_237362

-- Defining the total transaction amount for online and offline booths
variables (x y : ℝ)

-- Defining conditions
def offlineMoreThanOnline (y x : ℝ) : Prop := y - 7 * x = 1.8
def averageTransactionDifference (y x : ℝ) : Prop := (y / 71) - (x / 44) = 0.3

-- The proof problem
theorem red_window_exchange (x y : ℝ) :
  offlineMoreThanOnline y x ∧ averageTransactionDifference y x := 
sorry

end NUMINAMATH_GPT_red_window_exchange_l2373_237362


namespace NUMINAMATH_GPT_each_friend_paid_l2373_237352

def cottage_cost_per_hour : ℕ := 5
def rental_duration_hours : ℕ := 8
def total_cost := cottage_cost_per_hour * rental_duration_hours
def cost_per_person := total_cost / 2

theorem each_friend_paid : cost_per_person = 20 :=
by 
  sorry

end NUMINAMATH_GPT_each_friend_paid_l2373_237352


namespace NUMINAMATH_GPT_min_value_seq_l2373_237373

theorem min_value_seq (a : ℕ → ℕ) (n : ℕ) (h₁ : a 1 = 26) (h₂ : ∀ n, a (n + 1) - a n = 2 * n + 1) :
  ∃ m, (m > 0) ∧ (∀ k, k > 0 → (a k / k : ℚ) ≥ 10) ∧ (a m / m : ℚ) = 10 :=
by
  sorry

end NUMINAMATH_GPT_min_value_seq_l2373_237373


namespace NUMINAMATH_GPT_fraction_1790s_l2373_237346

def total_states : ℕ := 30
def states_1790s : ℕ := 16

theorem fraction_1790s : (states_1790s / total_states : ℚ) = 8 / 15 :=
by
  -- We claim that the fraction of states admitted during the 1790s is exactly 8/15
  sorry

end NUMINAMATH_GPT_fraction_1790s_l2373_237346


namespace NUMINAMATH_GPT_angle_in_third_quadrant_l2373_237379

theorem angle_in_third_quadrant (θ : ℤ) (hθ : θ = -510) : 
  (210 % 360 > 180 ∧ 210 % 360 < 270) := 
by
  have h : 210 % 360 = 210 := by norm_num
  sorry

end NUMINAMATH_GPT_angle_in_third_quadrant_l2373_237379


namespace NUMINAMATH_GPT_total_canoes_built_l2373_237387

-- Defining basic variables and functions for the proof
variable (a : Nat := 5) -- Initial number of canoes in January
variable (r : Nat := 3) -- Common ratio
variable (n : Nat := 6) -- Number of months including January

-- Function to compute sum of the first n terms of a geometric series
def geometric_sum (a r n : Nat) : Nat :=
  a * (r^n - 1) / (r - 1)

-- The proposition we want to prove
theorem total_canoes_built : geometric_sum a r n = 1820 := by
  sorry

end NUMINAMATH_GPT_total_canoes_built_l2373_237387


namespace NUMINAMATH_GPT_find_cost_price_l2373_237303

variable (C : ℝ)

theorem find_cost_price (h : 56 - C = C - 42) : C = 49 :=
by
  sorry

end NUMINAMATH_GPT_find_cost_price_l2373_237303


namespace NUMINAMATH_GPT_xy_sq_is_37_over_36_l2373_237309

theorem xy_sq_is_37_over_36 (x y : ℚ) (h : 2002 * (x - 1)^2 + |x - 12 * y + 1| = 0) : x^2 + y^2 = 37 / 36 :=
sorry

end NUMINAMATH_GPT_xy_sq_is_37_over_36_l2373_237309


namespace NUMINAMATH_GPT_sum_adjacent_odd_l2373_237386

/-
  Given 2020 natural numbers written in a circle, prove that the sum of any two adjacent numbers is odd.
-/

noncomputable def numbers_in_circle : Fin 2020 → ℕ := sorry

theorem sum_adjacent_odd (k : Fin 2020) :
  (numbers_in_circle k + numbers_in_circle (k + 1)) % 2 = 1 :=
sorry

end NUMINAMATH_GPT_sum_adjacent_odd_l2373_237386


namespace NUMINAMATH_GPT_axis_of_symmetry_shifted_sine_function_l2373_237399

open Real

noncomputable def axisOfSymmetry (k : ℤ) : ℝ := k * π / 2 + π / 6

theorem axis_of_symmetry_shifted_sine_function (x : ℝ) (k : ℤ) :
  ∃ k : ℤ, x = axisOfSymmetry k := by
sorry

end NUMINAMATH_GPT_axis_of_symmetry_shifted_sine_function_l2373_237399


namespace NUMINAMATH_GPT_apple_price_l2373_237374

variable (p q : ℝ)

theorem apple_price :
  (30 * p + 3 * q = 168) →
  (30 * p + 6 * q = 186) →
  (20 * p = 100) →
  p = 5 :=
by
  intros h1 h2 h3
  have h4 : p = 5 := sorry
  exact h4

end NUMINAMATH_GPT_apple_price_l2373_237374


namespace NUMINAMATH_GPT_parallel_lines_implies_m_opposite_sides_implies_m_range_l2373_237311

-- Definitions of the given lines and points
def l1 (x y : ℝ) : Prop := 2 * x + y - 1 = 0
def A (m : ℝ) : ℝ × ℝ := (-2, m)
def B (m : ℝ) : ℝ × ℝ := (m, 4)

-- Problem Part (I)
theorem parallel_lines_implies_m (m : ℝ) : 
  (∀ (x y : ℝ), l1 x y → false) ∧ (∀ (x2 y2 : ℝ), (x2, y2) = A m ∨ (x2, y2) = B m → false) →
  (∃ m, 2 * m + 3 = 0 ∧ m + 5 = 0) :=
sorry

-- Problem Part (II)
theorem opposite_sides_implies_m_range (m : ℝ) :
  ((2 * (-2) + m - 1) * (2 * m + 4 - 1) < 0) →
  m ∈ Set.Ioo (-3/2 : ℝ) (5 : ℝ) :=
sorry

end NUMINAMATH_GPT_parallel_lines_implies_m_opposite_sides_implies_m_range_l2373_237311


namespace NUMINAMATH_GPT_problem_inequality_l2373_237325

theorem problem_inequality (x y z : ℝ) (h1 : x + y + z = 0) (h2 : |x| + |y| + |z| ≤ 1) :
  x + (y / 2) + (z / 3) ≤ 1 / 3 :=
sorry

end NUMINAMATH_GPT_problem_inequality_l2373_237325


namespace NUMINAMATH_GPT_probability_at_least_one_die_shows_three_l2373_237397

theorem probability_at_least_one_die_shows_three : 
  let outcomes := 36
  let not_three_outcomes := 25
  (outcomes - not_three_outcomes) / outcomes = 11 / 36 := sorry

end NUMINAMATH_GPT_probability_at_least_one_die_shows_three_l2373_237397


namespace NUMINAMATH_GPT_probability_right_triangle_in_3x3_grid_l2373_237328

theorem probability_right_triangle_in_3x3_grid : 
  let vertices := (3 + 1) * (3 + 1)
  let total_combinations := Nat.choose vertices 3
  let right_triangles_on_gridlines := 144
  let right_triangles_off_gridlines := 24 + 32
  let total_right_triangles := right_triangles_on_gridlines + right_triangles_off_gridlines
  (total_right_triangles : ℚ) / total_combinations = 5 / 14 :=
by 
  sorry

end NUMINAMATH_GPT_probability_right_triangle_in_3x3_grid_l2373_237328


namespace NUMINAMATH_GPT_hexagon_longest_side_l2373_237350

theorem hexagon_longest_side (x : ℝ) (h₁ : 6 * x = 20) (h₂ : x < 20 - x) : (10 / 3) ≤ x ∧ x < 10 :=
sorry

end NUMINAMATH_GPT_hexagon_longest_side_l2373_237350


namespace NUMINAMATH_GPT_copper_alloy_proof_l2373_237357

variable (x p : ℝ)

theorem copper_alloy_proof
  (copper_content1 copper_content2 weight1 weight2 total_weight : ℝ)
  (h1 : weight1 = 3)
  (h2 : copper_content1 = 0.4)
  (h3 : weight2 = 7)
  (h4 : copper_content2 = 0.3)
  (h5 : total_weight = 8)
  (h6 : 1 ≤ x ∧ x ≤ 3)
  (h7 : p = 100 * (copper_content1 * x + copper_content2 * (total_weight - x)) / total_weight) :
  31.25 ≤ p ∧ p ≤ 33.75 := 
  sorry

end NUMINAMATH_GPT_copper_alloy_proof_l2373_237357


namespace NUMINAMATH_GPT_length_of_garden_l2373_237345

theorem length_of_garden (P B : ℕ) (hP : P = 1800) (hB : B = 400) : 
  ∃ L : ℕ, L = 500 ∧ P = 2 * (L + B) :=
by
  sorry

end NUMINAMATH_GPT_length_of_garden_l2373_237345


namespace NUMINAMATH_GPT_pyramid_base_side_length_l2373_237330

theorem pyramid_base_side_length
  (area : ℝ)
  (slant_height : ℝ)
  (h : area = 90)
  (sh : slant_height = 15) :
  ∃ (s : ℝ), 90 = 1 / 2 * s * 15 ∧ s = 12 :=
by
  sorry

end NUMINAMATH_GPT_pyramid_base_side_length_l2373_237330


namespace NUMINAMATH_GPT_distance_between_first_and_last_tree_l2373_237369

theorem distance_between_first_and_last_tree (n : ℕ) (d : ℕ) 
  (h₁ : n = 8)
  (h₂ : d = 75)
  : (d / ((4 - 1) : ℕ)) * (n - 1) = 175 := sorry

end NUMINAMATH_GPT_distance_between_first_and_last_tree_l2373_237369


namespace NUMINAMATH_GPT_Hillary_left_with_amount_l2373_237367

theorem Hillary_left_with_amount :
  let price_per_craft := 12
  let crafts_sold := 3
  let extra_earnings := 7
  let deposit_amount := 18
  let total_earnings := crafts_sold * price_per_craft + extra_earnings
  let remaining_amount := total_earnings - deposit_amount
  remaining_amount = 25 :=
by
  let price_per_craft := 12
  let crafts_sold := 3
  let extra_earnings := 7
  let deposit_amount := 18
  let total_earnings := crafts_sold * price_per_craft + extra_earnings
  let remaining_amount := total_earnings - deposit_amount
  sorry

end NUMINAMATH_GPT_Hillary_left_with_amount_l2373_237367


namespace NUMINAMATH_GPT_temperature_drop_l2373_237344

-- Define the initial temperature and the drop in temperature
def initial_temperature : ℤ := -6
def drop : ℤ := 5

-- Define the resulting temperature after the drop
def resulting_temperature : ℤ := initial_temperature - drop

-- The theorem to be proved
theorem temperature_drop : resulting_temperature = -11 :=
by
  sorry

end NUMINAMATH_GPT_temperature_drop_l2373_237344


namespace NUMINAMATH_GPT_man_late_minutes_l2373_237377

theorem man_late_minutes (v t t' : ℝ) (hv : v' = 3 / 4 * v) (ht : t = 2) (ht' : t' = 4 / 3 * t) :
  t' * 60 - t * 60 = 40 :=
by
  sorry

end NUMINAMATH_GPT_man_late_minutes_l2373_237377


namespace NUMINAMATH_GPT_tens_digit_23_pow_1987_l2373_237300

def tens_digit_of_power (a b n : ℕ) : ℕ :=
  ((a^b % n) / 10) % 10

theorem tens_digit_23_pow_1987 : tens_digit_of_power 23 1987 100 = 4 := by
  sorry

end NUMINAMATH_GPT_tens_digit_23_pow_1987_l2373_237300


namespace NUMINAMATH_GPT_divisibility_3804_l2373_237312

theorem divisibility_3804 (n : ℕ) (h : 0 < n) :
    3804 ∣ ((n ^ 3 - n) * (5 ^ (8 * n + 4) + 3 ^ (4 * n + 2))) :=
sorry

end NUMINAMATH_GPT_divisibility_3804_l2373_237312


namespace NUMINAMATH_GPT_meiosis_fertilization_correct_l2373_237398

theorem meiosis_fertilization_correct :
  (∀ (half_nuclear_sperm half_nuclear_egg mitochondrial_egg : Prop)
     (recognition_basis_clycoproteins : Prop)
     (fusion_basis_nuclei : Prop)
     (meiosis_eukaryotes : Prop)
     (random_fertilization : Prop),
    (half_nuclear_sperm ∧ half_nuclear_egg ∧ mitochondrial_egg ∧ recognition_basis_clycoproteins ∧ fusion_basis_nuclei ∧ meiosis_eukaryotes ∧ random_fertilization) →
    (D : Prop) ) := 
sorry

end NUMINAMATH_GPT_meiosis_fertilization_correct_l2373_237398


namespace NUMINAMATH_GPT_find_m_pure_imaginary_l2373_237319

noncomputable def find_m (m : ℝ) : ℝ := m

theorem find_m_pure_imaginary (m : ℝ) (h : (m^2 - 5 * m + 6 : ℂ) = 0) :
  find_m m = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_m_pure_imaginary_l2373_237319


namespace NUMINAMATH_GPT_football_game_spectators_l2373_237314

theorem football_game_spectators (total_wristbands wristbands_per_person : ℕ) (h1 : total_wristbands = 234) (h2 : wristbands_per_person = 2) :
  total_wristbands / wristbands_per_person = 117 := by
  sorry

end NUMINAMATH_GPT_football_game_spectators_l2373_237314


namespace NUMINAMATH_GPT_ceil_minus_floor_eq_one_implies_ceil_minus_y_l2373_237358

noncomputable def fractional_part (y : ℝ) : ℝ := y - ⌊y⌋

theorem ceil_minus_floor_eq_one_implies_ceil_minus_y (y : ℝ) (h : ⌈y⌉ - ⌊y⌋ = 1) : ⌈y⌉ - y = 1 - fractional_part y :=
by
  sorry

end NUMINAMATH_GPT_ceil_minus_floor_eq_one_implies_ceil_minus_y_l2373_237358


namespace NUMINAMATH_GPT_siamese_cats_initial_l2373_237306

theorem siamese_cats_initial (S : ℕ) : S + 25 - 45 = 18 -> S = 38 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_siamese_cats_initial_l2373_237306


namespace NUMINAMATH_GPT_remainder_of_91_pow_92_mod_100_l2373_237396

theorem remainder_of_91_pow_92_mod_100 : (91 ^ 92) % 100 = 81 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_91_pow_92_mod_100_l2373_237396


namespace NUMINAMATH_GPT_value_of_a_l2373_237376

theorem value_of_a (a b : ℝ) (h1 : b = 4 * a) (h2 : b = 20 - 7 * a) : a = 20 / 11 := by
  sorry

end NUMINAMATH_GPT_value_of_a_l2373_237376


namespace NUMINAMATH_GPT_factorize_quadratic_l2373_237354

theorem factorize_quadratic (x : ℝ) : 2 * x^2 + 12 * x + 18 = 2 * (x + 3)^2 :=
by
  sorry

end NUMINAMATH_GPT_factorize_quadratic_l2373_237354


namespace NUMINAMATH_GPT_tan_five_pi_over_four_l2373_237359

theorem tan_five_pi_over_four : Real.tan (5 * Real.pi / 4) = 1 :=
  by
  sorry

end NUMINAMATH_GPT_tan_five_pi_over_four_l2373_237359


namespace NUMINAMATH_GPT_percentage_of_students_passed_l2373_237329

theorem percentage_of_students_passed
  (students_failed : ℕ)
  (total_students : ℕ)
  (H_failed : students_failed = 260)
  (H_total : total_students = 400)
  (passed := total_students - students_failed) :
  (passed * 100 / total_students : ℝ) = 35 := 
by
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_percentage_of_students_passed_l2373_237329


namespace NUMINAMATH_GPT_train_stops_time_l2373_237363

theorem train_stops_time 
  (speed_excluding_stoppages : ℝ)
  (speed_including_stoppages : ℝ)
  (h1 : speed_excluding_stoppages = 60)
  (h2 : speed_including_stoppages = 40) : 
  ∃ (stoppage_time : ℝ), stoppage_time = 20 := 
by
  sorry

end NUMINAMATH_GPT_train_stops_time_l2373_237363


namespace NUMINAMATH_GPT_angle_BDC_is_30_l2373_237318

theorem angle_BDC_is_30 
    (A E C B D : ℝ) 
    (hA : A = 50) 
    (hE : E = 60) 
    (hC : C = 40) : 
    BDC = 30 :=
by
  sorry

end NUMINAMATH_GPT_angle_BDC_is_30_l2373_237318


namespace NUMINAMATH_GPT_polar_coordinate_conversion_l2373_237334

theorem polar_coordinate_conversion :
  ∃ (r θ : ℝ), (r = 2) ∧ (θ = 11 * Real.pi / 8) ∧ 
    ∀ (r1 θ1 : ℝ), (r1 = -2) ∧ (θ1 = 3 * Real.pi / 8) →
      (abs r1 = r) ∧ (θ1 + Real.pi = θ) :=
by
  sorry

end NUMINAMATH_GPT_polar_coordinate_conversion_l2373_237334


namespace NUMINAMATH_GPT_coefficient_x2y6_expansion_l2373_237313

theorem coefficient_x2y6_expansion :
  let x : ℤ := 1
  let y : ℤ := 1
  ∃ a : ℤ, a = -28 ∧ (a • x ^ 2 * y ^ 6) = (1 - y / x) * (x + y) ^ 8 :=
by
  sorry

end NUMINAMATH_GPT_coefficient_x2y6_expansion_l2373_237313


namespace NUMINAMATH_GPT_icing_time_is_30_l2373_237391

def num_batches : Nat := 4
def baking_time_per_batch : Nat := 20
def total_time : Nat := 200

def baking_time_total : Nat := num_batches * baking_time_per_batch
def icing_time_total : Nat := total_time - baking_time_total
def icing_time_per_batch : Nat := icing_time_total / num_batches

theorem icing_time_is_30 :
  icing_time_per_batch = 30 := by
  sorry

end NUMINAMATH_GPT_icing_time_is_30_l2373_237391


namespace NUMINAMATH_GPT_rectangle_perimeter_l2373_237332

-- We first define the side lengths of the squares and their relationships
def b1 : ℕ := 3
def b2 : ℕ := 9
def b3 := b1 + b2
def b4 := 2 * b1 + b2
def b5 := 3 * b1 + 2 * b2
def b6 := 3 * b1 + 3 * b2
def b7 := 4 * b1 + 3 * b2

-- Dimensions of the rectangle
def L := 37
def W := 52

-- Theorem to prove the perimeter of the rectangle
theorem rectangle_perimeter : 2 * L + 2 * W = 178 := by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_l2373_237332


namespace NUMINAMATH_GPT_partial_fraction_identity_l2373_237364

theorem partial_fraction_identity
  (P Q R : ℝ)
  (h1 : -2 = P + Q)
  (h2 : 1 = Q + R)
  (h3 : -1 = P + R) :
  (P, Q, R) = (-2, 0, 1) :=
by
  sorry

end NUMINAMATH_GPT_partial_fraction_identity_l2373_237364


namespace NUMINAMATH_GPT_fractions_arith_l2373_237347

theorem fractions_arith : (3 / 50) + (2 / 25) - (5 / 1000) = 0.135 := by
  sorry

end NUMINAMATH_GPT_fractions_arith_l2373_237347


namespace NUMINAMATH_GPT_tan_y_eq_tan_x_plus_one_over_cos_x_l2373_237395

theorem tan_y_eq_tan_x_plus_one_over_cos_x 
  (x y : ℝ) 
  (hx : 0 < x) 
  (hxy : x < y) 
  (hy : y < π / 2) 
  (h_tan : Real.tan y = Real.tan x + (1 / Real.cos x)) 
  : y - (x / 2) = π / 6 :=
sorry

end NUMINAMATH_GPT_tan_y_eq_tan_x_plus_one_over_cos_x_l2373_237395


namespace NUMINAMATH_GPT_x_squared_plus_y_squared_l2373_237304

theorem x_squared_plus_y_squared (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y)
(h3 : x * y + x + y = 71)
(h4 : x^2 * y + x * y^2 = 880) :
x^2 + y^2 = 146 :=
sorry

end NUMINAMATH_GPT_x_squared_plus_y_squared_l2373_237304


namespace NUMINAMATH_GPT_dihedral_minus_solid_equals_expression_l2373_237308

-- Definitions based on the conditions provided.
noncomputable def sumDihedralAngles (P : Polyhedron) : ℝ := sorry
noncomputable def sumSolidAngles (P : Polyhedron) : ℝ := sorry
def numFaces (P : Polyhedron) : ℕ := sorry

-- Theorem statement we want to prove.
theorem dihedral_minus_solid_equals_expression (P : Polyhedron) :
  sumDihedralAngles P - sumSolidAngles P = 2 * Real.pi * (numFaces P - 2) :=
sorry

end NUMINAMATH_GPT_dihedral_minus_solid_equals_expression_l2373_237308


namespace NUMINAMATH_GPT_circle_center_l2373_237301

theorem circle_center (x y : ℝ) :
  4 * x^2 - 16 * x + 4 * y^2 + 8 * y - 12 = 0 →
  (∃ h k : ℝ, (x - h)^2 + (y - k)^2 = 8 ∧ h = 2 ∧ k = -1) :=
sorry

end NUMINAMATH_GPT_circle_center_l2373_237301


namespace NUMINAMATH_GPT_cornflowers_count_l2373_237324

theorem cornflowers_count
  (n k : ℕ)
  (total_flowers : 9 * n + 17 * k = 70)
  (equal_dandelions_daisies : 5 * n = 7 * k) :
  (9 * n - 20 - 14 = 2) ∧ (17 * k - 20 - 14 = 0) :=
by
  sorry

end NUMINAMATH_GPT_cornflowers_count_l2373_237324


namespace NUMINAMATH_GPT_arctan_sum_of_roots_eq_pi_div_4_l2373_237389

theorem arctan_sum_of_roots_eq_pi_div_4 (x₁ x₂ x₃ : ℝ) 
  (h₁ : Polynomial.eval x₁ (Polynomial.C 11 - Polynomial.C 10 * Polynomial.X + Polynomial.X ^ 3) = 0)
  (h₂ : Polynomial.eval x₂ (Polynomial.C 11 - Polynomial.C 10 * Polynomial.X + Polynomial.X ^ 3) = 0)
  (h₃ : Polynomial.eval x₃ (Polynomial.C 11 - Polynomial.C 10 * Polynomial.X + Polynomial.X ^ 3) = 0)
  (h_intv : -5 < x₁ ∧ x₁ < 5 ∧ -5 < x₂ ∧ x₂ < 5 ∧ -5 < x₃ ∧ x₃ < 5) :
  Real.arctan x₁ + Real.arctan x₂ + Real.arctan x₃ = Real.pi / 4 :=
sorry

end NUMINAMATH_GPT_arctan_sum_of_roots_eq_pi_div_4_l2373_237389


namespace NUMINAMATH_GPT_find_g_zero_l2373_237333

variable {g : ℝ → ℝ}

theorem find_g_zero (h : ∀ x y : ℝ, g (x + y) = g x + g y - 1) : g 0 = 1 :=
sorry

end NUMINAMATH_GPT_find_g_zero_l2373_237333


namespace NUMINAMATH_GPT_calculate_difference_l2373_237305

variable (σ : ℝ) -- Let \square be represented by a real number σ
def correct_answer := 4 * (σ - 3)
def incorrect_answer := 4 * σ - 3
def difference := correct_answer σ - incorrect_answer σ

theorem calculate_difference : difference σ = -9 := by
  sorry

end NUMINAMATH_GPT_calculate_difference_l2373_237305


namespace NUMINAMATH_GPT_simplify_expression_l2373_237315

theorem simplify_expression (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = 2 * x⁻¹ * y⁻¹ * z⁻¹ :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2373_237315


namespace NUMINAMATH_GPT_subtraction_correctness_l2373_237384

theorem subtraction_correctness : 25.705 - 3.289 = 22.416 := 
by
  sorry

end NUMINAMATH_GPT_subtraction_correctness_l2373_237384


namespace NUMINAMATH_GPT_simplify_expression_l2373_237302

noncomputable def sin_30 := 1 / 2
noncomputable def cos_30 := Real.sqrt 3 / 2

theorem simplify_expression :
  (sin_30 ^ 3 + cos_30 ^ 3) / (sin_30 + cos_30) = 1 - Real.sqrt 3 / 4 := sorry

end NUMINAMATH_GPT_simplify_expression_l2373_237302


namespace NUMINAMATH_GPT_paula_paint_cans_l2373_237383

variables (rooms_per_can total_rooms_lost initial_rooms final_rooms cans_lost : ℕ)

theorem paula_paint_cans
  (h1 : initial_rooms = 50)
  (h2 : cans_lost = 2)
  (h3 : final_rooms = 42)
  (h4 : total_rooms_lost = initial_rooms - final_rooms)
  (h5 : rooms_per_can = total_rooms_lost / cans_lost) :
  final_rooms / rooms_per_can = 11 :=
by sorry

end NUMINAMATH_GPT_paula_paint_cans_l2373_237383


namespace NUMINAMATH_GPT_evaluate_expression_l2373_237375

theorem evaluate_expression : 4 * (8 - 3) - 6 / 3 = 18 :=
by sorry

end NUMINAMATH_GPT_evaluate_expression_l2373_237375


namespace NUMINAMATH_GPT_arithmetic_geometric_progression_l2373_237381

theorem arithmetic_geometric_progression (a b c : ℤ) (h1 : a < b) (h2 : b < c)
  (h3 : b = 3 * a) (h4 : 2 * b = a + c) (h5 : b * b = a * c) : c = 9 :=
sorry

end NUMINAMATH_GPT_arithmetic_geometric_progression_l2373_237381


namespace NUMINAMATH_GPT_calculate_polynomial_value_l2373_237340

theorem calculate_polynomial_value (a a1 a2 a3 a4 a5 : ℝ) : 
  (∀ x : ℝ, (1 - x)^2 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5) → 
  (a + a2 + a4) * (a1 + a3 + a5) = -256 := 
by 
  intro h
  sorry

end NUMINAMATH_GPT_calculate_polynomial_value_l2373_237340


namespace NUMINAMATH_GPT_window_design_ratio_l2373_237353

theorem window_design_ratio (AB AD r : ℝ)
  (h1 : AB = 40)
  (h2 : AD / AB = 4 / 3)
  (h3 : r = AB / 2) :
  ((AD - AB) * AB) / (π * r^2 / 2) = 8 / (3 * π) :=
by
  sorry

end NUMINAMATH_GPT_window_design_ratio_l2373_237353


namespace NUMINAMATH_GPT_average_age_of_women_l2373_237342

theorem average_age_of_women (A : ℝ) (W1 W2 : ℝ)
  (cond1 : 10 * (A + 6) - 10 * A = 60)
  (cond2 : W1 + W2 = 60 + 40) :
  (W1 + W2) / 2 = 50 := 
by
  sorry

end NUMINAMATH_GPT_average_age_of_women_l2373_237342


namespace NUMINAMATH_GPT_distinct_dress_designs_l2373_237317

theorem distinct_dress_designs : 
  let num_colors := 5
  let num_patterns := 6
  num_colors * num_patterns = 30 :=
by
  sorry

end NUMINAMATH_GPT_distinct_dress_designs_l2373_237317


namespace NUMINAMATH_GPT_maximum_value_of_func_l2373_237394

noncomputable def func (x : ℝ) : ℝ := 4 * x - 2 + 1 / (4 * x - 5)

theorem maximum_value_of_func (x : ℝ) (h : x < 5 / 4) : ∃ y, y = 1 ∧ ∀ z, z = func x → z ≤ y :=
sorry

end NUMINAMATH_GPT_maximum_value_of_func_l2373_237394


namespace NUMINAMATH_GPT_cylinder_volume_relation_l2373_237371

theorem cylinder_volume_relation (r h : ℝ) (π_pos : 0 < π) :
  (∀ B_h B_r A_h A_r : ℝ, B_h = r ∧ B_r = h ∧ A_h = h ∧ A_r = r 
   → 3 * (π * h^2 * r) = π * r^2 * h) → 
  ∃ N : ℝ, (π * (3 * h)^2 * h) = N * π * h^3 ∧ N = 9 :=
by 
  sorry

end NUMINAMATH_GPT_cylinder_volume_relation_l2373_237371


namespace NUMINAMATH_GPT_problem_conditions_l2373_237382

noncomputable def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x < 1 then x / (1 + x)
else if -1 < x ∧ x < 0 then x / (1 - x)
else 0

theorem problem_conditions (a b : ℝ) (x : ℝ) :
  (∀ x : ℝ, -1 < x → x < 1 → f (-x) = -f x) ∧ 
  (∀ x : ℝ, 0 ≤ x → x < 1 → f x = (-a * x - b) / (1 + x)) ∧ 
  (f (1 / 2) = 1 / 3) →
  (a = -1) ∧ (b = 0) ∧
  (∀ x :  ℝ, -1 < x ∧ x < 1 → 
    (if 0 ≤ x ∧ x < 1 then f x = x / (1 + x) else if -1 < x ∧ x < 0 then f x = x / (1 - x) else True)) ∧ 
  (∀ x1 x2 : ℝ, -1 < x1 → x1 < x2 → x2 < 1 → f x1 < f x2) ∧ 
  (∀ x : ℝ, f (x - 1) + f x > 0 → (1 / 2 < x ∧ x < 1)) :=
by
  sorry

end NUMINAMATH_GPT_problem_conditions_l2373_237382


namespace NUMINAMATH_GPT_total_arrangements_l2373_237326

theorem total_arrangements (students communities : ℕ) 
  (h_students : students = 5) 
  (h_communities : communities = 3)
  (h_conditions :
    ∀(student : Fin students) (community : Fin communities), 
      true 
  ) : 150 = 150 :=
by sorry

end NUMINAMATH_GPT_total_arrangements_l2373_237326


namespace NUMINAMATH_GPT_strands_of_duct_tape_used_l2373_237388

-- Define the conditions
def hannah_cut_rate : ℕ := 8  -- Hannah's cutting rate
def son_cut_rate : ℕ := 3     -- Son's cutting rate
def minutes : ℕ := 2          -- Time taken to free the younger son

-- Define the total cutting rate
def total_cut_rate : ℕ := hannah_cut_rate + son_cut_rate

-- Define the total number of strands
def total_strands : ℕ := total_cut_rate * minutes

-- State the theorem to prove
theorem strands_of_duct_tape_used : total_strands = 22 :=
by
  sorry

end NUMINAMATH_GPT_strands_of_duct_tape_used_l2373_237388


namespace NUMINAMATH_GPT_daughters_meet_days_count_l2373_237323

noncomputable def days_elder_returns := 5
noncomputable def days_second_returns := 4
noncomputable def days_youngest_returns := 3

noncomputable def total_days := 100

-- Defining the count of individual and combined visits
noncomputable def count_individual_visits (period : ℕ) : ℕ := total_days / period
noncomputable def count_combined_visits (period1 : ℕ) (period2 : ℕ) : ℕ := total_days / Nat.lcm period1 period2
noncomputable def count_all_together_visits (periods : List ℕ) : ℕ := total_days / periods.foldr Nat.lcm 1

-- Specific counts
noncomputable def count_youngest_visits : ℕ := count_individual_visits days_youngest_returns
noncomputable def count_second_visits : ℕ := count_individual_visits days_second_returns
noncomputable def count_elder_visits : ℕ := count_individual_visits days_elder_returns

noncomputable def count_youngest_and_second : ℕ := count_combined_visits days_youngest_returns days_second_returns
noncomputable def count_youngest_and_elder : ℕ := count_combined_visits days_youngest_returns days_elder_returns
noncomputable def count_second_and_elder : ℕ := count_combined_visits days_second_returns days_elder_returns

noncomputable def count_all_three : ℕ := count_all_together_visits [days_youngest_returns, days_second_returns, days_elder_returns]

-- Final Inclusion-Exclusion principle application
noncomputable def days_at_least_one_returns : ℕ := 
  count_youngest_visits + count_second_visits + count_elder_visits
  - count_youngest_and_second
  - count_youngest_and_elder
  - count_second_and_elder
  + count_all_three

theorem daughters_meet_days_count : days_at_least_one_returns = 60 := by
  sorry

end NUMINAMATH_GPT_daughters_meet_days_count_l2373_237323
