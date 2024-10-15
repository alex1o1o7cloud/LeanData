import Mathlib

namespace NUMINAMATH_GPT_problem1_problem2_l1240_124079

namespace TriangleProofs

-- Problem 1: Prove that A + B = π / 2
theorem problem1 (a b c : ℝ) (A B C : ℝ) 
  (m n : ℝ × ℝ) 
  (h1 : m = (a, Real.cos B))
  (h2 : n = (b, Real.cos A))
  (h_parallel : m.1 * n.2 = m.2 * n.1)
  (h_neq : m ≠ n)
  : A + B = Real.pi / 2 :=
sorry

-- Problem 2: Determine the range of x
theorem problem2 (A B : ℝ) (x : ℝ) 
  (h : A + B = Real.pi / 2) 
  (hx : x * Real.sin A * Real.sin B = Real.sin A + Real.sin B) 
  : 2 * Real.sqrt 2 ≤ x :=
sorry

end TriangleProofs

end NUMINAMATH_GPT_problem1_problem2_l1240_124079


namespace NUMINAMATH_GPT_chocolate_chip_cookies_l1240_124095

theorem chocolate_chip_cookies (chocolate_chips_per_recipe : ℕ) (num_recipes : ℕ) (total_chocolate_chips : ℕ) 
  (h1 : chocolate_chips_per_recipe = 2) 
  (h2 : num_recipes = 23) 
  (h3 : total_chocolate_chips = chocolate_chips_per_recipe * num_recipes) : 
  total_chocolate_chips = 46 :=
by
  rw [h1, h2] at h3
  exact h3

-- sorry

end NUMINAMATH_GPT_chocolate_chip_cookies_l1240_124095


namespace NUMINAMATH_GPT_total_amount_saved_l1240_124050

def priceX : ℝ := 575
def surcharge_rateX : ℝ := 0.04
def installation_chargeX : ℝ := 82.50
def total_chargeX : ℝ := priceX + surcharge_rateX * priceX + installation_chargeX

def priceY : ℝ := 530
def surcharge_rateY : ℝ := 0.03
def installation_chargeY : ℝ := 93.00
def total_chargeY : ℝ := priceY + surcharge_rateY * priceY + installation_chargeY

def savings : ℝ := total_chargeX - total_chargeY

theorem total_amount_saved : savings = 41.60 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_saved_l1240_124050


namespace NUMINAMATH_GPT_solution_x_chemical_b_l1240_124025

theorem solution_x_chemical_b (percentage_x_a percentage_y_a percentage_y_b : ℝ) :
  percentage_x_a = 0.3 →
  percentage_y_a = 0.4 →
  percentage_y_b = 0.6 →
  (0.8 * percentage_x_a + 0.2 * percentage_y_a = 0.32) →
  (100 * (1 - percentage_x_a) = 70) :=
by {
  sorry
}

end NUMINAMATH_GPT_solution_x_chemical_b_l1240_124025


namespace NUMINAMATH_GPT_find_max_problems_l1240_124038

def max_problems_in_7_days (P : ℕ → ℕ) : Prop :=
  (∀ i : ℕ, i ∈ Finset.range 7 → P i ≤ 10) ∧
  (∀ i : ℕ, i ∈ Finset.range 5 → (P i > 7) → (P (i + 1) ≤ 5 ∧ P (i + 2) ≤ 5))

theorem find_max_problems : ∃ P : ℕ → ℕ, max_problems_in_7_days P ∧ (Finset.range 7).sum P = 50 :=
by
  sorry

end NUMINAMATH_GPT_find_max_problems_l1240_124038


namespace NUMINAMATH_GPT_complex_number_real_imag_equal_l1240_124092

theorem complex_number_real_imag_equal (a : ℝ) (h : (a + 6) = (3 - 2 * a)) : a = -1 :=
by
  sorry

end NUMINAMATH_GPT_complex_number_real_imag_equal_l1240_124092


namespace NUMINAMATH_GPT_joe_paint_left_after_third_week_l1240_124039

def initial_paint : ℕ := 360

def paint_used_first_week (initial_paint : ℕ) : ℕ := initial_paint / 4

def paint_left_after_first_week (initial_paint : ℕ) : ℕ := initial_paint - paint_used_first_week initial_paint

def paint_used_second_week (paint_left_after_first_week : ℕ) : ℕ := paint_left_after_first_week / 2

def paint_left_after_second_week (paint_left_after_first_week : ℕ) : ℕ := paint_left_after_first_week - paint_used_second_week paint_left_after_first_week

def paint_used_third_week (paint_left_after_second_week : ℕ) : ℕ := paint_left_after_second_week * 2 / 3

def paint_left_after_third_week (paint_left_after_second_week : ℕ) : ℕ := paint_left_after_second_week - paint_used_third_week paint_left_after_second_week

theorem joe_paint_left_after_third_week : 
  paint_left_after_third_week (paint_left_after_second_week (paint_left_after_first_week initial_paint)) = 45 :=
by 
  sorry

end NUMINAMATH_GPT_joe_paint_left_after_third_week_l1240_124039


namespace NUMINAMATH_GPT_sin_2B_minus_5pi_over_6_area_of_triangle_l1240_124051

-- Problem (I)
theorem sin_2B_minus_5pi_over_6 {A B C : ℝ} (a b c : ℝ)
  (h: 3 * Real.cos A * Real.cos C * (Real.tan A * Real.tan C - 1) = 1) :
  Real.sin (2 * B - 5 * Real.pi / 6) = (7 - 4 * Real.sqrt 6) / 18 :=
sorry

-- Problem (II)
theorem area_of_triangle {A B C : ℝ} (a b c : ℝ)
  (h1: a + c = 3 * Real.sqrt 3 / 2) (h2: b = Real.sqrt 3) :
  Real.sqrt (a * c) * Real.sin B / 2 = 15 * Real.sqrt 2 / 32 :=
sorry

end NUMINAMATH_GPT_sin_2B_minus_5pi_over_6_area_of_triangle_l1240_124051


namespace NUMINAMATH_GPT_canal_depth_l1240_124032

theorem canal_depth (A : ℝ) (W_top : ℝ) (W_bottom : ℝ) (d : ℝ) (h: ℝ)
  (h₁ : A = 840) 
  (h₂ : W_top = 12) 
  (h₃ : W_bottom = 8)
  (h₄ : A = (1/2) * (W_top + W_bottom) * d) : 
  d = 84 :=
by 
  sorry

end NUMINAMATH_GPT_canal_depth_l1240_124032


namespace NUMINAMATH_GPT_vasya_can_interfere_with_petya_goal_l1240_124006

theorem vasya_can_interfere_with_petya_goal :
  ∃ (evens odds : ℕ), evens + odds = 50 ∧ (evens + odds) % 2 = 1 :=
sorry

end NUMINAMATH_GPT_vasya_can_interfere_with_petya_goal_l1240_124006


namespace NUMINAMATH_GPT_find_OH_squared_l1240_124010

theorem find_OH_squared (R a b c : ℝ) (hR : R = 10) (hsum : a^2 + b^2 + c^2 = 50) : 
  9 * R^2 - (a^2 + b^2 + c^2) = 850 :=
by
  sorry

end NUMINAMATH_GPT_find_OH_squared_l1240_124010


namespace NUMINAMATH_GPT_relay_race_order_count_l1240_124040

-- Definitions based on the given conditions
def team_members : List String := ["Sam", "Priya", "Jordan", "Luis"]
def first_runner := "Sam"
def last_runner := "Jordan"

-- Theorem stating the number of different possible orders
theorem relay_race_order_count {team_members first_runner last_runner} :
  (team_members = ["Sam", "Priya", "Jordan", "Luis"]) →
  (first_runner = "Sam") →
  (last_runner = "Jordan") →
  (2 = 2) :=
by
  intros _ _ _
  sorry

end NUMINAMATH_GPT_relay_race_order_count_l1240_124040


namespace NUMINAMATH_GPT_probability_of_death_each_month_l1240_124078

-- Defining the variables and expressions used in conditions
def p : ℝ := 0.1
def N : ℝ := 400
def surviving_after_3_months : ℝ := 291.6

-- The main theorem to be proven
theorem probability_of_death_each_month (prob : ℝ) :
  (N * (1 - prob)^3 = surviving_after_3_months) → (prob = p) :=
by
  sorry

end NUMINAMATH_GPT_probability_of_death_each_month_l1240_124078


namespace NUMINAMATH_GPT_B_finish_work_alone_in_12_days_l1240_124041

theorem B_finish_work_alone_in_12_days (A_days B_days both_days : ℕ) :
  A_days = 6 →
  both_days = 4 →
  (1 / A_days + 1 / B_days = 1 / both_days) →
  B_days = 12 :=
by
  intros hA hBoth hRate
  sorry

end NUMINAMATH_GPT_B_finish_work_alone_in_12_days_l1240_124041


namespace NUMINAMATH_GPT_candy_distribution_l1240_124008

theorem candy_distribution (n k : ℕ) (h1 : 3 < n) (h2 : n < 15) (h3 : 195 - n * k = 8) : k = 17 :=
  by
    sorry

end NUMINAMATH_GPT_candy_distribution_l1240_124008


namespace NUMINAMATH_GPT_triangle_angle_and_side_l1240_124020

theorem triangle_angle_and_side (A B C : ℝ)
  (a b c : ℝ)
  (h1 : b * Real.cos A + a * Real.cos B = -2 * c * Real.cos C)
  (h2 : a + b = 6)
  (h3 : 1 / 2 * a * b * Real.sin C = 2 * Real.sqrt 3)
  : C = 2 * Real.pi / 3 ∧ c = 2 * Real.sqrt 7 := by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_triangle_angle_and_side_l1240_124020


namespace NUMINAMATH_GPT_cost_per_mile_l1240_124045

theorem cost_per_mile (m x : ℝ) (h_cost_eq : 2.50 + x * m = 2.50 + 5.00 + x * 14) : 
  x = 5 / 14 :=
by
  sorry

end NUMINAMATH_GPT_cost_per_mile_l1240_124045


namespace NUMINAMATH_GPT_range_of_a_l1240_124057

noncomputable def f (x : ℝ) : ℝ := Real.log x + 3 * x^2
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := 4 * x^2 - a * x

theorem range_of_a (a : ℝ) :
  (∃ x0 : ℝ, x0 > 0 ∧ f x0 = g (-x0) a) → a ≤ -1 := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1240_124057


namespace NUMINAMATH_GPT_anne_speed_ratio_l1240_124066

theorem anne_speed_ratio (B A A' : ℝ) (h_A : A = 1/12) (h_together_current : (B + A) * 4 = 1) (h_together_new : (B + A') * 3 = 1) :
  A' / A = 2 := 
by
  sorry

end NUMINAMATH_GPT_anne_speed_ratio_l1240_124066


namespace NUMINAMATH_GPT_line_properties_l1240_124005

theorem line_properties : 
  ∃ (m b : ℝ), 
  (∀ x : ℝ, ∀ y : ℝ, (x = 1 ∧ y = 3) ∨ (x = 3 ∧ y = 7) → y = m * x + b) ∧
  m + b = 3 ∧
  (∀ x : ℝ, ∀ y : ℝ, (x = 0 ∧ y = 1) → y = m * x + b) :=
sorry

end NUMINAMATH_GPT_line_properties_l1240_124005


namespace NUMINAMATH_GPT_at_least_one_greater_than_zero_l1240_124084

noncomputable def a (x : ℝ) : ℝ := x^2 - 2 * x + (Real.pi / 2)
noncomputable def b (y : ℝ) : ℝ := y^2 - 2 * y + (Real.pi / 2)
noncomputable def c (z : ℝ) : ℝ := z^2 - 2 * z + (Real.pi / 2)

theorem at_least_one_greater_than_zero (x y z : ℝ) : (a x > 0) ∨ (b y > 0) ∨ (c z > 0) :=
by sorry

end NUMINAMATH_GPT_at_least_one_greater_than_zero_l1240_124084


namespace NUMINAMATH_GPT_units_digit_2016_pow_2017_add_2017_pow_2016_l1240_124073

theorem units_digit_2016_pow_2017_add_2017_pow_2016 :
  (2016 ^ 2017 + 2017 ^ 2016) % 10 = 7 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_2016_pow_2017_add_2017_pow_2016_l1240_124073


namespace NUMINAMATH_GPT_part1_part2_part3_l1240_124048

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := (-2^x + b) / (2^(x+1) + 2)

theorem part1 (h_odd : ∀ x : ℝ, f x b = -f (-x) b) : b = 1 :=
sorry

theorem part2 (h_b : b = 1) : ∀ (x1 x2 : ℝ), x1 < x2 → f x1 1 > f x2 1 :=
sorry

theorem part3 (h_monotonic : ∀ (x1 x2 : ℝ), x1 < x2 → f x1 1 > f x2 1) 
  : ∀ t : ℝ, f (t^2 - 2 * t) 1 + f (2 * t^2 - k) 1 < 0 → k < -1/3 :=
sorry

end NUMINAMATH_GPT_part1_part2_part3_l1240_124048


namespace NUMINAMATH_GPT_worker_followed_instructions_l1240_124081

def initial_trees (grid_size : ℕ) : ℕ := grid_size * grid_size

noncomputable def rows_of_trees (rows left each_row : ℕ) : ℕ := rows * each_row

theorem worker_followed_instructions :
  initial_trees 7 = 49 →
  rows_of_trees 5 20 4 = 20 →
  rows_of_trees 5 10 4 = 39 →
  (∃ T : Finset (Fin 7 × Fin 7), T.card = 10) :=
by
  sorry

end NUMINAMATH_GPT_worker_followed_instructions_l1240_124081


namespace NUMINAMATH_GPT_discount_percentage_l1240_124062

theorem discount_percentage (discount amount_paid : ℝ) (h_discount : discount = 40) (h_paid : amount_paid = 120) : 
  (discount / (discount + amount_paid)) * 100 = 25 := by
  sorry

end NUMINAMATH_GPT_discount_percentage_l1240_124062


namespace NUMINAMATH_GPT_find_number_that_gives_200_9_when_8_036_divided_by_it_l1240_124053

theorem find_number_that_gives_200_9_when_8_036_divided_by_it (
  x : ℝ
) : (8.036 / x = 200.9) → (x = 0.04) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_number_that_gives_200_9_when_8_036_divided_by_it_l1240_124053


namespace NUMINAMATH_GPT_sum_underlined_numbers_non_negative_l1240_124044

def sum_underlined_numbers (seq : Fin 100 → Int) : Bool :=
  let underlined_indices : List (Fin 100) :=
    List.range 100 |>.filter (λ i =>
      seq i > 0 ∨ (i < 99 ∧ seq i + seq (i + 1) > 0) ∨ (i < 98 ∧ seq i + seq (i + 1) + seq (i + 2) > 0))
  let underlined_sum : Int := underlined_indices.map (λ i => seq i) |>.sum
  underlined_sum ≤ 0

theorem sum_underlined_numbers_non_negative {seq : Fin 100 → Int} :
  ¬ sum_underlined_numbers seq :=
sorry

end NUMINAMATH_GPT_sum_underlined_numbers_non_negative_l1240_124044


namespace NUMINAMATH_GPT_abc_inequality_l1240_124068

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) : 
    (ab / (a^5 + ab + b^5)) + (bc / (b^5 + bc + c^5)) + (ca / (c^5 + ca + a^5)) ≤ 1 := 
sorry

end NUMINAMATH_GPT_abc_inequality_l1240_124068


namespace NUMINAMATH_GPT_parabola_problem_l1240_124018

noncomputable def p_value_satisfy_all_conditions (p : ℝ) : Prop :=
  ∃ (F : ℝ × ℝ) (A B : ℝ × ℝ),
    F = (p / 2, 0) ∧
    (A.2 = A.1 - p / 2 ∧ (A.2)^2 = 2 * p * A.1) ∧
    (B.2 = B.1 - p / 2 ∧ (B.2)^2 = 2 * p * B.1) ∧
    (A.1 + B.1) / 2 = 3 * p / 2 ∧
    (A.2 + B.2) / 2 = p ∧
    (p - 2 = -3 * p / 2)

theorem parabola_problem : ∃ (p : ℝ), p_value_satisfy_all_conditions p ∧ p = 4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_parabola_problem_l1240_124018


namespace NUMINAMATH_GPT_range_of_a_l1240_124075

theorem range_of_a (a : ℝ) :
  (∃ (x : ℝ), (2 - 2^(-|x - 3|))^2 = 3 + a) ↔ -2 ≤ a ∧ a < 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1240_124075


namespace NUMINAMATH_GPT_problem_statement_l1240_124043

noncomputable def a : ℝ := 13 / 2
noncomputable def b : ℝ := -4

theorem problem_statement :
  ∀ k : ℝ, ∃ x : ℝ, (2 * k * x + a) / 3 = 2 + (x - b * k) / 6 ↔ x = 1 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1240_124043


namespace NUMINAMATH_GPT_kim_boxes_sold_on_tuesday_l1240_124034

theorem kim_boxes_sold_on_tuesday :
  ∀ (T W Th F : ℕ),
  (T = 3 * W) →
  (W = 2 * Th) →
  (Th = 3 / 2 * F) →
  (F = 600) →
  T = 5400 :=
by
  intros T W Th F h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_kim_boxes_sold_on_tuesday_l1240_124034


namespace NUMINAMATH_GPT_usual_time_to_school_l1240_124037

theorem usual_time_to_school (R T : ℕ) (h : 7 * R * (T - 4) = 6 * R * T) : T = 28 :=
sorry

end NUMINAMATH_GPT_usual_time_to_school_l1240_124037


namespace NUMINAMATH_GPT_tangent_line_at_P_exists_c_for_a_l1240_124055

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem tangent_line_at_P :
  ∀ x y : ℝ, y = f x → x = 1 → y = 0 → x - y - 1 = 0 := 
by 
  sorry

theorem exists_c_for_a :
  ∀ a : ℝ, 1 < a → ∃ c : ℝ, 0 < c ∧ c < 1 / a ∧ ∀ x : ℝ, c < x → x < 1 → f x > a * x * (x - 1) :=
by 
  sorry

end NUMINAMATH_GPT_tangent_line_at_P_exists_c_for_a_l1240_124055


namespace NUMINAMATH_GPT_time_for_A_l1240_124029

noncomputable def work_days (A B C D E : ℝ) : Prop :=
  (1/A + 1/B + 1/C + 1/D = 1/8) ∧
  (1/B + 1/C + 1/D + 1/E = 1/6) ∧
  (1/A + 1/E = 1/12)

theorem time_for_A (A B C D E : ℝ) (h : work_days A B C D E) : A = 48 :=
  by
    sorry

end NUMINAMATH_GPT_time_for_A_l1240_124029


namespace NUMINAMATH_GPT_part1_l1240_124002

theorem part1 (m n p : ℝ) (h1 : m > n) (h2 : n > 0) (h3 : p > 0) : 
  (n / m) < (n + p) / (m + p) := 
sorry

end NUMINAMATH_GPT_part1_l1240_124002


namespace NUMINAMATH_GPT_paula_bought_two_shirts_l1240_124035

-- Define the conditions
def total_money : Int := 109
def shirt_cost : Int := 11
def pants_cost : Int := 13
def remaining_money : Int := 74

-- Calculate the expenditure on shirts and pants
def expenditure : Int := total_money - remaining_money

-- Define the number of shirts bought
def number_of_shirts (S : Int) : Prop := expenditure = shirt_cost * S + pants_cost

-- The theorem stating that Paula bought 2 shirts
theorem paula_bought_two_shirts : number_of_shirts 2 :=
by
  -- The proof is omitted as per instructions
  sorry

end NUMINAMATH_GPT_paula_bought_two_shirts_l1240_124035


namespace NUMINAMATH_GPT_find_F_of_circle_l1240_124093

def circle_equation (x y F : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 2*y + F = 0

def is_circle_with_radius (x y F r : ℝ) : Prop := 
  ∃ k h, (x - k)^2 + (y + h)^2 = r

theorem find_F_of_circle {F : ℝ} :
  (∀ x y : ℝ, circle_equation x y F) ∧ 
  is_circle_with_radius 1 1 F 4 → F = -2 := 
by
  sorry

end NUMINAMATH_GPT_find_F_of_circle_l1240_124093


namespace NUMINAMATH_GPT_volume_of_smaller_cube_l1240_124016

noncomputable def volume_of_larger_cube : ℝ := 343
noncomputable def number_of_smaller_cubes : ℝ := 343
noncomputable def surface_area_difference : ℝ := 1764

theorem volume_of_smaller_cube (v_lc : ℝ) (n_sc : ℝ) (sa_diff : ℝ) :
  v_lc = volume_of_larger_cube →
  n_sc = number_of_smaller_cubes →
  sa_diff = surface_area_difference →
  ∃ (v_sc : ℝ), v_sc = 1 :=
by sorry

end NUMINAMATH_GPT_volume_of_smaller_cube_l1240_124016


namespace NUMINAMATH_GPT_division_value_l1240_124099

theorem division_value (x : ℝ) (h : 1376 / x - 160 = 12) : x = 8 := 
by sorry

end NUMINAMATH_GPT_division_value_l1240_124099


namespace NUMINAMATH_GPT_g_of_g_of_g_of_g_of_3_l1240_124059

def g (x : ℕ) : ℕ :=
if x % 3 = 0 then x / 3 else x^2 + 2

theorem g_of_g_of_g_of_g_of_3 : g (g (g (g 3))) = 3 :=
by sorry

end NUMINAMATH_GPT_g_of_g_of_g_of_g_of_3_l1240_124059


namespace NUMINAMATH_GPT_find_interest_rate_l1240_124085

noncomputable def amount : ℝ := 896
noncomputable def principal : ℝ := 799.9999999999999
noncomputable def time : ℝ := 2 + 2 / 5
noncomputable def interest : ℝ := amount - principal
noncomputable def rate : ℝ := interest / (principal * time)

theorem find_interest_rate :
  rate * 100 = 5 := by
  sorry

end NUMINAMATH_GPT_find_interest_rate_l1240_124085


namespace NUMINAMATH_GPT_find_p_l1240_124088

variables (p q : ℚ)
variables (h1 : 2 * p + 5 * q = 10) (h2 : 5 * p + 2 * q = 20)

theorem find_p : p = 80 / 21 :=
by sorry

end NUMINAMATH_GPT_find_p_l1240_124088


namespace NUMINAMATH_GPT_total_cost_of_bicycles_is_2000_l1240_124089

noncomputable def calculate_total_cost_of_bicycles (SP1 SP2 : ℝ) (profit1 profit2 : ℝ) : ℝ :=
  let C1 := SP1 / (1 + profit1)
  let C2 := SP2 / (1 - profit2)
  C1 + C2

theorem total_cost_of_bicycles_is_2000 :
  calculate_total_cost_of_bicycles 990 990 0.10 0.10 = 2000 :=
by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_total_cost_of_bicycles_is_2000_l1240_124089


namespace NUMINAMATH_GPT_theater_cost_per_square_foot_l1240_124021

theorem theater_cost_per_square_foot
    (n_seats : ℕ)
    (space_per_seat : ℕ)
    (cost_ratio : ℕ)
    (partner_coverage : ℕ)
    (tom_expense : ℕ)
    (total_seats := 500)
    (square_footage := total_seats * space_per_seat)
    (construction_cost := cost_ratio * land_cost)
    (total_cost := land_cost + construction_cost)
    (partner_expense := total_cost * partner_coverage / 100)
    (tom_expense_ratio := 100 - partner_coverage)
    (cost_equation := tom_expense = total_cost * tom_expense_ratio / 100)
    (land_cost := 30000) :
    tom_expense = 54000 → 
    space_per_seat = 12 → 
    cost_ratio = 2 →
    partner_coverage = 40 → 
    tom_expense_ratio = 60 → 
    total_cost = 90000 → 
    total_cost / 3 = land_cost →
    land_cost / square_footage = 5 :=
    sorry

end NUMINAMATH_GPT_theater_cost_per_square_foot_l1240_124021


namespace NUMINAMATH_GPT_probability_A_will_receive_2_awards_l1240_124022

def classes := Fin 4
def awards := 8

-- The number of ways to distribute 4 remaining awards to 4 classes
noncomputable def total_distributions : ℕ :=
  Nat.choose (awards - 4 + 4 - 1) (4 - 1)

-- The number of ways when class A receives exactly 2 awards
noncomputable def favorable_distributions : ℕ :=
  Nat.choose (2 + 3 - 1) (4 - 1)

-- The probability that class A receives exactly 2 out of 8 awards
noncomputable def probability_A_receives_2_awards : ℚ :=
  favorable_distributions / total_distributions

theorem probability_A_will_receive_2_awards :
  probability_A_receives_2_awards = 2 / 7 := by
  sorry

end NUMINAMATH_GPT_probability_A_will_receive_2_awards_l1240_124022


namespace NUMINAMATH_GPT_a9_proof_l1240_124027

variable {a : ℕ → ℝ}

-- Conditions
axiom a1 : a 1 = 1
axiom an_recurrence : ∀ n > 1, a n = (a (n - 1)) * 2^(n - 1)

-- Goal
theorem a9_proof : a 9 = 2^36 := 
by 
  sorry

end NUMINAMATH_GPT_a9_proof_l1240_124027


namespace NUMINAMATH_GPT_correct_quotient_remainder_sum_l1240_124090

theorem correct_quotient_remainder_sum :
  ∃ N : ℕ, (N % 23 = 17 ∧ N / 23 = 3) ∧ (∃ q r : ℕ, N = 32 * q + r ∧ r < 32 ∧ q + r = 24) :=
by
  sorry

end NUMINAMATH_GPT_correct_quotient_remainder_sum_l1240_124090


namespace NUMINAMATH_GPT_total_distance_yards_remaining_yards_l1240_124042

structure Distance where
  miles : Nat
  yards : Nat

def marathon_distance : Distance :=
  { miles := 26, yards := 385 }

def miles_to_yards (miles : Nat) : Nat :=
  miles * 1760

def total_yards_in_marathon (d : Distance) : Nat :=
  miles_to_yards d.miles + d.yards

def total_distance_in_yards (d : Distance) (n : Nat) : Nat :=
  n * total_yards_in_marathon d

def remaining_yards (total_yards : Nat) (yards_in_mile : Nat) : Nat :=
  total_yards % yards_in_mile

theorem total_distance_yards_remaining_yards :
    let total_yards := total_distance_in_yards marathon_distance 15
    remaining_yards total_yards 1760 = 495 :=
by
  sorry

end NUMINAMATH_GPT_total_distance_yards_remaining_yards_l1240_124042


namespace NUMINAMATH_GPT_evaluate_expression_l1240_124083

theorem evaluate_expression : 
  (196 * (1 / 17 - 1 / 21) + 361 * (1 / 21 - 1 / 13) + 529 * (1 / 13 - 1 / 17)) /
    (14 * (1 / 17 - 1 / 21) + 19 * (1 / 21 - 1 / 13) + 23 * (1 / 13 - 1 / 17)) = 56 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1240_124083


namespace NUMINAMATH_GPT_thirteen_pow_seven_mod_eight_l1240_124091

theorem thirteen_pow_seven_mod_eight : 
  (13^7) % 8 = 5 := by
  sorry

end NUMINAMATH_GPT_thirteen_pow_seven_mod_eight_l1240_124091


namespace NUMINAMATH_GPT_train_length_l1240_124049

theorem train_length {L : ℝ} (h_equal_lengths : ∃ (L: ℝ), L = L) (h_cross_time : ∃ (t : ℝ), t = 60) (h_speed : ∃ (v : ℝ), v = 20) : L = 600 :=
by
  sorry

end NUMINAMATH_GPT_train_length_l1240_124049


namespace NUMINAMATH_GPT_triangle_angle_bisector_l1240_124000

theorem triangle_angle_bisector 
  (a b l : ℝ) (h1: a > 0) (h2: b > 0) (h3: l > 0) :
  ∃ α : ℝ, α = 2 * Real.arccos (l * (a + b) / (2 * a * b)) :=
by
  sorry

end NUMINAMATH_GPT_triangle_angle_bisector_l1240_124000


namespace NUMINAMATH_GPT_loss_percent_l1240_124012

theorem loss_percent (C S : ℝ) (h : 100 * S = 40 * C) : ((C - S) / C) * 100 = 60 :=
by
  sorry

end NUMINAMATH_GPT_loss_percent_l1240_124012


namespace NUMINAMATH_GPT_min_value_expression_l1240_124030

theorem min_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) :
  (∃ c : ℝ, c = (1 / (2 * x) + x / (y + 1)) ∧ c = 5 / 4) :=
sorry

end NUMINAMATH_GPT_min_value_expression_l1240_124030


namespace NUMINAMATH_GPT_angle_W_in_quadrilateral_l1240_124056

theorem angle_W_in_quadrilateral 
  (W X Y Z : ℝ) 
  (h₀ : W + X + Y + Z = 360) 
  (h₁ : W = 3 * X) 
  (h₂ : W = 4 * Y) 
  (h₃ : W = 6 * Z) : 
  W = 206 :=
by
  sorry

end NUMINAMATH_GPT_angle_W_in_quadrilateral_l1240_124056


namespace NUMINAMATH_GPT_geometric_series_sum_eq_l1240_124003

theorem geometric_series_sum_eq :
  let a := (5 : ℚ)
  let r := (-1/2 : ℚ)
  (∑' n : ℕ, a * r^n) = (10 / 3 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_geometric_series_sum_eq_l1240_124003


namespace NUMINAMATH_GPT_roses_in_each_bouquet_l1240_124061

theorem roses_in_each_bouquet (R : ℕ)
(roses_bouquets daisies_bouquets total_bouquets total_flowers daisies_per_bouquet total_daisies : ℕ)
(h1 : total_bouquets = 20)
(h2 : roses_bouquets = 10)
(h3 : daisies_bouquets = 10)
(h4 : total_flowers = 190)
(h5 : daisies_per_bouquet = 7)
(h6 : total_daisies = daisies_bouquets * daisies_per_bouquet)
(h7 : total_flowers - total_daisies = roses_bouquets * R) :
R = 12 :=
by
  sorry

end NUMINAMATH_GPT_roses_in_each_bouquet_l1240_124061


namespace NUMINAMATH_GPT_max_xy_l1240_124031

theorem max_xy (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (h_eq : 2 * x + 3 * y = 6) : 
  xy ≤ (3/2) :=
sorry

end NUMINAMATH_GPT_max_xy_l1240_124031


namespace NUMINAMATH_GPT_unique_a_for_fx_eq_2ax_l1240_124070

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - 2 * a * Real.log x

theorem unique_a_for_fx_eq_2ax (a : ℝ) (h : 0 < a) :
  (∀ x : ℝ, f x a = 2 * a * x → x = (a + Real.sqrt (a^2 + 4 * a)) / 2) →
  a = 1 / 2 :=
sorry

end NUMINAMATH_GPT_unique_a_for_fx_eq_2ax_l1240_124070


namespace NUMINAMATH_GPT_cubic_eq_root_nature_l1240_124072

-- Definitions based on the problem statement
def cubic_eq (x : ℝ) : Prop := x^3 + 3 * x^2 - 4 * x - 12 = 0

-- The main theorem statement
theorem cubic_eq_root_nature :
  (∃ p n₁ n₂ : ℝ, cubic_eq p ∧ cubic_eq n₁ ∧ cubic_eq n₂ ∧ p > 0 ∧ n₁ < 0 ∧ n₂ < 0 ∧ p ≠ n₁ ∧ p ≠ n₂ ∧ n₁ ≠ n₂) :=
sorry

end NUMINAMATH_GPT_cubic_eq_root_nature_l1240_124072


namespace NUMINAMATH_GPT_angle_in_second_quadrant_l1240_124063

def inSecondQuadrant (θ : ℤ) : Prop :=
  90 < θ ∧ θ < 180

theorem angle_in_second_quadrant :
  ∃ k : ℤ, inSecondQuadrant (-2015 + 360 * k) :=
by {
  sorry
}

end NUMINAMATH_GPT_angle_in_second_quadrant_l1240_124063


namespace NUMINAMATH_GPT_jesters_on_stilts_count_l1240_124047

theorem jesters_on_stilts_count :
  ∃ j e : ℕ, 3 * j + 4 * e = 50 ∧ j + e = 18 ∧ j = 22 :=
by 
  sorry

end NUMINAMATH_GPT_jesters_on_stilts_count_l1240_124047


namespace NUMINAMATH_GPT_committee_problem_solution_l1240_124033

def committee_problem : Prop :=
  let total_committees := Nat.choose 15 5
  let zero_profs_committees := Nat.choose 8 5
  let one_prof_committees := (Nat.choose 7 1) * (Nat.choose 8 4)
  let undesirable_committees := zero_profs_committees + one_prof_committees
  let desired_committees := total_committees - undesirable_committees
  desired_committees = 2457

theorem committee_problem_solution : committee_problem :=
by
  sorry

end NUMINAMATH_GPT_committee_problem_solution_l1240_124033


namespace NUMINAMATH_GPT_find_coefficients_l1240_124052

variables {x1 x2 x3 x4 x5 x6 x7 : ℝ}

theorem find_coefficients
  (h1 : x1 + 4*x2 + 9*x3 + 16*x4 + 25*x5 + 36*x6 + 49*x7 = 5)
  (h2 : 4*x1 + 9*x2 + 16*x3 + 25*x4 + 36*x5 + 49*x6 + 64*x7 = 14)
  (h3 : 9*x1 + 16*x2 + 25*x3 + 36*x4 + 49*x5 + 64*x6 + 81*x7 = 30)
  (h4 : 16*x1 + 25*x2 + 36*x3 + 49*x4 + 64*x5 + 81*x6 + 100*x7 = 70) :
  25*x1 + 36*x2 + 49*x3 + 64*x4 + 81*x5 + 100*x6 + 121*x7 = 130 :=
sorry

end NUMINAMATH_GPT_find_coefficients_l1240_124052


namespace NUMINAMATH_GPT_extremum_f_at_neg_four_thirds_monotonicity_g_l1240_124058

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x ^ 3 + x ^ 2
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (f a x) * Real.exp x
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x ^ 2 + 2 * x
noncomputable def g' (a : ℝ) (x : ℝ) : ℝ := 
  let f_a_x := f a x
  ( f' a x * Real.exp x ) + ( f_a_x * Real.exp x)

theorem extremum_f_at_neg_four_thirds (a : ℝ) :
  f' a (-4/3) = 0 ↔ a = 1/2 := sorry

-- Assuming a = 1/2 from the previous theorem
theorem monotonicity_g :
  let a := 1/2
  ∀ x : ℝ, 
    ((x < -4 → g' a x < 0) ∧ 
     (-4 < x ∧ x < -1 → g' a x > 0) ∧
     (-1 < x ∧ x < 0 → g' a x < 0) ∧
     (x > 0 → g' a x > 0)) := sorry

end NUMINAMATH_GPT_extremum_f_at_neg_four_thirds_monotonicity_g_l1240_124058


namespace NUMINAMATH_GPT_rational_operation_example_l1240_124014

def rational_operation (a b : ℚ) : ℚ := a^3 - 2 * a * b + 4

theorem rational_operation_example : rational_operation 4 (-9) = 140 := 
by
  sorry

end NUMINAMATH_GPT_rational_operation_example_l1240_124014


namespace NUMINAMATH_GPT_trigonometric_identity_l1240_124019

theorem trigonometric_identity : 
  let sin := Real.sin
  let cos := Real.cos
  sin 18 * cos 63 - sin 72 * sin 117 = - (Real.sqrt 2 / 2) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1240_124019


namespace NUMINAMATH_GPT_part1_part2_l1240_124082

noncomputable def f (a c x : ℝ) : ℝ :=
  if x >= c then a * Real.log x + (x - c) ^ 2
  else a * Real.log x - (x - c) ^ 2

theorem part1 (a c : ℝ)
  (h_a : a = 2 * c - 2)
  (h_c_gt_0 : c > 0)
  (h_f_geq : ∀ x, x ∈ (Set.Ioi c) → f a c x >= 1 / 4) :
    a ∈ Set.Icc (-2 : ℝ) (-1 : ℝ) :=
  sorry

theorem part2 (a c x1 x2 : ℝ)
  (h_a_lt_0 : a < 0)
  (h_c_gt_0 : c > 0)
  (h_x1 : x1 = Real.sqrt (- a / 2))
  (h_x2 : x2 = c)
  (h_tangents_intersect : deriv (f a c) x1 * deriv (f a c) x2 = -1) :
    c >= 3 * Real.sqrt 3 / 2 :=
  sorry

end NUMINAMATH_GPT_part1_part2_l1240_124082


namespace NUMINAMATH_GPT_log_inequality_l1240_124024

theorem log_inequality
  (a : ℝ := Real.log 4 / Real.log 5)
  (b : ℝ := (Real.log 3 / Real.log 5)^2)
  (c : ℝ := Real.log 5 / Real.log 4) :
  b < a ∧ a < c :=
by
  sorry

end NUMINAMATH_GPT_log_inequality_l1240_124024


namespace NUMINAMATH_GPT_length_of_platform_l1240_124017

-- Definitions for the given conditions
def speed_of_train_kmph : ℕ := 54
def speed_of_train_mps : ℕ := 15
def time_to_pass_platform : ℕ := 16
def time_to_pass_man : ℕ := 10

-- Main statement of the problem
theorem length_of_platform (v_kmph : ℕ) (v_mps : ℕ) (t_p : ℕ) (t_m : ℕ) 
    (h1 : v_kmph = 54) 
    (h2 : v_mps = 15) 
    (h3 : t_p = 16) 
    (h4 : t_m = 10) : 
    v_mps * t_p - v_mps * t_m = 90 := 
sorry

end NUMINAMATH_GPT_length_of_platform_l1240_124017


namespace NUMINAMATH_GPT_part1_part2_l1240_124023

variables {a m n : ℝ}

theorem part1 (h1 : a^m = 2) (h2 : a^n = 3) : a^(4*m + 3*n) = 432 :=
by sorry

theorem part2 (h1 : a^m = 2) (h2 : a^n = 3) : a^(5*m - 2*n) = 32 / 9 :=
by sorry

end NUMINAMATH_GPT_part1_part2_l1240_124023


namespace NUMINAMATH_GPT_sphere_surface_area_l1240_124015

-- Define the conditions
def points_on_sphere (A B C : Type) := 
  ∃ (AB BC AC : Real), AB = 6 ∧ BC = 8 ∧ AC = 10

-- Define the distance condition
def distance_condition (R : Real) := 
  ∃ (d : Real), d = R / 2

-- Define the main theorem
theorem sphere_surface_area 
  (A B C : Type) 
  (h_points : points_on_sphere A B C) 
  (h_distance : ∃ R : Real, distance_condition R) : 
  4 * Real.pi * (10 / 3 * Real.sqrt 3) ^ 2 = 400 / 3 * Real.pi := 
by 
  sorry

end NUMINAMATH_GPT_sphere_surface_area_l1240_124015


namespace NUMINAMATH_GPT_not_sufficient_nor_necessary_geometric_seq_l1240_124046

theorem not_sufficient_nor_necessary_geometric_seq {a : ℕ → ℝ} (q : ℝ) (h_geom : ∀ n, a (n + 1) = a n * q) :
    (a 1 < a 3) ↔ (¬(a 2 < a 4) ∨ ¬(a 4 < a 2)) :=
by
  sorry

end NUMINAMATH_GPT_not_sufficient_nor_necessary_geometric_seq_l1240_124046


namespace NUMINAMATH_GPT_find_percentage_l1240_124097

noncomputable def percentage_condition (P : ℝ) : Prop :=
  9000 + (P / 100) * 9032 = 10500

theorem find_percentage (P : ℝ) (h : percentage_condition P) : P = 16.61 :=
sorry

end NUMINAMATH_GPT_find_percentage_l1240_124097


namespace NUMINAMATH_GPT_problem_evaluation_l1240_124054

theorem problem_evaluation (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c + d)⁻¹ * (a⁻¹ + b⁻¹ + c⁻¹ + d⁻¹) * (ab + bc + cd + da + ac + bd)⁻¹ * ((ab)⁻¹ + (bc)⁻¹ + (cd)⁻¹ + (da)⁻¹ + (ac)⁻¹ + (bd)⁻¹) = 
  (1 / (a * b * c * d)) * (1 / (a * b * c * d)) :=
by
  sorry

end NUMINAMATH_GPT_problem_evaluation_l1240_124054


namespace NUMINAMATH_GPT_range_of_a_for_increasing_function_l1240_124071

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (6 - a) * x - 2 * a else a ^ x

theorem range_of_a_for_increasing_function (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x ≤ f a y) ↔ (3/2 ≤ a ∧ a < 6) := sorry

end NUMINAMATH_GPT_range_of_a_for_increasing_function_l1240_124071


namespace NUMINAMATH_GPT_factorization_correct_l1240_124007

theorem factorization_correct (x : ℝ) :
  16 * x ^ 2 + 8 * x - 24 = 8 * (2 * x ^ 2 + x - 3) ∧ (2 * x ^ 2 + x - 3) = (2 * x + 3) * (x - 1) :=
by
  sorry

end NUMINAMATH_GPT_factorization_correct_l1240_124007


namespace NUMINAMATH_GPT_problem_solution_l1240_124076

def sequence_graphical_representation_isolated (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, ∃ x : ℝ, x = a n

def sequence_terms_infinite (a : ℕ → ℝ) : Prop :=
  ∃ l : List ℝ, ∃ n : ℕ, l.length = n

def sequence_general_term_formula_unique (a : ℕ → ℝ) : Prop :=
  ∀ f g : ℕ → ℝ, (∀ n, f n = g n) → f = g

theorem problem_solution
  (h1 : ∀ a : ℕ → ℝ, sequence_graphical_representation_isolated a)
  (h2 : ¬ ∀ a : ℕ → ℝ, sequence_terms_infinite a)
  (h3 : ¬ ∀ a : ℕ → ℝ, sequence_general_term_formula_unique a) :
  ∀ a : ℕ → ℝ, sequence_graphical_representation_isolated a ∧ 
                ¬ (sequence_terms_infinite a) ∧
                ¬ (sequence_general_term_formula_unique a) := by
  sorry

end NUMINAMATH_GPT_problem_solution_l1240_124076


namespace NUMINAMATH_GPT_extremum_areas_extremum_areas_case_b_equal_areas_l1240_124074

variable (a b x : ℝ)
variable (h1 : b > 0) (h2 : a ≥ b) (h_cond : 0 < x ∧ x ≤ b)

def area_t1 (a b x : ℝ) : ℝ := 2 * x^2 - (a + b) * x + a * b
def area_t2 (a b x : ℝ) : ℝ := -2 * x^2 + (a + b) * x

noncomputable def x0 (a b : ℝ) : ℝ := (a + b) / 4

-- Problem 1
theorem extremum_areas :
  b ≥ a / 3 → area_t1 a b (x0 a b) ≤ area_t1 a b x ∧ area_t2 a b (x0 a b) ≥ area_t2 a b x :=
sorry

theorem extremum_areas_case_b :
  b < a / 3 → (area_t1 a b b = b^2) ∧ (area_t2 a b b = a * b - b^2) :=
sorry

-- Problem 2
theorem equal_areas :
  b ≤ a ∧ a ≤ 2 * b → (area_t1 a b (a / 2) = area_t2 a b (a / 2)) ∧ (area_t1 a b (b / 2) = area_t2 a b (b / 2)) :=
sorry

end NUMINAMATH_GPT_extremum_areas_extremum_areas_case_b_equal_areas_l1240_124074


namespace NUMINAMATH_GPT_num_divisors_360_l1240_124028

theorem num_divisors_360 :
  ∀ n : ℕ, n = 360 → (∀ (p q r : ℕ), p = 2 ∧ q = 3 ∧ r = 5 →
    (∃ (a b c : ℕ), 360 = p^a * q^b * r^c ∧ a = 3 ∧ b = 2 ∧ c = 1) →
    (3+1) * (2+1) * (1+1) = 24) :=
  sorry

end NUMINAMATH_GPT_num_divisors_360_l1240_124028


namespace NUMINAMATH_GPT_more_people_needed_to_paint_fence_l1240_124026

theorem more_people_needed_to_paint_fence :
  ∀ (n t m t' : ℕ), n = 8 → t = 3 → t' = 2 → (n * t = m * t') → m - n = 4 :=
by
  intros n t m t'
  intro h1
  intro h2
  intro h3
  intro h4
  sorry

end NUMINAMATH_GPT_more_people_needed_to_paint_fence_l1240_124026


namespace NUMINAMATH_GPT_pure_imaginary_condition_l1240_124060

theorem pure_imaginary_condition (a b : ℝ) : 
  (a = 0) ↔ (∃ b : ℝ, b ≠ 0 ∧ z = a + b * I) :=
sorry

end NUMINAMATH_GPT_pure_imaginary_condition_l1240_124060


namespace NUMINAMATH_GPT_LynsDonation_l1240_124087

theorem LynsDonation (X : ℝ)
  (h1 : 1/3 * X + 1/2 * X + 1/4 * (X - (1/3 * X + 1/2 * X)) = 3/4 * X)
  (h2 : (X - 3/4 * X)/4 = 30) :
  X = 240 := by
  sorry

end NUMINAMATH_GPT_LynsDonation_l1240_124087


namespace NUMINAMATH_GPT_asia_paid_140_l1240_124065

noncomputable def original_price : ℝ := 350
noncomputable def discount_percentage : ℝ := 0.60
noncomputable def discount_amount : ℝ := original_price * discount_percentage
noncomputable def final_price : ℝ := original_price - discount_amount

theorem asia_paid_140 : final_price = 140 := by
  unfold final_price
  unfold discount_amount
  unfold original_price
  unfold discount_percentage
  sorry

end NUMINAMATH_GPT_asia_paid_140_l1240_124065


namespace NUMINAMATH_GPT_max_value_expression_l1240_124096

theorem max_value_expression (x : ℝ) : 
  ∃ m : ℝ, m = 1 / 37 ∧ ∀ x : ℝ, (x^6) / (x^12 + 3*x^9 - 5*x^6 + 15*x^3 + 27) ≤ m :=
sorry

end NUMINAMATH_GPT_max_value_expression_l1240_124096


namespace NUMINAMATH_GPT_coords_reflect_origin_l1240_124086

def P : Type := (ℤ × ℤ)

def reflect_origin (p : P) : P :=
  (-p.1, -p.2)

theorem coords_reflect_origin (p : P) (hx : p = (2, -1)) : reflect_origin p = (-2, 1) :=
by
  sorry

end NUMINAMATH_GPT_coords_reflect_origin_l1240_124086


namespace NUMINAMATH_GPT_third_pipe_empty_time_l1240_124004

theorem third_pipe_empty_time (x : ℝ) :
  (1 / 60 : ℝ) + (1 / 120) - (1 / x) = (1 / 60) →
  x = 120 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_third_pipe_empty_time_l1240_124004


namespace NUMINAMATH_GPT_find_angle_B_find_triangle_area_l1240_124011

open Real

theorem find_angle_B (B : ℝ) (h : sqrt 3 * sin (2 * B) = 1 - cos (2 * B)) : B = π / 3 :=
sorry

theorem find_triangle_area (BC A B : ℝ) (hBC : BC = 2) (hA : A = π / 4) (hB : B = π / 3) :
  let AC := BC * (sin B / sin A)
  let C := π - A - B
  let area := (1 / 2) * AC * BC * sin C
  area = (3 + sqrt 3) / 2 :=
sorry


end NUMINAMATH_GPT_find_angle_B_find_triangle_area_l1240_124011


namespace NUMINAMATH_GPT_calculate_expression_l1240_124069

theorem calculate_expression : 
  -3^2 + Real.sqrt ((-2)^4) - (-27)^(1/3 : ℝ) = -2 := 
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1240_124069


namespace NUMINAMATH_GPT_num_positive_integers_l1240_124067

-- Definitions
def is_divisor (a b : ℕ) : Prop := ∃ k, b = k * a

-- Problem statement
theorem num_positive_integers (n : ℕ) (h : n = 2310) :
  (∃ count, count = 3 ∧ (∀ m : ℕ, m > 0 → is_divisor (m^2 - 2) n → count = 3)) := by
  sorry

end NUMINAMATH_GPT_num_positive_integers_l1240_124067


namespace NUMINAMATH_GPT_combined_students_yellow_blue_l1240_124094

theorem combined_students_yellow_blue {total_students blue_percent red_percent yellow_combined : ℕ} :
  total_students = 200 →
  blue_percent = 30 →
  red_percent = 40 →
  yellow_combined = (total_students * 3 / 10) + ((total_students - (total_students * 3 / 10)) * 6 / 10) →
  yellow_combined = 144 :=
by
  intros
  sorry

end NUMINAMATH_GPT_combined_students_yellow_blue_l1240_124094


namespace NUMINAMATH_GPT_jill_basket_total_weight_l1240_124036

def jill_basket_capacity : ℕ := 24
def type_a_weight : ℕ := 150
def type_b_weight : ℕ := 170
def jill_basket_type_a_count : ℕ := 12
def jill_basket_type_b_count : ℕ := 12

theorem jill_basket_total_weight :
  (jill_basket_type_a_count * type_a_weight + jill_basket_type_b_count * type_b_weight) = 3840 :=
by
  -- We provide the calculations for clarification; not essential to the theorem statement
  -- (12 * 150) + (12 * 170) = 1800 + 2040 = 3840
  -- Started proof to provide context; actual proof steps are omitted
  sorry

end NUMINAMATH_GPT_jill_basket_total_weight_l1240_124036


namespace NUMINAMATH_GPT_complement_M_l1240_124001

noncomputable def U : Set ℝ := Set.univ

def M : Set ℝ := { x | x^2 - 4 ≤ 0 }

theorem complement_M : U \ M = { x | x < -2 ∨ x > 2 } :=
by 
  sorry

end NUMINAMATH_GPT_complement_M_l1240_124001


namespace NUMINAMATH_GPT_johns_allowance_is_3_45_l1240_124064

noncomputable def johns_weekly_allowance (A : ℝ) : Prop :=
  -- Condition 1: John spent 3/5 of his allowance at the arcade
  let spent_at_arcade := (3/5) * A
  -- Remaining allowance
  let remaining_after_arcade := A - spent_at_arcade
  -- Condition 2: He spent 1/3 of the remaining allowance at the toy store
  let spent_at_toy_store := (1/3) * remaining_after_arcade
  let remaining_after_toy_store := remaining_after_arcade - spent_at_toy_store
  -- Condition 3: He spent his last $0.92 at the candy store
  let spent_at_candy_store := 0.92
  -- Remaining amount after the candy store expenditure should be 0
  remaining_after_toy_store = spent_at_candy_store

theorem johns_allowance_is_3_45 : johns_weekly_allowance 3.45 :=
sorry

end NUMINAMATH_GPT_johns_allowance_is_3_45_l1240_124064


namespace NUMINAMATH_GPT_unique_solution_conditions_l1240_124080

-- Definitions based on the conditions
variables {x y a : ℝ}

def inequality_condition (x y a : ℝ) : Prop := 
  x^2 + y^2 + 2 * x ≤ 1

def equation_condition (x y a : ℝ) : Prop := 
  x - y = -a

-- Main Theorem Statement
theorem unique_solution_conditions (a : ℝ) : 
  (∃! x y : ℝ, inequality_condition x y a ∧ equation_condition x y a) ↔ (a = 1 + Real.sqrt 2 ∨ a = 1 - Real.sqrt 2) :=
sorry

end NUMINAMATH_GPT_unique_solution_conditions_l1240_124080


namespace NUMINAMATH_GPT_gcd_45_75_l1240_124013

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end NUMINAMATH_GPT_gcd_45_75_l1240_124013


namespace NUMINAMATH_GPT_inequality_xyz_l1240_124098

theorem inequality_xyz (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 :=
sorry

end NUMINAMATH_GPT_inequality_xyz_l1240_124098


namespace NUMINAMATH_GPT_product_of_sum_positive_and_quotient_negative_l1240_124009

-- Definitions based on conditions in the problem
def sum_positive (a b : ℝ) : Prop := a + b > 0
def quotient_negative (a b : ℝ) : Prop := a / b < 0

-- Problem statement as a theorem
theorem product_of_sum_positive_and_quotient_negative (a b : ℝ)
  (h1 : sum_positive a b)
  (h2 : quotient_negative a b) :
  a * b < 0 := by
  sorry

end NUMINAMATH_GPT_product_of_sum_positive_and_quotient_negative_l1240_124009


namespace NUMINAMATH_GPT_ice_cream_flavors_l1240_124077

theorem ice_cream_flavors (n k : ℕ) (h1 : n = 6) (h2 : k = 4) :
  (n + k - 1).choose (k - 1) = 84 :=
by
  have h3 : n = 6 := h1
  have h4 : k = 4 := h2
  rw [h3, h4]
  sorry

end NUMINAMATH_GPT_ice_cream_flavors_l1240_124077
