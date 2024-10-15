import Mathlib

namespace NUMINAMATH_GPT_hemisphere_surface_area_ratio_l2110_211055

theorem hemisphere_surface_area_ratio 
  (r : ℝ) (sphere_surface_area : ℝ) (hemisphere_surface_area : ℝ) 
  (eq1 : sphere_surface_area = 4 * π * r^2) 
  (eq2 : hemisphere_surface_area = 3 * π * r^2) : 
  hemisphere_surface_area / sphere_surface_area = 3 / 4 :=
by sorry

end NUMINAMATH_GPT_hemisphere_surface_area_ratio_l2110_211055


namespace NUMINAMATH_GPT_students_watching_l2110_211098

theorem students_watching (b g : ℕ) (h : b + g = 33) : (2 / 3 : ℚ) * b + (2 / 3 : ℚ) * g = 22 := by
  sorry

end NUMINAMATH_GPT_students_watching_l2110_211098


namespace NUMINAMATH_GPT_least_five_digit_integer_congruent_3_mod_17_l2110_211085

theorem least_five_digit_integer_congruent_3_mod_17 : 
  ∃ n, n ≥ 10000 ∧ n % 17 = 3 ∧ ∀ m, (m ≥ 10000 ∧ m % 17 = 3) → n ≤ m := 
sorry

end NUMINAMATH_GPT_least_five_digit_integer_congruent_3_mod_17_l2110_211085


namespace NUMINAMATH_GPT_remainder_2_pow_305_mod_9_l2110_211032

theorem remainder_2_pow_305_mod_9 :
  2^305 % 9 = 5 :=
by sorry

end NUMINAMATH_GPT_remainder_2_pow_305_mod_9_l2110_211032


namespace NUMINAMATH_GPT_triangle_inequality_power_sum_l2110_211049

theorem triangle_inequality_power_sum
  (a b c : ℝ) (n : ℕ)
  (h_a_bc : a + b + c = 1)
  (h_a_b_c : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c)
  (h_a_triangl : a + b > c)
  (h_b_triangl : b + c > a)
  (h_c_triangl : c + a > b)
  (h_n : n > 1) :
  (a^n + b^n)^(1/n : ℝ) + (b^n + c^n)^(1/n : ℝ) + (c^n + a^n)^(1/n : ℝ) < 1 + (2^(1/n : ℝ)) / 2 :=
by
  sorry

end NUMINAMATH_GPT_triangle_inequality_power_sum_l2110_211049


namespace NUMINAMATH_GPT_cold_brew_cost_l2110_211078

theorem cold_brew_cost :
  let drip_coffee_cost := 2.25
  let espresso_cost := 3.50
  let latte_cost := 4.00
  let vanilla_syrup_cost := 0.50
  let cappuccino_cost := 3.50
  let total_order_cost := 25.00
  let drip_coffee_total := 2 * drip_coffee_cost
  let lattes_total := 2 * latte_cost
  let known_costs := drip_coffee_total + espresso_cost + lattes_total + vanilla_syrup_cost + cappuccino_cost
  total_order_cost - known_costs = 5.00 →
  5.00 / 2 = 2.50 := by sorry

end NUMINAMATH_GPT_cold_brew_cost_l2110_211078


namespace NUMINAMATH_GPT_sum_of_roots_l2110_211031

theorem sum_of_roots (α β : ℝ)
  (hα : α^3 - 3*α^2 + 5*α - 4 = 0)
  (hβ : β^3 - 3*β^2 + 5*β - 2 = 0) :
  α + β = 2 :=
sorry

end NUMINAMATH_GPT_sum_of_roots_l2110_211031


namespace NUMINAMATH_GPT_average_annual_growth_rate_l2110_211037

theorem average_annual_growth_rate (x : ℝ) (h1 : 6.4 * (1 + x)^2 = 8.1) : x = 0.125 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_average_annual_growth_rate_l2110_211037


namespace NUMINAMATH_GPT_scalene_triangles_count_l2110_211001

/-- Proving existence of exactly 3 scalene triangles with integer side lengths and perimeter < 13. -/
theorem scalene_triangles_count : 
  ∃ (triangles : Finset (ℕ × ℕ × ℕ)), 
    triangles.card = 3 ∧ 
    ∀ (a b c : ℕ), (a, b, c) ∈ triangles → a < b ∧ b < c ∧ a + b + c < 13 :=
sorry

end NUMINAMATH_GPT_scalene_triangles_count_l2110_211001


namespace NUMINAMATH_GPT_meaningful_fraction_l2110_211091

theorem meaningful_fraction (x : ℝ) : (x ≠ 5) ↔ (∃ y : ℝ, y = 1 / (x - 5)) :=
by
  sorry

end NUMINAMATH_GPT_meaningful_fraction_l2110_211091


namespace NUMINAMATH_GPT_exact_time_now_l2110_211073

noncomputable def minute_hand_position (t : ℝ) : ℝ := 6 * (t + 4)
noncomputable def hour_hand_position (t : ℝ) : ℝ := 0.5 * (t - 2) + 270
noncomputable def is_opposite (x y : ℝ) : Prop := |x - y| = 180

theorem exact_time_now (t : ℝ) (h1 : 0 ≤ t) (h2 : t < 60)
  (h3 : is_opposite (minute_hand_position t) (hour_hand_position t)) :
  t = 591/50 :=
by
  sorry

end NUMINAMATH_GPT_exact_time_now_l2110_211073


namespace NUMINAMATH_GPT_triangle_problem_l2110_211002

noncomputable def a : ℝ := 2 * Real.sqrt 3
noncomputable def B : ℝ := 45
noncomputable def S : ℝ := 3 + Real.sqrt 3

noncomputable def c : ℝ := Real.sqrt 2 + Real.sqrt 6
noncomputable def C : ℝ := 75

theorem triangle_problem
  (a_val : a = 2 * Real.sqrt 3)
  (B_val : B = 45)
  (S_val : S = 3 + Real.sqrt 3) :
  c = Real.sqrt 2 + Real.sqrt 6 ∧ C = 75 :=
by
  sorry

end NUMINAMATH_GPT_triangle_problem_l2110_211002


namespace NUMINAMATH_GPT_observations_count_l2110_211064

theorem observations_count (n : ℕ) 
  (original_mean : ℚ) (wrong_value_corrected : ℚ) (corrected_mean : ℚ)
  (h1 : original_mean = 36)
  (h2 : wrong_value_corrected = 1)
  (h3 : corrected_mean = 36.02) :
  n = 50 :=
by
  sorry

end NUMINAMATH_GPT_observations_count_l2110_211064


namespace NUMINAMATH_GPT_fraction_zero_solution_l2110_211023

theorem fraction_zero_solution (x : ℝ) (h1 : |x| - 3 = 0) (h2 : x + 3 ≠ 0) : x = 3 := 
sorry

end NUMINAMATH_GPT_fraction_zero_solution_l2110_211023


namespace NUMINAMATH_GPT_vasya_wins_l2110_211034

-- Define the grid size and initial setup
def grid_size : ℕ := 13
def initial_stones : ℕ := 2023

-- Define a condition that checks if a move can put a stone on the 13th cell
def can_win (position : ℕ) : Prop :=
  position = grid_size

-- Define the game logic for Petya and Vasya
def next_position (pos : ℕ) (move : ℕ) : ℕ :=
  pos + move

-- Ensure a win by always ensuring the next move does not leave Petya on positions 4, 7, 10, 13
def winning_strategy_for_vasya (current_pos : ℕ) (move : ℕ) : Prop :=
  (next_position current_pos move) ≠ 4 ∧
  (next_position current_pos move) ≠ 7 ∧
  (next_position current_pos move) ≠ 10 ∧
  (next_position current_pos move) ≠ 13

theorem vasya_wins : ∃ strategy : ℕ → ℕ → Prop,
  ∀ current_pos move, winning_strategy_for_vasya current_pos move → can_win (next_position current_pos move) :=
by
  sorry -- To be provided

end NUMINAMATH_GPT_vasya_wins_l2110_211034


namespace NUMINAMATH_GPT_smallest_three_digit_plus_one_multiple_l2110_211088

theorem smallest_three_digit_plus_one_multiple (x : ℕ) : 
  (421 = x) →
  (x ≥ 100 ∧ x < 1000) ∧ 
  ∃ k : ℕ, x = k * Nat.lcm (Nat.lcm 3 4) * Nat.lcm 5 7 + 1 :=
by
  sorry

end NUMINAMATH_GPT_smallest_three_digit_plus_one_multiple_l2110_211088


namespace NUMINAMATH_GPT_find_c_l2110_211059

def is_midpoint (p1 p2 mid : ℝ × ℝ) : Prop :=
(mid.1 = (p1.1 + p2.1) / 2) ∧ (mid.2 = (p1.2 + p2.2) / 2)

def is_perpendicular_bisector (line : ℝ → ℝ → ℝ) (p1 p2 : ℝ × ℝ) : Prop := 
∃ mid : ℝ × ℝ, 
is_midpoint p1 p2 mid ∧ line mid.1 mid.2 = 0

theorem find_c (c : ℝ) : 
is_perpendicular_bisector (λ x y => 3 * x - y - c) (2, 4) (6, 8) → c = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_c_l2110_211059


namespace NUMINAMATH_GPT_no_such_function_exists_l2110_211036

theorem no_such_function_exists :
  ¬ ∃ f : ℝ → ℝ, ∀ x y : ℝ, f (x - f y) = 1 + x - y := by
  sorry

end NUMINAMATH_GPT_no_such_function_exists_l2110_211036


namespace NUMINAMATH_GPT_weekly_rental_fee_percentage_l2110_211089

theorem weekly_rental_fee_percentage
  (camera_value : ℕ)
  (rental_period_weeks : ℕ)
  (friend_percentage : ℚ)
  (john_paid : ℕ)
  (percentage : ℚ)
  (total_rental_fee : ℚ)
  (weekly_rental_fee : ℚ)
  (P : ℚ)
  (camera_value_pos : camera_value = 5000)
  (rental_period_weeks_pos : rental_period_weeks = 4)
  (friend_percentage_pos : friend_percentage = 0.40)
  (john_paid_pos : john_paid = 1200)
  (percentage_pos : percentage = 1 - friend_percentage)
  (total_rental_fee_calc : total_rental_fee = john_paid / percentage)
  (weekly_rental_fee_calc : weekly_rental_fee = total_rental_fee / rental_period_weeks)
  (weekly_rental_fee_equation : weekly_rental_fee = P * camera_value)
  (P_calc : P = weekly_rental_fee / camera_value) :
  P * 100 = 10 := 
by 
  sorry

end NUMINAMATH_GPT_weekly_rental_fee_percentage_l2110_211089


namespace NUMINAMATH_GPT_grid_X_value_l2110_211041

theorem grid_X_value :
  ∃ X, (∃ b d1 d2 d3 d4, 
    b = 16 ∧
    d1 = (25 - 20) ∧
    d2 = (16 - 15) / 3 ∧
    d3 = (d1 * 5) / 4 ∧
    d4 = d1 - d3 ∧
    (-12 - d4 * 4) = -30 ∧ 
    X = d4 ∧
    X = 10.5) :=
sorry

end NUMINAMATH_GPT_grid_X_value_l2110_211041


namespace NUMINAMATH_GPT_find_f_ln_log_52_l2110_211017

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := ((x + 1)^2 + a * Real.sin x) / (x^2 + 1) + 3

axiom given_condition (a : ℝ) : f a (Real.log (Real.log 5 / Real.log 2)) = 5

theorem find_f_ln_log_52 (a : ℝ) : f a (Real.log (Real.log 2 / Real.log 5)) = 3 :=
by
  -- The details of the proof are omitted
  sorry

end NUMINAMATH_GPT_find_f_ln_log_52_l2110_211017


namespace NUMINAMATH_GPT_emily_widgets_production_l2110_211050

variable (w t : ℕ) (work_hours_monday work_hours_tuesday production_monday production_tuesday : ℕ)

theorem emily_widgets_production :
  (w = 2 * t) → 
  (work_hours_monday = t) →
  (work_hours_tuesday = t - 3) →
  (production_monday = w * work_hours_monday) → 
  (production_tuesday = (w + 6) * work_hours_tuesday) →
  (production_monday - production_tuesday) = 18 :=
by
  intros hw hwm hwmt hpm hpt
  sorry

end NUMINAMATH_GPT_emily_widgets_production_l2110_211050


namespace NUMINAMATH_GPT_sum_modulo_remainder_l2110_211008

theorem sum_modulo_remainder :
  ((82 + 83 + 84 + 85 + 86 + 87 + 88 + 89) % 17) = 12 :=
by
  sorry

end NUMINAMATH_GPT_sum_modulo_remainder_l2110_211008


namespace NUMINAMATH_GPT_convex_quadrilateral_inequality_l2110_211090

variable (a b c d : ℝ) -- lengths of sides of quadrilateral
variable (S : ℝ) -- Area of the quadrilateral

-- Given condition: a, b, c, d are lengths of the sides of a convex quadrilateral
def is_convex_quadrilateral (a b c d : ℝ) (S : ℝ) : Prop :=
  S ≤ (a^2 + b^2 + c^2 + d^2) / 4

theorem convex_quadrilateral_inequality (a b c d : ℝ) (S : ℝ) 
  (h : is_convex_quadrilateral a b c d S) : 
  S ≤ (a^2 + b^2 + c^2 + d^2) / 4 := 
by
  sorry

end NUMINAMATH_GPT_convex_quadrilateral_inequality_l2110_211090


namespace NUMINAMATH_GPT_average_of_roots_l2110_211030

theorem average_of_roots (a b: ℝ) (h : a ≠ 0) (hr : ∃ x1 x2: ℝ, a * x1 ^ 2 - 3 * a * x1 + b = 0 ∧ a * x2 ^ 2 - 3 * a * x2 + b = 0 ∧ x1 ≠ x2):
  (∃ r1 r2: ℝ, a * r1 ^ 2 - 3 * a * r1 + b = 0 ∧ a * r2 ^ 2 - 3 * a * r2 + b = 0 ∧ r1 ≠ r2) →
  ((r1 + r2) / 2 = 3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_average_of_roots_l2110_211030


namespace NUMINAMATH_GPT_first_equation_value_l2110_211053

theorem first_equation_value (x y : ℝ) (V : ℝ) 
  (h1 : x + |x| + y = V) 
  (h2 : x + |y| - y = 6) 
  (h3 : x + y = 12) : 
  V = 18 := 
by
  sorry

end NUMINAMATH_GPT_first_equation_value_l2110_211053


namespace NUMINAMATH_GPT_monotonic_intervals_max_min_values_on_interval_l2110_211079

noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.exp x

theorem monotonic_intervals :
  (∀ x > -2, 0 < (x + 2) * Real.exp x) ∧ (∀ x < -2, (x + 2) * Real.exp x < 0) :=
by
  sorry

theorem max_min_values_on_interval :
  let a := -4
  let b := 0
  let f_a := (-4 + 1) * Real.exp (-4)
  let f_b := (0 + 1) * Real.exp 0
  let f_c := (-2 + 1) * Real.exp (-2)
  (f b = 1) ∧ (f_c = -1 / Real.exp 2) ∧ (f_a < f_b) ∧ (f_a < f_c) ∧ (f_c < f_b) :=
by
  sorry

end NUMINAMATH_GPT_monotonic_intervals_max_min_values_on_interval_l2110_211079


namespace NUMINAMATH_GPT_find_t_l2110_211015

-- Define the utility function
def utility (r j : ℕ) : ℕ := r * j

-- Define the Wednesday and Thursday utilities
def utility_wednesday (t : ℕ) : ℕ := utility (t + 1) (7 - t)
def utility_thursday (t : ℕ) : ℕ := utility (3 - t) (t + 4)

theorem find_t : (utility_wednesday t = utility_thursday t) → t = 5 / 8 :=
by
  sorry

end NUMINAMATH_GPT_find_t_l2110_211015


namespace NUMINAMATH_GPT_impossible_grid_arrangement_l2110_211027

theorem impossible_grid_arrangement :
  ¬ ∃ (f : Fin 25 → Fin 41 → ℤ),
    (∀ i j, abs (f i j - f (i + 1) j) ≤ 16 ∧ abs (f i j - f i (j + 1)) ≤ 16 ∧
            f i j ≠ f (i + 1) j ∧ f i j ≠ f i (j + 1)) := 
sorry

end NUMINAMATH_GPT_impossible_grid_arrangement_l2110_211027


namespace NUMINAMATH_GPT_minimum_value_expression_l2110_211019

theorem minimum_value_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ z, (z = a^2 + b^2 + 1 / a^2 + 2 * b / a) ∧ z ≥ 2 :=
sorry

end NUMINAMATH_GPT_minimum_value_expression_l2110_211019


namespace NUMINAMATH_GPT_counting_numbers_dividing_56_greater_than_2_l2110_211087

theorem counting_numbers_dividing_56_greater_than_2 :
  (∃ (A : Finset ℕ), A = {n ∈ (Finset.range 57) | n > 2 ∧ 56 % n = 0} ∧ A.card = 5) :=
sorry

end NUMINAMATH_GPT_counting_numbers_dividing_56_greater_than_2_l2110_211087


namespace NUMINAMATH_GPT_jar_last_days_l2110_211099

theorem jar_last_days :
  let serving_size := 0.5 -- each serving is 0.5 ounces
  let daily_servings := 3  -- James uses 3 servings every day
  let quart_ounces := 32   -- 1 quart = 32 ounces
  let jar_size := quart_ounces - 2 -- container is 2 ounces less than 1 quart
  let daily_consumption := daily_servings * serving_size
  let number_of_days := jar_size / daily_consumption
  number_of_days = 20 := by
  sorry

end NUMINAMATH_GPT_jar_last_days_l2110_211099


namespace NUMINAMATH_GPT_find_x2_y2_l2110_211035

variable (x y : ℝ)

-- Given conditions
def average_commute_time (x y : ℝ) := (x + y + 10 + 11 + 9) / 5 = 10
def variance_commute_time (x y : ℝ) := ( (x - 10) ^ 2 + (y - 10) ^ 2 + (10 - 10) ^ 2 + (11 - 10) ^ 2 + (9 - 10) ^ 2 ) / 5 = 2

-- The theorem to prove
theorem find_x2_y2 (hx_avg : average_commute_time x y) (hx_var : variance_commute_time x y) : 
  x^2 + y^2 = 208 :=
sorry

end NUMINAMATH_GPT_find_x2_y2_l2110_211035


namespace NUMINAMATH_GPT_radius_circle_B_l2110_211096

theorem radius_circle_B (rA rB rD : ℝ) 
  (hA : rA = 2) (hD : rD = 2 * rA) (h_tangent : (rA + rB) ^ 2 = rD ^ 2) : 
  rB = 2 :=
by
  sorry

end NUMINAMATH_GPT_radius_circle_B_l2110_211096


namespace NUMINAMATH_GPT_bowling_ball_weight_l2110_211009

-- Definitions based on given conditions
variable (k b : ℕ)

-- Condition 1: one kayak weighs 35 pounds
def kayak_weight : Prop := k = 35

-- Condition 2: four kayaks weigh the same as five bowling balls
def balance_equation : Prop := 4 * k = 5 * b

-- Goal: prove the weight of one bowling ball is 28 pounds
theorem bowling_ball_weight (hk : kayak_weight k) (hb : balance_equation k b) : b = 28 :=
by
  sorry

end NUMINAMATH_GPT_bowling_ball_weight_l2110_211009


namespace NUMINAMATH_GPT_linear_eq_value_abs_sum_l2110_211038

theorem linear_eq_value_abs_sum (a m : ℤ)
  (h1: m^2 - 9 = 0)
  (h2: m ≠ 3)
  (h3: |a| ≤ 3) : 
  |a + m| + |a - m| = 6 :=
by
  sorry

end NUMINAMATH_GPT_linear_eq_value_abs_sum_l2110_211038


namespace NUMINAMATH_GPT_sugar_for_recipe_l2110_211065

theorem sugar_for_recipe (sugar_frosting sugar_cake : ℝ) (h1 : sugar_frosting = 0.6) (h2 : sugar_cake = 0.2) :
  sugar_frosting + sugar_cake = 0.8 :=
by
  sorry

end NUMINAMATH_GPT_sugar_for_recipe_l2110_211065


namespace NUMINAMATH_GPT_division_of_powers_of_ten_l2110_211068

theorem division_of_powers_of_ten : 10^8 / (2 * 10^6) = 50 := by 
  sorry

end NUMINAMATH_GPT_division_of_powers_of_ten_l2110_211068


namespace NUMINAMATH_GPT_solution_to_exponential_equation_l2110_211094

theorem solution_to_exponential_equation :
  ∃ x : ℕ, (8^12 + 8^12 + 8^12 = 2^x) ∧ x = 38 :=
by
  sorry

end NUMINAMATH_GPT_solution_to_exponential_equation_l2110_211094


namespace NUMINAMATH_GPT_factorize_expression_l2110_211075

-- Variables used in the expression
variables (m n : ℤ)

-- The expression to be factored
def expr := 4 * m^3 * n - 16 * m * n^3

-- The desired factorized form of the expression
def factored := 4 * m * n * (m + 2 * n) * (m - 2 * n)

-- The proof problem statement
theorem factorize_expression : expr m n = factored m n :=
by sorry

end NUMINAMATH_GPT_factorize_expression_l2110_211075


namespace NUMINAMATH_GPT_sequence_general_term_l2110_211048

theorem sequence_general_term (a : ℕ → ℕ) (n : ℕ) (h1 : a 1 = 1)
  (h2 : ∀ n, a (n + 1) = a n + 2 * n + 1) : a n = n * n :=
by
  sorry

end NUMINAMATH_GPT_sequence_general_term_l2110_211048


namespace NUMINAMATH_GPT_missing_fraction_is_two_l2110_211066

theorem missing_fraction_is_two :
  (1/2) + (-5/6) + (1/5) + (1/4) + (-9/20) + (-5/6) + 2 = 0.8333333333333334 := by
  sorry

end NUMINAMATH_GPT_missing_fraction_is_two_l2110_211066


namespace NUMINAMATH_GPT_jens_son_age_l2110_211033

theorem jens_son_age
  (J : ℕ)
  (S : ℕ)
  (h1 : J = 41)
  (h2 : J = 3 * S - 7) :
  S = 16 :=
by
  sorry

end NUMINAMATH_GPT_jens_son_age_l2110_211033


namespace NUMINAMATH_GPT_bob_age_is_eleven_l2110_211076

/-- 
Susan, Arthur, Tom, and Bob are siblings. Arthur is 2 years older than Susan, 
Tom is 3 years younger than Bob. Susan is 15 years old, 
and the total age of all four family members is 51 years. 
This theorem states that Bob is 11 years old.
-/

theorem bob_age_is_eleven
  (S A T B : ℕ)
  (h1 : A = S + 2)
  (h2 : T = B - 3)
  (h3 : S = 15)
  (h4 : S + A + T + B = 51) : 
  B = 11 :=
  sorry

end NUMINAMATH_GPT_bob_age_is_eleven_l2110_211076


namespace NUMINAMATH_GPT_ticket_cost_l2110_211010

-- Conditions
def seats : ℕ := 400
def capacity_percentage : ℝ := 0.8
def performances : ℕ := 3
def total_revenue : ℝ := 28800

-- Question: Prove that the cost of each ticket is $30
theorem ticket_cost : (total_revenue / (seats * capacity_percentage * performances)) = 30 := 
by
  sorry

end NUMINAMATH_GPT_ticket_cost_l2110_211010


namespace NUMINAMATH_GPT_total_cost_of_pencils_and_erasers_l2110_211028

theorem total_cost_of_pencils_and_erasers 
  (pencil_cost : ℕ)
  (eraser_cost : ℕ)
  (pencils_bought : ℕ)
  (erasers_bought : ℕ)
  (total_cost_dollars : ℝ)
  (cents_to_dollars : ℝ)
  (hc : pencil_cost = 2)
  (he : eraser_cost = 5)
  (hp : pencils_bought = 500)
  (he2 : erasers_bought = 250)
  (cents_to_dollars_def : cents_to_dollars = 100)
  (total_cost_calc : total_cost_dollars = 
    ((pencils_bought * pencil_cost + erasers_bought * eraser_cost : ℕ) : ℝ) / cents_to_dollars) 
  : total_cost_dollars = 22.50 :=
sorry

end NUMINAMATH_GPT_total_cost_of_pencils_and_erasers_l2110_211028


namespace NUMINAMATH_GPT_average_last_30_l2110_211097

theorem average_last_30 (avg_first_40 : ℝ) 
  (avg_all_70 : ℝ) 
  (sum_first_40 : ℝ := 40 * avg_first_40)
  (sum_all_70 : ℝ := 70 * avg_all_70) 
  (total_results: ℕ := 70):
  (30 : ℝ) * (40: ℝ) + (30: ℝ) * (40: ℝ) = 70 * 34.285714285714285 :=
by
  sorry

end NUMINAMATH_GPT_average_last_30_l2110_211097


namespace NUMINAMATH_GPT_triangle_area_range_l2110_211081

theorem triangle_area_range (A B C : ℝ) (a b c : ℝ) 
  (h1 : a * Real.sin B = Real.sqrt 3 * b * Real.cos A)
  (h2 : a = 3) :
  0 < (1 / 2) * b * c * Real.sin A ∧ 
  (1 / 2) * b * c * Real.sin A ≤ (9 * Real.sqrt 3) / 4 := 
  sorry

end NUMINAMATH_GPT_triangle_area_range_l2110_211081


namespace NUMINAMATH_GPT_integer_solutions_of_inequality_count_l2110_211004

theorem integer_solutions_of_inequality_count :
  let a := -2 - Real.sqrt 6
  let b := -2 + Real.sqrt 6
  ∃ n, n = 5 ∧ ∀ x : ℤ, x < a ∨ b < x ↔ (4 * x^2 + 16 * x + 15 ≤ 23) → n = 5 :=
by sorry

end NUMINAMATH_GPT_integer_solutions_of_inequality_count_l2110_211004


namespace NUMINAMATH_GPT_probability_of_BEI3_is_zero_l2110_211080

def isVowelOrDigit (s : Char) : Prop :=
  (s ∈ ['A', 'E', 'I', 'O', 'U']) ∨ (s.isDigit)

def isNonVowel (s : Char) : Prop :=
  s ∈ ['B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z']

def isHexDigit (s : Char) : Prop :=
  s.isDigit ∨ s ∈ ['A', 'B', 'C', 'D', 'E', 'F']

noncomputable def numPossiblePlates : Nat :=
  13 * 21 * 20 * 16

theorem probability_of_BEI3_is_zero :
    ∃ (totalPlates : Nat), 
    (totalPlates = numPossiblePlates) ∧
    ¬(isVowelOrDigit 'B') →
    (1 : ℚ) / (totalPlates : ℚ) = 0 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_BEI3_is_zero_l2110_211080


namespace NUMINAMATH_GPT_right_pyramid_volume_l2110_211044

noncomputable def volume_of_right_pyramid (base_area lateral_face_area total_surface_area : ℝ) : ℝ := 
  let height := (10 : ℝ) / 3
  (1 / 3) * base_area * height

theorem right_pyramid_volume (total_surface_area base_area lateral_face_area : ℝ)
  (h0 : total_surface_area = 300)
  (h1 : base_area + 3 * lateral_face_area = total_surface_area)
  (h2 : lateral_face_area = base_area / 3) 
  : volume_of_right_pyramid base_area lateral_face_area total_surface_area = 500 / 3 := 
by
  sorry

end NUMINAMATH_GPT_right_pyramid_volume_l2110_211044


namespace NUMINAMATH_GPT_unique_solution_set_l2110_211012

theorem unique_solution_set :
  {a : ℝ | ∃! x : ℝ, (x^2 - 4) / (x + a) = 1} = { -17 / 4, -2, 2 } :=
by sorry

end NUMINAMATH_GPT_unique_solution_set_l2110_211012


namespace NUMINAMATH_GPT_remainder_of_polynomial_l2110_211056

-- Define the polynomial
def P (x : ℝ) : ℝ := x^4 - 4 * x^2 + 7 * x - 8

-- State the theorem
theorem remainder_of_polynomial (x : ℝ) : P 3 = 50 := sorry

end NUMINAMATH_GPT_remainder_of_polynomial_l2110_211056


namespace NUMINAMATH_GPT_ice_cubes_total_l2110_211077

theorem ice_cubes_total (initial_cubes made_cubes : ℕ) (h_initial : initial_cubes = 2) (h_made : made_cubes = 7) : initial_cubes + made_cubes = 9 :=
by
  sorry

end NUMINAMATH_GPT_ice_cubes_total_l2110_211077


namespace NUMINAMATH_GPT_correct_mark_l2110_211025

theorem correct_mark (x : ℕ) (h1 : 73 - x = 10) : x = 63 :=
by
  sorry

end NUMINAMATH_GPT_correct_mark_l2110_211025


namespace NUMINAMATH_GPT_tank_capacity_l2110_211057

theorem tank_capacity (C : ℕ) 
  (leak_rate : C / 4 = C / 4)               -- Condition: Leak rate is C/4 litres per hour
  (inlet_rate : 6 * 60 = 360)                -- Condition: Inlet rate is 360 litres per hour
  (net_emptying_rate : C / 12 = (360 - C / 4))  -- Condition: Net emptying rate for 12 hours
  : C = 1080 := 
by 
  -- Conditions imply that C = 1080 
  sorry

end NUMINAMATH_GPT_tank_capacity_l2110_211057


namespace NUMINAMATH_GPT_arithmetic_seq_infinitely_many_squares_l2110_211067

theorem arithmetic_seq_infinitely_many_squares 
  (a d : ℕ) 
  (h : ∃ (n y : ℕ), a + n * d = y^2) : 
  ∃ (m : ℕ), ∀ k : ℕ, ∃ n' y' : ℕ, a + n' * d = y'^2 :=
by sorry

end NUMINAMATH_GPT_arithmetic_seq_infinitely_many_squares_l2110_211067


namespace NUMINAMATH_GPT_compute_expression_l2110_211022

theorem compute_expression (x1 y1 x2 y2 x3 y3 : ℝ)
  (h1 : x1^3 - 3 * x1 * y1^2 = 2017)
  (h2 : y1^3 - 3 * x1^2 * y1 = 2016)
  (h3 : x2^3 - 3 * x2 * y2^2 = 2017)
  (h4 : y2^3 - 3 * x2^2 * y2 = 2016)
  (h5 : x3^3 - 3 * x3 * y3^2 = 2017)
  (h6 : y3^3 - 3 * x3^2 * y3 = 2016) :
  (2 - x1 / y1) * (2 - x2 / y2) * (2 - x3 / y3) = 26219 / 2016 := 
by
  sorry

end NUMINAMATH_GPT_compute_expression_l2110_211022


namespace NUMINAMATH_GPT_x_plus_2y_equals_5_l2110_211082

theorem x_plus_2y_equals_5 (x y : ℝ) (h1 : 2 * x + y = 6) (h2 : (x + y) / 3 = 1.222222222222222) : x + 2 * y = 5 := 
by sorry

end NUMINAMATH_GPT_x_plus_2y_equals_5_l2110_211082


namespace NUMINAMATH_GPT_prime_sq_mod_12_l2110_211062

theorem prime_sq_mod_12 (p : ℕ) (hp_prime : Nat.Prime p) (hp_gt_3 : p > 3) : (p * p) % 12 = 1 := by
  sorry

end NUMINAMATH_GPT_prime_sq_mod_12_l2110_211062


namespace NUMINAMATH_GPT_solve_equation_l2110_211040

theorem solve_equation (x : ℝ) (h : x ≠ 3) : 
  -x^2 = (3*x - 3) / (x - 3) → x = 1 :=
by
  intro h1
  sorry

end NUMINAMATH_GPT_solve_equation_l2110_211040


namespace NUMINAMATH_GPT_second_frog_hops_eq_18_l2110_211058

-- Define the given conditions
variables (x : ℕ) (h3 : ℕ)

def second_frog_hops := 2 * h3
def first_frog_hops := 4 * second_frog_hops
def total_hops := h3 + second_frog_hops + first_frog_hops

-- The proof goal
theorem second_frog_hops_eq_18 (H : total_hops = 99) : second_frog_hops = 18 :=
by
  sorry

end NUMINAMATH_GPT_second_frog_hops_eq_18_l2110_211058


namespace NUMINAMATH_GPT_simplify_expression_l2110_211007

-- Define constants
variables (z : ℝ)

-- Define the problem and its solution
theorem simplify_expression :
  (5 - 2 * z) - (4 + 5 * z) = 1 - 7 * z := 
sorry

end NUMINAMATH_GPT_simplify_expression_l2110_211007


namespace NUMINAMATH_GPT_area_of_rhombus_l2110_211013

variable (a b θ : ℝ)
variable (h_a : 0 < a) (h_b : 0 < b)

theorem area_of_rhombus (h : true) : (2 * a) * (2 * b) / 2 = 2 * a * b := by
  sorry

end NUMINAMATH_GPT_area_of_rhombus_l2110_211013


namespace NUMINAMATH_GPT_minimum_value_of_expression_l2110_211092

theorem minimum_value_of_expression (x : ℝ) (hx : x > 0) : 6 * x + 1 / x ^ 6 ≥ 7 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l2110_211092


namespace NUMINAMATH_GPT_combined_instruments_l2110_211005

def charlie_flutes := 1
def charlie_horns := 2
def charlie_harps := 1

def carli_flutes := 2 * charlie_flutes
def carli_horns := charlie_horns / 2
def carli_harps := 0

def charlie_total := charlie_flutes + charlie_horns + charlie_harps
def carli_total := carli_flutes + carli_horns + carli_harps
def combined_total := charlie_total + carli_total

theorem combined_instruments :
  combined_total = 7 :=
by
  sorry

end NUMINAMATH_GPT_combined_instruments_l2110_211005


namespace NUMINAMATH_GPT_set_intersection_eq_l2110_211051

def setA : Set ℝ := { x | x^2 - 3 * x - 4 > 0 }
def setB : Set ℝ := { x | -2 < x ∧ x < 5 }
def setC : Set ℝ := { x | (-2 < x ∧ x < -1) ∨ (4 < x ∧ x < 5) }

theorem set_intersection_eq : setA ∩ setB = setC := by
  sorry

end NUMINAMATH_GPT_set_intersection_eq_l2110_211051


namespace NUMINAMATH_GPT_cost_small_and_large_puzzle_l2110_211021

-- Define the cost of a large puzzle L and the cost equation for large and small puzzles
def cost_large_puzzle : ℤ := 15

def cost_equation (S : ℤ) : Prop := cost_large_puzzle + 3 * S = 39

-- Theorem to prove the total cost of a small puzzle and a large puzzle together
theorem cost_small_and_large_puzzle : ∃ S : ℤ, cost_equation S ∧ (S + cost_large_puzzle = 23) :=
by
  sorry

end NUMINAMATH_GPT_cost_small_and_large_puzzle_l2110_211021


namespace NUMINAMATH_GPT_xy_sum_of_squares_l2110_211095

theorem xy_sum_of_squares (x y : ℝ) (h1 : x - y = 18) (h2 : x + y = 22) : x^2 + y^2 = 404 := by
  sorry

end NUMINAMATH_GPT_xy_sum_of_squares_l2110_211095


namespace NUMINAMATH_GPT_sum_of_unit_fractions_l2110_211026

theorem sum_of_unit_fractions : (1 / 2) + (1 / 3) + (1 / 7) + (1 / 42) = 1 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_unit_fractions_l2110_211026


namespace NUMINAMATH_GPT_min_k_value_l2110_211083

variable (p q r s k : ℕ)

/-- Prove the smallest value of k for which p, q, r, and s are positive integers and 
    satisfy the given equations is 77
-/
theorem min_k_value (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (hs : 0 < s)
  (eq1 : p + 2 * q + 3 * r + 4 * s = k)
  (eq2 : 4 * p = 3 * q)
  (eq3 : 4 * p = 2 * r)
  (eq4 : 4 * p = s) : k = 77 :=
sorry

end NUMINAMATH_GPT_min_k_value_l2110_211083


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l2110_211018

theorem necessary_and_sufficient_condition (a b : ℝ) (h1 : a > b) (h2 : 1 / a > 1 / b) : a > 0 ∧ 0 > b :=
by
  sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l2110_211018


namespace NUMINAMATH_GPT_Christopher_joggers_eq_80_l2110_211016

variable (T A C : ℕ)

axiom Tyson_joggers : T > 0                  -- Tyson bought a positive number of joggers.

axiom Alexander_condition : A = T + 22        -- Alexander bought 22 more joggers than Tyson.

axiom Christopher_condition : C = 20 * T      -- Christopher bought twenty times as many joggers as Tyson.

axiom Christopher_Alexander : C = A + 54     -- Christopher bought 54 more joggers than Alexander.

theorem Christopher_joggers_eq_80 : C = 80 := 
by
  sorry

end NUMINAMATH_GPT_Christopher_joggers_eq_80_l2110_211016


namespace NUMINAMATH_GPT_polynomial_remainder_l2110_211006

-- Define the polynomial p(x)
def p (x : ℝ) : ℝ := x^3 + 2*x + 3

-- Define the divisor q(x)
def q (x : ℝ) : ℝ := x + 2

-- The theorem asserting the remainder when p(x) is divided by q(x)
theorem polynomial_remainder : (p (-2)) = -9 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_remainder_l2110_211006


namespace NUMINAMATH_GPT_zero_is_smallest_natural_number_l2110_211046

theorem zero_is_smallest_natural_number : ∀ n : ℕ, 0 ≤ n :=
by
  intro n
  exact Nat.zero_le n

#check zero_is_smallest_natural_number  -- confirming the theorem check

end NUMINAMATH_GPT_zero_is_smallest_natural_number_l2110_211046


namespace NUMINAMATH_GPT_mary_income_percentage_l2110_211047

-- Declare noncomputable as necessary
noncomputable def calculate_percentage_more
    (J : ℝ) -- Juan's income
    (T : ℝ) (M : ℝ)
    (hT : T = 0.70 * J) -- Tim's income is 30% less than Juan's income
    (hM : M = 1.12 * J) -- Mary's income is 112% of Juan's income
    : ℝ :=
  ((M - T) / T) * 100

theorem mary_income_percentage
    (J T M : ℝ)
    (hT : T = 0.70 * J)
    (hM : M = 1.12 * J) :
    calculate_percentage_more J T M hT hM = 60 :=
by sorry

end NUMINAMATH_GPT_mary_income_percentage_l2110_211047


namespace NUMINAMATH_GPT_Sasha_earnings_proof_l2110_211072

def Monday_hours : ℕ := 90  -- 1.5 hours * 60 minutes/hour
def Tuesday_minutes : ℕ := 75  -- 1 hour * 60 minutes/hour + 15 minutes
def Wednesday_minutes : ℕ := 115  -- 11:10 AM - 9:15 AM
def Thursday_minutes : ℕ := 45

def total_minutes_worked : ℕ := Monday_hours + Tuesday_minutes + Wednesday_minutes + Thursday_minutes

def hourly_rate : ℚ := 4.50
def total_hours : ℚ := total_minutes_worked / 60

def weekly_earnings : ℚ := total_hours * hourly_rate

theorem Sasha_earnings_proof : weekly_earnings = 24 := by
  sorry

end NUMINAMATH_GPT_Sasha_earnings_proof_l2110_211072


namespace NUMINAMATH_GPT_Adam_total_balls_l2110_211086

def number_of_red_balls := 20
def number_of_blue_balls := 10
def number_of_orange_balls := 5
def number_of_pink_balls := 3 * number_of_orange_balls

def total_number_of_balls := 
  number_of_red_balls + number_of_blue_balls + number_of_pink_balls + number_of_orange_balls

theorem Adam_total_balls : total_number_of_balls = 50 := by
  sorry

end NUMINAMATH_GPT_Adam_total_balls_l2110_211086


namespace NUMINAMATH_GPT_evaluate_expression_l2110_211071

theorem evaluate_expression (x : ℤ) (h : x = 4) : 3 * x + 5 = 17 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2110_211071


namespace NUMINAMATH_GPT_surface_area_of_cube_is_correct_l2110_211024

noncomputable def edge_length (a : ℝ) : ℝ := 5 * a

noncomputable def surface_area_of_cube (a : ℝ) : ℝ :=
  let edge := edge_length a
  6 * edge * edge

theorem surface_area_of_cube_is_correct (a : ℝ) :
  surface_area_of_cube a = 150 * a ^ 2 := by
  sorry

end NUMINAMATH_GPT_surface_area_of_cube_is_correct_l2110_211024


namespace NUMINAMATH_GPT_rabbit_speed_l2110_211042

theorem rabbit_speed (s : ℕ) (h : (s * 2 + 4) * 2 = 188) : s = 45 :=
sorry

end NUMINAMATH_GPT_rabbit_speed_l2110_211042


namespace NUMINAMATH_GPT_pure_imaginary_iff_a_eq_2_l2110_211039

theorem pure_imaginary_iff_a_eq_2 (a : ℝ) : (∃ k : ℝ, (∃ x : ℝ, (2-a) / 2 = x ∧ x = 0) ∧ (2+a)/2 = k ∧ k ≠ 0) ↔ a = 2 :=
by
  sorry

end NUMINAMATH_GPT_pure_imaginary_iff_a_eq_2_l2110_211039


namespace NUMINAMATH_GPT_difference_between_max_and_min_coins_l2110_211003

theorem difference_between_max_and_min_coins (n : ℕ) : 
  (∃ x y : ℕ, x * 10 + y * 25 = 45 ∧ x + y = n) →
  (∃ p q : ℕ, p * 10 + q * 25 = 45 ∧ p + q = n) →
  (n = 2) :=
by
  sorry

end NUMINAMATH_GPT_difference_between_max_and_min_coins_l2110_211003


namespace NUMINAMATH_GPT_valid_duty_schedules_l2110_211029

noncomputable def validSchedules : ℕ := 
  let A_schedule := Nat.choose 7 4  -- \binom{7}{4} for A
  let B_schedule := Nat.choose 4 4  -- \binom{4}{4} for B
  let C_schedule := Nat.choose 6 3  -- \binom{6}{3} for C
  let D_schedule := Nat.choose 5 5  -- \binom{5}{5} for D
  A_schedule * B_schedule * C_schedule * D_schedule

theorem valid_duty_schedules : validSchedules = 700 := by
  -- proof steps will go here
  sorry

end NUMINAMATH_GPT_valid_duty_schedules_l2110_211029


namespace NUMINAMATH_GPT_circles_radius_difference_l2110_211069

variable (s : ℝ)

theorem circles_radius_difference (h : (π * (2*s)^2) / (π * s^2) = 4) : (2 * s - s) = s :=
by
  sorry

end NUMINAMATH_GPT_circles_radius_difference_l2110_211069


namespace NUMINAMATH_GPT_sum_of_areas_of_sixteen_disks_l2110_211093

theorem sum_of_areas_of_sixteen_disks :
  let r := 1 - (2:ℝ).sqrt
  let area_one_disk := r^2 * Real.pi
  let total_area := 16 * area_one_disk
  total_area = Real.pi * (48 - 32 * (2:ℝ).sqrt) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_areas_of_sixteen_disks_l2110_211093


namespace NUMINAMATH_GPT_number_of_perfect_square_factors_of_360_l2110_211020

def is_perfect_square (n : ℕ) : Prop :=
  ∃ (m : ℕ), m * m = n

def number_of_perfect_square_factors (n : ℕ) : ℕ :=
  if n = 360 then 4 else 0

theorem number_of_perfect_square_factors_of_360 :
  number_of_perfect_square_factors 360 = 4 := 
by {
  -- Sorry is used here as a placeholder for the proof steps.
  sorry
}

end NUMINAMATH_GPT_number_of_perfect_square_factors_of_360_l2110_211020


namespace NUMINAMATH_GPT_sum_of_three_numbers_l2110_211054

theorem sum_of_three_numbers :
  ∃ A B C : ℕ, 
    (100 ≤ A ∧ A < 1000) ∧  -- A is a three-digit number
    (10 ≤ B ∧ B < 100) ∧     -- B is a two-digit number
    (10 ≤ C ∧ C < 100) ∧     -- C is a two-digit number
    (A + (if (B / 10 = 7 ∨ B % 10 = 7) then B else 0) + 
       (if (C / 10 = 7 ∨ C % 10 = 7) then C else 0) = 208) ∧
    (if (B / 10 = 3 ∨ B % 10 = 3) then B else 0) + 
    (if (C / 10 = 3 ∨ C % 10 = 3) then C else 0) = 76 ∧
    A + B + C = 247 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_three_numbers_l2110_211054


namespace NUMINAMATH_GPT_solve_x_l2110_211011

noncomputable def solveEquation (a b c d : ℝ) (x : ℝ) : Prop :=
  x = 3 * a * b + 33 * b^2 + 333 * c^3 + 3.33 * (Real.sin d)^4

theorem solve_x :
  solveEquation 2 (-1) 0.5 (Real.pi / 6) 68.833125 :=
by
  sorry

end NUMINAMATH_GPT_solve_x_l2110_211011


namespace NUMINAMATH_GPT_round_2741836_to_nearest_integer_l2110_211063

theorem round_2741836_to_nearest_integer :
  (2741836.4928375).round = 2741836 := 
by
  -- Explanation that 0.4928375 < 0.5 leading to rounding down
  sorry

end NUMINAMATH_GPT_round_2741836_to_nearest_integer_l2110_211063


namespace NUMINAMATH_GPT_above_265_is_234_l2110_211060

namespace PyramidArray

-- Definition of the pyramid structure and identifying important properties
def is_number_in_pyramid (n : ℕ) : Prop :=
  ∃ k : ℕ, (k^2 - (k - 1)^2) / 2 ≥ n ∧ (k^2 - (k - 1)^2) / 2 < n + (2 * k - 1)

def row_start (k : ℕ) : ℕ :=
  (k - 1)^2 + 1

def row_end (k : ℕ) : ℕ :=
  k^2

def number_above (n : ℕ) (r : ℕ) : ℕ :=
  row_start r + ((n - row_start (r + 1)) % (2 * (r + 1) - 1))

theorem above_265_is_234 : 
  (number_above 265 16) = 234 := 
sorry

end PyramidArray

end NUMINAMATH_GPT_above_265_is_234_l2110_211060


namespace NUMINAMATH_GPT_total_students_in_class_l2110_211061

theorem total_students_in_class 
  (avg_age_all : ℝ)
  (num_students1 : ℕ) (avg_age1 : ℝ)
  (num_students2 : ℕ) (avg_age2 : ℝ)
  (age_student17 : ℕ)
  (total_students : ℕ) :
  avg_age_all = 17 →
  num_students1 = 5 →
  avg_age1 = 14 →
  num_students2 = 9 →
  avg_age2 = 16 →
  age_student17 = 75 →
  total_students = num_students1 + num_students2 + 1 →
  total_students = 17 :=
by
  intro h_avg_all h_num1 h_avg1 h_num2 h_avg2 h_age17 h_total
  -- Additional proof steps would go here
  sorry

end NUMINAMATH_GPT_total_students_in_class_l2110_211061


namespace NUMINAMATH_GPT_mixed_solution_concentration_l2110_211000

def salt_amount_solution1 (weight1 : ℕ) (concentration1 : ℕ) : ℕ := (concentration1 * weight1) / 100
def salt_amount_solution2 (salt2 : ℕ) : ℕ := salt2
def total_salt (salt1 salt2 : ℕ) : ℕ := salt1 + salt2
def total_weight (weight1 weight2 : ℕ) : ℕ := weight1 + weight2
def concentration (total_salt : ℕ) (total_weight : ℕ) : ℚ := (total_salt : ℚ) / (total_weight : ℚ) * 100

theorem mixed_solution_concentration 
  (weight1 weight2 salt2 : ℕ) (concentration1 : ℕ)
  (h_weight1 : weight1 = 200)
  (h_weight2 : weight2 = 300)
  (h_concentration1 : concentration1 = 25)
  (h_salt2 : salt2 = 60) :
  concentration (total_salt (salt_amount_solution1 weight1 concentration1) (salt_amount_solution2 salt2)) (total_weight weight1 weight2) = 22 := 
sorry

end NUMINAMATH_GPT_mixed_solution_concentration_l2110_211000


namespace NUMINAMATH_GPT_cannot_make_it_in_time_l2110_211045

theorem cannot_make_it_in_time (time_available : ℕ) (distance_to_station : ℕ) (v1 : ℕ) :
  time_available = 2 ∧ distance_to_station = 2 ∧ v1 = 30 → 
  ¬ ∃ v2, (time_available - (distance_to_station / v1)) * v2 ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_cannot_make_it_in_time_l2110_211045


namespace NUMINAMATH_GPT_cost_per_pound_of_mixed_candy_l2110_211043

def w1 := 10
def p1 := 8
def w2 := 20
def p2 := 5

theorem cost_per_pound_of_mixed_candy : 
    (w1 * p1 + w2 * p2) / (w1 + w2) = 6 := by
  sorry

end NUMINAMATH_GPT_cost_per_pound_of_mixed_candy_l2110_211043


namespace NUMINAMATH_GPT_convert_base_5_to_base_10_l2110_211014

theorem convert_base_5_to_base_10 :
  let a3 := 2 * 5^3
  let a2 := 2 * 5^2
  let a1 := 0 * 5^1
  let a0 := 2 * 5^0
  a3 + a2 + a1 + a0 = 302 := by
  let a3 := 2 * 5^3
  let a2 := 2 * 5^2
  let a1 := 0 * 5^1
  let a0 := 2 * 5^0
  show a3 + a2 + a1 + a0 = 302
  sorry

end NUMINAMATH_GPT_convert_base_5_to_base_10_l2110_211014


namespace NUMINAMATH_GPT_simplify_sqrt_expression_l2110_211074

theorem simplify_sqrt_expression :
  (3 * (Real.sqrt (4 * 3)) - 2 * (Real.sqrt (1 / 3)) +
     Real.sqrt (16 * 3)) / (2 * Real.sqrt 3) = 14 / 3 := by
sorry

end NUMINAMATH_GPT_simplify_sqrt_expression_l2110_211074


namespace NUMINAMATH_GPT_sum_arithmetic_sequence_terms_l2110_211084

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n m : ℕ, a (n + m) = a n + m * (a 1 - a 0)

theorem sum_arithmetic_sequence_terms (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a) 
  (h₅ : a 5 = 8) :
  a 2 + a 4 + a 5 + a 9 = 32 :=
by
  sorry

end NUMINAMATH_GPT_sum_arithmetic_sequence_terms_l2110_211084


namespace NUMINAMATH_GPT_fraction_collectors_edition_is_correct_l2110_211052

-- Let's define the necessary conditions
variable (DinaDolls IvyDolls CollectorsEditionDolls : ℕ)
variable (FractionCollectorsEdition : ℚ)

-- Given conditions
axiom DinaHas60Dolls : DinaDolls = 60
axiom DinaHasTwiceAsManyDollsAsIvy : DinaDolls = 2 * IvyDolls
axiom IvyHas20CollectorsEditionDolls : CollectorsEditionDolls = 20

-- The statement to prove
theorem fraction_collectors_edition_is_correct :
  FractionCollectorsEdition = (CollectorsEditionDolls : ℚ) / (IvyDolls : ℚ) ∧
  DinaDolls = 60 →
  DinaDolls = 2 * IvyDolls →
  CollectorsEditionDolls = 20 →
  FractionCollectorsEdition = 2 / 3 := 
by
  sorry

end NUMINAMATH_GPT_fraction_collectors_edition_is_correct_l2110_211052


namespace NUMINAMATH_GPT_union_correct_l2110_211070

variable (x : ℝ)
def A := {x | -2 < x ∧ x < 1}
def B := {x | 0 < x ∧ x < 3}
def unionSet := {x | -2 < x ∧ x < 3}

theorem union_correct : ( {x | -2 < x ∧ x < 1} ∪ {x | 0 < x ∧ x < 3} ) = {x | -2 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_GPT_union_correct_l2110_211070
