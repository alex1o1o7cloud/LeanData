import Mathlib

namespace NUMINAMATH_GPT_elderly_in_sample_l2193_219336

variable (A E M : ℕ)
variable (total_employees : ℕ)
variable (total_young : ℕ)
variable (sample_size_young : ℕ)
variable (sampling_ratio : ℚ)
variable (sample_elderly : ℕ)

axiom condition_1 : total_young = 160
axiom condition_2 : total_employees = 430
axiom condition_3 : M = 2 * E
axiom condition_4 : A + M + E = total_employees
axiom condition_5 : sampling_ratio = sample_size_young / total_young
axiom sampling : sample_size_young = 32
axiom elderly_employees : sample_elderly = 18

theorem elderly_in_sample : sample_elderly = sampling_ratio * E := by
  -- Proof steps are not provided
  sorry

end NUMINAMATH_GPT_elderly_in_sample_l2193_219336


namespace NUMINAMATH_GPT_race_prob_l2193_219390

theorem race_prob :
  let pX := (1 : ℝ) / 8
  let pY := (1 : ℝ) / 12
  let pZ := (1 : ℝ) / 6
  pX + pY + pZ = (3 : ℝ) / 8 :=
by
  sorry

end NUMINAMATH_GPT_race_prob_l2193_219390


namespace NUMINAMATH_GPT_larger_number_is_299_l2193_219384

theorem larger_number_is_299 {a b : ℕ} (hcf : Nat.gcd a b = 23) (lcm_factors : ∃ k1 k2 : ℕ, Nat.lcm a b = 23 * k1 * k2 ∧ k1 = 12 ∧ k2 = 13) :
  max a b = 299 :=
by
  sorry

end NUMINAMATH_GPT_larger_number_is_299_l2193_219384


namespace NUMINAMATH_GPT_no_three_natural_numbers_l2193_219383

theorem no_three_natural_numbers (a b c : ℕ) (h1 : a > 1) (h2 : b > 1) (h3 : c > 1)
    (h4 : b ∣ a^2 - 1) (h5 : a ∣ c^2 - 1) (h6 : b ∣ c^2 - 1) : false :=
by
  sorry

end NUMINAMATH_GPT_no_three_natural_numbers_l2193_219383


namespace NUMINAMATH_GPT_mike_passing_percentage_l2193_219343

theorem mike_passing_percentage :
  ∀ (score shortfall max_marks : ℕ), 
    score = 212 ∧ shortfall = 25 ∧ max_marks = 790 →
    (score + shortfall) / max_marks * 100 = 30 :=
by
  intros score shortfall max_marks h
  have h1 : score = 212 := h.1
  have h2 : shortfall = 25 := h.2.1
  have h3 : max_marks = 790 := h.2.2
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_mike_passing_percentage_l2193_219343


namespace NUMINAMATH_GPT_second_month_sale_l2193_219367

theorem second_month_sale (S : ℝ) :
  (S + 5420 + 6200 + 6350 + 6500 = 30000) → S = 5530 :=
by
  sorry

end NUMINAMATH_GPT_second_month_sale_l2193_219367


namespace NUMINAMATH_GPT_power_sum_greater_than_linear_l2193_219378

theorem power_sum_greater_than_linear (x : ℝ) (n : ℕ) (hx1 : x > -1) (hx2 : x ≠ 0) (hn : n ≥ 2) :
  (1 + x) ^ n > 1 + n * x :=
sorry

end NUMINAMATH_GPT_power_sum_greater_than_linear_l2193_219378


namespace NUMINAMATH_GPT_machines_complete_job_in_12_days_l2193_219341

-- Given the conditions
variable (D : ℕ) -- The number of days for 12 machines to complete the job
variable (h1 : (1 : ℚ) / ((12 : ℚ) * D) = (1 : ℚ) / ((18 : ℚ) * 8))

-- Prove the number of days for 12 machines to complete the job
theorem machines_complete_job_in_12_days (h1 : (1 : ℚ) / ((12 : ℚ) * D) = (1 : ℚ) / ((18 : ℚ) * 8)) : D = 12 :=
by
  sorry

end NUMINAMATH_GPT_machines_complete_job_in_12_days_l2193_219341


namespace NUMINAMATH_GPT_larger_triangle_perimeter_l2193_219305

theorem larger_triangle_perimeter 
    (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) 
    (h1 : a = 6) (h2 : b = 8)
    (hypo_large : ∀ c : ℝ, c = 20) : 
    (2 * a + 2 * b + 20 = 48) :=
by {
  sorry
}

end NUMINAMATH_GPT_larger_triangle_perimeter_l2193_219305


namespace NUMINAMATH_GPT_rectangle_area_is_correct_l2193_219386

-- Define the conditions
def length : ℕ := 135
def breadth (l : ℕ) : ℕ := l / 3

-- Define the area of the rectangle
def area (l b : ℕ) : ℕ := l * b

-- The statement to prove
theorem rectangle_area_is_correct : area length (breadth length) = 6075 := by
  -- Proof goes here, this is just the statement
  sorry

end NUMINAMATH_GPT_rectangle_area_is_correct_l2193_219386


namespace NUMINAMATH_GPT_a_2n_is_square_l2193_219317

def a_n (n : ℕ) : ℕ := 
  if n = 0 then 0 
  else if n = 1 then 1
  else if n = 2 then 1
  else if n = 3 then 2
  else a_n (n - 1) + a_n (n - 3) + a_n (n - 4)

theorem a_2n_is_square (n : ℕ) : ∃ k : ℕ, a_n (2 * n) = k * k := by
  sorry

end NUMINAMATH_GPT_a_2n_is_square_l2193_219317


namespace NUMINAMATH_GPT_solve_system_equations_l2193_219345

noncomputable def system_equations : Prop :=
  ∃ x y : ℝ,
    (8 * x^2 - 26 * x * y + 15 * y^2 + 116 * x - 150 * y + 360 = 0) ∧
    (8 * x^2 + 18 * x * y - 18 * y^2 + 60 * x + 45 * y + 108 = 0) ∧
    ((x = 0 ∧ y = 4) ∨ (x = -7.5 ∧ y = 1) ∨ (x = -4.5 ∧ y = 0))

theorem solve_system_equations : system_equations := 
by
  sorry

end NUMINAMATH_GPT_solve_system_equations_l2193_219345


namespace NUMINAMATH_GPT_a_minus_b_eq_zero_l2193_219380

-- Definitions from the conditions
def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^2 + a * x + b
def tangent_line (x y : ℝ) : Prop := x - y + 1 = 0

-- The point (0, b)
def point_b (b : ℝ) : (ℝ × ℝ) := (0, b)

-- Slope condition at point (0, b)
def slope_of_f_at_0 (a : ℝ) : ℝ := a
def slope_of_tangent_line : ℝ := 1

-- Prove a - b = 0 given the conditions
theorem a_minus_b_eq_zero (a b : ℝ) 
    (h1 : f 0 a b = b)
    (h2 : tangent_line 0 b) 
    (h3 : slope_of_f_at_0 a = slope_of_tangent_line) : a - b = 0 :=
by
  sorry

end NUMINAMATH_GPT_a_minus_b_eq_zero_l2193_219380


namespace NUMINAMATH_GPT_correct_equation_l2193_219346

-- Define conditions as variables in Lean
def cost_price (x : ℝ) : Prop := x > 0
def markup_percentage : ℝ := 0.40
def discount_percentage : ℝ := 0.80
def selling_price : ℝ := 240

-- Define the theorem
theorem correct_equation (x : ℝ) (hx : cost_price x) :
  x * (1 + markup_percentage) * discount_percentage = selling_price :=
by
  sorry

end NUMINAMATH_GPT_correct_equation_l2193_219346


namespace NUMINAMATH_GPT_seven_digit_number_subtraction_l2193_219360

theorem seven_digit_number_subtraction 
  (n : ℕ)
  (d1 d2 d3 d4 d5 d6 d7 : ℕ)
  (h1 : n = d1 * 10^6 + d2 * 10^5 + d3 * 10^4 + d4 * 10^3 + d5 * 10^2 + d6 * 10 + d7)
  (h2 : d1 < 10 ∧ d2 < 10 ∧ d3 < 10 ∧ d4 < 10 ∧ d5 < 10 ∧ d6 < 10 ∧ d7 < 10)
  (h3 : n - (d1 + d3 + d4 + d5 + d6 + d7) = 9875352) :
  n - (d1 + d3 + d4 + d5 + d6 + d7 - d2) = 9875357 :=
sorry

end NUMINAMATH_GPT_seven_digit_number_subtraction_l2193_219360


namespace NUMINAMATH_GPT_least_possible_product_of_primes_l2193_219364

-- Define a prime predicate for a number greater than 20
def is_prime_over_20 (p : Nat) : Prop := Nat.Prime p ∧ p > 20

-- Define the two primes
def prime1 := 23
def prime2 := 29

-- Given the conditions, prove the least possible product of two distinct primes greater than 20 is 667
theorem least_possible_product_of_primes :
  ∃ p1 p2 : Nat, is_prime_over_20 p1 ∧ is_prime_over_20 p2 ∧ p1 ≠ p2 ∧ (p1 * p2 = 667) :=
by
  -- Theorem statement without proof
  existsi (prime1)
  existsi (prime2)
  have h1 : is_prime_over_20 prime1 := by sorry
  have h2 : is_prime_over_20 prime2 := by sorry
  have h3 : prime1 ≠ prime2 := by sorry
  have h4 : prime1 * prime2 = 667 := by sorry
  exact ⟨h1, h2, h3, h4⟩

end NUMINAMATH_GPT_least_possible_product_of_primes_l2193_219364


namespace NUMINAMATH_GPT_find_k_l2193_219387

theorem find_k (k : ℕ) (h1 : k > 0) (h2 : 15 * k^4 < 120) : k = 1 := 
  sorry

end NUMINAMATH_GPT_find_k_l2193_219387


namespace NUMINAMATH_GPT_decimal_to_base5_equiv_l2193_219356

def base5_representation (n : ℕ) : ℕ := -- Conversion function (implementation to be filled later)
  sorry

theorem decimal_to_base5_equiv : base5_representation 88 = 323 :=
by
  -- Proof steps go here.
  sorry

end NUMINAMATH_GPT_decimal_to_base5_equiv_l2193_219356


namespace NUMINAMATH_GPT_root_equivalence_l2193_219312

theorem root_equivalence (a_1 a_2 a_3 b : ℝ) :
  (∃ c_1 c_2 c_3 : ℝ, c_1 ≠ c_2 ∧ c_2 ≠ c_3 ∧ c_1 ≠ c_3 ∧
    (∀ x : ℝ, (x - a_1) * (x - a_2) * (x - a_3) = b ↔ (x = c_1 ∨ x = c_2 ∨ x = c_3))) →
  (∀ x : ℝ, (x + c_1) * (x + c_2) * (x + c_3) = b ↔ (x = -a_1 ∨ x = -a_2 ∨ x = -a_3)) :=
by 
  sorry

end NUMINAMATH_GPT_root_equivalence_l2193_219312


namespace NUMINAMATH_GPT_inequality_a3_minus_b3_l2193_219374

theorem inequality_a3_minus_b3 (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a < b) : a^3 - b^3 < 0 :=
by sorry

end NUMINAMATH_GPT_inequality_a3_minus_b3_l2193_219374


namespace NUMINAMATH_GPT_washing_machine_capacity_l2193_219303

theorem washing_machine_capacity 
  (shirts : ℕ) (sweaters : ℕ) (loads : ℕ) (total_clothing : ℕ) (n : ℕ)
  (h1 : shirts = 43) (h2 : sweaters = 2) (h3 : loads = 9)
  (h4 : total_clothing = shirts + sweaters)
  (h5 : total_clothing / loads = n) :
  n = 5 :=
sorry

end NUMINAMATH_GPT_washing_machine_capacity_l2193_219303


namespace NUMINAMATH_GPT_part_I_part_II_l2193_219379

noncomputable def f (x : ℝ) : ℝ := 5 + Real.log x
noncomputable def g (x k : ℝ) : ℝ := k * x / (x + 1)

theorem part_I (k : ℝ) : 
  (∃ x0, g x0 k = x0 + 4 ∧ (k / (x0 + 1)^2) = 1) ↔ (k = 1 ∨ k = 9) :=
by
  sorry

theorem part_II (k : ℕ) : (∀ x : ℝ, 1 < x → f x > g x k) → k ≤ 7 :=
by
  sorry

end NUMINAMATH_GPT_part_I_part_II_l2193_219379


namespace NUMINAMATH_GPT_train_pass_jogger_in_41_seconds_l2193_219316

-- Definitions based on conditions
def jogger_speed_kmh := 9 -- in km/hr
def train_speed_kmh := 45 -- in km/hr
def initial_distance_jogger := 200 -- in meters
def train_length := 210 -- in meters

-- Converting speeds from km/hr to m/s
def kmh_to_ms (kmh: ℕ) : ℕ := (kmh * 1000) / 3600

def jogger_speed_ms := kmh_to_ms jogger_speed_kmh -- in m/s
def train_speed_ms := kmh_to_ms train_speed_kmh -- in m/s

-- Relative speed of the train with respect to the jogger
def relative_speed := train_speed_ms - jogger_speed_ms -- in m/s

-- Total distance to be covered by the train to pass the jogger
def total_distance := initial_distance_jogger + train_length -- in meters

-- Time taken to pass the jogger
def time_to_pass (distance : ℕ) (speed : ℕ) : ℕ := distance / speed

theorem train_pass_jogger_in_41_seconds : time_to_pass total_distance relative_speed = 41 :=
by
  sorry

end NUMINAMATH_GPT_train_pass_jogger_in_41_seconds_l2193_219316


namespace NUMINAMATH_GPT_Jerry_weekly_earnings_l2193_219368

def hours_per_task : ℕ := 2
def pay_per_task : ℕ := 40
def hours_per_day : ℕ := 10
def days_per_week : ℕ := 7

theorem Jerry_weekly_earnings : pay_per_task * (hours_per_day / hours_per_task) * days_per_week = 1400 :=
by
  -- Carry out the proof here
  sorry

end NUMINAMATH_GPT_Jerry_weekly_earnings_l2193_219368


namespace NUMINAMATH_GPT_find_b_l2193_219344

theorem find_b 
  (a b c x : ℝ)
  (h : (3 * x^2 - 4 * x + 5 / 2) * (a * x^2 + b * x + c) 
       = 6 * x^4 - 17 * x^3 + 11 * x^2 - 7 / 2 * x + 5 / 3) 
  (ha : 3 * a = 6) : b = -3 := 
by 
  sorry

end NUMINAMATH_GPT_find_b_l2193_219344


namespace NUMINAMATH_GPT_spending_record_l2193_219391

-- Definitions based on conditions
def deposit_record (x : ℤ) : ℤ := x
def spend_record (x : ℤ) : ℤ := -x

-- Theorem statement
theorem spending_record (x : ℤ) (hx : x = 500) : spend_record x = -500 := by
  sorry

end NUMINAMATH_GPT_spending_record_l2193_219391


namespace NUMINAMATH_GPT_paint_needed_l2193_219382

-- Definitions from conditions
def total_needed_paint := 70
def initial_paint := 36
def bought_paint := 23

-- The main statement to prove
theorem paint_needed : total_needed_paint - (initial_paint + bought_ppaint) = 11 :=
by
  -- Definitions are already imported and stated
  -- Just need to refer these to the theorem assertion correctly
  sorry

end NUMINAMATH_GPT_paint_needed_l2193_219382


namespace NUMINAMATH_GPT_abs_ineq_solution_set_l2193_219366

theorem abs_ineq_solution_set (x : ℝ) : |x + 1| - |x - 5| < 4 ↔ x < 4 :=
sorry

end NUMINAMATH_GPT_abs_ineq_solution_set_l2193_219366


namespace NUMINAMATH_GPT_count_integers_log_condition_l2193_219350

theorem count_integers_log_condition :
  (∃! n : ℕ, n = 54 ∧ (∀ x : ℕ, x > 30 ∧ x < 90 ∧ ((x - 30) * (90 - x) < 1000) ↔ (31 <= x ∧ x <= 84))) :=
sorry

end NUMINAMATH_GPT_count_integers_log_condition_l2193_219350


namespace NUMINAMATH_GPT_part_I_solution_set_part_II_min_value_l2193_219399

-- Define the function f(x)
def f (x : ℝ) : ℝ := x + 1 + |3 - x|

-- Prove the solution set of the inequality f(x) ≤ 6 for x ≥ -1 is -1 ≤ x ≤ 4
theorem part_I_solution_set (x : ℝ) (h1 : x ≥ -1) : f x ≤ 6 ↔ (-1 ≤ x ∧ x ≤ 4) :=
by
  sorry

-- Define the condition for the minimum value of f(x)
def min_f := 4

-- Prove the minimum value of 2a + b under the given constraints
theorem part_II_min_value (a b : ℝ) (h2 : a > 0 ∧ b > 0) (h3 : 8 * a * b = a + 2 * b) : 2 * a + b ≥ 9 / 8 :=
by
  sorry

end NUMINAMATH_GPT_part_I_solution_set_part_II_min_value_l2193_219399


namespace NUMINAMATH_GPT_last_digit_one_over_three_pow_neg_ten_l2193_219332

theorem last_digit_one_over_three_pow_neg_ten : (3^10) % 10 = 9 := by
  sorry

end NUMINAMATH_GPT_last_digit_one_over_three_pow_neg_ten_l2193_219332


namespace NUMINAMATH_GPT_combined_perimeter_l2193_219301

theorem combined_perimeter (side_square : ℝ) (a b c : ℝ) (diameter : ℝ) 
  (h_square : side_square = 7) 
  (h_triangle : a = 5 ∧ b = 6 ∧ c = 7) 
  (h_diameter : diameter = 4) : 
  4 * side_square + (a + b + c) + (2 * Real.pi * (diameter / 2) + diameter) = 50 + 2 * Real.pi := 
by 
  sorry

end NUMINAMATH_GPT_combined_perimeter_l2193_219301


namespace NUMINAMATH_GPT_lasso_success_probability_l2193_219375

-- Let p be the probability of successfully placing a lasso in a single throw
def p := 1 / 2

-- Let q be the probability of failure in a single throw
def q := 1 - p

-- Let n be the number of attempts
def n := 4

-- The probability of failing all n times
def probFailAll := q ^ n

-- The probability of succeeding at least once
def probSuccessAtLeastOnce := 1 - probFailAll

-- Theorem statement
theorem lasso_success_probability : probSuccessAtLeastOnce = 15 / 16 := by
  sorry

end NUMINAMATH_GPT_lasso_success_probability_l2193_219375


namespace NUMINAMATH_GPT_speed_of_rest_distance_l2193_219306

theorem speed_of_rest_distance (D V : ℝ) (h1 : D = 26.67)
                                (h2 : (D / 2) / 5 + (D / 2) / V = 6) : 
  V = 20 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_rest_distance_l2193_219306


namespace NUMINAMATH_GPT_min_wins_required_l2193_219361

theorem min_wins_required 
  (total_matches initial_matches remaining_matches : ℕ)
  (points_for_win points_for_draw points_for_defeat current_points target_points : ℕ)
  (matches_played_points : ℕ)
  (h_total : total_matches = 20)
  (h_initial : initial_matches = 5)
  (h_remaining : remaining_matches = total_matches - initial_matches)
  (h_win_points : points_for_win = 3)
  (h_draw_points : points_for_draw = 1)
  (h_defeat_points : points_for_defeat = 0)
  (h_current_points : current_points = 8)
  (h_target_points : target_points = 40)
  (h_matches_played_points : matches_played_points = current_points)
  :
  (∃ min_wins : ℕ, min_wins * points_for_win + (remaining_matches - min_wins) * points_for_defeat >= target_points - matches_played_points ∧ min_wins ≤ remaining_matches) ∧
  (∀ other_wins : ℕ, other_wins < min_wins → (other_wins * points_for_win + (remaining_matches - other_wins) * points_for_defeat < target_points - matches_played_points)) :=
sorry

end NUMINAMATH_GPT_min_wins_required_l2193_219361


namespace NUMINAMATH_GPT_minimum_value_l2193_219377

theorem minimum_value (x : ℝ) (hx : 0 ≤ x) : ∃ y : ℝ, y = x^2 - 6 * x + 8 ∧ (∀ t : ℝ, 0 ≤ t → y ≤ t^2 - 6 * t + 8) :=
sorry

end NUMINAMATH_GPT_minimum_value_l2193_219377


namespace NUMINAMATH_GPT_boxcar_capacity_ratio_l2193_219324

-- The known conditions translated into Lean definitions
def red_boxcar_capacity (B : ℕ) : ℕ := 3 * B
def blue_boxcar_count : ℕ := 4
def red_boxcar_count : ℕ := 3
def black_boxcar_count : ℕ := 7
def black_boxcar_capacity : ℕ := 4000
def total_capacity : ℕ := 132000

-- The mathematical condition as a Lean theorem statement.
theorem boxcar_capacity_ratio 
  (B : ℕ)
  (h_condition : (red_boxcar_count * red_boxcar_capacity B + 
                  blue_boxcar_count * B + 
                  black_boxcar_count * black_boxcar_capacity = 
                  total_capacity)) : 
  black_boxcar_capacity / B = 1 / 2 := 
sorry

end NUMINAMATH_GPT_boxcar_capacity_ratio_l2193_219324


namespace NUMINAMATH_GPT_construct_rhombus_l2193_219319

-- Define data structure representing a point in a 2-dimensional Euclidean space.
structure Point where
  x : ℝ
  y : ℝ

-- Define what it means for four points to form a rhombus.
def isRhombus (A B C D : Point) : Prop :=
  (A.x - B.x) ^ 2 + (A.y - B.y) ^ 2 = (B.x - C.x) ^ 2 + (B.y - C.y) ^ 2 ∧
  (B.x - C.x) ^ 2 + (B.y - C.y) ^ 2 = (C.x - D.x) ^ 2 + (C.y - D.y) ^ 2 ∧
  (C.x - D.x) ^ 2 + (C.y - D.y) ^ 2 = (D.x - A.x) ^ 2 + (D.y - A.y) ^ 2

-- Define circumradius condition for triangle ABC
def circumradius (A B C : Point) (R : ℝ) : Prop := sorry -- Detailed definition would be added here.

-- Define inradius condition for triangle BCD
def inradius (B C D : Point) (r : ℝ) : Prop := sorry -- Detailed definition would be added here.

-- The proposition to be proved: We can construct the rhombus ABCD given R and r.
theorem construct_rhombus (A B C D : Point) (R r : ℝ) :
  (circumradius A B C R) →
  (inradius B C D r) →
  isRhombus A B C D :=
by
  sorry

end NUMINAMATH_GPT_construct_rhombus_l2193_219319


namespace NUMINAMATH_GPT_tire_price_l2193_219370

-- Definitions based on given conditions
def tire_cost (T : ℝ) (n : ℕ) : Prop :=
  n * T + 56 = 224

-- The equivalence we want to prove
theorem tire_price (T : ℝ) (n : ℕ) (h : tire_cost T n) : n * T = 168 :=
by
  sorry

end NUMINAMATH_GPT_tire_price_l2193_219370


namespace NUMINAMATH_GPT_median_and_mode_l2193_219313

theorem median_and_mode (data : List ℝ) (h : data = [6, 7, 4, 7, 5, 2]) :
  ∃ median mode, median = 5.5 ∧ mode = 7 := 
by {
  sorry
}

end NUMINAMATH_GPT_median_and_mode_l2193_219313


namespace NUMINAMATH_GPT_inequality_proof_l2193_219385

theorem inequality_proof {x y z : ℝ} (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hsum : x + y + z = 1) :
    (2 * x^2 / (y + z)) + (2 * y^2 / (z + x)) + (2 * z^2 / (x + y)) ≥ 1 := sorry

end NUMINAMATH_GPT_inequality_proof_l2193_219385


namespace NUMINAMATH_GPT_number_of_terms_in_arithmetic_sequence_l2193_219333

-- Define the first term, common difference, and the nth term of the sequence
def a : ℤ := -3
def d : ℤ := 4
def a_n : ℤ := 45

-- Define the number of terms in the arithmetic sequence
def num_of_terms : ℤ := 13

-- The theorem states that for the given arithmetic sequence, the number of terms n satisfies the sequence equation
theorem number_of_terms_in_arithmetic_sequence :
  a + (num_of_terms - 1) * d = a_n :=
by
  sorry

end NUMINAMATH_GPT_number_of_terms_in_arithmetic_sequence_l2193_219333


namespace NUMINAMATH_GPT_problem_solution_l2193_219329

noncomputable def root1 : ℝ := (3 + Real.sqrt 105) / 4
noncomputable def root2 : ℝ := (3 - Real.sqrt 105) / 4

theorem problem_solution :
  (∀ x : ℝ, x ≠ -2 → x ≠ -3 → (x^3 - x^2 - 4 * x) / (x^2 + 5 * x + 6) + x = -4
    → x = root1 ∨ x = root2) := 
by
  sorry

end NUMINAMATH_GPT_problem_solution_l2193_219329


namespace NUMINAMATH_GPT_solve_modulo_problem_l2193_219389

theorem solve_modulo_problem (n : ℤ) :
  0 ≤ n ∧ n < 19 ∧ 38574 % 19 = n % 19 → n = 4 := by
  sorry

end NUMINAMATH_GPT_solve_modulo_problem_l2193_219389


namespace NUMINAMATH_GPT_sum_of_possible_values_l2193_219331

theorem sum_of_possible_values (N : ℝ) (h : N * (N - 10) = -7) :
  ∃ N1 N2 : ℝ, (N1 * (N1 - 10) = -7 ∧ N2 * (N2 - 10) = -7) ∧ (N1 + N2 = 10) :=
sorry

end NUMINAMATH_GPT_sum_of_possible_values_l2193_219331


namespace NUMINAMATH_GPT_intersecting_lines_l2193_219308

theorem intersecting_lines (n c : ℝ) 
  (h1 : (15 : ℝ) = n * 5 + 5)
  (h2 : (15 : ℝ) = 4 * 5 + c) : 
  c + n = -3 := 
by
  sorry

end NUMINAMATH_GPT_intersecting_lines_l2193_219308


namespace NUMINAMATH_GPT_factorize_l2193_219321

variables (a b x y : ℝ)

theorem factorize : (a * x - b * y)^2 + (a * y + b * x)^2 = (x^2 + y^2) * (a^2 + b^2) :=
by
  sorry

end NUMINAMATH_GPT_factorize_l2193_219321


namespace NUMINAMATH_GPT_fraction_value_unchanged_l2193_219307

theorem fraction_value_unchanged (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (x / (x + y) = (2 * x) / (2 * (x + y))) :=
by sorry

end NUMINAMATH_GPT_fraction_value_unchanged_l2193_219307


namespace NUMINAMATH_GPT_sandy_paint_area_l2193_219323

-- Define the dimensions of the wall
def wall_height : ℕ := 10
def wall_length : ℕ := 15

-- Define the dimensions of the decorative region
def deco_height : ℕ := 3
def deco_length : ℕ := 5

-- Calculate the areas and prove the required area to paint
theorem sandy_paint_area :
  wall_height * wall_length - deco_height * deco_length = 135 := by
  sorry

end NUMINAMATH_GPT_sandy_paint_area_l2193_219323


namespace NUMINAMATH_GPT_sin_inequality_l2193_219355

theorem sin_inequality (x : ℝ) (hx1 : 0 < x) (hx2 : x < Real.pi / 4) : 
  Real.sin (Real.sin x) < Real.sin x ∧ Real.sin x < Real.sin (Real.tan x) :=
by 
  sorry

end NUMINAMATH_GPT_sin_inequality_l2193_219355


namespace NUMINAMATH_GPT_find_special_numbers_l2193_219371

/-- Define the sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

/-- Define the main statement to be proved -/
theorem find_special_numbers :
  { n : ℕ | sum_of_digits n * (sum_of_digits n - 1) = n - 1 } = {1, 13, 43, 91, 157} :=
by
  sorry

end NUMINAMATH_GPT_find_special_numbers_l2193_219371


namespace NUMINAMATH_GPT_amount_of_money_l2193_219354

variable (x : ℝ)

-- Conditions
def condition1 : Prop := x < 2000
def condition2 : Prop := 4 * x > 2000
def condition3 : Prop := 4 * x - 2000 = 2000 - x

theorem amount_of_money (h1 : condition1 x) (h2 : condition2 x) (h3 : condition3 x) : x = 800 :=
by
  sorry

end NUMINAMATH_GPT_amount_of_money_l2193_219354


namespace NUMINAMATH_GPT_blueberries_count_l2193_219393

theorem blueberries_count
  (initial_apples : ℕ)
  (initial_oranges : ℕ)
  (initial_blueberries : ℕ)
  (apples_eaten : ℕ)
  (oranges_eaten : ℕ)
  (remaining_fruits : ℕ)
  (h1 : initial_apples = 14)
  (h2 : initial_oranges = 9)
  (h3 : apples_eaten = 1)
  (h4 : oranges_eaten = 1)
  (h5 : remaining_fruits = 26) :
  initial_blueberries = 5 := 
by
  sorry

end NUMINAMATH_GPT_blueberries_count_l2193_219393


namespace NUMINAMATH_GPT_extreme_values_for_f_when_a_is_one_number_of_zeros_of_h_l2193_219357

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x / Real.exp x + x^2 / 2 - x
noncomputable def g (x : ℝ) : ℝ := Real.log x
noncomputable def h (a : ℝ) (x : ℝ) : ℝ := max (f a x) (g x)

theorem extreme_values_for_f_when_a_is_one :
  (∀ x : ℝ, (f 1 x) ≤ 0) ∧ f 1 0 = 0 ∧ f 1 1 = (1 / Real.exp 1) - 1 / 2 :=
sorry

theorem number_of_zeros_of_h (a : ℝ) :
  (0 ≤ a → 
   if 1 < a ∧ a < Real.exp 1 / 2 then
     ∃ x1 x2 : ℝ, 0 < x1 ∧ x1 < 1 ∧ 0 < x2 ∧ x2 < 1 ∧ h a x1 = 0 ∧ h a x2 = 0
   else if 0 ≤ a ∧ a ≤ 1 ∨ a = Real.exp 1 / 2 then
     ∃ x : ℝ, 0 < x ∧ x < 1 ∧ h a x = 0
   else
     ∀ x : ℝ, x > 0 → h a x ≠ 0) :=
sorry

end NUMINAMATH_GPT_extreme_values_for_f_when_a_is_one_number_of_zeros_of_h_l2193_219357


namespace NUMINAMATH_GPT_future_cup_defensive_analysis_l2193_219315

variables (avg_A : ℝ) (std_dev_A : ℝ) (avg_B : ℝ) (std_dev_B : ℝ)

-- Statement translations:
-- A: On average, Class B has better defensive skills than Class A.
def stat_A : Prop := avg_B < avg_A

-- C: Class B sometimes performs very well in defense, while other times it performs relatively poorly.
def stat_C : Prop := std_dev_B > std_dev_A

-- D: Class A rarely concedes goals.
def stat_D : Prop := avg_A <= 1.9 -- It's implied that 'rarely' indicates consistency and a lower average threshold, so this represents that.

theorem future_cup_defensive_analysis (h_avg_A : avg_A = 1.9) (h_std_dev_A : std_dev_A = 0.3) 
  (h_avg_B : avg_B = 1.3) (h_std_dev_B : std_dev_B = 1.2) :
  stat_A avg_A avg_B ∧ stat_C std_dev_A std_dev_B ∧ stat_D avg_A :=
by {
  -- Proof is omitted as per instructions
  sorry
}

end NUMINAMATH_GPT_future_cup_defensive_analysis_l2193_219315


namespace NUMINAMATH_GPT_shirt_wallet_ratio_l2193_219338

theorem shirt_wallet_ratio
  (F W S : ℕ)
  (hF : F = 30)
  (hW : W = F + 60)
  (h_total : S + W + F = 150) :
  S / W = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_shirt_wallet_ratio_l2193_219338


namespace NUMINAMATH_GPT_sqrt_12_eq_2_sqrt_3_sqrt_1_div_2_eq_sqrt_2_div_2_l2193_219358

theorem sqrt_12_eq_2_sqrt_3 : Real.sqrt 12 = 2 * Real.sqrt 3 := sorry

theorem sqrt_1_div_2_eq_sqrt_2_div_2 : Real.sqrt (1 / 2) = Real.sqrt 2 / 2 := sorry

end NUMINAMATH_GPT_sqrt_12_eq_2_sqrt_3_sqrt_1_div_2_eq_sqrt_2_div_2_l2193_219358


namespace NUMINAMATH_GPT_inequality_one_inequality_two_l2193_219381

-- First Inequality Problem
theorem inequality_one (a b c d : ℝ) (h : a + b + c + d = 3) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (1 / a^2) + (1 / b^2) + (1 / c^2) + (1 / d^2) ≤ (1 / (a^2 * b^2 * c^2 * d^2)) :=
sorry

-- Second Inequality Problem
theorem inequality_two (a b c d : ℝ) (h : a + b + c + d = 3) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (1 / a^3) + (1 / b^3) + (1 / c^3) + (1 / d^3) ≤ (1 / (a^3 * b^3 * c^3 * d^3)) :=
sorry

end NUMINAMATH_GPT_inequality_one_inequality_two_l2193_219381


namespace NUMINAMATH_GPT_paid_more_than_free_l2193_219351

def num_men : ℕ := 194
def num_women : ℕ := 235
def free_admission : ℕ := 68
def total_people (num_men num_women : ℕ) : ℕ := num_men + num_women
def paid_admission (total_people free_admission : ℕ) : ℕ := total_people - free_admission
def paid_over_free (paid_admission free_admission : ℕ) : ℕ := paid_admission - free_admission

theorem paid_more_than_free :
  paid_over_free (paid_admission (total_people num_men num_women) free_admission) free_admission = 293 := 
by
  sorry

end NUMINAMATH_GPT_paid_more_than_free_l2193_219351


namespace NUMINAMATH_GPT_hexagon_theorem_l2193_219376

-- Define a structure for the hexagon with its sides
structure Hexagon :=
(side1 side2 side3 side4 side5 side6 : ℕ)

-- Define the conditions of the problem
def hexagon_conditions (h : Hexagon) : Prop :=
  h.side1 = 5 ∧ h.side2 = 6 ∧ h.side3 = 7 ∧
  (h.side1 + h.side2 + h.side3 + h.side4 + h.side5 + h.side6 = 38)

-- Define the proposition that we need to prove
def hexagon_proposition (h : Hexagon) : Prop :=
  (h.side3 = 7 ∨ h.side4 = 7 ∨ h.side5 = 7 ∨ h.side6 = 7) → 
  (h.side1 = 5 ∧ h.side2 = 6 ∧ h.side3 = 7 ∧ h.side4 = 7 ∧ h.side5 = 7 ∧ h.side6 = 7 → 3 = 3)

-- The proof statement combining conditions and the to-be-proven proposition
theorem hexagon_theorem (h : Hexagon) (hc : hexagon_conditions h) : hexagon_proposition h :=
by
  sorry -- No proof is required

end NUMINAMATH_GPT_hexagon_theorem_l2193_219376


namespace NUMINAMATH_GPT_solution_pairs_l2193_219396

theorem solution_pairs (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b)
    (h_coprime: Nat.gcd (2 * a - 1) (2 * b + 1) = 1) 
    (h_divides : (a + b) ∣ (4 * a * b + 1)) :
    ∃ n : ℕ, a = n ∧ b = n + 1 :=
by
  -- statement
  sorry

end NUMINAMATH_GPT_solution_pairs_l2193_219396


namespace NUMINAMATH_GPT_triangle_perimeter_l2193_219342

theorem triangle_perimeter {a b c : ℕ} (ha : a = 10) (hb : b = 6) (hc : c = 7) :
    a + b + c = 23 := by
  sorry

end NUMINAMATH_GPT_triangle_perimeter_l2193_219342


namespace NUMINAMATH_GPT_fraction_of_married_men_is_two_fifths_l2193_219365

noncomputable def fraction_of_married_men (W : ℕ) (p : ℚ) (h : p = 1 / 3) : ℚ :=
  let W_s := p * W
  let W_m := W - W_s
  let M_m := W_m
  let T := W + M_m
  M_m / T

theorem fraction_of_married_men_is_two_fifths (W : ℕ) (p : ℚ) (h : p = 1 / 3) (hW : W = 6) : fraction_of_married_men W p h = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_married_men_is_two_fifths_l2193_219365


namespace NUMINAMATH_GPT_most_stable_student_l2193_219373

-- Define the variances for the four students
def variance_A (SA2 : ℝ) : Prop := SA2 = 0.15
def variance_B (SB2 : ℝ) : Prop := SB2 = 0.32
def variance_C (SC2 : ℝ) : Prop := SC2 = 0.5
def variance_D (SD2 : ℝ) : Prop := SD2 = 0.25

-- Theorem proving that the most stable student is A
theorem most_stable_student {SA2 SB2 SC2 SD2 : ℝ} 
  (hA : variance_A SA2) 
  (hB : variance_B SB2)
  (hC : variance_C SC2)
  (hD : variance_D SD2) : 
  SA2 < SB2 ∧ SA2 < SC2 ∧ SA2 < SD2 :=
by
  rw [variance_A, variance_B, variance_C, variance_D] at *
  sorry

end NUMINAMATH_GPT_most_stable_student_l2193_219373


namespace NUMINAMATH_GPT_chess_competition_players_l2193_219359

theorem chess_competition_players (J H : ℕ) (total_points : ℕ) (junior_points : ℕ) (high_school_points : ℕ → ℕ)
  (plays : ℕ → ℕ)
  (H_junior_points : junior_points = 8)
  (H_total_points : total_points = (J + H) * (J + H - 1) / 2)
  (H_total_points_contribution : total_points = junior_points + H * high_school_points H)
  (H_even_distribution : ∀ x : ℕ, 0 ≤ x ∧ x ≤ J → high_school_points H = x * (x - 1) / 2)
  (H_H_cases : H = 7 ∨ H = 9 ∨ H = 14) :
  H = 7 ∨ H = 14 :=
by
  have H_cases : H = 7 ∨ H = 14 :=
    by
      sorry
  exact H_cases

end NUMINAMATH_GPT_chess_competition_players_l2193_219359


namespace NUMINAMATH_GPT_min_value_of_polynomial_l2193_219304

theorem min_value_of_polynomial (a : ℝ) : 
  (∀ x : ℝ, (2 * x^3 - 3 * x^2 + a) ≥ 5) → a = 6 :=
by
  sorry   -- Proof omitted

end NUMINAMATH_GPT_min_value_of_polynomial_l2193_219304


namespace NUMINAMATH_GPT_power_function_not_origin_l2193_219348

theorem power_function_not_origin (m : ℝ) 
  (h1 : m^2 - 3 * m + 3 = 1) 
  (h2 : m^2 - m - 2 ≤ 0) : 
  m = 1 ∨ m = 2 :=
sorry

end NUMINAMATH_GPT_power_function_not_origin_l2193_219348


namespace NUMINAMATH_GPT_find_range_of_k_l2193_219326

-- Define the conditions and the theorem
def is_ellipse (k : ℝ) : Prop :=
  (3 + k > 0) ∧ (2 - k > 0) ∧ (3 + k ≠ 2 - k)

theorem find_range_of_k :
  {k : ℝ | is_ellipse k} = {k : ℝ | (-3 < k ∧ k < -1/2) ∨ (-1/2 < k ∧ k < 2)} :=
by
  sorry

end NUMINAMATH_GPT_find_range_of_k_l2193_219326


namespace NUMINAMATH_GPT_greatest_possible_perimeter_l2193_219372

def triangle_side_lengths (x : ℤ) : Prop :=
  (x > 0) ∧ (5 * x > 18) ∧ (x < 6)

def perimeter (x : ℤ) : ℤ :=
  x + 4 * x + 18

theorem greatest_possible_perimeter :
  ∃ x : ℤ, triangle_side_lengths x ∧ (perimeter x = 38) :=
by
  sorry

end NUMINAMATH_GPT_greatest_possible_perimeter_l2193_219372


namespace NUMINAMATH_GPT_problem1_problem2_l2193_219328

-- Problem 1: If a is parallel to b, then x = 4
theorem problem1 (x : ℝ) (u v : ℝ × ℝ) : 
  let a := (1, 1)
  let b := (4, x)
  (a.1 / b.1 = a.2 / b.2) → x = 4 := 
by 
  intros a b h
  dsimp [a, b] at h
  sorry

-- Problem 2: If (u - 2 * v) is perpendicular to (u + v), then x = -6
theorem problem2 (x : ℝ) (a u v : ℝ × ℝ) : 
  let a := (1, 1)
  let b := (4, x)
  let u := (a.1 + 2 * b.1, a.2 + 2 * b.2)
  let v := (2 * a.1 + b.1, 2 * a.2 + b.2)
  ((u.1 - 2 * v.1) * (u.1 + v.1) + (u.2 - 2 * v.2) * (u.2 + v.2) = 0) → x = -6 := 
by 
  intros a b u v h
  dsimp [a, b, u, v] at h
  sorry

end NUMINAMATH_GPT_problem1_problem2_l2193_219328


namespace NUMINAMATH_GPT_largest_is_B_l2193_219309

noncomputable def A := Real.sqrt (Real.sqrt (56 ^ (1 / 3)))
noncomputable def B := Real.sqrt (Real.sqrt (3584 ^ (1 / 3)))
noncomputable def C := Real.sqrt (Real.sqrt (2744 ^ (1 / 3)))
noncomputable def D := Real.sqrt (Real.sqrt (392 ^ (1 / 3)))
noncomputable def E := Real.sqrt (Real.sqrt (448 ^ (1 / 3)))

theorem largest_is_B : B > A ∧ B > C ∧ B > D ∧ B > E := by
  sorry

end NUMINAMATH_GPT_largest_is_B_l2193_219309


namespace NUMINAMATH_GPT_sequence_value_is_correct_l2193_219335

theorem sequence_value_is_correct (a : ℕ → ℤ) (h1 : a 1 = 1) (h2 : ∀ n : ℕ, n > 0 → a (n + 1) = a n + 2) : a 8 = 15 :=
sorry

end NUMINAMATH_GPT_sequence_value_is_correct_l2193_219335


namespace NUMINAMATH_GPT_quadratic_expression_value_l2193_219397

-- Given conditions
variables (a : ℝ) (h : 2 * a^2 + 3 * a - 2022 = 0)

-- Prove the main statement
theorem quadratic_expression_value :
  2 - 6 * a - 4 * a^2 = -4042 :=
sorry

end NUMINAMATH_GPT_quadratic_expression_value_l2193_219397


namespace NUMINAMATH_GPT_number_of_students_l2193_219314

theorem number_of_students (n S : ℕ) 
  (h1 : S = 15 * n) 
  (h2 : (S + 36) / (n + 1) = 16) : 
  n = 20 :=
by 
  sorry

end NUMINAMATH_GPT_number_of_students_l2193_219314


namespace NUMINAMATH_GPT_initial_people_in_line_l2193_219310

theorem initial_people_in_line (X : ℕ) 
  (h1 : X - 6 + 3 = 18) : X = 21 :=
  sorry

end NUMINAMATH_GPT_initial_people_in_line_l2193_219310


namespace NUMINAMATH_GPT_madhav_rank_from_last_is_15_l2193_219339

-- Defining the conditions
def class_size : ℕ := 31
def madhav_rank_from_start : ℕ := 17

-- Statement to be proved
theorem madhav_rank_from_last_is_15 :
  (class_size - madhav_rank_from_start + 1) = 15 := by
  sorry

end NUMINAMATH_GPT_madhav_rank_from_last_is_15_l2193_219339


namespace NUMINAMATH_GPT_sum_first_third_numbers_l2193_219398

theorem sum_first_third_numbers (A B C : ℕ)
    (h1 : A + B + C = 98)
    (h2 : A * 3 = B * 2)
    (h3 : B * 8 = C * 5)
    (h4 : B = 30) :
    A + C = 68 :=
by
-- Data is sufficient to conclude that A + C = 68
sorry

end NUMINAMATH_GPT_sum_first_third_numbers_l2193_219398


namespace NUMINAMATH_GPT_simplify_expression_l2193_219395

theorem simplify_expression (x : ℝ) : 
  (3 * x - 4) * (2 * x + 10) - (x + 3) * (3 * x - 2) = 3 * x^2 + 15 * x - 34 := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2193_219395


namespace NUMINAMATH_GPT_large_rectangle_perimeter_l2193_219325

-- Definitions from the conditions
def side_length_of_square (perimeter_square : ℕ) : ℕ := perimeter_square / 4
def width_of_small_rectangle (perimeter_rect : ℕ) (side_length : ℕ) : ℕ := (perimeter_rect / 2) - side_length

-- Given conditions
def perimeter_square := 24
def perimeter_rect := 16
def side_length := side_length_of_square perimeter_square
def rect_width := width_of_small_rectangle perimeter_rect side_length
def large_rectangle_height := side_length + rect_width
def large_rectangle_width := 3 * side_length

-- Perimeter calculation
def perimeter_large_rectangle (width height : ℕ) : ℕ := 2 * (width + height)

-- Proof problem statement
theorem large_rectangle_perimeter : 
  perimeter_large_rectangle large_rectangle_width large_rectangle_height = 52 :=
sorry

end NUMINAMATH_GPT_large_rectangle_perimeter_l2193_219325


namespace NUMINAMATH_GPT_train_speed_l2193_219340

def train_length : ℕ := 180
def crossing_time : ℕ := 12

theorem train_speed :
  train_length / crossing_time = 15 := sorry

end NUMINAMATH_GPT_train_speed_l2193_219340


namespace NUMINAMATH_GPT_electricity_fee_l2193_219337

theorem electricity_fee (a b : ℝ) : 
  let base_usage := 100
  let additional_usage := 160 - base_usage
  let base_cost := base_usage * a
  let additional_cost := additional_usage * b
  base_cost + additional_cost = 100 * a + 60 * b :=
by
  sorry

end NUMINAMATH_GPT_electricity_fee_l2193_219337


namespace NUMINAMATH_GPT_problem_statement_l2193_219353

theorem problem_statement (x y : ℝ) (h : x - 2 * y = -2) : 3 + 2 * x - 4 * y = -1 :=
  sorry

end NUMINAMATH_GPT_problem_statement_l2193_219353


namespace NUMINAMATH_GPT_range_of_a_l2193_219388

def proposition_p (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2 * x + a > 0

def proposition_q (a : ℝ) : Prop :=
  a - 1 > 1

theorem range_of_a (a : ℝ) :
  (proposition_p a ∨ proposition_q a) ∧ ¬ (proposition_p a ∧ proposition_q a) ↔ 1 < a ∧ a ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2193_219388


namespace NUMINAMATH_GPT_rhombus_area_l2193_219311

theorem rhombus_area (d1 d2 : ℝ) (h_d1 : d1 = 5) (h_d2 : d2 = 8) : 
  (1 / 2) * d1 * d2 = 20 :=
by
  sorry

end NUMINAMATH_GPT_rhombus_area_l2193_219311


namespace NUMINAMATH_GPT_daria_weeks_needed_l2193_219330

-- Defining the parameters and conditions
def initial_amount : ℕ := 20
def weekly_savings : ℕ := 10
def cost_of_vacuum_cleaner : ℕ := 120

-- Defining the total money Daria needs to add to her initial amount
def additional_amount_needed : ℕ := cost_of_vacuum_cleaner - initial_amount

-- Defining the number of weeks needed to save the additional amount, given weekly savings
def weeks_needed : ℕ := additional_amount_needed / weekly_savings

-- The theorem stating that Daria needs exactly 10 weeks to cover the expense of the vacuum cleaner
theorem daria_weeks_needed : weeks_needed = 10 := by
  sorry

end NUMINAMATH_GPT_daria_weeks_needed_l2193_219330


namespace NUMINAMATH_GPT_right_triangle_of_medians_l2193_219362

theorem right_triangle_of_medians
  (a b c m1 m2 m3 : ℝ)
  (h1 : 4 * m1^2 = 2 * (b^2 + c^2) - a^2)
  (h2 : 4 * m2^2 = 2 * (a^2 + c^2) - b^2)
  (h3 : 4 * m3^2 = 2 * (a^2 + b^2) - c^2)
  (h4 : m1^2 + m2^2 = 5 * m3^2) :
  c^2 = a^2 + b^2 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_of_medians_l2193_219362


namespace NUMINAMATH_GPT_abs_val_problem_l2193_219347

variable (a b : ℝ)

theorem abs_val_problem (h_abs_a : |a| = 2) (h_abs_b : |b| = 4) (h_sum_neg : a + b < 0) : a - b = 2 ∨ a - b = 6 :=
sorry

end NUMINAMATH_GPT_abs_val_problem_l2193_219347


namespace NUMINAMATH_GPT_cake_divided_into_equal_parts_l2193_219300

theorem cake_divided_into_equal_parts (cake_weight : ℕ) (pierre : ℕ) (nathalie : ℕ) (parts : ℕ) 
  (hw_eq : cake_weight = 400)
  (hp_eq : pierre = 100)
  (pn_eq : pierre = 2 * nathalie)
  (parts_eq : cake_weight / nathalie = parts)
  (hparts_eq : parts = 8) :
  cake_weight / nathalie = 8 := 
by
  sorry

end NUMINAMATH_GPT_cake_divided_into_equal_parts_l2193_219300


namespace NUMINAMATH_GPT_b_2018_eq_5043_l2193_219302

def b (n : Nat) : Nat :=
  if n % 2 = 1 then 5 * ((n + 1) / 2) - 3 else 5 * (n / 2) - 2

theorem b_2018_eq_5043 : b 2018 = 5043 := by
  sorry

end NUMINAMATH_GPT_b_2018_eq_5043_l2193_219302


namespace NUMINAMATH_GPT_number_of_pencil_boxes_l2193_219320

-- Define the total number of pencils and pencils per box as given conditions
def total_pencils : ℝ := 2592
def pencils_per_box : ℝ := 648.0

-- Problem statement: To prove the number of pencil boxes is 4
theorem number_of_pencil_boxes : total_pencils / pencils_per_box = 4 := by
  sorry

end NUMINAMATH_GPT_number_of_pencil_boxes_l2193_219320


namespace NUMINAMATH_GPT_expenditure_recording_l2193_219363

def income : ℕ := 200
def recorded_income : ℤ := 200
def expenditure (e : ℕ) : ℤ := -(e : ℤ)

theorem expenditure_recording (e : ℕ) :
  expenditure 150 = -150 := by
  sorry

end NUMINAMATH_GPT_expenditure_recording_l2193_219363


namespace NUMINAMATH_GPT_income_of_deceased_member_l2193_219369

theorem income_of_deceased_member
  (A B C : ℝ) -- Incomes of the three members
  (h1 : (A + B + C) / 3 = 735)
  (h2 : (A + B) / 2 = 650) :
  C = 905 :=
by
  sorry

end NUMINAMATH_GPT_income_of_deceased_member_l2193_219369


namespace NUMINAMATH_GPT_Laticia_knitted_socks_l2193_219318

theorem Laticia_knitted_socks (x : ℕ) (cond1 : x ≥ 0)
  (cond2 : ∃ y, y = x + 4)
  (cond3 : ∃ z, z = (x + (x + 4)) / 2)
  (cond4 : ∃ w, w = z - 3)
  (cond5 : x + (x + 4) + z + w = 57) : x = 13 := by
  sorry

end NUMINAMATH_GPT_Laticia_knitted_socks_l2193_219318


namespace NUMINAMATH_GPT_max_intersections_l2193_219349

-- Define the conditions
def num_points_x : ℕ := 15
def num_points_y : ℕ := 10

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Define the problem statement
theorem max_intersections (I : ℕ) :
  (15 : ℕ) == num_points_x →
  (10 : ℕ) == num_points_y →
  (I = binom 15 2 * binom 10 2) →
  I = 4725 := by
  -- We add sorry to skip the proof
  sorry

end NUMINAMATH_GPT_max_intersections_l2193_219349


namespace NUMINAMATH_GPT_candidate_majority_votes_l2193_219334

theorem candidate_majority_votes (total_votes : ℕ) (candidate_percentage other_percentage : ℕ) 
  (h_total_votes : total_votes = 5200)
  (h_candidate_percentage : candidate_percentage = 60)
  (h_other_percentage : other_percentage = 40) :
  (candidate_percentage * total_votes / 100) - (other_percentage * total_votes / 100) = 1040 := 
by
  sorry

end NUMINAMATH_GPT_candidate_majority_votes_l2193_219334


namespace NUMINAMATH_GPT_real_roots_range_of_m_l2193_219352

noncomputable def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem real_roots_range_of_m :
  (∃ x : ℝ, x^2 + 4 * m * x + 4 * m^2 + 2 * m + 3 = 0) ∨ 
  (∃ x : ℝ, x^2 + (2 * m + 1) * x + m^2 = 0) ↔ 
  m ≤ -3 / 2 ∨ m ≥ -1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_real_roots_range_of_m_l2193_219352


namespace NUMINAMATH_GPT_cube_edge_length_l2193_219392

theorem cube_edge_length (surface_area : ℝ) (h : surface_area = 96) : ∃ (edge_length : ℝ), edge_length = 4 := 
by 
  sorry

end NUMINAMATH_GPT_cube_edge_length_l2193_219392


namespace NUMINAMATH_GPT_regular_polygon_sides_l2193_219394

theorem regular_polygon_sides (n : ℕ) (h : 1 < n) (exterior_angle : ℝ) (h_ext : exterior_angle = 18) :
  n * exterior_angle = 360 → n = 20 :=
by 
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l2193_219394


namespace NUMINAMATH_GPT_A_eq_B_l2193_219327

variables (α : Type) (Q : α → Prop)
variables (A B C : α → Prop)

-- Conditions
-- 1. For the questions where both B and C answered "yes", A also answered "yes".
axiom h1 : ∀ q, B q ∧ C q → A q
-- 2. For the questions where A answered "yes", B also answered "yes".
axiom h2 : ∀ q, A q → B q
-- 3. For the questions where B answered "yes", at least one of A and C answered "yes".
axiom h3 : ∀ q, B q → (A q ∨ C q)

-- Prove that A and B gave the same answer to all questions
theorem A_eq_B : ∀ q, A q ↔ B q :=
sorry

end NUMINAMATH_GPT_A_eq_B_l2193_219327


namespace NUMINAMATH_GPT_remaining_days_to_finish_coke_l2193_219322

def initial_coke_in_ml : ℕ := 2000
def daily_consumption_in_ml : ℕ := 200
def days_already_drunk : ℕ := 3

theorem remaining_days_to_finish_coke : 
  (initial_coke_in_ml / daily_consumption_in_ml) - days_already_drunk = 7 := 
by
  sorry -- Proof placeholder

end NUMINAMATH_GPT_remaining_days_to_finish_coke_l2193_219322
