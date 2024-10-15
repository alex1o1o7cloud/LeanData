import Mathlib

namespace NUMINAMATH_GPT_medicine_supply_duration_l1905_190529

noncomputable def pillDuration (numPills : ℕ) (pillFractionPerThreeDays : ℚ) : ℚ :=
  let pillPerDay := pillFractionPerThreeDays / 3
  let daysPerPill := 1 / pillPerDay
  numPills * daysPerPill

theorem medicine_supply_duration (numPills : ℕ) (pillFractionPerThreeDays : ℚ) (daysPerMonth : ℚ) :
  numPills = 90 →
  pillFractionPerThreeDays = 1 / 3 →
  daysPerMonth = 30 →
  pillDuration numPills pillFractionPerThreeDays / daysPerMonth = 27 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp [pillDuration]
  sorry

end NUMINAMATH_GPT_medicine_supply_duration_l1905_190529


namespace NUMINAMATH_GPT_smallest_four_digit_divisible_by_34_l1905_190573

/-- Define a four-digit number. -/
def is_four_digit (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000

/-- Define a number to be divisible by another number. -/
def divisible_by (n k : ℕ) : Prop :=
  k ∣ n

/-- Prove that the smallest four-digit number divisible by 34 is 1020. -/
theorem smallest_four_digit_divisible_by_34 : ∃ n : ℕ, is_four_digit n ∧ divisible_by n 34 ∧ 
    (∀ m : ℕ, is_four_digit m → divisible_by m 34 → n ≤ m) :=
  sorry

end NUMINAMATH_GPT_smallest_four_digit_divisible_by_34_l1905_190573


namespace NUMINAMATH_GPT_proof_problem_l1905_190533

theorem proof_problem (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : a + b - 1 / (2 * a) - 2 / b = 3 / 2) :
  (a < 1 → b > 2) ∧ (∀ x y : ℝ, x > 0 → y > 0 → x + y - 1 / (2 * x) - 2 / y = 3 / 2 → x + y ≥ 3) :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l1905_190533


namespace NUMINAMATH_GPT_smallest_distance_l1905_190582

open Real

/-- Let A be a point on the circle (x-3)^2 + (y-4)^2 = 16,
and let B be a point on the parabola x^2 = 8y.
The smallest possible distance AB is √34 - 4. -/
theorem smallest_distance 
  (A B : ℝ × ℝ)
  (hA : (A.1 - 3)^2 + (A.2 - 4)^2 = 16)
  (hB : (B.1)^2 = 8 * B.2) :
  sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≥ sqrt 34 - 4 := 
sorry

end NUMINAMATH_GPT_smallest_distance_l1905_190582


namespace NUMINAMATH_GPT_unique_solution_for_exponential_eq_l1905_190507

theorem unique_solution_for_exponential_eq (a y : ℕ) (h_a : a ≥ 1) (h_y : y ≥ 1) :
  3^(2*a-1) + 3^a + 1 = 7^y ↔ (a = 1 ∧ y = 1) := by
  sorry

end NUMINAMATH_GPT_unique_solution_for_exponential_eq_l1905_190507


namespace NUMINAMATH_GPT_negation_correct_l1905_190554

def original_statement (x : ℝ) : Prop := x > 0 → x^2 + 3 * x - 2 > 0

def negated_statement (x : ℝ) : Prop := x > 0 ∧ x^2 + 3 * x - 2 ≤ 0

theorem negation_correct : (¬ ∀ x, original_statement x) ↔ ∃ x, negated_statement x := by
  sorry

end NUMINAMATH_GPT_negation_correct_l1905_190554


namespace NUMINAMATH_GPT_greatest_value_divisible_by_3_l1905_190551

theorem greatest_value_divisible_by_3 :
  ∃ (a : ℕ), (168026 + 1000 * a) % 3 = 0 ∧ a ≤ 9 ∧ ∀ b : ℕ, (168026 + 1000 * b) % 3 = 0 → b ≤ 9 → a ≥ b :=
sorry

end NUMINAMATH_GPT_greatest_value_divisible_by_3_l1905_190551


namespace NUMINAMATH_GPT_problem_l1905_190568

theorem problem (a : ℝ) : (∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) → (a > 3 ∨ a < -1) :=
by
  sorry

end NUMINAMATH_GPT_problem_l1905_190568


namespace NUMINAMATH_GPT_profit_difference_l1905_190589

-- Definitions of the conditions
def car_cost : ℕ := 100
def cars_per_month : ℕ := 4
def car_revenue : ℕ := 50

def motorcycle_cost : ℕ := 250
def motorcycles_per_month : ℕ := 8
def motorcycle_revenue : ℕ := 50

-- Calculation of profits
def car_profit : ℕ := (cars_per_month * car_revenue) - car_cost
def motorcycle_profit : ℕ := (motorcycles_per_month * motorcycle_revenue) - motorcycle_cost

-- Prove that the profit difference is 50 dollars
theorem profit_difference : (motorcycle_profit - car_profit) = 50 :=
by
  -- Statements to assert conditions and their proofs go here
  sorry

end NUMINAMATH_GPT_profit_difference_l1905_190589


namespace NUMINAMATH_GPT_lastTwoNonZeroDigits_of_80_fact_is_8_l1905_190562

-- Define the factorial function
def fac : ℕ → ℕ
  | 0     => 1
  | (n+1) => (n+1) * fac n

-- Define the function to find the last two nonzero digits of a factorial
def lastTwoNonZeroDigits (n : ℕ) : ℕ := sorry -- Placeholder logic for now

-- State the problem as a theorem
theorem lastTwoNonZeroDigits_of_80_fact_is_8 :
  lastTwoNonZeroDigits 80 = 8 :=
sorry

end NUMINAMATH_GPT_lastTwoNonZeroDigits_of_80_fact_is_8_l1905_190562


namespace NUMINAMATH_GPT_circus_accommodation_l1905_190528

theorem circus_accommodation : 246 * 4 = 984 := by
  sorry

end NUMINAMATH_GPT_circus_accommodation_l1905_190528


namespace NUMINAMATH_GPT_find_c_l1905_190511

-- Define the quadratic polynomial with given conditions
def quadratic (b c x y : ℝ) : Prop :=
  y = x^2 + b * x + c

-- Define the condition that the polynomial passes through two particular points
def passes_through_points (b c : ℝ) : Prop :=
  (quadratic b c 1 4) ∧ (quadratic b c 5 4)

-- The theorem stating c is 9 given the conditions
theorem find_c (b c : ℝ) (h : passes_through_points b c) : c = 9 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_c_l1905_190511


namespace NUMINAMATH_GPT_find_initial_cookies_l1905_190572

-- Definitions based on problem conditions
def initial_cookies (x : ℕ) : Prop :=
  let after_eating := x - 2
  let after_buying := after_eating + 37
  after_buying = 75

-- Main statement to be proved
theorem find_initial_cookies : ∃ x, initial_cookies x ∧ x = 40 :=
by
  sorry

end NUMINAMATH_GPT_find_initial_cookies_l1905_190572


namespace NUMINAMATH_GPT_variance_of_numbers_l1905_190548

noncomputable def variance (s : List ℕ) : ℚ :=
  let mean := (s.sum : ℚ) / s.length
  let sqDiffs := s.map (λ n => (n - mean) ^ 2)
  sqDiffs.sum / s.length

def avg_is_34 (s : List ℕ) : Prop := (s.sum : ℚ) / s.length = 34

theorem variance_of_numbers (x : ℕ) 
  (h : avg_is_34 [31, 38, 34, 35, x]) : variance [31, 38, 34, 35, x] = 6 := 
by
  sorry

end NUMINAMATH_GPT_variance_of_numbers_l1905_190548


namespace NUMINAMATH_GPT_percent_of_motorists_receive_speeding_tickets_l1905_190536

theorem percent_of_motorists_receive_speeding_tickets
    (p_exceed : ℝ)
    (p_no_ticket : ℝ)
    (h1 : p_exceed = 0.125)
    (h2 : p_no_ticket = 0.20) : 
    (0.8 * p_exceed) * 100 = 10 :=
by
  sorry

end NUMINAMATH_GPT_percent_of_motorists_receive_speeding_tickets_l1905_190536


namespace NUMINAMATH_GPT_train_speed_is_correct_l1905_190579

-- Define the conditions.
def length_of_train : ℕ := 1800 -- Length of the train in meters.
def time_to_cross_platform : ℕ := 60 -- Time to cross the platform in seconds (1 minute).

-- Define the statement that needs to be proved.
def speed_of_train : ℕ := (2 * length_of_train) / time_to_cross_platform

-- State the theorem.
theorem train_speed_is_correct :
  speed_of_train = 60 := by
  sorry -- Proof is not required.

end NUMINAMATH_GPT_train_speed_is_correct_l1905_190579


namespace NUMINAMATH_GPT_value_of_a6_l1905_190520

theorem value_of_a6 (a : ℕ → ℝ) (h_positive : ∀ n, 0 < a n)
  (h_a1 : a 1 = 1) (h_a2 : a 2 = 2)
  (h_recurrence : ∀ n, 2 * (a n)^2 = (a (n + 1))^2 + (a (n - 1))^2) :
  a 6 = 4 := 
sorry

end NUMINAMATH_GPT_value_of_a6_l1905_190520


namespace NUMINAMATH_GPT_find_x_l1905_190501

-- Define the variables and conditions
def x := 27
axiom h : (5 / 3) * x = 45

-- Main statement to be proved
theorem find_x : x = 27 :=
by
  have : (5 / 3) * x = 45 := h
  sorry

end NUMINAMATH_GPT_find_x_l1905_190501


namespace NUMINAMATH_GPT_highest_weekly_sales_is_60_l1905_190592

/-- 
Given that a convenience store sold 300 bags of chips in a month,
and the following weekly sales pattern:
1. In the first week, 20 bags were sold.
2. In the second week, there was a 2-for-1 promotion, tripling the sales to 60 bags.
3. In the third week, a 10% discount doubled the sales to 40 bags.
4. In the fourth week, sales returned to the first week's number, 20 bags.
Prove that the number of bags of chips sold during the week with the highest demand is 60.
-/
theorem highest_weekly_sales_is_60 
  (total_sales : ℕ)
  (week1_sales : ℕ)
  (week2_sales : ℕ)
  (week3_sales : ℕ)
  (week4_sales : ℕ)
  (h_total : total_sales = 300)
  (h_week1 : week1_sales = 20)
  (h_week2 : week2_sales = 3 * week1_sales)
  (h_week3 : week3_sales = 2 * week1_sales)
  (h_week4 : week4_sales = week1_sales) :
  max (max week1_sales week2_sales) (max week3_sales week4_sales) = 60 := 
sorry

end NUMINAMATH_GPT_highest_weekly_sales_is_60_l1905_190592


namespace NUMINAMATH_GPT_constant_term_of_expansion_l1905_190565

theorem constant_term_of_expansion (x : ℝ) : 
  (∃ c : ℝ, c = 15 ∧ ∀ r : ℕ, r = 1 → (Nat.choose 5 r * 3^r * x^((5-5*r)/2) = c)) :=
by
  sorry

end NUMINAMATH_GPT_constant_term_of_expansion_l1905_190565


namespace NUMINAMATH_GPT_inequality_solution_interval_l1905_190556

noncomputable def solve_inequality (x : ℝ) : Prop :=
  -2 < (x^2 - 16 * x + 15) / (x^2 - 4 * x + 5) ∧
  (x^2 - 4 * x + 5) ≠ 0 ∧
  (3 * x^2 - 24 * x + 25) / (x^2 - 4 * x + 5) > 0 ∧
  (- x^2 - 8 * x + 5) / (x^2 - 4 * x + 5) < 0

theorem inequality_solution_interval (x : ℝ) :
  solve_inequality x :=
sorry

end NUMINAMATH_GPT_inequality_solution_interval_l1905_190556


namespace NUMINAMATH_GPT_sum_of_15_consecutive_integers_perfect_square_l1905_190526

open Nat

-- statement that defines the conditions and the objective of the problem
theorem sum_of_15_consecutive_integers_perfect_square :
  ∃ n k : ℕ, 15 * (n + 7) = k^2 ∧ 15 * (n + 7) ≥ 225 := 
sorry

end NUMINAMATH_GPT_sum_of_15_consecutive_integers_perfect_square_l1905_190526


namespace NUMINAMATH_GPT_right_triangle_hypotenuse_l1905_190534

theorem right_triangle_hypotenuse (a b c : ℝ) 
  (h₁ : a + b + c = 40) 
  (h₂ : a * b = 60) 
  (h₃ : a^2 + b^2 = c^2) : c = 18.5 := 
by 
  sorry

end NUMINAMATH_GPT_right_triangle_hypotenuse_l1905_190534


namespace NUMINAMATH_GPT_solution1_solution2_l1905_190561

noncomputable def problem1 (x : ℝ) : Prop :=
  4 * x^2 - 25 = 0

theorem solution1 (x : ℝ) : problem1 x ↔ x = 5 / 2 ∨ x = -5 / 2 :=
by sorry

noncomputable def problem2 (x : ℝ) : Prop :=
  (x + 1)^3 = -27

theorem solution2 (x : ℝ) : problem2 x ↔ x = -4 :=
by sorry

end NUMINAMATH_GPT_solution1_solution2_l1905_190561


namespace NUMINAMATH_GPT_cookout_2006_kids_l1905_190544

def kids_2004 : ℕ := 60
def kids_2005 : ℕ := kids_2004 / 2
def kids_2006 : ℕ := (2 * kids_2005) / 3

theorem cookout_2006_kids : kids_2006 = 20 := by
  sorry

end NUMINAMATH_GPT_cookout_2006_kids_l1905_190544


namespace NUMINAMATH_GPT_find_a_l1905_190547

noncomputable def f' (x : ℝ) (a : ℝ) := 2 * x^3 + a * x^2 + x

theorem find_a (a : ℝ) (h : f' 1 a = 9) : a = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1905_190547


namespace NUMINAMATH_GPT_sandy_books_l1905_190597

theorem sandy_books (x : ℕ)
  (h1 : 1080 + 840 = 1920)
  (h2 : 16 = 1920 / (x + 55)) :
  x = 65 :=
by
  -- Theorem proof placeholder
  sorry

end NUMINAMATH_GPT_sandy_books_l1905_190597


namespace NUMINAMATH_GPT_area_of_remaining_figure_l1905_190585
noncomputable def π := Real.pi

theorem area_of_remaining_figure (R : ℝ) (chord_length : ℝ) (C : ℝ) 
  (h : chord_length = 8) (hC : C = R) : (π * R^2 - 2 * π * (R / 2)^2) = 12.57 := by
  sorry

end NUMINAMATH_GPT_area_of_remaining_figure_l1905_190585


namespace NUMINAMATH_GPT_three_digit_number_equality_l1905_190590

theorem three_digit_number_equality :
  ∃ (x y z : ℕ), 1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 ∧ 0 ≤ z ∧ z ≤ 9 ∧
  (100 * x + 10 * y + z = x^2 + y + z^3) ∧
  (100 * x + 10 * y + z = 357) :=
by
  sorry

end NUMINAMATH_GPT_three_digit_number_equality_l1905_190590


namespace NUMINAMATH_GPT_ninety_seven_squared_l1905_190538

theorem ninety_seven_squared :
  97 * 97 = 9409 :=
by sorry

end NUMINAMATH_GPT_ninety_seven_squared_l1905_190538


namespace NUMINAMATH_GPT_only_set_B_is_right_angle_triangle_l1905_190555

def is_right_angle_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

theorem only_set_B_is_right_angle_triangle :
  is_right_angle_triangle 3 4 5 ∧ ¬is_right_angle_triangle 1 2 2 ∧ ¬is_right_angle_triangle 3 4 9 ∧ ¬is_right_angle_triangle 4 5 7 :=
by
  -- proof steps omitted
  sorry

end NUMINAMATH_GPT_only_set_B_is_right_angle_triangle_l1905_190555


namespace NUMINAMATH_GPT_mean_daily_profit_l1905_190510

theorem mean_daily_profit 
  (mean_first_15_days : ℝ) 
  (mean_last_15_days : ℝ) 
  (n : ℝ) 
  (m1_days : ℝ) 
  (m2_days : ℝ) : 
  (mean_first_15_days = 245) → 
  (mean_last_15_days = 455) → 
  (m1_days = 15) → 
  (m2_days = 15) → 
  (n = 30) →
  (∀ P, P = (245 * 15 + 455 * 15) / 30) :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_mean_daily_profit_l1905_190510


namespace NUMINAMATH_GPT_part1_part2_l1905_190588

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - 1 / (x + a)

theorem part1 (a x : ℝ):
  a ≥ 1 → x > 0 → f x a ≥ 0 := 
sorry

theorem part2 (a : ℝ):
  0 < a ∧ a ≤ 2 / 3 → ∃! x, x > -a ∧ f x a = 0 :=
sorry

end NUMINAMATH_GPT_part1_part2_l1905_190588


namespace NUMINAMATH_GPT_first_bag_weight_l1905_190515

def weight_of_first_bag (initial_weight : ℕ) (second_bag : ℕ) (total_weight : ℕ) : ℕ :=
  total_weight - second_bag - initial_weight

theorem first_bag_weight : weight_of_first_bag 15 10 40 = 15 :=
by
  unfold weight_of_first_bag
  sorry

end NUMINAMATH_GPT_first_bag_weight_l1905_190515


namespace NUMINAMATH_GPT_smallest_possible_value_of_n_l1905_190517

theorem smallest_possible_value_of_n :
  ∃ n : ℕ, (60 * n = (x + 6) * x * (x + 6) ∧ (x > 0) ∧ gcd 60 n = x + 6) ∧ n = 93 :=
by
  sorry

end NUMINAMATH_GPT_smallest_possible_value_of_n_l1905_190517


namespace NUMINAMATH_GPT_at_least_one_not_less_than_two_l1905_190567

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c):
  (a + 1 / b) ≥ 2 ∨ (b + 1 / c) ≥ 2 ∨ (c + 1 / a) ≥ 2 :=
sorry

end NUMINAMATH_GPT_at_least_one_not_less_than_two_l1905_190567


namespace NUMINAMATH_GPT_f_comp_g_eq_g_comp_f_iff_l1905_190502

variable {R : Type} [CommRing R]

def f (m n : R) (x : R) : R := m * x ^ 2 + n
def g (p q : R) (x : R) : R := p * x + q

theorem f_comp_g_eq_g_comp_f_iff (m n p q : R) :
  (∀ x : R, f m n (g p q x) = g p q (f m n x)) ↔ n * (1 - p ^ 2) - q * (1 - m) = 0 :=
by
  sorry

end NUMINAMATH_GPT_f_comp_g_eq_g_comp_f_iff_l1905_190502


namespace NUMINAMATH_GPT_line_slope_and_intersection_l1905_190532

theorem line_slope_and_intersection:
  (∀ x y : ℝ, x^2 + x / 4 + y / 5 = 1 → ∀ m : ℝ, m = -5 / 4) ∧
  (∀ x y : ℝ, x^2 + y^2 = 1 → ¬ (x^2 + x / 4 + y / 5 = 1)) :=
by
  sorry

end NUMINAMATH_GPT_line_slope_and_intersection_l1905_190532


namespace NUMINAMATH_GPT_largest_constant_inequality_l1905_190523

theorem largest_constant_inequality :
  ∃ C, C = 3 ∧
  (∀ (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ),
  (x₁ + x₂ + x₃ + x₄ + x₅ + x₆)^2 ≥ 
  C * (x₁ * (x₂ + x₃) + x₂ * (x₃ + x₄) + x₃ * (x₄ + x₅) + x₄ * (x₅ + x₆) + x₅ * (x₆ + x₁) + x₆ * (x₁ + x₂))) :=

sorry

end NUMINAMATH_GPT_largest_constant_inequality_l1905_190523


namespace NUMINAMATH_GPT_walking_speed_l1905_190516

theorem walking_speed (total_time : ℕ) (distance : ℕ) (rest_interval : ℕ) (rest_time : ℕ) (rest_periods: ℕ) 
  (total_rest_time: ℕ) (total_walking_time: ℕ) (hours: ℕ) 
  (H1 : total_time = 332) 
  (H2 : distance = 50) 
  (H3 : rest_interval = 10) 
  (H4 : rest_time = 8)
  (H5 : rest_periods = distance / rest_interval - 1) 
  (H6 : total_rest_time = rest_periods * rest_time)
  (H7 : total_walking_time = total_time - total_rest_time) 
  (H8 : hours = total_walking_time / 60) : 
  (distance / hours) = 10 :=
by {
  -- proof omitted
  sorry
}

end NUMINAMATH_GPT_walking_speed_l1905_190516


namespace NUMINAMATH_GPT_sin_365_1_eq_m_l1905_190549

noncomputable def sin_value (θ : ℝ) : ℝ := Real.sin (Real.pi * θ / 180)
variables (m : ℝ) (h : sin_value 5.1 = m)

theorem sin_365_1_eq_m : sin_value 365.1 = m :=
by sorry

end NUMINAMATH_GPT_sin_365_1_eq_m_l1905_190549


namespace NUMINAMATH_GPT_jane_current_age_l1905_190527

theorem jane_current_age (J : ℕ) (h1 : ∀ t : ℕ, t = 13 → 25 + t = 2 * (J + t)) : J = 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_jane_current_age_l1905_190527


namespace NUMINAMATH_GPT_expand_polynomials_l1905_190550

variable (x : ℝ)

theorem expand_polynomials : 
  (3 * x^2 - 4 * x + 3) * (-4 * x^2 + 2 * x - 6) = -12 * x^4 + 22 * x^3 - 38 * x^2 + 30 * x - 18 :=
  by
  sorry

end NUMINAMATH_GPT_expand_polynomials_l1905_190550


namespace NUMINAMATH_GPT_toot_has_vertical_symmetry_l1905_190595

def has_vertical_symmetry (letter : Char) : Prop :=
  letter = 'T' ∨ letter = 'O'

def word_has_vertical_symmetry (word : List Char) : Prop :=
  ∀ letter ∈ word, has_vertical_symmetry letter

theorem toot_has_vertical_symmetry : word_has_vertical_symmetry ['T', 'O', 'O', 'T'] :=
  by
    sorry

end NUMINAMATH_GPT_toot_has_vertical_symmetry_l1905_190595


namespace NUMINAMATH_GPT_factorize_m_l1905_190564

theorem factorize_m (m : ℝ) : m^2 - 4 * m - 5 = (m + 1) * (m - 5) := 
sorry

end NUMINAMATH_GPT_factorize_m_l1905_190564


namespace NUMINAMATH_GPT_eggs_per_week_is_84_l1905_190508

-- Define the number of pens
def number_of_pens : Nat := 4

-- Define the number of emus per pen
def emus_per_pen : Nat := 6

-- Define the number of days in a week
def days_in_week : Nat := 7

-- Define the number of eggs per female emu per day
def eggs_per_female_emu_per_day : Nat := 1

-- Calculate the total number of emus
def total_emus : Nat := number_of_pens * emus_per_pen

-- Calculate the number of female emus
def female_emus : Nat := total_emus / 2

-- Calculate the number of eggs per day
def eggs_per_day : Nat := female_emus * eggs_per_female_emu_per_day

-- Calculate the number of eggs per week
def eggs_per_week : Nat := eggs_per_day * days_in_week

-- The theorem to prove
theorem eggs_per_week_is_84 : eggs_per_week = 84 := by
  sorry

end NUMINAMATH_GPT_eggs_per_week_is_84_l1905_190508


namespace NUMINAMATH_GPT_find_a_b_sum_l1905_190500

theorem find_a_b_sum (a b : ℕ) (h1 : 830 - (400 + 10 * a + 7) = 300 + 10 * b + 4)
    (h2 : ∃ k : ℕ, 300 + 10 * b + 4 = 7 * k) : a + b = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_b_sum_l1905_190500


namespace NUMINAMATH_GPT_calc_fraction_l1905_190598

theorem calc_fraction :
  ((1 / 3 + 1 / 6) * (4 / 7) * (5 / 9) = 10 / 63) :=
by
  sorry

end NUMINAMATH_GPT_calc_fraction_l1905_190598


namespace NUMINAMATH_GPT_positive_diff_solutions_l1905_190513

theorem positive_diff_solutions : 
  (∃ x₁ x₂ : ℝ, ( (9 - x₁^2 / 4)^(1/3) = -3) ∧ ((9 - x₂^2 / 4)^(1/3) = -3) ∧ ∃ (d : ℝ), d = |x₁ - x₂| ∧ d = 24) :=
by
  sorry

end NUMINAMATH_GPT_positive_diff_solutions_l1905_190513


namespace NUMINAMATH_GPT_old_geometry_book_pages_l1905_190503

def old_pages := 340
def new_pages := 450
def deluxe_pages := 915

theorem old_geometry_book_pages : 
  (new_pages = 2 * old_pages - 230) ∧ 
  (deluxe_pages = new_pages + old_pages + 125) ∧ 
  (deluxe_pages ≥ old_pages + old_pages / 10) 
  → old_pages = 340 := by
  sorry

end NUMINAMATH_GPT_old_geometry_book_pages_l1905_190503


namespace NUMINAMATH_GPT_total_lateness_l1905_190593

/-
  Conditions:
  Charlize was 20 minutes late.
  Ana was 5 minutes later than Charlize.
  Ben was 15 minutes less late than Charlize.
  Clara was twice as late as Charlize.
  Daniel was 10 minutes earlier than Clara.

  Total time for which all five students were late is 120 minutes.
-/

def charlize := 20
def ana := charlize + 5
def ben := charlize - 15
def clara := charlize * 2
def daniel := clara - 10

def total_time := charlize + ana + ben + clara + daniel

theorem total_lateness : total_time = 120 :=
by
  sorry

end NUMINAMATH_GPT_total_lateness_l1905_190593


namespace NUMINAMATH_GPT_equal_real_roots_of_quadratic_l1905_190559

theorem equal_real_roots_of_quadratic (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - m * x + 3 = 0 ∧ 
               (∀ y : ℝ, 3 * y^2 - m * y + 3 = 0 → y = x)) → 
  m = 6 ∨ m = -6 :=
by
  sorry  -- proof to be filled in.

end NUMINAMATH_GPT_equal_real_roots_of_quadratic_l1905_190559


namespace NUMINAMATH_GPT_algebraic_expression_values_l1905_190575

-- Defining the given condition
def condition (x y : ℝ) : Prop :=
  x^4 + 6 * x^2 * y + 9 * y^2 + 2 * x^2 + 6 * y + 4 = 7

-- Defining the target expression
def target_expression (x y : ℝ) : ℝ :=
  x^4 + 6 * x^2 * y + 9 * y^2 - 2 * x^2 - 6 * y - 1

-- Stating the theorem to be proved
theorem algebraic_expression_values (x y : ℝ) (h : condition x y) :
  target_expression x y = -2 ∨ target_expression x y = 14 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_values_l1905_190575


namespace NUMINAMATH_GPT_find_x_l1905_190584

def vector := ℝ × ℝ

def a : vector := (1, 1)
def b (x : ℝ) : vector := (2, x)

def vector_add (u v : vector) : vector :=
(u.1 + v.1, u.2 + v.2)

def scalar_mul (k : ℝ) (v : vector) : vector :=
(k * v.1, k * v.2)

def vector_sub (u v : vector) : vector :=
(u.1 - v.1, u.2 - v.2)

def are_parallel (u v : vector) : Prop :=
∃ k : ℝ, u = scalar_mul k v

theorem find_x (x : ℝ) : are_parallel (vector_add a (b x)) (vector_sub (scalar_mul 4 (b x)) (scalar_mul 2 a)) → x = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1905_190584


namespace NUMINAMATH_GPT_clock_chime_time_l1905_190524

theorem clock_chime_time (t : ℕ) (h : t = 12) (k : 4 * (t / (4 - 1)) = 12) :
  12 * (t / (4 - 1)) - (12 - 1) * (t / (4 - 1)) = 44 :=
by {
  sorry
}

end NUMINAMATH_GPT_clock_chime_time_l1905_190524


namespace NUMINAMATH_GPT_bus_seating_options_l1905_190521

theorem bus_seating_options :
  ∃! (x y : ℕ), 21*x + 10*y = 241 :=
sorry

end NUMINAMATH_GPT_bus_seating_options_l1905_190521


namespace NUMINAMATH_GPT_green_pill_cost_l1905_190505

theorem green_pill_cost (p g : ℕ) (h1 : g = p + 1) (h2 : 14 * (p + g) = 546) : g = 20 :=
by
  sorry

end NUMINAMATH_GPT_green_pill_cost_l1905_190505


namespace NUMINAMATH_GPT_total_tiles_in_square_hall_l1905_190545

theorem total_tiles_in_square_hall
  (s : ℕ) -- integer side length of the square hall
  (black_tiles : ℕ)
  (total_tiles : ℕ)
  (all_tiles_white_or_black : ∀ (x : ℕ), x ≤ total_tiles → x = black_tiles ∨ x = total_tiles - black_tiles)
  (black_tiles_count : black_tiles = 153 + 3) : total_tiles = 6084 :=
by
  sorry

end NUMINAMATH_GPT_total_tiles_in_square_hall_l1905_190545


namespace NUMINAMATH_GPT_team_matches_per_season_l1905_190537

theorem team_matches_per_season (teams_count total_games : ℕ) (h1 : teams_count = 50) (h2 : total_games = 4900) : 
  ∃ n : ℕ, n * (teams_count - 1) * teams_count / 2 = total_games ∧ n = 2 :=
by
  sorry

end NUMINAMATH_GPT_team_matches_per_season_l1905_190537


namespace NUMINAMATH_GPT_negation_abs_lt_zero_l1905_190512

theorem negation_abs_lt_zero : ¬ (∀ x : ℝ, |x| < 0) ↔ ∃ x : ℝ, |x| ≥ 0 := 
by 
  sorry

end NUMINAMATH_GPT_negation_abs_lt_zero_l1905_190512


namespace NUMINAMATH_GPT_bond_interest_percentage_l1905_190578

noncomputable def interest_percentage_of_selling_price (face_value interest_rate : ℝ) (selling_price : ℝ) : ℝ :=
  (face_value * interest_rate) / selling_price * 100

theorem bond_interest_percentage :
  let face_value : ℝ := 5000
  let interest_rate : ℝ := 0.07
  let selling_price : ℝ := 5384.615384615386
  interest_percentage_of_selling_price face_value interest_rate selling_price = 6.5 :=
by
  sorry

end NUMINAMATH_GPT_bond_interest_percentage_l1905_190578


namespace NUMINAMATH_GPT_tree_growth_per_year_l1905_190514

-- Defining the initial height and age.
def initial_height : ℕ := 5
def initial_age : ℕ := 1

-- Defining the height and age after a certain number of years.
def height_at_7_years : ℕ := 23
def age_at_7_years : ℕ := 7

-- Calculating the total growth and number of years.
def total_height_growth : ℕ := height_at_7_years - initial_height
def years_of_growth : ℕ := age_at_7_years - initial_age

-- Stating the theorem to be proven.
theorem tree_growth_per_year : total_height_growth / years_of_growth = 3 :=
by
  sorry

end NUMINAMATH_GPT_tree_growth_per_year_l1905_190514


namespace NUMINAMATH_GPT_number_of_people_l1905_190541

-- Definitions based on conditions
def per_person_cost (x : ℕ) : ℕ :=
  if x ≤ 30 then 100 else max 72 (100 - 2 * (x - 30))

def total_cost (x : ℕ) : ℕ :=
  x * per_person_cost x

-- Main theorem statement
theorem number_of_people (x : ℕ) (h1 : total_cost x = 3150) (h2 : x > 30) : x = 35 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_people_l1905_190541


namespace NUMINAMATH_GPT_point_D_not_in_region_l1905_190546

-- Define the condition that checks if a point is not in the region defined by 3x + 2y < 6
def point_not_in_region (x y : ℝ) : Prop :=
  ¬ (3 * x + 2 * y < 6)

-- Define the points
def A := (0, 0)
def B := (1, 1)
def C := (0, 2)
def D := (2, 0)

-- The proof problem as a Lean statement
theorem point_D_not_in_region : point_not_in_region (2:ℝ) (0:ℝ) :=
by
  show point_not_in_region 2 0
  sorry

end NUMINAMATH_GPT_point_D_not_in_region_l1905_190546


namespace NUMINAMATH_GPT_store_paid_price_l1905_190563

theorem store_paid_price (selling_price : ℕ) (less_amount : ℕ) 
(h1 : selling_price = 34) (h2 : less_amount = 8) : ∃ p : ℕ, p = selling_price - less_amount ∧ p = 26 := 
by
  sorry

end NUMINAMATH_GPT_store_paid_price_l1905_190563


namespace NUMINAMATH_GPT_set1_eq_set2_eq_set3_eq_set4_eq_set5_eq_l1905_190509

open Set

-- (1) The set of integers whose absolute value is not greater than 2
theorem set1_eq : { x : ℤ | |x| ≤ 2 } = {-2, -1, 0, 1, 2} := sorry

-- (2) The set of positive numbers less than 10 that are divisible by 3
theorem set2_eq : { x : ℕ | x < 10 ∧ x > 0 ∧ x % 3 = 0 } = {3, 6, 9} := sorry

-- (3) The set {x | x = |x|, x < 5, x ∈ 𝕫}
theorem set3_eq : { x : ℕ | x < 5 } = {0, 1, 2, 3, 4} := sorry

-- (4) The set {(x, y) | x + y = 6, x ∈ ℕ⁺, y ∈ ℕ⁺}
theorem set4_eq : { p : ℕ × ℕ | p.1 + p.2 = 6 ∧ p.1 > 0 ∧ p.2 > 0 } = {(1, 5), (2, 4), (3, 3), (4, 2), (5, 1) } := sorry

-- (5) The set {-3, -1, 1, 3, 5}
theorem set5_eq : {-3, -1, 1, 3, 5} = { x : ℤ | ∃ k : ℤ, x = 2 * k - 1 ∧ -1 ≤ k ∧ k ≤ 3 } := sorry

end NUMINAMATH_GPT_set1_eq_set2_eq_set3_eq_set4_eq_set5_eq_l1905_190509


namespace NUMINAMATH_GPT_plane_through_points_and_perpendicular_l1905_190518

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def plane_eq (A B C D : ℝ) (P : Point3D) : Prop :=
  A * P.x + B * P.y + C * P.z + D = 0

def vector_sub (P Q : Point3D) : Point3D :=
  ⟨Q.x - P.x, Q.y - P.y, Q.z - P.z⟩

def cross_product (u v : Point3D) : Point3D :=
  ⟨u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x⟩

def is_perpendicular (normal1 normal2 : Point3D) : Prop :=
  normal1.x * normal2.x + normal1.y * normal2.y + normal1.z * normal2.z = 0

theorem plane_through_points_and_perpendicular
  (P1 P2 : Point3D)
  (A B C D : ℝ)
  (n_perp : Point3D)
  (normal1_eq : n_perp = ⟨2, -1, 4⟩)
  (eqn_given : plane_eq 2 (-1) 4 7 P1)
  (vec := vector_sub P1 P2)
  (n := cross_product vec n_perp)
  (eqn : plane_eq 11 (-10) (-9) (-33) P1) :
  (plane_eq 11 (-10) (-9) (-33) P2 ∧ is_perpendicular n n_perp) :=
sorry

end NUMINAMATH_GPT_plane_through_points_and_perpendicular_l1905_190518


namespace NUMINAMATH_GPT_find_a_interval_l1905_190553

theorem find_a_interval :
  ∀ {a : ℝ}, (∃ b x y : ℝ, x = abs (y + a) + 4 / a ∧ x^2 + y^2 + 24 + b * (2 * y + b) = 10 * x) ↔ (a < 0 ∨ a ≥ 2 / 3) :=
by {
  sorry
}

end NUMINAMATH_GPT_find_a_interval_l1905_190553


namespace NUMINAMATH_GPT_not_directly_nor_inversely_proportional_A_not_directly_nor_inversely_proportional_D_l1905_190522

def equationA (x y : ℝ) : Prop := 2 * x + 3 * y = 5
def equationD (x y : ℝ) : Prop := 4 * x + 2 * y = 8

def directlyProportional (x y : ℝ) : Prop := ∃ k : ℝ, y = k * x
def inverselyProportional (x y : ℝ) : Prop := ∃ k : ℝ, x * y = k

theorem not_directly_nor_inversely_proportional_A (x y : ℝ) :
  equationA x y → ¬ (directlyProportional x y ∨ inverselyProportional x y) := 
sorry

theorem not_directly_nor_inversely_proportional_D (x y : ℝ) :
  equationD x y → ¬ (directlyProportional x y ∨ inverselyProportional x y) := 
sorry

end NUMINAMATH_GPT_not_directly_nor_inversely_proportional_A_not_directly_nor_inversely_proportional_D_l1905_190522


namespace NUMINAMATH_GPT_remainder_17_pow_2047_mod_23_l1905_190574

theorem remainder_17_pow_2047_mod_23 : (17 ^ 2047) % 23 = 11 := 
by
  sorry

end NUMINAMATH_GPT_remainder_17_pow_2047_mod_23_l1905_190574


namespace NUMINAMATH_GPT_smallest_percent_coffee_tea_l1905_190569

theorem smallest_percent_coffee_tea (C T : ℝ) (hC : C = 50) (hT : T = 60) : 
  ∃ x, x = C + T - 100 ∧ x = 10 :=
by
  sorry

end NUMINAMATH_GPT_smallest_percent_coffee_tea_l1905_190569


namespace NUMINAMATH_GPT_a_squared_plus_b_squared_less_than_c_squared_l1905_190504

theorem a_squared_plus_b_squared_less_than_c_squared 
  (a b c : Real) 
  (h : a^2 + b^2 + a * b + b * c + c * a < 0) : 
  a^2 + b^2 < c^2 := 
  by 
  sorry

end NUMINAMATH_GPT_a_squared_plus_b_squared_less_than_c_squared_l1905_190504


namespace NUMINAMATH_GPT_basil_has_winning_strategy_l1905_190586

-- Definitions based on conditions
def piles : Nat := 11
def stones_per_pile : Nat := 10
def peter_moves (n : Nat) := n = 1 ∨ n = 2 ∨ n = 3
def basil_moves (n : Nat) := n = 1 ∨ n = 2 ∨ n = 3

-- The main theorem to prove Basil has a winning strategy
theorem basil_has_winning_strategy 
  (total_stones : Nat := piles * stones_per_pile) 
  (peter_first : Bool := true) :
  exists winning_strategy_for_basil, 
    ∀ (piles_remaining : Nat) (sum_stones_remaining : Nat),
    sum_stones_remaining = piles_remaining * stones_per_pile ∨
    (1 ≤ piles_remaining ∧ piles_remaining ≤ piles) ∧
    (0 ≤ sum_stones_remaining ∧ sum_stones_remaining ≤ total_stones)
    → winning_strategy_for_basil = true := 
sorry -- The proof is omitted

end NUMINAMATH_GPT_basil_has_winning_strategy_l1905_190586


namespace NUMINAMATH_GPT_cos_product_l1905_190543

theorem cos_product : 
  (1 + Real.cos (Real.pi / 12)) * (1 + Real.cos (5 * Real.pi / 12)) * (1 + Real.cos (7 * Real.pi / 12)) * (1 + Real.cos (11 * Real.pi / 12)) = 1 / 8 := 
by
  sorry

end NUMINAMATH_GPT_cos_product_l1905_190543


namespace NUMINAMATH_GPT_statement_A_statement_B_statement_C_statement_D_l1905_190539

variable (a b : ℝ)

-- Given conditions
axiom positive_a : 0 < a
axiom positive_b : 0 < b
axiom condition : a + 2 * b = 2 * a * b

-- Prove the statements
theorem statement_A : a + 2 * b ≥ 4 := sorry
theorem statement_B : ¬ (a + b ≥ 4) := sorry
theorem statement_C : ¬ (a * b ≤ 2) := sorry
theorem statement_D : a^2 + 4 * b^2 ≥ 8 := sorry

end NUMINAMATH_GPT_statement_A_statement_B_statement_C_statement_D_l1905_190539


namespace NUMINAMATH_GPT_total_tea_consumption_l1905_190557

variables (S O P : ℝ)

theorem total_tea_consumption : 
  S + O = 11 →
  P + O = 15 →
  P + S = 13 →
  S + O + P = 19.5 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_total_tea_consumption_l1905_190557


namespace NUMINAMATH_GPT_total_clips_correct_l1905_190535

def clips_in_april : ℕ := 48
def clips_in_may : ℕ := clips_in_april / 2
def total_clips : ℕ := clips_in_april + clips_in_may

theorem total_clips_correct : total_clips = 72 := by
  sorry

end NUMINAMATH_GPT_total_clips_correct_l1905_190535


namespace NUMINAMATH_GPT_original_selling_price_l1905_190560

variable (P : ℝ)
variable (S : ℝ) 

-- Conditions
axiom profit_10_percent : S = 1.10 * P
axiom profit_diff : 1.17 * P - S = 42

-- Goal
theorem original_selling_price : S = 660 := by
  sorry

end NUMINAMATH_GPT_original_selling_price_l1905_190560


namespace NUMINAMATH_GPT_part1_part2_l1905_190540

open Set Real

noncomputable def A : Set ℝ := {x | x^2 - 2 * x - 3 ≤ 0}
noncomputable def B (m : ℝ) : Set ℝ := {x | m - 3 ≤ x ∧ x ≤ m + 3}
noncomputable def C : Set ℝ := {x | 2 ≤ x ∧ x ≤ 3}

theorem part1 (m : ℝ) (h : A ∩ B m = C) : m = 5 :=
  sorry

theorem part2 (m : ℝ) (h : A ⊆ (B m)ᶜ) : m < -4 ∨ 6 < m :=
  sorry

end NUMINAMATH_GPT_part1_part2_l1905_190540


namespace NUMINAMATH_GPT_polygon_sides_l1905_190583

theorem polygon_sides (n : ℕ) 
  (h1 : ∀ (i : ℕ), i < n → 180 - 360 / n = 150) : n = 12 := by
  sorry

end NUMINAMATH_GPT_polygon_sides_l1905_190583


namespace NUMINAMATH_GPT_parabola_vertex_l1905_190576

theorem parabola_vertex :
  (∃ h k, ∀ x, (x^2 - 2 = ((x - h) ^ 2) + k) ∧ (h = 0) ∧ (k = -2)) :=
by
  sorry

end NUMINAMATH_GPT_parabola_vertex_l1905_190576


namespace NUMINAMATH_GPT_pink_highlighters_count_l1905_190558

-- Define the necessary constants and types
def total_highlighters : ℕ := 12
def yellow_highlighters : ℕ := 2
def blue_highlighters : ℕ := 4

-- We aim to prove that the number of pink highlighters is 6
theorem pink_highlighters_count : ∃ (pink_highlighters : ℕ), 
  pink_highlighters = total_highlighters - (yellow_highlighters + blue_highlighters) ∧
  pink_highlighters = 6 :=
by
  sorry

end NUMINAMATH_GPT_pink_highlighters_count_l1905_190558


namespace NUMINAMATH_GPT_min_abs_phi_l1905_190580

theorem min_abs_phi {f : ℝ → ℝ} (h : ∀ x, f x = 3 * Real.sin (2 * x + φ) ∧ ∀ x, f (x) = f (2 * π / 3 - x)) :
  |φ| = π / 6 :=
by
  sorry

end NUMINAMATH_GPT_min_abs_phi_l1905_190580


namespace NUMINAMATH_GPT_solve_fun_problem_l1905_190530

variable (f : ℝ → ℝ)

-- Definitions of the conditions
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x
def is_monotonic_on_pos (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, 0 < x → x < y → f x < f y

-- The main theorem
theorem solve_fun_problem (h_even : is_even f) (h_monotonic : is_monotonic_on_pos f) :
  {x : ℝ | f (x + 1) = f (2 * x)} = {1, -1 / 3} := 
sorry

end NUMINAMATH_GPT_solve_fun_problem_l1905_190530


namespace NUMINAMATH_GPT_geom_arith_seq_first_term_is_two_l1905_190566

theorem geom_arith_seq_first_term_is_two (b q a d : ℝ) 
  (hq : q ≠ 1) 
  (h_geom_first : b = a + d) 
  (h_geom_second : b * q = a + 3 * d) 
  (h_geom_third : b * q^2 = a + 6 * d) 
  (h_prod : b * b * q * b * q^2 = 64) :
  b = 2 :=
by
  sorry

end NUMINAMATH_GPT_geom_arith_seq_first_term_is_two_l1905_190566


namespace NUMINAMATH_GPT_smallest_class_size_l1905_190591

theorem smallest_class_size (n : ℕ) (h : 5 * n + 1 > 40) : ∃ k : ℕ, k >= 41 :=
by sorry

end NUMINAMATH_GPT_smallest_class_size_l1905_190591


namespace NUMINAMATH_GPT_box_volume_l1905_190525

theorem box_volume (l w h V : ℝ) 
  (h1 : l * w = 30) 
  (h2 : w * h = 18) 
  (h3 : l * h = 10) 
  : V = l * w * h → V = 90 :=
by 
  intro volume_eq
  sorry

end NUMINAMATH_GPT_box_volume_l1905_190525


namespace NUMINAMATH_GPT_percent_non_filler_l1905_190577

def burger_weight : ℕ := 120
def filler_weight : ℕ := 30

theorem percent_non_filler : 
  let total_weight := burger_weight
  let filler := filler_weight
  let non_filler := total_weight - filler
  (non_filler / total_weight : ℚ) * 100 = 75 := by
  sorry

end NUMINAMATH_GPT_percent_non_filler_l1905_190577


namespace NUMINAMATH_GPT_height_difference_after_3_years_l1905_190599

/-- Conditions for the tree's and boy's growth rates per season. --/
def tree_spring_growth : ℕ := 4
def tree_summer_growth : ℕ := 6
def tree_fall_growth : ℕ := 2
def tree_winter_growth : ℕ := 1

def boy_spring_growth : ℕ := 2
def boy_summer_growth : ℕ := 2
def boy_fall_growth : ℕ := 0
def boy_winter_growth : ℕ := 0

/-- Initial heights. --/
def initial_tree_height : ℕ := 16
def initial_boy_height : ℕ := 24

/-- Length of each season in months. --/
def season_length : ℕ := 3

/-- Time period in years. --/
def years : ℕ := 3

/-- Prove the height difference between the tree and the boy after 3 years is 73 inches. --/
theorem height_difference_after_3_years :
    let tree_annual_growth := tree_spring_growth * season_length +
                             tree_summer_growth * season_length +
                             tree_fall_growth * season_length +
                             tree_winter_growth * season_length
    let tree_final_height := initial_tree_height + tree_annual_growth * years
    let boy_annual_growth := boy_spring_growth * season_length +
                            boy_summer_growth * season_length +
                            boy_fall_growth * season_length +
                            boy_winter_growth * season_length
    let boy_final_height := initial_boy_height + boy_annual_growth * years
    tree_final_height - boy_final_height = 73 :=
by sorry

end NUMINAMATH_GPT_height_difference_after_3_years_l1905_190599


namespace NUMINAMATH_GPT_cola_cost_l1905_190571

theorem cola_cost (h c : ℝ) (h1 : 3 * h + 2 * c = 360) (h2 : 2 * h + 3 * c = 390) : c = 90 :=
by
  sorry

end NUMINAMATH_GPT_cola_cost_l1905_190571


namespace NUMINAMATH_GPT_consecutive_numbers_perfect_square_l1905_190506

theorem consecutive_numbers_perfect_square (a : ℕ) (h : a ≥ 1) : 
  (a * (a + 1) * (a + 2) * (a + 3) + 1) = (a^2 + 3 * a + 1)^2 :=
by sorry

end NUMINAMATH_GPT_consecutive_numbers_perfect_square_l1905_190506


namespace NUMINAMATH_GPT_temperature_difference_l1905_190542

theorem temperature_difference 
    (freezer_temp : ℤ) (room_temp : ℤ) (temperature_difference : ℤ) 
    (h1 : freezer_temp = -4) 
    (h2 : room_temp = 18) : 
    temperature_difference = room_temp - freezer_temp := 
by 
  sorry

end NUMINAMATH_GPT_temperature_difference_l1905_190542


namespace NUMINAMATH_GPT_probability_calculation_correct_l1905_190594

def total_balls : ℕ := 100
def white_balls : ℕ := 50
def green_balls : ℕ := 20
def yellow_balls : ℕ := 10
def red_balls : ℕ := 17
def purple_balls : ℕ := 3

def number_of_non_red_or_purple_balls : ℕ := total_balls - (red_balls + purple_balls)

def probability_of_non_red_or_purple : ℚ := number_of_non_red_or_purple_balls / total_balls

theorem probability_calculation_correct :
  probability_of_non_red_or_purple = 0.8 := 
  by 
    -- proof goes here
    sorry

end NUMINAMATH_GPT_probability_calculation_correct_l1905_190594


namespace NUMINAMATH_GPT_yellow_red_chair_ratio_l1905_190531

variable (Y B : ℕ)
variable (red_chairs : ℕ := 5)
variable (total_chairs : ℕ := 43)

-- Condition: There are 2 fewer blue chairs than yellow chairs
def blue_chairs_condition : Prop := B = Y - 2

-- Condition: Total number of chairs
def total_chairs_condition : Prop := red_chairs + Y + B = total_chairs

-- Prove the ratio of yellow chairs to red chairs is 4:1
theorem yellow_red_chair_ratio (h1 : blue_chairs_condition Y B) (h2 : total_chairs_condition Y B) :
  (Y / red_chairs) = 4 := 
sorry

end NUMINAMATH_GPT_yellow_red_chair_ratio_l1905_190531


namespace NUMINAMATH_GPT_fraction_meaningful_iff_nonzero_l1905_190596

theorem fraction_meaningful_iff_nonzero (x : ℝ) : (∃ y : ℝ, y = 1 / x) ↔ x ≠ 0 :=
by sorry

end NUMINAMATH_GPT_fraction_meaningful_iff_nonzero_l1905_190596


namespace NUMINAMATH_GPT_minimum_value_problem_l1905_190587

theorem minimum_value_problem (x y : ℝ) (h1 : 0 < x) (h2 : x < 1) (h3 : 0 < y) (h4 : y < 1) (h5 : x * y = 1 / 2) : 
  ∃ m : ℝ, m = 10 ∧ ∀ z, z = (2 / (1 - x) + 1 / (1 - y)) → z ≥ m :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_problem_l1905_190587


namespace NUMINAMATH_GPT_constant_term_binomial_expansion_l1905_190570

theorem constant_term_binomial_expansion : ∃ T, (∀ x : ℝ, T = (2 * x - 1 / (2 * x)) ^ 6) ∧ T = -20 := 
by
  sorry

end NUMINAMATH_GPT_constant_term_binomial_expansion_l1905_190570


namespace NUMINAMATH_GPT_exists_ints_for_inequalities_l1905_190519

theorem exists_ints_for_inequalities (a b : ℝ) (ε : ℝ) (hε : ε > 0) :
  ∃ (n : ℕ) (k m : ℤ), |(n * a) - k| < ε ∧ |(n * b) - m| < ε :=
by
  sorry

end NUMINAMATH_GPT_exists_ints_for_inequalities_l1905_190519


namespace NUMINAMATH_GPT_exist_distinct_xy_divisibility_divisibility_implies_equality_l1905_190581

-- Part (a)
theorem exist_distinct_xy_divisibility (n : ℕ) (h_n : n > 0) :
  ∃ (x y : ℕ), x ≠ y ∧ (∀ j : ℕ, 1 ≤ j ∧ j ≤ n → (x + j) ∣ (y + j)) :=
sorry

-- Part (b)
theorem divisibility_implies_equality (x y : ℕ) (h : ∀ j : ℕ, (x + j) ∣ (y + j)) : 
  x = y :=
sorry

end NUMINAMATH_GPT_exist_distinct_xy_divisibility_divisibility_implies_equality_l1905_190581


namespace NUMINAMATH_GPT_eq1_solution_eq2_solution_eq3_solution_eq4_solution_l1905_190552

-- Equation 1: 3x^2 - 2x - 1 = 0
theorem eq1_solution (x : ℝ) : 3 * x ^ 2 - 2 * x - 1 = 0 ↔ (x = -1/3 ∨ x = 1) :=
by sorry

-- Equation 2: (y + 1)^2 - 4 = 0
theorem eq2_solution (y : ℝ) : (y + 1) ^ 2 - 4 = 0 ↔ (y = 1 ∨ y = -3) :=
by sorry

-- Equation 3: t^2 - 6t - 7 = 0
theorem eq3_solution (t : ℝ) : t ^ 2 - 6 * t - 7 = 0 ↔ (t = 7 ∨ t = -1) :=
by sorry

-- Equation 4: m(m + 3) - 2m = 0
theorem eq4_solution (m : ℝ) : m * (m + 3) - 2 * m = 0 ↔ (m = 0 ∨ m = -1) :=
by sorry

end NUMINAMATH_GPT_eq1_solution_eq2_solution_eq3_solution_eq4_solution_l1905_190552
