import Mathlib

namespace NUMINAMATH_GPT_total_books_l122_12239

def books_per_shelf : ℕ := 78
def number_of_shelves : ℕ := 15

theorem total_books : books_per_shelf * number_of_shelves = 1170 := 
by
  sorry

end NUMINAMATH_GPT_total_books_l122_12239


namespace NUMINAMATH_GPT_latest_start_time_is_correct_l122_12251

noncomputable def doughComingToRoomTemp : ℕ := 1  -- 1 hour
noncomputable def shapingDough : ℕ := 15         -- 15 minutes
noncomputable def proofingDough : ℕ := 2         -- 2 hours
noncomputable def bakingBread : ℕ := 30          -- 30 minutes
noncomputable def coolingBread : ℕ := 15         -- 15 minutes
noncomputable def bakeryOpeningTime : ℕ := 6     -- 6:00 am

-- Total preparation time in minutes
noncomputable def totalPreparationTimeInMinutes : ℕ :=
  (doughComingToRoomTemp * 60) + shapingDough + (proofingDough * 60) + bakingBread + coolingBread

-- Total preparation time in hours
noncomputable def totalPreparationTimeInHours : ℕ :=
  totalPreparationTimeInMinutes / 60

-- Latest time the baker can start working
noncomputable def latestTimeBakerCanStart : ℕ :=
  if (bakeryOpeningTime - totalPreparationTimeInHours) < 0 then 24 + (bakeryOpeningTime - totalPreparationTimeInHours)
  else bakeryOpeningTime - totalPreparationTimeInHours

theorem latest_start_time_is_correct : latestTimeBakerCanStart = 2 := by
  sorry

end NUMINAMATH_GPT_latest_start_time_is_correct_l122_12251


namespace NUMINAMATH_GPT_distance_between_cities_A_B_l122_12299

-- Define the problem parameters
def train_1_speed : ℝ := 60 -- km/hr
def train_2_speed : ℝ := 75 -- km/hr
def start_time_train_1 : ℝ := 8 -- 8 a.m.
def start_time_train_2 : ℝ := 9 -- 9 a.m.
def meeting_time : ℝ := 12 -- 12 p.m.

-- Define the times each train travels
def hours_train_1_travelled := meeting_time - start_time_train_1
def hours_train_2_travelled := meeting_time - start_time_train_2

-- Calculate the distances covered by each train
def distance_train_1_cover := train_1_speed * hours_train_1_travelled
def distance_train_2_cover := train_2_speed * hours_train_2_travelled

-- Define the total distance between cities A and B
def distance_AB := distance_train_1_cover + distance_train_2_cover

-- The theorem to be proved
theorem distance_between_cities_A_B : distance_AB = 465 := 
  by
    -- placeholder for the proof
    sorry

end NUMINAMATH_GPT_distance_between_cities_A_B_l122_12299


namespace NUMINAMATH_GPT_investment_of_c_l122_12231

theorem investment_of_c (P_b P_a P_c C_a C_b C_c : ℝ)
  (h1 : P_b = 2000) 
  (h2 : P_a - P_c = 799.9999999999998)
  (h3 : C_a = 8000)
  (h4 : C_b = 10000)
  (h5 : P_b / C_b = P_a / C_a)
  (h6 : P_c / C_c = P_a / C_a)
  : C_c = 4000 :=
by 
  sorry

end NUMINAMATH_GPT_investment_of_c_l122_12231


namespace NUMINAMATH_GPT_average_temperature_problem_l122_12291

variable {T W Th F : ℝ}

theorem average_temperature_problem (h1 : (W + Th + 44) / 3 = 34) (h2 : T = 38) : 
  (T + W + Th) / 3 = 32 := by
  sorry

end NUMINAMATH_GPT_average_temperature_problem_l122_12291


namespace NUMINAMATH_GPT_smallest_value_at_x_5_l122_12250

-- Define the variable x
def x : ℕ := 5

-- Define each expression
def exprA := 8 / x
def exprB := 8 / (x + 2)
def exprC := 8 / (x - 2)
def exprD := x / 8
def exprE := (x + 2) / 8

-- The goal is to prove that exprD yields the smallest value
theorem smallest_value_at_x_5 : exprD = min (min (min exprA exprB) (min exprC exprE)) :=
sorry

end NUMINAMATH_GPT_smallest_value_at_x_5_l122_12250


namespace NUMINAMATH_GPT_find_5_minus_c_l122_12219

theorem find_5_minus_c (c d : ℤ) (h₁ : 5 + c = 6 - d) (h₂ : 3 + d = 8 + c) : 5 - c = 7 := by
  sorry

end NUMINAMATH_GPT_find_5_minus_c_l122_12219


namespace NUMINAMATH_GPT_cos_pi_over_4_minus_alpha_l122_12255

theorem cos_pi_over_4_minus_alpha (α : ℝ) (h : Real.sin (Real.pi / 4 + α) = 2 / 3) :
  Real.cos (Real.pi / 4 - α) = 2 / 3 := 
by
  sorry

end NUMINAMATH_GPT_cos_pi_over_4_minus_alpha_l122_12255


namespace NUMINAMATH_GPT_tom_total_amount_l122_12292

-- Definitions of the initial conditions
def initial_amount : ℕ := 74
def amount_earned : ℕ := 86

-- Main statement to prove
theorem tom_total_amount : initial_amount + amount_earned = 160 := 
by
  -- sorry added to skip the proof
  sorry

end NUMINAMATH_GPT_tom_total_amount_l122_12292


namespace NUMINAMATH_GPT_sum_of_squares_of_rates_l122_12293

theorem sum_of_squares_of_rates :
  ∃ (b j s : ℕ), 3 * b + j + 5 * s = 89 ∧ 4 * b + 3 * j + 2 * s = 106 ∧ b^2 + j^2 + s^2 = 821 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_rates_l122_12293


namespace NUMINAMATH_GPT_inequality_abc_l122_12228

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b) / (a^2 + b^2) + (b + c) / (b^2 + c^2) + (c + a) / (c^2 + a^2) ≤ 1/a + 1/b + 1/c := 
by
  sorry

end NUMINAMATH_GPT_inequality_abc_l122_12228


namespace NUMINAMATH_GPT_max_collisions_l122_12225

-- Define the problem
theorem max_collisions (n : ℕ) (hn : n > 0) : 
  ∃ C : ℕ, C = (n * (n - 1)) / 2 := 
sorry

end NUMINAMATH_GPT_max_collisions_l122_12225


namespace NUMINAMATH_GPT_price_increase_for_1620_profit_maximizing_profit_l122_12276

-- To state the problem, we need to define some variables and the associated conditions.

def cost_price : ℝ := 13
def initial_selling_price : ℝ := 20
def initial_monthly_sales : ℝ := 200
def decrease_in_sales_per_yuan : ℝ := 10
def profit_condition (x : ℝ) : ℝ := (initial_selling_price + x - cost_price) * (initial_monthly_sales - decrease_in_sales_per_yuan * x)
def profit_function (x : ℝ) : ℝ := -(10 * x ^ 2) + (130 * x) + 140

-- Part (1): Prove the price increase x such that the profit is 1620 yuan
theorem price_increase_for_1620_profit :
  ∃ (x : ℝ), profit_condition x = 1620 ∧ (x = 2 ∨ x = 11) :=
sorry

-- Part (2): Prove that the selling price that maximizes profit is 26.5 yuan and max profit is 1822.5 yuan
theorem maximizing_profit :
  ∃ (x : ℝ), (x = 13 / 2) ∧ profit_function (13 / 2) = 3645 / 2 :=
sorry

end NUMINAMATH_GPT_price_increase_for_1620_profit_maximizing_profit_l122_12276


namespace NUMINAMATH_GPT_solution_l122_12216

variable (x y z : ℝ)

noncomputable def problem := 
  ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 →
  x^2 + x * y + y^2 = 48 →
  y^2 + y * z + z^2 = 25 →
  z^2 + z * x + x^2 = 73 →
  x * y + y * z + z * x = 40

theorem solution : problem := by
  intros
  sorry

end NUMINAMATH_GPT_solution_l122_12216


namespace NUMINAMATH_GPT_candidate_function_is_odd_and_increasing_l122_12279

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_increasing_function (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x < f y

def candidate_function (x : ℝ) : ℝ := x * |x|

theorem candidate_function_is_odd_and_increasing :
  is_odd_function candidate_function ∧ is_increasing_function candidate_function :=
by
  sorry

end NUMINAMATH_GPT_candidate_function_is_odd_and_increasing_l122_12279


namespace NUMINAMATH_GPT_f_at_count_l122_12232

def f (a b c : ℕ) : ℕ := (a * b * c) / (Nat.gcd (Nat.gcd a b) c * Nat.lcm (Nat.lcm a b) c)

def is_f_at (n : ℕ) : Prop :=
  ∃ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ x ≤ 60 ∧ y ≤ 60 ∧ z ≤ 60 ∧ f x y z = n

theorem f_at_count : ∃ (n : ℕ), n = 70 ∧ ∀ k, is_f_at k → k ≤ 70 := 
sorry

end NUMINAMATH_GPT_f_at_count_l122_12232


namespace NUMINAMATH_GPT_sports_day_results_l122_12234

-- Conditions and questions
variables (a b c : ℕ)
variables (class1_score class2_score class3_score class4_score : ℕ)

-- Conditions given in the problem
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom a_gt_b_gt_c : a > b ∧ b > c
axiom no_ties : (class1_score ≠ class2_score) ∧ (class2_score ≠ class3_score) ∧ (class3_score ≠ class4_score) ∧ (class1_score ≠ class3_score) ∧ (class1_score ≠ class4_score) ∧ (class2_score ≠ class4_score)
axiom class_scores : class1_score + class2_score + class3_score + class4_score = 40

-- To prove
theorem sports_day_results : a + b + c = 8 ∧ a = 5 :=
by
  sorry

end NUMINAMATH_GPT_sports_day_results_l122_12234


namespace NUMINAMATH_GPT_sin_A_sin_C_eq_3_over_4_triangle_is_equilateral_l122_12288

variable {α : Type*}

-- Part 1
theorem sin_A_sin_C_eq_3_over_4
  (A B C : Real)
  (a b c : Real)
  (h1 : b ^ 2 = a * c)
  (h2 : (Real.cos (A - C)) + (Real.cos B) = 3 / 2) :
  Real.sin A * Real.sin C = 3 / 4 :=
sorry

-- Part 2
theorem triangle_is_equilateral
  (A B C : Real)
  (a b c : Real)
  (h1 : b ^ 2 = a * c)
  (h2 : (Real.cos (A - C)) + (Real.cos B) = 3 / 2) :
  A = B ∧ B = C :=
sorry

end NUMINAMATH_GPT_sin_A_sin_C_eq_3_over_4_triangle_is_equilateral_l122_12288


namespace NUMINAMATH_GPT_solve_equation_l122_12236

theorem solve_equation (a b n : ℕ) (p : ℕ) [hp : Fact (Nat.Prime p)] :
  (a > 0) → (b > 0) → (n > 0) → (a ^ 2013 + b ^ 2013 = p ^ n) ↔ 
  ∃ k : ℕ, a = 2 ^ k ∧ b = 2 ^ k ∧ p = 2 ∧ n = 2013 * k + 1 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l122_12236


namespace NUMINAMATH_GPT_divisible_by_120_l122_12201

theorem divisible_by_120 (n : ℕ) (hn_pos : n > 0) : 120 ∣ n * (n^2 - 1) * (n^2 - 5 * n + 26) := 
by
  sorry

end NUMINAMATH_GPT_divisible_by_120_l122_12201


namespace NUMINAMATH_GPT_fraction_sum_l122_12262

theorem fraction_sum : (3 / 4 : ℚ) + (6 / 9 : ℚ) = 17 / 12 := 
by 
  -- Sorry placeholder to indicate proof is not provided.
  sorry

end NUMINAMATH_GPT_fraction_sum_l122_12262


namespace NUMINAMATH_GPT_negation_of_proposition_l122_12230

theorem negation_of_proposition (a b c : ℝ) (h1 : a + b + c ≥ 0) (h2 : abc ≤ 0) : 
  (∃ x y z : ℝ, (x < 0) ∧ (y < 0) ∧ (x = a ∨ x = b ∨ x = c) ∧ (y = a ∨ y = b ∨ y = c) ∧ (x ≠ y)) →
  ¬(∀ x y z : ℝ, (x < 0 ∨ y < 0 ∨ z < 0) → (x ≠ y → x ≠ z → y ≠ z → (x = a ∨ x = b ∨ x = c) ∧ (y = a ∨ y = b ∨ y = c) ∧ (z = a ∨ z = b ∨ z = c))) :=
sorry

end NUMINAMATH_GPT_negation_of_proposition_l122_12230


namespace NUMINAMATH_GPT_sean_less_points_than_combined_l122_12296

def tobee_points : ℕ := 4
def jay_points : ℕ := tobee_points + 6
def combined_points_tobee_jay : ℕ := tobee_points + jay_points
def total_team_points : ℕ := 26
def sean_points : ℕ := total_team_points - combined_points_tobee_jay

theorem sean_less_points_than_combined : (combined_points_tobee_jay - sean_points) = 2 := by
  sorry

end NUMINAMATH_GPT_sean_less_points_than_combined_l122_12296


namespace NUMINAMATH_GPT_floor_T_value_l122_12254

noncomputable def floor_T : ℝ := 
  let p := (0 : ℝ)
  let q := (0 : ℝ)
  let r := (0 : ℝ)
  let s := (0 : ℝ)
  p + q + r + s

theorem floor_T_value (p q r s : ℝ) (hpq: p^2 + q^2 = 2500) (hrs: r^2 + s^2 = 2500) (hpr: p * r = 1200) (hqs: q * s = 1200) (hpos: p > 0 ∧ q > 0 ∧ r > 0 ∧ s > 0) :
  ∃ T : ℝ, T = p + q + r + s ∧ ⌊T⌋ = 140 := 
  by
  sorry

end NUMINAMATH_GPT_floor_T_value_l122_12254


namespace NUMINAMATH_GPT_problem_statement_l122_12227

-- Defining the properties of the function
def even_function (f : ℝ → ℝ) := ∀ x, f x = f (-x)
def symmetric_about_2 (f : ℝ → ℝ) := ∀ x, f (2 + (2 - x)) = f x

-- Given the function f, even function, and symmetric about line x = 2,
-- and given that f(3) = 3, we need to prove f(-1) = 3.
theorem problem_statement (f : ℝ → ℝ) 
  (h1 : even_function f) 
  (h2 : symmetric_about_2 f) 
  (h3 : f 3 = 3) : 
  f (-1) = 3 := 
sorry

end NUMINAMATH_GPT_problem_statement_l122_12227


namespace NUMINAMATH_GPT_combined_work_rate_l122_12256

-- Define the context and the key variables
variable (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b)

-- State the theorem corresponding to the proof problem
theorem combined_work_rate (h_a : a ≠ 0) (h_b : b ≠ 0) : 
  1/a + 1/b = (a * b) / (a + b) * (1/a * 1/b) :=
sorry

end NUMINAMATH_GPT_combined_work_rate_l122_12256


namespace NUMINAMATH_GPT_ratio_a_f_l122_12267

theorem ratio_a_f (a b c d e f : ℕ)
  (h1 : a / b = 1 / 4)
  (h2 : b / c = 13 / 9)
  (h3 : c / d = 5 / 13)
  (h4 : d / e = 2 / 3)
  (h5 : e / f = 7 / 5) :
  a / f = 7 / 6 := by
  sorry

end NUMINAMATH_GPT_ratio_a_f_l122_12267


namespace NUMINAMATH_GPT_red_notebooks_count_l122_12235

variable (R B : ℕ)

-- Conditions
def cost_condition : Prop := 4 * R + 4 + 3 * B = 37
def count_condition : Prop := R + 2 + B = 12
def blue_notebooks_expr : Prop := B = 10 - R

-- Prove the number of red notebooks
theorem red_notebooks_count : cost_condition R B ∧ count_condition R B ∧ blue_notebooks_expr R B → R = 3 := by
  sorry

end NUMINAMATH_GPT_red_notebooks_count_l122_12235


namespace NUMINAMATH_GPT_sum_first_four_terms_eq_12_l122_12297

noncomputable def a : ℕ → ℤ := sorry -- An arithmetic sequence aₙ

-- Given conditions
axiom h1 : a 2 = 4
axiom h2 : a 1 + a 5 = 4 * a 3 - 4

theorem sum_first_four_terms_eq_12 : (a 1 + a 2 + a 3 + a 4) = 12 := 
by {
  sorry
}

end NUMINAMATH_GPT_sum_first_four_terms_eq_12_l122_12297


namespace NUMINAMATH_GPT_total_number_of_boys_in_class_is_40_l122_12261

theorem total_number_of_boys_in_class_is_40 
  (n : ℕ) (h : 27 - 7 = n / 2):
  n = 40 :=
by
  sorry

end NUMINAMATH_GPT_total_number_of_boys_in_class_is_40_l122_12261


namespace NUMINAMATH_GPT_math_problem_l122_12243

/-
Two mathematicians take a morning coffee break each day.
They arrive at the cafeteria independently, at random times between 9 a.m. and 10:30 a.m.,
and stay for exactly m minutes.
Given the probability that either one arrives while the other is in the cafeteria is 30%,
and m = a - b√c, where a, b, and c are positive integers, and c is not divisible by the square of any prime,
prove that a + b + c = 127.

-/

noncomputable def is_square_free (c : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → p * p ∣ c → False

theorem math_problem
  (m a b c : ℕ)
  (h1 : 0 < m)
  (h2 : m = a - b * Real.sqrt c)
  (h3 : is_square_free c)
  (h4 : 30 * (90 * 90) / 100 = (90 - m) * (90 - m)) :
  a + b + c = 127 :=
sorry

end NUMINAMATH_GPT_math_problem_l122_12243


namespace NUMINAMATH_GPT_p_iff_q_l122_12278

def f (x a : ℝ) := x * (x - a) * (x - 2)

def p (a : ℝ) := 0 < a ∧ a < 2

def q (a : ℝ) := 
  let f' x := 3 * x^2 - 2 * (a + 2) * x + 2 * a
  f' a < 0

theorem p_iff_q (a : ℝ) : (p a) ↔ (q a) := by
  sorry

end NUMINAMATH_GPT_p_iff_q_l122_12278


namespace NUMINAMATH_GPT_polynomial_product_is_square_l122_12210

theorem polynomial_product_is_square (x a : ℝ) :
  (x + a) * (x + 2 * a) * (x + 3 * a) * (x + 4 * a) + a^4 = (x^2 + 5 * a * x + 5 * a^2)^2 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_product_is_square_l122_12210


namespace NUMINAMATH_GPT_trains_crossing_time_l122_12244

theorem trains_crossing_time
  (L1 : ℕ) (L2 : ℕ) (T1 : ℕ) (T2 : ℕ)
  (H1 : L1 = 150) (H2 : L2 = 180)
  (H3 : T1 = 10) (H4 : T2 = 15) :
  (L1 + L2) / ((L1 / T1) + (L2 / T2)) = 330 / 27 := sorry

end NUMINAMATH_GPT_trains_crossing_time_l122_12244


namespace NUMINAMATH_GPT_solve_inequalities_l122_12270

theorem solve_inequalities :
  (∀ x : ℝ, x^2 + 3 * x - 10 ≥ 0 ↔ (x ≤ -5 ∨ x ≥ 2)) ∧
  (∀ x : ℝ, x^2 - 3 * x - 2 ≤ 0 ↔ (3 - Real.sqrt 17) / 2 ≤ x ∧ x ≤ (3 + Real.sqrt 17) / 2) :=
by
  sorry

end NUMINAMATH_GPT_solve_inequalities_l122_12270


namespace NUMINAMATH_GPT_min_height_regular_quadrilateral_pyramid_l122_12277

theorem min_height_regular_quadrilateral_pyramid (r : ℝ) (a : ℝ) (h : 2 * r < a / 2) : 
  ∃ x : ℝ, (0 < x) ∧ (∃ V : ℝ, ∀ x' : ℝ, V = (a^2 * x) / 3 ∧ (∀ x' ≠ x, V < (a^2 * x') / 3)) ∧ x = (r * (5 + Real.sqrt 17)) / 2 :=
sorry

end NUMINAMATH_GPT_min_height_regular_quadrilateral_pyramid_l122_12277


namespace NUMINAMATH_GPT_no_base6_digit_d_divisible_by_7_l122_12260

theorem no_base6_digit_d_divisible_by_7 : 
∀ d : ℕ, (d < 6) → ¬ (654 + 42 * d) % 7 = 0 :=
by
  intro d h
  -- Proof is omitted as requested
  sorry

end NUMINAMATH_GPT_no_base6_digit_d_divisible_by_7_l122_12260


namespace NUMINAMATH_GPT_lightning_distance_l122_12209

/--
Linus observed a flash of lightning and then heard the thunder 15 seconds later.
Given:
- speed of sound: 1088 feet/second
- 1 mile = 5280 feet
Prove that the distance from Linus to the lightning strike is 3.25 miles.
-/
theorem lightning_distance (time_seconds : ℕ) (speed_sound : ℕ) (feet_per_mile : ℕ) (distance_miles : ℚ) :
  time_seconds = 15 →
  speed_sound = 1088 →
  feet_per_mile = 5280 →
  distance_miles = 3.25 :=
by
  sorry

end NUMINAMATH_GPT_lightning_distance_l122_12209


namespace NUMINAMATH_GPT_river_bank_bottom_width_l122_12247

/-- 
The cross-section of a river bank is a trapezium with a 12 m wide top and 
a certain width at the bottom. The area of the cross-section is 500 sq m 
and the depth is 50 m. Prove that the width at the bottom is 8 m.
-/
theorem river_bank_bottom_width (area height top_width : ℝ) (h_area: area = 500) 
(h_height : height = 50) (h_top_width : top_width = 12) : ∃ b : ℝ, (1 / 2) * (top_width + b) * height = area ∧ b = 8 :=
by
  use 8
  sorry

end NUMINAMATH_GPT_river_bank_bottom_width_l122_12247


namespace NUMINAMATH_GPT_complement_of_A_in_U_l122_12226

-- Define the universal set U as the set of integers
def U : Set ℤ := Set.univ

-- Define the set A as the set of odd integers
def A : Set ℤ := {x : ℤ | ∃ k : ℤ, x = 2 * k + 1}

-- Define the complement of A in U
def complement_A : Set ℤ := U \ A

-- State the equivalence to be proved
theorem complement_of_A_in_U :
  complement_A = {x : ℤ | ∃ k : ℤ, x = 2 * k} :=
by
  sorry

end NUMINAMATH_GPT_complement_of_A_in_U_l122_12226


namespace NUMINAMATH_GPT_elysse_bags_per_trip_l122_12253

-- Definitions from the problem conditions
def total_bags : ℕ := 30
def total_trips : ℕ := 5
def bags_per_trip : ℕ := total_bags / total_trips

def carries_same_amount (elysse_bags brother_bags : ℕ) : Prop := elysse_bags = brother_bags

-- Statement to prove
theorem elysse_bags_per_trip :
  ∀ (elysse_bags brother_bags : ℕ), 
  bags_per_trip = elysse_bags + brother_bags → 
  carries_same_amount elysse_bags brother_bags → 
  elysse_bags = 3 := 
by 
  intros elysse_bags brother_bags h1 h2
  sorry

end NUMINAMATH_GPT_elysse_bags_per_trip_l122_12253


namespace NUMINAMATH_GPT_prove_p_false_and_q_true_l122_12205

variables (p q : Prop)

theorem prove_p_false_and_q_true (h1 : p ∨ q) (h2 : ¬p) : ¬p ∧ q :=
by {
  -- proof placeholder
  sorry
}

end NUMINAMATH_GPT_prove_p_false_and_q_true_l122_12205


namespace NUMINAMATH_GPT_greatest_difference_areas_l122_12294

theorem greatest_difference_areas (l w l' w' : ℕ) (h₁ : 2*l + 2*w = 120) (h₂ : 2*l' + 2*w' = 120) : 
  l * w ≤ 900 ∧ (l = 30 → w = 30) ∧ l' * w' ≤ 900 ∧ (l' = 30 → w' = 30)  → 
  ∃ (A₁ A₂ : ℕ), (A₁ = l * w ∧ A₂ = l' * w') ∧ (841 = l * w - l' * w') := 
sorry

end NUMINAMATH_GPT_greatest_difference_areas_l122_12294


namespace NUMINAMATH_GPT_value_of_k_l122_12284

theorem value_of_k (x y k : ℝ) (h1 : 4 * x - 3 * y = k) (h2 : 2 * x + 3 * y = 5) (h3 : x = y) : k = 1 :=
sorry

end NUMINAMATH_GPT_value_of_k_l122_12284


namespace NUMINAMATH_GPT_sum_of_digits_l122_12298

theorem sum_of_digits (x : ℕ) (hx : 1 ≤ x ∧ x ≤ 9) (h : 10 * x + 6 * x = 16) : x + 6 * x = 7 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_sum_of_digits_l122_12298


namespace NUMINAMATH_GPT_simplify_polynomial_l122_12207

variable {R : Type} [CommRing R] (s : R)

theorem simplify_polynomial :
  (2 * s^2 + 5 * s - 3) - (2 * s^2 + 9 * s - 4) = -4 * s + 1 :=
by
  sorry

end NUMINAMATH_GPT_simplify_polynomial_l122_12207


namespace NUMINAMATH_GPT_sum_of_x_y_l122_12290

theorem sum_of_x_y (x y : ℚ) (h1 : 1/x + 1/y = 5) (h2 : 1/x - 1/y = -9) : x + y = -5/14 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_x_y_l122_12290


namespace NUMINAMATH_GPT_true_product_of_two_digit_number_l122_12211

theorem true_product_of_two_digit_number (a b : ℕ) (h1 : b = 2 * a) (h2 : 136 * (10 * b + a) = 136 * (10 * a + b) + 1224) : 136 * (10 * a + b) = 1632 := 
by sorry

end NUMINAMATH_GPT_true_product_of_two_digit_number_l122_12211


namespace NUMINAMATH_GPT_average_additional_minutes_per_day_l122_12248

def daily_differences : List ℤ := [20, 5, -5, 0, 15, -10, 10]

theorem average_additional_minutes_per_day :
  (List.sum daily_differences / daily_differences.length) = 5 := by
  sorry

end NUMINAMATH_GPT_average_additional_minutes_per_day_l122_12248


namespace NUMINAMATH_GPT_company_hired_22_additional_males_l122_12286

theorem company_hired_22_additional_males
  (E M : ℕ) 
  (initial_percentage_female : ℝ)
  (final_total_employees : ℕ)
  (final_percentage_female : ℝ)
  (initial_female_count : initial_percentage_female * E = 0.6 * E)
  (final_employee_count : E + M = 264) 
  (final_female_count : initial_percentage_female * E = final_percentage_female * (E + M)) :
  M = 22 := 
by
  sorry

end NUMINAMATH_GPT_company_hired_22_additional_males_l122_12286


namespace NUMINAMATH_GPT_sum_of_solutions_l122_12259

theorem sum_of_solutions (x1 x2 : ℝ) (h1 : (x1 - 2)^2 = 81) (h2 : (x2 - 2)^2 = 81) :
  x1 + x2 = 4 := by
  sorry

end NUMINAMATH_GPT_sum_of_solutions_l122_12259


namespace NUMINAMATH_GPT_permutations_containing_substring_l122_12249

open Nat

/-- Prove that the number of permutations of the string "000011112222" that contain the substring "2020" is equal to 3575. -/
theorem permutations_containing_substring :
  let total_permutations := factorial 8 / (factorial 2 * factorial 4 * factorial 2)
  let num_positions := 9
  let non_overlap_count := total_permutations * num_positions
  let overlap_subtract := 7 * (factorial 6 / (factorial 1 * factorial 4 * factorial 1))
  let add_back := 5 * (factorial 4 / factorial 4)
  non_overlap_count - overlap_subtract + add_back = 3575 := 
by
  let total_permutations := factorial 8 / (factorial 2 * factorial 4 * factorial 2)
  let num_positions := 9
  let non_overlap_count := total_permutations * num_positions
  let overlap_subtract := 7 * (factorial 6 / (factorial 1 * factorial 4 * factorial 1))
  let add_back := 5 * (factorial 4 / factorial 4)
  have h: non_overlap_count - overlap_subtract + add_back = 3575 := by sorry
  exact h

end NUMINAMATH_GPT_permutations_containing_substring_l122_12249


namespace NUMINAMATH_GPT_geometric_sequence_property_l122_12224

noncomputable def S_n (n : ℕ) (a_n : ℕ → ℕ) : ℕ := 3 * 2^n - 3

noncomputable def a_n (n : ℕ) : ℕ := 3 * 2^(n-1)

noncomputable def b_n (n : ℕ) : ℕ := 2^(n-1)

noncomputable def T_n (n : ℕ) : ℕ := 2^n - 1

theorem geometric_sequence_property (n : ℕ) (hn : n ≥ 0) :
  T_n n < b_n (n+1) :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_property_l122_12224


namespace NUMINAMATH_GPT_garden_area_increase_l122_12223

-- Definitions derived directly from the conditions
def length := 50
def width := 10
def perimeter := 2 * (length + width)
def side_length_square := perimeter / 4
def area_rectangle := length * width
def area_square := side_length_square * side_length_square

-- The proof statement
theorem garden_area_increase :
  area_square - area_rectangle = 400 := 
by
  sorry

end NUMINAMATH_GPT_garden_area_increase_l122_12223


namespace NUMINAMATH_GPT_floor_sqrt_n_sqrt_n_plus1_eq_floor_sqrt_4n_plus_2_l122_12220

theorem floor_sqrt_n_sqrt_n_plus1_eq_floor_sqrt_4n_plus_2 (n : ℕ) (hn : n > 0) :
  Int.floor (Real.sqrt n + Real.sqrt (n + 1)) = Int.floor (Real.sqrt (4 * n + 2)) := 
  sorry

end NUMINAMATH_GPT_floor_sqrt_n_sqrt_n_plus1_eq_floor_sqrt_4n_plus_2_l122_12220


namespace NUMINAMATH_GPT_am_hm_inequality_l122_12282

theorem am_hm_inequality (a1 a2 a3 : ℝ) (h1 : 0 < a1) (h2 : 0 < a2) (h3 : 0 < a3) (h_sum : a1 + a2 + a3 = 1) : 
  (1 / a1) + (1 / a2) + (1 / a3) ≥ 9 :=
by
  sorry

end NUMINAMATH_GPT_am_hm_inequality_l122_12282


namespace NUMINAMATH_GPT_micrometer_conversion_l122_12240

theorem micrometer_conversion :
  (0.01 * (1 * 10 ^ (-6))) = (1 * 10 ^ (-8)) :=
by 
  -- sorry is used to skip the actual proof but ensure the theorem is recognized
  sorry

end NUMINAMATH_GPT_micrometer_conversion_l122_12240


namespace NUMINAMATH_GPT_rotation_problem_l122_12203

-- Define the coordinates of the points in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the triangles with given vertices
def P : Point := {x := 0, y := 0}
def Q : Point := {x := 0, y := 13}
def R : Point := {x := 17, y := 0}

def P' : Point := {x := 34, y := 26}
def Q' : Point := {x := 46, y := 26}
def R' : Point := {x := 34, y := 0}

-- Rotation parameters
variables (n : ℝ) (x y : ℝ) (h₀ : 0 < n) (h₁ : n < 180)

-- The mathematical proof problem
theorem rotation_problem :
  n + x + y = 180 := by
  sorry

end NUMINAMATH_GPT_rotation_problem_l122_12203


namespace NUMINAMATH_GPT_cylinder_radius_l122_12258

theorem cylinder_radius
  (r₁ r₂ : ℝ)
  (rounds₁ rounds₂ : ℕ)
  (H₁ : r₁ = 14)
  (H₂ : rounds₁ = 70)
  (H₃ : rounds₂ = 49)
  (L₁ : rounds₁ * 2 * Real.pi * r₁ = rounds₂ * 2 * Real.pi * r₂) :
  r₂ = 20 := 
sorry

end NUMINAMATH_GPT_cylinder_radius_l122_12258


namespace NUMINAMATH_GPT_unoccupied_volume_l122_12212

/--
Given:
1. Three congruent cones, each with a radius of 8 cm and a height of 8 cm.
2. The cones are enclosed within a cylinder such that the bases of two cones are at each base of the cylinder, and one cone is inverted in the middle touching the other two cones at their vertices.
3. The height of the cylinder is 16 cm.

Prove:
The volume of the cylinder not occupied by the cones is 512π cubic cm.
-/
theorem unoccupied_volume 
  (r h : ℝ) 
  (hr : r = 8) 
  (hh_cone : h = 8) 
  (hh_cyl : h_cyl = 16) 
  : (π * r^2 * h_cyl) - (3 * (1/3 * π * r^2 * h)) = 512 * π := 
by 
  sorry

end NUMINAMATH_GPT_unoccupied_volume_l122_12212


namespace NUMINAMATH_GPT_kindergarten_library_models_l122_12263

theorem kindergarten_library_models
  (paid : ℕ)
  (reduced_price : ℕ)
  (models_total_gt_5 : ℕ)
  (bought : ℕ) 
  (condition : paid = 570 ∧ reduced_price = 95 ∧ models_total_gt_5 > 5 ∧ bought = 3 * (2 : ℕ)) :
  exists x : ℕ, bought / 3 = x ∧ x = 2 :=
by
  sorry

end NUMINAMATH_GPT_kindergarten_library_models_l122_12263


namespace NUMINAMATH_GPT_solve_fraction_l122_12222

theorem solve_fraction : (3.242 * 10) / 100 = 0.3242 := by
  sorry

end NUMINAMATH_GPT_solve_fraction_l122_12222


namespace NUMINAMATH_GPT_distinct_meals_l122_12271

def num_entrees : ℕ := 4
def num_drinks : ℕ := 2
def num_desserts : ℕ := 2

theorem distinct_meals : num_entrees * num_drinks * num_desserts = 16 := by
  sorry

end NUMINAMATH_GPT_distinct_meals_l122_12271


namespace NUMINAMATH_GPT_slope_of_line_inclination_l122_12245

theorem slope_of_line_inclination (α : ℝ) (h1 : 0 ≤ α) (h2 : α < 180) 
  (h3 : Real.tan (α * Real.pi / 180) = Real.sqrt 3 / 3) : α = 30 :=
by
  sorry

end NUMINAMATH_GPT_slope_of_line_inclination_l122_12245


namespace NUMINAMATH_GPT_triangle_angle_split_l122_12218

-- Conditions
variables (A B C C1 C2 : ℝ)
-- Axioms/Assumptions
axiom angle_order : A < B
axiom angle_partition : A + C1 = 90 ∧ B + C2 = 90

-- The theorem to prove
theorem triangle_angle_split : C1 - C2 = B - A :=
by {
  sorry
}

end NUMINAMATH_GPT_triangle_angle_split_l122_12218


namespace NUMINAMATH_GPT_inequality_proof_equality_condition_l122_12269

theorem inequality_proof (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ( (3 * a * b * c / (a * b + a * c + b * c)) ^ (a^2 + b^2 + c^2) ) ≥ (a ^ (b * c) * b ^ (a * c) * c ^ (a * b)) := 
sorry

theorem equality_condition (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ( (3 * a * b * c / (a * b + a * c + b * c)) ^ (a^2 + b^2 + c^2) ) = (a ^ (b * c) * b ^ (a * c) * c ^ (a * b)) ↔ a = b ∧ b = c := 
sorry

end NUMINAMATH_GPT_inequality_proof_equality_condition_l122_12269


namespace NUMINAMATH_GPT_simplification_evaluation_l122_12215

theorem simplification_evaluation (x : ℝ) (h : x = Real.sqrt 5 + 2) :
  (x + 2) / (x - 1) / (x + 1 - 3 / (x - 1)) = Real.sqrt 5 / 5 :=
by
  sorry

end NUMINAMATH_GPT_simplification_evaluation_l122_12215


namespace NUMINAMATH_GPT_length_LM_in_triangle_l122_12275

theorem length_LM_in_triangle 
  (A B C K L M : Type*) 
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace K] [MetricSpace L] [MetricSpace M]
  (angle_A: Real) (angle_B: Real) (angle_C: Real)
  (AK: Real) (BL: Real) (MC: Real) (KL: Real) (KM: Real)
  (H1: angle_A = 90) (H2: angle_B = 30) (H3: angle_C = 60) 
  (H4: AK = 4) (H5: BL = 31) (H6: MC = 3) 
  (H7: KL = KM) : 
  (LM = 20) :=
sorry

end NUMINAMATH_GPT_length_LM_in_triangle_l122_12275


namespace NUMINAMATH_GPT_hilary_regular_toenails_in_jar_l122_12202

-- Conditions
def jar_capacity : Nat := 100
def big_toenail_size : Nat := 2
def num_big_toenails : Nat := 20
def remaining_regular_toenails_space : Nat := 20

-- Question & Answer
theorem hilary_regular_toenails_in_jar : 
  (jar_capacity - remaining_regular_toenails_space - (num_big_toenails * big_toenail_size)) = 40 :=
by
  sorry

end NUMINAMATH_GPT_hilary_regular_toenails_in_jar_l122_12202


namespace NUMINAMATH_GPT_mean_combined_set_l122_12238

noncomputable def mean (s : Finset ℚ) : ℚ :=
  (s.sum id) / s.card

theorem mean_combined_set :
  ∀ (s1 s2 : Finset ℚ),
  s1.card = 7 →
  s2.card = 8 →
  mean s1 = 15 →
  mean s2 = 18 →
  mean (s1 ∪ s2) = 249 / 15 :=
by
  sorry

end NUMINAMATH_GPT_mean_combined_set_l122_12238


namespace NUMINAMATH_GPT_inverse_variation_z_x_square_l122_12281

theorem inverse_variation_z_x_square (x z : ℝ) (K : ℝ) 
  (h₀ : z * x^2 = K) 
  (h₁ : x = 3 ∧ z = 2)
  (h₂ : z = 8) :
  x = 3 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_inverse_variation_z_x_square_l122_12281


namespace NUMINAMATH_GPT_average_student_age_before_leaving_l122_12246

theorem average_student_age_before_leaving
  (A : ℕ)
  (student_count : ℕ := 30)
  (leaving_student_age : ℕ := 11)
  (teacher_age : ℕ := 41)
  (new_avg_age : ℕ := 11)
  (new_total_students : ℕ := 30)
  (initial_total_age : ℕ := 30 * A)
  (remaining_students : ℕ := 29)
  (total_age_after_leaving : ℕ := initial_total_age - leaving_student_age)
  (total_age_including_teacher : ℕ := total_age_after_leaving + teacher_age) :
  total_age_including_teacher / new_total_students = new_avg_age → A = 10 := 
  by
    intros h
    sorry

end NUMINAMATH_GPT_average_student_age_before_leaving_l122_12246


namespace NUMINAMATH_GPT_probability_exactly_two_even_dice_l122_12266

theorem probability_exactly_two_even_dice :
  let p_even := 1 / 2
  let p_not_even := 1 / 2
  let number_of_ways := 3
  let probability_each_way := (p_even * p_even * p_not_even)
  3 * probability_each_way = 3 / 8 :=
by
  sorry

end NUMINAMATH_GPT_probability_exactly_two_even_dice_l122_12266


namespace NUMINAMATH_GPT_percent_increase_in_maintenance_time_l122_12233

theorem percent_increase_in_maintenance_time (original_time new_time : ℝ) (h1 : original_time = 25) (h2 : new_time = 30) : 
  ((new_time - original_time) / original_time) * 100 = 20 :=
by
  sorry

end NUMINAMATH_GPT_percent_increase_in_maintenance_time_l122_12233


namespace NUMINAMATH_GPT_license_plate_combinations_l122_12242

theorem license_plate_combinations :
  let num_consonants := 21
  let num_vowels := 5
  let num_digits := 10
  num_consonants^2 * num_vowels^2 * num_digits = 110250 :=
by
  let num_consonants := 21
  let num_vowels := 5
  let num_digits := 10
  sorry

end NUMINAMATH_GPT_license_plate_combinations_l122_12242


namespace NUMINAMATH_GPT_peg_arrangement_l122_12272

theorem peg_arrangement :
  let Y := 5
  let R := 4
  let G := 3
  let B := 2
  let O := 1
  (Y! * R! * G! * B! * O!) = 34560 :=
by
  sorry

end NUMINAMATH_GPT_peg_arrangement_l122_12272


namespace NUMINAMATH_GPT_tiles_needed_l122_12213

/-- 
Given:
- The cafeteria is tiled with the same floor tiles.
- It takes 630 tiles to cover an area of 18 square decimeters of tiles.
- We switch to square tiles with a side length of 6 decimeters.

Prove:
- The number of new tiles needed to cover the same area is 315.
--/
theorem tiles_needed (n_tiles : ℕ) (area_per_tile : ℕ) (new_tile_side_length : ℕ) 
  (h1 : n_tiles = 630) (h2 : area_per_tile = 18) (h3 : new_tile_side_length = 6) :
  (630 * 18) / (6 * 6) = 315 :=
by
  sorry

end NUMINAMATH_GPT_tiles_needed_l122_12213


namespace NUMINAMATH_GPT_empty_set_l122_12237

def setA := {x : ℝ | x^2 - 4 = 0}
def setB := {x : ℝ | x > 9 ∨ x < 3}
def setC := {p : ℝ × ℝ | p.1^2 + p.2^2 = 0}
def setD := {x : ℝ | x > 9 ∧ x < 3}

theorem empty_set : setD = ∅ := 
  sorry

end NUMINAMATH_GPT_empty_set_l122_12237


namespace NUMINAMATH_GPT_trains_pass_time_l122_12206

def length_train1 : ℕ := 200
def length_train2 : ℕ := 280

def speed_train1_kmph : ℕ := 42
def speed_train2_kmph : ℕ := 30

def kmph_to_mps (speed_kmph : ℕ) : ℚ :=
  speed_kmph * 1000 / 3600

def relative_speed_mps : ℚ :=
  kmph_to_mps (speed_train1_kmph + speed_train2_kmph)

def total_length : ℕ :=
  length_train1 + length_train2

def time_to_pass_trains : ℚ :=
  total_length / relative_speed_mps

theorem trains_pass_time :
  time_to_pass_trains = 24 := by
  sorry

end NUMINAMATH_GPT_trains_pass_time_l122_12206


namespace NUMINAMATH_GPT_fraction_equality_l122_12273

theorem fraction_equality
  (a b : ℝ)
  (x : ℝ)
  (h1 : x = (a^2) / (b^2))
  (h2 : a ≠ b)
  (h3 : b ≠ 0) :
  (a^2 + b^2) / (a^2 - b^2) = (x + 1) / (x - 1) :=
by
  sorry

end NUMINAMATH_GPT_fraction_equality_l122_12273


namespace NUMINAMATH_GPT_problem_statement_l122_12264

theorem problem_statement (a b c : ℝ) (h_nonneg_a : a ≥ 0) (h_nonneg_b : b ≥ 0) (h_nonneg_c : c ≥ 0) (h_condition : a * b + b * c + c * a = 1 / 3) :
  1 / (a^2 - b * c + 1) + 1 / (b^2 - c * a + 1) + 1 / (c^2 - a * b + 1) ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l122_12264


namespace NUMINAMATH_GPT_transaction_loss_l122_12274

theorem transaction_loss :
  let house_sale_price := 10000
  let store_sale_price := 15000
  let house_loss_percentage := 0.25
  let store_gain_percentage := 0.25
  let h := house_sale_price / (1 - house_loss_percentage)
  let s := store_sale_price / (1 + store_gain_percentage)
  let total_cost_price := h + s
  let total_selling_price := house_sale_price + store_sale_price
  let difference := total_selling_price - total_cost_price
  difference = -1000 / 3 :=
by
  sorry

end NUMINAMATH_GPT_transaction_loss_l122_12274


namespace NUMINAMATH_GPT_jasper_drinks_more_than_hot_dogs_l122_12200

-- Definition of conditions based on the problem
def bags_of_chips := 27
def fewer_hot_dogs_than_chips := 8
def drinks_sold := 31

-- Definition to compute the number of hot dogs
def hot_dogs_sold := bags_of_chips - fewer_hot_dogs_than_chips

-- Lean 4 statement to prove the final result
theorem jasper_drinks_more_than_hot_dogs : drinks_sold - hot_dogs_sold = 12 :=
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_jasper_drinks_more_than_hot_dogs_l122_12200


namespace NUMINAMATH_GPT_measure_of_angle_C_l122_12204

variable (C D : ℕ)
variable (h1 : C + D = 180)
variable (h2 : C = 5 * D)

theorem measure_of_angle_C : C = 150 :=
by
  sorry

end NUMINAMATH_GPT_measure_of_angle_C_l122_12204


namespace NUMINAMATH_GPT_average_income_A_B_l122_12214

theorem average_income_A_B (A B C : ℝ)
  (h1 : (B + C) / 2 = 5250)
  (h2 : (A + C) / 2 = 4200)
  (h3 : A = 3000) : (A + B) / 2 = 4050 :=
by
  sorry

end NUMINAMATH_GPT_average_income_A_B_l122_12214


namespace NUMINAMATH_GPT_m_not_equal_n_possible_l122_12217

-- Define the touching relation on an infinite chessboard
structure Chessboard :=
(colored_square : ℤ × ℤ → Prop)
(touches : ℤ × ℤ → ℤ × ℤ → Prop)

-- Define the properties
def colors_square (board : Chessboard) : Prop :=
∃ i j : ℤ, board.colored_square (i, j) ∧ board.colored_square (i + 1, j + 1)

def black_square_touches_m_black_squares (board : Chessboard) (m : ℕ) : Prop :=
∀ i j : ℤ, board.colored_square (i, j) →
    (board.touches (i, j) (i + 1, j) → board.colored_square (i + 1, j)) ∧ 
    (board.touches (i, j) (i - 1, j) → board.colored_square (i - 1, j)) ∧
    (board.touches (i, j) (i, j + 1) → board.colored_square (i, j + 1)) ∧
    (board.touches (i, j) (i, j - 1) → board.colored_square (i, j - 1))
    -- Add additional conditions to ensure exactly m black squares are touched

def white_square_touches_n_white_squares (board : Chessboard) (n : ℕ) : Prop :=
∀ i j : ℤ, ¬board.colored_square (i, j) →
    (board.touches (i, j) (i + 1, j) → ¬board.colored_square (i + 1, j)) ∧ 
    (board.touches (i, j) (i - 1, j) → ¬board.colored_square (i - 1, j)) ∧
    (board.touches (i, j) (i, j + 1) → ¬board.colored_square (i, j + 1)) ∧
    (board.touches (i, j) (i, j - 1) → ¬board.colored_square (i, j - 1))
    -- Add additional conditions to ensure exactly n white squares are touched

theorem m_not_equal_n_possible (board : Chessboard) (m n : ℕ) :
colors_square board →
black_square_touches_m_black_squares board m →
white_square_touches_n_white_squares board n →
m ≠ n :=
by {
    sorry
}

end NUMINAMATH_GPT_m_not_equal_n_possible_l122_12217


namespace NUMINAMATH_GPT_find_f_of_3_l122_12285

noncomputable def f : ℝ → ℝ :=
  sorry

theorem find_f_of_3 (h : ∀ x : ℝ, x ≠ 0 → f x - 3 * f (1 / x) = 3 ^ x) :
  f 3 = (-27 + 3 * (3 ^ (1 / 3))) / 8 :=
sorry

end NUMINAMATH_GPT_find_f_of_3_l122_12285


namespace NUMINAMATH_GPT_sum_ratio_l122_12252

variable {a_n : ℕ → ℕ}
variable {S_n : ℕ → ℕ}
variable {d : ℕ}

axiom arithmetic_sequence : ∀ n, a_n n = a_n 1 + (n - 1) * d
axiom sum_of_first_n_terms : ∀ n, S_n n = n * (a_n 1 + a_n n) / 2
axiom condition_a4 : a_n 4 = 2 * (a_n 2 + a_n 3)
axiom non_zero_difference : d ≠ 0

theorem sum_ratio : S_n 7 / S_n 4 = 7 / 4 := 
by
  sorry

end NUMINAMATH_GPT_sum_ratio_l122_12252


namespace NUMINAMATH_GPT_ninth_term_l122_12268

variable (a d : ℤ)
variable (h1 : a + 2 * d = 20)
variable (h2 : a + 5 * d = 26)

theorem ninth_term (a d : ℤ) (h1 : a + 2 * d = 20) (h2 : a + 5 * d = 26) : a + 8 * d = 32 :=
sorry

end NUMINAMATH_GPT_ninth_term_l122_12268


namespace NUMINAMATH_GPT_chess_group_players_l122_12265

theorem chess_group_players (n : ℕ) (h : n * (n - 1) / 2 = 91) : n = 14 :=
sorry

end NUMINAMATH_GPT_chess_group_players_l122_12265


namespace NUMINAMATH_GPT_ratio_XY_7_l122_12208

variable (Z : ℕ)
variable (population_Z : ℕ := Z)
variable (population_Y : ℕ := 2 * Z)
variable (population_X : ℕ := 14 * Z)

theorem ratio_XY_7 :
  population_X / population_Y = 7 := by
  sorry

end NUMINAMATH_GPT_ratio_XY_7_l122_12208


namespace NUMINAMATH_GPT_consecutive_numbers_count_l122_12241

theorem consecutive_numbers_count (n x : ℕ) (h_avg : (2 * n * 20 = n * (2 * x + n - 1))) (h_largest : x + n - 1 = 23) : n = 7 :=
by
  sorry

end NUMINAMATH_GPT_consecutive_numbers_count_l122_12241


namespace NUMINAMATH_GPT_zeta_1_8_add_zeta_2_8_add_zeta_3_8_l122_12287

noncomputable def compute_s8 (s : ℕ → ℂ) : ℂ :=
  s 8

theorem zeta_1_8_add_zeta_2_8_add_zeta_3_8 {ζ : ℕ → ℂ} 
  (h1 : ζ 1 + ζ 2 + ζ 3 = 2)
  (h2 : ζ 1^2 + ζ 2^2 + ζ 3^2 = 6)
  (h3 : ζ 1^3 + ζ 2^3 + ζ 3^3 = 18)
  (rec : ∀ n, ζ (n + 3) = 2 * ζ (n + 2) + ζ (n + 1) - (4 / 3) * ζ n)
  (s0 : ζ 0 = 3)
  (s1 : ζ 1 = 2)
  (s2 : ζ 2 = 6)
  (s3 : ζ 3 = 18)
  : ζ 8 = compute_s8 ζ := 
sorry

end NUMINAMATH_GPT_zeta_1_8_add_zeta_2_8_add_zeta_3_8_l122_12287


namespace NUMINAMATH_GPT_quadrilateral_area_l122_12289

theorem quadrilateral_area (a b x : ℝ)
  (h1: ∀ (y z : ℝ), y^2 + z^2 = a^2 ∧ (x + y)^2 + (x + z)^2 = b^2)
  (hx_perp: ∀ (p q : ℝ), x * q = 0 ∧ x * p = 0) :
  S = (1 / 4) * |b^2 - a^2| :=
by
  sorry

end NUMINAMATH_GPT_quadrilateral_area_l122_12289


namespace NUMINAMATH_GPT_find_x_l122_12221

def custom_op (a b : ℝ) : ℝ :=
  a^2 - 3 * b

theorem find_x (x : ℝ) : 
  (custom_op (custom_op 7 x) 3 = 18) ↔ (x = 17.71 ∨ x = 14.96) := 
by
  sorry

end NUMINAMATH_GPT_find_x_l122_12221


namespace NUMINAMATH_GPT_problem_f_f2_equals_16_l122_12280

noncomputable def f (x : ℝ) : ℝ :=
if h : x < 3 then x^2 else 2^x

theorem problem_f_f2_equals_16 : f (f 2) = 16 :=
by
  sorry

end NUMINAMATH_GPT_problem_f_f2_equals_16_l122_12280


namespace NUMINAMATH_GPT_cost_price_radio_l122_12229

theorem cost_price_radio (SP : ℝ) (loss_percentage : ℝ) (C : ℝ) 
  (h1 : SP = 1305) 
  (h2 : loss_percentage = 0.13) 
  (h3 : SP = C * (1 - loss_percentage)) :
  C = 1500 := 
by 
  sorry

end NUMINAMATH_GPT_cost_price_radio_l122_12229


namespace NUMINAMATH_GPT_correct_statement_l122_12295

open Set

variable (U : Set ℕ) (M : Set ℕ)
variables (hU : U = {1, 2, 3, 4, 5}) (hM : U \ M = {1, 3})

theorem correct_statement : 2 ∈ M :=
by
  sorry

end NUMINAMATH_GPT_correct_statement_l122_12295


namespace NUMINAMATH_GPT_fred_games_last_year_proof_l122_12257

def fred_games_last_year (this_year: ℕ) (diff: ℕ) : ℕ := this_year + diff

theorem fred_games_last_year_proof : 
  ∀ (this_year: ℕ) (diff: ℕ),
  this_year = 25 → 
  diff = 11 →
  fred_games_last_year this_year diff = 36 := 
by 
  intros this_year diff h_this_year h_diff
  rw [h_this_year, h_diff]
  sorry

end NUMINAMATH_GPT_fred_games_last_year_proof_l122_12257


namespace NUMINAMATH_GPT_find_a_b_monotonicity_l122_12283

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := (x^2 + a * x + b) / x

theorem find_a_b (a b : ℝ) (h_odd : ∀ x ≠ 0, f (-x) a b = -f x a b) (h_eq : f 1 a b = f 4 a b) :
  a = 0 ∧ b = 4 := by sorry

theorem monotonicity (f : ℝ → ℝ) (h : ∀ x ≠ 0, f x = x + 4 / x) :
  (∀ x1 x2, 0 < x1 ∧ x1 ≤ 2 ∧ x1 < x2 ∧ x2 ≤ 2 → f x1 > f x2) ∧
  (∀ x1 x2, 2 < x1 ∧ x1 < x2 → f x1 < f x2) := by sorry

end NUMINAMATH_GPT_find_a_b_monotonicity_l122_12283
