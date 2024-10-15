import Mathlib

namespace NUMINAMATH_GPT_arithmetic_mean_l1443_144377

theorem arithmetic_mean (x y : ℝ) (h1 : x = Real.sqrt 2 - 1) (h2 : y = 1 / (Real.sqrt 2 - 1)) :
  (x + y) / 2 = Real.sqrt 2 := sorry

end NUMINAMATH_GPT_arithmetic_mean_l1443_144377


namespace NUMINAMATH_GPT_sara_jim_savings_eq_l1443_144311

theorem sara_jim_savings_eq (w : ℕ) : 
  let sara_init_savings := 4100
  let sara_weekly_savings := 10
  let jim_weekly_savings := 15
  (sara_init_savings + sara_weekly_savings * w = jim_weekly_savings * w) → w = 820 :=
by
  intros
  sorry

end NUMINAMATH_GPT_sara_jim_savings_eq_l1443_144311


namespace NUMINAMATH_GPT_point_on_parabola_l1443_144356

theorem point_on_parabola (c m n x1 x2 : ℝ) (h : x1 < x2)
  (hx1 : x1^2 + 2*x1 + c = 0)
  (hx2 : x2^2 + 2*x2 + c = 0)
  (hp : n = m^2 + 2*m + c)
  (hn : n < 0) :
  x1 < m ∧ m < x2 :=
sorry

end NUMINAMATH_GPT_point_on_parabola_l1443_144356


namespace NUMINAMATH_GPT_find_AX_l1443_144398

variable (A B X C : Point)
variable (AB AC BC AX XB : ℝ)
variable (angleACX angleXCB : Angle)
variable (eqAngle : angleACX = angleXCB)

axiom length_AB : AB = 80
axiom length_AC : AC = 36
axiom length_BC : BC = 72

theorem find_AX (AB AC BC AX XB : ℝ) (angleACX angleXCB : Angle)
  (eqAngle : angleACX = angleXCB)
  (h1 : AB = 80)
  (h2 : AC = 36)
  (h3 : BC = 72) : AX = 80 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_AX_l1443_144398


namespace NUMINAMATH_GPT_integer_sequence_unique_l1443_144337

theorem integer_sequence_unique (a : ℕ → ℤ) :
  (∀ n : ℕ, ∃ p q : ℕ, p ≠ q ∧ a p > 0 ∧ a q < 0) ∧
  (∀ n : ℕ, ∀ i j : ℕ, i < n → j < n → i ≠ j → a i % (n : ℤ) ≠ a j % (n : ℤ))
  → ∀ x : ℤ, ∃! i : ℕ, a i = x :=
by
  sorry

end NUMINAMATH_GPT_integer_sequence_unique_l1443_144337


namespace NUMINAMATH_GPT_least_number_subtracted_divisible_by_5_l1443_144361

def subtract_least_number (n : ℕ) (m : ℕ) : ℕ :=
  n % m

theorem least_number_subtracted_divisible_by_5 : subtract_least_number 9671 5 = 1 :=
by
  sorry

end NUMINAMATH_GPT_least_number_subtracted_divisible_by_5_l1443_144361


namespace NUMINAMATH_GPT_power_of_2_multiplication_l1443_144381

theorem power_of_2_multiplication : (16^3) * (4^4) * (32^2) = 2^30 := by
  sorry

end NUMINAMATH_GPT_power_of_2_multiplication_l1443_144381


namespace NUMINAMATH_GPT_cary_wage_after_two_years_l1443_144363

theorem cary_wage_after_two_years (initial_wage raise_percentage cut_percentage : ℝ) (wage_after_first_year wage_after_second_year : ℝ) :
  initial_wage = 10 ∧ raise_percentage = 0.2 ∧ cut_percentage = 0.75 ∧ 
  wage_after_first_year = initial_wage * (1 + raise_percentage) ∧
  wage_after_second_year = wage_after_first_year * cut_percentage → 
  wage_after_second_year = 9 :=
by
  sorry

end NUMINAMATH_GPT_cary_wage_after_two_years_l1443_144363


namespace NUMINAMATH_GPT_ted_age_l1443_144329

variables (t s j : ℕ)

theorem ted_age
  (h1 : t = 2 * s - 20)
  (h2 : j = s + 6)
  (h3 : t + s + j = 90) :
  t = 32 :=
by
  sorry

end NUMINAMATH_GPT_ted_age_l1443_144329


namespace NUMINAMATH_GPT_students_answered_both_correctly_l1443_144357

theorem students_answered_both_correctly (x y z w total : ℕ) (h1 : x = 22) (h2 : y = 20) 
  (h3 : z = 3) (h4 : total = 25) (h5 : x + y - w - z = total) : w = 17 :=
by
  sorry

end NUMINAMATH_GPT_students_answered_both_correctly_l1443_144357


namespace NUMINAMATH_GPT_sum_of_roots_l1443_144392

noncomputable def f (x : ℝ) : ℝ :=
  x^2 - 2016 * x + 2015

theorem sum_of_roots (a b c : ℝ) (h1 : f a = c) (h2 : f b = c) (h3 : a ≠ b) :
  a + b = 2016 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_roots_l1443_144392


namespace NUMINAMATH_GPT_remainder_of_large_number_l1443_144336

theorem remainder_of_large_number :
  (1235678901 % 101) = 1 :=
by
  have h1: (10^8 % 101) = 1 := sorry
  have h2: (10^6 % 101) = 1 := sorry
  have h3: (10^4 % 101) = 1 := sorry
  have h4: (10^2 % 101) = 1 := sorry
  have large_number_decomposition: 1235678901 = 12 * 10^8 + 35 * 10^6 + 67 * 10^4 + 89 * 10^2 + 1 := sorry
  -- Proof using the decomposition and modulo properties
  sorry

end NUMINAMATH_GPT_remainder_of_large_number_l1443_144336


namespace NUMINAMATH_GPT_broken_line_coverable_l1443_144307

noncomputable def cover_broken_line (length_of_line : ℝ) (radius_of_circle : ℝ) : Prop :=
  length_of_line = 5 ∧ radius_of_circle > 1.25

theorem broken_line_coverable :
  ∃ radius_of_circle, cover_broken_line 5 radius_of_circle :=
by sorry

end NUMINAMATH_GPT_broken_line_coverable_l1443_144307


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l1443_144323

noncomputable def hyperbola_eccentricity_range (a b e : ℝ) (h_a_pos : 0 < a) (h_a_less_1 : a < 1) (h_b_pos : 0 < b) : Prop :=
  let c := Real.sqrt ((5 * a^2 - a^4) / (1 - a^2))
  let e := c / a
  e > Real.sqrt 5

theorem hyperbola_eccentricity (a b e : ℝ) (h_a_pos : 0 < a) (h_a_less_1 : a < 1) (h_b_pos : 0 < b) :
  hyperbola_eccentricity_range a b e h_a_pos h_a_less_1 h_b_pos := 
sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l1443_144323


namespace NUMINAMATH_GPT_projectile_highest_point_l1443_144314

noncomputable def highest_point (v w_h w_v θ g : ℝ) : ℝ × ℝ :=
  let t := (v * Real.sin θ + w_v) / g
  let x := (v * t + w_h * t) * Real.cos θ
  let y := (v * t + w_v * t) * Real.sin θ - (1/2) * g * t^2
  (x, y)

theorem projectile_highest_point : highest_point 100 10 (-2) (Real.pi / 4) 9.8 = (561.94, 236) :=
  sorry

end NUMINAMATH_GPT_projectile_highest_point_l1443_144314


namespace NUMINAMATH_GPT_vectors_parallel_l1443_144343

theorem vectors_parallel (x : ℝ) :
    ∀ (a b : ℝ × ℝ × ℝ),
    a = (2, -1, 3) →
    b = (x, 2, -6) →
    (∃ k : ℝ, b = (k * 2, k * -1, k * 3)) →
    x = -4 :=
by
  intro a b ha hb hab
  sorry

end NUMINAMATH_GPT_vectors_parallel_l1443_144343


namespace NUMINAMATH_GPT_function_has_two_zeros_l1443_144384

/-- 
Given the function y = x + 1/(2x) + t has two zeros under the condition t > 0,
prove that the range of the real number t is (-∞, -√2).
-/
theorem function_has_two_zeros (t : ℝ) (ht : t > 0) : t < -Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_function_has_two_zeros_l1443_144384


namespace NUMINAMATH_GPT_distinct_ways_to_place_digits_l1443_144352

theorem distinct_ways_to_place_digits :
  let n := 4 -- number of digits
  let k := 5 -- number of boxes
  (k * (n!)) = 120 := by
  sorry

end NUMINAMATH_GPT_distinct_ways_to_place_digits_l1443_144352


namespace NUMINAMATH_GPT_birds_count_214_l1443_144383

def two_legged_birds_count (b m i : Nat) : Prop :=
  b + m + i = 300 ∧ 2 * b + 4 * m + 3 * i = 686 → b = 214

theorem birds_count_214 (b m i : Nat) : two_legged_birds_count b m i :=
by
  sorry

end NUMINAMATH_GPT_birds_count_214_l1443_144383


namespace NUMINAMATH_GPT_find_f2_l1443_144313

def f (x a b : ℝ) : ℝ := x^5 + a * x^3 + b * x - 8

theorem find_f2 (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -26 :=
sorry

end NUMINAMATH_GPT_find_f2_l1443_144313


namespace NUMINAMATH_GPT_geometric_sequence_a2_a4_sum_l1443_144382

theorem geometric_sequence_a2_a4_sum :
  ∃ (a : ℕ → ℝ) (q : ℝ), (∀ n, a n = a 1 * q ^ (n - 1)) ∧
    (a 2 * a 4 = 9) ∧
    (9 * (a 1 * (1 - q^4) / (1 - q)) = 10 * (a 1 * (1 - q^2) / (1 - q))) ∧
    (a 2 + a 4 = 10) :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_a2_a4_sum_l1443_144382


namespace NUMINAMATH_GPT_total_tiles_to_be_replaced_l1443_144306

-- Define the given conditions
def horizontal_paths : List ℕ := [30, 50, 30, 20, 20, 50]
def vertical_paths : List ℕ := [20, 50, 20, 50, 50]
def intersections : ℕ := List.sum [2, 3, 3, 4, 4]

-- Problem statement: Prove that the total number of tiles to be replaced is 374
theorem total_tiles_to_be_replaced : List.sum horizontal_paths + List.sum vertical_paths - intersections = 374 := 
by sorry

end NUMINAMATH_GPT_total_tiles_to_be_replaced_l1443_144306


namespace NUMINAMATH_GPT_integer_solutions_xy_l1443_144385

theorem integer_solutions_xy :
  ∃ (x y : ℤ), (x + y + x * y = 500) ∧ 
               ((x = 0 ∧ y = 500) ∨ 
                (x = -2 ∧ y = -502) ∨ 
                (x = 2 ∧ y = 166) ∨ 
                (x = -4 ∧ y = -168)) :=
by
  sorry

end NUMINAMATH_GPT_integer_solutions_xy_l1443_144385


namespace NUMINAMATH_GPT_max_value_condition_min_value_condition_l1443_144372

theorem max_value_condition (x : ℝ) (h : x < 0) : (x^2 + x + 1) / x ≤ -1 :=
sorry

theorem min_value_condition (x : ℝ) (h : x > -1) : ((x + 5) * (x + 2)) / (x + 1) ≥ 9 :=
sorry

end NUMINAMATH_GPT_max_value_condition_min_value_condition_l1443_144372


namespace NUMINAMATH_GPT_a_plus_b_eq_neg2_l1443_144318

noncomputable def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x + 14

variable (a b : ℝ)

axiom h1 : f a = 1
axiom h2 : f b = 19

theorem a_plus_b_eq_neg2 : a + b = -2 :=
sorry

end NUMINAMATH_GPT_a_plus_b_eq_neg2_l1443_144318


namespace NUMINAMATH_GPT_probability_point_below_x_axis_l1443_144322

structure Point2D :=
  (x : ℝ)
  (y : ℝ)

structure Parallelogram :=
  (P Q R S : Point2D)

def vertices_of_PQRS : Parallelogram :=
  ⟨⟨4, 4⟩, ⟨-2, -2⟩, ⟨-8, -2⟩, ⟨-2, 4⟩⟩

def point_lies_below_x_axis_probability (parallelogram : Parallelogram) : ℝ :=
  sorry

theorem probability_point_below_x_axis :
  point_lies_below_x_axis_probability vertices_of_PQRS = 1 / 2 :=
sorry

end NUMINAMATH_GPT_probability_point_below_x_axis_l1443_144322


namespace NUMINAMATH_GPT_train_speed_calculation_l1443_144373

variable (p : ℝ) (h_p : p > 0)

/-- The speed calculation of a train that covers 200 meters in p seconds is correctly given by 720 / p km/hr. -/
theorem train_speed_calculation (h_p : p > 0) : (200 / p * 3.6 = 720 / p) :=
by
  sorry

end NUMINAMATH_GPT_train_speed_calculation_l1443_144373


namespace NUMINAMATH_GPT_percentage_of_volume_is_P_l1443_144386

noncomputable def volumeOfSolutionP {P Q : ℝ} (h : 0.80 * P + 0.55 * Q = 0.675 * (P + Q)) : ℝ := 
(P / (P + Q)) * 100

theorem percentage_of_volume_is_P {P Q: ℝ} (h : 0.80 * P + 0.55 * Q = 0.675 * (P + Q)) : 
  volumeOfSolutionP h = 50 :=
sorry

end NUMINAMATH_GPT_percentage_of_volume_is_P_l1443_144386


namespace NUMINAMATH_GPT_sum_of_numbers_is_60_l1443_144319

-- Define the primary values used in the conditions
variables (a b c : ℝ)

-- Define the conditions in the problem
def mean_condition_1 : Prop := (a + b + c) / 3 = a + 20
def mean_condition_2 : Prop := (a + b + c) / 3 = c - 30
def median_condition : Prop := b = 10

-- Prove that the sum of the numbers is 60 given the conditions
theorem sum_of_numbers_is_60 (hac1 : mean_condition_1 a b c) (hac2 : mean_condition_2 a b c) (hbm : median_condition b) : a + b + c = 60 :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_numbers_is_60_l1443_144319


namespace NUMINAMATH_GPT_justine_more_than_bailey_l1443_144328

-- Definitions from conditions
def J : ℕ := 22 -- Justine's initial rubber bands
def B : ℕ := 12 -- Bailey's initial rubber bands

-- Theorem to prove
theorem justine_more_than_bailey : J - B = 10 := by
  -- Proof will be done here
  sorry

end NUMINAMATH_GPT_justine_more_than_bailey_l1443_144328


namespace NUMINAMATH_GPT_ratio_chest_of_drawers_to_treadmill_l1443_144367

theorem ratio_chest_of_drawers_to_treadmill :
  ∀ (C T TV : ℕ),
  T = 100 →
  TV = 3 * 100 →
  100 + C + TV = 600 →
  C / T = 2 :=
by
  intros C T TV ht htv heq
  sorry

end NUMINAMATH_GPT_ratio_chest_of_drawers_to_treadmill_l1443_144367


namespace NUMINAMATH_GPT_edward_made_in_summer_l1443_144333

theorem edward_made_in_summer
  (spring_earnings : ℤ)
  (spent_on_supplies : ℤ)
  (final_amount : ℤ)
  (S : ℤ)
  (h1 : spring_earnings = 2)
  (h2 : spent_on_supplies = 5)
  (h3 : final_amount = 24)
  (h4 : spring_earnings + S - spent_on_supplies = final_amount) :
  S = 27 := 
by
  sorry

end NUMINAMATH_GPT_edward_made_in_summer_l1443_144333


namespace NUMINAMATH_GPT_rabbit_speed_l1443_144350

theorem rabbit_speed (dog_speed : ℝ) (head_start : ℝ) (catch_time_minutes : ℝ) 
  (H1 : dog_speed = 24) (H2 : head_start = 0.6) (H3 : catch_time_minutes = 4) :
  let catch_time_hours := catch_time_minutes / 60
  let distance_dog_runs := dog_speed * catch_time_hours
  let distance_rabbit_runs := distance_dog_runs - head_start
  let rabbit_speed := distance_rabbit_runs / catch_time_hours
  rabbit_speed = 15 :=
  sorry

end NUMINAMATH_GPT_rabbit_speed_l1443_144350


namespace NUMINAMATH_GPT_reasoning_classification_correct_l1443_144388

def analogical_reasoning := "reasoning from specific to specific"
def inductive_reasoning := "reasoning from part to whole and from individual to general"
def deductive_reasoning := "reasoning from general to specific"

theorem reasoning_classification_correct : 
  (analogical_reasoning, inductive_reasoning, deductive_reasoning) =
  ("reasoning from specific to specific", "reasoning from part to whole and from individual to general", "reasoning from general to specific") := 
by 
  sorry

end NUMINAMATH_GPT_reasoning_classification_correct_l1443_144388


namespace NUMINAMATH_GPT_tangent_line_to_parabola_l1443_144354

theorem tangent_line_to_parabola : ∃ k : ℝ, (∀ x y : ℝ, 4 * x + 6 * y + k = 0) ∧ (∀ y : ℝ, ∃ x : ℝ, y^2 = 32 * x) ∧ (48^2 - 4 * (1 : ℝ) * 8 * k = 0) := by
  use 72
  sorry

end NUMINAMATH_GPT_tangent_line_to_parabola_l1443_144354


namespace NUMINAMATH_GPT_zoe_spent_amount_l1443_144303

def flower_price : ℕ := 3
def roses_bought : ℕ := 8
def daisies_bought : ℕ := 2

theorem zoe_spent_amount :
  roses_bought + daisies_bought = 10 ∧
  flower_price = 3 →
  (roses_bought + daisies_bought) * flower_price = 30 :=
by
  sorry

end NUMINAMATH_GPT_zoe_spent_amount_l1443_144303


namespace NUMINAMATH_GPT_no_positive_integer_solutions_l1443_144371

theorem no_positive_integer_solutions (x y z : ℕ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) : x^2 + y^2 ≠ 7 * z^2 := by
  sorry

end NUMINAMATH_GPT_no_positive_integer_solutions_l1443_144371


namespace NUMINAMATH_GPT_mean_value_of_quadrilateral_angles_l1443_144397

theorem mean_value_of_quadrilateral_angles : 
  (∀ (A B C D : ℕ), A + B + C + D = 360 → (A + B + C + D) / 4 = 90) :=
by {
  sorry
}

end NUMINAMATH_GPT_mean_value_of_quadrilateral_angles_l1443_144397


namespace NUMINAMATH_GPT_infinitely_many_good_primes_infinitely_many_non_good_primes_l1443_144394

def is_good_prime (p : ℕ) : Prop :=
∀ a b : ℕ, a ≡ b [ZMOD p] ↔ a^3 ≡ b^3 [ZMOD p]

theorem infinitely_many_good_primes :
  ∃ᶠ p in at_top, is_good_prime p := sorry

theorem infinitely_many_non_good_primes :
  ∃ᶠ p in at_top, ¬ is_good_prime p := sorry

end NUMINAMATH_GPT_infinitely_many_good_primes_infinitely_many_non_good_primes_l1443_144394


namespace NUMINAMATH_GPT_gcd_b_squared_plus_11b_plus_28_and_b_plus_6_l1443_144300

theorem gcd_b_squared_plus_11b_plus_28_and_b_plus_6 (b : ℤ) (h : ∃ k : ℤ, b = 1573 * k) : 
  Int.gcd (b^2 + 11 * b + 28) (b + 6) = 2 := 
sorry

end NUMINAMATH_GPT_gcd_b_squared_plus_11b_plus_28_and_b_plus_6_l1443_144300


namespace NUMINAMATH_GPT_cost_per_semester_correct_l1443_144326

variable (cost_per_semester total_cost : ℕ)
variable (years semesters_per_year : ℕ)

theorem cost_per_semester_correct :
    years = 13 →
    semesters_per_year = 2 →
    total_cost = 520000 →
    cost_per_semester = total_cost / (years * semesters_per_year) →
    cost_per_semester = 20000 := by
  sorry

end NUMINAMATH_GPT_cost_per_semester_correct_l1443_144326


namespace NUMINAMATH_GPT_least_number_subtracted_378461_l1443_144359

def least_number_subtracted (n : ℕ) : ℕ :=
  n % 13

theorem least_number_subtracted_378461 : least_number_subtracted 378461 = 5 :=
by
  -- actual proof would go here
  sorry

end NUMINAMATH_GPT_least_number_subtracted_378461_l1443_144359


namespace NUMINAMATH_GPT_ratio_surface_area_l1443_144304

open Real

theorem ratio_surface_area (R a : ℝ) 
  (h1 : 4 * R^2 = 6 * a^2) 
  (H : R = (sqrt 6 / 2) * a) : 
  3 * π * R^2 / (6 * a^2) = 3 * π / 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_ratio_surface_area_l1443_144304


namespace NUMINAMATH_GPT_find_missing_number_l1443_144309

theorem find_missing_number
  (mean : ℝ)
  (n : ℕ)
  (nums : List ℝ)
  (total_sum : ℝ)
  (sum_known_numbers : ℝ)
  (missing_number : ℝ) :
  mean = 20 → 
  n = 8 →
  nums = [1, 22, 23, 24, 25, missing_number, 27, 2] →
  total_sum = mean * n →
  sum_known_numbers = 1 + 22 + 23 + 24 + 25 + 27 + 2 →
  missing_number = total_sum - sum_known_numbers :=
by
  intros
  sorry

end NUMINAMATH_GPT_find_missing_number_l1443_144309


namespace NUMINAMATH_GPT_andrea_avg_km_per_day_l1443_144305

theorem andrea_avg_km_per_day
  (total_distance : ℕ := 168)
  (total_days : ℕ := 6)
  (completed_fraction : ℚ := 3/7)
  (completed_days : ℕ := 3) :
  (total_distance * (1 - completed_fraction)) / (total_days - completed_days) = 32 := 
sorry

end NUMINAMATH_GPT_andrea_avg_km_per_day_l1443_144305


namespace NUMINAMATH_GPT_ellipse_equation_no_match_l1443_144310

-- Definitions based on conditions in a)
def a : ℝ := 6
def c : ℝ := 1

-- Calculation for b² based on solution steps
def b_squared := a^2 - c^2

-- Standard forms of ellipse equations
def standard_ellipse_eq1 (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b_squared) = 1
def standard_ellipse_eq2 (x y : ℝ) : Prop := (y^2 / a^2) + (x^2 / b_squared) = 1

-- The proof problem statement
theorem ellipse_equation_no_match : 
  ∀ (x y : ℝ), ¬(standard_ellipse_eq1 x y) ∧ ¬(standard_ellipse_eq2 x y) := 
sorry

end NUMINAMATH_GPT_ellipse_equation_no_match_l1443_144310


namespace NUMINAMATH_GPT_initial_cost_of_milk_l1443_144302

theorem initial_cost_of_milk (total_money : ℝ) (bread_cost : ℝ) (detergent_cost : ℝ) (banana_cost_per_pound : ℝ) (banana_pounds : ℝ) (detergent_coupon : ℝ) (milk_discount_rate : ℝ) (money_left : ℝ)
  (h_total_money : total_money = 20) (h_bread_cost : bread_cost = 3.50) (h_detergent_cost : detergent_cost = 10.25) (h_banana_cost_per_pound : banana_cost_per_pound = 0.75) (h_banana_pounds : banana_pounds = 2)
  (h_detergent_coupon : detergent_coupon = 1.25) (h_milk_discount_rate : milk_discount_rate = 0.5) (h_money_left : money_left = 4) : 
  ∃ (initial_milk_cost : ℝ), initial_milk_cost = 4 := 
sorry

end NUMINAMATH_GPT_initial_cost_of_milk_l1443_144302


namespace NUMINAMATH_GPT_BANANA_distinct_arrangements_l1443_144380

theorem BANANA_distinct_arrangements : 
  (Nat.factorial 6) / ((Nat.factorial 1) * (Nat.factorial 3) * (Nat.factorial 2)) = 60 := 
by
  sorry

end NUMINAMATH_GPT_BANANA_distinct_arrangements_l1443_144380


namespace NUMINAMATH_GPT_paul_digs_the_well_l1443_144325

theorem paul_digs_the_well (P : ℝ) (h1 : 1 / 16 + 1 / P + 1 / 48 = 1 / 8) : P = 24 :=
sorry

end NUMINAMATH_GPT_paul_digs_the_well_l1443_144325


namespace NUMINAMATH_GPT_vector_addition_l1443_144327

variable (a : ℝ × ℝ)
variable (b : ℝ × ℝ)

theorem vector_addition (h1 : a = (-1, 2)) (h2 : b = (1, 0)) :
  3 • a + b = (-2, 6) :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_vector_addition_l1443_144327


namespace NUMINAMATH_GPT_compute_105_times_95_l1443_144332

theorem compute_105_times_95 : (105 * 95 = 9975) :=
by
  sorry

end NUMINAMATH_GPT_compute_105_times_95_l1443_144332


namespace NUMINAMATH_GPT_jihye_marbles_l1443_144339

theorem jihye_marbles (Y : ℕ) (h1 : Y + (Y + 11) = 85) : Y + 11 = 48 := by
  sorry

end NUMINAMATH_GPT_jihye_marbles_l1443_144339


namespace NUMINAMATH_GPT_fraction_eq_zero_iff_l1443_144342

theorem fraction_eq_zero_iff (x : ℝ) : (3 * x - 1) / (x ^ 2 + 1) = 0 ↔ x = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_fraction_eq_zero_iff_l1443_144342


namespace NUMINAMATH_GPT_range_of_m_value_of_m_l1443_144315

variable (α β m : ℝ)

open Real

-- Conditions: α and β are positive roots.
def quadratic_roots (α β m : ℝ) : Prop :=
  (α > 0) ∧ (β > 0) ∧ (α + β = 1 - 2*m) ∧ (α * β = m^2)

-- Part 1: Range of values for m.
theorem range_of_m (h : quadratic_roots α β m) : m ≤ 1/4 ∧ m ≠ 0 :=
sorry

-- Part 2: Given α^2 + β^2 = 49, find the value of m.
theorem value_of_m (h : quadratic_roots α β m) (h' : α^2 + β^2 = 49) : m = -4 :=
sorry

end NUMINAMATH_GPT_range_of_m_value_of_m_l1443_144315


namespace NUMINAMATH_GPT_least_positive_integer_l1443_144364

theorem least_positive_integer (n : ℕ) (h₁ : n % 3 = 0) (h₂ : n % 4 = 1) (h₃ : n % 5 = 2) : n = 57 :=
by
  -- sorry to skip the proof
  sorry

end NUMINAMATH_GPT_least_positive_integer_l1443_144364


namespace NUMINAMATH_GPT_gcd_79_pow7_plus_1_and_79_pow7_plus_79_pow2_plus_1_l1443_144348

theorem gcd_79_pow7_plus_1_and_79_pow7_plus_79_pow2_plus_1 (h_prime : Nat.Prime 79) : 
  Nat.gcd (79^7 + 1) (79^7 + 79^2 + 1) = 1 := 
by
  sorry

end NUMINAMATH_GPT_gcd_79_pow7_plus_1_and_79_pow7_plus_79_pow2_plus_1_l1443_144348


namespace NUMINAMATH_GPT_count_positive_integers_l1443_144312

theorem count_positive_integers (n : ℕ) : ∃ k : ℕ, k = 9 ∧  ∀ n, 1 ≤ n → n < 10 → 3 * n + 20 < 50 :=
by
  sorry

end NUMINAMATH_GPT_count_positive_integers_l1443_144312


namespace NUMINAMATH_GPT_evaluate_expression_l1443_144334

theorem evaluate_expression : 2 + 3 / (4 + 5 / 6) = 76 / 29 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1443_144334


namespace NUMINAMATH_GPT_speed_of_stream_l1443_144376

-- Define the speed of the boat in still water
def speed_of_boat_in_still_water : ℝ := 39

-- Define the effective speed upstream and downstream
def effective_speed_upstream (v : ℝ) : ℝ := speed_of_boat_in_still_water - v
def effective_speed_downstream (v : ℝ) : ℝ := speed_of_boat_in_still_water + v

-- Define the condition that time upstream is twice the time downstream
def time_condition (D v : ℝ) : Prop := 
  (D / effective_speed_upstream v = 2 * (D / effective_speed_downstream v))

-- The main theorem stating the speed of the stream
theorem speed_of_stream (D : ℝ) (h : D > 0) : (v : ℝ) → time_condition D v → v = 13 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_stream_l1443_144376


namespace NUMINAMATH_GPT_four_pow_sub_divisible_iff_l1443_144320

open Nat

theorem four_pow_sub_divisible_iff (m n k : ℕ) (h₁ : m > n) : 
  (3^(k + 1)) ∣ (4^m - 4^n) ↔ (3^k) ∣ (m - n) := 
by sorry

end NUMINAMATH_GPT_four_pow_sub_divisible_iff_l1443_144320


namespace NUMINAMATH_GPT_f_zero_eq_zero_f_one_eq_one_f_n_is_n_l1443_144360

variable (f : ℤ → ℤ)

axiom functional_eq : ∀ m n : ℤ, f (m^2 + f n) = f (f m) + n

theorem f_zero_eq_zero : f 0 = 0 :=
sorry

theorem f_one_eq_one : f 1 = 1 :=
sorry

theorem f_n_is_n : ∀ n : ℤ, f n = n :=
sorry

end NUMINAMATH_GPT_f_zero_eq_zero_f_one_eq_one_f_n_is_n_l1443_144360


namespace NUMINAMATH_GPT_mary_change_received_l1443_144391

def cost_of_adult_ticket : ℝ := 2
def cost_of_child_ticket : ℝ := 1
def discount_first_child : ℝ := 0.5
def discount_second_child : ℝ := 0.75
def discount_third_child : ℝ := 1
def sales_tax_rate : ℝ := 0.08
def amount_paid : ℝ := 20

def total_ticket_cost_before_tax : ℝ :=
  cost_of_adult_ticket + (cost_of_child_ticket * discount_first_child) + 
  (cost_of_child_ticket * discount_second_child) + (cost_of_child_ticket * discount_third_child)

def sales_tax : ℝ :=
  total_ticket_cost_before_tax * sales_tax_rate

def total_ticket_cost_with_tax : ℝ :=
  total_ticket_cost_before_tax + sales_tax

def change_received : ℝ :=
  amount_paid - total_ticket_cost_with_tax

theorem mary_change_received :
  change_received = 15.41 :=
by
  sorry

end NUMINAMATH_GPT_mary_change_received_l1443_144391


namespace NUMINAMATH_GPT_eval_expression_l1443_144317

theorem eval_expression : 6 + 15 / 3 - 4^2 + 1 = -4 := by
  sorry

end NUMINAMATH_GPT_eval_expression_l1443_144317


namespace NUMINAMATH_GPT_correctCountForDivisibilityBy15_l1443_144347

namespace Divisibility

noncomputable def countWaysToMakeDivisibleBy15 : Nat := 
  let digits := [0, 2, 4, 5, 7, 9]
  let baseSum := 2 + 0 + 1 + 6 + 0 + 2
  let validLastDigit := [0, 5]
  let totalCombinations := 6^4
  let ways := 2 * totalCombinations
  let adjustment := (validLastDigit.length * digits.length * digits.length * digits.length * validLastDigit.length) / 4 -- Correcting multiplier as per reference
  adjustment

theorem correctCountForDivisibilityBy15 : countWaysToMakeDivisibleBy15 = 864 := 
  by
    sorry

end Divisibility

end NUMINAMATH_GPT_correctCountForDivisibilityBy15_l1443_144347


namespace NUMINAMATH_GPT_A_and_C_amount_l1443_144321

variables (A B C : ℝ)

def amounts_satisfy_conditions : Prop :=
  (A + B + C = 500) ∧ (B + C = 320) ∧ (C = 20)

theorem A_and_C_amount (h : amounts_satisfy_conditions A B C) : A + C = 200 :=
by {
  sorry
}

end NUMINAMATH_GPT_A_and_C_amount_l1443_144321


namespace NUMINAMATH_GPT_smallest_number_l1443_144365

theorem smallest_number (x : ℕ) : (∃ y : ℕ, y = x - 16 ∧ (y % 4 = 0) ∧ (y % 6 = 0) ∧ (y % 8 = 0) ∧ (y % 10 = 0)) → x = 136 := by
  sorry

end NUMINAMATH_GPT_smallest_number_l1443_144365


namespace NUMINAMATH_GPT_math_proof_problem_l1443_144330

variable {a_n : ℕ → ℝ} -- sequence a_n
variable {b_n : ℕ → ℝ} -- sequence b_n

-- Given that a_n is an arithmetic sequence with common difference d
def isArithmeticSequence (a_n : ℕ → ℝ) (d : ℝ) : Prop := ∀ n, a_n (n + 1) = a_n n + d

-- Given condition for sequence b_n
def b_n_def (a_n b_n : ℕ → ℝ) : Prop := ∀ n, b_n n = a_n (n + 1) * a_n (n + 2) - a_n n ^ 2

-- Both sequences have common difference d ≠ 0
def common_difference_ne_zero (a_n b_n : ℕ → ℝ) (d : ℝ) : Prop :=
  isArithmeticSequence a_n d ∧ isArithmeticSequence b_n d ∧ d ≠ 0

-- Condition involving positive integers s and t
def integer_condition (a_n b_n : ℕ → ℝ) (s t : ℕ) : Prop :=
  1 ≤ s ∧ 1 ≤ t ∧ ∃ (x : ℤ), a_n s + b_n t = x

-- Theorem to prove that the sequence {b_n} is arithmetic and find minimum value of |a_1|
theorem math_proof_problem
  (a_n b_n : ℕ → ℝ) (d : ℝ) (s t : ℕ)
  (arithmetic_a : isArithmeticSequence a_n d)
  (defined_b : b_n_def a_n b_n)
  (common_diff : common_difference_ne_zero a_n b_n d)
  (int_condition : integer_condition a_n b_n s t) :
  (isArithmeticSequence b_n (3 * d ^ 2)) ∧ (∃ m : ℝ, m = |a_n 1| ∧ m = 1 / 36) :=
  by sorry

end NUMINAMATH_GPT_math_proof_problem_l1443_144330


namespace NUMINAMATH_GPT_problem_proof_l1443_144375

noncomputable def f (x : ℝ) : ℝ := Real.sin x
noncomputable def h (x : ℝ) : ℝ := Real.sin (x + Real.pi / 3)
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem problem_proof :
  (∀ x, g (x + Real.pi) = g x) ∧ (∀ y, g (2 * (Real.pi / 12) - y) = g y) :=
by
  sorry

end NUMINAMATH_GPT_problem_proof_l1443_144375


namespace NUMINAMATH_GPT_max_value_of_quadratic_function_l1443_144389

def quadratic_function (x : ℝ) : ℝ := -x^2 + 2*x + 4

theorem max_value_of_quadratic_function : ∃ x : ℝ, quadratic_function x = 5 ∧ ∀ y : ℝ, quadratic_function y ≤ 5 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_quadratic_function_l1443_144389


namespace NUMINAMATH_GPT_altitudes_sum_of_triangle_formed_by_line_and_axes_l1443_144316

noncomputable def sum_of_altitudes (x y : ℝ) : ℝ :=
  let intercept_x := 6
  let intercept_y := 16
  let altitude_3 := 48 / Real.sqrt (8^2 + 3^2)
  intercept_x + intercept_y + altitude_3

theorem altitudes_sum_of_triangle_formed_by_line_and_axes :
  ∀ (x y : ℝ), (8 * x + 3 * y = 48) →
  sum_of_altitudes x y = 22 + 48 / Real.sqrt 73 :=
by
  sorry

end NUMINAMATH_GPT_altitudes_sum_of_triangle_formed_by_line_and_axes_l1443_144316


namespace NUMINAMATH_GPT_ant_positions_l1443_144366

noncomputable def side_lengths (a b c : ℝ) : Prop :=
  (a + 2 = b) ∧ (b + 2 = c) ∧ (4 * c / c - 2 + 1) = 3 ∧ (4 * c / (c - 4) - 1) = 3

theorem ant_positions (a b c : ℝ) (v : ℝ) (ha : side_lengths a b c) : 
  a = 4 ∧ b = 6 ∧ c = 8 :=
by
  sorry

end NUMINAMATH_GPT_ant_positions_l1443_144366


namespace NUMINAMATH_GPT_part1_part2_l1443_144331

noncomputable def f (a x : ℝ) : ℝ := a * x^2 - (a + 2) * x + Real.log x

theorem part1 (a : ℝ) (h : (2 * a - (a + 2) + 1) = 0) : a = 1 :=
by
  sorry

theorem part2 (a x : ℝ) (ha : a ≥ 1) (hx : 1 ≤ x ∧ x ≤ Real.exp 1) : (2 * a * x - (a + 2) + 1 / x) ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l1443_144331


namespace NUMINAMATH_GPT_least_three_digit_with_factors_correct_l1443_144324

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000
def has_factors (n : ℕ) : Prop := n % 3 = 0 ∧ n % 4 = 0 ∧ n % 9 = 0
def least_three_digit_with_factors : ℕ := 108

theorem least_three_digit_with_factors_correct : 
  is_three_digit least_three_digit_with_factors ∧ has_factors least_three_digit_with_factors ∧
  ∀ m : ℕ, is_three_digit m → has_factors m → least_three_digit_with_factors ≤ m := 
by 
  sorry

end NUMINAMATH_GPT_least_three_digit_with_factors_correct_l1443_144324


namespace NUMINAMATH_GPT_container_volume_ratio_l1443_144338

theorem container_volume_ratio (A B : ℚ) (h : (2 / 3 : ℚ) * A = (1 / 2 : ℚ) * B) : A / B = 3 / 4 :=
by sorry

end NUMINAMATH_GPT_container_volume_ratio_l1443_144338


namespace NUMINAMATH_GPT_balance_difference_is_7292_83_l1443_144341

noncomputable def angela_balance : ℝ := 7000 * (1 + 0.05)^15
noncomputable def bob_balance : ℝ := 9000 * (1 + 0.03)^30
noncomputable def balance_difference : ℝ := bob_balance - angela_balance

theorem balance_difference_is_7292_83 : balance_difference = 7292.83 := by
  sorry

end NUMINAMATH_GPT_balance_difference_is_7292_83_l1443_144341


namespace NUMINAMATH_GPT_car_stops_at_three_seconds_l1443_144301

theorem car_stops_at_three_seconds (t : ℝ) (h : -3 * t^2 + 18 * t = 0) : t = 3 := 
sorry

end NUMINAMATH_GPT_car_stops_at_three_seconds_l1443_144301


namespace NUMINAMATH_GPT_correct_operation_l1443_144399

variable {a b : ℝ}

theorem correct_operation : (3 * a^2 * b - 3 * b * a^2 = 0) :=
by sorry

end NUMINAMATH_GPT_correct_operation_l1443_144399


namespace NUMINAMATH_GPT_lcm_1_to_5_l1443_144335

theorem lcm_1_to_5 : Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5 = 60 := by
  sorry

end NUMINAMATH_GPT_lcm_1_to_5_l1443_144335


namespace NUMINAMATH_GPT_math_problem_l1443_144340

theorem math_problem (a b c : ℝ) (h₁ : a = 85) (h₂ : b = 32) (h₃ : c = 113) :
  (a + b / c) * c = 9637 :=
by
  rw [h₁, h₂, h₃]
  sorry

end NUMINAMATH_GPT_math_problem_l1443_144340


namespace NUMINAMATH_GPT_kathleen_savings_in_july_l1443_144351

theorem kathleen_savings_in_july (savings_june savings_august spending_school spending_clothes money_left savings_target add_from_aunt : ℕ) 
  (h_june : savings_june = 21)
  (h_august : savings_august = 45)
  (h_school : spending_school = 12)
  (h_clothes : spending_clothes = 54)
  (h_left : money_left = 46)
  (h_target : savings_target = 125)
  (h_aunt : add_from_aunt = 25)
  (not_received_from_aunt : (savings_june + savings_august + money_left + add_from_aunt) ≤ savings_target)
  : (savings_june + savings_august + money_left + spending_school + spending_clothes - (savings_june + savings_august + spending_school + spending_clothes)) = 46 := 
by 
  -- These conditions narrate the problem setup
  -- We can proceed to show the proof here
  sorry 

end NUMINAMATH_GPT_kathleen_savings_in_july_l1443_144351


namespace NUMINAMATH_GPT_probability_x_gt_2y_is_1_over_3_l1443_144358

noncomputable def probability_x_gt_2y_in_rectangle : ℝ :=
  let A_rect := 6 * 1
  let A_triangle := (1/2) * 4 * 1
  A_triangle / A_rect

theorem probability_x_gt_2y_is_1_over_3 :
  probability_x_gt_2y_in_rectangle = 1 / 3 :=
sorry

end NUMINAMATH_GPT_probability_x_gt_2y_is_1_over_3_l1443_144358


namespace NUMINAMATH_GPT_total_pieces_correct_l1443_144378

theorem total_pieces_correct :
  let bell_peppers := 10
  let onions := 7
  let zucchinis := 15
  let bell_peppers_slices := (2 * 20)  -- 25% of 10 bell peppers sliced into 20 slices each
  let bell_peppers_large_pieces := (7 * 10)  -- Remaining 75% cut into 10 pieces each
  let bell_peppers_smaller_pieces := (35 * 3)  -- Half of large pieces cut into 3 pieces each
  let onions_slices := (3 * 18)  -- 50% of onions sliced into 18 slices each
  let onions_pieces := (4 * 8)  -- Remaining 50% cut into 8 pieces each
  let zucchinis_slices := (4 * 15)  -- 30% of zucchinis sliced into 15 pieces each
  let zucchinis_pieces := (10 * 8)  -- Remaining 70% cut into 8 pieces each
  let total_slices := bell_peppers_slices + onions_slices + zucchinis_slices
  let total_pieces := bell_peppers_large_pieces + bell_peppers_smaller_pieces + onions_pieces + zucchinis_pieces
  total_slices + total_pieces = 441 :=
by
  sorry

end NUMINAMATH_GPT_total_pieces_correct_l1443_144378


namespace NUMINAMATH_GPT_constant_max_value_l1443_144345

theorem constant_max_value (n : ℤ) (c : ℝ) (h1 : c * (n^2) ≤ 8100) (h2 : n = 8) :
  c ≤ 126.5625 :=
sorry

end NUMINAMATH_GPT_constant_max_value_l1443_144345


namespace NUMINAMATH_GPT_subtraction_result_l1443_144344

open Matrix

namespace Vector

def a : (Fin 3 → ℝ) :=
  ![5, -3, 2]

def b : (Fin 3 → ℝ) :=
  ![-2, 4, 1]

theorem subtraction_result : a - (2 • b) = ![9, -11, 0] :=
by
  -- Skipping the proof
  sorry

end Vector

end NUMINAMATH_GPT_subtraction_result_l1443_144344


namespace NUMINAMATH_GPT_evaluate_expression_l1443_144349

variable (x y z : ℚ) -- assuming x, y, z are rational numbers

theorem evaluate_expression (h1 : x = 1 / 4) (h2 : y = 3 / 4) (h3 : z = -8) :
  x^2 * y^3 * z^2 = 108 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1443_144349


namespace NUMINAMATH_GPT_diana_hits_seven_l1443_144368

-- Define the participants
inductive Player 
| Alex 
| Brooke 
| Carlos 
| Diana 
| Emily 
| Fiona

open Player

-- Define a function to get the total score of a participant
def total_score (p : Player) : ℕ :=
match p with
| Alex => 20
| Brooke => 23
| Carlos => 28
| Diana => 18
| Emily => 26
| Fiona => 30

-- Function to check if a dart target is hit within the range and unique
def is_valid_target (x y z : ℕ) :=
x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ 1 ≤ x ∧ x ≤ 12 ∧ 1 ≤ y ∧ y ≤ 12 ∧ 1 ≤ z ∧ z ≤ 12

-- Check if the sum equals the score of the player
def valid_score (p : Player) (x y z : ℕ) :=
is_valid_target x y z ∧ x + y + z = total_score p

-- Lean 4 theorem statement, asking if Diana hits the region 7
theorem diana_hits_seven : ∃ x y z, valid_score Diana x y z ∧ (x = 7 ∨ y = 7 ∨ z = 7) :=
sorry

end NUMINAMATH_GPT_diana_hits_seven_l1443_144368


namespace NUMINAMATH_GPT_find_second_expression_l1443_144393

theorem find_second_expression (a x : ℕ) (h₁ : (2 * a + 16 + x) / 2 = 84) (h₂ : a = 32) : x = 88 :=
  sorry

end NUMINAMATH_GPT_find_second_expression_l1443_144393


namespace NUMINAMATH_GPT_three_person_subcommittees_from_seven_l1443_144387

-- Definition of the combinations formula (binomial coefficient)
def choose : ℕ → ℕ → ℕ
| n, k => if k = 0 then 1 else (n * choose (n - 1) (k - 1)) / k 

-- Problem statement in Lean 4
theorem three_person_subcommittees_from_seven : choose 7 3 = 35 :=
by
  -- We would fill in the steps here or use a sorry to skip the proof
  sorry

end NUMINAMATH_GPT_three_person_subcommittees_from_seven_l1443_144387


namespace NUMINAMATH_GPT_minimal_positive_sum_circle_integers_l1443_144353

-- Definitions based on the conditions in the problem statement
def cyclic_neighbors (l : List Int) (i : ℕ) : Int :=
  l.getD (Nat.mod (i - 1) l.length) 0 + l.getD (Nat.mod (i + 1) l.length) 0

-- Problem statement in Lean: 
theorem minimal_positive_sum_circle_integers :
  ∃ (l : List Int), l.length ≥ 5 ∧ (∀ (i : ℕ), i < l.length → l.getD i 0 ∣ cyclic_neighbors l i) ∧ (0 < l.sum) ∧ l.sum = 2 :=
sorry

end NUMINAMATH_GPT_minimal_positive_sum_circle_integers_l1443_144353


namespace NUMINAMATH_GPT_train_cross_time_approx_l1443_144369

noncomputable def time_to_cross_pole (length : ℝ) (speed_kmh : ℝ) : ℝ :=
  let speed_ms := speed_kmh * (5 / 18)
  length / speed_ms

theorem train_cross_time_approx
  (d : ℝ) (v_kmh : ℝ)
  (h_d : d = 120)
  (h_v : v_kmh = 121) :
  abs (time_to_cross_pole d v_kmh - 3.57) < 0.01 :=
by {
  sorry
}

end NUMINAMATH_GPT_train_cross_time_approx_l1443_144369


namespace NUMINAMATH_GPT_correct_factorization_l1443_144370

-- Define the conditions from the problem
def conditionA (a b : ℝ) : Prop := a * (a - b) - b * (b - a) = (a - b) * (a + b)
def conditionB (a b : ℝ) : Prop := a^2 - 4 * b^2 = (a + 4 * b) * (a - 4 * b)
def conditionC (a b : ℝ) : Prop := a^2 + 2 * a * b - b^2 = (a + b)^2
def conditionD (a : ℝ) : Prop := a^2 - a - 2 = a * (a - 1) - 2

-- Main theorem statement verifying that only conditionA holds
theorem correct_factorization (a b : ℝ) : 
  conditionA a b ∧ ¬ conditionB a b ∧ ¬ conditionC a b ∧ ¬ conditionD a :=
by 
  sorry

end NUMINAMATH_GPT_correct_factorization_l1443_144370


namespace NUMINAMATH_GPT_skateboarded_one_way_distance_l1443_144308

-- Define the total skateboarded distance and the walked distance.
def total_skateboarded : ℕ := 24
def walked_distance : ℕ := 4

-- Define the proof theorem.
theorem skateboarded_one_way_distance : 
    (total_skateboarded - walked_distance) / 2 = 10 := 
by sorry

end NUMINAMATH_GPT_skateboarded_one_way_distance_l1443_144308


namespace NUMINAMATH_GPT_second_part_of_ratio_l1443_144379

theorem second_part_of_ratio (h_ratio : ∀ (x : ℝ), 25 = 0.5 * (25 + x)) : ∃ x : ℝ, x = 25 :=
by
  sorry

end NUMINAMATH_GPT_second_part_of_ratio_l1443_144379


namespace NUMINAMATH_GPT_smallest_a_plus_b_l1443_144395

theorem smallest_a_plus_b (a b : ℕ) (h1: 0 < a) (h2: 0 < b) (h3 : 2^10 * 7^3 = a^b) : a + b = 31 :=
sorry

end NUMINAMATH_GPT_smallest_a_plus_b_l1443_144395


namespace NUMINAMATH_GPT_find_A_l1443_144390

variable (p q r s A : ℝ)

theorem find_A (H1 : (p + q + r + s) / 4 = 5) (H2 : (p + q + r + s + A) / 5 = 8) : A = 20 := 
by
  sorry

end NUMINAMATH_GPT_find_A_l1443_144390


namespace NUMINAMATH_GPT_all_a_n_are_perfect_squares_l1443_144346

noncomputable def c : ℕ → ℤ 
| 0 => 1
| 1 => 0
| 2 => 2005
| n+2 => -3 * c n - 4 * c (n-1) + 2008

noncomputable def a (n : ℕ) : ℤ :=
  5 * (c (n + 2) - c n) * (502 - c (n - 1) - c (n - 2)) + 4 ^ n * 2004 * 501

theorem all_a_n_are_perfect_squares (n : ℕ) (h : n > 2) : ∃ k : ℤ, a n = k^2 :=
by
  sorry

end NUMINAMATH_GPT_all_a_n_are_perfect_squares_l1443_144346


namespace NUMINAMATH_GPT_second_number_is_90_l1443_144355

theorem second_number_is_90 (x y z : ℕ) 
  (h1 : z = 4 * y) 
  (h2 : y = 2 * x) 
  (h3 : (x + y + z) / 3 = 165) : y = 90 := 
by
  sorry

end NUMINAMATH_GPT_second_number_is_90_l1443_144355


namespace NUMINAMATH_GPT_total_students_l1443_144362

variables (F G B N : ℕ)
variables (hF : F = 41) (hG : G = 22) (hB : B = 9) (hN : N = 6)

theorem total_students (F G B N : ℕ) (hF : F = 41) (hG : G = 22) (hB : B = 9) (hN : N = 6) : 
  F + G - B + N = 60 := by
sorry

end NUMINAMATH_GPT_total_students_l1443_144362


namespace NUMINAMATH_GPT_find_vector_b_coordinates_l1443_144374

theorem find_vector_b_coordinates 
  (a b : ℝ × ℝ) 
  (h₁ : a = (-3, 4)) 
  (h₂ : ∃ m : ℝ, m < 0 ∧ b = (-3 * m, 4 * m)) 
  (h₃ : ‖b‖ = 10) : 
  b = (6, -8) := 
by
  sorry

end NUMINAMATH_GPT_find_vector_b_coordinates_l1443_144374


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1443_144396

theorem solution_set_of_inequality (a : ℝ) :
  (a > 1 → {x : ℝ | ax + 1 < a^2 + x} = {x : ℝ | x < a + 1}) ∧
  (a < 1 → {x : ℝ | ax + 1 < a^2 + x} = {x : ℝ | x > a + 1}) ∧
  (a = 1 → {x : ℝ | ax + 1 < a^2 + x} = ∅) := 
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1443_144396
