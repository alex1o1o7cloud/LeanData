import Mathlib

namespace NUMINAMATH_GPT_cos_240_eq_neg_half_l2318_231854

/-- Prove that cos 240 degrees equals -1/2 --/
theorem cos_240_eq_neg_half : Real.cos (240 * Real.pi / 180) = -1/2 :=
  sorry

end NUMINAMATH_GPT_cos_240_eq_neg_half_l2318_231854


namespace NUMINAMATH_GPT_g_at_5_eq_9_l2318_231881

-- Define the polynomial function g as given in the conditions
def g (a b c x : ℝ) : ℝ := a * x^5 + b * x^3 + c * x + 3

-- Define the hypothesis that g(-5) = -3
axiom g_neg5 (a b c : ℝ) : g a b c (-5) = -3

-- State the theorem to prove that g(5) = 9 given the conditions
theorem g_at_5_eq_9 (a b c : ℝ) : g a b c 5 = 9 := 
by sorry

end NUMINAMATH_GPT_g_at_5_eq_9_l2318_231881


namespace NUMINAMATH_GPT_rectangle_area_12_l2318_231810

theorem rectangle_area_12
  (L W : ℝ)
  (h1 : L + W = 7)
  (h2 : L^2 + W^2 = 25) :
  L * W = 12 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_12_l2318_231810


namespace NUMINAMATH_GPT_area_to_be_painted_l2318_231862

def wall_height : ℕ := 8
def wall_length : ℕ := 15
def glass_painting_height : ℕ := 3
def glass_painting_length : ℕ := 5

theorem area_to_be_painted :
  (wall_height * wall_length) - (glass_painting_height * glass_painting_length) = 105 := by
  sorry

end NUMINAMATH_GPT_area_to_be_painted_l2318_231862


namespace NUMINAMATH_GPT_last_three_digits_of_2_pow_9000_l2318_231842

-- The proof statement
theorem last_three_digits_of_2_pow_9000 (h : 2 ^ 300 ≡ 1 [MOD 1000]) : 2 ^ 9000 ≡ 1 [MOD 1000] :=
by
  sorry

end NUMINAMATH_GPT_last_three_digits_of_2_pow_9000_l2318_231842


namespace NUMINAMATH_GPT_solve_system_correct_l2318_231801

noncomputable def solve_system (a b c d e : ℝ) : Prop :=
  3 * a = (b + c + d) ^ 3 ∧ 
  3 * b = (c + d + e) ^ 3 ∧ 
  3 * c = (d + e + a) ^ 3 ∧ 
  3 * d = (e + a + b) ^ 3 ∧ 
  3 * e = (a + b + c) ^ 3

theorem solve_system_correct :
  ∀ (a b c d e : ℝ), solve_system a b c d e → 
    (a = 1/3 ∧ b = 1/3 ∧ c = 1/3 ∧ d = 1/3 ∧ e = 1/3) ∨ 
    (a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 ∧ e = 0) ∨ 
    (a = -1/3 ∧ b = -1/3 ∧ c = -1/3 ∧ d = -1/3 ∧ e = -1/3) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_correct_l2318_231801


namespace NUMINAMATH_GPT_tangency_condition_l2318_231845

def functions_parallel (a b c : ℝ) (f g: ℝ → ℝ)
       (parallel: ∀ x, f x = a * x + b ∧ g x = a * x + c) := 
  ∀ x, f x = a * x + b ∧ g x = a * x + c

theorem tangency_condition (a b c A : ℝ)
    (h_parallel : a ≠ 0)
    (h_tangency : (∀ x, (a * x + b)^2 = 7 * (a * x + c))) :
  A = 0 ∨ A = -7 :=
sorry

end NUMINAMATH_GPT_tangency_condition_l2318_231845


namespace NUMINAMATH_GPT_relationship_of_points_on_inverse_proportion_l2318_231840

theorem relationship_of_points_on_inverse_proportion :
  let y_1 := - 3 / - 3
  let y_2 := - 3 / - 1
  let y_3 := - 3 / (1 / 3)
  y_3 < y_1 ∧ y_1 < y_2 :=
by
  let y_1 := - 3 / - 3
  let y_2 := - 3 / - 1
  let y_3 := - 3 / (1 / 3)
  sorry

end NUMINAMATH_GPT_relationship_of_points_on_inverse_proportion_l2318_231840


namespace NUMINAMATH_GPT_cube_surface_area_l2318_231806

theorem cube_surface_area (Q : ℝ) (a : ℝ) (H : (3 * a^2 * Real.sqrt 3) / 2 = Q) :
    (6 * (a * Real.sqrt 2) ^ 2) = (8 * Q * Real.sqrt 3) / 3 :=
by
  sorry

end NUMINAMATH_GPT_cube_surface_area_l2318_231806


namespace NUMINAMATH_GPT_students_not_visiting_any_l2318_231883

-- Define the given conditions as Lean definitions
def total_students := 52
def visited_botanical := 12
def visited_animal := 26
def visited_technology := 23
def visited_botanical_animal := 5
def visited_botanical_technology := 2
def visited_animal_technology := 4
def visited_all_three := 1

-- Translate the problem statement and proof goal
theorem students_not_visiting_any :
  total_students - (visited_botanical + visited_animal + visited_technology 
  - visited_botanical_animal - visited_botanical_technology 
  - visited_animal_technology + visited_all_three) = 1 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_students_not_visiting_any_l2318_231883


namespace NUMINAMATH_GPT_present_population_l2318_231871

variable (P : ℝ)
variable (H1 : P * 1.20 = 2400)

theorem present_population (H1 : P * 1.20 = 2400) : P = 2000 :=
by {
  sorry
}

end NUMINAMATH_GPT_present_population_l2318_231871


namespace NUMINAMATH_GPT_algebraic_expression_evaluation_l2318_231804

theorem algebraic_expression_evaluation (x y : ℝ) : 
  3 * (x^2 - 2 * x * y + y^2) - 3 * (x^2 - 2 * x * y + y^2 - 1) = 3 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_evaluation_l2318_231804


namespace NUMINAMATH_GPT_product_of_sum_and_reciprocal_ge_four_l2318_231828

theorem product_of_sum_and_reciprocal_ge_four (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (a + b) * (1 / a + 1 / b) ≥ 4 :=
sorry

end NUMINAMATH_GPT_product_of_sum_and_reciprocal_ge_four_l2318_231828


namespace NUMINAMATH_GPT_sum_of_imaginary_parts_l2318_231820

theorem sum_of_imaginary_parts (x y u v w z : ℝ) (h1 : y = 5) 
  (h2 : w = -x - u) (h3 : (x + y * I) + (u + v * I) + (w + z * I) = 4 * I) :
  v + z = -1 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_imaginary_parts_l2318_231820


namespace NUMINAMATH_GPT_greatest_mondays_in_45_days_l2318_231818

-- Define the days in a week
def days_in_week : ℕ := 7

-- Define the total days being considered
def total_days : ℕ := 45

-- Calculate the complete weeks in the total days
def complete_weeks : ℕ := total_days / days_in_week

-- Calculate the extra days
def extra_days : ℕ := total_days % days_in_week

-- Define that the period starts on Monday (condition)
def starts_on_monday : Bool := true

-- Prove that the greatest number of Mondays in the first 45 days is 7
theorem greatest_mondays_in_45_days (h1 : days_in_week = 7) (h2 : total_days = 45) (h3 : starts_on_monday = true) : 
  (complete_weeks + if starts_on_monday && extra_days >= 1 then 1 else 0) = 7 := 
by
  sorry

end NUMINAMATH_GPT_greatest_mondays_in_45_days_l2318_231818


namespace NUMINAMATH_GPT_find_f_neg_one_l2318_231843

open Real

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * sin x + b * tan x + 3

theorem find_f_neg_one (a b : ℝ) (h : f a b 1 = 1) : f a b (-1) = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_f_neg_one_l2318_231843


namespace NUMINAMATH_GPT_find_ab_l2318_231859

theorem find_ab (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 39) : a * b = 15 :=
by
  sorry

end NUMINAMATH_GPT_find_ab_l2318_231859


namespace NUMINAMATH_GPT_trisha_interest_l2318_231839

noncomputable def total_amount (P : ℝ) (r : ℝ) (D : ℝ) (t : ℕ) : ℝ :=
  let rec compute (n : ℕ) (A : ℝ) :=
    if n = 0 then A
    else let A_next := A * (1 + r) + D
         compute (n - 1) A_next
  compute t P

noncomputable def total_deposits (D : ℝ) (t : ℕ) : ℝ :=
  D * t

noncomputable def total_interest (P : ℝ) (r : ℝ) (D : ℝ) (t : ℕ) : ℝ :=
  total_amount P r D t - P - total_deposits D t

theorem trisha_interest :
  total_interest 2000 0.05 300 5 = 710.25 :=
by
  sorry

end NUMINAMATH_GPT_trisha_interest_l2318_231839


namespace NUMINAMATH_GPT_expression_evaluates_to_2023_l2318_231821

theorem expression_evaluates_to_2023 : 
  (2023^3 - 2 * 2023^2 * 2024 + 3 * 2023 * 2024^2 - 2024^3 + 1) / (2023 * 2024) = 2023 :=
by 
  sorry

end NUMINAMATH_GPT_expression_evaluates_to_2023_l2318_231821


namespace NUMINAMATH_GPT_find_annual_interest_rate_l2318_231875

/-- 
  Given:
  - Principal P = 10000
  - Interest I = 450
  - Time period T = 0.75 years

  Prove that the annual interest rate is 0.08.
-/
theorem find_annual_interest_rate (P I : ℝ) (T : ℝ) (hP : P = 10000) (hI : I = 450) (hT : T = 0.75) : 
  (I / (P * T) / T) = 0.08 :=
by
  sorry

end NUMINAMATH_GPT_find_annual_interest_rate_l2318_231875


namespace NUMINAMATH_GPT_polynomial_sum_squares_l2318_231838

theorem polynomial_sum_squares (a0 a1 a2 a3 a4 a5 a6 a7 : ℤ)
  (h₁ : (1 - 2) ^ 7 = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7)
  (h₂ : (1 + -2) ^ 7 = a0 - a1 + a2 - a3 + a4 - a5 + a6 - a7) :
  (a0 + a2 + a4 + a6) ^ 2 - (a1 + a3 + a5 + a7) ^ 2 = -2187 := 
  sorry

end NUMINAMATH_GPT_polynomial_sum_squares_l2318_231838


namespace NUMINAMATH_GPT_total_interest_proof_l2318_231834

open Real

def initial_investment : ℝ := 10000
def interest_6_months : ℝ := 0.02 * initial_investment
def reinvested_amount_6_months : ℝ := initial_investment + interest_6_months
def interest_10_months : ℝ := 0.03 * reinvested_amount_6_months
def reinvested_amount_10_months : ℝ := reinvested_amount_6_months + interest_10_months
def interest_18_months : ℝ := 0.04 * reinvested_amount_10_months

def total_interest : ℝ := interest_6_months + interest_10_months + interest_18_months

theorem total_interest_proof : total_interest = 926.24 := by
    sorry

end NUMINAMATH_GPT_total_interest_proof_l2318_231834


namespace NUMINAMATH_GPT_total_distance_of_race_is_150_l2318_231819

variable (D : ℝ)

-- Conditions
def A_covers_distance_in_45_seconds (D : ℝ) : Prop := ∃ A_speed, A_speed = D / 45
def B_covers_distance_in_60_seconds (D : ℝ) : Prop := ∃ B_speed, B_speed = D / 60
def A_beats_B_by_50_meters_in_60_seconds (D : ℝ) : Prop := (D / 45) * 60 = D + 50

theorem total_distance_of_race_is_150 :
  A_covers_distance_in_45_seconds D ∧ 
  B_covers_distance_in_60_seconds D ∧ 
  A_beats_B_by_50_meters_in_60_seconds D → 
  D = 150 :=
by
  sorry

end NUMINAMATH_GPT_total_distance_of_race_is_150_l2318_231819


namespace NUMINAMATH_GPT_probability_of_getting_a_prize_l2318_231832

theorem probability_of_getting_a_prize {prizes blanks : ℕ} (h_prizes : prizes = 10) (h_blanks : blanks = 25) :
  (prizes / (prizes + blanks) : ℚ) = 2 / 7 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_getting_a_prize_l2318_231832


namespace NUMINAMATH_GPT_smallest_positive_y_l2318_231874

theorem smallest_positive_y (y : ℕ) (h : 42 * y + 8 ≡ 4 [MOD 24]) : y = 2 :=
sorry

end NUMINAMATH_GPT_smallest_positive_y_l2318_231874


namespace NUMINAMATH_GPT_a5_equals_2_l2318_231812

variable {a : ℕ → ℝ}  -- a_n represents the nth term of the arithmetic sequence

-- Define the arithmetic sequence property
def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n m : ℕ, a (n + 1) = a 1 + n * d 

-- Given condition
axiom arithmetic_condition (h : is_arithmetic_sequence a) : a 1 + a 5 + a 9 = 6

-- The goal is to prove a_5 = 2
theorem a5_equals_2 (h : is_arithmetic_sequence a) (h_cond : a 1 + a 5 + a 9 = 6) : a 5 = 2 := 
by 
  sorry

end NUMINAMATH_GPT_a5_equals_2_l2318_231812


namespace NUMINAMATH_GPT_car_travel_distance_l2318_231814

theorem car_travel_distance (distance : ℝ) 
  (speed1 : ℝ := 80) 
  (speed2 : ℝ := 76.59574468085106) 
  (time_difference : ℝ := 2 / 3600) : 
  (distance / speed2 = distance / speed1 + time_difference) → 
  distance = 0.998177 :=
by
  -- assuming the above equation holds, we need to conclude the distance
  sorry

end NUMINAMATH_GPT_car_travel_distance_l2318_231814


namespace NUMINAMATH_GPT_smallest_n_satisfies_condition_l2318_231848

theorem smallest_n_satisfies_condition : 
  ∃ (n : ℕ), n = 1806 ∧ ∀ (p : ℕ), Nat.Prime p → n % (p - 1) = 0 → n % p = 0 := 
sorry

end NUMINAMATH_GPT_smallest_n_satisfies_condition_l2318_231848


namespace NUMINAMATH_GPT_range_of_a_l2318_231863

noncomputable def func (x a : ℝ) : ℝ := -x^2 - 2 * a * x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → func x a ≤ a^2) →
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ func x a = a^2) →
  -1 ≤ a ∧ a ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2318_231863


namespace NUMINAMATH_GPT_maximum_sum_set_l2318_231802

def no_two_disjoint_subsets_have_equal_sums (S : Finset ℕ) : Prop :=
  ∀ (A B : Finset ℕ), A ≠ B ∧ A ∩ B = ∅ → (A.sum id) ≠ (B.sum id)

theorem maximum_sum_set (S : Finset ℕ) (h : ∀ x ∈ S, x ≤ 15) (h_subset_sum : no_two_disjoint_subsets_have_equal_sums S) : S.sum id = 61 :=
sorry

end NUMINAMATH_GPT_maximum_sum_set_l2318_231802


namespace NUMINAMATH_GPT_more_oaks_than_willows_l2318_231809

theorem more_oaks_than_willows (total_trees willows : ℕ) (h1 : total_trees = 83) (h2 : willows = 36) :
  (total_trees - willows) - willows = 11 :=
by
  sorry

end NUMINAMATH_GPT_more_oaks_than_willows_l2318_231809


namespace NUMINAMATH_GPT_remainder_product_l2318_231829

theorem remainder_product (a b c : ℤ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
sorry

end NUMINAMATH_GPT_remainder_product_l2318_231829


namespace NUMINAMATH_GPT_same_terminal_side_eq_l2318_231898

theorem same_terminal_side_eq (α : ℝ) : 
    (∃ k : ℤ, α = 2 * k * Real.pi - Real.pi / 3) ↔ α = 5 * Real.pi / 3 :=
by sorry

end NUMINAMATH_GPT_same_terminal_side_eq_l2318_231898


namespace NUMINAMATH_GPT_value_of_b_l2318_231879

theorem value_of_b (a b c y1 y2 y3 : ℝ)
( h1 : y1 = a + b + c )
( h2 : y2 = a - b + c )
( h3 : y3 = 4 * a + 2 * b + c )
( h4 : y1 - y2 = 8 )
( h5 : y3 = y1 + 2 )
: b = 4 :=
sorry

end NUMINAMATH_GPT_value_of_b_l2318_231879


namespace NUMINAMATH_GPT_compute_f_g_f_3_l2318_231868

def f (x : ℤ) : ℤ := 5 * x + 5
def g (x : ℤ) : ℤ := 6 * x + 4

theorem compute_f_g_f_3 : f (g (f 3)) = 625 := sorry

end NUMINAMATH_GPT_compute_f_g_f_3_l2318_231868


namespace NUMINAMATH_GPT_no_real_solution_equation_l2318_231855

theorem no_real_solution_equation (x : ℝ) (h : x ≠ -9) : 
  ¬ ∃ x, (8*x^2 + 90*x + 2) / (3*x + 27) = 4*x + 2 :=
by
  sorry

end NUMINAMATH_GPT_no_real_solution_equation_l2318_231855


namespace NUMINAMATH_GPT_right_triangle_segments_l2318_231833

open Real

theorem right_triangle_segments 
  (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) (h_ab : a > b)
  (P Q : ℝ × ℝ) (P_on_ellipse : P.1^2 / a^2 + P.2^2 / b^2 = 1)
  (Q_on_ellipse : Q.1^2 / a^2 + Q.2^2 / b^2 = 1)
  (Q_in_first_quad : Q.1 > 0 ∧ Q.2 > 0)
  (OQ_parallel_AP : ∃ k : ℝ, Q.1 = k * P.1 ∧ Q.2 = k * P.2)
  (M : ℝ × ℝ) (M_midpoint : M = ((P.1 + 0) / 2, (P.2 + 0) / 2))
  (R : ℝ × ℝ) (R_on_ellipse : R.1^2 / a^2 + R.2^2 / b^2 = 1)
  (OM_intersects_R : ∃ k : ℝ, R = (k * M.1, k * M.2))
: dist (0,0) Q ≠ 0 →
  dist (0,0) R ≠ 0 →
  dist (Q, R) ≠ 0 →
  dist (0,0) Q ^ 2 + dist (0,0) R ^ 2 = dist ((-a), (b)) ((a), (b)) ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_segments_l2318_231833


namespace NUMINAMATH_GPT_tennis_tournament_l2318_231824

theorem tennis_tournament (n x : ℕ) 
    (p : ℕ := 4 * n) 
    (m : ℕ := (p * (p - 1)) / 2) 
    (r_women : ℕ := 3 * x) 
    (r_men : ℕ := 2 * x) 
    (total_wins : ℕ := r_women + r_men) 
    (h_matches : m = total_wins) 
    (h_ratio : r_women = 3 * x ∧ r_men = 2 * x ∧ 4 * n * (4 * n - 1) = 10 * x): 
    n = 4 :=
by
  sorry

end NUMINAMATH_GPT_tennis_tournament_l2318_231824


namespace NUMINAMATH_GPT_smallest_coins_l2318_231837

theorem smallest_coins (n : ℕ) (n_min : ℕ) (h1 : ∃ n, n % 8 = 5 ∧ n % 7 = 4 ∧ n = 53) (h2 : n_min = n):
  (n_min ≡ 5 [MOD 8]) ∧ (n_min ≡ 4 [MOD 7]) ∧ (n_min = 53) ∧ (53 % 9 = 8) :=
by
  sorry

end NUMINAMATH_GPT_smallest_coins_l2318_231837


namespace NUMINAMATH_GPT_other_solution_of_quadratic_l2318_231851

theorem other_solution_of_quadratic (x : ℚ) (h₁ : 81 * 2/9 * 2/9 + 220 = 196 * 2/9 - 15) (h₂ : 81*x^2 - 196*x + 235 = 0) : x = 2/9 ∨ x = 5/9 :=
by
  sorry

end NUMINAMATH_GPT_other_solution_of_quadratic_l2318_231851


namespace NUMINAMATH_GPT_quadratic_equation_m_l2318_231873

theorem quadratic_equation_m (m b : ℝ) (h : (m - 2) * x ^ |m| - b * x - 1 = 0) : m = -2 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_equation_m_l2318_231873


namespace NUMINAMATH_GPT_average_salary_all_employees_l2318_231887

-- Define the given conditions
def average_salary_officers : ℝ := 440
def average_salary_non_officers : ℝ := 110
def number_of_officers : ℕ := 15
def number_of_non_officers : ℕ := 480

-- Define the proposition we need to prove
theorem average_salary_all_employees :
  let total_salary_officers := average_salary_officers * number_of_officers
  let total_salary_non_officers := average_salary_non_officers * number_of_non_officers
  let total_salary_all_employees := total_salary_officers + total_salary_non_officers
  let total_number_of_employees := number_of_officers + number_of_non_officers
  let average_salary_all_employees := total_salary_all_employees / total_number_of_employees
  average_salary_all_employees = 120 :=
by {
  -- Skipping the proof steps
  sorry
}

end NUMINAMATH_GPT_average_salary_all_employees_l2318_231887


namespace NUMINAMATH_GPT_exists_quadratic_polynomial_distinct_remainders_l2318_231847

theorem exists_quadratic_polynomial_distinct_remainders :
  ∃ (a b c : ℤ), 
    (¬ (2014 ∣ a)) ∧ 
    (∀ x y : ℤ, (1 ≤ x ∧ x ≤ 2014) ∧ (1 ≤ y ∧ y ≤ 2014) → x ≠ y → 
      (1007 * x^2 + 1008 * x + c) % 2014 ≠ (1007 * y^2 + 1008 * y + c) % 2014) :=
  sorry

end NUMINAMATH_GPT_exists_quadratic_polynomial_distinct_remainders_l2318_231847


namespace NUMINAMATH_GPT_smallest_A_l2318_231891

theorem smallest_A (A B C D E : ℕ) 
  (hA_even : A % 2 = 0)
  (hB_even : B % 2 = 0)
  (hC_even : C % 2 = 0)
  (hD_even : D % 2 = 0)
  (hE_even : E % 2 = 0)
  (hA_three_digit : 100 ≤ A ∧ A < 1000)
  (hB_three_digit : 100 ≤ B ∧ B < 1000)
  (hC_three_digit : 100 ≤ C ∧ C < 1000)
  (hD_three_digit : 100 ≤ D ∧ D < 1000)
  (hE_three_digit : 100 ≤ E ∧ E < 1000)
  (h_sorted : A < B ∧ B < C ∧ C < D ∧ D < E)
  (h_sum : A + B + C + D + E = 4306) :
  A = 326 :=
sorry

end NUMINAMATH_GPT_smallest_A_l2318_231891


namespace NUMINAMATH_GPT_volleyballs_remaining_l2318_231844

def initial_volleyballs := 9
def lent_volleyballs := 5

theorem volleyballs_remaining : initial_volleyballs - lent_volleyballs = 4 := 
by
  sorry

end NUMINAMATH_GPT_volleyballs_remaining_l2318_231844


namespace NUMINAMATH_GPT_marbles_total_l2318_231849

theorem marbles_total (fabian kyle miles : ℕ) (h1 : fabian = 3 * kyle) (h2 : fabian = 5 * miles) (h3 : fabian = 15) : kyle + miles = 8 := by
  sorry

end NUMINAMATH_GPT_marbles_total_l2318_231849


namespace NUMINAMATH_GPT_cristine_initial_lemons_l2318_231811

theorem cristine_initial_lemons (L : ℕ) (h : (3 / 4 : ℚ) * L = 9) : L = 12 :=
sorry

end NUMINAMATH_GPT_cristine_initial_lemons_l2318_231811


namespace NUMINAMATH_GPT_sum_of_integers_l2318_231886

theorem sum_of_integers (a b : ℕ) (h1 : a * b + a + b = 255) (h2 : a < 30) (h3 : b < 30) (h4 : a % 2 = 1) :
  a + b = 30 := 
sorry

end NUMINAMATH_GPT_sum_of_integers_l2318_231886


namespace NUMINAMATH_GPT_number_of_ways_to_enter_and_exit_l2318_231858

theorem number_of_ways_to_enter_and_exit (n : ℕ) (h : n = 4) : (n * n) = 16 := by
  sorry

end NUMINAMATH_GPT_number_of_ways_to_enter_and_exit_l2318_231858


namespace NUMINAMATH_GPT_pool_surface_area_l2318_231888

/-
  Given conditions:
  1. The width of the pool is 3 meters.
  2. The length of the pool is 10 meters.

  To prove:
  The surface area of the pool is 30 square meters.
-/
def width : ℕ := 3
def length : ℕ := 10
def surface_area (length width : ℕ) : ℕ := length * width

theorem pool_surface_area : surface_area length width = 30 := by
  unfold surface_area
  rfl

end NUMINAMATH_GPT_pool_surface_area_l2318_231888


namespace NUMINAMATH_GPT_coeff_x3_in_product_l2318_231846

theorem coeff_x3_in_product :
  let p1 := 3 * (Polynomial.X ^ 3) + 4 * (Polynomial.X ^ 2) + 5 * Polynomial.X + 6
  let p2 := 7 * (Polynomial.X ^ 2) + 8 * Polynomial.X + 9
  (Polynomial.coeff (p1 * p2) 3) = 94 :=
by
  sorry

end NUMINAMATH_GPT_coeff_x3_in_product_l2318_231846


namespace NUMINAMATH_GPT_selection_methods_count_l2318_231827

-- Define the number of female students
def num_female_students : ℕ := 3

-- Define the number of male students
def num_male_students : ℕ := 2

-- Define the total number of different selection methods
def total_selection_methods : ℕ := num_female_students + num_male_students

-- Prove that the total number of different selection methods is 5
theorem selection_methods_count : total_selection_methods = 5 := by
  sorry

end NUMINAMATH_GPT_selection_methods_count_l2318_231827


namespace NUMINAMATH_GPT_max_side_length_triangle_l2318_231878

theorem max_side_length_triangle (a b c : ℕ) (h_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_perimeter : a + b + c = 20) (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) : max a (max b c) = 9 := 
sorry

end NUMINAMATH_GPT_max_side_length_triangle_l2318_231878


namespace NUMINAMATH_GPT_inscribed_circle_radius_eq_3_l2318_231800

open Real

theorem inscribed_circle_radius_eq_3
  (a : ℝ) (A : ℝ) (p : ℝ) (r : ℝ)
  (h_eq_tri : ∀ (a : ℝ), A = (sqrt 3 / 4) * a^2)
  (h_perim : ∀ (a : ℝ), p = 3 * a)
  (h_area_perim : ∀ (a : ℝ), A = (3 / 2) * p) :
  r = 3 :=
by sorry

end NUMINAMATH_GPT_inscribed_circle_radius_eq_3_l2318_231800


namespace NUMINAMATH_GPT_math_problem_l2318_231856

-- Define the first part of the problem
def line_area_to_axes (line_eq : ℝ → ℝ → Prop) (x y : ℝ) : Prop :=
  line_eq x y ∧ x = 4 ∧ y = -4

-- Define the second part of the problem
def line_through_fixed_point (m : ℝ) : Prop :=
  ∃ (x y : ℝ), (m * x) + y + m = 0 ∧ x = -1 ∧ y = 0

-- Theorem combining both parts
theorem math_problem (line_eq : ℝ → ℝ → Prop) (m : ℝ) :
  (∃ x y, line_area_to_axes line_eq x y → 8 = (1 / 2) * 4 * 4) ∧ line_through_fixed_point m :=
sorry

end NUMINAMATH_GPT_math_problem_l2318_231856


namespace NUMINAMATH_GPT_new_shoes_last_for_two_years_l2318_231823

theorem new_shoes_last_for_two_years :
  let cost_repair := 11.50
  let cost_new := 28.00
  let increase_factor := 1.2173913043478261
  (cost_new / ((increase_factor) * cost_repair)) ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_new_shoes_last_for_two_years_l2318_231823


namespace NUMINAMATH_GPT_compute_expression_l2318_231893

theorem compute_expression : (3 + 7)^2 + (3^2 + 7^2 + 5^2) = 183 := by
  sorry

end NUMINAMATH_GPT_compute_expression_l2318_231893


namespace NUMINAMATH_GPT_inequality_proof_l2318_231889

theorem inequality_proof {a b c d e f : ℝ} (h : b^2 ≥ a^2 + c^2) : 
  (a * f - c * d)^2 ≤ (a * e - b * d)^2 + (b * f - c * e)^2 :=
sorry

end NUMINAMATH_GPT_inequality_proof_l2318_231889


namespace NUMINAMATH_GPT_circles_intersect_l2318_231896

theorem circles_intersect (m c : ℝ) (h1 : (1:ℝ) = (5 + (-m))) (h2 : (3:ℝ) = (5 + (c - (-2)))) :
  m + c = 3 :=
sorry

end NUMINAMATH_GPT_circles_intersect_l2318_231896


namespace NUMINAMATH_GPT_sum_of_inner_segments_l2318_231825

/-- Given the following conditions:
  1. The sum of the perimeters of the three quadrilaterals is 25 centimeters.
  2. The sum of the perimeters of the four triangles is 20 centimeters.
  3. The perimeter of triangle ABC is 19 centimeters.
Prove that AD + BE + CF = 13 centimeters. -/
theorem sum_of_inner_segments 
  (perimeter_quads : ℝ)
  (perimeter_tris : ℝ)
  (perimeter_ABC : ℝ)
  (hq : perimeter_quads = 25)
  (ht : perimeter_tris = 20)
  (hABC : perimeter_ABC = 19) 
  : AD + BE + CF = 13 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_inner_segments_l2318_231825


namespace NUMINAMATH_GPT_not_product_of_consecutive_integers_l2318_231866

theorem not_product_of_consecutive_integers (n k : ℕ) (hn : n > 0) (hk : k > 0) :
  ∀ (m : ℕ), 2 * (n ^ k) ^ 3 + 4 * (n ^ k) + 10 ≠ m * (m + 1) := by
sorry

end NUMINAMATH_GPT_not_product_of_consecutive_integers_l2318_231866


namespace NUMINAMATH_GPT_hyperbola_eccentricity_range_l2318_231850

theorem hyperbola_eccentricity_range (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) :
  (∃ P₁ P₂ : { p : ℝ × ℝ // p ≠ (0, b) ∧ p ≠ (c, 0) ∧ ((0, b) - p).1 * ((c, 0) - p).1 + ((0, b) - p).2 * ((c, 0) - p).2 = 0},
   true) -- This encodes the existence of the required points P₁ and P₂ on line segment BF excluding endpoints
  → 1 < (Real.sqrt ((a^2 + b^2) / a^2)) ∧ (Real.sqrt ((a^2 + b^2) / a^2)) < (Real.sqrt 5 + 1)/2 :=
sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_range_l2318_231850


namespace NUMINAMATH_GPT_set_P_equals_set_interval_l2318_231895

def A : Set ℝ := {x | x < 5}
def B : Set ℝ := {x | x <= 1 ∨ x >= 3}
def P : Set ℝ := {x | x ∈ A ∧ ¬ (x ∈ A ∧ x ∈ B)}

theorem set_P_equals_set_interval :
  P = {x | 1 < x ∧ x < 3} :=
sorry

end NUMINAMATH_GPT_set_P_equals_set_interval_l2318_231895


namespace NUMINAMATH_GPT_contrapositive_eq_inverse_l2318_231884

variable (p q : Prop)

theorem contrapositive_eq_inverse (h1 : p → q) :
  (¬ p → ¬ q) ↔ (q → p) := by
  sorry

end NUMINAMATH_GPT_contrapositive_eq_inverse_l2318_231884


namespace NUMINAMATH_GPT_chessboard_piece_arrangements_l2318_231897

-- Define the problem in Lean
theorem chessboard_piece_arrangements (black_pos white_pos : ℕ)
  (black_pos_neq_white_pos : black_pos ≠ white_pos)
  (valid_position : black_pos < 64 ∧ white_pos < 64) :
  ¬(∀ (move : ℕ → ℕ → Prop), (move black_pos white_pos) → ∃! (p : ℕ × ℕ), move (p.fst) (p.snd)) :=
by sorry

end NUMINAMATH_GPT_chessboard_piece_arrangements_l2318_231897


namespace NUMINAMATH_GPT_simplify_tan_expression_l2318_231870

noncomputable def tan_15 := Real.tan (Real.pi / 12)
noncomputable def tan_30 := Real.tan (Real.pi / 6)

theorem simplify_tan_expression : (1 + tan_15) * (1 + tan_30) = 2 := by
  sorry

end NUMINAMATH_GPT_simplify_tan_expression_l2318_231870


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l2318_231885

noncomputable 
def f (x : ℝ) : ℝ := Real.exp x

theorem problem1 
  (a b : ℝ)
  (h1 : f 1 = a) 
  (h2 : b = 0) : f x = Real.exp x :=
sorry

theorem problem2 
  (k : ℝ) 
  (h : ∀ x : ℝ, f x ≥ k * x) : 0 ≤ k ∧ k ≤ Real.exp 1 :=
sorry

theorem problem3 
  (t : ℝ)
  (h : t ≤ 2) : ∀ x : ℝ, f x > t + Real.log x :=
sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l2318_231885


namespace NUMINAMATH_GPT_second_lock_less_than_three_times_first_l2318_231872

variable (first_lock_time : ℕ := 5)
variable (second_lock_time : ℕ)
variable (combined_lock_time : ℕ := 60)

-- Assuming the second lock time is a fraction of the combined lock time
axiom h1 : 5 * second_lock_time = combined_lock_time

theorem second_lock_less_than_three_times_first : (3 * first_lock_time - second_lock_time) = 3 :=
by
  -- prove that the theorem is true based on given conditions.
  sorry

end NUMINAMATH_GPT_second_lock_less_than_three_times_first_l2318_231872


namespace NUMINAMATH_GPT_youngest_child_age_l2318_231882

theorem youngest_child_age :
  ∃ x : ℕ, x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 65 ∧ x = 7 :=
by
  sorry

end NUMINAMATH_GPT_youngest_child_age_l2318_231882


namespace NUMINAMATH_GPT_f_above_g_l2318_231815

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := (Real.exp x) / (x - m)
def g (x : ℝ) : ℝ := x^2 + x

theorem f_above_g (m : ℝ) (h₀ : 0 < m) (h₁ : m < 1/2) : 
  ∀ x, m ≤ x ∧ x ≤ m + 1 → f x m > g x := 
sorry

end NUMINAMATH_GPT_f_above_g_l2318_231815


namespace NUMINAMATH_GPT_reduced_fraction_numerator_l2318_231841

theorem reduced_fraction_numerator :
  let numerator := 4128 
  let denominator := 4386 
  let gcd := Nat.gcd numerator denominator
  let reduced_numerator := numerator / gcd 
  let reduced_denominator := denominator / gcd 
  (reduced_numerator : ℚ) / (reduced_denominator : ℚ) = 16 / 17 → reduced_numerator = 16 :=
by
  intros
  sorry

end NUMINAMATH_GPT_reduced_fraction_numerator_l2318_231841


namespace NUMINAMATH_GPT_graph_three_lines_no_common_point_l2318_231869

theorem graph_three_lines_no_common_point :
  ∀ x y : ℝ, x^2 * (x + 2*y - 3) = y^2 * (x + 2*y - 3) →
    x + 2*y - 3 = 0 ∨ x = y ∨ x = -y :=
by sorry

end NUMINAMATH_GPT_graph_three_lines_no_common_point_l2318_231869


namespace NUMINAMATH_GPT_count_ordered_triples_l2318_231894

theorem count_ordered_triples (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : a^2 = b^2 + c^2) (h5 : b^2 = a^2 + c^2) (h6 : c^2 = a^2 + b^2) : 
  (a = b ∧ b = c ∧ a ≠ 0) ∨ (a = -b ∧ b = c ∧ a ≠ 0) ∨ (a = b ∧ b = -c ∧ a ≠ 0) ∨ (a = -b ∧ b = -c ∧ a ≠ 0) :=
sorry

end NUMINAMATH_GPT_count_ordered_triples_l2318_231894


namespace NUMINAMATH_GPT_extremum_values_l2318_231836

def f (x : ℝ) : ℝ := x^3 - 3 * x^2 - 9 * x

theorem extremum_values :
  (∀ x, f x ≤ 5) ∧ f (-1) = 5 ∧ (∀ x, f x ≥ -27) ∧ f 3 = -27 :=
by
  sorry

end NUMINAMATH_GPT_extremum_values_l2318_231836


namespace NUMINAMATH_GPT_cricketer_average_score_l2318_231899

variable {A : ℤ} -- A represents the average score after 18 innings

theorem cricketer_average_score
  (h1 : (19 * (A + 4) = 18 * A + 98)) :
  A + 4 = 26 := by
  sorry

end NUMINAMATH_GPT_cricketer_average_score_l2318_231899


namespace NUMINAMATH_GPT_integer_cube_less_than_triple_l2318_231822

theorem integer_cube_less_than_triple (x : ℤ) : x^3 < 3 * x ↔ x = 0 :=
by 
  sorry

end NUMINAMATH_GPT_integer_cube_less_than_triple_l2318_231822


namespace NUMINAMATH_GPT_total_third_graders_l2318_231864

theorem total_third_graders (num_girls : ℕ) (num_boys : ℕ) (h1 : num_girls = 57) (h2 : num_boys = 66) : num_girls + num_boys = 123 :=
by
  sorry

end NUMINAMATH_GPT_total_third_graders_l2318_231864


namespace NUMINAMATH_GPT_mod_calculation_l2318_231835

theorem mod_calculation : (9^7 + 8^8 + 7^9) % 5 = 2 := by
  sorry

end NUMINAMATH_GPT_mod_calculation_l2318_231835


namespace NUMINAMATH_GPT_singles_percentage_l2318_231808

-- Definitions based on conditions
def total_hits : ℕ := 50
def home_runs : ℕ := 2
def triples : ℕ := 3
def doubles : ℕ := 7
def non_single_hits : ℕ := home_runs + triples + doubles
def singles : ℕ := total_hits - non_single_hits

-- Theorem based on the proof problem
theorem singles_percentage :
  singles = 38 ∧ (singles / total_hits : ℚ) * 100 = 76 := 
  by
    sorry

end NUMINAMATH_GPT_singles_percentage_l2318_231808


namespace NUMINAMATH_GPT_katie_more_games_l2318_231813

noncomputable def katie_games : ℕ := 57 + 39
noncomputable def friends_games : ℕ := 34
noncomputable def games_difference : ℕ := katie_games - friends_games

theorem katie_more_games : games_difference = 62 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_katie_more_games_l2318_231813


namespace NUMINAMATH_GPT_clay_capacity_second_box_l2318_231853

-- Define the dimensions and clay capacity of the first box
def height1 : ℕ := 4
def width1 : ℕ := 2
def length1 : ℕ := 3
def clay1 : ℕ := 24

-- Define the dimensions of the second box
def height2 : ℕ := 3 * height1
def width2 : ℕ := 2 * width1
def length2 : ℕ := length1

-- The volume relation
def volume_relation (height width length clay: ℕ) : ℕ :=
  height * width * length * clay

theorem clay_capacity_second_box (height1 width1 length1 clay1 : ℕ) (height2 width2 length2 : ℕ) :
  height1 = 4 →
  width1 = 2 →
  length1 = 3 →
  clay1 = 24 →
  height2 = 3 * height1 →
  width2 = 2 * width1 →
  length2 = length1 →
  volume_relation height2 width2 length2 1 = 6 * volume_relation height1 width1 length1 1 →
  volume_relation height2 width2 length2 clay1 / volume_relation height1 width1 length1 1 = 144 :=
by
  intros h1 w1 l1 c1 h2 w2 l2 vol_rel
  sorry

end NUMINAMATH_GPT_clay_capacity_second_box_l2318_231853


namespace NUMINAMATH_GPT_part_one_equation_of_line_part_two_equation_of_line_l2318_231880

-- Definition of line passing through a given point
def line_through_point (a b : ℝ) (P : ℝ × ℝ) : Prop := P.1 / a + P.2 / b = 1

-- Condition: the sum of intercepts is 12
def sum_of_intercepts (a b : ℝ) : Prop := a + b = 12

-- Condition: area of triangle is 12
def area_of_triangle (a b : ℝ) : Prop := (1/2) * (abs (a * b)) = 12

-- First part: equation of the line when the sum of intercepts is 12
theorem part_one_equation_of_line (a b : ℝ) : 
  (line_through_point a b (3, 2)) ∧ (sum_of_intercepts a b) →
  (∃ x, (x = 2 ∧ (2*x)+x - 8 = 0) ∨ (x = 3 ∧ x + 3*x - 9 = 0)) :=
by
  sorry

-- Second part: equation of the line when the area of the triangle is 12
theorem part_two_equation_of_line (a b : ℝ) : 
  (line_through_point a b (3, 2)) ∧ (area_of_triangle a b) →
  ∃ x, x = 2 ∧ (2*x + 3*x - 12 = 0) :=
by
  sorry

end NUMINAMATH_GPT_part_one_equation_of_line_part_two_equation_of_line_l2318_231880


namespace NUMINAMATH_GPT_triangle_30_60_90_PQ_l2318_231867

theorem triangle_30_60_90_PQ (PR : ℝ) (hPR : PR = 18 * Real.sqrt 3) : 
  ∃ PQ : ℝ, PQ = 54 :=
by
  sorry

end NUMINAMATH_GPT_triangle_30_60_90_PQ_l2318_231867


namespace NUMINAMATH_GPT_sugar_ratio_l2318_231877

theorem sugar_ratio (r : ℝ) (H1 : 24 * r^3 = 3) : (24 * r / 24 = 1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_sugar_ratio_l2318_231877


namespace NUMINAMATH_GPT_certain_number_is_11_l2318_231852

theorem certain_number_is_11 (x : ℝ) (h : 15 * x = 165) : x = 11 :=
by {
  sorry
}

end NUMINAMATH_GPT_certain_number_is_11_l2318_231852


namespace NUMINAMATH_GPT_alex_buys_15_pounds_of_rice_l2318_231807

theorem alex_buys_15_pounds_of_rice (r b : ℝ) 
  (h1 : r + b = 30)
  (h2 : 75 * r + 35 * b = 1650) : 
  r = 15.0 := sorry

end NUMINAMATH_GPT_alex_buys_15_pounds_of_rice_l2318_231807


namespace NUMINAMATH_GPT_avg_move_to_california_l2318_231892

noncomputable def avg_people_per_hour (total_people : ℕ) (total_days : ℕ) : ℕ :=
  let total_hours := total_days * 24
  let avg_per_hour := total_people / total_hours
  let remainder := total_people % total_hours
  if remainder * 2 < total_hours then avg_per_hour else avg_per_hour + 1

theorem avg_move_to_california : avg_people_per_hour 3500 5 = 29 := by
  sorry

end NUMINAMATH_GPT_avg_move_to_california_l2318_231892


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l2318_231816

theorem sufficient_but_not_necessary (x : ℝ) :
  (|x - 3| - |x - 1| < 2) → x ≠ 1 ∧ ¬ (∀ x : ℝ, x ≠ 1 → |x - 3| - |x - 1| < 2) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l2318_231816


namespace NUMINAMATH_GPT_value_of_b_l2318_231860

-- Definitions
def A := 45  -- in degrees
def B := 60  -- in degrees
def a := 10  -- length of side a

-- Assertion
theorem value_of_b : (b : ℝ) = 5 * Real.sqrt 6 :=
by
  -- Definitions used in previous problem conditions
  let sin_A := Real.sin (Real.pi * A / 180)
  let sin_B := Real.sin (Real.pi * B / 180)
  -- Applying the Law of Sines
  have law_of_sines := (a / sin_A) = (b / sin_B)
  -- Simplified calculation of b (not provided here; proof required later)
  sorry

end NUMINAMATH_GPT_value_of_b_l2318_231860


namespace NUMINAMATH_GPT_solve_inequality_l2318_231876

theorem solve_inequality (a : ℝ) : 
    (∀ x : ℝ, x^2 + (a + 2)*x + 2*a < 0 ↔ 
        (if a < 2 then -2 < x ∧ x < -a
         else if a = 2 then false
         else -a < x ∧ x < -2)) :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l2318_231876


namespace NUMINAMATH_GPT_cos_alpha_value_l2318_231861

-- Define our conditions
variables (α : ℝ)
axiom sin_alpha : Real.sin α = -5 / 13
axiom tan_alpha_pos : Real.tan α > 0

-- State our goal
theorem cos_alpha_value : Real.cos α = -12 / 13 :=
by
  sorry

end NUMINAMATH_GPT_cos_alpha_value_l2318_231861


namespace NUMINAMATH_GPT_kenny_trumpet_hours_l2318_231865

variables (x y : ℝ)
def basketball_hours := 10
def running_hours := 2 * basketball_hours
def trumpet_hours := 2 * running_hours

theorem kenny_trumpet_hours (x y : ℝ) (H : basketball_hours + running_hours + trumpet_hours = x + y) :
  trumpet_hours = 40 :=
by
  sorry

end NUMINAMATH_GPT_kenny_trumpet_hours_l2318_231865


namespace NUMINAMATH_GPT_people_owning_only_cats_and_dogs_l2318_231803

theorem people_owning_only_cats_and_dogs 
  (total_people : ℕ) 
  (only_dogs : ℕ) 
  (only_cats : ℕ) 
  (cats_dogs_snakes : ℕ) 
  (total_snakes : ℕ) 
  (only_cats_and_dogs : ℕ) 
  (h1 : total_people = 89) 
  (h2 : only_dogs = 15) 
  (h3 : only_cats = 10) 
  (h4 : cats_dogs_snakes = 3) 
  (h5 : total_snakes = 59) 
  (h6 : total_people = only_dogs + only_cats + only_cats_and_dogs + cats_dogs_snakes + (total_snakes - cats_dogs_snakes)) : 
  only_cats_and_dogs = 5 := 
by 
  sorry

end NUMINAMATH_GPT_people_owning_only_cats_and_dogs_l2318_231803


namespace NUMINAMATH_GPT_parallel_line_through_point_l2318_231857

theorem parallel_line_through_point :
  ∃ c : ℝ, ∀ x y : ℝ, (x = -1) → (y = 3) → (x - 2*y + 3 = 0) → (x - 2*y + c = 0) :=
sorry

end NUMINAMATH_GPT_parallel_line_through_point_l2318_231857


namespace NUMINAMATH_GPT_linear_function_not_in_second_quadrant_l2318_231831

theorem linear_function_not_in_second_quadrant (m : ℤ) (h1 : m + 4 > 0) (h2 : m + 2 ≤ 0) : 
  m = -3 ∨ m = -2 := 
sorry

end NUMINAMATH_GPT_linear_function_not_in_second_quadrant_l2318_231831


namespace NUMINAMATH_GPT_find_age_of_30th_student_l2318_231826

theorem find_age_of_30th_student :
  let avg1 := 23.5
  let n1 := 30
  let avg2 := 21.3
  let n2 := 9
  let avg3 := 19.7
  let n3 := 12
  let avg4 := 24.2
  let n4 := 7
  let avg5 := 35
  let n5 := 1
  let total_age_30 := n1 * avg1
  let total_age_9 := n2 * avg2
  let total_age_12 := n3 * avg3
  let total_age_7 := n4 * avg4
  let total_age_1 := n5 * avg5
  let total_age_29 := total_age_9 + total_age_12 + total_age_7 + total_age_1
  let age_30th := total_age_30 - total_age_29
  age_30th = 72.5 :=
by
  sorry

end NUMINAMATH_GPT_find_age_of_30th_student_l2318_231826


namespace NUMINAMATH_GPT_larger_integer_of_two_integers_diff_8_prod_120_l2318_231805

noncomputable def larger_integer (a b : ℕ) : ℕ :=
if a > b then a else b

theorem larger_integer_of_two_integers_diff_8_prod_120 (a b : ℕ) 
  (h_diff : a - b = 8) 
  (h_product : a * b = 120) 
  (h_positive_a : 0 < a) 
  (h_positive_b : 0 < b) : larger_integer a b = 20 := by
  sorry

end NUMINAMATH_GPT_larger_integer_of_two_integers_diff_8_prod_120_l2318_231805


namespace NUMINAMATH_GPT_proof_probability_second_science_given_first_arts_l2318_231830

noncomputable def probability_second_science_given_first_arts : ℚ :=
  let total_questions := 5
  let science_questions := 3
  let arts_questions := 2

  -- Event A: drawing an arts question in the first draw.
  let P_A := arts_questions / total_questions

  -- Event AB: drawing an arts question in the first draw and a science question in the second draw.
  let P_AB := (arts_questions / total_questions) * (science_questions / (total_questions - 1))

  -- Conditional probability P(B|A): drawing a science question in the second draw given drawing an arts question in the first draw.
  P_AB / P_A

theorem proof_probability_second_science_given_first_arts :
  probability_second_science_given_first_arts = 3 / 4 :=
by
  -- Lean does not include the proof in the statement as required.
  sorry

end NUMINAMATH_GPT_proof_probability_second_science_given_first_arts_l2318_231830


namespace NUMINAMATH_GPT_color_nat_two_colors_no_sum_power_of_two_l2318_231890

theorem color_nat_two_colors_no_sum_power_of_two :
  ∃ (f : ℕ → ℕ), (∀ a b : ℕ, a ≠ b → f a = f b → ∃ c : ℕ, c > 0 ∧ c ≠ 1 ∧ c ≠ 2 ∧ (a + b ≠ 2 ^ c)) :=
sorry

end NUMINAMATH_GPT_color_nat_two_colors_no_sum_power_of_two_l2318_231890


namespace NUMINAMATH_GPT_factorization_correct_l2318_231817

theorem factorization_correct :
  (∀ x : ℝ, x^2 - 6*x + 9 = (x - 3)^2) :=
by
  sorry

end NUMINAMATH_GPT_factorization_correct_l2318_231817
