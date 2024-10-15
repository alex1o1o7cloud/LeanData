import Mathlib

namespace NUMINAMATH_GPT_lowest_position_l2398_239822

theorem lowest_position (num_cyclists : ℕ) (num_stages : ℕ) (vasya_position : ℕ) :
  num_cyclists = 500 →
  num_stages = 15 →
  vasya_position = 7 →
  ∃ n, n = 91 :=
by
  intros
  sorry

end NUMINAMATH_GPT_lowest_position_l2398_239822


namespace NUMINAMATH_GPT_geom_prog_identity_l2398_239887

-- Define that A, B, C are the n-th, p-th, and k-th terms respectively of the same geometric progression.
variables (a r : ℝ) (n p k : ℕ) (A B C : ℝ)

-- Assume A = ar^(n-1), B = ar^(p-1), C = ar^(k-1)
def isGP (a r : ℝ) (n p k : ℕ) (A B C : ℝ) : Prop :=
  A = a * r^(n-1) ∧ B = a * r^(p-1) ∧ C = a * r^(k-1)

-- Define the statement to be proved
theorem geom_prog_identity (h : isGP a r n p k A B C) : A^(p-k) * B^(k-n) * C^(n-p) = 1 :=
sorry

end NUMINAMATH_GPT_geom_prog_identity_l2398_239887


namespace NUMINAMATH_GPT_opposite_of_neg_two_is_two_l2398_239829

theorem opposite_of_neg_two_is_two (a : ℤ) (h : a = -2) : a + 2 = 0 := by
  rw [h]
  norm_num

end NUMINAMATH_GPT_opposite_of_neg_two_is_two_l2398_239829


namespace NUMINAMATH_GPT_registration_methods_for_5_students_l2398_239865

def number_of_registration_methods (students groups : ℕ) : ℕ :=
  groups ^ students

theorem registration_methods_for_5_students : number_of_registration_methods 5 2 = 32 := by
  sorry

end NUMINAMATH_GPT_registration_methods_for_5_students_l2398_239865


namespace NUMINAMATH_GPT_odd_and_increasing_l2398_239880

-- Define the function f(x) = e^x - e^{-x}
noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)

-- We want to prove that this function is both odd and increasing.
theorem odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f (x)) ∧ (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂) :=
sorry

end NUMINAMATH_GPT_odd_and_increasing_l2398_239880


namespace NUMINAMATH_GPT_range_of_m_l2398_239899

theorem range_of_m (m : ℝ) :
  (¬ ∃ x : ℝ, 0 < x ∧ mx^2 + 2 * x + m > 0) →
  m ≤ -1 := by
  sorry

end NUMINAMATH_GPT_range_of_m_l2398_239899


namespace NUMINAMATH_GPT_interval_1_5_frequency_is_0_70_l2398_239836

-- Define the intervals and corresponding frequencies
def intervals : List (ℤ × ℤ) := [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]

def frequencies : List ℕ := [1, 1, 2, 3, 1, 2]

-- Sample capacity
def sample_capacity : ℕ := 10

-- Calculate the frequency of the sample in the interval [1,5)
noncomputable def frequency_in_interval_1_5 : ℝ := (frequencies.take 4).sum / sample_capacity

-- Prove that the frequency in the interval [1,5) is 0.70
theorem interval_1_5_frequency_is_0_70 : frequency_in_interval_1_5 = 0.70 := by
  sorry

end NUMINAMATH_GPT_interval_1_5_frequency_is_0_70_l2398_239836


namespace NUMINAMATH_GPT_total_coins_l2398_239879

-- Define the number of stacks and the number of coins per stack
def stacks : ℕ := 5
def coins_per_stack : ℕ := 3

-- State the theorem to prove the total number of coins
theorem total_coins (s c : ℕ) (hs : s = stacks) (hc : c = coins_per_stack) : s * c = 15 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_total_coins_l2398_239879


namespace NUMINAMATH_GPT_mean_of_remaining_four_numbers_l2398_239819

theorem mean_of_remaining_four_numbers (a b c d : ℝ) 
  (h_mean_five : (a + b + c + d + 120) / 5 = 100) : 
  (a + b + c + d) / 4 = 95 :=
by
  sorry

end NUMINAMATH_GPT_mean_of_remaining_four_numbers_l2398_239819


namespace NUMINAMATH_GPT_expression_equality_l2398_239811

theorem expression_equality :
  (5 + 2) * (5^2 + 2^2) * (5^4 + 2^4) * (5^8 + 2^8) * (5^16 + 2^16) * (5^32 + 2^32) * (5^64 + 2^64) = 5^128 - 2^128 := 
  sorry

end NUMINAMATH_GPT_expression_equality_l2398_239811


namespace NUMINAMATH_GPT_volleyball_practice_start_time_l2398_239831

def homework_time := 1 * 60 + 59  -- convert 1:59 p.m. to minutes since 12:00 p.m.
def homework_duration := 96        -- duration in minutes
def buffer_time := 25              -- time between finishing homework and practice
def practice_start_time := 4 * 60  -- convert 4:00 p.m. to minutes since 12:00 p.m.

theorem volleyball_practice_start_time :
  homework_time + homework_duration + buffer_time = practice_start_time := 
by
  sorry

end NUMINAMATH_GPT_volleyball_practice_start_time_l2398_239831


namespace NUMINAMATH_GPT_find_BE_l2398_239850

-- Definitions from the conditions
variable {A B C D E : Point}
variable (AB BC CA BD BE CE : ℝ)
variable (angleBAE angleCAD : Real.Angle)

-- Given conditions
axiom h1 : AB = 12
axiom h2 : BC = 17
axiom h3 : CA = 15
axiom h4 : BD = 7
axiom h5 : angleBAE = angleCAD

-- Required proof statement
theorem find_BE :
  BE = 1632 / 201 := by
  sorry

end NUMINAMATH_GPT_find_BE_l2398_239850


namespace NUMINAMATH_GPT_probability_at_least_one_card_each_cousin_correct_l2398_239867

noncomputable def probability_at_least_one_card_each_cousin : ℚ :=
  let total_cards := 16
  let cards_per_cousin := 8
  let selections := 3
  let total_ways := Nat.choose total_cards selections
  let ways_all_from_one_cousin := Nat.choose cards_per_cousin selections * 2  -- twice: once for each cousin
  let prob_all_from_one_cousin := (ways_all_from_one_cousin : ℚ) / total_ways
  1 - prob_all_from_one_cousin

theorem probability_at_least_one_card_each_cousin_correct :
  probability_at_least_one_card_each_cousin = 4 / 5 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_probability_at_least_one_card_each_cousin_correct_l2398_239867


namespace NUMINAMATH_GPT_championship_outcomes_l2398_239846

theorem championship_outcomes (students events : ℕ) (h_students : students = 3) (h_events : events = 2) : 
  students ^ events = 9 :=
by
  rw [h_students, h_events]
  have h : 3 ^ 2 = 9 := by norm_num
  exact h

end NUMINAMATH_GPT_championship_outcomes_l2398_239846


namespace NUMINAMATH_GPT_sum_of_permutations_of_1234567_l2398_239872

theorem sum_of_permutations_of_1234567 : 
  let factorial_7 := 5040
  let sum_of_digits := 1 + 2 + 3 + 4 + 5 + 6 + 7
  let geometric_series_sum := (10 ^ 7 - 1) / (10 - 1)
  sum_of_digits * factorial_7 * geometric_series_sum = 22399997760 :=
by
  let factorial_7 := 5040
  let sum_of_digits := 1 + 2 + 3 + 4 + 5 + 6 + 7
  let geometric_series_sum := (10^7 - 1) / (10 - 1)
  sorry

end NUMINAMATH_GPT_sum_of_permutations_of_1234567_l2398_239872


namespace NUMINAMATH_GPT_find_N_l2398_239877

theorem find_N (N : ℕ) (h_pos : N > 0) (h_small_factors : 1 + 3 = 4) 
  (h_large_factors : N + N / 3 = 204) : N = 153 :=
  by sorry

end NUMINAMATH_GPT_find_N_l2398_239877


namespace NUMINAMATH_GPT_ratio_of_good_states_l2398_239895

theorem ratio_of_good_states (n : ℕ) :
  let total_states := 2^(2*n)
  let good_states := Nat.choose (2 * n) n
  good_states / total_states = (List.range n).foldr (fun i acc => acc * (2*i+1)) 1 / (2^n * Nat.factorial n) := sorry

end NUMINAMATH_GPT_ratio_of_good_states_l2398_239895


namespace NUMINAMATH_GPT_maximize_probability_sum_is_15_l2398_239883

def initial_list : List ℤ := [-1, 0, 1, 2, 3, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16]

def valid_pairs (lst : List ℤ) : List (ℤ × ℤ) :=
  (lst.product lst).filter (λ ⟨x, y⟩ => x < y ∧ x + y = 15)

def remove_one_element (lst : List ℤ) (x : ℤ) : List ℤ :=
  lst.erase x

theorem maximize_probability_sum_is_15 :
  (List.length (valid_pairs (remove_one_element initial_list 8))
   = List.maximum (List.map (λ x => List.length (valid_pairs (remove_one_element initial_list x))) initial_list)) :=
sorry

end NUMINAMATH_GPT_maximize_probability_sum_is_15_l2398_239883


namespace NUMINAMATH_GPT_value_of_n_l2398_239882

-- Definitions of the question and conditions
def is_3_digit_integer (x : ℕ) : Prop := 100 ≤ x ∧ x < 1000
def not_divisible_by (x : ℕ) (d : ℕ) : Prop := ¬ (d ∣ x)

def problem (m n : ℕ) : Prop :=
  lcm m n = 690 ∧ is_3_digit_integer n ∧ not_divisible_by n 3 ∧ not_divisible_by m 2

-- The theorem to prove
theorem value_of_n {m n : ℕ} (h : problem m n) : n = 230 :=
sorry

end NUMINAMATH_GPT_value_of_n_l2398_239882


namespace NUMINAMATH_GPT_employed_population_percentage_l2398_239897

theorem employed_population_percentage
  (P : ℝ) -- Total population
  (E : ℝ) -- Fraction of population that is employed
  (employed_males : ℝ) -- Fraction of population that is employed males
  (employed_females_fraction : ℝ)
  (h1 : employed_males = 0.8 * P)
  (h2 : employed_females_fraction = 1 / 3) :
  E = 0.6 :=
by
  -- We don't need the proof here.
  sorry

end NUMINAMATH_GPT_employed_population_percentage_l2398_239897


namespace NUMINAMATH_GPT_provisions_last_days_l2398_239878

def num_soldiers_initial : ℕ := 1200
def daily_consumption_initial : ℝ := 3
def initial_duration : ℝ := 30
def extra_soldiers : ℕ := 528
def daily_consumption_new : ℝ := 2.5

noncomputable def total_provisions : ℝ := num_soldiers_initial * daily_consumption_initial * initial_duration
noncomputable def total_soldiers_after_joining : ℕ := num_soldiers_initial + extra_soldiers
noncomputable def new_daily_consumption : ℝ := total_soldiers_after_joining * daily_consumption_new

theorem provisions_last_days : (total_provisions / new_daily_consumption) = 25 := by
  sorry

end NUMINAMATH_GPT_provisions_last_days_l2398_239878


namespace NUMINAMATH_GPT_complement_A_in_U_range_of_a_l2398_239853

open Set Real

noncomputable def U : Set ℝ := univ
noncomputable def f (x : ℝ) : ℝ := (1 / (sqrt (x + 2))) + log (3 - x)
noncomputable def A : Set ℝ := {x | -2 < x ∧ x < 3}
noncomputable def B (a : ℝ) : Set ℝ := {x | a < x ∧ x < (2 * a - 1)}

theorem complement_A_in_U : compl A = {x | x ≤ -2 ∨ 3 ≤ x} :=
by {
  sorry
}

theorem range_of_a (a : ℝ) (h : A ∪ B a = A) : a ∈ Iic 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_complement_A_in_U_range_of_a_l2398_239853


namespace NUMINAMATH_GPT_condition_two_eqn_l2398_239886

def line_through_point_and_perpendicular (x1 y1 : ℝ) (c : ℝ) : Prop :=
  ∀ x y : ℝ, (y - y1) = -1/(x - x1) * (x - x1 + c) → x - y + c = 0

theorem condition_two_eqn :
  line_through_point_and_perpendicular 1 (-2) (-3) :=
sorry

end NUMINAMATH_GPT_condition_two_eqn_l2398_239886


namespace NUMINAMATH_GPT_triangle_angle_identity_l2398_239859

def triangle_angles_arithmetic_sequence (A B C : ℝ) : Prop :=
  A + C = 2 * B

def sum_of_triangle_angles (A B C : ℝ) : Prop :=
  A + B + C = 180

def angle_B_is_60 (B : ℝ) : Prop :=
  B = 60

theorem triangle_angle_identity (A B C a b c : ℝ)
  (h1 : triangle_angles_arithmetic_sequence A B C)
  (h2 : sum_of_triangle_angles A B C)
  (h3 : angle_B_is_60 B) : 
  1 / (a + b) + 1 / (b + c) = 3 / (a + b + c) :=
by 
  sorry

end NUMINAMATH_GPT_triangle_angle_identity_l2398_239859


namespace NUMINAMATH_GPT_perimeter_of_large_square_l2398_239821

theorem perimeter_of_large_square (squares : List ℕ) (h : squares = [1, 1, 2, 3, 5, 8, 13]) : 2 * (21 + 13) = 68 := by
  sorry

end NUMINAMATH_GPT_perimeter_of_large_square_l2398_239821


namespace NUMINAMATH_GPT_hyperbola_eccentricity_cond_l2398_239868

def hyperbola_eccentricity_condition (m : ℝ) : Prop :=
  let a := Real.sqrt m
  let b := Real.sqrt 3
  let c := Real.sqrt (m + 3)
  let e := 2
  (e * e) = (c * c) / (a * a)

theorem hyperbola_eccentricity_cond (m : ℝ) :
  hyperbola_eccentricity_condition m ↔ m = 1 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_cond_l2398_239868


namespace NUMINAMATH_GPT_total_children_in_circle_l2398_239861

theorem total_children_in_circle 
  (n : ℕ)  -- number of children
  (h_even : Even n)   -- condition: the circle is made up of an even number of children
  (h_pos : n > 0) -- condition: there are some children
  (h_opposite : (15 % n + 15 % n) % n = 0)  -- condition: the 15th child clockwise from Child A is facing Child A (implies opposite)
  : n = 30 := 
sorry

end NUMINAMATH_GPT_total_children_in_circle_l2398_239861


namespace NUMINAMATH_GPT_polynomial_abs_sum_l2398_239815

theorem polynomial_abs_sum (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ)
  (h : (2*X - 1)^5 = a_5 * X^5 + a_4 * X^4 + a_3 * X^3 + a_2 * X^2 + a_1 * X + a_0) :
  |a_0| + |a_1| + |a_2| + |a_3| + |a_4| + |a_5| = 243 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_abs_sum_l2398_239815


namespace NUMINAMATH_GPT_inequality_always_holds_l2398_239892

variable {a b : ℝ}

theorem inequality_always_holds (ha : a > 0) (hb : b < 0) : 1 / a > 1 / b :=
by
  sorry

end NUMINAMATH_GPT_inequality_always_holds_l2398_239892


namespace NUMINAMATH_GPT_total_questions_l2398_239858

theorem total_questions (qmc : ℕ) (qtotal : ℕ) (h1 : 10 = qmc) (h2 : qmc = (20 / 100) * qtotal) : qtotal = 50 :=
sorry

end NUMINAMATH_GPT_total_questions_l2398_239858


namespace NUMINAMATH_GPT_john_speed_above_limit_l2398_239848

def distance : ℝ := 150
def time : ℝ := 2
def speed_limit : ℝ := 60

theorem john_speed_above_limit :
  distance / time - speed_limit = 15 :=
by
  sorry

end NUMINAMATH_GPT_john_speed_above_limit_l2398_239848


namespace NUMINAMATH_GPT_new_average_is_ten_l2398_239863

-- Define the initial conditions
def initial_sum (x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ x₉ : ℝ) : Prop :=
  x₁ + x₂ + x₃ + x₄ + x₅ + x₆ + x₇ + x₈ + x₉ = 9 * 7

-- Define the transformation on the nine numbers
def transformed_sum (x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ x₉ : ℝ) : ℝ :=
  (x₁ - 3) + (x₂ - 3) + (x₃ - 3) +
  (x₄ + 5) + (x₅ + 5) + (x₆ + 5) +
  (2 * x₇) + (2 * x₈) + (2 * x₉)

-- The theorem to prove the new average is 10
theorem new_average_is_ten (x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ x₉ : ℝ) 
  (h : initial_sum x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ x₉) :
  transformed_sum x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ x₉ / 9 = 10 :=
by 
  sorry

end NUMINAMATH_GPT_new_average_is_ten_l2398_239863


namespace NUMINAMATH_GPT_inequality_l2398_239830

variable (a b c : ℝ)

noncomputable def condition (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 1 / 8

theorem inequality (h : condition a b c) : 
  a^2 + b^2 + c^2 + a^2 * b^2 + a^2 * c^2 + b^2 * c^2 ≥ 15 / 16 :=
sorry

end NUMINAMATH_GPT_inequality_l2398_239830


namespace NUMINAMATH_GPT_min_teachers_required_l2398_239893

-- Define the conditions
def num_english_teachers : ℕ := 9
def num_history_teachers : ℕ := 7
def num_geography_teachers : ℕ := 6
def max_subjects_per_teacher : ℕ := 2

-- The proposition we want to prove
theorem min_teachers_required :
  ∃ (t : ℕ), t = 13 ∧
    t * max_subjects_per_teacher ≥ num_english_teachers + num_history_teachers + num_geography_teachers :=
sorry

end NUMINAMATH_GPT_min_teachers_required_l2398_239893


namespace NUMINAMATH_GPT_distance_point_C_to_line_is_2_inch_l2398_239816

/-- 
Four 2-inch squares are aligned in a straight line. The second square from the left is rotated 90 degrees, 
and then shifted vertically downward until it touches the adjacent squares. Prove that the distance from 
point C, the top vertex of the rotated square, to the original line on which the bases of the squares were 
placed is 2 inches.
-/
theorem distance_point_C_to_line_is_2_inch :
  ∀ (squares : Fin 4 → ℝ) (rotation : ℝ) (vertical_shift : ℝ) (C_position : ℝ),
  (∀ n : Fin 4, squares n = 2) →
  rotation = 90 →
  vertical_shift = 0 →
  C_position = 2 →
  C_position = 2 :=
by
  intros squares rotation vertical_shift C_position
  sorry

end NUMINAMATH_GPT_distance_point_C_to_line_is_2_inch_l2398_239816


namespace NUMINAMATH_GPT_yield_and_fertilization_correlated_l2398_239856

-- Define the variables and conditions
def yield_of_crops : Type := sorry
def fertilization : Type := sorry

-- State the condition
def yield_depends_on_fertilization (Y : yield_of_crops) (F : fertilization) : Prop :=
  -- The yield of crops depends entirely on fertilization
  sorry

-- State the theorem with the given condition and the conclusion
theorem yield_and_fertilization_correlated {Y : yield_of_crops} {F : fertilization} :
  yield_depends_on_fertilization Y F → sorry := 
  -- There is a correlation between the yield of crops and fertilization
  sorry

end NUMINAMATH_GPT_yield_and_fertilization_correlated_l2398_239856


namespace NUMINAMATH_GPT_proof_problem_l2398_239857

noncomputable def a {α : Type*} [LinearOrderedField α] : α := sorry
noncomputable def b {α : Type*} [LinearOrderedField α] : α := sorry
noncomputable def c {α : Type*} [LinearOrderedField α] : α := sorry
noncomputable def d {α : Type*} [LinearOrderedField α] : α := sorry

theorem proof_problem (a b c d : ℝ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) 
(hprod : a * b * c * d = 1) : 
a^5 + b^5 + c^5 + d^5 ≥ a + b + c + d :=
sorry

end NUMINAMATH_GPT_proof_problem_l2398_239857


namespace NUMINAMATH_GPT_lcm_24_150_is_600_l2398_239804

noncomputable def lcm_24_150 : ℕ :=
  let a := 24
  let b := 150
  have h₁ : a = 2^3 * 3 := by sorry
  have h₂ : b = 2 * 3 * 5^2 := by sorry
  Nat.lcm a b

theorem lcm_24_150_is_600 : lcm_24_150 = 600 := by
  -- Use provided primes conditions to derive the result
  sorry

end NUMINAMATH_GPT_lcm_24_150_is_600_l2398_239804


namespace NUMINAMATH_GPT_find_amount_after_two_years_l2398_239824

noncomputable def initial_value : ℝ := 64000
noncomputable def yearly_increase (amount : ℝ) : ℝ := amount / 9
noncomputable def amount_after_year (amount : ℝ) : ℝ := amount + yearly_increase amount
noncomputable def amount_after_two_years : ℝ := amount_after_year (amount_after_year initial_value)

theorem find_amount_after_two_years : amount_after_two_years = 79012.34 :=
by
  sorry

end NUMINAMATH_GPT_find_amount_after_two_years_l2398_239824


namespace NUMINAMATH_GPT_some_employee_not_team_leader_l2398_239827

variables (Employee : Type) (isTeamLeader : Employee → Prop) (meetsDeadline : Employee → Prop)

-- Conditions
axiom some_employee_not_meets_deadlines : ∃ e : Employee, ¬ meetsDeadline e
axiom all_team_leaders_meet_deadlines : ∀ e : Employee, isTeamLeader e → meetsDeadline e

-- Theorem to prove
theorem some_employee_not_team_leader : ∃ e : Employee, ¬ isTeamLeader e :=
sorry

end NUMINAMATH_GPT_some_employee_not_team_leader_l2398_239827


namespace NUMINAMATH_GPT_problem_mod_1000_l2398_239838

noncomputable def M : ℕ := Nat.choose 18 9

theorem problem_mod_1000 : M % 1000 = 620 := by
  sorry

end NUMINAMATH_GPT_problem_mod_1000_l2398_239838


namespace NUMINAMATH_GPT_factorization_identity_sum_l2398_239898

theorem factorization_identity_sum (a b c : ℤ)
  (h1 : ∀ x : ℤ, x^2 + 15 * x + 36 = (x + a) * (x + b))
  (h2 : ∀ x : ℤ, x^2 + 7 * x - 60 = (x + b) * (x - c)) :
  a + b + c = 20 :=
sorry

end NUMINAMATH_GPT_factorization_identity_sum_l2398_239898


namespace NUMINAMATH_GPT_triangle_ratio_l2398_239875

theorem triangle_ratio
  (D E F X : Type)
  [DecidableEq D] [DecidableEq E] [DecidableEq F] [DecidableEq X]
  (DE DF : ℝ)
  (hDE : DE = 36)
  (hDF : DF = 40)
  (DX_bisects_EDF : ∀ EX FX, (DE * FX = DF * EX)) :
  ∃ (EX FX : ℝ), EX / FX = 9 / 10 :=
sorry

end NUMINAMATH_GPT_triangle_ratio_l2398_239875


namespace NUMINAMATH_GPT_find_angle_C_find_max_area_l2398_239874

variable {A B C a b c : ℝ}

-- Given Conditions
def condition1 (c B a b C : ℝ) := c * Real.cos B + (b - 2 * a) * Real.cos C = 0
def condition2 (c : ℝ) := c = 2 * Real.sqrt 3

-- Problem (1): Prove the size of angle C
theorem find_angle_C (h : condition1 c B a b C) (h2 : condition2 c) : C = Real.pi / 3 := 
  sorry

-- Problem (2): Prove the maximum area of ΔABC
theorem find_max_area (h : condition1 c B a b C) (h2 : condition2 c) :
  ∃ (A B : ℝ), B = 2 * Real.pi / 3 - A ∧ 
    (∀ (A B : ℝ), Real.sin (2 * A - Real.pi / 6) = 1 → 
    1 / 2 * a * b * Real.sin C = 3 * Real.sqrt 3 ∧ 
    a = b ∧ b = c) := 
  sorry

end NUMINAMATH_GPT_find_angle_C_find_max_area_l2398_239874


namespace NUMINAMATH_GPT_least_possible_integer_for_friends_statements_l2398_239809

theorem least_possible_integer_for_friends_statements 
    (M : Nat)
    (statement_divisible_by : Nat → Prop)
    (h1 : ∀ n, 1 ≤ n ∧ n ≤ 30 → statement_divisible_by n = (M % n = 0))
    (h2 : ∃ m, 1 ≤ m ∧ m < 30 ∧ (statement_divisible_by m = false ∧ 
                                    statement_divisible_by (m + 1) = false)) :
    M = 12252240 :=
by
  sorry

end NUMINAMATH_GPT_least_possible_integer_for_friends_statements_l2398_239809


namespace NUMINAMATH_GPT_simplify_expression_l2398_239833

theorem simplify_expression (x : ℝ) : 2 * x + 1 - (x + 1) = x := 
by 
sorry

end NUMINAMATH_GPT_simplify_expression_l2398_239833


namespace NUMINAMATH_GPT_real_condition_proof_l2398_239871

noncomputable def real_condition_sufficient_but_not_necessary : Prop := 
∀ x : ℝ, (|x - 2| < 1) → ((x^2 + x - 2) > 0) ∧ (¬ ( ∀ y : ℝ, (y^2 + y - 2) > 0 → |y - 2| < 1))

theorem real_condition_proof : real_condition_sufficient_but_not_necessary :=
by
  sorry

end NUMINAMATH_GPT_real_condition_proof_l2398_239871


namespace NUMINAMATH_GPT_bren_age_indeterminate_l2398_239862

/-- The problem statement: The ratio of ages of Aman, Bren, and Charlie are in 
the ratio 5:8:7 respectively. A certain number of years ago, the sum of their ages was 76. 
We need to prove that without additional information, it is impossible to uniquely 
determine Bren's age 10 years from now. -/
theorem bren_age_indeterminate
  (x y : ℕ) 
  (h_ratio : true)
  (h_sum : 20 * x - 3 * y = 76) : 
  ∃ x y : ℕ, (20 * x - 3 * y = 76) ∧ ∀ bren_age_future : ℕ, ∃ x' y' : ℕ, (20 * x' - 3 * y' = 76) ∧ (8 * x' + 10) ≠ bren_age_future :=
sorry

end NUMINAMATH_GPT_bren_age_indeterminate_l2398_239862


namespace NUMINAMATH_GPT_fairfield_middle_school_geography_players_l2398_239832

/-- At Fairfield Middle School, there are 24 players on the football team.
All players are enrolled in at least one of the subjects: history or geography.
There are 10 players taking history and 6 players taking both subjects.
We need to prove that the number of players taking geography is 20. -/
theorem fairfield_middle_school_geography_players
  (total_players : ℕ)
  (history_players : ℕ)
  (both_subjects_players : ℕ)
  (h1 : total_players = 24)
  (h2 : history_players = 10)
  (h3 : both_subjects_players = 6) :
  total_players - (history_players - both_subjects_players) = 20 :=
by {
  sorry
}

end NUMINAMATH_GPT_fairfield_middle_school_geography_players_l2398_239832


namespace NUMINAMATH_GPT_angle_proof_l2398_239812

-- Variables and assumptions
variable {α : Type} [LinearOrderedField α]    -- using a general type for angles
variable {A B C D E : α}                       -- points of the triangle and extended segment

-- Given conditions
variable (angle_ACB angle_ABC : α)
variable (H1 : angle_ACB = 2 * angle_ABC)      -- angle condition
variable (CD BD AD DE : α)
variable (H2 : CD = 2 * BD)                    -- segment length condition
variable (H3 : AD = DE)                        -- extended segment condition

-- The proof goal in Lean format
theorem angle_proof (H1 : angle_ACB = 2 * angle_ABC) 
  (H2 : CD = 2 * BD) 
  (H3 : AD = DE) :
  angle_ECB + 180 = 2 * angle_EBC := 
sorry  -- proof to be filled in

end NUMINAMATH_GPT_angle_proof_l2398_239812


namespace NUMINAMATH_GPT_bert_kangaroos_equal_to_kameron_in_40_days_l2398_239885

theorem bert_kangaroos_equal_to_kameron_in_40_days
  (k_count : ℕ) (b_count : ℕ) (rate : ℕ) (days : ℕ)
  (h1 : k_count = 100)
  (h2 : b_count = 20)
  (h3 : rate = 2)
  (h4 : days = 40) :
  b_count + days * rate = k_count := 
by
  sorry

end NUMINAMATH_GPT_bert_kangaroos_equal_to_kameron_in_40_days_l2398_239885


namespace NUMINAMATH_GPT_equation_relating_price_and_tax_and_discount_l2398_239873

variable (c t d : ℚ)

theorem equation_relating_price_and_tax_and_discount
  (h1 : 1.30 * c * ((100 + t) / 100) * ((100 - d) / 100) = 351) :
    1.30 * c * (100 + t) * (100 - d) = 3510000 := by
  sorry

end NUMINAMATH_GPT_equation_relating_price_and_tax_and_discount_l2398_239873


namespace NUMINAMATH_GPT_find_a_find_m_l2398_239869

noncomputable def f (x a : ℝ) : ℝ := Real.exp 1 * x - a * Real.log x

theorem find_a {a : ℝ} (h : ∀ x, f x a = Real.exp 1 - a / x)
  (hx : f (1 / Real.exp 1) a = 0) :
  a = 1 :=
by
  sorry

theorem find_m (a : ℝ) (h_a : a = 1)
  (h_exists : ∃ (x₀ : ℝ), x₀ ∈ Set.Icc (1 / Real.exp 1) (Real.exp 1) 
    ∧ f x₀ a < x₀ + m) :
  1 + Real.log (Real.exp 1 - 1) < m :=
by
  sorry

end NUMINAMATH_GPT_find_a_find_m_l2398_239869


namespace NUMINAMATH_GPT_set_clock_correctly_l2398_239814

noncomputable def correct_clock_time
  (T_depart T_arrive T_depart_friend T_return : ℕ) 
  (T_visit := T_depart_friend - T_arrive) 
  (T_return_err := T_return - T_depart) 
  (T_total_travel := T_return_err - T_visit) 
  (T_travel_oneway := T_total_travel / 2) : ℕ :=
  T_depart + T_visit + T_travel_oneway

theorem set_clock_correctly 
  (T_depart T_arrive T_depart_friend T_return : ℕ)
  (h1 : T_depart ≤ T_return) -- The clock runs without accounting for the time away
  (h2 : T_arrive ≤ T_depart_friend) -- The friend's times are correct
  (h3 : T_return ≠ T_depart) -- The man was away for some non-zero duration
: 
  (correct_clock_time T_depart T_arrive T_depart_friend T_return) = 
  (T_depart + (T_depart_friend - T_arrive) + ((T_return - T_depart - (T_depart_friend - T_arrive)) / 2)) :=
sorry

end NUMINAMATH_GPT_set_clock_correctly_l2398_239814


namespace NUMINAMATH_GPT_shifted_function_is_correct_l2398_239855

-- Define the original function
def original_function (x : ℝ) : ℝ := -2 * x

-- Define the shifted function
def shifted_function (x : ℝ) : ℝ := original_function (x - 3)

-- State the theorem to be proven
theorem shifted_function_is_correct :
  ∀ x : ℝ, shifted_function x = -2 * x + 6 :=
by
  sorry

end NUMINAMATH_GPT_shifted_function_is_correct_l2398_239855


namespace NUMINAMATH_GPT_train_length_l2398_239852

theorem train_length (speed : ℝ) (time_seconds : ℝ) (time_hours : ℝ) (distance_km : ℝ) (distance_m : ℝ) 
  (h1 : speed = 60) 
  (h2 : time_seconds = 42) 
  (h3 : time_hours = time_seconds / 3600)
  (h4 : distance_km = speed * time_hours) 
  (h5 : distance_m = distance_km * 1000) :
  distance_m = 700 :=
by 
  sorry

end NUMINAMATH_GPT_train_length_l2398_239852


namespace NUMINAMATH_GPT_minimum_n_l2398_239810

-- Assume the sequence a_n is defined as part of an arithmetic sequence
def arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ := a + n * d

-- Define S_n as the sum of the first n terms in the sequence
def sum_arithmetic_sequence (a d n : ℕ) : ℕ := n * a + (n * (n - 1)) / 2 * d

-- Given conditions
def a1 := 2
def d := 1  -- Derived from the condition a1 + a4 = a5

-- Problem Statement
theorem minimum_n (n : ℕ) :
  (sum_arithmetic_sequence a1 d n > 32) ↔ n = 6 :=
sorry

end NUMINAMATH_GPT_minimum_n_l2398_239810


namespace NUMINAMATH_GPT_bob_wins_even_n_l2398_239851

def game_of_islands (n : ℕ) (even_n : n % 2 = 0) : Prop :=
  ∃ strategy : (ℕ → ℕ), -- strategy is a function representing each player's move
    ∀ A B : ℕ → ℕ, -- A and B represent the moves of Alice and Bob respectively
    (A 0 + B 1) = n → (A (A 0 + 1) ≠ B (A 0 + 1)) -- Bob can always mirror Alice’s move.

theorem bob_wins_even_n (n : ℕ) (h : n % 2 = 0) : game_of_islands n h :=
sorry

end NUMINAMATH_GPT_bob_wins_even_n_l2398_239851


namespace NUMINAMATH_GPT_rectangle_area_l2398_239866

noncomputable def circle_radius := 8
noncomputable def rect_ratio : ℕ × ℕ := (3, 1)
noncomputable def rect_area (width length : ℕ) : ℕ := width * length

theorem rectangle_area (width length : ℕ) 
  (h1 : 2 * circle_radius = width) 
  (h2 : rect_ratio.1 * width = length) : 
  rect_area width length = 768 := 
sorry

end NUMINAMATH_GPT_rectangle_area_l2398_239866


namespace NUMINAMATH_GPT_min_balloon_count_l2398_239828

theorem min_balloon_count 
(R B : ℕ) (burst_red burst_blue : ℕ) 
(h1 : R = 7 * B) 
(h2 : burst_red = burst_blue / 3) 
(h3 : burst_red ≥ 1) :
R + B = 24 :=
by 
    sorry

end NUMINAMATH_GPT_min_balloon_count_l2398_239828


namespace NUMINAMATH_GPT_incorrect_quotient_l2398_239803

theorem incorrect_quotient
    (correct_quotient : ℕ)
    (correct_divisor : ℕ)
    (incorrect_divisor : ℕ)
    (h1 : correct_quotient = 28)
    (h2 : correct_divisor = 21)
    (h3 : incorrect_divisor = 12) :
  correct_divisor * correct_quotient / incorrect_divisor = 49 :=
by
  sorry

end NUMINAMATH_GPT_incorrect_quotient_l2398_239803


namespace NUMINAMATH_GPT_total_money_shared_l2398_239807

-- Conditions
def ken_share : ℕ := 1750
def tony_share : ℕ := 2 * ken_share

-- Question and proof to be demonstrated
theorem total_money_shared : ken_share + tony_share = 5250 :=
by sorry

end NUMINAMATH_GPT_total_money_shared_l2398_239807


namespace NUMINAMATH_GPT_horizontal_asymptote_value_l2398_239840

theorem horizontal_asymptote_value :
  ∀ (x : ℝ),
  ((8 * x^4 + 6 * x^3 + 7 * x^2 + 2 * x + 4) / 
  (2 * x^4 + 5 * x^3 + 3 * x^2 + x + 6)) = (4 : ℝ) :=
by sorry

end NUMINAMATH_GPT_horizontal_asymptote_value_l2398_239840


namespace NUMINAMATH_GPT_distribute_neg3_l2398_239806

theorem distribute_neg3 (x y : ℝ) : -3 * (x - x * y) = -3 * x + 3 * x * y :=
by sorry

end NUMINAMATH_GPT_distribute_neg3_l2398_239806


namespace NUMINAMATH_GPT_sum_of_series_l2398_239805

theorem sum_of_series :
  (∑' n : ℕ, (3^n) / (3^(3^n) + 1)) = 1 / 2 :=
sorry

end NUMINAMATH_GPT_sum_of_series_l2398_239805


namespace NUMINAMATH_GPT_three_alpha_four_plus_eight_beta_three_eq_876_l2398_239817

variable (α β : ℝ)

-- Condition 1: α and β are roots of the equation x^2 - 3x - 4 = 0
def roots_of_quadratic : Prop := α^2 - 3 * α - 4 = 0 ∧ β^2 - 3 * β - 4 = 0

-- Question: 3α^4 + 8β^3 = ?
theorem three_alpha_four_plus_eight_beta_three_eq_876 
  (h : roots_of_quadratic α β) : (3 * α^4 + 8 * β^3 = 876) := sorry

end NUMINAMATH_GPT_three_alpha_four_plus_eight_beta_three_eq_876_l2398_239817


namespace NUMINAMATH_GPT_total_flowers_sold_l2398_239894

-- Definitions for conditions
def roses_per_bouquet : ℕ := 12
def daisies_per_bouquet : ℕ := 12  -- Assuming each daisy bouquet contains the same number of daisies as roses
def total_bouquets : ℕ := 20
def rose_bouquets_sold : ℕ := 10
def daisy_bouquets_sold : ℕ := 10

-- Statement of the equivalent Lean theorem
theorem total_flowers_sold :
  (rose_bouquets_sold * roses_per_bouquet) + (daisy_bouquets_sold * daisies_per_bouquet) = 240 :=
by
  sorry

end NUMINAMATH_GPT_total_flowers_sold_l2398_239894


namespace NUMINAMATH_GPT_parabola_equation_l2398_239839

theorem parabola_equation (a : ℝ) :
  (∀ x, (x + 1) * (x - 3) = 0 ↔ x = -1 ∨ x = 3) →
  (∀ y, y = a * (0 + 1) * (0 - 3) → y = 3) →
  a = -1 → 
  (∀ x, y = a * (x + 1) * (x - 3) → y = -x^2 + 2 * x + 3) :=
by
  intros h₁ h₂ ha
  sorry

end NUMINAMATH_GPT_parabola_equation_l2398_239839


namespace NUMINAMATH_GPT_factorial_multiple_of_3_l2398_239808

theorem factorial_multiple_of_3 (n : ℤ) (h : n ≥ 9) : 3 ∣ (n+1) * (n+3) :=
sorry

end NUMINAMATH_GPT_factorial_multiple_of_3_l2398_239808


namespace NUMINAMATH_GPT_matt_days_alone_l2398_239876

noncomputable def work_rate (days : ℝ) : ℝ := 1 / days

theorem matt_days_alone (M P : ℝ) (h1 : work_rate M + work_rate P = work_rate 20) 
  (h2 : 1 - 12 * (work_rate M + work_rate P) = 2 / 5) 
  (h3 : 10 * work_rate M = 2 / 5) : M = 25 :=
by
  sorry

end NUMINAMATH_GPT_matt_days_alone_l2398_239876


namespace NUMINAMATH_GPT_find_m_value_l2398_239860

theorem find_m_value (m : ℝ) (h : (m - 4)^2 + 1^2 + 2^2 = 30) : m = 9 ∨ m = -1 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_m_value_l2398_239860


namespace NUMINAMATH_GPT_combined_stickers_l2398_239843

theorem combined_stickers (k j a : ℕ) (h : 7 * j + 5 * a = 54) (hk : k = 42) (hk_ratio : k = 7 * 6) :
  j + a = 54 :=
by
  sorry

end NUMINAMATH_GPT_combined_stickers_l2398_239843


namespace NUMINAMATH_GPT_negation_of_universal_l2398_239834

theorem negation_of_universal :
  (¬ (∀ x : ℝ, x > 0 → x^2 + x ≥ 0)) ↔ (∃ x_0 : ℝ, x_0 > 0 ∧ x_0^2 + x_0 < 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_universal_l2398_239834


namespace NUMINAMATH_GPT_regular_polygon_sides_l2398_239842

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (x : ℕ), x > 2 → n = x)
  (h2 : ∀ (θ : ℕ), θ = 18 → 360 / n = θ) : n = 20 := by
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l2398_239842


namespace NUMINAMATH_GPT_exists_special_function_l2398_239841

theorem exists_special_function : ∃ (s : ℚ → ℤ), (∀ x y : ℚ, x ≠ y → (x * y = 1 ∨ x + y = 0 ∨ x + y = 1) → s x * s y = -1) ∧ (∀ x : ℚ, s x = 1 ∨ s x = -1) :=
by
  sorry

end NUMINAMATH_GPT_exists_special_function_l2398_239841


namespace NUMINAMATH_GPT_coloring_satisfies_conditions_l2398_239896

-- Definitions of point colors
inductive Color
| Red
| White
| Black

def color_point (x y : ℤ) : Color :=
  if (x + y) % 2 = 1 then Color.Red
  else if (x % 2 = 1 ∧ y % 2 = 0) then Color.White
  else Color.Black

-- Problem statement
theorem coloring_satisfies_conditions :
  (∀ y : ℤ, ∃ x1 x2 x3 : ℤ, 
    color_point x1 y = Color.Red ∧ 
    color_point x2 y = Color.White ∧
    color_point x3 y = Color.Black)
  ∧ 
  (∀ (x1 y1 x2 y2 x3 y3 : ℤ),
    color_point x1 y1 = Color.White →
    color_point x2 y2 = Color.Red →
    color_point x3 y3 = Color.Black →
    ∃ x4 y4, 
      color_point x4 y4 = Color.Red ∧ 
      x4 = x3 + (x1 - x2) ∧ 
      y4 = y3 + (y1 - y2)) :=
by
  sorry

end NUMINAMATH_GPT_coloring_satisfies_conditions_l2398_239896


namespace NUMINAMATH_GPT_sum_of_base_8_digits_888_l2398_239837

def base_8_representation (n : ℕ) : ℕ := 
  let d0 := n % 8
  let n  := n / 8
  let d1 := n % 8
  let n  := n / 8
  let d2 := n % 8
  let n  := n / 8
  let d3 := n % 8
  d3 * 1000 + d2 * 100 + d1 * 10 + d0

def sum_of_digits (n : ℕ) : ℕ :=
  n / 1000 + (n % 1000) / 100 + (n % 100) / 10 + (n % 10)

theorem sum_of_base_8_digits_888 : 
  sum_of_digits (base_8_representation 888) = 13 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_base_8_digits_888_l2398_239837


namespace NUMINAMATH_GPT_set_B_equals_1_4_l2398_239864

open Set

def U : Set ℕ := {1, 2, 3, 4}
def C_U_B : Set ℕ := {2, 3}

theorem set_B_equals_1_4 : 
  ∃ B : Set ℕ, B = {1, 4} ∧ U \ B = C_U_B := by
  sorry

end NUMINAMATH_GPT_set_B_equals_1_4_l2398_239864


namespace NUMINAMATH_GPT_books_in_special_collection_at_beginning_of_month_l2398_239818

theorem books_in_special_collection_at_beginning_of_month
  (loaned_out_real : Real)
  (loaned_out_books : Int)
  (returned_ratio : Real)
  (books_at_end : Int)
  (B : Int)
  (h1 : loaned_out_real = 49.99999999999999)
  (h2 : loaned_out_books = 50)
  (h3 : returned_ratio = 0.70)
  (h4 : books_at_end = 60)
  (h5 : loaned_out_books = Int.floor loaned_out_real)
  (h6 : ∀ (loaned_books : Int), loaned_books ≤ loaned_out_books → returned_ratio * loaned_books + (loaned_books - returned_ratio * loaned_books) = loaned_books)
  : B = 75 :=
by
  sorry

end NUMINAMATH_GPT_books_in_special_collection_at_beginning_of_month_l2398_239818


namespace NUMINAMATH_GPT_max_volume_of_acetic_acid_solution_l2398_239870

theorem max_volume_of_acetic_acid_solution :
  (∀ (V : ℝ), 0 ≤ V ∧ (V * 0.09) = (25 * 0.7 + (V - 25) * 0.05)) →
  V = 406.25 :=
by
  sorry

end NUMINAMATH_GPT_max_volume_of_acetic_acid_solution_l2398_239870


namespace NUMINAMATH_GPT_primes_diff_power_of_two_divisible_by_three_l2398_239823

theorem primes_diff_power_of_two_divisible_by_three
  (p q : ℕ) (m n : ℕ)
  (hp : Prime p) (hq : Prime q) (hp_gt : p > 3) (hq_gt : q > 3)
  (diff : q - p = 2^n ∨ p - q = 2^n) :
  3 ∣ (p^(2*m+1) + q^(2*m+1)) := by
  sorry

end NUMINAMATH_GPT_primes_diff_power_of_two_divisible_by_three_l2398_239823


namespace NUMINAMATH_GPT_valve_solution_l2398_239849

noncomputable def valve_problem : Prop :=
  ∀ (x y z : ℝ),
  (1 / (x + y + z) = 2) →
  (1 / (x + z) = 4) →
  (1 / (y + z) = 3) →
  (1 / (x + y) = 2.4)

theorem valve_solution : valve_problem :=
by
  -- proof omitted
  intros x y z h1 h2 h3
  sorry

end NUMINAMATH_GPT_valve_solution_l2398_239849


namespace NUMINAMATH_GPT_prove_total_bill_is_correct_l2398_239826

noncomputable def totalCostAfterDiscounts : ℝ :=
  let adultsMealsCost := 8 * 12
  let teenagersMealsCost := 4 * 10
  let childrenMealsCost := 3 * 7
  let adultsSodasCost := 8 * 3.5
  let teenagersSodasCost := 4 * 3.5
  let childrenSodasCost := 3 * 1.8
  let appetizersCost := 4 * 8
  let dessertsCost := 5 * 5

  let subtotal := adultsMealsCost + teenagersMealsCost + childrenMealsCost +
                  adultsSodasCost + teenagersSodasCost + childrenSodasCost +
                  appetizersCost + dessertsCost

  let discountAdultsMeals := 0.10 * adultsMealsCost
  let discountDesserts := 5
  let discountChildrenMealsAndSodas := 0.15 * (childrenMealsCost + childrenSodasCost)

  let adjustedSubtotal := subtotal - discountAdultsMeals - discountDesserts - discountChildrenMealsAndSodas

  let additionalDiscount := if subtotal > 200 then 0.05 * adjustedSubtotal else 0
  let total := adjustedSubtotal - additionalDiscount

  total

theorem prove_total_bill_is_correct : totalCostAfterDiscounts = 230.70 :=
by sorry

end NUMINAMATH_GPT_prove_total_bill_is_correct_l2398_239826


namespace NUMINAMATH_GPT_capacity_of_each_type_l2398_239889

def total_capacity_barrels : ℕ := 7000

def increased_by_first_type : ℕ := 8000

def decreased_by_second_type : ℕ := 3000

theorem capacity_of_each_type 
  (x y : ℕ) 
  (n k : ℕ)
  (h1 : x + y = total_capacity_barrels)
  (h2 : x * (n + k) / n = increased_by_first_type)
  (h3 : y * (n + k) / k = decreased_by_second_type) :
  x = 6400 ∧ y = 600 := sorry

end NUMINAMATH_GPT_capacity_of_each_type_l2398_239889


namespace NUMINAMATH_GPT_rope_length_before_folding_l2398_239845

theorem rope_length_before_folding (L : ℝ) (h : L / 4 = 10) : L = 40 :=
by
  sorry

end NUMINAMATH_GPT_rope_length_before_folding_l2398_239845


namespace NUMINAMATH_GPT_division_problem_l2398_239847

theorem division_problem (n : ℕ) (h : n / 4 = 12) : n / 3 = 16 := by
  sorry

end NUMINAMATH_GPT_division_problem_l2398_239847


namespace NUMINAMATH_GPT_tasty_residue_count_2016_l2398_239813

def tasty_residue (n : ℕ) (a : ℕ) : Prop :=
  1 < a ∧ a < n ∧ ∃ m : ℕ, m > 1 ∧ a ^ m ≡ a [MOD n]

theorem tasty_residue_count_2016 : 
  (∃ count : ℕ, count = 831 ∧ ∀ a : ℕ, 1 < a ∧ a < 2016 ↔ tasty_residue 2016 a) :=
sorry

end NUMINAMATH_GPT_tasty_residue_count_2016_l2398_239813


namespace NUMINAMATH_GPT_gcd_repeated_three_digit_integers_l2398_239800

theorem gcd_repeated_three_digit_integers : 
  ∀ m ∈ {n | 100 ≤ n ∧ n < 1000}, 
  gcd (1001 * m) (1001 * (m + 1)) = 1001 :=
by
  sorry

end NUMINAMATH_GPT_gcd_repeated_three_digit_integers_l2398_239800


namespace NUMINAMATH_GPT_correct_equation_l2398_239801

theorem correct_equation (x : ℕ) :
  (30 * x + 8 = 31 * x - 26) := by
  sorry

end NUMINAMATH_GPT_correct_equation_l2398_239801


namespace NUMINAMATH_GPT_problem_statement_l2398_239820

theorem problem_statement (x : ℚ) (h : 8 * x = 3) : 200 * (1 / x) = 1600 / 3 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l2398_239820


namespace NUMINAMATH_GPT_divide_by_10_result_l2398_239881

theorem divide_by_10_result (x : ℕ) (h : 5 * x = 100) : x / 10 = 2 := by
  sorry

end NUMINAMATH_GPT_divide_by_10_result_l2398_239881


namespace NUMINAMATH_GPT_irreducible_fraction_for_any_n_l2398_239890

theorem irreducible_fraction_for_any_n (n : ℤ) : Int.gcd (14 * n + 3) (21 * n + 4) = 1 := 
by {
  sorry
}

end NUMINAMATH_GPT_irreducible_fraction_for_any_n_l2398_239890


namespace NUMINAMATH_GPT_mod_equivalence_l2398_239891

theorem mod_equivalence (a b : ℤ) (d : ℕ) (hd : d ≠ 0) 
  (a' b' : ℕ) (ha' : a % d = a') (hb' : b % d = b') : (a ≡ b [ZMOD d]) ↔ a' = b' := 
sorry

end NUMINAMATH_GPT_mod_equivalence_l2398_239891


namespace NUMINAMATH_GPT_time_to_cross_pole_correct_l2398_239825

-- Definitions of the conditions
def trainSpeed_kmh : ℝ := 120 -- km/hr
def trainLength_m : ℝ := 300 -- meters

-- Assumed conversions
def kmToMeters : ℝ := 1000 -- meters in a km
def hoursToSeconds : ℝ := 3600 -- seconds in an hour

-- Conversion of speed from km/hr to m/s
noncomputable def trainSpeed_ms := (trainSpeed_kmh * kmToMeters) / hoursToSeconds

-- Time to cross the pole
noncomputable def timeToCrossPole := trainLength_m / trainSpeed_ms

-- The theorem stating the proof problem
theorem time_to_cross_pole_correct : timeToCrossPole = 9 := by
  sorry

end NUMINAMATH_GPT_time_to_cross_pole_correct_l2398_239825


namespace NUMINAMATH_GPT_range_of_m_l2398_239835

noncomputable def point := (ℝ × ℝ)
noncomputable def P : point := (-1, 1)
noncomputable def Q : point := (2, 2)
noncomputable def M : point := (0, -1)
noncomputable def line_eq (m : ℝ) := ∀ p : point, p.1 + m * p.2 + m = 0

theorem range_of_m (m : ℝ) (l : line_eq m) : -3 < m ∧ m < -2/3 := 
by
  sorry

end NUMINAMATH_GPT_range_of_m_l2398_239835


namespace NUMINAMATH_GPT_identity_eq_a_minus_b_l2398_239884

theorem identity_eq_a_minus_b (a b : ℚ) (x : ℚ) (h : ∀ x, x > 0 → 
  (a / (2^x - 2) + b / (2^x + 3) = (5 * 2^x + 4) / ((2^x - 2) * (2^x + 3)))) : 
  a - b = 3 / 5 := 
by 
  sorry

end NUMINAMATH_GPT_identity_eq_a_minus_b_l2398_239884


namespace NUMINAMATH_GPT_part_I_part_I_correct_interval_part_II_min_value_l2398_239844

noncomputable def f (x : ℝ) : ℝ := |2 * x + 1| - |x - 4|

theorem part_I : ∀ x : ℝ, (f x > 2) ↔ ( x < -7 ∨ (5 / 3 < x ∧ x < 4) ∨ x ≥ 4) := sorry

theorem part_I_correct_interval : ∀ x : ℝ, (f x > 2) → (x < -7 ∨ (5 / 3 < x ∧ x < 4) ∨ x ≥ 4) := sorry

theorem part_II_min_value : ∀ x : ℝ, ∃ y : ℝ, y = f x ∧ ∀ x : ℝ, f x ≥ y := 
sorry

end NUMINAMATH_GPT_part_I_part_I_correct_interval_part_II_min_value_l2398_239844


namespace NUMINAMATH_GPT_min_dot_product_trajectory_l2398_239802

-- Definitions of points and conditions
def point (x y : ℝ) : Prop := True

def trajectory (P : ℝ × ℝ) : Prop := 
  let x := P.1
  let y := P.2
  x * x - y * y = 2 ∧ x ≥ Real.sqrt 2

-- Definition of dot product over vectors from origin
def dot_product (A B : ℝ × ℝ) : ℝ :=
  A.1 * B.1 + A.2 * B.2

-- Stating the theorem for minimum value of dot product
theorem min_dot_product_trajectory (A B : ℝ × ℝ) (hA : trajectory A) (hB : trajectory B) : 
  dot_product A B ≥ 2 := 
sorry

end NUMINAMATH_GPT_min_dot_product_trajectory_l2398_239802


namespace NUMINAMATH_GPT_slope_angle_of_perpendicular_line_l2398_239888

theorem slope_angle_of_perpendicular_line (l : ℝ → ℝ) (h_perp : ∀ x y : ℝ, l x = y ↔ x - y - 1 = 0) : ∃ α : ℝ, α = 135 :=
by
  sorry

end NUMINAMATH_GPT_slope_angle_of_perpendicular_line_l2398_239888


namespace NUMINAMATH_GPT_average_books_per_month_l2398_239854

-- Definitions based on the conditions
def books_sold_january : ℕ := 15
def books_sold_february : ℕ := 16
def books_sold_march : ℕ := 17
def total_books_sold : ℕ := books_sold_january + books_sold_february + books_sold_march
def number_of_months : ℕ := 3

-- The theorem we need to prove
theorem average_books_per_month : total_books_sold / number_of_months = 16 :=
by
  sorry

end NUMINAMATH_GPT_average_books_per_month_l2398_239854
