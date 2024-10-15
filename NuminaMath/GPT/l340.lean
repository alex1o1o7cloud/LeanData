import Mathlib

namespace NUMINAMATH_GPT_grant_earnings_proof_l340_34087

noncomputable def total_earnings (X Y Z W : ℕ): ℕ :=
  let first_month := X
  let second_month := 3 * X + Y
  let third_month := 2 * second_month - Z
  let average := (first_month + second_month + third_month) / 3
  let fourth_month := average + W
  first_month + second_month + third_month + fourth_month

theorem grant_earnings_proof : total_earnings 350 30 20 50 = 5810 := by
  sorry

end NUMINAMATH_GPT_grant_earnings_proof_l340_34087


namespace NUMINAMATH_GPT_arithmetic_seq_and_general_formula_find_Tn_l340_34056

-- Given definitions
def S : ℕ → ℕ := sorry
def a : ℕ → ℕ := sorry

-- Conditions
axiom a1 : a 1 = 1
axiom a2 : ∀ n : ℕ, n > 0 → n * S n.succ = (n+1) * S n + n^2 + n

-- Problem 1: Prove and derive general formula for Sₙ
theorem arithmetic_seq_and_general_formula (n : ℕ) (h : n > 0) :
  ∃ S : ℕ → ℕ, (∀ n : ℕ, n > 0 → (S (n+1)) / (n+1) - (S n) / n = 1) ∧ (S n = n^2) := sorry

-- Problem 2: Given bₙ and Tₙ, find Tₙ
def b (n : ℕ) : ℕ := 1 / (a n * a (n+1))
def T : ℕ → ℕ := sorry

axiom b1 : ∀ n : ℕ, n > 0 → b 1 = 1
axiom b2 : ∀ n : ℕ, n > 0 → T n = 1 / (2 * n + 1)

theorem find_Tn (n : ℕ) (h : n > 0) : T n = n / (2 * n + 1) := sorry

end NUMINAMATH_GPT_arithmetic_seq_and_general_formula_find_Tn_l340_34056


namespace NUMINAMATH_GPT_coefficient_x9_l340_34009

theorem coefficient_x9 (p : Polynomial ℚ) : 
  p = (1 + 3 * Polynomial.X - Polynomial.X^2)^5 →
  Polynomial.coeff p 9 = 15 := 
by
  intro h
  rw [h]
  -- additional lean tactics to prove the statement would go here
  sorry

end NUMINAMATH_GPT_coefficient_x9_l340_34009


namespace NUMINAMATH_GPT_probability_of_selection_is_equal_l340_34003

-- Define the conditions of the problem
def total_students := 2004
def eliminated_students := 4
def remaining_students := total_students - eliminated_students -- 2000
def selected_students := 50
def k := remaining_students / selected_students -- 40

-- Define the probability calculation
def probability_selected := selected_students / remaining_students

-- The theorem stating that every student has a 1/40 probability of being selected
theorem probability_of_selection_is_equal :
  probability_selected = 1 / 40 :=
by
  -- insert proof logic here
  sorry

end NUMINAMATH_GPT_probability_of_selection_is_equal_l340_34003


namespace NUMINAMATH_GPT_puppy_cost_l340_34098

variable (P : ℕ)  -- Cost of one puppy

theorem puppy_cost (P : ℕ) (kittens : ℕ) (cost_kitten : ℕ) (total_value : ℕ) :
  kittens = 4 → cost_kitten = 15 → total_value = 100 → 
  2 * P + kittens * cost_kitten = total_value → P = 20 :=
by sorry

end NUMINAMATH_GPT_puppy_cost_l340_34098


namespace NUMINAMATH_GPT_ratio_of_boys_to_girls_l340_34080

def boys_girls_ratio (b g : ℕ) : ℚ := b / g

theorem ratio_of_boys_to_girls (b g : ℕ) (h1 : b = g + 6) (h2 : g + b = 40) :
  boys_girls_ratio b g = 23 / 17 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_boys_to_girls_l340_34080


namespace NUMINAMATH_GPT_incorrect_statement_among_props_l340_34024

theorem incorrect_statement_among_props 
    (A: Prop := True)  -- Axioms in mathematics are accepted truths that do not require proof.
    (B: Prop := True)  -- A mathematical proof can proceed in different valid sequences depending on the approach and insights.
    (C: Prop := True)  -- All concepts utilized in a proof must be clearly defined before their use in arguments.
    (D: Prop := False) -- Logical deductions based on false premises can lead to valid conclusions.
    (E: Prop := True): -- Proof by contradiction only needs one assumption to be negated and shown to lead to a contradiction to be valid.
  ¬D := 
by sorry

end NUMINAMATH_GPT_incorrect_statement_among_props_l340_34024


namespace NUMINAMATH_GPT_rhombus_perimeter_l340_34076

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 72) (h2 : d2 = 30) 
  (h3 : ∀ {x y : ℝ}, (x = d1 / 2 ∧ y = d2 / 2) → (x^2 + y^2 = (d1 / 2)^2 + (d2 / 2)^2)) : 
  4 * (Real.sqrt ((d1/2)^2 + (d2/2)^2)) = 156 :=
by 
  rw [h1, h2]
  simp
  sorry

end NUMINAMATH_GPT_rhombus_perimeter_l340_34076


namespace NUMINAMATH_GPT_initial_passengers_is_350_l340_34021

variable (N : ℕ)

def initial_passengers (N : ℕ) : Prop :=
  let after_first_train := 9 * N / 10
  let after_second_train := 27 * N / 35
  let after_third_train := 108 * N / 175
  after_third_train = 216

theorem initial_passengers_is_350 : initial_passengers 350 := 
  sorry

end NUMINAMATH_GPT_initial_passengers_is_350_l340_34021


namespace NUMINAMATH_GPT_a_beats_b_by_10_seconds_l340_34033

theorem a_beats_b_by_10_seconds :
  ∀ (T_A T_B D_A D_B : ℕ),
    T_A = 615 →
    D_A = 1000 →
    D_A - D_B = 16 →
    T_B = (D_A * T_A) / D_B →
    T_B - T_A = 10 :=
by
  -- Placeholder to ensure the theorem compiles
  intros T_A T_B D_A D_B h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_a_beats_b_by_10_seconds_l340_34033


namespace NUMINAMATH_GPT_words_with_mistakes_percentage_l340_34030

theorem words_with_mistakes_percentage (n x : ℕ) 
  (h1 : (x - 1 : ℝ) / n = 0.24)
  (h2 : (x - 1 : ℝ) / (n - 1) = 0.25) :
  (x : ℝ) / n * 100 = 28 := 
by 
  sorry

end NUMINAMATH_GPT_words_with_mistakes_percentage_l340_34030


namespace NUMINAMATH_GPT_find_person_10_number_l340_34031

theorem find_person_10_number (n : ℕ) (a : ℕ → ℕ)
  (h1 : n = 15)
  (h2 : 2 * a 10 = a 9 + a 11)
  (h3 : 2 * a 3 = a 2 + a 4)
  (h4 : a 10 = 8)
  (h5 : a 3 = 7) :
  a 10 = 8 := 
by sorry

end NUMINAMATH_GPT_find_person_10_number_l340_34031


namespace NUMINAMATH_GPT_find_m_l340_34002

theorem find_m (m x : ℝ) 
  (h1 : (m - 1) * x^2 + 5 * x + m^2 - 3 * m + 2 = 0) 
  (h2 : m^2 - 3 * m + 2 = 0)
  (h3 : m ≠ 1) : 
  m = 2 := 
sorry

end NUMINAMATH_GPT_find_m_l340_34002


namespace NUMINAMATH_GPT_never_return_to_start_l340_34064

variable {City : Type} [MetricSpace City]

-- Conditions
variable (C : ℕ → City)  -- C is the sequence of cities
variable (dist : City → City → ℝ)  -- distance function
variable (furthest : City → City)  -- function that maps each city to the furthest city from it
variable (start : City)  -- initial city

-- Assuming C satisfies the properties in the problem statement
axiom initial_city : C 1 = start
axiom furthest_city_step : ∀ n, C (n + 1) = furthest (C n)
axiom no_ambiguity : ∀ c1 c2, (dist c1 (furthest c1) > dist c1 c2 ↔ c2 ≠ furthest c1)

-- Define the problem to prove that if C₁ ≠ C₃, then ∀ n ≥ 4, Cₙ ≠ C₁
theorem never_return_to_start (h : C 1 ≠ C 3) : ∀ n ≥ 4, C n ≠ start := sorry

end NUMINAMATH_GPT_never_return_to_start_l340_34064


namespace NUMINAMATH_GPT_freshman_class_total_students_l340_34001

theorem freshman_class_total_students (N : ℕ) 
    (h1 : 90 ≤ N) 
    (h2 : 100 ≤ N)
    (h3 : 20 ≤ N) 
    (h4: (90 : ℝ) / N * (20 : ℝ) / 100 = (20 : ℝ) / N):
    N = 450 :=
  sorry

end NUMINAMATH_GPT_freshman_class_total_students_l340_34001


namespace NUMINAMATH_GPT_evaluate_g_at_2_l340_34074

def g (x : ℝ) : ℝ := x^3 - 2 * x + 1

theorem evaluate_g_at_2 : g 2 = 5 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_g_at_2_l340_34074


namespace NUMINAMATH_GPT_inequation_proof_l340_34099

theorem inequation_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a^2 + b^2 + c^2 = 1) :
  (a / (1 - a^2)) + (b / (1 - b^2)) + (c / (1 - c^2)) ≥ (3 * Real.sqrt 3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_inequation_proof_l340_34099


namespace NUMINAMATH_GPT_find_sides_of_triangle_ABC_find_angle_A_l340_34079

variable (a b c A B C : ℝ)

-- Part (Ⅰ)
theorem find_sides_of_triangle_ABC
  (hC : C = Real.pi / 3)
  (hc : c = 2)
  (hArea : 1/2 * a * b * Real.sin (Real.pi / 3) = Real.sqrt 3) :
  a = 2 ∧ b = 2 := sorry

-- Part (Ⅱ)
theorem find_angle_A
  (hC : C = Real.pi / 3)
  (hc : c = 2)
  (hTrig : Real.sin C + Real.sin (B - A) = 2 * Real.sin (2 * A)) :
  A = Real.pi / 2 ∨ A = Real.pi / 6 := sorry

end NUMINAMATH_GPT_find_sides_of_triangle_ABC_find_angle_A_l340_34079


namespace NUMINAMATH_GPT_base6_number_divisibility_l340_34011

/-- 
Given that 45x2 in base 6 converted to its decimal equivalent is 6x + 1046,
and it is divisible by 19. Prove that x = 5 given that x is a base-6 digit.
-/
theorem base6_number_divisibility (x : ℕ) (h1 : 0 ≤ x ∧ x ≤ 5) (h2 : (6 * x + 1046) % 19 = 0) : x = 5 :=
sorry

end NUMINAMATH_GPT_base6_number_divisibility_l340_34011


namespace NUMINAMATH_GPT_polynomial_bound_l340_34023

noncomputable def P (a b c d : ℝ) (x : ℝ) : ℝ :=
  a * x^3 + b * x^2 + c * x + d

theorem polynomial_bound (a b c d : ℝ) (h : ∀ x : ℝ, |x| < 1 → |P a b c d x| ≤ 1) :
  |a| + |b| + |c| + |d| ≤ 7 :=
sorry

end NUMINAMATH_GPT_polynomial_bound_l340_34023


namespace NUMINAMATH_GPT_area_of_table_l340_34034

-- Definitions of the given conditions
def free_side_conditions (L W : ℝ) : Prop :=
  (L = 2 * W) ∧ (2 * W + L = 32)

-- Statement to prove the area of the rectangular table
theorem area_of_table {L W : ℝ} (h : free_side_conditions L W) : L * W = 128 := by
  sorry

end NUMINAMATH_GPT_area_of_table_l340_34034


namespace NUMINAMATH_GPT_pass_rate_l340_34006

theorem pass_rate (total_students : ℕ) (students_not_passed : ℕ) (pass_rate : ℚ) :
  total_students = 500 → 
  students_not_passed = 40 → 
  pass_rate = (total_students - students_not_passed) / total_students * 100 →
  pass_rate = 92 :=
by 
  intros ht hs hpr 
  sorry

end NUMINAMATH_GPT_pass_rate_l340_34006


namespace NUMINAMATH_GPT_smallest_nineteen_multiple_l340_34018

theorem smallest_nineteen_multiple (n : ℕ) 
  (h₁ : 19 * n ≡ 5678 [MOD 11]) : n = 8 :=
by sorry

end NUMINAMATH_GPT_smallest_nineteen_multiple_l340_34018


namespace NUMINAMATH_GPT_area_of_square_l340_34097

theorem area_of_square (r : ℝ) (b : ℝ) (ℓ : ℝ) (area_rect : ℝ) 
    (h₁ : ℓ = 2 / 3 * r) 
    (h₂ : r = b) 
    (h₃ : b = 13) 
    (h₄ : area_rect = 598) 
    (h₅ : area_rect = ℓ * b) : 
    r^2 = 4761 := 
sorry

end NUMINAMATH_GPT_area_of_square_l340_34097


namespace NUMINAMATH_GPT_trig_identity_example_l340_34066

open Real

noncomputable def tan_alpha_eq_two_tan_pi_fifths (α : ℝ) :=
  tan α = 2 * tan (π / 5)

theorem trig_identity_example (α : ℝ) (h : tan_alpha_eq_two_tan_pi_fifths α) :
  (cos (α - 3 * π / 10) / sin (α - π / 5)) = 3 :=
sorry

end NUMINAMATH_GPT_trig_identity_example_l340_34066


namespace NUMINAMATH_GPT_total_population_is_700_l340_34039

-- Definitions for the problem conditions
def L : ℕ := 200
def P : ℕ := L / 2
def E : ℕ := (L + P) / 2
def Z : ℕ := E + P

-- Proof statement (with sorry)
theorem total_population_is_700 : L + P + E + Z = 700 :=
by
  sorry

end NUMINAMATH_GPT_total_population_is_700_l340_34039


namespace NUMINAMATH_GPT_orchestra_musicians_l340_34069

theorem orchestra_musicians : ∃ (m n : ℕ), (m = n^2 + 11) ∧ (m = n * (n + 5)) ∧ m = 36 :=
by {
  sorry
}

end NUMINAMATH_GPT_orchestra_musicians_l340_34069


namespace NUMINAMATH_GPT_triangle_angle_sum_l340_34094

open scoped Real

theorem triangle_angle_sum (A B C : ℝ) 
  (hA : A = 25) (hB : B = 55) : C = 100 :=
by
  have h1 : A + B + C = 180 := sorry
  rw [hA, hB] at h1
  linarith

end NUMINAMATH_GPT_triangle_angle_sum_l340_34094


namespace NUMINAMATH_GPT_three_digit_numbers_with_properties_l340_34073

noncomputable def valid_numbers_with_properties : List Nat :=
  [179, 239, 299, 359, 419, 479, 539, 599, 659, 719, 779, 839, 899, 959]

theorem three_digit_numbers_with_properties (N : ℕ) :
  N >= 100 ∧ N < 1000 ∧ 
  N ≡ 1 [MOD 2] ∧
  N ≡ 2 [MOD 3] ∧
  N ≡ 3 [MOD 4] ∧
  N ≡ 4 [MOD 5] ∧
  N ≡ 5 [MOD 6] ↔ N ∈ valid_numbers_with_properties :=
by
  sorry

end NUMINAMATH_GPT_three_digit_numbers_with_properties_l340_34073


namespace NUMINAMATH_GPT_least_isosceles_triangles_cover_rectangle_l340_34010

-- Define the dimensions of the rectangle
def rectangle_height : ℕ := 10
def rectangle_width : ℕ := 100

-- Define the least number of isosceles right triangles needed to cover the rectangle
def least_number_of_triangles (h w : ℕ) : ℕ :=
  if h = rectangle_height ∧ w = rectangle_width then 11 else 0

-- The theorem statement
theorem least_isosceles_triangles_cover_rectangle :
  least_number_of_triangles rectangle_height rectangle_width = 11 :=
by
  -- skip the proof
  sorry

end NUMINAMATH_GPT_least_isosceles_triangles_cover_rectangle_l340_34010


namespace NUMINAMATH_GPT_smallest_sum_of_20_consecutive_integers_twice_perfect_square_l340_34060

theorem smallest_sum_of_20_consecutive_integers_twice_perfect_square :
  ∃ n : ℕ, ∃ k : ℕ, (∀ m : ℕ, m ≥ n → 0 < m) ∧ 10 * (2 * n + 19) = 2 * k^2 ∧ 10 * (2 * n + 19) = 450 :=
by
  sorry

end NUMINAMATH_GPT_smallest_sum_of_20_consecutive_integers_twice_perfect_square_l340_34060


namespace NUMINAMATH_GPT_tournament_matches_l340_34016

theorem tournament_matches (n : ℕ) (total_matches : ℕ) (matches_three_withdrew : ℕ) (matches_after_withdraw : ℕ) :
  ∀ (x : ℕ), total_matches = (n * (n - 1) / 2) → matches_three_withdrew = 6 - x → matches_after_withdraw = total_matches - (3 * 2 - x) → 
  matches_after_withdraw = 50 → x = 1 :=
by
  intros
  sorry

end NUMINAMATH_GPT_tournament_matches_l340_34016


namespace NUMINAMATH_GPT_trig_identity_cosine_powers_l340_34084

theorem trig_identity_cosine_powers :
  12 * (Real.cos (Real.pi / 8)) ^ 4 + 
  (Real.cos (3 * Real.pi / 8)) ^ 4 + 
  (Real.cos (5 * Real.pi / 8)) ^ 4 + 
  (Real.cos (7 * Real.pi / 8)) ^ 4 = 
  3 / 2 := 
  sorry

end NUMINAMATH_GPT_trig_identity_cosine_powers_l340_34084


namespace NUMINAMATH_GPT_hundredth_number_is_100_l340_34093

/-- Define the sequence of numbers said by Jo, Blair, and Parker following the conditions described. --/
def next_number (turn : ℕ) : ℕ :=
  -- Each turn increments by one number starting from 1
  turn

-- Prove that the 100th number in the sequence is 100
theorem hundredth_number_is_100 :
  next_number 100 = 100 := 
by sorry

end NUMINAMATH_GPT_hundredth_number_is_100_l340_34093


namespace NUMINAMATH_GPT_tan_difference_identity_l340_34045

theorem tan_difference_identity (a b : ℝ) (h1 : Real.tan a = 2) (h2 : Real.tan b = 3 / 4) :
  Real.tan (a - b) = 1 / 2 :=
sorry

end NUMINAMATH_GPT_tan_difference_identity_l340_34045


namespace NUMINAMATH_GPT_lisa_flew_distance_l340_34032

-- Define the given conditions
def speed := 32  -- speed in miles per hour
def time := 8    -- time in hours

-- Define the derived distance
def distance := speed * time  -- using the formula Distance = Speed × Time

-- Prove that the calculated distance is 256 miles
theorem lisa_flew_distance : distance = 256 :=
by
  sorry

end NUMINAMATH_GPT_lisa_flew_distance_l340_34032


namespace NUMINAMATH_GPT_comparison_of_A_and_B_l340_34070

noncomputable def A (m : ℝ) : ℝ := Real.sqrt (m + 1) - Real.sqrt m
noncomputable def B (m : ℝ) : ℝ := Real.sqrt m - Real.sqrt (m - 1)

theorem comparison_of_A_and_B (m : ℝ) (h : m > 1) : A m < B m :=
by
  sorry

end NUMINAMATH_GPT_comparison_of_A_and_B_l340_34070


namespace NUMINAMATH_GPT_largest_possible_sum_l340_34014

def max_sum_pair_mult_48 : Prop :=
  ∃ (heartsuit clubsuit : ℕ), (heartsuit * clubsuit = 48) ∧ (heartsuit + clubsuit = 49) ∧ 
  (∀ (h c : ℕ), (h * c = 48) → (h + c ≤ 49))

theorem largest_possible_sum : max_sum_pair_mult_48 :=
  sorry

end NUMINAMATH_GPT_largest_possible_sum_l340_34014


namespace NUMINAMATH_GPT_area_of_triangle_is_sqrt3_l340_34037

theorem area_of_triangle_is_sqrt3
  (a b c : ℝ)
  (B : ℝ)
  (h_geom_prog : b^2 = a * c)
  (h_b : b = 2)
  (h_B : B = Real.pi / 3) :
  (1/2) * a * c * Real.sin B = Real.sqrt 3 := 
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_is_sqrt3_l340_34037


namespace NUMINAMATH_GPT_quadratic_has_one_solution_implies_m_l340_34005

theorem quadratic_has_one_solution_implies_m (m : ℚ) :
  (∀ x : ℚ, 3 * x^2 - 7 * x + m = 0 → (b^2 - 4 * a * m = 0)) ↔ m = 49 / 12 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_has_one_solution_implies_m_l340_34005


namespace NUMINAMATH_GPT_log_base_change_l340_34053

-- Define the conditions: 8192 = 2 ^ 13 and change of base formula
def x : ℕ := 8192
def a : ℕ := 2
def n : ℕ := 13
def b : ℕ := 5

theorem log_base_change (log : ℕ → ℕ → ℝ) 
  (h1 : x = a ^ n) 
  (h2 : ∀ (x b c: ℕ), c ≠ 1 → log x b = (log x c) / (log b c) ): 
  log x b = 13 / (log 5 2) :=
by
  sorry

end NUMINAMATH_GPT_log_base_change_l340_34053


namespace NUMINAMATH_GPT_seq_2011_l340_34043

-- Definition of the sequence
def seq (a : ℕ → ℤ) := (a 1 = a 201) ∧ a 201 = 2 ∧ ∀ n : ℕ, a n + a (n + 1) = 0

-- The main theorem to prove that a_2011 = 2
theorem seq_2011 : ∀ a : ℕ → ℤ, seq a → a 2011 = 2 :=
by
  intros a h
  let seq := h
  sorry

end NUMINAMATH_GPT_seq_2011_l340_34043


namespace NUMINAMATH_GPT_derivative_and_value_l340_34050

-- Given conditions
def eqn (x y : ℝ) : Prop := 10 * x^3 + 4 * x^2 * y + y^2 = 0

-- The derivative y'
def y_prime (x y y' : ℝ) : Prop := y' = (-15 * x^2 - 4 * x * y) / (2 * x^2 + y)

-- Specific values derivatives
def y_prime_at_x_neg2_y_4 (y' : ℝ) : Prop := y' = -7 / 3

-- The main theorem
theorem derivative_and_value (x y y' : ℝ) 
  (h1 : eqn x y) (x_neg2 : x = -2) (y_4 : y = 4) : 
  y_prime x y y' ∧ y_prime_at_x_neg2_y_4 y' :=
sorry

end NUMINAMATH_GPT_derivative_and_value_l340_34050


namespace NUMINAMATH_GPT_negative_product_implies_negatives_l340_34042

theorem negative_product_implies_negatives (a b c : ℚ) (h : a * b * c < 0) :
  (∃ n : ℕ, n = 1 ∨ n = 3 ∧ (n = 1 ↔ (a < 0 ∧ b > 0 ∧ c > 0 ∨ a > 0 ∧ b < 0 ∧ c > 0 ∨ a > 0 ∧ b > 0 ∧ c < 0)) ∨ 
                                n = 3 ∧ (n = 3 ↔ (a < 0 ∧ b < 0 ∧ c < 0 ∨ a < 0 ∧ b < 0 ∧ c > 0 ∨ a < 0 ∧ b > 0 ∧ c < 0 ∨ a > 0 ∧ b < 0 ∧ c < 0))) :=
  sorry

end NUMINAMATH_GPT_negative_product_implies_negatives_l340_34042


namespace NUMINAMATH_GPT_union_of_sets_l340_34092

noncomputable def A : Set ℕ := {1, 2, 4}
noncomputable def B : Set ℕ := {2, 4, 6}

theorem union_of_sets : A ∪ B = {1, 2, 4, 6} := 
by 
sorry

end NUMINAMATH_GPT_union_of_sets_l340_34092


namespace NUMINAMATH_GPT_men_in_second_group_l340_34044

theorem men_in_second_group (M : ℕ) (h1 : 16 * 30 = 480) (h2 : M * 24 = 480) : M = 20 :=
by
  sorry

end NUMINAMATH_GPT_men_in_second_group_l340_34044


namespace NUMINAMATH_GPT_prove_y_value_l340_34007

theorem prove_y_value (x y : ℤ) (h1 : x - y = 8) (h2 : x + y = 10) : y = 1 :=
by
  sorry

end NUMINAMATH_GPT_prove_y_value_l340_34007


namespace NUMINAMATH_GPT_smallest_coprime_gt_one_l340_34083

theorem smallest_coprime_gt_one (x : ℕ) (h1 : 1 < x) (h2 : Nat.gcd x 180 = 1) : x = 7 := sorry

end NUMINAMATH_GPT_smallest_coprime_gt_one_l340_34083


namespace NUMINAMATH_GPT_nancy_pictures_l340_34025

theorem nancy_pictures (z m b d : ℕ) (hz : z = 120) (hm : m = 75) (hb : b = 45) (hd : d = 93) :
  (z + m + b) - d = 147 :=
by {
  -- Theorem definition capturing the problem statement
  sorry
}

end NUMINAMATH_GPT_nancy_pictures_l340_34025


namespace NUMINAMATH_GPT_crayon_colors_correct_l340_34020

-- The Lean code will define the conditions and the proof statement as follows:
noncomputable def crayon_problem := 
  let crayons_per_box := (160 / (5 * 4)) -- Total crayons / Total boxes
  let colors := (crayons_per_box / 2) -- Crayons per box / Crayons per color
  colors = 4

-- This is the theorem that needs to be proven:
theorem crayon_colors_correct : crayon_problem := by
  sorry

end NUMINAMATH_GPT_crayon_colors_correct_l340_34020


namespace NUMINAMATH_GPT_find_a_l340_34058

-- Define given parameters and conditions
def parabola_eq (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * a * x + 3

def shifted_parabola_eq (a : ℝ) (x : ℝ) : ℝ := parabola_eq a x - 3 * |a|

-- Define axis of symmetry function
def axis_of_symmetry (a : ℝ) : ℝ := 1

-- Conditions: a ≠ 0
variable (a : ℝ)
variable (h : a ≠ 0)

-- Define value for discriminant check
def discriminant (a : ℝ) (c : ℝ) : ℝ := (-2 * a)^2 - 4 * a * c

-- Problem statement
theorem find_a (ha : a ≠ 0) : 
  (axis_of_symmetry a = 1) ∧ (discriminant a (3 - 3 * |a|) = 0 → (a = 3 / 4 ∨ a = -3 / 2)) := 
by
  sorry -- proof to be filled in

end NUMINAMATH_GPT_find_a_l340_34058


namespace NUMINAMATH_GPT_two_pow_ge_two_mul_l340_34067

theorem two_pow_ge_two_mul (n : ℕ) : 2^n ≥ 2 * n :=
sorry

end NUMINAMATH_GPT_two_pow_ge_two_mul_l340_34067


namespace NUMINAMATH_GPT_final_segment_distance_l340_34040

theorem final_segment_distance :
  let north_distance := 2
  let east_distance := 1
  let south_distance := 1
  let net_north := north_distance - south_distance
  let net_east := east_distance
  let final_distance := Real.sqrt (net_north ^ 2 + net_east ^ 2)
  final_distance = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_final_segment_distance_l340_34040


namespace NUMINAMATH_GPT_min_sum_is_11_over_28_l340_34091

-- Definition of the problem
def digits : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

-- Defining the minimum sum problem
def min_sum (A B C D : ℕ) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  A ∈ digits ∧ B ∈ digits ∧ C ∈ digits ∧ D ∈ digits →
  ((A : ℚ) / B + (C : ℚ) / D) = (11 : ℚ) / 28

-- The theorem statement
theorem min_sum_is_11_over_28 :
  ∃ A B C D : ℕ, min_sum A B C D :=
sorry

end NUMINAMATH_GPT_min_sum_is_11_over_28_l340_34091


namespace NUMINAMATH_GPT_arc_length_of_circle_l340_34068

theorem arc_length_of_circle (r θ : ℝ) (h_r : r = 2) (h_θ : θ = 120) : 
  (θ / 180 * r * Real.pi) = (4 / 3) * Real.pi := by
  sorry

end NUMINAMATH_GPT_arc_length_of_circle_l340_34068


namespace NUMINAMATH_GPT_number_of_petri_dishes_l340_34057

def germs_in_lab : ℕ := 3700
def germs_per_dish : ℕ := 25
def num_petri_dishes : ℕ := germs_in_lab / germs_per_dish

theorem number_of_petri_dishes : num_petri_dishes = 148 :=
by
  sorry

end NUMINAMATH_GPT_number_of_petri_dishes_l340_34057


namespace NUMINAMATH_GPT_sum_powers_l340_34026

theorem sum_powers :
  ∃ (α β γ : ℂ), α + β + γ = 2 ∧ α^2 + β^2 + γ^2 = 5 ∧ α^3 + β^3 + γ^3 = 8 ∧ α^5 + β^5 + γ^5 = 46.5 :=
by
  sorry

end NUMINAMATH_GPT_sum_powers_l340_34026


namespace NUMINAMATH_GPT_person_dining_minutes_l340_34086

theorem person_dining_minutes
  (initial_angle : ℕ)
  (final_angle : ℕ)
  (time_spent : ℕ)
  (minute_angle_per_minute : ℕ)
  (hour_angle_per_minute : ℕ)
  (h1 : initial_angle = 110)
  (h2 : final_angle = 110)
  (h3 : minute_angle_per_minute = 6)
  (h4 : hour_angle_per_minute = minute_angle_per_minute / 12)
  (h5 : time_spent = (final_angle - initial_angle) / (minute_angle_per_minute / (minute_angle_per_minute / 12) - hour_angle_per_minute)) :
  time_spent = 40 := sorry

end NUMINAMATH_GPT_person_dining_minutes_l340_34086


namespace NUMINAMATH_GPT_hardcover_books_count_l340_34012

theorem hardcover_books_count
  (h p : ℕ)
  (h_plus_p_eq_10 : h + p = 10)
  (total_cost_eq_250 : 30 * h + 20 * p = 250) :
  h = 5 :=
by
  sorry

end NUMINAMATH_GPT_hardcover_books_count_l340_34012


namespace NUMINAMATH_GPT_legally_drive_after_hours_l340_34061

theorem legally_drive_after_hours (n : ℕ) :
  (∀ t ≥ n, 0.8 * (0.5 : ℝ) ^ t ≤ 0.2) ↔ n = 2 :=
by
  sorry

end NUMINAMATH_GPT_legally_drive_after_hours_l340_34061


namespace NUMINAMATH_GPT_henrys_distance_from_start_l340_34041

noncomputable def meters_to_feet (x : ℝ) : ℝ := x * 3.281
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem henrys_distance_from_start :
  let west_walk_feet := meters_to_feet 15
  let north_walk_feet := 60
  let east_walk_feet := 156
  let south_walk_meter_backwards := 30
  let south_walk_feet_backwards := 12
  let total_south_feet := meters_to_feet south_walk_meter_backwards + south_walk_feet_backwards
  let net_south_feet := total_south_feet - north_walk_feet
  let net_east_feet := east_walk_feet - west_walk_feet
  distance 0 0 net_east_feet (-net_south_feet) = 118 := 
by
  sorry

end NUMINAMATH_GPT_henrys_distance_from_start_l340_34041


namespace NUMINAMATH_GPT_range_of_m_l340_34089

def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*x + m ≠ 0
def q (m : ℝ) : Prop := ∃ y : ℝ, ∀ x : ℝ, (x^2)/(m-1) + y^2 = 1
def not_p (m : ℝ) : Prop := ¬ (p m)
def p_and_q (m : ℝ) : Prop := (p m) ∧ (q m)

theorem range_of_m (m : ℝ) : (¬ (not_p m) ∧ ¬ (p_and_q m)) → 1 < m ∧ m ≤ 2 :=
sorry

end NUMINAMATH_GPT_range_of_m_l340_34089


namespace NUMINAMATH_GPT_composite_function_l340_34059

def f (x : ℝ) : ℝ := 2 * x - 1
def g (x : ℝ) : ℝ := x + 1

theorem composite_function : ∀ (x : ℝ), f (g x) = 2 * x + 1 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_composite_function_l340_34059


namespace NUMINAMATH_GPT_sisterPassesMeInOppositeDirection_l340_34075

noncomputable def numberOfPasses (laps_sister : ℕ) : ℕ :=
if laps_sister > 1 then 2 * laps_sister else 0

theorem sisterPassesMeInOppositeDirection
  (my_laps : ℕ) (laps_sister : ℕ) (passes_in_same_direction : ℕ) :
  my_laps = 1 ∧ passes_in_same_direction = 2 ∧ laps_sister > 1 →
  passes_in_same_direction * 2 = 4 :=
by intros; sorry

end NUMINAMATH_GPT_sisterPassesMeInOppositeDirection_l340_34075


namespace NUMINAMATH_GPT_problem1_l340_34029

theorem problem1 (x y : ℝ) 
  (h1 : 3 * x + 2 * y = 10)
  (h2 : x / 2 - (y + 1) / 3 = 1) :
  x = 3 ∧ y = 1 / 2 := 
sorry

end NUMINAMATH_GPT_problem1_l340_34029


namespace NUMINAMATH_GPT_parabola_vertex_coordinates_l340_34013

theorem parabola_vertex_coordinates :
  ∀ x y : ℝ, y = 3 * (x - 7)^2 + 5 → (7, 5) = (7, 5) :=
by
  intros x y h
  exact rfl

end NUMINAMATH_GPT_parabola_vertex_coordinates_l340_34013


namespace NUMINAMATH_GPT_find_S_9_l340_34081

variable (a : ℕ → ℝ)

def arithmetic_sum_9 (S_9 : ℝ) : Prop :=
  (a 1 + a 3 + a 5 = 39) ∧ (a 5 + a 7 + a 9 = 27) ∧ (S_9 = (9 * (a 3 + a 7)) / 2)

theorem find_S_9 
  (h1 : a 1 + a 3 + a 5 = 39)
  (h2 : a 5 + a 7 + a 9 = 27) :
  ∃ S_9, arithmetic_sum_9 a S_9 ∧ S_9 = 99 := 
by
  sorry

end NUMINAMATH_GPT_find_S_9_l340_34081


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l340_34085

-- Define the conditions and parameters for the problem
variables (m : ℝ) (c a e : ℝ)

-- Given conditions
def hyperbola_eq (m : ℝ) := ∀ x y : ℝ, (x^2 / m^2 - y^2 = 4)
def focal_distance : Prop := c = 4
def standard_hyperbola_form : Prop := a^2 = 4 * m^2 ∧ 4 = 4

-- Eccentricity definition
def eccentricity : Prop := e = c / a

-- Main theorem
theorem hyperbola_eccentricity (m : ℝ) (h_pos : 0 < m) (h_foc_dist : focal_distance c) (h_form : standard_hyperbola_form a m) :
  eccentricity e a c :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l340_34085


namespace NUMINAMATH_GPT_solve_for_b_l340_34035

def is_imaginary (z : ℂ) : Prop := z.re = 0

theorem solve_for_b (b : ℝ) (i_is_imag_unit : ∀ (z : ℂ), i * z = z * i):
  is_imaginary (i * (b * i + 1)) → b = 0 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_b_l340_34035


namespace NUMINAMATH_GPT_gcd_lcm_sum_l340_34022

theorem gcd_lcm_sum (a b : ℕ) (h₁ : a = 120) (h₂ : b = 3507) :
  Nat.gcd a b + Nat.lcm a b = 140283 := by 
  sorry

end NUMINAMATH_GPT_gcd_lcm_sum_l340_34022


namespace NUMINAMATH_GPT_equal_tuesdays_thursdays_l340_34078

theorem equal_tuesdays_thursdays (days_in_month : ℕ) (tuesdays : ℕ) (thursdays : ℕ) : (days_in_month = 30) → (tuesdays = thursdays) → (∃ (start_days : Finset ℕ), start_days.card = 2) :=
by
  sorry

end NUMINAMATH_GPT_equal_tuesdays_thursdays_l340_34078


namespace NUMINAMATH_GPT_not_right_triangle_if_angle_A_eq_angle_B_eq_2_angle_C_l340_34071

theorem not_right_triangle_if_angle_A_eq_angle_B_eq_2_angle_C (A B C : ℝ) (h1 : A = 2 * C) (h2 : B = 2 * C) (h3 : A + B + C = 180) : A ≠ 90 ∧ B ≠ 90 ∧ C ≠ 90 := 
by 
  sorry

end NUMINAMATH_GPT_not_right_triangle_if_angle_A_eq_angle_B_eq_2_angle_C_l340_34071


namespace NUMINAMATH_GPT_max_value_of_x2_plus_y2_l340_34095

noncomputable def max_value (x y : ℝ) : ℝ := x^2 + y^2

theorem max_value_of_x2_plus_y2 {x y : ℝ} (h : 5*x^2 + 4*y^2 = 10*x) : max_value x y ≤ 4 := sorry

end NUMINAMATH_GPT_max_value_of_x2_plus_y2_l340_34095


namespace NUMINAMATH_GPT_max_sum_of_digits_l340_34038

theorem max_sum_of_digits (X Y Z : ℕ) (hX : 1 ≤ X ∧ X ≤ 9) (hY : 1 ≤ Y ∧ Y ≤ 9) (hZ : 1 ≤ Z ∧ Z ≤ 9) (hXYZ : X > Y ∧ Y > Z) : 
  10 * X + 11 * Y + Z ≤ 185 :=
  sorry

end NUMINAMATH_GPT_max_sum_of_digits_l340_34038


namespace NUMINAMATH_GPT_lcm_24_36_45_l340_34054

theorem lcm_24_36_45 : Nat.lcm (Nat.lcm 24 36) 45 = 360 := by
  sorry

end NUMINAMATH_GPT_lcm_24_36_45_l340_34054


namespace NUMINAMATH_GPT_proof_problem_l340_34015

def sum_of_sequence (S : ℕ → ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, 0 < n → S n = n * (n + 1) + 2 ∧ S 1 = a 1 ∧ (∀ n, 1 < n → a n = S n - S (n - 1))

def general_term_a (a : ℕ → ℕ) : Prop :=
  a 1 = 4 ∧ (∀ n, 1 < n → a n = 2 * n)

def geometric_sequence (a : ℕ → ℕ) (b : ℕ → ℕ) : Prop :=
  ∀ k : ℕ, 0 < k → 
  a 2 = 4 ∧ a (k+2) = 2 * (k + 2) ∧ a (3 * k + 2) = 2 * (3 * k + 2) →
  b 1 = a 2 ∧ b 2 = a (k + 2) ∧ b 3 = a (3 * k + 2) ∧ 
  (∀ n, b n = 2^(n + 1))

theorem proof_problem :
  ∃ (a b S : ℕ → ℕ),
  sum_of_sequence S a ∧ general_term_a a ∧ geometric_sequence a b :=
sorry

end NUMINAMATH_GPT_proof_problem_l340_34015


namespace NUMINAMATH_GPT_checkerboard_7_strips_l340_34096

theorem checkerboard_7_strips (n : ℤ) :
  (n % 7 = 3) →
  ∃ m : ℤ, n^2 = 9 + 7 * m :=
by
  intro h
  sorry

end NUMINAMATH_GPT_checkerboard_7_strips_l340_34096


namespace NUMINAMATH_GPT_number_of_possible_values_of_a_l340_34047

theorem number_of_possible_values_of_a :
  ∃ (a_values : Finset ℕ), 
    (∀ a ∈ a_values, 3 ∣ a ∧ a ∣ 18 ∧ a ∣ 27 ∧ 0 < a) ∧
    a_values.card = 2 :=
by
  sorry

end NUMINAMATH_GPT_number_of_possible_values_of_a_l340_34047


namespace NUMINAMATH_GPT_prism_volume_is_25_l340_34063

noncomputable def triangle_area (a b : ℝ) : ℝ := (1 / 2) * a * b

noncomputable def prism_volume (base_area height : ℝ) : ℝ := base_area * height

theorem prism_volume_is_25 :
  let a := Real.sqrt 5
  let base_area := triangle_area a a
  let volume := prism_volume base_area 10
  volume = 25 :=
by
  intros
  sorry

end NUMINAMATH_GPT_prism_volume_is_25_l340_34063


namespace NUMINAMATH_GPT_abe_job_time_l340_34051

theorem abe_job_time (A G C: ℕ) : G = 70 → C = 21 → (1 / G + 1 / A = 1 / C) → A = 30 := by
sorry

end NUMINAMATH_GPT_abe_job_time_l340_34051


namespace NUMINAMATH_GPT_lucy_50_cent_items_l340_34055

theorem lucy_50_cent_items :
  ∃ (a b c : ℕ), a + b + c = 30 ∧ 50 * a + 150 * b + 300 * c = 4500 ∧ a = 6 :=
by
  sorry

end NUMINAMATH_GPT_lucy_50_cent_items_l340_34055


namespace NUMINAMATH_GPT_compute_expression_l340_34082

def f (x : ℝ) : ℝ := x + 3
def g (x : ℝ) : ℝ := 2 * x
noncomputable def f_inv (x : ℝ) : ℝ := x - 3
noncomputable def g_inv (x : ℝ) : ℝ := x / 2

theorem compute_expression : 
  f (g_inv (f_inv (f_inv (g (f 15))))) = 18 := by
  sorry

end NUMINAMATH_GPT_compute_expression_l340_34082


namespace NUMINAMATH_GPT_plants_remaining_l340_34072

theorem plants_remaining (plants_initial plants_first_day plants_second_day_eaten plants_third_day_eaten : ℕ)
  (h1 : plants_initial = 30)
  (h2 : plants_first_day = 20)
  (h3 : plants_second_day_eaten = (plants_initial - plants_first_day) / 2)
  (h4 : plants_third_day_eaten = 1)
  : (plants_initial - plants_first_day - plants_second_day_eaten - plants_third_day_eaten) = 4 := 
by
  sorry

end NUMINAMATH_GPT_plants_remaining_l340_34072


namespace NUMINAMATH_GPT_sequence_terms_proof_l340_34088

theorem sequence_terms_proof (P Q R T U V W : ℤ) (S : ℤ) 
  (h1 : S = 10) 
  (h2 : P + Q + R + S = 40) 
  (h3 : Q + R + S + T = 40) 
  (h4 : R + S + T + U = 40) 
  (h5 : S + T + U + V = 40) 
  (h6 : T + U + V + W = 40) : 
  P + W = 40 := 
by 
  have h7 : P + Q + R + 10 = 40 := by rwa [h1] at h2
  have h8 : Q + R + 10 + T = 40 := by rwa [h1] at h3
  have h9 : R + 10 + T + U = 40 := by rwa [h1] at h4
  have h10 : 10 + T + U + V = 40 := by rwa [h1] at h5
  have h11 : T + U + V + W = 40 := h6
  sorry

end NUMINAMATH_GPT_sequence_terms_proof_l340_34088


namespace NUMINAMATH_GPT_minimum_candies_l340_34017

variables (c z : ℕ) (total_candies : ℕ)

def remaining_red_candies := (3 * c) / 5
def remaining_green_candies := (2 * z) / 5
def remaining_total_candies := remaining_red_candies + remaining_green_candies
def red_candies_fraction := remaining_red_candies * 8 = 3 * remaining_total_candies

theorem minimum_candies (h1 : 5 * c = 2 * z) (h2 : red_candies_fraction) :
  total_candies ≥ 35 := sorry

end NUMINAMATH_GPT_minimum_candies_l340_34017


namespace NUMINAMATH_GPT_students_not_good_at_either_l340_34049

theorem students_not_good_at_either (total good_at_english good_at_chinese both_good : ℕ) 
(h₁ : total = 45) 
(h₂ : good_at_english = 35) 
(h₃ : good_at_chinese = 31) 
(h₄ : both_good = 24) : total - (good_at_english + good_at_chinese - both_good) = 3 :=
by sorry

end NUMINAMATH_GPT_students_not_good_at_either_l340_34049


namespace NUMINAMATH_GPT_first_product_of_digits_of_98_l340_34004

theorem first_product_of_digits_of_98 : (9 * 8 = 72) :=
by simp [mul_eq_mul_right_iff] -- This will handle the basic arithmetic automatically

end NUMINAMATH_GPT_first_product_of_digits_of_98_l340_34004


namespace NUMINAMATH_GPT_milo_cash_reward_l340_34000

theorem milo_cash_reward : 
  let three_2s := [2, 2, 2]
  let four_3s := [3, 3, 3, 3]
  let one_4 := [4]
  let one_5 := [5]
  let all_grades := three_2s ++ four_3s ++ one_4 ++ one_5
  let total_grades := all_grades.length
  let total_sum := all_grades.sum
  let average_grade := total_sum / total_grades
  5 * average_grade = 15 := by
  sorry

end NUMINAMATH_GPT_milo_cash_reward_l340_34000


namespace NUMINAMATH_GPT_divisible_by_120_l340_34008

theorem divisible_by_120 (n : ℕ) : 120 ∣ n * (n^2 - 1) * (n^2 - 5 * n + 26) := sorry

end NUMINAMATH_GPT_divisible_by_120_l340_34008


namespace NUMINAMATH_GPT_total_samples_correct_l340_34062

-- Define the conditions as constants
def samples_per_shelf : ℕ := 65
def number_of_shelves : ℕ := 7

-- Define the total number of samples and the expected result
def total_samples : ℕ := samples_per_shelf * number_of_shelves
def expected_samples : ℕ := 455

-- State the theorem to be proved
theorem total_samples_correct : total_samples = expected_samples := by
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_total_samples_correct_l340_34062


namespace NUMINAMATH_GPT_range_of_m_l340_34019

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, x ≤ 3 → (x ≤ m → (x < y → y < m))) → m ≥ 3 := 
by
  sorry

end NUMINAMATH_GPT_range_of_m_l340_34019


namespace NUMINAMATH_GPT_flower_beds_fraction_l340_34028

noncomputable def isosceles_right_triangle_area (leg : ℝ) : ℝ :=
  (1 / 2) * leg^2

noncomputable def fraction_of_yard_occupied_by_flower_beds : ℝ :=
  let leg_length := (25 - 15) / 2
  let area_of_one_triangle := isosceles_right_triangle_area leg_length
  let total_area_of_flower_beds := 2 * area_of_one_triangle
  let area_of_yard := 25 * 5
  total_area_of_flower_beds / area_of_yard

theorem flower_beds_fraction : 
  let leg_length := (25 - 15) / 2
  let area_of_one_triangle := isosceles_right_triangle_area leg_length
  let total_area_of_flower_beds := 2 * area_of_one_triangle
  let area_of_yard := 25 * 5
  total_area_of_flower_beds / area_of_yard = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_flower_beds_fraction_l340_34028


namespace NUMINAMATH_GPT_calculate_discount_l340_34090

theorem calculate_discount
  (original_cost : ℝ)
  (amount_spent : ℝ)
  (h1 : original_cost = 35.00)
  (h2 : amount_spent = 18.00) :
  original_cost - amount_spent = 17.00 :=
by
  sorry

end NUMINAMATH_GPT_calculate_discount_l340_34090


namespace NUMINAMATH_GPT_r_exceeds_s_by_six_l340_34077

theorem r_exceeds_s_by_six (x y : ℚ) (h1 : 3 * x + 2 * y = 16) (h2 : x + 3 * y = 26 / 5) :
  x - y = 6 := by
  sorry

end NUMINAMATH_GPT_r_exceeds_s_by_six_l340_34077


namespace NUMINAMATH_GPT_fibonacci_money_problem_l340_34046

variable (x : ℕ)

theorem fibonacci_money_problem (h : 0 < x - 6) (eq_amounts : 90 / (x - 6) = 120 / x) : 
    90 / (x - 6) = 120 / x :=
sorry

end NUMINAMATH_GPT_fibonacci_money_problem_l340_34046


namespace NUMINAMATH_GPT_number_of_cubes_with_at_least_two_faces_painted_is_56_l340_34027

def one_inch_cubes_with_at_least_two_faces_painted 
  (side_length : ℕ) (face_colors : ℕ) (cubes_per_side : ℕ) :=
  if side_length = 4 ∧ face_colors = 6 ∧ cubes_per_side = 1 then 56 else 0

theorem number_of_cubes_with_at_least_two_faces_painted_is_56 :
  one_inch_cubes_with_at_least_two_faces_painted 4 6 1 = 56 :=
by
  sorry

end NUMINAMATH_GPT_number_of_cubes_with_at_least_two_faces_painted_is_56_l340_34027


namespace NUMINAMATH_GPT_first_representation_second_representation_third_representation_l340_34052

theorem first_representation :
  1 + 2 + 3 + 4 + 5 + 6 + 7 + (8 * 9) = 100 := 
by 
  sorry

theorem second_representation:
  1 + 2 + 3 + 47 + (5 * 6) + 8 + 9 = 100 :=
by
  sorry

theorem third_representation:
  1 + 2 + 3 + 4 + 5 - 6 - 7 + 8 + 92 = 100 := 
by
  sorry

end NUMINAMATH_GPT_first_representation_second_representation_third_representation_l340_34052


namespace NUMINAMATH_GPT_simplify_sqrt_sum_l340_34048

noncomputable def sqrt_72 : ℝ := Real.sqrt 72
noncomputable def sqrt_32 : ℝ := Real.sqrt 32
noncomputable def sqrt_27 : ℝ := Real.sqrt 27
noncomputable def result : ℝ := 10 * Real.sqrt 2 + 3 * Real.sqrt 3

theorem simplify_sqrt_sum :
  sqrt_72 + sqrt_32 + sqrt_27 = result :=
by
  sorry

end NUMINAMATH_GPT_simplify_sqrt_sum_l340_34048


namespace NUMINAMATH_GPT_slower_train_speed_l340_34065

theorem slower_train_speed (v : ℝ) (faster_train_speed : ℝ) (time_pass : ℝ) (train_length : ℝ) :
  (faster_train_speed = 46) →
  (time_pass = 36) →
  (train_length = 50) →
  (v = 36) :=
by
  intro h1 h2 h3
  -- Formal proof goes here
  sorry

end NUMINAMATH_GPT_slower_train_speed_l340_34065


namespace NUMINAMATH_GPT_solve_for_m_l340_34036

theorem solve_for_m (m : ℝ) (h : (4 * m + 6) * (2 * m - 5) = 159) : m = 5.3925 :=
sorry

end NUMINAMATH_GPT_solve_for_m_l340_34036
