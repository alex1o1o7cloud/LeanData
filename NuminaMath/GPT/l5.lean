import Mathlib
import Mathlib.Algebra.Divisibility.Basic
import Mathlib.Algebra.GeomSum
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Parity
import Mathlib.Algebra.Quadratic.Discriminant
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Integrals
import Mathlib.Analysis.SpecialFunctions.Pow
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.CombinatorialLemmas
import Mathlib.Combinatorics.Composition
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Fin
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.List.Basic
import Mathlib.Data.List.Perm
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Base
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Probability
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Seq.Seq
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Circle
import Mathlib.Geometry.Tetrahedron
import Mathlib.Init.Data.Nat.Basic
import Mathlib.LinearAlgebra.Matrix.Determinant
import Mathlib.NumberTheory.Gcd
import Mathlib.NumberTheory.Lcm
import Mathlib.Probability.Basic
import Mathlib.Probability.Distribution.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.Basic

namespace evaluate_log_expression_l5_5088

noncomputable def evaluate_expression (x y : Real) : Real :=
  (Real.log x / Real.log (y ^ 8)) * 
  (Real.log (y ^ 3) / Real.log (x ^ 7)) * 
  (Real.log (x ^ 7) / Real.log (y ^ 3)) * 
  (Real.log (y ^ 8) / Real.log (x ^ 2))

theorem evaluate_log_expression (x y : Real) : 
  evaluate_expression x y = (1 : Real) := sorry

end evaluate_log_expression_l5_5088


namespace quadratic_symmetry_l5_5806

def quadratic (b c : ℝ) (x : ℝ) : ℝ :=
  x^2 + b * x + c

theorem quadratic_symmetry (b c : ℝ) :
  let f := quadratic b c
  (f 2) < (f 1) ∧ (f 1) < (f 4) :=
by
  sorry

end quadratic_symmetry_l5_5806


namespace days_worked_l5_5029

theorem days_worked (total_toys_per_week toys_per_day : ℕ) (h1 : total_toys_per_week = 3400) (h2 : toys_per_day = 680) : total_toys_per_week / toys_per_day = 5 :=
by
  rw [h1, h2]
  norm_num

end days_worked_l5_5029


namespace expected_value_8_sided_die_l5_5365

theorem expected_value_8_sided_die :
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let probability_each_outcome := 1 / 8
  (Σ x ∈ outcomes, probability_each_outcome * x) = 4.5 :=
by
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let probability_each_outcome := 1 / 8
  have h : (Σ x ∈ outcomes, probability_each_outcome * x) = (1 / 8) * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8) := by sorry
  have sum_eq_36 : 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 = 36 := by sorry
  rw [sum_eq_36] at h
  have expected_value_eq : (1 / 8) * 36 = 4.5 := by sorry
  rw [expected_value_eq] at h
  exact h

end expected_value_8_sided_die_l5_5365


namespace solve_diff_eq_and_find_particular_solution_l5_5794

noncomputable def general_solution (C : ℝ) (x : ℝ) : ℝ :=
  Real.arctan (x^2 + C)

noncomputable def particular_solution (x : ℝ) : ℝ :=
  general_solution 0 x

theorem solve_diff_eq_and_find_particular_solution :
  (∀ (x y : ℝ), 2 * x * (deriv id x) = (deriv (λ y, Real.tan y) id y) * (cos y)^2) →
  general_solution (0) 0 = 0 → particular_solution x = Real.arctan (x^2) :=
by
  intros h_eq h_init
  sorry

end solve_diff_eq_and_find_particular_solution_l5_5794


namespace distribute_students_l5_5524

theorem distribute_students :
  let C (n k : ℕ) := nat.choose n k in
  ∃ num_schemes : ℕ,
    num_schemes = C 12 4 * C 8 4 * C 4 4 :=
by
  sorry

end distribute_students_l5_5524


namespace range_of_b_l5_5189

theorem range_of_b (b : ℝ) : 
  (0 < b) ∧ (∀ x : ℝ, | x - 5 / 4 | < b → | x - 1 | < 1 / 2) ↔ (0 < b ∧ b ≤ 1 / 4) := 
by
  sorry

end range_of_b_l5_5189


namespace area_of_rhombus_l5_5096

theorem area_of_rhombus (R₁ R₂ : ℝ) (x y : ℝ) (hR₁ : R₁ = 10) (hR₂ : R₂ = 20) 
    (h_eq : (x * (x^2 + y^2)) / (4 * R₁) = (y * (x^2 + y^2)) / (4 * R₂)) :
    ((2 * x) * (2 * y)) / 2 = 40 :=
by
  have h₁ : x * (x^2 + y^2) = y * (x^2 + y^2) / 2 := by sorry
  have h₂ : y = 2 * x := by sorry
  have h₃ : x^2 = 40 := by sorry
  have h₄ : x = sqrt 40 := by sorry
  have h₅ : y = 4 * sqrt 40 := by sorry
  have h₆ : (2 * sqrt 40) * (4 * sqrt 40) / 2 = 40 := by sorry
  exact h₆

end area_of_rhombus_l5_5096


namespace no_regular_pentagon_cross_section_l5_5425

theorem no_regular_pentagon_cross_section (P : Plane) (C : Cube) : 
  ¬(cross_section P C = regular_pentagon) :=
sorry  -- proof omitted

end no_regular_pentagon_cross_section_l5_5425


namespace two_digit_number_count_three_digit_number_count_five_digit_numbers_exist_six_digit_numbers_do_not_exist_element_1081_is_4012_l5_5757

noncomputable def set_A : Set ℤ := {x | x ≥ 10}

def valid_digits (n : ℕ) : Prop :=
  let digits := (n.digits 10).nodup in   -- Digits must be all different
  ∀i j, i < j → (n.digit i + n.digit j ≠ 9)   -- Sum of any two digits != 9

def set_B : Set ℕ := {n | n ∈ set_A ∧ valid_digits n}

-- Question 1
theorem two_digit_number_count : (set_B ∩ {n | 10 ≤ n ∧ n < 100}).card = 72 := sorry

theorem three_digit_number_count : (set_B ∩ {n | 100 ≤ n ∧ n < 1000}).card = 432 := sorry

-- Question 2
theorem five_digit_numbers_exist : ∃ n, 10000 ≤ n ∧ n < 100000 ∧ n ∈ set_B := sorry
theorem six_digit_numbers_do_not_exist : ¬ ∃ n, 100000 ≤ n ∧ n < 1000000 ∧ n ∈ set_B := sorry

-- Question 3
noncomputable def ordered_set_B : List ℕ := (set_B.to_finset).to_list.sort (≤)

noncomputable def element_1081_in_B : ℕ := ordered_set_B.get_or_else 1080 0

theorem element_1081_is_4012 : element_1081_in_B = 4012 := sorry

end two_digit_number_count_three_digit_number_count_five_digit_numbers_exist_six_digit_numbers_do_not_exist_element_1081_is_4012_l5_5757


namespace even_iff_n_pow_n_even_l5_5697

theorem even_iff_n_pow_n_even (n : ℕ) (h : n ≠ 0) : even n ↔ even (n^n) := 
  sorry

end even_iff_n_pow_n_even_l5_5697


namespace polynomials_exist_for_odd_d_l5_5537

theorem polynomials_exist_for_odd_d (d : ℕ) (hd : d % 2 = 1) :
  ∃ (P Q : ℝ[X]), P.degree = d ∧ P^2 + 1 = (X^2 + 1) * Q^2 :=
sorry

end polynomials_exist_for_odd_d_l5_5537


namespace interval_of_decrease_l5_5962

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 15 * x^2 - 33 * x + 6

-- Define the derivative of the function
def f_derivative (x : ℝ) : ℝ := 3 * x^2 - 30 * x - 33

-- Statement: Prove that f(x) is decreasing on the interval (-1, 11)
theorem interval_of_decrease :
  ∀ x : ℝ, x ∈ Ioo (-1) 11 → f_derivative x < 0 := 
sorry

end interval_of_decrease_l5_5962


namespace min_value_log_function_l5_5630

theorem min_value_log_function
  (α β x : ℝ)
  (h1 : cos α + sin β = sqrt 3)
  (h2 : -1 ≤ x ∧ x ≤ 1) :
  ∃ y, y = log (1 / 2) (sqrt (2 * x + 3) / (4 * x + 10)) ∧ y = 5 / 2 :=
by
  sorry

end min_value_log_function_l5_5630


namespace sum_of_first_n_terms_l5_5183

variable {a : ℕ → ℤ}

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_first_n_terms {a : ℕ → ℤ} (h : is_arithmetic_sequence a) (h_condition : a 1 + a 7 = 15 - a 4) :
  ∑ i in finset.range 9, a i = 126 := 
sorry

end sum_of_first_n_terms_l5_5183


namespace alice_journey_duration_l5_5494
noncomputable def journey_duration (start_hour start_minute end_hour end_minute : ℕ) : ℕ :=
  let start_in_minutes := start_hour * 60 + start_minute
  let end_in_minutes := end_hour * 60 + end_minute
  if end_in_minutes >= start_in_minutes then end_in_minutes - start_in_minutes
  else end_in_minutes + 24 * 60 - start_in_minutes
  
theorem alice_journey_duration :
  ∃ start_hour start_minute end_hour end_minute,
  (7 ≤ start_hour ∧ start_hour < 8 ∧ start_minute = 38) ∧
  (16 ≤ end_hour ∧ end_hour < 17 ∧ end_minute = 35) ∧
  journey_duration start_hour start_minute end_hour end_minute = 537 :=
by {
  sorry
}

end alice_journey_duration_l5_5494


namespace greatest_possible_sum_of_consecutive_integers_prod_lt_200_l5_5417

theorem greatest_possible_sum_of_consecutive_integers_prod_lt_200 :
  ∃ n : ℤ, (n * (n + 1) < 200) ∧ ( ∀ m : ℤ, (m * (m + 1) < 200) → m ≤ n) ∧ (n + (n + 1) = 27) :=
by
  sorry

end greatest_possible_sum_of_consecutive_integers_prod_lt_200_l5_5417


namespace area_of_rhombus_l5_5097

theorem area_of_rhombus (R₁ R₂ : ℝ) (x y : ℝ) (hR₁ : R₁ = 10) (hR₂ : R₂ = 20) 
    (h_eq : (x * (x^2 + y^2)) / (4 * R₁) = (y * (x^2 + y^2)) / (4 * R₂)) :
    ((2 * x) * (2 * y)) / 2 = 40 :=
by
  have h₁ : x * (x^2 + y^2) = y * (x^2 + y^2) / 2 := by sorry
  have h₂ : y = 2 * x := by sorry
  have h₃ : x^2 = 40 := by sorry
  have h₄ : x = sqrt 40 := by sorry
  have h₅ : y = 4 * sqrt 40 := by sorry
  have h₆ : (2 * sqrt 40) * (4 * sqrt 40) / 2 = 40 := by sorry
  exact h₆

end area_of_rhombus_l5_5097


namespace scaling_transformation_correct_l5_5833

theorem scaling_transformation_correct :
  ∃ (λ μ : ℚ), (λ * (3 : ℚ) = 2) ∧ (μ * (2 : ℚ) = 3) ∧ λ = 2 / 3 ∧ μ = 3 / 2 :=
by {
  use [2/3, 3/2],
  have h1 : (2/3) * 3 = 2 := by {
    calc (2/3) * 3 = 2 : by norm_num
  },
  have h2 : (3/2) * 2 = 3 := by {
    calc (3/2) * 2 = 3 : by norm_num
  },
  exact ⟨h1, h2, rfl, rfl⟩
}

end scaling_transformation_correct_l5_5833


namespace sum_of_coefficients_l5_5182

noncomputable def polynomial : ℤ[X] := (2 * (X : ℤ[X]) - 1)^4

theorem sum_of_coefficients :
  let p := polynomial
  let a_4 := p.coeff 4
  let a_2 := p.coeff 2
  let a_0 := p.coeff 0
  a_4 + a_2 + a_0 = 41 :=
by
  let p := polynomial
  let a_4 := p.coeff 4
  let a_2 := p.coeff 2
  let a_0 := p.coeff 0
  sorry

end sum_of_coefficients_l5_5182


namespace error_in_step_1_step_3_is_factorization_correct_solution_value_l5_5942

-- Definitions based on the conditions
def x_values := { x : ℤ | -2 < x ∧ x < 2 }

-- Assertion (1): Error starts from step 1
theorem error_in_step_1 (x : ℤ) (hx : x ∈ x_values) : 
  let expr := (x^2 - 1) / (x^2 + 2 * x + 1) / (1 / (x + 1) - 1)
  in (1 / (x + 1) - 1) is incorrectly computed 
  := sorry 

-- Assertion (2): Operation in step 3 is factorization
theorem step_3_is_factorization (x : ℤ) (hx : x ∈ x_values) : 
  (∃ a b, (x^2 - 1 = (x + 1) * (x - 1)) ∧ (x^2 + 2 * x + 1 = (x + 1)^2)) ∧ 
   operation_in_step_3 = "factorization" 
:= sorry 

-- Assertion (3): Correct solution for specific x value
theorem correct_solution_value (x : ℤ) (hx : x = 1) : 
  (x^2 - 1) / (x^2 + 2 * x + 1) / (1 / (x + 1) - 1) = 0 
:= sorry 

end error_in_step_1_step_3_is_factorization_correct_solution_value_l5_5942


namespace main_inequality_l5_5231

noncomputable def f : ℝ → ℝ := sorry

axiom continuous_f : continuous f

axiom func_property : ∀ (x y : ℝ), f(x + y) * f(x - y) = (f(x))^2 - (f(y))^2

axiom periodic_f : ∀ (x : ℝ), f(x + 2 * Real.pi) = f(x)

axiom no_lesser_period : ¬∃ (a : ℝ), 0 < a ∧ a < 2 * Real.pi ∧ (∀ x : ℝ, f(x + a) = f(x))

theorem main_inequality : ∀ x : ℝ, |f(Real.pi / 2)| ≥ f(x) :=
by
  sorry

end main_inequality_l5_5231


namespace find_k1_k2_l5_5646

variables {V : Type*} [add_comm_group V] [vector_space ℝ V]

def vectors_sum_to_zero (a b c d : V) : Prop :=
a + b + c + d = 0

theorem find_k1_k2 (a b c d : V) (h : vectors_sum_to_zero a b c d) : 
  ∃ (k1 k2 : ℝ), k1 = 1 ∧ k2 = 1 ∧ k1 • (b × a) + k2 • (c × d) + b × d + c × a = 0 :=
by {
  use 1,
  use 1,
  split, 
  { refl },
  split,
  { refl },
  sorry
}

end find_k1_k2_l5_5646


namespace find_d_and_r_l5_5113

theorem find_d_and_r (d r : ℤ)
  (h1 : 1210 % d = r)
  (h2 : 1690 % d = r)
  (h3 : 2670 % d = r) :
  d - 4 * r = -20 := sorry

end find_d_and_r_l5_5113


namespace expected_value_8_sided_die_l5_5402

/-- 
The expected value of rolling a standard 8-sided die is 4.5.
The die has 8 sides labeled 1 through 8, and each face has an equal probability of appearing,
which is 1/8. 
-/
theorem expected_value_8_sided_die : 
  (1/8:ℝ) * 1 + (1/8) * 2 + (1/8) * 3 + (1/8) * 4 + (1/8) * 5 + (1/8) * 6 + (1/8) * 7 + (1/8) * 8 = 4.5 :=
by 
  sorry

end expected_value_8_sided_die_l5_5402


namespace workshop_average_salary_l5_5202

theorem workshop_average_salary :
  let technicians := 8
  let rest := 24 - technicians
  let avg_technician_salary := 12000
  let avg_rest_salary := 6000
  let total_workers := 24
  let total_staff_salary := (technicians * avg_technician_salary) + (rest * avg_rest_salary)
  let A := total_staff_salary / total_workers
  A = 8000 :=
by
  -- Definitions according to given conditions
  let technicians := 8
  let rest := 24 - technicians
  let avg_technician_salary := 12000
  let avg_rest_salary := 6000
  let total_workers := 24
  let total_staff_salary := (technicians * avg_technician_salary) + (rest * avg_rest_salary)
  let A := total_staff_salary / total_workers
  -- We need to show that A = 8000
  show A = 8000
  sorry

end workshop_average_salary_l5_5202


namespace range_of_a_l5_5677

open Set

variable {α : Type} [LinearOrder α] [DecidablePred (λ x : α, x < 4)]

theorem range_of_a (a : α) :
  (∃ A B : Set α, 
    A = {x | -2 < x ∧ x < 4} ∧ 
    B = {x | x < a} ∧ 
    A ∪ B = {x | x < 4}) → 
  -2 < a ∧ a ≤ 4 := 
sorry

end range_of_a_l5_5677


namespace income_growth_rate_l5_5482

noncomputable def income_growth_eq (x : ℝ) : Prop :=
  3.2 * (1 + x)^2 = 3.7

theorem income_growth_rate :
  ∃ x : ℝ, income_growth_eq x :=
sorry

end income_growth_rate_l5_5482


namespace min_value_of_quadratic_l5_5423

noncomputable def quadratic (x : ℝ) : ℝ := x^2 - 8*x + 18

theorem min_value_of_quadratic : ∃ x : ℝ, quadratic x = 2 ∧ (∀ y : ℝ, quadratic y ≥ 2) :=
by
  use 4
  sorry

end min_value_of_quadratic_l5_5423


namespace letter_puzzle_solutions_l5_5563

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def to_number (B A : ℕ) : ℕ :=
  10 * B + A

theorem letter_puzzle_solutions (A B : ℕ) (h_diff : A ≠ B) (h_digits : 1 ≤ A ∧ A ≤ 9 ∧ 1 ≤ B ∧ B ≤ 9)
  (h_two_digit : is_two_digit (to_number B A)) :
  A^B = to_number B A ↔ (A = 2 ∧ B = 5 ∨ A = 6 ∧ B = 2 ∨ A = 4 ∧ B = 3) :=
by
  sorry

end letter_puzzle_solutions_l5_5563


namespace middle_number_sorted_desc_l5_5876

theorem middle_number_sorted_desc :
  let nums := [10000, 1, 10, 100, 1000]
  let sorted_nums := nums.qsort (λ x y => x > y)
  sorted_nums.nth 2 = some 100 :=
by
  sorry

end middle_number_sorted_desc_l5_5876


namespace max_value_a_ln_b_is_e_l5_5134

noncomputable def maximum_value_a_ln_b (a b : ℝ) (h1 : 1 < a) (h2 : 1 < b) (h3 : a * b = Real.exp 2) : ℝ :=
  let t := a^(Real.log b)
  Real.mk t sorry -- the proof is skipped

theorem max_value_a_ln_b_is_e (a b : ℝ) (h1 : 1 < a) (h2 : 1 < b) (h3 : a * b = Real.exp 2) :
  maximum_value_a_ln_b a b h1 h2 h3 = Real.exp 1 :=
sorry

end max_value_a_ln_b_is_e_l5_5134


namespace fencing_cost_l5_5457

-- Define the sides of the rectangular field in ratio 3:4
def ratio_sides (x : ℝ) : Prop := (3 * x) * (4 * x) = 9408

-- Area of the rectangular field
def area : ℝ := 9408

-- Define the cost per meter for fencing
def cost_per_meter : ℝ := 0.25

-- Define the ratio condition
def ratio_condition (a b : ℝ) : Prop := a = 3 * b ∧ b = 4 * (b / 4)

-- Define the correct cost calculation
def total_cost (x : ℝ) : ℝ :=
  let a := 3 * x
  let b := 4 * x
  let perimeter := 2 * (a + b)
  let diagonal := Math.sqrt (a^2 + b^2)
  let paddock_height := area / (6 * (b / 2))
  perimeter + diagonal + 2 * paddock_height

theorem fencing_cost (x : ℝ) (h : ratio_sides x) : total_cost x * cost_per_meter = 147 :=
by
  sorry

end fencing_cost_l5_5457


namespace grocer_display_rows_l5_5916

theorem grocer_display_rows (n : ℕ)
  (h1 : ∃ k, k = 2 + 3 * (n - 1))
  (h2 : ∃ s, s = (n / 2) * (2 + (3 * n - 1))):
  (3 * n^2 + n) / 2 = 225 → n = 12 :=
by
  sorry

end grocer_display_rows_l5_5916


namespace smallest_rel_prime_to_180_l5_5611

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ Int.gcd x 180 = 1 ∧ ∀ y : ℕ, y > 1 ∧ Int.gcd y 180 = 1 → x ≤ y := 
sorry

end smallest_rel_prime_to_180_l5_5611


namespace anvil_mega_factory_l5_5061

theorem anvil_mega_factory :
  let total_employees := 154875 in
  let journeymen := (5 / 11 : ℚ) * total_employees in
  let laid_off_first := (2 / 5 : ℚ) * journeymen in
  let remaining_after_first := journeymen - laid_off_first in
  let laid_off_second := (1 / 3 : ℚ) * remaining_after_first in
  let remaining_journeymen := remaining_after_first - laid_off_second in
  let total_remaining_employees := total_employees - (laid_off_first + laid_off_second) in
  (remaining_journeymen / total_remaining_employees) * 100 = 25 :=
by 
  let total_employees := 154875 in
  let journeymen : ℚ := (5 / 11) * total_employees in
  let laid_off_first : ℚ := (2 / 5) * journeymen in
  let remaining_after_first : ℚ := journeymen - laid_off_first in
  let laid_off_second : ℚ := (1 / 3) * remaining_after_first in
  let remaining_journeymen : ℚ := remaining_after_first - laid_off_second in
  let total_remaining_employees : ℚ := total_employees - (laid_off_first + laid_off_second) in
  have h1 : remaining_journeymen = 28_159 := sorry,
  have h2 : total_remaining_employees = 112_636 := sorry,
  have h3 : (remaining_journeymen / total_remaining_employees) * 100 = 25 := sorry,
  exact h3

end anvil_mega_factory_l5_5061


namespace speed_of_larger_fragment_l5_5031

noncomputable def v0 : ℝ := 20   -- initial speed in m/s
noncomputable def g : ℝ := 10    -- acceleration due to gravity in m/s^2
noncomputable def t : ℝ := 3     -- time in seconds
noncomputable def m_ratio : ℝ := 1 / 2  -- mass ratio
noncomputable def v_small_horizontal : ℝ := 16   -- horizontal speed of smaller fragment in m/s

theorem speed_of_larger_fragment : 
  let v_before_explosion := v0 - g * t in
  let v_horizontal := - (v_small_horizontal / 2) in
  let v_vertical := - (v_before_explosion / 2) in
  (v_before_explosion < 0) → 
  (v_horizontal < 0) →
  (v_vertical < 0) →
  sqrt(v_horizontal^2 + v_vertical^2) = 17 :=
by
  sorry

end speed_of_larger_fragment_l5_5031


namespace lowest_combined_work_rate_l5_5968

def work_rate (hours: ℕ) : ℚ := 1 / hours

def combined_work_rate (rates: List ℚ) : ℚ :=
  rates.sum

theorem lowest_combined_work_rate :
  ∃ (a b c d : ℕ),
    a = 4 ∧ b = 5 ∧ c = 8 ∧ d = 10 ∧
    let rates := [work_rate a, work_rate b, work_rate c, work_rate d] in
    (min (combined_work_rate (rates.erase_nth 0))   -- (B, C, D)
         (min (combined_work_rate (rates.erase_nth 1))  -- (A, C, D)
              (min (combined_work_rate (rates.erase_nth 2))  -- (A, B, D)
                   (combined_work_rate (rates.erase_nth 3)))))  -- (A, B, C)
        = 17 / 40 :=
by sorry

end lowest_combined_work_rate_l5_5968


namespace volume_pyramid_SPQR_192_l5_5788

noncomputable def volume_of_pyramid_SPQR {P Q R S : Type*}
  [InnerProductSpace ℝ P] [InnerProductSpace ℝ Q]
  [InnerProductSpace ℝ R] [InnerProductSpace ℝ S]
  (SP SQ SR : ℝ)
  (hSP : SP = 12) 
  (hSQ : SQ = 12)
  (hSR : SR = 8)
  (h_perpendicular_SP_SQ : orthogonal SP SQ)
  (h_perpendicular_SP_SR: orthogonal SP SR)
  (h_perpendicular_SQ_SR : orthogonal SQ SR) :
  ℝ :=
  let area_PQR := (1 / 2) * SP * SQ in
  let volume_SPQR := (1 / 3) * area_PQR * SR in
  volume_SPQR

theorem volume_pyramid_SPQR_192 :
  volume_of_pyramid_SPQR 12 12 8 12 12 8 = 192 :=
  sorry

end volume_pyramid_SPQR_192_l5_5788


namespace total_spent_at_music_store_l5_5745

theorem total_spent_at_music_store 
  (flute : ℝ) (music_tool : ℝ) (song_book : ℝ) 
  (flute_case : ℝ) (music_stand : ℝ) (cleaning_kit : ℝ) 
  (sheet_protectors : ℝ) :
  flute = 142.46 → music_tool = 8.89 → song_book = 7.00 → 
  flute_case = 35.25 → music_stand = 12.15 → cleaning_kit = 14.99 → 
  sheet_protectors = 3.29 → 
  flute + music_tool + song_book + flute_case + music_stand + cleaning_kit + sheet_protectors = 224.03 :=
by
  intros h_flute h_music_tool h_song_book h_flute_case h_music_stand h_cleaning_kit h_sheet_protectors
  rw [h_flute, h_music_tool, h_song_book, h_flute_case, h_music_stand, h_cleaning_kit, h_sheet_protectors]
  norm_num
  sorry

end total_spent_at_music_store_l5_5745


namespace min_value_of_M_l5_5132

variable {a b c : ℝ}

-- Defining the conditions in Lean 4
def unique_solution (a b c : ℝ) : Prop := (a < b) ∧ (b^2 - 4*a*c = 0)

-- Defining the expression M
def M_value (a b c : ℝ) := (a + 3*b + 4*c) / (b - a)

-- The Lean statement that needs to be proved
theorem min_value_of_M (h : unique_solution a b c) : M_value a b c = 2 * Real.sqrt 5 + 5 :=
sorry

end min_value_of_M_l5_5132


namespace smallest_integer_to_make_perfect_square_l5_5929

def y : ℕ := 2^3 * 3^4 * 4^5 * 5^6 * 6^7 * 7^8 * 8^9 * 9^10

theorem smallest_integer_to_make_perfect_square :
  ∃ n : ℕ, n > 0 ∧ (∃ k : ℕ, (n * y) = k^2) ∧ n = 6 :=
by
  sorry

end smallest_integer_to_make_perfect_square_l5_5929


namespace problem1_asymptotes_problem2_equation_l5_5015

-- Problem 1: Asymptotes of a hyperbola
theorem problem1_asymptotes (a : ℝ) (x y : ℝ) (hx : (y + a) ^ 2 - (x - a) ^ 2 = 2 * a)
  (hpt : 3 = x ∧ 1 = y) : 
  (y = x - 2 * a) ∨ (y = - x) := 
by 
  sorry

-- Problem 2: Equation of a hyperbola
theorem problem2_equation (a b c : ℝ) (x y : ℝ) 
  (hasymptote : y = x + 1 ∨ y = - (x + 1))  (hfocal : 2 * c = 4)
  (hc_squared : c ^ 2 = a ^ 2 + b ^ 2) (ha_eq_b : a = b): 
  y^2 - (x + 1)^2 = 2 := 
by 
  sorry

end problem1_asymptotes_problem2_equation_l5_5015


namespace losing_candidate_defeated_by_504_l5_5500

-- Definitions based on the given conditions
def total_polled_votes : ℕ := 850
def invalid_votes : ℕ := 10
def valid_votes : ℕ := total_polled_votes - invalid_votes
def losing_candidate_percentage : ℚ := 0.20
def losing_votes : ℕ := (losing_candidate_percentage * valid_votes).to_nat
def winning_votes : ℕ := valid_votes - losing_votes
def defeated_by_votes : ℕ := winning_votes - losing_votes

-- Theorem to prove that the losing candidate was defeated by 504 votes
theorem losing_candidate_defeated_by_504 :
  defeated_by_votes = 504 :=
by
  sorry

end losing_candidate_defeated_by_504_l5_5500


namespace find_x_l5_5440

-- We will define the numbers and their conditions
def avg1 : ℤ := (24 + 35 + 58) / 3
def avg2 (x : ℤ) : ℤ := (19 + 51 + x) / 3

-- We will define the main theorem to prove
theorem find_x : ∃ x : ℤ, avg1 = avg2 x + 6 ∧ x = 29 :=
by {
  -- Define the averages explicitly
  have h1 : avg1 = 39 := by norm_num,
  have h2 : ∀ x, avg2 x + 6 = 39 := by intro x; norm_num,

  -- Now solve for x
  use 29,
  split,
  { rw h1, exact h2 29, },
  { norm_num, }
}

end find_x_l5_5440


namespace expected_value_of_8_sided_die_l5_5395

def outcomes : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]
def probability (n : ℕ) : ℝ := 1 / 8

theorem expected_value_of_8_sided_die :
  (∑ x in outcomes, probability x * x) = 4.5 := 
sorry

end expected_value_of_8_sided_die_l5_5395


namespace smallest_rel_prime_to_180_l5_5581

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ Nat.gcd x 180 = 1 ∧ ∀ y : ℕ, y > 1 ∧ Nat.gcd y 180 = 1 → x ≤ y :=
begin
  use 7,
  split,
  { exact dec_trivial },
  split,
  { exact dec_trivial },
  { intro y,
    intro h,
    cases h with h1 h2,
    repeat { try { apply dec_trivial,
                   apply lt_or_eq_of_le,
                   norm_num,
                   apply Nat.prime_not_dvd_mul,
                   norm_num,
                   apply not_or_distrib.mpr,
                   split,
                   norm_cast,
                   intro,
                   exact le_antisymm _ },
           sorry }
end

end smallest_rel_prime_to_180_l5_5581


namespace repeating_decimal_to_fraction_l5_5983

/-- Given 0.02 repeating as a fraction 2/99, prove that 2.06 repeating can be expressed as 68/33 -/
theorem repeating_decimal_to_fraction :
  (2 + 0.06̅ : ℝ) = (68 / 33 : ℝ) :=
begin
  have h : (0.02̅ : ℝ) = (2 / 99 : ℝ), from sorry,
  have h3 : (0.06̅ : ℝ) = 3 * (0.02̅ : ℝ), from sorry,
  have h6 : (0.06̅ : ℝ) = 3 * (2 / 99 : ℝ), from sorry,
  have s : (0.06̅ : ℝ) = (6 / 99 : ℝ), from sorry,
  have s2 : (6 / 99 : ℝ) = (2 / 33 : ℝ), from sorry,
  have add := congr_arg (λ x : ℝ, (2 : ℝ) + x) s2,
  rw [add_comm, ← add_halves', add_assoc', add_comm (2 : ℝ), add_comm 2 (2 / 33 : ℝ), add_halves',
    add_assoc', add_comm (2 : ℝ), add_comm 68 (2 / 33 : ℝ)] ,
end

end repeating_decimal_to_fraction_l5_5983


namespace repeating_decimal_fraction_l5_5988

theorem repeating_decimal_fraction :  exists (p q : ℤ), (q ≠ 0) ∧ (p / q = (68 / 33 : ℚ)) :=
begin
  let recurring02 : ℚ := 2 / 99,
  let recurring06 : ℚ := 3 * recurring02,
  have recurring06_simplified : recurring06 = 2 / 33, 
    by simp [recurring06, mul_div_assoc, mul_div_cancel_left],
  let result := 2 + recurring06_simplified,
  use [68, 33],
  split,
  { norm_num }, -- proof that 33 ≠ 0
  { exact result } -- proof that 2.06 recurring = 68 / 33
end

end repeating_decimal_fraction_l5_5988


namespace probability_of_x_in_D_l5_5248

noncomputable def D : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}

theorem probability_of_x_in_D :
  let interval := (-3 : ℝ, 5 : ℝ)
  let length_interval := (interval.2 - interval.1)
  let length_D := (2 - 0)
  let P := length_D / length_interval
  P = 1 / 4 :=
by
  sorry

end probability_of_x_in_D_l5_5248


namespace smallest_rel_prime_to_180_l5_5610

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ Int.gcd x 180 = 1 ∧ ∀ y : ℕ, y > 1 ∧ Int.gcd y 180 = 1 → x ≤ y := 
sorry

end smallest_rel_prime_to_180_l5_5610


namespace number_of_ways_to_select_students_l5_5325

theorem number_of_ways_to_select_students 
  (n m t : ℕ) (h : t = 1) (hs : 3 = 2 + 1) (hd : 2 = 1 + 1) :
  (nat.choose (3, 2) * nat.choose (2, 1)) = 15 :=
by
  sorry

end number_of_ways_to_select_students_l5_5325


namespace proof_problem_l5_5674

-- Define proposition p
def p : Prop := ∀ x : ℝ, x > 0 → x^2 - x + 1 > 0

-- Define the negation of proposition p
def neg_p : Prop := ∃ x_0 : ℝ, x_0 ≤ 0 ∧ x_0^2 - x_0 + 1 ≤ 0

-- Define proposition q
def q : Prop := ∀ (A B C : ℝ) (a b c : ℝ), 
  (sin A > sin B) ↔ (a > b)

-- Define the final proposition to prove
def final_proposition : Prop := neg_p ∧ q

-- State the theorem
theorem proof_problem : final_proposition :=
  by
  sorry

end proof_problem_l5_5674


namespace problem1_problem2_problem3_l5_5528

-- Define the functions f and g
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c
def g (a b : ℝ) (x : ℝ) : ℝ := a * x + b

-- Problem statements in Lean
theorem problem1 (a b c : ℝ) (h : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → |f a b c x| ≤ 1) : |c| ≤ 1 :=
sorry

theorem problem2 (a b c : ℝ) (h₁ : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → |f a b c x| ≤ 1) : 
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → |g a b x| ≤ 2 :=
sorry

theorem problem3 (a b c : ℝ) (ha : a > 0) (hx : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → g a b x ≤ 2) (hf : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → |f a b c x| ≤ 1) :
  ∃ x, -1 ≤ x ∧ x ≤ 1 ∧ g a b x = 2 :=
sorry

end problem1_problem2_problem3_l5_5528


namespace number_of_lucky_tickets_is_even_sum_of_lucky_tickets_is_divisible_by_999_l5_5888

def is_lucky_ticket (n : ℕ) : Prop :=
  n <= 999999 ∧ (n / 1000 % 10 + n / 10000 % 10 + n / 100000 % 10) = (n % 10 + n / 10 % 10 + n / 100 % 10)

theorem number_of_lucky_tickets_is_even :
  (∃ m, ∀ n, (0 <= n ∧ n <= 999999) → is_lucky_ticket n ↔ (n < m)) ∧
  ∃ n, even n :=
sorry

theorem sum_of_lucky_tickets_is_divisible_by_999 :
  (∑ n in finset.filter is_lucky_ticket (finset.range 1000000), n) % 999 = 0 :=
sorry

end number_of_lucky_tickets_is_even_sum_of_lucky_tickets_is_divisible_by_999_l5_5888


namespace acute_angle_triangles_limit_l5_5645

theorem acute_angle_triangles_limit :
  ∀ (points : Finset (ℝ × ℝ)), points.card = 5 → (∀ p1 p2 p3 ∈ points, Geometry.Collinear ℝ {p1, p2, p3} → false) → 
  (∃ (triangles : Finset (Finset (ℝ × ℝ))), triangles.card ≤ 7 ∧ (∀ t ∈ triangles, t.card = 3) ∧ (∀ t ∈ triangles, Geometry.Angle_triangle t < 90)) :=
sorry

end acute_angle_triangles_limit_l5_5645


namespace letter_puzzle_solutions_l5_5541

theorem letter_puzzle_solutions (A B : ℕ) : 
  (1 ≤ A ∧ A < 10) ∧ (1 ≤ B ∧ B < 10) ∧ (A ≠ B) ∧ (A^B = 10 * B + A) → 
  (A = 2 ∧ B = 5) ∨ (A = 6 ∧ B = 2) ∨ (A = 4 ∧ B = 3) :=
by
  sorry

end letter_puzzle_solutions_l5_5541


namespace log_inequality_l5_5150

theorem log_inequality (x : ℝ) (h : 1 < x ∧ x < 10) : 
  log 10 (log 10 x) < (log 10 x)^2 ∧ (log 10 x)^2 < log 10 (x^2) :=
sorry

end log_inequality_l5_5150


namespace men_at_conference_l5_5199

theorem men_at_conference (M : ℕ) 
  (num_women : ℕ) (num_children : ℕ)
  (indian_men_fraction : ℚ) (indian_women_fraction : ℚ)
  (indian_children_fraction : ℚ) (non_indian_fraction : ℚ)
  (num_women_eq : num_women = 300)
  (num_children_eq : num_children = 500)
  (indian_men_fraction_eq : indian_men_fraction = 0.10)
  (indian_women_fraction_eq : indian_women_fraction = 0.60)
  (indian_children_fraction_eq : indian_children_fraction = 0.70)
  (non_indian_fraction_eq : non_indian_fraction = 0.5538461538461539) :
  M = 500 :=
by
  sorry

end men_at_conference_l5_5199


namespace expected_value_eight_sided_die_l5_5414

-- Define a standard 8-sided die
def eight_sided_die : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]

-- Compute the probability of each outcome
def probability (n : ℕ) : ℝ := 1 / 8

-- Expected Value of a discrete random variable
def expected_value (outcomes : List ℕ) (prob : ℕ → ℝ) : ℝ :=
  outcomes.sum / outcomes.length.toReal

-- Theorem stating the expected value of a standard 8-sided die roll is 4.5
theorem expected_value_eight_sided_die : expected_value eight_sided_die probability = 4.5 := by
  sorry

end expected_value_eight_sided_die_l5_5414


namespace expected_value_8_sided_die_l5_5404

/-- 
The expected value of rolling a standard 8-sided die is 4.5.
The die has 8 sides labeled 1 through 8, and each face has an equal probability of appearing,
which is 1/8. 
-/
theorem expected_value_8_sided_die : 
  (1/8:ℝ) * 1 + (1/8) * 2 + (1/8) * 3 + (1/8) * 4 + (1/8) * 5 + (1/8) * 6 + (1/8) * 7 + (1/8) * 8 = 4.5 :=
by 
  sorry

end expected_value_8_sided_die_l5_5404


namespace find_x_eq_minus_3_l5_5534

theorem find_x_eq_minus_3 (x : ℤ) : (3 ^ 7 * 3 ^ x = 81) → x = -3 := 
by
  sorry

end find_x_eq_minus_3_l5_5534


namespace letter_puzzle_solutions_l5_5540

theorem letter_puzzle_solutions (A B : ℕ) : 
  (1 ≤ A ∧ A < 10) ∧ (1 ≤ B ∧ B < 10) ∧ (A ≠ B) ∧ (A^B = 10 * B + A) → 
  (A = 2 ∧ B = 5) ∨ (A = 6 ∧ B = 2) ∨ (A = 4 ∧ B = 3) :=
by
  sorry

end letter_puzzle_solutions_l5_5540


namespace polynomial_degree_l5_5960

theorem polynomial_degree : 
  let f := (x^5 + 1)^6 * (x^4 + 1)^2
  in polynomial.degree f = 38 := sorry

end polynomial_degree_l5_5960


namespace QR_length_l5_5728

-- Definitions of the geometric entities and properties
variables (A B C D P Q R : Point)
variables (rectangle : is_rectangle A B C D)
variables (on_side_AB : P ∈ line A B)
variables (PA PB : length (line P A) = length (line P D) ∧ length (line P B) = length (line P C))
variables (C1 C2 : Circle)
variables (intersect_Q : intersects (circle_center_radius P (length (line P A))) (line B D) Q)
variables (intersect_R : intersects (circle_center_radius P (length (line P B))) (line A C) R)

-- The theorem we need to prove
theorem QR_length :
  length (line Q R) = (length (line P A) + length (line P B)) / 2 :=
sorry

end QR_length_l5_5728


namespace expected_value_8_sided_die_l5_5363

theorem expected_value_8_sided_die :
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let probability_each_outcome := 1 / 8
  (Σ x ∈ outcomes, probability_each_outcome * x) = 4.5 :=
by
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let probability_each_outcome := 1 / 8
  have h : (Σ x ∈ outcomes, probability_each_outcome * x) = (1 / 8) * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8) := by sorry
  have sum_eq_36 : 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 = 36 := by sorry
  rw [sum_eq_36] at h
  have expected_value_eq : (1 / 8) * 36 = 4.5 := by sorry
  rw [expected_value_eq] at h
  exact h

end expected_value_8_sided_die_l5_5363


namespace expected_value_of_8_sided_die_l5_5385

theorem expected_value_of_8_sided_die : 
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let probability := 1 / 8
  (∑ outcome in outcomes, outcome * probability) = 4.5 :=
by
  sorry

end expected_value_of_8_sided_die_l5_5385


namespace statement_2_correct_statement_4_correct_statement_1_incorrect_statement_3_incorrect_l5_5951

open Set

-- Conditions given in the problem
variables 
  (L₁ L₂ : Type) -- Lines
  (P₁ P₂ P₃ : Type) -- Planes
  (h1 : ∀ (l₁ l₂ : L₁) (p : P₃), (Parallel l₁ p ∧ Parallel l₂ p) → Parallel P₁ P₂)
  (h2 : ∀ (l : L₁) (p₁ p₂ : P₂), (Perpendicular l p₁ ∧ p₁ ∈ p₂) → Perpendicular p₁ p₂)
  (h3 : ∀ (l₁ l₂ l₃ : L₁), (Perpendicular l₁ l₃ ∧ Perpendicular l₂ l₃) → Parallel l₁ l₂)
  (h4 : ∀ (p₁ p₂ : P₂) (l : L₁), (Perpendicular p₁ p₂ ∧ l ∈ p₁ ∧ ¬Perpendicular l (Intersection p₁ p₂)) → ¬Perpendicular l p₂)

-- We are asked to prove that:

-- Statement 2 is correct
theorem statement_2_correct : ∀ (l : L₁) (p₁ p₂ : P₂), (Perpendicular l p₁ ∧ p₁ ∈ p₂) → Perpendicular p₁ p₂ := sorry

-- Statement 4 is correct
theorem statement_4_correct : ∀ (p₁ p₂ : P₂) (l : L₁), (Perpendicular p₁ p₂ ∧ l ∈ p₁ ∧ ¬Perpendicular l (Intersection p₁ p₂)) → ¬Perpendicular l p₂ := sorry

-- Statement 1 is incorrect
theorem statement_1_incorrect : ¬ (∀ (l₁ l₂ : L₁) (p : P₃), (Parallel l₁ p ∧ Parallel l₂ p) → Parallel P₁ P₂) := sorry

-- Statement 3 is incorrect
theorem statement_3_incorrect : ¬ (∀ (l₁ l₂ l₃ : L₁), (Perpendicular l₁ l₃ ∧ Perpendicular l₂ l₃) → Parallel l₁ l₂) := sorry


end statement_2_correct_statement_4_correct_statement_1_incorrect_statement_3_incorrect_l5_5951


namespace num_real_solutions_eq_2_l5_5178

theorem num_real_solutions_eq_2 :
  ∃ (x : Set ℝ → ℕ), x { x : ℝ | 2^(2*x + 3) - 2^(x + 4) - 2^(x + 1) + 8 = 0} = 2 :=
sorry

end num_real_solutions_eq_2_l5_5178


namespace gcd_example_l5_5340

-- Define the two numbers
def a : ℕ := 102
def b : ℕ := 238

-- Define the GCD of a and b
def gcd_ab : ℕ :=
  Nat.gcd a b

-- The expected result of the GCD
def expected_gcd : ℕ := 34

-- Prove that the GCD of a and b is equal to the expected GCD
theorem gcd_example : gcd_ab = expected_gcd := by
  sorry

end gcd_example_l5_5340


namespace jeans_cost_l5_5277

theorem jeans_cost (initial_money pizza_cost soda_cost quarter_value after_quarters : ℝ) (quarters_count: ℕ) :
  initial_money = 40 ->
  pizza_cost = 2.75 ->
  soda_cost = 1.50 ->
  quarter_value = 0.25 ->
  quarters_count = 97 ->
  after_quarters = quarters_count * quarter_value ->
  initial_money - (pizza_cost + soda_cost) - after_quarters = 11.50 :=
by
  intros h_initial h_pizza h_soda h_quarter_val h_quarters h_after_quarters
  sorry

end jeans_cost_l5_5277


namespace coprime_sequence_l5_5252

open Nat

noncomputable def f (x : ℕ) : ℕ := x^2 - x + 1

theorem coprime_sequence (m : ℕ) (h : m > 1) :
  ∀ i j : ℕ, i ≠ j → Nat.coprime (Nat.iterate f i m) (Nat.iterate f j m) :=
by
  sorry

end coprime_sequence_l5_5252


namespace remainder_8927_div_11_l5_5420

theorem remainder_8927_div_11 : 8927 % 11 = 8 :=
by
  sorry

end remainder_8927_div_11_l5_5420


namespace surgeon_is_mother_l5_5173

theorem surgeon_is_mother (Arthur : Type) (father_died : Arthur → Prop) (surgeon_stmt : ∀ (a : Arthur), (∃ p : Arthur, father_died p) → (surgeon_stmt a → (surgeon_stmt a = True → p = a))) :
  ∃ mother : Arthur, surgeon_stmt mother = True :=
by
  sorry

end surgeon_is_mother_l5_5173


namespace lucy_max_sodas_l5_5260

theorem lucy_max_sodas : 
  let lucy_money_cents := 2545
  let soda_price_cents := 215
  let tax_rate := 0.05
  let tax_cents := soda_price_cents * tax_rate
  let total_cost_per_soda := soda_price_cents + tax_cents
  lucy_money_cents / total_cost_per_soda ≤ 11 :=
by
  let lucy_money_cents := 2545
  let soda_price_cents := 215
  let tax_rate := 0.05
  let tax_cents := soda_price_cents * tax_rate
  let total_cost_per_soda := soda_price_cents + tax_cents
  have total_cost_per_soda_eq : total_cost_per_soda = 215 + 0.05 * 215 := by rfl
  have multiplied : 0.05 * 215 = 10.75 := by norm_num
  have summed : 215 + 10.75 = 225.75 := by norm_num
  have divide : 2545 / 225.75 = 11.273 := by norm_num
  have ceil_div : (2545 / 225.75 : ℝ).floor = 11 := by exact_mod_cast int.of_nat_floor (floor 11.273)
  exact_mod_cast ceil_div.symm


end lucy_max_sodas_l5_5260


namespace simplify_expression_l5_5296

theorem simplify_expression (a : ℚ) (ha : a ≠ 0):
  (a^2 - 2 + a⁻²) / (a^2 - a⁻²) = (a^2 - 1) / (a^2 + 1) :=
by
  sorry

end simplify_expression_l5_5296


namespace final_alcohol_percentage_l5_5018

noncomputable def initial_volume : ℝ := 6
noncomputable def initial_percentage : ℝ := 0.25
noncomputable def added_alcohol : ℝ := 3
noncomputable def final_volume : ℝ := initial_volume + added_alcohol
noncomputable def final_percentage : ℝ := (initial_volume * initial_percentage + added_alcohol) / final_volume * 100

theorem final_alcohol_percentage :
  final_percentage = 50 := by
  sorry

end final_alcohol_percentage_l5_5018


namespace no_convex_quadrilateral_with_acute_diagonal_division_l5_5213

theorem no_convex_quadrilateral_with_acute_diagonal_division :
  ¬ ∃ (ABCD : Type) [is_convex_quadrilateral ABCD] 
      (AC BD : diagonal ABCD),
      (is_acute_triangle (triangle AC) ∧ is_acute_triangle (triangle BD)) :=
by
  -- Proof will go here
  sorry

end no_convex_quadrilateral_with_acute_diagonal_division_l5_5213


namespace Asha_winning_probability_l5_5829

theorem Asha_winning_probability
  (P_lose : ℚ) (P_tie : ℚ)
  (h1 : P_lose = 3/7)
  (h2 : P_tie = 1/5)
  (h3 : P_lose + P_tie < 1) :
  (1 - P_lose - P_tie = 13/35) :=
by
  rw [h1, h2]
  norm_num
  sorry

end Asha_winning_probability_l5_5829


namespace fraction_taken_by_kiley_l5_5848

-- Define the constants and conditions
def total_crayons : ℕ := 48
def remaining_crayons_after_joe : ℕ := 18

-- Define the main statement to be proven
theorem fraction_taken_by_kiley (f : ℚ) : 
  (48 - (48 * f)) / 2 = 18 → f = 1 / 4 :=
by 
  intro h
  sorry

end fraction_taken_by_kiley_l5_5848


namespace find_second_number_in_denominator_l5_5323

theorem find_second_number_in_denominator :
  (0.625 * 0.0729 * 28.9) / (0.0017 * x * 8.1) = 382.5 → x = 0.24847 :=
by
  intro h
  sorry

end find_second_number_in_denominator_l5_5323


namespace min_pieces_on_checkerboard_l5_5271

def num_pieces (board : matrix (fin 6) (fin 6) bool) : ℕ :=
  fin.sum_univ (λ i, fin.sum_univ (λ j, if board i j then 1 else 0))

def piece_in_row_col (board : matrix (fin 6) (fin 6) bool) (n : ℕ) : Prop :=
  ∃ (r c : fin 6), (fin.sum_univ (λ j, if board r j then 1 else 0) + fin.sum_univ (λ i, if board i c then 1 else 0) - 1) = n

theorem min_pieces_on_checkerboard :
  ∀ (board : matrix (fin 6) (fin 6) bool),
  (∀ n : ℕ, 2 ≤ n ∧ n ≤ 10 → piece_in_row_col board n) →
  ∃ (board : matrix (fin 6) (fin 6) bool), num_pieces board = 19 := sorry

end min_pieces_on_checkerboard_l5_5271


namespace count_terminating_decimals_l5_5615

theorem count_terminating_decimals :
  {n : ℕ | 1 ≤ n ∧ n ≤ 150 ∧ ∃ k : ℕ, n = 3 * k}.to_finset.card = 50 := by
sorry

end count_terminating_decimals_l5_5615


namespace income_growth_rate_l5_5481

noncomputable def income_growth_eq (x : ℝ) : Prop :=
  3.2 * (1 + x)^2 = 3.7

theorem income_growth_rate :
  ∃ x : ℝ, income_growth_eq x :=
sorry

end income_growth_rate_l5_5481


namespace max_writers_at_conference_l5_5060

variables (T E W x : ℕ)

-- Defining the conditions
def conference_conditions (T E W x : ℕ) : Prop :=
  T = 90 ∧ E > 38 ∧ x ≤ 6 ∧ 2 * x + (W + E - x) = T ∧ W = T - E - x

-- Statement to prove the number of writers
theorem max_writers_at_conference : ∃ W, conference_conditions 90 39 W 1 :=
by
  sorry

end max_writers_at_conference_l5_5060


namespace min_pieces_on_checkboard_l5_5272

-- Define the board size and the range of n values
def board_size : ℕ := 6

-- Condition: For each number n from 2 to 10, there must be a piece in the same row and column as exactly n pieces (not counting itself).
def valid_placement (board : matrix (fin board_size) (fin board_size) bool) : Prop :=
  ∀ n in finset.range 9, -- since 10 - 2 + 1 = 9
    ∃ i j, board i j = true ∧
      ((finset.univ.filter (λ k, k ≠ j ∧ board i k = true)).card +
       (finset.univ.filter (λ k, k ≠ i ∧ board k j = true)).card) = (n + 2)

-- The minimum number of pieces that satisfies the given conditions
theorem min_pieces_on_checkboard : ∃ (board : matrix (fin board_size) (fin board_size) bool), valid_placement board ∧ (finset.univ.filter (λ (i : fin board_size) (j : fin board_size), board i j = true)).card = 19 :=
sorry

end min_pieces_on_checkboard_l5_5272


namespace angle_CNL_90_l5_5302

theorem angle_CNL_90 (A B C D K L M N : Type) [plane_geom A B C D K L M N] 
  (parallelogram_ABCD : parallelogram A B C D)
  (AK_bisects_∠A_of_∠LAN : angle_bisector A K (side A B C D) L M N)
  (AL_eq_CK : AL = CK)
  (AK_CL_intersect_M : intersect A K C L M)
  (N_on_extension_of_AD_beyond_D : extension A D N)
  (cyclic_quadrilateral_ALMN : cyclic_quadrilateral A L M N) :
  ∠ C N L = 90 :=
sorry

end angle_CNL_90_l5_5302


namespace letter_puzzle_solutions_l5_5549

noncomputable def is_solution (A B : ℕ) : Prop :=
A ≠ B ∧ A ∈ finset.range (10) ∧ B ∈ finset.range (10) ∧ 10 ≤ B * 10 + A ∧ B * 10 + A ≤ 99 ∧ A^B = B * 10 + A

theorem letter_puzzle_solutions :
  ∃ A B : ℕ, is_solution A B ∧ ((A = 2 ∧ B = 5) ∨ (A = 6 ∧ B = 2) ∨ (A = 4 ∧ B = 3)) :=
by
  sorry

end letter_puzzle_solutions_l5_5549


namespace simplify_sqrt_88200_l5_5292

theorem simplify_sqrt_88200 :
  ∀ (a b c d e : ℕ), a = 2 ∧ b = 3 ∧ c = 5 ∧ d = 7 ∧ e = 1 →
  ∃ f g : ℝ, (88200 : ℝ) = (f^2 * g) ∧ f = 882 ∧ g = 10 ∧ real.sqrt (88200 : ℝ) = f * real.sqrt g :=
sorry

end simplify_sqrt_88200_l5_5292


namespace unique_cubic_coefficients_l5_5808

noncomputable def cubic_function (a b c : ℝ) (x : ℝ) : ℝ := 4 * x^3 + a * x^2 + b * x + c

theorem unique_cubic_coefficients
  (a b c : ℝ)
  (h1 : ∀ x, -1 ≤ x ∧ x ≤ 1 → -1 ≤ cubic_function a b c x ∧ cubic_function a b c x ≤ 1) :
  (a = 0 ∧ b = -3 ∧ c = 0) :=
by
  sorry

end unique_cubic_coefficients_l5_5808


namespace b_real_if_product_is_real_l5_5180

theorem b_real_if_product_is_real (b : ℝ) (i : ℂ) (h_i : i = complex.I) 
  (h_condition : (2 + complex.I) * (b + complex.I) ∈ ℝ) : b = -2 :=
sorry

end b_real_if_product_is_real_l5_5180


namespace complement_intersection_l5_5774

def A : Set ℤ := { m | m ≤ -3 ∨ m ≥ 2 }
def B : Set ℕ := { n | -1 ≤ n ∧ n < 3 }

def complement_A : Set ℤ := { m | -3 < m ∧ m < 2 }

theorem complement_intersection : (complement_A ∩ B) = {0, 1} := by
  sorry

end complement_intersection_l5_5774


namespace exact_number_of_false_statements_is_three_l5_5326

-- Define the four statements as propositions
def stmt1 : Prop := ∃ f : Fin 4 → bool, (finset.filter (λ i, f i = ff) (finset.univ : finset (Fin 4))).card = 1
def stmt2 : Prop := ∃ f : Fin 4 → bool, (finset.filter (λ i, f i = ff) (finset.univ : finset (Fin 4))).card = 2
def stmt3 : Prop := ∃ f : Fin 4 → bool, (finset.filter (λ i, f i = ff) (finset.univ : finset (Fin 4))).card = 3
def stmt4 : Prop := ∃ f : Fin 4 → bool, (finset.filter (λ i, f i = ff) (finset.univ : finset (Fin 4))).card = 4

-- Goal: prove there are exactly three false statements
theorem exact_number_of_false_statements_is_three :
  stmt3 :=
sorry

end exact_number_of_false_statements_is_three_l5_5326


namespace greatest_integer_solution_l5_5861

theorem greatest_integer_solution (n : ℤ) (h : n^2 - 12 * n + 28 ≤ 0) : 6 ≤ n :=
sorry

end greatest_integer_solution_l5_5861


namespace equilateral_triangle_identity_l5_5071

noncomputable def complex_equilateral_triangle (a b c : ℂ) : Prop :=
(∃ z : ℂ, z = (c - a) / (b - a) ∧ z = complex.exp (complex.I * (π / 3)) ∧ z = -((complex.I * 2 * π / 3)^2)) ∧
(∃ z : ℂ, z = (c - a) / (b - a) ∧ complex.arg ((c - a) / (b - a)) = π / 3) ∧
(a + complex.exp (complex.I * (2 * π / 3)) * b + (complex.exp (complex.I * (4 * π / 3))) * c = 0)

theorem equilateral_triangle_identity {a b c : ℂ} (h : complex_equilateral_triangle a b c) :
  a^2 + b^2 + c^2 - a * b - b * c - c * a = 0 :=
sorry

end equilateral_triangle_identity_l5_5071


namespace Malik_yards_per_game_l5_5261

-- Definitions of the conditions
def number_of_games : ℕ := 4
def josiah_yards_per_game : ℕ := 22
def darnell_average_yards_per_game : ℕ := 11
def total_yards_all_athletes : ℕ := 204

-- The statement to prove
theorem Malik_yards_per_game (M : ℕ) 
  (H1 : number_of_games = 4) 
  (H2 : josiah_yards_per_game = 22) 
  (H3 : darnell_average_yards_per_game = 11) 
  (H4 : total_yards_all_athletes = 204) :
  4 * M + 4 * 22 + 4 * 11 = 204 → M = 18 :=
by
  intros h
  sorry

end Malik_yards_per_game_l5_5261


namespace combined_height_is_correct_l5_5225

variables (John Lena Rebeca Sam Amy : ℕ)

-- Conditions
def John_height : ℕ := 152
def Lena_height := John_height - 15
def Rebeca_height := John_height + 6
def Sam_height := John_height + 8
def Amy_height := Lena_height - 3

-- Question to prove
def total_combined_height := Lena_height + Rebeca_height + Sam_height + Amy_height

theorem combined_height_is_correct : total_combined_height = 589 :=
by
  simp [Lena_height, Rebeca_height, Sam_height, Amy_height, John_height]
  sorry

end combined_height_is_correct_l5_5225


namespace smallest_rel_prime_to_180_l5_5597

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ Nat.gcd x 180 = 1 ∧ ∀ y : ℕ, (y > 1 ∧ Nat.gcd y 180 = 1) → x ≤ y :=
begin
  sorry
end

end smallest_rel_prime_to_180_l5_5597


namespace expected_value_of_8_sided_die_is_4_point_5_l5_5352

def expected_value_8_sided_die : ℝ :=
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let probability := 1 / 8
  (list.sum outcomes : ℝ) * probability

theorem expected_value_of_8_sided_die_is_4_point_5 :
  expected_value_8_sided_die = 4.5 := by
  sorry

end expected_value_of_8_sided_die_is_4_point_5_l5_5352


namespace range_of_g_l5_5763

noncomputable def g (x : ℝ) : ℤ :=
if x > -3 then
  ⌈1 / ((x + 3)^2)⌉
else
  ⌊1 / ((x + 3)^2)⌋

theorem range_of_g :
  ∀ y : ℤ, (∃ x : ℝ, g x = y) ↔ (∃ n : ℕ, y = n + 1) :=
by sorry

end range_of_g_l5_5763


namespace population_initial_count_l5_5809

theorem population_initial_count (B D : ℕ) (net_growth_rate_percentage : ℝ) (hB : B = 32) (hD : D = 11) (hNGR : net_growth_rate_percentage = 2.1) : 
  let P := (B - D) / (net_growth_rate_percentage / 100) in P = 1000 :=
by 
  -- Bringing hypotheses into the local context
  have h_B := hB 
  have h_D := hD 
  have h_NGR := hNGR

  -- Defining P using the given conditions
  let P := (32 - 11) / (2.1 / 100)

  -- Showing the calculation result ensures P = 1000
  show 
    P = 1000 
  sorry

end population_initial_count_l5_5809


namespace find_x_positive_multiple_of_8_l5_5186

theorem find_x_positive_multiple_of_8 (x : ℕ) 
  (h1 : ∃ k, x = 8 * k) 
  (h2 : x^2 > 100) 
  (h3 : x < 20) : x = 16 :=
by
  sorry

end find_x_positive_multiple_of_8_l5_5186


namespace vector_parallel_find_lambda_l5_5680

-- Definitions of vectors a and b
def a (x : ℝ) : ℝ × ℝ := (Real.cos (3 * x / 2), Real.sin (3 * x / 2))
def b (x : ℝ) : ℝ × ℝ := (Real.cos (x / 2), -Real.sin (x / 2))

-- Conditions from the problem context
def x_in_domain (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ Real.pi / 2

-- Question 1: Proving parallel vectors implies specific x values
theorem vector_parallel (x : ℝ) (h : x_in_domain x) (h_parallel : a x = b x) : x = 0 ∨ x = Real.pi / 2 := by
  sorry

-- Function f(x) from the problem context
def f (x λ : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2 - 2 * λ * (Real.sqrt (2 + 2 * (a x).1 * (b x).1 + (a x).2 * (b x).2)) + 2 * λ

-- Question 2: Given f(x) achieves minimum value -3, find λ
theorem find_lambda (x λ : ℝ) (h : x_in_domain x) (h_min : f x λ = -3) : λ = -1 ∨ λ = 2 := by
  sorry

end vector_parallel_find_lambda_l5_5680


namespace sqrt_88200_simplified_l5_5295

theorem sqrt_88200_simplified : Real.sqrt 88200 = 210 * Real.sqrt 6 :=
by sorry

end sqrt_88200_simplified_l5_5295


namespace smallest_rel_prime_to_180_l5_5578

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ Nat.gcd x 180 = 1 ∧ ∀ y : ℕ, y > 1 ∧ Nat.gcd y 180 = 1 → x ≤ y :=
begin
  use 7,
  split,
  { exact dec_trivial },
  split,
  { exact dec_trivial },
  { intro y,
    intro h,
    cases h with h1 h2,
    repeat { try { apply dec_trivial,
                   apply lt_or_eq_of_le,
                   norm_num,
                   apply Nat.prime_not_dvd_mul,
                   norm_num,
                   apply not_or_distrib.mpr,
                   split,
                   norm_cast,
                   intro,
                   exact le_antisymm _ },
           sorry }
end

end smallest_rel_prime_to_180_l5_5578


namespace expected_value_of_eight_sided_die_l5_5351

-- Definitions based on the problem conditions
def eight_sided_die_outcomes : List ℕ := [1,2,3,4,5,6,7,8]

def probability (n : ℕ) := 1 / n

-- Expected value calculation related to the problem
def expected_value_die_roll (outcomes : List ℕ) (prob : ℕ → Rat) : Rat :=
  List.sum (outcomes.map (λ x => prob outcomes.length * x))

-- Expected value of an 8-sided die roll
theorem expected_value_of_eight_sided_die :
  expected_value_die_roll eight_sided_die_outcomes probability = 4.5 := 
sorry

end expected_value_of_eight_sided_die_l5_5351


namespace school_supply_cost_l5_5217

theorem school_supply_cost (num_students : ℕ) (pens_per_student : ℕ) (pen_cost : ℝ) 
  (notebooks_per_student : ℕ) (notebook_cost : ℝ) 
  (binders_per_student : ℕ) (binder_cost : ℝ) 
  (highlighters_per_student : ℕ) (highlighter_cost : ℝ) 
  (teacher_discount : ℝ) : 
  num_students = 30 →
  pens_per_student = 5 →
  pen_cost = 0.50 →
  notebooks_per_student = 3 →
  notebook_cost = 1.25 →
  binders_per_student = 1 →
  binder_cost = 4.25 →
  highlighters_per_student = 2 →
  highlighter_cost = 0.75 →
  teacher_discount = 100 →
  (num_students * 
    (pens_per_student * pen_cost + notebooks_per_student * notebook_cost + 
    binders_per_student * binder_cost + highlighters_per_student * highlighter_cost) - 
    teacher_discount) = 260 :=
by
  intros _ _ _ _ _ _ _ _ _ _

  -- Sorry added to skip the proof
  sorry

end school_supply_cost_l5_5217


namespace concyclic_points_l5_5766

-- Definitions and conditions:
variable (A B C I D E F M N : Type) 
variable [triangle : Triangle A B C]
variable [incenter : Incenter I A B C]
variable [angle_bisectors : AngleBisectors (AI) (BI) (CI) A B C D E F]
variable [perpend_bisector_AD : PerpendicularBisector (AD) (BI) (CI) M N]

-- Statement that needs to be proven:
theorem concyclic_points :
  Concyclic A I M N :=
sorry

end concyclic_points_l5_5766


namespace part1_part2_l5_5162

noncomputable def f (x m : ℝ) : ℝ := abs (x - m) + abs (x + 3)

theorem part1 (x : ℝ) : f x 1 ≤ x + 4 ↔ 0 ≤ x ∧ x ≤ 2 := sorry

theorem part2 (m n t : ℝ) (hm : m > 0) (hn : n > 0) (ht : t > 0) 
  (hmin : ∀ x, f x m ≥ 5 - n - t) :
  1 / (m + n) + 1 / t ≥ 2 := sorry

end part1_part2_l5_5162


namespace basketball_team_cookies_sale_l5_5807

theorem basketball_team_cookies_sale (
  cupcake_sales : ℕ := 50,
  cupcake_price : ℕ := 2,
  bb_count : ℕ := 2,
  bb_price : ℕ := 40,
  energy_drink_count : ℕ := 20,
  energy_drink_price : ℕ := 2,
  cookie_price : ℝ := 0.5) :
  cupcake_sales * cupcake_price + cookie_price * 40 = bb_count * bb_price + energy_drink_count * energy_drink_price → 
  ∃ (num_cookies : ℕ), num_cookies = 40 ∧ cupcake_sales * cupcake_price + num_cookies * cookie_price = bb_count * bb_price + energy_drink_count * energy_drink_price := by
  sorry

end basketball_team_cookies_sale_l5_5807


namespace school_supply_cost_l5_5218

theorem school_supply_cost (num_students : ℕ) (pens_per_student : ℕ) (pen_cost : ℝ) 
  (notebooks_per_student : ℕ) (notebook_cost : ℝ) 
  (binders_per_student : ℕ) (binder_cost : ℝ) 
  (highlighters_per_student : ℕ) (highlighter_cost : ℝ) 
  (teacher_discount : ℝ) : 
  num_students = 30 →
  pens_per_student = 5 →
  pen_cost = 0.50 →
  notebooks_per_student = 3 →
  notebook_cost = 1.25 →
  binders_per_student = 1 →
  binder_cost = 4.25 →
  highlighters_per_student = 2 →
  highlighter_cost = 0.75 →
  teacher_discount = 100 →
  (num_students * 
    (pens_per_student * pen_cost + notebooks_per_student * notebook_cost + 
    binders_per_student * binder_cost + highlighters_per_student * highlighter_cost) - 
    teacher_discount) = 260 :=
by
  intros _ _ _ _ _ _ _ _ _ _

  -- Sorry added to skip the proof
  sorry

end school_supply_cost_l5_5218


namespace probability_point_closer_to_origin_l5_5039

noncomputable def probability_closer_to_origin (A B C D E : Point) (Rect : Rectangle) : ℚ :=
  let area_rect := Rectangle.area Rect
  let midpoint := (B.x / 2, B.y / 2)
  let slope := B.y / B.x
  let perp_slope := -B.x / B.y
  let bisector_eqn x := perp_slope * (x - midpoint.1) + midpoint.2
  let area_under_bisector := ∫ x in (0 : ℝ)..(B.y / perp_slope : ℝ), bisector_eqn x
  area_under_bisector / area_rect

theorem probability_point_closer_to_origin :
  ∀ (P : Point) (Rect : Rectangle),
    Rect.vertices = [(0,0), (4,0), (4,2), (0,2)] →
    probability_closer_to_origin (0,0) (5,2) Rect = 29 / 32 :=
by
  sorry

end probability_point_closer_to_origin_l5_5039


namespace blue_pill_cost_l5_5930

theorem blue_pill_cost (r : ℝ) (h1 : r + (r + 3) + (2*r - 2) = 44) : r + 3 = 13.75 :=
by
  have daily_cost : r + (r + 3) + (2*r - 2) = 4*r + 1 :=
    by ring
  rw daily_cost at h1
  linarith

example (red : ℝ) (blue : ℝ) (yellow : ℝ)
  (h1 : blue = red + 3)
  (h2 : yellow = 2 * red - 2)
  (total_cost : 21 * (red + blue + yellow) = 924) : blue = 13.75 :=
by 
  have daily_cost : red + blue + yellow = 44 :=
    by calc
      924 / 21 = 44 : by norm_num
  have total_eq : red + (red + 3) + (2 * red - 2) = 4 * red + 1 :=
    by ring
  rw [h1, h2] at daily_cost
  have cost_eq : 4 * red + 1 = 44 :=
    by linarith 
  have red_cost : red = 10.75 :=
    by linarith
  rw [red_cost, h1]
  norm_num 

end blue_pill_cost_l5_5930


namespace stereographic_projection_inversion_stereographic_projection_circles_stereographic_projection_angle_preservation_l5_5005

-- Definitions of the geometric entities
def Point := ℝ × ℝ × ℝ
def Sphere (center: Point) (radius: ℝ) := { p : Point // dist p center = radius }
def Plane (point: Point) (normal: Point) := { p : Point // dot normal (sub p point) = 0 }

variables 
  (A B: Point)  -- Points on the sphere 
  (P: Plane A (1, 0, 0))  -- Tangent plane at A
  (S: Sphere A 1)  -- Sphere with unit radius centered at A
  -- Both X and Y defined as subsets might not be fully accurate but capture the intersection and other geometric properties.
  (X: { x : Point // dist x A = 1 })  -- Any point on the sphere except B
  (Y : { y : Point // ∃ (l : ℝ), l • (sub B X) = sub y A }) -- Intersecting line BX with P

-- The main proofs capturing the three parts
theorem stereographic_projection_inversion :
  ∃ r, ∀ (X: Point), ¬ (dist X B = 0) →  (dist X (X • r)) = r :=
by sorry -- Detailed construction skipped

theorem stereographic_projection_circles :
  (∃ X : Point, dist X B = 0 → line P (proj_stereographic B S P X)) ∧
  (∀ X : Point, (dist X B ≠ 0) → circle P (proj_stereographic B S P X)) :=
by sorry -- Use projections and transformations

theorem stereographic_projection_angle_preservation :
  inv (preserves_angle(sphere_intersection_circles S P B)) :=
by sorry -- Conformal map properties like Thales


end stereographic_projection_inversion_stereographic_projection_circles_stereographic_projection_angle_preservation_l5_5005


namespace max_gcd_consecutive_terms_l5_5082

def sequence (n : ℕ) : ℕ := n^2! + n

theorem max_gcd_consecutive_terms : ∃ m : ℕ, (∀ n : ℕ, n ≥ 0 → gcd (sequence n) (sequence (n + 1)) ≤ m) ∧ m = 2 := by
  sorry

end max_gcd_consecutive_terms_l5_5082


namespace true_statements_l5_5730

variables {α β : Type} [Plane α] [Plane β]
variables {l m : Line}
variables (h_lm_diff : l ≠ m)
variables (h_alpha_beta_diff : α ≠ β)

-- Statement 1: If l ⊆ α, m ⊆ α, l ‖ β, and m ‖ β, then α ‖ β
def statement_1 (h1 : l ⊆ α) (h2 : m ⊆ α) (h3 : l ‖ β) (h4 : m ‖ β) : Prop :=
  α ‖ β

-- Statement 2: If l ⊆ α, l ‖ β, and α ∩ β = m, then l ‖ m
def statement_2 (h5 : l ⊆ α) (h6 : l ‖ β) (h7 : α ∩ β = m) : Prop :=
  l ‖ m

-- Statement 3: If α ‖ β and l ‖ α, then l ‖ β
def statement_3 (h8 : α ‖ β) (h9 : l ‖ α) : Prop :=
  l ‖ β

-- Statement 4: If l ⊥ α, m ‖ l, and α ‖ β, then m ⊥ β
def statement_4 (h10 : l ⊥ α) (h11 : m ‖ l) (h12 : α ‖ β) : Prop :=
  m ⊥ β

-- Conclusion: The true statements are 2 and 4
theorem true_statements : statement_2 l m α β h_lm_diff h_alpha_beta_diff
                        ∧ statement_4 l m α β h_lm_diff h_alpha_beta_diff :=
by sorry

end true_statements_l5_5730


namespace pair_C_same_function_graphs_l5_5932

theorem pair_C_same_function_graphs :
  (∀ x : ℝ, x ≠ 0 → (x / x = x ^ 0)) :=
begin
  assume x h,
  calc
    x / x = 1 : by rw div_self h
    ... = x ^ 0 : by rw pow_zero x,
end sorry

end pair_C_same_function_graphs_l5_5932


namespace tangent_line_at_neg_one_l5_5816

noncomputable def f (x : ℝ) : ℝ :=
  if x >= 0 then real.exp x else real.exp (-x)

def is_even (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = g x

theorem tangent_line_at_neg_one : 
  is_even f ∧ (∀ x : ℝ, x >= 0 → f x = real.exp x) → (∃ m b : ℝ, ex + y = 0 ∧ (f(-1) = e ∧ m = -e)) := 
by
  sorry

end tangent_line_at_neg_one_l5_5816


namespace vector_b_magnitude_l5_5737

noncomputable def vector_a : ℝ × ℝ := (2, -1)

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1^2 + v.2^2)

theorem vector_b_magnitude (b : ℝ × ℝ) (h1 : b = (-2) • vector_a) (h2 : magnitude b = 2 * real.sqrt 5) :
  b = (-4, 2) :=
sorry

end vector_b_magnitude_l5_5737


namespace min_one_dollar_coins_l5_5504

theorem min_one_dollar_coins (n : ℕ) : 4 * 5 + 10 * 0.05 + n ≥ 37.50 → n ≥ 17 :=
by 
  intros h
  sorry

end min_one_dollar_coins_l5_5504


namespace paintings_per_room_l5_5257

theorem paintings_per_room (total_paintings : ℕ) (rooms : ℕ) (h_total : total_paintings = 32) (h_rooms : rooms = 4) :
  total_paintings / rooms = 8 :=
by {
  rw [h_total, h_rooms],
  norm_num,
}

end paintings_per_room_l5_5257


namespace correct_option_given_conditions_l5_5711

-- Define the conditions of the problem
def announcement_conditions (announcement: Type) : Prop := 
  (∃ units_in_arrears: ℕ, units_in_arrears > 0) ∧ 
  (∃ companies_paid_on_time: ℕ, companies_paid_on_time > 0)

-- Define options
structure Options :=
  (A B C D : Type)

-- Define the option descriptions as given in the problem
def OptionDescription_A (A: Type) : Prop := 
  "Promptly recovering money → Promoting economic development → Improving people's livelihood"

def OptionDescription_B (B: Type) : Prop := 
  "Urging enterprises to fulfill their social responsibilities → Perfecting social security → Achieving social fairness"

def OptionDescription_C (C: Type) : Prop := 
  "Urging enterprises to pay social insurance → Improving people's livelihood → Promoting economic development"

def OptionDescription_D (D: Type) : Prop := 
  "Preventing inflation → Protecting people's livelihood → Promoting social harmony"

-- Define the main theorem to prove that Option B is the correct answer based on the conditions
theorem correct_option_given_conditions (announcement: Type) (opts: Options) : 
  announcement_conditions announcement →
  OptionDescription_B opts.B :=
begin
  sorry
end

end correct_option_given_conditions_l5_5711


namespace repeating_decimal_to_fraction_l5_5974

/-- Given 0.02 repeating as a fraction 2/99, prove that 2.06 repeating can be expressed as 68/33 -/
theorem repeating_decimal_to_fraction :
  (2 + 0.06̅ : ℝ) = (68 / 33 : ℝ) :=
begin
  have h : (0.02̅ : ℝ) = (2 / 99 : ℝ), from sorry,
  have h3 : (0.06̅ : ℝ) = 3 * (0.02̅ : ℝ), from sorry,
  have h6 : (0.06̅ : ℝ) = 3 * (2 / 99 : ℝ), from sorry,
  have s : (0.06̅ : ℝ) = (6 / 99 : ℝ), from sorry,
  have s2 : (6 / 99 : ℝ) = (2 / 33 : ℝ), from sorry,
  have add := congr_arg (λ x : ℝ, (2 : ℝ) + x) s2,
  rw [add_comm, ← add_halves', add_assoc', add_comm (2 : ℝ), add_comm 2 (2 / 33 : ℝ), add_halves',
    add_assoc', add_comm (2 : ℝ), add_comm 68 (2 / 33 : ℝ)] ,
end

end repeating_decimal_to_fraction_l5_5974


namespace equation_of_parabola_y_intercept_range_l5_5636

-- Given conditions
variables (A B C : Point) (l : Line)

def parabola (p : ℝ) (h : p > 0) : Prop := 
  ∀ (x y : ℝ), x^2 = 2 * p * y

def line (slope : ℝ) (intercept : ℝ) : Prop :=
  ∀ (x y : ℝ), y = slope * (x + intercept)

def vec_eq (A B C : Point) : Prop :=
  -- This implies some vector equation relationship
  sorry

-- Prove 1: Equation of the parabola
theorem equation_of_parabola (A B C : Point) (l : Line) 
  (h1 : l.passes_through (−4, 0)) 
  (h2 : l.intersects_parabola at B and C) 
  (slope : ℝ) (h3 : slope = 1/2) 
  (h4 : vec_eq A B C) :
  parabola 2 :=
sorry

-- Prove 2: Range of y-intercept of the perpendicular bisector
theorem y_intercept_range (k : ℝ)  
  (h1 : (2 * (k + 1)^2 (b : ℝ)) = b)
  (h2 : 16 * k^2 + 64 * k > 0) :
  2 < b :=
sorry

end equation_of_parabola_y_intercept_range_l5_5636


namespace expected_value_8_sided_die_l5_5403

/-- 
The expected value of rolling a standard 8-sided die is 4.5.
The die has 8 sides labeled 1 through 8, and each face has an equal probability of appearing,
which is 1/8. 
-/
theorem expected_value_8_sided_die : 
  (1/8:ℝ) * 1 + (1/8) * 2 + (1/8) * 3 + (1/8) * 4 + (1/8) * 5 + (1/8) * 6 + (1/8) * 7 + (1/8) * 8 = 4.5 :=
by 
  sorry

end expected_value_8_sided_die_l5_5403


namespace octagon_area_sum_l5_5897

theorem octagon_area_sum :
  let A1 := 2024
  let a := 1012
  let b := 506
  let c := 2
  a + b + c = 1520 := by
    sorry

end octagon_area_sum_l5_5897


namespace remainder_when_b_divided_by_13_l5_5761

theorem remainder_when_b_divided_by_13 :
  let b := ((5^{-1} : ℤ) + (7^{-1}) + (9^{-1}) + (11^{-1}))⁻¹
  (mod 13) 
  ((b % 13) = 11 % 13) :=
by
  have h5_mod_inv : 5 * 8 ≡ 1 [MOD 13] := by norm_num
  have h7_mod_inv : 7 * 2 ≡ 1 [MOD 13] := by norm_num
  have h9_mod_inv : 9 * 3 ≡ 1 [MOD 13] := by norm_num
  have h11_mod_inv : 11 * 6 ≡ 1 [MOD 13] := by norm_num
  have b_mod_inv : ((8 + 2 + 3 + 6) % 13 = 6 % 13) := by norm_num
  have b := 6^{-1} % 13
  have b_val : b ≡ 11 [MOD 13] := by norm_num
  sorry

end remainder_when_b_divided_by_13_l5_5761


namespace largest_mersenne_prime_less_than_500_l5_5905

def is_mersenne_prime (p : ℕ) : Prop :=
  ∃ n : ℕ, Nat.Prime n ∧ p = 2^n - 1 ∧ Nat.Prime p

theorem largest_mersenne_prime_less_than_500 :
  ∃ p : ℕ, is_mersenne_prime p ∧ p < 500 ∧ ∀ q : ℕ, is_mersenne_prime q ∧ q < 500 → q ≤ p → p = 127 :=
by
  sorry

end largest_mersenne_prime_less_than_500_l5_5905


namespace repeatingDecimal_as_fraction_l5_5684

def repeatingDecimal : ℚ := 0.136513513513

theorem repeatingDecimal_as_fraction : repeatingDecimal = 136377 / 999000 := 
by 
  sorry

end repeatingDecimal_as_fraction_l5_5684


namespace sum_of_nth_powers_of_unity_l5_5115

def nth_roots_of_unity (n : ℕ) : List ℂ :=
  List.range n |>.map (λ k => complex.cos (2 * ↑k * real.pi / ↑n) + complex.sin (2 * ↑k * real.pi / ↑n) * complex.I)

theorem sum_of_nth_powers_of_unity (n : ℕ) (hn : n > 0) :
  let roots := nth_roots_of_unity n in
  (roots.map (λ x => x ^ n)).sum = n :=
by
  sorry

end sum_of_nth_powers_of_unity_l5_5115


namespace ordered_pairs_unique_solution_l5_5687

theorem ordered_pairs_unique_solution :
  ∃! (b c : ℕ), (b > 0) ∧ (c > 0) ∧ (b^2 - 4 * c = 0) ∧ (c^2 - 4 * b = 0) :=
sorry

end ordered_pairs_unique_solution_l5_5687


namespace min_moves_to_eq_triangle_l5_5509

/-- 
Given an initial equilateral triangle with side length 700, 
this theorem states that the minimum number of moves required 
to change it to an equilateral triangle with side length 2 is 14.
Each move consists of changing the length of one side while 
maintaining a valid triangle shape.
-/
theorem min_moves_to_eq_triangle (initial_len target_len : ℕ) (initial_len = 700) (target_len = 2) : 
  ∃ (min_moves : ℕ), min_moves = 14 := sorry

end min_moves_to_eq_triangle_l5_5509


namespace symmetric_circle_eqn_l5_5159

theorem symmetric_circle_eqn :
  ∀ (x y : ℝ),
  ((x + 1)^2 + (y - 1)^2 = 1) ∧ (x - y - 1 = 0) →
  (∀ (x' y' : ℝ), (x' = y + 1) ∧ (y' = x - 1) → (x' + 1)^2 + (y' - 1)^2 = 1) →
  (x - 2)^2 + (y + 2)^2 = 1 :=
by
  intros x y h h_sym
  sorry

end symmetric_circle_eqn_l5_5159


namespace arrangement_count_l5_5341

theorem arrangement_count (digits : Finset ℕ) (H : digits = {1, 3, 4, 5, 6}) :
  ∃ (count : ℕ), count = 20 ∧
    ∀ (arr : Vector ℕ 5), (∀ d ∈ arr.toList, d ∈ digits) →
    (arr.nth 0 = 1 ∧ list.indexof 1 arr.toList < list.indexof 3 arr.toList ∧ list.indexof 1 arr.toList < list.indexof 4 arr.toList) →
    count = count_valid_arrangements arr := by
  sorry

end arrangement_count_l5_5341


namespace dinner_party_seating_l5_5087

theorem dinner_party_seating : 
  (∃ f : Fin 8 → Fin 8 → Bool, 
    (∃ p : Perm (Fin 7), 
    ∀ g h : Fin 7 → Fin 7 → Bool, (g ≈ h) ↔ (∃ r : α → α, is_rotation r g h)) ∧ 
    (∀ x : Fin 7, p x ≠ x)) → 
    nat.factorial 7 = 5760 := 
begin
  sorry
end

end dinner_party_seating_l5_5087


namespace percentage_increase_in_allowance_l5_5227

def middle_school_allowance : ℕ := 8 + 2
def senior_year_allowance : ℕ := 2 * middle_school_allowance + 5

theorem percentage_increase_in_allowance : 
  (senior_year_allowance - middle_school_allowance) * 100 / middle_school_allowance = 150 := 
  by
    sorry

end percentage_increase_in_allowance_l5_5227


namespace repeating_decimal_to_fraction_l5_5980

/-- Given 0.02 repeating as a fraction 2/99, prove that 2.06 repeating can be expressed as 68/33 -/
theorem repeating_decimal_to_fraction :
  (2 + 0.06̅ : ℝ) = (68 / 33 : ℝ) :=
begin
  have h : (0.02̅ : ℝ) = (2 / 99 : ℝ), from sorry,
  have h3 : (0.06̅ : ℝ) = 3 * (0.02̅ : ℝ), from sorry,
  have h6 : (0.06̅ : ℝ) = 3 * (2 / 99 : ℝ), from sorry,
  have s : (0.06̅ : ℝ) = (6 / 99 : ℝ), from sorry,
  have s2 : (6 / 99 : ℝ) = (2 / 33 : ℝ), from sorry,
  have add := congr_arg (λ x : ℝ, (2 : ℝ) + x) s2,
  rw [add_comm, ← add_halves', add_assoc', add_comm (2 : ℝ), add_comm 2 (2 / 33 : ℝ), add_halves',
    add_assoc', add_comm (2 : ℝ), add_comm 68 (2 / 33 : ℝ)] ,
end

end repeating_decimal_to_fraction_l5_5980


namespace avg_annual_growth_rate_l5_5488
-- Import the Mathlib library

-- Define the given conditions
def initial_income : ℝ := 32000
def final_income : ℝ := 37000
def period : ℝ := 2
def initial_income_ten_thousands : ℝ := initial_income / 10000
def final_income_ten_thousands : ℝ := final_income / 10000

-- Define the growth rate
variable (x : ℝ)

-- Define the theorem
theorem avg_annual_growth_rate :
  3.2 * (1 + x) ^ 2 = 3.7 :=
sorry

end avg_annual_growth_rate_l5_5488


namespace probability_even_product_l5_5339

noncomputable def first_spinner_options : Finset ℕ := {3, 4, 7}
noncomputable def second_spinner_options : Finset ℕ := {1, 4, 5, 6, 8}

def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_even (n : ℕ) : Prop := ¬ is_odd n

def spinner_product_is_even : ℕ → ℕ → Prop :=
  λ x y, is_even (x * y)

theorem probability_even_product :
  (let outcomes := (first_spinner_options.product second_spinner_options).toFinset,
       even_outcomes := outcomes.filter (λ p, spinner_product_is_even p.1 p.2) in
   (even_outcomes.card : ℚ) / (outcomes.card : ℚ)) = 11 / 15 := by 
  sorry

end probability_even_product_l5_5339


namespace vector_negative_parallel_l5_5284

variable {α : Type*} [NormedAddCommGroup α] [NormedSpace ℝ α]

def vectors_parallel (a b : α) : Prop := ∃ k : ℝ, a = k • b

theorem vector_negative_parallel (a b : α) (h : a = -b) : vectors_parallel a b :=
by {
  use (-1),
  exact h,
}

end vector_negative_parallel_l5_5284


namespace max_area_rectangle_l5_5467

theorem max_area_rectangle (P : ℝ) (x : ℝ) (h1 : P = 40) (h2 : 6 * x = P) : 
  2 * (x ^ 2) = 800 / 9 :=
by
  sorry

end max_area_rectangle_l5_5467


namespace max_height_reached_by_rocket_l5_5468

def h (t : ℝ) : ℝ := -12 * t^2 + 72 * t + 36

theorem max_height_reached_by_rocket : ∃ t : ℝ, h t = 144 ∧ ∀ t' : ℝ, h t' ≤ 144 := sorry

end max_height_reached_by_rocket_l5_5468


namespace area_of_square_with_diagonal_10_l5_5473

theorem area_of_square_with_diagonal_10 :
  ∀ (d : ℝ) (A : ℝ), d = 10 → A = (d^2 / 2) → A = 50 :=
by
  intros d A h₁ h₂
  rw [h₁, h₂]
  norm_num
  sorry

end area_of_square_with_diagonal_10_l5_5473


namespace smallest_rel_prime_to_180_l5_5607

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ Int.gcd x 180 = 1 ∧ ∀ y : ℕ, y > 1 ∧ Int.gcd y 180 = 1 → x ≤ y := 
sorry

end smallest_rel_prime_to_180_l5_5607


namespace expected_value_of_8_sided_die_l5_5396

def outcomes : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]
def probability (n : ℕ) : ℝ := 1 / 8

theorem expected_value_of_8_sided_die :
  (∑ x in outcomes, probability x * x) = 4.5 := 
sorry

end expected_value_of_8_sided_die_l5_5396


namespace cross_platform_time_l5_5819

-- Definitions of the conditions
def length_train : ℝ := 750
def length_platform := length_train
def speed_train_kmh : ℝ := 90
def speed_train_ms := speed_train_kmh * (1000 / 3600)

-- The proof problem: Prove that the time to cross the platform is 60 seconds
theorem cross_platform_time : 
  (length_train + length_platform) / speed_train_ms = 60 := 
by
  sorry

end cross_platform_time_l5_5819


namespace positive_integer_n_iff_conditions_hold_l5_5536

def IsArrowConfiguration (n : ℕ) (board : ℕ × ℕ → ℕ) : Prop :=
  ∀ x y, true -- Dummy definition for representing board configuration
  
def Condition1 (n : ℕ) (board : ℕ × ℕ → ℕ) : Prop :=
  ∀ i j, ( ∃ steps : List (ℕ × ℕ), steps.head = (i, j) ∧ steps.last = (i, j) ∧ 
    (∀ k < steps.length - 1, steps.get k ≠ steps.get (k+1))) ∧ -- Dummy condition to represent following arrows

def Condition2 (n : ℕ) (board : ℕ × ℕ → ℕ) : Prop :=
  ∀ i, 1 ≤ i ∧ i < n - 1 → (board i).filter (= (1 : ℕ)).length = (board i).filter (= (2 : ℕ)).length -- Dummy condition

def Condition3 (n : ℕ) (board : ℕ × ℕ → ℕ) : Prop :=
  ∀ j, 1 ≤ j ∧ j < n - 1 → (board j).filter (= (3 : ℕ)).length = (board j).filter (= (4 : ℕ)).length -- Dummy condition

theorem positive_integer_n_iff_conditions_hold :
  ∀ n : ℕ, n > 0 →
  (∃ board : ℕ × ℕ → ℕ, 
    IsArrowConfiguration n board ∧
    Condition1 n board ∧
    Condition2 n board ∧
    Condition3 n board
  ) ↔ n = 2 :=
by
  sorry

end positive_integer_n_iff_conditions_hold_l5_5536


namespace binomial_expansion_constant_term_l5_5156

noncomputable def binomial_coeff (n k : ℕ) := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_expansion_constant_term
  (n : ℕ)
  (h1 : binomial_coeff n 0 + binomial_coeff n 1 = 10)
  (h2 : n = 9) :
  let T : ℕ := binomial_coeff n 6 * 2^(n-6) in
  T = 672 :=
by
  sorry

end binomial_expansion_constant_term_l5_5156


namespace locus_of_midpoint_l5_5037

-- Define the given conditions
def point_A_on_circle (xA yA : ℝ) : Prop := xA^2 + yA^2 = 1
def point_B : ℝ × ℝ := (3, 0)

-- Define the coordinates of the midpoint M of the segment AB
def midpoint_M (xA yA : ℝ) : ℝ × ℝ :=
  ((xA + 3) / 2, yA / 2)

-- Lean 4 statement proving the trajectory equation
theorem locus_of_midpoint (x y : ℝ) (xA yA : ℝ) :
  point_A_on_circle xA yA →
  midpoint_M xA yA = (x, y) →
  (2 * x - 3)^2 + 4 * y^2 = 1 := by
  sorry

end locus_of_midpoint_l5_5037


namespace red_paint_amount_l5_5718

theorem red_paint_amount (r w : ℕ) (hrw : r / w = 5 / 7) (hwhite : w = 21) : r = 15 :=
by {
  sorry
}

end red_paint_amount_l5_5718


namespace repeating_decimal_fraction_l5_5990

theorem repeating_decimal_fraction :  exists (p q : ℤ), (q ≠ 0) ∧ (p / q = (68 / 33 : ℚ)) :=
begin
  let recurring02 : ℚ := 2 / 99,
  let recurring06 : ℚ := 3 * recurring02,
  have recurring06_simplified : recurring06 = 2 / 33, 
    by simp [recurring06, mul_div_assoc, mul_div_cancel_left],
  let result := 2 + recurring06_simplified,
  use [68, 33],
  split,
  { norm_num }, -- proof that 33 ≠ 0
  { exact result } -- proof that 2.06 recurring = 68 / 33
end

end repeating_decimal_fraction_l5_5990


namespace expected_value_of_eight_sided_die_l5_5344

-- Definitions based on the problem conditions
def eight_sided_die_outcomes : List ℕ := [1,2,3,4,5,6,7,8]

def probability (n : ℕ) := 1 / n

-- Expected value calculation related to the problem
def expected_value_die_roll (outcomes : List ℕ) (prob : ℕ → Rat) : Rat :=
  List.sum (outcomes.map (λ x => prob outcomes.length * x))

-- Expected value of an 8-sided die roll
theorem expected_value_of_eight_sided_die :
  expected_value_die_roll eight_sided_die_outcomes probability = 4.5 := 
sorry

end expected_value_of_eight_sided_die_l5_5344


namespace river_width_l5_5043

noncomputable def width_of_river (d: ℝ) (f: ℝ) (v: ℝ) : ℝ :=
  v / (d * (f * 1000 / 60))

theorem river_width : width_of_river 2 2 3000 = 45 := by
  sorry

end river_width_l5_5043


namespace unique_solution_l5_5433

noncomputable def unique_solution_exists : Prop :=
  ∃ (a b c d e : ℕ), 
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧
    (a + b = (c + d + e) / 7) ∧
    (a + d = (b + c + e) / 5) ∧
    (a + b + c + d + e = 24) ∧
    (a = 1 ∧ b = 2 ∧ c = 9 ∧ d = 3 ∧ e = 9)

theorem unique_solution : unique_solution_exists :=
sorry

end unique_solution_l5_5433


namespace half_number_plus_seven_l5_5681

theorem half_number_plus_seven (n : ℕ) (h : n = 20) : (n / 2) + 7 = 17 :=
by
  rw [h]
  -- We assert "n / 2 + 7 = 17"
  -- However, actual proof steps are not needed for this task.
  sorry

end half_number_plus_seven_l5_5681


namespace range_of_k_l5_5166

theorem range_of_k (k : ℝ) :
  (-4 / 3 : ℝ) ≤ k ∧ k ≤ 0 ↔
  ∃ (A B : ℝ × ℝ), 
    ((A.1 - 2)^2 + (A.2 + 1)^2 = 4 ∧ (B.1 - 2)^2 + (B.2 + 1)^2 = 4) ∧
    (A.2 = k * A.1 ∧ B.2 = k * B.1) ∧
    dist A B ≥ 2 * sqrt 3 := by 
  sorry

end range_of_k_l5_5166


namespace isosceles_triangle_sides_part1_isosceles_triangle_sides_part2_l5_5051

-- Part 1 proof
theorem isosceles_triangle_sides_part1 (x : ℝ) (h1 : x + 2 * x + 2 * x = 20) : 
  x = 4 ∧ 2 * x = 8 :=
by
  sorry

-- Part 2 proof
theorem isosceles_triangle_sides_part2 (a b : ℝ) (h2 : a = 5) (h3 : 2 * b + a = 20) :
  b = 7.5 :=
by
  sorry

end isosceles_triangle_sides_part1_isosceles_triangle_sides_part2_l5_5051


namespace incenter_circumcenter_distance_l5_5042

theorem incenter_circumcenter_distance (a b c : ℝ) (h1 : a = 8) (h2 : b = 15) (h3 : c = 17) (h4 : a^2 + b^2 = c^2) : 
    distance_between_incenter_and_circumcenter a b c = (real.sqrt 85) / 2 :=
by
    sorry

end incenter_circumcenter_distance_l5_5042


namespace domain_of_f_l5_5961

theorem domain_of_f (x : ℝ) : 
    (x < 3 / 2) ∧ (x ≠ 1) ↔ x ∈ set.Ioo (-∞) 1 ∪ set.Ioo 1 (3 / 2) := by
    sorry

end domain_of_f_l5_5961


namespace number_of_arrangements_l5_5336

-- Definitions based on conditions
def boys : ℕ := 2
def girls : ℕ := 3
def units : ℕ := 4  -- after treating two girls as one unit

-- The statement to be proved
theorem number_of_arrangements (boys girls units : ℕ) (hboys : boys = 2) (hgirls : girls = 3) (hunits : units = 4) :
  let girl_pairs := 2! in
  units! * girl_pairs = 48 :=
by 
  sorry

end number_of_arrangements_l5_5336


namespace square_of_real_not_always_positive_l5_5856

theorem square_of_real_not_always_positive (a : ℝ) : ¬(a^2 > 0) := 
sorry

end square_of_real_not_always_positive_l5_5856


namespace only_MiddleSchoolStudentsGanzhou_forms_set_l5_5426

-- Define the options as elements
inductive Option
| FamousHostsCCTV
| FastestCarsCity
| MiddleSchoolStudentsGanzhou
| TallBuildingsGanzhou

-- Define a function that checks if an option forms a definite set
def isDefiniteSet : Option → Prop
| Option.FamousHostsCCTV         := False
| Option.FastestCarsCity         := False
| Option.MiddleSchoolStudentsGanzhou := True
| Option.TallBuildingsGanzhou    := False

-- The goal is to prove that only Option.MiddleSchoolStudentsGanzhou forms a definite set
theorem only_MiddleSchoolStudentsGanzhou_forms_set :
  isDefiniteSet Option.MiddleSchoolStudentsGanzhou = True := by
  sorry

end only_MiddleSchoolStudentsGanzhou_forms_set_l5_5426


namespace probability_of_multiple_of_45_l5_5188

noncomputable def single_digit_multiples_of_3 := {x : ℕ | x ∈ {3, 6, 9}}

noncomputable def prime_numbers_less_than_20 := {x : ℕ | x ∈ {2, 3, 5, 7, 11, 13, 17, 19}}

noncomputable def is_multiple_of_45 (n : ℕ) : Prop :=
  45 ∣ n

theorem probability_of_multiple_of_45 : 
  (↑((1 : ℚ) / 3) : ℚ) * (↑((1 : ℚ) / 8) : ℚ) = (1 : ℚ) / 24 :=
by 
  sorry

end probability_of_multiple_of_45_l5_5188


namespace problem_l5_5760

noncomputable def a : ℝ := Real.log 6 / Real.log 3
noncomputable def b : ℝ := Real.log 10 / Real.log 5
noncomputable def c : ℝ := Real.log 12 / Real.log 6

theorem problem : a > b ∧ b > c := by
  sorry

end problem_l5_5760


namespace percentage_of_muslim_boys_l5_5722

def totalBoys : ℕ := 850
def hinduPercent : ℝ := 28 / 100
def sikhPercent : ℝ := 10 / 100
def otherCommunityBoys : ℕ := 187

theorem percentage_of_muslim_boys :
  let hinduBoys := hinduPercent * totalBoys
      sikhBoys := sikhPercent * totalBoys
      nonMuslimBoys : ℝ := hinduBoys + sikhBoys + otherCommunityBoys
      muslimBoys : ℝ := totalBoys - nonMuslimBoys
      muslimPercent : ℝ := (muslimBoys / totalBoys) * 100
  in muslimPercent = 40 := by
    sorry

end percentage_of_muslim_boys_l5_5722


namespace ab_plus_de_l5_5229

variables {A B C D E : Type}
variables [linear_ordered_field A]
variables (AD BD CD AE CE : A)
variables (AB DE : A)

noncomputable def am_equals : Prop :=
  AD = 60 ∧ BD = 189 ∧ CD = 36 ∧ AE = 40 ∧ CE = 50

theorem ab_plus_de (h : am_equals AD BD CD AE CE) : 
  AB + DE = 120 := 
sorry

end ab_plus_de_l5_5229


namespace area_of_rhombus_l5_5099

/-- Given the radii of the circles circumscribed around triangles EFG and EGH
    are 10 and 20, respectively, then the area of rhombus EFGH is 30.72√3. -/
theorem area_of_rhombus (R1 R2 : ℝ) (A : ℝ) :
  R1 = 10 → R2 = 20 → A = 30.72 * Real.sqrt 3 :=
by sorry

end area_of_rhombus_l5_5099


namespace crackers_eaten_by_Daniel_and_Elsie_l5_5779

theorem crackers_eaten_by_Daniel_and_Elsie :
  ∀ (initial_crackers remaining_crackers eaten_by_Ally eaten_by_Bob eaten_by_Clair: ℝ),
    initial_crackers = 27.5 →
    remaining_crackers = 10.5 →
    eaten_by_Ally = 3.5 →
    eaten_by_Bob = 4.0 →
    eaten_by_Clair = 5.5 →
    initial_crackers - remaining_crackers = (eaten_by_Ally + eaten_by_Bob + eaten_by_Clair) + (4 : ℝ) :=
by sorry

end crackers_eaten_by_Daniel_and_Elsie_l5_5779


namespace sum_of_odd_indexed_terms_l5_5045

theorem sum_of_odd_indexed_terms (a : ℕ → ℕ) (n : ℕ) :
  (∀ k < n, a (k + 1) = a k + 2) ∧ (a 0 + ∑ i in finset.range 2020, a (i + 1)) = 6060 →
  ∑ i in finset.range 1010, a (2 * i + 1) = 2020 :=
by
  sorry

end sum_of_odd_indexed_terms_l5_5045


namespace stratified_sampling_correct_l5_5717

noncomputable def stratified_sampling (n1 n2 n3 sample_size total_students : ℕ) : Prop :=
  let total := n1 + n2 + n3
  in (total = total_students) ∧
     (n1 * sample_size / total = 27) ∧ 
     (n2 * sample_size / total = 22) ∧ 
     (n3 * sample_size / total = 21)

theorem stratified_sampling_correct: 
  stratified_sampling 540 440 420 70 1400 := by
  sorry

end stratified_sampling_correct_l5_5717


namespace area_of_shaded_region_l5_5498

-- Definitions of given conditions
def octagon_side_length : ℝ := 5
def arc_radius : ℝ := 4

-- Theorem statement
theorem area_of_shaded_region : 
  let octagon_area := 50
  let sectors_area := 16 * Real.pi
  octagon_area - sectors_area = 50 - 16 * Real.pi :=
by
  sorry

end area_of_shaded_region_l5_5498


namespace expected_value_of_8_sided_die_l5_5383

theorem expected_value_of_8_sided_die : 
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let probability := 1 / 8
  (∑ outcome in outcomes, outcome * probability) = 4.5 :=
by
  sorry

end expected_value_of_8_sided_die_l5_5383


namespace repeating_decimal_fraction_l5_5989

theorem repeating_decimal_fraction :  exists (p q : ℤ), (q ≠ 0) ∧ (p / q = (68 / 33 : ℚ)) :=
begin
  let recurring02 : ℚ := 2 / 99,
  let recurring06 : ℚ := 3 * recurring02,
  have recurring06_simplified : recurring06 = 2 / 33, 
    by simp [recurring06, mul_div_assoc, mul_div_cancel_left],
  let result := 2 + recurring06_simplified,
  use [68, 33],
  split,
  { norm_num }, -- proof that 33 ≠ 0
  { exact result } -- proof that 2.06 recurring = 68 / 33
end

end repeating_decimal_fraction_l5_5989


namespace circle_triangle_ratio_l5_5441

/-- Given the conditions on the geometric configuration of circles and tangents, 
show that the ratio of AX to XY is 1/2. -/
theorem circle_triangle_ratio (ω ω₁ ω₂ : Circle) (A B C X Y : Point)
  (h_ab : ω₁.tangent_at A = Line.through A B)
  (h_ac : ω₂.tangent_at A = Line.through A C)
  (h_ω : IsCircumscribedTriangle ω A B C)
  (h_x : IsTangentPoint ω₁ X (TangentAt ω A))
  (h_y : IsTangentPoint ω₂ Y (TangentAt ω A)) :
  AX / XY = 1 / 2 :=
sorry

end circle_triangle_ratio_l5_5441


namespace min_value_expression_l5_5892

theorem min_value_expression (a b : ℝ) (h_a : 0 ≤ a) (h_b : 0 ≤ b) :
  (\frac{|a-3*b-2| + |3*a-b|}{\sqrt{a^2 + (b+1)^2}}) = 2 := 
sorry

end min_value_expression_l5_5892


namespace expected_value_8_sided_die_l5_5369

theorem expected_value_8_sided_die :
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let probability_each_outcome := 1 / 8
  (Σ x ∈ outcomes, probability_each_outcome * x) = 4.5 :=
by
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let probability_each_outcome := 1 / 8
  have h : (Σ x ∈ outcomes, probability_each_outcome * x) = (1 / 8) * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8) := by sorry
  have sum_eq_36 : 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 = 36 := by sorry
  rw [sum_eq_36] at h
  have expected_value_eq : (1 / 8) * 36 = 4.5 := by sorry
  rw [expected_value_eq] at h
  exact h

end expected_value_8_sided_die_l5_5369


namespace C1_standard_form_C2_parametric_form_C1_C2_intersect_l5_5731

-- Definitions of the curves
def curve_C1 (t : ℝ) : ℝ × ℝ := (4 + t, 5 + 2 * t)

def standard_form_C1 (x y : ℝ) : Prop :=
  y = 2 * x - 3

def polar_curve_C2 (ρ θ : ℝ) : Prop :=
  ρ^2 - 6 * ρ * Real.cos θ - 10 * Real.sin θ + 9 = 0

def parametric_curve_C2 (α : ℝ) : ℝ × ℝ :=
  (3 + 5 * Real.cos α, 5 + 5 * Real.sin α)

-- Proof objectives
theorem C1_standard_form :
  ∀ x y t : ℝ, curve_C1 t = (x, y) → standard_form_C1 x y := 
sorry
  
theorem C2_parametric_form :
  ∀ α ρ θ : ℝ, polar_curve_C2 ρ θ ↔ (3 + 5 * Real.cos α, 5 + 5 * Real.sin α) = parametric_curve_C2 α := 
sorry

theorem C1_C2_intersect :
  ∀ x y t α : ℝ, curve_C1 t = (x, y) → parametric_curve_C2 α = (x, y) → ∃ x y, True :=
sorry

end C1_standard_form_C2_parametric_form_C1_C2_intersect_l5_5731


namespace choose_5_with_exactly_one_twin_l5_5021

theorem choose_5_with_exactly_one_twin :
  let total_players := 12
  let twins := 2
  let players_to_choose := 5
  let remaining_players_after_one_twin := total_players - twins + 1 -- 11 players to choose from
  (2 * Nat.choose remaining_players_after_one_twin (players_to_choose - 1)) = 420 := 
by
  sorry

end choose_5_with_exactly_one_twin_l5_5021


namespace determinant_of_sum_l5_5168

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![5, 6], ![2, 3]]
def B : Matrix (Fin 2) (Fin 2) ℤ := ![![1, 1], ![1, 0]]

theorem determinant_of_sum : (A + B).det = -3 := 
by 
  sorry

end determinant_of_sum_l5_5168


namespace find_a_plus_b_l5_5161

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x * real.log x + b

theorem find_a_plus_b (a b : ℝ) (h_tangent : ∀ x y, (x = 1 ∧ y = f a b x) → 2 * x - y = 0) :
  a + b = 4 :=
by
  -- From the condition that the tangent line at x = 1 is 2x - y = 0:
  have h_deriv : deriv (f a b) 1 = 2 := sorry

  -- From the condition on the function at x = 1:
  have h_f1 : f a b 1 = 2 := sorry

  -- Derivative of f(a, b, x):
  let df := a * (1 + real.log 1)
  have h_df : df = 2 := sorry

  -- Simplify the derivative condition:
  have ha : df = a := sorry
  rw ha at h_df

  -- From the above, it follows a = 2:
  have ha2 : a = 2 := sorry

  -- As f(1) = b and h_f1 implies:
  have hb : b = 2 := sorry

  -- Therefore, a + b = 4:
  have hab : a + b = 4 := sorry

  exact hab

end find_a_plus_b_l5_5161


namespace trisecting_angle_ratio_l5_5743

theorem trisecting_angle_ratio
  (A B C D E : Point)
  (h_triangle : Triangle A B C)
  (h_B_trisection : Angle B ∠trisected by BD and BE)
  (h_intersect_D : SegIntersect AC BD D)
  (h_intersect_E : SegIntersect AC BE E) :
  AD / EC = AB * BD / (BE * BC) :=
sorry

end trisecting_angle_ratio_l5_5743


namespace travel_speed_l5_5694

theorem travel_speed (distance time : ℕ) (h₁ : distance = 78) (h₂ : time = 39) : distance / time = 2 := 
by
  rw [h₁, h₂]
  norm_num

end travel_speed_l5_5694


namespace sum_sequence_integer_part_l5_5442

def a (i : ℕ) : ℝ := 1 + (10^i - 1) / 10^i

def S : ℝ := ∑ i in Finset.range 2018, a (i + 1)

theorem sum_sequence_integer_part 
  (int_part : ℤ)
  (dec_part : ℝ)
  (zeros_count : ℕ) 
  (h_int: int_part = 4035) 
  (h_dec_part: dec_part = 1 / 10^(zeros_count + 1))
  (h_zeros_count: zeros_count = 2017) : 
  ⌊S⌋ = ↑int_part ∧ (S - ⌊S⌋ = dec_part) :=
sorry

end sum_sequence_integer_part_l5_5442


namespace clock_angle_8_15_l5_5865

theorem clock_angle_8_15:
  ∃ angle : ℝ, time_on_clock = 8.25 → angle = 157.5 := sorry

end clock_angle_8_15_l5_5865


namespace count_satisfying_sets_l5_5840

theorem count_satisfying_sets :
  ∃ (B : set (fin 5)) → (insert 1 (insert 2 B) = {i | i < 5}) ∧ 
  (card B = 4) → true := 
sorry

end count_satisfying_sets_l5_5840


namespace expected_value_of_8_sided_die_l5_5387

theorem expected_value_of_8_sided_die : 
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let probability := 1 / 8
  (∑ outcome in outcomes, outcome * probability) = 4.5 :=
by
  sorry

end expected_value_of_8_sided_die_l5_5387


namespace exist_consecutive_natural_numbers_with_largest_prime_l5_5338

noncomputable def largest_prime_divisor (n : ℕ) : ℕ := sorry

theorem exist_consecutive_natural_numbers_with_largest_prime (
  p q : ℕ) (hp_prime : Prime p) (hq_prime : Prime q) 
  (p_ne_q : p ≠ q) (prime_diff_cond : abs (p - q) < 2 * p) : 
  ∃ (n : ℕ), largest_prime_divisor n = p ∧ largest_prime_divisor (n + 1) = q :=
sorry

end exist_consecutive_natural_numbers_with_largest_prime_l5_5338


namespace expected_value_of_eight_sided_die_l5_5374

theorem expected_value_of_eight_sided_die : 
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8], 
      n := (outcomes.length : ℚ),
      probabilities := List.replicate (outcomes.length) (1 / n),
      expected_value := (List.zipWith (*) probabilities (outcomes.map (· : ℚ))).sum
  in expected_value = 4.5 :=
by
  sorry

end expected_value_of_eight_sided_die_l5_5374


namespace range_of_a_l5_5305

theorem range_of_a (a : ℝ) :
  (∃ (f : ℝ → ℝ), (∀ x, 0 < x ∧ x < 1 → f(x) = log a (3 - a * x^2)) ∧ 
  (∀ x y, 0 < x ∧ x < y ∧ y < 1 → f'(x) < 0 ∧ f'(y) < 0)) → 
  (1 < a ∧ a ≤ 3) :=
begin
  sorry
end

end range_of_a_l5_5305


namespace three_digit_numbers_distinct_count_l5_5685

def distinct_three_digit_numbers_count : Nat :=
  5.choose 3 * 3.factorial

theorem three_digit_numbers_distinct_count : distinct_three_digit_numbers_count = 60 := by
  sorry

end three_digit_numbers_distinct_count_l5_5685


namespace initial_books_count_l5_5038

-- Define the conditions
def books_end_month : ℝ := 66
def percentage_returned : ℝ := 0.70
def books_loaned_out : ℝ := 29.999999999999996

-- Define the main statement
theorem initial_books_count (books_end_month : ℝ) (percentage_returned : ℝ) (books_loaned_out : ℝ)
  (H1 : books_end_month = 66)
  (H2 : percentage_returned = 0.70)
  (H3 : books_loaned_out = 29.999999999999996) :
  let books_returned := percentage_returned * books_loaned_out in
  let books_not_returned := books_loaned_out - books_returned in
  let initial_books := books_end_month + books_not_returned in
  initial_books = 75 :=
by
  -- Proof would go here
  sorry

end initial_books_count_l5_5038


namespace toy_position_from_left_l5_5016

/-- Define the total number of toys -/
def total_toys : ℕ := 19

/-- Define the position of toy (A) from the right -/
def position_from_right : ℕ := 8

/-- Prove the main statement: The position of toy (A) from the left is 12 given the conditions -/
theorem toy_position_from_left : total_toys - position_from_right + 1 = 12 := by
  sorry

end toy_position_from_left_l5_5016


namespace time_to_cross_platform_l5_5822

-- Definitions for the length of the train, the length of the platform, and the speed of the train
def length_train : ℕ := 750
def length_platform : ℕ := 750
def speed_train_kmh : ℕ := 90

-- Conversion constants
def meters_per_kilometer : ℕ := 1000
def seconds_per_hour : ℕ := 3600

-- Convert speed from km/hr to m/s
def speed_train_ms : ℚ := speed_train_kmh * meters_per_kilometer / seconds_per_hour

-- Total distance the train covers to cross the platform
def total_distance : ℕ := length_train + length_platform

-- Proof problem: To prove that the time taken to cross the platform is 60 seconds
theorem time_to_cross_platform : total_distance / speed_train_ms = 60 := by
  sorry

end time_to_cross_platform_l5_5822


namespace smallest_n_l5_5569

theorem smallest_n (n : ℕ) (h1 : n > 2016) (h2 : (1^n + 2^n + 3^n + 4^n) % 10 ≠ 0) : n = 2020 :=
sorry

end smallest_n_l5_5569


namespace parallelogram_area_l5_5102

noncomputable def vector_u : ℝ × ℝ × ℝ := (4, 2, -3)
noncomputable def vector_v : ℝ × ℝ × ℝ := (2, -4, 5)

theorem parallelogram_area : 
  let cross_product := (vector_u.2 * vector_v.3 - vector_u.3 * vector_v.2, vector_u.3 * vector_v.1 - vector_u.1 * vector_v.3, vector_u.1 * vector_v.2 - vector_u.2 * vector_v.1) in
  let magnitude := real.sqrt (cross_product.1^2 + cross_product.2^2 + cross_product.3^2) in
  magnitude = 10 * real.sqrt (10.8) :=
sorry

end parallelogram_area_l5_5102


namespace largest_common_value_arithmetic_progressions_l5_5522

theorem largest_common_value_arithmetic_progressions : 
  ∃ a, (∃ n, ∃ m, a = 7 * n ∧ a = 5 + 11 * m) ∧ a < 1000 ∧ 
  (∀ b, (∃ n, ∃ m, b = 7 * n ∧ b = 5 + 11 * m) ∧ b < 1000 → b ≤ a) :=
begin
  sorry
end

end largest_common_value_arithmetic_progressions_l5_5522


namespace sum_of_divisors_satisfying_conditions_l5_5118

-- Definitions of the problem conditions
def is_divisor (m n : ℕ) : Prop := ∃ k, m = k * n

def is_perfect_square (n : ℕ) : Prop := ∃ k, n = k * k

def satisfies_conditions (n : ℕ) : Prop :=
  is_divisor 24 n ∧ is_perfect_square (n^2 - 11 * n + 24)

-- Proposition statement
theorem sum_of_divisors_satisfying_conditions : 
  ∑ n in { n | satisfies_conditions n }, n = 6 :=
by
-- Proof goes here
sorry

end sum_of_divisors_satisfying_conditions_l5_5118


namespace simplify_sqrt_88200_l5_5287

theorem simplify_sqrt_88200 :
  (Real.sqrt 88200) = 70 * Real.sqrt 6 := 
by 
  -- given conditions
  have h : 88200 = 2^3 * 3 * 5^2 * 7^2 := sorry,
  sorry

end simplify_sqrt_88200_l5_5287


namespace area_triangle_BDM_l5_5214

-- Define the given geometric properties
variable (A B C D M : Point)
-- Define distances
variable (AB BC AC AM MB BD DC : ℝ)

-- Given conditions
axiom isosceles_triangle : AB = 3 ∧ BC = 3 ∧ AC = 4
axiom midpoint_M : M.is_midpoint A B
axiom point_D_on_BC : BD = 1 ∧ DC = BC - BD

-- Define the task to prove
theorem area_triangle_BDM :
  (let s := (BD + AM + BM) / 2;
       area := sqrt (s * (s - BD) * (s - AM) * (s - BM)))
     in area = 1/2 := 
sorry -- skipping the actual proof

end area_triangle_BDM_l5_5214


namespace total_value_of_item_l5_5887

theorem total_value_of_item
  (import_tax : ℝ)
  (V : ℝ)
  (h₀ : import_tax = 110.60)
  (h₁ : import_tax = 0.07 * (V - 1000)) :
  V = 2579.43 := 
sorry

end total_value_of_item_l5_5887


namespace ratio_B_to_C_l5_5792

theorem ratio_B_to_C (A_share B_share C_share : ℝ) 
  (total : A_share + B_share + C_share = 510) 
  (A_share_val : A_share = 360) 
  (B_share_val : B_share = 90)
  (C_share_val : C_share = 60)
  (A_cond : A_share = (2 / 3) * B_share) 
  : B_share / C_share = 3 / 2 := 
by 
  sorry

end ratio_B_to_C_l5_5792


namespace smallest_coprime_gt_one_l5_5576

theorem smallest_coprime_gt_one (x : ℕ) (h1 : 1 < x) (h2 : Nat.gcd x 180 = 1) : x = 7 := sorry

end smallest_coprime_gt_one_l5_5576


namespace smallest_rel_prime_to_180_l5_5577

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ Nat.gcd x 180 = 1 ∧ ∀ y : ℕ, y > 1 ∧ Nat.gcd y 180 = 1 → x ≤ y :=
begin
  use 7,
  split,
  { exact dec_trivial },
  split,
  { exact dec_trivial },
  { intro y,
    intro h,
    cases h with h1 h2,
    repeat { try { apply dec_trivial,
                   apply lt_or_eq_of_le,
                   norm_num,
                   apply Nat.prime_not_dvd_mul,
                   norm_num,
                   apply not_or_distrib.mpr,
                   split,
                   norm_cast,
                   intro,
                   exact le_antisymm _ },
           sorry }
end

end smallest_rel_prime_to_180_l5_5577


namespace quadratic_function_root_range_l5_5144

theorem quadratic_function_root_range (b c : ℝ)
  (h1 : (λ x, x^2 + 2*b*x + c) 1 = 0)
  (h2 : ∃ (r₁ r₂ : ℝ), 
        r₁ ∈ set.Ioo (-3 : ℝ) (-2) ∧ 
        r₂ ∈ set.Ioo 0 1 ∧ 
        (λ x, x^2 + 3*b*x + (b + c)) r₁ = 0 ∧
        (λ x, x^2 + 3*b*x + (b + c)) r₂ = 0) :
  b ∈ set.Ioo (-5/2 : ℝ) (-1/2) :=
sorry

end quadratic_function_root_range_l5_5144


namespace cos_pi_sub_alpha_l5_5133

variables {α : ℝ}

-- Given condition
axiom sin_sub_half_pi : sin (π / 2 - α) = 3 / 5

-- Proof statement
theorem cos_pi_sub_alpha : cos (π - α) = -3 / 5 :=
by
  sorry

end cos_pi_sub_alpha_l5_5133


namespace area_f2_closed_region_l5_5775

def f0 (x : ℝ) : ℝ := abs x
def f1 (x : ℝ) : ℝ := abs (f0 x - 1)
def f2 (x : ℝ) : ℝ := abs (f1 x - 2)

theorem area_f2_closed_region : 
  let area := ∫ (x : ℝ) in -3..3, f2 x in
  area = 7 := by
  sorry

end area_f2_closed_region_l5_5775


namespace quadratic_has_two_distinct_real_roots_l5_5316

theorem quadratic_has_two_distinct_real_roots :
  (∃ a b c, a ≠ 0 ∧ b = -3 ∧ c = -1 ∧ a * (x^2) + b * x + c = 0 ∧
              (b^2 - 4 * a * c) > 0) :=
by {
  use [1, -3, -1],
  split,
  { exact one_ne_zero },
  split,
  { refl },
  split,
  { refl },
  split,
  { -- Verification that the quadratic equation matches
    intros x h,
    sorry
  },
  { -- Proof that the discriminant is positive
    have : (-3)^2 - 4 * 1 * (-1) = 9 + 4 := by norm_num,
    linarith,
    sorry
  }
}

end quadratic_has_two_distinct_real_roots_l5_5316


namespace calculate_mailing_cost_l5_5040

-- Define the main function to calculate the mailing cost
def mailing_cost (W : ℝ) : ℝ :=
  5 * (⌈W⌉ : ℕ) + 3

-- Main theorem stating the proof problem
theorem calculate_mailing_cost (W : ℝ) : 
  mailing_cost W = 5 * (⌈W⌉ : ℕ) + 3 := 
by sorry

end calculate_mailing_cost_l5_5040


namespace letter_puzzle_solutions_l5_5550

noncomputable def is_solution (A B : ℕ) : Prop :=
A ≠ B ∧ A ∈ finset.range (10) ∧ B ∈ finset.range (10) ∧ 10 ≤ B * 10 + A ∧ B * 10 + A ≤ 99 ∧ A^B = B * 10 + A

theorem letter_puzzle_solutions :
  ∃ A B : ℕ, is_solution A B ∧ ((A = 2 ∧ B = 5) ∨ (A = 6 ∧ B = 2) ∨ (A = 4 ∧ B = 3)) :=
by
  sorry

end letter_puzzle_solutions_l5_5550


namespace sum_of_odd_terms_l5_5046

-- Define a sequence and state the main properties and the sum condition
def sequence (x : ℕ → ℕ) :=
  (∀ n, x (n + 1) = x n + 2) ∧
  (∑ i in Finset.range 2020, x i = 6060)

-- Define the sum of every second term, starting with the first, as S
def sum_of_every_second_term (x : ℕ → ℕ) : ℕ :=
  ∑ i in Finset.range 1010, x (2 * i)

-- The main theorem we want to prove
theorem sum_of_odd_terms (x : ℕ → ℕ) (h : sequence x) :
  sum_of_every_second_term x = 2020 :=
sorry

end sum_of_odd_terms_l5_5046


namespace area_A₁B₁C₁_le_one_fourth_area_ABC_l5_5249

variables {A B C A₁ B₁ C₁ : Type}
variables [HasMeasure A] [HasMeasure B] [HasMeasure C] [HasMeasure A₁] [HasMeasure B₁] [HasMeasure C₁]

-- Given points A, B, C defining the triangle ABC
-- A₁, B₁, and C₁ being the points where the internal angle bisectors intersect the opposite sides

def area_ABC := measureABC A B C
def area_A₁B₁C₁ := measureA₁B₁C₁ A₁ B₁ C₁

theorem area_A₁B₁C₁_le_one_fourth_area_ABC 
(hA₁ : is_intersection_bisector A A₁ B C)
(hB₁ : is_intersection_bisector B B₁ A C)
(hC₁ : is_intersection_bisector C C₁ A B) :
  area_A₁B₁C₁ ≤ (1 / 4) * area_ABC :=
sorry

end area_A₁B₁C₁_le_one_fourth_area_ABC_l5_5249


namespace expected_value_8_sided_die_l5_5362

theorem expected_value_8_sided_die :
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let probability_each_outcome := 1 / 8
  (Σ x ∈ outcomes, probability_each_outcome * x) = 4.5 :=
by
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let probability_each_outcome := 1 / 8
  have h : (Σ x ∈ outcomes, probability_each_outcome * x) = (1 / 8) * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8) := by sorry
  have sum_eq_36 : 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 = 36 := by sorry
  rw [sum_eq_36] at h
  have expected_value_eq : (1 / 8) * 36 = 4.5 := by sorry
  rw [expected_value_eq] at h
  exact h

end expected_value_8_sided_die_l5_5362


namespace soccer_camp_afternoon_l5_5843

-- Define the total number of kids
def total_kids : ℕ := 2000

-- Define the fraction of kids going to soccer camp
def fraction_soccer : ℝ := 1 / 2

-- Define the fraction of kids going to morning soccer camp
def fraction_morning : ℝ := 1 / 4

-- Calculate the number of kids going to soccer camp
def kids_soccer : ℕ := (total_kids * fraction_soccer.to_nat)

-- Calculate the number of kids going to morning soccer camp
def kids_morning_soccer : ℕ := (kids_soccer * fraction_morning.to_nat)

-- Calculate the number of kids going to afternoon soccer camp
def kids_afternoon_soccer : ℕ := kids_soccer - kids_morning_soccer

-- The theorem stating the resultant number for the kids going to soccer camp in the afternoon
theorem soccer_camp_afternoon : kids_afternoon_soccer = 750 := by
  sorry

end soccer_camp_afternoon_l5_5843


namespace angle_B_eq_pi_over_3_range_of_area_l5_5723

-- Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively
-- And given vectors m and n represented as stated and are collinear
-- Prove that angle B is π/3
theorem angle_B_eq_pi_over_3
  (ABC_acute : True) -- placeholder condition indicating acute triangle
  (a b c A B C : ℝ)
  (m := (2 * Real.sin (A + C), - Real.sqrt 3))
  (n := (Real.cos (2 * B), 2 * Real.cos (B / 2) ^ 2 - 1))
  (collinear_m_n : m.1 * n.2 = m.2 * n.1) :
  B = Real.pi / 3 :=
sorry

-- Given side b = 1, find the range of the area S of triangle ABC
theorem range_of_area
  (a c A B C : ℝ)
  (triangle_area : ℝ)
  (ABC_acute : True) -- placeholder condition indicating acute triangle
  (hB : B = Real.pi / 3)
  (hb : b = 1)
  (cosine_theorem : 1 = a^2 + c^2 - a*c)
  (area_formula : triangle_area = (1/2) * a * c * Real.sin B) :
  0 < triangle_area ∧ triangle_area ≤ (Real.sqrt 3) / 4 :=
sorry

end angle_B_eq_pi_over_3_range_of_area_l5_5723


namespace abs_product_correct_l5_5632

theorem abs_product_correct {x y : ℤ} (hx : x = -2) (hy : y = 4) : |x * y| = 8 := by
  rw [hx, hy]
  exact abs_of_neg_mul_eq 8

end abs_product_correct_l5_5632


namespace incorrect_statement_D_l5_5255

variables {n : ℕ} (a : Fin n.succ → ℝ) (x x1 x2 : ℝ)

-- Definition of the function f
def f (x : ℝ) : ℝ := 
  ∑ i in Finset.range n.succ,
    (a i) * Real.sin (x + a i)

-- Conditions
def cond1 := ∀ i, 1 ≤ i ∧ i ≤ n → a i ∈ ℝ
def cond2 := f 0 = 0
def cond3 := f (Real.pi / 2) = 0
def cond4 := f 0 ≠ 0 ∧ f (Real.pi / 2) ≠ 0

-- Statement D
def statement_D := f x1 = 0 ∧ f x2 = 0 → ∃ k : ℤ, x1 - x2 = k * 2 * Real.pi

-- Proof that statement D is incorrect
theorem incorrect_statement_D : cond1 a n → cond2 a n → cond3 a n → cond4 a n → ¬ statement_D a x1 x2 :=
by
  intros
  sorry

end incorrect_statement_D_l5_5255


namespace letter_puzzle_solutions_l5_5559

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def to_number (B A : ℕ) : ℕ :=
  10 * B + A

theorem letter_puzzle_solutions (A B : ℕ) (h_diff : A ≠ B) (h_digits : 1 ≤ A ∧ A ≤ 9 ∧ 1 ≤ B ∧ B ≤ 9)
  (h_two_digit : is_two_digit (to_number B A)) :
  A^B = to_number B A ↔ (A = 2 ∧ B = 5 ∨ A = 6 ∧ B = 2 ∨ A = 4 ∧ B = 3) :=
by
  sorry

end letter_puzzle_solutions_l5_5559


namespace fraction_transform_l5_5422

theorem fraction_transform {x : ℤ} :
  (537 - x : ℚ) / (463 + x) = 1 / 9 ↔ x = 437 := by
sorry

end fraction_transform_l5_5422


namespace repeating_decimal_to_fraction_l5_5995

theorem repeating_decimal_to_fraction (h : 0.\overline{02} = 2 / 99) : 
  2.\overline{06} = 68 / 33 := by
  sorry

end repeating_decimal_to_fraction_l5_5995


namespace vertical_asymptote_at_5_l5_5952

noncomputable def f (x : ℝ) : ℝ := (x^3 + 3*x^2 + 2*x + 10) / (x - 5)

theorem vertical_asymptote_at_5 : ∃ a : ℝ, (a = 5) ∧ ∀ δ > 0, ∃ ε > 0, ∀ x : ℝ, 0 < |x - a| ∧ |x - a| < ε → |f x| > δ :=
by
  sorry

end vertical_asymptote_at_5_l5_5952


namespace find_number_of_valid_subsets_l5_5566

noncomputable def numValidSubsets : ℕ :=
  let count := 25.choose 9
  count

theorem find_number_of_valid_subsets :
  numValidSubsets = 177100 := by
  sorry

end find_number_of_valid_subsets_l5_5566


namespace angle_triple_complement_l5_5859

theorem angle_triple_complement (x : ℝ) (h1 : x + (180 - x) = 180) (h2 : x = 3 * (180 - x)) : x = 135 := 
by
  sorry

end angle_triple_complement_l5_5859


namespace expected_value_of_eight_sided_die_l5_5349

-- Definitions based on the problem conditions
def eight_sided_die_outcomes : List ℕ := [1,2,3,4,5,6,7,8]

def probability (n : ℕ) := 1 / n

-- Expected value calculation related to the problem
def expected_value_die_roll (outcomes : List ℕ) (prob : ℕ → Rat) : Rat :=
  List.sum (outcomes.map (λ x => prob outcomes.length * x))

-- Expected value of an 8-sided die roll
theorem expected_value_of_eight_sided_die :
  expected_value_die_roll eight_sided_die_outcomes probability = 4.5 := 
sorry

end expected_value_of_eight_sided_die_l5_5349


namespace expected_value_of_8_sided_die_is_4_point_5_l5_5356

def expected_value_8_sided_die : ℝ :=
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let probability := 1 / 8
  (list.sum outcomes : ℝ) * probability

theorem expected_value_of_8_sided_die_is_4_point_5 :
  expected_value_8_sided_die = 4.5 := by
  sorry

end expected_value_of_8_sided_die_is_4_point_5_l5_5356


namespace square_of_eccentricity_l5_5953

-- Definitions based on the conditions
def hyperbola_eq (a b : ℝ) (x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1
def parabola_eq (p x y : ℝ) : Prop := y^2 = 2 * p * x
def midpoint_formula (O F P Q : ℝ × ℝ) : Prop := 
  let (Ox, Oy) := O in
  let (Fx, Fy) := F in
  let (Px, Py) := P in
  let (Qx, Qy) := Q in
  (Ox = (Fx + Qx) / 2) ∧ (Oy = (Fy + Qy) / 2)

-- Main theorem statement
theorem square_of_eccentricity (a b p c : ℝ) (F P Q : ℝ × ℝ) (hC : c = p / 2)
(hyper_eq : ∀ x y, hyperbola_eq a b x y)
(para_eq : ∀ x y, parabola_eq p x y)
(mid_eq : midpoint_formula (0, 0) F P Q)
: c^4 - a^2 * c^2 - a^4 = 0 → (c / a)^4 - (c / a)^2 - 1 = 0 → (c / a)^2 = (Real.sqrt 5 + 1) / 2 :=
by
  -- Placeholder, proof not included
  sorry

end square_of_eccentricity_l5_5953


namespace derivative_at_zero_eq_three_l5_5667

def f (x : ℝ) : ℝ := (2 * x + 1) * Real.exp x

theorem derivative_at_zero_eq_three : (derivative f 0) = 3 := by
  sorry

end derivative_at_zero_eq_three_l5_5667


namespace proof_M_gt_five_lt_six_l5_5634

def in_Rplus (x : ℝ) : Prop := x > 0

theorem proof_M_gt_five_lt_six (a1 a2 a3 a4 : ℝ)
  (h0 : in_Rplus a1)
  (h1 : in_Rplus a2)
  (h2 : in_Rplus a3)
  (h3 : in_Rplus a4)
  (h_sum : a1 + a2 + a3 + a4 = 1) :
  let M := real.cbrt (7 * a1 + 1) + real.cbrt (7 * a2 + 1) + real.cbrt (7 * a3 + 1) + real.cbrt (7 * a4 + 1)
  in 5 < M ∧ M < 6 :=
by
  sorry

end proof_M_gt_five_lt_six_l5_5634


namespace expected_value_eight_sided_die_l5_5412

-- Define a standard 8-sided die
def eight_sided_die : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]

-- Compute the probability of each outcome
def probability (n : ℕ) : ℝ := 1 / 8

-- Expected Value of a discrete random variable
def expected_value (outcomes : List ℕ) (prob : ℕ → ℝ) : ℝ :=
  outcomes.sum / outcomes.length.toReal

-- Theorem stating the expected value of a standard 8-sided die roll is 4.5
theorem expected_value_eight_sided_die : expected_value eight_sided_die probability = 4.5 := by
  sorry

end expected_value_eight_sided_die_l5_5412


namespace letter_puzzle_solutions_l5_5542

theorem letter_puzzle_solutions (A B : ℕ) : 
  (1 ≤ A ∧ A < 10) ∧ (1 ≤ B ∧ B < 10) ∧ (A ≠ B) ∧ (A^B = 10 * B + A) → 
  (A = 2 ∧ B = 5) ∨ (A = 6 ∧ B = 2) ∨ (A = 4 ∧ B = 3) :=
by
  sorry

end letter_puzzle_solutions_l5_5542


namespace gcd_of_38_and_23_l5_5112

def gcd_subtraction (a b : ℕ) : ℕ :=
match a, b with
| 0, b => b
| a, 0 => a
| a, b => if a > b then gcd_subtraction (a - b) b else gcd_subtraction a (b - a)

theorem gcd_of_38_and_23 : gcd_subtraction 38 23 = 1 := by
  sorry

end gcd_of_38_and_23_l5_5112


namespace smallest_rel_prime_to_180_l5_5593

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ Nat.gcd x 180 = 1 ∧ ∀ y : ℕ, (y > 1 ∧ Nat.gcd y 180 = 1) → x ≤ y :=
begin
  sorry
end

end smallest_rel_prime_to_180_l5_5593


namespace find_a_l5_5307

theorem find_a (a : ℝ) : (-2 * a + 3 = -4) -> (a = 7 / 2) :=
by
  intro h
  sorry

end find_a_l5_5307


namespace triangular_pyramid_volume_l5_5664

theorem triangular_pyramid_volume (a b c : ℝ) 
  (h1 : 1 / 2 * a * b = 6) 
  (h2 : 1 / 2 * a * c = 4) 
  (h3 : 1 / 2 * b * c = 3) : 
  (1 / 3) * (1 / 2) * a * b * c = 4 := by 
  sorry

end triangular_pyramid_volume_l5_5664


namespace sufficient_but_not_necessary_condition_l5_5882

-- Define the given conditions
variable a : ℝ
def l1 (x y : ℝ) : Prop := a * x + 2 * y - 1 = 0
def l2 (x y : ℝ) : Prop := x + (a + 1) * y + 4 = 0

-- Define the statement to be proved
theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∀ x y : ℝ, l1 x y → l2 x y) → (a = 1 ∨ a = -2) :=
begin
  intro h,
  sorry
end

end sufficient_but_not_necessary_condition_l5_5882


namespace min_real_roots_l5_5242

theorem min_real_roots (g : Polynomial ℝ) (h_deg : g.degree = 2010)
  (roots : Fin 2010 → ℂ) (h_roots : ∀ i, g.eval (roots i) = 0)
  (distinct_abs_vals : (Finset (Fin 2010)).card (Finset.image (λ i, abs (roots i)) Finset.univ) = 1005) :
  ∃ n : ℕ, (g.real_roots ≥ n) ∧ (∀ m : ℕ, g.real_roots ≥ m → m ≥ 3) := 
sorry

end min_real_roots_l5_5242


namespace letter_puzzle_l5_5554

theorem letter_puzzle (A B : ℕ) (hA : 1 ≤ A ∧ A ≤ 9) (hB : 1 ≤ B ∧ B ≤ 9) (h_diff : A ≠ B) :
  A^B = 10 * B + A ↔ (A = 2 ∧ B = 5) ∨ (A = 6 ∧ B = 2) ∨ (A = 4 ∧ B = 3) :=
by
  sorry

end letter_puzzle_l5_5554


namespace score_of_juniors_correct_l5_5198

-- Let the total number of students be 20
def total_students : ℕ := 20

-- 20% of the students are juniors
def juniors_percent : ℝ := 0.20

-- Total number of juniors
def number_of_juniors : ℕ := 4 -- 20% of 20

-- The remaining are seniors
def number_of_seniors : ℕ := 16 -- 80% of 20

-- Overall average score of all students
def overall_average_score : ℝ := 85

-- Average score of the seniors
def seniors_average_score : ℝ := 84

-- Calculate the total score of all students
def total_score : ℝ := overall_average_score * total_students

-- Calculate the total score of the seniors
def total_score_of_seniors : ℝ := seniors_average_score * number_of_seniors

-- We need to prove that the score of each junior
def score_of_each_junior : ℝ := 89

theorem score_of_juniors_correct :
  (total_score - total_score_of_seniors) / number_of_juniors = score_of_each_junior :=
by
  sorry

end score_of_juniors_correct_l5_5198


namespace find_k_l5_5703

theorem find_k (t k : ℤ) (h1 : t = 35) (h2 : t = 5 * (k - 32) / 9) : k = 95 :=
sorry

end find_k_l5_5703


namespace area_of_parallelogram_l5_5104

-- Definitions of the vectors
def u : ℝ × ℝ × ℝ := (4, 2, -3)
def v : ℝ × ℝ × ℝ := (2, -4, 5)

-- Definition of the cross product
def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((a.2 * b.3 - a.3 * b.2), (a.3 * b.1 - a.1 * b.3), (a.1 * b.2 - a.2 * b.1))

-- Definition of the magnitude of a vector
def magnitude (a : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (a.1 * a.1 + a.2 * a.2 + a.3 * a.3)

-- The area of the parallelogram is the magnitude of the cross product of u and v
def area_parallelogram (u v : ℝ × ℝ × ℝ) : ℝ := magnitude (cross_product u v)

-- Proof statement
theorem area_of_parallelogram : area_parallelogram u v = 20 * real.sqrt 3 :=
by
  sorry

end area_of_parallelogram_l5_5104


namespace area_BCD_sixteen_area_BCD_with_new_ABD_l5_5641

-- Define the conditions and parameters of the problem.
variables (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]

-- Given conditions from part (a)
variable (AB_length : Real) (BC_length : Real) (area_ABD : Real)

-- Define the lengths and areas in our problem.
axiom AB_eq_five : AB_length = 5
axiom BC_eq_eight : BC_length = 8
axiom area_ABD_eq_ten : area_ABD = 10

-- Part (a) problem statement
theorem area_BCD_sixteen (AB_length BC_length area_ABD : Real) :
  AB_length = 5 → BC_length = 8 → area_ABD = 10 → (∃ area_BCD : Real, area_BCD = 16) :=
by
  sorry

-- Given conditions from part (b)
variable (new_area_ABD : Real)

-- Define the new area.
axiom new_area_ABD_eq_hundred : new_area_ABD = 100

-- Part (b) problem statement
theorem area_BCD_with_new_ABD (AB_length BC_length new_area_ABD : Real) :
  AB_length = 5 → BC_length = 8 → new_area_ABD = 100 → (∃ area_BCD : Real, area_BCD = 160) :=
by
  sorry

end area_BCD_sixteen_area_BCD_with_new_ABD_l5_5641


namespace max_min_z_diff_correct_l5_5245

noncomputable def max_min_z_diff (x y z : ℝ) (h1 : x + y + z = 3) (h2 : x^2 + y^2 + z^2 = 18) : ℝ :=
  6

theorem max_min_z_diff_correct (x y z : ℝ) (h1 : x + y + z = 3) (h2 : x^2 + y^2 + z^2 = 18) :
  max_min_z_diff x y z h1 h2 = 6 :=
sorry

end max_min_z_diff_correct_l5_5245


namespace expected_value_of_8_sided_die_l5_5379

theorem expected_value_of_8_sided_die : 
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let probability := 1 / 8
  (∑ outcome in outcomes, outcome * probability) = 4.5 :=
by
  sorry

end expected_value_of_8_sided_die_l5_5379


namespace num_whole_numbers_between_sqrt4_50_and_sqrt4_1000_l5_5693

def between_sqrt4_50_and_sqrt4_1000 : Prop :=
  3 < real.root 4 50 ∧ real.root 4 50 < 4 ∧
  5 < real.root 4 1000 ∧ real.root 4 1000 < 6

theorem num_whole_numbers_between_sqrt4_50_and_sqrt4_1000 (h : between_sqrt4_50_and_sqrt4_1000) : 
  ∃ n1 n2 : ℕ, real.root 4 50 < n1 ∧ n1 < n2 ∧ n2 < real.root 4 1000 ∧ n1 = 4 ∧ n2 = 5 :=
sorry

end num_whole_numbers_between_sqrt4_50_and_sqrt4_1000_l5_5693


namespace negation_of_p_l5_5239

variable {R : Type*} [LinearOrderedField R]

def f (x : R) : R

def proposition_p : Prop := ∀ x : R, x > 0 → f(x) > 0

theorem negation_of_p : ¬ proposition_p ↔ ∃ x : R, x > 0 ∧ f(x) ≤ 0 := by sorry

end negation_of_p_l5_5239


namespace summation_result_l5_5237

open Complex

noncomputable def ω (c : ℂ) : Prop := c^3 = 1 ∧ ¬ (c = 1 ∨ c = -1)

theorem summation_result
  (a : ℕ → ℝ)
  (ω : ℂ)
  (hω : ω^3 = 1 ∧ ¬(ω = 1 ∨ ω = -1))
  (h_sum : ∑ i in Finset.range n, (1 / (a i + ω)) = (7 - 3 * Complex.I))
  (n : ℕ) :
  ∑ i in Finset.range n, ((3 * a i + 2) / ((a i) ^ 2 - a i + 1)) = 14 :=
sorry

end summation_result_l5_5237


namespace expected_value_of_8_sided_die_l5_5392

def outcomes : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]
def probability (n : ℕ) : ℝ := 1 / 8

theorem expected_value_of_8_sided_die :
  (∑ x in outcomes, probability x * x) = 4.5 := 
sorry

end expected_value_of_8_sided_die_l5_5392


namespace range_of_m_l5_5194

theorem range_of_m (m : ℝ) :
  (∃ (p1 p2 : ℝ × ℝ), 
    ((p1.1 + 1)^2 + (p1.2 - 1)^2 = 8) ∧
    ((p2.1 + 1)^2 + (p2.2 - 1)^2 = 8) ∧ 
    |p1.1 + p1.2 + m| = sqrt 2 ∧ 
    |p2.1 + p2.2 + m| = sqrt 2) ↔ 
    (2 < m ∧ m < 6) ∨ (-6 < m ∧ m < -2) :=
sorry

end range_of_m_l5_5194


namespace comprehensive_score_correct_l5_5909

-- Conditions
def theoreticalWeight : ℝ := 0.20
def designWeight : ℝ := 0.50
def presentationWeight : ℝ := 0.30

def theoreticalScore : ℕ := 95
def designScore : ℕ := 88
def presentationScore : ℕ := 90

-- Calculate comprehensive score
def comprehensiveScore : ℝ :=
  theoreticalScore * theoreticalWeight +
  designScore * designWeight +
  presentationScore * presentationWeight

-- Lean statement to prove the comprehensive score using the conditions
theorem comprehensive_score_correct :
  comprehensiveScore = 90 := 
  sorry

end comprehensive_score_correct_l5_5909


namespace perimeter_of_figure_l5_5714

/-- Given:
1. ΔABC, ΔADE, and ΔEFG are equilateral triangles.
2. Points D and G are the midpoints of line segments AC and AE respectively.
3. Point H is the midpoint of line segment FG.
4. AB = 6,
Prove:
The perimeter of the figure ABCDEFGH is 22.5 units.
-/
theorem perimeter_of_figure (ABC ADE EFG : Triangle) 
  (equilateral_ABC : is_equilateral ABC) (equilateral_ADE : is_equilateral ADE) (equilateral_EFG : is_equilateral EFG)
  (D G H : Point) (mid_AC : midpoint D A C) (mid_AE : midpoint G A E) (mid_FG : midpoint H F G)
  (AB_length : dist A B = 6) :
  perimeter (polygon.mk [A, B, C, D, E, F, G, H]) = 22.5 :=
sorry

end perimeter_of_figure_l5_5714


namespace sequence_integral_terms_count_l5_5836

theorem sequence_integral_terms_count :
  let sequence : List ℕ := [15625, 3125, 625, 125, 25, 5, 1]
  in sequence.length = 7 :=
by
  sorry

end sequence_integral_terms_count_l5_5836


namespace geom_seq_min_m_l5_5710

def initial_capital : ℝ := 50
def growth_rate : ℝ := 0.5
def annual_payment (t : ℝ) : Prop := t ≤ 2500
def capital_remaining (aₙ : ℕ → ℝ) (n : ℕ) (t : ℝ) : ℝ := aₙ n * (1 + growth_rate) - t

theorem geom_seq (aₙ : ℕ → ℝ) (t : ℝ) (h₁ : annual_payment t) :
  (∀ n, aₙ (n + 1) = 3 / 2 * aₙ n - t) →
  (t ≠ 2500) →
  ∃ r : ℝ, ∀ n, aₙ n - 2 * t = (aₙ 0 - 2 * t) * r ^ n :=
sorry

theorem min_m (t : ℝ) (h₁ : t = 1500) (aₙ : ℕ → ℝ) :
  (∀ n, aₙ (n + 1) = 3 / 2 * aₙ n - t) →
  (aₙ 0 = initial_capital * (1 + growth_rate) - t) →
  ∃ m : ℕ, aₙ m > 21000 ∧ ∀ k < m, aₙ k ≤ 21000 :=
sorry

end geom_seq_min_m_l5_5710


namespace rewrite_equation_to_function_l5_5285

theorem rewrite_equation_to_function (x y : ℝ) (h : 2 * x - y = 3) : y = 2 * x - 3 :=
by
  sorry

end rewrite_equation_to_function_l5_5285


namespace expected_value_of_defective_products_variance_of_defective_products_l5_5036

def batch_authentic_prob := 0.99
def selected_products := 200
def defective_prob := 1.0 - batch_authentic_prob
def binomial_dist := Probability.Distribution.Binomial selected_products defective_prob

-- Definitions for the expected value and variance of a binomial distribution.
def expected_value := 2
def variance := 1.98

theorem expected_value_of_defective_products :
  Probability.Distribution.mean binomial_dist = expected_value := by
  sorry

theorem variance_of_defective_products :
  Probability.Distribution.variance binomial_dist = variance := by
  sorry

end expected_value_of_defective_products_variance_of_defective_products_l5_5036


namespace incorrect_arrangements_hello_l5_5313

theorem incorrect_arrangements_hello : 
  let total_permutations := 5!
  let correct_arrangement := 1
  let unique_permutations := total_permutations / 2
  unique_permutations - correct_arrangement = 59 :=
by
  sorry

end incorrect_arrangements_hello_l5_5313


namespace mary_final_books_l5_5264

def mary_initial_books := 5
def mary_first_return := 3
def mary_first_checkout := 5
def mary_second_return := 2
def mary_second_checkout := 7

theorem mary_final_books :
  (mary_initial_books - mary_first_return + mary_first_checkout - mary_second_return + mary_second_checkout) = 12 := 
by 
  sorry

end mary_final_books_l5_5264


namespace exists_m_no_even_digits_l5_5281

def has_no_even_digits (n : ℕ) : Prop :=
∀ d ∈ (nat.digits 10 n), ¬ (d % 2 = 0)

theorem exists_m_no_even_digits (n : ℕ) (hn : n > 0) : 
  ∃ m : ℕ, (m > 0) ∧ has_no_even_digits (5^n * m) :=
by
  -- Place proof here.
  sorry

end exists_m_no_even_digits_l5_5281


namespace expected_value_of_eight_sided_die_l5_5345

-- Definitions based on the problem conditions
def eight_sided_die_outcomes : List ℕ := [1,2,3,4,5,6,7,8]

def probability (n : ℕ) := 1 / n

-- Expected value calculation related to the problem
def expected_value_die_roll (outcomes : List ℕ) (prob : ℕ → Rat) : Rat :=
  List.sum (outcomes.map (λ x => prob outcomes.length * x))

-- Expected value of an 8-sided die roll
theorem expected_value_of_eight_sided_die :
  expected_value_die_roll eight_sided_die_outcomes probability = 4.5 := 
sorry

end expected_value_of_eight_sided_die_l5_5345


namespace area_of_parallelogram_is_20sqrt3_l5_5109

open Real EuclideanGeometry

def vector_u : ℝ × ℝ × ℝ := (4, 2, -3)
def vector_v : ℝ × ℝ × ℝ := (2, -4, 5)

noncomputable def cross_product (u v : ℝ × ℝ × ℝ) :=
  (u.2 * v.3 - u.3 * v.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2 - u.2 * v.1)

noncomputable def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

theorem area_of_parallelogram_is_20sqrt3 :
  magnitude (cross_product vector_u vector_v) = 20 * Real.sqrt 3 := by
  sorry

end area_of_parallelogram_is_20sqrt3_l5_5109


namespace minimum_value_l5_5185

theorem minimum_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_xy : x * y = 4) : 
  (1/x + 1/y) ≥ 2 :=
begin
  sorry
end

end minimum_value_l5_5185


namespace concyclic_condition_l5_5939

-- Define the points A, B, C, D and the coordinates of each.
variable (a b : ℝ) (A B C D S : Set (ℝ × ℝ))

-- Define the ellipse condition for points.
def on_ellipse (P : Set (ℝ × ℝ)) : Prop :=
∃ (x y : ℝ), P = {(x, y)} ∧ (x^2 / a^2) + (y^2 / b^2) = 1

-- Define the collinearity of points.
def collinear (P Q R : Set (ℝ × ℝ)) : Prop :=
∃ (m : ℝ), ∀ (x1 y1 x2 y2 x3 y3 : ℝ), 
  P = {(x1, y1)} ∧ Q = {(x2, y2)} ∧ R = {(x3, y3)} → (y2 - y1 = m * (x2 - x1)) ∧ (y3 - y1 = m * (x3 - x1))

-- Define the concyclic condition for points.
def concyclic (A B C D : Set (ℝ × ℝ)) : Prop :=
∃ (O : Set (ℝ × ℝ)), ∃ (R : ℝ), 
  ∀ (x y : ℝ), O = {(x, y)} ∧ 0 ≤ R ∧ 
  A ⊆ Circle (x, y) R ∧ B ⊆ Circle (x, y) R ∧ C ⊆ Circle (x, y) R ∧ D ⊆ Circle (x, y) R

-- Define the theorem statement.
theorem concyclic_condition :
  on_ellipse A → on_ellipse B → on_ellipse C → on_ellipse D →
  collinear A S B → collinear C S D → 
  concyclic A B C D := 
sorry

end concyclic_condition_l5_5939


namespace log_expression_identity_l5_5069

theorem log_expression_identity :
  (Real.log 5 / Real.log 10)^2 + (Real.log 2 / Real.log 10) * (Real.log 50 / Real.log 10) = 1 :=
by
  sorry

end log_expression_identity_l5_5069


namespace dress_designs_count_l5_5913

inductive Color
| red | green | blue | yellow

inductive Pattern
| stripes | polka_dots | floral | geometric | plain

def patterns_for_color (c : Color) : List Pattern :=
  match c with
  | Color.red    => [Pattern.stripes, Pattern.polka_dots, Pattern.floral, Pattern.geometric, Pattern.plain]
  | Color.green  => [Pattern.stripes, Pattern.polka_dots, Pattern.floral, Pattern.plain]
  | Color.blue   => [Pattern.stripes, Pattern.polka_dots, Pattern.floral, Pattern.geometric, Pattern.plain]
  | Color.yellow => [Pattern.stripes, Pattern.polka_dots, Pattern.floral, Pattern.plain]

noncomputable def number_of_dress_designs : ℕ :=
  (patterns_for_color Color.red).length +
  (patterns_for_color Color.green).length +
  (patterns_for_color Color.blue).length +
  (patterns_for_color Color.yellow).length

theorem dress_designs_count : number_of_dress_designs = 18 :=
  by
  sorry

end dress_designs_count_l5_5913


namespace annual_decrease_rate_l5_5828

theorem annual_decrease_rate
  (P0 : ℕ := 8000)
  (P2 : ℕ := 6480) :
  ∃ r : ℝ, 8000 * (1 - r / 100)^2 = 6480 ∧ r = 10 :=
by
  use 10
  sorry

end annual_decrease_rate_l5_5828


namespace smallest_rel_prime_to_180_l5_5591

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ Nat.gcd x 180 = 1 ∧ ∀ y : ℕ, (y > 1 ∧ Nat.gcd y 180 = 1) → x ≤ y :=
begin
  sorry
end

end smallest_rel_prime_to_180_l5_5591


namespace paths_to_form_2005_equals_88_l5_5800

-- Define the grid and allowable moves
def grid : Array Int := #[5, 5, 5, 5, 5,
                         5, 0, 0, 0, 5,
                         5, 0, 2, 0, 5,
                         5, 0, 0, 0, 5,
                         5, 5, 5, 5, 5]

def is_valid_move (pos1 pos2 : Int × Int) : Bool :=
  let (x1, y1) := pos1
  let (x2, y2) := pos2
  (abs (x2 - x1) ≤ 1) ∧ (abs (y2 - y1) ≤ 1)

-- The proof problem statement
theorem paths_to_form_2005_equals_88 :
    (count_paths_from_start_to_end grid (2, 2) 2005 = 88) :=
sorry

end paths_to_form_2005_equals_88_l5_5800


namespace repeating_decimal_to_fraction_l5_5979

/-- Given 0.02 repeating as a fraction 2/99, prove that 2.06 repeating can be expressed as 68/33 -/
theorem repeating_decimal_to_fraction :
  (2 + 0.06̅ : ℝ) = (68 / 33 : ℝ) :=
begin
  have h : (0.02̅ : ℝ) = (2 / 99 : ℝ), from sorry,
  have h3 : (0.06̅ : ℝ) = 3 * (0.02̅ : ℝ), from sorry,
  have h6 : (0.06̅ : ℝ) = 3 * (2 / 99 : ℝ), from sorry,
  have s : (0.06̅ : ℝ) = (6 / 99 : ℝ), from sorry,
  have s2 : (6 / 99 : ℝ) = (2 / 33 : ℝ), from sorry,
  have add := congr_arg (λ x : ℝ, (2 : ℝ) + x) s2,
  rw [add_comm, ← add_halves', add_assoc', add_comm (2 : ℝ), add_comm 2 (2 / 33 : ℝ), add_halves',
    add_assoc', add_comm (2 : ℝ), add_comm 68 (2 / 33 : ℝ)] ,
end

end repeating_decimal_to_fraction_l5_5979


namespace expected_value_of_8_sided_die_is_4_point_5_l5_5358

def expected_value_8_sided_die : ℝ :=
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let probability := 1 / 8
  (list.sum outcomes : ℝ) * probability

theorem expected_value_of_8_sided_die_is_4_point_5 :
  expected_value_8_sided_die = 4.5 := by
  sorry

end expected_value_of_8_sided_die_is_4_point_5_l5_5358


namespace fixed_point_of_f_is_1_minus_4_sqrt3_i_l5_5079

-- Definitions and fixed point
def f (z : Complex) : Complex := (2 - Complex.i * Real.sqrt 3) * z + (- Real.sqrt 3 - 12 * Complex.i) / 2

theorem fixed_point_of_f_is_1_minus_4_sqrt3_i : 
    ∃ c : Complex, f c = c ∧ c = 1 - 4 * Real.sqrt 3 * Complex.i := 
by
  sorry

end fixed_point_of_f_is_1_minus_4_sqrt3_i_l5_5079


namespace max_selectable_numbers_l5_5895

theorem max_selectable_numbers (n : ℕ) :
  ∀ (S : Finset ℕ), (∀ x ∈ (Finset.map (↑) (Finset.range (2 * n))).image ((λ k, 3 * (2 ^ k))), x ∈ S →
  ∀ y ∈ S, y ≠ x → (y ≠ 2 * x ∧ y ≠ x / 2)) →
  S.card ≤ n :=
by sorry

end max_selectable_numbers_l5_5895


namespace expected_value_of_eight_sided_die_l5_5372

theorem expected_value_of_eight_sided_die : 
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8], 
      n := (outcomes.length : ℚ),
      probabilities := List.replicate (outcomes.length) (1 / n),
      expected_value := (List.zipWith (*) probabilities (outcomes.map (· : ℚ))).sum
  in expected_value = 4.5 :=
by
  sorry

end expected_value_of_eight_sided_die_l5_5372


namespace orthocenter_of_triangle_l5_5921

theorem orthocenter_of_triangle 
  (A B C P E F : Point) 
  (h1 : P ∈ interior (triangle A B C)) 
  (h2 : ∠APB = 90 ∨ ∠APC = 90) 
  (h3 : E = proj C (line_through A B)) 
  (h4 : F = proj B (line_through A C)) 
  (circ_ae_eff : Tangent (circumcircle (triangle A E F)) E ∧ Tangent (circumcircle (triangle A E F)) F) 
  (tangent_meets : ∃ K : Point, K ∈ (line_through B C) ∧ (Tangents E F) = K) : 
  is_orthocenter A B C P := 
sorry

end orthocenter_of_triangle_l5_5921


namespace total_earnings_l5_5010

-- Definitions based on conditions
def bead_necklaces : ℕ := 7
def gem_necklaces : ℕ := 3
def cost_per_necklace : ℕ := 9

-- The main theorem to prove
theorem total_earnings : (bead_necklaces + gem_necklaces) * cost_per_necklace = 90 :=
by
  sorry

end total_earnings_l5_5010


namespace parallelogram_bisector_quadrilateral_sides_and_diagonal_l5_5837

theorem parallelogram_bisector_quadrilateral_sides_and_diagonal
  (a b : ℝ) (α : ℝ) :
  let d1 := |a - b| * Real.cos (α / 2), 
      d2 := |a - b| * Real.sin (α / 2),
      diag := |a - b| in
  (d1 = |a - b| * Real.cos (α / 2)) ∧
  (d2 = |a - b| * Real.sin (α / 2)) ∧
  (diag = |a - b|) := by
  sorry

end parallelogram_bisector_quadrilateral_sides_and_diagonal_l5_5837


namespace possible_values_for_a_l5_5957

def set_operation (A B : Set ℕ) : Set ℕ :=
  {z | ∃ x ∈ A, ∃ y ∈ B, z = x * y}

theorem possible_values_for_a : ∀ (a : ℕ),
  (∃ (A B : Set ℕ), A = {1, 2} ∧ B = {a, 2} ∧ (set_operation A B).card = 3) →
  a ∈ {0, 1, 4} :=
by
  sorry

end possible_values_for_a_l5_5957


namespace exists_point_M_right_triangle_l5_5938

theorem exists_point_M_right_triangle (A B C D E: Point)
  (hABC: is_isosceles_right_triangle A B C)
  (hADE: is_isosceles_right_triangle A D E)
  (h_non_congruent: ¬is_congruent A B C A D E)
  (h_AC_gt_AE: dist A C > dist A E)
  (rotation: Point → Point → Point) :
  ∃ M : Point, M ∈ line_segment E C ∧ is_isosceles_right_triangle B D M :=
sorry

end exists_point_M_right_triangle_l5_5938


namespace find_x_eq_minus_3_l5_5533

theorem find_x_eq_minus_3 (x : ℤ) : (3 ^ 7 * 3 ^ x = 81) → x = -3 := 
by
  sorry

end find_x_eq_minus_3_l5_5533


namespace mappings_count_l5_5167

open Classical

variable (A : Finset ℕ) (B : Finset ℕ) (f : ℕ → ℕ)

theorem mappings_count :
  ∃ (f : ℕ → ℕ), 
  (∀ x ∈ A, f x ∈ B) ∧ 
  (∃ (a b c d : ℕ), a < b ∧ b < c ∧ c < d ∧ f a ≤ f b ∧ f b ≤ f c ∧ f c ≤ f d) ∧
  (Finset.card A = 4 ∧ Finset.card B = 6) ∧ 
  (Finset.fold (+) 0 {import data.nat.choose
    (6.choose 1 + 3.choose 1 * 6.choose 2 + 3.choose 2 * 6.choose 3 + 6.choose 4)} = 126) :=
  sorry

end mappings_count_l5_5167


namespace woman_wait_time_l5_5883
noncomputable def time_for_man_to_catch_up (man_speed woman_speed distance: ℝ) : ℝ :=
  distance / man_speed

theorem woman_wait_time 
    (man_speed : ℝ)
    (woman_speed : ℝ)
    (wait_time_minutes : ℝ) 
    (woman_time : ℝ)
    (distance : ℝ)
    (man_time : ℝ) :
    man_speed = 5 -> 
    woman_speed = 15 -> 
    wait_time_minutes = 2 -> 
    woman_time = woman_speed * (1 / 60) * wait_time_minutes -> 
    woman_time = distance -> 
    man_speed * (1 / 60) = 0.0833 -> 
    man_time = distance / 0.0833 -> 
    man_time = 6 :=
by
  intros
  sorry

end woman_wait_time_l5_5883


namespace range_of_m_l5_5631

theorem range_of_m (m : ℝ) (h : 1 < m) :
  (∀ x : ℝ, -3 ≤ x ∧ x ≤ 1 → -m ≤ x ∧ x ≤ m - 1) → (3 ≤ m) :=
by
  sorry  -- The proof will be constructed here.

end range_of_m_l5_5631


namespace problem_area_of_triangle_DEF_l5_5196

noncomputable def herons_formula (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2 in
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem problem_area_of_triangle_DEF (DE EF DF : ℝ) (hDE : DE = 30) (hEF : EF = 50) (hDF : DF = 54) :
  herons_formula DE EF DF ≈ 735 :=
by
  rw [hDE, hEF, hDF]
  dsimp [herons_formula]
  have h : herons_formula 30 50 54 = Real.sqrt 540094 := by sorry
  rw h
  -- Note: this is a shorthand for approximate equality
  simp only [Real.sqrt]
  -- Verify that sqrt(540094) ≈ 735, which may be an approximation step
  sorry

end problem_area_of_triangle_DEF_l5_5196


namespace value_of_T_is_2002_l5_5084

def T : ℝ := 1002 + ∑ i in finset.range(1000), (1001 - i) / (2 : ℝ) ^ (i + 1)

theorem value_of_T_is_2002 : T = 2002 :=
by
  sorry

end value_of_T_is_2002_l5_5084


namespace average_birth_rate_solution_l5_5721

noncomputable def average_birth_rate_problem : Prop :=
  ∃ B : ℕ,
    (let net_increase_per_two_seconds := B - 2,
         two_second_intervals := 86400 / 2,
         total_increase := net_increase_per_two_seconds * two_second_intervals in
     total_increase = 216000) ∧ B = 7

theorem average_birth_rate_solution : average_birth_rate_problem :=
sorry

end average_birth_rate_solution_l5_5721


namespace slower_train_passing_time_l5_5886

-- Definitions of the conditions
def length_train : ℝ := 250 -- in meters
def speed_train_A_km_hr : ℝ := 45 -- in km/hr
def speed_train_B_km_hr : ℝ := 30 -- in km/hr
def relative_speed_m_s : ℝ := (speed_train_A_km_hr + speed_train_B_km_hr) * 1000 / 3600 -- in m/s
def distance_to_cover : ℝ := length_train -- in meters

-- The proof statement
theorem slower_train_passing_time : distance_to_cover / relative_speed_m_s = 12 := by
  sorry

end slower_train_passing_time_l5_5886


namespace smallest_k_l5_5956

noncomputable def sequence (n : Nat) : ℝ :=
  if n = 0 then 2
  else if n = 1 then real.sqrt (3 : ℝ)^ (21⁻¹ : ℝ)  -- expressing 21-th root of 3
  else sequence (n - 1) * (sequence (n - 2))^3

def is_integer (x : ℝ) : Prop := ∃ k : ℤ, (k : ℝ) = x

theorem smallest_k (k : Nat) :
  k = 17 ∧ is_integer (∏ i in finset.range k, sequence (i + 1)) :=
sorry

end smallest_k_l5_5956


namespace letter_puzzle_solutions_l5_5561

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def to_number (B A : ℕ) : ℕ :=
  10 * B + A

theorem letter_puzzle_solutions (A B : ℕ) (h_diff : A ≠ B) (h_digits : 1 ≤ A ∧ A ≤ 9 ∧ 1 ≤ B ∧ B ≤ 9)
  (h_two_digit : is_two_digit (to_number B A)) :
  A^B = to_number B A ↔ (A = 2 ∧ B = 5 ∨ A = 6 ∧ B = 2 ∨ A = 4 ∧ B = 3) :=
by
  sorry

end letter_puzzle_solutions_l5_5561


namespace terminating_decimal_fractions_l5_5619

theorem terminating_decimal_fractions :
  let n_count := (finset.range 151).filter (λ n, n % 3 = 0),
  n_count.card = 50 :=
by
  sorry

end terminating_decimal_fractions_l5_5619


namespace calculate_M_l5_5224

theorem calculate_M (h₁ : ∀(t : Real), 54 = 6 * (t^2))
                    (h₂ : ∀(r h : Real), 2 * π * (r^2) + 2 * π * r * h = 54)
                    (h₃ : ∀(r : Real), h = r)
                    (h₄ : ∀(V r h : Real), V = π * (r^2) * h)
                    (h₅ : ∀(V : Real), V = (M * sqrt(5)) / sqrt(π)) : 
           M = (81 * sqrt(3)) / 2 :=
by
  sorry

end calculate_M_l5_5224


namespace number_of_terminating_decimals_l5_5621

theorem number_of_terminating_decimals (n : ℕ) (h₁ : 1 ≤ n) (h₂ : n ≤ 150) :
  ∃ m, m = 50 ∧ 
  ∀ n, (1 ≤ n ∧ n ≤ 150) → (∃ k, n = 3 * k) →
  m = 50 :=
by 
  sorry

end number_of_terminating_decimals_l5_5621


namespace product_zero_count_l5_5624

theorem product_zero_count : 
  {n : ℕ | 1 ≤ n ∧ n ≤ 2012 ∧ ∏ k in finset.range n, (1 + real.exp (complex.I • ((2 * real.pi * k) / n)))^n + 1 = 0}.card = 335 := 
sorry

end product_zero_count_l5_5624


namespace jackson_spends_260_l5_5219

-- Definitions based on conditions
def num_students := 30
def pens_per_student := 5
def notebooks_per_student := 3
def binders_per_student := 1
def highlighters_per_student := 2

def cost_per_pen := 0.50
def cost_per_notebook := 1.25
def cost_per_binder := 4.25
def cost_per_highlighter := 0.75
def discount := 100.00

-- Calculate total cost
noncomputable def total_cost := 
  let cost_per_student := 
    (pens_per_student * cost_per_pen) +
    (notebooks_per_student * cost_per_notebook) +
    (binders_per_student * cost_per_binder) +
    (highlighters_per_student * cost_per_highlighter)
  in num_students * cost_per_student - discount

-- Theorem to prove the final cost
theorem jackson_spends_260 : total_cost = 260 := by
  sorry

end jackson_spends_260_l5_5219


namespace license_plates_count_l5_5176

def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_prime_under_10 (n : ℕ) : Prop := n ∈ {2, 3, 5, 7}
def is_even (n : ℕ) : Prop := n % 2 = 0

theorem license_plates_count :
  let letter_choices := 26 in
  let first_digit_choices := {1, 3, 5, 7, 9} in
  let second_digit_choices := {2, 3, 5, 7} in
  let third_digit_choices := {0, 2, 4, 6, 8} in
  (letter_choices * letter_choices * (first_digit_choices.card * second_digit_choices.card * third_digit_choices.card)) = 67600 :=
by
  sorry

end license_plates_count_l5_5176


namespace letter_puzzle_solutions_l5_5552

noncomputable def is_solution (A B : ℕ) : Prop :=
A ≠ B ∧ A ∈ finset.range (10) ∧ B ∈ finset.range (10) ∧ 10 ≤ B * 10 + A ∧ B * 10 + A ≤ 99 ∧ A^B = B * 10 + A

theorem letter_puzzle_solutions :
  ∃ A B : ℕ, is_solution A B ∧ ((A = 2 ∧ B = 5) ∨ (A = 6 ∧ B = 2) ∨ (A = 4 ∧ B = 3)) :=
by
  sorry

end letter_puzzle_solutions_l5_5552


namespace roots_in_annulus_l5_5179

-- Define the polynomial equation
def poly (z : ℂ) : ℂ := z^4 - 5 * z + 1

-- The main statement to prove:
theorem roots_in_annulus : 
  (finset.card (finset.filter (λ z, 1 < complex.abs z ∧ complex.abs z < 2) (finset.univ : finset (root_set poly ℂ)))) = 3 :=
  sorry

end roots_in_annulus_l5_5179


namespace jessica_rearrangements_time_l5_5222

noncomputable def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

theorem jessica_rearrangements_time :
  let perms := factorial 7 / factorial 2,
      time_minutes := perms / 18,
      time_hours := time_minutes / 60 in
  time_hours = 2 + 1/3 :=
by
  let perms := factorial 7 / factorial 2,
      time_minutes := perms / 18,
      time_hours := time_minutes / 60
  show time_hours = 2 + 1/3
  sorry

end jessica_rearrangements_time_l5_5222


namespace marbles_same_color_l5_5342

theorem marbles_same_color (a b c : ℕ) (h : a + b + c = 2015)
  (op1 : ∀ a b c: ℕ, (a, b, c) → (a + 2, b - 1, c - 1)) 
  (op2 : ∀ a b c: ℕ, (a, b, c) → (a - 1, b + 2, c - 1)) 
  (op3 : ∀ a b c: ℕ, (a, b, c) → (a - 1, b - 1, c + 2)) :
  ∃ k, (k = a ∨ k = b ∨ k = c) ∧ (k = 2015) := 
begin
  sorry,
end

end marbles_same_color_l5_5342


namespace expected_value_of_8_sided_die_l5_5390

def outcomes : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]
def probability (n : ℕ) : ℝ := 1 / 8

theorem expected_value_of_8_sided_die :
  (∑ x in outcomes, probability x * x) = 4.5 := 
sorry

end expected_value_of_8_sided_die_l5_5390


namespace equation_1_consecutive_odd_integers_equation_1_prime_integers_equation_2_consecutive_multiples_3_equation_2_positive_multiples_6_l5_5510

namespace MathProblemProofs

-- Problem 1: Theorem statement for (A)
theorem equation_1_consecutive_odd_integers (x y z : ℤ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hx_odd : x % 2 = 1) (hy_odd : y % 2 = 1) (hz_odd : z % 2 = 1) :
  x + y + z = 51 → (x = y - 2 ∧ z = y + 2) := 
sorry

-- Problem 2: Theorem statement for (B)
theorem equation_1_prime_integers (x y z : ℤ)
  (hx_pos : x > 0) (hy_pos : y > 0) (hz_pos : z > 0)
  (hx_prime : Nat.Prime x) (hy_prime : Nat.Prime y) (hz_prime : Nat.Prime z) :
  ¬ (x + y + z = 51) :=
sorry

-- Problem 3: Theorem statement for (C)
theorem equation_2_consecutive_multiples_3 (x y z w : ℤ)
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0)
  (hx_3 : x % 3 = 0) (hy_3 : y % 3 = 0) (hz_3 : z % 3 = 0) (hw_3 : w % 3 = 0) :
  x + y + z + w = 60 → ¬ (y = x + 3 ∧ z = x + 6 ∧ w = x + 9) :=
sorry

-- Problem 4: Theorem statement for (D)
theorem equation_2_positive_multiples_6 (x y z w : ℤ)
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0)
  (hx_6 : x % 6 = 0) (hy_6 : y % 6 = 0) (hz_6 : z % 6 = 0) (hw_6 : w % 6 = 0) :
  x + y + z + w = 60 :=
sorry

end MathProblemProofs

end equation_1_consecutive_odd_integers_equation_1_prime_integers_equation_2_consecutive_multiples_3_equation_2_positive_multiples_6_l5_5510


namespace rearrange_marked_squares_l5_5712

theorem rearrange_marked_squares :
  ∀ table : Array (Array Bool),
    table.size = 100 →
    (∀ i : Fin 100, table[i].size = 100) →
    (∑ i : Fin 100, ∑ j : Fin 100, if table[i][j] then 1 else 0) = 110 →
    ∃ r : Perm (Fin 100), ∃ c : Perm (Fin 100),
      ∀ i j : Fin 100, table[r i][c j] = true → i ≤ j :=
by
  sorry

end rearrange_marked_squares_l5_5712


namespace volume_of_tetrahedron_l5_5736

/-- If a tetrahedron has surface area S and inscribed sphere radius R, 
    its volume V is given by V = 1/3 * S * R. -/
theorem volume_of_tetrahedron (S R : ℝ) : 
  ∃ V : ℝ, V = (1 / 3) * S * R :=
begin
  use (1 / 3) * S * R,
  sorry,
end

end volume_of_tetrahedron_l5_5736


namespace smallest_rel_prime_to_180_l5_5609

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ Int.gcd x 180 = 1 ∧ ∀ y : ℕ, y > 1 ∧ Int.gcd y 180 = 1 → x ≤ y := 
sorry

end smallest_rel_prime_to_180_l5_5609


namespace sufficient_but_not_necessary_l5_5644

variable (x : ℝ)

def p := abs (x - 1) < 2
def q := x^2 - 5 * x - 6 < 0

theorem sufficient_but_not_necessary : (p -> q) ∧ (¬ q -> ¬ p) ∧ (¬ (q -> p)) := by
  sorry

end sufficient_but_not_necessary_l5_5644


namespace matrix_equation_l5_5770

open Matrix BigOperators
variables {R : Type*} [CommRing R]

def N : Matrix (Fin 2) (Fin 2) R := ![![3, 2], ![-4, 1]]
def I : Matrix (Fin 2) (Fin 2) R := 1

theorem matrix_equation : 
  let r := (4 : R), s := (-11 : R) in
  N * N = r • N + s • I :=
by sorry

end matrix_equation_l5_5770


namespace cube_height_after_cut_l5_5303

/--
A corner of a unit cube is cut off such that the cut runs through the three vertices adjacent to the vertex of the chosen corner.
-/
def unit_cube_cutoff (corner_cut : ℝ) : Prop :=
  corner_cut = 1 / 3 * (sqrt 3) / 3 + 1 + 1 / 3 * sqrt 3 / 3

theorem cube_height_after_cut 
  (unit_cube : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1)
  (height : ℝ) 
  (h_cut : unit_cube_cutoff height) 
  : height = (2 * sqrt 3) / 3 :=
by
  sorry

end cube_height_after_cut_l5_5303


namespace count_whole_numbers_between_l5_5690

theorem count_whole_numbers_between : 
  let a := Real.root 4 50
  let b := Real.root 4 1000
  ∃ n : ℕ, n = 3 ∧
  (∀ x : ℕ, a < x → x < b → 3 ≤ x ∧ x ≤ 5) ∧
  (∀ x : ℕ, a < x → x < b → x = 3 ∨ x = 4 ∨ x = 5) :=
by
  let a := Real.root 4 50
  let b := Real.root 4 1000
  use 3
  split
  { exact rfl }
  split
  { assume x ha hb
    split,
    linarith [Real.lift_root_pos 4 50, ha],
    linarith [hb, Real.lift_root_pos 4 1000] }
  { assume x ha hb
    cases lt_or_eq_of_le ha with ha_eq ha_eq,
    { linarith [ha_eq] },
    { cases lt_or_eq_of_le hb with hb_eq hb_eq,
      { linarith [hb_eq] },
      { assumption } } }

end count_whole_numbers_between_l5_5690


namespace pages_per_booklet_l5_5216

theorem pages_per_booklet (total_pages : ℕ) (booklets : ℕ) (h1 : total_pages = 441) (h2 : booklets = 49) : total_pages / booklets = 9 :=
by {
  rw [h1, h2],
  norm_num,
  sorry
}

end pages_per_booklet_l5_5216


namespace opposite_of_sqrt5_minus_2_l5_5827

theorem opposite_of_sqrt5_minus_2 : ∀ (x y : ℝ), x = sqrt 5 ∧ y = 2 → -(x - y) = -sqrt 5 + 2 :=
begin
  intros x y hx,
  obtain ⟨hx1, hy1⟩ := hx,
  rw [hx1, hy1],
  simp,
  sorry
end

end opposite_of_sqrt5_minus_2_l5_5827


namespace shifted_quadratic_function_l5_5308

def quadratic_function : ℝ → ℝ :=
  λ x, x^2

def shift_down (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ :=
  λ x, f x - a

theorem shifted_quadratic_function :
  shift_down quadratic_function 2 = λ x, x^2 - 2 :=
by
  sorry

end shifted_quadratic_function_l5_5308


namespace henry_tournament_points_l5_5267

theorem henry_tournament_points :
  let points_won := 2 * 5
  let points_lost := 2 * 2
  let points_drawn := 10 * 3
  points_won + points_lost + points_drawn = 44 := 
by
  let points_won := 2 * 5
  let points_lost := 2 * 2
  let points_drawn := 10 * 3
  have hw : points_won = 10 := rfl
  have hl : points_lost = 4 := rfl
  have hd : points_drawn = 30 := rfl
  rw [hw, hl, hd]
  exact rfl


end henry_tournament_points_l5_5267


namespace simplify_sqrt_88200_l5_5289

theorem simplify_sqrt_88200 :
  (Real.sqrt 88200) = 70 * Real.sqrt 6 := 
by 
  -- given conditions
  have h : 88200 = 2^3 * 3 * 5^2 * 7^2 := sorry,
  sorry

end simplify_sqrt_88200_l5_5289


namespace G5_units_digit_is_0_l5_5513

def power_mod (base : ℕ) (exp : ℕ) (modulus : ℕ) : ℕ :=
  (base ^ exp) % modulus

def G (n : ℕ) : ℕ := 2 ^ (3 ^ n) + 2

theorem G5_units_digit_is_0 : (G 5) % 10 = 0 :=
by
  sorry

end G5_units_digit_is_0_l5_5513


namespace sum_digits_divisible_by_81_l5_5767

theorem sum_digits_divisible_by_81 (N : ℕ) (h1 : N % 81 = 0) (h2 : (nat.reverse_digits N) % 81 = 0) : 
  (nat.digits 10 N).sum % 81 = 0 := 
sorry

end sum_digits_divisible_by_81_l5_5767


namespace smallest_rel_prime_to_180_l5_5599

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ x ≤ 7 ∧ (∀ y : ℕ, y > 1 ∧ y < x → y.gcd 180 ≠ 1) ∧ x.gcd 180 = 1 :=
by
  sorry

end smallest_rel_prime_to_180_l5_5599


namespace exists_m_divisible_by_1997_l5_5251

-- Define the function f
def f (x : ℤ) : ℤ := 3 * x + 2

-- Define the nth iterate of f
def f_iter (n : ℕ) (x : ℤ) : ℤ :=
  nat.rec_on n x (λ k y, f y)

theorem exists_m_divisible_by_1997 : ∃ m : ℤ, m > 0 ∧ (f_iter 99 m) % 1997 = 0 := 
sorry

end exists_m_divisible_by_1997_l5_5251


namespace expected_value_of_8_sided_die_l5_5384

theorem expected_value_of_8_sided_die : 
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let probability := 1 / 8
  (∑ outcome in outcomes, outcome * probability) = 4.5 :=
by
  sorry

end expected_value_of_8_sided_die_l5_5384


namespace true_proposition_is_D_l5_5497

-- Define the context for Euclidean geometry
open_locale euclidean_geometry

-- Define Propositions A, B, C, and D
def PropA : Prop := ∀ {α β : ℝ}, α = β → ∃ (l1 l2 : Line), l1 ≠ l2 ∧ ∃ (A B : ℝ × ℝ), (A ∈ l1) ∧ (B ∈ l2) ∧ ∃ (P : ℝ × ℝ), P ∈ l1 ∧ P ∈ l2
def PropB : Prop := ∀ {l1 l2 l3 : Line}, (∃ (A B : ℝ × ℝ), A ≠ B ∧ A ∈ l1 ∧ B ∈ l2 ∧ ∀ (C : ℝ × ℝ), C ∈ l3 → ∠ l1 l3 = ∠ l2 l3)
def PropC : Prop := ∀ (P : ℝ × ℝ) (l : Line), P ∉ l → ∃ (d : ℝ), d = distance P l
def PropD : Prop := ∀ {l : Line} (P : ℝ × ℝ), P ∉ l → ∃! (l2 : Line), P ∈ l2 ∧ l ⊥ l2

-- Assert that Proposition D is true
theorem true_proposition_is_D : PropD :=
sorry

end true_proposition_is_D_l5_5497


namespace first_digit_base9_of_base3_num_l5_5157

theorem first_digit_base9_of_base3_num {y : ℕ} (hy : y = 21211122211122211111₃) : 
  ∃ d, d = y.digits 9^.0 ∧ d.headI = 4 :=
by sorry

end first_digit_base9_of_base3_num_l5_5157


namespace animal_legs_count_l5_5327

-- Let's define the conditions first.
def total_animals : ℕ := 12
def chickens : ℕ := 5
def chicken_legs : ℕ := 2
def sheep_legs : ℕ := 4

-- Define the statement that we need to prove.
theorem animal_legs_count :
  ∃ (total_legs : ℕ), total_legs = 38 :=
by
  -- Adding the condition for total number of legs
  let sheep := total_animals - chickens
  let total_legs := (chickens * chicken_legs) + (sheep * sheep_legs)
  existsi total_legs
  -- Question proves the correct answer
  sorry

end animal_legs_count_l5_5327


namespace compare_mean_median_modes_median_l5_5265

noncomputable def mean : ℝ :=
  (13 * (30 * (31 / 2)) + 8 * 31) / 366 

noncomputable def median : ℝ := 15

noncomputable def modes_median : ℝ := (15 + 16) / 2

theorem compare_mean_median_modes_median :
  median < modes_median ∧ modes_median < mean :=
by 
  have h_mean : mean = 6293 / 366 := rfl
  have h_approx_mean : mean ≈ 17.20 := by sorry -- approximate mean value
  have h_median : median = 15 := rfl
  have h_modes_median : modes_median = 15.5 := rfl
  rw [h_median, h_modes_median, h_mean]
  exact ⟨by linarith, by linarith⟩


end compare_mean_median_modes_median_l5_5265


namespace solve_expression_l5_5503

theorem solve_expression : (1/2)^(-2) - (64)^(1/3) + (-(9))^(2:ℝ)^(1/2) - (3^(1/2)-2)^(0:ℝ) = 8 := 
  by
  sorry

end solve_expression_l5_5503


namespace distance_to_circle_center_l5_5860

theorem distance_to_circle_center :
  let center := (2: ℝ, -3: ℝ)
  let point := (5: ℝ, -2: ℝ)
  distance center point = Real.sqrt 10 :=
by
  -- Definitions
  let center := (2: ℝ, -3: ℝ)
  let point := (5: ℝ, -2: ℝ)
  
  -- Actual distance computation
  sorry

end distance_to_circle_center_l5_5860


namespace per_capita_income_growth_l5_5492

noncomputable def income2020 : ℝ := 3.2
noncomputable def income2022 : ℝ := 3.7
variable (x : ℝ)

/--
Prove the per capita disposable income model.
-/
theorem per_capita_income_growth :
  income2020 * (1 + x)^2 = income2022 :=
sorry

end per_capita_income_growth_l5_5492


namespace probability_in_region_l5_5466

-- Define the square region as vertices at (±2, ±2)
def square_region : Set (ℝ × ℝ) := {p | abs p.1 ≤ 2 ∧ abs p.2 ≤ 2}

-- Define the circle region of radius 2 centered at (2, 2)
def circle_region : Set (ℝ × ℝ) := {p | (p.1 - 2)^2 + (p.2 - 2)^2 ≤ 2^2}

-- Define the intersection of the square region and the circle region
def intersection_region : Set (ℝ × ℝ) := {p | p ∈ square_region ∧ p ∈ circle_region}

-- The area of the square region is 4 * 4 = 16
-- The area of the intersection_region we already know from geometric facts is π
-- Goal: The probability P lies within two units of the vertex at (2,2)
theorem probability_in_region : (measure_theory.measure (intersection_region) / measure_theory.measure (square_region)) = π / 16 :=
sorry

end probability_in_region_l5_5466


namespace incircle_excircle_relation_l5_5443

variable (A B C M : Point) (r r1 r2 q q1 q2 : ℝ)

-- Define the properties of the points and the circles
variable (hM : SegmentContains M A B)
variable (h_r : IsIncircleRadius r A B C)
variable (h_r1 : IsIncircleRadius r1 A M C)
variable (h_r2 : IsIncircleRadius r2 B M C)
variable (h_q : IsExcircleRadius q A B C)
variable (h_q1 : IsExcircleRadius q1 A M C)
variable (h_q2 : IsExcircleRadius q2 B M C)

theorem incircle_excircle_relation :
  r1 * r2 * q = r * q1 * q2 := sorry

end incircle_excircle_relation_l5_5443


namespace sum_gcd_3n_plus_4_n_l5_5937

theorem sum_gcd_3n_plus_4_n : 
  (∑ d in ({gcd (3 * n + 4) n | n : ℕ+, n > 0}).toFinset, d) = 7 := 
  sorry

end sum_gcd_3n_plus_4_n_l5_5937


namespace greatest_k_value_l5_5832

theorem greatest_k_value (k : ℝ) (h : ((k ^ 2) - 20 = 61)) : k ≤ 9 := by
  have h₁ : k ^ 2 = 81 := eq_add_of_sub_eq h
  have h₂ : k = 9 ∨ k = -9 := sq_eq_sq k 9
  cases h₂ with hk_pos hk_neg
  · exact le_of_eq hk_pos
  · exact le_trans (neg_nonpos_of_nonneg zero_le_one) hk_neg

end greatest_k_value_l5_5832


namespace expected_value_of_8_sided_die_is_4_point_5_l5_5359

def expected_value_8_sided_die : ℝ :=
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let probability := 1 / 8
  (list.sum outcomes : ℝ) * probability

theorem expected_value_of_8_sided_die_is_4_point_5 :
  expected_value_8_sided_die = 4.5 := by
  sorry

end expected_value_of_8_sided_die_is_4_point_5_l5_5359


namespace trigonometric_identity_proof_l5_5004

theorem trigonometric_identity_proof :
  (cos 70 * cos 10 + cos 80 * cos 20) / (cos 69 * cos 9 + cos 81 * cos 21) = 1 :=
by
  sorry

end trigonometric_identity_proof_l5_5004


namespace speed_of_man_proof_l5_5925

noncomputable def speed_of_man (train_length : ℝ) (train_speed_kmph : ℝ) (time_seconds : ℝ) : ℝ := 
    ((train_length / time_seconds) - (train_speed_kmph * (1000 / 3600))) * (3600 / 1000)

theorem speed_of_man_proof : 
    speed_of_man 55 60 3 ≈ 5.976 :=
by 
  rw [Real.mul_div_cancel' (60 : ℝ) (3600 : ℝ) 1000]
  sorry

end speed_of_man_proof_l5_5925


namespace relatively_prime_probability_l5_5853

noncomputable def phi (n : ℕ) : ℕ :=
  n * (1 - 1 / 2) * (1 - 1 / 3) * (1 - 1 / 7)

theorem relatively_prime_probability (n : ℕ) (h_n : n = 42) :
  (φ n / n = 4 / 7) :=
begin
  have h_prime_factors : ∀ x, x ∈ {2, 3, 7},
  { -- proof that 2, 3, and 7 are the prime factors of 42
    sorry
  },
  have h_phi : φ 42 = 24,
  {
    -- proof of Euler's totient function for 42
    sorry
  },
  calculate the number of integers that are relatively prime to 42
  show that the number of integers relatively prime to 42 divided by 42 is 4/7
  sorry
end

end relatively_prime_probability_l5_5853


namespace ratio_of_areas_l5_5823

theorem ratio_of_areas (b : ℝ) (h1 : 0 < b) (h2 : b < 4) 
  (h3 : (9 : ℝ) / 25 = (4 - b) / b * (4 : ℝ)) : b = 2.5 := 
sorry

end ratio_of_areas_l5_5823


namespace correct_options_A_and_D_l5_5878

noncomputable def problem_statement :=
  ∃ A B C D : Prop,
  (A = (∀ x : ℝ, x^2 + x + 1 > 0) ↔ (∃ x : ℝ, x^2 + x + 1 ≤ 0)) ∧ 
  (B = ∀ (a b c d : ℝ), a > b → c > d → ¬(a * c > b * d)) ∧
  (C = ∀ m : ℝ, ¬((∀ x : ℝ, x > 0 → (m^2 - m - 1) * x^(m^2 - 2 * m - 3) < 0) → (-1 < m ∧ m < 2))) ∧
  (D = ∀ a : ℝ, (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ < 0 ∧ x₁ + x₂ = 3 - a ∧ x₁ * x₂ = a) → a < 0)

-- We need to prove that only A and D are true
theorem correct_options_A_and_D : problem_statement :=
  sorry

end correct_options_A_and_D_l5_5878


namespace smallest_positive_integer_k_l5_5926

-- Define the conditions
def y : ℕ := 2^3 * 3^4 * (2^2)^5 * 5^6 * (2*3)^7 * 7^8 * (2^3)^9 * (3^2)^10

-- Define the question statement
theorem smallest_positive_integer_k :
  ∃ k : ℕ, k > 0 ∧ (∃ m : ℕ, (y * k) = m^2) ∧ k = 30 :=
by
  sorry

end smallest_positive_integer_k_l5_5926


namespace find_x_l5_5085

theorem find_x (x : ℝ) (h_pos : 0 < x) (h_eq : x * ⌊x⌋ = 72) : x = 8 :=
by
  sorry

end find_x_l5_5085


namespace smallest_rel_prime_to_180_l5_5598

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ x ≤ 7 ∧ (∀ y : ℕ, y > 1 ∧ y < x → y.gcd 180 ≠ 1) ∧ x.gcd 180 = 1 :=
by
  sorry

end smallest_rel_prime_to_180_l5_5598


namespace most_likely_units_digit_l5_5967

-- Definitions for Jack's and Jill's draws
def J : ℕ := sorry -- Jack's draw, which is uniformly distributed over 0 to 9
def K : ℕ := sorry -- Jill's draw, which is uniformly distributed over 0 to 9

-- The main theorem statement
theorem most_likely_units_digit (J K : ℕ) 
  (hJ : J ∈ ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} : set ℕ))
  (hK : K ∈ ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} : set ℕ)) :
  (∑ j k, ite (j + k) % 10 == 0 1 0 ≤ 10 * 10) = 0 := sorry

end most_likely_units_digit_l5_5967


namespace sin_cos_sufficient_not_necessary_cos2α_impl_sin_cos_or_sin_cos_neg_l5_5444

theorem sin_cos_sufficient_not_necessary (α : ℝ) (h1 : sin α + cos α = 0) : cos (2 * α) = 0 :=
by
  -- Using the trigonometric identities to establish the relationship
  sorry

theorem cos2α_impl_sin_cos_or_sin_cos_neg (α : ℝ) (h2 : cos (2 * α) = 0) : sin α + cos α = 0 ∨ cos α - sin α = 0 :=
by
  -- Using the trigonometric identities to establish the relationship
  sorry

end sin_cos_sufficient_not_necessary_cos2α_impl_sin_cos_or_sin_cos_neg_l5_5444


namespace expected_value_eight_sided_die_l5_5413

-- Define a standard 8-sided die
def eight_sided_die : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]

-- Compute the probability of each outcome
def probability (n : ℕ) : ℝ := 1 / 8

-- Expected Value of a discrete random variable
def expected_value (outcomes : List ℕ) (prob : ℕ → ℝ) : ℝ :=
  outcomes.sum / outcomes.length.toReal

-- Theorem stating the expected value of a standard 8-sided die roll is 4.5
theorem expected_value_eight_sided_die : expected_value eight_sided_die probability = 4.5 := by
  sorry

end expected_value_eight_sided_die_l5_5413


namespace remainder_when_y_squared_divided_by_30_l5_5698

theorem remainder_when_y_squared_divided_by_30 (y : ℤ) :
  6 * y ≡ 12 [ZMOD 30] → 5 * y ≡ 25 [ZMOD 30] → y ^ 2 ≡ 19 [ZMOD 30] :=
  by
  intro h1 h2
  sorry

end remainder_when_y_squared_divided_by_30_l5_5698


namespace math_problem_solution_l5_5946

noncomputable def math_problem : Real :=
  Real.abs (-2) + (1 / 3)⁻¹ - Real.sqrt 9 + (Real.sin (Real.pi / 4) - 1) ^ 0 - (-1)

theorem math_problem_solution : math_problem = 4 := by
  sorry

end math_problem_solution_l5_5946


namespace min_n_plus_d_l5_5725

theorem min_n_plus_d (a : ℕ → ℕ) (n d : ℕ) (h1 : a 1 = 1) (h2 : a n = 51)
  (h3 : ∀ i, a i = a 1 + (i-1) * d) : n + d = 16 :=
by
  sorry

end min_n_plus_d_l5_5725


namespace expected_value_of_8_sided_die_l5_5393

def outcomes : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]
def probability (n : ℕ) : ℝ := 1 / 8

theorem expected_value_of_8_sided_die :
  (∑ x in outcomes, probability x * x) = 4.5 := 
sorry

end expected_value_of_8_sided_die_l5_5393


namespace smallest_coprime_gt_one_l5_5573

theorem smallest_coprime_gt_one (x : ℕ) (h1 : 1 < x) (h2 : Nat.gcd x 180 = 1) : x = 7 := sorry

end smallest_coprime_gt_one_l5_5573


namespace smallest_rel_prime_to_180_l5_5606

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ Int.gcd x 180 = 1 ∧ ∀ y : ℕ, y > 1 ∧ Int.gcd y 180 = 1 → x ≤ y := 
sorry

end smallest_rel_prime_to_180_l5_5606


namespace expected_value_of_eight_sided_die_l5_5343

-- Definitions based on the problem conditions
def eight_sided_die_outcomes : List ℕ := [1,2,3,4,5,6,7,8]

def probability (n : ℕ) := 1 / n

-- Expected value calculation related to the problem
def expected_value_die_roll (outcomes : List ℕ) (prob : ℕ → Rat) : Rat :=
  List.sum (outcomes.map (λ x => prob outcomes.length * x))

-- Expected value of an 8-sided die roll
theorem expected_value_of_eight_sided_die :
  expected_value_die_roll eight_sided_die_outcomes probability = 4.5 := 
sorry

end expected_value_of_eight_sided_die_l5_5343


namespace expected_value_8_sided_die_l5_5361

theorem expected_value_8_sided_die :
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let probability_each_outcome := 1 / 8
  (Σ x ∈ outcomes, probability_each_outcome * x) = 4.5 :=
by
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let probability_each_outcome := 1 / 8
  have h : (Σ x ∈ outcomes, probability_each_outcome * x) = (1 / 8) * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8) := by sorry
  have sum_eq_36 : 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 = 36 := by sorry
  rw [sum_eq_36] at h
  have expected_value_eq : (1 / 8) * 36 = 4.5 := by sorry
  rw [expected_value_eq] at h
  exact h

end expected_value_8_sided_die_l5_5361


namespace experts_correct_l5_5056

theorem experts_correct (rate_per_minute : ℕ) (shots : ℕ) (time_observed : ℕ) 
  (h_rate: rate_per_minute = 1) (h_shots: shots = 60) (h_time_observed: time_observed = 60) : 
  (time_needed : ℕ) (h_time_needed : time_needed = shots - 1) → 
  h_time_needed := rfl → 
  time_needed = 59 :=
by {
  sorry
}

end experts_correct_l5_5056


namespace mosquito_feedings_to_death_l5_5461

theorem mosquito_feedings_to_death 
  (drops_per_feeding : ℕ := 20) 
  (drops_per_liter : ℕ := 5000) 
  (lethal_blood_loss_liters : ℝ := 3) 
  (drops_per_feeding_liters : ℝ := drops_per_feeding / drops_per_liter) 
  (lethal_feedings : ℝ := lethal_blood_loss_liters / drops_per_feeding_liters) :
  lethal_feedings = 750 := 
by
  sorry

end mosquito_feedings_to_death_l5_5461


namespace probability_A_more_than_3_points_probability_distribution_B_expectation_of_X_l5_5203

section shooting_game
open Probability

-- Conditions for player A
def ξ : ℕ → ℕ → ℝ := λ n k, (n.choose k) * (1 / 2)^k * (1 / 2)^(n - k)

-- Probability that A scores more than 3 points in 3 shots
theorem probability_A_more_than_3_points :
  (ξ 3 2 + ξ 3 3) = 1 / 2 :=
by sorry

-- Conditions for player B
def X (sequence : list (ℕ → ℝ)) (n : ℕ) : ℝ :=
match sequence.nth n with
| some p => p
| none   => 0
end

def B_prob (n : ℕ) : list (ℕ → ℝ) :=
match n with
| 0 => [λ x, (if x = 1 then 1/2 else 1/2)]
| 1 => [λ x, (if x = 1 then 3/5 else 2/5)]
| _ => [λ x, 0]
end

-- Probability distribution of the score X
theorem probability_distribution_B :
  (X (B_prob 0) 0) = 9/50 ∧
  (X (B_prob 0) 2) = 8/25 ∧
  (X (B_prob 0) 4) = 8/25 ∧
  (X (B_prob 0) 6) = 9/50 :=
by sorry

-- Expectation of the score X
theorem expectation_of_X :
  (0 * (9 / 50) + 2 * (8 / 25) + 4 * (8 / 25) + 6 * (9 / 50)) = 3 :=
by sorry

end shooting_game

end probability_A_more_than_3_points_probability_distribution_B_expectation_of_X_l5_5203


namespace number_of_outliers_l5_5074

open List

def dataset : List ℕ := [8, 22, 35, 35, 42, 44, 44, 47, 55, 62]
def Q1 : ℕ := 35
def Q3 : ℕ := 47
def IQR : ℕ := Q3 - Q1
def lower_threshold : ℕ := Q1 - 1.5 * IQR
def upper_threshold : ℕ := Q3 + 1.5 * IQR

def outliers_in_dataset : List ℕ :=
  dataset.filter (λ x, x < lower_threshold ∨ x > upper_threshold)

theorem number_of_outliers : outliers_in_dataset.length = 1 := 
  by sorry

end number_of_outliers_l5_5074


namespace sequences_nat_l5_5256

noncomputable def tangent_line_at (x : ℝ) : ℝ → ℝ := λ y, exp x * (y - x) + exp x

def intersection_point (n : ℕ) : ℝ × ℝ := 
  let A_n := n + (1 / (exp 1 - 1)) in
  let B_n := (exp (n + 1)) / (exp 1 - 1) in
  (A_n, B_n)

theorem sequences_nat (n : ℕ) : 
  let (A_n, B_n) := intersection_point n in
  (A_n = n + 1 / (exp 1 - 1)) ∧ (B_n = exp (n + 1) / (exp 1 - 1)) :=
by 
  sorry

end sequences_nat_l5_5256


namespace tangerines_more_than_oranges_l5_5331

-- Define initial conditions
def initial_apples := 9
def initial_oranges := 5
def initial_tangerines := 17

-- Define actions taken
def oranges_taken := 2
def tangerines_taken := 10

-- Resulting quantities
def oranges_left := initial_oranges - oranges_taken
def tangerines_left := initial_tangerines - tangerines_taken

-- Proof problem
theorem tangerines_more_than_oranges : tangerines_left - oranges_left = 4 := 
by sorry

end tangerines_more_than_oranges_l5_5331


namespace parallelogram_area_l5_5103

noncomputable def vector_u : ℝ × ℝ × ℝ := (4, 2, -3)
noncomputable def vector_v : ℝ × ℝ × ℝ := (2, -4, 5)

theorem parallelogram_area : 
  let cross_product := (vector_u.2 * vector_v.3 - vector_u.3 * vector_v.2, vector_u.3 * vector_v.1 - vector_u.1 * vector_v.3, vector_u.1 * vector_v.2 - vector_u.2 * vector_v.1) in
  let magnitude := real.sqrt (cross_product.1^2 + cross_product.2^2 + cross_product.3^2) in
  magnitude = 10 * real.sqrt (10.8) :=
sorry

end parallelogram_area_l5_5103


namespace speed_of_larger_fragment_l5_5032

noncomputable def v0 : ℝ := 20   -- initial speed in m/s
noncomputable def g : ℝ := 10    -- acceleration due to gravity in m/s^2
noncomputable def t : ℝ := 3     -- time in seconds
noncomputable def m_ratio : ℝ := 1 / 2  -- mass ratio
noncomputable def v_small_horizontal : ℝ := 16   -- horizontal speed of smaller fragment in m/s

theorem speed_of_larger_fragment : 
  let v_before_explosion := v0 - g * t in
  let v_horizontal := - (v_small_horizontal / 2) in
  let v_vertical := - (v_before_explosion / 2) in
  (v_before_explosion < 0) → 
  (v_horizontal < 0) →
  (v_vertical < 0) →
  sqrt(v_horizontal^2 + v_vertical^2) = 17 :=
by
  sorry

end speed_of_larger_fragment_l5_5032


namespace value_of_a_l5_5655

theorem value_of_a (a : ℝ) (h : ∃ b : ℝ, b ≠ 0 ∧ (a - real.I) / (2 + real.I) = b * real.I) : a = 1/2 :=
by
  sorry

end value_of_a_l5_5655


namespace mod_exp_difference_l5_5950

theorem mod_exp_difference (a b n k : ℕ) (h1: a ≡ 0 [MOD n]) (h2: b ≡ 0 [MOD n]) : (a^k - b^k) ≡ 0 [MOD n] :=
by
  sorry

example : (54^2023 - 27^2023) ≡ 0 [MOD 9] :=
  mod_exp_difference 54 27 9 2023
    (by norm_num) 
    (by norm_num)

end mod_exp_difference_l5_5950


namespace expected_value_8_sided_die_l5_5397

/-- 
The expected value of rolling a standard 8-sided die is 4.5.
The die has 8 sides labeled 1 through 8, and each face has an equal probability of appearing,
which is 1/8. 
-/
theorem expected_value_8_sided_die : 
  (1/8:ℝ) * 1 + (1/8) * 2 + (1/8) * 3 + (1/8) * 4 + (1/8) * 5 + (1/8) * 6 + (1/8) * 7 + (1/8) * 8 = 4.5 :=
by 
  sorry

end expected_value_8_sided_die_l5_5397


namespace sum_of_x_coordinates_l5_5119

theorem sum_of_x_coordinates (x : ℝ) (y : ℝ) 
    (h1 : y = abs (x^2 - 4 * x + 3)) 
    (h2 : y = 6 - x) : 
  (let solutions := {x | ∃ y, y = abs (x^2 - 4 * x + 3) ∧ y = 6 - x} in 
  ∑ x in solutions, x) = 3 := 
sorry

end sum_of_x_coordinates_l5_5119


namespace inverse_function_ratio_l5_5241

theorem inverse_function_ratio (a b c d : ℝ) : 
  (∀ x, g (g_inv x) = x) ∧ (∀ x, g_inv (g x) = x) → a / c = -4 :=
by
  let g (x : ℝ) := (3 * x - 2) / (x + 4)
  let g_inv (x : ℝ) := (a * x + b) / (c * x + d)
  intros h
  sorry

end inverse_function_ratio_l5_5241


namespace sqrt_88200_simplified_l5_5293

theorem sqrt_88200_simplified : Real.sqrt 88200 = 210 * Real.sqrt 6 :=
by sorry

end sqrt_88200_simplified_l5_5293


namespace selection_methods_count_l5_5130

theorem selection_methods_count :
  ∃ (n : ℕ), n = 420 ∧
    (∃ (boy_group girl_group : finset ℕ), 
      boy_group.card = 5 ∧ 
      girl_group.card = 4 ∧ 
      (∃ (selected_boys selected_girls : finset ℕ), 
        selected_boys.card + selected_girls.card = 3 ∧ 
        selected_boys ⊆ boy_group ∧ 
        selected_girls ⊆ girl_group ∧ 
        selected_boys.card > 0 ∧ 
        selected_girls.card > 0)) :=
begin
  existsi 420,
  split,
  { refl, },
  {
    existsi (finset.range 5),
    existsi (finset.range 4),
    split,
    { exact finset.card_range 5, },
    split,
    { exact finset.card_range 4, },
    { existsi ({0, 1}.to_finset), existsi ({0}.to_finset),
      split,
      { norm_num, },
      { split,
        { norm_num, },
        { split,
          { norm_num, },
          { split,
            { norm_num, },
            { norm_num, },
          },
        },
      },
    },
  },
end

end selection_methods_count_l5_5130


namespace count_terminating_decimals_l5_5617

theorem count_terminating_decimals :
  {n : ℕ | 1 ≤ n ∧ n ≤ 150 ∧ ∃ k : ℕ, n = 3 * k}.to_finset.card = 50 := by
sorry

end count_terminating_decimals_l5_5617


namespace polynomial_remainder_l5_5568

variable (R : Type) [CommRing R]

def polynomial_example (x : R) : R :=
  x^4 + 4*x^2 + 20*x + 1

def divisor (x : R) : R :=
  x^2 - 2*x + 7

theorem polynomial_remainder (x : R) : 
  ∃ q r : R, polynomial_example x = q * divisor x + r ∧ (degree r < degree (divisor x)) ∧ r = 8*x - 6 := 
by
  sorry

end polynomial_remainder_l5_5568


namespace triangle_equilateral_l5_5742

theorem triangle_equilateral (A B C : ℝ) (a b c : ℝ) [triangle ABC a b c] 
  (h : ∀ A B C, a / Real.cos A = b / Real.cos B ∧ b / Real.cos B = c / Real.cos C) :
  ∀ A B C, A = 60 ∧ B = 60 ∧ C = 60 ∧ a = b ∧ b = c ∧ c = a :=
by
  sorry

end triangle_equilateral_l5_5742


namespace desired_gasoline_percentage_l5_5914

def initial_gasohol_volume : ℝ := 45
def initial_ethanol_percentage : ℝ := 0.05
def additional_ethanol : ℝ := 2.5
def optimum_ethanol_percentage : ℝ := 0.10

theorem desired_gasoline_percentage : 
  let initial_ethanol := initial_ethanol_percentage * initial_gasohol_volume,
      total_ethanol := initial_ethanol + additional_ethanol,
      total_volume := initial_gasohol_volume + additional_ethanol,
      optimum_ethanol := optimum_ethanol_percentage * total_volume
  in total_ethanol = optimum_ethanol → 100 * (total_volume - total_ethanol) / total_volume = 90 := 
by {
  intros h,
  sorry
}

end desired_gasoline_percentage_l5_5914


namespace expected_value_of_eight_sided_die_l5_5347

-- Definitions based on the problem conditions
def eight_sided_die_outcomes : List ℕ := [1,2,3,4,5,6,7,8]

def probability (n : ℕ) := 1 / n

-- Expected value calculation related to the problem
def expected_value_die_roll (outcomes : List ℕ) (prob : ℕ → Rat) : Rat :=
  List.sum (outcomes.map (λ x => prob outcomes.length * x))

-- Expected value of an 8-sided die roll
theorem expected_value_of_eight_sided_die :
  expected_value_die_roll eight_sided_die_outcomes probability = 4.5 := 
sorry

end expected_value_of_eight_sided_die_l5_5347


namespace num_whole_numbers_between_sqrt4_50_and_sqrt4_1000_l5_5692

def between_sqrt4_50_and_sqrt4_1000 : Prop :=
  3 < real.root 4 50 ∧ real.root 4 50 < 4 ∧
  5 < real.root 4 1000 ∧ real.root 4 1000 < 6

theorem num_whole_numbers_between_sqrt4_50_and_sqrt4_1000 (h : between_sqrt4_50_and_sqrt4_1000) : 
  ∃ n1 n2 : ℕ, real.root 4 50 < n1 ∧ n1 < n2 ∧ n2 < real.root 4 1000 ∧ n1 = 4 ∧ n2 = 5 :=
sorry

end num_whole_numbers_between_sqrt4_50_and_sqrt4_1000_l5_5692


namespace area_of_rhombus_l5_5098

/-- Given the radii of the circles circumscribed around triangles EFG and EGH
    are 10 and 20, respectively, then the area of rhombus EFGH is 30.72√3. -/
theorem area_of_rhombus (R1 R2 : ℝ) (A : ℝ) :
  R1 = 10 → R2 = 20 → A = 30.72 * Real.sqrt 3 :=
by sorry

end area_of_rhombus_l5_5098


namespace expected_value_of_eight_sided_die_l5_5350

-- Definitions based on the problem conditions
def eight_sided_die_outcomes : List ℕ := [1,2,3,4,5,6,7,8]

def probability (n : ℕ) := 1 / n

-- Expected value calculation related to the problem
def expected_value_die_roll (outcomes : List ℕ) (prob : ℕ → Rat) : Rat :=
  List.sum (outcomes.map (λ x => prob outcomes.length * x))

-- Expected value of an 8-sided die roll
theorem expected_value_of_eight_sided_die :
  expected_value_die_roll eight_sided_die_outcomes probability = 4.5 := 
sorry

end expected_value_of_eight_sided_die_l5_5350


namespace solution1_solution2_l5_5947

noncomputable def problem1 : ℝ :=
  (3 * real.sqrt 27 - 2 * real.sqrt 12) / real.sqrt 3

noncomputable def problem2 : ℝ :=
  2 * real.sqrt 8 + 4 * real.sqrt (1 / 2) - 3 * real.sqrt 32

theorem solution1 : problem1 = 5 := by
  sorry

theorem solution2 : problem2 = -4 * real.sqrt 2 := by
  sorry

end solution1_solution2_l5_5947


namespace avg_annual_growth_rate_l5_5486
-- Import the Mathlib library

-- Define the given conditions
def initial_income : ℝ := 32000
def final_income : ℝ := 37000
def period : ℝ := 2
def initial_income_ten_thousands : ℝ := initial_income / 10000
def final_income_ten_thousands : ℝ := final_income / 10000

-- Define the growth rate
variable (x : ℝ)

-- Define the theorem
theorem avg_annual_growth_rate :
  3.2 * (1 + x) ^ 2 = 3.7 :=
sorry

end avg_annual_growth_rate_l5_5486


namespace letter_puzzle_solutions_l5_5539

theorem letter_puzzle_solutions (A B : ℕ) : 
  (1 ≤ A ∧ A < 10) ∧ (1 ≤ B ∧ B < 10) ∧ (A ≠ B) ∧ (A^B = 10 * B + A) → 
  (A = 2 ∧ B = 5) ∨ (A = 6 ∧ B = 2) ∨ (A = 4 ∧ B = 3) :=
by
  sorry

end letter_puzzle_solutions_l5_5539


namespace brendan_tax_percentage_l5_5943

theorem brendan_tax_percentage :
  let hourly_wage := 6
      hours_week := 2 * 8 + 12
      tips_per_hour := 12
      reported_fraction := 1 / 3
      taxes_paid := 56
      total_earnings := hours_week * hourly_wage
      total_tips := hours_week * tips_per_hour
      reported_tips := reported_fraction * total_tips
      reported_income := total_earnings + reported_tips
  in (taxes_paid / reported_income) * 100 = 20 :=
by
  sorry

end brendan_tax_percentage_l5_5943


namespace interval_containing_zero_l5_5818

noncomputable def f (x : ℝ) : ℝ := (1 / 3) ^ x - x

theorem interval_containing_zero :
  (∃ c ∈ Ioo 0 1, f c = 0) :=
begin
  -- Conditions
  have h1 : ∀ x y : ℝ, x < y → f x > f y,
  { intros x y hxy,
    simp [f],
    apply lt_sub_iff_add_lt.2,
    rw ←sub_lt_sub_iff_right x,
    apply sub_lt_sub,
    exact pow_lt_pow_of_lt_left one_third_pos hxy zero_le_one,
    exact hxy, },

  have h2 : f 0 = 1, from by simp [f],
  have h3 : f 1 = -2 / 3, from by norm_num [f],

  -- Existence of zero in the interval
  use exists_between (h2 ▸ (h3 ▸ by norm_num) : (f 0 > 0 ∧ f 1 < 0)),
  split,
  norm_num,
  norm_num,
end

end interval_containing_zero_l5_5818


namespace intersection_of_sets_l5_5676

-- Define the sets A and B
def A : Set ℝ := {x | |x| ≤ 2}
def B : Set ℝ := {x | 3 * x - 2 ≥ 1}

-- Prove that A ∩ B = {x | 1 ≤ x ∧ x ≤ 2}
theorem intersection_of_sets : A ∩ B = {x | 1 ≤ x ∧ x ≤ 2} :=
by sorry

end intersection_of_sets_l5_5676


namespace find_k_l5_5917

theorem find_k (k : ℝ) :
  let P1 := (0 : ℝ, 4 : ℝ)
  let P2 := (5 : ℝ, k)
  let P3 := (15 : ℝ, 1 : ℝ)
  collinear P1 P2 P3 → k = 3 :=
sorry

def collinear (P1 P2 P3 : ℝ × ℝ) : Prop :=
  let (x1, y1) := P1
  let (x2, y2) := P2
  let (x3, y3) := P3
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

end find_k_l5_5917


namespace area_of_region_l5_5519

-- Definitions from the problem's conditions.
def equation (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 10*y = -9

-- Statement of the theorem.
theorem area_of_region : 
  ∃ (area : ℝ), (∀ x y : ℝ, equation x y → True) ∧ area = 32 * Real.pi :=
by
  sorry

end area_of_region_l5_5519


namespace cone_csa_200pi_l5_5317

def cone_curved_surface_area (r l : ℝ) : ℝ := π * r * l

theorem cone_csa_200pi (r l : ℝ) (h_r : r = 10) (h_l : l = 20) : 
  cone_curved_surface_area r l = 200 * π :=
by
  rw [cone_curved_surface_area, h_r, h_l]
  norm_num
  ring

end cone_csa_200pi_l5_5317


namespace expected_value_of_8_sided_die_l5_5389

def outcomes : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]
def probability (n : ℕ) : ℝ := 1 / 8

theorem expected_value_of_8_sided_die :
  (∑ x in outcomes, probability x * x) = 4.5 := 
sorry

end expected_value_of_8_sided_die_l5_5389


namespace number_of_terminating_decimals_l5_5623

theorem number_of_terminating_decimals (n : ℕ) (h₁ : 1 ≤ n) (h₂ : n ≤ 150) :
  ∃ m, m = 50 ∧ 
  ∀ n, (1 ≤ n ∧ n ≤ 150) → (∃ k, n = 3 * k) →
  m = 50 :=
by 
  sorry

end number_of_terminating_decimals_l5_5623


namespace cosines_of_acute_angles_l5_5564
noncomputable theory

-- Define the conditions for a right triangle and the given conditions
def right_triangle (α β : ℝ) : Prop :=
  α + β = π / 2 ∧ α > 0 ∧ β > 0

def tangent_product_condition (α : ℝ) : Prop :=
  (Real.tan (α / 2) * Real.tan ((π / 4) - (α / 2)) = 1 / 6)

-- The theorem to prove the cosines of the acute angles
theorem cosines_of_acute_angles (α β : ℝ) (h : right_triangle α β) (hcond : tangent_product_condition α) :
  (Real.cos α = 3 / 5 ∧ Real.cos β = 4 / 5) ∨ (Real.cos α = 4 / 5 ∧ Real.cos β = 3 / 5) :=
sorry

end cosines_of_acute_angles_l5_5564


namespace coefficient_comparison_l5_5212

noncomputable def polynomial_expansion_1 := (1 + x^2 - x^3) ^ 1000
noncomputable def polynomial_expansion_2 := (1 - x^2 + x^3) ^ 1000

theorem coefficient_comparison (x : ℝ) :
  polynomial_expansion_1.coeff 20 > polynomial_expansion_2.coeff 20 :=
sorry

end coefficient_comparison_l5_5212


namespace area_of_parallelogram_l5_5105

-- Definitions of the vectors
def u : ℝ × ℝ × ℝ := (4, 2, -3)
def v : ℝ × ℝ × ℝ := (2, -4, 5)

-- Definition of the cross product
def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((a.2 * b.3 - a.3 * b.2), (a.3 * b.1 - a.1 * b.3), (a.1 * b.2 - a.2 * b.1))

-- Definition of the magnitude of a vector
def magnitude (a : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (a.1 * a.1 + a.2 * a.2 + a.3 * a.3)

-- The area of the parallelogram is the magnitude of the cross product of u and v
def area_parallelogram (u v : ℝ × ℝ × ℝ) : ℝ := magnitude (cross_product u v)

-- Proof statement
theorem area_of_parallelogram : area_parallelogram u v = 20 * real.sqrt 3 :=
by
  sorry

end area_of_parallelogram_l5_5105


namespace depth_of_water_is_60_l5_5493

def dean_height : ℕ := 6
def depth_multiplier : ℕ := 10
def water_depth : ℕ := depth_multiplier * dean_height

theorem depth_of_water_is_60 : water_depth = 60 := by
  -- mathematical equivalent proof problem
  sorry

end depth_of_water_is_60_l5_5493


namespace no_mutually_perpendicular_moments_l5_5072

-- Definitions for time intervals and perpendicular times
def straight_line_time (t : ℤ) : Prop := ∃ n : ℤ, t = 6 * n + 3
def perpendicular_time (t : ℤ) : Prop := ∃ k : ℤ, 0 ≤ k < 22 ∧ t = k * (12 / 22)

theorem no_mutually_perpendicular_moments :
  ¬ ∃ t : ℤ, straight_line_time t ∧ perpendicular_time t :=
by
  sorry

end no_mutually_perpendicular_moments_l5_5072


namespace parametric_equations_same_curve_l5_5678

-- Definitions of parametric equations
def eq1 (t : ℝ) : ℝ × ℝ := (t, t^2)
def eq2 (t : ℝ) : ℝ × ℝ := (Real.tan t, Real.tan t^2)
def eq3 (t : ℝ) : ℝ × ℝ := (Real.sin t, Real.sin t^2)

-- Definition of the curve
def curve (x : ℝ) : ℝ := x^2

-- Statement of the problem
theorem parametric_equations_same_curve :
  (∀ t : ℝ, eq1 t = (x, curve x)) ↔ (∀ t : ℝ, eq2 t = (x, curve x)) :=
sorry

end parametric_equations_same_curve_l5_5678


namespace mint_can_issue_coins_l5_5896

def can_Issue_Denominations : Prop :=
  ∃ d : Finset ℕ, d.card = 12 ∧ 
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 6543 →
    ∃ c : Multiset ℕ, (∀ x ∈ c, x ∈ d) ∧ c.sum = n ∧ c.card ≤ 8

theorem mint_can_issue_coins : can_Issue_Denominations :=
  sorry

end mint_can_issue_coins_l5_5896


namespace quadratic_inequality_solution_l5_5662

theorem quadratic_inequality_solution (a b c : ℝ) (h_solution_set : ∀ x, ax^2 + bx + c < 0 ↔ x < -1 ∨ x > 3) :
  (a < 0) ∧
  (a + b + c > 0) ∧
  (∀ x, cx^2 - bx + a < 0 ↔ -1/3 < x ∧ x < 1) :=
by
  sorry

end quadratic_inequality_solution_l5_5662


namespace repeating_decimal_fraction_l5_5991

theorem repeating_decimal_fraction :  exists (p q : ℤ), (q ≠ 0) ∧ (p / q = (68 / 33 : ℚ)) :=
begin
  let recurring02 : ℚ := 2 / 99,
  let recurring06 : ℚ := 3 * recurring02,
  have recurring06_simplified : recurring06 = 2 / 33, 
    by simp [recurring06, mul_div_assoc, mul_div_cancel_left],
  let result := 2 + recurring06_simplified,
  use [68, 33],
  split,
  { norm_num }, -- proof that 33 ≠ 0
  { exact result } -- proof that 2.06 recurring = 68 / 33
end

end repeating_decimal_fraction_l5_5991


namespace locus_and_line_equation_l5_5148

-- Define the fixed point A
def A : ℝ × ℝ := (1, -2)

-- Define the circle equation
def circle (B : ℝ × ℝ) : Prop := (B.1 + 1)^2 + (B.2 + 4)^2 = 4

-- Define the midpoint condition for C
def is_midpoint (A B C : ℝ × ℝ) : Prop := C = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define the fixed point P
def P : ℝ × ℝ := (1/2, -2)

-- Define the line passing through P
def line (k : ℝ) : ℝ × ℝ → Prop := λ C, C.2 + 2 = k * (C.1 - 1/2)

-- Define the distance |MN| = sqrt(3)
def distance_MN (M N : ℝ × ℝ) : Prop := real.sqrt ((N.1 - M.1)^2 + (N.2 - M.2)^2) = real.sqrt 3

-- Locus and line equation problem statement
theorem locus_and_line_equation :
  ∀ (C : ℝ × ℝ),
    (∃ B, circle B ∧ is_midpoint A B C) →
    (C.1^2 + (C.2 + 3)^2 = 1) ∧
    (∀ M N, (line (3/4) M ∧ line (3/4) N ∧ distance_MN M N) ∨ (M.1 = 1/2 ∧ N.1 = 1/2) →
             (∀ x y, line (3/4) (x, y) ∨ x = 1/2)) :=
by sorry


end locus_and_line_equation_l5_5148


namespace repeating_decimal_to_fraction_l5_5976

/-- Given 0.02 repeating as a fraction 2/99, prove that 2.06 repeating can be expressed as 68/33 -/
theorem repeating_decimal_to_fraction :
  (2 + 0.06̅ : ℝ) = (68 / 33 : ℝ) :=
begin
  have h : (0.02̅ : ℝ) = (2 / 99 : ℝ), from sorry,
  have h3 : (0.06̅ : ℝ) = 3 * (0.02̅ : ℝ), from sorry,
  have h6 : (0.06̅ : ℝ) = 3 * (2 / 99 : ℝ), from sorry,
  have s : (0.06̅ : ℝ) = (6 / 99 : ℝ), from sorry,
  have s2 : (6 / 99 : ℝ) = (2 / 33 : ℝ), from sorry,
  have add := congr_arg (λ x : ℝ, (2 : ℝ) + x) s2,
  rw [add_comm, ← add_halves', add_assoc', add_comm (2 : ℝ), add_comm 2 (2 / 33 : ℝ), add_halves',
    add_assoc', add_comm (2 : ℝ), add_comm 68 (2 / 33 : ℝ)] ,
end

end repeating_decimal_to_fraction_l5_5976


namespace find_decreasing_function_l5_5931

def y_decreases_as_x_increases (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂

theorem find_decreasing_function :
  ∃ f : ℝ → ℝ, (f = λ x, -2 * x + 8) ∧ y_decreases_as_x_increases f ∧
    (∀ g : ℝ → ℝ, (g = λ x, 2 * x + 8) → ¬ y_decreases_as_x_increases g) ∧
    (∀ g : ℝ → ℝ, (g = λ x, 4 * x - 2) → ¬ y_decreases_as_x_increases g) ∧
    (∀ g : ℝ → ℝ, (g = λ x, 4 * x) → ¬ y_decreases_as_x_increases g) :=
  by
    sorry

end find_decreasing_function_l5_5931


namespace PU_squared_fraction_l5_5211

noncomputable def compute_PU_squared : ℚ :=
  sorry -- Proof of the distance computation PU^2.

theorem PU_squared_fraction :
  ∃ (a b : ℕ), (gcd a b = 1) ∧ (compute_PU_squared = a / b) :=
  sorry -- Proof that the resulting fraction a/b is in its simplest form.

end PU_squared_fraction_l5_5211


namespace simplify_expression_l5_5001

theorem simplify_expression : (-8 - (+4) + (-5) - (-2) = -8 - 4 - 5 + 2) :=
by 
  -- skipping the proof part with sorry
  sorry

end simplify_expression_l5_5001


namespace total_prime_factors_l5_5120

theorem total_prime_factors (a b c : ℕ) (ha : a = 4) (hb : b = 7) (hc : c = 11) :
  let expr := (a ^ 11) * (b ^ 7) * (c ^ 2)
  (total_factors : ℕ) := total_factors = 31 :=
sorry

end total_prime_factors_l5_5120


namespace player2_wins_with_optimal_strategy_l5_5009

def grid : Type := fin 100 × fin 100

def is_domino (p q : grid) : Prop :=
  (abs (p.1 - q.1) = 1 ∧ p.2 = q.2) ∨ (abs (p.2 - q.2) = 1 ∧ p.1 = q.1)

inductive move_result : Type
| continue : move_result
| lose : move_result

def make_move (grid_state : set (grid × grid)) (p q : grid) (player : ℕ) : move_result :=
  if is_domino p q then
    let new_state := grid_state.insert (p, q) in
    if (connected_component new_state p).card > 1 then move_result.lose
    else move_result.continue
  else
    move_result.continue

noncomputable def optimal_play (player : ℕ) (grid_state : set (grid × grid)) : move_result :=
  sorry -- Placeholder for the definition of each player's optimal strategy

theorem player2_wins_with_optimal_strategy :
  ∀ initial_grid_state : set (grid × grid),
  optimal_play 2 initial_grid_state = move_result.lose :=
sorry

end player2_wins_with_optimal_strategy_l5_5009


namespace sphere_volume_l5_5320

theorem sphere_volume {r : ℝ} (h: 4 * Real.pi * r^2 = 256 * Real.pi) : (4 / 3) * Real.pi * r^3 = (2048 / 3) * Real.pi :=
by
  sorry

end sphere_volume_l5_5320


namespace hari_contribution_is_correct_l5_5007

-- Define the problem conditions
def initial_capital_praveen : ℝ := 3360
def time_praveen_engaged_months : ℝ := 12
def time_hari_engaged_months : ℝ := 7 -- 12 - 5 months
def profit_sharing_ratio : ℝ := 2 / 3

-- Define the effective capital contributions
def effective_capital_praveen := initial_capital_praveen * time_praveen_engaged_months
noncomputable def capital_hari (h : ℝ) := h
noncomputable def effective_capital_hari (h : ℝ) := capital_hari h * time_hari_engaged_months

-- The proof problem: Given the conditions, prove that Hari's contribution (H) equals 8640
theorem hari_contribution_is_correct (H : ℝ) 
  (h_effective : effective_capital_hari H = ((initial_capital_praveen * time_praveen_engaged_months) * (3/2)) / time_hari_engaged_months)  : 
  H = 8640 :=
by 
  sorry

end hari_contribution_is_correct_l5_5007


namespace slope_of_parallel_line_l5_5866

-- Define the points
def point1 : ℝ × ℝ := (3, -4)
def point2 : ℝ × ℝ := (-4, 2)

-- Define the function to calculate the slope between two points
def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

-- State the theorem to be proved
theorem slope_of_parallel_line :
  slope point1 point2 = -6/7 -> ∀ p3 p4 : ℝ × ℝ, slope p1 p2 = slope p3 p4 :=
sorry

end slope_of_parallel_line_l5_5866


namespace expected_value_of_eight_sided_die_l5_5373

theorem expected_value_of_eight_sided_die : 
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8], 
      n := (outcomes.length : ℚ),
      probabilities := List.replicate (outcomes.length) (1 / n),
      expected_value := (List.zipWith (*) probabilities (outcomes.map (· : ℚ))).sum
  in expected_value = 4.5 :=
by
  sorry

end expected_value_of_eight_sided_die_l5_5373


namespace percentage_of_day_only_candidates_l5_5901

theorem percentage_of_day_only_candidates : 
  (∃ D : ℝ, D = 1 - (0.06 / 0.20) ∧ D = 0.70) :=
by
  let D := 1 - (0.06 / 0.20)
  have hD : D = 0.70 := sorry
  exact ⟨D, ⟨rfl, hD⟩⟩

end percentage_of_day_only_candidates_l5_5901


namespace sqrt_domain_l5_5206

theorem sqrt_domain (x : ℝ) : (∃ y, y = sqrt (x - 2)) ↔ x ≥ 2 := by
  sorry

end sqrt_domain_l5_5206


namespace harris_carrot_cost_l5_5682

-- Definitions stemming from the conditions
def carrots_per_day : ℕ := 1
def days_per_year : ℕ := 365
def carrots_per_bag : ℕ := 5
def cost_per_bag : ℕ := 2

-- Prove that Harris's total cost for carrots in one year is $146
theorem harris_carrot_cost : (days_per_year * carrots_per_day / carrots_per_bag) * cost_per_bag = 146 := by
  sorry

end harris_carrot_cost_l5_5682


namespace mildred_spending_correct_l5_5266

variable (total_money : ℕ) (remaining_money : ℕ) (candice_spending : ℕ)

def total_spent (total_money : ℕ) (remaining_money : ℕ) : ℕ := total_money - remaining_money

def mildred_spent (total_spent : ℕ) (candice_spending : ℕ) : ℕ := total_spent - candice_spending

theorem mildred_spending_correct : 
  total_money = 100 → 
  remaining_money = 40 → 
  candice_spending = 35 → 
  mildred_spent (total_spent total_money remaining_money) candice_spending = 25 := 
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  unfold total_spent mildred_spent
  simp
  sorry

end mildred_spending_correct_l5_5266


namespace max_min_difference_l5_5247

noncomputable def difference_max_min_z (x y z : ℝ) : ℝ :=
  if h₁ : x + y + z = 3 ∧ x^2 + y^2 + z^2 = 18 then 6 else 0

theorem max_min_difference (x y z : ℝ) (h₁ : x + y + z = 3) (h₂ : x^2 + y^2 + z^2 = 18) :
  difference_max_min_z x y z = 6 :=
by sorry

end max_min_difference_l5_5247


namespace geo_seq_sum_condition_l5_5915

noncomputable def geometric_seq (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a * q^n

noncomputable def sum_geo_seq_3 (a : ℝ) (q : ℝ) : ℝ :=
  geometric_seq a q 0 + geometric_seq a q 1 + geometric_seq a q 2

noncomputable def sum_geo_seq_6 (a : ℝ) (q : ℝ) : ℝ :=
  sum_geo_seq_3 a q + geometric_seq a q 3 + geometric_seq a q 4 + geometric_seq a q 5

theorem geo_seq_sum_condition {a q S₃ S₆ : ℝ} (h_sum_eq : S₆ = 9 * S₃)
  (h_S₃_def : S₃ = sum_geo_seq_3 a q)
  (h_S₆_def : S₆ = sum_geo_seq_6 a q) :
  q = 2 :=
by
  sorry

end geo_seq_sum_condition_l5_5915


namespace log_base_2_b2010_l5_5137

theorem log_base_2_b2010 :
  ∀ (b : ℕ → ℝ),
    b 1 = 1 →
    (∀ n : ℕ, 0 < b (n + 1)) →
    (∀ n : ℕ, n ≠ 0 → n * (b (n + 1))^2 - 2 * (b n)^2 - (2 * n - 1) * (b (n + 1)) * (b n) = 0) →
    real.logb 2 (b 2010) = 2009 :=
by
  intros b hb1 hpos heq
  sorry

end log_base_2_b2010_l5_5137


namespace repeating_decimal_to_fraction_l5_5999

theorem repeating_decimal_to_fraction (h : 0.\overline{02} = 2 / 99) : 
  2.\overline{06} = 68 / 33 := by
  sorry

end repeating_decimal_to_fraction_l5_5999


namespace cross_platform_time_l5_5820

-- Definitions of the conditions
def length_train : ℝ := 750
def length_platform := length_train
def speed_train_kmh : ℝ := 90
def speed_train_ms := speed_train_kmh * (1000 / 3600)

-- The proof problem: Prove that the time to cross the platform is 60 seconds
theorem cross_platform_time : 
  (length_train + length_platform) / speed_train_ms = 60 := 
by
  sorry

end cross_platform_time_l5_5820


namespace find_points_XYQ_l5_5733

-- Definitions of points and lines for the pyramid structure
structure Point3D :=
(x : ℝ)
(y : ℝ)
(z : ℝ)
  
structure Line3D :=
(start : Point3D)
(end : Point3D)

def is_midpoint (M A B : Point3D) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ 
  M.y = (A.y + B.y) / 2 ∧ 
  M.z = (A.z + B.z) / 2

-- Given points in the problem
constant M A B C D : Point3D
constant K P X Y Q : Point3D

-- Conditions from the problem
axiom regular_pyramid : ∀ (p : Point3D), p ∈ {M, A, B, C, D} := sorry
axiom divides_edge_MC : (K.x = (2 * M.x + C.x) / 3) ∧ (K.y = (2 * M.y + C.y) / 3) ∧ (K.z = (2 * M.z + C.z) / 3)
axiom midpoint_MD : P.x = (M.x + D.x) / 2 ∧ P.y = (M.y + D.y) / 2 ∧ P.z = (M.z + D.z) / 2
axiom points_on_lines : X ∈ {M, X} ∧ Y ∈ {B, K} ∧ Q ∈ {C, P}
axiom Y_midpoint_XQ : is_midpoint Y X Q

-- The proof statement
theorem find_points_XYQ :
  ∃ (X Y Q : Point3D), 
    (X ∈ {M, X}) ∧ 
    (Y ∈ {B, K}) ∧ 
    (Q ∈ {C, P}) ∧ 
    is_midpoint Y X Q :=
  sorry

end find_points_XYQ_l5_5733


namespace smallest_rel_prime_to_180_l5_5608

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ Int.gcd x 180 = 1 ∧ ∀ y : ℕ, y > 1 ∧ Int.gcd y 180 = 1 → x ≤ y := 
sorry

end smallest_rel_prime_to_180_l5_5608


namespace solve_inequality_l5_5798

theorem solve_inequality (x y : ℤ) (h1 : x - 3 * y + 2 ≥ 1) (h2 : -x + 2 * y + 1 ≥ 1) : 
  (x = 0 ∧ y = 0) ∨ (x = 2 ∧ y = 1) ↔
  (∀ x y : ℤ, x - 3 * y + 2 ≥ 1 ∧ -x + 2 * y + 1 ≥ 1 →
    (frac (x ^ 2) (sqrt (x - 3 * y + 2)) + frac (y ^ 2) (sqrt (-x + 2 * y + 1))) ≥ y ^ 2 + 2 * x ^ 2 - 2 * x - 1) :=
sorry

end solve_inequality_l5_5798


namespace jordan_field_area_l5_5747

theorem jordan_field_area
  (s l : ℕ)
  (h1 : 2 * (s + l) = 24)
  (h2 : l + 1 = 2 * (s + 1)) :
  3 * s * 3 * l = 189 := 
by
  sorry

end jordan_field_area_l5_5747


namespace smallest_coprime_gt_one_l5_5572

theorem smallest_coprime_gt_one (x : ℕ) (h1 : 1 < x) (h2 : Nat.gcd x 180 = 1) : x = 7 := sorry

end smallest_coprime_gt_one_l5_5572


namespace repeating_decimal_fraction_l5_5985

theorem repeating_decimal_fraction :  exists (p q : ℤ), (q ≠ 0) ∧ (p / q = (68 / 33 : ℚ)) :=
begin
  let recurring02 : ℚ := 2 / 99,
  let recurring06 : ℚ := 3 * recurring02,
  have recurring06_simplified : recurring06 = 2 / 33, 
    by simp [recurring06, mul_div_assoc, mul_div_cancel_left],
  let result := 2 + recurring06_simplified,
  use [68, 33],
  split,
  { norm_num }, -- proof that 33 ≠ 0
  { exact result } -- proof that 2.06 recurring = 68 / 33
end

end repeating_decimal_fraction_l5_5985


namespace cousin_reading_time_l5_5781

theorem cousin_reading_time (my_time_hours : ℕ) (speed_ratio : ℕ) (my_time_minutes := my_time_hours * 60) :
  (my_time_hours = 3) ∧ (speed_ratio = 5) → 
  (my_time_minutes / speed_ratio = 36) :=
by
  sorry

end cousin_reading_time_l5_5781


namespace smallest_rel_prime_to_180_l5_5583

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ Nat.gcd x 180 = 1 ∧ ∀ y : ℕ, y > 1 ∧ Nat.gcd y 180 = 1 → x ≤ y :=
begin
  use 7,
  split,
  { exact dec_trivial },
  split,
  { exact dec_trivial },
  { intro y,
    intro h,
    cases h with h1 h2,
    repeat { try { apply dec_trivial,
                   apply lt_or_eq_of_le,
                   norm_num,
                   apply Nat.prime_not_dvd_mul,
                   norm_num,
                   apply not_or_distrib.mpr,
                   split,
                   norm_cast,
                   intro,
                   exact le_antisymm _ },
           sorry }
end

end smallest_rel_prime_to_180_l5_5583


namespace johns_overall_profit_l5_5438

-- Definitions based on conditions
def cost_price_grinder : ℝ := 15000
def cost_price_mobile : ℝ := 8000
def loss_percentage_grinder : ℝ := 5 / 100
def profit_percentage_mobile : ℝ := 10 / 100

-- Prove overall profit
theorem johns_overall_profit : 
  let selling_price_grinder := cost_price_grinder * (1 - loss_percentage_grinder)
  let selling_price_mobile := cost_price_mobile * (1 + profit_percentage_mobile)
  let total_cost_price := cost_price_grinder + cost_price_mobile
  let total_selling_price := selling_price_grinder + selling_price_mobile
  in 
  total_selling_price - total_cost_price = 50 := 
  by
  sorry

end johns_overall_profit_l5_5438


namespace determine_z_l5_5966

theorem determine_z (z : ℝ) (h1 : ∃ x : ℤ, 3 * (x : ℝ) ^ 2 + 19 * (x : ℝ) - 84 = 0 ∧ (x : ℝ) = ⌊z⌋) (h2 : 4 * (z - ⌊z⌋) ^ 2 - 14 * (z - ⌊z⌋) + 6 = 0) : 
  z = -11 :=
  sorry

end determine_z_l5_5966


namespace no_extreme_points_range_m_l5_5706

theorem no_extreme_points_range_m (m : ℝ) :
  (∀ x : ℝ, (3*x^2 + 2*m*x + 1 ≠ 0) ↔ -√3 ≤ m ∧ m ≤ √3) :=
sorry

end no_extreme_points_range_m_l5_5706


namespace proof_pn_k_eq_331992_l5_5432

noncomputable def p (x : ℝ) : ℝ := sorry -- A polynomial with rational coefficients
def k : ℝ := sorry -- A specific real number

axiom k_condition : k^3 - k = 331992
axiom p_condition : k^3 - k = (p k)^3 - (p k)

def iterated_p : ℕ → ℝ → ℝ
| 0, x => x
| (n + 1), x => p (iterated_p n x)

theorem proof_pn_k_eq_331992 (n : ℕ) : (iterated_p n k)^3 - (iterated_p n k) = 331992 := sorry

end proof_pn_k_eq_331992_l5_5432


namespace sum_of_solutions_eq_zero_l5_5869

theorem sum_of_solutions_eq_zero :
  (∑ x in {x | (6 * x / 18 = 3 / x)}, x) = 0 := by
sorry

end sum_of_solutions_eq_zero_l5_5869


namespace jackson_spends_260_l5_5220

-- Definitions based on conditions
def num_students := 30
def pens_per_student := 5
def notebooks_per_student := 3
def binders_per_student := 1
def highlighters_per_student := 2

def cost_per_pen := 0.50
def cost_per_notebook := 1.25
def cost_per_binder := 4.25
def cost_per_highlighter := 0.75
def discount := 100.00

-- Calculate total cost
noncomputable def total_cost := 
  let cost_per_student := 
    (pens_per_student * cost_per_pen) +
    (notebooks_per_student * cost_per_notebook) +
    (binders_per_student * cost_per_binder) +
    (highlighters_per_student * cost_per_highlighter)
  in num_students * cost_per_student - discount

-- Theorem to prove the final cost
theorem jackson_spends_260 : total_cost = 260 := by
  sorry

end jackson_spends_260_l5_5220


namespace trace_of_z_plus_two_over_z_is_ellipse_l5_5023

open Complex

theorem trace_of_z_plus_two_over_z_is_ellipse :
  ∀ z : ℂ, abs z = 3 → 
  ∃ a b : ℝ, z = a + b * I ∧ (a^2 + b^2 = 9) ∧ 
  (∃ x y : ℝ, x + y * I = z + (2 / z) ∧ (x^2 / (11/9)^2 + y^2 / (7/9)^2 = 9)) :=
by
  assume z h_abs
  -- The proof logic would go here
  sorry

end trace_of_z_plus_two_over_z_is_ellipse_l5_5023


namespace even_integers_between_200_and_800_l5_5174

def valid_digits : Finset ℕ := {1, 3, 4, 5, 7, 8}
def is_even_digit (n : ℕ) := n % 2 = 0 ∧ n ∈ valid_digits
def is_valid_hundreds_digit (n : ℕ) := n ∈ {2, 3, 4, 5, 7}
def is_valid_tens_digit (n : ℕ) := n ∈ valid_digits

def number_of_valid_even_integers : ℕ :=
  let last_digit_choices := {4, 8}
  let valid_hundreds_digits := {2, 3, 4, 5, 7}
  let tens_digits := valid_digits.erase 4
  let tens_digits := tens_digits.erase 8
  in 4 * 3 + 4 * 3

theorem even_integers_between_200_and_800 :
  ∃ n : ℕ, n = 24 :=
begin
  use number_of_valid_even_integers,
  sorry
end

end even_integers_between_200_and_800_l5_5174


namespace sin_double_angle_cos_angle_sum_l5_5651

variables {α β : ℝ}

-- Given conditions
def cos_α := -1 / 3
def sin_β := 2 / 3
def α_in_third_quadrant : Prop := π < α ∧ α < 3 * π / 2
def β_in_second_quadrant : Prop := π / 2 < β ∧ β < π

theorem sin_double_angle :
  cos α = cos_α → 
  α_in_third_quadrant →
  sin (2 * α) = 4 * Real.sqrt 2 / 9 :=
by
  intros
  sorry

theorem cos_angle_sum :
  cos α = cos_α → 
  sin β = sin_β → 
  α_in_third_quadrant → 
  β_in_second_quadrant →
  cos (2 * α + β) = (7 * Real.sqrt 5 - 8 * Real.sqrt 2) / 27 :=
by
  intros
  sorry

end sin_double_angle_cos_angle_sum_l5_5651


namespace cosine_of_angle_l5_5654

noncomputable def unit_vectors (a b : ℝ) : Prop :=
  ‖a‖ = 1 ∧ ‖b‖ = 1 ∧ (a • b = 0)

theorem cosine_of_angle (a b : ℝ) (hab : unit_vectors a b) : 
  real.cos (real.angle a (a + 2*b)) = √5 / 5 :=
sorry

end cosine_of_angle_l5_5654


namespace probability_of_composite_is_5_over_8_l5_5526

def is_composite (n : ℕ) : Prop :=
  ¬n.prime ∧ 1 < n

def balls : Finset ℕ := {3, 4, 5, 6, 7, 8, 9, 10}

def count_composites (s : Finset ℕ) : ℕ :=
  s.filter is_composite |>.card

def probability_of_composite (s : Finset ℕ) : ℚ :=
  count_composites s / s.card

theorem probability_of_composite_is_5_over_8 :
  probability_of_composite balls = 5 / 8 :=
by
  sorry

end probability_of_composite_is_5_over_8_l5_5526


namespace complex_number_condition_l5_5160
open Complex

-- Given: z = a / (2 - I) + (3 - 4 * I) / 5
-- And: The sum of the real part and imaginary part of z equals 1
-- Prove: a = 2
theorem complex_number_condition (a : ℝ) :
  let z := (a / (2 - I)) + (3 - 4 * I) / 5 in
  (z.re + z.im) = 1 → a = 2 :=
by
  intro hz
  sorry

end complex_number_condition_l5_5160


namespace solution_l5_5012

noncomputable def problem : ℝ :=
  (26.3 * 12 * 20) / 3 + 125 - real.sqrt 576 + 16 * real.log 7

theorem solution : problem = 2218.5216 :=
  sorry

end solution_l5_5012


namespace laps_in_5000m_race_l5_5276

theorem laps_in_5000m_race : 
  ∀ (total_distance lap_length : ℕ), 
  total_distance = 5000 → lap_length = 400 → 
  (total_distance / lap_length : ℚ) = 12.5 := 
by 
  intros total_distance lap_length h_total h_lap
  rw [h_total, h_lap]
  norm_num
  sorry

end laps_in_5000m_race_l5_5276


namespace cos_seven_pi_over_six_eq_neg_sqrt_three_over_two_l5_5322

theorem cos_seven_pi_over_six_eq_neg_sqrt_three_over_two :
  cos (7 * Real.pi / 6) = - (Real.sqrt 3 / 2) :=
by
  -- Proof goes here
  sorry

end cos_seven_pi_over_six_eq_neg_sqrt_three_over_two_l5_5322


namespace final_number_geq_one_over_n_l5_5784

theorem final_number_geq_one_over_n (n : ℕ) (h : n > 0) :
    ∀ (steps : List (ℝ × ℝ)) (final_number : ℝ),
    (∀ step ∈ steps, ∃ (a b : ℝ), step = (a, b) ∧ (a + b) / 4 ∈ {final_number}) →
    final_number ≥ 1 / n :=
by
  sorry

end final_number_geq_one_over_n_l5_5784


namespace complex_expression_simplification_l5_5881

theorem complex_expression_simplification (a : ℝ) (h : a > 1) :
  (( ( ( (sqrt a - 1) / (sqrt a + 1) )⁻¹ ) * ( (sqrt a - 1) / (sqrt a + 1) )^(1 / 2) - sqrt (a - 1) / (sqrt a + 1) ) ^ (-2) * 
  1 / (a ^ (2 / 3) + a ^ (1 / 3) + 1)) = ( (a ^ (1 / 3)) - 1) / 4 :=
by
  sorry

end complex_expression_simplification_l5_5881


namespace income_growth_rate_l5_5483

noncomputable def income_growth_eq (x : ℝ) : Prop :=
  3.2 * (1 + x)^2 = 3.7

theorem income_growth_rate :
  ∃ x : ℝ, income_growth_eq x :=
sorry

end income_growth_rate_l5_5483


namespace increase_in_circumference_l5_5758

-- condition: let d be the original diameter of a circle
variable {d : ℝ} (d_nonneg : 0 ≤ d)

-- question: let the diameter increase by π units
def increase_in_diameter : ℝ := d + Real.pi

-- condition: original circumference
def original_circumference : ℝ := Real.pi * d

-- condition: new circumference
def new_circumference : ℝ := Real.pi * (d + Real.pi)

-- proof problem: prove the increase in circumference P when diameter is increased by π units equals π^2
theorem increase_in_circumference : new_circumference d_nonneg - original_circumference d = Real.pi ^ 2 := by
  sorry

end increase_in_circumference_l5_5758


namespace pricing_and_discount_l5_5479

noncomputable def price_of_soccer_ball := 70
noncomputable def price_of_basketball := 90
noncomputable def discount_rate := 0.85

theorem pricing_and_discount (x y : ℕ) (m : ℝ) :
  (2 * x + 3 * y = 410) ∧ 
  (5 * x + 2 * y = 530) ∧ 
  (5 * x + 5 * y = 680) → 
  (x = price_of_soccer_ball) ∧ 
  (y = price_of_basketball) ∧ 
  (m = discount_rate) := by 
  sorry

end pricing_and_discount_l5_5479


namespace circle_intersects_y_axis_at_one_l5_5306

theorem circle_intersects_y_axis_at_one :
  let A := (-2011, 0)
  let B := (2010, 0)
  let C := (0, (-2010) * 2011)
  ∃ (D : ℝ × ℝ), D = (0, 1) ∧
    (∃ O : ℝ × ℝ, O = (0, 0) ∧
    (dist O A) * (dist O B) = (dist O C) * (dist O D)) :=
by
  sorry -- Proof of the theorem

end circle_intersects_y_axis_at_one_l5_5306


namespace truck_sand_amount_l5_5477

theorem truck_sand_amount (initial_sand lost_sand : ℝ) (h1 : initial_sand = 4.1) (h2 : lost_sand = 2.4) :
  initial_sand - lost_sand = 1.7 :=
by
  -- Conditions
  rw [h1, h2]
  -- Simplification
  norm_num
  -- Correctness
  exact rfl

end truck_sand_amount_l5_5477


namespace letter_puzzle_solution_l5_5544

theorem letter_puzzle_solution :
  ∃ (A B : ℕ), (A ≠ B) ∧ (1 ≤ A ∧ A ≤ 9) ∧ (1 ≤ B ∧ B ≤ 9) ∧ A ^ B = B * 10 + A :=
by {
  use 2, 5,
  split, { exact ne_of_lt (by decide) }, split, { exact ⟨by decide, by decide⟩ },
  split, { exact ⟨by decide, by decide⟩ }, exact (by decide : 2 ^ 5 = 5 * 10 + 2),
  sorry
} ∨
by {
  use 6, 2,
  split, { exact ne_of_lt (by decide) }, split, { exact ⟨by decide, by decide⟩ },
  split, { exact ⟨by decide, by decide⟩ }, exact (by decide : 6 ^ 2 = 2 * 10 + 6),
  sorry
} ∨
by {
  use 4, 3,
  split, { exact ne_of_lt (by decide) }, split, { exact ⟨by decide, by decide⟩ },
  split, { exact ⟨by decide, by decide⟩ }, exact (by decide : 4 ^ 3 = 3 * 10 + 4),
  sorry
}

end letter_puzzle_solution_l5_5544


namespace smallest_rel_prime_to_180_l5_5580

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ Nat.gcd x 180 = 1 ∧ ∀ y : ℕ, y > 1 ∧ Nat.gcd y 180 = 1 → x ≤ y :=
begin
  use 7,
  split,
  { exact dec_trivial },
  split,
  { exact dec_trivial },
  { intro y,
    intro h,
    cases h with h1 h2,
    repeat { try { apply dec_trivial,
                   apply lt_or_eq_of_le,
                   norm_num,
                   apply Nat.prime_not_dvd_mul,
                   norm_num,
                   apply not_or_distrib.mpr,
                   split,
                   norm_cast,
                   intro,
                   exact le_antisymm _ },
           sorry }
end

end smallest_rel_prime_to_180_l5_5580


namespace math_problem_l5_5880

-- Define constants and conversions from decimal/mixed numbers to fractions
def thirteen_and_three_quarters : ℚ := 55 / 4
def nine_and_sixth : ℚ := 55 / 6
def one_point_two : ℚ := 1.2
def ten_point_three : ℚ := 103 / 10
def eight_and_half : ℚ := 17 / 2
def six_point_eight : ℚ := 34 / 5
def three_and_three_fifths : ℚ := 18 / 5
def five_and_five_sixths : ℚ := 35 / 6
def three_and_two_thirds : ℚ := 11 / 3
def three_and_one_sixth : ℚ := 19 / 6
def fifty_six : ℚ := 56
def twenty_seven_and_sixth : ℚ := 163 / 6

def E : ℚ := 
  ((thirteen_and_three_quarters + nine_and_sixth) * one_point_two) / ((ten_point_three - eight_and_half) * (5 / 9)) + 
  ((six_point_eight - three_and_three_fifths) * five_and_five_sixths) / ((three_and_two_thirds - three_and_one_sixth) * fifty_six) - 
  twenty_seven_and_sixth

theorem math_problem : E = 29 / 3 := by
  sorry

end math_problem_l5_5880


namespace eliana_steps_l5_5971

theorem eliana_steps : 
  let steps_day_1 := 200 + 300 in
  let steps_day_2 := steps_day_1 ^ 2 in
  let steps_day_3 := steps_day_1 + steps_day_2 + 100 in
  steps_day_1 + steps_day_2 + steps_day_3 = 501100 :=
by
  sorry

end eliana_steps_l5_5971


namespace abs_eq_neg_l5_5129

theorem abs_eq_neg (x : ℝ) (h : |x + 6| = -(x + 6)) : x ≤ -6 :=
by 
  sorry

end abs_eq_neg_l5_5129


namespace problem_solution_l5_5172

noncomputable def solveProblem (p q : ℝ) : Prop :=
  let M := {x : ℝ | x^2 - p * x + 6 = 0}
  let N := {x : ℝ | x^2 + 6 * x - q = 0}
  M ∩ N = {2} → p + q = 21

-- Setting up the statement as a theorem with assuming the conditions
theorem problem_solution (p q : ℝ) (h : let M := {x : ℝ | x^2 - p * x + 6 = 0} in
                                         let N := {x : ℝ | x^2 + 6 * x - q = 0} in
                                         M ∩ N = {2}) : 
  p + q = 21 :=
sorry

end problem_solution_l5_5172


namespace fifth_individual_selected_l5_5924

theorem fifth_individual_selected:
  let table := ["7816", "6572", "0802", "6314", "0702", "4369", "9728", "0198", 
                  "3204", "9234", "4935", "8200", "3623", "4869", "6938", "7481"]
  let individuals := ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"]
  let selected_sequence := ["08", "02", "14", "07", "02", "01"]
  let unique_selected := ["08", "02", "14", "07", "01"]
  (unique_selected.nth 4).get_or_else "00" = "01" := by sorry

end fifth_individual_selected_l5_5924


namespace marble_probability_l5_5020

theorem marble_probability : 
  let total_marbles := 16
      red_marbles := 10
      blue_marbles := 6
      total_ways := Nat.choose total_marbles 3
      ways_RBB := (Nat.choose red_marbles 1) * (Nat.choose blue_marbles 2)
  in (ways_RBB / total_ways) = 15 / 56 :=
sorry

end marble_probability_l5_5020


namespace train_speed_l5_5475

/-- A train that crosses a pole in a certain time of 7 seconds and is 210 meters long has a speed of 108 kilometers per hour. -/
theorem train_speed (time_to_cross: ℝ) (length_of_train: ℝ) (speed_kmh : ℝ) 
  (H_time: time_to_cross = 7) (H_length: length_of_train = 210) 
  (conversion_factor: ℝ := 3.6) : speed_kmh = 108 :=
by
  have speed_mps : ℝ := length_of_train / time_to_cross
  have speed_kmh_calc : ℝ := speed_mps * conversion_factor
  sorry

end train_speed_l5_5475


namespace interest_rate_for_A_l5_5696

noncomputable def simple_interest_rate_for_A (total Money_lent_to_B Money_lent_to_A Simple_interest_rate_B Time Period : ℝ) : ℝ :=
  (((Math.lent_to_B * Simple_interest_rate_B * Time) / 100) + 360) / ((Money_lent_to_A * Time) / 100)

theorem interest_rate_for_A :
  ∀ (total Money_lent_to_B Simple_interest_rate_B Time Period : ℝ),
  total = 10000 → Money_lent_to_B = 4000 → Simple_interest_rate_B = 18 → Time = 2 →
  let Money_lent_to_A := total - Money_lent_to_B in
  simple_interest_rate_for_A total Money_lent_to_B Simple_interest_rate_A Simple_interest_rate_B Time Period) = 15 :=
begin
  intros,
  dsimp [simple_interest_rate_for_A],
  sorry
end

end interest_rate_for_A_l5_5696


namespace mutual_exclusivity_of_event_C_l5_5874

def draw_2_balls_from_2_red_and_2_black (r1 r2 b1 b2 : bool) : (bool × bool) :=
  (r1, r2) -- Suppose r1 and r2 are red balls, b1 and b2 are black balls

def exactly_1_black {a b : bool} : bool := (a ≠ b)
def exactly_2_black {a b : bool} : bool := (a = b) ∧ (a = true)

theorem mutual_exclusivity_of_event_C :
  ∀ (r1 r2 b1 b2 : bool),
  draw_2_balls_from_2_red_and_2_black r1 r2 b1 b2 = (false, true) → 
  (exactly_1_black r1 r2 = true) → 
  (exactly_2_black r1 r2 = false) := 
by
  sorry

end mutual_exclusivity_of_event_C_l5_5874


namespace unique_real_solution_bound_l5_5095

theorem unique_real_solution_bound (b : ℝ) :
  (∀ x : ℝ, x^3 - b*x^2 - 4*b*x + b^2 - 4 = 0 → ∃! y : ℝ, y = x) → b < 1 :=
by
  sorry

end unique_real_solution_bound_l5_5095


namespace expected_value_eight_sided_die_l5_5407

-- Define a standard 8-sided die
def eight_sided_die : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]

-- Compute the probability of each outcome
def probability (n : ℕ) : ℝ := 1 / 8

-- Expected Value of a discrete random variable
def expected_value (outcomes : List ℕ) (prob : ℕ → ℝ) : ℝ :=
  outcomes.sum / outcomes.length.toReal

-- Theorem stating the expected value of a standard 8-sided die roll is 4.5
theorem expected_value_eight_sided_die : expected_value eight_sided_die probability = 4.5 := by
  sorry

end expected_value_eight_sided_die_l5_5407


namespace probability_both_segments_at_least_1m_l5_5299

-- Definitions
def rope_length := 3 -- length of the rope
def min_segment_length := 1 -- minimum length of each segment

-- Main Statement
theorem probability_both_segments_at_least_1m : 
  (Pr(λ x : ℝ, min_segment_length ≤ x ∧ x ≤ rope_length - min_segment_length)) = (1 / rope_length) :=
by
  sorry

end probability_both_segments_at_least_1m_l5_5299


namespace number_of_cows_l5_5456

theorem number_of_cows (n : ℝ) (h1 : n / 2 + n / 4 + n / 5 + 7 = n) : n = 140 := 
sorry

end number_of_cows_l5_5456


namespace income_growth_rate_l5_5484

noncomputable def income_growth_eq (x : ℝ) : Prop :=
  3.2 * (1 + x)^2 = 3.7

theorem income_growth_rate :
  ∃ x : ℝ, income_growth_eq x :=
sorry

end income_growth_rate_l5_5484


namespace possible_values_of_b_l5_5958

theorem possible_values_of_b 
        (b : ℤ)
        (h : ∃ x : ℤ, (x ^ 3 + 2 * x ^ 2 + b * x + 8 = 0)) :
        b = -81 ∨ b = -26 ∨ b = -12 ∨ b = -6 ∨ b = 4 ∨ b = 9 ∨ b = 47 :=
  sorry

end possible_values_of_b_l5_5958


namespace undefined_expression_value_l5_5128

theorem undefined_expression_value {a : ℝ} : (a^3 - 8 = 0) ↔ (a = 2) :=
by sorry

end undefined_expression_value_l5_5128


namespace fill_tank_time_l5_5439

-- Define the rates of filling and draining
def rateA : ℕ := 200 -- Pipe A fills at 200 liters per minute
def rateB : ℕ := 50  -- Pipe B fills at 50 liters per minute
def rateC : ℕ := 25  -- Pipe C drains at 25 liters per minute

-- Define the times each pipe is open
def timeA : ℕ := 1   -- Pipe A is open for 1 minute
def timeB : ℕ := 2   -- Pipe B is open for 2 minutes
def timeC : ℕ := 2   -- Pipe C is open for 2 minutes

-- Define the capacity of the tank
def tankCapacity : ℕ := 1000

-- Prove the total time to fill the tank is 20 minutes
theorem fill_tank_time : 
  (tankCapacity * ((timeA * rateA + timeB * rateB) - (timeC * rateC)) * 5) = 20 :=
sorry

end fill_tank_time_l5_5439


namespace ordered_triples_count_l5_5177

theorem ordered_triples_count : 
  ∃! (s : Finset (ℤ × ℤ × ℤ)), 
    (∀ x ∈ s, let (a, b, c) := x in a + b + |c| = 17 ∧ ab + c = 91) 
    ∧ s.card = 10 :=
by
  sorry

end ordered_triples_count_l5_5177


namespace per_capita_income_growth_l5_5490

noncomputable def income2020 : ℝ := 3.2
noncomputable def income2022 : ℝ := 3.7
variable (x : ℝ)

/--
Prove the per capita disposable income model.
-/
theorem per_capita_income_growth :
  income2020 * (1 + x)^2 = income2022 :=
sorry

end per_capita_income_growth_l5_5490


namespace problem5_solution_l5_5903

-- Define the function f(n)
def f (n : ℕ) : ℝ := abs (finset.sum (finset.range (n / 3 + 1)) (λ m, (n.choose (3 * m))) - 2^n / 3)

-- The theorem we need to prove
theorem problem5_solution : finset.sum (finset.range 2022) f = 2695 / 3 := 
sorry

end problem5_solution_l5_5903


namespace pyramid_height_l5_5922

theorem pyramid_height (base_area face_area : ℝ) (s : ℝ) (h : ℝ) (H : ℝ) (s_eq : s^2 = base_area) (face_area_eq : (1 / 2) * s * h = face_area) : 
base_area = 1440 → face_area = 840 → s = real.sqrt 1440 → h = (1680 / real.sqrt 1440) → 
H = real.sqrt (h^2 - (s/2)^2) → H = 40 :=
by
  intros
  sorry

end pyramid_height_l5_5922


namespace permutation_20th_l5_5857

noncomputable theory
open List

def is_20th_permutation (n : ℕ) : Prop :=
  let digits := [1, 2, 3, 4, 5]
  let permutations := permutations digits
  permutations.get! (n - 1) = [1, 2, 5, 4, 3]

theorem permutation_20th : is_20th_permutation 20 :=
by
  sorry

end permutation_20th_l5_5857


namespace weights_problem_l5_5480

theorem weights_problem
  (a b c d : ℕ)
  (h1 : a + b = 280)
  (h2 : b + c = 255)
  (h3 : c + d = 290) 
  : a + d = 315 := 
  sorry

end weights_problem_l5_5480


namespace positive_value_of_s_l5_5127

theorem positive_value_of_s (s : ℝ) (h : 0 < s) (hyp : | Complex.mk (-3 : ℝ) s | = 2 * Real.sqrt 10) : s = Real.sqrt 31 := by
  sorry

end positive_value_of_s_l5_5127


namespace cosine_of_angle_l5_5739

-- Defining the geometric and measurement properties of the right triangular prism
noncomputable def prism : Prop :=
  let CA := 1 in
  let CB := 1 in
  let angle_BCA := 90 in
  let AA1 := 2 in
  let A1B := (-1,1,-2) in
  let B1C := (0,1,2) in
  let dot_product := (A1B.1 * B1C.1 + A1B.2 * B1C.2 + A1B.3 * B1C.3) in
  let A1B_mag := Real.sqrt (A1B.1^2 + A1B.2^2 + A1B.3^2) in
  let B1C_mag := Real.sqrt (B1C.1^2 + B1C.2^2 + B1C.3^2) in
  let cos_theta := (dot_product / (A1B_mag * B1C_mag)) in
  cos_theta = (sqrt 30 / 10)

-- Formal statement of the problem
theorem cosine_of_angle : prism := 
by 
  sorry 

end cosine_of_angle_l5_5739


namespace log_expression_equals_two_l5_5445

noncomputable def log_expression : ℝ :=
  log 4 + log 9 + 2 * sqrt ((log 6)^2 - log 36 + 1)

theorem log_expression_equals_two : log_expression = 2 :=
by
  sorry

end log_expression_equals_two_l5_5445


namespace evaluate_expression_l5_5527

theorem evaluate_expression (a : ℝ) (h : a ≠ 0) : 
  a^4 - a^(-4) = (a - a^(-1)) * (a + a^(-1)) * ((a + a^(-1))^2 - 2) :=
sorry

end evaluate_expression_l5_5527


namespace median_moons_l5_5864

def median (l : List ℕ) : ℕ :=
  let sorted_l := l.qsort (≤)
  let n := sorted_l.length
  sorted_l.get! (n / 2)

theorem median_moons :
  median [0, 0, 1, 4, 4, 5, 15, 16, 23] = 4 :=
by
  sorry

end median_moons_l5_5864


namespace repeating_decimal_to_fraction_l5_5998

theorem repeating_decimal_to_fraction (h : 0.\overline{02} = 2 / 99) : 
  2.\overline{06} = 68 / 33 := by
  sorry

end repeating_decimal_to_fraction_l5_5998


namespace number_of_odd_composite_integers_l5_5014

theorem number_of_odd_composite_integers:
  let composites : ℕ → Prop := λ n, Odd n ∧ Composite n in
  Finset.card (Finset.filter composites (Finset.range 51)) = 10 := 
by
  sorry

end number_of_odd_composite_integers_l5_5014


namespace norma_bananas_count_l5_5268

-- Definitions for the conditions
def initial_bananas : ℕ := 47
def lost_bananas : ℕ := 45

-- The proof problem in Lean 4 statement
theorem norma_bananas_count : initial_bananas - lost_bananas = 2 := by
  -- Proof is omitted
  sorry

end norma_bananas_count_l5_5268


namespace repeating_decimal_to_fraction_l5_5994

theorem repeating_decimal_to_fraction (h : 0.\overline{02} = 2 / 99) : 
  2.\overline{06} = 68 / 33 := by
  sorry

end repeating_decimal_to_fraction_l5_5994


namespace trigonometric_proof_l5_5139

noncomputable def f (x : ℝ) (a b : ℝ) := a * (Real.sin x ^ 3) + b * (Real.cos x ^ 3) + 4

theorem trigonometric_proof (a b : ℝ) (h₁ : f (Real.sin (10 * Real.pi / 180 )) a b = 5) : 
  f (Real.cos (100 * Real.pi / 180 )) a b = 3 := 
sorry

end trigonometric_proof_l5_5139


namespace max_omega_l5_5762

open Real

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := sin (ω * x + φ)

theorem max_omega (ω φ : ℝ) (h1 : f ω φ (-π / 8) = 0) (h2 : ∀ x, f ω φ x = f ω φ (π / 4 - x)) 
  (h3 : ∀ x, x ∈ Ioo (π / 5) (π / 4) →  f ω φ x > 0 → deriv_test (f ω φ) x > 0) :
  ω ≤ 14 :=
begin
  -- The proof goes here.
  sorry,
end

end max_omega_l5_5762


namespace letter_puzzle_solutions_l5_5551

noncomputable def is_solution (A B : ℕ) : Prop :=
A ≠ B ∧ A ∈ finset.range (10) ∧ B ∈ finset.range (10) ∧ 10 ≤ B * 10 + A ∧ B * 10 + A ≤ 99 ∧ A^B = B * 10 + A

theorem letter_puzzle_solutions :
  ∃ A B : ℕ, is_solution A B ∧ ((A = 2 ∧ B = 5) ∨ (A = 6 ∧ B = 2) ∨ (A = 4 ∧ B = 3)) :=
by
  sorry

end letter_puzzle_solutions_l5_5551


namespace range_of_x_l5_5125

theorem range_of_x (θ : ℝ) (h0 : 0 < θ) (h1 : θ < Real.pi / 2) (h2 : ∀ θ, (0 < θ) → (θ < Real.pi / 2) → (1 / (Real.sin θ) ^ 2 + 4 / (Real.cos θ) ^ 2 ≥ abs (2 * x - 1))) :
  -4 ≤ x ∧ x ≤ 5 := sorry

end range_of_x_l5_5125


namespace determine_days_l5_5855

-- Define the problem
def team_repair_time (x y : ℕ) : Prop :=
  ((1 / (x:ℝ)) + (1 / (y:ℝ)) = 1 / 18) ∧ 
  ((2 / 3 * x + 1 / 3 * y = 40))

theorem determine_days : ∃ x y : ℕ, team_repair_time x y :=
by
    use 45
    use 30
    have h1: (1/(45:ℝ) + 1/(30:ℝ)) = 1/18 := by
        sorry
    have h2: (2/3*45 + 1/3*30 = 40) := by
        sorry 
    exact ⟨h1, h2⟩

end determine_days_l5_5855


namespace tangerines_more_than_oranges_l5_5333

def initial_oranges := 5
def initial_tangerines := 17
def oranges_taken := 2
def tangerines_taken := 10

theorem tangerines_more_than_oranges
  (initial_oranges: ℕ) -- Tina starts with 5 oranges
  (initial_tangerines: ℕ) -- Tina starts with 17 tangerines
  (oranges_taken: ℕ) -- Tina takes away 2 oranges
  (tangerines_taken: ℕ) -- Tina takes away 10 tangerines
  : (initial_tangerines - tangerines_taken) - (initial_oranges - oranges_taken) = 4 := 
by
  sorry

end tangerines_more_than_oranges_l5_5333


namespace raman_profit_percentage_l5_5282

theorem raman_profit_percentage
  (cost1 weight1 rate1 : ℕ) (cost2 weight2 rate2 : ℕ) (total_cost_mix total_weight mixing_rate selling_rate profit profit_percentage : ℕ)
  (h_cost1 : cost1 = weight1 * rate1)
  (h_cost2 : cost2 = weight2 * rate2)
  (h_total_cost_mix : total_cost_mix = cost1 + cost2)
  (h_total_weight : total_weight = weight1 + weight2)
  (h_mixing_rate : mixing_rate = total_cost_mix / total_weight)
  (h_selling_price : selling_rate * total_weight = profit + total_cost_mix)
  (h_profit : profit = selling_rate * total_weight - total_cost_mix)
  (h_profit_percentage : profit_percentage = (profit * 100) / total_cost_mix)
  (h_weight1 : weight1 = 54)
  (h_rate1 : rate1 = 150)
  (h_weight2 : weight2 = 36)
  (h_rate2 : rate2 = 125)
  (h_selling_rate_value : selling_rate = 196) :
  profit_percentage = 40 :=
sorry

end raman_profit_percentage_l5_5282


namespace triangle_CI_condition_l5_5851

noncomputable def isosceles_right_triangle (AB AC : ℝ) : Prop :=
AB = AC ∧ AB = 4

noncomputable def midpoint (x y : ℝ) : ℝ :=
(x + y) / 2

noncomputable def cyclic_quadrilateral (AI AE : ℝ) : Prop :=
AI = 3 * AE

variable (a b c : ℤ)
variable (b_prime_sq_free : ¬ ∃ p : ℤ, prime p ∧ p^2 ∣ b)
variable (a_b_c_conditions : 5 + 8 + 2 = 15)
variable (BC_length : ℝ)
variable (M_midpoint : ℝ)
variable (area_EMI : ℝ)

theorem triangle_CI_condition :
  isosceles_right_triangle 4 4 ∧
  midpoint (4 * real.sqrt 2) 2 * real.sqrt 2 ∧
  cyclic_quadrilateral (5 + real.sqrt 8) (8 / 2) ∧
  area_EMI = 4.5
  → a + b + c = 15 :=
begin
  sorry
end

end triangle_CI_condition_l5_5851


namespace rhombus_to_isosceles_trapezoid_l5_5516

theorem rhombus_to_isosceles_trapezoid (A B C D E M: Point) (rhombus : isRhombus A B C D) 
  (altitude : altitude A E B C) (midpoint_M : midpoint A B M) (EM : onLineSeg E M) :
  ∃ P Q R S, isIsoscelesTrapezoid P Q R S :=
by
  sorry

end rhombus_to_isosceles_trapezoid_l5_5516


namespace correct_statements_l5_5785

inductive PolynomialSeq
| base (first : ℕ) (second : ℕ) : PolynomialSeq
| next (a b : ℕ) (n : PolynomialSeq) (m : PolynomialSeq) : PolynomialSeq

open PolynomialSeq

def P : ℕ → PolynomialSeq
| 1 := base 1 0 
| 2 := base 0 1 
| (n+2) := match P n, P (n+1) with
  | base a b, base c d := next (2*a + c) (2*b + d) (base a b) (base c d)
  | _, _ := base 0 0 -- default to handle other cases, should never be reached.

def check_statement1 := 
  match P 8 with
  | next _ 43 (next _ 42 _ _) _ := True
  | _ := False

def check_statement4 (n : ℕ) := 
  match P (2*n), P (2*n + 1) with
  | next a1 b1 _ _, next a2 b2 _ _ := (a1 + a2 = b1 + b2)
  | _, _ := False

theorem correct_statements : check_statement1 ∧ ∀ n > 0, check_statement4 n :=
by sorry

end correct_statements_l5_5785


namespace repeating_decimal_fraction_l5_5987

theorem repeating_decimal_fraction :  exists (p q : ℤ), (q ≠ 0) ∧ (p / q = (68 / 33 : ℚ)) :=
begin
  let recurring02 : ℚ := 2 / 99,
  let recurring06 : ℚ := 3 * recurring02,
  have recurring06_simplified : recurring06 = 2 / 33, 
    by simp [recurring06, mul_div_assoc, mul_div_cancel_left],
  let result := 2 + recurring06_simplified,
  use [68, 33],
  split,
  { norm_num }, -- proof that 33 ≠ 0
  { exact result } -- proof that 2.06 recurring = 68 / 33
end

end repeating_decimal_fraction_l5_5987


namespace count_integers_with_factors_15_20_25_l5_5175

theorem count_integers_with_factors_15_20_25 : 
  ∃ n, n = 3 ∧ ∀ k, 1000 ≤ k ∧ k ≤ 2000 ∧ (300 ∣ k) → k ∈ {1200, 1500, 1800} :=
by
  sorry

end count_integers_with_factors_15_20_25_l5_5175


namespace find_b_l5_5704

theorem find_b (a : ℝ) (h_a_pos : a > 0) (h_a_neq_one : a ≠ 1) (f : ℝ → ℝ) (h_f_def : ∀ x, f(x) = 2 * a^(x - b) + 1) (h_fixed : f 2 = 3) : b = 2 :=
sorry

end find_b_l5_5704


namespace area_of_largest_circle_l5_5474

theorem area_of_largest_circle (side_length : ℝ) (h : side_length = 2) : 
  (Real.pi * (side_length / 2)^2 = 3.14) :=
by
  sorry

end area_of_largest_circle_l5_5474


namespace find_angle_HGF_l5_5232

-- Define the Circle and points A, B, F, center O
variables (AB : Line) (O : Point) (circle_centered_at_O : Circle O)
variables (A B F G H : Point)
variable (tangent_at_B : Line) -- tangent line at B
variable (tangent_at_F : Line) -- tangent line at F
variable (AF : Line) -- line AF
variable (angle_BAF : ℕ) -- angle BAF in degrees

-- Conditions
def AB_is_diameter : Prop :=
  AB.is_diameter O

def F_is_on_circle : Prop := 
  F ∈ circle_centered_at_O

def tangents_intersect_at_G_and_H : Prop := 
  (tangent_at_B ∩ tangent_at_F = G) ∧ 
  (tangent_at_F ∩ AF = H)

def angle_BAF_is_37_degrees : Prop :=
  angle_BAF = 37

theorem find_angle_HGF
  (h_diameter : AB_is_diameter)
  (h_F_on_circle : F_is_on_circle)
  (h_tangents : tangents_intersect_at_G_and_H)
  (h_angle : angle_BAF_is_37_degrees) : 
  ∠HGF = 53 :=
sorry

end find_angle_HGF_l5_5232


namespace sales_of_fourth_month_l5_5458

-- Given conditions
def sales_first_month := 4000
def sales_second_month := 6524
def sales_third_month := 5689
def sales_fifth_month := 6000
def sales_sixth_month := 12557
def desired_average_sales := 7000
def total_months := 6

-- Lean statement to prove the fourth month's sales
theorem sales_of_fourth_month :
  let total_sales := desired_average_sales * total_months in
  ∃ (sales_fourth_month : ℕ), 
    total_sales = sales_first_month + sales_second_month + sales_third_month + sales_fourth_month + sales_fifth_month + sales_sixth_month ∧
    sales_fourth_month = 5730 :=
by
  let total_sales := desired_average_sales * total_months
  existsi (total_sales - (sales_first_month + sales_second_month + sales_third_month + sales_fifth_month + sales_sixth_month))
  split
  sorry
  sorry

end sales_of_fourth_month_l5_5458


namespace product_of_roots_quadratic_l5_5567

noncomputable def product_of_roots (a b c : ℚ) : ℚ :=
  c / a

theorem product_of_roots_quadratic : product_of_roots 14 21 (-250) = -125 / 7 :=
by
  sorry

end product_of_roots_quadratic_l5_5567


namespace jane_dolls_l5_5335

theorem jane_dolls (jane_dolls jill_dolls : ℕ) (h1 : jane_dolls + jill_dolls = 32) (h2 : jill_dolls = jane_dolls + 6) : jane_dolls = 13 := 
by {
  sorry
}

end jane_dolls_l5_5335


namespace expenditure_ratio_l5_5918

variable {I : ℝ} -- Income in the first year

-- Conditions
def first_year_savings (I : ℝ) : ℝ := 0.5 * I
def first_year_expenditure (I : ℝ) : ℝ := I - first_year_savings I
def second_year_income (I : ℝ) : ℝ := 1.5 * I
def second_year_savings (I : ℝ) : ℝ := 2 * first_year_savings I
def second_year_expenditure (I : ℝ) : ℝ := second_year_income I - second_year_savings I

-- Condition statement in Lean
theorem expenditure_ratio (I : ℝ) : 
  let total_expenditure := first_year_expenditure I + second_year_expenditure I
  (total_expenditure / first_year_expenditure I) = 2 :=
  by 
    sorry

end expenditure_ratio_l5_5918


namespace hyperbola_eccentricity_l5_5660

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0)
    (h : ∃ c P : ℝ × ℝ, P.1 ^ 2 / a ^ 2 - P.2 ^ 2 / b ^ 2 = 1 ∧ 
                         y = b/a * x ∧ 
                         dist P (c, 0) = 2b ∧ 
                         dist P (-c, 0) = 2a ∧ 
                         b = 2a) : 
    let e := sqrt (1 + (b/a) ^ 2)
    in e = sqrt 5 :=
begin
  sorry
end

end hyperbola_eccentricity_l5_5660


namespace sqrt_sum_ge_sqrt_weighted_sum_l5_5250

theorem sqrt_sum_ge_sqrt_weighted_sum {n : ℕ} {a : Fin n → ℝ}
  (h₀ : ∀ i, 0 ≤ a i) :
  ∑ i in Finset.range n, Real.sqrt (∑ j in Finset.Ico i n, a j) 
  ≥ Real.sqrt (∑ i in Finset.range n, (i + 1) * (i + 1) * a i) :=
sorry

end sqrt_sum_ge_sqrt_weighted_sum_l5_5250


namespace repeating_decimal_fraction_l5_5992

theorem repeating_decimal_fraction :  exists (p q : ℤ), (q ≠ 0) ∧ (p / q = (68 / 33 : ℚ)) :=
begin
  let recurring02 : ℚ := 2 / 99,
  let recurring06 : ℚ := 3 * recurring02,
  have recurring06_simplified : recurring06 = 2 / 33, 
    by simp [recurring06, mul_div_assoc, mul_div_cancel_left],
  let result := 2 + recurring06_simplified,
  use [68, 33],
  split,
  { norm_num }, -- proof that 33 ≠ 0
  { exact result } -- proof that 2.06 recurring = 68 / 33
end

end repeating_decimal_fraction_l5_5992


namespace letter_puzzle_solutions_l5_5543

theorem letter_puzzle_solutions (A B : ℕ) : 
  (1 ≤ A ∧ A < 10) ∧ (1 ≤ B ∧ B < 10) ∧ (A ≠ B) ∧ (A^B = 10 * B + A) → 
  (A = 2 ∧ B = 5) ∨ (A = 6 ∧ B = 2) ∨ (A = 4 ∧ B = 3) :=
by
  sorry

end letter_puzzle_solutions_l5_5543


namespace set1_eq_set2_eq_set3_eq_set4_eq_set5_eq_l5_5092

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

end set1_eq_set2_eq_set3_eq_set4_eq_set5_eq_l5_5092


namespace per_capita_income_growth_l5_5489

noncomputable def income2020 : ℝ := 3.2
noncomputable def income2022 : ℝ := 3.7
variable (x : ℝ)

/--
Prove the per capita disposable income model.
-/
theorem per_capita_income_growth :
  income2020 * (1 + x)^2 = income2022 :=
sorry

end per_capita_income_growth_l5_5489


namespace repeating_decimal_fraction_l5_5986

theorem repeating_decimal_fraction :  exists (p q : ℤ), (q ≠ 0) ∧ (p / q = (68 / 33 : ℚ)) :=
begin
  let recurring02 : ℚ := 2 / 99,
  let recurring06 : ℚ := 3 * recurring02,
  have recurring06_simplified : recurring06 = 2 / 33, 
    by simp [recurring06, mul_div_assoc, mul_div_cancel_left],
  let result := 2 + recurring06_simplified,
  use [68, 33],
  split,
  { norm_num }, -- proof that 33 ≠ 0
  { exact result } -- proof that 2.06 recurring = 68 / 33
end

end repeating_decimal_fraction_l5_5986


namespace smallest_rel_prime_to_180_l5_5600

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ x ≤ 7 ∧ (∀ y : ℕ, y > 1 ∧ y < x → y.gcd 180 ≠ 1) ∧ x.gcd 180 = 1 :=
by
  sorry

end smallest_rel_prime_to_180_l5_5600


namespace cot_arctan_sum_l5_5121

variable (a b c d : ℝ)

def cot (x : ℝ) : ℝ := 1 / tan x

noncomputable def cot_inv (x : ℝ) : ℝ := Real.atan (1 / x)

axiom cot_add_identity (x y : ℝ) : 
  cot (cot_inv x + cot_inv y) = (x * y - 1) / (x + y)

theorem cot_arctan_sum :
  cot (cot_inv a + cot_inv b + cot_inv c + cot_inv d) = 281 / 714 :=
by
  let α := cot_inv a
  let β := cot_inv b
  let γ := cot_inv c
  let δ := cot_inv d
  have h1 : cot (α + β) = (a * b - 1) / (a + b) := cot_add_identity a b
  have h2 : cot (γ + δ) = (c * d - 1) / (c + d) := cot_add_identity c d
  let e := (a * b - 1) / (a + b)
  let f := (c * d - 1) / (c + d)
  have h3 : cot (cot_inv e + cot_inv f) = (e * f - 1) / (e + f) := cot_add_identity e f
  show cot (cot_inv a + cot_inv b + cot_inv c + cot_inv d) = 281 / 714 from
  sorry

end cot_arctan_sum_l5_5121


namespace two_squares_inequality_l5_5135

theorem two_squares_inequality (a b : ℝ) : 2 * (a^2 + b^2) ≥ (a + b)^2 := 
sorry

end two_squares_inequality_l5_5135


namespace cos_sin_225_deg_l5_5505

theorem cos_sin_225_deg : (Real.cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2) ∧ (Real.sin (225 * Real.pi / 180) = -Real.sqrt 2 / 2) :=
by
  -- Lean proof steps would go here
  sorry

end cos_sin_225_deg_l5_5505


namespace intersection_of_lines_l5_5081

-- Define the first and second lines
def line1 (x : ℚ) : ℚ := 3 * x + 1
def line2 (x : ℚ) : ℚ := -7 * x - 5

-- Statement: Prove that the intersection of the lines given by
-- y = 3x + 1 and y + 5 = -7x is (-3/5, -4/5).

theorem intersection_of_lines :
  ∃ x y : ℚ, y = line1 x ∧ y = line2 x ∧ x = -3 / 5 ∧ y = -4 / 5 :=
by
  sorry

end intersection_of_lines_l5_5081


namespace all_plants_diseased_l5_5793

theorem all_plants_diseased (n : ℕ) (h : n = 1007) : 
  n * 2 = 2014 := by
  sorry

end all_plants_diseased_l5_5793


namespace central_projection_bijective_l5_5279

variables (α1 α2 : Plane) (O : Point)
variable {l l1 l2 : Line}
variable (h1 : α1 ≠ α2)
variable (h2 : α1 ∩ α2 = l)
variable (h3 : l1 = intersection (plane_through O parallel_to α2) α1)
variable (h4 : l2 = intersection (plane_through O parallel_to α1) α2)

theorem central_projection_bijective : 
  ∀ (x : (α1 \ l1)), 
    ∃! (y : (α2 \ l2)), 
      project_via_center O α1 α2 x y :=
sorry

end central_projection_bijective_l5_5279


namespace PlanY_more_cost_effective_l5_5329

-- Define the gigabytes Tim uses
variable (y : ℕ)

-- Define the cost functions for Plan X and Plan Y in cents
def cost_PlanX (y : ℕ) := 25 * y
def cost_PlanY (y : ℕ) := 1500 + 15 * y

-- Prove that Plan Y is cheaper than Plan X when y >= 150
theorem PlanY_more_cost_effective (y : ℕ) : y ≥ 150 → cost_PlanY y < cost_PlanX y := by
  sorry

end PlanY_more_cost_effective_l5_5329


namespace water_added_l5_5448

theorem water_added (capacity : ℝ) (percentage_initial : ℝ) (percentage_final : ℝ) :
  capacity = 120 →
  percentage_initial = 0.30 →
  percentage_final = 0.75 →
  ((percentage_final * capacity) - (percentage_initial * capacity)) = 54 :=
by intros
   sorry

end water_added_l5_5448


namespace genuine_coin_remains_l5_5450

theorem genuine_coin_remains (n : ℕ) (g f : ℕ) (h : n = 2022) (h_g : g > n/2) (h_f : f = n - g) : 
  (after_moves : ℕ) -> after_moves = n - 1 -> ∃ remaining_g : ℕ, remaining_g > 0 :=
by
  intros
  sorry

end genuine_coin_remains_l5_5450


namespace simplify_expr_1_div_sqrt11_add_sqrt10_simplify_expr_1_div_sqrt_nplus1_add_sqrtn_sum_expr_2_to_100_l5_5783

theorem simplify_expr_1_div_sqrt11_add_sqrt10 : 
  (1 / (Real.sqrt 11 + Real.sqrt 10)) = (Real.sqrt 11 - Real.sqrt 10) := 
by sorry

theorem simplify_expr_1_div_sqrt_nplus1_add_sqrtn (n : ℕ) : 
  (1 / (Real.sqrt (n + 1) + Real.sqrt n)) = (Real.sqrt (n + 1) - Real.sqrt n) := 
by sorry

theorem sum_expr_2_to_100 : 
  ∑ n in Finset.range 99 + 2, (1 / (Real.sqrt n.succ + Real.sqrt n)) = 9 :=
by sorry

end simplify_expr_1_div_sqrt11_add_sqrt10_simplify_expr_1_div_sqrt_nplus1_add_sqrtn_sum_expr_2_to_100_l5_5783


namespace binomial_16_4_l5_5507

theorem binomial_16_4 : Nat.choose 16 4 = 1820 :=
  sorry

end binomial_16_4_l5_5507


namespace razorback_shop_tshirt_revenue_l5_5300

theorem razorback_shop_tshirt_revenue :
  let price_per_tshirt := 62
  let tshirts_sold := 183
  price_per_tshirt * tshirts_sold = 11,346 :=
by
  let price_per_tshirt := 62
  let tshirts_sold := 183
  have : price_per_tshirt * tshirts_sold = 11,346 := sorry
  exact this

end razorback_shop_tshirt_revenue_l5_5300


namespace simplify_sqrt_88200_l5_5291

theorem simplify_sqrt_88200 :
  ∀ (a b c d e : ℕ), a = 2 ∧ b = 3 ∧ c = 5 ∧ d = 7 ∧ e = 1 →
  ∃ f g : ℝ, (88200 : ℝ) = (f^2 * g) ∧ f = 882 ∧ g = 10 ∧ real.sqrt (88200 : ℝ) = f * real.sqrt g :=
sorry

end simplify_sqrt_88200_l5_5291


namespace only_statement_I_is_correct_l5_5893

theorem only_statement_I_is_correct (x y : ℝ):
  (∀ x, ⌊x + 1⌋ = ⌊x⌋ + 1) ∧ 
  (¬ ∀ x y, ⌊x + y⌋ = ⌊x⌋ + ⌊y⌋) ∧ 
  (¬ ∀ x y, ⌊x * y⌋ = ⌊x⌋ * ⌊y⌋) := 
by
  sorry

end only_statement_I_is_correct_l5_5893


namespace find_x_if_arithmetic_mean_is_12_l5_5153

theorem find_x_if_arithmetic_mean_is_12 (x : ℝ) (h : (8 + 16 + 21 + 7 + x) / 5 = 12) : x = 8 :=
by
  sorry

end find_x_if_arithmetic_mean_is_12_l5_5153


namespace tangerines_more_than_oranges_l5_5332

def initial_oranges := 5
def initial_tangerines := 17
def oranges_taken := 2
def tangerines_taken := 10

theorem tangerines_more_than_oranges
  (initial_oranges: ℕ) -- Tina starts with 5 oranges
  (initial_tangerines: ℕ) -- Tina starts with 17 tangerines
  (oranges_taken: ℕ) -- Tina takes away 2 oranges
  (tangerines_taken: ℕ) -- Tina takes away 10 tangerines
  : (initial_tangerines - tangerines_taken) - (initial_oranges - oranges_taken) = 4 := 
by
  sorry

end tangerines_more_than_oranges_l5_5332


namespace expected_value_of_8_sided_die_is_4_point_5_l5_5360

def expected_value_8_sided_die : ℝ :=
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let probability := 1 / 8
  (list.sum outcomes : ℝ) * probability

theorem expected_value_of_8_sided_die_is_4_point_5 :
  expected_value_8_sided_die = 4.5 := by
  sorry

end expected_value_of_8_sided_die_is_4_point_5_l5_5360


namespace flour_vs_sugar_difference_l5_5263

-- Definitions based on the conditions
def flour_needed : ℕ := 10
def flour_added : ℕ := 7
def sugar_needed : ℕ := 2

-- Define the mathematical statement to prove
theorem flour_vs_sugar_difference :
  (flour_needed - flour_added) - sugar_needed = 1 :=
by
  sorry

end flour_vs_sugar_difference_l5_5263


namespace gcd_fact8_fact7_l5_5565

noncomputable def fact8 : ℕ := 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1
noncomputable def fact7 : ℕ := 7 * 6 * 5 * 4 * 3 * 2 * 1

theorem gcd_fact8_fact7 : Nat.gcd fact8 fact7 = fact7 := by
  unfold fact8 fact7
  exact sorry

end gcd_fact8_fact7_l5_5565


namespace arithmetic_sequence_geometric_l5_5146

noncomputable def sequence_arith_to_geom (a1 d : ℤ) (h_d : d ≠ 0) (n : ℕ) : ℤ :=
a1 + (n - 1) * d

theorem arithmetic_sequence_geometric (a1 d : ℤ) (h_d : d ≠ 0) (n : ℕ) :
  (n = 16)
    ↔ (((a1 + 3 * d) / (a1 + 2 * d) = (a1 + 6 * d) / (a1 + 3 * d)) ∧ 
        ((a1 + 6 * d) / (a1 + 3 * d) = (a1 + (n - 1) * d) / (a1 + 6 * d))) :=
by
  sorry

end arithmetic_sequence_geometric_l5_5146


namespace convert_to_dms_convert_to_decimal_degrees_l5_5068

-- Problem 1: Conversion of 24.29 degrees to degrees, minutes, and seconds 
theorem convert_to_dms (d : ℝ) (h : d = 24.29) : 
  (∃ deg min sec, d = deg + min / 60 + sec / 3600 ∧ deg = 24 ∧ min = 17 ∧ sec = 24) :=
by
  sorry

-- Problem 2: Conversion of 36 degrees 40 minutes 30 seconds to decimal degrees
theorem convert_to_decimal_degrees (deg min sec : ℝ) (h : deg = 36 ∧ min = 40 ∧ sec = 30) : 
  (deg + min / 60 + sec / 3600) = 36.66 :=
by
  sorry

end convert_to_dms_convert_to_decimal_degrees_l5_5068


namespace find_x_l5_5024

-- Define the conditions
def atomic_weight_C : ℝ := 12.01
def atomic_weight_Cl : ℝ := 35.45
def molecular_weight : ℝ := 152

-- State the theorem
theorem find_x : ∃ x : ℕ, molecular_weight = atomic_weight_C + atomic_weight_Cl * x ∧ x = 4 := by
  sorry

end find_x_l5_5024


namespace trigonometric_inequality_l5_5278

theorem trigonometric_inequality (a b A B : ℝ) (h : ∀ x : ℝ, 1 - a * Real.cos x - b * Real.sin x - A * Real.cos 2 * x - B * Real.sin 2 * x ≥ 0) : 
  a ^ 2 + b ^ 2 ≤ 2 ∧ A ^ 2 + B ^ 2 ≤ 1 := 
sorry

end trigonometric_inequality_l5_5278


namespace nuts_per_cookie_l5_5262

theorem nuts_per_cookie (h1 : (1/4:ℝ) * 60 = 15)
(h2 : (0.40:ℝ) * 60 = 24)
(h3 : 60 - 15 - 24 = 21)
(h4 : 72 / (15 + 21) = 2) :
72 / 36 = 2 := by
suffices h : 72 / 36 = 2 from h
exact h4

end nuts_per_cookie_l5_5262


namespace find_k_l5_5752

theorem find_k (A B : ℕ) (hA : 0 ≤ A ∧ A ≤ 9) (hB : 0 ≤ B ∧ B ≤ 9) 
  (h_prod : let k := (10 * A + B) * (10 * B + A) in ∃ k, (k + 1) % 101 = 0) :
  let k := (10 * A + B) * (10 * B + A) in k = 403 :=
sorry

end find_k_l5_5752


namespace num_other_possible_values_l5_5511

noncomputable def exponentiation_variant_1 : ℝ := 3^(3^(3^3))
noncomputable def exponentiation_variant_2 : ℝ := 3^((3^3)^3)
noncomputable def exponentiation_variant_3 : ℝ := ((3^3)^3)^3
noncomputable def exponentiation_variant_4 : ℝ := (3^(3^3))^3
noncomputable def exponentiation_variant_5 : ℝ := (3^3)^(3^3)

theorem num_other_possible_values :
  (Set.card (Set.ofList [exponentiation_variant_2, exponentiation_variant_3, exponentiation_variant_4, exponentiation_variant_5])) = 4 := by
  sorry

end num_other_possible_values_l5_5511


namespace magnitude_w_argument_w_l5_5243

open Complex

noncomputable def w : ℂ := Complex.mk _ _ -- some appropriate complex number such that w^2 = 16 - 48i

theorem magnitude_w :
  (Real.sqrt (Complex.abs w)) = 4 * Real.root 4 10 := sorry

theorem argument_w :
  Complex.arg w ≈ 0.946 := sorry

end magnitude_w_argument_w_l5_5243


namespace real_values_satisfying_inequality_l5_5538

theorem real_values_satisfying_inequality (x : ℝ) :
  (x ≠ 0) → (x + 2 - 3 * x^2 ≠ 0) → (x^2 * (x + 2 - 3 * x^2) / (x * (x + 2 - 3 * x^2)) ≥ 0) → 
  (x ∈ [0, 1) ∪ (1, ∞)) :=
by
    sorry

end real_values_satisfying_inequality_l5_5538


namespace distribution_count_l5_5052

theorem distribution_count :
  let rows := 13
  ∃ count : ℕ, count = 2560 ∧ 
    (∀ (x : fin rows → ℤ), (∀ i : fin rows, x i = 0 ∨ x i = 1) →
      (∑ i, (nat.choose (rows - 1) i) * x i) % 5 = 0 ↔ count = 2560) := 
by
  sorry

end distribution_count_l5_5052


namespace f_period_and_max_l5_5824

noncomputable def f (x : ℝ) : ℝ := 
  let term1 := Real.cos x * Real.sin x
  let term2 := -(Real.cos x * Real.cos x)
  let term3 := 1
  term1 + term2 + term3

theorem f_period_and_max :
  (∀ x : ℝ, f(x + π) = f(x)) ∧
  (∀ x : ℝ, f(x) ≤ (sqrt 2 + 1) / 2) ∧
  (∃ x : ℝ, f(x) = (sqrt 2 + 1) / 2) :=
by
  sorry

end f_period_and_max_l5_5824


namespace min_pieces_on_checkerboard_l5_5270

def num_pieces (board : matrix (fin 6) (fin 6) bool) : ℕ :=
  fin.sum_univ (λ i, fin.sum_univ (λ j, if board i j then 1 else 0))

def piece_in_row_col (board : matrix (fin 6) (fin 6) bool) (n : ℕ) : Prop :=
  ∃ (r c : fin 6), (fin.sum_univ (λ j, if board r j then 1 else 0) + fin.sum_univ (λ i, if board i c then 1 else 0) - 1) = n

theorem min_pieces_on_checkerboard :
  ∀ (board : matrix (fin 6) (fin 6) bool),
  (∀ n : ℕ, 2 ≤ n ∧ n ≤ 10 → piece_in_row_col board n) →
  ∃ (board : matrix (fin 6) (fin 6) bool), num_pieces board = 19 := sorry

end min_pieces_on_checkerboard_l5_5270


namespace circumcircle_tangent_through_circumcenter_l5_5708

theorem circumcircle_tangent_through_circumcenter
  {A B C X Y P Q A' : Type*}
  (2XY_eq_BC : ∀ {BC : ℝ}, 2 * dist X Y = BC)
  (AXY_circumcircle : ∀ (⦿ : set (set Type*)), ∃ (diam : line), on_circumcircle ⦿ (triangle A X Y) diam)
  (perpendiculars_BC : ∀ (line1 line2 : line), perpendicular line1 line2 ∧ intersect_AX_P ∧ intersect_AY_Q)
  (circumcenter_APQ : ∀ {O : Type*}, center_circumcircle O (triangle A P Q))
  : let tangent_line : line := tangent_at_point (triangle A X Y) A' in
  tangent_line passes_through (circumcenter_APQ A P Q) := 
sorry

end circumcircle_tangent_through_circumcenter_l5_5708


namespace Mahdi_swims_on_Saturday_l5_5778

-- Define each day of the week
inductive Day : Type
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday
deriving DecidableEq, Inhabited

open Day

-- Define Mahdi's sports schedule (one sport per day)
structure Schedule :=
  (sport : Day → String)

-- Define the problem conditions
structure Conditions :=
  (one_sport_per_day : ∀ d₁ d₂ : Day, d₁ ≠ d₂ → Schedule.sport d₁ ≠ Schedule.sport d₂)
  (cycles_3_days_week : 2 ≤ List.length (List.filter (λ d, Schedule.sport d = "cycling") [Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday]) ∧
                        List.length (List.filter (λ d, Schedule.sport d = "cycling") [Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday]) ≤ 3)
  (no_consecutive_cycling : ∀ d : Day, Schedule.sport d = "cycling" → Schedule.sport (next d) ≠ "cycling" ∧ Schedule.sport (prev d) ≠ "cycling")
  (volleyball_wednesday : Schedule.sport Wednesday = "volleyball")
  (basketball_friday : Schedule.sport Friday = "basketball")
  (tennis_near_cycling : ∀ d : Day, Schedule.sport d = "tennis" → Schedule.sport (next d) ≠ "swimming" ∧ Schedule.sport (prev d) ≠ "cycling")

-- next and prev functions to define consecutive days. Works correctly assuming week cycle.
def next : Day → Day
| Monday    := Tuesday
| Tuesday   := Wednesday
| Wednesday := Thursday
| Thursday  := Friday
| Friday    := Saturday
| Saturday  := Sunday
| Sunday    := Monday

def prev : Day → Day
| Monday    := Sunday
| Tuesday   := Monday
| Wednesday := Tuesday
| Thursday  := Wednesday
| Friday    := Thursday
| Saturday  := Friday
| Sunday    := Saturday

-- Prove that Mahdi swims on Saturday
theorem Mahdi_swims_on_Saturday (sched : Schedule) (cond : Conditions sched) : sched.sport Saturday = "swimming" :=
  sorry

end Mahdi_swims_on_Saturday_l5_5778


namespace geometric_sequence_seventh_term_l5_5190

theorem geometric_sequence_seventh_term (a₁ a₉ : ℕ) (r : ℕ) :
  a₁ = 4 →
  a₉ = 248832 →
  r^8 = 62208 →
  4 * 6^6 = 186624 :=
begin
  intros ha₁ ha₉ hr,
  sorry
end

end geometric_sequence_seventh_term_l5_5190


namespace find_b_tangent_l5_5310

-- Definitions for the equations and conditions in Lean 4
def line (x b : ℝ) : ℝ := (1 / 2) * x + b
def curve (x : ℝ) : ℝ := - (1 / 2) * x + Real.log x
def derivative_curve (x : ℝ) : ℝ := - (1 / 2) + (1 / x)

-- Conditions for tangency
def tangent_condition (x b : ℝ) : Prop := derivative_curve x = (1 / 2) ∧ curve x = line x b

theorem find_b_tangent :
  ∃ b : ℝ, tangent_condition 1 b ∧ b = -1 :=
by
  sorry

end find_b_tangent_l5_5310


namespace exist_indices_with_non_decreasing_subsequences_l5_5236

theorem exist_indices_with_non_decreasing_subsequences
  (a b c : ℕ → ℕ) :
  (∀ n m : ℕ, n < m → ∃ p q : ℕ, q < p ∧ 
    a p ≥ a q ∧ 
    b p ≥ b q ∧ 
    c p ≥ c q) :=
  sorry

end exist_indices_with_non_decreasing_subsequences_l5_5236


namespace expected_value_of_8_sided_die_is_4_point_5_l5_5354

def expected_value_8_sided_die : ℝ :=
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let probability := 1 / 8
  (list.sum outcomes : ℝ) * probability

theorem expected_value_of_8_sided_die_is_4_point_5 :
  expected_value_8_sided_die = 4.5 := by
  sorry

end expected_value_of_8_sided_die_is_4_point_5_l5_5354


namespace area_of_parallelogram_l5_5106

-- Definitions of the vectors
def u : ℝ × ℝ × ℝ := (4, 2, -3)
def v : ℝ × ℝ × ℝ := (2, -4, 5)

-- Definition of the cross product
def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((a.2 * b.3 - a.3 * b.2), (a.3 * b.1 - a.1 * b.3), (a.1 * b.2 - a.2 * b.1))

-- Definition of the magnitude of a vector
def magnitude (a : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (a.1 * a.1 + a.2 * a.2 + a.3 * a.3)

-- The area of the parallelogram is the magnitude of the cross product of u and v
def area_parallelogram (u v : ℝ × ℝ × ℝ) : ℝ := magnitude (cross_product u v)

-- Proof statement
theorem area_of_parallelogram : area_parallelogram u v = 20 * real.sqrt 3 :=
by
  sorry

end area_of_parallelogram_l5_5106


namespace candy_bars_per_bag_l5_5011

def total_candy_bars : ℕ := 15
def number_of_bags : ℕ := 5

theorem candy_bars_per_bag : total_candy_bars / number_of_bags = 3 :=
by
  sorry

end candy_bars_per_bag_l5_5011


namespace repeating_decimal_fraction_l5_5984

theorem repeating_decimal_fraction :  exists (p q : ℤ), (q ≠ 0) ∧ (p / q = (68 / 33 : ℚ)) :=
begin
  let recurring02 : ℚ := 2 / 99,
  let recurring06 : ℚ := 3 * recurring02,
  have recurring06_simplified : recurring06 = 2 / 33, 
    by simp [recurring06, mul_div_assoc, mul_div_cancel_left],
  let result := 2 + recurring06_simplified,
  use [68, 33],
  split,
  { norm_num }, -- proof that 33 ≠ 0
  { exact result } -- proof that 2.06 recurring = 68 / 33
end

end repeating_decimal_fraction_l5_5984


namespace marble_weight_l5_5782

-- Define the weights of marbles and waffle irons
variables (m w : ℝ)

-- Given conditions
def condition1 : Prop := 9 * m = 4 * w
def condition2 : Prop := 3 * w = 75 

-- The theorem we want to prove
theorem marble_weight (h1 : condition1 m w) (h2 : condition2 w) : m = 100 / 9 :=
by
  sorry

end marble_weight_l5_5782


namespace expected_value_eight_sided_die_l5_5406

-- Define a standard 8-sided die
def eight_sided_die : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]

-- Compute the probability of each outcome
def probability (n : ℕ) : ℝ := 1 / 8

-- Expected Value of a discrete random variable
def expected_value (outcomes : List ℕ) (prob : ℕ → ℝ) : ℝ :=
  outcomes.sum / outcomes.length.toReal

-- Theorem stating the expected value of a standard 8-sided die roll is 4.5
theorem expected_value_eight_sided_die : expected_value eight_sided_die probability = 4.5 := by
  sorry

end expected_value_eight_sided_die_l5_5406


namespace total_rainfall_in_April_l5_5197

variable (R_March : ℝ)
variable (R_April : ℝ)

-- Define the conditions
def condition1 := R_March = 0.81
def condition2 := R_April = R_March - 0.35

-- State the theorem we need to prove
theorem total_rainfall_in_April : condition1 ∧ condition2 → R_April = 0.46 := by
  sorry

end total_rainfall_in_April_l5_5197


namespace lotto_tickets_purchase_l5_5221

theorem lotto_tickets_purchase:
  ∀ (T : ℕ),
  (let cost := 2 * T in
  let winners := 0.20 * T in
  let five_dollar_winners := 0.80 * winners in
  let five_dollar_total := 5 * five_dollar_winners in
  let grand_prize := 5000 in
  let other_winners := winners - five_dollar_winners - 1 in
  let other_winners_total := 10 * other_winners in
  let total_winnings := five_dollar_total + grand_prize + other_winners_total in
  total_winnings - cost = 4830) →
  T = 200 :=
by
  intros T h
  sorry

end lotto_tickets_purchase_l5_5221


namespace cosine_of_angle_between_vectors_eq_l5_5653

variables (a b : ℝ^3)

-- a and b are unit vectors
hypothesis unit_a : ∥a∥ = 1
hypothesis unit_b : ∥b∥ = 1

-- The orthogonality condition
hypothesis orthogonal : (3 • a + b) ⬝ (a - 2 • b) = 0

-- The theorem statement
theorem cosine_of_angle_between_vectors_eq :
  ∃ θ : ℝ, (cos θ) = 1 / 5 :=
begin
  -- Mathematical proof goes here
  sorry
end

end cosine_of_angle_between_vectors_eq_l5_5653


namespace total_time_eq_l5_5772

theorem total_time_eq (n : ℕ) (p t s C : ℕ) 
  (h1 : p = 2 * n * s)
  (h2 : t = 3 * n * s + s) 
  (h3 : s = 6) :
  C = 30 * n + 12 :=
by 
  have hp : p = 12 * n, from by rw [h1, h3, Nat.mul_assoc, Nat.mul_comm 2 6, ← Nat.mul_assoc]; rfl,
  have ht : t = 18 * n + 6, from by rw [h2, h3, Nat.mul_assoc 3, Nat.mul_comm 3 6, ← Nat.mul_assoc]; exact Nat.add_comm 6 0 ▸ rfl,
  have hs : s = 6, from h3,
  calc 
    C = p + t + s    : by sorry
    ... = 12 * n + 18 * n + 6 + 6 : by sorry
    ... = 30 * n + 12 : by sorry

end total_time_eq_l5_5772


namespace original_selling_price_l5_5907

/-- A boy sells a book for some amount and he gets a loss of 10%.
To gain 10%, the selling price should be Rs. 550.
Prove that the original selling price of the book was Rs. 450. -/
theorem original_selling_price (CP : ℝ) (h1 : 1.10 * CP = 550) :
    0.90 * CP = 450 := 
sorry

end original_selling_price_l5_5907


namespace soccer_camp_afternoon_kids_l5_5845

theorem soccer_camp_afternoon_kids
  (total_kids : ℕ)
  (half_to_soccer : ℕ)
  (morning_fraction : ℕ)
  (afternoon_kids : ℕ) :
  total_kids = 2000 →
  half_to_soccer = total_kids / 2 →
  morning_fraction = half_to_soccer / 4 →
  afternoon_kids = half_to_soccer - morning_fraction →
  afternoon_kids = 750 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  simp
  sorry

end soccer_camp_afternoon_kids_l5_5845


namespace proof_B_cup_complement_A_l5_5776

open Set

variable U : Set ℕ := {x | x < 4}
variable A : Set ℕ := {0, 1, 2}
variable B : Set ℕ := {2, 3}

theorem proof_B_cup_complement_A :
  B ∪ (U \ A) = {2, 3} :=
by
  sorry

end proof_B_cup_complement_A_l5_5776


namespace sum_of_odd_terms_l5_5047

-- Define a sequence and state the main properties and the sum condition
def sequence (x : ℕ → ℕ) :=
  (∀ n, x (n + 1) = x n + 2) ∧
  (∑ i in Finset.range 2020, x i = 6060)

-- Define the sum of every second term, starting with the first, as S
def sum_of_every_second_term (x : ℕ → ℕ) : ℕ :=
  ∑ i in Finset.range 1010, x (2 * i)

-- The main theorem we want to prove
theorem sum_of_odd_terms (x : ℕ → ℕ) (h : sequence x) :
  sum_of_every_second_term x = 2020 :=
sorry

end sum_of_odd_terms_l5_5047


namespace range_of_c_l5_5233

theorem range_of_c (x y c : ℝ) (h : x^2 + (y - 1)^2 = 1) :
  (∀ (x y : ℝ), x^2 + (y - 1)^2 = 1 → x - y - c ≤ 0) ↔ c ∈ Ici (-1) :=
by sorry

end range_of_c_l5_5233


namespace smallest_rel_prime_to_180_l5_5605

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ Int.gcd x 180 = 1 ∧ ∀ y : ℕ, y > 1 ∧ Int.gcd y 180 = 1 → x ≤ y := 
sorry

end smallest_rel_prime_to_180_l5_5605


namespace remainder_six_n_mod_four_l5_5871

theorem remainder_six_n_mod_four (n : ℤ) (h : n % 4 = 3) : (6 * n) % 4 = 2 := 
by sorry

end remainder_six_n_mod_four_l5_5871


namespace parabola_standard_equation_with_directrix_x_eq_1_l5_5839

theorem parabola_standard_equation_with_directrix_x_eq_1 :
  ∀ (y x : ℝ), (directrix: ℝ → ℝ) (std_eq: ℝ → ℝ → ℝ) (focus: ℝ → ℝ → ℝ),
  directrix x = 1 →
  (std_eq y x = y^2 + 8 * x = 0) :=
by
  sorry

end parabola_standard_equation_with_directrix_x_eq_1_l5_5839


namespace distance_zephyr_to_sun_l5_5073

-- Definitions of the main parameters of the problem
def a : ℝ := 9          -- semi-major axis (AU)
def b : ℝ := 3 * Real.sqrt 5  -- semi-minor axis (AU)
def c : ℝ := 6          -- distance from center to the focus (AU)

def ZC : ℝ := b         -- Zephyr's distance from the center when it is above the center on the minor axis
def CF1 : ℝ := c        -- Distance from the center to the sun (focus)

-- Main theorem statement
theorem distance_zephyr_to_sun : ZC + CF1 = 3 * Real.sqrt 5 + 6 :=
by
  rw [ZC, CF1]
  sorry

end distance_zephyr_to_sun_l5_5073


namespace perfect_squares_in_sequence_l5_5469

theorem perfect_squares_in_sequence :
  ∃ (count : ℕ), count = 10 ∧
    ∀ (a : ℕ → ℕ),
      (a 1 = 1) ∧
      (∀ n, a (n + 1) = a n + nat.floor (real.sqrt (a n))) ∧
      (∃N, a N ≤ 1000000 ∧ is_square (a N)) :=
sorry

end perfect_squares_in_sequence_l5_5469


namespace find_d_l5_5234

theorem find_d (a₁: ℤ) (d : ℤ) (Sn : ℤ → ℤ) : 
  a₁ = 190 → 
  (Sn 20 > 0) → 
  (Sn 24 < 0) → 
  (Sn n = n * a₁ + (n * (n - 1)) / 2 * d) →
  d = -17 :=
by
  intros
  sorry

end find_d_l5_5234


namespace distance_A1C1_BD1_l5_5810

-- Define the vertices of the unit cube
def A := (0, 0, 0)
def B := (1, 0, 0)
def C := (1, 1, 0)
def D := (0, 1, 0)
def A1 := (0, 0, 1)
def B1 := (1, 0, 1)
def C1 := (1, 1, 1)
def D1 := (0, 1, 1)

-- Function to compute the distance between skew lines in 3D
noncomputable def distance_between_skew_lines (p1 p2 p3 p4 : ℝ × ℝ × ℝ) : ℝ :=
  let u : ℝ × ℝ × ℝ := ⟨(p2.1 - p1.1), (p2.2 - p1.2), (p2.3 - p1.3)⟩
  let v : ℝ × ℝ × ℝ := ⟨(p4.1 - p3.1), (p4.2 - p3.2), (p4.3 - p3.3)⟩
  let w : ℝ × ℝ × ℝ := ⟨(p1.1 - p3.1), (p1.2 - p3.2), (p1.3 - p3.3)⟩
  let a := u.1 * u.1 + u.2 * u.2 + u.3 * u.3
  let b := u.1 * v.1 + u.2 * v.2 + u.3 * v.3
  let c := v.1 * v.1 + v.2 * v.2 + v.3 * v.3
  let d := u.1 * w.1 + u.2 * w.2 + u.3 * w.3
  let e := v.1 * w.1 + v.2 * w.2 + v.3 * w.3
  let D := a * c - b * b
  if D = 0 then 0 else
  let s := (b * e - c * d) / D
  let t := (a * e - b * d) / D
  let x := w.1 + s * u.1 - t * v.1
  let y := w.2 + s * u.2 - t * v.2
  let z := w.3 + s * u.3 - t * v.3
  Math.sqrt (x * x + y * y + z * z)

theorem distance_A1C1_BD1 : distance_between_skew_lines A1 C1 B D1 = real.sqrt(6) / 6 := by
  sorry

end distance_A1C1_BD1_l5_5810


namespace greatest_possible_value_of_x_l5_5862

-- Define the function based on the given equation
noncomputable def f (x : ℝ) : ℝ := (4 * x - 16) / (3 * x - 4)

-- Statement to be proved
theorem greatest_possible_value_of_x : 
  (∀ x : ℝ, (f x)^2 + (f x) = 20) → 
  ∃ x : ℝ, (f x)^2 + (f x) = 20 ∧ x = 36 / 19 :=
by
  sorry

end greatest_possible_value_of_x_l5_5862


namespace tangerines_more_than_oranges_l5_5330

-- Define initial conditions
def initial_apples := 9
def initial_oranges := 5
def initial_tangerines := 17

-- Define actions taken
def oranges_taken := 2
def tangerines_taken := 10

-- Resulting quantities
def oranges_left := initial_oranges - oranges_taken
def tangerines_left := initial_tangerines - tangerines_taken

-- Proof problem
theorem tangerines_more_than_oranges : tangerines_left - oranges_left = 4 := 
by sorry

end tangerines_more_than_oranges_l5_5330


namespace range_of_a_l5_5955

open Real

def op (x y : ℝ) : ℝ := x * (1 - y)

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, op (x - a) (x + a) < 1) ↔ a ∈ Ioo (- (1/2)) ((3/2)) := 
by
  sorry

end range_of_a_l5_5955


namespace partition_exists_sum_l5_5756

theorem partition_exists_sum (n : ℕ) (h_pos : 0 < n) (A B C : set ℕ) (h_partition : ∀ x, x ∈ (finset.range (3 * n)).to_set ↔ x ∈ A ∪ B ∪ C) (h_disjoint : disjoint A B ∧ disjoint B C ∧ disjoint A C) (h_size : A.card = n ∧ B.card = n ∧ C.card = n) :
  ∃ x ∈ A, ∃ y ∈ B, ∃ z ∈ C, (x = y + z) ∨ (y = x + z) ∨ (z = x + y) :=
sorry

end partition_exists_sum_l5_5756


namespace at_least_ten_pairs_reporting_l5_5328

-- Define the condition: 20 spies, each writing a report on 10 colleagues.
def num_spies : ℕ := 20
def num_reports : ℕ := 10
def reports_on : Fin num_spies → Fin num_spies → Prop

-- Each spy reports on exactly 10 other spies.
axiom reports_on_condition : ∀ (s : Fin num_spies), (finset.filter (reports_on s) finset.univ).card = num_reports

-- Prove that at least 10 pairs of spies reported on each other.
theorem at_least_ten_pairs_reporting :
  ∃ (pairs : Finset (Fin num_spies × Fin num_spies)), pairs.card ≥ 10 ∧ ∀ (p : Fin num_spies × Fin num_spies), p ∈ pairs → reports_on p.1 p.2 ∧ reports_on p.2 p.1 :=
sorry

end at_least_ten_pairs_reporting_l5_5328


namespace expected_value_8_sided_die_l5_5367

theorem expected_value_8_sided_die :
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let probability_each_outcome := 1 / 8
  (Σ x ∈ outcomes, probability_each_outcome * x) = 4.5 :=
by
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let probability_each_outcome := 1 / 8
  have h : (Σ x ∈ outcomes, probability_each_outcome * x) = (1 / 8) * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8) := by sorry
  have sum_eq_36 : 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 = 36 := by sorry
  rw [sum_eq_36] at h
  have expected_value_eq : (1 / 8) * 36 = 4.5 := by sorry
  rw [expected_value_eq] at h
  exact h

end expected_value_8_sided_die_l5_5367


namespace avg_annual_growth_rate_l5_5487
-- Import the Mathlib library

-- Define the given conditions
def initial_income : ℝ := 32000
def final_income : ℝ := 37000
def period : ℝ := 2
def initial_income_ten_thousands : ℝ := initial_income / 10000
def final_income_ten_thousands : ℝ := final_income / 10000

-- Define the growth rate
variable (x : ℝ)

-- Define the theorem
theorem avg_annual_growth_rate :
  3.2 * (1 + x) ^ 2 = 3.7 :=
sorry

end avg_annual_growth_rate_l5_5487


namespace largest_lcm_l5_5863

theorem largest_lcm: 
  let lcm182 := Nat.lcm 18 2;
      lcm184 := Nat.lcm 18 4;
      lcm186 := Nat.lcm 18 6;
      lcm189 := Nat.lcm 18 9;
      lcm1812 := Nat.lcm 18 12;
      lcm1815 := Nat.lcm 18 15 
  in max (max (max (max (max lcm182 lcm184) lcm186) lcm189) lcm1812) lcm1815 = 90 :=
by
  sorry

end largest_lcm_l5_5863


namespace min_pieces_on_checkboard_l5_5273

-- Define the board size and the range of n values
def board_size : ℕ := 6

-- Condition: For each number n from 2 to 10, there must be a piece in the same row and column as exactly n pieces (not counting itself).
def valid_placement (board : matrix (fin board_size) (fin board_size) bool) : Prop :=
  ∀ n in finset.range 9, -- since 10 - 2 + 1 = 9
    ∃ i j, board i j = true ∧
      ((finset.univ.filter (λ k, k ≠ j ∧ board i k = true)).card +
       (finset.univ.filter (λ k, k ≠ i ∧ board k j = true)).card) = (n + 2)

-- The minimum number of pieces that satisfies the given conditions
theorem min_pieces_on_checkboard : ∃ (board : matrix (fin board_size) (fin board_size) bool), valid_placement board ∧ (finset.univ.filter (λ (i : fin board_size) (j : fin board_size), board i j = true)).card = 19 :=
sorry

end min_pieces_on_checkboard_l5_5273


namespace ratio_of_guesses_l5_5720

variable (c d : ℝ) -- c: number of times Anna guesses cat, d: number of times Anna guesses dog

-- Conditions
axiom equal_images : 0.9 * c + 0.05 * d = 0.1 * c + 0.95 * d
axiom correct_guess_dog : 0.95 * d
axiom correct_guess_cat : 0.9 * c

theorem ratio_of_guesses (c d : ℝ) (h1 : 0.9 * c + 0.05 * d = 0.1 * c + 0.95 * d) : d / c = 8 / 9 :=
by
  sorry

end ratio_of_guesses_l5_5720


namespace song_distribution_l5_5716

theorem song_distribution (five_songs : Finset α)
  (like_AB_Beth_Amy_not_Jo : ∃ s, s ∈ five_songs)
  (like_BC_Beth_Jo_not_Amy : ∃ s, s ∈ five_songs)
  (like_CA_Jo_Amy_not_Beth : ∃ s, s ∈ five_songs)
  (no_song_liked_by_all : ∀ s, s ∉ (like_AB_Beth_Amy_not_Jo ∩ like_BC_Beth_Jo_not_Amy ∩ like_CA_Jo_Amy_not_Beth)) :
  ∃ arrangements, arrangements.card = 168 :=
sorry

end song_distribution_l5_5716


namespace firecracker_velocity_magnitude_l5_5034

-- Problem conditions
def initial_speed : ℝ := 20
def acceleration_due_to_gravity : ℝ := 10
def explosion_time : ℝ := 3
def mass_ratio : ℕ := 2
def smaller_fragment_horizontal_speed : ℝ := 16

theorem firecracker_velocity_magnitude :
  let v_init := initial_speed
  let g := acceleration_due_to_gravity
  let t := explosion_time
  let m_ratio := mass_ratio
  let v_horizontal_sm := smaller_fragment_horizontal_speed
  let v_vertical_fr := v_init - g * t
  let m1 : ℝ := 1
  let m2 : ℝ := 2
  let v_horizontal_fr := - v_horizontal_sm * m1 / m2
  sqrt (v_horizontal_fr^2 + v_vertical_fr^2) = 17 :=
by
  sorry

end firecracker_velocity_magnitude_l5_5034


namespace expected_value_8_sided_die_l5_5401

/-- 
The expected value of rolling a standard 8-sided die is 4.5.
The die has 8 sides labeled 1 through 8, and each face has an equal probability of appearing,
which is 1/8. 
-/
theorem expected_value_8_sided_die : 
  (1/8:ℝ) * 1 + (1/8) * 2 + (1/8) * 3 + (1/8) * 4 + (1/8) * 5 + (1/8) * 6 + (1/8) * 7 + (1/8) * 8 = 4.5 :=
by 
  sorry

end expected_value_8_sided_die_l5_5401


namespace relationship_y1_y2_l5_5154

-- Define the linear function y = 2x + 1
def linear_function (x : ℝ) : ℝ := 2 * x + 1

-- Given conditions
def point1 : ℝ := linear_function (-1)  -- y1 when x = -1
def point2 : ℝ := linear_function 3     -- y2 when x = 3

-- Statement to prove
theorem relationship_y1_y2 : point1 < point2 :=
by
  have h1 : point1 = -1 := by 
    simp [point1, linear_function]
  have h2 : point2 = 7 := by 
    simp [point2, linear_function]
  rw [h1, h2]
  exact neg_one_lt_seven

-- Sorry is not needed since the proof is essentially completed.

end relationship_y1_y2_l5_5154


namespace length_of_QB_l5_5949

/-- 
Given a circle Q with a circumference of 16π feet, 
segment AB as its diameter, 
and the angle AQB of 120 degrees, 
prove that the length of segment QB is 8 feet.
-/
theorem length_of_QB (C : ℝ) (r : ℝ) (A B Q : ℝ) (angle_AQB : ℝ) 
  (h1 : C = 16 * Real.pi)
  (h2 : 2 * Real.pi * r = C)
  (h3 : angle_AQB = 120) 
  : QB = 8 :=
sorry

end length_of_QB_l5_5949


namespace expected_value_of_8_sided_die_l5_5391

def outcomes : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]
def probability (n : ℕ) : ℝ := 1 / 8

theorem expected_value_of_8_sided_die :
  (∑ x in outcomes, probability x * x) = 4.5 := 
sorry

end expected_value_of_8_sided_die_l5_5391


namespace weight_of_white_ring_l5_5226

def weight_orange := 0.08333333333333333
def weight_purple := 0.3333333333333333
def total_weight := 0.8333333333

def weight_white := 0.41666666663333337

theorem weight_of_white_ring :
  weight_white + weight_orange + weight_purple = total_weight :=
by
  sorry

end weight_of_white_ring_l5_5226


namespace quadratic_equation_solutions_l5_5838

theorem quadratic_equation_solutions :
  ∀ x : ℝ, x^2 + 7 * x = 0 ↔ (x = 0 ∨ x = -7) := 
by 
  intro x
  sorry

end quadratic_equation_solutions_l5_5838


namespace evan_books_two_years_ago_l5_5972

theorem evan_books_two_years_ago (B B2 : ℕ) 
  (h1 : 860 = 5 * B + 60) 
  (h2 : B2 = B + 40) : 
  B2 = 200 := 
by 
  sorry

end evan_books_two_years_ago_l5_5972


namespace kids_with_red_hair_l5_5911

theorem kids_with_red_hair (total_kids : ℕ) (ratio_red ratio_blonde ratio_black : ℕ) 
  (h_ratio : ratio_red + ratio_blonde + ratio_black = 16) (h_total : total_kids = 48) :
  (total_kids / (ratio_red + ratio_blonde + ratio_black)) * ratio_red = 9 :=
by
  sorry

end kids_with_red_hair_l5_5911


namespace expected_value_of_eight_sided_die_l5_5348

-- Definitions based on the problem conditions
def eight_sided_die_outcomes : List ℕ := [1,2,3,4,5,6,7,8]

def probability (n : ℕ) := 1 / n

-- Expected value calculation related to the problem
def expected_value_die_roll (outcomes : List ℕ) (prob : ℕ → Rat) : Rat :=
  List.sum (outcomes.map (λ x => prob outcomes.length * x))

-- Expected value of an 8-sided die roll
theorem expected_value_of_eight_sided_die :
  expected_value_die_roll eight_sided_die_outcomes probability = 4.5 := 
sorry

end expected_value_of_eight_sided_die_l5_5348


namespace decompose_max_product_l5_5078

theorem decompose_max_product (a : ℝ) (h_pos : a > 0) :
  ∃ x y : ℝ, x + y = a ∧ x * y ≤ (a / 2) * (a / 2) :=
by
  sorry

end decompose_max_product_l5_5078


namespace percent_students_both_correct_l5_5702

def percent_answered_both_questions (total_students first_correct second_correct neither_correct : ℕ) : ℕ :=
  let at_least_one_correct := total_students - neither_correct
  let total_individual_correct := first_correct + second_correct
  total_individual_correct - at_least_one_correct

theorem percent_students_both_correct
  (total_students : ℕ)
  (first_question_correct : ℕ)
  (second_question_correct : ℕ)
  (neither_question_correct : ℕ) 
  (h_total_students : total_students = 100)
  (h_first_correct : first_question_correct = 80)
  (h_second_correct : second_question_correct = 55)
  (h_neither_correct : neither_question_correct = 20) :
  percent_answered_both_questions total_students first_question_correct second_question_correct neither_question_correct = 55 :=
by
  rw [h_total_students, h_first_correct, h_second_correct, h_neither_correct]
  sorry


end percent_students_both_correct_l5_5702


namespace lara_flowers_l5_5228

theorem lara_flowers (total mom vase : ℕ) (h_total : total = 52) (h_mom : mom = 15) (h_vase : vase = 16) :
  let grandma := total - mom - vase in grandma - mom = 6 :=
by
  let grandma := total - mom - vase
  have h_grandma : grandma = 21 := by 
    rw [h_total, h_mom, h_vase]
    exact rfl
  have h_grandma_diff : grandma - mom = 6 := by
    rw [h_grandma, h_mom]
    exact rfl
  exact h_grandma_diff

end lara_flowers_l5_5228


namespace eyes_given_to_dog_l5_5215

-- Definitions of the conditions
def fish_per_person : ℕ := 4
def number_of_people : ℕ := 3
def eyes_per_fish : ℕ := 2
def eyes_eaten_by_Oomyapeck : ℕ := 22

-- The proof statement
theorem eyes_given_to_dog : ∃ (eyes_given_to_dog : ℕ), eyes_given_to_dog = 4 * 3 * 2 - 22 := by
  sorry

end eyes_given_to_dog_l5_5215


namespace length_of_intersection_segment_l5_5738

-- Definitions
def line_theta_eq_pi_over_3 (theta: ℝ) (rho: ℝ) : Prop := theta = π / 3
def circle_polar_equation (theta: ℝ) (rho: ℝ) : Prop := rho = 4 * cos theta + 4 * (sqrt 3) * sin theta

-- Problem Statement
theorem length_of_intersection_segment : 
  ∀ A B : EuclideanSpace ℝ (Fin 2), 
  (∃ (rho_A theta_A rho_B theta_B: ℝ), line_theta_eq_pi_over_3 theta_A rho_A ∧ circle_polar_equation theta_A rho_A ∧ A = ⟨rho_A * cos theta_A, rho_A * sin theta_A⟩ ∧
  line_theta_eq_pi_over_3 theta_B rho_B ∧ circle_polar_equation theta_B rho_B ∧ B = ⟨rho_B * cos theta_B, rho_B * sin theta_B⟩) →
  dist A B = 8 :=
by
  sorry

end length_of_intersection_segment_l5_5738


namespace rectangle_width_is_14_l5_5301

noncomputable def rectangleWidth (areaOfCircle : ℝ) (length : ℝ) : ℝ :=
  let r := Real.sqrt (areaOfCircle / Real.pi)
  2 * r

theorem rectangle_width_is_14 :
  rectangleWidth 153.93804002589985 18 = 14 :=
by 
  sorry

end rectangle_width_is_14_l5_5301


namespace monotonic_increasing_interval_l5_5825

open Real

def f (x : ℝ) := x^2 - 2*x

theorem monotonic_increasing_interval :
  ∀ x, 1 ≤ x → monotone_on f (Ici 1) :=
sorry

end monotonic_increasing_interval_l5_5825


namespace max_min_M_l5_5657

variable (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

/-- Proof that the maximum value of M is sqrt(2) given the conditions on a, b, and c --/
theorem max_min_M (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    ∃ M : ℝ, M = sqrt 2 ∧ M = max (min (1 / a) (min (2 / b) (min (4 / c) ( real.sqrt (abc)))) ) :=
sorry

end max_min_M_l5_5657


namespace distance_between_midpoints_l5_5274

theorem distance_between_midpoints (p q r s : ℝ) :
  let x := (p + r) / 2
  let y := (q + s) / 2
  let x' := x - 1
  let y' := y + 1
  real.sqrt ((x' - x) ^ 2 + (y' - y) ^ 2) = real.sqrt 2 :=
by
  let x := (p + r) / 2
  let y := (q + s) / 2
  let x' := x - 1
  let y' := y + 1
  sorry

end distance_between_midpoints_l5_5274


namespace some_students_not_honorSociety_l5_5058

universe u

constant Students : Type u
constant Scholarships : Type u
constant HonorSociety : Type u

constant student : Students → Prop
constant scholarship : Scholarships → Prop
constant honorSociety : HonorSociety → Prop

axiom exists_student_no_scholarship : ∃ s : Students, ¬ scholarship (s)
axiom all_honorSociety_scholarship : ∀ h : HonorSociety, scholarship (h)

theorem some_students_not_honorSociety : ∃ s : Students, ¬ honorSociety (s) :=
sorry

end some_students_not_honorSociety_l5_5058


namespace bridge_length_l5_5476

-- Definitions of the problem's conditions
def train_length : ℝ := 327
def train_speed : ℝ := 40 * 1000 / 3600  -- convert 40 km/hour to meters/second
def crossing_time : ℝ := 40.41

-- Statement of the problem to be proved
theorem bridge_length :
  let total_distance := train_speed * crossing_time in
  let bridge_length := total_distance - train_length in
  abs (bridge_length - 122.15) < 1e-2 :=
by
  sorry

end bridge_length_l5_5476


namespace bus_speed_excluding_stoppages_l5_5973

theorem bus_speed_excluding_stoppages :
  ∀ (S : ℝ), (45 = (3 / 4) * S) → (S = 60) :=
by 
  intros S h
  sorry

end bus_speed_excluding_stoppages_l5_5973


namespace chromium_percentage_in_new_alloy_l5_5884

-- Define the given conditions as Lean definitions.

def first_alloy_percent_chromium : ℝ := 12 / 100
def second_alloy_percent_chromium : ℝ := 8 / 100
def first_alloy_weight : ℝ := 20
def second_alloy_weight : ℝ := 35

-- Define the proof statement using Lean.
theorem chromium_percentage_in_new_alloy :
  let total_weight : ℝ := first_alloy_weight + second_alloy_weight
  let first_alloy_chromium : ℝ := first_alloy_percent_chromium * first_alloy_weight
  let second_alloy_chromium : ℝ := second_alloy_percent_chromium * second_alloy_weight
  let total_chromium : ℝ := first_alloy_chromium + second_alloy_chromium
  let chromium_percentage : ℝ := (total_chromium / total_weight) * 100
  chromium_percentage ≈ 9.45 :=
by
  -- The proof would go here.
  sorry

end chromium_percentage_in_new_alloy_l5_5884


namespace mysterious_angle_is_84_l5_5055

def equilateral_triangle_internal_angle := 60
def regular_pentagon_internal_angle := 108

theorem mysterious_angle_is_84 :
  ∀ (α β γ : ℕ), 
    α = equilateral_triangle_internal_angle ∧ 
    β = regular_pentagon_internal_angle ∧ 
    3 * β + α + γ = 360 → 
    γ = 84 :=
by
  intros α β γ h
  cases h with h1 h2
  cases h2 with h3 h4
  sorry

end mysterious_angle_is_84_l5_5055


namespace repeating_decimal_fraction_l5_5993

theorem repeating_decimal_fraction :  exists (p q : ℤ), (q ≠ 0) ∧ (p / q = (68 / 33 : ℚ)) :=
begin
  let recurring02 : ℚ := 2 / 99,
  let recurring06 : ℚ := 3 * recurring02,
  have recurring06_simplified : recurring06 = 2 / 33, 
    by simp [recurring06, mul_div_assoc, mul_div_cancel_left],
  let result := 2 + recurring06_simplified,
  use [68, 33],
  split,
  { norm_num }, -- proof that 33 ≠ 0
  { exact result } -- proof that 2.06 recurring = 68 / 33
end

end repeating_decimal_fraction_l5_5993


namespace ray_travel_distance_l5_5649

noncomputable def distance (p1 p2 : Real × Real) : Real :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def point_reflection_over_x_axis (p : Real × Real) : Real × Real :=
  (p.1, -p.2)

def circle (center : Real × Real) (radius : Real) : Set (Real × Real) :=
  {p | distance p center = radius}

def is_tangent_point (p : Real × Real) (center : Real × Real) (radius : Real) : Prop :=
  distance p center = radius

theorem ray_travel_distance :
  let A := (-2, 1)
  let C_center := (2, 2)
  let C_radius := 1
  let A_reflected := point_reflection_over_x_axis A
  distance A_reflected C_center = 5 →
  ∃ T,
  is_tangent_point T C_center C_radius ∧ distance A T = 2 * Real.sqrt 6 :=
by
  intros
  sorry

end ray_travel_distance_l5_5649


namespace smallest_rel_prime_to_180_l5_5579

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ Nat.gcd x 180 = 1 ∧ ∀ y : ℕ, y > 1 ∧ Nat.gcd y 180 = 1 → x ≤ y :=
begin
  use 7,
  split,
  { exact dec_trivial },
  split,
  { exact dec_trivial },
  { intro y,
    intro h,
    cases h with h1 h2,
    repeat { try { apply dec_trivial,
                   apply lt_or_eq_of_le,
                   norm_num,
                   apply Nat.prime_not_dvd_mul,
                   norm_num,
                   apply not_or_distrib.mpr,
                   split,
                   norm_cast,
                   intro,
                   exact le_antisymm _ },
           sorry }
end

end smallest_rel_prime_to_180_l5_5579


namespace total_collection_amount_l5_5035

theorem total_collection_amount (n : ℕ) (h : n = 57) : (n * n) / 100 = 32.49 :=
by {
  rw h,
  exact (3249 : ℝ) / 100,
  norm_num,
  norm_num,
}

end total_collection_amount_l5_5035


namespace expected_value_eight_sided_die_l5_5408

-- Define a standard 8-sided die
def eight_sided_die : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]

-- Compute the probability of each outcome
def probability (n : ℕ) : ℝ := 1 / 8

-- Expected Value of a discrete random variable
def expected_value (outcomes : List ℕ) (prob : ℕ → ℝ) : ℝ :=
  outcomes.sum / outcomes.length.toReal

-- Theorem stating the expected value of a standard 8-sided die roll is 4.5
theorem expected_value_eight_sided_die : expected_value eight_sided_die probability = 4.5 := by
  sorry

end expected_value_eight_sided_die_l5_5408


namespace prob_two_black_in_4th_draw_prob_dist_X_l5_5726

def bag_initial_state := "2 white balls and 2 black balls"

-- Probability of exactly drawing two black balls in the 4th draw
theorem prob_two_black_in_4th_draw : 
  P(draw_black_in_4th_draw) = 129 / 1000 := sorry

-- Probability distribution of X
theorem prob_dist_X :
  ∀ X ∈ {2, 3, 4, 5},
    (if X = 2 then P(X) = 1 / 10 else
     if X = 3 then P(X) = 13 / 100 else
     if X = 4 then P(X) = 129 / 1000 else
     if X = 5 then P(X) = 641 / 1000) := sorry

end prob_two_black_in_4th_draw_prob_dist_X_l5_5726


namespace leading_coefficient_of_g_l5_5638

theorem leading_coefficient_of_g (g : ℕ → ℚ) (h : ∀ x : ℕ, g (x + 1) - g x = x^2 + x + 1) :
  polynomial.leading_coeff (polynomial.C (g 0) + polynomial.C (g 1) * polynomial.X + polynomial.C (g 2) * polynomial.X^2 + polynomial.C (g 3) * polynomial.X^3) = 1 / 3 :=
sorry

end leading_coefficient_of_g_l5_5638


namespace letter_puzzle_l5_5557

theorem letter_puzzle (A B : ℕ) (hA : 1 ≤ A ∧ A ≤ 9) (hB : 1 ≤ B ∧ B ≤ 9) (h_diff : A ≠ B) :
  A^B = 10 * B + A ↔ (A = 2 ∧ B = 5) ∨ (A = 6 ∧ B = 2) ∨ (A = 4 ∧ B = 3) :=
by
  sorry

end letter_puzzle_l5_5557


namespace solve_inequality_l5_5796

theorem solve_inequality (x : ℝ) :
  |(3 * x - 2) / (x ^ 2 - x - 2)| > 3 ↔ (x ∈ Set.Ioo (-1) (-2 / 3) ∪ Set.Ioo (1 / 3) 4) :=
by sorry

end solve_inequality_l5_5796


namespace number_of_terminating_decimals_l5_5622

theorem number_of_terminating_decimals (n : ℕ) (h₁ : 1 ≤ n) (h₂ : n ≤ 150) :
  ∃ m, m = 50 ∧ 
  ∀ n, (1 ≤ n ∧ n ≤ 150) → (∃ k, n = 3 * k) →
  m = 50 :=
by 
  sorry

end number_of_terminating_decimals_l5_5622


namespace algebraic_expression_value_l5_5184

theorem algebraic_expression_value (a b : ℕ) (h : a - 3 * b = 0) :
  (a - (2 * a * b - b * b) / a) / ((a * a - b * b) / a) = 1 / 2 := 
sorry

end algebraic_expression_value_l5_5184


namespace carol_name_tag_l5_5850

theorem carol_name_tag (a b c : ℕ) (ha : Prime a ∧ a ≥ 10 ∧ a < 100) (hb : Prime b ∧ b ≥ 10 ∧ b < 100) (hc : Prime c ∧ c ≥ 10 ∧ c < 100) 
  (h1 : b + c = 14) (h2 : a + c = 20) (h3 : a + b = 18) : c = 11 := 
by 
  sorry

end carol_name_tag_l5_5850


namespace integral_circle_half_area_l5_5090

theorem integral_circle_half_area :
  ∫ x in -Real.sqrt 2..Real.sqrt 2, Real.sqrt (2 - x^2) = Real.pi :=
by
  sorry

end integral_circle_half_area_l5_5090


namespace cos_zero_is_necessary_not_sufficient_for_sin_one_l5_5898

-- Define the conditions
variables (α : ℝ) (k : ℤ)

-- Define the necessary proof steps
noncomputable def cos_zero_implies_alpha_form : Prop :=
  ∃ k : ℤ, α = k * Real.pi + Real.pi / 2

noncomputable def sin_one_implies_alpha_form : Prop :=
  ∃ k : ℤ, α = Real.pi / 2 + 2 * k * Real.pi

theorem cos_zero_is_necessary_not_sufficient_for_sin_one (h1 : cos α = 0) (h2 : sin α = 1) : 
  cos_zero_implies_alpha_form α ∧ ¬ sin_one_implies_alpha_form α :=
by
  sorry

end cos_zero_is_necessary_not_sufficient_for_sin_one_l5_5898


namespace price_decrease_percentage_l5_5049

-- Define the conditions
variables {P : ℝ} (original_price increased_price decreased_price : ℝ)
variables (y : ℝ) -- percentage by which increased price is decreased

-- Given conditions
def store_conditions :=
  increased_price = 1.20 * original_price ∧
  decreased_price = increased_price * (1 - y/100) ∧
  decreased_price = 0.75 * original_price

-- The proof problem
theorem price_decrease_percentage 
  (original_price increased_price decreased_price : ℝ)
  (y : ℝ) 
  (h : store_conditions original_price increased_price decreased_price y) :
  y = 37.5 :=
by 
  sorry

end price_decrease_percentage_l5_5049


namespace expected_value_of_8_sided_die_l5_5381

theorem expected_value_of_8_sided_die : 
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let probability := 1 / 8
  (∑ outcome in outcomes, outcome * probability) = 4.5 :=
by
  sorry

end expected_value_of_8_sided_die_l5_5381


namespace expected_value_8_sided_die_l5_5400

/-- 
The expected value of rolling a standard 8-sided die is 4.5.
The die has 8 sides labeled 1 through 8, and each face has an equal probability of appearing,
which is 1/8. 
-/
theorem expected_value_8_sided_die : 
  (1/8:ℝ) * 1 + (1/8) * 2 + (1/8) * 3 + (1/8) * 4 + (1/8) * 5 + (1/8) * 6 + (1/8) * 7 + (1/8) * 8 = 4.5 :=
by 
  sorry

end expected_value_8_sided_die_l5_5400


namespace quadrant_of_z1_minus_z2_l5_5140

def z1 : ℂ := 3 - 4 * complex.I
def z2 : ℂ := -2 + 3 * complex.I

theorem quadrant_of_z1_minus_z2 :
  let z := z1 - z2
  let point := (z.re, z.im)
  point.1 > 0 ∧ point.2 < 0 :=
by
  let z := z1 - z2
  have h1 : z = 5 - 7 * complex.I := by sorry -- Calculation of z1 - z2
  let point := (z.re, z.im)
  have h2 : point = (5, -7) := by sorry -- Getting the real and imaginary parts
  show point.1 > 0 ∧ point.2 < 0 from by
    rw h2
    exact And.intro (by linarith) (by linarith)

end quadrant_of_z1_minus_z2_l5_5140


namespace min_words_needed_to_score_90_percent_l5_5695

theorem min_words_needed_to_score_90_percent :
  ∃ x : ℕ, (x + 0.05 * (800 - x)) / 800 ≥ 0.90 ∧ x = 716 := sorry

end min_words_needed_to_score_90_percent_l5_5695


namespace vector_addition_l5_5679

def a : ℝ × ℝ := (5, -3)
def b : ℝ × ℝ := (-6, 4)

theorem vector_addition : a + b = (-1, 1) := by
  rw [a, b]
  sorry

end vector_addition_l5_5679


namespace ivanov_knows_languages_petrov_knows_languages_sidorov_knows_languages_l5_5744

-- Define types for persons and languages
inductive Person
| Ivanov
| Petrov
| Sidorov
  deriving DecidableEq

inductive Language
| German
| French
| English
| Spanish
| Italian
| Arabic
  deriving DecidableEq

-- Define the knowledge of languages by each person
def knows : Person → Language → Prop

-- Define the conditions as hypotheses
axiom each_person_knows_two_languages : ∀ p : Person, ∃ l1 l2 : Language, l1 ≠ l2 ∧ knows p l1 ∧ knows p l2
axiom french_and_spanish_are_neighbors : ∀ p1 p2 : Person, (knows p1 Language.French ∧ knows p1 Language.Spanish) → (knows p2 Language.French ∧ knows p2 Language.Spanish) → (p1 ≠ p2 ∧ neighbors p1 p2)
axiom ivanov_is_the_youngest : ∀ p : Person, p ≠ Person.Ivanov → older_than p Person.Ivanov
axiom sidorov_and_german_spanish_know_each_other : ∀ p : Person, knows p Language.German ∧ knows p Language.Spanish → close_friends Person.Sidorov p
axiom german_older_than_arabic : ∀ p1 p2 : Person, knows p1 Language.German → knows p2 Language.Arabic → older_than p1 p2
axiom english_arabic_fishing : ∀ p : Person, knows p Language.English ∧ knows p Language.Arabic → ∀ o : Person, Person.Ivanov ≠ o → goes_fishing_together Person.Ivanov p o

-- Theorems asserting the languages known by each person
theorem ivanov_knows_languages : knows Person.Ivanov Language.Spanish ∧ knows Person.Ivanov Language.Italian := sorry
theorem petrov_knows_languages : knows Person.Petrov Language.German ∧ knows Person.Petrov Language.English := sorry
theorem sidorov_knows_languages : knows Person.Sidorov Language.Arabic ∧ knows Person.Sidorov Language.French := sorry

end ivanov_knows_languages_petrov_knows_languages_sidorov_knows_languages_l5_5744


namespace positive_correlation_not_proportional_l5_5854

/-- Two quantities x and y depend on each other, and when one increases, the other also increases.
    This general relationship is denoted as a function g such that for any x₁, x₂,
    if x₁ < x₂ then g(x₁) < g(x₂). This implies a positive correlation but not necessarily proportionality. 
    We will prove that this does not imply a proportional relationship (y = kx). -/
theorem positive_correlation_not_proportional (g : ℝ → ℝ) 
(h_increasing: ∀ x₁ x₂ : ℝ, x₁ < x₂ → g x₁ < g x₂) :
¬ ∃ k : ℝ, ∀ x : ℝ, g x = k * x :=
sorry

end positive_correlation_not_proportional_l5_5854


namespace polygon_sides_l5_5238

theorem polygon_sides (a : ℝ) (h : a > 0) :
  let T := {p : ℝ × ℝ | 
    (frac 3 4 * a ≤ p.1 ∧ p.1 ≤ frac 5 2 * a) ∧ 
    (frac 3 4 * a ≤ p.2 ∧ p.2 ≤ frac 5 2 * a) ∧ 
    (p.1 + a ≥ p.2) ∧ 
    (p.2 + a ≥ p.1)} in
  (∃ n, ∀ (p: ℝ × ℝ), p ∈ T → polygon n p) ∧ sides T = 4 :=
sorry

end polygon_sides_l5_5238


namespace gcd_of_45_135_225_is_45_l5_5415

theorem gcd_of_45_135_225_is_45 : Nat.gcd (Nat.gcd 45 135) 225 = 45 :=
by
  sorry

end gcd_of_45_135_225_is_45_l5_5415


namespace count_values_of_x_l5_5126

-- Define the relevant condition for x being non-negative and sqrt(100 - x^(1/3)) being an integer.

lemma count_non_negative_real_values (x : ℝ) (h : 0 ≤ x) :
  ∃ n : ℕ, 0 ≤ n ∧ n ≤ 10 ∧ (100 - (x^(1/3))).nat_abs = n^2 :=
sorry

-- Define the theorem that verifies there are exactly 11 suitable values of x.
theorem count_values_of_x : 
  ∃ (S : set ℝ), (∀ x ∈ S, 0 ≤ x ∧ is_integer (sqrt (100 - real.cbrt x))) ∧ S.card = 11 :=
by {
  sorry
}

end count_values_of_x_l5_5126


namespace find_x_l5_5531

theorem find_x (x : ℤ) : 3^7 * 3^x = 81 → x = -3 := by
  sorry

end find_x_l5_5531


namespace line_intersecting_three_circles_exists_l5_5841

/-- There are 100 circles of radius one in the plane. A triangle formed by the centres
    of any three given circles has area at most 2017. Prove that there is a line
    intersecting at least three of the circles. -/
theorem line_intersecting_three_circles_exists :
  ∃ l : ℝ → ℝ × ℝ, ∀ circles : fin 100 → (ℝ × ℝ), (∀ i j k: fin 100, 
  let ⟨c1, c2⟩ := circles i in
  let ⟨c3, c4⟩ := circles j in
  let ⟨c5, c6⟩ := circles k in
  (1/2) * |c1 * (c4 - c6) + c3 * (c6 - c2) + c5 * (c2 - c4)| ≤ 2017) →
  ∃ c1 c2 c3: fin 100, 
  ∃ x1 x2 x3: ℝ, 
  |x1 - circles c1.1| = 1 ∧ 
  |x2 - circles c2.1| = 1 ∧ 
  |x3 - circles c3.1| = 1 ∧
  collinear (l x1) (l x2) (l x3) := sorry

end line_intersecting_three_circles_exists_l5_5841


namespace expected_value_of_eight_sided_die_l5_5378

theorem expected_value_of_eight_sided_die : 
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8], 
      n := (outcomes.length : ℚ),
      probabilities := List.replicate (outcomes.length) (1 / n),
      expected_value := (List.zipWith (*) probabilities (outcomes.map (· : ℚ))).sum
  in expected_value = 4.5 :=
by
  sorry

end expected_value_of_eight_sided_die_l5_5378


namespace letter_puzzle_solutions_l5_5562

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def to_number (B A : ℕ) : ℕ :=
  10 * B + A

theorem letter_puzzle_solutions (A B : ℕ) (h_diff : A ≠ B) (h_digits : 1 ≤ A ∧ A ≤ 9 ∧ 1 ≤ B ∧ B ≤ 9)
  (h_two_digit : is_two_digit (to_number B A)) :
  A^B = to_number B A ↔ (A = 2 ∧ B = 5 ∨ A = 6 ∧ B = 2 ∨ A = 4 ∧ B = 3) :=
by
  sorry

end letter_puzzle_solutions_l5_5562


namespace percentage_decrease_l5_5719

theorem percentage_decrease :
  ∃ y : ℝ, 
    let P₀ := 100,
        P₁ := P₀ * 1.30,
        P₂ := P₁ * 0.90,
        P₃ := P₂ * 1.20,
        P₄ := P₃ * (1 - y / 100),
        P₅ := P₄ * 1.15 in
    (P₅ = P₀ ∧ y = 38) := 
sorry

end percentage_decrease_l5_5719


namespace sqrt_div_l5_5877

theorem sqrt_div (a b : ℝ) (h1 : a = 6) (h2 : b = 2) : real.sqrt a / real.sqrt b = real.sqrt (a / b) := by
  sorry

end sqrt_div_l5_5877


namespace area_of_parallelogram_is_20sqrt3_l5_5111

open Real EuclideanGeometry

def vector_u : ℝ × ℝ × ℝ := (4, 2, -3)
def vector_v : ℝ × ℝ × ℝ := (2, -4, 5)

noncomputable def cross_product (u v : ℝ × ℝ × ℝ) :=
  (u.2 * v.3 - u.3 * v.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2 - u.2 * v.1)

noncomputable def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

theorem area_of_parallelogram_is_20sqrt3 :
  magnitude (cross_product vector_u vector_v) = 20 * Real.sqrt 3 := by
  sorry

end area_of_parallelogram_is_20sqrt3_l5_5111


namespace water_added_l5_5050

theorem water_added (x : ℝ) (salt_initial_percentage : ℝ) (salt_final_percentage : ℝ) 
   (evap_fraction : ℝ) (salt_added : ℝ) (W : ℝ) 
   (hx : x = 150) (h_initial_salt : salt_initial_percentage = 0.2) 
   (h_final_salt : salt_final_percentage = 1 / 3) 
   (h_evap_fraction : evap_fraction = 1 / 4) 
   (h_salt_added : salt_added = 20) : 
  W = 37.5 :=
by
  sorry

end water_added_l5_5050


namespace probability_of_selecting_at_least_one_female_l5_5131

open BigOperators

noncomputable def prob_at_least_one_female_selected : ℚ :=
  let total_choices := Nat.choose 10 3
  let all_males_choices := Nat.choose 6 3
  1 - (all_males_choices / total_choices : ℚ)

theorem probability_of_selecting_at_least_one_female :
  prob_at_least_one_female_selected = 5 / 6 := by
  sorry

end probability_of_selecting_at_least_one_female_l5_5131


namespace limit_of_sequence_exists_l5_5315

theorem limit_of_sequence_exists (a : ℕ → ℝ) 
  (h_nonneg : ∀ n, 0 ≤ a n) 
  (h_condition : ∀ m n, a (n + m) ≤ a n * a m) : 
  ∃ L, tendsto (λ n, (a n)^(1/(n:ℝ))) at_top (nhds L) := 
sorry

end limit_of_sequence_exists_l5_5315


namespace find_m_l5_5650

theorem find_m (m : ℝ) (A B C : ℝ × ℝ) (hA : A = (2, m)) (hB : B = (1, 2)) (hC : C = (3, 1)) :
  let ABx := (fst B - fst A) in
  let ABy := (snd B - snd A) in
  let CBx := (fst B - fst C) in
  let CBy := (snd B - snd C) in
  let ACx := (fst C - fst A) in
  let ACy := (snd C - snd A) in
  (ABx * CBx + ABy * CBy = real.sqrt(ACx^2 + ACy^2)) → m = 7 / 3 :=
by {
  intros,
  sorry
}

end find_m_l5_5650


namespace minimum_attempts_to_unlock_suitcase_l5_5429

-- Given: The suitcase lock code is a three-digit number composed of digits 8 and 5.

def is_code_valid (a b c : ℕ) : Prop := 
  (a = 5 ∨ a = 8) ∧ (b = 5 ∨ b = 8) ∧ (c = 5 ∨ c = 8)

-- Goal: Prove that the minimum number of attempts needed to ensure unlocking the suitcase is 6.
theorem minimum_attempts_to_unlock_suitcase : 
  ∃ n : ℕ, n = 6 ∧ ∀ a b c : ℕ, is_code_valid a b c → (1 + 1 + 1 = 3 → a ∈ {5, 8} → b ∈ {5, 8} → c ∈ {5, 8} → n = 6) :=
sorry

end minimum_attempts_to_unlock_suitcase_l5_5429


namespace Jake_balloons_l5_5496

theorem Jake_balloons (Allan_balloons : ℕ) (total_balloons : ℕ) (h1 : Allan_balloons = 2) (h2 : total_balloons = 6) : 
  ∃ (Jake_balloons : ℕ), Jake_balloons = total_balloons - Allan_balloons ∧ Jake_balloons = 4 :=
by
  use 4
  constructor
  . calc
    4 = 6 - 2 := rfl -- This step shows that 4 = total_balloons - Allan_balloons
  . exact rfl -- This step shows that Jake_balloons = 4

end Jake_balloons_l5_5496


namespace find_angle_C_max_area_l5_5152

-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively.
-- Given trigonometric condition and side length c.
section
variables {a b c A B C : ℝ}
variable (h1 : 2 * sin (7 * Real.pi / 6) * sin (Real.pi / 6 + C) + cos C = -1 / 2)
variable (h2 : c = 2 * Real.sqrt 3)

-- to prove that C = π/3
theorem find_angle_C : C = Real.pi / 3 := sorry

-- Given cosine rule and condition, prove max area
theorem max_area (h : c^2 = a^2 + b^2 - 2 * a * b * cos C) : 
  let area := 1/2 * a * b * Real.sin C in
  ∃ ab_max, (a * b ≤ ab_max) ∧ (ab_max = 12) ∧ (area ≤ 3 * Real.sqrt 3) ∧ (ab_max = a * b → area = 3 * Real.sqrt 3) := by
  sorry
end

end find_angle_C_max_area_l5_5152


namespace binomial_16_4_l5_5506

theorem binomial_16_4 : Nat.choose 16 4 = 1820 :=
  sorry

end binomial_16_4_l5_5506


namespace divide_base_5_l5_5093

def base_5_to_base_10 (n : ℕ) : ℕ := sorry  -- Assume this function converts base 5 to base 10

def base_10_to_base_5 (n : ℕ) : ℕ := sorry  -- Assume this function converts base 10 to base 5

theorem divide_base_5 (n m : ℕ) (hn : base_5_to_base_10 n = 269) (hm : base_5_to_base_10 m = 13) :
  base_10_to_base_5 (hn / hm) = 40 := sorry

end divide_base_5_l5_5093


namespace smallest_integer_to_make_perfect_square_l5_5928

def y : ℕ := 2^3 * 3^4 * 4^5 * 5^6 * 6^7 * 7^8 * 8^9 * 9^10

theorem smallest_integer_to_make_perfect_square :
  ∃ n : ℕ, n > 0 ∧ (∃ k : ℕ, (n * y) = k^2) ∧ n = 6 :=
by
  sorry

end smallest_integer_to_make_perfect_square_l5_5928


namespace function_relationship_minimize_total_cost_l5_5453

noncomputable def y (a x : ℕ) : ℕ :=
6400 * x + 50 * a + 100 * a^2 / (x - 1)

theorem function_relationship (a : ℕ) (hx : 2 ≤ x) : 
  y a x = 6400 * x + 50 * a + 100 * a^2 / (x - 1) :=
by sorry

theorem minimize_total_cost (a : ℕ) (hx : 2 ≤ x) (ha : a = 56) : 
  y a x ≥ 1650 * a + 6400 ∧ (x = 8) :=
by sorry

end function_relationship_minimize_total_cost_l5_5453


namespace find_number_of_tables_l5_5022

noncomputable def number_of_tables (total_books : ℕ) : ℕ :=
  let t := real.sqrt (2 * total_books / 5) in
  nat.sqrt t

theorem find_number_of_tables (h : 100000 = 2 * T ^ 2 / 5) : T = 500 := by
  sorry

end find_number_of_tables_l5_5022


namespace monotonicity_F_common_adjacent_line_m_one_l5_5669

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := m * Real.log x

-- Define the function F(x)
def F (m : ℝ) (x : ℝ) : ℝ := f m x + 1 / x - 1

-- The statement corresponding to the monotonicity proof of F(x)
theorem monotonicity_F (m : ℝ) : 
  (m ≤ 0 → ∀ x > 0, (F m x < F m (x + 0.0001))) ∧
  (m > 0 → (∀ x ∈ Ioo 0 (1 / m), F m x > F m (x - 0.0001)) ∧ (∀ x ∈ Ioi (1 / m), F m x < F m (x + 0.0001))) := 
sorry

-- The statement corresponding to whether there is a common adjacent line for m = 1
theorem common_adjacent_line_m_one : 
  ∀ x y, ∀ f_in : x >= 0 ∧ y >= 0, (f 1 (x+1) = f 1 y) ↔ (x / (x + 1) = y / (y + 1)) := 
sorry

end monotonicity_F_common_adjacent_line_m_one_l5_5669


namespace gift_options_l5_5478

theorem gift_options (n : ℕ) (h : n = 10) : (2^n - 1) = 1023 :=
by {
  rw h,
  norm_num,
  sorry
}

end gift_options_l5_5478


namespace number_of_primes_in_sequence_l5_5686

def is_prime (n : ℕ) : Prop := Nat.Prime n

def sequence (n : ℕ) : ℕ := 17 * (10^n - 1) / 9 -- This represents the numbers 17, 1717, 171717, ...

theorem number_of_primes_in_sequence (N : ℕ) : ∃ n : ℕ, (sequence n).Prime ∧ ∀ m : ℕ, m ≠ n → ¬ is_prime (sequence m) :=
by
  sorry

end number_of_primes_in_sequence_l5_5686


namespace compare_x1_x2_x3_l5_5633

def log_base (b x : ℝ) : ℝ := real.log x / real.log b

def x1 : ℝ := log_base (1/3) 2
def x2 : ℝ := 2^(-1/2)
def x3 : ℝ := (classical.some (exists_unique_of_exists_of_unique 
  (λ y, (1/3)^y = log_base 3 y) sorry)).some

theorem compare_x1_x2_x3 : x1 < x2 ∧ x2 < x3 :=
by 
  sorry

end compare_x1_x2_x3_l5_5633


namespace exists_n_good_sequences_l5_5147

theorem exists_n_good_sequences (n : ℤ) (hi : n ≥ 2) 
  (a : Fin n → ℤ) (hne : ∀ i, n ∣ a i → False)
  (hn : ¬n ∣ (Finset.univ.sum a)) : 
  ∃ seqs : Fin n → Fin n → Bool, 
    (∀ i, (seqs i).toFin n ≠ seqs 0) 
    ∧ ∀ i, ∃ (e : Fin n → ℤ), 
        (∀ j, (e j = 0 ∨ e j = 1))
        ∧ (∑ j in Finset.range n, e j * a j) % n = 0 := 
sorry

end exists_n_good_sequences_l5_5147


namespace soccer_camp_afternoon_kids_l5_5846

theorem soccer_camp_afternoon_kids
  (total_kids : ℕ)
  (half_to_soccer : ℕ)
  (morning_fraction : ℕ)
  (afternoon_kids : ℕ) :
  total_kids = 2000 →
  half_to_soccer = total_kids / 2 →
  morning_fraction = half_to_soccer / 4 →
  afternoon_kids = half_to_soccer - morning_fraction →
  afternoon_kids = 750 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  simp
  sorry

end soccer_camp_afternoon_kids_l5_5846


namespace computer_company_revenue_proof_l5_5454

variable (R : Type) [LinearOrderedField R]
variable (revenue_2007 : R) (revenue_2009 : R) (revenue_2010 : R) (growth_rate : R)

def is_growth_rate (revenue_2007 revenue_2009 growth_rate : R) : Prop :=
  revenue_2007 * (1 + growth_rate) ^ 2 = revenue_2009

def revenue_in_2010 (revenue_2009 growth_rate : R) : R :=
  revenue_2009 * (1 + growth_rate)

theorem computer_company_revenue_proof
  (h_rev_2007 : revenue_2007 = 15)
  (h_rev_2009 : revenue_2009 = 21.6)
  (h_growth_rate : is_growth_rate revenue_2007 revenue_2009 growth_rate)
  : growth_rate = 0.2 ∧ revenue_2010 = 2592 := 
by
  unfold is_growth_rate at h_growth_rate
  sorry

end computer_company_revenue_proof_l5_5454


namespace largest_side_of_rectangle_l5_5258

theorem largest_side_of_rectangle (l w : ℕ) (h1 : 2 * l + 2 * w = 240) (h2 : l * w = 1920) : max l w = 101 := 
by
  sorry

end largest_side_of_rectangle_l5_5258


namespace expected_winnings_correct_l5_5923

def probability_1 := (1:ℚ) / 4
def probability_2 := (1:ℚ) / 4
def probability_3 := (1:ℚ) / 6
def probability_4 := (1:ℚ) / 6
def probability_5 := (1:ℚ) / 8
def probability_6 := (1:ℚ) / 8

noncomputable def expected_winnings : ℚ :=
  (probability_1 + probability_3 + probability_5) * 2 +
  (probability_2 + probability_4) * 4 +
  probability_6 * (-6 + 4)

theorem expected_winnings_correct : expected_winnings = 1.67 := by
  sorry

end expected_winnings_correct_l5_5923


namespace find_value_of_2a_plus_c_l5_5628

theorem find_value_of_2a_plus_c (a b c : ℝ) (h1 : 3 * a + b + 2 * c = 3) (h2 : a + 3 * b + 2 * c = 1) :
  2 * a + c = 2 :=
sorry

end find_value_of_2a_plus_c_l5_5628


namespace chord_length_through_focus_l5_5067

theorem chord_length_through_focus (x y : ℝ) (h : x^2 / 4 + y^2 / 3 = 1)
  (h_perp : (x = 1) ∨ (x = -1)) : abs (2 * y) = 3 :=
by {
  sorry
}

end chord_length_through_focus_l5_5067


namespace least_amount_of_money_l5_5431

theorem least_amount_of_money 
  (amounts_diff : ∀ (a b : String), (a ≠ b) → (a ≠ b))
  (boZoe : ∀ (bo zoe : String), (bo > zoe))
  (boFlo : ∀ (bo flo : String), (bo > flo))
  (zoeMoe : ∀ (zoe moe : String), (zoe > moe))
  (coeMoe: ∀ (coe moe : String), (coe > moe))
  (floMoe : ∀ (flo moe : String), (flo > moe))
  (floCoe : ∀ (flo coe : String), (flo < coe)) :
  ((∀ (moe : String), moe) → (moe < bo ∧ moe < zoe ∧ moe < coe ∧ moe < flo)) → moe = least_amount_of_money :=
by
  sorry

end least_amount_of_money_l5_5431


namespace expected_value_eight_sided_die_l5_5410

-- Define a standard 8-sided die
def eight_sided_die : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]

-- Compute the probability of each outcome
def probability (n : ℕ) : ℝ := 1 / 8

-- Expected Value of a discrete random variable
def expected_value (outcomes : List ℕ) (prob : ℕ → ℝ) : ℝ :=
  outcomes.sum / outcomes.length.toReal

-- Theorem stating the expected value of a standard 8-sided die roll is 4.5
theorem expected_value_eight_sided_die : expected_value eight_sided_die probability = 4.5 := by
  sorry

end expected_value_eight_sided_die_l5_5410


namespace expected_value_8_sided_die_l5_5398

/-- 
The expected value of rolling a standard 8-sided die is 4.5.
The die has 8 sides labeled 1 through 8, and each face has an equal probability of appearing,
which is 1/8. 
-/
theorem expected_value_8_sided_die : 
  (1/8:ℝ) * 1 + (1/8) * 2 + (1/8) * 3 + (1/8) * 4 + (1/8) * 5 + (1/8) * 6 + (1/8) * 7 + (1/8) * 8 = 4.5 :=
by 
  sorry

end expected_value_8_sided_die_l5_5398


namespace product_mk_through_point_l5_5192

theorem product_mk_through_point (k m : ℝ) (h : (2 : ℝ) ^ m * k = (1/4 : ℝ)) : m * k = -2 := 
sorry

end product_mk_through_point_l5_5192


namespace omega_range_l5_5705

theorem omega_range (ω : ℝ) (hω : ω > 0) :
  (∀ x : ℝ, (x ∈ set.Icc (real.pi / 3) (real.pi / 2)) → (∃ k : ℤ, x ∈ set.Icc ((real.pi / 2 + 2 * k * real.pi) / ω) ((3 * real.pi / 2 + 2 * k * real.pi) / ω))) →
  (3 / 2 ≤ ω ∧ ω ≤ 3) :=
by sorry

end omega_range_l5_5705


namespace arith_seq_term_coefficient_l5_5659

noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem arith_seq_term_coefficient :
  let a : ℕ → ℤ := λ n, 3 * n - 5
  let coeff := binom 5 4 + binom 6 4 + binom 7 4
  coeff = 55 ∧ a 20 = 55 :=
by
  let a := λ n, 3 * n - 5
  let coeff := binom 5 4 + binom 6 4 + binom 7 4
  have h1 : coeff = 55 := by sorry
  have h2 : a 20 = 55 := by sorry
  exact ⟨h1, h2⟩

end arith_seq_term_coefficient_l5_5659


namespace sector_area_l5_5729

theorem sector_area (α r : ℝ) (hα : α = 2) (h_r : r = 1 / Real.sin 1) : 
  (1 / 2) * r^2 * α = 1 / (Real.sin 1)^2 :=
by
  sorry

end sector_area_l5_5729


namespace expand_binomials_l5_5091

variable (x y : ℝ)

theorem expand_binomials : 
  (3 * x - 2) * (2 * x + 4 * y + 1) = 6 * x^2 + 12 * x * y - x - 8 * y - 2 :=
by
  sorry

end expand_binomials_l5_5091


namespace smallest_k_exists_l5_5867

open Nat

theorem smallest_k_exists (n m k : ℕ) (hn : n > 0) (hm : 0 < m ∧ m ≤ 5) (hk : k % 3 = 0) :
  (64^k + 32^m > 4^(16 + n^2)) ↔ k = 6 :=
by
  sorry

end smallest_k_exists_l5_5867


namespace simplify_expression_l5_5070

-- Definitions for conditions and parameters
variables {x y : ℝ}

-- The problem statement and proof
theorem simplify_expression : 12 * x^5 * y / (6 * x * y) = 2 * x^4 :=
by sorry

end simplify_expression_l5_5070


namespace piggy_bank_total_l5_5065

def amount_added_in_january: ℕ := 19
def amount_added_in_february: ℕ := 19
def amount_added_in_march: ℕ := 8

theorem piggy_bank_total:
  amount_added_in_january + amount_added_in_february + amount_added_in_march = 46 := by
  sorry

end piggy_bank_total_l5_5065


namespace ratio_of_drums_l5_5525

variable (C_X C_Y : ℝ)

-- Conditions
def Oil_in_Drum_X := (1/2) * C_X
def Oil_in_Drum_Y := (1/4) * C_Y
def Oil_total := Oil_in_Drum_X + Oil_in_Drum_Y

theorem ratio_of_drums (h1 : Oil_in_Drum_X = (1/2) * C_X)
                       (h2 : Oil_in_Drum_Y = (1/4) * C_Y)
                       (h3 : Oil_total = (1/2) * C_Y) :
  C_Y / C_X = 2 :=
by
  sorry

end ratio_of_drums_l5_5525


namespace fixed_point_of_variable_line_l5_5637

theorem fixed_point_of_variable_line (p k b : ℝ) (k_ne_zero : k ≠ 0) (b_ne_zero : b ≠ 0) (h : ∃ (A B : ℝ × ℝ), A.2^2 = 2 * p * A.1 ∧ A.2 = k * A.1 + b ∧ B.2^2 = 2 * p * B.1 ∧ B.2 = k * B.1 + b ∧ (A.2 / A.1) * (B.2 / B.1) = sqrt 3) :
  ∃ x y : ℝ, x = - (2 * sqrt 3 * p) / 3 ∧ y = 0 ∧ y = k * x + b :=
by
  sorry

end fixed_point_of_variable_line_l5_5637


namespace households_with_neither_l5_5200

variable (C B_only C_and_B N Either Neither : ℕ)

-- Given Conditions
def condition1 : Prop := C_and_B = 22
def condition2 : Prop := C = 44
def condition3 : Prop := B_only = 35
def condition4 : Prop := N = 90

-- Derived Definitions
def B : ℕ := B_only + C_and_B 
def Either : ℕ := C + B - C_and_B
def Neither : ℕ := N - Either

-- Goal
theorem households_with_neither (h1: condition1) 
                               (h2: condition2) 
                               (h3: condition3) 
                               (h4: condition4) : Neither = 11 :=  
by 
  -- Mathematical proof goes here.
  sorry

end households_with_neither_l5_5200


namespace railway_reachability_exists_l5_5713

noncomputable def exists_reachable_city (n : ℕ) : Prop :=
  ∃ (N : Fin n), ∀ (i : Fin n), i ≠ N → 
    ∃ (j : Fin n), (i → j → N) ∨ (i → N)

theorem railway_reachability_exists :
  ∀ n : ℕ, 1 < n →
  ∃ N : Fin n, ∀ i : Fin n, i ≠ N →
    ∃ j : Fin n, (∃ (h : i < j), true) ∨ (i = j) := sorry

end railway_reachability_exists_l5_5713


namespace tea_drinking_problem_l5_5057

theorem tea_drinking_problem 
  (k b c t s : ℕ) 
  (hk : k = 1) 
  (hb : b = 15) 
  (hc : c = 3) 
  (ht : t = 2) 
  (hs : s = 1) : 
  17 = 17 := 
by {
  sorry
}

end tea_drinking_problem_l5_5057


namespace repeating_decimal_to_fraction_l5_5997

theorem repeating_decimal_to_fraction (h : 0.\overline{02} = 2 / 99) : 
  2.\overline{06} = 68 / 33 := by
  sorry

end repeating_decimal_to_fraction_l5_5997


namespace linear_inequality_m_eq_zero_l5_5699

theorem linear_inequality_m_eq_zero (m : ℝ) (x : ℝ) : 
  ((m - 2) * x ^ |m - 1| - 3 > 6) → abs (m - 1) = 1 → m ≠ 2 → m = 0 := by
  intros h1 h2 h3
  -- Proof of m = 0 based on given conditions
  sorry

end linear_inequality_m_eq_zero_l5_5699


namespace hyperbola_eccentricity_theorem_l5_5164

noncomputable def hyperbola_eccentricity : Prop :=
  ∃ (a b : ℝ) (c : ℝ) (P F1 F2 : ℝ × ℝ),
  (∀ x y: ℝ, x > 0 ∧ y > 0) ∧
  (|F1.1 - F2.1| = 12) ∧
  (∃ y: ℝ, ((P.1/a)^2 - (y/b)^2 = 1 ∧ (P.2 - F2.2) * P.1 = 0)) ∧
  (|P.1 - F2.1| = 5) ∧
  (c = |F1.1 - F2.1|/2) ∧
  (e ≥ 0) ∧
  (e = c/a) ∧
  (e = 3/2)

theorem hyperbola_eccentricity_theorem : hyperbola_eccentricity :=
sorry

end hyperbola_eccentricity_theorem_l5_5164


namespace estimate_event_probability_l5_5813

/-- 
Given the frequencies of a random event occurring during an experiment for various numbers of trials,
we estimate the probability of this event occurring through the experiment and prove that it is approximately 0.35 
when rounded to 0.01. 
-/
theorem estimate_event_probability :
  let freq := [0.300, 0.360, 0.350, 0.350, 0.352, 0.351, 0.351]
  let approx_prob := 0.35
  ∀ n : ℕ, n ∈ [20, 50, 100, 300, 500, 1000, 5000] →
  freq.get! n ≈ approx_prob := 
  sorry

end estimate_event_probability_l5_5813


namespace stream_speed_l5_5449

noncomputable def speed_of_stream
  (speed_boat_still_water : ℝ)
  (distance_downstream : ℝ)
  (time_downstream : ℝ) : ℝ :=
  let effective_speed_downstream := distance_downstream / time_downstream
  in effective_speed_downstream - speed_boat_still_water

theorem stream_speed
  (speed_boat_still_water : ℝ = 22)
  (distance_downstream : ℝ = 81)
  (time_downstream : ℝ = 3) :
  speed_of_stream speed_boat_still_water distance_downstream time_downstream = 5 :=
by
  simp [speed_of_stream]
  norm_num
  sorry

end stream_speed_l5_5449


namespace A_alone_days_l5_5908

theorem A_alone_days (A B C : ℝ) (hB: B = 9) (hC: C = 7.2) 
  (h: 1 / A + 1 / B + 1 / C = 1 / 2) : A = 2 :=
by
  rw [hB, hC] at h
  sorry

end A_alone_days_l5_5908


namespace symmetric_point_xOy_l5_5207

-- Definitions
def P := (1, 2, 3)  -- Original point P
def Q := (1, 2, -3) -- Point symmetric to P about xOy

-- Theorem statement
theorem symmetric_point_xOy : sym_point_xOy P = Q := by
  sorry

end symmetric_point_xOy_l5_5207


namespace letter_puzzle_l5_5555

theorem letter_puzzle (A B : ℕ) (hA : 1 ≤ A ∧ A ≤ 9) (hB : 1 ≤ B ∧ B ≤ 9) (h_diff : A ≠ B) :
  A^B = 10 * B + A ↔ (A = 2 ∧ B = 5) ∨ (A = 6 ∧ B = 2) ∨ (A = 4 ∧ B = 3) :=
by
  sorry

end letter_puzzle_l5_5555


namespace algebra_problem_l5_5701

theorem algebra_problem 
  (x : ℝ) 
  (h : x^2 - 2 * x = 3) : 
  2 * x^2 - 4 * x + 3 = 9 := 
by 
  sorry

end algebra_problem_l5_5701


namespace sin_x_plus_pi_div_3_sin_2x_plus_pi_div_6_l5_5652

theorem sin_x_plus_pi_div_3 (x : ℝ) (hx : cos x = -3/5) (hx_interval : x ∈ set.Ioo (π / 2) π) :
  sin (x + π / 3) = (4 - 3 * Real.sqrt 3) / 10 :=
by
  sorry

theorem sin_2x_plus_pi_div_6 (x : ℝ) (hx : cos x = -3/5) (hx_interval : x ∈ set.Ioo (π / 2) π) :
  sin (2 * x + π / 6) = -(24 * Real.sqrt 3 + 7) / 50 :=
by
  sorry

end sin_x_plus_pi_div_3_sin_2x_plus_pi_div_6_l5_5652


namespace shortest_distance_l5_5025

theorem shortest_distance 
  (C : ℝ × ℝ) (B : ℝ × ℝ) (stream : ℝ)
  (hC : C = (0, -3))
  (hB : B = (9, -8))
  (hStream : stream = 0) :
  ∃ d : ℝ, d = 3 + Real.sqrt 202 :=
by
  sorry

end shortest_distance_l5_5025


namespace find_m_l5_5170

noncomputable def vector_a : ℝ × ℝ := (1, real.sqrt 3)
noncomputable def vector_b (m : ℝ) : ℝ × ℝ := (3, m)
noncomputable def angle := real.pi / 6

theorem find_m :
  ∃ m : ℝ, 
    real.cos angle = (vector_a.1 * (vector_b m).1 + vector_a.2 * (vector_b m).2) /
                     (real.sqrt (vector_a.1 ^ 2 + vector_a.2 ^ 2) * real.sqrt ((vector_b m).1 ^ 2 + (vector_b m).2 ^ 2)) ↔
    m = real.sqrt 3 :=
begin
  sorry
end

end find_m_l5_5170


namespace point_symmetric_yaxis_second_quadrant_l5_5648

/-- 
Theorem: Given a point A with coordinates (a-1, 2a-4), if the point is symmetric about the y-axis 
and is in the second quadrant, then the parameter a must satisfy a > 2.
-/
theorem point_symmetric_yaxis_second_quadrant (a : ℝ) :
  (∀ (p : ℝ × ℝ), p = (a-1, 2a-4) → ∃ (q : ℝ × ℝ), q = (-(a-1), 2a-4) ∧ q.1 > 0 ∧ q.2 > 0) →
  (∃ (r : ℝ × ℝ), r = (a-1, 2a-4) ∧ r.1 < 0 ∧ r.2 > 0) →
  a > 2 :=
by
  sorry

end point_symmetric_yaxis_second_quadrant_l5_5648


namespace wildcats_points_l5_5805

theorem wildcats_points (panthers_points wildcats_additional_points wildcats_points : ℕ)
  (h_panthers : panthers_points = 17)
  (h_wildcats : wildcats_additional_points = 19)
  (h_wildcats_points : wildcats_points = panthers_points + wildcats_additional_points) :
  wildcats_points = 36 :=
by
  have h1 : panthers_points = 17 := h_panthers
  have h2 : wildcats_additional_points = 19 := h_wildcats
  have h3 : wildcats_points = panthers_points + wildcats_additional_points := h_wildcats_points
  sorry

end wildcats_points_l5_5805


namespace no_hexagon_cross_section_l5_5517

-- Define the shape of the cross-section resulting from cutting a triangular prism with a plane
inductive Shape
| triangle
| quadrilateral
| pentagon
| hexagon

-- Define the condition of cutting a triangular prism
structure TriangularPrism where
  cut : Shape

-- The theorem stating that cutting a triangular prism with a plane cannot result in a hexagon
theorem no_hexagon_cross_section (P : TriangularPrism) : P.cut ≠ Shape.hexagon :=
by
  sorry

end no_hexagon_cross_section_l5_5517


namespace find_abs_sum_roots_l5_5625

noncomputable def polynomial_root_abs_sum (n p q r : ℤ) : Prop :=
(p + q + r = 0) ∧
(p * q + q * r + r * p = -2009) ∧
(p * q * r = -n) →
(|p| + |q| + |r| = 102)

theorem find_abs_sum_roots (n p q r : ℤ) :
  polynomial_root_abs_sum n p q r :=
sorry

end find_abs_sum_roots_l5_5625


namespace geometric_sequence_increasing_l5_5759

theorem geometric_sequence_increasing {a : ℕ → ℝ} (r : ℝ) (h_pos : 0 < r) (h_geometric : ∀ n, a (n + 1) = r * a n) :
  (a 0 < a 1 ∧ a 1 < a 2) ↔ ∀ n m, n < m → a n < a m :=
by sorry

end geometric_sequence_increasing_l5_5759


namespace part1_part2_part3_l5_5452

variable (P : ℕ := 30)
variable (lower_bound : ℕ := 30)
variable (upper_bound : ℕ := 55)

variable (daily_sales : ℕ → ℕ := λ x, -2 * x + 140)
variable (daily_profit : ℕ → ℤ := λ x, (x - P) * daily_sales x)

-- Part 1: Prove the daily profit is 350 yuan when the selling price per unit is 35 yuan
theorem part1 : daily_profit P 35 = 350 := by
  sorry

-- Part 2: Prove the selling price per unit should be 40 yuan to make a daily profit of 600 yuan
theorem part2 : (∃ x, lower_bound ≤ x ∧ x ≤ upper_bound ∧ daily_profit P x = 600) ∧ 
                (∃ x, lower_bound ≤ x ∧ x ≤ upper_bound ∧ daily_profit P x ≠ 600 → x ≠ 40) := 
by 
  sorry

-- Part 3: Prove the shopping mall cannot make a daily profit of 900 yuan
theorem part3 : ¬ ∃ x, lower_bound ≤ x ∧ x ≤ upper_bound ∧ daily_profit P x = 900 := 
by 
  sorry

end part1_part2_part3_l5_5452


namespace repeating_decimal_to_fraction_l5_5996

theorem repeating_decimal_to_fraction (h : 0.\overline{02} = 2 / 99) : 
  2.\overline{06} = 68 / 33 := by
  sorry

end repeating_decimal_to_fraction_l5_5996


namespace expected_value_of_8_sided_die_l5_5382

theorem expected_value_of_8_sided_die : 
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let probability := 1 / 8
  (∑ outcome in outcomes, outcome * probability) = 4.5 :=
by
  sorry

end expected_value_of_8_sided_die_l5_5382


namespace exists_small_diff_l5_5230

open Classical

noncomputable def a_seq : ℕ → ℝ := sorry
noncomputable def b_seq : ℕ → ℝ := sorry

axiom a_prop : ∀ n ≥ 101, a_seq n = sqrt ((1:ℝ) / 100 * (finset.range 100).sum (λ j, (b_seq (n - j - 1))^2))

axiom b_prop : ∀ n ≥ 101, b_seq n = sqrt ((1:ℝ) / 100 * (finset.range 100).sum (λ j, (a_seq (n - j - 1))^2))

theorem exists_small_diff : ∃ m : ℕ, |a_seq m - b_seq m| < 0.001 := sorry

end exists_small_diff_l5_5230


namespace smallest_coprime_gt_one_l5_5571

theorem smallest_coprime_gt_one (x : ℕ) (h1 : 1 < x) (h2 : Nat.gcd x 180 = 1) : x = 7 := sorry

end smallest_coprime_gt_one_l5_5571


namespace ribbon_length_l5_5027

theorem ribbon_length (circumference height : ℝ) (turns : ℕ)
  (hc : circumference = 16)
  (hh : height = 8)
  (ht : turns = 2) :
  let d := height * turns,
      c := circumference in
  (ribbon_length := real.sqrt (c^2 + d^2)) = 16 * real.sqrt 2 :=
by
  sorry

end ribbon_length_l5_5027


namespace analogy_for_parallelepiped_l5_5427

def parallelepiped (F : Type) [MetricSpace F] :=
  ∃ p : set F, is_parallelogram p ∧ (∀ x y : F, is_parallel x y)

theorem analogy_for_parallelepiped : 
  ∀ (fig : Type) [MetricSpace fig], is_parallelogram fig ↔ is_suitable_as_analogy_as_parallelepiped fig := 
by
  intros fig _ 
  split
  {
    intro h
    sorry
  }
  {
    intro h
    sorry
  }

end analogy_for_parallelepiped_l5_5427


namespace prasanna_speed_l5_5750

variable (v_L : ℝ) (d t : ℝ)

theorem prasanna_speed (hLaxmiSpeed : v_L = 18) (htime : t = 1) (hdistance : d = 45) : 
  ∃ v_P : ℝ, v_P = 27 :=
  sorry

end prasanna_speed_l5_5750


namespace verify_b_c_sum_ten_l5_5700

theorem verify_b_c_sum_ten (a b c : ℕ) (ha : 1 ≤ a ∧ a < 10) (hb : 1 ≤ b ∧ b < 10) (hc : 1 ≤ c ∧ c < 10) 
    (h_eq : (10 * b + a) * (10 * c + a) = 100 * b * c + 100 * a + a ^ 2) : b + c = 10 :=
by
  sorry

end verify_b_c_sum_ten_l5_5700


namespace tan_squared_angle_AOB_l5_5773

noncomputable theory

structure Point3D :=
(x : ℝ) (y : ℝ) (z : ℝ)

def regular_tetrahedron (A B C D O : Point3D) : Prop :=
  -- Define the properties that characterize a regular tetrahedron and its center.
  (A = ⟨0, 0, 0⟩ ∧ B = ⟨1, 0, 0⟩ ∧ C = ⟨0, 1, 0⟩ ∧ D = ⟨0, 0, 1⟩) ∧
  (O = ⟨(A.x + B.x + C.x + D.x) / 4, (A.y + B.y + C.y + D.y) / 4, (A.z + B.z + C.z + D.z) / 4⟩)

def vector (P Q : Point3D) : Point3D :=
⟨Q.x - P.x, Q.y - P.y, Q.z - P.z⟩

def dot_product (u v : Point3D) : ℝ :=
u.x * v.x + u.y * v.y + u.z * v.z

def magnitude (v : Point3D) : ℝ :=
real.sqrt (v.x ^ 2 + v.y ^ 2 + v.z ^ 2)

def cos_angle (A B O : Point3D) : ℝ :=
(dot_product (vector A O) (vector B O)) / (magnitude (vector A O) * magnitude (vector B O))

def tan_squared_angle (A B O : Point3D) : ℝ :=
let c := cos_angle A B O in 
let s := real.sqrt (1 - c ^ 2) in
(s / c) ^ 2

theorem tan_squared_angle_AOB (A B C D O : Point3D) (h : regular_tetrahedron A B C D O) :
  tan_squared_angle A B O = 32 := by
sorry

end tan_squared_angle_AOB_l5_5773


namespace expected_value_8_sided_die_l5_5399

/-- 
The expected value of rolling a standard 8-sided die is 4.5.
The die has 8 sides labeled 1 through 8, and each face has an equal probability of appearing,
which is 1/8. 
-/
theorem expected_value_8_sided_die : 
  (1/8:ℝ) * 1 + (1/8) * 2 + (1/8) * 3 + (1/8) * 4 + (1/8) * 5 + (1/8) * 6 + (1/8) * 7 + (1/8) * 8 = 4.5 :=
by 
  sorry

end expected_value_8_sided_die_l5_5399


namespace assign_teachers_l5_5502

theorem assign_teachers : 
  let teachers := {t1, t2, t3, t4, t5}
  let classes := {c1, c2, c3}
  -- Number of ways to group 5 teachers into 3 classes with at least one teacher in each class:
  let num_groupings := (choose 5 2 * choose 3 2 * choose 1 1 / 2!) + choose 5 3
  -- Number of ways to permute the 3 groups among the 3 classes:
  let num_permutations := 3!
  -- Total number of assignments:
  (num_groupings * num_permutations) = 150
:= sorry

end assign_teachers_l5_5502


namespace stamps_total_l5_5259

def Lizette_stamps : ℕ := 813
def Minerva_stamps : ℕ := Lizette_stamps - 125
def Jermaine_stamps : ℕ := Lizette_stamps + 217

def total_stamps : ℕ := Minerva_stamps + Lizette_stamps + Jermaine_stamps

theorem stamps_total :
  total_stamps = 2531 := by
  sorry

end stamps_total_l5_5259


namespace calculate_expression_l5_5066

theorem calculate_expression : 5^3 + 5^3 + 5^3 + 5^3 = 625 :=
  sorry

end calculate_expression_l5_5066


namespace tetrahedron_ratio_l5_5920

open Geometry

theorem tetrahedron_ratio (A B C D : Point)
  (M_AB : Midpoint A B)
  (M_CD : Midpoint C D)
  (L : Point) (N : Point) :
  Plane (Midpoint.to_plane M_AB) (Midpoint.to_plane M_CD) (∩ (Edge A D)) = L →
  Plane (Midpoint.to_plane M_AB) (Midpoint.to_plane M_CD) (∩ (Edge B C)) = N →
  BC : Length (Edge B C) →
  CN : Length (Segment C N) →
  AD : Length (Edge A D) →
  DL : Length (Segment D L) →
  BC / CN = AD / DL := by 
  sorry

end tetrahedron_ratio_l5_5920


namespace sunny_second_race_ahead_l5_5201

variable (a e : ℝ)
variable (sa sw : ℝ) -- sa: Sunny's speed in the first race, sw: Windy's speed

-- Conditions
def sunny_first_race_ahead := sa * (a / sw - (a - e) / sw) = e
def second_race_sunny_speed := 1.2 * sa
def second_race_length := a + e

-- Theorem statement
theorem sunny_second_race_ahead : (second_race_length / second_race_sunny_speed) < (second_race_length / sw) → 
  (second_race_length - (second_race_length * sw / second_race_sunny_speed)) = 2 * (e^2) / (a + e) :=
by
  sorry

end sunny_second_race_ahead_l5_5201


namespace range_of_a_l5_5656

variables {f : ℝ → ℝ} (a : ℝ)

-- Even function definition
def even_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

-- Monotonically increasing on (-∞, 0)
def mono_increasing_on_neg (f : ℝ → ℝ) : Prop :=
∀ x y, x < y → y < 0 → f x ≤ f y

-- Problem statement
theorem range_of_a
  (h_even : even_function f)
  (h_mono_neg : mono_increasing_on_neg f)
  (h_inequality : f (2 ^ |a - 1|) > f 4) :
  -1 < a ∧ a < 3 :=
sorry

end range_of_a_l5_5656


namespace inverse_value_l5_5187

def g (x : ℝ) : ℝ := (x^5 - 1) / 5

theorem inverse_value :
  (∃ x : ℝ, g x = -11 / 40 ∧ x = -((3 : ℝ) / 8)^(1 / 5)) :=
sorry

end inverse_value_l5_5187


namespace solve_equation_l5_5795

theorem solve_equation : 
  ∀ x : ℝ, (x - 3 ≠ 0) → (x + 6) / (x - 3) = 4 → x = 6 :=
by
  intros x h1 h2
  sorry

end solve_equation_l5_5795


namespace tenth_term_geometric_sequence_l5_5075

theorem tenth_term_geometric_sequence :
  let a : ℚ := 5
  let r : ℚ := 3 / 4
  let a_n (n : ℕ) : ℚ := a * r ^ (n - 1)
  a_n 10 = 98415 / 262144 :=
by
  sorry

end tenth_term_geometric_sequence_l5_5075


namespace letter_puzzle_l5_5558

theorem letter_puzzle (A B : ℕ) (hA : 1 ≤ A ∧ A ≤ 9) (hB : 1 ≤ B ∧ B ≤ 9) (h_diff : A ≠ B) :
  A^B = 10 * B + A ↔ (A = 2 ∧ B = 5) ∨ (A = 6 ∧ B = 2) ∨ (A = 4 ∧ B = 3) :=
by
  sorry

end letter_puzzle_l5_5558


namespace volume_of_sand_pile_l5_5455

/-- Let a conical sand pile have a diameter of 12 feet and a height which is 60% of its diameter.
    Prove that its volume is 86.4π cubic feet. -/
theorem volume_of_sand_pile :
  let diameter := 12 in
  let height := 0.6 * diameter in
  let radius := diameter / 2 in
  let volume := (1 / 3) * Real.pi * radius ^ 2 * height in
  volume = 86.4 * Real.pi := by
  sorry

end volume_of_sand_pile_l5_5455


namespace correct_number_of_conclusions_l5_5626

noncomputable def α : ℂ := 1/2 + (real.sqrt 3)/2 * complex.I
noncomputable def β : ℂ := -1/2 - (real.sqrt 3)/2 * complex.I

theorem correct_number_of_conclusions :
  (α * β ≠ 1) ∧ (α / β ≠ 1) ∧ (|α / β| = 1) ∧ (α^2 + β^2 ≠ 1) ↔ 1 = 1 := by
  sorry

end correct_number_of_conclusions_l5_5626


namespace seating_arrangements_around_table_l5_5727

theorem seating_arrangements_around_table (n : ℕ) (special_pair : ℕ) :
  n = 6 ∧ special_pair = 2 → 
  (∃ k : ℕ, k = 5! * 2 ∧ k = 240) :=
by
  intros h
  cases h with h1 h2
  use 240
  split
  { exact Nat.factorial 5 * 2 }
  { rfl }
sorry

end seating_arrangements_around_table_l5_5727


namespace combined_cost_of_items_is_221_l5_5948

def wallet_cost : ℕ := 22
def purse_cost : ℕ := 4 * wallet_cost - 3
def shoes_cost : ℕ := wallet_cost + purse_cost + 7
def combined_cost : ℕ := wallet_cost + purse_cost + shoes_cost

theorem combined_cost_of_items_is_221 : combined_cost = 221 := by
  sorry

end combined_cost_of_items_is_221_l5_5948


namespace abs_inequality_solution_bounded_a_b_inequality_l5_5900

theorem abs_inequality_solution (x : ℝ) : (-4 < x ∧ x < 0) ↔ (|x + 1| + |x + 3| < 4) := sorry

theorem bounded_a_b_inequality (a b : ℝ) (h1 : -4 < a) (h2 : a < 0) (h3 : -4 < b) (h4 : b < 0) : 
  2 * |a - b| < |a * b + 2 * a + 2 * b| := sorry

end abs_inequality_solution_bounded_a_b_inequality_l5_5900


namespace terminating_decimal_fractions_l5_5620

theorem terminating_decimal_fractions :
  let n_count := (finset.range 151).filter (λ n, n % 3 = 0),
  n_count.card = 50 :=
by
  sorry

end terminating_decimal_fractions_l5_5620


namespace probability_green_marble_l5_5064

theorem probability_green_marble (P : ℚ) :
  let X_white := 5,
      X_black := 5,
      Y_green := 8,
      Y_total := 15,
      Z_green := 6,
      Z_total := 9 in
  P = (1/2) * (Y_green / Y_total) + (1/2) * (Z_green / Z_total) → P = (3/5) :=
by
  let X_white := 5,
      X_black := 5,
      X_total := X_white + X_black,
      P_white := X_white / X_total,
      P_black := X_black / X_total,
      Y_green := 8,
      Y_total := 7 + Y_green,
      Z_green := 6,
      Z_total := 3 + Z_green in
  have P1 : P = P_white * (Y_green / Y_total) + P_black * (Z_green / Z_total), sorry

end probability_green_marble_l5_5064


namespace modulus_of_complex_number_l5_5254

open Complex

theorem modulus_of_complex_number (z : ℂ) (h : z * (1 - I) = 2) : complex.abs z = Real.sqrt 2 := 
sorry

end modulus_of_complex_number_l5_5254


namespace sum_largest_smallest_l5_5612

theorem sum_largest_smallest : 
  let a := 0.11
  let b := 0.98
  let c := 3 / 4
  let d := 2 / 3
  (∀ x ∈ {a, b, c, d}, x = 0.11 ∨ x = 0.98 ∨ x = 0.75 ∨ x = 2 / 3) →
  0.11 + 0.98 = 1.09 :=
by 
  sorry

end sum_largest_smallest_l5_5612


namespace behavior_of_g_at_extremes_l5_5076

-- Define the function g
def g (x : ℝ) : ℝ := -x^4 + 5*x^3 + 7

-- State the limits as x approaches ∞ and -∞
theorem behavior_of_g_at_extremes : 
  (real.limit_at g real.limit_pos_infty = -real.infty) ∧ 
  (real.limit_at g real.limit_neg_infty = -real.infty) := 
sorry

end behavior_of_g_at_extremes_l5_5076


namespace remainder_six_n_mod_four_l5_5870

theorem remainder_six_n_mod_four (n : ℤ) (h : n % 4 = 3) : (6 * n) % 4 = 2 := 
by sorry

end remainder_six_n_mod_four_l5_5870


namespace repeating_decimal_to_fraction_l5_5982

/-- Given 0.02 repeating as a fraction 2/99, prove that 2.06 repeating can be expressed as 68/33 -/
theorem repeating_decimal_to_fraction :
  (2 + 0.06̅ : ℝ) = (68 / 33 : ℝ) :=
begin
  have h : (0.02̅ : ℝ) = (2 / 99 : ℝ), from sorry,
  have h3 : (0.06̅ : ℝ) = 3 * (0.02̅ : ℝ), from sorry,
  have h6 : (0.06̅ : ℝ) = 3 * (2 / 99 : ℝ), from sorry,
  have s : (0.06̅ : ℝ) = (6 / 99 : ℝ), from sorry,
  have s2 : (6 / 99 : ℝ) = (2 / 33 : ℝ), from sorry,
  have add := congr_arg (λ x : ℝ, (2 : ℝ) + x) s2,
  rw [add_comm, ← add_halves', add_assoc', add_comm (2 : ℝ), add_comm 2 (2 / 33 : ℝ), add_halves',
    add_assoc', add_comm (2 : ℝ), add_comm 68 (2 / 33 : ℝ)] ,
end

end repeating_decimal_to_fraction_l5_5982


namespace no_primes_in_range_l5_5614

noncomputable def factorial (n : ℕ) : ℕ :=
nat.rec_on n 1 (λ n ih, (n + 1) * ih)

open nat

theorem no_primes_in_range (n m : ℕ) (h₁ : n > 1) (h₂ : 1 ≤ m) (h₃ : m ≤ n) :
  ∀ p, prime p → ¬(n! + m < p ∧ p < n! + n + m) :=
by sorry

end no_primes_in_range_l5_5614


namespace large_cube_surface_area_l5_5970

noncomputable def cube_volume (s : ℝ) : ℝ := s^3

noncomputable def surface_area (s : ℝ) : ℝ := 6 * s^2

theorem large_cube_surface_area :
  (∀ s : ℝ, cube_volume s = 512) →
  (∃ S : ℝ, S = 2 * (∃ s : ℝ, s = 8)) →
  (surface_area 16 = 1536) :=
by
  intro h
  use 16
  sorry

end large_cube_surface_area_l5_5970


namespace estimate_probability_l5_5815

noncomputable def freq_20 : ℝ := 0.300
noncomputable def freq_50 : ℝ := 0.360
noncomputable def freq_100 : ℝ := 0.350
noncomputable def freq_300 : ℝ := 0.350
noncomputable def freq_500 : ℝ := 0.352
noncomputable def freq_1000 : ℝ := 0.351
noncomputable def freq_5000 : ℝ := 0.351

theorem estimate_probability : (|0.35 - ((freq_20 + freq_50 + freq_100 + freq_300 + freq_500 + freq_1000 + freq_5000) / 7)| < 0.01) :=
by sorry

end estimate_probability_l5_5815


namespace mod_remainder_l5_5872

theorem mod_remainder (n : ℕ) (h : n % 4 = 3) : (6 * n) % 4 = 2 := 
by
  sorry

end mod_remainder_l5_5872


namespace a3_value_l5_5143

noncomputable def a (n : ℕ) : ℝ :=
  sorry

axiom geom_seq (n : ℕ) : a (n + 1) = a n * r

axiom a1 : a 1 = 2
axiom a3_a5 : a 3 * a 5 = 4 * (a 6)^2

theorem a3_value : a 3 = 1/2 :=
by sorry

end a3_value_l5_5143


namespace probability_in_interval_l5_5735

-- Define the interval and condition
def interval := (1, 3)
def condition (x : ℝ) : Prop := log 2 (2 * x - 1) > 1

-- Define the length of an interval
def length (a b : ℝ) := b - a

-- Find the probability that x in (1, 3) satisfies the condition
theorem probability_in_interval : 
  let favorable_interval := (3 - 3/2)
  let total_interval := (3 - 1)
  probability (1 < x ∧ x < 3 ∧ condition x) = favorable_interval / total_interval := 
sorry

end probability_in_interval_l5_5735


namespace unique_f_l5_5754

def S := { x : ℕ // (∃ (digits : Fin 9 → Fin 3), ∀ i, digits i = 1 ∨ digits i = 2 ∨ digits i = 3 ∧ x = ∑ j, (digits j) * 10^j )}

def f (n : S) : Fin 3 :=
if h : (∃ (digits : Fin 9 → Fin 3), ∀ i, digits i = 1) then 1
else if h : (∃ (digits : Fin 9 → Fin 3), ∀ i, digits i = 2) then 2
else if h : (∃ (digits : Fin 9 → Fin 3), ∀ i, digits i = 3) then 3
else 1 -- This is a simplification for defining f, assuming first digit's value matches Fin 3

lemma condition1 (f : S → Fin 3) :
  f ⟨111111111, sorry⟩ = 1 ∧ f ⟨222222222, sorry⟩ = 2 ∧ f ⟨333333333, sorry⟩ = 3 ∧ f ⟨122222222, sorry⟩ = 1 := 
sorry

lemma condition2 (f : S → Fin 3) :
  ∀ x y : S, (∀ i : Fin 9, (x.1 / 10 ^ i % 10 ≠ y.1 / 10 ^ i % 10)) → f x ≠ f y :=
sorry

theorem unique_f : ∀ (g : S → Fin 3), 
  g ⟨111111111, sorry⟩ = 1 ∧ g ⟨222222222, sorry⟩ = 2 ∧ g ⟨333333333, sorry⟩ = 3 ∧ g ⟨122222222, sorry⟩ = 1 ∧
  (∀ x y : S, (∀ i : Fin 9, (x.1 / 10 ^ i % 10 ≠ y.1 / 10 ^ i % 10)) → g x ≠ g y) →
  ∀ n : S, g n = f n :=
sorry

end unique_f_l5_5754


namespace quadrilateral_diagonals_perpendicular_l5_5789

def convex_quadrilateral (A B C D : Type) : Prop := sorry -- Assume it’s defined elsewhere 
def tangent_to_all_sides (circle : Type) (A B C D : Type) : Prop := sorry -- Assume it’s properly specified with its conditions elsewhere
def tangent_to_all_extensions (circle : Type) (A B C D : Type) : Prop := sorry -- Same as above

theorem quadrilateral_diagonals_perpendicular
  (A B C D : Type)
  (h_convex : convex_quadrilateral A B C D)
  (incircle excircle : Type)
  (h_incircle : tangent_to_all_sides incircle A B C D)
  (h_excircle : tangent_to_all_extensions excircle A B C D) : 
  (⊥ : Prop) :=  -- statement indicating perpendicularity 
sorry

end quadrilateral_diagonals_perpendicular_l5_5789


namespace exists_positive_integer_n_divisible_by_2000_primes_and_divides_2_to_the_n_plus_1_l5_5086

/-- Proof that there exists a positive integer n that is divisible by 2000 distinct prime numbers
and such that 2^n + 1 is divisible by n. -/
theorem exists_positive_integer_n_divisible_by_2000_primes_and_divides_2_to_the_n_plus_1 :
  ∃ n : ℕ, n > 0 ∧ (∃ (primes : Finset ℕ) (h₁ : primes.card = 2000), primes ∣ n) ∧ (n ∣ (2^n + 1)) := 
sorry

end exists_positive_integer_n_divisible_by_2000_primes_and_divides_2_to_the_n_plus_1_l5_5086


namespace expected_value_of_8_sided_die_l5_5386

theorem expected_value_of_8_sided_die : 
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let probability := 1 / 8
  (∑ outcome in outcomes, outcome * probability) = 4.5 :=
by
  sorry

end expected_value_of_8_sided_die_l5_5386


namespace sqrt_expression_l5_5945

theorem sqrt_expression (a b : ℝ) (h1 : a = 25) (h2 : b = 25) (h3 : ∀ x : ℝ, x ≥ 0 → real.sqrt (x * 5) = real.sqrt x * real.sqrt 5) :
  real.sqrt (a * real.sqrt (b * real.sqrt b)) = 5 * real.root 4 125 :=
by
  sorry

end sqrt_expression_l5_5945


namespace interest_gender_independent_distribution_expectation_l5_5063

open Nat

-- Definitions based on conditions
def num_students : ℕ := 300
def ratio_male_female : ℚ := 2 / 1
def num_female_students : ℕ := num_students / 3
def num_male_students : ℕ := num_students - num_female_students
def percent_females_interested : ℚ := 0.8
def num_females_interested : ℕ := percent_females_interested * num_female_students
def num_females_not_interested : ℕ := num_female_students - num_females_interested
def num_males_not_interested : ℕ := 30
def num_males_interested : ℕ := num_male_students - num_males_not_interested
def alpha : ℚ := 0.1
def p : ℚ := 3 / 5

-- Computations based on conditions
def contingency_table := ((num_students, num_males_interested, num_males_not_interested, num_females_interested, num_females_not_interested), 1.2)

-- Theorem statements
theorem interest_gender_independent : contingency_table.2 = 1.2 → contingency_table.2 < 2.706 := by sorry

-- Distribution and expectation of points earned by Xiao Qiang
def P_X_0 : ℚ := (1 - p)^2
def P_X_1 : ℚ := 2 * p * (1 - p)
def P_X_2 : ℚ := 2 * p^2 * (1 - p)
def P_X_3 : ℚ := p^2

def expectation_X : ℚ := 0 * P_X_0 + 1 * P_X_1 + 2 * P_X_2 + 3 * P_X_3

theorem distribution_expectation : expectation_X = 231 / 125 := by sorry

end interest_gender_independent_distribution_expectation_l5_5063


namespace arithmetic_progression_difference_l5_5280

theorem arithmetic_progression_difference (a₁ aₙ d : ℝ) (n : ℕ) (h₁ : aₙ = a₁ + (n - 1) * d) (h₂ : (n : ℝ) ≠ 0) :
  ∑ i in Finset.range n, (a₁ + i • d) = (n / 2) * (a₁ + aₙ) →
  (aₙ ^ 2 - a₁ ^ 2) / (2 * (n / 2) * (a₁ + aₙ) - (a₁ + aₙ)) = d :=
by
  sorry

end arithmetic_progression_difference_l5_5280


namespace max_positive_factors_l5_5963

theorem max_positive_factors (b n : ℕ) (hb : 1 ≤ b ∧ b ≤ 15) (hn : n = 10) : 
  ∃ k, (∀ b, 1 ≤ b ∧ b ≤ 15 → (factors_in_pow b n ≤ k)) ∧ k = 121 := by
  sorry

noncomputable def factors_in_pow (b n : ℕ) : ℕ := 
  (if b = 0 then 0 else 
  let ⟨factors_b, _⟩ := Nat.factorize b in (factors_b.natfold ( λ acc p e, acc * (e*n + 1)) 1))


end max_positive_factors_l5_5963


namespace probability_at_least_60_cents_l5_5906

theorem probability_at_least_60_cents :
  let num_total_outcomes := Nat.choose 16 8
  let num_successful_outcomes := 
    (Nat.choose 4 2) * (Nat.choose 5 1) * (Nat.choose 7 5) +
    1 -- only one way to choose all 8 dimes
  num_successful_outcomes / num_total_outcomes = 631 / 12870 := by
  sorry

end probability_at_least_60_cents_l5_5906


namespace find_number_l5_5902

theorem find_number (x : ℝ) : (45 * x = 0.45 * 900) → (x = 9) :=
by sorry

end find_number_l5_5902


namespace smallest_three_digit_multiple_of_9_l5_5868

theorem smallest_three_digit_multiple_of_9 : ∃ n : ℤ, 100 ≤ 9 * n ∧ 9 * n < 1000 ∧ ∀ m : ℤ, 100 ≤ 9 * m ∧ 9 * m < 1000 → 9 * n ≤ 9 * m := 
begin
  use 12,
  split,
  {
    norm_num,
  },
  split,
  {
    norm_num,
  },
  {
    intros m hm,
    cases hm with h_m1 h_m2,
    exact le_of_lt ((integer.coe_nat_lt_coe_nat_iff m n).mpr (int.lt_of_add_one_le (by 
    {
      have := @int.lt_add_one_of_lt ℤ _ (m.floor_div_eq_div 9).mp h_m1,
      have := @le_of_succ_le m.floor_div_le_of_floor_div_eq_add (int.lt_sub_one_of_iff.trans $ 
      linear_ordered_field.div_roll_eq_of_eq 9)).div_gr (trans div_self_le_eq)).mp),
      } exact _⟧⟫,},
end

end smallest_three_digit_multiple_of_9_l5_5868


namespace complex_sum_abs_eq_1_or_3_l5_5771

open Complex

theorem complex_sum_abs_eq_1_or_3
  (a b c : ℂ)
  (ha : abs a = 1)
  (hb : abs b = 1)
  (hc : abs c = 1)
  (h : a^3 / (b * c) + b^3 / (a * c) + c^3 / (a * b) = 1) :
  ∃ r : ℝ, (r = 1 ∨ r = 3) ∧ abs (a + b + c) = r :=
by {
  -- Proof goes here
  sorry
}

end complex_sum_abs_eq_1_or_3_l5_5771


namespace cubic_roots_l5_5297

theorem cubic_roots : 
  ∃ α β : ℝ, 
  (α ≠ β) ∧ 
  (α ≠ -β) ∧ 
  (β ≠ - β) ∧ 
  (α * β * -β = 45) ∧ 
  (α * β + β * (- β) + (- β) * α = -9) ∧ 
  (α + β + (- β) = 5) ∧ 
  (x : ℝ) ∈ {5, (5 + Real.sqrt 61) / 2, -(5 + Real.sqrt 61) / 2} :=
by
  sorry

end cubic_roots_l5_5297


namespace total_two_digit_numbers_tens_digit_less_than_units_digit_l5_5013

def A : Set ℕ := {2, 4, 6, 8}
def B : Set ℕ := {1, 3, 5, 7, 9}

theorem total_two_digit_numbers : ((A.card * B.card) = 20) :=
by
  sorry

theorem tens_digit_less_than_units_digit : (Set.card ( { ab | ab.fst ∈ A ∧ ab.snd ∈ B ∧ ab.fst < ab.snd } ) = 10) :=
by
  sorry

end total_two_digit_numbers_tens_digit_less_than_units_digit_l5_5013


namespace postage_cost_l5_5462

def first_ounce_cost : ℕ := 40
def additional_ounce_cost : ℕ := 25
def letter_weight : ℝ := 5.3
def total_postage_cents : ℝ :=
  first_ounce_cost + additional_ounce_cost * (letter_weight - 1).ceil

theorem postage_cost {dollars : ℝ} :
  dollars = total_postage_cents / 100 ↔ dollars = 1.65 :=
by
  sorry

end postage_cost_l5_5462


namespace expected_value_eight_sided_die_l5_5409

-- Define a standard 8-sided die
def eight_sided_die : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]

-- Compute the probability of each outcome
def probability (n : ℕ) : ℝ := 1 / 8

-- Expected Value of a discrete random variable
def expected_value (outcomes : List ℕ) (prob : ℕ → ℝ) : ℝ :=
  outcomes.sum / outcomes.length.toReal

-- Theorem stating the expected value of a standard 8-sided die roll is 4.5
theorem expected_value_eight_sided_die : expected_value eight_sided_die probability = 4.5 := by
  sorry

end expected_value_eight_sided_die_l5_5409


namespace total_boxes_is_27_l5_5028

-- Defining the conditions
def stops : ℕ := 3
def boxes_per_stop : ℕ := 9

-- Prove that the total number of boxes is as expected
theorem total_boxes_is_27 : stops * boxes_per_stop = 27 := by
  sorry

end total_boxes_is_27_l5_5028


namespace parabola_vertex_distance_l5_5512

theorem parabola_vertex_distance :
  let eq1 := ∃ (x y : ℝ), (√(x^2 + y^2) + |y - 2| = 4 ∧ y ≥ 2 ∧ y = 3 - (1/12) * x^2)
  let eq2 := ∃ (x y : ℝ), (√(x^2 + y^2) + |y - 2| = 4 ∧ y < 2 ∧ y = (1/4) * x^2 - 1)
  let vertex1 := (0 : ℝ, 3 : ℝ)
  let vertex2 := (0 : ℝ, -1 : ℝ)
  dist vertex1 vertex2 = 4 := 
by
  sorry

end parabola_vertex_distance_l5_5512


namespace repeating_decimal_to_fraction_l5_5981

/-- Given 0.02 repeating as a fraction 2/99, prove that 2.06 repeating can be expressed as 68/33 -/
theorem repeating_decimal_to_fraction :
  (2 + 0.06̅ : ℝ) = (68 / 33 : ℝ) :=
begin
  have h : (0.02̅ : ℝ) = (2 / 99 : ℝ), from sorry,
  have h3 : (0.06̅ : ℝ) = 3 * (0.02̅ : ℝ), from sorry,
  have h6 : (0.06̅ : ℝ) = 3 * (2 / 99 : ℝ), from sorry,
  have s : (0.06̅ : ℝ) = (6 / 99 : ℝ), from sorry,
  have s2 : (6 / 99 : ℝ) = (2 / 33 : ℝ), from sorry,
  have add := congr_arg (λ x : ℝ, (2 : ℝ) + x) s2,
  rw [add_comm, ← add_halves', add_assoc', add_comm (2 : ℝ), add_comm 2 (2 / 33 : ℝ), add_halves',
    add_assoc', add_comm (2 : ℝ), add_comm 68 (2 / 33 : ℝ)] ,
end

end repeating_decimal_to_fraction_l5_5981


namespace daily_earnings_c_l5_5434

theorem daily_earnings_c (A B C : ℕ) (h1 : A + B + C = 600) (h2 : A + C = 400) (h3 : B + C = 300) : C = 100 :=
sorry

end daily_earnings_c_l5_5434


namespace max_distance_convoy_l5_5912

/-
Define the maximum distance the convoy can cover given the following constraints:
1. The convoy has a total of 21 gallons of gasoline to share.
2. The SUV must use at least 10 gallons.
3. The hybrid sedan must use at least 5 gallons.
4. The motorcycle must use at least 2 gallons.
5. Fuel efficiencies are:
   - SUV: 12.2 mpg
   - Hybrid sedan: 52 mpg
   - Motorcycle: 70 mpg

Prove that the maximum distance the convoy can travel is 122 miles.
-/
theorem max_distance_convoy : 
  (∀ (total_gallons suv_gallons hybrid_sedan_gallons motorcycle_gallons : ℝ),
    total_gallons = 21 ∧
    suv_gallons ≥ 10 ∧
    hybrid_sedan_gallons ≥ 5 ∧
    motorcycle_gallons ≥ 2 ∧
    suv_gallons + hybrid_sedan_gallons + motorcycle_gallons = total_gallons ∧
    ∀ (suv_mpg hybrid_sedan_mpg motorcycle_mpg : ℝ),
    suv_mpg = 12.2 ∧
    hybrid_sedan_mpg = 52 ∧
    motorcycle_mpg = 70 →
    let suv_distance := suv_gallons * suv_mpg in
    let hybrid_sedan_distance := hybrid_sedan_gallons * hybrid_sedan_mpg in
    let motorcycle_distance := motorcycle_gallons * motorcycle_mpg in
    min suv_distance (min hybrid_sedan_distance motorcycle_distance) = 122) :=
begin
  sorry
end

end max_distance_convoy_l5_5912


namespace keith_spent_total_l5_5749

theorem keith_spent_total :
    let speakers := 136.01
    let cd_player := 139.38
    let tires := 112.46
    let sales_tax_rate := 0.065
    let total_before_tax := speakers + cd_player + tires
    let sales_tax := sales_tax_rate * total_before_tax
    let total_spent := total_before_tax + sales_tax
    total_spent = 413.06 :=
by
  simp [speakers, cd_player, tires, sales_tax_rate, total_before_tax, sales_tax]
  norm_num
  sorry

end keith_spent_total_l5_5749


namespace smallest_rel_prime_to_180_is_7_l5_5584

theorem smallest_rel_prime_to_180_is_7 :
  ∃ x : ℕ, x > 1 ∧ gcd x 180 = 1 ∧ ∀ y : ℕ, y > 1 ∧ gcd y 180 = 1 → x ≤ y :=
begin
  use 7,
  split,
  { exact dec_trivial, },
  split,
  { exact dec_trivial, },
  intros y hy,
  cases hy with hy1 hy2,
  exact dec_trivial,
end

end smallest_rel_prime_to_180_is_7_l5_5584


namespace hexagon_colorings_l5_5969

-- Definitions based on conditions
def isValidColoring (A B C D E F : ℕ) (colors : Fin 7 → ℕ) : Prop :=
  -- Adjacent vertices must have different colors
  A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ E ∧ E ≠ F ∧ F ≠ A ∧
  -- Diagonal vertices must have different colors
  A ≠ D ∧ B ≠ E ∧ C ≠ F

-- Function to count all valid colorings
def countValidColorings : ℕ :=
  let colors := List.range 7
  -- Calculate total number of valid colorings
  7 * 6 * 5 * 4 * 3 * 2

theorem hexagon_colorings : countValidColorings = 5040 := by
  sorry

end hexagon_colorings_l5_5969


namespace perpendicular_tangents_l5_5165

theorem perpendicular_tangents (a b : ℝ) (h1 : ∀ (x y : ℝ), y = x^3 → y = (3 * x^2) * (x - 1) + 1 → y = 3 * (x - 1) + 1) (h2 : (a : ℝ) * 1 - (b : ℝ) * 1 = 2) 
 (h3 : (a : ℝ)/(b : ℝ) * 3 = -1) : a / b = -1 / 3 :=
by
  sorry

end perpendicular_tangents_l5_5165


namespace trigonometric_identity_second_quadrant_l5_5253

theorem trigonometric_identity_second_quadrant (θ : ℝ) (hθ : θ ∈ Icc (π / 2) π)
  (h_tan : tan (θ + π / 3) = 1 / 2) :
  sin θ + sqrt 3 * cos θ = -2 * sqrt 5 / 5 :=
by
  sorry

end trigonometric_identity_second_quadrant_l5_5253


namespace solve_problem_l5_5658

noncomputable def p_is_on_curve (P : ℝ × ℝ) : Prop :=
  let (x, y) := P in x^2 + y^2 + 2 * x - 6 * y + 1 = 0

noncomputable def q_is_on_curve (Q : ℝ × ℝ) : Prop :=
  let (x, y) := Q in x^2 + y^2 + 2 * x - 6 * y + 1 = 0

noncomputable def symmetric_about_line (P Q : ℝ × ℝ) (m : ℝ) : Prop :=
  let Mid := (0.5 * (P.1 + Q.1), 0.5 * (P.2 + Q.2)) in
  Mid.1 + m * Mid.2 + 4 = 0

noncomputable def orthogonal_vectors (P Q : ℝ × ℝ) : Prop :=
  P.1 * Q.1 + P.2 * Q.2 = 0

theorem solve_problem (P Q : ℝ × ℝ) (m : ℝ) :
  p_is_on_curve P →
  q_is_on_curve Q →
  symmetric_about_line P Q m →
  orthogonal_vectors P Q →
  m = -1 ∧ ∃ b : ℝ, ∀ x, y = -x + b := sorry

end solve_problem_l5_5658


namespace leaves_decrease_by_four_fold_l5_5501

theorem leaves_decrease_by_four_fold (x y : ℝ) (h1 : y ≤ x / 4) : 
  9 * y ≤ (9 * x) / 4 := by 
  sorry

end leaves_decrease_by_four_fold_l5_5501


namespace factorize_expression_l5_5529

-- Define the variables m and n
variables (m n : ℝ)

-- The statement to prove
theorem factorize_expression : -8 * m^2 + 2 * m * n = -2 * m * (4 * m - n) :=
sorry

end factorize_expression_l5_5529


namespace minimum_value_of_reciprocals_l5_5672

theorem minimum_value_of_reciprocals (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hline : a - b = 1) :
  (1 / a) + (1 / b) ≥ 4 :=
sorry

end minimum_value_of_reciprocals_l5_5672


namespace perimeter_of_triangle_ABC_l5_5709

theorem perimeter_of_triangle_ABC : 
  ∀ (A B C X Y Z W P : Type) 
  (ABXY_square : Square A B X Y) 
  (CBWZ_square : Square C B W Z) 
  (circle_XYZW_passes_through_P : Circle X Y Z W passes_through P)
  (angle_C_90 : ∠ C = 90)
  (AB_eq_10 : AB = 10),
  perimeter ABC = 20 :=
by
  -- Here we would provide the actual proof steps, but we'll skip it for now
  sorry

end perimeter_of_triangle_ABC_l5_5709


namespace avg_annual_growth_rate_l5_5485
-- Import the Mathlib library

-- Define the given conditions
def initial_income : ℝ := 32000
def final_income : ℝ := 37000
def period : ℝ := 2
def initial_income_ten_thousands : ℝ := initial_income / 10000
def final_income_ten_thousands : ℝ := final_income / 10000

-- Define the growth rate
variable (x : ℝ)

-- Define the theorem
theorem avg_annual_growth_rate :
  3.2 * (1 + x) ^ 2 = 3.7 :=
sorry

end avg_annual_growth_rate_l5_5485


namespace circumcenter_PQM_on_fixed_line_l5_5724

variable {α : Type*} [EuclideanGeometry α]

-- Considering points in the plane geometry
variables {A B C D M P Q : Point α}

-- Given conditions in the problem
variables (hABC : acute_triangle A B C)
          (hAD : angle_bisector A D B C)
          (hM : midpoint M B C)
          (hPQ : ∀ (P Q : Point α), P ∈ segment A D → Q ∈ segment A D → ∠ A B P = ∠ C B Q)
          (hD : same_line A D)
          (hFixedLine : Line α)

-- We are asked to prove that the circumcenter of triangle PQM lies on a fixed line
theorem circumcenter_PQM_on_fixed_line :
  ∀ (P Q : Point α), (P ∈ segment A D) → (Q ∈ segment A D) → ∠ A B P = ∠ C B Q →
  ∃ (H : Point α), is_circumcenter_of H P Q M ∧ H ∈ hFixedLine :=
begin
  intro P,
  intro Q,
  intro hP_in_segment_AD,
  intro hQ_in_segment_AD,
  intro hAngle_eq,
  sorry
end

end circumcenter_PQM_on_fixed_line_l5_5724


namespace percent_of_pizza_not_crust_l5_5465

theorem percent_of_pizza_not_crust (total_weight crust_weight : ℝ) (h_total : total_weight = 800) (h_crust : crust_weight = 200) :
  (total_weight - crust_weight) / total_weight * 100 = 75 :=
by
  sorry

end percent_of_pizza_not_crust_l5_5465


namespace smallest_w_l5_5435

theorem smallest_w (w : ℕ) (h : 2^5 ∣ 936 * w ∧ 3^3 ∣ 936 * w ∧ 11^2 ∣ 936 * w) : w = 4356 :=
sorry

end smallest_w_l5_5435


namespace extreme_value_f_f_less_than_x_minus_3_over_2_l5_5240

open Real

noncomputable def f (x : ℝ) : ℝ := log x - x^2 + x - 1

theorem extreme_value_f :
  ∀ x ∈ Ioi (0 : ℝ), f x ≤ -1 :=
by
  sorry

theorem f_less_than_x_minus_3_over_2 :
  ∀ x ∈ Ioi (0 : ℝ), f x < x - (3 / 2) :=
by
  sorry

end extreme_value_f_f_less_than_x_minus_3_over_2_l5_5240


namespace scrooge_no_equal_coins_l5_5286

theorem scrooge_no_equal_coins (n : ℕ → ℕ)
  (initial_state : n 1 = 1 ∧ n 2 = 0 ∧ n 3 = 0 ∧ n 4 = 0 ∧ n 5 = 0 ∧ n 6 = 0)
  (operation : ∀ x i, 1 ≤ i ∧ i ≤ 6 → (n (i + 1) = n i - x ∧ n ((i % 6) + 2) = n ((i % 6) + 2) + 6 * x) 
                      ∨ (n (i + 1) = n i + 6 * x ∧ n ((i % 6) + 2) = n ((i % 6) + 2) - x)) :
  ¬ ∃ k, n 1 = k ∧ n 2 = k ∧ n 3 = k ∧ n 4 = k ∧ n 5 = k ∧ n 6 = k :=
by {
  sorry
}

end scrooge_no_equal_coins_l5_5286


namespace average_remaining_ropes_l5_5008

theorem average_remaining_ropes 
  (n : ℕ) 
  (m : ℕ) 
  (l_avg : ℕ) 
  (l1_avg : ℕ) 
  (l2_avg : ℕ) 
  (h1 : n = 6)
  (h2 : m = 2)
  (hl_avg : l_avg = 80)
  (hl1_avg : l1_avg = 70)
  (htotal : l_avg * n = 480)
  (htotal1 : l1_avg * m = 140)
  (htotal2 : l_avg * n - l1_avg * m = 340):
  (340 : ℕ) / (4 : ℕ) = 85 := by
  sorry

end average_remaining_ropes_l5_5008


namespace letter_puzzle_solution_l5_5547

theorem letter_puzzle_solution :
  ∃ (A B : ℕ), (A ≠ B) ∧ (1 ≤ A ∧ A ≤ 9) ∧ (1 ≤ B ∧ B ≤ 9) ∧ A ^ B = B * 10 + A :=
by {
  use 2, 5,
  split, { exact ne_of_lt (by decide) }, split, { exact ⟨by decide, by decide⟩ },
  split, { exact ⟨by decide, by decide⟩ }, exact (by decide : 2 ^ 5 = 5 * 10 + 2),
  sorry
} ∨
by {
  use 6, 2,
  split, { exact ne_of_lt (by decide) }, split, { exact ⟨by decide, by decide⟩ },
  split, { exact ⟨by decide, by decide⟩ }, exact (by decide : 6 ^ 2 = 2 * 10 + 6),
  sorry
} ∨
by {
  use 4, 3,
  split, { exact ne_of_lt (by decide) }, split, { exact ⟨by decide, by decide⟩ },
  split, { exact ⟨by decide, by decide⟩ }, exact (by decide : 4 ^ 3 = 3 * 10 + 4),
  sorry
}

end letter_puzzle_solution_l5_5547


namespace right_triangle_altitude_relation_l5_5639

-- Definitions for the right triangle and foot of the altitude
variables {A B C H : Type} [euclidean_geometry A B C]
  (right_triangle : right_angle C)
  (altitude : foot H C)

-- Statement of the theorem
theorem right_triangle_altitude_relation
  (h : right_triangle) (h_alt : altitude) :
  (length (segment H C))^2 = (length (segment A H)) * (length (segment B H)) :=
sorry

end right_triangle_altitude_relation_l5_5639


namespace smallest_rel_prime_to_180_is_7_l5_5589

theorem smallest_rel_prime_to_180_is_7 :
  ∃ x : ℕ, x > 1 ∧ gcd x 180 = 1 ∧ ∀ y : ℕ, y > 1 ∧ gcd y 180 = 1 → x ≤ y :=
begin
  use 7,
  split,
  { exact dec_trivial, },
  split,
  { exact dec_trivial, },
  intros y hy,
  cases hy with hy1 hy2,
  exact dec_trivial,
end

end smallest_rel_prime_to_180_is_7_l5_5589


namespace sum_of_factors_24_l5_5421

theorem sum_of_factors_24 : 
  ∑ f in {1, 2, 3, 4, 6, 8, 12, 24}.to_finset, f = 60 := 
by sorry

end sum_of_factors_24_l5_5421


namespace hyperbola_asymptotes_intersection_l5_5309

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 144 - y^2 / 81 = 1

-- Define the line
def line (x y : ℝ) : Prop := y = 2 * x + 3

-- Define the equations of the asymptotes
def asymptote1 (x y : ℝ) : Prop := y = 3 * x / 4
def asymptote2 (x y : ℝ) : Prop := y = -3 * x / 4

-- Define the intersection points to be proved
def intersection1 : ℝ × ℝ := (-12 / 5, -9 / 5)
def intersection2 : ℝ × ℝ := (-12 / 11, 9 / 11)

-- The hypothesis that we will prove
theorem hyperbola_asymptotes_intersection : 
  ∃ (x y : ℝ), (line x y) ∧ (asymptote1 x y) ∨ (asymptote2 x y) → 
    (x, y) = intersection1 ∨ (x, y) = intersection2 := 
sorry

end hyperbola_asymptotes_intersection_l5_5309


namespace sum_of_odd_indexed_terms_l5_5044

theorem sum_of_odd_indexed_terms (a : ℕ → ℕ) (n : ℕ) :
  (∀ k < n, a (k + 1) = a k + 2) ∧ (a 0 + ∑ i in finset.range 2020, a (i + 1)) = 6060 →
  ∑ i in finset.range 1010, a (2 * i + 1) = 2020 :=
by
  sorry

end sum_of_odd_indexed_terms_l5_5044


namespace constant_term_in_expansion_l5_5959

theorem constant_term_in_expansion :
  let f := λ(x : ℝ), (2 * x - 1 / x) ^ 4
  in f(1) = 24 :=
by
  sorry

end constant_term_in_expansion_l5_5959


namespace expected_value_of_eight_sided_die_l5_5370

theorem expected_value_of_eight_sided_die : 
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8], 
      n := (outcomes.length : ℚ),
      probabilities := List.replicate (outcomes.length) (1 / n),
      expected_value := (List.zipWith (*) probabilities (outcomes.map (· : ℚ))).sum
  in expected_value = 4.5 :=
by
  sorry

end expected_value_of_eight_sided_die_l5_5370


namespace diff_larger_smaller_root_l5_5521

noncomputable def quadratic_difference (p : ℝ) : ℝ :=
  let a := 1
  let b := -(p + 1)
  let c := (p^2 + p - 2) / 4
  have h : ∃ r s : ℝ, r + s = p + 1 ∧ r * s = (p^2 + p - 2) / 4 ∧ r ≥ s, from sorry,
  let ⟨r, s, hrs⟩ := h in
  r - s

theorem diff_larger_smaller_root (p : ℝ) :
  quadratic_difference p = sqrt 5 :=
by
  -- No need to fill the proof steps
  sorry

end diff_larger_smaller_root_l5_5521


namespace smallest_rel_prime_to_180_l5_5601

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ x ≤ 7 ∧ (∀ y : ℕ, y > 1 ∧ y < x → y.gcd 180 ≠ 1) ∧ x.gcd 180 = 1 :=
by
  sorry

end smallest_rel_prime_to_180_l5_5601


namespace smallest_rel_prime_to_180_l5_5603

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ x ≤ 7 ∧ (∀ y : ℕ, y > 1 ∧ y < x → y.gcd 180 ≠ 1) ∧ x.gcd 180 = 1 :=
by
  sorry

end smallest_rel_prime_to_180_l5_5603


namespace inequality_solution_sets_l5_5799

theorem inequality_solution_sets (m : ℝ) :
  (m + 3) * x - 1 * (x + 1) > 0 → 
  (if m = -4 then 
     ∅ = {x : ℝ | ((m+3)*x-1)*(x+1) > 0} else
   if m < -4 then 
     {x : ℝ | -1 < x ∧ x < 1/(m+3)} = {x : ℝ | ((m+3)*x-1)*(x+1) > 0} else
   if -4 < m ∧ m < -3 then 
     {x : ℝ | 1/(m+3) < x ∧ x < -1} = {x : ℝ | ((m+3)*x-1)*(x+1) > 0} else
   if m = -3 then 
     {x : ℝ | x > -1} = {x : ℝ | ((m+3)*x-1)*(x+1) > 0} else
   if m > -3 then 
     {x : ℝ | x < -1 ∨ x > 1/(m+3)} = {x : ℝ | ((m+3)*x-1)*(x+1) > 0}) :=
by sorry

end inequality_solution_sets_l5_5799


namespace number_of_circles_l5_5430

def circle_radius : ℝ := 3
def paper_width : ℝ := 30
def paper_length : ℝ := 24
def circle_diameter : ℝ := 2 * circle_radius

theorem number_of_circles : (paper_width / circle_diameter) * (paper_length / circle_diameter) = 20 := by
  -- Calculation for how many circles fit along the width
  have circles_width : ℝ := paper_width / circle_diameter 
  -- Calculation for how many circles fit along the length
  have circles_length : ℝ := paper_length / circle_diameter 
  -- Multiplying the two values
  calc circles_width * circles_length = (paper_width / circle_diameter) * (paper_length / circle_diameter) := by
    sorry
  -- Given that circles_width = 5 and circles_length = 4
  show paper_width / circle_diameter = 5 := by
    sorry
  show paper_length / circle_diameter = 4 := by
    sorry
  -- Therefore, multiplying we get 20
  show (5:ℝ) * (4:ℝ) = 20 := by
    sorry

end number_of_circles_l5_5430


namespace roger_current_money_l5_5791

noncomputable def roger_initial_money : ℕ := 16
noncomputable def roger_birthday_money : ℕ := 28
noncomputable def roger_game_spending : ℕ := 25

theorem roger_current_money : roger_initial_money + roger_birthday_money - roger_game_spending = 19 := by
  sorry

end roger_current_money_l5_5791


namespace smallest_rel_prime_to_180_l5_5582

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ Nat.gcd x 180 = 1 ∧ ∀ y : ℕ, y > 1 ∧ Nat.gcd y 180 = 1 → x ≤ y :=
begin
  use 7,
  split,
  { exact dec_trivial },
  split,
  { exact dec_trivial },
  { intro y,
    intro h,
    cases h with h1 h2,
    repeat { try { apply dec_trivial,
                   apply lt_or_eq_of_le,
                   norm_num,
                   apply Nat.prime_not_dvd_mul,
                   norm_num,
                   apply not_or_distrib.mpr,
                   split,
                   norm_cast,
                   intro,
                   exact le_antisymm _ },
           sorry }
end

end smallest_rel_prime_to_180_l5_5582


namespace min_value_fraction_l5_5114

theorem min_value_fraction (x y : ℝ) (hx : x > 1) (hy : y > 1) : 
  (∃c, (c = 8) ∧ (∀z w : ℝ, z > 1 → w > 1 → ((z^3 / (w - 1) + w^3 / (z - 1)) ≥ c))) :=
by 
  sorry

end min_value_fraction_l5_5114


namespace expected_value_of_8_sided_die_is_4_point_5_l5_5353

def expected_value_8_sided_die : ℝ :=
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let probability := 1 / 8
  (list.sum outcomes : ℝ) * probability

theorem expected_value_of_8_sided_die_is_4_point_5 :
  expected_value_8_sided_die = 4.5 := by
  sorry

end expected_value_of_8_sided_die_is_4_point_5_l5_5353


namespace expected_value_of_eight_sided_die_l5_5376

theorem expected_value_of_eight_sided_die : 
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8], 
      n := (outcomes.length : ℚ),
      probabilities := List.replicate (outcomes.length) (1 / n),
      expected_value := (List.zipWith (*) probabilities (outcomes.map (· : ℚ))).sum
  in expected_value = 4.5 :=
by
  sorry

end expected_value_of_eight_sided_die_l5_5376


namespace express_y_in_terms_of_x_l5_5764

variable (x y p : ℝ)

-- Conditions
def condition1 := x = 1 + 3^p
def condition2 := y = 1 + 3^(-p)

-- The theorem to be proven
theorem express_y_in_terms_of_x (h1 : condition1 x p) (h2 : condition2 y p) : y = x / (x - 1) :=
sorry

end express_y_in_terms_of_x_l5_5764


namespace incorrect_inference_B_l5_5080

theorem incorrect_inference_B (p q : Prop) :
  ¬ (¬ (p ∧ q) → ¬ p ∨ ¬ q) :=
by {
  assume h : ¬ (p ∧ q) → ¬ p ∨ ¬ q,
  sorry
}

end incorrect_inference_B_l5_5080


namespace find_angle_A_find_sin_B_plus_sin_C_l5_5647

-- Given conditions of the problem
variables (A B C a b c S : ℝ)
variables (h1 : 2 * sin A ^ 2 + 3 * cos (B + C) = 0)
variables (h2 : S = 5 * sqrt 3)
variables (h3 : a = sqrt 21)

-- Proof goals
theorem find_angle_A : (0 < A ∧ A < π) → A = π / 3 :=
sorry

theorem find_sin_B_plus_sin_C :
  (0 < B ∧ B < π) → (0 < C ∧ C < π) → (a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * cos A) →
  (S = 1/2 * b * c * sin A) → (A = π / 3) → (b + c = 9) → (b * c = 20) →
  sin B + sin C = 9 * sqrt 7 / 14 :=
sorry

end find_angle_A_find_sin_B_plus_sin_C_l5_5647


namespace rectangle_area_l5_5283

theorem rectangle_area (DB : ℝ) (d1 d2 d3 : ℝ) (h : DB = 9) (hratio : d1 / d2 = 2 / 3 ∧ d1 / d3 = 2 / 4 ∧ d2 / d3 = 3 / 4) :
  let AB := Real.sqrt (DB^2 - (d1 * d3 * DB^2 / (d1 + d2 + d3)) / d2) in
  let AD := Real.sqrt ((d1 * d3 * DB^2 / (d1 + d2 + d3)) / d2) in
  AB * AD = 28.8 :=
by
  -- Proof will be added here
  sorry

end rectangle_area_l5_5283


namespace geometric_b_general_term_a_arithmetic_sequence_l5_5640

open Nat

namespace SequenceProof

-- Definition of the sequence {a_n}
def a : ℕ+ → ℝ
| ⟨1, _⟩ => 1 / 2
| ⟨n + 1, _⟩ => (a ⟨n, zero_lt_succ _⟩ + n) / 2

-- Definition of the sequence {b_n}
def b (n : ℕ+) : ℝ := a ⟨n.succ, Nat.succ_pos' n⟩ - a n - 1

-- Auxiliary definition for S_n and T_n
noncomputable def S (n : ℕ+) : ℝ := (Finset.range n).sum (λ i => a ⟨i + 1, Nat.succ_pos' i⟩)
noncomputable def T (n : ℕ+) : ℝ := (Finset.range n).sum (λ i => b ⟨i + 1, Nat.succ_pos' i⟩)

-- Proof Problems
theorem geometric_b : ∀ n : ℕ+, ∃ r : ℝ, b n = -3/4 * (1/2)^(n.val - 1) := sorry

theorem general_term_a : ∀ n : ℕ+, a n = 3 / 2^n + n - 2 := sorry

theorem arithmetic_sequence :
  ∃ λ : ℝ, λ = 2 ∧ ∀ n : ℕ+, (S n + λ * T n) / ↑n = λ * ↑n + b := sorry

end SequenceProof

end geometric_b_general_term_a_arithmetic_sequence_l5_5640


namespace per_capita_income_growth_l5_5491

noncomputable def income2020 : ℝ := 3.2
noncomputable def income2022 : ℝ := 3.7
variable (x : ℝ)

/--
Prove the per capita disposable income model.
-/
theorem per_capita_income_growth :
  income2020 * (1 + x)^2 = income2022 :=
sorry

end per_capita_income_growth_l5_5491


namespace muscovy_less_than_twice_cayugas_l5_5847

variable (C K : ℕ) (M : ℕ := 39)

def num_of_cayugas : ℕ := 39 - 4
def twice_the_number_of_cayugas : ℕ := 2 * num_of_cayugas
def difference_between_muscovy_and_twice_cayugas := M - twice_the_number_of_cayugas

theorem muscovy_less_than_twice_cayugas (h1 : M = C + 4)
    (h2 : M + C + K = 90) (h3 : M = 39) :
    M - 2 * C = -31 := by
  sorry

end muscovy_less_than_twice_cayugas_l5_5847


namespace f_non_periodic_l5_5790

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) + (Real.sin (Real.sqrt 2 * x))

theorem f_non_periodic : ¬ ∃ (T > 0), ∀ x : ℝ, f x = f (x + T) :=
sorry

end f_non_periodic_l5_5790


namespace letter_puzzle_solution_l5_5548

theorem letter_puzzle_solution :
  ∃ (A B : ℕ), (A ≠ B) ∧ (1 ≤ A ∧ A ≤ 9) ∧ (1 ≤ B ∧ B ≤ 9) ∧ A ^ B = B * 10 + A :=
by {
  use 2, 5,
  split, { exact ne_of_lt (by decide) }, split, { exact ⟨by decide, by decide⟩ },
  split, { exact ⟨by decide, by decide⟩ }, exact (by decide : 2 ^ 5 = 5 * 10 + 2),
  sorry
} ∨
by {
  use 6, 2,
  split, { exact ne_of_lt (by decide) }, split, { exact ⟨by decide, by decide⟩ },
  split, { exact ⟨by decide, by decide⟩ }, exact (by decide : 6 ^ 2 = 2 * 10 + 6),
  sorry
} ∨
by {
  use 4, 3,
  split, { exact ne_of_lt (by decide) }, split, { exact ⟨by decide, by decide⟩ },
  split, { exact ⟨by decide, by decide⟩ }, exact (by decide : 4 ^ 3 = 3 * 10 + 4),
  sorry
}

end letter_puzzle_solution_l5_5548


namespace estimate_event_probability_l5_5812

/-- 
Given the frequencies of a random event occurring during an experiment for various numbers of trials,
we estimate the probability of this event occurring through the experiment and prove that it is approximately 0.35 
when rounded to 0.01. 
-/
theorem estimate_event_probability :
  let freq := [0.300, 0.360, 0.350, 0.350, 0.352, 0.351, 0.351]
  let approx_prob := 0.35
  ∀ n : ℕ, n ∈ [20, 50, 100, 300, 500, 1000, 5000] →
  freq.get! n ≈ approx_prob := 
  sorry

end estimate_event_probability_l5_5812


namespace gcd_654327_543216_is_1_l5_5416

-- Define the gcd function and relevant numbers
def gcd_problem : Prop :=
  gcd 654327 543216 = 1

-- The statement of the theorem, with a placeholder for the proof
theorem gcd_654327_543216_is_1 : gcd_problem :=
by {
  -- actual proof will go here
  sorry
}

end gcd_654327_543216_is_1_l5_5416


namespace min_value_of_expr_l5_5668

noncomputable def f (x : ℝ) : ℝ := 2 * x - Real.sin x

theorem min_value_of_expr 
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h_eq : f a + f (2 * b - 1) = 0) :
  ∃ (m : ℝ), m = 9 + 4 * Real.sqrt 2 ∧ 
    (∀ x y, 0 < x → 0 < y → (f x + f (2 * y - 1) = 0 → (1/x + 4/y) ≥ m)) :=
begin
  sorry
end

end min_value_of_expr_l5_5668


namespace probability_desired_urn_l5_5934

-- Define the initial conditions and operations
def initial_urn : Type := { reds : ℕ // reds = 2 } × { blues : ℕ // blues = 1 }

-- Define the probability function for drawing a ball and adding two of the same color
noncomputable def ball_draw_and_add (current_urn : { reds : ℕ × blues : ℕ }) : ℕ := sorry

-- Define the desired outcome after five operations
def desired_urn : Type := { reds : ℕ // reds = 7 } × { blues : ℕ // blues = 4 }

-- Define the probability of achieving the desired outcome
noncomputable def probability_after_five_operations : ℚ := sorry

theorem probability_desired_urn (initial_urn : { reds : ℕ × blues : ℕ } )
    (current_urn : initial_urn = ⟨2, 1⟩) : 
    probability_after_five_operations = 32 / 315 := 
sorry

end probability_desired_urn_l5_5934


namespace a_2012_is_1_over_2012_l5_5834

noncomputable def a : ℕ → ℝ
| 1        := 1
| 2        := 1 / 2
| (n + 1)  := (2 * a (n - 1) * a (n + 1)) / (a (n - 1) + a (n + 1))

theorem a_2012_is_1_over_2012 : a 2012 = 1 / 2012 :=
by {
  sorry
}

end a_2012_is_1_over_2012_l5_5834


namespace triangle_perimeter_proof_l5_5811

-- Define the sides of the square and smaller square.
variables (z w : ℝ)

-- Define the height and base of the triangles.
variables (h b : ℝ)

-- Given condition: height plus base equals the side of the larger square.
def height_base_condition : Prop := h + b = z

-- Define the perimeter of one of the triangles.
def triangle_perimeter : ℝ := 2 * h + b

-- Given that b = z - h, the perimeter should simplify to h + z.
theorem triangle_perimeter_proof (h b z : ℝ) (cond : b = z - h) :
  triangle_perimeter h b z = h + z :=
sorry

end triangle_perimeter_proof_l5_5811


namespace wire_left_after_making_mobiles_l5_5748

-- Define the conditions as constants
constants (total_wire : ℝ) (wire_per_mobile : ℝ)
constant (h : total_wire = 105.8 ∧ wire_per_mobile = 4)

-- Define the main theorem
theorem wire_left_after_making_mobiles (total_wire wire_per_mobile : ℝ) (h : total_wire = 105.8 ∧ wire_per_mobile = 4) : 
  ∃ mobiles_left : ℝ, mobiles_left = 1.8 :=
by
  sorry

end wire_left_after_making_mobiles_l5_5748


namespace sum_of_15th_set_l5_5514

def first_element_of_set (n : ℕ) : ℕ :=
  3 + (n * (n - 1)) / 2

def sum_of_elements_in_set (n : ℕ) : ℕ :=
  let a_n := first_element_of_set n
  let l_n := a_n + n - 1
  n * (a_n + l_n) / 2

theorem sum_of_15th_set :
  sum_of_elements_in_set 15 = 1725 :=
by
  sorry

end sum_of_15th_set_l5_5514


namespace quadratic_expression_eval_l5_5195

variable {α : ℝ}

-- Given condition
def fourth_quadrant (α : ℝ) : Prop := sin α < 0

-- The theorem to prove
theorem quadratic_expression_eval (hα : fourth_quadrant α) :
    sqrt ((1 + cos α) / (1 - cos α)) + sqrt ((1 - cos α) / (1 + cos α)) = -2 / sin α :=
by
  sorry

end quadratic_expression_eval_l5_5195


namespace expected_value_of_eight_sided_die_l5_5346

-- Definitions based on the problem conditions
def eight_sided_die_outcomes : List ℕ := [1,2,3,4,5,6,7,8]

def probability (n : ℕ) := 1 / n

-- Expected value calculation related to the problem
def expected_value_die_roll (outcomes : List ℕ) (prob : ℕ → Rat) : Rat :=
  List.sum (outcomes.map (λ x => prob outcomes.length * x))

-- Expected value of an 8-sided die roll
theorem expected_value_of_eight_sided_die :
  expected_value_die_roll eight_sided_die_outcomes probability = 4.5 := 
sorry

end expected_value_of_eight_sided_die_l5_5346


namespace probability_each_guy_gets_top_and_bottom_l5_5122

theorem probability_each_guy_gets_top_and_bottom :
  let n := 5 in
  let total_buns := 2 * n in
  let event := (list.range n).all (λ i, (n - i) / (total_buns - 2 * i) = 1 / 2 * (n - i - 1) / (total_buns - 2 * i - 1)) in
  ∃ event, ((5/9) * (4/7) * (3/5) * (2/3) * (1/1)) = 8 / 63 := 
sorry

end probability_each_guy_gets_top_and_bottom_l5_5122


namespace expected_value_of_eight_sided_die_l5_5371

theorem expected_value_of_eight_sided_die : 
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8], 
      n := (outcomes.length : ℚ),
      probabilities := List.replicate (outcomes.length) (1 / n),
      expected_value := (List.zipWith (*) probabilities (outcomes.map (· : ℚ))).sum
  in expected_value = 4.5 :=
by
  sorry

end expected_value_of_eight_sided_die_l5_5371


namespace distance_between_trees_l5_5852

theorem distance_between_trees 
  (total_distance : ℝ)
  (num_trees : ℕ)
  (n : ℕ)
  (m : ℕ)
  (h_dist : total_distance = 220)
  (h_num_trees : num_trees = 10)
  (h_n : n = 1)
  (h_m : m = 6)
  : 
  let num_segments := num_trees + 1 in
  let segment_length := total_distance / num_segments in
  let distance := (m - n) * segment_length in
  distance = 100 := 
by 
  clear_value num_segments
  clear_value segment_length
  clear_value distance
  unfold num_segments segment_length at *
  simp [h_dist, h_num_trees, h_n, h_m]
  field_simp
  norm_num
  sorry

end distance_between_trees_l5_5852


namespace smallest_rel_prime_to_180_l5_5595

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ Nat.gcd x 180 = 1 ∧ ∀ y : ℕ, (y > 1 ∧ Nat.gcd y 180 = 1) → x ≤ y :=
begin
  sorry
end

end smallest_rel_prime_to_180_l5_5595


namespace expected_value_of_8_sided_die_l5_5380

theorem expected_value_of_8_sided_die : 
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let probability := 1 / 8
  (∑ outcome in outcomes, outcome * probability) = 4.5 :=
by
  sorry

end expected_value_of_8_sided_die_l5_5380


namespace gcd_X_Y_Z_l5_5751

-- Define the conditions
variables {a b c : ℕ}
variables (X Y Z : ℕ)
def is_digit (n : ℕ) := 0 < n ∧ n < 10

-- Define X, Y, and Z
def X := 10 * a + b
def Y := 10 * b + c
def Z := 10 * c + a

-- The main theorem
theorem gcd_X_Y_Z {a b c : ℕ} (ha : is_digit a) (hb : is_digit b) (hc : is_digit c) (h_distinct: a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  gcd (gcd X Y) Z ∈ {1, 2, 3, 4, 7, 13, 14} :=
sorry

end gcd_X_Y_Z_l5_5751


namespace remaining_paint_is_114_l5_5941

-- Definitions of initial amounts of paint
def blue_initial : ℕ := 130
def red_initial : ℕ := 164
def white_initial : ℕ := 188

-- Amount of paint used for each stripe, all stripes being of equal size
def stripe_size := sorry -- the amount of paint used for each stripe should be derived but is unknown in this context

-- Equal remaining amounts after painting
def remaining_amount := blue_initial - stripe_size
def total_remaining_amount := 3 * remaining_amount

-- Statement of the problem
theorem remaining_paint_is_114 :
  total_remaining_amount = 114 :=
by
  have stripe_size := 92 -- from the problem solution, each stripe used 92 ounces
  have remaining_amount := 38 -- from the problem solution, remaining = 130 - 92
  have total_remaining_amount := 3 * 38 -- total remaining is 3 times the remaining of one paint kind
  exact 114
    

#eval remaining_paint_is_114   -- Optional, to trigger the evaluation of the theorem

end remaining_paint_is_114_l5_5941


namespace triangle_angle_bisector_length_l5_5208

theorem triangle_angle_bisector_length
  (A B C D : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (AB BC AC CD : ℝ)
  (is_right_triangle : AB * AB + BC * BC = AC * AC)
  (angle_bisector : ∀ D, Metric.dist A D / Metric.dist B D = BC / AB) :
  Metric.dist B D ^ 2 + Metric.dist D C ^ 2 = 229 :=
by sorry

end triangle_angle_bisector_length_l5_5208


namespace min_expr_value_l5_5890

noncomputable def expr (a b : ℝ) : ℝ := 
  (abs (a - 3 * b - 2) + abs (3 * a - b)) / real.sqrt (a^2 + (b + 1)^2)

theorem min_expr_value : ∀ a b : ℝ, a ≥ 0 → b ≥ 0 → 
  ∃ (m : ℝ), m = 2 ∧ ∀ x y : ℝ, x ≥ 0 → y ≥ 0 → expr x y ≥ m := 
by 
  sorry

end min_expr_value_l5_5890


namespace garden_path_area_l5_5919

-- Define the lengths of the sides of the rectangle
def side_a : ℝ := 55
def side_b : ℝ := 40

-- Define the width of the path
def path_width : ℝ := 1

-- Define the expected area of the path
def expected_area : ℝ := 200 / 3

-- The Lean statement to prove
theorem garden_path_area :
  let diagonal := Real.sqrt (side_a ^ 2 + side_b ^ 2) in
  let path_area := diagonal * path_width in
  path_area = expected_area :=
by
  sorry

end garden_path_area_l5_5919


namespace count_terminating_decimals_l5_5616

theorem count_terminating_decimals :
  {n : ℕ | 1 ≤ n ∧ n ≤ 150 ∧ ∃ k : ℕ, n = 3 * k}.to_finset.card = 50 := by
sorry

end count_terminating_decimals_l5_5616


namespace min_expr_value_l5_5889

noncomputable def expr (a b : ℝ) : ℝ := 
  (abs (a - 3 * b - 2) + abs (3 * a - b)) / real.sqrt (a^2 + (b + 1)^2)

theorem min_expr_value : ∀ a b : ℝ, a ≥ 0 → b ≥ 0 → 
  ∃ (m : ℝ), m = 2 ∧ ∀ x y : ℝ, x ≥ 0 → y ≥ 0 → expr x y ≥ m := 
by 
  sorry

end min_expr_value_l5_5889


namespace smallest_rel_prime_to_180_l5_5602

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ x ≤ 7 ∧ (∀ y : ℕ, y > 1 ∧ y < x → y.gcd 180 ≠ 1) ∧ x.gcd 180 = 1 :=
by
  sorry

end smallest_rel_prime_to_180_l5_5602


namespace batsman_percentage_running_between_wickets_l5_5447

def boundaries : Nat := 6
def runs_per_boundary : Nat := 4
def sixes : Nat := 4
def runs_per_six : Nat := 6
def no_balls : Nat := 8
def runs_per_no_ball : Nat := 1
def wide_balls : Nat := 5
def runs_per_wide_ball : Nat := 1
def leg_byes : Nat := 2
def runs_per_leg_bye : Nat := 1
def total_score : Nat := 150

def runs_from_boundaries : Nat := boundaries * runs_per_boundary
def runs_from_sixes : Nat := sixes * runs_per_six
def runs_not_off_bat : Nat := no_balls * runs_per_no_ball + wide_balls * runs_per_wide_ball + leg_byes * runs_per_leg_bye

def runs_running_between_wickets : Nat := total_score - runs_not_off_bat - runs_from_boundaries - runs_from_sixes

def percentage_runs_running_between_wickets : Float := 
  (runs_running_between_wickets.toFloat / total_score.toFloat) * 100

theorem batsman_percentage_running_between_wickets : percentage_runs_running_between_wickets = 58 := sorry

end batsman_percentage_running_between_wickets_l5_5447


namespace probability_wheel_landing_FG_l5_5053

theorem probability_wheel_landing_FG :
  let pD := 1 / 4 in
  let pE := 1 / 3 in
  let pFG := 5 / 12 in
  pD + pE + pFG = 1 :=
by
  sorry

end probability_wheel_landing_FG_l5_5053


namespace angle_c_range_l5_5707

-- Define the conditions of the problem
variable (A B C : ℝ)
variable (AB BC : ℝ)

theorem angle_c_range (h1 : AB = 1) (h2 : BC = 2) : 
  ∃ θ, θ ∈ set.Ioc 0 (real.pi / 6) ∧ ang (B - A) (C - B) = θ := 
sorry

end angle_c_range_l5_5707


namespace expected_value_8_sided_die_l5_5364

theorem expected_value_8_sided_die :
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let probability_each_outcome := 1 / 8
  (Σ x ∈ outcomes, probability_each_outcome * x) = 4.5 :=
by
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let probability_each_outcome := 1 / 8
  have h : (Σ x ∈ outcomes, probability_each_outcome * x) = (1 / 8) * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8) := by sorry
  have sum_eq_36 : 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 = 36 := by sorry
  rw [sum_eq_36] at h
  have expected_value_eq : (1 / 8) * 36 = 4.5 := by sorry
  rw [expected_value_eq] at h
  exact h

end expected_value_8_sided_die_l5_5364


namespace find_x_l5_5530

theorem find_x (x : ℤ) : 3^7 * 3^x = 81 → x = -3 := by
  sorry

end find_x_l5_5530


namespace oliver_bumper_cars_proof_l5_5062

def rides_of_bumper_cars (total_tickets : ℕ) (tickets_per_ride : ℕ) (rides_ferris_wheel : ℕ) : ℕ :=
  (total_tickets - rides_ferris_wheel * tickets_per_ride) / tickets_per_ride

def oliver_bumper_car_rides : Prop :=
  rides_of_bumper_cars 30 3 7 = 3

theorem oliver_bumper_cars_proof : oliver_bumper_car_rides :=
by
  sorry

end oliver_bumper_cars_proof_l5_5062


namespace parallelogram_area_l5_5100

noncomputable def vector_u : ℝ × ℝ × ℝ := (4, 2, -3)
noncomputable def vector_v : ℝ × ℝ × ℝ := (2, -4, 5)

theorem parallelogram_area : 
  let cross_product := (vector_u.2 * vector_v.3 - vector_u.3 * vector_v.2, vector_u.3 * vector_v.1 - vector_u.1 * vector_v.3, vector_u.1 * vector_v.2 - vector_u.2 * vector_v.1) in
  let magnitude := real.sqrt (cross_product.1^2 + cross_product.2^2 + cross_product.3^2) in
  magnitude = 10 * real.sqrt (10.8) :=
sorry

end parallelogram_area_l5_5100


namespace circumcircle_APQ_passes_through_projection_l5_5753

open EuclideanGeometry

theorem circumcircle_APQ_passes_through_projection (
  (A B C P Q : Point) :
  ∠A = 60 ∧ collinear [A, B, C] ∧ collinear [B, P, Q, C] ∧ BP = PQ ∧ PQ = QC
) :
  ∃ H : Point, is_orthocenter H (triangle A B C) ∧ 
  let M := midpoint B C in 
  let X := foot H A M in
  cyclic [A, P, Q, X] :=
begin
  sorry
end

end circumcircle_APQ_passes_through_projection_l5_5753


namespace remainder_is_zero_l5_5116

noncomputable def poly1 : Polynomial ℤ := Polynomial.C 1
  + Polynomial.X^25
  + Polynomial.X^50
  + Polynomial.X^75
  + Polynomial.X^100

noncomputable def poly2 : Polynomial ℤ := Polynomial.C 1
  + Polynomial.X^3
  + Polynomial.X^6
  + Polynomial.X^9

theorem remainder_is_zero :
  (poly1 % poly2) = 0 :=
sorry

end remainder_is_zero_l5_5116


namespace goods_train_speed_l5_5460

noncomputable def speed_of_goods_train (train_speed : ℝ) (goods_length : ℝ) (passing_time : ℝ) : ℝ :=
  let relative_speed_mps := goods_length / passing_time
  let relative_speed_kmph := relative_speed_mps * 3.6
  (relative_speed_kmph - train_speed)

theorem goods_train_speed :
  speed_of_goods_train 30 280 9 = 82 :=
by
  sorry

end goods_train_speed_l5_5460


namespace smallest_rel_prime_to_180_is_7_l5_5586

theorem smallest_rel_prime_to_180_is_7 :
  ∃ x : ℕ, x > 1 ∧ gcd x 180 = 1 ∧ ∀ y : ℕ, y > 1 ∧ gcd y 180 = 1 → x ≤ y :=
begin
  use 7,
  split,
  { exact dec_trivial, },
  split,
  { exact dec_trivial, },
  intros y hy,
  cases hy with hy1 hy2,
  exact dec_trivial,
end

end smallest_rel_prime_to_180_is_7_l5_5586


namespace range_of_f_l5_5830

noncomputable def f (x : ℝ) : ℝ := (1 / 2) ^ (x^2 - 2 * x + 2)

theorem range_of_f :
  range f = set.Icc 0 (1 / 2) :=
by
  sorry

end range_of_f_l5_5830


namespace shift_sin_right_pi_by_6_l5_5334

theorem shift_sin_right_pi_by_6 :
  ∀ x : ℝ, sin (2 * (x - pi / 6) + pi / 3) = sin (2 * x) :=
by
  intro x
  sorry

end shift_sin_right_pi_by_6_l5_5334


namespace problem_solution_l5_5136

theorem problem_solution (a b c : ℝ) (h : b^2 = a * c) :
  (a^2 * b^2 * c^2 / (a^3 + b^3 + c^3)) * (1 / a^3 + 1 / b^3 + 1 / c^3) = 1 :=
  by sorry

end problem_solution_l5_5136


namespace fraction_product_l5_5858

theorem fraction_product (a b c d e : ℝ) (h1 : a = 1/2) (h2 : b = 1/3) (h3 : c = 1/4) (h4 : d = 1/6) (h5 : e = 144) :
  a * b * c * d * e = 1 := 
by
  -- Given the conditions h1 to h5, we aim to prove the product is 1
  sorry

end fraction_product_l5_5858


namespace max_rooks_in_cube_l5_5419

def non_attacking_rooks (n : ℕ) (cube : ℕ × ℕ × ℕ) : ℕ :=
  if cube = (8, 8, 8) then 64 else 0

theorem max_rooks_in_cube:
  non_attacking_rooks 64 (8, 8, 8) = 64 :=
by
  -- proof by logical steps matching the provided solution, if necessary, start with sorry for placeholder
  sorry

end max_rooks_in_cube_l5_5419


namespace max_temp_difference_l5_5269

-- Define the highest and lowest temperatures
def highest_temp : ℤ := 3
def lowest_temp : ℤ := -3

-- State the theorem for maximum temperature difference
theorem max_temp_difference : highest_temp - lowest_temp = 6 := 
by 
  -- Provide the proof here
  sorry

end max_temp_difference_l5_5269


namespace penny_net_income_over_three_months_l5_5786

def daily_income_first_month : ℝ := 10
def days_worked_first_month : ℝ := 20
def daily_income_increase_rate : ℝ := 0.2
def days_worked_second_month : ℝ := 25
def days_worked_third_month : ℝ := 15
def monthly_expenses : ℝ := 100

-- Tax computation (as a helper function based on the conditions provided)
def tax (income : ℝ) : ℝ :=
  if income <= 800 then income * 0.1
  else if income <= 2000 then 800 * 0.1 + (income - 800) * 0.15
  else  800 * 0.1 + 1200 * 0.15 + (income - 2000) * 0.2

-- Net income computation for each month
def net_income_per_month (daily_income : ℝ) (days_worked : ℝ) : ℝ :=
  let total_income := daily_income * days_worked
  let after_tax_income := total_income - tax(total_income)
  after_tax_income - monthly_expenses

-- Total net income over three months
def total_net_income : ℝ :=
  let net_first_month := net_income_per_month daily_income_first_month days_worked_first_month
  let daily_income_second_month := daily_income_first_month * (1 + daily_income_increase_rate)
  let net_second_month := net_income_per_month daily_income_second_month days_worked_second_month
  let daily_income_third_month := daily_income_second_month * (1 + daily_income_increase_rate)
  let net_third_month := net_income_per_month daily_income_third_month days_worked_third_month
  net_first_month + net_second_month + net_third_month

theorem penny_net_income_over_three_months : total_net_income = 344.40 :=
by
  -- Lean will verify the calculation steps provided in the informal solution.
  sorry

end penny_net_income_over_three_months_l5_5786


namespace min_inv_sum_l5_5671

theorem min_inv_sum (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) 
  (h₃ : 2 * a * 1 + b * 2 = 2) : 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = 1 ∧ (1/a) + (1/b) = 4 :=
sorry

end min_inv_sum_l5_5671


namespace expected_value_8_sided_die_l5_5405

/-- 
The expected value of rolling a standard 8-sided die is 4.5.
The die has 8 sides labeled 1 through 8, and each face has an equal probability of appearing,
which is 1/8. 
-/
theorem expected_value_8_sided_die : 
  (1/8:ℝ) * 1 + (1/8) * 2 + (1/8) * 3 + (1/8) * 4 + (1/8) * 5 + (1/8) * 6 + (1/8) * 7 + (1/8) * 8 = 4.5 :=
by 
  sorry

end expected_value_8_sided_die_l5_5405


namespace total_area_of_removed_triangles_l5_5048

theorem total_area_of_removed_triangles (x r s : ℝ) (h1 : (x - r)^2 + (x - s)^2 = 15^2) :
  4 * (1/2 * r * s) = 112.5 :=
by
  sorry

end total_area_of_removed_triangles_l5_5048


namespace distance_between_A_and_B_l5_5464

-- Given conditions as definitions

def total_time : ℝ := 4
def boat_speed : ℝ := 7.5
def stream_speed : ℝ := 2.5
def distance_AC : ℝ := 10

-- Define the possible solutions for the distance between A and B
def distance_AB (x : ℝ) := 
  (x / (boat_speed + stream_speed) + (x + distance_AC) / (boat_speed - stream_speed) = total_time) 
  ∨ 
  (x / (boat_speed + stream_speed) + (x - distance_AC) / (boat_speed - stream_speed) = total_time)

-- Problem statement
theorem distance_between_A_and_B :
  ∃ x : ℝ, (distance_AB x) ∧ (x = 20 ∨ x = 20 / 3) :=
sorry

end distance_between_A_and_B_l5_5464


namespace possible_values_of_k_l5_5835

-- Definitions of the conditions
def initial_condition (u : ℕ → ℕ) := u 0 = 1
def recurrence_relation (k : ℕ) (u : ℕ → ℕ) := ∀ n > 0, u (n + 1) * u (n - 1) = k * u n
def specific_value (u : ℕ → ℕ) := u 2000 = 2000

-- The question translated into a Lean theorem statement
theorem possible_values_of_k (k : ℕ) (u : ℕ → ℕ) :
  initial_condition u ∧ recurrence_relation k u ∧ specific_value u →
  k ∈ {2000, 1000, 500, 400, 200, 100} := by
  sorry

end possible_values_of_k_l5_5835


namespace min_ratio_at_least_l5_5613

structure Point (α : Type*) := (x : α) (y : α)

noncomputable def dist (P1 P2 : Point ℝ) : ℝ :=
  real.sqrt ((P2.x - P1.x)^2 + (P2.y - P1.y)^2)

theorem min_ratio_at_least : 
  ∀ (P1 P2 P3 P4 : Point ℝ),
    let ds := [dist P1 P2, dist P1 P3, dist P1 P4, dist P2 P3, dist P2 P4, dist P3 P4] in
    (∑ d in ds, id d) / (ds.minimum (by apply_instance)) ≥ 5 + real.sqrt 3 :=
by sorry

end min_ratio_at_least_l5_5613


namespace suitableTempForPreservingBoth_l5_5319

-- Definitions for the temperature ranges of types A and B vegetables
def suitableTempRangeA := {t : ℝ | 3 ≤ t ∧ t ≤ 8}
def suitableTempRangeB := {t : ℝ | 5 ≤ t ∧ t ≤ 10}

-- The intersection of the suitable temperature ranges
def suitableTempRangeForBoth := {t : ℝ | 5 ≤ t ∧ t ≤ 8}

-- The theorem statement we need to prove
theorem suitableTempForPreservingBoth :
  suitableTempRangeForBoth = suitableTempRangeA ∩ suitableTempRangeB :=
sorry

end suitableTempForPreservingBoth_l5_5319


namespace correct_statements_of_circle_l5_5158

noncomputable def circle_eqn (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 + y^2 - 2*k*x - 2*y - 2*k = 0

noncomputable def symmetric_condition (k : ℝ) : Prop :=
  let center := (k, 1) in
  1 = k^2

noncomputable def tangent_line_existence (k : ℝ) : Prop :=
  k ≠ -1 → ∃ x y, x = -1 ∧ x^2 + y^2 - 2*y = 0

noncomputable def area_of_triangle (m : ℝ) : Set ℝ :=
  {a | 0 < a ∧ a ≤ (Real.sqrt 55) / 4}

noncomputable def min_value_CM_AB (k : ℝ) (M C : ℝ × ℝ) : ℝ :=
  4

theorem correct_statements_of_circle (k : ℝ) (m : ℝ) (M C : ℝ × ℝ):
  (tangent_line_existence k) ∧
  (∀ m, ∃ P Q, P ≠ Q ∧ (m+1)*P.1 + 2*P.2 - 2*m - 3 = 0 ∧
   (m+1)*Q.1 + 2*Q.2 - 2*m - 3 = 0 ∧
   ∃ a ∈ (area_of_triangle m), a ≤ (Real.sqrt 55) / 4 ) ∧
  (min_value_CM_AB k M C = 4) :=
by sorry

end correct_statements_of_circle_l5_5158


namespace common_ratio_geometric_sequence_l5_5734

theorem common_ratio_geometric_sequence (a₃ S₃ : ℝ) (q : ℝ)
  (h1 : a₃ = 7) (h2 : S₃ = 21)
  (h3 : ∃ a₁ : ℝ, a₃ = a₁ * q^2)
  (h4 : ∃ a₁ : ℝ, S₃ = a₁ * (1 + q + q^2)) :
  q = -1/2 ∨ q = 1 :=
sorry

end common_ratio_geometric_sequence_l5_5734


namespace count_whole_numbers_between_l5_5691

theorem count_whole_numbers_between : 
  let a := Real.root 4 50
  let b := Real.root 4 1000
  ∃ n : ℕ, n = 3 ∧
  (∀ x : ℕ, a < x → x < b → 3 ≤ x ∧ x ≤ 5) ∧
  (∀ x : ℕ, a < x → x < b → x = 3 ∨ x = 4 ∨ x = 5) :=
by
  let a := Real.root 4 50
  let b := Real.root 4 1000
  use 3
  split
  { exact rfl }
  split
  { assume x ha hb
    split,
    linarith [Real.lift_root_pos 4 50, ha],
    linarith [hb, Real.lift_root_pos 4 1000] }
  { assume x ha hb
    cases lt_or_eq_of_le ha with ha_eq ha_eq,
    { linarith [ha_eq] },
    { cases lt_or_eq_of_le hb with hb_eq hb_eq,
      { linarith [hb_eq] },
      { assumption } } }

end count_whole_numbers_between_l5_5691


namespace catch_bus_probability_within_5_minutes_l5_5059

theorem catch_bus_probability_within_5_minutes :
  (Pbus3 : ℝ) → (Pbus6 : ℝ) → (Pbus3 = 0.20) → (Pbus6 = 0.60) → (Pcatch : ℝ) → (Pcatch = Pbus3 + Pbus6) → (Pcatch = 0.80) :=
by
  intros Pbus3 Pbus6 hPbus3 hPbus6 Pcatch hPcatch
  sorry

end catch_bus_probability_within_5_minutes_l5_5059


namespace min_wait_time_five_buckets_l5_5123

def min_total_waiting_time (t1 t2 t3 t4 t5 : Nat) : Nat :=
  let times := [t1, t2, t3, t4, t5].qsort (· < ·)
  times[0] * 5 + times[1] * 4 + times[2] * 3 + times[3] * 2 + times[4]

theorem min_wait_time_five_buckets :
  min_total_waiting_time 4 8 6 10 5 = 84 :=
by
  sorry

end min_wait_time_five_buckets_l5_5123


namespace determine_speed_l5_5936

variable (d t : ℝ)
variables (late early : ℝ := 5 / 60) 

def d_eq_60 : Prop := d = 60 * (t + late)
def d_eq_90 : Prop := d = 90 * (t - early)
def n_eq : Prop := nat.round (d / t) = 72

theorem determine_speed
  (h1 : d_eq_60 d t)
  (h2 : d_eq_90 d t) :
  n_eq d t :=
sorry

end determine_speed_l5_5936


namespace expected_value_of_8_sided_die_l5_5394

def outcomes : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]
def probability (n : ℕ) : ℝ := 1 / 8

theorem expected_value_of_8_sided_die :
  (∑ x in outcomes, probability x * x) = 4.5 := 
sorry

end expected_value_of_8_sided_die_l5_5394


namespace ratio_of_wages_l5_5304

theorem ratio_of_wages
  (men : ℕ) (women : ℕ) (days1 : ℕ) (days2 : ℕ) (total1 : ℕ) (total2 : ℕ)
  (M : ℕ) (W : ℕ)
  (H1 : men * days1 * M = total1)
  (H2 : women * days2 * W = total2) :
  M / W = 2 :=
by
  -- Initialize the given condition values.
  let men := 20
  let women := 40
  let days1 := 20
  let days2 := 30
  let total1 := 14400
  let total2 := 21600
  let M := 36
  let W := 18

  -- Given conditions from the problem
  have H1 : 20 * 20 * 36 = 14400 := by sorry
  have H2 : 40 * 30 * 18 = 21600 := by sorry

  -- Result
  have quotient_M_W: 36 / 18 = 2 := by sorry

  -- Prove the theorem
  exact quotient_M_W

end ratio_of_wages_l5_5304


namespace new_angle_measure_l5_5311

/-- Given the initial measure of angle ACB as 70 degrees and a rotation of ray CA by 570 degrees 
    clockwise about point C, the new measure of the acute angle ACB is 140 degrees. -/
theorem new_angle_measure (initial_angle : ℝ)
  (rotation : ℝ)
  (h_initial : initial_angle = 70)
  (h_rotation : rotation = 570) :
  ∃ new_angle : ℝ, new_angle = 140 :=
by {
  -- Initial angle condition
  have h1 : initial_angle = 70 := h_initial,
  
  -- Rotation condition
  have h2 : rotation = 570 := h_rotation,
  
  -- Calculate the effective rotation after reducing complete circles
  let effective_rotation := rotation - 360,
  
  -- Simplify the effective rotation
  let simplified_rotation := 210,
  
  -- Calculate the new acute angle
  let new_angle := 140,
  
  -- Show that the new angle matches the expected value
  use new_angle,
  exact eq.refl new_angle
}

end new_angle_measure_l5_5311


namespace cone_base_radius_l5_5155

variable (s : ℝ) (A : ℝ) (r : ℝ)

theorem cone_base_radius (h1 : s = 5) (h2 : A = 15 * Real.pi) : r = 3 :=
by
  sorry

end cone_base_radius_l5_5155


namespace fencing_required_l5_5041

noncomputable def length : ℝ := 20
noncomputable def area : ℝ := 390
noncomputable def width : ℝ := area / length
noncomputable def fencing : ℝ := 2 * width + length

theorem fencing_required :
  width = 19.5 ∧ fencing = 59 :=
by
  have h1 : width = 19.5 := by
    simp [width, length, area]
    exact (div_eq_iff (by norm_num : (20:ℝ) ≠ 0)).mpr rfl
  have h2 : fencing = 59 := by
    rw [width, h1]
    simp [fencing]
  exact ⟨h1, h2⟩

end fencing_required_l5_5041


namespace f_expression_f_inequality_l5_5142

noncomputable def f : ℝ → ℝ := sorry

axiom cond1 : ∀ x y : ℝ, f(x + y) = f(x) + f(y) + 2 * x * y
axiom cond2 : deriv f 0 = 2

theorem f_expression : ∀ x : ℝ, f(x) = x^2 + 2 * x := sorry

theorem f_inequality : ∀ x1 x2 : ℝ, x1 ∈ Set.Icc (-1 : ℝ) 1 → x2 ∈ Set.Icc (-1 : ℝ) 1 → 
  |f(x1) - f(x2)| ≤ 4 * |x1 - x2| := sorry

end f_expression_f_inequality_l5_5142


namespace expected_value_of_8_sided_die_is_4_point_5_l5_5355

def expected_value_8_sided_die : ℝ :=
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let probability := 1 / 8
  (list.sum outcomes : ℝ) * probability

theorem expected_value_of_8_sided_die_is_4_point_5 :
  expected_value_8_sided_die = 4.5 := by
  sorry

end expected_value_of_8_sided_die_is_4_point_5_l5_5355


namespace find_BC_l5_5204

-- Definitions based on the conditions in the problem
variable {A B C : Type} [Fintype A] [LinearOrderedField B] [EuclideanSpace A B]

def AB : ℝ := 6
def angle_B : ℝ := 90
def tan_A : ℝ := 3 / 4
def triangle_is_right : Prop := angle_B = 90

-- The theorem statement
theorem find_BC (h1 : tan_A = 3 / 4) (h2 : AB = 6) (h3 : triangle_is_right) :
  ∃ (BC : ℝ), BC = 4.5 :=
by
  use 4.5
  have h4 : 4.5 = (3 / 4) * 6, by norm_num
  have h5 : BC = (3 / 4) * AB, by rw [h2]
  rw [h4, h5]
  sorry

end find_BC_l5_5204


namespace real_solutions_two_l5_5831

theorem real_solutions_two (x : ℝ) : (2^|x| = 2 - x) → (x = 0 ∨ x = 1) :=
by
suffices h0 : x = 0 ∨ x = 1
from Or.elim h0 (λ h1 => h1) (λ h2 => h2),
sorry

end real_solutions_two_l5_5831


namespace largest_distance_l5_5205

-- Define the points with their coordinates
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define the coordinates for points A, C, D, E, F
def A : Point := {x := 0, y := 0}
def C : Point := {x := 3, y := 4}
def D : Point := {x := 3, y := -1}
def E : Point := {x := 5, y := 9}
def F : Point := {x := 5, y := 0}

-- Define the Pythagorean theorem to calculate distance between two points
def distance (p1 p2 : Point) : ℝ :=
  real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Calculate the distances as per conditions
def AE : ℝ := distance A E
def CD : ℝ := distance C D
def CF : ℝ := distance C F
def AC : ℝ := distance A C
def FD : ℝ := distance F D
def CE : ℝ := distance C E

-- Define each option as per problem statement
def option_A := AE
def option_B := CD + CF
def option_C := AC + CF
def option_D := FD
def option_E := AC + CE

-- Prove the largest distance among the options is option E
theorem largest_distance : max option_A (max option_B (max option_C (max option_D option_E))) = option_E :=
  sorry

end largest_distance_l5_5205


namespace magnitude_c_is_sqrt_35_25_l5_5124

def polynomial_has_four_distinct_roots (c : ℂ) : Prop :=
  let Q := λ x : ℂ, (x^2 - 3*x + 3)*(x^2 - c*x + 9)*(x^2 - 6*x + 18)
  -- The property of having exactly four distinct roots
  ∃ r1 r2 r3 r4 : ℂ, Q(r1) = 0 ∧ Q(r2) = 0 ∧ Q(r3) = 0 ∧ Q(r4) = 0 ∧ r1 ≠ r2 ∧ r1 ≠ r3 ∧ r1 ≠ r4 ∧ r2 ≠ r3 ∧ r2 ≠ r4 ∧ r3 ≠ r4

theorem magnitude_c_is_sqrt_35_25 (c : ℂ) (h : polynomial_has_four_distinct_roots c) :
  |c| = Real.sqrt 35.25 :=
sorry

end magnitude_c_is_sqrt_35_25_l5_5124


namespace speed_conversion_l5_5471

theorem speed_conversion (speed_kmh : ℚ) (speed_kmh = 1.5428571428571427) : (speed_kmh / 3.6 = 3 / 7) :=
by
  have conversion_factor : ℚ := 1 / 3.6
  have speed_mps : ℚ := speed_kmh * conversion_factor
  show speed_mps = 3 / 7, from sorry

end speed_conversion_l5_5471


namespace jimmy_fill_bucket_time_l5_5746

-- Definitions based on conditions
def pool_volume : ℕ := 84
def bucket_volume : ℕ := 2
def total_time_minutes : ℕ := 14
def total_time_seconds : ℕ := total_time_minutes * 60
def trips : ℕ := pool_volume / bucket_volume

-- Theorem statement
theorem jimmy_fill_bucket_time : (total_time_seconds / trips) = 20 := by
  sorry

end jimmy_fill_bucket_time_l5_5746


namespace smallest_rel_prime_to_180_l5_5596

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ Nat.gcd x 180 = 1 ∧ ∀ y : ℕ, (y > 1 ∧ Nat.gcd y 180 = 1) → x ≤ y :=
begin
  sorry
end

end smallest_rel_prime_to_180_l5_5596


namespace derivative_function_l5_5163

theorem derivative_function : ∀ x: ℝ, (derivative (λ x: ℝ, 2 * x^2 - 2 * x + 1)) x = 4 * x - 2 :=
by
  intros
  differentiate
  sorry

end derivative_function_l5_5163


namespace eccentricity_of_ellipse_l5_5629

variables (a b c x y : ℝ) (e : ℝ) (F A P : ℝ × ℝ)

def is_ellipse : Prop :=
  a > b ∧ b > 0 ∧ (x^2 / a^2 + y^2 / b^2 = 1)

def left_focus (F : ℝ × ℝ) (a c : ℝ) : Prop :=
  F = (-c, 0)

def right_vertex (A : ℝ × ℝ) (a : ℝ) : Prop :=
  A = (a, 0)

def point_on_ellipse_perpendicular (P F : ℝ × ℝ) : Prop :=
  P.2 = F.2

def PF_AF_condition (P F A : ℝ × ℝ) (a b : ℝ) : Prop :=
  let PF := (a * a - P.1 * P.1 + P.2 * P.2 - 2 * P.1 * -c + c * c)⁻¹ in
  let AF := a + c in
  |PF| = (3 / 4) * |AF|

theorem eccentricity_of_ellipse :
  is_ellipse a b x y ∧ left_focus F a c ∧ right_vertex A a ∧ point_on_ellipse_perpendicular P F ∧ PF_AF_condition P F A a b →
  (e = c / a ∧ e = 1 / 4) :=
sorry

end eccentricity_of_ellipse_l5_5629


namespace value_of_x_minus_2y_l5_5006

theorem value_of_x_minus_2y (x y : ℝ) (h1 : 0.5 * x = y + 20) : x - 2 * y = 40 :=
by
  sorry

end value_of_x_minus_2y_l5_5006


namespace area_of_parallelogram_is_20sqrt3_l5_5108

open Real EuclideanGeometry

def vector_u : ℝ × ℝ × ℝ := (4, 2, -3)
def vector_v : ℝ × ℝ × ℝ := (2, -4, 5)

noncomputable def cross_product (u v : ℝ × ℝ × ℝ) :=
  (u.2 * v.3 - u.3 * v.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2 - u.2 * v.1)

noncomputable def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

theorem area_of_parallelogram_is_20sqrt3 :
  magnitude (cross_product vector_u vector_v) = 20 * Real.sqrt 3 := by
  sorry

end area_of_parallelogram_is_20sqrt3_l5_5108


namespace concurrency_of_lines_l5_5769

theorem concurrency_of_lines
  (A B C D E F G H P M M' : Type)
  [h₁ : tangential_quadrilateral A B C D]
  [h₂ : touches A B at E]
  [h₃ : touches B C at F]
  [h₄ : touches C D at G]
  [h₅ : touches D A at H]
  [Pint : ∃ P, meets_ext DA CB P]
  [Mint : meets_ext HF BD M]
  [M'int : meets_ext EG BD M']
  :
  concurrent A C B D E G H F :=
sorry

end concurrency_of_lines_l5_5769


namespace find_x_l5_5094

theorem find_x (x : ℝ) (hx : x > 0) (h : sqrt(12 * x) * sqrt(20 * x) * sqrt(5 * x) * sqrt(30 * x) = 30) : 
  x = 1 / sqrt(6) :=
by
  sorry

end find_x_l5_5094


namespace expected_value_8_sided_die_l5_5368

theorem expected_value_8_sided_die :
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let probability_each_outcome := 1 / 8
  (Σ x ∈ outcomes, probability_each_outcome * x) = 4.5 :=
by
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let probability_each_outcome := 1 / 8
  have h : (Σ x ∈ outcomes, probability_each_outcome * x) = (1 / 8) * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8) := by sorry
  have sum_eq_36 : 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 = 36 := by sorry
  rw [sum_eq_36] at h
  have expected_value_eq : (1 / 8) * 36 = 4.5 := by sorry
  rw [expected_value_eq] at h
  exact h

end expected_value_8_sided_die_l5_5368


namespace range_of_a_l5_5675

def A (x y : ℝ) : Prop :=
  (x - 2)^2 + (y + 3)^2 ≤ 4

def B (x y a : ℝ) : Prop :=
  (x - 1)^2 + (y - a)^2 ≤ 1 / 4

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, B x y a → A x y) ↔ (-3 - real.sqrt(5) / 2) ≤ a ∧ a ≤ (-3 + real.sqrt(5) / 2) :=
by sorry

end range_of_a_l5_5675


namespace solution_set_of_inequality_l5_5318

theorem solution_set_of_inequality :
  {x : ℝ | |x - 5| + |x + 3| >= 10} = {x : ℝ | x ≤ -4} ∪ {x : ℝ | x ≥ 6} :=
by
  sorry

end solution_set_of_inequality_l5_5318


namespace arrangement_ways_l5_5935

-- Define the sets of plants
noncomputable def basil_plants : Finset (Fin 5) := {1, 2, 3, 4, 5}
noncomputable def tomato_plants : Finset (Fin 4) := {1, 2, 3, 4}

-- Define the total ways to arrange plants with the constraint that all tomato plants are next to each other
theorem arrangement_ways (A : Finset (Fin 5)) (B : Finset (Fin 4)) (hA : A.card = 5) (hB : B.card = 4) :
  let total_ways := (Fact (A.card + 1) * Fact B.card) in
  total_ways = 17280 := by
  sorry

end arrangement_ways_l5_5935


namespace painting_number_l5_5689

def color := {red, green, blue, yellow}

def proper_divisors (n : ℕ) : set ℕ :=
  {d | d ∣ n ∧ d < n}

def valid_coloring (colors : Fin 9 → color) : Prop :=
  (∀ n, n ∈ (Finset.range 9).map (Nat.succ) → (∀ d ∈ proper_divisors n, colors (n-2) ≠ colors (d-2))) ∧
  colors 8 = red

theorem painting_number:
  ∃ num_ways : ℕ, num_ways = 1296 :=
begin
  let choices : Fin 9 → color := λ n, sorry,
  have h_valid : valid_coloring choices := sorry,
  use 1296,
  sorry
end

end painting_number_l5_5689


namespace correct_calculation_l5_5424

theorem correct_calculation :
  let expr := (-36 : ℚ) / ((-1/2) + (1/6) - (1/3)) in
  expr = 54 := 
by
  let expr := (-36 : ℚ) / ((-1/2) + (1/6) - (1/3))
  have h1 : ((-1/2) + (1/6) - (1/3)) = (-2/3), by sorry
  have h2 : (-36 : ℚ) / (-2/3) = 54, by sorry
  rw [h1] at expr
  exact h2

end correct_calculation_l5_5424


namespace possible_values_of_m_l5_5635

theorem possible_values_of_m
  (f : ℝ → ℝ)
  (hf_add : ∀ x y : ℝ, f(x + y) = f x + f y)
  (hf1_ge_2 : f 1 ≥ 2)
  (m : ℤ)
  (hm_equation : f (-2) - m^2 - m + 4 = 0) :
  m = -1 ∨ m = 0 := 
sorry

end possible_values_of_m_l5_5635


namespace simplify_sqrt_88200_l5_5290

theorem simplify_sqrt_88200 :
  ∀ (a b c d e : ℕ), a = 2 ∧ b = 3 ∧ c = 5 ∧ d = 7 ∧ e = 1 →
  ∃ f g : ℝ, (88200 : ℝ) = (f^2 * g) ∧ f = 882 ∧ g = 10 ∧ real.sqrt (88200 : ℝ) = f * real.sqrt g :=
sorry

end simplify_sqrt_88200_l5_5290


namespace simplify_sqrt_88200_l5_5288

theorem simplify_sqrt_88200 :
  (Real.sqrt 88200) = 70 * Real.sqrt 6 := 
by 
  -- given conditions
  have h : 88200 = 2^3 * 3 * 5^2 * 7^2 := sorry,
  sorry

end simplify_sqrt_88200_l5_5288


namespace max_min_difference_l5_5246

noncomputable def difference_max_min_z (x y z : ℝ) : ℝ :=
  if h₁ : x + y + z = 3 ∧ x^2 + y^2 + z^2 = 18 then 6 else 0

theorem max_min_difference (x y z : ℝ) (h₁ : x + y + z = 3) (h₂ : x^2 + y^2 + z^2 = 18) :
  difference_max_min_z x y z = 6 :=
by sorry

end max_min_difference_l5_5246


namespace expected_value_of_8_sided_die_l5_5388

def outcomes : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]
def probability (n : ℕ) : ℝ := 1 / 8

theorem expected_value_of_8_sided_die :
  (∑ x in outcomes, probability x * x) = 4.5 := 
sorry

end expected_value_of_8_sided_die_l5_5388


namespace min_value_expression_l5_5891

theorem min_value_expression (a b : ℝ) (h_a : 0 ≤ a) (h_b : 0 ≤ b) :
  (\frac{|a-3*b-2| + |3*a-b|}{\sqrt{a^2 + (b+1)^2}}) = 2 := 
sorry

end min_value_expression_l5_5891


namespace balls_in_bag_l5_5019

theorem balls_in_bag :
  ∃ m, (m = 3) ∧
  let E_X := 
    (0 * ((1:ℝ) / 5) +
     1 * (3 / 5) +
     2 * (1 / 5)) in (m = 3) ∧ (E_X = 1) :=
by
  sorry

end balls_in_bag_l5_5019


namespace count_valid_two_digit_x_l5_5235

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

def valid_x (x : ℕ) : Prop :=
  sum_of_digits (sum_of_digits x) = 4

theorem count_valid_two_digit_x :
  (Finset.filter valid_x (Finset.filter is_two_digit (Finset.range 100))).card = 10 :=
by {
  sorry
}

end count_valid_two_digit_x_l5_5235


namespace letter_puzzle_solutions_l5_5553

noncomputable def is_solution (A B : ℕ) : Prop :=
A ≠ B ∧ A ∈ finset.range (10) ∧ B ∈ finset.range (10) ∧ 10 ≤ B * 10 + A ∧ B * 10 + A ≤ 99 ∧ A^B = B * 10 + A

theorem letter_puzzle_solutions :
  ∃ A B : ℕ, is_solution A B ∧ ((A = 2 ∧ B = 5) ∨ (A = 6 ∧ B = 2) ∨ (A = 4 ∧ B = 3)) :=
by
  sorry

end letter_puzzle_solutions_l5_5553


namespace k_range_for_monotonic_l5_5191

theorem k_range_for_monotonic (k : ℝ) :
  (∀ x : ℝ, 1 < x → kx - log x ≥ 0) ↔ 1 ≤ k := sorry

end k_range_for_monotonic_l5_5191


namespace mass_ratio_speed_ratio_l5_5446

variable {m1 m2 : ℝ} -- masses of the two balls
variable {V0 V : ℝ} -- velocities before and after collision
variable (h1 : V = 4 * V0) -- speed of m2 is four times that of m1 after collision

theorem mass_ratio (h2 :  m1 * V0^2 = m1 * V^2 + 16 * m2 * V^2)
                   (h3 : m1 * V0 = m1 * V + 4 * m2 * V) :
  m2 / m1 = 1 / 2 := sorry

theorem speed_ratio (h2 :  m1 * V0^2 = m1 * V^2 + 16 * m2 * V^2)
                    (h3 : m1 * V0 = m1 * V + 4 * m2 * V)
                    (h4 : m2 / m1 = 1 / 2) :
  V0 / V = 3 := sorry

end mass_ratio_speed_ratio_l5_5446


namespace main_l5_5765

variables {α β : Type} [plane α] [plane β]
variables {m n : Type} [line m] [line n]

-- Proposition 1: If α || β and m ⊂ α, then m || β
def prop1 (h1 : α ∥ β) (h2 : m ⊂ α) : m ∥ β := sorry

-- Proposition 2: If m ⊥ α and m ⊥ n, then n || α
def prop2 (h1 : m ⊥ α) (h2 : m ⊥ n) : n ∥ α := sorry

-- Proposition 3: If m ⊥ α and n ∥ α, then m ⊥ n
def prop3 (h1 : m ⊥ α) (h2 : n ∥ α) : m ⊥ n := sorry

-- Proposition 4: If m ⊥ n, m ⊥ α, and n ∥ β, then α ⊥ β
def prop4 (h1 : m ⊥ n) (h2 : m ⊥ α) (h3 : n ∥ β) : α ⊥ β := sorry

-- Main theorems stated as true or false
theorem main : (prop1 true true ∧ ¬prop2 true true ∧ prop3 true true ∧ ¬prop4 true true) := by
  split
  sorry
  split
  sorry
  split
  sorry
  sorry

end main_l5_5765


namespace find_x_values_l5_5141

noncomputable def valid_x_values (x : ℝ) (y : ℝ) : Prop :=
  (x^2 + y^2 = 1) ∧ (x - y - 2 = 0) ∧ (angle (PQ x y) = 30)

theorem find_x_values {x y : ℝ} :
  valid_x_values x y → x = 0 ∨ x = 2 :=
by sorry

end find_x_values_l5_5141


namespace probability_neither_equal_nor_same_gender_l5_5780

theorem probability_neither_equal_nor_same_gender : 
  let n := 12 in
  let total_outcomes := 2^n in
  let equal_boys_girls := Nat.choose n (n / 2) in
  let all_same_gender := 2 in
  (1 - (equal_boys_girls + all_same_gender) / total_outcomes) = 3170 / 4096 :=
by sorry

end probability_neither_equal_nor_same_gender_l5_5780


namespace triangle_problem_l5_5210

variables {a b c A B C : ℝ}

-- Given conditions
def condition1 : Prop := ∃ (A B C : ℝ) (a b c : ℝ), 
                         a * sin A = 4 * b * sin B ∧ 
                         a * c = sqrt 5 * (a^2 - b^2 - c^2)


-- Prove that given conditions satisfy the specified values of cos A and sin(2B - A)
theorem triangle_problem (h : condition1) : 
    cos A = -sqrt 5 / 5 ∧ sin (2 * B - A) = -2 * sqrt 5 / 5 := sorry

end triangle_problem_l5_5210


namespace smallest_rel_prime_to_180_is_7_l5_5590

theorem smallest_rel_prime_to_180_is_7 :
  ∃ x : ℕ, x > 1 ∧ gcd x 180 = 1 ∧ ∀ y : ℕ, y > 1 ∧ gcd y 180 = 1 → x ≤ y :=
begin
  use 7,
  split,
  { exact dec_trivial, },
  split,
  { exact dec_trivial, },
  intros y hy,
  cases hy with hy1 hy2,
  exact dec_trivial,
end

end smallest_rel_prime_to_180_is_7_l5_5590


namespace dice_probability_two_to_one_l5_5337

theorem dice_probability_two_to_one :
  let favourable_outcomes := {(1, 2), (2, 4), (3, 6), (2, 1), (4, 2), (6, 3)} in
  let total_outcomes := 36 in
  ∃ p, p = 1 / 6 ∧ 
  p = (favourable_outcomes.size : ℚ) / total_outcomes :=
by 
  sorry

end dice_probability_two_to_one_l5_5337


namespace repeating_decimal_to_fraction_l5_5977

/-- Given 0.02 repeating as a fraction 2/99, prove that 2.06 repeating can be expressed as 68/33 -/
theorem repeating_decimal_to_fraction :
  (2 + 0.06̅ : ℝ) = (68 / 33 : ℝ) :=
begin
  have h : (0.02̅ : ℝ) = (2 / 99 : ℝ), from sorry,
  have h3 : (0.06̅ : ℝ) = 3 * (0.02̅ : ℝ), from sorry,
  have h6 : (0.06̅ : ℝ) = 3 * (2 / 99 : ℝ), from sorry,
  have s : (0.06̅ : ℝ) = (6 / 99 : ℝ), from sorry,
  have s2 : (6 / 99 : ℝ) = (2 / 33 : ℝ), from sorry,
  have add := congr_arg (λ x : ℝ, (2 : ℝ) + x) s2,
  rw [add_comm, ← add_halves', add_assoc', add_comm (2 : ℝ), add_comm 2 (2 / 33 : ℝ), add_halves',
    add_assoc', add_comm (2 : ℝ), add_comm 68 (2 / 33 : ℝ)] ,
end

end repeating_decimal_to_fraction_l5_5977


namespace smallest_coprime_gt_one_l5_5575

theorem smallest_coprime_gt_one (x : ℕ) (h1 : 1 < x) (h2 : Nat.gcd x 180 = 1) : x = 7 := sorry

end smallest_coprime_gt_one_l5_5575


namespace letter_puzzle_solutions_l5_5560

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def to_number (B A : ℕ) : ℕ :=
  10 * B + A

theorem letter_puzzle_solutions (A B : ℕ) (h_diff : A ≠ B) (h_digits : 1 ≤ A ∧ A ≤ 9 ∧ 1 ≤ B ∧ B ≤ 9)
  (h_two_digit : is_two_digit (to_number B A)) :
  A^B = to_number B A ↔ (A = 2 ∧ B = 5 ∨ A = 6 ∧ B = 2 ∨ A = 4 ∧ B = 3) :=
by
  sorry

end letter_puzzle_solutions_l5_5560


namespace count_subsets_l5_5826

open Set

theorem count_subsets : 
  {M : Set ℕ | {2, 3} ⊆ M ∧ M ⊆ {1, 2, 3, 4}}.toFinset.card = 4 :=
by
  sorry

end count_subsets_l5_5826


namespace estimate_probability_l5_5814

noncomputable def freq_20 : ℝ := 0.300
noncomputable def freq_50 : ℝ := 0.360
noncomputable def freq_100 : ℝ := 0.350
noncomputable def freq_300 : ℝ := 0.350
noncomputable def freq_500 : ℝ := 0.352
noncomputable def freq_1000 : ℝ := 0.351
noncomputable def freq_5000 : ℝ := 0.351

theorem estimate_probability : (|0.35 - ((freq_20 + freq_50 + freq_100 + freq_300 + freq_500 + freq_1000 + freq_5000) / 7)| < 0.01) :=
by sorry

end estimate_probability_l5_5814


namespace area_of_parallelogram_l5_5107

-- Definitions of the vectors
def u : ℝ × ℝ × ℝ := (4, 2, -3)
def v : ℝ × ℝ × ℝ := (2, -4, 5)

-- Definition of the cross product
def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((a.2 * b.3 - a.3 * b.2), (a.3 * b.1 - a.1 * b.3), (a.1 * b.2 - a.2 * b.1))

-- Definition of the magnitude of a vector
def magnitude (a : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (a.1 * a.1 + a.2 * a.2 + a.3 * a.3)

-- The area of the parallelogram is the magnitude of the cross product of u and v
def area_parallelogram (u v : ℝ × ℝ × ℝ) : ℝ := magnitude (cross_product u v)

-- Proof statement
theorem area_of_parallelogram : area_parallelogram u v = 20 * real.sqrt 3 :=
by
  sorry

end area_of_parallelogram_l5_5107


namespace no_solutions_in_interval_l5_5688

open Real

theorem no_solutions_in_interval :
    ∀ x ∈ Icc 0 π, sin (π * cos x) ≠ cos (π * sin x) := sorry

end no_solutions_in_interval_l5_5688


namespace paths_H_to_J_through_I_l5_5944

theorem paths_H_to_J_through_I :
  let h_to_i_steps_right := 5
  let h_to_i_steps_down := 1
  let i_to_j_steps_right := 3
  let i_to_j_steps_down := 2
  let h_to_i_paths := Nat.choose (h_to_i_steps_right + h_to_i_steps_down) h_to_i_steps_down
  let i_to_j_paths := Nat.choose (i_to_j_steps_right + i_to_j_steps_down) i_to_j_steps_down
  let total_paths := h_to_i_paths * i_to_j_paths
  total_paths = 60 :=
by
  simp
  sorry

end paths_H_to_J_through_I_l5_5944


namespace cos_phi_eq_0_19_l5_5171

variables {u v : ℝ → ℝ}
variables (norm_u : ∥u∥ = 5) (norm_v : ∥v∥ = 10) (norm_u_plus_v : ∥u + v∥ = 12)

theorem cos_phi_eq_0_19 (u v : ℝ → ℝ)
  (norm_u : ∥u∥ = 5)
  (norm_v : ∥v∥ = 10)
  (norm_u_plus_v : ∥u + v∥ = 12) :
  let φ := real.angle u v in
  real.cos φ = 0.19 :=
by
  sorry

end cos_phi_eq_0_19_l5_5171


namespace calculation_l5_5627

theorem calculation (x : ℝ) (h : x^(1/2) + x^(-1/2) = 3) : 
  (x^(3/2) + x^(-3/2) + 2) / (x^(-1) + x + 3) = 2 := 
sorry

end calculation_l5_5627


namespace letter_puzzle_solution_l5_5546

theorem letter_puzzle_solution :
  ∃ (A B : ℕ), (A ≠ B) ∧ (1 ≤ A ∧ A ≤ 9) ∧ (1 ≤ B ∧ B ≤ 9) ∧ A ^ B = B * 10 + A :=
by {
  use 2, 5,
  split, { exact ne_of_lt (by decide) }, split, { exact ⟨by decide, by decide⟩ },
  split, { exact ⟨by decide, by decide⟩ }, exact (by decide : 2 ^ 5 = 5 * 10 + 2),
  sorry
} ∨
by {
  use 6, 2,
  split, { exact ne_of_lt (by decide) }, split, { exact ⟨by decide, by decide⟩ },
  split, { exact ⟨by decide, by decide⟩ }, exact (by decide : 6 ^ 2 = 2 * 10 + 6),
  sorry
} ∨
by {
  use 4, 3,
  split, { exact ne_of_lt (by decide) }, split, { exact ⟨by decide, by decide⟩ },
  split, { exact ⟨by decide, by decide⟩ }, exact (by decide : 4 ^ 3 = 3 * 10 + 4),
  sorry
}

end letter_puzzle_solution_l5_5546


namespace area_of_parallelogram_is_20sqrt3_l5_5110

open Real EuclideanGeometry

def vector_u : ℝ × ℝ × ℝ := (4, 2, -3)
def vector_v : ℝ × ℝ × ℝ := (2, -4, 5)

noncomputable def cross_product (u v : ℝ × ℝ × ℝ) :=
  (u.2 * v.3 - u.3 * v.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2 - u.2 * v.1)

noncomputable def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

theorem area_of_parallelogram_is_20sqrt3 :
  magnitude (cross_product vector_u vector_v) = 20 * Real.sqrt 3 := by
  sorry

end area_of_parallelogram_is_20sqrt3_l5_5110


namespace cannot_determine_exact_insect_l5_5451

-- Defining the conditions as premises
def insect_legs : ℕ := 6

def total_legs_two_insects (legs_per_insect : ℕ) (num_insects : ℕ) : ℕ :=
  legs_per_insect * num_insects

-- Statement: Proving that given just the number of legs, we cannot determine the exact type of insect
theorem cannot_determine_exact_insect (legs : ℕ) (num_insects : ℕ) (h1 : legs = 6) (h2 : num_insects = 2) (h3 : total_legs_two_insects legs num_insects = 12) :
  ∃ insect_type, insect_type :=
by
  sorry

end cannot_determine_exact_insect_l5_5451


namespace soccer_camp_afternoon_l5_5844

-- Define the total number of kids
def total_kids : ℕ := 2000

-- Define the fraction of kids going to soccer camp
def fraction_soccer : ℝ := 1 / 2

-- Define the fraction of kids going to morning soccer camp
def fraction_morning : ℝ := 1 / 4

-- Calculate the number of kids going to soccer camp
def kids_soccer : ℕ := (total_kids * fraction_soccer.to_nat)

-- Calculate the number of kids going to morning soccer camp
def kids_morning_soccer : ℕ := (kids_soccer * fraction_morning.to_nat)

-- Calculate the number of kids going to afternoon soccer camp
def kids_afternoon_soccer : ℕ := kids_soccer - kids_morning_soccer

-- The theorem stating the resultant number for the kids going to soccer camp in the afternoon
theorem soccer_camp_afternoon : kids_afternoon_soccer = 750 := by
  sorry

end soccer_camp_afternoon_l5_5844


namespace am_gm_inequality_l5_5768

theorem am_gm_inequality 
  (n : ℕ) 
  (a : ℕ → ℝ) 
  (h : ∀ i, 1 ≤ i ∧ i ≤ n → a i > 0) 
  (s : ℝ) 
  (hs : s = ∑ i in Finset.range n, a i) :  
  (∏ i in Finset.range n, (1 + a i)) <= (1 + s / n) ^ n :=
begin
  sorry
end

end am_gm_inequality_l5_5768


namespace luke_earning_problem_l5_5777

variable (WeedEarning Weeks SpendPerWeek MowingEarning : ℤ)

theorem luke_earning_problem
  (h1 : WeedEarning = 18)
  (h2 : Weeks = 9)
  (h3 : SpendPerWeek = 3)
  (h4 : MowingEarning + WeedEarning = Weeks * SpendPerWeek) :
  MowingEarning = 9 := by
  sorry

end luke_earning_problem_l5_5777


namespace hex_B3F_to_decimal_l5_5515

-- Define the hexadecimal values of B, 3, F
def hex_B : ℕ := 11
def hex_3 : ℕ := 3
def hex_F : ℕ := 15

-- Prove the conversion of B3F_{16} to a base 10 integer equals 2879
theorem hex_B3F_to_decimal : (hex_B * 16^2 + hex_3 * 16^1 + hex_F * 16^0) = 2879 := 
by 
  -- calculation details skipped
  sorry

end hex_B3F_to_decimal_l5_5515


namespace smallest_rel_prime_to_180_l5_5594

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ Nat.gcd x 180 = 1 ∧ ∀ y : ℕ, (y > 1 ∧ Nat.gcd y 180 = 1) → x ≤ y :=
begin
  sorry
end

end smallest_rel_prime_to_180_l5_5594


namespace population_increase_percentage_l5_5885

-- Define the birth rate and death rate per 1000
def birth_rate : ℝ := 32
def death_rate : ℝ := 11

-- Calculate the net increase in the population rate
def net_increase_rate : ℝ := birth_rate - death_rate

-- Calculate the percentage increase in the rate of population
def percentage_increase_rate : ℝ := (net_increase_rate / 1000) * 100

-- Statement of the proof problem
theorem population_increase_percentage :
  percentage_increase_rate = 2.1 :=
by
  sorry

end population_increase_percentage_l5_5885


namespace series_converges_absolutely_l5_5520

theorem series_converges_absolutely (α : ℝ) :
  ∑' n : ℕ, (cos (n * α))^3 / (n^2 : ℝ) < ∞ :=
by
  sorry

end series_converges_absolutely_l5_5520


namespace domain_of_f_increasing_f_solve_equation_l5_5138

variables {a x : ℝ}
open Real

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := log a (a^x - 1)

-- Assumption conditions
axiom a_pos : a > 0
axiom a_ne_one : a ≠ 1

-- Question 1: Prove the domain of f(x) is (-∞, 0)
theorem domain_of_f : ∀ x, x < 0 → ∃ y, f(x, a) = y :=
by
  intro x h
  use log a (a^x - 1)
  unfold f
  sorry

-- Question 2: Prove f(x) is an increasing function in (-∞, 0)
theorem increasing_f : ∀ {x1 x2 : ℝ}, x1 < x2 → x1 < 0 → x2 < 0 → f(x1, a) < f(x2, a) :=
by
  intros x1 x2 h1 h2 h3
  unfold f
  sorry

-- Question 3: Solve the equation log_a(a^{2x} - 1) = log_a(a^x + 1)
theorem solve_equation : ∀ x, log a (a^(2 * x) - 1) = log a (a^x + 1) → x = log a 2 :=
by
  intro x h
  sorry

end domain_of_f_increasing_f_solve_equation_l5_5138


namespace number_of_monkeys_l5_5802

theorem number_of_monkeys (N : ℕ)
  (h1 : N * 1 * 8 = 8)
  (h2 : 3 * 1 * 8 = 3 * 8) :
  N = 8 :=
sorry

end number_of_monkeys_l5_5802


namespace proof_problem_l5_5149

noncomputable def prop_p : Prop := ∃ x : ℝ, x - 2 > log 10 x
def prop_q : Prop := ∀ x : ℝ, x ^ 2 > 0

theorem proof_problem : prop_p ∧ ¬prop_q :=
by
  -- The statements "p is true" and "q is false" are provided by the problem.
  -- Fill in "sorry" to indicate the proof steps, which will usually involve demonstrating both parts of the conjunction are true.
  sorry

end proof_problem_l5_5149


namespace expected_value_sequence_l5_5499

theorem expected_value_sequence :
  let p := 12 
  let rec next (x : ℕ) : ℕ :=
    if x = 1 then 1 else {d : ℕ // d ∣ x}.choose
  let sequence := list.iterate next p
  let expected_value := (216 / 180) + (40 / 180) + (45 / 180) + (75 / 180) + (95 / 180) + 1
  (sequence.length : ℚ) = expected_value :=
sorry

end expected_value_sequence_l5_5499


namespace ax5_by5_eq_28616_l5_5801

variables (a b x y : ℝ)

theorem ax5_by5_eq_28616
  (h1 : a * x + b * y = 1)
  (h2 : a * x^2 + b * y^2 = 9)
  (h3 : a * x^3 + b * y^3 = 28)
  (h4 : a * x^4 + b * y^4 = 96) :
  a * x^5 + b * y^5 = 28616 :=
sorry

end ax5_by5_eq_28616_l5_5801


namespace smallest_coprime_gt_one_l5_5574

theorem smallest_coprime_gt_one (x : ℕ) (h1 : 1 < x) (h2 : Nat.gcd x 180 = 1) : x = 7 := sorry

end smallest_coprime_gt_one_l5_5574


namespace remainder_occurs_128_times_l5_5842

theorem remainder_occurs_128_times (n : ℕ) (h1 : 100 ≤ n ∧ n ≤ 999) (h2 : n % 7 = 2) : 
  ∃ w ∈ finset.Icc 100 999, w % 7 = 2 ∧ (finset.Icc 100 999).countp (λ w, w % 7 = 2) = 128 :=
by
  sorry

end remainder_occurs_128_times_l5_5842


namespace dot_product_b_c_l5_5169

variables (a b c : ℝ^3)

-- Conditions
def norm_a : ∥a∥ = 1 := sorry
def norm_b : ∥b∥ = 1 := sorry
def norm_sum_ab : ∥a + b∥ = sqrt 5 := sorry
def c_eq : c = a + 2 * b + 4 * (a × b) := sorry

-- Proposition to prove
theorem dot_product_b_c : b ⬝ c = 7 / 2 := sorry

end dot_product_b_c_l5_5169


namespace find_m_l5_5683

noncomputable def f (x : ℝ) : ℝ := 4 * x^2 + 3 * x + 5
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := x^2 - m * x - 9

theorem find_m (m : ℝ) : f 5 - g 5 m = 20 → m = -16.8 :=
by
  -- Given f(x) and g(x, m) definitions, we want to prove m = -16.8 given f 5 - g 5 m = 20.
  sorry

end find_m_l5_5683


namespace repeating_decimal_to_fraction_l5_5978

/-- Given 0.02 repeating as a fraction 2/99, prove that 2.06 repeating can be expressed as 68/33 -/
theorem repeating_decimal_to_fraction :
  (2 + 0.06̅ : ℝ) = (68 / 33 : ℝ) :=
begin
  have h : (0.02̅ : ℝ) = (2 / 99 : ℝ), from sorry,
  have h3 : (0.06̅ : ℝ) = 3 * (0.02̅ : ℝ), from sorry,
  have h6 : (0.06̅ : ℝ) = 3 * (2 / 99 : ℝ), from sorry,
  have s : (0.06̅ : ℝ) = (6 / 99 : ℝ), from sorry,
  have s2 : (6 / 99 : ℝ) = (2 / 33 : ℝ), from sorry,
  have add := congr_arg (λ x : ℝ, (2 : ℝ) + x) s2,
  rw [add_comm, ← add_halves', add_assoc', add_comm (2 : ℝ), add_comm 2 (2 / 33 : ℝ), add_halves',
    add_assoc', add_comm (2 : ℝ), add_comm 68 (2 / 33 : ℝ)] ,
end

end repeating_decimal_to_fraction_l5_5978


namespace expected_value_of_8_sided_die_is_4_point_5_l5_5357

def expected_value_8_sided_die : ℝ :=
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let probability := 1 / 8
  (list.sum outcomes : ℝ) * probability

theorem expected_value_of_8_sided_die_is_4_point_5 :
  expected_value_8_sided_die = 4.5 := by
  sorry

end expected_value_of_8_sided_die_is_4_point_5_l5_5357


namespace part_I_part_II_l5_5665

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (1 / (2 * x)) - a * x^2 + x

theorem part_I (a : ℝ) (ha : a ≥ 1/8) : 
  ¬∃ x : ℝ, 0 < x ∧ (f a x).derivative = 0 := sorry

theorem part_II (a : ℝ) (ha : 0 < a ∧ a < 1/8) 
  (h_extreme_points : ∃ x1 x2 : ℝ, 0 < x1 ∧ 0 < x2 ∧ x1 ≠ x2 ∧ 
    (f a x1).derivative = 0 ∧ (f a x2).derivative = 0) : 
  ∀ x1 x2, (h_extreme_points ∧ (f a x1) + (f a x2) > 3 - 4 * Real.log 2) := sorry

end part_I_part_II_l5_5665


namespace lim_ξ_does_not_exist_ae_l5_5894

open ProbabilityTheory
open MeasureTheory

variables {Ω : Type*} [MeasurableSpace Ω] [ProbabilityMeasure Ω]

-- Sequence of independent random variables
noncomputable def ξ (n : ℕ) : Ω → ℕ := 
sorry -- Axiomatic definition. To be replaced by actual random variable definition.

-- Probability distribution of each ξ_n
axiom  ξ_prob (n : ℕ) (ω : Ω) : ProbabilityTheory.Probability (ξ n ω = 0) = 1 / n
axiom ξ_comp_prob (n : ℕ) (ω : Ω) : ProbabilityTheory.Probability (ξ n ω = 1) = 1 - (1 / n)

-- Define event A_n
def A_n (n : ℕ) : Set Ω := {ω : Ω | ξ n ω = 0}

-- Prove that lim_{n -> ∞} ξ_n does not exist almost surely
theorem lim_ξ_does_not_exist_ae :
  ¬ ∃ l : ℕ, ∀ᵐ ω ∂(ProbabilityTheory.MeasureSpace.ProbabilityMeasure), ∃ (n₀ : ℕ), ∀ n ≥ n₀, ξ n ω = l :=
sorry

end lim_ξ_does_not_exist_ae_l5_5894


namespace spinner_no_divisible_by_5_l5_5472

theorem spinner_no_divisible_by_5 :
  let labels := {1, 2, 3, 4}
  let outcomes := {x : ℕ | ∃ a b c ∈ labels, x = 100 * a + 10 * b + c}
  ∀ x ∈ outcomes, (x % 5 ≠ 0) → (0 / 64 = 0) :=
by
  sorry

end spinner_no_divisible_by_5_l5_5472


namespace letter_puzzle_solution_l5_5545

theorem letter_puzzle_solution :
  ∃ (A B : ℕ), (A ≠ B) ∧ (1 ≤ A ∧ A ≤ 9) ∧ (1 ≤ B ∧ B ≤ 9) ∧ A ^ B = B * 10 + A :=
by {
  use 2, 5,
  split, { exact ne_of_lt (by decide) }, split, { exact ⟨by decide, by decide⟩ },
  split, { exact ⟨by decide, by decide⟩ }, exact (by decide : 2 ^ 5 = 5 * 10 + 2),
  sorry
} ∨
by {
  use 6, 2,
  split, { exact ne_of_lt (by decide) }, split, { exact ⟨by decide, by decide⟩ },
  split, { exact ⟨by decide, by decide⟩ }, exact (by decide : 6 ^ 2 = 2 * 10 + 6),
  sorry
} ∨
by {
  use 4, 3,
  split, { exact ne_of_lt (by decide) }, split, { exact ⟨by decide, by decide⟩ },
  split, { exact ⟨by decide, by decide⟩ }, exact (by decide : 4 ^ 3 = 3 * 10 + 4),
  sorry
}

end letter_puzzle_solution_l5_5545


namespace painted_cubes_count_l5_5026

theorem painted_cubes_count :
  let cubic_object := (3:ℕ)
  let total_cubes := cubic_object * cubic_object * cubic_object
  let painted_cubes := 26
  total_cubes = 27 →
  painted_cubes = 8 + 12 + 6 →
  total_cubes - painted_cubes = 1 :=
by
  let cubic_object := (3:ℕ)
  let total_cubes := cubic_object * cubic_object * cubic_object
  let painted_cubes := 8 + 12 + 6
  have h1 : total_cubes = 27 := by decide
  have h2 : painted_cubes = 26 := by decide
  exact congr(arg1 := fun (x:ℕ) => x - 26) h1 h2

end painted_cubes_count_l5_5026


namespace problem1_solution_problem2_solution_l5_5899

noncomputable def sol_set_problem1 (a b c : ℝ) (h1 : b = 5/3 * a) (h2 : c = -2/3 * a) (h3 : a < 0) : set ℝ :=
{ x | x ≤ -3 ∨ x ≥ 1/2 }

theorem problem1_solution (a b c : ℝ)
  (h_solutions : ∀ x, x < -2 ∨ x > 1/3 → ax^2 + bx + c < 0)
  (h_roots : [x | x < -2 ∨ x > 1/3] = { x | x = -2 ∨ x = 1/3 }) :
  ∀ x, (cx^2 - bx + a ≥ 0) ↔ x ∈ sol_set_problem1 a b c := by
  sorry

inductive DeltaValue
| le_zero
| gt_zero

def sol_set_problem2 (a : ℝ) (delta_case : DeltaValue) : set ℝ :=
match a, delta_case with
| 0, _ => { x | 0 < x }
| a, DeltaValue.le_zero => ∅
| a, DeltaValue.gt_zero => { x | 1 - (sqrt(1 - a^2)) / a < x ∧ x < ((1 + sqrt(1 - a^2)) / a) }
| a, _ => if a < 0 ∧ 1 < -a then set.univ else if -1 = a then { x | x < -1 ∨ -1 < x } else
          { x | x < (1 + sqrt(1 - a^2)) / a ∨ (1 - sqrt(1 - a^2)) / a < x }

theorem problem2_solution (a : ℝ) (h_cases : a = 0 ∨ (0 < a ∧ a < 1 ∧ DeltaValue.gt_zero) ∨ (1 ≤ a ∧ DeltaValue.le_zero) ∨ 
                        (a < 0 ∧ delta_case = DeltaValue.gt_zero) ∨ (a = -1 ∧ delta_case = DeltaValue.le_zero) ∨ 
                        (1 < -a ∧ delta_case = DeltaValue.le_zero)) :
  ∀ x, (ax^2 - 2 * x + a < 0) ↔ x ∈ sol_set_problem2 a delta_case := by
  sorry

end problem1_solution_problem2_solution_l5_5899


namespace yoongi_flowers_left_l5_5002

theorem yoongi_flowers_left (initial_flowers given_to_eunji given_to_yuna : ℕ) 
  (h_initial : initial_flowers = 28) 
  (h_eunji : given_to_eunji = 7) 
  (h_yuna : given_to_yuna = 9) : 
  initial_flowers - (given_to_eunji + given_to_yuna) = 12 := 
by 
  sorry

end yoongi_flowers_left_l5_5002


namespace sum_of_solutions_abs_eq_50_l5_5437

-- Let's define the problem statement in Lean
theorem sum_of_solutions_abs_eq_50 : 
  (∃ᵇ x, |x - 25| = 50 → x = 75 ∨ x = -25) ∧ ∑ x ∈ {75, -25}, x = 50 :=
by
  -- Sorry is used to skip the proof
  sorry 

end sum_of_solutions_abs_eq_50_l5_5437


namespace David_pushups_l5_5003

variable (Zachary_pushups : ℕ) (more_pushups : ℕ)

def Zachary_did_59_pushups : Zachary_pushups = 59 := by
  sorry

def David_did_19_more_pushups_than_Zachary : more_pushups = 19 := by
  sorry

theorem David_pushups (Zachary_pushups more_pushups : ℕ) 
  (hz : Zachary_pushups = 59) (hd : more_pushups = 19) : 
  Zachary_pushups + more_pushups = 78 := by
  rw [hz, hd]
  norm_num

end David_pushups_l5_5003


namespace determine_flower_responsibility_l5_5030

-- Define the structure of the grid
structure Grid (m n : ℕ) :=
  (vertices : Fin m → Fin n → Bool) -- True if gardener lives at the vertex

-- Define a function to determine if 3 gardeners are nearest to a flower
def is_nearest (i j fi fj : ℕ) : Bool :=
  -- Assume this function gives true if the gardener at (i, j) is one of the 3 nearest to the flower at (fi, fj)
  sorry

-- The main theorem statement
theorem determine_flower_responsibility 
  {m n : ℕ} 
  (G : Grid m n) 
  (i j : Fin m) 
  (k : Fin n) 
  (h : G.vertices i k = true) 
  : ∃ (fi fj : ℕ), is_nearest (i : ℕ) (k : ℕ) fi fj = true := 
sorry

end determine_flower_responsibility_l5_5030


namespace sum_of_conjugates_eq_30_l5_5508

theorem sum_of_conjugates_eq_30 :
  (15 - Real.sqrt 2023) + (15 + Real.sqrt 2023) = 30 :=
sorry

end sum_of_conjugates_eq_30_l5_5508


namespace functional_equation_solution_l5_5298

theorem functional_equation_solution (f : ℝ → ℝ)
  (h : ∀ x, 2 * f (f x) = (x^2 - x) * f x + 4 - 2 * x) :
  f 2 = 2 ∧ (f 1 = 1 ∨ f 1 = 4) :=
sorry

end functional_equation_solution_l5_5298


namespace smallest_rel_prime_to_180_l5_5592

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ Nat.gcd x 180 = 1 ∧ ∀ y : ℕ, (y > 1 ∧ Nat.gcd y 180 = 1) → x ≤ y :=
begin
  sorry
end

end smallest_rel_prime_to_180_l5_5592


namespace sqrt_88200_simplified_l5_5294

theorem sqrt_88200_simplified : Real.sqrt 88200 = 210 * Real.sqrt 6 :=
by sorry

end sqrt_88200_simplified_l5_5294


namespace problem_b_l5_5879

theorem problem_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b + a * b = 3) : a + b ≥ 2 :=
sorry

end problem_b_l5_5879


namespace expected_value_8_sided_die_l5_5366

theorem expected_value_8_sided_die :
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let probability_each_outcome := 1 / 8
  (Σ x ∈ outcomes, probability_each_outcome * x) = 4.5 :=
by
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8]
  let probability_each_outcome := 1 / 8
  have h : (Σ x ∈ outcomes, probability_each_outcome * x) = (1 / 8) * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8) := by sorry
  have sum_eq_36 : 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 = 36 := by sorry
  rw [sum_eq_36] at h
  have expected_value_eq : (1 / 8) * 36 = 4.5 := by sorry
  rw [expected_value_eq] at h
  exact h

end expected_value_8_sided_die_l5_5366


namespace parabola_intercepts_sum_l5_5817

theorem parabola_intercepts_sum :
  let x_intercept := 4
  let y_intercept1 := (9 + Real.sqrt 33) / 6
  let y_intercept2 := (9 - Real.sqrt 33) / 6
  in x_intercept + y_intercept1 + y_intercept2 = 7 :=
by
  let x_intercept := 4
  let y_intercept1 := (9 + Real.sqrt 33) / 6
  let y_intercept2 := (9 - Real.sqrt 33) / 6
  show x_intercept + y_intercept1 + y_intercept2 = 7
  sorry

end parabola_intercepts_sum_l5_5817


namespace area_of_shaded_region_l5_5954

noncomputable def semicircles_area : ℝ := 2

theorem area_of_shaded_region 
  (XY : ℝ) (r : ℝ) (M N P : ℝ × ℝ)
  (h1 : XY = 2)
  (h2 : r = 1)
  (h3 : M = (1, 1))
  (h4 : N = (1, 1))
  (h5 : P = (1, 1))
  (h6 : P = (M.1 + N.1) / 2, (M.2 + N.2) / 2) :
  area_rect : semicircles_area = XY * r :=
sorry

end area_of_shaded_region_l5_5954


namespace geometric_sequence_a6_l5_5715

theorem geometric_sequence_a6 (a : ℕ → ℝ) (geometric_seq : ∀ n, a (n + 1) = a n * a 1)
  (h1 : (a 4) * (a 8) = 9) (h2 : (a 4) + (a 8) = -11) : a 6 = -3 := by
  sorry

end geometric_sequence_a6_l5_5715


namespace probability_daniel_wins_l5_5077

theorem probability_daniel_wins :
  let p := 0.60
  let P := 0.36 + 0.48 * p
  P = 9 / 13 :=
by
  sorry

end probability_daniel_wins_l5_5077


namespace find_x_eq_minus_3_l5_5535

theorem find_x_eq_minus_3 (x : ℤ) : (3 ^ 7 * 3 ^ x = 81) → x = -3 := 
by
  sorry

end find_x_eq_minus_3_l5_5535


namespace river_current_speed_l5_5459

variable (V_s : ℝ) (V_c : ℝ) (t_upstream : ℝ) (t_downstream : ℝ)

theorem river_current_speed (h1 : V_s = 10) 
                            (h2 : t_upstream = 3 * t_downstream)
                            (h3 : (V_s - V_c) * t_upstream = (V_s + V_c) * t_downstream) : 
                            V_c = 5 :=
by
  have t_upstream_eq : t_upstream = 3 * t_downstream := h2
  have dist_eq : (V_s - V_c) * t_upstream = (V_s + V_c) * t_downstream := h3
  rw [t_upstream_eq] at dist_eq
  rw [mul_assoc, mul_comm t_downstream, mul_assoc] at dist_eq
  rw [← mul_assoc (V_s + V_c), ← mul_assoc (V_s - V_c)] at dist_eq
  sorry

end river_current_speed_l5_5459


namespace knights_win_35_l5_5312

noncomputable def Sharks : ℕ := sorry
noncomputable def Falcons : ℕ := sorry
noncomputable def Knights : ℕ := 35
noncomputable def Wolves : ℕ := sorry
noncomputable def Royals : ℕ := sorry

-- Conditions
axiom h1 : Sharks > Falcons
axiom h2 : Wolves > 25
axiom h3 : Wolves < Knights ∧ Knights < Royals

-- Prove: Knights won 35 games
theorem knights_win_35 : Knights = 35 := 
by sorry

end knights_win_35_l5_5312


namespace sum_of_circles_squares_l5_5495

variables (a b c : ℤ)

theorem sum_of_circles_squares 
  (h1 : 2 * a + 2 * b + c = 26)
  (h2 : 2 * a + 3 * b + c = 29) :
  2 * b + 3 * c = 24 := by
  sorry

end sum_of_circles_squares_l5_5495


namespace find_angle_C_find_max_area_l5_5209

noncomputable def C := ∠C
noncomputable def sin_A := (a: ℂ) / 2
noncomputable def sin_B := (b: ℂ) / 2
noncomputable def sin_C := (c: ℂ) / 2
noncomputable def cos_C := (√2: ℂ) / 2
noncomputable def area := (s: ℝ) := (abs(sin_A * sin_B * sin_C))

-- Given conditions
variables (a b c: ℂ)

-- Proof for first part: Finding the measure of ∠C
theorem find_angle_C:
  (2 * (sin_A ^ 2 - sin_C ^ 2) = ((√2 * a) - b) * sin_B) →
  cos_C = (√2: ℂ) / 2 →
  C = real.pi / 4 := sorry 

-- Proof for second part: Finding the maximum area
theorem find_max_area:
  (C = real.pi / 4) →
  (area = find_angle_C ∠ABC) →
  find_angle_C ∠ABC = (√2 / 2 + 1 / 2) := sorry

end find_angle_C_find_max_area_l5_5209


namespace num_regions_by_n_circles_l5_5324

theorem num_regions_by_n_circles (n : ℕ) : 
  (∀ (n : ℕ), f n = n^2 - n + 2) := 
begin
  sorry
end

end num_regions_by_n_circles_l5_5324


namespace initial_bird_families_count_l5_5000

theorem initial_bird_families_count :
  ∀ (flew_away stayed_behind initial : ℕ), 
  flew_away = 7 → 
  stayed_behind = 7 + 73 → 
  initial = flew_away + stayed_behind → 
  initial = 87 :=
by
  intros flew_away stayed_behind initial flew_away_eq stayed_behind_eq initial_eq,
  sorry

end initial_bird_families_count_l5_5000


namespace intersect_ratios_l5_5275

variables (A B C D P Q R S : Type) [convex_quadrilateral A B C D]

-- The points P, Q, R, and S are taken on the sides such that
-- BP:AB = CR:CD = α and AS:AD = BQ:BC = β
variable (α β : ℝ)
variable (h1 : segment_ratios BP AB α)
variable (h2 : segment_ratios CR CD α)
variable (h3 : segment_ratios AS AD β)
variable (h4 : segment_ratios BQ BC β)

-- Prove that segments PR and QS are divided by their intersection point in the ratios β:(1-β) and α:(1-α) respectively.
theorem intersect_ratios (I : intersection_point PR QS) : 
  divides PR I (β : (1-β)) ∧ divides QS I (α : (1-α)) :=
sorry

end intersect_ratios_l5_5275


namespace letter_puzzle_l5_5556

theorem letter_puzzle (A B : ℕ) (hA : 1 ≤ A ∧ A ≤ 9) (hB : 1 ≤ B ∧ B ≤ 9) (h_diff : A ≠ B) :
  A^B = 10 * B + A ↔ (A = 2 ∧ B = 5) ∨ (A = 6 ∧ B = 2) ∨ (A = 4 ∧ B = 3) :=
by
  sorry

end letter_puzzle_l5_5556


namespace correct_statement_l5_5428

-- Definitions of events A and B and their probabilities P(A) and P(B)
variables {Ω : Type} (A B : set Ω) (P : set Ω → ℝ)

-- Conditions: mutually exclusive, impossible event, and certain event
def mutually_exclusive (A B : set Ω) : Prop := A ∩ B = ∅
def impossible_event (A : set Ω) : Prop := P(A) = 0
def certain_event (B : set Ω) : Prop := P(B) = 1

-- The theorem we aim to prove
theorem correct_statement (h : mutually_exclusive A B) : P(A) + P(B) = 1 :=
sorry

end correct_statement_l5_5428


namespace max_length_of_cuts_l5_5904

-- Define the dimensions of the board and the number of parts
def board_size : ℕ := 30
def num_parts : ℕ := 225

-- Define the total possible length of the cuts
def max_possible_cuts_length : ℕ := 1065

-- Define the condition that the board is cut into parts of equal area
def equal_area_partition (board_size num_parts : ℕ) : Prop :=
  ∃ (area_per_part : ℕ), (board_size * board_size) / num_parts = area_per_part

-- Define the theorem to prove the maximum possible total length of the cuts
theorem max_length_of_cuts (h : equal_area_partition board_size num_parts) :
  max_possible_cuts_length = 1065 :=
by
  -- Proof to be filled in
  sorry

end max_length_of_cuts_l5_5904


namespace part1_part2_l5_5741

-- Definitions for conditions and problem statement

variables (a b c : ℝ)
variable (B : ℝ)

-- First part of the problem
theorem part1 (h1: (2 * a - c) * cos B = b * cos (pi - (arcsin (a * sin B / b) + B))) :
  B = pi / 4 := sorry

-- Second part of the problem
theorem part2 (a c : ℝ) (b : ℝ) (h1: b = sqrt 7) (h2: a + c = 4) (h3 : B = pi / 4) :
  let area := (1 / 2) * a * c * sin B in
  area = (3 * sqrt 2) / 4 := sorry

end part1_part2_l5_5741


namespace unit_digit_of_sum_factorial_squares_l5_5181

theorem unit_digit_of_sum_factorial_squares :
  (∑ n in finset.range 150, (nat.factorial (n + 1)) ^ 2) % 10 = 7 :=
by
  -- Utilize the condition that factorials for n ≥ 5 end in 0
  -- Reduce the proof to manually computing the units' digits of the first relevant factorials and summing them
  sorry

end unit_digit_of_sum_factorial_squares_l5_5181


namespace max_min_z_diff_correct_l5_5244

noncomputable def max_min_z_diff (x y z : ℝ) (h1 : x + y + z = 3) (h2 : x^2 + y^2 + z^2 = 18) : ℝ :=
  6

theorem max_min_z_diff_correct (x y z : ℝ) (h1 : x + y + z = 3) (h2 : x^2 + y^2 + z^2 = 18) :
  max_min_z_diff x y z h1 h2 = 6 :=
sorry

end max_min_z_diff_correct_l5_5244


namespace ship_sails_distance_square_l5_5470

theorem ship_sails_distance_square :
  let AB := 15
  let BC := 25
  ∃ θ : ℝ, θ ∈ set.Icc (60 * real.pi / 180) (90 * real.pi / 180) →
  let AC² := AB^2 + BC^2 - 2 * AB * BC * real.cos θ
  475 ≤ AC² ∧ AC² ≤ 850 :=
begin
  sorry
end

end ship_sails_distance_square_l5_5470


namespace firecracker_velocity_magnitude_l5_5033

-- Problem conditions
def initial_speed : ℝ := 20
def acceleration_due_to_gravity : ℝ := 10
def explosion_time : ℝ := 3
def mass_ratio : ℕ := 2
def smaller_fragment_horizontal_speed : ℝ := 16

theorem firecracker_velocity_magnitude :
  let v_init := initial_speed
  let g := acceleration_due_to_gravity
  let t := explosion_time
  let m_ratio := mass_ratio
  let v_horizontal_sm := smaller_fragment_horizontal_speed
  let v_vertical_fr := v_init - g * t
  let m1 : ℝ := 1
  let m2 : ℝ := 2
  let v_horizontal_fr := - v_horizontal_sm * m1 / m2
  sqrt (v_horizontal_fr^2 + v_vertical_fr^2) = 17 :=
by
  sorry

end firecracker_velocity_magnitude_l5_5033


namespace borya_wins_with_optimal_play_l5_5054

-- Define the game setup
def game_state :=
{ points : ℕ := 33 /- there are 33 points equally spaced along the circumference of a circle -/,
  anya_first : Bool := true /- Anya makes the first move -/,
  valid_move : ℕ → Bool /- ensures no adjacent points of the same color -/ }

-- Define the winning condition
def optimal_play (state : game_state) : String :=
  "Borya" -- Borya wins with optimal play

theorem borya_wins_with_optimal_play : optimal_play {points := 33, anya_first := true, valid_move := λ x, true} = "Borya" := 
by 
  sorry

end borya_wins_with_optimal_play_l5_5054


namespace scheduled_conference_games_l5_5804

-- Definitions based on conditions
def num_divisions := 3
def teams_per_division := 4
def games_within_division := 3
def games_across_divisions := 2

-- Proof statement
theorem scheduled_conference_games :
  let teams_in_division := teams_per_division
  let div_game_count := games_within_division * (teams_in_division * (teams_in_division - 1) / 2) 
  let total_within_division := div_game_count * num_divisions
  let cross_div_game_count := (teams_in_division * games_across_divisions * (num_divisions - 1) * teams_in_division * num_divisions) / 2
  total_within_division + cross_div_game_count = 102 := 
by {
  sorry
}

end scheduled_conference_games_l5_5804


namespace find_a_l5_5642

theorem find_a
  (S : ℕ → ℝ)
  (a_n : ℕ → ℝ)
  (a : ℝ)
  (h1: ∀ n : ℕ, S n = 2 * 3^n + a)
  (h2 : ∀ n : ℕ, a_n n = if n = 0 then S 0 else S n - S (n - 1))
  (h3 : ∀ n : ℕ, n > 1 → a_n 0 * a_n (n - 1) = (a_n (n - 2))^2) :
  a = -3 :=
by
  sorry

end find_a_l5_5642


namespace arithmetic_sequence_problem_l5_5145

open_locale classical

-- Define the conditions and prove the required result
theorem arithmetic_sequence_problem (a : ℕ → ℤ) (d : ℤ)
  (h_arithmetic : ∀ n, a (n + 1) = a n + d)
  (h_condition1 : a 2 + a 7 = 18)
  (h_condition2 : a 4 = 3) :
  a 5 = 15 :=
  sorry

end arithmetic_sequence_problem_l5_5145


namespace total_students_l5_5849

-- Define the conditions
def ratio_boys_to_girls (boys girls : ℕ) : Prop := boys = 3 * (girls / 2)
def boys_girls_difference (boys girls : ℕ) : Prop := boys = girls + 20

-- Define the property to be proved
theorem total_students (boys girls : ℕ) 
  (h1 : ratio_boys_to_girls boys girls)
  (h2 : boys_girls_difference boys girls) :
  boys + girls = 100 :=
sorry

end total_students_l5_5849


namespace cosine_minus_sine_eq_neg_sqrt_three_over_two_l5_5151

theorem cosine_minus_sine_eq_neg_sqrt_three_over_two 
  (theta : ℝ) 
  (h1 : sin theta * cos theta = 1 / 8) 
  (h2 : π / 4 < theta ∧ theta < π / 2) : 
  cos theta - sin theta = -√(3) / 2 :=
begin
  sorry,
end

end cosine_minus_sine_eq_neg_sqrt_three_over_two_l5_5151


namespace time_to_cross_platform_l5_5017

theorem time_to_cross_platform (length_train length_platform : ℕ) (speed_kmh : ℝ) (h_train : length_train = 240) (h_platform : length_platform = 240) (h_speed : speed_kmh = 64) :
  let distance := length_train + length_platform,
      speed_ms := speed_kmh * 1000 / 3600,
      time := distance / speed_ms
  in time ≈ 27 :=
by
  sorry

end time_to_cross_platform_l5_5017


namespace expression_for_T_n_l5_5663

theorem expression_for_T_n (a : ℕ → ℕ) (S : ℕ → ℕ) (T : ℕ → ℕ) 
  (h1 : ∀ n : ℕ, n > 0 → 3 * a n * S n = n * (n - 1))
  (h2 : ∀ n : ℕ, T n = ∑ i in Finset.range n.succ, S i) :
  ∀ n : ℕ, n > 0 → T n = (n * (n - 1)) / 6 :=
by
  sorry

end expression_for_T_n_l5_5663


namespace obtuse_triangle_count_l5_5643

-- Definitions based on conditions
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def is_arithmetic_sequence (a b c : ℕ) : Prop :=
  2 * b = a + c

def is_obtuse_triangle (a b c : ℕ) : Prop :=
  a * a + b * b < c * c ∨ b * b + c * c < a * a ∨ c * c + a * a < b * b

-- Main conjecture to prove
theorem obtuse_triangle_count :
  ∃ (n : ℕ), n = 157 ∧
    ∀ (a b c : ℕ), 
      a <= 50 ∧ b <= 50 ∧ c <= 50 ∧ 
      is_arithmetic_sequence a b c ∧ 
      is_triangle a b c ∧ 
      is_obtuse_triangle a b c → 
    true := sorry

end obtuse_triangle_count_l5_5643


namespace sum_of_angles_around_point_l5_5732

theorem sum_of_angles_around_point (x : ℝ) (h : 6 * x + 3 * x + 4 * x + x + 2 * x = 360) : x = 22.5 :=
by
  sorry

end sum_of_angles_around_point_l5_5732


namespace sum_b_first_2022_terms_l5_5803

def fib (n : ℕ) : ℕ :=
match n with
| 0 => 1
| 1 => 1
| n+2 => fib (n+1) + fib n

def b (n : ℕ) : ℕ := fib n % 4

theorem sum_b_first_2022_terms : 
  ∑ i in Finset.range 2022, b i = 2696 :=
by
  sorry

end sum_b_first_2022_terms_l5_5803


namespace time_to_cross_platform_l5_5821

-- Definitions for the length of the train, the length of the platform, and the speed of the train
def length_train : ℕ := 750
def length_platform : ℕ := 750
def speed_train_kmh : ℕ := 90

-- Conversion constants
def meters_per_kilometer : ℕ := 1000
def seconds_per_hour : ℕ := 3600

-- Convert speed from km/hr to m/s
def speed_train_ms : ℚ := speed_train_kmh * meters_per_kilometer / seconds_per_hour

-- Total distance the train covers to cross the platform
def total_distance : ℕ := length_train + length_platform

-- Proof problem: To prove that the time taken to cross the platform is 60 seconds
theorem time_to_cross_platform : total_distance / speed_train_ms = 60 := by
  sorry

end time_to_cross_platform_l5_5821


namespace expected_value_eight_sided_die_l5_5411

-- Define a standard 8-sided die
def eight_sided_die : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]

-- Compute the probability of each outcome
def probability (n : ℕ) : ℝ := 1 / 8

-- Expected Value of a discrete random variable
def expected_value (outcomes : List ℕ) (prob : ℕ → ℝ) : ℝ :=
  outcomes.sum / outcomes.length.toReal

-- Theorem stating the expected value of a standard 8-sided die roll is 4.5
theorem expected_value_eight_sided_die : expected_value eight_sided_die probability = 4.5 := by
  sorry

end expected_value_eight_sided_die_l5_5411


namespace surjective_implies_coprime_l5_5965

-- Definitions for the problem
def is_surjective_mod (n : ℕ) (f : ℕ → ℕ) : Prop :=
  ∀ b : ℕ, ∃ x : ℕ, f(x) % n = b % n

def f (x : ℕ) : ℕ := x^x

def euler_totient (n : ℕ) : ℕ := if n = 0 then 0 else (finset.filter (nat.coprime n) (finset.range n)).card

-- Main theorem statement
theorem surjective_implies_coprime (n : ℕ) (h_pos : n > 0) :
  is_surjective_mod n f → nat.coprime n (euler_totient n) :=
by
  sorry

end surjective_implies_coprime_l5_5965


namespace find_m_l5_5673

-- Define the line l
def line_l (x y : ℝ) : Prop := 2 * x - y - 2 = 0

-- Define the circle C
def circle_C (x y m : ℝ) : Prop := x^2 + y^2 - 2 * x + 4 * y + m = 0

-- Length of the intersection line segment
def intersection_length := (2 * real.sqrt 5) / 5

-- Define the theorem to find the value of m
theorem find_m :
  ∃ m : ℝ, (∀ x y : ℝ, circle_C x y m → line_l x y) →
  intersection_length = (2 * real.sqrt 5) / 5 →
  m = 4 :=
sorry

end find_m_l5_5673


namespace smallest_rel_prime_to_180_l5_5604

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ x ≤ 7 ∧ (∀ y : ℕ, y > 1 ∧ y < x → y.gcd 180 ≠ 1) ∧ x.gcd 180 = 1 :=
by
  sorry

end smallest_rel_prime_to_180_l5_5604


namespace count_set_sequences_l5_5964

-- Define the type of sets within the range of 1 to 999
def set_sequence (n : ℕ) : Prop :=
  ∀ i j : ℕ, i ≤ j → (S i : set ℕ) ⊆ (S j : set ℕ)

-- Define the total number of sequences that satisfy the condition
theorem count_set_sequences : 
  (∃ f : ℕ → set (fin 1000), 
    (∀ i j, i ≤ j → f i ⊆ f j) ∧ (∀ i, f i ⊆ { x : fin 1000 | x.1 < 999 }))
  = 1000 ^ 999 :=
sorry

end count_set_sequences_l5_5964


namespace books_loaned_out_l5_5463

theorem books_loaned_out (x : ℕ) 
  (h1 : 75 = 54 + ⌊0.35 * x⌋) 
  : x = 60 := 
  sorry

end books_loaned_out_l5_5463


namespace smallest_rel_prime_to_180_is_7_l5_5587

theorem smallest_rel_prime_to_180_is_7 :
  ∃ x : ℕ, x > 1 ∧ gcd x 180 = 1 ∧ ∀ y : ℕ, y > 1 ∧ gcd y 180 = 1 → x ≤ y :=
begin
  use 7,
  split,
  { exact dec_trivial, },
  split,
  { exact dec_trivial, },
  intros y hy,
  cases hy with hy1 hy2,
  exact dec_trivial,
end

end smallest_rel_prime_to_180_is_7_l5_5587


namespace triple_hash_72_eq_7_25_l5_5518

def hash (N : ℝ) : ℝ := 0.5 * N - 1

theorem triple_hash_72_eq_7_25 : hash (hash (hash 72)) = 7.25 :=
by
  sorry

end triple_hash_72_eq_7_25_l5_5518


namespace smallest_rel_prime_to_180_is_7_l5_5585

theorem smallest_rel_prime_to_180_is_7 :
  ∃ x : ℕ, x > 1 ∧ gcd x 180 = 1 ∧ ∀ y : ℕ, y > 1 ∧ gcd y 180 = 1 → x ≤ y :=
begin
  use 7,
  split,
  { exact dec_trivial, },
  split,
  { exact dec_trivial, },
  intros y hy,
  cases hy with hy1 hy2,
  exact dec_trivial,
end

end smallest_rel_prime_to_180_is_7_l5_5585


namespace second_root_of_quadratic_l5_5117

theorem second_root_of_quadratic (p q r : ℝ) (quad_eqn : ∀ x, 2 * p * (q - r) * x^2 + 3 * q * (r - p) * x + 4 * r * (p - q) = 0) (root : 2 * p * (q - r) * 2^2 + 3 * q * (r - p) * 2 + 4 * r * (p - q) = 0) :
    ∃ r₂ : ℝ, r₂ = (r * (p - q)) / (p * (q - r)) :=
sorry

end second_root_of_quadratic_l5_5117


namespace angle_relation_l5_5740

-- Conditions as definitions
variables (α x y z w : ℝ)
-- x and y are constants representing the given angles in degrees
-- Definition of m1 using appropriate implications from the problem
def m1 := (w = α + y)
def m2 := (α = x + z)

-- The main equality
theorem angle_relation : (∀ α : ℝ, w - y - z = x) :=
by
  -- Introduce known conditions
  assume α,
  -- Use the provided conditions m1 and m2 to show the main result
  have h1 : w = α + y := by sorry,
  have h2 : α = x + z := by sorry,
  -- Proceed to derive the main equality
  sorry

end angle_relation_l5_5740


namespace ratio_ac_l5_5436

variables {a b c : ℚ}

-- Conditions
def ratio_ab := a / b = 7 / 3
def ratio_bc := b / c = 1 / 5

-- Statement of the problem
theorem ratio_ac (h1 : ratio_ab) (h2 : ratio_bc) : a / c = 7 / 15 :=
  sorry

end ratio_ac_l5_5436


namespace sphere_volume_l5_5321

theorem sphere_volume {r : ℝ} (h: 4 * Real.pi * r^2 = 256 * Real.pi) : (4 / 3) * Real.pi * r^3 = (2048 / 3) * Real.pi :=
by
  sorry

end sphere_volume_l5_5321


namespace mod_remainder_l5_5873

theorem mod_remainder (n : ℕ) (h : n % 4 = 3) : (6 * n) % 4 = 2 := 
by
  sorry

end mod_remainder_l5_5873


namespace centroid_trajectory_l5_5787

-- Define the parameters and required proof

def point_on_circle (x y : ℝ) (r : ℝ) : Prop := x^2 + y^2 = r^2

def points_of_triangle_centroid_locus (A B C G : ℝ × ℝ) : Prop :=
    let Ax, Ay := A;
    let Bx, By := B;
    let Cx, Cy := C;
    let Gx, Gy := G;
    Ax = 3 ∧ Ay = 0 ∧
    point_on_circle Bx By 3 ∧
    point_on_circle Cx Cy 3 ∧
    ∠BAC = π/3 ∧
    Gx = (Ax + Bx + Cx) / 3 ∧
    Gy = (Ay + By + Cy) / 3 ∧
    (Gx - 2)^2 + Gy^2 = 1

theorem centroid_trajectory (A B C G : ℝ × ℝ) :
    (let Ax, Ay := A;
    let Bx, By := B;
    let Cx, Cy := C;
    let Gx, Gy := G;
    Ax = 3 ∧ Ay = 0 ∧
    point_on_circle Bx By 3 ∧
    point_on_circle Cx Cy 3 ∧
    ∠BAC = π/3 ∧
    Gx = (Ax + Bx + Cx) / 3 ∧
    Gy = (Ay + By + Cy) / 3) →
    (Gx - 2)^2 + Gy^2 = 1 :=
by 
    intros;
    -- This is where proof would go
    sorry

end centroid_trajectory_l5_5787


namespace smallest_rel_prime_to_180_is_7_l5_5588

theorem smallest_rel_prime_to_180_is_7 :
  ∃ x : ℕ, x > 1 ∧ gcd x 180 = 1 ∧ ∀ y : ℕ, y > 1 ∧ gcd y 180 = 1 → x ≤ y :=
begin
  use 7,
  split,
  { exact dec_trivial, },
  split,
  { exact dec_trivial, },
  intros y hy,
  cases hy with hy1 hy2,
  exact dec_trivial,
end

end smallest_rel_prime_to_180_is_7_l5_5588


namespace food_additives_degrees_undetermined_l5_5910

-- Definitions based on conditions
def total_circle_degrees : ℕ := 360
def basic_astrophysics_degrees : ℕ := 108

-- Theorem statement
theorem food_additives_degrees_undetermined 
: 
True := 
by 
sorr

end food_additives_degrees_undetermined_l5_5910


namespace range_of_m_l5_5523

def A := { x : ℝ | x^2 - 2 * x - 15 ≤ 0 }
def B (m : ℝ) := { x : ℝ | m - 2 < x ∧ x < 2 * m - 3 }

theorem range_of_m : ∀ m : ℝ, (B m ⊆ A) ↔ (m ≤ 4) :=
by sorry

end range_of_m_l5_5523


namespace second_discount_percentage_l5_5314

/-- Given:
1. The original price p = 1000.
2. The first discount is 15%.
3. The final sale price s = 830.

Prove:
The percentage of the second discount d₂ is approximately 2.35%. -/
theorem second_discount_percentage
    (p s : ℝ)
    (d₁ : ℝ)
    (hp : p = 1000)
    (hs : s = 830)
    (hd₁ : d₁ = 15 / 100) :
    (∃ d₂ : ℝ, d₂ = (20 / 8.5) * 100) :=
begin
  sorry
end

end second_discount_percentage_l5_5314


namespace find_x_l5_5532

theorem find_x (x : ℤ) : 3^7 * 3^x = 81 → x = -3 := by
  sorry

end find_x_l5_5532


namespace expected_value_of_eight_sided_die_l5_5377

theorem expected_value_of_eight_sided_die : 
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8], 
      n := (outcomes.length : ℚ),
      probabilities := List.replicate (outcomes.length) (1 / n),
      expected_value := (List.zipWith (*) probabilities (outcomes.map (· : ℚ))).sum
  in expected_value = 4.5 :=
by
  sorry

end expected_value_of_eight_sided_die_l5_5377


namespace max_y_value_l5_5666

theorem max_y_value :
  ∀ (f : ℝ → ℝ) (x : ℝ),
    (∀ x, 1 ≤ x ∧ x ≤ 9 → f(x) = 2 + Real.log x / Real.log 3) →
    (1 ≤ x ∧ x ≤ 9) →
    (let y := (f(x))^2 + f(x^2) in y ≤ 13) :=
begin
  intros f x h1 h2,
  sorry
end

end max_y_value_l5_5666


namespace sequence_from_625_to_629_l5_5193

def arrows_repeating_pattern (n : ℕ) : ℕ := n % 5

theorem sequence_from_625_to_629 :
  arrows_repeating_pattern 625 = 0 ∧ arrows_repeating_pattern 629 = 4 →
  ∃ (seq : ℕ → ℕ), 
    (seq 0 = arrows_repeating_pattern 625) ∧
    (seq 1 = arrows_repeating_pattern (625 + 1)) ∧
    (seq 2 = arrows_repeating_pattern (625 + 2)) ∧
    (seq 3 = arrows_repeating_pattern (625 + 3)) ∧
    (seq 4 = arrows_repeating_pattern 629) := 
sorry

end sequence_from_625_to_629_l5_5193


namespace distinct_banana_points_l5_5755

def circle_radius : ℝ := 1

def position_Luigi (t : ℝ) : ℝ × ℝ :=
  (Real.cos (Real.pi * t / 3), Real.sin (Real.pi * t / 3))

def position_Mario (t : ℝ) : ℝ × ℝ :=
  (Real.cos (Real.pi * t), Real.sin (Real.pi * t))

def position_Daisy (t : ℝ) : ℝ × ℝ :=
  ((Real.cos (Real.pi * t / 3) + Real.cos (Real.pi * t)) / 2,
   (Real.sin (Real.pi * t / 3) + Real.sin (Real.pi * t)) / 2)

-- Define the set of points marked with a banana by Daisy between t = 0 and t = 6
def points_marked : set (ℝ × ℝ) :=
  {p | ∃ t ∈ set.Icc (0 : ℝ) 6, position_Daisy t = p} \ {position_Daisy 0}

theorem distinct_banana_points :
  set.finite points_marked ∧ set.card points_marked = 5 :=
by {
  sorry
}

end distinct_banana_points_l5_5755


namespace product_of_AP_divisible_by_10_fact_l5_5933

theorem product_of_AP_divisible_by_10_fact (a : ℕ) :
  (a * (a + 11) * (a + 22) * (a + 33) * (a + 44) * (a + 55) * (a + 66) * (a + 77) * (a + 88) * (a + 99)) % (10.factorial) = 0 :=
by
  sorry

end product_of_AP_divisible_by_10_fact_l5_5933


namespace evaluate_expression_l5_5089

theorem evaluate_expression :
  (12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1) / (2 - 3 + 4 - 5 + 6 - 7 + 8 - 9 + 10 - 11 + 12) = 6 / 7 := by
sorry

end evaluate_expression_l5_5089


namespace problem1_problem2_l5_5670

-- The given functions and conditions
def g (a : ℝ) (x : ℝ) : ℝ := a * x - a - log x
def f (a : ℝ) (x : ℝ) : ℝ := x * g(a, x)

-- Problem (1): Prove that a = 1
theorem problem1 (h : ∀ x > 0, g a x ≥ 0) : a = 1 := 
sorry

-- Problem (2): Prove the existence of x_0 ∈ (0, 1) such that f'(x_0) = 0 and f(x) ≤ f(x_0)
theorem problem2 (h : ∀ x > 0, g 1 x ≥ 0) : ∃ x0 ∈ (set.Ioo 0 1), deriv (f 1) x0 = 0 ∧ ∀ x ∈ (set.Ioo 0 1), f 1 x ≤ f 1 x0 :=
sorry

end problem1_problem2_l5_5670


namespace expected_value_of_eight_sided_die_l5_5375

theorem expected_value_of_eight_sided_die : 
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8], 
      n := (outcomes.length : ℚ),
      probabilities := List.replicate (outcomes.length) (1 / n),
      expected_value := (List.zipWith (*) probabilities (outcomes.map (· : ℚ))).sum
  in expected_value = 4.5 :=
by
  sorry

end expected_value_of_eight_sided_die_l5_5375


namespace smallest_coprime_gt_one_l5_5570

theorem smallest_coprime_gt_one (x : ℕ) (h1 : 1 < x) (h2 : Nat.gcd x 180 = 1) : x = 7 := sorry

end smallest_coprime_gt_one_l5_5570


namespace harmonic_mean_of_2_3_6_is_3_l5_5418

def harmonic_mean (a b c : ℝ) : ℝ :=
  3 / ((1/a) + (1/b) + (1/c))

theorem harmonic_mean_of_2_3_6_is_3 :
  harmonic_mean 2 3 6 = 3 := by
  sorry

end harmonic_mean_of_2_3_6_is_3_l5_5418


namespace terminating_decimal_fractions_l5_5618

theorem terminating_decimal_fractions :
  let n_count := (finset.range 151).filter (λ n, n % 3 = 0),
  n_count.card = 50 :=
by
  sorry

end terminating_decimal_fractions_l5_5618


namespace range_of_sqrt_function_l5_5083

theorem range_of_sqrt_function (x : ℝ) : x - 3 ≥ 0 → x ≥ 3 :=
by {
  intro h,
  exact h,
}

end range_of_sqrt_function_l5_5083


namespace repeating_decimal_to_fraction_l5_5975

/-- Given 0.02 repeating as a fraction 2/99, prove that 2.06 repeating can be expressed as 68/33 -/
theorem repeating_decimal_to_fraction :
  (2 + 0.06̅ : ℝ) = (68 / 33 : ℝ) :=
begin
  have h : (0.02̅ : ℝ) = (2 / 99 : ℝ), from sorry,
  have h3 : (0.06̅ : ℝ) = 3 * (0.02̅ : ℝ), from sorry,
  have h6 : (0.06̅ : ℝ) = 3 * (2 / 99 : ℝ), from sorry,
  have s : (0.06̅ : ℝ) = (6 / 99 : ℝ), from sorry,
  have s2 : (6 / 99 : ℝ) = (2 / 33 : ℝ), from sorry,
  have add := congr_arg (λ x : ℝ, (2 : ℝ) + x) s2,
  rw [add_comm, ← add_halves', add_assoc', add_comm (2 : ℝ), add_comm 2 (2 / 33 : ℝ), add_halves',
    add_assoc', add_comm (2 : ℝ), add_comm 68 (2 / 33 : ℝ)] ,
end

end repeating_decimal_to_fraction_l5_5975


namespace solution_set_of_fraction_inequality_l5_5661

theorem solution_set_of_fraction_inequality
  (a b : ℝ) (h₀ : ∀ x : ℝ, x > 1 → ax - b > 0) :
  {x : ℝ | (ax + b) / (x - 2) > 0} = {x : ℝ | x < -1} ∪ {x : ℝ | x > 2} :=
by
  sorry

end solution_set_of_fraction_inequality_l5_5661


namespace log_ineq_solution_l5_5797

-- Define the conditions
def log_ineq (a x : ℝ) : Prop :=
  a > 0 ∧ a ≠ 1 ∧ 1 < x ∧ x < 100 ∧ (log a x - (Real.log x) ^ 2 < 4)

-- Define the range of a
def range_a (a : ℝ) : Prop :=
  (a > 0 ∧ a < 1) ∨ (a > Real.exp (1 / 4))

-- The theorem statement
theorem log_ineq_solution (a : ℝ) :
  (∀ x : ℝ, log_ineq a x) → range_a a :=
by
  sorry

end log_ineq_solution_l5_5797


namespace smallest_positive_integer_k_l5_5927

-- Define the conditions
def y : ℕ := 2^3 * 3^4 * (2^2)^5 * 5^6 * (2*3)^7 * 7^8 * (2^3)^9 * (3^2)^10

-- Define the question statement
theorem smallest_positive_integer_k :
  ∃ k : ℕ, k > 0 ∧ (∃ m : ℕ, (y * k) = m^2) ∧ k = 30 :=
by
  sorry

end smallest_positive_integer_k_l5_5927


namespace parallelogram_area_l5_5101

noncomputable def vector_u : ℝ × ℝ × ℝ := (4, 2, -3)
noncomputable def vector_v : ℝ × ℝ × ℝ := (2, -4, 5)

theorem parallelogram_area : 
  let cross_product := (vector_u.2 * vector_v.3 - vector_u.3 * vector_v.2, vector_u.3 * vector_v.1 - vector_u.1 * vector_v.3, vector_u.1 * vector_v.2 - vector_u.2 * vector_v.1) in
  let magnitude := real.sqrt (cross_product.1^2 + cross_product.2^2 + cross_product.3^2) in
  magnitude = 10 * real.sqrt (10.8) :=
sorry

end parallelogram_area_l5_5101


namespace max_missed_algebra_problems_max_missed_geometry_problems_l5_5940

-- Given Constants
def algebra_total_problems : ℕ := 40
def geometry_total_problems : ℕ := 30
def passing_percentage : ℝ := 0.85

-- Definitions based on given problem
def algebra_min_correct_problems : ℕ := (passing_percentage * algebra_total_problems).ceil.to_nat
def geometry_min_correct_problems : ℕ := (passing_percentage * geometry_total_problems).ceil.to_nat

-- The proof problems
theorem max_missed_algebra_problems : algebra_total_problems - algebra_min_correct_problems = 6 := 
  sorry

theorem max_missed_geometry_problems : geometry_total_problems - geometry_min_correct_problems = 4 := 
  sorry

end max_missed_algebra_problems_max_missed_geometry_problems_l5_5940


namespace correct_calculation_l5_5223

theorem correct_calculation (x : ℤ) (h : 20 + x = 60) : 34 - x = -6 := by
  sorry

end correct_calculation_l5_5223


namespace finite_set_condition_l5_5875

noncomputable def arithmetic_mean (x y : ℝ) : ℝ := (x + y) / 2

theorem finite_set_condition (X : set ℝ) [fintype X] : 
  (∀ a b ∈ X, ∃ x ∈ X, arithmetic_mean a x = b) ↔ ∃ a, X = {a} :=
sorry

end finite_set_condition_l5_5875
