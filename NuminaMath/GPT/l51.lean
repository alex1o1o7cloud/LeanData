import Mathlib

namespace NUMINAMATH_GPT_left_person_truthful_right_person_lies_l51_5154

theorem left_person_truthful_right_person_lies
  (L R M : Prop)
  (L_truthful_or_false : L ∨ ¬L)
  (R_truthful_or_false : R ∨ ¬R)
  (M_always_answers : M = (L → M) ∨ (¬L → M))
  (left_statement : L → (M = (L → M)))
  (right_statement : R → (M = (¬L → M))) :
  (L ∧ ¬R) ∨ (¬L ∧ R) :=
by
  sorry

end NUMINAMATH_GPT_left_person_truthful_right_person_lies_l51_5154


namespace NUMINAMATH_GPT_find_f2_l51_5195

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := sorry
noncomputable def a : ℝ := sorry

axiom odd_f : ∀ x, f (-x) = -f x
axiom even_g : ∀ x, g (-x) = g x
axiom fg_eq : ∀ x, f x + g x = a^x - a^(-x) + 2
axiom g2_a : g 2 = a
axiom a_pos : a > 0
axiom a_ne1 : a ≠ 1

theorem find_f2 : f 2 = 15 / 4 := 
by sorry

end NUMINAMATH_GPT_find_f2_l51_5195


namespace NUMINAMATH_GPT_rounding_estimate_lt_exact_l51_5100

variable (a b c a' b' c' : ℕ)

theorem rounding_estimate_lt_exact (ha : a' ≤ a) (hb : b' ≥ b) (hc : c' ≤ c) (hb_pos : b > 0) (hb'_pos : b' > 0) :
  (a':ℚ) / (b':ℚ) + (c':ℚ) < (a:ℚ) / (b:ℚ) + (c:ℚ) :=
sorry

end NUMINAMATH_GPT_rounding_estimate_lt_exact_l51_5100


namespace NUMINAMATH_GPT_f_neg_one_f_eq_half_l51_5190

noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 0 then 2^(-x) else Real.log x / Real.log 2

theorem f_neg_one : f (-1) = 2 := by
  sorry

theorem f_eq_half (x : ℝ) : f x = 1 / 2 ↔ x = Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_f_neg_one_f_eq_half_l51_5190


namespace NUMINAMATH_GPT_total_gallons_l51_5134

def gallons_used (A F : ℕ) := F = 4 * A - 5

theorem total_gallons
  (A F : ℕ)
  (h1 : gallons_used A F)
  (h2 : F = 23) :
  A + F = 30 :=
by
  sorry

end NUMINAMATH_GPT_total_gallons_l51_5134


namespace NUMINAMATH_GPT_credit_card_balance_l51_5171

theorem credit_card_balance :
  ∀ (initial_balance groceries_charge gas_charge return_credit : ℕ),
  initial_balance = 126 →
  groceries_charge = 60 →
  gas_charge = groceries_charge / 2 →
  return_credit = 45 →
  initial_balance + groceries_charge + gas_charge - return_credit = 171 :=
by
  intros initial_balance groceries_charge gas_charge return_credit
  intros h_initial h_groceries h_gas h_return
  rw [h_initial, h_groceries, h_gas, h_return]
  norm_num
  sorry

end NUMINAMATH_GPT_credit_card_balance_l51_5171


namespace NUMINAMATH_GPT_solution1_solution2_l51_5110

noncomputable def problem1 : ℝ :=
  40 + ((1 / 6) - (2 / 3) + (3 / 4)) * 12

theorem solution1 : problem1 = 43 := by
  sorry

noncomputable def problem2 : ℝ :=
  (-1 : ℝ) ^ 2021 + |(-9 : ℝ)| * (2 / 3) + (-3) / (1 / 5)

theorem solution2 : problem2 = -10 := by
  sorry

end NUMINAMATH_GPT_solution1_solution2_l51_5110


namespace NUMINAMATH_GPT_sin_beta_value_l51_5139

theorem sin_beta_value (α β : ℝ) (hα : 0 < α ∧ α < π/2) (hβ : 0 < β ∧ β < π/2)
  (h1 : Real.cos α = 4 / 5) (h2 : Real.cos (α + β) = 5 / 13) :
  Real.sin β = 33 / 65 :=
sorry

end NUMINAMATH_GPT_sin_beta_value_l51_5139


namespace NUMINAMATH_GPT_inverse_proposition_false_l51_5113

-- Definitions for the conditions
def congruent (A B C D E F: ℝ) : Prop := 
  A = D ∧ B = E ∧ C = F

def angles_equal (α β γ δ ε ζ: ℝ) : Prop := 
  α = δ ∧ β = ε ∧ γ = ζ

def original_proposition (A B C D E F α β γ : ℝ) : Prop :=
  congruent A B C D E F → angles_equal α β γ A B C

-- The inverse proposition
def inverse_proposition (α β γ δ ε ζ A B C D E F : ℝ) : Prop :=
  angles_equal α β γ δ ε ζ → congruent A B C D E F

-- The main theorem: the inverse proposition is false
theorem inverse_proposition_false (α β γ δ ε ζ A B C D E F : ℝ) :
  ¬(inverse_proposition α β γ δ ε ζ A B C D E F) := by
  sorry

end NUMINAMATH_GPT_inverse_proposition_false_l51_5113


namespace NUMINAMATH_GPT_fraction_subtraction_proof_l51_5150

theorem fraction_subtraction_proof : 
  (21 / 12) - (18 / 15) = 11 / 20 := 
by 
  sorry

end NUMINAMATH_GPT_fraction_subtraction_proof_l51_5150


namespace NUMINAMATH_GPT_find_n_in_arithmetic_sequence_l51_5162

noncomputable def arithmetic_sequence (a1 d n : ℕ) := a1 + (n - 1) * d

theorem find_n_in_arithmetic_sequence (a1 d an : ℕ) (h1 : a1 = 1) (h2 : d = 5) (h3 : an = 2016) :
  ∃ n : ℕ, an = arithmetic_sequence a1 d n :=
  by
  sorry

end NUMINAMATH_GPT_find_n_in_arithmetic_sequence_l51_5162


namespace NUMINAMATH_GPT_water_tower_excess_consumption_l51_5178

def water_tower_problem : Prop :=
  let initial_water := 2700
  let first_neighborhood := 300
  let second_neighborhood := 2 * first_neighborhood
  let third_neighborhood := second_neighborhood + 100
  let fourth_neighborhood := 3 * first_neighborhood
  let fifth_neighborhood := third_neighborhood / 2
  let leakage := 50
  let first_neighborhood_final := first_neighborhood + 0.10 * first_neighborhood
  let second_neighborhood_final := second_neighborhood - 0.05 * second_neighborhood
  let third_neighborhood_final := third_neighborhood + 0.10 * third_neighborhood
  let fifth_neighborhood_final := fifth_neighborhood - 0.05 * fifth_neighborhood
  let total_consumption := 
    first_neighborhood_final + second_neighborhood_final + third_neighborhood_final +
    fourth_neighborhood + fifth_neighborhood_final + leakage
  let excess_consumption := total_consumption - initial_water
  excess_consumption = 252.5

theorem water_tower_excess_consumption : water_tower_problem := by
  sorry

end NUMINAMATH_GPT_water_tower_excess_consumption_l51_5178


namespace NUMINAMATH_GPT_banana_price_l51_5199

theorem banana_price (x y : ℕ) (b : ℕ) 
  (hx : x + y = 4) 
  (cost_eq : 50 * x + 60 * y + b = 275) 
  (banana_cheaper_than_pear : b < 60) 
  : b = 35 ∨ b = 45 ∨ b = 55 :=
by
  sorry

end NUMINAMATH_GPT_banana_price_l51_5199


namespace NUMINAMATH_GPT_find_B_current_age_l51_5186

variable {A B C : ℕ}

theorem find_B_current_age (h1 : A + 10 = 2 * (B - 10))
                          (h2 : A = B + 7)
                          (h3 : C = (A + B) / 2) :
                          B = 37 := by
  sorry

end NUMINAMATH_GPT_find_B_current_age_l51_5186


namespace NUMINAMATH_GPT_exists_natural_number_starting_and_ending_with_pattern_l51_5176

theorem exists_natural_number_starting_and_ending_with_pattern (n : ℕ) : 
  ∃ (m : ℕ), 
  (m % 10 = 1) ∧ 
  (∃ t : ℕ, 
    m^2 / 10^t = 10^(n - 1) * (10^n - 1) / 9) ∧ 
  (m^2 % 10^n = 1 ∨ m^2 % 10^n = 2) :=
sorry

end NUMINAMATH_GPT_exists_natural_number_starting_and_ending_with_pattern_l51_5176


namespace NUMINAMATH_GPT_g_does_not_pass_through_fourth_quadrant_l51_5109

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x-1) - 2
noncomputable def g (x : ℝ) : ℝ := 1 + (1 / x)

theorem g_does_not_pass_through_fourth_quadrant (a : ℝ) (h : a > 0 ∧ a ≠ 1) : 
    ¬(∃ x, x > 0 ∧ g x < 0) :=
by
    sorry

end NUMINAMATH_GPT_g_does_not_pass_through_fourth_quadrant_l51_5109


namespace NUMINAMATH_GPT_system_solution_l51_5169

theorem system_solution :
  ∃ x y : ℝ, (16 * x^2 + 8 * x * y + 4 * y^2 + 20 * x + 2 * y = -7) ∧ 
            (8 * x^2 - 16 * x * y + 2 * y^2 + 20 * x - 14 * y = -11) ∧
            x = -3 / 4 ∧ y = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_system_solution_l51_5169


namespace NUMINAMATH_GPT_certain_amount_eq_3_l51_5187

theorem certain_amount_eq_3 (x A : ℕ) (hA : A = 5) (h : A + (11 + x) = 19) : x = 3 :=
by
  sorry

end NUMINAMATH_GPT_certain_amount_eq_3_l51_5187


namespace NUMINAMATH_GPT_smallest_Y_l51_5122

theorem smallest_Y (U : ℕ) (Y : ℕ) (hU : U = 15 * Y) 
  (digits_U : ∀ d ∈ Nat.digits 10 U, d = 0 ∨ d = 1) 
  (div_15 : U % 15 = 0) : Y = 74 :=
sorry

end NUMINAMATH_GPT_smallest_Y_l51_5122


namespace NUMINAMATH_GPT_factorization_and_evaluation_l51_5121

noncomputable def polynomial_q1 (x : ℝ) : ℝ := x
noncomputable def polynomial_q2 (x : ℝ) : ℝ := x^2 - 2
noncomputable def polynomial_q3 (x : ℝ) : ℝ := x^2 + x + 1
noncomputable def polynomial_q4 (x : ℝ) : ℝ := x^2 + 1

theorem factorization_and_evaluation :
  polynomial_q1 3 + polynomial_q2 3 + polynomial_q3 3 + polynomial_q4 3 = 33 := by
  sorry

end NUMINAMATH_GPT_factorization_and_evaluation_l51_5121


namespace NUMINAMATH_GPT_single_jalapeno_strips_l51_5168

-- Definitions based on conditions
def strips_per_sandwich : ℕ := 4
def minutes_per_sandwich : ℕ := 5
def hours_per_day : ℕ := 8
def total_jalapeno_peppers_used : ℕ := 48
def minutes_per_hour : ℕ := 60

-- Calculate intermediate steps
def total_minutes : ℕ := hours_per_day * minutes_per_hour
def total_sandwiches_served : ℕ := total_minutes / minutes_per_sandwich
def total_strips_needed : ℕ := total_sandwiches_served * strips_per_sandwich

theorem single_jalapeno_strips :
  total_strips_needed / total_jalapeno_peppers_used = 8 := 
by
  sorry

end NUMINAMATH_GPT_single_jalapeno_strips_l51_5168


namespace NUMINAMATH_GPT_probability_not_siblings_l51_5133

-- Define the number of people and the sibling condition
def number_of_people : ℕ := 6
def siblings_count (x : Fin number_of_people) : ℕ := 2

-- Define the probability that two individuals randomly selected are not siblings
theorem probability_not_siblings (P Q : Fin number_of_people) (h : P ≠ Q) :
  let K := number_of_people - 1
  let non_siblings := K - siblings_count P
  (non_siblings / K : ℚ) = 3 / 5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_probability_not_siblings_l51_5133


namespace NUMINAMATH_GPT_add_base6_l51_5130

def base6_to_base10 (n : Nat) : Nat :=
  let rec aux (n : Nat) (exp : Nat) : Nat :=
    match n with
    | 0     => 0
    | n + 1 => aux n (exp + 1) + (n % 6) * (6 ^ exp)
  aux n 0

def base10_to_base6 (n : Nat) : Nat :=
  let rec aux (n : Nat) : Nat :=
    if n = 0 then 0
    else
      let q := n / 6
      let r := n % 6
      r + 10 * aux q
  aux n

theorem add_base6 (a b : Nat) (h1 : base6_to_base10 a = 5) (h2 : base6_to_base10 b = 13) : base10_to_base6 (base6_to_base10 a + base6_to_base10 b) = 30 :=
by
  sorry

end NUMINAMATH_GPT_add_base6_l51_5130


namespace NUMINAMATH_GPT_solveSALE_l51_5188

namespace Sherlocked

open Nat

def areDistinctDigits (d₁ d₂ d₃ d₄ d₅ d₆ : Nat) : Prop :=
  d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₁ ≠ d₄ ∧ d₁ ≠ d₅ ∧ d₁ ≠ d₆ ∧ 
  d₂ ≠ d₃ ∧ d₂ ≠ d₄ ∧ d₂ ≠ d₅ ∧ d₂ ≠ d₆ ∧ 
  d₃ ≠ d₄ ∧ d₃ ≠ d₅ ∧ d₃ ≠ d₆ ∧ 
  d₄ ≠ d₅ ∧ d₄ ≠ d₆ ∧ 
  d₅ ≠ d₆

theorem solveSALE :
  ∃ (S C A L E T : ℕ),
    SCALE - SALE = SLATE ∧
    areDistinctDigits S C A L E T ∧
    S < 10 ∧ C < 10 ∧ A < 10 ∧
    L < 10 ∧ E < 10 ∧ T < 10 ∧
    SALE = 1829 :=
by
  sorry

end Sherlocked

end NUMINAMATH_GPT_solveSALE_l51_5188


namespace NUMINAMATH_GPT_solve_m_correct_l51_5160

noncomputable def solve_for_m (Q t h : ℝ) : ℝ :=
  if h >= 0 ∧ Q > 0 ∧ t > 0 then
    (Real.log (t / Q)) / (Real.log (1 + Real.sqrt h))
  else
    0 -- Define default output for invalid inputs

theorem solve_m_correct (Q t h : ℝ) (m : ℝ) :
  Q = t / (1 + Real.sqrt h)^m → m = (Real.log (t / Q)) / (Real.log (1 + Real.sqrt h)) :=
by
  intros h1
  rw [h1]
  sorry

end NUMINAMATH_GPT_solve_m_correct_l51_5160


namespace NUMINAMATH_GPT_barrel_contents_lost_l51_5124

theorem barrel_contents_lost (initial_amount remaining_amount : ℝ) 
  (h1 : initial_amount = 220) 
  (h2 : remaining_amount = 198) : 
  (initial_amount - remaining_amount) / initial_amount * 100 = 10 :=
by
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_barrel_contents_lost_l51_5124


namespace NUMINAMATH_GPT_imaginary_part_of_complex_num_l51_5185

def imaginary_unit : ℂ := Complex.I

noncomputable def complex_num : ℂ := 10 * imaginary_unit / (1 - 2 * imaginary_unit)

theorem imaginary_part_of_complex_num : complex_num.im = 2 := by
  sorry

end NUMINAMATH_GPT_imaginary_part_of_complex_num_l51_5185


namespace NUMINAMATH_GPT_line_equation_mb_l51_5103

theorem line_equation_mb (b m : ℤ) (h_b : b = -2) (h_m : m = 5) : m * b = -10 :=
by
  rw [h_b, h_m]
  norm_num

end NUMINAMATH_GPT_line_equation_mb_l51_5103


namespace NUMINAMATH_GPT_equilateral_triangle_area_l51_5119

theorem equilateral_triangle_area (h : ℝ) (h_eq : h = 2 * Real.sqrt 3) : 
  (Real.sqrt 3 / 4) * (2 * h / (Real.sqrt 3))^2 = 4 * Real.sqrt 3 := 
by
  rw [h_eq]
  sorry

end NUMINAMATH_GPT_equilateral_triangle_area_l51_5119


namespace NUMINAMATH_GPT_find_a10_l51_5181

def arithmetic_sequence (a1 d : ℤ) (n : ℕ) : ℤ := a1 + (n - 1) * d

theorem find_a10 
  (a1 d : ℤ)
  (h_condition : a1 + (a1 + 18 * d) = -18) :
  arithmetic_sequence a1 d 10 = -9 := 
by
  sorry

end NUMINAMATH_GPT_find_a10_l51_5181


namespace NUMINAMATH_GPT_system_solution_l51_5137

theorem system_solution (m n : ℚ) (x y : ℚ) 
  (h₁ : 2 * x + m * y = 5) 
  (h₂ : n * x - 3 * y = 2) 
  (h₃ : x = 3)
  (h₄ : y = 1) : 
  m / n = -3 / 5 :=
by sorry

end NUMINAMATH_GPT_system_solution_l51_5137


namespace NUMINAMATH_GPT_gcd_of_360_and_150_l51_5159

theorem gcd_of_360_and_150 : Nat.gcd 360 150 = 30 := 
by
  sorry

end NUMINAMATH_GPT_gcd_of_360_and_150_l51_5159


namespace NUMINAMATH_GPT_no_solution_l51_5172

theorem no_solution : ∀ x : ℝ, ¬ (3 * x + 2 < (x + 2)^2 ∧ (x + 2)^2 < 5 * x + 1) :=
by
  intro x
  -- Solve each part of the inequality
  have h1 : ¬ (3 * x + 2 < (x + 2)^2) ↔ x^2 + x + 2 ≤ 0 := by sorry
  have h2 : ¬ ((x + 2)^2 < 5 * x + 1) ↔ x^2 - x + 3 ≥ 0 := by sorry
  -- Combine the results
  exact sorry

end NUMINAMATH_GPT_no_solution_l51_5172


namespace NUMINAMATH_GPT_exam_time_ratio_l51_5170

-- Lean statements to define the problem conditions and goal
theorem exam_time_ratio (x M : ℝ) (h1 : x > 0) (h2 : M = x / 18) : 
  (5 * x / 6 + 2 * M) / (x / 6 - 2 * M) = 17 := by
  sorry

end NUMINAMATH_GPT_exam_time_ratio_l51_5170


namespace NUMINAMATH_GPT_triangle_area_l51_5173

theorem triangle_area (a b c : ℝ) (C : ℝ) (h1 : c^2 = (a - b)^2 + 6) (h2 : C = π / 3) :
    abs ((1 / 2) * a * b * Real.sin C) = 3 * Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_l51_5173


namespace NUMINAMATH_GPT_range_of_m_to_satisfy_quadratic_l51_5191

def quadratic_positive_forall_m (m : ℝ) : Prop :=
  ∀ x : ℝ, m * x^2 + m * x + 100 > 0

theorem range_of_m_to_satisfy_quadratic :
  {m : ℝ | quadratic_positive_forall_m m} = {m : ℝ | 0 ≤ m ∧ m < 400} :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_to_satisfy_quadratic_l51_5191


namespace NUMINAMATH_GPT_inequality_solution_l51_5193

theorem inequality_solution (x : ℝ) : (x < -4 ∨ x > -4) → (x + 3) / (x + 4) > (2 * x + 7) / (3 * x + 12) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_inequality_solution_l51_5193


namespace NUMINAMATH_GPT_range_of_a_for_f_increasing_l51_5138

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 - a) * x - a else Real.log x / Real.log a

theorem range_of_a_for_f_increasing :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ↔ (3 / 2 ≤ a ∧ a < 3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_for_f_increasing_l51_5138


namespace NUMINAMATH_GPT_largest_value_of_d_l51_5148

noncomputable def maximum_possible_value_of_d 
  (a b c d : ℝ) 
  (h1 : a + b + c + d = 10) 
  (h2 : ab + ac + ad + bc + bd + cd = 17) : ℝ :=
  (5 + Real.sqrt 123) / 2

theorem largest_value_of_d 
  (a b c d : ℝ) 
  (h1 : a + b + c + d = 10) 
  (h2 : ab + ac + ad + bc + bd + cd = 17) : 
  d ≤ maximum_possible_value_of_d a b c d h1 h2 :=
sorry

end NUMINAMATH_GPT_largest_value_of_d_l51_5148


namespace NUMINAMATH_GPT_sample_size_stratified_sampling_l51_5164

theorem sample_size_stratified_sampling :
  let N_business := 120
  let N_management := 24
  let N_logistics := 16
  let N_total := N_business + N_management + N_logistics
  let n_management_chosen := 3
  let sampling_fraction := n_management_chosen / N_management
  let sample_size := N_total * sampling_fraction
  sample_size = 20 :=
by
  -- Definitions:
  let N_business := 120
  let N_management := 24
  let N_logistics := 16
  let N_total := N_business + N_management + N_logistics
  let n_management_chosen := 3
  let sampling_fraction := n_management_chosen / N_management
  let sample_size := N_total * sampling_fraction
  
  -- Proof:
  sorry

end NUMINAMATH_GPT_sample_size_stratified_sampling_l51_5164


namespace NUMINAMATH_GPT_triangle_base_l51_5145

theorem triangle_base (A h b : ℝ) (hA : A = 15) (hh : h = 6) (hbase : A = 0.5 * b * h) : b = 5 := by
  sorry

end NUMINAMATH_GPT_triangle_base_l51_5145


namespace NUMINAMATH_GPT_number_of_factors_l51_5101

theorem number_of_factors (b n : ℕ) (hb1 : b = 6) (hn1 : n = 15) (hb2 : b > 0) (hb3 : b ≤ 15) (hn2 : n > 0) (hn3 : n ≤ 15) :
  let factors := (15 + 1) * (15 + 1)
  factors = 256 :=
by
  sorry

end NUMINAMATH_GPT_number_of_factors_l51_5101


namespace NUMINAMATH_GPT_find_x_l51_5118

/-- Given real numbers x and y,
    under the condition that (y^3 + 2y - 1)/(y^3 + 2y - 3) = x/(x - 1),
    we want to prove that x = (y^3 + 2y - 1)/2 -/
theorem find_x (x y : ℝ) (h1 : y^3 + 2*y - 3 ≠ 0) (h2 : y^3 + 2*y - 1 ≠ 0)
  (h : x / (x - 1) = (y^3 + 2*y - 1) / (y^3 + 2*y - 3)) :
  x = (y^3 + 2*y - 1) / 2 :=
by sorry

end NUMINAMATH_GPT_find_x_l51_5118


namespace NUMINAMATH_GPT_Aiyanna_has_more_cookies_l51_5123

theorem Aiyanna_has_more_cookies (Alyssa_cookies : ℕ) (Aiyanna_cookies : ℕ) (hAlyssa : Alyssa_cookies = 129) (hAiyanna : Aiyanna_cookies = 140) : Aiyanna_cookies - Alyssa_cookies = 11 := 
by sorry

end NUMINAMATH_GPT_Aiyanna_has_more_cookies_l51_5123


namespace NUMINAMATH_GPT_pounds_of_sugar_l51_5126

theorem pounds_of_sugar (x p : ℝ) (h1 : x * p = 216) (h2 : (x + 3) * (p - 1) = 216) : x = 24 :=
sorry

end NUMINAMATH_GPT_pounds_of_sugar_l51_5126


namespace NUMINAMATH_GPT_project_completion_by_B_l51_5116

-- Definitions of the given conditions
def person_A_work_rate := 1 / 10
def person_B_work_rate := 1 / 15
def days_A_worked := 3

-- Definition of the mathematical proof problem
theorem project_completion_by_B {x : ℝ} : person_A_work_rate * days_A_worked + person_B_work_rate * x = 1 :=
by
  sorry

end NUMINAMATH_GPT_project_completion_by_B_l51_5116


namespace NUMINAMATH_GPT_james_profit_l51_5132

-- Definitions and Conditions
def head_of_cattle : ℕ := 100
def purchase_price : ℕ := 40000
def feeding_percentage : ℕ := 20
def weight_per_head : ℕ := 1000
def price_per_pound : ℕ := 2

def feeding_cost : ℕ := (purchase_price * feeding_percentage) / 100
def total_cost : ℕ := purchase_price + feeding_cost
def selling_price_per_head : ℕ := weight_per_head * price_per_pound
def total_selling_price : ℕ := head_of_cattle * selling_price_per_head
def profit : ℕ := total_selling_price - total_cost

-- Theorem to Prove
theorem james_profit : profit = 112000 := by
  sorry

end NUMINAMATH_GPT_james_profit_l51_5132


namespace NUMINAMATH_GPT_circumference_difference_l51_5149

theorem circumference_difference (r : ℝ) (width : ℝ) (hp : width = 10.504226244065093) : 
  2 * Real.pi * (r + width) - 2 * Real.pi * r = 66.00691339889247 := by
  sorry

end NUMINAMATH_GPT_circumference_difference_l51_5149


namespace NUMINAMATH_GPT_tire_circumference_l51_5183

variable (rpm : ℕ) (car_speed_kmh : ℕ) (circumference : ℝ)

-- Define the conditions
def conditions : Prop :=
  rpm = 400 ∧ car_speed_kmh = 24

-- Define the statement to prove
theorem tire_circumference (h : conditions rpm car_speed_kmh) : circumference = 1 :=
sorry

end NUMINAMATH_GPT_tire_circumference_l51_5183


namespace NUMINAMATH_GPT_part_one_part_two_l51_5111

-- Definitions based on the conditions
def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B : Set ℝ := {x | 0 < x ∧ x ≤ 1}
def C (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ 2 - a}

-- Prove intersection A ∩ B = (0, 1)
theorem part_one : A ∩ B = { x | 0 < x ∧ x < 1 } := by
  sorry

-- Prove range of a when A ∪ C = A
theorem part_two (a : ℝ) (h : A ∪ C a = A) : 1 < a := by
  sorry

end NUMINAMATH_GPT_part_one_part_two_l51_5111


namespace NUMINAMATH_GPT_zeros_of_f_l51_5167

def f (x : ℝ) : ℝ := (x^2 - 3 * x) * (x + 4)

theorem zeros_of_f : ∀ x, f x = 0 ↔ x = 0 ∨ x = 3 ∨ x = -4 := by
  sorry

end NUMINAMATH_GPT_zeros_of_f_l51_5167


namespace NUMINAMATH_GPT_a4_value_a_n_formula_l51_5189

theorem a4_value : a_4 = 30 := 
by
    sorry

noncomputable def a_n (n : ℕ) : ℕ :=
    (n * (n + 1)^2 * (2 * n + 1)) / 6

theorem a_n_formula (n : ℕ) : a_n n = (n * (n + 1)^2 * (2 * n + 1)) / 6 := 
by
    sorry

end NUMINAMATH_GPT_a4_value_a_n_formula_l51_5189


namespace NUMINAMATH_GPT_second_expression_l51_5180

theorem second_expression (a x : ℕ) (h₁ : (2 * a + 16 + x) / 2 = 79) (h₂ : a = 30) : x = 82 := by
  sorry

end NUMINAMATH_GPT_second_expression_l51_5180


namespace NUMINAMATH_GPT_frank_ryan_problem_ratio_l51_5142

theorem frank_ryan_problem_ratio 
  (bill_problems : ℕ)
  (h1 : bill_problems = 20)
  (ryan_problems : ℕ)
  (h2 : ryan_problems = 2 * bill_problems)
  (frank_problems_per_type : ℕ)
  (h3 : frank_problems_per_type = 30)
  (types : ℕ)
  (h4 : types = 4) : 
  frank_problems_per_type * types / ryan_problems = 3 := by
  sorry

end NUMINAMATH_GPT_frank_ryan_problem_ratio_l51_5142


namespace NUMINAMATH_GPT_bundles_burned_in_afternoon_l51_5153

theorem bundles_burned_in_afternoon 
  (morning_burn : ℕ)
  (start_bundles : ℕ)
  (end_bundles : ℕ)
  (h_morning_burn : morning_burn = 4)
  (h_start : start_bundles = 10)
  (h_end : end_bundles = 3)
  : (start_bundles - morning_burn - end_bundles) = 3 := 
by 
  sorry

end NUMINAMATH_GPT_bundles_burned_in_afternoon_l51_5153


namespace NUMINAMATH_GPT_optimal_years_minimize_cost_l51_5198

noncomputable def initial_cost : ℝ := 150000
noncomputable def annual_expenses (n : ℕ) : ℝ := 15000 * n
noncomputable def maintenance_cost (n : ℕ) : ℝ := (n * (3000 + 3000 * n)) / 2
noncomputable def total_cost (n : ℕ) : ℝ := initial_cost + annual_expenses n + maintenance_cost n
noncomputable def average_annual_cost (n : ℕ) : ℝ := total_cost n / n

theorem optimal_years_minimize_cost : ∀ n : ℕ, n = 10 ↔ average_annual_cost 10 ≤ average_annual_cost n :=
by sorry

end NUMINAMATH_GPT_optimal_years_minimize_cost_l51_5198


namespace NUMINAMATH_GPT_chairs_left_to_move_l51_5127

theorem chairs_left_to_move (total_chairs : ℕ) (carey_chairs : ℕ) (pat_chairs : ℕ) (h1 : total_chairs = 74)
  (h2 : carey_chairs = 28) (h3 : pat_chairs = 29) : total_chairs - carey_chairs - pat_chairs = 17 :=
by 
  sorry

end NUMINAMATH_GPT_chairs_left_to_move_l51_5127


namespace NUMINAMATH_GPT_domain_of_sqrt_fraction_l51_5161

theorem domain_of_sqrt_fraction (x : ℝ) : 
  (x - 2 ≥ 0 ∧ 5 - x > 0) ↔ (2 ≤ x ∧ x < 5) :=
by
  sorry

end NUMINAMATH_GPT_domain_of_sqrt_fraction_l51_5161


namespace NUMINAMATH_GPT_eval_expression_l51_5136

theorem eval_expression :
  -((18 / 3 * 8) - 80 + (4 ^ 2 * 2)) = 0 :=
by
  sorry

end NUMINAMATH_GPT_eval_expression_l51_5136


namespace NUMINAMATH_GPT_smallest_k_sum_sequence_l51_5104

theorem smallest_k_sum_sequence (n : ℕ) (h : 100 = (n + 1) * (2 * 9 + n) / 2) : k = 9 := 
sorry

end NUMINAMATH_GPT_smallest_k_sum_sequence_l51_5104


namespace NUMINAMATH_GPT_Heather_total_distance_walked_l51_5106

theorem Heather_total_distance_walked :
  let d1 := 0.645
  let d2 := 1.235
  let d3 := 0.875
  let d4 := 1.537
  let d5 := 0.932
  (d1 + d2 + d3 + d4 + d5) = 5.224 := 
by
  sorry -- Proof goes here

end NUMINAMATH_GPT_Heather_total_distance_walked_l51_5106


namespace NUMINAMATH_GPT_min_value_sqrt_sum_l51_5108

open Real

theorem min_value_sqrt_sum (x : ℝ) : 
    ∃ c : ℝ, (∀ x : ℝ, c ≤ sqrt (x^2 - 4 * x + 13) + sqrt (x^2 - 10 * x + 26)) ∧ 
             (sqrt ((17/4)^2 - 4 * (17/4) + 13) + sqrt ((17/4)^2 - 10 * (17/4) + 26) = 5 ∧ c = 5) := 
by
  sorry

end NUMINAMATH_GPT_min_value_sqrt_sum_l51_5108


namespace NUMINAMATH_GPT_solve_system_of_equations_l51_5155

theorem solve_system_of_equations (x y : ℝ) 
  (h1 : x + y = 55) 
  (h2 : x - y = 15) 
  (h3 : x > y) : 
  x = 35 ∧ y = 20 := 
sorry

end NUMINAMATH_GPT_solve_system_of_equations_l51_5155


namespace NUMINAMATH_GPT_tan_alpha_values_l51_5165

theorem tan_alpha_values (α : ℝ) (h : 2 * Real.sin α ^ 2 - Real.sin α * Real.cos α + 5 * Real.cos α ^ 2 = 3) : 
  Real.tan α = 1 ∨ Real.tan α = -2 := 
by sorry

end NUMINAMATH_GPT_tan_alpha_values_l51_5165


namespace NUMINAMATH_GPT_sum_of_possible_values_l51_5152

theorem sum_of_possible_values (x y : ℝ) 
  (h : x * y - 2 * x / y ^ 3 - 2 * y / x ^ 3 = 4) : 
  (x - 2) * (y - 2) = 1 := 
sorry

end NUMINAMATH_GPT_sum_of_possible_values_l51_5152


namespace NUMINAMATH_GPT_stock_return_to_original_l51_5125

theorem stock_return_to_original (x : ℝ) : 
  let price_2006 := x
  let price_end_2006 := 1.30 * price_2006
  let price_end_2007 := 1.20 * price_end_2006
  let p := (0.56 * 100 / 1.56)
  1.56 * (1 - p/100) = 1 :=
by
  intro x
  let price_2006 := x
  let price_end_2006 := 1.30 * price_2006
  let price_end_2007 := 1.20 * price_end_2006
  let p := (0.56 * 100 / 1.56)
  show 1.56 * (1 - p / 100) = 1
  sorry

end NUMINAMATH_GPT_stock_return_to_original_l51_5125


namespace NUMINAMATH_GPT_min_value_expression_l51_5147

theorem min_value_expression (x y : ℝ) (h1 : x * y > 0) (h2 : x^2 * y = 2) : (x * y + x^2) ≥ 4 :=
sorry

end NUMINAMATH_GPT_min_value_expression_l51_5147


namespace NUMINAMATH_GPT_tangent_line_at_point_l51_5174

theorem tangent_line_at_point (x y : ℝ) (h_curve : y = Real.exp x - 2 * x) (h_point : (0, 1) = (x, y)) :
  x + y - 1 = 0 := 
by 
  sorry

end NUMINAMATH_GPT_tangent_line_at_point_l51_5174


namespace NUMINAMATH_GPT_value_of_a_l51_5143

variable (a : ℝ)

theorem value_of_a (h1 : (0.5 / 100) * a = 0.80) : a = 160 := by
  sorry

end NUMINAMATH_GPT_value_of_a_l51_5143


namespace NUMINAMATH_GPT_carpet_area_l51_5146

theorem carpet_area (length_ft : ℕ) (width_ft : ℕ) (ft_per_yd : ℕ) (A_y : ℕ) 
  (h_length : length_ft = 15) (h_width : width_ft = 12) (h_ft_per_yd : ft_per_yd = 9) :
  A_y = (length_ft * width_ft) / ft_per_yd := 
by sorry

#check carpet_area

end NUMINAMATH_GPT_carpet_area_l51_5146


namespace NUMINAMATH_GPT_johns_total_working_hours_l51_5107

theorem johns_total_working_hours (d h t : Nat) (h_d : d = 5) (h_h : h = 8) : t = d * h := by
  rewrite [h_d, h_h]
  sorry

end NUMINAMATH_GPT_johns_total_working_hours_l51_5107


namespace NUMINAMATH_GPT_dany_farm_bushels_l51_5120

theorem dany_farm_bushels :
  let cows := 5
  let cows_bushels_per_day := 3
  let sheep := 4
  let sheep_bushels_per_day := 2
  let chickens := 8
  let chickens_bushels_per_day := 1
  let pigs := 6
  let pigs_bushels_per_day := 4
  let horses := 2
  let horses_bushels_per_day := 5
  cows * cows_bushels_per_day +
  sheep * sheep_bushels_per_day +
  chickens * chickens_bushels_per_day +
  pigs * pigs_bushels_per_day +
  horses * horses_bushels_per_day = 65 := by
  sorry

end NUMINAMATH_GPT_dany_farm_bushels_l51_5120


namespace NUMINAMATH_GPT_find_intervals_of_monotonicity_find_value_of_a_l51_5194

noncomputable def f (x a : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6) + a + 1

theorem find_intervals_of_monotonicity (k : ℤ) (a : ℝ) :
  ∀ x ∈ Set.Icc (k * Real.pi - Real.pi / 6) (k * Real.pi + Real.pi / 3), MonotoneOn (λ x => f x a) (Set.Icc (k * Real.pi - Real.pi / 6) (k * Real.pi + Real.pi / 3)) :=
sorry

theorem find_value_of_a (a : ℝ) (max_value_condition : ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x a ≤ 4) :
  a = 1 :=
sorry

end NUMINAMATH_GPT_find_intervals_of_monotonicity_find_value_of_a_l51_5194


namespace NUMINAMATH_GPT_remainder_3_pow_1503_mod_7_l51_5184

theorem remainder_3_pow_1503_mod_7 : 
  (3 ^ 1503) % 7 = 6 := 
by sorry

end NUMINAMATH_GPT_remainder_3_pow_1503_mod_7_l51_5184


namespace NUMINAMATH_GPT_complement_union_correct_l51_5166

noncomputable def U : Set ℕ := {2, 3, 4, 5, 6}
noncomputable def A : Set ℕ := {x | x^2 - 6*x + 8 = 0}
noncomputable def B : Set ℕ := {2, 5, 6}

theorem complement_union_correct :
  (U \ A) ∪ B = {2, 3, 5, 6} := 
by
  sorry

end NUMINAMATH_GPT_complement_union_correct_l51_5166


namespace NUMINAMATH_GPT_stratified_sampling_l51_5129

-- Definition of conditions as hypothesis
def total_employees : ℕ := 100
def under_35 : ℕ := 45
def between_35_49 : ℕ := 25
def over_50 : ℕ := total_employees - under_35 - between_35_49
def sample_size : ℕ := 20
def sampling_ratio : ℚ := sample_size / total_employees

-- The target number of people from each group
def under_35_sample : ℚ := sampling_ratio * under_35
def between_35_49_sample : ℚ := sampling_ratio * between_35_49
def over_50_sample : ℚ := sampling_ratio * over_50

-- Problem statement
theorem stratified_sampling : 
  under_35_sample = 9 ∧ 
  between_35_49_sample = 5 ∧ 
  over_50_sample = 6 :=
  by
  sorry

end NUMINAMATH_GPT_stratified_sampling_l51_5129


namespace NUMINAMATH_GPT_carl_speed_l51_5196

theorem carl_speed 
  (time : ℝ) (distance : ℝ) 
  (h_time : time = 5) 
  (h_distance : distance = 10) 
  : (distance / time) = 2 :=
by
  rw [h_time, h_distance]
  sorry

end NUMINAMATH_GPT_carl_speed_l51_5196


namespace NUMINAMATH_GPT_weight_shifted_count_l51_5151

def is_weight_shifted (a b x y : ℕ) : Prop :=
  a + b = 2 * (x + y) ∧ 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9

theorem weight_shifted_count : 
  ∃ count : ℕ, count = 225 ∧ 
  (∀ (a b x y : ℕ), is_weight_shifted a b x y → count = 225) := 
sorry

end NUMINAMATH_GPT_weight_shifted_count_l51_5151


namespace NUMINAMATH_GPT_ratio_addition_l51_5141

theorem ratio_addition (x : ℤ) (h : 4 + x = 3 * (15 + x) / 4): x = 29 :=
by
  sorry

end NUMINAMATH_GPT_ratio_addition_l51_5141


namespace NUMINAMATH_GPT_find_unknown_rate_l51_5131

variable (x : ℕ)

theorem find_unknown_rate
    (c3 : ℕ := 3 * 100)
    (c5 : ℕ := 5 * 150)
    (n : ℕ := 10)
    (avg_price : ℕ := 160) 
    (h : c3 + c5 + 2 * x = avg_price * n) :
    x = 275 := 
by
  -- Proof goes here.
  sorry

end NUMINAMATH_GPT_find_unknown_rate_l51_5131


namespace NUMINAMATH_GPT_find_m_collinear_l51_5115

theorem find_m_collinear (m : ℝ) 
    (a : ℝ × ℝ := (m + 3, 2)) 
    (b : ℝ × ℝ := (m, 1)) 
    (collinear : a.1 * 1 - 2 * b.1 = 0) : 
    m = 3 :=
by {
    sorry
}

end NUMINAMATH_GPT_find_m_collinear_l51_5115


namespace NUMINAMATH_GPT_DVDs_per_season_l51_5182

theorem DVDs_per_season (total_DVDs : ℕ) (seasons : ℕ) (h1 : total_DVDs = 40) (h2 : seasons = 5) : total_DVDs / seasons = 8 :=
by
  sorry

end NUMINAMATH_GPT_DVDs_per_season_l51_5182


namespace NUMINAMATH_GPT_expression_value_l51_5128

-- The problem statement definition
def expression := 2 + 3 * 4 - 5 / 5 + 7

-- Theorem statement asserting the final result
theorem expression_value : expression = 20 := 
by sorry

end NUMINAMATH_GPT_expression_value_l51_5128


namespace NUMINAMATH_GPT_least_number_to_subtract_l51_5144

theorem least_number_to_subtract (n : ℕ) (h : n = 42739) : 
    ∃ k, k = 4 ∧ (n - k) % 15 = 0 := by
  sorry

end NUMINAMATH_GPT_least_number_to_subtract_l51_5144


namespace NUMINAMATH_GPT_halfway_fraction_l51_5175

theorem halfway_fraction (a b : ℚ) (h_a : a = 3 / 4) (h_b : b = 5 / 7) :
  ((a + b) / 2) = 41 / 56 :=
by
  rw [h_a, h_b]
  sorry

end NUMINAMATH_GPT_halfway_fraction_l51_5175


namespace NUMINAMATH_GPT_betty_needs_more_flies_l51_5102

-- Definitions for the number of flies consumed by the frog each day
def fliesMonday : ℕ := 3
def fliesTuesday : ℕ := 2
def fliesWednesday : ℕ := 4
def fliesThursday : ℕ := 5
def fliesFriday : ℕ := 1
def fliesSaturday : ℕ := 2
def fliesSunday : ℕ := 3

-- Definition for the total number of flies eaten by the frog in a week
def totalFliesEaten : ℕ :=
  fliesMonday + fliesTuesday + fliesWednesday + fliesThursday + fliesFriday + fliesSaturday + fliesSunday

-- Definitions for the number of flies caught by Betty
def fliesMorning : ℕ := 5
def fliesAfternoon : ℕ := 6
def fliesEscaped : ℕ := 1

-- Definition for the total number of flies caught by Betty considering the escape
def totalFliesCaught : ℕ := fliesMorning + fliesAfternoon - fliesEscaped

-- Lean 4 statement to prove the number of additional flies Betty needs to catch
theorem betty_needs_more_flies : 
  totalFliesEaten - totalFliesCaught = 10 := 
by
  sorry

end NUMINAMATH_GPT_betty_needs_more_flies_l51_5102


namespace NUMINAMATH_GPT_total_amount_divided_l51_5117

variables (T x : ℝ)
variables (h₁ : 0.06 * x + 0.05 * (T - x) = 85) 
variables (h₂ : T - x = 1100)

theorem total_amount_divided (T x : ℝ) 
  (h₁ : 0.06 * x + 0.05 * (T - x) = 85) 
  (h₂ : T - x = 1100) : 
  T = 1600 := 
sorry

end NUMINAMATH_GPT_total_amount_divided_l51_5117


namespace NUMINAMATH_GPT_minimum_value_l51_5157

theorem minimum_value (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_sum : a + b = 2) :
  (1 / a) + (1 / b) ≥ 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_minimum_value_l51_5157


namespace NUMINAMATH_GPT_ten_times_ten_thousand_ten_times_one_million_ten_times_ten_million_tens_of_thousands_in_hundred_million_l51_5105

theorem ten_times_ten_thousand : 10 * 10000 = 100000 :=
by sorry

theorem ten_times_one_million : 10 * 1000000 = 10000000 :=
by sorry

theorem ten_times_ten_million : 10 * 10000000 = 100000000 :=
by sorry

theorem tens_of_thousands_in_hundred_million : 100000000 / 10000 = 10000 :=
by sorry

end NUMINAMATH_GPT_ten_times_ten_thousand_ten_times_one_million_ten_times_ten_million_tens_of_thousands_in_hundred_million_l51_5105


namespace NUMINAMATH_GPT_marathon_finishers_l51_5197

-- Define the conditions
def totalParticipants : ℕ := 1250
def peopleGaveUp (F : ℕ) : ℕ := F + 124

-- Define the final statement to be proved
theorem marathon_finishers (F : ℕ) (h1 : totalParticipants = F + peopleGaveUp F) : F = 563 :=
by sorry

end NUMINAMATH_GPT_marathon_finishers_l51_5197


namespace NUMINAMATH_GPT_age_difference_is_18_l51_5135

variable (A B C : ℤ)
variable (h1 : A + B > B + C)
variable (h2 : C = A - 18)

theorem age_difference_is_18 : (A + B) - (B + C) = 18 :=
by
  sorry

end NUMINAMATH_GPT_age_difference_is_18_l51_5135


namespace NUMINAMATH_GPT_number_of_suits_sold_l51_5179

theorem number_of_suits_sold
  (commission_rate: ℝ)
  (price_per_suit: ℝ)
  (price_per_shirt: ℝ)
  (price_per_loafer: ℝ)
  (number_of_shirts: ℕ)
  (number_of_loafers: ℕ)
  (total_commission: ℝ)
  (suits_sold: ℕ)
  (total_sales: ℝ)
  (total_sales_from_non_suits: ℝ)
  (sales_needed_from_suits: ℝ)
  : 
  (commission_rate = 0.15) → 
  (price_per_suit = 700.0) → 
  (price_per_shirt = 50.0) → 
  (price_per_loafer = 150.0) → 
  (number_of_shirts = 6) → 
  (number_of_loafers = 2) → 
  (total_commission = 300.0) →
  (total_sales = total_commission / commission_rate) →
  (total_sales_from_non_suits = number_of_shirts * price_per_shirt + number_of_loafers * price_per_loafer) →
  (sales_needed_from_suits = total_sales - total_sales_from_non_suits) →
  (suits_sold = sales_needed_from_suits / price_per_suit) →
  suits_sold = 2 :=
by
  sorry

end NUMINAMATH_GPT_number_of_suits_sold_l51_5179


namespace NUMINAMATH_GPT_manuscript_pages_l51_5112

theorem manuscript_pages (P : ℝ)
  (h1 : 10 * (0.05 * P) + 10 * 5 = 250) : P = 400 :=
sorry

end NUMINAMATH_GPT_manuscript_pages_l51_5112


namespace NUMINAMATH_GPT_inverse_proportion_quadrants_l51_5156

theorem inverse_proportion_quadrants (m : ℝ) : (∀ (x : ℝ), x ≠ 0 → y = (m - 2) / x → (x > 0 ∧ y > 0 ∨ x < 0 ∧ y < 0)) ↔ m > 2 :=
by
  sorry

end NUMINAMATH_GPT_inverse_proportion_quadrants_l51_5156


namespace NUMINAMATH_GPT_minimize_G_l51_5140

noncomputable def F (p q : ℝ) : ℝ :=
  2 * p * q + 4 * p * (1 - q) + 2 * (1 - p) * q - 5 * (1 - p) * (1 - q)

noncomputable def G (p : ℝ) : ℝ :=
  max (F p 0) (F p 1)

theorem minimize_G :
  ∀ (p : ℝ), 0 ≤ p ∧ p ≤ 0.75 → G p = G 0 → p = 0 :=
by
  intro p hp hG
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_minimize_G_l51_5140


namespace NUMINAMATH_GPT_equivalent_statements_l51_5114

variable (P Q : Prop)

theorem equivalent_statements : 
  (P → Q) ↔ ((¬ Q → ¬ P) ∧ (¬ P ∨ Q)) :=
by 
  sorry

end NUMINAMATH_GPT_equivalent_statements_l51_5114


namespace NUMINAMATH_GPT_total_toys_correct_l51_5163

-- Define the conditions
def JaxonToys : ℕ := 15
def GabrielToys : ℕ := 2 * JaxonToys
def JerryToys : ℕ := GabrielToys + 8

-- Define the total number of toys
def TotalToys : ℕ := JaxonToys + GabrielToys + JerryToys

-- State the theorem
theorem total_toys_correct : TotalToys = 83 :=
by
  -- Skipping the proof as per instruction
  sorry

end NUMINAMATH_GPT_total_toys_correct_l51_5163


namespace NUMINAMATH_GPT_trig_expression_zero_l51_5192

theorem trig_expression_zero (θ : ℝ) (h : Real.tan θ = 3) :
  (1 - Real.cos θ) / Real.sin θ - Real.sin θ / (1 + Real.cos θ) = 0 :=
sorry

end NUMINAMATH_GPT_trig_expression_zero_l51_5192


namespace NUMINAMATH_GPT_inequality_proof_l51_5177

theorem inequality_proof (s r : ℝ) (h1 : s > 0) (h2 : r > 0) (h3 : r < s) :
  (s^3 - r^3) / (s^3 + r^3) > (s^2 - r^2) / (s^2 + r^2) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l51_5177


namespace NUMINAMATH_GPT_no_such_function_exists_l51_5158

open Set

theorem no_such_function_exists
  (f : ℝ → ℝ)
  (h1 : ∀ x y : ℝ, 0 < x → 0 < y → y > x → f y > (y - x) * f x ^ 2) :
  False :=
sorry

end NUMINAMATH_GPT_no_such_function_exists_l51_5158
