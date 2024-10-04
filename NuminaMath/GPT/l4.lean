import Mathlib
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Order.AbsoluteValue
import Mathlib.Analysis.Calculus.Integral
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.Trigonometry.Basic
import Mathlib.Combinatorics
import Mathlib.Combinatorics.Binomial
import Mathlib.Combinatorics.CombinatorialSpecies
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Gcd.Basic
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Quad
import Mathlib.MeasureTheory.Probability.ProbabilityMassFunction
import Mathlib.ProbabilityTheory.ProbabilitySpace
import Mathlib.Tactic

namespace shifted_cosine_function_l4_4957

noncomputable def original_function : ℝ → ℝ :=
  λ x, 3 * Real.cos (0.5 * x - Real.pi / 3)

noncomputable def shifted_function : ℝ → ℝ :=
  λ x, 3 * Real.cos (0.5 * x - Real.pi / 12)

theorem shifted_cosine_function :
  original_function (x + 2 * Real.pi / 8) = shifted_function x := by
  sorry

end shifted_cosine_function_l4_4957


namespace complex_solution_l4_4260

open Complex

theorem complex_solution (z : ℂ) (h : z + Complex.abs z = 1 + Complex.I) : z = Complex.I := 
by
  sorry

end complex_solution_l4_4260


namespace polynomial_sum_l4_4064

noncomputable def g (a b c d : ℝ) (x : ℂ) : ℂ := x^4 + a * x^3 + b * x^2 + c * x + d

theorem polynomial_sum : ∃ a b c d : ℝ, 
  (g a b c d (-3 * Complex.I) = 0) ∧
  (g a b c d (1 + Complex.I) = 0) ∧
  (g a b c d (3 * Complex.I) = 0) ∧
  (g a b c d (1 - Complex.I) = 0) ∧ 
  (a + b + c + d = 9) := by
  sorry

end polynomial_sum_l4_4064


namespace graph_eq_y_eq_x_l4_4616

section
variable {a : ℝ} (h1 : a > 0) (h2 : a ≠ 1)

theorem graph_eq_y_eq_x : ∀ x : ℝ, log a (a ^ x) = x :=
by
  -- This will be the place for the proof
  sorry
end

end graph_eq_y_eq_x_l4_4616


namespace pseudocode_result_l4_4109

def pseudocode_output : Nat := 
  let mut I := 2
  for n in [2, 4, 6, 8, 10] do
    I := 2 * I
    if I > 20 then
      I := I - 20
  I
-- pseudocode_output should be equal to 16
  
theorem pseudocode_result : pseudocode_output = 16 := by
  sorry

end pseudocode_result_l4_4109


namespace percentage_of_exports_from_fruits_l4_4162

theorem percentage_of_exports_from_fruits:
  (orange_exports yearly_exports : ℝ) 
  (H₁ : orange_exports = 4.25) 
  (H₂ : yearly_exports = 127.5) 
  (H₃ : 6 * orange_exports = 25.5) 
  : (25.5 / yearly_exports) * 100 = 20 := 
sorry

end percentage_of_exports_from_fruits_l4_4162


namespace z_when_y_six_l4_4041

theorem z_when_y_six
    (k : ℝ)
    (h1 : ∀ y (z : ℝ), y^2 * Real.sqrt z = k)
    (h2 : ∃ (y : ℝ) (z : ℝ), y = 3 ∧ z = 4 ∧ y^2 * Real.sqrt z = k) :
  ∃ z : ℝ, y = 6 ∧ z = 1 / 4 := 
sorry

end z_when_y_six_l4_4041


namespace cos_alpha_value_l4_4833

theorem cos_alpha_value (α : ℝ) (h₁ : 0 < α ∧ α < π/2)
  (h₂ : cos (α + π/4) = -3/5) (h₃ : sin (α + π/4) = 4/5) :
  cos α = sqrt 2 / 10 :=
by
  sorry

end cos_alpha_value_l4_4833


namespace feet_of_pipe_per_bolt_l4_4724

-- Definition of the initial conditions
def total_pipe_length := 40 -- total feet of pipe
def washers_per_bolt := 2
def initial_washers := 20
def remaining_washers := 4

-- The proof statement
theorem feet_of_pipe_per_bolt :
  ∀ (total_pipe_length washers_per_bolt initial_washers remaining_washers : ℕ),
  initial_washers - remaining_washers = 16 → -- 16 washers used
  16 / washers_per_bolt = 8 → -- 8 bolts used
  total_pipe_length / 8 = 5 :=
by
  intros
  sorry

end feet_of_pipe_per_bolt_l4_4724


namespace number_of_integers_in_union_l4_4811

def A : Set ℝ := { x | x^2 + 2 * x - 8 < 0 }
def B : Set ℝ := { x | |x - 1| < 1 }
def union_set := { x | -4 < x ∧ x < 2 }
    
theorem number_of_integers_in_union : 
    (∃ n, n = 5 ∧ {x : ℤ | x ∈ union_set }.card = n) :=
    sorry

end number_of_integers_in_union_l4_4811


namespace largest_possible_d_l4_4889

theorem largest_possible_d (a b c d : ℝ) (h1 : a + b + c + d = 10) (h2 : ab + ac + ad + bc + bd + cd = 20) : 
    d ≤ (5 + Real.sqrt 105) / 2 :=
by
  sorry

end largest_possible_d_l4_4889


namespace metallic_sheet_width_l4_4654

theorem metallic_sheet_width :
  ∀ (l s v w : ℝ),
    l = 50 ∧ s = 8 ∧ v = 5440 →
    (v = (l - 2 * s) * (w - 2 * s) * s) →
    w = 36 :=
by
  intros l s v w h₁ h₂
  cases h₁ with h_l hs_v
  cases hs_v with hs hv
  subst_vars
  sorry

end metallic_sheet_width_l4_4654


namespace grooming_time_equals_640_seconds_l4_4868

variable (cat_claws_per_foot : Nat) (cat_foot_count : Nat)
variable (nissa_clip_time_per_claw : Nat) (nissa_clean_time_per_ear : Nat) (nissa_shampoo_time_minutes : Nat) 
variable (cat_ear_count : Nat)
variable (seconds_per_minute : Nat)

def total_grooming_time (cat_claws_per_foot * cat_foot_count : nissa_clip_time_per_claw) (nissa_clean_time_per_ear * cat_ear_count) (nissa_shampoo_time_minutes * seconds_per_minute) := sorry

theorem grooming_time_equals_640_seconds : 
  cat_claws_per_foot = 4 →
  cat_foot_count = 4 →
  nissa_clip_time_per_claw = 10 →
  nissa_clean_time_per_ear = 90 →
  nissa_shampoo_time_minutes = 5 →
  cat_ear_count = 2 →
  seconds_per_minute = 60 →
  total_grooming_time = 160 + 180 + 300 → 
  total_grooming_time = 640 := sorry

end grooming_time_equals_640_seconds_l4_4868


namespace total_female_employees_l4_4545

variable (E M Male_E : ℕ)
variable (h1 : M = 2 / 5 * E)
variable (h2 : (2 / 5) * Male_E = M - 280)
variable (h3 : E - Male_E = 280 + 2 / 5 * E)

theorem total_female_employees : ∃ Female_E, E - Male_E = 700 :=
by
  existsi 700
  sorry

end total_female_employees_l4_4545


namespace foma_gives_ierema_55_l4_4552

-- Defining the amounts that Foma, Ierema, and Yuliy have initially
variables (F E Y : ℝ)

-- Given conditions
def condition_1 : Prop := F - E = 140
def condition_2 : Prop := F - Y = 40
def condition_3 : Prop := Y = E + 70

-- Prove that Foma should give Ierema 55 gold coins
theorem foma_gives_ierema_55 (h1 : condition_1) (h2 : condition_2) (h3 : condition_3) : 
  (F - E) / 2 = 55 :=
by
  sorry

end foma_gives_ierema_55_l4_4552


namespace exists_squares_sum_l4_4374

theorem exists_squares_sum (p k : ℤ) (hp : prime p) (hp_odd : odd p) : 
  ∃ (a b : ℤ), p ∣ (a^2 + b^2 - k) :=
sorry

end exists_squares_sum_l4_4374


namespace investment_duration_p_l4_4529

-- Given the investments ratio, profits ratio, and time period for q,
-- proving the time period of p's investment is 7 months.
theorem investment_duration_p (T_p T_q : ℕ) 
  (investment_ratio : 7 * T_p = 5 * T_q) 
  (profit_ratio : 7 * T_p / T_q = 7 / 10)
  (T_q_eq : T_q = 14) : T_p = 7 :=
by
  sorry

end investment_duration_p_l4_4529


namespace smallest_c_for_inverse_l4_4895

-- Define the function f
def f (x : ℝ) : ℝ := (x + 1)^2 - 3

-- Prove that the smallest value of c for which f restricted to [c, ∞) is invertible is -1
theorem smallest_c_for_inverse : ∃ c : ℝ, (∀ x1 x2 ∈ set.Ici c, x1 ≠ x2 → f x1 ≠ f x2) ∧ (c = -1) :=
by
  sorry

end smallest_c_for_inverse_l4_4895


namespace solution_sum_of_eq_zero_l4_4116

open Real

theorem solution_sum_of_eq_zero : 
  let f (x : ℝ) := (4*x + 6) * (3*x - 8)
  in (∀ x, f x = 0 → x = -3/2 ∨ x = 8/3) → 
     (-3/2 + 8/3 = 7/6) :=
by 
  let f (x : ℝ) := (4*x + 6) * (3*x - 8)
  intro h
  have h₁ : f(-3/2) = 0 := by sorry
  have h₂ : f(8/3) = 0 := by sorry
  have sum_of_solutions : -3/2 + 8/3 = 7/6 := by 
    sorry
  exact sum_of_solutions

end solution_sum_of_eq_zero_l4_4116


namespace intersection_range_l4_4809

-- Define the line equation
def line (k x : ℝ) : ℝ := k * x - k + 1

-- Define the curve equation
def curve (x y m : ℝ) : Prop := x^2 + 2 * y^2 = m

-- State the problem: Given the line and the curve have a common point, prove the range of m is m >= 3
theorem intersection_range (k m : ℝ) (h : ∃ x y, line k x = y ∧ curve x y m) : m ≥ 3 :=
by {
  sorry
}

end intersection_range_l4_4809


namespace probability_of_ascending_two_digit_number_l4_4212

def is_ascending (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 < d2

theorem probability_of_ascending_two_digit_number :
  let total := 90
  let ascending_count := 36
  let probability := ascending_count / total
  probability = (2/5 : ℚ) :=
begin
  sorry
end

end probability_of_ascending_two_digit_number_l4_4212


namespace D_nonnegative_l4_4022

def f (n : ℕ) : ℕ := 
  -- Placeholder for the actual implementation of f
  sorry

def g (n : ℕ) : ℕ := 
  -- Placeholder for the actual implementation of g
  sorry

def D (n : ℕ) : ℕ := f(n) - g(n)

theorem D_nonnegative (n : ℕ) : D(n) ≥ 0 := 
  sorry

end D_nonnegative_l4_4022


namespace total_apples_eq_l4_4078

-- Define the conditions for the problem
def baskets : ℕ := 37
def apples_per_basket : ℕ := 17

-- Define the theorem to prove the total number of apples
theorem total_apples_eq : baskets * apples_per_basket = 629 :=
by
  sorry

end total_apples_eq_l4_4078


namespace solve_trig_eq_l4_4506

open Real -- Open real number structure

theorem solve_trig_eq (x : ℝ) :
  (sin x)^2 + (sin (2 * x))^2 + (sin (3 * x))^2 = 2 ↔ 
  (∃ n : ℤ, x = π / 4 + (π * n) / 2)
  ∨ (∃ n : ℤ, x = π / 2 + π * n)
  ∨ (∃ n : ℤ, x = π / 6 + π * n ∨ x = -π / 6 + π * n) := by sorry

end solve_trig_eq_l4_4506


namespace given_roots_find_coefficients_l4_4245

theorem given_roots_find_coefficients {a b c : ℝ} :
  (1:ℝ)^5 + 2*(1)^4 + a * (1:ℝ)^2 + b * (1:ℝ) = c →
  (-1:ℝ)^5 + 2*(-1:ℝ)^4 + a * (-1:ℝ)^2 + b * (-1:ℝ) = c →
  a = -6 ∧ b = -1 ∧ c = -4 :=
by
  intros h1 h2
  sorry

end given_roots_find_coefficients_l4_4245


namespace remainder_is_12_l4_4611

noncomputable def dividend : Polynomial ℝ := 2 * X^2 - 17 * X + 47
noncomputable def divisor : Polynomial ℝ := X - 5

theorem remainder_is_12 :
  ∃ q : Polynomial ℝ, ∃ r : ℝ, dividend = divisor * q + Polynomial.C r ∧ r = 12 := 
sorry

end remainder_is_12_l4_4611


namespace total_length_T_l4_4389

def T : set (ℝ × ℝ) := { p | ∃ x y : ℝ, p = (x, y) ∧ 
  (| (| x | - 3) - 2 | + | (| y | - 3) - 2 | = 2) }

theorem total_length_T : 
  (∃ l : ℝ, l = 128 * real.sqrt 2 ∧ ∀ (p ∈ T), ∃ s : list (set (ℝ × ℝ)), 
    (p ∈ ⋃₀ set_of (λ t, t ∈ s) ∧ 
    (∀ t ∈ s, is_diamond t) ∧ 
    (∀ t ∈ s, perimeter t = 8 * real.sqrt 2) ∧ 
    length_of_lines s = l)) :=
sorry

end total_length_T_l4_4389


namespace medians_perpendicular_l4_4971

-- Define the triangle sides and the condition
variables {a b c : ℝ}

-- Define medians using vectors notation, but without relying on vector module specifics
noncomputable def median_1 : ℝ := c + 0.5 * a
noncomputable def median_2 : ℝ := a + 0.5 * b

-- Main theorem statement that combines both the direct proof and the converse
theorem medians_perpendicular (h : a^2 + b^2 = 5 * c^2) : 
  ((median_1 c a) * (median_2 a b) = 0) ↔ (a^2 + b^2 = 5 * c^2) :=
sorry

end medians_perpendicular_l4_4971


namespace floor_identity_l4_4020

theorem floor_identity (x : ℝ) : 
    (⌊(3 + x) / 6⌋ - ⌊(4 + x) / 6⌋ + ⌊(5 + x) / 6⌋ = ⌊(1 + x) / 2⌋ - ⌊(1 + x) / 3⌋) :=
by
  sorry

end floor_identity_l4_4020


namespace range_of_bc_div_a_l4_4361

theorem range_of_bc_div_a (a b c : ℝ) (A B C : ℝ)
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π)
  (h_sides : c = a + b - ab ∧ b = a - c + c)
  (h_angles_sides : B = atan (b / a) ∧ C = π - A - B) :
  (1 < (b + c) / a) ∧ ((b + c) / a ≤ 2) :=
by
  sorry

end range_of_bc_div_a_l4_4361


namespace largest_d_value_l4_4887

noncomputable def max_d (a b c d : ℝ) (h1 : a + b + c + d = 10) (h2 : ab + ac + ad + bc + bd + cd = 20) : ℝ :=
  if h : (4 * d ^ 2 - 20 * d - 80) ≤ 0 then d else 0

theorem largest_d_value (a b c d : ℝ) (h1 : a + b + c + d = 10) (h2 : ab + ac + ad + bc + bd + cd = 20) :
  max_d a b c d h1 h2 = (5 + 5 * real.sqrt 21) / 2 :=
sorry

end largest_d_value_l4_4887


namespace total_length_of_T_l4_4386

noncomputable def T : Set (ℝ × ℝ) :=
  { p : ℝ × ℝ | let (x, y) := p in abs (abs x - 3 - 2) + abs (abs y - 3 - 2) = 2 }

theorem total_length_of_T : 
  (4 * 8 * Real.sqrt 2) = 128 * Real.sqrt 2 :=
by
  sorry

end total_length_of_T_l4_4386


namespace exists_n_with_sum_of_digits_and_divisible_l4_4486

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits_b 10 |>.sum

theorem exists_n_with_sum_of_digits_and_divisible (s : ℕ) (h : s > 0) :
  ∃ n : ℕ, sum_of_digits n = s ∧ s ∣ n :=
by sorry

end exists_n_with_sum_of_digits_and_divisible_l4_4486


namespace consistent_number_proof_l4_4240

-- Define the conditions
def isConsistentNumber (m : ℕ) : Prop :=
  let a := m / 1000
  let b := (m % 1000) / 100
  let c := (m % 100) / 10
  let d := m % 10
  d = 1 ∧ 1 ≤ a ∧ a ≤ 8 ∧ a + b = c + 1

def swapDigits (m : ℕ) : ℕ :=
  let a := m / 1000
  let b := (m % 1000) / 100
  let c := (m % 100) / 10
  let d := m % 10
  1000 * c + 100 * d + 10 * a + b

def F (m : ℕ) : ℕ :=
  (m + swapDigits m) / 101

def G (N : ℕ) : ℕ :=
  let a := N / 10
  let b := N % 10
  if b ≤ 4 then a + 2 * b else a + 2 * b - 9

-- k value equation
def kEquation (m N : ℕ) (k : ℕ) : Prop :=
  F m - G N - 4 * (m / 1000) = k^2 + 3

-- Lean 4 statement to prove
theorem consistent_number_proof :
  ∃ k m, isConsistentNumber m ∧ G (m / 111)  % 2 = 0 ∧ kEquation m (m / 111) k ∧ m = 2231 ∧ (k = 6 ∨ k = -6) := 
sorry

end consistent_number_proof_l4_4240


namespace find_coordinates_B_l4_4776

variable (B : ℝ × ℝ)

def A : ℝ × ℝ := (2, 3)
def C : ℝ × ℝ := (0, 1)
def vec (P Q : ℝ × ℝ) : ℝ × ℝ := (Q.1 - P.1, Q.2 - P.2)

theorem find_coordinates_B (h : vec A B = (-2) • vec B C) : B = (-2, 5/3) :=
by
  -- Here you would provide proof steps
  sorry

end find_coordinates_B_l4_4776


namespace smallest_weights_to_measure_1_to_100_l4_4613

-- Define the problem as a theorem
theorem smallest_weights_to_measure_1_to_100 :
    (∀ n : ℕ, 1 ≤ n ∧ n ≤ 100 → ∃ (weights : list ℕ), (∀ w ∈ weights, ∃ k : ℕ, w = 2^k) ∧ list.sum weights = n) ∧
    (∀ w : list ℕ, (∀ w' ∈ w, ∃ k : ℕ, w' = 2^k) ∧ (∀ n : ℕ, 1 ≤ n ∧ n ≤ 100 → ∃ (subset : list ℕ), (∀ s ∈ subset, s ∈ w) ∧ list.sum subset = n) → w.length ≥ 7) :=
sorry

end smallest_weights_to_measure_1_to_100_l4_4613


namespace number_of_ways_to_choose_students_l4_4973

theorem number_of_ways_to_choose_students :
  let female_students := 4
  let male_students := 3
  (female_students * male_students) = 12 :=
by
  sorry

end number_of_ways_to_choose_students_l4_4973


namespace Dvaneft_percentage_bounds_l4_4650

noncomputable def percentageDvaneftShares (x y z : ℤ) (n m : ℕ) : ℚ :=
  n * 100 / (2 * (n + m))

theorem Dvaneft_percentage_bounds
  (x y z : ℤ) (n m : ℕ)
  (h1 : 4 * x * n = y * m)
  (h2 : x * n + y * m = z * (n + m))
  (h3 : 16 ≤ y - x)
  (h4 : y - x ≤ 20)
  (h5 : 42 ≤ z)
  (h6 : z ≤ 60) :
  12.5 ≤ percentageDvaneftShares x y z n m ∧ percentageDvaneftShares x y z n m ≤ 15 := by
  sorry

end Dvaneft_percentage_bounds_l4_4650


namespace coeff_expansion_l4_4336

theorem coeff_expansion (a : ℚ) (h : a = 1/2) :
  binom 6 3 * (-a)^3 * 2^3 = -20 :=
sorry

end coeff_expansion_l4_4336


namespace average_visitors_30_day_month_l4_4135

def visitors_per_day (total_visitors : ℕ) (days : ℕ) : ℕ := total_visitors / days

theorem average_visitors_30_day_month (visitors_sunday : ℕ) (visitors_other_days : ℕ) 
  (total_days : ℕ) (sundays : ℕ) (other_days : ℕ) :
  visitors_sunday = 510 →
  visitors_other_days = 240 →
  total_days = 30 →
  sundays = 4 →
  other_days = 26 →
  visitors_per_day (sundays * visitors_sunday + other_days * visitors_other_days) total_days = 276 :=
by
  intros h1 h2 h3 h4 h5
  -- Proof goes here
  sorry

end average_visitors_30_day_month_l4_4135


namespace find_ratio_l4_4907

variable {a b : ℝ} (hnza : a ≠ 0) (hnzb : b ≠ 0)
          (hpure_imag : (3 - 4 * Complex.i) * (a + b * Complex.i)).im ≠ 0

theorem find_ratio (hnza : a ≠ 0) (hnzb : b ≠ 0) (hpure_imag : Complex.re ((3 - 4 * Complex.i) * (a + b * Complex.i)) = 0) : a / b = -4 / 3 := sorry

end find_ratio_l4_4907


namespace fomagive_55_l4_4589

variables (F E Y : ℕ)

-- Conditions
def condition1 := F - E = 140
def condition2 := F - Y = 40
def condition3 := Y = E + 70

-- Proposition: Foma should give 55 coins to Ierema
theorem fomagive_55 (h1 : condition1 F E) (h2 : condition2 F Y) (h3 : condition3 E Y) : ∃ G, G = 55 :=
by
  have := h1,
  have := h2,
  have := h3,
  sorry

end fomagive_55_l4_4589


namespace foma_should_give_ierema_55_coins_l4_4560

theorem foma_should_give_ierema_55_coins (F E Y : ℤ)
  (h1 : F - 70 = E + 70)
  (h2 : F - 40 = Y)
  (h3 : E + 70 = Y) :
  F - E = 110 → F - E = 55 :=
by
  sorry

end foma_should_give_ierema_55_coins_l4_4560


namespace at_least_one_negative_l4_4013

theorem at_least_one_negative (a : Fin 7 → ℤ) :
  (∀ i j : Fin 7, i ≠ j → a i ≠ a j) ∧
  (∀ l1 l2 l3 : Fin 7, 
    a l1 + a l2 + a l3 = a l1 + a l2 + a l3) ∧
  (∃ i : Fin 7, a i = 0) →
  (∃ i : Fin 7, a i < 0) :=
  by
  sorry

end at_least_one_negative_l4_4013


namespace field_ratio_l4_4959

theorem field_ratio
  (l w : ℕ)
  (pond_length : ℕ)
  (pond_area_ratio : ℚ)
  (field_length : ℕ)
  (field_area : ℕ)
  (hl : l = 24)
  (hp : pond_length = 6)
  (hr : pond_area_ratio = 1 / 8)
  (hm : l % w = 0)
  (ha : field_area = 36 * 8)
  (hf : l * w = field_area) :
  l / w = 2 :=
by
  sorry

end field_ratio_l4_4959


namespace mean_problem_l4_4962

theorem mean_problem : 
  (8 + 12 + 24) / 3 = (16 + z) / 2 → z = 40 / 3 :=
by
  intro h
  sorry

end mean_problem_l4_4962


namespace two_digit_primes_l4_4314

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def reverse_digits (n : ℕ) : ℕ :=
  let t := n / 10
  let u := n % 10
  10 * u + t

def is_valid_n (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ is_prime (n + reverse_digits n)

theorem two_digit_primes (N : ℕ) : ∃! n, is_valid_n n :=
  ∃! n, n = 10 :=
begin
  sorry
end

end two_digit_primes_l4_4314


namespace diameter_increase_factor_l4_4615

theorem diameter_increase_factor (V : ℝ) (d : ℝ) (h : V = (π * d^3) / 6) :
  let d2 := (∛2) * d in
  let V2 := 2 * V in
  V2 = (π * d2^3) / 6 :=
by sorry

end diameter_increase_factor_l4_4615


namespace problem_statement_l4_4342

variables {Ω : Type*} [probability_space Ω]
def P (E : event Ω) : ℝ := probability E

-- Definitions of events as sets
def A : event Ω := {ω | ω ∈ γυναίκες ω}, -- All three students are female
def B : event Ω := {ω | ω ∈ άνδρες ω}, -- All three students are male
def C : event Ω := {ω | ∃ x ω, ω ∉ γυναίκες ω}, -- At least one male student
def D : event Ω := {ω | ω ∉ γυναίκες ω} -- Not all students are female

-- Probability values
axiom P_women : P (A) = (1/2) ^ 3

theorem problem_statement : 
  (P (A) = 1 / 8) ∧
  (disjoint A B) ∧
  (Aᶜ = C) :=
by
  sorry

end problem_statement_l4_4342


namespace gcd_hcf_of_36_and_84_l4_4743

theorem gcd_hcf_of_36_and_84 : Nat.gcd 36 84 = 12 := sorry

end gcd_hcf_of_36_and_84_l4_4743


namespace find_angle_between_vectors_l4_4284

noncomputable def angle_between_vectors 
  (a b : ℝ) (theta : ℝ) : Prop :=
  let mag_a := 1
  let mag_b := 2
  let mag_sum := sqrt 7
  7 = mag_a^2 + mag_b^2 + 2 * mag_a * mag_b * real.cos theta ∧ theta = real.arccos 0.5

theorem find_angle_between_vectors 
  (a b : ℝ) (theta : ℝ) 
  (ha : ∥a∥ = 1) 
  (hb : ∥b∥ = 2) 
  (hab : ∥a + b∥ = sqrt 7) : 
  theta = real.arccos (1 / 2) :=
begin
  sorry
end

end find_angle_between_vectors_l4_4284


namespace largest_prime_factor_of_S_l4_4266

-- Define p(n) as the product of non-zero digits of n
def p (n : ℕ) : ℕ :=
  n.digits 10 |> List.filter (≠ 0) |> List.prod

-- Define S as the sum of p(n) from 1 to 999
def S : ℕ :=
  (List.range 1000).filter (≠ 0) |>.map p |>.sum

-- State the proof problem
theorem largest_prime_factor_of_S : ∃ p : ℕ, Nat.Prime p ∧ p ∣ S ∧ ∀ q : ℕ, Nat.Prime q ∧ q ∣ S → q ≤ p := by sorry

end largest_prime_factor_of_S_l4_4266


namespace probability_larger_than_two_thirds_l4_4023

noncomputable def prob_larger_than_two_thirds : ℝ :=
  let I : set ℝ := set.Icc 0 2
  let prob_interval (a b : ℝ) : ℝ := (b - a) / (2 - 0)
  let prob_less_than_two_thirds := prob_interval 0 (2 / 3)
  let prob_both_less_than_two_thirds := prob_less_than_two_thirds ^ 2
  1 - prob_both_less_than_two_thirds

theorem probability_larger_than_two_thirds :
  prob_larger_than_two_thirds = 8 / 9 :=
by sorry

end probability_larger_than_two_thirds_l4_4023


namespace simplify_abs_expr_l4_4933

noncomputable def piecewise_y (x : ℝ) : ℝ :=
  if h1 : x < -3 then -3 * x
  else if h2 : -3 ≤ x ∧ x < 1 then 6 - x
  else if h3 : 1 ≤ x ∧ x < 2 then 4 + x
  else 3 * x

theorem simplify_abs_expr : 
  ∀ x : ℝ, (|x - 1| + |x - 2| + |x + 3|) = piecewise_y x :=
by
  intro x
  sorry

end simplify_abs_expr_l4_4933


namespace unit_disks_cover_parallelogram_l4_4900

axiom parallelogram (A B C D : Type) : Prop
axiom acute_triangle (A B D : Type) : Prop
axiom unit_length (AD : ℝ) : AD = 1
axiom side_length (AB : ℝ) : ℝ
axiom angle_alpha (α : ℝ) : ℝ
axiom unit_radius_disks (K_A K_B K_C K_D : Type) : Prop

theorem unit_disks_cover_parallelogram (A B C D : Type) 
  (ABCD : parallelogram A B C D)
  (AB_length : AB = side_length A B)
  (AD_length : AD = 1)
  (angle_DAB : angle_alpha α)
  (acute : acute_triangle A B D)
  (K_A_center : K_A = unit_radius_disks A B C D)
  (K_B_center : K_B = unit_radius_disks A B C D)
  (K_C_center : K_C = unit_radius_disks A B C D)
  (K_D_center : K_D = unit_radius_disks A B C D) :
  AB_length ≤ (cos α + sqrt 3 * sin α) := 
sorry

end unit_disks_cover_parallelogram_l4_4900


namespace largest_possible_sum_of_two_largest_angles_in_ABCD_l4_4058

noncomputable def largest_sum_of_angles (ABCD : Type) : ℝ :=
  let a := ℝ
  let d := ℝ
  
  -- Condition 1: Internal angles of $ABCD$ form an arithmetic progression
  let angle1 := a
  let angle2 := a + d
  let angle3 := a + 2 * d
  let angle4 := a + 3 * d

  -- Condition 2: The sum of the internal angles of any quadrilateral is 360 degrees
  have h_sum_angles : angle1 + angle2 + angle3 + angle4 = 360 := by sorry

  -- Condition 3: Triangles $ABD$ and $DCB$ are similar
  -- We denote those angles with respect to their similarity
  let α := ℝ
  let β := ℝ
  let γ := ℝ

  let angle_a_db_dcb := β
  let angle_a_ad_cbd := α
  let angle_a_ba_cdb := γ

  -- Condition 4: Angles in each of these triangles form an arithmetic progression
  let angle_tri1_1 := α
  let angle_tri1_2 := α + (180 - 3 * α) / 3
  let angle_tri1_3 := α + 2 * (180 - 3 * α) / 3

  let angle_tri2_1 := α
  let angle_tri2_2 := α + (180 - 3 * α) / 3
  let angle_tri2_3 := α + 2 * (180 - 3 * α) / 3

  -- Given all conditions, validate the largest possible sum of the two largest angles
  -- in $ABCD$ is 240 degrees
  240

theorem largest_possible_sum_of_two_largest_angles_in_ABCD :
  ∃ (ABCD : Type), largest_sum_of_angles ABCD = 240 := by
  -- Proof will be filled here.
  sorry

end largest_possible_sum_of_two_largest_angles_in_ABCD_l4_4058


namespace exists_root_in_interval_l4_4059

noncomputable def f (x : ℝ) : ℝ := x^3 - x - 3

theorem exists_root_in_interval : ∃ x ∈ set.Ioo (1 : ℝ) (2 : ℝ), f x = 0 := 
by {
  -- Use Intermediate Value Theorem and given function evaluations
  sorry
}

end exists_root_in_interval_l4_4059


namespace solution_set_is_interval_l4_4340

def f (x : ℝ) : ℝ := (3/4) * x^2 - 3 * x + 4

lemma min_value_of_f_at_2 : f 2 = 1 := by
  calc
    f 2 = (3/4) * (2 : ℝ)^2 - 3 * (2 : ℝ) + 4 : by rfl
       ... = 3 - 6 + 4 : by norm_num
       ... = 1 : by norm_num

theorem solution_set_is_interval {a b : ℝ} (h1 : a < b)
    (h2 : ∀ x, a ≤ f x ∧ f x ≤ b ↔ x ∈ set.Icc a b)
    : a + b = 4 :=
  sorry

end solution_set_is_interval_l4_4340


namespace average_multiples_of_10_l4_4623

theorem average_multiples_of_10 (a l n : ℕ) (h1 : a = 10) (h2 : l = 600)
  (h3 : ∀ x, a ≤ x ∧ x ≤ l → x % 10 = 0) : 
  (a + l) / 2 = 305 :=
by
  rw [h1, h2]
  norm_num [10 + 600, (10 + 600) / 2]
  sorry

end average_multiples_of_10_l4_4623


namespace evaluate_expression_l4_4727

theorem evaluate_expression (b : ℕ) (h : b = 5) : b^3 * b^4 * 2 = 156250 :=
by
  sorry

end evaluate_expression_l4_4727


namespace square_side_length_l4_4045

theorem square_side_length (d : ℝ) (h : d = 2 * Real.sqrt 2) : ∃ s : ℝ, s = 2 :=
by
  let s := d / Real.sqrt 2
  use s
  rw [h, ← mul_div_assoc, mul_comm (Real.sqrt 2), div_self (Real.sqrt_ne_zero 2)]
  norm_num
  sorry

end square_side_length_l4_4045


namespace factorize_x_squared_minus_25_l4_4221

theorem factorize_x_squared_minus_25 : ∀ (x : ℝ), (x^2 - 25) = (x + 5) * (x - 5) :=
by
  intros x
  sorry

end factorize_x_squared_minus_25_l4_4221


namespace age_difference_l4_4468

-- Defining the age variables as fractions
variables (x y : ℚ)

-- Given conditions
axiom ratio1 : 2 * x / y = 2 / y
axiom ratio2 : (5 * x + 20) / (y + 20) = 8 / 3

-- The main theorem to prove the difference between Mahesh's and Suresh's ages.
theorem age_difference : 5 * x - y = (125 / 8) := sorry

end age_difference_l4_4468


namespace joaozinho_multiplication_l4_4875

theorem joaozinho_multiplication :
  12345679 * 9 = 111111111 :=
by
  sorry

end joaozinho_multiplication_l4_4875


namespace integral_solution_l4_4735

noncomputable def integral_expression (x : ℝ) : ℝ :=
∫ (d : derivative x), 1 / ((x + 3)^(1/2) + (x + 3)^(2/3))

theorem integral_solution (x C : ℝ) :
  integral_expression x = 3 * (x + 3)^(1/3) - 6 * (x + 3)^(1/6) + 6 * log (abs ((x + 3)^(1/6) + 1)) + C := sorry

end integral_solution_l4_4735


namespace find_a_condition_l4_4224

theorem find_a_condition (a : ℚ) : (∀ n : ℕ, (a * n * (n + 2) * (n + 4)).denom = 1) ↔ ∃ k : ℤ, a = k / 3 := 
sorry

end find_a_condition_l4_4224


namespace oranges_in_each_box_l4_4697

theorem oranges_in_each_box (O B : ℕ) (h1 : O = 24) (h2 : B = 3) :
  O / B = 8 :=
by
  sorry

end oranges_in_each_box_l4_4697


namespace no_common_root_l4_4927

theorem no_common_root (a b c d : ℝ) (h0 : 0 < a) (h1 : a < b) (h2 : b < c) (h3 : c < d) :
  ¬ ∃ x : ℝ, x^2 + b * x + c = 0 ∧ x^2 + a * x + d = 0 :=
by
  sorry

end no_common_root_l4_4927


namespace largest_d_value_l4_4888

noncomputable def max_d (a b c d : ℝ) (h1 : a + b + c + d = 10) (h2 : ab + ac + ad + bc + bd + cd = 20) : ℝ :=
  if h : (4 * d ^ 2 - 20 * d - 80) ≤ 0 then d else 0

theorem largest_d_value (a b c d : ℝ) (h1 : a + b + c + d = 10) (h2 : ab + ac + ad + bc + bd + cd = 20) :
  max_d a b c d h1 h2 = (5 + 5 * real.sqrt 21) / 2 :=
sorry

end largest_d_value_l4_4888


namespace xy_zero_l4_4926

theorem xy_zero (x y : ℝ) (h1 : 2^x = 16^(y+1)) (h2 : 64^y = 4^(x-2)) : x * y = 0 :=
by
  sorry

end xy_zero_l4_4926


namespace ian_number_is_1021_l4_4725

-- Define the sequences each student skips
def alice_skips (n : ℕ) := ∃ k : ℕ, n = 4 * k
def barbara_skips (n : ℕ) := ∃ k : ℕ, n = 16 * (k + 1)
def candice_skips (n : ℕ) := ∃ k : ℕ, n = 64 * (k + 1)
-- Similar definitions for Debbie, Eliza, Fatima, Greg, and Helen

-- Define the condition under which Ian says a number
def ian_says (n : ℕ) :=
  ¬(alice_skips n) ∧ ¬(barbara_skips n) ∧ ¬(candice_skips n) -- and so on for Debbie, Eliza, Fatima, Greg, Helen

theorem ian_number_is_1021 : ian_says 1021 :=
by
  sorry

end ian_number_is_1021_l4_4725


namespace score_combinations_count_l4_4842

theorem score_combinations_count :
  let scores := {70, 85, 88, 90, 98, 100}
  ∃ (f : ℕ → ℤ) (n : ℕ), n ∈ {1, 2, 3, 4} → f n ∈ scores ∧ f 1 < f 2 ∧ f 2 ≤ f 3 ∧ f 3 < f 4 
  → ∃ combinations_count : ℕ, combinations_count = 35 :=
by sorry

end score_combinations_count_l4_4842


namespace quadrilateral_side_lengths_l4_4049

variable {A B C D M : Point}
variable {CD : Line}

-- Given conditions
variable h1 : dist_to_side A CD = dist_to_side B CD
variable h2 : dist A C + dist C B = dist A D + dist D B

-- The goal to prove
theorem quadrilateral_side_lengths (h1 : dist_to_side A CD = dist_to_side B CD) 
(h2 : dist A C + dist C B = dist A D + dist D B) : dist A D = dist B C ∧ dist A C = dist B D :=
sorry

end quadrilateral_side_lengths_l4_4049


namespace Karsyn_payment_l4_4184

def percentage : ℝ := 20
def initial_price : ℝ := 600

theorem Karsyn_payment : (percentage / 100) * initial_price = 120 :=
by 
  sorry

end Karsyn_payment_l4_4184


namespace sufficient_not_necessary_condition_l4_4145

theorem sufficient_not_necessary_condition (a : ℝ) : 
  a = 1 → ∀ x y : ℝ, 1 ≤ x → x ≤ y → f x ≤ f y :=
by
  let f := λ x, abs (x - a)
  intros ha hx hy hxy
  have hax1 : f x = x - 1, 
    { rw [ha, abs_of_nonneg (sub_nonneg_of_le hx)] }
  have hay1 : f y = y - 1,
    { rw [ha, abs_of_nonneg (sub_nonneg_of_le (le_trans hx hxy))] }
  rw [hax1, hay1]
  linarith

end sufficient_not_necessary_condition_l4_4145


namespace sum_fn_a_eq_sum_fn_1_minus_a_l4_4878

noncomputable def f : ℕ → (ℝ → ℝ)
| 0 := id
| (n + 1) := λ x, f n (x ^ 2 * (3 - 2 * x))

theorem sum_fn_a_eq_sum_fn_1_minus_a (a : ℝ) (h : 0 ≤ a ∧ a ≤ 1) :
  (∑ n in Finset.range 2018, f n a) + (∑ n in Finset.range 2018, f n (1 - a)) = 2018 :=
sorry

end sum_fn_a_eq_sum_fn_1_minus_a_l4_4878


namespace mixed_feed_total_pounds_l4_4090

theorem mixed_feed_total_pounds 
  (cheap_feed_cost : ℝ) (expensive_feed_cost : ℝ) (mix_cost : ℝ) 
  (cheap_feed_amount : ℕ) :
  cheap_feed_cost = 0.18 → 
  expensive_feed_cost = 0.53 → 
  mix_cost = 0.36 → 
  cheap_feed_amount = 17 → 
  (∃ (expensive_feed_amount : ℕ), 
    (cheap_feed_amount + expensive_feed_amount = 35)) :=
begin
  intros,
  use 18, -- We introduce 18 as the amount of more expensive feed
  sorry, -- Proof goes here
end

end mixed_feed_total_pounds_l4_4090


namespace sqrt_5_is_quadratic_radical_l4_4993

variable (a : ℝ) -- a is a real number

-- Definition to check if a given expression is a quadratic radical
def is_quadratic_radical (x : ℝ) : Prop := ∃ y : ℝ, y^2 = x

theorem sqrt_5_is_quadratic_radical : is_quadratic_radical 5 :=
by
  -- Here, 'by' indicates the start of the proof block,
  -- but the actual content of the proof is replaced with 'sorry' as instructed.
  sorry

end sqrt_5_is_quadratic_radical_l4_4993


namespace total_length_T_l4_4391

def T : set (ℝ × ℝ) := { p | ∃ x y : ℝ, p = (x, y) ∧ 
  (| (| x | - 3) - 2 | + | (| y | - 3) - 2 | = 2) }

theorem total_length_T : 
  (∃ l : ℝ, l = 128 * real.sqrt 2 ∧ ∀ (p ∈ T), ∃ s : list (set (ℝ × ℝ)), 
    (p ∈ ⋃₀ set_of (λ t, t ∈ s) ∧ 
    (∀ t ∈ s, is_diamond t) ∧ 
    (∀ t ∈ s, perimeter t = 8 * real.sqrt 2) ∧ 
    length_of_lines s = l)) :=
sorry

end total_length_T_l4_4391


namespace tom_search_cost_l4_4094

theorem tom_search_cost (n : ℕ) (h1 : n = 10) :
  let first_5_days_cost := 5 * 100 in
  let remaining_days_cost := (n - 5) * 60 in
  let total_cost := first_5_days_cost + remaining_days_cost in
  total_cost = 800 :=
by
  -- conditions
  have h2 : first_5_days_cost = 500 := by rfl
  have h3 : remaining_days_cost = 300 := by
    have rem_day_count : n - 5 = 5 := by
      rw [h1]
      rfl
    rfl
  have h4 : total_cost = 500 + 300 :=
    by rfl
  -- conclusion
  rw [h2, h3] at h4
  exact h4

end tom_search_cost_l4_4094


namespace tuesday_bought_toys_is_5_l4_4476

-- Defining conditions
def monday_dog_toys := 5
def tuesday_retained_dog_toys := 3
def wednesday_new_dog_toys := 5
def total_dog_toys_if_found := 13

-- Defining the number of dog toys bought on Tuesday
noncomputable def tuesday_bought_dog_toys : ℕ := sorry -- Placeholder to define T

-- The theorem to prove the number of dog toys bought on Tuesday
theorem tuesday_bought_toys_is_5 : tuesday_bought_dog_toys = 5 :=
by
  have eq1 : tuesday_retained_dog_toys + tuesday_bought_dog_toys + wednesday_new_dog_toys = total_dog_toys_if_found,
  { sorry }, -- This should be proved, but is marked as sorry
  sorry -- This will be the final part of the proof, using eq1 to conclude tuesday_bought_dog_toys = 5

end tuesday_bought_toys_is_5_l4_4476


namespace x_intercept_of_perpendicular_line_l4_4987

theorem x_intercept_of_perpendicular_line :
  ∃ x : ℝ, (0,y) ∈ line {a=3/2, b=-4} → x = 8/3 := 
sorry

end x_intercept_of_perpendicular_line_l4_4987


namespace foma_should_give_ierema_l4_4600

variables (F E Y : ℕ)

-- Given conditions in Lean definitions
def condition1 := F - 70 = E + 70
def condition2 := F - 40 = Y

-- Final statement to be proven
theorem foma_should_give_ierema : 
  condition1 → condition2 → F - E = 110 → F - E = 110 / 2 + 55 :=
by
  intros h1 h2 h3
  sorry

end foma_should_give_ierema_l4_4600


namespace area_of_rhombus_l4_4768

theorem area_of_rhombus (a : ℝ) (angle : ℝ) (h : angle = 60) (s : a = 20) :
  let height := a * Real.sin (angle * Real.pi / 180) in
  let area := a * height in
  area = 200 * Real.sqrt 3 :=
by
  sorry

end area_of_rhombus_l4_4768


namespace foma_gives_ierema_55_l4_4554

-- Defining the amounts that Foma, Ierema, and Yuliy have initially
variables (F E Y : ℝ)

-- Given conditions
def condition_1 : Prop := F - E = 140
def condition_2 : Prop := F - Y = 40
def condition_3 : Prop := Y = E + 70

-- Prove that Foma should give Ierema 55 gold coins
theorem foma_gives_ierema_55 (h1 : condition_1) (h2 : condition_2) (h3 : condition_3) : 
  (F - E) / 2 = 55 :=
by
  sorry

end foma_gives_ierema_55_l4_4554


namespace probability_vertices_sum_is_five_l4_4631

-- Define conditions: the square's vertices and the particle's starting point
def square_vertices : set (ℤ × ℤ) := {(2, 2), (-2, 2), (-2, -2), (2, -2)}

def initial_point : (ℤ × ℤ) := (0, 0)

-- Define the probability of moving to the 8 neighboring points
def move_prob : ℚ := 1 / 8

-- The set of boundary points of the square includes vertices
def boundary_points : set (ℤ × ℤ) :=
  square_vertices ∪ {(1, 2), (0, 2), (-1, 2), (1, -2), (0, -2), (-1, -2), 
                     (2, 1), (2, 0), (2, -1), (-2, 1), (-2, 0), (-2, -1)}

-- Statement: Prove that the sum of m and n, where m/n is the probability of landing on a vertex, is 5
theorem probability_vertices_sum_is_five :
  ∃ (m n : ℕ), (m.gcd n = 1) ∧ (4 = boundary_points.card) ∧ (boundary_points.card = 16) ∧ ((m : ℚ) / (n : ℚ) = 1 / 4) ∧ (m + n = 5) := sorry

end probability_vertices_sum_is_five_l4_4631


namespace problem1_problem2a_problem2b_l4_4147

-- Problem 1: Prove \(\sqrt{8} - \sqrt{6} < \sqrt{5} - \sqrt{3}\)
theorem problem1 : real.sqrt 8 - real.sqrt 6 < real.sqrt 5 - real.sqrt 3 :=
sorry

-- Problem 2(a): Prove \(\sin^2 15^\circ + \cos^2 15^\circ - \sin 15^\circ \cos 15^\circ = \frac{3}{4}\)
theorem problem2a : real.sin (15 * real.pi / 180)^2 + real.cos (15 * real.pi / 180)^2 - real.sin (15 * real.pi / 180) * real.cos (15 * real.pi / 180) = 3 / 4 :=
sorry

-- Problem 2(b): Generalized form
theorem problem2b (α : ℝ) : real.sin (α * real.pi / 180)^2 + real.cos ((30 - α) * real.pi / 180)^2 - real.sin (α * real.pi / 180) * real.cos ((30 - α) * real.pi / 180) = 3 / 4 :=
sorry

end problem1_problem2a_problem2b_l4_4147


namespace cube_cut_l4_4679

theorem cube_cut (
  (large_cube : Type) 
  (is_cube : large_cube → Prop) 
  (all_faces_painted_red : Prop) 
  (large_cube_cut : Prop) 
  (num_small_cubes_with_three_faces_colored : ℕ) 
  (num_corners_of_large_cube : ℕ) 
  (n : ℕ)
) : 
  ∀ large_cube, is_cube large_cube ∧ all_faces_painted_red ∧ large_cube_cut ∧ (num_small_cubes_with_three_faces_colored = 8) → 
  (num_corners_of_large_cube = 8) → 
  (n = 2) ∧ (n^3 = 8) :=
by
  intros
  sorry

end cube_cut_l4_4679


namespace penny_identified_species_of_sharks_l4_4216

theorem penny_identified_species_of_sharks (total_species : ℕ) (species_of_eels : ℕ) (species_of_whales : ℕ) :
  total_species = 55 →
  species_of_eels = 15 →
  species_of_whales = 5 →
  (total_species - (species_of_eels + species_of_whales)) = 35 :=
by
  intros
  rw [a_1, a_2, a_3]
  exact rfl

end penny_identified_species_of_sharks_l4_4216


namespace find_two_numbers_l4_4129

theorem find_two_numbers (A B : ℕ) (h1 : A ≠ B) (h2 : 32 - A = 23) (h3 : 32 - B = 13) (h4 : abs (A - B) ≠ 11 * n for some n ∈ ℕ) : (A = 9 ∧ B = 19) ∨ (A = 19 ∧ B = 9) :=
by
  sorry

end find_two_numbers_l4_4129


namespace light_time_at_12_23_PM_l4_4982

-- conditions
def length_initial : ℝ := sorry
def burn_out_time_candle1 : ℝ := 300
def burn_out_time_candle2 : ℝ := 360

-- functions representing remaining length of candles
def f (t : ℝ) : ℝ := length_initial * (300 - t) / 300
def g (t : ℝ) : ℝ := length_initial * (360 - t) / 360

-- main statement
theorem light_time_at_12_23_PM : ∃ t : ℝ, t = 277 ∧ (g(277) = 3 * f(277)) ∧ t = 277 ∧ (some_function_to_convert 277) = "12:23 PM" :=
by sorry

end light_time_at_12_23_PM_l4_4982


namespace intersecting_lines_l4_4060

theorem intersecting_lines (c d : ℝ) 
  (h1 : 3 = (1 / 3) * 3 + c) 
  (h2 : 3 = (1 / 3) * 3 + d) : c + d = 4 :=
begin
  sorry
end

end intersecting_lines_l4_4060


namespace books_on_shelves_l4_4977

-- Definitions based on the problem conditions.
def bookshelves : ℕ := 1250
def books_per_shelf : ℕ := 45
def total_books : ℕ := 56250

-- Theorem statement
theorem books_on_shelves : bookshelves * books_per_shelf = total_books := 
by
  sorry

end books_on_shelves_l4_4977


namespace triangle_relation_l4_4452

-- Define the given conditions and the hypothesis
variables {A B C M N K : Type*}
variables (triangle : A → B → C → Type*)
variables [midpoint M A C] [on_segment N A M] [extension K N B]
variables [right_angle (angle BMK)] [angle_eq (angle MBN) (angle CBM)]

-- Define segments and lengths
variables [segment BC] [segment AK] [segment BK]

-- State the theorem
theorem triangle_relation 
  (h1: midpoint M A C) 
  (h2: on_segment N A M) 
  (h3: angle_eq (angle MBN) (angle CBM))
  (h4: right_angle (angle BMK)) : 
  length BC = length AK + length BK := by
  sorry

end triangle_relation_l4_4452


namespace number_is_correct_l4_4331

theorem number_is_correct (x : ℝ) (h : (30 / 100) * x = (25 / 100) * 40) : x = 100 / 3 :=
by
  -- Proof would go here.
  sorry

end number_is_correct_l4_4331


namespace smallest_k_l4_4239

theorem smallest_k (n k : ℕ) (h1: 2000 < n) (h2: n < 3000)
  (h3: ∀ i, 2 ≤ i → i ≤ k → n % i = i - 1) :
  k = 9 :=
by
  sorry

end smallest_k_l4_4239


namespace original_student_count_l4_4899

variable (A B C N D : ℕ)
variable (hA : A = 40)
variable (hB : B = 32)
variable (hC : C = 36)
variable (hD : D = N * A)
variable (hNewSum : D + 8 * B = (N + 8) * C)

theorem original_student_count (hA : A = 40) (hB : B = 32) (hC : C = 36) (hD : D = N * A) (hNewSum : D + 8 * B = (N + 8) * C) : 
  N = 8 :=
by
  sorry

end original_student_count_l4_4899


namespace equalize_foma_ierema_l4_4574

theorem equalize_foma_ierema (F E Y : ℕ) 
  (h1 : F - 70 = E + 70) 
  (h2 : F - 40 = Y) 
  (h3 : Y = E + 70) 
  : ∃ x : ℕ, x = 55 ∧ F - x = E + x :=
by
  use 55
  sorry

end equalize_foma_ierema_l4_4574


namespace max_projection_area_tetrahedron_l4_4981

-- Definitions based on conditions
def side_length : ℝ := 3
def dihedral_angle : ℝ := 30 * Real.pi / 180 -- converting degrees to radians

-- Prove the maximum projection area
theorem max_projection_area_tetrahedron (S : ℝ)
  (hS : S = (Real.sqrt 3) / 4 * side_length ^ 2) :
  ∃ max_area, max_area = S := by
  exists (3 * Real.sqrt 3 / 4)
  sorry

end max_projection_area_tetrahedron_l4_4981


namespace sum_floor_eq_1994_l4_4902

theorem sum_floor_eq_1994 (n : ℕ) (S : ℝ) 
  (hS : S = (Finset.range (n + 1)).sum (λ k, 1 / Real.sqrt (k + 1))) :
  n = 997506 → ⌊S⌋ = 1994 :=
by
  intros h
  rw [h]
  sorry

end sum_floor_eq_1994_l4_4902


namespace remainder_of_groups_mod_100_l4_4163

def tenors := 7
def basses := 9

def valid_conditions (t b : Nat) : Prop :=
  (t + b > 0) ∧
  ((t - b) % 3 = 0) ∧
  ((t + b) % 2 = 0)

noncomputable def number_of_groups : Nat :=
  (∑ t in Finset.range (tenors + 1), ∑ b in Finset.range (basses + 1),
    if valid_conditions t b then Mathlib.combinatorics.choose tenors t * Mathlib.combinatorics.choose basses b else 0)

theorem remainder_of_groups_mod_100 : number_of_groups % 100 = 56 := 
by 
  sorry

end remainder_of_groups_mod_100_l4_4163


namespace count_four_digit_numbers_l4_4273

theorem count_four_digit_numbers :
  let cards := [2, 0, 0, 9] in
  let alternative := 6 in
  (∃ n : ℕ, n = 12 ∧ (
    ∃ positions, positions ⊆ finset.range 3 ∧
    finset.card positions = 2 ∧
    ∃ number_choice, number_choice ∈ {9, alternative} ∧
    let remaining := finset.erase (finset.erase (finset.univ) positions) number_choice in
    finset.card remaining = 2 ∧
    ∃ arrangements, finset.prod arrangements (λ _, 1) = 2!
  )) := sorry

end count_four_digit_numbers_l4_4273


namespace minimum_m_exists_l4_4371

theorem minimum_m_exists (n : ℕ) (h : n ≥ 2) : ∃ m, (∀ (x : Fin n → Fin n → ℝ),
     (∀ i j : Fin n, x i j = (Finset.range (j + 1)).sup (x i) ∨ x i j = (Finset.range (i + 1)).sup (λ k, x k j)) ∧
     (∀ i : Fin n, (Finset.univ.filter (λ k, x i k = (Finset.range (k + 1)).sup (x i))).card ≤ m) ∧
     (∀ j : Fin n, (Finset.univ.filter (λ k, x k j = (Finset.range (k + 1)).sup (λ i, x i j))).card ≤ m))
  → m = 1 + Int.ceil (n/2 : ℝ) :=
sorry

end minimum_m_exists_l4_4371


namespace foma_should_give_ierema_l4_4595

variables (F E Y : ℕ)

-- Given conditions in Lean definitions
def condition1 := F - 70 = E + 70
def condition2 := F - 40 = Y

-- Final statement to be proven
theorem foma_should_give_ierema : 
  condition1 → condition2 → F - E = 110 → F - E = 110 / 2 + 55 :=
by
  intros h1 h2 h3
  sorry

end foma_should_give_ierema_l4_4595


namespace largest_possible_degree_l4_4055

-- Define the rational function and the degree conditions
def rational_function (p : polynomial ℤ) : fraction_ring (polynomial ℤ) :=
  p / (3 * (polynomial.X ^ 7) - (polynomial.X ^ 3) + 5)

theorem largest_possible_degree (p : polynomial ℤ) (h : degree p ≤ 7) :
  ∃ L, tendsto (λ x : ℝ, (eval x p) / (3 * x^7 - x^3 + 5)) at_top (nhds L) :=
sorry

end largest_possible_degree_l4_4055


namespace problem_proof_l4_4470

noncomputable def 𝕌 := set.univ
noncomputable def M := {x : ℝ | (x + 3)^2 ≤ 0}
noncomputable def N := {x : ℝ | x^2 + x - 6 = 0}
noncomputable def complement_M := {x : ℝ | x ≠ -3}
noncomputable def A := complement_M ∩ N
noncomputable def B (a : ℝ) := {x : ℝ | a - 1 ≤ x ∧ x ≤ 5 - a}
noncomputable def R := {a : ℝ | a ≥ 3}

theorem problem_proof : 
  ((complement_M ∩ N) = {2}) ∧ 
  (∀ a : ℝ, (A ∪ B a = A) → a ∈ R) := 
by sorry

end problem_proof_l4_4470


namespace triangle_proof_l4_4429

-- Definition of a point in the plane.
structure Point : Type :=
(x : ℝ)
(y : ℝ)

-- Definitions for the points involved.
variables (A B C M N K : Point)

-- Midpoint condition
def is_midpoint (M A C : Point) : Prop :=
  (M.x = (A.x + C.x) / 2) ∧ (M.y = (A.y + C.y) / 2)

-- Angular condition
noncomputable def angle (P Q R : Point) : ℝ := sorry

-- Given Conditions as Lean Definitions
def problem_conditions : Prop :=
  is_midpoint M A C ∧
  N.x = (A.x + M.x) / 2 ∧ N.y = (A.y + M.y) / 2 ∧ -- N lies on AM such that
  angle M B N = angle C B M ∧
  angle B M K = π / 2

-- Final proof statement to be proved
theorem triangle_proof (h : problem_conditions A B C M N K) : (dist B C) = (dist A K) + (dist B K) :=
sorry

end triangle_proof_l4_4429


namespace symmetry_center_f_l4_4804

noncomputable def f (x : ℝ) : ℝ := sin x ^ 4 + cos x ^ 2 + 1 / 4 * sin (2 * x) * cos (2 * x)

theorem symmetry_center_f :
  ∃ x0 y0, x0 = -π / 16 ∧ y0 = 7 / 8 ∧ ∀ x : ℝ, f (x0 - x) = f (x0 + x) + 2 * y0 := 
sorry

end symmetry_center_f_l4_4804


namespace custom_op_theorem_l4_4211

def custom_op (x y : ℕ) : ℕ := x * y - 3 * x + y

theorem custom_op_theorem : (custom_op 6 5) - (custom_op 5 6) = -4 := by
  sorry

end custom_op_theorem_l4_4211


namespace eldest_sister_age_l4_4602

/-- Given three sisters with different ages whose average age is 10,
    the average age of one pair is 11, and the average age of another pair is 12,
    prove that the age of the eldest sister is 16. -/
theorem eldest_sister_age (a b c : ℕ) (h_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  (a + b + c) / 3 = 10 →
  (a + b) / 2 = 11 ∨ (a + c) / 2 = 11 ∨ (b + c) / 2 = 11 →
  (a + b) / 2 = 12 ∨ (a + c) / 2 = 12 ∨ (b + c) / 2 = 12 →
  max (max a b) c = 16 :=
begin
  sorry
end

end eldest_sister_age_l4_4602


namespace sequence_convergence_l4_4533

noncomputable def sequence (x0 : ℝ) : ℕ → ℝ
| 0       := x0
| (n + 1) := sqrt (sequence n + 1)

theorem sequence_convergence (x0 : ℝ) (h0 : x0 > 0) :
  ∃ (A C : ℝ), A = (1 + Real.sqrt 5) / 2 ∧ C = |x0 - A| ∧ ∀ n : ℕ, |sequence x0 n - A| < C / A^n :=
sorry

end sequence_convergence_l4_4533


namespace sum_integer_solutions_abs_lt_l4_4115

noncomputable def abs_lt (a b : ℤ) : Prop :=
  abs a < abs b

theorem sum_integer_solutions_abs_lt (n : ℤ) :
  (abs n < abs (n - 3) ∧ abs (n - 3) < 9) →
  n ∈ [-5, -4, -3, -2, -1, 0, 1] →
  ∑ x in [-5, -4, -3, -2, -1, 0, 1], x = -14 :=
by
  sorry

end sum_integer_solutions_abs_lt_l4_4115


namespace part_a_part_b_l4_4137

-- Part (a)
theorem part_a (x y z : ℤ) : (x^2 + y^2 + z^2 = 2 * x * y * z) → (x = 0 ∧ y = 0 ∧ z = 0) :=
by
  sorry

-- Part (b)
theorem part_b : ∃ (x y z v : ℤ), (x^2 + y^2 + z^2 + v^2 = 2 * x * y * z * v) → (x = 0 ∧ y = 0 ∧ z = 0 ∧ v = 0) :=
by
  sorry

end part_a_part_b_l4_4137


namespace fomagive_55_l4_4593

variables (F E Y : ℕ)

-- Conditions
def condition1 := F - E = 140
def condition2 := F - Y = 40
def condition3 := Y = E + 70

-- Proposition: Foma should give 55 coins to Ierema
theorem fomagive_55 (h1 : condition1 F E) (h2 : condition2 F Y) (h3 : condition3 E Y) : ∃ G, G = 55 :=
by
  have := h1,
  have := h2,
  have := h3,
  sorry

end fomagive_55_l4_4593


namespace triangle_angle_sum_l4_4420

variable (ABC : Type) [triangle ABC]

-- Definitions from conditions in a)
variable (A B C M N K : point ABC)
variable (M_midpoint : midpoint M A C)
variable (N_on_AM : lies_on N A M)
variable (angle_eq1 : ∠ M B N = ∠ C B M)
variable (K_ext_BN : lies_on_ext K B N)
variable (angle_eq2 : ∠ B M K = 90)

-- Goal to prove
theorem triangle_angle_sum (HC : M_midpoint) (HN : N_on_AM) (HB : angle_eq1) (HK : K_ext_BN) (HA : angle_eq2) :
  segment_length B C = segment_length A K + segment_length B K := sorry

end triangle_angle_sum_l4_4420


namespace find_number_l4_4161

theorem find_number (number : ℚ) 
  (H1 : 8 * 60 = 480)
  (H2 : number / 6 = 16 / 480) :
  number = 1 / 5 := 
by
  sorry

end find_number_l4_4161


namespace count_even_three_digit_numbers_l4_4816

def digits : Set ℕ := {0, 1, 2, 3, 4}

def is_three_digit_number(x : ℕ) : Prop :=
  100 ≤ x ∧ x < 1000

def is_even(x : ℕ) : Prop := 
  x % 2 = 0

def no_digit_repeated(x : ℕ) : Prop :=
  let digit_list := x.digits in 
  digit_list.length = digit_list.to_set.size

theorem count_even_three_digit_numbers :
  (finset.filter (λ x : ℕ, 
    is_three_digit_number x ∧ 
    is_even x ∧ 
    no_digit_repeated x
  ) (finset.of_list $ (list.range 1000))).card = 30 :=
sorry

end count_even_three_digit_numbers_l4_4816


namespace solve_system_of_equations_l4_4029

theorem solve_system_of_equations :
  ∀ (x y : ℝ),
    (3 * x^2 + 3 * y^2 - x^2 * y^2 = 3) ∧ (x^4 + y^4 - x^2 * y^2 = 31) ↔ 
      ((x =  sqrt 5 ∧ y =  sqrt 6) ∨ (x = -sqrt 5 ∧ y =  sqrt 6) ∨
       (x =  sqrt 5 ∧ y = -sqrt 6) ∨ (x = -sqrt 5 ∧ y = -sqrt 6) ∨
       (x =  sqrt 6 ∧ y =  sqrt 5) ∨ (x = -sqrt 6 ∧ y =  sqrt 5) ∨
       (x =  sqrt 6 ∧ y = -sqrt 5) ∨ (x = -sqrt 6 ∧ y = -sqrt 5)) :=
by {
  sorry
}

end solve_system_of_equations_l4_4029


namespace foma_should_give_ierema_55_coins_l4_4565

theorem foma_should_give_ierema_55_coins (F E Y : ℤ)
  (h1 : F - 70 = E + 70)
  (h2 : F - 40 = Y)
  (h3 : E + 70 = Y) :
  F - E = 110 → F - E = 55 :=
by
  sorry

end foma_should_give_ierema_55_coins_l4_4565


namespace triangle_equality_lemma_l4_4408

open EuclideanGeometry

theorem triangle_equality_lemma 
  (A B C M N K: Point)
  (hM_midpoint: midpoint M A C)
  (hN_on_AM: lies_on N A M)
  (h_angle_eq: ∠ M B N = ∠ C B M)
  (hK_right: ∠ B M K = 90) :
  dist B C = dist A K + dist B K :=
by
  sorry

end triangle_equality_lemma_l4_4408


namespace set_eq_implies_sum_eq_l4_4880

theorem set_eq_implies_sum_eq (a b : ℕ) (h : {4, a} = {2, a * b}) : a + b = 4 :=
by
  sorry

end set_eq_implies_sum_eq_l4_4880


namespace divide_connected_cities_l4_4035

theorem divide_connected_cities :
  ∀ (G : Type) (V : Finset G) (E : G → G → Prop) [DecidableRel E],
  (card V = 100) →
  (∀ v ∈ V, ∃ u ∈ V, u ≠ v) →
  (∀ v ∈ V, IsConnected (V.erase v) E) →
  ∃ (A B : Finset G),
  (card A = 50) ∧ (card B = 50) ∧
  (∀ a₁ a₂ ∈ A, E a₁ a₂) ∧
  (∀ b₁ b₂ ∈ B, E b₁ b₂) :=
by
  sorry

end divide_connected_cities_l4_4035


namespace obtain_one_fifth_from_zero_and_one_obtain_all_rationals_between_zero_and_one_l4_4841

theorem obtain_one_fifth_from_zero_and_one : 
  ∃ (S : Set ℚ), 
    {0, 1} ⊆ S ∧ 
    (∀ a b ∈ S, (a + b) / 2 ∉ S → (S ∪ { (a + b) / 2 }) = S) ∧ 
    (∃ q : ℚ, q = 1/5 ∧ q ∈ S) := 
sorry

theorem obtain_all_rationals_between_zero_and_one : 
  ∀ (q : ℚ), 
    0 < q ∧ q < 1 → 
    ∃ (S : Set ℚ), 
      {0, 1} ⊆ S ∧ 
      (∀ a b ∈ S, (a + b) / 2 ∉ S → (S ∪ { (a + b) / 2 }) = S) ∧ 
      q ∈ S := 
sorry

end obtain_one_fifth_from_zero_and_one_obtain_all_rationals_between_zero_and_one_l4_4841


namespace count_irreducible_fractions_l4_4347

theorem count_irreducible_fractions : 
  let nums := [226, 227, 229, 232, 233, 236, 238, 239]
  in (∀ n ∈ nums, by apply (1 / 16 < n / 15 ∧ n / 15 < 1 / 15 ∧ Nat.gcd n 15 = 1)) 
  ∧ nums.length = 8 := 
sorry

end count_irreducible_fractions_l4_4347


namespace factorization_of_w4_minus_81_l4_4732

theorem factorization_of_w4_minus_81 (w : ℝ) : 
  (w^4 - 81) = (w - 3) * (w + 3) * (w^2 + 9) :=
by sorry

end factorization_of_w4_minus_81_l4_4732


namespace inverse_variation_y_squared_sqrt_z_l4_4038

theorem inverse_variation_y_squared_sqrt_z (k : ℝ) :
  (∀ y z : ℝ, y^2 * sqrt z = k) →
  (∃ y z : ℝ, y = 3 ∧ z = 4 ∧ y^2 * sqrt z = k) →
  (∃ z : ℝ, (6 : ℝ)^2 * sqrt z = k ∧ z = 1/4) :=
by
  intros h₁ h₂
  sorry

end inverse_variation_y_squared_sqrt_z_l4_4038


namespace common_chord_passes_through_P_l4_4945

theorem common_chord_passes_through_P (k1 k2 : Circle) (P A B C D E : Point) (e f : Line)
  (condition1 : k1 ∩ k2 = {P})
  (condition2 : tangent e k1 A ∧ tangent e k2 B)
  (condition3 : parallel e f ∧ tangent f k1 C ∧ intersects f k2 = {D, E}) :
  chord (circumcircle (triangle A B C)) (circumcircle (triangle A D E)) ∋ P :=
sorry

end common_chord_passes_through_P_l4_4945


namespace remainder_when_squared_l4_4128

theorem remainder_when_squared (n : ℕ) (h : n % 8 = 6) : (n * n) % 32 = 4 := by
  sorry

end remainder_when_squared_l4_4128


namespace triangle_relation_l4_4455

-- Define the given conditions and the hypothesis
variables {A B C M N K : Type*}
variables (triangle : A → B → C → Type*)
variables [midpoint M A C] [on_segment N A M] [extension K N B]
variables [right_angle (angle BMK)] [angle_eq (angle MBN) (angle CBM)]

-- Define segments and lengths
variables [segment BC] [segment AK] [segment BK]

-- State the theorem
theorem triangle_relation 
  (h1: midpoint M A C) 
  (h2: on_segment N A M) 
  (h3: angle_eq (angle MBN) (angle CBM))
  (h4: right_angle (angle BMK)) : 
  length BC = length AK + length BK := by
  sorry

end triangle_relation_l4_4455


namespace circle_tangent_radius_l4_4085

-- Define the radii of the three given circles
def radius1 : ℝ := 1.0
def radius2 : ℝ := 2.0
def radius3 : ℝ := 3.0

-- Define the problem statement: finding the radius of the fourth circle externally tangent to the given three circles
theorem circle_tangent_radius (r1 r2 r3 : ℝ) (cond1 : r1 = 1) (cond2 : r2 = 2) (cond3 : r3 = 3) : 
  ∃ R : ℝ, R = 6 := by
  sorry

end circle_tangent_radius_l4_4085


namespace product_is_49_or_not_l4_4131

theorem product_is_49_or_not :
  (7 * 7 = 49) ∧
  ((-7) * (-7) = 49) ∧
  ((1 / 2) * 98 = 49) ∧
  (1 * 49 = 49) ∧
  ((3 / 2) * 35 ≠ 49) :=
by {
  split; -- Handling each condition separately
  -- solving condition for prodA
  norm_num,
  -- solving condition for prodB
  norm_num,
  -- solving condition for prodC
  norm_num,
  -- solving condition for prodD
  norm_num,
  -- solving condition for prodE
  norm_num,
  linarith, -- we use linarith to handle the last not equal condition
}

end product_is_49_or_not_l4_4131


namespace foma_should_give_ierema_55_coins_l4_4563

theorem foma_should_give_ierema_55_coins (F E Y : ℤ)
  (h1 : F - 70 = E + 70)
  (h2 : F - 40 = Y)
  (h3 : E + 70 = Y) :
  F - E = 110 → F - E = 55 :=
by
  sorry

end foma_should_give_ierema_55_coins_l4_4563


namespace equalize_foma_ierema_l4_4582

variables 
  (F E Y : ℕ)
  (h1 : F - 70 = E + 70)
  (h2 : F - 40 = Y)

def foma_should_give_ierema : ℕ := (F - E) / 2

theorem equalize_foma_ierema (F E Y : ℕ) (h1 : F - 70 = E + 70) (h2 : F - 40 = Y) :
  foma_should_give_ierema F E = 55 := 
by
  sorry

end equalize_foma_ierema_l4_4582


namespace max_value_right_triangle_ratio_l4_4663

theorem max_value_right_triangle_ratio 
  (k l a b c : ℝ) (hk : 0 < k) (hl : 0 < l)
  (h_pythag : k^2 * a^2 + l^2 * b^2 = c^2) :
  (ka + lb) / c ≤ sqrt 2 :=
sorry

end max_value_right_triangle_ratio_l4_4663


namespace solve_discriminant_l4_4702

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem solve_discriminant : 
  discriminant 2 (2 + 1/2) (1/2) = 2.25 :=
by
  -- The proof can be filled in here
  -- Assuming a = 2, b = 2.5, c = 1/2
  -- discriminant 2 2.5 0.5 will be computed
  sorry

end solve_discriminant_l4_4702


namespace intersection_forms_parallelogram_l4_4480

-- Definition of point and parallelogram
structure Point (α : Type) := (x y : α)

structure Parallelogram (α : Type) :=
(A B C D : Point α)
(center : Point α)
(is_center_bisected : true) -- Placeholder property to indicate the center

def divides_in_ratio (α : Type) [LinearOrder α] (p q : Point α) (r : Point α) (k : α) :=
(r.x = p.x + k * (q.x - p.x) / (1 + k) ∧ r.y = p.y + k * (q.y - p.y) / (1 + k))

theorem intersection_forms_parallelogram
  {α : Type} [LinearOrder α] [Field α]
  (P : Parallelogram α)
  (M N K L : Point α)
  (k : α)
  (M_on_AB : divides_in_ratio α P.A P.B M k)
  (N_on_BC : divides_in_ratio α P.B P.C N k)
  (K_on_CD : divides_in_ratio α P.C P.D K k)
  (L_on_DA : divides_in_ratio α P.D P.A L k) :
  ∃ P', Parallelogram α ∧ P'.center = P.center := 
sorry

end intersection_forms_parallelogram_l4_4480


namespace triangle_ABC_proof_l4_4440

variables {A B C M N K : Type} [inhabited A] [inhabited B] [inhabited C]
variables [between A B C] [within_segment A M] [within_segment M C] [on_segment A M]
variables [angle_eq (M B N) (C B M)] [right_angle (B M K)]
variables [midpoint_of_segment A C M]
variables [extension_of_segment B N K beyond N]

theorem triangle_ABC_proof (hM : midpoint_of_segment A C M)
    (hN : on_segment A M N)
    (hAngle : angle_eq (M B N) (C B M))
    (hPerp : right_angle (B M K))
    (hExt : extension_of_segment B N K beyond N) :
    BC = AK + BK :=
by
  sorry

end triangle_ABC_proof_l4_4440


namespace student_probability_at_least_9_correct_l4_4671

-- Define the conditions
def total_questions : ℕ := 10
def probability_of_success : ℚ := 1 / 4

-- Define the binomial probability calculation
noncomputable def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

-- Calculate the probability of getting exactly 9 correct answers
noncomputable def probability_9_correct : ℚ :=
  binomial_probability total_questions 9 probability_of_success

-- Calculate the probability of getting exactly 10 correct answers
noncomputable def probability_10_correct : ℚ :=
  binomial_probability total_questions 10 probability_of_success

-- The combined probability of answering at least 9 questions correctly
noncomputable def total_probability : ℚ :=
  probability_9_correct + probability_10_correct

-- Statement to be proved
theorem student_probability_at_least_9_correct :
  (total_probability ≈ 3 * 10^(-5) : Prop) :=
sorry

end student_probability_at_least_9_correct_l4_4671


namespace dividend_divisor_quotient_l4_4950

theorem dividend_divisor_quotient (x y z : ℕ) 
  (h1 : x = 6 * y) 
  (h2 : y = 6 * z) 
  (h3 : x = y * z) : 
  x = 216 ∧ y = 36 ∧ z = 6 := 
by
  sorry

end dividend_divisor_quotient_l4_4950


namespace average_stoppage_time_is_10_l4_4729

-- Define the speeds of trains excluding and including stoppages
def speed_excluding_1 : ℝ := 48
def speed_including_1 : ℝ := 40
def speed_excluding_2 : ℝ := 54
def speed_including_2 : ℝ := 45
def speed_excluding_3 : ℝ := 60
def speed_including_3 : ℝ := 50

-- Calculate the stoppage times for each train in minutes
def stoppage_time_1 : ℝ := (speed_excluding_1 - speed_including_1) / speed_excluding_1 * 60
def stoppage_time_2 : ℝ := (speed_excluding_2 - speed_including_2) / speed_excluding_2 * 60
def stoppage_time_3 : ℝ := (speed_excluding_3 - speed_including_3) / speed_excluding_3 * 60

-- Define the target average stoppage time
def average_stoppage_time : ℝ := (stoppage_time_1 + stoppage_time_2 + stoppage_time_3) / 3

-- State the theorem to prove
theorem average_stoppage_time_is_10 : average_stoppage_time = 10 := by
  sorry

end average_stoppage_time_is_10_l4_4729


namespace sequence_a_n_sum_b_n_l4_4636

noncomputable def f (x : ℝ) : ℝ := sin x * sin (x + 2 * π) * sin (x + 3 * π)

def a_n (n : ℕ+) : ℝ := π * (n - 1 / 2)
def b_n (n : ℕ+) : ℝ := 2 * n * a_n n
def T_n (n : ℕ+) : ℝ := ∑ i in finset.range n, b_n (i + 1)

theorem sequence_a_n (n : ℕ+) : ∀ n, a_n n = π * (n - 1 / 2) := by
  intro n
  sorry

theorem sum_b_n (n : ℕ+) : T_n n = π * ((2 * n - 3) * 2^n + 3) := by
  intro n
  sorry

end sequence_a_n_sum_b_n_l4_4636


namespace find_f2_f5_l4_4755

theorem find_f2_f5 (f : ℝ → ℝ) (h : ∀ x : ℝ, f (3 ^ x) = x * log 9) : f 2 + f 5 = 2 :=
sorry

end find_f2_f5_l4_4755


namespace triangle_angle_sum_l4_4425

variable (ABC : Type) [triangle ABC]

-- Definitions from conditions in a)
variable (A B C M N K : point ABC)
variable (M_midpoint : midpoint M A C)
variable (N_on_AM : lies_on N A M)
variable (angle_eq1 : ∠ M B N = ∠ C B M)
variable (K_ext_BN : lies_on_ext K B N)
variable (angle_eq2 : ∠ B M K = 90)

-- Goal to prove
theorem triangle_angle_sum (HC : M_midpoint) (HN : N_on_AM) (HB : angle_eq1) (HK : K_ext_BN) (HA : angle_eq2) :
  segment_length B C = segment_length A K + segment_length B K := sorry

end triangle_angle_sum_l4_4425


namespace negation_example_l4_4965

theorem negation_example :
  ¬ (∀ x : ℝ, x^2 - x + 1 ≥ 0) ↔ ∃ x : ℝ, x^2 - x + 1 < 0 :=
sorry

end negation_example_l4_4965


namespace beautiful_dates_in_2023_l4_4689

/-- A date is defined as beautiful if all digits in DD.MM.YY are unique -/
def is_beautiful_date (d m y : Nat) : Prop :=
  let digits := [d / 10, d % 10, m / 10, m % 10, y / 100 % 10, y / 10 % 10, y % 10]
  digit_pairs_no_repetition : Prop := (∀ i, ∀ j, i ≠ j → digits[i] ≠ digits[j])

/-- Given the year 2023, we calculate the number of beautiful dates in that year -/
def number_of_beautiful_dates_in_2023 : Nat :=
  let valid_months := [4, 5, 6, 7, 8, 9]
  let valid_days := [14, 15, 16, 17, 18, 19]
  valid_months.length * valid_days.length
  
theorem beautiful_dates_in_2023 (y : Nat):
  y = 2023 → number_of_beautiful_dates_in_2023 = 30 := by
  intros H
  rw [number_of_beautiful_dates_in_2023]
  sorry

end beautiful_dates_in_2023_l4_4689


namespace bella_travel_time_l4_4677

theorem bella_travel_time :
  let alice_speed := 110 * 50 in
  let alice_time := 20 in
  let alice_distance := alice_speed * alice_time in
  let bella_speed := 120 * 47 in
  alice_distance / bella_speed = 20 :=
by
  sorry

end bella_travel_time_l4_4677


namespace Susan_initial_amount_l4_4939

def initial_amount (S : ℝ) : Prop :=
  let Spent_in_September := (1/6) * S
  let Spent_in_October := (1/8) * S
  let Spent_in_November := 0.3 * S
  let Spent_in_December := 100
  let Remaining := 480
  S - (Spent_in_September + Spent_in_October + Spent_in_November + Spent_in_December) = Remaining

theorem Susan_initial_amount : ∃ S : ℝ, initial_amount S ∧ S = 1420 :=
by
  sorry

end Susan_initial_amount_l4_4939


namespace segments_form_quadrilateral_l4_4175

theorem segments_form_quadrilateral (a d : ℝ) (h_pos : a > 0 ∧ d > 0) (h_sum : 4 * a + 6 * d = 3) : 
  (∃ s1 s2 s3 s4 : ℝ, s1 + s2 + s3 > s4 ∧ s1 + s2 + s4 > s3 ∧ s1 + s3 + s4 > s2 ∧ s2 + s3 + s4 > s1) :=
sorry

end segments_form_quadrilateral_l4_4175


namespace travel_probability_l4_4723

theorem travel_probability (P_A P_B P_C : ℝ) (hA : P_A = 1/3) (hB : P_B = 1/4) (hC : P_C = 1/5) :
  let P_none_travel := (1 - P_A) * (1 - P_B) * (1 - P_C)
  ∃ (P_at_least_one : ℝ), P_at_least_one = 1 - P_none_travel ∧ P_at_least_one = 3/5 :=
by {
  sorry
}

end travel_probability_l4_4723


namespace foma_should_give_ierema_55_coins_l4_4567

variables (F E Y : ℝ)

-- Conditions
def condition1 : Prop := F - 70 = E + 70
def condition2 : Prop := F - 40 = Y
def condition3 : Prop := E + 70 = Y

theorem foma_should_give_ierema_55_coins
  (h1 : condition1 F E) (h2 : condition2 F Y) (h3 : condition3 E Y) :
  F - (E + 55) = E + 55 :=
by
  sorry

end foma_should_give_ierema_55_coins_l4_4567


namespace general_term_a_n_sum_series_l4_4787

noncomputable def a_n (n : ℕ) : ℕ :=
  if n = 0 then 0 else 2 * 3^(n-1)

def S_n (n : ℕ) : ℤ :=
  (3 / 2 : ℚ) * a_n n - 1

def b_n (n : ℕ) : ℚ :=
  2 * log 3 (a_n n / 2) + 1

theorem general_term_a_n (n : ℕ) (hn : n ≠ 0) : a_n n = 2 * 3^(n-1) :=
by sorry

theorem sum_series (n : ℕ) (hn : n > 0) :
  (finset.sum (finset.range (n-1)) (λ k, 1 / (b_n k * b_n (k+1)))) = (n-1) / (2 * n - 1) :=
by sorry

end general_term_a_n_sum_series_l4_4787


namespace compare_abc_l4_4780

noncomputable def a : ℝ := (0.6)^(2/5)
noncomputable def b : ℝ := (0.4)^(2/5)
noncomputable def c : ℝ := (0.4)^(3/5)

theorem compare_abc : a > b ∧ b > c := 
by
  sorry

end compare_abc_l4_4780


namespace steve_speed_back_from_work_l4_4139

-- Definitions relevant to the problem
def distance_to_work : ℝ := 40
def total_time : ℝ := 6

-- We define the speeds for the way to and back from work
def speed_to_work (v : ℝ) : ℝ := v
def speed_back_from_work (v : ℝ) : ℝ := 2 * v

-- Define and state the proof goal
theorem steve_speed_back_from_work :
  ∃ (v : ℝ), speed_to_work v > 0 ∧ speed_back_from_work v = 20 ∧
  (distance_to_work / speed_to_work v + distance_to_work / speed_back_from_work v = total_time) :=
begin
  sorry
end

end steve_speed_back_from_work_l4_4139


namespace initial_gift_card_value_l4_4024

-- The price per pound of coffee
def cost_per_pound : ℝ := 8.58

-- The number of pounds of coffee bought by Rita
def pounds_bought : ℝ := 4.0

-- The remaining balance on Rita's gift card after buying coffee
def remaining_balance : ℝ := 35.68

-- The total cost of the coffee Rita bought
def total_cost_of_coffee : ℝ := cost_per_pound * pounds_bought

-- The initial value of Rita's gift card
def initial_value_of_gift_card : ℝ := total_cost_of_coffee + remaining_balance

-- Statement of the proof problem
theorem initial_gift_card_value : initial_value_of_gift_card = 70.00 :=
by
  -- Placeholder for the proof
  sorry

end initial_gift_card_value_l4_4024


namespace triangle_relation_l4_4457

-- Define the given conditions and the hypothesis
variables {A B C M N K : Type*}
variables (triangle : A → B → C → Type*)
variables [midpoint M A C] [on_segment N A M] [extension K N B]
variables [right_angle (angle BMK)] [angle_eq (angle MBN) (angle CBM)]

-- Define segments and lengths
variables [segment BC] [segment AK] [segment BK]

-- State the theorem
theorem triangle_relation 
  (h1: midpoint M A C) 
  (h2: on_segment N A M) 
  (h3: angle_eq (angle MBN) (angle CBM))
  (h4: right_angle (angle BMK)) : 
  length BC = length AK + length BK := by
  sorry

end triangle_relation_l4_4457


namespace melanie_attended_games_l4_4551

/-- Melanie attended 5 football games if there were 12 total games and she missed 7. -/
theorem melanie_attended_games (totalGames : ℕ) (missedGames : ℕ) (h₁ : totalGames = 12) (h₂ : missedGames = 7) :
  totalGames - missedGames = 5 := 
sorry

end melanie_attended_games_l4_4551


namespace tom_search_cost_l4_4095

theorem tom_search_cost (n : ℕ) (h1 : n = 10) :
  let first_5_days_cost := 5 * 100 in
  let remaining_days_cost := (n - 5) * 60 in
  let total_cost := first_5_days_cost + remaining_days_cost in
  total_cost = 800 :=
by
  -- conditions
  have h2 : first_5_days_cost = 500 := by rfl
  have h3 : remaining_days_cost = 300 := by
    have rem_day_count : n - 5 = 5 := by
      rw [h1]
      rfl
    rfl
  have h4 : total_cost = 500 + 300 :=
    by rfl
  -- conclusion
  rw [h2, h3] at h4
  exact h4

end tom_search_cost_l4_4095


namespace largest_possible_d_l4_4891

theorem largest_possible_d (a b c d : ℝ) (h1 : a + b + c + d = 10) (h2 : ab + ac + ad + bc + bd + cd = 20) : 
    d ≤ (5 + Real.sqrt 105) / 2 :=
by
  sorry

end largest_possible_d_l4_4891


namespace find_number_l4_4325

theorem find_number (y : ℝ) (h : (30 / 100) * y = (25 / 100) * 40) : y = 100 / 3 :=
by
  sorry

end find_number_l4_4325


namespace area_of_rotated_squares_l4_4979

theorem area_of_rotated_squares : 
  let side_length := 8
  let radius := (side_length * Real.sqrt 2) / 2
  in
  let area_of_one_triangle := (1 / 2) * radius^2 * Real.sin (Real.pi / 12) -- 15 degrees in radians
  in
  let total_area := 24 * area_of_one_triangle
  in
  total_area ≈ 99.38 :=
by
  let side_length := 8
  let radius := (side_length * Real.sqrt 2) / 2
  let area_of_one_triangle := (1 / 2) * radius^2 * Real.sin (Real.pi / 12)
  let total_area := 24 * area_of_one_triangle
  have h : total_area ≈ 99.38, sorry
  exact h

end area_of_rotated_squares_l4_4979


namespace age_of_B_l4_4646

theorem age_of_B (A B C : ℕ) (h1 : A = 2 * C + 2) (h2 : B = 2 * C) (h3 : A + B + C = 27) : B = 10 :=
by
  sorry

end age_of_B_l4_4646


namespace sine_condition_l4_4146

variable {α β : ℝ}

theorem sine_condition (h₁ : α = β → sin α = sin β) (h₂ : sin α ≠ sin β → α ≠ β) :  
  (α ≠ β → sin α ≠ sin β) ∧ (sin α ≠ sin β → α ≠ β) :=
by 
  sorry

end sine_condition_l4_4146


namespace sum_of_solutions_l4_4120

theorem sum_of_solutions : ∀ x : ℚ, (4 * x + 6) * (3 * x - 8) = 0 → 
  (x = -3 / 2 ∨ x = 8 / 3) → 
  (-3 / 2 + 8 / 3) = 7 / 6 :=
by
  intros x h1 h2
  sorry

end sum_of_solutions_l4_4120


namespace greatest_x_integer_l4_4106

theorem greatest_x_integer (x : ℤ) (h : ∃ n : ℤ, x^2 + 2 * x + 7 = (x - 4) * n) : x ≤ 35 :=
sorry

end greatest_x_integer_l4_4106


namespace triangle_relation_l4_4451

-- Define the given conditions and the hypothesis
variables {A B C M N K : Type*}
variables (triangle : A → B → C → Type*)
variables [midpoint M A C] [on_segment N A M] [extension K N B]
variables [right_angle (angle BMK)] [angle_eq (angle MBN) (angle CBM)]

-- Define segments and lengths
variables [segment BC] [segment AK] [segment BK]

-- State the theorem
theorem triangle_relation 
  (h1: midpoint M A C) 
  (h2: on_segment N A M) 
  (h3: angle_eq (angle MBN) (angle CBM))
  (h4: right_angle (angle BMK)) : 
  length BC = length AK + length BK := by
  sorry

end triangle_relation_l4_4451


namespace fomagive_55_l4_4587

variables (F E Y : ℕ)

-- Conditions
def condition1 := F - E = 140
def condition2 := F - Y = 40
def condition3 := Y = E + 70

-- Proposition: Foma should give 55 coins to Ierema
theorem fomagive_55 (h1 : condition1 F E) (h2 : condition2 F Y) (h3 : condition3 E Y) : ∃ G, G = 55 :=
by
  have := h1,
  have := h2,
  have := h3,
  sorry

end fomagive_55_l4_4587


namespace Tom_search_cost_l4_4096

theorem Tom_search_cost (first_5_days_rate: ℕ) (first_5_days: ℕ) (remaining_days_rate: ℕ) (total_days: ℕ) : 
  first_5_days_rate = 100 → 
  first_5_days = 5 → 
  remaining_days_rate = 60 → 
  total_days = 10 → 
  (first_5_days * first_5_days_rate + (total_days - first_5_days) * remaining_days_rate) = 800 := 
by 
  intros h1 h2 h3 h4 
  sorry

end Tom_search_cost_l4_4096


namespace certain_fraction_exists_l4_4742

theorem certain_fraction_exists (a b : ℚ) (h : a / b = 3 / 4) :
  (a / b) / (1 / 5) = (3 / 4) / (2 / 5) :=
by
  sorry

end certain_fraction_exists_l4_4742


namespace determine_parameters_l4_4247

theorem determine_parameters
(eq_poly : ∀ x : ℝ, x^5 + 2*x^4 + a*x^2 + b*x = c) :
  ({ -1, 1 } : set ℝ) = { x : ℝ | x^5 + 2*x^4 + a*x^2 + b*x = c } →
  a = -6 ∧ b = -1 ∧ c = -4 :=
by 
  -- Proof can go here
  sorry

end determine_parameters_l4_4247


namespace total_people_on_trip_l4_4166

-- Define the conditions
def vans := 3
def vans_student_capacity := 7

def buses := 5
def buses_student_capacity := 25
def buses_teacher_capacity := 2

def minibuses := 2
def minibuses_student_capacity := 12
def minibuses_teacher_capacity := 1

def science_students := 60
def science_teachers := 6

def language_students := 65
def language_teachers := 5

def total_students := science_students + language_students
def total_teachers := science_teachers + language_teachers

def total_students := 60 + 65
def total_teachers := 6 + 5

-- Proof that the total number of people is equal to 136
theorem total_people_on_trip : 
  total_students + total_teachers = 136 := by 
  sorry

end total_people_on_trip_l4_4166


namespace b_plus_c_eq_neg3_l4_4785

theorem b_plus_c_eq_neg3 (b c : ℝ)
  (h1 : ∀ x : ℝ, x^2 + b * x + c > 0 ↔ (x < -1 ∨ x > 2)) :
  b + c = -3 :=
sorry

end b_plus_c_eq_neg3_l4_4785


namespace tic_tac_toe_board_configurations_l4_4346

theorem tic_tac_toe_board_configurations :
  let sections := 4
  let horizontal_vertical_configurations := 6 * 18
  let diagonal_configurations := 2 * 20
  let configurations_per_section := horizontal_vertical_configurations + diagonal_configurations
  let total_configurations := sections * configurations_per_section
  total_configurations = 592 :=
by 
  let sections := 4
  let horizontal_vertical_configurations := 6 * 18
  let diagonal_configurations := 2 * 20
  let configurations_per_section := horizontal_vertical_configurations + diagonal_configurations
  let total_configurations := sections * configurations_per_section
  sorry

end tic_tac_toe_board_configurations_l4_4346


namespace equalize_foma_ierema_l4_4577

theorem equalize_foma_ierema (F E Y : ℕ) 
  (h1 : F - 70 = E + 70) 
  (h2 : F - 40 = Y) 
  (h3 : Y = E + 70) 
  : ∃ x : ℕ, x = 55 ∧ F - x = E + x :=
by
  use 55
  sorry

end equalize_foma_ierema_l4_4577


namespace foma_should_give_ierema_55_coins_l4_4570

variables (F E Y : ℝ)

-- Conditions
def condition1 : Prop := F - 70 = E + 70
def condition2 : Prop := F - 40 = Y
def condition3 : Prop := E + 70 = Y

theorem foma_should_give_ierema_55_coins
  (h1 : condition1 F E) (h2 : condition2 F Y) (h3 : condition3 E Y) :
  F - (E + 55) = E + 55 :=
by
  sorry

end foma_should_give_ierema_55_coins_l4_4570


namespace price_on_friday_l4_4008

variables (P : ℝ) (monday_price friday_price : ℝ)
-- Conditions
def condition1 : Prop := monday_price = P * 1.10
def condition2 : Prop := friday_price = P * 0.90
def condition3 : Prop := monday_price = 5.5

theorem price_on_friday : condition1 ∧ condition2 ∧ condition3 → friday_price = 4.5 :=
by
  assume h,
  sorry

end price_on_friday_l4_4008


namespace fomagive_55_l4_4588

variables (F E Y : ℕ)

-- Conditions
def condition1 := F - E = 140
def condition2 := F - Y = 40
def condition3 := Y = E + 70

-- Proposition: Foma should give 55 coins to Ierema
theorem fomagive_55 (h1 : condition1 F E) (h2 : condition2 F Y) (h3 : condition3 E Y) : ∃ G, G = 55 :=
by
  have := h1,
  have := h2,
  have := h3,
  sorry

end fomagive_55_l4_4588


namespace range_of_a_l4_4778

theorem range_of_a (a : ℝ) 
  (P : (∀ x : ℝ, x > 0 → log_base a x > log_base a x) → Prop) 
  (Q : (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) → Prop) 
  (hP_or_Q : P ∨ Q) (hP_and_Q_false : ¬(P ∧ Q)) : 
  a > 2 ∨ -2 < a ∧ a ≤ 1 :=
sorry

end range_of_a_l4_4778


namespace equations_of_motion_velocity_components_l4_4711

noncomputable def omega := 10 -- rad/s
noncomputable def OA := 90 -- cm
noncomputable def AB := 90 -- cm
noncomputable def AL := (1 / 3) * AB -- cm

def x_A (t : ℝ) : ℝ := OA * Real.cos (omega * t)
def y_A (t : ℝ) : ℝ := OA * Real.sin (omega * t)

def x_L (t θ : ℝ) : ℝ := x_A t + AL * Real.cos θ
def y_L (t θ : ℝ) : ℝ := y_A t + AL * Real.sin θ

def v_x_L (t : ℝ) : ℝ := -1200 * Real.sin (omega * t)
def v_y_L (t : ℝ) : ℝ := 1200 * Real.cos (omega * t)

theorem equations_of_motion (t θ : ℝ) :
  x_L t θ = 90 * Real.cos (10 * t) + 30 * Real.cos θ ∧
  y_L t θ = 90 * Real.sin (10 * t) + 30 * Real.sin θ :=
sorry

theorem velocity_components (t : ℝ) :
  v_x_L t = -1200 * Real.sin (10 * t) ∧
  v_y_L t = 1200 * Real.cos (10 * t) :=
sorry

end equations_of_motion_velocity_components_l4_4711


namespace find_angle_QRT_l4_4767

-- Definitions of given angles and geometric properties
variable {P Q R S T : Type} [InCircle P Q R S]
variable (angle_PQS angle_PSR : ℝ)
variable (extended_QR : Extends QR T)

-- Conditions from the problem statement
def cyclic_quadrilateral : Prop := InCircle P Q R S
def angle_PQS_value : Prop := angle_PQS = 82
def angle_PSR_value : Prop := angle_PSR = 58

-- Question to prove
theorem find_angle_QRT
  (h1 : cyclic_quadrilateral)
  (h2 : angle_PQS_value)
  (h3 : angle_PSR_value)
  (h4 : extended_QR) : 
  ∃ angle_QRT, angle_QRT = 58 := 
sorry

end find_angle_QRT_l4_4767


namespace tan_angle_QDE_l4_4484

theorem tan_angle_QDE
  (Q D E F : Type)
  [InnerProductSpace ℝ Q D E F] 
  (a b c : ℝ)
  (DE EF FD : ℝ)
  (h1 : DE = 8)
  (h2 : EF = 10)
  (h3 : FD = 12)
  (h4 : ∃ (phi : ℝ), phi = ∠QDE ∧ phi = ∠QEF ∧ phi = ∠QFD) :
  ∃ (tan_phi : ℝ), tan_phi = (45 * Real.sqrt 7) / 77 :=
by {
  sorry
}

end tan_angle_QDE_l4_4484


namespace function_not_below_line_l4_4294

noncomputable def f (x : ℝ) := Real.exp x * Real.sin x

theorem function_not_below_line (k : ℝ) :
  (∀ x ∈ Icc (0 : ℝ) (Real.pi / 2), f x ≥ k * x) ↔ k ∈ Set.Iic (1 : ℝ) := by
sorry

end function_not_below_line_l4_4294


namespace find_a_l4_4756

noncomputable def f (x : ℝ) : ℝ :=
  3 * x^2 + 2 * x + 1

theorem find_a :
  (∫ x in -1..1, f x) = 2 * f a → 
  (a = 1/3 ∨ a = -1) :=
begin
  sorry
end

end find_a_l4_4756


namespace problem1_problem2_l4_4807

noncomputable def line_pass through_point (k : ℝ) : Prop :=
  ∃ P : ℝ × ℝ, P = (2, -1)

noncomputable def line_intersects_axes (k : ℝ) : Prop :=
  let A := (1 + 2 * k) / k in
  let B := -2 * k - 1 in
  A = B

theorem problem1 (k : ℝ) :
  line_pass k ↔ ∃ P: ℝ × ℝ, P = (2, -1) :=
by
  sorry

theorem problem2 (k : ℝ) :
  line_intersects_axes k → k = -1 :=
by
  sorry

end problem1_problem2_l4_4807


namespace smallest_positive_period_monotonically_decreasing_intervals_l4_4805

noncomputable def f (x : ℝ) : ℝ := sin(x) ^ 2 + sqrt 3 * sin(x) * cos(x) + 2 * cos(x) ^ 2

theorem smallest_positive_period (x : ℝ) : ∃ T > 0, ∀ x, f(x + T) = f(x) :=
by {
  use π,
  sorry
}

theorem monotonically_decreasing_intervals (k : ℤ) : ∃ a b : ℝ, a = π / 6 + k * π ∧ b = 2 * π / 3 + k * π ∧ ∀ x ∈ Icc a b, f'(x) < 0 :=
by {
  use (π/6 + k * π),
  use (2 * π / 3 + k * π),
  sorry
}

end smallest_positive_period_monotonically_decreasing_intervals_l4_4805


namespace green_tea_price_in_july_l4_4837

theorem green_tea_price_in_july :
  ∃ x : ℝ, ( 0.2 * x + 6 * x + 1.5 * x = 8.35 ) ∧ ( (0.1 * x) = 0.1084 ) :=
by
  -- Let x be the cost per pound of green tea, coffee, and black tea in June
  let x := 8.35 / 7.7
  use x
  split
  -- The total cost equation for the mixture in July
  { calc
      0.2 * x + 6 * x + 1.5 * x
          = 7.7 * x : by ring
      ... = 8.35 : by simp [mul_div_cancel' _ (ne_of_gt (by norm_num : (7.7 : ℝ) > 0))] }
  -- The cost of green tea per pound in July
  { calc
      0.1 * x
          = 0.1 * (8.35 / 7.7) : by congr
      ... = 0.1084 : by norm_num }

end green_tea_price_in_july_l4_4837


namespace find_second_sum_l4_4136

def sum : ℕ := 2717
def interest_rate_first : ℚ := 3 / 100
def interest_rate_second : ℚ := 5 / 100
def time_first : ℕ := 8
def time_second : ℕ := 3

theorem find_second_sum (x : ℚ) (h : x * interest_rate_first * time_first = (sum - x) * interest_rate_second * time_second) : 
  sum - x = 2449 :=
by
  sorry

end find_second_sum_l4_4136


namespace limit_arcsin_3x_over_sqrt_2x_minus_sqrt_2_l4_4626

/-- Definition of the limit problem to be proved -/
def limit_problem : Prop :=
  (∀ (x : ℝ), x ≠ 0 → 
  (∃ (d : ℝ), 0 < d ∧ 
  (∀ y (h : |y| < d), abs ((arcsin (3 * y)) / (sqrt (2 + y) - sqrt 2) - 6 * sqrt 2) < x)))

theorem limit_arcsin_3x_over_sqrt_2x_minus_sqrt_2 :
  limit_problem :=
by
  sorry

end limit_arcsin_3x_over_sqrt_2x_minus_sqrt_2_l4_4626


namespace min_value_of_reciprocal_sum_l4_4351

noncomputable def arithmetic_sequence_condition (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ ((2016 * (a 1 + a 2016)) / 2 = 1008)

theorem min_value_of_reciprocal_sum (a : ℕ → ℝ) (h : arithmetic_sequence_condition a) :
  ∃ x : ℝ, x = 4 ∧ (∀ y, y = (1 / a 1001 + 1 / a 1016) → x ≤ y) :=
sorry

end min_value_of_reciprocal_sum_l4_4351


namespace grooming_time_equals_640_seconds_l4_4866

variable (cat_claws_per_foot : Nat) (cat_foot_count : Nat)
variable (nissa_clip_time_per_claw : Nat) (nissa_clean_time_per_ear : Nat) (nissa_shampoo_time_minutes : Nat) 
variable (cat_ear_count : Nat)
variable (seconds_per_minute : Nat)

def total_grooming_time (cat_claws_per_foot * cat_foot_count : nissa_clip_time_per_claw) (nissa_clean_time_per_ear * cat_ear_count) (nissa_shampoo_time_minutes * seconds_per_minute) := sorry

theorem grooming_time_equals_640_seconds : 
  cat_claws_per_foot = 4 →
  cat_foot_count = 4 →
  nissa_clip_time_per_claw = 10 →
  nissa_clean_time_per_ear = 90 →
  nissa_shampoo_time_minutes = 5 →
  cat_ear_count = 2 →
  seconds_per_minute = 60 →
  total_grooming_time = 160 + 180 + 300 → 
  total_grooming_time = 640 := sorry

end grooming_time_equals_640_seconds_l4_4866


namespace probability_of_product_ending_with_zero_l4_4665
open BigOperators

def probability_product_ends_with_zero :=
  let no_zero := (9 / 10) ^ 20
  let at_least_one_zero := 1 - no_zero
  let no_even := (5 / 9) ^ 20
  let at_least_one_even := 1 - no_even
  let no_five_among_19 := (8 / 9) ^ 19
  let at_least_one_five := 1 - no_five_among_19
  let no_zero_and_conditions :=
    no_zero * at_least_one_even * at_least_one_five
  at_least_one_zero + no_zero_and_conditions

theorem probability_of_product_ending_with_zero :
  abs (probability_product_ends_with_zero - 0.988) < 0.001 :=
by
  sorry

end probability_of_product_ending_with_zero_l4_4665


namespace time_to_eat_potatoes_l4_4332

theorem time_to_eat_potatoes (rate : ℕ → ℕ → ℝ) (potatoes : ℕ → ℕ → ℝ) 
    (minutes : ℕ) (hours : ℝ) (total_potatoes : ℕ) : 
    rate 3 20 = 9 / 1 -> potatoes 27 9 = 3 := 
by
  intro h1
  -- You can add intermediate steps here as optional comments for clarity during proof construction
  /- 
  Given: 
  rate 3 20 = 9 -> Jason's rate of eating potatoes is 9 potatoes per hour
  time = potatoes / rate -> 27 potatoes / 9 potatoes/hour = 3 hours
  -/
  sorry

end time_to_eat_potatoes_l4_4332


namespace calculate_radius_of_film_l4_4471

def density_X := 0.8 -- g/cm^3
def thickness_film := 0.2 -- cm
def box_length := 8 -- cm
def box_width := 6 -- cm
def box_height := 4.5 -- cm
def density_water := 1 -- g/cm^3
def volume_X := box_length * box_width * box_height -- volume of liquid X

def radius_of_film : Real := sqrt (864 / π)

theorem calculate_radius_of_film :
  let mass_X := volume_X * density_X
  let equivalent_volume := mass_X / density_water
  let R := (equivalent_volume / (π * thickness_film))
  radius_of_film = sqrt R :=
by
  sorry

end calculate_radius_of_film_l4_4471


namespace ellipse_standard_eq_slope_sum_constant_l4_4774

-- Definitions of the given conditions
def ellipse_equation (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) (x y : ℝ) :=
  (x^2) / (a^2) + (y^2) / (b^2) = 1

def line_equation (k x : ℝ) := k * x + 2

def intersects (p1 p2 : ℝ × ℝ) := p1.1 = p2.1

-- Questions converted into Lean statements

-- 1. Prove the standard equation of the ellipse
theorem ellipse_standard_eq (h1 : ∀ x y : ℝ, ellipse_equation (sqrt 2) 1 (by norm_num) (by norm_num) x y) : 
  ∀ x y : ℝ, (x^2) / 2 + (y^2) = 1 :=
sorry

-- 2. Existence of point D on y-axis with constant sum of slopes
theorem slope_sum_constant (h1 : ∀ x y : ℝ, ellipse_equation (sqrt 2) 1 (by norm_num) (by norm_num) x y)
  (h2 : ∃ A B : ℝ × ℝ, (line_equation k A.1 = A.2) ∧ (line_equation k B.1 = B.2) ∧ (intersects A B)) : 
  ∃ D : ℝ × ℝ, D = (0, 1/2) ∧ (λ A B D, ((A.2 - D.2) / (A.1 - D.1)) + ((B.2 - D.2) / (B.1 - D.1))) = 0 :=
sorry

end ellipse_standard_eq_slope_sum_constant_l4_4774


namespace year_with_greatest_temp_increase_l4_4949

def avg_temp (year : ℕ) : ℝ :=
  match year with
  | 2000 => 2.0
  | 2001 => 2.3
  | 2002 => 2.5
  | 2003 => 2.7
  | 2004 => 3.9
  | 2005 => 4.1
  | 2006 => 4.2
  | 2007 => 4.4
  | 2008 => 3.9
  | 2009 => 3.1
  | _    => 0.0

theorem year_with_greatest_temp_increase : ∃ year, year = 2004 ∧
  (∀ y, 2000 < y ∧ y ≤ 2009 → avg_temp y - avg_temp (y - 1) ≤ avg_temp 2004 - avg_temp 2003) := by
  sorry

end year_with_greatest_temp_increase_l4_4949


namespace triangle_relation_l4_4453

-- Define the given conditions and the hypothesis
variables {A B C M N K : Type*}
variables (triangle : A → B → C → Type*)
variables [midpoint M A C] [on_segment N A M] [extension K N B]
variables [right_angle (angle BMK)] [angle_eq (angle MBN) (angle CBM)]

-- Define segments and lengths
variables [segment BC] [segment AK] [segment BK]

-- State the theorem
theorem triangle_relation 
  (h1: midpoint M A C) 
  (h2: on_segment N A M) 
  (h3: angle_eq (angle MBN) (angle CBM))
  (h4: right_angle (angle BMK)) : 
  length BC = length AK + length BK := by
  sorry

end triangle_relation_l4_4453


namespace x_sum_inequality_l4_4761

theorem x_sum_inequality (n : ℕ) (h1 : 2 ≤ n) (x : ℕ → ℝ)
  (h2 : (∑ i in Finset.range n, |x i|) = 1)
  (h3 : (∑ i in Finset.range n, x i) = 0) :
  |∑ i in Finset.range n, x i / (i + 1)| ≤ 1/2 - 1/(2 * n) :=
by
  sorry

end x_sum_inequality_l4_4761


namespace walking_time_l4_4168

theorem walking_time 
  (speed_km_hr : ℝ := 10) 
  (distance_km : ℝ := 6) 
  : (distance_km / (speed_km_hr / 60)) = 36 :=
by
  sorry

end walking_time_l4_4168


namespace sqrt_23_parts_xy_diff_l4_4491

-- Problem 1: Integer and decimal parts of sqrt(23)
theorem sqrt_23_parts : ∃ (integer_part : ℕ) (decimal_part : ℝ), 
  integer_part = 4 ∧ decimal_part = Real.sqrt 23 - 4 ∧
  (integer_part : ℝ) + decimal_part = Real.sqrt 23 :=
by
  sorry

-- Problem 2: x - y for 9 + sqrt(3) = x + y with given conditions
theorem xy_diff : 
  ∀ (x y : ℝ), x = 10 → y = Real.sqrt 3 - 1 → x - y = 11 - Real.sqrt 3 :=
by
  sorry

end sqrt_23_parts_xy_diff_l4_4491


namespace minimum_questions_to_determine_sequence_l4_4607

theorem minimum_questions_to_determine_sequence (n : ℕ) (x : ℕ → ℤ) (a : ℕ → ℤ) :
  (∀ i, 1 ≤ i → i ≤ n → (1 ≤ x i ∧ x i ≤ 9) ∨ (-9 ≤ x i ∧ x i ≤ -1)) →
  (∑ i in finset.range n, a i * x (i + 1)) = 
  (∑ i in finset.range n, (100 ^ (i + 1)) * x (i + 1)) →
  (∃ f : ℤ → ℤ, (∀ i, 1 ≤ i → i ≤ n → f (100 * i) = x i)) :=
sorry

end minimum_questions_to_determine_sequence_l4_4607


namespace incorrect_statement_is_A_l4_4157

open List

def reading_times : List ℕ := [2, 2, 4, 4, 4, 6, 6, 6, 6, 8]

-- Definitions for the different statistics
def mode (l : List ℕ) : ℕ := modeOf l
def mean (l : List ℕ) : ℝ := ((l.sum : ℝ) / l.length)
def median (l : List ℕ) : ℝ :=
  let sorted_l := sort l in
  if sorted_l.length % 2 = 1 then
    sorted_l[(sorted_l.length / 2)] -- for odd length
  else
    ((sorted_l[(sorted_l.length / 2) - 1] + sorted_l[(sorted_l.length / 2)]) / 2 : ℝ) -- for even length

-- The theorem states that the incorrect statement is A
theorem incorrect_statement_is_A : 
  (mode reading_times ≠ 1) ∧ 
  (mean reading_times = 4.8) ∧ 
  (reading_times.length = 10) ∧ 
  (median reading_times ≠ 5) := sorry

end incorrect_statement_is_A_l4_4157


namespace red_robin_team_arrangements_l4_4940

theorem red_robin_team_arrangements :
  let boys := 3
  let girls := 4
  let choose2 (n : ℕ) (k : ℕ) := Nat.choose n k
  let permutations (n : ℕ) := Nat.factorial n
  let waysToPositionBoys := choose2 boys 2 * permutations 2
  let waysToPositionRemainingMembers := permutations (boys - 2 + girls)
  waysToPositionBoys * waysToPositionRemainingMembers = 720 :=
by
  let boys := 3
  let girls := 4
  let choose2 (n : ℕ) (k : ℕ) := Nat.choose n k
  let permutations (n : ℕ) := Nat.factorial n
  let waysToPositionBoys := choose2 boys 2 * permutations 2
  let waysToPositionRemainingMembers := permutations (boys - 2 + girls)
  have : waysToPositionBoys * waysToPositionRemainingMembers = 720 := 
    by sorry -- Proof omitted here
  exact this

end red_robin_team_arrangements_l4_4940


namespace digit_a2008_l4_4034

noncomputable def three_prime_factors (n : ℕ) : Prop :=
  ∃ p q r : ℕ, nat.prime p ∧ nat.prime q ∧ nat.prime r ∧ p ≠ q ∧ q ≠ r ∧ r ≠ p ∧ n = p * q * r

theorem digit_a2008 (a : ℕ → ℕ) (h : ∀ i : ℕ, 1 ≤ i ∧ i ≤ 2007 → three_prime_factors (10 * a i + a (i + 1))) :
  a 2008 = 6 := 
sorry

end digit_a2008_l4_4034


namespace triangle_relation_l4_4454

-- Define the given conditions and the hypothesis
variables {A B C M N K : Type*}
variables (triangle : A → B → C → Type*)
variables [midpoint M A C] [on_segment N A M] [extension K N B]
variables [right_angle (angle BMK)] [angle_eq (angle MBN) (angle CBM)]

-- Define segments and lengths
variables [segment BC] [segment AK] [segment BK]

-- State the theorem
theorem triangle_relation 
  (h1: midpoint M A C) 
  (h2: on_segment N A M) 
  (h3: angle_eq (angle MBN) (angle CBM))
  (h4: right_angle (angle BMK)) : 
  length BC = length AK + length BK := by
  sorry

end triangle_relation_l4_4454


namespace triangle_equality_lemma_l4_4404

open EuclideanGeometry

theorem triangle_equality_lemma 
  (A B C M N K: Point)
  (hM_midpoint: midpoint M A C)
  (hN_on_AM: lies_on N A M)
  (h_angle_eq: ∠ M B N = ∠ C B M)
  (hK_right: ∠ B M K = 90) :
  dist B C = dist A K + dist B K :=
by
  sorry

end triangle_equality_lemma_l4_4404


namespace sum_of_angles_isosceles_triangles_l4_4985

theorem sum_of_angles_isosceles_triangles :
  (∃ A B : ℕ, A + A + 70 = 180 ∧ 2 ∣ A) ∧
  (∃ C D : ℕ, C + C + 70 = 180 ∧ ¬ 2 ∣ C) →
  let S := 2 * A,
      T := 2 * C in
  S + T = 250 := 
by
  sorry -- Proof not provided

end sum_of_angles_isosceles_triangles_l4_4985


namespace sqrt_9_eq_pos_neg_3_l4_4114

theorem sqrt_9_eq_pos_neg_3 : ∀ x : ℝ, x^2 = 9 ↔ x = 3 ∨ x = -3 :=
by
  sorry

end sqrt_9_eq_pos_neg_3_l4_4114


namespace sum_first_100_terms_l4_4268

def seq (n : ℕ) : ℕ → ℕ
| 1 := 2
| (bit0 n) := seq n + 1
| (bit1 n) := n - seq n

theorem sum_first_100_terms : (Finset.range 100).sum seq = 1289 := by
  sorry

end sum_first_100_terms_l4_4268


namespace part1_part2_l4_4876

-- Definitions for the first problem
def Sn (n : ℕ) : ℕ := n ^ 2

def a (n : ℕ) : ℕ := 2 * n - 1

theorem part1 : ∀ n : ℕ, n > 0 → a n = Sn n - Sn (n - 1) ∧ ∃ d : ℕ, ∀ n : ℕ, n > 0 → a (n + 1) - a n = d :=
by
  sorry

-- Definitions for the second problem
def b (n : ℕ) : ℝ := 1 / (a n * a (n + 1))

def T (n : ℕ) : ℝ := ∑ i in finset.range (n + 1), b i

theorem part2 : ∀ n : ℕ, n > 0 → ∀ m : ℝ, (T n > m) → m < 1 / 3 :=
by
  sorry

end part1_part2_l4_4876


namespace ben_last_10_shots_l4_4182

theorem ben_last_10_shots :
  let initial_shots := 30 in
  let initial_success_rate := 0.60 in
  let additional_shots := 10 in
  let new_success_rate := 0.62 in
  let initial_successful_shots := initial_shots * initial_success_rate in
  let total_shots := initial_shots + additional_shots in
  let total_successful_shots := total_shots * new_success_rate in
  let additional_successful_shots := total_successful_shots - initial_successful_shots in
  additional_successful_shots = 7 :=
sorry

end ben_last_10_shots_l4_4182


namespace sum_of_solutions_l4_4119

theorem sum_of_solutions : ∀ x : ℚ, (4 * x + 6) * (3 * x - 8) = 0 → 
  (x = -3 / 2 ∨ x = 8 / 3) → 
  (-3 / 2 + 8 / 3) = 7 / 6 :=
by
  intros x h1 h2
  sorry

end sum_of_solutions_l4_4119


namespace dvaneft_shares_percentage_range_l4_4653

theorem dvaneft_shares_percentage_range :
  ∀ (x y z n m : ℝ),
    (4 * x * n = y * m) →
    (x * n + y * m = z * (m + n)) →
    (16 ≤ y - x ∧ y - x ≤ 20) →
    (42 ≤ z ∧ z ≤ 60) →
    (12.5 ≤ (n / (2 * (n + m)) * 100) ∧ (n / (2 * (n + m)) * 100) ≤ 15) :=
by
  intros x y z n m h1 h2 h3 h4
  sorry

end dvaneft_shares_percentage_range_l4_4653


namespace vector_parallel_solution_l4_4814

theorem vector_parallel_solution (x : ℝ) : 
  let a := (matrix.vec_cons (-2 : ℝ) (matrix.vec_cons (1 : ℝ) matrix.vec_nil))
  let b := (matrix.vec_cons (x : ℝ) (matrix.vec_cons (3 : ℝ) matrix.vec_nil))
  in a ∥ b → x = -6 := 
by
  sorry

end vector_parallel_solution_l4_4814


namespace triangle_angle_bisector_segment_length_l4_4856

theorem triangle_angle_bisector_segment_length
  (A B C D : Type) [metric_space A]
  (AD DC DB AB : ℝ)
  (h1 : D ∈ line_segment A C)
  (h2 : D ∈ line_segment B A)
  (h3 : angle ∠ A D B = angle ∠ A D C)
  (hAD : AD = 15)
  (hDC : DC = 45)
  (hDB : DB = 24)
  : AB = 39 :=
sorry

end triangle_angle_bisector_segment_length_l4_4856


namespace problem_equivalent_statement_l4_4217

open Real

noncomputable def p' := 
  (∏ i in [2, 3, 3, 4, 4, 2, 2], Nat.factorial i)⁻¹ * Nat.factorial 20

noncomputable def q' := 
  (binom 7 2) * (∏ i in [4, 4, 4, 4, 4, 0, 0], Nat.factorial i)⁻¹ * Nat.factorial 20

-- Main goal
theorem problem_equivalent_statement : p' / q' = 37 :=
sorry

end problem_equivalent_statement_l4_4217


namespace mixed_feed_total_pounds_l4_4089

theorem mixed_feed_total_pounds 
  (cheap_feed_cost : ℝ) (expensive_feed_cost : ℝ) (mix_cost : ℝ) 
  (cheap_feed_amount : ℕ) :
  cheap_feed_cost = 0.18 → 
  expensive_feed_cost = 0.53 → 
  mix_cost = 0.36 → 
  cheap_feed_amount = 17 → 
  (∃ (expensive_feed_amount : ℕ), 
    (cheap_feed_amount + expensive_feed_amount = 35)) :=
begin
  intros,
  use 18, -- We introduce 18 as the amount of more expensive feed
  sorry, -- Proof goes here
end

end mixed_feed_total_pounds_l4_4089


namespace simplify_fraction_l4_4931

theorem simplify_fraction (x y z : ℕ) (hx : x = 5) (hy : y = 2) (hz : z = 4) :
  (10 * x^2 * y^3 * z) / (15 * x * y^2 * z^2) = 4 / 3 :=
by
  sorry

end simplify_fraction_l4_4931


namespace sum_of_angles_l4_4044

theorem sum_of_angles (h₁ : ∀ i, i ∈ finset.range 18 → segment_of_circle i)
    (h₂ : central_angle_of_segment 1 = 20)
    (h₃ : central_angle_of_partial_circle 3 = 60)
    (h₄ : central_angle_of_partial_circle 6 = 120)
    (h₅ : inscribed_angle_of_partial_circle 3 = 30)
    (h₆ : inscribed_angle_of_partial_circle 6 = 60) :
    (30 + 60 = 90) := by 
    sorry

end sum_of_angles_l4_4044


namespace expression_evaluation_l4_4728

theorem expression_evaluation :
  (4 * 6 / (12 * 14) * (8 * 12 * 14) / (4 * 6 * 8) - 1 = 0) :=
by sorry

end expression_evaluation_l4_4728


namespace foma_should_give_ierema_55_coins_l4_4568

variables (F E Y : ℝ)

-- Conditions
def condition1 : Prop := F - 70 = E + 70
def condition2 : Prop := F - 40 = Y
def condition3 : Prop := E + 70 = Y

theorem foma_should_give_ierema_55_coins
  (h1 : condition1 F E) (h2 : condition2 F Y) (h3 : condition3 E Y) :
  F - (E + 55) = E + 55 :=
by
  sorry

end foma_should_give_ierema_55_coins_l4_4568


namespace min_positive_period_interval_monotonic_increase_max_min_values_l4_4802

noncomputable def f (x : ℝ) : ℝ := cos x * (sin x + cos x) - 1 / 2

theorem min_positive_period :
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧ T = π := by sorry

theorem interval_monotonic_increase (k : ℤ) :
  ∃ a b, (∀ x, a ≤ x ∧ x ≤ b → f' x > 0) ∧ a = -3 * π / 8 + k * π ∧ b = π / 8 + k * π := by sorry

theorem max_min_values :
  ∃ f_max f_min, (∀ x ∈ set.Icc (-π / 4) (π / 2), f x ≤ f_max ∧ f x ≥ f_min) ∧ f_max = sqrt 2 / 2 ∧ f_min = - 1 / 2 := by sorry

end min_positive_period_interval_monotonic_increase_max_min_values_l4_4802


namespace find_a_for_even_function_l4_4830

theorem find_a_for_even_function (a : ℝ) (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = (x + 1)*(x - a) ∧ f (-x) = f x) : a = 1 :=
sorry

end find_a_for_even_function_l4_4830


namespace triangle_equality_BC_AK_BK_l4_4413

-- Define the main geometric constructs
variables {A B C M N K : Type}

-- Given conditions
variables [Midpoint M A C] -- M is the midpoint of AC
variables [OnSegment N A M] -- N lies on segment AM
variables [EqualAngle M B N C B M] -- ∠MBN = ∠CBM
variables [OnExtension K B N (90 : Angle)] -- K is on the extension of BN such that ∠BMK is right

-- Define the main theorem
theorem triangle_equality_BC_AK_BK :
  BC = AK + BK :=
sorry

end triangle_equality_BC_AK_BK_l4_4413


namespace angle_bisector_divides_CD_in_right_trapezoid_l4_4522

theorem angle_bisector_divides_CD_in_right_trapezoid
    {A B C D O : Type} [IsRightTrapezoid ABCD]
    (h1 : height_AB = base_AD + base_BC)
    (h2 : AB ⊥ AD)
    (h3 : AB ⊥ BC)
    (h4 : O = midpoint CD) :
    divides_angle_bisector B CD 1 1 :=
sorry

end angle_bisector_divides_CD_in_right_trapezoid_l4_4522


namespace oranges_per_box_l4_4696

theorem oranges_per_box (total_oranges : ℕ) (num_boxes : ℕ) (h1 : total_oranges = 24) (h2 : num_boxes = 3) :
  total_oranges / num_boxes = 8 :=
by
  rw [h1, h2]
  exact Nat.div_eq_of_eq_mul (by norm_num : 24 = 3 * 8)

end oranges_per_box_l4_4696


namespace factorial_solution_l4_4738

theorem factorial_solution (k n : ℕ) (hk : 0 < k) (hn : 0 < n) :
  k! = (2^n - 1) * (2^n - 2) * (2^n - 4) * ... * (2^n - 2^(n-1)) ↔ (k, n) = (1, 1) ∨ (k, n) = (3, 2) := 
sorry

end factorial_solution_l4_4738


namespace find_y_l4_4850

-- Define the problem environment
variable (AB : LineSegment)
variable (angleACD : ℝ) (angleECB : ℝ) (y : ℝ)

-- Specify the conditions
axiom angleACD_eq_90 : angleACD = 90
axiom angleECB_eq_65 : angleECB = 65

-- Define the desired property to prove
theorem find_y : angleACD + y + angleECB = 180 → y = 25 := by
  intros h
  have h1 : 90 + y + 65 = 180 := by
    rw [←angleACD_eq_90, ←angleECB_eq_65] at h
    exact h
  linarith
  sorry

end find_y_l4_4850


namespace coefficient_x3_in_expansion_l4_4716

theorem coefficient_x3_in_expansion :
  let general_term (r : ℕ) := (Nat.choose 5 r) * (2 : ℤ)^(5 - r) * (1 / 4)^(r : ℤ) * (x^(5 - 2 * r) : ℤ)
  (r := 1) :
  (2*x + 1/(4*x))^5 = 20 * x^3 + ... := 
by
  intros
  sorry

end coefficient_x3_in_expansion_l4_4716


namespace union_sets_l4_4335

open Set

def A : Set ℝ := {x | -2 < x ∧ x < 1}
def B : Set ℝ := {x | 0 < x ∧ x < 2}

theorem union_sets :
  A ∪ B = {x | -2 < x ∧ x < 2} :=
by
  sorry

end union_sets_l4_4335


namespace cot_difference_l4_4360

theorem cot_difference (A B C D : Point) (h_triangle : triangle A B C)
  (h_median : median A D B C) (h_angle : ∠ A D = 60°)
  (h_midpoint : midpoint D B C) (h_length : dist B D = 2 * dist D C) :
  |cot (angle B A D) - cot (angle C A D)| = (10 * sqrt 3 + 18) / 9 := 
by
  sorry

end cot_difference_l4_4360


namespace blue_area_percentage_of_flag_area_l4_4178

theorem blue_area_percentage_of_flag_area (s : ℕ) (flag_area blue_area cross_area : ℝ) (h1 : flag_area = 100) 
  (h2 : cross_area = 36) (h3 : blue_area = 9) (h4 : cross_area = 0.36 * flag_area) : blue_area / flag_area * 100 = 2 := 
by
  rw [h1, h2] at h4
  have h5 : blue_area = 9 := by sorry
  have h6 : 100 * 0.36 = 36 := by norm_num
  have h7 : blue_area / 100 * 100 = blue_area := by ring
  have h8 : blue_area / 100 * 100 = 9 := by rw [h5]
  have h9 : blue_area / 100 * 100 = 9 := by norm_num
  exact rfl

end blue_area_percentage_of_flag_area_l4_4178


namespace smallest_number_is_numA_l4_4681

-- Define the four numbers from the problem statement
def numA : ℝ := -1
def numB : ℝ := abs (-4)
def numC : ℝ := - (-3)
def numD : ℝ := - (1 / 2)

-- The theorem to prove that numA is the smallest number
theorem smallest_number_is_numA : numA = min numA (min numB (min numC numD)) :=
  by
    -- skipping the actual proof with sorry
    sorry

end smallest_number_is_numA_l4_4681


namespace sum_of_acute_angles_l4_4781

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < π / 2)
variable (hβ : 0 < β ∧ β < π / 2)
variable (hcosα : Real.cos α = 1 / Real.sqrt 10)
variable (hcosβ : Real.cos β = 1 / Real.sqrt 5)

theorem sum_of_acute_angles :
  α + β = 3 * Real.pi / 4 := by
  sorry

end sum_of_acute_angles_l4_4781


namespace central_angle_unchanged_l4_4339

theorem central_angle_unchanged (r s : ℝ) (h_r : r > 0) (h_s : s > 0) :
  let new_r := 2 * r,
      new_s := 2 * s,
      θ := s / r,
      new_θ := new_s / new_r
  in θ = new_θ :=
by
  sorry

end central_angle_unchanged_l4_4339


namespace oranges_per_box_l4_4695

theorem oranges_per_box (total_oranges : ℕ) (num_boxes : ℕ) (h1 : total_oranges = 24) (h2 : num_boxes = 3) :
  total_oranges / num_boxes = 8 :=
by
  rw [h1, h2]
  exact Nat.div_eq_of_eq_mul (by norm_num : 24 = 3 * 8)

end oranges_per_box_l4_4695


namespace one_color_present_in_all_boxes_l4_4836

theorem one_color_present_in_all_boxes:
  ∀ (boxes : Fin 25 → Set ℕ),
  (∀ k (hks : 1 ≤ k ∧ k ≤ 25) (s : Finset (Fin 25)) (hksize : s.card = k),
    (⋃ i ∈ s, boxes i).size = k + 1) →
  ∃ c, ∀ i, c ∈ boxes i := 
sorry

end one_color_present_in_all_boxes_l4_4836


namespace regular_pentagon_diagonal_square_l4_4396

variable (a d : ℝ)
def is_regular_pentagon (a d : ℝ) : Prop :=
d ^ 2 = a ^ 2 + a * d

theorem regular_pentagon_diagonal_square :
  is_regular_pentagon a d :=
sorry

end regular_pentagon_diagonal_square_l4_4396


namespace slope_angle_of_tangent_line_at_1_slope_angle_is_45_degrees_l4_4537

noncomputable def function_expr (x : ℝ) : ℝ :=
  x^3 - 2 * x + 4

noncomputable def derivative_at_1 : ℝ :=
  3 * 1^2 - 2

theorem slope_angle_of_tangent_line_at_1 : derivative_at_1 = 1 :=
by
  sorry

theorem slope_angle_is_45_degrees : derivative_at_1 = 1 → (atan 1) = Real.pi / 4 :=
by
  sorry

end slope_angle_of_tangent_line_at_1_slope_angle_is_45_degrees_l4_4537


namespace scientific_notation_for_70_million_l4_4635

-- Define the parameters for the problem
def scientific_notation (x : ℕ) (a : ℝ) (n : ℤ) : Prop :=
  x = a * 10 ^ n ∧ 1 ≤ |a| ∧ |a| < 10

-- Problem statement
theorem scientific_notation_for_70_million :
  scientific_notation 70000000 7.0 7 :=
by
  sorry

end scientific_notation_for_70_million_l4_4635


namespace largest_prime_factor_4872_l4_4108

theorem largest_prime_factor_4872 : ∀ (f : ℕ → Prop), 
  (∀ p, prime p → f p ↔ p ∣ 4872) → 
  ∃ p, prime p ∧ f p ∧ ∀ q, prime q ∧ f q → q ≤ p :=
by {
  sorry
}

end largest_prime_factor_4872_l4_4108


namespace simplify_sqrt_l4_4026

theorem simplify_sqrt :
  (sqrt (3 * 5) * sqrt (5^3 * 3^3)) = 225 := by
  sorry

end simplify_sqrt_l4_4026


namespace contrapositive_statement_l4_4948

theorem contrapositive_statement (a b : ℝ) (h : a^2 < b → -sqrt(b) < a ∧ a < sqrt(b)) : 
    (a ≥ sqrt(b) ∨ a ≤ -sqrt(b)) → a^2 ≥ b :=
by
  sorry

end contrapositive_statement_l4_4948


namespace tan_identity_proof_l4_4823

theorem tan_identity_proof
  (α β : ℝ)
  (h₁ : Real.tan (α + β) = 3)
  (h₂ : Real.tan (α + π / 4) = -3) :
  Real.tan (β - π / 4) = -3 / 4 := 
sorry

end tan_identity_proof_l4_4823


namespace impossible_n_gon_l4_4478

theorem impossible_n_gon (n : ℕ) (l : Fin n → Line) (h : Odd n) :
  ¬ ∃ (A : Fin n → Point),
    (∀ i : Fin n, perpendicular (l i) (midpoint (segment (A i) (A (i + 1)))) ∨ 
    bisector (l i) (angle (A i) (A (i - 1)) (A (i + 1)))) := 
sorry

end impossible_n_gon_l4_4478


namespace number_is_fraction_l4_4327

theorem number_is_fraction (x : ℝ) : (0.30 * x = 0.25 * 40) → (x = 100 / 3) :=
begin
  intro h,
  sorry
end

end number_is_fraction_l4_4327


namespace find_a10_l4_4849

-- Define the arithmetic sequence with its common difference and initial term
axiom a_seq : ℕ → ℝ
axiom a1 : ℝ
axiom d : ℝ

-- Conditions
axiom a3 : a_seq 3 = a1 + 2 * d
axiom a5_a8 : a_seq 5 + a_seq 8 = 15

-- Theorem statement
theorem find_a10 : a_seq 10 = 13 :=
by sorry

end find_a10_l4_4849


namespace z_when_y_six_l4_4040

theorem z_when_y_six
    (k : ℝ)
    (h1 : ∀ y (z : ℝ), y^2 * Real.sqrt z = k)
    (h2 : ∃ (y : ℝ) (z : ℝ), y = 3 ∧ z = 4 ∧ y^2 * Real.sqrt z = k) :
  ∃ z : ℝ, y = 6 ∧ z = 1 / 4 := 
sorry

end z_when_y_six_l4_4040


namespace time_to_fill_tank_l4_4180

variable (X Y Z : ℝ)

def rate_XY (T : ℝ) : Prop := X + Y = T / 3
def rate_XZ (T : ℝ) : Prop := X + Z = T / 6
def rate_YZ (T : ℝ) : Prop := Y + Z = T / 4.5

theorem time_to_fill_tank (T : ℝ) (h1 : rate_XY T) (h2 : rate_XZ T) (h3 : rate_YZ T) :
  T / (X + Y + Z) = 3.27 :=
by
  sorry

end time_to_fill_tank_l4_4180


namespace mode_is_six_l4_4159

variable (weekly_reading_hours : List ℕ := [2, 2, 4, 4, 4, 6, 6, 6, 6, 8])

theorem mode_is_six :
  mode weekly_reading_hours = 6 :=
sorry

end mode_is_six_l4_4159


namespace correct_operation_l4_4995

theorem correct_operation : ∀ (x : ℝ), 4 * x^3 - 3 * x^3 = x^3 :=
by
  intro x
  calc
    4 * x^3 - 3 * x^3 = (4 - 3) * x^3 : by rw [sub_mul]
                   ... = 1 * x^3 : by norm_num
                   ... = x^3 : by rw one_mul

# Documentation for understanding mathematical reasoning

end correct_operation_l4_4995


namespace largest_possible_d_l4_4890

theorem largest_possible_d (a b c d : ℝ) (h1 : a + b + c + d = 10) (h2 : ab + ac + ad + bc + bd + cd = 20) : 
    d ≤ (5 + Real.sqrt 105) / 2 :=
by
  sorry

end largest_possible_d_l4_4890


namespace right_triangle_medians_eq_semiperimeter_l4_4917

theorem right_triangle_medians_eq_semiperimeter :
  ∀ (AC BC: ℕ) (H: AC = 6 ∧ BC = 4),
  let AB := Real.sqrt(AC^2 + BC^2),
      CM := AB / 2,
      BM := Real.sqrt(2 * AC^2 + 2 * BC^2 - BC^2) / 2,
      AM := Real.sqrt(2 * BC^2 + 2 * AC^2 - AC^2) / 2,
      s  := (AC + BC + AB) / 2
  in BM + AM = s :=
by
  intros AC BC H
  sorry

end right_triangle_medians_eq_semiperimeter_l4_4917


namespace dice_arithmetic_progression_l4_4991

theorem dice_arithmetic_progression :
  let valid_combinations := [
     (1, 1, 1), (1, 3, 2), (1, 5, 3), 
     (2, 4, 3), (2, 6, 4), (3, 3, 3),
     (3, 5, 4), (4, 6, 5), (5, 5, 5)
  ]
  (valid_combinations.length : ℚ) / (6^3 : ℚ) = 1 / 24 :=
  sorry

end dice_arithmetic_progression_l4_4991


namespace foma_gives_ierema_55_l4_4557

-- Defining the amounts that Foma, Ierema, and Yuliy have initially
variables (F E Y : ℝ)

-- Given conditions
def condition_1 : Prop := F - E = 140
def condition_2 : Prop := F - Y = 40
def condition_3 : Prop := Y = E + 70

-- Prove that Foma should give Ierema 55 gold coins
theorem foma_gives_ierema_55 (h1 : condition_1) (h2 : condition_2) (h3 : condition_3) : 
  (F - E) / 2 = 55 :=
by
  sorry

end foma_gives_ierema_55_l4_4557


namespace weighted_power_mean_inequality_l4_4632

variables {n : ℕ} {p q : ℝ} {a x : Finₓ n → ℝ}

theorem weighted_power_mean_inequality (hpq : 0 < p < q)
    (hx : ∀ i, 0 < x i)
    (ha : ∀ i, 0 ≤ a i)
    (ha_nonzero : ∃ i, a i ≠ 0) :
    ( (∑ i, a i * (x i) ^ p) / (∑ i, a i) ) ^ (1 / p) 
    ≤ ( (∑ i, a i * (x i) ^ q) / (∑ i, a i) ) ^ (1 / q) :=
begin
  sorry
end

end weighted_power_mean_inequality_l4_4632


namespace snail_maximum_distance_l4_4177

theorem snail_maximum_distance
  (journey_duration : ℕ)
  (total_scientists : ℕ)
  (observation_time : ℕ)
  (distance_per_hour : ℕ)
  (scientists_cover_whole_duration : ∀ t : ℕ, t < journey_duration -> ∃ (s : ℕ), s < total_scientists ∧ ∀ τ : ℕ, τ = observation_time -> τ.cover_interval t) :
  distance_per_hour * total_scientists ≤ 10 := 
sorry

end snail_maximum_distance_l4_4177


namespace find_complex_number_l4_4261

-- We will define the complex number z
variable {z : ℂ}

-- We state the proof problem
theorem find_complex_number (h : (1 + complex.i) * z = 2 * complex.i) : z = complex.i + 1 :=
sorry

end find_complex_number_l4_4261


namespace min_value_of_f_when_x_neg_l4_4231
noncomputable def f (x : ℝ) : ℝ := -x - 2/x

theorem min_value_of_f_when_x_neg : ∀ x : ℝ, x < 0 → f(x) ≥ 2 * Real.sqrt 2 :=
by
  intros x hx
  have : f x = -x - 2 / x := rfl
  sorry

end min_value_of_f_when_x_neg_l4_4231


namespace birds_joined_l4_4152

theorem birds_joined (B : ℕ) : 
  let total_birds := 2 + B in
  total_birds + 1 = 6 → B = 3 :=
by
  intro h
  simp [total_birds] at h
  omega

end birds_joined_l4_4152


namespace trigonometric_identity_l4_4747

theorem trigonometric_identity :
  sin (21 * (Math.pi / 180)) * cos (81 * (Math.pi / 180)) - 
  sin (69 * (Math.pi / 180)) * cos (9 * (Math.pi / 180)) = 
  - (Real.sqrt 3 / 2) := 
by 
  sorry

end trigonometric_identity_l4_4747


namespace abs_eq_cases_l4_4130

theorem abs_eq_cases (a b : ℝ) : (|a| = |b|) → (a = b ∨ a = -b) :=
sorry

end abs_eq_cases_l4_4130


namespace scientific_notation_of_258000000_l4_4730

theorem scientific_notation_of_258000000 :
  258000000 = 2.58 * 10^8 :=
sorry

end scientific_notation_of_258000000_l4_4730


namespace sum_of_squares_ge_mean_square_l4_4904

variable {n : ℕ} (a : Fin n → ℝ)

theorem sum_of_squares_ge_mean_square :
  ∑ i, (a i)^2 ≥ (∑ i, a i)^2 / n := by
sorry

end sum_of_squares_ge_mean_square_l4_4904


namespace equalize_foma_ierema_l4_4576

theorem equalize_foma_ierema (F E Y : ℕ) 
  (h1 : F - 70 = E + 70) 
  (h2 : F - 40 = Y) 
  (h3 : Y = E + 70) 
  : ∃ x : ℕ, x = 55 ∧ F - x = E + x :=
by
  use 55
  sorry

end equalize_foma_ierema_l4_4576


namespace find_m_find_monotonic_intervals_l4_4290

noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  m * (1 + Real.sin x) + Real.cos x

theorem find_m (m : ℝ) (h : f m (Real.pi / 2) = 2) : m = 1 :=
  sorry

theorem find_monotonic_intervals :
  let f (x : ℝ) := 1 + Real.sin x + Real.cos x in
  (∀ k : ℤ, ∀ x, - (3 / 4) * Real.pi + 2 * k * Real.pi ≤ x ∧ x ≤ (1 / 4) * Real.pi + 2 * k * Real.pi ↔
              ∀ x, (1 / 4) * Real.pi + 2 * k * Real.pi ≤ x ∧ x ≤ (5 / 4) * Real.pi + 2 * k * Real.pi ↔
             is_monotonic f) :=
  sorry

end find_m_find_monotonic_intervals_l4_4290


namespace strange_seq_empty_iff_nilpotent_l4_4903

variable (n : ℕ)
variable (A : Matrix (Fin n) (Fin n) ℕ) -- Assuming entries are 0 or 1
variable (S : Set (Fin n)) := {i | i.val < n}
variable (strange_seq : ℕ → Fin n)

-- Define conditions for a strange sequence
def is_strange_seq (x : ℕ → Fin n) : Prop :=
  ∀ k : ℕ, k < n - 1 → A x[k] x[k+1] = 1

-- Define the set of strange sequences
def strange_set_empty : Prop :=
  ¬(∃ x : ℕ → Fin n, is_strange_seq A x)

-- Define nilpotency of a matrix
def is_nilpotent (A : Matrix (Fin n) (Fin n) ℕ) : Prop :=
  ∃ m : ℕ, m > 0 ∧ A ^ m = 0

theorem strange_seq_empty_iff_nilpotent :
  strange_set_empty n A ↔ is_nilpotent A := by
  sorry

end strange_seq_empty_iff_nilpotent_l4_4903


namespace selection_ways_l4_4093

/-- The number of ways to select 4 athletes from 5 male and 4 female athletes,
ensuring the selection includes both genders and at least one of athlete A or athlete B,
is 86. -/
theorem selection_ways : 
  let num_males := 5
  let num_females := 4
  let total_athletes := 9
  let select_count := 4
  let m_ways := Nat.choose 5 4
  let f_ways := Nat.choose 4 4
  let total_ways := Nat.choose 9 4
  let both_ways := total_ways - m_ways - f_ways
  let ex_both_ways := (Nat.choose (total_athletes - 2) select_count) - 1
  in both_ways - ex_both_ways = 86 :=
by
  let num_males := 5
  let num_females := 4
  let total_athletes := 9
  let select_count := 4
  let m_ways := Nat.choose 5 4
  let f_ways := Nat.choose 4 4
  let total_ways := Nat.choose 9 4
  let both_ways := total_ways - m_ways - f_ways
  let ex_both_ways := (Nat.choose (total_athletes - 2) select_count) - 1
  exact (both_ways - ex_both_ways = 86)
  sorry

end selection_ways_l4_4093


namespace cubic_expression_value_l4_4826

theorem cubic_expression_value (x : ℝ) (h : x^2 + 3 * x - 1 = 0) : x^3 + 5 * x^2 + 5 * x + 18 = 20 :=
by
  sorry

end cubic_expression_value_l4_4826


namespace reg_17gon_symmetry_l4_4174

theorem reg_17gon_symmetry : 
  let L := 17,
      R := 360 / 17 in
  L + R = 38 :=
by
  let L := 17
  let R := 21
  sorry

end reg_17gon_symmetry_l4_4174


namespace safe_lock_problem_l4_4686

-- Definitions of the conditions
def num_people := 9
def min_people_needed := 6

-- Binomial Coefficient Function
def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Statement of the problem and correct answer
theorem safe_lock_problem :
  (binomial_coefficient num_people (num_people - min_people_needed + 1) = 126) ∧
  (∀ lock, lock ∈ Finset.range 126 → (Finset.card (Finset.powersetLen 4 (Finset.range num_people)) = 4)) :=
by
  sorry

end safe_lock_problem_l4_4686


namespace equalize_foma_ierema_l4_4581

variables 
  (F E Y : ℕ)
  (h1 : F - 70 = E + 70)
  (h2 : F - 40 = Y)

def foma_should_give_ierema : ℕ := (F - E) / 2

theorem equalize_foma_ierema (F E Y : ℕ) (h1 : F - 70 = E + 70) (h2 : F - 40 = Y) :
  foma_should_give_ierema F E = 55 := 
by
  sorry

end equalize_foma_ierema_l4_4581


namespace sum_of_four_natural_numbers_smallest_of_four_natural_numbers_l4_4010

theorem sum_of_four_natural_numbers (a b c d : ℕ) 
  (h : {a + b, b + c, c + d, d + a, a + c, b + d}.count 23 = 3)
  (h' : {a + b, b + c, c + d, d + a, a + c, b + d}.count 34 = 3) : 
  a + b + c + d = 57 :=
sorry

theorem smallest_of_four_natural_numbers (a b c d : ℕ) 
  (h : {a + b, b + c, c + d, d + a, a + c, b + d}.count 23 = 3)
  (h' : {a + b, b + c, c + d, d + a, a + c, b + d}.count 34 = 3) : 
  min a (min b (min c d)) = 6 :=
sorry

end sum_of_four_natural_numbers_smallest_of_four_natural_numbers_l4_4010


namespace equal_share_each_shopper_l4_4864

theorem equal_share_each_shopper 
  (amount_giselle : ℕ)
  (amount_isabella : ℕ)
  (amount_sam : ℕ)
  (H1 : amount_isabella = amount_sam + 45)
  (H2 : amount_isabella = amount_giselle + 15)
  (H3 : amount_giselle = 120) : 
  (amount_isabella + amount_sam + amount_giselle) / 3 = 115 :=
by
  -- The proof is omitted.
  sorry

end equal_share_each_shopper_l4_4864


namespace third_group_count_l4_4474

-- Defining the total number of students
def total_students : ℕ := 45

-- Defining the proportion of students in the first group
def first_group_fraction : ℚ := 1 / 3

-- Defining the proportion of students in the second group
def second_group_fraction : ℚ := 2 / 5

-- Calculating the number of students in the first group
def first_group_students : ℕ := (first_group_fraction * total_students).natAbs

-- Calculating the number of students in the second group
def second_group_students : ℕ := (second_group_fraction * total_students).natAbs

-- Calculating the number of remaining students after the first group
def remaining_after_first_group : ℕ := total_students - first_group_students

-- Calculating the number of remaining students after the second group
def third_group_students : ℕ := remaining_after_first_group - second_group_students

-- The theorem stating that the number of students in the third group is 12
theorem third_group_count : third_group_students = 12 := by
  -- Sorry to skip the proof
  sorry

end third_group_count_l4_4474


namespace num_ways_to_choose_starting_lineup_l4_4170

-- Define conditions as Lean definitions
def team_size : ℕ := 12
def outfield_players : ℕ := 4

-- Define the function to compute the number of ways to choose the starting lineup
def choose_starting_lineup (team_size : ℕ) (outfield_players : ℕ) : ℕ :=
  team_size * Nat.choose (team_size - 1) outfield_players

-- The theorem to prove that the number of ways to choose the lineup is 3960
theorem num_ways_to_choose_starting_lineup : choose_starting_lineup team_size outfield_players = 3960 :=
  sorry

end num_ways_to_choose_starting_lineup_l4_4170


namespace foma_should_give_ierema_l4_4597

variables (F E Y : ℕ)

-- Given conditions in Lean definitions
def condition1 := F - 70 = E + 70
def condition2 := F - 40 = Y

-- Final statement to be proven
theorem foma_should_give_ierema : 
  condition1 → condition2 → F - E = 110 → F - E = 110 / 2 + 55 :=
by
  intros h1 h2 h3
  sorry

end foma_should_give_ierema_l4_4597


namespace volume_of_soil_l4_4621

theorem volume_of_soil (length width height : ℕ) (h_length : length = 20) (h_width : width = 10) (h_height : height = 8) : 
  length * width * height = 1600 :=
by {
  -- start with the assumptions
  rw [h_length, h_width, h_height],
  -- calculate the product
  norm_num,
  sorry
}

end volume_of_soil_l4_4621


namespace find_m_l4_4972

open Matrix

def vectors := (λ (m : ℝ), ({⟨3, ⟨4, 5⟩⟩, ⟨2, ⟨m, 3⟩⟩, ⟨2, ⟨3, m⟩⟩} : Matrix 3 3 ℝ))

def volume (m : ℝ) := abs (det (vectors m))

theorem find_m (m : ℝ) (h : volume m = 20) (hm : m > 0) : 
  m = 3 + (2 * Real.sqrt 15) / 3 :=
sorry -- Proof not included

end find_m_l4_4972


namespace laborers_employed_l4_4684

theorem laborers_employed 
    (H L : ℕ) 
    (h1 : H + L = 35) 
    (h2 : 140 * H + 90 * L = 3950) : 
    L = 19 :=
by
  sorry

end laborers_employed_l4_4684


namespace remainder_is_zero_l4_4745

theorem remainder_is_zero :
  (86 * 87 * 88 * 89 * 90 * 91 * 92) % 7 = 0 := 
by 
  sorry

end remainder_is_zero_l4_4745


namespace cylinder_height_l4_4069

theorem cylinder_height (h : ℝ)
  (circumference : ℝ)
  (rectangle_diagonal : ℝ)
  (C_eq : circumference = 12)
  (d_eq : rectangle_diagonal = 20) :
  h = 16 :=
by
  -- We derive the result based on the given conditions and calculations
  sorry -- Skipping the proof part

end cylinder_height_l4_4069


namespace red_balls_count_l4_4352

theorem red_balls_count (total_balls : ℕ) (freq_red_ball : ℚ) (h1 : total_balls = 20) (h2 : freq_red_ball = 0.25) : ∃ (x : ℕ), x = 5 ∧ (x / total_balls.toQ = freq_red_ball) :=
begin
  sorry
end

end red_balls_count_l4_4352


namespace limit_arcsin_3x_over_sqrt_2x_minus_sqrt_2_l4_4627

/-- Definition of the limit problem to be proved -/
def limit_problem : Prop :=
  (∀ (x : ℝ), x ≠ 0 → 
  (∃ (d : ℝ), 0 < d ∧ 
  (∀ y (h : |y| < d), abs ((arcsin (3 * y)) / (sqrt (2 + y) - sqrt 2) - 6 * sqrt 2) < x)))

theorem limit_arcsin_3x_over_sqrt_2x_minus_sqrt_2 :
  limit_problem :=
by
  sorry

end limit_arcsin_3x_over_sqrt_2x_minus_sqrt_2_l4_4627


namespace equation_solution_l4_4027

theorem equation_solution (x : ℚ) (h₁ : (5 * x^2 + 4 * x + 2) / (x + 2) = 5 * x - 3) : x = 8 / 3 :=
by
  sorry

end equation_solution_l4_4027


namespace foma_should_give_ierema_55_coins_l4_4562

theorem foma_should_give_ierema_55_coins (F E Y : ℤ)
  (h1 : F - 70 = E + 70)
  (h2 : F - 40 = Y)
  (h3 : E + 70 = Y) :
  F - E = 110 → F - E = 55 :=
by
  sorry

end foma_should_give_ierema_55_coins_l4_4562


namespace nissa_grooming_time_correct_l4_4871

def clipping_time_per_claw : ℕ := 10
def cleaning_time_per_ear : ℕ := 90
def shampooing_time_minutes : ℕ := 5

def claws_per_foot : ℕ := 4
def feet_count : ℕ := 4
def ear_count : ℕ := 2

noncomputable def total_grooming_time_in_seconds : ℕ := 
  (clipping_time_per_claw * claws_per_foot * feet_count) + 
  (cleaning_time_per_ear * ear_count) + 
  (shampooing_time_minutes * 60) -- converting minutes to seconds

theorem nissa_grooming_time_correct :
  total_grooming_time_in_seconds = 640 := by
  sorry

end nissa_grooming_time_correct_l4_4871


namespace problem_statement_l4_4893

noncomputable def max_value_d (a b c d : ℝ) : Prop :=
a + b + c + d = 10 ∧
(ab + ac + ad + bc + bd + cd = 20) ∧
∀ x, (a + b + c + x = 10 ∧ ab + ac + ad + bc + bd + cd = 20) → x ≤ 5 + Real.sqrt 105 / 2

theorem problem_statement (a b c d : ℝ) :
  max_value_d a b c d → d = (5 + Real.sqrt 105) / 2 :=
sorry

end problem_statement_l4_4893


namespace increasing_function_condition_l4_4806

noncomputable theory
open Real

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a * x + 1 / x

def f' (a : ℝ) (x : ℝ) : ℝ := 2 * x + a - 1 / x^2

def g (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 + a * x^2 - 1

def g' (a : ℝ) (x : ℝ) : ℝ := 6 * x^2 + 2 * a * x

theorem increasing_function_condition (a : ℝ) :
  (∀ x : ℝ, 1 / 2 < x → 0 ≤ f' a x) ↔ (3 ≤ a) := 
by
  sorry

end increasing_function_condition_l4_4806


namespace triangle_ABC_proof_l4_4439

variables {A B C M N K : Type} [inhabited A] [inhabited B] [inhabited C]
variables [between A B C] [within_segment A M] [within_segment M C] [on_segment A M]
variables [angle_eq (M B N) (C B M)] [right_angle (B M K)]
variables [midpoint_of_segment A C M]
variables [extension_of_segment B N K beyond N]

theorem triangle_ABC_proof (hM : midpoint_of_segment A C M)
    (hN : on_segment A M N)
    (hAngle : angle_eq (M B N) (C B M))
    (hPerp : right_angle (B M K))
    (hExt : extension_of_segment B N K beyond N) :
    BC = AK + BK :=
by
  sorry

end triangle_ABC_proof_l4_4439


namespace triangle_property_l4_4443

variables {ABC : Type*} [inner_product_space ℝ ABC]
variables (A B C M K N : ABC)

noncomputable def midpoint (A C : ABC) (M : ABC) : Prop := dist A M = dist M C

noncomputable def angle_equality (B M N : ABC) : Prop := ∠M B N = ∠C B M

noncomputable def right_angle (B M K : ABC) : Prop := ∠B M K = π / 2

theorem triangle_property (h1 : midpoint A C M) (h2 : angle_equality B M N) (h3 : right_angle B M K) : dist B C = dist A K + dist B K :=
sorry

end triangle_property_l4_4443


namespace foma_should_give_ierema_l4_4594

variables (F E Y : ℕ)

-- Given conditions in Lean definitions
def condition1 := F - 70 = E + 70
def condition2 := F - 40 = Y

-- Final statement to be proven
theorem foma_should_give_ierema : 
  condition1 → condition2 → F - E = 110 → F - E = 110 / 2 + 55 :=
by
  intros h1 h2 h3
  sorry

end foma_should_give_ierema_l4_4594


namespace triangle_relation_l4_4456

-- Define the given conditions and the hypothesis
variables {A B C M N K : Type*}
variables (triangle : A → B → C → Type*)
variables [midpoint M A C] [on_segment N A M] [extension K N B]
variables [right_angle (angle BMK)] [angle_eq (angle MBN) (angle CBM)]

-- Define segments and lengths
variables [segment BC] [segment AK] [segment BK]

-- State the theorem
theorem triangle_relation 
  (h1: midpoint M A C) 
  (h2: on_segment N A M) 
  (h3: angle_eq (angle MBN) (angle CBM))
  (h4: right_angle (angle BMK)) : 
  length BC = length AK + length BK := by
  sorry

end triangle_relation_l4_4456


namespace pages_needed_l4_4923

theorem pages_needed (packs : ℕ) (cards_per_pack : ℕ) (cards_per_page : ℕ) (total_packs : packs = 60) (cards_in_pack : cards_per_pack = 7) (capacity_per_page : cards_per_page = 10) : (packs * cards_per_pack) / cards_per_page = 42 := 
by
  -- Utilize the conditions
  have H1 : packs = 60 := total_packs
  have H2 : cards_per_pack = 7 := cards_in_pack
  have H3 : cards_per_page = 10 := capacity_per_page
  -- Use these to simplify and prove the target expression 
  sorry

end pages_needed_l4_4923


namespace nissa_grooming_time_correct_l4_4870

def clipping_time_per_claw : ℕ := 10
def cleaning_time_per_ear : ℕ := 90
def shampooing_time_minutes : ℕ := 5

def claws_per_foot : ℕ := 4
def feet_count : ℕ := 4
def ear_count : ℕ := 2

noncomputable def total_grooming_time_in_seconds : ℕ := 
  (clipping_time_per_claw * claws_per_foot * feet_count) + 
  (cleaning_time_per_ear * ear_count) + 
  (shampooing_time_minutes * 60) -- converting minutes to seconds

theorem nissa_grooming_time_correct :
  total_grooming_time_in_seconds = 640 := by
  sorry

end nissa_grooming_time_correct_l4_4870


namespace parallel_vectors_x_values_l4_4250

theorem parallel_vectors_x_values (x : ℝ) :
  let a := (2, -3 : ℝ)
  let b := (4, x^2 - 5 * x : ℝ)
  (2 ≠ 0 ∧ -3 ≠ 0 ∧ (x^2 - 5 * x)_2 ≠ 0) →
  (2 / 4 = -3 / (x^2 - 5 * x)) →
  (x = 2 ∨ x = 3) :=
by
  intros
  have h1 : 2 ≠ 0 := by norm_num
  have h2 : -3 ≠ 0 := by norm_num
  have h3 : (x^2 - 5 * x) ≠ 0 := 
  sorry
  have collinear: (2 / 4 = -3 / (x^2 - 5 * x)) :=
  sorry
  have x_values: x = 2 ∨ x = 3 :=
  sorry
  exact x_values

end parallel_vectors_x_values_l4_4250


namespace work_completion_l4_4133

theorem work_completion (a b : ℝ) 
  (h1 : a + b = 6) 
  (h2 : a = 10) : 
  a + b = 6 :=
by sorry

end work_completion_l4_4133


namespace find_number_l4_4323

theorem find_number (y : ℝ) (h : (30 / 100) * y = (25 / 100) * 40) : y = 100 / 3 :=
by
  sorry

end find_number_l4_4323


namespace foma_gives_ierema_55_l4_4556

-- Defining the amounts that Foma, Ierema, and Yuliy have initially
variables (F E Y : ℝ)

-- Given conditions
def condition_1 : Prop := F - E = 140
def condition_2 : Prop := F - Y = 40
def condition_3 : Prop := Y = E + 70

-- Prove that Foma should give Ierema 55 gold coins
theorem foma_gives_ierema_55 (h1 : condition_1) (h2 : condition_2) (h3 : condition_3) : 
  (F - E) / 2 = 55 :=
by
  sorry

end foma_gives_ierema_55_l4_4556


namespace jame_annual_earnings_difference_l4_4367

-- Define conditions
def new_hourly_wage := 20
def new_hours_per_week := 40
def old_hourly_wage := 16
def old_hours_per_week := 25
def weeks_per_year := 52

-- Define annual earnings calculations
def annual_earnings_old (hourly_wage : ℕ) (hours_per_week : ℕ) (weeks_per_year : ℕ) : ℕ :=
  hourly_wage * hours_per_week * weeks_per_year

def annual_earnings_new (hourly_wage : ℕ) (hours_per_week : ℕ) (weeks_per_year : ℕ) : ℕ :=
  hourly_wage * hours_per_week * weeks_per_year

-- Problem statement to prove
theorem jame_annual_earnings_difference :
  annual_earnings_new new_hourly_wage new_hours_per_week weeks_per_year -
  annual_earnings_old old_hourly_wage old_hours_per_week weeks_per_year = 20800 := by
  sorry

end jame_annual_earnings_difference_l4_4367


namespace evolute_of_cycloid_is_cycloid_l4_4488

noncomputable theory

/-- Proof that the evolute of a cycloid is another cycloid obtained via a parallel translation -/
theorem evolute_of_cycloid_is_cycloid :
  ∀ (Ox : ℝ), ∀ (O X P Q O₁ : ℝ × ℝ),
  ∃ (S₁ S₁' : ℝ), 
  (radius S₁ = 1) ∧
  (rolling S₁) ∧
  (fixed_point_initial (0, 0)) ∧
  (fixed_point_at_time (X)) ∧
  (axis_touch_point S₁ Ox P) ∧
  (point (O₁) = (π, 0)) →
  evolute_cycloid S₁ = translated_cycloid S₁' :=
begin
  sorry
end

end evolute_of_cycloid_is_cycloid_l4_4488


namespace cos_phi_eq_2_sqrt5_over_5_l4_4634

noncomputable def cos_angle_PXQ : ℝ :=
  let P := (0, 0) in
  let Q := (0, 4) in
  let R := (4, 4) in
  let S := (4, 0) in
  let X := (2, 4) in
  let Y := (4, 2) in
  let PX := real.sqrt ((X.1 - P.1)^2 + (X.2 - P.2)^2) in
  let PQ := 4 in
  let XQ := 2 in
  let numerator := PX^2 + PQ^2 - XQ^2 in
  let denominator := 2 * PX * PQ in
  numerator / denominator

theorem cos_phi_eq_2_sqrt5_over_5 : cos_angle_PXQ = 2 * real.sqrt 5 / 5 := 
  sorry

end cos_phi_eq_2_sqrt5_over_5_l4_4634


namespace problem_inequality_l4_4801

-- Definitions and conditions
noncomputable def f (x : ℝ) : ℝ := x * Real.log x
noncomputable def g (x : ℝ) (k : ℝ) : ℝ := f x + f (k - x)

-- The Lean proof problem
theorem problem_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  f a + (a + b) * Real.log 2 ≥ f (a + b) - f b := sorry

end problem_inequality_l4_4801


namespace stream_speed_is_3_l4_4154

-- definitions for conditions
def boat_speed_downstream := 100 / 8 -- in km/h
def boat_speed_upstream := 75 / 15   -- in km/h

-- definition of speed in still water
def B : ℝ := 8.75

-- definition of effective speeds including stream
def downstream_speed (B S : ℝ) := B + S
def upstream_speed (B S : ℝ) := B - S

-- Theorem statement: speed of the stream is 3.75 km/h
theorem stream_speed_is_3.75 : 
  ∃ S : ℝ, downstream_speed B S = boat_speed_downstream ∧ upstream_speed B S = boat_speed_upstream ∧ S = 3.75 :=
by 
  sorry

end stream_speed_is_3_l4_4154


namespace tangent_line_exists_unique_l4_4789

theorem tangent_line_exists_unique {a : ℝ} 
  (P : ℝ × ℝ := (-1, -2)) 
  (C : ℝ → ℝ → Prop := λ x y, x^2 + y^2 + a * x + 2 * a * y + 2 * a^2 - a - 1 = 0) :
  (∃! (l : ℝ → ℝ → Prop), ∀ x y, l x y ↔ (y = -2 * x + (2*a - 2))) → a = 1 :=
begin
  sorry
end

end tangent_line_exists_unique_l4_4789


namespace find_cost_price_l4_4670

theorem find_cost_price
  (cost_price : ℝ)
  (increase_rate : ℝ := 0.2)
  (decrease_rate : ℝ := 0.1)
  (profit : ℝ := 8):
  (1 + increase_rate) * cost_price * (1 - decrease_rate) - cost_price = profit → 
  cost_price = 100 := 
by 
  sorry

end find_cost_price_l4_4670


namespace sequence_2009_l4_4269

noncomputable def sequence (n : ℕ) : ℚ :=
  if n = 1 then 1 / 2
  else (1 : ℚ) / (1 - sequence (n - 1))

theorem sequence_2009 : sequence 2009 = 2 :=
sorry

end sequence_2009_l4_4269


namespace angle_between_vectors_l4_4281

variables {a b : ℝ^3}
variables (θ : ℝ)

// Given conditions
def a_norm : ℝ := 1
def b_norm : ℝ := 2
def a_minus_2b_norm : ℝ := real.sqrt 13

-- Main theorem statement
theorem angle_between_vectors (h1 : ∥a∥ = a_norm) (h2 : ∥b∥ = b_norm) (h3 : ∥a - 2 • b∥ = a_minus_2b_norm) : θ = real.arccos (1/2) :=
sorry

end angle_between_vectors_l4_4281


namespace triangle_equality_lemma_l4_4403

open EuclideanGeometry

theorem triangle_equality_lemma 
  (A B C M N K: Point)
  (hM_midpoint: midpoint M A C)
  (hN_on_AM: lies_on N A M)
  (h_angle_eq: ∠ M B N = ∠ C B M)
  (hK_right: ∠ B M K = 90) :
  dist B C = dist A K + dist B K :=
by
  sorry

end triangle_equality_lemma_l4_4403


namespace cevian_concurrency_l4_4363

noncomputable theory
open_locale classical

variables {α : Type*} [nondiscrete_normed_field α] {A B C P L M N: EuclideanSpace α (fin 2)}

def is_triangle (A B C : EuclideanSpace α (fin 2)) : Prop := 
  ¬ collinear ({A, B, C} : set (EuclideanSpace α (fin 2)))

def angle_eq (u v w : EuclideanSpace α (fin 2)) (θ : ℝ) : Prop :=
  real.angle.cos (real.angle u v) = real.angle.cos θ

def concurrent (u v w : EuclideanSpace α (fin 2)) : Prop :=
  ∃ t1 t2 s1 s2 r1 r2, t1 * u + t2 * v = s1 * w + s2 * u ∧ r1 * w + r2 * v = 0

theorem cevian_concurrency
  (hABC : is_triangle A B C)
  (hP : P ∈ interior (triangle A B C))
  (hAL : angle_eq A L P (angle_eq C P A))
  (hBM : angle_eq B M P (angle_eq A P B))
  (hCN : angle_eq C N P (angle_eq B P C)) :
  concurrent (line A L) (line B M) (line C N) := 
sorry

end cevian_concurrency_l4_4363


namespace fomagive_55_l4_4590

variables (F E Y : ℕ)

-- Conditions
def condition1 := F - E = 140
def condition2 := F - Y = 40
def condition3 := Y = E + 70

-- Proposition: Foma should give 55 coins to Ierema
theorem fomagive_55 (h1 : condition1 F E) (h2 : condition2 F Y) (h3 : condition3 E Y) : ∃ G, G = 55 :=
by
  have := h1,
  have := h2,
  have := h3,
  sorry

end fomagive_55_l4_4590


namespace negation_of_exists_l4_4524

theorem negation_of_exists :
  ¬ (∃ x₀ : ℝ, sin x₀ + 2 * x₀^2 > cos x₀) ↔ ∀ x : ℝ, sin x + 2 * x^2 ≤ cos x :=
by
  sorry

end negation_of_exists_l4_4524


namespace women_fraction_l4_4843

/-- In a room with 100 people, 1/4 of whom are married, the maximum number of unmarried women is 40.
    We need to prove that the fraction of women in the room is 2/5. -/
theorem women_fraction (total_people : ℕ) (married_fraction : ℚ) (unmarried_women : ℕ) (W : ℚ) 
  (h1 : total_people = 100) 
  (h2 : married_fraction = 1 / 4) 
  (h3 : unmarried_women = 40) 
  (hW : W = 2 / 5) : 
  W = 2 / 5 := 
by
  sorry

end women_fraction_l4_4843


namespace mean_equality_l4_4963

theorem mean_equality (z : ℚ) :
  (8 + 12 + 24) / 3 = (16 + z) / 2 ↔ z = 40 / 3 :=
by
  sorry

end mean_equality_l4_4963


namespace exponents_subtraction_l4_4105

theorem exponents_subtraction : (2 ^ (-3) ^ 0) - (2 ^ 1 ^ 4) = -15 := by
  sorry

end exponents_subtraction_l4_4105


namespace find_f_2015_l4_4518

noncomputable def f (x : ℝ) : ℝ := 
  if h : 0 ≤ x ∧ x < 2 then 3^x - 1 
  else -f (x - 2)  -- we use the recursive definition from f(x+2) = -f(x)

theorem find_f_2015 : f 2015 = -2 := by
  sorry

end find_f_2015_l4_4518


namespace volume_region_between_spheres_l4_4550

theorem volume_region_between_spheres 
    (r1 r2 : ℝ) 
    (h1 : r1 = 4) 
    (h2 : r2 = 7) 
    : 
    ( (4/3) * π * r2^3 - (4/3) * π * r1^3 ) = 372 * π := 
    sorry

end volume_region_between_spheres_l4_4550


namespace arc_length_of_sector_l4_4344

theorem arc_length_of_sector (r : ℝ) (θ : ℝ) (h_r : r = 2) (h_θ : θ = π / 3) :
  l = r * θ := by
  sorry

end arc_length_of_sector_l4_4344


namespace triangle_height_correct_l4_4603

-- Definitions based on conditions in a)
structure Triangle :=
  (A B C D E F : ℝ × ℝ)
  (midpoint_D : D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (line_angle : ∃ (θ : ℝ), θ = 30 ∧ 
                (E.2 - D.2) / (E.1 - D.1) = Real.tan (θ * Real.pi / 180))
  (ED_length : Real.dist D E = 6)
  (FD_length : Real.dist D F = 4)

noncomputable def height_of_triangle (T : Triangle) : ℝ :=
  let h := Real.sqrt (36 + 3 * 16)
  h

-- The theorem that states the correct answer
theorem triangle_height_correct (T : Triangle) : height_of_triangle T = 12 :=
by
  sorry

end triangle_height_correct_l4_4603


namespace triangle_inequality_inequality_l4_4462

-- Definitions:
variable {a b c : ℝ}

-- conditions: a, b, c are the sides of a triangle
-- This means they must satisfy the triangle inequalities:
axiom triangle_inequalities :
  a + b > c ∧ a + c > b ∧ b + c > a

-- Lean statement to prove:
theorem triangle_inequality_inequality (h : triangle_inequalities) :
  2 * (a * b + a * c + b * c) > a^2 + b^2 + c^2 :=
by
  sorry
 
end triangle_inequality_inequality_l4_4462


namespace find_ellipse_eq_l4_4285

noncomputable def ellipse_eq (x y : ℝ) (m n : ℝ) : Prop :=
  x^2 / m^2 + y^2 / n^2 = 1

def parabola_eq (x y : ℝ) : Prop :=
  y^2 = 8 * x

def focus_parabola : ℝ × ℝ :=
  (2, 0)

def eccentricity_ellipse (m c : ℝ) : ℝ :=
  c / m

theorem find_ellipse_eq (m n : ℝ) (h_focus : sqrt (m^2 - n^2) = 2) (h_eccentricity : 2 * m = 4) :
  ellipse_eq x y 4 (2 * sqrt 3) :=
sorry

end find_ellipse_eq_l4_4285


namespace trip_time_correct_l4_4477

-- Define the conditions

-- Lin travels 100 miles on the highway
def highway_distance : ℝ := 100

-- Lin travels 20 miles on a forest trail
def forest_distance : ℝ := 20

-- She drove four times as fast on the highway as on the forest trail
def speed_ratio : ℝ := 4

-- Lin spent 40 minutes driving on the forest trail
def forest_time : ℝ := 40

-- Define the speed on the forest trail
def forest_speed : ℝ := forest_distance / forest_time

-- Define the speed on the highway
def highway_speed : ℝ := speed_ratio * forest_speed

-- Calculate the time spent on the highway
def highway_time : ℝ := highway_distance / highway_speed

-- Calculate the total time taken
def total_time : ℝ := forest_time + highway_time

-- The theorem to prove
theorem trip_time_correct :
  total_time = 90 := by
  sorry

end trip_time_correct_l4_4477


namespace triangle_angle_sum_l4_4422

variable (ABC : Type) [triangle ABC]

-- Definitions from conditions in a)
variable (A B C M N K : point ABC)
variable (M_midpoint : midpoint M A C)
variable (N_on_AM : lies_on N A M)
variable (angle_eq1 : ∠ M B N = ∠ C B M)
variable (K_ext_BN : lies_on_ext K B N)
variable (angle_eq2 : ∠ B M K = 90)

-- Goal to prove
theorem triangle_angle_sum (HC : M_midpoint) (HN : N_on_AM) (HB : angle_eq1) (HK : K_ext_BN) (HA : angle_eq2) :
  segment_length B C = segment_length A K + segment_length B K := sorry

end triangle_angle_sum_l4_4422


namespace gold_balloons_count_l4_4547

-- Definitions of the conditions
def num_gold_balloons : ℕ
def num_silver_balloons : ℕ := 2 * num_gold_balloons
def total_balloons : ℕ := num_gold_balloons + num_silver_balloons + 150

-- The proof statement
theorem gold_balloons_count (num_gold_balloons : ℕ) (h1 : total_balloons = 573) : num_gold_balloons = 141 :=
sorry

end gold_balloons_count_l4_4547


namespace factorize_x_squared_minus_25_l4_4222

theorem factorize_x_squared_minus_25 : ∀ (x : ℝ), (x^2 - 25) = (x + 5) * (x - 5) :=
by
  intros x
  sorry

end factorize_x_squared_minus_25_l4_4222


namespace T_9_eq_274_l4_4394

def T (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if n = 2 then 3
  else if n = 3 then 7
  else if n = 4 then 13
  else T (n-1) + T (n-2) + T (n-3) + T (n-4)

theorem T_9_eq_274 : T 9 = 274 :=
by {
  have h1 : T 5 = 24, from calc
    T 5 = T 4 + T 3 + T 2 + T 1 : by rfl
    ... = 13 + 7 + 3 + 1 : by rfl
    ... = 24 : by rfl,
  have h2 : T 6 = 44, from calc
    T 6 = T 5 + T 4 + T 3 + T 2 : by rfl
    ... = 24 + 13 + 7 + 3 : by rfl
    ... = 44 : by rfl,
  have h3 : T 7 = 81, from calc
    T 7 = T 6 + T 5 + T 4 + T 3 : by rfl
    ... = 44 + 24 + 13 + 7 : by rfl
    ... = 81 : by rfl,
  have h4 : T 8 = 149, from calc
    T 8 = T 7 + T 6 + T 5 + T 4 : by rfl
    ... = 81 + 44 + 24 + 13 : by rfl
    ... = 149 : by rfl,
  show T 9 = 274, from calc
    T 9 = T 8 + T 7 + T 6 + T 5 : by rfl
    ... = 149 + 81 + 44 + 24 : by rfl
    ... = 298 : by rfl
}

end T_9_eq_274_l4_4394


namespace general_term_a_general_term_b_smallest_n_l4_4469

noncomputable def a₁ : ℕ := 3
noncomputable def b₁ : ℕ := 1

noncomputable def a : ℕ → ℕ
| 1       := a₁
| (n + 1) := a n + b n + Nat.sqrt ((a n)^2 - (a n) * (b n) + (b n)^2)

noncomputable def b : ℕ → ℕ
| 1       := b₁
| (n + 1) := a n + b n - Nat.sqrt ((a n)^2 - (a n) * (b n) + (b n)^2)

noncomputable def S (n : ℕ) : ℕ :=
(∑ i in Finset.range n, a i)

noncomputable def T (n : ℕ) : ℕ :=
(∑ i in Finset.range n, b i)

noncomputable def find_n : ℕ :=
  Nat.find (λ n => (∑ k in Finset.range n, S k + T k) > 2017)

theorem general_term_a : ∀ n, a (n + 1) = 3 * 2^n := sorry

theorem general_term_b : ∀ n, b (n + 1) = 2^n := sorry

theorem smallest_n : find_n = 9 := sorry

end general_term_a_general_term_b_smallest_n_l4_4469


namespace distance_to_other_focus_l4_4271

theorem distance_to_other_focus 
  (x y : ℝ) 
  (h1 : x ^ 2 / 2 + y ^ 2 = 1)
  (h2 : ∀ (c : ℝ), c = 1 → (x - c) ^ 2 + y ^ 2 = 1) :
  ∀ (c : ℝ), c = 1 → ((x + c) ^ 2 + y ^ 2 = 2 * sqrt 2 - 1) :=
begin
  sorry
end

end distance_to_other_focus_l4_4271


namespace problemI_problemII_l4_4683

/-- 
Define the probabilities for answering questions A, B, C, and D correctly.
--/
def PA := 1 / 2 
def PB := 1 / 3
def PC := 1 / 2
def PD := 1 / 2

/-- 
Prove the first statement: 
The probability of the candidate being hired 
(i.e., answering A and B correctly and at least one of C or D correctly) is 1/8.
--/
theorem problemI : PA * PB * (1 - (1 - PC) * (1 - PD)) = 1 / 8 :=
by
  sorry

/-- 
Prove the second statement: 
The sum of the probabilities that the candidate answers exactly 1 or exactly 3 questions correctly is 7/12.
--/
theorem problemII (ξ : ℕ) : (ξ = 1 → (1 - PA) * PB + PA * (1 - PB)) + (ξ = 3 → PA * PB * ((1 - PC) * PD + (1 - PD) * PC)) = 7 / 12 :=
by
  sorry

end problemI_problemII_l4_4683


namespace diagonals_form_triangle_l4_4922

variables {A B C D E : Type} [ConvexPentagon A B C D E]
variable (longest_diagonal_BE : ∀ (d : Diagonal A B C D E), d ≤ BE)

theorem diagonals_form_triangle :
  ∃ BE EC BD : ℝ,
    BE < EC + BD  :=
by sorry

end diagonals_form_triangle_l4_4922


namespace log_base_change_l4_4275

variable (a b : ℝ)

theorem log_base_change (h1 : log 2 / log 3 = a) (h2 : log 7 / log 2 = b) : log 7 / log 3 = a * b := 
sorry

end log_base_change_l4_4275


namespace find_b_additive_inverse_l4_4791

noncomputable def complex_equation (b : ℝ) : ℂ := (4 + b * complex.I) / (1 + complex.I)

theorem find_b_additive_inverse (b : ℝ) (h : complex.re (complex_equation b) + complex.im (complex_equation b) = 0) : b = 0 := 
sorry

end find_b_additive_inverse_l4_4791


namespace angle_between_polar_lines_l4_4853

def angle_between_lines_in_polar_coordinates (ρ θ : ℝ) : ℝ := 
  arctan (1 / 2)

theorem angle_between_polar_lines :
  ∀ (ρ θ : ℝ), (ρ * (cos θ + 2 * sin θ) = 1) → (ρ * sin θ = 1) → 
  angle_between_lines_in_polar_coordinates ρ θ = arctan(1 / 2) :=
by 
  intros ρ θ h₁ h₂
  sorry

end angle_between_polar_lines_l4_4853


namespace number_of_defective_pens_l4_4839

noncomputable def defective_pens (total : ℕ) (prob : ℚ) : ℕ :=
  let N := 6 -- since we already know the steps in the solution leading to N = 6
  let D := total - N
  D

theorem number_of_defective_pens (total : ℕ) (prob : ℚ) :
  (total = 12) → (prob = 0.22727272727272727) → defective_pens total prob = 6 :=
by
  intros ht hp
  unfold defective_pens
  sorry

end number_of_defective_pens_l4_4839


namespace trajectory_of_center_l4_4984

theorem trajectory_of_center (O₁ O₂ O : Type)
  (r₁ r₂ R : ℝ)
  (O₁_center O₂_center O_center : ℝ × ℝ)
  (h1 : r₁ ≠ r₂)
  (h2 : dist O₁_center O₂_center > r₁ + r₂)
  (h3 : dist O₁_center O₂_center < abs (r₁ - r₂))
  (h4 : ∀ p : ℝ × ℝ, dist p O₁_center = r₁ - R ∨ dist p O₁_center = R + r₂)
  (h5 : ∀ p : ℝ × ℝ, dist p O₂_center = R + r₂ ∨ dist p O₂_center = r₁ - R) :
  (trajectory O_center O₁_center O₂_center = one_branch_of_hyperbola ∨ trajectory O_center O₁_center O₂_center = ellipse) :=
sorry

end trajectory_of_center_l4_4984


namespace distance_from_A_to_plane_yoz_l4_4265

-- Definitions for the conditions
structure Point :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

-- Define the point A
def A : Point := ⟨-3, 1, -4⟩

-- Define the plane yoz
def plane_yoz (P : Point) : Prop := P.x = 0

-- Prove the distance from point A to the plane yoz is 3
theorem distance_from_A_to_plane_yoz : ∀ (P : Point), plane_yoz P → real.abs (A.x) = 3 :=
by
  intro P h
  sorry

end distance_from_A_to_plane_yoz_l4_4265


namespace binomial_inequality_l4_4749

theorem binomial_inequality (n : ℕ) (x : ℕ → ℕ) 
  (h_sum : ∑ i in finset.range n, x i = 101 * n) 
  (h_pos : ∀ i, i < n → 0 < x i)
  : (∑ i in finset.range n, x i * (x i - 1) / 2) ≥ 5050 * n := 
by
  sorry

end binomial_inequality_l4_4749


namespace ne_suff_nec_2_pow_x_lt_1_x_sq_lt_1_l4_4908

theorem ne_suff_nec_2_pow_x_lt_1_x_sq_lt_1 :
  ¬ ((∀ x : ℝ, 2^x < 1 → x^2 < 1) ∧ (∀ x : ℝ, x^2 < 1 → 2^x < 1)) :=
by
  sorry

end ne_suff_nec_2_pow_x_lt_1_x_sq_lt_1_l4_4908


namespace triangle_equality_BC_AK_BK_l4_4410

-- Define the main geometric constructs
variables {A B C M N K : Type}

-- Given conditions
variables [Midpoint M A C] -- M is the midpoint of AC
variables [OnSegment N A M] -- N lies on segment AM
variables [EqualAngle M B N C B M] -- ∠MBN = ∠CBM
variables [OnExtension K B N (90 : Angle)] -- K is on the extension of BN such that ∠BMK is right

-- Define the main theorem
theorem triangle_equality_BC_AK_BK :
  BC = AK + BK :=
sorry

end triangle_equality_BC_AK_BK_l4_4410


namespace expression_equals_minus_0p125_l4_4708

-- Define the expression
def compute_expression : ℝ := 0.125^8 * (-8)^7

-- State the theorem to prove
theorem expression_equals_minus_0p125 : compute_expression = -0.125 :=
by {
  sorry
}

end expression_equals_minus_0p125_l4_4708


namespace no_odd_integers_satisfy_equation_l4_4720

theorem no_odd_integers_satisfy_equation :
  ¬ ∃ (x y z : ℤ), (x % 2 ≠ 0) ∧ (y % 2 ≠ 0) ∧ (z % 2 ≠ 0) ∧ 
  (x + y)^2 + (x + z)^2 = (y + z)^2 :=
by
  sorry

end no_odd_integers_satisfy_equation_l4_4720


namespace discriminant_divisible_l4_4515

theorem discriminant_divisible (a b: ℝ) (n: ℤ) (h: (∃ x1 x2: ℝ, 2018*x1^2 + a*x1 + b = 0 ∧ 2018*x2^2 + a*x2 + b = 0 ∧ x1 - x2 = n)): 
  ∃ k: ℤ, a^2 - 4 * 2018 * b = 2018^2 * k := 
by 
  sorry

end discriminant_divisible_l4_4515


namespace all_lines_through_intersections_meet_at_single_point_l4_4678

noncomputable def circles_inscribed_in_segment (segment : Set Point) : Prop :=
  ∀ (S1 S2 : Circle), 
    (S1 ⊆ segment) → 
    (S2 ⊆ segment) → 
    (∃ M N : Point, M ≠ N ∧ M ∈ IntersectionPoints S1 S2 ∧ N ∈ IntersectionPoints S1 S2) 

theorem all_lines_through_intersections_meet_at_single_point
  (segment : Set Point)
  (P : Point)
  (h_circles : circles_inscribed_in_segment segment)
  (h_lines : ∀ (S1 S2 : Circle) (M N : Point), 
      M ≠ N ∧ M ∈ IntersectionPoints S1 S2 ∧ N ∈ IntersectionPoints S1 S2 →
      ∃ Q, lies_on_line Q M N →
      lies_on_line Q P) :
  ∀ (S1 S2 : Circle) (M N : Point),
    M ≠ N ∧ M ∈ IntersectionPoints S1 S2 ∧ N ∈ IntersectionPoints S1 S2 →
    lies_on_line P M N := 
sorry

end all_lines_through_intersections_meet_at_single_point_l4_4678


namespace tom_reaches_virgo_in_correct_time_l4_4098

def first_flight_time : ℝ := 5
def layover_after_first_flight : ℝ := 1
def second_flight_time : ℝ := 2 * first_flight_time
def layover_after_second_flight : ℝ := 2
def third_flight_time : ℝ := first_flight_time / 2
def layover_after_third_flight : ℝ := 3
def first_boat_ride : ℝ := 1.5
def layover_before_final_boat_ride : ℝ := 0.75
def final_boat_ride : ℝ := (first_flight_time - third_flight_time) * 2

def total_time_taken : ℝ :=
  first_flight_time + layover_after_first_flight + second_flight_time +
  layover_after_second_flight + third_flight_time + layover_after_third_flight +
  first_boat_ride + layover_before_final_boat_ride + final_boat_ride

theorem tom_reaches_virgo_in_correct_time :
  total_time_taken = 30.75 := by
  sorry

end tom_reaches_virgo_in_correct_time_l4_4098


namespace batsman_average_after_12th_innings_l4_4618

noncomputable def batsman_average (runs_in_12th_innings : ℕ) (average_increase : ℕ) (initial_average_after_11_innings : ℕ) : ℕ :=
initial_average_after_11_innings + average_increase

theorem batsman_average_after_12th_innings
(score_in_12th_innings : ℕ)
(average_increase : ℕ)
(initial_average_after_11_innings : ℕ)
(total_runs_after_11_innings := 11 * initial_average_after_11_innings)
(total_runs_after_12_innings := total_runs_after_11_innings + score_in_12th_innings)
(new_average_after_12_innings := total_runs_after_12_innings / 12)
:
score_in_12th_innings = 80 ∧ average_increase = 3 ∧ initial_average_after_11_innings = 44 → 
batsman_average score_in_12th_innings average_increase initial_average_after_11_innings = 47 := 
by
  -- skipping the actual proof for now
  sorry

end batsman_average_after_12th_innings_l4_4618


namespace equalize_foma_ierema_l4_4583

variables 
  (F E Y : ℕ)
  (h1 : F - 70 = E + 70)
  (h2 : F - 40 = Y)

def foma_should_give_ierema : ℕ := (F - E) / 2

theorem equalize_foma_ierema (F E Y : ℕ) (h1 : F - 70 = E + 70) (h2 : F - 40 = Y) :
  foma_should_give_ierema F E = 55 := 
by
  sorry

end equalize_foma_ierema_l4_4583


namespace second_day_hike_ratio_l4_4731

theorem second_day_hike_ratio (full_hike_distance first_day_distance third_day_distance : ℕ) 
(h_full_hike: full_hike_distance = 50)
(h_first_day: first_day_distance = 10)
(h_third_day: third_day_distance = 15) : 
(full_hike_distance - (first_day_distance + third_day_distance)) / full_hike_distance = 1 / 2 := by
  sorry

end second_day_hike_ratio_l4_4731


namespace complex_division_result_l4_4398

-- Define the imaginary unit i
def i : ℂ := complex.I

-- Define the complex number and its division result
def complex_division : ℂ := 2 / (1 - i)

-- State the theorem to be proved
theorem complex_division_result : complex_division = 1 + i :=
by
  sorry

end complex_division_result_l4_4398


namespace f_2017_eq_cos_l4_4280

noncomputable def f : ℕ → (ℝ → ℝ)
| 0       => λ x, Real.sin x
| (n + 1) => λ x, (f n x)' 

theorem f_2017_eq_cos (x : ℝ) : f 2017 x = Real.cos x := 
by
  sorry

end f_2017_eq_cos_l4_4280


namespace find_angle_A_find_sin_B_sin_C_l4_4857

variable (a b c : ℝ)
variable (A B C : ℝ)
variable (S : ℝ)

-- Definitions based on the problem conditions
def is_triangle (A B C : ℝ) := A + B + C = π

def valid_angle (angle : ℝ) := 0 < angle ∧ angle < π

def triangle_S : Prop := S = 5 * Real.sqrt 3

def side_b : Prop := b = 5

def equation_cos : Prop := Real.cos (2 * A) - 3 * Real.cos (B + C) = 1

-- Stating the problems (No proofs needed)
theorem find_angle_A (h₁ : is_triangle A B C) (h₂ : valid_angle A)
  (h₃ : equation_cos A B C) : A = Real.pi / 3 := sorry

theorem find_sin_B_sin_C (h₁ : is_triangle A B C) (h₂ : valid_angle A)
  (h₃ : side_b b) (h₄ : triangle_S S) (h₅ : A = Real.pi / 3)
  (h₆ : valid_angle B) (h₇ : valid_angle C) 
  (h₈ : equation_cos A B C) : Real.sin B * Real.sin C = 5 / 7 := sorry

end find_angle_A_find_sin_B_sin_C_l4_4857


namespace solve_equation1_solve_equation2_l4_4507

theorem solve_equation1 (x : ℝ) : x^2 - 2 * x - 2 = 0 ↔ (x = 1 + Real.sqrt 3 ∨ x = 1 - Real.sqrt 3) :=
by
  sorry

theorem solve_equation2 (x : ℝ) : 2 * (x - 3)^2 = x - 3 ↔ (x = 3/2 ∨ x = 7/2) :=
by
  sorry

end solve_equation1_solve_equation2_l4_4507


namespace triangle_equality_lemma_l4_4409

open EuclideanGeometry

theorem triangle_equality_lemma 
  (A B C M N K: Point)
  (hM_midpoint: midpoint M A C)
  (hN_on_AM: lies_on N A M)
  (h_angle_eq: ∠ M B N = ∠ C B M)
  (hK_right: ∠ B M K = 90) :
  dist B C = dist A K + dist B K :=
by
  sorry

end triangle_equality_lemma_l4_4409


namespace ted_worked_hours_l4_4132

variable (t : ℝ)
variable (julie_rate ted_rate combined_rate : ℝ)
variable (julie_alone_time : ℝ)
variable (job_done : ℝ)

theorem ted_worked_hours :
  julie_rate = 1 / 10 →
  ted_rate = 1 / 8 →
  combined_rate = julie_rate + ted_rate →
  julie_alone_time = 0.9999999999999998 →
  job_done = combined_rate * t + julie_rate * julie_alone_time →
  t = 4 :=
by
  sorry

end ted_worked_hours_l4_4132


namespace linear_function_intersects_x_axis_at_2_0_l4_4492

theorem linear_function_intersects_x_axis_at_2_0
  (f : ℝ → ℝ)
  (h : ∀ x, f x = -x + 2) :
  ∃ x, f x = 0 ∧ x = 2 :=
by
  sorry

end linear_function_intersects_x_axis_at_2_0_l4_4492


namespace system_of_equations_no_solution_fractional_eq_l4_4935

section Problem1

variable (x y : ℤ)

theorem system_of_equations :
  (x - y = 8 ∧ 3 * x + y = 12) -> (x = 5 ∧ y = -3) :=
by
  sorry

end Problem1

section Problem2

open set

noncomputable def fractional_eq (x : ℝ) :=
  (3 / (x - 1) - (x + 2) / (x * (x - 1)) = 0)

theorem no_solution_fractional_eq :
  ¬ ∃ x : ℝ, fractional_eq x :=
by
  sorry

end Problem2

end system_of_equations_no_solution_fractional_eq_l4_4935


namespace count_zeros_in_decimal_representation_l4_4304

theorem count_zeros_in_decimal_representation : 
  let f := 1 / (2 ^ 3 * 5 ^ 6)
  in (count_zeros (to_decimal_representation f)) = 5 :=
by
  -- Definitions and further details would be provided here in an actual proof
  sorry

end count_zeros_in_decimal_representation_l4_4304


namespace triangle_equality_lemma_l4_4407

open EuclideanGeometry

theorem triangle_equality_lemma 
  (A B C M N K: Point)
  (hM_midpoint: midpoint M A C)
  (hN_on_AM: lies_on N A M)
  (h_angle_eq: ∠ M B N = ∠ C B M)
  (hK_right: ∠ B M K = 90) :
  dist B C = dist A K + dist B K :=
by
  sorry

end triangle_equality_lemma_l4_4407


namespace division_possible_55_division_possible_54_l4_4644

-- Define the square field and another associated properties
def square_side_length : ℝ := 33
def total_area : ℝ := square_side_length * square_side_length
def plot_area : ℝ := total_area / 3

-- First problem: At most 55 m of fencing
def feasible_fencing_55 (fence_length : ℝ) : Prop :=
  fence_length <= 55

theorem division_possible_55 :
  ∃ division : ℝ → ℝ, feasible_fencing_55 (division square_side_length) := 
sorry

-- Second problem: At most 54 m of fencing
def feasible_fencing_54 (fence_length : ℝ) : Prop :=
  fence_length <= 54

theorem division_possible_54 :
  ∃ division : ℝ → ℝ, feasible_fencing_54 (division square_side_length) := 
sorry

end division_possible_55_division_possible_54_l4_4644


namespace sandy_more_tokens_than_siblings_l4_4497

-- Define the initial conditions
def initial_tokens : ℕ := 3000000
def initial_transaction_fee_percent : ℚ := 0.10
def value_increase_percent : ℚ := 0.20
def additional_tokens : ℕ := 500000
def additional_transaction_fee_percent : ℚ := 0.07
def sandy_keep_percent : ℚ := 0.40
def siblings : ℕ := 7
def sibling_transaction_fee_percent : ℚ := 0.05

-- Define the main theorem to prove
theorem sandy_more_tokens_than_siblings :
  let received_initial_tokens := initial_tokens * (1 - initial_transaction_fee_percent)
  let increased_tokens := received_initial_tokens * (1 + value_increase_percent)
  let received_additional_tokens := additional_tokens * (1 - additional_transaction_fee_percent)
  let total_tokens := increased_tokens + received_additional_tokens
  let sandy_tokens := total_tokens * sandy_keep_percent
  let remaining_tokens := total_tokens * (1 - sandy_keep_percent)
  let each_sibling_tokens := remaining_tokens / siblings * (1 - sibling_transaction_fee_percent)
  sandy_tokens - each_sibling_tokens = 1180307.1428 := sorry

end sandy_more_tokens_than_siblings_l4_4497


namespace foma_should_give_ierema_55_coins_l4_4566

variables (F E Y : ℝ)

-- Conditions
def condition1 : Prop := F - 70 = E + 70
def condition2 : Prop := F - 40 = Y
def condition3 : Prop := E + 70 = Y

theorem foma_should_give_ierema_55_coins
  (h1 : condition1 F E) (h2 : condition2 F Y) (h3 : condition3 E Y) :
  F - (E + 55) = E + 55 :=
by
  sorry

end foma_should_give_ierema_55_coins_l4_4566


namespace highest_place_joker_can_achieve_is_6_l4_4348

-- Define the total number of teams
def total_teams : ℕ := 16

-- Define conditions for points in football
def points_win : ℕ := 3
def points_draw : ℕ := 1
def points_loss : ℕ := 0

-- Condition definitions for Joker's performance in the tournament
def won_against_strong_teams (j k : ℕ) : Prop := j < k
def lost_against_weak_teams (j k : ℕ) : Prop := j > k

-- Define the performance of all teams
def teams (t : ℕ) := {n // n < total_teams}

-- Function to calculate Joker's points based on position k
def joker_points (k : ℕ) : ℕ := (total_teams - k) * points_win

theorem highest_place_joker_can_achieve_is_6 : ∃ k, k = 6 ∧ 
  (∀ j, 
    (j < k → won_against_strong_teams j k) ∧ 
    (j > k → lost_against_weak_teams j k) ∧
    (∃! p, p = joker_points k)) :=
by
  sorry

end highest_place_joker_can_achieve_is_6_l4_4348


namespace vacation_cost_l4_4541

theorem vacation_cost (n : ℕ) (h : 480 / n + 40 = 120) : n = 6 :=
sorry

end vacation_cost_l4_4541


namespace cat_ate_14_grams_l4_4914

theorem cat_ate_14_grams (bowl_weight_empty : ℕ) (food_per_day : ℕ) (refill_every_days : ℕ) (bowl_weight_after_cat_eats : ℕ) :
  bowl_weight_empty = 420 → food_per_day = 60 → refill_every_days = 3 → bowl_weight_after_cat_eats = 586 →
  (bowl_weight_empty + food_per_day * refill_every_days) - bowl_weight_after_cat_eats = 14 :=
begin
  intros h1 h2 h3 h4,
  rw [h1, h2, h3, h4],
  norm_num,
end

end cat_ate_14_grams_l4_4914


namespace foma_should_give_ierema_55_coins_l4_4561

theorem foma_should_give_ierema_55_coins (F E Y : ℤ)
  (h1 : F - 70 = E + 70)
  (h2 : F - 40 = Y)
  (h3 : E + 70 = Y) :
  F - E = 110 → F - E = 55 :=
by
  sorry

end foma_should_give_ierema_55_coins_l4_4561


namespace china_nhsm_league_2021_zhejiang_p15_l4_4018

variable (x y z : ℝ)

theorem china_nhsm_league_2021_zhejiang_p15 (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h : Real.sqrt x + Real.sqrt y + Real.sqrt z = 1) : 
  (x ^ 4 + y ^ 2 * z ^ 2) / (x ^ (5 / 2) * (y + z)) + 
  (y ^ 4 + z ^ 2 * x ^ 2) / (y ^ (5 / 2) * (z + x)) + 
  (z ^ 4 + y ^ 2 * x ^ 2) / (z ^ (5 / 2) * (y + x)) ≥ 1 := 
sorry

end china_nhsm_league_2021_zhejiang_p15_l4_4018


namespace ratio_solution_A_to_B_l4_4067

theorem ratio_solution_A_to_B :
  (∀ (a b : ℝ), 
    (0 < a ∧ 0 < b) → 
    (let frac_alc_A := 4 / (5 + 4);
         frac_alc_B := 5 / (6 + 5);
         new_concentration := 0.45 in
      frac_alc_A * a + frac_alc_B * b = new_concentration * (a + b))
  → a / b = 1) :=
by {
  intros a b hab h,
  let frac_alc_A := 4 / 9,
  let frac_alc_B := 5 / 11,
  let new_concentration := 0.45,
  have h1: frac_alc_A * a + frac_alc_B * b = new_concentration * (a + b), from h,
  sorry
}

end ratio_solution_A_to_B_l4_4067


namespace num_factors_of_2310_with_more_than_three_factors_l4_4817

theorem num_factors_of_2310_with_more_than_three_factors : 
  (∃ n : ℕ, n > 0 ∧ ∀ d : ℕ, d ∣ 2310 → (∀ f : ℕ, f ∣ d → f = 1 ∨ f = d ∨ f ∣ d) → 26 = n) := sorry

end num_factors_of_2310_with_more_than_three_factors_l4_4817


namespace find_constant_a_l4_4946

theorem find_constant_a (a : ℝ) (h : ∀ x : ℝ, x ≠ 0 → (a + 1/x) * (1-x)^4 = (a + 1/x) * (1 - 4*x + 6*x^2 + ... ) ∧ coeff_expr1_x_term (a + 1/x, (1 - x)^4) = -6) : a = 3 := by
  sorry

end find_constant_a_l4_4946


namespace folding_paper_ratio_l4_4660

-- The condition that the length is 2 times the width
def length_eq_2_times_width (width length: ℝ) : Prop :=
  length = 2 * width

-- The area of the paper should be A
def area (width length area: ℝ) : Prop :=
  area = width * length

-- The ratio of the new area to the original area
def ratio (new_area area ratio: ℝ) : Prop :=
  ratio = new_area / area

-- to prove the statement
theorem folding_paper_ratio (w : ℝ) (A : ℝ) (B : ℝ) (l : ℝ) :
  length_eq_2_times_width w l →
  area w l A →
  B = A - (A * (√2 / 4)) →
  ratio B A (1 - (√2 / 4)) :=
by
  intros h1 h2 h3
  sorry

end folding_paper_ratio_l4_4660


namespace simplify_fraction_l4_4501

theorem simplify_fraction (x : ℝ) : (x + 2) / 4 + (3 - 4 * x) / 3 = (-13 * x + 18) / 12 :=
by
  sorry

end simplify_fraction_l4_4501


namespace prime_diff_of_cubes_sum_of_square_and_three_times_square_l4_4019

theorem prime_diff_of_cubes_sum_of_square_and_three_times_square 
  (p : ℕ) (a b : ℕ) (h_prime : Nat.Prime p) (h_diff : p = a^3 - b^3) :
  ∃ c d : ℤ, p = c^2 + 3 * d^2 := 
  sorry

end prime_diff_of_cubes_sum_of_square_and_three_times_square_l4_4019


namespace triangle_property_l4_4447

variables {ABC : Type*} [inner_product_space ℝ ABC]
variables (A B C M K N : ABC)

noncomputable def midpoint (A C : ABC) (M : ABC) : Prop := dist A M = dist M C

noncomputable def angle_equality (B M N : ABC) : Prop := ∠M B N = ∠C B M

noncomputable def right_angle (B M K : ABC) : Prop := ∠B M K = π / 2

theorem triangle_property (h1 : midpoint A C M) (h2 : angle_equality B M N) (h3 : right_angle B M K) : dist B C = dist A K + dist B K :=
sorry

end triangle_property_l4_4447


namespace no_pair_of_primes_l4_4737

theorem no_pair_of_primes (p q : ℕ) (hp_prime : Prime p) (hq_prime : Prime q) (h_gt : p > q) :
  ¬ (∃ (h : ℤ), 2 * (p^2 - q^2) = 8 * h + 4) :=
by
  sorry

end no_pair_of_primes_l4_4737


namespace tangent_line_eq_l4_4071

-- Define the curve and its derivative
def curve (x : ℝ) : ℝ := Real.log x + x + 1
def derivative (x : ℝ) : ℝ := 1/x + 1

-- Define the target slope and find the corresponding x-coordinate
def target_slope : ℝ := 2

-- Assertion that the equation of the tangent line is y = 2x
theorem tangent_line_eq (x y : ℝ) (h₁ : x = 1) (h₂ : y = 2) :
    curve x = y → derivative x = target_slope → y = 2 * x := 
by
  sorry

end tangent_line_eq_l4_4071


namespace determine_parameters_l4_4246

theorem determine_parameters
(eq_poly : ∀ x : ℝ, x^5 + 2*x^4 + a*x^2 + b*x = c) :
  ({ -1, 1 } : set ℝ) = { x : ℝ | x^5 + 2*x^4 + a*x^2 + b*x = c } →
  a = -6 ∧ b = -1 ∧ c = -4 :=
by 
  -- Proof can go here
  sorry

end determine_parameters_l4_4246


namespace triangle_area_from_perimeter_and_inradius_l4_4967

theorem triangle_area_from_perimeter_and_inradius
  (P : ℝ) (r : ℝ) (A : ℝ)
  (h₁ : P = 24)
  (h₂ : r = 2.5) :
  A = 30 := 
by
  sorry

end triangle_area_from_perimeter_and_inradius_l4_4967


namespace noncongruent_triangles_count_l4_4016

-- Define the points and the triangle relationships
variables {R S T X Y Z : Type}
variables [EquilateralTriangle R S T]
variables [Midpoint X R S]
variables [Midpoint Y S T]
variables [Midpoint Z T R]

-- Define the proposition that needs to be proven
theorem noncongruent_triangles_count : 
  (number_of_noncongruent_triangles (R, S, T, X, Y, Z) = 4) :=
sorry

end noncongruent_triangles_count_l4_4016


namespace total_points_each_team_l4_4081

def score_touchdown := 7
def score_field_goal := 3
def score_safety := 2

def team_hawks_first_match_score := 3 * score_touchdown + 2 * score_field_goal + score_safety
def team_eagles_first_match_score := 5 * score_touchdown + 4 * score_field_goal
def team_hawks_second_match_score := 4 * score_touchdown + 3 * score_field_goal
def team_falcons_second_match_score := 6 * score_touchdown + 2 * score_safety

def total_score_hawks := team_hawks_first_match_score + team_hawks_second_match_score
def total_score_eagles := team_eagles_first_match_score
def total_score_falcons := team_falcons_second_match_score

theorem total_points_each_team :
  total_score_hawks = 66 ∧ total_score_eagles = 47 ∧ total_score_falcons = 46 :=
by
  unfold total_score_hawks team_hawks_first_match_score team_hawks_second_match_score
           total_score_eagles team_eagles_first_match_score
           total_score_falcons team_falcons_second_match_score
           score_touchdown score_field_goal score_safety
  sorry

end total_points_each_team_l4_4081


namespace oplus_self_twice_l4_4794

def my_oplus (x y : ℕ) := 3^x - y

theorem oplus_self_twice (a : ℕ) : my_oplus a (my_oplus a a) = a := by
  sorry

end oplus_self_twice_l4_4794


namespace problem1_problem2_l4_4800

open Real

-- Definition of the function f(x)
def f (x a : ℝ) : ℝ := exp x - a * x

-- Definition of the function h(x)
def h (x a : ℝ) : ℝ := exp x - a * x - (1 / 2) * x^2

-- Minimum value of the function f(x)
def g (a : ℝ) : ℝ := infi (fun x => f x a)

-- Theorem 1: Prove g(a) <= 1 for a > 0
theorem problem1 (a : ℝ) (ha : a > 0) : g a ≤ 1 := by
  sorry

-- Theorem 2: Prove h(x_1) + h(x_2) > 2 if h(x) has two critical points x_1, x_2 with x_1 < x_2
theorem problem2 (a x1 x2 : ℝ) (ha : a > 1) (hx : h' x1 a = 0 ∧ h' x2 a = 0) (h_crit : x1 < x2) : h x1 a + h x2 a > 2 := by
  sorry

end problem1_problem2_l4_4800


namespace mean_problem_l4_4961

theorem mean_problem : 
  (8 + 12 + 24) / 3 = (16 + z) / 2 → z = 40 / 3 :=
by
  intro h
  sorry

end mean_problem_l4_4961


namespace combined_percentage_grade4_l4_4539

-- Definitions based on the given conditions
def Pinegrove_total_students : ℕ := 120
def Maplewood_total_students : ℕ := 180

def Pinegrove_grade4_percentage : ℕ := 10
def Maplewood_grade4_percentage : ℕ := 20

theorem combined_percentage_grade4 :
  let combined_total_students := Pinegrove_total_students + Maplewood_total_students
  let Pinegrove_grade4_students := Pinegrove_grade4_percentage * Pinegrove_total_students / 100
  let Maplewood_grade4_students := Maplewood_grade4_percentage * Maplewood_total_students / 100 
  let combined_grade4_students := Pinegrove_grade4_students + Maplewood_grade4_students
  (combined_grade4_students * 100 / combined_total_students) = 16 := by
  sorry

end combined_percentage_grade4_l4_4539


namespace find_m_squared_l4_4337

def line (m : ℝ) (x : ℝ) : ℝ := m * x + 2
def ellipse (x y : ℝ) : Prop := x^2 + 9 * y^2 = 9
def circle (x y : ℝ) : Prop := x^2 + y^2 = 4

theorem find_m_squared (m : ℝ) :
  (∀ x y : ℝ, ellipse x y → y = line m x → (9 * m^2 * x^2 + 36 * m * x + 27) = 0) →
  (∀ x y : ℝ, circle x y → y = line m x → ((1 + m^2) * x^2 + 4 * m * x = 0)) →
  m^2 = 1 / 3 :=
begin
  -- The proof is omitted as stated in the instructions.
  sorry
end

end find_m_squared_l4_4337


namespace tall_trees_unspecified_l4_4974

-- Conditions definitions
def short_trees_initial := 41
def short_trees_planted := 57
def short_trees_final := 98

-- The statement to prove that currently the number of tall trees remains unspecified
theorem tall_trees_unspecified
  (short_trees_initial : ℕ)
  (short_trees_planted : ℕ)
  (short_trees_final : ℕ)
  (h1 : short_trees_initial + short_trees_planted = short_trees_final)
  : ∃ (tall_trees : ℕ), true :=
by
  exists 0 -- Since the number of tall trees is unspecified
  trivial

end tall_trees_unspecified_l4_4974


namespace percentage_of_boys_from_A_studying_science_is_30_l4_4341

-- Definitions based on given conditions
def T : ℝ := 350
def num_from_A : ℝ := 0.20 * T
def boys_not_studying_science : ℝ := 49
def boys_studying_science : ℝ := num_from_A - boys_not_studying_science
def percentage_science : ℝ := (boys_studying_science / num_from_A) * 100

-- The proof problem stating the desired result
theorem percentage_of_boys_from_A_studying_science_is_30 :
  percentage_science = 30 :=
by
  -- Proof to be filled in
  sorry

end percentage_of_boys_from_A_studying_science_is_30_l4_4341


namespace degree_poly_sum_l4_4032

noncomputable def f (z : ℤ) : ℤ := a_4 * z^4 + a_3 * z^3 + a_2 * z^2 + a_1 * z + a_0
noncomputable def g (z : ℤ) : ℤ := b_3 * z^3 + b_2 * z^2 + b_1 * z + b_0

theorem degree_poly_sum (h₁ : a_4 ≠ 0) :
    degree (f + g) = 4 :=
sorry

end degree_poly_sum_l4_4032


namespace number_of_zeros_f_part1_value_of_a_part2_l4_4803
noncomputable def f_part1 (x : ℝ) : ℝ := (1 / x) + 4 * Real.log x

theorem number_of_zeros_f_part1 : 
  ∃ x1 x2 : ℝ, 0 < x1 ∧ x1 < 1/4 ∧ x1 > 0 ∧ f_part1 x1 = 0 ∧ 
  x2 > 1/4 ∧ f_part1 x2 = 0 := 
sorry

noncomputable def f_part2 (x : ℝ) (a : ℝ) : ℝ := e^x + a * Real.log (x + 1)

theorem value_of_a_part2 : 
  (∀ x : ℝ, f_part2 (x + 1) a - 1 / (x + 1) ≥ 1) → a = -1 :=
sorry

end number_of_zeros_f_part1_value_of_a_part2_l4_4803


namespace sum_of_digits_y_coordinate_C_l4_4375

def is_on_parabola (P : ℝ × ℝ) : Prop := P.2 = P.1 ^ 2
def is_right_triangle (A B C : ℝ × ℝ) : Prop :=
  (A ≠ B ∧ B ≠ C ∧ C ≠ A) ∧
  ((B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0 ∨
   (C.1 - B.1) * (A.1 - B.1) + (C.2 - B.2) * (A.2 - B.2) = 0 ∨
   (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0)

def is_parallel_to_x_axis (A B : ℝ × ℝ) : Prop := A.2 = B.2

def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  (1 / 2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem sum_of_digits_y_coordinate_C (A B C : ℝ × ℝ) :
  is_on_parabola A →
  is_on_parabola B →
  is_on_parabola C →
  is_parallel_to_x_axis A B →
  is_right_triangle A B C →
  triangle_area A B C = 2008 →
  (nat.digits 10 (int.nat_abs (C.2.natAbs))).sum = 18 :=
by
  sorry

end sum_of_digits_y_coordinate_C_l4_4375


namespace probability_both_white_balls_probability_at_least_one_white_ball_l4_4149

open Classical

noncomputable def num_white_balls : ℕ := 3
noncomputable def num_black_balls : ℕ := 2
noncomputable def total_balls : ℕ := num_white_balls + num_black_balls

def event_A : Set (ℕ × ℕ) := { (1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2) }
def event_B : Set (ℕ × ℕ) := { (1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2),
                               (1, 4), (1, 5), (2, 4), (2, 5), (3, 4), (3, 5),
                               (4, 1), (4, 2), (4, 3), (5, 1), (5, 2), (5, 3) }

theorem probability_both_white_balls :
  (event_A.card : ℚ) / (Finset.univ.card : ℚ) = 3 / 10 :=
by
  -- skipping the proof
  sorry

theorem probability_at_least_one_white_ball :
  (event_B.card : ℚ) / (Finset.univ.card : ℚ) = 9 / 10 :=
by
  -- skipping the proof
  sorry

end probability_both_white_balls_probability_at_least_one_white_ball_l4_4149


namespace probability_of_duplicate_in_8_dice_rolls_l4_4919

/-- The probability that at least two of the 8 fair 8-sided dice show the same number --/
theorem probability_of_duplicate_in_8_dice_rolls : 
  let total_outcomes := (8 : ℕ)^8
  let all_different_outcomes := Nat.factorial 8
  let probability_all_different := (all_different_outcomes : ℝ) / (total_outcomes : ℝ)
  in 1 - probability_all_different = 1 - (40320 / 16777216) := by
  let total_outcomes := (8 : ℕ)^8
  let all_different_outcomes := Nat.factorial 8
  let probability_all_different := (all_different_outcomes : ℝ) / (total_outcomes : ℝ)
  show 1 - probability_all_different = 1 - (40320 / 16777216)
  sorry

end probability_of_duplicate_in_8_dice_rolls_l4_4919


namespace linear_function_intersects_x_axis_at_2_0_l4_4493

theorem linear_function_intersects_x_axis_at_2_0
  (f : ℝ → ℝ)
  (h : ∀ x, f x = -x + 2) :
  ∃ x, f x = 0 ∧ x = 2 :=
by
  sorry

end linear_function_intersects_x_axis_at_2_0_l4_4493


namespace dora_rate_correct_l4_4694

noncomputable def betty_rate : ℕ := 10
noncomputable def dora_rate : ℕ := 8
noncomputable def total_time : ℕ := 5
noncomputable def betty_break_time : ℕ := 2
noncomputable def cupcakes_difference : ℕ := 10

theorem dora_rate_correct :
  ∃ D : ℕ, 
  (D = dora_rate) ∧ 
  ((total_time - betty_break_time) * betty_rate = 30) ∧ 
  (total_time * D - 30 = cupcakes_difference) :=
sorry

end dora_rate_correct_l4_4694


namespace find_f_six_l4_4956

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_six (f : ℝ → ℝ)
  (h1 : ∀ x y : ℝ, x * f y = y * f x)
  (h2 : f 18 = 24) :
  f 6 = 8 :=
sorry

end find_f_six_l4_4956


namespace min_real_roots_l4_4897

noncomputable def polynomial : Type := 
  {p : polynomial ℝ // p.degree = 2010}

def has_real_roots (p : polynomial) (n : ℕ) :=
  ∃ (roots : finset ℝ), roots.card = n ∧ ∀ x, x ∈ roots → polynomial.eval x p = 0

theorem min_real_roots (p : polynomial)
  (h_distinct_magnitudes : (finset.image (norm ∘ polynomial.root) (polynomial.roots p)).card = 1010) :
  ∃ (n : ℕ), has_real_roots p n ∧ n = 10 := 
sorry

end min_real_roots_l4_4897


namespace equal_share_each_shopper_l4_4862

theorem equal_share_each_shopper 
  (amount_giselle : ℕ)
  (amount_isabella : ℕ)
  (amount_sam : ℕ)
  (H1 : amount_isabella = amount_sam + 45)
  (H2 : amount_isabella = amount_giselle + 15)
  (H3 : amount_giselle = 120) : 
  (amount_isabella + amount_sam + amount_giselle) / 3 = 115 :=
by
  -- The proof is omitted.
  sorry

end equal_share_each_shopper_l4_4862


namespace problem_l4_4921

open Real 

noncomputable def sqrt_log_a (a : ℝ) : ℝ := sqrt (log a / log 10)
noncomputable def sqrt_log_b (b : ℝ) : ℝ := sqrt (log b / log 10)

theorem problem (a b : ℝ) 
  (ha_pos : 0 < a)
  (hb_pos : 0 < b)
  (condition1 : sqrt_log_a a + 2 * sqrt_log_b b + 2 * log (sqrt a) / log 10 + log (sqrt b) / log 10 = 150)
  (int_sqrt_log_a : ∃ (m : ℕ), sqrt_log_a a = m)
  (int_sqrt_log_b : ∃ (n : ℕ), sqrt_log_b b = n)
  (condition2 : a^2 * b = 10^81) :
  a * b = 10^85 :=
sorry

end problem_l4_4921


namespace exists_point_condition_l4_4765

def condition_ellipse (x y : ℝ) := 
  sqrt ((x + 1)^2 + y^2) + sqrt ((x - 1)^2 + y^2) = 2 * sqrt 2

def trajectory_ellipse (x y : ℝ) := 
  x^2 / 2 + y^2 = 1

def on_line (N : ℝ × ℝ) := 
  N.1 = -1 / 2

noncomputable def exists_point_N (P Q : ℝ × ℝ) := 
  P ∈ set_of (λ (x y : ℝ), trajectory_ellipse x y) ∧
  Q ∈ set_of (λ (x y : ℝ), trajectory_ellipse x y) ∧
  ∃ N : ℝ × ℝ, on_line N ∧ N.2 = (sqrt 19 / 19) ∧
  (P.1 - 1) * (Q.1 - 1) + P.2 * Q.2 = 0 ∧
  (P.1 + Q.1) / 2 = N.1 ∧
  (P.2 + Q.2) / 2 = N.2

theorem exists_point_condition : 
  ∃ N : ℝ × ℝ, on_line N ∧ 
  (N.2 = sqrt 19 / 19 ∨ N.2 = -sqrt 19 / 19) ∧
  ( ∃ P Q : ℝ × ℝ, exists_point_N P Q) := 
sorry

end exists_point_condition_l4_4765


namespace football_game_total_collection_l4_4674

theorem football_game_total_collection (adult_price child_price : ℝ) (total_attendees total_adults : ℕ)
  (h1 : adult_price = 0.60)
  (h2 : child_price = 0.25)
  (h3 : total_attendees = 280)
  (h4 : total_adults = 200) :
  (total_adults * adult_price + (total_attendees - total_adults) * child_price) = 140 :=
by
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end football_game_total_collection_l4_4674


namespace candy_pieces_total_l4_4928

def number_of_packages_of_candy := 45
def pieces_per_package := 9

theorem candy_pieces_total : number_of_packages_of_candy * pieces_per_package = 405 :=
by
  sorry

end candy_pieces_total_l4_4928


namespace Tom_search_cost_l4_4097

theorem Tom_search_cost (first_5_days_rate: ℕ) (first_5_days: ℕ) (remaining_days_rate: ℕ) (total_days: ℕ) : 
  first_5_days_rate = 100 → 
  first_5_days = 5 → 
  remaining_days_rate = 60 → 
  total_days = 10 → 
  (first_5_days * first_5_days_rate + (total_days - first_5_days) * remaining_days_rate) = 800 := 
by 
  intros h1 h2 h3 h4 
  sorry

end Tom_search_cost_l4_4097


namespace sum_of_solutions_l4_4124

-- Define the quadratic equation as a product of linear factors
def quadratic_eq (x : ℚ) : Prop := (4 * x + 6) * (3 * x - 8) = 0

-- Define the roots of the quadratic equation
def root1 : ℚ := -3 / 2
def root2 : ℚ := 8 / 3

-- Sum of the roots of the quadratic equation
def sum_of_roots : ℚ := root1 + root2

-- Theorem stating that the sum of the roots is 7/6
theorem sum_of_solutions : sum_of_roots = 7 / 6 := by
  sorry

end sum_of_solutions_l4_4124


namespace value_is_200_l4_4322

variable (x value : ℝ)
variable (h1 : 0.20 * x = value)
variable (h2 : 1.20 * x = 1200)

theorem value_is_200 : value = 200 :=
by
  sorry

end value_is_200_l4_4322


namespace period_of_cos_3x_l4_4110

theorem period_of_cos_3x :
  ∃ T : ℝ, (∀ x : ℝ, (Real.cos (3 * (x + T))) = Real.cos (3 * x)) ∧ (T = (2 * Real.pi) / 3) :=
sorry

end period_of_cos_3x_l4_4110


namespace smallest_sum_of_three_numbers_l4_4538

theorem smallest_sum_of_three_numbers : 
  ∃ (x y z : ℤ), x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ x ∈ {0, 9, -4, 16, -8} ∧ y ∈ {0, 9, -4, 16, -8} ∧ z ∈ {0, 9, -4, 16, -8} ∧ x + y + z = -12 :=
by {
  sorry
}

end smallest_sum_of_three_numbers_l4_4538


namespace triangle_angle_sum_l4_4421

variable (ABC : Type) [triangle ABC]

-- Definitions from conditions in a)
variable (A B C M N K : point ABC)
variable (M_midpoint : midpoint M A C)
variable (N_on_AM : lies_on N A M)
variable (angle_eq1 : ∠ M B N = ∠ C B M)
variable (K_ext_BN : lies_on_ext K B N)
variable (angle_eq2 : ∠ B M K = 90)

-- Goal to prove
theorem triangle_angle_sum (HC : M_midpoint) (HN : N_on_AM) (HB : angle_eq1) (HK : K_ext_BN) (HA : angle_eq2) :
  segment_length B C = segment_length A K + segment_length B K := sorry

end triangle_angle_sum_l4_4421


namespace single_elimination_31_games_l4_4473

/-- In a single-elimination tournament with 32 teams, 31 games are played to determine the winner. -/
theorem single_elimination_31_games (n : ℕ) (h : n = 32) : n - 1 = 31 := 
by
  rw [h]
  norm_num

end single_elimination_31_games_l4_4473


namespace triangle_property_l4_4449

variables {ABC : Type*} [inner_product_space ℝ ABC]
variables (A B C M K N : ABC)

noncomputable def midpoint (A C : ABC) (M : ABC) : Prop := dist A M = dist M C

noncomputable def angle_equality (B M N : ABC) : Prop := ∠M B N = ∠C B M

noncomputable def right_angle (B M K : ABC) : Prop := ∠B M K = π / 2

theorem triangle_property (h1 : midpoint A C M) (h2 : angle_equality B M N) (h3 : right_angle B M K) : dist B C = dist A K + dist B K :=
sorry

end triangle_property_l4_4449


namespace mr_smith_buys_boxes_l4_4913

theorem mr_smith_buys_boxes :
  ∀ (mr_smith_initial_markers mr_smith_final_markers markers_per_box : ℕ),
  mr_smith_initial_markers = 32 →
  mr_smith_final_markers = 86 →
  markers_per_box = 9 →
  (mr_smith_final_markers - mr_smith_initial_markers) / markers_per_box = 6 :=
by
  intros mr_smith_initial_markers mr_smith_final_markers markers_per_box
  intros h_init h_final h_per_box
  rw [h_init, h_final, h_per_box]
  norm_num
  sorry

end mr_smith_buys_boxes_l4_4913


namespace tan_Pi_div_7_is_root_14_l4_4717

theorem tan_Pi_div_7_is_root_14 (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 13) :
  (tan (Real.pi / 7) + Complex.I) / (tan (Real.pi / 7) - Complex.I) =
  Complex.exp (Complex.I * (2 * n * Real.pi / 14)) :=
  n = 3 :=
sorry

end tan_Pi_div_7_is_root_14_l4_4717


namespace number_of_jars_pasta_sauce_l4_4002

-- Conditions
def pasta_cost_per_kg := 1.5
def pasta_weight_kg := 2.0
def ground_beef_cost_per_kg := 8.0
def ground_beef_weight_kg := 1.0 / 4.0
def quesadilla_cost := 6.0
def jar_sauce_cost := 2.0
def total_money := 15.0

-- Helper definitions for total costs
def pasta_total_cost := pasta_weight_kg * pasta_cost_per_kg
def ground_beef_total_cost := ground_beef_weight_kg * ground_beef_cost_per_kg
def other_total_cost := quesadilla_cost + pasta_total_cost + ground_beef_total_cost
def remaining_money := total_money - other_total_cost

-- Proof statement
theorem number_of_jars_pasta_sauce :
  (remaining_money / jar_sauce_cost) = 2 := by
  sorry

end number_of_jars_pasta_sauce_l4_4002


namespace angle_between_a_and_c_max_value_of_f_l4_4815

def vec_a (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
def vec_b (x : ℝ) : ℝ × ℝ := (-Real.cos x, Real.cos x)
def vec_c : ℝ × ℝ := (-1, 0)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

noncomputable def norm (u : ℝ × ℝ) : ℝ := Real.sqrt (u.1 * u.1 + u.2 * u.2)

theorem angle_between_a_and_c (x : ℝ) (hx : x = Real.pi / 6) :
  let angle := Real.arccos ((dot_product (vec_a x) vec_c) / (norm (vec_a x) * norm vec_c))
  angle = 5 * Real.pi / 6 :=
by
  sorry

noncomputable def f (x : ℝ) : ℝ := 2 * dot_product (vec_a x) (vec_b x) + 1

theorem max_value_of_f :
  let max_f := Real.sqrt 2 * Real.sin (2 * (Real.pi / 2) - Real.pi / 4)
  x ∈ Set.Icc (Real.pi / 2) (9 * Real.pi / 8) →
  ∃ x₀, x₀ = Real.pi / 2 ∧ f x₀ = 1 :=
by
  sorry

end angle_between_a_and_c_max_value_of_f_l4_4815


namespace car_catches_truck_in_7_hours_l4_4144

-- Definitions based on the conditions
def initial_distance := 175 -- initial distance in kilometers
def truck_speed := 40 -- speed of the truck in km/h
def car_initial_speed := 50 -- initial speed of the car in km/h
def car_speed_increase := 5 -- speed increase per hour for the car in km/h

-- The main statement to prove
theorem car_catches_truck_in_7_hours :
  ∃ n : ℕ, (n ≥ 0) ∧ 
  (car_initial_speed - truck_speed) * n + (car_speed_increase * n * (n - 1) / 2) = initial_distance :=
by
  existsi 7
  -- Check the equation for n = 7
  -- Simplify: car initial extra speed + sum of increase terms should equal initial distance
  -- (50 - 40) * 7 + 5 * 7 * 6 / 2 = 175
  -- (10) * 7 + 35 * 3 / 2 = 175
  -- 70 + 105 = 175
  sorry

end car_catches_truck_in_7_hours_l4_4144


namespace total_length_of_T_l4_4385

noncomputable def T : Set (ℝ × ℝ) :=
  { p : ℝ × ℝ | let (x, y) := p in abs (abs x - 3 - 2) + abs (abs y - 3 - 2) = 2 }

theorem total_length_of_T : 
  (4 * 8 * Real.sqrt 2) = 128 * Real.sqrt 2 :=
by
  sorry

end total_length_of_T_l4_4385


namespace foma_should_give_ierema_55_coins_l4_4559

theorem foma_should_give_ierema_55_coins (F E Y : ℤ)
  (h1 : F - 70 = E + 70)
  (h2 : F - 40 = Y)
  (h3 : E + 70 = Y) :
  F - E = 110 → F - E = 55 :=
by
  sorry

end foma_should_give_ierema_55_coins_l4_4559


namespace fraction_of_blue_cars_l4_4975

-- Definitions of the conditions
def total_cars : ℕ := 516
def red_cars : ℕ := total_cars / 2
def black_cars : ℕ := 86
def blue_cars : ℕ := total_cars - (red_cars + black_cars)

-- Statement to prove that the fraction of blue cars is 1/3
theorem fraction_of_blue_cars :
  (blue_cars : ℚ) / total_cars = 1 / 3 :=
by
  sorry -- Proof to be filled in

end fraction_of_blue_cars_l4_4975


namespace expected_value_of_fair_12_sided_die_l4_4610

noncomputable def fair_die_probability (n : ℕ) : ℚ := 1 / 12

theorem expected_value_of_fair_12_sided_die :
  (∑ x in Finset.range 12, (x + 1) * fair_die_probability 12) = 6.5 := by
  sorry

end expected_value_of_fair_12_sided_die_l4_4610


namespace center_in_triangle_probability_l4_4086

theorem center_in_triangle_probability (n : ℕ) :
  let vertices := 2 * n + 1
  let total_ways := vertices.choose 3
  let no_center_ways := vertices * (n.choose 2) / 2
  let p_no_center := no_center_ways / total_ways
  let p_center := 1 - p_no_center
  p_center = (n + 1) / (4 * n - 2) := sorry

end center_in_triangle_probability_l4_4086


namespace M_identically_zero_l4_4882

noncomputable def M (x y : ℝ) : ℝ := sorry

theorem M_identically_zero (a : ℝ) (h1 : a > 1) (h2 : ∀ x, M x (a^x) = 0) : ∀ x y, M x y = 0 :=
sorry

end M_identically_zero_l4_4882


namespace foma_should_give_ierema_55_coins_l4_4569

variables (F E Y : ℝ)

-- Conditions
def condition1 : Prop := F - 70 = E + 70
def condition2 : Prop := F - 40 = Y
def condition3 : Prop := E + 70 = Y

theorem foma_should_give_ierema_55_coins
  (h1 : condition1 F E) (h2 : condition2 F Y) (h3 : condition3 E Y) :
  F - (E + 55) = E + 55 :=
by
  sorry

end foma_should_give_ierema_55_coins_l4_4569


namespace max_cos_a_l4_4464

theorem max_cos_a (a b : ℝ) (h : Real.cos (a - b) = Real.cos a - Real.cos b) : 
  Real.cos a ≤ 1 :=
by
  -- Proof goes here
  sorry

end max_cos_a_l4_4464


namespace pipe_length_l4_4658

theorem pipe_length (L_short : ℕ) (hL_short : L_short = 59) : 
    L_short + 2 * L_short = 177 := by
  sorry

end pipe_length_l4_4658


namespace cannot_represent_parabola_l4_4241

theorem cannot_represent_parabola (k : ℝ) :
  ∀ x y : ℝ, ¬(x^2 + k*y^2 = 1) ∧ (k = 0 → False) → 
  (0 < k ∧ k ≠ 1 → False) → 
  (k = 1 → False) → 
  (k > 1 → False) := 
begin
  intros x y h1 h2 h3 h4,
  sorry
end

end cannot_represent_parabola_l4_4241


namespace five_digit_numbers_last_two_different_l4_4305

def total_five_digit_numbers : ℕ := 90000

def five_digit_numbers_last_two_same : ℕ := 9000

theorem five_digit_numbers_last_two_different :
  (total_five_digit_numbers - five_digit_numbers_last_two_same) = 81000 := 
by 
  sorry

end five_digit_numbers_last_two_different_l4_4305


namespace length_of_MN_l4_4401

noncomputable def midpoint (p1 p2 : Point ℝ) : Point ℝ :=
{ x := (p1.x + p2.x) / 2,
  y := (p1.y + p2.y) / 2,
  z := (p1.z + p2.z) / 2 }

theorem length_of_MN
  (A B C D M N : Point ℝ)
  (a : ℝ)
  (hAC : dist A C = a)
  (hM : is_centroid (Face.ABD A B D) M)
  (hN : is_centroid (Face.BCD B C D) N) :
  dist M N = a / 3 :=
sorry

end length_of_MN_l4_4401


namespace house_floors_l4_4978

theorem house_floors :
  (∀ (d : ℕ) (b : ℕ), b = 3 → d * b * 100 = 18_000) →
  (∀ (total_cost total_floors floors_per_house houses : ℕ),
    total_cost = 270_000 →
    total_floors * 18_000 = total_cost →
    total_floors = floors_per_house * houses →
    houses = 5 →
    floors_per_house = 3) →
  true :=
by
  intros
  sorry

end house_floors_l4_4978


namespace count_pairs_satisfying_conditions_l4_4309

theorem count_pairs_satisfying_conditions :
  ∃ n : ℕ, n = 6 ∧ 
  (∀ a b : ℕ, 0 < a ∧ 0 < b → (
    (a * b * (a + 3) / (a + 3 * b^2) = 7 ∧ a + b ≤ 150) →
    (a, b) ∈ {(21, 1), (42, 2), (63, 3), (84, 4), (105, 5), (126, 6)}
  )) :=
sorry

end count_pairs_satisfying_conditions_l4_4309


namespace perimeter_of_shaded_region_l4_4193

-- Definitions
def point : Type := ℝ × ℝ

def is_equilateral_triangle (B C E : point) : Prop :=
  ∃ (r : ℝ), r = 1 ∧ √((fst C - fst B)^2 + (snd C - snd B)^2) = r 
  ∧ √((fst E - fst B)^2 + (snd E - snd B)^2) = r 
  ∧ √((fst E - fst C)^2 + (snd E - snd C)^2) = r

def radius (B C : point) : ℝ := 1

noncomputable 
def arc_length (angle_rad : ℝ) : ℝ := radius B C * angle_rad

-- Main theorem
theorem perimeter_of_shaded_region (B C E : point) 
  (h_eq_tri : is_equilateral_triangle B C E)
  (h_radius : radius B C = 1) : 
  ∃ p : ℝ, p = 3 := sorry

end perimeter_of_shaded_region_l4_4193


namespace triangle_proof_l4_4432

-- Definition of a point in the plane.
structure Point : Type :=
(x : ℝ)
(y : ℝ)

-- Definitions for the points involved.
variables (A B C M N K : Point)

-- Midpoint condition
def is_midpoint (M A C : Point) : Prop :=
  (M.x = (A.x + C.x) / 2) ∧ (M.y = (A.y + C.y) / 2)

-- Angular condition
noncomputable def angle (P Q R : Point) : ℝ := sorry

-- Given Conditions as Lean Definitions
def problem_conditions : Prop :=
  is_midpoint M A C ∧
  N.x = (A.x + M.x) / 2 ∧ N.y = (A.y + M.y) / 2 ∧ -- N lies on AM such that
  angle M B N = angle C B M ∧
  angle B M K = π / 2

-- Final proof statement to be proved
theorem triangle_proof (h : problem_conditions A B C M N K) : (dist B C) = (dist A K) + (dist B K) :=
sorry

end triangle_proof_l4_4432


namespace val_total_value_l4_4609

theorem val_total_value : 
  let initial_nickels := 20 in 
  let dimes := 3 * initial_nickels in
  let quarters := 2 * dimes in
  let additional_nickels := 2 * initial_nickels in
  let total_nickels := initial_nickels + additional_nickels in
  let value_of_nickels := total_nickels * 0.05 in
  let value_of_dimes := dimes * 0.10 in
  let value_of_quarters := quarters * 0.25 in
  value_of_nickels + value_of_dimes + value_of_quarters = 39.00 :=
by
  sorry

end val_total_value_l4_4609


namespace min_value_fraction_l4_4257

theorem min_value_fraction (a b : ℝ) (hac : a ≠ 0) (hbc : b ≠ 0) : 
  (2 / (∫ x in (-1 : ℝ)..1, Real.sqrt (1 - x^2) / Math.pi) = 1) ∧ 
  (∃ A B : ℝ × ℝ, (sqrt 2 * a * A.1 + b * A.2 = 2) ∧ (sqrt 2 * a * B.1 + b * B.2 = 2) ∧
  (A.1^2 + A.2^2 = 1) ∧ (B.1^2 + B.2^2 = 1) ∧ (A.1 * B.2 - A.2 * B.1 = 1)) ∧ 
  (2 * a^2 + b^2 = 8) → 
  ∃ (a' b' : ℝ), (2 * a'^2 + b'^2 = 8) ∧ (1 / a'^2 + 2 / b'^2 = 1) :=
by
  sorry

end min_value_fraction_l4_4257


namespace dvaneft_shares_percentage_range_l4_4652

theorem dvaneft_shares_percentage_range :
  ∀ (x y z n m : ℝ),
    (4 * x * n = y * m) →
    (x * n + y * m = z * (m + n)) →
    (16 ≤ y - x ∧ y - x ≤ 20) →
    (42 ≤ z ∧ z ≤ 60) →
    (12.5 ≤ (n / (2 * (n + m)) * 100) ∧ (n / (2 * (n + m)) * 100) ≤ 15) :=
by
  intros x y z n m h1 h2 h3 h4
  sorry

end dvaneft_shares_percentage_range_l4_4652


namespace sum_of_binomial_coeffs_equality_l4_4879

theorem sum_of_binomial_coeffs_equality :
  - (Nat.choose 21 11) + (Nat.choose 21 10) = 0 :=
by
  sorry

end sum_of_binomial_coeffs_equality_l4_4879


namespace best_initial_method_best_method_after_adding_grades_method_A_not_best_after_adding_grades_l4_4699

open BigOperators

def initial_grades : List ℕ := [4, 1, 2, 5, 2]
def additional_grades : List ℕ := [5, 5]

def mean (grades : List ℕ) : Float := (grades.sum.toFloat / grades.length.toFloat)
def rounded_mean (grades : List ℕ) : Int := Float.toInt (mean grades).round
def median (grades : List ℕ) : ℕ :=
  let sorted_grades := grades.qsort (· ≤ ·)
  sorted_grades[sorted_grades.length / 2]

theorem best_initial_method : rounded_mean initial_grades = 3 ∧ median initial_grades = 2 :=
  by
  -- proof omitted
  sorry

theorem best_method_after_adding_grades :
  rounded_mean (initial_grades ++ additional_grades) = 3 ∧ median (initial_grades ++ additional_grades) = 4 :=
  by
  -- proof omitted
  sorry

theorem method_A_not_best_after_adding_grades :
  rounded_mean (initial_grades ++ additional_grades) < median (initial_grades ++ additional_grades) :=
  by
  -- proof omitted
  sorry

end best_initial_method_best_method_after_adding_grades_method_A_not_best_after_adding_grades_l4_4699


namespace parabola_b_value_l4_4527

theorem parabola_b_value (a b c p : ℝ) (hp : p ≠ 0) :
  (∀ x, (y = ax^2 + bx + c) → (y = a * (x - p)^2 + p)) ∧ 
  (0, -2p) is y-intercept of parabola → b = 6/p :=
begin
  sorry
end

end parabola_b_value_l4_4527


namespace red_light_wavelength_rounded_l4_4542

def given_data : ℝ := 0.000077
def target_precision : ℝ := 0.00001
def expected_result : ℝ := 8 * 10^(-5)

theorem red_light_wavelength_rounded :
  let rounded_data := Float.round (given_data / target_precision) * target_precision in
  let scientific_notation := rounded_data * 10^5 in
  scientific_notation = expected_result :=
by
  sorry

end red_light_wavelength_rounded_l4_4542


namespace plant_initial_mass_l4_4655

theorem plant_initial_mass (x : ℕ) :
  (27 * x + 52 = 133) → x = 3 :=
by
  intro h
  sorry

end plant_initial_mass_l4_4655


namespace find_number_l4_4303

-- Define the conditions and prove the statement
variable (n : ℝ)
axiom condition_half_plus_six_eq_eleven : (1 / 2) * n + 6 = 11

theorem find_number : n = 10 :=
by
  have h : (1 / 2) * n + 6 = 11 := condition_half_plus_six_eq_eleven
  sorry

end find_number_l4_4303


namespace number_of_ways_correct_l4_4706

noncomputable def number_of_ways := 
  let digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}.to_finset
  ∃ (a b c d : ℕ), 
    a > b ∧ b > c ∧ c > d ∧
    a ≥ 5 ∧ 1 ≤ d ∧ d ≤ 4 ∧
    (digits.count (λ x, x > 1) = a) ∧
    (digits.count (λ x, x > 2) = b) ∧
    (digits.count (λ x, x > 3) = c) ∧
    (digits.count (λ x, x > 4) = d)

theorem number_of_ways_correct : number_of_ways = 2 := 
sorry

end number_of_ways_correct_l4_4706


namespace binomial_17_4_eq_2380_l4_4205

theorem binomial_17_4_eq_2380 : nat.choose 17 4 = 2380 :=
begin
  sorry
end

end binomial_17_4_eq_2380_l4_4205


namespace can_draw_perpendicular_line_l4_4015

def theorem_of_three_perpendiculars (l : ℝ → ℝ × ℝ × ℝ) (floor : ℝ × ℝ → ℝ → ℝ) : Prop :=
∀ (p: ℝ × ℝ × ℝ), ∃ (proj_on_floor : ℝ × ℝ), 
(perpendicular_to_floor : ℝ × ℝ → ℝ → ℝ),
is_perpendicular (floor proj_on_floor) (perpendicular_to_floor proj_on_floor)

noncomputable def any_position_of_pen (pen_orientation : ℝ → ℝ × ℝ × ℝ) : Prop :=
∃ line_on_floor : ℝ × ℝ → ℝ, ∀ point: ℝ, 
(line_on_floor point ≠ pen_orientation point) ∧ theorem_of_three_perpendiculars pen_orientation line_on_floor

theorem can_draw_perpendicular_line (pen : ℝ → ℝ × ℝ × ℝ) 
    (floor : ℝ × ℝ → ℝ → ℝ) 
    (H : any_position_of_pen pen) : ∃ line_on_floor : ℝ × ℝ → ℝ, theorem_of_three_perpendiculars pen line_on_floor := 
by 
  sorry

end can_draw_perpendicular_line_l4_4015


namespace triangle_proof_l4_4426

-- Definition of a point in the plane.
structure Point : Type :=
(x : ℝ)
(y : ℝ)

-- Definitions for the points involved.
variables (A B C M N K : Point)

-- Midpoint condition
def is_midpoint (M A C : Point) : Prop :=
  (M.x = (A.x + C.x) / 2) ∧ (M.y = (A.y + C.y) / 2)

-- Angular condition
noncomputable def angle (P Q R : Point) : ℝ := sorry

-- Given Conditions as Lean Definitions
def problem_conditions : Prop :=
  is_midpoint M A C ∧
  N.x = (A.x + M.x) / 2 ∧ N.y = (A.y + M.y) / 2 ∧ -- N lies on AM such that
  angle M B N = angle C B M ∧
  angle B M K = π / 2

-- Final proof statement to be proved
theorem triangle_proof (h : problem_conditions A B C M N K) : (dist B C) = (dist A K) + (dist B K) :=
sorry

end triangle_proof_l4_4426


namespace relationship_of_x_vals_l4_4829

variables {k x1 x2 x3 : ℝ}

noncomputable def inverse_proportion_function (k x : ℝ) : ℝ := k / x

theorem relationship_of_x_vals (h1 : inverse_proportion_function k x1 = 1)
                              (h2 : inverse_proportion_function k x2 = -5)
                              (h3 : inverse_proportion_function k x3 = 3)
                              (hk : k < 0) :
                              x1 < x3 ∧ x3 < x2 :=
by
  sorry

end relationship_of_x_vals_l4_4829


namespace triangle_ABC_proof_l4_4434

variables {A B C M N K : Type} [inhabited A] [inhabited B] [inhabited C]
variables [between A B C] [within_segment A M] [within_segment M C] [on_segment A M]
variables [angle_eq (M B N) (C B M)] [right_angle (B M K)]
variables [midpoint_of_segment A C M]
variables [extension_of_segment B N K beyond N]

theorem triangle_ABC_proof (hM : midpoint_of_segment A C M)
    (hN : on_segment A M N)
    (hAngle : angle_eq (M B N) (C B M))
    (hPerp : right_angle (B M K))
    (hExt : extension_of_segment B N K beyond N) :
    BC = AK + BK :=
by
  sorry

end triangle_ABC_proof_l4_4434


namespace sequence_integer_iff_t_integer_l4_4520

variables (r s : ℤ) (k : ℝ)

def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = r ∧ a 2 = s ∧ ∀ n, a (n + 2) = (a (n + 1))^2 + k / a n

theorem sequence_integer_iff_t_integer
  (h_r_nonzero : r ≠ 0)
  (h_s_nonzero : s ≠ 0)
  (h_k_positive : 0 < k) :
  (∀ n, ∃ a : ℕ → ℝ, sequence r s k a ∧ ∀ m, a m ∈ ℤ) ↔
  ∃ t : ℤ, t = (r^2 + s^2 + (k : ℤ)) / (r * s) :=
sorry

end sequence_integer_iff_t_integer_l4_4520


namespace cube_root_four_l4_4786

-- Begin by defining the conditions given in the problem
variable {a : ℝ} (x : ℝ)
variable (h_pos : 0 < x)  -- x is positive
variable (h_roots : ∀ r, (r = 3 * a + 1 ∨ r = a + 11) → r^2 = x)

-- The statement we want to prove:
theorem cube_root_four (h : ∀ r, (r = 3 * a + 1 ∨ r = a + 11) → r^2 = x) (h_pos : 0 < x) :
  real.cbrt x = 4 := sorry

end cube_root_four_l4_4786


namespace probability_at_least_one_vowel_l4_4620

-- Define the English alphabet and its properties
def alphabet : List Char := ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
                             'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
                             'U', 'V', 'W', 'X', 'Y', 'Z']

def is_vowel (c : Char) : Prop := c ∈ ['A', 'E', 'I', 'O', 'U']

def consecutive_sets (n : ℕ) (lst : List α) : List (List α) :=
  List.filter_map (λ i, List.take? 4 (list.drop i lst)) (List.range (List.length lst - n + 1))

def no_vowel_sets : List (List Char) :=
  [['B', 'C', 'D', 'F'], ['F', 'G', 'H', 'J'], ['J','K','L','M'],
   ['N','P','Q','R'], ['S','T','V','W']]

-- Total number of 4 consecutive letter sets
def total_sets : ℕ := 23

-- Number of sets without vowels
def sets_without_vowel : ℕ := 5

-- Calculate the desired probability
theorem probability_at_least_one_vowel :
  (total_sets - sets_without_vowel) / total_sets.to_rat = 18 / 23 := by
  sorry

end probability_at_least_one_vowel_l4_4620


namespace grooming_time_equals_640_seconds_l4_4867

variable (cat_claws_per_foot : Nat) (cat_foot_count : Nat)
variable (nissa_clip_time_per_claw : Nat) (nissa_clean_time_per_ear : Nat) (nissa_shampoo_time_minutes : Nat) 
variable (cat_ear_count : Nat)
variable (seconds_per_minute : Nat)

def total_grooming_time (cat_claws_per_foot * cat_foot_count : nissa_clip_time_per_claw) (nissa_clean_time_per_ear * cat_ear_count) (nissa_shampoo_time_minutes * seconds_per_minute) := sorry

theorem grooming_time_equals_640_seconds : 
  cat_claws_per_foot = 4 →
  cat_foot_count = 4 →
  nissa_clip_time_per_claw = 10 →
  nissa_clean_time_per_ear = 90 →
  nissa_shampoo_time_minutes = 5 →
  cat_ear_count = 2 →
  seconds_per_minute = 60 →
  total_grooming_time = 160 + 180 + 300 → 
  total_grooming_time = 640 := sorry

end grooming_time_equals_640_seconds_l4_4867


namespace ann_independent_work_time_l4_4617

noncomputable def ann_time : ℝ :=
  let tina_time : ℝ := 12 in  -- Tina can do the job in 12 hours
  let tina_work : ℝ := 8 in    -- Tina worked for 8 hours
  let ann_work : ℝ := 3 in     -- Ann took 3 hours to complete the remaining job
  have tina_rate : ℝ := 1 / tina_time, by sorry, -- Tina's work rate
  have tina_done : ℝ := tina_rate * tina_work, by sorry, -- Work done by Tina
  have remaining_job : ℝ := 1 - tina_done, by sorry, -- Remaining job for Ann
  have ann_rate : ℝ := remaining_job / ann_work, by sorry, -- Ann's work rate
  have answer : ℝ := 1 / ann_rate, by sorry,  -- Time Ann needs to complete the job
  answer

theorem ann_independent_work_time : ann_time = 9 := by
  sorry

end ann_independent_work_time_l4_4617


namespace part1_part2_l4_4143

-- Part (1)
theorem part1 (k : ℝ) (n : ℝ) (P : ℝ × ℝ) : 
  P = (3, 2) → 
  n = 3 → 
  ∃ k, 2 = k * 3 + 3 ∧ k = -(1 : ℝ) / 3 :=
sorry

-- Part (2)
theorem part2 (n : ℕ) (f : ℝ → ℝ) :
  f = (λ x, if x ≥ 3 then -x^2 + 2*x + 4 + (n : ℝ) else -x^2 + 2*x + 4 - (n : ℝ)) →
  (1 < n ∧ n < 5) →
  ∃ x1 x2 x3,
  x1 = -2 ∧ x2 = 4 ∧ x3 = 4 ∧ x1 + x2 + x3 = 6 :=
sorry

end part1_part2_l4_4143


namespace nell_ace_cards_l4_4004

theorem nell_ace_cards (baseball_cards_original : ℕ) (ace_cards_original : ℕ)
(baseball_cards_left : ℕ) (ace_baseball_diff : ℕ)
(h1 : baseball_cards_original = 239)
(h2 : ace_cards_original = 38)
(h3 : baseball_cards_left = 111)
(h4 : ace_baseball_diff = 265) :
  let A := baseball_cards_left + ace_baseball_diff in A = 376 :=
by
  let A := baseball_cards_left + ace_baseball_diff
  have h5 : A = 111 + 265 := by rfl
  have h6 : A = 376 := by { rw h5, norm_num }
  exact h6

end nell_ace_cards_l4_4004


namespace white_animals_count_l4_4007

-- Definitions
def total : ℕ := 13
def black : ℕ := 6
def white : ℕ := total - black

-- Theorem stating the number of white animals
theorem white_animals_count : white = 7 :=
by {
  -- The proof would go here, but we'll use sorry to skip it.
  sorry
}

end white_animals_count_l4_4007


namespace sample_data_set_std_dev_l4_4298

theorem sample_data_set_std_dev :
  let data := [3, 3, 4, 4, 5, 6, 6, 7, 7]
  let n := data.length
  let mean := (data.sum : ℝ) / n
  let variance := (data.map (λ x, (x - mean)^2)).sum / n
  let std_dev := real.sqrt variance
  std_dev = 2 * real.sqrt 5 / 3 :=
by
  let data := [3, 3, 4, 4, 5, 6, 6, 7, 7]
  let n := data.length
  let mean := (data.sum : ℝ) / n
  let variance := (data.map (λ x, (x - mean)^2)).sum / n
  let std_dev := real.sqrt variance
  show std_dev = 2 * real.sqrt 5 / 3
  sorry

end sample_data_set_std_dev_l4_4298


namespace cleaning_times_l4_4187

theorem cleaning_times (alice_time : ℕ) (bob_ratio : ℚ) (charlie_ratio : ℚ)
  (h_alice : alice_time = 40)
  (h_bob_ratio : bob_ratio = 1/4)
  (h_charlie_ratio : charlie_ratio = 3/8) :
  let bob_time := bob_ratio * alice_time,
      charlie_time := charlie_ratio * alice_time
  in bob_time = 10 ∧ charlie_time = 15 := by
  sorry

end cleaning_times_l4_4187


namespace range_of_a_to_have_two_distinct_zeros_l4_4955

def f (x a : ℝ) : ℝ := 2^(x - 1) - real.log x - a

theorem range_of_a_to_have_two_distinct_zeros :
  ∀ a : ℝ, (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0) ↔ 1 < a :=
sorry

end range_of_a_to_have_two_distinct_zeros_l4_4955


namespace polynomial_abs_sum_l4_4253

theorem polynomial_abs_sum {a : ℕ → ℤ} (h : (λ x : ℤ, (2 - x)^2023) = (λ x : ℤ, ∑ i in finset.range 2024, a i * (x + 1)^i)) :
  (finset.range 2024).sum (λ i, |a i|) = 2^4046 :=
by
  sorry

end polynomial_abs_sum_l4_4253


namespace brick_tower_height_variations_l4_4272

theorem brick_tower_height_variations : 
  let bricks := 88
  let min_height := 3 * bricks
  let max_height := 18 * bricks
  let increment_small := 8 - 3
  let increment_large := 18 - 3
  let height_difference := max_height - min_height
  let steps := height_difference / increment_small
  in (steps + 1 = 265) :=
by
  let bricks := 88
  let min_height := 3 * bricks
  let max_height := 18 * bricks
  let increment_small := 8 - 3
  let increment_large := 18 - 3
  let height_difference := max_height - min_height
  let steps := height_difference / increment_small
  sorry

end brick_tower_height_variations_l4_4272


namespace triangle_ratios_equiv_l4_4877

theorem triangle_ratios_equiv
  (A B C X Y Z A₁ A₂ B₁ B₂ C₁ C₂ : Type)
  [LocallyCompactSpace A] [LocallyCompactSpace B] [LocallyCompactSpace C]
  [LocallyCompactSpace X] [LocallyCompactSpace Y] [LocallyCompactSpace Z]
  [LocallyCompactSpace A₁] [LocallyCompactSpace A₂]
  [LocallyCompactSpace B₁] [LocallyCompactSpace B₂]
  [LocallyCompactSpace C₁] [LocallyCompactSpace C₂]
  (BC ZX XY : Line) 
  (CA YZ : Line)
  (AB YZ' : Line)
  (BC_ZX : Intersect BC ZX A₁)
  (BC_XY : Intersect BC XY A₂)
  (CA_XY : Intersect CA XY B₁)
  (CA_YZ : Intersect CA YZ B₂)
  (AB_YZ : Intersect AB YZ C₁)
  (AB_ZX : Intersect AB ZX C₂) :
  (C₁C₂.length / AB.length = A₁A₂.length / BC.length ∧ A₁A₂.length / BC.length = B₁B₂.length / CA.length) ↔
  (A₁C₂.length / XZ.length = C₁B₂.length / ZY.length ∧ C₁B₂.length / ZY.length = B₁A₂.length / YX.length) :=
by
  sorry

end triangle_ratios_equiv_l4_4877


namespace hexagonal_pyramid_has_7_vertices_l4_4820

-- Definition of a pyramid with a regular hexagonal base
def hexagonal_pyramid_vertices : ℕ :=
  let base_vertices := 6   -- Hexagon has 6 vertices
  let apex_vertex := 1     -- Top vertex
  base_vertices + apex_vertex

-- Proof Statement: The number of vertices in a hexagonal pyramid is 7
theorem hexagonal_pyramid_has_7_vertices : hexagonal_pyramid_vertices = 7 :=
by
  unfold hexagonal_pyramid_vertices
  simp
  sorry

end hexagonal_pyramid_has_7_vertices_l4_4820


namespace range_of_a_l4_4400

theorem range_of_a (a : ℝ) (h₁ : 1/2 ≤ 1) (h₂ : a ≤ a + 1)
    (h_condition : ∀ x:ℝ, (1/2 ≤ x ∧ x ≤ 1) → (a ≤ x ∧ x ≤ a + 1)) :
  0 ≤ a ∧ a ≤ 1/2 :=
by
  sorry

end range_of_a_l4_4400


namespace sum_second_third_smallest_l4_4215

theorem sum_second_third_smallest :
  let digits := [1, 6, 8] in
  let combinations := 
    [ 100 * digits[0] + 10 * digits[1] + digits[2],
      100 * digits[0] + 10 * digits[2] + digits[1],
      100 * digits[1] + 10 * digits[0] + digits[2],
      100 * digits[1] + 10 * digits[2] + digits[0],
      100 * digits[2] + 10 * digits[0] + digits[1],
      100 * digits[2] + 10 * digits[1] + digits[0] ]
  in
  let sorted := List.sort combinations (λ x y => x < y) in
  sorted[1] + sorted[2] = 804 :=
by
  let digits := [1, 6, 8]
  let combinations := 
    [ 100 * digits[0] + 10 * digits[1] + digits[2],
      100 * digits[0] + 10 * digits[2] + digits[1],
      100 * digits[1] + 10 * digits[0] + digits[2],
      100 * digits[1] + 10 * digits[2] + digits[0],
      100 * digits[2] + 10 * digits[0] + digits[1],
      100 * digits[2] + 10 * digits[1] + digits[0] ]
  let sorted := List.sort combinations (λ x y => x < y)
  sorry

end sum_second_third_smallest_l4_4215


namespace triangle_property_l4_4446

variables {ABC : Type*} [inner_product_space ℝ ABC]
variables (A B C M K N : ABC)

noncomputable def midpoint (A C : ABC) (M : ABC) : Prop := dist A M = dist M C

noncomputable def angle_equality (B M N : ABC) : Prop := ∠M B N = ∠C B M

noncomputable def right_angle (B M K : ABC) : Prop := ∠B M K = π / 2

theorem triangle_property (h1 : midpoint A C M) (h2 : angle_equality B M N) (h3 : right_angle B M K) : dist B C = dist A K + dist B K :=
sorry

end triangle_property_l4_4446


namespace percentage_of_oil_in_mixture_l4_4920

theorem percentage_of_oil_in_mixture :
  ∀ (initial_oil initial_vinegar added_oil : ℕ), 
  initial_oil = 30 → 
  initial_vinegar = 15 → 
  added_oil = 15 → 
  (initial_oil + added_oil) * 100 / (initial_oil + initial_vinegar + added_oil) = 75 :=
by
  intros initial_oil initial_vinegar added_oil hoil hvinegar hadded
  rw [hoil, hvinegar, hadded]
  sorry

end percentage_of_oil_in_mixture_l4_4920


namespace find_moles_of_benzene_l4_4989

-- Definitions for atomic weights of Carbon and Hydrogen
def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008

-- Definition for molecular formula of Benzene (C6H6)
def molecular_weight_benzene : ℝ := 6 * atomic_weight_C + 6 * atomic_weight_H

-- Given molecular weight of Benzene corresponds to 312 g
def given_molecular_weight : ℝ := 312

-- Lean statement to prove the number of moles 'n' for 312 g is approximately 3.994
theorem find_moles_of_benzene (n : ℝ) (h : given_molecular_weight = n * molecular_weight_benzene) :
  n ≈ 3.994 :=
sorry

end find_moles_of_benzene_l4_4989


namespace number_of_elements_in_union_l4_4881

open Set

noncomputable def A (a : ℝ) : Set ℝ := {x | (x - 2) * (x + a) = 0}
noncomputable def B : Set ℕ := {x | x ∈ {2, 3, 4}}

theorem number_of_elements_in_union (a : ℝ) : 
  (if a = -2 then 3 else 4) = 4 :=
  sorry

end number_of_elements_in_union_l4_4881


namespace boats_distance_three_minutes_before_collision_l4_4099

variables (current_speed boat1_speed boat2_speed : ℝ)
variables (initial_distance time_before_collision : ℝ)

-- Defining the conditions
def conditions : Prop :=
  (current_speed = 2) ∧
  (boat1_speed = 5) ∧
  (boat2_speed = 25) ∧
  (initial_distance = 20) ∧
  (time_before_collision = 3 / 60)

-- Defining the speeds relative to the ground
def boat1_ground_speed := boat1_speed - current_speed
def boat2_ground_speed := boat2_speed - current_speed

-- Defining the relative speed of the boats
def relative_speed := boat1_ground_speed + boat2_ground_speed

-- Converting minutes to hours
def time_in_hours := 3 / 60

-- Calculating the distance covered in three minutes
def distance_covered := relative_speed * time_in_hours

-- Main statement to prove
theorem boats_distance_three_minutes_before_collision :
  conditions →
  distance_covered = 1.3 :=
by
  intros,
  sorry

end boats_distance_three_minutes_before_collision_l4_4099


namespace point_of_symmetry_is_1_0_l4_4718

def g (x : ℝ) : ℝ := |⌊x⌋| - |⌊2 - x⌋|

theorem point_of_symmetry_is_1_0 : ∀ x : ℝ, g(2 - x) = g(x) ↔ x = 1 :=
sorry

end point_of_symmetry_is_1_0_l4_4718


namespace minimum_value_of_a_l4_4338

theorem minimum_value_of_a :
  (∃ x : ℝ, x ∈ Icc (-1 : ℝ) 1 ∧ 1 + 2^x + a * 4^x ≥ 0) ↔ a >= -6 := 
begin
  sorry
end

end minimum_value_of_a_l4_4338


namespace triangle_ABC_proof_l4_4441

variables {A B C M N K : Type} [inhabited A] [inhabited B] [inhabited C]
variables [between A B C] [within_segment A M] [within_segment M C] [on_segment A M]
variables [angle_eq (M B N) (C B M)] [right_angle (B M K)]
variables [midpoint_of_segment A C M]
variables [extension_of_segment B N K beyond N]

theorem triangle_ABC_proof (hM : midpoint_of_segment A C M)
    (hN : on_segment A M N)
    (hAngle : angle_eq (M B N) (C B M))
    (hPerp : right_angle (B M K))
    (hExt : extension_of_segment B N K beyond N) :
    BC = AK + BK :=
by
  sorry

end triangle_ABC_proof_l4_4441


namespace opposite_of_neg_quarter_l4_4966

theorem opposite_of_neg_quarter : -(- (1/4 : ℝ)) = (1/4 : ℝ) :=
by
  sorry

end opposite_of_neg_quarter_l4_4966


namespace count_arrangements_of_books_l4_4079

-- Definitions based on the conditions
def num_books : ℕ := 6
def num_chinese_books : ℕ := 3
def num_math_books : ℕ := 3
def first_chinese_book_not_at_ends : Prop := true

-- Define the problem as a theorem in Lean
theorem count_arrangements_of_books 
  (n_books : ℕ := num_books)
  (n_chinese_books : ℕ := num_chinese_books)
  (n_math_books : ℕ := num_math_books)
  (condition : first_chinese_book_not_at_ends) :
  n_books = 6 ∧ n_chinese_books = 3 ∧ n_math_books = 3 ∧ condition →
  (∃ arrangements : ℕ, arrangements = 288) :=
by
  -- The proof steps would go here
  sorry

end count_arrangements_of_books_l4_4079


namespace temperature_at_midnight_l4_4343

-- Define temperature in the morning
def T_morning := -2 -- in degrees Celsius

-- Temperature change at noon
def delta_noon := 12 -- in degrees Celsius

-- Temperature change at midnight
def delta_midnight := -8 -- in degrees Celsius

-- Function to compute temperature
def compute_temperature (T : ℤ) (delta1 : ℤ) (delta2 : ℤ) : ℤ :=
  T + delta1 + delta2

-- The proposition to prove
theorem temperature_at_midnight :
  compute_temperature T_morning delta_noon delta_midnight = 2 :=
by
  sorry

end temperature_at_midnight_l4_4343


namespace form_square_from_cut_squares_l4_4302

theorem form_square_from_cut_squares :
  let a1 := 2 * 2
  let a2 := 3 * 3
  let a3 := 6 * 6
  a1 + a2 + a3 = 7 * 7 := by
  let a1 := 2 * 2
  let a2 := 3 * 3
  let a3 := 6 * 6
  calc a1 + a2 + a3 = 4 + 9 + 36 := by rfl
                 ... = 49 := by rfl
                 ... = 7 * 7 := by rfl
  sorry

end form_square_from_cut_squares_l4_4302


namespace angle_diff_complement_supplement_l4_4046

theorem angle_diff_complement_supplement (α : ℝ) : (180 - α) - (90 - α) = 90 := by
  sorry

end angle_diff_complement_supplement_l4_4046


namespace abs_gt_3_sufficient_but_not_necessary_for_x_x_minus_3_gt_0_l4_4142

theorem abs_gt_3_sufficient_but_not_necessary_for_x_x_minus_3_gt_0 :
  (∀ x : ℝ, |x| > 3 → x * (x - 3) > 0) ∧ (∃ x : ℝ, x * (x - 3) > 0 ∧ ¬(|x| > 3)) :=
begin
  sorry
end

end abs_gt_3_sufficient_but_not_necessary_for_x_x_minus_3_gt_0_l4_4142


namespace function_identity_l4_4459

variable (f : ℕ+ → ℕ+)

theorem function_identity (h : ∀ n : ℕ+, f (n + 1) > f (f n)) : ∀ n : ℕ+, f n = n := sorry

end function_identity_l4_4459


namespace area_of_transformed_parallelogram_l4_4043

variables (u v : ℝ^3)
-- Conditions
def parallelogram_area_condition : Prop := ∥u × v∥ = 12

-- Problem statement
theorem area_of_transformed_parallelogram (huv : parallelogram_area_condition u v) :
  ∥(3 • u - 2 • v) × (4 • u + v)∥ = 132 :=
sorry

end area_of_transformed_parallelogram_l4_4043


namespace statement_D_correct_l4_4996

theorem statement_D_correct :
  (∀ (x y : ℝ) (i : ℕ) (x_i y_i: list (ℝ × ℝ)) (a b : ℝ),
  (a > b → a^2 > b^2) ∧
  (∀ x : ℝ, 2^x > 0) ∧ ∃ x_0 : ℝ, 2^x_0 < 0 ∧
  (|correlation_coefficient x y| = |1|) ∧
  (∀ (x y : ℝ) (x_i x_f y_i y_f : ℝ),
    regression_equation (x_i, y_i) = 2*x_f - 0.4 ∧ mean x = 2 ∧
    new_regression_slope (remove_samples (x_i, y_i) [(-3, 1), (3, -1)] = 3 → 
    new_regression_equation (remove_samples (x_i, y_i) [(-3, 1), (3, -1)] = 3*x_f - 3))
  → statement_D_correct
sorry

end statement_D_correct_l4_4996


namespace projection_is_correct_l4_4969

theorem projection_is_correct :
  let v : ℝ × ℝ × ℝ := (2, -1, 1)
  let u1 : ℝ × ℝ × ℝ := (1, 2, 3)
  let p1 : ℝ × ℝ × ℝ := (2, -1, 1)
  let u2 : ℝ × ℝ × ℝ := (4, 1, -3)
  let expected_projection : ℝ × ℝ × ℝ := (4/3, -2/3, 2/3)
  ((u1.1 * v.1 + u1.2 * v.2 + u1.3 * v.3) / (v.1 * v.1 + v.2 * v.2 + v.3 * v.3) * v.1 = p1.1) →
  ((u1.1 * v.1 + u1.2 * v.2 + u1.3 * v.3) / (v.1 * v.1 + v.2 * v.2 + v.3 * v.3) * v.2 = p1.2) →
  ((u1.1 * v.1 + u1.2 * v.2 + u1.3 * v.3) / (v.1 * v.1 + v.2 * v.2 + v.3 * v.3) * v.3 = p1.3) →
  ((u2.1 * v.1 + u2.2 * v.2 + u2.3 * v.3) / (v.1 * v.1 + v.2 * v.2 + v.3 * v.3) * v.1 = expected_projection.1) ∧
  ((u2.1 * v.1 + u2.2 * v.2 + u2.3 * v.3) / (v.1 * v.1 + v.2 * v.2 + v.3 * v.3) * v.2 = expected_projection.2) ∧
  ((u2.1 * v.1 + u2.2 * v.2 + u2.3 * v.3) / (v.1 * v.1 + v.2 * v.2 + v.3 * v.3) * v.3 = expected_projection.3) :=
by
  intros
  sorry

end projection_is_correct_l4_4969


namespace Seokhyung_drank_the_most_l4_4472

-- Define the conditions
def Mina_Amount := 0.6
def Seokhyung_Amount := 1.5
def Songhwa_Amount := Seokhyung_Amount - 0.6

-- Statement to prove that Seokhyung drank the most cola
theorem Seokhyung_drank_the_most : Seokhyung_Amount > Mina_Amount ∧ Seokhyung_Amount > Songhwa_Amount :=
by
  -- Proof skipped
  sorry

end Seokhyung_drank_the_most_l4_4472


namespace total_length_of_T_l4_4380

def T : set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |(|p.1| - 3)| - 2| + |(|p.2| - 3)| - 2| = 2}

theorem total_length_of_T : 
  let L := 128 * Real.sqrt 2 in
  ∑ p in T, length_of_line_making_up_T = L := 
by sorry

end total_length_of_T_l4_4380


namespace length_of_BC_is_four_cbrt_two_l4_4188

theorem length_of_BC_is_four_cbrt_two
  (h1 : ∃ A B C : ℝ × ℝ, A = (0, 0) ∧
      (B.2 = 4 * B.1^2) ∧ (C.2 = 4 * C.1^2) ∧ 
      (A, B, C ∈ set_of (λ p, p.2 = 4 * p.1^2)) ∧
      (B.1 ≠ 0) ∧ (B.2 = C.2) ∧ 
      (A ≠ B) ∧ (A ≠ C)) 
  (h2 : (by simp [triangle_area (0 : ℝ) B.1 C.1, eq_div_2] : 64 = 2 * A.2 * (4 * B.1^2))) :
  ∃ a : ℝ, (4 * a^3 = 128) → 2 * a = 4*real.cbrt 2 := 
by {
  sorry
}

end length_of_BC_is_four_cbrt_two_l4_4188


namespace orthocenter_of_triangle_l4_4845

open EuclideanGeometry

noncomputable def orthocenter_triangle_ABC : Point ℝ :=
  let A : Point ℝ := (2, 3, 1)
  let B : Point ℝ := (4, -1, 5)
  let C : Point ℝ := (1, 5, 2)
  let H : Point ℝ := (5/3, 29/3, 8/3)
  in H

-- Statement of the proof problem
theorem orthocenter_of_triangle :
  let A : Point ℝ := (2, 3, 1)
  let B : Point ℝ := (4, -1, 5)
  let C : Point ℝ := (1, 5, 2)
  let H : Point ℝ := (5/3, 29/3, 8/3)
  orthocenter_triangle_ABC A B C = H := by
  sorry -- Proof omitted

end orthocenter_of_triangle_l4_4845


namespace power_modulo_l4_4111

theorem power_modulo {n : ℕ} : 
  let base := 5 in
  let exp := 2006 in
  let modulus := 13 in
  (base ^ exp) % modulus = 12 :=
by
  let base := 5
  let exp := 2006
  let modulus := 13
  have h1: (base ^ exp) % modulus = 5 ^ 2006 % 13, from sorry
  have h2: 2006 % 4 = 2, from sorry
  have h3: (base ^ exp) % modulus = (5 ^ 2) % 13, from sorry
  show (5 ^ 2006) % 13 = 12, from sorry

end power_modulo_l4_4111


namespace circle_tangency_angle_bisector_l4_4983

-- Define the conditions and the final statement to be proved.

theorem circle_tangency_angle_bisector (LargerCircle SmallerCircle : Circle) (M T A B : Point)
  (h_tangent_internal : TouchesInternally LargerCircle SmallerCircle M) -- condition 1
  (h_chord_larger : IsChordOfCircle LargerCircle AB) -- condition 2
  (h_touch_smaller : Touches SmallerCircle AB T) : -- condition 2

  (IsAngleBisector M T (Angle A M B)) :=      -- question to prove; conclusion
sorry

end circle_tangency_angle_bisector_l4_4983


namespace triangle_equality_BC_AK_BK_l4_4416

-- Define the main geometric constructs
variables {A B C M N K : Type}

-- Given conditions
variables [Midpoint M A C] -- M is the midpoint of AC
variables [OnSegment N A M] -- N lies on segment AM
variables [EqualAngle M B N C B M] -- ∠MBN = ∠CBM
variables [OnExtension K B N (90 : Angle)] -- K is on the extension of BN such that ∠BMK is right

-- Define the main theorem
theorem triangle_equality_BC_AK_BK :
  BC = AK + BK :=
sorry

end triangle_equality_BC_AK_BK_l4_4416


namespace modulus_of_complex_number_l4_4262

theorem modulus_of_complex_number : 
  let i := complex.I in
  let z := (1 - i^3) * (1 + 2 * i) in
  complex.abs z = real.sqrt 10 := 
by
  sorry

end modulus_of_complex_number_l4_4262


namespace number_of_valid_permutations_l4_4306

noncomputable def count_valid_permutations : Nat :=
  let multiples_of_77 := [154, 231, 308, 385, 462, 539, 616, 693, 770, 847, 924]
  let total_count := multiples_of_77.foldl (fun acc x =>
    if x == 770 then
      acc + 3
    else if x == 308 then
      acc + 6 - 2
    else
      acc + 6) 0
  total_count

theorem number_of_valid_permutations : count_valid_permutations = 61 :=
  sorry

end number_of_valid_permutations_l4_4306


namespace expectation_transform_l4_4466

-- Given E(X) = 6, prove E[3(X - 2)] = 12
variable (X : ℝ) -- X is a real-valued random variable
variable [has_Expectation X] -- X has an expectation defined

-- hypothesis or condition
axiom E_X_eq_6 : E[X] = 6

-- Proof goal
theorem expectation_transform : E[3 * (X - 2)] = 12 :=
by
  sorry -- The actual proof will be done here

end expectation_transform_l4_4466


namespace log_shift_fixed_point_l4_4006

theorem log_shift_fixed_point (a : ℝ) (h : 0 < a ∧ a ≠ 1) : 
  1 + log_a (2 - 1) = 1 :=
by
  sorry

end log_shift_fixed_point_l4_4006


namespace sum_of_solutions_l4_4123

-- Define the quadratic equation as a product of linear factors
def quadratic_eq (x : ℚ) : Prop := (4 * x + 6) * (3 * x - 8) = 0

-- Define the roots of the quadratic equation
def root1 : ℚ := -3 / 2
def root2 : ℚ := 8 / 3

-- Sum of the roots of the quadratic equation
def sum_of_roots : ℚ := root1 + root2

-- Theorem stating that the sum of the roots is 7/6
theorem sum_of_solutions : sum_of_roots = 7 / 6 := by
  sorry

end sum_of_solutions_l4_4123


namespace tangent_line_with_smallest_slope_l4_4793

-- Define the curve
def curve (x : ℝ) : ℝ := x^3 + 3 * x^2 + 6 * x + 4

-- The lean statement for the proof problem
theorem tangent_line_with_smallest_slope :
  ∃ (m b : ℝ), (∀ x, deriv curve x = m → m = 3 ∧ b = -3) ∧
  (∀ x, deriv curve x > m := λ _ → 3) ∧
  ∃ (x y : ℝ), y = curve x ∧ x = -1 ∧ y = 0 ∧
  3 * x - y + 3 = 0 :=
by
  sorry

end tangent_line_with_smallest_slope_l4_4793


namespace johns_pieces_of_gum_l4_4874

theorem johns_pieces_of_gum : 
  (∃ (john cole aubrey : ℕ), 
    cole = 45 ∧ 
    aubrey = 0 ∧ 
    (john + cole + aubrey) = 3 * 33) → 
  ∃ john : ℕ, john = 54 :=
by 
  sorry

end johns_pieces_of_gum_l4_4874


namespace even_positive_factors_of_n_l4_4898

-- Define the given value of n
def n : ℕ := 2^4 * 3^3 * 5 * 7

-- State the theorem to prove the number of even positive factors of n
theorem even_positive_factors_of_n : (finset.range (4 + 1)).card * (finset.range (3 + 1)).card * (finset.range (1 + 1)).card * (finset.range (1 + 1)).card = 64 := by
  sorry

end even_positive_factors_of_n_l4_4898


namespace exists_P_n_on_arc_l4_4901

open Set

variables {α : Type*} [TopologicalSpace α] {O : α} {C : Set α}
variable {P_0 : α}
variable {P : α}
variable {Q : α}

def is_center (O : α) (C : Set α) : Prop := True -- Define is_center as a placeholder, in practice, you would use geometric definitions or proofs.

def on_circle (P : α) (C : Set α) : Prop := True -- Define on_circle as a placeholder, similar to above.

def angle_condition (P_n P_(n-1) O : α) : Prop := True -- Define angle_condition as a placeholder.

theorem exists_P_n_on_arc (O : α) (C : Set α) (P_0 : α)
  (hO_center : is_center O C) (hP_0_on_circle : on_circle P_0 C)
  (theta_rational : irrational π)
  (hAngle : ∀ n : ℤ, on_circle (P_n n) C ∧ angle_condition (P_n (n+1)) (P_n n) O)
  (P Q : α) (hP_on_circle : on_circle P C) (hQ_on_circle : on_circle Q C) (hDistinct : P ≠ Q):
  ∃ n : ℤ, arc_contains_point P Q (P_n n) := sorry

end exists_P_n_on_arc_l4_4901


namespace judah_goals_less_twice_shelby_l4_4705

def carter_goals := 4
def shelby_goals := carter_goals / 2
def team_total_goals := 7
def judah_goals := 2 * shelby_goals - X

theorem judah_goals_less_twice_shelby :
  carter_goals + shelby_goals + judah_goals = team_total_goals → 
  (∃ X, judah_goals = 4 - X ∧ X = 3) :=
by
  intro h
  have hs: shelby_goals = 2 := by norm_num
  have hj: judah_goals = 4 - X := by sorry
  have ht: carter_goals + shelby_goals + judah_goals = 7 := h
  sorry

end judah_goals_less_twice_shelby_l4_4705


namespace principal_period_function_l4_4232

noncomputable def f (x : ℝ) : ℝ := (Real.arcsin (Real.sin (Real.arccos (Real.cos (3 * x))))) ^ (-5)

theorem principal_period_function :
  ∀ x : ℝ, f (x + π/3) = f x :=
sorry

end principal_period_function_l4_4232


namespace domain_of_function_l4_4951

theorem domain_of_function :
  { x : ℝ | x + 2 ≥ 0 ∧ x - 1 ≠ 0 } = { x : ℝ | x ≥ -2 ∧ x ≠ 1 } :=
by
  sorry

end domain_of_function_l4_4951


namespace ellipse_problem_l4_4050

-- Definitions of points, ellipse properties, and line tangency conditions

noncomputable def ellipse_standard_eq (a b : ℝ) (h : a > 0 ∧ b > 0 ∧ a > b) : Prop :=
  ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 ↔ (x, y) = (sqrt 2, sqrt 3)

noncomputable def ellipse_passes_point (a b : ℝ) (h : a > 0 ∧ b > 0 ∧ a > b) : Prop :=
  (sqrt 2)^2 / a^2 + (sqrt 3)^2 / b^2 = 1

noncomputable def ellipse_eccentricity (a b : ℝ) (h : a > 0 ∧ b > 0 ∧ a > b) : Prop :=
  let c := sqrt (a^2 - b^2) in
  c / a = sqrt 2 / 2

noncomputable def line_eq (a b : ℝ) (h : a > 0 ∧ b > 0 ∧ a > b) : Prop :=
  A = (0, -2) →
  (∀x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 → B) →
  F2 = (a, 0) ∧ 3 • M = F2 ∧
  M = (a / 3, 0) →
  ∀ x y : ℝ, (x - y = 2 ∨ x - 2 * y = 4)

theorem ellipse_problem :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a > b ∧
  ellipse_standard_eq a b ∧
  ellipse_passes_point a b ∧
  ellipse_eccentricity a b ∧
  line_eq a b :=
sorry

end ellipse_problem_l4_4050


namespace crayons_in_box_l4_4012

variable {a b c : ℕ}

def initial_crayons (a b c : ℕ) : ℕ := a + b

theorem crayons_in_box
    (h1 : a = 571)
    (h2 : b = 161)
    (h3 : c = 410)
    (h4 : a - b = c) :
    initial_crayons a b c = 732 := by
    rw [initial_crayons, h1, h2]
    rfl

end crayons_in_box_l4_4012


namespace min_value_of_f_l4_4229

noncomputable def f (x : ℝ) : ℝ :=
  ∑ k in finset.range 51, (x - (2 * k + 1))^2

theorem min_value_of_f :
  ∃ x : ℝ, f 51 = 44200 :=
begin
  sorry
end

end min_value_of_f_l4_4229


namespace triangle_ABC_proof_l4_4436

variables {A B C M N K : Type} [inhabited A] [inhabited B] [inhabited C]
variables [between A B C] [within_segment A M] [within_segment M C] [on_segment A M]
variables [angle_eq (M B N) (C B M)] [right_angle (B M K)]
variables [midpoint_of_segment A C M]
variables [extension_of_segment B N K beyond N]

theorem triangle_ABC_proof (hM : midpoint_of_segment A C M)
    (hN : on_segment A M N)
    (hAngle : angle_eq (M B N) (C B M))
    (hPerp : right_angle (B M K))
    (hExt : extension_of_segment B N K beyond N) :
    BC = AK + BK :=
by
  sorry

end triangle_ABC_proof_l4_4436


namespace boxes_per_crate_l4_4349

theorem boxes_per_crate (num_crates : ℕ) (wm_per_box : ℕ) (wm_removed : ℕ) (total_wm_removed : ℕ) (boxes_per_crate : ℕ) :
  num_crates = 10 →
  wm_per_box = 4 →
  wm_removed = 1 →
  total_wm_removed = 60 →
  boxes_per_crate = total_wm_removed / num_crates :=
begin
  sorry
end

end boxes_per_crate_l4_4349


namespace range_of_a_l4_4832

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, (a - 1) * x < a - 1 ↔ x > 1) : a < 1 := 
sorry

end range_of_a_l4_4832


namespace not_cheap_necessary_but_not_sufficient_l4_4532

-- Definitions for conditions and questions based on logical relationships
def not_cheap := Prop
def good_quality_product := Prop

-- Necessary condition means: good_quality_product implies not_cheap.
-- Not sufficient condition means: we can't deduce good_quality_product from not_cheap.

theorem not_cheap_necessary_but_not_sufficient (h₁ : good_quality_product → not_cheap) (h₂ : ¬ (not_cheap → good_quality_product)):
  ∃ (not_cheap : Prop) (good_quality_product : Prop), (good_quality_product → not_cheap) ∧ ¬ (not_cheap → good_quality_product) :=
by
  sorry

end not_cheap_necessary_but_not_sufficient_l4_4532


namespace parabola_circle_properties_l4_4283

theorem parabola_circle_properties :
  ∀ (a : ℝ) (P Q : ℝ × ℝ) (M : ℝ × ℝ),
  let C : set (ℝ × ℝ) := {p | p.2 ^ 2 = 4 * p.1},
      T : set (ℝ × ℝ) := {p | (p.1 + 2) ^ 2 + (p.2 + 7) ^ 2 = a ^ 2},
      F : ℝ × ℝ := (1, 0),
      l : set (ℝ × ℝ) := {p | p.2 = -p.1 + 1},
      D : set (ℝ × ℝ) := {p | (p.1 - 3) ^ 2 + (p.2 + 2) ^ 2 = 16},
      PQ_angle := ∃ P Q, angle (M - P) (Q - M) = π / 2 in
  (∀ A B ∈ l ∩ C, ∃ D_eq : (D = (λ x y, (x - 3)^2 + (y + 2)^2 = 16)),
  (∃ M ∈ T, PQ_angle → (\(\sqrt{2}, 9 * sqrt 2))) :=
sorry

end parabola_circle_properties_l4_4283


namespace probability_valid_assignment_l4_4526

-- Define a type for Faces and Numbers
structure Dodecahedron :=
(faces : Fin 12 → ℕ) -- 12 faces with numbers 1 to 12 assigned uniquely to each face

-- Define adjacency relation
def is_adjacent (d : Dodecahedron) (i j : Fin 12) : Prop := sorry  -- needs dodecahedron's adjacency relation

-- Define directly opposite relation
def is_opposite (d : Dodecahedron) (i j : Fin 12) : Prop := sorry -- needs dodecahedron's opposite relation

-- Define a valid assignment
def valid_assignment (d : Dodecahedron) : Prop :=
∀ i j, (is_adjacent d i j ∨ is_opposite d i j) → ¬ (nat.succ d.faces i = d.faces j ∨ d.faces i = nat.succ d.faces j ∨ (d.faces i = 12 ∧ d.faces j = 1) ∨ (d.faces i = 1 ∧ d.faces j = 12))

-- Define the problem statement
theorem probability_valid_assignment : 
  ∃ m n : ℕ, (∀ d, valid_assignment d) → (m + n) = sorry := 
sorry

end probability_valid_assignment_l4_4526


namespace at_least_one_genuine_l4_4751

theorem at_least_one_genuine :
  ∀ (total_products genuine_products defective_products selected_products : ℕ),
  total_products = 12 →
  genuine_products = 10 →
  defective_products = 2 →
  selected_products = 3 →
  (∃ g d : ℕ, g + d = selected_products ∧ g = 0 ∧ d = selected_products) = false :=
by
  intros total_products genuine_products defective_products selected_products
  intros H_total H_gen H_def H_sel
  sorry

end at_least_one_genuine_l4_4751


namespace abs_inequalities_equiv_l4_4465

theorem abs_inequalities_equiv (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  abs (a - b * Real.sqrt c) < 1 / (2 * b) ↔ abs (a ^ 2 - b ^ 2 * c) < Real.sqrt c :=
by {
  sorry
}

end abs_inequalities_equiv_l4_4465


namespace smallest_C_exists_l4_4237

theorem smallest_C_exists :
  ∃ C > 0, ∀ (a : Fin 5 → ℝ), (∀ i, 0 < a i) → 
    (∃ i j k l : Fin 5, i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧ 
      abs ((a i / a j) - (a k / a l)) ≤ C) :=
begin
  use 1/2,
  split,
  { norm_num, },
  {
    intros a h,
    -- Proof would go here
    sorry
  }
end

end smallest_C_exists_l4_4237


namespace point_A_outside_circle_iff_l4_4011

-- Define the conditions
def B : ℝ := 16
def radius : ℝ := 4
def A_position (t : ℝ) : ℝ := 2 * t

-- Define the theorem
theorem point_A_outside_circle_iff (t : ℝ) : (A_position t < B - radius) ∨ (A_position t > B + radius) ↔ (t < 6 ∨ t > 10) :=
by
  sorry

end point_A_outside_circle_iff_l4_4011


namespace find_word_l4_4673

theorem find_word (antonym : Nat) (cond : antonym = 26) : String :=
  "seldom"

end find_word_l4_4673


namespace equal_black_white_diagonals_l4_4661

theorem equal_black_white_diagonals (P : Finset (Fin 20))
  (h_black : P.card = 10)
  (h_white : (univ \ P).card = 10) 
  (h_regular : ∀ i ∈ P, ∃ j ∈ univ, j = ((i + 3) % 20)) :
  (∃ bs ws : Finset (Fin 20), bs.card = 45 ∧ ws.card = 45 ∧
    ∀ d ∈ bs, ∀ d' ∈ ws, are_diagonals_equal d d') := 
sorry

end equal_black_white_diagonals_l4_4661


namespace area_ECODF_l4_4605

-- Definitions of points and circles
structure Point :=
(x : ℝ) (y : ℝ)

def O := Point.mk 0 0
def A := Point.mk (-5) 0
def B := Point.mk 5 0
def OA := 5
def OB := 5
def O_mid_A_B := O = Point.mk ((A.x + B.x) / 2) ((A.y + B.y) / 2) -- Midpoint condition

-- Radii of circles
def radius_A := 3
def radius_B := 4

-- Length of line AB
def AB := 10 -- Given OA = 5 and O is midpoint

-- Tangents OC and OD
def OC := 4 -- From Pythagorean theorem: OC^2 + AC^2 = OA^2
def OD := 3 -- From Pythagorean theorem: OD^2 + BD^2 = OB^2

-- Areas of geometric entities
def area_rectangle_ABFE := 80
def area_triangle_ACO := 6
def area_triangle_BDO := 6
def area_sector_CAE := 4.18
def area_sector_DBF := 5.12

-- Definition of the shaded area ECODF
def area_shaded := area_rectangle_ABFE - area_triangle_ACO - area_triangle_BDO - area_sector_CAE - area_sector_DBF

-- Theorem statement for the proof
theorem area_ECODF : area_shaded = 58.7 :=
by
  -- Here's where the proof would go
  sorry

end area_ECODF_l4_4605


namespace rhombus_area_correct_l4_4714

-- Function to calculate the area of the rhombus
def rhombus_area (a d1 d2 : ℝ) : ℝ :=
  (d1 * d2) / 2

-- Problem conditions
def side (a : ℝ) : Prop :=
  a = 11

def diagonals_diff (d1 d2 : ℝ) : Prop :=
  abs (d1 - d2) = 8

-- Main theorem to prove
theorem rhombus_area_correct (a d1 d2 : ℝ) (h_side : side a) (h_diff : diagonals_diff d1 d2) :
  rhombus_area a d1 d2 = 104.81 :=
sorry

end rhombus_area_correct_l4_4714


namespace eccentricity_correct_l4_4295

noncomputable def eccentricity_of_hyperbola (a : ℝ) (ha : a > real.sqrt 2) : ℝ :=
  real.sqrt ((a^2 + 2) / a^2)

theorem eccentricity_correct (a : ℝ) (ha : a > real.sqrt 2) (hangle : real.angle.pi / 3 = real.angle.pi / 3) :
  eccentricity_of_hyperbola a ha = 2 * real.sqrt 3 / 3 :=
sorry

end eccentricity_correct_l4_4295


namespace find_k_l4_4827

theorem find_k (k : ℕ) : (1 / 2) ^ 16 * (1 / 81) ^ k = 1 / 18 ^ 16 → k = 8 :=
by
  intro h
  sorry

end find_k_l4_4827


namespace probability_of_condition_l4_4172

def Q_within_square (x y : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1

def condition (x y : ℝ) : Prop :=
  y > (1/2) * x

theorem probability_of_condition : 
  ∀ x y, Q_within_square x y → (0.75 = 3 / 4) :=
by
  sorry

end probability_of_condition_l4_4172


namespace foma_should_give_ierema_l4_4599

variables (F E Y : ℕ)

-- Given conditions in Lean definitions
def condition1 := F - 70 = E + 70
def condition2 := F - 40 = Y

-- Final statement to be proven
theorem foma_should_give_ierema : 
  condition1 → condition2 → F - E = 110 → F - E = 110 / 2 + 55 :=
by
  intros h1 h2 h3
  sorry

end foma_should_give_ierema_l4_4599


namespace triangle_equality_BC_AK_BK_l4_4417

-- Define the main geometric constructs
variables {A B C M N K : Type}

-- Given conditions
variables [Midpoint M A C] -- M is the midpoint of AC
variables [OnSegment N A M] -- N lies on segment AM
variables [EqualAngle M B N C B M] -- ∠MBN = ∠CBM
variables [OnExtension K B N (90 : Angle)] -- K is on the extension of BN such that ∠BMK is right

-- Define the main theorem
theorem triangle_equality_BC_AK_BK :
  BC = AK + BK :=
sorry

end triangle_equality_BC_AK_BK_l4_4417


namespace binomial_17_4_eq_2380_l4_4204

theorem binomial_17_4_eq_2380 : nat.choose 17 4 = 2380 :=
begin
  sorry
end

end binomial_17_4_eq_2380_l4_4204


namespace range_of_k_tan_alpha_l4_4813

noncomputable section

open Real

def a (x : ℝ) : ℝ × ℝ := (sin x, 1)
def b (k : ℝ) : ℝ × ℝ := (1, k)
def f (x k : ℝ) : ℝ := (a x).1 * (b k).1 + (a x).2 * (b k).2

theorem range_of_k (k : ℝ) : 
  (∃ x : ℝ, f x k = 1) ↔ k ∈ Icc 0 2 :=
by
  sorry

theorem tan_alpha (k α : ℝ) (hα : α ∈ Ioo 0 π) :
  f α k = (1 / 3) + k → 
  tan α ∈ {1 / (3 * sqrt (8 / 9)), -1 / (3 * sqrt (8 / 9))} :=
by
  sorry

end range_of_k_tan_alpha_l4_4813


namespace min_value_of_f_l4_4230

noncomputable def f (x : ℝ) : ℝ :=
  ∑ k in finset.range 51, (x - (2 * k + 1))^2

theorem min_value_of_f :
  ∃ x : ℝ, f 51 = 44200 :=
begin
  sorry
end

end min_value_of_f_l4_4230


namespace initial_principal_amount_l4_4179

noncomputable def compound_interest (P r n t : ℝ) : ℝ :=
  P * (1 + r / n)^(n * t)

theorem initial_principal_amount :
  let P := 4410 / (compound_interest 1 0.07 4 2 * compound_interest 1 0.09 2 2)
  abs (P - 3238.78) < 0.01 :=
by
  sorry

end initial_principal_amount_l4_4179


namespace tan_alpha_sin_cos_l4_4276

theorem tan_alpha_sin_cos (α : ℝ) (h : Real.tan α = 2) : Real.sin α * Real.cos α = 2 / 5 :=
begin
  sorry
end

end tan_alpha_sin_cos_l4_4276


namespace probability_all_selected_l4_4601

variables (P_Ram P_Ravi P_Rina : ℚ)

theorem probability_all_selected (hRam : P_Ram = 4/7) (hRavi : P_Ravi = 1/5) (hRina : P_Rina = 3/8) :
  P_Ram * P_Ravi * P_Rina = 3/70 :=
by
  -- Given conditions are already stated.
  -- Proof will be provided to complete the theorem.
  sorry

end probability_all_selected_l4_4601


namespace two_digit_primes_l4_4315

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def reverse_digits (n : ℕ) : ℕ :=
  let t := n / 10
  let u := n % 10
  10 * u + t

def is_valid_n (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ is_prime (n + reverse_digits n)

theorem two_digit_primes (N : ℕ) : ∃! n, is_valid_n n :=
  ∃! n, n = 10 :=
begin
  sorry
end

end two_digit_primes_l4_4315


namespace triangle_property_l4_4444

variables {ABC : Type*} [inner_product_space ℝ ABC]
variables (A B C M K N : ABC)

noncomputable def midpoint (A C : ABC) (M : ABC) : Prop := dist A M = dist M C

noncomputable def angle_equality (B M N : ABC) : Prop := ∠M B N = ∠C B M

noncomputable def right_angle (B M K : ABC) : Prop := ∠B M K = π / 2

theorem triangle_property (h1 : midpoint A C M) (h2 : angle_equality B M N) (h3 : right_angle B M K) : dist B C = dist A K + dist B K :=
sorry

end triangle_property_l4_4444


namespace cyclists_meet_time_l4_4141

theorem cyclists_meet_time (v1 v2 d : ℕ) (h₀ : v1 = 7) (h₁ : v2 = 8) (h₂ : d = 600) :
  d / (v1 + v2) = 40 := by
  rw [h₀, h₁, h₂]
  norm_num
  sorry

end cyclists_meet_time_l4_4141


namespace speed_of_water_l4_4657

variable (v : ℝ) -- the speed of the water in km/h
variable (t : ℝ) -- time taken to swim back in hours
variable (d : ℝ) -- distance swum against the current in km
variable (s : ℝ) -- speed in still water

theorem speed_of_water :
  ∀ (v t d s : ℝ),
  s = 20 -> t = 5 -> d = 40 -> d = (s - v) * t -> v = 12 :=
by
  intros v t d s ht hs hd heq
  sorry

end speed_of_water_l4_4657


namespace equalize_foma_ierema_l4_4579

theorem equalize_foma_ierema (F E Y : ℕ) 
  (h1 : F - 70 = E + 70) 
  (h2 : F - 40 = Y) 
  (h3 : Y = E + 70) 
  : ∃ x : ℕ, x = 55 ∧ F - x = E + x :=
by
  use 55
  sorry

end equalize_foma_ierema_l4_4579


namespace union_A_B_inter_A_B_compl_inter_A_B_union_A_B_l4_4299

namespace SetOperations

-- Definition of sets
def A := {x : ℝ | 3 ≤ x ∧ x < 10}
def B := {x : ℝ | 2 * x - 8 ≥ 0}

-- Questions to prove
theorem union_A_B : A ∪ B = {x : ℝ | 3 ≤ x} := sorry

theorem inter_A_B : A ∩ B = {x : ℝ | 4 ≤ x ∧ x < 10} := sorry

theorem compl_inter_A_B_union_A_B : (⦃x : ℝ | ¬ (4 ≤ x ∧ x < 10)⦄ ∩ {x : ℝ | 3 ≤ x}) = {x : ℝ | (3 ≤ x ∧ x < 4) ∨ 10 ≤ x} := sorry

end SetOperations

end union_A_B_inter_A_B_compl_inter_A_B_union_A_B_l4_4299


namespace smallest_M_l4_4885

noncomputable def smallest_possible_M : ℕ :=
  1010

theorem smallest_M (a b c d e : ℕ) (h_pos: a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0)
                  (h_sum: a + b + c + d + e = 2020) : 
  ∃ (M : ℕ), M = smallest_possible_M ∧ M = max (max (a + b) (max (b + c) (max (c + d) (d + e)))) := 
by {
  use 1010,
  split,
  { refl, },
  { sorry },
}

end smallest_M_l4_4885


namespace kim_no_tests_probability_l4_4066

theorem kim_no_tests_probability :
  let p_math_test := 5 / 8
  let p_history_test := 1 / 3
  let p_no_math_test := 1 - p_math_test
  let p_no_history_test := 1 - p_history_test
  let p_no_tests := p_no_math_test * p_no_history_test
  in p_no_tests = 1 / 4 :=
by
  let p_math_test := 5 / 8
  let p_history_test := 1 / 3
  let p_no_math_test := 1 - p_math_test
  let p_no_history_test := 1 - p_history_test
  let p_no_tests := p_no_math_test * p_no_history_test
  show p_no_tests = 1 / 4
  sorry

end kim_no_tests_probability_l4_4066


namespace transformed_center_coordinates_l4_4201

theorem transformed_center_coordinates :
  let initial_center : ℝ × ℝ := (3, -5) in
  let reflected_center := (-initial_center.1, initial_center.2) in
  let rotated_center := (reflected_center.2, -reflected_center.1) in
  let translated_center := (rotated_center.1, rotated_center.2 + 4) in
  translated_center = (-5, 7) :=
by 
  -- transformation steps
  let initial_center : ℝ × ℝ := (3, -5)
  let reflected_center := (-initial_center.1, initial_center.2)
  let rotated_center := (reflected_center.2, -reflected_center.1)
  let translated_center := (rotated_center.1, rotated_center.2 + 4)

  -- assertion
  have h : translated_center = (-5, 7) := by
    -- reflection
    have h_reflection : reflected_center = (-3, -5) := by
      simp [initial_center, reflected_center]

    -- rotation
    have h_rotation : rotated_center = (-5, 3) := by
      simp [reflected_center, h_reflection, rotated_center]

    -- translation
    have h_translation : translated_center = (-5, 7) := by
      simp [rotated_center, h_rotation, translated_center]

    exact h_translation

  exact h

end transformed_center_coordinates_l4_4201


namespace xy_inequality_l4_4775

theorem xy_inequality (x y : ℝ) (h: x^8 + y^8 ≤ 2) : 
  x^2 * y^2 + |x^2 - y^2| ≤ π / 2 :=
sorry

end xy_inequality_l4_4775


namespace sin_identity_l4_4753

theorem sin_identity (α : ℝ) (h_tan : Real.tan α = -3 / 4) : 
  Real.sin α * (Real.sin α - Real.cos α) = 21 / 25 :=
sorry

end sin_identity_l4_4753


namespace find_principal_sum_l4_4624

theorem find_principal_sum (P : ℝ) (r : ℝ) (A2 : ℝ) (A3 : ℝ) : 
  (A2 = 7000) → (A3 = 9261) → 
  (A2 = P * (1 + r)^2) → (A3 = P * (1 + r)^3) → 
  P = 4000 :=
by
  intro hA2 hA3 hA2_eq hA3_eq
  -- here, we assume the proof steps leading to P = 4000
  sorry

end find_principal_sum_l4_4624


namespace sin2_sum_ge_cos_sum_squared_equality_case_l4_4277

variable {A B C : ℝ}

-- Conditions
def non_obtuse_triangle : Prop := 
  A + B + C = 180 ∧ A ≤ 90 ∧ B ≤ 90 ∧ C ≤ 90

-- Theorem statement
theorem sin2_sum_ge_cos_sum_squared (h : non_obtuse_triangle A B C) : 
  sin A ^ 2 + sin B ^ 2 + sin C ^ 2 ≥ (cos A + cos B + cos C) ^ 2 :=
sorry

-- To state the condition when equality holds
theorem equality_case (h : non_obtuse_triangle A B C) :
  (sin A ^ 2 + sin B ^ 2 + sin C ^ 2 = (cos A + cos B + cos C) ^ 2) ↔
  (A = B ∧ B = C ∧ A = 60) ∨ -- Equilateral triangle
  (A = 45 ∧ B = 45 ∧ C = 90) := -- Isosceles right triangle
sorry

end sin2_sum_ge_cos_sum_squared_equality_case_l4_4277


namespace max_value_p_l4_4784

theorem max_value_p (x y : ℝ) (h : 3 * x ^ 2 + 2 * y ^ 2 ≤ 6) : 
  ∃ t : ℝ, t = 2 * x + y ∧ t ≤ sqrt 11 :=
sorry

end max_value_p_l4_4784


namespace binomial_17_4_l4_4203

theorem binomial_17_4 : nat.choose 17 4 = 2380 := by
  sorry

end binomial_17_4_l4_4203


namespace cost_of_shoes_l4_4200

-- Define the conditions
def saved : Nat := 30
def earn_per_lawn : Nat := 5
def lawns_per_weekend : Nat := 3
def weekends_needed : Nat := 6

-- Prove the total amount saved is the cost of the shoes
theorem cost_of_shoes : saved + (earn_per_lawn * lawns_per_weekend * weekends_needed) = 120 := by
  sorry

end cost_of_shoes_l4_4200


namespace distance_difference_l4_4206

theorem distance_difference (t : ℕ) (speed_alice speed_bob : ℕ) :
  speed_alice = 15 → speed_bob = 10 → t = 6 → (speed_alice * t) - (speed_bob * t) = 30 :=
by
  intros h1 h2 h3
  sorry

end distance_difference_l4_4206


namespace max_min_sum_l4_4754

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (5 * real.exp (x * real.log a) + 1) / (real.exp (x * real.log a) - 1) + real.log (real.sqrt (1 + x^2) - x)

theorem max_min_sum (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) : 
  let M := real.Sup (set.range (f a))
  let N := real.Inf (set.range (f a))
  M + N = 4 := by
  sorry

end max_min_sum_l4_4754


namespace triangle_equality_BC_AK_BK_l4_4411

-- Define the main geometric constructs
variables {A B C M N K : Type}

-- Given conditions
variables [Midpoint M A C] -- M is the midpoint of AC
variables [OnSegment N A M] -- N lies on segment AM
variables [EqualAngle M B N C B M] -- ∠MBN = ∠CBM
variables [OnExtension K B N (90 : Angle)] -- K is on the extension of BN such that ∠BMK is right

-- Define the main theorem
theorem triangle_equality_BC_AK_BK :
  BC = AK + BK :=
sorry

end triangle_equality_BC_AK_BK_l4_4411


namespace solution_sum_of_eq_zero_l4_4118

open Real

theorem solution_sum_of_eq_zero : 
  let f (x : ℝ) := (4*x + 6) * (3*x - 8)
  in (∀ x, f x = 0 → x = -3/2 ∨ x = 8/3) → 
     (-3/2 + 8/3 = 7/6) :=
by 
  let f (x : ℝ) := (4*x + 6) * (3*x - 8)
  intro h
  have h₁ : f(-3/2) = 0 := by sorry
  have h₂ : f(8/3) = 0 := by sorry
  have sum_of_solutions : -3/2 + 8/3 = 7/6 := by 
    sorry
  exact sum_of_solutions

end solution_sum_of_eq_zero_l4_4118


namespace find_ab_l4_4884

-- Given conditions
variables (a b : ℝ)
-- Polynomial definition
def polynomial (x : ℂ) := x^3 + (a : ℂ) * x^2 - x + (b : ℂ)

-- Root condition
theorem find_ab (h_root : polynomial a b (2 - 3 * complex.I) = 0) :
  (a = 7.5 ∧ b = -45.5) :=
sorry

end find_ab_l4_4884


namespace dihedral_angle_truncated_tetrahedron_l4_4740

theorem dihedral_angle_truncated_tetrahedron
  (a : ℝ)  -- side length of the larger base
  (α : ℝ)  -- dihedral angle between the base and lateral face
  (inscribed_sphere : ∃ r : ℝ, r = (a * sqrt 3 / 6) * tan (α / 2))  -- condition for inscribed sphere
  (edge_touching_sphere : ∃ l : ℝ, l = (a * sqrt 3 / 6) * sqrt (tan α ^ 2 + 4))  -- condition for sphere touching all edges
  : α = 2 * arctan (sqrt 3 - sqrt 2) := sorry

end dihedral_angle_truncated_tetrahedron_l4_4740


namespace find_f_of_5_l4_4289

def f : ℝ → ℝ
| x := if x ≤ 0 then 2^x else f (x - 3)

theorem find_f_of_5 : f 5 = 1 / 2 :=
  sorry

end find_f_of_5_l4_4289


namespace trigonometric_expression_simplification_l4_4287

theorem trigonometric_expression_simplification (θ : ℝ) (h : Real.tan θ = 3) :
  (Real.sin (3 * Real.pi / 2 + θ) + 2 * Real.cos (Real.pi - θ)) /
  (Real.sin (Real.pi / 2 - θ) - Real.sin (Real.pi - θ)) = 3 / 2 := 
sorry

end trigonometric_expression_simplification_l4_4287


namespace license_plate_combinations_l4_4692

open Finset

theorem license_plate_combinations :
  let letters_combinations := choose 25 2
  let positioning := choose 4 2
  let digit_combinations := 10 * 9
  letters_combinations * positioning * digit_combinations = 162000 :=
by
  sorry

end license_plate_combinations_l4_4692


namespace only_one_number_as_sum_of_two_primes_l4_4308

/-- Define the nth term of the sequence. -/
def sequence (n : ℕ) : ℕ := 5 + 10 * (n - 1)

/-- Check if a number can be written as the sum of two primes. -/
def can_be_written_as_sum_of_two_primes (n : ℕ) : Prop :=
  ∃ p1 p2 : ℕ, nat.prime p1 ∧ nat.prime p2 ∧ n = p1 + p2

/-- Prove that only one number in the first 10 terms of the set can be written as the sum of two primes. -/
theorem only_one_number_as_sum_of_two_primes :
  finset.card (finset.filter can_be_written_as_sum_of_two_primes (finset.image sequence (finset.range 10))) = 1 :=
sorry

end only_one_number_as_sum_of_two_primes_l4_4308


namespace triangle_equality_BC_AK_BK_l4_4412

-- Define the main geometric constructs
variables {A B C M N K : Type}

-- Given conditions
variables [Midpoint M A C] -- M is the midpoint of AC
variables [OnSegment N A M] -- N lies on segment AM
variables [EqualAngle M B N C B M] -- ∠MBN = ∠CBM
variables [OnExtension K B N (90 : Angle)] -- K is on the extension of BN such that ∠BMK is right

-- Define the main theorem
theorem triangle_equality_BC_AK_BK :
  BC = AK + BK :=
sorry

end triangle_equality_BC_AK_BK_l4_4412


namespace parabola_circle_intersection_l4_4063

theorem parabola_circle_intersection :
  let parabola := λ x : ℝ, x^2 - 2*x - 3 in
  let A := (1, -1) in
  let B := (3, 0) in
  let C := (0, -3) in 
  -- Conditions: The intersection points are on the circle
  let circle_center := (1, -1) in
  let r := sqrt(5) in
  (∀ x y, y = parabola x → 
         (x, y) = A ∨ (x, y) = B ∨ (x, y) = C) →
  ( ∃ (h: ℝ → ℝ → Prop), (h x y → (x - circle_center.1)^2 + (y - circle_center.2)^2 = r^2)) :=
by {
  sorry
}

end parabola_circle_intersection_l4_4063


namespace solve_equation_1_solve_equation_2_l4_4934

theorem solve_equation_1 (x : ℝ) : x^2 - 7 * x = 0 ↔ (x = 0 ∨ x = 7) :=
by sorry

theorem solve_equation_2 (x : ℝ) : 2 * x^2 - 6 * x + 1 = 0 ↔ (x = (3 + Real.sqrt 7) / 2 ∨ x = (3 - Real.sqrt 7) / 2) :=
by sorry

end solve_equation_1_solve_equation_2_l4_4934


namespace sum_of_squares_of_sines_l4_4703

theorem sum_of_squares_of_sines (α : ℝ) :
  (∑ k in Finset.range 180, (Real.sin (α + k * (Real.pi / 180)))^2) = 90 :=
by
  sorry

end sum_of_squares_of_sines_l4_4703


namespace fraction_simplifies_l4_4710

-- Define the integers
def a : ℤ := 1632
def b : ℤ := 1625
def c : ℤ := 1645
def d : ℤ := 1612

-- Define the theorem to prove
theorem fraction_simplifies :
  (a^2 - b^2) / (c^2 - d^2) = 7 / 33 := by
  sorry

end fraction_simplifies_l4_4710


namespace arrangement_count_l4_4546

-- Define the set of students
def students : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def front_row : Finset ℕ := {1, 2, 3, 4}
def back_row : Finset ℕ := {5, 6, 7, 8}

-- Define constraints
def is_front_row (s : ℕ) : Prop := s ∈ front_row
def is_back_row (s : ℕ) : Prop := s ∈ back_row

-- The specific students A, B (in front row), and C (in back row)
def A : ℕ := 1
def B : ℕ := 2
def C : ℕ := 5

-- Main theorem statement to be proved
theorem arrangement_count :
  (∀ s ∈ {A, B}, is_front_row s) ∧ is_back_row C →
  (finset.card (finset.image id students) = 8) →
  finset.card front_row = 4 →
  finset.card back_row = 4 →
  -- Total count of different arrangements satisfying the constraints
  let arrangement_count := 12 * 4 * 120 
  arrangement_count = 5760 :=
sorry

end arrangement_count_l4_4546


namespace minimum_distance_between_A_and_B_l4_4777

def A (t : ℝ) : ℝ × ℝ × ℝ := (1 - t, 1 - t, t)
def B (t : ℝ) : ℝ × ℝ × ℝ := (2, t, t)

noncomputable def distance (A B : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2 + (A.3 - B.3) ^ 2)

theorem minimum_distance_between_A_and_B : ∃ t : ℝ, distance (A t) (B t) = C :=
by sorry

end minimum_distance_between_A_and_B_l4_4777


namespace clock_angle_at_3_20_l4_4990

def deg_per_hour : ℝ := 360 / 12

def min_hand_angle (minute: ℕ) : ℝ := (minute / 60) * 360

def hour_hand_angle (hour minute: ℕ) : ℝ := (hour * deg_per_hour) + (minute / 60) * deg_per_hour

def smaller_angle (angle1 angle2: ℝ) : ℝ := abs (angle1 - angle2)

theorem clock_angle_at_3_20 :
  smaller_angle (min_hand_angle 20) (hour_hand_angle 3 20) = 20 := by
  sorry

end clock_angle_at_3_20_l4_4990


namespace largest_coefficient_term_in_expansion_l4_4540

theorem largest_coefficient_term_in_expansion :
  ∃ r : ℕ, r = 4 ∧
    (∀ k : ℕ, k ≠ 4 → 
    let coeff := binomial 7 k * (-2 : ℤ)^k in 
    coeff ≤ binomial 7 4 * (-2 : ℤ)^4) :=
sorry

end largest_coefficient_term_in_expansion_l4_4540


namespace starting_number_prime_factors_210_l4_4548

theorem starting_number_prime_factors_210
  (x : Nat)
  (h1 : ∀ p ∈ {2, 3, 5, 7}, Nat.Prime p ∧ p ≤ 100)
  (h2 : {p | p ∈ {2, 3, 5, 7} ∧ x ≤ p ≤ 100}.card = 4) :
  x = 1 :=
sorry

end starting_number_prime_factors_210_l4_4548


namespace angle_bisector_ratio_l4_4858

-- Definitions and conditions for the problem
variables {X Y Z Q U V : Type} [LinearOrder X] [LinearOrder Y] [LinearOrder Z] [LinearOrder Q] [LinearOrder U] [LinearOrder V]

-- Distances given in the problem
constants (XY XZ YZ : ℝ)
constants (angle_bisector_XU angle_bisector_YV : X → Y → Z → Q → U → V)

-- Given side lengths
axiom XY_length : XY = 8
axiom XZ_length : XZ = 6
axiom YZ_length : YZ = 4

-- Given angle bisectors
axiom angle_bisector_intersection : angle_bisector_XU X Y Z Q U ∧ angle_bisector_YV Y Z X Q V

-- The proof statement for the ratio YQ/QV
theorem angle_bisector_ratio : ∀ (YQ QV : ℝ), ∃ Q, 
  ∃ U V, 
  XY = 8 ∧ XZ = 6 ∧ YZ = 4 ∧ angle_bisector_intersection → 
  YQ / QV = 4 / 3 :=
by
  sorry

end angle_bisector_ratio_l4_4858


namespace determine_b_c_d_l4_4052

noncomputable def a (n : ℕ) (b c d : ℤ) : ℤ := b * int.floor (real.sqrt (n + int.to_nat c)) + d

theorem determine_b_c_d :
  ∃ (b c d : ℤ), (∀ n, a (n + 1) b c d ≥ a n b c d) ∧ 
                 (∀ m, ∃ k, a (k + 1) b c d = a k b c d + 2 * m ∧ ∀i < m, a (k + i) b c d = 2 * m - 1) ∧
                 b + c + d = 2 :=
sorry

end determine_b_c_d_l4_4052


namespace polyhedron_edge_count_l4_4164

-- Definitions for vertices and polyhedrons
variables {V : Type} {Q : Type} [Graph V Q]

-- Definitions for the properties of polyhedron Q and resultant polyhedron R
def edges (G : Type) [Graph V G] : ℕ := sorry  -- placeholder for edge count function

def cutting_plane (V : Type) : V → Set (Set V) := sorry  -- placeholder for cutting plane definition

theorem polyhedron_edge_count 
  (vertices : Finset V) (edges_Q : ℕ) (cutting_planes : V → Set (Set V)) :
  edges_Q = 150 ∧
  ∀ v ∈ vertices, v ∉ cutting_planes v ∧
  ∀ (P_k P_j : Set V), P_k ≠ P_j → (∃ e, e ∈ (P_k ∩ P_j)) →
  edges_R (resulting_polyhedron vertices edges_Q cutting_planes) = 2 * (|vertices|) + 600 :=
by
  sorry

end polyhedron_edge_count_l4_4164


namespace triangle_property_l4_4442

variables {ABC : Type*} [inner_product_space ℝ ABC]
variables (A B C M K N : ABC)

noncomputable def midpoint (A C : ABC) (M : ABC) : Prop := dist A M = dist M C

noncomputable def angle_equality (B M N : ABC) : Prop := ∠M B N = ∠C B M

noncomputable def right_angle (B M K : ABC) : Prop := ∠B M K = π / 2

theorem triangle_property (h1 : midpoint A C M) (h2 : angle_equality B M N) (h3 : right_angle B M K) : dist B C = dist A K + dist B K :=
sorry

end triangle_property_l4_4442


namespace spot_area_outside_l4_4712

theorem spot_area_outside (side length: ℝ) (rope length: ℝ) :
  side length = 2 → rope length = 5 → 
  ∃ (area: ℝ), area = 25 * Real.pi := 
by
  intro hside hrope
  use 25 * Real.pi
  sorry

end spot_area_outside_l4_4712


namespace sum_of_abc_l4_4530

theorem sum_of_abc (a b c : ℕ) (h : a + b + c = 12) 
  (area_ratio : ℝ) (side_length_ratio : ℝ) 
  (ha : area_ratio = 50 / 98) 
  (hb : side_length_ratio = (Real.sqrt 50) / (Real.sqrt 98))
  (hc : side_length_ratio = (a * (Real.sqrt b)) / c) :
  a + b + c = 12 :=
by
  sorry

end sum_of_abc_l4_4530


namespace volume_invariance_of_tetrahedron_motion_l4_4847

noncomputable def tetrahedron_volume_constant 
  (a b c : ℝ × ℝ × ℝ) -- representing the lines as points for simplicity
  (length_KS : ℝ) -- constant length of edge KS
  (constant_volume : Prop) :
  Prop :=
∀ (K S : ℝ × ℝ × ℝ), -- K moves along line a, S moves along line b
  is_on_line K a →
  is_on_line S b →
  tetrahedron_volume K S a b c = constant_volume → constant_volume

axiom is_on_line (P L : ℝ × ℝ × ℝ) : Prop
axiom tetrahedron_volume (K S : ℝ × ℝ × ℝ) (a b c : ℝ × ℝ × ℝ) : ℝ → Prop

theorem volume_invariance_of_tetrahedron_motion 
  (a b c : ℝ × ℝ × ℝ)
  (length_KS : ℝ)
  (constant_volume : ℝ) : 
  Prop :=
tetrahedron_volume_constant a b c length_KS constant_volume

end volume_invariance_of_tetrahedron_motion_l4_4847


namespace projection_correct_l4_4659

def vec1 := ⟨2, -4⟩ : ℝ × ℝ
def vec2 := ⟨3, -3⟩ : ℝ × ℝ
def vec3 := ⟨3, 5⟩ : ℝ × ℝ
def expected_projection := ⟨-1, 1⟩ : ℝ × ℝ

theorem projection_correct : 
  projection vec1 vec2 vec3 = expected_projection := sorry

end projection_correct_l4_4659


namespace find_k_value_l4_4796

theorem find_k_value (k : ℝ) : 
  (-x ^ 2 - (k + 11) * x - 8 = -( (x - 2) * (x - 4) ) ) → k = -17 := 
by 
  sorry

end find_k_value_l4_4796


namespace final_price_percentage_of_original_l4_4669

theorem final_price_percentage_of_original (original_price sale_price final_price : ℝ)
  (h1 : sale_price = original_price * 0.5)
  (h2 : final_price = sale_price * 0.9) :
  final_price = original_price * 0.45 :=
by
  sorry

end final_price_percentage_of_original_l4_4669


namespace find_minimum_value_of_f_l4_4228

noncomputable def f (x: ℝ) : ℝ :=
  ∑ k in Finset.range 51, (x - (2 * k + 1))^2

theorem find_minimum_value_of_f :
  ∃ x, f x = 44200 :=
sorry

end find_minimum_value_of_f_l4_4228


namespace distance_P_to_origin_l4_4048

-- Define the point P.
def P : ℝ × ℝ := (-2, -4)

-- Define a function to calculate the Euclidean distance from a point to the origin.
def distance_to_origin (point : ℝ × ℝ) : ℝ :=
  Real.sqrt (point.1^2 + point.2^2)

-- The theorem stating that the distance from point P to the origin is sqrt(20).
theorem distance_P_to_origin : distance_to_origin P = Real.sqrt 20 :=
  sorry

end distance_P_to_origin_l4_4048


namespace integer_for_all_n_l4_4333

theorem integer_for_all_n
  (x y : ℝ)
  (f : ℕ → ℤ)
  (h : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 4 → f n = ((x^n - y^n) / (x - y))) :
  ∀ n : ℕ, 0 < n → f n = ((x^n - y^n) / (x - y)) :=
by sorry

end integer_for_all_n_l4_4333


namespace three_digit_even_numbers_count_l4_4976

-- Given three cards with numbers on both sides: (0, 1), (2, 3), (4, 5)
-- We need to prove that the number of different three-digit even numbers formed is 16.

theorem three_digit_even_numbers_count : 
  let cards := [(0, 1), (2, 3), (4, 5)] in
  let is_even := λ n: Nat, n % 2 = 0 in
  let valid_choices (a: Nat) (b: Nat) (c: Nat) := 
    a ≠ 0 ∧ is_even c ∧ 
    ((a ∈ [0, 1, 2, 3, 4, 5]) ∧ (b ∈ [0, 1, 2, 3, 4, 5]) ∧ (c ∈ [0, 1, 2, 3, 4, 5])) ∧
    distinct [a, b, c] in
  (∃ (count: Nat), count = 16 ∧ 
    count = (cards.length * cards.length * cards.length)) → 
  count = 16 :=
by 
  sorry

end three_digit_even_numbers_count_l4_4976


namespace proof_problem_l4_4797

variables (x y : ℝ)

theorem proof_problem 
  (h1 : y > 0) 
  (h2 : | x - real.log (y^2) | = x + real.log (y^2))
  : x * (y - 1)^2 = 0 :=
sorry

end proof_problem_l4_4797


namespace area_of_triangle_XYZ_l4_4030

open Real

/-- Let \( XYZ \) be a scalene right triangle. Suppose \( Q \) is a point on hypotenuse \( \overline{XZ} \) such that \( \angle{XYQ} = 60^\circ \), \( XQ = 2 \), and \( QZ = 1 \). Then the area of triangle \( XYZ \) is \( \frac{3\sqrt{5}}{5} \). -/
theorem area_of_triangle_XYZ
  (XYZ : Type*)
  [metric_space XYZ]
  [ordered_ring XYZ]
  (X Y Z Q : XYZ)
  (h_triangle : ∃ (hXY : Prop) (hXZ_original_length : Prop), hXY ∧ hXZ_original_length ∧ true)
  (h_scalene : (Δ XYZ))
  (h_right : triangle_is_right XYZ)
  (h_geom : (∠ XYQ = π/3))
  (h_XQ : dist X Q = 2)
  (h_QZ: dist Q Z = 1) : 
  (area_of_triangle XYZ = 3 * (sqrt 5) / 5)
:= by
  sorry

end area_of_triangle_XYZ_l4_4030


namespace sum_of_abs_coefficients_eq_2_pow_4046_l4_4255

theorem sum_of_abs_coefficients_eq_2_pow_4046
  {a : ℕ → ℤ}
  (h : ∀ x, (2 - x) ^ 2023 = ∑ i in finset.range 2024, a i * (x + 1) ^ i) :
  ∑ i in finset.range 2024, |a i| = 2 ^ 4046 :=
sorry

end sum_of_abs_coefficients_eq_2_pow_4046_l4_4255


namespace problem1_problem2_l4_4633

-- Proof problem based on Question 1
theorem problem1 :
  (real.sqrt 2 - 1)^0 + 2 * (1 / 3) + (-1)^2023 - (-1 / 3)^(-1) = 11 / 3 :=
by sorry

-- Proof problem based on Question 2
theorem problem2 (x : ℝ) (h : x = 3) :
  (x^2 / (x - 2) - x - 2) / (4 * x / (x^2 - 4)) = 5 / 3 :=
by rw [h]; sorry

end problem1_problem2_l4_4633


namespace ice_cream_volume_correct_l4_4057

noncomputable def ice_cream_volume : ℝ :=
  let r := 3
  let h_cone := 12
  let h_cylinder := 2
  let V_cone := (1/3) * Mathlib.pi * r^2 * h_cone
  let V_cylinder := Mathlib.pi * r^2 * h_cylinder
  let V_hemisphere := (2/3) * Mathlib.pi * r^3
  V_cone + V_cylinder + V_hemisphere

theorem ice_cream_volume_correct :
  ice_cream_volume = 72 * Mathlib.pi :=
by
  sorry

end ice_cream_volume_correct_l4_4057


namespace A_and_C_work_together_in_2_hours_l4_4156

theorem A_and_C_work_together_in_2_hours
  (A_rate : ℚ)
  (B_rate : ℚ)
  (C_rate : ℚ)
  (A_4_hours : A_rate = 1 / 4)
  (B_12_hours : B_rate = 1 / 12)
  (B_and_C_3_hours : B_rate + C_rate = 1 / 3) :
  (A_rate + C_rate = 1 / 2) :=
by
  sorry

end A_and_C_work_together_in_2_hours_l4_4156


namespace odd_n_never_all_tails_even_n_possible_all_tails_l4_4076

-- Part (a)
theorem odd_n_never_all_tails (n : ℕ) (h_n : 3 ≤ n) (h_odd : n % 2 = 1) :
  ∀ (coins : Fin n → Bool), (∀ k : Fin n, if k = 0 then coins k = true else coins k = false) →
  ∀ k : Fin n, coins k = false → false :=
sorry

-- Part (b)
theorem even_n_possible_all_tails (n : ℕ) (h_n : 3 ≤ n) :
  (∃ k, n = 2 * k) ↔ (∃ numOfOperations, ∀ (coins : Fin n → Bool), (∀ k : Fin n, if k = 0 then coins k = true else coins k = false) →
  ∃ (turnCoin : ℕ → Bool), turnCoin numOfOperations = false) :=
sorry

end odd_n_never_all_tails_even_n_possible_all_tails_l4_4076


namespace subtraction_result_l4_4936

theorem subtraction_result :
  let x := 567.89
  let y := 123.45
  (x - y) = 444.44 :=
by
  sorry

end subtraction_result_l4_4936


namespace total_length_of_lines_in_T_l4_4378

def T (x y : ℝ) : Prop :=
  abs (abs (abs x - 3) - 2) + abs (abs (abs y - 3) - 2) = 2

theorem total_length_of_lines_in_T : (∑ (x y : ℝ), T x y) = 64 * real.sqrt 2 := 
sorry

end total_length_of_lines_in_T_l4_4378


namespace direction_cosines_AB_l4_4701

/-- Points in 3-dimensional space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Vector from point A to point B -/
def vectorAB (A B : Point3D) : Point3D :=
  ⟨B.x - A.x, B.y - A.y, B.z - A.z⟩

/-- Magnitude of the vector -/
def magnitude (v : Point3D) : ℝ :=
  real.sqrt ((v.x ^ 2) + (v.y ^ 2) + (v.z ^ 2))

/-- Direction cosines of a vector -/
def directionCosines (v : Point3D) (mag : ℝ) : Point3D :=
  ⟨v.x / mag, v.y / mag, v.z / mag⟩

/-- Problem Statement -/
theorem direction_cosines_AB :
  let A := Point3D.mk (-3) 2 0
  let B := Point3D.mk 3 (-3) 1
  let AB := vectorAB A B
  let magAB := magnitude AB
  let cosines := directionCosines AB magAB
  cosines = ⟨6 / real.sqrt 62, -5 / real.sqrt 62, 1 / real.sqrt 62⟩ := by
  sorry

end direction_cosines_AB_l4_4701


namespace sum_of_abs_coefficients_eq_2_pow_4046_l4_4254

theorem sum_of_abs_coefficients_eq_2_pow_4046
  {a : ℕ → ℤ}
  (h : ∀ x, (2 - x) ^ 2023 = ∑ i in finset.range 2024, a i * (x + 1) ^ i) :
  ∑ i in finset.range 2024, |a i| = 2 ^ 4046 :=
sorry

end sum_of_abs_coefficients_eq_2_pow_4046_l4_4254


namespace fomagive_55_l4_4592

variables (F E Y : ℕ)

-- Conditions
def condition1 := F - E = 140
def condition2 := F - Y = 40
def condition3 := Y = E + 70

-- Proposition: Foma should give 55 coins to Ierema
theorem fomagive_55 (h1 : condition1 F E) (h2 : condition2 F Y) (h3 : condition3 E Y) : ∃ G, G = 55 :=
by
  have := h1,
  have := h2,
  have := h3,
  sorry

end fomagive_55_l4_4592


namespace min_sum_of_distances_l4_4808

-- Define the line l1
def line_l1 : ℝ × ℝ → Prop :=
  λ (x y), 4 * x - 3 * y + 6 = 0

-- Define the line l2
def line_l2 : ℝ → Prop :=
  λ x, x = -1

-- Define the parabola
def parabola : ℝ × ℝ → Prop :=
  λ (x y), y^2 = 4 * x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the distance from a point to a line
def distance_to_line {α : Type*} [metric_space α] (p : α) (line : α → Prop) : ℝ :=
  -- Add a placeholder distance formula
  sorry

-- Define the minimum sum of distances problem
theorem min_sum_of_distances : ∀ (P : ℝ × ℝ), parabola P → 
    let dist1 := distance_to_line P line_l1 in
    let dist2 := distance_to_line P line_l2 in
    dist1 + dist2 = 2 :=
sorry

end min_sum_of_distances_l4_4808


namespace binomial_17_4_l4_4202

theorem binomial_17_4 : nat.choose 17 4 = 2380 := by
  sorry

end binomial_17_4_l4_4202


namespace Vasya_has_more_ways_to_place_kings_l4_4483

-- Definitions based on problem conditions
def Petya_king_placements : Nat := 500
def Vasya_king_placements : Nat := 500
def Petya_board : Nat × Nat := (100, 50)
def Vasya_board : Nat × Nat := (100, 100)
def Vasya_white_cells : Nat := 5000 -- White cells count in 100x100 checkered board

-- Proposition for the proof problem
theorem Vasya_has_more_ways_to_place_kings (p_king_placements : Nat) 
    (v_king_placements : Nat)
    (p_board : Nat × Nat)
    (v_board : Nat × Nat)
    (v_white_cells : Nat)
    (no_attack_p : (Nat × Nat) → List (Nat × Nat) → Prop)
    (no_attack_v : (Nat × Nat) → List (Nat × Nat) → Prop) : 
    p_king_placements = 500 ∧ v_king_placements = 500 ∧ 
    p_board = (100, 50) ∧ v_board = (100, 100) ∧ 
    v_white_cells = 5000 →
    (∃ (p_arr : List (Nat × Nat)), 
       length p_arr = p_king_placements ∧ 
       ∀ k ∈ p_arr, no_attack_p k p_arr) → 
    (∃ (v_arr : List (Nat × Nat)), 
       length v_arr = v_king_placements ∧ 
       ∀ k ∈ v_arr, no_attack_v k v_arr) → 
    v_king_placements > p_king_placements :=
by
  intros h1 h2 h3
  sorry

end Vasya_has_more_ways_to_place_kings_l4_4483


namespace projection_of_v_onto_plane_l4_4236

def vector := ℝ × ℝ × ℝ

def projection_onto_plane (v : vector) (n : vector) (c : ℝ) : vector :=
  let vn := (v.1 * n.1 + v.2 * n.2 + v.3 * n.3) / (n.1 * n.1 + n.2 * n.2 + n.3 * n.3)
  let n_proj := (vn * n.1, vn * n.2, vn * n.3)
  (v.1 - n_proj.1, v.2 - n_proj.2, v.3 - n_proj.3)

theorem projection_of_v_onto_plane :
  projection_onto_plane (2, 3, -1) (4, 2, -1) 0 = (-6/7, 11/7, -2/7) :=
by
  sorry

end projection_of_v_onto_plane_l4_4236


namespace votes_difference_l4_4638

variables (total_votes : ℕ) (john_votes james_votes jacob_votes joey_votes jack_votes jane_votes : ℕ)

-- Given conditions
def john_votes_condition := john_votes = 0.12 * total_votes
def james_votes_condition := james_votes = john_votes + 0.20 * john_votes
def jacob_votes_condition := jacob_votes = james_votes / 2
def joey_votes_condition := joey_votes = 2 * jacob_votes
def jack_votes_condition := jack_votes = 0.75 * (john_votes + jacob_votes)
def jane_votes_condition := jane_votes = jack_votes - 0.30 * jack_votes
def total_votes_condition := total_votes = 1425

-- Define the proof goal
theorem votes_difference (h1 : john_votes_condition)
                          (h2 : james_votes_condition)
                          (h3 : jacob_votes_condition)
                          (h4 : joey_votes_condition)
                          (h5 : jack_votes_condition)
                          (h6 : jane_votes_condition)
                          (h7 : total_votes_condition) :
                          jack_votes = john_votes + 35 :=
by
  sorry

end votes_difference_l4_4638


namespace triangle_proof_l4_4428

-- Definition of a point in the plane.
structure Point : Type :=
(x : ℝ)
(y : ℝ)

-- Definitions for the points involved.
variables (A B C M N K : Point)

-- Midpoint condition
def is_midpoint (M A C : Point) : Prop :=
  (M.x = (A.x + C.x) / 2) ∧ (M.y = (A.y + C.y) / 2)

-- Angular condition
noncomputable def angle (P Q R : Point) : ℝ := sorry

-- Given Conditions as Lean Definitions
def problem_conditions : Prop :=
  is_midpoint M A C ∧
  N.x = (A.x + M.x) / 2 ∧ N.y = (A.y + M.y) / 2 ∧ -- N lies on AM such that
  angle M B N = angle C B M ∧
  angle B M K = π / 2

-- Final proof statement to be proved
theorem triangle_proof (h : problem_conditions A B C M N K) : (dist B C) = (dist A K) + (dist B K) :=
sorry

end triangle_proof_l4_4428


namespace arithmetic_sequence_general_term_sum_of_bn_l4_4771

-- Define the arithmetic sequence {a_n} with its given properties.
def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∃ d, ∀ n, a (n + 1) = a n + d

-- Define the geometric sequence condition.
def forms_geometric_seq (a : ℕ → ℕ) : Prop :=
  ∃ r, r > 1 ∧ (2 * a 2) * (a 8 + 1) = (a 6) ^ 2

-- Define the sequence {b_n} in terms of {a_n}.
def b (a : ℕ → ℕ) (n : ℕ) : ℕ → ℝ :=
  λ n, a n / (2 ^ n)

-- Define the sum of the first n terms of sequence {b_n}.
def T (b : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, b (i + 1)

theorem arithmetic_sequence_general_term (a : ℕ → ℕ) (h₁ : arithmetic_sequence a) (h₂ : forms_geometric_seq a) :
  ∀ n, a n = n :=
sorry

theorem sum_of_bn (a : ℕ → ℕ) (h₁ : arithmetic_sequence a) (h₂ : forms_geometric_seq a) :
  ∀ n, T (b a) n = 2 - (n + 2) / 2^n :=
sorry

end arithmetic_sequence_general_term_sum_of_bn_l4_4771


namespace total_bricks_proof_l4_4369

-- Define the initial conditions
def initial_courses := 3
def bricks_per_course := 400
def additional_courses := 2

-- Compute the number of bricks removed from the last course
def bricks_removed_from_last_course (bricks_per_course: ℕ) : ℕ :=
  bricks_per_course / 2

-- Calculate the total number of bricks
def total_bricks (initial_courses : ℕ) (bricks_per_course : ℕ) (additional_courses : ℕ) (bricks_removed : ℕ) : ℕ :=
  (initial_courses + additional_courses) * bricks_per_course - bricks_removed

-- Given values and the proof problem
theorem total_bricks_proof :
  total_bricks initial_courses bricks_per_course additional_courses (bricks_removed_from_last_course bricks_per_course) = 1800 :=
by
  sorry

end total_bricks_proof_l4_4369


namespace find_2017th_pair_l4_4769

def sequence : ℕ → ℕ × ℕ 
| n :=
  let group := Nat.ite ((2 * n <= n * n + n)) (λ k, (k, n + 1 - k)) 
              (λ k, (n + 1 - k, k)) in
  group ((Nat.sqrt_int (2 * n) + 1) * Nat.sqrt_int (2 * n) / 2 + 1)

theorem find_2017th_pair : sequence 2017 = (1, 64) := sorry

end find_2017th_pair_l4_4769


namespace determine_constant_c_l4_4831

theorem determine_constant_c
  (f : ℝ → ℝ)
  (h0 : f = (λ x, 2 * x * (x-c)^2 + 3))
  (h1 : ∃ x, x = 2 ∧ (∀ δ > 0, ∃ ε > 0, ∀ y, abs (y - x) < ε → f x ≤ f y)) :
  c = 2 := 
sorry

end determine_constant_c_l4_4831


namespace triangle_equality_lemma_l4_4402

open EuclideanGeometry

theorem triangle_equality_lemma 
  (A B C M N K: Point)
  (hM_midpoint: midpoint M A C)
  (hN_on_AM: lies_on N A M)
  (h_angle_eq: ∠ M B N = ∠ C B M)
  (hK_right: ∠ B M K = 90) :
  dist B C = dist A K + dist B K :=
by
  sorry

end triangle_equality_lemma_l4_4402


namespace sin_120_eq_sqrt3_div_2_l4_4709

theorem sin_120_eq_sqrt3_div_2 :
  sin (120 * real.pi / 180) = real.sqrt 3 / 2 :=
sorry

end sin_120_eq_sqrt3_div_2_l4_4709


namespace colin_speed_l4_4707

variable (B T Bn C : ℝ)
variable (m : ℝ)

-- Given conditions
def condition1 := B = 1
def condition2 := T = m * B
def condition3 := Bn = T / 3
def condition4 := C = 6 * Bn
def condition5 := C = 4

-- We need to prove C = 4 given the conditions
theorem colin_speed :
  (B = 1) →
  (T = m * B) →
  (Bn = T / 3) →
  (C = 6 * Bn) →
  C = 4 :=
by
  intros _ _ _ _
  sorry

end colin_speed_l4_4707


namespace probability_hungarian_deck_correct_probability_french_deck_correct_l4_4249

noncomputable def probability_hungarian_deck : ℚ :=
  let k_I := 4 * (56) * (8 ^ 3)
  let k_II := 6 * ((28 ^ 2) * 64)
  let l := (32.choose 6)
  (k_I + k_II) / l
    
theorem probability_hungarian_deck_correct :
  probability_hungarian_deck ≈ 0.0459 :=
by
  sorry

noncomputable def probability_french_deck : ℚ :=
  let k_I := 4 * (286) * (13 ^ 3)
  let k_II := 6 * ((78 ^ 2) * 169)
  let l := (52.choose 6)
  (k_I + k_II) / l

theorem probability_french_deck_correct :
  probability_french_deck ≈ 0.426 :=
by
  sorry

end probability_hungarian_deck_correct_probability_french_deck_correct_l4_4249


namespace equalize_foma_ierema_l4_4578

theorem equalize_foma_ierema (F E Y : ℕ) 
  (h1 : F - 70 = E + 70) 
  (h2 : F - 40 = Y) 
  (h3 : Y = E + 70) 
  : ∃ x : ℕ, x = 55 ∧ F - x = E + x :=
by
  use 55
  sorry

end equalize_foma_ierema_l4_4578


namespace bowling_ball_weight_l4_4005

theorem bowling_ball_weight :
  (∀ b c : ℝ, 9 * b = 2 * c → c = 35 → b = 70 / 9) :=
by
  intros b c h1 h2
  sorry

end bowling_ball_weight_l4_4005


namespace triangle_proof_l4_4430

-- Definition of a point in the plane.
structure Point : Type :=
(x : ℝ)
(y : ℝ)

-- Definitions for the points involved.
variables (A B C M N K : Point)

-- Midpoint condition
def is_midpoint (M A C : Point) : Prop :=
  (M.x = (A.x + C.x) / 2) ∧ (M.y = (A.y + C.y) / 2)

-- Angular condition
noncomputable def angle (P Q R : Point) : ℝ := sorry

-- Given Conditions as Lean Definitions
def problem_conditions : Prop :=
  is_midpoint M A C ∧
  N.x = (A.x + M.x) / 2 ∧ N.y = (A.y + M.y) / 2 ∧ -- N lies on AM such that
  angle M B N = angle C B M ∧
  angle B M K = π / 2

-- Final proof statement to be proved
theorem triangle_proof (h : problem_conditions A B C M N K) : (dist B C) = (dist A K) + (dist B K) :=
sorry

end triangle_proof_l4_4430


namespace area_of_triangle_LRK_l4_4353

noncomputable def area_triang_lrk (JL LM JP QM : ℝ) : ℝ :=
  let PQ := JL - (JP + QM) -- Given JL and length of JP and QM
  let ratio := PQ / JL -- Ratio of sides PQ to JL
  let height_TRK := LM * (JL / PQ) -- Compute scaled height of triangle RLK
  1/2 * JL * height_TRK -- Area of triangle RLK

theorem area_of_triangle_LRK (JL LM JP QM : ℝ) (h1 : JL = 8) (h2 : LM = 4) (h3 : JP = 2) (h4 : QM = 1) :
  area_triang_lrk JL LM JP QM = 25.6 :=
by
  rw [h1, h2, h3, h4]
  let PQ := JL - (JP + QM)
  have hPQ : PQ = 5 := by linarith
  let ratio := PQ / JL
  have hRatio : ratio = 5/8 := by norm_num [ratio, JL, PQ, hPQ]
  let height_TRK := LM * (JL / PQ)
  have hHeight_TRK : height_TRK = 6.4 := by norm_num [height_TRK, LM, JL, PQ, hPQ]
  suffices : 1 / 2 * JL * height_TRK = 25.6 by
    exact this
  norm_num [JL, height_TRK, hHeight_TRK]
  sorry

end area_of_triangle_LRK_l4_4353


namespace right_triangle_ratio_l4_4719

theorem right_triangle_ratio (b e : ℝ) (h1 : b ≤ b + 2 * e)
  (h2 : b + 2 * e ≤ b + 3 * e) (h3 : (b + 3 * e) * (b + 3 * e) = b * b + (b + 2 * e) * (b + 2 * e)) :
  b / e = (1 + real.sqrt 11) / 2 :=
sorry

end right_triangle_ratio_l4_4719


namespace max_groups_needed_l4_4693

-- Definitions
def Cow := ℕ
def is_boss (A B : Cow) : Prop := sorry -- A placeholder for the boss relationship predicate

-- Main theorem
theorem max_groups_needed (cows : fin 2016) 
  (h1 : ∀ (A : Cow), ¬ is_boss A A)
  (h2 : ∀ (A B C : Cow), is_boss A B → is_boss B C → is_boss A C) : 
  ∃ G : ℕ, G = 63 :=
sorry

end max_groups_needed_l4_4693


namespace triangle_ABC_proof_l4_4435

variables {A B C M N K : Type} [inhabited A] [inhabited B] [inhabited C]
variables [between A B C] [within_segment A M] [within_segment M C] [on_segment A M]
variables [angle_eq (M B N) (C B M)] [right_angle (B M K)]
variables [midpoint_of_segment A C M]
variables [extension_of_segment B N K beyond N]

theorem triangle_ABC_proof (hM : midpoint_of_segment A C M)
    (hN : on_segment A M N)
    (hAngle : angle_eq (M B N) (C B M))
    (hPerp : right_angle (B M K))
    (hExt : extension_of_segment B N K beyond N) :
    BC = AK + BK :=
by
  sorry

end triangle_ABC_proof_l4_4435


namespace fraction_simplification_l4_4320

variable {x y z : ℝ}
variable (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z - z / x ≠ 0)

theorem fraction_simplification :
  (x^2 - 1 / y^2) / (z - z / x) = x / z :=
by
  sorry

end fraction_simplification_l4_4320


namespace triangle_angle_sum_l4_4419

variable (ABC : Type) [triangle ABC]

-- Definitions from conditions in a)
variable (A B C M N K : point ABC)
variable (M_midpoint : midpoint M A C)
variable (N_on_AM : lies_on N A M)
variable (angle_eq1 : ∠ M B N = ∠ C B M)
variable (K_ext_BN : lies_on_ext K B N)
variable (angle_eq2 : ∠ B M K = 90)

-- Goal to prove
theorem triangle_angle_sum (HC : M_midpoint) (HN : N_on_AM) (HB : angle_eq1) (HK : K_ext_BN) (HA : angle_eq2) :
  segment_length B C = segment_length A K + segment_length B K := sorry

end triangle_angle_sum_l4_4419


namespace spherical_to_rectangular_l4_4209

theorem spherical_to_rectangular {ρ θ φ x y z : ℝ} 
  (hρ : ρ = 6) (hθ : θ = 7 * Real.pi / 4) (hφ : φ = Real.pi / 4) :
  x = ρ * Real.sin φ * Real.cos θ ∧
  y = ρ * Real.sin φ * Real.sin θ ∧
  z = ρ * Real.cos φ ∧
  (x = 3 ∧ y = -3 ∧ z = 3 * Real.sqrt 2) → 
  (z ≠ x) :=
by
  intros h1 h2 h3 h4
  simp [*]
  sorry

end spherical_to_rectangular_l4_4209


namespace ratio_x_y_l4_4279

theorem ratio_x_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * log (x - 2 * y) = log x + log y) : x / y = 4 :=
by
  sorry

end ratio_x_y_l4_4279


namespace constant_term_in_expansion_l4_4947

theorem constant_term_in_expansion (x : ℝ) (hx : x ≠ 0) :
  let term (k : ℕ) := (nat.choose 6 k) * 2^k * x ^ (6 - (3 / 2) * k) in
  nat.choose 6 4 * 2^4 = 240 :=
by
  sorry

end constant_term_in_expansion_l4_4947


namespace correct_option_B_l4_4494

def linear_function (x : ℝ) : ℝ := -x + 2

theorem correct_option_B :
  ∃ x : ℝ, linear_function x = 0 ∧ x = 2 :=
by
  sorry

end correct_option_B_l4_4494


namespace equalize_foma_ierema_l4_4585

variables 
  (F E Y : ℕ)
  (h1 : F - 70 = E + 70)
  (h2 : F - 40 = Y)

def foma_should_give_ierema : ℕ := (F - E) / 2

theorem equalize_foma_ierema (F E Y : ℕ) (h1 : F - 70 = E + 70) (h2 : F - 40 = Y) :
  foma_should_give_ierema F E = 55 := 
by
  sorry

end equalize_foma_ierema_l4_4585


namespace exists_non_translatable_polygons_l4_4489

def F (polygon : Polygon) (AB: Ray) : ℝ := sorry -- Definition of F based on given conditions
   
theorem exists_non_translatable_polygons
    (PQR PQS : Polygon)
    (AB : Ray)
    (a : ℝ)
    (h_PQ_equal_area : area PQR = area PQS)
    (h_PQR_F : F PQR AB = a)
    (h_PQS_F : F PQS AB = -a) :
    ∃ (M1 M2 : Polygon), area M1 = area M2 ∧ ¬ (∃ (sm1 sm2 : Polygon), sm1 ⊆ M1 ∧ sm2 ⊆ M2 ∧ translated sm1 sm2) :=
begin
  sorry
end

end exists_non_translatable_polygons_l4_4489


namespace caroline_citrus_drinks_l4_4198

-- Definitions based on problem conditions
def citrus_drinks (oranges : ℕ) : ℕ := (oranges * 8) / 3

-- Define problem statement
theorem caroline_citrus_drinks : citrus_drinks 21 = 56 :=
by
  sorry

end caroline_citrus_drinks_l4_4198


namespace largest_n_with_triangle_property_l4_4715

def has_triangle_property (s : Set ℕ) : Prop :=
∀ {a b c d e f g h : ℕ}, -- Assume s has eight distinct elements a, b, c, d, e, f, g, h
  a ∈ s → b ∈ s → c ∈ s → d ∈ s → e ∈ s → f ∈ s → g ∈ s → h ∈ s →
  (a + b > c) ∧ (a + b > d) ∧ (a + b > e) ∧ (a + b > f) ∧ (a + b > g) ∧ (a + b > h) ∧
  (c + d > a) ∧ (c + d > b) ∧ (c + d > e) ∧ (c + d > f) ∧ (c + d > g) ∧ (c + d > h) ∧
  -- similar conditions for all pairs of elements in s

theorem largest_n_with_triangle_property :
  ∀ n : ℕ, 
  (∀ s : Set ℕ, (∀ a ∈ s, a ≥ 6) → (∀ a ∈ s, a ≤ n) → (s.card = 8) → has_triangle_property s) →
  n ≤ 138 :=
by
  intros n H
  sorry

end largest_n_with_triangle_property_l4_4715


namespace triangle_area_ratio_l4_4855

theorem triangle_area_ratio 
  (AB BC CA : ℝ)
  (p q r : ℝ)
  (ABC_area DEF_area : ℝ)
  (hAB : AB = 12)
  (hBC : BC = 16)
  (hCA : CA = 20)
  (h1 : p + q + r = 3 / 4)
  (h2 : p^2 + q^2 + r^2 = 1 / 2)
  (area_DEF_to_ABC : DEF_area / ABC_area = 385 / 512)
  : 897 = 385 + 512 := 
by
  sorry

end triangle_area_ratio_l4_4855


namespace part1_part2_l4_4854

-- Step 1: Defining the relevant points and conditions
def Point := ℝ × ℝ × ℝ -- A point in 3D space

-- Defining the points (P, A, B, C, D, and E)
def P : Point := (0, 0, 1)   -- Assuming PA is along the z-axis for simplicity
def A : Point := (0, 0, 0)
def B : Point := (1, 0, 0)
def C : Point := (1, 1, 0)
def D : Point := (0, 1, 0)

def E : Point := (1, 1, 1) -- Assuming some valid point E on PC for simplicity

-- Step 2: Defining perpendicularity relationships
def perpendicular (v1 v2 : Point) : Prop :=
    (v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3) = 0

-- Step 3: Conditions for the given problem
-- PA ⊥ plane ABCD
def PA_perp_plane_ABCD : Prop := perpendicular (P.1 - A.1, P.2 - A.2, P.3 - A.3)
                                               (B.1 - A.1, B.2 - A.2, B.3 - A.3) ∧ 
                                  perpendicular (P.1 - A.1, P.2 - A.2, P.3 - A.3)
                                               (D.1 - A.1, D.2 - A.2, D.3 - A.3)

-- DE ⊥ PC
def DE_perp_PC : Prop := perpendicular (D.1 - E.1, D.2 - E.2, D.3 - E.3)
                                      (P.1 - C.1, P.2 - C.2, P.3 - C.3)

-- Step 4: Theorem statements for each part of the problem
theorem part1 : PA_perp_plane_ABCD → DE_perp_PC → perpendicular (P.1 - C.1, P.2 - C.2, P.3 - C.3)
                                                             ((B.1 - D.1, B.2 - D.2, B.3 - D.3)) := sorry

theorem part2 : PA_perp_plane_ABCD → DE_perp_PC →
    -- When volume of E-BCD is maximized, find surface area of P-ABCD
    ∀ (E : Point), ∃ (vol_max : ℝ), surface_area_PABCD = √2 + √3 + 1 := sorry

end part1_part2_l4_4854


namespace complement_of_angle_correct_l4_4752

def complement_of_angle (a : ℚ) : ℚ := 90 - a

theorem complement_of_angle_correct : complement_of_angle (40 + 30/60) = 49 + 30/60 :=
by
  -- placeholder for the proof
  sorry

end complement_of_angle_correct_l4_4752


namespace three_digit_numbers_sorted_desc_l4_4313

theorem three_digit_numbers_sorted_desc :
  ∃ n, n = 84 ∧
    ∀ (h t u : ℕ), 100 <= 100 * h + 10 * t + u ∧ 100 * h + 10 * t + u <= 999 →
    1 ≤ h ∧ h ≤ 9 ∧ 0 ≤ t ∧ t ≤ 9 ∧ 0 ≤ u ∧ u ≤ 9 ∧ h > t ∧ t > u → 
    n = 84 := 
by
  sorry

end three_digit_numbers_sorted_desc_l4_4313


namespace symmetry_reflection_l4_4905

variables {l₁ l₂ l₃ : Type} (S : Type → Type) [Function S]

-- Hypothesis: l₃ is the image of l₂ under the reflection by l₁
def condition := l₃ = S l₁ l₂

-- Theorem: Prove the desired equality
theorem symmetry_reflection (h : condition S l₁ l₂ l₃) : S l₃ = λ x, S l₁ (S l₂ (S l₁ x)) :=
by
  sorry

end symmetry_reflection_l4_4905


namespace hit_ball_center_l4_4479

theorem hit_ball_center (a b : ℝ) (h₁ : 2 * b = 5 * a) :
  ∀ θ : ℝ, θ = real.arctan (9 / 25) →
  ∃ x y : ℝ, (hit_at_angle θ) ∧ (strikes_sides A B C D x y) ∧ (hits_center x y) :=
  sorry

def hit_at_angle (θ : ℝ) : Prop := sorry -- precise definitions of the functions omitted for now
def strikes_sides (a b c d x y : ℝ) : Prop := sorry
def hits_center (x y : ℝ) : Prop := sorry

end hit_ball_center_l4_4479


namespace simplify_expression_l4_4906

variables {a b c : ℝ}
-- Assume a, b, and c are nonzero
axiom nonzero_a : a ≠ 0
axiom nonzero_b : b ≠ 0
axiom nonzero_c : c ≠ 0

-- Definitions of x, y, z
def x : ℝ := (b / c) + 2 * (c / b)
def y : ℝ := (a / c) + 2 * (c / a)
def z : ℝ := (a / b) + 2 * (b / a)

theorem simplify_expression : x^2 + y^2 + z^2 - x * y * z = 4 :=
by
  sorry

end simplify_expression_l4_4906


namespace tetrahedron_volume_height_l4_4630

theorem tetrahedron_volume_height :
  let A1 := (0 : ℝ, -3, 1)
  let A2 := (-4 : ℝ, 1, 2)
  let A3 := (2 : ℝ, -1, 5)
  let A4 := (3 : ℝ, 1, -4)
  let vol := (32 + 1 / 3 : ℝ)
  let height := Real.sqrt (97 / 2)
  volume_of_tetrahedron A1 A2 A3 A4 = vol ∧
  height_from_vertex A4 to_plane_containing A1 A2 A3 = height :=
by
  sorry

end tetrahedron_volume_height_l4_4630


namespace ramesh_transport_cost_l4_4490

-- Definitions for conditions
def labelled_price (P : ℝ) : Prop := P = 13500 / 0.80
def selling_price (P : ℝ) : Prop := P * 1.10 = 18975
def transport_cost (T : ℝ) (extra_amount : ℝ) (installation_cost : ℝ) : Prop := T = extra_amount - installation_cost

-- The theorem statement to be proved
theorem ramesh_transport_cost (P T extra_amount installation_cost: ℝ) 
  (h1 : labelled_price P) 
  (h2 : selling_price P) 
  (h3 : extra_amount = 18975 - P)
  (h4 : installation_cost = 250) : 
  transport_cost T extra_amount installation_cost :=
by
  sorry

end ramesh_transport_cost_l4_4490


namespace combinatorial_difference_zero_combinatorial_sum_466_l4_4150

-- Statement for the first problem
theorem combinatorial_difference_zero : 
  Nat.choose 10 4 - Nat.choose 7 3 * 3.factorial = 0 :=
by
  sorry

-- Statement for the second problem
theorem combinatorial_sum_466 (n : ℕ) (h1 : 9.5 ≤ n) (h2 : n ≤ 10.5) (h3 : n = 10) :
  Nat.choose (3 * n) (38 - n) + Nat.choose (21 + n) (3 * n) = 466 :=
by
  sorry

end combinatorial_difference_zero_combinatorial_sum_466_l4_4150


namespace triangle_property_l4_4445

variables {ABC : Type*} [inner_product_space ℝ ABC]
variables (A B C M K N : ABC)

noncomputable def midpoint (A C : ABC) (M : ABC) : Prop := dist A M = dist M C

noncomputable def angle_equality (B M N : ABC) : Prop := ∠M B N = ∠C B M

noncomputable def right_angle (B M K : ABC) : Prop := ∠B M K = π / 2

theorem triangle_property (h1 : midpoint A C M) (h2 : angle_equality B M N) (h3 : right_angle B M K) : dist B C = dist A K + dist B K :=
sorry

end triangle_property_l4_4445


namespace principal_period_of_f_l4_4235

noncomputable def f (x : ℝ) : ℝ :=
  (Real.arcsin (Real.sin (Real.arccos (Real.cos (3 * x))))) ^ (-5)

theorem principal_period_of_f : ∀ x : ℝ, f (x + (π / 3)) = f x :=
by
  intro x
  sorry

end principal_period_of_f_l4_4235


namespace tangent_line_equation_ln_x_l4_4073

theorem tangent_line_equation_ln_x (y : ℝ) (x : ℝ) (h: x > 0) (slope : ℝ) (tangent_line : ℝ → ℝ) :
  (∀ x, y = log x + x + 1) →
  slope = 2 →
  tangent_line 1 = 2 →
  tangent_line = λ x, 2 * x :=
sorry

end tangent_line_equation_ln_x_l4_4073


namespace line_properties_l4_4649

def line_eq (x y : ℝ) : Prop := x / 4 + y / 3 = 1

def slope_of_line (m : ℝ) : Prop := m = -3 / 4

def midpoint_of_segment (p : ℝ × ℝ) : Prop := p = (2, 1.5)

theorem line_properties (x y m : ℝ) (p : ℝ × ℝ) :
  line_eq x y → slope_of_line m → midpoint_of_segment p := by
  intro h_line_eq
  intro h_slope
  intro h_midpoint
  /-
  Here, we associate the line equation with its slope and the midpoint
  of the segment formed by the intercepts.
  Each condition and property should be proved based on the provided information.
  -/
  sorry

end line_properties_l4_4649


namespace sqrt5_integer_decimal_part_one_plus_sqrt2_integer_decimal_part_two_plus_sqrt3_integer_decimal_part_l4_4924

-- (1) Integer part and decimal part of sqrt(5)
theorem sqrt5_integer_decimal_part :
  ∀ x : ℝ, 2 < x ∧ x < 3 → ∃ i d : ℝ, i = 2 ∧ d = x - 2 :=
by
  sorry

-- (2) Integer part and decimal part of 1 + sqrt(2)
theorem one_plus_sqrt2_integer_decimal_part :
  ∀ x : ℝ, 2 < x ∧ x < 3 → ∃ i d : ℝ, i = 2 ∧ d = x - 2 :=
by
  sorry

-- (3) The value of x - sqrt(3)y when integer part of 2 + sqrt(3) is x and decimal is y
theorem two_plus_sqrt3_integer_decimal_part :
  ∀ x : ℝ, ∃ y : ℝ, x = 3 → x - real.sqrt 3 * y = real.sqrt 3 ∧ y = real.sqrt 3 - 1 :=
by
  sorry

end sqrt5_integer_decimal_part_one_plus_sqrt2_integer_decimal_part_two_plus_sqrt3_integer_decimal_part_l4_4924


namespace eventually_periodic_l4_4534

variable (u : ℕ → ℤ)

def bounded (u : ℕ → ℤ) : Prop :=
  ∃ (m M : ℤ), ∀ (n : ℕ), m ≤ u n ∧ u n ≤ M

def recurrence (u : ℕ → ℤ) (n : ℕ) : Prop := 
  u (n) = (u (n-1) + u (n-2) + u (n-3) * u (n-4)) / (u (n-1) * u (n-2) + u (n-3) + u (n-4))

theorem eventually_periodic (hu_bounded : bounded u) (hu_recurrence : ∀ n ≥ 4, recurrence u n) :
  ∃ N M, ∀ k ≥ 0, u (N + k) = u (N + M + k) :=
sorry

end eventually_periodic_l4_4534


namespace problem_statement_l4_4892

noncomputable def max_value_d (a b c d : ℝ) : Prop :=
a + b + c + d = 10 ∧
(ab + ac + ad + bc + bd + cd = 20) ∧
∀ x, (a + b + c + x = 10 ∧ ab + ac + ad + bc + bd + cd = 20) → x ≤ 5 + Real.sqrt 105 / 2

theorem problem_statement (a b c d : ℝ) :
  max_value_d a b c d → d = (5 + Real.sqrt 105) / 2 :=
sorry

end problem_statement_l4_4892


namespace slope_of_line_l4_4612

theorem slope_of_line : ∀ (x y : ℝ), (x / 4 - y / 3 = 1) → ((3 * x / 4) - 3) = 0 → (y = (3 / 4) * x - 3) :=
by 
  intros x y h_eq h_slope 
  sorry

end slope_of_line_l4_4612


namespace triangle_right_angle_l4_4359

theorem triangle_right_angle {a b c : ℝ} {A B C : ℝ} (h : a * Real.cos A + b * Real.cos B = c * Real.cos C) :
  (A = Real.pi / 2) ∨ (B = Real.pi / 2) ∨ (C = Real.pi / 2) :=
sorry

end triangle_right_angle_l4_4359


namespace min_value_condition_l4_4790

theorem min_value_condition 
  (a b : ℝ) 
  (h1 : 4 * a + b = 1) 
  (h2 : a > 0) 
  (h3 : b > 0) : 
  ∃ x : ℝ, ∀ y : ℝ, (y = 1 - 4 * x → x = 16) := 
sorry

end min_value_condition_l4_4790


namespace product_of_20_random_digits_ends_with_zero_l4_4668

noncomputable def probability_product_ends_in_zero : ℝ := 
  (1 - (9 / 10)^20) +
  (9 / 10)^20 * (1 - (5 / 9)^20) * (1 - (8 / 9)^19)

theorem product_of_20_random_digits_ends_with_zero : 
  abs (probability_product_ends_in_zero - 0.988) < 0.001 :=
by
  sorry

end product_of_20_random_digits_ends_with_zero_l4_4668


namespace sum_distances_monotonically_decreases_l4_4930

-- Definitions used in the conditions
variables (n : ℕ) -- number of travelers
variables (P : fin n → ℝ → ℝ) -- positions of travelers as a function of time

-- Condition: the sum of pairwise distances monotonically decreases over a period of time
def pairwise_distance_sum (t : ℝ) : ℝ := 
  ∑ i j in finset.univ.filter (λ p, p.1 < p.2), abs (P i t - P j t)

-- Assuming the above sum is monotonically decreasing
axiom pairwise_distance_monotone_decreasing : 
  ∀ t1 t2, t1 ≤ t2 → pairwise_distance_sum n P t1 ≥ pairwise_distance_sum n P t2

-- Prove the sum of distances from a particular traveler to all other travelers also decreases
theorem sum_distances_monotonically_decreases :
  ∃ j : fin n, ∀ t1 t2, t1 ≤ t2 → 
  ∑ i in finset.univ.filter (λ p, p ≠ j), abs (P j t1 - P i t1) ≥ 
  ∑ i in finset.univ.filter (λ p, p ≠ j), abs (P j t2 - P i t2) :=
sorry

end sum_distances_monotonically_decreases_l4_4930


namespace comparison_f_values_l4_4757

def f (x : ℝ) := 3 * x^2 + 2^(x + 1)

def a : ℝ := 2 ^ (Real.log 11 / Real.log 2)  -- Since lg is base 10 logarithm, simplified in Lean as base change
def b : ℝ := (1 / 2) ^ (-1 / 3)
def c (t : ℝ) := t^2 - 4 * t + 9

theorem comparison_f_values (t : ℝ) (h_t : t ≥ 0) :
  f a < f (c t) ∧ f Real.pi < f (c t) :=
by
  have h_a : a = 11 := by {
    calc
      a = 2^(Real.log 11 / Real.log 2) : by rfl
      _ = 11 : by exact Real.two_pow_log_div_log_two_eq_self 11
  }
  have h_b : b = Real.cbrt 2 := by {
    calc
      b = (1 / 2) ^ (-1 / 3) : by rfl
      _ = 2^(1 / 3) : by field_simp [Real.inv_pow', h]
      _ = Real.cbrt 2 : by rfl
  }
  have h_c : ∀ t, c t ≥ 5 := by {
    intro t
    calc
      c t = (t - 2)^2 + 5 : by ring -- completes the square
      _ ≥ 5 : by nlinarith
  }
  split
  -- Proof for f(a) < f(c t)
  { calc
      f a = f 11 : by rw h_a
      _ < f (c t) : by {
        have : 11 < c t := by linarith [h_t]
        exact f_strict_mono this
      } },
  -- Proof for f(pi) < f(c t)
  { calc
      f Real.pi = f Real.pi : rfl
      _ < f (c t) : by {
        have : Real.pi < c t := by linarith [h_t]
        exact f_strict_mono this
      } }
  }

end comparison_f_values_l4_4757


namespace truth_values_of_p_and_q_l4_4783

variable (p q : Prop)

theorem truth_values_of_p_and_q
  (h1 : ¬ (p ∧ q))
  (h2 : (¬ p ∨ q)) :
  ¬ p ∧ (q ∨ ¬ q) :=
by {
  sorry
}

end truth_values_of_p_and_q_l4_4783


namespace arithmetic_sequence_sum_l4_4074

variable {α : Type*} [LinearOrder α] [AddCommMonoid α] [MulAction ℕ α] [HasDistribNeg α]

theorem arithmetic_sequence_sum :
  ∀ (a : ℕ → α) (S : ℕ → α),
    (∀ n, S n = (n + 1) * (-1:ℤ)⁻¹ * (a 0 + a n)) →
    a 0 + a 2 + a 4 + a 6 + a 8 = 55 →
    S 8 = 110 :=
by
  intros a S h1 h2
  sorry

end arithmetic_sequence_sum_l4_4074


namespace correct_option_B_l4_4495

def linear_function (x : ℝ) : ℝ := -x + 2

theorem correct_option_B :
  ∃ x : ℝ, linear_function x = 0 ∧ x = 2 :=
by
  sorry

end correct_option_B_l4_4495


namespace nissa_grooming_time_correct_l4_4869

def clipping_time_per_claw : ℕ := 10
def cleaning_time_per_ear : ℕ := 90
def shampooing_time_minutes : ℕ := 5

def claws_per_foot : ℕ := 4
def feet_count : ℕ := 4
def ear_count : ℕ := 2

noncomputable def total_grooming_time_in_seconds : ℕ := 
  (clipping_time_per_claw * claws_per_foot * feet_count) + 
  (cleaning_time_per_ear * ear_count) + 
  (shampooing_time_minutes * 60) -- converting minutes to seconds

theorem nissa_grooming_time_correct :
  total_grooming_time_in_seconds = 640 := by
  sorry

end nissa_grooming_time_correct_l4_4869


namespace total_length_of_T_l4_4382

def T : set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |(|p.1| - 3)| - 2| + |(|p.2| - 3)| - 2| = 2}

theorem total_length_of_T : 
  let L := 128 * Real.sqrt 2 in
  ∑ p in T, length_of_line_making_up_T = L := 
by sorry

end total_length_of_T_l4_4382


namespace smallest_N_l4_4399

theorem smallest_N (p q r s t u : ℕ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (hs : 0 < s) (ht : 0 < t) (hu : 0 < u)
  (h_sum : p + q + r + s + t + u = 2023) :
  ∃ N : ℕ, N = max (max (max (max (p + q) (q + r)) (r + s)) (s + t)) (t + u) ∧ N = 810 :=
sorry

end smallest_N_l4_4399


namespace side_length_of_triangle_l4_4643

-- Definitions based on conditions
def circle_area (r : ℝ) : ℝ := π * r^2

def is_equilateral (ABC : Type) [MetricSpace ABC] [Triangle ABC] :=
  ∀ (a b c : ABC), (dist a b = dist b c) ∧ (dist b c = dist c a)

def is_chord (circle : Type) [MetricSpace circle] (BC : circle)
  (O : circle) (r : ℝ) := ∃ X, dist O X = r ∧ (X ∈ BC) 

-- The conditions provided
axiom area_condition (r : ℝ) : circle_area r = 156 * π
axiom equilateral_condition (ABC : Type) [MetricSpace ABC] [Triangle ABC] : is_equilateral ABC
axiom chord_condition (circle : Type) [MetricSpace circle] (BC : circle)
 (O : circle) (r : ℝ) : is_chord circle BC O r
axiom OA_condition (O A : circle) : dist O A = 4 * sqrt 3
axiom outside_condition (O : circle) (ABC : Type) [MetricSpace ABC] [Triangle ABC] : O ∉ ABC

-- The statement of the theorem
theorem side_length_of_triangle {ABC : Type} [MetricSpace ABC] [Triangle ABC] 
  (A B C : ABC) (circle : Type) [MetricSpace circle] (O : circle) (OA : circle) (BC : circle)
  (r : ℝ) :
  circle_area r = 156 * π →
  is_equilateral ABC →
  is_chord circle BC O r →
  dist O OA = 4 * sqrt 3 →
  O ∉ ABC →
  ∃ s : ℝ, s = 6 :=
by sorry

end side_length_of_triangle_l4_4643


namespace A_n_plus_B_n_eq_2n_cubed_l4_4017

-- Definition of A_n given the grouping of positive integers
def A_n (n : ℕ) : ℕ :=
  let sum_first_n_squared := n * n * (n * n + 1) / 2
  let sum_first_n_minus_1_squared := (n - 1) * (n - 1) * ((n - 1) * (n - 1) + 1) / 2
  sum_first_n_squared - sum_first_n_minus_1_squared

-- Definition of B_n given the array of cubes of natural numbers
def B_n (n : ℕ) : ℕ := n * n * n - (n - 1) * (n - 1) * (n - 1)

-- The theorem to prove that A_n + B_n = 2n^3
theorem A_n_plus_B_n_eq_2n_cubed (n : ℕ) : A_n n + B_n n = 2 * n^3 := by
  sorry

end A_n_plus_B_n_eq_2n_cubed_l4_4017


namespace hummus_serving_amount_proof_l4_4084

/-- Given conditions: 
    one_can is the number of ounces of chickpeas in one can,
    total_cans is the number of cans Thomas buys,
    total_servings is the number of servings of hummus Thomas needs to make,
    to_produce_one_serving is the amount of chickpeas needed for one serving,
    we prove that to_produce_one_serving = 6.4 given the above conditions. -/
theorem hummus_serving_amount_proof 
  (one_can : ℕ) 
  (total_cans : ℕ) 
  (total_servings : ℕ) 
  (to_produce_one_serving : ℚ) 
  (h_one_can : one_can = 16) 
  (h_total_cans : total_cans = 8)
  (h_total_servings : total_servings = 20) 
  (h_total_ounces : total_cans * one_can = 128) : 
  to_produce_one_serving = 128 / 20 := 
by
  sorry

end hummus_serving_amount_proof_l4_4084


namespace problem_equivalence_l4_4772

noncomputable def a_n (n : ℕ) : ℤ := 2 * n + 1
noncomputable def S_n (n : ℕ) : ℤ := n^2 + 2 * n
noncomputable def b_n (n : ℕ) : ℚ := 1 / (n^2 + n)
noncomputable def T_n (n : ℕ) : ℚ := ∑ i in Finset.range n, (1 / (i + 1) - 1 / (i + 2))

theorem problem_equivalence (n : ℕ) (hn : 0 < n):
  (a_n 3 = 7) ∧ (a_n 5 + a_n 7 = 26) →
  (a_n n = 2 * n + 1) ∧ 
  (S_n n = n^2 + 2 * n) ∧
  (b_n n = 1 / (S_n n - n)) ∧ 
  (T_n n = (n : ℚ) / (n + 1)) :=
by
  sorry

end problem_equivalence_l4_4772


namespace difference_between_heads_and_feet_l4_4844

-- Definitions based on the conditions
def penguins := 30
def zebras := 22
def tigers := 8
def zookeepers := 12

-- Counting heads
def heads := penguins + zebras + tigers + zookeepers

-- Counting feet
def feet := (2 * penguins) + (4 * zebras) + (4 * tigers) + (2 * zookeepers)

-- Proving the difference between the number of feet and heads is 132
theorem difference_between_heads_and_feet : (feet - heads) = 132 :=
by
  sorry

end difference_between_heads_and_feet_l4_4844


namespace no_solution_inequality_l4_4296

theorem no_solution_inequality (a : ℝ) : (¬ ∃ x : ℝ, x > 1 ∧ x < a - 1) → a ≤ 2 :=
by
  sorry

end no_solution_inequality_l4_4296


namespace mean_equality_l4_4964

theorem mean_equality (z : ℚ) :
  (8 + 12 + 24) / 3 = (16 + z) / 2 ↔ z = 40 / 3 :=
by
  sorry

end mean_equality_l4_4964


namespace price_per_working_game_l4_4003

theorem price_per_working_game 
  (total_games : ℕ) (non_working_games : ℕ) (total_earnings : ℕ)
  (h1 : total_games = 16) (h2 : non_working_games = 8) (h3 : total_earnings = 56) :
  total_earnings / (total_games - non_working_games) = 7 :=
by {
  sorry
}

end price_per_working_game_l4_4003


namespace mode_is_six_l4_4160

variable (weekly_reading_hours : List ℕ := [2, 2, 4, 4, 4, 6, 6, 6, 6, 8])

theorem mode_is_six :
  mode weekly_reading_hours = 6 :=
sorry

end mode_is_six_l4_4160


namespace anna_pizza_fraction_l4_4167

theorem anna_pizza_fraction :
  let total_slices := 16
  let anna_eats := 2
  let shared_slices := 1
  let anna_share := shared_slices / 3
  let fraction_alone := anna_eats / total_slices
  let fraction_shared := anna_share / total_slices
  fraction_alone + fraction_shared = 7 / 48 :=
by
  sorry

end anna_pizza_fraction_l4_4167


namespace smallest_integer_is_840_l4_4915

def is_prime_digit (d : ℕ) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

def all_divide (N : ℕ) : Prop :=
  (2 ∣ N) ∧ (3 ∣ N) ∧ (5 ∣ N) ∧ (7 ∣ N)

def no_prime_digit (N : ℕ) : Prop :=
  ∀ d ∈ N.digits 10, ¬ is_prime_digit d

def smallest_satisfying_N (N : ℕ) : Prop :=
  no_prime_digit N ∧ all_divide N ∧ ∀ M, no_prime_digit M → all_divide M → N ≤ M

theorem smallest_integer_is_840 : smallest_satisfying_N 840 :=
by
  sorry

end smallest_integer_is_840_l4_4915


namespace foma_gives_ierema_55_l4_4558

-- Defining the amounts that Foma, Ierema, and Yuliy have initially
variables (F E Y : ℝ)

-- Given conditions
def condition_1 : Prop := F - E = 140
def condition_2 : Prop := F - Y = 40
def condition_3 : Prop := Y = E + 70

-- Prove that Foma should give Ierema 55 gold coins
theorem foma_gives_ierema_55 (h1 : condition_1) (h2 : condition_2) (h3 : condition_3) : 
  (F - E) / 2 = 55 :=
by
  sorry

end foma_gives_ierema_55_l4_4558


namespace sequence_general_formula_and_sum_l4_4267

theorem sequence_general_formula_and_sum (a : ℕ → ℕ) (S : ℕ → ℕ) (b : ℕ → ℕ) (T : ℕ → ℕ) :
  (a 1 = 1) →
  (∀ n : ℕ, S n = ∑ i in (Finset.range n).map Nat.succ, a i) →
  (∀ n : ℕ, 0 < n → a (n + 1) = 2 * S n + 1) →
  (∀ n : ℕ, b n = Nat.log 3 (a (n + 1))) →
  (∀ n : ℕ, T n = ∑ i in (Finset.range n).map Nat.succ, a i + b i) →
  (∀ n : ℕ, a n = 3^(n - 1)) ∧ (T n = (3^n + n^2 + n - 1) / 2) :=
by
  intros h₁ h₂ h₃ h₄ h₅
  sorry

end sequence_general_formula_and_sum_l4_4267


namespace find_enclosed_area_l4_4625

noncomputable def enclosed_area : ℝ :=
  (1 / 2) * ∫ φ in (0 : ℝ)..(π / 3), (sin φ)^2 + 
  (3 / 2) * ∫ φ in (π / 3)..(π / 2), (cos φ)^2

theorem find_enclosed_area :
  enclosed_area = (5 * π / 24) - (sqrt 3 / 4) :=
sorry

end find_enclosed_area_l4_4625


namespace magician_can_determine_area_of_convex_2008_gon_l4_4960

-- Define the problem
def can_determine_polygon_area (n : ℕ) (polygon : Fin n → ℝ × ℝ) : Prop :=
  ∃ (questions : Fin n.succ → Option ((ℕ × ℕ) ⊕ ((ℕ × ℝ) × (ℕ × ℝ)))),
  ∀ (pts : Fin (n+2) → ℝ × ℝ),
    (∀ i, pts i = polygon (n-1-mod i))
    → sorry -- This is the part where we define the rigorous mathematical condition ensuring area determination 

-- State the theorem
theorem magician_can_determine_area_of_convex_2008_gon :
  can_determine_polygon_area 2008 :=
begin
  sorry,
end

end magician_can_determine_area_of_convex_2008_gon_l4_4960


namespace volume_of_regular_tetrahedron_with_edge_length_1_l4_4125

-- We define the concepts needed: regular tetrahedron, edge length, and volume.
open Real

noncomputable def volume_of_regular_tetrahedron (a : ℝ) : ℝ :=
  let base_area := (sqrt 3 / 4) * a^2
  let height := sqrt (a^2 - (a * (sqrt 3 / 3))^2)
  (1 / 3) * base_area * height

-- The problem statement and our goal to prove:
theorem volume_of_regular_tetrahedron_with_edge_length_1 :
  volume_of_regular_tetrahedron 1 = sqrt 2 / 12 := sorry

end volume_of_regular_tetrahedron_with_edge_length_1_l4_4125


namespace sum_of_legs_of_larger_triangle_l4_4101

theorem sum_of_legs_of_larger_triangle (area_small : ℝ) (area_large : ℝ) (hypotenuse_small : ℝ) :
    (area_small = 8 ∧ area_large = 200 ∧ hypotenuse_small = 6) →
    ∃ sum_of_legs : ℝ, sum_of_legs = 41.2 :=
by
  sorry

end sum_of_legs_of_larger_triangle_l4_4101


namespace solve_quadratic_l4_4788

theorem solve_quadratic : {x : ℝ} (h : x^2 - 2 * x - 3 = x + 7) : x = 5 ∨ x = -2 :=
by
  sorry

end solve_quadratic_l4_4788


namespace same_number_of_groups_l4_4543

theorem same_number_of_groups (members : Fin 12 → Type) (groups : Type)
  (group_size : groups → ℕ)
  (H1 : ∀ g : groups, group_size g = 3 ∨ group_size g = 4)
  (member_of : members → groups → Prop)
  (H2 : ∀ (m1 m2 : members), m1 ≠ m2 → ∃! g : groups, member_of m1 g ∧ member_of m2 g)
  (num_groups : members → ℕ) :
  ∃ n, ∀ m : members, num_groups m = n :=
sorry

end same_number_of_groups_l4_4543


namespace simplify_fraction_l4_4502

theorem simplify_fraction (x : ℚ) : 
  (↑(x + 2) / 4 + ↑(3 - 4 * x) / 3 : ℚ) = ((-13 * x + 18) / 12 : ℚ) :=
by 
  sorry

end simplify_fraction_l4_4502


namespace magnification_proof_l4_4190

-- Define the conditions: actual diameter of the tissue and diameter of the magnified image
def actual_diameter := 0.0002
def magnified_diameter := 0.2

-- Define the magnification factor
def magnification_factor := magnified_diameter / actual_diameter

-- Prove that the magnification factor is 1000
theorem magnification_proof : magnification_factor = 1000 := by
  unfold magnification_factor
  unfold magnified_diameter
  unfold actual_diameter
  norm_num
  sorry

end magnification_proof_l4_4190


namespace limit_arcsin_sqrt_l4_4628

theorem limit_arcsin_sqrt :
  (Real.limit (λ x : ℝ, (Real.arcsin (3 * x)) / ((Real.sqrt (2 + x)) - Real.sqrt 2)) 0 = 6 * Real.sqrt 2) :=
sorry

end limit_arcsin_sqrt_l4_4628


namespace geometric_sequence_sum_a_l4_4263

theorem geometric_sequence_sum_a (a : ℝ) (S : ℕ → ℝ) (h : ∀ n, S n = 4^n + a) :
  a = -1 :=
sorry

end geometric_sequence_sum_a_l4_4263


namespace base_conversion_l4_4513

theorem base_conversion (b : ℝ) : (53 = 5 * 6 + 3) → (103_b = 1 * b^2 + 3) → 33 = b^2 + 3 → b = real.sqrt 30 :=
by 
  sorry

end base_conversion_l4_4513


namespace locks_and_keys_for_safe_l4_4687

theorem locks_and_keys_for_safe (n : ℕ) (required_members : ℕ) (total_members : ℕ) (locks : ℕ) (keys_per_lock : ℕ) :
  total_members = 9 →
  required_members = 6 →
  locks = Nat.choose total_members (required_members - 4) →
  keys_per_lock = (required_members - 2) →
  locks = 126 ∧ keys_per_lock = 4 :=
by
  intros h1 h2 h3 h4
  rw [←h1, ←h2, h3, h4]
  sorry

end locks_and_keys_for_safe_l4_4687


namespace sunny_ahead_in_second_race_l4_4350

variables (h d s w : ℝ)
-- Conditions
def race_condition_1 (h d : ℝ) (s w : ℝ) : Prop :=
  h / s = (h - 2*d) / w
  
def race_condition_2 (h d : ℝ) : Prop :=
  ∀ s w, race_condition_1 h d s w → d > 0 ∧ h > 0 ∧ s > 0 ∧ w > 0

-- Theorem to prove
theorem sunny_ahead_in_second_race (h d : ℝ) (hs : h > 0) (hd : d > 0) :
  ∀ s w, race_condition_1 h d s w → (s / w = h / (h - 2*d)) →
  let t := (h + 2*d) / s
  in ((s * t - (w * t)) = (4 * d^2 / h)) :=
begin
  sorry
end

end sunny_ahead_in_second_race_l4_4350


namespace number_is_fraction_l4_4328

theorem number_is_fraction (x : ℝ) : (0.30 * x = 0.25 * 40) → (x = 100 / 3) :=
begin
  intro h,
  sorry
end

end number_is_fraction_l4_4328


namespace num_shapes_folded_four_times_sum_of_areas_after_n_folds_l4_4127

-- Conditions
def initial_paper_width : ℝ := 20
def initial_paper_height : ℝ := 12
def area_one_fold : ℝ := 240
def area_two_fold : ℝ := 180

-- Number of shapes after four folds
def num_shapes_after_four_folds : Nat := 5

-- Sum of areas after n folds
def sum_of_areas (n : Nat) : ℝ := 240 * (3 - (n + 3) / 2^n)

-- Theorem statements

theorem num_shapes_folded_four_times : num_shapes_after_four_folds = 5 := by
  sorry

theorem sum_of_areas_after_n_folds (n : Nat) : ℝ := 
  240 * (3 - (n + 3) / 2^n) := by
  sorry

end num_shapes_folded_four_times_sum_of_areas_after_n_folds_l4_4127


namespace balloons_remain_intact_l4_4645

   theorem balloons_remain_intact (total_balloons : ℕ)
                                  (initial_blow_up_fraction : ℚ)
                                  (second_blow_up_multiplier : ℚ) :
     total_balloons = 200 →
     initial_blow_up_fraction = 1 / 5 →
     second_blow_up_multiplier = 2 →
     let first_blow_up := initial_blow_up_fraction * total_balloons in
     let second_blow_up := second_blow_up_multiplier * first_blow_up in
     let remain_after_first := total_balloons - first_blow_up in
     let remain_after_second := remain_after_first - second_blow_up in
     remain_after_second = 80 := 
   by
     sorry
   
end balloons_remain_intact_l4_4645


namespace min_positive_t_l4_4519

def sin_period (ω : ℝ) : ℝ :=
  2 * Real.pi / ω

theorem min_positive_t (t : ℕ) :
  (∃ x1 x2 : ℝ, 0 ≤ x1 ∧ x1 ≤ t ∧ 0 ≤ x2 ∧ x2 ≤ t ∧ 
  sin (Real.pi / 3 * x1) = 1 ∧ sin (Real.pi / 3 * x2) = 1 ∧ x1 ≠ x2) → 
  t ≥ 8 :=
by
  sorry

end min_positive_t_l4_4519


namespace valid_four_digit_numbers_l4_4062

def is_valid_number (n : ℕ) : Prop := 
  let a := n / 1000
  let b := (n % 1000) / 100
  let c := (n % 100) / 10
  let d := n % 10
  (100 * b + 10 * c + a + d) * 10 = 100 * a + 10 * b + c + d ∧ a ≠ 0

theorem valid_four_digit_numbers :
  ∀ n, is_valid_number n ↔ n ∈ {2019, 3028, 4037, 5046, 6055, 7064, 8073, 9082} :=
by sorry

end valid_four_digit_numbers_l4_4062


namespace coral_remaining_pages_l4_4210

def pages_after_week1 (total_pages : ℕ) : ℕ :=
  total_pages / 2

def pages_after_week2 (remaining_pages_week1 : ℕ) : ℕ :=
  remaining_pages_week1 - (3 * remaining_pages_week1 / 10)

def pages_after_week3 (remaining_pages_week2 : ℕ) (reading_hours : ℕ) (reading_speed : ℕ) : ℕ :=
  remaining_pages_week2 - (reading_hours * reading_speed)

theorem coral_remaining_pages (total_pages remaining_pages_week1 remaining_pages_week2 remaining_pages_week3 : ℕ) 
  (reading_hours reading_speed unread_pages : ℕ)
  (h1 : total_pages = 600)
  (h2 : remaining_pages_week1 = pages_after_week1 total_pages)
  (h3 : remaining_pages_week2 = pages_after_week2 remaining_pages_week1)
  (h4 : reading_hours = 10)
  (h5 : reading_speed = 15)
  (h6 : remaining_pages_week3 = pages_after_week3 remaining_pages_week2 reading_hours reading_speed)
  (h7 : unread_pages = remaining_pages_week3) :
  unread_pages = 60 :=
by
  sorry

end coral_remaining_pages_l4_4210


namespace _l4_4736

noncomputable theorem common_tangent_vector :
    (∃ s t : ℝ, (y1 : ℝ → ℝ) (y1 x) = e^x - 1 ∧ (y2 : ℝ → ℝ) (y2 x) = Real.log (x + 1) ∧ 
    (dy1 : ℝ → ℝ) (dy1 x) = Real.exp x ∧ (dy2 : ℝ → ℝ) (dy2 x) = 1 / (x + 1) ∧ 
    dy1 s = dy2 t ∧ e^s - s * e^s + e^s - 1 = -t / (t + 1) + Real.log (t + 1) ∧ t = 0) → 
    directional_vector = (1, 1) := 
sorry

end _l4_4736


namespace total_length_of_lines_in_T_l4_4376

def T (x y : ℝ) : Prop :=
  abs (abs (abs x - 3) - 2) + abs (abs (abs y - 3) - 2) = 2

theorem total_length_of_lines_in_T : (∑ (x y : ℝ), T x y) = 64 * real.sqrt 2 := 
sorry

end total_length_of_lines_in_T_l4_4376


namespace subtraction_result_l4_4937

theorem subtraction_result :
  let x := 567.89
  let y := 123.45
  (x - y) = 444.44 :=
by
  sorry

end subtraction_result_l4_4937


namespace euler_children_mean_age_l4_4511

-- Define the ages of each child
def ages : List ℕ := [8, 8, 8, 13, 13, 16]

-- Define the total number of children
def total_children := 6

-- Define the correct sum of ages
def total_sum_ages := 66

-- Define the correct answer (mean age)
def mean_age := 11

-- Prove that the mean (average) age of these children is 11
theorem euler_children_mean_age : (List.sum ages) / total_children = mean_age :=
by
  sorry

end euler_children_mean_age_l4_4511


namespace triangle_angle_sum_l4_4424

variable (ABC : Type) [triangle ABC]

-- Definitions from conditions in a)
variable (A B C M N K : point ABC)
variable (M_midpoint : midpoint M A C)
variable (N_on_AM : lies_on N A M)
variable (angle_eq1 : ∠ M B N = ∠ C B M)
variable (K_ext_BN : lies_on_ext K B N)
variable (angle_eq2 : ∠ B M K = 90)

-- Goal to prove
theorem triangle_angle_sum (HC : M_midpoint) (HN : N_on_AM) (HB : angle_eq1) (HK : K_ext_BN) (HA : angle_eq2) :
  segment_length B C = segment_length A K + segment_length B K := sorry

end triangle_angle_sum_l4_4424


namespace calculate_spring_decrease_l4_4672

variable (initial_members : ℕ) (fall_increase_percent total_change_percent : ℝ)

def fall_members (initial_members : ℕ) (fall_increase_percent : ℝ) : ℝ :=
  initial_members + (initial_members * fall_increase_percent / 100)

def spring_members (initial_members : ℕ) (total_change_percent : ℝ) : ℝ :=
  initial_members + (initial_members * total_change_percent / 100)

def percentage_decrease_in_spring (fall_members spring_members : ℝ) : ℝ :=
  (fall_members - spring_members) / fall_members * 100

theorem calculate_spring_decrease :
  initial_members = 100 →
  fall_increase_percent = 7 →
  total_change_percent = -13.33 →
  percentage_decrease_in_spring (fall_members initial_members fall_increase_percent)
                                (spring_members initial_members total_change_percent)
  ≈ 19 :=
by
  intros h_initial h_fall h_total
  rw [h_initial, h_fall, h_total]
  sorry

end calculate_spring_decrease_l4_4672


namespace set_intersection_l4_4000

open Set

variable (x : ℝ)

def U : Set ℝ := univ
def A : Set ℝ := { x | |x - 1| > 2 }
def B : Set ℝ := { x | x^2 - 6 * x + 8 < 0 }

theorem set_intersection (x : ℝ) : x ∈ (U \ A) ∩ B ↔ 2 < x ∧ x ≤ 3 := sorry

end set_intersection_l4_4000


namespace fraction_sum_eq_one_l4_4196

theorem fraction_sum_eq_one (m n : ℝ) (h : m ≠ n) : (m / (m - n) + n / (n - m) = 1) :=
by
  sorry

end fraction_sum_eq_one_l4_4196


namespace total_boys_in_class_l4_4680

theorem total_boys_in_class :
  ∃ N : ℕ, 
    (N / 2 = 27 - 7 ∨ N / 2 = N - (27 - 7)) ∧ 
    (N / 2 = 36 - 11 ∨ N / 2 = N - (36 - 11)) ∧ 
    (N / 2 = 42 - 15 ∨ N / 2 = N - (42 - 15)) ∧ 
    N = 54 :=
by
  let N := 54
  use N
  split
  case left => 
    split
    case left => 
      exact or.inl (by norm_num)
    case right => 
      split
      case left => 
        exact or.inl (by norm_num)
      case right =>
        exact or.inl (by norm_num)
  split
  case left => 
    split
    case left =>
      exact or.inr (by norm_num)
    case right =>
      split
      case left => 
        exact or.inr (by norm_num)
      case right =>
        exact or.inl (by norm_num)
  split
  case left =>
    split
    case left => 
      exact or.inl (by norm_num)
    case right =>
      split
      case left =>
        exact or.inr (by norm_num)
      case right =>
        exact or.inl (by norm_num)

-- Adding sorry to skip the proof part, as discussed
sorry

end total_boys_in_class_l4_4680


namespace cos_sum_of_arctan_roots_l4_4075

theorem cos_sum_of_arctan_roots (α β : ℝ) (hα : -π/2 < α ∧ α < 0) (hβ : -π/2 < β ∧ β < 0) 
  (h1 : Real.tan α + Real.tan β = -3 * Real.sqrt 3) 
  (h2 : Real.tan α * Real.tan β = 4) : 
  Real.cos (α + β) = - 1 / 2 :=
sorry

end cos_sum_of_arctan_roots_l4_4075


namespace find_valid_digits_l4_4364

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem find_valid_digits :
  ∃ (a b c : ℕ), 
  (a, b, c) ∈ { (1, 3, 7), (3, 7, 1), (7, 1, 3), (1, 7, 3), (7, 3, 1), (3, 1, 7) } ∧ 
  (is_prime (10 * a + b)) ∧ (is_prime (10 * b + a)) ∧
  (is_prime (10 * b + c)) ∧ (is_prime (10 * c + b)) ∧
  (is_prime (10 * c + a)) ∧ (is_prime (10 * a + c)) :=
by
  sorry

end find_valid_digits_l4_4364


namespace euler_polyhedron_l4_4458

-- Define the necessary concepts about the convex polyhedron
variables {M K N : ℕ} -- M: number of faces, K: number of edges, N: number of vertices

-- Euler's formula for convex polyhedron
theorem euler_polyhedron (h : convex_polyhedron M K N) : M - K + N = 2 := 
sorry -- Proof is omitted

end euler_polyhedron_l4_4458


namespace equal_share_each_shopper_l4_4863

theorem equal_share_each_shopper 
  (amount_giselle : ℕ)
  (amount_isabella : ℕ)
  (amount_sam : ℕ)
  (H1 : amount_isabella = amount_sam + 45)
  (H2 : amount_isabella = amount_giselle + 15)
  (H3 : amount_giselle = 120) : 
  (amount_isabella + amount_sam + amount_giselle) / 3 = 115 :=
by
  -- The proof is omitted.
  sorry

end equal_share_each_shopper_l4_4863


namespace estimate_planes_l4_4310

noncomputable def numberOfPlanes : ℝ → ℝ
| 15 := 134
| _  := 0

theorem estimate_planes :
  ∃ n : ℝ, numberOfPlanes 15 = n :=
begin
  use 134,
  sorry
end

end estimate_planes_l4_4310


namespace equalize_foma_ierema_l4_4573

theorem equalize_foma_ierema (F E Y : ℕ) 
  (h1 : F - 70 = E + 70) 
  (h2 : F - 40 = Y) 
  (h3 : Y = E + 70) 
  : ∃ x : ℕ, x = 55 ∧ F - x = E + x :=
by
  use 55
  sorry

end equalize_foma_ierema_l4_4573


namespace distinct_scores_count_l4_4641

def points (a b c : ℕ) : ℕ :=
  a + 2 * b + 3 * c

def valid_shots (a b c : ℕ) : Prop :=
  a + b + c = 8

def num_unique_scores : ℕ :=
  (finset.univ.filter (λ n, ∃ a b c, valid_shots a b c ∧ points a b c = n)).card

theorem distinct_scores_count : num_unique_scores = 19 :=
sorry

end distinct_scores_count_l4_4641


namespace A_alone_work_days_l4_4828

-- Define the problem conditions
variables (W : ℝ) (A B : ℝ) -- Work amount and rates for A and B

-- Given conditions
def is_thrice_as_fast (A B : ℝ) : Prop := A = 3 * B
def combined_work_rate (A B : ℝ) : Prop := A + B = W / 21

-- The fact to be proved
theorem A_alone_work_days (A B : ℝ) (W : ℝ) (h1 : is_thrice_as_fast A B) (h2 : combined_work_rate A B) : 
  W / A = 28 :=
sorry

end A_alone_work_days_l4_4828


namespace true_propositions_l4_4395

variables (a b : ℝ^3 → ℝ^3)  -- non-coincident lines represented as mappings
variables (α β : set (ℝ^3))  -- non-coincident planes represented as sets in ℝ^3
variables [plane α] [plane β] -- asserting that α and β are indeed planes

-- Propositions
def proposition2 (a b : ℝ^3 → ℝ^3) (α : set (ℝ^3)) [plane α] : Prop :=
  (∀ p, p ∈ α → a p = p) ∧ (∀ p, p ∈ α → b p = p) → (∀ p q, a p = a q)

def proposition4 (a : ℝ^3 → ℝ^3) (α β : set (ℝ^3)) [plane α] [plane β] : Prop :=
  (∀ p, p ∈ α → a p = p) ∧ (∀ p, p ∈ β → a p = p) → ∀ p q, p ∈ α → q ∈ β → α.parallel β

-- The true propositions are 2 and 4
theorem true_propositions (a b : ℝ^3 → ℝ^3) (α β : set (ℝ^3)) [plane α] [plane β] :
  proposition2 a b α ∧ proposition4 a α β :=
by
  sorry

end true_propositions_l4_4395


namespace foma_should_give_ierema_l4_4598

variables (F E Y : ℕ)

-- Given conditions in Lean definitions
def condition1 := F - 70 = E + 70
def condition2 := F - 40 = Y

-- Final statement to be proven
theorem foma_should_give_ierema : 
  condition1 → condition2 → F - E = 110 → F - E = 110 / 2 + 55 :=
by
  intros h1 h2 h3
  sorry

end foma_should_give_ierema_l4_4598


namespace waiter_earnings_l4_4691

theorem waiter_earnings (total_customers : ℕ) (no_tip_customers : ℕ) (tip_per_customer : ℕ)
  (h1 : total_customers = 10)
  (h2 : no_tip_customers = 5)
  (h3 : tip_per_customer = 3) :
  (total_customers - no_tip_customers) * tip_per_customer = 15 :=
by sorry

end waiter_earnings_l4_4691


namespace total_length_T_l4_4390

def T : set (ℝ × ℝ) := { p | ∃ x y : ℝ, p = (x, y) ∧ 
  (| (| x | - 3) - 2 | + | (| y | - 3) - 2 | = 2) }

theorem total_length_T : 
  (∃ l : ℝ, l = 128 * real.sqrt 2 ∧ ∀ (p ∈ T), ∃ s : list (set (ℝ × ℝ)), 
    (p ∈ ⋃₀ set_of (λ t, t ∈ s) ∧ 
    (∀ t ∈ s, is_diamond t) ∧ 
    (∀ t ∈ s, perimeter t = 8 * real.sqrt 2) ∧ 
    length_of_lines s = l)) :=
sorry

end total_length_T_l4_4390


namespace solve_system_of_equations_l4_4508

theorem solve_system_of_equations :
  ∃ x1 x2 x3 x4 : ℝ, 
  (x1 + 2 * x2 + 3 * x3 + x4 = 1) ∧
  (3 * x1 + 13 * x2 + 13 * x3 + 5 * x4 = 3) ∧
  (3 * x1 + 7 * x2 + 7 * x3 + 2 * x4 = 12) ∧
  (x1 + 5 * x2 + 3 * x3 + x4 = 7) ∧
  (4 * x1 + 5 * x2 + 6 * x3 + x4 = 19) ∧
  (x1, x2, x3, x4) = (4, 2, 0, -7) := 
begin
  use [4, 2, 0, -7],
  split,
  { sorry },
  split,
  { sorry },
  split,
  { sorry },
  split,
  { sorry },
  { sorry }
end

end solve_system_of_equations_l4_4508


namespace find_b_l4_4851

theorem find_b (FA DC FE BC : Line) (b : ℝ)
  (h1 : Parallel FA DC) (h2 : Parallel FE BC) :
  b = 73 := 
  sorry

end find_b_l4_4851


namespace simplify_to_ellipse_l4_4932

theorem simplify_to_ellipse (x y : ℝ) :
  (Real.sqrt ((x - 2)^2 + y^2) + Real.sqrt ((x + 2)^2 + y^2) = 10) →
  (x^2 / 25 + y^2 / 21 = 1) :=
by
  sorry

end simplify_to_ellipse_l4_4932


namespace tooth_fairy_left_amount_l4_4126

-- Define the values of the different types of coins
def quarter_value : ℝ := 0.25
def half_dollar_value : ℝ := 0.50
def dime_value : ℝ := 0.10

-- Define the number of each type of coins Joan received
def num_quarters : ℕ := 14
def num_half_dollars : ℕ := 14
def num_dimes : ℕ := 14

-- Calculate the total values for each type of coin
def total_quarters_value : ℝ := num_quarters * quarter_value
def total_half_dollars_value : ℝ := num_half_dollars * half_dollar_value
def total_dimes_value : ℝ := num_dimes * dime_value

-- The total amount of money left by the tooth fairy
def total_amount_left := total_quarters_value + total_half_dollars_value + total_dimes_value

-- The theorem stating that the total amount is $11.90
theorem tooth_fairy_left_amount : total_amount_left = 11.90 := by 
  sorry

end tooth_fairy_left_amount_l4_4126


namespace probability_non_expired_probability_at_least_one_expired_l4_4865

-- Define the initial conditions
def total_bottles : ℕ := 6
def expired_bottles : ℕ := 2
def non_expired_bottles : ℕ := total_bottles - expired_bottles

-- (I) Prove the probability of drawing a non-expired bottle
theorem probability_non_expired :
  (non_expired_bottles : ℚ) / total_bottles = 2 / 3 :=
begin
  sorry
end

-- (II) Prove the probability of drawing at least one expired bottle
theorem probability_at_least_one_expired :
  (9 : ℚ) / 15 = 3 / 5 :=
begin
  sorry
end

end probability_non_expired_probability_at_least_one_expired_l4_4865


namespace cos_sub_identity_l4_4779

theorem cos_sub_identity (A B : ℝ) (h1 : sin A + sin B = 1/2) (h2 : cos A + cos B = 1) : 
  cos (A - B) = -3 / 8 := 
by 
  sorry

end cos_sub_identity_l4_4779


namespace emails_in_afternoon_l4_4872

variable (e_m e_t e_a : Nat)
variable (h1 : e_m = 3)
variable (h2 : e_t = 8)

theorem emails_in_afternoon : e_a = 5 :=
by
  -- (Proof steps would go here)
  sorry

end emails_in_afternoon_l4_4872


namespace equalize_foma_ierema_l4_4575

theorem equalize_foma_ierema (F E Y : ℕ) 
  (h1 : F - 70 = E + 70) 
  (h2 : F - 40 = Y) 
  (h3 : Y = E + 70) 
  : ∃ x : ℕ, x = 55 ∧ F - x = E + x :=
by
  use 55
  sorry

end equalize_foma_ierema_l4_4575


namespace volume_of_dug_out_earth_l4_4134

theorem volume_of_dug_out_earth
  (diameter depth : ℝ)
  (h_diameter : diameter = 2) 
  (h_depth : depth = 14) 
  : abs ((π * (1 / 2 * diameter / 2) ^ 2 * depth) - 44) < 0.1 :=
by
  -- Provide a placeholder for the proof
  sorry

end volume_of_dug_out_earth_l4_4134


namespace workers_days_not_worked_l4_4619

theorem workers_days_not_worked (W N : ℕ) (h1 : W + N = 30) (h2 : 100 * W - 25 * N = 0) : N = 24 :=
sorry

end workers_days_not_worked_l4_4619


namespace given_roots_find_coefficients_l4_4244

theorem given_roots_find_coefficients {a b c : ℝ} :
  (1:ℝ)^5 + 2*(1)^4 + a * (1:ℝ)^2 + b * (1:ℝ) = c →
  (-1:ℝ)^5 + 2*(-1:ℝ)^4 + a * (-1:ℝ)^2 + b * (-1:ℝ) = c →
  a = -6 ∧ b = -1 ∧ c = -4 :=
by
  intros h1 h2
  sorry

end given_roots_find_coefficients_l4_4244


namespace number_of_paths_l4_4713

-- Conditions as definitions
def width : ℕ := 7
def height : ℕ := 3

-- The question translated to a math proof problem
theorem number_of_paths (w h : ℕ) (A B : fin (w + 1) × fin (h + 1)) 
  (A_at_bottom_left : A = (⟨0, by simp⟩, ⟨0, by simp⟩)) 
  (B_at_top_right : B = (⟨w, by simp⟩, ⟨h, by simp⟩)) : 
  (w = width) → (h = height) → 
  (finset.card (finset.univ.filter (λ p : (w + h).choose w, p.1 + p.2 = w + h)) = 120) :=
by 
  intros hw hh
  have step_count := hw.symm ▸ hh.symm ▸ 10
  calc finset.card (finset.univ.filter (λ p : (width + height).choose width, p.1 + p.2 = width + height)) 
      = (width + height).choose width : by sorry
      ... = 120 : by norm_num [nat.choose]

#check number_of_paths

end number_of_paths_l4_4713


namespace root_conditions_l4_4031

noncomputable def polynomial : ℤ → ℤ → ℤ → ℤ → (ℤ → ℤ) :=
  λ a b r s, λ x, x^3 + a*x^2 + b*x + 16*a

theorem root_conditions (a b r s : ℤ) (h₁ : a ≠ 0) (h₂ : b ≠ 0)
  (h₃ : polynomial a b r s = (λ x, (x - r)^2 * (x - s)))
  (h₄ : (x - r)^2 * (x - s) = x^3 - 2*r*x^2 + (r^2 + r*s)*x - r^2*s)
  (h₅ : -2*r = a)
  (h₆ : r^2 + r*s = b)
  (h₇ : -r^2*s = 16*a) :
  |a * b| = 272 :=
  sorry

end root_conditions_l4_4031


namespace total_feed_amount_l4_4092

theorem total_feed_amount (x : ℝ) : 
  (17 * 0.18) + (x * 0.53) = (17 + x) * 0.36 → 17 + x = 35 :=
by
  intros h
  sorry

end total_feed_amount_l4_4092


namespace equalize_foma_ierema_l4_4580

variables 
  (F E Y : ℕ)
  (h1 : F - 70 = E + 70)
  (h2 : F - 40 = Y)

def foma_should_give_ierema : ℕ := (F - E) / 2

theorem equalize_foma_ierema (F E Y : ℕ) (h1 : F - 70 = E + 70) (h2 : F - 40 = Y) :
  foma_should_give_ierema F E = 55 := 
by
  sorry

end equalize_foma_ierema_l4_4580


namespace tangent_intersections_l4_4054

noncomputable def f (x : ℝ) (ω : ℝ) : ℝ := Real.tan (ω * x)

theorem tangent_intersections (ω > 0) 
  (h1 : ∀ x1 x2, f x1 ω = 2 → f x2 ω = 2 → |x2 - x1| = π / 2) :
  f (π / 6) 2 = Real.sqrt 3 := 
by
  sorry

end tangent_intersections_l4_4054


namespace proof_N_is_12_l4_4481

/-- 
  Let N be a number such that:
  1. One half of N exceeds its one fourth by 3.
  2. The sum of the digits of N is 3.
  Prove that N = 12.
-/
noncomputable def proof_problem (N : ℕ) : Prop :=
  (N / 2 = N / 4 + 3) ∧ ((N.digits.sum = 3) → N = 12)

theorem proof_N_is_12 (N : ℕ) (h1 : N / 2 = N / 4 + 3) (h2 : N.digits.sum = 3) : N = 12 :=
  sorry

end proof_N_is_12_l4_4481


namespace simplified_form_of_expression_l4_4214

theorem simplified_form_of_expression (x : ℝ) :
  (sqrt (4 * x^2 - 8 * x + 4) + sqrt (4 * x^2 + 8 * x + 4) + 5) = 
  (2 * |x - 1| + 2 * |x + 1| + 5) := 
sorry

end simplified_form_of_expression_l4_4214


namespace triangle_angle_sum_l4_4423

variable (ABC : Type) [triangle ABC]

-- Definitions from conditions in a)
variable (A B C M N K : point ABC)
variable (M_midpoint : midpoint M A C)
variable (N_on_AM : lies_on N A M)
variable (angle_eq1 : ∠ M B N = ∠ C B M)
variable (K_ext_BN : lies_on_ext K B N)
variable (angle_eq2 : ∠ B M K = 90)

-- Goal to prove
theorem triangle_angle_sum (HC : M_midpoint) (HN : N_on_AM) (HB : angle_eq1) (HK : K_ext_BN) (HA : angle_eq2) :
  segment_length B C = segment_length A K + segment_length B K := sorry

end triangle_angle_sum_l4_4423


namespace angle_between_chords_l4_4357

theorem angle_between_chords
    (O A M N : Point) (R : ℝ) (α β : ℝ)
    (hOn_sphere : dist O A = R ∧ dist O M = R ∧ dist O N = R)
    (h_equal_chords : dist A M = dist A N)
    (h_angle_to_diameter : ∠A O M = α ∧ ∠A O N = α)
    (h_angle_MN : ∠M O N = β) :
    ∠M A N = 2 * real.arcsin (real.sin (β / 2) / (2 * real.cos α)) :=
sorry

end angle_between_chords_l4_4357


namespace largest_apartment_size_l4_4192

theorem largest_apartment_size (cost_per_sqft : ℝ) (budget : ℝ) (s : ℝ) 
    (h₁ : cost_per_sqft = 1.20) 
    (h₂ : budget = 600) 
    (h₃ : 1.20 * s = 600) : 
    s = 500 := 
  sorry

end largest_apartment_size_l4_4192


namespace triangle_equality_BC_AK_BK_l4_4415

-- Define the main geometric constructs
variables {A B C M N K : Type}

-- Given conditions
variables [Midpoint M A C] -- M is the midpoint of AC
variables [OnSegment N A M] -- N lies on segment AM
variables [EqualAngle M B N C B M] -- ∠MBN = ∠CBM
variables [OnExtension K B N (90 : Angle)] -- K is on the extension of BN such that ∠BMK is right

-- Define the main theorem
theorem triangle_equality_BC_AK_BK :
  BC = AK + BK :=
sorry

end triangle_equality_BC_AK_BK_l4_4415


namespace count_two_digit_prime_sum_l4_4317

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def reverse_digits (N : ℕ) : ℕ :=
  let t := N / 10
  let u := N % 10
  10 * u + t

def prime_sum_condition (N : ℕ) : Prop :=
  is_prime (N + reverse_digits N)

def two_digit_number (N : ℕ) : Prop :=
  10 ≤ N ∧ N < 100

theorem count_two_digit_prime_sum : (finset.filter prime_sum_condition (finset.Ico 10 100)).card = 1 :=
sorry

end count_two_digit_prime_sum_l4_4317


namespace range_of_a_l4_4614

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x ∈ Icc (-1:ℝ) 2 → a ≥ x^2 - 2 * x - 1) ↔ (a ≥ 2) :=
by
  sorry

end range_of_a_l4_4614


namespace value_of_a1_plus_a3_l4_4821

theorem value_of_a1_plus_a3 (a a1 a2 a3 a4 : ℝ) :
  (∀ x : ℝ, (1 + x)^4 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4) →
  a1 + a3 = 8 :=
by
  sorry

end value_of_a1_plus_a3_l4_4821


namespace number_of_subsets_of_A_plus_B_l4_4213

def set_plus (A B : Set ℕ) : Set ℕ := { z | ∃ x ∈ A, ∃ y ∈ B, z = x + y }

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {0, 1}

theorem number_of_subsets_of_A_plus_B : (set_plus A B).toFinset.card.powerset.card = 16 := 
by 
  sorry

end number_of_subsets_of_A_plus_B_l4_4213


namespace knowledge_challenge_probabilities_knowledge_challenge_comparative_prob_l4_4608

theorem knowledge_challenge_probabilities
  (p q : ℝ)
  (h1 : 1 - (1 - p) * (1 - q) = 5 / 6)
  (h2 : p * q = 1 / 3)
  (h3 : p > q) :
  p = 2 / 3 ∧ q = 1 / 2 := 
  sorry

theorem knowledge_challenge_comparative_prob
  (p q : ℝ)
  (h1 : p = 2 / 3)
  (h2 : q = 1 / 2) :
  let Pm_lt_n := ((1 - p) ^ 2 * 2 * (1 - q) * q + (1 - p) ^ 2 * q ^ 2 + 2 * (1 - p) * p * q ^ 2) in
  Pm_lt_n = 7 / 36 :=
  sorry

end knowledge_challenge_probabilities_knowledge_challenge_comparative_prob_l4_4608


namespace valid_permutation_count_l4_4176

def num_ways (n : ℕ) : ℕ := n!

def speaker_permutations (total_speakers : ℕ) (condition : (total_speakers ≥ 5)) : list (list ℕ) :=
  -- To simplify, we assume a list [1, 2, 3, 4, 5] corresponds to Dr. White at place 1, etc.
  (list.permutations (list.range (total_speakers))).filter (λ l, list.index_of 1 l < list.index_of 2 l)

theorem valid_permutation_count : (n : ℕ) (h : n = 5) (perm : list (list ℕ)) (total : ℕ)
  (condition_1 : total_speakers ≥ 5) (condition_2 : perm = speaker_permutations 5 condition_1)
  (condition_3 : total = num_ways n) :
  perm.length = total / 2 := sorry

end valid_permutation_count_l4_4176


namespace convex_pentagon_property_l4_4498

theorem convex_pentagon_property
  (A B C D E : Type) [convex_pentagon A B C D E]
  : ∃ (X : Type), 
    (∃ (oppX : Type), distance_to_side X oppX < 
     distance_to_side (adjacent_vertex1 X) oppX + 
     distance_to_side (adjacent_vertex2 X) oppX) := 
sorry

end convex_pentagon_property_l4_4498


namespace find_a_l4_4762

-- Define the domain of real numbers
variable (a : ℝ)
variable (f : ℝ → ℝ)

-- Define the function f
def f_def : f = λ x, (Real.sin x - |a|) := sorry

-- Define the condition that f is odd
def odd_function (g : ℝ → ℝ) := ∀ x, g (-x) = -g x

-- Now, the theorem to be proved
theorem find_a
  (h : odd_function f)
  (h_def : f = λ x, Real.sin x - |a|) : a = 0 :=
begin
  sorry -- proof is omitted
end

end find_a_l4_4762


namespace triangle_proof_l4_4433

-- Definition of a point in the plane.
structure Point : Type :=
(x : ℝ)
(y : ℝ)

-- Definitions for the points involved.
variables (A B C M N K : Point)

-- Midpoint condition
def is_midpoint (M A C : Point) : Prop :=
  (M.x = (A.x + C.x) / 2) ∧ (M.y = (A.y + C.y) / 2)

-- Angular condition
noncomputable def angle (P Q R : Point) : ℝ := sorry

-- Given Conditions as Lean Definitions
def problem_conditions : Prop :=
  is_midpoint M A C ∧
  N.x = (A.x + M.x) / 2 ∧ N.y = (A.y + M.y) / 2 ∧ -- N lies on AM such that
  angle M B N = angle C B M ∧
  angle B M K = π / 2

-- Final proof statement to be proved
theorem triangle_proof (h : problem_conditions A B C M N K) : (dist B C) = (dist A K) + (dist B K) :=
sorry

end triangle_proof_l4_4433


namespace polynomial_division_l4_4896

noncomputable def f (x : ℝ) : ℝ := 3 * x ^ 4 + 8 * x ^ 3 - 4 * x ^ 2 + 2 * x + 6
noncomputable def d (x : ℝ) : ℝ := x ^ 2 + 2 * x - 3
noncomputable def q (x : ℝ) : ℝ := 3 * x ^ 2 + 2 * x
noncomputable def r (x : ℝ) : ℝ := -6 * x + 3

theorem polynomial_division :
  f(2) = q(2) * d(2) + r(2) ∧ f(-2) = q(-2) * d(-2) + r(-2) → q(2) + r(-2) = 31 :=
by
  intro h
  sorry

end polynomial_division_l4_4896


namespace incorrect_propositions_l4_4682

-- Define the conditions
def condition_1 (a : ℝ) : Prop := ∃ x y : ℝ, x + y = 3 - a ∧ x * y = a ∧ x > 0 ∧ y < 0

def condition_2 : Prop := ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → sqrt (x^2 - 1) + sqrt (1 - x^2) = sqrt (x^2 - 1) + sqrt (1 - x^2)

def condition_3 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, -2 ≤ f x ∧ f x ≤ 2

def condition_4 (a : ℝ) : Prop := ∃ m : ℕ, m = 1 ∧ ∃ x : ℝ, |3 - x^2| = a

-- Define what needs to be proven i.e., the incorrect propositions
theorem incorrect_propositions (f : ℝ → ℝ) (a : ℝ) :
  ¬condition_2 ∧ ¬condition_3 (f ∘ (λ x, x + 1)) ∧ ¬condition_4 a :=
by
  sorry

end incorrect_propositions_l4_4682


namespace selling_price_is_correct_l4_4873

-- Define all the conditions as constants
def cost_per_widget : ℝ := 3
def monthly_rent : ℝ := 10000
def worker_payment_per_worker : ℝ := 2500
def number_of_workers : ℕ := 4
def tax_rate : ℝ := 0.2
def number_of_widgets_sold : ℕ := 5000
def total_profit : ℝ := 4000

-- Compute total payment to workers
def total_worker_payment : ℝ := worker_payment_per_worker * number_of_workers

-- Calculate total expenses excluding cost of widgets and taxes
def total_expenses_excluding_widgets_and_taxes : ℝ := monthly_rent + total_worker_payment

-- Calculate cost of widgets
def cost_of_widgets : ℝ := number_of_widgets_sold * cost_per_widget

-- Calculate taxes
def taxes : ℝ := tax_rate * total_profit

-- Calculate the total expenses including the cost of widgets and taxes
def total_expenses_including_widgets_and_taxes : ℝ := total_expenses_excluding_widgets_and_taxes + cost_of_widgets + taxes

-- Calculate total revenue
def total_revenue : ℝ := total_expenses_including_widgets_and_taxes + total_profit

-- Calculate selling price per widget
def selling_price_per_widget : ℝ := total_revenue / number_of_widgets_sold

-- Prove that Jenna sells each widget for $7.96
theorem selling_price_is_correct : selling_price_per_widget = 7.96 :=
by {
  -- Proof omitted
  sorry
}

end selling_price_is_correct_l4_4873


namespace problem_solution_l4_4148

-- Define the necessary conditions
def f (x : ℤ) : ℤ := 2 * x^4 + 3 * x^3 + 5 * x - 4

-- Define the main theorem
theorem problem_solution :
  (Nat.gcd 840 1785 = 105) ∧ (f 2 = 62) :=
by {
  -- We include sorry here to indicate that the proof is omitted.
  sorry
}

end problem_solution_l4_4148


namespace Joey_swimming_days_l4_4676

-- Define the conditions and required proof statement
theorem Joey_swimming_days (E : ℕ) (h1 : 3 * E / 4 = 9) : E / 2 = 6 :=
by
  sorry

end Joey_swimming_days_l4_4676


namespace line_parallel_or_contained_l4_4334

variable {Point : Type}
variable {Line : Type}
variable {Plane : Type}

-- Definitions for perpendicular, parallel, and containment relationships
variable perpendicular : Line → Plane → Prop
variable parallel : Line → Plane → Prop
variable contained_in : Line → Plane → Prop

-- Variables for the given line and planes
variable l : Line
variable α β : Plane

-- Given conditions
axiom h1 : perpendicular l β
axiom h2 : perpendicular α β

-- The theorem to prove
theorem line_parallel_or_contained :
  parallel l α ∨ contained_in l α :=
sorry

end line_parallel_or_contained_l4_4334


namespace find_values_f_zero_l4_4739

theorem find_values_f_zero :
  ∀ (f : ℝ → ℝ), 
  (∀ x y, f(x + y) = f(x) * f(y) / f(x * y)) → 
  (f(0) = 0) ∨ (∃ c ≠ 0, f(0) = c) := 
by
  sorry

end find_values_f_zero_l4_4739


namespace number_of_right_triangles_l4_4954

-- Define the points and the rectangle
variables (E F G H R S : Type)

-- Define that EFGH forms a rectangle
axiom rectangle_EFGH : ∀ (E F G H : Type), (rectangle E F G H)

-- Define that RS divides the rectangle into two congruent rectangles
axiom RS_divides_rect : ∀ (RS E F G H : Type), (divides RS (rectangle E F G H) into (congruent_rectangles E F R S) (congruent_rectangles G H R S))

-- Statement to prove the number of right triangles
theorem number_of_right_triangles : ∃ (E F G H R S : Type), 
  rectangle E F G H ∧ divides RS (rectangle E F G H) into (congruent_rectangles E F R S) (congruent_rectangles G H R S) ∧ 
  (count_right_triangles E F G H R S = 12) := 
sorry

end number_of_right_triangles_l4_4954


namespace arithmetic_geometric_sequence_sum_formula_l4_4773

noncomputable theory
open_locale classical

-- Define the arithmetic and geometric sequences and their properties
def arithmetic_sequence (d : ℕ) : (ℕ → ℕ) := λ n, 1 + (n - 1) * d
def geometric_sequence (q : ℕ) : (ℕ → ℕ) := λ n, q^(n - 1)

-- Define the sequences a_n and b_n based on the problem conditions
def a_n (d : ℕ) (n : ℕ) := arithmetic_sequence d n
def b_n (q : ℕ) (n : ℕ) := geometric_sequence q n 

-- Define c_n as the product of a_n and b_n
def c_n (d q : ℕ) (n : ℕ) := a_n d n * b_n q n

-- Define S_n as the sum of the first n terms of c_n
def S_n (d q : ℕ) (n : ℕ) := ∑ i in finset.range n, c_n d q (i + 1)

-- Theorem stating our equivalent math problem
theorem arithmetic_geometric_sequence_sum_formula (d q n : ℕ) :
  (d = 0 ∧ q = 1 → S_n d q n = n) ∧ 
  (d = 2 ∧ q = 3 → S_n d q n = (n-1) * 3^n / 2 + 1 / 2) :=
by { sorry }

end arithmetic_geometric_sequence_sum_formula_l4_4773


namespace three_digit_multiples_of_15_not_45_count_l4_4311

theorem three_digit_multiples_of_15_not_45_count : 
  (∃ n : ℕ, n = 40 ∧ 
    let multiples_15 := {x | 100 ≤ x ∧ x ≤ 999 ∧ x % 15 = 0} in
    let multiples_45 := {x | 100 ≤ x ∧ x ≤ 999 ∧ x % 45 = 0} in
    n = (multiples_15 \ multiples_45).card) :=
by
  sorry

end three_digit_multiples_of_15_not_45_count_l4_4311


namespace triangle_equality_lemma_l4_4405

open EuclideanGeometry

theorem triangle_equality_lemma 
  (A B C M N K: Point)
  (hM_midpoint: midpoint M A C)
  (hN_on_AM: lies_on N A M)
  (h_angle_eq: ∠ M B N = ∠ C B M)
  (hK_right: ∠ B M K = 90) :
  dist B C = dist A K + dist B K :=
by
  sorry

end triangle_equality_lemma_l4_4405


namespace linear_expressions_constant_multiple_l4_4243

theorem linear_expressions_constant_multiple 
    (a b c p q r : ℝ)
    (h : (a*x + p)^2 + (b*x + q)^2 = (c*x + r)^2) : 
    a*b ≠ 0 → p*q ≠ 0 → (a / b = p / q) :=
by
  -- Given: (ax + p)^2 + (bx + q)^2 = (cx + r)^2
  -- Prove: a / b = p / q, implying that A(x) and B(x) can be expressed as the constant times C(x)
  sorry

end linear_expressions_constant_multiple_l4_4243


namespace inverse_variation_y_squared_sqrt_z_l4_4039

theorem inverse_variation_y_squared_sqrt_z (k : ℝ) :
  (∀ y z : ℝ, y^2 * sqrt z = k) →
  (∃ y z : ℝ, y = 3 ∧ z = 4 ∧ y^2 * sqrt z = k) →
  (∃ z : ℝ, (6 : ℝ)^2 * sqrt z = k ∧ z = 1/4) :=
by
  intros h₁ h₂
  sorry

end inverse_variation_y_squared_sqrt_z_l4_4039


namespace largest_value_of_M_l4_4242

theorem largest_value_of_M (a S : ℤ) (hS : S ≠ 0) : 
  let D := (0, 0)
  let E := (3 * S, 0)
  let F := (3 * S - 2, 35)
  let parabola := ∃ b c, ∀ x y, (x = 0 ∧ y = 0) ∨ (x = 3 * S ∧ y = 0) ∨ (x = 3 * S - 2 ∧ y = 35) → y = a * x^2 + b * x + c
  ∃ M : ℚ, parabola ∧ (∑ i in [D, E, F], a * i.1) / (2 * S) = M ∧ M = 1485 / 4 :=
sorry

end largest_value_of_M_l4_4242


namespace find_time_to_match_avg_speed_l4_4509

-- Given conditions and average speed
variables (t v : ℝ)
variables (h1 : v = 88.00333333333333)  -- Given average speed, though incorrect

-- Definitions of total distance and total time
def distance : ℝ := 40 * t + 240
def time : ℝ := t + 3

-- Definition of average speed
def avg_speed : ℝ := distance t / time t

-- Proof statement
theorem find_time_to_match_avg_speed : avg_speed t = v → False :=
by
  intros h
  rw [avg_speed, distance, time] at h
  sorry

end find_time_to_match_avg_speed_l4_4509


namespace greatest_integer_sequence_l4_4068

def sequence (u : ℕ → ℚ) : Prop :=
  u 0 = 2 ∧ u 1 = 5/2 ∧ ∀ n, u (n + 1) = u n * (u (n - 1) ^ 2 - 2) - u 1

theorem greatest_integer_sequence (u : ℕ → ℚ)
  (h : sequence u)
  (n : ℕ) (hn: n > 0) :
  ⌊u n⌋ = 2 ^ (2 ^ n - (-1) ^ n) / 3 :=
sorry

end greatest_integer_sequence_l4_4068


namespace sum_of_solutions_l4_4122

-- Define the quadratic equation as a product of linear factors
def quadratic_eq (x : ℚ) : Prop := (4 * x + 6) * (3 * x - 8) = 0

-- Define the roots of the quadratic equation
def root1 : ℚ := -3 / 2
def root2 : ℚ := 8 / 3

-- Sum of the roots of the quadratic equation
def sum_of_roots : ℚ := root1 + root2

-- Theorem stating that the sum of the roots is 7/6
theorem sum_of_solutions : sum_of_roots = 7 / 6 := by
  sorry

end sum_of_solutions_l4_4122


namespace problem_statement_l4_4033

-- Given: x, y, z are real numbers such that x < 0 and x < y < z
variables {x y z : ℝ} 

-- Conditions
axiom h1 : x < 0
axiom h2 : x < y
axiom h3 : y < z

-- Statement to prove: x + y < y + z
theorem problem_statement : x + y < y + z :=
by {
  sorry
}

end problem_statement_l4_4033


namespace total_length_of_T_l4_4381

def T : set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |(|p.1| - 3)| - 2| + |(|p.2| - 3)| - 2| = 2}

theorem total_length_of_T : 
  let L := 128 * Real.sqrt 2 in
  ∑ p in T, length_of_line_making_up_T = L := 
by sorry

end total_length_of_T_l4_4381


namespace smallest_m_divisible_54_and_8_l4_4165

noncomputable def reverse_digits (m : ℕ) : ℕ :=
let digits := m.toString.data in
(digits.reverse.asString.toNat : ℕ)

theorem smallest_m_divisible_54_and_8 (m : ℕ) (n : ℕ) :
  digits_count m = 4 → 
  (∀ d, is_digit d → is_digit (reverse_digit d)) →
  (m % 54 = 0) →
  (m % 8 = 0) →
  (n = reverse_digits m) →
  (n % 54 = 0) →
  m >= 1000 →
  m <= 9999 →
  m = 1080 :=
by
  sorry

end smallest_m_divisible_54_and_8_l4_4165


namespace line_through_intersection_of_circles_l4_4952

theorem line_through_intersection_of_circles :
  ∀ (x y : ℝ),
    (x^2 + y^2 + 4 * x - 4 * y - 12 = 0) ∧
    (x^2 + y^2 + 2 * x + 4 * y - 4 = 0) →
    (x - 4 * y - 4 = 0) :=
by sorry

end line_through_intersection_of_circles_l4_4952


namespace paint_cube_cost_l4_4138

-- Definitions for the problem
def edge_length := 10
def area_per_quart := 120
def cost_per_quart := 3.20

-- The function to compute the cost of painting the cube
def paint_cost (edge_length : ℕ) (area_per_quart : ℕ) (cost_per_quart : ℝ) : ℝ :=
  let face_area := edge_length * edge_length
  let total_area := 6 * face_area
  let quarts_needed := total_area / area_per_quart
  quarts_needed * cost_per_quart

-- The proof statement that painting the cube costs $16.00
theorem paint_cube_cost : paint_cost edge_length area_per_quart cost_per_quart = 16 := by
  sorry

end paint_cube_cost_l4_4138


namespace cannot_determine_y_coordinate_due_to_insufficient_information_l4_4355

-- Define the assumptions
def passes_through_pts (x1 x2 xintr : ℝ) : Prop :=
  ∃ m b, m * x1 + b = y1 ∧ m * x2 + b = y2 ∧ m * xintr + b = 0

-- Lean statement for the problem:
theorem cannot_determine_y_coordinate_due_to_insufficient_information (x1 x2 xintr : ℝ) :
  passes_through_pts x1 x2 xintr →
  (xintr = 4) →
  (x1 = -10) →
  (x2 = 10) →
  ∃ m b, m * (-10) + b = y1 :=
sorry

end cannot_determine_y_coordinate_due_to_insufficient_information_l4_4355


namespace range_of_a_l4_4274

noncomputable def p (a : ℝ) := ∀ x : ℝ, x^2 + a ≥ 0
noncomputable def q (a : ℝ) := ∃ x : ℝ, x^2 + (2 + a) * x + 1 = 0

theorem range_of_a (a : ℝ) : (p a ∧ q a) → (a ≥ 0) := by
  sorry

end range_of_a_l4_4274


namespace principal_period_of_f_l4_4234

noncomputable def f (x : ℝ) : ℝ :=
  (Real.arcsin (Real.sin (Real.arccos (Real.cos (3 * x))))) ^ (-5)

theorem principal_period_of_f : ∀ x : ℝ, f (x + (π / 3)) = f x :=
by
  intro x
  sorry

end principal_period_of_f_l4_4234


namespace number_of_possible_teams_l4_4194

-- Define the number of girls and boys in the math club
def num_girls : ℕ := 5
def num_boys : ℕ := 8

-- Define how many girls and boys should be in the team
def girls_in_team : ℕ := 3
def boys_in_team : ℕ := 2

-- Define the combination function
def combination (n k : ℕ) : ℕ :=
  if h : k ≤ n then (finset.range n).card.combination k else 0

-- Main theorem to prove the number of different possible teams
theorem number_of_possible_teams : 
  combination num_girls 1 * combination (num_girls - 1) (girls_in_team - 1) * combination num_boys boys_in_team = 840 := 
by sorry

end number_of_possible_teams_l4_4194


namespace polynomial_abs_sum_l4_4252

theorem polynomial_abs_sum {a : ℕ → ℤ} (h : (λ x : ℤ, (2 - x)^2023) = (λ x : ℤ, ∑ i in finset.range 2024, a i * (x + 1)^i)) :
  (finset.range 2024).sum (λ i, |a i|) = 2^4046 :=
by
  sorry

end polynomial_abs_sum_l4_4252


namespace length_vector_P1P7_l4_4792

noncomputable def point_intersection (n : ℕ) : ℝ := 
  if n % 2 = 0 then (2 * n + 1) * π / 12 
  else (2 * n + 5) * π / 12 

-- Predicate to check that point n+6 is at (n+6)th position in intersection sequence
def points_distance (n : ℕ) : ℝ := abs (point_intersection (n + 6) - point_intersection n)

theorem length_vector_P1P7: points_distance 1 = 3 * π :=
by
  sorry

end length_vector_P1P7_l4_4792


namespace probability_of_product_ending_with_zero_l4_4666
open BigOperators

def probability_product_ends_with_zero :=
  let no_zero := (9 / 10) ^ 20
  let at_least_one_zero := 1 - no_zero
  let no_even := (5 / 9) ^ 20
  let at_least_one_even := 1 - no_even
  let no_five_among_19 := (8 / 9) ^ 19
  let at_least_one_five := 1 - no_five_among_19
  let no_zero_and_conditions :=
    no_zero * at_least_one_even * at_least_one_five
  at_least_one_zero + no_zero_and_conditions

theorem probability_of_product_ending_with_zero :
  abs (probability_product_ends_with_zero - 0.988) < 0.001 :=
by
  sorry

end probability_of_product_ending_with_zero_l4_4666


namespace ball_hits_ground_at_4_5_seconds_l4_4640

-- Define the initial conditions
def initial_velocity : ℝ := 32  -- feet per second
def initial_height : ℝ := 180   -- feet
def height_equation (t : ℝ) : ℝ := -16 * t^2 + 32 * t + 180

-- Prove that the ball will hit the ground at t = 4.5 seconds
theorem ball_hits_ground_at_4_5_seconds : ∃ t : ℝ, height_equation t = 0 ∧ t = 4.5 :=
by
  use 4.5
  split
  {
    -- Show that the height at t = 4.5 is 0
    calc height_equation 4.5
      = -16 * (4.5)^2 + 32 * 4.5 + 180 : by rfl
      = -16 * 20.25 + 144 + 180       : by norm_num
      = -324 + 144 + 180              : by norm_num
      = 0                             : by norm_num
  }
  {
    -- State the value of t
    refl
  }

end ball_hits_ground_at_4_5_seconds_l4_4640


namespace triangle_property_l4_4448

variables {ABC : Type*} [inner_product_space ℝ ABC]
variables (A B C M K N : ABC)

noncomputable def midpoint (A C : ABC) (M : ABC) : Prop := dist A M = dist M C

noncomputable def angle_equality (B M N : ABC) : Prop := ∠M B N = ∠C B M

noncomputable def right_angle (B M K : ABC) : Prop := ∠B M K = π / 2

theorem triangle_property (h1 : midpoint A C M) (h2 : angle_equality B M N) (h3 : right_angle B M K) : dist B C = dist A K + dist B K :=
sorry

end triangle_property_l4_4448


namespace sum_f_eq_sqrt3_l4_4318

def f (n : ℕ) : ℝ :=
  Real.tan (n * Real.pi / 3)

theorem sum_f_eq_sqrt3 :
  (∑ n in Finset.range 100, f (n + 1)) = Real.sqrt 3 :=
by
  sorry

end sum_f_eq_sqrt3_l4_4318


namespace Jerry_age_l4_4912

theorem Jerry_age (M J : ℕ) (h1 : M = 2 * J - 6) (h2 : M = 22) : J = 14 :=
by
  sorry

end Jerry_age_l4_4912


namespace max_min_value_among_elements_l4_4798

def matrix : List (List Nat) := [
  [11, 17, 25, 19, 16],
  [24, 10, 13, 15, 3],
  [12, 5, 14, 2, 18],
  [23, 4, 1, 8, 22],
  [6, 20, 7, 21, 9]
]

theorem max_min_value_among_elements :
  ∃ (elements : List Nat), 
  (∀ i j, i ≠ j → matrix[i] j ∉ elements) ∧ 
  (∃ minElem ∈ elements, ∀ e ∈ elements, minElem ≤ e) ∧ 
  17 ≥ minElem := by
  sorry

end max_min_value_among_elements_l4_4798


namespace fibonacci_p_arithmetic_periodic_l4_4487

-- Define p-arithmetic system and its properties
def p_arithmetic (p : ℕ) : Prop :=
  ∀ (a : ℤ), a ≠ 0 → a^(p-1) = 1

-- Define extraction of sqrt(5)
def sqrt5_extractable (p : ℕ) : Prop :=
  ∃ (r : ℝ), r^2 = 5

-- Define Fibonacci sequence in p-arithmetic
def fibonacci_p_arithmetic (p : ℕ) (v : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, v (n+2) = v (n+1) + v n

-- Main Theorem
theorem fibonacci_p_arithmetic_periodic (p : ℕ) (v : ℕ → ℤ) :
  p_arithmetic p →
  sqrt5_extractable p →
  fibonacci_p_arithmetic p v →
  (∀ k : ℕ, v (k + p) = v k) :=
by
  intros _ _ _
  sorry

end fibonacci_p_arithmetic_periodic_l4_4487


namespace zero_point_necessary_but_not_sufficient_condition_l4_4759

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := (sqrt (4 - x^2) / (x + 4)) - m

theorem zero_point_necessary_but_not_sufficient_condition
  (m : ℝ) (h1 : ∃ x : ℝ, f x m = 0)
  (h2 : |m| ≤ sqrt 3 / 3) :
  (∀ m : ℝ, (∃ x : ℝ, f x m = 0) ↔ |m| ≤ sqrt 3 / 3) ∧
  ¬ (∀ m : ℝ, |m| ≤ sqrt 3 / 3 → ∃ x : ℝ, f x m = 0) :=
sorry

end zero_point_necessary_but_not_sufficient_condition_l4_4759


namespace number_of_lines_through_five_points_l4_4307

def is_valid_point (p : ℕ × ℕ × ℕ) : Prop :=
  let (i, j, k) := p in 1 ≤ i ∧ i ≤ 5 ∧ 1 ≤ j ∧ j ≤ 5 ∧ 1 ≤ k ∧ k ≤ 5

def direction_vector_valid (a b c : ℤ) : Prop :=
  (-1 ≤ a ∧ a ≤ 1) ∧ (-1 ≤ b ∧ b ≤ 1) ∧ (-1 ≤ c ∧ c ≤ 1) ∧ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0)

def make_point (start : ℕ × ℕ × ℕ) (dir : ℤ × ℤ × ℤ) (n : ℕ) : ℕ × ℕ × ℕ :=
  let (i, j, k) := start
  let (a, b, c) := dir
  ((i : ℤ + n * a), (j : ℤ + n * b), (k : ℤ + n * c))

def within_bounds (p : ℕ × ℕ × ℕ) : Prop :=
  let (i, j, k) := p in 1 ≤ i ∧ i ≤ 5 ∧ 1 ≤ j ∧ j ≤ 5 ∧ 1 ≤ k ∧ k ≤ 5

theorem number_of_lines_through_five_points : 
  (set.count (λ l : (ℕ × ℕ × ℕ) × (ℤ × ℤ × ℤ),
    let (start, dir) := l
    is_valid_point start ∧ direction_vector_valid dir ∧
    (∀ n, n = 0 → is_valid_point (make_point start dir n) ∧ 
             n = 1 → is_valid_point (make_point start dir n) ∧ 
             n = 2 → is_valid_point (make_point start dir n) ∧
             n = 3 → is_valid_point (make_point start dir n) ∧ 
             n = 4 → is_valid_point (make_point start dir n)
    ) = 150 :=
sorry

end number_of_lines_through_five_points_l4_4307


namespace principal_period_function_l4_4233

noncomputable def f (x : ℝ) : ℝ := (Real.arcsin (Real.sin (Real.arccos (Real.cos (3 * x))))) ^ (-5)

theorem principal_period_function :
  ∀ x : ℝ, f (x + π/3) = f x :=
sorry

end principal_period_function_l4_4233


namespace x_minus_y_div_x_eq_4_7_l4_4822

-- Definitions based on the problem's conditions
axiom y_div_x_eq_3_7 (x y : ℝ) : y / x = 3 / 7

-- The main problem to prove
theorem x_minus_y_div_x_eq_4_7 (x y : ℝ) (h : y / x = 3 / 7) : (x - y) / x = 4 / 7 := by
  sorry

end x_minus_y_div_x_eq_4_7_l4_4822


namespace angle_EFD_70_l4_4362

-- Define the conditions from the problem
def pointsDandE (A B C D E F : Type) [MetricSpace D] [MetricSpace E] [MetricSpace F] :=
  D ∈ Line A B ∧ E ∈ Line A C ∧ DE ∥ BC

-- Define the angles provided in the problem
def givenAngles (BAC ABC : ℝ) : Prop := 
  BAC = 50 ∧ ABC = 60

-- Define the target angle to prove based on the conditions
def measureAngleEFD (A B C D E F : Type) [MetricSpace D] [MetricSpace E] [MetricSpace F] 
   (BAC ABC EFD : ℝ) : Prop :=
  pointsDandE A B C D E F ∧ givenAngles BAC ABC → EFD = 70

-- Lean statement
theorem angle_EFD_70 (A B C D E F : Type) [MetricSpace D] [MetricSpace E] [MetricSpace F] 
  (BAC ABC EFD : ℝ) :
  measureAngleEFD A B C D E F BAC ABC EFD :=
by
  sorry

end angle_EFD_70_l4_4362


namespace possible_colorings_l4_4153

open Set

/-- Define the set of natural numbers starting from 1 -/
def PosInt : Set ℕ := {n : ℕ | n ≥ 1}

/-- Definition for the coloring function -/
def coloring (c : ℕ → Bool) : Prop :=
  (∀ n m ∈ PosInt, c n = c m → c (n + m) = true)

theorem possible_colorings (c : ℕ → Bool) :
  coloring c →
  ∃ t : ℕ, ∀ n : ℕ, (n % 2 = 0 ∨ n > 2 * t ∨ c n = true) :=
by
  sorry

end possible_colorings_l4_4153


namespace max_value_of_f_g_shifted_is_f_l4_4292

def f (x : ℝ) : ℝ := 
  2 * (Real.cos (x + π/2))^2 + Real.sin (2 * x + π/6) - 1

def g (x : ℝ) : ℝ := 
  Real.sin (2 * x)

theorem max_value_of_f :
  ∃ x : ℝ, f x = 1 :=
sorry

theorem g_shifted_is_f :
  ∀ x : ℝ, g (x + π/12) = f x :=
sorry

end max_value_of_f_g_shifted_is_f_l4_4292


namespace each_shopper_receives_equal_amount_l4_4861

variables (G I S total_final : ℝ)

-- Given conditions
def conditions : Prop :=
  G = 120 ∧
  I = G + 15 ∧
  I = S + 45

noncomputable def amount_each_shopper_receives : ℝ :=
  let total := I + S + G in total / 3

theorem each_shopper_receives_equal_amount (h : conditions) : amount_each_shopper_receives G I S = 115 := by
  -- Given conditions for Giselle, Isabella, and Sam
  rcases h with ⟨hG, hI1, hI2⟩
    
  -- Define total_final from conditions
  let total_final := G + (G + 15) + (G + 15 - 45)
  
  -- Default proof
  sorry

end each_shopper_receives_equal_amount_l4_4861


namespace foma_should_give_ierema_55_coins_l4_4571

variables (F E Y : ℝ)

-- Conditions
def condition1 : Prop := F - 70 = E + 70
def condition2 : Prop := F - 40 = Y
def condition3 : Prop := E + 70 = Y

theorem foma_should_give_ierema_55_coins
  (h1 : condition1 F E) (h2 : condition2 F Y) (h3 : condition3 E Y) :
  F - (E + 55) = E + 55 :=
by
  sorry

end foma_should_give_ierema_55_coins_l4_4571


namespace total_length_of_T_l4_4383

def T : set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |(|p.1| - 3)| - 2| + |(|p.2| - 3)| - 2| = 2}

theorem total_length_of_T : 
  let L := 128 * Real.sqrt 2 in
  ∑ p in T, length_of_line_making_up_T = L := 
by sorry

end total_length_of_T_l4_4383


namespace exists_infinite_repeated_sum_of_digits_l4_4766

-- Define the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the sequence a_n which is the sum of digits of P(n)
def a (P : ℕ → ℤ) (n : ℕ) : ℕ :=
  sum_of_digits (P n).natAbs

theorem exists_infinite_repeated_sum_of_digits (P : ℕ → ℤ) (h_nat_coeffs : ∀ n, (P n) ≥ 0) :
  ∃ s : ℕ, ∀ N : ℕ, ∃ n : ℕ, n ≥ N ∧ a P n = s :=
sorry

end exists_infinite_repeated_sum_of_digits_l4_4766


namespace total_length_T_l4_4388

def T : set (ℝ × ℝ) := { p | ∃ x y : ℝ, p = (x, y) ∧ 
  (| (| x | - 3) - 2 | + | (| y | - 3) - 2 | = 2) }

theorem total_length_T : 
  (∃ l : ℝ, l = 128 * real.sqrt 2 ∧ ∀ (p ∈ T), ∃ s : list (set (ℝ × ℝ)), 
    (p ∈ ⋃₀ set_of (λ t, t ∈ s) ∧ 
    (∀ t ∈ s, is_diamond t) ∧ 
    (∀ t ∈ s, perimeter t = 8 * real.sqrt 2) ∧ 
    length_of_lines s = l)) :=
sorry

end total_length_T_l4_4388


namespace solve_for_x_l4_4504

theorem solve_for_x (x : ℝ) : (4 * 5^x = 5000) → x = 4 :=
by
  intro h
  sorry

end solve_for_x_l4_4504


namespace greatest_multiple_of_4_with_cube_less_than_1728_l4_4938

theorem greatest_multiple_of_4_with_cube_less_than_1728 :
  ∃ (x : ℕ), (x > 0) ∧ (x % 4 = 0) ∧ (x^3 < 1728) ∧ (∀ y, (y > 0) ∧ (y % 4 = 0) ∧ (y^3 < 1728) → y ≤ x) := 
begin
  sorry
end

end greatest_multiple_of_4_with_cube_less_than_1728_l4_4938


namespace puzzles_and_board_games_count_l4_4077

def num_toys : ℕ := 200
def num_action_figures : ℕ := num_toys / 4
def num_dolls : ℕ := num_toys / 3

theorem puzzles_and_board_games_count :
  num_toys - num_action_figures - num_dolls = 84 := 
  by
    -- TODO: Prove this theorem
    sorry

end puzzles_and_board_games_count_l4_4077


namespace overall_difference_l4_4358

def total_students_A := 1300
def total_students_B := 1100
def total_students_C := 1500

def freshmen_percent_A := 0.80
def sophomores_percent_A := 0.50

def freshmen_percent_B := 0.75
def sophomores_percent_B := 0.45

def freshmen_percent_C := 0.70
def sophomores_percent_C := 0.40

def freshmen_A := freshmen_percent_A * total_students_A
def sophomores_A := sophomores_percent_A * total_students_A

def freshmen_B := freshmen_percent_B * total_students_B
def sophomores_B := sophomores_percent_B * total_students_B

def freshmen_C := freshmen_percent_C * total_students_C
def sophomores_C := sophomores_percent_C * total_students_C

def total_freshmen := freshmen_A + freshmen_B + freshmen_C
def total_sophomores := sophomores_A + sophomores_B + sophomores_C

theorem overall_difference : total_freshmen - total_sophomores = 1170 :=
  by
    sorry -- proof not required

end overall_difference_l4_4358


namespace basketball_team_starters_l4_4482

open Finset

noncomputable def binom (n k : ℕ) : ℕ := nat.choose n k

theorem basketball_team_starters :
  ∃ (team : Finset ℕ), card team = 16 ∧
  ∃ (quadruplets : Finset ℕ), (card quadruplets = 4 ∧ quadruplets ⊆ team) ∧
  (∀ (starters : Finset ℕ), card starters = 7 →
    ∃ (chosen_from_quadruplets : Finset ℕ) (chosen_from_others : Finset ℕ),
    chosen_from_quadruplets ⊆ quadruplets ∧
    card chosen_from_quadruplets = 3 ∧
    chosen_from_others ⊆ (team \ quadruplets) ∧
    card chosen_from_others = 4 →
    (chosen_from_quadruplets ∪ chosen_from_others) = starters →
    ∑ _ in finset.range 1, (binom 4 3 * binom 12 4) = 1980) :=
sorry

end basketball_team_starters_l4_4482


namespace employees_count_l4_4941

theorem employees_count (n : ℕ) (avg_salary : ℝ) (manager_salary : ℝ)
  (new_avg_salary : ℝ) (total_employees_with_manager : ℝ) : 
  avg_salary = 1500 → 
  manager_salary = 3600 → 
  new_avg_salary = avg_salary + 100 → 
  total_employees_with_manager = (n + 1) * 1600 → 
  (n * avg_salary + manager_salary) / total_employees_with_manager = new_avg_salary →
  n = 20 := by
  intros
  sorry

end employees_count_l4_4941


namespace number_of_team_members_l4_4195

theorem number_of_team_members (x x1 x2 : ℕ) (h₀ : x = x1 + x2) (h₁ : 3 * x1 + 4 * x2 = 33) : x = 6 :=
sorry

end number_of_team_members_l4_4195


namespace true_discount_approx_l4_4065

noncomputable def true_discount (BD PW : ℝ) : ℝ :=
BD / (1 + BD / PW)

theorem true_discount_approx :
  true_discount 37.62 800 ≈ 35.92 :=
by
  sorry

end true_discount_approx_l4_4065


namespace range_of_t_l4_4523

theorem range_of_t (t : ℝ) :
  (∀ x : ℝ, (1 ≤ 12 / (x + 3) ∧ 12 / (x + 3) ≤ 3) → x ∈ set.Icc 1 9) ∧
  (∀ t : ℝ, t ≠ 0 → ∀ x : ℝ, x^2 + 3 * t * x - 4 * t ^ 2 < 0 → 
    (t > 0 → x ∈ set.Ioo (-4 * t) t) ∧ (t < 0 → x ∈ set.Ioo t (-4 * t))) ∧
  (∀ x : ℝ, 1 ≤ 12 / (x + 3) ∧ 12 / (x + 3) ≤ 3 → set.Icc 1 9) → 
  (set.Icc 1 9).measure = 8 →
  t ∈ set.Iic (-9/4) ∨ t ∈ set.Ici 9 := 
by
  sorry

end range_of_t_l4_4523


namespace total_feed_amount_l4_4091

theorem total_feed_amount (x : ℝ) : 
  (17 * 0.18) + (x * 0.53) = (17 + x) * 0.36 → 17 + x = 35 :=
by
  intros h
  sorry

end total_feed_amount_l4_4091


namespace triangle_ABC_proof_l4_4438

variables {A B C M N K : Type} [inhabited A] [inhabited B] [inhabited C]
variables [between A B C] [within_segment A M] [within_segment M C] [on_segment A M]
variables [angle_eq (M B N) (C B M)] [right_angle (B M K)]
variables [midpoint_of_segment A C M]
variables [extension_of_segment B N K beyond N]

theorem triangle_ABC_proof (hM : midpoint_of_segment A C M)
    (hN : on_segment A M N)
    (hAngle : angle_eq (M B N) (C B M))
    (hPerp : right_angle (B M K))
    (hExt : extension_of_segment B N K beyond N) :
    BC = AK + BK :=
by
  sorry

end triangle_ABC_proof_l4_4438


namespace gamma_sequence_convergence_l4_4770

noncomputable def converging_sequence (α : ℝ) : ℕ → ℝ
| 0 := 0  -- This would be based on your initial angle configuration
| n + 1 := (π - 2 * converging_sequence n - α) / 2

theorem gamma_sequence_convergence (α : ℝ) : 
  ∃ β, (∀ ε > 0, ∃ N, ∀ n ≥ N, |converging_sequence α n - β| < ε) ∧ β = (π - α) / 3 :=
begin
  sorry
end

end gamma_sequence_convergence_l4_4770


namespace new_person_weight_l4_4943

noncomputable def weight_of_new_person (avg_increase_weight : ℕ) (n : ℕ) (replaced_person_weight : ℕ) : ℕ :=
  let total_increase := n * avg_increase_weight
  in replaced_person_weight + total_increase

theorem new_person_weight :
  weight_of_new_person 6 8 40 = 88 :=
by
  have total_increase := 8 * 6
  have W := 40 + total_increase
  show W = 88
  sorry

end new_person_weight_l4_4943


namespace prob_kong_meng_is_one_sixth_l4_4750

variable (bag : List String := ["孔", "孟", "之", "乡"])
variable (draws : List String := [])
def total_events : ℕ := 4 * 3
def favorable_events : ℕ := 2
def probability_kong_meng : ℚ := favorable_events / total_events

theorem prob_kong_meng_is_one_sixth :
  (probability_kong_meng = 1 / 6) :=
by
  sorry

end prob_kong_meng_is_one_sixth_l4_4750


namespace sum_of_solutions_l4_4121

theorem sum_of_solutions : ∀ x : ℚ, (4 * x + 6) * (3 * x - 8) = 0 → 
  (x = -3 / 2 ∨ x = 8 / 3) → 
  (-3 / 2 + 8 / 3) = 7 / 6 :=
by
  intros x h1 h2
  sorry

end sum_of_solutions_l4_4121


namespace overall_percentage_of_favor_l4_4169

theorem overall_percentage_of_favor
    (n_starting : ℕ)
    (n_experienced : ℕ)
    (perc_starting_favor : ℝ)
    (perc_experienced_favor : ℝ)
    (in_favor_from_starting : ℕ)
    (in_favor_from_experienced : ℕ)
    (total_surveyed : ℕ)
    (total_in_favor : ℕ)
    (overall_percentage : ℝ) :
    n_starting = 300 →
    n_experienced = 500 →
    perc_starting_favor = 0.40 →
    perc_experienced_favor = 0.70 →
    in_favor_from_starting = 120 →
    in_favor_from_experienced = 350 →
    total_surveyed = 800 →
    total_in_favor = 470 →
    overall_percentage = (470 / 800) * 100 →
    overall_percentage = 58.75 :=
by
  sorry

end overall_percentage_of_favor_l4_4169


namespace locks_and_keys_for_safe_l4_4688

theorem locks_and_keys_for_safe (n : ℕ) (required_members : ℕ) (total_members : ℕ) (locks : ℕ) (keys_per_lock : ℕ) :
  total_members = 9 →
  required_members = 6 →
  locks = Nat.choose total_members (required_members - 4) →
  keys_per_lock = (required_members - 2) →
  locks = 126 ∧ keys_per_lock = 4 :=
by
  intros h1 h2 h3 h4
  rw [←h1, ←h2, h3, h4]
  sorry

end locks_and_keys_for_safe_l4_4688


namespace determine_B_l4_4301

open Set

variable {α : Type*}
noncomputable def U : Set α := {2, 4, 6, 8, 10}
variable (A B : Set α)

theorem determine_B :
  (compl (A ∪ B)) = {8, 10} →
  (A ∩ (U \ B)) = {2} →
  B = {4, 6} :=
begin
  intros h1 h2,
  sorry
end

end determine_B_l4_4301


namespace total_length_of_lines_in_T_l4_4379

def T (x y : ℝ) : Prop :=
  abs (abs (abs x - 3) - 2) + abs (abs (abs y - 3) - 2) = 2

theorem total_length_of_lines_in_T : (∑ (x y : ℝ), T x y) = 64 * real.sqrt 2 := 
sorry

end total_length_of_lines_in_T_l4_4379


namespace samantha_spending_l4_4517

/-- 
The dog toys Samantha buys are "buy one get one half off" and each costs $12.00.
She buys 4 toys. Prove that her total spending is $36.00.
-/
theorem samantha_spending (cost per_toy : ℕ) (num_toys : ℕ) 
    (h1 : cost per_toy = 1200) (h2 : num_toys = 4) : 
    let half_price := cost per_toy / 2 in
    let set_cost := cost per_toy + half_price in
    let total_spending := set_cost * (num_toys / 2) in
    total_spending = 3600 :=
by
  sorry

end samantha_spending_l4_4517


namespace identical_trapezoids_form_parallelogram_l4_4100

-- Definitions of a trapezoid and a parallelogram need to be formalized in Lean
def is_trapezoid (A B C D : Type) (quadrilateral : A → B → C → D → Prop) : Prop :=
  ∃ a b c d : Type, quadrilateral a b c d ∧ -- some condition ensuring it's a trapezoid

def ident_trapezoids (t1 t2 : Type) (quadrilateral1 : t1 → Prop) (quadrilateral2 : t2 → Prop) :=
  quadrilateral1 = quadrilateral2

def can_be_combined_into_parallelogram (t1 t2 : Type) : Prop :=
  is_trapezoid t1 ∧ is_trapezoid t2 ∧ ident_trapezoids t1 t2

-- Given two identical trapezoids, we need to prove they form a parallelogram
theorem identical_trapezoids_form_parallelogram (t1 t2 : Type)
  (h1 : is_trapezoid t1)
  (h2 : is_trapezoid t2)
  (h3 : ident_trapezoids t1 t2) :
  can_be_combined_into_parallelogram t1 t2 → true := 
by 
  sorry -- proof skipped

end identical_trapezoids_form_parallelogram_l4_4100


namespace tangent_line_eq_l4_4070

-- Define the curve and its derivative
def curve (x : ℝ) : ℝ := Real.log x + x + 1
def derivative (x : ℝ) : ℝ := 1/x + 1

-- Define the target slope and find the corresponding x-coordinate
def target_slope : ℝ := 2

-- Assertion that the equation of the tangent line is y = 2x
theorem tangent_line_eq (x y : ℝ) (h₁ : x = 1) (h₂ : y = 2) :
    curve x = y → derivative x = target_slope → y = 2 * x := 
by
  sorry

end tangent_line_eq_l4_4070


namespace count_five_digit_odd_number_with_odd_digits_l4_4248

theorem count_five_digit_odd_number_with_odd_digits :
  let digits := {1, 2, 3, 4, 5}
  ∑ num in {n | n ∈ List.permutations digits 
                ∧ (num.head ∈ {1, 3, 5})
                ∧ num.to_multiset.card = 5}, 1 = 72 :=
sorry

end count_five_digit_odd_number_with_odd_digits_l4_4248


namespace number_is_correct_l4_4330

theorem number_is_correct (x : ℝ) (h : (30 / 100) * x = (25 / 100) * 40) : x = 100 / 3 :=
by
  -- Proof would go here.
  sorry

end number_is_correct_l4_4330


namespace find_d_l4_4970

theorem find_d 
  (a b c d : ℝ)
  (h : a^2 + b^2 + c^2 + 2 = d + real.sqrt (a + b + c - d + 1)) :
  d = 9 / 4 :=
sorry

end find_d_l4_4970


namespace range_of_m_l4_4300

noncomputable def problem_statement
  (x y m : ℝ) : Prop :=
  (x - 2 * y + 5 ≥ 0) ∧
  (3 - x ≥ 0) ∧
  (x + y ≥ 0) ∧
  (m > 0)

theorem range_of_m (x y m : ℝ) :
  problem_statement x y m →
  ((∀ x y, problem_statement x y m → x^2 + y^2 ≤ m^2) ↔ m ≥ 3 * Real.sqrt 2) :=
by 
  intro h
  sorry

end range_of_m_l4_4300


namespace total_posts_in_a_day_l4_4173

theorem total_posts_in_a_day (num_members : ℕ) (questions_per_hour : ℕ) (hours_per_day : ℕ)
  (answers_to_questions_ratio : ℕ) (num_members_eq : num_members = 200)
  (questions_per_hour_eq : questions_per_hour = 3) (hours_per_day_eq : hours_per_day = 24)
  (answers_to_questions_ratio_eq : answers_to_questions_ratio = 3):
  let questions_per_day_per_member := questions_per_hour * hours_per_day
  let total_questions_per_day := questions_per_day_per_member * num_members
  let answers_per_day_per_member := answers_to_questions_ratio * questions_per_day_per_member
  let total_answers_per_day := answers_per_day_per_member * num_members
  total_questions_per_day + total_answers_per_day = 57600 := by {
  rw [num_members_eq, questions_per_hour_eq, hours_per_day_eq, answers_to_questions_ratio_eq],
  sorry
}

end total_posts_in_a_day_l4_4173


namespace number_exceeds_35_percent_by_245_l4_4622

theorem number_exceeds_35_percent_by_245 : 
  ∃ (x : ℝ), (0.35 * x + 245 = x) ∧ x = 376.92 := 
by
  sorry

end number_exceeds_35_percent_by_245_l4_4622


namespace triangular_weight_is_60_l4_4535

variable (w_round w_triangular w_rectangular : ℝ)

axiom rectangular_weight : w_rectangular = 90
axiom balance1 : w_round + w_triangular = 3 * w_round
axiom balance2 : 4 * w_round + w_triangular = w_triangular + w_round + w_rectangular

theorem triangular_weight_is_60 :
  w_triangular = 60 :=
by
  sorry

end triangular_weight_is_60_l4_4535


namespace foma_should_give_ierema_55_coins_l4_4572

variables (F E Y : ℝ)

-- Conditions
def condition1 : Prop := F - 70 = E + 70
def condition2 : Prop := F - 40 = Y
def condition3 : Prop := E + 70 = Y

theorem foma_should_give_ierema_55_coins
  (h1 : condition1 F E) (h2 : condition2 F Y) (h3 : condition3 E Y) :
  F - (E + 55) = E + 55 :=
by
  sorry

end foma_should_give_ierema_55_coins_l4_4572


namespace trig_system_solution_l4_4726

theorem trig_system_solution 
  (a b c : ℝ)
  (x y : ℝ)
  (h1 : sin x + sin y = a)
  (h2 : cos x + cos y = b)
  (h3 : cot x * cot y = c) :
  (a^2 + b^2)^2 - 4a^2 = c * ((a^2 + b^2)^2 - 4b^2) := 
by
  sorry

end trig_system_solution_l4_4726


namespace count_three_digit_integers_with_repeating_digits_l4_4818

theorem count_three_digit_integers_with_repeating_digits : 
  ( { n // 100 ≤ n ∧ n < 300 ∧ (n / 10 % 10 = n % 10 ∨ n / 100 = n % 10 ∨ n / 100 = n / 10 % 10) }.to_finset.card = 56 ) :=
by sorry

end count_three_digit_integers_with_repeating_digits_l4_4818


namespace volume_difference_l4_4662

noncomputable def sphere_radius : ℝ := 7
noncomputable def cylinder_radius : ℝ := 4
noncomputable def cylinder_height : ℝ := Real.sqrt (sphere_radius ^ 2 * 4 - cylinder_radius ^ 2 * 4)

noncomputable def sphere_volume : ℝ := (4 / 3) * Real.pi * sphere_radius ^ 3
noncomputable def cylinder_volume : ℝ := Real.pi * cylinder_radius ^ 2 * cylinder_height

theorem volume_difference : sphere_volume - cylinder_volume = (1372 / 3) * Real.pi - 32 * Real.pi * Real.sqrt(33) :=
by
  sorry

end volume_difference_l4_4662


namespace balance_angles_l4_4637

variable {A B C O : Type} 
variable [inhabited O] [inhabited A] [inhabited B] [inhabited C]
variable {p1 p2 p3 : ℝ} (p1 p2 p3 : ℝ)

theorem balance_angles (h_balance : true) :
  let θ_AOB := real.cos ((p3^2 - p1^2 - p2^2) / (2 * p1 * p2)),
      θ_BOC := real.cos ((p1^2 - p2^2 - p3^2) / (2 * p2 * p3)),
      θ_COA := real.cos ((p2^2 - p3^2 - p1^2) / (2 * p3 * p1)) in
  θ_AOB = θ_AOB ∧ θ_BOC = θ_BOC ∧ θ_COA = θ_COA :=
by
  skip
  sorry

end balance_angles_l4_4637


namespace third_number_is_507_l4_4942

theorem third_number_is_507 (x : ℕ) 
  (h1 : (55 + 48 + x + 2 + 684 + 42) / 6 = 223) : 
  x = 507 := by
  sorry

end third_number_is_507_l4_4942


namespace simplify_fraction_l4_4500

theorem simplify_fraction (x : ℝ) : (x + 2) / 4 + (3 - 4 * x) / 3 = (-13 * x + 18) / 12 :=
by
  sorry

end simplify_fraction_l4_4500


namespace total_length_of_T_l4_4387

noncomputable def T : Set (ℝ × ℝ) :=
  { p : ℝ × ℝ | let (x, y) := p in abs (abs x - 3 - 2) + abs (abs y - 3 - 2) = 2 }

theorem total_length_of_T : 
  (4 * 8 * Real.sqrt 2) = 128 * Real.sqrt 2 :=
by
  sorry

end total_length_of_T_l4_4387


namespace proof_volume_l4_4208

noncomputable def volume_set (a b c h r : ℝ) : ℝ := 
  let v_box := a * b * c
  let v_extensions := 2 * (a * b * h) + 2 * (a * c * h) + 2 * (b * c * h)
  let v_cylinder := Real.pi * r^2 * h
  let v_spheres := 8 * (1/6) * (Real.pi * r^3)
  v_box + v_extensions + v_cylinder + v_spheres

theorem proof_volume : 
  let a := 2; let b := 3; let c := 6
  let r := 2; let h := 3
  volume_set a b c h r = (540 + 48 * Real.pi) / 3 ∧ (540 + 48 + 3) = 591 :=
by 
  sorry

end proof_volume_l4_4208


namespace hyperbola_equation_line_equation_tangent_l4_4764

theorem hyperbola_equation (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0)
(h_c_tangent_circle : C_tangent_to_circle C O)
(h_focus_line_tangent : focus_line_tangent (sqrt 3) O C)
: C_equation = (x^2 / 3 - y^2 = 1) :=
sorry

theorem line_equation_tangent (P : point) (h_P_circle : point_on_circle P O)
(h_l_tangent : line_tangent_to_circle l O P)
(h_l_intersects_hyperbola : line_intersects_hyperbola l C A B)
(h_area_AOB : area_triangle_OAB A B 3sqrt(2))
: l_equation = (y = -x + sqrt(6)) :=
sorry

end hyperbola_equation_line_equation_tangent_l4_4764


namespace sum_cubes_geq_sum_squares_l4_4760

open Nat

theorem sum_cubes_geq_sum_squares (n : ℕ) (a : ℕ → ℝ)
  (h₀ : a 1 ≥ 1)
  (h₁ : ∀ k, 1 ≤ k → k < n → a (k + 1) ≥ a k + 1) :
  (∑ k in range n, (a (k + 1))^3) ≥ (∑ k in range n, a (k + 1))^2 :=
sorry

end sum_cubes_geq_sum_squares_l4_4760


namespace largest_d_value_l4_4886

noncomputable def max_d (a b c d : ℝ) (h1 : a + b + c + d = 10) (h2 : ab + ac + ad + bc + bd + cd = 20) : ℝ :=
  if h : (4 * d ^ 2 - 20 * d - 80) ≤ 0 then d else 0

theorem largest_d_value (a b c d : ℝ) (h1 : a + b + c + d = 10) (h2 : ab + ac + ad + bc + bd + cd = 20) :
  max_d a b c d h1 h2 = (5 + 5 * real.sqrt 21) / 2 :=
sorry

end largest_d_value_l4_4886


namespace coefficient_of_x_in_expression_l4_4700

theorem coefficient_of_x_in_expression : 
  let expr := 2 * (x - 5) + 5 * (8 - 3 * x^2 + 6 * x) - 9 * (3 * x - 2) + 3 * (x + 4)
  ∃ k : ℤ, (expr = k * x + term) ∧ 
  (∃ coefficient_x : ℤ, coefficient_x = 8) := 
sorry

end coefficient_of_x_in_expression_l4_4700


namespace find_DZ_l4_4264

-- Define the problem setting and the specific distances given in the problem
variables (A B C D A1 B1 C1 D1 X Y Z : Type) [AddCommGroup A]
          [Inhabited A] [HasNorm A] [MetricSpace A]
          (AX AD BC BC1a BY BX AXD1 A1X BY BC1)

-- Define the distances given
def A1X : ℕ := 5
def BY : ℕ := 3
def B1C1 : ℕ := 14

-- Define the proof problem
theorem find_DZ : 
  ∃ DZ : ℕ, DZ = 20 
:=
  sorry

end find_DZ_l4_4264


namespace tangent_line_equation_ln_x_l4_4072

theorem tangent_line_equation_ln_x (y : ℝ) (x : ℝ) (h: x > 0) (slope : ℝ) (tangent_line : ℝ → ℝ) :
  (∀ x, y = log x + x + 1) →
  slope = 2 →
  tangent_line 1 = 2 →
  tangent_line = λ x, 2 * x :=
sorry

end tangent_line_equation_ln_x_l4_4072


namespace total_cost_of_purchase_l4_4909

variable (x y z : ℝ)

theorem total_cost_of_purchase (h₁ : 4 * x + (9 / 2) * y + 12 * z = 6) (h₂ : 12 * x + 6 * y + 6 * z = 8) :
  4 * x + 3 * y + 6 * z = 4 :=
sorry

end total_cost_of_purchase_l4_4909


namespace evaluate_expression_121point5_l4_4319

theorem evaluate_expression_121point5 :
  let x := (2 / 3 : ℝ)
  let y := (9 / 2 : ℝ)
  (1 / 3) * x^4 * y^5 = 121.5 :=
by
  let x := (2 / 3 : ℝ)
  let y := (9 / 2 : ℝ)
  sorry

end evaluate_expression_121point5_l4_4319


namespace part1_part2_l4_4910

-- Definitions from conditions
def U := ℝ
def A := {x : ℝ | -x^2 + 12*x - 20 > 0}
def B (a : ℝ) := {x : ℝ | 5 - a < x ∧ x < a}

-- (1) If "x ∈ A" is a necessary condition for "x ∈ B", find the range of a
theorem part1 (a : ℝ) : (∀ x : ℝ, x ∈ B a → x ∈ A) → a ≤ 3 :=
by sorry

-- (2) If A ∩ B ≠ ∅, find the range of a
theorem part2 (a : ℝ) : (∃ x : ℝ, x ∈ A ∧ x ∈ B a) → a > 5 / 2 :=
by sorry

end part1_part2_l4_4910


namespace probability_correct_l4_4207

noncomputable def probability_parallel_not_coincident : ℚ :=
  let total_points := 6
  let lines := total_points.choose 2
  let total_ways := lines * lines
  let parallel_not_coincident_pairs := 12
  parallel_not_coincident_pairs / total_ways

theorem probability_correct :
  probability_parallel_not_coincident = 4 / 75 := by
  sorry

end probability_correct_l4_4207


namespace correct_proposition_is_D_l4_4189

theorem correct_proposition_is_D :
  (¬ (∃ x_0 : ℝ, x_0^2 - 1 < 0) ↔ (∀ x : ℝ, x^2 - 1 ≥ 0)) ∧
  (∀ (Q : Type) (q : Q), (∀ P : Q → Prop, q = P q ↔ P q)) ∧  -- This is a placeholder for the quadrilateral proposition
  ((∀ x y : ℝ, x^2 = y^2 → x = y) ↔ ∀ x y : ℝ, x ≠ y → x^2 ≠ y^2) ∧ 
  (¬ (x = 3 → x^2 - 2*x - 3 = 0) ↔ (x ≠ 3 → x^2 - 2*x - 3 ≠ 0)) :=
begin
  sorry
end

end correct_proposition_is_D_l4_4189


namespace max_value_at_1_l4_4226

noncomputable def max_value (x : ℝ) (h : 0 ≤ x ∧ x ≤ 2) : ℝ :=
  x / Real.exp x

theorem max_value_at_1 : ∀ x ∈ set.Icc 0 2, max_value x ⟨_, _⟩ ≤ 1 / Real.exp 1 :=
by
  intro x hx
  sorry

end max_value_at_1_l4_4226


namespace equalize_foma_ierema_l4_4586

variables 
  (F E Y : ℕ)
  (h1 : F - 70 = E + 70)
  (h2 : F - 40 = Y)

def foma_should_give_ierema : ℕ := (F - E) / 2

theorem equalize_foma_ierema (F E Y : ℕ) (h1 : F - 70 = E + 70) (h2 : F - 40 = Y) :
  foma_should_give_ierema F E = 55 := 
by
  sorry

end equalize_foma_ierema_l4_4586


namespace exists_f_i_l4_4782

noncomputable def f_periodic (f : ℝ → ℝ) := ∀ x : ℝ, f (x + 2 * Real.pi) = f x
def even_function (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)
def pi_periodic (f : ℝ → ℝ) := ∀ x : ℝ, f (x + Real.pi) = f x

theorem exists_f_i (f : ℝ → ℝ)
  (Hf : f_periodic f) :
  ∃ (f1 f2 f3 f4 : ℝ → ℝ),
    (∀ i, i = 1 ∨ i = 2 ∨ i = 3 ∨ i = 4 → 
     even_function (λ x, if i = 1 then f1 x else if i = 2 then f2 x else if i = 3 then f3 x else f4 x)) ∧
    (∀ i, i = 1 ∨ i = 2 ∨ i = 3 ∨ i = 4 → 
     pi_periodic (λ x, if i = 1 then f1 x else if i = 2 then f2 x else if i = 3 then f3 x else f4 x)) ∧
    (∀ x : ℝ, f x = f1 x + f2 x * Real.cos x + f3 x * Real.sin x + f4 x * Real.sin (2 * x)) :=
sorry

end exists_f_i_l4_4782


namespace probability_3a_minus_1_lt_0_l4_4102

noncomputable def uniform_random_variable (a : ℝ) : Prop :=
    0 ≤ a ∧ a ≤ 1

theorem probability_3a_minus_1_lt_0 (a : ℝ) 
    (ha : uniform_random_variable a) :
    (MeasureTheory.MeasureSpace.volume {x : ℝ | 3 * x - 1 < 0 ∧ 0 ≤ x ∧ x ≤ 1} / 
     MeasureTheory.MeasureSpace.volume {x : ℝ | 0 ≤ x ∧ x ≤ 1}) = 1 / 3 :=
  sorry

end probability_3a_minus_1_lt_0_l4_4102


namespace function_characterization_l4_4223

theorem function_characterization (f : ℤ → ℤ) 
  (h : ∀ x y z : ℤ, x + y + z = 0 → f(x) + f(y) + f(z) = x * y * z) :
  ∃ c : ℤ, ∀ x : ℤ, f(x) = (x^3 - x) / 3 + c * x :=
begin
  sorry,
end

end function_characterization_l4_4223


namespace part_I_part_II_l4_4758

open Real

def f (x m n : ℝ) := abs (x - m) + abs (x + n)

theorem part_I (m n M : ℝ) (h1 : m + n = 9) (h2 : ∀ x : ℝ, f x m n ≥ M) : M ≤ 9 := 
sorry

theorem part_II (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + b^2 = 9) : (a + b) * (a^3 + b^3) ≥ 81 := 
sorry

end part_I_part_II_l4_4758


namespace angle_bisector_condition_l4_4393

noncomputable def vector_a : ℝ × ℝ × ℝ := (8, -5, -3)
noncomputable def vector_c : ℝ × ℝ × ℝ := (-3, -2, 3)
noncomputable def vector_b : ℝ × ℝ × ℝ := (3/5, -11/5, 3/5)

-- Prove that vector_b bisects the angle between vector_a and vector_c
theorem angle_bisector_condition :
  let a := vector_a
  let b := vector_b
  let c := vector_c
  (a.1 * b.1 + a.2 * b.2 + a.3 * b.3) / (real.sqrt (a.1 ^ 2 + a.2 ^ 2 + a.3 ^ 2) * real.sqrt (b.1 ^ 2 + b.2 ^ 2 + b.3 ^ 2)) =
  (b.1 * c.1 + b.2 * c.2 + b.3 * c.3) / (real.sqrt (b.1 ^ 2 + b.2 ^ 2 + b.3 ^ 2) * real.sqrt (c.1 ^ 2 + c.2 ^ 2 + c.3 ^ 2)) :=
sorry

end angle_bisector_condition_l4_4393


namespace landscape_length_is_240_l4_4521

noncomputable def length_of_landscape (b : ℝ) : ℝ :=
  8 * b

axiom playground_area : ℝ := 1200
axiom playground_fraction : ℝ := 1 / 6
axiom statue_base_area : ℝ := 5 * 5
axiom fountain_area : ℝ := Real.pi * (5^2)
axiom flower_bed_area : ℝ := 100

theorem landscape_length_is_240 :
  ∃ b : ℝ, 
  length_of_landscape b = 240 ∧ 
  playground_fraction * (length_of_landscape b * b) = playground_area ∧ 
  (length_of_landscape b * b) - playground_area - statue_base_area - fountain_area - flower_bed_area = 5875 - 25 * Real.pi :=
begin
  sorry
end

end landscape_length_is_240_l4_4521


namespace product_of_20_random_digits_ends_with_zero_l4_4667

noncomputable def probability_product_ends_in_zero : ℝ := 
  (1 - (9 / 10)^20) +
  (9 / 10)^20 * (1 - (5 / 9)^20) * (1 - (8 / 9)^19)

theorem product_of_20_random_digits_ends_with_zero : 
  abs (probability_product_ends_in_zero - 0.988) < 0.001 :=
by
  sorry

end product_of_20_random_digits_ends_with_zero_l4_4667


namespace mark_paintable_area_l4_4838

theorem mark_paintable_area :
  let num_bedrooms := 4
  let length := 14
  let width := 11
  let height := 9
  let area_excluded := 70
  let area_wall_one_bedroom := 2 * (length * height) + 2 * (width * height) - area_excluded 
  (area_wall_one_bedroom * num_bedrooms) = 1520 :=
by
  sorry

end mark_paintable_area_l4_4838


namespace quotient_of_division_l4_4009

theorem quotient_of_division 
  (dividend divisor remainder : ℕ) 
  (h_dividend : dividend = 265) 
  (h_divisor : divisor = 22) 
  (h_remainder : remainder = 1) 
  (h_div : dividend = divisor * (dividend / divisor) + remainder) : 
  (dividend / divisor) = 12 := 
by
  sorry

end quotient_of_division_l4_4009


namespace reciprocal_of_lcm_l4_4531

open Classical

noncomputable def hcf (a b : ℕ) : ℕ := gcd a b
noncomputable def lcm (a b : ℕ) : ℕ := a * b / (gcd a b)

theorem reciprocal_of_lcm (A B : ℕ) (hcf_reciprocal : ℚ) (hcf_value : ℕ) (lcm_reciprocal : ℚ) :
  hcf A B = hcf_value ∧ 
  A = 24 ∧ 
  B = 156 ∧ 
  hcf A B = 12 ∧ 
  hcf_reciprocal = 1 / 12 ∧ 
  lcm_reciprocal = 1 / (lcm A B) 
  → lcm_reciprocal = 1 / 312 := 
by 
  sorry

end reciprocal_of_lcm_l4_4531


namespace smallest_n_for_multiple_of_7_l4_4510

theorem smallest_n_for_multiple_of_7 (x y n : ℤ) (hx : x ≡ 2 [MOD 7]) (hy : y ≡ -2 [MOD 7])
  (h : x^2 + x * y + y^2 + n ≡ 0 [MOD 7]) :
  n = 3 :=
sorry

end smallest_n_for_multiple_of_7_l4_4510


namespace smallest_sum_arithmetic_geometric_sequence_l4_4467

theorem smallest_sum_arithmetic_geometric_sequence :
  ∀ (E F G H : ℕ), E > 0 → F > 0 → G > 0 → H > 0 →
  -- Conditions for arithmetic sequence
  2 * F = E + G →
  -- Conditions for geometric sequence
  F * H = G * G →
  -- Given \(\frac{G}{F} = \frac{7}{4}\)
  4 * G = 7 * F →
  E + F + G + H = 97 :=
by
  assume E F G H hE_pos hF_pos hG_pos hH_pos h_arithmetic h_geometric h_ratio
  sorry

end smallest_sum_arithmetic_geometric_sequence_l4_4467


namespace limit_arcsin_sqrt_l4_4629

theorem limit_arcsin_sqrt :
  (Real.limit (λ x : ℝ, (Real.arcsin (3 * x)) / ((Real.sqrt (2 + x)) - Real.sqrt 2)) 0 = 6 * Real.sqrt 2) :=
sorry

end limit_arcsin_sqrt_l4_4629


namespace count_two_digit_prime_sum_l4_4316

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def reverse_digits (N : ℕ) : ℕ :=
  let t := N / 10
  let u := N % 10
  10 * u + t

def prime_sum_condition (N : ℕ) : Prop :=
  is_prime (N + reverse_digits N)

def two_digit_number (N : ℕ) : Prop :=
  10 ≤ N ∧ N < 100

theorem count_two_digit_prime_sum : (finset.filter prime_sum_condition (finset.Ico 10 100)).card = 1 :=
sorry

end count_two_digit_prime_sum_l4_4316


namespace calc1_calc2_calc3_calc4_l4_4197

-- Proof problem definitions
theorem calc1 : 15 + (-22) = -7 := sorry

theorem calc2 : (-13) + (-8) = -21 := sorry

theorem calc3 : (-0.9) + 1.5 = 0.6 := sorry

theorem calc4 : (1 / 2) + (-2 / 3) = -1 / 6 := sorry

end calc1_calc2_calc3_calc4_l4_4197


namespace floor_sum_eq_2018_implies_floor_1010a_eq_1009_l4_4392

theorem floor_sum_eq_2018_implies_floor_1010a_eq_1009 {a : ℝ} (h₀ : 0 < a ∧ a < 1)
  (h₁ : (finset.sum (finset.range 2019) (λ k, ⌊a + (k+1)/2020⌋) = 2018)) :
  ⌊1010 * a⌋ = 1009 :=
sorry

end floor_sum_eq_2018_implies_floor_1010a_eq_1009_l4_4392


namespace donna_total_episodes_per_week_l4_4722

-- Defining the conditions
def episodes_per_weekday : ℕ := 8
def weekday_count : ℕ := 5
def weekend_factor : ℕ := 3
def weekend_count : ℕ := 2

-- Theorem statement
theorem donna_total_episodes_per_week :
  (episodes_per_weekday * weekday_count) + ((episodes_per_weekday * weekend_factor) * weekend_count) = 88 := 
  by sorry

end donna_total_episodes_per_week_l4_4722


namespace factorize_x_squared_minus_25_l4_4220

theorem factorize_x_squared_minus_25 : ∀ (x : ℝ), (x^2 - 25) = (x + 5) * (x - 5) :=
by
  intros x
  sorry

end factorize_x_squared_minus_25_l4_4220


namespace total_project_hours_l4_4368

def research_hours : ℕ := 10
def proposal_hours : ℕ := 2
def report_hours_left : ℕ := 8

theorem total_project_hours :
  research_hours + proposal_hours + report_hours_left = 20 := 
  sorry

end total_project_hours_l4_4368


namespace first_three_digits_right_of_decimal_l4_4103

noncomputable def a : ℝ := 10^2003 + 1

theorem first_three_digits_right_of_decimal (a : ℝ) (h : a = 10^2003 + 1) : 
  let x := (10^2003 + 1)^(11/8) in
  (x - ⌊x⌋) * 1000 = 375 :=
by
  sorry

end first_three_digits_right_of_decimal_l4_4103


namespace minimum_positive_period_of_f_is_pi_l4_4061

noncomputable def f (x : ℝ) : ℝ :=
  (Real.cos (x + Real.pi / 4))^2 - (Real.sin (x + Real.pi / 4))^2

theorem minimum_positive_period_of_f_is_pi :
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T', T' > 0 ∧ (∀ x, f (x + T') = f x) → T' ≥ T) ∧ T = Real.pi :=
sorry

end minimum_positive_period_of_f_is_pi_l4_4061


namespace total_length_of_T_l4_4384

noncomputable def T : Set (ℝ × ℝ) :=
  { p : ℝ × ℝ | let (x, y) := p in abs (abs x - 3 - 2) + abs (abs y - 3 - 2) = 2 }

theorem total_length_of_T : 
  (4 * 8 * Real.sqrt 2) = 128 * Real.sqrt 2 :=
by
  sorry

end total_length_of_T_l4_4384


namespace correct_calculation_l4_4992

theorem correct_calculation (x : ℝ) : 
(x + x = 2 * x) ∧
(x * x = x^2) ∧
(2 * x * x^2 = 2 * x^3) ∧
(x^6 / x^3 = x^3) →
(2 * x * x^2 = 2 * x^3) := 
by
  intro h
  exact h.2.2.1

end correct_calculation_l4_4992


namespace each_shopper_receives_equal_amount_l4_4859

variables (G I S total_final : ℝ)

-- Given conditions
def conditions : Prop :=
  G = 120 ∧
  I = G + 15 ∧
  I = S + 45

noncomputable def amount_each_shopper_receives : ℝ :=
  let total := I + S + G in total / 3

theorem each_shopper_receives_equal_amount (h : conditions) : amount_each_shopper_receives G I S = 115 := by
  -- Given conditions for Giselle, Isabella, and Sam
  rcases h with ⟨hG, hI1, hI2⟩
    
  -- Define total_final from conditions
  let total_final := G + (G + 15) + (G + 15 - 45)
  
  -- Default proof
  sorry

end each_shopper_receives_equal_amount_l4_4859


namespace find_a_11_l4_4270

noncomputable def sequence (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, S n + S m = S (n + m)

theorem find_a_11 :
  ∃ a S : ℕ → ℕ, sequence a S ∧ a 1 = 1 ∧ a 11 = 1 :=
sorry

end find_a_11_l4_4270


namespace foma_gives_ierema_55_l4_4555

-- Defining the amounts that Foma, Ierema, and Yuliy have initially
variables (F E Y : ℝ)

-- Given conditions
def condition_1 : Prop := F - E = 140
def condition_2 : Prop := F - Y = 40
def condition_3 : Prop := Y = E + 70

-- Prove that Foma should give Ierema 55 gold coins
theorem foma_gives_ierema_55 (h1 : condition_1) (h2 : condition_2) (h3 : condition_3) : 
  (F - E) / 2 = 55 :=
by
  sorry

end foma_gives_ierema_55_l4_4555


namespace vasya_wins_l4_4080

-- Define the initial conditions of the game
def initial_stone_piles : List Nat := [40, 40, 40]

-- Define a structure to represent the state of the game
structure GameState where
  piles : List Nat
  current_player : Bool -- true for Petya, false for Vasya

-- Define a predicate that checks if a move is possible
def can_combine_two_piles (state : GameState) : Prop :=
  ∃ p1 p2, p1 ≠ p2 ∧ p1 ∈ state.piles ∧ p2 ∈ state.piles ∧
  (∃ piles', piles'.length = (state.piles.length + 2) ∧
  state.piles.erase p1 = state.piles.erase p2 ∧
  piles' = {piles := state.piles.erase p1.erase p2 ++ [1, 1, 1, 1]})

-- Define the final state condition
def final_state_condition (state : GameState) : Prop :=
  ¬(can_combine_two_piles state)

-- The theorem stating that Vasya wins
theorem vasya_wins : ∀ (state : GameState), state.piles = initial_stone_piles → 
  (state.current_player = false → final_state_condition state) :=
by
  sorry

end vasya_wins_l4_4080


namespace integral_of_f_l4_4288

noncomputable def f : ℝ → ℝ 
| x := if x ∈ Icc (0 : ℝ) 1 then x^2  
        else if x ∈ Ioc 1 real.exp then 1 / x 
        else 0

theorem integral_of_f :
  ∫ x in 0..real.exp, f x = 4 / 3 := 
sorry

end integral_of_f_l4_4288


namespace safe_lock_problem_l4_4685

-- Definitions of the conditions
def num_people := 9
def min_people_needed := 6

-- Binomial Coefficient Function
def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Statement of the problem and correct answer
theorem safe_lock_problem :
  (binomial_coefficient num_people (num_people - min_people_needed + 1) = 126) ∧
  (∀ lock, lock ∈ Finset.range 126 → (Finset.card (Finset.powersetLen 4 (Finset.range num_people)) = 4)) :=
by
  sorry

end safe_lock_problem_l4_4685


namespace find_S_find_a_l4_4810

noncomputable def set_S : Set ℝ := {x : ℝ | log 0.5 (x + 2) > log 0.25 49}
noncomputable def set_P (a : ℝ) : Set ℝ := {x : ℝ | a + 1 < x ∧ x < 2 * a + 15}

theorem find_S :
  set_S = {x : ℝ | x < 5} :=
sorry

theorem find_a (a : ℝ) (h : set_S ⊆ set_P a) :
  -17 / 2 < a ∧ a < 4 :=
sorry

end find_S_find_a_l4_4810


namespace samantha_spending_l4_4516

/-- 
The dog toys Samantha buys are "buy one get one half off" and each costs $12.00.
She buys 4 toys. Prove that her total spending is $36.00.
-/
theorem samantha_spending (cost per_toy : ℕ) (num_toys : ℕ) 
    (h1 : cost per_toy = 1200) (h2 : num_toys = 4) : 
    let half_price := cost per_toy / 2 in
    let set_cost := cost per_toy + half_price in
    let total_spending := set_cost * (num_toys / 2) in
    total_spending = 3600 :=
by
  sorry

end samantha_spending_l4_4516


namespace greatest_multiple_of_four_l4_4037

theorem greatest_multiple_of_four (x : ℕ) (hx : x > 0) (h4 : x % 4 = 0) (hcube : x^3 < 800) : x ≤ 8 :=
by {
  sorry
}

end greatest_multiple_of_four_l4_4037


namespace equalize_foma_ierema_l4_4584

variables 
  (F E Y : ℕ)
  (h1 : F - 70 = E + 70)
  (h2 : F - 40 = Y)

def foma_should_give_ierema : ℕ := (F - E) / 2

theorem equalize_foma_ierema (F E Y : ℕ) (h1 : F - 70 = E + 70) (h2 : F - 40 = Y) :
  foma_should_give_ierema F E = 55 := 
by
  sorry

end equalize_foma_ierema_l4_4584


namespace graph_passes_through_fixed_point_l4_4514

theorem graph_passes_through_fixed_point (a : ℝ) (h_pos : 0 < a) (h_ne_one : a ≠ 1) :
  ∃ (x y : ℝ), (x = 1 ∧ y = 3) ∧ (y = a^(x-1) + 2) :=
by
  use [1, 3]
  split
  · exact and.intro rfl rfl
  · sorry

end graph_passes_through_fixed_point_l4_4514


namespace total_digits_in_book_l4_4140

open Nat

theorem total_digits_in_book (n : Nat) (h : n = 10000) : 
    let pages_1_9 := 9
    let pages_10_99 := 90 * 2
    let pages_100_999 := 900 * 3
    let pages_1000_9999 := 9000 * 4
    let page_10000 := 5
    pages_1_9 + pages_10_99 + pages_100_999 + pages_1000_9999 + page_10000 = 38894 :=
by
    sorry

end total_digits_in_book_l4_4140


namespace find_number_l4_4324

theorem find_number (y : ℝ) (h : (30 / 100) * y = (25 / 100) * 40) : y = 100 / 3 :=
by
  sorry

end find_number_l4_4324


namespace min_keystrokes_243_l4_4155

noncomputable def minStepsToReach (start end_ : ℕ) : ℕ :=
if start = end_ then 0
else if end_ % 3 = 0 then 1 + minStepsToReach start (end_ / 3)
else 1 + minStepsToReach start (end_ - 1)

theorem min_keystrokes_243 : minStepsToReach 1 243 = 5 := 
sorry

end min_keystrokes_243_l4_4155


namespace incorrect_statement_is_A_l4_4158

open List

def reading_times : List ℕ := [2, 2, 4, 4, 4, 6, 6, 6, 6, 8]

-- Definitions for the different statistics
def mode (l : List ℕ) : ℕ := modeOf l
def mean (l : List ℕ) : ℝ := ((l.sum : ℝ) / l.length)
def median (l : List ℕ) : ℝ :=
  let sorted_l := sort l in
  if sorted_l.length % 2 = 1 then
    sorted_l[(sorted_l.length / 2)] -- for odd length
  else
    ((sorted_l[(sorted_l.length / 2) - 1] + sorted_l[(sorted_l.length / 2)]) / 2 : ℝ) -- for even length

-- The theorem states that the incorrect statement is A
theorem incorrect_statement_is_A : 
  (mode reading_times ≠ 1) ∧ 
  (mean reading_times = 4.8) ∧ 
  (reading_times.length = 10) ∧ 
  (median reading_times ≠ 5) := sorry

end incorrect_statement_is_A_l4_4158


namespace cover_parallelepiped_with_squares_l4_4968

theorem cover_parallelepiped_with_squares :
  ∃ (a b c : ℕ), a = 4 ∧ b = 1 ∧ c = 1 ∧
    ∀ (V T B : ℕ), V = 4 * 4 ∧ T = 1 * 1 ∧ B = 1 * 1 ∧
      (
        -- The condition that the parallelepiped's vertical faces are covered by one square of dimension 4x4
        (4 = a * b) ∧ 
        -- The condition that the parallelepiped's top and bottom faces are each covered by one square of dimension 1x1
        (1 = c ∧ 1 = b)
      ) :=
begin
  -- Proof goes here.
  sorry
end

end cover_parallelepiped_with_squares_l4_4968


namespace number_is_fraction_l4_4326

theorem number_is_fraction (x : ℝ) : (0.30 * x = 0.25 * 40) → (x = 100 / 3) :=
begin
  intro h,
  sorry
end

end number_is_fraction_l4_4326


namespace smallest_n_reducible_fraction_l4_4746

theorem smallest_n_reducible_fraction : ∀ (n : ℕ), (∃ (k : ℕ), gcd (n - 13) (5 * n + 6) = k ∧ k > 1) ↔ n = 84 := by
  sorry

end smallest_n_reducible_fraction_l4_4746


namespace prism_volume_l4_4944

theorem prism_volume (a α β : ℝ) : 
  let H := a * Real.tan β
      AD := a * (Real.cos α)
      BC := a * (Real.cos α)
      E := a
      BE := a * (Real.sin α)
      ED := a * (Real.cos α)
      S := a^2 * (Real.cos α * Real.sin α)
  in H = a * Real.tan β →
     AD + BC = 2 * a * (Real.cos α) →
     S = (AD + BC) / 2 * BE →
     V = S * H → 
    V = (a^3 / 2) * (Real.sin (2 * α)) * (Real.tan β) := 
by
  sorry

end prism_volume_l4_4944


namespace greatest_value_2q_sub_r_l4_4525

theorem greatest_value_2q_sub_r : 
  ∃ (q r : ℕ), 965 = 22 * q + r ∧ 2 * q - r = 67 := 
by 
  sorry

end greatest_value_2q_sub_r_l4_4525


namespace hyperbola_focal_length_l4_4051

theorem hyperbola_focal_length :
  (∃ (a b c : ℝ), a^2 = 4 ∧ b^2 = 5 ∧ c^2 = a^2 + b^2 ∧ 2 * c = 6) :=
begin
  sorry
end

end hyperbola_focal_length_l4_4051


namespace carter_drum_sticks_l4_4199

def sets_per_show (used : ℕ) (tossed : ℕ) : ℕ := used + tossed

def total_sets (sets_per_show : ℕ) (num_shows : ℕ) : ℕ := sets_per_show * num_shows

theorem carter_drum_sticks :
  sets_per_show 8 10 * 45 = 810 :=
by
  sorry

end carter_drum_sticks_l4_4199


namespace cube_surface_area_remains_constant_l4_4639

-- Step d): Lean 4 statement

/-- 
Given a cube of dimensions 5 cm x 5 cm x 5 cm and 8 corner cubes each of dimensions 
2 cm x 2 cm x 2 cm are removed, prove that the surface area of the resulting figure 
is 150 cm².
-/
theorem cube_surface_area_remains_constant :
  let original_cube_surface_area := 6 * (5 * 5)
  let reduced_cube_surface_area := original_cube_surface_area - 8 * 3 * (2 * 2) + 8 * 3 * (2 * 2)
  reduced_cube_surface_area = 150 := 
by
  let original_cube_surface_area := 6 * (5 * 5)
  let reduced_cube_surface_area := original_cube_surface_area - 8 * 3 * (2 * 2) + 8 * 3 * (2 * 2)
  have : reduced_cube_surface_area = original_cube_surface_area := by
    calculate_steps sorry
  simp [original_cube_surface_area, reduced_cube_surface_area]
  exact original_cube_surface_area


end cube_surface_area_remains_constant_l4_4639


namespace angle_between_vectors_l4_4812

noncomputable def a : ℝ × ℝ := (1, real.sqrt 3)
noncomputable def b (m : ℝ) : ℝ × ℝ := (3, m)

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

noncomputable def norm (u : ℝ × ℝ) : ℝ :=
  real.sqrt (u.1 ^ 2 + u.2 ^ 2)

noncomputable def proj_length (u v : ℝ × ℝ) : ℝ :=
  dot_product u v / norm u

noncomputable def cos_theta (u v : ℝ × ℝ) : ℝ :=
  dot_product u v / (norm u * norm v)

theorem angle_between_vectors :
  ∀ (m : ℝ),
  proj_length a (b m) = -3 →
  let θ := real.arccos (cos_theta a (b m)) in
  θ = (2 * real.pi) / 3 :=
begin
  intros,
  sorry
end

end angle_between_vectors_l4_4812


namespace smallest_number_of_students_l4_4690

/--
At a school, the ratio of 10th-graders to 8th-graders is 3:2, 
and the ratio of 10th-graders to 9th-graders is 5:3. 
Prove that the smallest number of students from these grades is 34.
-/
theorem smallest_number_of_students {G8 G9 G10 : ℕ} 
  (h1 : 3 * G8 = 2 * G10) (h2 : 5 * G9 = 3 * G10) : 
  G10 + G8 + G9 = 34 :=
by
  sorry

end smallest_number_of_students_l4_4690


namespace red_jellybeans_count_l4_4647

theorem red_jellybeans_count (total_jellybeans : ℕ)
  (blue_jellybeans : ℕ)
  (purple_jellybeans : ℕ)
  (orange_jellybeans : ℕ)
  (H1 : total_jellybeans = 200)
  (H2 : blue_jellybeans = 14)
  (H3 : purple_jellybeans = 26)
  (H4 : orange_jellybeans = 40) :
  total_jellybeans - (blue_jellybeans + purple_jellybeans + orange_jellybeans) = 120 :=
by sorry

end red_jellybeans_count_l4_4647


namespace equal_number_of_experienced_fishermen_and_children_l4_4151

theorem equal_number_of_experienced_fishermen_and_children 
  (n : ℕ)
  (total_fish : ℕ)
  (children_catch : ℕ)
  (fishermen_catch : ℕ)
  (h1 : total_fish = n^2 + 5 * n + 22)
  (h2 : fishermen_catch - 10 = children_catch)
  (h3 : total_fish = n * children_catch + 11 * fishermen_catch)
  (h4 : fishermen_catch > children_catch)
  : n = 11 := 
sorry

end equal_number_of_experienced_fishermen_and_children_l4_4151


namespace calculate_expression_l4_4704

theorem calculate_expression :
  - (1 : ℝ) ^ 2023 + real.sqrt ((-2 : ℝ) ^ 2) + 27 + |real.sqrt 5 - 2| = 26 + real.sqrt 5 :=
by
  sorry

end calculate_expression_l4_4704


namespace find_AX_bisect_ACB_l4_4734

theorem find_AX_bisect_ACB (AC BX BC : ℝ) (h₁ : AC = 21) (h₂ : BX = 28) (h₃ : BC = 30) :
  ∃ (AX : ℝ), AX = 98 / 5 :=
by
  existsi 98 / 5
  sorry

end find_AX_bisect_ACB_l4_4734


namespace minimum_m_value_l4_4397

theorem minimum_m_value (f : ℝ → ℝ)
  (h1 : ∀ x y ∈ Icc (0:ℝ) 1, x ≠ y → |f x - f y| < (1/2) * |x - y|) :
  ∃ (m : ℝ), (∀ x y ∈ Icc (0:ℝ) 1, |f x - f y| < m) ∧ m = 1/4 :=
by
  sorry

end minimum_m_value_l4_4397


namespace perfect_square_iff_all_perfect_squares_l4_4499

theorem perfect_square_iff_all_perfect_squares
  (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0):
  (∃ k : ℕ, (xy + 1) * (yz + 1) * (zx + 1) = k^2) ↔
  (∃ a b c : ℕ, xy + 1 = a^2 ∧ yz + 1 = b^2 ∧ zx + 1 = c^2) := 
sorry

end perfect_square_iff_all_perfect_squares_l4_4499


namespace product_signs_l4_4278

theorem product_signs (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) : 
  ( 
    (((-a * b > 0) ∧ (a * c < 0) ∧ (b * d < 0) ∧ (c * d < 0)) ∨ 
    ((-a * b < 0) ∧ (a * c > 0) ∧ (b * d > 0) ∧ (c * d > 0))) ∨
    (((-a * b < 0) ∧ (a * c > 0) ∧ (b * d < 0) ∧ (c * d > 0)) ∨ 
    ((-a * b > 0) ∧ (a * c < 0) ∧ (b * d > 0) ∧ (c * d < 0))) 
  ) := 
sorry

end product_signs_l4_4278


namespace amount_paid_after_discount_l4_4186

def phone_initial_price : ℝ := 600
def discount_percentage : ℝ := 0.2

theorem amount_paid_after_discount : (phone_initial_price - discount_percentage * phone_initial_price) = 480 :=
by
  sorry

end amount_paid_after_discount_l4_4186


namespace triangle_relation_l4_4450

-- Define the given conditions and the hypothesis
variables {A B C M N K : Type*}
variables (triangle : A → B → C → Type*)
variables [midpoint M A C] [on_segment N A M] [extension K N B]
variables [right_angle (angle BMK)] [angle_eq (angle MBN) (angle CBM)]

-- Define segments and lengths
variables [segment BC] [segment AK] [segment BK]

-- State the theorem
theorem triangle_relation 
  (h1: midpoint M A C) 
  (h2: on_segment N A M) 
  (h3: angle_eq (angle MBN) (angle CBM))
  (h4: right_angle (angle BMK)) : 
  length BC = length AK + length BK := by
  sorry

end triangle_relation_l4_4450


namespace min_value_expression_l4_4258

theorem min_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : 2^(x-3) = (1/2)^y) : 
    (1/x) + (4/y) = 3 := 
  sorry

end min_value_expression_l4_4258


namespace triangle_proof_l4_4427

-- Definition of a point in the plane.
structure Point : Type :=
(x : ℝ)
(y : ℝ)

-- Definitions for the points involved.
variables (A B C M N K : Point)

-- Midpoint condition
def is_midpoint (M A C : Point) : Prop :=
  (M.x = (A.x + C.x) / 2) ∧ (M.y = (A.y + C.y) / 2)

-- Angular condition
noncomputable def angle (P Q R : Point) : ℝ := sorry

-- Given Conditions as Lean Definitions
def problem_conditions : Prop :=
  is_midpoint M A C ∧
  N.x = (A.x + M.x) / 2 ∧ N.y = (A.y + M.y) / 2 ∧ -- N lies on AM such that
  angle M B N = angle C B M ∧
  angle B M K = π / 2

-- Final proof statement to be proved
theorem triangle_proof (h : problem_conditions A B C M N K) : (dist B C) = (dist A K) + (dist B K) :=
sorry

end triangle_proof_l4_4427


namespace pentagon_perimeter_coeffs_l4_4656

theorem pentagon_perimeter_coeffs :
  let points := [(0,0), (1,2), (3,2), (4,0), (2,-1), (0,0)]
  let dist (p1 p2 : ℕ × ℕ) : ℝ := real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  let perimeter := dist points[0] points[1] + dist points[1] points[2] + 
                   dist points[2] points[3] + dist points[3] points[4] + dist points[4] points[5]
  (ℝ.floor (perimeter)) = 2 ∧
  (perimeter - ℝ.floor (perimeter)) / (real.sqrt 5) % 1 = 0 :=
begin
  sorry
end

end pentagon_perimeter_coeffs_l4_4656


namespace isosceles_triangle_vertex_angle_l4_4191

theorem isosceles_triangle_vertex_angle (T : Type) [triangle T]
  (isosceles : is_isosceles T) (exterior_angle : ∃ (A : T), exterior_angle A = 140)
  (height_outside : ∃ (H : T), height H is outside)
  : vertex_angle T = 100 := sorry

end isosceles_triangle_vertex_angle_l4_4191


namespace Dvaneft_percentage_bounds_l4_4651

noncomputable def percentageDvaneftShares (x y z : ℤ) (n m : ℕ) : ℚ :=
  n * 100 / (2 * (n + m))

theorem Dvaneft_percentage_bounds
  (x y z : ℤ) (n m : ℕ)
  (h1 : 4 * x * n = y * m)
  (h2 : x * n + y * m = z * (n + m))
  (h3 : 16 ≤ y - x)
  (h4 : y - x ≤ 20)
  (h5 : 42 ≤ z)
  (h6 : z ≤ 60) :
  12.5 ≤ percentageDvaneftShares x y z n m ∧ percentageDvaneftShares x y z n m ≤ 15 := by
  sorry

end Dvaneft_percentage_bounds_l4_4651


namespace fomagive_55_l4_4591

variables (F E Y : ℕ)

-- Conditions
def condition1 := F - E = 140
def condition2 := F - Y = 40
def condition3 := Y = E + 70

-- Proposition: Foma should give 55 coins to Ierema
theorem fomagive_55 (h1 : condition1 F E) (h2 : condition2 F Y) (h3 : condition3 E Y) : ∃ G, G = 55 :=
by
  have := h1,
  have := h2,
  have := h3,
  sorry

end fomagive_55_l4_4591


namespace travelers_not_same_station_l4_4087

theorem travelers_not_same_station :
  (let S := 10 in
   let T := S * S * S in
   let same_station := S in
   T - same_station = 990) :=
by
  let S := 10
  let T := S * S * S
  let same_station := S
  have h: T - same_station = 990 := sorry  -- Placeholder for the proof
  exact h

end travelers_not_same_station_l4_4087


namespace find_b_perpendicular_lines_l4_4986

variable (b : ℝ)

theorem find_b_perpendicular_lines :
  (2 * b + (-4) * 3 + 7 * (-1) = 0) → b = 19 / 2 := 
by
  intro h
  sorry

end find_b_perpendicular_lines_l4_4986


namespace tile_C_is_TileIV_l4_4088

-- Define the tiles with their respective sides
structure Tile :=
(top right bottom left : ℕ)

def TileI : Tile := { top := 1, right := 2, bottom := 5, left := 6 }
def TileII : Tile := { top := 6, right := 3, bottom := 1, left := 5 }
def TileIII : Tile := { top := 5, right := 7, bottom := 2, left := 3 }
def TileIV : Tile := { top := 3, right := 5, bottom := 7, left := 2 }

-- Define Rectangles for reasoning
inductive Rectangle
| A
| B
| C
| D

open Rectangle

-- Define the mathematical statement to prove
theorem tile_C_is_TileIV : ∃ tile, tile = TileIV :=
  sorry

end tile_C_is_TileIV_l4_4088


namespace five_n_plus_three_composite_l4_4373

theorem five_n_plus_three_composite (n x y : ℕ) 
  (h_pos : 0 < n)
  (h1 : 2 * n + 1 = x ^ 2)
  (h2 : 3 * n + 1 = y ^ 2) : 
  ∃ a b : ℕ, 1 < a ∧ 1 < b ∧ a * b = 5 * n + 3 := 
sorry

end five_n_plus_three_composite_l4_4373


namespace find_2alpha_minus_beta_l4_4251

theorem find_2alpha_minus_beta (α β : ℝ) 
  (h1 : 0 < α ∧ α < π)
  (h2 : 0 < β ∧ β < π)
  (h3 : tan (α - β) = 1 / 3)
  (h4 : tan β = 1 / 7) :
  2 * α - β = π / 4 :=
sorry

end find_2alpha_minus_beta_l4_4251


namespace solve_for_x_l4_4028

theorem solve_for_x (x : ℝ) : (2 / 7) * (1 / 8) * x = 14 ↔ x = 392 :=
by {
  sorry
}

end solve_for_x_l4_4028


namespace area_of_triangle_l4_4953

variable (a b c S : ℝ)
variable (A B C : ℝ) -- Represents angles A, B, and C

noncomputable def herons_formula_area (a b c : ℝ) : ℝ :=
  real.sqrt (1/4 * (a^2 * c^2 - ( (a^2 + c^2 - b^2) / 2 )^2))

theorem area_of_triangle 
  (h₁ : S = herons_formula_area a b c)
  (h₂ : a^2 * real.sin C = 24 * real.sin A)
  (h₃ : a * (real.sin C - real.sin B) * (c + b) = (27 - a^2) * real.sin A) :
  S = 15 * real.sqrt 7 / 4 :=
sorry

end area_of_triangle_l4_4953


namespace pipe_b_time_to_fill_tank_alone_l4_4014

theorem pipe_b_time_to_fill_tank_alone :
  ∃ x : ℝ, (1 / 6) + (1 / x) - (1 / 12) = 1 / 3 ∧ x = 4 :=
by
  use 4
  split
  sorry

end pipe_b_time_to_fill_tank_alone_l4_4014


namespace perfect_cube_prime_l4_4485

theorem perfect_cube_prime (p : ℕ) (h_prime : Nat.Prime p) (h_cube : ∃ x : ℕ, 2 * p + 1 = x^3) : 
  2 * p + 1 = 27 ∧ p = 13 :=
by
  sorry

end perfect_cube_prime_l4_4485


namespace original_triangle_area_l4_4047

theorem original_triangle_area (area_new_triangle : ℝ)
  (scaling_factor : ℝ)
  (h1 : scaling_factor = 5)
  (h2 : area_new_triangle = 125) :
  let area_original_triangle := area_new_triangle / (scaling_factor^2)
  in area_original_triangle = 5 :=
by
  sorry

end original_triangle_area_l4_4047


namespace swimming_day_is_sunday_l4_4001

-- Define the days of the week
inductive Day : Type
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday

-- Definitions for individual days
open Day

-- Define what Mahdi does on each day
structure WeekSchedule :=
  (basketball_day : Day)
  (tennis_day : Day)
  (running_days : List Day)
  (swimming_day : Day)
  (cycling_day : Day)

-- Define properties/conditions for Mahdi's schedule
def valid_schedule (schedule : WeekSchedule) : Prop :=
  schedule.basketball_day = Monday ∧
  schedule.tennis_day = Wednesday ∧
  (schedule.running_days.length = 3) ∧
  (∃ d1 d2, d1 ≠ d2 ∧ (d2 = d1 + 1) ∧ d1 ∈ schedule.running_days ∧ d2 ∈ schedule.running_days) ∧
  (∀ d, d ∈ schedule.running_days → d ≠ Monday ∧ d ≠ Wednesday) ∧
  (schedule.cycling_day ≠ schedule.swimming_day + 1) ∧
  (∀ d, d ∈ schedule.running_days → schedule.cycling_day ≠ d + 1 ∧ schedule.cycling_day ≠ d - 1)

-- Prove that the swimming day is Sunday given the valid schedule
theorem swimming_day_is_sunday (schedule : WeekSchedule) (h : valid_schedule schedule) :
  schedule.swimming_day = Sunday :=
by {
  sorry
}

end swimming_day_is_sunday_l4_4001


namespace locus_is_plane_l4_4225

-- Definitions for skew lines and points on those lines
variable {P : Type} [EuclideanSpace P]

def line (P : Type) [EuclideanSpace P] := AffineSubspace ℝ P
def skew_lines (a b : line P) : Prop := ¬ ∃ (P : P), P ∈ a ∧ P ∈ b

-- Locus of midpoints of segments with endpoints on two given skew lines
def loci_midpoints (a b : line P) : AffineSubspace ℝ P :=
  { x | ∃ (M ∈ a) (N ∈ b), x = (M +ᵥ N) / 2 }

theorem locus_is_plane (a b : line P) (h : skew_lines a b) :
  ∃ α : AffineSubspace ℝ P, (affine_span ℝ (loci_midpoints a b)) = α ∧
  affine_subspace.is_plane α :=
sorry

end locus_is_plane_l4_4225


namespace no_positive_integer_solutions_exists_k0_l4_4021

theorem no_positive_integer_solutions_exists_k0 
  (k : ℕ) (a b n : ℕ) 
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_pos_n : 0 < n)
  (h_ge_k0 : k ≥ 4028) :
  ¬(a^{2 * n} + b^{4 * n} + 2013 = k * a^n * b^{2 * n}) :=
sorry

end no_positive_integer_solutions_exists_k0_l4_4021


namespace number_of_isosceles_triangles_is_four_l4_4918

structure Point where
  x : ℝ
  y : ℝ

def distance (p1 p2 : Point) : ℝ :=
  real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

def is_isosceles (a b c : Point) : Prop :=
  let dab := distance a b
  let dbc := distance b c
  let dac := distance a c
  dab = dbc ∨ dbc = dac ∨ dac = dab

def triangle1 : (Point × Point × Point) := (⟨1, 4⟩, ⟨3, 4⟩, ⟨2, 2⟩)
def triangle2 : (Point × Point × Point) := (⟨4, 3⟩, ⟨4, 5⟩, ⟨6, 3⟩)
def triangle3 : (Point × Point × Point) := (⟨0, 1⟩, ⟨2, 2⟩, ⟨4, 1⟩)
def triangle4 : (Point × Point × Point) := (⟨5, 1⟩, ⟨6, 3⟩, ⟨7, 0⟩)

def isosceles_triangles : Nat :=
  [triangle1, triangle2, triangle3, triangle4].count (λ ⟨a, b, c⟩ => is_isosceles a b c)

theorem number_of_isosceles_triangles_is_four :
  isosceles_triangles = 4 :=
by
  sorry

end number_of_isosceles_triangles_is_four_l4_4918


namespace num_adult_tickets_l4_4171

variables (A C : ℕ)

def num_tickets (A C : ℕ) : Prop := A + C = 900
def total_revenue (A C : ℕ) : Prop := 7 * A + 4 * C = 5100

theorem num_adult_tickets : ∃ A, ∃ C, num_tickets A C ∧ total_revenue A C ∧ A = 500 := 
by
  sorry

end num_adult_tickets_l4_4171


namespace f_periodic_f_28_eq_l4_4297

def f1 (x : ℚ) : ℚ := (2 * x - 1) / (x + 1)

def f (n : ℕ) (x : ℚ) : ℚ :=
  Nat.recOn n x (λ _ fn, f1 fn)

theorem f_periodic {x : ℚ} : f 35 x = f 5 x :=
  sorry

theorem f_28_eq {x : ℚ} : f 28 x = (1 / (1 - x)) :=
  sorry

end f_periodic_f_28_eq_l4_4297


namespace tangent_line_condition_l4_4852

-- statement only, no proof required
theorem tangent_line_condition {m n u v x y : ℝ}
  (hm : m > 1)
  (curve_eq : x^m + y^m = 1)
  (line_eq : u * x + v * y = 1)
  (u_v_condition : u^n + v^n = 1)
  (mn_condition : 1/m + 1/n = 1)
  : (u * x + v * y = 1) ↔ (u^n + v^n = 1 ∧ 1/m + 1/n = 1) :=
sorry

end tangent_line_condition_l4_4852


namespace exists_positive_integer_n_l4_4036

theorem exists_positive_integer_n (M : Set ℝ) (hM_card : M.toFinset.card = 2003)
  (hM_rational : ∀ (a b c : ℝ), a ≠ b → b ≠ c → c ≠ a → a ∈ M → b ∈ M → c ∈ M → a^2 + b * c ∈ ℚ) :
  ∃ n : ℕ, 0 < n ∧ ∀ a : ℝ, a ∈ M → a * Real.sqrt n ∈ ℚ := 
by
  sorry

end exists_positive_integer_n_l4_4036


namespace noelle_speed_A_to_B_l4_4916

noncomputable def speed_from_A_to_B (speed_B_to_A : ℝ) (average_speed : ℝ) : ℝ :=
  let v := 5 in
  have h1 : speed_B_to_A = 20 := rfl,
  have h2 : average_speed = 8 := rfl,
  v

theorem noelle_speed_A_to_B : 
  ∀ (d : ℝ), 
  (∀ speed_B_to_A average_speed, speed_B_to_A = 20 → average_speed = 8 → speed_from_A_to_B speed_B_to_A average_speed = 5) :=
by
  intros d speed_B_to_A average_speed h_speed_B_to_A h_average_speed
  rw h_speed_B_to_A
  rw h_average_speed
  exact rfl

end noelle_speed_A_to_B_l4_4916


namespace solution_sum_of_eq_zero_l4_4117

open Real

theorem solution_sum_of_eq_zero : 
  let f (x : ℝ) := (4*x + 6) * (3*x - 8)
  in (∀ x, f x = 0 → x = -3/2 ∨ x = 8/3) → 
     (-3/2 + 8/3 = 7/6) :=
by 
  let f (x : ℝ) := (4*x + 6) * (3*x - 8)
  intro h
  have h₁ : f(-3/2) = 0 := by sorry
  have h₂ : f(8/3) = 0 := by sorry
  have sum_of_solutions : -3/2 + 8/3 = 7/6 := by 
    sorry
  exact sum_of_solutions

end solution_sum_of_eq_zero_l4_4117


namespace total_length_of_lines_in_T_l4_4377

def T (x y : ℝ) : Prop :=
  abs (abs (abs x - 3) - 2) + abs (abs (abs y - 3) - 2) = 2

theorem total_length_of_lines_in_T : (∑ (x y : ℝ), T x y) = 64 * real.sqrt 2 := 
sorry

end total_length_of_lines_in_T_l4_4377


namespace solve_arithmetic_sequence_l4_4505

theorem solve_arithmetic_sequence :
  ∃ x > 0, (x * x = (4 + 25) / 2) :=
by
  sorry

end solve_arithmetic_sequence_l4_4505


namespace gcd_of_720_120_168_is_24_l4_4056

theorem gcd_of_720_120_168_is_24 : Int.gcd (Int.gcd 720 120) 168 = 24 := 
by sorry

end gcd_of_720_120_168_is_24_l4_4056


namespace find_vector_c_l4_4256

-- Definitions of the given vectors
def vector_a : ℝ × ℝ := (3, -1)
def vector_b : ℝ × ℝ := (-1, 2)
def vector_c : ℝ × ℝ := (2 * vector_a.1 + vector_b.1, 2 * vector_a.2 + vector_b.2)

-- The goal is to prove that vector_c = (5, 0)
theorem find_vector_c : vector_c = (5, 0) :=
by
  -- proof steps would go here
  sorry

end find_vector_c_l4_4256


namespace greatest_sum_possible_l4_4664

theorem greatest_sum_possible (a b c d e x y z w v : ℕ) (h_sums : 
  multiset.card ({210, 350, 300, 250, 400, x, y, z, w, v} : multiset ℕ) = 10) :
  x + y + z + w + v ≤ 1510 := 
sorry

end greatest_sum_possible_l4_4664


namespace smallest_x_multiple_of_53_l4_4112

theorem smallest_x_multiple_of_53 :
  ∃ x : ℕ, (3 * x + 41) % 53 = 0 ∧ x > 0 ∧ x = 4 :=
by 
  sorry

end smallest_x_multiple_of_53_l4_4112


namespace complex_power_sum_l4_4219

noncomputable theory

open Complex

theorem complex_power_sum :
  let i := Complex.I in
  i^8 + i^{20} + i^{-32} + 2 * i = 3 + 2 * i :=
by
  sorry

end complex_power_sum_l4_4219


namespace foma_should_give_ierema_l4_4596

variables (F E Y : ℕ)

-- Given conditions in Lean definitions
def condition1 := F - 70 = E + 70
def condition2 := F - 40 = Y

-- Final statement to be proven
theorem foma_should_give_ierema : 
  condition1 → condition2 → F - E = 110 → F - E = 110 / 2 + 55 :=
by
  intros h1 h2 h3
  sorry

end foma_should_give_ierema_l4_4596


namespace find_m_b_sum_does_not_prove_l4_4958

theorem find_m_b_sum_does_not_prove :
  ∃ m b : ℝ, 
  let original_point := (2, 3)
  let image_point := (10, 9)
  let midpoint := ((original_point.1 + image_point.1) / 2, (original_point.2 + image_point.2) / 2)
  m = -4 / 3 ∧ 
  midpoint = (6, 6) ∧ 
  6 = m * 6 + b 
  ∧ m + b = 38 / 3 := sorry

end find_m_b_sum_does_not_prove_l4_4958


namespace complex_number_power_identity_l4_4321

noncomputable def complex_number_condition (z : ℂ) : Prop := z + z⁻¹ = Real.sqrt 2

theorem complex_number_power_identity (z : ℂ) (hc : complex_number_condition z) : z^12 + z^(-12) = -2 := 
sorry

end complex_number_power_identity_l4_4321


namespace sum_coordinates_X_l4_4883

variables (X Y Z : ℝ × ℝ)
variables (a b c : ℝ)

-- Defining the conditions
def condition1 : Prop := ∃ (q : ℝ), a = 1 + q ∧ b = 9 - 6 * q ∧ q = 1/3
def condition2 : Prop := ∃ (r : ℝ), c = 1 + r ∧ b = 9 + 18 * r ∧ r = 2/3

theorem sum_coordinates_X (X Y Z : ℝ × ℝ) 
  (h1 : condition1) 
  (h2 : condition2) 
  (hy : Y = (1, 9)) 
  (hz : Z = (-1, 3)) : X.1 + X.2 = 34 :=
sorry

end sum_coordinates_X_l4_4883


namespace necessary_and_sufficient_condition_l4_4286

theorem necessary_and_sufficient_condition (t : ℝ) (S : ℕ → ℝ) (a : ℕ → ℝ) :
    (∀ n, S n = n^2 + 5*n + t) →
    (t = 0 ↔ (∀ n, a n = 2*n + 4 ∧ (n > 0 → a n = S n - S (n - 1)))) :=
by
  sorry

end necessary_and_sufficient_condition_l4_4286


namespace find_total_students_l4_4549

-- Definitions
def total_students (T : ℝ) : Prop := 0.88 * T = 44

-- The statement to be proven
theorem find_total_students : ∃ (T : ℝ), total_students T ∧ T = 50 :=
by
  use 50
  split
  · unfold total_students
    norm_num
  · norm_num
  sorry

end find_total_students_l4_4549


namespace factorization_of_w4_minus_81_l4_4733

theorem factorization_of_w4_minus_81 (w : ℝ) : 
  (w^4 - 81) = (w - 3) * (w + 3) * (w^2 + 9) :=
by sorry

end factorization_of_w4_minus_81_l4_4733


namespace general_term_of_sequence_l4_4356

noncomputable def a : ℕ → ℝ
| 0       := 1
| (n + 1) := 2 * a n + 3

theorem general_term_of_sequence (n : ℕ) : a n = 2^(n + 1) - 3 :=
sorry

end general_term_of_sequence_l4_4356


namespace right_triangle_leg_lengths_l4_4104

theorem right_triangle_leg_lengths 
  (α β : ℝ) -- α and β are the angles
  (a b : ℝ) -- a and b are the lengths of the legs
  (a_bisector b_bisector : ℝ) -- lengths of the angle bisectors of the acute angles
  (h1 : a_bisector = 1)
  (h2 : b_bisector = 2)
  (h3 : α + β = Real.pi / 4) -- since α + β = 45 degrees = π/4
  (h4 : b / (Real.sin (2 * β)) = 2 * Real.sin (α) / Real.sin (2 * β)) -- given by the angle bisector properties
  (h5 : 2 * Real.sin(2 * β) * Real.cos(β) = Real.sin((Real.pi / 4) - β) * (1 + Real.sin(2 * β))) -- simplified equation from solution
  : (a ≈ 0.8341) ∧ (b ≈ 1.9596) :=
begin
  sorry
end

end right_triangle_leg_lengths_l4_4104


namespace each_shopper_receives_equal_amount_l4_4860

variables (G I S total_final : ℝ)

-- Given conditions
def conditions : Prop :=
  G = 120 ∧
  I = G + 15 ∧
  I = S + 45

noncomputable def amount_each_shopper_receives : ℝ :=
  let total := I + S + G in total / 3

theorem each_shopper_receives_equal_amount (h : conditions) : amount_each_shopper_receives G I S = 115 := by
  -- Given conditions for Giselle, Isabella, and Sam
  rcases h with ⟨hG, hI1, hI2⟩
    
  -- Define total_final from conditions
  let total_final := G + (G + 15) + (G + 15 - 45)
  
  -- Default proof
  sorry

end each_shopper_receives_equal_amount_l4_4860


namespace percent_non_union_women_l4_4840

variables (E : ℝ) (percentage_unionized : ℝ) (percentage_union_men : ℝ) (percentage_nonunion_women : ℝ)

-- Given conditions
def condition1 : percentage_unionized = 0.60 := sorry
def condition2 : percentage_union_men = 0.70 := sorry
def condition3 : percentage_nonunion_women = 0.85 := sorry

-- Statement to prove
theorem percent_non_union_women (E : ℝ) 
  (h1 : condition1) 
  (h2 : condition2) 
  (h3 : condition3): (percentage_nonunion_women = 0.85) :=
begin
  sorry
end

end percent_non_union_women_l4_4840


namespace smallest_x_palindrome_l4_4113

def is_palindrome (n : ℕ) : Prop :=
  let s := n.repr in s = s.reverse

noncomputable def smallest_palindrome_x : ℕ :=
  77 -- directly from the solution

theorem smallest_x_palindrome : smallest_palindrome_x + 8921 = 8998 ∧ is_palindrome (smallest_palindrome_x + 8921) :=
by 
  -- Given in the problem, the palindrome condition is explicitly checked
  have h1 : is_palindrome 8998 := by sorry,
  have h2 : smallest_palindrome_x + 8921 = 8998 := by sorry,
  exact ⟨h2, h1⟩

end smallest_x_palindrome_l4_4113


namespace intersection_of_line_with_x_axis_l4_4053

theorem intersection_of_line_with_x_axis 
  (k : ℝ) 
  (h : ∀ x y : ℝ, y = k * x + 4 → (x = -1 ∧ y = 2)) 
  : ∃ x : ℝ, (2 : ℝ) * x + 4 = 0 ∧ x = -2 :=
by {
  sorry
}

end intersection_of_line_with_x_axis_l4_4053


namespace part1_part2_l4_4848

def parametric_eq_circle (phi : ℝ) : ℝ × ℝ :=
  (1 + Real.cos phi, Real.sin phi)

def polar_eq_circle (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 1

theorem part1 (phi : ℝ) : polar_eq_circle (1 + Real.cos phi) (Real.sin phi) :=
  sorry

def rho (theta : ℝ) : ℝ := 2 * Real.cos theta

def intersect1 (theta : ℝ) : ℝ :=
  rho theta

def intersect2 (theta : ℝ) : ℝ :=
  3 * Real.sqrt 3 / (Real.sin theta + Real.cos theta)

theorem part2 : |intersect1 (Real.pi / 3) - intersect2 (Real.pi / 3)| =
  abs (rho (Real.pi / 3) - (3 * Real.sqrt 3 / (Real.sin (Real.pi / 3) + Real.cos (Real.pi / 3)))) := sorry

end part1_part2_l4_4848


namespace find_x_l4_4604

-- Define the isosceles triangle properties and conditions
def is_isosceles_triangle (A B C : Type) (x : ℝ) :=
  ∃ α β γ : ℝ, α + β + γ = 180 ∧ (α = β ∨ β = γ ∨ γ = α) ∧ β = x

-- Define the possible measures of ∠BAC in each case
def angle_BAC_case1 (x : ℝ) : ℝ :=
  180 - 2 * x

def angle_BAC_case2 (x : ℝ) : ℝ :=
  x

def angle_BAC_case3 (x : ℝ) : ℝ :=
  (180 - x) / 2

-- Define the sum of possible measures
def sum_of_angles (x : ℝ) : ℝ :=
  angle_BAC_case1 x + angle_BAC_case2 x + angle_BAC_case3 x

theorem find_x (x : ℝ) : is_isosceles_triangle A B C x → sum_of_angles x = 240 → x = 20 :=
by
  sorry

end find_x_l4_4604


namespace circle_equation_l4_4642

-- Define the conditions in terms of Lean 4 variables and equations
variable {t : ℝ} -- t is a real number parameter for the center of the circle

-- Condition 1: Center lies on the line x - 2y = 0
def center_on_line (x y : ℝ) : Prop :=
  x - 2 * y = 0

-- Condition 2: Circle is tangent to the positive half of the y-axis
def tangent_to_yaxis (x y : ℝ) (r : ℝ) : Prop :=
  x = r ∧ r > 0

-- Condition 3: Length of chord cut from x-axis is 2sqrt(3)
def chord_length_eq (r : ℝ) : Prop :=
  (r = 2) ∧ (r = |2t|) ∧ (4 * t^2 = (sqrt(3))^2 + t^2)

-- The goal is to prove the standard equation of the circle
theorem circle_equation 
  (x y r : ℝ)
  (h1 : center_on_line x y) 
  (h2 : tangent_to_yaxis x y r) 
  (h3 : chord_length_eq r) : 
  (x = 2 ∧ y = 1 ∧ r = 2) → 
  (∀ (p q : ℝ), (p - 2)^2 + (q - 1)^2 = 4) :=
sorry

end circle_equation_l4_4642


namespace chord_length_reciprocal_sum_const_find_line_eq_l4_4259

-- Define the given conditions
def circle_center : ℝ × ℝ := (-1, 0)
def circle_radius : ℝ := 2
def tangent_point : ℚ × ℚ := (3/5, 6/5)
def tangent_line (x y : ℝ) : Prop := 4 * x + 3 * y - 6 = 0
def chord_line (x y : ℝ) : Prop := 12 * x - 5 * y - 1 = 0
def point_N : ℝ × ℝ := (2, 1)
def positive_slope (k : ℝ) : Prop := k > 0

-- Questions to prove
theorem chord_length :
  dist_circle_to_line circle_center circle_radius chord_line = 2 * sqrt 3 :=
sorry

theorem reciprocal_sum_const {x1 x2 : ℝ} (hx1 hx2 : real.is_root (λ x, (1 + k^2) * x^2 + 2 * x - 3)) :
  1 / x1 + 1 / x2 = 2 / 3 :=
sorry
  
theorem find_line_eq : 
  let k := 1 in
  line_through_origin_with_slope k = (λ x y, y = x) :=
sorry

end chord_length_reciprocal_sum_const_find_line_eq_l4_4259


namespace ancient_chinese_silver_problem_l4_4354

theorem ancient_chinese_silver_problem :
  ∃ (x y : ℤ), 7 * x = y - 4 ∧ 9 * x = y + 8 :=
by
  sorry

end ancient_chinese_silver_problem_l4_4354


namespace distinct_dragons_count_l4_4365

theorem distinct_dragons_count : 
  {n : ℕ // n = 7} :=
sorry

end distinct_dragons_count_l4_4365


namespace foma_should_give_ierema_55_coins_l4_4564

theorem foma_should_give_ierema_55_coins (F E Y : ℤ)
  (h1 : F - 70 = E + 70)
  (h2 : F - 40 = Y)
  (h3 : E + 70 = Y) :
  F - E = 110 → F - E = 55 :=
by
  sorry

end foma_should_give_ierema_55_coins_l4_4564


namespace amount_paid_after_discount_l4_4185

def phone_initial_price : ℝ := 600
def discount_percentage : ℝ := 0.2

theorem amount_paid_after_discount : (phone_initial_price - discount_percentage * phone_initial_price) = 480 :=
by
  sorry

end amount_paid_after_discount_l4_4185


namespace students_present_l4_4544

theorem students_present (total_students : ℕ) (absent_percentage : ℕ) (h1 : total_students = 100) (h2 : absent_percentage = 14) :
  total_students * (100 - absent_percentage) / 100 = 86 :=
by
  rw [h1, h2]
  norm_num
  sorry

end students_present_l4_4544


namespace triangle_proof_l4_4431

-- Definition of a point in the plane.
structure Point : Type :=
(x : ℝ)
(y : ℝ)

-- Definitions for the points involved.
variables (A B C M N K : Point)

-- Midpoint condition
def is_midpoint (M A C : Point) : Prop :=
  (M.x = (A.x + C.x) / 2) ∧ (M.y = (A.y + C.y) / 2)

-- Angular condition
noncomputable def angle (P Q R : Point) : ℝ := sorry

-- Given Conditions as Lean Definitions
def problem_conditions : Prop :=
  is_midpoint M A C ∧
  N.x = (A.x + M.x) / 2 ∧ N.y = (A.y + M.y) / 2 ∧ -- N lies on AM such that
  angle M B N = angle C B M ∧
  angle B M K = π / 2

-- Final proof statement to be proved
theorem triangle_proof (h : problem_conditions A B C M N K) : (dist B C) = (dist A K) + (dist B K) :=
sorry

end triangle_proof_l4_4431


namespace abc_value_l4_4825

theorem abc_value (a b c : ℝ) (h1 : ab = 30 * (4^(1/3))) (h2 : ac = 40 * (4^(1/3))) (h3 : bc = 24 * (4^(1/3))) :
  a * b * c = 120 :=
sorry

end abc_value_l4_4825


namespace intervals_of_monotonicity_find_a_for_max_value_l4_4799

noncomputable def f (x a : ℝ) : ℝ := 2 * x^3 - 3 * a * x^2 + 3 * a - 2
noncomputable def f' (x a : ℝ) : ℝ := 6 * x * (x - a)

-- Statement for part (1)
theorem intervals_of_monotonicity (x : ℝ) : 
  let a := 1 in
  (f' x a < 0 ↔ 0 < x ∧ x < 1) ∧ 
  (f' x a > 0 ↔ (x < 0 ∨ x > 1)) := by
  intro x
  let a := 1
  sorry

-- Statement for part (2)
theorem find_a_for_max_value (a : ℝ) (hmax : ∃ x, f x a = 0) :
  a = -2 ∨ a = 2 / 3 := by
  intro a hmax
  sorry

end intervals_of_monotonicity_find_a_for_max_value_l4_4799


namespace angle_C_of_acute_triangle_l4_4282

theorem angle_C_of_acute_triangle (A B C : ℝ) 
  (h_area : (1/2) * 4 * 3 * sin C = 3 * sqrt 3)
  (h0 : 0 < C)
  (h90 : C < 90) :
  C = 60 := 
sorry

end angle_C_of_acute_triangle_l4_4282


namespace probability_X_between_neg1_and_5_probability_X_le_8_probability_X_ge_5_probability_X_between_neg3_and_9_l4_4744

noncomputable def normalCDF (z : ℝ) : ℝ :=
  sorry -- Assuming some CDF function for the sake of the example.

variable (X : ℝ → ℝ)
variable (μ : ℝ := 3)
variable (σ : ℝ := sqrt 4)

-- 1. Proof that P(-1 < X < 5) = 0.8185
theorem probability_X_between_neg1_and_5 : 
  ((-1 < X) ∧ (X < 5) → (normalCDF 1 - normalCDF (-2)) = 0.8185) :=
  sorry

-- 2. Proof that P(X ≤ 8) = 0.9938
theorem probability_X_le_8 : 
  (X ≤ 8 → normalCDF 2.5 = 0.9938) :=
  sorry

-- 3. Proof that P(X ≥ 5) = 0.1587
theorem probability_X_ge_5 : 
  (X ≥ 5 → (1 - normalCDF 1) = 0.1587) :=
  sorry

-- 4. Proof that P(-3 < X < 9) = 0.9972
theorem probability_X_between_neg3_and_9 : 
  ((-3 < X) ∧ (X < 9) → (2 * normalCDF 3 - 1) = 0.9972) :=
  sorry

end probability_X_between_neg1_and_5_probability_X_le_8_probability_X_ge_5_probability_X_between_neg3_and_9_l4_4744


namespace arithmetic_geometric_mean_l4_4925

theorem arithmetic_geometric_mean (x y : ℝ) (h1 : (x + y) / 2 = 18) (h2 : (xy = 92) : x^2 + y^2 = 1112 :=
by 
  sorry

end arithmetic_geometric_mean_l4_4925


namespace correct_probability_statement_l4_4998

theorem correct_probability_statement (Ω : Type) [probability_space Ω] (A : set Ω) :
  (0 : ℝ) < P(A) ∧ P(A) < 1 := sorry

end correct_probability_statement_l4_4998


namespace max_number_ahn_can_get_l4_4675

theorem max_number_ahn_can_get :
  ∃ n : ℤ, (10 ≤ n ∧ n ≤ 99) ∧ ∀ m : ℤ, (10 ≤ m ∧ m ≤ 99) → (3 * (300 - n) ≥ 3 * (300 - m)) ∧ 3 * (300 - n) = 870 :=
by sorry

end max_number_ahn_can_get_l4_4675


namespace arithmeticSeq_100th_and_sum_secondOrderSeq_6th_secondOrderSeq_2013th_l4_4795

-- Define the arithmetic sequence
def arithmeticSeq (n : Nat) : Nat := 2 * n

-- Prove the 100th term and sum of the first 100 terms
theorem arithmeticSeq_100th_and_sum :
  arithmeticSeq 100 = 200 ∧ (List.range 100).map arithmeticSeq |>.sum = 10100 :=
by
  sorry

-- Define the second-order arithmetic sequence difference
def secondOrderDiff (n : Nat) : Nat := 2 * (n - 1)

-- Define the second-order arithmetic sequence
def secondOrderSeq : Nat → Nat
| 0     => 1
| n + 1 => secondOrderSeq n + secondOrderDiff (n + 1)

-- Prove the 6th term of the second-order arithmetic sequence
theorem secondOrderSeq_6th : secondOrderSeq 5 = 31 :=
by
  sorry

-- Define the general formula for the second-order arithmetic sequence
def secondOrderSeqFormula (n : Nat) : Nat := 1 + n * (n - 1)

-- Prove the 2013th term of the second-order arithmetic sequence
theorem secondOrderSeq_2013th : secondOrderSeqFormula 2013 = 4050157 :=
by
  sorry

end arithmeticSeq_100th_and_sum_secondOrderSeq_6th_secondOrderSeq_2013th_l4_4795


namespace count_mod_13_eq_3_l4_4312

theorem count_mod_13_eq_3 :
  { x : ℕ | x < 2000 ∧ x % 13 = 3 }.card = 154 :=
by { sorry }

end count_mod_13_eq_3_l4_4312


namespace triangle_equality_lemma_l4_4406

open EuclideanGeometry

theorem triangle_equality_lemma 
  (A B C M N K: Point)
  (hM_midpoint: midpoint M A C)
  (hN_on_AM: lies_on N A M)
  (h_angle_eq: ∠ M B N = ∠ C B M)
  (hK_right: ∠ B M K = 90) :
  dist B C = dist A K + dist B K :=
by
  sorry

end triangle_equality_lemma_l4_4406


namespace triangle_ABC_proof_l4_4437

variables {A B C M N K : Type} [inhabited A] [inhabited B] [inhabited C]
variables [between A B C] [within_segment A M] [within_segment M C] [on_segment A M]
variables [angle_eq (M B N) (C B M)] [right_angle (B M K)]
variables [midpoint_of_segment A C M]
variables [extension_of_segment B N K beyond N]

theorem triangle_ABC_proof (hM : midpoint_of_segment A C M)
    (hN : on_segment A M N)
    (hAngle : angle_eq (M B N) (C B M))
    (hPerp : right_angle (B M K))
    (hExt : extension_of_segment B N K beyond N) :
    BC = AK + BK :=
by
  sorry

end triangle_ABC_proof_l4_4437


namespace suff_but_not_necc_condition_l4_4824

theorem suff_but_not_necc_condition
  (a b c : ℂ) : (a^2 + b^2 > c^2) → (a^2 + b^2 - c^2 > 0) ∧ (¬ (a^2 + b^2 - c^2 > 0) → ¬ (a^2 + b^2 > c^2)) :=
  by
  sorry

end suff_but_not_necc_condition_l4_4824


namespace real_numbers_division_l4_4999

def is_non_neg (x : ℝ) : Prop := x ≥ 0

theorem real_numbers_division :
  ∀ x : ℝ, x < 0 ∨ is_non_neg x :=
by
  intro x
  by_cases h : x < 0
  · left
    exact h
  · right
    push_neg at h
    exact h

end real_numbers_division_l4_4999


namespace correct_statement_C_l4_4997

-- Definitions based on the problem's conditions
def in_the_same_plane (l1 l2 : ℝ^2) : Prop := ∃ plane, l1 ⊆ plane ∧ l2 ⊆ plane
def do_not_intersect (l1 l2 : ℝ^2) : Prop := ∀ x, x ∉ l1 ∨ x ∉ l2
def no_common_point (l1 l2 : ℝ^2) : Prop := ∀ x, x ∉ l1 ∨ x ∉ l2
def parallel_lines (l1 l2 : ℝ^2) : Prop := in_the_same_plane l1 l2 ∧ do_not_intersect l1 l2

-- Statement C to be proven
theorem correct_statement_C (l1 l2 : ℝ^2) (h1 : in_the_same_plane l1 l2) (h2 : no_common_point l1 l2) : parallel_lines l1 l2 :=
by sorry

end correct_statement_C_l4_4997


namespace inequality_positive_numbers_l4_4461

theorem inequality_positive_numbers (n : ℕ) (n_pos : 1 ≤ n)
  (x : Fin n → ℝ) (x_pos : ∀ i, 0 < x i) :
  (∑ i in Finset.finRange n, (x i)^2 / (x ((i + 1) % n))) ≥ ∑ i in Finset.finRange n, x i :=
by
  sorry

end inequality_positive_numbers_l4_4461


namespace sqrt_37_range_l4_4218

theorem sqrt_37_range : 6 < Real.sqrt 37 ∧ Real.sqrt 37 < 7 :=
by
  sorry

end sqrt_37_range_l4_4218


namespace min_digits_fraction_l4_4988

theorem min_digits_fraction :
  ∀ (n d : ℕ),
    n = 987654321 →
    d = 2^24 * 5^6 →
    ∃ k : ℕ, k = 24 ∧ (∃ decimal_repr : ℚ, decimal_repr = (n : ℚ) / (d : ℚ) ∧ has_decimal_places decimal_repr k) :=
by
  intros n d h_n h_d
  use 24
  split
  { refl }
  { use ((987654321 : ℚ) / (2^24 * 5^6 : ℚ))
    split
    { rw [h_n, h_d] }
    { sorry } }

end min_digits_fraction_l4_4988


namespace math_solution_l4_4460

noncomputable def math_problem (x y z : ℝ) : Prop :=
  (0 ≤ x) ∧ (0 ≤ y) ∧ (0 ≤ z) ∧ (x + y + z = 1) → 
  (x^2 * y^2 + y^2 * z^2 + z^2 * x^2 + x^2 * y^2 * z^2 ≤ 1 / 16)

theorem math_solution (x y z : ℝ) :
  math_problem x y z := 
by
  sorry

end math_solution_l4_4460


namespace find_minimum_value_of_f_l4_4227

noncomputable def f (x: ℝ) : ℝ :=
  ∑ k in Finset.range 51, (x - (2 * k + 1))^2

theorem find_minimum_value_of_f :
  ∃ x, f x = 44200 :=
sorry

end find_minimum_value_of_f_l4_4227


namespace train_passes_jogger_in_36_seconds_l4_4648

-- Definitions for initial conditions
def speed_jogger : ℝ := 9 * (1000.0 / 3600.0)
def speed_train : ℝ := 45 * (1000.0 / 3600.0)
def distance_ahead : ℝ := 240
def train_length : ℝ := 120

-- Total distance the train needs to cover to pass the jogger
def total_distance : ℝ := distance_ahead + train_length

-- Relative speed of the train with respect to the jogger
def relative_speed : ℝ := speed_train - speed_jogger

-- Time taken for the train to pass the jogger
def time_to_pass_train : ℝ := total_distance / relative_speed

theorem train_passes_jogger_in_36_seconds :
  time_to_pass_train = 36 := by
  sorry

end train_passes_jogger_in_36_seconds_l4_4648


namespace pine_taller_than_maple_l4_4366

def height_maple : ℚ := 13 + 1 / 4
def height_pine : ℚ := 19 + 3 / 8

theorem pine_taller_than_maple :
  (height_pine - height_maple = 6 + 1 / 8) :=
sorry

end pine_taller_than_maple_l4_4366


namespace AK_length_l4_4372

/-- Let K be the point of intersection of AB and the line touching the circumcircle of ΔABC at C where m(∠A) > m(∠B). 
Let L be a point on [BC] such that m(∠ALB) = m(∠CAK), 5 * |LC| = 4 * |BL|, and |KC| = 12. 
Show that |AK| = 8. -/
theorem AK_length (A B C K L : Point) 
  (hK : K = intersection_point_of_AB_and_tangent_circumcircle_line_through_C A B C)
  (h_angle_A_gt_B : ∠A > ∠B)
  (hL_on_BC : L ∈ segment BC)
  (h_angle_ALB_eq_CAK : ∠ALB = ∠CAK)
  (h_5LC_eq_4BL : 5 * |LC| = 4 * |BL|)
  (h_KC_eq_12 : |KC| = 12) : 
  |AK| = 8 := 
sorry

end AK_length_l4_4372


namespace prob_TeamA_wins_2_1_proof_prob_TeamB_wins_proof_best_of_five_increases_prob_l4_4042

noncomputable def prob_TeamA_wins_game : ℝ := 0.6
noncomputable def prob_TeamB_wins_game : ℝ := 0.4

-- Probability of Team A winning 2-1 in a best-of-three
noncomputable def prob_TeamA_wins_2_1 : ℝ := 2 * prob_TeamA_wins_game * prob_TeamB_wins_game * prob_TeamA_wins_game 

-- Probability of Team B winning in a best-of-three
noncomputable def prob_TeamB_wins_2_0 : ℝ := prob_TeamB_wins_game * prob_TeamB_wins_game
noncomputable def prob_TeamB_wins_2_1 : ℝ := 2 * prob_TeamB_wins_game * prob_TeamA_wins_game * prob_TeamB_wins_game
noncomputable def prob_TeamB_wins : ℝ := prob_TeamB_wins_2_0 + prob_TeamB_wins_2_1

-- Probability of Team A winning in a best-of-three
noncomputable def prob_TeamA_wins_best_of_three : ℝ := 1 - prob_TeamB_wins

-- Probability of Team A winning in a best-of-five
noncomputable def prob_TeamA_wins_3_0 : ℝ := prob_TeamA_wins_game * prob_TeamA_wins_game * prob_TeamA_wins_game
noncomputable def prob_TeamA_wins_3_1 : ℝ := 3 * (prob_TeamA_wins_game * prob_TeamA_wins_game * prob_TeamB_wins_game * prob_TeamA_wins_game)
noncomputable def prob_TeamA_wins_3_2 : ℝ := 6 * (prob_TeamA_wins_game * prob_TeamA_wins_game * prob_TeamB_wins_game * prob_TeamB_wins_game * prob_TeamA_wins_game)

noncomputable def prob_TeamA_wins_best_of_five : ℝ := prob_TeamA_wins_3_0 + prob_TeamA_wins_3_1 + prob_TeamA_wins_3_2

theorem prob_TeamA_wins_2_1_proof :
  prob_TeamA_wins_2_1 = 0.288 :=
sorry

theorem prob_TeamB_wins_proof :
  prob_TeamB_wins = 0.352 :=
sorry

theorem best_of_five_increases_prob :
  prob_TeamA_wins_best_of_three < prob_TeamA_wins_best_of_five :=
sorry

end prob_TeamA_wins_2_1_proof_prob_TeamB_wins_proof_best_of_five_increases_prob_l4_4042


namespace students_play_basketball_l4_4345

theorem students_play_basketball 
  (total_students : ℕ)
  (cricket_players : ℕ)
  (both_players : ℕ)
  (total_students_eq : total_students = 880)
  (cricket_players_eq : cricket_players = 500)
  (both_players_eq : both_players = 220) 
  : ∃ B : ℕ, B = 600 :=
by
  sorry

end students_play_basketball_l4_4345


namespace circle_m_eq_d1_d2_eq_l4_4763

noncomputable def circle_c (x y : ℝ) (r : ℝ) := (x + 2) ^ 2 + y ^ 2 = r ^ 2
noncomputable def circle_m (x y : ℝ) := (x - 2) ^ 2 + y ^ 2 = 4

theorem circle_m_eq :
  ∃ r > 0, ∀ x y, 
  circle_c x y r ∧ (1, -real.sqrt 3) = (x, y) ∧ (x - 2) ^ 2 + y ^ 2 = r ^ 2 → 
  (circle_m x y) :=
sorry

theorem d1_d2_eq (d1 d2 : ℝ) :
  ∀ (l1 l2 : set (ℝ × ℝ)), -- assuming l1 and l2 are lines
  (mutually_perpendicular l1 l2) ∧ 
  ∃ (a : ℝ × ℝ), a = (-1, 0) ∧ 
  lengths_of_chords_intercepted_by_circle l1 l2 (circle_c x y r) (d1, d2) ∧ 
  (d1 = length_of_chord_intercepted_by_circle l1 (circle_c x y r)) ∧
  (d2 = length_of_chord_intercepted_by_circle l2 (circle_c x y r)) →
  (d1 ^ 2 + d2 ^ 2 = 28) :=
sorry

end circle_m_eq_d1_d2_eq_l4_4763


namespace number_of_boys_l4_4512

theorem number_of_boys
  (average_weight_boys : ℕ → ℝ)
  (total_students : ℕ)
  (average_weight_class : ℝ)
  (average_weight_girls : ℕ → ℝ)
  (number_of_girls : ℕ)
  (B : ℕ)
  (average_weight_boys_val : ∀ B, average_weight_boys B = 48)
  (total_students_val : total_students = 25)
  (average_weight_class_val : average_weight_class = 45)
  (average_weight_girls_val : ∀ G, average_weight_girls G = 40.5)
  (number_of_girls_val : number_of_girls = 15) :
  B = total_students - number_of_girls :=
by
  have : total_students = 25 := total_students_val
  have : number_of_girls = 15 := number_of_girls_val
  have B_val : B = 25 - 15 := by
    rw [this, this]
    sorry
  exact B_val

end number_of_boys_l4_4512


namespace smallest_k_l4_4463

noncomputable def f (a b : ℕ) (M : ℤ) (n : ℤ) : ℤ :=
  if n < M then n + a else n - b

noncomputable def f_iter (a b : ℕ) (M : ℤ) (n : ℤ) (i : ℕ) : ℤ :=
  Nat.recOn i n (λ _ acc, f a b M acc)

theorem smallest_k (a b : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ b) :
  ∃ k : ℕ, k = (a + b) / Nat.gcd a b ∧ f_iter a b (Int.ofNat (Nat.ediv (a + b) 2)) 0 k = 0 := 
by
  sorry
\
end smallest_k_l4_4463


namespace cube_side_length_increase_20_percent_l4_4536

variable {s : ℝ} (initial_side_length_increase : ℝ) (percentage_surface_area_increase : ℝ) (percentage_volume_increase : ℝ)
variable (new_surface_area : ℝ) (new_volume : ℝ)

theorem cube_side_length_increase_20_percent :
  ∀ (s : ℝ),
  (initial_side_length_increase = 1.2 * s) →
  (new_surface_area = 6 * (1.2 * s)^2) →
  (new_volume = (1.2 * s)^3) →
  (percentage_surface_area_increase = ((new_surface_area - (6 * s^2)) / (6 * s^2)) * 100) →
  (percentage_volume_increase = ((new_volume - s^3) / s^3) * 100) →
  5 * (percentage_volume_increase - percentage_surface_area_increase) = 144 := by
  sorry

end cube_side_length_increase_20_percent_l4_4536


namespace percentage_of_cars_on_monday_compared_to_tuesday_l4_4083

theorem percentage_of_cars_on_monday_compared_to_tuesday : 
  ∀ (cars_mon cars_tue cars_wed cars_thu cars_fri cars_sat cars_sun : ℕ),
    cars_mon + cars_tue + cars_wed + cars_thu + cars_fri + cars_sat + cars_sun = 97 →
    cars_tue = 25 →
    cars_wed = cars_mon + 2 →
    cars_thu = 10 →
    cars_fri = 10 →
    cars_sat = 5 →
    cars_sun = 5 →
    (cars_mon * 100 / cars_tue = 80) :=
by
  intros cars_mon cars_tue cars_wed cars_thu cars_fri cars_sat cars_sun
  intro h_total
  intro h_tue
  intro h_wed
  intro h_thu
  intro h_fri
  intro h_sat
  intro h_sun
  sorry

end percentage_of_cars_on_monday_compared_to_tuesday_l4_4083


namespace determine_segment_length_l4_4835

noncomputable def segment_length_in_triangle (XY YZ XZ : ℝ) (P : Type) (d : ℝ) :=
  XY = 500 ∧ YZ = 550 ∧ XZ = 600 ∧
  ∃ (interior_point : P), 
  ∀ (through_P_is_parallel_to_sides : P → Prop), 
  (through_P_is_parallel_to_sides interior_point → length_of_segment d = 187.5)

theorem determine_segment_length :
  segment_length_in_triangle 500 550 600 _

end determine_segment_length_l4_4835


namespace find_smallest_a_l4_4238

theorem find_smallest_a :
  let θ := Real.pi / 6,
      cos_θ := Real.cos θ,
      sin_θ := Real.sin θ in
  (cos_θ = Real.sqrt 3 / 2 ∧ sin_θ = 1 / 2) →
  ∃ (a : ℝ), a = Real.sqrt (35 / 6) ∧
    (9 * Real.sqrt ((3 * a)^2 + cos_θ^2) - 6 * a^2 - sin_θ^2) / (Real.sqrt (1 + 6 * a^2) + 4) = 3 :=
by
  -- Definitions for clarity
  let θ := Real.pi / 6
  let cos_θ := Real.cos θ
  let sin_θ := Real.sin θ
  -- Assume the trigonometric identities given in the problem
  have h_cos : cos_θ = Real.sqrt 3 / 2 := by sorry
  have h_sin : sin_θ = 1 / 2 := by sorry
  -- Use the assumptions to show the required statement
  use Real.sqrt (35 / 6)
  split
  · exact rfl
  · sorry

end find_smallest_a_l4_4238


namespace count_triangles_in_figure_l4_4819

theorem count_triangles_in_figure : 
  let figure := (draw_rectangle 40 30) ++ 
                (draw_vertical_line 20 0 30) ++ 
                (draw_diagonal_line 0 0 20 30) ++
                (draw_diagonal_line 20 0 0 30) ++
                (draw_diagonal_line 20 0 40 30) ++
                (draw_diagonal_line 40 0 20 30) ++
                (draw_horizontal_line 0 15 40) ++
                (draw_vertical_line 10 0 30) ++
                (draw_vertical_line 30 0 30) ++
                (draw_diagonal_line 0 0 40 30) 
  in count_triangles figure = 68 := by
  sorry

end count_triangles_in_figure_l4_4819


namespace number_is_correct_l4_4329

theorem number_is_correct (x : ℝ) (h : (30 / 100) * x = (25 / 100) * 40) : x = 100 / 3 :=
by
  -- Proof would go here.
  sorry

end number_is_correct_l4_4329


namespace ratio_of_comedies_to_action_movies_l4_4846

theorem ratio_of_comedies_to_action_movies (comedies rented action_movies : ℕ) (h1 : comedies = 15) (h2 : action_movies = 5) : comedies / action_movies = 3 :=
by
  rw [h1, h2]
  norm_num

end ratio_of_comedies_to_action_movies_l4_4846


namespace simplify_fraction_l4_4503

theorem simplify_fraction (x : ℚ) : 
  (↑(x + 2) / 4 + ↑(3 - 4 * x) / 3 : ℚ) = ((-13 * x + 18) / 12 : ℚ) :=
by 
  sorry

end simplify_fraction_l4_4503


namespace player_A_winning_strategy_l4_4082

-- Definitions for the problem conditions
def initial_count : ℕ := 10000000

def valid_moves (n : ℕ) : Prop :=
  (∃ p : ℕ, p.prime ∧ ∃ k : ℕ, n = p^k) ∧ n ≤ 5

def valid_move_set : set ℕ := {1, 2, 3, 4, 5}

-- The main theorem stating that Player A has a winning strategy
theorem player_A_winning_strategy : ∃ winning_strategy : (ℕ → ℕ), 
  (∀ n, n = initial_count → winning_strategy 4 = 9999996 ∧ 
  ∀ k ∈ valid_move_set, 
    (∃ m ∈ valid_move_set, (n - winning_strategy m) % 6 = 0))
:= sorry

end player_A_winning_strategy_l4_4082


namespace reimbursement_correct_l4_4475

noncomputable def chairs := 3 * 15
noncomputable def tables := 5 * 15
noncomputable def cabinets := 2 * 15

noncomputable def base_price_chairs := chairs * 120
noncomputable def base_price_tables := tables * 220
noncomputable def base_price_cabinets := cabinets * 340

noncomputable def total_base_price := base_price_chairs + base_price_tables + base_price_cabinets

noncomputable def discounted_price_chairs := base_price_chairs * 0.85
noncomputable def discounted_price_tables := base_price_tables * 0.92
noncomputable def discounted_price_cabinets := base_price_cabinets * 0.93

noncomputable def total_discounted_price := discounted_price_chairs + discounted_price_tables + discounted_price_cabinets

noncomputable def sales_tax := total_discounted_price * 0.07
noncomputable def delivery_charge := 250
noncomputable def assembly_charge := 350

noncomputable def correct_total := total_discounted_price + sales_tax + delivery_charge + assembly_charge
noncomputable def amount_paid := 20700
noncomputable def reimbursement := correct_total - amount_paid

theorem reimbursement_correct : reimbursement = 11203.92 := sorry

end reimbursement_correct_l4_4475


namespace samuel_faster_than_sarah_l4_4929

theorem samuel_faster_than_sarah
  (efficiency_samuel : ℝ := 0.90)
  (efficiency_sarah : ℝ := 0.75)
  (efficiency_tim : ℝ := 0.80)
  (time_tim : ℝ := 45)
  : (time_tim * efficiency_tim / efficiency_sarah) - (time_tim * efficiency_tim / efficiency_samuel) = 8 :=
by
  sorry

end samuel_faster_than_sarah_l4_4929


namespace triangle_angle_sum_l4_4418

variable (ABC : Type) [triangle ABC]

-- Definitions from conditions in a)
variable (A B C M N K : point ABC)
variable (M_midpoint : midpoint M A C)
variable (N_on_AM : lies_on N A M)
variable (angle_eq1 : ∠ M B N = ∠ C B M)
variable (K_ext_BN : lies_on_ext K B N)
variable (angle_eq2 : ∠ B M K = 90)

-- Goal to prove
theorem triangle_angle_sum (HC : M_midpoint) (HN : N_on_AM) (HB : angle_eq1) (HK : K_ext_BN) (HA : angle_eq2) :
  segment_length B C = segment_length A K + segment_length B K := sorry

end triangle_angle_sum_l4_4418


namespace right_triangle_unique_k_l4_4528

/-- The perimeter of a triangle is 30, two sides are 8 and 12. 
    The number of values of k (a positive integer representing the third side) 
    such that the triangle is a right triangle is 1. -/
theorem right_triangle_unique_k :
  let per := 30
  let a := 8
  let b := 12
  ∃! (k : ℕ), (a + b + k = per) ∧ 
               ((k * k = a * a + b * b) ∨ (a * a + k * k = b * b) ∨ (b * b + k * k = a * a)) :=
begin
  sorry  -- the proof goes here
end

end right_triangle_unique_k_l4_4528


namespace problem_statement_l4_4894

noncomputable def max_value_d (a b c d : ℝ) : Prop :=
a + b + c + d = 10 ∧
(ab + ac + ad + bc + bd + cd = 20) ∧
∀ x, (a + b + c + x = 10 ∧ ab + ac + ad + bc + bd + cd = 20) → x ≤ 5 + Real.sqrt 105 / 2

theorem problem_statement (a b c d : ℝ) :
  max_value_d a b c d → d = (5 + Real.sqrt 105) / 2 :=
sorry

end problem_statement_l4_4894


namespace triangle_equality_BC_AK_BK_l4_4414

-- Define the main geometric constructs
variables {A B C M N K : Type}

-- Given conditions
variables [Midpoint M A C] -- M is the midpoint of AC
variables [OnSegment N A M] -- N lies on segment AM
variables [EqualAngle M B N C B M] -- ∠MBN = ∠CBM
variables [OnExtension K B N (90 : Angle)] -- K is on the extension of BN such that ∠BMK is right

-- Define the main theorem
theorem triangle_equality_BC_AK_BK :
  BC = AK + BK :=
sorry

end triangle_equality_BC_AK_BK_l4_4414


namespace domain_of_f_l4_4741

noncomputable def f (x : ℝ) : ℝ := (x^5 - 4*x^3 + 6*x^2 - 4*x + 1) / (x^2 - 9)

theorem domain_of_f :
  ∀ x : ℝ, f x ∈ ℝ ↔ (x ≠ 3 ∧ x ≠ -3) := 
by
  sorry

end domain_of_f_l4_4741


namespace range_of_g_in_interval_l4_4748

def f (x : ℝ) := Real.sin (x - Real.pi / 6)
def g (x : ℝ) := Real.sin (2 * x - 5 * Real.pi / 6)
def interval : Set ℝ := Set.Ioo (Real.pi / 4) (3 * Real.pi / 4)
def range_g : Set ℝ := (Set.Ioc (-Real.sqrt 3 / 2) 1)

theorem range_of_g_in_interval : (Set.maps_to g interval range_g) :=
sorry

end range_of_g_in_interval_l4_4748


namespace part1_l4_4293

section
variable {a : ℝ}

def f (x : ℝ) : ℝ := a * Real.exp x

theorem part1 (h : ∀ x, f x = a * Real.exp x)
    (h_tangent : ∀ x, ∃ k b, y = k * (x - 1) + f 1)
    (h_point : y = 3 ∧ y = f (1) ∧ f (3) = 3) : 
    (a = 1 / Real.exp 1) ∧
    ( ∀ x, (x + 1) * Real.exp (x - 1) > 0 ↔ x > -1) ∧
    ( ∀ x, (x + 1) * Real.exp (x - 1) < 0 ↔ x < -1) :=
sorry
end

end part1_l4_4293


namespace greatest_product_of_sum_246_l4_4107

theorem greatest_product_of_sum_246 :
  ∃ x : ℤ, x * (246 - x) = 15129 ∧ (∀ y : ℤ, y + (246 - y) = 246 → y * (246 - y) ≤ 15129) :=
begin
  sorry,
end

end greatest_product_of_sum_246_l4_4107


namespace probability_product_multiple_of_105_l4_4834

def S : Set ℕ := {3, 5, 7, 21, 25, 35, 42, 51, 70}

def is_multiple_of_105 (a b : ℕ) : Prop :=
  a * b % 105 = 0

def num_success := (finset.powerset_len 2 S.to_finset).count (λ p, match p.val with
  | [a, b] => is_multiple_of_105 a b
  | _      => false
  end)

def num_total := finset.card (finset.powerset_len 2 S.to_finset)

def P := (num_success : ℚ) / num_total

theorem probability_product_multiple_of_105 :
  P = 2 / 9 :=
by
  sorry

end probability_product_multiple_of_105_l4_4834


namespace irrational_triangle_area_in_quad_l4_4721

theorem irrational_triangle_area_in_quad (ABCD : Type) [trapezoid ABCD] 
  (BC AD : ℝ) (h : ℝ) (A : ℝ) 
  [BC_eq : BC = 1] 
  [AD_eq : AD = real.cbrt 2]
  [area_eq : A = 1] :
  ∃ O : Type, ∃ (O_inside : O ∈ interior ABCD), 
  (∀ α β : ℝ, α = height O AD → β = height O BC → 
  α + β / real.cbrt 2 = 2 / (1 + real.cbrt 2)) → 
  irrational (area (triangle O A B)) ∨ irrational (area (triangle O B C)) ∨ irrational (area (triangle O C D)) ∨ irrational (area (triangle O D A)) :=
by 
  sorry

end irrational_triangle_area_in_quad_l4_4721


namespace percent_of_ducks_among_non_swans_l4_4370

theorem percent_of_ducks_among_non_swans
  (total_birds : ℕ) 
  (percent_ducks percent_swans percent_eagles percent_sparrows : ℕ)
  (h1 : percent_ducks = 40) 
  (h2 : percent_swans = 20) 
  (h3 : percent_eagles = 15) 
  (h4 : percent_sparrows = 25)
  (h_sum : percent_ducks + percent_swans + percent_eagles + percent_sparrows = 100) :
  (percent_ducks * 100) / (100 - percent_swans) = 50 :=
by
  sorry

end percent_of_ducks_among_non_swans_l4_4370


namespace foma_gives_ierema_55_l4_4553

-- Defining the amounts that Foma, Ierema, and Yuliy have initially
variables (F E Y : ℝ)

-- Given conditions
def condition_1 : Prop := F - E = 140
def condition_2 : Prop := F - Y = 40
def condition_3 : Prop := Y = E + 70

-- Prove that Foma should give Ierema 55 gold coins
theorem foma_gives_ierema_55 (h1 : condition_1) (h2 : condition_2) (h3 : condition_3) : 
  (F - E) / 2 = 55 :=
by
  sorry

end foma_gives_ierema_55_l4_4553


namespace maximum_value_of_f_inequality_holds_for_all_x_l4_4291

noncomputable def f (a x : ℝ) : ℝ := (a * x^2 + x + a) * Real.exp (-x)

theorem maximum_value_of_f (a : ℝ) (h : 0 ≤ a) : 
  (∀ x, f a x ≤ f a 1) → f a 1 = 3 / Real.exp 1 → a = 1 := 
by 
  sorry

theorem inequality_holds_for_all_x (b : ℝ) : 
  (∀ a ≤ 0, ∀ x, 0 ≤ x → f a x ≤ b * Real.log (x + 1)) → 1 ≤ b := 
by 
  sorry

end maximum_value_of_f_inequality_holds_for_all_x_l4_4291


namespace proof_problem_l4_4911

-- Definitions of sets A, B, and C based on the conditions in the problem
def setA : Set (List String) :=
  { l | ∃ a1 a2 a3 a4 a5 : String, l = ["(" ++ "(" ++ (a1 ++ a2) ++ ")" ++ a3 ++ ")" ++ "(" ++ (a4 ++ a5) ++ ")"] }

def setB : Set (List (Nat × Nat)) :=
  { l | ∃ hexagon : List (Nat × Nat), hexagon.length = 6 ∧ 
        ∀ i, 0 ≤ i ∧ i < hexagon.length → 
        ∃ t1 t2 t3 t4 : (Nat × Nat), 
            [t1, t2, t3, t4] ⊆ hexagon ∧
            (t1.1 = t2.1 → t1.2 ≠ t2.2) ∧ 
            (t2.1 = t3.1 → t2.2 ≠ t3.2) ∧ 
            (t3.1 = t4.1 → t3.2 ≠ t4.2) }

def setC : Set (List Char) :=
  { l | l.perm (List.replicate 4 'W' ++ List.replicate 4 'B') ∧
        ∀ i (h : 0 ≤ i ∧ i < l.length), 
        l.take i.count ('W') ≥ l.take i.count ('B') }

theorem proof_problem : ∀ (A B C : Set (List String)), setA = A ∧ setB = B ∧ setC = C → A.card = B.card ∧ B.card = C.card :=
by
  intros A B C
  assume h
  have h1 : A.card = B.card := sorry
  have h2 : B.card = C.card := sorry
  exact ⟨h1, h2⟩

end proof_problem_l4_4911


namespace option_c_is_not_equal_l4_4994

theorem option_c_is_not_equal :
  let A := 14 / 12
  let B := 1 + 1 / 6
  let C := 1 + 1 / 2
  let D := 1 + 7 / 42
  let E := 1 + 14 / 84
  A = 7 / 6 ∧ B = 7 / 6 ∧ D = 7 / 6 ∧ E = 7 / 6 ∧ C ≠ 7 / 6 :=
by
  sorry

end option_c_is_not_equal_l4_4994


namespace min_odd_integers_l4_4606

-- Definitions of the conditions
variable (a b c d e f : ℤ)

-- The mathematical theorem statement
theorem min_odd_integers 
  (h1 : a + b = 30)
  (h2 : a + b + c + d = 50) 
  (h3 : a + b + c + d + e + f = 70)
  (h4 : e + f % 2 = 1) : 
  ∃ n, n ≥ 1 ∧ n = (if a % 2 = 1 then 1 else 0) + (if b % 2 = 1 then 1 else 0) + 
                    (if c % 2 = 1 then 1 else 0) + (if d % 2 = 1 then 1 else 0) + 
                    (if e % 2 = 1 then 1 else 0) + (if f % 2 = 1 then 1 else 0) :=
sorry

end min_odd_integers_l4_4606


namespace round_to_nearest_tenth_l4_4496

theorem round_to_nearest_tenth (x : ℝ) (h1 : x = 3.45) : (Float.round (10 * x) / 10 : ℝ) = 3.5 :=
by
  have h2 : 10 * x = 34.5 := by 
    rw [h1]
    norm_num
  rw [h2]
  norm_num
  -- Placeholder for proof completion
  sorry

end round_to_nearest_tenth_l4_4496


namespace Karsyn_payment_l4_4183

def percentage : ℝ := 20
def initial_price : ℝ := 600

theorem Karsyn_payment : (percentage / 100) * initial_price = 120 :=
by 
  sorry

end Karsyn_payment_l4_4183


namespace oranges_in_each_box_l4_4698

theorem oranges_in_each_box (O B : ℕ) (h1 : O = 24) (h2 : B = 3) :
  O / B = 8 :=
by
  sorry

end oranges_in_each_box_l4_4698


namespace Sandy_loses_2_marks_per_incorrect_sum_l4_4025

theorem Sandy_loses_2_marks_per_incorrect_sum
    (marks_per_correct_sum : ℕ)
    (total_attempted_sums : ℕ)
    (total_marks_obtained : ℕ)
    (correct_sums : ℕ)
    (incorrect_sums : ℕ)
    (total_lost_marks : ℕ)
    (marks_lost_per_incorrect_sum : ℕ) 
    (h1 : marks_per_correct_sum = 3)
    (h2 : total_attempted_sums = 30)
    (h3 : total_marks_obtained = 55)
    (h4 : correct_sums = 23)
    (h5 : incorrect_sums = total_attempted_sums - correct_sums)
    (h6 : total_lost_marks = (correct_sums * marks_per_correct_sum) - total_marks_obtained)
    (h7 : marks_lost_per_incorrect_sum = total_lost_marks / incorrect_sums) :
  marks_lost_per_incorrect_sum = 2 :=
by
  simp [h1, h2, h3, h4, h5, h6, h7]
  exact sorry

end Sandy_loses_2_marks_per_incorrect_sum_l4_4025


namespace train_length_l4_4181

theorem train_length 
  (speed_train_kmph : ℝ) (speed_man_kmph : ℝ) (time_seconds : ℝ)
  (h1 : speed_train_kmph = 90) 
  (h2 : speed_man_kmph = 6) 
  (h3 : time_seconds = 6) :
  let relative_speed_kmph := speed_train_kmph + speed_man_kmph in
  let relative_speed_mps := (relative_speed_kmph * 1000) / 3600 in
  let length_train_meters := relative_speed_mps * time_seconds in
  length_train_meters = 480 :=
by
  sorry

end train_length_l4_4181


namespace photographer_choices_l4_4980

theorem photographer_choices (P L M S : ℕ) (hP : P = 10) (hL : L = 8) (hM : M = 5) (hS : S = 4) :
  (nat.choose P 2) * (nat.choose L 2) * (nat.choose M 1) * (nat.choose S 1) = 25200 :=
by {
  rw [hP, hL, hM, hS],
  -- Note: Calculation steps can be written here, but we use sorry as a placeholder
  sorry
}

end photographer_choices_l4_4980
