import Mathlib
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Equiv
import Mathlib.Algebra.Field.Basic
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Combinatorics
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Nat.Gcd
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.LinearCombination
import Mathlib.Tactic.NormNum
import Mathlib.Topology.Basic
import Mathlib.Topology.EuclideanSpace.Basic

namespace log3_of_9_to_3_l242_242810

theorem log3_of_9_to_3 : Real.logb 3 (9^3) = 6 :=
by
  have h1 : 9 = 3^2 := by norm_num
  have h2 : 9^3 = (3^2)^3 := by rw [h1]
  have h3 : (3^2)^3 = 3^(2 * 3) := by rw [pow_mul]
  have h4 : Real.logb 3 (3^(2 * 3)) = 2 * 3 := Real.logb_pow 3 (2 * 3)
  rw [h4]
  norm_num

end log3_of_9_to_3_l242_242810


namespace swimming_speed_l242_242329

theorem swimming_speed
    (s : ℝ) (h₁ : s = 5)
    (h₂ : ∀ (v t : ℝ), t > 0 → (v + s) * t = (v - s) * (2 * t)) :
    ∃ v : ℝ, v = 15 :=
        by
        have eq1 : ∀ v t : ℝ, t > 0 -> (v + s) * t = (v - s) * (2 * t) := h₂
        have eq2 : (15 + s) * t = (15 - s) * (2 * t) by sorry
        exact ⟨15, eq2⟩

end swimming_speed_l242_242329


namespace first_term_exceeding_10000_l242_242225

noncomputable def sequence (n : ℕ) : ℕ :=
  if n = 0 then 2
  else (Finset.range n).sum (λ i, sequence i)

theorem first_term_exceeding_10000 :
  ∃ n : ℕ, sequence n > 10000 ∧ sequence n = 16384 :=
sorry

end first_term_exceeding_10000_l242_242225


namespace cube_identity_l242_242114

theorem cube_identity (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 := by
  sorry

end cube_identity_l242_242114


namespace solve_system_l242_242061

variable (a b : ℝ)
noncomputable def x_sol : ℝ := (a + b * Real.sqrt (a^2 - b^2)) / Real.sqrt (1 - a^2 + b^2)
noncomputable def y_sol : ℝ := (b + a * Real.sqrt (a^2 - b^2)) / Real.sqrt (1 - a^2 + b^2)

theorem solve_system (a b : ℝ) (h1 : 0 ≤ a^2 - b^2) (h2 : a^2 - b^2 < 1) :
  let x := x_sol a b in
  let y := y_sol a b in
  (x - y * Real.sqrt(x^2 - y^2)) / Real.sqrt(1 - x^2 + y^2) = a ∧
  (y - x * Real.sqrt(x^2 - y^2)) / Real.sqrt(1 - x^2 + y^2) = b :=
by
  sorry

end solve_system_l242_242061


namespace sum_even_integers_200_to_400_l242_242284

theorem sum_even_integers_200_to_400 : 
  let seq := list.range' 200 ((400 - 200) / 2 + 1)
  in seq.filter (λ n, n % 2 = 0) = list.range' 200 101 ∧ 
     seq.sum = 30300 := 
by
  sorry

end sum_even_integers_200_to_400_l242_242284


namespace ratio_spent_on_movies_l242_242986

-- Definitions for given conditions
def weekly_allowance : ℕ := 16
def end_amount : ℕ := 14
def earned_washing_car : ℕ := 6

-- Lean statement to prove the problem
theorem ratio_spent_on_movies :
  let total_before_movies := end_amount + earned_washing_car in
  let spent_on_movies := total_before_movies - weekly_allowance in
  spent_on_movies / weekly_allowance = 1 / 4 :=
by
  sorry

end ratio_spent_on_movies_l242_242986


namespace isosceles_in_27_gon_l242_242417

def vertices := {x : ℕ // x < 27}

def is_isosceles_triangle (a b c : vertices) : Prop :=
  (a.val + c.val) / 2 % 27 = b.val

def is_isosceles_trapezoid (a b c d : vertices) : Prop :=
  (a.val + d.val) / 2 % 27 = (b.val + c.val) / 2 % 27

def seven_points_form_isosceles (s : Finset vertices) : Prop :=
  ∃ (a b c : vertices) (h1 : a ∈ s) (h2 : b ∈ s) (h3 : c ∈ s), is_isosceles_triangle a b c

def seven_points_form_isosceles_trapezoid (s : Finset vertices) : Prop :=
  ∃ (a b c d : vertices) (h1 : a ∈ s) (h2 : b ∈ s) (h3 : c ∈ s) (h4 : d ∈ s), is_isosceles_trapezoid a b c d

theorem isosceles_in_27_gon :
  ∀ (s : Finset vertices), s.card = 7 → 
  (seven_points_form_isosceles s) ∨ (seven_points_form_isosceles_trapezoid s) :=
by sorry

end isosceles_in_27_gon_l242_242417


namespace inequality_a2b3c_l242_242182

theorem inequality_a2b3c {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : 1 / a + 1 / (2 * b) + 1 / (3 * c) = 1) : 
  a + 2 * b + 3 * c ≥ 9 :=
sorry

end inequality_a2b3c_l242_242182


namespace value_f_at_pi_over_3_decreasing_interval_and_symmetry_axis_l242_242050

noncomputable def f (x: ℝ) : ℝ := 2 * sqrt(3) * sin (x / 2) * cos (x / 2) - 2 * (cos (x / 2))^2

theorem value_f_at_pi_over_3 : f (π / 3) = 0 := sorry

theorem decreasing_interval_and_symmetry_axis : 
  (∀ k : ℤ, ∀ x ∈ set.Icc ((2 * π / 3) + 2 * k * π) ((5 * π / 3) + 2 * k * π), 
    (2 * cos (x - π / 6)) < 0) ∧ 
  (∀ k : ℤ, axis_of_symmetry (f) (x = (2 * π / 3) + k * π)) := sorry

end value_f_at_pi_over_3_decreasing_interval_and_symmetry_axis_l242_242050


namespace total_days_1996_to_2000_l242_242487

def is_leap_year (year : ℕ) : Prop :=
  (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ (year % 400 = 0)

def days_in_year (year : ℕ) : ℕ :=
  if is_leap_year year then 366 else 365

theorem total_days_1996_to_2000 :
  (days_in_year 1996) + (days_in_year 1997) + (days_in_year 1998) + (days_in_year 1999) + (days_in_year 2000) = 1827 :=
by sorry

end total_days_1996_to_2000_l242_242487


namespace probability_heads_even_after_100_flips_l242_242728

variable (n : Nat) (P : Nat → ℝ)

-- Define the recursive relation for P_n based on the problem conditions
def recurrence_relation (n : Nat) (P : Nat → ℝ) : ℝ :=
  3/4 - 1/2 * P (n - 1)

-- Define the function representing the probability at a given n
noncomputable def probability_even_heads (n : Nat) (P : Nat → ℝ) : ℝ :=
  if n = 0 then 1 else recurrence_relation n P

-- Formulate the final proof problem statement
theorem probability_heads_even_after_100_flips :
  probability_even_heads 100 (λ n, 1/2 * (1 + (1 / 4^n))) = 1/2 * (1 + 1/4^100) :=
by
  sorry

end probability_heads_even_after_100_flips_l242_242728


namespace cubic_sum_l242_242108

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 :=
by {
  sorry
}

end cubic_sum_l242_242108


namespace cubic_sum_identity_l242_242091

theorem cubic_sum_identity (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 :=
sorry

end cubic_sum_identity_l242_242091


namespace cubic_sum_identity_l242_242096

theorem cubic_sum_identity (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 :=
sorry

end cubic_sum_identity_l242_242096


namespace min_value_point_on_line_l242_242437

theorem min_value_point_on_line (m n : ℝ) (h : m + 2 * n = 1) : 
  2^m + 4^n ≥ 2 * Real.sqrt 2 :=
by
  sorry

end min_value_point_on_line_l242_242437


namespace first_term_exceeds_10000_l242_242223

def sequence : ℕ → ℕ
| 0     := 2
| (n+1) := (finset.sum (finset.range n.succ) sequence) 

theorem first_term_exceeds_10000 :
  ∃ n, sequence n > 10000 ∧ sequence n = 16384 :=
sorry

end first_term_exceeds_10000_l242_242223


namespace log_base_3_of_9_cubed_l242_242773

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 :=
by
  -- Given conditions:
  -- 1. 9 = 3^2
  -- 2. log_b (a^n) = n * log_b a
  sorry

end log_base_3_of_9_cubed_l242_242773


namespace convex_cyclic_quadrilaterals_count_l242_242073

theorem convex_cyclic_quadrilaterals_count :
  let num_quadrilaterals := ∑ i in (finset.range 36).powerset.filter(λ s, s.card = 4 
    ∧ let (a, b, c, d) := classical.some (vector.sorted_enum s)
    in a + b + c + d = 36 ∧ a < b + c + d ∧ b < a + c + d ∧ c < a + b + d ∧ d < a + b + c 
  ),
  finset.count :=
  num_quadrilaterals = 819 :=
begin
  sorry
end

end convex_cyclic_quadrilaterals_count_l242_242073


namespace sum_b_50_l242_242625

def S (n : ℕ) : ℕ := n^2 + n + 1

def a : ℕ → ℕ
| 0       := 0  -- a_0 is not defined in the original sequence, but let's assign it 0 to make our function total
| 1       := S 1
| (n + 2) := S (n + 2) - S (n + 1)

def b (n : ℕ) : ℕ := (-1 : ℤ)^(n + 1) * a n

def sum_b (n : ℕ) : ℤ :=
  (List.range n).map b |> List.sum

theorem sum_b_50 : sum_b 50 = 49 :=
sorry

end sum_b_50_l242_242625


namespace rhombus_area_l242_242257

theorem rhombus_area (a : ℝ) (θ : ℝ) (h₁ : a = 4) (h₂ : θ = π / 4) : 
    (a * a * Real.sin θ) = 16 :=
by
    have s1 : Real.sin (π / 4) = Real.sqrt 2 / 2 := Real.sin_pi_div_four
    rw [h₁, h₂, s1]
    have s2 : 4 * 4 * (Real.sqrt 2 / 2) = 16 := by norm_num
    exact s2

end rhombus_area_l242_242257


namespace time_interval_for_7_students_l242_242188

-- Definitions from conditions
def students_per_ride : ℕ := 7
def total_students : ℕ := 21
def total_time : ℕ := 15

-- Statement of the problem
theorem time_interval_for_7_students : (total_time / (total_students / students_per_ride)) = 5 := 
by sorry

end time_interval_for_7_students_l242_242188


namespace quadratic_has_real_roots_iff_l242_242428

theorem quadratic_has_real_roots_iff (a : ℝ) :
  (∃ (x : ℝ), a * x^2 - 4 * x - 2 = 0) ↔ (a ≥ -2 ∧ a ≠ 0) := by
  sorry

end quadratic_has_real_roots_iff_l242_242428


namespace evaluate_log_l242_242789

noncomputable def log_base (b x : ℝ) : ℝ := real.log x / real.log b

theorem evaluate_log : log_base 3 (9^3) = 6 := by
  have h1 : 9 = 3^2 := by norm_num
  have h2 : 9^3 = (3^2)^3 := by norm_num
  rw [h2, pow_mul]
  have h3 : log_base 3 (3^6) = 6 := by
    rw [← real.log_pow, real.log_mul, mul_comm, ← real.log_rpow, mul_div, real.log_self]
    norm_num
  exact h3

end evaluate_log_l242_242789


namespace log_base_3_of_9_cubed_l242_242772

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 :=
by
  -- Given conditions:
  -- 1. 9 = 3^2
  -- 2. log_b (a^n) = n * log_b a
  sorry

end log_base_3_of_9_cubed_l242_242772


namespace transformed_cosine_eq_translated_reduced_expanded_cosine_l242_242617

noncomputable def translated_reduced_expanded_cosine : ℝ → ℝ :=
  λ x, 3 * cos (2 * x + π / 3)

theorem transformed_cosine_eq_translated_reduced_expanded_cosine :
  ∀ x, translated_reduced_expanded_cosine x = 3 * cos (2 * x + π / 3) :=
by
  intro x
  sorry

end transformed_cosine_eq_translated_reduced_expanded_cosine_l242_242617


namespace circumradius_isosceles_triangle_l242_242232

theorem circumradius_isosceles_triangle (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : b < 2 * a) :
  let R := a^2 / (Real.sqrt (4 * a^2 - b^2)) in
  (∀ (circumradius : ℝ), circumradius = R) :=
by
  sorry

end circumradius_isosceles_triangle_l242_242232


namespace log_base_3_of_9_cubed_l242_242840

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 := by
  -- Define the known conditions
  have h1 : 9 = 3^2 := by norm_num
  have h2 : log 3 (a^n) = n * log 3 a := sorry  -- Statement of the logarithmic identity
  
  -- Apply the conditions to prove the result
  calc
    log 3 (9^3)
        = log 3 ((3^2)^3) : by rw h1
    ... = log 3 (3^(2 * 3)) : by congr; norm_num
    ... = log 3 (3^6) : by norm_num
    ... = 6 : by rw [h2]; norm_num

end log_base_3_of_9_cubed_l242_242840


namespace log_three_nine_cubed_l242_242949

theorem log_three_nine_cubed : ∀ (log : ℕ → ℕ → ℕ), log 3 9 = 2 → log 3 3 = 1 → (∀ (a b n : ℕ), log a (b ^ n) = n * log a b) → log 3 (9 ^ 3) = 6 :=
by
  intros log h1 h2 h3
  sorry

end log_three_nine_cubed_l242_242949


namespace log_base_3_of_9_cubed_l242_242819

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 := by
  sorry

end log_base_3_of_9_cubed_l242_242819


namespace gcd_poly_multiple_l242_242445

theorem gcd_poly_multiple {x : ℤ} (h : ∃ k : ℤ, x = 54321 * k) :
  Int.gcd ((3 * x + 4) * (8 * x + 5) * (15 * x + 11) * (x + 14)) x = 1 :=
sorry

end gcd_poly_multiple_l242_242445


namespace log_base_3_of_9_cubed_l242_242927
-- Import the necessary parts of the standard library

-- Define the statement to be proved
theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 :=
by
  sorry

end log_base_3_of_9_cubed_l242_242927


namespace log_base_3_of_9_cubed_l242_242835

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 := by
  -- Define the known conditions
  have h1 : 9 = 3^2 := by norm_num
  have h2 : log 3 (a^n) = n * log 3 a := sorry  -- Statement of the logarithmic identity
  
  -- Apply the conditions to prove the result
  calc
    log 3 (9^3)
        = log 3 ((3^2)^3) : by rw h1
    ... = log 3 (3^(2 * 3)) : by congr; norm_num
    ... = log 3 (3^6) : by norm_num
    ... = 6 : by rw [h2]; norm_num

end log_base_3_of_9_cubed_l242_242835


namespace probability_four_1s_in_five_rolls_l242_242499

open ProbabilityTheory

theorem probability_four_1s_in_five_rolls :
  let p1 := (1 / 8 : ℚ) -- probability of rolling a 1
  let p_not_1 := (7 / 8 : ℚ) -- probability of not rolling a 1
  let comb := (Nat.choose 5 4 : ℚ) -- number of ways to choose four positions out of five
  let prob := comb * (p1 ^ 4) * p_not_1 -- required probability
  prob = 35 / 32768 := 
by
  sorry

end probability_four_1s_in_five_rolls_l242_242499


namespace eval_expression_at_neg_half_l242_242604

theorem eval_expression_at_neg_half :
  let a := (-1)/2 in 
  (a + 3)^2 + (a + 3) * (a - 3) - 2 * a * (3 - a) = 1 :=
by
  sorry

end eval_expression_at_neg_half_l242_242604


namespace radius_of_surrounding_circles_l242_242685

theorem radius_of_surrounding_circles :
  ∀ (r : ℝ), 
    (∃ (C : ℝ → Prop), 
      (C 1 ∧ C r) ∧
      (∀ (x y : ℝ), x ≠ y → |x - y| = 2 * r) ∧ 
      (∀ (x : ℝ), (x ≠ 1 → x ≠ r) → |x - (1 + r)| = 1 + r) →
    r = 3 + 2 * real.sqrt 3) :=
begin
  sorry
end

end radius_of_surrounding_circles_l242_242685


namespace derivative_of_f_l242_242504

def f (x : ℝ) : ℝ := 2 * Real.cos x

theorem derivative_of_f : deriv f x = -2 * Real.sin x := sorry

end derivative_of_f_l242_242504


namespace log_base_3_of_9_cubed_l242_242817

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 := by
  sorry

end log_base_3_of_9_cubed_l242_242817


namespace binom_30_3_is_4060_l242_242368

theorem binom_30_3_is_4060 : Nat.choose 30 3 = 4060 := by
  sorry

end binom_30_3_is_4060_l242_242368


namespace cos_sum_sin_sum_cos_diff_sin_diff_l242_242526

section

variables (A B : ℝ)

-- Definition of cos and sin of angles
def cos (θ : ℝ) : ℝ := sorry
def sin (θ : ℝ) : ℝ := sorry

-- Cosine of the sum of angles
theorem cos_sum : cos (A + B) = cos A * cos B - sin A * sin B := sorry

-- Sine of the sum of angles
theorem sin_sum : sin (A + B) = sin A * cos B + cos A * sin B := sorry

-- Cosine of the difference of angles
theorem cos_diff : cos (A - B) = cos A * cos B + sin A * sin B := sorry

-- Sine of the difference of angles
theorem sin_diff : sin (A - B) = sin A * cos B - cos A * sin B := sorry

end

end cos_sum_sin_sum_cos_diff_sin_diff_l242_242526


namespace repeating_decimal_sum_is_467_l242_242233

noncomputable def repeating_decimal_fraction_sum : ℕ :=
  let x := (3.71717171 : ℚ) in
  let frac := (368 / 99 : ℚ) in
  if x = frac then (368 + 99) else 0

theorem repeating_decimal_sum_is_467 :
  repeating_decimal_fraction_sum = 467 := by
  sorry

end repeating_decimal_sum_is_467_l242_242233


namespace log_base_3_of_9_cubed_l242_242903

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 :=
by assumptions
  have h1 : 9 = 3^2 := sorry,
  have h2 : 9^3 = (3^2)^3 := sorry,
  have h3 : (3^2)^3 = 3^(2*3) := sorry,
  have h4 : 3^(2*3) = 3^6 := sorry,
  have h5 : log 3 (3^6) = 6 := by apply log_pow,
  sorry

end log_base_3_of_9_cubed_l242_242903


namespace evaluate_log_l242_242793

noncomputable def log_base (b x : ℝ) : ℝ := real.log x / real.log b

theorem evaluate_log : log_base 3 (9^3) = 6 := by
  have h1 : 9 = 3^2 := by norm_num
  have h2 : 9^3 = (3^2)^3 := by norm_num
  rw [h2, pow_mul]
  have h3 : log_base 3 (3^6) = 6 := by
    rw [← real.log_pow, real.log_mul, mul_comm, ← real.log_rpow, mul_div, real.log_self]
    norm_num
  exact h3

end evaluate_log_l242_242793


namespace gcd_84_126_l242_242407

-- Conditions
def a : ℕ := 84
def b : ℕ := 126

-- Theorem to prove gcd(a, b) = 42
theorem gcd_84_126 : Nat.gcd a b = 42 := by
  sorry

end gcd_84_126_l242_242407


namespace intersection_point_x_value_l242_242631

theorem intersection_point_x_value :
  ∃ x y : ℚ, (y = 3 * x - 22) ∧ (3 * x + y = 100) ∧ (x = 20 + 1 / 3) := by
  sorry

end intersection_point_x_value_l242_242631


namespace log_three_nine_cubed_l242_242953

theorem log_three_nine_cubed : ∀ (log : ℕ → ℕ → ℕ), log 3 9 = 2 → log 3 3 = 1 → (∀ (a b n : ℕ), log a (b ^ n) = n * log a b) → log 3 (9 ^ 3) = 6 :=
by
  intros log h1 h2 h3
  sorry

end log_three_nine_cubed_l242_242953


namespace find_simple_interest_sum_l242_242623

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r / 100) ^ n

noncomputable def simple_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * r * n / 100

theorem find_simple_interest_sum (P CIsum : ℝ)
  (simple_rate : ℝ) (simple_years : ℕ)
  (compound_rate : ℝ) (compound_years : ℕ)
  (compound_principal : ℝ)
  (hP : simple_interest P simple_rate simple_years = CIsum)
  (hCI : CIsum = (compound_interest compound_principal compound_rate compound_years - compound_principal) / 2) :
  P = 1272 :=
by
  sorry

end find_simple_interest_sum_l242_242623


namespace computation_equal_l242_242361

theorem computation_equal (a b c d : ℕ) (inv : ℚ → ℚ) (mul : ℚ → ℕ → ℚ) : 
  a = 3 → b = 1 → c = 6 → d = 2 → 
  inv ((a^b - d + c^2 + b) : ℚ) * 6 = (3 / 19) := by
  intros ha hb hc hd
  rw [ha, hb, hc, hd]
  sorry

end computation_equal_l242_242361


namespace shortest_distance_on_cuboid_surface_l242_242231

theorem shortest_distance_on_cuboid_surface (AA' AB AD : ℝ) (h1 : AA' = 1) (h2 : AB = 2) (h3 : AD = 4) :
  let d := real.sqrt (AD ^ 2 + AB ^ 2)
  in d = real.sqrt 20 :=
by
  sorry

end shortest_distance_on_cuboid_surface_l242_242231


namespace log3_of_9_to_3_l242_242802

theorem log3_of_9_to_3 : Real.logb 3 (9^3) = 6 :=
by
  have h1 : 9 = 3^2 := by norm_num
  have h2 : 9^3 = (3^2)^3 := by rw [h1]
  have h3 : (3^2)^3 = 3^(2 * 3) := by rw [pow_mul]
  have h4 : Real.logb 3 (3^(2 * 3)) = 2 * 3 := Real.logb_pow 3 (2 * 3)
  rw [h4]
  norm_num

end log3_of_9_to_3_l242_242802


namespace sequence_formula_and_sum_l242_242058

theorem sequence_formula_and_sum (a b T : ℕ → ℝ) (h : ∀ n : ℕ, n > 0 →
  (finset.sum (finset.range (n + 1)) (λ k, (k + 1) * a (k + 1)) = (n - 1) * 2^(n + 1) + 2)) :
  (∀ n : ℕ, n > 0 → a n = 2^n) ∧ 
  (∀ n : ℕ, n > 0 → 
    let b n := 1 / (real.logb 2 (a n) * real.logb 2 (a (n + 1))) in
    let T n := finset.sum (finset.range (n + 1)) b in
    T n < 1) :=
sorry

end sequence_formula_and_sum_l242_242058


namespace log_base_3_of_9_cubed_l242_242826

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 := by
  sorry

end log_base_3_of_9_cubed_l242_242826


namespace problem_solution_set_l242_242240

open Nat

def combination (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)
def permutation (n k : ℕ) : ℕ := n.factorial / (n - k).factorial

theorem problem_solution_set :
  {n : ℕ | 2 * combination n 3 ≤ permutation n 2} = {n | n = 3 ∨ n = 4 ∨ n = 5} :=
by
  sorry

end problem_solution_set_l242_242240


namespace segment_connecting_midpoints_l242_242676

variables (A B C D M N : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace M] [MetricSpace N]

-- Define a trapezoid with specific conditions
def is_trapezoid (ABCD : Set (A × B × C × D)) : Prop :=
∃ A B C D : ℝ,
  let α := ∠A + ∠D in
  α = 90 ∧
  AD > BC ∧
  M = midpoint A C ∧
  N = midpoint B D

-- Define the midpoint function on a trapezoid
def midpoint (x y : ℝ) : ℝ := (x + y) / 2

-- State the main theorem
theorem segment_connecting_midpoints (trapezoid_ABCD : Set (A × B × C × D))
  (h_trapezoid : is_trapezoid trapezoid_ABCD)
  (M N : Type) [MetricSpace M] [MetricSpace N]
  (hM : M = midpoint A C)
  (hN : N = midpoint B D) :
  distance M N = ½ * (distance A D - distance B C) :=
sorry

end segment_connecting_midpoints_l242_242676


namespace sum_coeff_eq_neg_two_l242_242529

theorem sum_coeff_eq_neg_two (a a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℝ) :
  (1 - 2*x)^7 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + a_7 * x^7 →
  a = 1 →
  a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 = -2 :=
by
  sorry

end sum_coeff_eq_neg_two_l242_242529


namespace rope_length_l242_242692

def grazed_area (r : ℝ) := (1 / 2) * r^2 * (Real.pi / 2)

theorem rope_length (A : ℝ) (hA : A = 38.48451000647496) : 
  ∃ r : ℝ, (grazed_area r = 38.48451000647496) ∧ |r - 7| < 0.1 :=
by
  sorry

end rope_length_l242_242692


namespace log_base_3_of_9_cubed_l242_242780

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 :=
by
  -- Given conditions:
  -- 1. 9 = 3^2
  -- 2. log_b (a^n) = n * log_b a
  sorry

end log_base_3_of_9_cubed_l242_242780


namespace coefficient_x2y4_in_expansion_2x_y_6_l242_242213

theorem coefficient_x2y4_in_expansion_2x_y_6 : 
  ∀ (x y : ℝ), (coeff (expand (2*x + y)^6) x^2 y^4) = 60 :=
by
  sorry

end coefficient_x2y4_in_expansion_2x_y_6_l242_242213


namespace cone_base_circumference_l242_242337

theorem cone_base_circumference
  (V : ℝ) (h : ℝ) (C : ℝ)
  (volume_eq : V = 18 * Real.pi)
  (height_eq : h = 3) :
  C = 6 * Real.sqrt 2 * Real.pi :=
sorry

end cone_base_circumference_l242_242337


namespace minimum_value_of_g_l242_242422

noncomputable def f (m x : ℝ) : ℝ := x + real.sqrt (100 - m * x)

noncomputable def g (m : ℝ) : ℝ := Real.max (λ x, f m x)

theorem minimum_value_of_g (m : ℝ) (h : m > 0) : ∃ m, m > 0 ∧ g m = 10 := 
by
  sorry

end minimum_value_of_g_l242_242422


namespace not_function_age_height_l242_242725

theorem not_function_age_height (f : ℕ → ℝ) :
  ¬(∀ (a b : ℕ), a = b → f a = f b) := sorry

end not_function_age_height_l242_242725


namespace cos2theta_minus_tantheta_l242_242044

theorem cos2theta_minus_tantheta (θ : ℝ) (h1 : 0 < θ ∧ θ < π / 2) (h2 : Real.sin θ = sqrt(5) / 5) :
  Real.cos (2 * θ) - Real.tan θ = -11 / 50 :=
sorry

end cos2theta_minus_tantheta_l242_242044


namespace ellipse_equation_quadrilateral_area_l242_242045

-- Let C be the ellipse defined with given parameters and conditions:
def ellipse_standard_equation (a b c : ℝ) :=
  a > b ∧ b > 0 ∧ c = 2 ∧ c/a = sqrt(6) / 3 ∧ a^2 = b^2 + c^2

-- (I) We need to prove that the standard equation of the ellipse is as given.
theorem ellipse_equation (a b : ℝ) (h : ellipse_standard_equation a b 2) : 
  (a = sqrt 6 ∧ b = sqrt 2) → ∀ x y : ℝ, (x^2) / 6 + (y^2) / 2 = 1 :=
by 
  sorry

-- (II) Given the conditions, prove the area of quadrilateral OPTQ is 2√3.
noncomputable def point (x y : ℝ) := (x, y)

def conditions (O F T P Q : ℝ × ℝ) :=
  O = (0, 0) ∧ F = (-2, 0) ∧ T.1 = -3 ∧ 
  (P = (x, y)) ∧ (Q = (x', y')) ∧
  ∃ m : ℝ, T = (-3, m) ∧ P.1 + Q.1 = -3 ∧ P.2 + Q.2 = m ∧ 
  quadrilateral_parallelogram O P T Q

-- We need a condition for a quadrilateral to be a parallelogram
def quadrilateral_parallelogram (O P T Q : ℝ × ℝ) :=
  P.1 + Q.1 = -3 ∧ P.2 + Q.2 = T.2

theorem quadrilateral_area (O F T P Q : ℝ × ℝ) (h1 : ellipse_standard_equation (sqrt 6) (sqrt 2) 2) (h2 : conditions O F T P Q) :
  area_of_quadrilateral O P T Q = 2 * sqrt 3 :=
by
  sorry

end ellipse_equation_quadrilateral_area_l242_242045


namespace log_base_3_of_9_cubed_l242_242821

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 := by
  sorry

end log_base_3_of_9_cubed_l242_242821


namespace third_number_in_5th_power_l242_242413

noncomputable def cube_decomposition_start: ℕ → ℕ
| 2 := 3
| 3 := 7
| _ := 0 -- not explicitly given

noncomputable def fourth_power_decomposition_start: ℕ := 7

theorem third_number_in_5th_power (n : ℕ) (h1 : 2 ≤ n) (h2 : n = 5): 
  ∃ k, (k = 3) → (5 ^ 4 = 121 + (121 + 2 * ( k - 1 ))) := by
  sorry

end third_number_in_5th_power_l242_242413


namespace company_allocation_salary_l242_242199

-- Definitions
def initial_salary : ℝ := 30000
def raise_rate : ℝ := 0.1
def tax_rate : ℝ := 0.13

-- Calculation of post-tax salary
def post_tax_salary : ℝ := initial_salary * (1 + raise_rate)

-- Formula to calculate the pre-tax salary from post-tax salary
def pre_tax_salary (post_tax : ℝ) (tax_rate : ℝ) : ℝ :=
  post_tax / (1 - tax_rate)

-- Theorem: proving the allocation amount
theorem company_allocation_salary : 
  pre_tax_salary post_tax_salary tax_rate = 37931 := 
by 
  sorry

end company_allocation_salary_l242_242199


namespace binom_30_3_eq_4060_l242_242365

theorem binom_30_3_eq_4060 : Nat.choose 30 3 = 4060 := by
  sorry

end binom_30_3_eq_4060_l242_242365


namespace large_bucket_capacity_l242_242695

variables (S L : ℝ)

theorem large_bucket_capacity (h1 : L = 2 * S + 3) (h2 : 2 * S + 5 * L = 63) : L = 11 :=
by sorry

end large_bucket_capacity_l242_242695


namespace ratio_B_C_l242_242602

def total_money := 595
def A_share := 420
def B_share := 105
def C_share := 70

-- The main theorem stating the expected ratio
theorem ratio_B_C : (B_share / C_share : ℚ) = 3 / 2 := by
  sorry

end ratio_B_C_l242_242602


namespace log_base_3_of_9_cubed_l242_242897

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 :=
by assumptions
  have h1 : 9 = 3^2 := sorry,
  have h2 : 9^3 = (3^2)^3 := sorry,
  have h3 : (3^2)^3 = 3^(2*3) := sorry,
  have h4 : 3^(2*3) = 3^6 := sorry,
  have h5 : log 3 (3^6) = 6 := by apply log_pow,
  sorry

end log_base_3_of_9_cubed_l242_242897


namespace log_base_3_of_9_cubed_l242_242870

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 := by
  -- Assuming the necessary properties of logarithms and exponents
  sorry

end log_base_3_of_9_cubed_l242_242870


namespace subsets_A_B_l242_242250

noncomputable def choices_of_subsets (C : set ℕ) (n : ℕ) : ℕ :=
  if h : C.card = n then 3^n else 0

theorem subsets_A_B (C : set ℕ) (n : ℕ) (h : C.card = n) :
  ∃ (A B : set ℕ), (A ⊆ C) ∧ (B ⊆ C) ∧ ((A ∩ B = ∅) ∨ (A ⊆ B)) → (choices_of_subsets C n) = 3^n := by
  sorry

end subsets_A_B_l242_242250


namespace terrell_sunday_hike_l242_242209

variables (total_distance : ℝ) (distance_saturday : ℝ)

-- Given conditions
def conditions : Prop :=
  total_distance = 9.8 ∧ distance_saturday = 8.2

-- Expected result
def distance_sunday : ℝ :=
  total_distance - distance_saturday

theorem terrell_sunday_hike (h : conditions total_distance distance_saturday) :
  distance_sunday total_distance distance_saturday = 1.6 :=
by
  cases h with ht hs
  dsimp [distance_sunday]
  rw [ht, hs]
  norm_num

end terrell_sunday_hike_l242_242209


namespace monotonically_decreasing_intervals_max_and_min_values_on_interval_l242_242461

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := -x^3 + 3 * x^2 + 9 * x + a

theorem monotonically_decreasing_intervals (a : ℝ) : 
  ∀ x : ℝ, (x < -1 ∨ x > 3) → f x a < f (x+1) a :=
sorry

theorem max_and_min_values_on_interval : 
  (f (-1) (-2) = -7) ∧ (max (f (-2) (-2)) (f 2 (-2)) = 20) :=
sorry

end monotonically_decreasing_intervals_max_and_min_values_on_interval_l242_242461


namespace cos_three_pi_over_two_l242_242400

theorem cos_three_pi_over_two : Real.cos (3 * Real.pi / 2) = 0 :=
by
  -- Provided as correct by the solution steps role
  sorry

end cos_three_pi_over_two_l242_242400


namespace imaginary_part_of_z_l242_242620

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Define the complex number z
def z : ℂ := (3 * i) / (1 + i)

-- State the theorem: the imaginary part of z is 3/2
theorem imaginary_part_of_z : z.im = 3 / 2 := by
  sorry

end imaginary_part_of_z_l242_242620


namespace x_cubed_plus_y_cubed_l242_242084

variable (x y : ℝ)
variable (h₁ : x + y = 5)
variable (h₂ : x^2 + y^2 = 17)

theorem x_cubed_plus_y_cubed :
  x^3 + y^3 = 65 :=
by sorry

end x_cubed_plus_y_cubed_l242_242084


namespace problem_l242_242162

variable {a b c x y z : ℝ}

theorem problem 
  (h1 : 5 * x + b * y + c * z = 0)
  (h2 : a * x + 7 * y + c * z = 0)
  (h3 : a * x + b * y + 9 * z = 0)
  (h4 : a ≠ 5)
  (h5 : x ≠ 0) :
  (a / (a - 5)) + (b / (b - 7)) + (c / (c - 9)) = 1 :=
by
  sorry

end problem_l242_242162


namespace binomial_expansion_properties_l242_242454

theorem binomial_expansion_properties (x : ℝ) : -- we use real numbers for simplicity.
  (∑ (i : ℕ) in Finset.range 9, (Nat.choose 8 i) * (-2)^i = 1) ∧
  (∃ (t : ℕ), t = 5 ∧ ∀ (k : ℕ), k ≠ 5 → Nat.choose 8 k * (-2)^k < Nat.choose 8 5 * (-2)^5) :=
by
  sorry

end binomial_expansion_properties_l242_242454


namespace number_of_boxes_initially_l242_242589

theorem number_of_boxes_initially (B : ℕ) (h1 : ∃ B, 8 * B - 17 = 15) : B = 4 :=
  by
  sorry

end number_of_boxes_initially_l242_242589


namespace range_of_a_for_extreme_value_l242_242121

-- Statement of the problem:
theorem range_of_a_for_extreme_value {a : ℝ} :
  (∃ x ∈ Ioo 1 2, (deriv (λ x : ℝ, x^3 + a * x^2 - x)) x = 0) →
  a ∈ Ioo (-11 / 4) (-1) :=
begin
  sorry
end

end range_of_a_for_extreme_value_l242_242121


namespace correct_sampling_methods_l242_242267

inductive SamplingMethod
| SimpleRandom
| Systematic
| Stratified

def scenario1 : (boxes : ℕ) → SamplingMethod :=
  fun boxes => if boxes = 10 then SamplingMethod.SimpleRandom else SamplingMethod.SimpleRandom

def scenario2 : (rows seats : ℕ) → SamplingMethod :=
  fun rows seats => if rows = 32 ∧ seats = 40 then SamplingMethod.Systematic else SamplingMethod.Systematic

def scenario3 : (total staff general admin logistics : ℕ) → SamplingMethod :=
  fun total staff general admin logistics =>
    if total = 160 ∧ staff = 120 ∧ admin = 16 ∧ logistics = 24 then
      SamplingMethod.Stratified
    else
      SamplingMethod.Stratified

theorem correct_sampling_methods :
  scenario1 10 = SamplingMethod.SimpleRandom ∧
  scenario2 32 40 = SamplingMethod.Systematic ∧
  scenario3 160 120 16 24 = SamplingMethod.Stratified :=
by
  -- proof omitted
  sorry

end correct_sampling_methods_l242_242267


namespace smallest_c_geometric_arithmetic_progression_l242_242545

theorem smallest_c_geometric_arithmetic_progression (a b c : ℕ) (h1 : a > b) (h2 : b > c) (h3 : 0 < c) 
(h4 : b ^ 2 = a * c) (h5 : a + b = 2 * c) : c = 1 :=
sorry

end smallest_c_geometric_arithmetic_progression_l242_242545


namespace actual_average_height_correct_l242_242669

theorem actual_average_height_correct : 
  (∃ (avg_height : ℚ), avg_height = 181 ) →
  (∃ (num_boys : ℕ), num_boys = 35) →
  (∃ (incorrect_height : ℚ), incorrect_height = 166) →
  (∃ (actual_height : ℚ), actual_height = 106) →
  (179.29 : ℚ) = 
    (round ((6315 + 106 : ℚ) / 35 * 100) / 100 ) :=
by
sorry

end actual_average_height_correct_l242_242669


namespace exists_transversal_l242_242721

open Real

noncomputable section

structure Line3D (p : Point) (d : Vector) : Type where
  point : p
  direction : d

def is_skew (ℓ1 ℓ2 : Line3D) : Prop :=
  -- Define the condition for two lines to be skew 
  ∀ t1 t2, ℓ1.point + t1 • ℓ1.direction ≠ ℓ2.point + t2 • ℓ2.direction
  ∧ ℓ1.direction ≠ ℓ2.direction

def is_plane (n : Vector) (p0 : Point) : Prop :=
  -- Define the condition for a vector and point to represent a plane
  true

def is_perpendicular (ℓ : Line3D) (n : Vector) : Prop :=
  -- Define the condition for a line and a vector to be perpendicular
  InnerProduct ℓ.direction n = 0

def intersects (ℓ : Line3D) (p : Point) : Prop :=
  -- Define the condition for a line to intersect a point
  ∃ t, ℓ.point + t • ℓ.direction = p

def find_transversal (ℓ1 ℓ2 : Line3D) (Π : Vector) : Line3D := sorry

theorem exists_transversal
  (ℓ1 ℓ2 : Line3D)
  (Π : Vector)
  (h_skew : is_skew ℓ1 ℓ2)
  (h_plane : is_plane Π ℓ1.point) :
  ∃ ℓ_t : Line3D, intersects ℓ_t ℓ1.point 
                  ∧ intersects ℓ_t ℓ2.point 
                  ∧ is_perpendicular ℓ_t Π := sorry

end exists_transversal_l242_242721


namespace log3_of_9_to_3_l242_242812

theorem log3_of_9_to_3 : Real.logb 3 (9^3) = 6 :=
by
  have h1 : 9 = 3^2 := by norm_num
  have h2 : 9^3 = (3^2)^3 := by rw [h1]
  have h3 : (3^2)^3 = 3^(2 * 3) := by rw [pow_mul]
  have h4 : Real.logb 3 (3^(2 * 3)) = 2 * 3 := Real.logb_pow 3 (2 * 3)
  rw [h4]
  norm_num

end log3_of_9_to_3_l242_242812


namespace gaokun_population_scientific_notation_l242_242314

theorem gaokun_population_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ (425000 = a * 10^n) ∧ (a = 4.25) ∧ (n = 5) :=
by
  sorry

end gaokun_population_scientific_notation_l242_242314


namespace log_base_3_l242_242857

theorem log_base_3 (log_identity : ∀ (b a c : ℝ), log b (a^c) = c * log b a) :
  log 3 (9^3) = 6 :=
by
  have : 9 = 3^2 := by sorry
  have : log 3 3 = 1 := by sorry
  exact (log_identity 3 (3^2) 3).trans sorry

end log_base_3_l242_242857


namespace Ray_wrote_35_l242_242601

theorem Ray_wrote_35 :
  ∃ (x y : ℕ), (10 * x + y = 35) ∧ (10 * x + y = 4 * (x + y) + 3) ∧ (10 * x + y + 18 = 10 * y + x) :=
by
  sorry

end Ray_wrote_35_l242_242601


namespace gcd_euclidean_algorithm_1813_333_l242_242646

theorem gcd_euclidean_algorithm_1813_333 :
  let (a, b) := (1813, 333) in
  Nat.gcd a b = 37 ∧
  ∃ n, n = 3 ∧ 
  (euclidean_algorithm_steps a b = n) :=
by
  sorry

end gcd_euclidean_algorithm_1813_333_l242_242646


namespace binomial_product_l242_242375

open Nat

theorem binomial_product : (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 := by
  sorry

end binomial_product_l242_242375


namespace log_base_3_of_9_cubed_l242_242926
-- Import the necessary parts of the standard library

-- Define the statement to be proved
theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 :=
by
  sorry

end log_base_3_of_9_cubed_l242_242926


namespace solution_set_for_fx_over_x_l242_242169

theorem solution_set_for_fx_over_x (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_f2 : f 2 = 0)
  (h_condition : ∀ x > 0, (x * f'' x - f x) / x^2 < 0) :
  {x : ℝ | f x / x > 0} = {x : ℝ | -2 < x ∧ x < 0} ∪ {x : ℝ | 0 < x ∧ x < 2} :=
by
  sorry

end solution_set_for_fx_over_x_l242_242169


namespace compare_sin_ratios_l242_242735

-- Define the problem conditions.
def sin_ratio_1 := sin (2014 * Real.pi / 180) / sin (2015 * Real.pi / 180)
def sin_ratio_2 := sin (2016 * Real.pi / 180) / sin (2017 * Real.pi / 180)

-- State the proof goal.
theorem compare_sin_ratios : sin_ratio_1 < sin_ratio_2 :=
by sorry

end compare_sin_ratios_l242_242735


namespace probability_sum_9_is_correct_l242_242658

def num_faces : ℕ := 6

def possible_outcomes : ℕ := num_faces * num_faces

def favorable_outcomes : ℕ := 4  -- (3,6), (6,3), (4,5), (5,4)

def probability_sum_9 : ℚ := favorable_outcomes / possible_outcomes

theorem probability_sum_9_is_correct :
  probability_sum_9 = 1/9 :=
sorry

end probability_sum_9_is_correct_l242_242658


namespace problem_statement_l242_242972

open Nat

theorem problem_statement (k : ℕ) (hk : k > 0) : 
  (∀ n : ℕ, n > 0 → 2^((k-1)*n+1) * (factorial (k*n) / factorial n) ≤ (factorial (k*n) / factorial n))
  ↔ ∃ a : ℕ, k = 2^a := 
sorry

end problem_statement_l242_242972


namespace seq1_odd_pos_seq2_odd_pos_seq3_even_pos_seq4_even_pos_l242_242330

def odd_pos_removed_same (s : ℕ → ℕ) : Prop :=
  ∀ n, s n = s (2 * n + 1)

def even_pos_removed_same (s : ℕ → ℕ) : Prop :=
  ∀ n, s n = s (2 * n)

-- Define sequences according to the problem:
-- Note: Lean sequences are typically represented as functions from ℕ to ℕ or lists.
def seq1 : ℕ → ℕ := λ n, if n % 3 = 2 then 0 else 1
def seq2 : ℕ → ℕ := λ n, if n % 5 = 4 then 0 else 1
def seq3 : ℕ → ℕ := λ n, if n % 3 = 1 then 0 else 1
def seq4 : ℕ → ℕ := λ n, if n % 6 = 4 then 0 else 1

theorem seq1_odd_pos : odd_pos_removed_same seq1 :=
by sorry

theorem seq2_odd_pos : odd_pos_removed_same seq2 :=
by sorry

theorem seq3_even_pos : even_pos_removed_same seq3 :=
by sorry

theorem seq4_even_pos : even_pos_removed_same seq4 :=
by sorry

end seq1_odd_pos_seq2_odd_pos_seq3_even_pos_seq4_even_pos_l242_242330


namespace total_pages_eq_95_l242_242346

-- Define the conditions as variables
variable (pages_first_day : ℕ) (pages_second_day : ℕ) (pages_left : ℕ)

-- Define the total number of pages in the book
def total_pages := pages_first_day + pages_second_day + pages_left

-- Given conditions
axiom pages_first_day_eq : pages_first_day = 18
axiom pages_second_day_eq : pages_second_day = 58
axiom pages_left_eq : pages_left = 19

-- Prove that the total number of pages is 95
theorem total_pages_eq_95 : total_pages pages_first_day pages_second_day pages_left = 95 :=
by
  rw [pages_first_day_eq, pages_second_day_eq, pages_left_eq]
  sorry

end total_pages_eq_95_l242_242346


namespace ellipse_and_slope_range_l242_242019

theorem ellipse_and_slope_range :
  ∃ (a b : ℝ) (E : set (ℝ × ℝ)), 
  (a > b ∧ b > 0) ∧
  E = {p | (p.fst ^ 2 / a^2) + (p.snd ^ 2 / b^2) = 1} ∧
  ∀ P Q B : ℝ × ℝ, 
    P = (0, -2) ∧ 
    B = (2, 0) ∧ 
    (Q.fst, Q.snd) = ((6 : ℝ) / 5, - (4 : ℝ) / 5) →
    2 • (P - Q) = 3 • (Q - B) ∧
    (triangle_is_isosceles_right (line_through A B) (line_through A P)) →
    a = 2 ∧
    b = 1 ∧
    E = {p | (p.fst^2 / 4) + p.snd^2 = 1} ∧
      ∀ l : set (ℝ × ℝ), 
      (∃ k : ℝ, l = {p | p.snd = k * p.fst - 2}) →
      ∃ M N: ℝ × ℝ, M ∈ E ∧ N ∈ E ∧ 
      let x1 := M.fst, x2 := N.fst in 
      O ∉ circle_with_diameter M N →
      (-2 < k ∧ k < - (Real.sqrt 3) / 2) ∨ (Real.sqrt 3 / 2 < k ∧ k < 2) :=
sorry

end ellipse_and_slope_range_l242_242019


namespace angle_between_vectors_l242_242063

open Real

def vector3 := (ℝ × ℝ × ℝ)

noncomputable def magnitude (v : vector3) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

noncomputable def dot_product (v₁ v₂ : vector3) : ℝ :=
  v₁.1 * v₂.1 + v₁.2 * v₂.2 + v₁.3 * v₂.3

noncomputable def cos_theta (v₁ v₂ : vector3) : ℝ :=
  dot_product v₁ v₂ / (magnitude v₁ * magnitude v₂)

theorem angle_between_vectors : 
  let a : vector3 := (-2, 2, 0)
  let b : vector3 := (1, 0, -1)
  cos_theta a b = - 1 / 2 → 
  acos (-1 / 2) = 2 * π / 3 :=
by
  let a : vector3 := (-2, 2, 0)
  let b : vector3 := (1, 0, -1)
  sorry

end angle_between_vectors_l242_242063


namespace _l242_242135

variable (ABC : Type) [triangle ABC]
variable {A B C O G H : point ABC}

-- Begin Lean definitions and statement
noncomputable def is_acute_angled (ABC : Type) := ∀ (A B C : point ABC), 
  ∠A + ∠B + ∠C = 180 ∧ ∠A < 90 ∧ ∠B < 90 ∧ ∠C < 90

noncomputable def circumcenter (ABC : Type) := O
noncomputable def centroid (ABC : Type) := G
noncomputable def orthocenter (ABC : Type) := H

noncomputable def euler_line (ABC : Type) := line G O

noncomputable def angle_60_degrees (ABC : Type) := ∠A C B = 60

noncomputable theorem equilateral_triangle_on_euler_line
  (h1 : is_acute_angled ABC)
  (h2 : angle_60_degrees ABC) :
  euler_line ABC = equilateral_triangle O G :=
sorry

end _l242_242135


namespace log3_of_9_to_3_l242_242809

theorem log3_of_9_to_3 : Real.logb 3 (9^3) = 6 :=
by
  have h1 : 9 = 3^2 := by norm_num
  have h2 : 9^3 = (3^2)^3 := by rw [h1]
  have h3 : (3^2)^3 = 3^(2 * 3) := by rw [pow_mul]
  have h4 : Real.logb 3 (3^(2 * 3)) = 2 * 3 := Real.logb_pow 3 (2 * 3)
  rw [h4]
  norm_num

end log3_of_9_to_3_l242_242809


namespace perfect_square_trinomial_l242_242078

theorem perfect_square_trinomial (m : ℝ) (h : ∃ a : ℝ, x^2 + 2 * x + m = (x + a)^2) : m = 1 := 
sorry

end perfect_square_trinomial_l242_242078


namespace log_base_three_of_nine_cubed_l242_242883

theorem log_base_three_of_nine_cubed : log 3 (9^3) = 6 :=
by
  have h1 : 9 = 3^2 := by sorry
  have h2 : ∀ (a m n: ℕ), (a^m)^n = a^(m * n) := by 
    intros a m n
    exact pow_mul a m n
  have h3 : ∀ (b n: ℕ), log b (b^n) = n := by sorry
  sorry

end log_base_three_of_nine_cubed_l242_242883


namespace birch_tree_count_l242_242512

theorem birch_tree_count:
  let total_trees := 8000
  let spruces := 0.12 * total_trees
  let pines := 0.15 * total_trees
  let maples := 0.18 * total_trees
  let cedars := 0.09 * total_trees
  let oaks := spruces + pines
  let calculated_trees := spruces + pines + maples + cedars + oaks
  let birches := total_trees - calculated_trees
  spruces = 960 → pines = 1200 → maples = 1440 → cedars = 720 → oaks = 2160 →
  birches = 1520 :=
by
  intros
  sorry

end birch_tree_count_l242_242512


namespace smartphones_discount_l242_242715

theorem smartphones_discount
  (discount : ℝ)
  (cost_per_iphone : ℝ)
  (total_saving : ℝ)
  (num_people : ℕ)
  (num_iphones : ℕ)
  (total_cost : ℝ)
  (required_num : ℕ) :
  discount = 0.05 →
  cost_per_iphone = 600 →
  total_saving = 90 →
  num_people = 3 →
  num_iphones = 3 →
  total_cost = num_iphones * cost_per_iphone →
  required_num = num_iphones →
  required_num * cost_per_iphone * discount = total_saving →
  required_num = 3 :=
by
  intros
  sorry

end smartphones_discount_l242_242715


namespace line_perpendicular_to_parallel_planes_l242_242998

variables (α β : Plane) (m : Line)

-- Given conditions:
-- α is parallel to β
axiom α_parallel_β : α ∥ β

-- m is perpendicular to α
axiom m_perpendicular_α : m ⟂ α

-- Question to prove: If α is parallel to β and m is perpendicular to α, then m is perpendicular to β
theorem line_perpendicular_to_parallel_planes 
  (α_parallel_β : α ∥ β) 
  (m_perpendicular_α : m ⟂ α) : 
  m ⟂ β := 
sorry

end line_perpendicular_to_parallel_planes_l242_242998


namespace log_base_3_of_9_cubed_l242_242825

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 := by
  sorry

end log_base_3_of_9_cubed_l242_242825


namespace fixed_points_distinct_l242_242555

def p1 (x : ℝ) : ℝ := x^2 - 2

def pn (n : ℕ) (x : ℝ) : ℝ :=
  nat.rec_on n (λ x, x) (λ n pn, p1 (pn x)) x

theorem fixed_points_distinct (n : ℕ) :
  ∀ x y : ℝ, pn n x = x → pn n y = y → x = y :=
by sorry

end fixed_points_distinct_l242_242555


namespace log_base_three_of_nine_cubed_l242_242875

theorem log_base_three_of_nine_cubed : log 3 (9^3) = 6 :=
by
  have h1 : 9 = 3^2 := by sorry
  have h2 : ∀ (a m n: ℕ), (a^m)^n = a^(m * n) := by 
    intros a m n
    exact pow_mul a m n
  have h3 : ∀ (b n: ℕ), log b (b^n) = n := by sorry
  sorry

end log_base_three_of_nine_cubed_l242_242875


namespace log_base_3_of_9_cubed_l242_242902

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 :=
by assumptions
  have h1 : 9 = 3^2 := sorry,
  have h2 : 9^3 = (3^2)^3 := sorry,
  have h3 : (3^2)^3 = 3^(2*3) := sorry,
  have h4 : 3^(2*3) = 3^6 := sorry,
  have h5 : log 3 (3^6) = 6 := by apply log_pow,
  sorry

end log_base_3_of_9_cubed_l242_242902


namespace min_trips_to_fill_hole_l242_242720

def hole_filling_trips (initial_gallons : ℕ) (required_gallons : ℕ) (capacity_2gallon : ℕ)
  (capacity_5gallon : ℕ) (capacity_8gallon : ℕ) (time_limit : ℕ) (time_per_trip : ℕ) : ℕ :=
  if initial_gallons < required_gallons then
    let remaining_gallons := required_gallons - initial_gallons
    let num_8gallon := remaining_gallons / capacity_8gallon
    let remaining_after_8gallon := remaining_gallons % capacity_8gallon
    let num_2gallon := if remaining_after_8gallon = 3 then 1 else 0
    let num_5gallon := if remaining_after_8gallon = 3 then 1 else remaining_after_8gallon / capacity_5gallon
    let total_trips := num_8gallon + num_2gallon + num_5gallon
    if total_trips <= time_limit / time_per_trip then
      total_trips
    else
      sorry -- If calculations overflow time limit
  else
    0

theorem min_trips_to_fill_hole : 
  hole_filling_trips 676 823 2 5 8 45 1 = 20 :=
by rfl

end min_trips_to_fill_hole_l242_242720


namespace evaluate_log_l242_242784

noncomputable def log_base (b x : ℝ) : ℝ := real.log x / real.log b

theorem evaluate_log : log_base 3 (9^3) = 6 := by
  have h1 : 9 = 3^2 := by norm_num
  have h2 : 9^3 = (3^2)^3 := by norm_num
  rw [h2, pow_mul]
  have h3 : log_base 3 (3^6) = 6 := by
    rw [← real.log_pow, real.log_mul, mul_comm, ← real.log_rpow, mul_div, real.log_self]
    norm_num
  exact h3

end evaluate_log_l242_242784


namespace series_value_l242_242748

theorem series_value :
  ∑ n in Finset.range 120, (-1) ^ n * (n^3 + (n - 1)^3) = 1728000 :=
by
  sorry

end series_value_l242_242748


namespace find_angle_C_find_perimeter_l242_242527

theorem find_angle_C (A B C : ℝ) (a b c : ℝ) (h1 : 2 * cos C * (a * cos B + b * cos A) = c) : 
  C = π / 3 := 
sorry

theorem find_perimeter (a b c : ℝ) (A B C : ℝ)
  (h1 : 2 * cos C * (a * cos B + b * cos A) = c)
  (h2 : c = sqrt 7)
  (h3 : 1/2 * a * b * sin C = 3 * sqrt 3 / 2) :
  a + b + c = 5 + sqrt 7 :=
sorry

end find_angle_C_find_perimeter_l242_242527


namespace unique_root_interval_l242_242049

noncomputable def f (x k : ℝ) : ℝ := x * Real.log x + x - k * (x - 1)

theorem unique_root_interval (k: ℝ) (n: ℤ) 
  (h1 : 1 < (∃ x : ℝ, x ∈ Set.Ioi 1 ∧ f x k = 0))
  (h2 : k ∈ Set.Ioo (n : ℝ) (n + 1))
  : n = 3 :=
by
  sorry

end unique_root_interval_l242_242049


namespace xiaoming_climb_stairs_five_steps_l242_242663

def count_ways_to_climb (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 1
  else count_ways_to_climb (n - 1) + count_ways_to_climb (n - 2)

theorem xiaoming_climb_stairs_five_steps :
  count_ways_to_climb 5 = 5 :=
by
  sorry

end xiaoming_climb_stairs_five_steps_l242_242663


namespace copper_percentage_l242_242349

theorem copper_percentage (copperFirst copperSecond totalWeight1 totalWeight2: ℝ) 
    (h1 : copperFirst = 0.25)
    (h2 : copperSecond = 0.50) 
    (h3 : totalWeight1 = 200) 
    (h4 : totalWeight2 = 800) : 
    (copperFirst * totalWeight1 + copperSecond * totalWeight2) / (totalWeight1 + totalWeight2) * 100 = 45 := 
by 
  sorry

end copper_percentage_l242_242349


namespace problem_1_problem_2_l242_242127

-- Define the given conditions
variables (a c : ℝ) (cosB : ℝ)
variables (b : ℝ) (S : ℝ)

-- Assuming the values for the variables
axiom h₁ : a = 4
axiom h₂ : c = 3
axiom h₃ : cosB = 1 / 8

-- Prove that b = sqrt(22)
theorem problem_1 : b = Real.sqrt 22 := by
  sorry

-- Prove that the area of triangle ABC is 9 * sqrt(7) / 4
theorem problem_2 : S = 9 * Real.sqrt 7 / 4 := by 
  sorry

end problem_1_problem_2_l242_242127


namespace problem_statement_l242_242993

theorem problem_statement (x y z : ℝ) (h : x^2 + y^2 + z^2 - 2*x + 4*y - 6*z + 14 = 0) : (x - y - z) ^ 2002 = 0 :=
sorry

end problem_statement_l242_242993


namespace lights_off_after_evaluation_l242_242247

/-!
There are 20 rooms. Initially, 10 rooms have their lights on, and 10 rooms have their lights off.
The light in the first room is on. Each person in a room will turn their light to match the majority state of the remaining rooms.
After everyone has had their turn, all 20 rooms will have their lights turned off.
-/

theorem lights_off_after_evaluation :
  ∀ (rooms : fin 20 → bool), 
  (∃ l : fin 20, ∑ i, if rooms i then 1 else 0 = 10 ∧ l = 0 ∧ rooms 0 = tt) →
  ∃ rooms' : fin 20 → bool, (∀ r, r < 20 → rooms' r = ff) :=
by 
  sorry


end lights_off_after_evaluation_l242_242247


namespace log_base_3_of_9_cubed_l242_242859

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 := by
  -- Assuming the necessary properties of logarithms and exponents
  sorry

end log_base_3_of_9_cubed_l242_242859


namespace xy_cubed_identity_l242_242097

namespace ProofProblem

theorem xy_cubed_identity (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 :=
by
  sorry

end ProofProblem

end xy_cubed_identity_l242_242097


namespace cubic_sum_identity_l242_242092

theorem cubic_sum_identity (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 :=
sorry

end cubic_sum_identity_l242_242092


namespace orchestra_total_people_l242_242197

def percussion_players := 3
def brass_players := 13
def strings_players := 18
def woodwinds_players := 10
def keyboards_and_harp_players := 2
def conductor := 1

theorem orchestra_total_people : 
  percussion_players + brass_players + strings_players + woodwinds_players + keyboards_and_harp_players + conductor = 47 := 
by
  sorry

end orchestra_total_people_l242_242197


namespace log_base_3_of_9_cubed_l242_242841

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 := by
  -- Define the known conditions
  have h1 : 9 = 3^2 := by norm_num
  have h2 : log 3 (a^n) = n * log 3 a := sorry  -- Statement of the logarithmic identity
  
  -- Apply the conditions to prove the result
  calc
    log 3 (9^3)
        = log 3 ((3^2)^3) : by rw h1
    ... = log 3 (3^(2 * 3)) : by congr; norm_num
    ... = log 3 (3^6) : by norm_num
    ... = 6 : by rw [h2]; norm_num

end log_base_3_of_9_cubed_l242_242841


namespace area_ratio_is_four_l242_242560

-- Definitions based on the given conditions
variables (k a b c d : ℝ)
variables (ka kb kc kd : ℝ)

-- Equations from the conditions
def eq1 : a = k * ka := sorry
def eq2 : b = k * kb := sorry
def eq3 : c = k * kc := sorry
def eq4 : d = k * kd := sorry

-- Ratios provided in the problem
def ratio1 : ka / kc = 2 / 5 := sorry
def ratio2 : kb / kd = 2 / 5 := sorry

-- The theorem to prove the ratio of areas is 4:1
theorem area_ratio_is_four : (k * ka * k * kb) / (k * kc * k * kd) = 4 :=
by sorry

end area_ratio_is_four_l242_242560


namespace log_three_pow_nine_pow_three_eq_six_l242_242946

open Real

noncomputable def eval_log_three_pow_nine_pow_three : ℝ :=
  log 3 (9^3)

theorem log_three_pow_nine_pow_three_eq_six :
  eval_log_three_pow_nine_pow_three = 6 := by
  sorry

end log_three_pow_nine_pow_three_eq_six_l242_242946


namespace sum_even_integers_between_200_and_400_l242_242289

theorem sum_even_integers_between_200_and_400 : 
  (Finset.sum (Finset.filter (λ x, x % 2 = 0) (Finset.range 401)) 
    - Finset.sum (Finset.filter (λ x, x % 2 = 0) (Finset.range 201)))  = 30100 :=
begin
  sorry
end

end sum_even_integers_between_200_and_400_l242_242289


namespace binom_computation_l242_242377

noncomputable def binom : ℕ → ℕ → ℕ
| n, 0       => 1
| 0, k+1     => 0
| n+1, k+1   => binom n k + binom n (k+1)

theorem binom_computation :
  (binom 10 3) * (binom 8 3) = 6720 := by
  sorry

end binom_computation_l242_242377


namespace parabola_focus_coordinates_l242_242059

open Real

theorem parabola_focus_coordinates (x y : ℝ) (h : y^2 = 6 * x) : (x, y) = (3 / 2, 0) :=
  sorry

end parabola_focus_coordinates_l242_242059


namespace log_pow_evaluation_l242_242905

-- Define the condition that 9 is equal to 3^2
def nine_eq_three_squared : Prop := 9 = 3^2

-- The main theorem that we need to prove
theorem log_pow_evaluation (h : nine_eq_three_squared) : log 3 (9^3) = 6 :=
  sorry

end log_pow_evaluation_l242_242905


namespace log_three_pow_nine_pow_three_eq_six_l242_242935

open Real

noncomputable def eval_log_three_pow_nine_pow_three : ℝ :=
  log 3 (9^3)

theorem log_three_pow_nine_pow_three_eq_six :
  eval_log_three_pow_nine_pow_three = 6 := by
  sorry

end log_three_pow_nine_pow_three_eq_six_l242_242935


namespace fish_per_white_duck_l242_242984

theorem fish_per_white_duck 
  (white_ducks : ℕ) (black_ducks : ℕ) (multicolor_ducks : ℕ) (total_fish : ℕ)
  (white_ducks_fish : ∀ w, w = 3 → ∃ x, total_fish = 3 * x + 7 * 10 + 6 * 12) : 
  ∃ x, total_fish = 3 * x + 70 + 72 → x = 5 :=
by
  intros h1 h2
  have hfish : total_fish = 3 * 5 + 70 + 72 := by
    sorry
  exact ⟨5, hfish⟩
sorry

end fish_per_white_duck_l242_242984


namespace find_length_PQ_l242_242152

-- Definitions used as conditions in the problem
variables (P Q R X Y Z : Type)
variables [MetricSpace P] [MetricSpace Q] [MetricSpace R] [MetricSpace X] [MetricSpace Y] [MetricSpace Z]

-- Notation for points and lengths
def angle_PQR_90 (P Q R : P) : Prop := true -- given condition
def length_QR : ℝ := 8 -- given condition
def least_XZ (X Z : X) : ℝ := 1.6 -- given condition

-- Right-angled triangle definition
def right_angle_triangle (P Q R : P) : Prop := angle_PQR_90 P Q R

-- Main theorem statement
theorem find_length_PQ (P Q R : P) 
  (h1 : angle_PQR_90 P Q R) -- Q is a 90-degree angle.
  (h2 : dist Q R = length_QR) -- QR is 8 cm.
  (h3 : true) -- X is a variable point on PQ.
  (h4 : true) -- The line through X parallel to QR intersects PR at Y.
  (h5 : true) -- The line through Y parallel to PQ intersects QR at Z.
  (h6 : dist X Z = least_XZ X Z) -- The least possible length of XZ is 1.6 cm.
  : dist P Q = 4 := 
  sorry -- Proof to be completed

end find_length_PQ_l242_242152


namespace valid_relationship_l242_242420

noncomputable def proof_statement (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^2 + c^2 = 2 * b * c) : Prop :=
  b > a ∧ a > c

theorem valid_relationship (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^2 + c^2 = 2 * b * c) : proof_statement a b c h_distinct h_pos h_eq :=
  sorry

end valid_relationship_l242_242420


namespace evaluate_log_l242_242794

noncomputable def log_base (b x : ℝ) : ℝ := real.log x / real.log b

theorem evaluate_log : log_base 3 (9^3) = 6 := by
  have h1 : 9 = 3^2 := by norm_num
  have h2 : 9^3 = (3^2)^3 := by norm_num
  rw [h2, pow_mul]
  have h3 : log_base 3 (3^6) = 6 := by
    rw [← real.log_pow, real.log_mul, mul_comm, ← real.log_rpow, mul_div, real.log_self]
    norm_num
  exact h3

end evaluate_log_l242_242794


namespace log_evaluation_l242_242768

theorem log_evaluation : ∀ (a : ℕ), a = 3 → log 3 (9^3) = 6 :=
by
  intro a ha
  have h1 : 9 = 3^2 := by sorry
  have h2 : (9^3) = (3^2)^3 := by rw [h1]
  have h3 : (3^2)^3 = 3^6 := by sorry
  have h4 : log 3 (3^6) = 6 := by sorry
  rw [h2, h3, h4]
  exact ha

end log_evaluation_l242_242768


namespace first_term_exceeds_10000_l242_242220

-- The sequence is defined such that
-- a_1 = 2
-- a_n = sum of all previous terms for n > 1

noncomputable def seq : ℕ → ℕ
| 0     => 2
| (n+1) => ∑ i in Finset.range (n+1), seq i

-- Prove first term that exceeds 10000 is 16384
theorem first_term_exceeds_10000 : ∃ n, seq n > 10000 ∧ seq n = 16384 := by
  sorry

-- Additional helper lemma for the geometric progression relation
lemma seq_geometric : ∀ n, n ≥ 1 → seq (n+1) = 2^(n - 1) := by
  sorry

end first_term_exceeds_10000_l242_242220


namespace gcd_fact8_fact10_l242_242271

-- Define the factorials
def fact : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * fact n

-- State the problem conditions
theorem gcd_fact8_fact10 : gcd (fact 8) (fact 10) = fact 8 := by
  sorry

end gcd_fact8_fact10_l242_242271


namespace find_coordinates_condition1_find_coordinates_condition2_find_coordinates_condition3_l242_242018

-- Define the coordinate functions for point P
def coord_x (m : ℚ) : ℚ := 3 * m + 6
def coord_y (m : ℚ) : ℚ := m - 3

-- Definitions for each condition
def condition1 (m : ℚ) : Prop := coord_x m = coord_y m
def condition2 (m : ℚ) : Prop := coord_y m = coord_x m + 5
def condition3 (m : ℚ) : Prop := coord_x m = 3

-- Proof statements for the coordinates based on each condition
theorem find_coordinates_condition1 : 
  ∃ m, condition1 m ∧ coord_x m = -7.5 ∧ coord_y m = -7.5 :=
by sorry

theorem find_coordinates_condition2 : 
  ∃ m, condition2 m ∧ coord_x m = -15 ∧ coord_y m = -10 :=
by sorry

theorem find_coordinates_condition3 : 
  ∃ m, condition3 m ∧ coord_x m = 3 ∧ coord_y m = -4 :=
by sorry

end find_coordinates_condition1_find_coordinates_condition2_find_coordinates_condition3_l242_242018


namespace calculate_rhombus_area_l242_242263

def rhombus_adj_sides_eq4_angle_eq_45_area : Prop :=
  ∀ (A B C D : ℝ), 
  ∃ (AB CD : ℝ) (angle_Dab : ℝ) (area : ℝ), 
  AB = 4 ∧ CD = 4 ∧ angle_Dab = 45 * (π / 180) ∧ ( area = 8 * √2 )

theorem calculate_rhombus_area :
  rhombus_adj_sides_eq4_angle_eq_45_area :=
by
  sorry

end calculate_rhombus_area_l242_242263


namespace no_integer_factors_l242_242709

def is_prime (p : ℕ) : Prop := Nat.Prime p

def has_decimal_digits (p : ℕ) (digits : List ℕ) : Prop :=
  p = digits.foldr (λ (c : ℕ) (acc : ℕ), c + acc * 10) 0

def P (digits : List ℕ) (x : ℤ) : ℤ :=
  digits.enum.reverse.foldl (λ acc (i_c : ℕ × ℕ), acc + i_c.snd * x^i_c.fst) 0

theorem no_integer_factors (p : ℕ) (digits : List ℕ) (n : ℕ) (hn : n > 0) (hn_digits : digits.length = n + 1)
  (hp : is_prime p) (hdigits : has_decimal_digits p digits) (hpositivedigits : digits.last (by sorry) > 1) :
  ¬ ∃ (f g : ℤ → ℤ), f.degree > 0 ∧ g.degree > 0 ∧ f.degree + g.degree = n ∧ (∀ m, f.coeff m ∈ ℤ) ∧ (∀ m, g.coeff m ∈ ℤ) ∧ P digits = f * g := 
sorry

end no_integer_factors_l242_242709


namespace ratio_AG_GC_l242_242450

-- Define the areas of the given shapes as constants
noncomputable def area_ABCD : ℝ := 48  -- Area of rectangle ABCD in cm^2
noncomputable def area_ADF : ℝ := 8    -- Area of triangle ADF in cm^2
noncomputable def area_ABE : ℝ := 9    -- Area of triangle ABE in cm^2

-- Define the theorem to prove the ratio AG:GC
theorem ratio_AG_GC : AG : GC = 21 : 10 :=
by
  sorry

end ratio_AG_GC_l242_242450


namespace log_base_3_of_9_cubed_l242_242838

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 := by
  -- Define the known conditions
  have h1 : 9 = 3^2 := by norm_num
  have h2 : log 3 (a^n) = n * log 3 a := sorry  -- Statement of the logarithmic identity
  
  -- Apply the conditions to prove the result
  calc
    log 3 (9^3)
        = log 3 ((3^2)^3) : by rw h1
    ... = log 3 (3^(2 * 3)) : by congr; norm_num
    ... = log 3 (3^6) : by norm_num
    ... = 6 : by rw [h2]; norm_num

end log_base_3_of_9_cubed_l242_242838


namespace find_k_l242_242970

open Nat

def S (n : ℕ) : ℕ :=
  Integer.toNat $ (n.toBinary).count 1 -- toBinary converts n to its binary representation and count counts 1's in that binary list

def v (n : ℕ) : ℕ :=
  n - S n

theorem find_k (k : ℕ) : (∀ n : ℕ, n > 0 → 2 ^ ((k - 1) * n + 1) ∣ factorial (k * n) / factorial n) ↔ (∃ m : ℕ, k = 2 ^ m) :=
by
  sorry

end find_k_l242_242970


namespace log_base_3_of_9_cubed_l242_242771

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 :=
by
  -- Given conditions:
  -- 1. 9 = 3^2
  -- 2. log_b (a^n) = n * log_b a
  sorry

end log_base_3_of_9_cubed_l242_242771


namespace log_base_3_of_9_cubed_l242_242892

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 :=
by assumptions
  have h1 : 9 = 3^2 := sorry,
  have h2 : 9^3 = (3^2)^3 := sorry,
  have h3 : (3^2)^3 = 3^(2*3) := sorry,
  have h4 : 3^(2*3) = 3^6 := sorry,
  have h5 : log 3 (3^6) = 6 := by apply log_pow,
  sorry

end log_base_3_of_9_cubed_l242_242892


namespace find_s_l242_242402

theorem find_s (s : ℝ) (h : 4 * log 3 s = log 3 (4 * s^2)) : s = 2 := 
sorry

end find_s_l242_242402


namespace area_of_triangle_arith_seq_l242_242037

theorem area_of_triangle_arith_seq (A B C : Type) [Real HilbertSpace A] [InnerProductSpace ℝ A]
  (angle_ABC : angle B A C = real.angle.of_real 120)
  (a b c : ℝ)
  (h_seq : ∃ k, a = k - 2 ∧ b = k ∧ c = k + 2) :
  area (triangle.mk A B C) = real.sqrt 3 * 15 / 4 := 
sorry

end area_of_triangle_arith_seq_l242_242037


namespace class_mean_l242_242510

theorem class_mean :
  (let n1 := 32 in
   let s1 := 68 in
   let n2 := 8 in
   let s2 := 82 in
   let total_students := n1 + n2 in
   let total_score := (n1 * s1) + (n2 * s2) in
   let mean_score := total_score / total_students in
   mean_score / 1 == 70.8) := sorry

end class_mean_l242_242510


namespace common_area_is_216_l242_242264

-- Define the structure of a 30-60-90 triangle
structure Triangle306090 :=
  (hypotenuse : ℝ)
  (leg_short : ℝ := hypotenuse / 2)
  (leg_long : ℝ := (hypotenuse / 2) * real.sqrt 3)

-- Define the problem conditions
def congruentTriangles := Triangle306090.mk 24

-- The problem statement
theorem common_area_is_216 (t1 t2 : Triangle306090)
  (h_condition1 : t1.hypotenuse = t2.hypotenuse)
  (h_condition2 : t1.hypotenuse = 24)
  (h_condition3 : t1.leg_short = t2.leg_short)
  (h_condition4 : t1.leg_long = t2.leg_long)
  (rotated_30_degrees : ∀ t1 t2 : Triangle306090, t1.hypotenuse = t2.hypotenuse) :
  -- Prove that the area common to both triangles is 216
  (let commonBase := (congruentTriangles.leg_long * real.cos (real.pi / 6))) in
  (let commonHeight := congruentTriangles.leg_long) in
  (commonBase * commonHeight = 216) :=
sorry

end common_area_is_216_l242_242264


namespace log_base_3_of_9_cubed_l242_242814

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 := by
  sorry

end log_base_3_of_9_cubed_l242_242814


namespace total_nominal_income_l242_242579

theorem total_nominal_income :
  let principal := 8700
  let rate := 0.06 / 12
  let income (n : ℕ) := principal * ((1 + rate) ^ n - 1)
  income 6 + income 5 + income 4 + income 3 + income 2 + income 1 = 921.15 := by
  sorry

end total_nominal_income_l242_242579


namespace log_base_3_l242_242856

theorem log_base_3 (log_identity : ∀ (b a c : ℝ), log b (a^c) = c * log b a) :
  log 3 (9^3) = 6 :=
by
  have : 9 = 3^2 := by sorry
  have : log 3 3 = 1 := by sorry
  exact (log_identity 3 (3^2) 3).trans sorry

end log_base_3_l242_242856


namespace evaluate_log_l242_242788

noncomputable def log_base (b x : ℝ) : ℝ := real.log x / real.log b

theorem evaluate_log : log_base 3 (9^3) = 6 := by
  have h1 : 9 = 3^2 := by norm_num
  have h2 : 9^3 = (3^2)^3 := by norm_num
  rw [h2, pow_mul]
  have h3 : log_base 3 (3^6) = 6 := by
    rw [← real.log_pow, real.log_mul, mul_comm, ← real.log_rpow, mul_div, real.log_self]
    norm_num
  exact h3

end evaluate_log_l242_242788


namespace evaluate_log_l242_242796

noncomputable def log_base (b x : ℝ) : ℝ := real.log x / real.log b

theorem evaluate_log : log_base 3 (9^3) = 6 := by
  have h1 : 9 = 3^2 := by norm_num
  have h2 : 9^3 = (3^2)^3 := by norm_num
  rw [h2, pow_mul]
  have h3 : log_base 3 (3^6) = 6 := by
    rw [← real.log_pow, real.log_mul, mul_comm, ← real.log_rpow, mul_div, real.log_self]
    norm_num
  exact h3

end evaluate_log_l242_242796


namespace totalNominalIncomeIsCorrect_l242_242581

def nominalIncomeForMonth (principal rate divisor months : ℝ) : ℝ :=
  principal * ((1 + rate / divisor) ^ months - 1)

def totalNominalIncomeForSixMonths : ℝ :=
  nominalIncomeForMonth 8700 0.06 12 6 +
  nominalIncomeForMonth 8700 0.06 12 5 +
  nominalIncomeForMonth 8700 0.06 12 4 +
  nominalIncomeForMonth 8700 0.06 12 3 +
  nominalIncomeForMonth 8700 0.06 12 2 +
  nominalIncomeForMonth 8700 0.06 12 1

theorem totalNominalIncomeIsCorrect : totalNominalIncomeForSixMonths = 921.15 := by
  sorry

end totalNominalIncomeIsCorrect_l242_242581


namespace power_function_value_at_quarter_l242_242618

noncomputable def f (x : ℝ) (α : ℝ) : ℝ := x^α

theorem power_function_value_at_quarter (α : ℝ) (h : f 4 α = 1 / 2) : f (1 / 4) α = 2 := 
  sorry

end power_function_value_at_quarter_l242_242618


namespace log_three_nine_cubed_l242_242950

theorem log_three_nine_cubed : ∀ (log : ℕ → ℕ → ℕ), log 3 9 = 2 → log 3 3 = 1 → (∀ (a b n : ℕ), log a (b ^ n) = n * log a b) → log 3 (9 ^ 3) = 6 :=
by
  intros log h1 h2 h3
  sorry

end log_three_nine_cubed_l242_242950


namespace complex_quadrant_z_l242_242559

noncomputable def quadrant (z : Complex) : String :=
  if z.re > 0 ∧ z.im > 0 then "first quadrant"
  else if z.re < 0 ∧ z.im > 0 then "second quadrant"
  else if z.re < 0 ∧ z.im < 0 then "third quadrant"
  else if z.re > 0 ∧ z.im < 0 then "fourth quadrant"
  else "axis"

theorem complex_quadrant_z (z : Complex) (h : z * (1 + Complex.i) = Complex.abs (⟨√3, -1⟩)) :
  quadrant z = "fourth quadrant" :=
sorry

end complex_quadrant_z_l242_242559


namespace binom_30_3_eq_4060_l242_242364

theorem binom_30_3_eq_4060 : Nat.choose 30 3 = 4060 := by
  sorry

end binom_30_3_eq_4060_l242_242364


namespace solution_set_of_b_inequality_l242_242043

-- Definitions
def sol_set_ineq1 (a b : ℝ) : set ℝ := {x : ℝ | -3 < x ∧ x < -2}

def quadratic_eq_roots (a b : ℝ) (x1 x2 : ℝ) : Prop :=
  x1 + x2 = (5 / a) ∧ x1 * x2 = (b / a)

-- Condition
axiom sol_set_condition (a b : ℝ) : sol_set_ineq1 a b = {x : ℝ | -3 < x ∧ x < -2}

-- Question as a Proof Problem
theorem solution_set_of_b_inequality (a b : ℝ) (h : sol_set_condition a b) :
  {x : ℝ | bx^2 - 5x + a < 0} = {x : ℝ | x < -1 / 2 ∨ x > -1 / 3} :=
sorry

end solution_set_of_b_inequality_l242_242043


namespace find_abs_ab_ac_bc_l242_242737

noncomputable def a : ℂ := sorry
noncomputable def b : ℂ := sorry
noncomputable def c : ℂ := sorry

axiom equilateral_triangle (a b c : ℂ) : (a - b).abs = 24 ∧ (b - c).abs = 24 ∧ (c - a).abs = 24
axiom sum_of_complex (a b c : ℂ) : (a + b + c).abs = 48

theorem find_abs_ab_ac_bc : |a * b + a * c + b * c| = 768 :=
by 
  have h1 := equilateral_triangle a b c,
  have h2 := sum_of_complex a b c,
  sorry

end find_abs_ab_ac_bc_l242_242737


namespace x_cubed_plus_y_cubed_l242_242080

variable (x y : ℝ)
variable (h₁ : x + y = 5)
variable (h₂ : x^2 + y^2 = 17)

theorem x_cubed_plus_y_cubed :
  x^3 + y^3 = 65 :=
by sorry

end x_cubed_plus_y_cubed_l242_242080


namespace proof_l242_242011

def arithmetic_sequence (a b d : ℕ → ℕ) := 
  (a 3 = 7) ∧ 
  (∀ n, b n = a n * (d n + 1))

def find_sequences (a b d : ℕ → ℕ) :=
  (∀ n, a n = 2 * n + 1) ∧ 
  (∀ n, (S : ℕ → ℕ), S n = n^2 + 2 * n) ∧
  (∀ n, b n = (\sum k in finset.range n, (1 / k) - 1 / (k + 1)) / (n + 1))

theorem proof (a b d : ℕ → ℕ) : arithmetic_sequence a b d → find_sequences a b d :=
by
  intros h₁ h₂,
  sorry

end proof_l242_242011


namespace fencing_three_sides_l242_242335

noncomputable def fencing_required (L W : ℕ) (A : ℕ) : ℕ :=
  if L * W = A then L + 2 * W else 0

theorem fencing_three_sides (L W A : ℕ) (hL : L = 20) (hA : A = 120) (hAW : A = L * W) : 
  fencing_required L W A = 32 :=
by
  rw [hL, hA, fencing_required]
  simp only [Nat.mul_eq_mul, Nat.add_mul]
  clear hAW
  ring_nf
  split_ifs
  · exact rfl
  · contradiction

end fencing_three_sides_l242_242335


namespace sum_even_integers_between_200_and_400_l242_242286

theorem sum_even_integers_between_200_and_400 : 
  (Finset.sum (Finset.filter (λ x, x % 2 = 0) (Finset.range 401)) 
    - Finset.sum (Finset.filter (λ x, x % 2 = 0) (Finset.range 201)))  = 30100 :=
begin
  sorry
end

end sum_even_integers_between_200_and_400_l242_242286


namespace expressions_even_odd_l242_242590

theorem expressions_even_odd (a b : ℤ) : 
  (∃ x y, x = a - b ∧ y = a + b + 1 ∧ (even x ∧ odd y) ∨ (odd x ∧ even y)) :=
by
  sorry

end expressions_even_odd_l242_242590


namespace a_2023_eq_neg2_l242_242492

def seq : ℕ → ℤ
| 0     := 2
| (n+1) := -|seq n + 5|

theorem a_2023_eq_neg2 : seq 2022 = -2 :=
  sorry

end a_2023_eq_neg2_l242_242492


namespace log_base_3_of_9_cubed_l242_242815

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 := by
  sorry

end log_base_3_of_9_cubed_l242_242815


namespace total_nominal_income_l242_242571

theorem total_nominal_income
  (c1 : 8700 * ((1 + 0.06 / 12) ^ 6 - 1) = 264.28)
  (c2 : 8700 * ((1 + 0.06 / 12) ^ 5 - 1) = 219.69)
  (c3 : 8700 * ((1 + 0.06 / 12) ^ 4 - 1) = 175.31)
  (c4 : 8700 * ((1 + 0.06 / 12) ^ 3 - 1) = 131.15)
  (c5 : 8700 * ((1 + 0.06 / 12) ^ 2 - 1) = 87.22)
  (c6 : 8700 * (1 + 0.06 / 12 - 1) = 43.5) :
  264.28 + 219.69 + 175.31 + 131.15 + 87.22 + 43.5 = 921.15 := by
  sorry

end total_nominal_income_l242_242571


namespace championship_outcomes_l242_242979

theorem championship_outcomes (students events : ℕ) (hs : students = 5) (he : events = 3) :
  ∃ outcomes : ℕ, outcomes = 5 ^ 3 := by
  sorry

end championship_outcomes_l242_242979


namespace minimum_value_of_distance_l242_242996

noncomputable def minimum_distance : ℂ := Complex.norm (1 - Complex.I)

theorem minimum_value_of_distance (z : ℂ) (h : Complex.abs (z - 1) = Complex.abs (z + 2 * Complex.I)) :
  minimum_distance = (9 * Real.sqrt 5) / 10 :=
sorry

end minimum_value_of_distance_l242_242996


namespace focus_of_parabola_l242_242614

theorem focus_of_parabola : ∀ x : ℝ, ∀ y : ℝ, (y = (1 / 4) * x^2) → (0, 1) = (focus_of_parabola y) :=
by
  sorry
  
-- Helper function to define focus of a parabola for the given y
def focus_of_parabola (y : ℝ) := (0, 1)

end focus_of_parabola_l242_242614


namespace log_base_3_of_9_cubed_l242_242776

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 :=
by
  -- Given conditions:
  -- 1. 9 = 3^2
  -- 2. log_b (a^n) = n * log_b a
  sorry

end log_base_3_of_9_cubed_l242_242776


namespace relationship_between_y_values_l242_242234

-- Define the quadratic function given the constraints
def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + abs b * x + c

-- Define the points (x1, y1), (x2, y2), (x3, y3)
def x1 := -14 / 3
def x2 := 5 / 2
def x3 := 3

def y1 (a b c : ℝ) : ℝ := quadratic_function a b c x1
def y2 (a b c : ℝ) : ℝ := quadratic_function a b c x2
def y3 (a b c : ℝ) : ℝ := quadratic_function a b c x3

theorem relationship_between_y_values 
  (a b c : ℝ) (h1 : - (abs b) / (2 * a) = -1) 
  (y1_value : ℝ := y1 a b c) 
  (y2_value : ℝ := y2 a b c) 
  (y3_value : ℝ := y3 a b c) : 
  y2_value < y1_value ∧ y1_value < y3_value := 
by 
  sorry

end relationship_between_y_values_l242_242234


namespace sum_of_painted_sides_l242_242336

theorem sum_of_painted_sides (L W : ℕ) (hL : L = 99) (hA : L * W = 126) : L + 2 * (126 / 99) ≈ 101.55 := 
by
  sorry

end sum_of_painted_sides_l242_242336


namespace xy_cubed_identity_l242_242102

namespace ProofProblem

theorem xy_cubed_identity (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 :=
by
  sorry

end ProofProblem

end xy_cubed_identity_l242_242102


namespace parabola_directrix_l242_242056

theorem parabola_directrix (p : ℝ) (h_focus : ∃ x y : ℝ, y^2 = 2*p*x ∧ 2*x + 3*y - 4 = 0) : 
  ∀ x y : ℝ, y^2 = 2*p*x → x = -p/2 := 
sorry

end parabola_directrix_l242_242056


namespace pasta_ratio_l242_242677

theorem pasta_ratio (students_surveyed : ℕ) (spaghetti_preferred : ℕ) (manicotti_preferred : ℕ) 
(h_total : students_surveyed = 800) 
(h_spaghetti : spaghetti_preferred = 320) 
(h_manicotti : manicotti_preferred = 160) : 
(spaghetti_preferred / manicotti_preferred : ℚ) = 2 := by
  sorry

end pasta_ratio_l242_242677


namespace valbonne_middle_school_l242_242609

theorem valbonne_middle_school (students : Finset ℕ) (h : students.card = 367) :
  ∃ (date1 date2 : ℕ), date1 ≠ date2 ∧ date1 = date2 ∧ date1 ∈ students ∧ date2 ∈ students :=
by {
  sorry
}

end valbonne_middle_school_l242_242609


namespace cube_identity_l242_242110

theorem cube_identity (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 := by
  sorry

end cube_identity_l242_242110


namespace log_base_3_l242_242858

theorem log_base_3 (log_identity : ∀ (b a c : ℝ), log b (a^c) = c * log b a) :
  log 3 (9^3) = 6 :=
by
  have : 9 = 3^2 := by sorry
  have : log 3 3 = 1 := by sorry
  exact (log_identity 3 (3^2) 3).trans sorry

end log_base_3_l242_242858


namespace sum_a4_a5_a6_l242_242141

variable (a : ℕ → ℤ)
variable (d : ℤ)

-- Conditions
axiom h1 : a 1 = 2
axiom h2 : a 3 = -10

-- Definition of arithmetic sequence
axiom h_arith_seq : ∀ n : ℕ, a (n + 1) = a n + d

-- Proof problem statement
theorem sum_a4_a5_a6 : a 4 + a 5 + a 6 = -66 :=
by
  sorry

end sum_a4_a5_a6_l242_242141


namespace slope_angle_OA_l242_242438

-- Defining points O and A
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (1, 1)

-- Definition of the slope angle
def slope_angle (P Q : ℝ × ℝ) : ℝ := Real.arctan ((Q.2 - P.2) / (Q.1 - P.1))

-- Statement to prove
theorem slope_angle_OA : slope_angle O A = Real.pi / 4 := by
  sorry

end slope_angle_OA_l242_242438


namespace discount_percentage_l242_242585

theorem discount_percentage (original_price sale_price : ℝ) (h₁ : original_price = 128) (h₂ : sale_price = 83.2) :
  (original_price - sale_price) / original_price * 100 = 35 :=
by
  sorry

end discount_percentage_l242_242585


namespace first_generation_l242_242307

-- Definition for each generation multiplication factor
def split_factor : ℕ := 8
def survival_rate : ℕ := 50
def effective_factor : ℕ := split_factor * survival_rate / 100

-- Condition for the seventh generation number
def G7 : ℕ := 4096 * 10^6

-- Define the recursive relation for generations
def generation (n : ℕ) : ℕ :=
  if n = 7 then G7
  else generation (n + 1) / effective_factor

-- Theorem to prove the first generation number
theorem first_generation : generation 1 = 1 * 10^6 :=
by
  sorry

end first_generation_l242_242307


namespace find_value_l242_242004

noncomputable def imaginary_unit := Complex.I

theorem find_value (x y : ℝ) (h1 : (x - 2) * imaginary_unit - y = -1 + imaginary_unit) :
  (1 - imaginary_unit)^(x + y) = -4 := 
by
  -- Proof will be provided here
  sorry

end find_value_l242_242004


namespace spring_hamburger_sales_l242_242317

theorem spring_hamburger_sales
  (winter_sales : ℕ)
  (summer_sales : ℕ)
  (fall_sales : ℕ)
  (total_sales : ℕ)
  (winter_percent : ℕ)
  (H : winter_sales = (winter_percent * total_sales) / 100)
  (W : winter_sales = 3)
  (S : summer_sales = 6)
  (F : fall_sales = 4)
  (T : total_sales = 15) :
  ∃ (spring_sales : ℕ), spring_sales = 2 :=
by
  let spring_sales := total_sales - (winter_sales + summer_sales + fall_sales)
  have : spring_sales = 2 := by
    sorry
  use spring_sales
  exact this

end spring_hamburger_sales_l242_242317


namespace sum_of_decimals_l242_242359

theorem sum_of_decimals : 5.46 + 2.793 + 3.1 = 11.353 := by
  sorry

end sum_of_decimals_l242_242359


namespace find_m_l242_242436

-- Define vector a
def a : ℝ × ℝ := (4, 3)

-- Define vector b with an unknown second component m
def b (m : ℝ) : ℝ × ℝ := (6, m)

-- Define the condition that a is parallel to 2b - a
def is_parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = (k • v).fst

-- Define the statement to be proven
theorem find_m (m : ℝ) : a.parallel (2 • b m - a) ↔ m = 9 / 2 :=
by 
  sorry

end find_m_l242_242436


namespace income_second_day_l242_242318

theorem income_second_day (x : ℕ) 
  (h_condition : (200 + x + 750 + 400 + 500) / 5 = 400) : x = 150 :=
by 
  -- Proof omitted.
  sorry

end income_second_day_l242_242318


namespace log_pow_evaluation_l242_242915

-- Define the condition that 9 is equal to 3^2
def nine_eq_three_squared : Prop := 9 = 3^2

-- The main theorem that we need to prove
theorem log_pow_evaluation (h : nine_eq_three_squared) : log 3 (9^3) = 6 :=
  sorry

end log_pow_evaluation_l242_242915


namespace log_three_nine_cubed_l242_242961

theorem log_three_nine_cubed : ∀ (log : ℕ → ℕ → ℕ), log 3 9 = 2 → log 3 3 = 1 → (∀ (a b n : ℕ), log a (b ^ n) = n * log a b) → log 3 (9 ^ 3) = 6 :=
by
  intros log h1 h2 h3
  sorry

end log_three_nine_cubed_l242_242961


namespace angle_case_I_angle_case_II_angle_case_III_l242_242189

-- Define the types and parameters
def d1 := 1
def e1 := 1
def f1 := 2

def d2 := 11
def e2 := 8
def f2 := 11

def d3 := 4
def e3 := 3
def f3 := 5

-- The problem stated in Lean:
theorem angle_case_I : (cos^2 (π / 3)) * 2 * (π / 3) = 90 :=
by sorry

theorem angle_case_II : (d2 + e2) * (e2 + f2) = 90 :=
by sorry

theorem angle_case_III : (cos^2 (99.6333 / 3)) * 2 * (99.6333 / 3) = 99.6333 :=
by sorry

end angle_case_I_angle_case_II_angle_case_III_l242_242189


namespace number_of_people_after_10_years_l242_242351

def number_of_people_after_n_years (n : ℕ) : ℕ :=
  Nat.recOn n 30 (fun k a_k => 3 * a_k - 20)

theorem number_of_people_after_10_years :
  number_of_people_after_n_years 10 = 1180990 := by
  sorry

end number_of_people_after_10_years_l242_242351


namespace fraction_of_juice_l242_242154

-- Define the initial conditions
variables (milk1 juice2 : ℕ) (first_transfer : ℕ)

-- Let's assume the milk1 and juice2 are the initial contents in ounces
-- and the first_transfer represents the amount of milk transferred from Cup 1 to Cup 2.
variables (milk1_initial juice2_initial first_transfer second_transfer_total : ℕ)

-- Define the amounts in the initial setup
def initial_setup := 
  milk1 = 6 ∧ juice2 = 6 ∧ first_transfer = 2 ∧ second_transfer_total = 2

-- Define the final composition in Cup 1 after transfers
def final_cup1 (milk1 juice1 : ℕ) := 
  milk1 = 4.5 ∧ juice1 = 1.5

-- Define the fraction of juice in Cup 1
def fraction_juice (juice1 total1 : ℚ) := juice1 / total1 = 1 / 4

-- Theorem to show that fraction of juice in Cup 1 is 1/4
theorem fraction_of_juice (h: initial_setup) : fraction_juice 1.5 6 := 
by
  sorry

end fraction_of_juice_l242_242154


namespace A_intersection_B_l242_242470

def A (x : ℝ) : Prop := x^2 - 16 < 0
def B (x : ℝ) : Prop := x^2 - 4x + 3 > 0
def intersection (x : ℝ) : Prop := (-4 < x ∧ x < 1) ∨ (3 < x ∧ x < 4)

theorem A_intersection_B :
  { x : ℝ | A x } ∩ { x | B x } = { x | intersection x } :=
sorry

end A_intersection_B_l242_242470


namespace log3_of_9_to_3_l242_242800

theorem log3_of_9_to_3 : Real.logb 3 (9^3) = 6 :=
by
  have h1 : 9 = 3^2 := by norm_num
  have h2 : 9^3 = (3^2)^3 := by rw [h1]
  have h3 : (3^2)^3 = 3^(2 * 3) := by rw [pow_mul]
  have h4 : Real.logb 3 (3^(2 * 3)) = 2 * 3 := Real.logb_pow 3 (2 * 3)
  rw [h4]
  norm_num

end log3_of_9_to_3_l242_242800


namespace max_integer_solutions_l242_242333

def is_antibalanced (p : ℤ[x]) : Prop :=
  p.coeff 0 = 50 ∧ ∀ n, p.coeff n ∈ ℤ

theorem max_integer_solutions : ∀ (p : ℤ[x]),
  is_antibalanced p →
  (∀ k : ℤ, p.eval k = k^3 - k) →
  ∃ n : ℕ, n = 8 :=
by
  sorry

end max_integer_solutions_l242_242333


namespace log_base_3_of_9_cubed_l242_242896

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 :=
by assumptions
  have h1 : 9 = 3^2 := sorry,
  have h2 : 9^3 = (3^2)^3 := sorry,
  have h3 : (3^2)^3 = 3^(2*3) := sorry,
  have h4 : 3^(2*3) = 3^6 := sorry,
  have h5 : log 3 (3^6) = 6 := by apply log_pow,
  sorry

end log_base_3_of_9_cubed_l242_242896


namespace cos_squared_minus_sin_squared_double_angle_formula_for_cosine_reduction_formula_for_cosine_known_value_cos_pi_fourth_cos_squared_minus_sin_squared_eq_negsqrt2_div2_l242_242362

theorem cos_squared_minus_sin_squared (α : ℝ) :
  cos (3 * π / 4) = -√2 / 2 :=
begin
  sorry
end

theorem double_angle_formula_for_cosine {α : ℝ} :
  cos(2 * α) = cos α * cos α - sin α * sin α :=
by sorry

theorem reduction_formula_for_cosine {θ : ℝ} :
  cos(π - θ) = -cos θ :=
by sorry

theorem known_value_cos_pi_fourth :
  cos (π / 4) = √2 / 2 :=
by sorry

-- The main theorem to be proved
theorem cos_squared_minus_sin_squared_eq_negsqrt2_div2 :
  (cos ((3 * π) / 8)^2 - sin ((3 * π) / 8)^2) = -(√2 / 2) :=
begin
  -- We will make use of the above theorems as necessary and sorry for the proof step
  have h1 : cos(2 * (3 * π / 8)) = cos(3 * π / 4), by {
      rw double_angle_formula_for_cosine ((3 * π) / 8),
      sorry,
  },
  have h2 : cos(3 * π / 4) = -√2 / 2, by {
      rw reduction_formula_for_cosine (π / 4),
      rw known_value_cos_pi_fourth,
      sorry,
  },
  rw ←h1,
  rw h2,
  sorry
end

end cos_squared_minus_sin_squared_double_angle_formula_for_cosine_reduction_formula_for_cosine_known_value_cos_pi_fourth_cos_squared_minus_sin_squared_eq_negsqrt2_div2_l242_242362


namespace total_cost_for_james_l242_242531

-- Prove that James will pay a total of $250 for his new pair of glasses.

theorem total_cost_for_james
  (frame_cost : ℕ := 200)
  (lens_cost : ℕ := 500)
  (insurance_cover_percentage : ℚ := 0.80)
  (coupon_on_frames : ℕ := 50) :
  (frame_cost - coupon_on_frames + lens_cost * (1 - insurance_cover_percentage)) = 250 :=
by
  -- Declare variables for the described values
  let total_frame_cost := frame_cost - coupon_on_frames
  let insurance_cover := lens_cost * insurance_cover_percentage
  let total_lens_cost := lens_cost - insurance_cover
  let total_cost := total_frame_cost + total_lens_cost

  -- We need to show total_cost = 250
  have h1 : total_frame_cost = 150 := by sorry
  have h2 : insurance_cover = 400 := by sorry
  have h3 : total_lens_cost = 100 := by sorry
  have h4 : total_cost = 250 := by
    rw [←h1, ←h3]
    sorry

  exact h4

end total_cost_for_james_l242_242531


namespace log_pow_evaluation_l242_242907

-- Define the condition that 9 is equal to 3^2
def nine_eq_three_squared : Prop := 9 = 3^2

-- The main theorem that we need to prove
theorem log_pow_evaluation (h : nine_eq_three_squared) : log 3 (9^3) = 6 :=
  sorry

end log_pow_evaluation_l242_242907


namespace probability_of_6_green_marbles_l242_242206

-- Define the conditions as constants
def green_marbles := 10
def blue_marbles := 5
def total_marbles := green_marbles + blue_marbles
def green_probability := green_marbles / total_marbles.toReal
def blue_probability := blue_marbles / total_marbles.toReal
def draws := 10
def green_draws := 6

-- Define the binomial coefficient
noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define the binomial probability formula
noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (binomial_coefficient n k).toReal * p^k * (1 - p)^(n - k)

-- The Lean 4 theorem statement to prove the question == answer given the conditions
theorem probability_of_6_green_marbles :
  binomial_probability draws green_draws green_probability ≈ 0.230 :=
by
  sorry

end probability_of_6_green_marbles_l242_242206


namespace ordered_pairs_count_l242_242481

theorem ordered_pairs_count : 
  (set.univ : set (ℕ × ℕ)).count (λ p => let (M, N) := p in M * N = 64 ∧ M ≤ 64) = 7 :=
by {
  sorry
}

end ordered_pairs_count_l242_242481


namespace ratio_spaghetti_pizza_l242_242313

/-- Define the number of students who participated in the survey and their preferences --/
def students_surveyed : ℕ := 800
def lasagna_pref : ℕ := 150
def manicotti_pref : ℕ := 120
def ravioli_pref : ℕ := 180
def spaghetti_pref : ℕ := 200
def pizza_pref : ℕ := 150

/-- Prove the ratio of students who preferred spaghetti to those who preferred pizza is 4/3 --/
theorem ratio_spaghetti_pizza : (200 / 150 : ℚ) = 4 / 3 :=
by sorry

end ratio_spaghetti_pizza_l242_242313


namespace cubic_sum_l242_242106

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 :=
by {
  sorry
}

end cubic_sum_l242_242106


namespace log_base_3_of_9_cubed_l242_242869

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 := by
  -- Assuming the necessary properties of logarithms and exponents
  sorry

end log_base_3_of_9_cubed_l242_242869


namespace convex_cyclic_quadrilaterals_count_l242_242074

theorem convex_cyclic_quadrilaterals_count :
  let num_quadrilaterals := ∑ i in (finset.range 36).powerset.filter(λ s, s.card = 4 
    ∧ let (a, b, c, d) := classical.some (vector.sorted_enum s)
    in a + b + c + d = 36 ∧ a < b + c + d ∧ b < a + c + d ∧ c < a + b + d ∧ d < a + b + c 
  ),
  finset.count :=
  num_quadrilaterals = 819 :=
begin
  sorry
end

end convex_cyclic_quadrilaterals_count_l242_242074


namespace total_nominal_income_l242_242576

theorem total_nominal_income :
  let principal := 8700
  let rate := 0.06 / 12
  let income (n : ℕ) := principal * ((1 + rate) ^ n - 1)
  income 6 + income 5 + income 4 + income 3 + income 2 + income 1 = 921.15 := by
  sorry

end total_nominal_income_l242_242576


namespace arithmetic_seq_sum_mod_9_l242_242357

def sum_arithmetic_seq := 88230 + 88231 + 88232 + 88233 + 88234 + 88235 + 88236 + 88237 + 88238 + 88239 + 88240

theorem arithmetic_seq_sum_mod_9 : 
  sum_arithmetic_seq % 9 = 0 :=
by
-- proof will be provided here
sorry

end arithmetic_seq_sum_mod_9_l242_242357


namespace log_base_3_of_9_cubed_l242_242822

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 := by
  sorry

end log_base_3_of_9_cubed_l242_242822


namespace total_nominal_income_l242_242575

noncomputable def monthly_income (principal : ℝ) (rate : ℝ) (months : ℕ) : ℝ :=
  principal * ((1 + rate) ^ months - 1)

def total_income : ℝ :=
  let rate := 0.06 / 12
  let principal := 8700
  (monthly_income principal rate 6) + 
  (monthly_income principal rate 5) + 
  (monthly_income principal rate 4) + 
  (monthly_income principal rate 3) + 
  (monthly_income principal rate 2) + 
  (monthly_income principal rate 1)

theorem total_nominal_income :
  total_income = 921.15 :=
by
  sorry

end total_nominal_income_l242_242575


namespace number_of_green_marbles_in_basketB_l242_242251

structure Basket where
  red : ℕ
  yellow : ℕ
  green : Option ℕ
  white : Option ℕ

def basketA : Basket := { red := 4, yellow := 2, green := none, white := none }
def basketB (g : ℕ) : Basket := { red := 0, yellow := 1, green := some g, white := none }
def basketC : Basket := { red := 0, yellow := 9, green := none, white := some 3 }

theorem number_of_green_marbles_in_basketB (g : ℕ) :
  (let diffA := abs (basketA.red - basketA.yellow) in
   let diffB := abs (g - basketB(g).yellow) in
   let diffC := abs (basketC.white.get_or_else 0 - basketC.yellow) in
   max (max diffA diffB) diffC = 6) → g = 7 :=
  by
    intros
    sorry

end number_of_green_marbles_in_basketB_l242_242251


namespace find_a_l242_242456

variable {x a : ℝ}

def A (x : ℝ) : Prop := x ≤ -1 ∨ x > 2
def B (x a : ℝ) : Prop := x < a ∨ x > a + 1

theorem find_a (hA : ∀ x, (x + 1) / (x - 2) ≥ 0 ↔ A x)
                (hB : ∀ x, x^2 - (2 * a + 1) * x + a^2 + a > 0 ↔ B x a)
                (hSub : ∀ x, A x → B x a) :
  -1 < a ∧ a ≤ 1 :=
sorry

end find_a_l242_242456


namespace initial_apps_count_l242_242744

theorem initial_apps_count (X : ℕ) 
  (added_apps : X + 71)
  (remaining_apps : X + 71 - (71 + 1) = 14) 
  (deleted_apps : 71 + 1) : 
  X = 15 :=
by 
  sorry

end initial_apps_count_l242_242744


namespace geometric_sequence_first_term_l242_242411

theorem geometric_sequence_first_term (a b : ℕ) (h1 : b = 243 / 3) (h2 : a = 81 / 3) 
(h3 : 243 / 3 = 81) (h4 : 81 / 3 = 27) (h5 : 27 / 3 = 9) (h6 : 9 / 3 = 3) : 
PROP (a, b = 81, 243, 729) (x = 3). sorry


end geometric_sequence_first_term_l242_242411


namespace probability_is_two_thirds_l242_242453

noncomputable def probabilityOfEvent : ℚ :=
  let Ω := { p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 6 }
  let A := { p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 6 ∧ 2 * p.1 - p.2 + 2 ≥ 0 }
  let area_Ω := (2 - 0) * (6 - 0)
  let area_A := area_Ω - (1 / 2) * 2 * 4
  (area_A / area_Ω : ℚ)

theorem probability_is_two_thirds : probabilityOfEvent = (2 / 3 : ℚ) :=
  sorry

end probability_is_two_thirds_l242_242453


namespace log_base_3_of_9_cubed_l242_242782

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 :=
by
  -- Given conditions:
  -- 1. 9 = 3^2
  -- 2. log_b (a^n) = n * log_b a
  sorry

end log_base_3_of_9_cubed_l242_242782


namespace train_length_l242_242665

theorem train_length (train_speed_kmh : ℝ) (bike_speed_kmh : ℝ) (overtake_time_s : ℝ) 
  (h_train_speed : train_speed_kmh = 100) (h_bike_speed : bike_speed_kmh = 64) (h_overtake_time : overtake_time_s = 85) : 
  let train_speed_ms := train_speed_kmh * (1000 / 3600), bike_speed_ms := bike_speed_kmh * (1000 / 3600) in
  let relative_speed_ms := train_speed_ms - bike_speed_ms in
  let length_of_train := relative_speed_ms * overtake_time_s in
  length_of_train = 850 := 
by
  sorry

end train_length_l242_242665


namespace log_base_3_of_9_cubed_l242_242868

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 := by
  -- Assuming the necessary properties of logarithms and exponents
  sorry

end log_base_3_of_9_cubed_l242_242868


namespace tangent_product_is_product_of_radii_l242_242214

-- Definitions for given conditions
variables {R r : ℝ} (O1 O2 A B C : Point)

-- Conditions
axiom circles_have_radii : Circle O1 R ∧ Circle O2 r
axiom common_tangent_intersects_at_AB : Tangent (CommonInteriorTangent O1 O2) A B
axiom tangent_touches_circle_at_C : PointOnCircle C O1 R

-- The Proof statement
theorem tangent_product_is_product_of_radii :
  AC * CB = R * r :=
sorry

end tangent_product_is_product_of_radii_l242_242214


namespace cottage_rental_hours_l242_242153

-- Condition: the cost of renting the cottage is $5 per hour
def cost_per_hour : ℕ := 5

-- Condition: Jack and Jill each paid $20, so the total paid is $40
def total_paid : ℕ := 20 + 20

-- Define the main statement in Lean
theorem cottage_rental_hours : total_paid / cost_per_hour = 8 := by
  have h := total_paid / cost_per_hour
  show h = 8
  sorry

end cottage_rental_hours_l242_242153


namespace roots_of_quadratic_l242_242025

theorem roots_of_quadratic (α β : ℝ) (h1 : α^2 - 4*α - 5 = 0) (h2 : β^2 - 4*β - 5 = 0) :
  3*α^4 + 10*β^3 = 2593 := 
by
  sorry

end roots_of_quadratic_l242_242025


namespace minimum_value_of_f_l242_242408

noncomputable def f (x : ℝ) : ℝ :=
  x - 1 - (Real.log x) / x

theorem minimum_value_of_f : (∀ x > 0, f x ≥ 0) ∧ (∃ x > 0, f x = 0) :=
by
  sorry

end minimum_value_of_f_l242_242408


namespace cylinder_original_radius_l242_242752

-- Define the original height and the conditions
def original_height : ℝ := 5
def increased_radius_Vol (r : ℝ) : ℝ := π * (r + 4)^2 * original_height
def increased_height_Vol (r : ℝ) : ℝ := π * r^2 * (original_height + 10)

-- Stating the theorem
theorem cylinder_original_radius (r : ℝ) (h : increased_radius_Vol r = increased_height_Vol r) : 
  r = 2 + 2 * Real.sqrt 3 :=
sorry

end cylinder_original_radius_l242_242752


namespace exists_odd_game_count_l242_242514

theorem exists_odd_game_count (n : ℕ) (h : n = 15) :
  ∃ i j ∈ Finset.range n, i ≠ j ∧ ((games_played_before i j) + (games_played_before j i)) % 2 = 1 := 
sorry

end exists_odd_game_count_l242_242514


namespace magnitude_sum_l242_242434

-- Given conditions
variables {z1 z2 : ℂ}

-- Definitions based on the conditions
def magnitude_condition (z : ℂ) : Prop := complex.abs z = 1
def distance_condition (z1 z2 : ℂ) : Prop := complex.abs (z1 - z2) = √3

-- Theorem statement
theorem magnitude_sum (hz1 : magnitude_condition z1) (hz2 : magnitude_condition z2) (hdist : distance_condition z1 z2) :
  complex.abs (z1 + z2) = 1 :=
sorry

end magnitude_sum_l242_242434


namespace log_base_3_of_9_cubed_l242_242901

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 :=
by assumptions
  have h1 : 9 = 3^2 := sorry,
  have h2 : 9^3 = (3^2)^3 := sorry,
  have h3 : (3^2)^3 = 3^(2*3) := sorry,
  have h4 : 3^(2*3) = 3^6 := sorry,
  have h5 : log 3 (3^6) = 6 := by apply log_pow,
  sorry

end log_base_3_of_9_cubed_l242_242901


namespace find_m_l242_242539

def M (m : ℝ) := {1, 2, (m^2 - 2*m - 5) + (m^2 + 5*m + 6)*complex.I}
def N : set ℝ := {3, 10}

theorem find_m (m : ℝ) :
  (M m ∩ N).nonempty ↔ (m = -2 ∨ m = -3) :=
by 
  sorry

end find_m_l242_242539


namespace possible_values_of_m_l242_242473

theorem possible_values_of_m (m : ℝ) :
  let A := {x | x^2 - 4 * x + 3 = 0}
  let B := {x | ∃ m : ℝ, m * x + 1 = 0}
  (∀ x, x ∈ B → x ∈ A) ↔ m = 0 ∨ m = -1 ∨ m = -1 / 3 :=
by
  let A := {x | x^2 - 4 * x + 3 = 0}
  let B := {x | ∃ m : ℝ, m * x + 1 = 0}
  sorry -- Proof needed

end possible_values_of_m_l242_242473


namespace value_S3_S2_S5_S3_l242_242012

variable {a : ℕ → ℝ} {S : ℕ → ℝ}
variable {d : ℝ}
variable (h_arith_seq : ∀ n, a (n + 1) = a n + d)
variable (d_ne_zero : d ≠ 0)
variable (h_geom_seq : (a 1 + 2 * d) ^ 2 = (a 1) * (a 1 + 3 * d))
variable (S_def : ∀ n, S n = n * a 1 + d * (n * (n - 1)) / 2)

theorem value_S3_S2_S5_S3 : (S 3 - S 2) / (S 5 - S 3) = 2 := by
  sorry

end value_S3_S2_S5_S3_l242_242012


namespace combined_tax_rate_l242_242187

-- Define the incomes and tax rates
def income_Mork := m : ℝ
def tax_rate_Mork := 0.45
def income_Mindy := 4 * m
def tax_rate_Mindy := 0.15
def income_Orson := 2 * m
def tax_rate_Orson := 0.25

-- Define taxes paid by each individual
def tax_Mork := tax_rate_Mork * income_Mork
def tax_Mindy := tax_rate_Mindy * income_Mindy
def tax_Orson := tax_rate_Orson * income_Orson

-- Define total tax and income
def total_tax := tax_Mork + tax_Mindy + tax_Orson
def total_income := income_Mork + income_Mindy + income_Orson

-- Statement to prove the combined tax rate is 0.2214 (or 22.14%)
theorem combined_tax_rate :
  (total_tax / total_income) = 0.2214 :=
by 
  -- Insert the full mathematical proof steps here
  sorry

end combined_tax_rate_l242_242187


namespace problem_statement_l242_242973

open Nat

theorem problem_statement (k : ℕ) (hk : k > 0) : 
  (∀ n : ℕ, n > 0 → 2^((k-1)*n+1) * (factorial (k*n) / factorial n) ≤ (factorial (k*n) / factorial n))
  ↔ ∃ a : ℕ, k = 2^a := 
sorry

end problem_statement_l242_242973


namespace log_base_3_of_9_cubed_l242_242770

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 :=
by
  -- Given conditions:
  -- 1. 9 = 3^2
  -- 2. log_b (a^n) = n * log_b a
  sorry

end log_base_3_of_9_cubed_l242_242770


namespace kid_ticket_cost_l242_242683

theorem kid_ticket_cost :
  ∀ (cost_adult total_sales num_adults total_people : ℕ), 
  cost_adult = 28 → 
  total_sales = 3864 → 
  num_adults = 51 → 
  total_people = 254 → 
  let num_kids := total_people - num_adults in
  let sales_adult := num_adults * cost_adult in
  let sales_kid := total_sales - sales_adult in
  let cost_kid := sales_kid / num_kids in
  cost_kid = 12 :=
by
  intros cost_adult total_sales num_adults total_people h1 h2 h3 h4;
  let num_kids := total_people - num_adults;
  let sales_adult := num_adults * cost_adult;
  let sales_kid := total_sales - sales_adult;
  let cost_kid := sales_kid / num_kids;
  sorry

end kid_ticket_cost_l242_242683


namespace find_h_l242_242228

theorem find_h
  (h j k : ℝ)
  (y1 y2 : ℝ → ℝ)
  (intercept_cond1 : y1 0 = 4027)
  (intercept_cond2 : y2 0 = 4028)
  (intercept_formula1 : y1 = (λ x, 4 * (x - h)^2 + j))
  (intercept_formula2 : y2 = (λ x, 5 * (x - h)^2 + k))
  (graph1_has_two_pos_int_x_intercepts : ∃ x1 x2 : ℝ, (y1 x1 = 0 ∧ y1 x2 = 0) ∧ (0 < x1 ∧ 0 < x2))
  (graph2_has_two_pos_int_x_intercepts : ∃ x1 x2 : ℝ, (y2 x1 = 0 ∧ y2 x2 = 0) ∧ (0 < x1 ∧ 0 < x2)) :
  h = 36 := 
sorry

end find_h_l242_242228


namespace year_2023_AD_denotation_l242_242508

theorem year_2023_AD_denotation :
  (∀ y : ℕ, (-y = -500) ↔ (y = 500)) → 
  (∀ x : ℕ, x > 0 ↔ x = 2023) → 
  2023 = 2023 :=
by
  intros h1 h2
  sorry

end year_2023_AD_denotation_l242_242508


namespace prove_radii_diff_equals_OI_l242_242696

noncomputable def triangle (ABC: Type) : Type := sorry

structure Triangle :=
  (A B C O I : ABC)
  (D : ABC -> Point := λ triangle, sorry)

axiom circumradius (ABC: Type) (t : Triangle ABC) : ABC -> ℝ
  | .r1 => sorry   -- Circumradius of triangle ABD
  | .r2 => sorry   -- Circumradius of triangle ACD

axiom EulerTheorem (ABC: Type) (t : Triangle ABC) : ℝ
  := sorry -- theorem: OI^2 = R^2 - 2Rr
  
theorem prove_radii_diff_equals_OI (ABC : Type) (t : Triangle ABC) :
  circumradius ABC t .r1 - circumradius ABC t .r2 = distance t.O t.I :=
sorry

end prove_radii_diff_equals_OI_l242_242696


namespace problem_convex_quad_cos_A_l242_242516

theorem problem_convex_quad_cos_A
  (ABCD : ConvexQuadrilateral)
  (h_angle : ∠ ABCD.A ≅ ∠ ABCD.C)
  (h_AB_CD : ABCD.AB = 200 ∧ ABCD.CD = 200)
  (h_AD_neq_BC : ABCD.AD ≠ ABCD.BC)
  (h_perimeter : ABCD.perimeter = 680) :
  floor (1000 * cos ABCD.A) = 700 :=
sorry

end problem_convex_quad_cos_A_l242_242516


namespace inequality_proof_l242_242421

theorem inequality_proof (x y : ℝ) (n : ℕ) (hx : 0 < x) (hy : 0 < y) (hn : 0 < n):
  x^n / (1 + x^2) + y^n / (1 + y^2) ≤ (x^n + y^n) / (1 + x * y) :=
by
  sorry

end inequality_proof_l242_242421


namespace log_three_nine_cubed_l242_242958

theorem log_three_nine_cubed : ∀ (log : ℕ → ℕ → ℕ), log 3 9 = 2 → log 3 3 = 1 → (∀ (a b n : ℕ), log a (b ^ n) = n * log a b) → log 3 (9 ^ 3) = 6 :=
by
  intros log h1 h2 h3
  sorry

end log_three_nine_cubed_l242_242958


namespace log_base_3_l242_242853

theorem log_base_3 (log_identity : ∀ (b a c : ℝ), log b (a^c) = c * log b a) :
  log 3 (9^3) = 6 :=
by
  have : 9 = 3^2 := by sorry
  have : log 3 3 = 1 := by sorry
  exact (log_identity 3 (3^2) 3).trans sorry

end log_base_3_l242_242853


namespace area_of_circular_garden_l242_242254

-- Definitions of the conditions
def center (C A B: Type) := (C ∈ Affine.looksLike (Metric.sphere (20/2 : ℝ)))
def midpoint (D A B: Type) := (D = (A + B) / 2)
def distance (x y: Type) := real.norm (x - y)

-- Define the problem
theorem area_of_circular_garden (C A B D: Type) 
(h_center: center C A B)
(h_AB : distance A B = 20)
(h_midpoint : midpoint D A B)
(h_D_to_C : distance D C = 12) : 
area = 244 * π := 
sorry

end area_of_circular_garden_l242_242254


namespace value_of_M_is_800_l242_242383

theorem value_of_M_is_800 : 
  let M := (sum (list.map (λ n, if n % 3 == 1 then (list.nth_le [50, 47, ..., 5] n (by sorry))^2 - (list.nth_le [49, 46, ..., 4] n (by sorry))^2 else (list.nth_le [48, 45, ..., 3] n (by sorry))^2) [0..15]))
  M = 800 :=
by sorry

end value_of_M_is_800_l242_242383


namespace donation_amounts_l242_242398

theorem donation_amounts 
  (d : Fin 5 → ℕ)
  (h_avg : (∑ i, d i) = 2800)
  (h_int_mult : ∀ i, d i % 100 = 0)
  (h_min : ∃ i, d i = 200)
  (h_max : ∃ i, d i = 800 ∧ ∀ j ≠ i, d j ≠ 800)
  (h_specific : ∃ i, d i = 600)
  (h_median : (d (Fin.ofNat 2)) = 600) : 
  ∃ a b, (a = 500 ∧ b = 700) ∨ (a = 600 ∧ b = 600) :=
sorry

end donation_amounts_l242_242398


namespace log3_of_9_to_3_l242_242799

theorem log3_of_9_to_3 : Real.logb 3 (9^3) = 6 :=
by
  have h1 : 9 = 3^2 := by norm_num
  have h2 : 9^3 = (3^2)^3 := by rw [h1]
  have h3 : (3^2)^3 = 3^(2 * 3) := by rw [pow_mul]
  have h4 : Real.logb 3 (3^(2 * 3)) = 2 * 3 := Real.logb_pow 3 (2 * 3)
  rw [h4]
  norm_num

end log3_of_9_to_3_l242_242799


namespace angus_tokens_l242_242561

theorem angus_tokens (x : ℕ) (h1 : x = 60 - (25 / 100) * 60) : x = 45 :=
by
  sorry

end angus_tokens_l242_242561


namespace correct_degree_l242_242660

variable (a b x y : ℝ)
variable π : ℝ

-- Definitions based on the conditions
def is_monomial (e : ℝ) : Prop := 
  -- a monomial has only one term
  ∃ c : ℝ, e = c

def coefficient (t : ℝ) : ℝ :=
  -- coefficient is the numerical part of the term
  t

def degree (mon : ℝ) : ℕ :=
  -- Degree is the sum of the exponents for a monomial
  if mon = -5 * a^2 * b then 3 else 0

-- Theorem to prove statement D is correct
theorem correct_degree : 
  degree (-5 * a^2 * b) = 3 := sorry

end correct_degree_l242_242660


namespace proof_part1_proof_part2_l242_242458

noncomputable def f (x a : ℝ) : ℝ := x^3 - a * x^2 + 3 * x

def condition1 (a : ℝ) : Prop := ∀ x : ℝ, x ≥ 1 → 3 * x^2 - 2 * a * x + 3 ≥ 0

def condition2 (a : ℝ) : Prop := 3 * 3^2 - 2 * a * 3 + 3 = 0

theorem proof_part1 (a : ℝ) : condition1 a → a ≤ 3 := 
sorry

theorem proof_part2 (a : ℝ) (ha : a = 5) : 
  f 1 a = -1 ∧ f 3 a = -9 ∧ f 5 a = 15 :=
sorry

end proof_part1_proof_part2_l242_242458


namespace log_base_3_of_9_cubed_l242_242920
-- Import the necessary parts of the standard library

-- Define the statement to be proved
theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 :=
by
  sorry

end log_base_3_of_9_cubed_l242_242920


namespace Paco_cookies_l242_242190

theorem Paco_cookies :
  let initial_cookies := 120
  let cookies_to_first_friend := 34
  let cookies_to_second_friend := 29
  let cookies_eaten := 20
  let cookies_bought := 45
  let cookies_to_third_friend := 15
  let final_cookies :=
    initial_cookies - cookies_to_first_friend - cookies_to_second_friend - cookies_eaten + cookies_bought - cookies_to_third_friend
  final_cookies = 67 := by 
  let initial_cookies := 120
  let cookies_to_first_friend := 34
  let cookies_to_second_friend := 29
  let cookies_eaten := 20
  let cookies_bought := 45
  let cookies_to_third_friend := 15
  let final_cookies :=
    initial_cookies - cookies_to_first_friend - cookies_to_second_friend - cookies_eaten + cookies_bought - cookies_to_third_friend
  show final_cookies = 67 from by
    sorry

end Paco_cookies_l242_242190


namespace standard_deviation_scaled_l242_242430

variable {α : Type*} [Nonempty α] [HasSmul ℝ α] [AddCommGroup α] [Module ℝ α]

variable (σ : α → ℝ)

theorem standard_deviation_scaled (x : α) (h : σ x = sqrt 2) : σ (3 • x) = 3 * sqrt 2 :=
by sorry

end standard_deviation_scaled_l242_242430


namespace unique_polynomial_in_base_4_l242_242537

-- Define the conditions: n is a positive integer
variable (n : ℕ)
variable (hn : n > 0)

-- Define the problem conditions:
-- Find the number of polynomials P(x) with the coefficients in {0, 1, 2, 3} 
-- such that P(2) = n and prove it's exactly 1.
theorem unique_polynomial_in_base_4 (n : ℕ) (hn : n > 0) : 
  ∃! P : Polynomial ℕ, (∀ a ∈ P.coeffs, a ∈ {0, 1, 2, 3}) ∧ P.eval 2 = n := 
sorry

end unique_polynomial_in_base_4_l242_242537


namespace equal_expr_count_l242_242480

theorem equal_expr_count :
  ∀ (x : ℕ), x > 0 →
    ((x^x * x! = x! * x^x) ∧ ¬(x^x * x! = x^(x+1)) ∧
     ¬(x^x * x! = (x!)^x) ∧ ¬(x^x * x! = x^(x!)) ) → 1 :=
begin
  intros x x_pos,
  repeat { split },
  { -- x! * x^x = x^x * x! (True)
    exact eq_comm },
  { -- x^x * x! ≠ x^(x+1) (True)
    intro h, 
    -- verifying it's not equal
    sorry },
  { -- x^x * x! ≠ (x!)^x (True)
    intro h,
    -- verifying it's not equal for some x
    sorry },
  { -- x^x * x! ≠ x^(x!) (True)
    intro h, 
    -- verifying it's not equal for some x
    sorry }
end

end equal_expr_count_l242_242480


namespace sum_even_integers_200_to_400_l242_242290

def is_even (n : ℤ) : Prop := n % 2 = 0
def is_in_range (n : ℤ) : Prop := 200 <= n ∧ n <= 400
def is_valid_even_number (n : ℤ) : Prop := is_even n ∧ is_in_range n

theorem sum_even_integers_200_to_400 :
  ∃ (sum : ℤ), (sum = ∑ n in (finset.filter is_valid_even_number (finset.Icc 200 400)), n) ∧ sum = 29700 :=
begin
  sorry
end

end sum_even_integers_200_to_400_l242_242290


namespace pins_after_30_days_l242_242195

-- Define the initial state and parameters
variables (X Y Z : ℝ)
constants (initial_pins : ℝ := 1000) (group_size : ℕ := 20) (days_in_month : ℕ := 30)

-- Define the average daily group contribution
def group_contribution_per_day := X * group_size

-- Define Reese's additional daily contribution over the group average
def reese_extra_contribution_per_day := Z - X

-- Define the owner's weekly deletion rate
def owner_weekly_deletion_per_person := Y

-- Define the total weekly deletion rate
def owner_weekly_deletion := owner_weekly_deletion_per_person * group_size

-- Define the total deletion for the extra 2 days beyond the 4 weeks
def extra_days_deletion := (owner_weekly_deletion_per_person / 7) * 2 * group_size

-- Define the total pins calculation after 30 days
def total_pins_after_30_days := initial_pins
                              + group_contribution_per_day * days_in_month
                              + reese_extra_contribution_per_day * days_in_month
                              - owner_weekly_deletion * (days_in_month / 7)
                              - extra_days_deletion

-- The goal is to prove this equality
theorem pins_after_30_days (h1 : 30 % 7 = 2) : total_pins_after_30_days X Y Z = 
  1000 + 570 * X + 30 * Z - 80 * Y - (40 * Y / 7) :=
by
  simp [total_pins_after_30_days, group_contribution_per_day, reese_extra_contribution_per_day, owner_weekly_deletion, extra_days_deletion]
  sorry -- Proof to be filled in

end pins_after_30_days_l242_242195


namespace log_three_pow_nine_pow_three_eq_six_l242_242940

open Real

noncomputable def eval_log_three_pow_nine_pow_three : ℝ :=
  log 3 (9^3)

theorem log_three_pow_nine_pow_three_eq_six :
  eval_log_three_pow_nine_pow_three = 6 := by
  sorry

end log_three_pow_nine_pow_three_eq_six_l242_242940


namespace equal_max_clique_partition_l242_242355

open Finite

noncomputable theory

def max_clique_size {V : Type*} (G : SimpleGraph V) (s : Set V) : ℕ :=
  (s.powerset.filter (λ t, G.subgraph (coe t) t.pairwise)).image (λ t, t.to_finset.card).sup id

theorem equal_max_clique_partition {V : Type*} (G : SimpleGraph V) (h_even : ∃ k, 2 * k = @max_clique_size _ G univ) :
  ∃ (A B : Set V), A ∪ B = univ ∧ A ∩ B = ∅ ∧ max_clique_size G A = max_clique_size G B :=
sorry

end equal_max_clique_partition_l242_242355


namespace sum_even_integers_200_to_400_l242_242293

def is_even (n : ℤ) : Prop := n % 2 = 0
def is_in_range (n : ℤ) : Prop := 200 <= n ∧ n <= 400
def is_valid_even_number (n : ℤ) : Prop := is_even n ∧ is_in_range n

theorem sum_even_integers_200_to_400 :
  ∃ (sum : ℤ), (sum = ∑ n in (finset.filter is_valid_even_number (finset.Icc 200 400)), n) ∧ sum = 29700 :=
begin
  sorry
end

end sum_even_integers_200_to_400_l242_242293


namespace number_of_solutions_l242_242443

noncomputable def f : ℝ → ℝ
| x := if x >= 0 then -(x-1)^2 + 1 else -(abs(x)-1)^2 + 1

theorem number_of_solutions :
  ∃ (a:ℕ), a = 8 ∧ ∃ (x:ℝ → ℝ), (∀ x, f (-x) = f x) ∧ ∀ x, f[x] = (if x >= 0 then -(x-1)^2 + 1 else -(abs(x)-1)^2 + 1) ∧
  (∃ (a : ℝ), f (f a) = 1/2) ∧ 
  (∃ (x : ℝ) (solutions : finset ℝ), solutions.card = 8 ∧ ∀ x ∈ solutions, f (f x) = 1/2) := 
by
  sorry

end number_of_solutions_l242_242443


namespace side_length_of_third_octagon_l242_242252

theorem side_length_of_third_octagon (k : ℝ) : 
  let a := 7 in
  let b := 42 in
  let ratio := 1/6 in
  ∀ x : ℝ, (a^2 + b^2 = 7 * (2058:x)) → x = 7 * real.sqrt(6)
:= 
begin
  intros,
  sorry -- skipping the proof part as per instruction
end

end side_length_of_third_octagon_l242_242252


namespace MN_bisects_PQ_l242_242415

-- Define conditions as separate statements
variables {A M N K L P Q : Point}
variables (circle : Circle)
variables (AM AN : Line)
variables (secant : Line)
variables (line_l : Line)

-- Assume all properties as definitions or axioms
axiom point_A_outside_circle : outside A circle
axiom tangents_from_point : is_tangent AM circle A M ∧ is_tangent AN circle A N
axiom secant_definition : secant_intersects_circle secant circle K L
axiom line_parallel_AM : parallel line_l AM
axiom KM_intersects_line : intersects_line KM line_l P
axiom LM_intersects_line : intersects_line LM line_l Q

theorem MN_bisects_PQ (MN : Line) (cond : is_tangent AM circle A M ∧ is_tangent AN circle A N
                                  ∧ secant_intersects_circle secant circle K L
                                  ∧ parallel line_l AM
                                  ∧ intersects_line KM line_l P
                                  ∧ intersects_line LM line_l Q) :
  bisects MN PQ := sorry

end MN_bisects_PQ_l242_242415


namespace prove_tan_A_prove_a_l242_242149

noncomputable def triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  a * Real.cos C + (c - 3 * b) * Real.cos A = 0

theorem prove_tan_A (A B C a b c : ℝ) (h : triangle A B C a b c) :
  Real.tan A = 2 * Real.sqrt 2 :=
sorry

noncomputable def triangle_area (a b c : ℝ) (A : ℝ) : Prop :=
  (1 / 2) * b * c * Real.sin A = Real.sqrt 2 

theorem prove_a (A B C a b c : ℝ)
  (h1 : triangle A B C a b c)
  (h2 : triangle_area b c A)
  (h3 : b - c = 2) :
  a = 2 * Real.sqrt 2 :=
sorry

end prove_tan_A_prove_a_l242_242149


namespace log_three_pow_nine_pow_three_eq_six_l242_242945

open Real

noncomputable def eval_log_three_pow_nine_pow_three : ℝ :=
  log 3 (9^3)

theorem log_three_pow_nine_pow_three_eq_six :
  eval_log_three_pow_nine_pow_three = 6 := by
  sorry

end log_three_pow_nine_pow_three_eq_six_l242_242945


namespace inequality_solution_l242_242607

noncomputable def solve_inequality (a : ℝ) (h : a > 0) : set ℝ :=
  if ha1 : a > 1 then
    { x : ℝ | x < 1 / a ∨ x > a }
  else if ha2 : a = 1 then
    { x : ℝ | x ≠ 1 }
  else
    { x : ℝ | x < a ∨ x > 1 / a }

theorem inequality_solution (a : ℝ) (h : a > 0) :
  ∀ x : ℝ, (ax^2 - (a^2 + 1)x + a > 0) ↔ x ∈ (solve_inequality a h) :=
sorry

end inequality_solution_l242_242607


namespace class_fund_after_trip_l242_242564

variable (initial_fund contribution_per_student trip_cost_per_student num_students : ℕ)
variable (fund_after_contributions total_trip_cost final_fund : ℕ)

def total_contributed_by_students := contribution_per_student * num_students
def total_fund := initial_fund + total_contributed_by_students
def total_trip_cost := trip_cost_per_student * num_students
def final_fund := total_fund - total_trip_cost

theorem class_fund_after_trip
    (h1 : initial_fund = 50)
    (h2 : contribution_per_student = 5)
    (h3 : trip_cost_per_student = 7)
    (h4 : num_students = 20) :
    final_fund = 10 := by
  sorry

end class_fund_after_trip_l242_242564


namespace log_evaluation_l242_242758

theorem log_evaluation : ∀ (a : ℕ), a = 3 → log 3 (9^3) = 6 :=
by
  intro a ha
  have h1 : 9 = 3^2 := by sorry
  have h2 : (9^3) = (3^2)^3 := by rw [h1]
  have h3 : (3^2)^3 = 3^6 := by sorry
  have h4 : log 3 (3^6) = 6 := by sorry
  rw [h2, h3, h4]
  exact ha

end log_evaluation_l242_242758


namespace log_base_3_l242_242851

theorem log_base_3 (log_identity : ∀ (b a c : ℝ), log b (a^c) = c * log b a) :
  log 3 (9^3) = 6 :=
by
  have : 9 = 3^2 := by sorry
  have : log 3 3 = 1 := by sorry
  exact (log_identity 3 (3^2) 3).trans sorry

end log_base_3_l242_242851


namespace pt_length_l242_242137

variables (PQ RS PR PT TR : ℝ)
variables (T : Type)
variables [convex_quadrilateral PQRS]

theorem pt_length 
  (hPQ : PQ = 10)
  (hRS : RS = 15)
  (hPR : PR = 18)
  (h: PT + TR = PR)
  (hArea: ∀ (PTS QTR : T), area PTS = area QTR) :
  PT = 7.2 :=
sorry

end pt_length_l242_242137


namespace exists_points_with_midpoint_l242_242179

-- Definition of the problem
theorem exists_points_with_midpoint 
  (C : set (ℝ × ℝ)) -- C is a closed curve in the plane
  (P : ℝ × ℝ)       -- P is a point in the plane
  (hC_cont : continuous (λ (θ : ℝ), (C θ)) ) -- C is continuous
  (hC_closed : is_closed C )  -- C is closed
  (hC_non_intersect : ∀ (θ₁ θ₂ : ℝ), C θ₁ = C θ₂ → θ₁ = θ₂) -- C does not intersect itself
  (hP_inside : P ∈ interior C)  -- P is inside C
  : ∃ Q Q' ∈ C, midpoint Q Q' = P := 
sorry

end exists_points_with_midpoint_l242_242179


namespace total_nominal_income_l242_242569

theorem total_nominal_income
  (c1 : 8700 * ((1 + 0.06 / 12) ^ 6 - 1) = 264.28)
  (c2 : 8700 * ((1 + 0.06 / 12) ^ 5 - 1) = 219.69)
  (c3 : 8700 * ((1 + 0.06 / 12) ^ 4 - 1) = 175.31)
  (c4 : 8700 * ((1 + 0.06 / 12) ^ 3 - 1) = 131.15)
  (c5 : 8700 * ((1 + 0.06 / 12) ^ 2 - 1) = 87.22)
  (c6 : 8700 * (1 + 0.06 / 12 - 1) = 43.5) :
  264.28 + 219.69 + 175.31 + 131.15 + 87.22 + 43.5 = 921.15 := by
  sorry

end total_nominal_income_l242_242569


namespace log_base_3_l242_242844

theorem log_base_3 (log_identity : ∀ (b a c : ℝ), log b (a^c) = c * log b a) :
  log 3 (9^3) = 6 :=
by
  have : 9 = 3^2 := by sorry
  have : log 3 3 = 1 := by sorry
  exact (log_identity 3 (3^2) 3).trans sorry

end log_base_3_l242_242844


namespace log_base_3_of_9_cubed_l242_242834

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 := by
  -- Define the known conditions
  have h1 : 9 = 3^2 := by norm_num
  have h2 : log 3 (a^n) = n * log 3 a := sorry  -- Statement of the logarithmic identity
  
  -- Apply the conditions to prove the result
  calc
    log 3 (9^3)
        = log 3 ((3^2)^3) : by rw h1
    ... = log 3 (3^(2 * 3)) : by congr; norm_num
    ... = log 3 (3^6) : by norm_num
    ... = 6 : by rw [h2]; norm_num

end log_base_3_of_9_cubed_l242_242834


namespace log_product_identity_l242_242496

theorem log_product_identity (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x ≠ y) : 
  log x y * log y x * log x x = 1 := 
by sorry

end log_product_identity_l242_242496


namespace solve_equation_l242_242205

-- Defining the equation and the conditions for positive integers
def equation (x y z v : ℕ) : Prop :=
  x + 1 / (y + 1 / (z + 1 / (v : ℚ))) = 101 / 91

-- Stating the problem in Lean
theorem solve_equation :
  ∃ x y z v : ℕ, equation x y z v ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧ v > 0 ∧
  x = 1 ∧ y = 9 ∧ z = 9 ∧ v = 1 :=
by
  sorry

end solve_equation_l242_242205


namespace max_area_of_triangle_l242_242509

theorem max_area_of_triangle 
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : a = 2)
  (h2 : 4 * (Real.cos (A / 2))^2 -  Real.cos (2 * (B + C)) = 7 / 2)
  (h3 : A + B + C = Real.pi) :
  (Real.sqrt 3 / 2 * b * c) ≤ Real.sqrt 3 :=
sorry

end max_area_of_triangle_l242_242509


namespace solution_set_l242_242242

theorem solution_set:
  (∃ x y : ℝ, x - y = 0 ∧ x^2 + y = 2) ↔ (∃ x y : ℝ, (x = 1 ∧ y = 1) ∨ (x = -2 ∧ y = -2)) :=
by
  sorry

end solution_set_l242_242242


namespace limit_bn_over_n_eq_zero_l242_242035

-- Define our sequence a_n of positive integers
variable {a : ℕ → ℕ}
-- Define the condition that a_n are positive integers
axiom positive_integers : ∀ n, a n > 0

-- Define the condition that the sum of 1/a_n converges
axiom series_converges : ∃ l : ℝ, has_sum (λ n, 1 / (a n : ℝ)) l

-- Define b_n, the count of elements a_i with a_i ≤ n
def b (n : ℕ) : ℕ := {i : ℕ | i ≤ n ∧ a i ≤ n}.card

-- State the theorem to be proven
theorem limit_bn_over_n_eq_zero : 
  tendsto (λ n, (b n : ℝ) / n) at_top (𝓝 0) :=
sorry

end limit_bn_over_n_eq_zero_l242_242035


namespace determine_transformation_constants_l242_242055

def h (x : ℝ) : ℝ :=
if x ∈ Icc (-4 : ℝ) 0 then -3 - x
else if x ∈ Icc (0 : ℝ) 6 then Real.sqrt (9 - (x - 3)^2) - 3
else if x ∈ Icc (3 : ℝ) 4 then 3 * (x - 3)
else 0

def k (x : ℝ) (p q r : ℝ) : ℝ := p * h (q * x) + r

theorem determine_transformation_constants : ∃ (p q r : ℝ), 
  (∀ x : ℝ, k x p q r = h (x / 3) - 5) ∧ 
  p = 1 ∧ q = 1 / 3 ∧ r = -5 :=
by 
  use 1, (1 / 3 : ℝ), -5
  split
  intro x
  simp [k, h]
  sorry

end determine_transformation_constants_l242_242055


namespace divisibility_by_9_l242_242191

theorem divisibility_by_9 (d : ℕ) (n_k : ℕ → ℕ) (n : ℕ) (S : ℕ) :
  (n = ∑ k in Finset.range (d + 1), 10 ^ k * n_k k) →
  (S = ∑ k in Finset.range (d + 1), n_k k) →
  (9 ∣ n ↔ 9 ∣ S) :=
by
  intro n_def S_def
  sorry

end divisibility_by_9_l242_242191


namespace binom_computation_l242_242379

noncomputable def binom : ℕ → ℕ → ℕ
| n, 0       => 1
| 0, k+1     => 0
| n+1, k+1   => binom n k + binom n (k+1)

theorem binom_computation :
  (binom 10 3) * (binom 8 3) = 6720 := by
  sorry

end binom_computation_l242_242379


namespace distribution_ways_l242_242345

theorem distribution_ways (n_problems n_friends : ℕ) (h_problems : n_problems = 6) (h_friends : n_friends = 8) : (n_friends ^ n_problems) = 262144 :=
by
  rw [h_problems, h_friends]
  norm_num

end distribution_ways_l242_242345


namespace determine_erased_number_l242_242203

theorem determine_erased_number 
  (C : Fin 6 → ℕ) -- representing the six circles
  (S : Fin 6 → ℕ) -- representing the six segments
  (h1 : C 0 = S 0 + S 1)
  (h2 : C 1 = S 1 + S 2)
  (h3 : C 2 = S 2 + S 3)
  (h4 : C 3 = S 3 + S 4)
  (h5 : C 4 = S 4 + S 5)
  (h6 : C 5 = S 5 + S 0) :
  ∃ (x : ℕ), ∀ (i : Fin 6), C i = x → x can be determined by using S and {C j | j ≠ i} :=
begin
  -- proof will be here
  sorry
end

end determine_erased_number_l242_242203


namespace initial_water_amount_l242_242681

theorem initial_water_amount 
  (W : ℝ) 
  (evap_rate : ℝ) 
  (days : ℕ) 
  (percentage_evaporated : ℝ) 
  (evap_rate_eq : evap_rate = 0.012) 
  (days_eq : days = 50) 
  (percentage_evaporated_eq : percentage_evaporated = 0.06) 
  (total_evaporated_eq : evap_rate * days = 0.6) 
  (percentage_condition : percentage_evaporated * W = evap_rate * days) 
  : W = 10 := 
  by sorry

end initial_water_amount_l242_242681


namespace inequality_proof_l242_242675

theorem inequality_proof 
  (x y z : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hz : 0 < z) 
  (hxyz : x + y + z = 1) : 
  (x^2 + y^2) / z + (y^2 + z^2) / x + (z^2 + x^2) / y ≥ 2 :=
by
  sorry

end inequality_proof_l242_242675


namespace log_base_3_of_9_cubed_l242_242829

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 := by
  -- Define the known conditions
  have h1 : 9 = 3^2 := by norm_num
  have h2 : log 3 (a^n) = n * log 3 a := sorry  -- Statement of the logarithmic identity
  
  -- Apply the conditions to prove the result
  calc
    log 3 (9^3)
        = log 3 ((3^2)^3) : by rw h1
    ... = log 3 (3^(2 * 3)) : by congr; norm_num
    ... = log 3 (3^6) : by norm_num
    ... = 6 : by rw [h2]; norm_num

end log_base_3_of_9_cubed_l242_242829


namespace factors_60_multiples_of_5_not_3_l242_242482

theorem factors_60_multiples_of_5_not_3 : 
  ∃ n : ℕ, n = 2 ∧ (∀ k : ℕ, k ∣ 60 → k % 5 = 0 → k % 3 ≠ 0 → (∃ count : ℕ, count = n)) :=
begin
  sorry
end

end factors_60_multiples_of_5_not_3_l242_242482


namespace rate_per_meter_for_fencing_l242_242230

-- Conditions
def width : ℝ := 40
def length : ℝ := width + 10
def perimeter : ℝ := 180
def total_cost : ℝ := 1170
def rate := total_cost / perimeter

-- Statement
theorem rate_per_meter_for_fencing :
  rate = 6.5 :=
sorry

end rate_per_meter_for_fencing_l242_242230


namespace log_base_3_of_9_cubed_l242_242923
-- Import the necessary parts of the standard library

-- Define the statement to be proved
theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 :=
by
  sorry

end log_base_3_of_9_cubed_l242_242923


namespace sum_diagonal_l242_242423

-- Definitions for the conditions of the problem
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ i : ℕ, a (i + 1) = a i + d

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ i : ℕ, a (i + 1) = a i * q

def matrix_conditions (a : ℕ → ℕ → ℝ) (d : ℝ) (q : ℝ) :=
  (∀ i : ℕ, is_arithmetic_sequence (λ j, a i (j + 1)) d) ∧
  (∀ j : ℕ, is_geometric_sequence (λ i, a (i + 1) j) q) ∧
  (∀ i j : ℕ, i > 0 ∧ j > 0 → 0 < a i j)

def given_values (a : ℕ → ℕ → ℝ) :=
  a 2 4 = 1 ∧ a 4 2 = 1 / 8 ∧ a 4 3 = 3 / 16

-- Main theorem statement
theorem sum_diagonal (n : ℕ) (a : ℕ → ℕ → ℝ) (d q : ℝ) (h1 : n ≥ 4) 
  (h2 : matrix_conditions a d q) (h3 : given_values a) :
  (Σ i in finset.range n, a (i + 1) (i + 1)) = 2 - 1 / 2^(n-1) - n / 2^n :=
sorry

end sum_diagonal_l242_242423


namespace log_three_nine_cubed_l242_242954

theorem log_three_nine_cubed : ∀ (log : ℕ → ℕ → ℕ), log 3 9 = 2 → log 3 3 = 1 → (∀ (a b n : ℕ), log a (b ^ n) = n * log a b) → log 3 (9 ^ 3) = 6 :=
by
  intros log h1 h2 h3
  sorry

end log_three_nine_cubed_l242_242954


namespace least_x_cos_identity_l242_242549

theorem least_x_cos_identity :
  ∃ x : ℝ, x > 1 ∧ cos (x * real.pi / 180) = cos ((x^2) * real.pi / 180) ∧ x = 18 := 
sorry

end least_x_cos_identity_l242_242549


namespace gcd_8_10_l242_242278

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem gcd_8_10 : Nat.gcd (factorial 8) (factorial 10) = factorial 8 := 
by
  sorry

end gcd_8_10_l242_242278


namespace log_base_three_of_nine_cubed_l242_242879

theorem log_base_three_of_nine_cubed : log 3 (9^3) = 6 :=
by
  have h1 : 9 = 3^2 := by sorry
  have h2 : ∀ (a m n: ℕ), (a^m)^n = a^(m * n) := by 
    intros a m n
    exact pow_mul a m n
  have h3 : ∀ (b n: ℕ), log b (b^n) = n := by sorry
  sorry

end log_base_three_of_nine_cubed_l242_242879


namespace solve_theta_count_l242_242249

theorem solve_theta_count (n : ℕ) : 
  (∀ θ : ℝ, (sin θ)^2 - 1 = 0 ∨ 2 * (sin θ)^2 - 1 = 0 → 0 ≤ θ ∧ θ ≤ 2 * Real.pi) → n = 6 :=
sorry

end solve_theta_count_l242_242249


namespace part_a_part_b_l242_242338

def is_balanced (n : ℕ) (S : Finset ℕ) : Prop :=
  ∀ a ∈ S, ∃ b ∈ S, b ≠ a ∧ (a + b) / 2 ∈ S

theorem part_a (k : ℕ) (h : k > 1) (S : Finset (Fin (2^k))) (hS : S.card > 3 * 2^k / 4) :
  is_balanced (2^k) S := sorry

theorem part_b (k : ℕ) (h : k > 1) :
  ∃ (S : Finset (Fin (2^k))), S.card > 2 * 2^k / 3 ∧ ¬is_balanced (2^k) S := sorry

end part_a_part_b_l242_242338


namespace satisfies_equation_l242_242603

variable (c : ℝ) (x : ℝ)

noncomputable def y : ℝ := Real.log (c + Real.exp x)

theorem satisfies_equation :
  deriv y x = Real.exp (x - y x) :=
sorry

end satisfies_equation_l242_242603


namespace alice_bob_speed_l242_242347

theorem alice_bob_speed (x : ℝ) (h : x = 3 + 2 * Real.sqrt 7) :
  x^2 - 5 * x - 14 = 8 + 2 * Real.sqrt 7 - 5 := by
sorry

end alice_bob_speed_l242_242347


namespace find_d_l242_242172

noncomputable def f (x : ℝ) (c : ℝ) : ℝ := 5 * x + c
noncomputable def g (x : ℝ) (c : ℝ) : ℝ := c * x - 3

theorem find_d (c d : ℝ) (h : ∀ x, f (g x c) c = 15 * x + d) : d = -12 :=
by
  have h1 : ∀ x, f (g x c) c = 5 * (c * x - 3) + c := by intros; simp [f, g]
  have h2 : ∀ x, 5 * (c * x - 3) + c = 5 * c * x + c - 15 := by intros; ring
  specialize h 0
  rw [h1, h2] at h
  sorry

end find_d_l242_242172


namespace triangle_side_difference_l242_242147

theorem triangle_side_difference :
  ∀ (x : ℝ), x + 7 > 10 ∧ x + 10 > 7 ∧ 7 + 10 > x → ∃ (a b : ℝ), (a = 4) ∧ (b = 16) ∧ (b - a = 12) :=
by 
  intros x h,
  let lower_bound := 4,
  let upper_bound := 16,
  have h1 : x > 3 := by linarith,
  have h2 : x < 17 := by linarith,
  have h3 : (∀ z : ℤ, z ≥ lower_bound ∧ z ≤ upper_bound → (z ≥ 4 ∧ z ≤ 16)) := 
    by intros z hz; exact mod_cast hz,
  use [lower_bound, upper_bound],
  split, {refl}, 
  split, {refl},
  norm_num,
sorry

end triangle_side_difference_l242_242147


namespace together_finish_in_12_days_l242_242303

noncomputable def work_rate_a := 1 / 20
noncomputable def work_rate_b := 1 / 30.000000000000007
noncomputable def combined_work_rate := work_rate_a + work_rate_b
noncomputable def days_to_complete := 1 / combined_work_rate

theorem together_finish_in_12_days : days_to_complete = 12 := by
  sorry

end together_finish_in_12_days_l242_242303


namespace log_three_nine_cubed_l242_242963

theorem log_three_nine_cubed : ∀ (log : ℕ → ℕ → ℕ), log 3 9 = 2 → log 3 3 = 1 → (∀ (a b n : ℕ), log a (b ^ n) = n * log a b) → log 3 (9 ^ 3) = 6 :=
by
  intros log h1 h2 h3
  sorry

end log_three_nine_cubed_l242_242963


namespace program_output_l242_242122

def initial_state : Nat × Nat := (1, 1)

def loop_step (state : Nat × Nat) : Nat × Nat :=
  (state.1 + 9, state.2 + 1)

def loop (state : Nat × Nat) : Nat :=
  if state.2 = 3 then state.1
  else loop (loop_step state)

theorem program_output :
  loop initial_state = 19 :=
by
  sorry

end program_output_l242_242122


namespace simplify_trig_expression_l242_242201

noncomputable def tan (x : ℝ) : ℝ := Real.sin x / Real.cos x

theorem simplify_trig_expression :
  (tan (20 * Real.pi / 180) + tan (30 * Real.pi / 180) + tan (60 * Real.pi / 180) + tan (70 * Real.pi / 180)) / Real.sin (10 * Real.pi / 180) =
  1 / (2 * Real.sin (10 * Real.pi / 180) ^ 2 * Real.cos (20 * Real.pi / 180)) + 4 / (Real.sqrt 3 * Real.sin (10 * Real.pi / 180)) :=
by
  sorry

end simplify_trig_expression_l242_242201


namespace log_pow_evaluation_l242_242910

-- Define the condition that 9 is equal to 3^2
def nine_eq_three_squared : Prop := 9 = 3^2

-- The main theorem that we need to prove
theorem log_pow_evaluation (h : nine_eq_three_squared) : log 3 (9^3) = 6 :=
  sorry

end log_pow_evaluation_l242_242910


namespace minimum_unions_needed_l242_242592

def country := Fin 100 -- There are 100 countries
def union (A : Type) := Set A -- A union is a set of countries

noncomputable def min_unions := Nat.MinSorry 6 -- We know minimum number of unions needed is 6

theorem minimum_unions_needed : 
  ∃ U : Set (union country), 
    (∀ u ∈ U, u.card ≤ 50) ∧          -- Each union has at most 50 countries
    (∀ c1 c2 : country, c1 ≠ c2 → ∃ u ∈ U, c1 ∈ u ∧ c2 ∈ u) ∧ -- Every two countries share at least one union
    U.card = min_unions :=             -- The number of unions is minimized (min_unions = 6)
sorry

end minimum_unions_needed_l242_242592


namespace complex_addition_l242_242269

-- Define the complex numbers c and d
def c : ℂ := 3 + 2 * complex.I
def d : ℂ := -2 - complex.I

-- State the theorem
theorem complex_addition : 3 * c + 4 * d = 1 + 2 * complex.I := 
by 
  sorry

end complex_addition_l242_242269


namespace rhombus_area_l242_242255

theorem rhombus_area (a : ℝ) (θ : ℝ) (h₁ : a = 4) (h₂ : θ = π / 4) : 
    (a * a * Real.sin θ) = 16 :=
by
    have s1 : Real.sin (π / 4) = Real.sqrt 2 / 2 := Real.sin_pi_div_four
    rw [h₁, h₂, s1]
    have s2 : 4 * 4 * (Real.sqrt 2 / 2) = 16 := by norm_num
    exact s2

end rhombus_area_l242_242255


namespace fund_remaining_after_trip_l242_242567

-- Define the conditions in Lean 4
variables (initial_fund : ℕ) (student_contribution : ℕ) (num_students : ℕ) (cost_per_student : ℕ)

-- Set the specific values for the conditions
def initial_fund := 50
def student_contribution := 5
def num_students := 20
def cost_per_student := 7

-- Statement of the problem in Lean 4
theorem fund_remaining_after_trip :
  initial_fund + (num_students * student_contribution) - (num_students * cost_per_student) = 10 := 
by 
  sorry

end fund_remaining_after_trip_l242_242567


namespace gcd_fact8_fact10_l242_242272

-- Define the factorials
def fact : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * fact n

-- State the problem conditions
theorem gcd_fact8_fact10 : gcd (fact 8) (fact 10) = fact 8 := by
  sorry

end gcd_fact8_fact10_l242_242272


namespace arrangement_ends_arrangement_restrictions_arrangement_next_to_each_other_arrangement_descending_boys_l242_242634

-- Number of different arrangements when Person A and B must stand at the two ends
theorem arrangement_ends (A B : Type) (remaining : Fin 5) :
  arrangement_ends A B remaining = 240 := 
sorry

-- Number of different arrangements when Person A cannot stand at the left end, and person B cannot stand at the right end
theorem arrangement_restrictions (A B : Type) (remaining : Fin 5) :
  arrangement_restrictions A B remaining = 3720 :=
sorry

-- Number of different arrangements when Person A and B must stand next to each other
theorem arrangement_next_to_each_other (A B : Type) (remaining : Fin 6) :
  arrangement_next_to_each_other A B remaining = 1440 :=
sorry

-- Number of different arrangements when the 3 boys are arranged from left to right in descending order of height
theorem arrangement_descending_boys (boys girls : Fin 7) :
  arrangement_descending_boys boys girls = 840 :=
sorry

end arrangement_ends_arrangement_restrictions_arrangement_next_to_each_other_arrangement_descending_boys_l242_242634


namespace distance_from_origin_l242_242704

open Complex

noncomputable def distance_after_n_moves (n : ℕ) (θ : ℝ) : ℝ :=
  abs ((sin (n * θ / 2)) / (sin (θ / 2)))

theorem distance_from_origin (n : ℕ) (θ : ℝ) :
  (∑ k in Finset.range n, exp (k * θ * I)).abs = distance_after_n_moves n θ :=
sorry

end distance_from_origin_l242_242704


namespace calculate_rhombus_area_l242_242262

def rhombus_adj_sides_eq4_angle_eq_45_area : Prop :=
  ∀ (A B C D : ℝ), 
  ∃ (AB CD : ℝ) (angle_Dab : ℝ) (area : ℝ), 
  AB = 4 ∧ CD = 4 ∧ angle_Dab = 45 * (π / 180) ∧ ( area = 8 * √2 )

theorem calculate_rhombus_area :
  rhombus_adj_sides_eq4_angle_eq_45_area :=
by
  sorry

end calculate_rhombus_area_l242_242262


namespace evaluate_expression_l242_242964

theorem evaluate_expression (b : ℚ) (h : b = 4 / 3) :
  (6 * b ^ 2 - 17 * b + 8) * (3 * b - 4) = 0 :=
by 
  -- Proof goes here
  sorry

end evaluate_expression_l242_242964


namespace focal_point_on_AO_eccentricity_range_l242_242311

section EllipseProof

variables (a b : ℝ) (h : a > b > 0)
variables (A : (ℝ × ℝ)) (B : (ℝ × ℝ)) (F : (ℝ × ℝ))
variables (P Q O : (ℝ × ℝ))
variables (c e : ℝ)

-- Definitions from conditions
def ellipse_eqn := ∀ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1
def endpoint_A := A = (a, 0)
def endpoint_B := B = (0, b)
def focal_point_F := F = (c, 0) ∨ F = (-c, 0)
def symmetric_PQ := P = (a * cos θ, b * sin θ) ∧ Q = (-a * cos θ, -b * sin θ)
def dot_product_condition := (F.1, F.2 - P.2) • (F.1, F.2 - Q.2) + (F.1 - A.1, F.2 - A.2) • (F.1 - B.1, F.2 - B.2) = (a^2 + b^2)

-- Proof 1 Statement
theorem focal_point_on_AO :  
  ∃ F, ∀ (P Q : ℝ × ℝ), ellipse_eqn P.1 P.2 ∧ ellipse_eqn Q.1 Q.2 ∧ symmetric_PQ → 
  dot_product_condition → 
  F = (c, 0) ∨ F = (-c, 0) := 
sorry

-- Proof 2 Statement
theorem eccentricity_range : 
  let e := c / a in 
  3 / 4 ≤ e ∧ e ≤ (sqrt 37 - 1) / 6 :=
sorry

end EllipseProof

end focal_point_on_AO_eccentricity_range_l242_242311


namespace sum_possible_x_coordinates_l242_242707

-- Define the vertices of the parallelogram
def A := (1, 2)
def B := (3, 8)
def C := (4, 1)

-- Definition of what it means to be a fourth vertex that forms a parallelogram
def is_fourth_vertex (D : ℤ × ℤ) : Prop :=
  (D = (6, 7)) ∨ (D = (2, -5)) ∨ (D = (0, 9))

-- The sum of possible x-coordinates for the fourth vertex
def sum_x_coordinates : ℤ :=
  6 + 2 + 0

theorem sum_possible_x_coordinates :
  (∃ D, is_fourth_vertex D) → sum_x_coordinates = 8 :=
by
  -- Sorry is used to skip the detailed proof steps
  sorry

end sum_possible_x_coordinates_l242_242707


namespace angle_A_given_conditions_l242_242431

variables {A B C M : Type} [point : ℕ] [triangle : ℕ]

noncomputable def is_acute (ΔABC : triangle) : Prop :=
∃ A B C : point, ∀ (a b c : ℕ), a < 90 ∧ b < 90 ∧ c < 90 ∧ a + b + c = 180

def orthocenter (A B C M : point) : Prop := ∃ ΔABC : triangle, ∀ (a b c m : ℕ),
  -- Here, the 'orthocenter' definition would involve detailed geometric relations
  sorry

def heights_equal (A B C M : point) (AM BC : ℕ) : Prop :=
  AM = BC

theorem angle_A_given_conditions
  (h1 : is_acute ΔABC)
  (h2 : orthocenter A B C M)
  (h3 : heights_equal A B C M AM BC) :
  angle A B C = 45 :=
sorry

end angle_A_given_conditions_l242_242431


namespace sum_even_integers_200_to_400_l242_242282

theorem sum_even_integers_200_to_400 : 
  let seq := list.range' 200 ((400 - 200) / 2 + 1)
  in seq.filter (λ n, n % 2 = 0) = list.range' 200 101 ∧ 
     seq.sum = 30300 := 
by
  sorry

end sum_even_integers_200_to_400_l242_242282


namespace check_propositions_l242_242455

section Propositions

-- Define the first proposition
def prop1 (P : ℝ × ℝ) : Prop :=
  let F1 := (-2, 0)
  let F2 := (2, 0)
  dist P F1 + dist P F2 = 4 → isEllipse P

-- Define what it means for P to be on an ellipse (placeholder since false)
def isEllipse (P : ℝ × ℝ) : Prop := false

-- Define the second proposition
def prop2 : Prop :=
  let P : ℝ × ℝ := sorry
  let F : ℝ × ℝ := sorry
  let A := (2, 1)
  P ∈ parabola ∧ minDistSum P F A 3

-- Define the parabola y^2 = 4x
def parabola (P : ℝ × ℝ) : Prop := P.snd ^ 2 = 4 * P.fst

-- Define the minimum distance sum condition
def minDistSum (P F A : ℝ × ℝ) (minSum : ℝ) : Prop :=
  dist P F + dist P A = minSum

-- Define the third proposition
def prop3 (f : ℝ → ℝ) : Prop :=
  ∀ x ∈ set.Ioo (real.pi) (2 * real.pi), monotonic (deriv f x)

-- Define the function f(x) = x * cos(x) - sin(x)
noncomputable def f (x : ℝ) : ℝ := x * real.cos x - real.sin x

-- Define the derivative of the function
def deriv (f : ℝ → ℝ) (x : ℝ) : ℝ := -x * real.sin x

-- Define the monotonic predicate
def monotonic (f : ℝ → ℝ) (x : ℝ) : Prop := ∀ x y, x < y → f x ≤ f y

-- Define the fourth proposition
def prop4 (f : ℝ → ℝ) : Prop :=
  f'' 1 = 0 ∧ (∀ x, (x - 1) * deriv f x > 0) → f 0 + f 2 > 2 * f 1

-- Define the second derivative placeholder (for simplicity here)
noncomputable def f'' (x : ℝ) : ℝ := 0

-- Final theorem composing all propositions
theorem check_propositions : (¬ prop1 ∧ prop2 ∧ prop3 f ∧ prop4 f) := by
  unfold prop1 prop2 prop3 prop4 f parabola minDistSum
  sorry -- The proof is skipped
end

end check_propositions_l242_242455


namespace log_pow_evaluation_l242_242908

-- Define the condition that 9 is equal to 3^2
def nine_eq_three_squared : Prop := 9 = 3^2

-- The main theorem that we need to prove
theorem log_pow_evaluation (h : nine_eq_three_squared) : log 3 (9^3) = 6 :=
  sorry

end log_pow_evaluation_l242_242908


namespace sheila_hours_l242_242200

theorem sheila_hours :
  ∀ (H : ℕ),
    let h_tu_th := 6 * 2 in
    let earnings_tu_th := 12 * h_tu_th in
    let total_weekly_earnings := 432 in
    let earnings_mw_f := total_weekly_earnings - earnings_tu_th in
    let hourly_rate := 12 in
    H * hourly_rate = earnings_mw_f → H = 24 := by
  intros H h_tu_th earnings_tu_th total_weekly_earnings earnings_mw_f hourly_rate
  sorry

end sheila_hours_l242_242200


namespace incorrect_conclusion_d_l242_242354

theorem incorrect_conclusion_d {x y : ℝ} (height : ℝ) (weight : ℝ) (h : ∀ x, y = 0.85 * x - 85.71) :
  height = 170 → ¬ (weight = y) :=
by {
  assume h1,
  have h2 : h height = weight := h height,
  sorry
}

end incorrect_conclusion_d_l242_242354


namespace washing_machine_capacity_l242_242341

-- Definitions of the conditions
def total_pounds_per_day : ℕ := 200
def number_of_machines : ℕ := 8

-- Main theorem to prove the question == answer given the conditions
theorem washing_machine_capacity :
  total_pounds_per_day / number_of_machines = 25 :=
by
  sorry

end washing_machine_capacity_l242_242341


namespace smallest_positive_integer_n_l242_242654

noncomputable def smallest_n : ℕ :=
  Inf { n : ℕ | ∃ (k m : ℕ), 3 * n = (3 * k) ^ 2 ∧ 5 * n = (5 * m) ^ 3 }

theorem smallest_positive_integer_n : smallest_n = 1875 :=
sorry

end smallest_positive_integer_n_l242_242654


namespace determine_a_2013_l242_242980

noncomputable def sequence (a : ℕ → ℤ) (p : ℕ) [Fact (Nat.Prime p)] (k : ℕ) : Prop :=
  a (p * k + 1) = p * a k - 3 * a p + 13

theorem determine_a_2013 (a : ℕ → ℤ)
  (h : ∀ p k, Nat.Prime p → sequence a p k) : a 2013 = 13 :=
sorry

end determine_a_2013_l242_242980


namespace cubic_sum_identity_l242_242093

theorem cubic_sum_identity (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 :=
sorry

end cubic_sum_identity_l242_242093


namespace smallest_n_for_square_and_cube_l242_242651

theorem smallest_n_for_square_and_cube (n : ℕ) 
  (h1 : ∃ m : ℕ, 3 * n = m^2) 
  (h2 : ∃ k : ℕ, 5 * n = k^3) : 
  n = 675 :=
  sorry

end smallest_n_for_square_and_cube_l242_242651


namespace sequence_problems_l242_242518
open Nat

-- Define the arithmetic sequence conditions
def arith_seq_condition_1 (a : ℕ → ℤ) : Prop :=
  a 2 + a 7 = -23

def arith_seq_condition_2 (a : ℕ → ℤ) : Prop :=
  a 3 + a 8 = -29

-- Define the geometric sequence condition
def geom_seq_condition (a b : ℕ → ℤ) (c : ℤ) : Prop :=
  ∀ n, a n + b n = c^(n - 1)

-- Define the arithmetic sequence formula
def arith_seq_formula (a : ℕ → ℤ) : Prop :=
  ∀ n, a n = -3 * n + 2

-- Define the sum of the first n terms of the sequence b_n
def sum_b_n (b : ℕ → ℤ) (S_n : ℕ → ℤ) (c : ℤ) : Prop :=
  (c = 1 → ∀ n, S_n n = (3 * n^2 + n) / 2) ∧
  (c ≠ 1 → ∀ n, S_n n = (n * (3 * n - 1)) / 2 + ((1 - c^n) / (1 - c)))

-- Define the main theorem
theorem sequence_problems (a b : ℕ → ℤ) (c : ℤ) (S_n : ℕ → ℤ) :
  arith_seq_condition_1 a →
  arith_seq_condition_2 a →
  geom_seq_condition a b c →
  arith_seq_formula a ∧ sum_b_n b S_n c :=
by
  -- Proofs for the conditions to the formula
  sorry

end sequence_problems_l242_242518


namespace probability_of_both_selected_l242_242309

theorem probability_of_both_selected (pX pY : ℚ) (hX : pX = 1/7) (hY : pY = 2/5) : 
  pX * pY = 2 / 35 :=
by {
  sorry
}

end probability_of_both_selected_l242_242309


namespace find_face_value_l242_242613

-- Define the conditions as variables in Lean
variable (BD TD FV : ℝ)
variable (hBD : BD = 36)
variable (hTD : TD = 30)
variable (hRel : BD = TD + (TD * BD / FV))

-- State the theorem we want to prove
theorem find_face_value (BD TD : ℝ) (FV : ℝ) 
  (hBD : BD = 36) (hTD : TD = 30) (hRel : BD = TD + (TD * BD / FV)) : 
  FV = 180 := 
  sorry

end find_face_value_l242_242613


namespace binomial_30_3_l242_242370

theorem binomial_30_3 : nat.choose 30 3 = 4060 := by
  sorry

end binomial_30_3_l242_242370


namespace relationship_between_y_values_l242_242237

def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + |b| * x + c

theorem relationship_between_y_values
  (a b c y1 y2 y3 : ℝ)
  (h1 : quadratic_function a b c (-14 / 3) = y1)
  (h2 : quadratic_function a b c (5 / 2) = y2)
  (h3 : quadratic_function a b c 3 = y3)
  (axis_symmetry : -(|b| / (2 * a)) = -1)
  (h_pos : 0 < a) :
  y2 < y1 ∧ y1 < y3 :=
by
  sorry

end relationship_between_y_values_l242_242237


namespace student_weight_l242_242500

-- Define the weights of the student and sister
variables (S R : ℝ)

-- Define the conditions given in the problem
def condition1 : Prop := S - 5 = 1.25 * R
def condition2 : Prop := S + R = 104

-- The theorem we want to prove
theorem student_weight (h1 : condition1 S R) (h2 : condition2 S R) : S = 60 := 
by
  sorry

end student_weight_l242_242500


namespace tom_read_in_five_months_l242_242640

def books_in_may : ℕ := 2
def books_in_june : ℕ := 6
def books_in_july : ℕ := 12
def books_in_august : ℕ := 20
def books_in_september : ℕ := 30

theorem tom_read_in_five_months : 
  books_in_may + books_in_june + books_in_july + books_in_august + books_in_september = 70 := by
  sorry

end tom_read_in_five_months_l242_242640


namespace log_base_3_of_9_cubed_l242_242891

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 :=
by assumptions
  have h1 : 9 = 3^2 := sorry,
  have h2 : 9^3 = (3^2)^3 := sorry,
  have h3 : (3^2)^3 = 3^(2*3) := sorry,
  have h4 : 3^(2*3) = 3^6 := sorry,
  have h5 : log 3 (3^6) = 6 := by apply log_pow,
  sorry

end log_base_3_of_9_cubed_l242_242891


namespace find_x_eq_5_over_3_l242_242390

def g (x : ℝ) : ℝ := 4 * x - 5

def g_inv (y : ℝ) : ℝ := (y + 5) / 4

theorem find_x_eq_5_over_3 (x : ℝ) (hx : g x = g_inv x) : x = 5 / 3 :=
by
  sorry

end find_x_eq_5_over_3_l242_242390


namespace binom_multiplication_l242_242380

open BigOperators

noncomputable def choose_and_multiply (n k m l : ℕ) : ℕ :=
  Nat.choose n k * Nat.choose m l

theorem binom_multiplication : choose_and_multiply 10 3 8 3 = 6720 := by
  sorry

end binom_multiplication_l242_242380


namespace g_at_2023_l242_242547

noncomputable def g : ℝ → ℝ := sorry

-- Conditions
axiom g_pos : ∀ x > 0, g x > 0
axiom g_func_eq : ∀ x y > 0, x > y → g (x - y) = sqrt (g (x * y) + 3)

-- Goal
theorem g_at_2023 : g 2023 = 3 := sorry

end g_at_2023_l242_242547


namespace log_base_3_of_9_cubed_l242_242893

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 :=
by assumptions
  have h1 : 9 = 3^2 := sorry,
  have h2 : 9^3 = (3^2)^3 := sorry,
  have h3 : (3^2)^3 = 3^(2*3) := sorry,
  have h4 : 3^(2*3) = 3^6 := sorry,
  have h5 : log 3 (3^6) = 6 := by apply log_pow,
  sorry

end log_base_3_of_9_cubed_l242_242893


namespace factorial_division_l242_242739

theorem factorial_division : (10! / 9! : Nat) = 10 := by
  sorry

end factorial_division_l242_242739


namespace log_base_3_of_9_cubed_l242_242816

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 := by
  sorry

end log_base_3_of_9_cubed_l242_242816


namespace min_value_z_l242_242503

theorem min_value_z (z : ℂ) (h : |z + Complex.I| + |z - Complex.I| = 2) : ∃ (y : ℝ), -1 ≤ y ∧ y ≤ 1 ∧ z = Complex.I * y ∧ (∀ (w : ℂ), (w = Complex.I * y) → |w + Complex.I + 1| ≥ 1 ∧ |w + Complex.I + 1| = 1) :=
  sorry

end min_value_z_l242_242503


namespace cubic_sum_identity_l242_242095

theorem cubic_sum_identity (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 :=
sorry

end cubic_sum_identity_l242_242095


namespace problem_1_problem_2_l242_242472

-- Definitions of sets A and B
def A : Set ℝ := { x : ℝ | x^2 - 2 * x - 3 ≤ 0 }
def B (m : ℝ) : Set ℝ := { x : ℝ | m - 1 ≤ x ∧ x ≤ m + 1 }

-- Problem 1: Prove that if A ∩ B = [1, 3], then m = 2
theorem problem_1 (m : ℝ) (h : (A ∩ B m) = {x : ℝ | 1 ≤ x ∧ x ≤ 3}) : m = 2 :=
sorry

-- Problem 2: Prove that if A ⊆ complement ℝ B m, then m > 4 or m < -2
theorem problem_2 (m : ℝ) (h : A ⊆ { x : ℝ | x < m - 1 ∨ x > m + 1 }) : m > 4 ∨ m < -2 :=
sorry

end problem_1_problem_2_l242_242472


namespace log_three_pow_nine_pow_three_eq_six_l242_242936

open Real

noncomputable def eval_log_three_pow_nine_pow_three : ℝ :=
  log 3 (9^3)

theorem log_three_pow_nine_pow_three_eq_six :
  eval_log_three_pow_nine_pow_three = 6 := by
  sorry

end log_three_pow_nine_pow_three_eq_six_l242_242936


namespace festival_second_day_attendance_l242_242701

-- Define the conditions
variables (X Y Z A : ℝ)
variables (h1 : A = 2700) (h2 : Y = X / 2) (h3 : Z = 3 * X) (h4 : A = X + Y + Z)

-- Theorem stating the question and the conditions result in the correct answer
theorem festival_second_day_attendance (X Y Z A : ℝ) 
  (h1 : A = 2700) (h2 : Y = X / 2) (h3 : Z = 3 * X) (h4 : A = X + Y + Z) : 
  Y = 300 :=
sorry

end festival_second_day_attendance_l242_242701


namespace totalNominalIncomeIsCorrect_l242_242582

def nominalIncomeForMonth (principal rate divisor months : ℝ) : ℝ :=
  principal * ((1 + rate / divisor) ^ months - 1)

def totalNominalIncomeForSixMonths : ℝ :=
  nominalIncomeForMonth 8700 0.06 12 6 +
  nominalIncomeForMonth 8700 0.06 12 5 +
  nominalIncomeForMonth 8700 0.06 12 4 +
  nominalIncomeForMonth 8700 0.06 12 3 +
  nominalIncomeForMonth 8700 0.06 12 2 +
  nominalIncomeForMonth 8700 0.06 12 1

theorem totalNominalIncomeIsCorrect : totalNominalIncomeForSixMonths = 921.15 := by
  sorry

end totalNominalIncomeIsCorrect_l242_242582


namespace find_s_l242_242403

theorem find_s (s : ℝ) (h : 4 * log 3 s = log 3 (4 * s^2)) : s = 2 :=
sorry

end find_s_l242_242403


namespace exists_pairs_with_ratio_condition_l242_242599

theorem exists_pairs_with_ratio_condition :
  ∀ (S : set ℝ), S.card = 2000 →
  ∃ (a b c d : ℝ), a ≠ b ∧ c ≠ d ∧ (a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S) ∧ (a > b ∧ c > d) ∧ ((a ≠ c ∨ b ≠ d) ∧ (|((a - b) / (c - d)) - 1| < 1 / 100000)) :=
by
  sorry

end exists_pairs_with_ratio_condition_l242_242599


namespace rhombus_area_l242_242259

-- Given a rhombus with sides of 4 cm and an included angle of 45 degrees,
-- prove that the area is 8 square centimeters.

theorem rhombus_area :
  ∀ (s : ℝ) (α : ℝ), s = 4 ∧ α = π / 4 → 
    let area := s * s * Real.sin α in
    area = 8 := 
by
  intros s α h
  sorry

end rhombus_area_l242_242259


namespace pulley_centers_distance_l242_242642

-- Define the problem parameters
def radius1 : ℝ := 15
def radius2 : ℝ := 5
def distance_contact : ℝ := 26

-- Define the problem question
def center_distance : ℝ := 2 * Real.sqrt (194)

theorem pulley_centers_distance : 
  ∀ (r1 r2 d : ℝ), r1 = radius1 → r2 = radius2 → d = distance_contact → 
  (Real.sqrt (d ^ 2 + (r1 - r2) ^ 2) = center_distance) := 
by
  intros r1 r2 d hr1 hr2 hd
  rw [hr1, hr2, hd]
  sorry

end pulley_centers_distance_l242_242642


namespace log_pow_evaluation_l242_242914

-- Define the condition that 9 is equal to 3^2
def nine_eq_three_squared : Prop := 9 = 3^2

-- The main theorem that we need to prove
theorem log_pow_evaluation (h : nine_eq_three_squared) : log 3 (9^3) = 6 :=
  sorry

end log_pow_evaluation_l242_242914


namespace find_other_number_l242_242670

-- Defining the conditions
def A : ℕ := 210
def LCM : ℕ := 2310
def HCF : ℕ := 47

-- Proving the main statement
theorem find_other_number (B : ℕ) 
  (h1 : nat.lcm A B = LCM)
  (h2 : nat.gcd A B = HCF) :
  B = 517 :=
by
  sorry

end find_other_number_l242_242670


namespace log_base_3_l242_242845

theorem log_base_3 (log_identity : ∀ (b a c : ℝ), log b (a^c) = c * log b a) :
  log 3 (9^3) = 6 :=
by
  have : 9 = 3^2 := by sorry
  have : log 3 3 = 1 := by sorry
  exact (log_identity 3 (3^2) 3).trans sorry

end log_base_3_l242_242845


namespace problem_part1_problem_part2_l242_242429

-- Define the conditions for sequences
def S (n : ℕ) : ℕ := 3 * n^2 + 8 * n
def a (n : ℕ) : ℕ := S n - S (n - 1)
def b (n : ℕ) := 3 * n + 1
def c (n : ℕ) := (a n + 1)^(n + 1) / (b n + 2)^n
def T (n : ℕ) : ℕ := 3 * n * 2^(n + 2)

-- Statement for the two problems
theorem problem_part1 (n : ℕ) : b (n) = 3 * n + 1 := sorry

theorem problem_part2 (n : ℕ) : T n = ∑ i in finset.range (n + 1), c i := sorry

end problem_part1_problem_part2_l242_242429


namespace cos_double_angle_eq_l242_242538

theorem cos_double_angle_eq (A1 A2 A3 A4 A5 A6 A7 A8 M1 M3 M5 M7 B1 B3 B5 B7 : Point)
  (length A1 A2 = L) (is_regular_octagon A1 A2 A3 A4 A5 A6 A7 A8)
  (midpoints :=
    (M1 = midpoint A1 A2) ∧ 
    (M3 = midpoint A3 A4) ∧ 
    (M5 = midpoint A5 A6) ∧ 
    (M7 = midpoint A7 A8))
  (rays :=
    (perpendicular (ray_from_to M1 B1) (ray_from_to M3 B3)) ∧ 
    (perpendicular (ray_from_to M3 B3) (ray_from_to M5 B5)) ∧ 
    (perpendicular (ray_from_to M5 B5) (ray_from_to M7 B7)) ∧
    (perpendicular (ray_from_to M7 B7) (ray_from_to M1 B1)))
  (equal_distances := (dist B1 B3 = dist A1 A2))
  : ∃ m n : ℕ, m - sqrt n = cos (2 * (angle A3 M3 B1)) ∧ (m + n = 37) := 
sorry

end cos_double_angle_eq_l242_242538


namespace xy_cubed_identity_l242_242099

namespace ProofProblem

theorem xy_cubed_identity (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 :=
by
  sorry

end ProofProblem

end xy_cubed_identity_l242_242099


namespace derivative_periodic_and_even_l242_242497

noncomputable def is_periodic {α : Type*} [add_group α] [topological_space α] 
  (f : α → α) (T : α) : Prop := ∀ x, f (x + T) = f x

noncomputable def is_odd {α : Type*} [has_neg α] (f : α → α) : Prop := ∀ x, f (-x) = -f x

noncomputable def is_even {α : Type*} [has_neg α] (f : α → α) : Prop := ∀ x, f (-x) = f x

theorem derivative_periodic_and_even 
  {α : Type*} [add_group α] [topological_space α] 
  (f : α → α) (T : α) 
  (h_periodic : is_periodic f T) (h_odd : is_odd f) 
  : is_periodic (deriv f) T ∧ is_even (deriv f) :=
by
  sorry

end derivative_periodic_and_even_l242_242497


namespace vector_computation_l242_242476

noncomputable def a : ℝ × ℝ := (1, 2)
noncomputable def b (m : ℝ) : ℝ × ℝ := (2, -m)
noncomputable def m : ℝ := 1 -- proven from a ⋅ b = 0

theorem vector_computation 
  (h1 : a = (1, 2))
  (h2 : b m = (2, -m))
  (h3 : m = 1)
  (h4 : (a.1 * b m.1 + a.2 * b m.2) = 0) : 3 • a + 2 • b m = (7, 4) := by
  sorry

end vector_computation_l242_242476


namespace incircle_circumradius_half_l242_242551

noncomputable def orthogonal (circle1 circle2 : Circle) : Prop :=
sorry

noncomputable def passes_through (circle : Circle) (point1 point2 : Point) : Prop :=
sorry

noncomputable def intersection_points (circle1 circle2 : Circle) : Set Point :=
sorry

theorem incircle_circumradius_half {ABC : Triangle} (Ω : Circle)
  (Ω_a Ω_b Ω_c : Circle)
  (r : ℝ) -- radius of Ω
  (A' B' C' : Point) :
  (Ω.radius = r) →
  (orthogonal Ω Ω_a) → (orthogonal Ω Ω_b) → (orthogonal Ω Ω_c) →
  (passes_through Ω_a B C) → (passes_through Ω_b A C) → (passes_through Ω_c A B) →
  (intersection_points Ω_a Ω_b = {C'}) →
  (intersection_points Ω_b Ω_c = {A'}) →
  (intersection_points Ω_c Ω_a = {B'}) →
  (circumcircle_radius A' B' C' = r / 2) := 
sorry

end incircle_circumradius_half_l242_242551


namespace elephant_weight_l242_242248

def kg_to_pound (kg : ℕ) : ℝ := kg / 0.4536

theorem elephant_weight :
  kg_to_pound 1200 ≈ 2646 :=
by sorry

end elephant_weight_l242_242248


namespace log_base_3_of_9_cubed_l242_242779

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 :=
by
  -- Given conditions:
  -- 1. 9 = 3^2
  -- 2. log_b (a^n) = n * log_b a
  sorry

end log_base_3_of_9_cubed_l242_242779


namespace new_concentration_of_mixture_l242_242304

/--
Problem:
A vessel of capacity 2 liters has 40% alcohol 
and another vessel of capacity 6 liters has 60% alcohol. 
The total liquid of 8 liters was poured out in a vessel of capacity 10 liters 
and the rest part of the vessel was filled with water. 
What is the new concentration of the mixture?

Given conditions:
- Vessel 1 has a capacity of 2 liters with 40% alcohol.
- Vessel 2 has a capacity of 6 liters with 60% alcohol.
- The total liquid from both vessels (8 liters) is poured into a 10-liter vessel, and the remaining volume is filled with water.

Prove that the new concentration of the mixture is equal to 44%.
-/

theorem new_concentration_of_mixture 
    (V1 V2 : ℝ) (A1 A2 : ℝ)
    (V1_eq : V1 = 2) (A1_eq : A1 = 0.4) 
    (V2_eq : V2 = 6) (A2_eq : A2 = 0.6) :
    let total_volume := V1 + V2 + (10 - (V1 + V2))
    let total_alcohol := (A1 * V1) + (A2 * V2)
    total_alcohol / total_volume = 0.44 :=
by {
  have h₁ : V1 = 2 := V1_eq,
  have h₂ : A1 = 0.4 := A1_eq,
  have h₃ : V2 = 6 := V2_eq,
  have h₄ : A2 = 0.6 := A2_eq,
  let total_volume := V1 + V2 + (10 - (V1 + V2)),
  let total_alcohol := (A1 * V1) + (A2 * V2),
  have h₅ : total_volume = 10,
  {
    rw [h₁, h₃],
    calc 2 + 6 + (10 - (2 + 6)) = 2 + 6 + (10 - 8)    : rfl
    ...                        = 2 + 6 + 2          : rfl
    ...                        = 10                 : rfl
  },
  have h₆ : total_alcohol = 4.4,
  {
    rw [h₁, h₂, h₃, h₄],
    calc (0.4 * 2) + (0.6 * 6) = 0.8 + 3.6   : rfl
    ...                       = 4.4         : rfl
  },
  rw [h₅, h₆],
  calc 4.4 / 10 = 0.44 : rfl
}

end new_concentration_of_mixture_l242_242304


namespace dot_product_property_l242_242477

variables {ι : Type*}
variables (a b c : ι → ℝ)

-- Definitions from conditions
def parallel (a b : ι → ℝ) : Prop := ∃ (λ : ℝ), ∀ i, b i = λ * a i
def perpendicular (a c : ι → ℝ) : Prop := ∑ i, a i * c i = 0

-- Main theorem to prove the desired relationship
theorem dot_product_property
  (ha_parallel_b : parallel a b)
  (ha_perp_c : perpendicular a c) :
  (∑ i, c i * (a i + 2 * b i)) = 0 := by
  sorry

end dot_product_property_l242_242477


namespace equilateral_triangle_circumcircle_area_l242_242686

theorem equilateral_triangle_circumcircle_area (side_length : ℝ) (h : side_length = 4) 
: ∃ (π : ℝ), (let r := 2 * (side_length / (2 * Real.sqrt 3)))
    area = π * r^2 := 
    ∃ (π : ℝ), (π * (4 * Rat.sqrt 3 / 3)^2 = 16 * π / 3)
      sorry

end equilateral_triangle_circumcircle_area_l242_242686


namespace round_to_nearest_tenth_l242_242196

theorem round_to_nearest_tenth (x : ℝ) (h : x = 24.3642) : Real.round (10 * x) / 10 = 24.4 :=
by
  rw [←h]
  apply sorry

end round_to_nearest_tenth_l242_242196


namespace log_base_3_of_9_cubed_l242_242872

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 := by
  -- Assuming the necessary properties of logarithms and exponents
  sorry

end log_base_3_of_9_cubed_l242_242872


namespace binomial_coeffs_not_arith_seq_l242_242597

def binomial (n k : ℕ) : ℕ := Nat.choose n k

def are_pos_integer (n : ℕ) : Prop := n > 0

def is_arith_seq (a b c d : ℕ) : Prop := 
  2 * b = a + c ∧ 2 * c = b + d 

theorem binomial_coeffs_not_arith_seq (n r : ℕ) : 
  are_pos_integer n → are_pos_integer r → n ≥ r + 3 → ¬ is_arith_seq (binomial n r) (binomial n (r+1)) (binomial n (r+2)) (binomial n (r+3)) :=
by
  sorry

end binomial_coeffs_not_arith_seq_l242_242597


namespace min_distance_angle_l242_242062

variables {a b : ℝ^3} [nonzero a] [nonzero b]

def magnitude (v : ℝ^3) := real.sqrt (v.1^2 + v.2^2 + v.3^2)

def angle_between_vectors (a b : ℝ^3) : ℝ :=
  let cos_theta := (a.dot b) / (magnitude a * magnitude b) in
  real.acos cos_theta * 180 / real.pi

theorem min_distance_angle (h : magnitude b = 2) :
  ∃ θ, θ ∈ {60, 120} →
  ∃ t, (magnitude (b - t • a) = real.sqrt 3) :=
sorry

end min_distance_angle_l242_242062


namespace proof_of_math_problem_l242_242159

open_locale euclidean_geometry

noncomputable def math_problem_statement : Prop :=
  ∀ (A B C I D X Y : Point) (ω1 ω2 : Circle),
    (is_incenter I A B C) ∧
    (dist A B = 20) ∧ 
    (dist B C = 15) ∧ 
    (dist B I = 12) ∧ 
    (CI_intersects_ω1_at_D D I C ω1) ∧
    (intersects_at_minor_arc_AC D X ω1) ∧
    (intersects_circumcircle_ω2_at_Y D X Y A I C ω1 ω2) ∧
    (is_right_triangle I D X Y) → 
    (dist I Y = 13)

-- Each definition like is_incenter, dist, CI_intersects_ω1_at_D, 
-- intersects_at_minor_arc_AC, intersects_circumcircle_ω2_at_Y, is_right_triangle 
-- needs to be properly defined based on the conditions stated above.

theorem proof_of_math_problem : math_problem_statement :=
by sorry

end proof_of_math_problem_l242_242159


namespace sum_of_coefficients_l242_242990

noncomputable def binomial_coefficient (n k : ℕ) : ℚ :=
  (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k)))

theorem sum_of_coefficients (n : ℕ) (h1 : 0 < n)
  (h2 : binomial_coefficient n 2 = binomial_coefficient n 7) :
  ((1 - 2 : ℚ) ^ n) = -1 := 
sorry

end sum_of_coefficients_l242_242990


namespace avg_mark_correct_l242_242611

noncomputable def avg_mark_excluded_students 
    (total_students : ℕ) (avg_mark_total : ℕ) 
    (excluded_students : ℕ) (avg_mark_remaining : ℕ) : ℕ :=
  let total_marks := total_students * avg_mark_total in
  let remaining_students := total_students - excluded_students in
  let marks_remaining := remaining_students * avg_mark_remaining in
  let marks_excluded := total_marks - marks_remaining in
  marks_excluded / excluded_students

theorem avg_mark_correct : 
  avg_mark_excluded_students 20 80 5 90 = 50 :=
by
  unfold avg_mark_excluded_students
  /- Now insert the calculations from the solution as a proof step:
      total_marks (T) = 20 * 80 = 1600,
      marks_remaining (15 * 90) = 1350,
      marks_excluded (5 * A) = 1600 - 1350 = 250,
      A = 250 / 5 = 50 
  -/
  sorry

end avg_mark_correct_l242_242611


namespace sum_even_200_to_400_l242_242297

theorem sum_even_200_to_400 : 
  let t₁ := 202
  let tₙ := 398
  let n := 99
  let sum := (t₁ + tₙ) * n / 2 in
  sum = 29700 := 
by
  let t₁ := 202
  let tₙ := 398
  let n := 99
  let sum := (t₁ + tₙ) * n / 2
  show sum = 29700
  sorry

end sum_even_200_to_400_l242_242297


namespace plane_eq_of_proj_l242_242165

noncomputable def vector_eq {α : Type*} [Field α] (v1 v2 : (Fin 3) → α) : Prop :=
∀ i, v1 i = v2 i

noncomputable def proj_w (v w : (Fin 3) → ℝ) : (Fin 3) → ℝ :=
let numerator := (Real.dot_product v w) in
let denominator := (Real.dot_product w w) in
λ i, (numerator / denominator) * w i

theorem plane_eq_of_proj {x y z : ℝ} :
  let w := ![3, -3, 3] in
  let v := ![x, y, z] in
  vector_eq (proj_w v w) ![6, -6, 6] →
  x - y + z = 18 :=
by
  intros
  unfold proj_w at *
  sorry

end plane_eq_of_proj_l242_242165


namespace sum_divisors_divisible_by_24_l242_242174

theorem sum_divisors_divisible_by_24 {n : ℕ} (h : (n + 1) % 24 = 0) :
  ∃ k : ℕ, n = 24 * k - 1 ∧ 
  ∀ d ∈ divisors n, d % 24 = 0 := sorry

end sum_divisors_divisible_by_24_l242_242174


namespace min_value_fraction_l242_242425

theorem min_value_fraction (m n : ℕ) (hmn : m + n = 6) : ∃ (min_val : ℚ), min_val = 3 / 4 ∧ min_val = Real.Inf (Set.image (λ m_n : ℕ × ℕ, 2 / m_n.fst + 1 / (2 * m_n.snd)) {p | p.1 + p.2 = 6}) :=
by
  sorry

end min_value_fraction_l242_242425


namespace pond_water_after_45_days_l242_242691

theorem pond_water_after_45_days :
  let initial_amount := 300
  let daily_evaporation := 1
  let rain_every_third_day := 2
  let total_days := 45
  let non_third_days := total_days - (total_days / 3)
  let third_days := total_days / 3
  let total_net_change := (non_third_days * (-daily_evaporation)) + (third_days * (rain_every_third_day - daily_evaporation))
  let final_amount := initial_amount + total_net_change
  final_amount = 285 :=
by
  sorry

end pond_water_after_45_days_l242_242691


namespace fraction_paint_left_after_third_day_l242_242133

noncomputable def original_paint : ℝ := 2
noncomputable def paint_after_first_day : ℝ := original_paint - (1 / 2 * original_paint)
noncomputable def paint_after_second_day : ℝ := paint_after_first_day - (1 / 4 * paint_after_first_day)
noncomputable def paint_after_third_day : ℝ := paint_after_second_day - (1 / 2 * paint_after_second_day)

theorem fraction_paint_left_after_third_day :
  paint_after_third_day / original_ppaint = 3 / 8 :=
sorry

end fraction_paint_left_after_third_day_l242_242133


namespace sum_even_integers_200_to_400_l242_242292

def is_even (n : ℤ) : Prop := n % 2 = 0
def is_in_range (n : ℤ) : Prop := 200 <= n ∧ n <= 400
def is_valid_even_number (n : ℤ) : Prop := is_even n ∧ is_in_range n

theorem sum_even_integers_200_to_400 :
  ∃ (sum : ℤ), (sum = ∑ n in (finset.filter is_valid_even_number (finset.Icc 200 400)), n) ∧ sum = 29700 :=
begin
  sorry
end

end sum_even_integers_200_to_400_l242_242292


namespace log_evaluation_l242_242760

theorem log_evaluation : ∀ (a : ℕ), a = 3 → log 3 (9^3) = 6 :=
by
  intro a ha
  have h1 : 9 = 3^2 := by sorry
  have h2 : (9^3) = (3^2)^3 := by rw [h1]
  have h3 : (3^2)^3 = 3^6 := by sorry
  have h4 : log 3 (3^6) = 6 := by sorry
  rw [h2, h3, h4]
  exact ha

end log_evaluation_l242_242760


namespace stickers_used_to_decorate_l242_242563

theorem stickers_used_to_decorate (initial: ℕ) (bought: ℕ) (gifted: ℕ) (given: ℕ) (left: ℕ) :
    initial = 20 → bought = 26 → gifted = 20 → given = 6 → left = 2 →
    (initial + bought + gifted - given - left = 58) :=
begin
  intros initial_eq twenty_eq bought_eq gifted_eq given_eq left_eq,
  rw [initial_eq, twenty_eq, bought_eq, gifted_eq, given_eq, left_eq],
  norm_num,
end

end stickers_used_to_decorate_l242_242563


namespace income_increase_l242_242323

-- Definitions based on conditions
def original_price := 1.0
def original_items := 100.0
def discount := 0.10
def increased_sales := 0.15

-- Calculations for new values
def new_price := original_price * (1 - discount)
def new_items := original_items * (1 + increased_sales)
def original_income := original_price * original_items
def new_income := new_price * new_items

-- The percentage increase in income
def percentage_increase := ((new_income - original_income) / original_income) * 100

-- The theorem to prove that the percentage increase in gross income is 3.5%
theorem income_increase : percentage_increase = 3.5 := 
by
  -- This is where the proof would go
  sorry

end income_increase_l242_242323


namespace value_of_a_is_3_l242_242463

def symmetric_about_x1 (a : ℝ) : Prop :=
  ∀ x : ℝ, |x + 1| + |x - a| = |2 - x + 1| + |2 - x - a|

theorem value_of_a_is_3 : symmetric_about_x1 3 :=
sorry

end value_of_a_is_3_l242_242463


namespace log_base_three_of_nine_cubed_l242_242878

theorem log_base_three_of_nine_cubed : log 3 (9^3) = 6 :=
by
  have h1 : 9 = 3^2 := by sorry
  have h2 : ∀ (a m n: ℕ), (a^m)^n = a^(m * n) := by 
    intros a m n
    exact pow_mul a m n
  have h3 : ∀ (b n: ℕ), log b (b^n) = n := by sorry
  sorry

end log_base_three_of_nine_cubed_l242_242878


namespace part_one_part_two_l242_242448

noncomputable def seq_a (n : ℕ) : ℚ := sorry -- Define the arithmetic sequence a_n
noncomputable def seq_b (n : ℕ) : ℚ := sorry -- Define the arithmetic sequence b_n

axiom arithmetic_seq (a_1 a_d b_1 b_d : ℚ) : ∀ n : ℕ, seq_a n = a_1 + n * a_d ∧ seq_b n = b_1 + n * b_d

noncomputable def S (n : ℕ) : ℚ := ∑ i in finset.range (n + 1), seq_a i
noncomputable def T (n : ℕ) : ℚ := ∑ i in finset.range (n + 1), seq_b i

axiom sum_ratio (S T : ℕ → ℚ) : ∀ n : ℕ, S n / T n = (3 * n + 31) / (31 * n + 3)

theorem part_one : ∀ a_1 a_d b_1 b_d : ℚ, ∃ (a b : ℚ), arithmetic_seq a_1 a_d b_1 b_d → seq_a 28 / seq_b 28 = 7 :=
sorry

theorem part_two : ∀ a_1 a_d b_1 b_d : ℚ, ∃ (a b : ℚ), arithmetic_seq a_1 a_d b_1 b_d → 
  (∀ n : ℕ, (seq_b n / seq_a n) ∈ ℤ ↔ n ∈ {1, 18, 35, 154}) :=
sorry

end part_one_part_two_l242_242448


namespace log_pow_evaluation_l242_242909

-- Define the condition that 9 is equal to 3^2
def nine_eq_three_squared : Prop := 9 = 3^2

-- The main theorem that we need to prove
theorem log_pow_evaluation (h : nine_eq_three_squared) : log 3 (9^3) = 6 :=
  sorry

end log_pow_evaluation_l242_242909


namespace find_abs_ab_ac_bc_l242_242738

noncomputable def a : ℂ := sorry
noncomputable def b : ℂ := sorry
noncomputable def c : ℂ := sorry

axiom equilateral_triangle (a b c : ℂ) : (a - b).abs = 24 ∧ (b - c).abs = 24 ∧ (c - a).abs = 24
axiom sum_of_complex (a b c : ℂ) : (a + b + c).abs = 48

theorem find_abs_ab_ac_bc : |a * b + a * c + b * c| = 768 :=
by 
  have h1 := equilateral_triangle a b c,
  have h2 := sum_of_complex a b c,
  sorry

end find_abs_ab_ac_bc_l242_242738


namespace log_three_pow_nine_pow_three_eq_six_l242_242948

open Real

noncomputable def eval_log_three_pow_nine_pow_three : ℝ :=
  log 3 (9^3)

theorem log_three_pow_nine_pow_three_eq_six :
  eval_log_three_pow_nine_pow_three = 6 := by
  sorry

end log_three_pow_nine_pow_three_eq_six_l242_242948


namespace value_of_a_l242_242164

variable (a : ℤ)
def U : Set ℤ := {2, 4, 3 - a^2}
def P : Set ℤ := {2, a^2 + 2 - a}

theorem value_of_a (h : (U a) \ (P a) = {-1}) : a = 2 :=
sorry

end value_of_a_l242_242164


namespace log_evaluation_l242_242759

theorem log_evaluation : ∀ (a : ℕ), a = 3 → log 3 (9^3) = 6 :=
by
  intro a ha
  have h1 : 9 = 3^2 := by sorry
  have h2 : (9^3) = (3^2)^3 := by rw [h1]
  have h3 : (3^2)^3 = 3^6 := by sorry
  have h4 : log 3 (3^6) = 6 := by sorry
  rw [h2, h3, h4]
  exact ha

end log_evaluation_l242_242759


namespace maximize_Sn_l242_242451

noncomputable def a1 := 5
noncomputable def d := -5/7

-- Sum of the first n terms for the given arithmetic sequence
noncomputable def Sn (n : ℕ) : ℚ :=
  (n / 2 : ℚ) * (2 * a1 + (n - 1) * d)

theorem maximize_Sn : Sn 8 ≥ Sn n ∀ n : ℕ >= 0 := by
  sorry

end maximize_Sn_l242_242451


namespace find_k_l242_242741

def f (x : ℝ) : ℝ := 8 * x^2 - (3 / x) + 5
def g (x : ℝ) (k : ℝ) : ℝ := x^2 - k

theorem find_k (h : f 3 - g 3 k = 3) : k = -64 := by
  sorry

end find_k_l242_242741


namespace amy_more_than_connor_l242_242384

-- Define the variables for scores of Connor, Amy, and Jason
variables (C A J : ℕ)

-- Define the conditions
def cond1 : Prop := C = 2
def cond2 : Prop := A > C
def cond3 : Prop := J = 2 * A
def cond4 : Prop := C + A + J = 20

-- Prove the statement
theorem amy_more_than_connor (h1 : cond1) (h2 : cond2) (h3 : cond3) (h4 : cond4) : A - C = 4 := 
by sorry

end amy_more_than_connor_l242_242384


namespace num_true_statements_l242_242017

/-- Given non-intersecting lines m, n, l and non-coincident planes α, β -/
variable {m n l : Line}
variable {α β : Plane}

/-- propositions -/
def prop1 (m α : Line) (l : Line) (A : Point) : Prop :=
(m ⊆ α) ∧ (l ∩ α = {A}) ∧ (A ∉ m) → ¬coplanar l m

def prop2 (l m : Line) (α : Plane) (n : Line) : Prop :=
(skew_lines l m) ∧ (l ∥ α) ∧ (m ∥ α) ∧ (n ⟂ l ∧ n ⟂ m) → (n ⟂ α)

def prop3 (l m : Line) (α β : Plane) (A : Point) : Prop :=
(l ⊆ α) ∧ (m ⊆ α) ∧ (l ∩ m = {A}) ∧ (m ∥ β) ∧ (l ∥ β) → (α ∥ β)

def prop4 (l : Line) (m : Line) (α β : Plane) : Prop :=
(l ∥ α) ∧ (m ∥ β) ∧ (α ∥ β) → (l ∥ m)

/-- proof problem -/
theorem num_true_statements (m n l : Line) (α β : Plane) (A : Point) :
  ∃ (num_true : ℕ), (prop1 m α l A) ∧ (prop2 l m α n) ∧ (prop3 l m α β A) ∧ (¬prop4 l m α β) ∧ (num_true = 3) :=
by
  sorry

end num_true_statements_l242_242017


namespace probability_of_inequality_l242_242143

noncomputable def probability_satisfy_inequality : ℝ :=
  let θ := measureTheory.measure_space.random_variable(0, real.pi)
  let condition := sqrt 2 ≤ sqrt 2 * real.sin θ + sqrt 2 * real.cos θ ∧ sqrt 2 * real.sin θ + sqrt 2 * real.cos θ ≤ 2
  measureTheory.probability θ condition

theorem probability_of_inequality (h0 : 0 ≤ θ) (hπ : θ ≤ real.pi) :
  probability_satisfy_inequality = 1 / 2 :=
sorry

end probability_of_inequality_l242_242143


namespace amanda_car_round_trip_time_l242_242723

theorem amanda_car_round_trip_time :
  let bus_time := 40
  let bus_distance := 120
  let detour := 15
  let reduced_time := 5
  let amanda_trip_one_way_time := bus_time - reduced_time
  let amanda_round_trip_distance := (bus_distance * 2) + (detour * 2)
  let required_time := amanda_round_trip_distance * amanda_trip_one_way_time / bus_distance
  required_time = 79 :=
by
  sorry

end amanda_car_round_trip_time_l242_242723


namespace find_s_l242_242401

theorem find_s (s : ℝ) (h : 4 * log 3 s = log 3 (4 * s^2)) : s = 2 := 
sorry

end find_s_l242_242401


namespace M_inter_N_l242_242315

def M : Set ℝ := {y | ∃ x, y = 3 - x^2}
def N : Set ℝ := {y | ∃ x, y = 2 * x^2 - 1}

theorem M_inter_N : M ∩ N = Icc (-1 : ℝ) 3 :=
by
  sorry

end M_inter_N_l242_242315


namespace five_digit_palindrome_difference_l242_242673

-- Definition of a five-digit palindrome.
def is_five_digit_palindrome (n : ℕ) : Prop :=
  n >= 10000 ∧ n < 100000 ∧ (n % 10) = (n / 10000) ∧ ((n / 10) % 10) = ((n / 1000) % 10)

-- Main theorem statement
theorem five_digit_palindrome_difference (a b : ℕ)
  (ha : is_five_digit_palindrome a)
  (hb : is_five_digit_palindrome b)
  (h_less : a < b)
  (h_between : ∀ c : ℕ, is_five_digit_palindrome c → a < c → c < b → false) :
  b - a ∈ {100, 110, 11} :=
by sorry

end five_digit_palindrome_difference_l242_242673


namespace two_pipes_fill_time_l242_242639

theorem two_pipes_fill_time (R : ℝ) (h : 3 * R = 1 / 8) : 2 * R = 1 / 12 := 
by sorry

end two_pipes_fill_time_l242_242639


namespace omega_range_l242_242616

-- Definitions for the original problem conditions
def sin_omega_x (ω x : ℝ) : ℝ := Real.sin (ω * x) -- function definition y = sin(ωx)
def max_in_interval (ω : ℝ) : Prop := -- condition about max values in [0,1]
  ∃ k : ℕ, k = 50 ∧ ∀ n : ℤ, (n * (2 * Mathlib.pi)) / ω ∈ Icc (0 : ℝ) (1 : ℝ)

theorem omega_range (ω : ℝ) (h : ω > 0) (h_max : max_in_interval ω) : ω ≥ 100 * Real.pi :=
sorry

end omega_range_l242_242616


namespace total_nominal_income_l242_242577

theorem total_nominal_income :
  let principal := 8700
  let rate := 0.06 / 12
  let income (n : ℕ) := principal * ((1 + rate) ^ n - 1)
  income 6 + income 5 + income 4 + income 3 + income 2 + income 1 = 921.15 := by
  sorry

end total_nominal_income_l242_242577


namespace imaginary_unit_multiplication_l242_242029

-- Statement of the problem   
theorem imaginary_unit_multiplication (i : ℂ) (hi : i ^ 2 = -1) : i * (1 + i) = -1 + i :=
by sorry

end imaginary_unit_multiplication_l242_242029


namespace log_base_3_of_9_cubed_l242_242860

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 := by
  -- Assuming the necessary properties of logarithms and exponents
  sorry

end log_base_3_of_9_cubed_l242_242860


namespace license_plate_increase_factor_l242_242130

def old_license_plates := 26^2 * 10^3
def new_license_plates := 26^3 * 10^4

theorem license_plate_increase_factor : (new_license_plates / old_license_plates) = 260 := by
  sorry

end license_plate_increase_factor_l242_242130


namespace part_I_part_II_1_part_II_2_l242_242013

-- Define the ellipse equation and conditions
variable (a b : ℝ)
variable (a_gt_b : a > b) (b_gt_0 : b > 0)
variable (D : ℝ × ℝ) (E : ℝ × ℝ)
variable (H_D : D = (2, 0)) (H_E : E = (1, (Real.sqrt 3) / 2))

-- Define the ellipse at the given points
def ellipse := ∃ (a' b' : ℝ), 0 < b' ∧ b' < a' ∧ 
  ( (2 : ℝ)^2 / a'^2 + 0^2 / b'^2 = 1 ) ∧ 
  ( (1 : ℝ)^2 / a'^2 + (Real.sqrt 3 / 2)^2 / b'^2 = 1 )

-- Part I: Find the equation of ellipse F
theorem part_I : ellipse -> (a = 2 ∧ b = 1) := sorry

-- Part II-1: Given line l intersects F and other conditions, prove 4m^2 = 4k^2 + 1
variable (k m : ℝ)
variable (x1 y1 x2 y2 : ℝ)
variable (G : ℝ × ℝ)
variable (H_G : G = ((x1 + x2) / 2, (y1 + y2) / 2))
variable (O : ℝ × ℝ)
variable (H_O : O = (0, 0))
variable (Q : ℝ × ℝ)
variable (H_Q : Q = (2 * (G.fst - O.fst), 2 * (G.snd - O.snd)))
variable (A B : ℝ × ℝ)
variable (H_AB : A = (x1, y1) ∧ B = (x2, y2))
variable (intersect_line : line_intersect_ellipse l F A B)
def line_intersect_ellipse := 
  ∃ (x1 y1 x2 y2 : ℝ), 
  (y1 = k * x1 + m) ∧ (y2 = k * x2 + m) ∧ 
  (x1^2 / a^2 + y1^2 / b^2 = 1) ∧ (x2^2 / a^2 + y2^2 / b^2 = 1)

theorem part_II_1 (H2 : line_intersect_ellipse ) : 4 * m^2 = 4 * k^2 + 1 := sorry

-- Part II-2: Find the area of triangle AOB
def triangle_area (A B O : ℝ × ℝ) := 
  (1 / 2) * abs (O.snd * B.fst - O.fst * B.snd + B.snd * A.fst - B.fst * A.snd + A.snd * O.fst - A.fst * O.snd)

theorem part_II_2 (H3 : line_intersect_ellipse ): triangle_area A O B = Real.sqrt(3) / 2 := sorry

end part_I_part_II_1_part_II_2_l242_242013


namespace solve_for_y_l242_242204

open Real

theorem solve_for_y (y : ℝ) : 3^y + 18 = 4 * 3^y - 44 ↔ 
                              y = log 3 (62 / 3) :=
by
  split
  · intro h
    have h1 : 3^y + 18 = 4 * 3^y - 44 := h
    -- sorry, solving steps will go here
    sorry 
  · intro h
    rw h
    -- sorry, verification steps that y = log 3 (62 / 3) satisfy the original equation
    sorry 

end solve_for_y_l242_242204


namespace problem_solution_set_l242_242241

open Nat

def combination (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)
def permutation (n k : ℕ) : ℕ := n.factorial / (n - k).factorial

theorem problem_solution_set :
  {n : ℕ | 2 * combination n 3 ≤ permutation n 2} = {n | n = 3 ∨ n = 4 ∨ n = 5} :=
by
  sorry

end problem_solution_set_l242_242241


namespace triangle_properties_l242_242010

open Real

noncomputable def vector_m (a b c : ℝ) : ℝ × ℝ := (2 * a - c, b)
noncomputable def vector_n (B C : ℝ) : ℝ × ℝ := (cos C, cos B)

theorem triangle_properties
  (A B C a b c : ℝ)
  (area : ℝ)
  (angle_condition : vector_m a b c = λ (v : ℝ × ℝ), ∃ k: ℝ, k * vector_n B C)
  (area_condition : area = sqrt 3) :
  B = π / 3 ∧ (c = 2 ∧ a = c) :=
by
  sorry

end triangle_properties_l242_242010


namespace log_evaluation_l242_242757

theorem log_evaluation : ∀ (a : ℕ), a = 3 → log 3 (9^3) = 6 :=
by
  intro a ha
  have h1 : 9 = 3^2 := by sorry
  have h2 : (9^3) = (3^2)^3 := by rw [h1]
  have h3 : (3^2)^3 = 3^6 := by sorry
  have h4 : log 3 (3^6) = 6 := by sorry
  rw [h2, h3, h4]
  exact ha

end log_evaluation_l242_242757


namespace function_properties_l242_242662

-- Define the conditions for the function f
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

def max_value (f : ℝ → ℝ) (v : ℝ) : Prop := ∃ x : ℝ, f x = v ∧ ∀ y : ℝ, f y ≤ v

def is_not_quadratic (f : ℝ → ℝ) : Prop := ∀ a b c : ℝ, f ≠ λ x, a * x^2 + b * x + c

-- Prove that f(x) = 2cos(x) satisfies all conditions
theorem function_properties :
  let f := λ x : ℝ, 2 * Real.cos x
  in is_even f ∧ max_value f 2 ∧ is_not_quadratic f :=
by
  let f := λ x : ℝ, 2 * Real.cos x
  split
  { unfold is_even
    intro x
    exact Real.cos_neg x }
  { split
    { unfold max_value
      use 0
      split
      { exact Real.cos_zero }
      { intro y
        exact mul_le_mul_of_nonneg_left (Real.cos_le_one y) (by norm_num) } }
    { unfold is_not_quadratic
      -- sketch of the proof: for any coefficients a, b, c, show f ≠ λ x, a * x^2 + b * x + c
      sorry } }

end function_properties_l242_242662


namespace log_pow_evaluation_l242_242918

-- Define the condition that 9 is equal to 3^2
def nine_eq_three_squared : Prop := 9 = 3^2

-- The main theorem that we need to prove
theorem log_pow_evaluation (h : nine_eq_three_squared) : log 3 (9^3) = 6 :=
  sorry

end log_pow_evaluation_l242_242918


namespace find_m_l242_242002

theorem find_m : ∃ (m : ℝ), (x = 1 ∧ y = 3 → 3 * m * x - 2 * y = 9) ∧ m = 5 :=
by
  let x := 1
  let y := 3
  use 5
  split
  case left =>
    intro h
    rw [h.1, h.2]
    norm_num
  case right =>
    rfl

end find_m_l242_242002


namespace sum_even_integers_between_200_and_400_l242_242288

theorem sum_even_integers_between_200_and_400 : 
  (Finset.sum (Finset.filter (λ x, x % 2 = 0) (Finset.range 401)) 
    - Finset.sum (Finset.filter (λ x, x % 2 = 0) (Finset.range 201)))  = 30100 :=
begin
  sorry
end

end sum_even_integers_between_200_and_400_l242_242288


namespace max_value_f_0_f_2017_l242_242997

noncomputable def f : ℝ → ℝ := sorry

axiom f_condition : ∀ x : ℝ, f(x + 1) = 0.5 + real.sqrt(f x - (f x)^2)

theorem max_value_f_0_f_2017 : f 0 + f 2017 = 1 + real.sqrt(2) / 2 :=
sorry

end max_value_f_0_f_2017_l242_242997


namespace frequency_of_fifth_group_is_288_l242_242523

-- Defining the conditions as they appear in the problem
def total_sample_size : ℕ := 1600
def num_rectangles : ℕ := 9
def area_first_rectangle : ℝ := 0.02
def total_area_histogram : ℝ := 1

-- Conditions for arithmetic sequences
def first_arith_sequence (d : ℝ) (i : ℕ) : ℝ := area_first_rectangle + (i - 1 : ℝ) * d
def last_arith_sequence (d : ℝ) (i : ℕ) : ℝ := first_arith_sequence d 5 + (i - 5 : ℝ) * (-d)

-- Statement of the problem as a theorem in Lean 4
theorem frequency_of_fifth_group_is_288 : 
  ∃ (d : ℝ), 
    (∑ i in finset.range 5, first_arith_sequence d (i + 1)) + 
    (∑ i in finset.range 5, last_arith_sequence d (i + 5)) = total_area_histogram ∧
    let area_fifth_rectangle := first_arith_sequence d 5 in
    (total_sample_size : ℝ) * area_fifth_rectangle = 288 :=
begin
  sorry
end

end frequency_of_fifth_group_is_288_l242_242523


namespace log3_of_9_to_3_l242_242804

theorem log3_of_9_to_3 : Real.logb 3 (9^3) = 6 :=
by
  have h1 : 9 = 3^2 := by norm_num
  have h2 : 9^3 = (3^2)^3 := by rw [h1]
  have h3 : (3^2)^3 = 3^(2 * 3) := by rw [pow_mul]
  have h4 : Real.logb 3 (3^(2 * 3)) = 2 * 3 := Real.logb_pow 3 (2 * 3)
  rw [h4]
  norm_num

end log3_of_9_to_3_l242_242804


namespace log_pow_evaluation_l242_242911

-- Define the condition that 9 is equal to 3^2
def nine_eq_three_squared : Prop := 9 = 3^2

-- The main theorem that we need to prove
theorem log_pow_evaluation (h : nine_eq_three_squared) : log 3 (9^3) = 6 :=
  sorry

end log_pow_evaluation_l242_242911


namespace number_of_diagonals_in_octagon_l242_242348

theorem number_of_diagonals_in_octagon : 
  let n := 8 in n * (n - 3) / 2 = 20 :=
by sorry

end number_of_diagonals_in_octagon_l242_242348


namespace log_three_nine_cubed_l242_242955

theorem log_three_nine_cubed : ∀ (log : ℕ → ℕ → ℕ), log 3 9 = 2 → log 3 3 = 1 → (∀ (a b n : ℕ), log a (b ^ n) = n * log a b) → log 3 (9 ^ 3) = 6 :=
by
  intros log h1 h2 h3
  sorry

end log_three_nine_cubed_l242_242955


namespace solve_g_eq_g_inv_l242_242388

noncomputable def g (x : ℝ) : ℝ := 4 * x - 5

noncomputable def g_inv (x : ℝ) : ℝ := (x + 5) / 4

theorem solve_g_eq_g_inv : 
  ∃ x : ℝ, g x = g_inv x ∧ x = 5 / 3 :=
by
  sorry

end solve_g_eq_g_inv_l242_242388


namespace log_base_3_of_9_cubed_l242_242861

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 := by
  -- Assuming the necessary properties of logarithms and exponents
  sorry

end log_base_3_of_9_cubed_l242_242861


namespace calculate_rhombus_area_l242_242261

def rhombus_adj_sides_eq4_angle_eq_45_area : Prop :=
  ∀ (A B C D : ℝ), 
  ∃ (AB CD : ℝ) (angle_Dab : ℝ) (area : ℝ), 
  AB = 4 ∧ CD = 4 ∧ angle_Dab = 45 * (π / 180) ∧ ( area = 8 * √2 )

theorem calculate_rhombus_area :
  rhombus_adj_sides_eq4_angle_eq_45_area :=
by
  sorry

end calculate_rhombus_area_l242_242261


namespace system_of_equations_solution_l242_242173

theorem system_of_equations_solution (n : ℕ) (h : 1 ≤ n) 
  (x : Fin n → ℝ) 
  (h1 : ∑ i in Finset.range n, (finprod (λ j : Fin (i + 1), x j)) ^ (2 * n) = 1)
  (h2 : ∑ i in Finset.range n, (finprod (λ j : Fin (i + 1), x j)) ^ (2 * n + 1) = 1) : 
  x 0 = 1 ∧ ∀ i : Fin n, i ≠ 0 → x i = 0 := 
  sorry

end system_of_equations_solution_l242_242173


namespace positive_difference_two_numbers_l242_242626

theorem positive_difference_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 80) : |x - y| = 8 := by
  sorry

end positive_difference_two_numbers_l242_242626


namespace sum_of_divisors_divisible_by_24_l242_242177

theorem sum_of_divisors_divisible_by_24 (n : ℕ) (h : (n + 1) % 24 = 0) :
    ((Finset.filter (λ d, n % d = 0) (Finset.range (n + 1))).sum id) % 24 = 0 := 
by 
  sorry

end sum_of_divisors_divisible_by_24_l242_242177


namespace find_selling_price_l242_242682

-- Define the variables
constant CurrentPrice : ℝ := 60
constant CurrentVolume : ℝ := 300
constant CostPrice : ℝ := 40
constant DesiredProfit : ℝ := 6080

-- Define how the new volume is calculated
def NewVolume (d : ℝ) : ℝ := CurrentVolume + 20 * (CurrentPrice - d)

-- Define the profit equation
def Profit (d : ℝ) : ℝ := (d - CostPrice) * NewVolume(d)

-- The statement to prove
theorem find_selling_price (d : ℝ) : 
  (Profit d = DesiredProfit) → d = 56 :=
sorry

end find_selling_price_l242_242682


namespace intersection_is_correct_l242_242471

def A : Set ℝ := { x | x * (x - 2) < 0 }
def B : Set ℝ := { x | Real.log x > 0 }

theorem intersection_is_correct : A ∩ B = { x | 1 < x ∧ x < 2 } := by
  sorry

end intersection_is_correct_l242_242471


namespace value_of_m_l242_242999

theorem value_of_m (m : ℝ) (α : ℝ) (h1 : cos α = 3 / 5) (h2 : ∀ (x y : ℝ), P x y → (P 3 m)) : m = 4 ∨ m = -4 :=
by
  -- Conditions
  have h3 : 3 / real.sqrt (9 + m ^ 2) = cos α, from sorry
 
  -- Solving the equation
  have h4 : h3 = 3 / 5, from h1
  
  sorry

end value_of_m_l242_242999


namespace relationship_between_m_and_n_l242_242992

theorem relationship_between_m_and_n
  (a : ℝ) (b : ℝ) (ha : a > 2) (hb : b ≠ 0)
  (m : ℝ := a + 1 / (a - 2))
  (n : ℝ := 2^(2 - b^2)) :
  m > n :=
sorry

end relationship_between_m_and_n_l242_242992


namespace log_three_pow_nine_pow_three_eq_six_l242_242942

open Real

noncomputable def eval_log_three_pow_nine_pow_three : ℝ :=
  log 3 (9^3)

theorem log_three_pow_nine_pow_three_eq_six :
  eval_log_three_pow_nine_pow_three = 6 := by
  sorry

end log_three_pow_nine_pow_three_eq_six_l242_242942


namespace common_point_on_incircle_l242_242138

variable (A B C K_a K_b K_c : Type)

-- Define scalene triangle ABC
axiom scalene_triangle (A B C : Type) [triangle A B C] : Prop

-- Define K_a, K_b, K_c as described in the problem statement
axiom feet_of_bisectors (A B C K_a K_b K_c : Type)
  [incircle A B C K_a]
  [incircle A B C K_b]
  [incircle A B C K_c] : Prop

axiom meet_at_common_point (A B C K_a K_b K_c : Type)
  [midpoint A B C K_a]
  [midpoint A B C K_b]
  [midpoint A B C K_c] : Prop

theorem common_point_on_incircle (A B C K_a K_b K_c : Type)
  [scalene_triangle A B C]
  [feet_of_bisectors A B C K_a K_b K_c]
  [meet_at_common_point A B C K_a K_b K_c] :
  ∃ P, incircle A B C P :=
sorry

end common_point_on_incircle_l242_242138


namespace sum_even_200_to_400_l242_242296

theorem sum_even_200_to_400 : 
  let t₁ := 202
  let tₙ := 398
  let n := 99
  let sum := (t₁ + tₙ) * n / 2 in
  sum = 29700 := 
by
  let t₁ := 202
  let tₙ := 398
  let n := 99
  let sum := (t₁ + tₙ) * n / 2
  show sum = 29700
  sorry

end sum_even_200_to_400_l242_242296


namespace log_base_3_of_9_cubed_l242_242924
-- Import the necessary parts of the standard library

-- Define the statement to be proved
theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 :=
by
  sorry

end log_base_3_of_9_cubed_l242_242924


namespace range_of_a_l242_242505

noncomputable def f (a : ℝ) : ℝ → ℝ :=
  λ x, if x < 0 then -x + 3 * a else a^x + 1

theorem range_of_a (a : ℝ) (h_pos : a > 0) (h_neq : a ≠ 1) (h_decreasing : ∀ x y : ℝ, x < y → f a y ≤ f a x) : 
  (2/3 : ℝ) ≤ a ∧ a < 1 :=
by
  sorry

end range_of_a_l242_242505


namespace hyperbola_eccentricity_is_correct_l242_242041

noncomputable def hyperbola_eccentricity (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) 
(asymptote : 2 * x - sqrt 3 * y = 0 → y = (2 / sqrt 3) * x)
: ℝ :=
sqrt 21 / 3

theorem hyperbola_eccentricity_is_correct (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) 
(asymptote : 2 * x - sqrt 3 * y = 0 → y = (2 / sqrt 3) * x) :
  hyperbola_eccentricity a b a_pos b_pos asymptote = sqrt 21 / 3 := 
sorry

end hyperbola_eccentricity_is_correct_l242_242041


namespace cubic_sum_l242_242087

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 := by
  sorry

end cubic_sum_l242_242087


namespace log_base_3_of_9_cubed_l242_242919
-- Import the necessary parts of the standard library

-- Define the statement to be proved
theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 :=
by
  sorry

end log_base_3_of_9_cubed_l242_242919


namespace coordinates_of_point_P_l242_242126

noncomputable def tangent_slope_4 : Prop :=
  ∀ (x y : ℝ), y = 1 / x → (-1 / (x^2)) = -4 → (x = 1 / 2 ∧ y = 2) ∨ (x = -1 / 2 ∧ y = -2)

theorem coordinates_of_point_P : tangent_slope_4 :=
by sorry

end coordinates_of_point_P_l242_242126


namespace sum_primes_and_composites_l242_242726

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

noncomputable def is_composite (n : ℕ) : Prop :=
  ¬is_prime n ∧ n > 1

theorem sum_primes_and_composites {n : ℕ} (h₁ : n = 20) :
  ∑ k in Finset.filter is_prime (Finset.range (n + 1)) + ∑ k in Finset.filter is_composite (Finset.range (n + 1)) = 209 :=
by
  sorry

end sum_primes_and_composites_l242_242726


namespace marian_baked_cookies_l242_242185

theorem marian_baked_cookies :
  let cookies_per_tray := 12
  let trays_used := 23
  trays_used * cookies_per_tray = 276 :=
by
  sorry

end marian_baked_cookies_l242_242185


namespace geometric_sequence_a_div_n_sum_first_n_terms_l242_242008

variable {a : ℕ → ℝ} -- sequence a_n
variable {S : ℕ → ℝ} -- sum of first n terms S_n

axiom S_recurrence {n : ℕ} (hn : n > 0) : 
  S (n + 1) = S n + (n + 1) / (3 * n) * a n

axiom a_1 : a 1 = 1

theorem geometric_sequence_a_div_n :
  ∃ (r : ℝ), ∀ {n : ℕ} (hn : n > 0), (a n / n) = r^n := 
sorry

theorem sum_first_n_terms (n : ℕ) :
  S n = (9 / 4) - ((9 / 4) + (3 * n / 2)) * (1 / 3) ^ n :=
sorry

end geometric_sequence_a_div_n_sum_first_n_terms_l242_242008


namespace log_three_nine_cubed_l242_242956

theorem log_three_nine_cubed : ∀ (log : ℕ → ℕ → ℕ), log 3 9 = 2 → log 3 3 = 1 → (∀ (a b n : ℕ), log a (b ^ n) = n * log a b) → log 3 (9 ^ 3) = 6 :=
by
  intros log h1 h2 h3
  sorry

end log_three_nine_cubed_l242_242956


namespace count_convex_cyclic_quadrilaterals_is_1505_l242_242069

-- Define a quadrilateral using its sides
structure Quadrilateral where
  a b c d : ℕ
  deriving Repr, DecidableEq

-- Define what a convex cyclic quadrilateral is, given the integer sides and the perimeter condition
def isConvexCyclicQuadrilateral (q : Quadrilateral) : Prop :=
 q.a + q.b + q.c + q.d = 36 ∧
 q.a > 0 ∧ q.b > 0 ∧ q.c > 0 ∧ q.d > 0 ∧
 q.a + q.b > q.c + q.d ∧ q.c + q.d > q.a + q.b ∧
 q.b + q.c > q.d + q.a ∧ q.d + q.a > q.b + q.c

-- Noncomputable definition to count all convex cyclic quadrilaterals
noncomputable def countConvexCyclicQuadrilaterals : ℕ :=
  sorry

-- The theorem stating the count is equal to 1505
theorem count_convex_cyclic_quadrilaterals_is_1505 :
  countConvexCyclicQuadrilaterals = 1505 :=
  sorry

end count_convex_cyclic_quadrilaterals_is_1505_l242_242069


namespace log_base_3_of_9_cubed_l242_242836

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 := by
  -- Define the known conditions
  have h1 : 9 = 3^2 := by norm_num
  have h2 : log 3 (a^n) = n * log 3 a := sorry  -- Statement of the logarithmic identity
  
  -- Apply the conditions to prove the result
  calc
    log 3 (9^3)
        = log 3 ((3^2)^3) : by rw h1
    ... = log 3 (3^(2 * 3)) : by congr; norm_num
    ... = log 3 (3^6) : by norm_num
    ... = 6 : by rw [h2]; norm_num

end log_base_3_of_9_cubed_l242_242836


namespace geometric_sequence_problem_l242_242441

variable {a : ℕ → ℝ}
variable (r a1 : ℝ)
variable (h_pos : ∀ n, a n > 0)
variable (h_geom : ∀ n, a (n + 1) = a 1 * r ^ n)
variable (h_eq : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 2025)

theorem geometric_sequence_problem :
  a 3 + a 5 = 45 :=
by
  sorry

end geometric_sequence_problem_l242_242441


namespace collinear_ABD_determine_k_collinear_l242_242475

variables {a b : Type} [AddCommGroup a] [AddCommGroup b]
variables (a b : a)
variables (k : ℝ)

-- Non-collinearity and non-zero assumptions
axiom a_nonzero : a ≠ 0
axiom b_nonzero : b ≠ 0
axiom non_collinear : ¬ (∃ (α : ℝ), a = α • b)

-- Given conditions for the first problem
def vec_AB := (2 : ℝ) • a + (3 : ℝ) • b
def vec_BC := (6 : ℝ) • a + (23 : ℝ) • b
def vec_CD := (4 : ℝ) • a - (8 : ℝ) • b
def vec_BD := vec_BC + vec_CD

-- Problem 1
theorem collinear_ABD : vec_BD = 5 • vec_AB :=
by sorry

-- Given conditions for the second problem
def vec_AB_k := (2 : ℝ) • a + k • b
def vec_CB := (1 : ℝ) • a + (3 : ℝ) • b
def vec_CD_2 := (2 : ℝ) • a - (1 : ℝ) • b
def vec_CA := vec_CB - vec_AB_k
def vec_DA := vec_CA + vec_CD_2

-- Problem 2
theorem determine_k_collinear :
  (∃ (k : ℝ), vec_DA = 0) → k = -8 :=
by sorry

end collinear_ABD_determine_k_collinear_l242_242475


namespace nonstudent_ticket_cost_l242_242982

theorem nonstudent_ticket_cost :
  ∃ x : ℝ, (530 * 2 + (821 - 530) * x = 1933) ∧ x = 3 :=
by 
  sorry

end nonstudent_ticket_cost_l242_242982


namespace sqrt_sum_comparison_l242_242736

theorem sqrt_sum_comparison:
  sqrt 3 + sqrt 5 > sqrt 2 + sqrt 6 :=
by
  sorry

end sqrt_sum_comparison_l242_242736


namespace find_a_b_l242_242490

theorem find_a_b (a b : ℤ) (h : ∀ x : ℤ, (x - 2) * (x + 3) = x^2 + a * x + b) : a = 1 ∧ b = -6 :=
by
  sorry

end find_a_b_l242_242490


namespace target_heart_rate_sprinting_l242_242727

def max_heart_rate (age : ℕ) : ℕ := 225 - age
def jogging_target_heart_rate (max_hr : ℕ) : ℝ := 0.75 * max_hr
def sprinting_target_heart_rate (jogging_hr : ℝ) : ℝ := jogging_hr + 10

theorem target_heart_rate_sprinting (age : ℕ) (h : age = 30) : 
  let max_hr := max_heart_rate age in
  let jogging_hr := jogging_target_heart_rate max_hr in
  let sprinting_hr := sprinting_target_heart_rate jogging_hr in
  sprinting_hr.to_nat = 156 :=
by 
  sorry

end target_heart_rate_sprinting_l242_242727


namespace gcd_factorial_8_10_l242_242276

theorem gcd_factorial_8_10 (n : ℕ) (hn : n = 10! - 8!): gcd 8! 10! = 8! := by
  sorry

end gcd_factorial_8_10_l242_242276


namespace distance_feet_perpendiculars_invariant_l242_242416

theorem distance_feet_perpendiculars_invariant (A B C D E F : Point) (h : Triangle A B C) 
  (h1 : RightAngle A D B) (h2 : Perpendicular D A A B) (h3 : Perpendicular D F A C) 
  (h4 : RightAngle D E B) (h5 : RightAngle D F C) : ∀ (hA B C : Triangle A B C),
  distance E F = distance_from_some_function (Area A B C) (sides_of_triangle A B C) := sorry

end distance_feet_perpendiculars_invariant_l242_242416


namespace remainder_when_divided_by_DE_l242_242163

theorem remainder_when_divided_by_DE (P D E Q R M S C : ℕ) 
  (h1 : P = Q * D + R) 
  (h2 : Q = E * M + S) :
  (∃ quotient : ℕ, P = quotient * (D * E) + (S * D + R + C)) :=
by {
  sorry
}

end remainder_when_divided_by_DE_l242_242163


namespace attendance_second_day_l242_242698

theorem attendance_second_day (total_attendance first_day_attendance second_day_attendance third_day_attendance : ℕ) 
  (h_total : total_attendance = 2700)
  (h_second_day : second_day_attendance = first_day_attendance / 2)
  (h_third_day : third_day_attendance = 3 * first_day_attendance) :
  second_day_attendance = 300 :=
by
  sorry

end attendance_second_day_l242_242698


namespace log_base_3_of_9_cubed_l242_242824

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 := by
  sorry

end log_base_3_of_9_cubed_l242_242824


namespace polynomial_negativity_l242_242193

theorem polynomial_negativity (a x : ℝ) (h₀ : 0 < x) (h₁ : x < a) (h₂ : 0 < a) : 
  (a - x)^6 - 3 * a * (a - x)^5 + (5 / 2) * a^2 * (a - x)^4 - (1 / 2) * a^4 * (a - x)^2 < 0 := 
by
  sorry

end polynomial_negativity_l242_242193


namespace x_cubed_plus_y_cubed_l242_242079

variable (x y : ℝ)
variable (h₁ : x + y = 5)
variable (h₂ : x^2 + y^2 = 17)

theorem x_cubed_plus_y_cubed :
  x^3 + y^3 = 65 :=
by sorry

end x_cubed_plus_y_cubed_l242_242079


namespace xy_cubed_identity_l242_242098

namespace ProofProblem

theorem xy_cubed_identity (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 :=
by
  sorry

end ProofProblem

end xy_cubed_identity_l242_242098


namespace problem1_problem2_l242_242548

-- problem (1): Prove that if a = 1 and (p ∨ q) is true, then the range of x is 1 < x < 3
def p (a x : ℝ) : Prop := x ^ 2 - 4 * a * x + 3 * a ^ 2 < 0
def q (x : ℝ) : Prop := (x - 3) / (x - 2) < 0

theorem problem1 (x : ℝ) (a : ℝ) (h₁ : a = 1) (h₂ : p a x ∨ q x) : 
    1 < x ∧ x < 3 :=
sorry

-- problem (2): Prove that if p is a necessary but not sufficient condition for q,
-- then the range of a is 1 ≤ a ≤ 2
theorem problem2 (a : ℝ) :
  (∀ x : ℝ, q x → p a x) ∧ (∃ x : ℝ, p a x ∧ ¬q x) → 
  1 ≤ a ∧ a ≤ 2 := 
sorry

end problem1_problem2_l242_242548


namespace complex_modulus_l242_242502

def is_pure_imaginary (z : ℂ) : Prop :=
  z.re = 0

theorem complex_modulus (m : ℝ) (h_pure_imaginary : is_pure_imaginary ((1 : ℂ) + (m : ℂ) * complex.I) * ((3 : ℂ) + complex.I)) :
  complex.abs ((m : ℂ) + 3 * complex.I / (1 - complex.I)) = 3 := 
begin
  sorry,
end

end complex_modulus_l242_242502


namespace relation_between_a_b_c_l242_242040

-- Definitions based on given conditions
def even_function (f : ℝ → ℝ) : Prop :=
  ∃ g : ℝ → ℝ, ∀ x, f (x + 2) = g x ∧ g x = g (-x)

def monotone_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ ⦃x₁ x₂⦄, x₁ ∈ s → x₂ ∈ s → x₁ < x₂ → f x₁ < f x₂

-- Defining the values based on problem statements
def a (f : ℝ → ℝ) : ℝ := f (Real.log 18 / Real.log 3)
def b (f : ℝ → ℝ) : ℝ := f (Real.log (e^2 / Real.sqrt 2))
def c (f : ℝ → ℝ) : ℝ := f (Real.exp (Real.log 10 / 2))

-- Main theorem
theorem relation_between_a_b_c (f : ℝ → ℝ) :
  even_function f ∧ monotone_increasing_on f {x | 2 ≤ x} →
  b f < a f ∧ a f < c f :=
by
  sorry

end relation_between_a_b_c_l242_242040


namespace even_sum_formula_even_sum_102_to_200_l242_242608

-- Define the sum of the first n even numbers
def even_sum (n : ℕ) : ℕ :=
  ∑ i in Finset.range n, 2 * (i + 1)

-- Prove that the sum of the first n even numbers is n^2 + n
theorem even_sum_formula (n : ℕ) : even_sum n = n^2 + n := 
  sorry

-- Specific calculation for the range from 102 to 200
theorem even_sum_102_to_200 : 
  (∑ i in Finset.range 50, 2*(i + 51)) = 7550 :=
by 
  sorry

end even_sum_formula_even_sum_102_to_200_l242_242608


namespace icosahedron_path_count_l242_242310

-- Definitions from the conditions
def vertices := 12
def edges := 30
def top_adjacent := 5
def bottom_adjacent := 5

-- Define the total paths calculation based on the given structural conditions
theorem icosahedron_path_count (v e ta ba : ℕ) (hv : v = 12) (he : e = 30) (hta : ta = 5) (hba : ba = 5) : 
  (ta * (ta - 1) * (ba - 1)) * 2 = 810 :=
by
-- Insert calculation logic here if needed or detailed structure definitions
  sorry

end icosahedron_path_count_l242_242310


namespace no_primes_between_factorial_and_addition_l242_242412

theorem no_primes_between_factorial_and_addition (n : ℕ) (h : n > 1) :
  ∀ p : ℕ, prime p → (n! + 1 < p) → (p < n! + n) → false :=
by
  intros p hp hp_gt hp_lt
  -- Proof steps would follow here to show the contradiction that p can't be prime in this range.
  sorry

end no_primes_between_factorial_and_addition_l242_242412


namespace rhombus_area_l242_242256

theorem rhombus_area (a : ℝ) (θ : ℝ) (h₁ : a = 4) (h₂ : θ = π / 4) : 
    (a * a * Real.sin θ) = 16 :=
by
    have s1 : Real.sin (π / 4) = Real.sqrt 2 / 2 := Real.sin_pi_div_four
    rw [h₁, h₂, s1]
    have s2 : 4 * 4 * (Real.sqrt 2 / 2) = 16 := by norm_num
    exact s2

end rhombus_area_l242_242256


namespace lawn_area_inside_right_triangle_l242_242340

-- Define the sides of the main triangle
def a : ℝ := 7
def b : ℝ := 24
def c : ℝ := 25

-- Define the inner (decreased) inradius
def inradius_inner : ℝ := 1

-- Define the area of the lawn which we need to prove
def area_lawn : ℝ := 28 / 3

-- Formalize the problem statement
theorem lawn_area_inside_right_triangle :
  let s := (a + b + c) / 2 in
  let area_triangle := a * b / 2 in
  let inradius_outer := area_triangle / s in
  let k := inradius_inner / inradius_outer in
  let area_inner := k^2 * area_triangle in
  area_inner = area_lawn := 
sorry

end lawn_area_inside_right_triangle_l242_242340


namespace total_days_1996_to_2000_l242_242484

theorem total_days_1996_to_2000 : 
  let is_leap_year (year : ℕ) : Bool :=
    (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ (year % 400 = 0)
  in 
  let days_in_year (year : ℕ) : ℕ :=
    if is_leap_year year then 366 else 365
  in 
  (days_in_year 1996 + days_in_year 1997 + days_in_year 1998 + days_in_year 1999 + days_in_year 2000) = 1827 := 
by 
  sorry

end total_days_1996_to_2000_l242_242484


namespace log_pow_evaluation_l242_242917

-- Define the condition that 9 is equal to 3^2
def nine_eq_three_squared : Prop := 9 = 3^2

-- The main theorem that we need to prove
theorem log_pow_evaluation (h : nine_eq_three_squared) : log 3 (9^3) = 6 :=
  sorry

end log_pow_evaluation_l242_242917


namespace log3_of_9_to_3_l242_242808

theorem log3_of_9_to_3 : Real.logb 3 (9^3) = 6 :=
by
  have h1 : 9 = 3^2 := by norm_num
  have h2 : 9^3 = (3^2)^3 := by rw [h1]
  have h3 : (3^2)^3 = 3^(2 * 3) := by rw [pow_mul]
  have h4 : Real.logb 3 (3^(2 * 3)) = 2 * 3 := Real.logb_pow 3 (2 * 3)
  rw [h4]
  norm_num

end log3_of_9_to_3_l242_242808


namespace log_three_pow_nine_pow_three_eq_six_l242_242939

open Real

noncomputable def eval_log_three_pow_nine_pow_three : ℝ :=
  log 3 (9^3)

theorem log_three_pow_nine_pow_three_eq_six :
  eval_log_three_pow_nine_pow_three = 6 := by
  sorry

end log_three_pow_nine_pow_three_eq_six_l242_242939


namespace convert_to_scientific_notation_l242_242139

theorem convert_to_scientific_notation :
  40.25 * 10^9 = 4.025 * 10^9 :=
by
  -- Sorry is used here to skip the proof
  sorry

end convert_to_scientific_notation_l242_242139


namespace increase_by_percentage_l242_242679

theorem increase_by_percentage (x : ℝ) (percent : ℝ) (final : ℝ) : 
  x = 784.3 → percent = 28.5 → final = 1007.8255 :=
by
  assume h1 : x = 784.3
  assume h2 : percent = 28.5
  have h3 : final = x + (percent / 100) * x := by sorry
  exact h3

end increase_by_percentage_l242_242679


namespace ratio_of_radii_l242_242245

noncomputable
theorem ratio_of_radii 
  {r1 r2 r3 : ℝ} 
  (h1 : 0 < r1) 
  (h2 : r1 < r2) 
  (h3 : r2 < r3) 
  (h4 : ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 12 → (π * r1^2 / 12 = π * (r2^2 - r1^2) / 12) 
                     ∧ (π * (r2^2 - r1^2) / 12 = π * (r3^2 - r2^2) / 12)) : 
  r1 = 1 ∧ r2 = Real.sqrt 2 * r1 ∧ r3 = Real.sqrt 3 * r1 :=
sorry

end ratio_of_radii_l242_242245


namespace sum_even_odd_difference_l242_242270

theorem sum_even_odd_difference : 
  let N := 1500,
      sum_even := (N * (N + 1)),
      sum_odd := N * N
  in sum_even - sum_odd = 1500 := by
  let N := 1500
  let sum_even := N * (N + 1)
  let sum_odd := N * N
  calc
    sum_even - sum_odd = N * (N + 1) - N * N : by rfl
                  ... = N^2 + N - N^2       : by rw mul_add
                  ... = N                     : by ring
                  ... = 1500                  : by rfl

end sum_even_odd_difference_l242_242270


namespace total_nominal_income_l242_242578

theorem total_nominal_income :
  let principal := 8700
  let rate := 0.06 / 12
  let income (n : ℕ) := principal * ((1 + rate) ^ n - 1)
  income 6 + income 5 + income 4 + income 3 + income 2 + income 1 = 921.15 := by
  sorry

end total_nominal_income_l242_242578


namespace log_base_3_l242_242847

theorem log_base_3 (log_identity : ∀ (b a c : ℝ), log b (a^c) = c * log b a) :
  log 3 (9^3) = 6 :=
by
  have : 9 = 3^2 := by sorry
  have : log 3 3 = 1 := by sorry
  exact (log_identity 3 (3^2) 3).trans sorry

end log_base_3_l242_242847


namespace f_zero_twice_centroid_midpoint_three_chords_l242_242664

noncomputable def real : Type := ℝ

-- Part 1: Show that f(x) is zero for at least two points in (0, π)
theorem f_zero_twice 
  (f : real → real) 
  [h_cont : continuous_on f (Icc 0 π)]
  (h_sin : ∫ x in 0..π, f x * sin x = 0)
  (h_cos : ∫ x in 0..π, f x * cos x = 0)
  : ∃ x1 x2 ∈ Ioo 0 π, f x1 = 0 ∧ f x2 = 0 ∧ x1 ≠ x2 :=
by
  sorry

-- Part 2: Show the centroid property for bounded convex open regions
theorem centroid_midpoint_three_chords
  (region : set (real × real))
  (h_region : is_open region ∧ convex region)
  (h_bounded : ∃ (x y : real), ∀ p ∈ region, (complex.abs (p.1)) ≤ x ∧ (complex.abs (p.2)) ≤ y)
  : ∃ (chord1 chord2 chord3 : set (real × real)), 
      is_chord chord1 region ∧ is_chord chord2 region ∧ is_chord chord3 region ∧ 
      (chord1 ∩ centroid region ≠ ∅) ∧ (chord2 ∩ centroid region ≠ ∅) ∧ (chord3 ∩ centroid region ≠ ∅) ∧ 
      chord1 ≠ chord2 ∧ chord2 ≠ chord3 ∧ chord1 ≠ chord3 :=
by
  sorry

end f_zero_twice_centroid_midpoint_three_chords_l242_242664


namespace quadratic_fraction_formula_l242_242468

theorem quadratic_fraction_formula (p q α β : ℝ) 
  (h1 : α + β = p) 
  (h2 : α * β = 6) 
  (h3 : p^2 ≠ 12) 
  (h4 : ∃ x : ℝ, x^2 - p * x + q = 0) :
  (α + β) / (α^2 + β^2) = p / (p^2 - 12) :=
sorry

end quadratic_fraction_formula_l242_242468


namespace log_evaluation_l242_242762

theorem log_evaluation : ∀ (a : ℕ), a = 3 → log 3 (9^3) = 6 :=
by
  intro a ha
  have h1 : 9 = 3^2 := by sorry
  have h2 : (9^3) = (3^2)^3 := by rw [h1]
  have h3 : (3^2)^3 = 3^6 := by sorry
  have h4 : log 3 (3^6) = 6 := by sorry
  rw [h2, h3, h4]
  exact ha

end log_evaluation_l242_242762


namespace log_base_3_of_9_cubed_l242_242873

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 := by
  -- Assuming the necessary properties of logarithms and exponents
  sorry

end log_base_3_of_9_cubed_l242_242873


namespace prove_value_of_x_l242_242622

theorem prove_value_of_x (x y : ℝ) (data : Finset ℝ) (H_data : data = {80, 120, x, 60, y, 300, 110, 50, 90})
(H_mean : data.sum / 9 = x) (H_median : median(data) = x) (H_mode : mode(data) = x) : x = 90 :=
by 
  -- proof steps would go here
  sorry

end prove_value_of_x_l242_242622


namespace log_three_nine_cubed_l242_242960

theorem log_three_nine_cubed : ∀ (log : ℕ → ℕ → ℕ), log 3 9 = 2 → log 3 3 = 1 → (∀ (a b n : ℕ), log a (b ^ n) = n * log a b) → log 3 (9 ^ 3) = 6 :=
by
  intros log h1 h2 h3
  sorry

end log_three_nine_cubed_l242_242960


namespace collinear_D_M_L_l242_242674

-- Define the parallelogram and its properties
variables {A B C D A' C' K L M : Type*}

-- Given the parallelogram ABCD
-- with diagonal BD
-- Define the points A', C' on BD such that AA' is parallel to CC'
-- K lies on segment A'C
-- Line AK intersects line CC' at L
-- Through K, a line parallel to BC is drawn, and through C a line parallel to BD is drawn. These two lines intersect at M
-- Prove that points D, M, and L are collinear
theorem collinear_D_M_L
  (h_parallelogram : parallelogram A B C D)
  (h_A'_on_BD : point_on_line_segment A' B D)
  (h_C'_on_BD : point_on_line_segment C' B D)
  (h_AA'_||CC' : parallel (line_through A A') (line_through C C'))
  (h_K_on_A'_C : point_on_line_segment K A' C)
  (h_L_intersection : intersection_point (line_through A K) (line_through C C') L)
  (h_M_intersection : intersection_point (parallel_through K B C) (parallel_through C B D) M) :
  collinear D M L := 
sorry

end collinear_D_M_L_l242_242674


namespace gina_earnings_per_hour_l242_242418

def painting_speeds := 
  (rose_speed : ℕ := 6, lily_speed : ℕ := 7, sunflower_speed : ℕ := 5, orchid_speed : ℕ := 8)

def orders :=
  (order1 : (rose_cups : ℕ := 6, lily_cups : ℕ := 14, sunflower_cups : ℕ := 4, payment : ℕ := 120, max_time : ℕ := 6),
   order2 : (orchid_cups : ℕ := 10, rose_cups : ℕ := 2, payment : ℕ := 80, max_time : ℕ := 3),
   order3 : (sunflower_cups : ℕ := 8, orchid_cups : ℕ := 4, payment : ℕ := 70, max_time : ℕ := 4))

def schedule_constraints := 
  (total_available_time : ℕ := 12, daily_limit : ℕ := 4, days : ℕ := 3)

theorem gina_earnings_per_hour : 
  let total_time := (6/6 + 14/7 + 4/5) + (10/8 + 2/6) + (8/5 + 4/8)
  let total_earnings := 120 + 80 + 70 in
  total_time < 12 → (total_earnings / total_time) ≈ 36.10 :=
by
  have total_time_approx : total_time ≈ 7.48 := sorry
  have total_earnings := 270 := sorry
  show total_earnings / total_time ≈ 36.10 from sorry

end gina_earnings_per_hour_l242_242418


namespace marias_workday_end_time_l242_242562

theorem marias_workday_end_time :
  ∀ (start_time : ℕ) (lunch_time : ℕ) (work_duration : ℕ) (lunch_break : ℕ) (total_work_time : ℕ),
  start_time = 8 ∧ lunch_time = 13 ∧ work_duration = 8 ∧ lunch_break = 1 →
  (total_work_time = work_duration - (lunch_time - start_time - lunch_break)) →
  lunch_time + 1 + (work_duration - (lunch_time - start_time)) = 17 :=
by
  sorry

end marias_workday_end_time_l242_242562


namespace ivan_can_determine_abc_l242_242533

theorem ivan_can_determine_abc (a b c X Y Z : ℕ)
  (ha : 10 ≤ a ∧ a < 100) (hb : 10 ≤ b ∧ b < 100) (hc : 10 ≤ c ∧ c < 100)
  (hX : X = 1) (hY : Y = 100) (hZ : Z = 10000) :
  ∃ a b c, a * 1 + b * 100 + c * 10000 = a + 100 * b + 10000 * c :=
by {
  sorry,
}

end ivan_can_determine_abc_l242_242533


namespace only_solution_is_linear_l242_242396

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation (x y : ℝ) : f (f x * f y) = f x * y

theorem only_solution_is_linear : ∀ x, f x = x := 
begin
  sorry
end

end only_solution_is_linear_l242_242396


namespace binomial_product_l242_242376

open Nat

theorem binomial_product : (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 := by
  sorry

end binomial_product_l242_242376


namespace polar_eq_of_parametric_lambda_range_l242_242057

noncomputable theory

def parametric_eq_C (θ : ℝ) : ℝ × ℝ :=
  (-1 + 2 * Real.cos θ, 1 + 2 * Real.sin θ)

def polar_eq_C (ρ θ : ℝ) : Prop :=
  ρ^2 + 2 * Real.sqrt 2 * ρ * Real.cos (θ + Real.pi / 4) = 2

def midpoint_MN (ρ1 ρ2 : ℝ) : ℝ :=
  (ρ1 + ρ2) / 2

theorem polar_eq_of_parametric {θ ρ : ℝ} :
  (∃ θ, parametric_eq_C θ = (ρ * Real.cos θ, ρ * Real.sin θ)) → 
  polar_eq_C ρ θ :=
sorry

theorem lambda_range (α : ℝ) (hα : 0 ≤ α ∧ α < Real.pi) (ρ1 ρ2 : ℝ) :
  √2 ≥ |midpoint_MN ρ1 ρ2| → ∃ λ, λ = √2 :=
sorry

end polar_eq_of_parametric_lambda_range_l242_242057


namespace M_tensor_N_is_correct_l242_242022

variable (A B : Set ℝ) 
variable (M N : Set ℝ)

-- Conditions
def A_nonempty : Prop := A.nonempty
def B_nonempty : Prop := B.nonempty
def A_tensor_B : Set ℝ := {x | x ∈ (A ∪ B) ∧ x ∉ (A ∩ B)}
def M : Set ℝ := {y | ∃ (x : ℝ), (0 < x ∧ x < 2) ∧ y = -x^2 + 2*x}
def N : Set ℝ := {y | ∃ (x : ℝ), (0 < x) ∧ y = (2^(x-1))}

-- Correct Answer
def correct_answer : Set ℝ := (Ioc 0 (1 / 2)) ∪ Ioi 1

-- Proof Problem Statement
theorem M_tensor_N_is_correct :
  (M \ N ∪ N \ M) = correct_answer :=
sorry

end M_tensor_N_is_correct_l242_242022


namespace cubic_sum_l242_242103

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 :=
by {
  sorry
}

end cubic_sum_l242_242103


namespace focus_to_directrix_distance_tangent_line_equation_locus_of_Q_l242_242426

-- Define the parabola and the circle
def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def circle (x y : ℝ) : Prop := (x - 5)^2 + y^2 = 16

-- Prove the distance from the focus to the directrix of the given parabola
theorem focus_to_directrix_distance :
  ∀ (y : ℝ), parabola 0 y → 2 = 2 :=
by sorry

-- Prove the equation of line l given it is tangent to the circle at the midpoint of A and B
theorem tangent_line_equation {A B M : ℝ × ℝ} :
  ∃ (l : ℝ → ℝ), (circle M.1 M.2 ∧ M = (A + B) / 2) →
    (∀ x, l x = 1) ∨ (∀ x, l x = 9) :=
by sorry

-- Prove the locus of point Q given certain conditions
theorem locus_of_Q {A B Q O : ℝ × ℝ} :
  ∀ (x y : ℝ), 
    (parabola A.1 A.2 ∧ parabola B.1 B.2) ∧
    ((O, A) = ((0, 0), A) ∧ (O, B) = ((0, 0), B)) ∧
    (A ≠ B ∧ (O.1 * A.1 + O.2 * A.2) = 0 ∧
    Q ∈ segment ℝ ((0,0) : ℝ × ℝ) ((A + B) / 2) ∧
    ∀ x y, (x, y) = Q → O ⊥ AB) →
    x^2 - 4 * x + y^2 = 0 :=
by sorry

end focus_to_directrix_distance_tangent_line_equation_locus_of_Q_l242_242426


namespace quotient_of_numbers_l242_242965

noncomputable def larger_number : ℕ := 22
noncomputable def smaller_number : ℕ := 8

theorem quotient_of_numbers : (larger_number.toFloat / smaller_number.toFloat) = 2.75 := by
  sorry

end quotient_of_numbers_l242_242965


namespace only_number_smaller_than_neg3_l242_242724

theorem only_number_smaller_than_neg3 : 
  ∀ (x : ℤ), (x = -3 ∨ x = 2 ∨ x = 0 ∨ x = -4) → (x < -3 ↔ x = -4) := by
  intros x h
  cases h with h1 h23
  { left, apply gt.irrefl },
  cases h23 with h2 h3
  { right, left, apply gt_of_gt_of_ge nat.zero_lt_succ (le_of_eq rfl) },
  cases h3 with h0 h4
  { right, right, left, apply lt_irrefl },
  { right, right, right,
    show x < -3,
    exact lt_of_neg_of_lt dec_trivial (lt_neg_add' neg_succ_lt_zero) },
  sorry

end only_number_smaller_than_neg3_l242_242724


namespace binom_30_3_is_4060_l242_242367

theorem binom_30_3_is_4060 : Nat.choose 30 3 = 4060 := by
  sorry

end binom_30_3_is_4060_l242_242367


namespace tangent_line_eq_l242_242462

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := (1 / 3) * x^3 - a * x^2 + b * x

def f' (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^2 - 2 * a * x + b

def g (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := f x a b - 4 * x

theorem tangent_line_eq (a b : ℝ)
  (h1 : f' 0 a b = 1)
  (h2 : f' 2 a b = 1) :
  4 * 3 - (f 3 a b) - 9 = 0 :=
sorry

end tangent_line_eq_l242_242462


namespace company_allocation_salary_l242_242198

-- Definitions
def initial_salary : ℝ := 30000
def raise_rate : ℝ := 0.1
def tax_rate : ℝ := 0.13

-- Calculation of post-tax salary
def post_tax_salary : ℝ := initial_salary * (1 + raise_rate)

-- Formula to calculate the pre-tax salary from post-tax salary
def pre_tax_salary (post_tax : ℝ) (tax_rate : ℝ) : ℝ :=
  post_tax / (1 - tax_rate)

-- Theorem: proving the allocation amount
theorem company_allocation_salary : 
  pre_tax_salary post_tax_salary tax_rate = 37931 := 
by 
  sorry

end company_allocation_salary_l242_242198


namespace log_evaluation_l242_242756

theorem log_evaluation : ∀ (a : ℕ), a = 3 → log 3 (9^3) = 6 :=
by
  intro a ha
  have h1 : 9 = 3^2 := by sorry
  have h2 : (9^3) = (3^2)^3 := by rw [h1]
  have h3 : (3^2)^3 = 3^6 := by sorry
  have h4 : log 3 (3^6) = 6 := by sorry
  rw [h2, h3, h4]
  exact ha

end log_evaluation_l242_242756


namespace log_base_3_l242_242855

theorem log_base_3 (log_identity : ∀ (b a c : ℝ), log b (a^c) = c * log b a) :
  log 3 (9^3) = 6 :=
by
  have : 9 = 3^2 := by sorry
  have : log 3 3 = 1 := by sorry
  exact (log_identity 3 (3^2) 3).trans sorry

end log_base_3_l242_242855


namespace find_k_l242_242971

open Nat

def S (n : ℕ) : ℕ :=
  Integer.toNat $ (n.toBinary).count 1 -- toBinary converts n to its binary representation and count counts 1's in that binary list

def v (n : ℕ) : ℕ :=
  n - S n

theorem find_k (k : ℕ) : (∀ n : ℕ, n > 0 → 2 ^ ((k - 1) * n + 1) ∣ factorial (k * n) / factorial n) ↔ (∃ m : ℕ, k = 2 ^ m) :=
by
  sorry

end find_k_l242_242971


namespace probability_odd_product_l242_242000

theorem probability_odd_product :
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  let pairs := { (a, b) | a ∈ S ∧ b ∈ S ∧ a < b }
  let odd_product_pairs := { (a, b) | a ∈ S ∧ b ∈ S ∧ a < b ∧ a % 2 = 1 ∧ b % 2 = 1 }
  (finset.card odd_product_pairs : ℚ) / (finset.card pairs : ℚ) = 5 / 18 :=
by
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  let pairs := finset.filter (λ (p : ℕ × ℕ), p.1 < p.2) (finset.product (finset.filter (λ x, true) (finset.range 10)) (finset.filter (λ x, true) (finset.range 10)))
  let odd_product_pairs := finset.filter (λ (p : ℕ × ℕ), p.1 % 2 = 1 ∧ p.2 % 2 = 1) pairs
  have h_pairs : finset.card pairs = 36 := sorry
  have h_odd_product_pairs : finset.card odd_product_pairs = 10 := sorry
  exact (congr_arg (λ x, x : ℚ) h_odd_product_pairs) / (congr_arg (λ x, x : ℚ) h_pairs) ▸ sorry

end probability_odd_product_l242_242000


namespace log_base_3_of_9_cubed_l242_242900

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 :=
by assumptions
  have h1 : 9 = 3^2 := sorry,
  have h2 : 9^3 = (3^2)^3 := sorry,
  have h3 : (3^2)^3 = 3^(2*3) := sorry,
  have h4 : 3^(2*3) = 3^6 := sorry,
  have h5 : log 3 (3^6) = 6 := by apply log_pow,
  sorry

end log_base_3_of_9_cubed_l242_242900


namespace inequality_solution_set_l242_242039

open Set

variable {f : ℝ → ℝ}

-- Definition of the domain and the conditions of the function f(x).
def domain_f : Set ℝ := {x | 0 < x}

def condition_derivative (x : ℝ) : Prop := 
  x ∈ domain_f → differentiable_at ℝ f x

def condition_inequality (x : ℝ) : Prop := 
  x ∈ domain_f → f x > deriv f x

theorem inequality_solution_set :
  (∀ x, condition_derivative x) →
  (∀ x, condition_inequality x) →
  {x | e^(x+2) * f (x^2 - x) > e^(x^2) * f 2} = Ioo (-1 : ℝ) 0 ∪ Ioo 1 2 :=
sorry

end inequality_solution_set_l242_242039


namespace number_of_integer_length_chords_l242_242684

theorem number_of_integer_length_chords (A : ℝ × ℝ) (hA : A = (11, 2)) :
  ∀ (C : ℝ × ℝ) (r : ℝ), (C = (1, -2)) ∧ (r = 2) ∧ (∀ l : ℝ, (l ∈ ℤ) → 
  (∃ x1 y1 x2 y2, (x1^2 + y1^2 - 2*x1 + 4*y1 + 1 = 0) ∧ (x2^2 + y2^2 - 2*x2 + 4*y2 + 1 = 0) ∧ 
  ((x1, y1) = A ∨ (x2, y2) = A) ∧ sqrt ((x2 - x1)^2 + (y2 - y1)^2) = l) → l ∈ {1, 2, 3, 4}) →
  ∃ n : ℕ, n = 7 :=
sorry

end number_of_integer_length_chords_l242_242684


namespace equilateral_triangle_tangent_circles_radius_l242_242552

theorem equilateral_triangle_tangent_circles_radius :
  ∃ (a b : ℤ), (ABC side_length 16 ∧ mutually_tangent_circles_tangent_to_sides) →
  (∃ r, radius_expression r a b ∧ a + b = 52) :=
sorry

end equilateral_triangle_tangent_circles_radius_l242_242552


namespace sum_f_a_n_first_10_terms_l242_242015

noncomputable def f (x : ℝ) : ℝ := sorry

def a_n (n : ℕ) : ℝ := 1 + (n - 1)

lemma f_odd_and_periodic : 
  (∀ x : ℝ, f(-x) = -f(x)) ∧ 
  (∀ x k : ℤ, f(x + 2 * k) = f(x)) := sorry

theorem sum_f_a_n_first_10_terms :
  f(a_n 1) + f(a_n 2) + f(a_n 3) + f(a_n 4) + f(a_n 5) + 
  f(a_n 6) + f(a_n 7) + f(a_n 8) + f(a_n 9) + f(a_n 10) = 0 :=
by 
  have h_odd_periodic := f_odd_and_periodic,
  sorry

end sum_f_a_n_first_10_terms_l242_242015


namespace ernie_income_ratio_l242_242753

-- Define constants and properties based on the conditions
def previous_income := 6000
def jack_income := 2 * previous_income
def combined_income := 16800

-- Lean proof statement that the ratio of Ernie's current income to his previous income is 2/3
theorem ernie_income_ratio (current_income : ℕ) (h1 : current_income + jack_income = combined_income) :
    current_income / previous_income = 2 / 3 :=
sorry

end ernie_income_ratio_l242_242753


namespace sum_of_closest_integers_to_sqrt_40_l242_242244

theorem sum_of_closest_integers_to_sqrt_40 :
  ∃ (a b : ℤ), (a < real.sqrt 40 ∧ real.sqrt 40 < b) ∧ (int.of_nat a = 6) ∧ (int.of_nat b = 7) ∧ (a + b = 13) :=
by
  use 6, 7
  have h1 : real.sqrt 36 < real.sqrt 40, by sorry
  have h2 : real.sqrt 40 < real.sqrt 49, by sorry
  have h6 : real.sqrt 36 = 6, by sorry
  have h7 : real.sqrt 49 = 7, by sorry
  exact ⟨⟨h1, h2⟩, h6, h7, rfl⟩

end sum_of_closest_integers_to_sqrt_40_l242_242244


namespace log_base_3_of_9_cubed_l242_242820

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 := by
  sorry

end log_base_3_of_9_cubed_l242_242820


namespace min_modulus_complex_l242_242452

theorem min_modulus_complex (t : ℝ) : ∃ z : ℂ, z = (t-1) + (t+1) * complex.I ∧ (∀ t : ℝ, complex.abs z ≥ real.sqrt 2) := by
  sorry

end min_modulus_complex_l242_242452


namespace must_hold_inequality_l242_242392

variable (f : ℝ → ℝ)

noncomputable def condition : Prop := ∀ x > 0, x * (deriv^[2] f) x < 1

theorem must_hold_inequality (h : condition f) : f (Real.exp 1) < f 1 + 1 := 
sorry

end must_hold_inequality_l242_242392


namespace log_base_3_of_9_cubed_l242_242921
-- Import the necessary parts of the standard library

-- Define the statement to be proved
theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 :=
by
  sorry

end log_base_3_of_9_cubed_l242_242921


namespace min_value_48_l242_242021

noncomputable def min_value {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (h : 3 * a + b = 1) : ℝ :=
  1 / a + 27 / b

theorem min_value_48 {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (h : 3 * a + b = 1) : 
  min_value ha hb h = 48 := 
sorry

end min_value_48_l242_242021


namespace trajectory_equation_of_M_l242_242020

def F1 : (ℝ × ℝ) := (-5, 0)
def F2 : (ℝ × ℝ) := (5, 0)

def M (x y : ℝ) : Prop := abs (sqrt ((x + 5)^2 + y^2) - sqrt ((x - 5)^2 + y^2)) = 8

theorem trajectory_equation_of_M : ∀ x y : ℝ, M x y → x^2 / 16 - y^2 / 9 = 1 ∧ 0 < x := sorry

end trajectory_equation_of_M_l242_242020


namespace relationship_between_y_values_l242_242236

def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + |b| * x + c

theorem relationship_between_y_values
  (a b c y1 y2 y3 : ℝ)
  (h1 : quadratic_function a b c (-14 / 3) = y1)
  (h2 : quadratic_function a b c (5 / 2) = y2)
  (h3 : quadratic_function a b c 3 = y3)
  (axis_symmetry : -(|b| / (2 * a)) = -1)
  (h_pos : 0 < a) :
  y2 < y1 ∧ y1 < y3 :=
by
  sorry

end relationship_between_y_values_l242_242236


namespace log_base_3_of_9_cubed_l242_242830

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 := by
  -- Define the known conditions
  have h1 : 9 = 3^2 := by norm_num
  have h2 : log 3 (a^n) = n * log 3 a := sorry  -- Statement of the logarithmic identity
  
  -- Apply the conditions to prove the result
  calc
    log 3 (9^3)
        = log 3 ((3^2)^3) : by rw h1
    ... = log 3 (3^(2 * 3)) : by congr; norm_num
    ... = log 3 (3^6) : by norm_num
    ... = 6 : by rw [h2]; norm_num

end log_base_3_of_9_cubed_l242_242830


namespace find_amplitude_l242_242730

theorem find_amplitude (a b c d : ℝ) 
  (h1 : ∀ x, a * cos (b * x + c) + d ≤ 4)
  (h2 : ∀ x, a * cos (b * x + c) + d ≥ 0) :
  a = 2 := 
sorry

end find_amplitude_l242_242730


namespace log_base_3_of_9_cubed_l242_242769

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 :=
by
  -- Given conditions:
  -- 1. 9 = 3^2
  -- 2. log_b (a^n) = n * log_b a
  sorry

end log_base_3_of_9_cubed_l242_242769


namespace chords_intersect_probability_l242_242641

noncomputable def probability_chords_intersect (total_points : ℕ) (selected_points : fin 2024 → bool) : ℚ :=
  if total_points = 2024 ∧ selected_points 0 ∧ selected_points 1 = tt ∧ ¬selected_points 2 ∧ ¬selected_points 3 
  then 1 / 3 
  else 0

theorem chords_intersect_probability : 
  ∀ (total_points : ℕ) (selected_points : fin 2024 → bool),
  total_points = 2024 ∧ selected_points 0 ∧ selected_points 1 = tt ∧ ¬selected_points 2 ∧ ¬selected_points 3 
  → probability_chords_intersect total_points selected_points = 1 / 3 := 
by sorry

end chords_intersect_probability_l242_242641


namespace repeating_decimals_l242_242342

def max_repeating_decimal := "0.20120415" → "0.\overline{20120415}"
def min_repeating_decimal := "0.20120415" → "0.20\overline{120415}"

theorem repeating_decimals (x : String) 
  (h : x = "0.20120415") : 
  max_repeating_decimal x = "0.\overline{20120415}" ∧ min_repeating_decimal x = "0.20\overline{120415}" :=
by
  sorry

end repeating_decimals_l242_242342


namespace remainder_p11_minus_3_div_p_minus_2_l242_242409

def f (p : ℕ) : ℕ := p^11 - 3

theorem remainder_p11_minus_3_div_p_minus_2 : f 2 = 2045 := 
by 
  sorry

end remainder_p11_minus_3_div_p_minus_2_l242_242409


namespace sum_divisors_divisible_by_24_l242_242175

theorem sum_divisors_divisible_by_24 {n : ℕ} (h : (n + 1) % 24 = 0) :
  ∃ k : ℕ, n = 24 * k - 1 ∧ 
  ∀ d ∈ divisors n, d % 24 = 0 := sorry

end sum_divisors_divisible_by_24_l242_242175


namespace inverse_proportion_points_l242_242117

theorem inverse_proportion_points (x1 x2 x3 : ℝ) :
  (10 / x1 = -5) →
  (10 / x2 = 2) →
  (10 / x3 = 5) →
  x1 < x3 ∧ x3 < x2 :=
by sorry

end inverse_proportion_points_l242_242117


namespace evaluate_log_l242_242795

noncomputable def log_base (b x : ℝ) : ℝ := real.log x / real.log b

theorem evaluate_log : log_base 3 (9^3) = 6 := by
  have h1 : 9 = 3^2 := by norm_num
  have h2 : 9^3 = (3^2)^3 := by norm_num
  rw [h2, pow_mul]
  have h3 : log_base 3 (3^6) = 6 := by
    rw [← real.log_pow, real.log_mul, mul_comm, ← real.log_rpow, mul_div, real.log_self]
    norm_num
  exact h3

end evaluate_log_l242_242795


namespace find_ac_sum_l242_242619

-- Define the absolute value function for utility
def abs_val (x: ℝ) : ℝ := |x|

-- Define the first function as given
def f1 (x a b : ℝ) : ℝ := -abs_val (x - (a + 1)) + b

-- Define the second function as given
def f2 (x c d : ℝ) : ℝ := abs_val (x - (c - 1)) + (d - 1)

-- Define the existence of the intersection points
def intersects (f1 f2 : ℝ → ℝ) (x y : ℝ) : Prop := (f1 x = y) ∧ (f2 x = y)

-- Given the conditions as stated in the problem
def conditions (a b c d : ℝ) : Prop :=
  intersects (f1 _ a b) (f2 _ c d) 3 4 ∧ intersects (f1 _ a b) (f2 _ c d) 7 2

-- Main theorem to prove
theorem find_ac_sum {a b c d : ℝ} (h : conditions a b c d) : a + c = 10 := 
by sorry

end find_ac_sum_l242_242619


namespace smallest_integer_base_cube_l242_242650

theorem smallest_integer_base_cube (b : ℤ) (h1 : b > 5) (h2 : ∃ k : ℤ, 1 * b + 2 = k^3) : b = 6 :=
sorry

end smallest_integer_base_cube_l242_242650


namespace log_three_nine_cubed_l242_242957

theorem log_three_nine_cubed : ∀ (log : ℕ → ℕ → ℕ), log 3 9 = 2 → log 3 3 = 1 → (∀ (a b n : ℕ), log a (b ^ n) = n * log a b) → log 3 (9 ^ 3) = 6 :=
by
  intros log h1 h2 h3
  sorry

end log_three_nine_cubed_l242_242957


namespace arithmetic_sequence_common_difference_l242_242028

theorem arithmetic_sequence_common_difference (a : ℕ → ℤ) (d : ℤ) (h_arith_seq: ∀ n, a n = a 1 + (n - 1) * d) 
  (h_cond1 : a 3 + a 9 = 4 * a 5) (h_cond2 : a 2 = -8) : 
  d = 4 :=
sorry

end arithmetic_sequence_common_difference_l242_242028


namespace eval_expression_l242_242734

theorem eval_expression : (2 * sin (Real.pi / 6)) - Real.cbrt 8 + (2 - Real.pi)^0 + (-1)^(2023 : ℕ) = -1 := by
  -- insert necessary proofs and steps here
  sorry

end eval_expression_l242_242734


namespace deposit_amount_l242_242316

theorem deposit_amount (P : ℝ) (deposit remaining : ℝ) (h1 : deposit = 0.1 * P) (h2 : remaining = P - deposit) (h3 : remaining = 1350) : 
  deposit = 150 := 
by
  sorry

end deposit_amount_l242_242316


namespace binom_30_3_eq_4060_l242_242366

theorem binom_30_3_eq_4060 : Nat.choose 30 3 = 4060 := by
  sorry

end binom_30_3_eq_4060_l242_242366


namespace find_b_l242_242030

-- Definitions
variable (k : ℤ) (b : ℤ)
def x := 3 * k
def y := 4 * k
def z := 7 * k

-- Conditions
axiom ratio : x / y = 3 / 4 ∧ y / z = 4 / 7
axiom equation : y = 15 * b - 5

-- Theorem statement
theorem find_b : ∃ b : ℤ, 4 * k = 15 * b - 5 ∧ b = 3 :=
by
  sorry

end find_b_l242_242030


namespace count_three_digit_numbers_with_odd_sum_eq_eighteen_l242_242525

open List

def digits : List ℕ := [1, 2, 3, 4, 5]

def is_odd (n : ℕ) : Prop := n % 2 = 1

def three_digit_numbers_with_odd_sum : List (ℕ × ℕ × ℕ) :=
  digits.bind (fun d1 =>
  (digits.erase d1).bind (fun d2 =>
  (digits.erase d1).erase d2).map (fun d3 => (d1, d2, d3)))

def has_odd_sum (n : ℕ × ℕ × ℕ) : Prop :=
  is_odd (n.1 + n.2 + n.3)

def count_odd_sum_numbers : ℕ :=
  (three_digit_numbers_with_odd_sum.filter has_odd_sum).length

theorem count_three_digit_numbers_with_odd_sum_eq_eighteen :
  count_odd_sum_numbers = 18 :=
  sorry

end count_three_digit_numbers_with_odd_sum_eq_eighteen_l242_242525


namespace gcd_fact8_fact10_l242_242273

-- Define the factorials
def fact : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * fact n

-- State the problem conditions
theorem gcd_fact8_fact10 : gcd (fact 8) (fact 10) = fact 8 := by
  sorry

end gcd_fact8_fact10_l242_242273


namespace count_convex_cyclic_quadrilaterals_is_1505_l242_242071

-- Define a quadrilateral using its sides
structure Quadrilateral where
  a b c d : ℕ
  deriving Repr, DecidableEq

-- Define what a convex cyclic quadrilateral is, given the integer sides and the perimeter condition
def isConvexCyclicQuadrilateral (q : Quadrilateral) : Prop :=
 q.a + q.b + q.c + q.d = 36 ∧
 q.a > 0 ∧ q.b > 0 ∧ q.c > 0 ∧ q.d > 0 ∧
 q.a + q.b > q.c + q.d ∧ q.c + q.d > q.a + q.b ∧
 q.b + q.c > q.d + q.a ∧ q.d + q.a > q.b + q.c

-- Noncomputable definition to count all convex cyclic quadrilaterals
noncomputable def countConvexCyclicQuadrilaterals : ℕ :=
  sorry

-- The theorem stating the count is equal to 1505
theorem count_convex_cyclic_quadrilaterals_is_1505 :
  countConvexCyclicQuadrilaterals = 1505 :=
  sorry

end count_convex_cyclic_quadrilaterals_is_1505_l242_242071


namespace relationship_between_y_values_l242_242235

-- Define the quadratic function given the constraints
def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + abs b * x + c

-- Define the points (x1, y1), (x2, y2), (x3, y3)
def x1 := -14 / 3
def x2 := 5 / 2
def x3 := 3

def y1 (a b c : ℝ) : ℝ := quadratic_function a b c x1
def y2 (a b c : ℝ) : ℝ := quadratic_function a b c x2
def y3 (a b c : ℝ) : ℝ := quadratic_function a b c x3

theorem relationship_between_y_values 
  (a b c : ℝ) (h1 : - (abs b) / (2 * a) = -1) 
  (y1_value : ℝ := y1 a b c) 
  (y2_value : ℝ := y2 a b c) 
  (y3_value : ℝ := y3 a b c) : 
  y2_value < y1_value ∧ y1_value < y3_value := 
by 
  sorry

end relationship_between_y_values_l242_242235


namespace number_of_tiles_in_row_l242_242211

noncomputable def width_of_room_is_feet (w : ℝ) : Prop := 2 * w^2 = 360

noncomputable def room_width_in_inches (w : ℝ) : ℝ := w * 12

noncomputable def tiles_in_each_row_along_width (w_inches : ℝ) : ℝ := w_inches / 8

theorem number_of_tiles_in_row (w : ℝ) (h_w : width_of_room_is_feet w) :
  tiles_in_each_row_along_width (room_width_in_inches w) = 9 * real.sqrt 5 :=
sorry

end number_of_tiles_in_row_l242_242211


namespace max_value_of_rocks_l242_242343

-- Definition of the value and weight of each type of rock
def rock (weight value : Nat) : Type := (weight, value)

-- The rocks available
def rocks : List (Nat × Nat) := [(7, 20), (3, 10), (2, 4)]

-- The maximum weight Carl can carry
def maxWeight : Nat := 21

-- Carl can carry up to 21 pounds. Prove the maximum value he can carry is $70.
theorem max_value_of_rocks : 
    ∃ (x y z : Nat), 
    (x * 7 + y * 3 + z * 2 ≤ maxWeight) ∧ 
    (x + y + z > 0) ∧ 
    (∀ (x y z : Nat), 
     x * 7 + y * 3 + z * 2 ≤ maxWeight → 
     20 * x + 10 * y + 4 * z ≤ 70) := 
sorry

end max_value_of_rocks_l242_242343


namespace prime_intersection_even_is_two_l242_242541

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def set_prime : set ℕ := {n | is_prime n}
def set_even : set ℕ := {n | is_even n}

theorem prime_intersection_even_is_two : set_prime ∩ set_even = {2} :=
sorry

end prime_intersection_even_is_two_l242_242541


namespace convex_cyclic_quadrilaterals_count_l242_242072

theorem convex_cyclic_quadrilaterals_count :
  let num_quadrilaterals := ∑ i in (finset.range 36).powerset.filter(λ s, s.card = 4 
    ∧ let (a, b, c, d) := classical.some (vector.sorted_enum s)
    in a + b + c + d = 36 ∧ a < b + c + d ∧ b < a + c + d ∧ c < a + b + d ∧ d < a + b + c 
  ),
  finset.count :=
  num_quadrilaterals = 819 :=
begin
  sorry
end

end convex_cyclic_quadrilaterals_count_l242_242072


namespace log_base_three_of_nine_cubed_l242_242877

theorem log_base_three_of_nine_cubed : log 3 (9^3) = 6 :=
by
  have h1 : 9 = 3^2 := by sorry
  have h2 : ∀ (a m n: ℕ), (a^m)^n = a^(m * n) := by 
    intros a m n
    exact pow_mul a m n
  have h3 : ∀ (b n: ℕ), log b (b^n) = n := by sorry
  sorry

end log_base_three_of_nine_cubed_l242_242877


namespace sum_of_sequence_transform_l242_242036

noncomputable def sequence_a (n : ℕ) : ℕ := 2^n

noncomputable def sum_S (n : ℕ) : ℕ := (Finset.range n).sum sequence_a

noncomputable def sum_T (n : ℕ) : ℕ := (Finset.range n).sum (λ k, (sequence_a k)^2)

noncomputable def sum_of_transformed_sequence (n : ℕ) : ℝ := (Finset.range n).sum (λ k, ((k-1) / (k * (k+1))) * sequence_a k)

variables {n : ℕ}

theorem sum_of_sequence_transform :
  sum_S n^2 + 4 * sum_S n - 3 * sum_T n = 0 → 
  sum_of_transformed_sequence n = (2^(n+1) / (n+1) - 2) :=
sorry

end sum_of_sequence_transform_l242_242036


namespace log_base_three_of_nine_cubed_l242_242874

theorem log_base_three_of_nine_cubed : log 3 (9^3) = 6 :=
by
  have h1 : 9 = 3^2 := by sorry
  have h2 : ∀ (a m n: ℕ), (a^m)^n = a^(m * n) := by 
    intros a m n
    exact pow_mul a m n
  have h3 : ∀ (b n: ℕ), log b (b^n) = n := by sorry
  sorry

end log_base_three_of_nine_cubed_l242_242874


namespace complement_M_eq_interval_l242_242060

-- Definition of the set M
def M : Set ℝ := { x | x * (x - 3) > 0 }

-- Universal set is ℝ
def U : Set ℝ := Set.univ

-- Theorem to prove the complement of M in ℝ is [0, 3]
theorem complement_M_eq_interval :
  U \ M = { x | 0 ≤ x ∧ x ≤ 3 } :=
by
  sorry

end complement_M_eq_interval_l242_242060


namespace log_base_3_of_9_cubed_l242_242842

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 := by
  -- Define the known conditions
  have h1 : 9 = 3^2 := by norm_num
  have h2 : log 3 (a^n) = n * log 3 a := sorry  -- Statement of the logarithmic identity
  
  -- Apply the conditions to prove the result
  calc
    log 3 (9^3)
        = log 3 ((3^2)^3) : by rw h1
    ... = log 3 (3^(2 * 3)) : by congr; norm_num
    ... = log 3 (3^6) : by norm_num
    ... = 6 : by rw [h2]; norm_num

end log_base_3_of_9_cubed_l242_242842


namespace total_nominal_income_l242_242572

noncomputable def monthly_income (principal : ℝ) (rate : ℝ) (months : ℕ) : ℝ :=
  principal * ((1 + rate) ^ months - 1)

def total_income : ℝ :=
  let rate := 0.06 / 12
  let principal := 8700
  (monthly_income principal rate 6) + 
  (monthly_income principal rate 5) + 
  (monthly_income principal rate 4) + 
  (monthly_income principal rate 3) + 
  (monthly_income principal rate 2) + 
  (monthly_income principal rate 1)

theorem total_nominal_income :
  total_income = 921.15 :=
by
  sorry

end total_nominal_income_l242_242572


namespace count_distinct_ways_l242_242302

theorem count_distinct_ways (p : ℕ × ℕ → ℕ) (h_condition : ∃ j : ℕ × ℕ, j ∈ [(0, 0), (0, 1)] ∧ p j = 4)
  (h_grid_size : ∀ i : ℕ × ℕ, i ∈ [(0, 0), (0, 1), (1, 0), (1, 1)] → 1 ≤ p i ∧ p i ≤ 4)
  (h_distinct : ∀ i j : ℕ × ℕ, i ∈ [(0, 0), (0, 1), (1, 0), (1, 1)] → j ∈ [(0, 0), (0, 1), (1, 0), (1, 1)] → i ≠ j → p i ≠ p j) :
  ∃! l : Finset (ℕ × ℕ → ℕ), l.card = 12 :=
by
  sorry

end count_distinct_ways_l242_242302


namespace difference_q_r_l242_242305

-- Definitions for the problem
variables (p q r : ℕ) (x : ℕ)

-- Conditions derived from the problem
axiom ratio_condition : p = 3 * x ∧ q = 7 * x ∧ r = 12 * x
axiom difference_p_q : q - p = 4400

-- Theorem stating the proof problem
theorem difference_q_r : q - p = 4400 → r - q = 5500 :=
begin
  intro h1,
  rcases ratio_condition with ⟨hp, hq, hr⟩,
  have h2: 7 * x - 3 * x = 4400 := by { rw [hq, hp], exact h1 },
  have h3: 4 * x = 4400 := by linarith,
  have h4: x = 1100 := by linarith,
  have h5: 12 * x - 7 * x = 5500 := by { rw h4, norm_num },
  rw [hr, hq],
  exact h5,
end

end difference_q_r_l242_242305


namespace angle_part_a_angle_part_b_l242_242405

noncomputable def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

noncomputable def magnitude (a : ℝ × ℝ) : ℝ :=
  Real.sqrt (a.1^2 + a.2^2)

noncomputable def angle_between_vectors (a b : ℝ × ℝ) : ℝ :=
  Real.arccos ((dot_product a b) / (magnitude a * magnitude b))

theorem angle_part_a :
  angle_between_vectors (4, 0) (2, -2) = Real.arccos (Real.sqrt 2 / 2) :=
by
  sorry

theorem angle_part_b :
  angle_between_vectors (5, -3) (3, 5) = Real.pi / 2 :=
by
  sorry

end angle_part_a_angle_part_b_l242_242405


namespace coefficient_of_6th_term_expansion_l242_242167

noncomputable def integral_value : ℝ := ∫ x in (-1:ℝ)..1, (sin x + 1)

theorem coefficient_of_6th_term_expansion :
  (∫ x in (-1:ℝ)..1, (sin x + 1)) = integral_value →
  let a := integral_value in
  let expansion := (a * x^2 - (1/x))^6 in
  let T6 := -12 * x^(-3) in
  (coeff (expansion.coeff) 5).1 = -12 :=
begin
  sorry
end

end coefficient_of_6th_term_expansion_l242_242167


namespace sequence_general_term_l242_242406

noncomputable def a_sequence : ℕ → ℝ
| 0     := 0
| 1     := 0
| (n+2) := 6 * a_sequence (n+1) - 9 * a_sequence n + 2^n + n

theorem sequence_general_term (n : ℕ) : 
  a_sequence n = (n + 1) / 4 + 2^n - (5 / 3) * 3^n + (5 / 12) * n * 3^n :=
sorry

end sequence_general_term_l242_242406


namespace log_base_3_l242_242848

theorem log_base_3 (log_identity : ∀ (b a c : ℝ), log b (a^c) = c * log b a) :
  log 3 (9^3) = 6 :=
by
  have : 9 = 3^2 := by sorry
  have : log 3 3 = 1 := by sorry
  exact (log_identity 3 (3^2) 3).trans sorry

end log_base_3_l242_242848


namespace total_nominal_income_l242_242573

noncomputable def monthly_income (principal : ℝ) (rate : ℝ) (months : ℕ) : ℝ :=
  principal * ((1 + rate) ^ months - 1)

def total_income : ℝ :=
  let rate := 0.06 / 12
  let principal := 8700
  (monthly_income principal rate 6) + 
  (monthly_income principal rate 5) + 
  (monthly_income principal rate 4) + 
  (monthly_income principal rate 3) + 
  (monthly_income principal rate 2) + 
  (monthly_income principal rate 1)

theorem total_nominal_income :
  total_income = 921.15 :=
by
  sorry

end total_nominal_income_l242_242573


namespace log_three_pow_nine_pow_three_eq_six_l242_242944

open Real

noncomputable def eval_log_three_pow_nine_pow_three : ℝ :=
  log 3 (9^3)

theorem log_three_pow_nine_pow_three_eq_six :
  eval_log_three_pow_nine_pow_three = 6 := by
  sorry

end log_three_pow_nine_pow_three_eq_six_l242_242944


namespace sum_lent_is_300_l242_242331

-- Define the conditions
def interest_rate : ℕ := 4
def time_period : ℕ := 8
def interest_amounted_less : ℕ := 204

-- Prove that the sum lent P is 300 given the conditions
theorem sum_lent_is_300 (P : ℕ) : 
  (P * interest_rate * time_period / 100 = P - interest_amounted_less) -> P = 300 := by
  sorry

end sum_lent_is_300_l242_242331


namespace cost_of_paving_l242_242621

noncomputable def length : Float := 5.5
noncomputable def width : Float := 3.75
noncomputable def cost_per_sq_meter : Float := 600

theorem cost_of_paving :
  (length * width * cost_per_sq_meter) = 12375 := by
  sorry

end cost_of_paving_l242_242621


namespace cubic_polynomial_root_l242_242968

theorem cubic_polynomial_root : ∃ Q : Polynomial ℤ, Q.monic ∧ (Q (∛3 + 2) = 0) :=
by
  let x := (∛3 + 2)
  let Q := Polynomial.Cubic {coeffs := [ -11, 12, -6, 1 ]}
  use Q
  show Q.monic ∧ Q x = 0
  sorry

end cubic_polynomial_root_l242_242968


namespace french_fries_count_l242_242131

theorem french_fries_count 
    (total_students : ℕ = 25) 
    (likes_burgers : ℕ = 10) 
    (likes_both : ℕ = 6) 
    (likes_neither : ℕ = 6)
    : ∃ (F : ℕ), F = 15 :=
by
  -- here goes the proof
  sorry

end french_fries_count_l242_242131


namespace log_three_nine_cubed_l242_242952

theorem log_three_nine_cubed : ∀ (log : ℕ → ℕ → ℕ), log 3 9 = 2 → log 3 3 = 1 → (∀ (a b n : ℕ), log a (b ^ n) = n * log a b) → log 3 (9 ^ 3) = 6 :=
by
  intros log h1 h2 h3
  sorry

end log_three_nine_cubed_l242_242952


namespace octagon_area_l242_242644

theorem octagon_area (O : Point) (A B C D E F G H : Point) 
  (square1 square2 : square) 
  (h_shared_center : square1.center = O ∧ square2.center = O)
  (h_side_length : square1.side_length = 2 ∧ square2.side_length = 2) 
  (h_length_AB : distance A B = 20 / 49) :
  ∃ (m n : ℕ), m.gcd n = 1 ∧ 8 * (1/2 * (20 / 49) * square1.side_length) = m / n ∧ m+n = 209 := 
sorry

end octagon_area_l242_242644


namespace units_digit_sum_factorials_l242_242733

theorem units_digit_sum_factorials : 
  (∑ i in Finset.range 100, Nat.factorial (i + 1)) % 10 = 3 :=
by
  sorry

end units_digit_sum_factorials_l242_242733


namespace smallest_period_of_periodic_and_anti_symmetric_l242_242424

noncomputable def smallest_positive_period (f : ℝ → ℝ) (a : ℝ) : ℝ :=
  if h : a > 0 ∧ ∀ (x : ℝ), f(x - a) = -f(x) then 2 * a else 0

theorem smallest_period_of_periodic_and_anti_symmetric (f : ℝ → ℝ) (a : ℝ) (ha : a > 0) 
  (hf : ∀ (x : ℝ), f(x - a) = -f(x)) : smallest_positive_period f a = 2 * a := by 
  sorry

end smallest_period_of_periodic_and_anti_symmetric_l242_242424


namespace problem_1_problem_2_problem_3_l242_242460

-- Problem 1
theorem problem_1 (a : ℝ) (h_pos : a > 0) 
  (h_increasing : ∀ x : ℝ, (1 < x) → deriv (λ x, (1 - x) / (a * x) + log x) x ≥ 0) : 
  1 ≤ a := sorry

-- Problem 2
theorem problem_2 : 
  ∃ x₀, ∀ x ∈ set.Ici (0 : ℝ), g x ≤ g x₀ ∧ g x₀ = 0 :=
sorry
  where g (x : ℝ) := log (1 + x) - x 

-- Problem 3
theorem problem_3 (a b : ℝ) (h_a : a > 1) (h_b : b > 0) :
  1 / (a + b) ≤ log ((a + b) / b) ∧ log ((a + b) / b) < a / b :=
sorry

end problem_1_problem_2_problem_3_l242_242460


namespace evaluate_log_l242_242790

noncomputable def log_base (b x : ℝ) : ℝ := real.log x / real.log b

theorem evaluate_log : log_base 3 (9^3) = 6 := by
  have h1 : 9 = 3^2 := by norm_num
  have h2 : 9^3 = (3^2)^3 := by norm_num
  rw [h2, pow_mul]
  have h3 : log_base 3 (3^6) = 6 := by
    rw [← real.log_pow, real.log_mul, mul_comm, ← real.log_rpow, mul_div, real.log_self]
    norm_num
  exact h3

end evaluate_log_l242_242790


namespace symmetric_points_sum_l242_242038

theorem symmetric_points_sum (a b : ℤ) (h1 : A = (a, 2022)) (h2 : A' = (-2023, b)) (symmetric : (A, A') are_symmetric_origin) : a + b = 1 := by
  sorry

end symmetric_points_sum_l242_242038


namespace percentage_of_3rd_graders_l242_242627

theorem percentage_of_3rd_graders (students_jackson students_madison : ℕ)
  (percent_3rd_grade_jackson percent_3rd_grade_madison : ℝ) :
  students_jackson = 200 → percent_3rd_grade_jackson = 25 →
  students_madison = 300 → percent_3rd_grade_madison = 35 →
  ((percent_3rd_grade_jackson / 100 * students_jackson +
    percent_3rd_grade_madison / 100 * students_madison) /
   (students_jackson + students_madison) * 100) = 31 :=
by 
  intros hjackson_percent hmpercent 
    hpercent_jack_percent hpercent_mad_percent
  -- Proof Placeholder
  sorry

end percentage_of_3rd_graders_l242_242627


namespace profit_450_l242_242321

-- Define the conditions
def cost_per_garment : ℕ := 40
def wholesale_price : ℕ := 60

-- Define the piecewise function for wholesale price P
noncomputable def P (x : ℕ) : ℕ :=
  if h : 0 < x ∧ x ≤ 100 then wholesale_price
  else if h : 100 < x ∧ x ≤ 500 then 62 - x / 50
  else 0

-- Define the profit function L
noncomputable def L (x : ℕ) : ℕ :=
  if h : 0 < x ∧ x ≤ 100 then (P x - cost_per_garment) * x
  else if h : 100 < x ∧ x ≤ 500 then (22 * x - x^2 / 50)
  else 0

-- State the theorem
theorem profit_450 : L 450 = 5850 :=
by
  sorry

end profit_450_l242_242321


namespace number_of_valid_arrangements_l242_242325

-- Define the types for crops and sections
inductive Crop
| corn | wheat | soybeans | potatoes | oats

-- Define the 2 by 2 grid as a tuple of crops
structure Field :=
  (sec1 : Crop)
  (sec2 : Crop)
  (sec3 : Crop)
  (sec4 : Crop)

-- Define a function to check crop adjacency constraints
def valid_arrangement (f : Field) : Prop :=
  (f.sec1 ≠ Crop.corn ∨ f.sec2 ≠ Crop.wheat) ∧
  (f.sec1 ≠ Crop.corn ∨ f.sec3 ≠ Crop.wheat) ∧
  (f.sec2 ≠ Crop.corn ∨ f.sec4 ≠ Crop.wheat) ∧
  (f.sec2 ≠ Crop.soybeans ∨ f.sec4 ≠ Crop.potatoes) ∧
  (f.sec3 ≠ Crop.corn ∨ f.sec4 ≠ Crop.oats) ∧
  (f.sec1 ≠ Crop.corn ∨ f.sec4 ≠ Crop.oats)

-- The main theorem stating the number of valid field configurations
theorem number_of_valid_arrangements : ∃ n : ℕ, n = 157 ∧ ∀ f : Field, valid_arrangement f -> (number_of_configurations f = 157) :=
sorry

end number_of_valid_arrangements_l242_242325


namespace find_other_integer_l242_242535

theorem find_other_integer (x y : ℤ) (h1 : 4 * x + 3 * y = 150) (h2 : x = 15 ∨ y = 15) : y = 30 :=
by
  sorry

end find_other_integer_l242_242535


namespace correct_calculation_l242_242488

theorem correct_calculation (x : ℝ) (h : 5.46 - x = 3.97) : 5.46 + x = 6.95 := by
  sorry

end correct_calculation_l242_242488


namespace log_base_3_of_9_cubed_l242_242843

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 := by
  -- Define the known conditions
  have h1 : 9 = 3^2 := by norm_num
  have h2 : log 3 (a^n) = n * log 3 a := sorry  -- Statement of the logarithmic identity
  
  -- Apply the conditions to prove the result
  calc
    log 3 (9^3)
        = log 3 ((3^2)^3) : by rw h1
    ... = log 3 (3^(2 * 3)) : by congr; norm_num
    ... = log 3 (3^6) : by norm_num
    ... = 6 : by rw [h2]; norm_num

end log_base_3_of_9_cubed_l242_242843


namespace sum_even_integers_between_200_and_400_l242_242287

theorem sum_even_integers_between_200_and_400 : 
  (Finset.sum (Finset.filter (λ x, x % 2 = 0) (Finset.range 401)) 
    - Finset.sum (Finset.filter (λ x, x % 2 = 0) (Finset.range 201)))  = 30100 :=
begin
  sorry
end

end sum_even_integers_between_200_and_400_l242_242287


namespace total_nominal_income_l242_242568

theorem total_nominal_income
  (c1 : 8700 * ((1 + 0.06 / 12) ^ 6 - 1) = 264.28)
  (c2 : 8700 * ((1 + 0.06 / 12) ^ 5 - 1) = 219.69)
  (c3 : 8700 * ((1 + 0.06 / 12) ^ 4 - 1) = 175.31)
  (c4 : 8700 * ((1 + 0.06 / 12) ^ 3 - 1) = 131.15)
  (c5 : 8700 * ((1 + 0.06 / 12) ^ 2 - 1) = 87.22)
  (c6 : 8700 * (1 + 0.06 / 12 - 1) = 43.5) :
  264.28 + 219.69 + 175.31 + 131.15 + 87.22 + 43.5 = 921.15 := by
  sorry

end total_nominal_income_l242_242568


namespace solve_for_y_l242_242212

theorem solve_for_y (y : ℝ) (h_sum : (1 + 99) * 99 / 2 = 4950)
  (h_avg : (4950 + y) / 100 = 50 * y) : y = 4950 / 4999 :=
by
  sorry

end solve_for_y_l242_242212


namespace triangle_area_constant_l242_242465

noncomputable def parabola (p : ℝ) (h : p > 0) : set (ℝ × ℝ) :=
{ P | ∃ (x y : ℝ), y^2 = 2 * p * x ∧ P = (x, y) }

noncomputable def hyperbola : set (ℝ × ℝ) :=
{ Q | ∃ (x y : ℝ), y = - (1 / x) ∧ Q = (x, y) }

noncomputable def intersection_point (p : ℝ) (h : p > 0) : ℝ × ℝ :=
(1 / real.cbrt (2 * p), - real.cbrt (2 * p))

noncomputable def tangent_point_parabola (p : ℝ) (h : p > 0) : ℝ × ℝ :=
(4 / real.cbrt (2 * p), 2 * real.cbrt (2 * p))

noncomputable def tangent_point_hyperbola (p : ℝ) (h : p > 0) : ℝ × ℝ :=
(-2 / real.cbrt (2 * p), real.cbrt (2 * p) / 2)

def triangle_area (A B C : ℝ × ℝ) : ℝ :=
0.5 * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)|

theorem triangle_area_constant (p : ℝ) (hp : p > 0) :
  triangle_area (intersection_point p hp) (tangent_point_parabola p hp) (tangent_point_hyperbola p hp) = 27 / 4 :=
sorry

end triangle_area_constant_l242_242465


namespace evaluate_log_l242_242786

noncomputable def log_base (b x : ℝ) : ℝ := real.log x / real.log b

theorem evaluate_log : log_base 3 (9^3) = 6 := by
  have h1 : 9 = 3^2 := by norm_num
  have h2 : 9^3 = (3^2)^3 := by norm_num
  rw [h2, pow_mul]
  have h3 : log_base 3 (3^6) = 6 := by
    rw [← real.log_pow, real.log_mul, mul_comm, ← real.log_rpow, mul_div, real.log_self]
    norm_num
  exact h3

end evaluate_log_l242_242786


namespace triangle_ratio_is_right_triangle_triangle_is_right_angled_l242_242507

theorem triangle_ratio (k : ℝ) (h_sum : k + 2 * k + 3 * k = 180) : k = 30 :=
  sorry

theorem is_right_triangle (a b c : ℝ) (h1 : a + b + c = 180)
  (h2 : a = 30) (h3 : b = 60) (h4 : c = 90) : 
  c = 90 := by 
  rw h4
  rfl

-- Main theorem combining the results
theorem triangle_is_right_angled (a b c : ℝ) 
  (h_ratio : ∃ k, a = k ∧ b = 2 * k ∧ c = 3 * k)
  (h_sum : a + b + c = 180) : c = 90 := by
  obtain ⟨k, h_a, h_b, h_c⟩ := h_ratio
  have h_k : k = 30 := triangle_ratio k (by rw [h_a, h_b, h_c]; exact h_sum)
  rw [h_a, h_b, h_c, h_k]
  exact is_right_triangle 30 60 90 (by norm_num) (by rfl) (by rfl) (by rfl)

end triangle_ratio_is_right_triangle_triangle_is_right_angled_l242_242507


namespace find_x_arithmetic_sequence_l242_242967

def fractional_part (x : ℝ) : ℝ := x - Real.floor x

theorem find_x_arithmetic_sequence (x : ℝ) (h_nonzero : x ≠ 0)
  (h_arith_seq : fractional_part (fractional_part x ^ 2) + (2 * fractional_part x)) 
  (h_floor_eq : Real.floor x = 2 * fractional_part x) :
  x = 3 / 2 := 
sorry

end find_x_arithmetic_sequence_l242_242967


namespace max_distance_from_origin_to_curve_l242_242024

noncomputable def curve (θ : ℝ) : ℝ × ℝ :=
  (3 + Real.cos θ, Real.sin θ)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem max_distance_from_origin_to_curve : ∀ θ : ℝ, distance (0, 0) (curve θ) ≤ 4 :=
sorry  -- Proof is omitted

end max_distance_from_origin_to_curve_l242_242024


namespace class_fund_after_trip_l242_242565

variable (initial_fund contribution_per_student trip_cost_per_student num_students : ℕ)
variable (fund_after_contributions total_trip_cost final_fund : ℕ)

def total_contributed_by_students := contribution_per_student * num_students
def total_fund := initial_fund + total_contributed_by_students
def total_trip_cost := trip_cost_per_student * num_students
def final_fund := total_fund - total_trip_cost

theorem class_fund_after_trip
    (h1 : initial_fund = 50)
    (h2 : contribution_per_student = 5)
    (h3 : trip_cost_per_student = 7)
    (h4 : num_students = 20) :
    final_fund = 10 := by
  sorry

end class_fund_after_trip_l242_242565


namespace shortest_ribbon_length_is_10_l242_242532

noncomputable def shortest_ribbon_length (L : ℕ) : Prop :=
  (∃ k1 : ℕ, L = 2 * k1) ∧ (∃ k2 : ℕ, L = 5 * k2)

theorem shortest_ribbon_length_is_10 : shortest_ribbon_length 10 :=
by
  sorry

end shortest_ribbon_length_is_10_l242_242532


namespace smallest_positive_integer_n_l242_242653

noncomputable def smallest_n : ℕ :=
  Inf { n : ℕ | ∃ (k m : ℕ), 3 * n = (3 * k) ^ 2 ∧ 5 * n = (5 * m) ^ 3 }

theorem smallest_positive_integer_n : smallest_n = 1875 :=
sorry

end smallest_positive_integer_n_l242_242653


namespace solve_geometric_sequence_and_sum_l242_242524

noncomputable theory

open_locale big_operators

variables {a : ℕ → ℝ} {b : ℕ → ℝ} {T : ℕ → ℝ}

-- ℝ is considered for logarithmic calculations, n is ℕ (Natural numbers)

def geom_seq (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = a n * q

theorem solve_geometric_sequence_and_sum (a : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ)
  (h1 : geom_seq a 2)
  (h2 : a 1 * a 6 = 32)
  (h3 : a 2 + a 5 = 18)
  (b_def : ∀ n, b n = a n + log (a (n + 1)) / log 2)
  (T_def : ∀ n, T n = ∑ i in finset.range n, b i) :
  (∀ n, a n = 2 ^ (n - 1)) ∧ (∀ n, T n = 2 ^ n - 1 + n * (n + 1) / 2) :=
by { sorry }

end solve_geometric_sequence_and_sum_l242_242524


namespace largest_among_ab_ab_logb_a_l242_242076

-- definitions and conditions
variables (a b : ℝ)
variable h : 0 < a ∧ a < b ∧ b < 1

theorem largest_among_ab_ab_logb_a :
  (ab < log b a ∧ a^b < log b a) :=
begin
  -- sorry used to skip the proof
  sorry,
end

end largest_among_ab_ab_logb_a_l242_242076


namespace total_digits_in_book_l242_242628

theorem total_digits_in_book : 
  let pages1 := 9
  let pages2 := 99 - 10 + 1
  let pages3 := 266 - 100 + 1
  let total := (pages1 * 1) + (pages2 * 2) + (pages3 * 3)
  in total = 690 :=
by
  let pages1 := 9
  let pages2 := 99 - 10 + 1
  let pages3 := 266 - 100 + 1
  let total := (pages1 * 1) + (pages2 * 2) + (pages3 * 3)
  have h1 : pages1 = 9 := rfl
  have h2 : pages2 = 90 := rfl
  have h3 : pages3 = 167 := rfl
  have h_total : total = 690 := by
    calc total
       = (pages1 * 1) + (pages2 * 2) + (pages3 * 3) : rfl
     ... = 9 * 1 + 90 * 2 + 167 * 3 : by rw [h1, h2, h3]
     ... = 9 + 180 + 501 : rfl
     ... = 690 : rfl
  exact h_total

end total_digits_in_book_l242_242628


namespace chord_length_l242_242697

noncomputable def parabola : set (ℝ × ℝ) := {p | p.2 ^ 2 = 4 * p.1}
def line_through_point_with_slope_angle (p : ℝ × ℝ) (θ : ℝ) : set (ℝ × ℝ) := 
  {q | q.2 = tan(θ) * (q.1 - p.1)}

theorem chord_length
  (p : ℝ × ℝ) (θ : ℝ) (h : p = (1, 0))
  (parabola_eq : parabola = {z | z.snd ^ 2 = 4 * z.fst})
  (line_theta_eq : θ = (3 * π) / 4) :
  ∀ A B ∈ (parabola ∩ line_through_point_with_slope_angle p θ), dist A B = 8 :=
by
  intro A B hA hB
  sorry

end chord_length_l242_242697


namespace exists_irrational_r_l242_242598

theorem exists_irrational_r (k : ℕ) (hk : 2 ≤ k) :
  ∃ (r : ℝ), irrational r ∧ ∀ (m : ℕ), (⌊r^m⌋ : ℤ) ≡ -1 [ZMOD k] :=
sorry

end exists_irrational_r_l242_242598


namespace log_three_nine_cubed_l242_242951

theorem log_three_nine_cubed : ∀ (log : ℕ → ℕ → ℕ), log 3 9 = 2 → log 3 3 = 1 → (∀ (a b n : ℕ), log a (b ^ n) = n * log a b) → log 3 (9 ^ 3) = 6 :=
by
  intros log h1 h2 h3
  sorry

end log_three_nine_cubed_l242_242951


namespace first_term_exceeds_10000_l242_242224

def sequence : ℕ → ℕ
| 0     := 2
| (n+1) := (finset.sum (finset.range n.succ) sequence) 

theorem first_term_exceeds_10000 :
  ∃ n, sequence n > 10000 ∧ sequence n = 16384 :=
sorry

end first_term_exceeds_10000_l242_242224


namespace apple_tree_width_l242_242194

theorem apple_tree_width :
  ∃ A : ℝ, (2 * A + 12) + (12 + 15 + 12) = 71 ∧ A = 10 :=
by
  use 10
  split
  calc
    (2 * 10 + 12) + (12 + 15 + 12) = (20 + 12) + (12 + 15 + 12) := by ring
    ... = 32 + 39 := by ring
    ... = 71 := by ring
  rfl

end apple_tree_width_l242_242194


namespace log_evaluation_l242_242764

theorem log_evaluation : ∀ (a : ℕ), a = 3 → log 3 (9^3) = 6 :=
by
  intro a ha
  have h1 : 9 = 3^2 := by sorry
  have h2 : (9^3) = (3^2)^3 := by rw [h1]
  have h3 : (3^2)^3 = 3^6 := by sorry
  have h4 : log 3 (3^6) = 6 := by sorry
  rw [h2, h3, h4]
  exact ha

end log_evaluation_l242_242764


namespace cos_squared_1_l242_242636

theorem cos_squared_1 (k : ℕ) (θ : ℝ) (h₁ : cos^2 (k^2 + 7^2) = 1) (h₂ : θ = (k^2 + 7^2) * π / 180) : (k = 13 ∨ k = 23) :=
sorry

end cos_squared_1_l242_242636


namespace calc_1_calc_2_calc_3_calc_4_l242_242363

-- Problem 1
theorem calc_1 : 26 - 7 + (-6) + 17 = 30 := 
by
  sorry

-- Problem 2
theorem calc_2 : -81 / (9 / 4) * (-4 / 9) / (-16) = -1 := 
by
  sorry

-- Problem 3
theorem calc_3 : ((2 / 3) - (3 / 4) + (1 / 6)) * (-36) = -3 := 
by
  sorry

-- Problem 4
theorem calc_4 : -1^4 + 12 / (-2)^2 + (1 / 4) * (-8) = 0 := 
by
  sorry


end calc_1_calc_2_calc_3_calc_4_l242_242363


namespace pineapple_rings_per_pineapple_l242_242157

def pineapples_purchased : Nat := 6
def cost_per_pineapple : Nat := 3
def rings_sold_per_set : Nat := 4
def price_per_set_of_4_rings : Nat := 5
def profit_made : Nat := 72

theorem pineapple_rings_per_pineapple : (90 / 5 * 4 / 6) = 12 := 
by 
  sorry

end pineapple_rings_per_pineapple_l242_242157


namespace correct_probability_of_three_of_a_kind_l242_242605

-- Definitions and conditions based on provided problem statement
def standard_dice := {n : ℕ // n ∈ finset.range 1 7}

def initial_roll (d1 d2 d3 d4 d5 d6 : standard_dice) :=
  ¬ (∃ n, finset.card (finset.filter (λ x, x = n) {d1, d2, d3, d4, d5, d6}) ≥ 3)
  ∧ ∃ n1 n2,
      n1 ≠ n2 ∧
      finset.card (finset.filter (λ x, x = n1) {d1, d2, d3, d4, d5, d6}) = 2 ∧
      finset.card (finset.filter (λ x, x = n2) {d1, d2, d3, d4, d5, d6}) = 2

-- Re-rolling two dice and analyzing outcomes
def re_rolled_dice (r1 r2 : standard_dice) (set_aside_1 set_aside_2 set_aside_3 set_aside_4 : standard_dice) :=
  finset.card (finset.filter (λ x, x = r1) {r1, r2, set_aside_1, set_aside_2, set_aside_3, set_aside_4}) ≥ 3 ∨
  finset.card (finset.filter (λ x, x = r2) {r1, r2, set_aside_1, set_aside_2, set_aside_3, set_aside_4}) ≥ 3

-- Probability calculation
noncomputable def probability_at_least_three_of_a_kind :=
  34 / 36 : ℚ

theorem correct_probability_of_three_of_a_kind
  (d1 d2 d3 d4 d5 d6 : standard_dice)
  (initial_condition: initial_roll d1 d2 d3 d4 d5 d6) :
  ∃ r1 r2, re_rolled_dice r1 r2 d1 d2 d3 d4 ∨
           re_rolled_dice r1 r2 d1 d2 d5 d6 ∨
           re_rolled_dice r1 r2 d3 d4 d5 d6 →
  probability_at_least_three_of_a_kind = 17 / 18 :=
by
  sorry

end correct_probability_of_three_of_a_kind_l242_242605


namespace sum_of_integers_ending_in_1_or_7_l242_242360

theorem sum_of_integers_ending_in_1_or_7 (a b : ℕ) (f : ℕ → ℕ) (g : ℕ → ℕ) (n m : ℕ) :
  a = 51 → b = 57 → f 0 = 51 → f n = 51 + 10 * n → g 0 = 57 → g m = 57 + 10 * m →
  (50 ≤ f n ∧ f n ≤ 450) → (50 ≤ g m ∧ g m ≤ 450) →
  (n = 39) → (m = 39) →
  (Finset.range (n + 1)).sum (λ i, f i) + (Finset.range (m + 1)).sum (λ i, g i) = 19920 := 
by
  intros
  sorry

end sum_of_integers_ending_in_1_or_7_l242_242360


namespace evaluate_log_l242_242785

noncomputable def log_base (b x : ℝ) : ℝ := real.log x / real.log b

theorem evaluate_log : log_base 3 (9^3) = 6 := by
  have h1 : 9 = 3^2 := by norm_num
  have h2 : 9^3 = (3^2)^3 := by norm_num
  rw [h2, pow_mul]
  have h3 : log_base 3 (3^6) = 6 := by
    rw [← real.log_pow, real.log_mul, mul_comm, ← real.log_rpow, mul_div, real.log_self]
    norm_num
  exact h3

end evaluate_log_l242_242785


namespace rhombus_area_l242_242260

-- Given a rhombus with sides of 4 cm and an included angle of 45 degrees,
-- prove that the area is 8 square centimeters.

theorem rhombus_area :
  ∀ (s : ℝ) (α : ℝ), s = 4 ∧ α = π / 4 → 
    let area := s * s * Real.sin α in
    area = 8 := 
by
  intros s α h
  sorry

end rhombus_area_l242_242260


namespace log_base_3_of_9_cubed_l242_242778

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 :=
by
  -- Given conditions:
  -- 1. 9 = 3^2
  -- 2. log_b (a^n) = n * log_b a
  sorry

end log_base_3_of_9_cubed_l242_242778


namespace total_cost_for_james_l242_242530

-- Prove that James will pay a total of $250 for his new pair of glasses.

theorem total_cost_for_james
  (frame_cost : ℕ := 200)
  (lens_cost : ℕ := 500)
  (insurance_cover_percentage : ℚ := 0.80)
  (coupon_on_frames : ℕ := 50) :
  (frame_cost - coupon_on_frames + lens_cost * (1 - insurance_cover_percentage)) = 250 :=
by
  -- Declare variables for the described values
  let total_frame_cost := frame_cost - coupon_on_frames
  let insurance_cover := lens_cost * insurance_cover_percentage
  let total_lens_cost := lens_cost - insurance_cover
  let total_cost := total_frame_cost + total_lens_cost

  -- We need to show total_cost = 250
  have h1 : total_frame_cost = 150 := by sorry
  have h2 : insurance_cover = 400 := by sorry
  have h3 : total_lens_cost = 100 := by sorry
  have h4 : total_cost = 250 := by
    rw [←h1, ←h3]
    sorry

  exact h4

end total_cost_for_james_l242_242530


namespace min_n_for_binomial_constant_term_l242_242119

theorem min_n_for_binomial_constant_term:
  (∃ (n : ℕ), (∃ (k : ℕ), 3 * n = 5 * k) ∧ (k + 1 ≤ n) ∧ 
    ((λ (T : ℕ → ℝ), T k + 1 = (nat.choose n k) * (3^k) * (0)) / ((λ x, T x) (k) = (λ (k : ℕ), 3^k * nat.choose n k * x ^ (3 * n - 5 * k / 6) ) ) ∧ 
    (∃ (k : ℕ), 3 * n = 5 * k)) ⇔ n = 5) 
  : sorry

end min_n_for_binomial_constant_term_l242_242119


namespace log_pow_evaluation_l242_242913

-- Define the condition that 9 is equal to 3^2
def nine_eq_three_squared : Prop := 9 = 3^2

-- The main theorem that we need to prove
theorem log_pow_evaluation (h : nine_eq_three_squared) : log 3 (9^3) = 6 :=
  sorry

end log_pow_evaluation_l242_242913


namespace angle_between_vectors_l242_242005

variable {V : Type*} [InnerProductSpace ℝ V]
variable (a b : V)

theorem angle_between_vectors (ha : ‖a‖ = 1) (hb : ‖b‖ = √2) (h_perp : (inner a (a - b)) = 0) :
  real.angle a b = real.pi / 4 :=
sorry

end angle_between_vectors_l242_242005


namespace question_equals_answer_l242_242498

theorem question_equals_answer (x y : ℝ) (h : abs (x - 6) + (y + 4)^2 = 0) : x + y = 2 :=
sorry

end question_equals_answer_l242_242498


namespace p_and_q_work_together_l242_242667

-- Given conditions
variable (Wp Wq : ℝ)

-- Condition that p is 50% more efficient than q
def efficiency_relation : Prop := Wp = 1.5 * Wq

-- Condition that p can complete the work in 25 days
def work_completion_by_p : Prop := Wp = 1 / 25

-- To be proved that p and q working together can complete the work in 15 days
theorem p_and_q_work_together (h1 : efficiency_relation Wp Wq)
                              (h2 : work_completion_by_p Wp) :
                              1 / (Wp + (Wp / 1.5)) = 15 :=
by
  sorry

end p_and_q_work_together_l242_242667


namespace log_evaluation_l242_242766

theorem log_evaluation : ∀ (a : ℕ), a = 3 → log 3 (9^3) = 6 :=
by
  intro a ha
  have h1 : 9 = 3^2 := by sorry
  have h2 : (9^3) = (3^2)^3 := by rw [h1]
  have h3 : (3^2)^3 = 3^6 := by sorry
  have h4 : log 3 (3^6) = 6 := by sorry
  rw [h2, h3, h4]
  exact ha

end log_evaluation_l242_242766


namespace simplify_expression_l242_242202

theorem simplify_expression :
    (1 / (Real.logBase 18 3 + 1)) + (1 / (Real.logBase 12 2 + 1)) + (1 / (Real.logBase 8 7 + 1)) = 13 / 12 := 
by
  sorry

end simplify_expression_l242_242202


namespace students_taking_all_three_l242_242246

-- Definitions and Conditions
def total_students : ℕ := 25
def coding_students : ℕ := 12
def chess_students : ℕ := 15
def photography_students : ℕ := 10
def at_least_two_classes : ℕ := 10

-- Request to prove: Number of students taking all three classes
theorem students_taking_all_three (x y w z : ℕ) :
  (x + y + z + w = 10) →
  (coding_students - (10 - y) + chess_students - (10 - w) + (10 - x) = 21) →
  z = 4 :=
by
  intros
  -- Proof will go here
  sorry

end students_taking_all_three_l242_242246


namespace angle_opposite_shared_vertex_l242_242716

-- Define the square and right triangle sharing a common vertex and inscribed in a circle.
variables {A B C D E F : Type} [has_angle_deg A] [has_angle_deg B] [has_angle_deg C] [has_angle_deg D] [has_angle_deg E] [has_angle_deg F]

-- Define the points and properties
variable (V : Type) -- Common vertex
variable (X Y : Type) -- Other vertices of the triangle
variable (square_side : Type) -- One side of the square which is also the hypotenuse of the triangle

-- Given conditions
variables [right_angle (angle_deg D V square_side)] -- Each angle in the square is 90 degrees
variables [isosceles_right_triangle V X Y] -- Triangle VXY is isosceles and right
variables [shared_vertex V] -- V is the common vertex

-- Prove
theorem angle_opposite_shared_vertex :
  angle_deg (angle X V Y) = 90 :=
sorry

end angle_opposite_shared_vertex_l242_242716


namespace right_triangle_area_l242_242594

noncomputable def num_possible_locations (P Q : EuclideanSpace ℝ (Fin 2)) (hPQ : dist P Q = 10) : Nat :=
  8

theorem right_triangle_area (P Q R : EuclideanSpace ℝ (Fin 2)) 
  (hPQ : dist P Q = 10)
  (hArea : euclidean_dist P Q * height R = 32) : 
  num_possible_locations P Q hPQ = 8 :=
by
  sorry

end right_triangle_area_l242_242594


namespace log_three_pow_nine_pow_three_eq_six_l242_242938

open Real

noncomputable def eval_log_three_pow_nine_pow_three : ℝ :=
  log 3 (9^3)

theorem log_three_pow_nine_pow_three_eq_six :
  eval_log_three_pow_nine_pow_three = 6 := by
  sorry

end log_three_pow_nine_pow_three_eq_six_l242_242938


namespace problem_statement_l242_242740

-- Define the function g with known values
def g : ℝ → ℝ
| -1 => 3
| 0 => 1
| 1 => 4
| 2 => 5
| 3 => 6
| _ => sorry  -- g defined only on the given points

-- Given that g is invertible
noncomputable def g_inv : ℝ → ℝ
| 3 => -1
| 1 => 0
| 4 => 1
| 5 => 2
| 6 => 3
| _ => sorry  -- g_inv defined only on known inverse points

theorem problem_statement : g (g 1) + g (g_inv 2) + g_inv (g_inv 6) = 8 :=
by
  -- Proof omitted
  sorry

end problem_statement_l242_242740


namespace cubic_sum_l242_242107

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 :=
by {
  sorry
}

end cubic_sum_l242_242107


namespace speed_of_current_l242_242328

theorem speed_of_current(
  (rowing_speed_still_water : ℝ), 
  (downstream_distance : ℝ), 
  (downstream_time : ℝ)) 
  (h_rowing_speed : rowing_speed_still_water = 6) 
  (h_downstream_distance : downstream_distance = 80) 
  (h_downstream_time : downstream_time = 31.99744020478362) : 
  (speed_of_current : ℝ) 
  (h_speed_of_current : speed_of_current = 3) := 
sorry

end speed_of_current_l242_242328


namespace george_blocks_count_l242_242988

theorem george_blocks_count :
  (boxes : ℕ) (blocks_per_box : ℕ) (total_blocks : ℕ) :=
  (boxes = 2) →
  (blocks_per_box = 6) →
  total_blocks = (boxes * blocks_per_box) →
  total_blocks = 12 :=
by
  sorry

end george_blocks_count_l242_242988


namespace log_base_3_of_9_cubed_l242_242774

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 :=
by
  -- Given conditions:
  -- 1. 9 = 3^2
  -- 2. log_b (a^n) = n * log_b a
  sorry

end log_base_3_of_9_cubed_l242_242774


namespace max_bishops_with_two_mountains_is_19_l242_242649

-- Define the problem conditions
def chessboard : Type := Array (Array (Option Bool)) -- Bool indicates if there's a mountain, Option None indicates an empty square

def is_bishop_move_valid (board : chessboard) (x1 y1 x2 y2 : Nat) : Bool :=
  let valid_move := (x1 ≠ x2 ∧ y1 ≠ y2) ∧ ((x2 - x1).abs = (y2 - y1).abs)
  valid_move ∧ (board[x2]![y2].isNone) ∧ (board[x1]![y1].isNone) -- Cannot move through mountains or other bishops

def max_non_attacking_bishops (board : chessboard) : Nat := 
  -- Calculate based on the given solution approach of regions
  -- We'll just define this as a constant based on the solution steps
  19

-- Theorem to prove
theorem max_bishops_with_two_mountains_is_19 (board : chessboard) (condition : ∀ x y, board[x]![y] ≠ some true) :
  max_non_attacking_bishops board = 19 :=
  sorry

end max_bishops_with_two_mountains_is_19_l242_242649


namespace pos_int_satisfy_inequality_l242_242239

open Nat

def C (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))
def A (n k : ℕ) : ℕ := factorial n / factorial (n - k)

theorem pos_int_satisfy_inequality :
  {n : ℕ // 0 < n ∧ 2 * C n 3 ≤ A n 2} = {n // n = 3 ∨ n = 4 ∨ n = 5} :=
by
  sorry

end pos_int_satisfy_inequality_l242_242239


namespace triangle_is_isosceles_right_l242_242128

theorem triangle_is_isosceles_right (A B C a b c : ℝ) 
  (h : a / (Real.cos A) = b / (Real.cos B) ∧ b / (Real.cos B) = c / (Real.sin C)) :
  A = π/4 ∧ B = π/4 ∧ C = π/2 := 
sorry

end triangle_is_isosceles_right_l242_242128


namespace train_crossing_time_l242_242717

def train_length : ℕ := 1000
def train_speed_km_per_h : ℕ := 18
def train_speed_m_per_s := train_speed_km_per_h * 1000 / 3600

theorem train_crossing_time :
  train_length / train_speed_m_per_s = 200 := by
sorry

end train_crossing_time_l242_242717


namespace stratified_sampling_players_l242_242661

theorem stratified_sampling_players 
  (total_players_class5 : ℕ) (total_players_class16 : ℕ) (total_players_class33 : ℕ) 
  (selected_players : ℕ) (sampling_ratio : ℚ)
  (players_class5_sampled : ℕ) (players_class16_sampled : ℕ) :
  total_players_class5 = 6 → total_players_class16 = 8 → total_players_class33 = 10 → 
  selected_players = 12 → sampling_ratio = (selected_players : ℚ) / (total_players_class5 + total_players_class16 + total_players_class33) → 
  players_class5_sampled = (total_players_class5 : ℚ * sampling_ratio).toNat → 
  players_class16_sampled = (total_players_class16 : ℚ * sampling_ratio).toNat → 
  players_class5_sampled = 3 ∧ players_class16_sampled = 4 :=
by 
  intros h_class5 h_class16 h_class33 h_selected_players h_sampling_ratio h_sampled_class5 h_sampled_class16
  sorry

end stratified_sampling_players_l242_242661


namespace log_base_3_of_9_cubed_l242_242823

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 := by
  sorry

end log_base_3_of_9_cubed_l242_242823


namespace not_or_false_imp_and_false_l242_242495

variable (p q : Prop)

theorem not_or_false_imp_and_false (h : ¬ (p ∨ q) = False) : ¬ (p ∧ q) :=
by
  sorry

end not_or_false_imp_and_false_l242_242495


namespace log_evaluation_l242_242761

theorem log_evaluation : ∀ (a : ℕ), a = 3 → log 3 (9^3) = 6 :=
by
  intro a ha
  have h1 : 9 = 3^2 := by sorry
  have h2 : (9^3) = (3^2)^3 := by rw [h1]
  have h3 : (3^2)^3 = 3^6 := by sorry
  have h4 : log 3 (3^6) = 6 := by sorry
  rw [h2, h3, h4]
  exact ha

end log_evaluation_l242_242761


namespace trisection_eccentricity_unique_l242_242014

-- Definitions and assumptions leveraging given problem.
def ellipse_equation (x y a b : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

def major_axis (a b : ℝ) : Prop :=
  a > b ∧ b > 0

def foci_condition (a b c e : ℝ) : Prop :=
  c = a * e ∧ e ∈ (Set.Ioo 0 1)

def trisection_condition (a b c : ℝ) : Prop :=
  ∃ P : ℝ × ℝ, 
  (ellipse_equation P.1 P.2 a b) ∧
  ((∠P (0, b) A = ∠P F1 F2) ∧ (∠P F1 F2 = ∠P A B / 3))

noncomputable
def distinct_eccentricity_count : ℝ → ℝ → ℝ :=
sorry

theorem trisection_eccentricity_unique (a b : ℝ) (h : major_axis a b) :
  ∃ e : ℝ, 
  (foci_condition a b (a * e) e) ∧ 
  (trisection_condition a b (a * e)) ∧ 
  (distinct_eccentricity_count a b = 1) :=
sorry

end trisection_eccentricity_unique_l242_242014


namespace extreme_values_monotonic_decreasing_interval_l242_242048

noncomputable def f (x : ℝ) := 4 * Real.log x + (1 / 2) * x^2 - 5 * x

theorem extreme_values :
  (∀ x, Real.log x > -1) →
  (Sup (f '' (Set.Icc 1 4)) = f 1 ∧ f 1 = -9 / 2 ∧ Inf (f '' (Set.Icc 1 4)) = f 4 ∧ f 4 = 8 * Real.log 2 - 12) :=
by
  intros
  sorry

theorem monotonic_decreasing_interval (m : ℝ) :
  (∀ x, Real.log x > -1) →
  (∀ x, 2 * m <= x → x <= m + 1 → (f x)' < 0) →
  (1 / 2 ≤ m ∧ m < 1) :=
by
  intros
  sorry

end extreme_values_monotonic_decreasing_interval_l242_242048


namespace ab_neither_sufficient_nor_necessary_l242_242546

theorem ab_neither_sufficient_nor_necessary (a b : ℝ) (h : a * b ≠ 0) :
  (¬ ((a * b > 1) → (a > 1 / b))) ∧ (¬ ((a > 1 / b) → (a * b > 1))) :=
by
  sorry

end ab_neither_sufficient_nor_necessary_l242_242546


namespace log_base_3_of_9_cubed_l242_242866

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 := by
  -- Assuming the necessary properties of logarithms and exponents
  sorry

end log_base_3_of_9_cubed_l242_242866


namespace main_l242_242435

section
variables (a b c : ℕ) (ast : ℕ → ℕ → ℕ)

-- Conditions
axiom cond1 : ∀ a b c : ℕ, (ast a b) (ast c) = ast a (b * c) 
axiom cond2 : ∀ a b c : ℕ, (ast a b) * (ast a c) = ast a (b + c) 

-- Main Goal
theorem main : ast 2 4 = 16 := 
sorry
end

end main_l242_242435


namespace volume_of_inscribed_sphere_is_2304_pi_l242_242710

-- Formulate the problem's conditions formally.
structure RightCircularCone :=
(base_diameter : ℝ)
(height : ℝ)
(vertex_angle : ℝ := 90)

structure Sphere :=
(radius : ℝ)

-- Given data
def cone : RightCircularCone :=
{ base_diameter := 24,
  height := 12 * Real.sqrt 2 }

-- The sphere is inscribed in the cone
def inscribed_sphere (c : RightCircularCone) : Sphere :=
{ radius := c.base_diameter / 2 }

-- Prove that the volume of the inscribed sphere is 2304π cubic inches
theorem volume_of_inscribed_sphere_is_2304_pi :
  let r := (inscribed_sphere cone).radius in
  (4 / 3) * Real.pi * r^3 = 2304 * Real.pi :=
by
  sorry

end volume_of_inscribed_sphere_is_2304_pi_l242_242710


namespace symmetric_prob_15_55_l242_242218

-- Definition of summing the top faces of 10 dice to 15
def is_sum_15 (outcomes : Fin 10 → Fin 6) : Prop :=
  (∑ i, (outcomes i).val + 1) = 15

-- Definition of summing the top faces of 10 dice to 55
def is_sum_55 (outcomes : Fin 10 → Fin 6) : Prop :=
  (∑ i, (outcomes i).val + 1) = 55

-- Statement expressing the symmetry of the sums around 35
theorem symmetric_prob_15_55 :
  (∑ i, (outcomes i).val + 1) = 15 ↔ (∑ i, (outcomes i).val + 1) = 55 :=
by
  sorry

end symmetric_prob_15_55_l242_242218


namespace log_base_three_of_nine_cubed_l242_242884

theorem log_base_three_of_nine_cubed : log 3 (9^3) = 6 :=
by
  have h1 : 9 = 3^2 := by sorry
  have h2 : ∀ (a m n: ℕ), (a^m)^n = a^(m * n) := by 
    intros a m n
    exact pow_mul a m n
  have h3 : ∀ (b n: ℕ), log b (b^n) = n := by sorry
  sorry

end log_base_three_of_nine_cubed_l242_242884


namespace a_dot_b_norm_a_norm_b_angle_between_a_b_l242_242207

variable (e1 e2 : ℝ^3) -- ℝ^3 to represent unit vectors context
variable (a b : ℝ^3) 

-- Conditions
axiom e1_unit : ‖e1‖ = 1
axiom e2_unit : ‖e2‖ = 1
axiom angle_60 : e1 ⬝ e2 = 1/2

-- Definitions of a and b
def a : ℝ^3 := 2 • e1 + e2
def b : ℝ^3 := -3 • e1 + 2 • e2

-- Problem statements
theorem a_dot_b : a ⬝ b = -7 / 2 := sorry

theorem norm_a : ‖a‖ = Real.sqrt 7 := sorry
theorem norm_b : ‖b‖ = Real.sqrt 7 := sorry

theorem angle_between_a_b : Real.arccos ((a ⬝ b) / (‖a‖ * ‖b‖)) = 2 * Real.pi / 3 := sorry

end a_dot_b_norm_a_norm_b_angle_between_a_b_l242_242207


namespace min_value_of_squares_find_p_l242_242003

open Real

theorem min_value_of_squares (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (eqn : a + sqrt 2 * b + sqrt 3 * c = 2 * sqrt 3) :
  a^2 + b^2 + c^2 = 2 :=
by sorry

theorem find_p (m : ℝ) (hm : m = 2) (p q : ℝ) :
  (∀ x, |x - 3| ≥ m ↔ x^2 + p * x + q ≥ 0) → p = -6 :=
by sorry

end min_value_of_squares_find_p_l242_242003


namespace smallest_positive_period_of_f_l242_242978

def smallest_positive_period_function : Real := 
  π

def f (x : ℝ) : ℝ := 
  Real.cos (2 * x + π / 6)

theorem smallest_positive_period_of_f :
  ∃ T > 0, ∀ x : ℝ, f (x + T) = f x ∧ T = smallest_positive_period_function := 
by
  sorry

end smallest_positive_period_of_f_l242_242978


namespace sector_area_l242_242712

-- Given the radius r of the circle and the arc length s of the sector,
-- prove that the area of the sector equals 30.
theorem sector_area (r s : ℝ) (h1 : r = 6) (h2 : s = 10) : 
  (s / (2 * real.pi * r)) * (real.pi * r^2) = 30 := by
  sorry

end sector_area_l242_242712


namespace inequality_solution_l242_242606

theorem inequality_solution (x : ℝ) : 
    (2 ^ (log x / log 2) ^ 2 - 12 * x ^ (log x / log 0.5) < 3 - log (x ^ 2 - 6 * x + 9) / log (3 - x)) ↔ 
    (x ∈ Set.Ioo (2 ^ -Real.sqrt 2) 2 ∪ Set.Ioo 2 (2 ^ Real.sqrt 2)) :=
by
  sorry

end inequality_solution_l242_242606


namespace pos_int_satisfy_inequality_l242_242238

open Nat

def C (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))
def A (n k : ℕ) : ℕ := factorial n / factorial (n - k)

theorem pos_int_satisfy_inequality :
  {n : ℕ // 0 < n ∧ 2 * C n 3 ≤ A n 2} = {n // n = 3 ∨ n = 4 ∨ n = 5} :=
by
  sorry

end pos_int_satisfy_inequality_l242_242238


namespace derivative_at_one_is_four_l242_242215

-- Define the function y = x^2 + 2x + 1
def f (x : ℝ) := x^2 + 2*x + 1

-- State the theorem: The derivative of f at x = 1 is 4
theorem derivative_at_one_is_four : (deriv f 1) = 4 :=
by
  -- The proof is omitted here.
  sorry

end derivative_at_one_is_four_l242_242215


namespace log3_of_9_to_3_l242_242806

theorem log3_of_9_to_3 : Real.logb 3 (9^3) = 6 :=
by
  have h1 : 9 = 3^2 := by norm_num
  have h2 : 9^3 = (3^2)^3 := by rw [h1]
  have h3 : (3^2)^3 = 3^(2 * 3) := by rw [pow_mul]
  have h4 : Real.logb 3 (3^(2 * 3)) = 2 * 3 := Real.logb_pow 3 (2 * 3)
  rw [h4]
  norm_num

end log3_of_9_to_3_l242_242806


namespace four_digit_number_count_four_digit_number_with_exactly_two_identical_digits_count_l242_242647

-- Define the constraints for forming a four-digit number using specific digits
def valid_digits := {0, 1, 2, 3, 4, 5}
def is_valid_digit (d : ℕ) := d ∈ valid_digits
def is_nonzero_digit (d : ℕ) := d ∈ valid_digits ∧ d ≠ 0

-- Define what it means for a number to be four digits
def is_four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ (∀ d : ℕ, d ∈ (digits 10 n) → is_valid_digit d) ∧ is_nonzero_digit (digits 10 n).head!

-- Question I: Prove total number of valid four-digit numbers
theorem four_digit_number_count : ∃ n, n = 1080 ∧
  ∀ x : ℕ, is_four_digit_number x ↔ x ∈ (finset.range n) :=
sorry

-- Question II: Prove number of four-digit numbers with exactly two identical digits
theorem four_digit_number_with_exactly_two_identical_digits_count : ∃ n, n = 600 ∧
  ∀ x : ℕ, is_four_digit_number x ∧ (∃ d ∈ (digits 10 x), count (digits 10 x) d = 2) ↔ x ∈ (finset.range n) :=
sorry

end four_digit_number_count_four_digit_number_with_exactly_two_identical_digits_count_l242_242647


namespace percentage_of_millet_in_Brand_A_l242_242689

variable (A B : ℝ)
variable (B_percent : B = 0.65)
variable (mix_millet_percent : 0.60 * A + 0.40 * B = 0.50)

theorem percentage_of_millet_in_Brand_A :
  A = 0.40 :=
by
  sorry

end percentage_of_millet_in_Brand_A_l242_242689


namespace ab_sum_l242_242115

theorem ab_sum (a b : ℝ) (h₁ : ∀ x : ℝ, (x + a) * (x + 8) = x^2 + b * x + 24) (h₂ : 8 * a = 24) : a + b = 14 :=
by
  sorry

end ab_sum_l242_242115


namespace parabola_equation_l242_242983

theorem parabola_equation (x m : ℝ) (P : ℝ × ℝ) (hP : P = (1, m)) (h_axis_symmetry : ∀ t : ℝ, (t, 0) = t • (1, 0))
  (vertex_origin : (0, 0) = (0, 0))
  (distance_focus : ∀ a : ℝ, (1 - a)^2 + m^2 = 9 → a = 2) :
  ∃ a : ℝ, y^2 = 4 * a * x := begin
  use 2,
  sorry  -- This is where the proof would go.
end

end parabola_equation_l242_242983


namespace max_value_of_expression_l242_242544

variable {V : Type _} [InnerProductSpace ℝ V]
variables (p q r : V)

def norm_squared (x : V) : ℝ := inner x x

theorem max_value_of_expression (hp : ∥p∥ = 2) (hq : ∥q∥ = 3) (hr : ∥r∥ = 4) :
  ∥p - 3 • q∥^2 + ∥q - 3 • r∥^2 + ∥r - 3 • p∥^2 ≤ 377 :=
sorry

end max_value_of_expression_l242_242544


namespace a_plus_b_values_l242_242994

theorem a_plus_b_values (a b : ℝ) (h1 : abs a = 5) (h2 : abs b = 3) (h3 : abs (a - b) = b - a) : a + b = -2 ∨ a + b = -8 :=
sorry

end a_plus_b_values_l242_242994


namespace sum_even_integers_200_to_400_l242_242280

theorem sum_even_integers_200_to_400 : 
  let seq := list.range' 200 ((400 - 200) / 2 + 1)
  in seq.filter (λ n, n % 2 = 0) = list.range' 200 101 ∧ 
     seq.sum = 30300 := 
by
  sorry

end sum_even_integers_200_to_400_l242_242280


namespace find_length_PB_l242_242540

variable (A B C D P : Type)
variable (PA PD AC PB : ℝ)

def is_point_in_rectangle (A B C D P : Type) : Prop := sorry
def diagonal_is_correct (AC : ℝ) : Prop := AC = 8
def PA_is_correct (PA : ℝ) : Prop := PA = 5
def PD_is_correct (PD : ℝ) : Prop := PD = 3
def PB_is_correct (PB : ℝ) : Prop := PB = sqrt 39

theorem find_length_PB (h1 : is_point_in_rectangle A B C D P)
  (h2 : diagonal_is_correct AC) (h3 : PA_is_correct PA)
  (h4 : PD_is_correct PD) : PB_is_correct PB :=
sorry

end find_length_PB_l242_242540


namespace product_of_common_ratios_l242_242550

theorem product_of_common_ratios (x p r a2 a3 b2 b3 : ℝ)
  (h1 : a2 = x * p) (h2 : a3 = x * p^2)
  (h3 : b2 = x * r) (h4 : b3 = x * r^2)
  (h5 : 3 * a3 - 4 * b3 = 5 * (3 * a2 - 4 * b2))
  (h_nonconstant : x ≠ 0) (h_diff_ratios : p ≠ r) :
  p * r = 9 :=
by
  sorry

end product_of_common_ratios_l242_242550


namespace log_base_3_of_9_cubed_l242_242777

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 :=
by
  -- Given conditions:
  -- 1. 9 = 3^2
  -- 2. log_b (a^n) = n * log_b a
  sorry

end log_base_3_of_9_cubed_l242_242777


namespace cost_of_each_ball_number_of_plans_highest_profit_l242_242320

-- Part 1: Cost of each ball
theorem cost_of_each_ball (x y : ℝ) 
  (h1 : 10 * x + 5 * y = 100) 
  (h2 : 5 * x + 3 * y = 55) : 
  x = 5 ∧ y = 10 :=
sorry

-- Part 2: Number of purchasing plans
theorem number_of_plans 
  (x y m : ℝ) 
  (h_costA : x = 5) 
  (h_costB : y = 10)
  (h_budget : 1000 = x * (200 - 2 * m) + y * m) 
  (h_quantity_A : 200 - 2 * m ≥ 6 * m) 
  (h_minimum_m : 23 ≤ m) :
  ∃ l, l = {23, 24, 25}.card ∧ (∀ k ∈ l, 23 ≤ k ∧ k ≤ 25) ∧ set.card l = 3 :=
sorry

-- Part 3: Maximum profit
theorem highest_profit 
  (m : ℝ)
  (h_costB : 10 = 10)
  (h_plans : m ∈ {23, 24, 25})
  (h_max_profit : ∀ m₁ ∈ {23, 24, 25}, 
    let profit := 3 * (200 - 2 * m) + 4 * m in 
    profit ≤ (3 * (200 - 2 * 23) + 4 * 23) :=
  ∀ m₁ ∈ {23, 24, 25}, 
    3 * (200 - 2 * m) + 4 * m ≤ 554 :=
sorry

end cost_of_each_ball_number_of_plans_highest_profit_l242_242320


namespace gcd_factorial_8_10_l242_242274

theorem gcd_factorial_8_10 (n : ℕ) (hn : n = 10! - 8!): gcd 8! 10! = 8! := by
  sorry

end gcd_factorial_8_10_l242_242274


namespace find_a_l242_242387

def operation (a b : ℤ) : ℤ := 2 * a - b * b

theorem find_a (a : ℤ) : operation a 3 = 15 → a = 12 := by
  sorry

end find_a_l242_242387


namespace probability_not_face_card_l242_242319

-- Definitions based on the conditions
def total_cards : ℕ := 52
def face_cards  : ℕ := 12
def non_face_cards : ℕ := total_cards - face_cards

-- Statement of the theorem
theorem probability_not_face_card : (non_face_cards : ℚ) / (total_cards : ℚ) = 10 / 13 := by
  sorry

end probability_not_face_card_l242_242319


namespace distance_AO_eq_3_symmetric_point_A_eq_negA_l242_242515

-- Define the points A and O
def A : ℝ × ℝ × ℝ := (2, 1, 2)
def O : ℝ × ℝ × ℝ := (0, 0, 0)

-- Define the distance function between two points in 3D space
def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2)

-- Define the symmetric point function
def symmetric_point (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (-p.1, -p.2, -p.3)

-- Statement to prove the distance from A to O is 3
theorem distance_AO_eq_3 : distance A O = 3 := sorry

-- Statement to prove the symmetric point of A is (-2, -1, -2)
theorem symmetric_point_A_eq_negA : symmetric_point A = (-2, -1, -2) := sorry

end distance_AO_eq_3_symmetric_point_A_eq_negA_l242_242515


namespace log_three_pow_nine_pow_three_eq_six_l242_242941

open Real

noncomputable def eval_log_three_pow_nine_pow_three : ℝ :=
  log 3 (9^3)

theorem log_three_pow_nine_pow_three_eq_six :
  eval_log_three_pow_nine_pow_three = 6 := by
  sorry

end log_three_pow_nine_pow_three_eq_six_l242_242941


namespace equation_of_circle_through_points_l242_242974

def point := (ℤ × ℤ)
def circle_eq (d e f : ℤ) := Π (x y : ℤ), x^2 + y^2 + d * x + e * y + f = 0

theorem equation_of_circle_through_points (A B O : point)
  (hA : A = (-1, 1))
  (hB : B = (1, 1))
  (hO : O = (0, 0)) :
  ∃ d e f : ℤ, d = 0 ∧ e = -2 ∧ f = 0 ∧ circle_eq d e f A.1 A.2 ∧ circle_eq d e f B.1 B.2 ∧ circle_eq d e f O.1 O.2 := by
  sorry

end equation_of_circle_through_points_l242_242974


namespace monotonic_interval_l242_242006

open Real

noncomputable def f (a : ℝ) : ℝ → ℝ := λ x, exp (abs (x - a))

theorem monotonic_interval (a m : ℝ)
  (h1 : ∀ x, f a (1 + x) = f a (-x))
  (h2 : monotonic_on (f a) (set.Icc m (m + 1))) :
  m ∈ set.Iic (-1/2) ∪ set.Ici (1/2) :=
begin
  sorry
end

end monotonic_interval_l242_242006


namespace log_base_three_of_nine_cubed_l242_242888

theorem log_base_three_of_nine_cubed : log 3 (9^3) = 6 :=
by
  have h1 : 9 = 3^2 := by sorry
  have h2 : ∀ (a m n: ℕ), (a^m)^n = a^(m * n) := by 
    intros a m n
    exact pow_mul a m n
  have h3 : ∀ (b n: ℕ), log b (b^n) = n := by sorry
  sorry

end log_base_three_of_nine_cubed_l242_242888


namespace log_base_3_of_9_cubed_l242_242862

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 := by
  -- Assuming the necessary properties of logarithms and exponents
  sorry

end log_base_3_of_9_cubed_l242_242862


namespace log_base_3_of_9_cubed_l242_242827

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 := by
  sorry

end log_base_3_of_9_cubed_l242_242827


namespace log_base_3_of_9_cubed_l242_242837

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 := by
  -- Define the known conditions
  have h1 : 9 = 3^2 := by norm_num
  have h2 : log 3 (a^n) = n * log 3 a := sorry  -- Statement of the logarithmic identity
  
  -- Apply the conditions to prove the result
  calc
    log 3 (9^3)
        = log 3 ((3^2)^3) : by rw h1
    ... = log 3 (3^(2 * 3)) : by congr; norm_num
    ... = log 3 (3^6) : by norm_num
    ... = 6 : by rw [h2]; norm_num

end log_base_3_of_9_cubed_l242_242837


namespace tokyo_hurricane_damage_l242_242694

theorem tokyo_hurricane_damage (damage_in_yen : ℝ) (exchange_rate : ℝ) (damage_in_usd : ℝ) 
  (h_damage_in_yen : damage_in_yen = 5_000_000_000) (h_exchange_rate : exchange_rate = 110) 
  : damage_in_usd = 45_454_545 :=
by
  sorry

end tokyo_hurricane_damage_l242_242694


namespace students_no_more_than_397_l242_242632

theorem students_no_more_than_397 (initial_students : ℕ) (rounds : ℕ) : 
  initial_students = 2010 ∧ rounds = 5 → 
  (let after_first_round := initial_students - (initial_students / 3),
       after_second_round := after_first_round - (after_first_round / 3),
       after_third_round := after_second_round - (after_second_round / 3),
       after_fourth_round := after_third_round - (after_third_round / 3),
       after_fifth_round := after_fourth_round - (after_fourth_round / 3)
   in after_fifth_round ≤ 397) :=
by
  sorry

end students_no_more_than_397_l242_242632


namespace nalani_net_amount_l242_242586

-- Definitions based on the conditions
def luna_birth := 10 -- Luna gave birth to 10 puppies
def stella_birth := 14 -- Stella gave birth to 14 puppies
def luna_sold := 8 -- Nalani sold 8 puppies from Luna's litter
def stella_sold := 10 -- Nalani sold 10 puppies from Stella's litter
def luna_price := 200 -- Price per puppy for Luna's litter is $200
def stella_price := 250 -- Price per puppy for Stella's litter is $250
def luna_cost := 80 -- Cost of raising each puppy from Luna's litter is $80
def stella_cost := 90 -- Cost of raising each puppy from Stella's litter is $90

-- Theorem stating the net amount received by Nalani
theorem nalani_net_amount : 
        luna_sold * luna_price + stella_sold * stella_price - 
        (luna_birth * luna_cost + stella_birth * stella_cost) = 2040 :=
by 
  sorry

end nalani_net_amount_l242_242586


namespace cube_identity_l242_242112

theorem cube_identity (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 := by
  sorry

end cube_identity_l242_242112


namespace log_base_3_of_9_cubed_l242_242925
-- Import the necessary parts of the standard library

-- Define the statement to be proved
theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 :=
by
  sorry

end log_base_3_of_9_cubed_l242_242925


namespace sum_even_200_to_400_l242_242295

theorem sum_even_200_to_400 : 
  let t₁ := 202
  let tₙ := 398
  let n := 99
  let sum := (t₁ + tₙ) * n / 2 in
  sum = 29700 := 
by
  let t₁ := 202
  let tₙ := 398
  let n := 99
  let sum := (t₁ + tₙ) * n / 2
  show sum = 29700
  sorry

end sum_even_200_to_400_l242_242295


namespace log_three_pow_nine_pow_three_eq_six_l242_242934

open Real

noncomputable def eval_log_three_pow_nine_pow_three : ℝ :=
  log 3 (9^3)

theorem log_three_pow_nine_pow_three_eq_six :
  eval_log_three_pow_nine_pow_three = 6 := by
  sorry

end log_three_pow_nine_pow_three_eq_six_l242_242934


namespace domain_of_f_l242_242216

noncomputable def f (x : ℝ) := 1 / (Real.sqrt (1 - Real.log x / Real.log 3))

theorem domain_of_f :
  { x : ℝ | x > 0 ∧ x < 3 } = { x : ℝ | ∃ y, y ∈ set.Ioo 0 3 ∧ x = y } :=
by
  simp [set.ext_iff, Ioo]
  sorry

end domain_of_f_l242_242216


namespace cubic_sum_identity_l242_242094

theorem cubic_sum_identity (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 :=
sorry

end cubic_sum_identity_l242_242094


namespace area_of_pentagon_exists_l242_242385

noncomputable def area_pentagon : ℝ :=
  (9 * Real.sqrt 3) / 4 + 25 * Real.sin (100 * Real.pi / 180)

theorem area_of_pentagon_exists
  (convex_FGHIJ : convex FGHIJ)
  (angle_F : ∠F = 100)
  (angle_G : ∠G = 100)
  (FI_eq : FI = 3)
  (IJ_eq : IJ = 3)
  (JG_eq : JG = 3)
  (GH_eq : GH = 5)
  (HF_eq : HF = 5) : 
  area FGHIJ = area_pentagon := 
  sorry

end area_of_pentagon_exists_l242_242385


namespace range_of_a_l242_242208

noncomputable def f (a x : ℝ) : ℝ := log ((1 + 2^x + 4^x * a) / 3)

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ≤ 1 → 1 + 2^x + 4^x * a > 0) → a > -3/4 :=
by
  intro h
  have h_max : ∀ x : ℝ, x ≤ 1 → -((1/4)^x) - ((1/2)^x) ≤ -3/4 := sorry
  sorry

end range_of_a_l242_242208


namespace proof_expression_value_l242_242989

noncomputable def calculate_expression (α : ℝ) : ℝ :=
  (sin (2 * α) - 2 * (cos α)^2) / (sin (α - π / 4))

theorem proof_expression_value (α : ℝ) (h1 : tan (α + π / 4) = -1 / 2) (h2 : π / 2 < α ∧ α < π) :
  calculate_expression α = -2 * sqrt 5 / 5 :=
by
  sorry

end proof_expression_value_l242_242989


namespace doubled_dimensions_volume_l242_242718

noncomputable def original_volume (r h : ℝ) : ℝ :=
  π * r^2 * h

noncomputable def new_volume (r h : ℝ) : ℝ :=
  π * (2 * r)^2 * (2 * h)

theorem doubled_dimensions_volume (r h : ℝ) (h_volume : original_volume r h = 2) : 
  new_volume r h = 16 := by
  sorry

end doubled_dimensions_volume_l242_242718


namespace plumber_max_earnings_l242_242332

-- Define the costs of fixing sink, shower, and toilet
def cost_sink : ℕ := 30
def cost_shower : ℕ := 40
def cost_toilet : ℕ := 50

-- Define the earnings for each job
def job1_earnings : ℕ := 3 * cost_toilet + 3 * cost_sink
def job2_earnings : ℕ := 2 * cost_toilet + 5 * cost_sink
def job3_earnings : ℕ := 1 * cost_toilet + 2 * cost_shower + 3 * cost_sink

-- Define the maximum earnings
def max_earnings : ℕ := max (max job1_earnings job2_earnings) job3_earnings

-- Prove that job2_earnings is the maximum
theorem plumber_max_earnings : max_earnings = job2_earnings :=
by
  -- We need to prove that $250 is the maximum earnings
  have h1 : job1_earnings = 240 := by
    calc
      3 * cost_toilet + 3 * cost_sink
      _ = 3 * 50 + 3 * 30 := rfl
      _ = 150 + 90 := rfl
      _ = 240 := rfl
  have h2 : job2_earnings = 250 := by
    calc
      2 * cost_toilet + 5 * cost_sink
      _ = 2 * 50 + 5 * 30 := rfl
      _ = 100 + 150 := rfl
      _ = 250 := rfl
  have h3 : job3_earnings = 220 := by
    calc
      1 * cost_toilet + 2 * cost_shower + 3 * cost_sink
      _ = 1 * 50 + 2 * 40 + 3 * 30 := rfl
      _ = 50 + 80 + 90 := rfl
      _ = 220 := rfl
  calc
    max_earnings
    _ = max (max 240 250) 220 := by rw [h1, h2, h3]
    _ = max 250 220 := rfl
    _ = 250 := rfl

end plumber_max_earnings_l242_242332


namespace thomas_total_bill_l242_242350

def item_prices : List (String × Float) := [
  ("shirt", 12.00),
  ("pack_of_socks", 5.00),
  ("shorts", 15.00),
  ("swim_trunks", 14.00),
  ("hat", 6.00),
  ("sunglasses", 30.00)
]

def item_quantities : List (String × Nat) := [
  ("shirt", 3),
  ("pack_of_socks", 1),
  ("shorts", 2),
  ("swim_trunks", 1),
  ("hat", 1),
  ("sunglasses", 1)
]

def is_accessory (item : String) : Bool :=
  item = "hat" || item = "sunglasses"

def shipping_cost (total : Float) (is_clothes : Bool) : Float :=
  if total < 50.00 then 5.00
  else if is_clothes then 0.20 * total else 0.10 * total

def total_cost : Float :=
  let total_clothes := item_quantities.foldl (fun acc (item, qty) =>
                        if is_accessory item then acc
                        else acc + qty * (item_prices.lookup item).getD 0)
                      0
  let total_accessories := item_quantities.foldl (fun acc (item, qty) =>
                            if is_accessory item then acc + qty * (item_prices.lookup item).getD 0
                            else acc)
                          0
  let shipping_clothes := shipping_cost total_clothes true
  let shipping_accessories := shipping_cost total_accessories false
  total_clothes + total_accessories + shipping_clothes + shipping_accessories

theorem thomas_total_bill : total_cost = 141.60 :=
by
  sorry

end thomas_total_bill_l242_242350


namespace evaluate_log_l242_242797

noncomputable def log_base (b x : ℝ) : ℝ := real.log x / real.log b

theorem evaluate_log : log_base 3 (9^3) = 6 := by
  have h1 : 9 = 3^2 := by norm_num
  have h2 : 9^3 = (3^2)^3 := by norm_num
  rw [h2, pow_mul]
  have h3 : log_base 3 (3^6) = 6 := by
    rw [← real.log_pow, real.log_mul, mul_comm, ← real.log_rpow, mul_div, real.log_self]
    norm_num
  exact h3

end evaluate_log_l242_242797


namespace range_of_a_l242_242042

theorem range_of_a
  (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
  (h1 : x + 2 * y + 4 = 4 * x * y)
  (h2 : ∀ a : ℝ, (x + 2 * y) * a ^ 2 + 2 * a + 2 * x * y - 34 ≥ 0) : 
  ∀ a : ℝ, a ≤ -3 ∨ a ≥ 5 / 2 :=
by
  sorry

end range_of_a_l242_242042


namespace continuous_g_l242_242181

def g (c d : ℝ) (x : ℝ) : ℝ :=
if x > 3 then c * x + 4
else if x >= -3 then x - 6
else 3 * x - d

theorem continuous_g (c d : ℝ) (H1 : 3 * c + 4 = -3) (H2 : -3 - 6 = -9 - d) : 
  c + d = -7 / 3 :=
by
  sorry

end continuous_g_l242_242181


namespace captivating_quadruples_count_l242_242393

theorem captivating_quadruples_count :
  let n : ℕ := 15
  let condition (a b c d : ℕ) := (1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d ≤ n) ∧ (a + d > 2 * (b + c))
  let count (A : List (ℕ × ℕ × ℕ × ℕ)) := A.length
  let acceptable (quad : (ℕ × ℕ × ℕ × ℕ)) := condition quad.1 quad.2 quad.3 quad.4
  count (List.filter acceptable [ (a, b, c, d) | a ← List.range (n+1), b ← List.range (n+1), c ← List.range (n+1), d ← List.range (n+1)]) = 200 :=
by
  -- Proof will be inserted here
  sorry

end captivating_quadruples_count_l242_242393


namespace log_base_3_of_9_cubed_l242_242898

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 :=
by assumptions
  have h1 : 9 = 3^2 := sorry,
  have h2 : 9^3 = (3^2)^3 := sorry,
  have h3 : (3^2)^3 = 3^(2*3) := sorry,
  have h4 : 3^(2*3) = 3^6 := sorry,
  have h5 : log 3 (3^6) = 6 := by apply log_pow,
  sorry

end log_base_3_of_9_cubed_l242_242898


namespace xy_cubed_identity_l242_242101

namespace ProofProblem

theorem xy_cubed_identity (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 :=
by
  sorry

end ProofProblem

end xy_cubed_identity_l242_242101


namespace isosceles_right_triangle_incenter_distance_l242_242629

theorem isosceles_right_triangle_incenter_distance
  (A B C I : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace I]
  (hABC : IsoscelesRightTriangle A B C)
  (hAC : right_angle A C B)
  (hBC_eq : distance B C = 6)
  (hI : incenter I A B C) :
  distance B I = 6 - 3 * sqrt 2 := 
sorry

end isosceles_right_triangle_incenter_distance_l242_242629


namespace cos_sum_identity_l242_242440

theorem cos_sum_identity (α : ℝ) (h1 : cos α = 3 / 5) (h2 : 0 < α ∧ α < π / 2) :
  cos (π / 3 + α) = (3 - 4 * real.sqrt 3) / 10 := 
by
  sorry

end cos_sum_identity_l242_242440


namespace first_term_exceeding_10000_l242_242226

noncomputable def sequence (n : ℕ) : ℕ :=
  if n = 0 then 2
  else (Finset.range n).sum (λ i, sequence i)

theorem first_term_exceeding_10000 :
  ∃ n : ℕ, sequence n > 10000 ∧ sequence n = 16384 :=
sorry

end first_term_exceeding_10000_l242_242226


namespace problem_theorem_converse_theorem_l242_242645

noncomputable def problem_statement (A B C A' B' C' X Y Z S : Type) [Plane A] [Plane A'] [Plane B] [Plane B'] [Plane C] [Plane C'] : Prop :=
  let α := Plane.create (Line.create B C) (Line.create B' C')
  let β := Plane.create (Line.create C A) (Line.create C' A')
  let γ := Plane.create (Line.create A B) (Line.create A' B')
  ((Line.intersect (Line.create B C) (Line.create B' C') == X) ∧
   (Line.intersect (Line.create C A) (Line.create C' A') == Y) ∧
   (Line.intersect (Line.create A B) (Line.create A' B') == Z)) →
  (X ∈ Line.create Y Z) ∧
  ((Line.intersect (Plane.intersect β γ) (Plane.intersect γ α) == S) ∨ 
  (Plane.parallel (Plane.intersect β γ) (Plane.intersect γ α) (Plane.intersect α β)))

noncomputable def converse_statement (A B C A' B' C' X Y Z S : Type) [Plane A] [Plane A'] [Plane B] [Plane B'] [Plane C] [Plane C'] : Prop :=
  let α := Plane.create (Line.create B C) (Line.create B' C')
  let β := Plane.create (Line.create C A) (Line.create C' A')
  let γ := Plane.create (Line.create A B) (Line.create A' B')
  ((Line.intersect (Plane.intersect β γ) (Plane.intersect γ α) == S) ∨ 
  (Plane.parallel (Plane.intersect β γ) (Plane.intersect γ α) (Plane.intersect α β))) →
  (Line.intersect (Line.create B C) (Line.create B' C') == X) ∧
  (Line.intersect (Line.create C A) (Line.create C' A') == Y) ∧
  (Line.intersect (Line.create A B) (Line.create A' B') == Z) →
  (X ∈ Line.create Y Z)

theorem problem_theorem : problem_statement A B C A' B' C' X Y Z S := sorry

theorem converse_theorem : converse_statement A B C A' B' C' X Y Z S := sorry

end problem_theorem_converse_theorem_l242_242645


namespace min_value_of_alpha_beta_l242_242180

theorem min_value_of_alpha_beta 
  (k : ℝ)
  (h_k : k ≤ -4 ∨ k ≥ 5)
  (α β : ℝ)
  (h_αβ : α^2 - 2 * k * α + (k + 20) = 0 ∧ β^2 - 2 * k * β + (k + 20) = 0) :
  (α + 1) ^ 2 + (β + 1) ^ 2 = 18 → k = -4 :=
sorry

end min_value_of_alpha_beta_l242_242180


namespace cube_identity_l242_242109

theorem cube_identity (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 := by
  sorry

end cube_identity_l242_242109


namespace problem1_part1_problem1_part2_l242_242053

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x - (2 * a / x) - (a + 2) * real.log x

theorem problem1_part1 (a : ℝ) (a_nonneg : a = 0) (x : ℝ) (h : 1 ≤ x ∧ x ≤ 3) :
  (f x a = x - 2 * real.log x) → 
  (∀ x, 1 ≤ x ∧ x ≤ 3 → f x a ≥ 2 - 2 * real.log 2) ∧
  (∀ x, 1 ≤ x ∧ x ≤ 3 → f x a ≤ 1) := 
sorry

theorem problem1_part2 (a : ℝ) (a_pos : a > 0) :
  (∀ x, f x a = x - (2 * a / x) - (a + 2) * real.log x) →
  (if a = 2 then ∀ x, deriv (λ x, f x a) x ≥ 0 else true) ∧
  (if a > 2 then 
    (∀ x, x < 2 → deriv (λ x, f x a) x < 0) ∧
    (∀ x, x > a → deriv (λ x, f x a) x > 0) ∧
    (∀ x, 2 < x ∧ x < a → deriv (λ x, f x a) x < 0) else true) ∧
  (if a < 2 then 
    (∀ x, x < a → deriv (λ x, f x a) x > 0) ∧
    (∀ x, x > 2 → deriv (λ x, f x a) x < 0) ∧
    (∀ x, a < x ∧ x < 2 → deriv (λ x, f x a) x > 0) else true) := 
sorry

end problem1_part1_problem1_part2_l242_242053


namespace consistent_values_for_a_l242_242985

def eq1 (x a : ℚ) : Prop := 10 * x^2 + x - a - 11 = 0
def eq2 (x a : ℚ) : Prop := 4 * x^2 + (a + 4) * x - 3 * a - 8 = 0

theorem consistent_values_for_a : ∃ x, (eq1 x 0 ∧ eq2 x 0) ∨ (eq1 x (-2) ∧ eq2 x (-2)) ∨ (eq1 x (54) ∧ eq2 x (54)) :=
by
  sorry

end consistent_values_for_a_l242_242985


namespace max_product_condition_l242_242414

theorem max_product_condition (x y : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 12) (h3 : 0 ≤ y) (h4 : y ≤ 12) (h_eq : x * y = (12 - x) ^ 2 * (12 - y) ^ 2) : x * y ≤ 81 :=
sorry

end max_product_condition_l242_242414


namespace find_c_l242_242706

theorem find_c 
  (b c : ℝ) 
  (h1 : 4 = 2 * (1:ℝ)^2 + b * (1:ℝ) + c)
  (h2 : 4 = 2 * (5:ℝ)^2 + b * (5:ℝ) + c) : 
  c = 14 := 
sorry

end find_c_l242_242706


namespace max_interior_angles_l242_242536

theorem max_interior_angles (n : ℕ) (h_n : n ≥ 3) (is_regular : ∀ (s : fin n → ℝ), 
  (∀ i, s i = s 0) ∧ ∀ i j, i ≠ j → dist (s i) (s j) = dist (s 0) (s 1)) : 
  if n = 4 then 0 else (n - 3) :=
sorry

end max_interior_angles_l242_242536


namespace arithmetic_seq_sum_l242_242116

noncomputable theory

variable {a : ℕ → ℝ}

def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a 1 - a 0

theorem arithmetic_seq_sum (h_arith : arithmetic_seq a) (h_sum : a 2 + a 3 + a 4 = 12) :
  a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 = 28 :=
sorry

end arithmetic_seq_sum_l242_242116


namespace sum_even_integers_200_to_400_l242_242291

def is_even (n : ℤ) : Prop := n % 2 = 0
def is_in_range (n : ℤ) : Prop := 200 <= n ∧ n <= 400
def is_valid_even_number (n : ℤ) : Prop := is_even n ∧ is_in_range n

theorem sum_even_integers_200_to_400 :
  ∃ (sum : ℤ), (sum = ∑ n in (finset.filter is_valid_even_number (finset.Icc 200 400)), n) ∧ sum = 29700 :=
begin
  sorry
end

end sum_even_integers_200_to_400_l242_242291


namespace card_A_inter_B_l242_242501

noncomputable def A := {x : ℝ | x^2 - 9 * x < 0}
noncomputable def B := {y : ℤ | 4 % y = 0 ∧ y ≠ 0}

theorem card_A_inter_B : (A ∩ ↑B).to_finset.card = 3 := by
  sorry

end card_A_inter_B_l242_242501


namespace movie_of_the_year_l242_242253

theorem movie_of_the_year (members : ℕ) (h : members = 1500) : 
  ∃ (n : ℕ), n = members / 2 ∧ n ≥ 750 :=
by
  use members / 2
  rw h
  have : 1500 / 2 = 750 := rfl
  exact ⟨this.symm, this.ge⟩
«◊
Sorry, there was a misunderstanding. Here is a correct translation of the specific problem:


end movie_of_the_year_l242_242253


namespace sphere_surface_area_l242_242324

theorem sphere_surface_area (a : ℝ) (a_pos : a = 4) : 
  ∃ (r : ℝ), r = 2 * real.sqrt 3 ∧ 4 * real.pi * r^2 = 48 * real.pi :=
by {
  sorry
}

end sphere_surface_area_l242_242324


namespace range_of_a_l242_242991

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x^3 + x

theorem range_of_a 
    (h : ∃ x : ℝ, f(a - x) + f(a * x^2 - 1) < 0) : 
    a ∈ set.Ioo (-(1 : ℝ)) (1 + sqrt 2) / 2 := 
sorry

end range_of_a_l242_242991


namespace intersection_in_fourth_quadrant_l242_242300

def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a
def g (a : ℝ) (x : ℝ) : ℝ := (1 - a) * x

theorem intersection_in_fourth_quadrant (a x : ℝ) (h₁ : a > 1) (h₂ : f a x = g a x) : x > 0 ∧ g a x < 0 :=
by
  sorry

end intersection_in_fourth_quadrant_l242_242300


namespace first_term_exceeds_10000_l242_242219

-- The sequence is defined such that
-- a_1 = 2
-- a_n = sum of all previous terms for n > 1

noncomputable def seq : ℕ → ℕ
| 0     => 2
| (n+1) => ∑ i in Finset.range (n+1), seq i

-- Prove first term that exceeds 10000 is 16384
theorem first_term_exceeds_10000 : ∃ n, seq n > 10000 ∧ seq n = 16384 := by
  sorry

-- Additional helper lemma for the geometric progression relation
lemma seq_geometric : ∀ n, n ≥ 1 → seq (n+1) = 2^(n - 1) := by
  sorry

end first_term_exceeds_10000_l242_242219


namespace normal_prob_l242_242334

noncomputable def ξ : ℝ → ℝ := sorry
variable (σ : ℝ) (h_norm : ξ ∼ NormalDistrib 40 σ^2)
variable (h_prob : P(ξ < 30) = 0.2)

theorem normal_prob (h_norm: ξ ∼ NormalDistrib 40 σ^2) (h_prob: P(ξ < 30) = 0.2) :
    P(30 < ξ < 50) = 0.6 := 
sorry

end normal_prob_l242_242334


namespace log_base_3_of_9_cubed_l242_242867

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 := by
  -- Assuming the necessary properties of logarithms and exponents
  sorry

end log_base_3_of_9_cubed_l242_242867


namespace cubic_sum_l242_242088

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 := by
  sorry

end cubic_sum_l242_242088


namespace arithmetic_sequence_common_difference_l242_242140

-- Define the arithmetic sequence conditions and the target statement
theorem arithmetic_sequence_common_difference :
  ∃ d : ℤ, ∀ (a : ℕ → ℤ), (a 1 = 13) ∧ (a 4 = 1) ∧ 
  (∀ n : ℕ, a n = a 1 + (n - 1) * d) → d = -4 :=
begin
  sorry
end

end arithmetic_sequence_common_difference_l242_242140


namespace number_of_rocks_in_bucket_l242_242155

noncomputable def average_weight_rock : ℝ := 1.5
noncomputable def total_money_made : ℝ := 60
noncomputable def price_per_pound : ℝ := 4

theorem number_of_rocks_in_bucket : 
  let total_weight_rocks := total_money_made / price_per_pound
  let number_of_rocks := total_weight_rocks / average_weight_rock
  number_of_rocks = 10 :=
by
  sorry

end number_of_rocks_in_bucket_l242_242155


namespace length_of_BD_is_six_l242_242517

-- Definitions of the conditions
def AB : ℕ := 6
def BC : ℕ := 11
def CD : ℕ := 6
def DA : ℕ := 8
def BD : ℕ := 6 -- adding correct answer into definition

-- The statement we want to prove
theorem length_of_BD_is_six (hAB : AB = 6) (hBC : BC = 11) (hCD : CD = 6) (hDA : DA = 8) (hBD_int : BD = 6) : 
  BD = 6 :=
by
  -- Proof placeholder
  sorry

end length_of_BD_is_six_l242_242517


namespace log_base_3_of_9_cubed_l242_242831

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 := by
  -- Define the known conditions
  have h1 : 9 = 3^2 := by norm_num
  have h2 : log 3 (a^n) = n * log 3 a := sorry  -- Statement of the logarithmic identity
  
  -- Apply the conditions to prove the result
  calc
    log 3 (9^3)
        = log 3 ((3^2)^3) : by rw h1
    ... = log 3 (3^(2 * 3)) : by congr; norm_num
    ... = log 3 (3^6) : by norm_num
    ... = 6 : by rw [h2]; norm_num

end log_base_3_of_9_cubed_l242_242831


namespace total_logs_in_both_stacks_l242_242265

-- Define the number of logs in the first stack
def first_stack_logs : Nat :=
  let bottom_row := 15
  let top_row := 4
  let number_of_terms := bottom_row - top_row + 1
  let average_logs := (bottom_row + top_row) / 2
  average_logs * number_of_terms

-- Define the number of logs in the second stack
def second_stack_logs : Nat :=
  let bottom_row := 5
  let top_row := 10
  let number_of_terms := top_row - bottom_row + 1
  let average_logs := (bottom_row + top_row) / 2
  average_logs * number_of_terms

-- Prove the total number of logs in both stacks
theorem total_logs_in_both_stacks : first_stack_logs + second_stack_logs = 159 := by
  sorry

end total_logs_in_both_stacks_l242_242265


namespace Martha_reading_challenge_l242_242186

theorem Martha_reading_challenge :
  ∀ x : ℕ,
  (12 + 18 + 14 + 20 + 11 + 13 + 19 + 15 + 17 + x) / 10 = 15 ↔ x = 11 :=
by sorry

end Martha_reading_challenge_l242_242186


namespace Laura_runs_at_6_point_5_mph_l242_242534

theorem Laura_runs_at_6_point_5_mph :
  ∃ x : ℝ, 0 < x ∧
    ((15 / (3 * x)) + (8 / x) = 2) →
    x = 6.5 :=
begin
  sorry
end

end Laura_runs_at_6_point_5_mph_l242_242534


namespace distinct_cyclic_quadrilaterals_perimeter_36_l242_242066

noncomputable def count_distinct_cyclic_quadrilaterals : Nat :=
  1026

theorem distinct_cyclic_quadrilaterals_perimeter_36 :
  (∃ (a b c d : ℕ), a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ a + b + c + d = 36 ∧ a < b + c + d) → count_distinct_cyclic_quadrilaterals = 1026 :=
by
  rintro ⟨a, b, c, d, hab, hbc, hcd, hsum, hlut⟩
  sorry

end distinct_cyclic_quadrilaterals_perimeter_36_l242_242066


namespace largest_divisible_by_88_l242_242672

theorem largest_divisible_by_88 (n : ℕ) (h₁ : n = 9999) (h₂ : n % 88 = 55) : n - 55 = 9944 := by
  sorry

end largest_divisible_by_88_l242_242672


namespace sum_even_200_to_400_l242_242299

theorem sum_even_200_to_400 : 
  let t₁ := 202
  let tₙ := 398
  let n := 99
  let sum := (t₁ + tₙ) * n / 2 in
  sum = 29700 := 
by
  let t₁ := 202
  let tₙ := 398
  let n := 99
  let sum := (t₁ + tₙ) * n / 2
  show sum = 29700
  sorry

end sum_even_200_to_400_l242_242299


namespace mutual_fund_percent_increase_l242_242729

def percent_increase (initial final : ℝ) : ℝ :=
  ((final - initial) / initial) * 100

theorem mutual_fund_percent_increase (P : ℝ) (h₁ : 1.25 * P = P + 0.25 * P)
    (h₂ : 1.80 * P = P + 0.80 * P) :
    percent_increase (1.25 * P) (1.80 * P) = 44 :=
  sorry

end mutual_fund_percent_increase_l242_242729


namespace log_base_three_of_nine_cubed_l242_242881

theorem log_base_three_of_nine_cubed : log 3 (9^3) = 6 :=
by
  have h1 : 9 = 3^2 := by sorry
  have h2 : ∀ (a m n: ℕ), (a^m)^n = a^(m * n) := by 
    intros a m n
    exact pow_mul a m n
  have h3 : ∀ (b n: ℕ), log b (b^n) = n := by sorry
  sorry

end log_base_three_of_nine_cubed_l242_242881


namespace largest_angle_triangle_DEF_l242_242528

noncomputable theory

def triangle_largest_angle
  (d e f : ℝ)
  (h1 : d + 3 * e + 3 * f = d^2)
  (h2 : d + 3 * e - 3 * f = -5) : ℝ :=
  120

theorem largest_angle_triangle_DEF (d e f : ℝ) (h1 : d + 3 * e + 3 * f = d^2) (h2 : d + 3 * e - 3 * f = -5) :
  triangle_largest_angle d e f h1 h2 = 120 :=
sorry

end largest_angle_triangle_DEF_l242_242528


namespace part1_part2_l242_242151
noncomputable theory

-- Define the function for part (1) of the problem.
def f (x A : Real) : Real := Real.sin (2 * x + A)

-- Part (1): Proving the value of f(-π/6) when A = π/2
theorem part1 (A : Real) (hA : A = Real.pi / 2) : f (-Real.pi / 6) A = -1 / 2 :=
by
  sorry

-- Part (2): Given f(π/12) = 1, a = 3, cos B = 4/5, find the length of b.
theorem part2 (A B a b : Real) (k : Int) (h1 : f (Real.pi / 12) A = 1) (h2 : a = 3) (h3 : Real.cos B = 4 / 5) (h4 : b = 6 * Real.sqrt 3 / 5) :
  b = 6 * Real.sqrt 3 / 5 :=
by
  sorry

end part1_part2_l242_242151


namespace inequality_holds_for_positive_reals_l242_242556

theorem inequality_holds_for_positive_reals (x y : ℝ) (m n : ℤ) 
  (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) :
  (1 - x^n)^m + (1 - y^m)^n ≥ 1 :=
sorry

end inequality_holds_for_positive_reals_l242_242556


namespace log_pow_evaluation_l242_242916

-- Define the condition that 9 is equal to 3^2
def nine_eq_three_squared : Prop := 9 = 3^2

-- The main theorem that we need to prove
theorem log_pow_evaluation (h : nine_eq_three_squared) : log 3 (9^3) = 6 :=
  sorry

end log_pow_evaluation_l242_242916


namespace cost_price_approx_l242_242714

noncomputable def cost_price (selling_price : ℝ) (profit_percent : ℝ) : ℝ :=
  selling_price / (1 + profit_percent / 100)

theorem cost_price_approx :
  ∀ (selling_price profit_percent : ℝ),
  selling_price = 2552.36 →
  profit_percent = 6 →
  abs (cost_price selling_price profit_percent - 2407.70) < 0.01 :=
by
  intros selling_price profit_percent h1 h2
  sorry

end cost_price_approx_l242_242714


namespace fund_remaining_after_trip_l242_242566

-- Define the conditions in Lean 4
variables (initial_fund : ℕ) (student_contribution : ℕ) (num_students : ℕ) (cost_per_student : ℕ)

-- Set the specific values for the conditions
def initial_fund := 50
def student_contribution := 5
def num_students := 20
def cost_per_student := 7

-- Statement of the problem in Lean 4
theorem fund_remaining_after_trip :
  initial_fund + (num_students * student_contribution) - (num_students * cost_per_student) = 10 := 
by 
  sorry

end fund_remaining_after_trip_l242_242566


namespace water_lilies_half_coverage_l242_242132

theorem water_lilies_half_coverage (doubles : ∀ n : ℕ, area (n + 1) = 2 * area n) 
  (full_coverage_day : area 48 = lake_area) : 
  area 47 = lake_area / 2 := sorry

end water_lilies_half_coverage_l242_242132


namespace evaluate_log_l242_242792

noncomputable def log_base (b x : ℝ) : ℝ := real.log x / real.log b

theorem evaluate_log : log_base 3 (9^3) = 6 := by
  have h1 : 9 = 3^2 := by norm_num
  have h2 : 9^3 = (3^2)^3 := by norm_num
  rw [h2, pow_mul]
  have h3 : log_base 3 (3^6) = 6 := by
    rw [← real.log_pow, real.log_mul, mul_comm, ← real.log_rpow, mul_div, real.log_self]
    norm_num
  exact h3

end evaluate_log_l242_242792


namespace seq_sum_five_l242_242146

noncomputable def a : ℕ → ℤ
| 0 := 1
| (n + 1) := a n ^ 2 - 1

theorem seq_sum_five : a 0 + a 1 + a 2 + a 3 + a 4 = -1 :=
by
  sorry

end seq_sum_five_l242_242146


namespace least_positive_base_ten_number_seven_binary_digits_l242_242656

theorem least_positive_base_ten_number_seven_binary_digits : 
  ∃ n : ℕ, (n > 0) ∧ (n < 2^7) ∧ (n ≥ 2^6) ∧ (n = 64) :=
by {
  use 64,
  split,
  { exact Nat.zero_lt_succ _ },
  split,
  { norm_num },
  split,
  { norm_num },
  { reflexivity }
}

end least_positive_base_ten_number_seven_binary_digits_l242_242656


namespace log_base_3_l242_242849

theorem log_base_3 (log_identity : ∀ (b a c : ℝ), log b (a^c) = c * log b a) :
  log 3 (9^3) = 6 :=
by
  have : 9 = 3^2 := by sorry
  have : log 3 3 = 1 := by sorry
  exact (log_identity 3 (3^2) 3).trans sorry

end log_base_3_l242_242849


namespace validColoringsCount_l242_242386

-- Define the initial conditions
def isValidColoring (n : ℕ) (color : ℕ → ℕ) : Prop :=
  ∀ i ∈ Finset.range (n - 1), 
    (i % 2 = 1 → (color i = 1 ∨ color i = 3)) ∧
    color i ≠ color (i + 1)

noncomputable def countValidColorings : ℕ → ℕ
| 0     => 1
| 1     => 2
| (n+2) => 
    match n % 2 with
    | 0 => 2 * 3^(n/2)
    | _ => 4 * 3^((n-1)/2)

-- Main theorem
theorem validColoringsCount (n : ℕ) :
  (∀ color : ℕ → ℕ, isValidColoring n color) →
  (if n % 2 = 0 then countValidColorings n = 4 * 3^((n / 2) - 1) 
     else countValidColorings n = 2 * 3^(n / 2)) :=
by
  sorry

end validColoringsCount_l242_242386


namespace quadratic_inequality_sufficient_not_necessary_condition_l242_242170

theorem quadratic_inequality_sufficient_not_necessary_condition
  (m : ℝ)
  (p : ∀ x : ℝ, x^2 - m*x + 1 > 0)
  (q : -2 ≤ m ∧ m ≤ 2) :
  (∀ x, x^2 - m*x + 1 > 0) → (-2 < m ∧ m < 2) :=
begin
  assume h,
  have H : m^2 - 4 < 0,
  { specialize h 0,
    linarith },
  linarith,
end

example (m : ℝ) (p : ∀ x : ℝ, x^2 - m*x + 1 > 0) (q : -2 ≤ m ∧ m ≤ 2) :
  (∀ x, x^2 - m*x + 1 > 0) → (-2 < m ∧ m < 2) :=
by { apply quadratic_inequality_sufficient_not_necessary_condition, }

end quadratic_inequality_sufficient_not_necessary_condition_l242_242170


namespace binomial_30_3_l242_242372

theorem binomial_30_3 : nat.choose 30 3 = 4060 := by
  sorry

end binomial_30_3_l242_242372


namespace sum_third_column_l242_242750

variable (a b c d e f g h i : ℕ)

theorem sum_third_column :
  (a + b + c = 24) →
  (d + e + f = 26) →
  (g + h + i = 40) →
  (a + d + g = 27) →
  (b + e + h = 20) →
  (c + f + i = 43) :=
by
  intros
  sorry

end sum_third_column_l242_242750


namespace x_value_for_divisibility_l242_242410

theorem x_value_for_divisibility (x : ℕ) (h1 : x = 0 ∨ x = 5) (h2 : (8 * 10 + x) % 4 = 0) : x = 0 :=
by
  sorry

end x_value_for_divisibility_l242_242410


namespace cubic_sum_l242_242089

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 := by
  sorry

end cubic_sum_l242_242089


namespace find_initial_price_l242_242344

def discount_price (initial_price : ℝ) : ℝ :=
  initial_price * 0.75 * 0.85 * 0.90 * 0.95 * 0.97

theorem find_initial_price (final_price : ℝ) (P : ℝ) : 
  discount_price P = final_price → P ≈ 11691.89 := by
  sorry

end find_initial_price_l242_242344


namespace true_proposition_l242_242439

def p1 (x : ℝ) : Prop :=
  (sin x ≠ 0) → (sin x + 1 / sin x ≥ 2)

def p2 (x y : ℝ) : Prop :=
  (x + y = 0) ↔ (x / y = -1)

theorem true_proposition : (¬ p1 x) ∨ p2 x y := 
begin
  sorry -- The actual proof will be here
end

end true_proposition_l242_242439


namespace a_3_eq_33_l242_242001

noncomputable def a : ℕ → ℤ
| 0       := 1
| 1       := 3
| (n + 1) := if h : n ≥ 1 then (a n) ^ 2 + (-1) ^ n / (a (n - 1)) else 1

theorem a_3_eq_33 : a 3 = 33 := 
by {
    -- Using the conditions to prove the assertion
    sorry
}

end a_3_eq_33_l242_242001


namespace first_term_exceeding_10000_l242_242227

noncomputable def sequence (n : ℕ) : ℕ :=
  if n = 0 then 2
  else (Finset.range n).sum (λ i, sequence i)

theorem first_term_exceeding_10000 :
  ∃ n : ℕ, sequence n > 10000 ∧ sequence n = 16384 :=
sorry

end first_term_exceeding_10000_l242_242227


namespace binom_computation_l242_242378

noncomputable def binom : ℕ → ℕ → ℕ
| n, 0       => 1
| 0, k+1     => 0
| n+1, k+1   => binom n k + binom n (k+1)

theorem binom_computation :
  (binom 10 3) * (binom 8 3) = 6720 := by
  sorry

end binom_computation_l242_242378


namespace neighbor_receives_correct_amount_of_chocolate_l242_242322

-- Definitions based on the conditions:
def total_chocolate : ℚ := 72 / 7
def num_packages : ℚ := 6
def packages_for_neighbor : ℚ := 2

-- The main statement to be proven:
theorem neighbor_receives_correct_amount_of_chocolate :
  let chocolate_per_package := total_chocolate / num_packages in
  let neighbor_chocolate := packages_for_neighbor * chocolate_per_package in
  neighbor_chocolate = 24 / 7 :=
by
  sorry

end neighbor_receives_correct_amount_of_chocolate_l242_242322


namespace convex_polygon_interior_angle_l242_242243

theorem convex_polygon_interior_angle (n : ℕ) (h1 : 3 ≤ n)
  (h2 : (n - 2) * 180 = 2570 + x) : x = 130 :=
sorry

end convex_polygon_interior_angle_l242_242243


namespace angle_ACB_eq_60_l242_242158

variable (A B C I M_a M_b A' B' : Type*)

variables [Incenter I] [Midpoint M_b AC] [Midpoint M_a BC]
variables [Intersect M_aI AC A'] [Intersect M_bI BC B']

theorem angle_ACB_eq_60 (h_area : area (triangle A B C) = area (triangle A' B' C)) : angle A C B = 60 :=
by
  -- Skipping proof for now
  sorry

end angle_ACB_eq_60_l242_242158


namespace sum_even_integers_between_200_and_400_l242_242285

theorem sum_even_integers_between_200_and_400 : 
  (Finset.sum (Finset.filter (λ x, x % 2 = 0) (Finset.range 401)) 
    - Finset.sum (Finset.filter (λ x, x % 2 = 0) (Finset.range 201)))  = 30100 :=
begin
  sorry
end

end sum_even_integers_between_200_and_400_l242_242285


namespace cubic_sum_l242_242090

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 := by
  sorry

end cubic_sum_l242_242090


namespace first_term_exceeds_10000_l242_242222

def sequence : ℕ → ℕ
| 0     := 2
| (n+1) := (finset.sum (finset.range n.succ) sequence) 

theorem first_term_exceeds_10000 :
  ∃ n, sequence n > 10000 ∧ sequence n = 16384 :=
sorry

end first_term_exceeds_10000_l242_242222


namespace log_base_three_of_nine_cubed_l242_242887

theorem log_base_three_of_nine_cubed : log 3 (9^3) = 6 :=
by
  have h1 : 9 = 3^2 := by sorry
  have h2 : ∀ (a m n: ℕ), (a^m)^n = a^(m * n) := by 
    intros a m n
    exact pow_mul a m n
  have h3 : ∀ (b n: ℕ), log b (b^n) = n := by sorry
  sorry

end log_base_three_of_nine_cubed_l242_242887


namespace log_base_3_of_9_cubed_l242_242931
-- Import the necessary parts of the standard library

-- Define the statement to be proved
theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 :=
by
  sorry

end log_base_3_of_9_cubed_l242_242931


namespace log_base_3_of_9_cubed_l242_242833

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 := by
  -- Define the known conditions
  have h1 : 9 = 3^2 := by norm_num
  have h2 : log 3 (a^n) = n * log 3 a := sorry  -- Statement of the logarithmic identity
  
  -- Apply the conditions to prove the result
  calc
    log 3 (9^3)
        = log 3 ((3^2)^3) : by rw h1
    ... = log 3 (3^(2 * 3)) : by congr; norm_num
    ... = log 3 (3^6) : by norm_num
    ... = 6 : by rw [h2]; norm_num

end log_base_3_of_9_cubed_l242_242833


namespace x_cubed_plus_y_cubed_l242_242083

variable (x y : ℝ)
variable (h₁ : x + y = 5)
variable (h₂ : x^2 + y^2 = 17)

theorem x_cubed_plus_y_cubed :
  x^3 + y^3 = 65 :=
by sorry

end x_cubed_plus_y_cubed_l242_242083


namespace pentagons_count_at_least_13_l242_242693

noncomputable def pentagons_count :=
    let V : ℕ := ... -- number of vertices
    let E : ℕ := ... -- number of edges
    let F : ℕ := ... -- number of faces
    let a : ℕ := ... -- number of pentagons
    let b : ℕ := ... -- number of hexagons
    -- assuming the conditions from the problem
    (V - E + F = 2) ∧
    (E ≥ 3 * V / 2) ∧
    (E ≤ 3 * F - 6) ∧
    (F = a + b + 1) ∧
    (5 * a + 6 * b + 7 = 2 * E)
    → a ≥ 13

-- declare the proof function
theorem pentagons_count_at_least_13 : pentagons_count :=
by
  sorry

end pentagons_count_at_least_13_l242_242693


namespace cistern_capacity_l242_242671

noncomputable theory

/-- The capacity of the cistern is determined given the rates at which the pipes fill and empty it. -/
theorem cistern_capacity
  (C : ℝ) -- capacity of the cistern in liters
  (rate_A : ℝ) -- rate of pipe A in liters per minute
  (rate_B : ℝ) -- rate of pipe B in liters per minute
  (rate_C : ℝ) -- rate of pipe C in liters per minute
  (emptying_time : ℝ) -- time in minutes to empty the cistern when all pipes are open
  (hA : rate_A = C / (15 / 2)) -- pipe A fills the cistern in 15/2 minutes
  (hB : rate_B = C / 5) -- pipe B fills the cistern in 5 minutes
  (hC : rate_C = 14) -- pipe C empties the cistern at a rate of 14 liters per minute
  (ht : emptying_time = 60) -- the cistern is emptied in 60 minutes
  : C = 840 / 17 :=
by {
  sorry
}

end cistern_capacity_l242_242671


namespace proof_A_n_plus_B_n_l242_242595

-- Define the sums and differences based on given conditions

noncomputable def A_n (n : ℕ) : ℕ := (2*n-1).choose(2) - (2*n-3).choose(2)

noncomputable def B_n (n : ℕ) : ℕ := (2*n-1)^3 - (2*(n-1))^3

-- Theorem to prove that A_n + B_n = 2n^3
theorem proof_A_n_plus_B_n (n : ℕ) : A_n n + B_n n = 2*n^3 := 
by sorry

end proof_A_n_plus_B_n_l242_242595


namespace circle_equation_l242_242975

variable (x y : ℝ)

def center : ℝ × ℝ := (4, -6)
def radius : ℝ := 3

theorem circle_equation : (x - center.1)^2 + (y - center.2)^2 = radius^2 ↔ (x - 4)^2 + (y + 6)^2 = 9 :=
by
  sorry

end circle_equation_l242_242975


namespace binomial_constant_term_l242_242519

theorem binomial_constant_term : 
  let T_r(r : ℕ) := binomial 6 r * ((-2)^r) * (x : ℚ) ^ (6 - 2 * r)
  in (6 - 2 * 3 = 0) → (6.choose 3 * (-2)^3) = -160 := 
begin
  intro,
  sorry
end

end binomial_constant_term_l242_242519


namespace initial_candies_count_l242_242587

-- Definitions based on conditions
def NelliesCandies : Nat := 12
def JacobsCandies : Nat := NelliesCandies / 2
def LanasCandies : Nat := JacobsCandies - 3
def TotalCandiesEaten : Nat := NelliesCandies + JacobsCandies + LanasCandies
def RemainingCandies : Nat := 3 * 3
def InitialCandies := TotalCandiesEaten + RemainingCandies

-- Theorem stating the initial candies count
theorem initial_candies_count : InitialCandies = 30 := by 
  sorry

end initial_candies_count_l242_242587


namespace total_nominal_income_l242_242570

theorem total_nominal_income
  (c1 : 8700 * ((1 + 0.06 / 12) ^ 6 - 1) = 264.28)
  (c2 : 8700 * ((1 + 0.06 / 12) ^ 5 - 1) = 219.69)
  (c3 : 8700 * ((1 + 0.06 / 12) ^ 4 - 1) = 175.31)
  (c4 : 8700 * ((1 + 0.06 / 12) ^ 3 - 1) = 131.15)
  (c5 : 8700 * ((1 + 0.06 / 12) ^ 2 - 1) = 87.22)
  (c6 : 8700 * (1 + 0.06 / 12 - 1) = 43.5) :
  264.28 + 219.69 + 175.31 + 131.15 + 87.22 + 43.5 = 921.15 := by
  sorry

end total_nominal_income_l242_242570


namespace common_tangent_line_b_l242_242124

theorem common_tangent_line_b
  (f g : ℝ → ℝ)
  (l : ℝ → ℝ)
  (b : ℝ)
  (hx : ∃ x₁ x₂, f = λ x, Real.exp x ∧ g = λ x, Real.exp 2 * Real.log x ∧
    l = λ x, (Real.exp x₁) * x + ((1 - x₁) * Real.exp x₁) ∧
    l = λ x, (Real.exp 2 / x₂) * x + Real.exp 2 * (Real.log x₂ - 1))
  : b = 0 ∨ b = -Real.exp 2 :=
sorry

end common_tangent_line_b_l242_242124


namespace log_base_3_of_9_cubed_l242_242929
-- Import the necessary parts of the standard library

-- Define the statement to be proved
theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 :=
by
  sorry

end log_base_3_of_9_cubed_l242_242929


namespace log_three_nine_cubed_l242_242962

theorem log_three_nine_cubed : ∀ (log : ℕ → ℕ → ℕ), log 3 9 = 2 → log 3 3 = 1 → (∀ (a b n : ℕ), log a (b ^ n) = n * log a b) → log 3 (9 ^ 3) = 6 :=
by
  intros log h1 h2 h3
  sorry

end log_three_nine_cubed_l242_242962


namespace saved_fireworks_l242_242065

theorem saved_fireworks (h_bought : 2) (f_bought : 3) (total_fireworks : 11) : 6 = total_fireworks - (h_bought + f_bought) :=
by
  -- Using the provided conditions
  let new_fireworks := h_bought + f_bought
  let saved_from_last_year := total_fireworks - new_fireworks
  -- Verify the hypothesis matches the calculated saved fireworks
  have h_save : saved_from_last_year = (11 - (2 + 3)) := rfl
  -- Show that 6 = 11 - (2 + 3)
  sorry

end saved_fireworks_l242_242065


namespace a_2023_is_neg_2_l242_242493

def seq (n : ℕ) : ℤ :=
  if n = 1 then 2
  else -|seq (n - 1) + 5|

theorem a_2023_is_neg_2 : seq 2023 = -2 := sorry

end a_2023_is_neg_2_l242_242493


namespace find_s_l242_242404

theorem find_s (s : ℝ) (h : 4 * log 3 s = log 3 (4 * s^2)) : s = 2 :=
sorry

end find_s_l242_242404


namespace constant_term_in_expansion_l242_242521

theorem constant_term_in_expansion :
  (∃ n : ℕ, (n = 6) ∧ 
   (∃ k : ℕ, (k = 4) ∧ 
   ((x^2 - (1/x))^n).coeff (12 - 3*k) = 15)) := by
  sorry

end constant_term_in_expansion_l242_242521


namespace log_base_3_of_9_cubed_l242_242863

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 := by
  -- Assuming the necessary properties of logarithms and exponents
  sorry

end log_base_3_of_9_cubed_l242_242863


namespace find_x_l242_242543

noncomputable def x : ℚ := 4/3

def intPart (x: ℚ) : ℤ := ⌊x⌋

def fracPart (x: ℚ) : ℚ := x - ⌊x⌋

def satisfiesEquation (x : ℚ) : Prop :=
  2 * x + 3 * (intPart x) - 5 * (fracPart x) = 4

theorem find_x : ∃ x : ℚ, satisfiesEquation x ∧ x = 4/3 :=
by
  use 4/3
  split
  {
    unfold satisfiesEquation
    norm_num
    sorry
  }
  exact rfl

end find_x_l242_242543


namespace sqrt_inequality_l242_242600

theorem sqrt_inequality (x : ℝ) (hx : x ≥ 4) : 
  sqrt (x - 3) + sqrt (x - 2) > sqrt (x - 4) + sqrt (x - 1) := 
sorry

end sqrt_inequality_l242_242600


namespace problem_solution_l242_242184

noncomputable def f : ℕ → ℝ → ℝ
| 0, x => Real.sin x
| (n+1), x => Real.derivative (f n) x

theorem problem_solution : 
  ∑ i in Finset.range 2017, f (i + 1) (15 * Real.pi / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 :=
by
  sorry

end problem_solution_l242_242184


namespace log_b_2024_l242_242745

noncomputable def star (a b : ℝ) : ℝ := a ^ (Real.log10 b)
noncomputable def op (a b : ℝ) : ℝ := a ^ (2 / (Real.log10 b))

def b : ℕ → ℝ
| 4     := op 4 3
| (n+1) := star (op n n.pred) (b n) if n ≥ 4

def log_b (n : ℕ) : ℝ := Real.log10 (b n)

theorem log_b_2024 : log_b 2024 = 10 :=
by
  sorry

end log_b_2024_l242_242745


namespace locus_of_midpoint_l242_242624

-- Define the problem conditions
variables {E F G : Point} -- Points in space
variables (e f : Line) -- Perpendicular lines 
variables (A B : Point) -- Endpoints of segment AB
variables [unit_length_AB : (A.distance B = 1)] -- AB is of unit length
variables [A_on_e : A ∈ e]
variables [B_on_f : B ∈ f]
variables [perpendicular_e_f : Perpendicular e f]

-- Definition of the midpoint \( G \)
noncomputable def midpoint (A B : Point) : Point := ...
-- Distance between points
def distance (P Q : Point) : ℝ := ...

-- The definition implying G is the midpoint of \( AB \)
axiom midpoint_eq : G = midpoint A B

-- The main theorem stating the locus of G
theorem locus_of_midpoint (AB : Line) (G : Point) (E F : Point) (r : ℝ) :
  (A ∈ e) → (B ∈ f) → Perpendicular e f → (A.distance B = 1) →
  (G = midpoint A B) → 
  (G.distance (midpoint E F) = sqrt (1/4 - (E.distance F / 2)^2)) :=
sorry

end locus_of_midpoint_l242_242624


namespace log_base_three_of_nine_cubed_l242_242886

theorem log_base_three_of_nine_cubed : log 3 (9^3) = 6 :=
by
  have h1 : 9 = 3^2 := by sorry
  have h2 : ∀ (a m n: ℕ), (a^m)^n = a^(m * n) := by 
    intros a m n
    exact pow_mul a m n
  have h3 : ∀ (b n: ℕ), log b (b^n) = n := by sorry
  sorry

end log_base_three_of_nine_cubed_l242_242886


namespace log_base_3_l242_242850

theorem log_base_3 (log_identity : ∀ (b a c : ℝ), log b (a^c) = c * log b a) :
  log 3 (9^3) = 6 :=
by
  have : 9 = 3^2 := by sorry
  have : log 3 3 = 1 := by sorry
  exact (log_identity 3 (3^2) 3).trans sorry

end log_base_3_l242_242850


namespace log3_of_9_to_3_l242_242807

theorem log3_of_9_to_3 : Real.logb 3 (9^3) = 6 :=
by
  have h1 : 9 = 3^2 := by norm_num
  have h2 : 9^3 = (3^2)^3 := by rw [h1]
  have h3 : (3^2)^3 = 3^(2 * 3) := by rw [pow_mul]
  have h4 : Real.logb 3 (3^(2 * 3)) = 2 * 3 := Real.logb_pow 3 (2 * 3)
  rw [h4]
  norm_num

end log3_of_9_to_3_l242_242807


namespace A_is_11_years_older_than_B_l242_242129

-- Define the constant B as given in the problem
def B : ℕ := 41

-- Define the condition based on the problem statement
def condition (A : ℕ) := A + 10 = 2 * (B - 10)

-- Prove the main statement that A is 11 years older than B
theorem A_is_11_years_older_than_B (A : ℕ) (h : condition A) : A - B = 11 :=
by
  sorry

end A_is_11_years_older_than_B_l242_242129


namespace total_days_1996_to_2000_l242_242486

def is_leap_year (year : ℕ) : Prop :=
  (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ (year % 400 = 0)

def days_in_year (year : ℕ) : ℕ :=
  if is_leap_year year then 366 else 365

theorem total_days_1996_to_2000 :
  (days_in_year 1996) + (days_in_year 1997) + (days_in_year 1998) + (days_in_year 1999) + (days_in_year 2000) = 1827 :=
by sorry

end total_days_1996_to_2000_l242_242486


namespace divisible_by_five_l242_242034

theorem divisible_by_five (x y z : ℤ) (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x) :
  ∃ k : ℤ, (x-y)^5 + (y-z)^5 + (z-x)^5 = 5 * k * (y-z) * (z-x) * (x-y) :=
  sorry

end divisible_by_five_l242_242034


namespace binom_multiplication_l242_242382

open BigOperators

noncomputable def choose_and_multiply (n k m l : ℕ) : ℕ :=
  Nat.choose n k * Nat.choose m l

theorem binom_multiplication : choose_and_multiply 10 3 8 3 = 6720 := by
  sorry

end binom_multiplication_l242_242382


namespace find_f_inv_64_l242_242077

def f (x : ℝ) : ℝ := sorry

theorem find_f_inv_64 :
  f(3) = 1 ∧ (∀ x, f(2 * x) = 2 * f(x)) → f⁻¹ 64 = 192 := 
sorry

end find_f_inv_64_l242_242077


namespace first_term_exceeds_10000_l242_242221

-- The sequence is defined such that
-- a_1 = 2
-- a_n = sum of all previous terms for n > 1

noncomputable def seq : ℕ → ℕ
| 0     => 2
| (n+1) => ∑ i in Finset.range (n+1), seq i

-- Prove first term that exceeds 10000 is 16384
theorem first_term_exceeds_10000 : ∃ n, seq n > 10000 ∧ seq n = 16384 := by
  sorry

-- Additional helper lemma for the geometric progression relation
lemma seq_geometric : ∀ n, n ≥ 1 → seq (n+1) = 2^(n - 1) := by
  sorry

end first_term_exceeds_10000_l242_242221


namespace log_base_three_of_nine_cubed_l242_242880

theorem log_base_three_of_nine_cubed : log 3 (9^3) = 6 :=
by
  have h1 : 9 = 3^2 := by sorry
  have h2 : ∀ (a m n: ℕ), (a^m)^n = a^(m * n) := by 
    intros a m n
    exact pow_mul a m n
  have h3 : ∀ (b n: ℕ), log b (b^n) = n := by sorry
  sorry

end log_base_three_of_nine_cubed_l242_242880


namespace mrs_hilt_baked_pecan_pies_l242_242584

/-- Mrs. Hilt baked a certain number of pecan pies. This proof establishes 
that given the conditions on the number of apple pies and the total pie arrangement 
in rows, the number of pecan pies baked is as determined. -/
theorem mrs_hilt_baked_pecan_pies :
  let apple_pies := 14 in
  let total_pies := 30 * 5 in
  let pecan_pies := total_pies - apple_pies in
  pecan_pies = 136 :=
by
  let apple_pies := 14
  let total_pies := 30 * 5
  let pecan_pies := total_pies - apple_pies
  show pecan_pies = 136 from sorry

end mrs_hilt_baked_pecan_pies_l242_242584


namespace number_of_1989_periodic_points_l242_242171

noncomputable theory

open Complex

def unit_circle : Set ℂ := {z : ℂ | Complex.abs z = 1}

def f (m : ℕ) (z : ℂ) : ℂ := z^m

def is_periodic_point (f : ℂ → ℂ) (z : ℂ) (n : ℕ) : Prop :=
  (∀ k, 1 ≤ k ∧ k < n → f^[k] z ≠ z) ∧ (f^[n] z = z)

theorem number_of_1989_periodic_points (m : ℕ) (h_m : 1 < m) :
  ∃ count : ℕ,
    count = m^1989 - m^663 - m^153 - m^117 + m^51 + m^39 + m^9 - m^3 := sorry

end number_of_1989_periodic_points_l242_242171


namespace derivative_of_f_l242_242615

noncomputable def f (x : ℝ) : ℝ := x * Real.cos x - Real.sin x

theorem derivative_of_f : ∀ x : ℝ, deriv f x = -x * Real.sin x := by
  sorry

end derivative_of_f_l242_242615


namespace xy_cubed_identity_l242_242100

namespace ProofProblem

theorem xy_cubed_identity (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 :=
by
  sorry

end ProofProblem

end xy_cubed_identity_l242_242100


namespace sum_even_integers_200_to_400_l242_242294

def is_even (n : ℤ) : Prop := n % 2 = 0
def is_in_range (n : ℤ) : Prop := 200 <= n ∧ n <= 400
def is_valid_even_number (n : ℤ) : Prop := is_even n ∧ is_in_range n

theorem sum_even_integers_200_to_400 :
  ∃ (sum : ℤ), (sum = ∑ n in (finset.filter is_valid_even_number (finset.Icc 200 400)), n) ∧ sum = 29700 :=
begin
  sorry
end

end sum_even_integers_200_to_400_l242_242294


namespace a_2023_is_neg_2_l242_242494

def seq (n : ℕ) : ℤ :=
  if n = 1 then 2
  else -|seq (n - 1) + 5|

theorem a_2023_is_neg_2 : seq 2023 = -2 := sorry

end a_2023_is_neg_2_l242_242494


namespace log_pow_evaluation_l242_242906

-- Define the condition that 9 is equal to 3^2
def nine_eq_three_squared : Prop := 9 = 3^2

-- The main theorem that we need to prove
theorem log_pow_evaluation (h : nine_eq_three_squared) : log 3 (9^3) = 6 :=
  sorry

end log_pow_evaluation_l242_242906


namespace prove_correct_statement_l242_242301

-- Definitions of the given statements
def statementA : Prop := 
  "It is suitable to use a comprehensive survey to understand the service life of a certain brand of mobile phone."

def statementB : Prop := 
  "It is suitable to use a sampling survey to investigate whether athletes at the Beijing Winter Olympics use stimulants."

def statementC : Prop := 
  "Encountering a red light at a traffic intersection with traffic lights is a random event."

def statementD : Prop := 
  "To understand the 'double reduction' effect, it is reasonable for a school to randomly select homework from ninth-grade students for comparison."

-- Main theorem to be proved
theorem prove_correct_statement : statementC := sorry

end prove_correct_statement_l242_242301


namespace length_of_PR_l242_242520

-- Given conditions and target proof statement
noncomputable def length_PR (x y : ℝ) : ℝ := Real.sqrt (x ^ 2 + y ^ 2)

theorem length_of_PR {x y : ℝ} (h_total_area : (x ^ 2 + y ^ 2) / 2 = 300) 
  (h_eq_600 : x^2 + y^2 = 600) : length_PR x y = 10 * Real.sqrt 6 :=
by
  -- Assume x^2 + y^2 = 600 from the conditions
  rw [←h_eq_600]
  -- Directly compute the length of PR
  unfold length_PR
  -- Simplify the square root operation
  simp
  -- The statement will be:
  sorry

end length_of_PR_l242_242520


namespace binom_multiplication_l242_242381

open BigOperators

noncomputable def choose_and_multiply (n k m l : ℕ) : ℕ :=
  Nat.choose n k * Nat.choose m l

theorem binom_multiplication : choose_and_multiply 10 3 8 3 = 6720 := by
  sorry

end binom_multiplication_l242_242381


namespace periodic_f_sum_l242_242690

def f (x : ℝ) : ℝ :=
  if -2 ≤ x ∧ x < 0 then -(x + 3)^2
  else if 0 ≤ x ∧ x < 3 then x
  else 0  -- periodic extension

theorem periodic_f_sum : 
  (f 1) + (f 2) + (f 3) + (f 4) + (f 5) + (f 6) + (f 7) + (f 8) + (f 9) + (f 10) + 
  (f 11) + (f 12) + (f 13) + (f 14) + (f 15) + (f 16) + (f 17) + (f 18) + (f 19) + (f 20) + 
  (f 21) + (f 22) + (f 23) + (f 24) + (f 25) + (f 26) + (f 27) + (f 28) + (f 29) + (f 30) +
  -- continue this for all terms up to 2013
  sorry -- sum up to f(2013)
= 810 := by
  sorry

end periodic_f_sum_l242_242690


namespace log3_of_9_to_3_l242_242813

theorem log3_of_9_to_3 : Real.logb 3 (9^3) = 6 :=
by
  have h1 : 9 = 3^2 := by norm_num
  have h2 : 9^3 = (3^2)^3 := by rw [h1]
  have h3 : (3^2)^3 = 3^(2 * 3) := by rw [pow_mul]
  have h4 : Real.logb 3 (3^(2 * 3)) = 2 * 3 := Real.logb_pow 3 (2 * 3)
  rw [h4]
  norm_num

end log3_of_9_to_3_l242_242813


namespace length_of_angle_bisector_l242_242148

theorem length_of_angle_bisector (A B C D : Type) 
  (distAB : ℝ) (distAC : ℝ) (cosA : ℝ)
  (h_distAB : distAB = 4)
  (h_distAC : distAC = 5)
  (h_cosA : cosA = 1 / 10) : 
  Exists (lambda (lenAD : ℝ), -- Existence of length of angle bisector AD
    lenAD^2 = ((4:ℝ)^2 + (((4 * real.sqrt 37) / 9)^2) - 2 * 4 * ((4 * real.sqrt 37) / 9) * real.cos (arc_cos 1/10))) := 
by
  sorry

end length_of_angle_bisector_l242_242148


namespace measure_angle_I_l242_242136

-- Defining the angles in the pentagon
def angle_F : ℝ := y
def angle_G : ℝ := y
def angle_H : ℝ := y
def angle_I : ℝ := y + 80
def angle_J : ℝ := y + 80

-- Total sum of interior angles in a pentagon
def sum_angles_pentagon : ℝ := 540

theorem measure_angle_I (y : ℝ) :
  angle_F + angle_G + angle_H + angle_I + angle_J = sum_angles_pentagon → angle_I = 156 :=
by
  unfold angle_F angle_G angle_H angle_I angle_J sum_angles_pentagon
  sorry

end measure_angle_I_l242_242136


namespace gcd_8_10_l242_242279

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem gcd_8_10 : Nat.gcd (factorial 8) (factorial 10) = factorial 8 := 
by
  sorry

end gcd_8_10_l242_242279


namespace log_base_3_of_9_cubed_l242_242930
-- Import the necessary parts of the standard library

-- Define the statement to be proved
theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 :=
by
  sorry

end log_base_3_of_9_cubed_l242_242930


namespace log_base_3_of_9_cubed_l242_242895

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 :=
by assumptions
  have h1 : 9 = 3^2 := sorry,
  have h2 : 9^3 = (3^2)^3 := sorry,
  have h3 : (3^2)^3 = 3^(2*3) := sorry,
  have h4 : 3^(2*3) = 3^6 := sorry,
  have h5 : log 3 (3^6) = 6 := by apply log_pow,
  sorry

end log_base_3_of_9_cubed_l242_242895


namespace g_neg2_l242_242031

noncomputable def f : ℝ → ℝ := sorry

noncomputable def g (x : ℝ) : ℝ := f(x) + 1

axiom even_function (x : ℝ) : f(-x) + -x = f(x) + x

axiom f_2_value : f(2) = real.log 32 / real.log 10 + real.log 16 / real.log 4 + 6 * real.log (1 / 2) / real.log 10 + real.log (1 / 5) / real.log 10

theorem g_neg2 : g(-2) = 6 := by
  sorry

end g_neg2_l242_242031


namespace exists_subsum_divisible_by_n_l242_242160

theorem exists_subsum_divisible_by_n (n : ℕ) (a : ℕ → ℤ) (h_len : ∀ i, 1 ≤ i ∧ i ≤ n) :
  ∃ k r, (k < n) ∧ (r < n) ∧ (n ∣ ∑ i in finset.range (r + 1), a (k + i)) :=
by sorry

end exists_subsum_divisible_by_n_l242_242160


namespace prob_sum_eq_101_l242_242557

def sequence_a : List ℝ :=
  let rec a (k : ℕ) : ℝ :=
  match k with
  | 1 => 0.303
  | _ =>
    let prev := a (k - 1)
    let base := if k % 2 = 0 then
                  let female_k := "0.303" ++ "01".repeat (k/2 - 1) ++ "011" -- even
                  (String.toList female_k).map fun x => (x.toNat - '0'.toNat : ℝ)
                else
                  let male_k:= "0.303" ++ "01".repeat ((k - 1)/2) -- odd
                  (String.toList male_k).map fun x => (x.toNat - '0'.toNat : ℝ)
    base^.a (k - 1)
  List.range 102).map a

def sequence_b : List ℝ :=
  (sequence_a.greatest)

def same_index_eq_sum : ℕ := 
  let {k : ℕ | 1 ≤ k ≤ 101 ∧ sequence_a[n] = sequence_b[n]} 
  k.sum

theorem prob_sum_eq_101 : same_index_eq_sum = 101 :=
sorry

end prob_sum_eq_101_l242_242557


namespace log_base_three_of_nine_cubed_l242_242885

theorem log_base_three_of_nine_cubed : log 3 (9^3) = 6 :=
by
  have h1 : 9 = 3^2 := by sorry
  have h2 : ∀ (a m n: ℕ), (a^m)^n = a^(m * n) := by 
    intros a m n
    exact pow_mul a m n
  have h3 : ∀ (b n: ℕ), log b (b^n) = n := by sorry
  sorry

end log_base_three_of_nine_cubed_l242_242885


namespace alvin_age_l242_242722

theorem alvin_age (A S : ℕ) (h_s : S = 10) (h_cond : S = 1/2 * A - 5) : A = 30 := by
  sorry

end alvin_age_l242_242722


namespace op_eq_example_l242_242125

def op (x y : ℕ) : ℕ := 2 * x * y

theorem op_eq (x y z : ℕ) (h : z = op x y) : op 7 z = 560 := by
  sorry

theorem example : op 7 (op 4 5) = 560 := by
  rw [op, op]
  show 2 * 7 * (2 * 4 * 5) = 560
  sorry

end op_eq_example_l242_242125


namespace binomial_30_3_l242_242371

theorem binomial_30_3 : nat.choose 30 3 = 4060 := by
  sorry

end binomial_30_3_l242_242371


namespace speed_of_ship_with_two_sails_l242_242731

noncomputable def nautical_mile : ℝ := 1.15
noncomputable def land_miles_traveled : ℝ := 345
noncomputable def time_with_one_sail : ℝ := 4
noncomputable def time_with_two_sails : ℝ := 4
noncomputable def speed_with_one_sail : ℝ := 25

theorem speed_of_ship_with_two_sails :
  ∃ S : ℝ, 
    (S * time_with_two_sails + speed_with_one_sail * time_with_one_sail = land_miles_traveled / nautical_mile) → 
    S = 50  :=
by
  sorry

end speed_of_ship_with_two_sails_l242_242731


namespace log_three_pow_nine_pow_three_eq_six_l242_242947

open Real

noncomputable def eval_log_three_pow_nine_pow_three : ℝ :=
  log 3 (9^3)

theorem log_three_pow_nine_pow_three_eq_six :
  eval_log_three_pow_nine_pow_three = 6 := by
  sorry

end log_three_pow_nine_pow_three_eq_six_l242_242947


namespace longest_third_side_of_triangle_l242_242123

theorem longest_third_side_of_triangle {a b : ℕ} (ha : a = 8) (hb : b = 9) : 
  ∃ c : ℕ, 1 < c ∧ c < 17 ∧ ∀ (d : ℕ), (1 < d ∧ d < 17) → d ≤ c :=
by
  sorry

end longest_third_side_of_triangle_l242_242123


namespace totalNominalIncomeIsCorrect_l242_242583

def nominalIncomeForMonth (principal rate divisor months : ℝ) : ℝ :=
  principal * ((1 + rate / divisor) ^ months - 1)

def totalNominalIncomeForSixMonths : ℝ :=
  nominalIncomeForMonth 8700 0.06 12 6 +
  nominalIncomeForMonth 8700 0.06 12 5 +
  nominalIncomeForMonth 8700 0.06 12 4 +
  nominalIncomeForMonth 8700 0.06 12 3 +
  nominalIncomeForMonth 8700 0.06 12 2 +
  nominalIncomeForMonth 8700 0.06 12 1

theorem totalNominalIncomeIsCorrect : totalNominalIncomeForSixMonths = 921.15 := by
  sorry

end totalNominalIncomeIsCorrect_l242_242583


namespace log_pow_evaluation_l242_242912

-- Define the condition that 9 is equal to 3^2
def nine_eq_three_squared : Prop := 9 = 3^2

-- The main theorem that we need to prove
theorem log_pow_evaluation (h : nine_eq_three_squared) : log 3 (9^3) = 6 :=
  sorry

end log_pow_evaluation_l242_242912


namespace log_base_3_l242_242854

theorem log_base_3 (log_identity : ∀ (b a c : ℝ), log b (a^c) = c * log b a) :
  log 3 (9^3) = 6 :=
by
  have : 9 = 3^2 := by sorry
  have : log 3 3 = 1 := by sorry
  exact (log_identity 3 (3^2) 3).trans sorry

end log_base_3_l242_242854


namespace proof1_proof2_l242_242459

def f (x : ℝ) : ℝ :=
  if x < 0 then 1 / (1 - x) else (1 / 3) ^ x

theorem proof1 : f 1 + f (-1) = 5 / 6 :=
  sorry

theorem proof2 : {x : ℝ | f x ≥ 1 / 3} = {x : ℝ | -2 ≤ x ∧ x ≤ 1} :=
  sorry

end proof1_proof2_l242_242459


namespace log_base_3_of_9_cubed_l242_242933
-- Import the necessary parts of the standard library

-- Define the statement to be proved
theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 :=
by
  sorry

end log_base_3_of_9_cubed_l242_242933


namespace log_base_3_of_9_cubed_l242_242922
-- Import the necessary parts of the standard library

-- Define the statement to be proved
theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 :=
by
  sorry

end log_base_3_of_9_cubed_l242_242922


namespace canadian_math_olympiad_1992_l242_242033

theorem canadian_math_olympiad_1992
    (n : ℤ) (a : ℕ → ℤ) (k : ℕ)
    (h1 : n ≥ a 1) 
    (h2 : ∀ i, 1 ≤ i → i ≤ k → a i > 0)
    (h3 : ∀ i j, 1 ≤ i → i ≤ k → 1 ≤ j → j ≤ k → n ≥ Int.lcm (a i) (a j))
    (h4 : ∀ i, 1 ≤ i → i < k → a i > a (i + 1)) :
  ∀ i, 1 ≤ i → i ≤ k → i * a i ≤ n :=
sorry

end canadian_math_olympiad_1992_l242_242033


namespace trigonometric_identity_proof_l242_242399

theorem trigonometric_identity_proof :
  ( (Real.cos (40 * Real.pi / 180) + Real.sin (50 * Real.pi / 180) * (1 + Real.sqrt 3 * Real.tan (10 * Real.pi / 180)))
  / (Real.sin (70 * Real.pi / 180) * Real.sqrt (1 + Real.cos (40 * Real.pi / 180))) ) =
  Real.sqrt 2 :=
by
  sorry

end trigonometric_identity_proof_l242_242399


namespace circle_eq_standard_eq_l242_242447

-- Definition of the points A, B, C and the circle M in terms of conditions
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := -3, y := -1 }
def B : Point := { x := -6, y := 8 }
def C : Point := { x := 1, y := 1 }
def P : Point := { x := 2, y := 3 }

-- Hypothesis of the circle passing through given points
def circle_through_points (center : Point) (radius : ℝ) : Point → Prop :=
  λ p, (p.x - center.x)^2 + (p.y - center.y)^2 = radius^2

-- The circle M with center (-3,4) and radius 5
def M_center : Point := { x := -3, y := 4 }
def radius : ℝ := 5

-- Tangent lines problem
def tangent_line (center : Point) (radius : ℝ) (P : Point) : ℝ → Prop :=
  λ k, let L : Point → Prop := λ p, k * (p.x - P.x) = (p.y - P.y) in
    ∃ tangent : Point → Prop,
      (∀ p, tangent p → circle_through_points center radius p) ∧
      L P ∧
      (tangent P ∨ P = center)

-- Proof statements without proofs
theorem circle_eq_standard_eq : 
  (circle_through_points M_center radius A) ∧ 
  (circle_through_points M_center radius B) ∧ 
  (circle_through_points M_center radius C) → 
  ((∀ p : Point, circle_through_points M_center radius p → (p.x + 3)^2 + (p.y - 4)^2 = 25) ∧ 
  (∀ k, tangent_line M_center radius P k → 
    ((k = 2) ∨ (12 * P.x - 5 * P.y - 9 = 0))) :=
sorry

end circle_eq_standard_eq_l242_242447


namespace thirtieth_triangular_number_sum_thirtieth_thirtyfirst_triangular_numbers_l242_242648

def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem thirtieth_triangular_number :
  triangular_number 30 = 465 :=
by
  sorry

theorem sum_thirtieth_thirtyfirst_triangular_numbers :
  triangular_number 30 + triangular_number 31 = 961 :=
by
  sorry

end thirtieth_triangular_number_sum_thirtieth_thirtyfirst_triangular_numbers_l242_242648


namespace cos_diff_eq_sin_prod_angle_double_range_m_l242_242312

-- Part 1: Proving \(\cos 2x - \cos 2y = -2\sin(x+y)\sin(x-y)\)
theorem cos_diff_eq_sin_prod (x y : ℝ) : 
  cos (2 * x) - cos (2 * y) = -2 * sin (x + y) * sin (x - y) := 
sorry

-- Part 2(i): Proving \(A = 2B\) given \(a\sin A = (b+c)\sin B\)
theorem angle_double (a b c A B : ℝ) (h : a * sin A = (b + c) * sin B) : 
  A = 2 * B :=
sorry

-- Part 2(ii): Proving \(m \in [-3, \frac{3}{2}]\) given \(A = 2B\) and \((b-c)(m+2\cos^2B) \leq 2b\)
theorem range_m (b c A B m : ℝ) (h1 : A = 2 * B) (h2 : (b - c) * (m + 2 * cos B ^ 2) ≤ 2 * b) : 
  -3 ≤ m ∧ m ≤ 3 / 2 :=
sorry

end cos_diff_eq_sin_prod_angle_double_range_m_l242_242312


namespace evaluate_log_l242_242787

noncomputable def log_base (b x : ℝ) : ℝ := real.log x / real.log b

theorem evaluate_log : log_base 3 (9^3) = 6 := by
  have h1 : 9 = 3^2 := by norm_num
  have h2 : 9^3 = (3^2)^3 := by norm_num
  rw [h2, pow_mul]
  have h3 : log_base 3 (3^6) = 6 := by
    rw [← real.log_pow, real.log_mul, mul_comm, ← real.log_rpow, mul_div, real.log_self]
    norm_num
  exact h3

end evaluate_log_l242_242787


namespace constant_function_inequality_l242_242969

theorem constant_function_inequality (f : ℝ → ℝ) :
  (∀ (x y z : ℝ), f(x + y) + f(y + z) + f(z + x) ≥ 3 * f(x + 2 * y + 3 * z)) →
  ∃ c : ℝ, ∀ t : ℝ, f(t) = c :=
by 
  sorry

end constant_function_inequality_l242_242969


namespace log_base_3_l242_242846

theorem log_base_3 (log_identity : ∀ (b a c : ℝ), log b (a^c) = c * log b a) :
  log 3 (9^3) = 6 :=
by
  have : 9 = 3^2 := by sorry
  have : log 3 3 = 1 := by sorry
  exact (log_identity 3 (3^2) 3).trans sorry

end log_base_3_l242_242846


namespace sum_of_coefficients_l242_242183

noncomputable def integral_result : ℝ := 6 * (Real.sin (π / 2) - Real.sin 0)

noncomputable def f (x a : ℝ) : ℝ := (x - a) ^ integral_result

theorem sum_of_coefficients (a : ℝ) (h : f'.eval 0 / f.eval 0 = -3) :
  ∑ i in Finset.range (integral_result + 1), f.coeff i = 1 := by
  sorry

end sum_of_coefficients_l242_242183


namespace log_base_3_of_9_cubed_l242_242783

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 :=
by
  -- Given conditions:
  -- 1. 9 = 3^2
  -- 2. log_b (a^n) = n * log_b a
  sorry

end log_base_3_of_9_cubed_l242_242783


namespace solve_for_x_l242_242705

theorem solve_for_x : ∃ x : ℝ, (6 * x) / 1.5 = 3.8 ∧ x = 0.95 := by
  use 0.95
  exact ⟨by norm_num, by norm_num⟩

end solve_for_x_l242_242705


namespace log_base_3_of_9_cubed_l242_242839

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 := by
  -- Define the known conditions
  have h1 : 9 = 3^2 := by norm_num
  have h2 : log 3 (a^n) = n * log 3 a := sorry  -- Statement of the logarithmic identity
  
  -- Apply the conditions to prove the result
  calc
    log 3 (9^3)
        = log 3 ((3^2)^3) : by rw h1
    ... = log 3 (3^(2 * 3)) : by congr; norm_num
    ... = log 3 (3^6) : by norm_num
    ... = 6 : by rw [h2]; norm_num

end log_base_3_of_9_cubed_l242_242839


namespace unique_angles_sum_l242_242522

theorem unique_angles_sum (a1 a2 a3 a4 e4 e5 e6 e7 : ℝ) 
  (h_abcd: a1 + a2 + a3 + a4 = 360) 
  (h_efgh: e4 + e5 + e6 + e7 = 360) 
  (h_shared: a4 = e4) : 
  a1 + a2 + a3 + e4 + e5 + e6 + e7 - a4 = 360 := 
by 
  sorry

end unique_angles_sum_l242_242522


namespace log3_of_9_to_3_l242_242801

theorem log3_of_9_to_3 : Real.logb 3 (9^3) = 6 :=
by
  have h1 : 9 = 3^2 := by norm_num
  have h2 : 9^3 = (3^2)^3 := by rw [h1]
  have h3 : (3^2)^3 = 3^(2 * 3) := by rw [pow_mul]
  have h4 : Real.logb 3 (3^(2 * 3)) = 2 * 3 := Real.logb_pow 3 (2 * 3)
  rw [h4]
  norm_num

end log3_of_9_to_3_l242_242801


namespace no_good_subset_of_888_cells_l242_242995

   theorem no_good_subset_of_888_cells (table : Fin 32 → Fin 32 → Prop)
   (mouse_start : table (Fin.ofNat 31) (Fin.ofNat 0))
   (good_subset : Set (Fin 32 × Fin 32))
   (cheese_cells : good_subset.card = 888)
   (mouse_moves : ∀ p ∈ good_subset, mouse_turns_right p)
   (mouse_falls_off : ∃ p ∈ good_subset, next_position p = off_table) :
   false := 
   sorry
   
end no_good_subset_of_888_cells_l242_242995


namespace mutually_exclusive_events_l242_242987

theorem mutually_exclusive_events {qualified defective : ℕ} (hq : 2 < qualified) (hd : 2 < defective) :
  ∀ (A B : Prop),
  (A = (∃ (x y : bool), x ≠ y ∧ (x = true → defective > 1) ∧ (y = true → qualified > 1))) ∧
  (B = (∀ (x y : bool), x = false ∧ y = false)) →
  (A → ¬ B) :=
by
  intros A B h
  sorry

end mutually_exclusive_events_l242_242987


namespace AnaWinsAnyK_l242_242352

-- Define a type for a block of letters
structure Block where
  letter : Char
  count : Nat

-- Define the word as a list of blocks
def Word := List Block

-- Ana's win condition
def AnaCanAlwaysWin (A : Word) : Prop :=
  ∃ b : Block, b.count = 1 ∧ b ∈ A

-- The main theorem
theorem AnaWinsAnyK (A : Word) : AnaCanAlwaysWin A ↔
  ∀ k : Nat, ∃ W : Word, (numRainbows W A = k) :=
sorry

-- Auxiliary function to count the number of rainbows
noncomputable def numRainbows (W : Word) (A : Word) : Nat :=
sorry

end AnaWinsAnyK_l242_242352


namespace cubic_sum_l242_242104

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 :=
by {
  sorry
}

end cubic_sum_l242_242104


namespace log_base_3_of_9_cubed_l242_242775

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 :=
by
  -- Given conditions:
  -- 1. 9 = 3^2
  -- 2. log_b (a^n) = n * log_b a
  sorry

end log_base_3_of_9_cubed_l242_242775


namespace constant_term_of_expansion_l242_242746

open BigOperators

noncomputable def binomialCoeff (n k : ℕ) : ℕ := Nat.choose n k

theorem constant_term_of_expansion :
  ∑ r in Finset.range (6 + 1), binomialCoeff 6 r * (2^r * (x : ℚ)^r) / (x^3 : ℚ) = 160 :=
by
  sorry

end constant_term_of_expansion_l242_242746


namespace largest_k_divisible_by_3_l242_242542

-- Definition of the product of the first 50 positive odd integers
def Q : ℕ := (List.range 100).filter (λ n => n % 2 = 1).map (λ n => n + 1).prod

-- Lean statement to prove the largest k such that 3^k divides Q is 26
theorem largest_k_divisible_by_3 : ∃ k : ℕ, (3^k ∣ Q) ∧ k = 26 := by
  sorry

end largest_k_divisible_by_3_l242_242542


namespace determine_omega_varphi_l242_242446

theorem determine_omega_varphi (ω : ℝ) (φ : ℝ) (h1 : |φ| < π / 2) (h2 : 2 * sin φ = 1) : 
  (ω = 2 ∧ φ = π / 6) ∨ (ω = 2 ∧ φ = -π / 6) :=
begin
  sorry
end

end determine_omega_varphi_l242_242446


namespace find_m_l242_242466

open Real

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem find_m :
  let a := (-sqrt 3, m)
  let b := (2, 1)
  (dot_product a b = 0) → m = 2 * sqrt 3 :=
by
  sorry

end find_m_l242_242466


namespace symmetric_sum_8_dice_l242_242217

theorem symmetric_sum_8_dice (S : ℕ) (h : S = 11) : 
  ∃ other_sum : ℕ, other_sum = 45 ∧ 
  (∀ x : ℕ, x >= 8 ∧ x <= 48 → P(x)= P(2 * 28 - x)) :=
by
  sorry

end symmetric_sum_8_dice_l242_242217


namespace log_evaluation_l242_242763

theorem log_evaluation : ∀ (a : ℕ), a = 3 → log 3 (9^3) = 6 :=
by
  intro a ha
  have h1 : 9 = 3^2 := by sorry
  have h2 : (9^3) = (3^2)^3 := by rw [h1]
  have h3 : (3^2)^3 = 3^6 := by sorry
  have h4 : log 3 (3^6) = 6 := by sorry
  rw [h2, h3, h4]
  exact ha

end log_evaluation_l242_242763


namespace area_of_triangles_ABC_and_ABD_l242_242142

noncomputable def calculate_area := 
  let AB := 15 
  let AC := 10 
  let BD := 8 
  let Area_ABC := (1 / 2) * AB * AC
  let Area_ABD := (1 / 2) * AB * BD
  let Total_Area := Area_ABC + Area_ABD
  Total_Area

theorem area_of_triangles_ABC_and_ABD : 
  calculate_area = 135 := 
by 
  -- Conditions 
  let AB : ℝ := 15
  let AC : ℝ := 10
  let BD : ℝ := 8

  -- Area of triangle ABC
  have Area_ABC : ℝ := (1 / 2) * AB * AC
  have h1 : Area_ABC = 75 := by norm_num

  -- Area of triangle ABD
  have Area_ABD : ℝ := (1 / 2) * AB * BD
  have h2 : Area_ABD = 60 := by norm_num

  -- Total area
  have Total_Area : ℝ := Area_ABC + Area_ABD
  have h3 : Total_Area = 135 := by norm_num

  -- Conclude
  exact h3

end area_of_triangles_ABC_and_ABD_l242_242142


namespace negative_column_exists_l242_242678

theorem negative_column_exists (matrix : Matrix (Fin 5) (Fin 5) ℤ)
  (h : ∀ i : Fin 5, (∏ j : Fin 5, matrix i j) < 0) :
  ∃ j : Fin 5, (∏ i : Fin 5, matrix i j) < 0 :=
sorry

end negative_column_exists_l242_242678


namespace factorization_correct_l242_242659

theorem factorization_correct (a x y : ℝ) : a * x - a * y = a * (x - y) := by sorry

end factorization_correct_l242_242659


namespace nina_investment_l242_242588

theorem nina_investment:
  ∀ (total_inheritance invest_first_rate total_yearly_interest first_investment : ℝ),
    total_inheritance = 12000 →
    invest_first_rate = 0.06 →
    total_yearly_interest = 860 →
    first_investment = 5000 →
    7000 * (total_yearly_interest - (invest_first_rate * first_investment)) / 7000 = 0.08 :=
by
  intros total_inheritance invest_first_rate total_yearly_interest first_investment
  intros h_total h_rate h_interest h_first
  rw [h_total, h_rate, h_interest, h_first]
  norm_num
  sorry

end nina_investment_l242_242588


namespace cubic_sum_l242_242086

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 := by
  sorry

end cubic_sum_l242_242086


namespace distinct_cyclic_quadrilaterals_perimeter_36_l242_242068

noncomputable def count_distinct_cyclic_quadrilaterals : Nat :=
  1026

theorem distinct_cyclic_quadrilaterals_perimeter_36 :
  (∃ (a b c d : ℕ), a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ a + b + c + d = 36 ∧ a < b + c + d) → count_distinct_cyclic_quadrilaterals = 1026 :=
by
  rintro ⟨a, b, c, d, hab, hbc, hcd, hsum, hlut⟩
  sorry

end distinct_cyclic_quadrilaterals_perimeter_36_l242_242068


namespace function_increasing_on_interval_l242_242747

noncomputable def f (x : ℝ) := Real.sin x - Real.sqrt 3 * Real.cos x

theorem function_increasing_on_interval :
∀ x ∈ Icc (-π) 0, 
  f' x > 0 -> x ∈ Icc (-π / 6) 0 := 
by
  sorry

end function_increasing_on_interval_l242_242747


namespace general_formula_arithmetic_sequence_sum_of_sequence_b_l242_242433

-- Definitions of arithmetic sequence {a_n} and geometric sequence conditions
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
 ∀ n : ℕ, a (n + 1) = a n + d

def sum_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
 ∀ n : ℕ, S n = n * (a 1 + a n) / 2

def geometric_sequence (a : ℕ → ℤ) : Prop :=
  a 3 ^ 2 = a 1 * a 7

def arithmetic_sum_S3 (S : ℕ → ℤ) : Prop :=
  S 3 = 9

def general_formula (a : ℕ → ℤ) : Prop :=
 ∀ n : ℕ, a n = n + 1

def sum_first_n_terms_b (b : ℕ → ℤ) (T : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, T n = (n-1) * 2^(n+1) + 2

-- The Lean theorem statements
theorem general_formula_arithmetic_sequence
  (a : ℕ → ℤ) (d : ℤ) (S : ℕ → ℤ)
  (h1 : arithmetic_sequence a d)
  (h2 : sum_first_n_terms a S)
  (h3 : geometric_sequence a)
  (h4 : arithmetic_sum_S3 S) :
  general_formula a :=
  sorry

theorem sum_of_sequence_b
  (a b : ℕ → ℤ) (T : ℕ → ℤ)
  (h1 : general_formula a)
  (h2 : ∀ n : ℕ, b n = (a n - 1) * 2^n)
  (h3 : sum_first_n_terms_b b T) :
  ∀ n : ℕ, T n = (n-1) * 2^(n+1) + 2 :=
  sorry

end general_formula_arithmetic_sequence_sum_of_sequence_b_l242_242433


namespace trig_identity_l242_242596

theorem trig_identity (x : ℝ) :
  sin x * cos x + (sin x) ^ 3 * cos x + (sin x) ^ 5 * (sec x) = tan x := 
by
  sorry

end trig_identity_l242_242596


namespace who_received_q_in_first_round_l242_242719

variables (p q r A B C : ℕ)
variables (n : ℕ)
variable (rounds : Fin n → Fin 3 × Fin 3 × Fin 3)
variable (total_marbles_A total_marbles_B total_marbles_C : ℕ)
variable (received_in_final_round : Fin 3) -- 0 for A, 1 for B, 2 for C

-- Assumptions
axiom p_lt_q_lt_r : 0 < p ∧ p < q ∧ q < r
axiom total_A : total_marbles_A = 20
axiom total_B : total_marbles_B = 10
axiom total_C : total_marbles_C = 9
axiom B_received_r_in_final : rounds (Fin.last n) = (1, ⟨r, sorry⟩, sorry)

theorem who_received_q_in_first_round : (rounds 0).1 = 0 :=
sorry

end who_received_q_in_first_round_l242_242719


namespace sum_of_divisors_divisible_by_24_l242_242176

theorem sum_of_divisors_divisible_by_24 (n : ℕ) (h : (n + 1) % 24 = 0) :
    ((Finset.filter (λ d, n % d = 0) (Finset.range (n + 1))).sum id) % 24 = 0 := 
by 
  sorry

end sum_of_divisors_divisible_by_24_l242_242176


namespace triangle_angle_l242_242150

theorem triangle_angle (A B C : ℝ) (h1 : A - C = B) (h2 : A + B + C = 180) : A = 90 :=
by
  sorry

end triangle_angle_l242_242150


namespace attendance_second_day_l242_242699

theorem attendance_second_day (total_attendance first_day_attendance second_day_attendance third_day_attendance : ℕ) 
  (h_total : total_attendance = 2700)
  (h_second_day : second_day_attendance = first_day_attendance / 2)
  (h_third_day : third_day_attendance = 3 * first_day_attendance) :
  second_day_attendance = 300 :=
by
  sorry

end attendance_second_day_l242_242699


namespace multiples_of_12_in_range_24_250_l242_242479

theorem multiples_of_12_in_range_24_250 : 
  let mults := list.filter (λ x, x % 12 = 0) (list.range' 24 (250 + 1))
  in mults.length = 19 := 
by
  sorry

end multiples_of_12_in_range_24_250_l242_242479


namespace find_m_l242_242327

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := 
  if 2 ≤ x ∧ x ≤ 4 then 1 - |x - 3| 
  else if 1 ≤ x ∧ x < 2 then (1 / m) * (1 - |2 * x - 3|) 
  else if 4 < x ∧ x ≤ 8 then m * (1 - |x / 2 - 3|) 
  else 0

theorem find_m (m : ℝ) (hm : 0 < m) :
  ((∀ x ∈ set.Ici 1, f (2 * x) m = m * f x m) ∧
  (∀ (x : ℝ), 2 ≤ x ∧ x ≤ 4 → f x m = 1 - |x - 3|) ∧
  (∃ x₁ x₂ x₃ : ℝ, 1 < x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧
  ∀ y₁ y₂ y₃, (f x₁ m = y₁ ∧ f x₂ m = y₂ ∧ f x₃ m = y₃) →
  collinear ({(x₁, y₁), (x₂, y₂), (x₃, y₃)}))) →
  m = 1 ∨ m = 2 := by
  sorry

end find_m_l242_242327


namespace fill_table_correct_l242_242966

def table := {f a e x k b c : ℤ}
def filled_table (t : table) : Prop :=
  t.f = -3 ∧ t.a = 7 ∧ t.e = 12 ∧ t.x = -7 ∧ t.k = -2 ∧ t.b = 2 ∧ t.c = 8

theorem fill_table_correct (t : table) :
  (t.f = -3) ∧ (t.a = 7) ∧ (t.e = 12) ∧ (t.x = -7) ∧ (t.k = -2) ∧ (t.b = 2) ∧ (t.c = 8) →
  (∀ p q r s t, p + q + r + s + t = 0 → 
  (t.f = -3 ∧ t.a = 7 ∧ t.e = 12 ∧ t.x = -7 ∧ t.k = -2 ∧ t.b = 2 ∧ t.c = 8)) :=
by
  intros h_cond sum_row_eq_zero
  sorry

end fill_table_correct_l242_242966


namespace incorrect_option_c_l242_242749

theorem incorrect_option_c (a b : ℝ) (h : b < a ∧ a < 0) :
  (1/2)^b ≥ (1/2)^a := 
sorry

end incorrect_option_c_l242_242749


namespace triangle_exterior_angle_ratio_l242_242610

theorem triangle_exterior_angle_ratio (a b c : ℝ) (h : a = 2) (h1 : b = 3) (h2 : c = 4) :
  let α := 2 * 20 := α 
  let β := 3 * 20 := β 
  let γ := 4 * 20 := γ 
  let eα := 180 - α := eα 
  let eβ := 180 - β := eβ 
  let eγ := 180 - γ := eγ 
  eα / 20 = 7 ∧ eβ / 20 = 6 ∧ eγ / 20 = 5 :=
sorry

end triangle_exterior_angle_ratio_l242_242610


namespace partition_ratio_l242_242554

theorem partition_ratio (n : ℕ) (h_even : even n) (A B : Finset ℕ)
  (hAB : (1 : Finset ℕ).range (n^2 + 1) = A ∪ B)
  (h_size : A.card = B.card)
  (h_sum : (∑ x in A, x) * 64 = (∑ x in B, x) * 39) : 206 ∣ n :=
sorry

end partition_ratio_l242_242554


namespace smallest_n_for_square_and_cube_l242_242652

theorem smallest_n_for_square_and_cube (n : ℕ) 
  (h1 : ∃ m : ℕ, 3 * n = m^2) 
  (h2 : ∃ k : ℕ, 5 * n = k^3) : 
  n = 675 :=
  sorry

end smallest_n_for_square_and_cube_l242_242652


namespace division_addition_example_l242_242358

theorem division_addition_example : 12 / (1 / 6) + 3 = 75 := by
  sorry

end division_addition_example_l242_242358


namespace a_2023_eq_neg2_l242_242491

def seq : ℕ → ℤ
| 0     := 2
| (n+1) := -|seq n + 5|

theorem a_2023_eq_neg2 : seq 2022 = -2 :=
  sorry

end a_2023_eq_neg2_l242_242491


namespace evaluate_log_l242_242791

noncomputable def log_base (b x : ℝ) : ℝ := real.log x / real.log b

theorem evaluate_log : log_base 3 (9^3) = 6 := by
  have h1 : 9 = 3^2 := by norm_num
  have h2 : 9^3 = (3^2)^3 := by norm_num
  rw [h2, pow_mul]
  have h3 : log_base 3 (3^6) = 6 := by
    rw [← real.log_pow, real.log_mul, mul_comm, ← real.log_rpow, mul_div, real.log_self]
    norm_num
  exact h3

end evaluate_log_l242_242791


namespace maximum_b_value_l242_242464

noncomputable def f (x : ℝ) := Real.exp x - x - 1
def g (x : ℝ) := -x^2 + 4 * x - 3

theorem maximum_b_value (a b : ℝ) (h1 : 0 ≤ a) (h2 : f a = g b) : b ≤ 3 := by
  sorry

end maximum_b_value_l242_242464


namespace length_of_bridge_l242_242339

-- Definitions of the given conditions
def train_length : ℕ := 385  -- meters
def train_speed : ℕ := 45    -- km/hour
def time_to_pass : ℕ := 42   -- seconds

-- The required proof statement
theorem length_of_bridge (train_length = 385) (train_speed = 45) (time_to_pass = 42) : 
  140 :=
  sorry

end length_of_bridge_l242_242339


namespace lowest_temperature_at_noon_l242_242612

theorem lowest_temperature_at_noon
  (L : ℤ) -- Denote lowest temperature as L
  (avg_temp : ℤ) -- Average temperature from Monday to Friday
  (max_range : ℤ) -- Maximum possible range of the temperature
  (h1 : avg_temp = 50) -- Condition 1: average temperature is 50
  (h2 : max_range = 50) -- Condition 2: maximum range is 50
  (total_temp : ℤ) -- Sum of temperatures from Monday to Friday
  (h3 : total_temp = 250) -- Sum of temperatures equals 5 * 50
  (h4 : total_temp = L + (L + 50) + (L + 50) + (L + 50) + (L + 50)) -- Sum represented in terms of L
  : L = 10 := -- Prove that L equals 10
sorry

end lowest_temperature_at_noon_l242_242612


namespace sum_even_integers_200_to_400_l242_242281

theorem sum_even_integers_200_to_400 : 
  let seq := list.range' 200 ((400 - 200) / 2 + 1)
  in seq.filter (λ n, n % 2 = 0) = list.range' 200 101 ∧ 
     seq.sum = 30300 := 
by
  sorry

end sum_even_integers_200_to_400_l242_242281


namespace problem_proof_l242_242016

def f : ℕ → ℕ
| 1 := 1
| 2 := 3
| 3 := 1
| _ := 0  -- Default case if table values are exhausted

def g : ℕ → ℕ
| 1 := 3
| 2 := 2
| 3 := 1
| _ := 0  -- Default case if table values are exhausted

theorem problem_proof :
  f (g 1) = 1 ∧ { x | f (g x) > g (f x) } = {2} := 
by
  sorry

end problem_proof_l242_242016


namespace log_base_3_of_9_cubed_l242_242894

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 :=
by assumptions
  have h1 : 9 = 3^2 := sorry,
  have h2 : 9^3 = (3^2)^3 := sorry,
  have h3 : (3^2)^3 = 3^(2*3) := sorry,
  have h4 : 3^(2*3) = 3^6 := sorry,
  have h5 : log 3 (3^6) = 6 := by apply log_pow,
  sorry

end log_base_3_of_9_cubed_l242_242894


namespace flight_relation_not_preserved_l242_242511

noncomputable def swap_city_flights (cities : Finset ℕ) (flights : ℕ → ℕ → Bool) : Prop := sorry

theorem flight_relation_not_preserved (cities : Finset ℕ) (flights : ℕ → ℕ → Bool) (M N : ℕ) (hM : M ∈ cities) (hN : N ∈ cities) : 
  ¬ swap_city_flights cities flights :=
sorry

end flight_relation_not_preserved_l242_242511


namespace power_function_monotonically_increasing_range_condition_l242_242467

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m-1)^2 * x^(m^2 - 4*m + 2)
noncomputable def g (k : ℝ) (x : ℝ) : ℝ := 2^x - k

theorem power_function_monotonically_increasing (m : ℝ) :
  (∀ x : ℝ, 0 < x → f m x > 0 → m = 0) :=
sorry

theorem range_condition (k : ℝ) : 
  (∀ x : ℝ, x ∈ Icc (1 : ℝ) 2 → (2 - k ≥ 1) ∧ (4 - k ≤ 4) → 0 ≤ k ∧ k ≤ 1) :=
sorry

end power_function_monotonically_increasing_range_condition_l242_242467


namespace log_base_3_of_9_cubed_l242_242828

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 := by
  sorry

end log_base_3_of_9_cubed_l242_242828


namespace range_of_x_l242_242054

def f (x : ℝ) : ℝ := 3^(1 + |x|) - 1 / (1 + x^2)

theorem range_of_x (x : ℝ) : f(x) < f(2 * x + 1) ↔ x < -1 ∨ -1/3 < x := sorry

end range_of_x_l242_242054


namespace log_base_3_of_9_cubed_l242_242871

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 := by
  -- Assuming the necessary properties of logarithms and exponents
  sorry

end log_base_3_of_9_cubed_l242_242871


namespace tangent_line_equation_at_0_l242_242457

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x^2 - x + Real.sin x

theorem tangent_line_equation_at_0 : 
  let y := f 0
  let f' := fun x => Real.exp x + 2 * x - 1 + Real.cos x
  let slope := f' 0
  (∀ x : ℝ, y = x + 1)
sorry

end tangent_line_equation_at_0_l242_242457


namespace cosine_of_angle_between_vectors_l242_242064

theorem cosine_of_angle_between_vectors :
  let a := (1:ℝ, 1:ℝ)
  let b := (-2:ℝ, 1:ℝ)
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_a := Real.sqrt (a.1 * a.1 + a.2 * a.2)
  let magnitude_b := Real.sqrt (b.1 * b.1 + b.2 * b.2)
  cos_angle := dot_product / (magnitude_a * magnitude_b)
  cos_angle = (-Real.sqrt 10 / 10) := 
by
  let a := (1:ℝ, 1:ℝ)
  let b := (-2:ℝ, 1:ℝ)
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_a := Real.sqrt (a.1 * a.1 + a.2 * a.2)
  let magnitude_b := Real.sqrt (b.1 * b.1 + b.2 * b.2)
  let cos_angle := dot_product / (magnitude_a * magnitude_b)
  sorry

end cosine_of_angle_between_vectors_l242_242064


namespace investment_plans_count_l242_242326

variable (Projects Cities : Type) [Fintype Projects] [Fintype Cities]
variable {investment_plan : Projects → Cities}

theorem investment_plans_count 
  [fintype Projects] [fintype Cities]
  (h_projects : Fintype.card Projects = 3)
  (h_cities : Fintype.card Cities = 4)
  (h_at_most_two_projects_per_city : ∀ c : Cities, (finset.univ.filter (λ p, investment_plan p = c)).card ≤ 2) :
  fintype.card {investment_plan // 
    ∀ c : Cities, (finset.univ.filter (λ p, investment_plan p = c)).card ≤ 2} = 60 := 
  sorry

end investment_plans_count_l242_242326


namespace infinite_sum_equals_one_fourth_l242_242732

theorem infinite_sum_equals_one_fourth :
  ∑' n : ℕ, (3^n / (1 + 3^n + 3^(n + 1) + 3^(2 * n + 1))) = 1 / 4 :=
sorry

end infinite_sum_equals_one_fourth_l242_242732


namespace sum_even_integers_200_to_400_l242_242283

theorem sum_even_integers_200_to_400 : 
  let seq := list.range' 200 ((400 - 200) / 2 + 1)
  in seq.filter (λ n, n % 2 = 0) = list.range' 200 101 ∧ 
     seq.sum = 30300 := 
by
  sorry

end sum_even_integers_200_to_400_l242_242283


namespace find_m_n_find_cos_angle_AOC_l242_242474

variables (m n : ℝ)

def vec_A := (-3, m + 1)
def vec_B := (n, 3)
def vec_C := (7, 4)

-- Proving Orthogonality 
def orthogonality_condition := (-3, m + 1) ⬝ (n, 3) = 0

-- Defining vectors
def OA := vec_A
def OB := vec_B
def OC := vec_C

-- Proof of colinearity 
def colinearity_condition := ∃ (k : ℝ), (7 - n, 1) = k • (n + 3, 2 - m)

-- Definition of centroid G
def G := (OA.1 + OC.1) / 2, (OA.2 + OC.2) / 2

-- Given the centroid condition
def centroid_condition := (vec _ / 2 , vec _/ 2 ) = 2 / 3 • (vec_B /2 ) 

-- Conclusion 1
theorem find_m_n : orthogonality_condition → colinearity_condition → 
  (m = 1 ∧ n = 2) ∨ (m = 8 ∧ n = 9) := 
sorry

-- Conclusion 2
theorem find_cos_angle_AOC 
  (H : orthogonality_condition) 
  (G_eq : centroid_condition)
  (m_eq: m = 1) (n_eq: n = 2) :
  let OA := (-3, 2)
  let OC := (7, 4) in
  (OA ⬝ OC) / (|OA| * |OC|) = -sqrt(5)/5 :=
sorry

end find_m_n_find_cos_angle_AOC_l242_242474


namespace x_cubed_plus_y_cubed_l242_242081

variable (x y : ℝ)
variable (h₁ : x + y = 5)
variable (h₂ : x^2 + y^2 = 17)

theorem x_cubed_plus_y_cubed :
  x^3 + y^3 = 65 :=
by sorry

end x_cubed_plus_y_cubed_l242_242081


namespace log3_of_9_to_3_l242_242811

theorem log3_of_9_to_3 : Real.logb 3 (9^3) = 6 :=
by
  have h1 : 9 = 3^2 := by norm_num
  have h2 : 9^3 = (3^2)^3 := by rw [h1]
  have h3 : (3^2)^3 = 3^(2 * 3) := by rw [pow_mul]
  have h4 : Real.logb 3 (3^(2 * 3)) = 2 * 3 := Real.logb_pow 3 (2 * 3)
  rw [h4]
  norm_num

end log3_of_9_to_3_l242_242811


namespace log_base_3_of_9_cubed_l242_242889

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 :=
by assumptions
  have h1 : 9 = 3^2 := sorry,
  have h2 : 9^3 = (3^2)^3 := sorry,
  have h3 : (3^2)^3 = 3^(2*3) := sorry,
  have h4 : 3^(2*3) = 3^6 := sorry,
  have h5 : log 3 (3^6) = 6 := by apply log_pow,
  sorry

end log_base_3_of_9_cubed_l242_242889


namespace sum_of_possible_x_eq_27_l242_242009

noncomputable def data_points (x : ℝ) : list ℝ := [10, 8, 8, 11, 16, 8, x]

def mean (d : list ℝ) : ℝ := (d.sum) / d.length

def median (d : list ℝ) : ℝ :=
  let sorted := d.qsort (· ≤ ·)
  sorted.nth (sorted.length / 2) |>.get

def mode (d : list ℝ) : ℝ :=
  d.foldl (λ (acc : ℕ × ℝ) (val : ℝ) =>
    let count := d.count (λ y, y = val)
    if count > acc.fst then (count, val) else acc) (0, 0.0) |>.snd

def is_arithmetic_sequence (a b c : ℝ) : Prop :=
  2 * b = a + c

theorem sum_of_possible_x_eq_27 : 
  let xs := [10, 8, 8, 11, 16, 8] in
  let valid_x (x : ℝ) := 
    let d := data_points x in
    is_arithmetic_sequence (mean d) (median d) (mode d) in
  (list.filter valid_x ([-5, 9, 23])).sum = 27 :=
by 
  let xs := [10, 8, 8, 11, 16, 8]
  let valid_x (x : ℝ) := 
    let d := data_points x
    is_arithmetic_sequence (mean d) (median d) (mode d)
  have : valid_x (-5) := sorry
  have : valid_x (9) := sorry
  have : valid_x (23) := sorry
  show ([-5, 9, 23]).sum = 27 from sorry

end sum_of_possible_x_eq_27_l242_242009


namespace gcd_8_10_l242_242277

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem gcd_8_10 : Nat.gcd (factorial 8) (factorial 10) = factorial 8 := 
by
  sorry

end gcd_8_10_l242_242277


namespace cos_alpha_l242_242032

-- Definitions
variable (α : ℝ) (h1 : 0 < α ∧ α < π) (h2 : cos (α - π / 6) = 3 / 5)

-- Theorem Statement
theorem cos_alpha (h1 : 0 < α ∧ α < π) (h2 : cos (α - π / 6) = 3 / 5) :
  cos α = (3 * Real.sqrt 3 - 4) / 10 :=
sorry

end cos_alpha_l242_242032


namespace total_weight_correct_l242_242688

-- Definitions of the given weights of materials
def weight_concrete : ℝ := 0.17
def weight_bricks : ℝ := 0.237
def weight_sand : ℝ := 0.646
def weight_stone : ℝ := 0.5
def weight_steel : ℝ := 1.73
def weight_wood : ℝ := 0.894

-- Total weight of all materials
def total_weight : ℝ := 
  weight_concrete + weight_bricks + weight_sand + weight_stone + weight_steel + weight_wood

-- The proof statement
theorem total_weight_correct : total_weight = 4.177 := by
  sorry

end total_weight_correct_l242_242688


namespace sum_of_roots_l242_242007

theorem sum_of_roots (x₁ x₂ : ℝ) 
  (h₁ : x₁^2 - 2 * x₁ - 8 = 0) 
  (h₂ : x₂^2 - 2 * x₂ - 8 = 0)
  (h_distinct : x₁ ≠ x₂) : 
  x₁ + x₂ = 2 := 
sorry

end sum_of_roots_l242_242007


namespace festival_second_day_attendance_l242_242700

-- Define the conditions
variables (X Y Z A : ℝ)
variables (h1 : A = 2700) (h2 : Y = X / 2) (h3 : Z = 3 * X) (h4 : A = X + Y + Z)

-- Theorem stating the question and the conditions result in the correct answer
theorem festival_second_day_attendance (X Y Z A : ℝ) 
  (h1 : A = 2700) (h2 : Y = X / 2) (h3 : Z = 3 * X) (h4 : A = X + Y + Z) : 
  Y = 300 :=
sorry

end festival_second_day_attendance_l242_242700


namespace find_x_l242_242419

theorem find_x (lg2 lg3 : ℝ) (h₁ : lg2 = 0.3010) (h₂ : lg3 = 0.4771) :
  ∃ x : ℝ, 3^(x + 3) = 135 ∧ x ≈ 1.47 :=
by
  sorry

end find_x_l242_242419


namespace number_of_articles_l242_242120

theorem number_of_articles (C S : ℝ) (N : ℝ) 
    (h1 : N * C = 40 * S) 
    (h2 : (S - C) / C * 100 = 49.999999999999986) : 
    N = 60 :=
sorry

end number_of_articles_l242_242120


namespace smallest_multiple_of_45_and_60_not_divisible_by_18_l242_242655

noncomputable def smallest_multiple_not_18 (n : ℕ) : Prop :=
  (n % 45 = 0) ∧
  (n % 60 = 0) ∧
  (n % 18 ≠ 0) ∧
  ∀ m : ℕ, (m % 45 = 0) ∧ (m % 60 = 0) ∧ (m % 18 ≠ 0) → n ≤ m

theorem smallest_multiple_of_45_and_60_not_divisible_by_18 : ∃ n : ℕ, smallest_multiple_not_18 n ∧ n = 810 := 
by
  existsi 810
  sorry

end smallest_multiple_of_45_and_60_not_divisible_by_18_l242_242655


namespace solve_g_eq_g_inv_l242_242389

noncomputable def g (x : ℝ) : ℝ := 4 * x - 5

noncomputable def g_inv (x : ℝ) : ℝ := (x + 5) / 4

theorem solve_g_eq_g_inv : 
  ∃ x : ℝ, g x = g_inv x ∧ x = 5 / 3 :=
by
  sorry

end solve_g_eq_g_inv_l242_242389


namespace rhombus_area_l242_242258

-- Given a rhombus with sides of 4 cm and an included angle of 45 degrees,
-- prove that the area is 8 square centimeters.

theorem rhombus_area :
  ∀ (s : ℝ) (α : ℝ), s = 4 ∧ α = π / 4 → 
    let area := s * s * Real.sin α in
    area = 8 := 
by
  intros s α h
  sorry

end rhombus_area_l242_242258


namespace log_base_3_of_9_cubed_l242_242890

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 :=
by assumptions
  have h1 : 9 = 3^2 := sorry,
  have h2 : 9^3 = (3^2)^3 := sorry,
  have h3 : (3^2)^3 = 3^(2*3) := sorry,
  have h4 : 3^(2*3) = 3^6 := sorry,
  have h5 : log 3 (3^6) = 6 := by apply log_pow,
  sorry

end log_base_3_of_9_cubed_l242_242890


namespace cost_of_each_item_number_of_purchasing_plans_l242_242210

-- Question 1: Cost of each item
theorem cost_of_each_item : 
  ∃ (x y : ℕ), 
    (10 * x + 5 * y = 2000) ∧ 
    (5 * x + 3 * y = 1050) ∧ 
    (x = 150) ∧ 
    (y = 100) :=
by
    sorry

-- Question 2: Number of different purchasing plans
theorem number_of_purchasing_plans : 
  (∀ (a b : ℕ), 
    (150 * a + 100 * b = 4000) → 
    (a ≥ 12) → 
    (b ≥ 12) → 
    (4 = 4)) :=
by
    sorry

end cost_of_each_item_number_of_purchasing_plans_l242_242210


namespace students_in_band_and_chorus_not_orchestra_l242_242633

def total_students : ℕ := 250
def band_students : ℕ := 80
def chorus_students : ℕ := 110
def orchestra_students : ℕ := 60
def total_involved_students : ℕ := 190

theorem students_in_band_and_chorus_not_orchestra : 
    let total_overlaps := (band_students + chorus_students + orchestra_students) - total_involved_students in
    let students_double_counted := total_overlaps in
    let students_just_orchestra := orchestra_students / 2 in
    let students_band_chorus_only := students_double_counted - students_just_orchestra in
    students_band_chorus_only = 30 :=
by
  sorry

end students_in_band_and_chorus_not_orchestra_l242_242633


namespace cube_identity_l242_242111

theorem cube_identity (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 := by
  sorry

end cube_identity_l242_242111


namespace instantaneous_velocity_at_t_eq_5_is_40_l242_242708

theorem instantaneous_velocity_at_t_eq_5_is_40 :
  (∀ t : ℝ, s t = 4 * t^2 - 3) →
  (∀ t : ℝ, instantaneous_velocity s t = derivative (s t) t) →
  instantaneous_velocity s 5 = 40 :=
by
  sorry

-- Definitions used
def s : ℝ → ℝ := λ t, 4 * t^2 - 3
def instantaneous_velocity (s : ℝ → ℝ) (t : ℝ) : ℝ :=
  deriv s t

end instantaneous_velocity_at_t_eq_5_is_40_l242_242708


namespace find_x_eq_5_over_3_l242_242391

def g (x : ℝ) : ℝ := 4 * x - 5

def g_inv (y : ℝ) : ℝ := (y + 5) / 4

theorem find_x_eq_5_over_3 (x : ℝ) (hx : g x = g_inv x) : x = 5 / 3 :=
by
  sorry

end find_x_eq_5_over_3_l242_242391


namespace totalNominalIncomeIsCorrect_l242_242580

def nominalIncomeForMonth (principal rate divisor months : ℝ) : ℝ :=
  principal * ((1 + rate / divisor) ^ months - 1)

def totalNominalIncomeForSixMonths : ℝ :=
  nominalIncomeForMonth 8700 0.06 12 6 +
  nominalIncomeForMonth 8700 0.06 12 5 +
  nominalIncomeForMonth 8700 0.06 12 4 +
  nominalIncomeForMonth 8700 0.06 12 3 +
  nominalIncomeForMonth 8700 0.06 12 2 +
  nominalIncomeForMonth 8700 0.06 12 1

theorem totalNominalIncomeIsCorrect : totalNominalIncomeForSixMonths = 921.15 := by
  sorry

end totalNominalIncomeIsCorrect_l242_242580


namespace evaluate_log_l242_242798

noncomputable def log_base (b x : ℝ) : ℝ := real.log x / real.log b

theorem evaluate_log : log_base 3 (9^3) = 6 := by
  have h1 : 9 = 3^2 := by norm_num
  have h2 : 9^3 = (3^2)^3 := by norm_num
  rw [h2, pow_mul]
  have h3 : log_base 3 (3^6) = 6 := by
    rw [← real.log_pow, real.log_mul, mul_comm, ← real.log_rpow, mul_div, real.log_self]
    norm_num
  exact h3

end evaluate_log_l242_242798


namespace log_three_pow_nine_pow_three_eq_six_l242_242943

open Real

noncomputable def eval_log_three_pow_nine_pow_three : ℝ :=
  log 3 (9^3)

theorem log_three_pow_nine_pow_three_eq_six :
  eval_log_three_pow_nine_pow_three = 6 := by
  sorry

end log_three_pow_nine_pow_three_eq_six_l242_242943


namespace find_a_find_n_l242_242432

noncomputable def arithmetic_sequence (a d n : ℕ) : ℕ := a + (n - 1) * d
noncomputable def sum_of_first_n_terms (a d n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2
noncomputable def S (a d n : ℕ) : ℕ := if n = 1 then a else sum_of_first_n_terms a d n
noncomputable def arithmetic_sum_property (a d n : ℕ) : Prop :=
  ∀ n ≥ 2, (S a d n) ^ 2 = 3 * n ^ 2 * arithmetic_sequence a d n + (S a d (n - 1)) ^ 2

theorem find_a (a : ℕ) (h1 : ∀ n ≥ 2, S a 3 n ^ 2 = 3 * n ^ 2 * arithmetic_sequence a 3 n + S a 3 (n - 1) ^ 2) :
  a = 3 :=
sorry

noncomputable def c (n : ℕ) (a5 : ℕ) : ℕ := 3 ^ (n - 1) + a5
noncomputable def sum_of_first_n_terms_c (n a5 : ℕ) : ℕ := (3^n - 1) / 2 + 15 * n
noncomputable def T (n a5 : ℕ) : ℕ := sum_of_first_n_terms_c n a5

theorem find_n (a : ℕ) (a5 : ℕ) (h1 : ∀ n ≥ 2, S a 3 n ^ 2 = 3 * n ^ 2 * arithmetic_sequence a 3 n + S a 3 (n - 1) ^ 2)
  (h2 : a = 3) (h3 : a5 = 15) :
  ∃ n : ℕ, 4 * T n a5 > S a 3 10 ∧ n = 3 :=
sorry

end find_a_find_n_l242_242432


namespace proof_intersection_l242_242469

def setA : Set ℤ := {x | abs x ≤ 2}

def setB : Set ℝ := {x | x^2 - 2 * x - 8 ≥ 0}

def complementB : Set ℝ := {x | x^2 - 2 * x - 8 < 0}

def intersectionAComplementB : Set ℤ := {x | x ∈ setA ∧ (x : ℝ) ∈ complementB}

theorem proof_intersection : intersectionAComplementB = {-1, 0, 1, 2} := by
  sorry

end proof_intersection_l242_242469


namespace proof_problem_l242_242046

noncomputable def problem1 :=
  let e := (Real.sqrt 3) / 3 in
  let a := Real.sqrt 3 in
  let c := 1 in
  let b := Real.sqrt (a^2 - c^2) in
  ∃ (x y : ℝ), (a > b ∧ b > 0 ∧ (x^2 / a^2 + y^2 / b^2 = 1) ∧ x = a ∧ y = 0 ∧ e = c / a)

noncomputable def problem2 :=
  let k : ℝ := sorry in
  ∃ (x1 x2 : ℝ),  
    (x1 + x2 = -6 * k / (3 * k^2 + 2) ∧ 
     x1 * x2 = -3 / (3 * k^2 + 2) ∧ 
     ∀ m : ℝ, |m| ≤ Real.sqrt 6 / 12 → x1^2 / 3 + ((k * x1 + 1)^2 / 2) = 1 ∧ 
     x2^2 / 3 + ((k * x2 + 1)^2 / 2) = 1)

theorem proof_problem : 
  problem1 ∧ problem2 :=
by
  sorry

end proof_problem_l242_242046


namespace log_base_3_of_9_cubed_l242_242864

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 := by
  -- Assuming the necessary properties of logarithms and exponents
  sorry

end log_base_3_of_9_cubed_l242_242864


namespace log_base_3_of_9_cubed_l242_242928
-- Import the necessary parts of the standard library

-- Define the statement to be proved
theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 :=
by
  sorry

end log_base_3_of_9_cubed_l242_242928


namespace flags_left_l242_242266

theorem flags_left (interval circumference : ℕ) (total_flags : ℕ) (h1 : interval = 20) (h2 : circumference = 200) (h3 : total_flags = 12) : 
  total_flags - (circumference / interval) = 2 := 
by 
  -- Using the conditions h1, h2, h3
  sorry

end flags_left_l242_242266


namespace rectangle_length_reduction_l242_242229

theorem rectangle_length_reduction:
  ∀ (L W : ℝ) (X : ℝ),
  W > 0 →
  L > 0 →
  (L * (1 - X / 100) * (4 / 3)) * W = L * W →
  X = 25 :=
by
  intros L W X hW hL hEq
  sorry

end rectangle_length_reduction_l242_242229


namespace log_evaluation_l242_242754

theorem log_evaluation : ∀ (a : ℕ), a = 3 → log 3 (9^3) = 6 :=
by
  intro a ha
  have h1 : 9 = 3^2 := by sorry
  have h2 : (9^3) = (3^2)^3 := by rw [h1]
  have h3 : (3^2)^3 = 3^6 := by sorry
  have h4 : log 3 (3^6) = 6 := by sorry
  rw [h2, h3, h4]
  exact ha

end log_evaluation_l242_242754


namespace man_average_interest_rate_l242_242702

noncomputable def average_rate_of_interest (total_investment : ℝ) (rate1 rate2 rate_average : ℝ) 
    (x : ℝ) (same_return : (rate1 * (total_investment - x) = rate2 * x)) : Prop :=
  (rate_average = ((rate1 * (total_investment - x) + rate2 * x) / total_investment))

theorem man_average_interest_rate
    (total_investment : ℝ) 
    (rate1 : ℝ)
    (rate2 : ℝ)
    (rate_average : ℝ)
    (x : ℝ)
    (same_return : rate1 * (total_investment - x) = rate2 * x) :
    total_investment = 4500 ∧ rate1 = 0.04 ∧ rate2 = 0.06 ∧ x = 1800 ∧ rate_average = 0.048 → 
    average_rate_of_interest total_investment rate1 rate2 rate_average x same_return := 
by
  sorry

end man_average_interest_rate_l242_242702


namespace distance_point_to_line_l242_242395

-- Define the points
def point1 := (2, 3, -1 : ℝ×ℝ×ℝ)
def point2 := (3, -1, 4 : ℝ×ℝ×ℝ)
def point3 := (5, 0, 1 : ℝ×ℝ×ℝ)

-- The distance from point1 to the line passing through point2 and point3 is sqrt(3667)/14.

theorem distance_point_to_line :
  let d := dist_point_to_line point1 point2 point3 in
  d = sqrt(3667) / 14 :=
sorry

-- Function to compute distance (this needs appropriate math definition but for now we acknowledge it)
noncomputable def dist_point_to_line (P₁ P₂ P₃ : ℝ×ℝ×ℝ) : ℝ :=
  sorry

end distance_point_to_line_l242_242395


namespace cubic_sum_l242_242105

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 :=
by {
  sorry
}

end cubic_sum_l242_242105


namespace theater_loss_l242_242703

theorem theater_loss 
  (capacity : ℕ) (ticket_price : ℕ) (tickets_sold : ℕ)
  (h_capacity : capacity = 50)
  (h_ticket_price : ticket_price = 8)
  (h_tickets_sold : tickets_sold = 24) : 
  (capacity * ticket_price) - (tickets_sold * ticket_price) = 208 := 
by 
  rw [h_capacity, h_ticket_price, h_tickets_sold]
  exact Nat.sub_mul_eq_sub_mul capacity tickets_sold ticket_price
  sorry -- Skip the proof for this exercise

end theater_loss_l242_242703


namespace log_base_3_of_9_cubed_l242_242781

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 :=
by
  -- Given conditions:
  -- 1. 9 = 3^2
  -- 2. log_b (a^n) = n * log_b a
  sorry

end log_base_3_of_9_cubed_l242_242781


namespace binomial_product_l242_242374

open Nat

theorem binomial_product : (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 := by
  sorry

end binomial_product_l242_242374


namespace cos_2alpha_value_l242_242506

def f (x α : ℝ) : ℝ := Real.sin (x + α - Real.pi / 12)

theorem cos_2alpha_value (α : ℝ) 
  (h : ∀ x : ℝ, f x α = f (-x) α) : Real.cos (2 * α) = -Real.sqrt 3 / 2 := 
by
  sorry

end cos_2alpha_value_l242_242506


namespace three_common_subsets_l242_242161

theorem three_common_subsets 
  (n : ℕ) (hn : n ≥ 2) 
  (X : Finset ℕ) (hX : X.card = n)
  (A : Fin 101 → Finset ℕ)
  (hA : ∀ S : Finset (Fin 101), S.card = 50 → (S.val.bind (λ i, (A i).val)).toFinset.card > (50 * n) / 51) :
  ∃ i j k : Fin 101, i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ (A i ∩ A j ≠ ∅) ∧ (A i ∩ A k ≠ ∅) ∧ (A j ∩ A k ≠ ∅) := 
sorry

end three_common_subsets_l242_242161


namespace simplify_f_find_value_of_f_l242_242026

variables (α : ℝ)

def f (α : ℝ) : ℝ :=
  (Real.sin (α - (Real.pi / 2)) * Real.cos ((Real.pi / 2) + α) * Real.tan (Real.pi - α)) /
  (Real.tan (-α - Real.pi) * Real.sin (-α - Real.pi))

theorem simplify_f (α : ℝ) (hα : α > Real.pi ∧ α < 3 * Real.pi / 2) : f α = Real.cos α :=
by sorry

theorem find_value_of_f (α : ℝ) (hα : α > Real.pi ∧ α < 3 * Real.pi / 2)
  (h1 : Real.cos (α - (Real.pi / 2)) = -1 / 4) : f α = -Real.sqrt 15 / 4 :=
by sorry

end simplify_f_find_value_of_f_l242_242026


namespace log_base_3_of_9_cubed_l242_242818

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 := by
  sorry

end log_base_3_of_9_cubed_l242_242818


namespace general_term_a_sum_Tn_l242_242449

open_locale big_operators

-- Definitions:
def a (n : ℕ) : ℕ := 2^(n-1)

def b (n : ℕ) : ℕ := (2 * n - 1) * a n

def S (n : ℕ) : ℕ := ∑ i in finset.range (n + 1), a i

def T (n : ℕ) : ℕ := ∑ i in finset.range (n + 1), b i

-- Conditions:
axiom a1_a3_eq_five : a 1 + a 3 = 5
axiom S3_eq_seven : S 3 = 7

-- Proof statements:
theorem general_term_a : ∀ n : ℕ, a n = 2^(n-1) :=
sorry

theorem sum_Tn : ∀ n : ℕ, T n = (2 * n - 3) * 2^n + 3 :=
sorry

end general_term_a_sum_Tn_l242_242449


namespace total_nominal_income_l242_242574

noncomputable def monthly_income (principal : ℝ) (rate : ℝ) (months : ℕ) : ℝ :=
  principal * ((1 + rate) ^ months - 1)

def total_income : ℝ :=
  let rate := 0.06 / 12
  let principal := 8700
  (monthly_income principal rate 6) + 
  (monthly_income principal rate 5) + 
  (monthly_income principal rate 4) + 
  (monthly_income principal rate 3) + 
  (monthly_income principal rate 2) + 
  (monthly_income principal rate 1)

theorem total_nominal_income :
  total_income = 921.15 :=
by
  sorry

end total_nominal_income_l242_242574


namespace divisor_of_z_in_form_4n_minus_1_l242_242178

theorem divisor_of_z_in_form_4n_minus_1
  (x y : ℕ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (z : ℕ) 
  (hz : z = 4 * x * y / (x + y)) 
  (hz_odd : z % 2 = 1) :
  ∃ n : ℕ, n > 0 ∧ ∃ d : ℕ, d ∣ z ∧ d = 4 * n - 1 :=
sorry

end divisor_of_z_in_form_4n_minus_1_l242_242178


namespace cube_identity_l242_242113

theorem cube_identity (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 := by
  sorry

end cube_identity_l242_242113


namespace min_distance_diff_l242_242023

noncomputable def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

def point (x y : ℝ) := (x, y)

noncomputable def distance (p q : ℝ × ℝ) :=
  real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

noncomputable def min_value_of_MN_MF : ℝ :=
  sorry

theorem min_distance_diff (F M N : ℝ × ℝ) (a b : ℝ)
  (h1F : F = (-1, 0))
  (h2ell : ellipse a b M.1 M.2)
  (h3N : N = (5, 3))
  (h4ab : a = sqrt 3 ∧ b = sqrt 2)
  : min_value_of_MN_MF = 5 - 2 * sqrt 3 :=
sorry

end min_distance_diff_l242_242023


namespace intersect_count_l242_242742

def quadratic1 (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 2
def quadratic2 (x : ℝ) : ℝ := 3 * x^2 + 5 * x + 2
def line (x : ℝ) : ℝ := x + 1

theorem intersect_count : 
  let intersections : ℕ := 
    if (0, quadratic1 0) = (0, quadratic2 0) 
    then 1 + (if (1, quadratic1 1) = (1, line 1) then 1 else 0) + (if ((1/3 : ℝ), quadratic1 (1/3)) = (1/3, line (1/3)) then 1 else 0)
    else 
    (if (1, quadratic1 1) = (1, line 1) then 1 else 0) + (if ((1/3 : ℝ), quadratic1 (1/3)) = (1/3, line (1/3)) then 1 else 0) in
  intersections = 3 := sorry

end intersect_count_l242_242742


namespace sin_double_angle_l242_242166

theorem sin_double_angle (θ : ℝ) 
  (h₁ : θ > π/2 ∧ θ < π) 
  (h₂ : cos (π / 2 - θ) = 3 / 5) : 
  sin (2 * θ) = -24 / 25 := 
sorry

end sin_double_angle_l242_242166


namespace log_pow_evaluation_l242_242904

-- Define the condition that 9 is equal to 3^2
def nine_eq_three_squared : Prop := 9 = 3^2

-- The main theorem that we need to prove
theorem log_pow_evaluation (h : nine_eq_three_squared) : log 3 (9^3) = 6 :=
  sorry

end log_pow_evaluation_l242_242904


namespace gcd_polynomial_multiple_of_532_l242_242442

theorem gcd_polynomial_multiple_of_532 (a : ℤ) (h : ∃ k : ℤ, a = 532 * k) :
  Int.gcd (5 * a ^ 3 + 2 * a ^ 2 + 6 * a + 76) a = 76 :=
by
  sorry

end gcd_polynomial_multiple_of_532_l242_242442


namespace focal_length_hyperbola_eq_8_l242_242976

noncomputable def focal_length_of_hyperbola (m : ℝ) : ℝ :=
  let a_squared := m^2 + 12
  let b_squared := 4 - m^2
  let c_squared := a_squared + b_squared
  let c := Real.sqrt c_squared
  2 * c

theorem focal_length_hyperbola_eq_8 (m : ℝ) : focal_length_of_hyperbola m = 8 :=
by
  have a_squared := m^2 + 12
  have b_squared := 4 - m^2
  have c_squared := a_squared + b_squared
  have h1 : c_squared = 16 := by linarith
  have c := Real.sqrt c_squared
  have h2 : c = 4 := by simp [h1]
  show focal_length_of_hyperbola m = 8
  rw [focal_length_of_hyperbola]
  rw [h2]
  simp
  sorry

end focal_length_hyperbola_eq_8_l242_242976


namespace pradeep_maximum_marks_l242_242668

theorem pradeep_maximum_marks (M : ℝ) (h1 : 0.35 * M = 175) :
  M = 500 :=
by
  sorry

end pradeep_maximum_marks_l242_242668


namespace smallest_value_at_12_l242_242981

theorem smallest_value_at_12 :
  ∀ (x : ℕ), x = 12 → 
  let option_A := (8 : ℚ) / x
  let option_B := (8 : ℚ) / (x + 2)
  let option_C := (8 : ℚ) / (x - 2)
  let option_D := (x : ℚ) / 8
  let option_E := (x + 2 : ℚ) / 8
  option_B = min option_A (min option_B (min option_C (min option_D option_E))) :=
by
  intro x hx
  simp [hx]
  have hA : 8 / 12 = 2 / 3 := by norm_num
  have hB : 8 / (12 + 2) = 4 / 7 := by norm_num
  have hC : 8 / (12 - 2) = 4 / 5 := by norm_num
  have hD : 12 / 8 = 3 / 2 := by norm_num
  have hE : (12 + 2) / 8 = 7 / 4 := by norm_num
  rw [hA, hB, hC, hD, hE]
  norm_num
  sorry

end smallest_value_at_12_l242_242981


namespace log_base_3_of_9_cubed_l242_242865

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 := by
  -- Assuming the necessary properties of logarithms and exponents
  sorry

end log_base_3_of_9_cubed_l242_242865


namespace log_evaluation_l242_242767

theorem log_evaluation : ∀ (a : ℕ), a = 3 → log 3 (9^3) = 6 :=
by
  intro a ha
  have h1 : 9 = 3^2 := by sorry
  have h2 : (9^3) = (3^2)^3 := by rw [h1]
  have h3 : (3^2)^3 = 3^6 := by sorry
  have h4 : log 3 (3^6) = 6 := by sorry
  rw [h2, h3, h4]
  exact ha

end log_evaluation_l242_242767


namespace value_of_N_l242_242553

noncomputable def N : ℚ := (sqrt (sqrt 8 + 3) + sqrt (sqrt 8 - 3)) / sqrt (sqrt 8 + 2) - sqrt (4 - 2 * sqrt 3)
def correct_answer : ℚ := (1 + sqrt 6 - sqrt 3) / 2

theorem value_of_N :
  N = correct_answer :=
sorry

end value_of_N_l242_242553


namespace log3_of_9_to_3_l242_242803

theorem log3_of_9_to_3 : Real.logb 3 (9^3) = 6 :=
by
  have h1 : 9 = 3^2 := by norm_num
  have h2 : 9^3 = (3^2)^3 := by rw [h1]
  have h3 : (3^2)^3 = 3^(2 * 3) := by rw [pow_mul]
  have h4 : Real.logb 3 (3^(2 * 3)) = 2 * 3 := Real.logb_pow 3 (2 * 3)
  rw [h4]
  norm_num

end log3_of_9_to_3_l242_242803


namespace digit_assignment_correct_l242_242144

noncomputable def digit_assignment : Prop :=
  ∃ (a b c d e f g h i j z : ℕ),
    a = 7 ∧
    b = 3 ∧
    c = 4 ∧
    d = 9 ∧
    e = 1 ∧
    f = 5 ∧
    g = 7 ∧
    h = 6 ∧
    i = 4 ∧
    j = 2 ∧
    z = 4 ∧
    (a * 1_000_000 + b * 100_000 + z * 10_000 + c * 1_000 + d * 100 + e * 10 + z) *
    (f * 100_000 + g * 10_000 + h * 1_000 + i * 100 + z * 10 + j) =
    4234162045288

theorem digit_assignment_correct : digit_assignment :=
sorry

end digit_assignment_correct_l242_242144


namespace perfect_square_factors_count_l242_242483

-- Statement of the problem in Lean
theorem perfect_square_factors_count : 
  let n := (2 ^ 12) * (3 ^ 15) * (5 ^ 18) * (7 ^ 8)
  ∃ count, count = 2800 ∧ 
    (∀ d, d > 0 → d * d ≤ n → n % (d * d) = 0 → 
      (∃ a b c d, d = 2 ^ a * 3 ^ b * 5 ^ c * 7 ^ d) ∧ a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0 ∧ d % 2 = 0) :=
begin
  let n := (2 ^ 12) * (3 ^ 15) * (5 ^ 18) * (7 ^ 8),
  use 2800,
  split,
  exact rfl,
  intros d h_pos h_le h_factor,
  sorry,
end

end perfect_square_factors_count_l242_242483


namespace joel_age_when_dad_twice_l242_242156

theorem joel_age_when_dad_twice (x joel_age dad_age: ℕ) (h₁: joel_age = 12) (h₂: dad_age = 47) 
(h₃: dad_age + x = 2 * (joel_age + x)) : joel_age + x = 35 :=
by
  rw [h₁, h₂] at h₃ 
  sorry

end joel_age_when_dad_twice_l242_242156


namespace circle_diameter_of_right_triangle_l242_242711

-- Conditions
def right_triangle (area : ℝ) (leg1 : ℝ) (leg2 : ℝ) : Prop :=
  area = (leg1 * leg2) / 2

def inscribed_circle_diameter (leg1 leg2 : ℝ) : ℝ :=
  let hypotenuse := real.sqrt ((leg1^2) + (leg2^2))
  in hypotenuse

-- Theorem statement
theorem circle_diameter_of_right_triangle (area : ℝ) (leg1 : ℝ) (leg2 : ℝ) (hypotenuse : ℝ) (d : ℝ) :
  right_triangle area leg1 leg2 ∧ leg1 = 30 ∧ area = 150 ∧ leg2 = 10 ∧ hypotenuse = 10 * real.sqrt 10 ∧ d = hypotenuse →
  d = 10 * real.sqrt(10) :=
by sorry

end circle_diameter_of_right_triangle_l242_242711


namespace log_evaluation_l242_242755

theorem log_evaluation : ∀ (a : ℕ), a = 3 → log 3 (9^3) = 6 :=
by
  intro a ha
  have h1 : 9 = 3^2 := by sorry
  have h2 : (9^3) = (3^2)^3 := by rw [h1]
  have h3 : (3^2)^3 = 3^6 := by sorry
  have h4 : log 3 (3^6) = 6 := by sorry
  rw [h2, h3, h4]
  exact ha

end log_evaluation_l242_242755


namespace same_side_probability_l242_242657

theorem same_side_probability (p : ℝ) (n : ℕ) (h_fair_coin : p = 0.5) (h_num_tosses : n = 4) :
  let prob_side_repeats := p^(n) + p^(n) in
  prob_side_repeats = 0.125 :=
by
  let prob_heads := p^n
  let prob_tails := p^n
  let prob_side_repeats := prob_heads + prob_tails
  have h_prob : prob_side_repeats = 0.125 := sorry
  exact h_prob

end same_side_probability_l242_242657


namespace construct_triangle_given_sides_and_median_l242_242743

theorem construct_triangle_given_sides_and_median
  (A B C E : Type)
  (c b m : ℝ)
  (triangle_ABC : Triangle A B C)
  (h1 : distance A B = c)
  (h2 : distance A C = b)
  (h3 : midpoint E B C)
  (h4 : distance A E = m)
  : ∃ (ABC : Triangle A B C), true := 
  sorry

end construct_triangle_given_sides_and_median_l242_242743


namespace distinguishable_colorings_of_octahedron_l242_242751

theorem distinguishable_colorings_of_octahedron :
  ∃ (σ : (Fin 6) → Fin 6), σ.perm.support.card = 720 := 
sorry

end distinguishable_colorings_of_octahedron_l242_242751


namespace bella_steps_to_meet_ella_l242_242356

theorem bella_steps_to_meet_ella
  (total_distance_miles : ℕ)
  (distance_in_feet : ℕ)
  (feet_per_step : ℕ)
  (speed_ratio : ℕ)
  : total_distance_miles = 3 →
    distance_in_feet = 15840 →
    feet_per_step = 3 →
    speed_ratio = 4 →
    (forall (b_speed : ℕ), b_speed * (7920 / (5 * b_speed)) / feet_per_step = 528) :=
begin
  intros,
  sorry
end

end bella_steps_to_meet_ella_l242_242356


namespace log_base_three_of_nine_cubed_l242_242876

theorem log_base_three_of_nine_cubed : log 3 (9^3) = 6 :=
by
  have h1 : 9 = 3^2 := by sorry
  have h2 : ∀ (a m n: ℕ), (a^m)^n = a^(m * n) := by 
    intros a m n
    exact pow_mul a m n
  have h3 : ∀ (b n: ℕ), log b (b^n) = n := by sorry
  sorry

end log_base_three_of_nine_cubed_l242_242876


namespace sum_difference_20_l242_242666

def sum_of_even_integers (n : ℕ) : ℕ := (n / 2) * (2 + 2 * (n - 1))

def sum_of_odd_integers (n : ℕ) : ℕ := (n / 2) * (1 + 2 * (n - 1))

theorem sum_difference_20 : sum_of_even_integers (20) - sum_of_odd_integers (20) = 20 := by
  sorry

end sum_difference_20_l242_242666


namespace total_waiting_time_difference_l242_242635

theorem total_waiting_time_difference :
  let n_swings := 6
  let n_slide := 4 * n_swings
  let t_swings := 3.5 * 60
  let t_slide := 45
  let T_swings := n_swings * t_swings
  let T_slide := n_slide * t_slide
  let T_difference := T_swings - T_slide
  T_difference = 180 :=
by
  sorry

end total_waiting_time_difference_l242_242635


namespace election_votes_l242_242308

variable (V : ℕ)
variable (winner_votes : ℕ)
variable (loser_votes : ℕ)

theorem election_votes (h1 : winner_votes = 0.60 * V)
                      (h2 : loser_votes = 0.40 * V)
                      (h3 : winner_votes - loser_votes = 288) :
                      winner_votes = 864 :=
by
  sorry

end election_votes_l242_242308


namespace segments_in_semicircles_l242_242134

noncomputable def right_triangle := 
  (leg1 leg2 hypotenuse height segment1 segment2 : ℝ)
  (h_right_triangle : leg1 = 75 ∧ leg2 = 100 ∧ hypotenuse = 125 ∧ height = 60 ∧ segment1 = 48 ∧ segment2 = 36)

theorem segments_in_semicircles :
  ∃ hypotenuse height segment1 segment2,
    right_triangle 75 100 hypotenuse height segment1 segment2 ∧
    hypotenuse = 125 ∧ height = 60 ∧ segment1 = 48 ∧ segment2 = 36 :=
by
  sorry

end segments_in_semicircles_l242_242134


namespace distinct_cyclic_quadrilaterals_perimeter_36_l242_242067

noncomputable def count_distinct_cyclic_quadrilaterals : Nat :=
  1026

theorem distinct_cyclic_quadrilaterals_perimeter_36 :
  (∃ (a b c d : ℕ), a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ a + b + c + d = 36 ∧ a < b + c + d) → count_distinct_cyclic_quadrilaterals = 1026 :=
by
  rintro ⟨a, b, c, d, hab, hbc, hcd, hsum, hlut⟩
  sorry

end distinct_cyclic_quadrilaterals_perimeter_36_l242_242067


namespace part1_part2_l242_242052

-- Define the function f
def f (x : ℝ) : ℝ := 5 * (Real.sin x)^2 - 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 3 * (Real.cos x)^2

-- Prove that f is monotonically decreasing in the given interval
theorem part1 (k : ℤ) : 
  monotonic_decreasing_on f (Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6)) := sorry

-- Given conditions for part 2
variable {α : ℝ}
axiom h1 : f (α + Real.pi / 6) = 12 / 5
axiom h2 : Real.pi / 2 < α ∧ α < Real.pi

-- Prove the value of tan(2α + π/4)
theorem part2 : Real.tan (2 * α + Real.pi / 4) = 1 / 7 := sorry

end part1_part2_l242_242052


namespace circumcircle_radius_and_area_l242_242687

theorem circumcircle_radius_and_area (N P Q F : Type*)
  (hNQ : N ≠ Q)
  (h_isosceles : is_isosceles_triangle N P Q)
  (h_midpoint_arc : is_midpoint_arc F P N (λ f : Nat, true))
  (hdist1 : dist F (line P N) = 5)
  (hdist2 : dist F (line Q N) = (20 / 3)) :
  ∃ (R : ℝ), (circumradius N P Q = 6) ∧ (triangle_area N P Q = (35 * real.sqrt 35 / 9)) :=
by
  sorry

end circumcircle_radius_and_area_l242_242687


namespace log_three_nine_cubed_l242_242959

theorem log_three_nine_cubed : ∀ (log : ℕ → ℕ → ℕ), log 3 9 = 2 → log 3 3 = 1 → (∀ (a b n : ℕ), log a (b ^ n) = n * log a b) → log 3 (9 ^ 3) = 6 :=
by
  intros log h1 h2 h3
  sorry

end log_three_nine_cubed_l242_242959


namespace roses_in_vase_l242_242638

theorem roses_in_vase (initial_roses : ℕ) (roses_left : ℕ) (put_in_vase : ℕ) 
  (h1 : initial_roses = 29) (h2 : roses_left = 12) : put_in_vase = 17 :=
by
  have h3 : put_in_vase = initial_roses - roses_left, from sorry
  rw [h1, h2] at h3
  exact h3

end roses_in_vase_l242_242638


namespace gcd_factorial_8_10_l242_242275

theorem gcd_factorial_8_10 (n : ℕ) (hn : n = 10! - 8!): gcd 8! 10! = 8! := by
  sorry

end gcd_factorial_8_10_l242_242275


namespace evaluate_expression_l242_242427

noncomputable def g (A B C D x : ℝ) : ℝ := A * x^3 + B * x^2 - C * x + D

theorem evaluate_expression (A B C D : ℝ) (h1 : g A B C D 2 = 5) (h2 : g A B C D (-1) = -8) (h3 : g A B C D 0 = 2) :
  -12 * A + 6 * B - 3 * C + D = 27.5 :=
by
  sorry

end evaluate_expression_l242_242427


namespace lambda_range_l242_242478

def vector (α : Type*) := list α

def dot_product (a b : vector ℝ) : ℝ :=
  (a.headD 0) * (b.headD 0) + (a.tail.headD 0) * (b.tail.headD 0)

def is_acute_angle (a b : vector ℝ) : Prop := 
  dot_product a b > 0

theorem lambda_range (λ : ℝ) : 
  is_acute_angle [1, -2] [2, λ] → λ ∈ set.Ioo ⊥ (-4) ∪ set.Ioo (-4) 1 :=
by
  sorry

end lambda_range_l242_242478


namespace cubic_sum_l242_242085

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 := by
  sorry

end cubic_sum_l242_242085


namespace log_base_3_l242_242852

theorem log_base_3 (log_identity : ∀ (b a c : ℝ), log b (a^c) = c * log b a) :
  log 3 (9^3) = 6 :=
by
  have : 9 = 3^2 := by sorry
  have : log 3 3 = 1 := by sorry
  exact (log_identity 3 (3^2) 3).trans sorry

end log_base_3_l242_242852


namespace range_of_eccentricity_l242_242630

noncomputable def eccentricity_of_ellipse_with_conditions (a c : ℝ) (e : ℝ) : Prop :=
  a > 0 ∧ c > 0 ∧ e = c / a ∧ e ∈ (set.Icc (1 / 4) (1 / 2))

theorem range_of_eccentricity (a c : ℝ) (h : |3 * c| = 3 * c) :
  eccentricity_of_ellipse_with_conditions a c (c / a) :=
by {
  sorry
}

end range_of_eccentricity_l242_242630


namespace fraction_deviation_l242_242192

theorem fraction_deviation (x : ℝ) (h : 1 ≤ x ∧ x ≤ 9) :
  abs ((real.sqrt x) - ((6 * x + 6) / (x + 11))) < 0.05 :=
sorry

end fraction_deviation_l242_242192


namespace max_value_y_l242_242444

noncomputable def y (x : ℝ) : ℝ := 3 - 3*x - 1/x

theorem max_value_y : (∃ x > 0, ∀ x' > 0, y x' ≤ y x) ∧ (y (1 / Real.sqrt 3) = 3 - 2 * Real.sqrt 3) :=
by
  sorry

end max_value_y_l242_242444


namespace total_books_l242_242637

noncomputable def num_books_on_shelf : ℕ := 8

theorem total_books (p h s : ℕ) (assump1 : p = 2) (assump2 : h = 6) (assump3 : s = 36) :
  p + h = num_books_on_shelf :=
by {
  -- leaving the proof construction out as per instructions
  sorry
}

end total_books_l242_242637


namespace construct_line_segment_l242_242643

-- Define the conditions
variables (A B : ℝ^2) -- Points A and B in the 2D plane
hypothesis h_dist : |A - B| ≈ 37 -- Points A and B are approximately 37 cm apart
variables (straightedge : ℝ) (right_triangle_hypotenuse : ℝ)
hypothesis h_se_length : straightedge ≈ 20 -- Straightedge is approximately 20 cm
hypothesis h_rt_length : right_triangle_hypotenuse ≈ 15 -- Right-angled triangle has a hypotenuse of approximately 15 cm

-- State the problem
theorem construct_line_segment :
  ∃ (line_AB : ℝ → ℝ^2) (h_construct : true), -- Placeholder for the line function and a proof
  (line_AB 0 = A) ∧ (line_AB 1 = B) := 
sorry

end construct_line_segment_l242_242643


namespace g_f_of_3_l242_242168

def f (x : ℝ) : ℝ := x^3 - 4
def g (x : ℝ) : ℝ := 3 * x^2 + 5 * x + 2

theorem g_f_of_3 : g (f 3) = 1704 := by
  sorry

end g_f_of_3_l242_242168


namespace wendy_phone_pictures_l242_242268

theorem wendy_phone_pictures:
  (total_albums : ℕ) (pictures_per_album : ℕ) (camera_pictures : ℕ)
  (total_albums = 4) (pictures_per_album = 6) (camera_pictures = 2) :
  total_albums * pictures_per_album - camera_pictures = 22 :=
by
  sorry

end wendy_phone_pictures_l242_242268


namespace complex_number_count_l242_242977

theorem complex_number_count
  (z : ℂ) 
  (hz : ∥z∥ = 1)
  (h_real : (z ^ finset.factorial 7 - z ^ finset.factorial 6).im = 0) :
  ∃ (n : ℕ), n = 3440 :=
by sorry

end complex_number_count_l242_242977


namespace coefficient_x2_in_expansion_l242_242394

theorem coefficient_x2_in_expansion : 
  (λ x : ℚ, coefficient (x^2) ((1 + 2 * x) ^ 3 * (1 - x) ^ 4)) = -6 :=
sorry

end coefficient_x2_in_expansion_l242_242394


namespace count_convex_cyclic_quadrilaterals_is_1505_l242_242070

-- Define a quadrilateral using its sides
structure Quadrilateral where
  a b c d : ℕ
  deriving Repr, DecidableEq

-- Define what a convex cyclic quadrilateral is, given the integer sides and the perimeter condition
def isConvexCyclicQuadrilateral (q : Quadrilateral) : Prop :=
 q.a + q.b + q.c + q.d = 36 ∧
 q.a > 0 ∧ q.b > 0 ∧ q.c > 0 ∧ q.d > 0 ∧
 q.a + q.b > q.c + q.d ∧ q.c + q.d > q.a + q.b ∧
 q.b + q.c > q.d + q.a ∧ q.d + q.a > q.b + q.c

-- Noncomputable definition to count all convex cyclic quadrilaterals
noncomputable def countConvexCyclicQuadrilaterals : ℕ :=
  sorry

-- The theorem stating the count is equal to 1505
theorem count_convex_cyclic_quadrilaterals_is_1505 :
  countConvexCyclicQuadrilaterals = 1505 :=
  sorry

end count_convex_cyclic_quadrilaterals_is_1505_l242_242070


namespace binom_30_3_is_4060_l242_242369

theorem binom_30_3_is_4060 : Nat.choose 30 3 = 4060 := by
  sorry

end binom_30_3_is_4060_l242_242369


namespace archit_ayush_probability_l242_242353

/-
  Define the problem:
  - Set of points (x,y) for -1 ≤ x,y ≤ 1.
  - Archit starts at (1,1).
  - Ayush starts at (1,0).
  - Movements are to a random point at distance 1.
  - Prove that the probability Archit reaches (0,0) before Ayush is 2/5, and m+n = 7.
-/

theorem archit_ayush_probability :
  let points := {p : ℤ × ℤ | -1 ≤ p.1 ∧ p.1 ≤ 1 ∧ -1 ≤ p.2 ∧ p.2 ≤ 1},
      archit_initial := (1, 1),
      ayush_initial := (1, 0),
      moves := λ p : ℤ × ℤ, {q : ℤ × ℤ | abs (q.1 - p.1) + abs (q.2 - p.2) = 1 ∧ q ∈ points},
      archit_target := (0, 0) in
  (∃ m n, nat.gcd m n = 1 ∧ (m = 2 ∧ n = 5) ∧ m + n = 7) →
  true := 
sorry

end archit_ayush_probability_l242_242353


namespace possible_sums_l242_242047

-- Define the problem conditions
def equation (x a : ℝ) : Prop := |x^2 - 6*x| = a

-- Define the predicate for being a solution to the equation
def is_solution (x : ℝ) (a : ℝ) : Prop := equation x a

-- Define the possible sum of solutions
def sum_of_elements_in_P (a : ℝ) : ℝ :=
if h : 0 < a then
  if a = 9 then 9
  else if a < 9 then 12
  else 6
else 0

-- Main statement to be proved
theorem possible_sums (a : ℝ) (h : 0 < a) : (sum_of_elements_in_P a = 6 ∨ sum_of_elements_in_P a = 9 ∨ sum_of_elements_in_P a = 12) :=
sorry

end possible_sums_l242_242047


namespace circle_conditions_l242_242489

variables {P : Point} {r : ℝ}
variables (x y : ℝ)

-- Condition I
def condition_I : Prop := 
  |P.x| < 1/2 ∧ |P.y| < 1/2

-- Condition II
def condition_II : Prop := 
  |P.x| + |P.y| < 1 ∧ |P.x| ≠ |P.y|

-- Condition III
def condition_III : Prop := 
  ( |P.x| < 1 ∧ |P.y| < 1/2 ∧ P.x ≠ 0 ∧ |P.x| ≠ 1/2 ) ∨ 
  ( |P.x| < 1/2 ∧ |P.y| < 1 ∧ P.y ≠ 0 ∧ |P.y| ≠ 1/2 )

def satisfies_all_conditions : Prop :=
  condition_I ∨ condition_II ∨ condition_III

theorem circle_conditions
  (P : Point) 
  (r : ℝ) 
  (cond : satisfies_all_conditions P r):
  ∃ (center : Point), 
  ( ∀ (Q : Point), Q ≠ origin 
    → Q ∈ lattice_points 
    → Q ∉ circle(center, r)) :=
sorry

end circle_conditions_l242_242489


namespace log_base_3_of_9_cubed_l242_242832

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 := by
  -- Define the known conditions
  have h1 : 9 = 3^2 := by norm_num
  have h2 : log 3 (a^n) = n * log 3 a := sorry  -- Statement of the logarithmic identity
  
  -- Apply the conditions to prove the result
  calc
    log 3 (9^3)
        = log 3 ((3^2)^3) : by rw h1
    ... = log 3 (3^(2 * 3)) : by congr; norm_num
    ... = log 3 (3^6) : by norm_num
    ... = 6 : by rw [h2]; norm_num

end log_base_3_of_9_cubed_l242_242832


namespace log_evaluation_l242_242765

theorem log_evaluation : ∀ (a : ℕ), a = 3 → log 3 (9^3) = 6 :=
by
  intro a ha
  have h1 : 9 = 3^2 := by sorry
  have h2 : (9^3) = (3^2)^3 := by rw [h1]
  have h3 : (3^2)^3 = 3^6 := by sorry
  have h4 : log 3 (3^6) = 6 := by sorry
  rw [h2, h3, h4]
  exact ha

end log_evaluation_l242_242765


namespace log3_of_9_to_3_l242_242805

theorem log3_of_9_to_3 : Real.logb 3 (9^3) = 6 :=
by
  have h1 : 9 = 3^2 := by norm_num
  have h2 : 9^3 = (3^2)^3 := by rw [h1]
  have h3 : (3^2)^3 = 3^(2 * 3) := by rw [pow_mul]
  have h4 : Real.logb 3 (3^(2 * 3)) = 2 * 3 := Real.logb_pow 3 (2 * 3)
  rw [h4]
  norm_num

end log3_of_9_to_3_l242_242805


namespace sum_even_200_to_400_l242_242298

theorem sum_even_200_to_400 : 
  let t₁ := 202
  let tₙ := 398
  let n := 99
  let sum := (t₁ + tₙ) * n / 2 in
  sum = 29700 := 
by
  let t₁ := 202
  let tₙ := 398
  let n := 99
  let sum := (t₁ + tₙ) * n / 2
  show sum = 29700
  sorry

end sum_even_200_to_400_l242_242298


namespace x_cubed_plus_y_cubed_l242_242082

variable (x y : ℝ)
variable (h₁ : x + y = 5)
variable (h₂ : x^2 + y^2 = 17)

theorem x_cubed_plus_y_cubed :
  x^3 + y^3 = 65 :=
by sorry

end x_cubed_plus_y_cubed_l242_242082


namespace rectangle_length_l242_242118

theorem rectangle_length (s w : ℝ) (A : ℝ) (L : ℝ) (h1 : s = 9) (h2 : w = 3) (h3 : A = s * s) (h4 : A = w * L) : L = 27 :=
by
  sorry

end rectangle_length_l242_242118


namespace sequence_fifth_term_l242_242145

theorem sequence_fifth_term :
  let a : ℕ → ℝ := λ n, match n with
                        | 0        => 0                   -- The sequence starts with a₁, so we let a₀ be 0.
                        | 1        => 1
                        | (n + 1)  => 1 + (-1)^(n + 1) / a n
                      in a 5 = 2 / 3 := 
by
  sorry

end sequence_fifth_term_l242_242145


namespace total_red_cards_l242_242713

theorem total_red_cards (num_standard_decks : ℕ) (num_special_decks : ℕ)
  (red_standard_deck : ℕ) (additional_red_special_deck : ℕ)
  (total_decks : ℕ) (h1 : num_standard_decks = 5)
  (h2 : num_special_decks = 10)
  (h3 : red_standard_deck = 26)
  (h4 : additional_red_special_deck = 4)
  (h5 : total_decks = num_standard_decks + num_special_decks) :
  num_standard_decks * red_standard_deck +
  num_special_decks * (red_standard_deck + additional_red_special_deck) = 430 := by
  -- Proof is omitted.
  sorry

end total_red_cards_l242_242713


namespace plant_is_red_daisy_l242_242075

def plant_color := Prop
def plant_type := Prop

variable (red purple yellow : plant_color)
variable (rose daisy dahlia : plant_type)

-- Anika's statement
def Anika (c : plant_color) (t : plant_type) : Prop :=
  c = red ∧ t = rose

-- Bill's statement
def Bill (c : plant_color) (t : plant_type) : Prop :=
  c = purple ∧ t = daisy

-- Cathy's statement
def Cathy (c : plant_color) (t : plant_type) : Prop :=
  c = red ∧ t = dahlia

-- Plant identification
def plant := {c : plant_color // 
              {t : plant_type // 
                (Anika c t ∨ Anika t c) ∧
                (Bill c t ∨ Bill t c) ∧
                (Cathy c t ∨ Cathy t c) }}

-- Problem Statement: Prove the plant is a red daisy.
theorem plant_is_red_daisy : ∃ (c : plant_color) (t : plant_type), 
  Anika c t ∧ Bill c t ∧ Cathy c t → c = red ∧ t = daisy := 
sorry

end plant_is_red_daisy_l242_242075


namespace die_total_dots_l242_242593

theorem die_total_dots :
  ∀ (face1 face2 face3 face4 face5 face6 : ℕ),
    face1 < face2 ∧ face2 < face3 ∧ face3 < face4 ∧ face4 < face5 ∧ face5 < face6 ∧
    (face2 - face1 ≥ 2) ∧ (face3 - face2 ≥ 2) ∧ (face4 - face3 ≥ 2) ∧ (face5 - face4 ≥ 2) ∧ (face6 - face5 ≥ 2) ∧
    (face3 ≠ face1 + 2) ∧ (face4 ≠ face2 + 2) ∧ (face5 ≠ face3 + 2) ∧ (face6 ≠ face4 + 2)
    → face1 + face2 + face3 + face4 + face5 + face6 = 27 :=
by {
  sorry
}

end die_total_dots_l242_242593


namespace total_days_1996_to_2000_l242_242485

theorem total_days_1996_to_2000 : 
  let is_leap_year (year : ℕ) : Bool :=
    (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ (year % 400 = 0)
  in 
  let days_in_year (year : ℕ) : ℕ :=
    if is_leap_year year then 366 else 365
  in 
  (days_in_year 1996 + days_in_year 1997 + days_in_year 1998 + days_in_year 1999 + days_in_year 2000) = 1827 := 
by 
  sorry

end total_days_1996_to_2000_l242_242485


namespace find_intersection_l242_242397

def intersection_point (x y : ℚ) : Prop :=
  3 * x + 4 * y = 12 ∧ 7 * x - 2 * y = 14

theorem find_intersection :
  intersection_point (40 / 17) (21 / 17) :=
by
  sorry

end find_intersection_l242_242397


namespace find_f_at_1_l242_242051

noncomputable def f (x : ℝ) : ℝ := 3 * f' 1 * x - x^2 + Real.log x + 1 / 2
noncomputable def f' (x : ℝ) : ℝ := deriv f x

theorem find_f_at_1 : f 1 = 1 :=
by
  sorry

end find_f_at_1_l242_242051


namespace log_base_3_of_9_cubed_l242_242899

theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 :=
by assumptions
  have h1 : 9 = 3^2 := sorry,
  have h2 : 9^3 = (3^2)^3 := sorry,
  have h3 : (3^2)^3 = 3^(2*3) := sorry,
  have h4 : 3^(2*3) = 3^6 := sorry,
  have h5 : log 3 (3^6) = 6 := by apply log_pow,
  sorry

end log_base_3_of_9_cubed_l242_242899


namespace sum_of_arithmetic_sequence_l242_242027

noncomputable def a_n (a1 : ℤ) (d : ℤ) (n : ℤ) : ℤ :=
  a1 + (n - 1) * d

noncomputable def S_n (a1 : ℤ) (d : ℤ) (n : ℤ) : ℤ :=
  n * a1 + d * (n * (n - 1) / 2)

theorem sum_of_arithmetic_sequence :
  (∃ (d : ℤ),
    let a1 := 8 in
    (a_n a1 d 4 + a_n a1 d 6 = 0) ∧ S_n a1 d 8 = 8) :=
by
  sorry

end sum_of_arithmetic_sequence_l242_242027


namespace sum_reciprocal_inequality_l242_242558

theorem sum_reciprocal_inequality (n : ℕ) (hn : 0 < n) :
  (1 / 2) ≤ (∑ k in Finset.range n, 1 / (n + k + 1 : ℕ)) ∧ 
  (∑ k in Finset.range n, 1 / (n + k + 1 : ℕ)) < 1 :=
sorry

end sum_reciprocal_inequality_l242_242558


namespace initially_caught_and_tagged_is_30_l242_242513

open Real

-- Define conditions
def total_second_catch : ℕ := 50
def tagged_second_catch : ℕ := 2
def total_pond_fish : ℕ := 750

-- Define ratio condition
def ratio_condition (T : ℕ) : Prop :=
  (T : ℝ) / (total_pond_fish : ℝ) = (tagged_second_catch : ℝ) / (total_second_catch : ℝ)

-- Prove the number of fish initially caught and tagged is 30
theorem initially_caught_and_tagged_is_30 :
  ∃ T : ℕ, ratio_condition T ∧ T = 30 :=
by
  -- Skipping proof
  sorry

end initially_caught_and_tagged_is_30_l242_242513


namespace absent_children_l242_242591

theorem absent_children 
  (total_children : ℕ) (total_bananas : ℕ)
  (bananas_per_child_if_present : ℕ)
  (bananas_per_child_if_absent : ℕ) :
  total_children = 320 →
  total_bananas = total_children * bananas_per_child_if_present →
  bananas_per_child_if_absent = bananas_per_child_if_present + 2 →
  (∃ (absent_children : ℕ), 
    total_children - absent_children + 
    2 * (total_children - absent_children) = total_bananas
    ) → 
  absent_children = 160 := 
by
  sorry

end absent_children_l242_242591


namespace log_three_pow_nine_pow_three_eq_six_l242_242937

open Real

noncomputable def eval_log_three_pow_nine_pow_three : ℝ :=
  log 3 (9^3)

theorem log_three_pow_nine_pow_three_eq_six :
  eval_log_three_pow_nine_pow_three = 6 := by
  sorry

end log_three_pow_nine_pow_three_eq_six_l242_242937


namespace prob_divisible_by_3_is_one_third_l242_242306

-- Define prime digits
def prime_digits : set ℕ := {2, 3, 5, 7}

-- Define the set of two-digit numbers formed by prime digits
def two_digit_nums : set ℕ := { n | ∃ a b, a ∈ prime_digits ∧ b ∈ prime_digits ∧ n = 10 * a + b }

-- Define the condition for divisibility by 3
def divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

-- Calculate the set of numbers divisible by 3 from two_digit_nums
def favorable_set : set ℕ := { n ∈ two_digit_nums | divisible_by_3 n }

-- Calculate the probability
def probability_q : ℚ := (favorable_set.to_finset.card : ℚ) / (two_digit_nums.to_finset.card : ℚ)

-- Statement of the problem as a Lean theorem
theorem prob_divisible_by_3_is_one_third : probability_q = 1 / 3 := by
  sorry

end prob_divisible_by_3_is_one_third_l242_242306


namespace cos_arcsin_4_5_l242_242373

theorem cos_arcsin_4_5 (x : ℝ) (h₁ : sin x = 4/5) (h₂ : x = arcsin (4/5)) :
  cos (arcsin (4/5)) = 3/5 :=
sorry

end cos_arcsin_4_5_l242_242373


namespace log_base_three_of_nine_cubed_l242_242882

theorem log_base_three_of_nine_cubed : log 3 (9^3) = 6 :=
by
  have h1 : 9 = 3^2 := by sorry
  have h2 : ∀ (a m n: ℕ), (a^m)^n = a^(m * n) := by 
    intros a m n
    exact pow_mul a m n
  have h3 : ∀ (b n: ℕ), log b (b^n) = n := by sorry
  sorry

end log_base_three_of_nine_cubed_l242_242882


namespace container_capacity_l242_242680

theorem container_capacity
  (C : ℝ)  -- Total capacity of the container in liters
  (h1 : C / 2 + 20 = 3 * C / 4)  -- Condition combining the water added and the fractional capacities
  : C = 80 := 
sorry

end container_capacity_l242_242680


namespace log_base_3_of_9_cubed_l242_242932
-- Import the necessary parts of the standard library

-- Define the statement to be proved
theorem log_base_3_of_9_cubed : log 3 (9^3) = 6 :=
by
  sorry

end log_base_3_of_9_cubed_l242_242932
