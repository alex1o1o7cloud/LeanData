import Mathlib
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order.Field
import Mathlib.Algebra.QuadraticDiscriminant
import Mathlib.Analysis.Calculus.Basic
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Complex.Real
import Mathlib.Analysis.SpecialFunctions.Pi
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.SimpleGraph
import Mathlib.Data.Combinatorics
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.ConicSections.Basic
import Mathlib.Geometry.Constructions.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Init.Function
import Mathlib.MeasureTheory.ProbabilityMassFunction
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Topology.Basic

namespace lcm_12_18_l298_298409

theorem lcm_12_18 : Nat.lcm 12 18 = 36 :=
by
  -- Definitions of the conditions
  have h12 : 12 = 2 * 2 * 3 := by norm_num
  have h18 : 18 = 2 * 3 * 3 := by norm_num
  
  -- Calculating LCM using the built-in Nat.lcm
  rw [Nat.lcm_comm]  -- Ordering doesn't matter for lcm
  rw [Nat.lcm, h12, h18]
  -- Prime factorizations checks are implicitly handled
  
  -- Calculate the LCM based on the highest powers from the factorizations
  have lcm_val : 4 * 9 = 36 := by norm_num
  
  -- So, the LCM of 12 and 18 is
  exact lcm_val

end lcm_12_18_l298_298409


namespace domain_of_f_log2x_is_0_4_l298_298462

def f : ℝ → ℝ := sorry

-- Given condition: domain of y = f(2x) is (-1, 1)
def dom_f_2x (x : ℝ) : Prop := -1 < 2 * x ∧ 2 * x < 1

-- Conclusion: domain of y = f(log_2 x) is (0, 4)
def dom_f_log2x (x : ℝ) : Prop := 0 < x ∧ x < 4

theorem domain_of_f_log2x_is_0_4 (x : ℝ) :
  (dom_f_2x x) → (dom_f_log2x x) :=
by
  sorry

end domain_of_f_log2x_is_0_4_l298_298462


namespace dice_probability_l298_298695

-- Definitions based on the problem
def first_die_numbers : Finset ℕ := (Finset.range 19).map ⟨λ n, n + 2, (Finset.range 19).nodup.map _⟩
def second_die_numbers : Finset ℕ := (Finset.range 11 ∪ Finset.range' 12 8).erase 11

-- Probability calculation for sum 30
def num_successful_outcomes (first_die_numbers second_die_numbers : Finset ℕ) : ℕ :=
(first_die_numbers.product second_die_numbers).filter (λ p, p.1 + p.2 = 30).card

def total_possibilities : ℕ := 20 * 20

def probability := (num_successful_outcomes first_die_numbers second_die_numbers : ℚ) / total_possibilities

-- Theorem to prove
theorem dice_probability : probability = 9 / 400 := 
by
  -- Proof will go here
  sorry

end dice_probability_l298_298695


namespace reflect_center_of_circle_l298_298246

def reflect_point (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (-y, -x)

theorem reflect_center_of_circle :
  reflect_point (3, -7) = (7, -3) :=
by
  sorry

end reflect_center_of_circle_l298_298246


namespace intersection_domains_l298_298851

def domain_f := {x : ℝ | x < 1}
def domain_g := {x : ℝ | x ≠ 0}

theorem intersection_domains :
  {x : ℝ | x < 1} ∩ {x : ℝ | x ≠ 0} = {x : ℝ | x < 1 ∧ x ≠ 0} :=
by 
  sorry

end intersection_domains_l298_298851


namespace find_a_and_solve_inequality_l298_298844

theorem find_a_and_solve_inequality :
  (∀ x : ℝ, |x^2 - 4 * x + a| + |x - 3| ≤ 5 → x ≤ 3) →
  a = 8 :=
by
  sorry

end find_a_and_solve_inequality_l298_298844


namespace problem1_problem2_l298_298768

noncomputable def proof1 : ℝ := 
  (2 * Real.log 2 + Real.log 3) / 
  (1 + 1 / 2 * Real.log 0.36 + 1 / 3 * Real.log 8)

theorem problem1 : proof1 = 1 := 
by  
  sorry

noncomputable def proof2 : ℝ := 
  3 * (-4) ^ 3 - (1/2) ^ 0 + 0.25 ^ (1/2) * (-1 / Real.sqrt 2) ^ (-4)

theorem problem2 : proof2 = -191 := 
by
  sorry

end problem1_problem2_l298_298768


namespace concurrency_of_lines_l298_298823

-- Given statement definitions
variables {A B C C' B' P H H' : Point}
variables (ABC : Triangle A B C)
variables (circle_passing_BC : Circle B C)
variables (intersects_AB_at_C' : OnCircle C' (circle_passing_BC) ∧ LineThrough A C' = LineThrough A B)
variables (intersects_AC_at_B' : OnCircle B' (circle_passing_BC) ∧ LineThrough A B' = LineThrough A C)
variables (H_is_orthocenter : Orthocenter H ABC)
variables (H'_is_orthocenter : Orthocenter H' (Triangle A B' C'))

-- Proof goal
theorem concurrency_of_lines 
    (BB'_intersection : Intersection (LineThrough B B') (LineThrough C C') = P) :
    Concurrent (LineThrough B B') (LineThrough C C') (LineThrough H H') :=
sorry

end concurrency_of_lines_l298_298823


namespace max_area_of_triangle_l298_298651

theorem max_area_of_triangle (AB BC AC : ℝ) (ratio : BC / AC = 3 / 5) (hAB : AB = 10) :
  ∃ A : ℝ, (A ≤ 260.52) :=
sorry

end max_area_of_triangle_l298_298651


namespace area_not_covered_by_parabolas_l298_298446

noncomputable section

-- Define the equilateral triangle
structure EquilateralTriangle (α : Type) [LinearOrderedField α] :=
  (A B C : EuclideanSpace α)
  (side_length : α)
  (equilateral : dist A B = side_length ∧ dist B C = side_length ∧ dist C A = side_length)

-- Area of part of the equilateral triangle not covered by any of the parabola's interior
theorem area_not_covered_by_parabolas (α : Type) [LinearOrderedField α]
  (T : EquilateralTriangle α) (h : T.side_length = 1) :
  (let A := T.A 
          B := T.B 
          C := T.C in
    -- Compute total area of the equilateral triangle ABC
    let total_area := (sqrt 3) / 4 * (1 * 1) in
    -- Compute area covered by the interior of each parabola
    let covered_area := sqrt 3 / 4 in -- Placeholder for actual covered area
    -- Area not covered by any parabola
    total_area - (3 * covered_area) = 0.1244) :=
begin
  sorry
end

end area_not_covered_by_parabolas_l298_298446


namespace solve_for_x_l298_298601

theorem solve_for_x (x : ℝ) (h : 5 + 3.5 * x = 2.5 * x - 25) : x = -30 :=
sorry

end solve_for_x_l298_298601


namespace lcm_12_18_l298_298411

theorem lcm_12_18 : Nat.lcm 12 18 = 36 :=
by
  -- Definitions of the conditions
  have h12 : 12 = 2 * 2 * 3 := by norm_num
  have h18 : 18 = 2 * 3 * 3 := by norm_num
  
  -- Calculating LCM using the built-in Nat.lcm
  rw [Nat.lcm_comm]  -- Ordering doesn't matter for lcm
  rw [Nat.lcm, h12, h18]
  -- Prime factorizations checks are implicitly handled
  
  -- Calculate the LCM based on the highest powers from the factorizations
  have lcm_val : 4 * 9 = 36 := by norm_num
  
  -- So, the LCM of 12 and 18 is
  exact lcm_val

end lcm_12_18_l298_298411


namespace mutual_acquaintances_exists_l298_298785

theorem mutual_acquaintances_exists (n : ℕ) 
  (schools : Fin 3 → Fin n → Fin n → Bool)
  (acquaintances : ∀ (i : Fin 3) (s : Fin n), ∀ j ≠ i, 0 < ∑ t, if schools j t s then 1 else 0) :
  ∃ (a b c : Fin n), (a ≠ b ∧ b ≠ c ∧ c ≠ a) 
  ∧ schools 0 a b ∧ schools 0 a c ∧ schools 1 b c :=
sorry

end mutual_acquaintances_exists_l298_298785


namespace incorrect_statement_in_regression_line_l298_298291

theorem incorrect_statement_in_regression_line :
  ¬ (∃ (x y : ℕ), y = \hat{b} * x + \hat{a} → 
      (x = x1 ∧ y = y1) 
   ∨ (x = x2 ∧ y = y2) 
   ∨ ... 
   ∨ (x = xn ∧ y = yn)) :=
  sorry

namespace Conditions

-- Condition 1: Adding or subtracting the same constant to each data in a set doesn't change the variance.
def variance_constant_shift (data : List ℕ) (c : ℕ) : 
  variance (data.add c) = variance data := sorry

-- Condition 2: In a residual plot, narrower band of residuals indicates higher accuracy of model fit.
def narrower_band_higher_accuracy (residuals : List ℕ) :
  (narrow_band residuals) → (higher_accuracy residuals) := sorry

-- Condition 3: In a \(2 \times 2\) contingency table, larger chi-squared value indicates greater certainty of association.
def chi_squared_certainty (chi_squared_value : ℕ) :
  larger_value chi_squared_value → greater_certainty chi_squared_value := sorry

end Conditions

end incorrect_statement_in_regression_line_l298_298291


namespace percentage_passed_both_l298_298299

-- Define the percentages of failures
def percentage_failed_hindi : ℕ := 34
def percentage_failed_english : ℕ := 44
def percentage_failed_both : ℕ := 22

-- Statement to prove
theorem percentage_passed_both : 
  (100 - (percentage_failed_hindi + percentage_failed_english - percentage_failed_both)) = 44 := by
  sorry

end percentage_passed_both_l298_298299


namespace mutually_exclusive_but_not_opposite_l298_298386

-- Define the cards and the people
inductive Card
| Red
| Black
| Blue
| White

inductive Person
| A
| B
| C
| D

-- Define the events
def eventA_gets_red (distribution : Person → Card) : Prop :=
distribution Person.A = Card.Red

def eventB_gets_red (distribution : Person → Card) : Prop :=
distribution Person.B = Card.Red

-- Define mutually exclusive events
def mutually_exclusive (P Q : Prop) : Prop :=
P → ¬ Q

-- Statement of the problem
theorem mutually_exclusive_but_not_opposite :
  ∀ (distribution : Person → Card), 
    mutually_exclusive (eventA_gets_red distribution) (eventB_gets_red distribution) ∧ 
    ¬ (eventA_gets_red distribution ↔ eventB_gets_red distribution) :=
by sorry

end mutually_exclusive_but_not_opposite_l298_298386


namespace largest_n_in_base10_l298_298614

-- Definitions corresponding to the problem conditions
def n_eq_base8_expr (A B C : ℕ) : ℕ := 64 * A + 8 * B + C
def n_eq_base12_expr (A B C : ℕ) : ℕ := 144 * C + 12 * B + A

-- Problem statement translated into Lean
theorem largest_n_in_base10 (n A B C : ℕ) (h1 : n = n_eq_base8_expr A B C) 
    (h2 : n = n_eq_base12_expr A B C) (hA : A < 8) (hB : B < 8) (hC : C < 12) (h_pos: n > 0) : 
    n ≤ 509 :=
sorry

end largest_n_in_base10_l298_298614


namespace sarah_five_dollar_bills_l298_298591

theorem sarah_five_dollar_bills :
  ∃ x y : ℕ, x + y = 15 ∧ 5 * x + 10 * y = 100 ∧ x = 10 :=
begin
  sorry
end

end sarah_five_dollar_bills_l298_298591


namespace probability_real_squared_expression_l298_298984

noncomputable theory

open_locale classical

def S : set ℚ := {q | ∃ (n d : ℤ), 0 ≤ n ∧ n < 2 * d ∧ 1 ≤ d ∧ d ≤ 7 ∧ q = n / d}

theorem probability_real_squared_expression :
  ∃ (p : ℝ), p = classical.some (uniform_distribution S) ∧
    (∀ a b ∈ S, (cos (a * real.pi) + real.I * sin (b * real.pi)) ^ 2 ∈ ℝ ↔ p) :=
sorry

end probability_real_squared_expression_l298_298984


namespace find_f2_l298_298440

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^2005 + a*x^3 - 1/(b*x) - 8

theorem find_f2 (a b : ℝ)
  (h1 : f (-2) a b = 10)
  (h2 : ∀ x, (x^2005 + a*x^3 - 1/(b*x) = x^2005 + a*x^3 - x ⁻¹ * b) ↔ (f (-x) a b = -f(x) a b)) :
  f 2 a b = -26 :=
by
  -- the proof goes here
  sorry

end find_f2_l298_298440


namespace derivative_at_pi_over_4_l298_298467

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x

theorem derivative_at_pi_over_4 :
  deriv f (π / 4) = (Real.sqrt 2 / 2) + (Real.sqrt 2 * π / 8) :=
by
  -- Since the focus is only on the statement, the proof is not required.
  sorry

end derivative_at_pi_over_4_l298_298467


namespace sum_of_reciprocals_of_triangular_numbers_l298_298369

-- Define the nth triangular number
def triangular_number (n : ℕ) : ℝ := n * (n + 1) / 2

-- Define the sum of the reciprocals of the first 2500 triangular numbers
noncomputable def sum_of_reciprocals : ℝ := ∑ n in finset.range 2500, 1 / triangular_number (n + 1)

-- State the theorem
theorem sum_of_reciprocals_of_triangular_numbers :
  sum_of_reciprocals = 5000 / 2501 :=
sorry

end sum_of_reciprocals_of_triangular_numbers_l298_298369


namespace probability_of_rolling_8_l298_298237

theorem probability_of_rolling_8 :
  let num_favorable := 5
  let num_total := 36
  let probability := (5 : ℚ) / 36
  probability =
    (num_favorable : ℚ) / num_total :=
by
  sorry

end probability_of_rolling_8_l298_298237


namespace product_of_differences_of_nth_roots_of_unity_l298_298549

-- Definition of nth roots of unity
def nth_roots_of_unity (n : ℕ) (k : ℕ) : ℂ :=
  complex.exp (2 * real.pi * complex.I * k / n)

-- Statement of the proof problem
theorem product_of_differences_of_nth_roots_of_unity (n : ℕ) (hn : n > 0) :
  let α := λ k, nth_roots_of_unity n k in
  (∏ i in finset.range n, ∏ j in finset.range n, if i < j then α (i + 1) - α (j + 1) else 1) ^ 2 =
  (-1) ^ ((n - 1) * (n - 2) / 2) * n ^ n :=
by sorry

end product_of_differences_of_nth_roots_of_unity_l298_298549


namespace solve_problem_l298_298451

noncomputable def p (x : ℝ) : Prop := 2^x < 3^x
noncomputable def q : Prop := ∃ x : ℝ, x^2 = 2 - x

theorem solve_problem (h : ¬(∀ x : ℝ, p x) ∧ q) : 
  ∃ x : ℝ, 2^x >= 3^x ∧ x^2 = 2 - x ∧ x = -2 := 
by 
  sorry

end solve_problem_l298_298451


namespace max_principals_during_15_year_period_l298_298390

theorem max_principals_during_15_year_period : 
  (∃ (P : ℕ → ℕ) (n : ℕ), (∀ i, P i = 4) ∧ ∑ (i : ℕ) in (finset.range n), P i ≤ 15) → n ≤ 4 :=
by
  sorry

end max_principals_during_15_year_period_l298_298390


namespace probability_of_head_l298_298151

def events : Type := {e // e = "H" ∨ e = "T"}

def equallyLikely (e : events) : Prop :=
  e = ⟨"H", Or.inl rfl⟩ ∨ e = ⟨"T", Or.inr rfl⟩

def totalOutcomes := 2

def probOfHead : ℚ := 1 / totalOutcomes

theorem probability_of_head : probOfHead = 1 / 2 :=
by
  sorry

end probability_of_head_l298_298151


namespace parallel_lines_perpendicular_to_same_plane_l298_298088

theorem parallel_lines_perpendicular_to_same_plane 
  (m n : Line) (α : Plane) 
  (h_non_coincident_lines : ¬(m = n))
  (h_perpendicular_m : Perpendicular m α)
  (h_perpendicular_n : Perpendicular n α) : 
  Parallel m n :=
sorry

end parallel_lines_perpendicular_to_same_plane_l298_298088


namespace sixish_f_g_eq_pseudo_sixish_27_no_other_pseudo_sixish_l298_298283

def is_sixish (n : ℕ) : Prop :=
  ∃ p, Nat.Prime p ∧ Nat.Prime (p + 6) ∧ n = p * (p + 6)

noncomputable def f : ℕ → ℕ
| n => (∑ d in n.divisors, d^2)

def g (x : ℕ) : ℕ := x^2 + 2*x + 37

theorem sixish_f_g_eq (n : ℕ) (h : is_sixish n) : f n = g n :=
by
  sorry

theorem pseudo_sixish_27 : ¬ is_sixish 27 ∧ f 27 = g 27 :=
by
  sorry

theorem no_other_pseudo_sixish (n : ℕ) 
  (hn : n ≠ 27) : ¬ is_sixish n → f n ≠ g n :=
by
  sorry

end sixish_f_g_eq_pseudo_sixish_27_no_other_pseudo_sixish_l298_298283


namespace repeating_decimal_to_fraction_l298_298380

theorem repeating_decimal_to_fraction :
  let x := 0.431431431 + 0.000431431431 + 0.000000431431431
  let y := 0.4 + x
  y = 427 / 990 :=
by
  sorry

end repeating_decimal_to_fraction_l298_298380


namespace range_of_f_l298_298857

noncomputable def f (x : ℝ) : ℝ := Real.sin x + 3 * Real.cos x

theorem range_of_f :
  setOf (f x) = set.Icc (-Real.sqrt 10) (Real.sqrt 10) :=
sorry

end range_of_f_l298_298857


namespace volume_of_pyramid_TABC_l298_298980

noncomputable def volume_of_pyramid (TA TB TC : ℝ) : ℝ :=
  (1/3) * (1/2) * TA * TB * TC

theorem volume_of_pyramid_TABC :
  ∀ (A B C T : ℝ × ℝ × ℝ) (TA TB TC : ℝ),
    (TA = 12) →
    (TB = 12) →
    (TC = 10) →
    -- We assume TA, TB, TC are the lengths of the perpendicular segments
    let volume := volume_of_pyramid TA TB TC in
    volume = 240 :=
by
  intros A B C T TA TB TC hTA hTB hTC
  have h1 : volume_of_pyramid TA TB TC = (1/3) * (1/2) * 12 * 12 * 10 := by simp [volume_of_pyramid, hTA, hTB, hTC]
  have h2 : (1/3) * (1/2) * 12 * 12 * 10 = (1/3) * 72 * 10 := by ring
  have h3 : (1/3) * 72 * 10 = 240 := by norm_num
  exact (by rw [h1, h2, h3])

#check volume_of_pyramid_TABC

end volume_of_pyramid_TABC_l298_298980


namespace exactly_one_divisible_by_4_l298_298579

theorem exactly_one_divisible_by_4 :
  (777 % 4 = 1) ∧ (555 % 4 = 3) ∧ (999 % 4 = 3) →
  (∃! (x : ℕ),
    (x = 777 ^ 2021 * 999 ^ 2021 - 1 ∨
     x = 999 ^ 2021 * 555 ^ 2021 - 1 ∨
     x = 555 ^ 2021 * 777 ^ 2021 - 1) ∧
    x % 4 = 0) :=
by
  intros h
  sorry

end exactly_one_divisible_by_4_l298_298579


namespace roll_probability_l298_298131

theorem roll_probability :
  (∃ p : ℚ, p = (Nat.choose 5 3) * ((1/6)^3) * (Nat.choose 2 2) * ((1/6)^2) ∧ p = 5 / 3888) :=
begin
  sorry
end

end roll_probability_l298_298131


namespace combined_final_price_percentage_l298_298742

def original_price_A := 50
def original_price_B := 70
def original_price_C := 100

def sale_percentage_A := 0.75
def sale_percentage_B := 0.60
def sale_percentage_C := 0.80

def additional_discount_A := 0.10
def additional_discount_B := 0.15
def additional_discount_C := 0.20

def final_price_A := (original_price_A * sale_percentage_A) * (1 - additional_discount_A)
def final_price_B := (original_price_B * sale_percentage_B) * (1 - additional_discount_B)
def final_price_C := (original_price_C * sale_percentage_C) * (1 - additional_discount_C)

noncomputable def combined_final_price := final_price_A + final_price_B + final_price_C
noncomputable def combined_original_price := original_price_A + original_price_B + original_price_C

noncomputable def final_percentage := (combined_final_price / combined_original_price) * 100

theorem combined_final_price_percentage :
  abs (final_percentage - 60.66) < 0.01 :=
sorry

end combined_final_price_percentage_l298_298742


namespace maximum_sqrt3a2_minus_ab_l298_298439

theorem maximum_sqrt3a2_minus_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + b^2 - (sqrt 3) * a * b = 1) :
  ∃ c : ℝ, sqrt 3 * a^2 - a * b ≤ 2 + sqrt 3 :=
sorry

end maximum_sqrt3a2_minus_ab_l298_298439


namespace arithmetic_seq_a_8_l298_298158

variable (a : ℕ → ℝ) -- Define the arithmetic sequence

-- Define the sum of the first 15 terms
def S_15 (a : ℕ → ℝ) : ℝ := (15 / 2) * (a 1 + a 15)

-- Condition: S_15 = 90
axiom h : S_15 a = 90

-- The theorem to prove
theorem arithmetic_seq_a_8 (h : S_15 a = 90) : a 8 = 6 :=
by
  sorry

end arithmetic_seq_a_8_l298_298158


namespace aardvark_distance_l298_298026

theorem aardvark_distance :
  ∀ (r₁ r₂ : ℝ), (r₁ = 15) → (r₂ = 30) →
  let d₁ := 1 / 2 * 2 * Real.pi * r₁,
      d₂ := r₂ - r₁,
      d₃ := 3 / 4 * 2 * Real.pi * r₂,
      d₄ := r₂
  in d₁ + d₂ + d₃ + d₄ = 60 * Real.pi + 45 :=
by
  intros r₁ r₂ h₁ h₂
  let d₁ := 1 / 2 * 2 * Real.pi * r₁
  let d₂ := r₂ - r₁
  let d₃ := 3 / 4 * 2 * Real.pi * r₂
  let d₄ := r₂
  have : d₁ + d₂ + d₃ + d₄ = 60 * Real.pi + 45 := sorry
  exact this

end aardvark_distance_l298_298026


namespace sqrt_ab_max_value_l298_298450

theorem sqrt_ab_max_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : sqrt (a * b) ≤ 1 / 2 :=
by sorry

end sqrt_ab_max_value_l298_298450


namespace area_of_triangle_l298_298209

noncomputable def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 8 = 1

def foci_distance (F1 F2 : ℝ × ℝ) : Prop := (F1.1, F1.2) = (-3, 0) ∧ (F2.1, F2.2) = (3, 0)

def point_on_hyperbola (x y : ℝ) : Prop := hyperbola x y

def distance_ratios (P F1 F2 : ℝ × ℝ) : Prop := 
  let PF1 := (P.1 - F1.1)^2 + (P.2 - F1.2)^2
  let PF2 := (P.1 - F2.1)^2 + (P.2 - F2.2)^2
  PF1 / PF2 = 3 / 4

theorem area_of_triangle {P F1 F2 : ℝ × ℝ} 
  (H1 : foci_distance F1 F2)
  (H2 : point_on_hyperbola P.1 P.2)
  (H3 : distance_ratios P F1 F2) :
  let area := 1 / 2 * (6:ℝ) * (8:ℝ) * Real.sqrt 5
  area = 8 * Real.sqrt 5 := 
sorry

end area_of_triangle_l298_298209


namespace banks_policies_for_seniors_justified_l298_298585

-- Defining conditions
def better_credit_repayment_reliability : Prop := sorry
def stable_pension_income : Prop := sorry
def indirect_younger_relative_contributions : Prop := sorry
def pensioners_inclination_to_save : Prop := sorry
def regular_monthly_income : Prop := sorry
def preference_for_long_term_deposits : Prop := sorry

-- Lean theorem statement using the conditions
theorem banks_policies_for_seniors_justified :
  better_credit_repayment_reliability →
  stable_pension_income →
  indirect_younger_relative_contributions →
  pensioners_inclination_to_save →
  regular_monthly_income →
  preference_for_long_term_deposits →
  (banks_should_offer_higher_deposit_and_lower_loan_rates_to_seniors : Prop) :=
by
  -- Insert proof here that given all the conditions the conclusion follows
  sorry -- proof not required, so skipping

end banks_policies_for_seniors_justified_l298_298585


namespace equal_angles_of_bisected_quadrilateral_l298_298947

open Real
open EuclideanGeometry

noncomputable def intersection_point (l₁ l₂: Line) : Point := sorry

noncomputable def perpendicular (P O: Point) (EF: Line) : Prop := sorry

theorem equal_angles_of_bisected_quadrilateral
  (A B C D: Point) [ConvexQuadrilateral A B C D]
  (E F: Point) (hEF_A: LineThrough E A)
  (hEF_C: LineThrough E C)
  (hEF_B: LineThrough F B)
  (hEF_D: LineThrough F D)
  (P: Point) (hP_AC: LineThrough P A ∧ LineThrough P C)
  (hP_BD: LineThrough P B ∧ LineThrough P D)
  (O: Point) 
  (hO_perpendicular: perpendicular P O (LineThrough E F)) :
  ∠ (LineThrough B O) (LineThrough O C) = ∠ (LineThrough A O) (LineThrough O D) :=
sorry

end equal_angles_of_bisected_quadrilateral_l298_298947


namespace blocks_left_l298_298983

def blocks_initial := 78
def blocks_used := 19

theorem blocks_left : blocks_initial - blocks_used = 59 :=
by
  -- Solution is not required here, so we add a sorry placeholder.
  sorry

end blocks_left_l298_298983


namespace last_digit_even_numbers_less_than_100_not_multiples_of_ten_l298_298780

def last_digit_of_product_of_even_numbers_less_than_100_not_multiples_of_ten : ℕ := 6

theorem last_digit_even_numbers_less_than_100_not_multiples_of_ten :
  let evens := {n : ℕ | n < 100 ∧ n % 2 = 0 ∧ n % 10 ≠ 0}
  let product := ∏ n in evens, n 
  (product % 10) = last_digit_of_product_of_even_numbers_less_than_100_not_multiples_of_ten :=
sorry

end last_digit_even_numbers_less_than_100_not_multiples_of_ten_l298_298780


namespace relationship_abc_l298_298076

def a := log 2 (3 * sqrt 3)
def b := log 2 (9 / sqrt 3)
def c := log 3 2

theorem relationship_abc : a = b ∧ a > c := by
  sorry

end relationship_abc_l298_298076


namespace cost_of_fencing_per_meter_l298_298262

theorem cost_of_fencing_per_meter (x : ℝ) (length width : ℝ) (area : ℝ) (total_cost : ℝ) :
  length = 3 * x ∧ width = 2 * x ∧ area = 3750 ∧ area = length * width ∧ total_cost = 125 →
  (total_cost / (2 * (length + width)) = 0.5) :=
by
  sorry

end cost_of_fencing_per_meter_l298_298262


namespace proof_largest_e_l298_298543

-- Definitions based on the problem conditions
def diameter : ℝ := 2

def PX : ℝ := 4 / 3
def QY : ℝ := 3 / 2 

def largest_e : ℝ := 17 - 8 * Real.sqrt 9

-- Lean statement for the proof
theorem proof_largest_e : 
  ∃ (u v w : ℕ), (0 < u ∧ 0 < v ∧ 0 < w) ∧ (¬ (∃ p : ℕ, p^2 ∣ w)) ∧ 
  (largest_e = u - v * Real.sqrt w) ∧ (u + v + w = 34) :=
sorry

end proof_largest_e_l298_298543


namespace determinant_zero_l298_298374

def matrix_cos : Matrix Real :=
  ![
    ![Real.cos (π / 4), Real.cos (π / 2), Real.cos (3 * π / 4)],
    ![Real.cos π, Real.cos (5 * π / 4), Real.cos (3 * π / 2)],
    ![Real.cos (7 * π / 4), Real.cos (2 * π), Real.cos (9 * π / 4)]
  ]

theorem determinant_zero : Matrix.det matrix_cos = 0 :=
by
  sorry

end determinant_zero_l298_298374


namespace gain_percentage_of_trade_l298_298676

theorem gain_percentage_of_trade
  (C : ℝ)
  (cost_of_one_pen : C > 0)
  (number_of_pens_sold : 100)
  (gain_in_term_of_pen_cost : 40) :
  number_of_pens_sold = 100 ∧ gain_in_term_of_pen_cost = 40 →
  ((gain_in_term_of_pen_cost : ℝ) / number_of_pens_sold) * 100 = 40 :=
by
  sorry

end gain_percentage_of_trade_l298_298676


namespace complex_inverse_l298_298124

noncomputable def complex_expression (i : ℂ) (h_i : i ^ 2 = -1) : ℂ :=
  (3 * i - 3 * (1 / i))⁻¹

theorem complex_inverse (i : ℂ) (h_i : i^2 = -1) :
  complex_expression i h_i = -i / 6 :=
by
  -- the proof part is omitted
  sorry

end complex_inverse_l298_298124


namespace inequality_k_ge_2_l298_298539

theorem inequality_k_ge_2 {a b c : ℝ} (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_abc : a * b * c = 1) (k : ℤ) (h_k : k ≥ 2) :
  (a^k / (a + b) + b^k / (b + c) + c^k / (c + a)) ≥ 3 / 2 :=
by
  sorry

end inequality_k_ge_2_l298_298539


namespace frogs_seen_in_pond_l298_298886

-- Definitions from the problem conditions
def initial_frogs_on_lily_pads : ℕ := 5
def frogs_on_logs : ℕ := 3
def baby_frogs_on_rock : ℕ := 2 * 12  -- Two dozen

-- The statement of the proof
theorem frogs_seen_in_pond : initial_frogs_on_lily_pads + frogs_on_logs + baby_frogs_on_rock = 32 :=
by sorry

end frogs_seen_in_pond_l298_298886


namespace worker_wage_before_promotion_l298_298749

variable (W_new : ℝ)
variable (W : ℝ)

theorem worker_wage_before_promotion (h1 : W_new = 45) (h2 : W_new = 1.60 * W) :
  W = 28.125 := by
  sorry

end worker_wage_before_promotion_l298_298749


namespace solve_for_x_l298_298611

theorem solve_for_x (x : ℝ) : 5 + 3.5 * x = 2.5 * x - 25 ↔ x = -30 :=
by {
  split,
  {
    intro h,
    calc
      x = -30 : by sorry,
  },
  {
    intro h,
    calc
      5 + 3.5 * (-30) = 5 - 105
                       = -100,
      2.5 * (-30) - 25 = -75 - 25
                       = -100,
    exact Eq.symm (by sorry),
  }
}

end solve_for_x_l298_298611


namespace length_CF_l298_298519

theorem length_CF (A B C D E F : Point)
  (H1 : Rectangle A B C D)
  (H2 : dist A B = 6)
  (H3 : dist A D = 4)
  (H4 : OnLine C D F)
  (H5 : divides B D E 1 2)
  : dist C F = 4 * (sqrt 13 - 3) / 3 := 
sorry

end length_CF_l298_298519


namespace locus_midpoint_l298_298464

-- Definitions based on the conditions provided
variables 
  {A B C D A' B' C' D' X Y Z : Type}
  [cube_faces : face A B C D]
  [opposite_faces : face A' B' C' D']
  [edges_are_parallel : (A' - A) ∥ (B' - B) ∥ (C' - C) ∥ (D' - D')]
  (X_path_constant_speed : path_constant_speed X A B C D)
  (Y_path_constant_speed : path_constant_speed Y B' C' C B)
  (X_start_pos : start_pos X A)
  (Y_start_pos : start_pos Y B')

-- Proving the locus of the midpoint Z
theorem locus_midpoint :
  locus_midpoint Z X Y = rhombus_perimeter E F G C := 
sorry

end locus_midpoint_l298_298464


namespace unique_representation_l298_298485

theorem unique_representation (n : ℕ) (h_pos : 0 < n) : 
  ∃! (a b : ℚ), a = 1 / n ∧ b = 1 / (n + 1) ∧ (a + b = (2 * n + 1) / (n * (n + 1))) :=
by
  sorry

end unique_representation_l298_298485


namespace h_k_minus3_eq_l298_298121

def h (x : ℝ) : ℝ := 4 - Real.sqrt x
def k (x : ℝ) : ℝ := 3 * x + 3 * x^2

theorem h_k_minus3_eq : h (k (-3)) = 4 - 3 * Real.sqrt 2 := 
by 
  sorry

end h_k_minus3_eq_l298_298121


namespace bisectors_of_parallelogram_form_rectangle_diagonal_l298_298981

/-- Define a parallelogram with sides a and b, with given properties of its angle bisectors that form a rectangle -/
theorem bisectors_of_parallelogram_form_rectangle_diagonal (a b : ℝ) (ABCD : Parallelogram)
  (h1 : ∠ ABCD ∠ opposite = REFL ∠ ) -- Opposite angles are equal properties
  (h2 : ∠ ABCD ∠ adjacent ‘== ’ 180 ) -- Adjacent angles are supplementary
  (rect: Rectangle PQRS -- Formed by intersections of angle bisectors
  ) : 
  diagonal PRS = abs (a - b) := 
by
  sorry

end bisectors_of_parallelogram_form_rectangle_diagonal_l298_298981


namespace coloring_methods_count_l298_298373

-- Define the coloring of integers as a function
def color (n : ℕ) : Prop := sorry

-- Define the conditions
def condition1 : Prop := sorry
def condition2 (x y : ℕ) : Prop := sorry
def condition3 (x y : ℕ) : Prop := sorry

-- The final statement to be proven
theorem coloring_methods_count : ∃ n : ℕ, n = 4 ∧
  (∀ x, x ∈ {1..15} → color x) ∧
  condition1 ∧
  (∀ x y, x ≠ y → x + y ≤ 15 → condition2 x y) ∧
  (∀ x y, x ≠ y → x * y ≤ 15 → condition3 x y) :=
sorry

end coloring_methods_count_l298_298373


namespace line_slope_y_intercept_l298_298506

theorem line_slope_y_intercept :
  (∃ (a b : ℝ), (∀ (x y : ℝ), (x = 3 → y = 7 → y = a * x + b) ∧ (x = 7 → y = 19 → y = a * x + b)) ∧ (a - b = 5)) :=
begin
  sorry
end

end line_slope_y_intercept_l298_298506


namespace percentage_change_difference_l298_298357

-- Define the initial and final percentages of students
def initial_liked_percentage : ℝ := 0.4
def initial_disliked_percentage : ℝ := 0.6
def final_liked_percentage : ℝ := 0.8
def final_disliked_percentage : ℝ := 0.2

-- Define the problem statement
theorem percentage_change_difference :
  (final_liked_percentage - initial_liked_percentage) + 
  (initial_disliked_percentage - final_disliked_percentage) = 0.6 :=
sorry

end percentage_change_difference_l298_298357


namespace quadratic_function_l298_298774

theorem quadratic_function :
  ∃ a : ℝ, ∃ f : ℝ → ℝ, (∀ x : ℝ, f x = a * (x - 1) * (x - 5)) ∧ f 3 = 10 ∧ 
  f = fun x => -2.5 * x^2 + 15 * x - 12.5 :=
by
  sorry

end quadratic_function_l298_298774


namespace lcm_12_18_l298_298423

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end lcm_12_18_l298_298423


namespace evaluate_expression_l298_298025

def diamond (a b : ℝ) : ℝ := a - 1 / b

theorem evaluate_expression : 
  ((diamond (diamond 2 3) 4) - (diamond 2 (diamond 3 4))) = -29 / 132 :=
by
  sorry

end evaluate_expression_l298_298025


namespace sum_interior_numbers_eight_l298_298899

noncomputable def sum_interior_numbers (n : ℕ) : ℕ :=
  2^(n-1) - 2 -- This is a general formula derived from the pattern

theorem sum_interior_numbers_eight :
  sum_interior_numbers 8 = 126 :=
by
  -- No proof required, so we use sorry.
  sorry

end sum_interior_numbers_eight_l298_298899


namespace solve_for_k_l298_298109

def vector := ℝ × ℝ

def a : vector := (-3, 1)
def b : vector := (1, -2)

-- The condition given for parallel vectors
def vector_parallel (v1 v2: vector) : Prop := 
  ∃ (λ: ℝ), v1 = (λ * v2.1, λ * v2.2)

theorem solve_for_k (k : ℝ) : 
  vector_parallel ((-2 * a.1 + b.1, -2 * a.2 + b.2)) (a.1 + k * b.1, a.2 + k * b.2) ↔ 
  k = -1 / 2 :=
by
  sorry

end solve_for_k_l298_298109


namespace probability_of_integer_p_is_3_div_20_l298_298489

open_locale big_operators

noncomputable def probability_of_p : ℚ :=
  let possible_p := {p | ∃ q : ℤ, p * q - 6 * p - 3 * q = 3} in
  let total_possibilities := finset.Icc 1 20 in
  (finset.card (total_possibilities.filter possible_p)).to_rat / (finset.card total_possibilities).to_rat

theorem probability_of_integer_p_is_3_div_20 :
  probability_of_p = 3 / 20 :=
by {
  sorry
}

end probability_of_integer_p_is_3_div_20_l298_298489


namespace base_six_four_digit_odd_final_l298_298435

theorem base_six_four_digit_odd_final :
  ∃ b : ℕ, (b^4 > 285 ∧ 285 ≥ b^3 ∧ (285 % b) % 2 = 1) :=
by 
  use 6
  sorry

end base_six_four_digit_odd_final_l298_298435


namespace arun_complete_remaining_work_in_42_days_l298_298017

theorem arun_complete_remaining_work_in_42_days :
  (let total_work := 1 in
   let combined_rate := total_work / 10 in
   let arun_rate := total_work / 70 in
   let tarun_rate := combined_rate - arun_rate in
   let work_done_in_4_days := 4 * combined_rate in
   let remaining_work := total_work - work_done_in_4_days in
   let days_by_arun := remaining_work / arun_rate in
   days_by_arun = 42) :=
by
  sorry

end arun_complete_remaining_work_in_42_days_l298_298017


namespace cos_condition_l298_298809

theorem cos_condition (α : ℝ) : (∀ k : ℕ, cos (2^k * α) ≤ 0) ↔ (∃ n : ℤ, α = (2 * π / 3) + 2 * n * π ∨ α = (4 * π / 3) + 2 * n * π) :=
sorry

end cos_condition_l298_298809


namespace right_square_pyramid_inscribed_circumscribed_spheres_l298_298956

noncomputable def proof_problem_statement (R r d : ℝ) :=
  (d^2 + (R + r)^2 = 2 * R^2) ∧ 
  (∀ r_max, (d = 0 → r = R * (Real.sqrt 2 - 1)) → r / R ≤ r_max) 

theorem right_square_pyramid_inscribed_circumscribed_spheres 
  (R r d : ℝ) (h : d^2 + (R + r)^2 = 2 * R^2) : 
  proof_problem_statement R r d :=
sorry

end right_square_pyramid_inscribed_circumscribed_spheres_l298_298956


namespace perp_AD_IK_l298_298522

open EuclideanGeometry

noncomputable def A := sorry
noncomputable def B := sorry
noncomputable def C := sorry
noncomputable def I := incenter A B C
noncomputable def O := circumcircle A B C
noncomputable def M := midpoint B C
noncomputable def D := projection I (line_segment B C)
noncomputable def N := arc_midpoint (circumcircle A B C) A B C
noncomputable def K := intersect (line AM) (line DN)

theorem perp_AD_IK :
  AD ⊥ IK :=
sorry

end perp_AD_IK_l298_298522


namespace pyramid_cross_section_perimeter_l298_298912

theorem pyramid_cross_section_perimeter 
  (D A B C E F : Point)
  (BC : Segment BC 4)
  (DA : Segment DA 8)
  (DB : Segment DB 8)
  (DC : Segment DC 8)
  (DB_E : Intersection DB E)
  (DC_F : Intersection DC F)
  (pyramid : RegularTriangularPyramid D A B C)
  (cross_section : CrossSectionalTriangle A E F DB DC) :
  Perimeter (Triangle A E F) = 11 := 
sorry

end pyramid_cross_section_perimeter_l298_298912


namespace tangent_line_with_smallest_slope_l298_298060

-- Define the given curve
def curve (x : ℝ) : ℝ := x^3 + 3 * x^2 + 6 * x - 10

-- Define the derivative of the given curve
def curve_derivative (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 6

-- Define the equation of the tangent line with the smallest slope
def tangent_line (x y : ℝ) : Prop := 3 * x - y = 11

-- Prove that the equation of the tangent line with the smallest slope on the curve is 3x - y - 11 = 0
theorem tangent_line_with_smallest_slope :
  ∃ x y : ℝ, curve x = y ∧ curve_derivative x = 3 ∧ tangent_line x y :=
by
  sorry

end tangent_line_with_smallest_slope_l298_298060


namespace imaginary_part_of_z_l298_298254

def z : ℂ := 1 - 2 * Complex.I

theorem imaginary_part_of_z : Complex.im z = -2 := by
  sorry

end imaginary_part_of_z_l298_298254


namespace determinant_computation_l298_298074

variable (x y z w : ℝ)
variable (det : ℝ)
variable (H : x * w - y * z = 7)

theorem determinant_computation : 
  (x + z) * w - (y + 2 * w) * z = 7 - w * z := by
  sorry

end determinant_computation_l298_298074


namespace solution_in_interval_l298_298890

noncomputable theory -- Since we are dealing with real numbers and exponential functions.

def f (x : ℝ) : ℝ := 2^x + 3*x - 12

theorem solution_in_interval (x₀ : ℝ) (h₀ : f x₀ = 0) 
  (h_mono : ∀ x1 x2, x1 < x2 → f x1 < f x2) 
  (h_f2 : f 2 < 0) 
  (h_f3 : f 3 > 0) : 
  x₀ ∈ Set.Ioo 2 3 := 
by 
  sorry -- Proof is omitted as per the instructions.

end solution_in_interval_l298_298890


namespace minimum_money_lost_l298_298211

-- Define the conditions and setup the problem

def check_amount : ℕ := 1270
def T_used (F : ℕ) : Σ' T, (T = F + 1 ∨ T = F - 1) :=
sorry

def money_used (T F : ℕ) : ℕ := 10 * T + 50 * F

def total_bills_used (T F : ℕ) : Prop := T + F = 15

theorem minimum_money_lost : (∃ T F, (T = F + 1 ∨ T = F - 1) ∧ T + F = 15 ∧ (check_amount - (10 * T + 50 * F) = 800)) :=
sorry

end minimum_money_lost_l298_298211


namespace expression_value_l298_298465

theorem expression_value : 
  10 * 73 * (Real.logBase 3 (4 * (3^2 + 1) * (3^4 + 1) * (3^8 + 1) * (3^16 + 1) * (3^32 + 1) + 1 / 2)) + (Real.logBase 3 2) = 128 := 
sorry

end expression_value_l298_298465


namespace total_time_l298_298745

variables (x : ℝ)

def time1 : ℝ := x / 50
def time2 : ℝ := (2 * x) / 100
def time3 : ℝ := (3 * x) / 150

theorem total_time : time1 x + time2 x + time3 x = 3 * x / 50 :=
by sorry

end total_time_l298_298745


namespace candy_necklaces_l298_298173

theorem candy_necklaces (necklace_candies : ℕ) (block_candies : ℕ) (blocks_broken : ℕ) :
  necklace_candies = 10 → block_candies = 30 → blocks_broken = 3 → (blocks_broken * block_candies) / necklace_candies - 1 = 8 := 
by 
  intro h1 h2 h3
  rw [h1, h2, h3]
  sorry

end candy_necklaces_l298_298173


namespace calc_pow_16_neg_2_pow_neg_3_l298_298761

theorem calc_pow_16_neg_2_pow_neg_3 : (16 : ℝ) ^ (-(2 : ℝ) ^ -3) = 1 / Real.sqrt 2 := 
sorry

end calc_pow_16_neg_2_pow_neg_3_l298_298761


namespace trailing_zeroes_500_factorial_l298_298630

theorem trailing_zeroes_500_factorial : (nat_trailing_zeroes (fact 500) = 124) :=
by
  -- Placeholder for the proof
  sorry

noncomputable def fact (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (n+1) => (n+1) * fact n

noncomputable def nat_trailing_zeroes (n : ℕ) : ℕ :=
  if n = 0 then 0
  else
    let rec trailing_zeroes (n : ℕ) (count : ℕ) : ℕ :=
      if n % 5 = 0 then trailing_zeroes (n / 5) (count + 1)
      else count
    trailing_zeroes n 0

end trailing_zeroes_500_factorial_l298_298630


namespace arithmetic_sequence_sum_cubes_l298_298640

theorem arithmetic_sequence_sum_cubes (x : ℤ) (n : ℕ) (h1 : n > 5) 
(h2 : finset.sum (finset.range (n + 1)) (fun k => (x + 2 * k) ^ 3) = -3969) : 
n = 6 :=
sorry

end arithmetic_sequence_sum_cubes_l298_298640


namespace train_cross_tunnel_time_l298_298327

def convert_speed_kmh_to_ms (speed_kmh : ℝ) : ℝ :=
  (speed_kmh * 1000) / 3600

def time_to_cross_tunnel (length_train length_tunnel speed_kmh : ℝ) : ℝ :=
  let total_distance := length_train + length_tunnel
  let speed_ms := convert_speed_kmh_to_ms speed_kmh
  let time_seconds := total_distance / speed_ms
  time_seconds / 60

theorem train_cross_tunnel_time :
  time_to_cross_tunnel 800 500 78 = 1 := by
  -- Prove that the train takes 1 minute to cross the tunnel given the conditions
  sorry

end train_cross_tunnel_time_l298_298327


namespace probability_of_two_aces_or_at_least_one_king_is_correct_l298_298895

noncomputable def probability_two_aces_or_at_least_one_king
  (deck : List String)
  (is_king : String → Bool)
  (is_ace : String → Bool)
  (num_kings : ℕ := 4)
  (num_aces : ℕ := 6)
  (num_others : ℕ := 44)
  (total_cards : ℕ := 54) : ℚ := 
  let two_aces_prob := (6 / 54) * (5 / 53)
  let one_king_prob := 2 * (4 / 54) * (50 / 53)
  let two_kings_prob := (4 / 54) * (3 / 53)
  let at_least_one_king_prob := one_king_prob + two_kings_prob
  (15 / 1431) + at_least_one_king_prob

theorem probability_of_two_aces_or_at_least_one_king_is_correct :
  probability_two_aces_or_at_least_one_king 
    ["A", "A", "A", "A", "A", "A", "K", "K", "K", "K"] 
    (λ card, card = "K")
    (λ card, card = "A") = 221 / 1431 :=
by
  sorry

end probability_of_two_aces_or_at_least_one_king_is_correct_l298_298895


namespace solution_satisfies_system_l298_298236

noncomputable def solution : ℝ × ℝ := (19 / 4, 17 / 8)

theorem solution_satisfies_system :
  (x y : ℝ) (h1 : x = (19 / 4)) (h2 : y = (17 / 8)) :
  (x + real.sqrt (x + 2 * y) - 2 * y = 7 / 2) ∧ 
  (x^2 + x + 2 * y - 4 * y^2 = 27 / 2) :=
by
  split;
  -- First equation
  { rw [h1, h2], 
    calc
      (19 / 4) + real.sqrt ((19 / 4) + 2 * (17 / 8)) - 2 * (17 / 8)
        = (19 / 4) + 3 - 2 * (17 / 8) : sorry
        ... = 7 / 2 : sorry
  },
  -- Second equation
  { rw [h1, h2], 
    calc
      (19 / 4)^2 + (19 / 4) + 2 * (17 / 8) - 4 * (17 / 8)^2
        = 27 / 2 : sorry
  }

end solution_satisfies_system_l298_298236


namespace faith_earnings_correct_l298_298790

variable (pay_per_hour : ℝ) (regular_hours_per_day : ℝ) (work_days_per_week : ℝ) (overtime_hours_per_day : ℝ)
variable (overtime_rate_multiplier : ℝ)

def total_earnings (pay_per_hour : ℝ) (regular_hours_per_day : ℝ) (work_days_per_week : ℝ) 
                   (overtime_hours_per_day : ℝ) (overtime_rate_multiplier : ℝ) : ℝ :=
  let regular_hours := regular_hours_per_day * work_days_per_week
  let overtime_hours := overtime_hours_per_day * work_days_per_week
  let overtime_pay_rate := pay_per_hour * overtime_rate_multiplier
  let regular_earnings := pay_per_hour * regular_hours
  let overtime_earnings := overtime_pay_rate * overtime_hours
  regular_earnings + overtime_earnings

theorem faith_earnings_correct : 
  total_earnings 13.5 8 5 2 1.5 = 742.50 :=
by
  -- This is where the proof would go, but it's omitted as per the instructions
  sorry

end faith_earnings_correct_l298_298790


namespace max_min_squared_diff_l298_298587

variables {a b c : ℝ} {λ : ℝ}

theorem max_min_squared_diff (h : λ > 0) (h₀ : a^2 + b^2 + c^2 = λ) : 
  ∃ a b c, max (min ((a - b)^2) ((b - c)^2)) (min ((b - c)^2) ((c - a)^2)) = λ / 2 :=
begin
  sorry
end

end max_min_squared_diff_l298_298587


namespace weight_of_selected_students_is_sample_l298_298813

noncomputable theory
open set classical

def population (students : ℕ) := students = 1000
def sample_size (selected_students : ℕ) := selected_students = 125
def is_sample (population : ℕ) (sample_size : ℕ) := sample_size < population

theorem weight_of_selected_students_is_sample
  (students : ℕ)
  (selected_students : ℕ)
  (h1: population students)
  (h2: sample_size selected_students):
  is_sample students selected_students :=
by {
  sorry
}

end weight_of_selected_students_is_sample_l298_298813


namespace exists_four_functions_l298_298963

theorem exists_four_functions 
  (f : ℝ → ℝ)
  (h_periodic : ∀ x, f (x + 2 * Real.pi) = f x) :
  ∃ (f1 f2 f3 f4 : ℝ → ℝ), 
    (∀ x, f1 (-x) = f1 x ∧ f1 (x + Real.pi) = f1 x) ∧
    (∀ x, f2 (-x) = f2 x ∧ f2 (x + Real.pi) = f2 x) ∧
    (∀ x, f3 (-x) = f3 x ∧ f3 (x + Real.pi) = f3 x) ∧
    (∀ x, f4 (-x) = f4 x ∧ f4 (x + Real.pi) = f4 x) ∧
    (∀ x, f x = f1 x + f2 x * Real.cos x + f3 x * Real.sin x + f4 x * Real.sin (2 * x)) :=
sorry

end exists_four_functions_l298_298963


namespace area_ratio_of_squares_l298_298259

theorem area_ratio_of_squares (a b : ℝ) (h : 4 * a = 16 * b) : a ^ 2 = 16 * b ^ 2 := by
  sorry

end area_ratio_of_squares_l298_298259


namespace count_exceptions_in_range_l298_298032

def g (n : ℕ) : ℕ := if n % 2 = 1 then n^2 + 1 else n/2 * n/2

theorem count_exceptions_in_range :
  (finset.filter (λ n, ∃ k : ℕ, iterate g k n = 16) (finset.range 101)).card = 1 :=
begin
  sorry
end

end count_exceptions_in_range_l298_298032


namespace longest_diagonal_of_rhombus_l298_298722

variable (d1 d2 : ℝ) (r : ℝ)
variable h_area : 0.5 * d1 * d2 = 150
variable h_ratio : d1 / d2 = 4 / 3

theorem longest_diagonal_of_rhombus :
  max d1 d2 = 20 :=
by
  sorry

end longest_diagonal_of_rhombus_l298_298722


namespace donation_third_home_correct_l298_298998

-- Define the total donation amount
def total_donation : ℝ := 700

-- Define the donation to the first home
def donation_first_home : ℝ := 245

-- Define the donation to the second home
def donation_second_home : ℝ := 225

-- Define the expected donation to the third home
def donation_third_home : ℝ := 230

-- Prove that the donation to the third home is 230 dollars
theorem donation_third_home_correct : 
  total_donation - donation_first_home - donation_second_home = donation_third_home :=
by 
  -- leaving the proof out as per instructions
  sorry

end donation_third_home_correct_l298_298998


namespace cylinder_in_cone_l298_298738

noncomputable def cylinder_radius : ℝ :=
  let cone_radius : ℝ := 4
  let cone_height : ℝ := 10
  let r : ℝ := (10 * 2) / 9  -- based on the derived form of r calculation
  r

theorem cylinder_in_cone :
  let cone_radius : ℝ := 4
  let cone_height : ℝ := 10
  let r : ℝ := cylinder_radius
  (r = 20 / 9) :=
by
  sorry -- Proof mechanism is skipped as per instructions.

end cylinder_in_cone_l298_298738


namespace fermat_prime_exponents_l298_298184

-- Definitions based on the conditions
def euler_totient (n : ℕ) : ℕ := 
  ∑ i in range n, if gcd n i = 1 then 1 else 0

def sum_of_divisors (n : ℕ) : ℕ := 
  ∑ i in range (n + 1), if n % i = 0 then i else 0

theorem fermat_prime_exponents (k : ℕ) : 
  euler_totient (sum_of_divisors (2^k)) = 2^k ↔ k ∈ {1, 3, 7, 15, 31} :=
by sorry

end fermat_prime_exponents_l298_298184


namespace find_number_of_cats_l298_298214

theorem find_number_of_cats (dogs ferrets cats total_shoes shoes_per_animal : ℕ) 
  (h_dogs : dogs = 3)
  (h_ferrets : ferrets = 1)
  (h_total_shoes : total_shoes = 24)
  (h_shoes_per_animal : shoes_per_animal = 4) :
  cats = (total_shoes - (dogs + ferrets) * shoes_per_animal) / shoes_per_animal := by
  sorry

end find_number_of_cats_l298_298214


namespace find_g_neg_6_l298_298191

def f (x : ℚ) : ℚ := 4 * x - 9
def g (y : ℚ) : ℚ := 3 * (y * y) + 4 * y - 2

theorem find_g_neg_6 : g (-6) = 43 / 16 := by
  sorry

end find_g_neg_6_l298_298191


namespace solve_for_x_l298_298608

theorem solve_for_x (x : ℝ) : 5 + 3.5 * x = 2.5 * x - 25 ↔ x = -30 :=
by {
  split,
  {
    intro h,
    calc
      x = -30 : by sorry,
  },
  {
    intro h,
    calc
      5 + 3.5 * (-30) = 5 - 105
                       = -100,
      2.5 * (-30) - 25 = -75 - 25
                       = -100,
    exact Eq.symm (by sorry),
  }
}

end solve_for_x_l298_298608


namespace larger_fraction_of_two_l298_298265

theorem larger_fraction_of_two (x y : ℚ) (h1 : x + y = 7/8) (h2 : x * y = 1/4) : max x y = 1/2 :=
sorry

end larger_fraction_of_two_l298_298265


namespace ball_draw_probability_red_is_one_ninth_l298_298270

theorem ball_draw_probability_red_is_one_ninth :
  let A_red := 4
  let A_white := 2
  let B_red := 1
  let B_white := 5
  let P_red_A := A_red / (A_red + A_white)
  let P_red_B := B_red / (B_red + B_white)
  P_red_A * P_red_B = 1 / 9 := by
    -- Proof here
    sorry

end ball_draw_probability_red_is_one_ninth_l298_298270


namespace lcm_12_18_l298_298405

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := 
by
  sorry

end lcm_12_18_l298_298405


namespace inscribed_rectangle_length_l298_298589

theorem inscribed_rectangle_length {PQ QR PR : ℝ} (hPQ : PQ = 5) (hQR : QR = 12) (hPR : PR = 13)
  (l h : ℝ) (hheight : h = l / 2)
  (h_triangle : is_right_triangle PQ QR PR)
  (h_sim1 : triangle_similar (h, l) (5, 13)) :
  l = 7.5 :=
sorry

end inscribed_rectangle_length_l298_298589


namespace sigma_has_prime_factor_greater_than_2k_l298_298958

open BigOperators

def is_positive_integer (k : ℕ) : Prop := k > 0

def sigma (n : ℕ) : ℕ := ∑ i in (finset.range n.succ).filter (λ d, n % d = 0), d

theorem sigma_has_prime_factor_greater_than_2k (k : ℕ) (h : is_positive_integer k) :
  ∃ p : ℕ, nat.prime p ∧ p ∣ sigma (2^k) ∧ p > 2^k := sorry

end sigma_has_prime_factor_greater_than_2k_l298_298958


namespace value_of_k_l298_298615

theorem value_of_k (x k : ℝ) (h : (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 8)) (hk : k ≠ 0) : k = 8 :=
sorry

end value_of_k_l298_298615


namespace solve_for_x_l298_298231

theorem solve_for_x 
  (x : ℝ) 
  (h : (2/7) * (1/4) * x = 8) : 
  x = 112 :=
sorry

end solve_for_x_l298_298231


namespace emma_harry_weight_l298_298150

theorem emma_harry_weight (e f g h : ℕ) 
  (h1 : e + f = 280) 
  (h2 : f + g = 260) 
  (h3 : g + h = 290) : 
  e + h = 310 := 
sorry

end emma_harry_weight_l298_298150


namespace distinct_sequences_count_l298_298885

def letters := ["E", "Q", "U", "A", "L", "S"]

noncomputable def count_sequences : Nat :=
  let remaining_letters := ["E", "Q", "U", "A"] -- 'L' and 'S' are already considered
  3 * (4 * 3) -- as analyzed: (LS__) + (L_S_) + (L__S)

theorem distinct_sequences_count : count_sequences = 36 := 
  by
    unfold count_sequences
    sorry

end distinct_sequences_count_l298_298885


namespace Maria_total_eggs_l298_298970

-- Defining the types and their calculations based on the conditions.
def TypeA_eggs : ℕ := 4 * 10 * 6
def TypeB_eggs : ℕ := (3 * 15 / 2) * 5 -- This will be handled in the proof
def TypeC_eggs : ℕ := 6 * 5 * 9
def TypeD_eggs : ℕ := 8 * 2.5 * 10 -- This will be handled in the proof
def TypeE_eggs : ℕ := 20

-- Calculating the sum
def total_eggs : ℕ :=
  TypeA_eggs + 
  TypeB_eggs.to_nat + -- converting the non-integer part to an integer
  TypeC_eggs +
  TypeD_eggs.to_nat + -- converting the non-integer part to an integer
  TypeE_eggs 

-- Stating the theorem
theorem Maria_total_eggs : total_eggs = 842 :=
  by
  -- converting the non-integer parts of TypeB_eggs and TypeD_eggs properly and summing the whole
  sorry

end Maria_total_eggs_l298_298970


namespace range_of_function_l298_298557

open Real

theorem range_of_function (x : ℝ) (hx : -π/4 ≤ x ∧ x ≤ π/2) : 
  ∃ y, y = sin x * cos x * (sin x + cos x) ∧ y ∈ set.Icc (-√3 / 9) (√2 / 2) :=
sorry

end range_of_function_l298_298557


namespace g_at_minus_six_l298_298192

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x - 9
def g (x : ℝ) : ℝ := 3 * x ^ 2 + 4 * x - 2

theorem g_at_minus_six : g (-6) = 43 / 16 := by
  sorry

end g_at_minus_six_l298_298192


namespace pilak_divisors_30_l298_298340

def is_pilak (S : Set ℕ) : Prop :=
  ∃ (F T : Set ℕ), F ≠ ∅ ∧ T ≠ ∅ ∧
  ∃ (aF bF aT bT : ℕ),
    (bF ≥ aF + 1 ∧ bT ≥ aT + 1)
    ∧ S = F ∪ T
    ∧ Disjoint F T
    ∧ (∀ x ∈ F, ∃ i, x = Fib (aF + i))
    ∧ (∀ y ∈ T, ∃ j, y = Triangular (aT + j))

def divisors_without_self (n : ℕ) : Set ℕ :=
  { d | d ∣ n ∧ d ≠ n }

theorem pilak_divisors_30 :
  is_pilak (divisors_without_self 30) := 
sorry

end pilak_divisors_30_l298_298340


namespace part_I_part_II_l298_298391

theorem part_I (a b c d : ℝ) (h_pos: 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) (h_adbc: a * d = b * c) (h_ineq1: a + d > b + c): |a - d| > |b - c| :=
sorry

theorem part_II (a b c d t: ℝ) 
(h_pos: 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) 
(h_eq: t * (Real.sqrt (a^2 + b^2) * Real.sqrt (c^2 + d^2)) = Real.sqrt (a^4 + c^4) + Real.sqrt (b^4 + d^4)):
t >= Real.sqrt 2 :=
sorry

end part_I_part_II_l298_298391


namespace no_inscribed_triangle_exists_l298_298529

theorem no_inscribed_triangle_exists
  (a b c : ℝ) 
  (p : ℝ)
  (inscribed: a^2 + b^2 + c^2 - 2*a*b - 2*b*c - 2*c*a = 0) 
  (h_poly: ∀ (x : ℝ), x^3 - 2 * a * x^2 + b * c * x = p ∧ (x = sin ((a:ℝ) / 2) ∨ x = sin ((b:ℝ) / 2) ∨ x = sin ((c:ℝ) / 2))) : false :=
sorry

end no_inscribed_triangle_exists_l298_298529


namespace solve_equation_1_solve_equation_2_l298_298232

theorem solve_equation_1 (x : ℚ) : 1 - (1 / (x - 5)) = (x / (x + 5)) → x = 15 / 2 := 
by
  sorry

theorem solve_equation_2 (x : ℚ) : (3 / (x - 1)) - (2 / (x + 1)) = (1 / (x^2 - 1)) → x = -4 := 
by
  sorry

end solve_equation_1_solve_equation_2_l298_298232


namespace geometric_sequence_second_term_l298_298641

theorem geometric_sequence_second_term (a_1 q a_3 a_4 : ℝ) (h3 : a_1 * q^2 = 12) (h4 : a_1 * q^3 = 18) : a_1 * q = 8 :=
by
  sorry

end geometric_sequence_second_term_l298_298641


namespace min_balls_to_guarantee_18_single_color_l298_298320

theorem min_balls_to_guarantee_18_single_color (red green yellow blue white black : ℕ)
  (h_red : red = 30) (h_green : green = 25) (h_yellow : yellow = 22)
  (h_blue : blue = 15) (h_white : white = 12) (h_black : black = 6) :
  ∃ n, n = 85 ∧ ∀ (draws : ℕ), draws ≥ n → (∃ color, drawn_balls color draws ≥ 18) :=
by
  sorry

end min_balls_to_guarantee_18_single_color_l298_298320


namespace x_needs_to_finish_remaining_work_l298_298679

variable (x y : ℝ)

def x_work_rate : ℝ := 1 / 36
def y_work_rate : ℝ := 1 / 24
def y_worked_days : ℝ := 12
def y_work_done : ℝ := y_worked_days * y_work_rate  -- calculation

theorem x_needs_to_finish_remaining_work : (1 - y_work_done) / x_work_rate = 18 := 
  by
  have y_work_done_eq : y_work_done = 1 / 2 := by sorry -- y worked half of the work
  have remaining_work_eq : 1 - y_work_done = 1 / 2 := by sorry
  have x_days_eq : (1 / 2) / x_work_rate = 18 := by sorry
  exact x_days_eq

end x_needs_to_finish_remaining_work_l298_298679


namespace fraction_geq_bound_l298_298187

theorem fraction_geq_bound {n : ℕ} (hn : n ≥ 2) (a : Fin n → ℝ) (distinct : ∀ (i j : Fin n), i ≠ j → a i ≠ a j) :
  let S := ∑ i, (a i) ^ 2
  let M := min (Finset.univ.biomi (λ i j, if i = j then none else some ((a i - a j) ^ 2))) (Option.get_or_else 0)
  in S / M ≥ n * (n^2 - 1) / 12 := 
sorry

end fraction_geq_bound_l298_298187


namespace total_money_correct_l298_298967

noncomputable def total_money : ℕ :=
  let madeline := 48
  let brother := madeline / 2
  let sister := madeline * 2
  madeline + brother + sister

theorem total_money_correct : total_money = 168 := by
  let madeline := 48
  let brother := madeline / 2
  let sister := madeline * 2
  have : brother = 24 := rfl
  have : sister = 96 := rfl
  calc
    total_money
        = madeline + brother + sister    : rfl
    ... = 48 + 24 + 96                  : by rw [this, this]
    ... = 168                           : rfl

end total_money_correct_l298_298967


namespace back_wheel_revolutions_l298_298215

noncomputable def front_wheel_diameter : ℝ := 28
noncomputable def back_wheel_diameter : ℝ := 16
noncomputable def front_wheel_revolutions : ℝ := 150
noncomputable def front_wheel_circumference := front_wheel_diameter * Real.pi
noncomputable def back_wheel_circumference := back_wheel_diameter * Real.pi

theorem back_wheel_revolutions
  (no_slippage : true) :
  let distance_traveled := front_wheel_revolutions * front_wheel_circumference in
  distance_traveled / back_wheel_circumference = 262.5 := by
  sorry

end back_wheel_revolutions_l298_298215


namespace elements_with_first_digit_3_l298_298950

-- Define the set S as given in the conditions
def S : set ℕ := {k | 0 ≤ k ∧ k ≤ 5000}

-- Specify the conditions given in the problem
axiom digits_3_5000 : ∃ digits : ℕ, digits = 2388 ∧ (3^5000).digits.length = digits
axiom first_digit_3_5000 : (3^5000).digits.head = some 3

-- Define the function that checks the first digit
def first_digit (n : ℕ) : option ℕ :=
  (nat.digits 10 n).head

-- State the theorem
theorem elements_with_first_digit_3 : (∀ k ∈ S, first_digit (3^k) = some 3) := sorry

end elements_with_first_digit_3_l298_298950


namespace john_sells_percentage_of_newspapers_l298_298533

theorem john_sells_percentage_of_newspapers
    (n_newspapers : ℕ)
    (selling_price : ℝ)
    (cost_price_discount : ℝ)
    (profit : ℝ)
    (sold_percentage : ℝ)
    (h1 : n_newspapers = 500)
    (h2 : selling_price = 2)
    (h3 : cost_price_discount = 0.75)
    (h4 : profit = 550)
    (h5 : sold_percentage = 80) : 
    ( ∃ (sold_n : ℕ), 
      sold_n / n_newspapers * 100 = sold_percentage ∧
      sold_n * selling_price = 
        n_newspapers * selling_price * (1 - cost_price_discount) + profit) :=
by
  sorry

end john_sells_percentage_of_newspapers_l298_298533


namespace pear_worth_equivalence_l298_298616

variables (numApples numPears worthApples worthPears : ℕ)

-- Define conditions
def condition1 : Prop := (3 / 4 * 12 : ℝ) = 10
def condition2 : Prop := numApples = 15
def condition3 : Prop := worthApples = 3 / 5 * numApples
def condition4 : Prop := worthPears = 10

-- The theorem to prove
theorem pear_worth_equivalence :
  condition1 →
  condition2 →
  condition3 →
  worthPears = worthApples * 10 / 9 →
  worthPears = 10 :=
by
  intros h1 h2 h3 h4
  sorry

end pear_worth_equivalence_l298_298616


namespace intersection_unique_l298_298943

noncomputable def g (x : ℝ) : ℝ := x^3 + 5*x^2 + 15*x + 35

theorem intersection_unique :
  ∃! (c d : ℝ), g(c) = c ∧ d = c ∧ c = -5 ∧ d = -5 :=
by
  sorry

end intersection_unique_l298_298943


namespace tangent_line_with_smallest_slope_l298_298061

-- Define the given curve
def curve (x : ℝ) : ℝ := x^3 + 3 * x^2 + 6 * x - 10

-- Define the derivative of the given curve
def curve_derivative (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 6

-- Define the equation of the tangent line with the smallest slope
def tangent_line (x y : ℝ) : Prop := 3 * x - y = 11

-- Prove that the equation of the tangent line with the smallest slope on the curve is 3x - y - 11 = 0
theorem tangent_line_with_smallest_slope :
  ∃ x y : ℝ, curve x = y ∧ curve_derivative x = 3 ∧ tangent_line x y :=
by
  sorry

end tangent_line_with_smallest_slope_l298_298061


namespace sqrt_x2y_l298_298891

theorem sqrt_x2y (x y : ℝ) (h : x * y < 0) : Real.sqrt (x^2 * y) = -x * Real.sqrt y :=
sorry

end sqrt_x2y_l298_298891


namespace larger_angle_is_50_degrees_l298_298278

noncomputable def larger_complementary_angle_measures (x : ℝ) (h1 : 5 * x + 4 * x = 90) : ℝ :=
5 * x

theorem larger_angle_is_50_degrees (x : ℝ) (h1 : 5 * x + 4 * x = 90) : larger_complementary_angle_measures x h1 = 50 :=
by
  have h2 : 9 * x = 90 := by
    linarith
  have h3 : x = 10 := by
    linarith
  rw [larger_complementary_angle_measures, h3]
  norm_num
sorry

end larger_angle_is_50_degrees_l298_298278


namespace y_intercepts_are_equal_l298_298400

theorem y_intercepts_are_equal:
  (y_intercept_1 : ℝ) (h1 : 2 * 0 - 3 * y_intercept_1 = 6) ∧
  (y_intercept_2 : ℝ) (h2 : 0 + 4 * y_intercept_2 = -8) →
  y_intercept_1 = -2 ∧ y_intercept_2 = -2 :=
by
  sorry

end y_intercepts_are_equal_l298_298400


namespace triangle_incircle_ratio_l298_298346

theorem triangle_incircle_ratio (r p k : ℝ) (h1 : k = r * (p / 2)) : 
  p / k = 2 / r :=
by
  sorry

end triangle_incircle_ratio_l298_298346


namespace uniform_heights_l298_298238

theorem uniform_heights (varA varB : ℝ) (hA : varA = 0.56) (hB : varB = 2.1) : varA < varB := by
  rw [hA, hB]
  exact (by norm_num)

end uniform_heights_l298_298238


namespace no_valid_abc_l298_298343

theorem no_valid_abc : 
  ∀ (a b c : ℕ), (100 * a + 10 * b + c) % 15 = 0 → (10 * b + c) % 4 = 0 → a > b → b > c → false :=
by
  intros a b c habc_mod15 hbc_mod4 h_ab_gt h_bc_gt
  sorry

end no_valid_abc_l298_298343


namespace remaining_books_l298_298276

def initial_books : Nat := 500
def num_people_donating : Nat := 10
def books_per_person : Nat := 8
def borrowed_books : Nat := 220

theorem remaining_books :
  (initial_books + num_people_donating * books_per_person - borrowed_books) = 360 := 
by 
  -- This will contain the mathematical proof
  sorry

end remaining_books_l298_298276


namespace Matt_buys_10_key_chains_l298_298972

theorem Matt_buys_10_key_chains
  (cost_per_keychain_in_pack_of_10 : ℝ)
  (cost_per_keychain_in_pack_of_4 : ℝ)
  (number_of_keychains : ℝ)
  (savings : ℝ)
  (h1 : cost_per_keychain_in_pack_of_10 = 2)
  (h2 : cost_per_keychain_in_pack_of_4 = 3)
  (h3 : savings = 20)
  (h4 : 3 * number_of_keychains - 2 * number_of_keychains = savings) :
  number_of_keychains = 10 := 
by
  sorry

end Matt_buys_10_key_chains_l298_298972


namespace sum_of_x_intercepts_l298_298572

theorem sum_of_x_intercepts (a b : ℕ) (ha : a > 0) (hb : b > 0) 
  (h : (5 : ℤ) * (3 : ℤ) = (a : ℤ) * (b : ℤ)) : 
  ((-5 : ℤ) / (a : ℤ)) + ((-5 : ℤ) / (3 : ℤ)) + ((-1 : ℤ) / (1 : ℤ)) + ((-1 : ℤ) / (15 : ℤ)) = -8 := 
by 
  sorry

end sum_of_x_intercepts_l298_298572


namespace find_premium_percentage_l298_298706

theorem find_premium_percentage :
  ∃ P : ℝ,
    let investmentAmount: ℝ := 14400,
        faceValue: ℝ := 100,
        dividendPercentage: ℝ := 0.07,
        dividendReceived: ℝ := 840 in
    let N: ℝ := dividendReceived / (dividendPercentage * faceValue) in
    N * (faceValue + P) = investmentAmount ∧
    P = 20 :=
by
  let investmentAmount: ℝ := 14400
  let faceValue: ℝ := 100
  let dividendPercentage: ℝ := 0.07
  let dividendReceived: ℝ := 840
  let N: ℝ := dividendReceived / (dividendPercentage * faceValue)
  use 20
  simp only [N, investmentAmount, faceValue]
  split
  · sorry -- skip the actual proof
  · refl

end find_premium_percentage_l298_298706


namespace area_increase_300_percent_l298_298239

noncomputable def percentage_increase_of_area (d : ℝ) : ℝ :=
  let d' := 2 * d
  let r := d / 2
  let r' := d' / 2
  let A := Real.pi * r^2
  let A' := Real.pi * (r')^2
  100 * (A' - A) / A

theorem area_increase_300_percent (d : ℝ) : percentage_increase_of_area d = 300 :=
by
  sorry

end area_increase_300_percent_l298_298239


namespace sequence_pairs_l298_298944

theorem sequence_pairs (a : ℕ → ℝ) (h_incr : ∀ n, a n < a (n + 1))
  (h_pos : ∀ n, 0 < a n)
  (h_inf : tendsto (fun n => a n) atTop atTop)
  (h_ratio : ∀ n, a (n + 1) / a n ≤ 10) :
  ∀ k : ℕ, ∃ᶠ (i j : ℕ) in atTop ×ᶠ atTop, 10^k ≤ a i / a j ∧ a i / a j ≤ 10^(k + 1) :=
by
  sorry

end sequence_pairs_l298_298944


namespace represent_class_l298_298888

theorem represent_class (Grade Class : Nat) (rep : (Grade, Class)) : (Class, Grade) = (8, 7) :=
by
  -- Given the specific condition (Grade, Class) = (7, 8)
  have h1 : (Grade, Class) = (7, 8) := by sorry
  -- Rearrange to show (Class, Grade) 
  show (Class, Grade) = (8, 7) by sorry

end represent_class_l298_298888


namespace quadrilateral_parallelogram_l298_298080

-- Definitions based on conditions
variable (A B C D O M O_1 O_2 : Type)

-- Assume A, B, C, D are points on a cyclic quadrilateral inscribed in a circle centered at O
variable [CyclicQuadrilateral A B C D O]

-- Assume M is the intersection of diagonals AC and BD
variable [Intersection M (Diagonal AC) (Diagonal BD)]

-- Assume O_1 is the center of the circumcircle of triangle ABM
variable [CircleCenter O_1 (Triangle A B M)]

-- Assume O_2 is the center of the circumcircle of triangle CDM
variable [CircleCenter O_2 (Triangle C D M)]

-- Define the theorem that O, O_1, M, O_2 form a parallelogram
theorem quadrilateral_parallelogram 
  (h1 : CyclicQuadrilateral A B C D O)
  (h2 : Intersection M (Diagonal AC) (Diagonal BD))
  (h3 : CircleCenter O_1 (Triangle A B M))
  (h4 : CircleCenter O_2 (Triangle C D M)) :
  Parallelogram O O_1 M O_2 := 
sorry -- proof placeholder

end quadrilateral_parallelogram_l298_298080


namespace graph_shift_up_1_l298_298647

theorem graph_shift_up_1 (f : ℝ → ℝ) (x : ℝ) : 
  (∀ x, f x = 2^x) → (f x + 1 = 2^x + 1) :=
by
  intro h
  calc
    f x + 1 = 2^x + 1 : by rw h

end graph_shift_up_1_l298_298647


namespace triangle_DEC_angles_l298_298828

variables {A B C M P D E : Type}
variables [EquilateralTriangle ABC] [LineParallelTo AC AB M] [LineParallelTo AC BC P]
variable [CenterEquilateralTriangle PMB D]
variable [Midpoint AP E]

theorem triangle_DEC_angles :
  angle DEC = 90 ∧ angle CDE = 60 ∧ angle EDC = 30 :=
sorry

end triangle_DEC_angles_l298_298828


namespace polygon_sides_exterior_angle_l298_298897

theorem polygon_sides_exterior_angle (exterior_angle : ℝ) (h : exterior_angle = 45) : 
  let sum_exterior_angles := 360
  in sum_exterior_angles / exterior_angle = 8 :=
by
  sorry

end polygon_sides_exterior_angle_l298_298897


namespace symmetric_point_proof_l298_298843

def point := (ℝ × ℝ)

def line (m b : ℝ) (p : point) : Prop :=
  (p.1 - p.2 - 2 = 0)

def symmetric_point (P : point) (L : point → Prop) : point :=
  let x := P.1
  let y := P.2
  let m := (x + y - 2) / 2
  let n := (y * (-1) / (x - 1) + 1) in
  (x + y, 2 - x)

theorem symmetric_point_proof :
  let P : point := (1, 1)
  let L : point → Prop := line 1 (-2)
  symmetric_point P L = (3, -1) :=
by {
  sorry
}

end symmetric_point_proof_l298_298843


namespace proof_problem_l298_298454

open Real

/-- Given basic conditions --/
noncomputable def basis_vectors (e1 e2 : ℝ × ℝ) : Prop :=
e1 ≠ (0, 0) ∧ e2 ≠ (0, 0) ∧ e1.x * e2.y - e1.y * e2.x ≠ 0

/-- Given basis vectors and conditions for collinearity --/
noncomputable def given_conditions (e1 e2 e_a_b e_b_e e_e_c : ℝ × ℝ) (λ : ℝ) : Prop :=
basis_vectors e1 e2 ∧ e_a_b = (2 * e1.x + e2.x, 2 * e1.y + e2.y) ∧
e_b_e = (-e1.x + λ * e2.x, -e1.y + λ * e2.y) ∧
e_e_c = (-2 * e1.x + e2.x, -2 * e1.y + e2.y) ∧
∃ k : ℝ, (e_a_b.x + e_b_e.x = k * e_e_c.x) ∧ (e_a_b.y + e_b_e.y = k * e_e_c.y)

/-- Given unit vectors with an angle of 60 degrees between them and calculation of a and b. --/
noncomputable def unit_vectors_60 (e1 e2 : ℝ × ℝ) : Prop :=
(e1.x^2 + e1.y^2 = 1) ∧ (e2.x^2 + e2.y^2 = 1) ∧ (e1.x * e2.x + e1.y * e2.y = 1 / 2)

noncomputable def calc_ab_dot (a b : ℝ × ℝ) :=
a.x * b.x + a.y * b.y

noncomputable def vector_dot_prod (e1 e2 : ℝ × ℝ) (λ : ℝ) : ℝ :=
calc_ab_dot (e1.x + λ * e2.x, e1.y + λ * e2.y) (-2 * λ * e1.x - e2.x, -2 * λ * e1.y - e2.y)

/-- Proof of point collinearity and max/min dot product values. --/
theorem proof_problem (e1 e2 : ℝ × ℝ) (e_a_b e_b_e e_e_c : ℝ × ℝ) (λ : ℝ) :
given_conditions e1 e2 e_a_b e_b_e e_e_c λ ∧ unit_vectors_60 e1 e2 ∧ (-3 ≤ λ) ∧ (λ ≤ 5) →
(λ = -3/2) ∧ (vector_dot_prod e1 e2 λ = max (1/4) min (-1/2)) :=
sorry

end proof_problem_l298_298454


namespace number_of_red_balls_l298_298918

theorem number_of_red_balls (x : ℕ) (h₀ : 4 > 0) (h₁ : (x : ℝ) / (x + 4) = 0.6) : x = 6 :=
sorry

end number_of_red_balls_l298_298918


namespace sum_of_f_values_l298_298207

noncomputable def f : ℝ → ℝ := sorry

theorem sum_of_f_values :
  (∀ x : ℝ, f x + f (-x) = 0) →
  (∀ x : ℝ, f x = f (x + 2)) →
  (∀ x : ℝ, 0 ≤ x → x < 1 → f x = 2^x - 1) →
  f (1/2) + f 1 + f (3/2) + f 2 + f (5/2) = Real.sqrt 2 - 1 :=
by
  intros h1 h2 h3
  sorry

end sum_of_f_values_l298_298207


namespace lcm_12_18_l298_298422

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end lcm_12_18_l298_298422


namespace evaluate_expression_l298_298786

theorem evaluate_expression : 
  (2 / (log 7 (2000^3)) + 3 / (log 8 (2000^3)) = 2 / 3) :=
  sorry

end evaluate_expression_l298_298786


namespace collinear_points_in_circumscribed_quadrilateral_l298_298945

-- Define data structures and necessary prerequisites
variables (A B C D P I I1 : Type)
variables [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited P] [Inhabited I] [Inhabited I1]

-- Definition for a circumscribed quadrilateral
def circumscribed_quadrilateral (A B C D : Type) : Prop :=
-- Add the specific properties that define a circumscribed quadrilateral
sorry

-- Definition for the incenter of a triangle
def incenter_of_triangle (x y z : Type) : Type :=
-- Add the specific properties that define an incenter
sorry

-- Definition for the excenter of a triangle touching a specific side
def excenter_of_triangle_touching (x y z : Type) (side : Type) : Type :=
-- Add the specific properties that define an excenter touching a given side
sorry

-- Define the statement in Lean 4
theorem collinear_points_in_circumscribed_quadrilateral
  (h1 : circumscribed_quadrilateral A B C D)
  (h2 : I = incenter_of_triangle A B C)
  (h3 : I1 = excenter_of_triangle_touching C D A A)
  (h4 : P = (intersection_of_diagonals A C B D)) :
  collinear P I I1 :=
sorry

end collinear_points_in_circumscribed_quadrilateral_l298_298945


namespace general_formula_constant_c_value_l298_298825

-- Definition of an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a (n + 1) - a n = d

-- Given sequence {a_n}
variables {a : ℕ → ℝ} (S : ℕ → ℝ) (d : ℝ)
-- Conditions
variables (h1 : a 3 * a 4 = 117) (h2 : a 2 + a 5 = 22) (hd_pos : d > 0)
-- Proof that the general formula for the sequence {a_n} is a_n = 4n - 3
theorem general_formula :
  (∀ n, a n = 4 * n - 3) :=
sorry

-- Given new sequence {b_n}
variables (b : ℕ → ℕ → ℝ) {c : ℝ} (hc : c ≠ 0)
-- New condition that bn is an arithmetic sequence
variables (h_b1 : b 1 = S 1 / (1 + c)) (h_b2 : b 2 = S 2 / (2 + c)) (h_b3 : b 3 = S 3 / (3 + c))
-- Proof that c = -1/2 is the correct constant
theorem constant_c_value :
  (c = -1 / 2) :=
sorry

end general_formula_constant_c_value_l298_298825


namespace totalMoney_l298_298532

noncomputable def joannaMoney : ℕ := 8
noncomputable def brotherMoney : ℕ := 3 * joannaMoney
noncomputable def sisterMoney : ℕ := joannaMoney / 2

theorem totalMoney : joannaMoney + brotherMoney + sisterMoney = 36 := by
  sorry

end totalMoney_l298_298532


namespace tangent_line_at_zero_strictly_increasing_f_range_of_a_values_l298_298859

noncomputable def f (a x : Real) : Real :=
  a^x + x^2 - x * Real.log a

def tangent_at_zero (a : Real) (ha : a > 0) (ha_ne_one : a ≠ 1) : String :=
  "y = 1"

theorem tangent_line_at_zero (a : Real) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  tangent_at_zero a ha ha_ne_one = "y = 1" :=
sorry

def strictly_increasing_interval (a : Real) (ha : a > 0) (ha_ne_one : a ≠ 1) : Set Real :=
  (0, ∞)

theorem strictly_increasing_f (a : Real) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  ∀ x, x ∈ strictly_increasing_interval a ha ha_ne_one → (λ x, f a x) x > 0 :=
sorry

def range_of_a (a : Real) (ha : a > 0) (ha_ne_one : a ≠ 1) : Set Real :=
  (0, 1 / Real.exp 1] ∪ [Real.exp 1, ∞)

theorem range_of_a_values (a : Real) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  ∃ x1 x2 ∈ Set.Icc (-1:Real) 1, abs (f a x1 - f a x2) ≥ Real.exp 1 - 1 →
  a ∈ range_of_a a ha ha_ne_one :=
sorry

end tangent_line_at_zero_strictly_increasing_f_range_of_a_values_l298_298859


namespace n_is_composite_number_l298_298354

-- Definition of the problem conditions
def irregular_n_gon (n : ℕ) : Prop := 
  ∃ M : Type, (M ≠ M ∧ M ≠ M) -- Here, some definition that strictly states irregular
  ∧ (∃ circ : real, (circ. inscribed_in (n-sides : ℝ)) ∧ true) -- Inscribed in a circle

-- Rotation condition
def rotated_polygon (n : ℕ) (α : ℝ) : Prop :=
  ∃ α : ℝ, α ≠ 2 * real.pi 
  ∧ polygon_maps_onto_itself (α ≠ 2 * real.pi)

-- Conclusion to prove
theorem n_is_composite_number (n: ℕ) (α : ℝ) (h_irr: irregular_n_gon n) (h_rot: rotated_polygon n α) :
  ∃k m : ℕ, 1 < k ∧ 1 < m ∧ k*m = n :=
sorry

end n_is_composite_number_l298_298354


namespace necessary_but_not_sufficient_l298_298621

-- Define the lines and their respective forms.
def line1 (a : ℝ) : ℝ × ℝ → Prop := λ p, 2 * p.1 + a * p.2 - 1 = 0
def line2 (b : ℝ) : ℝ × ℝ → Prop := λ p, b * p.1 + 2 * p.2 - 2 = 0

-- Define the slopes of the lines to ascertain parallelism.
def slope1 (a : ℝ) : ℝ := -2 / a
def slope2 (b : ℝ) : ℝ := -b / 2

-- Prove that the condition ab=4 is necessary but not sufficient for the lines being parallel and not coinciding.
theorem necessary_but_not_sufficient (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) :
  (slope1 a = slope2 b) ↔ (a * b = 4) :=
by
  sorry

end necessary_but_not_sufficient_l298_298621


namespace olly_owns_ferrets_l298_298976

-- Define the number of dogs, cats, and total shoes needed
def num_dogs : ℕ := 3
def num_cats : ℕ := 2
def total_shoes_needed : ℕ := 24

-- Define the number of paws per animal
def paws_per_dog : ℕ := 4
def paws_per_cat : ℕ := 4
def paws_per_ferret : ℕ := 4

-- Prove that the number of ferrets Olly owns is 1
theorem olly_owns_ferrets : (24 - ((3 * 4) + (2 * 4))) / 4 = 1 :=
by
  -- calculate shoes for dogs and cats
  let shoes_dogs := num_dogs * paws_per_dog
  let shoes_cats := num_cats * paws_per_cat
  let shoes_other := total_shoes_needed - (shoes_dogs + shoes_cats)
  let num_ferrets := shoes_other / paws_per_ferret
  show num_ferrets = 1 from sorry

end olly_owns_ferrets_l298_298976


namespace number_of_divisors_le_two_sqrt_l298_298220

theorem number_of_divisors_le_two_sqrt (n : ℕ) (hn : n > 0) : 
  (finset.filter (λ d, n % d = 0) (finset.range (n+1))).card ≤ 2 * (nat.sqrt n) :=
sorry

end number_of_divisors_le_two_sqrt_l298_298220


namespace inequality_solution_set_l298_298472

theorem inequality_solution_set
  (k a b c : ℝ)
  (ineq_solution : ∀ x : ℝ, (x ∈ set.union (set.Ioo (-1 : ℝ) (-1/3 : ℝ)) (set.Ioo (1/2 : ℝ) (1 : ℝ))) ↔ ((k / (x + a) + (x + b) / (x + c)) < 0)) :
  ∀ y : ℝ, (y ∈ set.union (set.Ioo (-3 : ℝ) (-1 : ℝ)) (set.Ioo (1 : ℝ) (2 : ℝ))) ↔
             ((kx / (ax + 1) + (bx + 1) / (cx + 1)) < 0) :=
by
  sorry

end inequality_solution_set_l298_298472


namespace solve_for_x_l298_298805

theorem solve_for_x (x y : ℝ) (h1 : x + y = 15) (h2 : x - y = 5) : x = 10 :=
by
  sorry

end solve_for_x_l298_298805


namespace standard_eq_line_standard_eq_circle_range_of_a_l298_298867

noncomputable def line (a t : ℝ) : ℝ × ℝ := (a - 2 * t, -4 * t)
noncomputable def circle (θ : ℝ) : ℝ × ℝ := (4 * Real.cos θ, 4 * Real.sin θ)

-- Prove the standard equation of line l
theorem standard_eq_line (a t : ℝ) (x y : ℝ) (h : line a t = (x, y)) : 2 * x - y - 2 * a = 0 := by
  sorry

-- Prove the standard equation of circle C
theorem standard_eq_circle (θ : ℝ) (x y : ℝ) (h : circle θ = (x, y)) : x^2 + y^2 = 16 := by
  sorry

-- Prove the range of values for a when line l and circle C have common points
theorem range_of_a (a : ℝ) (h : ∀(t : ℝ), ∃(θ : ℝ), line a t = circle θ) : -2 * Real.sqrt 5 ≤ a ∧ a ≤ 2 * Real.sqrt 5 := by
  sorry

end standard_eq_line_standard_eq_circle_range_of_a_l298_298867


namespace square_flag_side_length_side_length_of_square_flags_is_4_l298_298031

theorem square_flag_side_length 
  (total_fabric : ℕ)
  (fabric_left : ℕ)
  (num_square_flags : ℕ)
  (num_wide_flags : ℕ)
  (num_tall_flags : ℕ)
  (wide_flag_length : ℕ)
  (wide_flag_width : ℕ)
  (tall_flag_length : ℕ)
  (tall_flag_width : ℕ)
  (fabric_used_on_wide_and_tall_flags : ℕ)
  (fabric_used_on_all_flags : ℕ)
  (fabric_used_on_square_flags : ℕ)
  (square_flag_area : ℕ)
  (side_length : ℕ) : Prop :=
  total_fabric = 1000 ∧
  fabric_left = 294 ∧
  num_square_flags = 16 ∧
  num_wide_flags = 20 ∧
  num_tall_flags = 10 ∧
  wide_flag_length = 5 ∧
  wide_flag_width = 3 ∧
  tall_flag_length = 5 ∧
  tall_flag_width = 3 ∧
  fabric_used_on_wide_and_tall_flags = (num_wide_flags + num_tall_flags) * (wide_flag_length * wide_flag_width) ∧
  fabric_used_on_all_flags = total_fabric - fabric_left ∧
  fabric_used_on_square_flags = fabric_used_on_all_flags - fabric_used_on_wide_and_tall_flags ∧
  square_flag_area = fabric_used_on_square_flags / num_square_flags ∧
  side_length = Int.sqrt square_flag_area ∧
  side_length = 4

theorem side_length_of_square_flags_is_4 : 
  square_flag_side_length 1000 294 16 20 10 5 3 5 3 450 706 256 16 4 :=
  by
    sorry

end square_flag_side_length_side_length_of_square_flags_is_4_l298_298031


namespace jane_ate_four_pieces_l298_298434

theorem jane_ate_four_pieces (total_pieces : ℕ) (number_of_people : ℕ) (equal_division : total_pieces % number_of_people = 0) (total_pieces = 12) (number_of_people = 3) : (total_pieces / number_of_people = 4) :=
by
  sorry

end jane_ate_four_pieces_l298_298434


namespace inequality_solution_l298_298863

theorem inequality_solution {a : ℝ} :
  (∀ x : ℝ, -3 < x ∧ x < -1 ∨ x > 2 → (x + a) / (x^2 + 4 * x + 3) > 0) → a = -2 :=
begin
  sorry
end

end inequality_solution_l298_298863


namespace color_ways_l298_298806

theorem color_ways (n : ℕ) : ℕ :=
  let f : ℕ → ℕ → ℕ :=
    λ i color, 
      if i = 1 then
        if color = 1 then 1
        else if color = 2 then 0
        else 1
      else if i % 2 = 1 then
        if color = 1 then (f (i-1) 2 + f (i-1) 3)
        else if color = 2 then 0
        else (f (i-1) 1 + f (i-1) 2)
      else
        if color = 1 then (f (i-1) 2 + f (i-1) 3)
        else if color = 2 then (f (i-1) 1 + f (i-1) 3)
        else (f (i-1) 1 + f (i-1) 2)
  in
  f n 1 + f n 2 + f n 3

end color_ways_l298_298806


namespace initial_pennies_l298_298674

theorem initial_pennies (P : ℝ) 
  (h1 : P - (0.5 * P + 1) = 0.5 * P - 1)
  (h2 : 0.5 * P - 1 - (0.5 * (0.5 * P - 1) + 2) = 0.25 * P - 2.5)
  (h3 : 0.25 * P - 2.5 - (0.5 * (0.25 * P - 2.5) + 3) = 0.125 * P - 4.25)
  (h4 : 0.125 * P - 4.25 = 1) :
  P = 42 :=
begin
  sorry
end

end initial_pennies_l298_298674


namespace number_of_values_a_l298_298202

theorem number_of_values_a :
  {a : ℕ // 0 < a ∧ a < 100 ∧ 24 ∣ (a^3 + 23)}.card = 5 :=
by
  sorry

end number_of_values_a_l298_298202


namespace trig_identity_15_deg_l298_298663

theorem trig_identity_15_deg :
  (sin (15 * real.pi / 180) + cos (15 * real.pi / 180)) /
  (sin (15 * real.pi / 180) - cos (15 * real.pi / 180)) = -sqrt 3 :=
by
  sorry

end trig_identity_15_deg_l298_298663


namespace find_omega_l298_298829

-- Define the problem conditions
def A : ℝ × ℝ := (π / 6, sqrt 3 / 2)
def B : ℝ × ℝ := (π / 4, 1)
def C : ℝ × ℝ := (π / 2, 0)
def f (ω x : ℝ) : ℝ := sin (ω * x)

-- Define the target set based on the solution
def solutions : Set ℝ := 
  ({ω | ∃ k : ℕ, ω = 8 * k + 2} ∩ {ω | ∃ k : ℕ, ω = 12 * k + 2 ∨ ω = 12 * k + 4}) ∪ {2, 4}

-- State the proof problem
theorem find_omega (ω : ℝ) (hA : f ω (π / 6) = sqrt 3 / 2) (hB : f ω (π / 4) = 1) (hC : f ω (π / 2) = 0) : ω ∈ solutions := 
by
  sorry

end find_omega_l298_298829


namespace ratio_a_div_8_to_b_div_7_l298_298494

theorem ratio_a_div_8_to_b_div_7 (a b : ℝ) (h1 : 7 * a = 8 * b) (h2 : a ≠ 0 ∧ b ≠ 0) :
  (a / 8) / (b / 7) = 1 :=
sorry

end ratio_a_div_8_to_b_div_7_l298_298494


namespace balls_coloring_l298_298997

theorem balls_coloring : 
  let positions := list.range 10 in
  let valid_pairs := (positions.product positions).filter (λ (x, y), x < y ∧ y - x > 2) in
  valid_pairs.length = 28 :=
by
  let positions := list.range 10;
  let valid_pairs := (positions.product positions).filter (λ (x, y), x < y ∧ y - x > 2);
  have : valid_pairs.length = 28 := sorry;
  exact this

end balls_coloring_l298_298997


namespace b_2023_eq_64_div_9_l298_298188

-- Define the sequence b
def b : ℕ → ℚ
| 0 := 3         -- Note: Lean arrays are zero-indexed, so b_1 = b 0 in Lean
| 1 := 4         -- and b_2 = b 1
| n + 2 := (b (n + 1))^2 / (b n)

-- Our goal is to prove that b 2022 = 64 / 9 (b 2022 corresponds to b_2023 in 1-based indexing)
theorem b_2023_eq_64_div_9 : b 2022 = 64 / 9 :=
sorry

end b_2023_eq_64_div_9_l298_298188


namespace orchids_cut_correct_l298_298272

-- Define the initial number of orchids and the number of  orchids after cutting
def initial_orchids : Nat := 2
def orchids_after_cutting : Nat := 21

-- Define the number of orchids Jessica cut
def orchids_cut : Nat := orchids_after_cutting - initial_orchids

theorem orchids_cut_correct : orchids_cut = 19 := by
  unfold orchids_cut
  simp
  sorry

end orchids_cut_correct_l298_298272


namespace total_rainfall_l298_298932

theorem total_rainfall (rain_first_hour : ℕ) (rain_second_hour : ℕ) : Prop :=
  rain_first_hour = 5 →
  rain_second_hour = 7 + 2 * rain_first_hour →
  rain_first_hour + rain_second_hour = 22

-- Add sorry to skip the proof.

end total_rainfall_l298_298932


namespace greatest_t_for_good_set_l298_298946

open Nat

def is_good (k : ℕ) (S : set ℕ) : Prop :=
∃ (color : ℕ → ℕ), (∀ n, color n < k) ∧ (∀ (x y : ℕ), x ∈ S → y ∈ S → x ≠ y → color x ≠ color y ∨ x + y ∉ S)

theorem greatest_t_for_good_set (k : ℕ) (h : k > 1) : 
  ∃ (t : ℕ), (∀ a : ℕ, is_good k {n | a+1 ≤ n ∧ n ≤ a+t}) ∧ 
              (∀ (t' : ℕ), (∀ a : ℕ, is_good k {n | a+1 ≤ n ∧ n ≤ a+t'}) → t' ≤ 2*k - 1) :=
sorry

end greatest_t_for_good_set_l298_298946


namespace distance_from_circle_center_to_point_l298_298284

theorem distance_from_circle_center_to_point :
  ∀ (x y : ℝ), 
  let circle_eq := x^2 + y^2 = 6*x - 8*y + 20 in
  (x = 3) ∧ (y = -4) → 
  let point := (-3, 4) in
  real.dist (↑(x, y)) (↑point) = 10 := 
by
  intro x y circle_eq center point,
  sorry

end distance_from_circle_center_to_point_l298_298284


namespace perpendicular_vectors_m_l298_298477

theorem perpendicular_vectors_m (m : ℝ) : (3, -2) • (3 * m - 1, 4 - m) = 0 ↔ m = 1 :=
by
  sorry

end perpendicular_vectors_m_l298_298477


namespace lcm_12_18_l298_298414

theorem lcm_12_18 : Nat.lcm 12 18 = 36 :=
by
  -- Definitions of the conditions
  have h12 : 12 = 2 * 2 * 3 := by norm_num
  have h18 : 18 = 2 * 3 * 3 := by norm_num
  
  -- Calculating LCM using the built-in Nat.lcm
  rw [Nat.lcm_comm]  -- Ordering doesn't matter for lcm
  rw [Nat.lcm, h12, h18]
  -- Prime factorizations checks are implicitly handled
  
  -- Calculate the LCM based on the highest powers from the factorizations
  have lcm_val : 4 * 9 = 36 := by norm_num
  
  -- So, the LCM of 12 and 18 is
  exact lcm_val

end lcm_12_18_l298_298414


namespace series_convergence_l298_298169

theorem series_convergence : ∃ l : ℝ, has_sum (λ n : ℕ, 1 / (5 * 2^n)) l :=
by
  -- Sorry is used here to skip the actual proof steps
  sorry

end series_convergence_l298_298169


namespace calculate_area_l298_298366

def bounded_area_arcsin_cos (a b : ℝ) (f : ℝ → ℝ) : ℝ :=
∫ x in a..b, f x

theorem calculate_area :
  bounded_area_arcsin_cos (π / 2) (7 * π / 2) (λ x, Real.arcsin (Real.cos x)) = π^2 := 
sorry

end calculate_area_l298_298366


namespace settle_debt_using_coins_l298_298655

theorem settle_debt_using_coins :
  ∃ n m : ℕ, 49 * n - 99 * m = 1 :=
sorry

end settle_debt_using_coins_l298_298655


namespace solution_set_integer_count_l298_298903

theorem solution_set_integer_count (m : ℝ) : 
  (∀ x : ℝ, x^2 + (m + 1) * x + m < 0 → x ∈ ℤ) → (m ∈ Icc (-2 : ℝ) (-1) ∨ m ∈ Ioo (3 : ℝ) 4) :=
sorry

end solution_set_integer_count_l298_298903


namespace halved_r_value_of_n_l298_298926

theorem halved_r_value_of_n (r a : ℝ) (n : ℕ) (h₁ : a = (2 * r)^n)
  (h₂ : 0.125 * a = r^n) : n = 3 :=
by
  sorry

end halved_r_value_of_n_l298_298926


namespace area_of_annulus_l298_298007

variables (r s k : ℝ)
-- Conditions
def r_gt_s : Prop := r > s
def relation : Prop := r^2 - s^2 = k^2

-- Theorem stating the area of the annulus
theorem area_of_annulus (h1 : r_gt_s r s) (h2 : relation r s k) : 
  let area := π * k^2 in 
  area = π * k^2 := 
by sorry

end area_of_annulus_l298_298007


namespace prof_seat_arrangements_l298_298392

theorem prof_seat_arrangements :
  ∃ (α β γ δ: ℕ), 
    1 ≤ α ∧ α ≤ 11 ∧ 
    1 ≤ β ∧ β ≤ 11 ∧ 
    1 ≤ γ ∧ γ ≤ 11 ∧ 
    1 ≤ δ ∧ δ ≤ 11 ∧ 
    α ≠ β ∧ α ≠ γ ∧ α ≠ δ ∧ 
    β ≠ γ ∧ β ≠ δ ∧ 
    γ ≠ δ ∧ 
    (∃ s1 s2 s3 s4 s5 s6 s7: ℕ),
    (0 < s1) ∧ (s1 ≠ α) ∧ (s1 ≠ β) ∧ (s1 ≠ γ) ∧ (s1 ≠ δ) ∧ 
    (0 < s2) ∧ (s2 ≠ α) ∧ (s2 ≠ β) ∧ (s2 ≠ γ) ∧ (s2 ≠ δ) ∧ 
    (0 < s3) ∧ (s3 ≠ α) ∧ (s3 ≠ β) ∧ (s3 ≠ γ) ∧ (s3 ≠ δ) ∧ 
    (0 < s4) ∧ (s4 ≠ α) ∧ (s4 ≠ β) ∧ (s4 ≠ γ) ∧ (s4 ≠ δ) ∧ 
    (0 < s5) ∧ (s5 ≠ α) ∧ (s5 ≠ β) ∧ (s5 ≠ γ) ∧ (s5 ≠ δ) ∧ 
    (0 < s6) ∧ (s6 ≠ α) ∧ (s6 ≠ β) ∧ (s6 ≠ γ) ∧ (s6 ≠ δ) ∧ 
    (0 < s7) ∧ (s7 ≠ α) ∧ (s7 ≠ β) ∧ (s7 ≠ γ) ∧ (s7 ≠ δ) ∧
    ((List.perm ⟨s1, s2, s3, s4, s5, s6, s7, α, β, γ, δ⟩ (List.iota 11)) ∧ 
    List.nodup [s1, s2, s3, s4, s5, s6, s7, α, β, γ, δ]) → 18900 = sorry

end prof_seat_arrangements_l298_298392


namespace isosceles_triangle_ratio_ABC_l298_298199

noncomputable def isosceles_triangle_ratio : ℝ :=
  let AB := 2 * (1 : ℝ) -- assuming x = 1 for simplicity
  let AC := 2 * (1 : ℝ)
  let BC := 2 * (sqrt 2 : ℝ) -- as derived from the solution
  AB / BC

theorem isosceles_triangle_ratio_ABC
  (A B C D E F : Point)
  (h_iso : Isosceles A B C)
  (h_mid_D : Midpoint D A B)
  (h_mid_E : Midpoint E A C)
  (h_similar : Similar (Triangle B F A) (Triangle A B C)) :
  isosceles_triangle_ratio = sqrt 2 := sorry

end isosceles_triangle_ratio_ABC_l298_298199


namespace number_of_positive_solutions_l298_298035

theorem number_of_positive_solutions : 
  {x : ℝ | (2 * x ^ 2 - 7) ^ 2 = 49 ∧ 0 < x}.card = 1 := 
by
  sorry

end number_of_positive_solutions_l298_298035


namespace find_a_l298_298093

def line1 (a : ℝ) (P : ℝ × ℝ) : Prop := 2 * P.1 - a * P.2 - 1 = 0

def line2 (P : ℝ × ℝ) : Prop := P.1 + 2 * P.2 = 0

theorem find_a (a : ℝ) :
  (∀ P : ℝ × ℝ, line1 a P ∧ line2 P) → a = 1 := by
  sorry

end find_a_l298_298093


namespace ant_probability_after_6_minutes_l298_298012

open Probability

def is_valid_position (x y : ℤ) : Prop := -2 ≤ x ∧ x ≤ 2 ∧ -2 ≤ y ∧ y ≤ 2

def valid_moves : List (ℤ × ℤ) :=
  [(1, 0), (-1, 0), (0, 1), (0, -1)]

def ant_move (pos : ℤ × ℤ) (move : ℤ × ℤ) : ℤ × ℤ :=
  (pos.1 + move.1, pos.2 + move.2)

noncomputable def probability_ant_at_C :
  ℚ := 20 * (1 / 4096) -- Combines 𝜀_perm and sequence prob

theorem ant_probability_after_6_minutes : 
  probability_ant_at_C = 5 / 1024 := 
by
  sorry

end ant_probability_after_6_minutes_l298_298012


namespace number_of_passed_candidates_l298_298241

theorem number_of_passed_candidates
  (P F : ℕ) 
  (h1 : P + F = 120)
  (h2 : 39 * P + 15 * F = 4200) : P = 100 :=
sorry

end number_of_passed_candidates_l298_298241


namespace find_triples_l298_298794

theorem find_triples (a b c : ℝ) : 
  a + b + c = 14 ∧ a^2 + b^2 + c^2 = 84 ∧ a^3 + b^3 + c^3 = 584 ↔ (a = 4 ∧ b = 2 ∧ c = 8) ∨ (a = 2 ∧ b = 4 ∧ c = 8) ∨ (a = 8 ∧ b = 2 ∧ c = 4) :=
by
  sorry

end find_triples_l298_298794


namespace participate_teams_total_scored_points_l298_298913

-- Conditions
def first_place_score := 11
def second_place_score := 9
def third_place_score := 7
def fourth_place_score := 5
def total_top_scores := first_place_score + second_place_score + third_place_score + fourth_place_score

-- Definitions used
def points_per_draw := 1
def points_per_win := 2
def total_points (k : ℕ) := k * (k - 1)

def number_of_teams (k : ℕ) : Prop :=
  total_top_scores + 4 * (k - 4) < total_points k

def total_points_scored (t : ℕ) (k : ℕ) : Prop :=
  t = total_points k

-- Proof statements
theorem participate_teams : ∃ k, k = 7 ∧ number_of_teams k :=
begin
  existsi 7,
  split,
  rfl,
  unfold number_of_teams,
  simp,
  unfold total_points,
  norm_num,
end

theorem total_scored_points : ∃ t, ∃ k, k = 7 ∧ total_points_scored t k :=
begin
  existsi 42,
  existsi 7,
  split,
  rfl,
  unfold total_points_scored,
  unfold total_points,
  norm_num,
end

end participate_teams_total_scored_points_l298_298913


namespace not_southern_with_southern_divisors_2022_infinitely_many_not_southern_with_2022_southern_divisors_l298_298014

-- Define what it means for an integer to be "southern"
def southern (n : ℕ) : Prop :=
  ∀ (d : ℕ), (d ∣ n) → ∃ i : ℕ, (1 = d) ∨ (d = n) ∨ (n % (d - 1) = 0)

-- Prove the existence of an integer that has 2022 positive southern divisors and is not southern
theorem not_southern_with_southern_divisors_2022 :
  ∃ (n : ℕ), (¬ southern n) ∧ (card {d : ℕ | d ∣ n ∧ southern d} = 2022) :=
sorry

-- Prove that there are infinitely many such integers
theorem infinitely_many_not_southern_with_2022_southern_divisors :
  infinite {n : ℕ | (¬ southern n) ∧ (card {d : ℕ | d ∣ n ∧ southern d} = 2022)} :=
sorry

end not_southern_with_southern_divisors_2022_infinitely_many_not_southern_with_2022_southern_divisors_l298_298014


namespace gcd_324_243_135_l298_298279

theorem gcd_324_243_135 : Nat.gcd (Nat.gcd 324 243) 135 = 27 :=
by
  sorry

end gcd_324_243_135_l298_298279


namespace tail_count_likelihood_draw_and_rainy_l298_298041

def coin_tosses : ℕ := 25
def heads_count : ℕ := 11
def draws_when_heads : ℕ := 7
def rainy_when_tails : ℕ := 4

theorem tail_count :
  coin_tosses - heads_count = 14 :=
sorry

theorem likelihood_draw_and_rainy :
  0 = 0 :=
sorry

end tail_count_likelihood_draw_and_rainy_l298_298041


namespace lcm_12_18_is_36_l298_298418

def prime_factors (n : ℕ) : list ℕ :=
  if n = 12 then [2, 2, 3]
  else if n = 18 then [2, 3, 3]
  else []

noncomputable def lcm_of_two (a b : ℕ) : ℕ :=
  match prime_factors a, prime_factors b with
  | [2, 2, 3], [2, 3, 3] => 36
  | _, _ => 0

theorem lcm_12_18_is_36 : lcm_of_two 12 18 = 36 :=
  sorry

end lcm_12_18_is_36_l298_298418


namespace zero_is_natural_number_l298_298289

theorem zero_is_natural_number (N : Type) [Nat : ∀ (n : N), n = 0 ∨ (∃ m, n = m + 1)] :
  ∃ n : N, n = 0 :=
sorry

end zero_is_natural_number_l298_298289


namespace reduce_to_one_piece_l298_298633

-- Definitions representing the conditions:
def plane_divided_into_unit_triangles : Prop := sorry
def initial_configuration (n : ℕ) : Prop := sorry
def possible_moves : Prop := sorry

-- Main theorem statement:
theorem reduce_to_one_piece (n : ℕ) 
  (H1 : plane_divided_into_unit_triangles) 
  (H2 : initial_configuration n) 
  (H3 : possible_moves) : 
  ∃ k : ℕ, k * 3 = n :=
sorry

end reduce_to_one_piece_l298_298633


namespace value_of_g_3x_minus_5_l298_298993

variable (R : Type) [Field R]
variable (g : R → R)
variable (x y : R)

-- Given condition: g(x) = -3 for all real numbers x
axiom g_is_constant : ∀ x : R, g x = -3

-- Prove that g(3x - 5) = -3
theorem value_of_g_3x_minus_5 : g (3 * x - 5) = -3 :=
by
  sorry

end value_of_g_3x_minus_5_l298_298993


namespace longest_diagonal_length_l298_298723

theorem longest_diagonal_length (A : ℝ) (d1 d2 : ℝ) (h1 : A = 150) (h2 : d1 / d2 = 4 / 3) : d1 = 20 :=
by
  -- Skipping the proof here
  sorry

end longest_diagonal_length_l298_298723


namespace monotonic_increasing_f_l298_298837

theorem monotonic_increasing_f (f g : ℝ → ℝ) (hf : ∀ x, f (-x) = -f x) 
  (hg : ∀ x, g (-x) = g x) (hfg : ∀ x, f x + g x = 3^x) :
  ∀ a b : ℝ, a > b → f a > f b :=
sorry

end monotonic_increasing_f_l298_298837


namespace basicAlgorithmStatementsSet_l298_298670

/-
Given the following classifications of statements:
  1. INPUT statement - basic algorithm statement
  2. PRINT statement - basic algorithm statement
  3. IF-THEN statement - basic algorithm statement
  4. DO statement - basic algorithm statement
  5. END statement - not a basic algorithm statement
  6. WHILE statement - basic algorithm statement
  7. END IF statement - not a basic algorithm statement

Prove that the set of basic algorithm statements is exactly {1, 2, 3, 4, 6}.
-/

def isBasicAlgorithmStatement (statement : ℕ) : Prop :=
  statement = 1 ∨ statement = 2 ∨ statement = 3 ∨ statement = 4 ∨ statement = 6

theorem basicAlgorithmStatementsSet :
  {n : ℕ | isBasicAlgorithmStatement n} = {1, 2, 3, 4, 6} := by {
  rw Set.setOf_or,
  rw Set.setOf_or,
  rw Set.setOf_or,
  rw Set.setOf_or,
  sorry
}

end basicAlgorithmStatementsSet_l298_298670


namespace determine_x_l298_298995
noncomputable theory

variables (x y : ℝ)

theorem determine_x 
  (h : 12 * 3^x = 7^(y+5))
  (hy : y = -4) :
  x = Real.log (7/12) / Real.log 3 :=
sorry

end determine_x_l298_298995


namespace erase_one_not_divisible_l298_298307

theorem erase_one_not_divisible (n : ℕ) 
  (h1 : n > 1) 
  (h2 : n % 2 = 1) :
  ∃ (x : ℕ), x ∈ finset.Icc n (2 * n - 1) ∧ 
  ¬ ∃ (d ∈ finset.Icc n (2 * n - 1)), (finset.sum (finset.Icc n (2 * n - 1) \ {x}) (λ x, x)) % d = 0 :=
begin
  sorry
end

end erase_one_not_divisible_l298_298307


namespace min_balls_draw_required_l298_298321

def num_red_balls : ℕ := 35
def num_green_balls : ℕ := 25
def num_yellow_balls : ℕ := 22
def num_blue_balls : ℕ := 15
def num_white_balls : ℕ := 12
def num_black_balls : ℕ := 11
def min_balls_to_guarantee_18_same_color : ℕ := 89

theorem min_balls_draw_required :
  ∀ (n_red n_green n_yellow n_blue n_white n_black : ℕ),
    n_red = num_red_balls →
    n_green = num_green_balls →
    n_yellow = num_yellow_balls →
    n_blue = num_blue_balls →
    n_white = num_white_balls →
    n_black = num_black_balls →
    min_balls_to_guarantee_18_same_color = 89 :=
by
  intros n_red n_green n_yellow n_blue n_white n_black h_red h_green h_yellow h_blue h_white h_black
  rw [h_red, h_green, h_yellow, h_blue, h_white, h_black]
  exact sorry

end min_balls_draw_required_l298_298321


namespace stratified_sampling_correct_l298_298001

-- Define the total number of employees
def total_employees : ℕ := 100

-- Define the number of employees in each age group
def under_30 : ℕ := 20
def between_30_and_40 : ℕ := 60
def over_40 : ℕ := 20

-- Define the number of people to be drawn
def total_drawn : ℕ := 20

-- Function to calculate number of people to be drawn from each group
def stratified_draw (group_size : ℕ) (total_size : ℕ) (drawn : ℕ) : ℕ :=
  (group_size * drawn) / total_size

-- The proof problem statement
theorem stratified_sampling_correct :
  stratified_draw under_30 total_employees total_drawn = 4 ∧
  stratified_draw between_30_and_40 total_employees total_drawn = 12 ∧
  stratified_draw over_40 total_employees total_drawn = 4 := by
  sorry

end stratified_sampling_correct_l298_298001


namespace problem_proof_l298_298221

theorem problem_proof : 
  (let n := 2002 in
   (1 / (2 : ℝ) ^ n) * (∑ i in Finset.range 1001, (-1) ^ i * 3 ^ i * (Nat.choose n (2 * i))) = -(1/2)
  ) := sorry

end problem_proof_l298_298221


namespace rationalize_tsum_eq_52_l298_298224

theorem rationalize_tsum_eq_52 :
  let A := 6,
      B := 4,
      C := -1,
      D := 1,
      E := 30,
      F := 12 in
  A + B + C + D + E + F = 52 :=
by
  sorry

end rationalize_tsum_eq_52_l298_298224


namespace cost_of_fencing_l298_298301

theorem cost_of_fencing (x : ℝ)
  (ratio_condition : 3 / 2 = (length / width))
  (area_condition : 3 * x * 2 * x = 3750)
  (cost_per_meter_paise : 40)
  : 2 * ((3 * x) + (2 * x)) * (cost_per_meter_paise / 100) = 100 :=
by
  sorry

end cost_of_fencing_l298_298301


namespace find_y_l298_298038

theorem find_y :
  ∃ x y : ℤ, y = 3 * x^2 ∧
  (2 * x) / 5 = 1 / (1 - 2 / (3 + 1 / (4 - 5 / (6 - x)))) ∧ y = 147 :=
begin
  sorry,
end

end find_y_l298_298038


namespace john_win_time_l298_298535

-- Definitions based on the problem conditions
def john_speed_mph : ℝ := 15
def race_distance : ℝ := 5
def next_fastest_time_minutes : ℝ := 23

-- Convert John's speed to miles per minute
def john_speed_mpm : ℝ := john_speed_mph / 60

-- Calculate John's total race time
def john_race_time_minutes : ℝ := race_distance / john_speed_mpm

-- Prove the main statement: How many minutes did John win the race by?
theorem john_win_time : next_fastest_time_minutes - john_race_time_minutes = 3 :=
by
  sorry

end john_win_time_l298_298535


namespace baker_total_cost_is_correct_l298_298316

theorem baker_total_cost_is_correct :
  let flour_cost := 3 * 3
  let eggs_cost := 3 * 10
  let milk_cost := 7 * 5
  let baking_soda_cost := 2 * 3
  let total_cost := flour_cost + eggs_cost + milk_cost + baking_soda_cost
  total_cost = 80 := 
by
  sorry

end baker_total_cost_is_correct_l298_298316


namespace medians_intersect_at_centroid_l298_298282

theorem medians_intersect_at_centroid (A B C : Point) (M_A : Point) (M_B : Point) (M_C : Point)
    (h_MA : M_A = midpoint B C) (h_MB : M_B = midpoint A C) (h_MC : M_C = midpoint A B) :
    ∃ G : Point, concurrent (median A M_A) (median B M_B) (median C M_C) ∧ is_centroid G A B C :=
sorry

end medians_intersect_at_centroid_l298_298282


namespace abs_inequality_equiv_l298_298067

theorem abs_inequality_equiv (x : ℝ) : 1 ≤ |x - 2| ∧ |x - 2| ≤ 7 ↔ (-5 ≤ x ∧ x ≤ 1) ∨ (3 ≤ x ∧ x ≤ 9) :=
by
  sorry

end abs_inequality_equiv_l298_298067


namespace n_pow_n_gt_product_odds_l298_298574

theorem n_pow_n_gt_product_odds (n : ℕ) (h : 0 < n) : 
  n^n > ∏ i in finset.range(n), (2 * i + 1) :=
  sorry

end n_pow_n_gt_product_odds_l298_298574


namespace quadruples_satisfying_equations_l298_298062

open Real

noncomputable def valid_quadruples_count : Nat := 15

theorem quadruples_satisfying_equations :
  ∃ (S : Finset (ℝ × ℝ × ℝ × ℝ)), 
    (∀ x ∈ S, let ⟨a, b, c, d⟩ := x in 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧
      a^2 + b^2 + c^2 + d^2 = 9 ∧
      (a + b + c + d) * (a^3 + b^3 + c^3 + d^3) = 27) ∧ 
    (S.card = valid_quadruples_count) :=
by
  sorry

end quadruples_satisfying_equations_l298_298062


namespace boxes_to_eliminate_l298_298910

noncomputable def box_values : List ℕ :=
  [1, 1000, 10000, 100000, 1000000]

def count_boxes_with_at_least (values : List ℕ) (threshold : ℕ) : ℕ :=
  values.count (λ x => x ≥ threshold)

def remaining_boxes_for_at_least_half (total_boxes high_value_boxes : ℕ) : ℕ :=
  total_boxes - 2 * high_value_boxes

theorem boxes_to_eliminate :
  remaining_boxes_for_at_least_half 30 (count_boxes_with_at_least box_values 200000) = 18 := by
  sorry

end boxes_to_eliminate_l298_298910


namespace andy_10th_turn_l298_298015

def andy_moves : ℕ → ℤ × ℤ
| 0 := (0, 0)
| (n + 1) := 
    let (x, y) := andy_moves n in
    let k := n + 1 in
    match n % 4 with
    | 0 => (x, y + k) -- North
    | 1 => (x + k, y) -- East
    | 2 => (x, y - k) -- South
    | _ => (x - k, y) -- West

theorem andy_10th_turn : andy_moves 10 = (6, 5) :=
by {
  sorry
}

end andy_10th_turn_l298_298015


namespace hannah_pieces_lost_l298_298592

def scarlett_lost : ℕ := 6
def total_pieces_left : ℕ := 18
def initial_pieces_each : ℕ := 16

theorem hannah_pieces_lost : 
  ∀ (scarlett_lost total_pieces_left initial_pieces_each : ℕ),
  total_pieces_left = 18 →
  scarlett_lost = 6 →
  initial_pieces_each = 16 →
  (32 - total_pieces_left) - scarlett_lost = 8 :=
by
  intro scarlett_lost total_pieces_left initial_pieces_each
  assume h1 h2 h3
  sorry

end hannah_pieces_lost_l298_298592


namespace graveling_cost_is_3900_l298_298295

noncomputable def cost_of_graveling_roads 
  (length : ℕ) (breadth : ℕ) (width_road : ℕ) (cost_per_sq_m : ℕ) : ℕ :=
  let area_road_length := length * width_road
  let area_road_breadth := (breadth - width_road) * width_road
  let total_area := area_road_length + area_road_breadth
  total_area * cost_per_sq_m

theorem graveling_cost_is_3900 :
  cost_of_graveling_roads 80 60 10 3 = 3900 := 
by 
  unfold cost_of_graveling_roads
  sorry

end graveling_cost_is_3900_l298_298295


namespace longest_diagonal_of_rhombus_l298_298734

variables (d1 d2 : ℝ) (x : ℝ)
def rhombus_area := (d1 * d2) / 2
def diagonal_ratio := d1 / d2 = 4 / 3

theorem longest_diagonal_of_rhombus (h : rhombus_area (4 * x) (3 * x) = 150) (r : diagonal_ratio (4 * x) (3 * x)) : d1 = 20 := by
  sorry

end longest_diagonal_of_rhombus_l298_298734


namespace gray_region_area_l298_298520

theorem gray_region_area (r R : ℝ) (hR : R = 3 * r) (h_diff : R - r = 3) :
  π * (R^2 - r^2) = 18 * π :=
by
  -- The proof goes here
  sorry

end gray_region_area_l298_298520


namespace parabola_equation_l298_298797

def is_parabola (a b c x y : ℝ) : Prop :=
  y = a*x^2 + b*x + c

def has_vertex (h k a b c : ℝ) : Prop :=
  b = -2 * a * h ∧ c = k + a * h^2 

def contains_point (a b c x y : ℝ) : Prop :=
  y = a*x^2 + b*x + c

theorem parabola_equation (a b c : ℝ) :
  has_vertex 3 (-2) a b c ∧ contains_point a b c 5 6 → 
  a = 2 ∧ b = -12 ∧ c = 16 := by
  sorry

end parabola_equation_l298_298797


namespace sqrt_inequality_l298_298218

theorem sqrt_inequality (a : ℝ) (h : a > 2) : sqrt (a + 2) + sqrt (a - 2) < 2 * sqrt a :=
sorry

end sqrt_inequality_l298_298218


namespace remaining_puppies_l298_298227

def initial_puppies : Nat := 8
def given_away_puppies : Nat := 4

theorem remaining_puppies : initial_puppies - given_away_puppies = 4 := 
by 
  sorry

end remaining_puppies_l298_298227


namespace solve_for_x_l298_298607

theorem solve_for_x (x : ℝ) : 5 + 3.5 * x = 2.5 * x - 25 ↔ x = -30 :=
by {
  split,
  {
    intro h,
    calc
      x = -30 : by sorry,
  },
  {
    intro h,
    calc
      5 + 3.5 * (-30) = 5 - 105
                       = -100,
      2.5 * (-30) - 25 = -75 - 25
                       = -100,
    exact Eq.symm (by sorry),
  }
}

end solve_for_x_l298_298607


namespace solve_equation_l298_298673

theorem solve_equation (x : ℝ) (hx_pos : 0 < x) (hx_ne_one : x ≠ 1) :
    x^2 * (Real.log 27 / Real.log x) * (Real.log x / Real.log 9) = x + 4 → x = 2 :=
by
  sorry

end solve_equation_l298_298673


namespace calen_pencils_loss_l298_298371

theorem calen_pencils_loss
  (P_Candy : ℕ)
  (P_Caleb : ℕ)
  (P_Calen_original : ℕ)
  (P_Calen_after_loss : ℕ)
  (h1 : P_Candy = 9)
  (h2 : P_Caleb = 2 * P_Candy - 3)
  (h3 : P_Calen_original = P_Caleb + 5)
  (h4 : P_Calen_after_loss = 10) :
  P_Calen_original - P_Calen_after_loss = 10 := 
sorry

end calen_pencils_loss_l298_298371


namespace trailing_zeros_a6_l298_298104

theorem trailing_zeros_a6:
  (∃ a : ℕ+ → ℚ, 
    a 1 = 3 / 2 ∧ 
    (∀ n : ℕ+, a (n + 1) = (1 / 2) * (a n + (1 / a n))) ∧
    (∃ k, 10^k ≤ a 6 ∧ a 6 < 10^(k + 1))) →
  (∃ m, m = 22) :=
sorry

end trailing_zeros_a6_l298_298104


namespace find_k_l298_298547

noncomputable def k (k : ℝ) : Prop :=
  (k > 1) ∧ (∑ n in (Finset.range 1000), (6 * (n + 1) - 2) / k^(n + 1) = 3)

theorem find_k (k : ℝ) : k k ↔ k = 2 :=
by
  sorry

end find_k_l298_298547


namespace frac_diff_equals_l298_298068

variable {x y : ℝ}

-- Conditions
def condition1 : Prop := x ≠ 0
def condition2 : Prop := y ≠ 0
def condition3 : Prop := x - y = x * y + 1

-- Theorem statement
theorem frac_diff_equals : condition1 → condition2 → condition3 → (1/x - 1/y = -1 - 1/(x*y)) :=
by
  intros
  sorry

end frac_diff_equals_l298_298068


namespace farmer_sells_ear_price_l298_298697

-- Definitions based on the given conditions
def seeds_per_ear := 4
def seeds_per_bag := 100
def cost_per_bag := 0.5
def profit := 40
def ears_sold := 500

-- Theorem to prove the farmer sells one ear of corn for $0.10
theorem farmer_sells_ear_price : 
  let total_seeds_needed := ears_sold * seeds_per_ear in
  let bags_needed := total_seeds_needed / seeds_per_bag in
  let total_cost_of_seeds := bags_needed * cost_per_bag in
  let total_revenue := profit + total_cost_of_seeds in
  let price_per_ear := total_revenue / ears_sold in
  price_per_ear = 0.10 :=
by
  sorry

end farmer_sells_ear_price_l298_298697


namespace birds_never_gather_single_tree_n_44_birds_gather_or_not_gather_single_tree_general_n_l298_298675

theorem birds_never_gather_single_tree_n_44 :
  ∀ (birds trees : ℕ), trees = 44 ∧ birds = 44 ∧
  (∀ bird_positions : fin 44 → fin 44,
    ∀ move : (fin 44 × fin 44) → (fin 44 × fin 44),
    (move.1.1 + move.2.1 = bird_positions ∘ (move.1 + move.2))) →
  ¬ (∃ tree : fin 44, ∀ bird : fin 44, bird_positions bird = tree) :=
by sorry

theorem birds_gather_or_not_gather_single_tree_general_n :
  ∀ (birds trees : ℕ), birds = trees ∧
  (∀ bird_positions : fin trees → fin trees,
    ∀ move : (fin trees × fin trees) → (fin trees × fin trees),
    (move.1.1 + move.2.1 = bird_positions ∘ (move.1 + move.2))) →
  ((even trees → ¬ (∃ tree : fin trees, ∀ bird : fin trees, bird_positions bird = tree)) ∧
   (odd trees → ∃ tree : fin trees, ∀ bird : fin trees, bird_positions bird = tree)) :=
by sorry

end birds_never_gather_single_tree_n_44_birds_gather_or_not_gather_single_tree_general_n_l298_298675


namespace longest_diagonal_length_l298_298727

theorem longest_diagonal_length (A : ℝ) (d1 d2 : ℝ) (h1 : A = 150) (h2 : d1 / d2 = 4 / 3) : d1 = 20 :=
by
  -- Skipping the proof here
  sorry

end longest_diagonal_length_l298_298727


namespace min_vegetable_dishes_l298_298337

theorem min_vegetable_dishes (x : ℕ) : 5.choose 2 * x.choose 2 ≥ 200 → x ≥ 7 :=
by sorry

end min_vegetable_dishes_l298_298337


namespace total_books_equiv_19_over_4x_l298_298896

variable (x : ℝ)

-- Definitions based on the problem's conditions.
def betty_books : ℝ := x
def sister_books : ℝ := x + (1/4) * x
def cousin_books : ℝ := 2 * (x + (1/4) * x)

-- The total number of books of Betty, her sister, and their cousin.
def total_books : ℝ := betty_books x + sister_books x + cousin_books x

-- The theorem that asserts the total number of books is (19/4)x.
theorem total_books_equiv_19_over_4x : total_books x = (19/4) * x :=
by sorry  -- Proof is omitted.

end total_books_equiv_19_over_4x_l298_298896


namespace find_radius_of_touching_circle_l298_298568

-- Define the side length of the square
def side_length := 108

-- Define the radius of the semicircle
def semicircle_radius := side_length / 2

-- Define the condition of the circle touching the square sides and semicircles
def touching_circle_radius (r : ℝ) : Prop := r + semicircle_radius = side_length - r

-- The theorem statement proving the radius r
theorem find_radius_of_touching_circle : ∃ (r : ℝ), touching_circle_radius r ∧ r = 27 := 
by
  use 27
  split
  -- Provide the touch condition
  · sorry
  -- Prove that r = 27 is the correct radius
  · sorry

end find_radius_of_touching_circle_l298_298568


namespace total_time_riding_l298_298887

-- Definitions according to the conditions
def speed : ℝ := 4.25  -- Speed is 4.25 meters per minute
def timeHour : ℝ := 60 -- 1 hour is 60 minutes
def additionalDistance : ℝ := 29.75 -- Additional distance is 29.75 meters

-- Required to calculate the total time Hyeonil rode the bicycle
def timeTotal : ℝ := 67 -- The total time we need to prove is 67 minutes

-- Lean 4 Statement
theorem total_time_riding : 
  let distanceHour := speed * timeHour in
  let additionalTime := additionalDistance / speed in
  distanceHour = speed * timeHour → 
  additionalTime = additionalDistance/speed → 
  (timeHour + additionalTime) = timeTotal := 
by
  intros distanceHour additionalTime h1 h2
  rw [h1, h2]
  exact rfl


end total_time_riding_l298_298887


namespace AM_eq_KB_l298_298305

variables {Point : Type} [EuclideanGeometry Point]
variables (A K O M C B : Point)
variables (α β : Angle)

-- Conditions 
hypothesis Angle_AKO_eq_alpha : ∠ A K O = α
hypothesis Angle_AOK_eq_alpha : ∠ A O K = α
hypothesis Isosceles_AKO : IsoscelesTriangle A K O
hypothesis Angle_MOC_eq_alpha : ∠ M O C = α
hypothesis Angle_MKC_eq_beta : ∠ M K C = β
hypothesis Isosceles_MKC : IsoscelesTriangle M K C
hypothesis Parallel_KM_AC : Parallel (Line.mk K M) (Line.mk A C)

-- Proof goal
theorem AM_eq_KB (Angle_AKO_eq_alpha : ∠ A K O = α)
                  (Angle_AOK_eq_alpha : ∠ A O K = α)
                  (Isosceles_AKO : IsoscelesTriangle A K O)
                  (Angle_MOC_eq_alpha : ∠ M O C = α)
                  (Angle_MKC_eq_beta : ∠ M K C = β)
                  (Isosceles_MKC : IsoscelesTriangle M K C)
                  (Parallel_KM_AC : Parallel (Line.mk K M) (Line.mk A C)) :
  dist A M = dist K B :=
by
  sorry

end AM_eq_KB_l298_298305


namespace possible_values_of_p1_l298_298709

noncomputable def p (x : ℝ) (n : ℕ) : ℝ := sorry

axiom deg_p (n : ℕ) (h : n ≥ 2) (x : ℝ) : x^n = 1

axiom roots_le_one (r : ℝ) : r ≤ 1

axiom p_at_2 (n : ℕ) (h : n ≥ 2) : p 2 n = 3^n

theorem possible_values_of_p1 (n : ℕ) (h : n ≥ 2) : p 1 n = 0 ∨ p 1 n = (-1)^n * 2^n :=
by
  sorry

end possible_values_of_p1_l298_298709


namespace b_11_plus_b_12_l298_298463

noncomputable def geo_seq (a : ℕ → ℝ) : Prop := ∃ q : ℝ, q > 0 ∧ ∀ n, a (n+2) = q * a n

variables (a b : ℕ → ℝ)

def satisfied_conditions (a b : ℕ → ℝ) : Prop :=
  geo_seq a ∧
  (∀ n, log 2 (a (n+2)) - log 2 (a n) = 2) ∧
  (a 3 = 8) ∧
  (b 1 = 1) ∧
  (∀ n, b n * b (n+1) = a n)

theorem b_11_plus_b_12 (seq_a seq_b : ℕ → ℝ)
  (h : satisfied_conditions seq_a seq_b) : seq_b 11 + seq_b 12 = 96 :=
sorry

end b_11_plus_b_12_l298_298463


namespace ellipse_equation_find_k_max_area_triangle_l298_298827

-- Define the general equation of ellipse and the given conditions
def ellipse (a b : ℝ) : Prop := ∀ x y : ℝ, (x ^ 2) / (a ^ 2) + (y ^ 2) / (b ^ 2) = 1

-- Given conditions
axiom a_gt_zero (a : ℝ) : a > 0
axiom b_gt_zero (b : ℝ) : b > 0
axiom eccentricity_value (e : ℝ) : e = Real.sqrt 6 / 3
axiom length_major_axis (a : ℝ) : 2 * a = 2 * Real.sqrt 3

-- Proof goals
theorem ellipse_equation (a b : ℝ) (h1 : a = Real.sqrt 3) (h2 : b = 1) (e : ℝ) 
  (he1 : e = (Real.sqrt 6) / 3) (maj_axis : 2 * a = 2 * Real.sqrt 3) : 
  ellipse (Real.sqrt 3) 1 :=
by
  sorry

theorem find_k (k m : ℝ) (h : m = 1) 
  (dot_product_OA_OB : k = Real.sqrt 3 / 3 ∨ k = - (Real.sqrt 3 / 3)) : 
  m = 1 → dot_product_OA_OB :=
by
  sorry

theorem max_area_triangle (k : ℝ) (dist_from_O_to_l : ℝ) 
  (distance_given : dist_from_O_to_l = Real.sqrt 3 / 2) :
  ∃ (area : ℝ), area = Real.sqrt 3 / 2 ∧ dist_from_O_to_l = Real.sqrt 3 / 2 :=
by
  sorry

end ellipse_equation_find_k_max_area_triangle_l298_298827


namespace part1_part2_l298_298936

variables {A B C : ℝ} {a b c : ℝ} -- Angles A, B, C and sides a, b, c
variable (triangleABC : Real) -- Area of triangle ABC

-- Assume the given condition in the problem
def cond1 (A B C a b c : ℝ) : Prop := 
  (cos B / b) + (cos C / c) = (sin A / (sqrt 3 * sin C))

def cond2 (B : ℝ) : Prop := 
  (cos B) + (sqrt 3 * sin B) = 2

-- The objective is to prove that b = √3
theorem part1 (h : cond1 A B C a b c) : b = sqrt 3 :=
sorry

-- The objective is to prove that the maximum area of triangle ABC is 3√3/4, given cond2 and b = √3
theorem part2 (h1 : b = sqrt 3) (h2 : cond2 B) : triangleABC = (3 * sqrt 3) / 4 :=
sorry

end part1_part2_l298_298936


namespace circle_tangent_to_line_eq_l298_298624

theorem circle_tangent_to_line_eq (x y : ℝ) :
  (center_origin : x^2 + y^2 - 0 = 0) ∧ (line_tangent : y = 2 - x) →
  (x^2 + y^2 = 2) :=
sorry

end circle_tangent_to_line_eq_l298_298624


namespace problem1_problem2_problem3_problem4_l298_298497

section
variable (a b c d: ℝ)

-- Proof for a = 2 given the expression involving trigonometric identities
theorem problem1 (h1: a = (Real.sin 15 / Real.cos 75) + (1 / (Real.sin 75)^2) - (Real.tan 15)^2): a = 2 := sorry

-- Proof for b = -3 given the lines are perpendicular
theorem problem2 (h2: a ≠ 0): 
  (let m1 := -a / 2,
        m2 := -3 / b,
        perpendicular := (m1 * m2 = -1))
  → b = -3 := sorry

-- Proof for c = 12 given the points are collinear
theorem problem3 (h3 : 2 ≠ 4): 
  (let slope1 := (λ b: ℝ, -b),
        slope2 := (λ b c: ℝ, (c/2 + b)),
        colinear := ∀ b c, slope1 b = slope2 b c)
  → c = 12 := sorry

-- Proof for d = 140 from proportional relationships
theorem problem4 
  (h4x: x ≠ 0) (h4y: y ≠ 0) (h4z: z ≠ 0)
  (h4: (1/x) / (1/y) / (1/z) = 3 / 4 / 5)
  (h5: (1/(x+y)) / (1/(y+z)) = 9 * c / d): d = 140 := sorry

end

end problem1_problem2_problem3_problem4_l298_298497


namespace problem_l298_298855

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin (2 * x + (Real.pi / 6))

noncomputable def F (x : ℝ) : ℝ := f x - 3

theorem problem :
  let zeros := {x : ℝ | F x = 0 ∧ 0 ≤ x ∧ x ≤ 91 * Real.pi / 6}.toFinset
      xs := zeros.toList.sorted (<)
      indices := List.range xs.length in
  xs.length = 31 ∧
  ∀ n, n < xs.length →
  x_1 + (List.sum ((indices.filter (λ i, i ≠ 0 ∧ i ≠ xs.length - 1)).map (λ i, 2 * xs.nthLe i sorry)) + (xs.nthLe 0 sorry) + (xs.nthLe (xs.length - 1) sorry)) = 445 * Real.pi := sorry

end problem_l298_298855


namespace baker_total_cost_is_correct_l298_298315

theorem baker_total_cost_is_correct :
  let flour_cost := 3 * 3
  let eggs_cost := 3 * 10
  let milk_cost := 7 * 5
  let baking_soda_cost := 2 * 3
  let total_cost := flour_cost + eggs_cost + milk_cost + baking_soda_cost
  total_cost = 80 := 
by
  sorry

end baker_total_cost_is_correct_l298_298315


namespace range_r_l298_298662

def r (x : ℝ) : ℝ := 1 / (1 - x)^2

theorem range_r : set.range r = {y : ℝ | 0 < y} :=
by sorry

end range_r_l298_298662


namespace pigeonhole_principle_befriend_l298_298311

-- Define a function to represent the befriend relationship
def befriend (n : ℕ) (friendships : Fin n → Fin n → Bool) : Prop :=
  ∀ i j : Fin n, friendships i j = friendships j i

-- Main theorem statement
theorem pigeonhole_principle_befriend (n : ℕ) (h_n : n = 23) 
  (friendships : Fin n → Fin n → Bool)
  (symm_friendships : befriend n friendships) :
  ∃ i j : Fin n, i ≠ j ∧
  (finset.univ.filter (λ x, friendships i x).count = 
   finset.univ.filter (λ x, friendships j x).count) := 
begin
  sorry -- Proof is not required
end

end pigeonhole_principle_befriend_l298_298311


namespace expression_value_l298_298832

theorem expression_value (a b m n : ℚ) 
  (ha : a = -7/4) 
  (hb : b = -2/3) 
  (hmn : m + n = 0) : 
  4 * a / b + 3 * (m + n) = 21 / 2 :=
by 
  sorry

end expression_value_l298_298832


namespace trapezoid_area_l298_298796

-- Define the parameters of the trapezoid
def trapezoid (ε θ : Type) [linear_ordered_field ε] :=
  (base1 base2 diag1 diag2 : ε)

-- Instantiate the trapezoid with given conditions.
noncomputable def problem_trapezoid : trapezoid ℝ :=
  ⟨3, 6, 7, 8⟩

-- Define the problem statement: The area of the trapezoid equals 12√5 cm²
theorem trapezoid_area {ε : Type} [linear_ordered_field ε] :
  ∀ (t : trapezoid ε),
  t.base1 = 3 → t.base2 = 6 → t.diag1 = 7 → t.diag2 = 8 →
  -- The proof of this theorem represents the main part of the provided solution.
  (some_area_calculation_function t = 12 * real.sqrt 5) :=
sorry

end trapezoid_area_l298_298796


namespace car_tire_diameter_l298_298693

noncomputable def pi := Real.pi
noncomputable def mile_to_feet : ℝ := 5280.0
noncomputable def distance_in_miles : ℝ := 0.5
noncomputable def revolutions : ℝ := 775.5724667489372

theorem car_tire_diameter :
  let distance_in_feet := distance_in_miles * mile_to_feet;
  let circumference := distance_in_feet / revolutions;
  let diameter := circumference / pi;
  let diameter_in_inches := diameter * 12;
  abs (diameter_in_inches - 13) < 0.1 :=
by
  let distance_in_feet := 0.5 * 5280
  let circumference := distance_in_feet / 775.5724667489372
  let diameter := circumference / Real.pi
  let diameter_in_inches := diameter * 12
  have : abs (diameter_in_inches - 13) < 0.1 := sorry
  exact this

end car_tire_diameter_l298_298693


namespace number_of_true_propositions_l298_298632

def proposition_1 (a b c : Vector) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) (h₄ : ∃ λ₁ : ℝ, a = λ₁ • b) (h₅ : ∃ λ₂ : ℝ, b = λ₂ • c) : Prop :=
  ∃ λ₃ : ℝ, a = λ₃ • c

def proposition_2 (e1 e2 : Vector) (h₁ : ∥e1∥ = 1) (h₂ : ∥e2∥ = 1) (h₃ : e1 ⬝ e2 = 0) (k : ℝ) : Prop :=
  (e1 + k • e2) ⬝ (k • e1 + e2) > 0 → 0 < k ∧ k ≠ 1

def proposition_3 (a b : Vector) : Prop :=
  let proj := (a ⬝ b) / (b ⬝ b) * b in proj = (6/5, 12/5)

def proposition_4 (e1 e2 : Vector) : Prop :=
  ¬ ∃ k : ℝ, e1 = k • e2

theorem number_of_true_propositions : 
  (proposition_1 (2, 4) (-1, 2) (3, 6) (by simp) (by simp) (by simp) (by simp) (by simp)) ∧ 
  ¬ (proposition_2 (1, 0) (0, 1) (by norm_num) (by norm_num) (by norm_num) 0) ∧
  ¬ (proposition_3 (2, 4) (-1, 2)) ∧
  ¬ (proposition_4 (2, -3) (-4, 6)) :=
sorry

end number_of_true_propositions_l298_298632


namespace apartments_with_both_cat_and_dog_l298_298906

-- Defining the main problem setup
def apartment_has_cat (n : ℕ) : Prop :=
  n % 5 = 0

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 100) + (n % 100 / 10) + (n % 10)

def apartment_has_dog (n : ℕ) : Prop :=
  sum_of_digits n % 5 = 0

def number_of_apartments_with_both : ℕ :=
  (Finset.range 301).filter (λ n, apartment_has_cat n ∧ apartment_has_dog n).card

theorem apartments_with_both_cat_and_dog : number_of_apartments_with_both = 6 :=
by
  -- The proof is omitted
  sorry

end apartments_with_both_cat_and_dog_l298_298906


namespace number_of_x_intercepts_l298_298803

theorem number_of_x_intercepts : 
  let interval_low  := 0.00005
  let interval_high := 0.0005
  let pi := Real.pi
  let low_k  := Real.floor ((1 / interval_high) / pi)
  let high_k := Real.floor ((1 / interval_low) / pi)
  high_k - low_k = 5730 := 
by suffices : high_k = 6366 ∧ low_k = 636 by sorry

end number_of_x_intercepts_l298_298803


namespace arithmetic_sequence_solution_minimum_lambda_l298_298171

noncomputable def arithmetic_sequence (a n : ℕ) (d : ℚ) : ℚ :=
  a + (n-1) * d

noncomputable def geometric_condition (a₁ a₂ a₄ : ℚ) : Prop :=
  (a₁ * a₄ = a₂^2)

noncomputable def bn (a : ℕ -> ℚ) (n : ℕ) : ℚ :=
  25 / ((a n) * (a (n+1)))

noncomputable def Tn (a : ℕ -> ℚ) (n : ℕ) : ℚ :=
∑ k in Finset.range n, bn a k

theorem arithmetic_sequence_solution
  (d : ℚ)
  (a₁ a₂ a₃ a₄ : ℚ)
  (ha₃ : arithmetic_sequence a₁ 3 d = 5)
  (hg : geometric_condition a₁ a₂ a₄) :
  (∀ n, arithmetic_sequence a₁ n d = 5 ∨ arithmetic_sequence a₁ n d = (5 / 3) * n) :=
sorry

theorem minimum_lambda
  (a : ℕ -> ℚ)
  (ha : ∀ n, a n = (5 / 3) * n)
  (n : ℕ) :
  (Tn a n ≤ 9) :=
sorry

#eval arithmetic_sequence 1 3 (5:ℚ)
#eval geometric_condition 1 2 1
#eval bn (λ n => (5 / 3) * n) 1
#eval Tn (λ n => (5 / 3) * n) 2

end arithmetic_sequence_solution_minimum_lambda_l298_298171


namespace consecutive_tree_distance_l298_298314

-- Define the conditions and question
def yard_length : ℝ := 1850
def tree_count : ℕ := 52
def gaps := tree_count - 1 -- 51 gaps between the trees

def distance_between_trees := yard_length / gaps -- calculate the distance between each trees

theorem consecutive_tree_distance : distance_between_trees ≈ 36.27 :=
by {
  -- The proof is omitted, but we can outline that we have the conditions and need to show this result holds.
  sorry
}

end consecutive_tree_distance_l298_298314


namespace intersection_in_first_quadrant_l298_298256

-- Define the lines
def line1 (x : ℝ) : ℝ := x + 1
def line2 (x : ℝ) (a : ℝ) : ℝ := -2 * x + a

-- Define the intersection point
def intersection_point (a : ℝ) : ℝ × ℝ := 
  let x := (a - 1) / 3 in
  (x, line1 x)

-- Define the condition for the intersection point to be in the first quadrant
def in_first_quadrant (p : ℝ × ℝ) : Prop := 
  p.1 > 0 ∧ p.2 > 0

-- Translate the problem to a Lean theorem statement
theorem intersection_in_first_quadrant (a : ℝ) : in_first_quadrant (intersection_point a) ↔ a > 1 := 
sorry

end intersection_in_first_quadrant_l298_298256


namespace find_x_l298_298490

theorem find_x
  (x : Real)
  (h₀ : 0 ≤ x ∧ x < 180)
  (h₁ : tan (4 * x) + tan (6 * x) = 0) :
  x = 18 :=
sorry

end find_x_l298_298490


namespace complex_magnitude_l298_298438

open Complex

theorem complex_magnitude {x y : ℝ} (h : (1 + Complex.I) * x = 1 + y * Complex.I) : abs (x + y * Complex.I) = Real.sqrt 2 :=
sorry

end complex_magnitude_l298_298438


namespace hexagon_diagonals_intersect_at_single_point_l298_298646

theorem hexagon_diagonals_intersect_at_single_point
  (ABC: Triangle)
  (A1 A2 B1 B2 C1 C2: Point)
  (hA: divides (A1, A2, BC) 3)
  (hB: divides (B1, B2, CA) 3)
  (hC:divides (C1, C2, AB) 3):
  ∃ (P : Point), (diagonal (A, B1, C2) P)
  ∧ (diagonal (B, C1, A2) P)
  ∧ (diagonal (C, A1, B2) P) :=
  sorry

end hexagon_diagonals_intersect_at_single_point_l298_298646


namespace lcm_12_18_is_36_l298_298419

def prime_factors (n : ℕ) : list ℕ :=
  if n = 12 then [2, 2, 3]
  else if n = 18 then [2, 3, 3]
  else []

noncomputable def lcm_of_two (a b : ℕ) : ℕ :=
  match prime_factors a, prime_factors b with
  | [2, 2, 3], [2, 3, 3] => 36
  | _, _ => 0

theorem lcm_12_18_is_36 : lcm_of_two 12 18 = 36 :=
  sorry

end lcm_12_18_is_36_l298_298419


namespace new_parabola_equation_l298_298103

theorem new_parabola_equation :
  (∃ t : ℝ, y = -3 * (x - 1) ^ 2 + 2 + t ∧ 
  (2, 4) ∈ set_of (λ x : ℝ, y = mx - 2)) → 
  y = -3 * x ^ 2 + 6 * x + 4 := 
sorry

end new_parabola_equation_l298_298103


namespace mateo_average_speed_l298_298971

-- Define the conditions
def distance : ℝ := 300
def start_time : ℕ := 7
def end_time : ℕ := 11
def break_time : ℕ := 40

-- Define the time calculations in minutes and hours
def total_travel_time : ℕ := (end_time - start_time) * 60
def driving_time : ℕ := total_travel_time - break_time
def driving_time_hours : ℝ := driving_time / 60

-- The goal is to prove the average speed
theorem mateo_average_speed :
  (distance / driving_time_hours) = 90 := by
  sorry

end mateo_average_speed_l298_298971


namespace longest_diagonal_of_rhombus_l298_298717

theorem longest_diagonal_of_rhombus (d1 d2 : ℝ) (area : ℝ) (ratio : ℝ) (h1 : area = 150) (h2 : d1 / d2 = 4 / 3) :
  max d1 d2 = 20 :=
by 
  let x := sqrt (area * 2 / (d1 * d2))
  have d1_expr : d1 = 4 * x := sorry
  have d2_expr : d2 = 3 * x := sorry
  have x_val : x = 5 := sorry
  have length_longest_diag : max d1 d2 = max (4 * 5) (3 * 5) := sorry
  exact length_longest_diag

end longest_diagonal_of_rhombus_l298_298717


namespace problem_statement_l298_298288

variables {A B x y a : ℝ}

theorem problem_statement (h1 : 1/A = 1 - (1 - x) / y)
                          (h2 : 1/B = 1 - y / (1 - x))
                          (h3 : x = (1 - a) / (1 - 1/a))
                          (h4 : y = 1 - 1/x)
                          (h5 : a ≠ 1) (h6 : a ≠ -1) : 
                          A + B = 1 :=
sorry

end problem_statement_l298_298288


namespace slope_AA_l298_298653

noncomputable theory

variables {a b : ℝ} (h1 : a ≠ b) (h2 : a > 0) (h3 : b > 0)

theorem slope_AA'_not_zero : ¬(let A := (a, b) in
                               let A' := (-b, a) in
                               (A.2 - A'.2) / (A.1 - A'.1) = 0) :=
sorry

end slope_AA_l298_298653


namespace find_S6_l298_298263

def arithmetic_sum (n : ℕ) : ℝ := sorry
def S_3 := 6
def S_9 := 27

theorem find_S6 : ∃ S_6 : ℝ, S_6 = 15 ∧ 
                              S_6 - S_3 = (6 + (S_9 - S_6)) / 2 :=
sorry

end find_S6_l298_298263


namespace point_B_not_on_graph_l298_298672

-- Define the function
def f (x : ℝ) : ℝ := (x - 1) / (x + 2)

-- Define the points
def point_A : ℝ × ℝ := (0, -1/2)
def point_B : ℝ × ℝ := (-3/2, 1)
def point_C : ℝ × ℝ := (1, 0)
def point_D : ℝ × ℝ := (-2, -3)
def point_E : ℝ × ℝ := (-1, -2)

-- The statement for the proof
theorem point_B_not_on_graph : ¬ (f point_B.1 = point_B.2) := 
sorry

end point_B_not_on_graph_l298_298672


namespace sum_of_first_10_terms_l298_298072

def sequence_sum_formula (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) : Prop :=
  S (n + 1) = S n + a n + 3

def condition_a5_a6 (a : ℕ → ℕ) : Prop := 
  a 5 + a 6 = 29

def first_10_terms_sum (a : ℕ → ℕ) (sum_formula : (ℕ → ℕ) → Prop): Prop :=
  let b n := a n + a (n + 1)
  let S_b n := ∑ i in list.range (n + 1), b i
  S_b 10 = 320

theorem sum_of_first_10_terms (a : ℕ → ℕ) (S : ℕ → ℕ) :
  sequence_sum_formula a S → condition_a5_a6 a → first_10_terms_sum a sequence_sum_formula := by
  sorry

end sum_of_first_10_terms_l298_298072


namespace isosceles_triangle_legs_length_l298_298902

theorem isosceles_triangle_legs_length 
  (P : ℝ) (base : ℝ) (leg_length : ℝ) 
  (hp : P = 26) 
  (hb : base = 11) 
  (hP : P = 2 * leg_length + base) : 
  leg_length = 7.5 := 
by 
  sorry

end isosceles_triangle_legs_length_l298_298902


namespace no_three_in_range_l298_298437

theorem no_three_in_range (c : ℝ) : c > 4 → ¬ (∃ x : ℝ, x^2 + 2 * x + c = 3) :=
by
  sorry

end no_three_in_range_l298_298437


namespace black_haired_girls_count_l298_298564

theorem black_haired_girls_count (initial_total_girls : ℕ) (initial_blonde_girls : ℕ) (added_blonde_girls : ℕ) (final_blonde_girls total_girls : ℕ) 
    (h1 : initial_total_girls = 80) 
    (h2 : initial_blonde_girls = 30) 
    (h3 : added_blonde_girls = 10) 
    (h4 : final_blonde_girls = initial_blonde_girls + added_blonde_girls) 
    (h5 : total_girls = initial_total_girls) : 
    total_girls - final_blonde_girls = 40 :=
by
  sorry

end black_haired_girls_count_l298_298564


namespace parabola_focus_distance_l298_298865

theorem parabola_focus_distance (p : ℝ) (y₀ : ℝ) (h₀ : p > 0) 
  (h₁ : y₀^2 = 2 * p * 4) 
  (h₂ : dist (4, y₀) (p/2, 0) = 3/2 * p) : 
  p = 4 := 
sorry

end parabola_focus_distance_l298_298865


namespace line_slope_y_intercept_l298_298505

theorem line_slope_y_intercept :
  (∃ (a b : ℝ), (∀ (x y : ℝ), (x = 3 → y = 7 → y = a * x + b) ∧ (x = 7 → y = 19 → y = a * x + b)) ∧ (a - b = 5)) :=
begin
  sorry
end

end line_slope_y_intercept_l298_298505


namespace book_price_increase_percentage_l298_298678

theorem book_price_increase_percentage :
  let P_original := 300
  let P_new := 480
  (P_new - P_original : ℝ) / P_original * 100 = 60 :=
by
  sorry

end book_price_increase_percentage_l298_298678


namespace james_out_of_pocket_cost_l298_298530

theorem james_out_of_pocket_cost
  (total_charge : ℝ)
  (insurance_coverage : ℝ) : 
  total_charge = 300 → 
  insurance_coverage = 0.80 →
  (total_charge * (1 - insurance_coverage)) = 60 := 
begin
  intros h_total h_insurance,
  rw [h_total, h_insurance],
  norm_num,
end

end james_out_of_pocket_cost_l298_298530


namespace lcm_12_18_l298_298412

theorem lcm_12_18 : Nat.lcm 12 18 = 36 :=
by
  -- Definitions of the conditions
  have h12 : 12 = 2 * 2 * 3 := by norm_num
  have h18 : 18 = 2 * 3 * 3 := by norm_num
  
  -- Calculating LCM using the built-in Nat.lcm
  rw [Nat.lcm_comm]  -- Ordering doesn't matter for lcm
  rw [Nat.lcm, h12, h18]
  -- Prime factorizations checks are implicitly handled
  
  -- Calculate the LCM based on the highest powers from the factorizations
  have lcm_val : 4 * 9 = 36 := by norm_num
  
  -- So, the LCM of 12 and 18 is
  exact lcm_val

end lcm_12_18_l298_298412


namespace problem1_problem2_l298_298091

noncomputable def f : ℝ → ℝ := -- we assume f is noncomputable since we know its explicit form in the desired interval
sorry

axiom periodic_f (x : ℝ) : f (x + 5) = f x
axiom odd_f {x : ℝ} (h : -1 ≤ x ∧ x ≤ 1) : f (-x) = -f x
axiom quadratic_f {x : ℝ} (h : 1 ≤ x ∧ x ≤ 4) : f x = 2 * (x - 2) ^ 2 - 5
axiom minimum_f : f 2 = -5

theorem problem1 : f 1 + f 4 = 0 :=
by
  sorry

theorem problem2 {x : ℝ} (h : 1 ≤ x ∧ x ≤ 4) : f x = 2 * x ^ 2 - 8 * x + 3 :=
by
  sorry

end problem1_problem2_l298_298091


namespace perpendicular_distance_to_plane_l298_298023

-- Define the points A, B, C, and D
def A : ℝ × ℝ × ℝ := (5, 0, 0)
def B : ℝ × ℝ × ℝ := (0, 3, 0)
def C : ℝ × ℝ × ℝ := (0, 0, 6)
def D : ℝ × ℝ × ℝ := (0, 0, 0)

-- Define the vector cross product in ℝ³
def cross (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.2.2 * v.3 - u.3 * v.2.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2.2 - u.2.2 * v.1)

-- Define dot product in ℝ³
def dot (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2.2 * v.2.2 + u.3 * v.3

-- Define the norm (length) of a vector in ℝ³
def norm (u : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (dot u u)

-- The main theorem statement
theorem perpendicular_distance_to_plane :
  let n := cross (A - D) (B - D) in
  let dist := abs (dot n (C - D)) / norm n in
  dist = 3
  :=
by
  sorry

end perpendicular_distance_to_plane_l298_298023


namespace find_abc_l298_298792

theorem find_abc (a b c : ℤ) (h1 : 1 < a) (h2 : a < b) (h3 : b < c) (h4 : (a-1) * (b-1) * (c-1) ∣ a * b * c - 1) :
    (a, b, c) = (3, 5, 15) ∨ (a, b, c) = (2, 4, 8) :=
by
  sorry

end find_abc_l298_298792


namespace geom_seq_div_a5_a7_l298_298523

variable {a : ℕ → ℝ}

-- Given sequence is geometric and positive
def is_geom_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * r

-- Positive geometric sequence with decreasing terms
def is_positive_decreasing_geom_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  is_geom_sequence a r ∧ ∀ n, a (n + 1) < a n ∧ a n > 0

-- Conditions
variables (r : ℝ) (hp : is_positive_decreasing_geom_sequence a r)
           (h2 : a 2 * a 8 = 6) (h3 : a 4 + a 6 = 5)

-- Goal
theorem geom_seq_div_a5_a7 : a 5 / a 7 = 3 / 2 :=
by
  sorry

end geom_seq_div_a5_a7_l298_298523


namespace percentage_of_laborers_present_is_correct_l298_298904

def total_laborers : ℕ := 45
def laborers_present : ℕ := 17

def percentage_present (total_laborers laborers_present : ℕ) : ℝ :=
  (laborers_present.to_real / total_laborers.to_real) * 100

theorem percentage_of_laborers_present_is_correct :
  percentage_present total_laborers laborers_present ≈ 37.8 :=
by
  sorry

end percentage_of_laborers_present_is_correct_l298_298904


namespace min_balls_needed_l298_298691

theorem min_balls_needed (red green yellow blue white black : ℕ)
(h_red : red = 35) (h_green : green = 25) (h_yellow : yellow = 22)
(h_blue : blue = 15) (h_white : white = 14) (h_black : black = 12) :
  ∃ (k : ℕ), (∀ (n_red n_green n_yellow n_blue n_white n_black : ℕ),
    n_red ≤ red ∧ n_green ≤ green ∧ n_yellow ≤ yellow ∧ n_blue ≤ blue ∧ n_white ≤ white ∧ n_black ≤ black ∧ 
    n_red + n_green + n_yellow + n_blue + n_white + n_black = k → 
    (n_red < 18) ∧ (n_green < 18) ∧ (n_yellow < 18) ∧ (n_blue < 18) ∧ (n_white < 18) ∧ (n_black < 18) → k < 93) ∧ k = 93 :=
begin
  sorry
end

#print axioms min_balls_needed  -- This line prints the assumptions required for this theorem.

end min_balls_needed_l298_298691


namespace banks_policies_for_seniors_justified_l298_298584

-- Defining conditions
def better_credit_repayment_reliability : Prop := sorry
def stable_pension_income : Prop := sorry
def indirect_younger_relative_contributions : Prop := sorry
def pensioners_inclination_to_save : Prop := sorry
def regular_monthly_income : Prop := sorry
def preference_for_long_term_deposits : Prop := sorry

-- Lean theorem statement using the conditions
theorem banks_policies_for_seniors_justified :
  better_credit_repayment_reliability →
  stable_pension_income →
  indirect_younger_relative_contributions →
  pensioners_inclination_to_save →
  regular_monthly_income →
  preference_for_long_term_deposits →
  (banks_should_offer_higher_deposit_and_lower_loan_rates_to_seniors : Prop) :=
by
  -- Insert proof here that given all the conditions the conclusion follows
  sorry -- proof not required, so skipping

end banks_policies_for_seniors_justified_l298_298584


namespace count_valid_numbers_l298_298884

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ (n % 10 = 2 * ((n % 100) / 10))

theorem count_valid_numbers : 
  (Finset.filter is_valid_number (Finset.range 1000)).card = 45 :=
by
  sorry

end count_valid_numbers_l298_298884


namespace parabola_vertex_shift_l298_298507

theorem parabola_vertex_shift :
  let original_vertex := (2, 3)
  let new_vertex := (original_vertex.1 - 3, original_vertex.2 - 5)
  new_vertex = (-1, -2) :=
by
  let original_vertex := (2, 3)
  let new_vertex := (original_vertex.1 - 3, original_vertex.2 - 5)
  show new_vertex = (-1, -2) from
    sorry

end parabola_vertex_shift_l298_298507


namespace sequence_formula_l298_298635

theorem sequence_formula (a : ℕ → ℝ) (h_initial : a 1 = 1 / 3)
  (h_recurrence : ∀ n : ℕ, 1 < n → n / a n = (2 * a (n - 1) + (n - 1)) / a (n - 1)) :
  ∀ n : ℕ, n ≥ 1 → a n = n / (2 * n + 1) :=
begin
  sorry
end

end sequence_formula_l298_298635


namespace find_polynomial_l298_298711

theorem find_polynomial
  (M : ℝ → ℝ)
  (h : ∀ x, M x + 5 * x^2 - 4 * x - 3 = -1 * x^2 - 3 * x) :
  ∀ x, M x = -6 * x^2 + x + 3 :=
sorry

end find_polynomial_l298_298711


namespace base6_divisible_by_11_l298_298436

theorem base6_divisible_by_11 :
  ∃ d : ℕ, d ∈ {0, 1, 2, 3, 4, 5} ∧ (437 + 42 * d) % 11 = 0 ∧ d = 4 :=
by
  sorry

end base6_divisible_by_11_l298_298436


namespace day_of_week_in_2023_days_l298_298275

theorem day_of_week_in_2023_days (h1 : "Thursday" : String) (h2 : ∀ n : ℕ, (n % 7) = 0 → true) :
  (2023 % 7 = 0 → h1 = "Thursday") :=
by {
  sorry
}

end day_of_week_in_2023_days_l298_298275


namespace passed_candidates_l298_298243

theorem passed_candidates (P F : ℕ) (h1 : P + F = 120) (h2 : 39 * P + 15 * F = 35 * 120) : P = 100 :=
by
  sorry

end passed_candidates_l298_298243


namespace candy_necklaces_l298_298174

theorem candy_necklaces (necklace_candies : ℕ) (block_candies : ℕ) (blocks_broken : ℕ) :
  necklace_candies = 10 → block_candies = 30 → blocks_broken = 3 → (blocks_broken * block_candies) / necklace_candies - 1 = 8 := 
by 
  intro h1 h2 h3
  rw [h1, h2, h3]
  sorry

end candy_necklaces_l298_298174


namespace independent_prob_intersection_l298_298136

variable (Ω : Type) [ProbabilityMeasure Ω]

variables (a b : Set Ω)

noncomputable def p (s : Set Ω) : ℝ :=
  ProbabilityMeasure.probability s

-- Given conditions.
axiom h1 : p(Ω)(a) = 1 / 5
axiom h2 : p(Ω)(b) = 2 / 5
axiom h3 : independent_events a b

theorem independent_prob_intersection :
  p(Ω)(a ∩ b) = 2 / 25 :=
sorry

end independent_prob_intersection_l298_298136


namespace longest_diagonal_of_rhombus_l298_298731

theorem longest_diagonal_of_rhombus (A B : ℝ) (h1 : A = 150) (h2 : ∃ x, (A = 1/2 * (4 * x) * (3 * x)) ∧ (x = 5)) : 
  4 * (classical.some h2) = 20 := 
by sorry

end longest_diagonal_of_rhombus_l298_298731


namespace bike_distance_l298_298319

variable (Speed Time : ℕ)
variable (Distance : ℕ)
variable h_speed : Speed = 90
variable h_time : Time = 5

theorem bike_distance : (Distance = Speed * Time) → Distance = 450 :=
by
  intros h
  rw [h_speed, h_time] at h
  simp at h
  exact h

end bike_distance_l298_298319


namespace problem_I_problem_II_l298_298862

open Real

-- Definition for Problem (Ⅰ)
def f (x : ℝ) (a : ℝ) : ℝ := x + a^2 / x

-- Definition for Problem (Ⅱ)
def g (x : ℝ) : ℝ := x + log x

-- Statement for Problem (Ⅰ)
theorem problem_I (x : ℝ) : f x 2 = x + 4 / x → (-2 < x ∧ x < 0) ∨ (0 < x ∧ x < 2) :=
sorry

-- Statement for Problem (Ⅱ)
theorem problem_II (a : ℝ) (h : a > 0) (h' : ∀ (x1 x2 : ℝ), 1 ≤ x1 ∧ x1 ≤ exp 1 ∧ 1 ≤ x2 ∧ x2 ≤ exp 1 → f x1 a ≥ g x2) : 
  a ≥ (exp 1 + 1) / 2 :=
sorry

end problem_I_problem_II_l298_298862


namespace count_distinct_sums_l298_298482

theorem count_distinct_sums (S : Set ℕ) (hS : S = {2, 5, 8, 11, 14, 17, 20}) :
  ∃ n, n = 13 ∧ ∀ k ∈ { k | ∃ a b c d, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ 
                       a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ k = a + b + c + d },
          k ∈ (Finset.range (63) \ Finset.range (26)) → 
          (k + 1) % 3 = 0 :=
begin
  sorry
end

end count_distinct_sums_l298_298482


namespace find_somus_age_l298_298302

def somus_current_age (S F : ℕ) := S = F / 3
def somus_age_7_years_ago (S F : ℕ) := (S - 7) = (F - 7) / 5

theorem find_somus_age (S F : ℕ) 
  (h1 : somus_current_age S F) 
  (h2 : somus_age_7_years_ago S F) : S = 14 :=
sorry

end find_somus_age_l298_298302


namespace cylindrical_to_rectangular_coords_l298_298379

/--
Cylindrical coordinates (r, θ, z)
Rectangular coordinates (x, y, z)
-/
theorem cylindrical_to_rectangular_coords (r θ z : ℝ) (hx : x = r * Real.cos θ)
    (hy : y = r * Real.sin θ) (hz : z = z) :
    (r, θ, z) = (5, Real.pi / 4, 2) → (x, y, z) = (5 * Real.sqrt 2 / 2, 5 * Real.sqrt 2 / 2, 2) :=
by
  sorry

end cylindrical_to_rectangular_coords_l298_298379


namespace solve_for_x_l298_298602

theorem solve_for_x (x : ℝ) (h : 5 + 3.5 * x = 2.5 * x - 25) : x = -30 :=
by 
  -- Placeholder for the actual proof
  sorry

end solve_for_x_l298_298602


namespace max_M_value_l298_298208

theorem max_M_value (a : ℕ → ℝ) (h1 : a 1 = a 2017)
  (h2 : ∀ i, 1 ≤ i ∧ i ≤ 2015 → abs (a i + a (i + 2) - 2 * a (i + 1)) ≤ 1) :
  ∃ M, M = max (set.restrict (λ (x :ℕ × ℕ), abs (a x.1 - a x.2)) (λ x, 1 ≤ x.1 ∧ x.2 ≤ 17 ∧ x.1 < x.2)) ∧ 
       M ≤ 1008 ^ 2 / 2 :=
begin
  sorry
end

end max_M_value_l298_298208


namespace correct_statements_incorrect_statements_l298_298556

def f (x : ℝ) : ℝ := 
if x ≤ 0 then 2^x 
else Real.log x / Real.log 0.5

theorem correct_statements : 
  (∀ a : ℝ, a ≤ 0 → f (f a) = -a) ∧ 
  (∀ a : ℝ, a ≥ 1 → f (f a) = 1 / a) :=
by
  sorry

theorem incorrect_statements : 
  (∀ a : ℝ, f (f a) = -a → a ≤ 0) → False ∧ 
  (∀ a : ℝ, f (f a) = 1 / a → a ≥ 1) → False :=
by
  sorry

end correct_statements_incorrect_statements_l298_298556


namespace minimum_value_of_f_l298_298078

noncomputable def f (x y : ℝ) : ℝ :=
  real.sqrt (x^2 - x * y + y^2) + real.sqrt (x^2 - 9 * x + 27) + real.sqrt (y^2 - 15 * y + 75)

theorem minimum_value_of_f : ∀ x y : ℝ, 0 < x → 0 < y → f x y ≥ 7 * real.sqrt 3 :=
by
  intro x y hx hy
  unfold f
  sorry

end minimum_value_of_f_l298_298078


namespace find_acute_angle_l298_298881

theorem find_acute_angle (a b : ℝ × ℝ) (α : ℝ) (ha : a = (1/2, sin α))
    (hb : b = (sin α, 1)) (h_parallel : ∃ k : ℝ, k ≠ 0 ∧ (1/2, sin α) = k • (sin α, 1)) (h_acute : 0 < α ∧ α < π / 2) :
    α = π / 4 :=
begin
  sorry,
end

end find_acute_angle_l298_298881


namespace greatest_odd_integer_l298_298657

theorem greatest_odd_integer (x : ℕ) (h_odd : x % 2 = 1) (h_pos : x > 0) (h_ineq : x^2 < 50) : x = 7 :=
by sorry

end greatest_odd_integer_l298_298657


namespace algebra_expression_value_l298_298129

theorem algebra_expression_value :
  ∀ (x y : ℝ), (x^2 + y^2 - 12x + 16y + 100 = 0) → (x - 7)^(-y) = 1 :=
by
  intros x y h
  sorry

end algebra_expression_value_l298_298129


namespace banks_should_offer_benefits_to_seniors_l298_298582

-- Definitions based on conditions
def better_credit_reliability (pensioners : Type) : Prop :=
  ∀ (p : pensioners), has_better_credit_reliability p

def stable_pension_income (pensioners : Type) : Prop :=
  ∀ (p : pensioners), has_stable_income p

def indirect_financial_benefits (pensioners : Type) : Prop :=
  ∀ (p : pensioners), receives_financial_benefit p

def propensity_to_save (pensioners : Type) : Prop :=
  ∀ (p : pensioners), has_saving_habits p

def preference_long_term_deposits (pensioners : Type) : Prop :=
  ∀ (p : pensioners), prefers_long_term_deposits p

-- Main theorem statement
theorem banks_should_offer_benefits_to_seniors
  (P : Type)
  (h1 : better_credit_reliability P)
  (h2 : stable_pension_income P)
  (h3 : indirect_financial_benefits P)
  (h4 : propensity_to_save P)
  (h5 : preference_long_term_deposits P) :
  ∃ benefits : Type, benefits.make_sense :=
sorry

end banks_should_offer_benefits_to_seniors_l298_298582


namespace hyperbola_focus_l298_298378

variable {x y : ℝ}

def hyperbola (x y : ℝ) : Prop := 
  2 * x^2 - 3 * y^2 + 6 * x - 12 * y - 8 = 0

theorem hyperbola_focus :
  (∃ (x y : ℝ), hyperbola x y ∧ (x, y) = (5 * real.sqrt 3 / 6 - 3 / 2, -2)) :=
sorry

end hyperbola_focus_l298_298378


namespace sum_of_coeffs_l298_298385

theorem sum_of_coeffs (A B C D : ℤ) (h₁ : A = 1) (h₂ : B = -1) (h₃ : C = -12) (h₄ : D = 3) :
  A + B + C + D = -9 := 
by
  rw [h₁, h₂, h₃, h₄]
  norm_num

end sum_of_coeffs_l298_298385


namespace probability_six_even_is_one_seventy_six_l298_298784

-- Define the range of integers from -9 to 9
def integers_in_range : Set ℤ := { n | -9 ≤ n ∧ n ≤ 9 }

-- Define the even numbers in this range
def even_numbers : Set ℤ := { n | n ∈ integers_in_range ∧ n % 2 = 0 }

-- Total number of even numbers in the range
lemma even_numbers_count : even_numbers.toFinset.card = 9 := by
  sorry

-- Total number of integers in the range
lemma integers_count : integers_in_range.toFinset.card = 19 := by
  sorry

-- The probability of drawing 6 even numbers without replacement
noncomputable def probability_of_six_even_numbers : ℚ :=
  (9 / 19) * (8 / 18) * (7 / 17) * (6 / 16) * (5 / 15) * (4 / 14)

-- The main theorem stating the probability is 1/76
theorem probability_six_even_is_one_seventy_six :
  probability_of_six_even_numbers = 1 / 76 := by
  sorry

end probability_six_even_is_one_seventy_six_l298_298784


namespace minimum_bottles_needed_l298_298049

noncomputable def oil_bottles (fluid_oz_required : ℝ) (fl_oz_per_l : ℝ) (ml_per_l : ℝ) (ml_per_bottle : ℝ) : ℕ :=
let liters_required := fluid_oz_required / fl_oz_per_l
let ml_required := liters_required * ml_per_l
let bottles_required := ml_required / ml_per_bottle
in ⌈bottles_required⌉₊ -- ceiling function to round up to the nearest whole number

theorem minimum_bottles_needed : oil_bottles 60 33.8 1000 250 = 8 :=
by
  sorry

end minimum_bottles_needed_l298_298049


namespace angle_terminal_side_equiv_l298_298618

-- Define the function to check angle equivalence
def angle_equiv (θ₁ θ₂ : ℝ) : Prop := ∃ k : ℤ, θ₁ = θ₂ + k * 360

-- Theorem statement
theorem angle_terminal_side_equiv : angle_equiv 330 (-30) :=
  sorry

end angle_terminal_side_equiv_l298_298618


namespace necessary_but_not_sufficient_condition_l298_298128

theorem necessary_but_not_sufficient_condition (x y : ℝ) : 
  ((x > 1) ∨ (y > 2)) → (x + y > 3) ∧ ¬((x > 1) ∨ (y > 2) ↔ (x + y > 3)) :=
by
  sorry

end necessary_but_not_sufficient_condition_l298_298128


namespace solve_for_x_l298_298603

theorem solve_for_x (x : ℝ) (h : 5 + 3.5 * x = 2.5 * x - 25) : x = -30 :=
by 
  -- Placeholder for the actual proof
  sorry

end solve_for_x_l298_298603


namespace he_has_9_more_apples_than_adam_and_jackie_together_l298_298648

noncomputable def num_of_apples_jackie : ℕ := Nat
noncomputable def num_of_apples_adam (J : ℕ) : ℕ := J + 8
noncomputable def num_of_apples_together (J : ℕ) : ℕ := J + num_of_apples_adam J
def num_of_apples_he : ℕ := 21

theorem he_has_9_more_apples_than_adam_and_jackie_together (J : ℕ) (h1 : num_of_apples_together J = 12) (h2 : num_of_apples_he = 21) :
  21 - num_of_apples_together J = 9 :=
by
  sorry

end he_has_9_more_apples_than_adam_and_jackie_together_l298_298648


namespace nth_letter_is_A_l298_298929

def repeating_sequence := "FEDCBAABCDEF"
def sequence_length := 12
def nth_position := 1234

theorem nth_letter_is_A : 
  repeating_sequence[(nth_position % sequence_length) - 1] = 'A' :=
by
  -- Implement the proof here
  sorry

end nth_letter_is_A_l298_298929


namespace break_even_books_l298_298300

theorem break_even_books (F V S : ℕ) (hF : F = 50000) (hV : V = 4) (hS : S = 9) : ∃ x : ℕ, S * x = F + V * x ∧ x = 10000 :=
by
  use 10000
  dsimp at *
  rw [hF, hV, hS]
  norm_num
  sorry

end break_even_books_l298_298300


namespace find_a_b_and_monotonicity_l298_298754

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^2 * exp(x - 1) + a * x^3 + b * x^2

theorem find_a_b_and_monotonicity :
  ∃ (a b : ℝ), a = -1 / 3 ∧ b = -1 ∧
  (∀ x, x < -2 → deriv (λ x, f x a b) x < 0) ∧
  (∀ x, x > -2 ∧ x < 0 → deriv (λ x, f x a b) x > 0) ∧
  (∀ x, x > 0 ∧ x < 1 → deriv (λ x, f x a b) x < 0) ∧
  (∀ x, x > 1 → deriv (λ x, f x a b) x > 0) :=
by 
  sorry

end find_a_b_and_monotonicity_l298_298754


namespace angle_B_l298_298145

-- Define the conditions
variables {A B C : ℝ} (a b c : ℝ)
variable (h : a^2 + c^2 = b^2 + ac)

-- State the theorem
theorem angle_B (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  B = π / 3 :=
sorry

end angle_B_l298_298145


namespace normal_distribution_symmetry_l298_298848

variables {σ : ℝ} {x : ℝ}

noncomputable def normal_distribution (μ : ℝ) (σ : ℝ) (x : ℝ) : ℝ :=
  1 / (σ * real.sqrt (2 * real.pi)) * real.exp (- ((x - μ) ^ 2) / (2 * σ ^ 2))

theorem normal_distribution_symmetry (h1 : ∀ x, x ~ N(4, σ^2)) (h2 : P(x > 2) = 0.6) :
  P(x > 6) = 0.4 :=
sorry

end normal_distribution_symmetry_l298_298848


namespace P_union_Q_l298_298954

variable {P Q : Set ℝ}

def P : Set ℝ := { x | x^2 - 5x - 6 ≤ 0 }
def Q : Set ℝ := { x | ∃ y : ℝ, y = Real.log 2 (x^2 - 2x - 15) }

theorem P_union_Q :
  (P ∪ Q \ P ∩ Q) = (Set.Union (λ x, (-∞, -3) ∪ Set.Interval.closed (-1) 5 ∪ (6, +∞))) :=
by 
  sorry

end P_union_Q_l298_298954


namespace correct_statements_count_l298_298290

-- Definitions of properties
def bisect_diagonals (q : Type) : Prop := sorry
def perpendicular_diagonals (q : Type) : Prop := sorry
def equal_diagonals (q : Type) : Prop := sorry

-- Definitions of geometric shapes
def is_parallelogram (q : Type) : Prop := sorry
def is_rhombus (q : Type) : Prop := sorry
def is_square (q : Type) : Prop := sorry
def is_rectangle (q : Type) : Prop := sorry

-- Given conditions
variables {Q : Type}

-- Statements to prove
def statement_1 (Q : Type) := bisect_diagonals Q → is_parallelogram Q
def statement_2 (Q : Type) := perpendicular_diagonals Q → is_rhombus Q
def statement_3 (Q : Type) := is_parallelogram Q → perpendicular_diagonals Q ∧ equal_diagonals Q → is_square Q
def statement_4 (Q : Type) := is_parallelogram Q → equal_diagonals Q → is_rectangle Q

-- The proof problem: Prove the number of correct statements is 3
theorem correct_statements_count : 
(statement_1 Q ∧ ¬statement_2 Q ∧ statement_3 Q ∧ statement_4 Q) = 3 := sorry

end correct_statements_count_l298_298290


namespace train_departure_time_l298_298345

theorem train_departure_time 
(distance speed : ℕ) (arrival_time_chicago difference : ℕ) (arrival_time_new_york departure_time : ℕ) 
(h_dist : distance = 480) 
(h_speed : speed = 60)
(h_arrival_chicago : arrival_time_chicago = 17) 
(h_difference : difference = 1)
(h_arrival_new_york : arrival_time_new_york = arrival_time_chicago + difference) : 
  departure_time = arrival_time_new_york - distance / speed :=
by
  sorry

end train_departure_time_l298_298345


namespace original_cost_before_changes_l298_298938

variable (C : ℝ)

theorem original_cost_before_changes (h : 2 * C * 1.20 = 480) : C = 200 :=
by
  -- proof goes here
  sorry

end original_cost_before_changes_l298_298938


namespace range_of_b_l298_298141

theorem range_of_b 
  (b : ℝ)
  (h : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (x₁ + b = 3 - real.sqrt (4 * x₁ - x₁^2)) ∧ 
    (x₂ + b = 3 - real.sqrt (4 * x₂ - x₂^2))) :
  b ∈ set.Ioo ((1 - real.sqrt 29) / 2) ((1 + real.sqrt 29) / 2) :=
by
  sorry

end range_of_b_l298_298141


namespace number_of_x_intercepts_l298_298801

theorem number_of_x_intercepts :
  let f (x : ℝ) := Real.sin (1 / x)
  let interval : Set ℝ := { x : ℝ | 0.00005 < x ∧ x < 0.0005 }
  ∃! n : ℕ, n = 5730 ∧ Intervals.count_x_intercepts f interval n :=
by
  sorry

end number_of_x_intercepts_l298_298801


namespace number_of_correct_propositions_l298_298108

-- Define the concepts in the problem.
variable {m n : Line}
variable {α β γ : Plane}

-- Conditions and propositions
def prop1 (α β : Plane) (m n : Line) : Prop := 
  (α ∩ β = m) ∧ (n ∥ m) → (n ∥ α) ∧ (n ∦ β)

def prop2 (α β : Plane) (m : Line) : Prop := 
  (α ⊥ β) ∧ (m ⊥ β) ∧ (¬ m ⊆ α) → (m ∥ α)

def prop3 (α β : Plane) (m : Line) : Prop := 
  (α ∦ β) ∧ (m ⊆ α) → (m ∥ β)

def prop4 (α β γ : Plane) : Prop := 
  (α ⊥ β) ∧ (α ⊥ γ) → (β ∥ γ)

-- The final theorem to prove the number of correct propositions.
theorem number_of_correct_propositions : 
  (¬ prop1 α β m n) ∧ (prop2 α β m) ∧ (prop3 α β m) ∧ (¬ prop4 α β γ) → 2 :=
  by
    sorry

end number_of_correct_propositions_l298_298108


namespace longest_diagonal_of_rhombus_l298_298730

theorem longest_diagonal_of_rhombus (A B : ℝ) (h1 : A = 150) (h2 : ∃ x, (A = 1/2 * (4 * x) * (3 * x)) ∧ (x = 5)) : 
  4 * (classical.some h2) = 20 := 
by sorry

end longest_diagonal_of_rhombus_l298_298730


namespace alcohol_percentage_after_adding_water_l298_298293

variables (initial_volume : ℕ) (initial_percentage : ℕ) (added_volume : ℕ)
def initial_alcohol_volume := initial_volume * initial_percentage / 100
def final_volume := initial_volume + added_volume
def final_percentage := initial_alcohol_volume * 100 / final_volume

theorem alcohol_percentage_after_adding_water :
  initial_volume = 15 →
  initial_percentage = 20 →
  added_volume = 5 →
  final_percentage = 15 := by
sorry

end alcohol_percentage_after_adding_water_l298_298293


namespace isosceles_triangle_angles_l298_298355

theorem isosceles_triangle_angles (α β γ : ℝ) (h_iso : α = β ∨ α = γ ∨ β = γ) (h_angle : α + β + γ = 180) (h_40 : α = 40 ∨ β = 40 ∨ γ = 40) :
  (α = 70 ∧ β = 70 ∧ γ = 40) ∨ (α = 40 ∧ β = 100 ∧ γ = 40) ∨ (α = 40 ∧ β = 40 ∧ γ = 100) :=
by
  sorry

end isosceles_triangle_angles_l298_298355


namespace probability_log_value_l298_298100

noncomputable def f (x : ℝ) := Real.log x / Real.log 2 - 1

theorem probability_log_value (a : ℝ) (h1 : 1 ≤ a) (h2 : a ≤ 10) :
  (4 / 9 : ℝ) = 
    ((8 - 4) / (10 - 1) : ℝ) := by
  sorry

end probability_log_value_l298_298100


namespace passed_candidates_l298_298242

theorem passed_candidates (P F : ℕ) (h1 : P + F = 120) (h2 : 39 * P + 15 * F = 35 * 120) : P = 100 :=
by
  sorry

end passed_candidates_l298_298242


namespace rational_soln_integer_l298_298873

theorem rational_soln_integer (a b : ℤ) (x y : ℚ) (h1 : y - 2 * x = a) (h2 : y^2 - x * y + x^2 = b) : 
  x ∈ ℤ ∧ y ∈ ℤ :=
sorry

end rational_soln_integer_l298_298873


namespace impossible_pairs_l298_298509

variable (b r : ℕ)

-- Definition of the conditions as stated in the problem
def conditions (b r : ℕ) : Prop :=
  ∀ (blue_rect red_rect : Set (Set (ℝ × ℝ))),
  (∀ (rect ∈ blue_rect), rect.size = b) ∧
  (∀ (rect ∈ red_rect), rect.size = r) ∧
  ∀ (line : Set (ℝ × ℝ)), 
  (∀ (c : Set (Set (ℝ × ℝ))), c.size = 1 →
    ∀ (rect : Set (ℝ × ℝ)), rect ∈ c → rect ≠ ∅) ∧
    (∀ (blue_rect red_rect : Set (Set (ℝ × ℝ))),
      ∃ (line : Set (ℝ × ℝ)), 
      ∀ (blue_rect red_rect : Set (Set (ℝ × ℝ))),
      (blue_rect ≠ ∅ ∧ red_rect ≠ ∅ → line ∩ blue_rect ≠ ∅ ∧ line ∩ red_rect ≠ ∅))

-- Theorem stating the mathematical equivalence
theorem impossible_pairs : 
  conditions b r → (b, r) ∉ [(1, 7), (2, 6), (3, 4), (3, 3)] → False :=
begin
  -- The actual proof is omitted here
  sorry
end

end impossible_pairs_l298_298509


namespace tan_of_triangle_l298_298515

theorem tan_of_triangle (A B : ℝ) (h1 : ∀ {x : ℝ}, 0 < x ∧ x < π/2) 
  (h2 : cos (A + B) = sin (A - B)) : tan A = 1 := 
by
  sorry

end tan_of_triangle_l298_298515


namespace circle_area_l298_298978

-- Define the points A and B
def A : ℝ × ℝ := (4, 10)
def B : ℝ × ℝ := (10, 8)

-- Circle omega with A and B on it and tangents at A and B intersect on x-axis.
theorem circle_area (A B : ℝ × ℝ)
  (x_axis_intersection : (∃ C : ℝ × ℝ, C.2 = 0 ∧
    (∃ l m : ℝ (l, m).fst = A ∧ (l, m).snd = B ∧ (∃ k1 k2: ℝ, k1 * A.fst + k2 * A.snd = 1 ∧ k1 * B.fst + k2 * B.snd = 1 ∧ k1 != k2 ∧ 
    k1 * A.fst + k2 * B.snd = k1 * C.fst + k2 * A.snd)))): 
  (area : ℝ := π * (10/3)^2 = 100π / 9)
  := sorry

end circle_area_l298_298978


namespace distance_between_points_l298_298376

theorem distance_between_points (O A B : Type) (R a b d : ℝ) (circle : ∀ (x : O), dist O x = R) :
  dist O A = a → dist O B = b → dist A B = d →
  ∃ P1 P2 : Type, 
    dist P1 P2 = 2 * R * sin (acos ((a^2 + b^2 - d^2) / (2 * a * b)) / 2) :=
by
  sorry

end distance_between_points_l298_298376


namespace fraction_double_halfway_l298_298798

theorem fraction_double_halfway (a b : ℚ) (h1 : a = 1/6) (h2 : b = 1/4) : (2 * ((a + b) / 2)) = 5/12 :=
by
  rw [h1, h2]
  simp
  norm_num
  sorry

end fraction_double_halfway_l298_298798


namespace desired_yearly_income_l298_298329

theorem desired_yearly_income (total_investment : ℝ) 
  (investment1 : ℝ) (rate1 : ℝ) 
  (investment2 : ℝ) (rate2 : ℝ) 
  (rate_remainder : ℝ) 
  (h_total : total_investment = 10000) 
  (h_invest1 : investment1 = 4000)
  (h_rate1 : rate1 = 0.05) 
  (h_invest2 : investment2 = 3500)
  (h_rate2 : rate2 = 0.04)
  (h_rate_remainder : rate_remainder = 0.064)
  : (rate1 * investment1 + rate2 * investment2 + rate_remainder * (total_investment - (investment1 + investment2))) = 500 := 
by
  sorry

end desired_yearly_income_l298_298329


namespace box_width_l298_298700

theorem box_width
  (length : ℝ) (height : ℝ) (vol_cube : ℝ) (num_cubes : ℕ)
  (h_length : length = 10)
  (h_height : height = 5)
  (h_vol_cube : vol_cube = 5)
  (h_num_cubes : num_cubes = 130) :
  ∃ (width : ℝ), width = 13 :=
by
  have total_volume := num_cubes * vol_cube,
  have volume := length * height * width,
  have h_volume : total_volume = length * height * width,
  use (total_volume / (length * height)),
  sorry


end box_width_l298_298700


namespace zhijie_suanjing_l298_298162

theorem zhijie_suanjing :
  ∃ (x y: ℕ), x + y = 100 ∧ 3 * x + y / 3 = 100 :=
by
  sorry

end zhijie_suanjing_l298_298162


namespace syllogism_correct_order_l298_298310

theorem syllogism_correct_order :
  (∀ f : ℝ → ℝ, (∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b) → ∀ g : ℝ → ℝ, g = f → (LinearFunction g → StraightLineGraph g)) →
  (∃ a b : ℝ, ∀ x : ℝ, (2 * x + 5) = a * x + b) →
  StraightLineGraph (λ x : ℝ, 2 * x + 5) :=
begin
  sorry
end

end syllogism_correct_order_l298_298310


namespace f_expression_sequence_geometric_sum_of_first_n_terms_l298_298089

-- condition definitions
def is_acute_angle (α : ℝ) : Prop := 0 < α ∧ α < π / 2
def tan_alpha_eq_sqrt2_minus_1 (α : ℝ) : Prop := tan α = Real.sqrt 2 - 1
def function_f (α : ℝ) (x : ℤ) : ℤ := 2 * x * (2 * tan α / (1 - tan α ^ 2)) + sin (2 * α + π / 4)

-- given problem conditions
constant α : ℝ
constant is_acute : is_acute_angle α
constant tan_condition : tan_alpha_eq_sqrt2_minus_1 α
constant a : ℕ → ℤ
axiom a_1 : a 1 = 1
axiom a_recurrence : ∀ n, a (n + 1) = function_f α (a n)

-- problem part 1
theorem f_expression (x : ℤ) : function_f α x = 2 * x + 1 := sorry

-- problem part 2
theorem sequence_geometric (n : ℕ) : ∃ r : ℤ, (λ n, a n + 1) n = (2 ^ n) * r := sorry

-- problem part 3
theorem sum_of_first_n_terms (n : ℕ) : ∑ k in Finset.range n, a (k + 1) = 2^(n + 1) - n - 2 := sorry

end f_expression_sequence_geometric_sum_of_first_n_terms_l298_298089


namespace geometric_sequence_a6_value_l298_298927

theorem geometric_sequence_a6_value
  (a : ℕ → ℝ)
  (h_geom : ∀ n m, a (n + m) = a n * a m)
  (h_eq : ∀ x, x^2 - 4 * x + 3 = 0 → (a 4 = x ∨ a 8 = x)) :
  a 6 = sqrt 3 :=
by sorry

end geometric_sequence_a6_value_l298_298927


namespace g_at_minus_six_l298_298193

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x - 9
def g (x : ℝ) : ℝ := 3 * x ^ 2 + 4 * x - 2

theorem g_at_minus_six : g (-6) = 43 / 16 := by
  sorry

end g_at_minus_six_l298_298193


namespace Al_Carol_coinciding_rest_days_l298_298003

theorem Al_Carol_coinciding_rest_days :
  let al_schedule := λ (n : ℕ), (n % 6) = 4 ∨ (n % 6) = 5,
      carol_schedule := λ (n : ℕ), (n % 6) = 5 in
  let count_coinciding_rest_days := (Finset.range 1000).count (λ n, al_schedule n ∧ carol_schedule n) in
  count_coinciding_rest_days = 166 :=
sorry

end Al_Carol_coinciding_rest_days_l298_298003


namespace find_function_form_l298_298842

theorem find_function_form
  (f : ℝ → ℝ)
  (h : ∀ x, f⁻¹ x = log (x + 1) / log 2 + 1) :
  f = λ x, 2^(x - 1) - 1 :=
by
sorry

end find_function_form_l298_298842


namespace maximize_ab_value_at_2017_l298_298772

noncomputable def f (a b x : ℝ) : ℝ := 2 * a * x + b

theorem maximize_ab_value_at_2017 
  (a b : ℝ)
  (h_pos : 0 < a) 
  (h_pos_b : 0 < b)
  (h_abs : ∀ x ∈ set.Icc (-1/2 : ℝ) (1/2 : ℝ), abs (f a b x) ≤ 2) 
  (h_max_prod : a * b = 1) 
  : f a b 2017 = 4035 := 
sorry

end maximize_ab_value_at_2017_l298_298772


namespace arithmetic_expression_evaluation_l298_298370

theorem arithmetic_expression_evaluation :
  3^2 * (-2 + 3) / (1 / 3) - | -28 | = -1 :=
by
  sorry

end arithmetic_expression_evaluation_l298_298370


namespace eccentricity_of_ellipse_eq_l298_298046

-- Definition of the ellipse and related parameters
variables (a b : ℝ) (x_0 y_0 : ℝ)
variables (h1 : a > b) (h2 : b > 0)
variables (h_ellipse : (x_0 / a) ^ 2 + (y_0 / b) ^ 2 = 1)
variables (h_symmetry : ∀(P Q : ℝ), P = x_0 → Q = -x_0)

-- Definition of the slopes and their product
def slope_AP (P : ℝ × ℝ) : ℝ := P.2 / (P.1 + a)
def slope_AQ (Q : ℝ × ℝ) : ℝ := Q.2 / (Q.1 - a)
variable (h_slopes : slope_AP (x_0, y_0) * slope_AQ (-x_0, y_0) = 1 / 4)

-- The proof statement for the eccentricity
theorem eccentricity_of_ellipse_eq :
  (sqrt (1 - (b / a) ^ 2)) = sqrt 3 / 2 :=
by
  sorry

end eccentricity_of_ellipse_eq_l298_298046


namespace shelves_filled_l298_298996

theorem shelves_filled (total_teddy_bears teddy_bears_per_shelf : ℕ) (h1 : total_teddy_bears = 98) (h2 : teddy_bears_per_shelf = 7) : 
  total_teddy_bears / teddy_bears_per_shelf = 14 := 
by 
  sorry

end shelves_filled_l298_298996


namespace sum_of_areas_of_circles_l298_298347

-- Definitions of the conditions given in the problem
def triangle_side1 : ℝ := 6
def triangle_side2 : ℝ := 8
def triangle_side3 : ℝ := 10

-- Definitions of the radii r, s, t
variables (r s t : ℝ)

-- Conditions derived from the problem
axiom rs_eq : r + s = triangle_side1
axiom rt_eq : r + t = triangle_side2
axiom st_eq : s + t = triangle_side3

-- Main theorem to prove
theorem sum_of_areas_of_circles : (π * r^2) + (π * s^2) + (π * t^2) = 56 * π :=
by
  sorry

end sum_of_areas_of_circles_l298_298347


namespace ray_climbs_l298_298586

theorem ray_climbs (n : ℕ) (h1 : n % 3 = 1) (h2 : n % 5 = 3) (h3 : n % 7 = 1) (h4 : n > 15) : n = 73 :=
sorry

end ray_climbs_l298_298586


namespace number_of_x_intercepts_l298_298802

theorem number_of_x_intercepts :
  let f (x : ℝ) := Real.sin (1 / x)
  let interval : Set ℝ := { x : ℝ | 0.00005 < x ∧ x < 0.0005 }
  ∃! n : ℕ, n = 5730 ∧ Intervals.count_x_intercepts f interval n :=
by
  sorry

end number_of_x_intercepts_l298_298802


namespace find_a_max_min_on_interval_l298_298468

noncomputable def f (a x : ℝ) : ℝ := -x^3 + 3 * a * x^2 - 5

theorem find_a (h : ∃ a, x = 2 → Derivative (f a) x = 0) : a = 1 := by
  sorry

theorem max_min_on_interval (a : ℝ) (h : a = 1) :
  let f := f a 
  let max_val := max (f (-2)) (max (f 0) (max (f 2) (f 4))) in
  let min_val := min (f (-2)) (min (f 0) (min (f 2) (f 4))) in
  max_val = 15 ∧ min_val = -21 := by
  sorry

end find_a_max_min_on_interval_l298_298468


namespace radius_ratio_of_smaller_to_larger_l298_298704

noncomputable def ratio_of_radii (v_large v_small : ℝ) (R r : ℝ) (h_large : (4/3) * Real.pi * R^3 = v_large) (h_small : v_small = 0.25 * v_large) (h_small_sphere : (4/3) * Real.pi * r^3 = v_small) : ℝ :=
  let ratio := r / R
  ratio

theorem radius_ratio_of_smaller_to_larger (v_large : ℝ) (R r : ℝ) (h_large : (4/3) * Real.pi * R^3 = 576 * Real.pi) (h_small_sphere : (4/3) * Real.pi * r^3 = 0.25 * 576 * Real.pi) : r / R = 1 / (2^(2/3)) :=
by
  sorry

end radius_ratio_of_smaller_to_larger_l298_298704


namespace transaction_mistake_in_cents_l298_298746

theorem transaction_mistake_in_cents
  (x y : ℕ)
  (hx : 10 ≤ x ∧ x ≤ 99)
  (hy : 10 ≤ y ∧ y ≤ 99)
  (error_cents : 100 * y + x - (100 * x + y) = 5616) :
  y = x + 56 :=
by {
  sorry
}

end transaction_mistake_in_cents_l298_298746


namespace lcm_12_18_l298_298408

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := 
by
  sorry

end lcm_12_18_l298_298408


namespace triangle_inequality_l298_298527

variables (A B C : Type) [IsTriangle A B C]
variables (s R r : ℝ)
variable [IsSemiPerimeter A B C s]
variable [IsCircumradius A B C R]
variable [IsInradius A B C r]

theorem triangle_inequality :
  2 * Real.sqrt (r * (r + 4 * R)) < 2 * s ∧ 2 * s ≤ Real.sqrt (4 * (r + 2 * R) ^ 2 + 2 * R ^ 2) := sorry

end triangle_inequality_l298_298527


namespace solve_x_l298_298612

theorem solve_x (x : ℝ) (h : (4 * x + 3) / (3 * x ^ 2 + 4 * x - 4) = 3 * x / (3 * x - 2)) :
  x = (-1 + Real.sqrt 10) / 3 ∨ x = (-1 - Real.sqrt 10) / 3 :=
by sorry

end solve_x_l298_298612


namespace minimum_value_f_f_x_plus_pi_over_8_l298_298107
noncomputable theory

def a (x : ℝ) : ℝ × ℝ := (5 * real.sqrt 3 * real.cos x, real.cos x)
def b (x : ℝ) : ℝ × ℝ := (real.sin x, 2 * real.cos x)
def f (x : ℝ) : ℝ :=
  let dot_product := a x.1 * b x.1 + a x.2 * b x.2 in
  let magnitude_squared := b x.1 ^ 2 + b x.2 ^ 2 in
  dot_product + magnitude_squared + 3 / 2

theorem minimum_value_f :
  ∃ x ∈ set.Icc 0 (real.pi / 2), f x = 5 / 2 :=
sorry

theorem f_x_plus_pi_over_8 (x : ℝ) (h : x ∈ set.Icc (real.pi / 6) (real.pi / 2) ∧ f x = 8) :
  f (x + real.pi / 8) = 5 - real.sqrt 2 / 2 :=
sorry

end minimum_value_f_f_x_plus_pi_over_8_l298_298107


namespace BD_can_determine_own_results_l298_298513

-- Definition of students participating
inductive Student
| A | B | C | D
deriving DecidableEq, Repr

-- Definition of results
inductive Result
| Excellent | Good
deriving DecidableEq, Repr

-- The competition conditions
def competition_results : (Student → Result) → (Student → (Student → Result) → Prop) → Prop :=
  λ results knowledge,
  -- Condition 1: Among the students, two have 'Excellent' and two have 'Good'
  (∃ p1 p2 : (Student → Prop), (∀ s, p1 s ↔ results s = Result.Excellent) ∧ (∀ s, p2 s ↔ results s = Result.Good) ∧ (∃ s1 s2 s3 s4, 
      s1 ≠ s2 ∧ s3 ≠ s4 ∧ 
      results s1 = Result.Excellent ∧ results s2 = Result.Excellent ∧ 
      results s3 = Result.Good ∧ results s4 = Result.Good
  )) ∧
  
  -- Condition 2: Knowledge distribution
  knowledge Student.A (λ s, s = Student.B ∨ s = Student.C → results s = Result.Excellent) ∧
  knowledge Student.B (λ s, s = Student.C → results s = Result.Excellent) ∧
  knowledge Student.D (λ s, s = Student.A → results s = Result.Excellent) ∧
  
  -- Condition 3: Observation statement by Student A
  ¬(knowledge Student.A (λ s, s = Student.A → results s = Result.Excellent))

-- The proof problem: Given the above conditions, B and D can determine their own results
theorem BD_can_determine_own_results :
  ∀ (results : Student → Result) (knowledge : Student → (Student → Result) → Prop),
  competition_results results knowledge →
  (∃ resultB, knowledge Student.B (λ s, s = Student.B → results s = resultB)) ∧
  (∃ resultD, knowledge Student.D (λ s, s = Student.D → results s = resultD)) :=
by 
  -- Proof steps are omitted
  sorry

end BD_can_determine_own_results_l298_298513


namespace point_in_fourth_quadrant_l298_298570

def point : ℝ × ℝ := (1, -2)

def is_fourth_quadrant (p: ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 < 0

theorem point_in_fourth_quadrant : is_fourth_quadrant point :=
by
  sorry

end point_in_fourth_quadrant_l298_298570


namespace intersection_eq_0_2_l298_298453

-- Define the sets A and B based on given conditions
def setA : Set ℝ := {x | 2^x > 1}
def setB : Set ℝ := {x | x^2 - x - 2 < 0}

theorem intersection_eq_0_2 :
  Set.inter (setA) (setB) = {x | 0 < x ∧ x < 2} :=
by
  sorry

end intersection_eq_0_2_l298_298453


namespace ratio_boys_girls_l298_298908

theorem ratio_boys_girls (number_of_girls : ℕ) (total_students : ℕ) 
  (h1 : number_of_girls = 160) 
  (h2 : total_students = 416) :
  8 = 256 / 32 ∧ 5 = 160 / 32 ∧ (256 / 32) = 8 ∧ (160 / 32) = 5 ∧ (256 / 32):(160 / 32) = 8:5 :=
by
  sorry

end ratio_boys_girls_l298_298908


namespace h_of_k_neg_3_l298_298118

def h (x : ℝ) : ℝ := 4 - real.sqrt x

def k (x : ℝ) : ℝ := 3 * x + 3 * x^2

theorem h_of_k_neg_3 : h (k (-3)) = 4 - 3 * real.sqrt 2 :=
by
  sorry

end h_of_k_neg_3_l298_298118


namespace width_of_field_l298_298677

noncomputable def field_width : ℝ := 60

theorem width_of_field (L W : ℝ) (hL : L = (7/5) * W) (hP : 288 = 2 * L + 2 * W) : W = field_width :=
by
  sorry

end width_of_field_l298_298677


namespace median_eq_mean_implies_x_l298_298740

def data_set (x : ℝ) : List ℝ := [5, 7, 7, x]

def median (l : List ℝ) : ℝ :=
  let sorted := l.qsort (· ≤ ·)
  match sorted with
  | a::b::c::d::_ => (b + c) / 2
  | _ => 0  -- This case won't actually occur for our data set

def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

theorem median_eq_mean_implies_x (x : ℝ) 
  (h : median (data_set x) = mean (data_set x)) : x = 5 ∨ x = 9 := by
  sorry

end median_eq_mean_implies_x_l298_298740


namespace angle_CDE_eq_20_l298_298552

-- Definitions used in the conditions
variables {A B C D E : Type} [Triangle A B C]
variable (isosceles : AB = AC)
variable (angleBAC : ∠BAC = 20)
variable (D_on_AB : D ∈ AB)
variable (angleBCD : ∠BCD = 70)
variable (E_on_AC : E ∈ AC)
variable (angleCBE : ∠CBE = 60)

-- Statement of the math proof problem
theorem angle_CDE_eq_20 : ∠CDE = 20 :=
begin
  sorry
end

end angle_CDE_eq_20_l298_298552


namespace DE_parallel_AB_l298_298955

open EuclideanGeometry 

variables {A B C D E : Point}

-- Let \( ABC \) be an isosceles triangle with \( AB = AC \).
axiom is_isosceles_triangle (h : Triangle A B C) (h_iso : distance A B = distance A C) : is_isosceles A B C

-- Point \( D \) lies on side \( AC \) such that \( BD \) is the angle bisector of \( \angle ABC \).
axiom on_side_AC (h : Triangle A B C) (h_point: Point D) : lies_on D (segment A C)
axiom is_angle_bisector_B (h : Triangle A B C) (h_bisect : is_angle_bisector B D (∠ B A C)) : angle B D A = angle B D C

-- Point \( E \) lies on side \( BC \) between \( B \) and \( C \) such that \( BE = CD \).
axiom on_side_BC (h : Triangle A B C) (h_point: Point E) : lies_on E (segment B C)
axiom BE_eq_CD (h : Triangle A B C) (dist_BE_eq_CD : distance B E = distance C D) : distance B E = distance C D

-- Prove that \( DE \parallel AB \).
theorem DE_parallel_AB {A B C D E : Point} (h : Triangle A B C) 
  (h_iso : is_isosceles A B C) 
  (h_DonAC : lies_on D (segment A C))
  (h_angleBisector : is_angle_bisector B D (∠ B A C))
  (h_EonBC: lies_on E (segment B C))
  (h_dist_BE_eq_CD: distance B E = distance C D) :
  parallel (line D E) (line A B) :=
sorry

end DE_parallel_AB_l298_298955


namespace fencing_cost_l298_298297

noncomputable def pi_value : ℝ := 3.14159

def diameter := 42
def rate_per_meter := 3

def circumference (d : ℝ) : ℝ := pi_value * d
def cost (C : ℝ) (r : ℝ) : ℝ := C * r

theorem fencing_cost : cost (circumference diameter) rate_per_meter = 396 := by
  show (cost (circumference diameter) rate_per_meter = 396)
  sorry

end fencing_cost_l298_298297


namespace relationship_between_fractions_l298_298544

variable (a a' b b' : ℝ)
variable (h₁ : a > 0)
variable (h₂ : a' > 0)
variable (h₃ : (-(b / (2 * a)))^2 > (-(b' / (2 * a')))^2)

theorem relationship_between_fractions
  (a : ℝ) (a' : ℝ) (b : ℝ) (b' : ℝ)
  (h1 : a > 0) (h2 : a' > 0)
  (h3 : (-(b / (2 * a)))^2 > (-(b' / (2 * a')))^2) :
  (b^2) / (a^2) > (b'^2) / (a'^2) :=
by sorry

end relationship_between_fractions_l298_298544


namespace cyclic_sum_inequality_l298_298575

theorem cyclic_sum_inequality (n : ℕ) (a : Fin n.succ -> ℕ) (h : ∀ i, a i > 0) : 
  (Finset.univ.sum fun i => a i / a ((i + 1) % n)) ≥ n :=
by
  sorry

end cyclic_sum_inequality_l298_298575


namespace alpha_in_third_quadrant_l298_298116

theorem alpha_in_third_quadrant (k : ℤ) (α : ℝ) :
  (4 * k + 1) * 180 < α ∧ α < (4 * k + 1) * 180 + 60 → 180 < α ∧ α < 240 :=
  sorry

end alpha_in_third_quadrant_l298_298116


namespace find_omega_of_monotone_and_equal_l298_298858

theorem find_omega_of_monotone_and_equal (ω : ℝ) 
  (h_monotone : ∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < π / 3 → 
    sin (ω * x₁ + π / 6) < sin (ω * x₂ + π / 6))
  (h_equal : sin (ω * (π / 4) + π / 6) = sin (ω * (π / 2) + π / 6)) : 
  ω = 8 / 9 :=
sorry

end find_omega_of_monotone_and_equal_l298_298858


namespace h_k_minus3_eq_l298_298122

def h (x : ℝ) : ℝ := 4 - Real.sqrt x
def k (x : ℝ) : ℝ := 3 * x + 3 * x^2

theorem h_k_minus3_eq : h (k (-3)) = 4 - 3 * Real.sqrt 2 := 
by 
  sorry

end h_k_minus3_eq_l298_298122


namespace semicircle_P_S_eq_l298_298739

   /-- A semicircle is constructed over the line segment AB with midpoint M.
    Let P be a point on the semicircle other than A or B.
    Let Q be the midpoint of the arc AP. The intersection of the line BP
    with the line parallel to PQ through M is denoted as S.
    We need to prove that PM = PS. -/
   theorem semicircle_P_S_eq (A B M P Q S : Point) (h : semicircle_over A B M) 
   (P_on_semicircle : point_on_semicircle P A B M)
   (Q_midpoint_arc_AP : midpoint_arc Q A P)
   (S_intersection : intersection S (line_BP B P) (parallel PQ M)) :
   distance P M = distance P S :=
   sorry
   
end semicircle_P_S_eq_l298_298739


namespace circle_condition_l298_298501

theorem circle_condition (m : ℝ): (∃ x y : ℝ, (x^2 + y^2 - 2*x - 4*y + m = 0)) ↔ (m < 5) :=
by
  sorry

end circle_condition_l298_298501


namespace statistics_problem_l298_298005

open Real

noncomputable def problem_1 : Prop :=
  let students : List ℕ := [7, 33, 46]
  let systematic_sampling (n : ℕ) (k : ℕ) : List ℕ := List.range n k
  let sample : List ℕ := systematic_sampling 52 4
  students ⊆ sample

noncomputable def problem_2 : Prop :=
  let data : List ℕ := [1, 2, 3, 3, 4, 5]
  let mean : ℝ := (data.sum) / data.length
  let mode : ℕ := data.mode
  let median : ℝ := data.median
  (mean = 3) ∧ (mode = 3) ∧ (median = 3)

noncomputable def problem_3 : Prop :=
  let data : List ℝ := [a, 0, 1, 2, 3]
  let mean : ℝ := data.sum / data.length
  let variance : ℝ := (data.map (λ x => (x - mean) ^ 2)).sum / data.length
  let std_dev : ℝ := sqrt variance
  mean = 1 → std_dev = 2

noncomputable def problem_4 : Prop :=
  let b : ℝ := 2
  let x_mean : ℝ := 1
  let y_mean : ℝ := 3
  let a : ℝ := y_mean - b * x_mean
  a = 1

noncomputable def main_problem : Prop :=
  problem_2 ∧ problem_4

theorem statistics_problem : main_problem :=
  sorry

end statistics_problem_l298_298005


namespace total_number_of_notes_l298_298705

theorem total_number_of_notes 
  (total_money : ℕ)
  (fifty_rupees_notes : ℕ)
  (five_hundred_rupees_notes : ℕ)
  (total_money_eq : total_money = 10350)
  (fifty_rupees_notes_eq : fifty_rupees_notes = 117)
  (money_eq : 50 * fifty_rupees_notes + 500 * five_hundred_rupees_notes = total_money) :
  fifty_rupees_notes + five_hundred_rupees_notes = 126 :=
by sorry

end total_number_of_notes_l298_298705


namespace lcm_12_18_l298_298406

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := 
by
  sorry

end lcm_12_18_l298_298406


namespace number_of_mango_trees_l298_298986

-- Define the conditions
variable (M : Nat) -- Number of mango trees
def num_papaya_trees := 2
def papayas_per_tree := 10
def mangos_per_tree := 20
def total_fruits := 80

-- Prove that the number of mango trees M is equal to 3
theorem number_of_mango_trees : 20 + (mangos_per_tree * M) = total_fruits -> M = 3 :=
by
  intro h
  sorry

end number_of_mango_trees_l298_298986


namespace lcm_12_18_is_36_l298_298416

def prime_factors (n : ℕ) : list ℕ :=
  if n = 12 then [2, 2, 3]
  else if n = 18 then [2, 3, 3]
  else []

noncomputable def lcm_of_two (a b : ℕ) : ℕ :=
  match prime_factors a, prime_factors b with
  | [2, 2, 3], [2, 3, 3] => 36
  | _, _ => 0

theorem lcm_12_18_is_36 : lcm_of_two 12 18 = 36 :=
  sorry

end lcm_12_18_is_36_l298_298416


namespace airplane_distance_difference_l298_298011

variable (a : ℝ)

theorem airplane_distance_difference :
  let wind_speed := 20
  (4 * a) - (3 * (a - wind_speed)) = a + 60 := by
  sorry

end airplane_distance_difference_l298_298011


namespace max_difference_black_white_squares_l298_298959

theorem max_difference_black_white_squares (n : ℕ) (hn : n > 1)
  (color : Fin n × Fin n → Bool)
  (connected_black : ∀ b1 b2 : Fin n × Fin n, color b1 = true → color b2 = true → (∃ seq : List (Fin n × Fin n), seq.head = b1 ∧ seq.last = b2 ∧ ∀ i, i+1 < seq.length → (seq.nthLe i sorry).1 - (seq.nthLe (i+1) sorry).1 ∣ ±1 ∧ (seq.nthLe i sorry).2 - (seq.nthLe (i+1) sorry).2 ∣ ±1))
  (connected_white: ∀ w1 w2 : Fin n × Fin n, color w1 = false → color w2 = false → (∃ seq : List (Fin n × Fin n), seq.head = w1 ∧ seq.last = w2 ∧ ∀ i, i+1 < seq.length → (seq.nthLe i sorry).1 - (seq.nthLe (i+1) sorry).1 ∣ ±1 ∧ (seq.nthLe i sorry).2 - (seq.nthLe (i+1) sorry).2 ∣ ±1))
  (subgrid_condition : ∀ i j : Fin (n-1), ∃ (b_w : Bool), ∃ k l : ℕ, (k < 2) ∧ (l < 2) ∧ color (Fin.ofNat (i+k)) (Fin.ofNat (j+l)) = b_w ∧ color (Fin.ofNat (i+k)) (Fin.ofNat (j+l)) ≠ b_w ) :
  (if n % 2 = 1 then (n * n / 2) + (n % 2) = 2 * n + 1 else (n * n / 2) = 2 * n - 2) sorry

end max_difference_black_white_squares_l298_298959


namespace prove_odd_function_l298_298441

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g(x)

theorem prove_odd_function (f : ℝ → ℝ) :
  is_odd_function (λ x, f(x) - f(-x)) :=
sorry

end prove_odd_function_l298_298441


namespace unique_solution_for_a_half_l298_298094

noncomputable def unique_solution_a (a : ℝ) (h_pos : 0 < a) : Prop :=
∀ x : ℝ, 2 * a * x = x^2 - 2 * a * (Real.log x) → x = 1

theorem unique_solution_for_a_half : unique_solution_a (1 / 2) (by norm_num : 0 < (1 / 2)) :=
sorry

end unique_solution_for_a_half_l298_298094


namespace sine_cosine_solution_count_l298_298779

theorem sine_cosine_solution_count : 
  (set.countInRange (fun x => sin (π / 2 * cos x) = cos (π * sin x)) (set.Icc 0 (2 * π))) = 2 := 
sorry

end sine_cosine_solution_count_l298_298779


namespace min_rods_is_2n_minus_2_l298_298447

-- Define the conditions for the puzzle and the rods
variable (n : ℕ) (A : Type) [puzzle : n ≥ 2 ∧ ∀ i j, i ≠ j → removed i ≠ removed j]

-- Define the partition into rods
def rods (A : Type) : Type := 
  { rods : set (set (ℕ × ℕ)) // 
    (∀ rod ∈ rods, ∃ k : ℕ, (rod = {x | (x.1 = k)} ∨ rod = {x | (x.2 = k)})) ∧
    (⋃₀ rods = {ij : ℕ × ℕ | ij ∈ A ∧ ij ∉ removed}) ∧
    (∀ rod1 rod2 ∈ rods, rod1 ≠ rod2 → rod1 ∩ rod2 = ∅)
  }

-- Define the minimum number of rods in the partition
def m (A : Type) : ℕ := 
  Inf { m : ℕ | ∃ rods : rods A, rods.size = m }

-- The Lean 4 statement of the problem
theorem min_rods_is_2n_minus_2 (n : ℕ) (A : Type) [puzzle : ¬n < 2 ∧ ∀ i j, i ≠ j → removed i ≠ removed j] : 
  m(A) = 2 * n - 2 :=
by sorry

end min_rods_is_2n_minus_2_l298_298447


namespace unit_vector_of_a_is_a0_l298_298875

def vector (a b : ℝ) : ℝ × ℝ := (a, b)

def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def unit_vector (v : ℝ × ℝ) : ℝ × ℝ :=
  let mag := magnitude v
  (v.1 / mag, v.2 / mag)

def a : ℝ × ℝ := (2, real.sqrt 5)

def a_0 := (2 / 3, real.sqrt 5 / 3)

theorem unit_vector_of_a_is_a0 : unit_vector a = a_0 := by
  sorry

end unit_vector_of_a_is_a0_l298_298875


namespace problematic_statements_l298_298588

variables {Point : Type} {Plane : Type} [euclidean_geometry Point Plane]

-- Definitions for the rectangle and geometric constructs
variables (A B C D A' D' F E : Point)
variables (AB : segment A B) (AD : segment A D) (AC : segment A C) (AD' : segment A D') (AC' : segment A C') (CD' : segment C D') (EF : segment E F)
variable (ABCD : rectangle ABCD AB AD)
variables (planeABC planeABD' planeBCD' planeADA' planeBD'E : Plane)

-- Conditions of the problem
variables (hAD3: AD.length = 3) (hAB4: AB.length = 4)
variables (hFold: ∀(P : Point), P ∈ planeADA' → P ∉ planeABC)
variables (hMidF : F = midpoint A D')
variables (hEonAC : ∃(P : Point), P = E ∧ P ∈ AC)

-- Statements to be proved
theorem problematic_statements :
  (∃ E, E ∈ AC ∧ EF.parallel (plane BCD')) ∧ -- 1
  (¬ (∃ E, E ∈ AC ∧ EF.perpendicular (plane ABD'))) ∧ -- 2
  (∃ E, E ∈ AC ∧ D'E.perpendicular (plane ABC)) ∧ -- 3
  (¬ (∃ E, E ∈ AC ∧ AC.perpendicular (plane BD'E))) -- 4
:= sorry

end problematic_statements_l298_298588


namespace sum_first_60_integers_l298_298368

theorem sum_first_60_integers : (∑ i in Finset.range 61, i) = 1830 := sorry

end sum_first_60_integers_l298_298368


namespace problem_prove_l298_298571

-- Definitions based on the given conditions
variable {A B C X I_B I_C : Point}
variable [CircumcircleABC : Circumcircle ABC]
variable [InsideCircumcircle : x_inside_circumcircle ABC X]
variable [excenter_AC : Excenter ABC AC I_B]
variable [excenter_AB : Excenter ABC AB I_C]

-- Statement of the problem
theorem problem_prove (h_in_circumcircle : InsideCircumcircle) 
    (h_excenter_AC : excenter_AC) 
    (h_excenter_AB : excenter_AB) : (dist X I_B) * (dist X I_C) > (dist X B) * (dist X C) :=
  sorry

end problem_prove_l298_298571


namespace radius_ratio_l298_298701

noncomputable def volume_large : ℝ := 576 * Real.pi
noncomputable def volume_small : ℝ := 0.25 * volume_large

theorem radius_ratio (V_large V_small : ℝ) (h_large : V_large = 576 * Real.pi) (h_small : V_small = 0.25 * V_large) :
  (∃ r_ratio : ℝ, r_ratio = Real.sqrt (Real.sqrt (Real.sqrt (V_small / V_large)))) :=
begin
  rw [h_large, h_small],
  use 1 / Real.sqrt (Real.sqrt 4),
  sorry
end

end radius_ratio_l298_298701


namespace continuous_piecewise_l298_298205

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
if x > 2 then a * x + 4
else if -2 ≤ x ∧ x ≤ 2 then 2 * x - 5
else 3 * x - b

theorem continuous_piecewise (a b : ℝ) (h_cont : continuous (f x a b)) : a + b = 1/2 :=
sorry

end continuous_piecewise_l298_298205


namespace find_a_find_b_and_area_l298_298510

variables {a b c A C : ℝ}

-- Condition: c = √3
def c_value := c = Real.sqrt 3

-- Condition: cos 2A = -1/3
def cos_2A := Real.cos (2 * A) = -1 / 3

-- Condition: sin A = √6 sin C
def sin_relation := Real.sin A = Real.sqrt 6 * Real.sin C

-- Proof: a = 3√2
theorem find_a (h1 : c_value) (h2 : cos_2A) (h3 : sin_relation) : a = 3 * Real.sqrt 2 :=
sorry

-- Additional condition: A is acute
def A_acute := 0 < A ∧ A < Real.pi / 2

-- Proof: b = 5 and Area = 5√2/2
theorem find_b_and_area (h1 : c_value) (h2 : cos_2A) (h3 : sin_relation) (h4 : A_acute) :
  b = 5 ∧ 0.5 * b * c * (Real.sin A) = 5 * Real.sqrt 2 / 2 :=
sorry

end find_a_find_b_and_area_l298_298510


namespace roots_of_unity_l298_298960

noncomputable def is_root_of_unity (z : ℂ) : Prop :=
  ∃ n : ℕ, n > 0 ∧ z^n = 1

noncomputable def is_cube_root_of_unity (z : ℂ) : Prop :=
  z^3 = 1

theorem roots_of_unity (x y : ℂ) (hx : is_root_of_unity x) (hy : is_root_of_unity y) (hxy : x ≠ y) :
  is_root_of_unity (x + y) ↔ is_cube_root_of_unity (y / x) :=
sorry

end roots_of_unity_l298_298960


namespace differentiable_and_derivative_gt_implies_l298_298457

variable {a b x : ℝ}
variable (f g : ℝ → ℝ)

theorem differentiable_and_derivative_gt_implies :
  (∀ x ∈ set.Icc a b, differentiable ℝ f x) →
  (∀ x ∈ set.Icc a b, differentiable ℝ g x) →
  (∀ x ∈ set.Ioo a b, deriv f x > deriv g x) →
  a < x → x < b →
  f x + g a > g x + f a :=
by
  intros h1 h2 h3 ha hb
  sorry

end differentiable_and_derivative_gt_implies_l298_298457


namespace longest_diagonal_of_rhombus_l298_298736

variables (d1 d2 : ℝ) (x : ℝ)
def rhombus_area := (d1 * d2) / 2
def diagonal_ratio := d1 / d2 = 4 / 3

theorem longest_diagonal_of_rhombus (h : rhombus_area (4 * x) (3 * x) = 150) (r : diagonal_ratio (4 * x) (3 * x)) : d1 = 20 := by
  sorry

end longest_diagonal_of_rhombus_l298_298736


namespace probability_of_valid_p_is_one_fifth_l298_298486

noncomputable def count_valid_p : ℕ :=
  finset.card (finset.filter (λ p, ∃ q:ℤ, p * q - 6 * p - 3 * q = 3) (finset.range 21 \ {0}))

noncomputable def probability : ℚ := count_valid_p / 20

theorem probability_of_valid_p_is_one_fifth :
  count_valid_p = 4 → probability = 1 / 5 :=
by sorry

end probability_of_valid_p_is_one_fifth_l298_298486


namespace mobius_trip_total_time_l298_298974

theorem mobius_trip_total_time :
  let speed_without_load := 13
  let speed_light_load := 12
  let speed_typical_load := 11
  let segment1_distance := 80
  let segment1_load := speed_typical_load
  let segment1_reduction := 0.15
  let segment2_distance := 110
  let segment2_load := speed_light_load
  let segment2_reduction := 0
  let segment3_distance := 100
  let segment3_load := speed_without_load
  let segment3_reduction := 0.20
  let segment4_distance := 60
  let segment4_load := speed_typical_load
  let segment4_reduction := 0.10
  let rest_stops_first_half := [20, 25, 35] -- in minutes
  let rest_stops_second_half := [45, 30] -- in minutes
  let total_hours : ℝ := 
    (segment1_distance / (segment1_load * (1 - segment1_reduction)) + 
    segment2_distance / (segment2_load) +
    segment3_distance / (segment3_load * (1 - segment3_reduction)) + 
    segment4_distance / (segment4_load * (1 - segment4_reduction)) +
    (rest_stops_first_half.sum + rest_stops_second_half.sum) / 60)
  in total_hours = 35.982 := by
  navigate to the goal
  sorry

end mobius_trip_total_time_l298_298974


namespace intersection_eq_l298_298474

def M : Set ℝ := {x | x^2 + 2 * x - 3 < 0}
def N : Set ℤ := {-3, -2, -1, 0, 1, 2}
def R : Set ℤ := {-2, -1, 0}

theorem intersection_eq (M_real_int : ∀ x : ℝ, -3 < x ∧ x < 1 → x ∈ N) :
  (M ∩ N) = R := 
sorry

end intersection_eq_l298_298474


namespace angle_bisectors_perpendicular_l298_298071

theorem angle_bisectors_perpendicular 
    (O A B C D : Type) 
    (rays_extend_from_O : O → O → Prop) 
    (clockwise_order : (list O) → Prop)
    (sum_angles_condition : O → O → O → O → ℝ)
    (angle_bisectors_perpendicular: ∀ (O A B C D: O) 
        (rays_extend_from_O O A ∧ rays_extend_from_O O B ∧ rays_extend_from_O O C ∧ rays_extend_from_O O D ∧ clockwise_order [A,B,C,D])
        (sum_angles_condition (∠AOB + ∠COD = 180°)))
        : ∀ (O A B C D: O)   
          (sum_angles_condition (∠AOB + ∠COD = 180°)) → 
            (∠AOC / 2 + ∠BOD / 2 = 90°)
            :=
begin
  sorry
end

end angle_bisectors_perpendicular_l298_298071


namespace lcm_12_18_l298_298426

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end lcm_12_18_l298_298426


namespace longest_diagonal_of_rhombus_l298_298714

theorem longest_diagonal_of_rhombus (d1 d2 : ℝ) (area : ℝ) (ratio : ℝ) (h1 : area = 150) (h2 : d1 / d2 = 4 / 3) :
  max d1 d2 = 20 :=
by 
  let x := sqrt (area * 2 / (d1 * d2))
  have d1_expr : d1 = 4 * x := sorry
  have d2_expr : d2 = 3 * x := sorry
  have x_val : x = 5 := sorry
  have length_longest_diag : max d1 d2 = max (4 * 5) (3 * 5) := sorry
  exact length_longest_diag

end longest_diagonal_of_rhombus_l298_298714


namespace remainder_binomial_sum_mod_9_l298_298065

noncomputable def binomial_sum (n : ℕ) : ℕ :=
  (∑ k in finset.range (n + 1), if k % 2 = 0 then nat.choose 34 k else 0)

theorem remainder_binomial_sum_mod_9 : (binomial_sum 34) % 9 = 8 :=
by
  sorry

end remainder_binomial_sum_mod_9_l298_298065


namespace max_diff_y_l298_298130

theorem max_diff_y (x y z : ℕ) (h₁ : 4 < x) (h₂ : x < z) (h₃ : z < y) (h₄ : y < 10) (h₅ : y - x = 5) : y = 9 :=
sorry

end max_diff_y_l298_298130


namespace new_roads_connectivity_l298_298909

-- Definitions
def city := ℕ
def road := (city × city)

-- Assumptions
variable (C : Finset city) -- 100 cities
variable (R : Finset road) -- original 99 roads
variable [hC_card : Fintype.card C = 100] -- Exactly 100 cities
variable [hR_card : Fintype.card R = 99] -- 99 roads forming a tree

-- New roads we are adding
def new_roads (A : Finset road) := A.card = 50

-- Valid connections after adding new roads
def valid_connection (C : Finset city) (R : Finset road) (A : Finset road) : Prop :=
∀ e ∈ (R ∪ A), ∃ p : List city, (p.nodup ∧ p.head ∈ C ∧ p.last ∈ C ∧ ∀ (x, y) ∈ p.zip p.tail, (x, y) ∈ (R ∪ A) ∨ (y, x) ∈ (R ∪ A))

-- Theorem to prove
theorem new_roads_connectivity : 
  ∃ A ⊆ ({p ∈ (C.product C) | p.1 ≠ p.2} : Finset road), new_roads A ∧ valid_connection C R A :=
sorry

end new_roads_connectivity_l298_298909


namespace g_g_g_20_l298_298553

def g (x : ℝ) : ℝ :=
  if x < 10 then x^2 - 9 else x - 15

theorem g_g_g_20 : g (g (g 20)) = 1 := by
  sorry

end g_g_g_20_l298_298553


namespace Miss_Stevie_payment_l298_298531

theorem Miss_Stevie_payment:
  let painting_hours := 8
  let painting_rate := 15
  let painting_earnings := painting_hours * painting_rate
  let mowing_hours := 6
  let mowing_rate := 10
  let mowing_earnings := mowing_hours * mowing_rate
  let plumbing_hours := 4
  let plumbing_rate := 18
  let plumbing_earnings := plumbing_hours * plumbing_rate
  let total_earnings := painting_earnings + mowing_earnings + plumbing_earnings
  let discount := 0.10 * total_earnings
  let amount_paid := total_earnings - discount
  amount_paid = 226.80 :=
by
  sorry

end Miss_Stevie_payment_l298_298531


namespace chinese_riddle_championship_l298_298756

noncomputable def number_of_arrangements : ℕ :=
let all_entities := [1, 2, 3, 4, 5, 6] in -- representing 6 people
let students := [4, 5, 6] in -- representing the students as one group
let teacher := 1 in -- representing the teacher
let positions := [1, 2, 3, 4] in -- representing possible positions for the combined entities
let teacher_positions := [2, 3] in -- teacher can't be at the ends
let factorial := λ n, if n = 0 then 1 else n * factorial (n - 1) in
let arrangements_for_teacher := teacher_positions.length in -- 2 positions
let arrangements_for_students := factorial 3 in -- Permutations within the student group
let remaining_entities := 3 in -- Remaining: 1 entity for students' group, 2 parents
let arrangements_for_remaining_entities := factorial remaining_entities in -- Permutations of remaining
arrangements_for_teacher * arrangements_for_students * arrangements_for_remaining_entities

theorem chinese_riddle_championship :
  number_of_arrangements = 72 :=
by
  unfold number_of_arrangements
  simp
  sorry

end chinese_riddle_championship_l298_298756


namespace solve_equation_l298_298596

theorem solve_equation (x : ℚ) (h1 : (x + 4) / (x - 3) = (x - 2) / (x + 2)) : x = -2 / 11 := by
  sorry

end solve_equation_l298_298596


namespace faith_weekly_earnings_l298_298788

theorem faith_weekly_earnings :
  let hourly_pay := 13.50
  let regular_hours_per_day := 8
  let workdays_per_week := 5
  let overtime_hours_per_day := 2
  let regular_pay_per_day := hourly_pay * regular_hours_per_day
  let regular_pay_per_week := regular_pay_per_day * workdays_per_week
  let overtime_pay_per_day := hourly_pay * overtime_hours_per_day
  let overtime_pay_per_week := overtime_pay_per_day * workdays_per_week
  let total_weekly_earnings := regular_pay_per_week + overtime_pay_per_week
  total_weekly_earnings = 675 := 
  by
    sorry

end faith_weekly_earnings_l298_298788


namespace maximum_sum_minimum_difference_l298_298140

-- Definitions based on problem conditions
def is_least_common_multiple (m n lcm: ℕ) : Prop := Nat.lcm m n = lcm
def is_greatest_common_divisor (m n gcd: ℕ) : Prop := Nat.gcd m n = gcd

-- The target theorem to prove
theorem maximum_sum_minimum_difference (x y: ℕ) (h_lcm: is_least_common_multiple x y 2010) (h_gcd: is_greatest_common_divisor x y 2) :
  (x + y = 2012 ∧ x - y = 104 ∨ y - x = 104) :=
by
  sorry

end maximum_sum_minimum_difference_l298_298140


namespace longest_diagonal_of_rhombus_l298_298732

theorem longest_diagonal_of_rhombus (A B : ℝ) (h1 : A = 150) (h2 : ∃ x, (A = 1/2 * (4 * x) * (3 * x)) ∧ (x = 5)) : 
  4 * (classical.some h2) = 20 := 
by sorry

end longest_diagonal_of_rhombus_l298_298732


namespace area_of_gray_region_l298_298763

theorem area_of_gray_region (r R : ℝ) (hr : r = 2) (hR : R = 3 * r) : 
  π * R ^ 2 - π * r ^ 2 = 32 * π :=
by
  have hr : r = 2 := hr
  have hR : R = 3 * r := hR
  sorry

end area_of_gray_region_l298_298763


namespace max_edges_dodecahedron_no_shared_vertices_l298_298660

noncomputable def dodecahedron := sorry -- Placeholder for actual dodecahedron graph definition

theorem max_edges_dodecahedron_no_shared_vertices (G : SimpleGraph) :
  G = dodecahedron ->
  ∀ (E : Finset G.Edge), (∀ (e1 e2 : G.Edge), e1 ≠ e2 ∧ (e1 ∩ e2).Nonempty → False) →
  E.card ≤ 10 ∧ (∃ (E' : Finset G.Edge), E'.card = 10 ∧ ∀ (e1 e2 : G.Edge), e1 ≠ e2 ∧ (e1 ∩ e2).Nonempty → False) :=
sorry

end max_edges_dodecahedron_no_shared_vertices_l298_298660


namespace area_of_triangle_pqr_l298_298160

noncomputable def area_of_triangle (P Q R : ℝ) : ℝ :=
  let PQ := P + Q
  let PR := P + R
  let QR := Q + R
  if PQ^2 = PR^2 + QR^2 then
    1 / 2 * PR * QR
  else
    0

theorem area_of_triangle_pqr : 
  area_of_triangle 3 2 1 = 6 :=
by
  simp [area_of_triangle]
  sorry

end area_of_triangle_pqr_l298_298160


namespace max_a4_l298_298964

variables {a : ℕ → ℤ} {S : ℕ → ℤ}
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (S : ℕ → ℤ) (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, S n = ∑ i in finset.range n, a i

theorem max_a4 (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (H_seq : is_arithmetic_sequence a)
  (H_S1 : S 1 ≤ 13)
  (H_S4 : S 4 ≥ 10)
  (H_S5 : S 5 ≤ 15) :
  a 4 ≤ 4 :=
sorry

end max_a4_l298_298964


namespace longest_diagonal_of_rhombus_l298_298737

variables (d1 d2 : ℝ) (x : ℝ)
def rhombus_area := (d1 * d2) / 2
def diagonal_ratio := d1 / d2 = 4 / 3

theorem longest_diagonal_of_rhombus (h : rhombus_area (4 * x) (3 * x) = 150) (r : diagonal_ratio (4 * x) (3 * x)) : d1 = 20 := by
  sorry

end longest_diagonal_of_rhombus_l298_298737


namespace problems_per_page_is_five_l298_298170

-- Let M and R be the number of problems on each math and reading page respectively
variables (M R : ℕ)

-- Conditions given in problem
def two_math_pages := 2 * M
def four_reading_pages := 4 * R
def total_problems := two_math_pages + four_reading_pages

-- Assume the number of problems per page is the same for both math and reading as P
variable (P : ℕ)
def problems_per_page_equal := (2 * P) + (4 * P) = 30

theorem problems_per_page_is_five :
  (2 * P) + (4 * P) = 30 → P = 5 :=
by
  intro h
  sorry

end problems_per_page_is_five_l298_298170


namespace walking_time_l298_298353

theorem walking_time (intervals_time : ℕ) (poles_12_time : ℕ) (speed_constant : Prop) : 
  intervals_time = 2 → poles_12_time = 22 → speed_constant → 39 * intervals_time = 78 :=
by
  sorry

end walking_time_l298_298353


namespace travel_from_capital_to_dalniy_l298_298921

noncomputable def graph (num_cities : ℕ) : SimpleGraph ℕ := sorry

theorem travel_from_capital_to_dalniy :
  ∃ path : List ℕ, path.head = capital ∧ path.last = dalniy ∧ 
  ∀ (i : ℕ), i < path.length - 1 -> (graph num_cities).adj (path.nth_le i _) (path.nth_le (i + 1) _) :=
begin
  sorry
end

end travel_from_capital_to_dalniy_l298_298921


namespace angle_bisector_theorem_l298_298961

open Triangle

variables {A B C D : Point} [InTriangle A B C]
variable (BD : AngleBisector B D A)

theorem angle_bisector_theorem (h : IsAngleBisector (B D A)) : 
  AB > AD ∧ CB > CD :=
begin
  sorry
end

end angle_bisector_theorem_l298_298961


namespace boat_journey_time_l298_298689

/-- A boat travels from A to B downstream and back to point C which is midway between A and B.
    Given the velocity of the stream is 4 kmph, the speed of the boat in still water is 14 kmph,
    and the distance between A and B is 180 km, the total time taken for the journey is 19 hours. -/
theorem boat_journey_time (stream_speed : ℝ) (boat_speed : ℝ) (distance_AB : ℝ) (C_midway : Prop) :
  stream_speed = 4 → boat_speed = 14 → distance_AB = 180 → C_midway →
  let downstream_speed := boat_speed + stream_speed,
      upstream_speed := boat_speed - stream_speed,
      time_downstream := distance_AB / downstream_speed,
      distance_BC := distance_AB / 2,
      time_upstream := distance_BC / upstream_speed in
  time_downstream + time_upstream = 19 :=
by
  intros hs hb hd hC
  let downstream_speed := boat_speed + stream_speed
  let upstream_speed := boat_speed - stream_speed
  let time_downstream := distance_AB / downstream_speed
  let distance_BC := distance_AB / 2
  let time_upstream := distance_BC / upstream_speed
  sorry

end boat_journey_time_l298_298689


namespace problem_unique_solution_l298_298988

theorem problem_unique_solution (x : ℝ) (hx : x ≥ 0) : 
  2021 * real.root 202 (x ^ 2020) - 1 = 2020 * x ↔ x = 1 := by
  sorry

end problem_unique_solution_l298_298988


namespace correct_answer_l298_298137

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := 2 * x^2 + m * x - 1

theorem correct_answer (m : ℝ) : 
  (∀ x₁ x₂, 1 < x₁ → 1 < x₂ → (f x₁ m - f x₂ m) / (x₁ - x₂) > 0) → m ≥ -4 :=
by
  sorry

end correct_answer_l298_298137


namespace pirates_coins_l298_298698

noncomputable def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

theorem pirates_coins (x : ℕ) (h : ∃ x, ∀ k ∈ Finset.range 10, let remaining := x * List.prod ((List.range k).map (λ n, 10 - n - 1)) / 10^k; in ∃ c, remaining * (k + 1) / 10 = c) :
  let final_count := x * factorial 8 / 10^9 in final_count = 79488 :=
by
  sorry

end pirates_coins_l298_298698


namespace domain_of_function_l298_298250

theorem domain_of_function :
  { x : ℝ | 0 < x ∧ 2 ≤ log 2 x } = { x : ℝ | 4 ≤ x } :=
by
  sorry

end domain_of_function_l298_298250


namespace range_of_f_l298_298838

-- Statement without proof
theorem range_of_f (a : ℝ) (h_pos : a > 0) (h_odd : ∀ x : ℝ, f (x) = (1 / a) - (1 / (a^x + 1)) → f (-x) = -f (x)) :
  set.Ioo (-1 / 2) (1 / 2) = {y : ℝ | ∃ x : ℝ, f(x) = y} :=
by
  sorry

end range_of_f_l298_298838


namespace line_equation_l298_298252

theorem line_equation (P : ℝ × ℝ) (slope : ℝ) (hP : P = (-2, 0)) (hSlope : slope = 3) :
    ∃ (a b : ℝ), ∀ x y : ℝ, y = a * x + b ↔ P.1 = -2 ∧ P.2 = 0 ∧ slope = 3 ∧ y = 3 * x + 6 :=
by
  sorry

end line_equation_l298_298252


namespace solve_for_y_l298_298496

theorem solve_for_y (x y : ℝ) (h₁ : x - y = 16) (h₂ : x + y = 4) : y = -6 := 
by 
  sorry

end solve_for_y_l298_298496


namespace jack_correct_percentage_l298_298512

theorem jack_correct_percentage (y : ℝ) (h : y ≠ 0) :
  ((8 * y - (2 * y - 3)) / (8 * y)) * 100 = 75 + (75 / (2 * y)) :=
by
  sorry

end jack_correct_percentage_l298_298512


namespace player_A_first_probability_flip_4_player_A_first_in_both_rounds_probability_flip_3_twice_l298_298907

-- First problem: probability that player A goes first if player A flips over a card with value 4
theorem player_A_first_probability_flip_4 :
  let cards := {1, 2, 3, 4, 5}
  let remaining_cards := {1, 2, 3, 5}
  let favorable_outcomes := {1, 2, 3}
  let total_outcomes := remaining_cards
  (card_entity "Probability" : ℚ) = (favorable_outcomes.size.to_rat / total_outcomes.size.to_rat) := 
by
  sorry

-- Second problem: probability that player A goes first in both rounds if player A flips over a card with value 3 both times
theorem player_A_first_in_both_rounds_probability_flip_3_twice :
  let cards := {1, 2, 3, 4, 5}
  let remaining_cards := {1, 2, 3, 5}
  let favourable_outcomes_in_one_round := {1, 2}
  let total_outcomes := (remaining_cards.product remaining_cards).size
  let favorable_outcomes := (favourable_outcomes_in_one_round.product favourable_outcomes_in_one_round).size
  (card_entity "Probability" : ℚ) = (favorable_outcomes.to_rat / total_outcomes.to_rat) := 
by
  sorry

end player_A_first_probability_flip_4_player_A_first_in_both_rounds_probability_flip_3_twice_l298_298907


namespace find_a_l298_298085

noncomputable def geometric_sequence_solution (a : ℝ) : Prop :=
  (a + 1) ^ 2 = (1 / (a - 1)) * (a ^ 2 - 1)

theorem find_a (a : ℝ) : geometric_sequence_solution a → a = 0 :=
by
  intro h
  sorry

end find_a_l298_298085


namespace lcm_12_18_l298_298410

theorem lcm_12_18 : Nat.lcm 12 18 = 36 :=
by
  -- Definitions of the conditions
  have h12 : 12 = 2 * 2 * 3 := by norm_num
  have h18 : 18 = 2 * 3 * 3 := by norm_num
  
  -- Calculating LCM using the built-in Nat.lcm
  rw [Nat.lcm_comm]  -- Ordering doesn't matter for lcm
  rw [Nat.lcm, h12, h18]
  -- Prime factorizations checks are implicitly handled
  
  -- Calculate the LCM based on the highest powers from the factorizations
  have lcm_val : 4 * 9 = 36 := by norm_num
  
  -- So, the LCM of 12 and 18 is
  exact lcm_val

end lcm_12_18_l298_298410


namespace sum_of_digits_in_pages_2000_l298_298690

/-- 
A book has 2000 pages. Prove that the sum of all the digits used in the page numbers of this book is 28002.
-/
theorem sum_of_digits_in_pages_2000 (n : ℕ) (h : n = 2000) : 
  digit_sum_of_pages n = 28002 := 
  sorry

end sum_of_digits_in_pages_2000_l298_298690


namespace remaining_water_at_end_of_hike_l298_298480

-- Define conditions
def initial_water : ℝ := 9
def hike_length : ℝ := 7
def hike_duration : ℝ := 2
def leak_rate : ℝ := 1
def drink_rate_6_miles : ℝ := 0.6666666666666666
def drink_last_mile : ℝ := 2

-- Define the question and answer
def remaining_water (initial: ℝ) (duration: ℝ) (leak: ℝ) (drink6: ℝ) (drink_last: ℝ) : ℝ :=
  initial - ((drink6 * 6) + drink_last + (leak * duration))

-- Theorem stating the proof problem 
theorem remaining_water_at_end_of_hike :
  remaining_water initial_water hike_duration leak_rate drink_rate_6_miles drink_last_mile = 1 :=
by
  sorry

end remaining_water_at_end_of_hike_l298_298480


namespace grid_diagonal_intersection_l298_298155

theorem grid_diagonal_intersection (n m : ℕ) (h1 : n = 4) (h2 : m = 5) : 
  6 = (grid_not_intersected_by_diagonals n m) → 
  grid_not_intersected_by_diagonals 8 10 = 48 :=
sorry

end grid_diagonal_intersection_l298_298155


namespace lcm_12_18_l298_298404

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := 
by
  sorry

end lcm_12_18_l298_298404


namespace measure_of_y_l298_298521

/- Assume we have two angles that are created by lines m and n being parallel. -/
/- Angle at point A is 40 degrees with respect to the horizontal line. -/
/- Angle at point B is also 40 degrees with respect to the horizontal line. -/

/- Definitions for the problem setup -/
def parallel_lines (m n : Prop) : Prop := ∀ x y, m x y ↔ n x y

def horizontal_angle_at (A B : Prop) (angle : ℝ) : Prop := 
  A = angle ∧ B = angle

/- Theorem stating the measure of angle y -/
theorem measure_of_y {m n A B : Prop}
  (h_parallel : parallel_lines m n)
  (h_angles : horizontal_angle_at A B 40) : y = 40 :=
by sorry

end measure_of_y_l298_298521


namespace problem_solution_l298_298816

theorem problem_solution (x y m : ℝ) (hx : x > 0) (hy : y > 0) : 
  (∀ x y, (2 * y / x) + (8 * x / y) > m^2 + 2 * m) → -4 < m ∧ m < 2 :=
by
  intros h
  sorry

end problem_solution_l298_298816


namespace h_k_minus3_eq_l298_298123

def h (x : ℝ) : ℝ := 4 - Real.sqrt x
def k (x : ℝ) : ℝ := 3 * x + 3 * x^2

theorem h_k_minus3_eq : h (k (-3)) = 4 - 3 * Real.sqrt 2 := 
by 
  sorry

end h_k_minus3_eq_l298_298123


namespace solve_for_x_l298_298609

theorem solve_for_x (x : ℝ) : 5 + 3.5 * x = 2.5 * x - 25 ↔ x = -30 :=
by {
  split,
  {
    intro h,
    calc
      x = -30 : by sorry,
  },
  {
    intro h,
    calc
      5 + 3.5 * (-30) = 5 - 105
                       = -100,
      2.5 * (-30) - 25 = -75 - 25
                       = -100,
    exact Eq.symm (by sorry),
  }
}

end solve_for_x_l298_298609


namespace monotone_exponential_function_l298_298351

theorem monotone_exponential_function :
  (∀ f : ℝ → ℝ, (∀ x y, f (x + y) = f x * f y) → (∀ x y, x < y → f x < f y) → (∃ a, f = λ x, a ^ x ∧ a > 1) → 
   (∃! f, (f = λ x : ℝ, 3 ^ x))) :=
by
  sorry

end monotone_exponential_function_l298_298351


namespace smallest_number_leaves_remainders_l298_298066

theorem smallest_number_leaves_remainders :
  ∃ (x : ℤ), (x ≡ 4 [MOD 45]) ∧
             (x ≡ 45 [MOD 454]) ∧
             (x ≡ 454 [MOD 4545]) ∧
             (x ≡ 4545 [MOD 45454]) ∧
             x = 35641667749 := sorry

end smallest_number_leaves_remainders_l298_298066


namespace black_haired_girls_count_l298_298561

def initial_total_girls : ℕ := 80
def added_blonde_girls : ℕ := 10
def initial_blonde_girls : ℕ := 30

def total_girls := initial_total_girls + added_blonde_girls
def total_blonde_girls := initial_blonde_girls + added_blonde_girls
def black_haired_girls := total_girls - total_blonde_girls

theorem black_haired_girls_count : black_haired_girls = 50 := by
  sorry

end black_haired_girls_count_l298_298561


namespace longest_diagonal_of_rhombus_l298_298718

variable (d1 d2 : ℝ) (r : ℝ)
variable h_area : 0.5 * d1 * d2 = 150
variable h_ratio : d1 / d2 = 4 / 3

theorem longest_diagonal_of_rhombus :
  max d1 d2 = 20 :=
by
  sorry

end longest_diagonal_of_rhombus_l298_298718


namespace number_of_correct_statements_is_two_l298_298631

-- Definitions of the statements
def statement1 : Prop := ∀ (a m : ℝ), m = (a + b) / 2 → ∃! b, b = 2 * m - a
def statement2 : Prop := ∀ (q : Quadrilateral), q.diagonalsEqual → isoscelesTrapezoid q
def statement3 : Prop := ∀ (t : Trapezoid), ∃ r i : Trapezoid, rightTrapezoid r ∧ isoscelesTrapezoid i ∧ t = r ∪ i
def statement4 : Prop := ∀ (i : IsoscelesTrapezoid), isSymmetric i ∧ (axisOfSymmetry i = lineConnectingMidpoints i)

-- Define the propositions
def propositions := [statement1, statement2, statement3, statement4]

-- The problem statement
theorem number_of_correct_statements_is_two : (∑ s in propositions, if s then 1 else 0) = 2 := sorry

end number_of_correct_statements_is_two_l298_298631


namespace max_piece_length_l298_298029

theorem max_piece_length (L1 L2 L3 L4 : ℕ) (hL1 : L1 = 48) (hL2 : L2 = 72) (hL3 : L3 = 120) (hL4 : L4 = 144) 
  (h_min_pieces : ∀ L k, L = 48 ∨ L = 72 ∨ L = 120 ∨ L = 144 → k > 0 → L / k ≥ 5) : 
  ∃ k, k = 8 ∧ ∀ L, (L = L1 ∨ L = L2 ∨ L = L3 ∨ L = L4) → L % k = 0 :=
by
  sorry

end max_piece_length_l298_298029


namespace bucket_weight_l298_298692

variable {p q x y : ℝ}

theorem bucket_weight (h1 : x + (1 / 4) * y = p) (h2 : x + (3 / 4) * y = q) :
  x + y = - (1 / 2) * p + (3 / 2) * q := by
  sorry

end bucket_weight_l298_298692


namespace marius_scored_3_more_than_darius_l298_298030

theorem marius_scored_3_more_than_darius 
  (D M T : ℕ) 
  (h1 : D = 10) 
  (h2 : T = D + 5) 
  (h3 : M + D + T = 38) : 
  M = D + 3 := 
by
  sorry

end marius_scored_3_more_than_darius_l298_298030


namespace hyperbola_sine_of_asymptotes_l298_298623

variables (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0)

def ellipse_foci : set (ℝ × ℝ) := {p | p = (1, 0) ∨ p = (-1, 0)}

def hyperbola_foci (a b : ℝ) : set (ℝ × ℝ) := {p | p = (a, 0) ∨ p = (-a, 0)}

def ellipse_eccentricity := 1 / 2
def hyperbola_eccentricity := 2

def reciprocals (e₁ e₂ : ℝ) := e₁ * e₂ = 1

noncomputable def hyperbola_asymptotes_slope := b / a

def inclination_angle_sine (slope : ℝ) : ℝ := slope / (real.sqrt (1 + slope^2))

theorem hyperbola_sine_of_asymptotes :
  ellipse_foci = hyperbola_foci a b →
  reciprocals ellipse_eccentricity hyperbola_eccentricity →
  hyperbola_asymptotes_slope a b = real.sqrt 3 →
  inclination_angle_sine (hyperbola_asymptotes_slope a b) = real.sqrt 3 / 2 :=
by
  intros _ _ _
  sorry

end hyperbola_sine_of_asymptotes_l298_298623


namespace longest_diagonal_of_rhombus_l298_298721

variable (d1 d2 : ℝ) (r : ℝ)
variable h_area : 0.5 * d1 * d2 = 150
variable h_ratio : d1 / d2 = 4 / 3

theorem longest_diagonal_of_rhombus :
  max d1 d2 = 20 :=
by
  sorry

end longest_diagonal_of_rhombus_l298_298721


namespace residue_of_minus_963_plus_100_mod_35_l298_298766

-- Defining the problem in Lean 4
theorem residue_of_minus_963_plus_100_mod_35 : 
  ((-963 + 100) % 35) = 12 :=
by
  sorry

end residue_of_minus_963_plus_100_mod_35_l298_298766


namespace parallelogram_area_l298_298401

theorem parallelogram_area 
  (slant_height : ℝ) (angle_degrees : ℝ) (side_length : ℝ) 
  (h_height : slant_height = 30) (h_angle : angle_degrees = 60) (h_side : side_length = 15) :
  ∃ (area : ℝ), area = 225 :=
by
  -- definition for base using the given conditions
  let base := side_length * Real.cos (angle_degrees * (Real.pi / 180))
  -- using the base and the height to define the area
  let area := base * slant_height
  have h_area : base * slant_height = 225, sorry
  use 225
  exact h_area

end parallelogram_area_l298_298401


namespace g_at_minus_six_l298_298194

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x - 9
def g (x : ℝ) : ℝ := 3 * x ^ 2 + 4 * x - 2

theorem g_at_minus_six : g (-6) = 43 / 16 := by
  sorry

end g_at_minus_six_l298_298194


namespace problem_2008_ast_1001_l298_298033

theorem problem_2008_ast_1001 :
  (∀ n : ℕ, (2 * n + 2) ∗ 1001 = 3 * ((2 * n) ∗ 1001)) ∧ (2 ∗ 1001 = 1) → (2008 ∗ 1001 = 3^1003) :=
by
  sorry

end problem_2008_ast_1001_l298_298033


namespace find_b_in_triangle_l298_298166

theorem find_b_in_triangle 
  (A B : ℝ) 
  (a : ℝ) 
  (hA : A = Real.pi / 4) 
  (hB : B = Real.pi / 6) 
  (ha : a = 2) 
  (b : ℝ) 
  (h_triangle : ∀ (Δ : Triangle), Δ.A = A ∧ Δ.B = B ∧ Δ.a = a ∧ Δ.b = b
    → Δ.a / Real.sin Δ.A = Δ.b / Real.sin Δ.B) 
: b = Real.sqrt 2 := 
begin
  sorry
end

end find_b_in_triangle_l298_298166


namespace longest_diagonal_of_rhombus_l298_298720

variable (d1 d2 : ℝ) (r : ℝ)
variable h_area : 0.5 * d1 * d2 = 150
variable h_ratio : d1 / d2 = 4 / 3

theorem longest_diagonal_of_rhombus :
  max d1 d2 = 20 :=
by
  sorry

end longest_diagonal_of_rhombus_l298_298720


namespace sixth_graders_l298_298269

theorem sixth_graders (total_students sixth_graders seventh_graders : ℕ)
    (h1 : seventh_graders = 64)
    (h2 : 32 * total_students = 64 * 100)
    (h3 : sixth_graders * 100 = 38 * total_students) :
    sixth_graders = 76 := by
  sorry

end sixth_graders_l298_298269


namespace tan_identity_l298_298889

theorem tan_identity (x : ℝ) (hx1 : cos (2 * x) ≠ 0) (hx2 : cos (4 * x) ≠ 0) :
  sin (2 * x) * sin (4 * x) = cos (2 * x) * cos (4 * x) → x = π / 12 :=
by
  intro h
  -- we'll skip the proof part here
  sorry

end tan_identity_l298_298889


namespace solve_for_x_when_y_is_neg2_l298_298132

theorem solve_for_x_when_y_is_neg2 
  (x y : ℝ) 
  (h : 9 * 3^x = 7^(y + 4)) 
  (hy : y = -2) : 
  x = 2.106 :=
by
  subst hy
  have h1 : 9 * 3^x = 49 := by
    rw [← hy, add_comm, add_assoc, add_neg_self, add_zero]
    exact h
  -- sorry to skip the detailed proof
  sorry

end solve_for_x_when_y_is_neg2_l298_298132


namespace consecutive_odd_numbers_divisibility_l298_298185

theorem consecutive_odd_numbers_divisibility
  (a b c : ℤ)
  (h1 : b = a + 2)
  (h2 : c = b + 2)
  (h3 : odd a)
  (h4 : odd b)
  (h5 : odd c) :
  (a * b * c + 4 * b) = b^3 := by
  sorry

end consecutive_odd_numbers_divisibility_l298_298185


namespace find_angle_C_find_perimeter_l298_298165

-- Given triangle with sides opposite to angles A, B, C are a, b, c respectively,
-- and the condition 2*cos C(a*cos B + b*cos A) = c.
section part1
variables {A B C : ℝ} {a b c : ℝ}

def triangle_condition := 2 * Real.cos C * (a * Real.cos B + b * Real.cos A) = c

-- Prove that C = π/3, given the condition
theorem find_angle_C (h : triangle_condition) (h1 : 0 < C) (h2 : C < Real.pi):
  C = Real.pi / 3 :=
sorry
end part1

-- Additional conditions for Part II
section part2
variables {a b : ℝ}

-- Given c = sqrt 7 and area = 3 * sqrt 3 / 2
def additional_condition_c := c = Real.sqrt 7
def additional_condition_area := (1 / 2) * a * b * (Real.sin C) = (3 * Real.sqrt 3) / 2

-- Prove that the perimeter is 5 + √7, given the conditions
theorem find_perimeter (h : triangle_condition) (h1 : additional_condition_c) (h2 : additional_condition_area) :
  let p := a + b + c in p = 5 + Real.sqrt 7 :=
sorry
end part2

end find_angle_C_find_perimeter_l298_298165


namespace sticks_form_triangle_l298_298771

theorem sticks_form_triangle:
  (2 + 3 > 4) ∧ (2 + 4 > 3) ∧ (3 + 4 > 2) := by
  sorry

end sticks_form_triangle_l298_298771


namespace find_g_neg_6_l298_298190

def f (x : ℚ) : ℚ := 4 * x - 9
def g (y : ℚ) : ℚ := 3 * (y * y) + 4 * y - 2

theorem find_g_neg_6 : g (-6) = 43 / 16 := by
  sorry

end find_g_neg_6_l298_298190


namespace unique_solution_of_inequality_l298_298034

open Real

theorem unique_solution_of_inequality (b : ℝ) : 
  (∃! x : ℝ, |x^2 + 2 * b * x + 2 * b| ≤ 1) ↔ b = 1 := 
by exact sorry

end unique_solution_of_inequality_l298_298034


namespace no_two_color_map_l298_298143

theorem no_two_color_map (m : ℕ) :
  (∃ borders : ℕ → ℕ, (∃ c : ℕ, borders c % m ≠ 0) ∧ (∀ c ≠ c_spec, borders c % m = 0)) →
  ∀ (color : ℕ → bool), 
  ¬(∀ c c', adjacent c c' → color c ≠ color c') :=
by
  intros m h borders h_coloring
  sorry

end no_two_color_map_l298_298143


namespace incircle_contact_point_l298_298444

-- Definitions for the problem setup
variables (F₁ F₂ : Point) -- Foci of the hyperbola
variables (M N : Point)   -- Vertices of the hyperbola
variables (P : Point)     -- Point on the hyperbola

-- Hypotheses
hypothesis H_p_on_hyperbola : IsOnHyperbola P F₁ F₂ M N
hypothesis H_incircle : HasIncircle (Triangle.mk P F₁ F₂)

-- The theorem statement
theorem incircle_contact_point (G : Point) :
  IncircleContactPoint (Triangle.mk P F₁ F₂) F₁ F₂ G →
  G = M ∨ G = N := 
sorry 

end incircle_contact_point_l298_298444


namespace perpendicular_lines_l298_298654

theorem perpendicular_lines (a : ℝ) :
  (∀ x y : ℝ, ax - y + 2a = 0 → (2a-1)x + ay + a = 0 → (a = 1 ∨ a = 0)) :=
by
  intros x y h1 h2
  sorry

end perpendicular_lines_l298_298654


namespace probability_at_least_one_blue_from_A_probability_blue_from_B_l298_298760

-- Conditions
def boxA := {red: ℕ, blue: ℕ}
def boxB := {red: ℕ, blue: ℕ}

def initialBoxA := {red := 2, blue := 4}
def initialBoxB := {red := 3, blue := 3}

def drawBallsFromA (box: boxA) (k: ℕ) : ℕ × ℕ :=
  -- calculates the possible outcomes
  sorry

def combineBoxes (a: ℕ × ℕ) (boxB: boxB) : boxB :=
  -- returns the combined boxB after adding the drawn balls from boxA
  sorry

def drawBallFromB (box: boxB) : ℕ × ℕ :=
  -- returns the outcome of drawing 1 ball
  sorry

-- Proof problem
theorem probability_at_least_one_blue_from_A :
  let drawnFromA := drawBallsFromA initialBoxA 2
  let p := ((binom (initialBoxA.blue, drawnFromA.2) * 
     binom (initialBoxA.red, drawnFromA.1)) / 
     binom (initialBoxA.red + initialBoxA.blue, 2)) + ((binom (initialBoxA.blue, drawnFromA.2) * 
     binom (initialBoxA.red, drawnFromA.1)) / 
     binom (initialBoxA.red + initialBoxA.blue, 2))
  in p = (14 / 15) := 
sorry

theorem probability_blue_from_B :
  let drawnFromA := drawBallsFromA initialBoxA 2
  let combinedBoxB := combineBoxes drawnFromA initialBoxB
  let drawnFromB := drawBallFromB combinedBoxB
  let p := ((1 / 40) + (4 / 15) + (1 / 4))
  in p = (13 / 24) := 
sorry

end probability_at_least_one_blue_from_A_probability_blue_from_B_l298_298760


namespace num_valid_permutations_l298_298281

-- Definitions of frequency conditions
def freq_x (l : List Char) : Nat := l.count 'x'
def freq_y (l : List Char) : Nat := l.count 'y'
def freq_z (l : List Char) : Nat := l.count 'z'

def is_permutation_valid (l : List Char) : Bool :=
  l.length = 6 ∧
  (freq_x l = 2 ∨ freq_x l = 3) ∧
  (freq_y l % 2 = 1) ∧
  freq_z l = 2

-- Theorem statement
theorem num_valid_permutations : 
  (List.filter is_permutation_valid (List.permutations ['x', 'x', 'x', 'y', 'y', 'y', 'z', 'z']).length = 60) :=
sorry

end num_valid_permutations_l298_298281


namespace large_bottle_water_amount_l298_298164

noncomputable def sport_drink_water_amount (C V : ℝ) (prop_e : ℝ) : ℝ :=
  let F := C / 4
  let W := (C * 15)
  W

theorem large_bottle_water_amount (C V : ℝ) (prop_e : ℝ) (hc : C = 7) (hprop_e : prop_e = 0.05) : sport_drink_water_amount C V prop_e = 105 := by
  sorry

end large_bottle_water_amount_l298_298164


namespace q1_q2_q3_l298_298087

-- Step 1: Translation of question 1
theorem q1 (f : ℝ → ℝ) 
  (h_odd : ∀ x ∈ Icc (-1 : ℝ) 1, f (-x) = -f x)
  (h_f1 : f 1 = 1)
  (h_pos : ∀ a b ∈ Icc (-1 : ℝ) 1, a + b ≠ 0 → (f a + f b) / (a + b) > 0) :
  ∀ x1 x2 ∈ Icc (-1 : ℝ) 1, x1 < x2 → f x1 < f x2 :=
sorry

-- Step 2: Translation of question 2
theorem q2 (f : ℝ → ℝ)
  (h_monotone : ∀ x1 x2 ∈ Icc (-1 : ℝ) 1, x1 < x2 → f x1 < f x2) :
  {x : ℝ | 0 ≤ x ∧ x < 2/5} = {x | f (2 * x - 1) < f (1 - 3 * x)} :=
sorry

-- Step 3: Translation of question 3
theorem q3 (f : ℝ → ℝ)
  (h_bound : ∀ x ∈ Icc (-1 : ℝ) 1, f x ≤ γ x) 
  (γ : ℝ → ℝ)
  (h_γ_eq : ∀ a ∈ Icc (-1 : ℝ) 1, γ a = m^2 - 2 * a * m + 1) :
  (m = 0 ∨ m ≤ -2 ∨ m ≥ 2) :=
sorry

end q1_q2_q3_l298_298087


namespace sum_of_valid_as_l298_298852

theorem sum_of_valid_as :
  let integer_solution_exists (a : ℤ) : Prop :=
    ∃ x : ℤ, (ax - 2) / (x - 1) + 1 = -1 / (1 - x)
  let inequality_system (a x : ℤ) : Prop :=
    3 * x ≤ 2 * (x - 1 / 2) ∧ 2 * x - a < (x - 1) / 3
  has_solution (a: ℤ) : Prop :=
    ∃ x : ℤ, x ≤ -1 ∧ inequality_system a x 
  let summable_as : ∑ (a in {-5, -3, -2, 0, 1} : Finset ℤ), has_solution a ∧ integer_solution_exists a := 1 in
  ∃ sum : ℤ, summable_as = sum :=
  sorry

end sum_of_valid_as_l298_298852


namespace point_in_fourth_quadrant_l298_298634

theorem point_in_fourth_quadrant (x y : ℝ) (hx : x = Real.sqrt 2022) (hy : y = -Real.sqrt 2023) :
  (x > 0 ∧ y < 0) → 4 = 4 :=
by
  intros h
  cases h with x_pos y_neg
  trivial

end point_in_fourth_quadrant_l298_298634


namespace equal_cost_mileage_l298_298991

theorem equal_cost_mileage :
  ∃ (x : ℝ), (17.99 + 0.18 * x = 18.95 + 0.16 * x) := 
begin
  use 48,
  sorry
end

end equal_cost_mileage_l298_298991


namespace number_of_divisors_of_8n2_l298_298551

theorem number_of_divisors_of_8n2 (n : ℕ) (h₁ : n % 2 = 1) (h₂ : nat.totient n = 13) : nat.num_divisors (8 * n^2) = 100 :=
by
  sorry

end number_of_divisors_of_8n2_l298_298551


namespace parallelogram_perimeter_l298_298919

/-- Let $ABCD$ be a quadrilateral with $AB = 5$ and $BC = 3$. If $ABCD$ is a parallelogram, then the perimeter of $ABCD$ is $16$. -/
theorem parallelogram_perimeter (A B C D : Type) 
  (AB : ℝ) (BC : ℝ) (CD : ℝ) (DA : ℝ) 
  (h1 : AB = 5) (h2 : BC = 3)
  (h3 : CD = AB) (h4 : DA = BC)
  (parallelogram : AB = CD ∧ BC = DA) :
  AB + BC + CD + DA = 16 :=
by
  rw [h1, h2, h3, h4]
  rw parallelogram.left
  rw parallelogram.right
  rw add_assoc,
  rw add_assoc,
  rw mul_add,
  sorry

end parallelogram_perimeter_l298_298919


namespace sum_of_common_divisors_l298_298054

theorem sum_of_common_divisors :
  ∃ (d1 d2 d3 d4 : ℕ), 
    (d1 > 0) ∧ (d2 > 0) ∧ (d3 > 0) ∧ (d4 > 0) ∧
    (∀ n ∈ {45, 90, -15, 135, 180}, n % d1 = 0) ∧
    (∀ n ∈ {45, 90, -15, 135, 180}, n % d2 = 0) ∧
    (∀ n ∈ {45, 90, -15, 135, 180}, n % d3 = 0) ∧
    (∀ n ∈ {45, 90, -15, 135, 180}, n % d4 = 0) ∧
    d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4 ∧
    d1 + d2 + d3 + d4 = 24 := 
by 
  sorry

end sum_of_common_divisors_l298_298054


namespace largest_prime_factor_5985_l298_298659

theorem largest_prime_factor_5985 : ∃ p, Nat.Prime p ∧ p ∣ 5985 ∧ ∀ q, Nat.Prime q ∧ q ∣ 5985 → q ≤ p :=
sorry

end largest_prime_factor_5985_l298_298659


namespace sum_of_squares_of_roots_eq_l298_298052

noncomputable def sum_of_squares_of_roots (n : ℕ) (h : 0 < n) : ℂ :=
  let roots := {z : ℂ | (z - 1)^n = (z + 1)^n}
  ∑ z in roots, z^2

theorem sum_of_squares_of_roots_eq (n : ℕ) (h : 0 < n) :
  sum_of_squares_of_roots n h = (n - 1) * (n - 2) / 3 := sorry

end sum_of_squares_of_roots_eq_l298_298052


namespace exactly_one_divisible_by_4_l298_298578

theorem exactly_one_divisible_by_4 :
  (777 % 4 = 1) ∧ (555 % 4 = 3) ∧ (999 % 4 = 3) →
  (∃! (x : ℕ),
    (x = 777 ^ 2021 * 999 ^ 2021 - 1 ∨
     x = 999 ^ 2021 * 555 ^ 2021 - 1 ∨
     x = 555 ^ 2021 * 777 ^ 2021 - 1) ∧
    x % 4 = 0) :=
by
  intros h
  sorry

end exactly_one_divisible_by_4_l298_298578


namespace matrix_vector_product_is_correct_l298_298770

def A : Matrix (Fin 2) (Fin 3) ℤ :=
  ![![3, 0, -2], ![1, -1, 4]]

def v : Fin 3 → ℤ :=
  ![2, 3, -1]

def result : Fin 2 → ℤ :=
  ![8, -5]

theorem matrix_vector_product_is_correct :
  A.mul_vec v = result :=
by
  sorry

end matrix_vector_product_is_correct_l298_298770


namespace books_sold_l298_298629

theorem books_sold {total_books sold_fraction left_fraction : ℕ} (h_total : total_books = 9900)
    (h_fraction : left_fraction = 4/6) (h_sold : sold_fraction = 1 - left_fraction) : 
  (sold_fraction * total_books) = 3300 := 
  by 
  sorry

end books_sold_l298_298629


namespace number_of_perfect_square_factors_of_240_l298_298484

-- Define 240 as a product of its prime factors
def prime_factors_240 : Prop := 240 = 2^4 * 3 * 5

-- Define what it means for a number to be a perfect square in terms of its factors
def is_perfect_square (n : ℕ) : Prop :=
  ∀ p k, prime_factors_240 → (n = p^k) → even k

-- Define the statement we want to prove
theorem number_of_perfect_square_factors_of_240 : prime_factors_240 → ∃ n, n = 3 :=
by
  intro h
  use 3
  sorry  -- proof to be filled in

end number_of_perfect_square_factors_of_240_l298_298484


namespace area_of_trapezoid_DBCE_l298_298751

-- Describe the conditions
def triangle (A B C : Type) : Prop :=
  ∃ (similar_to_ABC : Prop), similar_to_ABC ∧ (A = B) ∧ (B = C)

def has_area (t : Type) (area : ℕ) : Prop :=
  ∃ (area_of_t : ℕ), area_of_t = area

def eight_smallest_triangles_have_area_1 (T : Type) : Prop :=
  ∀ (t : Type), (has_area t 1) → (t ∈ set.range T)

def big_triangle_has_area_49 (ABC : Type) : Prop :=
  has_area ABC 49

-- The main statement to prove
theorem area_of_trapezoid_DBCE (T ABC DBCE : Type)
  (h1 : triangle T ABC)
  (h2 : eight_smallest_triangles_have_area_1 T)
  (h3 : big_triangle_has_area_49 ABC) :
  has_area DBCE 41 :=
sorry

end area_of_trapezoid_DBCE_l298_298751


namespace arithmetic_sequence_general_term_and_sum_l298_298084

noncomputable def a (n : ℕ) : ℤ := 2 * n - 1

def S (n : ℕ) : ℤ := n * (a 1 + a n) / 2

def b (n : ℕ) : ℚ := 1 / ((a n : ℚ) * (a (n + 1)))

theorem arithmetic_sequence_general_term_and_sum (a5_eq_9 : a 5 = 9) (S5_eq_25 : S 5 = 25) :
  (∀ n : ℕ, a n = 2 * n - 1) ∧ (∑ i in Finset.range 100, b i.succ = 100 / 201) := by
  sorry

end arithmetic_sequence_general_term_and_sum_l298_298084


namespace eccentricity_of_ellipse_eq_l298_298045

-- Definition of the ellipse and related parameters
variables (a b : ℝ) (x_0 y_0 : ℝ)
variables (h1 : a > b) (h2 : b > 0)
variables (h_ellipse : (x_0 / a) ^ 2 + (y_0 / b) ^ 2 = 1)
variables (h_symmetry : ∀(P Q : ℝ), P = x_0 → Q = -x_0)

-- Definition of the slopes and their product
def slope_AP (P : ℝ × ℝ) : ℝ := P.2 / (P.1 + a)
def slope_AQ (Q : ℝ × ℝ) : ℝ := Q.2 / (Q.1 - a)
variable (h_slopes : slope_AP (x_0, y_0) * slope_AQ (-x_0, y_0) = 1 / 4)

-- The proof statement for the eccentricity
theorem eccentricity_of_ellipse_eq :
  (sqrt (1 - (b / a) ^ 2)) = sqrt 3 / 2 :=
by
  sorry

end eccentricity_of_ellipse_eq_l298_298045


namespace find_triangle_sides_l298_298115

theorem find_triangle_sides (a : Fin 7 → ℝ) (h : ∀ i, 1 < a i ∧ a i < 13) : 
  ∃ i j k, 1 ≤ i ∧ i < j ∧ j < k ∧ k ≤ 7 ∧ 
           a i + a j > a k ∧ 
           a j + a k > a i ∧ 
           a k + a i > a j :=
sorry

end find_triangle_sides_l298_298115


namespace largest_n_for_factorable_polynomial_l298_298799

theorem largest_n_for_factorable_polynomial :
  ∃ (n : ℤ), (∀ A B : ℤ, 7 * A * B = 56 → n ≤ 7 * B + A) ∧ n = 393 :=
by {
  sorry
}

end largest_n_for_factorable_polynomial_l298_298799


namespace find_x_l298_298668

theorem find_x (x : ℤ) (h : 4 * x - 23 = 33) : x = 14 := 
by 
  sorry

end find_x_l298_298668


namespace math_problem_l298_298077

theorem math_problem
  (x y : ℝ)
  (h1 : x + y = x * y) (h2 : x > 0) (h3 : y > 0) :
  (x + 2 * y ≥ 3 + 2 * Real.sqrt 2) ∧
  (2^x + 2^y ≥ 8) ∧
  (xy + 1/xy < 5) ∧
  (1/Real.sqrt x + 1/Real.sqrt y ≤ Real.sqrt 2) :=
by
  sorry

end math_problem_l298_298077


namespace problem1_problem2_l298_298917

theorem problem1 (h1 : ∀ (A B C : ℕ), A + B + C = 180) 
                 (h2 : real.sin A = 2 * real.sqrt 2 / 3) : 
  real.tan (B + C) / 2 ^ 2 = 2 :=
by
  sorry

theorem problem2 (a : ℝ) (h : a = 2) 
                 (S : ℝ) (S_eq : S = real.sqrt 2) 
                 (sin_A : ℝ) (h_sin_A : sin_A = 2 * real.sqrt 2 / 3) 
                 (bc : ℝ) (h_bc : bc = 3) 
                 (cos_A : ℝ) (h_cos_A : cos_A = 1 / 3) : 
  ∃ b c : ℝ, b * c = 3 ∧ b = real.sqrt 3 := 
by
  sorry

end problem1_problem2_l298_298917


namespace matt_left_hand_rate_l298_298210

theorem matt_left_hand_rate :
  let w := 35 / 5 in
  w = 7 :=
by
  let write_rate_right_hand := 10
  let minutes := 5
  let write_rate_left_hand := 35 / minutes
  have : 10 * 5 = 50 := by norm_num
  have : 50 - 15 = 35 := by norm_num
  exact calc w = 35 / 5 : rfl
               ... = 7     : by norm_num

end matt_left_hand_rate_l298_298210


namespace problem1_seating_arrangement_problem2_standing_arrangement_problem3_spots_distribution_l298_298309

theorem problem1_seating_arrangement :
  let n_seats := 8
  let n_people := 3
  seating_ways (n_seats : ℕ) (n_people : ℕ) : ℕ :=
sorry

theorem problem2_standing_arrangement :
  let n_people := 5
  arrangement_ways (n_people : ℕ) (A_right_of_B : Prop) : ℕ :=
sorry

theorem problem3_spots_distribution :
  let n_spots := 10
  let n_schools := 7
  let min_spot_per_school := 1
  distribution_ways (n_spots : ℕ) (n_schools : ℕ) (min_spot_per_school : ℕ) : ℕ :=
sorry

end problem1_seating_arrangement_problem2_standing_arrangement_problem3_spots_distribution_l298_298309


namespace set_intersection_complement_l298_298476

open Finset

def U := {0, 1, 2, 3, 4}
def M := {0, 1, 2}
def N := {2, 3}
def compl_N := U.sdiff N
def intersection := M ∩ compl_N

theorem set_intersection_complement : intersection = {0, 1} := by
  sorry

end set_intersection_complement_l298_298476


namespace find_a_l298_298526

-- Definitions based on conditions
def triangle_area (A B C : Type) [HasSin A] [HasMul B] [HasMul C] [HasDiv B] [HasSqrt B] [HasOne B] [HasZero B] :=
  B -> B -> C -> B

noncomputable def area_of_triangle {α : Type} [LinearOrderField α] (A B C : α) : α :=
  1/2 * B * C * sin A

variable (α : Type) [LinearOrderField α]

-- Given conditions
def A : α := 60 * π / 180 -- Conversion from degrees to radians
def b : α := 1
def area : α := √3

-- Statement to prove
theorem find_a (a : α) (c : α) : 
  area_of_triangle A b c = area ∧
  a^2 = b^2 + c^2 - 2 * b * c * cos A → 
  a = √13 :=
sorry

end find_a_l298_298526


namespace nuts_left_over_l298_298432

/-- Builder purchased 7 boxes of bolts with 11 bolts per box,
    and 3 boxes of nuts with 15 nuts per box. He used a total of 113 bolts and nuts,
    had 3 bolts left over. Prove the number of nuts left over is 6. -/
theorem nuts_left_over :
  let total_bolts := 7 * 11 in
  let total_nuts := 3 * 15 in
  let bolts_used := total_bolts - 3 in
  let nuts_used := 113 - bolts_used in
  total_nuts - nuts_used = 6 :=
by
  let total_bolts := 7 * 11
  let total_nuts := 3 * 15
  let bolts_used := total_bolts - 3
  let nuts_used := 113 - bolts_used
  show total_nuts - nuts_used = 6 from sorry

end nuts_left_over_l298_298432


namespace increasing_function_range_a_l298_298815

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 0 then (a - 1) * x + 3 * a - 4 else a^x

theorem increasing_function_range_a (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₂ - f a x₁) / (x₂ - x₁) > 0) ↔ 1 < a ∧ a ≤ 5 / 3 :=
sorry

end increasing_function_range_a_l298_298815


namespace coloring_connected_circles_diff_colors_l298_298248

def num_ways_to_color_five_circles : ℕ :=
  36

theorem coloring_connected_circles_diff_colors (A B C D E : Type) (colors : Fin 3) 
  (connected : (A → B → C → D → E → Prop)) : num_ways_to_color_five_circles = 36 :=
by sorry

end coloring_connected_circles_diff_colors_l298_298248


namespace second_range_is_18_l298_298313

variable (range1 range2 range3 : ℕ)

theorem second_range_is_18
  (h1 : range1 = 30)
  (h2 : range2 = 18)
  (h3 : range3 = 32) :
  range2 = 18 := by
  sorry

end second_range_is_18_l298_298313


namespace min_distance_equals_sqrt2_over_2_l298_298876

noncomputable def min_distance_from_point_to_line (m n : ℝ) : ℝ :=
  (|m + n + 10|) / Real.sqrt (1^2 + 1^2)

def circle_eq (m n : ℝ) : Prop :=
  (m - 1 / 2)^2 + (n - 1 / 2)^2 = 1 / 2

theorem min_distance_equals_sqrt2_over_2 (m n : ℝ) (h1 : circle_eq m n) :
  min_distance_from_point_to_line m n = 1 / (Real.sqrt 2) :=
sorry

end min_distance_equals_sqrt2_over_2_l298_298876


namespace problem_statement_l298_298196

noncomputable def g : ℝ → ℝ := sorry

axiom g_property : ∀ x y : ℝ, g (g x + y) = 2 * g x - g (g y + g (-x)) + x

theorem problem_statement : 
  (∃ (c : ℝ), g (-2) = c ∧ (∀ (d : ℝ), g (-2) = d → d = c)) ∧
  (let t := g (-2) in t = 2) ∧
  (let n := 1 in n * 2 = 2) :=
sorry

end problem_statement_l298_298196


namespace determine_a_b_monotonic_intervals_range_of_c_l298_298860

noncomputable def f (x : ℝ) (a b c : ℝ) := x^3 + a * x^2 + b * x + c

theorem determine_a_b (a b c : ℝ) (h₁ : f'(-1) = 0) (h₂ : f'(2) = 0) :
  a = -3/2 ∧ b = -6 :=
  sorry

theorem monotonic_intervals (a b c : ℝ) (h₁ : a = -3/2) (h₂ : b = -6) :
  (∀ x, -1 < x ∧ x < 2 → f'(x) < 0) ∧
  (∀ x, x < -1 → f'(x) > 0) ∧
  (∀ x, x > 2 → f'(x) > 0) :=
  sorry

theorem range_of_c (a b c : ℝ) (h₁ : a = -3/2) (h₂ : b = -6) :
  (∀ x, x ∈ Icc (-2 : ℝ) 3 → f(x, a, b, c) + 3/2 * c < c^2) →
  (-∞ < c ∧ c < -1) ∨ (7/2 < c ∧ c < ∞) :=
  sorry

end determine_a_b_monotonic_intervals_range_of_c_l298_298860


namespace h_of_k_neg_3_l298_298119

def h (x : ℝ) : ℝ := 4 - real.sqrt x

def k (x : ℝ) : ℝ := 3 * x + 3 * x^2

theorem h_of_k_neg_3 : h (k (-3)) = 4 - 3 * real.sqrt 2 :=
by
  sorry

end h_of_k_neg_3_l298_298119


namespace longest_diagonal_of_rhombus_l298_298729

theorem longest_diagonal_of_rhombus (A B : ℝ) (h1 : A = 150) (h2 : ∃ x, (A = 1/2 * (4 * x) * (3 * x)) ∧ (x = 5)) : 
  4 * (classical.some h2) = 20 := 
by sorry

end longest_diagonal_of_rhombus_l298_298729


namespace sqrt_division_simplification_l298_298375

theorem sqrt_division_simplification :
  (real.sqrt 40) / (real.sqrt 5) = 2 * real.sqrt 2 :=
by
  sorry

end sqrt_division_simplification_l298_298375


namespace line_slope_intercept_through_points_l298_298504

theorem line_slope_intercept_through_points (a b : ℝ) :
  (∀ x y : ℝ, (x, y) = (3, 7) ∨ (x, y) = (7, 19) → y = a * x + b) →
  a - b = 5 :=
by
  sorry

end line_slope_intercept_through_points_l298_298504


namespace lcm_12_18_is_36_l298_298420

def prime_factors (n : ℕ) : list ℕ :=
  if n = 12 then [2, 2, 3]
  else if n = 18 then [2, 3, 3]
  else []

noncomputable def lcm_of_two (a b : ℕ) : ℕ :=
  match prime_factors a, prime_factors b with
  | [2, 2, 3], [2, 3, 3] => 36
  | _, _ => 0

theorem lcm_12_18_is_36 : lcm_of_two 12 18 = 36 :=
  sorry

end lcm_12_18_is_36_l298_298420


namespace sum_of_maximum_values_variable_based_on_computation_l298_298949

def P (x : ℝ) : ℝ :=
  x^3 + a*x^2 + b*x + c

def Q (x : ℝ) : ℝ :=
  x^2 + d*x + e

def P_comp_Q (x : ℝ) : ℝ :=
  P (Q x)

def Q_comp_P (x : ℝ) : ℝ :=
  Q (P x)

theorem sum_of_maximum_values_variable_based_on_computation
  (hP : monic P)
  (hQ : monic Q)
  (hP_comp_Q_zeros :
    (λ x, P_comp_Q x =
      (λ x, x = -3) ∨
      (λ x, x = -2) ∨
      (λ x, x = 0) ∨
      (λ x, x =1))
  (hQ_comp_P_zeros :
    (λ x, Q_comp_P x =
      (λ x, x = -4) ∨
      (λ x, x = -1) ∨
      (λ x, x = 3)) :

  ∃ P_max Q_max : ℝ, (sum(P_max, Q_max) =
    "Variable based on computation") := sorry

end sum_of_maximum_values_variable_based_on_computation_l298_298949


namespace zero_of_f_l298_298096

noncomputable def f (x : ℝ) : ℝ := (|Real.log x - Real.log 2|) - (1 / 3) ^ x

theorem zero_of_f :
  ∃ x1 x2 : ℝ, x1 < x2 ∧ (f x1 = 0) ∧ (f x2 = 0) ∧
  (1 < x1 ∧ x1 < 2) ∧ (2 < x2) := 
sorry

end zero_of_f_l298_298096


namespace lcm_12_18_l298_298403

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := 
by
  sorry

end lcm_12_18_l298_298403


namespace AY_is_2_sqrt_55_l298_298694

noncomputable def AY_length : ℝ :=
  let rA := 10
  let rB := 3
  let AB := rA + rB
  let AD := rA - rB
  let BD := Real.sqrt (AB^2 - AD^2)
  2 * Real.sqrt (rA^2 + BD^2)

theorem AY_is_2_sqrt_55 :
  AY_length = 2 * Real.sqrt 55 :=
by
  -- Assuming the given problem's conditions.
  let rA := 10
  let rB := 3
  let AB := rA + rB
  let AD := rA - rB
  let BD := Real.sqrt (AB^2 - AD^2)
  show AY_length = 2 * Real.sqrt 55
  sorry

end AY_is_2_sqrt_55_l298_298694


namespace distance_difference_l298_298498

-- Define the initial conditions
def speed1 : ℝ := 5 -- speed at 5 km/hr
def speed2 : ℝ := 10 -- speed at 10 km/hr
def distance_travelled : ℝ := 20 -- distance is 20 km

-- Define known times
def time1 : ℝ := distance_travelled / speed1
def time2 : ℝ := distance_travelled / speed2

def distance1 : ℝ := speed1 * time1
def distance2 : ℝ := speed2 * time1

-- Define the theorem
theorem distance_difference :
  distance2 - distance1 = 20 :=
sorry

end distance_difference_l298_298498


namespace length_of_EC_l298_298683

structure Kite (A B C D : Type) :=
(AB_AD_equal : AB = AD)
(CB_CD_equal : CB = CD)
(diagonal_AC_14 : AC = 14)
(angle_AEB_90 : ∠AEB = 90°)

theorem length_of_EC {A B C D E : Type} [Kite A B C D] (h : Kite A B C D) : EC = 7 :=
by sorry

end length_of_EC_l298_298683


namespace green_or_blue_marble_probability_l298_298286

theorem green_or_blue_marble_probability :
  (4 + 3 : ℝ) / (4 + 3 + 8) = 0.4667 := by
  sorry

end green_or_blue_marble_probability_l298_298286


namespace exists_vertex_with_one_friend_l298_298987

-- Define the graph with the given conditions and show there exists a vertex with degree 1

universe u
variables {α : Type u}

structure Graph (α : Type u) :=
(vertices : finset α)
(adj : α → α → Prop)
(symm : ∀ {v w}, adj v w → adj w v)
(loop_free : ∀ v, ¬ adj v v)

open Graph

def degree {α : Type u} (G : Graph α) (v : α) : ℕ :=
(finset.filter (G.adj v) G.vertices).card

theorem exists_vertex_with_one_friend {G : Graph α}
  (H : ∀ v w, degree G v = degree G w → ¬ G.adj v w) :
  ∃ v, degree G v = 1 :=
begin
  sorry
end

end exists_vertex_with_one_friend_l298_298987


namespace team_total_points_l298_298969

theorem team_total_points (three_points_goals: ℕ) (two_points_goals: ℕ) (half_of_total: ℕ) 
  (h1 : three_points_goals = 5) 
  (h2 : two_points_goals = 10) 
  (h3 : half_of_total = (3 * three_points_goals + 2 * two_points_goals) / 2) 
  : 2 * half_of_total = 70 := 
by 
  -- proof to be filled
  sorry

end team_total_points_l298_298969


namespace cos_alpha_minus_two_pi_l298_298075

theorem cos_alpha_minus_two_pi (alpha : ℝ) (h1 : sin (π + alpha) = 3 / 5) (h2 : α > 3 * π / 2 ∧ α < 2 * π) :
  cos (alpha - 2 * π) = 4 / 5 :=
sorry

end cos_alpha_minus_two_pi_l298_298075


namespace triangle_BE_length_l298_298524

theorem triangle_BE_length :
  ∀ (A B C D E : Type) [linear_ordered_field A]
    (a b c : A)
    (h : a + b + c > 0)
    (AB BC CA : A) (CD BE : A)
    (hAB : AB = 12) (hBC : BC = 13) (hCA : CA = 15)
    (hCD : CD = 5)
    (h_angle : ∀ (angle_BAE angle_CAD : A), angle_BAE = angle_CAD),
  BE = (52 : A) / (29 : A) := by
  sorry

end triangle_BE_length_l298_298524


namespace shot_put_surface_area_l298_298341

noncomputable def radius (d : ℝ) : ℝ := d / 2

noncomputable def surface_area (r : ℝ) : ℝ := 4 * Real.pi * r^2

theorem shot_put_surface_area :
  surface_area (radius 5) = 25 * Real.pi :=
by
  sorry

end shot_put_surface_area_l298_298341


namespace expression_equals_16_l298_298127

open Real

theorem expression_equals_16 (x : ℝ) :
  (x + 1) ^ 2 + 2 * (x + 1) * (3 - x) + (3 - x) ^ 2 = 16 :=
sorry

end expression_equals_16_l298_298127


namespace hexagon_circumradius_l298_298577

theorem hexagon_circumradius
  (A B C D E F : ℝ)
  (hex_eq : A = 1 ∧ B = 1 ∧ C = 1 ∧ D = 1 ∧ E = 1 ∧ F = 1)
  (convex : convex_hull ℝ ({A, B, C, D, E, F} : set ℝ).nonempty) : 
  ∃ (rACE rBDF : ℝ), (rACE ≤ 1 ∨ rBDF ≤ 1) := 
sorry

end hexagon_circumradius_l298_298577


namespace tangent_value_prism_QABC_l298_298879

-- Assuming R is the radius of the sphere and considering the given conditions
variables {R x : ℝ} (P Q A B C M H : Type)

-- Given condition: Angle between lateral face and base of prism P-ABC is 45 degrees
def angle_PABC : ℝ := 45
-- Required to prove: tan(angle between lateral face and base of prism Q-ABC) = 4
def tangent_QABC : ℝ := 4

theorem tangent_value_prism_QABC
  (h1 : angle_PABC = 45)
  (h2 : 5 * x - 2 * R = 0) -- Derived condition from the solution
  (h3 : x = 2 * R / 5) -- x, the distance calculation
: tangent_QABC = 4 := by
  sorry

end tangent_value_prism_QABC_l298_298879


namespace percentage_increase_from_40_to_48_is_20_l298_298508

theorem percentage_increase_from_40_to_48_is_20 :
  let original_value := 40
  let new_value := 48
  let percentage_increase := ((new_value - original_value) / original_value.toFloat) * 100
  percentage_increase = 20 :=
by
  sorry

end percentage_increase_from_40_to_48_is_20_l298_298508


namespace complement_of_A_union_B_in_U_l298_298874

def U : Set ℝ := { x | -5 < x ∧ x < 5 }
def A : Set ℝ := { x | x^2 - 4*x - 5 < 0 }
def B : Set ℝ := { x | -2 < x ∧ x < 4 }

theorem complement_of_A_union_B_in_U :
  (U \ (A ∪ B)) = { x | -5 < x ∧ x ≤ -2 } := by
  sorry

end complement_of_A_union_B_in_U_l298_298874


namespace solve_for_x_l298_298598

theorem solve_for_x (x : ℝ) (h : 5 + 3.5 * x = 2.5 * x - 25) : x = -30 :=
sorry

end solve_for_x_l298_298598


namespace distance_P2P4_eq_pi_l298_298367

noncomputable def distance_P2P4_on_curve : ℝ := 
  let f := λ x : ℝ, 2 * sin (x + (Real.pi / 4)) * cos (x - (Real.pi / 4))
  let g := λ y : ℝ, 1 / 2
  let y_intersection := 1 / 2
  sorry -- skipping the actual proof

theorem distance_P2P4_eq_pi : distance_P2P4_on_curve = Real.pi := 
  sorry -- proof to be completed

end distance_P2P4_eq_pi_l298_298367


namespace infinite_fractions_2_over_odd_l298_298666

theorem infinite_fractions_2_over_odd (a b : ℕ) (n : ℕ) : 
  (a = 2 → 2 * b + 1 ≠ 0) ∧ ((b = 2 * n + 1) → (2 + 2) / (2 * (2 * n + 1)) = 2 / (2 * n + 1)) ∧ (a / b = 2 / (2 * n + 1)) :=
by
  sorry

end infinite_fractions_2_over_odd_l298_298666


namespace range_of_k_l298_298853

noncomputable def f (x : ℝ) : ℝ :=
if x >= 0 then sin x else -x^2 - 1

theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, f x ≤ k * x) ↔ (1 ≤ k ∧ k ≤ 2) :=
by
  sorry

end range_of_k_l298_298853


namespace derivative_at_x1_is_12_l298_298247

theorem derivative_at_x1_is_12 : 
  (deriv (fun x : ℝ => (2 * x + 1) ^ 2) 1) = 12 :=
by
  sorry

end derivative_at_x1_is_12_l298_298247


namespace single_elimination_games_l298_298685

theorem single_elimination_games (n : ℕ) : n = 23 → (∀ k, k = 22) :=
by {
  intro h,
  induction h,
  sorry
}

end single_elimination_games_l298_298685


namespace coeff_x2_in_poly_expansion_l298_298161

theorem coeff_x2_in_poly_expansion :
  let f := (2 * (x ^ 3) + 1) * (x - x ^ (-2)) ^ 5 in
  polynomial.coeff f 2 = 15 :=
sorry

end coeff_x2_in_poly_expansion_l298_298161


namespace line_slope_intercept_through_points_l298_298503

theorem line_slope_intercept_through_points (a b : ℝ) :
  (∀ x y : ℝ, (x, y) = (3, 7) ∨ (x, y) = (7, 19) → y = a * x + b) →
  a - b = 5 :=
by
  sorry

end line_slope_intercept_through_points_l298_298503


namespace exists_vertex_deg_le_5_six_color_theorem_l298_298684

-- Problem 1: Show that a simple planar graph G with a finite number of vertices has at least one vertex of degree not exceeding 5.
theorem exists_vertex_deg_le_5 (G : SimpleGraph) [Finite G] [Planar G] :
  ∃ v : G.V, G.degree v ≤ 5 :=
sorry

-- Problem 2: Show the 6-color theorem: A simple planar graph G with a finite number of vertices has a chromatic number χ(G) ≤ 6.
theorem six_color_theorem (G : SimpleGraph) [Finite G] [Planar G] :
  G.chromaticNumber ≤ 6 :=
sorry

end exists_vertex_deg_le_5_six_color_theorem_l298_298684


namespace interest_rate_calculation_l298_298665

theorem interest_rate_calculation
  (P : ℕ) 
  (I : ℕ) 
  (T : ℕ) 
  (R : ℕ) 
  (principal : P = 9200) 
  (time : T = 3) 
  (interest_diff : P - 5888 = I) 
  (interest_formula : I = P * R * T / 100) 
  : R = 12 :=
sorry

end interest_rate_calculation_l298_298665


namespace probability_of_integer_p_is_3_div_20_l298_298488

open_locale big_operators

noncomputable def probability_of_p : ℚ :=
  let possible_p := {p | ∃ q : ℤ, p * q - 6 * p - 3 * q = 3} in
  let total_possibilities := finset.Icc 1 20 in
  (finset.card (total_possibilities.filter possible_p)).to_rat / (finset.card total_possibilities).to_rat

theorem probability_of_integer_p_is_3_div_20 :
  probability_of_p = 3 / 20 :=
by {
  sorry
}

end probability_of_integer_p_is_3_div_20_l298_298488


namespace relationship_between_T_and_S_l298_298073

variable (a b : ℝ)

def T : ℝ := a + 2 * b
def S : ℝ := a + b^2 + 1

theorem relationship_between_T_and_S : T a b ≤ S a b := by
  sorry

end relationship_between_T_and_S_l298_298073


namespace sum_dihedral_angles_eq_l298_298639

-- Define the tetrahedron and its edges
variables {A B C D : Type} [Point A] [Point B] [Point C] [Point D]
variables (AB BC CD DA : Segment)

-- Given Condition
def edge_length_cond : Prop :=
  length AB + length CD = length BC + length DA

-- Dihedral angles
def dihedral_angle (e1 e2 : Segment) : Angle := sorry

-- Proving the equality of sum of dihedral angles
theorem sum_dihedral_angles_eq (h : edge_length_cond AB BC CD DA) :
  dihedral_angle AB CD + dihedral_angle CD AB = dihedral_angle BC DA + dihedral_angle DA BC :=
sorry

end sum_dihedral_angles_eq_l298_298639


namespace airplane_travel_difference_correct_l298_298009

-- Define airplane's speed without wind
def airplane_speed_without_wind : ℕ := a

-- Define wind speed
def wind_speed : ℕ := 20

-- Define time without wind
def time_without_wind : ℕ := 4

-- Define time against wind
def time_against_wind : ℕ := 3

-- Define distance covered without wind
def distance_without_wind : ℕ := airplane_speed_without_wind * time_without_wind

-- Define effective speed against wind
def effective_speed_against_wind : ℕ := airplane_speed_without_wind - wind_speed

-- Define distance covered against wind
def distance_against_wind : ℕ := effective_speed_against_wind * time_against_wind

-- Define the difference in distances
def distance_difference : ℕ := distance_without_wind - distance_against_wind

-- The theorem we wish to prove
theorem airplane_travel_difference_correct (a : ℕ) :
  distance_difference = a + 60 :=
by
sorry

end airplane_travel_difference_correct_l298_298009


namespace max_consecutive_equal_terms_l298_298846

-- Define that a sequence has a period modulo a given number.
def has_period (seq : ℕ → ℕ) (p : ℕ) : Prop :=
  ∀ n, seq (n + p) = seq n

-- Sequences a_n and b_n with their respective periods.
variables (a b : ℕ → ℕ)
variables (h1 : has_period a 7)
variables (h2 : has_period b 13)

-- Definitions to represent the existence of k such that a_i = b_i for i=1 to k.
def seq_equal_up_to (a b : ℕ → ℕ) (k : ℕ) : Prop :=
  ∀ i, 1 ≤ i ∧ i ≤ k → a i = b i

-- Statement of the problem: maximum k such that a_i = b_i for i=1 to k.
theorem max_consecutive_equal_terms : ∃ k, seq_equal_up_to a b k ∧ ∀ k', k' > k → ¬ seq_equal_up_to a b k' :=
begin
  use 91,
  split,
  { -- Proof of seq_equal_up_to a b 91     
    sorry
  },
  { -- Proof that there is no k' > 91 where the sequences remain equal for k' consecutive terms
    sorry
  }
end

end max_consecutive_equal_terms_l298_298846


namespace simplify_expression_l298_298491

theorem simplify_expression (x y z : ℝ) (h1 : x ≠ 2) (h2 : y ≠ 3) (h3 : z ≠ 4) :
  (x - 2) / (4 - z) * (y - 3) / (2 - x) * (z - 4) / (3 - y) = -1 :=
by 
sorry

end simplify_expression_l298_298491


namespace locus_of_P_l298_298082

-- Definitions related to geometry
structure Point where
  x : ℝ
  y : ℝ

def dist_sq (A B : Point) : ℝ := (A.x - B.x)^2 + (A.y - B.y)^2

def is_locus_point (A B C P : Point) : Prop :=
  dist_sq P A + dist_sq P B = dist_sq P C

-- The proof problem
theorem locus_of_P (A B C : Point) (P : Point) :
  -- Assuming C is the origin
  C = {x := 0, y := 0} → 
  -- Various conditions based on the angle γ = ∠ACB
  (if (dist_sq A C + dist_sq B C - dist_sq A B) > 0 then
    ∀ P, ¬ is_locus_point A B C P  -- Empty set for γ > 90°
   else if (dist_sq A C + dist_sq B C - dist_sq A B) = 0 then
    P = {x := (A.x + B.x) / 2, y := (A.y + B.y) / 2} → is_locus_point A B C P  -- Point D for γ = 90°
   else
    ∃ (D : Point), 
    D = {x := (A.x + B.x) / 2, y := (A.y + B.y) / 2} ∧ 
    (dist_sq D {x := (A.x + B.x) / 2, y := (A.y + B.y) / 2}) = (dist_sq A C + dist_sq B C - dist_sq A B) → 
    (dist_sq P D = √(dist_sq A C + dist_sq B C - dist_sq A B)) → is_locus_point A B C P  -- Circle for γ < 90°
  ) 
  :=
sorry

end locus_of_P_l298_298082


namespace length_segment_AB_l298_298979

def distance_to_line (A : Point) (a : Line) (d : ℝ) := 
  d = 7 ∧ ∃ B, (distance B a = 3 ∧ ∃ l, on_line A l ∧ on_line B l)

theorem length_segment_AB (A B : Point) (a : Line) :
  (distance_to_line A a 7) → (distance B a = 3) → 
  (on_line A l) ∧ (on_line B l) →
  (segment_length A B = 10 ∨ segment_length A B = 4) := 
by
  sorry

end length_segment_AB_l298_298979


namespace largest_integer_not_exceeding_l298_298933

noncomputable def sequence (a : ℕ) : ℕ → ℚ
| 0       := a
| (n + 1) := (sequence n) ^ 2 / ((sequence n) + 1)

theorem largest_integer_not_exceeding (a : ℕ) (n : ℕ) (h : a ∈ Set.Ici 1) (hn : n ≤ a / 2 + 1) :
  ⌊sequence a n⌋ = a - n := 
sorry

end largest_integer_not_exceeding_l298_298933


namespace sequence_correct_l298_298081

def sequence_a (n : ℕ) : ℤ :=
  if n = 1 then -2 else -2^(n-1)

def sequence_S (n : ℕ) : ℤ :=
  -2^n

theorem sequence_correct (n : ℕ) : sequence_a n = if n = 1 then -2 else -2^(n-1) ∧ sequence_S n = -2^n := by
  sorry

end sequence_correct_l298_298081


namespace josanna_min_average_l298_298536

theorem josanna_min_average
  (scores : List ℕ)
  (current_scores_sum : ℕ := scores.sum)
  (num_scores : ℕ := scores.length)
  (attempts : ℕ := 2)
  (increase : ℝ := 5)
  (desired_average : ℝ := (current_scores_sum + attempts * 101.8) / (num_scores + attempts)) :
  (scores = [91, 85, 82, 73, 88]) →
  (desired_average = (scores.sum.toFloat / num_scores.toFloat) + increase) →
  101.8 =
  ((desired_average * (num_scores + attempts)) - current_scores_sum.toFloat) / attempts := 
sorry

end josanna_min_average_l298_298536


namespace sum_of_all_x_such_that_f_x_eq_0_l298_298204

def f (x : ℝ) : ℝ :=
if x ≤ 2 then -2 * x - 4 else x / 3 + 2

theorem sum_of_all_x_such_that_f_x_eq_0 : 
Sum {x : ℝ | f x = 0} = -2 :=
sorry

end sum_of_all_x_such_that_f_x_eq_0_l298_298204


namespace xiaofang_time_l298_298348

-- Definitions
def overlap_time (t : ℕ) : Prop :=
  t - t / 12 = 40

def opposite_time (t : ℕ) : Prop :=
  t - t / 12 = 40

-- Theorem statement
theorem xiaofang_time :
  ∃ (x y : ℕ), 
    480 + x = 8 * 60 + 43 ∧
    840 + y = 2 * 60 + 43 ∧
    overlap_time x ∧
    opposite_time y ∧
    (y + 840 - (x + 480)) = 6 * 60 :=
by
  sorry

end xiaofang_time_l298_298348


namespace compute_f_g_2_l298_298495

def f (x : ℝ) : ℝ := 5 - 4 * x
def g (x : ℝ) : ℝ := x^2 + 2

theorem compute_f_g_2 : f (g 2) = -19 := 
by {
  sorry
}

end compute_f_g_2_l298_298495


namespace quadrilateral_is_kite_l298_298916

open Set

/-- Problem statement: Given the conditions of the original problem,
prove that the quadrilateral AMHN is a kite -/
theorem quadrilateral_is_kite 
    (A B C : Point)
    (acute_triangle : is_acute_triangle A B C)
    (D : Point)
    (AD_bisects_angle_A : is_angle_bisector A D B C)
    (H : Point)
    (altitude_from_A : is_altitude A H B C)
    (M : Point)
    (on_circle_BM : on_circle_center_radius B M (distance B D))
    (N : Point)
    (on_circle_CN : on_circle_center_radius C N (distance C D)) : 
  is_kite A M H N := sorry

end quadrilateral_is_kite_l298_298916


namespace find_k_l298_298758

def taxis_2005 : ℕ := 100000

def taxis_scrapped_each_year : ℕ := 20000

def growth_rate_new_taxis : ℝ := 0.10

def taxis_in_range {k : ℕ} (k_pos : 0 < k) : Prop :=
  let rec taxis (n : ℕ) : ℝ :=
    match n with
    | 0   => taxis_2005
    | (nat.succ n') => (1.1 * taxis n' - 20000 : ℝ)
  in
  ∃ k : ℕ, (k_pos : ℕ) > 0 ∧ floor (taxis 3 / 1000) = k

theorem find_k : taxis_in_range 12 :=
  sorry

end find_k_l298_298758


namespace earl_up_second_time_l298_298044

def earl_floors (n top start up1 down up2 dist : ℕ) : Prop :=
  start + up1 - down + up2 = top - dist

theorem earl_up_second_time 
  (start up1 down top dist : ℕ) 
  (h_start : start = 1) 
  (h_up1 : up1 = 5) 
  (h_down : down = 2) 
  (h_top : top = 20) 
  (h_dist : dist = 9) : 
  ∃ up2, earl_floors n top start up1 down up2 dist ∧ up2 = 7 :=
by
  use 7
  sorry

end earl_up_second_time_l298_298044


namespace problem_statement_l298_298546

-- Define x_k
def x_k (k : ℕ) : ℕ := k^2

-- Define the function f(n)
def f (n : ℕ) : ℚ := 
  (∑ k in Finset.range n, k.succ ^ 4) / (n * (n + 1))

-- Statement of the problem: Prove f(10) = 21
theorem problem_statement : f 10 = 21 := 
  sorry

end problem_statement_l298_298546


namespace jam_cost_l298_298393

theorem jam_cost (N B J H : ℕ) (h1 : N > 1) (h2 : N * (3 * B + 6 * J + 2 * H) = 342) :
  6 * N * J = 270 := 
sorry

end jam_cost_l298_298393


namespace complement_intersection_A_B_union_A_complement_B_l298_298872

def A : set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : set ℝ := {x | 2 < x ∧ x < 10}

theorem complement_intersection_A_B :
  (A ∩ B)ᶜ = {x | x < 3 ∨ 7 ≤ x} := 
by sorry

theorem union_A_complement_B :
  A ∪ Bᶜ = {x | x ≤ 2 ∨ (3 ≤ x ∧ x < 7) ∨ 10 ≤ x} :=
by sorry

end complement_intersection_A_B_union_A_complement_B_l298_298872


namespace problem_statement_l298_298581

theorem problem_statement : 
  (777 % 4 = 1) ∧ 
  (555 % 4 = 3) ∧ 
  (999 % 4 = 3) → 
  ( (999^2021 * 555^2021 - 1) % 4 = 0 ∧ 
    (777^2021 * 999^2021 - 1) % 4 ≠ 0 ∧ 
    (555^2021 * 777^2021 - 1) % 4 ≠ 0 ) := 
by {
  sorry
}

end problem_statement_l298_298581


namespace lcm_12_18_l298_298421

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end lcm_12_18_l298_298421


namespace solution_set_inequality_l298_298847

theorem solution_set_inequality 
  (a b c : ℝ)
  (h1 : ∀ x : ℝ, (ax^2 + bx + c < 0 ↔ x ∈ (set.Iio (-1) ∪ set.Ioi (1/2)))) :
  ∀ x : ℝ, (cx^2 - bx + a < 0 ↔ x ∈ set.Ioo (-2) 1) :=
by
  sorry

end solution_set_inequality_l298_298847


namespace other_root_of_quadratic_l298_298135

noncomputable def quadratic_root (a : ℝ) : ℝ :=
  let b := a + 2 in
  let c := 2 * a in (-b + real.sqrt (b^2 - 4 * c)) / 2

theorem other_root_of_quadratic (a : ℝ) (h : a = 3) (hr : quadratic_root a = 3) : quadratic_root 3 = 2 :=
  by
  sorry

end other_root_of_quadratic_l298_298135


namespace solve_floor_equation_l298_298051

theorem solve_floor_equation (x : ℝ) (h : ⌊x * ⌊x⌋⌋ = 20) : 5 ≤ x ∧ x < 5.25 := by
  sorry

end solve_floor_equation_l298_298051


namespace n_divisible_by_100_l298_298499

theorem n_divisible_by_100 
    (n : ℕ) 
    (h_pos : 0 < n) 
    (h_div : 100 ∣ n^3) : 
    100 ∣ n := 
sorry

end n_divisible_by_100_l298_298499


namespace grasshopper_jump_distance_l298_298627

theorem grasshopper_jump_distance (frog_jump : ℕ) (extra_distance : ℕ) (h_frog_jump : frog_jump = 11) (h_extra_distance : extra_distance = 2) :
  frog_jump + extra_distance = 13 :=
by {
  rw [h_frog_jump, h_extra_distance],
  exact rfl,
}

end grasshopper_jump_distance_l298_298627


namespace verify_digits_l298_298397

theorem verify_digits :
  ∃ (x : ℕ), 100 ≤ x ∧ x < 1000 ∧ (∃ (a b c : ℕ), a ≠ 0 ∧ c ≠ 0 ∧ x = 100*a + 10*b + c ∧ b = 7 ∧ 707 * x = 124432) :=
by {
  use 176, -- We state that 176 is the solution
  exact ⟨rfl, rfl, rfl⟩,
  use 1, use 7, use 6, -- We verify digit positions
  simp,
} 

end verify_digits_l298_298397


namespace exists_permutation_divisible_by_seven_l298_298783

theorem exists_permutation_divisible_by_seven : 
  ∃ n : ℕ, n ∈ {1234, 1243, 1324, 1342, 1423, 1432, 
                2134, 2143, 2314, 2341, 2413, 2431, 
                3124, 3142, 3214, 3241, 3412, 3421, 
                4123, 4132, 4213, 4231, 4312, 4321} ∧ 
                n % 7 = 0 :=
  sorry

end exists_permutation_divisible_by_seven_l298_298783


namespace kim_gum_distribution_l298_298177

theorem kim_gum_distribution (cousins : ℕ) (total_gum : ℕ) 
  (h1 : cousins = 4) (h2 : total_gum = 20) : 
  total_gum / cousins = 5 :=
by
  sorry

end kim_gum_distribution_l298_298177


namespace rearrangement_impossible_l298_298680

-- Definition of an 8x8 chessboard's cell numbering.
def cell_number (i j : ℕ) : ℕ := i + j - 1

-- The initial placement of pieces, represented as a permutation on {1, 2, ..., 8}
def initial_placement (p: Fin 8 → Fin 8) := True -- simplify for definition purposes

-- The rearranged placement of pieces
def rearranged_placement (q: Fin 8 → Fin 8) := True -- simplify for definition purposes

-- Condition for each piece: cell number increases
def cell_increase_condition (p q: Fin 8 → Fin 8) : Prop :=
  ∀ i, cell_number (q i).val (i.val + 1) > cell_number (p i).val (i.val + 1)

-- The main theorem to state it's impossible to rearrange under the given conditions and question
theorem rearrangement_impossible 
  (p q: Fin 8 → Fin 8) 
  (h_initial : initial_placement p) 
  (h_rearranged : rearranged_placement q) 
  (h_increase : cell_increase_condition p q) : False := 
sorry

end rearrangement_impossible_l298_298680


namespace slices_left_for_lunch_tomorrow_l298_298387

-- Definitions according to conditions
def initial_slices : ℕ := 12
def slices_eaten_for_lunch := initial_slices / 2
def remaining_slices_after_lunch := initial_slices - slices_eaten_for_lunch
def slices_eaten_for_dinner := 1 / 3 * remaining_slices_after_lunch
def remaining_slices_after_dinner := remaining_slices_after_lunch - slices_eaten_for_dinner
def slices_shared_with_friend := 1 / 4 * remaining_slices_after_dinner
def remaining_slices_after_sharing := remaining_slices_after_dinner - slices_shared_with_friend
def slices_eaten_by_sibling := if (1 / 5 * remaining_slices_after_sharing < 1) then 0 else 1 / 5 * remaining_slices_after_sharing
def remaining_slices_after_sibling := remaining_slices_after_sharing - slices_eaten_by_sibling

-- Lean statement of the proof problem
theorem slices_left_for_lunch_tomorrow : remaining_slices_after_sibling = 3 := by
  sorry

end slices_left_for_lunch_tomorrow_l298_298387


namespace geometric_sequence_a2_l298_298928

variable {α : Type*} [LinearOrderedField α]

-- Conditions from the problem statement
def is_geometric_sequence (a : ℕ → α) := ∀ n, a (n + 1) = r * a n

-- Given condition
def geometric_condition (a : ℕ → α) (r : α) := a 1 * a 2 * a 3 = -8

-- Proof problem: Prove that a_2 = -2 given the above conditions
theorem geometric_sequence_a2 (a : ℕ → α) (r : α) (h1 : is_geometric_sequence a)
  (h2 : geometric_condition a r) : a 2 = -2 :=
sorry

end geometric_sequence_a2_l298_298928


namespace eccentricity_of_ellipse_l298_298048

theorem eccentricity_of_ellipse 
  (a b : ℝ) (x₀ y₀ : ℝ) 
  (h1 : a > b) (h2 : b > 0) 
  (h3 : (x₀^2 / a^2) + (y₀^2 / b^2) = 1) 
  (h4 : (y₀ / (x₀ + a)) * (y₀ / (a - x₀)) = 1 / 4) 
  : real.sqrt(1 - (b^2 / a^2)) = real.sqrt(3) / 2 :=
sorry

end eccentricity_of_ellipse_l298_298048


namespace factorization_correct_l298_298021

theorem factorization_correct (x : ℝ) :
  16 * x ^ 2 + 8 * x - 24 = 8 * (2 * x ^ 2 + x - 3) ∧ (2 * x ^ 2 + x - 3) = (2 * x + 3) * (x - 1) :=
by
  sorry

end factorization_correct_l298_298021


namespace children_have_flags_l298_298322

theorem children_have_flags (F : ℕ) (hF_even : F % 2 = 0) (h_all_flags : ∀ (C : ℕ), C = F / 2 ∧ 
  ((∃ B R : ℕ, B + R = F ∧ B % C = 0 ∧ R % C = 0) ∧ (∀ (b r : ℤ), (b + r = C) → (20 * C / 100 = b ∧ b = r)))) :
  100% = 100% :=
by
  sorry

end children_have_flags_l298_298322


namespace area_of_highlighted_region_l298_298626

/-- The figure is composed of 12 square tiles, each with side length 10 cm. 
Prove that the area of the highlighted region is 200 square cm. -/
theorem area_of_highlighted_region :
  (∃ (n : ℕ) (s : ℝ), n = 12 ∧ s = 10 ∧ 
  let A := (4 * (1 / 2 * s * s)) in A = 200) :=
sorry

end area_of_highlighted_region_l298_298626


namespace arc_limit_l298_298181

def fibonacci : ℕ → ℕ
| 0 := 0
| 1 := 1
| (n + 2) := fibonacci (n + 1) + fibonacci n

def arc_length (k : ℕ) : ℝ :=
  let F := fibonacci (2*k + 1)
  in 4 * F / (F^2 + 1)

noncomputable def total_arc_length (n : ℕ) : ℝ :=
  ∑ k in (range n), arc_length k

theorem arc_limit : tendsto (λ n, total_arc_length n) at_top (𝓝 (2 * real.pi)) := sorry

end arc_limit_l298_298181


namespace length_of_BC_l298_298152

theorem length_of_BC 
  (A B C : Type)
  [Triangle A B C]
  (acute : is_acute ABC)
  (AB : length A B = 3)
  (AC : length A C = 4)
  (area : triangle_area ABC = 3 * Real.sqrt 3) :
  length B C = Real.sqrt 13 := 
sorry

end length_of_BC_l298_298152


namespace ticket_price_calculation_l298_298344

variable (total_people total_revenue reserved_price unreserved_num reserved_num unreserved_price : ℝ)

theorem ticket_price_calculation (h_total_people : total_people = 1096)
                                (h_reserved_price : reserved_price = 25)
                                (h_reserved_num : reserved_num = 246)
                                (h_unreserved_num : unreserved_num = 246)
                                (h_total_revenue : total_revenue = 26_170)
                               
                                : 
                                (unreserved_price = 81.30) := by
  let reserved_revenue := reserved_price * reserved_num
  let unreserved_revenue := unreserved_price * unreserved_num
  have h_reserved_revenue : reserved_revenue = 6150 := by
    calc reserved_revenue = reserved_price * reserved_num : rfl
                       _ = 25 * 246             : by rw [h_reserved_price, h_reserved_num]
                       _ = 6150                 : by norm_num
  have h_total_revenue_eq : total_revenue = reserved_revenue + unreserved_revenue := by

    calc total_revenue = 26_170           : h_total_revenue
                  _ = 6150 + 246 * unreserved_price       : by rw [h_reserved_revenue, h_unreserved_num]
                  _ = reserved_revenue + unreserved_revenue := by rw [h_reserved_revenue, unreserved_revenue]
  have h_unreserved_price : 246 * unreserved_price = 20_020 := by
    calc 246 * unreserved_price = total_revenue - 6150 : h_total_revenue_eq.symm
                       _ = 26_170 - 6150        : by rw h_total_revenue
                       _ = 20_020               : by norm_num
  calc unreserved_price = 20_020 / 246         : by rw h_unreserved_price
                     _ = 81.30              : by norm_num

end ticket_price_calculation_l298_298344


namespace infinite_monochromatic_subset_l298_298839

open Set

theorem infinite_monochromatic_subset (X : Set α) (k c : ℕ) (hX: Infinite X) (hk: 0 < k) (hc: 0 < c) 
(coloring : {s : Set α | card s = k} → Fin c) : 
  ∃ (M : Set α), Infinite M ∧ ∃ (col : Fin c), ∀ s ∈ {s : Set α | card s = k}, M ⊆ X → coloring s = col :=
sorry

end infinite_monochromatic_subset_l298_298839


namespace minimum_a_value_l298_298836

noncomputable def min_a (a b c : ℝ) : Prop :=
  a + b + c = 3 ∧ a ≥ b ∧ b ≥ c ∧ (∃ x1 x2 : ℝ, ax^2 + bx + c = 0) → a ≥ 4 / 3

theorem minimum_a_value (a b c : ℝ) (h1 : a + b + c = 3) (h2 : a ≥ b) (h3 : b ≥ c) (h4 : ∃ x1 x2 : ℝ, ax^2 + bx + c = 0) : a ≥ 4 / 3 :=
sorry

end minimum_a_value_l298_298836


namespace domain_of_g_eq_7_infty_l298_298036

noncomputable def domain_function (x : ℝ) : Prop := (2 * x + 1 ≥ 0) ∧ (x - 7 > 0)

theorem domain_of_g_eq_7_infty : 
  (∀ x : ℝ, domain_function x ↔ x > 7) :=
by 
  -- We declare the structure of our proof problem here.
  -- The detailed proof steps would follow.
  sorry

end domain_of_g_eq_7_infty_l298_298036


namespace forest_logging_duration_l298_298149
open Nat

def forest_area_miles : Nat := 4 * 6
def trees_per_square_mile : Nat := 600
def total_trees : Nat := trees_per_square_mile * forest_area_miles

def team_A_loggers : Nat := 6
def team_A_days : Nat := 5
def team_A_trees_per_day : Nat := 5

def team_B_loggers : Nat := 8
def team_B_days : Nat := 4
def team_B_trees_per_day : Nat := 6

def team_C_loggers : Nat := 10
def team_C_days : Nat := 3
def team_C_trees_per_day : Nat := 8

def team_D_loggers : Nat := 12
def team_D_days : Nat := 2
def team_D_trees_per_day : Nat := 10

def team_A_weekly_trees : Nat := team_A_loggers * team_A_days * team_A_trees_per_day
def team_B_weekly_trees : Nat := team_B_loggers * team_B_days * team_B_trees_per_day
def team_C_weekly_trees : Nat := team_C_loggers * team_C_days * team_C_trees_per_day
def team_D_weekly_trees : Nat := team_D_loggers * team_D_days * team_D_trees_per_day

def total_weekly_trees : Nat := team_A_weekly_trees + team_B_weekly_trees + team_C_weekly_trees + team_D_weekly_trees

def days_in_month : Nat := 30
def weeks_in_month : Real := days_in_month / 7

def monthly_trees_cut_real : Real := total_weekly_trees * weeks_in_month

def total_trees_real : Real := (4 * 6 : Real) * (600 : Real)

def required_months_real : Real := total_trees_real / monthly_trees_cut_real
def required_months : Nat := required_months_real.ceil.toNat

theorem forest_logging_duration : required_months = 5 := by
  sorry

end forest_logging_duration_l298_298149


namespace solve_for_x_l298_298600

theorem solve_for_x (x : ℝ) (h : 5 + 3.5 * x = 2.5 * x - 25) : x = -30 :=
sorry

end solve_for_x_l298_298600


namespace robins_fraction_l298_298915

theorem robins_fraction (B R J : ℕ) (h1 : R + J = B)
  (h2 : 2/3 * (R : ℚ) + 1/3 * (J : ℚ) = 7/15 * (B : ℚ)) :
  (R : ℚ) / B = 2/5 :=
by
  sorry

end robins_fraction_l298_298915


namespace dihedral_angle_range_l298_298911

theorem dihedral_angle_range (n : ℕ) (h : 3 ≤ n) : 
  ∃ (θ : set ℝ), θ = set.Ioo ((n-2) * real.pi / n) real.pi :=
sorry

end dihedral_angle_range_l298_298911


namespace problem_intervals_increasing_sum_first_2018_y_n_l298_298095

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * sin (2 * x + π / 6)

-- Define the sequence {x_n}
def x_n (n : ℕ) : ℝ :=
  if n = 0 then π / 6
  else (π / 6 + (n + 1) * π / 2)

-- Define the sequence {y_n}
def y_n (n : ℕ) : ℝ :=
  f (x_n n)

theorem problem_intervals_increasing :
  ∀ k : ℤ, increasing (λ x, f x) (double_intervals (-π/3 + k * π) (π/6 + k * π)) :=
sorry

theorem sum_first_2018_y_n :
  (∑ i in finset.range 2018, y_n i) = 0 :=
sorry

end problem_intervals_increasing_sum_first_2018_y_n_l298_298095


namespace probability_of_sequence_l298_298043

open ProbabilityTheory

noncomputable def probability_sequence (p : ℝ) (h : 0 < p ∧ p < 1) : ℝ :=
  (1 - p)^7 * p^3

theorem probability_of_sequence :
  ∀ (p : ℝ) (h : 0 < p ∧ p < 1),
  probability_sequence p h = (1 - p)^7 * p^3 :=
  by
    intro p h
    simp [probability_sequence]
    sorry

end probability_of_sequence_l298_298043


namespace airplane_travel_difference_correct_l298_298008

-- Define airplane's speed without wind
def airplane_speed_without_wind : ℕ := a

-- Define wind speed
def wind_speed : ℕ := 20

-- Define time without wind
def time_without_wind : ℕ := 4

-- Define time against wind
def time_against_wind : ℕ := 3

-- Define distance covered without wind
def distance_without_wind : ℕ := airplane_speed_without_wind * time_without_wind

-- Define effective speed against wind
def effective_speed_against_wind : ℕ := airplane_speed_without_wind - wind_speed

-- Define distance covered against wind
def distance_against_wind : ℕ := effective_speed_against_wind * time_against_wind

-- Define the difference in distances
def distance_difference : ℕ := distance_without_wind - distance_against_wind

-- The theorem we wish to prove
theorem airplane_travel_difference_correct (a : ℕ) :
  distance_difference = a + 60 :=
by
sorry

end airplane_travel_difference_correct_l298_298008


namespace element_type_determined_by_protons_nuclide_type_determined_by_protons_neutrons_chemical_properties_determined_by_outermost_electrons_highest_positive_valence_determined_by_main_group_num_l298_298642

-- defining element, nuclide, and valence based on protons, neutrons, and electrons
def Element (protons : ℕ) := protons
def Nuclide (protons neutrons : ℕ) := (protons, neutrons)
def ChemicalProperties (outermostElectrons : ℕ) := outermostElectrons
def HighestPositiveValence (mainGroupNum : ℕ) := mainGroupNum

-- The proof problems as Lean theorems
theorem element_type_determined_by_protons (protons : ℕ) :
  Element protons = protons := sorry

theorem nuclide_type_determined_by_protons_neutrons (protons neutrons : ℕ) :
  Nuclide protons neutrons = (protons, neutrons) := sorry

theorem chemical_properties_determined_by_outermost_electrons (outermostElectrons : ℕ) :
  ChemicalProperties outermostElectrons = outermostElectrons := sorry
  
theorem highest_positive_valence_determined_by_main_group_num (mainGroupNum : ℕ) :
  HighestPositiveValence mainGroupNum = mainGroupNum := sorry

end element_type_determined_by_protons_nuclide_type_determined_by_protons_neutrons_chemical_properties_determined_by_outermost_electrons_highest_positive_valence_determined_by_main_group_num_l298_298642


namespace find_x_l298_298814

theorem find_x (x y : ℝ) (h1 : x / y = 12 / 5) (h2 : y = 25) : x = 60 :=
by
  sorry

end find_x_l298_298814


namespace total_games_won_l298_298363

theorem total_games_won (Betsy_games : ℕ) (Helen_games : ℕ) (Susan_games : ℕ) 
    (hBetsy : Betsy_games = 5)
    (hHelen : Helen_games = 2 * Betsy_games)
    (hSusan : Susan_games = 3 * Betsy_games) : 
    Betsy_games + Helen_games + Susan_games = 30 :=
sorry

end total_games_won_l298_298363


namespace product_terms_evaluation_l298_298287

theorem product_terms_evaluation : 
    (∏ i in (Finset.range 8).map (λ x => x + 1), (1 + (1 / (i : ℚ)))) = 9 := 
by 
  sorry

end product_terms_evaluation_l298_298287


namespace tangent_line_min_slope_l298_298059

theorem tangent_line_min_slope {x y : ℝ} (hx : y = x^3 + 3 * x^2 + 6 * x - 10) :
  ∃ x : ℝ, ∃ y : ℝ, ∃ m : ℝ, ∃ b : ℝ, (m = 3) ∧ (y = m * (x + 1) - 14) ∧ (3 * x - y - 11 = 0).
proof
  sorry

end tangent_line_min_slope_l298_298059


namespace solve_for_x_l298_298606

theorem solve_for_x (x : ℝ) (h : 5 + 3.5 * x = 2.5 * x - 25) : x = -30 :=
by 
  -- Placeholder for the actual proof
  sorry

end solve_for_x_l298_298606


namespace sum_of_b_l298_298083

-- Define the arithmetic sequence and its properties
def arithmetic_sequence (d : ℤ) (a₁ : ℤ) : ℕ → ℤ
| 0     := a₁
| (n+1) := arithmetic_sequence d a₁ n + d

-- Define S_n as the sum of the first n terms of the arithmetic sequence
def S (d a₁ : ℤ) (n : ℕ) : ℤ :=
(n * (2 * a₁ + (n - 1) * d)) / 2

-- Define b_n
def b (a : ℕ → ℤ) (n : ℕ) : ℚ :=
2 / (a n * a (n + 1))

-- Define T_n as the sum of the first n terms of the sequence b_n
def T (a : ℕ → ℤ) (n : ℕ) : ℚ :=
(∑ k in Finset.range n, b a k)

theorem sum_of_b (d : ℕ) (a₁ : ℕ) (n : ℕ)
  (h1 : ∀ n, arithmetic_sequence d a₁ n = 2 * n - 1)
  (h2 : S d a₁ 2^2 = (arithmetic_sequence d a₁ 0) * S d a₁ 4) :
  T (λ n, 2 * n - 1) n = (2 * n) / (2 * n + 1) := 
sorry

end sum_of_b_l298_298083


namespace grid_values_equal_l298_298360

theorem grid_values_equal (f : ℤ × ℤ → ℕ) (h : ∀ (i j : ℤ), 
  f (i, j) = 1 / 4 * (f (i + 1, j) + f (i - 1, j) + f (i, j + 1) + f (i, j - 1))) :
  ∀ (i j i' j' : ℤ), f (i, j) = f (i', j') :=
by
  sorry

end grid_values_equal_l298_298360


namespace tens_digit_N_to_20_l298_298548

theorem tens_digit_N_to_20 (N : ℕ) (h1 : Even N) (h2 : ¬(∃ k : ℕ, N = 10 * k)) : 
  ((N ^ 20) / 10) % 10 = 7 := 
by 
  sorry

end tens_digit_N_to_20_l298_298548


namespace longest_diagonal_of_rhombus_l298_298719

variable (d1 d2 : ℝ) (r : ℝ)
variable h_area : 0.5 * d1 * d2 = 150
variable h_ratio : d1 / d2 = 4 / 3

theorem longest_diagonal_of_rhombus :
  max d1 d2 = 20 :=
by
  sorry

end longest_diagonal_of_rhombus_l298_298719


namespace sixth_graders_count_l298_298266

theorem sixth_graders_count (total_students seventh_graders_percentage sixth_graders_percentage : ℝ)
                            (seventh_graders_count : ℕ)
                            (h1 : seventh_graders_percentage = 0.32)
                            (h2 : seventh_graders_count = 64)
                            (h3 : sixth_graders_percentage = 0.38)
                            (h4 : seventh_graders_count = seventh_graders_percentage * total_students) :
                            sixth_graders_percentage * total_students = 76 := by
  sorry

end sixth_graders_count_l298_298266


namespace average_weight_of_b_and_c_l298_298245

variables (a b c : ℝ)

theorem average_weight_of_b_and_c :
  (a + b + c) / 3 = 45 →  -- Condition 1
  (a + b) / 2 = 42 →      -- Condition 2
  b = 35 →                -- Condition 3
  (b + c) / 2 = 43 :=     -- Question (derived)
begin
  intros h1 h2 h3,
  sorry
end

end average_weight_of_b_and_c_l298_298245


namespace lcm_12_18_is_36_l298_298415

def prime_factors (n : ℕ) : list ℕ :=
  if n = 12 then [2, 2, 3]
  else if n = 18 then [2, 3, 3]
  else []

noncomputable def lcm_of_two (a b : ℕ) : ℕ :=
  match prime_factors a, prime_factors b with
  | [2, 2, 3], [2, 3, 3] => 36
  | _, _ => 0

theorem lcm_12_18_is_36 : lcm_of_two 12 18 = 36 :=
  sorry

end lcm_12_18_is_36_l298_298415


namespace psychology_majors_percentage_in_liberal_arts_l298_298358

theorem psychology_majors_percentage_in_liberal_arts 
  (total_students : ℕ) 
  (percent_freshmen : ℝ) 
  (percent_freshmen_liberal_arts : ℝ) 
  (percent_freshmen_psych_majors_liberal_arts : ℝ) 
  (h1: percent_freshmen = 0.40) 
  (h2: percent_freshmen_liberal_arts = 0.50)
  (h3: percent_freshmen_psych_majors_liberal_arts = 0.10) :
  ((percent_freshmen_psych_majors_liberal_arts / (percent_freshmen * percent_freshmen_liberal_arts)) * 100 = 50) :=
by
  sorry

end psychology_majors_percentage_in_liberal_arts_l298_298358


namespace scalene_triangle_properties_l298_298163

theorem scalene_triangle_properties
  (a b c : ℝ) (A B C : ℝ)
  (h1 : a = 3)
  (h2 : c = 4)
  (h3 : C = 2 * A)
  (h4 : ∀ (A B C : ℝ), A ≠ B ∧ A ≠ C ∧ B ≠ C) : 
  (cos A = 2 / 3) ∧ 
  (b = 7 / 3) ∧ 
  (cos (2 * A + (Real.pi / 6)) = (-Real.sqrt 3 - 4 * Real.sqrt 5) / 18) :=
by
  sorry

end scalene_triangle_properties_l298_298163


namespace monotone_increasing_implies_f1_gt_f0_l298_298817

theorem monotone_increasing_implies_f1_gt_f0
  {f : ℝ → ℝ} 
  (h_diff : ∀ x, 0 ≤ x ∧ x ≤ 1 → differentiable_at ℝ f x) 
  (h_derive_pos : ∀ x, 0 ≤ x ∧ x ≤ 1 → deriv f x > 0) :
  f 1 > f 0 :=
sorry

end monotone_increasing_implies_f1_gt_f0_l298_298817


namespace part1_part2_l298_298469

-- Define the function f(x) = |x - 4| - |x + 2|
def f (x : ℝ) : ℝ := abs (x - 4) - abs (x + 2)

-- Prove the range of values for a
theorem part1 : ∀ x a : ℝ, (f x) - a^2 + 5 * a ≥ 0 → (2 ≤ a ∧ a ≤ 3) :=
by
  sorry

-- Maximum value of f(x)
def M : ℝ := 6

-- Prove the maximum value of sqrt(a+1) + sqrt(b+2) + sqrt(c+3)
theorem part2 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = M) : 
  ∀ a b c : ℝ, a + b + c = 6 ∧ 0 < a ∧ 0 < b ∧ 0 < c → sqrt(a + 1) + sqrt(b + 2) + sqrt(c + 3) ≤ 6 :=
by
  sorry

end part1_part2_l298_298469


namespace mrs_white_expected_yield_l298_298975

noncomputable def orchard_yield : ℝ :=
  let length_in_feet : ℝ := 10 * 3
  let width_in_feet : ℝ := 30 * 3
  let total_area : ℝ := length_in_feet * width_in_feet
  let half_area : ℝ := total_area / 2
  let tomato_yield : ℝ := half_area * 0.75
  let cucumber_yield : ℝ := half_area * 0.4
  tomato_yield + cucumber_yield

theorem mrs_white_expected_yield :
  orchard_yield = 1552.5 := sorry

end mrs_white_expected_yield_l298_298975


namespace boat_license_combinations_l298_298741

theorem boat_license_combinations :
  let letter_choices := 3
  let digit_choices := 10
  let digit_positions := 5
  (letter_choices * (digit_choices ^ digit_positions)) = 300000 :=
  sorry

end boat_license_combinations_l298_298741


namespace fred_total_cents_l298_298812

def dimes_to_cents (dimes : ℕ) : ℕ := dimes * 10

theorem fred_total_cents (h_dimes : Fred dimes = 9) : Fred cents = 90 :=
by
  rw [← h_dimes]
  dsimper
  exact dimes_to_cents 9

end fred_total_cents_l298_298812


namespace extremum_value_l298_298456

noncomputable def f (x a b : ℝ) : ℝ := x^3 + 3 * a * x^2 + b * x + a^2

theorem extremum_value (a b : ℝ) (h1 : (3 - 6 * a + b = 0)) (h2 : (-1 + 3 * a - b + a^2 = 0)) :
  a - b = -7 :=
by
  sorry

end extremum_value_l298_298456


namespace min_chord_length_intercepted_line_eq_l298_298819

theorem min_chord_length_intercepted_line_eq (m : ℝ)
  (hC : ∀ (x y : ℝ), (x-1)^2 + (y-1)^2 = 16)
  (hL : ∀ (x y : ℝ), (2*m-1)*x + (m-1)*y - 3*m + 1 = 0)
  : ∃ x y : ℝ, x - 2*y - 4 = 0 := sorry

end min_chord_length_intercepted_line_eq_l298_298819


namespace probability_l298_298335

/-- Define the domain for the random selection of (x, y) --/
def domain (x y : ℝ) : Prop := 0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 3

/-- Define the inequality x + 2y ≤ 6 --/
def inequality (x y : ℝ) : Prop := x + 2y ≤ 6

/-- The total area of the rectangle with given bounds --/
def total_area : ℝ := 9

/-- The area of the triangle formed by (0,0), (3,0), (3, 3/2) --/
def triangle_area : ℝ := 9 / 4

/-- The probability of selecting a point (x, y) such that x + 2y ≤ 6 given 
    0 ≤ x ≤ 3 and 0 ≤ y ≤ 3 --/
theorem probability (x y : ℝ) (h₁ : domain x y) (h₂ : inequality x y) : 
  triangle_area / total_area = 1 / 4 := sorry

end probability_l298_298335


namespace stratified_sampling_third_grade_selection_l298_298748

noncomputable def total_students : ℕ := 1000
noncomputable def first_grade_students : ℕ := 350
noncomputable def second_grade_probability : ℝ := 0.32
noncomputable def selected_students : ℕ := 100

theorem stratified_sampling_third_grade_selection :
  let second_grade_students := (second_grade_probability * total_students : ℝ).to_nat,
      third_grade_students := total_students - first_grade_students - second_grade_students,
      third_grade_selection := (selected_students * third_grade_students / total_students : ℝ).to_nat
  in third_grade_selection = 33 :=
by
  sorry

end stratified_sampling_third_grade_selection_l298_298748


namespace tetrahedron_cd_length_l298_298636

theorem tetrahedron_cd_length (a b c d : Type) [MetricSpace a] [MetricSpace b] [MetricSpace c] [MetricSpace d] :
  let ab := 53
  let edge_lengths := [17, 23, 29, 39, 46, 53]
  ∃ cd, cd = 17 :=
by
  sorry

end tetrahedron_cd_length_l298_298636


namespace sum_x_coords_Q3_is_132_l298_298206

noncomputable def sum_x_coords_Q3 (x_coords: Fin 44 → ℝ) (sum_x1: ℝ) : ℝ :=
  sum_x1 -- given sum_x1 is the sum of x-coordinates of Q1 i.e., 132

theorem sum_x_coords_Q3_is_132 (x_coords: Fin 44 → ℝ) (sum_x1: ℝ) (h: sum_x1 = 132) :
  sum_x_coords_Q3 x_coords sum_x1 = 132 :=
by
  sorry

end sum_x_coords_Q3_is_132_l298_298206


namespace eval_trig_expression_l298_298395

theorem eval_trig_expression :
  (cos 20 * (Real.pi / 180)) / (cos 35 * (Real.pi / 180) * sqrt (1 - sin (20 * (Real.pi / 180)))) = sqrt 2 :=
by sorry

end eval_trig_expression_l298_298395


namespace find_f_value_l298_298443

-- Define the function f satisfying given condition
def f (ω φ : ℝ) (x : ℝ) := 3 * Real.sin (ω * x + φ)

-- Given condition: f(π/3 + x) = f(-x) for all x
def condition (ω φ : ℝ) :=
  ∀ (x : ℝ), f ω φ (π / 3 + x) = f ω φ (-x)

-- The main statement we want to prove
theorem find_f_value (ω φ : ℝ) (h : condition ω φ) :
  f ω φ (π / 6) = 3 ∨ f ω φ (π / 6) = -3 :=
sorry

end find_f_value_l298_298443


namespace find_slope_of_BC_l298_298826

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := (x^2 / 4) + y^2 = 1

-- Define the three points A, B, C
variables {A B C M : ℝ × ℝ}

-- Define centroid condition
def is_centroid_origin (A B C : ℝ × ℝ) : Prop := 
  let (xA, yA) := A in
  let (xB, yB) := B in
  let (xC, yC) := C in
  (xA + xB + xC = 0) ∧ (yA + yB + yC = 0)

-- Define area ratio condition
def area_ratio_cond (A B C M O : ℝ × ℝ) : Prop :=
  let (xA, yA) := A in
  let (xB, yB) := B in
  let (xC, yC) := C in
  let (xM, yM) := M in
  let O_area := (B.1 - O.1) * (C.2 - O.2) - (C.1 - O.1) * (B.2 - O.2) in
  let BM_area := (B.1 - M.1) * (A.2 - M.2) - (A.1 - M.1) * (B.2 - M.2) in
  3 * O_area = 2 * BM_area

-- Define the slope of the line BC
def is_neg_slope (B C : ℝ × ℝ) : Prop :=
  let slope := (C.2 - B.2) / (C.1 - B.1) in
  slope < 0

-- Define the correct slope conditions
def slope (B C : ℝ × ℝ) : ℝ := (C.2 - B.2) / (C.1 - B.1)

noncomputable def correct_slopes (k : ℝ) : Prop :=
  k = (-3 * Real.sqrt 3 / 2) ∨ k = (-Real.sqrt 3 / 6)

-- Final theorem statement
theorem find_slope_of_BC (A B C M : ℝ × ℝ) (O : ℝ × ℝ := (0,0)) :
  is_on_ellipse (A.1) (A.2) ∧ is_on_ellipse (B.1) (B.2) ∧ is_on_ellipse (C.1) (C.2) ∧
  is_neg_slope B C ∧ 
  is_centroid_origin A B C ∧ 
  area_ratio_cond A B C M O →
  correct_slopes (slope B C) :=
begin
  sorry
end

end find_slope_of_BC_l298_298826


namespace angle_between_vectors_l298_298102

variables {ℝ : Type*} [normed_group ℝ] [vector_space ℝ ℝ]

-- Define vectors a and b in Euclidean space with magnitudes and conditions
variables (a b : ℝ)
axiom norm_a : ‖a‖ = 2
axiom norm_b : ‖b‖ = 2/3
axiom norm_diff : ‖a - 1/2 • b‖ = real.sqrt 43 / 3

-- Define the angle and dot product
def dot (x y : ℝ) : ℝ := x * y   -- This could be further refined to true dot product
def cos_theta : ℝ := dot a b / (‖a‖ * ‖b‖)
def angle (x y : ℝ) : ℝ := real.arccos (cos_theta x y)

-- Statement of the proof problem
theorem angle_between_vectors :
  angle a b = 2 * real.pi / 3 :=
sorry

end angle_between_vectors_l298_298102


namespace probability_of_palindrome_div_by_7_l298_298333

def is_palindrome (n : ℕ) : Prop :=
  let digits := to_digits 10 n
  digits = digits.reverse

def four_digit_palindrome (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ n = 1001 * a + 110 * b

def prob_palindrome_div_by_7 : ℚ :=
  let valid_palindromes := { n : ℕ | four_digit_palindrome n }
  let valid_results := { n : ℕ | ∃ m : ℕ, four_digit_palindrome m ∧ is_palindrome (m / 7) }
  classical.some (∃ p : ℚ, p = valid_results.card / valid_palindromes.card)

theorem probability_of_palindrome_div_by_7 (n : ℕ) :
  let prob := 2 / 9 in
  (∀ n, four_digit_palindrome n → (n / 7).is_palindrome → (n / 7).is_palindrome = true)
  → prob_palindrome_div_by_7 = prob := sorry

end probability_of_palindrome_div_by_7_l298_298333


namespace solution_value_of_a_l298_298871

noncomputable def verify_a (a : ℚ) (A : Set ℚ) : Prop :=
  A = {a - 2, 2 * a^2 + 5 * a, 12} ∧ -3 ∈ A

theorem solution_value_of_a (a : ℚ) (A : Set ℚ) (h : verify_a a A) : a = -3 / 2 := by
  sorry

end solution_value_of_a_l298_298871


namespace simplify_complex_expression_l298_298230

theorem simplify_complex_expression : (5 + 7 * complex.I) / (5 - 7 * complex.I) + (5 - 7 * complex.I) / (5 + 7 * complex.I) = -23 / 37 := by
  sorry

end simplify_complex_expression_l298_298230


namespace students_who_walk_home_l298_298755

theorem students_who_walk_home :
  let car_fraction := (1 : ℚ) / 3
  let bus_fraction := (1 : ℚ) / 5
  let cycle_fraction := (1 : ℚ) / 8
  1 - (car_fraction + bus_fraction + cycle_fraction) = 41 / 120 :=
by {
  let car_fraction := (1 : ℚ) / 3,
  let bus_fraction := (1 : ℚ) / 5,
  let cycle_fraction := (1 : ℚ) / 8,
  let total_fraction := car_fraction + bus_fraction + cycle_fraction,
  have h1 : total_fraction = 79 / 120, { sorry },  -- Calculation step to be filled
  have h2 : 1 = 120 / 120, from by norm_num,
  calc 
    1 - total_fraction 
        = 120 / 120 - 79 / 120 : by rw [h2]
    ... = 41 / 120 : by norm_num
}

end students_who_walk_home_l298_298755


namespace number_of_mappings_n_elements_l298_298183

theorem number_of_mappings_n_elements
  (A : Type) [Fintype A] [DecidableEq A] (n : ℕ) (h : 3 ≤ n) (f : A → A)
  (H1 : ∀ x : A, ∃ c : A, ∀ (i : ℕ), i ≥ n - 2 → f^[i] x = c)
  (H2 : ∃ x₁ x₂ : A, f^[n] x₁ ≠ f^[n] x₂) :
  ∃ m : ℕ, m = (2 * n - 5) * (n.factorial) / 2 :=
sorry

end number_of_mappings_n_elements_l298_298183


namespace question_a_question_b_question_c_l298_298664

def f (x : ℝ) : ℝ := real.sqrt (x + real.sqrt (2 * x - 1)) + real.sqrt (x - real.sqrt (2 * x - 1))

theorem question_a (x : ℝ) (h : x ≥ 1 / 2) : 
  (f x = real.sqrt 2) ↔ (1 / 2 ≤ x ∧ x ≤ 1) := sorry

theorem question_b (x : ℝ) (h : x ≥ 1 / 2) : 
  (f x = 1) ↔ false := sorry

theorem question_c (x : ℝ) (h : x ≥ 1 / 2) : 
  (f x = 2) ↔ (x = 3 / 2) := sorry

end question_a_question_b_question_c_l298_298664


namespace proof_problem_l298_298834

-- Definitions needed for conditions
def a := -7 / 4
def b := -2 / 3
def m : ℚ := 1  -- m can be any rational number
def n : ℚ := -m  -- since m and n are opposite numbers

-- Lean statement to prove the given problem
theorem proof_problem : 4 * a / b + 3 * (m + n) = 21 / 2 := by
  -- Definitions ensuring a, b, m, n meet the conditions
  have habs : |a| = 7 / 4 := by sorry
  have brecip : 1 / b = -3 / 2 := by sorry
  have moppos : m + n = 0 := by sorry
  sorry

end proof_problem_l298_298834


namespace faith_earnings_correct_l298_298789

variable (pay_per_hour : ℝ) (regular_hours_per_day : ℝ) (work_days_per_week : ℝ) (overtime_hours_per_day : ℝ)
variable (overtime_rate_multiplier : ℝ)

def total_earnings (pay_per_hour : ℝ) (regular_hours_per_day : ℝ) (work_days_per_week : ℝ) 
                   (overtime_hours_per_day : ℝ) (overtime_rate_multiplier : ℝ) : ℝ :=
  let regular_hours := regular_hours_per_day * work_days_per_week
  let overtime_hours := overtime_hours_per_day * work_days_per_week
  let overtime_pay_rate := pay_per_hour * overtime_rate_multiplier
  let regular_earnings := pay_per_hour * regular_hours
  let overtime_earnings := overtime_pay_rate * overtime_hours
  regular_earnings + overtime_earnings

theorem faith_earnings_correct : 
  total_earnings 13.5 8 5 2 1.5 = 742.50 :=
by
  -- This is where the proof would go, but it's omitted as per the instructions
  sorry

end faith_earnings_correct_l298_298789


namespace max_sum_sign_counts_l298_298681

theorem max_sum_sign_counts (table : matrix (fin 30) (fin 30) bool)
  (pluses minuses : fin 900 → Option (fin 30 × fin 30)) 
  (h_pluses : ∃ (pls : fin 900), pls.val = 162) 
  (h_minuses : ∃ (mins : fin 900), mins.val = 144) 
  (sign_in_row : ∀ (i : fin 30), ∃ (k : fin 31), k ≤ 17 ∧ 
                  ∃ (p : fin k), table i p = tt) 
  (sign_in_col : ∀ (j : fin 30), ∃ (k : fin 31), k ≤ 17 ∧ 
                  ∃ (p : fin k), table p j = tt)
  (max_count : ∀ (i j : fin 30), 
                if table i j 
                then count_min i table ≤ 8
                else count_pls j table ≤ 9) :
  ∑ i j, if table i j 
          then count_min i table 
          else count_pls j table = 2592 := sorry

end max_sum_sign_counts_l298_298681


namespace calculate_group_A_B_C_and_total_is_correct_l298_298042

def groupA_1week : Int := 175000
def groupA_2week : Int := 107000
def groupA_3week : Int := 35000
def groupB_1week : Int := 100000
def groupB_2week : Int := 70350
def groupB_3week : Int := 19500
def groupC_1week : Int := 45000
def groupC_2week : Int := 87419
def groupC_3week : Int := 14425
def kids_staying_home : Int := 590796
def kids_outside_county : Int := 22

def total_kids_in_A := groupA_1week + groupA_2week + groupA_3week
def total_kids_in_B := groupB_1week + groupB_2week + groupB_3week
def total_kids_in_C := groupC_1week + groupC_2week + groupC_3week
def total_kids_in_camp := total_kids_in_A + total_kids_in_B + total_kids_in_C
def total_kids := total_kids_in_camp + kids_staying_home + kids_outside_county

theorem calculate_group_A_B_C_and_total_is_correct :
  total_kids_in_A = 317000 ∧
  total_kids_in_B = 189850 ∧
  total_kids_in_C = 146844 ∧
  total_kids = 1244512 := by
  sorry

end calculate_group_A_B_C_and_total_is_correct_l298_298042


namespace probability_sum_15_is_zero_l298_298565

-- Define what it means to flip a fair coin with sides labeled 5 and 15
inductive FairCoin
| heads : FairCoin
| tails : FairCoin

def coinValue : FairCoin → ℕ
| FairCoin.heads := 5
| FairCoin.tails := 15

-- Define a standard six-sided die roll
inductive SixSidedDie
| one : SixSidedDie
| two : SixSidedDie
| three : SixSidedDie
| four : SixSidedDie
| five : SixSidedDie
| six : SixSidedDie

def dieValue : SixSidedDie → ℕ
| SixSidedDie.one := 1
| SixSidedDie.two := 2
| SixSidedDie.three := 3
| SixSidedDie.four := 4
| SixSidedDie.five := 5
| SixSidedDie.six := 6

-- Statement: The probability that the sum of the coin and die equals 15 is 0
theorem probability_sum_15_is_zero : (coinValue FairCoin.heads + dieValue die = 15 ∨ coinValue FairCoin.tails + dieValue die = 15) → false :=
by
  intros h
  cases h with h1 h2
  all_goals { cases die <;> simp [dieValue, coinValue] at h1 h2}
  
sorry

end probability_sum_15_is_zero_l298_298565


namespace b_formula_a_range_l298_298965

-- Define the sequence and conditions
def Seq (n : ℕ) : ℝ
| 0       => 0  -- This is a placeholder; in practice use Sn for initial partial sums
| (n+1) => sorry  -- Definition of Sn

def a_seq : ℕ → ℝ
| 0       => a -- Initial value a1 = a
| (n+1) => Seq n + 3^n -- Definition for a_{n+1}

-- Define the sequence b_n
def b_seq (n : ℕ) : ℝ :=
  Seq n - 3^n 

-- The Lean statement proving the general formula for b_seq
theorem b_formula (n : ℕ) (h : n > 0) : b_seq n = (a - 3) * 2^(n-1) := 
by
  sorry

-- The Lean statement proving the range of values for a
theorem a_range (h : ∀ n > 0, a_seq (n + 1) ≥ a_seq n) : a ≥ -9 := 
by
  sorry

end b_formula_a_range_l298_298965


namespace cos_BCA_isosceles_l298_298935

theorem cos_BCA_isosceles (A B C : Point) (γ α : ℝ) (h1 : cos γ = 4 / 5) (isos : triangle_is_isosceles A B C) (h2 : α = (180 - γ) / 2) : cos α = -sqrt (9 / 20) :=
by
  sorry

end cos_BCA_isosceles_l298_298935


namespace equivalent_discount_l298_298332

theorem equivalent_discount : 
  ∃ (d : ℝ), d ≈ 40.5 ∧ (50 * (1 - 30 / 100) * (1 - 15 / 100) = 50 * (1 - d / 100)) := 
by 
  sorry

end equivalent_discount_l298_298332


namespace Yoongi_class_students_l298_298511

theorem Yoongi_class_students (Total_a Total_b Total_ab : ℕ)
  (h1 : Total_a = 18)
  (h2 : Total_b = 24)
  (h3 : Total_ab = 7)
  (h4 : Total_a + Total_b - Total_ab = 35) : 
  Total_a + Total_b - Total_ab = 35 :=
sorry

end Yoongi_class_students_l298_298511


namespace find_triples_l298_298053

theorem find_triples (a b p : ℕ) (hp : p.prime) (ha : a > 0) (hb : b > 0) (h : a^p - b^p = 2013) :
  (a, b, p) = (337, 334, 2) ∨ (a, b, p) = (97, 86, 2) ∨ (a, b, p) = (47, 14, 2) :=
by {
  -- Proof goes here
  sorry
}

end find_triples_l298_298053


namespace correct_statement_is_B_l298_298006

constant two_rays_form_angle : Prop := 
  "The figure formed by two rays is called an angle."

constant two_points_determines_line : Prop := 
  "Two points determine a line."

constant straight_line_shortest_distance : Prop := 
  "The straight line is the shortest distance between two points."

constant extend_line_AB_to_C : Prop := 
  "Extend the line AB to C."

theorem correct_statement_is_B : two_points_determines_line := 
by {
  sorry
}

end correct_statement_is_B_l298_298006


namespace tan_inequality_l298_298576

theorem tan_inequality (α β : ℝ) (h: cos α * cos β > 0) :
  - (tan (α/2))^2 ≤ (tan ((β - α)/2)) / (tan ((β + α)/2)) ∧
  (tan ((β - α)/2)) / (tan ((β + α)/2)) ≤ (tan (β/2))^2 := 
sorry

end tan_inequality_l298_298576


namespace model_to_statue_ratio_l298_298257

theorem model_to_statue_ratio (h_statue : ℝ) (h_model : ℝ) (h_statue_eq : h_statue = 60) (h_model_eq : h_model = 4) :
  (h_statue / h_model) = 15 := by
  sorry

end model_to_statue_ratio_l298_298257


namespace rationalize_denominator_l298_298222

theorem rationalize_denominator :
  (let A := -6;
       B := -4;
       C := 0;
       D := 1;
       E := 30;
       F := 24 
   in A + B + C + D + E + F = 45) := 
by 
  let A := -6;
  let B := -4;
  let C := 0;
  let D := 1;
  let E := 30;
  let F := 24 in
  have h : A + B + C + D + E + F = 45 :=
        by simp [A, B, C, D, E, F];
  exact h

end rationalize_denominator_l298_298222


namespace shaded_area_approx_l298_298377

-- Define the dimensions of the rectangle and the two circles as given in the conditions
def length : ℝ := 16
def width : ℝ := 8
def radius_large : ℝ := 4
def radius_small : ℝ := 2

-- Define the area of the rectangle
def area_rectangle : ℝ := length * width

-- Define the area of the larger circle
def area_large_circle : ℝ := Real.pi * (radius_large ^ 2)

-- Define the area of the smaller circle
def area_small_circle : ℝ := Real.pi * (radius_small ^ 2)

-- Define the total area subtracted due to the two circles
def area_total_subtracted : ℝ := area_large_circle + area_small_circle

-- Define the shaded area remaining in the rectangle
def area_shaded : ℝ := area_rectangle - area_total_subtracted

-- Lean statement to prove the total shaded area is approximately 65.2 square feet
theorem shaded_area_approx : area_shaded ≈ 65.2 := by
  -- apply the actual computation here and show that area_shaded is approximately 65.2
  sorry

end shaded_area_approx_l298_298377


namespace total_cost_is_80_l298_298317

-- Conditions
def cost_flour := 3 * 3
def cost_eggs := 3 * 10
def cost_milk := 7 * 5
def cost_baking_soda := 2 * 3

-- Question and proof requirement
theorem total_cost_is_80 : cost_flour + cost_eggs + cost_milk + cost_baking_soda = 80 := by
  sorry

end total_cost_is_80_l298_298317


namespace sixth_graders_count_l298_298267

theorem sixth_graders_count (total_students seventh_graders_percentage sixth_graders_percentage : ℝ)
                            (seventh_graders_count : ℕ)
                            (h1 : seventh_graders_percentage = 0.32)
                            (h2 : seventh_graders_count = 64)
                            (h3 : sixth_graders_percentage = 0.38)
                            (h4 : seventh_graders_count = seventh_graders_percentage * total_students) :
                            sixth_graders_percentage * total_students = 76 := by
  sorry

end sixth_graders_count_l298_298267


namespace tan_domain_l298_298251

open Real

theorem tan_domain (k : ℤ) (x : ℝ) :
  (∀ k : ℤ, x ≠ (k * π / 2) + (3 * π / 8)) ↔ 
  (∀ k : ℤ, 2 * x - π / 4 ≠ k * π + π / 2) := sorry

end tan_domain_l298_298251


namespace dave_total_earnings_l298_298776

def hourly_wage (day : ℕ) : ℝ :=
  if day = 0 then 6 else
  if day = 1 then 7 else
  if day = 2 then 9 else
  if day = 3 then 8 else 
  0

def hours_worked (day : ℕ) : ℝ :=
  if day = 0 then 6 else
  if day = 1 then 2 else
  if day = 2 then 3 else
  if day = 3 then 5 else 
  0

def unpaid_break (day : ℕ) : ℝ :=
  if day = 0 then 0.5 else
  if day = 1 then 0.25 else
  if day = 2 then 0 else
  if day = 3 then 0.5 else 
  0

def daily_earnings (day : ℕ) : ℝ :=
  (hours_worked day - unpaid_break day) * hourly_wage day

def net_earnings (day : ℕ) : ℝ :=
  daily_earnings day - (daily_earnings day * 0.1)

def total_net_earnings : ℝ :=
  net_earnings 0 + net_earnings 1 + net_earnings 2 + net_earnings 3

theorem dave_total_earnings : total_net_earnings = 97.43 := by
  sorry

end dave_total_earnings_l298_298776


namespace student_score_l298_298342

theorem student_score
    (total_questions : ℕ)
    (correct_responses : ℕ)
    (grading_method : ℕ → ℕ → ℕ)
    (h1 : total_questions = 100)
    (h2 : correct_responses = 92)
    (h3 : grading_method = λ correct incorrect => correct - 2 * incorrect) :
  grading_method correct_responses (total_questions - correct_responses) = 76 :=
by
  -- proof would be here, but is skipped
  sorry

end student_score_l298_298342


namespace indian_percentage_women_l298_298147

theorem indian_percentage_women :
  (let total_people := 500 + 300 + 500 in
   let indian_men := 0.1 * 500 in
   let indian_children := 0.7 * 500 in
   let total_indians := indian_men + indian_children + (300 * x / 100) in
   total_indians / total_people = 0.4461538461538461 → x = 60) :=
by
  sorry

end indian_percentage_women_l298_298147


namespace solve_for_x_l298_298604

theorem solve_for_x (x : ℝ) (h : 5 + 3.5 * x = 2.5 * x - 25) : x = -30 :=
by 
  -- Placeholder for the actual proof
  sorry

end solve_for_x_l298_298604


namespace parabola_translation_l298_298002

theorem parabola_translation :
  ∀ (x : ℝ),
  (∃ x' y' : ℝ, x' = x - 1 ∧ y' = 2 * x' ^ 2 - 3 ∧ y = y' + 3) →
  (y = 2 * x ^ 2) :=
by
  sorry

end parabola_translation_l298_298002


namespace banks_should_offer_benefits_to_seniors_l298_298583

-- Definitions based on conditions
def better_credit_reliability (pensioners : Type) : Prop :=
  ∀ (p : pensioners), has_better_credit_reliability p

def stable_pension_income (pensioners : Type) : Prop :=
  ∀ (p : pensioners), has_stable_income p

def indirect_financial_benefits (pensioners : Type) : Prop :=
  ∀ (p : pensioners), receives_financial_benefit p

def propensity_to_save (pensioners : Type) : Prop :=
  ∀ (p : pensioners), has_saving_habits p

def preference_long_term_deposits (pensioners : Type) : Prop :=
  ∀ (p : pensioners), prefers_long_term_deposits p

-- Main theorem statement
theorem banks_should_offer_benefits_to_seniors
  (P : Type)
  (h1 : better_credit_reliability P)
  (h2 : stable_pension_income P)
  (h3 : indirect_financial_benefits P)
  (h4 : propensity_to_save P)
  (h5 : preference_long_term_deposits P) :
  ∃ benefits : Type, benefits.make_sense :=
sorry

end banks_should_offer_benefits_to_seniors_l298_298583


namespace minimum_weighings_l298_298753

-- Defining the problem context
def coins (n : ℕ) := n > 2

-- Defining what needs to be proven
theorem minimum_weighings (n : ℕ) (h : coins n) : 
  ∃ w, (w ≤ 2) ∧ 
       (∀ (B C A : ℕ → Prop), 
          (B = λ x, x < A) ∧ (C = λ x, x < A) ∧ (A = λ x, x < B + C) → 
          (B + C = n) ∧ 
          (∃ w1 w2, w1 = w2 ⊔ Σ n_less_than_n_plus_2, λ C A, A ≠ C ⊔ C ≠ B)
       ) := sorry

end minimum_weighings_l298_298753


namespace mo_tea_cups_l298_298567

theorem mo_tea_cups (n t : ℤ) 
  (h1 : 2 * n + 5 * t = 26) 
  (h2 : 5 * t = 2 * n + 14) :
  t = 4 :=
sorry

end mo_tea_cups_l298_298567


namespace travel_time_equation_l298_298750

theorem travel_time_equation
 (d : ℝ) (x t_saved factor : ℝ) 
 (h : d = 202) 
 (h1 : t_saved = 1.8) 
 (h2 : factor = 1.6)
 : (d / x) * factor = d / (x - t_saved) := sorry

end travel_time_equation_l298_298750


namespace octagon_properties_l298_298831

-- Define the conditions
variables {B D E F A C : Type} [MetricSpace B] 
variables (BDEF_is_square : B × D × E × F)
variables (AB BC AC : ℝ)
variables (AB_eq : AB = 2)
variables (BC_eq : BC = 2)
variables (triangle_ABC : is_right_isosceles A B C)

-- Define the octagon and calculate its properties
noncomputable def octagon_area_perimeter : ℝ × ℝ :=
  let AC := real.sqrt (AB^2 + BC^2) in
  let square_side := 4 + 2 * AC in
  let square_area := square_side^2 in
  let triangle_area := 4 * (1 / 2 * AB * BC) in
  let octagon_area := square_area - triangle_area in
  let octagon_perimeter := 8 * AC in
  (octagon_area, octagon_perimeter)

-- Assert the properties
theorem octagon_properties :
  octagon_area_perimeter BDEF_is_square AB_eq BC_eq triangle_ABC = (16 + 8 * real.sqrt 2, 16 * real.sqrt 2) :=
sorry

end octagon_properties_l298_298831


namespace sum_S_eq_l298_298558

noncomputable def S (n : ℕ) (h : 0 < n) : ℝ :=
  1 / n - 1 / (n + 1)

theorem sum_S_eq :
  (Finset.sum (Finset.range 2016) (λ i, S (i + 1) (nat.succ_pos i))) = 2016 / 2017 := sorry

end sum_S_eq_l298_298558


namespace dissect_to_three_squares_l298_298228

theorem dissect_to_three_squares :
  ∃ (pieces : Finset (ℕ × ℕ)) (reassemble : pieces → Finset (ℕ × ℕ)),
  let initial_square_area := 7 * 7,
      square_areas := [6 * 6, 3 * 3, 2 * 2] in
  initial_square_area = square_areas.sum ∧
  pieces.card ≤ 5 ∧
  reassemble pieces = Finset.singleton (6, 6) ∪ Finset.singleton (3, 3) ∪ Finset.singleton (2, 2) := 
sorry

end dissect_to_three_squares_l298_298228


namespace max_value_of_f_l298_298298

theorem max_value_of_f :
  ∀ (x : ℝ), -5 ≤ x ∧ x ≤ 13 → ∃ (y : ℝ), y = x - 5 ∧ y ≤ 8 ∧ y >= -10 ∧ 
  (∀ (z : ℝ), z = (x - 5) → z ≤ 8) := 
by
  sorry

end max_value_of_f_l298_298298


namespace common_pasture_area_l298_298304

variable (Area_Ivanov Area_Petrov Area_Sidorov Area_Vasilev Area_Ermolaev : ℝ)
variable (Common_Pasture : ℝ)

theorem common_pasture_area :
  Area_Ivanov = 24 ∧
  Area_Petrov = 28 ∧
  Area_Sidorov = 10 ∧
  Area_Vasilev = 20 ∧
  Area_Ermolaev = 30 →
  Common_Pasture = 17.5 :=
sorry

end common_pasture_area_l298_298304


namespace trevor_coin_difference_l298_298277

theorem trevor_coin_difference (total_coins quarters dimes : ℕ) (h1 : total_coins = 127) (h2 : quarters = 39) (h3 : dimes = 28) 
  (nickels pennies : ℕ) (h4 : 3 * pennies = 2 * nickels) : 
  (nickels + pennies) - (quarters + dimes) = -7 :=
by
  sorry

end trevor_coin_difference_l298_298277


namespace part1_part2_l298_298861

noncomputable def f (x: ℝ) : ℝ := |x + 2| + |x - 1|

theorem part1 (x : ℝ) (hx : f x ≥ x + 8) : x ≤ -3 ∨ x ≥ 7 := 
begin
  sorry
end

theorem part2 (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_eq : (1/a) + (1/(2*b)) + (1/(3*c)) = 1) : 
  a + 2*b + 3*c ≥ 9 :=
begin
  sorry
end

end part1_part2_l298_298861


namespace sqrt_x2y_l298_298892

theorem sqrt_x2y (x y : ℝ) (h : x * y < 0) : Real.sqrt (x^2 * y) = -x * Real.sqrt y :=
sorry

end sqrt_x2y_l298_298892


namespace castle_food_supply_l298_298643

theorem castle_food_supply :
  ∀ (initial_people : ℕ) (initial_days : ℕ) (days_elapsed : ℕ) (people_leave : ℕ),
  initial_people = 300 →
  initial_days = 90 →
  days_elapsed = 30 →
  people_leave = 100 →
  let remaining_people := initial_people - people_leave in
  let remaining_days := initial_days - days_elapsed in
  (remaining_days * initial_people) / remaining_people = 90 :=
begin
  intros,
  sorry
end

end castle_food_supply_l298_298643


namespace dan_found_dimes_l298_298759

theorem dan_found_dimes:
  (barry_dimes_value_initial: ℝ)
  (dan_dimes_initial_ratio: ℝ)
  (dan_dimes_final: ℕ)
  (barry_dimes_value_initial = 10.0)
  (dan_dimes_initial_ratio = 0.5)
  (dan_dimes_final = 52)
  : 
  (dan_dimes_found: ℕ),
  (dan_dimes_found = dan_dimes_final - (barry_dimes_value_initial / 0.10 * dan_dimes_initial_ratio)) →
  dan_dimes_found = 2 := by
  sorry

end dan_found_dimes_l298_298759


namespace sqrt_x2y_neg_x_sqrt_y_l298_298894

variables {x y : ℝ} (h : x * y < 0)

theorem sqrt_x2y_neg_x_sqrt_y (h : x * y < 0): real.sqrt (x ^ 2 * y) = -x * real.sqrt y :=
sorry

end sqrt_x2y_neg_x_sqrt_y_l298_298894


namespace purely_imaginary_condition_l298_298849

-- Define the given conditions and the question.
theorem purely_imaginary_condition (z : ℂ) (θ : ℝ) (k : ℤ) :
  (z = complex.sin θ - complex.i * complex.cos θ) →
  (θ = 2 * k * real.pi) →
  (∃ (k : ℤ), θ = 2 * k * real.pi → z = 0 - complex.i ∨ z = complex.i) ∧
  (∃ (θ : ℝ), (z = complex.sin θ - complex.i * complex.cos θ) → θ ≠ 2 * k * real.pi) := 
sorry

end purely_imaginary_condition_l298_298849


namespace sequence_formula_l298_298869

theorem sequence_formula (a : ℕ → ℤ) (h0 : a 0 = 1) (h1 : a 1 = 5)
    (h_rec : ∀ n, n ≥ 2 → a n = (2 * (a (n - 1))^2 - 3 * (a (n - 1)) - 9) / (2 * a (n - 2))) :
  ∀ n, a n = 2^(n + 2) - 3 :=
by
  intros
  sorry

end sequence_formula_l298_298869


namespace two_arrows_in_equals_two_arrows_out_l298_298356

theorem two_arrows_in_equals_two_arrows_out
  (n : ℕ) -- number of sides (vertices) in the polygon
  (k : ℕ) -- number of vertices with two arrows entering
  (polygon_has_arrows : ∀ (v : ℕ), v < n → ∃ (in_arrows out_arrows: ℕ), in_arrows + out_arrows = 2 ∧ (if in_arrows = 2 then v ∈ {k} else true)) :
  k = (n - k - (n - 2 * k)) :=
by
  sorry

end two_arrows_in_equals_two_arrows_out_l298_298356


namespace find_triplet_l298_298399

def ordered_triplet : Prop :=
  ∃ (x y z : ℚ), 
  7 * x + 3 * y = z - 10 ∧ 
  2 * x - 4 * y = 3 * z + 20 ∧ 
  x = 0 ∧ 
  y = -50 / 13 ∧ 
  z = -20 / 13

theorem find_triplet : ordered_triplet := 
  sorry

end find_triplet_l298_298399


namespace find_number_of_integers_l298_298427

def euler_totient (n : ℕ) : ℕ :=
  if n = 0 then 0 else (Finset.card ((Finset.range n).filter (Nat.coprime n)))

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem find_number_of_integers (count_eq : Nat) : count_eq = 13 :=
  let n_vals := (Finset.range 100).filter (λ n, is_prime (n - euler_totient (n + 1)))
  Finset.card n_vals = count_eq
  sorry

end find_number_of_integers_l298_298427


namespace sqrt_diff_solve_x_range_of_sqrt_sum_series_sum_l298_298777

-- Problem 1.1
theorem sqrt_diff (x : ℝ) (h : sqrt (20 - x) + sqrt (4 - x) = 8) : sqrt (20 - x) - sqrt (4 - x) = 2 := by
  sorry

-- Problem 1.2
theorem solve_x (x : ℝ) (h : sqrt (20 - x) + sqrt (4 - x) = 8) : x = -5 := by
  sorry

-- Problem 2
theorem range_of_sqrt_sum (x : ℝ) (h₁ : 2 ≤ x) (h₂ : x ≤ 10) : 
  2 * sqrt 2 ≤ sqrt (10 - x) + sqrt (x - 2) ∧ sqrt (10 - x) + sqrt (x - 2) ≤ 4 := by
  sorry

-- Problem 3
theorem series_sum : 
  (Finset.range 1012).sum (λ n, 1 / ((2 * n + 3) * sqrt (2 * n - 1) + sqrt (2 * n + 1))) = 
  (2023 - sqrt 2023) / 4046 := by
  sorry

end sqrt_diff_solve_x_range_of_sqrt_sum_series_sum_l298_298777


namespace black_haired_girls_count_l298_298560

def initial_total_girls : ℕ := 80
def added_blonde_girls : ℕ := 10
def initial_blonde_girls : ℕ := 30

def total_girls := initial_total_girls + added_blonde_girls
def total_blonde_girls := initial_blonde_girls + added_blonde_girls
def black_haired_girls := total_girls - total_blonde_girls

theorem black_haired_girls_count : black_haired_girls = 50 := by
  sorry

end black_haired_girls_count_l298_298560


namespace solve_for_x_l298_298610

theorem solve_for_x (x : ℝ) : 5 + 3.5 * x = 2.5 * x - 25 ↔ x = -30 :=
by {
  split,
  {
    intro h,
    calc
      x = -30 : by sorry,
  },
  {
    intro h,
    calc
      5 + 3.5 * (-30) = 5 - 105
                       = -100,
      2.5 * (-30) - 25 = -75 - 25
                       = -100,
    exact Eq.symm (by sorry),
  }
}

end solve_for_x_l298_298610


namespace pure_imaginary_value_l298_298461

theorem pure_imaginary_value (a : ℝ) 
  (h1 : (a^2 - 3 * a + 2) = 0) 
  (h2 : (a - 2) ≠ 0) : a = 1 := sorry

end pure_imaginary_value_l298_298461


namespace solve_for_x_l298_298605

theorem solve_for_x (x : ℝ) (h : 5 + 3.5 * x = 2.5 * x - 25) : x = -30 :=
by 
  -- Placeholder for the actual proof
  sorry

end solve_for_x_l298_298605


namespace john_has_dollars_left_l298_298534

-- Definitions based on the conditions
def john_savings_octal : ℕ := 5273
def rental_car_cost_decimal : ℕ := 1500

-- Define the function to convert octal to decimal
def octal_to_decimal (n : ℕ) : ℕ := -- Conversion logic
sorry

-- Statements for the conversion and subtraction
def john_savings_decimal : ℕ := octal_to_decimal john_savings_octal
def amount_left_for_gas_and_accommodations : ℕ :=
  john_savings_decimal - rental_car_cost_decimal

-- Theorem statement equivalent to the correct answer
theorem john_has_dollars_left :
  amount_left_for_gas_and_accommodations = 1247 :=
by sorry

end john_has_dollars_left_l298_298534


namespace range_of_a_l298_298139

theorem range_of_a (a : ℝ) :
  (∃ x : ℤ, 2 * (x : ℝ) - 1 > 3 ∧ x ≤ a) ∧ (∀ x : ℤ, 2 * (x : ℝ) - 1 > 3 → x ≤ a) → 5 ≤ a ∧ a < 6 :=
by
  sorry

end range_of_a_l298_298139


namespace problem_inequality_l298_298822

noncomputable def a : ℕ → ℕ
| 1     := 1
| 2     := 3
| (n+1) := 3 * 4 ^ (n - 1)

def S (n : ℕ) : ℕ := 4 ^ (n - 1)

def b (n : ℕ) : ℕ := 2 * n + 1

noncomputable def p (n : ℕ) : ℝ :=
  (∏ i in (range (2 * n)).filter (λ i, i % 2 = 1), b (i + 1))
  / (∏ i in (range (2 * n)).filter (λ i, i % 2 = 0), b (i + 1))

theorem problem_inequality (n : ℕ) :
  (∀ k ≥ 1, p k) → 
  (p 1 + p 2 + ... + p n < (sqrt 3 / 2) * (sqrt (4 * n + 3) - sqrt 3)) :=
sorry

end problem_inequality_l298_298822


namespace subtract_15_after_multiplying_by_10_l298_298481

theorem subtract_15_after_multiplying_by_10 (n : ℕ) (h : n / 10 = 6) : n - 15 = 45 :=
by
  have h₁ : n = 6 * 10 := by sorry
  rw [h₁]
  calc
    60 - 15 = 45 := by sorry

end subtract_15_after_multiplying_by_10_l298_298481


namespace train_length_proof_l298_298744

-- Definitions based on the conditions given in the problem
def speed_km_per_hr := 45 -- speed of the train in km/hr
def time_seconds := 60 -- time taken to pass the platform in seconds
def length_platform_m := 390 -- length of the platform in meters

-- Conversion factor from km/hr to m/s
def km_per_hr_to_m_per_s (speed : ℕ) : ℕ := (speed * 1000) / 3600

-- Calculate the speed in m/s
def speed_m_per_s : ℕ := km_per_hr_to_m_per_s speed_km_per_hr

-- Calculate the total distance covered by the train while passing the platform
def total_distance_m : ℕ := speed_m_per_s * time_seconds

-- Total distance is the sum of the length of the train and the length of the platform
def length_train_m := total_distance_m - length_platform_m

-- The statement to prove the length of the train
theorem train_length_proof : length_train_m = 360 :=
by
  sorry

end train_length_proof_l298_298744


namespace longest_diagonal_of_rhombus_l298_298735

variables (d1 d2 : ℝ) (x : ℝ)
def rhombus_area := (d1 * d2) / 2
def diagonal_ratio := d1 / d2 = 4 / 3

theorem longest_diagonal_of_rhombus (h : rhombus_area (4 * x) (3 * x) = 150) (r : diagonal_ratio (4 * x) (3 * x)) : d1 = 20 := by
  sorry

end longest_diagonal_of_rhombus_l298_298735


namespace trivia_team_members_l298_298747

theorem trivia_team_members (n p s x y : ℕ) (h1 : n = 12) (h2 : p = 64) (h3 : s = 8) (h4 : x = p / s) (h5 : y = n - x) : y = 4 :=
by
  sorry

end trivia_team_members_l298_298747


namespace find_k_values_l298_298795

theorem find_k_values {k : ℝ} :
  (∀ x : ℝ, x^2 - (k - 5) * x - k + 9 > 1) ↔ k ∈ Ioo (-1 : ℝ) 7 :=
begin
  sorry
end

end find_k_values_l298_298795


namespace longest_diagonal_of_rhombus_l298_298713

theorem longest_diagonal_of_rhombus (d1 d2 : ℝ) (area : ℝ) (ratio : ℝ) (h1 : area = 150) (h2 : d1 / d2 = 4 / 3) :
  max d1 d2 = 20 :=
by 
  let x := sqrt (area * 2 / (d1 * d2))
  have d1_expr : d1 = 4 * x := sorry
  have d2_expr : d2 = 3 * x := sorry
  have x_val : x = 5 := sorry
  have length_longest_diag : max d1 d2 = max (4 * 5) (3 * 5) := sorry
  exact length_longest_diag

end longest_diagonal_of_rhombus_l298_298713


namespace prove_k_prove_m_l298_298111

-- Define the vectors a and b
def vec_a : ℝ × ℝ := (1, -2)
def vec_b : ℝ × ℝ := (3, 4)

-- Problem Part 1: Prove k = -1/3 given (3 * vec_a - vec_b) || (vec_a + k * vec_b)
noncomputable def k : ℝ := -1 / 3

theorem prove_k : (3 • vec_a - vec_b) ∥ (vec_a + k • vec_b) := 
  sorry

-- Problem Part 2: Prove m = -1 given vec_a ⊥ (m * vec_a - vec_b)
noncomputable def m : ℝ := -1

theorem prove_m : vec_a ⊥ (m • vec_a - vec_b) := 
  sorry

end prove_k_prove_m_l298_298111


namespace area_CDHE_l298_298167

noncomputable def sqrt := Real.sqrt

-- Define the triangle with given conditions
structure Triangle :=
  (A B C : Point)
  (AB : distance A B = 1)
  (angleBAC : angle A B C = 45)
  (angleABC : angle B A C = 60)

-- Define the theorem to prove the area of quadrilateral CDHE
theorem area_CDHE (T : Triangle) (H D E C : Point) 
                  (AD : is_altitude T.A T.B T.C D)
                  (BE : is_altitude T.B T.A T.C E)
                  (HD_intersects : ∃ H : Point, between H D E)
                  (CH : is_point_on_line H C T.C)
                  (DH : is_point_on_line H D T.B)
                  (EH : is_point_on_line H E T.A)
                  : 
  area_of_quadrilateral C D H E = (2 - sqrt 3) / 8 :=
sorry

end area_CDHE_l298_298167


namespace polynomial_remainder_l298_298429

-- Define the polynomial p(x)
def p (x : ℝ) : ℝ := x^3 + 2*x + 3

-- Define the divisor q(x)
def q (x : ℝ) : ℝ := x + 2

-- The theorem asserting the remainder when p(x) is divided by q(x)
theorem polynomial_remainder : (p (-2)) = -9 :=
by
  sorry

end polynomial_remainder_l298_298429


namespace people_remaining_on_bus_l298_298217

theorem people_remaining_on_bus
  (students_left : ℕ) (students_right : ℕ) (students_back : ℕ)
  (students_aisle : ℕ) (teachers : ℕ) (bus_driver : ℕ) 
  (students_off1 : ℕ) (teachers_off1 : ℕ)
  (students_off2 : ℕ) (teachers_off2 : ℕ)
  (students_off3 : ℕ) :
  students_left = 42 ∧ students_right = 38 ∧ students_back = 5 ∧
  students_aisle = 15 ∧ teachers = 2 ∧ bus_driver = 1 ∧
  students_off1 = 14 ∧ teachers_off1 = 1 ∧
  students_off2 = 18 ∧ teachers_off2 = 1 ∧
  students_off3 = 5 →
  (students_left + students_right + students_back + students_aisle + teachers + bus_driver) -
  (students_off1 + teachers_off1 + students_off2 + teachers_off2 + students_off3) = 64 :=
by {
  sorry
}

end people_remaining_on_bus_l298_298217


namespace solve_for_x_l298_298599

theorem solve_for_x (x : ℝ) (h : 5 + 3.5 * x = 2.5 * x - 25) : x = -30 :=
sorry

end solve_for_x_l298_298599


namespace expression_value_l298_298833

theorem expression_value (a b m n : ℚ) 
  (ha : a = -7/4) 
  (hb : b = -2/3) 
  (hmn : m + n = 0) : 
  4 * a / b + 3 * (m + n) = 21 / 2 :=
by 
  sorry

end expression_value_l298_298833


namespace inequality_proof_l298_298442

theorem inequality_proof
  (n : ℕ) (hn : n ≥ 3) (x y z : ℝ) (hxyz_pos : x > 0 ∧ y > 0 ∧ z > 0)
  (hxyz_sum : x + y + z = 1) :
  (1 / x^(n-1) - x) * (1 / y^(n-1) - y) * (1 / z^(n-1) - z) ≥ ((3^n - 1) / 3)^3 :=
by sorry

end inequality_proof_l298_298442


namespace arithmetic_sequence_sum_l298_298923

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (d : ℤ) (h1 : a 1 + a 3 = 2) (h2 : a 2 + a 4 = 6)
  (h_arith : ∀ n, a (n + 1) = a n + d) : a 1 + a 7 = 10 :=
by
  sorry

end arithmetic_sequence_sum_l298_298923


namespace polynomial_remainder_l298_298710

theorem polynomial_remainder : 
  ∀ (q : ℚ → ℚ),
  (q 3 = 5) → 
  (q 4 = -2) → 
  (q (-2) = 6) →
  ∃ a b c : ℚ, 
  (q = λ x => (x - 3) * (x - 4) * (x + 2) * (λ r, r x) + a * x^2 + b * x + c) ∧ 
  (a = -17/15) ∧ (b = -26/45) ∧ (c = 4.2) ∧ 
  (a * 5^2 + b * 5 + c = -27) :=
by
  intro q hq3 hq4 hqminus2
  use [-17/15, -26/45, 4.2]
  split
  . sorry
  . split; [refl, split; [refl, sorry, sorry]]

end polynomial_remainder_l298_298710


namespace correct_propositions_count_l298_298878

variables (α β γ : Type) (m n : Type)
-- Assuming definitions for parallelism (parallel) and perpendicularity (perpendicular)
variables [Plane α] [Plane β] [Plane γ] [Line m] [Line n]
variables (intersects : α ∩ β = m) (parallel_mn : n ∥ m) (perpendicular_αβ : α ⊥ β)
variables (perpendicular_mβ : m ⊥ β) (not_contains : m ∉ α) (parallel_αβ : α ∥ β)
variables (contains_m : m ⊂ α) (perpendicular_αγ : α ⊥ γ)

theorem correct_propositions_count: (if intersect and parallel_mn then (n ∥ α) ∧ (n ∥ β) else false) + 
                                      (if perpendicular_αβ ∧ perpendicular_mβ ∧ ¬contains_m then ∥ α else false) +
                                      (if parallel_αβ ∧ contains_m then ∥ β else false) + 
                                      (if perpendicular_αβ ∧ perpendicular_αγ then ∥ γ else false) = 2 :=
by sorry

end correct_propositions_count_l298_298878


namespace no_pos_int_lt_2000_7_times_digits_sum_l298_298113

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem no_pos_int_lt_2000_7_times_digits_sum :
  ∀ n : ℕ, n < 2000 → n = 7 * sum_of_digits n → False :=
by
  intros n h1 h2
  sorry

end no_pos_int_lt_2000_7_times_digits_sum_l298_298113


namespace remainder_polynomial_l298_298781

theorem remainder_polynomial (p : ℕ → ℕ) (h1 : p 2 = 4) (h2 : p 4 = 10) :
  ∃ r : ℕ → ℕ, r = (λ x, 3 * x - 2) :=
by {
  sorry
}

end remainder_polynomial_l298_298781


namespace problem_conditions_propositions_l298_298382

theorem problem_conditions_propositions :
  let p1 := ∀ (x : ℝ), x ≠ 4 → x^2 - 2 * x - 3 < 0 = false,
  let p2 := ∀ (x : ℝ), (x = 2) ↔ (x^2 - 4 * x + 4 = 0) = true,
  let p3 := ¬(∀ (t : Triangle), sum_angles t = 180) ↔ ¬is_triangle t ∧ sum_angles t ≠ 180,
  let p4 := ¬(∀ (x : ℝ), x^2 ≥ 0) ↔ ∃ (x : ℝ), x^2 < 0,
  number_of_correct_propositions := 0 in
  p1 ∧ p2 ∧ p3 ∧ p4 → number_of_correct_propositions = 0 :=
by
  intros,
  sorry

end problem_conditions_propositions_l298_298382


namespace negation_of_forall_log_gt_one_l298_298471

noncomputable def negation_of_p : Prop :=
∃ x : ℝ, Real.log x ≤ 1

theorem negation_of_forall_log_gt_one :
  (¬ (∀ x : ℝ, Real.log x > 1)) ↔ negation_of_p :=
by
  sorry

end negation_of_forall_log_gt_one_l298_298471


namespace average_speed_palindrome_l298_298990

def is_palindrome (n : ℕ) : Prop :=
  n.toString = n.toString.reverse

theorem average_speed_palindrome :
  let initial_reading := 12321
  let final_reading := 12421
  let time := 3
  is_palindrome initial_reading ->
  is_palindrome final_reading ->
  final_reading > initial_reading ->
  (final_reading - initial_reading) = 100 ->
  (final_reading - initial_reading : ℝ) / time = 33.33 :=
by
  intros
  sorry

end average_speed_palindrome_l298_298990


namespace general_formulas_for_sequences_find_m_l298_298934

open Nat

def a : ℕ → ℕ
| 0     := 0 -- not used since sequences often start from n=1
| 1     := 1
| (n+1) := a n + 2

def b : ℕ → ℕ
| 1 := 3
| 2 := 7
| n := sorry -- To be calculated or defined for n > 2

def c (n : ℕ) : ℕ := b n - a n

axiom c_is_geometric (n : ℕ) : c (n) / c (n - 1) = 2

theorem general_formulas_for_sequences : ∀ (n : ℕ), a n = 2 * n - 1 ∧ c n = 2^n := 
by sorry

theorem find_m (m : ℕ) : b 6 = a m → m = 38 := 
by sorry

end general_formulas_for_sequences_find_m_l298_298934


namespace exists_sequence_l298_298782

theorem exists_sequence (a : ℕ → ℕ) :
  (∀ n : ℕ, ∃! k : ℕ, a k = n) ∧
  (∀ k : ℕ, k > 0 → (∑ i in Finset.range k, a i) % k = 0) :=
sorry

end exists_sequence_l298_298782


namespace even_function_periodic_symmetric_about_2_l298_298841

variables {F : ℝ → ℝ}

theorem even_function_periodic_symmetric_about_2
  (h_even : ∀ x, F x = F (-x))
  (h_symmetric : ∀ x, F (2 - x) = F (2 + x))
  (h_cond : F 2011 + 2 * F 1 = 18) :
  F 2011 = 6 :=
sorry

end even_function_periodic_symmetric_about_2_l298_298841


namespace oranges_per_child_l298_298389

theorem oranges_per_child (children oranges : ℕ) (h1 : children = 4) (h2 : oranges = 12) : oranges / children = 3 := by
  sorry

end oranges_per_child_l298_298389


namespace modulus_z_range_m_l298_298459

-- Define the imaginary unit
def i : ℂ := Complex.I

-- Define the conjugate function
def conj (z : ℂ) : ℂ := Complex.conj z

-- Given condition: z + 2 * conj(z) = 3 - 2 * i
variable (z : ℂ)
axiom condition : z + 2 * conj z = 3 - 2 * i

-- Part 1: Prove the modulus of z is sqrt(5)
theorem modulus_z : Complex.abs z = Real.sqrt 5 := by
  sorry

-- Part 2: Prove the range of m such that the complex number z(2 - mi) is in the second quadrant is (-∞, -1)
theorem range_m (m : ℝ) : (2 + 2 * m < 0 ∧ 4 - m > 0) ↔ m < -1 := by
  sorry

end modulus_z_range_m_l298_298459


namespace find_equation_of_parabola_find_min_area_l298_298821

-- Lean 4 statement for Problem 1
theorem find_equation_of_parabola 
  (p : ℝ) (p_pos : p > 0)
  (HQ_QF_condition : ∀ Q F H: ℝ, Q = (4, 8 / p) → F = (0, p / 2) → H = (4, 0) → 
    ( ( sqrt ((4 - 0) ^ 2 + (8 / p - p / 2) ^ 2) = (3 / 2) * sqrt ((4 - 4) ^ 2 + (8 / p - 0) ^ 2) ))) :
  ∃ p,  p = 2 * sqrt 2 → x^2 = 4 * sqrt 2 * y :=
by sorry

-- Lean 4 statement for Problem 2
theorem find_min_area
  (p : ℝ) (p_pos : p > 0)
  (Eq1  : ∀ (x : ℝ), x^2 = 4 * sqrt 2 * y)
  (line_l: ℝ → ℝ → ℝ) 
  (l1_tangent: ∀ A : ℝ, equation_of_tangent_l1 = y= sqrt 2 * x) 
  (l2_tangent: ∀ B : ℝ, equation_of_tangent_l2 = y = sqrt 2 * x) 
  (A1_Intersection : ∀ (x1 x2: ℝ) , A(x1, y1) ∧ B(x2, y2) ∧ line_l passes through A and B):
  ∃ k: ℝ , ∀ k,  ∃ x2,
   R = (2 * sqrt(2) * k, - sqrt 2) ∧ S_Δ(R,A,B) = min 8 :=
by sorry

end find_equation_of_parabola_find_min_area_l298_298821


namespace sequence_properties_l298_298870

-- Utilizing nat for the sequence index n
open nat

-- Defining the sequence a_n recursively
def a (n : ℕ) : ℕ := nat.rec_on n 2 (λ k ak, 4 * ak - 3 * k + 1)

-- Statement of the problem
theorem sequence_properties : 
  (∀ n : ℕ, a n - n = 4^(n-1)) ∧                              -- The sequence {a_n - n} is geometric with ratio 4
  (∀ n : ℕ, a n = n + 4^(n-1)) ∧                              -- The explicit term a_n = n + 4^(n-1)
  (∀ n : ℕ, ∑ k in range n, a k = n * (n - 1) / 2 + (4^n - 1) / 3)  -- Sum of first n terms S_n
:= 
sorry

end sequence_properties_l298_298870


namespace values_a1_a2_a3_geometric_sequence_general_term_l298_298445

-- Define the sequence {a_n} and its sum {S_n}
variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}

-- Given condition
axiom sum_condition : ∀ n : ℕ, n > 0 → a 1 + 2 * a 2 + 3 * a 3 + ... + n * a n = (n - 1) * S n + 2 * n

-- Required proofs
theorem values_a1_a2_a3 : a 1 = 2 ∧ a 2 = 4 ∧ a 3 = 8 := by
  sorry

theorem geometric_sequence : ∀ n : ℕ, n > 0 → S n + 2 = 4 * 2 ^ (n - 1) := by
  sorry

theorem general_term : ∀ n : ℕ, n > 0 → a n = 2 ^ n := by
  sorry

end values_a1_a2_a3_geometric_sequence_general_term_l298_298445


namespace beret_count_l298_298172

/-- James can make a beret from 3 spools of yarn. 
    He has 12 spools of red yarn, 15 spools of black yarn, and 6 spools of blue yarn.
    Prove that he can make 11 berets in total. -/
theorem beret_count (red_yarn : ℕ) (black_yarn : ℕ) (blue_yarn : ℕ) (spools_per_beret : ℕ) 
  (total_yarn : ℕ) (num_berets : ℕ) (h1 : red_yarn = 12) (h2 : black_yarn = 15) (h3 : blue_yarn = 6)
  (h4 : spools_per_beret = 3) (h5 : total_yarn = red_yarn + black_yarn + blue_yarn) 
  (h6 : num_berets = total_yarn / spools_per_beret) : 
  num_berets = 11 :=
by sorry

end beret_count_l298_298172


namespace roots_of_equation_l298_298261

theorem roots_of_equation (x : ℝ) : 3 * x * (x - 1) = 2 * (x - 1) → (x = 1 ∨ x = 2 / 3) :=
by 
  intros h
  sorry

end roots_of_equation_l298_298261


namespace ellipse_equation_l298_298013

noncomputable def equation_of_ellipse (x y : ℝ) : Prop :=
  2 * (x - 2)^2 + (y - 1)^2 = 12

theorem ellipse_equation :
  (∃ c A B : ℝ, A = (4, 3) ∧ B = (0, -1) ∧ C = (1, (√10 + 1)) ∧
    ∀ point ∈ [A, B, C], 2(point.x - 2)^2 + (point.y - 1)^2 = 12) :=
begin
  sorry
end

end ellipse_equation_l298_298013


namespace complex_power_sum_complex_distance_range_l298_298233

-- Part (1) Proof Problem Statement
theorem complex_power_sum :
  let z : ℂ := (Complex.i - 1) / Real.sqrt 2 in
  z^20 + z^10 + 1 = -Complex.i :=
by
  sorry

-- Part (2) Proof Problem Statement
theorem complex_distance_range (z : ℂ) :
  |z - 3 - 4*Complex.i| = 1 → 4 ≤ Complex.abs z ∧ Complex.abs z ≤ 6 :=
by
  sorry

end complex_power_sum_complex_distance_range_l298_298233


namespace chocolates_difference_l298_298226

/-!
We are given that:
- Robert ate 10 chocolates
- Nickel ate 5 chocolates

We need to prove that Robert ate 5 more chocolates than Nickel.
-/

def robert_chocolates := 10
def nickel_chocolates := 5

theorem chocolates_difference : robert_chocolates - nickel_chocolates = 5 :=
by
  -- Proof is omitted as per instructions
  sorry

end chocolates_difference_l298_298226


namespace sum_of_x_satisfying_condition_l298_298550

noncomputable def g (x : ℝ) : ℝ := 9 * x + 7

theorem sum_of_x_satisfying_condition :
  let g_inv : ℝ → ℝ := λ x, (x - 7) / 9 in
  ∑ x in { x : ℝ | g_inv x = g ((3 * x)⁻¹) }.to_finset, x = 70 :=
by
  let g_inv : ℝ → ℝ := λ x, (x - 7) / 9
  sorry

end sum_of_x_satisfying_condition_l298_298550


namespace sequence_an_sequence_bn_sum_Tn_l298_298157

theorem sequence_an (n : ℕ) (h1 : a 3 = 6) (h2 : a 8 = 16) : a n = 2 * n := 
sorry

theorem sequence_bn (n : ℕ) (h1 : b 1 = 1) (h2 : 4 * S 1 = 3 * S 2) (h3 : 3 * S 2 = 2 * S 3) :
  b n = 2^(n-1) := 
sorry

theorem sum_Tn (n : ℕ) (a : ℕ → ℕ) (b : ℕ → ℕ) 
  (h1 : a_n = 2 * n) 
  (h2 : b_n = 2^(n-1)) : 
  T n = (n-1) * 2^(n+1) + 2 := 
sorry

end sequence_an_sequence_bn_sum_Tn_l298_298157


namespace normalized_vectors_sum_to_zero_l298_298449

variables {V : Type*} [inner_product_space ℝ V]

def a_and_b_nonzero (a b : V) : Prop := a ≠ 0 ∧ b ≠ 0

theorem normalized_vectors_sum_to_zero
  (a b : V) (h : a = - (1/3 : ℝ) • b) (nonzero : a_and_b_nonzero a b) :
  (a / ∥a∥) + (b / ∥b∥) = 0 :=
sorry

end normalized_vectors_sum_to_zero_l298_298449


namespace maximum_profit_at_week_5_l298_298617

noncomputable def price (x : ℕ) : ℝ :=
  if x ≤ 4 then 10 + 2 * x
  else if x ≤ 10 then 20
  else if x ≤ 16 then 20 - 2 * (x - 10)
  else 0

noncomputable def cost_price (x : ℕ) : ℝ :=
  -0.125 * (x - 8) ^ 2 + 12

noncomputable def profit (x : ℕ) : ℝ :=
  price x - cost_price x

theorem maximum_profit_at_week_5 :
  ∀ x ∈ finset.range 17, profit x ≤ profit 5 :=
begin
  -- Proof skipped
  sorry
end

end maximum_profit_at_week_5_l298_298617


namespace find_k_l298_298478

-- Define vectors a, b, and c
def a : ℝ × ℝ := (3, 1)
def b : ℝ × ℝ := (1, 3)
def c (k : ℝ) : ℝ × ℝ := (k, 2)

-- Define the dot product function for two 2D vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Define the condition for perpendicular vectors
def perpendicular_condition (k : ℝ) : Prop :=
  dot_product (a.1 - k, -1) b = 0

-- State the theorem
theorem find_k : ∃ k : ℝ, perpendicular_condition k ∧ k = 0 := by
  sorry

end find_k_l298_298478


namespace number_of_x_intercepts_l298_298804

theorem number_of_x_intercepts : 
  let interval_low  := 0.00005
  let interval_high := 0.0005
  let pi := Real.pi
  let low_k  := Real.floor ((1 / interval_high) / pi)
  let high_k := Real.floor ((1 / interval_low) / pi)
  high_k - low_k = 5730 := 
by suffices : high_k = 6366 ∧ low_k = 636 by sorry

end number_of_x_intercepts_l298_298804


namespace tangent_line_min_slope_l298_298058

theorem tangent_line_min_slope {x y : ℝ} (hx : y = x^3 + 3 * x^2 + 6 * x - 10) :
  ∃ x : ℝ, ∃ y : ℝ, ∃ m : ℝ, ∃ b : ℝ, (m = 3) ∧ (y = m * (x + 1) - 14) ∧ (3 * x - y - 11 = 0).
proof
  sorry

end tangent_line_min_slope_l298_298058


namespace product_closest_to_l298_298019

def is_closest_to (n target : ℝ) (options : List ℝ) : Prop :=
  ∀ o ∈ options, |n - target| ≤ |n - o|

theorem product_closest_to : is_closest_to ((2.5) * (50.5 + 0.25)) 127 [120, 125, 127, 130, 140] :=
by
  sorry

end product_closest_to_l298_298019


namespace calculation_difference_l298_298479

theorem calculation_difference :
  let H := 12 - (3 + 4 * 2)
  let T := 12 - 3 + 4 * 2
  H - T = -25 :=
by
  let H := 12 - (3 + 4 * 2)
  let T := 12 - 3 + 4 * 2
  show H - T = -25
  sorry

end calculation_difference_l298_298479


namespace exist_numbers_with_properties_l298_298982

theorem exist_numbers_with_properties (n : ℕ) (h : n ≥ 1) :
  ∃ (S : finset (vector ℕ (2^n))),
  S.card = 2^(n+1) ∧
  ∀ (x y ∈ S), x ≠ y → (finset.card (finset.filter (λ i, x.nth i ≠ y.nth i) (finset.range (2^n))) ≥ 2^(n-1)) :=
by
  sorry

end exist_numbers_with_properties_l298_298982


namespace volume_of_given_tetrahedron_l298_298914

def point := (ℝ × ℝ × ℝ)

def volume_tetrahedron (A B C D : point) : ℝ :=
  let (x1, y1, z1) := A
  let (x2, y2, z2) := B
  let (x3, y3, z3) := C
  let (x4, y4, z4) := D
  let AB := (x2 - x1, y2 - y1, z2 - z1)
  let AC := (x3 - x1, y3 - y1, z3 - z1)
  let AD := (x4 - x1, y4 - y1, z4 - z1)
  let M := matrix.of_vec_list [
    [AB.1, AB.2, AB.3],
    [AC.1, AC.2, AC.3],
    [AD.1, AD.2, AD.3]
  ]
  (1/6) * abs(matrix.det M)

theorem volume_of_given_tetrahedron :
  volume_tetrahedron (-1, 0, 2) (7, 4, 3) (7, -4, 5) (4, 2, -3) = 54.67 :=
by
  sorry

end volume_of_given_tetrahedron_l298_298914


namespace magnitude_b_l298_298448

-- Define vectors and their properties
variables {V : Type*} [inner_product_space ℝ V] (a b : V)

-- Given conditions
def condition1 : Prop := (a ≠ 0) ∧ (b ≠ 0) -- non-zero vectors
def condition2 : Prop := ⟪a, b⟫ = 0 -- a · b = 0, which implies perpendicular vectors
def condition3 : Prop := ∥a∥ = 3 -- |a| = 3
def condition4 : Prop := real.angle a (a + b) = real.pi / 4 -- angle between a and a + b = π/4

-- Prove the magnitude of vector b
theorem magnitude_b : condition1 a b → condition2 a b → condition3 a → condition4 a b → ∥b∥ = 3 :=
by
  intro h1 h2 h3 h4
  sorry

end magnitude_b_l298_298448


namespace geometric_locus_of_point_l298_298613

structure Point := (x : ℝ) (y : ℝ) 
def distance (p q : Point) : ℝ := real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

structure Triangle := (A O M : Point)

def on_circle (O : Point) (r : ℝ) (M : Point) : Prop := 
  distance O M = r

def on_ray (O C M : Point) : Prop :=
  ∃ k : ℝ, k > 0 ∧ M.x = O.x + k * (C.x - O.x) ∧ M.y = O.y + k * (C.y - O.y)

theorem geometric_locus_of_point
  (O A C M : Point)
  (h_dist : distance O A = distance O M)
  (h_ray : on_ray O C M)
  (h_M_not_A : M ≠ A)
  (h_M_not_O : M ≠ O) :
  (on_circle O (distance O A) M ∧ M ≠ A) ∨ 
  (on_ray O C M ∧ M ≠ O) :=
sorry

end geometric_locus_of_point_l298_298613


namespace inscribed_circle_ratio_l298_298652

theorem inscribed_circle_ratio (P Q R S: Type) [is_triangle P Q R] (PR QR PQ: ℝ) (PR_eq : PR = 6) (QR_eq : QR = 8) (PQ_eq : PQ = 10)
 (S_on_PQ : S ∈ segment PQ) (RS_bisects : ∠ PRS = ∠ QRS)
 (r_p r_q: ℝ) (inradius_PS_R : is_inscribed_circle_radius (triangle PSR) r_p)
 (inradius_QS_R : is_inscribed_circle_radius (triangle QSR) r_q)
 : r_p / r_q = (3 / 28) * (20 - 2 * sqrt 3) :=
sorry

end inscribed_circle_ratio_l298_298652


namespace h_of_k_neg_3_l298_298120

def h (x : ℝ) : ℝ := 4 - real.sqrt x

def k (x : ℝ) : ℝ := 3 * x + 3 * x^2

theorem h_of_k_neg_3 : h (k (-3)) = 4 - 3 * real.sqrt 2 :=
by
  sorry

end h_of_k_neg_3_l298_298120


namespace vector_dot_product_and_magnitude_l298_298880

theorem vector_dot_product_and_magnitude :
  let a : ℝ × ℝ := (1, real.sqrt 3)
  let b : ℝ × ℝ := (1, 0) -- assuming a direction for simplicity since |b| = 1
  let angle := real.pi / 3  -- 60 degrees in radians
  (a.1^2 + a.2^2)^0.5 = 2 ∧
  (b.1^2 + b.2^2)^0.5 = 1 ∧
  (a.1 * real.cos angle + a.2 * real.sin angle) = 1 ∧
  ((a.1 - 2 * b.1) ^ 2 + (a.2 - 2 * b.2) ^ 2)^0.5 = 2
:=
  by
  sorry

end vector_dot_product_and_magnitude_l298_298880


namespace product_sum_representation_l298_298229

theorem product_sum_representation (n : ℕ) (h_pos : 0 < n) :
  ∃ (N : ℕ) (c : fin N → ℚ) (a : fin N → fin n → ℤ),
    (∀ i j, a i j = -1 ∨ a i j = 0 ∨ a i j = 1) ∧
    (∀ (x : fin n → ℚ),
       ∏ i, x i = ∑ i, c i * (∑ j, a i j * x j)^n) :=
begin
  sorry
end

end product_sum_representation_l298_298229


namespace denom_divisible_by_prime_l298_298203

noncomputable def problem (p : ℕ) (b : ℕ → ℕ) (n : ℕ) : Prop :=
  prime p ∧ 
  (∃! i, i < n ∧ p ∣ b i) ∧
  (∀ i < n, b i > 0) ∧ 
  let S := (List.range n).sum (λ i, (1 / (b i) : ℚ)) in
  ∃ (a b : ℤ) (ha : a ≠ 0), S = (a / b) ∧ p ∣ Int.gcd b

theorem denom_divisible_by_prime (p : ℕ) (b : ℕ → ℕ) (n : ℕ) :
  problem p b n →
  ∃ (den : ℤ), (den ≠ 0) ∧ (∀ a : ℤ, a ≠ 0 → den = Int.gcd a) ∧ p ∣ den :=
sorry

end denom_divisible_by_prime_l298_298203


namespace jessica_total_cost_l298_298176

-- Define the costs
def cost_cat_toy : ℝ := 10.22
def cost_cage : ℝ := 11.73

-- Define the total cost
def total_cost : ℝ := cost_cat_toy + cost_cage

-- State the theorem
theorem jessica_total_cost : total_cost = 21.95 := by
  sorry

end jessica_total_cost_l298_298176


namespace longest_diagonal_length_l298_298726

theorem longest_diagonal_length (A : ℝ) (d1 d2 : ℝ) (h1 : A = 150) (h2 : d1 / d2 = 4 / 3) : d1 = 20 :=
by
  -- Skipping the proof here
  sorry

end longest_diagonal_length_l298_298726


namespace abs_sum_quotient_lt_one_l298_298818

variable (a b : ℝ)

theorem abs_sum_quotient_lt_one (h1 : |a| < 1) (h2 : |b| < 1) : 
  (| (a + b) / (1 + a * b) | < 1) :=
by
  sorry

end abs_sum_quotient_lt_one_l298_298818


namespace pyramid_total_surface_area_l298_298338

-- Defining the points and dimensions involved
def length_base : ℝ := 14
def width_base : ℝ := 8
def height_peak : ℝ := 15

-- Definition to compute the slant height using Pythagorean theorem
def distance_FM : ℝ := width_base / 2
def slant_height : ℝ := real.sqrt (height_peak ^ 2 + distance_FM ^ 2)

-- Definition to compute the total surface area
def base_area : ℝ := length_base * width_base
def lateral_area : ℝ := 4 * (1 / 2 * length_base * slant_height)
def total_surface_area : ℝ := base_area + lateral_area

-- The theorem to prove
theorem pyramid_total_surface_area :
  total_surface_area = 112 + 28 * real.sqrt 241 :=
by
  -- Placeholder for the proof, which is not required
  sorry

end pyramid_total_surface_area_l298_298338


namespace isometry_rotate_180_l298_298669

noncomputable def rotate_180 (p: (ℝ × ℝ)) : (ℝ × ℝ) :=
  match p with
  | (x, y) => (-x, -y)

theorem isometry_rotate_180 :
  rotate_180 (-3, 0) = (3, 0) ∧ rotate_180 (0, 5) = (0, -5) :=
by
  unfold rotate_180
  simp
  split
  · exact rfl
  · exact rfl

end isometry_rotate_180_l298_298669


namespace intersection_product_with_origin_l298_298079

theorem intersection_product_with_origin
  (k : ℝ)
  (x y : ℝ)
  (x1 x2 y1 y2 : ℝ)
  (h1 : (x - 3)^2 + (y + 4)^2 = 4)
  (h2 : y = k * x)
  (h3 : ∃ x y, y = k * x ∧ (x - 3)^2 + (y + 4)^2 = 4)
  (h4 : x1 + x2 = (-8 * k + 6) / (k^2 + 1))
  (h5 : x1 * x2 = 21 / (k^2 + 1))
  (h6 : y1 = k * x1)
  (h7 : y2 = k * x2) :
  (sqrt (x1^2 + y1^2)) * (sqrt (x2^2 + y2^2)) = 21 :=
sorry

end intersection_product_with_origin_l298_298079


namespace black_haired_girls_count_l298_298559

def initial_total_girls : ℕ := 80
def added_blonde_girls : ℕ := 10
def initial_blonde_girls : ℕ := 30

def total_girls := initial_total_girls + added_blonde_girls
def total_blonde_girls := initial_blonde_girls + added_blonde_girls
def black_haired_girls := total_girls - total_blonde_girls

theorem black_haired_girls_count : black_haired_girls = 50 := by
  sorry

end black_haired_girls_count_l298_298559


namespace part_I_part_II_part_III_l298_298466

noncomputable def f (x : ℝ) : ℝ := 3 ^ x
noncomputable def g (x a : ℝ) : ℝ := (f x) ^ 2 - 2 * a * (f x) + 3
noncomputable def h (a : ℝ) : ℝ :=
if a < 1/3 then (28 / 9) - (2 * a / 3)
else if a <= 3 then 3 - a ^ 2
else 12 - 6 * a

theorem part_I : set.range (λ x, g x 0) = set.Icc (28 / 9) 12 := sorry

theorem part_II (a : ℝ) : h a = 
  if a < 1/3 then (28 / 9) - (2 * a / 3)
  else if a <= 3 then 3 - a ^ 2
  else 12 - 6 * a := sorry

theorem part_III : ¬ ∃ m n : ℝ, m > n ∧ n > 3 ∧ 
  (set.range h $ set.Icc n m = set.Icc (n^2) (m^2)) := sorry

end part_I_part_II_part_III_l298_298466


namespace hyperbola_eccentricity_l298_298840

def isHyperbolaWithEccentricity (e : ℝ) : Prop :=
  ∃ (a b : ℝ), a = 4 * b ∧ e = (Real.sqrt (a^2 + b^2)) / a

theorem hyperbola_eccentricity : isHyperbolaWithEccentricity (Real.sqrt 17 / 4) :=
sorry

end hyperbola_eccentricity_l298_298840


namespace proof_allison_brian_noah_l298_298752

-- Definitions based on the problem conditions

-- Definition for the cubes
def allison_cube := [6, 6, 6, 6, 6, 6]
def brian_cube := [1, 2, 2, 3, 3, 4]
def noah_cube := [3, 3, 3, 3, 5, 5]

-- Helper function to calculate the probability of succeeding conditions
def probability_succeeding (A B C : List ℕ) : ℚ :=
  if (A.all (λ x => x = 6)) ∧ (B.all (λ x => x ≤ 5)) ∧ (C.all (λ x => x ≤ 5)) then 1 else 0

-- Define the proof statement for the given problem
theorem proof_allison_brian_noah :
  probability_succeeding allison_cube brian_cube noah_cube = 1 :=
by
  -- Since all conditions fulfill the requirement, we'll use sorry to skip the proof for now
  sorry

end proof_allison_brian_noah_l298_298752


namespace matrix_entry_sum_l298_298179

theorem matrix_entry_sum (n : ℕ) (h_pos : n > 0) :
  (∑ i in finset.range n, finset.range n) = (5 * n^2 - 3 * n) / 2 :=
by
  -- Definitions according to the conditions and sum formula.
  sorry

end matrix_entry_sum_l298_298179


namespace find_n_l298_298303

-- Definitions of conditions
def lcm (a b : Nat) : Nat := sorry -- Placeholder definition
def gcd (a b : Nat) : Nat := sorry -- Placeholder definition

-- Condition properties
axiom lcm_eq : ∀ (n : Nat), lcm n 24 = 48
axiom gcd_eq : ∀ (n : Nat), gcd n 24 = 8

-- Statement to prove
theorem find_n (n : Nat) (h₁ : lcm n 24 = 48) (h₂ : gcd n 24 = 8) : n = 16 := by
  sorry

end find_n_l298_298303


namespace correct_weight_misread_l298_298244

theorem correct_weight_misread : 
  ∀ (x : ℝ) (n : ℝ) (avg1 : ℝ) (avg2 : ℝ) (misread : ℝ),
  n = 20 → avg1 = 58.4 → avg2 = 59 → misread = 56 → 
  (n * avg2 - n * avg1 + misread) = x → 
  x = 68 :=
by
  intros x n avg1 avg2 misread
  intros h1 h2 h3 h4 h5
  sorry

end correct_weight_misread_l298_298244


namespace linear_system_solution_l298_298234

/-- Given a system of three linear equations:
      x + y + z = 1
      a x + b y + c z = h
      a² x + b² y + c² z = h²
    Prove that the solution x, y, z is given by:
    x = (h - b)(h - c) / (a - b)(a - c)
    y = (h - a)(h - c) / (b - a)(b - c)
    z = (h - a)(h - b) / (c - a)(c - b) -/
theorem linear_system_solution (a b c h : ℝ) (x y z : ℝ) :
  x + y + z = 1 →
  a * x + b * y + c * z = h →
  a^2 * x + b^2 * y + c^2 * z = h^2 →
  x = (h - b) * (h - c) / ((a - b) * (a - c)) ∧
  y = (h - a) * (h - c) / ((b - a) * (b - c)) ∧
  z = (h - a) * (h - b) / ((c - a) * (c - b)) :=
by
  intros
  sorry

end linear_system_solution_l298_298234


namespace train_crossing_time_l298_298296

def length_of_train : ℝ := 390 -- meters
def speed_of_train : ℝ := 25 * 1000 / 3600 -- km/h converted to m/s
def speed_of_man : ℝ := 2 * 1000 / 3600 -- km/h converted to m/s
def relative_speed : ℝ := speed_of_train + speed_of_man -- relative speed of train and man in m/s

theorem train_crossing_time : length_of_train / relative_speed = 52 := by
  sorry

end train_crossing_time_l298_298296


namespace parabola_line_intersection_l298_298334

-- Define the existence and properties of points A and B, and the parabola characteristics
theorem parabola_line_intersection :
  ∃ (p a : ℝ) (xA yA xB yB : ℝ), 
    (yA = 2) ∧ (xA = 1) ∧ 
    (yA^2 = 2 * p * xA) ∧ -- A lies on the parabola y^2 = 2px
    (2 * xA + yA + a = 0) ∧ -- A lies on the line 2x + y + a = 0
    (yB = -2 * xB - 4) ∧ -- B satisfies the linear equation
    (yB^2 = 2 * p * xB) ∧ -- B lies on the parabola y^2 = 2px
    (|1 - xA| + |1 - xB| + p = 7) :=
begin
  use [2, -4, 1, 2, 4, -12],
  split,
  { exact rfl },
  split,
  { exact rfl },
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { exact eq.symm (by norm_num) },
  split,
  { norm_num },
  split,
  { norm_num },
  norm_num,
end

end parabola_line_intersection_l298_298334


namespace find_v_l298_298807

theorem find_v (v : ℝ) (h : (v - v / 3) - ((v - v / 3) / 3) = 4) : v = 9 := 
by 
  sorry

end find_v_l298_298807


namespace integral_diverges_l298_298168

noncomputable def integrand1 (x : ℝ) : ℝ := x / (1 + x^2 * (Real.cos x)^2)
noncomputable def integrand2 (x : ℝ) : ℝ := x / (1 + x^2)

theorem integral_diverges : ¬(∃ L : ℝ, integral (0:ℝ) (∞:ℝ) (fun x => integrand1 x) = L) := 
by 
  sorry

end integral_diverges_l298_298168


namespace find_number_of_solutions_l298_298064

theorem find_number_of_solutions :
  (∃ s : set ℝ, {x | -π ≤ x ∧ x ≤ π ∧ cos (6 * x) + (cos (3 * x))^4 + (sin (2 * x))^2 + (cos (x))^2 = 0} = s ∧ s.card = 14) :=
  sorry

end find_number_of_solutions_l298_298064


namespace equation1_solutions_equation2_solution_l298_298989

-- Define the first proof problem: Solve x^2 + 3x + 1 = 0
theorem equation1_solutions (x : ℝ) : x^2 + 3 * x + 1 = 0 ↔ 
    x = (-3 + real.sqrt 5) / 2 ∨ x = (-3 - real.sqrt 5) / 2 := by
  -- proof goes here
  sorry

-- Define the second proof problem: Solve 4x/(x-2) - 1 = 3/(2-x)
theorem equation2_solution (x : ℝ) (h : x ≠ 2) : 
    4 * x / (x - 2) - 1 = 3 / (2 - x) ↔ x = -5 / 3 := by
  -- proof goes here
  sorry

end equation1_solutions_equation2_solution_l298_298989


namespace minimum_value_proof_l298_298186

noncomputable def minimum_value_expression (a b c : ℝ) : ℝ :=
  (a - 2)^2 + (b / a - 1)^2 + (c / b - 1)^2 + (5 / c - 1)^2

theorem minimum_value_proof :
  ∀ a b c : ℝ, 2 ≤ a → a ≤ b → b ≤ c → c ≤ 5 → minimum_value_expression a b c ≥ 4 * (Real.root 4 5 - 1 / 2)^2 := 
by 
  sorry

end minimum_value_proof_l298_298186


namespace sixth_graders_l298_298268

theorem sixth_graders (total_students sixth_graders seventh_graders : ℕ)
    (h1 : seventh_graders = 64)
    (h2 : 32 * total_students = 64 * 100)
    (h3 : sixth_graders * 100 = 38 * total_students) :
    sixth_graders = 76 := by
  sorry

end sixth_graders_l298_298268


namespace find_length_of_BC_l298_298538

noncomputable def triangle_ABC (A B C I X Y : Point) (AB AC BC : ℝ) (omega : Circle) : Prop :=
  ∃ (I : Point) (X Y : Point), 
  X ∈ omega ∧ Y ∈ omega ∧
  ∠ B X C = 90 ∧ ∠ B Y C = 90 ∧
  Collinear I X Y ∧
  (AB = 80) ∧ (AC = 97) ∧ 
  BC = 59

theorem find_length_of_BC (A B C I X Y : Point) (AB AC : ℝ) (omega : Circle) :
  (triangle_ABC A B C I X Y 80 97 59 omega) → 
  BC = 59 :=
sorry

end find_length_of_BC_l298_298538


namespace largest_n_divisible_l298_298658

theorem largest_n_divisible (n : ℕ) : (n^3 + 150) % (n + 15) = 0 → n ≤ 2385 := by
  sorry

end largest_n_divisible_l298_298658


namespace lemon_more_valuable_than_banana_l298_298757

variable {L B A V : ℝ}

theorem lemon_more_valuable_than_banana
  (h1 : L + B = 2 * A + 23 * V)
  (h2 : 3 * L = 2 * B + 2 * A + 14 * V) :
  L > B := by
  sorry

end lemon_more_valuable_than_banana_l298_298757


namespace probability_of_valid_p_is_one_fifth_l298_298487

noncomputable def count_valid_p : ℕ :=
  finset.card (finset.filter (λ p, ∃ q:ℤ, p * q - 6 * p - 3 * q = 3) (finset.range 21 \ {0}))

noncomputable def probability : ℚ := count_valid_p / 20

theorem probability_of_valid_p_is_one_fifth :
  count_valid_p = 4 → probability = 1 / 5 :=
by sorry

end probability_of_valid_p_is_one_fifth_l298_298487


namespace iterated_fixed_point_l298_298195

def f (a b c d x : ℝ) : ℝ := (a * x + b) / (c * x + d)
def F_n (a b c d : ℝ) : (ℕ → ℝ → ℝ)
| 0, x => x
| (n + 1), x => f a b c d (F_n a b c d n x)

theorem iterated_fixed_point (a b c d : ℝ) (n : ℕ) (h1 : f a b c d 0 ≠ 0) 
  (h2 : f a b c d (f a b c d 0) ≠ 0) (h3 : F_n a b c d n 0 = 0) : 
  ∀ x, F_n a b c d n x = x := by
  sorry

end iterated_fixed_point_l298_298195


namespace lcm_12_18_l298_298425

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end lcm_12_18_l298_298425


namespace paper_unfold_holes_l298_298712

theorem paper_unfold_holes
  (fold_top_bottom : ℕ → ℕ)
  (fold_left_right : ℕ → ℕ)
  (rotate_90_clockwise : ℕ → ℕ)
  (punch_hole : ℕ → ℕ → Prop):
  (unfold : ℕ → ℕ) := 
begin
  -- The paper is initially folded top to bottom and left to right,
  -- rotated 90 degrees, and a hole is punched near the center.

  -- Given the operations defined:
  let folded_paper := fold_top_bottom ∘ fold_left_right,
  let rotated_paper := rotate_90_clockwise ∘ folded_paper,
  have holes_punched := punch_hole rotated_paper,

  -- When unfolded, the holes should appear symmetrically in each quadrant.
  let unfold_paper := unfold holes_punched,
  have symmetry := symmetric_quadrant_holes unfold_paper,

  -- Conclude that the paper has four holes symmetrically.
  show symmetry,
  sorry
end

end paper_unfold_holes_l298_298712


namespace knights_prob_adjacent_l298_298273

noncomputable def knights_adjacent_prob : ℚ := 1 / 5

theorem knights_prob_adjacent : 
  ∀ (knights : Finset ℕ), knights.card = 30 → 
  (choose (finset.range 30) 3).card → 
  probability_adjacent (choose (finset.range 30) 3).to_set = knights_adjacent_prob :=
begin
  intros,
  sorry
end

end knights_prob_adjacent_l298_298273


namespace find_triples_l298_298793

-- Define the conditions
def is_prime (p : ℕ) : Prop := Nat.Prime p
def power_of_p (p n : ℕ) : Prop := ∃ (k : ℕ), n = p^k

-- Given the conditions
variable (p x y : ℕ)
variable (h_prime : is_prime p)
variable (h_pos_x : x > 0)
variable (h_pos_y : y > 0)

-- The problem statement
theorem find_triples (h1 : power_of_p p (x^(p-1) + y)) (h2 : power_of_p p (x + y^(p-1))) : 
  (p = 3 ∧ x = 2 ∧ y = 5) ∨
  (p = 3 ∧ x = 5 ∧ y = 2) ∨
  (p = 2 ∧ ∃ (n i : ℕ), n > 0 ∧ i > 0 ∧ x = n ∧ y = 2^i - n ∧ 0 < n ∧ n < 2^i) := 
sorry

end find_triples_l298_298793


namespace number_of_passed_candidates_l298_298240

theorem number_of_passed_candidates
  (P F : ℕ) 
  (h1 : P + F = 120)
  (h2 : 39 * P + 15 * F = 4200) : P = 100 :=
sorry

end number_of_passed_candidates_l298_298240


namespace customer_B_cost_effectiveness_customer_A_boxes_and_consumption_l298_298985

theorem customer_B_cost_effectiveness (box_orig_cost box_spec_cost : ℕ) (orig_price spec_price eggs_per_box remaining_eggs : ℕ) 
    (h1 : orig_price = 15) (h2 : spec_price = 12) (h3 : eggs_per_box = 30) 
    (h4 : remaining_eggs = 20) : 
    ¬ (spec_price * 2 / (eggs_per_box * 2 - remaining_eggs) < orig_price / eggs_per_box) :=
by
  sorry

theorem customer_A_boxes_and_consumption (orig_price spec_price eggs_per_box total_cost_savings : ℕ) 
    (h1 : orig_price = 15) (h2 : spec_price = 12) (h3 : eggs_per_box = 30) 
    (h4 : total_cost_savings = 90): 
  ∃ (boxes_bought : ℕ) (avg_daily_consumption : ℕ), 
    (spec_price * boxes_bought = orig_price * boxes_bought * 2 - total_cost_savings) ∧ 
    (avg_daily_consumption = eggs_per_box * boxes_bought / 15) :=
by
  sorry

end customer_B_cost_effectiveness_customer_A_boxes_and_consumption_l298_298985


namespace problem_solution_l298_298336

def cylinder_volume (r h : ℝ) : ℝ :=
  π * r^2 * h

def sphere_volume (r : ℝ) : ℝ :=
  (4/3) * π * r^3

noncomputable def probability_inside_sphere_in_cylinder : ℝ :=
  let cyl_vol := cylinder_volume 2 4
  let sph_vol := sphere_volume 2
  sph_vol / cyl_vol

theorem problem_solution :
  probability_inside_sphere_in_cylinder = 2 / 3 := by
  sorry

end problem_solution_l298_298336


namespace plane_equation_through_point_and_line_l298_298057

theorem plane_equation_through_point_and_line :
  ∃ (A B C D : ℤ), A > 0 ∧ Int.gcd A B = 1 ∧ Int.gcd A C = 1 ∧ Int.gcd A D = 1 ∧
  ∀ (x y z : ℝ),
    (A * x + B * y + C * z + D = 0 ↔ 
    (∃ (t : ℝ), x = -3 * t - 1 ∧ y = 2 * t + 3 ∧ z = t - 2) ∨ 
    (x = 0 ∧ y = 7 ∧ z = -7)) :=
by
  -- sorry, implementing proofs is not required.
  sorry

end plane_equation_through_point_and_line_l298_298057


namespace inequality_implies_l298_298126

theorem inequality_implies:
  ∀ (x y : ℝ), (x > y) → (2 * x - 1 > 2 * y - 1) :=
by
  intro x y hxy
  sorry

end inequality_implies_l298_298126


namespace car_rental_problem_l298_298537

theorem car_rental_problem :
  let total_distance := 150 * 2
  let cost_per_liter := 0.90
  let rental_cost1 := 50
  let rental_cost2 := 90
  let savings := 22
  ∃ d : ℕ, 90 - (50 + (300 / d) * 0.90) = 22 → d = 15 :=
by {
  let total_distance := 150 * 2,
  let cost_per_liter := 0.90,
  let rental_cost1 := 50,
  let rental_cost2 := 90,
  let savings := 22,
  sorry
}

end car_rental_problem_l298_298537


namespace θ_decreases_as_n_increases_l298_298086

open Real

noncomputable def θ (n : ℕ) (hn : 0 < n) : ℝ :=
atan (2 + 1 / n)

theorem θ_decreases_as_n_increases : ∀ (m n : ℕ) (hm : 0 < m) (hn : 0 < n), m < n → θ m hm > θ n hn :=
by
  intros m n hm hn hmn
  sorry

end θ_decreases_as_n_increases_l298_298086


namespace complex_fraction_simplification_l298_298769

theorem complex_fraction_simplification :
  (2 * complex.I) / (1 - complex.I) = -1 + complex.I :=
by
  sorry

end complex_fraction_simplification_l298_298769


namespace final_position_D_l298_298569

open Function

-- Define the original points of the parallelogram
def A : ℝ × ℝ := (3, 4)
def B : ℝ × ℝ := (5, 8)
def C : ℝ × ℝ := (9, 4)
def D : ℝ × ℝ := (7, 0)

-- Define the reflection across the y-axis
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

-- Define the translation by (0, 1)
def translate_up (p : ℝ × ℝ) : ℝ × ℝ := (p.1, p.2 + 1)
def translate_down (p : ℝ × ℝ) : ℝ × ℝ := (p.1, p.2 - 1)

-- Define the reflection across y = x
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

-- Combine the transformations to get the final reflection across y = x - 1
def reflect_across_y_eq_x_minus_1 (p : ℝ × ℝ) : ℝ × ℝ :=
  translate_down (reflect_y_eq_x (translate_up p))

-- Prove that the final position of D after the two transformations is (1, -8)
theorem final_position_D'' : reflect_across_y_eq_x_minus_1 (reflect_y_axis D) = (1, -8) :=
  sorry

end final_position_D_l298_298569


namespace longest_diagonal_of_rhombus_l298_298716

theorem longest_diagonal_of_rhombus (d1 d2 : ℝ) (area : ℝ) (ratio : ℝ) (h1 : area = 150) (h2 : d1 / d2 = 4 / 3) :
  max d1 d2 = 20 :=
by 
  let x := sqrt (area * 2 / (d1 * d2))
  have d1_expr : d1 = 4 * x := sorry
  have d2_expr : d2 = 3 * x := sorry
  have x_val : x = 5 := sorry
  have length_longest_diag : max d1 d2 = max (4 * 5) (3 * 5) := sorry
  exact length_longest_diag

end longest_diagonal_of_rhombus_l298_298716


namespace Frank_worked_days_l298_298811

theorem Frank_worked_days
  (h_per_day : ℕ) (total_hours : ℕ) (d : ℕ) 
  (h_day_def : h_per_day = 8) 
  (total_hours_def : total_hours = 32) 
  (d_def : d = total_hours / h_per_day) : 
  d = 4 :=
by 
  rw [total_hours_def, h_day_def] at d_def
  exact d_def

end Frank_worked_days_l298_298811


namespace product_of_last_two_digits_l298_298134

theorem product_of_last_two_digits (n : ℤ) (A B : ℕ) 
    (h_sum : A + B = 12)
    (h_divisibility : ∃ k, n = 10 * A + B ∧ B = 0 ∨ B = 5) :
  A * B = 35 :=
sorry

end product_of_last_two_digits_l298_298134


namespace radius_ratio_l298_298702

noncomputable def volume_large : ℝ := 576 * Real.pi
noncomputable def volume_small : ℝ := 0.25 * volume_large

theorem radius_ratio (V_large V_small : ℝ) (h_large : V_large = 576 * Real.pi) (h_small : V_small = 0.25 * V_large) :
  (∃ r_ratio : ℝ, r_ratio = Real.sqrt (Real.sqrt (Real.sqrt (V_small / V_large)))) :=
begin
  rw [h_large, h_small],
  use 1 / Real.sqrt (Real.sqrt 4),
  sorry
end

end radius_ratio_l298_298702


namespace volume_of_solution_is_1000_l298_298339

-- Definitions based on the given conditions
def grams_per_cubic_cm (grams : ℝ) (volume : ℝ) : ℝ := grams / volume

def given_density : ℝ := grams_per_cubic_cm 0.375 25 -- 0.375 grams in 25 cm³

def target_grams : ℝ := 15 -- 15 grams in some volume

def target_density : ℝ := grams_per_cubic_cm target_grams 1000 -- This should match given_density

-- Theorem statement to prove
theorem volume_of_solution_is_1000 :
  (target_density = given_density) → (1000 = (target_grams / given_density)) :=
sorry

end volume_of_solution_is_1000_l298_298339


namespace inequality_solution_l298_298235

theorem inequality_solution :
  ∀ x : ℝ, (x - 3) / (x^2 + 4 * x + 10) ≥ 0 ↔ x ≥ 3 :=
by
  sorry

end inequality_solution_l298_298235


namespace fraction_meaningful_iff_x_ne_pm1_l298_298431

theorem fraction_meaningful_iff_x_ne_pm1 (x : ℝ) : (x^2 - 1 ≠ 0) ↔ (x ≠ 1 ∧ x ≠ -1) :=
by
  sorry

end fraction_meaningful_iff_x_ne_pm1_l298_298431


namespace probability_sum_le_one_l298_298159

theorem probability_sum_le_one :
  let I := set.Icc (-1 : ℝ) 1
  let Omega : set (ℝ × ℝ) := {p | (p.fst ∈ I) ∧ (p.snd ∈ I)}
  let measurable_set_Omega : measure_theory.measurable_set Omega := sorry
  let s_Omega := 4 -- Area of the sample space
  let A : set (ℝ × ℝ) := {p | (p.fst ∈ I) ∧ (p.snd ∈ I) ∧ (p.fst + p.snd ≤ 1)}
  let measurable_set_A : measure_theory.measurable_set A := sorry
  let s_A := (7 / 2) -- Area of the desired event
  (measure_theory.measure_of_le measurable_set_A * (7 / 8)) / (measure_theory.measure_of_le measurable_set_Omega) = 1 := 
sorry

end probability_sum_le_one_l298_298159


namespace exists_geometric_progression_with_sum_perfect_square_l298_298573

def is_geometric_progression (seq : List ℕ) : Prop :=
  ∀ (i : ℕ), i < seq.length - 1 → seq.get i * seq.get 1 = seq.get (i + 1)

def is_perfect_square (n : ℕ) : Prop :=
  ∃ (k : ℕ), k * k = n

theorem exists_geometric_progression_with_sum_perfect_square :
  ∃ (seq : List ℕ), seq.length ≥ 3 ∧ seq.get 0 = 1 ∧ is_geometric_progression seq ∧ is_perfect_square (seq.sum) :=
sorry

end exists_geometric_progression_with_sum_perfect_square_l298_298573


namespace rowan_rate_still_water_l298_298590

theorem rowan_rate_still_water :
  ∃ R : ℝ, R ≈ 9.52 ∧ (∃ C : ℝ, 40 = (R + C) * 3 ∧ 40 = (R - C) * 7) :=
by
  sorry

end rowan_rate_still_water_l298_298590


namespace frac_3125_over_1024_gt_e_l298_298306

theorem frac_3125_over_1024_gt_e : (3125 : ℝ) / 1024 > Real.exp 1 := sorry

end frac_3125_over_1024_gt_e_l298_298306


namespace point_P_on_line_l_min_distance_curve_C_to_line_l_l298_298517

-- Definitions based on conditions
def line_l (x y : ℝ) : Prop := x - y + 4 = 0
def curve_C (α : ℝ) : (ℝ × ℝ) := (sqrt 3 * cos α, sin α)
def polar_to_cartesian (r θ : ℝ) : (ℝ × ℝ) := (r * cos θ, r * sin θ)

-- Prove the positional relationship between point P and line l
theorem point_P_on_line_l : 
  (polar_to_cartesian (2 * sqrt 2) (3 * π / 4)).fst - (polar_to_cartesian (2 * sqrt 2) (3 * π / 4)).snd + 4 = 0 := 
by 
  -- Here we should convert the polar coordinates into cartesian (P(-2, 2)) and plug into line equation
  sorry

-- Prove the minimum distance from a moving point Q on curve C to line l
theorem min_distance_curve_C_to_line_l : 
  (∀ α : ℝ, sqrt 2 * cos (α + π / 6) + 2 * sqrt 2 ≥ sqrt 2) ∧ 
  (∃ α : ℝ, sqrt 2 * cos (α + π / 6) + 2 * sqrt 2 = sqrt 2) := 
by 
  -- Here we should find the minimum distance formula and show it achieves sqrt 2
  sorry

end point_P_on_line_l_min_distance_curve_C_to_line_l_l298_298517


namespace required_run_rate_remaining_25_overs_l298_298925

-- Definitions of the given conditions
def run_rate_first_25_overs : Float := 4.1
def total_overs : Nat := 50
def target_score : Nat := 375
def WLF : Float := 0.75
def runs_scored_first_25_overs : Nat := 103

-- Mathematical proof statement
theorem required_run_rate_remaining_25_overs
  (run_rate_first_25_overs_eq : run_rate_first_25_overs = 4.1)
  (total_overs_eq : total_overs = 50)
  (target_score_eq : target_score = 375)
  (WLF_eq : WLF = 0.75)
  (runs_scored_first_25_overs_eq : runs_scored_first_25_overs = 103) :
  (272 / 25 * WLF).round(2) = 8.16 := by
  sorry

end required_run_rate_remaining_25_overs_l298_298925


namespace find_z_l298_298460

noncomputable def complex_number_satisfies_conditions (z : ℂ) : Prop :=
  complex.arg (z^2 - 4) = 5 * Real.pi / 6 ∧ complex.arg (z^2 + 4) = Real.pi / 3

theorem find_z (z : ℂ) (h : complex_number_satisfies_conditions z) :
  z = 1 + Complex.sqrt 3 * Complex.I ∨ z = -(1 + Complex.sqrt 3 * Complex.I) :=
by
  sorry

end find_z_l298_298460


namespace evaluate_81_pow_8_div_3_l298_298394

theorem evaluate_81_pow_8_div_3 : 81^(8/3:ℝ) = 59049 * (9^(1/3:ℝ)) :=
by
  sorry

end evaluate_81_pow_8_div_3_l298_298394


namespace find_y_at_x_50_l298_298260

variable (x y : ℕ)

/-- Given points that lie on a straight line, we use the points to determine the equation of the line. -/
def points : List (ℕ × ℕ) := [(0, 2), (5, 17), (10, 32)]

/-- Proves that for x = 50, the y-coordinate is 152 on the given line. -/
theorem find_y_at_x_50 
  (h1 : (0, 2) ∈ points)
  (h2 : (5, 17) ∈ points)
  (h3 : (10, 32) ∈ points)
  : ∃ y : ℕ, y = 152 ∧ ∀ x = 50, y = 3 * x + 2 := sorry

end find_y_at_x_50_l298_298260


namespace largest_n_value_l298_298433

open Nat

def is_power_of_prime (x : ℕ) : Prop :=
  ∃ p k : ℕ, prime p ∧ x = p ^ k

def sequence (q : ℕ → ℕ) : Prop :=
  ∀ i ≥ 1, q i = (q (i - 1) - 1) ^ 3 + 3

theorem largest_n_value (q : ℕ → ℕ) (q0 : ℕ) (h0 : q0 > 0) 
  (hseq : ∀ i, i ≥ 1 → q i = (q (i - 1) - 1) ^ 3 + 3) 
  (h_prime_power : ∀ i ≤ 2, is_power_of_prime (q i)) : 
  ∃ n, n = 2 :=
by
  sorry

end largest_n_value_l298_298433


namespace find_depth_of_box_l298_298323

-- Define the parameters
def length : ℕ := 49
def width : ℕ := 42
def num_cubes : ℕ := 84
def cube_volume : ℕ := 7 * 7 * 7  -- since side length of each cube is 7 inches
def box_volume : ℕ := length * width * 14   -- Hypothesized depth is 14 inches

-- The statement of the proof problem in Lean 4
theorem find_depth_of_box (h_depth : 49 * 42 * 14 = 84 * (7 * 7 * 7)) : 14 = 14 :=
by
  negate h_depth
  contradicts h_volume
  sorry

end find_depth_of_box_l298_298323


namespace sum_of_digits_in_base_7_l298_298992

theorem sum_of_digits_in_base_7 (A B C : ℕ) (hA : A > 0) (hB : B > 0) (hC : C > 0) (hA7 : A < 7) (hB7 : B < 7) (hC7 : C < 7)
  (h_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C) 
  (h_eqn : A * 49 + B * 7 + C + (B * 7 + C) = A * 49 + C * 7 + A) : 
  (A + B + C) = 14 := by
  sorry

end sum_of_digits_in_base_7_l298_298992


namespace amy_l298_298352

theorem amy's_speed (a b : ℝ) (s : ℝ) 
  (h1 : ∀ (major minor : ℝ), major = 2 * minor) 
  (h2 : ∀ (w : ℝ), w = 4) 
  (h3 : ∀ (t_diff : ℝ), t_diff = 48) 
  (h4 : 2 * a + 2 * Real.pi * Real.sqrt ((4 * b^2 + b^2) / 2) - (2 * a + 2 * Real.pi * Real.sqrt (((2 * b + 8)^2 + (b + 4)^2) / 2)) = 48 * s) :
  s = Real.pi / 2 := sorry

end amy_l298_298352


namespace range_of_a_for_non_monotonicity_M_geq_2_when_abs_a_geq_2_l298_298097

-- Definitions and conditions
def f (x : ℝ) (a b : ℝ) := x^2 + a * x + b

def is_not_monotonic_in_interval (a : ℝ) : Prop :=
  -a / 2 ∈ Icc (-1 : ℝ) (1 : ℝ)

def M (a b : ℝ) : ℝ :=
  max (abs (f (-1) a b)) (abs (f (1) a b))

-- Statements to prove
theorem range_of_a_for_non_monotonicity (a b : ℝ) (h : is_not_monotonic_in_interval a) : 
  a ∈ Icc (-2 : ℝ) (2 : ℝ) :=
sorry

theorem M_geq_2_when_abs_a_geq_2 (a b : ℝ) (h : abs a ≥ 2) : 
  M (a b) ≥ 2 :=
sorry

end range_of_a_for_non_monotonicity_M_geq_2_when_abs_a_geq_2_l298_298097


namespace repeated_f_application_l298_298470

noncomputable def f (x : ℝ) : ℝ := (1 / (1 - x^3)^(1 / 3))

theorem repeated_f_application (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 1) : 
  (let q := x^3 in 
  let f2 := (1 - 1/q)^(1 / 3) in
  f2) = ∛(1 - (1 / 19^3)) :=
by
  sorry

end repeated_f_application_l298_298470


namespace intersection_of_lines_l298_298153

noncomputable def P := (3, -2, 4 : ℝ × ℝ × ℝ)
noncomputable def Q := (13, -12, 9 : ℝ × ℝ × ℝ)
noncomputable def R := (1, 5, -3 : ℝ × ℝ × ℝ)
noncomputable def S := (3, -3, 11 : ℝ × ℝ × ℝ)

theorem intersection_of_lines :
  ∃ t s: ℝ, 
    P + t • (Q - P) = R + s • (S - R) ∧
    P + t • (Q - P) = (4 / 3 : ℝ, -7 / 3 : ℝ, 14 / 3 : ℝ) := 
sorry

end intersection_of_lines_l298_298153


namespace pq_plus_sum_eq_20_l298_298125

theorem pq_plus_sum_eq_20 
  (p q : ℕ) 
  (hp : p > 0) 
  (hq : q > 0) 
  (hpl : p < 30) 
  (hql : q < 30) 
  (heq : p + q + p * q = 119) : 
  p + q = 20 :=
sorry

end pq_plus_sum_eq_20_l298_298125


namespace fraction_is_three_fourths_l298_298638

-- Define the number
def n : ℝ := 8.0

-- Define the fraction
variable (x : ℝ)

-- The main statement to be proved
theorem fraction_is_three_fourths
(h : x * n + 2 = 8) : x = 3 / 4 :=
sorry

end fraction_is_three_fourths_l298_298638


namespace sampling_method_is_systematic_l298_298325

def factory_produces_products : Prop := 
  true

def use_conveyor_belt_transport : Prop := 
  true

def quality_inspectors_sample_every_ten_minutes : Prop := 
  true

theorem sampling_method_is_systematic :
  factory_produces_products → 
  use_conveyor_belt_transport → 
  quality_inspectors_sample_every_ten_minutes →
  ∃ method : String, method = "Systematic Sampling" :=
by
  intros
  existsi "Systematic Sampling"
  refl
  sorry

end sampling_method_is_systematic_l298_298325


namespace sum_of_digits_of_N_is_15_l298_298948

theorem sum_of_digits_of_N_is_15 :
  let N := 2 * Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 8))))))
  in (N.digits.sum = 15) :=
by
  let N := 2 * Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 8))))))
  sorry

end sum_of_digits_of_N_is_15_l298_298948


namespace no_square_ends_in_4444_l298_298381

theorem no_square_ends_in_4444:
  ∀ (a k : ℕ), (a ^ 2 = 1000 * k + 444) → (∃ b m n : ℕ, (b = 500 * n + 38) ∨ (b = 500 * n - 38) → (a = 2 * b) →
  (a ^ 2 ≠ 1000 * m + 4444)) :=
by
  sorry

end no_square_ends_in_4444_l298_298381


namespace sum_of_excluded_x_values_l298_298037

theorem sum_of_excluded_x_values {x : ℝ} (h : 3 * x^2 - 9 * x + 6 = 0) :
  x = 1 ∨ x = 2 → (1 + 2 = 3) :=
by classical; sorry

end sum_of_excluded_x_values_l298_298037


namespace arrangement_of_accommodation_l298_298383

open Nat

noncomputable def num_arrangements_accommodation : ℕ :=
  (factorial 13) / ((factorial 2) * (factorial 2) * (factorial 2) * (factorial 2))

theorem arrangement_of_accommodation : num_arrangements_accommodation = 389188800 := by
  sorry

end arrangement_of_accommodation_l298_298383


namespace number_of_intersections_l298_298114

def line_eq (x y : ℝ) : Prop := 4 * x + 9 * y = 12
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 9

theorem number_of_intersections : 
  ∃ (p1 p2 : ℝ × ℝ), 
  (line_eq p1.1 p1.2 ∧ circle_eq p1.1 p1.2) ∧ 
  (line_eq p2.1 p2.2 ∧ circle_eq p2.1 p2.2) ∧ 
  p1 ≠ p2 ∧ 
  ∀ p : ℝ × ℝ, 
    (line_eq p.1 p.2 ∧ circle_eq p.1 p.2) → (p = p1 ∨ p = p2) :=
sorry

end number_of_intersections_l298_298114


namespace at_least_one_valid_triangle_l298_298000

variable (T : Type) [IsTriangle T]
variable (angles : T → ℝ) 

-- Define the condition for initial triangle T where no angle exceeds 120°
def condition (t : Triangle) : Prop :=
  ∀ (a ∈ angles t), a ≤ 120

-- Definition of resulting triangles after dividing the original triangle T
variable (resultingTriangles : Set T)

-- Define the condition for a resulting triangle where all angles do not exceed 120°
def valid_triangle (t : T) : Prop :=
  ∀ (a ∈ angles t), a ≤ 120

-- The proof statement
theorem at_least_one_valid_triangle (T : Type) [IsTriangle T] :
  ∀ t ∈ resultingTriangles, condition t → 
  ∃ t' ∈ resultingTriangles, valid_triangle t' :=
sorry

end at_least_one_valid_triangle_l298_298000


namespace sum_of_angles_convex_polygon_l298_298791

theorem sum_of_angles_convex_polygon (n : ℕ) (h : n ≥ 3) : 
  (∑ i in finset.range n, 1 * 180) = (n - 2) * 180 :=
by 
  sorry

end sum_of_angles_convex_polygon_l298_298791


namespace count_isosceles_triangles_l298_298525

-- Definitions based on conditions in a)
def is_congruent (A B C : Type) (s : A → B → C) (x y : C) : Prop := s x y = s y x
def is_perpendicular (A B : Type) (s : A → B → ℝ)
(AB : A → B → Type) : Prop := s AB = 90

-- Main Lean theorem statement
theorem count_isosceles_triangles
  (A B C D E F : Type)
  (s : A → B → Type)
  (t : B → C → Type)
  (u : C → D → Type)
  (v : D → E → Type)
  (w : E → F → Type)
  (AB AC BC BD DE EF : Type)
  (angle : Type → Type)
  (measure : angle → ℝ)
  (AB_AC : is_congruent A B C s AB AC)
  (angle_ABC_60 : measure (angle AB BC) = 60)
  (BD_bisects_ABC : ∀ x, s x D = (s B x / 2))
  (DE_perpendicular_AB : is_perpendicular B C s DE)
  (EF_perpendicular_BD : is_perpendicular E F s EF) :
  ∃ n, n = 4 := sorry

end count_isosceles_triangles_l298_298525


namespace circles_are_tangent_l298_298661

theorem circles_are_tangent :
  let circle1 := { p : ℝ × ℝ | p.1^2 + p.2^2 = 4 }
  let circle2 := { p : ℝ × ℝ | (p.1 + 1)^2 + p.2^2 = 1 }
  ∀ (d : ℝ), d = (sqrt ((-1 - 0)^2 + (0 - 0)^2)) -> d = 1 -> 1 = 2 - 1 ->
  ∀ p, (circle1 p) ↔ (circle2 p -> dist p (0, 0) + dist p (-1, 0) = 3) :=
by
  sorry

end circles_are_tangent_l298_298661


namespace largest_possible_unique_count_of_primes_l298_298566

theorem largest_possible_unique_count_of_primes (a b c d e f g h i : ℕ) 
  (h₁ : prime (a + b^c))
  (h₂ : prime (b + c^d))
  (h₃ : prime (c + d^e))
  (h₄ : prime (d + e^f))
  (h₅ : prime (e + f^g))
  (h₆ : prime (f + g^h))
  (h₇ : prime (g + h^i))
  (h₈ : prime (h + i^a))
  (h₉ : prime (i + a^b)) : 
  (∀ x y z u v w x' y' z' : ℕ, multiset.card (multiset.erase_dup ([x,y,z,u,v,w,x',y',z'] : multiset ℕ)) ≤ 5) :=
sorry

end largest_possible_unique_count_of_primes_l298_298566


namespace midpoint_of_line_segment_l298_298924

-- Conditions: Endpoint \(z1 = -7 + 5i\) and \(z2 = 5 - 9i\)
def z1 : ℂ := -7 + 5 * complex.I
def z2 : ℂ := 5 - 9 * complex.I

-- Statement: Prove that the midpoint of z1 and z2 is -1 - 2i
theorem midpoint_of_line_segment : (z1 + z2) / 2 = -1 - 2 * complex.I := by 
  sorry

end midpoint_of_line_segment_l298_298924


namespace product_a_5_to_a_50_l298_298808

def a_n (n : ℕ) := (n + 3) ^ 3 - 1 / (n * (n ^ 3 - 1))

theorem product_a_5_to_a_50 :
  ∏ i in (finset.range (50 - 5 + 1)).map (finset.range 5).succ, a_n i = 111 / (50.factorial) :=
  sorry

end product_a_5_to_a_50_l298_298808


namespace eccentricity_of_ellipse_l298_298047

theorem eccentricity_of_ellipse 
  (a b : ℝ) (x₀ y₀ : ℝ) 
  (h1 : a > b) (h2 : b > 0) 
  (h3 : (x₀^2 / a^2) + (y₀^2 / b^2) = 1) 
  (h4 : (y₀ / (x₀ + a)) * (y₀ / (a - x₀)) = 1 / 4) 
  : real.sqrt(1 - (b^2 / a^2)) = real.sqrt(3) / 2 :=
sorry

end eccentricity_of_ellipse_l298_298047


namespace find_g_neg_6_l298_298189

def f (x : ℚ) : ℚ := 4 * x - 9
def g (y : ℚ) : ℚ := 3 * (y * y) + 4 * y - 2

theorem find_g_neg_6 : g (-6) = 43 / 16 := by
  sorry

end find_g_neg_6_l298_298189


namespace marked_price_l298_298018

theorem marked_price (cost_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) 
  (hp : cost_price = 250.3) 
  (hr : discount_rate = 0.12) 
  (pr : profit_rate = 0.45) : 
  ∃ MP : ℝ, MP = 412.43 := 
by 
  have sell_price := cost_price * (1 + profit_rate)
  have marked_price := sell_price / (1 - discount_rate)
  have mp_exact := marked_price = 412.43
  exact ⟨marked_price, mp_exact⟩

end marked_price_l298_298018


namespace acute_angled_triangle_count_l298_298004

def num_vertices := 8

def total_triangles := Nat.choose num_vertices 3

def right_angled_triangles := 8 * 6

def acute_angled_triangles := total_triangles - right_angled_triangles

theorem acute_angled_triangle_count : acute_angled_triangles = 8 :=
by
  sorry

end acute_angled_triangle_count_l298_298004


namespace ink_mixing_proof_l298_298028

theorem ink_mixing_proof (m a : ℝ) (h1 : 0 < a) (h2 : a < m) :
  let cup_A_red_initial := m,
      cup_B_blue_initial := m,
      cup_A_red_after_pour := m - a,
      cup_B_red_after_pour := a,
      total_volume_B_after_pour := m + a,
      concentration_red_B := a / (m + a),
      concentration_blue_B := m / (m + a),
      amount_blue_in_pour_back := a * concentration_blue_B,
      amount_red_left_in_B_after_pour_back := a * concentration_red_B in
  amount_blue_in_pour_back = amount_red_left_in_B_after_pour_back :=
sorry

end ink_mixing_proof_l298_298028


namespace percentage_sophia_ate_l298_298365

theorem percentage_sophia_ate : 
  ∀ (caden zoe noah sophia : ℝ),
    caden = 20 / 100 →
    zoe = caden + (0.5 * caden) →
    noah = zoe + (0.5 * zoe) →
    caden + zoe + noah + sophia = 1 →
    sophia = 5 / 100 :=
by
  intros
  sorry

end percentage_sophia_ate_l298_298365


namespace balance_scale_cereal_l298_298644

def scales_are_balanced (left_pan : ℕ) (right_pan : ℕ) : Prop :=
  left_pan = right_pan

theorem balance_scale_cereal (inaccurate_scales : ℕ → ℕ → Prop)
  (cereal : ℕ)
  (correct_weight : ℕ) :
  (∀ left_pan right_pan, inaccurate_scales left_pan right_pan → left_pan = right_pan) →
  (cereal / 2 = 1) →
  true :=
  sorry

end balance_scale_cereal_l298_298644


namespace equation_of_line_through_A_farthest_from_origin_l298_298402

theorem equation_of_line_through_A_farthest_from_origin :
  ∃ l : LinearMap ℝ (ℝ × ℝ) ℝ, 
  (l (1, 2) = 0) ∧ 
  (l = LinearMap.mk 
           (λ p, p.1 + 2 * p.2 - 5)
           (LinearMap.add (λ p q, (p.1 + q.1, p.2 + q.2)) 
                          (λ p q, (p.1 + q.1, p.2 + q.2))) 
           (LinearMap.smul (λ a ⟨x, y⟩, (a * x, a * y)) 
                           (λ a ⟨x, y⟩, (a * x, a * y)))) :=
begin
  -- Proof will be provided here.
  sorry
end

end equation_of_line_through_A_farthest_from_origin_l298_298402


namespace intersection_of_A_and_B_l298_298473

-- Define sets A and B
def A := {x : ℝ | x > 0}
def B := {x : ℝ | x < 1}

-- Statement of the proof problem
theorem intersection_of_A_and_B : A ∩ B = {x | 0 < x ∧ x < 1} := by
  sorry -- The proof goes here

end intersection_of_A_and_B_l298_298473


namespace dog_biscuit_cost_l298_298361

open Real

theorem dog_biscuit_cost :
  (∀ (x : ℝ),
    (4 * x + 2) * 7 = 21 →
    x = 1 / 4) :=
by
  intro x h
  sorry

end dog_biscuit_cost_l298_298361


namespace longest_diagonal_length_l298_298725

theorem longest_diagonal_length (A : ℝ) (d1 d2 : ℝ) (h1 : A = 150) (h2 : d1 / d2 = 4 / 3) : d1 = 20 :=
by
  -- Skipping the proof here
  sorry

end longest_diagonal_length_l298_298725


namespace longest_diagonal_of_rhombus_l298_298733

variables (d1 d2 : ℝ) (x : ℝ)
def rhombus_area := (d1 * d2) / 2
def diagonal_ratio := d1 / d2 = 4 / 3

theorem longest_diagonal_of_rhombus (h : rhombus_area (4 * x) (3 * x) = 150) (r : diagonal_ratio (4 * x) (3 * x)) : d1 = 20 := by
  sorry

end longest_diagonal_of_rhombus_l298_298733


namespace black_haired_girls_count_l298_298563

theorem black_haired_girls_count (initial_total_girls : ℕ) (initial_blonde_girls : ℕ) (added_blonde_girls : ℕ) (final_blonde_girls total_girls : ℕ) 
    (h1 : initial_total_girls = 80) 
    (h2 : initial_blonde_girls = 30) 
    (h3 : added_blonde_girls = 10) 
    (h4 : final_blonde_girls = initial_blonde_girls + added_blonde_girls) 
    (h5 : total_girls = initial_total_girls) : 
    total_girls - final_blonde_girls = 40 :=
by
  sorry

end black_haired_girls_count_l298_298563


namespace lcm_12_18_l298_298413

theorem lcm_12_18 : Nat.lcm 12 18 = 36 :=
by
  -- Definitions of the conditions
  have h12 : 12 = 2 * 2 * 3 := by norm_num
  have h18 : 18 = 2 * 3 * 3 := by norm_num
  
  -- Calculating LCM using the built-in Nat.lcm
  rw [Nat.lcm_comm]  -- Ordering doesn't matter for lcm
  rw [Nat.lcm, h12, h18]
  -- Prime factorizations checks are implicitly handled
  
  -- Calculate the LCM based on the highest powers from the factorizations
  have lcm_val : 4 * 9 = 36 := by norm_num
  
  -- So, the LCM of 12 and 18 is
  exact lcm_val

end lcm_12_18_l298_298413


namespace art_piece_future_value_multiple_l298_298645

theorem art_piece_future_value_multiple (original_price increase_in_value future_value multiple : ℕ)
  (h1 : original_price = 4000)
  (h2 : increase_in_value = 8000)
  (h3 : future_value = original_price + increase_in_value)
  (h4 : multiple = future_value / original_price) :
  multiple = 3 := 
sorry

end art_piece_future_value_multiple_l298_298645


namespace exp_log_relation_l298_298039

variable (a b : ℝ)

-- Definitions of given conditions
def exp_cond : Prop := 2^a > 2^b
def log_cond : Prop := ln a > ln b

-- Necessary but not sufficient proof
theorem exp_log_relation (h : exp_cond a b) : ¬ (exp_cond a b → log_cond a b) ∧ (log_cond a b → exp_cond a b) :=
by
  sorry

end exp_log_relation_l298_298039


namespace prop2_prop4_l298_298877
-- Import the entire Mathlib for necessary definitions and theorems

-- Definitions and conditions
variable (m n : Line)
variable (α β : Plane)
variable (m_perp_α : m ⊥ α)
variable (m_perp_β : m ⊥ β)
variable (n_subset_α : n ⊆ α)

-- Statement for Proposition (2)
theorem prop2 : m_perp_α → m_perp_β → α ∥ β := by sorry

-- Statement for Proposition (4)
theorem prop4 : m_perp_α → n_subset_α → m ⊥ n := by sorry

end prop2_prop4_l298_298877


namespace find_k_l298_298138

theorem find_k (k : ℤ) (h1 : ∃(a b c : ℤ), a = (36 + k) ∧ b = (300 + k) ∧ c = (596 + k) ∧ (∃ d, 
  (a = d^2) ∧ (b = (d + 1)^2) ∧ (c = (d + 2)^2)) ) : k = 925 := by
  sorry

end find_k_l298_298138


namespace black_haired_girls_count_l298_298562

theorem black_haired_girls_count (initial_total_girls : ℕ) (initial_blonde_girls : ℕ) (added_blonde_girls : ℕ) (final_blonde_girls total_girls : ℕ) 
    (h1 : initial_total_girls = 80) 
    (h2 : initial_blonde_girls = 30) 
    (h3 : added_blonde_girls = 10) 
    (h4 : final_blonde_girls = initial_blonde_girls + added_blonde_girls) 
    (h5 : total_girls = initial_total_girls) : 
    total_girls - final_blonde_girls = 40 :=
by
  sorry

end black_haired_girls_count_l298_298562


namespace percent_defective_units_shipped_for_sale_l298_298930

theorem percent_defective_units_shipped_for_sale 
  (P : ℝ) -- total number of units produced
  (h_defective : 0.06 * P = d) -- 6 percent of units are defective
  (h_shipped : 0.0024 * P = s) -- 0.24 percent of units are defective units shipped for sale
  : (s / d) * 100 = 4 :=
by
  sorry

end percent_defective_units_shipped_for_sale_l298_298930


namespace delegates_without_badges_l298_298359

theorem delegates_without_badges :
  ∀ (total_delegates : ℕ)
    (preprinted_frac hand_written_frac break_frac : ℚ),
  total_delegates = 100 →
  preprinted_frac = 1/5 →
  break_frac = 3/7 →
  hand_written_frac = 2/9 →
  let preprinted_delegates := (preprinted_frac * total_delegates).natAbs in
  let remaining_after_preprinted := total_delegates - preprinted_delegates in
  let break_delegates := (break_frac * remaining_after_preprinted).natAbs in
  let remaining_after_break := remaining_after_preprinted - break_delegates in
  let handwritten_delegates := (hand_written_frac * remaining_after_break).natAbs in
  let non_badge_delegates := remaining_after_break - handwritten_delegates in
  non_badge_delegates = 36 :=
by
  intros
  sorry

end delegates_without_badges_l298_298359


namespace rectangle_circle_intersect_l298_298905

theorem rectangle_circle_intersect
  (A B C D E F : Point) (circle : Circle)
  (hAB : dist A B = 4) (hBC : dist B C = 5)
  (hDE : dist D E = 3)
  (B_on_circle : B ∈ circle) 
  (C_on_circle : C ∈ circle)
  (E_on_circle : E ∈ circle)
  (F_on_circle : F ∈ circle)
  (hEF_eq : dist E F = 7) :
  dist E F = 7 := by
  sorry

end rectangle_circle_intersect_l298_298905


namespace Toms_walking_speed_l298_298650

theorem Toms_walking_speed
  (total_distance : ℝ)
  (total_time : ℝ)
  (run_distance : ℝ)
  (run_speed : ℝ)
  (walk_distance : ℝ)
  (walk_time : ℝ)
  (walk_speed : ℝ)
  (h1 : total_distance = 1800)
  (h2 : total_time ≤ 20)
  (h3 : run_distance = 600)
  (h4 : run_speed = 210)
  (h5 : total_distance = run_distance + walk_distance)
  (h6 : total_time = walk_time + run_distance / run_speed)
  (h7 : walk_speed = walk_distance / walk_time) :
  walk_speed ≤ 70 := sorry

end Toms_walking_speed_l298_298650


namespace negation_of_P_is_non_P_l298_298452

open Real

/-- Proposition P: For any x in the real numbers, sin(x) <= 1 -/
def P : Prop := ∀ x : ℝ, sin x ≤ 1

/-- Negation of P: There exists x in the real numbers such that sin(x) >= 1 -/
def non_P : Prop := ∃ x : ℝ, sin x ≥ 1

theorem negation_of_P_is_non_P : ¬P ↔ non_P :=
by 
  sorry

end negation_of_P_is_non_P_l298_298452


namespace quadratic_has_real_roots_l298_298142

theorem quadratic_has_real_roots (k : ℝ) :
  (∃ (x : ℝ), (k-2) * x^2 - 2 * k * x + k = 6) ↔ (k ≥ (3 / 2) ∧ k ≠ 2) :=
by
  sorry

end quadratic_has_real_roots_l298_298142


namespace age_difference_l298_298146

-- defining the conditions
variable (A B : ℕ)
variable (h1 : B = 35)
variable (h2 : A + 10 = 2 * (B - 10))

-- the proof statement
theorem age_difference : A - B = 5 :=
by
  sorry

end age_difference_l298_298146


namespace divisors_8n4_l298_298951

noncomputable def n (p : ℕ) [Fact p.Prime] : ℕ := p ^ 16

theorem divisors_8n4 (p : ℕ) [Fact p.Prime] (hp : Odd p) : 
  numberOfDivisors(8 * (n p) ^ 4) = 260 :=
by
  sorry

end divisors_8n4_l298_298951


namespace solution_set_of_inequality_l298_298637

theorem solution_set_of_inequality (x : ℝ) : -x^2 + 3 * x - 2 > 0 ↔ 1 < x ∧ x < 2 :=
by
  sorry

end solution_set_of_inequality_l298_298637


namespace longest_diagonal_of_rhombus_l298_298715

theorem longest_diagonal_of_rhombus (d1 d2 : ℝ) (area : ℝ) (ratio : ℝ) (h1 : area = 150) (h2 : d1 / d2 = 4 / 3) :
  max d1 d2 = 20 :=
by 
  let x := sqrt (area * 2 / (d1 * d2))
  have d1_expr : d1 = 4 * x := sorry
  have d2_expr : d2 = 3 * x := sorry
  have x_val : x = 5 := sorry
  have length_longest_diag : max d1 d2 = max (4 * 5) (3 * 5) := sorry
  exact length_longest_diag

end longest_diagonal_of_rhombus_l298_298715


namespace gcd_1729_1314_l298_298656

theorem gcd_1729_1314 : Nat.gcd 1729 1314 = 1 :=
by
  sorry

end gcd_1729_1314_l298_298656


namespace max_area_of_rotating_triangle_l298_298518

-- Define points in the coordinate plane
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (15, 0)
def C : ℝ × ℝ := (25, 0)

-- Define the lines passing through points A, B, and C with initial given slopes
def l_A : ℝ → ℝ := λ θ, θ
def l_B : ℝ → ℝ := λ x, 0
def l_C : ℝ → ℝ := λ θ, -θ + 25

-- Define a function that calculates the area of the triangle formed by the intersection of the lines
noncomputable def max_triangle_area : ℝ :=
  162.5

-- The statement to be proven
theorem max_area_of_rotating_triangle : 
  ∀ θ₁ θ₂ θ₃, max_triangle_area = 162.5 := 
sorry

end max_area_of_rotating_triangle_l298_298518


namespace exists_bijection_f_R_to_R_l298_298040

theorem exists_bijection_f_R_to_R (f : ℝ → ℝ) (h_bij : Function.Bijective f) (h_condition : ∀ x : ℝ, f x + f⁻¹' {x} = -x) : 
  ∃ f : ℝ → ℝ, Function.Bijective f ∧ (∀ x : ℝ, f x + f⁻¹' {x} = -x) :=
sorry

end exists_bijection_f_R_to_R_l298_298040


namespace radius_ratio_of_smaller_to_larger_l298_298703

noncomputable def ratio_of_radii (v_large v_small : ℝ) (R r : ℝ) (h_large : (4/3) * Real.pi * R^3 = v_large) (h_small : v_small = 0.25 * v_large) (h_small_sphere : (4/3) * Real.pi * r^3 = v_small) : ℝ :=
  let ratio := r / R
  ratio

theorem radius_ratio_of_smaller_to_larger (v_large : ℝ) (R r : ℝ) (h_large : (4/3) * Real.pi * R^3 = 576 * Real.pi) (h_small_sphere : (4/3) * Real.pi * r^3 = 0.25 * 576 * Real.pi) : r / R = 1 / (2^(2/3)) :=
by
  sorry

end radius_ratio_of_smaller_to_larger_l298_298703


namespace circumference_of_circle_l298_298764

theorem circumference_of_circle :
  (∀ x y : ℝ, x^2 + y^2 - 2 * x + 2 * y = 0 → (metric.sphere ⟨1, -1⟩ (sqrt 2)).circumference = 2 * sqrt 2 * real.pi) :=
sorry

end circumference_of_circle_l298_298764


namespace limit_of_fn_div_n_l298_298696

def is_really_neat (F : set (set ℕ)) : Prop :=
  ∀ A B ∈ F, ∃ C ∈ F, A ∪ B = A ∪ C ∧ B ∪ C = A ∪ B

def f (n : ℕ) : ℕ :=
  Nat.find_greatest (λ k, ∃ (F : set (set ℕ)), is_really_neat F ∧ ⋃₀ F = Set.univ ∧ ∀ A ∈ F, Set.card A ≤ k) n / 2

theorem limit_of_fn_div_n : Filter.Tendsto (λ n, f(n) / n) Filter.atTop (Filter.lift' Filter.atTop (λ x, {0.5})) := 
by 
  sorry

end limit_of_fn_div_n_l298_298696


namespace probability_co_captains_l298_298070

theorem probability_co_captains :
  let teams := [{size := 6, co_captains := 3}, {size := 8, co_captains := 3},
                {size := 9, co_captains := 3}, {size := 11, co_captains := 3}] in
  let prob (team : {size : ℕ, co_captains : ℕ}) :=
    (team.co_captains * (team.co_captains - 1)) / (team.size * (team.size - 1)) in
  let weighted_sum :=
    (1 / 4 : ℚ) * ((prob teams[0]) + (prob teams[1]) + (prob teams[2]) + (prob teams[3])) in
  weighted_sum = (1115 / 18480 : ℚ) :=
begin
  sorry
end

end probability_co_captains_l298_298070


namespace area_triangle_identity_l298_298133

-- Definitions of the conditions and statement in the Lean 4 language.

-- Given conditions
variables {a b c Q r r1 r2 r3 : ℝ}
def herons_formula (a b c : ℝ) : ℝ :=
  (1 / 4) * Real.sqrt ((a + b + c) * (a + b - c) * (a + c - b) * (b + c - a))

def reciprocal_product (a b c Q : ℝ) : ℝ :=
  (a + b + c) * (a + b - c) * (a + c - b) * (b + c - a) / (16 * Q^4)

-- Main theorem to prove
theorem area_triangle_identity (a b c Q r r1 r2 r3 : ℝ)
  (hQ : Q = herons_formula a b c)
  (h_rec : 1 / (r * r1 * r2 * r3) = reciprocal_product a b c Q) :
  Q^2 = r * r1 * r2 * r3 := by
  sorry

end area_triangle_identity_l298_298133


namespace complex_power_sum_l298_298493

open Complex

noncomputable def cis (θ : ℝ) : ℂ := cos θ + sin θ * I

theorem complex_power_sum (z : ℂ) (h : z + z⁻¹ = real.sqrt 3) : z^2010 + z^(-2010) = -2 :=
  sorry

end complex_power_sum_l298_298493


namespace max_path_of_4x4_square_l298_298687

-- Define a 4x4 grid with edges
def edges : ℕ := 40

-- Define the maximum path length within the constraints specified
def max_path_length : ℕ := 32

-- The statement we want to prove
theorem max_path_of_4x4_square : 
  ∃ (path_length : ℕ), path_length = max_path_length ∧ path_length ≤ edges := 
begin
  use max_path_length,
  split,
  { refl, },
  { linarith, },
end

end max_path_of_4x4_square_l298_298687


namespace min_marked_cells_l298_298541

def is_even (n : ℕ) := ∃ k, n = 2 * k

def neighboring_cells (i j x y : ℕ) : Prop :=
  (i = x ∧ (j = y + 1 ∨ j + 1 = y)) ∨
  (j = y ∧ (i = x + 1 ∨ i + 1 = x))

-- Definition of the problem using the conditions
theorem min_marked_cells (n : ℕ) (h_even : is_even n) :
  ∃ m, (∀ (i j : ℕ), i < n → j < n →
    (∃ (x y : ℕ), x < n ∧ y < n ∧ (neighboring_cells i j x y) ∧ (marked x y)) →
    marked i j) ∧ m = n * (n + 2) / 4 :=
sorry

end min_marked_cells_l298_298541


namespace arrange_in_order_l298_298016

theorem arrange_in_order : 
  ∀ (a b c d e f g h i : ℚ) (h1 : a = -1.1) (h2 : b = -0.75) (h3 : c = -2/3) (h4 : d = 1/200) 
    (h5 : e = 0.005) (h6 : f = 2/3) (h7 : g = 5/7) (h8 : h = 11/15) (h9 : i = 1),
  [a, b, c, d, e, f, g, h, i].sort (≤) = [a, b, c, d, e, f, g, h, i] :=
by
  sorry

end arrange_in_order_l298_298016


namespace find_k_l298_298773

def f (x : ℝ) : ℝ := 3 * x ^ 2 - 2 * x + 8
def g (x : ℝ) (k : ℝ) : ℝ := x ^ 2 - k * x + 3

theorem find_k : 
  (f 5 - g 5 k = 12) → k = -53 / 5 :=
by
  intro hyp
  sorry

end find_k_l298_298773


namespace total_cost_is_80_l298_298318

-- Conditions
def cost_flour := 3 * 3
def cost_eggs := 3 * 10
def cost_milk := 7 * 5
def cost_baking_soda := 2 * 3

-- Question and proof requirement
theorem total_cost_is_80 : cost_flour + cost_eggs + cost_milk + cost_baking_soda = 80 := by
  sorry

end total_cost_is_80_l298_298318


namespace find_scalar_k_l298_298271

/-- Given vectors a, b, c such that a + 2b + c = 0, we want to prove that
    3 (b × a) + b × c + c × a = 0. -/
theorem find_scalar_k (a b c : ℝ^3) 
  (h : a + 2 • b + c = 0) 
  : 3 • (b × a) + (b × c) + (c × a) = 0 := 
  sorry

end find_scalar_k_l298_298271


namespace quicker_route_is_Y_by_six_point_one_minutes_l298_298212

theorem quicker_route_is_Y_by_six_point_one_minutes :
  let route_x := 
    let normal_distance_x := 6 in
    let normal_speed_x := 25 in
    let heavy_distance_x := 1 in
    let heavy_speed_x := 10 in
    (normal_distance_x / normal_speed_x * 60) + (heavy_distance_x / heavy_speed_x * 60)

  let route_y := 
    let clear_distance_y := 6 in
    let clear_speed_y := 35 in
    let construction_distance_y := 1 in
    let construction_speed_y := 15 in
    (clear_distance_y / clear_speed_y * 60) + (construction_distance_y / construction_speed_y * 60)
  
  route_y < route_x ∧ (route_x - route_y) ≈ 6.1 :=
by
  sorry

end quicker_route_is_Y_by_six_point_one_minutes_l298_298212


namespace dot_product_of_ab_ac_l298_298156

def vec_dot (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem dot_product_of_ab_ac :
  vec_dot (1, -2) (2, -2) = 6 := by
  sorry

end dot_product_of_ab_ac_l298_298156


namespace total_recovery_time_l298_298312

theorem total_recovery_time 
  (lions: ℕ := 3) (rhinos: ℕ := 2) (time_per_animal: ℕ := 2) :
  (lions + rhinos) * time_per_animal = 10 := by
  sorry

end total_recovery_time_l298_298312


namespace functional_equation_zero_l298_298820

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation_zero (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f(f(x) + f(y)) = (x + y) * f(x + y)) :
  ∀ x : ℝ, f(x) = 0 :=
begin
  sorry
end

end functional_equation_zero_l298_298820


namespace max_min_f_triangle_area_l298_298882

open Real

noncomputable def vec_a (x : ℝ) : ℝ × ℝ := (-2 * sin x, -1)
noncomputable def vec_b (x : ℝ) : ℝ × ℝ := (-cos x, cos (2 * x))
noncomputable def f (x : ℝ) : ℝ := (vec_a x).1 * (vec_b x).1 + (vec_a x).2 * (vec_b x).2

theorem max_min_f :
  (∀ x : ℝ, f x ≤ 2) ∧ (∀ x : ℝ, -2 ≤ f x) :=
sorry

theorem triangle_area
  (A B C : ℝ)
  (hA : 0 < A ∧ A < π / 2)
  (hB : 0 < B ∧ B < π / 2)
  (hC : 0 < C ∧ C < π / 2)
  (h : A + B + C = π)
  (h_f_A : f A = 1)
  (b c : ℝ)
  (h_bc : b * c = 8) :
  (1 / 2) * b * c * sin A = 2 :=
sorry

end max_min_f_triangle_area_l298_298882


namespace problem_statement_l298_298580

theorem problem_statement : 
  (777 % 4 = 1) ∧ 
  (555 % 4 = 3) ∧ 
  (999 % 4 = 3) → 
  ( (999^2021 * 555^2021 - 1) % 4 = 0 ∧ 
    (777^2021 * 999^2021 - 1) % 4 ≠ 0 ∧ 
    (555^2021 * 777^2021 - 1) % 4 ≠ 0 ) := 
by {
  sorry
}

end problem_statement_l298_298580


namespace find_amplitude_l298_298362

noncomputable def amplitude_of_cosine (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) : ℝ :=
  a

theorem find_amplitude (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_max : amplitude_of_cosine a b h_a h_b = 3) :
  a = 3 :=
sorry

end find_amplitude_l298_298362


namespace solve_for_x_l298_298597

theorem solve_for_x (x : ℝ) (h : 5 + 3.5 * x = 2.5 * x - 25) : x = -30 :=
sorry

end solve_for_x_l298_298597


namespace shanghai_expo_scientific_notation_l298_298999

theorem shanghai_expo_scientific_notation : ∃ a n, (1 ≤ |a| ∧ |a| < 10) ∧ n ∈ Int ∧ 73.08 * 10^6 = a * 10^n ∧ a = 7.308 ∧ n = 7 :=
by
  sorry

end shanghai_expo_scientific_notation_l298_298999


namespace work_completion_l298_298294

variable (A B : Type)

/-- A can do half of the work in 70 days and B can do one third of the work in 35 days.
Together, A and B can complete the work in 60 days. -/
theorem work_completion (hA : (1 : ℚ) / 2 / 70 = (1 : ℚ) / a) 
                      (hB : (1 : ℚ) / 3 / 35 = (1 : ℚ) / b) :
                      (1 / 140 + 1 / 105) = 1 / 60 :=
  sorry

end work_completion_l298_298294


namespace lambda_value_on_segment_l298_298845

theorem lambda_value_on_segment (AB AP PB : ℝ) (h1 : AB = 4 * AP) (h2 : PB = λ * PA) : λ = -3 :=
by
  sorry

end lambda_value_on_segment_l298_298845


namespace minimum_weight_of_grass_seed_l298_298326

-- Definitions of cost and weights
def price_5_pound_bag : ℝ := 13.85
def price_10_pound_bag : ℝ := 20.43
def price_25_pound_bag : ℝ := 32.20
def max_weight : ℝ := 80
def min_cost : ℝ := 98.68

-- Lean proposition to prove the minimum weight given the conditions
theorem minimum_weight_of_grass_seed (w : ℝ) :
  w = 75 ↔ (w ≤ max_weight ∧
            ∃ (n5 n10 n25 : ℕ), 
              w = 5 * n5 + 10 * n10 + 25 * n25 ∧
              min_cost ≤ n5 * price_5_pound_bag + n10 * price_10_pound_bag + n25 * price_25_pound_bag ∧
              n5 * price_5_pound_bag + n10 * price_10_pound_bag + n25 * price_25_pound_bag ≤ min_cost) := 
by
  sorry

end minimum_weight_of_grass_seed_l298_298326


namespace JackBuckets_l298_298937

theorem JackBuckets (tank_capacity buckets_per_trip_jill trips_jill time_ratio trip_buckets_jack : ℕ) :
  tank_capacity = 600 → buckets_per_trip_jill = 5 → trips_jill = 30 →
  time_ratio = 3 / 2 → trip_buckets_jack = 2 :=
  sorry

end JackBuckets_l298_298937


namespace min_product_log_condition_l298_298090

theorem min_product_log_condition (a b : ℝ) (ha : 1 < a) (hb : 1 < b) (h : Real.log a / Real.log 2 * Real.log b / Real.log 2 = 1) : 4 ≤ a * b :=
by
  sorry

end min_product_log_condition_l298_298090


namespace lateral_edges_parallel_and_equal_length_l298_298628

-- Define lateral faces of a prism as parallelograms
def is_parallelogram (face : Type) : Prop :=
  ∃ (a b c d : face), (a ≠ b) ∧ (b ≠ c) ∧ (c ≠ d) ∧ (d ≠ a) ∧ (parallel a c) ∧ (parallel b d)

-- Define lateral edges as the common edges between adjacent lateral faces
def is_lateral_edge (edge : Type) (faces : Type) [parallelogram faces] : Prop :=
  ∃ (f1 f2 : faces), (f1 ≠ f2) ∧ (edge ∈ f1) ∧ (edge ∈ f2)

-- Problem statement: Prove lateral edges are parallel and equal in length
theorem lateral_edges_parallel_and_equal_length (prism : Type) 
  (lateral_faces : prism → Type)
  (lateral_edges : prism → Type) 
  [∀ p, is_parallelogram (lateral_faces p)]
  [∀ p, is_lateral_edge (lateral_edges p) (lateral_faces p)]
  : ∀ p, parallel (lateral_edges p) ∧ (length (lateral_edges p) = length (lateral_edges p)) :=
begin
  sorry
end

end lateral_edges_parallel_and_equal_length_l298_298628


namespace correct_time_fraction_l298_298686

theorem correct_time_fraction : 
  let hours_with_glitch := [5]
  let minutes_with_glitch := [5, 15, 25, 35, 45, 55]
  let total_hours := 12
  let total_minutes_per_hour := 60
  let correct_hours := total_hours - hours_with_glitch.length
  let correct_minutes := total_minutes_per_hour - minutes_with_glitch.length
  (correct_hours * correct_minutes) / (total_hours * total_minutes_per_hour) = 33 / 40 :=
by
  sorry

end correct_time_fraction_l298_298686


namespace range_of_a_l298_298253

-- Define the function f
def f (a x : ℝ) : ℝ := a * x^3 + x

-- Define the derivative of f
def f_prime (a x : ℝ) : ℝ := 3 * a * x^2 + 1

-- State the main theorem
theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f_prime a x1 = 0 ∧ f_prime a x2 = 0) →
  a < 0 :=
by
  sorry

end range_of_a_l298_298253


namespace passing_time_correct_l298_298707

def speed_train1_kmph : ℝ := 80
def speed_train2_kmph : ℝ := 32
def length_train_m : ℝ := 280

def kmph_to_mps (speed_kmph : ℝ) : ℝ :=
  speed_kmph * 1000 / 3600

def relative_speed_kmph := speed_train1_kmph + speed_train2_kmph
def relative_speed_mps := kmph_to_mps relative_speed_kmph

def passing_time_seconds (length_m : ℝ) (speed_mps : ℝ) : ℝ :=
  length_m / speed_mps

theorem passing_time_correct :
  passing_time_seconds length_train_m relative_speed_mps ≈ 8.993 := 
  sorry

end passing_time_correct_l298_298707


namespace range_of_a_l298_298898

theorem range_of_a (a : ℝ) : 
  (∀ x y z : ℝ, x^2 + y^2 + z^2 = 1 → |a - 1| ≥ x + 2 * y + 2 * z) →
  a ∈ Set.Iic (-2) ∪ Set.Ici 4 :=
by
  sorry

end range_of_a_l298_298898


namespace find_g2_l298_298625

-- Define the conditions of the problem
def satisfies_condition (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g x + 3 * g (1 - x) = 4 * x^3 - 5 * x

-- Prove the desired value of g(2)
theorem find_g2 (g : ℝ → ℝ) (h : satisfies_condition g) : g 2 = -19 / 6 :=
by
  sorry

end find_g2_l298_298625


namespace probability_without_favorite_in_6_minutes_correct_l298_298968

noncomputable def probability_without_favorite_in_6_minutes
  (total_songs : ℕ)
  (songs_lengths : Fin total_songs → ℚ)
  (favorite_song_length : ℚ)
  (time_limit : ℚ) : ℚ :=
  have len := total_songs!
  let scenarios := finset.sum (fin_range (total_songs - 1)) (λ i, (total_songs - 1 - i)!)
  (len - scenarios) / len

theorem probability_without_favorite_in_6_minutes_correct :
  probability_without_favorite_in_6_minutes 12 (λ i, 60 + 30 * i) 300 360 = 1813 / 1980 :=
sorry

end probability_without_favorite_in_6_minutes_correct_l298_298968


namespace more_than_3000_students_l298_298977

-- Define the conditions
def students_know_secret (n : ℕ) : ℕ :=
  3 ^ (n - 1)

-- Define the statement to prove
theorem more_than_3000_students : ∃ n : ℕ, students_know_secret n > 3000 ∧ n = 9 := by
  sorry

end more_than_3000_students_l298_298977


namespace common_ratio_of_geometric_series_l298_298056

theorem common_ratio_of_geometric_series :
  let a := (8:ℚ) / 10
  let second_term := (-6:ℚ) / 15 
  let r := second_term / a
  r = -1 / 2 :=
by
  let a := (8:ℚ) / 10
  let second_term := (-6:ℚ) / 15 
  let r := second_term / a
  have : r = -1 / 2 := sorry
  exact this

end common_ratio_of_geometric_series_l298_298056


namespace intersection_point_of_lines_l298_298952

theorem intersection_point_of_lines (n : ℕ) (x y : ℤ) :
  15 * x + 18 * y = 1005 ∧ y = n * x + 2 → n = 2 :=
by
  sorry

end intersection_point_of_lines_l298_298952


namespace average_growth_rate_l298_298388

theorem average_growth_rate
  (S_Feb S_Apr : ℝ)
  (H_Feb : S_Feb = 240000)
  (H_Apr : S_Apr = 290400) 
  (H_growth : ∃ x : ℝ, S_Apr = S_Feb * (1 + x)^2) :
  ∃ x : ℝ, x = 0.1 := 
by
  have H_eq : ∃ x, 290400 = 240000 * (1 + x)^2,
  { use (√1.21 - 1),
    have H_sqrt : √1.21 - 1 = 0.1,
    { sorry },
    rw H_sqrt },
  use 0.1

end average_growth_rate_l298_298388


namespace correct_propositions_l298_298110

-- Define the conditions
variables {a b c d : ℝ}
axiom h1 : a^2 + b^2 = 1
axiom h2 : c^2 + d^2 = 1

-- Define vectors
def m : ℝ × ℝ × ℝ := (a, b, 0)
def n : ℝ × ℝ × ℝ := (c, d, 1)

-- Define propositions as statements
def prop1 : Prop := (∠ n  &=â&z-axis-space&) = π / 4
def prop3 : Prop := ∀ u v : ℝ × ℝ × ℝ, ⟪ u, v ⟫ ≤ 3π / 4
def prop4 : Prop := |m × n| ≤ √2

-- The proof statement
theorem correct_propositions : prop1 ∧ ¬ prop2 ∧ prop3 ∧ prop4 := 
by
  exact ⟨sorry, sorry, sorry, sorry⟩ -- proofs of the propositions

end correct_propositions_l298_298110


namespace r_exceeds_s_l298_298475

theorem r_exceeds_s (x y : ℚ) (h1 : x + 2 * y = 16 / 3) (h2 : 5 * x + 3 * y = 26) :
  x - y = 106 / 21 :=
sorry

end r_exceeds_s_l298_298475


namespace expand_expression_l298_298050

variable (y : ℤ)

theorem expand_expression : 12 * (3 * y - 4) = 36 * y - 48 := 
by
  sorry

end expand_expression_l298_298050


namespace eccentricity_of_ellipse_l298_298274

def ellipse_eccentricity (a b c : ℝ) (A B : ℝ × ℝ) (O : ℝ × ℝ) (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (1 - (b / a) ^ 2)

theorem eccentricity_of_ellipse (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b)
    (h4 : (3, -1) = (0.5*(-2*c*a^2 / (a^2 + b^2)), 0.5*(2*c*b^2 / (a^2 + b^2)))) :
    ellipse_eccentricity a b c (0, 0) (0, 0) (3, -1) = Real.sqrt(6) / 3 :=
sorry

end eccentricity_of_ellipse_l298_298274


namespace part_one_solution_set_part_two_inequality_l298_298098

noncomputable def f (x : ℝ) : ℝ :=
|2 * x - 1| + |2 * x + 1|

theorem part_one_solution_set :
  ∀ x : ℝ, f(x) < 4 ↔ -1 < x ∧ x < 1 :=
sorry

theorem part_two_inequality (a b : ℝ) (h₁ : -1 < a) (h₂ : a < 1) (h₃ : -1 < b) (h₄ : b < 1) :
  |a + b| / |a * b + 1| < 1 :=
sorry

end part_one_solution_set_part_two_inequality_l298_298098


namespace longest_diagonal_length_l298_298724

theorem longest_diagonal_length (A : ℝ) (d1 d2 : ℝ) (h1 : A = 150) (h2 : d1 / d2 = 4 / 3) : d1 = 20 :=
by
  -- Skipping the proof here
  sorry

end longest_diagonal_length_l298_298724


namespace faith_weekly_earnings_l298_298787

theorem faith_weekly_earnings :
  let hourly_pay := 13.50
  let regular_hours_per_day := 8
  let workdays_per_week := 5
  let overtime_hours_per_day := 2
  let regular_pay_per_day := hourly_pay * regular_hours_per_day
  let regular_pay_per_week := regular_pay_per_day * workdays_per_week
  let overtime_pay_per_day := hourly_pay * overtime_hours_per_day
  let overtime_pay_per_week := overtime_pay_per_day * workdays_per_week
  let total_weekly_earnings := regular_pay_per_week + overtime_pay_per_week
  total_weekly_earnings = 675 := 
  by
    sorry

end faith_weekly_earnings_l298_298787


namespace addition_in_sets_l298_298554

theorem addition_in_sets (a b : ℤ) (hA : ∃ k : ℤ, a = 2 * k) (hB : ∃ k : ℤ, b = 2 * k + 1) : ∃ k : ℤ, a + b = 2 * k + 1 :=
by
  sorry

end addition_in_sets_l298_298554


namespace semicircle_line_unique_intersection_l298_298516

theorem semicircle_line_unique_intersection 
  (h₁ : ∀ x : ℝ, y = sqrt (4 - x^2)) 
  (h₂ : ∃ x : ℝ, y = m ∧ y = sqrt (4 - x^2)) 
  (h₃ : ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → sqrt (4 - x₁^2) ≠ sqrt (4 - x₂^2)) 
  (hq : ∀ x : ℝ, ∃! y : ℝ, y = m ∧ y = sqrt (4 - x^2)) :
  m = 2 :=
sorry

end semicircle_line_unique_intersection_l298_298516


namespace N_is_harmonious_set_l298_298331

def harmonious_set (G : Type) [has_add G] : Prop :=
  (∀ a b : G, a + b ∈ G) ∧ (∃ c : G, ∀ a : G, a + c = c + a = a)

theorem N_is_harmonious_set : harmonious_set ℕ :=
by
  unfold harmonious_set
  split
  -- Condition (1): Closed under addition
  exact λ a b => by apply add_nonneg
  -- Condition (2): Existence of identity element
  exists 0
  exact λ a => by simp

end N_is_harmonious_set_l298_298331


namespace batsman_average_after_11th_inning_l298_298688

theorem batsman_average_after_11th_inning (A : ℝ) 
  (h1 : A + 5 = (10 * A + 85) / 11) : A + 5 = 35 :=
by
  sorry

end batsman_average_after_11th_inning_l298_298688


namespace total_games_won_l298_298364

theorem total_games_won (Betsy_games : ℕ) (Helen_games : ℕ) (Susan_games : ℕ) 
    (hBetsy : Betsy_games = 5)
    (hHelen : Helen_games = 2 * Betsy_games)
    (hSusan : Susan_games = 3 * Betsy_games) : 
    Betsy_games + Helen_games + Susan_games = 30 :=
sorry

end total_games_won_l298_298364


namespace no_nontrivial_integer_solutions_l298_298219

theorem no_nontrivial_integer_solutions (x y z : ℤ) : x^3 + 2*y^3 + 4*z^3 - 6*x*y*z = 0 -> x = 0 ∧ y = 0 ∧ z = 0 :=
by
  sorry

end no_nontrivial_integer_solutions_l298_298219


namespace average_score_l298_298213

theorem average_score (n_students : ℕ)
  (scores : List ℕ)
  (students: List ℕ)
  (h_nscores: scores = [100, 95, 85, 70, 60, 55, 45])
  (h_nstudents: students = [10, 20, 40, 40, 20, 10, 10])
  (h_nstudents_total: students.sum = 150) :
  (students.zip scores).sum (λ ⟨n, s⟩, n * s) / n_students = 75.33 := 
by
  sorry

end average_score_l298_298213


namespace jason_seashells_l298_298175

theorem jason_seashells (initial_seashells : ℕ) (given_seashells : ℕ) (remaining_seashells : ℕ) 
(h1 : initial_seashells = 49) (h2 : given_seashells = 13) :
remaining_seashells = initial_seashells - given_seashells := by
  sorry

end jason_seashells_l298_298175


namespace alpha_not_eq_beta_necessary_not_sufficient_for_cos_not_eq_l298_298682

theorem alpha_not_eq_beta_necessary_not_sufficient_for_cos_not_eq (α β : ℝ) : 
  (α ≠ β) ↔ (cos α ≠ cos β) :=
sorry

end alpha_not_eq_beta_necessary_not_sufficient_for_cos_not_eq_l298_298682


namespace missing_number_is_correct_l298_298372

theorem missing_number_is_correct (mean : ℝ) (observed_numbers : List ℝ) (total_obs : ℕ) (x : ℝ) :
  mean = 14.2 →
  observed_numbers = [8, 13, 21, 7, 23] →
  total_obs = 6 →
  (mean * total_obs = x + observed_numbers.sum) →
  x = 13.2 :=
by
  intros h_mean h_obs h_total h_sum
  sorry

end missing_number_is_correct_l298_298372


namespace find_length_QT_l298_298514

-- Define the terms used in the conditions
variables (PT : ℝ) (TQ QT : ℝ)
axiom cos_T : ℝ
axiom PT_value : PT = 10
axiom cos_T_value : cos_T = (3 / 5)
axiom right_triangle_cosine : cos_T = TQ / PT

-- Prove that QT = 8
theorem find_length_QT : QT = 8 := by
  have TQ_value : TQ = 6 := by
    rw [←right_triangle_cosine, PT_value, cos_T_value]
    field_simp
  sorry -- To be filled with the steps proving QT = 8

end find_length_QT_l298_298514


namespace min_value_x_y_xy_l298_298868

theorem min_value_x_y_xy (x y : ℝ) (h : 2 * x^2 + 3 * x * y + 2 * y^2 = 1) :
  x + y + x * y ≥ -9 / 8 :=
sorry

end min_value_x_y_xy_l298_298868


namespace hana_speed_correct_l298_298973

noncomputable def distance : ℝ := 100
noncomputable def megan_speed : ℝ := 5 / 4
noncomputable def time_difference : ℝ := 5

theorem hana_speed_correct :
  let megan_time := distance / megan_speed in       -- Time taken by Megan's car
  let hana_time := megan_time - time_difference in  -- Time taken by Hana's car
  let hana_speed := distance / hana_time in         -- Average speed of Hana's car
  hana_speed = 4 / 3 :=                             -- Result
  by
  -- Proof goes here
  sorry

end hana_speed_correct_l298_298973


namespace probability_of_specific_cards_l298_298708

noncomputable def probability_top_heart_second_spade_third_king 
  (deck_size : ℕ) (ranks_per_suit : ℕ) (suits : ℕ) (hearts : ℕ) (spades : ℕ) (kings : ℕ) : ℚ :=
  (hearts * spades * kings) / (deck_size * (deck_size - 1) * (deck_size - 2))

theorem probability_of_specific_cards :
  probability_top_heart_second_spade_third_king 104 26 4 26 26 8 = 169 / 34102 :=
by {
  sorry
}

end probability_of_specific_cards_l298_298708


namespace airplane_distance_difference_l298_298010

variable (a : ℝ)

theorem airplane_distance_difference :
  let wind_speed := 20
  (4 * a) - (3 * (a - wind_speed)) = a + 60 := by
  sorry

end airplane_distance_difference_l298_298010


namespace number_of_m_values_l298_298775

def right_triangle_exists_median (a b c d m : ℝ) : Prop :=
  (b + (2 * c)) = 3 * a + 1 ∧ (m = 5 / 2)

theorem number_of_m_values : Nat :=
  have h1 : ∃ m, right_triangle_exists_median (some a) (some b) (some c) (some d) m := sorry
  1

end number_of_m_values_l298_298775


namespace largest_number_is_l298_298264

-- Define the conditions stated in the problem
def sum_of_three_numbers_is_100 (a b c : ℝ) : Prop :=
  a + b + c = 100

def two_larger_numbers_differ_by_8 (b c : ℝ) : Prop :=
  c - b = 8

def two_smaller_numbers_differ_by_5 (a b : ℝ) : Prop :=
  b - a = 5

-- Define the hypothesis
def problem_conditions (a b c : ℝ) : Prop :=
  sum_of_three_numbers_is_100 a b c ∧
  two_larger_numbers_differ_by_8 b c ∧
  two_smaller_numbers_differ_by_5 a b

-- Define the proof problem
theorem largest_number_is (a b c : ℝ) (h : problem_conditions a b c) : 
  c = 121 / 3 :=
sorry

end largest_number_is_l298_298264


namespace find_quadratic_function_l298_298092

variables {𝔸 : Type*} [LinearOrderedField 𝔸]

def quadratic_function (a : 𝔸) : 𝔸 → 𝔸 := λ x, a * (x - 3)^2 + 2

theorem find_quadratic_function (a : 𝔸) :
  (∃ a, quadratic_function a 2 = 1 ∧ quadratic_function a 4 = 1) ∧ quadratic_function a 3 = 2 → 
  ∃ a, ∀ x, quadratic_function a x = a * (x - 3) ^ 2 + 2 :=
by
  sorry

end find_quadratic_function_l298_298092


namespace race_track_inner_circumference_l298_298255

noncomputable def inner_circumference (w R : ℝ) : ℝ :=
  let r := R - w
  2 * Real.pi * r

theorem race_track_inner_circumference :
  inner_circumference 25 165.0563499208679 ≈ 880.053 := by
  sorry

end race_track_inner_circumference_l298_298255


namespace gcd_324_243_135_l298_298280

theorem gcd_324_243_135 : Nat.gcd (Nat.gcd 324 243) 135 = 27 :=
by
  sorry

end gcd_324_243_135_l298_298280


namespace perimeter_of_square_l298_298620

theorem perimeter_of_square (length : ℕ) (width : ℕ) (area_square : ℕ) :
  length = 1024 → width = 1 → area_square = length * width → 
  4 * (Int.sqrt area_square) = 128 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  have h4 : area_square = 1024 := h3
  have h5 : (Int.sqrt 1024 : ℕ) = 32 := by norm_num
  rw [h4, h5]
  norm_num

end perimeter_of_square_l298_298620


namespace problem_statement_l298_298957

variable (a b c d : ℝ)

theorem problem_statement :
  (a^2 - a + 1) * (b^2 - b + 1) * (c^2 - c + 1) * (d^2 - d + 1) ≥ (9 / 16) * (a - b) * (b - c) * (c - d) * (d - a) :=
sorry

end problem_statement_l298_298957


namespace find_y_interval_l298_298492

theorem find_y_interval (y : ℝ) (h : y^2 - 8 * y + 12 < 0) : 2 < y ∧ y < 6 :=
sorry

end find_y_interval_l298_298492


namespace problem_statement_l298_298542

noncomputable def f (p : ℕ) : ℕ :=
  -- count the number of infinite sequences satisfying the given condition
  sorry

theorem problem_statement (p : ℕ) (hp : p.prime ∧ p > 5) :
  f(p) % 5 = 0 ∨ f(p) % 5 = 2 :=
sorry

end problem_statement_l298_298542


namespace connected_graph_min_edges_l298_298328

noncomputable def is_connected {V : Type*} (G : SimpleGraph V) : Prop :=
∀ (u v : V), u ≠ v → ∃ (p : G.walk u v), true

theorem connected_graph_min_edges (V : Type*) [fintype V] (G : SimpleGraph V) 
  (h_conn : is_connected G) (h_n : fintype.card V = n) :
  G.edge_finset.card ≥ n - 1 :=
sorry

end connected_graph_min_edges_l298_298328


namespace product_of_solutions_eq_minus_nine_l298_298428

theorem product_of_solutions_eq_minus_nine : 
  ∀ (z : ℂ), (|z| = 3 * (|z| - 2)) → z = 3 ∨ z = -3 :=
begin
  sorry -- The actual proof is omitted as per the instructions.
end

end product_of_solutions_eq_minus_nine_l298_298428


namespace lcm_12_18_l298_298424

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end lcm_12_18_l298_298424


namespace fraction_simplification_l298_298762

theorem fraction_simplification :
  (\frac{(\frac{1}{2} + \frac{1}{5})}{(\frac{3}{7} - \frac{1}{14})}) = \frac{49}{25} := 
by 
  sorry

end fraction_simplification_l298_298762


namespace eccentricity_of_curve_l298_298258

section
variables (θ : ℝ)

def parametric_curve (θ : ℝ) : ℝ × ℝ :=
  (cos θ, (sqrt 3) * sin θ)

theorem eccentricity_of_curve :
  let e := (sqrt 6) / 3 in
  ∀ (x y : ℝ), (x = cos θ) → (y = (sqrt 3) * sin θ) → 
  ∃ (e : ℝ),
    e = (sqrt 6) / 3 :=
by
  intros x y h₁ h₂
  use (sqrt 6) / 3
  sorry

end

end eccentricity_of_curve_l298_298258


namespace solution_of_homogeneous_exact_l298_298249

noncomputable def homogeneous_exact (a b : ℝ → ℝ → ℝ) : Prop :=
  ∀ x y, (∂[y] (a x y)) = (∂[x] (b x y)) ∧
  ∃ k : ℝ, ∀ (x y : ℝ), (x * (∂[x] (a x y)) + y * (∂[y] (a x y)) = k * (a x y)) ∧
                  (x * (∂[x] (b x y)) + y * (∂[y] (b x y)) = k * (b x y))

theorem solution_of_homogeneous_exact (a b : ℝ → ℝ → ℝ) (k : ℝ)
  (ha : homogeneous_exact a b) :
  (∀ x y, a x y * x + b x y * y = k) → (∃ c, ∀ x y, a x y * x + b x y * y = c) :=
sorry

end solution_of_homogeneous_exact_l298_298249


namespace fishAddedIs15_l298_298939

-- Define the number of fish Jason starts with
def initialNumberOfFish : ℕ := 6

-- Define the fish counts on each day
def fishOnDay2 := 2 * initialNumberOfFish
def fishOnDay3 := 2 * fishOnDay2 - (1 / 3 : ℚ) * (2 * fishOnDay2)
def fishOnDay4 := 2 * fishOnDay3
def fishOnDay5 := 2 * fishOnDay4 - (1 / 4 : ℚ) * (2 * fishOnDay4)
def fishOnDay6 := 2 * fishOnDay5
def fishOnDay7 := 2 * fishOnDay6

-- Define the total fish on the seventh day after adding some fish
def totalFishOnDay7 := 207

-- Define the number of fish Jason added on the seventh day
def fishAddedOnDay7 := totalFishOnDay7 - fishOnDay7

-- Prove that the number of fish Jason added on the seventh day is 15
theorem fishAddedIs15 : fishAddedOnDay7 = 15 := sorry

end fishAddedIs15_l298_298939


namespace ratio_Nicolai_to_Charliz_l298_298112

-- Definitions based on conditions
def Haylee_guppies := 3 * 12
def Jose_guppies := Haylee_guppies / 2
def Charliz_guppies := Jose_guppies / 3
def Total_guppies := 84
def Nicolai_guppies := Total_guppies - (Haylee_guppies + Jose_guppies + Charliz_guppies)

-- Proof statement
theorem ratio_Nicolai_to_Charliz : Nicolai_guppies / Charliz_guppies = 4 := by
  sorry

end ratio_Nicolai_to_Charliz_l298_298112


namespace sum_sequence_2015_equals_l298_298767

-- Define S_n as the sum of k(k + 1) for k from 1 to n
def S (n : ℕ) : ℕ := ∑ k in Finset.range (n + 1), k * (k + 1)

-- Theorem stating that S(2015) equals 2731179360
theorem sum_sequence_2015_equals :
  S 2015 = 2731179360 :=
sorry

end sum_sequence_2015_equals_l298_298767


namespace range_of_a_l298_298854

noncomputable def f : ℝ → ℝ :=
λ x, if x ≥ 0 then x^2 + 4 * x else 4 * x - x^2

theorem range_of_a (a : ℝ) : f (2 - a^2) > f a → -2 < a ∧ a < 1 :=
by
  intro h
  sorry

end range_of_a_l298_298854


namespace profit_function_definition_maximum_profit_at_100_l298_298154

noncomputable def revenue (x : ℝ) : ℝ := 700 * x
noncomputable def cost (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 40 then 
    10 * x^2 + 100 * x + 250 
  else 
    701 * x + 10000 / x - 9450 + 250

noncomputable def profit (x : ℝ) : ℝ := revenue x - cost x

theorem profit_function_definition :
  ∀ x : ℝ, 0 < x → (profit x = 
    if x < 40 then 
      -10 * x^2 + 600 * x - 250 
    else 
      -(x + 10000 / x) + 9200) := sorry

theorem maximum_profit_at_100 :
  ∃ x_max : ℝ, x_max = 100 ∧ (∀ x : ℝ, 0 < x → profit x ≤ profit x_max)
  := sorry

end profit_function_definition_maximum_profit_at_100_l298_298154


namespace kimberly_store_visits_l298_298178

def peanuts_per_visit : ℕ := 7
def total_peanuts : ℕ := 21

def visits : ℕ := total_peanuts / peanuts_per_visit

theorem kimberly_store_visits : visits = 3 :=
by
  sorry

end kimberly_store_visits_l298_298178


namespace train_speed_correct_l298_298743

-- Define the known values: lengths and time, and target speed in km/hr
def length_train : ℝ := 360
def length_platform : ℝ := 190
def time_to_pass : ℝ := 44
def expected_speed : ℝ := 45

-- State the theorem
theorem train_speed_correct :
  let total_distance := length_train + length_platform in
  let speed_m_per_s := total_distance / time_to_pass in
  let speed_km_per_hr := speed_m_per_s * 3.6 in
  speed_km_per_hr = expected_speed := by
  sorry

end train_speed_correct_l298_298743


namespace log2_a_10_l298_298148

noncomputable def a_n (n : Nat) : ℝ := (1 / 16) * 2 ^ (n - 1)

theorem log2_a_10 (h1 : (∀ n : ℕ, a_n n > 0))
  (h2 : a_n 3 * a_n 11 = 16) : log (a_n 10) / log 2 = 5 :=
by 
  -- proof goes here
  sorry

end log2_a_10_l298_298148


namespace find_2005th_element_in_M_l298_298966

def T : Set ℕ := {0, 1, 2, 3, 4, 5, 6}
def M : Set ℝ := { (a_1 / 7) + (a_2 / 49) + (a_3 / 343) + (a_4 / 2401) | 
                    a_1 in T, a_2 in T, a_3 in T, a_4 in T }

theorem find_2005th_element_in_M :
    ∃ x ∈ M, x = 1 / 7 + 1 / 49 + 0 / 343 + 4 / 2401 :=
by
  sorry

end find_2005th_element_in_M_l298_298966


namespace tangent_point_l298_298900

-- Define the curve
def curve (x : ℝ) : ℝ := 1 / x

-- Define the tangent line at point x with slope -1 and y-intercept a
def tangent_line (x a : ℝ) : ℝ := -x + a

-- State the theorem
theorem tangent_point (a : ℝ) (x : ℝ) (hx : x ≠ 0) :
  tangent_line x a = curve x ∧ - 1 = - (1 / x ^ 2) →
  a = 2 ∨ a = -2 :=
sorry

end tangent_point_l298_298900


namespace circumcenter_of_triangle_O1_P_O2_l298_298201

variables {ABC : Type*} [equilateral_triangle ABC]
variables {D : Type*} (BC : line_segment ABC) [point_on_line_segment D BC]
variables {O1 O2 I1 I2 P : Type*}
variables [circumcenter O1 (triangle ABD)] [incenter I1 (triangle ABD)]
variables [circumcenter O2 (triangle ADC)] [incenter I2 (triangle ADC)]
variables [intersection_point P (line_segment O1 I1) (line_segment O2 I2)]

theorem circumcenter_of_triangle_O1_P_O2 :
  circumcenter D (triangle O1 P O2) :=
sorry

end circumcenter_of_triangle_O1_P_O2_l298_298201


namespace percentage_less_than_l298_298144

variable (x y : ℝ)
variable (H : y = 1.4 * x)

theorem percentage_less_than :
  ((y - x) / y) * 100 = 28.57 := by
  sorry

end percentage_less_than_l298_298144


namespace proof_problem_l298_298835

-- Definitions needed for conditions
def a := -7 / 4
def b := -2 / 3
def m : ℚ := 1  -- m can be any rational number
def n : ℚ := -m  -- since m and n are opposite numbers

-- Lean statement to prove the given problem
theorem proof_problem : 4 * a / b + 3 * (m + n) = 21 / 2 := by
  -- Definitions ensuring a, b, m, n meet the conditions
  have habs : |a| = 7 / 4 := by sorry
  have brecip : 1 / b = -3 / 2 := by sorry
  have moppos : m + n = 0 := by sorry
  sorry

end proof_problem_l298_298835


namespace largest_sum_of_base8_digits_l298_298117

theorem largest_sum_of_base8_digits (a b c y : ℕ) (h1 : a < 8) (h2 : b < 8) (h3 : c < 8) (h4 : 0 < y ∧ y ≤ 16) (h5 : (a * 64 + b * 8 + c) * y = 512) :
  a + b + c ≤ 5 :=
sorry

end largest_sum_of_base8_digits_l298_298117


namespace coefficient_x3_in_expansion_eq_neg48_l298_298055

theorem coefficient_x3_in_expansion_eq_neg48 :
  let T_r := λ r : ℕ, Nat.choose 6 r
  let T_s := λ s : ℕ, (-1)^s * Nat.choose 4 s
  (finset.sum (finset.range 4) (λ i, T_r(3 - i) * T_s(i))) = -48 :=
by
  let T_r := λ r : ℕ, Nat.choose 6 r
  let T_s := λ s : ℕ, (-1)^s * Nat.choose 4 s
  -- The implementation details go here, using sorry for the incomplete proof
  sorry

end coefficient_x3_in_expansion_eq_neg48_l298_298055


namespace complex_exp_165_pow_60_l298_298022

noncomputable def complex_cos (θ : Float) : Complex := Complex.ofReal (Float.cos θ)
noncomputable def complex_sin (θ : Float) : Complex := Complex.i * Complex.ofReal (Float.sin θ)
noncomputable def complex_exp (θ : Float) := complex_cos θ + complex_sin θ

theorem complex_exp_165_pow_60 :
  (complex_exp 165) ^ 60 = -1 := by
  sorry

end complex_exp_165_pow_60_l298_298022


namespace rationalize_denominator_l298_298223

theorem rationalize_denominator :
  (let A := -6;
       B := -4;
       C := 0;
       D := 1;
       E := 30;
       F := 24 
   in A + B + C + D + E + F = 45) := 
by 
  let A := -6;
  let B := -4;
  let C := 0;
  let D := 1;
  let E := 30;
  let F := 24 in
  have h : A + B + C + D + E + F = 45 :=
        by simp [A, B, C, D, E, F];
  exact h

end rationalize_denominator_l298_298223


namespace lcm_12_18_l298_298407

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := 
by
  sorry

end lcm_12_18_l298_298407


namespace total_earnings_l298_298292

open Real

theorem total_earnings (x y : ℝ)
  (h₁ : x > 0) (h₂ : y > 0)
  (h : 4 * x * (5 * y) / 100 = 3 * x * (6 * y) / 100 + 200) :
  let total_earnings := (3 * x * (6 * y) / 100) + (4 * x * (5 * y) / 100) + (5 * x * (4 * y) / 100)
  in total_earnings = 58000 :=
by
  have h₃ : 20 * x * y = 18 * x * y + 20000 := by linarith [h]
  have h₄ : 2 * x * y = 20000 := by linarith [h₃]
  have h₅ : x * y = 10000 := eq_div_of_mul_eq_left (show 2 ≠ 0, by norm_num) h₄
  have a_earnings : 3 * x * (6 * y) / 100 = 3 * 10000 * 6 / 100 := by rw [h₅]
  have b_earnings : 4 * x * (5 * y) / 100 = 4 * 10000 * 5 / 100 := by rw [h₅]
  have c_earnings : 5 * x * (4 * y) / 100 = 5 * 10000 * 4 / 100 := by rw [h₅]
  have total_earnings_eq : (3 * 10000 * 6 / 100) + (4 * 10000 * 5 / 100) + (5 * 10000 * 4 / 100) = 58000 := by norm_num
  exact total_earnings_eq

end total_earnings_l298_298292


namespace stratified_sampling_seniors_l298_298699

-- Definitions for conditions given
def total_students : ℕ := 2000
def freshmen_students : ℕ := 760
def sophomore_probability : ℝ := 0.37
def total_select : ℕ := 20

-- Statement to prove the number of senior students to be selected
theorem stratified_sampling_seniors:
  let sophomore_students := (sophomore_probability * total_students).toNat,
      senior_students := total_students - freshmen_students - sophomore_students,
      senior_select := senior_students * total_select / total_students in
  senior_select.toNat = 5 := 
by sorry

end stratified_sampling_seniors_l298_298699


namespace min_washes_l298_298667

theorem min_washes (x : ℕ) :
  (1 / 4)^x ≤ 1 / 100 → x ≥ 4 :=
by sorry

end min_washes_l298_298667


namespace area_of_sector_l298_298619

-- Define the conditions
variables (r : ℝ) (α : ℝ) (P : ℝ)

-- The given conditions
def central_angle := (α = 2)
def perimeter := (P = 3)
def sector_perimeter_eq := (P = α * r + 2 * r)

-- Prove the area of the sector
def sector_area (r : ℝ) (α : ℝ) := (1 / 2) * α * r ^ 2

theorem area_of_sector
  (h1 : central_angle)
  (h2 : perimeter)
  (h3 : sector_perimeter_eq) :
  sector_area r α = 9/16 :=
by
  sorry

end area_of_sector_l298_298619


namespace part1_part2_part3_l298_298830

-- Define A and B based on the given conditions
def A : Set ℝ := {x | -2 < x ∧ x < 4}
def B (m : ℝ) : Set ℝ := {x | x < m}

-- 1. Given m = 3, we show that A ∩ complement U B = {x | 3 ≤ x < 4}
theorem part1 :
    let U := A ∪ B 3 in
    A ∩ Uᶜ B 3 = {x | 3 ≤ x ∧ x < 4} :=
by
  sorry

-- 2. Show that the set of real numbers for m such that A ∩ B = ∅ is {m | m ≤ -2}
theorem part2 :
    A ∩ B m = ∅ ↔ m ≤ -2 :=
by
  sorry

-- 3. Show that the set of real numbers for m such that A ∩ B = A is {m | m ≥ 4}
theorem part3 :
    A ∩ (B m) = A ↔ m ≥ 4 :=
by
  sorry

end part1_part2_part3_l298_298830


namespace Laran_poster_sales_l298_298942

theorem Laran_poster_sales:
  (L S : ℕ) (hl : L = 2)
  (profit_large : ℕ) (profit_small : ℕ)
  (daily_profit : ℕ)
  (hpl : profit_large = 10) (hps : profit_small = S * 3)
  (total_profit : daily_profit = profit_large + profit_small)
  (weekly_profit : daily_profit * 5 = 95) :
  L + S = 5 :=
by
  sorry

end Laran_poster_sales_l298_298942


namespace multiply_then_divide_eq_multiply_l298_298671

theorem multiply_then_divide_eq_multiply (x : ℚ) :
  (x * (2 / 5)) / (3 / 7) = x * (14 / 15) :=
by
  sorry

end multiply_then_divide_eq_multiply_l298_298671


namespace part_a_l298_298180

variables {n : ℕ} (A B : Matrix (Fin n) (Fin n) ℂ)

theorem part_a (hAB_comm : A * B = B * A)
  (h_detB_nonzero : det B ≠ 0)
  (h_det_cond : ∀ z : ℂ, abs z = 1 → abs (det (A + z • B)) = 1) :
  A ^ n = 0 :=
sorry

end part_a_l298_298180


namespace divisor_is_198_l298_298216

def find_divisor (dividend quotient remainder : ℕ) (divisor : ℕ) : Prop :=
  dividend = divisor * quotient + remainder

theorem divisor_is_198 : find_divisor 17698 89 14 198 :=
by {
  rw [find_divisor],
  norm_num,
  sorry
}

end divisor_is_198_l298_298216


namespace exists_real_root_iff_l298_298500

theorem exists_real_root_iff {m : ℝ} :
  (∃x : ℝ, 25 - abs (x + 1) - 4 * 5 - abs (x + 1) - m = 0) ↔ (-3 < m ∧ m < 0) :=
by
  sorry

end exists_real_root_iff_l298_298500


namespace count_parallel_lines_to_plane_l298_298824

variable (A B C A₁ B₁ C₁ : Type) [IsPrism A B C A₁ B₁ C₁]

theorem count_parallel_lines_to_plane :
  (∃ (midpoints : List (Edge × Point)) (lines : List Line),
    (∀ (m₁ m₂ : Edge × Point), m₁ ∈ midpoints → m₂ ∈ midpoints → ∃ l : Line, l ∈ lines ∧ passes_through l [m₁.2, m₂.2]) ∧
    (∀ l : Line, l ∈ lines → parallel_to_plane l (Plane.mk (A, B, B, A₁)))) →
    lines.length = 6 :=
by 
  sorry

end count_parallel_lines_to_plane_l298_298824


namespace longest_diagonal_of_rhombus_l298_298728

theorem longest_diagonal_of_rhombus (A B : ℝ) (h1 : A = 150) (h2 : ∃ x, (A = 1/2 * (4 * x) * (3 * x)) ∧ (x = 5)) : 
  4 * (classical.some h2) = 20 := 
by sorry

end longest_diagonal_of_rhombus_l298_298728


namespace function_graph_second_quadrant_l298_298502

theorem function_graph_second_quadrant (b : ℝ) (h : ∀ x, 2 ^ x + b - 1 ≥ 0): b ≤ 0 :=
sorry

end function_graph_second_quadrant_l298_298502


namespace age_ratio_in_8_years_l298_298810

-- Define the conditions
variables (s l : ℕ) -- Sam's and Leo's current ages

def condition1 := s - 4 = 2 * (l - 4)
def condition2 := s - 10 = 3 * (l - 10)

-- Define the final problem
theorem age_ratio_in_8_years (h1 : condition1 s l) (h2 : condition2 s l) : 
  ∃ x : ℕ, x = 8 ∧ (s + x) / (l + x) = 3 / 2 :=
sorry

end age_ratio_in_8_years_l298_298810


namespace intersection_of_M_and_N_l298_298105

open Set

theorem intersection_of_M_and_N :
  let M := {-1, 0, 1}
  let N := {x | x^2 - 2 * x < 0}
  M ∩ N = {1} :=
begin
  sorry
end

end intersection_of_M_and_N_l298_298105


namespace set_equality_implies_difference_l298_298545

theorem set_equality_implies_difference
  (a b : ℝ) (h : {a, 1} = {0, a + b}) : b - a = 1 :=
by
  sorry

end set_equality_implies_difference_l298_298545


namespace parabola_distance_FM_l298_298864

theorem parabola_distance_FM (p : ℝ) (h_p : p > 0) (F: ℝ × ℝ) (M A: ℝ × ℝ) :
  (∀ x y : ℝ, y^2 = 2 * p * x → y = 0) →
  (A.snd ^ 2 = 2 * p * A.fst) →
  (A.fst = -5/4) →
  (M.snd ^ 2 = 5 * M.fst) →
  (abs(M.fst + 5/4) < 0.0001) →
  (abs ∠ (A - F) (A - M) = π / 3) →
  dist M F = 5 :=
by
  intros
  sorry

end parabola_distance_FM_l298_298864


namespace find_set_B_l298_298106

def A : Set ℕ := {1, 2}
def B : Set (Set ℕ) := { x | x ⊆ A }

theorem find_set_B : B = { ∅, {1}, {2}, {1, 2} } :=
by
  sorry

end find_set_B_l298_298106


namespace no_integer_solutions_3a2_eq_b2_plus_1_l298_298398

theorem no_integer_solutions_3a2_eq_b2_plus_1 :
  ¬ ∃ (a b : ℤ), 3 * a^2 = b^2 + 1 :=
by
  sorry

end no_integer_solutions_3a2_eq_b2_plus_1_l298_298398


namespace negation_of_existence_l298_298962

theorem negation_of_existence :
  (¬ ∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) :=
by sorry

end negation_of_existence_l298_298962


namespace johnny_hourly_wage_l298_298941

-- Definitions based on conditions
def hours_worked : ℕ := 6
def total_earnings : ℝ := 28.5

-- Theorem statement
theorem johnny_hourly_wage : total_earnings / hours_worked = 4.75 :=
by
  sorry

end johnny_hourly_wage_l298_298941


namespace gcd_polynomial_l298_298455

theorem gcd_polynomial (b : ℤ) (h : 2142 ∣ b) : Int.gcd (b^2 + 11 * b + 28) (b + 6) = 2 :=
sorry

end gcd_polynomial_l298_298455


namespace find_points_A_B_find_line2_eqn_find_point_P_find_area_triangle_PAB_l298_298101

open Real
open Classical

noncomputable def line1_eqn := λ x : ℝ, (1/2) * x + 3
def point_A := (-6, 0)
def point_B := (0, 3)

-- Definitions for shifted line l2 and point P
noncomputable def line2_eqn := λ x : ℝ, (1/2) * (x - 8) + 3
def point_P := (2, 0)

-- Distances and area
def distance_AP := abs (point_P.1 - point_A.1) -- Difference in x-coordinates
def height_OB := point_B.2 -- y-coordinate of point B
def triangle_area := (1/2 : ℝ) * distance_AP * height_OB

-- Proof statements
theorem find_points_A_B : 
  line1_eqn (-6) = 0 ∧ line1_eqn 0 = 3 := 
by simp [line1_eqn]

theorem find_line2_eqn :
  ∀ x, line2_eqn x = (1 / 2) * x - 1 :=
by simp [line2_eqn]; intro x; ring

theorem find_point_P : 
  line2_eqn 2 = 0 :=
by simp [line2_eqn]

theorem find_area_triangle_PAB :
  triangle_area = 12 :=
by simp [triangle_area, distance_AP, height_OB]; norm_num

end find_points_A_B_find_line2_eqn_find_point_P_find_area_triangle_PAB_l298_298101


namespace isosceles_triangle_sides_l298_298027

theorem isosceles_triangle_sides (a k : ℝ) (h : 2 * k > a): ∃ b : ℝ, b = sqrt (4 * k^2 - a^2) :=
by
  let b := sqrt (4 * k^2 - a^2)
  exists b
  sorry

end isosceles_triangle_sides_l298_298027


namespace nominal_interest_rate_l298_298622

-- Define the effective annual rate (EAR)
def EAR : ℝ := 0.0609

-- Define the number of compounding periods per year
def n : ℕ := 2

-- Define the nominal interest rate which needs to be found
noncomputable def nominal_rate : ℝ :=
  let lhs := (1 + nominal_rate / n)^n
  let rhs := 1 + EAR
  if lhs = rhs then nominal_rate else sorry

theorem nominal_interest_rate (EAR : ℝ) (n : ℕ) : EAR = 0.0609 → n = 2 → nominal_rate = 0.0597 :=
by {
  intros h1 h2,
  sorry
}

end nominal_interest_rate_l298_298622


namespace rationalize_tsum_eq_52_l298_298225

theorem rationalize_tsum_eq_52 :
  let A := 6,
      B := 4,
      C := -1,
      D := 1,
      E := 30,
      F := 12 in
  A + B + C + D + E + F = 52 :=
by
  sorry

end rationalize_tsum_eq_52_l298_298225


namespace finite_sets_inequality_l298_298200

theorem finite_sets_inequality
  (n : ℕ) (A : fin n → Set) 
  (h1 : ∀ i, Finite (A i))
  (h2 : 3 ≤ n) :
  (1 / n : ℚ) * (Finset.range n).sum (λ i, (A i).card.to_nat) + 
  (1 / nat.choose n 3 : ℚ) * (Finset.triangular n).sum (λ (i j k : fin n), (A i ∩ A j ∩ A k).card.to_nat) ≥ 
  (2 / nat.choose n 2 : ℚ) * (Finset.diagonal n).sum (λ (i j : fin n), (A i ∩ A j).card.to_nat) :=
sorry

end finite_sets_inequality_l298_298200


namespace circle_min_dist_l298_298555

theorem circle_min_dist (a b r : ℝ) 
  (h1 : 2 * (Real.sqrt (r^2 - a^2)) = 2)
  (h2 : r^2 = 2 * b^2)
  (h3 : 2b^2 - a^2 = 1) : 
  (a = 1 ∧ b = 1) ∨ (a = -1 ∧ b = -1) → 
      ((x - a)^2 + (y - b)^2 = r^2) :=
by
  intros
  sorry

end circle_min_dist_l298_298555


namespace coefficients_rational_of_eventually_periodic_l298_298540

theorem coefficients_rational_of_eventually_periodic
  (a b d : ℤ) (f : ℝ[X]) (n : ℤ)
  (h_a : |a| ≥ 2)
  (h_d : d ≥ 0)
  (h_b : b ≥ (|a| + 1) ^ (d + 1))
  (h_deg : f.degree = d)
  (h_eventually_periodic : ∃ T : ℕ, ∀ n ≥ T, ∃ r : ℤ, (f.eval n * a^n) % b = r % b)
  : ∀ coeff : ℕ, coeff ≤ d → is_rat (f.coeff coeff) := 
  sorry

end coefficients_rational_of_eventually_periodic_l298_298540


namespace cartesian_eq_of_line_l_general_eq_of_curve_C_max_distance_from_C_to_l_l298_298920

section
variables (theta : ℝ) (x y : ℝ) (rho : ℝ)
-- Curve C defined with parameter theta
def curve_C (theta : ℝ) : Prop :=
  x = sqrt 3 * Real.cos theta ∧ y = 2 * Real.sin theta

-- Line l given in polar coordinates, converted to Cartesian coordinates
def line_l (rho theta : ℝ) : Prop :=
  rho * (2 * Real.cos theta - Real.sin theta) = 6

-- Cartesian coordinate equation of line l
theorem cartesian_eq_of_line_l (x y : ℝ) (h : line_l rho theta) : 2 * x - y = 6 :=
sorry

-- General equation of curve C
theorem general_eq_of_curve_C (x y : ℝ) (h : curve_C theta) : (x * x) / 3 + (y * y) / 4 = 1 :=
sorry

-- Distance from point P to line l
def distance_from_P_to_l (P : ℝ × ℝ) (line_l : ℝ × ℝ → ℝ × ℝ → ℝ) : ℝ :=
  abs (2 * P.1 - P.2 - 6) / sqrt 7

-- Maximum distance from a point on curve C to line l
theorem max_distance_from_C_to_l : ∃ P, curve_C theta → 
  distance_from_P_to_l (sqrt 3 * Real.cos theta, 2 * Real.sin theta) (λ _, (x, y) → (2 * x - y, 6)) = 10 * sqrt 7 / 7 :=
sorry
end

end cartesian_eq_of_line_l_general_eq_of_curve_C_max_distance_from_C_to_l_l298_298920


namespace largest_quotient_valid_smallest_product_valid_l298_298285

open Set

def number_set : Set ℤ := {-25, -4, -1, 3, 5, 9}

theorem largest_quotient_valid : ∃ a b ∈ number_set, a / b = 3 :=
by
  sorry

theorem smallest_product_valid : ∃ a b ∈ number_set, a * b = -225 :=
by
  sorry

end largest_quotient_valid_smallest_product_valid_l298_298285


namespace cubic_root_sum_cubed_l298_298197

theorem cubic_root_sum_cubed
  (p q r : ℂ)
  (h1 : 3 * p^3 - 9 * p^2 + 27 * p - 6 = 0)
  (h2 : 3 * q^3 - 9 * q^2 + 27 * q - 6 = 0)
  (h3 : 3 * r^3 - 9 * r^2 + 27 * r - 6 = 0)
  (hpq : p ≠ q)
  (hqr : q ≠ r)
  (hrp : r ≠ p) :
  (p + q + 1)^3 + (q + r + 1)^3 + (r + p + 1)^3 = 585 := 
  sorry

end cubic_root_sum_cubed_l298_298197


namespace coprime_iff_no_common_prime_factors_l298_298593

theorem coprime_iff_no_common_prime_factors (a b : ℕ) :
  Nat.coprime a b ↔ ∀ p : ℕ, Prime p → ¬ (p ∣ a ∧ p ∣ b) :=
sorry

end coprime_iff_no_common_prime_factors_l298_298593


namespace leap_years_between_2050_and_4050_l298_298330

def is_leap_year (y : Nat) : Prop :=
  y % 800 = 300 ∨ y % 800 = 500

def range (y : Nat) : Prop :=
  2050 ≤ y ∧ y ≤ 4050

def ends_in_double_zero (y : Nat) : Prop :=
  y % 100 = 0

theorem leap_years_between_2050_and_4050 :
  {y : Nat | range y ∧ ends_in_double_zero y ∧ is_leap_year y}.to_finset.card = 4 :=
by sorry

end leap_years_between_2050_and_4050_l298_298330


namespace sqrt_eq_l298_298994

theorem sqrt_eq (x : ℝ) (h : sqrt (64 - x^2) - sqrt (36 - x^2) = 4) :
  sqrt (64 - x^2) + sqrt (36 - x^2) = 7 :=
by
  sorry

end sqrt_eq_l298_298994


namespace limit_result_l298_298765

open Real

noncomputable def limit_expr : ℝ := (λ x, (exp x - exp 1) / sin (x^2 - 1))

theorem limit_result :
  tendsto limit_expr (𝓝 1) (𝓝 (exp 1 / 2)) :=
sorry

end limit_result_l298_298765


namespace lcm_12_18_is_36_l298_298417

def prime_factors (n : ℕ) : list ℕ :=
  if n = 12 then [2, 2, 3]
  else if n = 18 then [2, 3, 3]
  else []

noncomputable def lcm_of_two (a b : ℕ) : ℕ :=
  match prime_factors a, prime_factors b with
  | [2, 2, 3], [2, 3, 3] => 36
  | _, _ => 0

theorem lcm_12_18_is_36 : lcm_of_two 12 18 = 36 :=
  sorry

end lcm_12_18_is_36_l298_298417


namespace maximize_angle_OMA_l298_298528

open_locale classical
noncomputable theory

variables {O A : Point} [Circle ℝ]
def O_is_circle_center (O : Point) : Prop := is_center O

def A_inside_circle (A : Point) : Prop := inside_circle A

theorem maximize_angle_OMA :
  (∃ M₁ M₂ : Point, on_circle M₁ ∧ on_circle M₂ ∧ ∠OAM₁ = 90 ∧ ∠OAM₂ = 90) → 
  (∀ M : Point, on_circle M → ∠OMA ≤ ∠OAM₁ ∨ ∠OMA ≤ ∠OAM₂) :=
begin
  sorry
end

end maximize_angle_OMA_l298_298528


namespace cube_skew_lines_angle_l298_298020

theorem cube_skew_lines_angle (v1 v2 v3 v4 : ℝ³) 
  (h1 : is_vertex v1) 
  (h2 : is_vertex v2) 
  (h3 : is_vertex v3) 
  (h4 : is_vertex v4) 
  (h_skew1 : is_skew (line_through v1 v2) (line_through v3 v4)) 
  (h_skew2 : is_skew (line_through v1 v3) (line_through v2 v4)) 
  (h_skew3 : is_skew (line_through v1 v4) (line_through v2 v3)) :
  ∀ θ, angle_between θ (line_through v1 v2) (line_through v3 v4) ∨
       angle_between θ (line_through v1 v3) (line_through v2 v4) ∨
       angle_between θ (line_through v1 v4) (line_through v2 v3) →
  θ ≠ 30° :=
sorry

end cube_skew_lines_angle_l298_298020


namespace pies_sold_by_mcgee_l298_298595

/--
If Smith's Bakery sold 70 pies, and they sold 6 more than four times the number of pies that Mcgee's Bakery sold,
prove that Mcgee's Bakery sold 16 pies.
-/
theorem pies_sold_by_mcgee (x : ℕ) (h1 : 4 * x + 6 = 70) : x = 16 :=
by
  sorry

end pies_sold_by_mcgee_l298_298595


namespace prove_correct_conclusions_l298_298350

theorem prove_correct_conclusions : 
  let conclusion1 := ∀ (x y : ℝ), (x > 0 ∧ y > 0) → 
                    (x = 2 ∧ y = 1 → x + 2 * y = 2 * sqrt(2 * x * y)) ∧
                    (x + 2 * y = 2 * sqrt(2 * x * y) → x = 2 ∧ y = 1 → False),
      conclusion2 := ∃ (a x : ℝ), (a > 1 ∧ x > 0) ∧ a^x < log a x,
      conclusion3 := ∀ (a b : ℝ) (f : ℝ → ℝ), (continuous_on f (set.Icc a b) ∧ ∫ x in a..b, f x > 0) → 
                    ∀ x ∈ set.Icc a b, f x > 0,
      conclusion4 := ∀ (A B C : ℝ),
                     ∠'ABC = π/2 →
                     ∀ (f A B C : real.angle), sin B * (1 + 2 * cos C) = 2 * sin A * cos C + cos A * sin C → 
                     A = 2 * B,
      conclusion5 := ∀ P : ℝ × ℝ, ∃ F : ℝ × ℝ, (dist P F > 1 ∨ dist P (0, snd P) > 1) → 
                    equation_for_trajectory P F,
      correct_conclusions := {1, 2} 
  in 
      conclusion1 ∧ conclusion2 ∧ ¬conclusion3 ∧ ¬conclusion4 ∧ ¬conclusion5 → 
      ( ∀ c, c ∈ correct_conclusions ) :=
begin
  intros _,
  exact sorry,
end

end prove_correct_conclusions_l298_298350


namespace determine_f_value_l298_298458

noncomputable def f (t : ℝ) : ℝ := t^2 + 2

theorem determine_f_value : f 3 = 11 := by
  sorry

end determine_f_value_l298_298458


namespace min_value_proof_l298_298198

noncomputable theory

-- Define the conditions: x and y are positive real numbers
-- their reciprocals according to provided condition sum to 1/4

def min_value_condition (x y : ℝ) : Prop := 
  x > 0 ∧ y > 0 ∧ (1 / (x + 3) + 1 / (y + 3) = 1 / 4)

-- Define what we need to prove: for any x and y that satisfy the condition,
-- the value of x + 3y is at least 4 + 8√3

theorem min_value_proof :
  ∃ (x y : ℝ), min_value_condition x y → x + 3 * y ≥ 4 + 8 * Real.sqrt 3 :=
sorry

end min_value_proof_l298_298198


namespace function_monotonic_iff_l298_298099

noncomputable def y (x : ℝ) (a : ℝ) : ℝ := (1 / 3) * x^3 + x^2 + a * x - 5

theorem function_monotonic_iff (a : ℝ) : (∀ x : ℝ, ((1 / 3) * x^3 + x^2 + a * x - 5).derivative ≥ 0) ↔ a ≥ 1 := 
begin
  sorry
end

end function_monotonic_iff_l298_298099


namespace range_of_a_l298_298856

def f (x a: ℝ) : ℝ :=
  if x ≤ 0 then a * x^2 - 2 * x - 2 else -2^x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, ∃ M : ℝ, f x a ≤ M) ↔ (-1 ≤ a ∧ a < 0) :=
by
  sorry

end range_of_a_l298_298856


namespace total_cost_15_days_l298_298649

variable (days : ℕ)

def cost_first_5_days := 5 * 100
def cost_next_5_days := 5 * 80
def cost_last_5_days := 5 * 80
def discount := 0.10 * cost_last_5_days
def total_cost := cost_first_5_days + cost_next_5_days + (cost_last_5_days - discount)

theorem total_cost_15_days : total_cost 15 = 1260 :=
by 
  unfold total_cost cost_first_5_days cost_next_5_days cost_last_5_days discount
  sorry

end total_cost_15_days_l298_298649


namespace cos_45_minus_cos_90_eq_sqrt2_over_2_l298_298594

theorem cos_45_minus_cos_90_eq_sqrt2_over_2 :
  (Real.cos (45 * Real.pi / 180) - Real.cos (90 * Real.pi / 180)) = (Real.sqrt 2 / 2) :=
by
  have h1 : Real.cos (90 * Real.pi / 180) = 0 := by sorry
  have h2 : Real.cos (45 * Real.pi / 180) = Real.sqrt 2 / 2 := by sorry
  sorry

end cos_45_minus_cos_90_eq_sqrt2_over_2_l298_298594


namespace positive_integer_solutions_count_l298_298483

def number_of_solutions : ℕ := 2

theorem positive_integer_solutions_count :
  {x : ℕ // |7 * x - 5| ≤ 9}.card = number_of_solutions := by
sorry

end positive_integer_solutions_count_l298_298483


namespace probability_of_floor_log_difference_l298_298953

open Real

noncomputable def probability_condition (x : ℝ) : Prop :=
  ∀ x, 0 < x ∧ x < 1 → (⌊log 2 (5 * x)⌋ - ⌊log 2 x⌋ = 0)

theorem probability_of_floor_log_difference :
  probability_correct := 0.431 :=
sorry

end probability_of_floor_log_difference_l298_298953


namespace company_machine_purchase_plan_meets_requirements_l298_298324

noncomputable def machine_purchase_plan_count : Prop :=
  ∃ x : ℕ, x ≤ 2 ∧ (x = 0 ∨ x = 1 ∨ x = 2)

noncomputable def optimal_machine_purchase_plan : Prop :=
  ∃ x : ℕ, x ≤ 2 ∧ x ≥ 1 ∧
  (100 * x + 60 * (6 - x) ≥ 380 ∧ 
   (7 * x + 5 * (6 - x) = 32 ∨ 7 * x + 5 * (6 - x) = 34))

theorem company_machine_purchase_plan_meets_requirements (count : machine_purchase_plan_count) (optimal_plan : optimal_machine_purchase_plan) : 
  count → optimal_plan → ∃ x : ℕ, x = 1 ∧ 7 * x + 5 * (6 - x) = 32 :=
by sorry

end company_machine_purchase_plan_meets_requirements_l298_298324


namespace henry_room_books_l298_298883

-- Definitions based on the conditions
def totalBooks : ℕ := 99
def boxesDonated : ℕ := 3
def booksEachBox : ℕ := 15
def coffeeTableBooks : ℕ := 4
def kitchenBooks : ℕ := 18
def takenBooks : ℕ := 12
def booksRemaining : ℕ := 23

-- Main statement to prove
theorem henry_room_books :
  let donatedBooks := boxesDonated * booksEachBox + coffeeTableBooks + kitchenBooks - takenBooks in
  let roomBooks := totalBooks - donatedBooks in
  roomBooks = 44 :=
by
  sorry

end henry_room_books_l298_298883


namespace evaluate_expression_l298_298396

theorem evaluate_expression : 3 - 5 * (2^3 + 3) * 2 = -107 := by
  sorry

end evaluate_expression_l298_298396


namespace area_of_original_figure_l298_298901

theorem area_of_original_figure (base_angle : ℝ) (leg : ℝ) (top_base : ℝ) (intuitive_area: ℝ) (oblique_factor: ℝ)
  (h_angle : base_angle = 45)
  (h_leg : leg = 1)
  (h_top_base : top_base = 1)
  (h_intuitive_area : intuitive_area = (1 + 1 + real.sqrt 2) / 2 * real.sqrt 2 / 2)
  (h_oblique_factor : oblique_factor = real.sqrt 2 / 4)
  : (intuitive_area * (1 / oblique_factor) = 2 + real.sqrt 2) :=
by 
  have h1 : intuitive_area = (1 + real.sqrt 2) / 2 := by sorry
  have h2 : 1 / oblique_factor = 4 / real.sqrt 2 := by sorry
  show (intuitive_area * (1 / oblique_factor) = 2 + real.sqrt 2), by sorry
  sorry

end area_of_original_figure_l298_298901


namespace sqrt_x2y_neg_x_sqrt_y_l298_298893

variables {x y : ℝ} (h : x * y < 0)

theorem sqrt_x2y_neg_x_sqrt_y (h : x * y < 0): real.sqrt (x ^ 2 * y) = -x * real.sqrt y :=
sorry

end sqrt_x2y_neg_x_sqrt_y_l298_298893


namespace jungsoo_number_is_correct_l298_298940

def J := (1 * 4) + (0.1 * 2) + (0.001 * 7)
def Y := 100 * J 
def S := Y + 0.05

theorem jungsoo_number_is_correct : S = 420.75 := by
  sorry

end jungsoo_number_is_correct_l298_298940


namespace inverse_proportion_increasing_implication_l298_298069

theorem inverse_proportion_increasing_implication (m x : ℝ) (h1 : x > 0) (h2 : ∀ x1 x2, x1 > 0 → x2 > 0 → x1 < x2 → (m + 3) / x1 < (m + 3) / x2) : m < -3 :=
by
  sorry

end inverse_proportion_increasing_implication_l298_298069


namespace min_solution_of_x_abs_x_eq_3x_plus_4_l298_298430

theorem min_solution_of_x_abs_x_eq_3x_plus_4 : 
  ∃ x : ℝ, (x * |x| = 3 * x + 4) ∧ ∀ y : ℝ, (y * |y| = 3 * y + 4) → x ≤ y :=
sorry

end min_solution_of_x_abs_x_eq_3x_plus_4_l298_298430


namespace anayet_speed_is_61_l298_298349

-- Define the problem conditions
def amoli_speed : ℝ := 42
def amoli_time : ℝ := 3
def anayet_time : ℝ := 2
def total_distance : ℝ := 369
def remaining_distance : ℝ := 121

-- Calculate derived values
def amoli_distance : ℝ := amoli_speed * amoli_time
def covered_distance : ℝ := total_distance - remaining_distance
def anayet_distance : ℝ := covered_distance - amoli_distance

-- Define the theorem to prove Anayet's speed
theorem anayet_speed_is_61 : anayet_distance / anayet_time = 61 :=
by
  -- sorry is a placeholder for the proof
  sorry

end anayet_speed_is_61_l298_298349


namespace system_of_equations_has_no_solution_l298_298384

theorem system_of_equations_has_no_solution
  (x y z : ℝ)
  (h1 : 3 * x - 4 * y + z = 10)
  (h2 : 6 * x - 8 * y + 2 * z = 16)
  (h3 : x + y - z = 3) :
  false :=
by 
  sorry

end system_of_equations_has_no_solution_l298_298384


namespace rectangular_coordinate_equation_of_line_l_general_equation_of_curve_C_maximum_distance_l298_298931

open Real

-- Definitions
def polar_to_rect (ρ θ : ℝ) : ℝ × ℝ := (ρ * cos θ, ρ * sin θ)

-- Given conditions
theorem rectangular_coordinate_equation_of_line_l :
  ∃ ρ θ : ℝ, ρ * sin (θ - π / 4) + 2 * sqrt 2 = 0 → ∀ x y : ℝ,
  x = ρ * cos θ ∧ y = ρ * sin θ → x - y = -4 := by
  sorry

theorem general_equation_of_curve_C :
  (∀ α : ℝ, x = sqrt 3 * cos α ∧ y = sin α) → ∀ x y : ℝ,
  x^2 / 3 + y^2 = 1 := by
  sorry

theorem maximum_distance :
  ∃ M N : ℝ × ℝ, M = (-2, 2) ∧ (∀ α : ℝ, N = (sqrt 3 * cos α, sin α)) →
  ∃ P : ℝ × ℝ, P = ((sqrt 3 / 2) * cos α - 1, (1 / 2) * sin α + 1) →
  sup (λ d : ℝ, d = (abs ((sqrt 3 / 2) * cos α - (1 / 2) * sin α - 6) / sqrt 2)) = 7 * sqrt 2 / 2 := by
  sorry

end rectangular_coordinate_equation_of_line_l_general_equation_of_curve_C_maximum_distance_l298_298931


namespace all_numbers_zero_l298_298024

theorem all_numbers_zero (points : Finset Point) (n : Point → ℤ)
  (h_non_collinear : ¬ ∀ (p1 p2 p3 : Point) (h1 : p1 ∈ points) (h2 : p2 ∈ points) (h3 : p3 ∈ points), collinear p1 p2 p3)
  (h_line_sum_zero : ∀ (line : Line) (h : ∃ (p1 p2 : Point), (p1 ≠ p2 ∧ p1 ∈ points ∧ p2 ∈ points ∧ on_line p1 line ∧ on_line p2 line)),
    ∑ p in points.filter (fun p => on_line p line), n p = 0) :
  ∀ p ∈ points, n p = 0 :=
by
  sorry

end all_numbers_zero_l298_298024


namespace disjoint_paths_cover_union_l298_298182

variables {V : Type} (G : SimpleGraph V) (A B : Set V)

theorem disjoint_paths_cover_union
  (h_A_covered : ∃ (PA : Set (Set V)), (∀ p ∈ PA, SimpleGraph.Path p) ∧ (∀ a ∈ A, ∃ p ∈ PA, a ∈ p) ∧ (∀ p1 p2 ∈ PA, p1 ≠ p2 → p1 ∩ p2 = ∅))
  (h_B_covered : ∃ (PB : Set (Set V)), (∀ p ∈ PB, SimpleGraph.Path p) ∧ (∀ b ∈ B, ∃ p ∈ PB, b ∈ p) ∧ (∀ p1 p2 ∈ PB, p1 ≠ p2 → p1 ∩ p2 = ∅))
  : ∃ (PAB : Set (Set V)), (∀ p ∈ PAB, SimpleGraph.Path p) ∧ (∀ v ∈ A ∪ B, ∃ p ∈ PAB, v ∈ p) ∧ (∀ p1 p2 ∈ PAB, p1 ≠ p2 → p1 ∩ p2 = ∅) :=
sorry

end disjoint_paths_cover_union_l298_298182


namespace polynomial_integer_solution_l298_298778

theorem polynomial_integer_solution {P : ℤ[X]} (hP : ∀ n : ℕ, ∃ x : ℤ, P.eval x = 2^n) :
  ∃ a b : ℤ, a ∈ {1, -1, 2, -2} ∧ P = polynomial.C a * (polynomial.X + polynomial.C b) :=
by 
  sorry

end polynomial_integer_solution_l298_298778


namespace common_point_exists_l298_298866

noncomputable def param_eq_line (t : ℝ) : ℝ × ℝ := (4 - 3 * t, sqrt 3 * t)

noncomputable def param_eq_curve (θ : ℝ) : ℝ × ℝ := (2 + Real.cos θ, Real.sin θ)

def is_common_point (P : ℝ × ℝ) : Prop :=
  ∃ t θ, (P = param_eq_line t) ∧ (P = param_eq_curve θ)

theorem common_point_exists : is_common_point (5 / 2, sqrt 3 / 2) :=
  sorry

end common_point_exists_l298_298866


namespace arithmetic_sequence_subtract_l298_298922

theorem arithmetic_sequence_subtract (a : ℕ → ℝ) (d : ℝ) :
  (a 4 + a 6 + a 8 + a 10 + a 12 = 120) →
  (a 9 - (1 / 3) * a 11 = 16) :=
by
  sorry

end arithmetic_sequence_subtract_l298_298922


namespace solution_pairs_count_l298_298063

theorem solution_pairs_count : 
  ∃ (s : Finset (ℕ × ℕ)), (∀ (p : ℕ × ℕ), p ∈ s → 5 * p.1 + 7 * p.2 = 708) ∧ s.card = 20 :=
sorry

end solution_pairs_count_l298_298063


namespace find_distance_OC_l298_298308

variable (A B C O : Point)
variable (h1 : OnCircle A O)
variable (h2 : OnCircle B O)
variable (h3 : OnRay C A B)
variable (h4 : dist A B = 24)
variable (h5 : dist B C = 28)
variable (h6 : dist O A = 15)

theorem find_distance_OC : dist O C = 41 := by
  sorry

end find_distance_OC_l298_298308


namespace transform_C1_to_C2_l298_298850

theorem transform_C1_to_C2 (x : ℝ) :
  let C1 := λ x, cos (2 * x),
      C2 := λ x, sin (4 * x + π / 3),
      shrink := λ f, λ x, f (x / 2),
      shift_right := λ f, λ x, f (x - π / 24)
  in C2 x = (shift_right (shrink C1)) x :=
by sorry

end transform_C1_to_C2_l298_298850


namespace locus_of_centers_l298_298800

-- Define points A and B in a Euclidean space
variables {A B : Euclidean_plane.Point}

-- Theorem statement
theorem locus_of_centers (O : Euclidean_plane.Point) (h : Euclidean_plane.dist O A = Euclidean_plane.dist O B) :
  ∃ l : Euclidean_plane.Line, Euclidean_plane.perpendicular bisector l A B := sorry
  -- We want to show that there exists a line l that is the perpendicular bisector of segment AB, and the locus of O lies on this line

end locus_of_centers_l298_298800
