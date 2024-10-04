import Mathlib
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.Order.Nonneg.Basic
import Mathlib.Algebra.Polynomial.Basic
import Mathlib.Algebra.QuadraticEquation
import Mathlib.Algebra.TrigonometricFunctions
import Mathlib.Analysis.Calculus
import Mathlib.Analysis.Circle
import Mathlib.Analysis.Probability.Independence
import Mathlib.Analysis.RegularFunctions
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.Trigonometry
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype
import Mathlib.Data.Int.Basic
import Mathlib.Data.Int.Mod
import Mathlib.Data.List.Range
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Primes
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Probability.ProbabilityMassFunction
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Euclidean.Circumscription
import Mathlib.Geometry.Plane.Basic
import Mathlib.NumberTheory.LinearCongruences
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.Basic

namespace sum_first_9_terms_l271_271362

noncomputable def a (n : ℕ) : ℝ := a_0 + (n - 1) * d
noncomputable def f (x : ℝ) : ℝ := (Real.sin (2 * x)) - (Real.cos x) - 1
noncomputable def c (n : ℕ) : ℝ := f (a n)

theorem sum_first_9_terms (a_0 d : ℝ) (h : a 5 = Real.pi / 2) :
  ∑ i in finset.range 9, c i = -9 :=
by
  intro h
  sorry

end sum_first_9_terms_l271_271362


namespace circumcircles_intersect_at_point_of_concentric_and_diameter_l271_271132

noncomputable theory
open_locale classical

variables {k k' : Type*} [metric_space k] [metric_space k'] 
variables (O A B E F : k') (circumcircle₁ circumcircle₂ : set k')

-- Constants and hypotheses setup
constants (circ : set k) (diam₁ diam₂ : set k')

-- Definitions based on problem conditions
def concentric_circles (O : k') (k k' : set k) : Prop :=
  ∃ (r₁ r₂ : ℝ), r₁ < r₂ ∧ ∀ (P : k), P ∈ k ↔ dist O P = r₁ ∧ P ∈ k' ↔ dist O P = r₂

def line_through_O (O A B : k') : Prop :=
  ∃ (l : set k'), (A ∈ l) ∧ (B ∈ l) ∧ (O ∈ l)

def lies_between (O A B : k') : Prop :=
  dist O A + dist O B = dist A B

-- The proof goal
theorem circumcircles_intersect_at_point_of_concentric_and_diameter
  (h_concentric : concentric_circles O k k')
  (h_line_O_AB : line_through_O O A B)
  (h_O_between_A_and_B : lies_between O A B)
  (h_line_O_EF : line_through_O O E F)
  (h_E_between_O_and_F : lies_between E O F) :
  ∃ (S : k'), S ∈ circumcircle₁ ∧ S ∈ diam₁ ∧ S ∈ diam₂ :=
sorry

end circumcircles_intersect_at_point_of_concentric_and_diameter_l271_271132


namespace part1_part2_l271_271013

def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1 (x : ℝ) : 
  ∀ x, f x 2 ≥ 4 ↔ (x ≤ 3 / 2 ∨ x ≥ 11 / 2) :=
by sorry

theorem part2 (x a : ℝ) :
  f x a ≥ 4 → (a ≤ -1 ∨ a ≥ 3) :=
by sorry

end part1_part2_l271_271013


namespace smallest_prime_b_l271_271306

def is_prime (n : ℕ) := nat.prime n

theorem smallest_prime_b (a b : ℕ) : 
  a + b = 90 ∧ is_prime a ∧ is_prime b ∧ a > b + 2 → b = 7 :=
by
  sorry

end smallest_prime_b_l271_271306


namespace max_det_value_l271_271340

theorem max_det_value : ∃ θ : ℝ, 
  (∀ θ' : ℝ, 
     (det ![
      ![1, 1, 1],
      ![1, 1 + Real.cos θ', 1],
      ![1 + Real.sin θ', 1, 1]
     ]) ≤ (det ![
      ![1, 1, 1],
      ![1, 1 + Real.cos θ, 1],
      ![1 + Real.sin θ, 1, 1]
     ])) ∧ 
  (det ![
    ![1, 1, 1],
    ![1, 1 + Real.cos θ, 1],
    ![1 + Real.sin θ, 1, 1]
  ] = 1 / 2) := 
sorry

end max_det_value_l271_271340


namespace part1_solution_set_part2_range_of_a_l271_271006

-- Define the function f for a general a
def f (x a : ℝ) : ℝ := |x - a^2| + |x - (2 * a) + 1|

-- Part 1: When a = 2
theorem part1_solution_set (x : ℝ) (h : f x 2 ≥ 4) : x ≤ 3/2 ∨ x ≥ 11/2 :=
  sorry

-- Part 2: Range of values for a
theorem part2_range_of_a (a : ℝ) (h : ∀ x, f x a ≥ 4) : a ≤ -1 ∨ a ≥ 3 :=
  sorry

end part1_solution_set_part2_range_of_a_l271_271006


namespace measure_Angle_F_l271_271093

axiom IsoscelesTriangle_DEF : 
  ∃ (D E F : ℝ), 
    (triangle D E F) ∧
    (angle D + angle E + angle F = 180) ∧
    (D = E) ∧
    (F = D + 40)

theorem measure_Angle_F :
  ∀ (D E F : ℝ), 
    (triangle D E F) ∧
    (angle D + angle E + angle F = 180) ∧
    (D = E) ∧
    (F = D + 40) → 
    F = 86.67 :=
begin
  intros D E F H,
  sorry
end

end measure_Angle_F_l271_271093


namespace widget_cost_reduction_l271_271944

theorem widget_cost_reduction:
  ∀ (C C_reduced : ℝ), 
  6 * C = 27.60 → 
  8 * C_reduced = 27.60 → 
  C - C_reduced = 1.15 := 
by
  intros C C_reduced h1 h2
  sorry

end widget_cost_reduction_l271_271944


namespace final_price_correct_l271_271308

def cost_price : ℝ := 20
def profit_percentage : ℝ := 0.30
def sale_discount_percentage : ℝ := 0.50
def local_tax_percentage : ℝ := 0.10
def packaging_fee : ℝ := 2

def selling_price_before_discount : ℝ := cost_price * (1 + profit_percentage)
def sale_discount : ℝ := sale_discount_percentage * selling_price_before_discount
def price_after_discount : ℝ := selling_price_before_discount - sale_discount
def tax : ℝ := local_tax_percentage * price_after_discount
def price_with_tax : ℝ := price_after_discount + tax
def final_price : ℝ := price_with_tax + packaging_fee

theorem final_price_correct : final_price = 16.30 :=
by
  sorry

end final_price_correct_l271_271308


namespace rudy_total_running_time_l271_271500

theorem rudy_total_running_time :
  let time_segment_1 := 5 * 10 in
  let time_segment_2 := 4 * 9.5 in
  let time_segment_3 := 3 * 8.5 in
  let time_segment_4 := 2 * 12 in
  time_segment_1 + time_segment_2 + time_segment_3 + time_segment_4 = 137.5 :=
by
  -- Proof explicitly not required as per instructions
  sorry

end rudy_total_running_time_l271_271500


namespace sum_of_solutions_correct_l271_271577

noncomputable def sum_of_solutions : ℕ :=
  { 
    val := (List.range' 1 30 / 2).sum,
    property := sorry
  }

theorem sum_of_solutions_correct :
  (∀ x : ℕ, (x > 0 ∧ x ≤ 30 ∧ (17 * (5 * x - 3)).mod 10 = 34.mod 10) →
    List.mem x ((List.range' 1 30).filter (λ n, n % 2 = 1)) →
    ((List.range' 1 30 / 2).sum = 225)) :=
by {
  intro x h1 h2,
  sorry
}

end sum_of_solutions_correct_l271_271577


namespace find_reduced_price_per_dozen_l271_271630

def reduced_price_per_dozen (P : ℝ) : ℝ := 40 / 72 * 12

theorem find_reduced_price_per_dozen
  (P : ℝ)
  (reduction : P = 0.60 * P)
  (cost_condition : 40 = 72 * 0.60 * P) :
  reduced_price_per_dozen P ≈ 6.67 :=
by
  sorry

end find_reduced_price_per_dozen_l271_271630


namespace willie_exchange_rate_l271_271587

theorem willie_exchange_rate :
  let euros := 70
  let normal_exchange_rate := 1 / 5 -- euros per dollar
  let airport_exchange_rate := 5 / 7
  let dollars := euros * normal_exchange_rate * airport_exchange_rate
  dollars = 10 := by
  sorry

end willie_exchange_rate_l271_271587


namespace percentage_less_than_l271_271278

namespace PercentProblem

noncomputable def A (C : ℝ) : ℝ := 0.65 * C
noncomputable def B (C : ℝ) : ℝ := 0.8923076923076923 * A C

theorem percentage_less_than (C : ℝ) (hC : C ≠ 0) : (C - B C) / C = 0.42 :=
by
  sorry

end PercentProblem

end percentage_less_than_l271_271278


namespace matrix_reflection_solution_l271_271973

theorem matrix_reflection_solution :
  let R : Matrix (Fin 2) (Fin 2) ℚ := ![
    [-(1/4 : ℚ), -(5/4 : ℚ)],
    [-(3 / 4), (1 / 4)]
  ]
  let I : Matrix (Fin 2) (Fin 2) ℚ := ![
    [1, 0],
    [0, 1]
  ]
  R * R = I := by
    sorry

end matrix_reflection_solution_l271_271973


namespace polynomial_zero_integer_l271_271623

theorem polynomial_zero_integer (p q r : ℤ) (c d : ℤ) 
    (hpqr : true) -- Placeholder for stating that p, q, r are real zeros
    (lead_coeff : true) -- Placeholder for the leading coefficient being 1
    (int_coeff : true) -- Placeholder for having integer coefficients
    (deg_five : true) -- Placeholder for polynomial degree being 5
    (c_eq_neg3 : c = -3)
    (d_eq_six : d = 6) :
    ( ∃ (x y : ℤ), ∀ (a b : ℂ), a = (3 + b * sqrt 15)/2 → true) := sorry

end polynomial_zero_integer_l271_271623


namespace temperature_drop_change_l271_271428

theorem temperature_drop_change (T : ℝ) (h1 : T + 2 = T + 2) :
  (T - 4) - T = -4 :=
by
  sorry

end temperature_drop_change_l271_271428


namespace _l271_271745

@[simp] theorem upper_base_length (ABCD is_trapezoid: Boolean)
  (point_M: Boolean)
  (perpendicular_DM_AB: Boolean)
  (MC_eq_CD: Boolean)
  (AD_eq_d: ℝ)
  : BC = d / 2 := sorry

end _l271_271745


namespace english_alphabet_is_set_l271_271584

-- Conditions definition: Elements of a set must have the properties of definiteness, distinctness, and unorderedness.
def is_definite (A : Type) : Prop := ∀ (a b : A), a = b ∨ a ≠ b
def is_distinct (A : Type) : Prop := ∀ (a b : A), a ≠ b → (a ≠ b)
def is_unordered (A : Type) : Prop := true  -- For simplicity, we assume unorderedness holds for any set

-- Property that verifies if the 26 letters of the English alphabet can form a set
def english_alphabet_set : Prop :=
  is_definite Char ∧ is_distinct Char ∧ is_unordered Char

theorem english_alphabet_is_set : english_alphabet_set :=
  sorry

end english_alphabet_is_set_l271_271584


namespace trapezoid_upper_base_BC_l271_271767

theorem trapezoid_upper_base_BC (A B C D M : Point) (d : ℝ)
  (h1 : Trapezoid A B C D)
  (h2 : OnLine M A B)
  (h3 : Perpendicular D M A B)
  (h4 : Distance M C = Distance C D)
  (h5 : Distance A D = d) : Distance B C = d / 2 := 
sorry

end trapezoid_upper_base_BC_l271_271767


namespace ace_then_king_same_suit_probability_l271_271557

theorem ace_then_king_same_suit_probability : 
  let deck_size := 52
  let ace_count := 4
  let king_count := 4
  let same_suit_aces := 1
  let same_suit_kings := 1
  P((draw1 = ace) ∧ (draw2 = king) | (same_suit)) = 1 / 663 :=
by
  sorry

end ace_then_king_same_suit_probability_l271_271557


namespace sticks_form_equilateral_triangle_l271_271711

theorem sticks_form_equilateral_triangle (n : ℕ) :
  (∃ (s : set ℕ), s = {1, 2, ..., n} ∧ ∃ t : ℕ → ℕ, t ∈ s ∧ t s ∩ { a, b, c} = ∅ 
   and t t  = t and a + b + {
    sum s = nat . multiplicity = S
}) <=> n

example: ℕ in {
  n ≥ 5 ∧ n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5 } equilateral triangle (sorry)
   sorry


Sorry  = fiancé in the propre case, {1,2, ..., n}

==> n.g.e. : natal sets {1, 2, ... n }  + elastical


end sticks_form_equilateral_triangle_l271_271711


namespace count_games_l271_271847

def total_teams : ℕ := 20
def games_per_pairing : ℕ := 7
def total_games := (total_teams * (total_teams - 1)) / 2 * games_per_pairing

theorem count_games : total_games = 1330 := by
  sorry

end count_games_l271_271847


namespace average_first_200_terms_l271_271307

def sequence_term (n : ℕ) : ℤ :=
  (-1 : ℤ) ^ n * n

def sequence_sum (n : ℕ) : ℤ :=
  ∑ i in finset.range n, sequence_term i

theorem average_first_200_terms : 
  (sequence_sum 200 : ℚ) / 200 = 0.5 := 
  sorry

end average_first_200_terms_l271_271307


namespace min_value_fraction_l271_271737

theorem min_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2 * y = 1) : 
  ( (x + 1) * (y + 1) / (x * y) ) >= 8 + 4 * Real.sqrt 3 :=
sorry

end min_value_fraction_l271_271737


namespace angle_BAC_eq_angle_DAE_l271_271165

-- Define types and points A, B, C, D, E
variables (A B C D E : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E]
variables (P Q R S T : Point)

-- Define angles
variable {α β γ δ θ ω : Angle}

-- Establish the conditions
axiom angle_ABC_eq_angle_ADE : α = θ
axiom angle_AEC_eq_angle_ADB : β = ω

-- State the theorem
theorem angle_BAC_eq_angle_DAE
  (h1 : α = θ) -- Given \(\angle ABC = \angle ADE\)
  (h2 : β = ω) -- Given \(\angle AEC = \angle ADB\)
  : γ = δ := sorry

end angle_BAC_eq_angle_DAE_l271_271165


namespace prime_factors_count_l271_271344

theorem prime_factors_count :
  let expr := (4 ^ 19) * (101 ^ 10) * (2 ^ 16) * (67 ^ 5) * (17 ^ 9) in
  total_prime_factors expr = 78 :=
by
  sorry

end prime_factors_count_l271_271344


namespace best_fitting_model_l271_271079

theorem best_fitting_model (R2_1 R2_2 R2_3 R2_4 : ℝ)
  (h1 : R2_1 = 0.25) 
  (h2 : R2_2 = 0.50) 
  (h3 : R2_3 = 0.80) 
  (h4 : R2_4 = 0.98) : 
  (R2_4 = max (max R2_1 (max R2_2 R2_3)) R2_4) :=
by
  sorry

end best_fitting_model_l271_271079


namespace abs_x_minus_1_lt_4_necessary_not_sufficient_l271_271825

open Real

theorem abs_x_minus_1_lt_4_necessary_not_sufficient (x : ℝ) :
  (|x - 1| < 4) → (\frac{x - 5}{2 - x} > 0) → False :=
by
  sorry

end abs_x_minus_1_lt_4_necessary_not_sufficient_l271_271825


namespace polynomial_constant_if_bounded_l271_271333

theorem polynomial_constant_if_bounded (k : ℕ) (h_pos : k ≥ 1)
  (F : ℤ[X]) (h_bound : ∀ c ∈ finset.range (k + 2), 0 ≤ F.eval c ∧ F.eval c ≤ k) :
  ∀ c₁ c₂ ∈ finset.range (k + 2), F.eval c₁ = F.eval c₂ := 
sorry

end polynomial_constant_if_bounded_l271_271333


namespace part1_part2_l271_271904

-- Define the condition for the first part
def condition (A B C : ℝ) : Prop :=
  (C = 2 * Real.pi / 3) ∧ (1 + Real.sin A ≠ 0) ∧ (2 * Real.cos B ≠ 0) ∧ 
  (Real.cos A / (1 + Real.sin A) = Real.sin (2 * B) / (1 + Real.cos (2 * B)))

-- Theorem for the first part: If \( C = \frac{2\pi}{3} \), then \( B = \frac{\pi}{6} \)
theorem part1 (A B C : ℝ) (h : condition A B C) : B = Real.pi / 6 :=
  sorry

-- Define the condition for the second part as the side ratios expression
def ratio_expression (a b c : ℝ) : ℝ :=
  (a^2 + b^2) / (c^2)

-- Theorem for the second part: Minimum value of \(\frac{a^2 + b^2}{c^2}\)
theorem part2 (a b c : ℝ) (A B C : ℝ) 
  (h : condition A B C) : ratio_expression a b c = 4 * Real.sqrt 2 - 5 :=
  sorry

end part1_part2_l271_271904


namespace Willie_dollars_exchange_l271_271590

theorem Willie_dollars_exchange:
  ∀ (euros : ℝ) (official_rate : ℝ) (airport_rate : ℝ),
  euros = 70 →
  official_rate = 5 →
  airport_rate = 5 / 7 →
  euros / official_rate * airport_rate = 10 :=
by
  intros euros official_rate airport_rate
  intros h_euros h_official_rate h_airport_rate
  rw [h_euros, h_official_rate, h_airport_rate]
  norm_num
  sorry

end Willie_dollars_exchange_l271_271590


namespace scientific_notation_of_8_36_billion_l271_271154

theorem scientific_notation_of_8_36_billion : 
  ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ 8.36 * 10^9 = a * 10^n := 
by
  use 8.36
  use 9
  simp
  sorry

end scientific_notation_of_8_36_billion_l271_271154


namespace find_C_and_MN_line_eq_l271_271728

noncomputable def find_C_coordinates (A B : (ℝ × ℝ)) (M N : (ℝ × ℝ)) : ℝ × ℝ :=
let (xa, ya) := A,
    (xb, yb) := B in
  (C.1, C.2) where
  C := (-5, -3) -- we prove that C's coordinates must be (-5, -3)

theorem find_C_and_MN_line_eq :
  ∀ (A B : (ℝ × ℝ)) (M N : (ℝ × ℝ)),
    A = (5, -2) →
    B = (7, 3) →
    (M = (0, (-2 + -3) / 2)) →
    (N = ((-5 + 7) / 2, 0)) →
    find_C_coordinates A B M N = (-5, -3) ∧
    (∀ x y : ℝ, 5 * x - 2 * y - 5 = 0) :=
by {
  sorry
}

end find_C_and_MN_line_eq_l271_271728


namespace sum_of_ages_l271_271145

theorem sum_of_ages (M C : ℝ) (h1 : M = C + 12) (h2 : M + 10 = 3 * (C - 6)) : M + C = 52 :=
by
  sorry

end sum_of_ages_l271_271145


namespace monocolor_isosceles_invariant_l271_271099

def regular_polygon (n : ℕ) (n_gt3 : n > 3) (odd_n : odd n) (n_not_divisible_by_3 : ¬ (3 ∣ n)): Type := sorry

def colored_vertices (n : ℕ) (m : ℕ) (h : m ≤ n) : Type := sorry

def monocolor_isosceles_triangles {n : ℕ} [regular_polygon n] (m : ℕ) [colored_vertices n m] : ℕ :=
  if m = 0 then 0 else
    2 * n / 3

theorem monocolor_isosceles_invariant (n : ℕ) (m : ℕ)
  (n_gt3 : n > 3) (odd_n : nat.odd n) (n_not_divisible_by_3 : ¬ (3 ∣ n))
  (m_leq_n : m ≤ n) :
  @monocolor_isosceles_triangles n (regular_polygon n n_gt3 odd_n n_not_divisible_by_3) m (colored_vertices n m m_leq_n) = (2 * n / 3) :=
by
  intros
  apply sorry

end monocolor_isosceles_invariant_l271_271099


namespace part1_solution_set_part2_values_of_a_l271_271038

-- Parameters and definitions for the function and conditions
def f (x a : ℝ) := |x - a^2| + |x - (2*a + 1)|

-- Part (1) When a = 2, the inequality's solution set
theorem part1_solution_set (x : ℝ) : f x 2 ≥ 4 ↔ (x ≤ 3/2 ∨ x ≥ 11/2) := 
sorry

-- Part (2) Range of values for a given f(x, a) ≥ 4
theorem part2_values_of_a (a : ℝ) : 
∀ x, f x a ≥ 4 → (a ≤ -1 ∨ a ≥ 3) :=
sorry

end part1_solution_set_part2_values_of_a_l271_271038


namespace firefighter_remaining_money_correct_l271_271270

noncomputable def firefighter_weekly_earnings : ℕ := 30 * 48
noncomputable def firefighter_monthly_earnings : ℕ := firefighter_weekly_earnings * 4
noncomputable def firefighter_rent_expense : ℕ := firefighter_monthly_earnings / 3
noncomputable def firefighter_food_expense : ℕ := 500
noncomputable def firefighter_tax_expense : ℕ := 1000
noncomputable def firefighter_total_expenses : ℕ := firefighter_rent_expense + firefighter_food_expense + firefighter_tax_expense
noncomputable def firefighter_remaining_money : ℕ := firefighter_monthly_earnings - firefighter_total_expenses

theorem firefighter_remaining_money_correct :
  firefighter_remaining_money = 2340 :=
by 
  rfl

end firefighter_remaining_money_correct_l271_271270


namespace max_valid_triples_l271_271887

def A : Set ℤ := {1, 2, 3, 4, 5, 6, 7}

def count_valid_triples (A : Set ℤ) : ℕ :=
  Set.count_triples (λ (x y z : ℤ), x < y ∧ x + y = z ∧ x ∈ A ∧ y ∈ A ∧ z ∈ A)

theorem max_valid_triples :
  ∃ (n : ℕ), (∀ (A : Set ℤ), count_valid_triples A ≤ n) ∧ n = 9 :=
  sorry

end max_valid_triples_l271_271887


namespace sin_cos_sum_eq_l271_271782

theorem sin_cos_sum_eq (θ : ℝ) 
  (h1 : θ ∈ Set.Ioo (π / 2) π) 
  (h2 : Real.tan (θ + π / 4) = 1 / 2): 
  Real.sin θ + Real.cos θ = -Real.sqrt 10 / 5 := 
  sorry

end sin_cos_sum_eq_l271_271782


namespace part1_part2_l271_271932

-- Define the problem conditions
def vector_a : ℝ × ℝ := (2, 4)
def vector_b (m : ℝ) : ℝ × ℝ := (m, -1)
def dot_product (u v : ℝ × ℝ) := u.1 * v.1 + u.2 * v.2
def magnitude (u : ℝ × ℝ) := real.sqrt (u.1^2 + u.2^2)
def add_vectors (u v : ℝ × ℝ) := (u.1 + v.1, u.2 + v.2)

-- Part 1: Given that vector_a is perpendicular to vector_b, prove that m = 2.
theorem part1 (m : ℝ) (h : dot_product vector_a (vector_b m) = 0) : m = 2 := sorry

-- Part 2: Given that the magnitude of vector_a + vector_b is 5, prove that m = 2 or m = -6.
theorem part2 (m : ℝ) (h : magnitude (add_vectors vector_a (vector_b m)) = 5) : m = 2 ∨ m = -6 := sorry

end part1_part2_l271_271932


namespace sticks_form_equilateral_triangle_l271_271700

theorem sticks_form_equilateral_triangle {n : ℕ} (h1 : n ≥ 5) (h2 : n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5) :
  (∃ a b c, a = b ∧ b = c ∧ a + b + c = (n * (n + 1)) / 2) :=
by
  sorry

end sticks_form_equilateral_triangle_l271_271700


namespace path_length_traversed_l271_271540

-- Definitions based on conditions in a)
def triangle_ABC := {a b c : ℝ // a = 6 ∧ b = 8 ∧ c = 10}
def circle_radius := (r : ℝ) (h : r = 1)

-- The proof statement based on the problem requirements
theorem path_length_traversed
  (T : triangle_ABC)
  (rP : circle_radius) :
  ∃ P : ℝ, P = 12 :=
sorry

end path_length_traversed_l271_271540


namespace joe_two_different_fruits_l271_271106

-- Define the set of fruits
inductive Fruit
| apple | orange | banana | grape

open Fruit

-- Define the probability space for one selection
def fruit_prob_space : MeasureSpace Fruit :=
  probability_space {Finset.univ}

-- Define the event Joe eats the same fruit for all meals
def same_fruit_for_all_meals : set (Fruit × Fruit × Fruit × Fruit) :=
  {x | x.1 = x.2 ∧ x.2 = x.3 ∧ x.3 = x.4}

-- Calculate the probability of Joe eating the same fruit for all meals
noncomputable def prob_same_fruit : ℚ :=
  (1 / 4) ^ 4 * 4

-- Define the probability of eating at least two different kinds of fruit
noncomputable def prob_at_least_two_different_fruits : ℚ :=
  1 - prob_same_fruit

theorem joe_two_different_fruits :
  prob_at_least_two_different_fruits = 63 / 64 :=
by
  unfold prob_at_least_two_different_fruits
  unfold prob_same_fruit
  sorry

end joe_two_different_fruits_l271_271106


namespace sum_first_5n_eq_630_l271_271071

theorem sum_first_5n_eq_630 (n : ℕ)
  (h : (4 * n * (4 * n + 1)) / 2 = (2 * n * (2 * n + 1)) / 2 + 300) :
  (5 * n * (5 * n + 1)) / 2 = 630 :=
sorry

end sum_first_5n_eq_630_l271_271071


namespace find_fourth_power_sum_l271_271059

theorem find_fourth_power_sum (a b c : ℝ) 
    (h1 : a + b + c = 2) 
    (h2 : a^2 + b^2 + c^2 = 3) 
    (h3 : a^3 + b^3 + c^3 = 4) : 
    a^4 + b^4 + c^4 = 7.833 :=
sorry

end find_fourth_power_sum_l271_271059


namespace taxi_fare_max_distance_l271_271187

-- Setting up the conditions
def starting_price : ℝ := 7
def additional_fare_per_km : ℝ := 2.4
def max_base_distance_km : ℝ := 3
def total_fare : ℝ := 19

-- Defining the maximum distance based on the given conditions
def max_distance : ℝ := 8

-- The theorem is to prove that the maximum distance is indeed 8 kilometers
theorem taxi_fare_max_distance :
  ∀ (x : ℝ), total_fare = starting_price + additional_fare_per_km * (x - max_base_distance_km) → x ≤ max_distance :=
by
  intros x h
  sorry

end taxi_fare_max_distance_l271_271187


namespace total_cost_of_purchases_l271_271311

theorem total_cost_of_purchases :
  let cost_of_snake_toy := 11.76
  let cost_of_cage := 14.54
  let found_dollar := 1.00
  cost_of_snake_toy + cost_of_cage - found_dollar = 25.30 :=
by {
  let cost_of_snake_toy := 11.76
  let cost_of_cage := 14.54
  let found_dollar := 1.00
  show cost_of_snake_toy + cost_of_cage - found_dollar = 25.30,
  sorry
}

end total_cost_of_purchases_l271_271311


namespace minimum_sum_is_correct_maximum_product_is_correct_expression_value_is_correct_l271_271291

def rational_numbers : List ℚ := [-3, -2, 0, 1, 3]

def minimum_sum : ℚ := -5
def maximum_product : ℚ := 18
def expression_value : ℚ := -15 / 4

-- Prove the minimum sum is -5
theorem minimum_sum_is_correct : 
  (∃ (a : ℚ), a = minimum_sum) ↔ 
  (∃ (nums : List ℚ), nums ⊆ rational_numbers ∧ nums.length = 3 ∧ nums.sum = minimum_sum) := 
  sorry

-- Prove the maximum product is 18
theorem maximum_product_is_correct : 
  (∃ (b : ℚ), b = maximum_product) ↔ 
  (∃ (nums : List ℚ), nums ⊆ rational_numbers ∧ nums.length = 3 ∧ (nums.prod = b)) := 
  sorry

-- Prove the value of the expression given x = 4 and y = 5
theorem expression_value_is_correct : 
  (∃ (x y : ℚ), x = 4 ∧ y = 5 ∧ (|x - 4| + (y + minimum_sum)^2 = 0) ∧ 
  (frac y^2 x - (1 / 2) * x * y = expression_value)) :=
  sorry

end minimum_sum_is_correct_maximum_product_is_correct_expression_value_is_correct_l271_271291


namespace triple_integral_value_l271_271302

theorem triple_integral_value :
  (∫ x in (-1 : ℝ)..1, ∫ y in (x^2 : ℝ)..1, ∫ z in (0 : ℝ)..y, (4 + z) ) = (16 / 3 : ℝ) :=
by
  sorry

end triple_integral_value_l271_271302


namespace square_subset_rectangle_l271_271804

-- Definitions based on given conditions
def is_parallelogram (x : Type) : Prop := x ∈ A
def is_rectangle (x : Type) : Prop := x ∈ B
def is_square (x : Type) : Prop := x ∈ C
def is_rhombus (x : Type) : Prop := x ∈ D

-- The lean statement for the problem
theorem square_subset_rectangle : C ⊆ B := by
  sorry -- Proof goes here

end square_subset_rectangle_l271_271804


namespace students_with_dogs_l271_271632

theorem students_with_dogs (total_students : ℕ) (half_students : total_students / 2 = 50)
  (percent_girls_with_dogs : ℕ → ℚ) (percent_boys_with_dogs : ℕ → ℚ)
  (girls_with_dogs : ∀ (total_girls: ℕ), percent_girls_with_dogs total_girls = 0.2)
  (boys_with_dogs : ∀ (total_boys: ℕ), percent_boys_with_dogs total_boys = 0.1) :
  ∀ (total_girls total_boys students_with_dogs: ℕ),
  total_students = 100 →
  total_girls = total_students / 2 →
  total_boys = total_students / 2 →
  total_girls = 50 →
  total_boys = 50 →
  students_with_dogs = (percent_girls_with_dogs (total_students / 2) * (total_students / 2) + 
                        percent_boys_with_dogs (total_students / 2) * (total_students / 2)) →
  students_with_dogs = 15 :=
by
  intros total_girls total_boys students_with_dogs h1 h2 h3 h4 h5 h6
  sorry

end students_with_dogs_l271_271632


namespace arithmetic_sequence_S7_geometric_sequence_k_l271_271770

noncomputable def S_n (a d : ℕ) (n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequence_S7 (a_4 : ℕ) (h : a_4 = 8) : S_n a_4 1 7 = 56 := by
  sorry

def Sn_formula (k : ℕ) : ℕ := k^2 + k
def a (i d : ℕ) := i * d

theorem geometric_sequence_k (a_1 k : ℕ) (h1 : a_1 = 2) (h2 : (2 * k + 2)^2 = 6 * (k^2 + k)) :
  k = 2 := by
  sorry

end arithmetic_sequence_S7_geometric_sequence_k_l271_271770


namespace find_area_and_coordinates_of_point_P_l271_271160

def is_on_ellipse (P : ℝ × ℝ) : Prop :=
  let (x, y) := P in (x^2) / 25 + (y^2) / 9 = 1

def is_focus (F : ℝ × ℝ) (c : ℝ) (b : ℝ) : Prop :=
  let (Fx, Fy) := F in Fy = 0 ∧ (Fx = c ∨ Fx = -c)

noncomputable def area_triangle (P F1 F2 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := F1
  let (x2, y2) := F2
  let (xP, yP) := P
  abs ((x1 * (y2 - yP) + x2 * (yP - y1) + xP * (y1 - y2)) / 2)

theorem find_area_and_coordinates_of_point_P (P : ℝ × ℝ) (F1 F2 : ℝ × ℝ) 
  (h1 : is_on_ellipse P) (h2 : is_focus F1 4 3) (h3 : is_focus F2 4 3) (angle : real.angle) 
  (h4 : angle = 60) : 
  area_triangle P F1 F2 = 3 * real.sqrt 3 ∧ 
    (P = (5 * real.sqrt 13 / 4, 3 * real.sqrt 3 / 4) ∨ 
     P = (5 * real.sqrt 13 / 4, -3 * real.sqrt 3 / 4) ∨ 
     P = (-5 * real.sqrt 13 / 4, 3 * real.sqrt 3 / 4) ∨ 
     P = (-5 * real.sqrt 13 / 4, -3 * real.sqrt 3 / 4)) :=
by 
  sorry

end find_area_and_coordinates_of_point_P_l271_271160


namespace trapezoid_upper_base_BC_l271_271768

theorem trapezoid_upper_base_BC (A B C D M : Point) (d : ℝ)
  (h1 : Trapezoid A B C D)
  (h2 : OnLine M A B)
  (h3 : Perpendicular D M A B)
  (h4 : Distance M C = Distance C D)
  (h5 : Distance A D = d) : Distance B C = d / 2 := 
sorry

end trapezoid_upper_base_BC_l271_271768


namespace probability_ace_second_draw_l271_271240

theorem probability_ace_second_draw
  (total_cards : ℕ := 52)
  (initial_aces : ℕ := 4)
  (drawn_aces : ℕ := 1)
  :
  let remaining_cards := total_cards - drawn_aces,
      remaining_aces := initial_aces - drawn_aces
  in remaining_aces / remaining_cards = 1 / 17 :=
by
  -- Conditions
  have card_count : total_cards = 52 := rfl
  have initial_ace_count : initial_aces = 4 := rfl
  have ace_drawn : drawn_aces = 1 := rfl
  -- Placeholder proof
  sorry

end probability_ace_second_draw_l271_271240


namespace part1_part2_l271_271914

namespace MathProof

-- defining the basic setup of the triangle and given constraints
variables (A B C : ℝ) (a b c : ℝ)

-- condition 1: the equation relating cosines and sines
def condition1 : Prop := (cos A) / (1 + sin A) = (sin (2 * B)) / (1 + cos (2 * B))

-- condition 2: specific value of angle C
def condition2 : Prop := C = 2 * π / 3

-- question 1: finding angle B
def findB : Prop := B = π / 6

-- The minimum value of (a^2 + b^2) / c^2
def min_value_expr : ℝ := (a^2 + b^2) / c^2
def min_value : ℝ := 4 * real.sqrt 2 - 5

-- Proving both parts
theorem part1 (h1 : condition1) (h2 : condition2) : findB := sorry

theorem part2 (h1 : condition1) (h2 : condition2) : min_value_expr = min_value := sorry

end MathProof

end part1_part2_l271_271914


namespace max_sum_geq_max_sorted_sums_l271_271809

theorem max_sum_geq_max_sorted_sums (n : ℕ) 
  (a b : Finₓ n → ℝ)
  (ha : ∀ i j : Finₓ n, i ≤ j → a i ≤ a j)
  (hb : ∀ i j : Finₓ n, i ≤ j → b i ≥ b j) :
  max (Finₓ.foldl (λ x y, max x y) (a 0 + b 0) (λ i, a i + b i)) 
      (Finₓ.foldl (λ x y, max x y) (a 0 + b n) (λ i, a i + b (n - i))) :=
sorry

end max_sum_geq_max_sorted_sums_l271_271809


namespace min_value_ineq_l271_271808

noncomputable def problem_statement : Prop :=
  ∃ (x y : ℝ), 0 < x ∧ 0 < y ∧ x + y = 4 ∧ (∀ (a b : ℝ), 0 < a → 0 < b → a + b = 4 → (1/a + 4/b) ≥ 9/4)

theorem min_value_ineq : problem_statement :=
by
  unfold problem_statement
  sorry

end min_value_ineq_l271_271808


namespace max_knights_with_courtiers_l271_271457

open Nat

def is_solution (a b : ℕ) : Prop :=
  12 ≤ a ∧ a ≤ 18 ∧
  10 ≤ b ∧ b ≤ 20 ∧
  1 / (a:ℝ).toNat + 1 / (b:ℝ).toNat = 1 / 7

theorem max_knights_with_courtiers : ∃ a b : ℕ, is_solution a b ∧ b = 14 ∧ a = 14 :=
by
  sorry

end max_knights_with_courtiers_l271_271457


namespace prob_B_given_A_l271_271091

theorem prob_B_given_A (P_A P_B P_A_and_B : ℝ) (h1 : P_A = 0.06) (h2 : P_B = 0.08) (h3 : P_A_and_B = 0.02) :
  (P_A_and_B / P_A) = (1 / 3) :=
by
  -- substitute values
  sorry

end prob_B_given_A_l271_271091


namespace sticks_form_equilateral_triangle_l271_271706

theorem sticks_form_equilateral_triangle (n : ℕ) :
  n ≥ 5 → (n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5) → 
  ∃ k : ℕ, ∑ i in finset.range (n + 1), i = 3 * k :=
by {
  sorry
}

end sticks_form_equilateral_triangle_l271_271706


namespace part1_problem_l271_271927

theorem part1_problem
  (A B C : Real.Angle)
  (a b c : ℝ)
  (cosA : Real.cos A)
  (sinA : Real.sin A)
  (sin2B : Real.sin (2 * B))
  (cos2B : Real.cos (2 * B))
  (hC : C = 2 * π / 3)
  (h : cosA / (1 + sinA) = sin2B / (1 + cos2B))
  : B = π / 6 := by
  sorry

end part1_problem_l271_271927


namespace cyclic_quadrilateral_angles_l271_271679

theorem cyclic_quadrilateral_angles (A B C D : ℝ) (h_cyclic : A + C = 180) (h_diag_bisect : (A = 2 * (B / 5 + B / 5)) ∧ (C = 2 * (D / 5 + D / 5))) (h_ratio : B / D = 2 / 3):
  A = 80 ∨ A = 100 ∨ A = 1080 / 11 ∨ A = 900 / 11 :=
  sorry

end cyclic_quadrilateral_angles_l271_271679


namespace solve_inequality_l271_271961

theorem solve_inequality :
  {x : ℝ | (x - 1) * (2 * x + 1) ≤ 0} = { x : ℝ | -1/2 ≤ x ∧ x ≤ 1 } :=
by
  sorry

end solve_inequality_l271_271961


namespace radius_of_circular_film_l271_271142

theorem radius_of_circular_film
  (a b c : ℝ)
  (thickness : ℝ)
  (volume : ℝ)
  (h_volume : volume = a * b * c)
  (h_thickness : thickness = 0.1):
  let r := Real.sqrt (volume / (π * thickness)) in
  r = Real.sqrt (2160 / π) :=
by
  sorry

end radius_of_circular_film_l271_271142


namespace income_calculation_l271_271194

theorem income_calculation
  (x : ℕ)
  (income : ℕ := 5 * x)
  (expenditure : ℕ := 4 * x)
  (savings : ℕ := income - expenditure)
  (savings_eq : savings = 3000) :
  income = 15000 :=
sorry

end income_calculation_l271_271194


namespace train_length_l271_271599

def speed : ℝ := 60 -- speed in km/hr
def time_sec : ℝ := 30 -- time in seconds
def time_hr : ℝ := time_sec / 3600 -- time converted to hours
def length_km : ℝ := speed * time_hr -- length in kilometers

theorem train_length
  (speed : ℝ := 60)
  (time_sec : ℝ := 30)
  (time_hr := time_sec / 3600)
  (length_km := speed * time_hr) :
  length_km * 1000 = 500 := 
sorry

end train_length_l271_271599


namespace problem_solution_l271_271474

theorem problem_solution (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)
    (h4 : 2^x = 3^y) (h5 : 3^y = 4^z) :
    2 * x = 4 * z ∧ 4 * z > 3 * y := 
sorry

end problem_solution_l271_271474


namespace range_of_half_x_l271_271202

noncomputable def function_range (f : ℝ → ℝ) (E : set ℝ) : set ℝ :=
  {y : ℝ | ∃ x ∈ E, f x = y}

theorem range_of_half_x (x : ℝ) (hx : x ≥ 8) : 
  function_range (λ x, (1/2)^x) {x | x ≥ 8} = {y | 0 < y ∧ y ≤ (1/2)^8} :=
by
  sorry

end range_of_half_x_l271_271202


namespace sufficient_condition_l271_271507

variables {P Q R S T : Prop}

theorem sufficient_condition (hpq : P → Q)
  (hqr : R → Q)
  (hqs : Q ↔ S)
  (hts : S → T)
  (htr : T → R) : P → T :=
by 
  assume hp : P,
  sorry

end sufficient_condition_l271_271507


namespace f_is_odd_l271_271796

def f (x : ℝ) : ℝ :=
  if x > 0 then x^2
  else -x^2

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f (x) := by
  sorry

end f_is_odd_l271_271796


namespace num_integers_containing_3_5_7_in_range_l271_271053

theorem num_integers_containing_3_5_7_in_range : 
  ∃ num_integers : ℕ, 
    num_integers = 6 ∧ 
    (∀ n, 600 ≤ n ∧ n < 2000 → 
         (∃ d₁ d₂ d₃, d₁ ∈ {3, 5, 7} ∧ d₂ ∈ {3, 5, 7} ∧ d₃ ∈ {3, 5, 7} ∧ ∀ d, d ∈ {d₁, d₂, d₃} → d=3 ∨ d=5 ∨ d=7) → 
         num_integers = 6) := 
by 
  sorry

end num_integers_containing_3_5_7_in_range_l271_271053


namespace largest_last_digit_l271_271192

theorem largest_last_digit (s : List ℕ) (h_length : s.length = 2003) (h_first : s.head = 1)
    (h_cond : ∀ n, n ∈ (List.zip s s.tail |>.map (λ p => p.1 * 10 + p.2)) 
        → (n % 17 = 0 ∨ n % 19 = 0)) : s.getLast '0 = 8 :=
by
  sorry

end largest_last_digit_l271_271192


namespace division_correct_l271_271331

-- Definitions based on conditions
def expr1 : ℕ := 12 + 15 * 3
def expr2 : ℚ := 180 / expr1

-- Theorem statement using the question and correct answer
theorem division_correct : expr2 = 180 / 57 := by
  sorry

end division_correct_l271_271331


namespace chocolate_bars_percentage_l271_271933

noncomputable def total_chocolate_bars (milk dark almond white caramel : ℕ) : ℕ :=
  milk + dark + almond + white + caramel

noncomputable def percentage (count total : ℕ) : ℚ :=
  (count : ℚ) / (total : ℚ) * 100

theorem chocolate_bars_percentage :
  let milk := 36
  let dark := 21
  let almond := 40
  let white := 15
  let caramel := 28
  let total := total_chocolate_bars milk dark almond white caramel
  total = 140 ∧
  percentage milk total = 25.71 ∧
  percentage dark total = 15 ∧
  percentage almond total = 28.57 ∧
  percentage white total = 10.71 ∧
  percentage caramel total = 20 :=
by
  sorry

end chocolate_bars_percentage_l271_271933


namespace v_expression_max_traffic_flow_l271_271945

-- Definitions for conditions
def v (x : ℝ) : ℝ :=
if h : 0 ≤ x ∧ x ≤ 30 then 50
else if h : 30 < x ∧ x ≤ 280 then -0.2 * x + 56
else 0  -- should never be used

def f (x : ℝ) : ℝ := x * v x

-- Theorem statements
theorem v_expression (x : ℝ) (h : 0 ≤ x ∧ x ≤ 280) :
  v x = (if 0 ≤ x ∧ x ≤ 30 then 50 else -0.2 * x + 56) :=
sorry

theorem max_traffic_flow :
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 280 ∧ f x = 3920 ∧ ∀ y : ℝ, 0 ≤ y ∧ y ≤ 280 → f y ≤ 3920 :=
sorry

end v_expression_max_traffic_flow_l271_271945


namespace num_possible_point_totals_l271_271077

-- Define the parameters of the problem
def num_teams : ℕ := 5
def num_matches_per_team : ℕ := 8  -- Each team plays 2 matches against 4 other teams
def points_win : ℕ := 3
def points_draw : ℕ := 1
def points_loss : ℕ := 0

-- State the theorem
theorem num_possible_point_totals : 
  ∃ n, n = 24 ∧ (∀ pts, pts ∈ (set.range (λ x : ℕ, if x ≤ 8 then x*3 + (8 - x) * 1 else 0)) ↔ (pts ≥ 0 ∧ pts ≤ 24)) := 
sorry

end num_possible_point_totals_l271_271077


namespace common_ratio_l271_271469

theorem common_ratio (a_3 S_3 : ℝ) (q : ℝ) 
  (h1 : a_3 = 3 / 2) 
  (h2 : S_3 = 9 / 2)
  (h3 : S_3 = (1 + q + q^2) * a_3 / q^2) :
  q = 1 ∨ q = -1 / 2 := 
by 
  sorry

end common_ratio_l271_271469


namespace find_B_min_of_sum_of_squares_l271_271924

-- Given conditions in a)
variables {A B C a b c : ℝ}
hypothesis (h1 : C = 2 * Real.pi / 3)
hypothesis (h2 : (Real.cos A) / (1 + Real.sin A) = Real.sin (2 * B) / (1 + Real.cos (2 * B)))

-- Part (1) prove B = π / 6
theorem find_B (h1 : C = 2 * Real.pi / 3) (h2 : (Real.cos A) / (1 + Real.sin A) = Real.sin (2 * B) / (1 + Real.cos (2 * B))) : B = Real.pi / 6 :=
by sorry

-- Part (2) find the minimum value of (a^2 + b^2) / (c^2)
theorem min_of_sum_of_squares (h1 : C = 2 * Real.pi / 3) (h2 : (Real.cos A) / (1 + Real.sin A) = Real.sin (2 * B) / (1 + Real.cos (2 * B))) : ∃ (m : ℝ), m = 4 * Real.sqrt 2 - 5 ∧ ∀ a b c, a > 0 ∧ b > 0 ∧ c > 0 → (a^2 + b^2) / c^2 ≥ m :=
by sorry

end find_B_min_of_sum_of_squares_l271_271924


namespace triangle_construction_l271_271366

-- Define the problem statement in Lean
theorem triangle_construction (a b c : ℝ) :
  correct_sequence = [3, 1, 4, 2] :=
sorry

end triangle_construction_l271_271366


namespace complex_number_identity_l271_271781

theorem complex_number_identity (m : ℝ) (h : m + ((m ^ 2 - 4) * Complex.I) = Complex.re 0 + 1 * Complex.I ↔ m > 0): 
  (Complex.mk m 2 * Complex.mk 2 (-2)⁻¹) = Complex.I := sorry

end complex_number_identity_l271_271781


namespace parallelogram_of_bisecting_diagonals_l271_271647

-- Define a quadrilateral with diagonals bisecting each other
structure Quadrilateral where
  A B C D : Type
  (diagonal1 : A → C)
  (diagonal2 : B → D)
  (bisect_each_other : ∃ M, diagonal1 A = diagonal2 B ∧ diagonal1 C = diagonal2 D)

-- Main theorem stating that if the diagonals of a quadrilateral bisect each other, then it is a parallelogram.
theorem parallelogram_of_bisecting_diagonals :
  ∀ (Q : Quadrilateral),
  (∃ M, Q.bisect_each_other) → (Q,A,B,C,D) is a parallelogram :=
sorry

end parallelogram_of_bisecting_diagonals_l271_271647


namespace expression_value_l271_271235

theorem expression_value :
  (2^4 - 3^4) / (2^(-4) + 2^(-4)) = -520 :=
by
  sorry

end expression_value_l271_271235


namespace second_player_wins_l271_271228

theorem second_player_wins :
  ∀ P : ℕ, P = 1000 ∨ P < 1000 →
  (∀ m : ℕ, 1 ≤ m ∧ m ≤ 5 →
   (∀ t : ℕ, t = 1000 - m ∨ (t < 1000 ∧ 0 ≤ t)) →
    ∃ n : ℕ, 1 ≤ n ∧ n ≤ 5 ∧
    ∃ N : ℕ, (N = 1000 - n ∨ (N < 1000 ∧ 0 ≤ N)) ∧
    ((P - m = P - n → false) ∨ (∃ k : ℕ, P - k = P - n ∧ k ≤ 5 ∧ 1 ≤ k))) :=
begin
  sorry
end

end second_player_wins_l271_271228


namespace clothing_price_190_l271_271611

/-- A clothing retailer purchases a first batch of children's clothing at a price of
  3 pieces for 160 yuan, and then purchases twice as many children's clothing as the
  first batch at a price of 4 pieces for 210 yuan. He wants to sell all these clothes
  and make a 20% profit. Therefore, he needs to sell them at a price of 3 pieces for 190 yuan. -/
theorem clothing_price_190 (a : ℕ) :
  let cost1 := 160 * a / 3,
      cost2 := 210 * 2 * a / 4,
      total_cost := cost1 + cost2,
      total_revenue := 3 * a * 190 / 3
  in  total_revenue = total_cost * 1.20 :=
sorry

end clothing_price_190_l271_271611


namespace first_question_second_question_l271_271260

/-- First Question Proof Statement: -/
theorem first_question (l m : ℝ → ℝ) (h1 : ∀ x : ℝ, m x = -sqrt 3 * x + 1) (h2 : l = λ x, -((sqrt 3) / 3) * x + d )
  (h3 : ∃ (d : ℝ), l = λ x, -((sqrt 3) / 3) * x + d) : 
  ∃ (a b c : ℝ), (a * 2 + b * 2 + c = 0) → (sqrt 3 * 2 + 3 * 2 - 6 - 2 * sqrt 3 = 0) :=
sorry

/-- Second Question Proof Statement: -/
theorem second_question (Q : ℝ × ℝ) (hQ : Q = (3, -2)) (l : ℝ → ℝ)
  (h1 : ∃ a b : ℝ, (a ≠ 0) ∧ (b ≠ 0) ∧ (l = λ x, (b/a) * x) ∧ (l 3 = -2)) :
  (∀ x : ℝ, (2 * x + 3 * (l x) = 0 ∨ x - l x - 5 = 0)) :=
sorry

end first_question_second_question_l271_271260


namespace penelope_savings_l271_271948

theorem penelope_savings (daily_savings : ℕ) (days_in_year : ℕ) (total_savings : ℕ) : 
  (daily_savings = 24) → 
  (days_in_year = 365) → 
  (total_savings = daily_savings * days_in_year) → 
  (total_savings = 8760) := 
by 
  intros h1 h2 h3 
  rw [h1, h2] at h3 
  exact h3
  sorry

end penelope_savings_l271_271948


namespace z_div_second_quadrant_l271_271832

noncomputable def z1 : ℂ := 2 - I

-- z1 and z2 are symmetric about the y-axis
noncomputable def z2 : ℂ := -2 - I

-- Define the division of z1 by z2
noncomputable def z_div : ℂ := z1 / z2

-- Prove that z_div is in the second quadrant
theorem z_div_second_quadrant (h1 : z1 = 2 - I) (h2 : z2 = -2 - I) : 
  0 < z_div.im ∧ z_div.re < 0 :=
by
  have h_div : z_div = -3 / 5 + 4 / 5 * I := by sorry
  rw [h_div]
  split
  sorry -- 0 < 4 / 5
  sorry -- -3 / 5 < 0

end z_div_second_quadrant_l271_271832


namespace sin_half_x_sin_five_half_x_l271_271402

theorem sin_half_x_sin_five_half_x (x : ℝ) (h : sin (5 * π / 2 - x) = 3 / 5) : 
  sin (x / 2) * sin (5 * x / 2) = 86 / 125 :=
by 
  sorry

end sin_half_x_sin_five_half_x_l271_271402


namespace find_a_l271_271412

theorem find_a (α β : ℝ) (h1 : α + β = 10) (h2 : α * β = 20) : (1 / α + 1 / β) = 1 / 2 :=
sorry

end find_a_l271_271412


namespace solve_rate_of_interest_l271_271597

noncomputable def principal : ℝ := 12500
noncomputable def amount_after_4_years : ℝ := 15500
noncomputable def time : ℝ := 4

theorem solve_rate_of_interest : 
  let SI := amount_after_4_years - principal in
  let rate := (SI * 100) / (principal * time) in
  rate = 6 := by 
  -- start proof

  sorry

end solve_rate_of_interest_l271_271597


namespace f_has_two_zeros_l271_271393

noncomputable def f (a x : ℝ) : ℝ := a * x^2 + (a - 2) * x - Real.log x

theorem f_has_two_zeros (a : ℝ) (ha : 0 < a ∧ a < 1) :
  ∃ x1 x2 : ℝ, 0 < x1 ∧ 0 < x2 ∧ x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0 := sorry

end f_has_two_zeros_l271_271393


namespace Jamie_earns_10_per_hour_l271_271461

noncomputable def JamieHourlyRate (days_per_week : ℕ) (hours_per_day : ℕ) (weeks : ℕ) (total_earnings : ℕ) : ℕ :=
  let total_hours := days_per_week * hours_per_day * weeks
  total_earnings / total_hours

theorem Jamie_earns_10_per_hour :
  JamieHourlyRate 2 3 6 360 = 10 := by
  sorry

end Jamie_earns_10_per_hour_l271_271461


namespace tangent_lines_to_circle_fixed_point_diameter_PM_l271_271667

theorem tangent_lines_to_circle (k a : ℝ) :
  let M := mk_circle 4 2 4 in
  ∃ m b, tangent_line M m b ∧ (∀ x: ℝ, y: ℝ, (m * x + b = y) → (x-intercept = 2 * y-intercept)) → 
  (tangent_line_eq M (m, b)) = (m = 4/3) ∨ 
  (tangent_line_eq M (m, b)) = (a = 4 + sqrt(5)) ∨ 
  (tangent_line_eq M (m, b)) = (a = 4 - sqrt(5)) := sorry

theorem fixed_point_diameter_PM (a b : ℝ) (p : Point) :
  ∀ (a b : ℝ), 2a + b = 2 →
  let P := mk_point a b in
  let O := origin() in
  tangent_to_circle P O (PA = PO) → 
  fixed_point_eq (x, y) with P M diameter
  ,
  (x, y) = (4/5, 2/5) := sorry

end tangent_lines_to_circle_fixed_point_diameter_PM_l271_271667


namespace common_difference_is_1_l271_271082

variable (a_2 a_5 : ℕ) (d : ℤ)

def arithmetic_sequence (n a_1 : ℤ) (d : ℤ) : ℤ := a_1 + (n - 1) * d

theorem common_difference_is_1 
  (h1 : arithmetic_sequence 2 a_1 d = 3) 
  (h2 : arithmetic_sequence 5 a_1 d = 6) : 
  d = 1 := 
sorry

end common_difference_is_1_l271_271082


namespace division_to_fraction_fraction_to_division_mixed_to_improper_fraction_whole_to_fraction_l271_271330

theorem division_to_fraction : (7 / 9) = 7 / 9 := by
  sorry

theorem fraction_to_division : 12 / 7 = 12 / 7 := by
  sorry

theorem mixed_to_improper_fraction : (3 + 5 / 8) = 29 / 8 := by
  sorry

theorem whole_to_fraction : 6 = 66 / 11 := by
  sorry

end division_to_fraction_fraction_to_division_mixed_to_improper_fraction_whole_to_fraction_l271_271330


namespace largest_multiple_of_8_smaller_than_neg_80_l271_271567

theorem largest_multiple_of_8_smaller_than_neg_80 :
  ∃ n : ℤ, (8 ∣ n) ∧ n < -80 ∧ ∀ m : ℤ, (8 ∣ m ∧ m < -80 → m ≤ n) :=
sorry

end largest_multiple_of_8_smaller_than_neg_80_l271_271567


namespace compute_abs_a_plus_b_plus_c_l271_271509

variable (a b c : ℝ)

theorem compute_abs_a_plus_b_plus_c (h1 : a^2 - b * c = 14)
                                   (h2 : b^2 - c * a = 14)
                                   (h3 : c^2 - a * b = -3) :
                                   |a + b + c| = 5 :=
sorry

end compute_abs_a_plus_b_plus_c_l271_271509


namespace prove_a_plus_b_l271_271823

-- Defining the function f(x)
def f (a b x: ℝ) : ℝ := a * x^2 + b * x

-- The given conditions
variable (a b : ℝ)
variable (h1 : f a b (a - 1) = f a b (2 * a))
variable (h2 : ∀ x : ℝ, f a b x = f a b (-x))

-- The objective is to show a + b = 1/3
theorem prove_a_plus_b (a b : ℝ) (h1 : f a b (a - 1) = f a b (2 * a)) (h2 : ∀ x : ℝ, f a b x = f a b (-x)) :
  a + b = 1 / 3 := 
sorry

end prove_a_plus_b_l271_271823


namespace trigonometric_identity_l271_271355

theorem trigonometric_identity (α : ℝ) 
  (h : Real.tan (Real.pi / 4 - α) = 1 / 2) :
  (Real.sin (2 * α) + Real.sin(α) ^ 2) / (1 + Real.cos (2 * α)) = 7 / 18 := 
  sorry

end trigonometric_identity_l271_271355


namespace sqrt3_times_3_minus_sqrt3_bound_l271_271323

theorem sqrt3_times_3_minus_sqrt3_bound : 2 < (Real.sqrt 3) * (3 - (Real.sqrt 3)) ∧ (Real.sqrt 3) * (3 - (Real.sqrt 3)) < 3 := 
by 
  sorry

end sqrt3_times_3_minus_sqrt3_bound_l271_271323


namespace not_prime_for_some_n_l271_271111

theorem not_prime_for_some_n (a : ℕ) (h : 1 < a) : ∃ n : ℕ, ¬ Nat.Prime (2^(2^n) + a) := 
sorry

end not_prime_for_some_n_l271_271111


namespace sum_of_solutions_is_1850pi_l271_271723

noncomputable def find_sum_of_solutions : ℝ :=
  let eq : ℝ → Prop := λ x, 3 * cos (2 * x) * (cos (2 * x) - cos (2000 * π^2 / x)) = cos (4 * x) - 1
  (∑ x in {x : ℝ | x > 0 ∧ eq x}, x)

theorem sum_of_solutions_is_1850pi : find_sum_of_solutions = 1850 * π :=
  sorry

end sum_of_solutions_is_1850pi_l271_271723


namespace incenter_of_triangle_l271_271371

variables {A B C E F O : Point}

-- Definition assuming A O is the altitude from vertex A to the base E F
def is_altitude (A O : Point) (EF : Line) : Prop :=
  O ∈ EF ∧ orthogonal A O EF

-- Definition assuming isosceles triangle A E F with AO = EF
def is_isosceles_triangle_and_altitude_eq_base (A E F O : Point) : Prop :=
  is_isosceles A E F ∧ is_altitude A O (line E F) ∧ dist A O = dist E F

-- Defined points BE = AE
def is_extension_equal (A E B : Point) : Prop :=
  dist A E = dist B E

-- Defined perpendicular from point B to line AF meeting at C
def is_perpendicular (B C A F : Point) : Prop :=
  orthogonal B C (line A F) ∧ B C ∈ (line AF)

-- Main statement
theorem incenter_of_triangle (A B C E F O : Point) 
  (h1 : is_isosceles_triangle_and_altitude_eq_base A E F O)
  (h2 : is_extension_equal A E B)
  (h3 : is_perpendicular B C A F) :
  is_incenter O (triangle A B C) :=
sorry

end incenter_of_triangle_l271_271371


namespace bounded_squares_after_deletion_l271_271348

open Nat

theorem bounded_squares_after_deletion (N : ℕ) :
  ∃ C : ℕ, ∀ (a_i : ℕ), (a_i ∈ { delete_digit N i | i : ℕ }) →
  (∃ k, k * k = a_i) → { a_i | ∃ k, k * k = a_i }.finite ∧
  {a_i | a_i ∈ (λ i, delete_digit N i) ∧ ∃ k : ℕ, k * k = a_i }.card ≤ C := 
sorry

def delete_digit (N : ℕ) (i : ℕ) : ℕ :=
  let digits := repr N;
  let new_digits := digits.remove nth digits i;
  val new_digits to ℕ

end bounded_squares_after_deletion_l271_271348


namespace total_cost_of_items_l271_271968

variable (M R F : ℝ)
variable (h1 : 10 * M = 24 * R)
variable (h2 : F = 2 * R)
variable (h3 : F = 21)

theorem total_cost_of_items : 4 * M + 3 * R + 5 * F = 237.3 :=
by
  sorry

end total_cost_of_items_l271_271968


namespace joe_fruit_probability_l271_271463

theorem joe_fruit_probability :
  let prob_same := (1 / 4) ^ 3
  let total_prob_same := 4 * prob_same
  let prob_diff := 1 - total_prob_same
  prob_diff = 15 / 16 :=
by
  sorry

end joe_fruit_probability_l271_271463


namespace always_true_inequality_l271_271674

theorem always_true_inequality (x : ℝ) : x^2 + 1 ≥ 2 * |x| := 
sorry

end always_true_inequality_l271_271674


namespace part1_part2_l271_271916

namespace MathProof

-- defining the basic setup of the triangle and given constraints
variables (A B C : ℝ) (a b c : ℝ)

-- condition 1: the equation relating cosines and sines
def condition1 : Prop := (cos A) / (1 + sin A) = (sin (2 * B)) / (1 + cos (2 * B))

-- condition 2: specific value of angle C
def condition2 : Prop := C = 2 * π / 3

-- question 1: finding angle B
def findB : Prop := B = π / 6

-- The minimum value of (a^2 + b^2) / c^2
def min_value_expr : ℝ := (a^2 + b^2) / c^2
def min_value : ℝ := 4 * real.sqrt 2 - 5

-- Proving both parts
theorem part1 (h1 : condition1) (h2 : condition2) : findB := sorry

theorem part2 (h1 : condition1) (h2 : condition2) : min_value_expr = min_value := sorry

end MathProof

end part1_part2_l271_271916


namespace coin_combinations_50_cents_l271_271404

-- Define the value of each coin
def penny : ℕ := 1
def nickel : ℕ := 5
def quarter : ℕ := 25

-- Define a function to count the combinations
def count_combinations (target : ℕ) : ℕ :=
  let combinations := (λ p n q, p * penny + n * nickel + q * quarter = target) in
  ((λ n, n * 1 + 0 * 5 + 0 * 25 = target) -- only pennies
  + (∑ n in finset.range (target / nickel + 1), (λ p, p * 1 + n * 5 + 0 * 25 = target)) -- pennies and nickels
  + (∑ q in finset.range (target / quarter + 1), (λ p, p * 1 + 0 * 5 + q * 25 = target) ) -- pennies and quarters
  + (λ n q, n * 5 + q * 25 = target) -- nickels and quarters
  + if target % 5 = 0 then 1 else 0) -- only nickels
  + if target % 25 = 0 then 1 else 0 -- only quarters
  sorry -- Implementation to count the combinations

-- A theorem that states the number of combinations of 1, 5, and 25 cent coins to make 50 cents is 20
theorem coin_combinations_50_cents : count_combinations 50 = 20 :=
  sorry

end coin_combinations_50_cents_l271_271404


namespace rect_faces_hexahedron_cuboid_l271_271586

-- assume the definitions of a prism, a cuboid and a hexahedron
def prism (P : Type) := ∃ (top bottom : P) (lateral_edges : set (P × P)), true 
def cuboid (C : Type) := prism C ∧ ∃ (rect_faces : set C), true 
def hexahedron (H : Type) := ∃ (faces : set H), true 

-- Given conditions
axiom prism_is_cuboid (P : Type) (h : prism P) : cuboid P
axiom rect_base_prism_is_cuboid (P : Type) (h : prism P) (hb : ∃ (base : P), true) : cuboid P
axiom hexahedron_is_cuboid (H : Type) (h : hexahedron H) : cuboid H
axiom rect_faces_hexahedron_is_cuboid (H : Type) (h : hexahedron H) (hf : ∃ (rect_faces : set H), true) : cuboid H

-- We need to prove that D is correct based on the given conditions
theorem rect_faces_hexahedron_cuboid (H : Type) (h : hexahedron H) (hf : ∃ (rect_faces : set H), true) :
  cuboid H := sorry

end rect_faces_hexahedron_cuboid_l271_271586


namespace min_trips_calculation_l271_271553

noncomputable def min_trips (total_weight : ℝ) (truck_capacity : ℝ) : ℕ :=
  ⌈total_weight / truck_capacity⌉₊

theorem min_trips_calculation : min_trips 18.5 3.9 = 5 :=
by
  -- Proof goes here
  sorry

end min_trips_calculation_l271_271553


namespace upper_base_length_l271_271764

-- Definitions based on the given conditions
variables (A B C D M : ℝ)
variables (d : ℝ)
variables (h : ℝ) -- height from D to AB

-- Conditions given in the problem
def is_trapezoid (ABCD : ℝ) : Prop := true
def is_perpendicular (DM AB : ℝ) : Prop := true
def point_on_side (M AB : ℝ) : Prop := true
def MC_eq_CD (MC CD : ℝ) : Prop := MC = CD
def AD_length (A D d : ℝ) : Prop := A - D = d -- Assuming some coordinate system 

-- Define the proof statement with conditions and the result
theorem upper_base_length
  (ht : is_trapezoid ABCD)
  (hp : is_perpendicular D M)
  (ps : point_on_side M AB)
  (mc_cd : MC_eq_CD M C D)
  (ad_len : AD_length A D d) :
  BC = d / 2 :=
sorry

end upper_base_length_l271_271764


namespace volume_of_solid_l271_271637

noncomputable def s : ℝ := 2 * Real.sqrt 2

noncomputable def h : ℝ := 3 * s

noncomputable def base_area (a b : ℝ) : ℝ := 1 / 2 * a * b

noncomputable def volume (base_area height : ℝ) : ℝ := base_area * height

theorem volume_of_solid : volume (base_area s s) h = 24 * Real.sqrt 2 :=
by
  -- The proof will go here
  sorry

end volume_of_solid_l271_271637


namespace ratio_of_perimeters_l271_271895

-- Define the side lengths of the squares
variables (x y : ℝ)

-- Define the diagonals in terms of the side lengths
def diag_first_square := x * real.sqrt 2
def diag_second_square := y * real.sqrt 2

-- State the given condition
axiom (h : diag_first_square = 1.5 * diag_second_square)

-- Define the perimeters of the squares
def perimeter_first_square := 4 * x
def perimeter_second_square := 4 * y

-- State the theorem for the required ratio
theorem ratio_of_perimeters (h : diag_first_square = 1.5 * diag_second_square) : 
  (perimeter_first_square / perimeter_second_square) = 1.5 := 
sorry

end ratio_of_perimeters_l271_271895


namespace certain_number_sum_l271_271210

/--
The sum of the even numbers between 1 and n is 85 times a certain number,
where n is an odd number. The value of n is 171. 
Prove that the certain number is 86.
-/
theorem certain_number_sum (n : ℕ) (h_odd : n % 2 = 1) (h_n_value : n = 171)
    (h_sum_eq : (finset.sum (finset.filter even (finset.range n)) = 85 * x)) :
    x = 86 := 
by
  sorry

end certain_number_sum_l271_271210


namespace sequence_sum_l271_271581

theorem sequence_sum (a b : ℤ) (h1 : ∃ d, d = 5 ∧ (∀ n : ℕ, (3 + n * d) = a ∨ (3 + (n-1) * d) = b ∨ (3 + (n-2) * d) = 33)) : 
  a + b = 51 :=
by
  sorry

end sequence_sum_l271_271581


namespace find_B_min_of_sum_of_squares_l271_271921

-- Given conditions in a)
variables {A B C a b c : ℝ}
hypothesis (h1 : C = 2 * Real.pi / 3)
hypothesis (h2 : (Real.cos A) / (1 + Real.sin A) = Real.sin (2 * B) / (1 + Real.cos (2 * B)))

-- Part (1) prove B = π / 6
theorem find_B (h1 : C = 2 * Real.pi / 3) (h2 : (Real.cos A) / (1 + Real.sin A) = Real.sin (2 * B) / (1 + Real.cos (2 * B))) : B = Real.pi / 6 :=
by sorry

-- Part (2) find the minimum value of (a^2 + b^2) / (c^2)
theorem min_of_sum_of_squares (h1 : C = 2 * Real.pi / 3) (h2 : (Real.cos A) / (1 + Real.sin A) = Real.sin (2 * B) / (1 + Real.cos (2 * B))) : ∃ (m : ℝ), m = 4 * Real.sqrt 2 - 5 ∧ ∀ a b c, a > 0 ∧ b > 0 ∧ c > 0 → (a^2 + b^2) / c^2 ≥ m :=
by sorry

end find_B_min_of_sum_of_squares_l271_271921


namespace three_f_l271_271062

noncomputable def f (x : ℝ) : ℝ := sorry

theorem three_f (x : ℝ) (hx : 0 < x) (h : ∀ y > 0, f (3 * y) = 5 / (3 + y)) :
  3 * f x = 45 / (9 + x) :=
by
  sorry

end three_f_l271_271062


namespace christopher_natalie_difference_l271_271128

-- Definitions
def T : ℕ := 6 -- Tyson's joggers based on condition solving
def Martha (T : ℕ) : ℕ := T - (2 * T / 3) -- Joggers Martha bought
def Alexander (T : ℕ) : ℕ := 3 * T / 2 -- Joggers Alexander bought
def Christopher (A M : ℕ) : ℕ := 18 * (A - M) -- Joggers Christopher bought

-- Natasha's joggers based on Christopher's count
def Natasha (C : ℕ) : ℕ := (4 * C) / 5 -- Natasha bought 80% of Christopher's joggers

-- Declaration of the theorem
theorem christopher_natalie_difference (T : ℕ) (C : ℕ) (A : ℕ) (M : ℕ) :
  Christopher A M = C → N = Natasha C → C = 126 → Alexander T = A → Martha T = M → 
  C - N = 26 :=
by {
  -- typical Lean proof formalities to ensure proper typing and conditions
  intro h1 h2 h3 h4 h5,
  rw [h1, h2], sorry
}

end christopher_natalie_difference_l271_271128


namespace sum_first_six_terms_l271_271448

-- Define the geometric sequence and conditions
def geom_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

-- Given conditions for the problem
variables (a : ℕ → ℝ)
variable (q : ℝ)
hypothesis h1 : a 1 = 2
hypothesis h2 : a 1 * q * a 1 * q^2 = 32

-- Define the sum of the first six terms of the sequence
def sum_geom_seq (a : ℕ → ℝ) (n : ℕ) :=
  if h : q ≠ 1 then
    (a 0 * (1 - q^n) / (1 - q))
  else
    a 0 * n

-- Statement to prove
theorem sum_first_six_terms : sum_geom_seq a 6 = 126 :=
by {
  sorry -- The proof would go here
}

end sum_first_six_terms_l271_271448


namespace simplify_expr1_simplify_expr2_simplify_expr3_l271_271301

-- For the first expression
theorem simplify_expr1 (a b : ℝ) : 2 * a - 3 * b + a - 5 * b = 3 * a - 8 * b :=
by
  sorry

-- For the second expression
theorem simplify_expr2 (a : ℝ) : (a^2 - 6 * a) - 3 * (a^2 - 2 * a + 1) + 3 = -2 * a^2 :=
by
  sorry

-- For the third expression
theorem simplify_expr3 (x y : ℝ) : 4*(x^2*y - 2*x*y^2) - 3*(-x*y^2 + 2*x^2*y) = -2*x^2*y - 5*x*y^2 :=
by
  sorry

end simplify_expr1_simplify_expr2_simplify_expr3_l271_271301


namespace expected_value_of_ratio_l271_271219

noncomputable def expected_value_ratio (N : ℕ) : ℚ :=
  if h : N ≥ 2 then
    let pairs := (List.range N).comb 2
    let ratios := pairs.map (λ l, if l.head! < l.get_last (by simp) then l.head! / l.get_last (by simp) else l.get_last (by simp) / l.head!)
    let sum_above := (ratios.map (λ r, (2 : ℚ) / (N * (N - 1) / 2) * r)).sum
    (sum_above / ratios.length : ℚ)
  else 
    0

theorem expected_value_of_ratio (N : ℕ) (h : N ≥ 2) :
  expected_value_ratio N = 1 / 2 := 
sorry

end expected_value_of_ratio_l271_271219


namespace largest_number_formed_l271_271805

-- Define the set of given numbers
def given_numbers : Set ℕ := {1, 7, 0}

-- Define what it means to form a number by using each given number exactly once
def form_number (nums : List ℕ) : ℕ :=
  nums.foldl (λ acc n => acc * 10 + n) 0

-- The main theorem to prove
theorem largest_number_formed :
  ∃ nums : List ℕ, (∀ n ∈ nums, n ∈ given_numbers) ∧ (List.length nums = given_numbers.toList.length) ∧ max (form_number [1,7,0]) (max (form_number [1,0,7]) (max (form_number [7,1,0]) (max (form_number [7,0,1]) (max (form_number [0,1,7]) (form_number [0,7,1]))))) = 710 :=
begin
  sorry
end

end largest_number_formed_l271_271805


namespace c_range_given_inequality_l271_271780

theorem c_range_given_inequality (a b c : ℝ) 
  (h1 : ∀ x : ℝ, sqrt(2 * x^2 + a * x + b) > x - c ↔ x ≤ 0 ∨ x > 1) : 
  c ∈ set.Ioo 0 1 :=
sorry

end c_range_given_inequality_l271_271780


namespace max_ratio_l271_271373

variables {P Q O : ℝ × ℝ}
variables (x y t a : ℝ)

def on_curve_C1 (P : ℝ × ℝ) : Prop := (P.2)^2 = 8 * P.1
def on_curve_C (Q : ℝ × ℝ) : Prop := (Q.1 - 2)^2 + (Q.2)^2 = 1
def origin (O : ℝ × ℝ) : Prop := O = (0, 0)

noncomputable def PO (P O : ℝ × ℝ) : ℝ := 
  real.sqrt ((P.1 - O.1)^2 + (P.2 - O.2)^2)

noncomputable def PQ (P Q : ℝ × ℝ) : ℝ := 
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

noncomputable def ratio (P Q O : ℝ × ℝ) : ℝ := 
  PO P O / PQ P Q

theorem max_ratio
  (hP : on_curve_C1 P)
  (hQ : on_curve_C Q)
  (hO : origin O) :
  ∃ a : ℝ, (P.1 + 1 = a⁻¹) → ratio P Q O = ∀ a, √(1 + 6a - 7a^2) = 4 * √7 / 7 := 
sorry

end max_ratio_l271_271373


namespace area_of_sector_l271_271422

-- Definition for central angle
def central_angle : Real := 120 * (Real.pi / 180)

-- Definition for radius
def radius : Real := 10

-- Definition for area of sector
def sector_area (central_angle radius : Real) : Real :=
  (1 / 2) * (central_angle * radius) * radius

-- The theorem to prove
theorem area_of_sector : sector_area central_angle radius = 100 * Real.pi / 3 :=
by
  sorry

end area_of_sector_l271_271422


namespace distance_N_to_AK_distance_MN_to_AK_distance_A1_to_plane_MNK_l271_271083

def Point := ℝ × ℝ × ℝ
def Line := Point × Point
def Plane := Point × Point × Point

-- Conditions
variables (A B C D A1 B1 C1 D1 M N K : Point)
variable edge_length : ℝ
variable midpoint_M : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2, (A.3 + B.3) / 2)
variable midpoint_N : N = ((B1.1 + C1.1) / 2, (B1.2 + C1.2) / 2, (B1.3 + C1.3) / 2)
variable position_K : 2 * K.1 = D.1 + C.1 ∧ K.2 = D.2 ∧ K.3 = D.3

-- Question a)
def distance_point_to_line (P : Point) (L : Line) : ℝ := sorry

theorem distance_N_to_AK : distance_point_to_line N (A, K) = 6 * Real.sqrt (17 / 13) :=
sorry

-- Question b)
def distance_between_lines (L1 L2 : Line) : ℝ := sorry

theorem distance_MN_to_AK : distance_between_lines (M, N) (A, K) = 18 / Real.sqrt 53 :=
sorry

-- Question c)
def distance_point_to_plane (P : Point) (π : Plane) : ℝ := sorry

theorem distance_A1_to_plane_MNK : distance_point_to_plane A1 (M, N, K) = 66 / Real.sqrt 173 :=
sorry

end distance_N_to_AK_distance_MN_to_AK_distance_A1_to_plane_MNK_l271_271083


namespace hyperbolic_cosine_identity_hyperbolic_tangent_identity_l271_271497

-- Part (a)
theorem hyperbolic_cosine_identity (x α: ℝ) 
  (h: cosh x * cosh x - sinh x * sinh x = 1) : 
  cosh x = 1 / cos α :=
sorry

-- Part (b)
theorem hyperbolic_tangent_identity (x α: ℝ) 
  (h1: exp x = sinh x + cosh x)
  (h2: sinh x = tan α)
  (h3: cosh x = 1 / cos α) : 
  tanh (x / 2) = tan (α / 2) :=
sorry

end hyperbolic_cosine_identity_hyperbolic_tangent_identity_l271_271497


namespace range_of_a_l271_271378

open Real

-- Definitions of vertices A, B, C as functions of a
def A (a : ℝ) : ℝ × ℝ := (a, a + 1)
def B (a : ℝ) : ℝ × ℝ := (a - 1, 2 * a)
def C: ℝ × ℝ := (1, 3)

-- The condition that all points inside and on the boundary of triangle ABC are within the region defined by the inequality 3x + y ≥ 2
def satisfies_inequality (a : ℝ) : Prop :=
  let (xA, yA) := A a in
  let (xB, yB) := B a in
  3 * xA + yA ≥ 2 ∧ 3 * xB + yB ≥ 2

-- The theorem to prove
theorem range_of_a (a : ℝ) : satisfies_inequality a → a ≥ 1 :=
by sorry

end range_of_a_l271_271378


namespace l_shape_area_l271_271296

theorem l_shape_area (P : ℝ) (L : ℝ) (x : ℝ)
  (hP : P = 52) 
  (hL : L = 16) 
  (h_x : L + (L - x) + 2 * (16 - x) = P)
  (h_split : 2 * (16 - x) * x = 120) :
  2 * ((16 - x) * x) = 120 :=
by
  -- This is the proof problem statement
  sorry

end l_shape_area_l271_271296


namespace teacher_wang_arrives_in_0_7_hours_l271_271513

-- Define the conditions as Lean definitions
def bicycle_speed : ℝ := 15 -- km/h
def bicycle_time : ℝ := 0.2 -- hours
def walking_speed : ℝ := 5 -- km/h
def walking_time_given : ℝ := 0.7 -- hours

-- Define the problem statement
theorem teacher_wang_arrives_in_0_7_hours :
  let distance := bicycle_speed * bicycle_time in
  (distance / walking_speed) < walking_time_given := by
  sorry

end teacher_wang_arrives_in_0_7_hours_l271_271513


namespace sticks_form_equilateral_triangle_l271_271702

theorem sticks_form_equilateral_triangle {n : ℕ} (h1 : n ≥ 5) (h2 : n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5) :
  (∃ a b c, a = b ∧ b = c ∧ a + b + c = (n * (n + 1)) / 2) :=
by
  sorry

end sticks_form_equilateral_triangle_l271_271702


namespace max_knights_with_courtiers_l271_271458

open Nat

def is_solution (a b : ℕ) : Prop :=
  12 ≤ a ∧ a ≤ 18 ∧
  10 ≤ b ∧ b ≤ 20 ∧
  1 / (a:ℝ).toNat + 1 / (b:ℝ).toNat = 1 / 7

theorem max_knights_with_courtiers : ∃ a b : ℕ, is_solution a b ∧ b = 14 ∧ a = 14 :=
by
  sorry

end max_knights_with_courtiers_l271_271458


namespace part1_solution_set_part2_values_of_a_l271_271032

-- Parameters and definitions for the function and conditions
def f (x a : ℝ) := |x - a^2| + |x - (2*a + 1)|

-- Part (1) When a = 2, the inequality's solution set
theorem part1_solution_set (x : ℝ) : f x 2 ≥ 4 ↔ (x ≤ 3/2 ∨ x ≥ 11/2) := 
sorry

-- Part (2) Range of values for a given f(x, a) ≥ 4
theorem part2_values_of_a (a : ℝ) : 
∀ x, f x a ≥ 4 → (a ≤ -1 ∨ a ≥ 3) :=
sorry

end part1_solution_set_part2_values_of_a_l271_271032


namespace pheasants_and_rabbits_l271_271549

theorem pheasants_and_rabbits (x y : Nat) 
    (heads_condition : x + y = 35)
    (legs_condition : 2 * x + 4 * y = 94) :
    x = 23 ∧ y = 12 :=
begin
    sorry
end

end pheasants_and_rabbits_l271_271549


namespace sum_of_vertices_l271_271243

theorem sum_of_vertices (vertices_rectangle : ℕ) (vertices_pentagon : ℕ) 
  (h_rect : vertices_rectangle = 4) (h_pent : vertices_pentagon = 5) : 
  vertices_rectangle + vertices_pentagon = 9 :=
by
  sorry

end sum_of_vertices_l271_271243


namespace group_partition_disjoint_l271_271163

theorem group_partition_disjoint (G : Type*) [DecidableEq G] [Fintype G] (acquaintance : G → G → Bool) 
  (h_reciprocal : ∀ {x y : G}, acquaintance x y = acquaintance y x) : 
  ∃ (A B : Finset G), A ∩ B = ∅ ∧ ∀ v ∈ A, (∑ w in Finset.filter (λ x, acquaintance v x) B).card ≥ (Finset.filter (λ x, acquaintance v x) A).card / 2 ∧
                            ∀ v ∈ B, (∑ w in Finset.filter (λ x, acquaintance v x) A).card ≥ (Finset.filter (λ x, acquaintance v x) B).card / 2 :=
sorry

end group_partition_disjoint_l271_271163


namespace sticks_form_equilateral_triangle_l271_271712

theorem sticks_form_equilateral_triangle (n : ℕ) :
  (∃ (s : set ℕ), s = {1, 2, ..., n} ∧ ∃ t : ℕ → ℕ, t ∈ s ∧ t s ∩ { a, b, c} = ∅ 
   and t t  = t and a + b + {
    sum s = nat . multiplicity = S
}) <=> n

example: ℕ in {
  n ≥ 5 ∧ n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5 } equilateral triangle (sorry)
   sorry


Sorry  = fiancé in the propre case, {1,2, ..., n}

==> n.g.e. : natal sets {1, 2, ... n }  + elastical


end sticks_form_equilateral_triangle_l271_271712


namespace joey_next_age_sum_digits_l271_271465

-- Definitions and conditions based on step a)
def joey_age_today : ℕ := 36
def zoe_age_today : ℕ := 1
def next_age_multiple (n : ℕ) : ℕ := joey_age_today * (n + 1)

-- The main theorem to prove the question
theorem joey_next_age_sum_digits : 
  let next_joey_age := next_age_multiple 1 in
  next_joey_age.digits.sum = 9 :=
by
  -- Given definitions
  have joey_is_36 : joey_age_today = 36 := rfl
  have zoe_is_1 : zoe_age_today = 1 := rfl

  -- Lemma and computation(sorry for the proof)
  sorry

end joey_next_age_sum_digits_l271_271465


namespace num_women_in_luxury_suite_l271_271685

theorem num_women_in_luxury_suite (total_passengers : ℕ) (pct_women : ℕ) (pct_women_luxury : ℕ)
  (h_total_passengers : total_passengers = 300)
  (h_pct_women : pct_women = 50)
  (h_pct_women_luxury : pct_women_luxury = 15) :
  (total_passengers * pct_women / 100) * pct_women_luxury / 100 = 23 := 
by
  sorry

end num_women_in_luxury_suite_l271_271685


namespace parking_ways_l271_271988

theorem parking_ways (n m : ℕ) (h_n : n = 7) (h_m : m = 3) :
  let empty_spaces := n - m
  in empty_spaces = 4 → number_of_parking_methods n m = 24 :=
begin
  intros,
  sorry
end

end parking_ways_l271_271988


namespace inclination_angle_at_neg1_l271_271211

noncomputable def curve (x : ℝ) : ℝ := (1 / 3) * x^3 - 2

def derivative (x : ℝ) : ℝ := x^2

theorem inclination_angle_at_neg1 :
  let slope := derivative (-1)
  let inclination_angle := real.atan slope * (180 / real.pi)
  inclination_angle = 45 :=
by
  sorry

end inclination_angle_at_neg1_l271_271211


namespace part1_solution_set_part2_values_of_a_l271_271036

-- Parameters and definitions for the function and conditions
def f (x a : ℝ) := |x - a^2| + |x - (2*a + 1)|

-- Part (1) When a = 2, the inequality's solution set
theorem part1_solution_set (x : ℝ) : f x 2 ≥ 4 ↔ (x ≤ 3/2 ∨ x ≥ 11/2) := 
sorry

-- Part (2) Range of values for a given f(x, a) ≥ 4
theorem part2_values_of_a (a : ℝ) : 
∀ x, f x a ≥ 4 → (a ≤ -1 ∨ a ≥ 3) :=
sorry

end part1_solution_set_part2_values_of_a_l271_271036


namespace corrected_avg_and_variance_l271_271848

-- Definitions from the conditions
def students : ℕ := 50
def initial_avg_score : ℝ := 70
def initial_variance : ℝ := 102
def incorrect_scores : list ℝ := [50, 90]
def correct_scores : list ℝ := [80, 60]

-- Proof statement
theorem corrected_avg_and_variance :
  let corrected_avg := initial_avg_score,
      corrected_variance := initial_variance - (800 - 200) / students
  in corrected_avg = 70 ∧ corrected_variance = 90 :=
by
  sorry

end corrected_avg_and_variance_l271_271848


namespace smallest_positive_period_of_f_intervals_where_f_is_monotonically_increasing_l271_271049

def vector_dot (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

noncomputable def f (x : ℝ) : ℝ :=
  vector_dot (2 * Real.cos x, Real.sin x) (Real.sqrt 3 * Real.cos x, 2 * Real.cos x) - Real.sqrt 3

theorem smallest_positive_period_of_f :
  ∃ T > 0, (∀ x : ℝ, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x : ℝ, f (x + T') = f x) → T ≤ T')
:= sorry

theorem intervals_where_f_is_monotonically_increasing :
  ∀ k : ℤ, ∀ x : ℝ, - (5 * Real.pi / 12) + (↑k * Real.pi) ≤ x ∧ x ≤ (↑k * Real.pi) + (Real.pi / 12) → 
            f' (x) > 0
:= sorry

end smallest_positive_period_of_f_intervals_where_f_is_monotonically_increasing_l271_271049


namespace total_round_robin_matches_l271_271859

theorem total_round_robin_matches (m : ℕ) (h : m = 12) : (m * (m - 1)) / 2 = 66 :=
by {
  rw h,
  norm_num,
  sorry
}

end total_round_robin_matches_l271_271859


namespace prove_concurrence_of_lines_l271_271656

noncomputable def trapezoid_and_tangents_concurrence
  (A B C D P Q : Point) (O1 O2 : Circle)
  (tangent1 : Tangent O1 A B P) (tangent2 : Tangent O1 D A P)
  (tangent3 : Tangent O1 B C P) (tangent4 : Tangent O2 D A Q)
  (tangent5 : Tangent O2 C D Q) (tangent6 : Tangent O2 B C Q)
  (h_parallel : is_parallel AB CD) : Prop :=
  are_concurrent (Line_through A C) (Line_through B D) (Line_through P Q)

theorem prove_concurrence_of_lines
  (A B C D P Q : Point) (O1 O2 : Circle)
  (tangent1 : Tangent O1 A B P) (tangent2 : Tangent O1 D A P)
  (tangent3 : Tangent O1 B C P) (tangent4 : Tangent O2 D A Q)
  (tangent5 : Tangent O2 C D Q) (tangent6 : Tangent O2 B C Q)
  (h_parallel : is_parallel AB CD) :
  trapezoid_and_tangents_concurrence A B C D P Q O1 O2
    tangent1 tangent2 tangent3 tangent4 tangent5 tangent6 h_parallel :=
  sorry

end prove_concurrence_of_lines_l271_271656


namespace floor_function_example_l271_271327

theorem floor_function_example : 
  let a := 15.3 in 
  (Int.floor (a * a)) - (Int.floor a) * (Int.floor a) = 9 :=
by 
  -- Introducing the "let" bindings in Lean scope
  let a := 15.3
  sorry

end floor_function_example_l271_271327


namespace trapezoid_upper_base_BC_l271_271769

theorem trapezoid_upper_base_BC (A B C D M : Point) (d : ℝ)
  (h1 : Trapezoid A B C D)
  (h2 : OnLine M A B)
  (h3 : Perpendicular D M A B)
  (h4 : Distance M C = Distance C D)
  (h5 : Distance A D = d) : Distance B C = d / 2 := 
sorry

end trapezoid_upper_base_BC_l271_271769


namespace cos_plus_sin_l271_271776

open Real

theorem cos_plus_sin (α : ℝ) (h₁ : tan α = -2) (h₂ : (π / 2) < α ∧ α < π) : 
  cos α + sin α = (sqrt 5) / 5 :=
sorry

end cos_plus_sin_l271_271776


namespace part1_part2_l271_271009

def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1 (x : ℝ) : 
  ∀ x, f x 2 ≥ 4 ↔ (x ≤ 3 / 2 ∨ x ≥ 11 / 2) :=
by sorry

theorem part2 (x a : ℝ) :
  f x a ≥ 4 → (a ≤ -1 ∨ a ≥ 3) :=
by sorry

end part1_part2_l271_271009


namespace no_real_solution_l271_271489

theorem no_real_solution (a b : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : 1 / a + 1 / b = 1 / (a + b)) : False :=
by
  sorry

end no_real_solution_l271_271489


namespace point_further_l271_271281

noncomputable def cheese_location : ℝ × ℝ := (15, 12)
noncomputable def mouse_start : ℝ × ℝ := (3, -3)
noncomputable def mouse_line (x : ℝ) : ℝ := -6 * x + 15

def perpendicular_line (x : ℝ) : ℝ := (1 / 6) * x + 11

def intersection_point_x : ℝ := 24 / 37
def intersection_point_y : ℝ := 123 / 37

theorem point_further (a b : ℝ) :
  (a, b) = (intersection_point_x, intersection_point_y) ∧ 
  (∃ d : ℝ, d = sqrt ((a - fst mouse_start) ^ 2 + (b - snd mouse_start) ^ 2) ∧ 
  d = 250 / 37) :=
by
  sorry

end point_further_l271_271281


namespace find_complex_number_z_l271_271785

noncomputable def complex_number_z (z : ℂ) : Prop :=
  (z + 1) * complex.I = 1 - complex.I

theorem find_complex_number_z (z : ℂ) (h : complex_number_z z) : z = -2 - complex.I :=
by
  sorry

end find_complex_number_z_l271_271785


namespace correct_propositions_count_l271_271400

theorem correct_propositions_count
  (m n : Line)
  (α β : Plane)
  (h1 : Intersect m n)
  (h2 : Outside m α)
  (h3 : Outside m β)
  (h4 : Parallel m α)
  (h5 : Parallel m β)
  (h6 : Parallel n α)
  (h7 : Parallel n β) :
  number_of_correct_propositions = 1 :=
by
  -- Here, we assume m and n are types for lines, and α and β are types for planes.
  -- Intersect, Outside, and Parallel are relational predicates defined appropriately for lines and planes.
  sorry

end correct_propositions_count_l271_271400


namespace percentage_change_in_revenue_l271_271253

theorem percentage_change_in_revenue (P V : ℝ) : 
  let R := P * V in
  let P_new := 1.5 * P in
  let V_new := 0.8 * V in
  let R_new := P_new * V_new in
  ((R_new - R) / R) * 100 = 20 :=
by
  sorry

end percentage_change_in_revenue_l271_271253


namespace log_sum_geometric_sequence_l271_271739

variable {q : ℝ} (a : ℕ → ℝ)

noncomputable def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = a n * q

theorem log_sum_geometric_sequence :
  is_geometric_sequence a q →
  a 6 = 2 →
  0 < q →
  (∑ i in Finset.range 1.succ (11), Real.log (a i) / Real.log 2) = 11
  :=
by
  intro h_seq h_a6 h_q_pos
  sorry

end log_sum_geometric_sequence_l271_271739


namespace number_of_students_l271_271985

variable (F S J R T : ℕ)

axiom freshman_more_than_junior : F = (5 * J) / 4
axiom sophomore_fewer_than_freshman : S = 9 * F / 10
axiom total_students : T = F + S + J + R
axiom seniors_total : R = T / 5
axiom given_sophomores : S = 144

theorem number_of_students (T : ℕ) : T = 540 :=
by 
  sorry

end number_of_students_l271_271985


namespace triangle_is_right_angle_perimeter_range_l271_271096

noncomputable def vec_m (A C : ℝ) : ℝ × ℝ := (Real.sin A, Real.cos C)
noncomputable def vec_n (B A : ℝ) : ℝ × ℝ := (Real.cos B, Real.sin A)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Conditions in the problem
variables (A B C : ℝ) (sinA sinB sinC cosB cosC : ℝ)
(h1 : ∀ A B C, vec_m A C ▸ vec_n B A = Real.sin B + Real.sin C)
(h2 : ∀ A B C, dot_product (vec_m A C) (vec_n B A) = Real.sin B + Real.sin C)

-- Questions in the problem
-- 1. Prove that ∆ABC is a right triangle
theorem triangle_is_right_angle : ∃ (a b c : ℝ), a^2 = b^2 + c^2 :=
sorry

-- 2. If the radius of the circumcircle of ∆ABC is 1, finding the range of its perimeter
theorem perimeter_range (a b c : ℝ) (h_a : a = 2) : 4 < a + b + c ∧ a + b + c ≤ 4 :=
sorry

end triangle_is_right_angle_perimeter_range_l271_271096


namespace sin_cos_addition_l271_271819

theorem sin_cos_addition
  (α : ℝ)
  (h : (cos (2 * α)) / (sin (α + 7 * π / 4)) = - (sqrt 2) / 2) :
  sin α + cos α = 1 / 2 := by
  sorry

end sin_cos_addition_l271_271819


namespace simplify_product_of_fractions_l271_271958

theorem simplify_product_of_fractions :
  8 * (15 / 4) * (-28 / 45) = -56 / 3 := by
  sorry

end simplify_product_of_fractions_l271_271958


namespace x_is_36_percent_of_z_l271_271429

variable (x y z : ℝ)

theorem x_is_36_percent_of_z (h1 : x = 1.20 * y) (h2 : y = 0.30 * z) : x = 0.36 * z :=
by
  sorry

end x_is_36_percent_of_z_l271_271429


namespace series_sum_to_one_l271_271130

theorem series_sum_to_one (a A : ℕ) (h : A ≥ a) :
  (∑ i in finset.range (A - a + 1), 
    (∏ j in list.range (i+1), (ite (j = 0) (a / (A - j)) ((A - a - j + 1) / (A - j)))) 
  ) = 1 :=
by
  sorry

end series_sum_to_one_l271_271130


namespace num_digits_in_base_ten_l271_271820

theorem num_digits_in_base_ten (x : ℝ) (h : log 2 (log 2 (log 2 x)) = 3) : 
  let digits := Nat.ceil (256 * log 10 2) in digits = 77 :=
begin
  sorry
end

end num_digits_in_base_ten_l271_271820


namespace milk_consumption_l271_271319

theorem milk_consumption (milk_0 consumption_ratio_rachel consumption_ratio_monica : ℚ) 
  (initial_milk_eq : milk_0 = 3 / 4)
  (rachel_ratio_eq : consumption_ratio_rachel = 1 / 2)
  (monica_ratio_eq : consumption_ratio_monica = 1 / 3):
  (rachel_ratio_eq * milk_0 + (monica_ratio_eq * (milk_0 - (rachel_ratio_eq * milk_0)))) = 1 / 2 := by
  sorry

end milk_consumption_l271_271319


namespace find_integer_n_l271_271233

theorem find_integer_n (n : ℕ) (hn1 : 0 ≤ n) (hn2 : n < 102) (hmod : 99 * n % 102 = 73) : n = 97 :=
  sorry

end find_integer_n_l271_271233


namespace eccentricity_of_ellipse_l271_271380

theorem eccentricity_of_ellipse :
  ∃ (e : ℝ), (∀ (a b : ℝ), (a^2 = 25) → (b^2 = 16) → e = real.sqrt (1 - (b^2 / a^2))) ∧ e = 3/5 :=
sorry

end eccentricity_of_ellipse_l271_271380


namespace find_c_l271_271834

def line1 (x y : ℝ) : Prop := 2 * y + x + 3 = 0
def line2 (x y : ℝ) (c : ℝ) : Prop := 3 * y + c * x + 2 = 0
def slope (m : ℝ) (x y b : ℝ) : Prop := y = m * x + b

theorem find_c (c : ℝ) : (∀ x y : ℝ, line1 x y → ∃ m b : ℝ, slope m x y b ∧ m = -1/2) →
                        (∀ x y : ℝ, line2 x y c → ∃ m' b' : ℝ, slope m' x y b' ∧ m' = -c/3) →
                        (∀ m m' : ℝ, m * m' = -1) → 
                        c = -6 :=
by sorry

end find_c_l271_271834


namespace unique_solution_of_quadratic_l271_271179

theorem unique_solution_of_quadratic :
  ∀ (b : ℝ), b ≠ 0 → (∃ x : ℝ, 3 * x^2 + b * x + 12 = 0 ∧ ∀ y : ℝ, 3 * y^2 + b * y + 12 = 0 → y = x) → 
  (b = 12 ∧ ∃ x : ℝ, x = -2 ∧ 3 * x^2 + 12 * x + 12 = 0 ∧ (∀ y : ℝ, 3 * y^2 + 12 * y + 12 = 0 → y = x)) ∨ 
  (b = -12 ∧ ∃ x : ℝ, x = 2 ∧ 3 * x^2 - 12 * x + 12 = 0 ∧ (∀ y : ℝ, 3 * y^2 - 12 * y + 12 = 0 → y = x)) :=
by 
  sorry

end unique_solution_of_quadratic_l271_271179


namespace increasing_function_probability_l271_271729

noncomputable def a_vals := {-2, 0, 1, 3, 4}
noncomputable def b_vals := {1, 2}

def is_increasing (a : ℤ) : Prop := (a^2 - 2) > 0

def increasing_probability : ℚ :=
  let suitable_a := {a ∈ a_vals | is_increasing a}
  let total_a := a_vals.card
  let suitable_a_count := suitable_a.card
  suitable_a_count / total_a

theorem increasing_function_probability :
  increasing_probability = 3 / 5 :=
by 
  sorry

end increasing_function_probability_l271_271729


namespace TangencyCondition_PerimeterHalfCondition_LengthEquality_SumPerimetersTriangle_PerimeterTriangleEqLength_IncircleRadiusCondition_l271_271447

/-- Definitions for the geometric setup -/
section GeometrySetup
variables (A B C D C' D' E F G : Point)
variables (ABCD : square A B C D)

/-- Given conditions for the problem -/
variables (Fold : FoldAcross EF C' D')
variables (AC_BC_radius : radius_circle C B)
variables (PerimHalf : PerimeterTriangle G A C' = HalfPerimeterSquare ABCD)
variables (Tangency : TangentLineCircle C' D' (center_circle C (radius_circle B C)))
variables (PerimEq : SumPerimeters (PerimeterTriangle C' B E) (PerimeterTriangle G D' F) = PerimeterTriangle G A C')
variables (PerimTriEq : PerimeterTriangle G D' F = LengthSegment A C')
variables (IncircleRadiusEq : RadiusIncircle G A C' = LengthSegment G D')

/-- Prove the first part -/
theorem TangencyCondition : TangentLineCircle C' D' (center_circle C (radius_circle B C)) := sorry

/-- Prove the second part -/
theorem PerimeterHalfCondition : PerimeterTriangle G A C' = HalfPerimeterSquare ABCD := sorry

/-- Prove the third part -/
theorem LengthEquality : Distance A G = Distance C' B + Distance G D' := sorry

/-- Prove the fourth part -/
theorem SumPerimetersTriangle : SumPerimeters (PerimeterTriangle C' B E) (PerimeterTriangle G D' F) = PerimeterTriangle G A C' := sorry

/-- Prove the fifth part -/
theorem PerimeterTriangleEqLength : PerimeterTriangle G D' F = LengthSegment A C' := sorry

/-- Prove the sixth part -/
theorem IncircleRadiusCondition : RadiusIncircle (Triangle G A C') = LengthSegment G D' := sorry

end GeometrySetup

end TangencyCondition_PerimeterHalfCondition_LengthEquality_SumPerimetersTriangle_PerimeterTriangleEqLength_IncircleRadiusCondition_l271_271447


namespace probability_eq_fraction_l271_271161

noncomputable def probability_inside_rectangle (a b : ℝ) : ℚ :=
let Area_Triangle := (1 / 2) * 2019 * (2019 / 9),
    Area_Rectangle := 2019 * 2021 in
(Area_Triangle / Area_Rectangle : ℚ)

theorem probability_eq_fraction :
  probability_inside_rectangle 2019 2021 = 22433 / 404200 :=
sorry

end probability_eq_fraction_l271_271161


namespace find_rate_from_simple_interest_l271_271570

-- Definitions from the conditions
def simple_interest (P R T : ℝ) : ℝ := P * R * T / 100

-- Main statement/problem to prove with the given conditions
theorem find_rate_from_simple_interest :
  (simple_interest 900 R 4 = 160) → R ≈ 4.44 :=
by
  -- We skip the proof as per instructions
  sorry

end find_rate_from_simple_interest_l271_271570


namespace solve_arccos_cos_within_interval_l271_271504

noncomputable def solve_arccos_cos_eq (x : ℝ) : Prop :=
  arccos (cos x) = x / 3
  
theorem solve_arccos_cos_within_interval :
  {x : ℝ | solve_arccos_cos_eq x} = {0, 3 * (π / 2), -3 * (π / 2)} :=
begin
  sorry
end

end solve_arccos_cos_within_interval_l271_271504


namespace good_numbers_1_to_17_not_good_number_18_l271_271133

def d (n : ℕ) := {k : ℕ | k > 0 ∧ n % k = 0}.card

def is_good_number (m : ℕ) : Prop :=
  ∃ n : ℕ, m = n / d(n)

theorem good_numbers_1_to_17 : 
  ∀ m : ℕ, 1 ≤ m ∧ m ≤ 17 → is_good_number m :=
by sorry

theorem not_good_number_18 : ¬ is_good_number 18 :=
by sorry

end good_numbers_1_to_17_not_good_number_18_l271_271133


namespace probability_exactly_three_cured_l271_271289

 noncomputable def combination (n k : ℕ) : ℕ :=
  Nat.fact n / (Nat.fact k * Nat.fact (n - k))

theorem probability_exactly_three_cured (p : ℝ) (n k : ℕ) (p_cure : p = 0.9) (n_pigs : n = 5) (k_cured : k = 3) :
  (combination n k) * p^k * (1 - p)^(n - k) = (combination 5 3) * 0.9^3 * 0.1^2 := 
begin
  -- proof steps go here
  sorry
end

end probability_exactly_three_cured_l271_271289


namespace game_goal_impossible_l271_271563

-- Definition for initial setup
def initial_tokens : ℕ := 2013
def initial_piles : ℕ := 1

-- Definition for the invariant
def invariant (tokens piles : ℕ) : ℕ := tokens + piles

-- Initial value of the invariant constant
def initial_invariant : ℕ :=
  invariant initial_tokens initial_piles

-- Goal is to check if the final configuration is possible
theorem game_goal_impossible (n : ℕ) :
  (invariant (3 * n) n = initial_invariant) → false :=
by
  -- The invariant states 4n = initial_invariant which is 2014.
  -- Thus, we need to check if 2014 / 4 results in an integer.
  have invariant_expr : 4 * n = 2014 := by sorry
  have n_is_integer : 2014 % 4 = 0 := by sorry
  sorry

end game_goal_impossible_l271_271563


namespace part1_part2_l271_271014

def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1 (x : ℝ) : 
  ∀ x, f x 2 ≥ 4 ↔ (x ≤ 3 / 2 ∨ x ≥ 11 / 2) :=
by sorry

theorem part2 (x a : ℝ) :
  f x a ≥ 4 → (a ≤ -1 ∨ a ≥ 3) :=
by sorry

end part1_part2_l271_271014


namespace solve_inequality_system_l271_271962

theorem solve_inequality_system (x : ℝ) :
  (x - 1 < 2 * x + 1) ∧ ((2 * x - 5) / 3 ≤ 1) → (-2 < x ∧ x ≤ 4) :=
by
  intro cond
  sorry

end solve_inequality_system_l271_271962


namespace upper_base_length_l271_271762

-- Definitions based on the given conditions
variables (A B C D M : ℝ)
variables (d : ℝ)
variables (h : ℝ) -- height from D to AB

-- Conditions given in the problem
def is_trapezoid (ABCD : ℝ) : Prop := true
def is_perpendicular (DM AB : ℝ) : Prop := true
def point_on_side (M AB : ℝ) : Prop := true
def MC_eq_CD (MC CD : ℝ) : Prop := MC = CD
def AD_length (A D d : ℝ) : Prop := A - D = d -- Assuming some coordinate system 

-- Define the proof statement with conditions and the result
theorem upper_base_length
  (ht : is_trapezoid ABCD)
  (hp : is_perpendicular D M)
  (ps : point_on_side M AB)
  (mc_cd : MC_eq_CD M C D)
  (ad_len : AD_length A D d) :
  BC = d / 2 :=
sorry

end upper_base_length_l271_271762


namespace calculate_fraction_l271_271345

-- Define the fractions we are working with
def fraction1 : ℚ := 3 / 4
def fraction2 : ℚ := 15 / 5
def one_half : ℚ := 1 / 2

-- Define the main calculation
def main_fraction (f1 f2 one_half : ℚ) : ℚ := f1 * f2 - one_half

-- State the theorem
theorem calculate_fraction : main_fraction fraction1 fraction2 one_half = (7 / 4) := by
  sorry

end calculate_fraction_l271_271345


namespace max_knights_and_courtiers_l271_271452

theorem max_knights_and_courtiers (a b : ℕ) (ha : 12 ≤ a ∧ a ≤ 18) (hb : 10 ≤ b ∧ b ≤ 20) :
  (1 / a : ℚ) + (1 / b) = (1 / 7) → a = 14 ∧ b = 14 :=
begin
  -- Proof would go here.
  sorry
end

end max_knights_and_courtiers_l271_271452


namespace smallest_number_of_marbles_with_17_proper_factors_l271_271464

theorem smallest_number_of_marbles_with_17_proper_factors : ∃ n : ℕ, (nat.factors n).length = 19 ∧ n = 4608 :=
by
  -- Proof steps would go here, omitted per instructions
  sorry

end smallest_number_of_marbles_with_17_proper_factors_l271_271464


namespace vertical_distance_compute_l271_271658

theorem vertical_distance_compute :
  let thickness := 1
  let top_ring_diameter := 36
  let smallest_ring_diameter := 4
  let decrease_per_ring := 2
  let number_of_rings := ((top_ring_diameter - smallest_ring_diameter) / decrease_per_ring) + 1 in
  let inside_diameter_start := top_ring_diameter - thickness in
  let inside_diameter_end := smallest_ring_diameter - thickness in
  let sum_of_inside_diameters := (number_of_rings * (inside_diameter_start + inside_diameter_end)) / 2 in
  let total_distance := sum_of_inside_diameters + (2 * thickness) in
  total_distance = 325 :=
by
  sorry

end vertical_distance_compute_l271_271658


namespace angle_C_formed_by_m_and_n_l271_271551

-- Angle types
variable {ℝ} [NonZeroDivisors ℝ] (l m n : Line ℝ) 

-- Conditions
axiom intersect_at_point : Intersect l m n
axiom l_parallel_m : Parallel l m
axiom angle_A : Angle (l, l.direction) = 130
axiom angle_B : Angle (m, m.direction) = 140
axiom angle_D : Angle (l, n.direction) = 50

-- Conclusion to prove
theorem angle_C_formed_by_m_and_n
    (l m n : Line ℝ)
    (intersect_at_point : Intersect l m n)
    (l_parallel_m : Parallel l m)
    (angle_A : Angle (l, l.direction) = 130)
    (angle_B : Angle (m, m.direction) = 140)
    (angle_D : Angle (l, n.direction) = 50) :
    Angle (m, n.direction) = 40 :=
sorry

end angle_C_formed_by_m_and_n_l271_271551


namespace part1_part2_l271_271010

def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1 (x : ℝ) : 
  ∀ x, f x 2 ≥ 4 ↔ (x ≤ 3 / 2 ∨ x ≥ 11 / 2) :=
by sorry

theorem part2 (x a : ℝ) :
  f x a ≥ 4 → (a ≤ -1 ∨ a ≥ 3) :=
by sorry

end part1_part2_l271_271010


namespace tysons_speed_in_ocean_l271_271560

theorem tysons_speed_in_ocean
  (speed_lake : ℕ) (half_races_lake : ℕ) (total_races : ℕ) (race_distance : ℕ) (total_time : ℕ)
  (speed_lake_val : speed_lake = 3)
  (half_races_lake_val : half_races_lake = 5)
  (total_races_val : total_races = 10)
  (race_distance_val : race_distance = 3)
  (total_time_val : total_time = 11) :
  ∃ (speed_ocean : ℚ), speed_ocean = 2.5 := 
by
  sorry

end tysons_speed_in_ocean_l271_271560


namespace trapezoid_upper_base_BC_l271_271765

theorem trapezoid_upper_base_BC (A B C D M : Point) (d : ℝ)
  (h1 : Trapezoid A B C D)
  (h2 : OnLine M A B)
  (h3 : Perpendicular D M A B)
  (h4 : Distance M C = Distance C D)
  (h5 : Distance A D = d) : Distance B C = d / 2 := 
sorry

end trapezoid_upper_base_BC_l271_271765


namespace find_upper_base_length_l271_271750

variables {A B C D M : Type}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space M]
variables (AB: line_segment A B) (CD: line_segment C D)
variables (AD : ℝ) (d : ℝ)

noncomputable def upper_base_length (A B C D M : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space M]
  (ABCD : bool)
  (DM_perp_AB : line_segment D M ⊥ line_segment A B)
  (MC_eq_CD : line_segment M C = line_segment C D)
  (AD_length : line_segment A D)
  (d_value : AD_length = d)
: Prop :=
BC = d / 2

theorem find_upper_base_length :
∀ (A B C D M : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space M]
  (ABCD : bool)
  (DM_perp_AB : line_segment D M ⊥ line_segment A B)
  (MC_eq_CD : line_segment M C = line_segment C D)
  (AD_length : line_segment A D)
  (d_value : AD_length = d),
  upper_base_length A B C D M ABCD DM_perp_AB MC_eq_CD AD_length d_value := sorry

end find_upper_base_length_l271_271750


namespace prime_count_between_20_and_40_l271_271817

open Nat

def is_prime_in_range_21_to_39 (n : ℕ) : Prop :=
  n ≥ 21 ∧ n ≤ 39 ∧ Prime n

theorem prime_count_between_20_and_40 :
  {n : ℕ | is_prime_in_range_21_to_39 n}.toFinset.card = 4 := by
  sorry

end prime_count_between_20_and_40_l271_271817


namespace zeros_in_expansion_l271_271411

theorem zeros_in_expansion : 
  let x := 10^12 - 5 in
  (x^2).toString.filter (· = '0').length = 12 
  := 
by 
  let x := 10^12 - 5
  -- Compute the square of (10^12 - 5)
  have h : (x^2).toString = "9999000000000000000025" := sorry
  -- Count the number of zeros in the resultant string
  have count_zeros : ("9999000000000000000025".filter (· = '0')).length = 12 := sorry
  exact count_zeros

end zeros_in_expansion_l271_271411


namespace largest_constant_C_l271_271672

theorem largest_constant_C :
  ∃ C, C = 1012 ∧ ∀ (a : Fin 2023 → ℝ), (∀ i j, i ≠ j → 0 < a i ∧ a i ≠ a j) →
    (∑ i : Fin 2023, a i / |a (i + 1 % 2023) - a (i + 2 % 2023)|) > C :=
by
  sorry

end largest_constant_C_l271_271672


namespace leaks_drain_time_l271_271626

-- Definitions from conditions
def pump_rate : ℚ := 1 / 2 -- tanks per hour
def leak1_rate : ℚ := 1 / 6 -- tanks per hour
def leak2_rate : ℚ := 1 / 9 -- tanks per hour

-- Proof statement
theorem leaks_drain_time : (leak1_rate + leak2_rate)⁻¹ = 3.6 :=
by
  sorry

end leaks_drain_time_l271_271626


namespace integer_b_if_integer_a_l271_271536

theorem integer_b_if_integer_a (a b : ℤ) (h : 2 * a + a^2 = 2 * b + b^2) : (∃ a' : ℤ, a = a') → ∃ b' : ℤ, b = b' :=
by
-- proof will be filled in here
sorry

end integer_b_if_integer_a_l271_271536


namespace estate_area_is_correct_l271_271284

noncomputable def actual_area_of_estate (length_in_inches : ℕ) (width_in_inches : ℕ) (scale : ℕ) : ℕ :=
  let actual_length := length_in_inches * scale
  let actual_width := width_in_inches * scale
  actual_length * actual_width

theorem estate_area_is_correct :
  actual_area_of_estate 9 6 350 = 6615000 := by
  -- Here, we would provide the proof steps, but for this exercise, we use sorry.
  sorry

end estate_area_is_correct_l271_271284


namespace quadratic_min_value_l271_271449

theorem quadratic_min_value
  (a b c : ℝ)
  (h1 : y = (λ x : ℝ, a * x ^ 2 + b * x + c))
  (h2 : (∀ x, ∃ y : ℝ, (x = -7 → y = -9) ∧ (x = -5 → y = -4) ∧ (x = -3 → y = -1) ∧ (x = -1 → y = 0) ∧ (x = 1 → y = -1)))
  (h_range : -7 ≤ x ∧ x ≤ 7) :
  ∃ x : ℝ, -7 ≤ x ∧ x ≤ 7 ∧ ∀ x', -7 ≤ x' ∧ x' ≤ 7 → (a * x ^ 2 + b * x + c) ≥ (a * x' ^ 2 + b * x' + c) :=
  sorry

end quadratic_min_value_l271_271449


namespace division_yields_square_l271_271773

theorem division_yields_square (a b : ℕ) (hab : ab + 1 ∣ a^2 + b^2) :
  ∃ m : ℕ, m^2 = (a^2 + b^2) / (ab + 1) :=
sorry

end division_yields_square_l271_271773


namespace find_a_max_min_f_l271_271799

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x * Real.exp x

theorem find_a (a : ℝ) (h : (deriv (f a) 0 = 1)) : a = 1 :=
by sorry

noncomputable def f_one (x : ℝ) : ℝ := f 1 x

theorem max_min_f (h : ∀ x, 0 ≤ x → x ≤ 2 → deriv (f_one) x > 0) :
  (f_one 0 = 0) ∧ (f_one 2 = 2 * Real.exp 2) :=
by sorry

end find_a_max_min_f_l271_271799


namespace part1_solution_set_part2_range_of_a_l271_271024

def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1_solution_set (x : ℝ) : 
  (f x 2 ≥ 4) ↔ (x ≤ 3 / 2 ∨ x ≥ 11 / 2) :=
sorry

theorem part2_range_of_a (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 4) → (a ≤ -1 ∨ a ≥ 3) :=
sorry

end part1_solution_set_part2_range_of_a_l271_271024


namespace total_amount_spent_l271_271606

theorem total_amount_spent (avg_price_goat : ℕ) (num_goats : ℕ) (avg_price_cow : ℕ) (num_cows : ℕ) (total_spent : ℕ) 
  (h1 : avg_price_goat = 70) (h2 : num_goats = 10) (h3 : avg_price_cow = 400) (h4 : num_cows = 2) :
  total_spent = 1500 :=
by
  have cost_goats := avg_price_goat * num_goats
  have cost_cows := avg_price_cow * num_cows
  have total := cost_goats + cost_cows
  sorry

end total_amount_spent_l271_271606


namespace sum_of_solutions_l271_271579

theorem sum_of_solutions : 
  let solutions := {x | 0 < x ∧ x ≤ 30 ∧ 17 * (5 * x - 3) % 10 = 34 % 10}
  in (∑ x in solutions, x) = 225 := by
  sorry

end sum_of_solutions_l271_271579


namespace max_swaps_sort_descending_l271_271545

theorem max_swaps_sort_descending (n : ℕ) (arr : Array ℕ) (h1: arr.size = n) 
(h2 : ∀ i, 0 < i → i ≤ n → ∃ j, arr[j] = i) 
(h3 : ∀ i, ∃ j, 0 ≤ j → j < n → arr[j] = i → arr[j+1] ≠ i) :
  ∃ (swap_operations : (fin n) → (fin n) → Prop), 
  ∀ sequence : List (fin n × fin n), 
  (∀ (⟨i,j⟩ : fin n × fin n), (i < n - 1 ∧ j = i + 1) → swap_operations i j)  ∧
  (sequence.length ≤ (n * (n - 1)) / 2) ∧
  sorted (>.reversed) (permutation arr sequence)  :=
sorry

end max_swaps_sort_descending_l271_271545


namespace upper_base_length_l271_271760

-- Definitions based on the given conditions
variables (A B C D M : ℝ)
variables (d : ℝ)
variables (h : ℝ) -- height from D to AB

-- Conditions given in the problem
def is_trapezoid (ABCD : ℝ) : Prop := true
def is_perpendicular (DM AB : ℝ) : Prop := true
def point_on_side (M AB : ℝ) : Prop := true
def MC_eq_CD (MC CD : ℝ) : Prop := MC = CD
def AD_length (A D d : ℝ) : Prop := A - D = d -- Assuming some coordinate system 

-- Define the proof statement with conditions and the result
theorem upper_base_length
  (ht : is_trapezoid ABCD)
  (hp : is_perpendicular D M)
  (ps : point_on_side M AB)
  (mc_cd : MC_eq_CD M C D)
  (ad_len : AD_length A D d) :
  BC = d / 2 :=
sorry

end upper_base_length_l271_271760


namespace willie_exchange_rate_l271_271588

theorem willie_exchange_rate :
  let euros := 70
  let normal_exchange_rate := 1 / 5 -- euros per dollar
  let airport_exchange_rate := 5 / 7
  let dollars := euros * normal_exchange_rate * airport_exchange_rate
  dollars = 10 := by
  sorry

end willie_exchange_rate_l271_271588


namespace angle_ABC_is_correct_l271_271810

noncomputable def vector_BA : ℝ × ℝ := (1/2, real.sqrt 2 / 2)
noncomputable def vector_BC : ℝ × ℝ := (real.sqrt 3 / 2, 1/2)

noncomputable def angle_ABC : ℝ :=
  real.arccos ((vector_BA.fst * vector_BC.fst + vector_BA.snd * vector_BC.snd) /
               (real.sqrt (vector_BA.fst^2 + vector_BA.snd^2) * real.sqrt (vector_BC.fst^2 + vector_BC.snd^2)))

theorem angle_ABC_is_correct : angle_ABC = (3 + real.sqrt 6) / 6 :=
by sorry

end angle_ABC_is_correct_l271_271810


namespace correct_statements_eq_l271_271585

-- Definitions used in the Lean 4 statement should only directly appear in the conditions
variable {a b c : ℝ} 

-- Use the condition directly
theorem correct_statements_eq (h : a / c = b / c) (hc : c ≠ 0) : a = b := 
by
  -- This is where the proof would go
  sorry

end correct_statements_eq_l271_271585


namespace ratio_of_sum_l271_271414

theorem ratio_of_sum (a b c : ℚ) (h1 : b / a = 3) (h2 : c / b = 4) : 
  (2 * a + 3 * b) / (b + 2 * c) = 11 / 27 := 
by
  sorry

end ratio_of_sum_l271_271414


namespace number_of_zeros_in_expansion_of_square_l271_271408

theorem number_of_zeros_in_expansion_of_square (h : 999999999995 = 10^12 - 5) :
  let n := 999999999995 in 
  let z := 11 in 
  ∃ m : ℕ, m = (n^2) ∧ (∃ k : ℕ, k ≤ m ∧ ((10^13 - 10^3) * 10^n == m ∧ k == z)) := 
sorry

end number_of_zeros_in_expansion_of_square_l271_271408


namespace projection_magnitude_is_five_l271_271121

variables {𝕜 : Type*} [IsROrC 𝕜] {E : Type*} [NormedAddCommGroup E] [InnerProductSpace 𝕜 E]
variables (v w : E)

-- Define the conditions
def dot_product_condition : Prop := ⟪v, w⟫ = (-5 : 𝕜)
def norm_condition : Prop := ‖w‖ = 7

-- Define the projection magnitude calculation
def projection_magnitude : 𝕜 := ‖(⟪v, w⟫ / (‖w‖ * ‖w‖)) • w‖

-- State the theorem
theorem projection_magnitude_is_five
  (h1 : dot_product_condition v w)
  (h2 : norm_condition w) :
  projection_magnitude v w = (5 : ℝ) :=
sorry

end projection_magnitude_is_five_l271_271121


namespace f_even_not_odd_l271_271885

def M_x_n (x : ℝ) (n : ℕ) : ℝ := ∏ i in Finset.range n, (x + i)

def f (x : ℝ) : ℝ := x * M_x_n (x - 9) 19

theorem f_even_not_odd : (∀ x : ℝ, f (-x) = f x) ∧ ¬ (∀ x : ℝ, f (-x) = -f x) :=
by
  sorry

end f_even_not_odd_l271_271885


namespace meeting_point_of_mark_and_sandy_l271_271146

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def shifted_right (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1 + d, p.2)

theorem meeting_point_of_mark_and_sandy :
  let mark := (2, 3)
  let sandy := (6, -5)
  let midpoint_coord := midpoint mark sandy
  let meeting_point := shifted_right midpoint_coord 2
  meeting_point = (6, -1) :=
by {
  let mark : ℝ × ℝ := (2, 3)
  let sandy : ℝ × ℝ := (6, -5)
  let midpoint_coord := midpoint mark sandy
  let meeting_point := shifted_right midpoint_coord 2
  have H1 : midpoint_coord = (4, -1) := by {
      unfold midpoint,
      simp,
  }
  have H2 : meeting_point = (6, -1) := by {
      unfold shifted_right,
      rw H1,
      simp,
  }
  exact H2,
}

end meeting_point_of_mark_and_sandy_l271_271146


namespace directrix_of_parabola_l271_271523

theorem directrix_of_parabola (y : ℝ) : (x^2 = 2 * y) → (directrix y = -1/2) :=
sorry

end directrix_of_parabola_l271_271523


namespace imaginary_part_of_complex_expression_l271_271802

def complex_imaginary_part (z : ℂ) : ℝ := z.im

theorem imaginary_part_of_complex_expression : 
  complex_imaginary_part ((3 + 2 * complex.I) / complex.I) = -3 := 
by 
  sorry

end imaginary_part_of_complex_expression_l271_271802


namespace find_y_expression_l271_271045

lemma quadratic_two_distinct_real_roots (m : ℝ) (h : m ≠ 0) :
  let Δ := (m^2 + 2)^2 - 4 * (m^2 + 1) in Δ > 0 :=
by {
  let Δ := (m^2 + 2)^2 - 4 * (m^2 + 1),
  have hΔ : Δ = m^4,
  { calc Δ = (m^2 + 2)^2 - 4 * (m^2 + 1) : by ring
       ... = m^4 : by ring },
  rw hΔ,
  exact pow_pos (ne_of_gt (by linarith)) 4,
  sorry,
}

theorem find_y_expression (m : ℝ) (h : m ≠ 0) :
  let y := (x₁ (m) + x₂ (m)) in y = m^2 - 2 :=
by {
  let x₁ := (1 : ℝ),
  let x₂ := m^2 + 1,
  let y := x₂ - 2 * x₁ - 1,
  have h_y : y = m^2 - 2,
  { calc y = (m^2 + 1) - 2 * (1 : ℝ) - 1 : by ring
        ... = m^2 - 2 : by ring },
  rw h_y,
  exact rfl,
  sorry,
}

end find_y_expression_l271_271045


namespace angle_between_skew_lines_l271_271831

-- Define the angle between direction vectors
def angle_between_direction_vectors (l1 l2 : ℝ) : ℝ := 135

-- State the theorem about the angle formed by two skew lines
theorem angle_between_skew_lines (l1 l2 : ℝ) (h : angle_between_direction_vectors l1 l2 = 135) :
  ∃ θ : ℝ, θ = 45 :=
by
  use 45
  sorry

end angle_between_skew_lines_l271_271831


namespace first_player_wins_l271_271852

def grid_size : ℕ := 99
def center : ℕ × ℕ := (grid_size / 2, grid_size / 2)

variable (move_X : ℕ × ℕ → Prop) (move_O : ℕ × ℕ → Prop)

-- Conditions
axiom init_X : move_X center
axiom O_moves : ∀ (p : ℕ × ℕ), 
  p = (center.1 + 1, center.2) ∨ p = (center.1 - 1, center.2) ∨ 
  p = (center.1, center.2 + 1) ∨ p = (center.1, center.2 - 1) ∨ 
  p = (center.1 + 1, center.2 + 1) ∨ p = (center.1 - 1, center.2 - 1) ∨ 
  p = (center.1 + 1, center.2 - 1) ∨ p = (center.1 - 1, center.2 + 1) → 
  move_O p

axiom player_turns : ∀ (n : ℕ), (move_X ∨ move_O) (if even n then (n, n) else center)

-- Question
theorem first_player_wins : 
  (move_X (0, 0) ∨ move_X (0, grid_size) ∨ move_X (grid_size, 0) ∨ move_X (grid_size, grid_size)) := sorry

end first_player_wins_l271_271852


namespace problem1_problem2_l271_271158

def num_balls := 4  -- Number of balls
def num_boxes := 3  -- Number of boxes

-- Conditions for the first problem
def balls := [1, 2, 3, 4]
def boxes := ['A', 'B', 'C']
def ball_3_in_box_b (a : (Fin num_balls → Fin num_boxes)) : Prop := a 2 = 1
def no_empty_boxes (a : (Fin num_balls → Fin num_boxes)) : Prop := 
  ∀ b, ∃ i, a i = b

-- Conditions for the second problem
def ball_1_not_in_box_a (a : (Fin num_balls → Fin num_boxes)) : Prop := a 0 ≠ 0
def ball_2_not_in_box_b (a : (Fin num_balls → Fin num_boxes)) : Prop := a 1 ≠ 1

-- Theorem for the first problem
theorem problem1 : 
  ∃ (a : (Fin num_balls → Fin num_boxes)), 
    ball_3_in_box_b a ∧ no_empty_boxes a := 
sorry

-- Theorem for the second problem
theorem problem2 : 
  ∃ (a : (Fin num_balls → Fin num_boxes)),
    ball_1_not_in_box_a a ∧ ball_2_not_in_box_b a := 
sorry

end problem1_problem2_l271_271158


namespace postage_extra_fee_l271_271181

theorem postage_extra_fee:
  let l1 := 7 in let h1 := 5 in
  let l2 := 8 in let h2 := 2 in
  let l3 := 7 in let h3 := 7 in
  let l4 := 12 in let h4 := 4 in
  (l1 / h1 < 1.4 ∨ l1 / h1 > 2.6) + 
  (l2 / h2 < 1.4 ∨ l2 / h2 > 2.6) + 
  (l3 / h3 < 1.4 ∨ l3 / h3 > 2.6) + 
  (l4 / h4 < 1.4 ∨ l4 / h4 > 2.6) = 3 := by
  sorry

end postage_extra_fee_l271_271181


namespace max_min_angle_rotation_l271_271125

variable {k : ℕ}
variable {a : Vector ℝ k} (hpos : ∀ i, 0 < a.get i)

/-- Represents the ordering of segments x₁, x₂, ..., xₖ. In lean, we don't need to specify 
the segment laying procedure explicitly, but we will consider their properties based on 
the indices i₁, i₂, ..., iₖ as permutations. -/
definition segment_arrangement (x : Vector ℝ k) (perm : Fin k → Fin k) :
  x = a.map perm.get := sorry

/-- We are to show that if the permutation order x₁ ≤ x₂ ≤ ... ≤ xₖ, it maximizes the angle,
and if x₁ ≥ x₂ ≥ ... ≥ xₖ, it minimizes the angle. -/
theorem max_min_angle_rotation (x : Vector ℝ k)
    (perm_max : ∀ i j, i < j → a.get (perm_max i) ≤ a.get (perm_max j)) 
    (perm_min : ∀ i j, i < j → a.get (perm_min i) ≥ a.get (perm_min j)) :
    θ (x.map perm_max.get) = max_angle ∧ θ (x.map perm_min.get) = min_angle :=
sorry

end max_min_angle_rotation_l271_271125


namespace bridge_length_l271_271195

def train_length : ℕ := 170 -- Train length in meters
def train_speed : ℕ := 45 -- Train speed in kilometers per hour
def crossing_time : ℕ := 30 -- Time to cross the bridge in seconds

noncomputable def speed_m_per_s : ℚ := (train_speed * 1000) / 3600

noncomputable def total_distance : ℚ := speed_m_per_s * crossing_time

theorem bridge_length : total_distance - train_length = 205 :=
by
  sorry

end bridge_length_l271_271195


namespace avg_mpg_sum_l271_271434

def first_car_gallons : ℕ := 25
def second_car_gallons : ℕ := 35
def total_miles : ℕ := 2275
def first_car_mpg : ℕ := 40

noncomputable def sum_of_avg_mpg_of_two_cars : ℝ := 76.43

theorem avg_mpg_sum :
  let first_car_miles := (first_car_gallons * first_car_mpg : ℕ)
  let second_car_miles := total_miles - first_car_miles
  let second_car_mpg := (second_car_miles : ℝ) / second_car_gallons
  let sum_avg_mpg := (first_car_mpg : ℝ) + second_car_mpg
  sum_avg_mpg = sum_of_avg_mpg_of_two_cars :=
by
  sorry

end avg_mpg_sum_l271_271434


namespace train_platform_probability_l271_271643

-- Definition of the problem parameters
def num_platforms : ℕ := 16
def distance_between_platforms : ℕ := 200
def max_distance : ℕ := 800
def total_pairs : ℕ := num_platforms * (num_platforms - 1)

-- Favorable outcomes calculations split for edge and central platforms
def favorable_edge_platforms : ℕ := 8 * 8
def favorable_central_platforms : ℕ := 8 * 10
def total_favorable : ℕ := favorable_edge_platforms + favorable_central_platforms

-- Probability calculation
def probability_feet_or_less : ℚ :=
  total_favorable / total_pairs

-- Proving the probability and thus p and q sum
theorem train_platform_probability:
  probability_feet_or_less = 3 / 5 ∧ 
  let p := 3, q := 5 in (p + q = 8) :=
by {
  sorry
}

end train_platform_probability_l271_271643


namespace kaleb_lives_left_l271_271249

theorem kaleb_lives_left (initial_lives : ℕ) (lives_lost : ℕ) (remaining_lives : ℕ) :
  initial_lives = 98 → lives_lost = 25 → remaining_lives = initial_lives - lives_lost → remaining_lives = 73 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end kaleb_lives_left_l271_271249


namespace sticks_form_equilateral_triangle_l271_271707

theorem sticks_form_equilateral_triangle (n : ℕ) :
  n ≥ 5 → (n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5) → 
  ∃ k : ℕ, ∑ i in finset.range (n + 1), i = 3 * k :=
by {
  sorry
}

end sticks_form_equilateral_triangle_l271_271707


namespace three_digit_even_numbers_l271_271218

theorem three_digit_even_numbers :
  let cards := [(0, 1), (2, 3), (4, 5)] in
  (∃ n ∈ finset.Icc 100 999, 
    let digits := (n % 10, (n / 10) % 10, (n / 100) % 10) in
    ∃ card1 ∈ cards, 
    ∃ card2 ∈ cards, 
    ∃ card3 ∈ cards, 
    ((digits.1 = card1.1 ∨ digits.1 = card1.2) ∧ 
     (digits.2 = card2.1 ∨ digits.2 = card2.2) ∧ 
     (digits.3 = card3.1 ∨ digits.3 = card3.2) ∧ 
     (n % 2 = 0)) ∧
    (list.nodup [card1, card2, card3])
  ) = 20 := sorry

end three_digit_even_numbers_l271_271218


namespace triangle_sides_and_angles_l271_271783

theorem triangle_sides_and_angles (a b c : ℝ) (A B C : ℝ) 
  (h_triangle: ∀ (A B C : ℝ), A + B + C = 180) 
  (h_c : c = 10) 
  (h_A : A = 45) 
  (h_C : C = 30) :
  a = 10 * real.sqrt 2 ∧
  B = 105 ∧
  b = 5 * (real.sqrt 2 + real.sqrt 6) := by
  sorry

end triangle_sides_and_angles_l271_271783


namespace simplify_expression_l271_271730

theorem simplify_expression (a b k : ℝ) (h1 : a + b = -k) (h2 : ab = -3) : 
  (a - 3) * (b - 3) = 6 + 3k := 
by 
  sorry

end simplify_expression_l271_271730


namespace max_value_of_f_l271_271974

def f (x : ℝ) : ℝ := |x| - |x - 3|

theorem max_value_of_f : ∃ x : ℝ, (∀ y : ℝ, f(x) ≥ f(y)) ∧ f(x) = 3 := 
sorry

end max_value_of_f_l271_271974


namespace part_one_part_two_l271_271387

-- Part (1)
theorem part_one (x : ℝ) (h : 0 ≤ x ∧ x ≤ π / 4) :
  let f := λ x : ℝ, √2 * (sin x + cos x) - 2 in
  let g := λ x : ℝ, f x + 1 / 2 in
  (∃! y, 0 ≤ y ∧ y ≤ π / 4 ∧ g y = 0) :=
by {
  let f := λ x : ℝ, √2 * (sin x + cos x) - 2,
  let g := λ x : ℝ, f x + 1 / 2,
  sorry
}

-- Part (2)
theorem part_two (a b : ℝ) :
  (∀ x : ℝ, √2 * a * (sin x + cos x) + 2 * b * sin (2 * x) - 2 ≤ 0) →
  -2 ≤ a + b ∧ a + b ≤ 1 :=
by {
  sorry 
}

end part_one_part_two_l271_271387


namespace sticks_form_equilateral_triangle_l271_271708

theorem sticks_form_equilateral_triangle (n : ℕ) :
  n ≥ 5 → (n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5) → 
  ∃ k : ℕ, ∑ i in finset.range (n + 1), i = 3 * k :=
by {
  sorry
}

end sticks_form_equilateral_triangle_l271_271708


namespace ellipse_ratio_condition_l271_271364

theorem ellipse_ratio_condition 
  (a b : ℝ) (h1: a > b) (h2: b > 0) 
  (h3: ∀ P, P ∈ { (x, y) | (x / a)^2 + (y / b)^2 = 1 }
    → ∠(P, (c, 0), (-c, 0)) > π/2) : 
  0 < (b / a) ∧ (b / a) < (Real.sqrt 2 / 2) := 
sorry

end ellipse_ratio_condition_l271_271364


namespace part1_solution_set_part2_range_of_a_l271_271007

-- Define the function f for a general a
def f (x a : ℝ) : ℝ := |x - a^2| + |x - (2 * a) + 1|

-- Part 1: When a = 2
theorem part1_solution_set (x : ℝ) (h : f x 2 ≥ 4) : x ≤ 3/2 ∨ x ≥ 11/2 :=
  sorry

-- Part 2: Range of values for a
theorem part2_range_of_a (a : ℝ) (h : ∀ x, f x a ≥ 4) : a ≤ -1 ∨ a ≥ 3 :=
  sorry

end part1_solution_set_part2_range_of_a_l271_271007


namespace least_positive_integer_exists_l271_271238

theorem least_positive_integer_exists 
  (exists_k : ∃ k, (1 ≤ k ∧ k ≤ 2 * 5) ∧ (5^2 - 5 + k) % k = 0)
  (not_all_k : ¬(∀ k, (1 ≤ k ∧ k ≤ 2 * 5) → (5^2 - 5 + k) % k = 0)) :
  5 = 5 := 
by
  trivial

end least_positive_integer_exists_l271_271238


namespace part1_part2_l271_271918

namespace MathProof

-- defining the basic setup of the triangle and given constraints
variables (A B C : ℝ) (a b c : ℝ)

-- condition 1: the equation relating cosines and sines
def condition1 : Prop := (cos A) / (1 + sin A) = (sin (2 * B)) / (1 + cos (2 * B))

-- condition 2: specific value of angle C
def condition2 : Prop := C = 2 * π / 3

-- question 1: finding angle B
def findB : Prop := B = π / 6

-- The minimum value of (a^2 + b^2) / c^2
def min_value_expr : ℝ := (a^2 + b^2) / c^2
def min_value : ℝ := 4 * real.sqrt 2 - 5

-- Proving both parts
theorem part1 (h1 : condition1) (h2 : condition2) : findB := sorry

theorem part2 (h1 : condition1) (h2 : condition2) : min_value_expr = min_value := sorry

end MathProof

end part1_part2_l271_271918


namespace tiffany_sequence_sum_l271_271552

theorem tiffany_sequence_sum :
  let original_seq := List.replicate 2000 [1, 2, 3, 4, 5, 6].join  -- Create the initial sequence
      seq1 := original_seq.enum.filter (λ idx_d, idx_d.1 % 4 ≠ 3) |>.map Prod.snd -- First erasure every 4th digit (0-based 3rd, 7th, ...)
      seq2 := seq1.enum.filter (λ idx_d, idx_d.1 % 5 ≠ 4) |>.map Prod.snd         -- Second erasure every 5th digit of the remaining
      final_seq := seq2.enum.filter (λ idx_d, idx_d.1 % 3 ≠ 2) |>.map Prod.snd    -- Third erasure every 3rd digit of the remaining
  in (final_seq !! 3030).getOrElse 0 + (final_seq !! 3031).getOrElse 0 + (final_seq !! 3032).getOrElse 0 = 6 :=
begin
  sorry
end

end tiffany_sequence_sum_l271_271552


namespace find_upper_base_length_l271_271759

-- Define the trapezoid and its properties.
variables (d : ℝ)
variables (A D : ℝ × ℝ) (M : ℝ × ℝ) (C : ℝ × ℝ)

-- Define the conditions of the problem.
def is_trapezoid (A B C D : ℝ × ℝ) : Prop := 
  ∃ M, M.1 = (A.1 + B.1) / 2 ∧ M.1 = (C.1 + D.1) / 2
  
def perpendicular (P Q : ℝ × ℝ) : Prop :=
  P.1 = Q.1

def equal_distance (P Q R : ℝ × ℝ) : Prop :=
  dist P Q = dist Q R

-- Setting the exact locations of points
def coordinates : Prop := 
  D = (0, 0) ∧ A = (d, 0) ∧ perpendicular (M) (D, A)

-- Required proof
theorem find_upper_base_length :
  coordinates d A D M ∧ equal_distance M C D → 
  dist (A, C) = d / 2 :=
by sorry

end find_upper_base_length_l271_271759


namespace centroids_coincide_l271_271155

variables {A B C P Q R : Type*} [vector_space ℝ A] [vector_space ℝ B] [vector_space ℝ C]
variables (k : ℝ)
variables [affine_space A ℝ] [affine_space B ℝ] [affine_space C ℝ]

/- Points P, Q, R divide the sides AB, BC, and AC in ratio k:1 -/
def divides_in_ratio (A B P : point) (k : ℝ) : Prop := 
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧ P = (A -ᵥ B) • (k / (k + 1)) +ᵥ B

noncomputable def centroid (X Y Z : point) : point :=
  (X + Y + Z) / 3

theorem centroids_coincide (A B C P Q R : point) (k : ℝ)
  (hP : divides_in_ratio A B P k)
  (hQ : divides_in_ratio B C Q k)
  (hR : divides_in_ratio C A R k) :
  centroid A B C = centroid P Q R :=
begin
  sorry
end

end centroids_coincide_l271_271155


namespace part1_part2_l271_271017

def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1 (x : ℝ) : 
  ∀ x, f x 2 ≥ 4 ↔ (x ≤ 3 / 2 ∨ x ≥ 11 / 2) :=
by sorry

theorem part2 (x a : ℝ) :
  f x a ≥ 4 → (a ≤ -1 ∨ a ≥ 3) :=
by sorry

end part1_part2_l271_271017


namespace half_angle_third_quadrant_l271_271372

open Real

theorem half_angle_third_quadrant (θ k : ℤ→ ℝ) 
  (h1 : θ ∈ Ioo ((2 * k : ℝ) * π + π / 2) ((2 * k : ℝ) * π + π))
  (h2 : abs (sin (θ / 2)) = -sin (θ / 2)) :
  θ / 2 ∈ Ioo ((k : ℝ) * π + π / 2) ((k : ℝ) * π + π) :=
begin
  sorry
end

end half_angle_third_quadrant_l271_271372


namespace street_length_l271_271280

theorem street_length (t : ℕ) (s : ℕ) (h1 : t = 5) (h2 : s = 9.6) : (s * 1000 / 60) * t = 800 :=
by sorry

end street_length_l271_271280


namespace amount_subtracted_correct_l271_271266

noncomputable def find_subtracted_amount (N : ℝ) (A : ℝ) : Prop :=
  0.40 * N - A = 23

theorem amount_subtracted_correct :
  find_subtracted_amount 85 11 :=
by
  sorry

end amount_subtracted_correct_l271_271266


namespace simplify_polynomial_l271_271564

-- Define the polynomial expression
def expr : ℝ[X] := (5 - 7 * X - 13 * X^2 + 10 + 15 * X - 25 * X^2 - 20 + 21 * X + 33 * X^2 - 15 * X^3)

-- Define the simplified expression
def simplified_expr : ℝ[X] := (-15 * X^3 - 5 * X^2 + 29 * X - 5)

-- The theorem that needs to be proved
theorem simplify_polynomial : expr = simplified_expr := by
  sorry

end simplify_polynomial_l271_271564


namespace find_functions_l271_271335

theorem find_functions (f : ℝ → ℝ) (h : ∀ x : ℝ, f (2002 * x - f 0) = 2002 * x^2) :
  (∀ x, f x = (x^2) / 2002) ∨ (∀ x, f x = (x^2) / 2002 + 2 * x + 2002) :=
sorry

end find_functions_l271_271335


namespace smallest_period_of_f_max_min_of_f_on_interval_l271_271797

noncomputable def f (x : ℝ) : ℝ := (sin x + cos x)^2 + cos (2 * x) - 1

theorem smallest_period_of_f : ∀ x, f (x + π) = f x :=
by
  sorry

theorem max_min_of_f_on_interval :
    (∀ x, -π / 4 ≤ x ∧ x ≤ π / 4 → -√2 ≤ f x ∧ f x ≤ √2) :=
by
  sorry

end smallest_period_of_f_max_min_of_f_on_interval_l271_271797


namespace quadratic_roots_l271_271654

theorem quadratic_roots (b c : ℝ) :
  (∀ x : ℝ, x^2 + b*x + c = 0 → (x = 2 ∨ x = -3)) → b = 1 ∧ c = -6 :=
by
  assume h
  have h_sum : 2 + (-3) = -b, from sorry
  have h_prod : 2 * (-3) = c, from sorry
  rw [neg_eq_iff_add_eq_zero] at h_sum
  rw h_sum at h_prod
  exact ⟨h_sum, h_prod⟩

end quadratic_roots_l271_271654


namespace area_of_right_triangle_with_given_altitude_l271_271182

open Real EuclideanGeometry

namespace ProofProblem

theorem area_of_right_triangle_with_given_altitude 
  (ABC : Triangle ℝ)
  (B C : Point ℝ)
  (h_right : angle ABC = π / 2)
  (h_B_45 : angle B = π / 4)
  (h_C_45 : angle C = π / 4)
  (h_altitude : altitude ABC from B to hypotenuse AC = 4) :
  area ABC = 8 * sqrt 2 :=
by 
  -- We start by considering the definition of a 45-45-90 triangle.
  -- Since altitude from B to AC is given as 4, we can compute the hypotenuse
  -- and the required area.
  sorry

end ProofProblem

end area_of_right_triangle_with_given_altitude_l271_271182


namespace absent_workers_efficiency_l271_271618

-- Define the total work, efficiency levels, and number of workers.
variables {W : ℝ} {N : ℕ} 
variables (E : Fin N → ℝ) -- Efficiency levels of the workers

-- Conditions from the problem
variable (H1 : W = (Finset.univ.sum E) * 20) -- Total work equation for the original group
variable (H2 : W = (Finset.univ.erase (Finset.univ.erase Finset.univ (Fin N (N-1))) (Fin N (N-2)).sum E) * 22) 
  -- Total work equation for the remaining group after 2 workers absent

-- Prove that the efficiency levels of the absent workers add up to a specific value
theorem absent_workers_efficiency :
  (E (Fin N (N-1)) + E (Fin N (N-2))) = (Finset.univ.erase (Finset.univ.erase Finset.univ (Fin N (N-1))) (Fin N (N-2)).sum E) / 11 :=
by
  sorry

end absent_workers_efficiency_l271_271618


namespace find_a_value_l271_271193

theorem find_a_value 
  (f : ℝ → ℝ)
  (a : ℝ)
  (h : ∀ x : ℝ, f x = x^3 + a*x^2 + 3*x - 9)
  (extreme_at_minus_3 : ∀ f' : ℝ → ℝ, (∀ x, f' x = 3*x^2 + 2*a*x + 3) → f' (-3) = 0) :
  a = 5 := 
sorry

end find_a_value_l271_271193


namespace unique_two_scoop_sundaes_l271_271620

theorem unique_two_scoop_sundaes (n : ℕ) (h_n : n = 8) : (nat.choose n 2) = 28 :=
by
  rw h_n
  exact nat.choose_eq_fact_div_fact_mul_fact 8 2
  -- sorry to skip the proof, since it is not required.
  sorry

end unique_two_scoop_sundaes_l271_271620


namespace upper_base_length_l271_271763

-- Definitions based on the given conditions
variables (A B C D M : ℝ)
variables (d : ℝ)
variables (h : ℝ) -- height from D to AB

-- Conditions given in the problem
def is_trapezoid (ABCD : ℝ) : Prop := true
def is_perpendicular (DM AB : ℝ) : Prop := true
def point_on_side (M AB : ℝ) : Prop := true
def MC_eq_CD (MC CD : ℝ) : Prop := MC = CD
def AD_length (A D d : ℝ) : Prop := A - D = d -- Assuming some coordinate system 

-- Define the proof statement with conditions and the result
theorem upper_base_length
  (ht : is_trapezoid ABCD)
  (hp : is_perpendicular D M)
  (ps : point_on_side M AB)
  (mc_cd : MC_eq_CD M C D)
  (ad_len : AD_length A D d) :
  BC = d / 2 :=
sorry

end upper_base_length_l271_271763


namespace problem1_solutions_problem2_solutions_l271_271960

-- Problem 1: Prove that the solutions to the equation x^2 - 4 = 0 are x = 2 and x = -2
theorem problem1_solutions (x : ℝ) : (x^2 - 4 = 0) → (x = 2 ∨ x = -2) :=
begin
  sorry
end

-- Problem 2: Prove that the solutions to the equation (2x + 3)^2 = 4(2x + 3) are x = -3/2 and x = 1/2
theorem problem2_solutions (x : ℝ) : ((2 * x + 3)^2 = 4 * (2 * x + 3)) → (x = -3/2 ∨ x = 1/2) :=
begin
  sorry
end

end problem1_solutions_problem2_solutions_l271_271960


namespace tan_A_eq_sqrt41_div_4_l271_271459

noncomputable def tangent_of_angle_A (AB BC : ℝ) : ℝ :=
  BC / AB

theorem tan_A_eq_sqrt41_div_4 (AB BC : ℝ) (h1 : AB = 4) (h2 : BC = Real.sqrt 41) :
  tangent_of_angle_A AB BC = Real.sqrt 41 / 4 :=
by
  rw [h1, h2]
  simp
  sorry

end tan_A_eq_sqrt41_div_4_l271_271459


namespace original_number_of_people_is_fifteen_l271_271965

/-!
The average age of all the people who gathered at a family celebration was equal to the number of attendees. 
Aunt Beta, who was 29 years old, soon excused herself and left. 
Even after Aunt Beta left, the average age of all the remaining attendees was still equal to their number.
Prove that the original number of people at the celebration is 15.
-/

theorem original_number_of_people_is_fifteen
  (n : ℕ)
  (s : ℕ)
  (h1 : s = n^2)
  (h2 : s - 29 = (n - 1)^2):
  n = 15 :=
by
  sorry

end original_number_of_people_is_fifteen_l271_271965


namespace graph_is_two_lines_l271_271314

theorem graph_is_two_lines (x y : ℝ) :
  x^2 - 50 * y^2 - 10 * x + 25 = 0 →
  (x = 5 + 5 * real.sqrt 2 * y ∨ x = 5 - 5 * real.sqrt 2 * y) :=
by
  sorry

end graph_is_two_lines_l271_271314


namespace zeros_in_expansion_l271_271410

theorem zeros_in_expansion : 
  let x := 10^12 - 5 in
  (x^2).toString.filter (· = '0').length = 12 
  := 
by 
  let x := 10^12 - 5
  -- Compute the square of (10^12 - 5)
  have h : (x^2).toString = "9999000000000000000025" := sorry
  -- Count the number of zeros in the resultant string
  have count_zeros : ("9999000000000000000025".filter (· = '0')).length = 12 := sorry
  exact count_zeros

end zeros_in_expansion_l271_271410


namespace find_remainder_l271_271342

noncomputable def remainder_x2023_div_x2_minus_1_x_plus_2 
  (p : ℕ) : Polynomial ℤ := 
Polynomial.X ^ (2 * 10^3 + 23)

theorem find_remainder (p : ℕ) : 
  remainder_x2023_div_x2_minus_1_x_plus_2 p % ((Polynomial.X ^ 2 - 1) * (Polynomial.X + 2)) = 
  Polynomial.X ^ 3 :=
sorry

end find_remainder_l271_271342


namespace exists_two_cos_inequality_l271_271951

open Real

theorem exists_two_cos_inequality (a b c d : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < c) (h4 : c < d) (h5 : d < π / 2) :
  ∃ (x y : ℝ), x ∈ {a, b, c, d} ∧ y ∈ {a, b, c, d} ∧ x ≠ y ∧ 8 * cos x * cos y * cos (x - y) + 1 > 4 * (cos x ^ 2 + cos y ^ 2) :=
by
  sorry

end exists_two_cos_inequality_l271_271951


namespace count_good_numbers_l271_271838

theorem count_good_numbers : 
  let is_good_number (n a b : ℕ) := n = 10 * a + b ∧ a > 0 ∧ b > 0 ∧ gcd n a > 1
  in (finsets.range 90).filter (λ n, ∃ a b, is_good_number n a b) = 41 :=
by sorry

end count_good_numbers_l271_271838


namespace students_with_dogs_l271_271633

theorem students_with_dogs (total_students : ℕ) (half_students : total_students / 2 = 50)
  (percent_girls_with_dogs : ℕ → ℚ) (percent_boys_with_dogs : ℕ → ℚ)
  (girls_with_dogs : ∀ (total_girls: ℕ), percent_girls_with_dogs total_girls = 0.2)
  (boys_with_dogs : ∀ (total_boys: ℕ), percent_boys_with_dogs total_boys = 0.1) :
  ∀ (total_girls total_boys students_with_dogs: ℕ),
  total_students = 100 →
  total_girls = total_students / 2 →
  total_boys = total_students / 2 →
  total_girls = 50 →
  total_boys = 50 →
  students_with_dogs = (percent_girls_with_dogs (total_students / 2) * (total_students / 2) + 
                        percent_boys_with_dogs (total_students / 2) * (total_students / 2)) →
  students_with_dogs = 15 :=
by
  intros total_girls total_boys students_with_dogs h1 h2 h3 h4 h5 h6
  sorry

end students_with_dogs_l271_271633


namespace equation_has_solution_implies_a_ge_2_l271_271336

theorem equation_has_solution_implies_a_ge_2 (a : ℝ) :
  (∃ x : ℝ, 4^x - a * 2^x - a + 3 = 0) → a ≥ 2 :=
by
  sorry

end equation_has_solution_implies_a_ge_2_l271_271336


namespace find_angle_B_l271_271841

open Real

theorem find_angle_B (A B : ℝ) 
  (h1 : 0 < B ∧ B < A ∧ A < π/2)
  (h2 : cos A = 1/7) 
  (h3 : cos (A - B) = 13/14) : 
  B = π/3 :=
sorry

end find_angle_B_l271_271841


namespace area_G1G2G3_l271_271115

variable {P A B C G1 G2 G3 : Type} [MetricSpace A B C P]

-- Define the point P inside the triangle ABC and the centroids G1, G2, and G3
-- Define the areas of triangles
noncomputable def area (t : Type) [metric_space t] : ℝ := sorry

-- Assume the conditions
variable [is_triangle A B C]
variable (P : A)
variable [is_inside_triangle P A B C]
variable (G1 : A) [is_centroid G1 P B C]
variable (G2 : A) [is_centroid G2 P C A]
variable (G3 : A) [is_centroid G3 P A B]

-- Given the area of triangle ABC is 24
axiom h1 : area (triangle A B C) = 24

-- Define the area of triangle G1 G2 G3
theorem area_G1G2G3 : area (triangle G1 G2 G3) = 2.67 := by
  sorry

end area_G1G2G3_l271_271115


namespace find_period_find_analytic_expression_find_sin_theta_l271_271381

-- Definitions based on conditions in the problem
def f (x : ℝ) (A : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := A * Real.sin (ω * x + φ) + 1

-- 1. Prove the smallest positive period is π
theorem find_period (A ω φ : ℝ) (hA : A > 0) (hω : ω > 0)
  (hφ1 : -Real.pi / 2 ≤ φ) (hφ2 : φ ≤ Real.pi / 2)
  (h_dist : ∀ x, f x A ω φ = f (x + Real.pi) A ω φ) :
  (∃ T > 0, ∀ x, f (x + T) A ω φ = f x A ω φ) ∧ 
  ∃ T > 0, T = Real.pi := 
by
  sorry

-- 2. Prove the analytic expression of f(x)
theorem find_analytic_expression (A ω φ : ℝ) (hA : A > 0) (h_ω_eq : ω = 2)
  (h_max : ∃ x, f x A ω φ = 3) (h_sym : ∃ x, f x A ω φ = f (-x + 2 * Real.pi / 3) A ω φ) 
  (h_phi_range: φ = -Real.pi / 6) :
  f x 2 2 (-Real.pi / 6) = 2 * Real.sin (2 * x - Real.pi / 6) + 1 :=
by
  sorry

-- 3. Prove the possible values of sin(θ) 
theorem find_sin_theta (A ω φ θ : ℝ)
  (h_eq : f (θ / 2 + Real.pi / 3) A ω φ = 7 / 5) 
  (h_x : f (θ / 2 + Real.pi / 3) 2 2 (-Real.pi / 6) = 2 *Real.sin (2 * (θ / 2 + Real.pi / 3) - Real.pi / 6) + 1):
  (Real.sin θ = 2 * Real.sqrt 6 / 5) ∨ (Real.sin θ = -2 * Real.sqrt 6 / 5) :=
by
  sorry

end find_period_find_analytic_expression_find_sin_theta_l271_271381


namespace sticks_forming_equilateral_triangle_l271_271698

theorem sticks_forming_equilateral_triangle (n : ℕ) (h1 : n ≥ 5) :
  (∃ l : list ℕ, (∀ x ∈ l, x ∈ finset.range (n + 1) ∧ x > 0) ∧ l.sum % 3 = 0) ↔ 
  (n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5) := 
sorry

end sticks_forming_equilateral_triangle_l271_271698


namespace fencing_required_l271_271595

theorem fencing_required (L W : ℕ) (area : ℕ) (hL : L = 20) (hA : area = 120) (hW : area = L * W) :
  2 * W + L = 32 :=
by
  -- Steps and proof logic to be provided here
  sorry

end fencing_required_l271_271595


namespace trapezoid_EFGH_circle_center_Q_EQ_l271_271556

noncomputable def EQ := 160 / 3

-- Define p and q based on EQ being in its simplest fraction form
noncomputable def p := 160
noncomputable def q := 3

theorem trapezoid_EFGH_circle_center_Q_EQ :
  let EF := 86
  let FG := 60
  let GH := 26
  let HE := 80
  let parallel : Prop := EF ∥ GH
  let Q_condition : Prop := true -- placeholder for the circle's positional conditions
  in p + q = 163 :=
by
  have EF : ℕ := 86
  have FG : ℕ := 60
  have GH : ℕ := 26
  have HE : ℕ := 80
  have parallel : EF ∥ GH := sorry -- proof that EF and GH are parallel
  have Q_condition : true := sorry -- positional conditions of Q on EF and tangent to FG and HE
  have p + q = 160 + 3 := sorry   -- the calculated sum of p and q
  exact this

end trapezoid_EFGH_circle_center_Q_EQ_l271_271556


namespace part_a_part_b_l271_271600

-- Define the setup for the problem
variables {A B C P A1 B1 : Type} [HasCircumcircle A B C P]
variables (PA PA1 PB1 : ℝ) (R d : ℝ)
variables (h1 : Perpendicular PA1 BC) (h2: Perpendicular PB1 AC)
variables (h3 : Distance P A1B1 = d)
variables (h4 : IsAngleBetween A1B1 BC α)

-- Define the problem statements
theorem part_a : PA * PA1 = 2 * R * d :=
by
  sorry

theorem part_b : cos α = PA / (2 * R) :=
by
  sorry

end part_a_part_b_l271_271600


namespace count_odd_ad_bc_l271_271744

open Finset

def is_odd (n : ℤ) : Prop := n % 2 ≠ 0

theorem count_odd_ad_bc :
  let s := {0, 1, 2, 3}
  finset.card {p : ℤ × ℤ × ℤ × ℤ | p.1.1 ∈ s ∧ p.1.2 ∈ s ∧ p.2.1 ∈ s ∧ p.2.2 ∈ s ∧
                                    is_odd (p.1.1 * p.2.2 - p.1.2 * p.2.1) } = 96 :=
by
  sorry

end count_odd_ad_bc_l271_271744


namespace find_start_time_l271_271230

def time_first_train_started 
  (distance_pq : ℝ) 
  (speed_train1 : ℝ) 
  (speed_train2 : ℝ) 
  (start_time_train2 : ℝ) 
  (meeting_time : ℝ) 
  (T : ℝ) : ℝ :=
  T

theorem find_start_time 
  (distance_pq : ℝ := 200)
  (speed_train1 : ℝ := 20)
  (speed_train2 : ℝ := 25)
  (start_time_train2 : ℝ := 8)
  (meeting_time : ℝ := 12) 
  : time_first_train_started distance_pq speed_train1 speed_train2 start_time_train2 meeting_time 7 = 7 :=
by
  sorry

end find_start_time_l271_271230


namespace quadrilateral_covered_by_circles_l271_271533

theorem quadrilateral_covered_by_circles
  {A B C D : Type}
  (distance : A → A → ℝ)
  (convex_quadrilateral : A → A → A → A → Prop)
  (side_length_at_most_7 : ∀ (X Y : A), distance X Y ≤ 7)
  (r : ℝ) : 
  r = 5 →
  convex_quadrilateral A B C D → 
  (∀ O : A, 
    (distance O A ≤ r ∨ distance O B ≤ r ∨ distance O C ≤ r ∨ distance O D ≤ r)) :=
by
  intros hr hcvq O
  sorry

end quadrilateral_covered_by_circles_l271_271533


namespace savings_in_july_l271_271223

-- Definitions based on the conditions
def savings_june : ℕ := 27
def savings_august : ℕ := 21
def expenses_books : ℕ := 5
def expenses_shoes : ℕ := 17
def final_amount_left : ℕ := 40

-- Main theorem stating the problem
theorem savings_in_july (J : ℕ) : 
  savings_june + J + savings_august - (expenses_books + expenses_shoes) = final_amount_left → 
  J = 14 :=
by
  sorry

end savings_in_july_l271_271223


namespace eval_expr_at_3_l271_271503

-- Define the expression
def expr (x : ℝ) : ℝ :=
  ((x^2 - x) / (x^2 - 2*x + 1) + 2 / (x - 1)) / ((x^2 - 4) / (x^2 - 1))

-- The assumption x ≠ 1 and x ≠ 2
def valid_x (x : ℝ) : Prop :=
  x ≠ 1 ∧ x ≠ 2

-- Prove that for x = 3, the expression evaluates to 4
theorem eval_expr_at_3 : expr 3 = 4 :=
by
  -- proof goes here
  sorry

end eval_expr_at_3_l271_271503


namespace sum_sqr_diff_ge_const_l271_271482

noncomputable def sum_abs_diff {n : ℕ} (a : Fin n → ℝ) : ℝ :=
  ∑ i in Finset.univ, ∑ j in Finset.univ, |a i - a j|

noncomputable def sum_sqr_diff {n : ℕ} (a : Fin n → ℝ) : ℝ :=
  ∑ i in Finset.univ, ∑ j in Finset.univ, (a i - a j)^2

theorem sum_sqr_diff_ge_const {n : ℕ} (a : Fin n → ℝ) (h1 : (∀ i j, i < j → a i < a j))
  (h2 : sum_abs_diff a = 1) :
  sum_sqr_diff a ≥ (3 / (2 * (n^2 - 1))) :=
by
  -- proof goes here
  sorry

end sum_sqr_diff_ge_const_l271_271482


namespace lamp_probability_approximation_l271_271624

/-- Given a network of 18000 lamps with each lamp having a 0.9 probability of being turned on, 
the probability that the number of lamps turned on differs from its expected value by no more than 200 
is approximately 0.955. -/
theorem lamp_probability_approximation :
  let n := 18000
  let p := 0.9
  let expected_value := n * p
  let variance := n * p * (1 - p)
  let k := 200
  let chebyshev_probability := 1 - variance / k^2
  let clt_standard_deviation := sqrt variance
  let standardized_value := k / clt_standard_deviation
  -- The final probability using CLT and normal approximation:
  let normal_approximation_probability := 0.955 in
  chebyshev_probability ≥ 0.955 ∧
  -- Using Chebyshev's inequality:
  chebyshev_probability = 1 - variance / k^2 ∧
  -- Using the Central Limit Theorem:
  normal_approximation_probability ≈ 0.955 :=
sorry

end lamp_probability_approximation_l271_271624


namespace classes_divided_by_3_then_added_12_is_20_l271_271990

-- Define the conditions
def three_erasers_per_class (C : ℕ) : ℕ := 3 * C
def remaining_erasers_after_throw (broken : ℕ) (remaining : ℕ) : ℕ := broken + remaining

-- Constants given by the problem
constant initial_erasers : ℕ := 72
constant broken_erasers : ℕ := 12
constant remaining_erasers : ℕ := 60

-- The final proof we are aiming for
theorem classes_divided_by_3_then_added_12_is_20 (C : ℕ) (h1 : 3 * C = initial_erasers) : C / 3 + 12 = 20 :=
by
  -- Begin proof here (proof skipped with "sorry")
  sorry

end classes_divided_by_3_then_added_12_is_20_l271_271990


namespace integers_a_b_c_d_arbitrarily_large_l271_271173

theorem integers_a_b_c_d_arbitrarily_large (n : ℤ) : 
  ∃ (a b c d : ℤ), (a^2 + b^2 + c^2 + d^2 = a * b * c + a * b * d + a * c * d + b * c * d) ∧ 
    min (min a b) (min c d) ≥ n := 
by sorry

end integers_a_b_c_d_arbitrarily_large_l271_271173


namespace sin_pi_plus_alpha_eq_neg_four_fifths_l271_271353

theorem sin_pi_plus_alpha_eq_neg_four_fifths 
  (α : ℝ) 
  (h1 : sin (π / 2 + α) = 3 / 5) 
  (h2 : 0 < α ∧ α < π / 2) : 
  sin (π + α) = -4 / 5 :=
by
  sorry

end sin_pi_plus_alpha_eq_neg_four_fifths_l271_271353


namespace percent_defective_units_l271_271088

-- Definition of the given problem conditions
variable (D : ℝ) -- D represents the percentage of defective units

-- The main statement we want to prove
theorem percent_defective_units (h1 : 0.04 * D = 0.36) : D = 9 := by
  sorry

end percent_defective_units_l271_271088


namespace exists_large_quadrilateral_l271_271952

theorem exists_large_quadrilateral (n : ℕ) (h : 4 ≤ n) (A : Type*) [convex_polygon A n] 
  (area_A : polygon_area A = 1) : 
  ∃ (Q : set (finset ℕ)), 
    (quadrilateral Q ∧ polygon_area Q ≥ 1/2) :=
by
  sorry

end exists_large_quadrilateral_l271_271952


namespace triangle_problem_l271_271073

theorem triangle_problem
  (A B C : ℝ)
  (a b c : ℝ)
  (hb : 0 < B ∧ B < Real.pi)
  (hc : 0 < C ∧ C < Real.pi)
  (ha : 0 < A ∧ A < Real.pi)
  (h_sum_angles : A + B + C = Real.pi)
  (h_sides : a > b)
  (h_perimeter : a + b + c = 20)
  (h_area : (1/2) * a * b * Real.sin C = 10 * Real.sqrt 3)
  (h_eq : a * (Real.sqrt 3 * Real.tan B - 1) = (b * Real.cos A / Real.cos B) + (c * Real.cos A / Real.cos C)) :
  C = Real.pi / 3 ∧ a = 8 ∧ b = 5 ∧ c = 7 := sorry

end triangle_problem_l271_271073


namespace same_grade_percentage_l271_271986

theorem same_grade_percentage (total_students: ℕ)
  (a_students: ℕ) (b_students: ℕ) (c_students: ℕ) (d_students: ℕ)
  (total: total_students = 30)
  (a: a_students = 2) (b: b_students = 4) (c: c_students = 5) (d: d_students = 1)
  : (a_students + b_students + c_students + d_students) * 100 / total_students = 40 := by
  sorry

end same_grade_percentage_l271_271986


namespace sum_of_sequence_l271_271526

noncomputable def a_n : ℕ → ℚ
| 0 := 2
| (n+1) := (1/2) * (finset.sum finset.range (λ i, a_n i))

def S_n (n : ℕ) : ℚ := finset.sum (finset.range n) a_n

theorem sum_of_sequence (n : ℕ) : S_n n = 2 * (3/2)^(n-1) :=
sorry

end sum_of_sequence_l271_271526


namespace find_upper_base_length_l271_271757

-- Define the trapezoid and its properties.
variables (d : ℝ)
variables (A D : ℝ × ℝ) (M : ℝ × ℝ) (C : ℝ × ℝ)

-- Define the conditions of the problem.
def is_trapezoid (A B C D : ℝ × ℝ) : Prop := 
  ∃ M, M.1 = (A.1 + B.1) / 2 ∧ M.1 = (C.1 + D.1) / 2
  
def perpendicular (P Q : ℝ × ℝ) : Prop :=
  P.1 = Q.1

def equal_distance (P Q R : ℝ × ℝ) : Prop :=
  dist P Q = dist Q R

-- Setting the exact locations of points
def coordinates : Prop := 
  D = (0, 0) ∧ A = (d, 0) ∧ perpendicular (M) (D, A)

-- Required proof
theorem find_upper_base_length :
  coordinates d A D M ∧ equal_distance M C D → 
  dist (A, C) = d / 2 :=
by sorry

end find_upper_base_length_l271_271757


namespace sticks_form_equilateral_triangle_l271_271689

theorem sticks_form_equilateral_triangle (n : ℕ) (h1 : n ≥ 5) (h2 : n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5) :
  ∃ k, k * 3 = (n * (n + 1)) / 2 :=
by
  sorry

end sticks_form_equilateral_triangle_l271_271689


namespace solution_set_equivalent_l271_271983

noncomputable def solution_set_inequality (a b c x : ℝ) :=
  a * x^2 - b * x + c 

axiom ax_smaller : ∀ x : ℝ, x ∈ {x | a * x^2 + b * x + c > 0} ↔ (2 < x ∧ x < 3)
axiom root1 : a * 2^2 + b * 2 + c = 0
axiom root2 : a * 3^2 + b * 3 + c = 0
axiom a_neg : a < 0

theorem solution_set_equivalent : ∀ x : ℝ, (a * x^2 - b * x + c > 0) ↔ (-3 < x ∧ x < -2) :=
by
  sorry -- Proof is omitted as instructed.

end solution_set_equivalent_l271_271983


namespace john_total_trip_cost_l271_271872

noncomputable def total_trip_cost
  (hotel_nights : ℕ) 
  (hotel_rate_per_night : ℝ) 
  (discount : ℝ) 
  (loyal_customer_discount_rate : ℝ) 
  (service_tax_rate : ℝ) 
  (room_service_cost_per_day : ℝ) 
  (cab_cost_per_ride : ℝ) : ℝ :=
  let hotel_cost := hotel_nights * hotel_rate_per_night
  let cost_after_discount := hotel_cost - discount
  let loyal_customer_discount := loyal_customer_discount_rate * cost_after_discount
  let cost_after_loyalty_discount := cost_after_discount - loyal_customer_discount
  let service_tax := service_tax_rate * cost_after_loyalty_discount
  let final_hotel_cost := cost_after_loyalty_discount + service_tax
  let room_service_cost := hotel_nights * room_service_cost_per_day
  let cab_cost := cab_cost_per_ride * 2 * hotel_nights
  final_hotel_cost + room_service_cost + cab_cost

theorem john_total_trip_cost : total_trip_cost 3 250 100 0.10 0.12 50 30 = 985.20 :=
by 
  -- We are skipping the proof but our focus is the statement
  sorry

end john_total_trip_cost_l271_271872


namespace part1_part2_l271_271917

namespace MathProof

-- defining the basic setup of the triangle and given constraints
variables (A B C : ℝ) (a b c : ℝ)

-- condition 1: the equation relating cosines and sines
def condition1 : Prop := (cos A) / (1 + sin A) = (sin (2 * B)) / (1 + cos (2 * B))

-- condition 2: specific value of angle C
def condition2 : Prop := C = 2 * π / 3

-- question 1: finding angle B
def findB : Prop := B = π / 6

-- The minimum value of (a^2 + b^2) / c^2
def min_value_expr : ℝ := (a^2 + b^2) / c^2
def min_value : ℝ := 4 * real.sqrt 2 - 5

-- Proving both parts
theorem part1 (h1 : condition1) (h2 : condition2) : findB := sorry

theorem part2 (h1 : condition1) (h2 : condition2) : min_value_expr = min_value := sorry

end MathProof

end part1_part2_l271_271917


namespace alpha_greater_than_three_f_decreasing_range_of_a_l271_271392

variable (a : ℝ)
variable (α β : ℝ)
variable (x : ℝ)

-- Problem 1: Prove that α > 3
theorem alpha_greater_than_three 
  (h1 : 0 < a ∧ a < 1) 
  (h2 : ∀ x, α ≤ x ∧ x < β → f(x) = log a ((x-3)/(x+3)))
  (h3 : log a (a * (β-1)) < f(x) ∧ f(x) ≤ log a (a * (α-1))) :
  α > 3 :=
sorry

-- Problem 2: Prove that f(x) is strictly decreasing within its domain
theorem f_decreasing 
  (h1 : 0 < a ∧ a < 1) 
  (h2 : α ≤ x ∧ x < β) :
  ∃ f : ℝ → ℝ, (f(x) = log a ((x - 3) / (x + 3))) → monotone_decreasing f :=
sorry

-- Problem 3: Find the range of the positive number a
theorem range_of_a 
  (h1 : f(x) = log a ((x-3)/(x+3)))
  (h2 : α > 3) 
  (h3 : ∀ x, 0 < a ∧ a < 1) :
  0 < a ∧ a < (2 - Real.sqrt 3) / 4 :=
sorry

end alpha_greater_than_three_f_decreasing_range_of_a_l271_271392


namespace sample_staff_urumqi_l271_271508

theorem sample_staff_urumqi :
  ∀ (total_staff admin_staff teachers staff_logistics sample_size : ℕ),
  total_staff = 160 →
  admin_staff = 16 →
  teachers = 112 →
  staff_logistics = 32 →
  sample_size = 20 →
  ∃ (lottery1 systematic lottery2 : ℕ),
    lottery1 = 2 ∧
    systematic = 14 ∧
    lottery2 = 4 ∧
    (lottery1 + systematic + lottery2 = sample_size) :=
by
  intros total_staff admin_staff teachers staff_logistics sample_size
  intro h1 h2 h3 h4 h5
  use 2, 14, 4
  simp [h5]
  repeat { split; try { refl }; sorry }

end sample_staff_urumqi_l271_271508


namespace find_B_min_of_sum_of_squares_l271_271925

-- Given conditions in a)
variables {A B C a b c : ℝ}
hypothesis (h1 : C = 2 * Real.pi / 3)
hypothesis (h2 : (Real.cos A) / (1 + Real.sin A) = Real.sin (2 * B) / (1 + Real.cos (2 * B)))

-- Part (1) prove B = π / 6
theorem find_B (h1 : C = 2 * Real.pi / 3) (h2 : (Real.cos A) / (1 + Real.sin A) = Real.sin (2 * B) / (1 + Real.cos (2 * B))) : B = Real.pi / 6 :=
by sorry

-- Part (2) find the minimum value of (a^2 + b^2) / (c^2)
theorem min_of_sum_of_squares (h1 : C = 2 * Real.pi / 3) (h2 : (Real.cos A) / (1 + Real.sin A) = Real.sin (2 * B) / (1 + Real.cos (2 * B))) : ∃ (m : ℝ), m = 4 * Real.sqrt 2 - 5 ∧ ∀ a b c, a > 0 ∧ b > 0 ∧ c > 0 → (a^2 + b^2) / c^2 ≥ m :=
by sorry

end find_B_min_of_sum_of_squares_l271_271925


namespace min_eq_floor_sqrt_l271_271164

theorem min_eq_floor_sqrt (n : ℕ) (h : n > 0) : 
  (∀ k : ℕ, k > 0 → (k + n / k) ≥ ⌊(Real.sqrt (4 * n + 1))⌋) := 
sorry

end min_eq_floor_sqrt_l271_271164


namespace probability_divisible_by_4_l271_271305

theorem probability_divisible_by_4 : 
  let x : ℕ := 0; let y : ℕ := 0;
  ∃ (M : ℕ) (n : ℕ), 100 * x + 10 * y + 4 = M ∧ 
  (M % 10 = 4) ∧ (∃ k : ℕ, 4 * k = M) → 
  (@Probability_theory.probability_space.univ (Type* := ℕ) _ _ (λ n, ∃ y, n = 10 * y + 4 ∧ y % 2 = 0) 1/2) :=
begin
  -- proof goes here
  sorry
end

end probability_divisible_by_4_l271_271305


namespace calculate_present_worth_l271_271603

variable (BG : ℝ) (r : ℝ) (t : ℝ)

theorem calculate_present_worth (hBG : BG = 24) (hr : r = 0.10) (ht : t = 2) : 
  ∃ PW : ℝ, PW = 120 := 
by
  sorry

end calculate_present_worth_l271_271603


namespace quadrilateral_coverage_l271_271531

-- Definition of a convex quadrilateral with side lengths no more than 7
structure ConvexQuadrilateral (A B C D : Type) :=
(convex : is_convex A B C D)
(side_lengths_leq_7 : ∀ (X Y : Type), {X, Y} ⊆ {A, B, C, D} → dist X Y ≤ 7)

-- The theorem statement in Lean 4
theorem quadrilateral_coverage 
  {A B C D : Type} 
  (quad : ConvexQuadrilateral A B C D) 
  (O : Type):
  (distance O A ≤ 5) ∨ (distance O B ≤ 5) ∨ (distance O C ≤ 5) ∨ (distance O D ≤ 5) :=
sorry

end quadrilateral_coverage_l271_271531


namespace part1_part2_l271_271012

def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1 (x : ℝ) : 
  ∀ x, f x 2 ≥ 4 ↔ (x ≤ 3 / 2 ∨ x ≥ 11 / 2) :=
by sorry

theorem part2 (x a : ℝ) :
  f x a ≥ 4 → (a ≤ -1 ∨ a ≥ 3) :=
by sorry

end part1_part2_l271_271012


namespace rectangle_width_is_pi_l271_271283

theorem rectangle_width_is_pi (w : ℝ) (h1 : real_w ≠ 0)
    (h2 : ∀ w, ∃ length, length = 2 * w)
    (h3 : ∀ w, 2 * (length + w) = 6 * w)
    (h4 : 2 * (2 * w + w) = 6 * π) : 
    w = π :=
by {
  sorry -- The proof would go here.
}

end rectangle_width_is_pi_l271_271283


namespace distribute_cousins_into_rooms_l271_271937

theorem distribute_cousins_into_rooms : 
  let num_cousins := 5
  let num_rooms := 5
  ∃ finset ℕ, num_ways = 52 :=
by 
  let partitions := [
    (5,0,0,0,0), 
    (4,1,0,0,0), 
    (3,2,0,0,0), 
    (3,1,1,0,0), 
    (2,2,1,0,0), 
    (2,1,1,1,0), 
    (1,1,1,1,1)
  ]
  let num_ways := 
    1 + 
    5 + 
    10 + 
    10 + 
    15 + 
    10 + 
    1
  sorry

end distribute_cousins_into_rooms_l271_271937


namespace marvin_combination_code_count_l271_271934
open Nat

def primes_between (a b : Nat) : List Nat :=
  (List.range' a (b - a + 1)).filter isPrime

def multiples_between (k a b : Nat) : List Nat :=
  (List.range (b + 1)).filter (λ x => x % k = 0 ∧ a ≤ x ∧ x ≤ b)

def even_numbers_between (a b : Nat) : List Nat :=
  (List.range' a (b - a + 1)).filter (λ x => x % 2 = 0)

theorem marvin_combination_code_count :
  let primes := primes_between 1 25 
  let multiples_of_4 := multiples_between 4 1 20 
  let multiples_of_5 := multiples_between 5 1 30 
  let evens := even_numbers_between 1 12 
  primes.length * multiples_of_4.length * multiples_of_5.length * evens.length = 1620 :=
by
  let primes := primes_between 1 25 
  let multiples_of_4 := multiples_between 4 1 20 
  let multiples_of_5 := multiples_between 5 1 30 
  let evens := even_numbers_between 1 12 
  have h_primes : primes.length = 9 := sorry
  have h_multiples_of_4 : multiples_of_4.length = 5 := sorry
  have h_multiples_of_5 : multiples_of_5.length = 6 := sorry
  have h_evens : evens.length = 6 := sorry
  calc
    primes.length * multiples_of_4.length * multiples_of_5.length * evens.length
        = 9 * 5 * 6 * 6 : by rw [h_primes, h_multiples_of_4, h_multiples_of_5, h_evens]
    ... = 1620 : by norm_num

end marvin_combination_code_count_l271_271934


namespace part1_part2_l271_271894

open Set Real

-- Definitions of sets A, B, and C
def setA : Set ℝ := { x | 2 ≤ x ∧ x < 5 }
def setB : Set ℝ := { x | 1 < x ∧ x < 8 }
def setC (a : ℝ) : Set ℝ := { x | x < a - 1 ∨ x > a }

-- Conditions:
-- - Complement of A
def complementA : Set ℝ := { x | x < 2 ∨ x ≥ 5 }

-- Question parts:
-- (1) Finding intersection of complementA and B
theorem part1 : (complementA ∩ setB) = { x | (1 < x ∧ x < 2) ∨ (5 ≤ x ∧ x < 8) } := sorry

-- (2) Finding range of a for specific condition on C
theorem part2 (a : ℝ) : (setA ∪ setC a = univ) → (a ≤ 2 ∨ a > 6) := sorry

end part1_part2_l271_271894


namespace conjugate_of_z_l271_271473

noncomputable def i : ℂ := complex.I

/-- Given conditions -/
def z : ℂ := (2 - i) / (1 + i^2 - i^5)

/-- Prove that the conjugate of z equals 1 - 2i -/
theorem conjugate_of_z : complex.conj z = 1 - 2 * i :=
begin
  sorry
end

end conjugate_of_z_l271_271473


namespace padic_zeros_l271_271123

variable {p : ℕ} (hp : p > 1)
variable {a : ℕ} (hnz : a % p ≠ 0)

theorem padic_zeros (k : ℕ) (hk : k ≥ 1) :
  (a^(p^(k-1)*(p-1)) - 1) % (p^k) = 0 :=
sorry

end padic_zeros_l271_271123


namespace ab_gt_ba_l271_271048

theorem ab_gt_ba (a b : ℝ) (h₀ : 0 < b) (h₁ : b < a) (h₂ : a < Real.exp 1) : a ^ b > b ^ a :=
by
  let f := λ x : ℝ, Real.log x / x
  have h₃ : ∀ x ∈ Ioo (0 : ℝ) (Real.exp 1), (1 - Real.log x) / x^2 > 0,
  { intros x hx,
    have x_pos : 0 < x := hx.1,
    have x_lt_e : x < Real.exp 1 := hx.2,
    let log_x := Real.log x,
    have h₄ : log_x < 1, from Real.log_lt_self x_pos x_lt_e,
    have h₅ : 1 > log_x, from h₄,
    have h₆ : 1 - log_x > 0, from sub_pos_of_lt h₅,
    have h₇ : x^2 > 0, from sq_pos_of_pos x_pos,
    exact div_pos h₆ h₇ },
  have f_incr : ∀ x y ∈ Ioo (0 : ℝ) (Real.exp 1), x < y → f x < f y,
  { intros x hx y hy h₈,
    exact Real.Ioox_lt_log_x_div_x_monotonic hx hy h₈ h₃ },
  have ha_pos : a ∈ Ioo (0 : ℝ) (Real.exp 1), from ⟨h₀.trans h₁, h₂⟩,
  have hb_pos : b ∈ Ioo (0 : ℝ) (Real.exp 1), from ⟨h₀, h₁⟩,
  have f_lt : f b < f a, from f_incr b hb_pos a ha_pos h₁,
  have f_ba := Real.log_b_over_b_pos b h₀ h₁,
  have f_ab := Real.log_a_over_a_pos a h₀ h₂,
  have f_neq : f a ≠ f b := (ne_of_lt f_lt).symm,
  linarith [f_ab, f_ba]

end ab_gt_ba_l271_271048


namespace part1_solution_set_part2_range_of_a_l271_271000

-- Define the function f for a general a
def f (x a : ℝ) : ℝ := |x - a^2| + |x - (2 * a) + 1|

-- Part 1: When a = 2
theorem part1_solution_set (x : ℝ) (h : f x 2 ≥ 4) : x ≤ 3/2 ∨ x ≥ 11/2 :=
  sorry

-- Part 2: Range of values for a
theorem part2_range_of_a (a : ℝ) (h : ∀ x, f x a ≥ 4) : a ≤ -1 ∨ a ≥ 3 :=
  sorry

end part1_solution_set_part2_range_of_a_l271_271000


namespace perfect_square_conditions_l271_271688

theorem perfect_square_conditions (k : ℕ) : 
  (∃ m : ℤ, k^2 - 101 * k = m^2) ↔ (k = 101 ∨ k = 2601) := 
by 
  sorry

end perfect_square_conditions_l271_271688


namespace triangle_divide_similar_l271_271828

theorem triangle_divide_similar (T : Type) [IsTriangle T] (divides_into_similar_triangle : ∃ (T1 T2 : T), T1 ∼ T ∧ T2 ∼ T) : IsRightTriangle T :=
sorry

end triangle_divide_similar_l271_271828


namespace case_a_case_b_case_c_l271_271359

-- Definitions of game manageable
inductive Player
| First
| Second

def sum_of_dimensions (m n : Nat) : Nat := m + n

def is_winning_position (m n : Nat) : Player :=
  if sum_of_dimensions m n % 2 = 1 then Player.First else Player.Second

-- Theorem statements for the given grid sizes
theorem case_a : is_winning_position 9 10 = Player.First := 
  sorry

theorem case_b : is_winning_position 10 12 = Player.Second := 
  sorry

theorem case_c : is_winning_position 9 11 = Player.Second := 
  sorry

end case_a_case_b_case_c_l271_271359


namespace card_draw_probability_l271_271846

-- Define the problem within Lean
theorem card_draw_probability : 
  let cards := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  in (1⁄10 : ℚ) =
    (∑' 
  (x y : ℕ) in cards ×' cards, if (x + y) % 10 == 0 then 1 else 0) / 
    (∑' (x y : ℕ) in cards × cards, 1) :=
sorry

end card_draw_probability_l271_271846


namespace volume_of_solid_correct_l271_271207

-- Define assumptions and conditions
variables {u : ℝ × ℝ × ℝ}
def dot_prod (a b : ℝ × ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2 + a.3 * b.3

-- Condition expressed as an equation
def condition (u : ℝ × ℝ × ℝ) : Prop :=
  dot_prod u u = dot_prod u (8, -28, 12)

-- Define the volume of the solid formed.
def volume_of_solid : ℝ :=
  (4 / 3) * Real.pi * 248^(3/2)

theorem volume_of_solid_correct :
  (∃ u : ℝ × ℝ × ℝ, condition u) → (volume_of_solid = (4 / 3) * Real.pi * 248^(3/2)) :=
by
  -- Proof goes here.
  sorry

end volume_of_solid_correct_l271_271207


namespace largest_angle_of_consecutive_interior_angles_pentagon_l271_271197

theorem largest_angle_of_consecutive_interior_angles_pentagon (x : ℕ)
  (h1 : (x - 3) + (x - 2) + (x - 1) + x + (x + 1) = 540) :
  x + 1 = 110 := sorry

end largest_angle_of_consecutive_interior_angles_pentagon_l271_271197


namespace maximum_knights_and_courtiers_l271_271454

theorem maximum_knights_and_courtiers :
  ∃ (a b : ℕ), 12 ≤ a ∧ a ≤ 18 ∧ 10 ≤ b ∧ b ≤ 20 ∧ (1 / a + 1 / b = 1 / 7) ∧
               (∀ (a' b' : ℕ), 12 ≤ a' ∧ a' ≤ 18 ∧ 10 ≤ b' ∧ b' ≤ 20 ∧ (1 / a' + 1 / b' = 1 / 7) → b ≤ b') → 
  a = 14 ∧ b = 14 :=
by
  use 14, 14
  split
  repeat { sorry }

end maximum_knights_and_courtiers_l271_271454


namespace possible_values_of_m_l271_271538

noncomputable def polynomial_integral_roots (b c d e : ℤ) (h_d : d ≠ 0) : set ℕ :=
  {m | ∃ (f : polynomial ℤ), f = X^4 + C b * X^3 + C c * X^2 + (7 * C d) * X + C e ∧
    m = (roots f).count_map (λ r, if r ∈ ℤ then 1 else 0)}

theorem possible_values_of_m (b c d e : ℤ) (h_d : d ≠ 0) :
  polynomial_integral_roots b c d e h_d ⊆ {0, 1, 2, 4} := sorry

end possible_values_of_m_l271_271538


namespace tangent_line_at_1_l271_271800

noncomputable def f (x : ℝ) (a b : ℝ) := x^3 - a * x + b

theorem tangent_line_at_1 (a b : ℝ) :
  let f_val := f 1 a b
  let f_deriv := 3 * 1^2 - a
    in
  f_deriv = 2 ∧ f_val = 5 → 
  f 1 1 5 = x^3 - x + 5 :=
by
  sorry

end tangent_line_at_1_l271_271800


namespace fraction_finding_l271_271064

theorem fraction_finding :
  ∃ F : ℝ, 0.40 * F * 150 = 36 ∧ F = 0.6 :=
by
  use 0.6
  split
  sorry

end fraction_finding_l271_271064


namespace number_of_bricks_required_l271_271593

def courtyard_length_m : ℕ := 25
def courtyard_width_m : ℕ := 16
def brick_length_cm : ℕ := 20
def brick_width_cm : ℕ := 10

theorem number_of_bricks_required :
  let courtyard_length_cm := courtyard_length_m * 100 in
  let courtyard_width_cm := courtyard_width_m * 100 in
  let courtyard_area_cm2 := courtyard_length_cm * courtyard_width_cm in
  let brick_area_cm2 := brick_length_cm * brick_width_cm in
  courtyard_area_cm2 / brick_area_cm2 = 20000 :=
by
  sorry

end number_of_bricks_required_l271_271593


namespace trajectory_is_eight_rays_l271_271542

open Real

def trajectory_of_point (x y : ℝ) : Prop :=
  abs (abs x - abs y) = 2

theorem trajectory_is_eight_rays :
  ∃ (x y : ℝ), trajectory_of_point x y :=
sorry

end trajectory_is_eight_rays_l271_271542


namespace derivative_of_f_l271_271338

noncomputable def f (x : ℝ) : ℝ :=
  (1 / Real.sqrt 2) * Real.arctan ((3 * x - 1) / Real.sqrt 2) + (1 / 3) * ((3 * x - 1) / (3 * x^2 - 2 * x + 1))

theorem derivative_of_f :
  ∀ x : ℝ, deriv f x = 4 / (3 * (3 * x^2 - 2 * x + 1)^2) :=
by intros; sorry

end derivative_of_f_l271_271338


namespace max_value_of_f_period_of_f_value_of_a_div_c_l271_271481

def f (x : ℝ) : ℝ := 6 * (Real.cos x) ^ 2 - Real.sqrt 3 * (Real.sin (2 * x))

theorem max_value_of_f : ∃ x : ℝ, f x = 2 * Real.sqrt 3 + 3 :=
sorry

theorem period_of_f : ∃ T : ℝ, T = π ∧ ∀ x, f (x + T) = f x :=
sorry

def in_triangle_AC_ratio (A B : ℝ) (fA B) (a c : ℝ) :=
  (f A = 3 - 2 * Real.sqrt 3) ∧ (B = π / 12) ∧ ∃ C : ℝ, (A + B + C = π)

theorem value_of_a_div_c (A B : ℝ) (a c : ℝ) (h1 : f A = 3 - 2 * Real.sqrt 3) (h2 : B = π / 12) :
  in_triangle_AC_ratio A B h1 h2 a c → a / c = (Real.sqrt 6 + Real.sqrt 2) / 4 :=
sorry

end max_value_of_f_period_of_f_value_of_a_div_c_l271_271481


namespace probability_distance_ge_sqrt2_div_2_l271_271118

theorem probability_distance_ge_sqrt2_div_2 (S : set (ℝ × ℝ)) (p1 p2 : ℝ × ℝ):
  (S = { (x, y) | 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 }) →
  p1 ∉ S ∨ p2 ∉ S ∨ (p1 = p2) →
  (∃ a b c : ℕ, gcd (gcd a b) c = 1 ∧ (36 - (1 : ℝ) * real.pi) / 16 =
    ((a : ℤ) - b * real.pi) / c ∧ a + b + c = 53) :=
by
  sorry

end probability_distance_ge_sqrt2_div_2_l271_271118


namespace mode_and_median_of_data_set_l271_271433

-- Conditions: The number of discarded plastic bags picked up by 6 students.
def data_set : List ℕ := [5, 4, 6, 8, 7, 7]

-- Questions: Proving mode and median are correct
theorem mode_and_median_of_data_set :
  (mode data_set = 7) ∧ (median data_set = 6.5) :=
by
  sorry -- Proof goes here

end mode_and_median_of_data_set_l271_271433


namespace point_on_y_axis_m_value_l271_271445

theorem point_on_y_axis_m_value (m : ℝ) (h : 6 - 2 * m = 0) : m = 3 := by
  sorry

end point_on_y_axis_m_value_l271_271445


namespace jasmine_laps_l271_271100

/-- Jasmine swims 12 laps every afternoon, Monday through Friday. Calculate the total number of laps she swims in five weeks. -/
theorem jasmine_laps : 
    ∀ (laps_per_day days_per_week number_of_weeks : ℕ),
        laps_per_day = 12 ∧ days_per_week = 5 ∧ number_of_weeks = 5 → 
        number_of_weeks * (days_per_week * laps_per_day) = 300 :=
by
    intros laps_per_day days_per_week number_of_weeks
    intro h
    cases h with hlaps h_days_weeks
    cases h_days_weeks with hdays hweeks
    rw [hlaps, hdays, hweeks]
    norm_num
    sorry

end jasmine_laps_l271_271100


namespace find_upper_base_length_l271_271756

-- Define the trapezoid and its properties.
variables (d : ℝ)
variables (A D : ℝ × ℝ) (M : ℝ × ℝ) (C : ℝ × ℝ)

-- Define the conditions of the problem.
def is_trapezoid (A B C D : ℝ × ℝ) : Prop := 
  ∃ M, M.1 = (A.1 + B.1) / 2 ∧ M.1 = (C.1 + D.1) / 2
  
def perpendicular (P Q : ℝ × ℝ) : Prop :=
  P.1 = Q.1

def equal_distance (P Q R : ℝ × ℝ) : Prop :=
  dist P Q = dist Q R

-- Setting the exact locations of points
def coordinates : Prop := 
  D = (0, 0) ∧ A = (d, 0) ∧ perpendicular (M) (D, A)

-- Required proof
theorem find_upper_base_length :
  coordinates d A D M ∧ equal_distance M C D → 
  dist (A, C) = d / 2 :=
by sorry

end find_upper_base_length_l271_271756


namespace sticks_form_equilateral_triangle_l271_271690

theorem sticks_form_equilateral_triangle (n : ℕ) (h1 : n ≥ 5) (h2 : n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5) :
  ∃ k, k * 3 = (n * (n + 1)) / 2 :=
by
  sorry

end sticks_form_equilateral_triangle_l271_271690


namespace no_integer_roots_if_coefficients_are_odd_l271_271950

theorem no_integer_roots_if_coefficients_are_odd (a b c x : ℤ) 
  (h1 : Odd a) (h2 : Odd b) (h3 : Odd c) (h4 : a * x^2 + b * x + c = 0) : False := 
by
  sorry

end no_integer_roots_if_coefficients_are_odd_l271_271950


namespace find_B_min_fraction_of_squares_l271_271913

-- Lean 4 statement for part (1)
theorem find_B (A B C a b c : ℝ) (h_triangle : A + B + C = π) 
(h_sides : a > 0 ∧ b > 0 ∧ c > 0) 
(h_identity : (cos A) / (1 + sin A) = (sin (2 * B)) / (1 + cos (2 * B))) 
(h_C : C = 2 * π / 3) : 
B = π / 6 := sorry

-- Lean 4 statement for part (2)
theorem min_fraction_of_squares (A B C a b c : ℝ) 
(h_triangle : A + B + C = π) 
(h_sides : a > 0 ∧ b > 0 ∧ c > 0) 
(h_identity : (cos A) / (1 + sin A) = (sin (2 * B)) / (1 + cos (2 * B))) 
(h_C : C = 2 * π / 3) : 
∀ a b c, ∃ m, m = 4 * sqrt 2 - 5 ∧ (a^2 + b^2) / c^2 = m := sorry

end find_B_min_fraction_of_squares_l271_271913


namespace conditions_necessary_sufficient_l271_271370

variables (p q r s : Prop)

theorem conditions_necessary_sufficient :
  ((p → r) ∧ (¬ (r → p)) ∧ (q → r) ∧ (s → r) ∧ (q → s)) →
  ((s ↔ q) ∧ ((p → q) ∧ ¬ (q → p)) ∧ ((¬ p → ¬ s) ∧ ¬ (¬ s → ¬ p))) := by
  sorry

end conditions_necessary_sufficient_l271_271370


namespace g_25_eq_zero_l271_271527

noncomputable def g : ℝ → ℝ := sorry

axiom g_def (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : x^2 * g y - y^2 * g x = g (x^2 / y^2)

theorem g_25_eq_zero : g 25 = 0 := by
  sorry

end g_25_eq_zero_l271_271527


namespace area_G1G2G3_l271_271114

variable {P A B C G1 G2 G3 : Type} [MetricSpace A B C P]

-- Define the point P inside the triangle ABC and the centroids G1, G2, and G3
-- Define the areas of triangles
noncomputable def area (t : Type) [metric_space t] : ℝ := sorry

-- Assume the conditions
variable [is_triangle A B C]
variable (P : A)
variable [is_inside_triangle P A B C]
variable (G1 : A) [is_centroid G1 P B C]
variable (G2 : A) [is_centroid G2 P C A]
variable (G3 : A) [is_centroid G3 P A B]

-- Given the area of triangle ABC is 24
axiom h1 : area (triangle A B C) = 24

-- Define the area of triangle G1 G2 G3
theorem area_G1G2G3 : area (triangle G1 G2 G3) = 2.67 := by
  sorry

end area_G1G2G3_l271_271114


namespace chord_sphere_tangent_squared_l271_271208

-- Defining the mathematical objects and properties
variables {R r : ℝ} -- Radii of spheres S and s respectively
variables (S s : Type*) [metric_space S] [metric_space s]
variables [sphere S R] [sphere s r]
variables {A B C O: S} -- Points on the sphere S
variables (passes_through_center : center s = O)
variables (C_on_AB : tangent_at_point s C (line_segment A B))

-- The proposition to be proven
theorem chord_sphere_tangent_squared (passes_through_center : center s = center S) 
  (tangent_C_AB : tangent_at_point s C (line_segment A B)) :
  distance A C ^ 2 + distance B C ^ 2 ≤ 2 * R ^ 2 + r ^ 2 := 
sorry

end chord_sphere_tangent_squared_l271_271208


namespace monotonicity_and_inequality_l271_271798

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x^2

theorem monotonicity_and_inequality (a : ℝ) (p q : ℝ) (hp : 0 < p ∧ p < 1) (hq : 0 < q ∧ q < 1)
  (h_distinct: p ≠ q) (h_a : a ≥ 10) : 
  (f a (p + 1) - f a (q + 1)) / (p - q) > 1 := by
  sorry

end monotonicity_and_inequality_l271_271798


namespace coin_flips_probability_ratio_l271_271149

theorem coin_flips_probability_ratio (n : ℕ) : 
  let factorial := λ k, Nat.factorial k
  let binomial := λ n k, factorial n / (factorial k * factorial (n - k))
  let prob := λ k, binomial (2 * n) k / 2^ (2 * n) 
  let joint_prob := (prob n)^2
  let cond_prob_equal_heads := binomial (4 * n) (2 * n) / 2^(4 * n)
  let prob_Mitya_equal_heads := joint_prob / cond_prob_equal_heads
  let ratio := prob_Mitya_equal_heads / (prob n)
  ratio = Real.sqrt 2 := 
  sorry

end coin_flips_probability_ratio_l271_271149


namespace evaluate_ceilings_l271_271325

theorem evaluate_ceilings : (⌈Real.sqrt (9 / 4)⌉ : ℤ) + ⌈(9 / 4)⌉ + ⌈(9 / 4) ^ 2⌉ = 11 :=
by
  -- Definitions pulled from condition in a):
  have h₁ : (⌈Real.sqrt (9 / 4)⌉ : ℤ) = 2 := sorry,
  have h₂ : (⌈(9 / 4)⌉ : ℤ) = 3 := sorry,
  have h₃ : (⌈(9 / 4 : ℝ) ^ 2⌉ : ℤ) = 6 := sorry,
  rw [h₁, h₂, h₃],
  norm_num,
  exact eq.refl 11

end evaluate_ceilings_l271_271325


namespace reciprocal_in_third_quadrant_l271_271188

variable (a b : ℝ)
-- Given conditions
def condition1 : Prop := a > 0
def condition2 : Prop := b > 0
def condition3 : Prop := a^2 + b^2 > 1
def F : ℂ := -a + b * complex.I

-- Lean statement to prove the locating of the reciprocal F lies in the third quadrant
theorem reciprocal_in_third_quadrant : 
  condition1 a → condition2 b → condition3 a b → 
  let reciprocal := 1 / F a b in
  ∃ (D : ℂ), D = reciprocal ∧ (D.re < 0) ∧ (D.im < 0) := 
by {
  intros,
  let reciprocal := 1 / F a b,
  use reciprocal,
  split,
  simp,
  split,
  sorry,
  sorry
}

end reciprocal_in_third_quadrant_l271_271188


namespace volume_of_polyhedron_l271_271085

def is_rectangle (a b : ℝ) : Prop := a * b ≠ 0

def is_regular_hexagon (s : ℝ) : Prop := s > 0

theorem volume_of_polyhedron :
  (∀ (H I J : ℝ × ℝ), is_rectangle H.1 H.2 ∧ H = (2, 1) ∧
                      is_rectangle I.1 I.2 ∧ I = (2, 1) ∧
                      is_rectangle J.1 J.2 ∧ J = (2, 1)) →
  (∀ (K L M : ℝ × ℝ), is_rectangle K.1 K.2 ∧ K = (1, 2) ∧
                      is_rectangle L.1 L.2 ∧ L = (1, 2) ∧
                      is_rectangle M.1 M.2 ∧ M = (1, 2)) →
  (∃ (N : ℝ), is_regular_hexagon N ∧ N = 1) →
  ∃ (V : ℝ), V = 3 * Real.sqrt 3 :=
begin
  intros,
  existsi (3 * Real.sqrt 3),
  sorry
end

end volume_of_polyhedron_l271_271085


namespace find_positive_integers_l271_271334

theorem find_positive_integers (n : ℕ) (h_pos : n > 0) : 
  (n^4 - 4 * n^3 + 22 * n^2 - 36 * n + 18 ∈ (Set.range (λ m : ℕ, m * m))) ↔ (n = 1 ∨ n = 3) :=
by
  sorry

end find_positive_integers_l271_271334


namespace num_mutually_exclusive_but_not_contradictory_pairs_l271_271653

namespace Archer

-- Definitions of events 
def E1 : Prop := "miss"
def E2 : Prop := "hit the target"
def E3 : Prop := "the number of rings hit is greater than 4"
def E4 : Prop := "the number of rings hit is at least 5"

-- Predicate for mutually exclusive but not contradictory events
def mutually_exclusive_but_not_contradictory (A B : Prop) : Prop :=
  (¬ (A ∧ B)) ∧ ¬ (A ↔ ¬ B)

-- The main theorem
theorem num_mutually_exclusive_but_not_contradictory_pairs :
  ((mutually_exclusive_but_not_contradictory E1 E3) ∨ 
   (mutually_exclusive_but_not_contradictory E1 E4) ∨ 
   (mutually_exclusive_but_not_contradictory E2 E3) ∨ 
   (mutually_exclusive_but_not_contradictory E2 E4) ∨ 
   (mutually_exclusive_but_not_contradictory E3 E4) ∨ 
   (mutually_exclusive_but_not_contradictory E1 E2)) = 2 :=
sorry

end Archer

end num_mutually_exclusive_but_not_contradictory_pairs_l271_271653


namespace find_B_min_value_a2_b2_c2_l271_271896

theorem find_B (A B C : ℝ) (h1 : C = 2 * π / 3) 
  (h2 : (cos A) / (1 + sin A) = (sin (2 * B)) / (1 + cos (2 * B))) : B = π / 6 :=
sorry

theorem min_value_a2_b2_c2 (A B C a b c : ℝ) (h1 : C = 2 * π / 3)
  (h2 : (cos A) / (1 + sin A) = (sin (2 * B)) / (1 + cos (2 * B))) 
  (h3 : triangles_side_identity a b c A B C AE BA CE) :
  (a^2 + b^2) / c^2 = 4 * Real.sqrt 2 - 5 :=
sorry

end find_B_min_value_a2_b2_c2_l271_271896


namespace percentage_increase_proof_l271_271108

def breakfast_calories : ℕ := 500
def shakes_total_calories : ℕ := 3 * 300
def total_daily_calories : ℕ := 3275

noncomputable def percentage_increase_in_calories (P : ℝ) : Prop :=
  let lunch_calories := breakfast_calories * (1 + P / 100)
  let dinner_calories := 2 * lunch_calories
  breakfast_calories + lunch_calories + dinner_calories + shakes_total_calories = total_daily_calories

theorem percentage_increase_proof : percentage_increase_in_calories 125 :=
by
  sorry

end percentage_increase_proof_l271_271108


namespace boxes_filled_l271_271687

theorem boxes_filled (boxes numbers : Finset ℕ) (h₁ : boxes = {1, 2, 3, 4}) (h₂ : numbers = {1, 2, 3, 4}) :
    let valid_fills := {fill : Fin 4 → Fin 4 // ∀ i : Fin 4, fill i ≠ i}
    |valid_fills.to_finset.card = 9 := 
by
  sorry

end boxes_filled_l271_271687


namespace x_minus_y_value_l271_271417

theorem x_minus_y_value (x y : ℝ) (h1 : x^2 = 4) (h2 : |y| = 3) (h3 : x + y < 0) : x - y = 1 ∨ x - y = 5 := by
  sorry

end x_minus_y_value_l271_271417


namespace measure_of_angle_A_l271_271568

-- Defining the measures of angles
def angle_B : ℝ := 50
def angle_C : ℝ := 40
def angle_D : ℝ := 30

-- Prove that measure of angle A is 120 degrees given the conditions
theorem measure_of_angle_A (B C D : ℝ) (hB : B = angle_B) (hC : C = angle_C) (hD : D = angle_D) : B + C + D + 60 = 180 -> 180 - (B + C + D + 60) = 120 :=
by sorry

end measure_of_angle_A_l271_271568


namespace coin_flips_probability_ratio_l271_271150

theorem coin_flips_probability_ratio (n : ℕ) : 
  let factorial := λ k, Nat.factorial k
  let binomial := λ n k, factorial n / (factorial k * factorial (n - k))
  let prob := λ k, binomial (2 * n) k / 2^ (2 * n) 
  let joint_prob := (prob n)^2
  let cond_prob_equal_heads := binomial (4 * n) (2 * n) / 2^(4 * n)
  let prob_Mitya_equal_heads := joint_prob / cond_prob_equal_heads
  let ratio := prob_Mitya_equal_heads / (prob n)
  ratio = Real.sqrt 2 := 
  sorry

end coin_flips_probability_ratio_l271_271150


namespace find_m_n_l271_271583

theorem find_m_n : ∃ (m n : ℕ), 2^n + 1 = m^2 ∧ m = 3 ∧ n = 3 :=
by {
  sorry
}

end find_m_n_l271_271583


namespace range_of_t_l271_271835

variable (t : ℝ)

def point_below_line (x y a b c : ℝ) : Prop :=
  a * x - b * y + c < 0

theorem range_of_t (t : ℝ) : point_below_line 2 (3 * t) 2 (-1) 6 → t < 10 / 3 :=
  sorry

end range_of_t_l271_271835


namespace length_BF_in_equilateral_triangle_is_six_l271_271439

theorem length_BF_in_equilateral_triangle_is_six :
  ∀ (A B C D E F : Type)
  (length : A → B → ℝ)
  (midpoint : B → B → B)
  (area : A → A → A → ℝ)
  (side_length : ℝ),
  (∀ (a b c : A), a ≠ b ∧ b ≠ c ∧ a ≠ c → length a b = side_length) →
  (∀ (b c : A), length b c = side_length → midpoint b c = D) →
  (∀ (a c : A), midpoint a c = E) →
  (∀ (b c f : A), area a b f = area a b d + area a d e) →
  (midpoint c, d = f → length b f = 6) :=
sorry

end length_BF_in_equilateral_triangle_is_six_l271_271439


namespace part1_solution_set_part2_values_of_a_l271_271029

-- Parameters and definitions for the function and conditions
def f (x a : ℝ) := |x - a^2| + |x - (2*a + 1)|

-- Part (1) When a = 2, the inequality's solution set
theorem part1_solution_set (x : ℝ) : f x 2 ≥ 4 ↔ (x ≤ 3/2 ∨ x ≥ 11/2) := 
sorry

-- Part (2) Range of values for a given f(x, a) ≥ 4
theorem part2_values_of_a (a : ℝ) : 
∀ x, f x a ≥ 4 → (a ≤ -1 ∨ a ≥ 3) :=
sorry

end part1_solution_set_part2_values_of_a_l271_271029


namespace find_a_and_extreme_values_l271_271375

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x^2 + a * Real.log x

theorem find_a_and_extreme_values :
  (∀ a, f' (2) = 0 → a = -8) ∧
  (f 2 (-8) = 4 - 8 * Real.log 2) ∧
  (∀ x ∈ Set.Icc (1 / Real.exp 1) Real.exp 1,
    IsLocalMin (f x) (4 - 8 * Real.log 2)
    ∧ IsLocalMax (f x) (1 / (Real.exp 2) + 8)) :=
by
  sorry

end find_a_and_extreme_values_l271_271375


namespace houses_count_l271_271144

theorem houses_count (n : ℕ) 
  (h1 : ∃ k : ℕ, k + 7 = 12)
  (h2 : ∃ m : ℕ, m + 25 = 30) :
  n = 32 :=
sorry

end houses_count_l271_271144


namespace group_size_l271_271516

theorem group_size:
  ∀ (n : ℕ),
  (60 - 40) = 2.5 * ↑n →
  n = 8 :=
by
  intros n h
  have h1 : 20 = 2.5 * ↑n := h
  sorry

end group_size_l271_271516


namespace parity_E_2023_2024_2025_l271_271636

def E : ℕ → ℕ
| 0 := 1
| 1 := 1
| 2 := 0
| (n+3) := E n + E (n+1) + E (n+2)

def parity (n : ℕ) : Prop := 
  (E n % 2 = 0 ∧ E (n + 1) % 2 = 0 ∧ E (n + 2) % 2 = 0)

theorem parity_E_2023_2024_2025 : parity 2023 :=
by {
  -- initialize sequence conditions and prove here
  sorry 
}

end parity_E_2023_2024_2025_l271_271636


namespace Anya_more_than_Vitya_l271_271277

/-- Definition of Anya's sequence: Arithmetic sequence where the first term is 1 and the common difference is 3 -/
def Anya_sequence (n : ℕ) : ℕ := 1 + (n - 1) * 3

/-- Definition of Vitya's sequence: Arithmetic sequence where the first term is 3 and the common difference is 3 -/
def Vitya_sequence (n : ℕ) : ℕ := 3 + (n - 1) * 3

/-- Theorem statement to prove that Anya receives 68 rubles more than Vitya by the end of the sequences -/
theorem Anya_more_than_Vitya (N : ℕ) (hAnya : Anya_sequence 68 = 202) (hVitya : 3 + (N - 1) * 3 = 204 - 68 * 1) :
  (∑ k in Finset.range 68, Anya_sequence k) - (∑ k in Finset.range 68, Vitya_sequence k) = 68 :=
by
  sorry

end Anya_more_than_Vitya_l271_271277


namespace quadratic_symmetry_axis_l271_271648

/-- 
  This theorem states that the quadratic function y = (x + 1)^2 
  has a symmetry axis at the line x = -1.
--/
theorem quadratic_symmetry_axis (x : ℝ) :
  let y := (x + 1) ^ 2
  in ∃ h : ℝ, h = -1 ∧ (∀ x1 x2 : ℝ, y x1 = y x2 → x1 = -1 ∨ x2 = -1) :=
sorry

end quadratic_symmetry_axis_l271_271648


namespace homothety_composition_proof_homothety_composition_translation_l271_271166

noncomputable def homothety_composition (k₁ k₂ : ℝ) (O₁ O₂ : ℝ × ℝ) (H₁ H₂ : (ℝ × ℝ) → (ℝ × ℝ)) :=
  H₂ ∘ H₁

theorem homothety_composition_proof
  (k₁ k₂ : ℝ) (O₁ O₂ : ℝ × ℝ)
  (H₁ : (ℝ × ℝ) → (ℝ × ℝ)) (H₂ : (ℝ × ℝ) → (ℝ × ℝ))
  (h₁ : ∀ A : ℝ × ℝ, H₁ A = O₁ + k₁ • (A - O₁))
  (h₂ : ∀ A : ℝ × ℝ, H₂ A = O₂ + k₂ • (A - O₂))
  (hk : k₁ * k₂ ≠ 1) :
  ∃ K : ℝ, K = k₁ * k₂ ∧
    (∃ C : ℝ × ℝ, ∀ P : ℝ × ℝ, (H₂ ∘ H₁) P = C + K • (P - C)) ∧
    (C - O₁ = (k₂ / (1 - k₁ * k₂)) • (O₂ - O₁)) :=
begin
  sorry
end

theorem homothety_composition_translation
  (k₁ k₂ : ℝ) (O₁ O₂ : ℝ × ℝ)
  (H₁ : (ℝ × ℝ) → (ℝ × ℝ)) (H₂ : (ℝ × ℝ) → (ℝ × ℝ))
  (h₁ : ∀ A : ℝ × ℝ, H₁ A = O₁ + k₁ • (A - O₁))
  (h₂ : ∀ A : ℝ × ℝ, H₂ A = O₂ + k₂ • (A - O₂))
  (hk : k₁ * k₂ = 1) :
  ∃ T : ℝ × ℝ, ∀ P : ℝ × ℝ, (H₂ ∘ H₁) P = P + T :=
begin
  sorry
end

end homothety_composition_proof_homothety_composition_translation_l271_271166


namespace maximum_knights_and_courtiers_l271_271453

theorem maximum_knights_and_courtiers :
  ∃ (a b : ℕ), 12 ≤ a ∧ a ≤ 18 ∧ 10 ≤ b ∧ b ≤ 20 ∧ (1 / a + 1 / b = 1 / 7) ∧
               (∀ (a' b' : ℕ), 12 ≤ a' ∧ a' ≤ 18 ∧ 10 ≤ b' ∧ b' ≤ 20 ∧ (1 / a' + 1 / b' = 1 / 7) → b ≤ b') → 
  a = 14 ∧ b = 14 :=
by
  use 14, 14
  split
  repeat { sorry }

end maximum_knights_and_courtiers_l271_271453


namespace jasmine_swims_laps_l271_271102

theorem jasmine_swims_laps (laps_per_day : ℕ) (days_per_week : ℕ) (weeks : ℕ) : 
  (laps_per_day = 12) → 
  (days_per_week = 5) → 
  (weeks = 5) → 
  laps_per_day * days_per_week * weeks = 300 := 
by 
  intros h1 h2 h3
  rw [h1, h2, h3]
  exact rfl

end jasmine_swims_laps_l271_271102


namespace speed_of_current_l271_271602
  
  theorem speed_of_current (v c : ℝ)
    (h1 : 64 = (v + c) * 8)
    (h2 : 24 = (v - c) * 8) :
    c = 2.5 :=
  by {
    sorry
  }
  
end speed_of_current_l271_271602


namespace sets_relation_l271_271775

def is_integer (x : ℚ) : Prop := ∃ n : ℤ, x = n

def M : Set ℚ := {x | ∃ (m : ℤ), x = m + 1/6}
def S : Set ℚ := {x | ∃ (s : ℤ), x = s/2 - 1/3}
def P : Set ℚ := {x | ∃ (p : ℤ), x = p/2 + 1/6}

theorem sets_relation : M ⊆ S ∧ S = P := by
  sorry

end sets_relation_l271_271775


namespace students_like_both_soda_and_coke_l271_271813

theorem students_like_both_soda_and_coke
  (T S C N : ℕ)
  (hT : T = 500)
  (hS : S = 337)
  (hC : C = 289)
  (hN : N = 56) :
  (S + C - T + N) = 182 :=
by
  rw [hT, hS, hC, hN]
  -- It simplifies directly to the answer
  by simp
  sorry

end students_like_both_soda_and_coke_l271_271813


namespace determine_initial_fund_l271_271972

def initial_amount_fund (n : ℕ) := 60 * n + 30 - 10

theorem determine_initial_fund (n : ℕ) (h : 50 * n + 110 = 60 * n - 10) : initial_amount_fund n = 740 :=
by
  -- we skip the proof steps here
  sorry

end determine_initial_fund_l271_271972


namespace concurrency_of_incircle_reflections_l271_271966

theorem concurrency_of_incircle_reflections
  {ABC : Triangle}
  {O : Point}
  {A1 B1 C1 A2 B2 C2 : Point}
  (hO : is_incenter O ABC)
  (hA1 : is_incircle_touching A1 ABC BC)
  (hB1 : is_incircle_touching B1 ABC AC)
  (hC1 : is_incircle_touching C1 ABC AB)
  (hA2 : intersection AO incircle = A2)
  (hB2 : intersection BO incircle = B2)
  (hC2 : intersection CO incircle = C2) :
  concurrent (line_through A1 A2) (line_through B1 B2) (line_through C1 C2) :=
begin
  sorry
end

end concurrency_of_incircle_reflections_l271_271966


namespace trigonometric_identity_l271_271778

noncomputable def cos_alpha (α : ℝ) : ℝ := -Real.sqrt (1 / (1 + (tan α)^2))
noncomputable def sin_alpha (α : ℝ) : ℝ := Real.sqrt (1 - (cos_alpha α)^2)

theorem trigonometric_identity
  (α : ℝ) (h1 : tan α = -2) (h2 : (π / 2) < α ∧ α < π) :
  cos_alpha α + sin_alpha α = Real.sqrt(5) / 5 :=
sorry

end trigonometric_identity_l271_271778


namespace largest_is_B_l271_271675

noncomputable def A : ℚ := ((2023:ℚ) / 2022) + ((2023:ℚ) / 2024)
noncomputable def B : ℚ := ((2024:ℚ) / 2023) + ((2026:ℚ) / 2023)
noncomputable def C : ℚ := ((2025:ℚ) / 2024) + ((2025:ℚ) / 2026)

theorem largest_is_B : B > A ∧ B > C := by
  sorry

end largest_is_B_l271_271675


namespace problem_solution_l271_271215

-- Definitions for the given conditions
def num_children : ℕ := 5555

def sum_consecutive (k : ℕ) (f : ℕ → ℤ) : ℤ :=
  (List.range k).map (λ i => f i).sum

def a (n : ℕ) : ℤ :=
  if n % num_children = 1 then 1
  else if n % num_children = 12 then 21
  else if n % num_children = 123 then 321
  else if n % num_children = 1234 then 4321
  else -1  -- placeholder for other values, congruence assumption will hold

noncomputable def problem_statement (f : ℕ → ℤ) : Prop :=
  (∀ i : ℕ, i < num_children → sum_consecutive 2005 (λ k => f (i + k)) = 2005) ∧
  f 1 = 1 ∧
  f 12 = 21 ∧
  f 123 = 321 ∧
  f 1234 = 4321

noncomputable def solution_statement (f : ℕ → ℤ) : Prop :=
  f 5555 = -4659

theorem problem_solution : ∃ f : ℕ → ℤ, problem_statement f ∧ solution_statement f :=
  by
    sorry

end problem_solution_l271_271215


namespace box_volume_is_correct_l271_271621

noncomputable def box_volume (length width cut_side : ℝ) : ℝ :=
  (length - 2 * cut_side) * (width - 2 * cut_side) * cut_side

theorem box_volume_is_correct : box_volume 48 36 5 = 9880 := by
  sorry

end box_volume_is_correct_l271_271621


namespace sum_of_solutions_l271_271580

theorem sum_of_solutions : 
  let solutions := {x | 0 < x ∧ x ≤ 30 ∧ 17 * (5 * x - 3) % 10 = 34 % 10}
  in (∑ x in solutions, x) = 225 := by
  sorry

end sum_of_solutions_l271_271580


namespace fraction_of_blue_eggs_is_0_8_l271_271963

-- Defining the problem conditions
variables (E : ℕ) -- Total number of Easter eggs
variables (blue purple : ℝ) -- Fraction of blue and purple eggs

-- Given conditions
def fraction_purple : ℝ := 1 / 5 -- 1/5 are purple
def fraction_purple_with_five_candies : ℝ := (1 / 2) * fraction_purple -- Half of the purple eggs have five pieces of candy
def fraction_blue_with_five_candies (b : ℝ) := (1 / 4) * b -- 1/4 of blue eggs have five pieces of candy
def probability_five_candies : ℝ := 0.3 -- Probability of picking an egg with 5 pieces of candy

-- Required to prove
theorem fraction_of_blue_eggs_is_0_8 (b : ℝ) (h : E > 0)
  (h1 : purple = fraction_purple)
  (h2 : b = blue)
  (h3 : fraction_purple_with_five_candies * E + fraction_blue_with_five_candies b * E = probability_five_candies * E) :
  b = 0.8 := 
by
  sorry

end fraction_of_blue_eggs_is_0_8_l271_271963


namespace shopkeeper_gain_percent_during_sale_l271_271171

theorem shopkeeper_gain_percent_during_sale :
  let SP := 30         -- Selling Price of the kite
  let gain_percent := 15 / 100 -- Gain percent when selling at Rs. 30
  let discount_percent := 10 / 100 -- Discount percent during clearance sale
  let CP := SP / (1 + gain_percent) -- Cost Price calculation
  let discount := SP * discount_percent -- Discount amount calculation
  let SP_sale := SP - discount -- Selling Price during sale
  let gain_during_sale := SP_sale - CP -- Gain during the sale
  let gain_percent_during_sale := (gain_during_sale / CP) * 100 -- Gain percent during sale
  gain_percent_during_sale ≈ 3.49 := sorry

end shopkeeper_gain_percent_during_sale_l271_271171


namespace sticks_form_equilateral_triangle_l271_271691

theorem sticks_form_equilateral_triangle (n : ℕ) (h1 : n ≥ 5) (h2 : n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5) :
  ∃ k, k * 3 = (n * (n + 1)) / 2 :=
by
  sorry

end sticks_form_equilateral_triangle_l271_271691


namespace investor_receives_l271_271232

-- Define the conditions
def principal := 4000
def annual_rate := 0.10
def compounding_per_year := 1
def years := 2

-- Compound interest formula
def compound_interest (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) : ℝ := P * (1 + r / n) ^ (n * t)

-- Statement to prove
theorem investor_receives :
  compound_interest principal annual_rate compounding_per_year years = 4840 :=
by
  sorry

end investor_receives_l271_271232


namespace cos_double_angle_l271_271060

variable (θ : Real)

theorem cos_double_angle (h : sin θ = 3 / 5) : cos (2 * θ) = 7 / 25 := sorry

end cos_double_angle_l271_271060


namespace conjugate_of_complex_l271_271518

theorem conjugate_of_complex : conj (2 + (1:ℂ) * I) / (1 - (2:ℂ) * I) = -I := sorry

end conjugate_of_complex_l271_271518


namespace arithmetic_sequence_general_term_a_sum_b_n_l271_271803

variable (a : ℕ → ℕ)
variable (b : ℕ → ℕ)
variable (T : ℕ → ℕ)

noncomputable def a_1 : ℕ := 4

-- condition for a_n for n >= 2
noncomputable def a_n (n : ℕ) : ℕ :=
  if n = 1 then a_1 else a (n-1) + 2^(n-1) + 3

-- b_n definition
noncomputable def b_n (n : ℕ) : ℕ := a n - 3 * n

-- Sn definition
noncomputable def T_n (n : ℕ) : ℕ := ∑ k in Finset.range n, b_n k

theorem arithmetic_sequence {n : ℕ} (h : n ≥ 2) :
  (a n - 2^n) - (a (n-1) - 2^(n-1)) = 3 := sorry

theorem general_term_a (n : ℕ) :
  a n = 2^n + 3 * n - 1 := sorry

theorem sum_b_n (n : ℕ) :
  T_n n = 2^(n+1) - n - 2 := sorry

end arithmetic_sequence_general_term_a_sum_b_n_l271_271803


namespace solve_for_a_l271_271789

theorem solve_for_a (a x : ℝ) (h₁ : 2 * x - 3 = 5 * x - 2 * a) (h₂ : x = 1) : a = 3 :=
by
  sorry

end solve_for_a_l271_271789


namespace area_of_tangency_triangle_l271_271550

theorem area_of_tangency_triangle 
  (r1 r2 r3 : ℝ) 
  (h1 : r1 = 2)
  (h2 : r2 = 3)
  (h3 : r3 = 4) 
  (mutually_tangent : ∀ {c1 c2 c3 : ℝ}, c1 + c2 = r1 + r2 ∧ c2 + c3 = r2 + r3 ∧ c1 + c3 = r1 + r3 ) :
  ∃ area : ℝ, area = 3 * (Real.sqrt 6) / 2 :=
by
  sorry

end area_of_tangency_triangle_l271_271550


namespace optimal_strategy_second_player_wins_l271_271559

structure Graph (V : Type) :=
  (edges : V → V → Prop)
  (no_duplicate_edges : ∀ {u v: V}, edges u v → ¬ edges v u)

structure Game :=
  (vertices : Type)
  (graph : Graph vertices)
  (initial_position : vertices)
  (degree : vertices → ℕ)
  (move : vertices → vertices → Prop)
  (initial_degree : initial_position → ℕ)
  (final_state : vertices)
  (move_condition : ∀ u, move initial_position u → graph.edges initial_position u)
  (end_condition : final_state = initial_position ∧ degree final_state = 0)

theorem optimal_strategy_second_player_wins (G : Game) :
  (∃ (strategy : (G.vertices → G.vertices) × (G.vertices → G.vertices)), 
    (∀ u v, G.move u v → strategy.1 u = v) ∧ 
    (∀ u v, G.move v u → strategy.2 v = u)) → 
  (G.initial_position = G.final_state ∧ G.degree G.final_state = 0) → 
  second_player_wins :=
sorry

end optimal_strategy_second_player_wins_l271_271559


namespace angle_between_chords_l271_271227

theorem angle_between_chords (R r : ℝ) 
    (ratio : R / r = 9 - 4 * Real.sqrt 3) 
    (equal_length_chords : ∀ c : Real.Circle, ∃ (chord1 chord2 : Real.Segment), 
      chord1.length = chord2.length ∧ 
      chord1.is_tangent_to c ∧ 
      chord2.is_tangent_to c ∧ 
      chord1.is_perpendicular_to (R - r)) :
  ∃ θ : ℝ, θ = 30 :=
sorry

end angle_between_chords_l271_271227


namespace circle_area_variables_l271_271865

-- Define the components of the problem
def pi : ℝ := Real.pi
def S (r : ℝ) : ℝ := pi * r^2
def is_variable (x : ℝ → ℝ) := ∀ r₁ r₂ : ℝ, r₁ ≠ r₂ → x r₁ ≠ x r₂

-- The formal statement to prove
theorem circle_area_variables : is_variable S ∧ is_variable (fun (r : ℝ) => r) :=
by sorry

end circle_area_variables_l271_271865


namespace number_of_planting_methods_l271_271285

noncomputable def num_planting_methods : ℕ :=
  -- Six different types of crops
  let crops := ['A', 'B', 'C', 'D', 'E', 'F']
  -- Six trial fields arranged in a row, numbered 1 through 6
  -- Condition: Crop A cannot be planted in the first two fields
  -- Condition: Crop B must not be adjacent to crop A
  -- Answer: 240 different planting methods
  240

theorem number_of_planting_methods :
  num_planting_methods = 240 :=
  by
    -- Proof omitted
    sorry

end number_of_planting_methods_l271_271285


namespace no_A_is_B_all_B_are_C_l271_271607

variable {α : Type*} -- Assume a universe α where elements of A, B, and C are housed

-- Definition of sets A, B, and C
variables (A B C : set α)

-- Conditions translated to Lean definitions
def no_A_is_B : Prop := ∀ x, x ∈ A → x ∉ B
def all_B_are_C : Prop := ∀ x, x ∈ B → x ∈ C

-- Theorem statement combining conditions
theorem no_A_is_B_all_B_are_C (h1 : no_A_is_B A B) (h2 : all_B_are_C B C) : A ∩ B = ∅ ∧ B ⊆ C :=
by
  sorry

end no_A_is_B_all_B_are_C_l271_271607


namespace quadratic_int_roots_iff_n_eq_3_or_4_l271_271735

theorem quadratic_int_roots_iff_n_eq_3_or_4 (n : ℕ) (hn : 0 < n) :
    (∃ m k : ℤ, (m ≠ k) ∧ (m^2 - 4 * m + n = 0) ∧ (k^2 - 4 * k + n = 0)) ↔ (n = 3 ∨ n = 4) := sorry

end quadratic_int_roots_iff_n_eq_3_or_4_l271_271735


namespace nonzero_fraction_power_zero_fraction_pow_zero_l271_271236

noncomputable def fraction := (-123456789012345 : ℚ) / 9876543210987654321

theorem nonzero_fraction : fraction ≠ 0 :=
by
  have h_num : (-123456789012345 : ℚ) ≠ 0 := by norm_num
  have h_den : (9876543210987654321 : ℚ) ≠ 0 := by norm_num
  rw fraction
  exact div_ne_zero h_num h_den

theorem power_zero (f : ℚ) (h : f ≠ 0) : f ^ 0 = 1 :=
by exact pow_zero f

theorem fraction_pow_zero : fraction ^ 0 = 1 :=
by apply power_zero fraction nonzero_fraction

end nonzero_fraction_power_zero_fraction_pow_zero_l271_271236


namespace likelihood_ratio_sqrt2_l271_271148

namespace ProbabilityProblem

open Nat

theorem likelihood_ratio_sqrt2 (n : ℕ) :
  let P (k : ℕ) := (Nat.choose (2 * n) k / 2 ^ (2 * n) : ℝ)
  let PA := P n
  let PB := (Nat.choose (2 * n) n / 2 ^ (2 * n)) ^ 2 / (Nat.choose (4 * n) (2 * n) / 2 ^ (4 * n))
  PA ≠ 0 ∧ PB ≠ 0 → PB / PA = Real.sqrt 2 :=
by
  sorry

end ProbabilityProblem

end likelihood_ratio_sqrt2_l271_271148


namespace fraction_filled_in_3_minutes_l271_271282

theorem fraction_filled_in_3_minutes (time_to_fill_cistern : ℕ) (fraction_time : ℕ) : 
  time_to_fill_cistern = 33 ∧ fraction_time = 3 → (fraction_time / time_to_fill_cistern) = (1 / 11) := 
by
  intro h
  cases h
  have fraction_filled_per_minute : ℝ := 1 / time_to_fill_cistern
  have fraction_in_3_minutes : ℝ := fraction_filled_per_minute * fraction_time
  calc 
    fraction_in_3_minutes = (1 / 33) * 3 : by
      rw [h_left, h_right]
  ... = 1 / 11 : by
      linarith

  sorry

end fraction_filled_in_3_minutes_l271_271282


namespace find_tangent_c_l271_271787

theorem find_tangent_c (c : ℝ) : 
  (∀ x y : ℝ, y = 3 * x + c ∧ y^2 = 12 * x → (-12)^2 - 4 * (1) * (12 * c) = 0) → c = 3 :=
sorry

end find_tangent_c_l271_271787


namespace range_of_a_l271_271068

theorem range_of_a (a : ℝ) : 
  (a * (2 + a) < 0) → a ∈ set.Ioo (-2 : ℝ) 0 :=
by
  sorry

end range_of_a_l271_271068


namespace trigonometric_identity_l271_271821

theorem trigonometric_identity :
  ∀ α : Real, sin (π / 6 - α) = 1 / 3 → 2 * cos (π / 6 + α / 2) ^ 2 - 1 = 1 / 3 :=
by
  intro α h
  sorry

end trigonometric_identity_l271_271821


namespace first_digit_base12_1025_l271_271237

theorem first_digit_base12_1025 : (1025 : ℕ) / (12^2 : ℕ) = 7 := by
  sorry

end first_digit_base12_1025_l271_271237


namespace magnitude_of_2a_plus_b_l271_271401

noncomputable def vector_a (x : ℝ) : ℝ × ℝ × ℝ := (x, 1, 1)
def vector_b (y : ℝ) : ℝ × ℝ × ℝ := (1, y, 1)
def vector_c : ℝ × ℝ × ℝ := (2, -4, 2)

-- Dot product definition
def dot_product (v w : ℝ × ℝ × ℝ) : ℝ :=
v.1 * w.1 + v.2 * w.2 + v.3 * w.3

-- Scalar multiplication
def scalar_mult (k : ℝ) (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
(k * v.1, k * v.2, k * v.3)

-- Vector addition
def vector_add (v w : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
(v.1 + w.1, v.2 + w.2, v.3 + w.3)

-- Magnitude of a vector
def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
real.sqrt (v.1^2 + v.2^2 + v.3^2)

theorem magnitude_of_2a_plus_b (x y : ℝ)
  (h1 : dot_product (vector_a x) vector_c = 0)
  (h2 : ∃ k : ℝ, vector_b y = scalar_mult k vector_c) :
  magnitude (vector_add (scalar_mult 2 (vector_a x)) (vector_b y)) = 3 * real.sqrt 2 :=
sorry

end magnitude_of_2a_plus_b_l271_271401


namespace find_five_digit_numbers_divisible_by_72_l271_271271

theorem find_five_digit_numbers_divisible_by_72 (n : ℤ) :
  (9999 < n) ∧ (n < 100000) ∧ (n % 72 = 0) ∧
  (Int.toDigits 10 n).count 1 = 3 ↔ n ∈ {41112, 14112, 11016, 11160} :=
by
  sorry

end find_five_digit_numbers_divisible_by_72_l271_271271


namespace total_value_correct_l271_271262

noncomputable def total_value (num_coins : ℕ) : ℕ :=
  let value_one_rupee := num_coins * 1
  let value_fifty_paise := (num_coins * 50) / 100
  let value_twentyfive_paise := (num_coins * 25) / 100
  value_one_rupee + value_fifty_paise + value_twentyfive_paise

theorem total_value_correct :
  let num_coins := 40
  total_value num_coins = 70 := by
  sorry

end total_value_correct_l271_271262


namespace no_roots_in_interval_l271_271384

noncomputable def f : ℝ → ℝ := λ x, x^4 - 4*x^3 + 10*x^2

theorem no_roots_in_interval : ∀ x ∈ Icc (1 : ℝ) 2, f x ≠ 0 :=
by
  sorry

end no_roots_in_interval_l271_271384


namespace upper_base_length_l271_271761

-- Definitions based on the given conditions
variables (A B C D M : ℝ)
variables (d : ℝ)
variables (h : ℝ) -- height from D to AB

-- Conditions given in the problem
def is_trapezoid (ABCD : ℝ) : Prop := true
def is_perpendicular (DM AB : ℝ) : Prop := true
def point_on_side (M AB : ℝ) : Prop := true
def MC_eq_CD (MC CD : ℝ) : Prop := MC = CD
def AD_length (A D d : ℝ) : Prop := A - D = d -- Assuming some coordinate system 

-- Define the proof statement with conditions and the result
theorem upper_base_length
  (ht : is_trapezoid ABCD)
  (hp : is_perpendicular D M)
  (ps : point_on_side M AB)
  (mc_cd : MC_eq_CD M C D)
  (ad_len : AD_length A D d) :
  BC = d / 2 :=
sorry

end upper_base_length_l271_271761


namespace max_value_l271_271718

-- Define the function whose maximum we want to find
def f (x y : ℝ) : ℝ :=
  (x + 3 * y + 5) / Real.sqrt (x^2 + y^2 + 4)

theorem max_value : ∃ x y : ℝ, (∀ x y : ℝ, f x y ≤ Real.sqrt 35) ∧ 
                                (f x y = Real.sqrt 35) :=
  sorry

end max_value_l271_271718


namespace universal_acquaintance_l271_271957

-- Definitions of the problem's elements
def round_table (G : Type) := G

def mutual_acquaintanceship (G : Type) (acquaintance : G → G → Prop) :=
  ∀ (a b : G), acquaintance a b → acquaintance b a

def equal_intervals (G : Type) (acquaintance : G → G → Prop) :=
  ∀ (g : G), ∃ (n : ℕ), ∀ (k : ℕ), acquaintance g (rotate (n * k) g)

def mutual_acquaintance (G : Type) (acquaintance : G → G → Prop) :=
  ∀ (a b : G), ∃ (c : G), acquaintance a c ∧ acquaintance b c

def complete_acquaintance (G : Type) (acquaintance : G → G → Prop) :=
  ∀ (a b : G), a ≠ b → acquaintance a b

-- Theorem Statement
theorem universal_acquaintance
  (G : Type) -- Guests type
  (acquaintance : G → G → Prop) -- Acquaintance relation
  (H1 : round_table G)
  (H2 : mutual_acquaintanceship G acquaintance)
  (H3 : equal_intervals G acquaintance)
  (H4 : mutual_acquaintance G acquaintance) :
  complete_acquaintance G acquaintance := 
sorry

end universal_acquaintance_l271_271957


namespace fraction_of_income_from_tips_l271_271288

theorem fraction_of_income_from_tips (S T : ℚ) (h : T = (11/4) * S) : (T / (S + T)) = (11/15) :=
by sorry

end fraction_of_income_from_tips_l271_271288


namespace negate_exists_real_l271_271535

theorem negate_exists_real (h : ¬ ∃ x : ℝ, x^2 - 2 ≤ 0) : ∀ x : ℝ, x^2 - 2 > 0 :=
by
  sorry

end negate_exists_real_l271_271535


namespace min_value_5_log_m_plus_9_log_n_squared_l271_271385

def has_two_distinct_zeros (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a ≠ b ∧ f a = 0 ∧ f b = 0

theorem min_value_5_log_m_plus_9_log_n_squared
  (m n : ℝ)
  (hm : m > 0)
  (hn : n > 0)
  (h : has_two_distinct_zeros (λ x : ℝ, 2 * m * x ^ 3 - 3 * n * x ^ 2 + 10)) :
  5 * (Real.log m)^2 + 9 * (Real.log n)^2 = 5/9 := sorry

end min_value_5_log_m_plus_9_log_n_squared_l271_271385


namespace largest_root_of_polynomial_l271_271339

theorem largest_root_of_polynomial (p q r : ℝ)
  (h1 : p + q + r = 1)
  (h2 : p * q + p * r + q * r = -8)
  (h3 : p * q * r = 15) :
  ∃ x ∈ {p, q, r}, x = 3 ∧ ∀ y ∈ {p, q, r}, y ≤ x :=
by
  sorry

end largest_root_of_polynomial_l271_271339


namespace quadratic_symmetry_axis_l271_271649

/-- 
  This theorem states that the quadratic function y = (x + 1)^2 
  has a symmetry axis at the line x = -1.
--/
theorem quadratic_symmetry_axis (x : ℝ) :
  let y := (x + 1) ^ 2
  in ∃ h : ℝ, h = -1 ∧ (∀ x1 x2 : ℝ, y x1 = y x2 → x1 = -1 ∨ x2 = -1) :=
sorry

end quadratic_symmetry_axis_l271_271649


namespace sticks_forming_equilateral_triangle_l271_271697

theorem sticks_forming_equilateral_triangle (n : ℕ) (h1 : n ≥ 5) :
  (∃ l : list ℕ, (∀ x ∈ l, x ∈ finset.range (n + 1) ∧ x > 0) ∧ l.sum % 3 = 0) ↔ 
  (n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5) := 
sorry

end sticks_forming_equilateral_triangle_l271_271697


namespace triangle_in_Q5_exists_smallest_m_exists_irrational_tangent_triangle_similar_triangle_in_Q6_exists_l271_271562

-- Part (a)
theorem triangle_in_Q5_exists (ABC : triangle (ℚ ^ n)) : 
  ∃ (A' B' C' : point (ℚ ^ 5)), ∠ B' A' C' = ∠ B A C :=
sorry

-- Part (b)
theorem smallest_m_exists : ∀ (ABC : triangle (ℚ ^ n)), ∃ (m : ℕ), (m = 5) ∧ embedded_in_Q_m ABC m :=
sorry

-- Part (c)
theorem irrational_tangent_triangle : ∃ (ABC : triangle (ℚ ^ n)), 
  ∀ (A' B' C' : triangle (ℚ ^ 3)), ¬ similar_triangle ABC A' B' C' :=
sorry

-- Part (d)
theorem similar_triangle_in_Q6_exists : ∀ (ABC : triangle (ℚ ^ n)), 
  ∃ (A' B' C' : point (ℚ ^ 6)), similar_triangle ABC (triangle.mk A' B' C') :=
sorry

end triangle_in_Q5_exists_smallest_m_exists_irrational_tangent_triangle_similar_triangle_in_Q6_exists_l271_271562


namespace quadrilateral_coverage_l271_271532

-- Definition of a convex quadrilateral with side lengths no more than 7
structure ConvexQuadrilateral (A B C D : Type) :=
(convex : is_convex A B C D)
(side_lengths_leq_7 : ∀ (X Y : Type), {X, Y} ⊆ {A, B, C, D} → dist X Y ≤ 7)

-- The theorem statement in Lean 4
theorem quadrilateral_coverage 
  {A B C D : Type} 
  (quad : ConvexQuadrilateral A B C D) 
  (O : Type):
  (distance O A ≤ 5) ∨ (distance O B ≤ 5) ∨ (distance O C ≤ 5) ∨ (distance O D ≤ 5) :=
sorry

end quadrilateral_coverage_l271_271532


namespace peter_lost_85_marbles_l271_271157

theorem peter_lost_85_marbles :
  ∀ (initial marbles lost_fraction remaining_fraction torn_loss : ℕ),
    initial = 120 →
    lost_fraction = 1/4 → 
    remaining_fraction = 1/2 → 
    torn_loss = 10 → 
    (lost_fraction * initial) + 
    (remaining_fraction * (initial - lost_fraction * initial)) + 
    torn_loss = 85 := 
by
  intros initial marbles lost_fraction remaining_fraction torn_loss
  assume h_initial : initial = 120
  assume h_lost_fraction : lost_fraction = 1/4
  assume h_remaining_fraction : remaining_fraction = 1/2
  assume h_torn_loss : torn_loss = 10
  sorry

end peter_lost_85_marbles_l271_271157


namespace problem_proof_l271_271140

noncomputable def a_n (n : ℕ) : ℕ :=
  n * (n + 1)

noncomputable def b_n (n : ℕ) : ℝ :=
  (1 / 2 ^ (n - 1))

noncomputable def S_n (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, sqrt (a_n (i + 1)) * ∑ j in Finset.range i, b_n (j + 1)

theorem problem_proof (n : ℕ) (hn : 0 < n) :
  a_n n = n * (n + 1) ∧
  b_n n = (1 / 2 ^ (n - 1)) ∧
  S_n n = (∑ i in Finset.range n, sqrt (a_n (i + 1))) * (∑ i in Finset.range n, b_n (i + 1)) ∧
  S_n n ≤ n * (n + 2) :=
by
  sorry

end problem_proof_l271_271140


namespace crossword_digit_intersection_l271_271849

theorem crossword_digit_intersection :
  (∃ m n : ℕ, (100 ≤ 2^m ∧ 2^m < 1000) ∧ (100 ≤ 3^n ∧ 3^n < 1000) ∧ digit2(2^m) = digit2(3^n) ∧ digit2(2^m) = 2) :=
sorry

def digit2 (n : ℕ) : ℕ := (n / 10) % 10

end crossword_digit_intersection_l271_271849


namespace nine_digit_palindrome_count_l271_271403

-- Defining the set of digits
def digits : Multiset ℕ := {1, 1, 2, 2, 2, 4, 4, 5, 5}

-- Defining the proposition of the number of 9-digit palindromes
def num_9_digit_palindromes (digs : Multiset ℕ) : ℕ := 36

-- The proof statement
theorem nine_digit_palindrome_count : num_9_digit_palindromes digits = 36 := 
sorry

end nine_digit_palindrome_count_l271_271403


namespace morleys_theorem_l271_271250

def is_trisector (A B C : Point) (p : Point) : Prop :=
sorry -- Definition that this point p is on one of the trisectors of ∠BAC

def triangle (A B C : Point) : Prop :=
sorry -- Definition that points A, B, C form a triangle

def equilateral (A B C : Point) : Prop :=
sorry -- Definition that triangle ABC is equilateral

theorem morleys_theorem (A B C D E F : Point)
  (hABC : triangle A B C)
  (hD : is_trisector A B C D)
  (hE : is_trisector B C A E)
  (hF : is_trisector C A B F) :
  equilateral D E F :=
sorry

end morleys_theorem_l271_271250


namespace part1_problem_l271_271928

theorem part1_problem
  (A B C : Real.Angle)
  (a b c : ℝ)
  (cosA : Real.cos A)
  (sinA : Real.sin A)
  (sin2B : Real.sin (2 * B))
  (cos2B : Real.cos (2 * B))
  (hC : C = 2 * π / 3)
  (h : cosA / (1 + sinA) = sin2B / (1 + cos2B))
  : B = π / 6 := by
  sorry

end part1_problem_l271_271928


namespace intersection_A_B_l271_271172

-- Define the sets A and B
def setA : Set ℤ := {x | (1 / 2 : ℝ) ≤ 2^x ∧ 2^x ≤ 2}
def setB : Set ℝ := {y | ∃ x ∈ setA, y = Real.cos x}

-- State the theorem
theorem intersection_A_B : {1} = setA ∩ setB :=
by
  -- Proof not provided, placeholder sorry
  sorry

end intersection_A_B_l271_271172


namespace medication_dose_function_one_bag_weight_range_l271_271521

variables (x y : ℝ)

-- Condition 1: A child weighing 10kg takes 110mg per dose
def medication_dose1 : Prop := y = 110 ∧ x = 10

-- Condition 2: A child weighing 15kg takes 160mg per dose
def medication_dose2 : Prop := y = 160 ∧ x = 15

-- Condition 3: The dosage y is linear in terms of weight x
def linear_dose (k b : ℝ): Prop := y = k * x + b

-- Condition 4: The dosage range for x is between 5 and 50 kg
def weight_range : Prop := 5 ≤ x ∧ x ≤ 50

-- Problem verification
theorem medication_dose_function :
  (∃ (k b : ℝ), linear_dose k b ∧ medication_dose1 
  ∧ medication_dose2 ∧ weight_range) →
    (∃ (a : ℝ), y = 10 * x + 10 ∧ weight_range) :=
sorry

-- Further verification: If one package is 300mg
def normal_dose : Prop := y = 300

-- Maximum safe dosage condition
def maximum_safe_dose : Prop := y = 300 * 1.2

-- Problem verification for the weight range with one package size
theorem one_bag_weight_range :
  (normal_dose ∨ y ≤ 300 * 1.2) →
    (24 ≤ x ∧ x ≤ 29) :=
sorry

end medication_dose_function_one_bag_weight_range_l271_271521


namespace odd_function_property_l271_271793

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (4 - x^2)) / x

theorem odd_function_property (a : ℝ) (h_a : -2 ≤ a ∧ a ≤ 2) (h_fa : f a = -4) : f (-a) = 4 :=
by
  sorry

end odd_function_property_l271_271793


namespace fg_neg_one_eq_neg_eight_l271_271471

def f (x : ℤ) : ℤ := x - 4
def g (x : ℤ) : ℤ := x^2 + 2*x - 3

theorem fg_neg_one_eq_neg_eight : f (g (-1)) = -8 := by
  sorry

end fg_neg_one_eq_neg_eight_l271_271471


namespace final_cash_amounts_l271_271151

theorem final_cash_amounts (a_initial_cash b_initial_cash a_house_value t1_sell_price t2_buy_price t3_sell_price : ℤ) :
    a_initial_cash = 12000 →
    b_initial_cash = 13000 →
    a_house_value = 12000 →
    t1_sell_price = 14000 →
    t2_buy_price = 11000 →
    t3_sell_price = 15000 →
    let a_final_cash := a_initial_cash + t1_sell_price - t2_buy_price + t3_sell_price in
    let b_final_cash := b_initial_cash - t1_sell_price + t2_buy_price - t3_sell_price in
    a_final_cash = 30000 ∧ b_final_cash = -5000 :=
by
  intros
  let a_final_cash := 12000 + 14000 - 11000 + 15000
  let b_final_cash := 13000 - 14000 + 11000 - 15000
  exact ⟨rfl, rfl⟩
  sorry

end final_cash_amounts_l271_271151


namespace find_B_min_value_a2_b2_c2_l271_271897

theorem find_B (A B C : ℝ) (h1 : C = 2 * π / 3) 
  (h2 : (cos A) / (1 + sin A) = (sin (2 * B)) / (1 + cos (2 * B))) : B = π / 6 :=
sorry

theorem min_value_a2_b2_c2 (A B C a b c : ℝ) (h1 : C = 2 * π / 3)
  (h2 : (cos A) / (1 + sin A) = (sin (2 * B)) / (1 + cos (2 * B))) 
  (h3 : triangles_side_identity a b c A B C AE BA CE) :
  (a^2 + b^2) / c^2 = 4 * Real.sqrt 2 - 5 :=
sorry

end find_B_min_value_a2_b2_c2_l271_271897


namespace quadrilateral_area_eq_half_l271_271543

noncomputable def sin_quad_area (n : ℕ) : ℝ :=
  let x := [Real.sin (π * n), Real.sin (π * (n + 1)), Real.sin (π * (n + 2)), Real.sin (π * (n + 3))] in 
  abs ((x[0] * π * (n + 1) + x[1] * π * (n + 2) + x[2] * π * (n + 3) + x[3] * π * n) - 
       (π * n * x[1] + π * (n + 1) * x[2] + π * (n + 2) * x[3] + π * (n + 3) * x[0])) / 2

theorem quadrilateral_area_eq_half (n : ℕ) (hpos : 0 < n) :
  sin_quad_area n = 0.5 := 
sorry

end quadrilateral_area_eq_half_l271_271543


namespace quotient_of_1575_210_l271_271189

theorem quotient_of_1575_210 (a b q : ℕ) (h1 : a = 1575) (h2 : b = a - 1365) (h3 : a % b = 15) : q = 7 :=
by {
  sorry
}

end quotient_of_1575_210_l271_271189


namespace students_with_dogs_l271_271634

theorem students_with_dogs (total_students : ℕ) (half_students : total_students = 100)
                           (girls_percentage : ℕ) (boys_percentage : ℕ)
                           (girls_dog_percentage : ℕ) (boys_dog_percentage : ℕ)
                           (P1 : half_students / 2 = 50)
                           (P2 : girls_dog_percentage = 20)
                           (P3 : boys_dog_percentage = 10) :
                           (50 * girls_dog_percentage / 100 + 
                            50 * boys_dog_percentage / 100) = 15 :=
by sorry

end students_with_dogs_l271_271634


namespace original_percentage_of_acid_l271_271276

theorem original_percentage_of_acid 
  (a w : ℝ) 
  (h1 : a + w = 6) 
  (h2 : a / (a + w + 2) = 15 / 100) 
  (h3 : (a + 2) / (a + w + 4) = 25 / 100) :
  (a / 6) * 100 = 20 :=
  sorry

end original_percentage_of_acid_l271_271276


namespace distribute_cousins_into_rooms_l271_271936

theorem distribute_cousins_into_rooms : 
  let num_cousins := 5
  let num_rooms := 5
  ∃ finset ℕ, num_ways = 52 :=
by 
  let partitions := [
    (5,0,0,0,0), 
    (4,1,0,0,0), 
    (3,2,0,0,0), 
    (3,1,1,0,0), 
    (2,2,1,0,0), 
    (2,1,1,1,0), 
    (1,1,1,1,1)
  ]
  let num_ways := 
    1 + 
    5 + 
    10 + 
    10 + 
    15 + 
    10 + 
    1
  sorry

end distribute_cousins_into_rooms_l271_271936


namespace jasmine_laps_l271_271101

/-- Jasmine swims 12 laps every afternoon, Monday through Friday. Calculate the total number of laps she swims in five weeks. -/
theorem jasmine_laps : 
    ∀ (laps_per_day days_per_week number_of_weeks : ℕ),
        laps_per_day = 12 ∧ days_per_week = 5 ∧ number_of_weeks = 5 → 
        number_of_weeks * (days_per_week * laps_per_day) = 300 :=
by
    intros laps_per_day days_per_week number_of_weeks
    intro h
    cases h with hlaps h_days_weeks
    cases h_days_weeks with hdays hweeks
    rw [hlaps, hdays, hweeks]
    norm_num
    sorry

end jasmine_laps_l271_271101


namespace sum_of_exponents_l271_271430

-- Given product of integers from 1 to 15
def y := Nat.factorial 15

-- Prime exponent variables in the factorization of y
variables (i j k m n p q : ℕ)

-- Conditions
axiom h1 : y = 2^i * 3^j * 5^k * 7^m * 11^n * 13^p * 17^q 

-- Prove that the sum of the exponents equals 24
theorem sum_of_exponents :
  i + j + k + m + n + p + q = 24 := 
sorry

end sum_of_exponents_l271_271430


namespace angle_translation_l271_271555

theorem angle_translation :
  ∃ α k : ℤ, -1485 = α + k * 360 ∧ 0 ≤ α ∧ α < 360 :=
begin
  use 315, -- α
  use -5, -- k
  split,
  { linarith, }, -- proving -1485 = 315 - 5 * 360
  split,
  { norm_num, }, -- proving 0 ≤ 315
  { norm_num, }, -- proving 315 < 360
end

end angle_translation_l271_271555


namespace sum_of_distinct_integers_eq_36_l271_271127

theorem sum_of_distinct_integers_eq_36
  (p q r s t : ℤ)
  (hpq : p ≠ q) (hpr : p ≠ r) (hps : p ≠ s) (hpt : p ≠ t)
  (hqr : q ≠ r) (hqs : q ≠ s) (hqt : q ≠ t)
  (hrs : r ≠ s) (hrt : r ≠ t)
  (hst : s ≠ t)
  (h : (8 - p) * (8 - q) * (8 - r) * (8 - s) * (8 - t) = 80) :
  p + q + r + s + t = 36 :=
by
  sorry

end sum_of_distinct_integers_eq_36_l271_271127


namespace number_of_m_gons_proof_l271_271479

noncomputable def number_of_m_gons_with_two_acute_angles (m n : ℕ) (h1 : 4 < m) (h2 : m < n) : ℕ :=
  (2 * n + 1) * (Nat.choose (n + 1) (m - 1) + Nat.choose n (m - 1))

theorem number_of_m_gons_proof {m n : ℕ} (h1 : 4 < m) (h2 : m < n) :
  number_of_m_gons_with_two_acute_angles m n h1 h2 =
  (2 * n + 1) * ((Nat.choose (n + 1) (m - 1)) + (Nat.choose n (m - 1))) :=
sorry

end number_of_m_gons_proof_l271_271479


namespace minimum_value_of_a_plus_b_l271_271067

noncomputable def f (x : ℝ) := Real.log x - (1 / x)
noncomputable def f' (x : ℝ) := 1 / x + 1 / (x^2)

theorem minimum_value_of_a_plus_b (a b m : ℝ) (h1 : a = 1 / m + 1 / (m^2)) 
  (h2 : b = Real.log m - 2 / m - 1) : a + b = -1 :=
by
  sorry

end minimum_value_of_a_plus_b_l271_271067


namespace sqrt_of_4_l271_271984

theorem sqrt_of_4 :
  {x | x * x = 4} = {2, -2} :=
sorry

end sqrt_of_4_l271_271984


namespace prob_at_least_two_same_l271_271935

open ProbabilityTheory

-- Define a fair 8-sided die
def eight_sided_die : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

-- Define the event space for rolling 7 eight-sided dice
def event_space := (Finset.product (λ _, eight_sided_die) (𝓝 n := 7))

-- Define the event where each die shows a different number
def all_different (outcome : Vector ℕ 7) : Prop :=
  outcome.to_list.nodup

-- Define the probability space
noncomputable def prob_space :=
  measure_theory.measure_space (event_space.to_finset.fintype)

-- Define the probability of having all different outcomes
noncomputable def P_all_different : ℚ :=
  prob_space.measure (measure_theory.measurable_set.prod
    (λ i, (eight_sided_die.erase (List.nth_le (outcome.to_list) i sorry))))

-- Define the probability of having at least two dice showing the same number
noncomputable def P_at_least_two_same : ℚ :=
  1 - P_all_different

-- Define the main conjecture
theorem prob_at_least_two_same :
  P_at_least_two_same = 319 / 320 := by
  sorry

end prob_at_least_two_same_l271_271935


namespace proof_of_a_neg_two_l271_271734

theorem proof_of_a_neg_two (a : ℝ) (i : ℂ) (h_i : i^2 = -1) (h_real : (1 + i)^2 - a / i = (a + 2) * i → ∃ r : ℝ, (1 + i)^2 - a / i = r) : a = -2 :=
sorry

end proof_of_a_neg_two_l271_271734


namespace sally_initial_orange_balloons_l271_271501

variable (initial_orange_balloons : ℕ)  -- The initial number of orange balloons Sally had
variable (lost_orange_balloons : ℕ := 2)  -- The number of orange balloons Sally lost
variable (current_orange_balloons : ℕ := 7)  -- The number of orange balloons Sally currently has

theorem sally_initial_orange_balloons : 
  current_orange_balloons + lost_orange_balloons = initial_orange_balloons := 
by
  sorry

end sally_initial_orange_balloons_l271_271501


namespace no_nondegenerate_triangle_l271_271258

def distinct_positive_integers (a b c : ℕ) : Prop :=
  (0 < a) ∧ (0 < b) ∧ (0 < c) ∧ (a ≠ b) ∧ (b ≠ c) ∧ (c ≠ a)

def nondegenerate_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem no_nondegenerate_triangle (a b c : ℕ)
  (h_distinct : distinct_positive_integers a b c)
  (h_gcd : Nat.gcd (Nat.gcd a b) c = 1)
  (h1 : a ∣ (b - c) ^ 2)
  (h2 : b ∣ (c - a) ^ 2)
  (h3 : c ∣ (a - b) ^ 2) :
  ¬nondegenerate_triangle a b c :=
sorry

end no_nondegenerate_triangle_l271_271258


namespace relationship_between_abc_l271_271732

open Real

variables (a b c : ℝ)

def definition_a := a = 1 / 2023
def definition_b := b = tan (exp (1 / 2023) / 2023)
def definition_c := c = sin (exp (1 / 2024) / 2024)

theorem relationship_between_abc (h1 : definition_a a) (h2 : definition_b b) (h3 : definition_c c) : 
  c < a ∧ a < b :=
by
  sorry

end relationship_between_abc_l271_271732


namespace area_of_triangle_l271_271829

theorem area_of_triangle {A B C : Type} [HasArea A B C]
  (CE : ℝ) (h : CE = 2) (h_30_60_90 : is_30_60_90_triangle A B C) :
  area A B C = 2 := 
  sorry

end area_of_triangle_l271_271829


namespace prove_angle_BAC_eq_theta_prove_cos_theta_l271_271858

variable (A B C D E : Type)
variables (AD_perp_BCD : ⟦ AD ⟧ ⊥ plane BCD)
variables (angle_ABD_eq_BDC : ∀ θ : ℝ, angle ABD = θ ∧ angle BDC = θ ∧ θ < (45 : ℝ))
variables (BE_eq_AD : ∀ BE AD BE', BE = AD ∧ AD = 1)
variables (CE_perp_BD : Π ⟨BD, CE⟩, CE ⊥ BD)

-- Part 1: Proving angle BAC = θ under given conditions
theorem prove_angle_BAC_eq_theta (θ : ℝ) (AD_perp_BCD : ⟦ AD ⟧ ⊥ plane BCD) 
    (angle_ABD_eq_BDC : ∀ θ : ℝ, angle ABD = θ ∧ angle BDC = θ ∧ θ < (45 : ℝ)) 
    (BE_eq_AD : ∀ BE AD BE', BE = AD ∧ AD = 1) 
    (CE_perp_BD : Π ⟨BD, CE⟩, CE ⊥ BD) : ∠ BAC = θ :=
sorry

-- Part 2: Proving cos θ = 4/5 given the distance from D to plane ABC is 4/13
theorem prove_cos_theta (θ : ℝ) (distance_from_D_to_plane_ABC : ℝ) 
    (given_distance : distance_from_D_to_plane_ABC = 4/13) : cos θ = 4/5 :=
sorry

end prove_angle_BAC_eq_theta_prove_cos_theta_l271_271858


namespace city_mileage_l271_271609

theorem city_mileage (x : ℝ): 
  (∀ (highway_gas city_gas : ℝ), 
  highway_gas = 4 / 34 ∧ city_gas = 4 / x → 
  (highway_gas + city_gas = (8 / 34 * 1.3499999999999999)) → 
  x = 10) :=
begin
  intros highway_gas city_gas,
  sorry
end

end city_mileage_l271_271609


namespace part1_part2_l271_271915

namespace MathProof

-- defining the basic setup of the triangle and given constraints
variables (A B C : ℝ) (a b c : ℝ)

-- condition 1: the equation relating cosines and sines
def condition1 : Prop := (cos A) / (1 + sin A) = (sin (2 * B)) / (1 + cos (2 * B))

-- condition 2: specific value of angle C
def condition2 : Prop := C = 2 * π / 3

-- question 1: finding angle B
def findB : Prop := B = π / 6

-- The minimum value of (a^2 + b^2) / c^2
def min_value_expr : ℝ := (a^2 + b^2) / c^2
def min_value : ℝ := 4 * real.sqrt 2 - 5

-- Proving both parts
theorem part1 (h1 : condition1) (h2 : condition2) : findB := sorry

theorem part2 (h1 : condition1) (h2 : condition2) : min_value_expr = min_value := sorry

end MathProof

end part1_part2_l271_271915


namespace parabola_vertex_relationship_l271_271168

theorem parabola_vertex_relationship (m x y : ℝ) :
  (y = x^2 - 2*m*x + 2*m^2 - 3*m + 1) → (y = x^2 - 3*x + 1) :=
by
  intro h
  sorry

end parabola_vertex_relationship_l271_271168


namespace find_j_l271_271126

theorem find_j (a b c j : ℤ) 
  (h1 : f 1 = 0 → a + b + c = 0) 
  (h2 : f (-1) = 0 → a - b + c = 0) 
  (h3 : 70 < f 7 < 90)
  (h4 : 110 < f 8 < 140)
  (h5 : 1000 * j < f 50 ∧ f 50 < 1000 * (j + 1)) :
  (∃ j, j = 4) := 
by
  sorry

end find_j_l271_271126


namespace lemon_cookies_amount_l271_271104

def cookies_problem 
  (jenny_pb_cookies : ℕ) (jenny_cc_cookies : ℕ) (marcus_pb_cookies : ℕ) (marcus_lemon_cookies : ℕ)
  (total_pb_cookies : ℕ) (total_non_pb_cookies : ℕ) : Prop :=
  jenny_pb_cookies = 40 ∧
  jenny_cc_cookies = 50 ∧
  marcus_pb_cookies = 30 ∧
  total_pb_cookies = jenny_pb_cookies + marcus_pb_cookies ∧
  total_pb_cookies = 70 ∧
  total_non_pb_cookies = jenny_cc_cookies + marcus_lemon_cookies ∧
  total_pb_cookies = total_non_pb_cookies

theorem lemon_cookies_amount
  (jenny_pb_cookies : ℕ) (jenny_cc_cookies : ℕ) (marcus_pb_cookies : ℕ) (marcus_lemon_cookies : ℕ)
  (total_pb_cookies : ℕ) (total_non_pb_cookies : ℕ) :
  cookies_problem jenny_pb_cookies jenny_cc_cookies marcus_pb_cookies marcus_lemon_cookies total_pb_cookies total_non_pb_cookies →
  marcus_lemon_cookies = 20 :=
by
  sorry

end lemon_cookies_amount_l271_271104


namespace Willie_dollars_exchange_l271_271589

theorem Willie_dollars_exchange:
  ∀ (euros : ℝ) (official_rate : ℝ) (airport_rate : ℝ),
  euros = 70 →
  official_rate = 5 →
  airport_rate = 5 / 7 →
  euros / official_rate * airport_rate = 10 :=
by
  intros euros official_rate airport_rate
  intros h_euros h_official_rate h_airport_rate
  rw [h_euros, h_official_rate, h_airport_rate]
  norm_num
  sorry

end Willie_dollars_exchange_l271_271589


namespace percentage_increase_after_initial_decrease_l271_271978

variable (P : ℝ) (initial_decrease_percent net_increase_percent final_increase_percent : ℝ)

-- Conditions
def initial_decrease : ℝ := initial_decrease_percent / 100 * P
def net_increase : ℝ := net_increase_percent / 100 * P

-- original and conditions described in plain words
def price_after_decrease : ℝ := P - initial_decrease

def price_after_increase : ℝ := price_after_decrease * (1 + final_increase_percent / 100)

def final_price : ℝ := P + net_increase

-- Proof problem statement
theorem percentage_increase_after_initial_decrease 
  (initial_decrease_percent : initial_decrease_percent = 50)
  (net_increase_percent : net_increase_percent = 20)
  : final_increase_percent = 140 := by
  sorry

end percentage_increase_after_initial_decrease_l271_271978


namespace sticks_form_equilateral_triangle_l271_271709

theorem sticks_form_equilateral_triangle (n : ℕ) :
  (∃ (s : set ℕ), s = {1, 2, ..., n} ∧ ∃ t : ℕ → ℕ, t ∈ s ∧ t s ∩ { a, b, c} = ∅ 
   and t t  = t and a + b + {
    sum s = nat . multiplicity = S
}) <=> n

example: ℕ in {
  n ≥ 5 ∧ n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5 } equilateral triangle (sorry)
   sorry


Sorry  = fiancé in the propre case, {1,2, ..., n}

==> n.g.e. : natal sets {1, 2, ... n }  + elastical


end sticks_form_equilateral_triangle_l271_271709


namespace subtract_real_numbers_l271_271298

theorem subtract_real_numbers : 3.56 - 1.89 = 1.67 :=
by
  sorry

end subtract_real_numbers_l271_271298


namespace pages_for_spreads_to_ads_l271_271625

-- Definitions of constants based on the conditions
def single_page_spreads : Nat := 20
def double_page_spreads : Nat := 2 * single_page_spreads
def pages_per_brochure : Nat := 5
def number_of_brochures : Nat := 25
def ads_per_block : Nat := 4
def ad_page_fraction : Rational := 1 / 4

-- Definition to calculate the total pages for brochures
def total_pages_needed : Nat := number_of_brochures * pages_per_brochure

-- Definition to calculate the pages provided by double-page spreads
def pages_from_double_spreads : Nat := double_page_spreads * 2

-- Definition to calculate remaining pages needed from single-page spreads
def remaining_pages_needed : Nat := total_pages_needed - pages_from_double_spreads
def remaining_single_pages_spreads : Nat := single_page_spreads - remaining_pages_needed

-- The theorem to prove the number of pages printed for the spreads for each block of 4 ads
theorem pages_for_spreads_to_ads : remaining_single_pages_spreads = 15 := by
  -- Proof will be provided here
  sorry

end pages_for_spreads_to_ads_l271_271625


namespace largest_binomial_term_l271_271446

theorem largest_binomial_term :
  let expr := (2 * x - 1 / 2) ^ 6
  (expr.coeff (2 * x)^3 * (-1 / 2)^3 : Polynomial ℚ) = -20 * x^3 :=
by
  -- sorry is used to denote that the proof is required but not provided here.
  sorry

end largest_binomial_term_l271_271446


namespace jason_total_payment_l271_271462

def total_cost (shorts jacket shoes socks tshirts : ℝ) : ℝ :=
  shorts + jacket + shoes + socks + tshirts

def discount_amount (total : ℝ) (discount_rate : ℝ) : ℝ :=
  total * discount_rate

def total_after_discount (total discount : ℝ) : ℝ :=
  total - discount

def sales_tax_amount (total : ℝ) (tax_rate : ℝ) : ℝ :=
  total * tax_rate

def final_amount (total after_discount tax : ℝ) : ℝ :=
  after_discount + tax

theorem jason_total_payment :
  let shorts := 14.28
  let jacket := 4.74
  let shoes := 25.95
  let socks := 6.80
  let tshirts := 18.36
  let discount_rate := 0.15
  let tax_rate := 0.07
  let total := total_cost shorts jacket shoes socks tshirts
  let discount := discount_amount total discount_rate
  let after_discount := total_after_discount total discount
  let tax := sales_tax_amount after_discount tax_rate
  let final := final_amount total after_discount tax
  final = 63.78 :=
by
  sorry

end jason_total_payment_l271_271462


namespace line_MN_tangent_to_inscribed_sphere_l271_271738

-- Definitions to initialize the cube and the inscribed sphere
variables {A B C D A₁ B₁ C₁ D₁ : Point}
variables {sphere : Sphere}
variables {plane : Plane}
variables (M N T : Point)

-- Conditions
def is_cube := 
  -- Definition ensuring given points form a cube (details omitted for brevity)
  sorry

def tangent_plane (plane : Plane) (sphere : Sphere) (point : Point) :=
  -- Definition ensuring given plane is tangent to sphere at the specified point
  sorry

def intersection_with_lines A₁ A (plane : Plane) B D : (Point × Point) :=
  -- Definition ensuring plane intersects lines A₁B and A₁D at points M and N
  sorry

-- Main theorem
theorem line_MN_tangent_to_inscribed_sphere (H1 : is_cube A B C D A₁ B₁ C₁ D₁)
  (H2 : tangent_plane plane sphere A)
  (H3 : (M, N) = intersection_with_lines A₁ A plane B D):
  tangent_line sphere (line_through M N) :=
begin
  sorry
end

end line_MN_tangent_to_inscribed_sphere_l271_271738


namespace solution_set_inequality_l271_271343

noncomputable def solution_set (x : ℝ) : Prop :=
  (2 * x - 1) / (x + 2) > 1

theorem solution_set_inequality :
  { x : ℝ | solution_set x } = { x : ℝ | x < -2 ∨ x > 3 } := by
  sorry

end solution_set_inequality_l271_271343


namespace angle_conversion_l271_271669

theorem angle_conversion (deg : ℝ) : (deg = -300) → (deg * (Real.pi / 180) = -5 * Real.pi / 3) :=
by 
  intro h_deg,
  rw h_deg,
  simp,
  linarith

end angle_conversion_l271_271669


namespace find_phi_monotonic_intervals_translation_for_odd_function_notation_l271_271386

noncomputable def A : ℝ := 1
def phi1 : ℝ := π / 3
def phi2 : ℝ := 2 * π / 3
def f (x : ℝ) : ℝ := A * Real.sin (2 * x + phi1)

theorem find_phi (hA_pos : A > 0) (hphi_range : 0 < phi1 ∧ phi1 < π) :
  ∃ phi, φ1 = π / 3 := by
  sorry

theorem monotonic_intervals (k: ℤ) :
  f(x) is increasing on the intervals [k * π - 5 * π / 12, k * π + π / 12] := by
  sorry

theorem translation_for_odd_function_notation :
  by f(x) = Real.sin(2 * (x + π / 6))
  ∃ g (translation : ℝ), g(x) = Real.sin 2x ∧ translation = π / 6 := by
  sorry

end find_phi_monotonic_intervals_translation_for_odd_function_notation_l271_271386


namespace toothpicks_in_100th_stage_l271_271078

theorem toothpicks_in_100th_stage : 
  let num_toothpicks (n : ℕ) := 5 + (n - 1) * 4 in 
  num_toothpicks 100 = 401 :=
by 
  let num_toothpicks (n : ℕ) := 5 + (n - 1) * 4 
  have h : num_toothpicks 100 = 5 + 99 * 4 := rfl 
  show 5 + 99 * 4 = 401
  sorry

end toothpicks_in_100th_stage_l271_271078


namespace pipe_A_filling_time_l271_271995

theorem pipe_A_filling_time (A : ℝ) : 
  (∀ t : ℝ, 
   let rate_A := 1 / A,
       rate_B := 1 / 32,
       combined_rate := rate_A + rate_B,
       volume_first_8_minutes := 8 * combined_rate,
       volume_next_10_minutes := 10 * rate_A,
       full_tank_volume := 1 in
   volume_first_8_minutes + volume_next_10_minutes = full_tank_volume) →
  A = 24 :=
sorry

end pipe_A_filling_time_l271_271995


namespace train_crossing_time_l271_271287

/--
A train requires 8 seconds to pass a pole while it requires some seconds to cross a stationary train which is 400 meters long. 
The speed of the train is 144 km/h. Prove that it takes 18 seconds for the train to cross the stationary train.
-/
theorem train_crossing_time
  (train_speed_kmh : ℕ)
  (time_to_pass_pole : ℕ)
  (length_stationary_train : ℕ)
  (speed_mps : ℕ)
  (length_moving_train : ℕ)
  (total_length : ℕ)
  (crossing_time : ℕ) :
  train_speed_kmh = 144 →
  time_to_pass_pole = 8 →
  length_stationary_train = 400 →
  speed_mps = (train_speed_kmh * 1000) / 3600 →
  length_moving_train = speed_mps * time_to_pass_pole →
  total_length = length_moving_train + length_stationary_train →
  crossing_time = total_length / speed_mps →
  crossing_time = 18 :=
by
  intros;
  sorry

end train_crossing_time_l271_271287


namespace correct_statement_l271_271416

-- We assume the existence of lines and planes with certain properties.
variables {Line : Type} {Plane : Type}
variables {m n : Line} {alpha beta gamma : Plane}

-- Definitions for perpendicular and parallel relations
def perpendicular (p1 p2 : Plane) : Prop := sorry
def parallel (p1 p2 : Plane) : Prop := sorry
def line_perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry
def line_parallel_to_plane (l : Line) (p : Plane) : Prop := sorry

-- The theorem we aim to prove given the conditions
theorem correct_statement :
  line_perpendicular_to_plane m beta ∧ line_parallel_to_plane m alpha → perpendicular alpha beta :=
by sorry

end correct_statement_l271_271416


namespace _l271_271749

@[simp] theorem upper_base_length (ABCD is_trapezoid: Boolean)
  (point_M: Boolean)
  (perpendicular_DM_AB: Boolean)
  (MC_eq_CD: Boolean)
  (AD_eq_d: ℝ)
  : BC = d / 2 := sorry

end _l271_271749


namespace mrs_smith_additional_money_needed_l271_271152

noncomputable def dresses_budget := 300
noncomputable def shoes_budget := 150
noncomputable def accessories_budget := 50
noncomputable def extra_fraction := 2 / 5
noncomputable def discount_dresses := 0.20
noncomputable def discount_shoes := 0.10
noncomputable def discount_accessories := 0.15
noncomputable def available_money := 500

noncomputable def additional_money_needed (dresses_budget shoes_budget accessories_budget : ℝ) 
(extra_fraction discount_dresses discount_shoes discount_accessories available_money : ℝ) : ℝ :=
let total_dresses := dresses_budget * (1 + extra_fraction) * (1 - discount_dresses)
    total_shoes := shoes_budget * (1 + extra_fraction) * (1 - discount_shoes)
    total_accessories := accessories_budget * (1 + extra_fraction) * (1 - discount_accessories)
    total_needed := total_dresses + total_shoes + total_accessories
in total_needed - available_money

theorem mrs_smith_additional_money_needed :
  additional_money_needed dresses_budget shoes_budget accessories_budget 
  extra_fraction discount_dresses discount_shoes discount_accessories available_money = 84.50 := 
by
  sorry

end mrs_smith_additional_money_needed_l271_271152


namespace general_formula_an_sum_bn_l271_271119

-- Define the sequence a_n and conditions
def seq_a (n : ℕ) : ℕ := 2 * n + 1

def seq_S (n : ℕ) : ℕ := ∑ i in range (n + 1), seq_a i

-- Condition: a_n > 0
lemma an_pos (n : ℕ) : (seq_a n) > 0 := by
  sorry

-- Condition: a_n^2 + 2a_n = 4S_n + 3
lemma an_eq (n : ℕ) : (seq_a n) ^ 2 + 2 * (seq_a n) = 4 * (seq_S n) + 3 := by
  sorry

-- Define the sequence b_n
def seq_b (n : ℕ) : ℕ := 1 / ((seq_a n) * (seq_a (n + 1)))

-- The general formula for {a_n}
theorem general_formula_an (n : ℕ) : seq_a n = 2 * n + 1 :=
  by sorry

-- The sum of the first n terms of sequence {b_n}
theorem sum_bn (n : ℕ) : (∑ i in range (n + 1), seq_b i) = n / (3 * (2 * n + 3)) :=
  by sorry

end general_formula_an_sum_bn_l271_271119


namespace only_possible_limit_value_l271_271134

open_locale topological_space

noncomputable def possible_value_of_limit (ϕ : ℕ → ℕ) (L : ℝ) : Prop :=
  bijective ϕ ∧ (tendsto (λ n, (ϕ n : ℝ) / n) at_top (𝓝 L)) → L = 1

theorem only_possible_limit_value (ϕ : ℕ → ℕ) (L : ℝ) :
  bijective ϕ →
  tendsto (λ n, (ϕ n : ℝ) / n) at_top (𝓝 L) →
  L = 1 :=
begin
  intros h_bij h_tendsto,
  apply possible_value_of_limit ϕ L,
  split,
  { exact h_bij },
  { exact h_tendsto }
end

end only_possible_limit_value_l271_271134


namespace midpoint_set_characterization_l271_271880

structure Line := 
  (point : ℝ × ℝ × ℝ) 
  (direction : ℝ × ℝ × ℝ)

def is_parallel (a b : Line) : Prop :=
  ∃ k : ℝ, b.direction = (k • a.direction)

def set_of_midpoints (a b : Line) : set (ℝ × ℝ × ℝ) :=
  {midpoint : ℝ × ℝ × ℝ | 
    ∃ (p₁ p₂ : ℝ × ℝ × ℝ), 
    (∃ t₁ t₂ : ℝ, p₁ = (a.point.1 + t₁ * a.direction.1, a.point.2 + t₁ * a.direction.2, a.point.3 + t₁ * a.direction.3) ∧
                      p₂ = (b.point.1 + t₂ * b.direction.1, b.point.2 + t₂ * b.direction.2, b.point.3 + t₂ * b.direction.3)) ∧
    midpoint = ((p₁.1 + p₂.1) / 2, (p₁.2 + p₂.2) / 2, (p₁.3 + p₂.3) / 2)}

theorem midpoint_set_characterization (a b : Line) : 
  (is_parallel a b → 
    ∃ l : Line, set_of_midpoints a b = {p | ∃ t : ℝ, p = (l.point.1 + t * l.direction.1, l.point.2 + t * l.direction.2, l.point.3 + t * l.direction.3)}) ∧
  (¬ is_parallel a b → 
    ∃ P : ℝ × ℝ × ℝ × ℝ × ℝ × ℝ, set_of_midpoints a b = {p | ∃ α β : ℝ, p = (P.1 + α * P.4 + β * P.3, P.2 + α * P.5 + β * P.4, P.3 + α * P.6 + β * P.5)}) :=
by
  sorry

end midpoint_set_characterization_l271_271880


namespace pa_pb_pc_pg_squared_constant_l271_271468

noncomputable def is_centroid (G A B C : Point) : Prop := 
  G = (A + B + C) / 3

noncomputable def on_circumcircle (P A B C : Point) (R : ℝ) : Prop :=
  dist P center = R where center := circumcenter A B C

theorem pa_pb_pc_pg_squared_constant (A B C P : Point) (a b c R : ℝ) (G : Point) 
  (hG : is_centroid G A B C) (hP : on_circumcircle P A B C R) : 
  PA^2 + PB^2 + PC^2 - PG^2 = (14 * R^2) / 3 :=
begin
  sorry -- Proof omitted
end

end pa_pb_pc_pg_squared_constant_l271_271468


namespace product_of_positive_solutions_l271_271719

theorem product_of_positive_solutions :
  ∃ n : ℕ, ∃ p : ℕ, Prime p ∧ (n^2 - 41*n + 408 = p) ∧ (∀ m : ℕ, (Prime p ∧ (m^2 - 41*m + 408 = p)) → m = n) ∧ (n = 406) := 
sorry

end product_of_positive_solutions_l271_271719


namespace find_parallel_lines_l271_271716

-- Define the original line equation as a condition
def original_line (x y : ℝ) : Prop := 2 * x + y - 3 = 0

-- Define the distance condition between two parallel lines
def distance_between_parallel_lines (a b c1 c2 : ℝ) (dist : ℝ) : Prop :=
  dist * real.sqrt (a^2 + b^2) = real.abs (c1 - c2)

-- Main statement to prove
theorem find_parallel_lines :
  ∃ (t₁ t₂ : ℝ), (t₁ = 2 ∧ t₂ = -8) ∧
  distance_between_parallel_lines 2 1 (-3) t₁ (real.sqrt 5) ∧
  distance_between_parallel_lines 2 1 (-3) t₂ (real.sqrt 5) :=
sorry

end find_parallel_lines_l271_271716


namespace measure_angle_E_l271_271485

-- Definitions based on conditions
variables {p q : Type} {A B E : ℝ}

noncomputable def measure_A (A B : ℝ) : ℝ := A
noncomputable def measure_B (A B : ℝ) : ℝ := 9 * A
noncomputable def parallel_lines (p q : Type) : Prop := true

-- Condition: measure of angle A is 1/9 of the measure of angle B
axiom angle_condition : A = (1 / 9) * B

-- Condition: p is parallel to q
axiom parallel_condition : parallel_lines p q

-- Prove that the measure of angle E is 18 degrees
theorem measure_angle_E (y : ℝ) (h1 : A = y) (h2 : B = 9 * y) : E = 18 :=
by
  sorry

end measure_angle_E_l271_271485


namespace find_B_min_fraction_of_squares_l271_271912

-- Lean 4 statement for part (1)
theorem find_B (A B C a b c : ℝ) (h_triangle : A + B + C = π) 
(h_sides : a > 0 ∧ b > 0 ∧ c > 0) 
(h_identity : (cos A) / (1 + sin A) = (sin (2 * B)) / (1 + cos (2 * B))) 
(h_C : C = 2 * π / 3) : 
B = π / 6 := sorry

-- Lean 4 statement for part (2)
theorem min_fraction_of_squares (A B C a b c : ℝ) 
(h_triangle : A + B + C = π) 
(h_sides : a > 0 ∧ b > 0 ∧ c > 0) 
(h_identity : (cos A) / (1 + sin A) = (sin (2 * B)) / (1 + cos (2 * B))) 
(h_C : C = 2 * π / 3) : 
∀ a b c, ∃ m, m = 4 * sqrt 2 - 5 ∧ (a^2 + b^2) / c^2 = m := sorry

end find_B_min_fraction_of_squares_l271_271912


namespace transform_flag_l271_271655

structure Flag :=
  (width height : ℕ)
  (upper_color : String)
  (lower_color : String)

structure NewFlag :=
  (width height : ℕ)
  (left_color : String)
  (right_color : String)

theorem transform_flag (old_flag : Flag) (new_flag : NewFlag) 
  (h_old : old_flag.upper_color = "red" ∧ old_flag.lower_color = "yellow")
  (h_new : new_flag.left_color = "red" ∧ new_flag.right_color = "yellow") :
  (old_flag.width = new_flag.width) ∧ (old_flag.height = new_flag.height) ∧ 
  ∃ pieces : list (String × String), 
    list.length pieces = 4 ∧
    pieces.nth 0 = some ("red", "top-left") ∧
    pieces.nth 1 = some ("red", "top-right") ∧
    pieces.nth 2 = some ("yellow", "bottom-left") ∧
    pieces.nth 3 = some ("yellow", "bottom-right") ∧
    (new_flag.left_color = pieces.nth 0.iget.fst ∧ new_flag.right_color = pieces.nth 1.iget.fst) :=
sorry

end transform_flag_l271_271655


namespace quadrilateral_equality_l271_271442

-- Variables definitions for points and necessary properties
variables {A B C D : Type*} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]

-- Assumptions based on given conditions
variables (AB : ℝ) (AD : ℝ) (BC : ℝ) (DC : ℝ) (beta : ℝ)
variables {angleB : ℝ} {angleD : ℝ}

-- Given conditions
axiom AB_eq_AD : AB = AD
axiom angleB_eq_angleD : angleB = angleD

-- The statement to be proven
theorem quadrilateral_equality (h1 : AB = AD) (h2 : angleB = angleD) : BC = DC :=
by
  sorry

end quadrilateral_equality_l271_271442


namespace magnitude_of_linear_combination_l271_271484

variables {V : Type*} [inner_product_space ℝ V]

theorem magnitude_of_linear_combination
  (a b : V) 
  (ha : ∥a∥ = 2) 
  (hb : ∥b∥ = 3) 
  (hab : ⟪a, b⟫ = 3) :
  ∥3 • a - 2 • b∥ = 6 :=
by
  sorry

end magnitude_of_linear_combination_l271_271484


namespace part1_solution_set_part2_range_of_a_l271_271021

def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1_solution_set (x : ℝ) : 
  (f x 2 ≥ 4) ↔ (x ≤ 3 / 2 ∨ x ≥ 11 / 2) :=
sorry

theorem part2_range_of_a (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 4) → (a ≤ -1 ∨ a ≥ 3) :=
sorry

end part1_solution_set_part2_range_of_a_l271_271021


namespace megatek_manufacturing_percentage_l271_271255

theorem megatek_manufacturing_percentage (total_degrees manufacturing_degrees : ℝ)
    (h_proportional : total_degrees = 360)
    (h_manufacturing_degrees : manufacturing_degrees = 180) :
    (manufacturing_degrees / total_degrees) * 100 = 50 := by
  -- The proof will go here.
  sorry

end megatek_manufacturing_percentage_l271_271255


namespace extra_taxes_paid_l271_271613

variables (initial_tax_rate new_tax_rate initial_income new_income : ℝ)

-- Conditions
def initial_tax_rate := 0.2
def new_tax_rate := 0.3
def initial_income := 1000000
def new_income := 1500000

-- The statement about the extra taxes paid
theorem extra_taxes_paid :
  (new_tax_rate * new_income) - (initial_tax_rate * initial_income) = 250000 :=
by
  sorry

end extra_taxes_paid_l271_271613


namespace gloria_coins_l271_271050

theorem gloria_coins (qd qda qdc : ℕ) (h1 : qdc = 350) (h2 : qda = qdc / 5) (h3 : qd = qda - (2 * qda / 5)) :
  qd + qdc = 392 :=
by sorry

end gloria_coins_l271_271050


namespace line_passes_through_vertex_parabola_l271_271352

theorem line_passes_through_vertex_parabola : {b : ℝ // (y = x + b).passes_the_vertex_of (y = x^2 + b^2)}.card = 2 :=
by
  sorry

end line_passes_through_vertex_parabola_l271_271352


namespace four_points_nonexistent_l271_271677

theorem four_points_nonexistent :
  ¬ (∃ (A B C D : ℝ × ℝ × ℝ), 
    dist A B = 8 ∧ 
    dist C D = 8 ∧ 
    dist A C = 10 ∧ 
    dist B D = 10 ∧ 
    dist A D = 13 ∧ 
    dist B C = 13) :=
by
  sorry

end four_points_nonexistent_l271_271677


namespace interior_angle_of_regular_pentagon_is_108_l271_271566

-- Define the sum of angles in a triangle
def sum_of_triangle_angles : ℕ := 180

-- Define the number of triangles in a convex pentagon
def num_of_triangles_in_pentagon : ℕ := 3

-- Define the total number of interior angles in a pentagon
def num_of_angles_in_pentagon : ℕ := 5

-- Define the total sum of the interior angles of a pentagon
def sum_of_pentagon_interior_angles : ℕ := num_of_triangles_in_pentagon * sum_of_triangle_angles

-- Define the degree measure of an interior angle of a regular pentagon
def interior_angle_of_regular_pentagon : ℕ := sum_of_pentagon_interior_angles / num_of_angles_in_pentagon

theorem interior_angle_of_regular_pentagon_is_108 :
  interior_angle_of_regular_pentagon = 108 :=
by
  -- Proof will be filled in here
  sorry

end interior_angle_of_regular_pentagon_is_108_l271_271566


namespace collinear_points_l271_271421

theorem collinear_points (a : ℝ) : 
  let A := (4 : ℝ, 3 : ℝ)
  let B := (5 : ℝ, a)
  let C := (6 : ℝ, 5 : ℝ)
  (a - 3) = (5 - a) →
  a = 4 :=
by
  sorry

end collinear_points_l271_271421


namespace profit_percentage_is_correct_l271_271596

-- Define the conditions
variables (market_price_per_pen : ℝ) (discount_percentage : ℝ) (total_pens_bought : ℝ) (cost_pens_market_price : ℝ)
variables (cost_price_per_pen : ℝ) (selling_price_per_pen : ℝ) (profit_per_pen : ℝ) (profit_percent : ℝ)

-- Conditions
def condition_1 : market_price_per_pen = 1 := by sorry
def condition_2 : discount_percentage = 0.01 := by sorry
def condition_3 : total_pens_bought = 80 := by sorry
def condition_4 : cost_pens_market_price = 36 := by sorry

-- Definitions based on conditions
def cost_price_per_pen_def : cost_price_per_pen = cost_pens_market_price / total_pens_bought := by sorry
def selling_price_per_pen_def : selling_price_per_pen = market_price_per_pen * (1 - discount_percentage) := by sorry
def profit_per_pen_def : profit_per_pen = selling_price_per_pen - cost_price_per_pen := by sorry
def profit_percent_def : profit_percent = (profit_per_pen / cost_price_per_pen) * 100 := by sorry

-- The statement to prove
theorem profit_percentage_is_correct : profit_percent = 120 :=
by
  have h1 : cost_price_per_pen = 36 / 80 := by sorry
  have h2 : selling_price_per_pen = 1 * (1 - 0.01) := by sorry
  have h3 : profit_per_pen = 0.99 - 0.45 := by sorry
  have h4 : profit_percent = (0.54 / 0.45) * 100 := by sorry
  sorry

end profit_percentage_is_correct_l271_271596


namespace max_knights_with_courtiers_l271_271456

open Nat

def is_solution (a b : ℕ) : Prop :=
  12 ≤ a ∧ a ≤ 18 ∧
  10 ≤ b ∧ b ≤ 20 ∧
  1 / (a:ℝ).toNat + 1 / (b:ℝ).toNat = 1 / 7

theorem max_knights_with_courtiers : ∃ a b : ℕ, is_solution a b ∧ b = 14 ∧ a = 14 :=
by
  sorry

end max_knights_with_courtiers_l271_271456


namespace oldest_bride_age_l271_271544

theorem oldest_bride_age (B G : ℕ) (h1 : B = G + 19) (h2 : B + G = 185) :
  B = 102 :=
by
  sorry

end oldest_bride_age_l271_271544


namespace line_equation_l271_271717

theorem line_equation 
    (passes_through_intersection : ∃ (P : ℝ × ℝ), P ∈ { (x, y) | 11 * x + 3 * y - 7 = 0 } ∧ P ∈ { (x, y) | 12 * x + y - 19 = 0 })
    (equidistant_from_A_and_B : ∃ (P : ℝ × ℝ), dist P (3, -2) = dist P (-1, 6)) :
    ∃ (a b c : ℝ), (a = 7 ∧ b = 1 ∧ c = -9) ∨ (a = 2 ∧ b = 1 ∧ c = 1) ∧ ∀ (x y : ℝ), a * x + b * y + c = 0 := 
sorry

end line_equation_l271_271717


namespace initial_speed_l271_271265

theorem initial_speed (v : ℝ) (v₀ : ℝ) (h₁ : v = 40) (h₂ : sqrt(2 * v^2) = v₀) : v₀ = 56.6 :=
by
  admit

end initial_speed_l271_271265


namespace intersecting_segments_l271_271946

open Set

def circle_points (n : ℕ) : Prop :=
  ∃ (points : Finset ℝ) (red blue : Finset ℝ),
    points.card = 4 * n ∧
    red ∪ blue = points ∧
    red ∩ blue = ∅ ∧
    red.card = 2 * n ∧
    blue.card = 2 * n ∧
    (∀ (r1 r2 : ℝ), r1 ∈ red → r2 ∈ red → r1 ≠ r2 → Segment r1 r2 ∩ {p | ∃ r, r ∈ red ∪ blue ∧ p = Segment r1 r2} = ∅) ∧
    (∀ (b1 b2 : ℝ), b1 ∈ blue → b2 ∈ blue → b1 ≠ b2 → Segment b1 b2 ∩ {p | ∃ r, r ∈ red ∪ blue ∧ p = Segment b1 b2} = ∅)

theorem intersecting_segments (n : ℕ) (h : circle_points n) : ∃ k : ℕ, k ≥ n ∧
  ∀ (redSeg : Finset (ℝ × ℝ)) (blueSeg : Finset (ℝ × ℝ)),
    redSeg.card = n ∧
    blueSeg.card = n ∧
    redSeg ⊆ {Segment r1 r2 | r1 r2 ∈ Finset.filter (∈ red ∧ ∈ blue) (Finset.pairs (red ∪ blue))} ∧
    blueSeg ⊆ {Segment b1 b2 | b1 b2 ∈ Finset.filter (∈ red ∧ ∈ blue) (Finset.pairs (red ∪ blue))} →
      ∃ i ∈ redSeg, ∃ j ∈ blueSeg, Segment.intersect i = Segment.intersect j :=
by sorry

end intersecting_segments_l271_271946


namespace log_expression_simplification_l271_271664

theorem log_expression_simplification (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (log x ^ 2 / log (y ^ 8)) * (log (y ^ 3) / log (x ^ 4)) * (log (x ^ 4) / log (y ^ 5)) * (log (y ^ 5) / log (x ^ 2)) * (log (x ^ 4) / log (y ^ 3)) =
  (1/2) * (log y x) :=
by
  sorry

end log_expression_simplification_l271_271664


namespace max_sum_of_factors_l271_271605

theorem max_sum_of_factors (h k : ℕ) (h_even : Even h) (prod_eq : h * k = 24) : h + k ≤ 14 :=
sorry

end max_sum_of_factors_l271_271605


namespace exists_subset_satisfying_condition_l271_271883

theorem exists_subset_satisfying_condition (n : ℕ) (r : Fin n -> ℝ) :
  ∃ S : Finset (Fin n), 
  (∀ i : Fin (n-2), (S ∩ {i, i+1, i+2}).card = 1 ∨ (S ∩ {i, i+1, i+2}).card = 2) ∧
  |∑ i in S, r i| ≥ (1 / 6) * ∑ i in Finset.univ, |r i| :=
sorry

end exists_subset_satisfying_condition_l271_271883


namespace find_positive_integer_n_l271_271671

noncomputable def cube_root_condition (n : ℕ) : Prop :=
  let a := n / 1000 in  -- the integer part after removing the last three digits
  let b := n % 1000 in  -- the last three decimal digits
  a = Nat.cbrt n ∧ n = a * 1000 + b  -- the two defining conditions

theorem find_positive_integer_n :
  ∃! (n : ℕ), n > 0 ∧ cube_root_condition n := by
  sorry

end find_positive_integer_n_l271_271671


namespace find_expression_l271_271390

-- Given the function f such that for all x, f(x+1) = 3x + 1
def satisfies_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(x + 1) = 3 * x + 1

-- We need to prove that f(x) = 3x - 2
theorem find_expression (f : ℝ → ℝ) (h : satisfies_function f) : 
  ∀ x : ℝ, f(x) = 3 * x - 2 :=
sorry

end find_expression_l271_271390


namespace trapezoid_problem_l271_271225

theorem trapezoid_problem
  (EF GH EH FH EG FH : ℝ)
  (JK : ℝ)
  (h1 : EF = 72) -- Since EF = 2*GH due to parallel lines and same length segments making it 72
  (h2 : GH = FG = 36)
  (h3 : EH = (6 * (Real.sqrt 369)))
  (h4 : EH ^ 2 + FH ^ 2 = EF ^ 2)
  (h5 : JK = 15)
  (h6 : FH = 90)
  (h7 : ∃ J K, J = (EG ∩ FH) ∧ K = midpoint FH ∧ JK = 15) :
  EH = 6 * Real.sqrt 369 :=
by
  sorry

end trapezoid_problem_l271_271225


namespace apples_minimum_count_l271_271548

theorem apples_minimum_count :
  ∃ n : ℕ, n ≡ 2 [MOD 3] ∧ n ≡ 2 [MOD 4] ∧ n ≡ 2 [MOD 5] ∧ n = 62 := by
sorry

end apples_minimum_count_l271_271548


namespace part1_solution_set_part2_values_of_a_l271_271031

-- Parameters and definitions for the function and conditions
def f (x a : ℝ) := |x - a^2| + |x - (2*a + 1)|

-- Part (1) When a = 2, the inequality's solution set
theorem part1_solution_set (x : ℝ) : f x 2 ≥ 4 ↔ (x ≤ 3/2 ∨ x ≥ 11/2) := 
sorry

-- Part (2) Range of values for a given f(x, a) ≥ 4
theorem part2_values_of_a (a : ℝ) : 
∀ x, f x a ≥ 4 → (a ≤ -1 ∨ a ≥ 3) :=
sorry

end part1_solution_set_part2_values_of_a_l271_271031


namespace find_A_max_area_l271_271853

-- Define the given conditions for the first part of the problem
variables {A B C a b c : ℝ}

-- Define an acute triangle condition 
def is_acute (A B C : ℝ) : Prop := 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2

-- Define the initial condition 
def initial_condition (a b c A C : ℝ) : Prop := 
  (b^2 - a^2 - c^2) * sin A * cos A = a * c * cos (A + C)

-- State the theorem to find angle A
theorem find_A (h_acute : is_acute A B C) (h_cond : initial_condition a b c A C) : 
  A = π / 4 := 
sorry

-- Define another condition for part 2 when a = sqrt(2)
def side_length_condition (b c : ℝ) := 
  (b^2 + c^2 - (sqrt 2) * b * c = 2)

-- Define the function to calculate the area of the triangle
def triangle_area (b c : ℝ) (A : ℝ) : ℝ := 1 / 2 * b * c * sin A

-- State the theorem for maximum area when a = sqrt(2)
theorem max_area (h_acute : is_acute A B C) (h_A : A = π / 4) (h_side : side_length_condition b c) : 
  ∃ b c, triangle_area b c A ≤ (sqrt 2 + 1) / 2 :=
sorry

end find_A_max_area_l271_271853


namespace smallest_four_digit_square_with_digits_1_to_9_l271_271561

theorem smallest_four_digit_square_with_digits_1_to_9 :
  ∃ (a b c : ℕ), 
    ( (a * a).digits == 2 ) ∧ 
    ( (b * b).digits == 3 ) ∧ 
    ( (c * c).digits == 4 ) ∧ 
    ( a * a + b * b + c * c == digits_combination ) ∧ 
    ( c * c == 1369 ) := 
  sorry

end smallest_four_digit_square_with_digits_1_to_9_l271_271561


namespace digit_difference_l271_271054

theorem digit_difference (n : ℕ) : 
  let num_base10 := 1600 in 
  let num_base4_digits := Nat.find (λ m, num_base10 < 4^m) in
  let num_base7_digits := Nat.find (λ m, num_base10 < 7^m) in
  num_base4_digits - num_base7_digits = 2 :=
by
  let num_base10 := 1600
  let num_base4_digits := Nat.find (λ m, num_base10 < 4^m)
  let num_base7_digits := Nat.find (λ m, num_base10 < 7^m)
  sorry

end digit_difference_l271_271054


namespace symmetric_function_l271_271388

-- Define the function f(x) = a^x
def f (a : ℝ) (x : ℝ) : ℝ := a^x

-- Define the inverse function g(x) = log_a(x)
def g (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Given conditions
variable (a : ℝ)
variable (h1 : 0 < a)
variable (h2 : a ≠ 1)
variable (h3 : f a 2 = 9)

theorem symmetric_function (hf : f a 3 = 27) : g a (1/9) + f a 3 = 25 := by
  sorry

end symmetric_function_l271_271388


namespace complement_of_A_in_U_l271_271047

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 4}
def complement_set (U A : Set ℕ) : Set ℕ := U \ A

theorem complement_of_A_in_U :
  complement_set U A = {2, 3, 5} :=
by
  apply Set.ext
  intro x
  simp [complement_set, U, A]
  sorry

end complement_of_A_in_U_l271_271047


namespace find_magnitude_of_w_l271_271884

noncomputable def magnitude_of_w (w : ℂ) (h : w^2 = -48 + 14 * complex.I) : ℝ :=
  complex.abs w

theorem find_magnitude_of_w (w : ℂ) (h : w^2 = -48 + 14 * complex.I) : magnitude_of_w w h = 5 * real.sqrt 2 := by
  sorry

end find_magnitude_of_w_l271_271884


namespace no_real_solution_f_eq_3_l271_271892

def f (x : ℝ) : ℝ :=
  if x < -3 then 3 * x - 7
  else - x^2 + 3 * x - 3

theorem no_real_solution_f_eq_3 : ¬∃ x : ℝ, f x = 3 :=
by
  sorry

end no_real_solution_f_eq_3_l271_271892


namespace otherWorkStations_accommodate_students_l271_271267

def numTotalStudents := 38
def numStations := 16
def numWorkStationsForTwo := 10
def capacityWorkStationsForTwo := 2

theorem otherWorkStations_accommodate_students : 
  (numTotalStudents - numWorkStationsForTwo * capacityWorkStationsForTwo) = 18 := 
by
  sorry

end otherWorkStations_accommodate_students_l271_271267


namespace part1_part2_l271_271015

def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1 (x : ℝ) : 
  ∀ x, f x 2 ≥ 4 ↔ (x ≤ 3 / 2 ∨ x ≥ 11 / 2) :=
by sorry

theorem part2 (x a : ℝ) :
  f x a ≥ 4 → (a ≤ -1 ∨ a ≥ 3) :=
by sorry

end part1_part2_l271_271015


namespace sum_of_solutions_correct_l271_271576

noncomputable def sum_of_solutions : ℕ :=
  { 
    val := (List.range' 1 30 / 2).sum,
    property := sorry
  }

theorem sum_of_solutions_correct :
  (∀ x : ℕ, (x > 0 ∧ x ≤ 30 ∧ (17 * (5 * x - 3)).mod 10 = 34.mod 10) →
    List.mem x ((List.range' 1 30).filter (λ n, n % 2 = 1)) →
    ((List.range' 1 30 / 2).sum = 225)) :=
by {
  intro x h1 h2,
  sorry
}

end sum_of_solutions_correct_l271_271576


namespace percentile_correct_l271_271638

def data_set : List ℕ := [72, 76, 78, 82, 86, 88, 92, 98]

def N : ℕ := data_set.length

def P : ℕ := 80

def percentile_position : ℝ := (P / 100) * N

def 80th_percentile : ℝ :=
  if percentile_position - percentile_position.floor = 0 then 
    data_set[percentile_position.to_nat - 1]
  else 
    data_set[percentile_position.to_nat]  

theorem percentile_correct : 
  80th_percentile = 92 := 
sorry

end percentile_correct_l271_271638


namespace label_baskets_l271_271546

variable {Basket : Type}
variable (A B C D E : Basket)
variable (contains : Basket → ℕ → Prop)

-- Given conditions
axiom hA : contains A 3 ∧ contains A 4
axiom hB : contains B 3 ∧ contains B 4
axiom hC : contains C 2 ∧ contains C 3
axiom hD : contains D 4 ∧ contains D 5
axiom hE : contains E 1 ∧ contains E 5

-- Prove that the baskets can be labeled correctly
theorem label_baskets :
  ∃ (B1 B2 B3 B4 B5 : Basket),
    (contains B1 1) ∧ (contains B2 2) ∧
    (contains B3 3 ∧ contains B3 4) ∧
    (contains B4 3 ∧ contains B4 4) ∧
    (contains B5 5) ∧
    {B1, B2, B3, B4, B5} = {A, B, C, D, E} :=
  sorry

end label_baskets_l271_271546


namespace distance_AC_eq_12_sqrt_13_l271_271226

theorem distance_AC_eq_12_sqrt_13 :
  ∀ (O₁ O₂ A B C : Point) (r₁ r₂ : ℝ), 
    r₁ = 30 → r₂ = 13 → dist O₁ O₂ = 41 →
    on_circle O₂ r₂ A → on_circle O₁ r₁ C →
    midpoint B A C → 
    dist A C = 12 * real.sqrt 13 := 
by 
  -- Proof goes here
  sorry

end distance_AC_eq_12_sqrt_13_l271_271226


namespace red_light_at_A_prob_calc_l271_271216

-- Defining the conditions
def count_total_permutations : ℕ := Nat.factorial 4 / Nat.factorial 1
def count_favorable_permutations : ℕ := Nat.factorial 3 / Nat.factorial 1

-- Calculating the probability
def probability_red_at_A : ℚ := count_favorable_permutations / count_total_permutations

-- Statement to be proved
theorem red_light_at_A_prob_calc : probability_red_at_A = 1 / 4 :=
by
  sorry

end red_light_at_A_prob_calc_l271_271216


namespace minimize_distance_AB_l271_271807

open Real

variable (x : ℝ)

def point_A (x : ℝ) : ℝ × ℝ × ℝ := (x, 5 - x, 2 * x - 1)
def point_B (x : ℝ) : ℝ × ℝ × ℝ := (1, x + 2, 2 - x)

def distance_AB (x : ℝ) : ℝ :=
  let (xA, yA, zA) := point_A x
  let (xB, yB, zB) := point_B x
  sqrt ((xB - xA) ^ 2 + (yB - yA) ^ 2 + (zB - zA) ^ 2)

theorem minimize_distance_AB : ArgMin distance_AB = 8 / 7 :=
  sorry

end minimize_distance_AB_l271_271807


namespace oe_ab_intersect_90_deg_l271_271476

variables {α : ℝ} {O M N A B E : Type}

-- Definitions from the conditions
variables [IsoscelesTriangle O M N] (h1 : ∠ OMN = α)
          [SimilarTriangles (Δ O M N) (Δ O B A)] (h2:  ∠ ABO = α)
          [RightTriangle O M N] (h3 : ∠ EON = π / 2 - α)

-- The theorem stating that OE and AB intersect at a right angle
theorem oe_ab_intersect_90_deg : 
  ∠ (OE ∩ AB) = π / 2 :=
by sorry

end oe_ab_intersect_90_deg_l271_271476


namespace radius_of_base_of_cone_correct_l271_271074

noncomputable def radius_of_base_of_cone (n : ℕ) (r α : ℝ) : ℝ :=
  r * (1 / Real.sin (Real.pi / n) - 1 / Real.tan (Real.pi / 4 + α / 2))

theorem radius_of_base_of_cone_correct :
  radius_of_base_of_cone 11 3 (Real.pi / 6) = 3 / Real.sin (Real.pi / 11) - Real.sqrt 3 :=
by
  sorry

end radius_of_base_of_cone_correct_l271_271074


namespace part1_part2_l271_271906

-- Define the condition for the first part
def condition (A B C : ℝ) : Prop :=
  (C = 2 * Real.pi / 3) ∧ (1 + Real.sin A ≠ 0) ∧ (2 * Real.cos B ≠ 0) ∧ 
  (Real.cos A / (1 + Real.sin A) = Real.sin (2 * B) / (1 + Real.cos (2 * B)))

-- Theorem for the first part: If \( C = \frac{2\pi}{3} \), then \( B = \frac{\pi}{6} \)
theorem part1 (A B C : ℝ) (h : condition A B C) : B = Real.pi / 6 :=
  sorry

-- Define the condition for the second part as the side ratios expression
def ratio_expression (a b c : ℝ) : ℝ :=
  (a^2 + b^2) / (c^2)

-- Theorem for the second part: Minimum value of \(\frac{a^2 + b^2}{c^2}\)
theorem part2 (a b c : ℝ) (A B C : ℝ) 
  (h : condition A B C) : ratio_expression a b c = 4 * Real.sqrt 2 - 5 :=
  sorry

end part1_part2_l271_271906


namespace cross_section_area_perimeter_constant_l271_271657

-- Defining the cube and plane perpendicular to the diagonal AC'
variables {V : Type*} [EuclideanSpace V] (A B C D A' B' C' D' : V)
variables (cube : Cube ABCD A' B' C' D')
variables (alpha : Plane)
variables (S l : Real)

-- Conditions
def plane_perpendicular_diagonal : Prop :=
  Plane.perpendicular alpha (Diagonal AC')

def area_cross_section (alpha : Plane) (S : Real) : Prop :=
  Exists (Polygon W, plane_intersects_cube alpha cube W ∧ Polygon.Area W = S)

def perimeter_cross_section (alpha : Plane) (l : Real) : Prop :=
  Exists (Polygon W, plane_intersects_cube alpha cube W ∧ Polygon.Perimeter W = l)

-- Proof problem statement
theorem cross_section_area_perimeter_constant :
  plane_perpendicular_diagonal alpha →
  (∃ S, area_cross_section alpha S → ∃ l, perimeter_cross_section alpha l) ∧ 
  S ≠ constant ∧ l = constant
:= by
  sorry

end cross_section_area_perimeter_constant_l271_271657


namespace scheduling_courses_constraints_l271_271818

theorem scheduling_courses_constraints :
  ∀ (P : Finset ℕ), 
  (∀ c ∈ P, c ∈ {1, 2, 3, 4}) → 
  ∀ S : Finset ℕ, 
  (∀ p ∈ S, 1 ≤ p ∧ p ≤ 8) → 
  (@Finset.card ℕ {s : ℕ // s ∈ S} = 4) →
  (∀ (a b c : ℕ), a, b, c ∈ S → a ≠ b → b ≠ c → c ≠ a → (a + 1 = b ∧ b + 1 = c) → False) →
  ∃ n, n = 1080 :=
by
  sorry

end scheduling_courses_constraints_l271_271818


namespace find_a_l271_271377

theorem find_a (a : ℝ) (k_l : ℝ) (h1 : k_l = -1)
  (h2 : a ≠ 3) 
  (h3 : (2 - (-1)) / (3 - a) * k_l = -1) : a = 6 :=
by
  sorry

end find_a_l271_271377


namespace circles_internally_tangent_l271_271977

theorem circles_internally_tangent :
  ∀ (x y : ℝ),
  (x - 6)^2 + y^2 = 1 → 
  (x - 3)^2 + (y - 4)^2 = 36 → 
  true := 
by 
  intros x y h1 h2
  sorry

end circles_internally_tangent_l271_271977


namespace find_total_amount_before_brokerage_l271_271186

noncomputable def total_amount_before_brokerage (realized_amount : ℝ) (brokerage_rate : ℝ) : ℝ :=
  realized_amount / (1 - brokerage_rate / 100)

theorem find_total_amount_before_brokerage :
  total_amount_before_brokerage 107.25 (1 / 4) = 107.25 * 400 / 399 := by
sorry

end find_total_amount_before_brokerage_l271_271186


namespace coefficient_of_x7_in_expansion_l271_271070

theorem coefficient_of_x7_in_expansion :
  (∑ k in finset.range 6, 
  binomial 5 k * ↑((2 : ℤ) ^ k) * ↑((-3) : ℤ) ^ (5 - k) * a) = 1 →
  a = 2 →
  (∑ k in finset.range 6, 
  binomial 5 k * ↑((2 : ℤ) ^ (5 - k)) * ↑((-3) : ℤ) ^ k * ↑((2 : ℤ)) ^ (7 - 2 * k - (5 - k))) = -2040 :=
by sorry

end coefficient_of_x7_in_expansion_l271_271070


namespace study_days_needed_l271_271295

theorem study_days_needed :
  let math_chapters := 4
  let math_worksheets := 7
  let physics_chapters := 5
  let physics_worksheets := 9
  let chemistry_chapters := 6
  let chemistry_worksheets := 8

  let math_chapter_hours := 2.5
  let math_worksheet_hours := 1.5
  let physics_chapter_hours := 3.0
  let physics_worksheet_hours := 2.0
  let chemistry_chapter_hours := 3.5
  let chemistry_worksheet_hours := 1.75

  let daily_study_hours := 7.0
  let breaks_first_3_hours := 3 * 10 / 60.0
  let breaks_next_3_hours := 3 * 15 / 60.0
  let breaks_final_hour := 1 * 20 / 60.0
  let snack_breaks := 2 * 20 / 60.0
  let lunch_break := 45 / 60.0

  let break_time_per_day := breaks_first_3_hours + breaks_next_3_hours + breaks_final_hour + snack_breaks + lunch_break
  let effective_study_time_per_day := daily_study_hours - break_time_per_day

  let total_math_hours := (math_chapters * math_chapter_hours) + (math_worksheets * math_worksheet_hours)
  let total_physics_hours := (physics_chapters * physics_chapter_hours) + (physics_worksheets * physics_worksheet_hours)
  let total_chemistry_hours := (chemistry_chapters * chemistry_chapter_hours) + (chemistry_worksheets * chemistry_worksheet_hours)

  let total_study_hours := total_math_hours + total_physics_hours + total_chemistry_hours
  let total_study_days := total_study_hours / effective_study_time_per_day
  
  total_study_days.ceil = 23 := by sorry

end study_days_needed_l271_271295


namespace range_of_a_l271_271397

def A (a : ℝ) : set ℝ := {x | 6 * x + a > 0}

theorem range_of_a (a : ℝ) (h : 1 ∉ A a) : a ≤ -6 :=
by
  -- Here we would write the proof, but we replace it with sorry for now
  sorry

end range_of_a_l271_271397


namespace side_lengths_inequality_l271_271357

variable (O : Type) [metric_space O] [circumference : ∀ (A B C : O), inscribed_triangle A B C → circle O]
variables (A B C : O) (a b c a1 b1 c1 : Real)
variable (h₁ : inscribed_triangle A B C)
variable (h₂ : side_lengths h₁ a b c)
variable (A1 B1 C1 : O)
variable (h₃ : midpoint_arcs A1 B1 C1)
variable (h₄ : inscribed_triangle A1 B1 C1)
variable (h₅ : side_lengths h₄ a1 b1 c1)

theorem side_lengths_inequality :
  ¬(a1 < a ∧ b1 < b ∧ c1 < c) :=
by
  sorry

end side_lengths_inequality_l271_271357


namespace range_f_when_a_1_range_of_a_values_l271_271795

noncomputable def f (x a : ℝ) : ℝ := |x - a| + |x + 4|

theorem range_f_when_a_1 : 
  (∀ x : ℝ, f x 1 ≥ 5) :=
sorry

theorem range_of_a_values :
  (∀ x, f x a ≥ 1) → (a ∈ Set.union (Set.Iic (-5)) (Set.Ici (-3))) :=
sorry

end range_f_when_a_1_range_of_a_values_l271_271795


namespace perpendicular_distance_from_D_to_plane_ABC_is_2_1_l271_271153

-- Definitions of the coordinates of vertices in the problem
def D := (0, 0, 0) : ℝ × ℝ × ℝ
def A := (4, 0, 0) : ℝ × ℝ × ℝ
def B := (0, 4, 0) : ℝ × ℝ × ℝ
def C := (0, 0, 3) : ℝ × ℝ × ℝ

-- Definition of the plane containing points A, B, and C
def plane_ABC : ℝ × ℝ × ℝ := sorry

-- Function to calculate the perpendicular distance from a point to a plane
noncomputable def perpendicular_distance (point : ℝ × ℝ × ℝ) (plane : ℝ × ℝ × ℝ) : ℝ := sorry

-- Proof statement claiming the perpendicular distance is 2.1
theorem perpendicular_distance_from_D_to_plane_ABC_is_2_1 :
  perpendicular_distance D plane_ABC = 2.1 :=
sorry

end perpendicular_distance_from_D_to_plane_ABC_is_2_1_l271_271153


namespace sticks_form_equilateral_triangle_l271_271693

theorem sticks_form_equilateral_triangle (n : ℕ) (h1 : n ≥ 5) (h2 : n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5) :
  ∃ k, k * 3 = (n * (n + 1)) / 2 :=
by
  sorry

end sticks_form_equilateral_triangle_l271_271693


namespace possible_values_for_D_l271_271862

noncomputable def distinct_digit_values (A B C D : Nat) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧ 
  B < 10 ∧ A < 10 ∧ D < 10 ∧ C < 10 ∧ C = 9 ∧ (B + A = 9 + D)

theorem possible_values_for_D :
  ∃ (Ds : Finset Nat), (∀ D ∈ Ds, ∃ A B C, distinct_digit_values A B C D) ∧
  Ds.card = 5 :=
sorry

end possible_values_for_D_l271_271862


namespace collinear_vectors_l271_271398

theorem collinear_vectors (k : ℝ) :
  let OA := (k, 12 : ℝ × ℝ),
      OB := (4, 5 : ℝ × ℝ),
      OC := (-k, 0 : ℝ × ℝ),
      AB := (4 - k, -7),
      AC := (-2 * k, -12) in
  (k ≠ 0) → (∃ λ : ℝ, AB = λ • AC) → k = -24 :=
by
  sorry

end collinear_vectors_l271_271398


namespace oakwood_math_team_ways_l271_271982

theorem oakwood_math_team_ways :
  let girls := 4,
      boys := 6 in 
  combinatorics.choose girls 3 * combinatorics.choose boys 4 = 60 :=
by
  sorry

end oakwood_math_team_ways_l271_271982


namespace inclusion_exclusion_three_events_l271_271167

variables {U : Type} (A B C : set U)

theorem inclusion_exclusion_three_events :
  (set.indicator (A ∪ B ∪ C) (λ _, (1 : ℕ))) = (set.indicator A (λ _, 1)) + (set.indicator B (λ _, 1)) + (set.indicator C (λ _, 1)) - (set.indicator (A ∩ B) (λ _, 1)) - (set.indicator (A ∩ C) (λ _, 1)) - (set.indicator (B ∩ C) (λ _, 1)) + (set.indicator (A ∩ B ∩ C) (λ _, 1)) :=
begin
  sorry
end

end inclusion_exclusion_three_events_l271_271167


namespace point_in_second_quadrant_l271_271830

theorem point_in_second_quadrant (m : ℝ) (h : 2 > 0 ∧ m < 0) : m < 0 :=
by
  sorry

end point_in_second_quadrant_l271_271830


namespace find_range_of_a_l271_271039

noncomputable def range_of_a : Set ℝ := 
  { a : ℝ | (4 - a / 2 > 0) ∧ 
            (a > 1) ∧ 
            ((4 - a / 2) + 2 ≤ a) }

theorem find_range_of_a :
  ∀ (a : ℝ), 
    (∀ x : ℝ, 
      (x ≤ 1 → ((4 - a / 2) * x + 2) ≤ (4 - a / 2) * (x + 1) + 2) ∧
      (x > 1 → a^x ≤ a^(x + 1))) → 
    (4 ≤ a ∧ a < 8) :=
begin
  sorry
end

end find_range_of_a_l271_271039


namespace area_of_triangle_l271_271999

theorem area_of_triangle : 
    let line1 := λ x, 3 * x - 3
    let line2 := λ x, -2 * x + 18
    let y_axis := λ x, 0
    let intersection_point := (21 / 5, 48 / 5)
    let y_intercept_line1 := (0, -3)
    let y_intercept_line2 := (0, 18)
    let base := y_intercept_line2.2 - y_intercept_line1.2
    let height := intersection_point.1
    let area := (base * height) / 2 
in
    area = 44.1 := 
by
    sorry

end area_of_triangle_l271_271999


namespace extra_taxes_paid_l271_271612

variables (initial_tax_rate new_tax_rate initial_income new_income : ℝ)

-- Conditions
def initial_tax_rate := 0.2
def new_tax_rate := 0.3
def initial_income := 1000000
def new_income := 1500000

-- The statement about the extra taxes paid
theorem extra_taxes_paid :
  (new_tax_rate * new_income) - (initial_tax_rate * initial_income) = 250000 :=
by
  sorry

end extra_taxes_paid_l271_271612


namespace unique_digit_puzzle_l271_271329

theorem unique_digit_puzzle :
  ∃ A B C D E : ℕ, 
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧
    C ≠ D ∧ C ≠ E ∧
    D ≠ E ∧
    1 ≤ A ∧ A ≤ 6 ∧
    1 ≤ B ∧ B ≤ 6 ∧
    1 ≤ C ∧ C ≤ 6 ∧
    1 ≤ D ∧ D ≤ 6 ∧
    1 ≤ E ∧ E ≤ 6 ∧
    (100 * A + 10 * B + C) + (10 * C + D) + E = 696 :=
by {
  existsi (6 : ℕ),
  existsi (2 : ℕ),
  existsi (4 : ℕ),
  existsi (6 : ℕ),
  existsi (3 : ℕ),
  exact sorry
}

end unique_digit_puzzle_l271_271329


namespace sticks_forming_equilateral_triangle_l271_271695

theorem sticks_forming_equilateral_triangle (n : ℕ) (h1 : n ≥ 5) :
  (∃ l : list ℕ, (∀ x ∈ l, x ∈ finset.range (n + 1) ∧ x > 0) ∧ l.sum % 3 = 0) ↔ 
  (n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5) := 
sorry

end sticks_forming_equilateral_triangle_l271_271695


namespace part1_solution_set_part2_range_of_a_l271_271004

-- Define the function f for a general a
def f (x a : ℝ) : ℝ := |x - a^2| + |x - (2 * a) + 1|

-- Part 1: When a = 2
theorem part1_solution_set (x : ℝ) (h : f x 2 ≥ 4) : x ≤ 3/2 ∨ x ≥ 11/2 :=
  sorry

-- Part 2: Range of values for a
theorem part2_range_of_a (a : ℝ) (h : ∀ x, f x a ≥ 4) : a ≤ -1 ∨ a ≥ 3 :=
  sorry

end part1_solution_set_part2_range_of_a_l271_271004


namespace angle_f1pf2_l271_271494

theorem angle_f1pf2 (P F1 F2 : ℝ × ℝ)
  (hP : P ∈ {p : ℝ × ℝ | p.1^2 / 16 + p.2^2 / 9 = 1})
  (hF1 : F1 = (-4, 0))
  (hF2 : F2 = (4, 0))
  (hP_dist : (real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2)) * (real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2)) = 12) :
  ∃ θ : ℝ, θ = 60 ∧ cosine (P F1 F2) = 1 / 2 :=
sorry

end angle_f1pf2_l271_271494


namespace num_even_multiple_5_perfect_squares_lt_1000_l271_271816

theorem num_even_multiple_5_perfect_squares_lt_1000 : 
  ∃ n, n = 3 ∧ ∀ x, (x < 1000) ∧ (x > 0) ∧ (∃ k, x = 100 * k^2) → (n = 3) := by 
  sorry

end num_even_multiple_5_perfect_squares_lt_1000_l271_271816


namespace radius_of_inscribed_circle_l271_271437

theorem radius_of_inscribed_circle (a b x : ℝ) (hx : 0 < x) 
  (h_side_length : a > 20) 
  (h_TM : a = x + 8) 
  (h_OM : b = x + 9) 
  (h_Pythagorean : (a - 8)^2 + (b - 9)^2 = x^2) :
  x = 29 :=
by
  -- Assume all conditions and continue to the proof part.
  sorry

end radius_of_inscribed_circle_l271_271437


namespace triangle_side_length_approx_l271_271869

theorem triangle_side_length_approx (a b C : ℝ) (cosC : ℝ) (h1 : a = 2) (h2 : b = sqrt 3 - 1) (h3 : C = π / 6) (h4 : cosC = sqrt 3 / 2) :
  ∃ c : ℝ, c ≈ 1.5 :=
by
  sorry

end triangle_side_length_approx_l271_271869


namespace problem_proof_l271_271998

noncomputable def product := (20 : ℚ) * y^3 * (8 : ℚ) * y^2 * (1 / (4 * y)^3)

theorem problem_proof (y : ℚ) : product = (5 / 2) * y^2 :=
by
  sorry

end problem_proof_l271_271998


namespace person_saves_2000_l271_271969

variable (income expenditure savings : ℕ)
variable (h_ratio : income / expenditure = 7 / 6)
variable (h_income : income = 14000)

theorem person_saves_2000 (h_ratio : income / expenditure = 7 / 6) (h_income : income = 14000) :
  savings = income - (6 * (14000 / 7)) :=
by
  sorry

end person_saves_2000_l271_271969


namespace functional_condition_specific_case_l271_271272

-- Define the main function f
noncomputable def f (x : ℝ) : ℝ := 
  if 0 ≤ x ∧ x ≤ 1 then x * (1 - x)
  else if -1 ≤ x ∧ x ≤ 0 then -0.5 * x * (x + 1)
  else sorry

-- Define the functional condition f(x+1) = 2 * f(x)
theorem functional_condition (x : ℝ) : f(x + 1) = 2 * f(x) := sorry

-- Prove the specific case when -1 ≤ x ≤ 0
theorem specific_case (x : ℝ) (h : -1 ≤ x ∧ x ≤ 0) : f(x) = -0.5 * x * (x + 1) := 
by 
  sorry

end functional_condition_specific_case_l271_271272


namespace sin_of_tan_eq_sqrt2_div_3_l271_271075

theorem sin_of_tan_eq_sqrt2_div_3 {A : Real.Angle} (h : Real.tan A = Real.sqrt 2 / 3) : 
  Real.sin A = Real.sqrt 22 / 11 := 
by
  sorry

end sin_of_tan_eq_sqrt2_div_3_l271_271075


namespace area_of_trajectory_l271_271483

theorem area_of_trajectory (z : ℂ) (hz1 : 0 < re (z / 10) ∧ re (z / 10) < 1)
                           (hz2 : 0 < im (z / 10) ∧ im (z / 10) < 1)
                           (hz3 : 0 < re (10 / conj z) ∧ re (10 / conj z) < 1)
                           (hz4 : 0 < im (10 / conj z) ∧ im (10 / conj z) < 1) :
  let x := re z,
      y := im z,
      region := { z : ℂ | 0 < re z ∧ re z < 10 ∧ 0 < im z ∧ im z < 10 ∧
                          (re z - 5)^2 + im z^2 > 25 ∧ re z^2 + (im z - 5)^2 > 25 } in
  ∃ area : ℝ, area = 75 - 25 / 2 * Real.pi :=
by
  sorry

end area_of_trajectory_l271_271483


namespace circle_equation_l271_271209

theorem circle_equation (center : ℝ × ℝ) (point : ℝ × ℝ) (h_center : center = (2, -1)) (h_point : point = (-1, 3)) :
  (let r := (Real.sqrt ((2 + 1)^2 + (-1 - 3)^2)) in 
  (x - 2)^2 + (y + 1)^2 = r^2) :=
by
  sorry

end circle_equation_l271_271209


namespace first_person_job_completion_time_l271_271558

noncomputable def job_completion_time :=
  let A := 1 - (1/5)
  let C := 1/8
  let combined_rate := A + C
  have h1 : combined_rate = 0.325 := by
    sorry
  have h2 : A ≠ 0 := by
    sorry
  (1 / A : ℝ)
  
theorem first_person_job_completion_time :
  job_completion_time = 1.25 :=
by
  sorry

end first_person_job_completion_time_l271_271558


namespace determine_identity_l271_271092

-- Define the statements made by the brothers
def FirstBrother : Prop := 
  ∀ (Trulya Tra_lya : Prop), (¬Trulya ∧ Tra_lya)

def SecondBrother : Prop := 
  ∀ (Trulya Tra_lya : Prop), (Trulya ∨ ¬Tra_lya)

def CardSuitStatement : Prop := 
  ∀ (cards_same_suit : Prop), cards_same_suit

-- Define the proof problem
theorem determine_identity :
  ∀ (Trulya Tra_lya cards_same_suit : Prop),
    FirstBrother Trulya Tra_lya →
    SecondBrother Trulya Tra_lya →
    ¬CardSuitStatement cards_same_suit →
    Tra_lya :=
by
  intros Trulya Tra_lya cards_same_suit first_brother_stmt second_brother_stmt card_suit_false
  sorry

end determine_identity_l271_271092


namespace part1_solution_set_part2_values_of_a_l271_271035

-- Parameters and definitions for the function and conditions
def f (x a : ℝ) := |x - a^2| + |x - (2*a + 1)|

-- Part (1) When a = 2, the inequality's solution set
theorem part1_solution_set (x : ℝ) : f x 2 ≥ 4 ↔ (x ≤ 3/2 ∨ x ≥ 11/2) := 
sorry

-- Part (2) Range of values for a given f(x, a) ≥ 4
theorem part2_values_of_a (a : ℝ) : 
∀ x, f x a ≥ 4 → (a ≤ -1 ∨ a ≥ 3) :=
sorry

end part1_solution_set_part2_values_of_a_l271_271035


namespace max_sector_area_l271_271361

-- Define the necessary components related to the sector
def perimeter (r : ℝ) (α : ℝ) : ℝ := 2 * r + r * α
def area (r : ℝ) (α : ℝ) : ℝ := (1/2) * r^2 * α

-- The main theorem
theorem max_sector_area :
  ∃ r α : ℝ, perimeter r α = 20 ∧ (∀ r' α', perimeter r' α' = 20 → area r α ≥ area r' α') ∧ area r α = 50 := 
sorry

end max_sector_area_l271_271361


namespace _l271_271748

@[simp] theorem upper_base_length (ABCD is_trapezoid: Boolean)
  (point_M: Boolean)
  (perpendicular_DM_AB: Boolean)
  (MC_eq_CD: Boolean)
  (AD_eq_d: ℝ)
  : BC = d / 2 := sorry

end _l271_271748


namespace asymptote_of_hyperbola_l271_271801

theorem asymptote_of_hyperbola (x y : ℝ) :
  (x^2 - (y^2 / 4) = 1) → (y = 2 * x ∨ y = -2 * x) := sorry

end asymptote_of_hyperbola_l271_271801


namespace max_knights_and_courtiers_l271_271451

theorem max_knights_and_courtiers (a b : ℕ) (ha : 12 ≤ a ∧ a ≤ 18) (hb : 10 ≤ b ∧ b ≤ 20) :
  (1 / a : ℚ) + (1 / b) = (1 / 7) → a = 14 ∧ b = 14 :=
begin
  -- Proof would go here.
  sorry
end

end max_knights_and_courtiers_l271_271451


namespace solve_system_of_equations_l271_271505

theorem solve_system_of_equations : 
  ∃ (x y : ℚ), 4 * x - 3 * y = -2 ∧ 5 * x + 2 * y = 8 ∧ x = 20 / 23 ∧ y = 42 / 23 :=
by
  sorry

end solve_system_of_equations_l271_271505


namespace triangles_same_euler_line_l271_271191

theorem triangles_same_euler_line
  (A B C D E F G H I: ℝ)
  (orthocenter_ABC : D = foot_of_altitude A B C ∧ E = foot_of_altitude B C A ∧ F = foot_of_altitude C A B)
  (incircle_DEF : G = touch_point_incircle D E F ∧ H = touch_point_incircle E F D ∧ I = touch_point_incircle F D E) :
  Euler_line ABC = Euler_line GHI := 
sorry

end triangles_same_euler_line_l271_271191


namespace part1_part2_l271_271018

def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1 (x : ℝ) : 
  ∀ x, f x 2 ≥ 4 ↔ (x ≤ 3 / 2 ∨ x ≥ 11 / 2) :=
by sorry

theorem part2 (x a : ℝ) :
  f x a ≥ 4 → (a ≤ -1 ∨ a ≥ 3) :=
by sorry

end part1_part2_l271_271018


namespace find_angle_ACE_l271_271863

theorem find_angle_ACE (DC_parallel_AB : ∀ (DC AB : Line), DC ∥ AB)
  (angle_DCA : ∃ x, ∠DCA = x ∧ x = 50)
  (angle_ABC : ∃ y, ∠ABC = y ∧ y = 60)
  (angle_BCE : ∃ z, ∠BCE = z ∧ z = 30) :
  ∃ w, ∠ACE = w ∧ w = 90 := 
by
  sorry

end find_angle_ACE_l271_271863


namespace cube_root_64_minus_sqrt_properties_l271_271502

theorem cube_root_64_minus_sqrt_properties :
  (∛(64 : ℝ) - real.sqrt(7 + 1/4)) ^ 2 = 93/4 - 4 * real.sqrt 29 := by
  sorry

end cube_root_64_minus_sqrt_properties_l271_271502


namespace determine_n_l271_271316

-- Define the condition as a proposition
def condition (n : ℕ) : Prop := 5^3 - 7 = 6^2 + n

-- Prove that n = 82, given the condition
theorem determine_n : ∃ n : ℕ, condition n ∧ n = 82 :=
begin
  use 82,
  unfold condition,
  exact ⟨rfl, rfl⟩,
end

end determine_n_l271_271316


namespace jennie_rental_cost_is_306_l271_271517

-- Definitions for the given conditions
def weekly_rate_mid_size : ℕ := 190
def daily_rate_mid_size_upto10 : ℕ := 25
def total_rental_days : ℕ := 13
def coupon_discount : ℝ := 0.10

-- Define the cost calculation
def rental_cost (days : ℕ) : ℕ :=
  let weeks := days / 7
  let extra_days := days % 7
  let cost_weeks := weeks * weekly_rate_mid_size
  let cost_extra := extra_days * daily_rate_mid_size_upto10
  cost_weeks + cost_extra

def discount (total : ℝ) (rate : ℝ) : ℝ := total * rate

def final_amount (initial_amount : ℝ) (discount_amount : ℝ) : ℝ := initial_amount - discount_amount

-- Main theorem to prove the final payment amount
theorem jennie_rental_cost_is_306 : 
  final_amount (rental_cost total_rental_days) (discount (rental_cost total_rental_days) coupon_discount) = 306 := 
by
  sorry

end jennie_rental_cost_is_306_l271_271517


namespace calculate_f_x_plus_3_l271_271419

def f (x : ℝ) : ℝ := x * (x - 3) / 3

theorem calculate_f_x_plus_3 (x : ℝ) : f (x + 3) = (x + 3) * x / 3 :=
by
  -- sorry to skip the proof
  sorry

end calculate_f_x_plus_3_l271_271419


namespace part1_solution_set_part2_range_of_a_l271_271028

def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1_solution_set (x : ℝ) : 
  (f x 2 ≥ 4) ↔ (x ≤ 3 / 2 ∨ x ≥ 11 / 2) :=
sorry

theorem part2_range_of_a (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 4) → (a ≤ -1 ∨ a ≥ 3) :=
sorry

end part1_solution_set_part2_range_of_a_l271_271028


namespace MrMartinSpent_l271_271941

theorem MrMartinSpent : 
  ∀ (C B : ℝ), 
    3 * C + 2 * B = 12.75 → 
    B = 1.5 → 
    2 * C + 5 * B = 14 := 
by
  intros C B h1 h2
  sorry

end MrMartinSpent_l271_271941


namespace min_oranges_in_new_box_l271_271217

theorem min_oranges_in_new_box (m n : ℕ) (x : ℕ) (h1 : m + n ≤ 60) 
    (h2 : 59 * m = 60 * n + x) : x = 30 :=
sorry

end min_oranges_in_new_box_l271_271217


namespace cyclic_parallelogram_diagonal_length_l271_271303

theorem cyclic_parallelogram_diagonal_length
  (A B C D O : Point)
  {r : ℝ}
  (h_circle : circle_circumscribed A B C D r)
  (h_parallelogram : parallelogram A B C D)
  (h_AB : dist A B = 6)
  (h_AD : dist A D = 5)
  (O_mid_AC : midpoint O A C)
  (O_mid_BD : midpoint O B D) :
  dist A O = dist O C := sorry

noncomputable def AO_length (A B C D O : Point) {r : ℝ} (h_circle : circle_circumscribed A B C D r) (h_parallelogram : parallelogram A B C D) (h_AB : dist A B = 6) (h_AD : dist A D = 5) (O_mid_AC : midpoint O A C) (O_mid_BD : midpoint O B D) : ℝ :=
  sqrt 61 / 2

#check @cyclic_parallelogram_diagonal_length

end cyclic_parallelogram_diagonal_length_l271_271303


namespace find_quadratic_with_axis_l271_271651

def quadratic_symmetry_axis (f : ℝ → ℝ) := ∃ h k : ℝ, ∀ x : ℝ, f x = (x + h)^2 + k

theorem find_quadratic_with_axis 
  (A B C D : ℝ → ℝ)
  (hA : quadratic_symmetry_axis A)
  (hB : quadratic_symmetry_axis B)
  (hC : quadratic_symmetry_axis C)
  (hD : quadratic_symmetry_axis D)
  (axis_A : ∃ h, h = -1 ∧ ∀ x, A x = (x + h)^2)
  (axis_B : ∀ h, ¬ (h = -1 ∧ ∀ x, B x = (x + h)^2))
  (axis_C : ∀ h, ¬ (h = -1 ∧ ∀ x, C x = (x + h)^2))
  (axis_D : ∀ h, ¬ (h = -1 ∧ ∀ x, D x = (x + h)^2)) :
  A = λ x, (x + 1)^2 :=
by sorry

end find_quadratic_with_axis_l271_271651


namespace baker_price_l271_271263

theorem baker_price
  (P : ℝ)
  (h1 : 8 * P = 320)
  (h2 : 10 * (0.80 * P) = 320)
  : P = 40 := sorry

end baker_price_l271_271263


namespace missing_number_l271_271668

theorem missing_number (mean : ℝ) (numbers : List ℝ) (x : ℝ) (h_mean : mean = 14.2) (h_numbers : numbers = [13.0, 8.0, 13.0, 21.0, 23.0]) :
  (numbers.sum + x) / (numbers.length + 1) = mean → x = 7.2 :=
by
  -- states the hypothesis about the mean calculation into the theorem structure
  intro h
  sorry

end missing_number_l271_271668


namespace count_sums_of_two_cubes_lt_400_l271_271055

theorem count_sums_of_two_cubes_lt_400 : 
  ∃ (s : Finset ℕ), 
    (∀ n ∈ s, ∃ a b, 1 ≤ a ∧ a ≤ 7 ∧ 1 ≤ b ∧ b ≤ 7 ∧ n = a^3 + b^3 ∧ (Odd a ∨ Odd b) ∧ n < 400) ∧
    s.card = 15 :=
by 
  sorry

end count_sums_of_two_cubes_lt_400_l271_271055


namespace sarah_more_than_cecily_l271_271221

theorem sarah_more_than_cecily (t : ℕ) (ht : t = 144) :
  let s := (1 / 3 : ℚ) * t
  let a := (3 / 8 : ℚ) * t
  let c := t - (s + a)
  s - c = 6 := by
  sorry

end sarah_more_than_cecily_l271_271221


namespace pipes_fill_together_in_16_hours_l271_271640

-- Definitions based on the conditions
def RA (RB : ℝ) : ℝ := 2 * RB
def RC (RB : ℝ) : ℝ := RB / 2
def RB := 1 / 56

-- Combined rate of A, B, and C
def RABC := RA RB + RB + RC RB

-- Time to fill the tank together
def time_to_fill := 1 / RABC

-- Main theorem statement
theorem pipes_fill_together_in_16_hours : time_to_fill = 16 := by
  sorry

end pipes_fill_together_in_16_hours_l271_271640


namespace tractor_efficiency_l271_271959

theorem tractor_efficiency (x y : ℝ) (h1 : 18 / x = 24 / y) (h2 : x + y = 7) :
  x = 3 ∧ y = 4 :=
by {
  sorry
}

end tractor_efficiency_l271_271959


namespace expected_number_of_digits_on_fair_icosahedral_die_l271_271269

noncomputable def expected_digits_fair_icosahedral_die : ℚ :=
  let prob_one_digit := (9 : ℚ) / 20
  let prob_two_digits := (11 : ℚ) / 20
  (prob_one_digit * 1) + (prob_two_digits * 2)

theorem expected_number_of_digits_on_fair_icosahedral_die : expected_digits_fair_icosahedral_die = 1.55 := by
  sorry

end expected_number_of_digits_on_fair_icosahedral_die_l271_271269


namespace pebbles_sum_at_12_days_l271_271942

def pebbles_collected (n : ℕ) : ℕ :=
  if n = 0 then 0 else n + pebbles_collected (n - 1)

theorem pebbles_sum_at_12_days : pebbles_collected 12 = 78 := by
  -- This would be the place for the proof, but adding sorry as instructed.
  sorry

end pebbles_sum_at_12_days_l271_271942


namespace part1_solution_set_part2_range_of_a_l271_271022

def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1_solution_set (x : ℝ) : 
  (f x 2 ≥ 4) ↔ (x ≤ 3 / 2 ∨ x ≥ 11 / 2) :=
sorry

theorem part2_range_of_a (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 4) → (a ≤ -1 ∨ a ≥ 3) :=
sorry

end part1_solution_set_part2_range_of_a_l271_271022


namespace find_B_min_of_sum_of_squares_l271_271923

-- Given conditions in a)
variables {A B C a b c : ℝ}
hypothesis (h1 : C = 2 * Real.pi / 3)
hypothesis (h2 : (Real.cos A) / (1 + Real.sin A) = Real.sin (2 * B) / (1 + Real.cos (2 * B)))

-- Part (1) prove B = π / 6
theorem find_B (h1 : C = 2 * Real.pi / 3) (h2 : (Real.cos A) / (1 + Real.sin A) = Real.sin (2 * B) / (1 + Real.cos (2 * B))) : B = Real.pi / 6 :=
by sorry

-- Part (2) find the minimum value of (a^2 + b^2) / (c^2)
theorem min_of_sum_of_squares (h1 : C = 2 * Real.pi / 3) (h2 : (Real.cos A) / (1 + Real.sin A) = Real.sin (2 * B) / (1 + Real.cos (2 * B))) : ∃ (m : ℝ), m = 4 * Real.sqrt 2 - 5 ∧ ∀ a b c, a > 0 ∧ b > 0 ∧ c > 0 → (a^2 + b^2) / c^2 ≥ m :=
by sorry

end find_B_min_of_sum_of_squares_l271_271923


namespace _l271_271747

@[simp] theorem upper_base_length (ABCD is_trapezoid: Boolean)
  (point_M: Boolean)
  (perpendicular_DM_AB: Boolean)
  (MC_eq_CD: Boolean)
  (AD_eq_d: ℝ)
  : BC = d / 2 := sorry

end _l271_271747


namespace number_of_female_puppies_number_of_female_puppies_l271_271268

theorem number_of_female_puppies (total_puppies : ℕ) (male_puppies : ℕ) (ratio_female_to_male : ℝ) : ℕ :=
  have female_puppies : ℕ := (ratio_female_to_male * male_puppies)
  by 
    sorry

# Parameters
noncomputable def total_puppies := 12
noncomputable def male_puppies := 10
noncomputable def ratio_female_to_male := 0.2

# Theorem statement
theorem number_of_female_puppies : (ratio_female_to_male * male_puppies) = 2 :=
  by sorry

end number_of_female_puppies_number_of_female_puppies_l271_271268


namespace profit_share_b_l271_271601

variables (Pa Pb Pc : ℕ)

def investment_ratios := Pa / Pb / Pc = 4 / 5 / 6

def profit_difference := Pc - Pa = 640

theorem profit_share_b (h1 : investment_ratios) (h2 : profit_difference) : Pb = 1600 :=
sorry

end profit_share_b_l271_271601


namespace part1_solution_set_part2_range_of_a_l271_271001

-- Define the function f for a general a
def f (x a : ℝ) : ℝ := |x - a^2| + |x - (2 * a) + 1|

-- Part 1: When a = 2
theorem part1_solution_set (x : ℝ) (h : f x 2 ≥ 4) : x ≤ 3/2 ∨ x ≥ 11/2 :=
  sorry

-- Part 2: Range of values for a
theorem part2_range_of_a (a : ℝ) (h : ∀ x, f x a ≥ 4) : a ≤ -1 ∨ a ≥ 3 :=
  sorry

end part1_solution_set_part2_range_of_a_l271_271001


namespace part1_solution_set_part2_range_of_a_l271_271020

def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1_solution_set (x : ℝ) : 
  (f x 2 ≥ 4) ↔ (x ≤ 3 / 2 ∨ x ≥ 11 / 2) :=
sorry

theorem part2_range_of_a (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 4) → (a ≤ -1 ∨ a ≥ 3) :=
sorry

end part1_solution_set_part2_range_of_a_l271_271020


namespace max_min_sum_l271_271368

theorem max_min_sum (a b : ℝ)
  (h1 : -1 ≤ a + b ∧ a + b ≤ 3)
  (h2 : 2 ≤ a - b ∧ a - b ≤ 4) :
  let m := max (2 * a + 3 * b) (-1 + 3)
      n := min (2 * a + 3 * b) (2 - 4)
  in m + n = 2 :=
by {
  sorry
}

end max_min_sum_l271_271368


namespace remove_two_points_preserve_property_l271_271480

theorem remove_two_points_preserve_property
  (n : ℕ) (h1 : n ≥ 2) (colors : Fin n → List α) (h2 : ∀ c, List.length (colors c) = 2)
  (property : ∀ (L : List α), 1 ≤ List.length L ∧ List.length L ≤ (2 * n - 1) → ∃ c, List.count c L = 1) :
  ∃ (new_colors : Fin (n-1) → List α), 
    ∀ (L : List α), 1 ≤ List.length L ∧ List.length L ≤ (2 * n - 3) → ∃ c, List.count c L = 1 :=
sorry

end remove_two_points_preserve_property_l271_271480


namespace sum_of_solutions_correct_l271_271575

noncomputable def sum_of_solutions : ℕ :=
  { 
    val := (List.range' 1 30 / 2).sum,
    property := sorry
  }

theorem sum_of_solutions_correct :
  (∀ x : ℕ, (x > 0 ∧ x ≤ 30 ∧ (17 * (5 * x - 3)).mod 10 = 34.mod 10) →
    List.mem x ((List.range' 1 30).filter (λ n, n % 2 = 1)) →
    ((List.range' 1 30 / 2).sum = 225)) :=
by {
  intro x h1 h2,
  sorry
}

end sum_of_solutions_correct_l271_271575


namespace trigonometric_identity_l271_271779

noncomputable def cos_alpha (α : ℝ) : ℝ := -Real.sqrt (1 / (1 + (tan α)^2))
noncomputable def sin_alpha (α : ℝ) : ℝ := Real.sqrt (1 - (cos_alpha α)^2)

theorem trigonometric_identity
  (α : ℝ) (h1 : tan α = -2) (h2 : (π / 2) < α ∧ α < π) :
  cos_alpha α + sin_alpha α = Real.sqrt(5) / 5 :=
sorry

end trigonometric_identity_l271_271779


namespace labyrinth_knights_correct_l271_271850

noncomputable def labyrinth_knights_bound (n : ℕ) : ℕ :=
  n + 1

theorem labyrinth_knights_correct (n : ℕ) 
  (h1 : ∀ i j, i ≠ j → ¬ (parallel (lines i) (lines j))) 
  (h2 : ∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → ¬ (concurrent (lines i) (lines j) (lines k)))
  (h3 : ∀ i, colors i = {red, blue}) 
  (h4 : ∀ i j, i ≠ j → has_door (intersection i j) ↔ (opposite_colors i j))
  : k (labyrinth n) = labyrinth_knights_bound n :=
  sorry

end labyrinth_knights_correct_l271_271850


namespace find_q_l271_271089

variable {m n q : ℝ}

theorem find_q (h1 : m = 3 * n + 5) (h2 : m + 2 = 3 * (n + q) + 5) : q = 2 / 3 := by
  sorry

end find_q_l271_271089


namespace sticks_form_equilateral_triangle_l271_271692

theorem sticks_form_equilateral_triangle (n : ℕ) (h1 : n ≥ 5) (h2 : n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5) :
  ∃ k, k * 3 = (n * (n + 1)) / 2 :=
by
  sorry

end sticks_form_equilateral_triangle_l271_271692


namespace students_in_fifth_and_sixth_classes_l271_271940

theorem students_in_fifth_and_sixth_classes :
  let c1 := 20
  let c2 := 25
  let c3 := 25
  let c4 := c1 / 2
  let total_students := 136
  let total_first_four_classes := c1 + c2 + c3 + c4
  let c5_and_c6 := total_students - total_first_four_classes
  c5_and_c6 = 56 :=
by
  sorry

end students_in_fifth_and_sixth_classes_l271_271940


namespace tim_total_payment_correct_l271_271222

-- Define the conditions stated in the problem
def doc_visit_cost : ℝ := 300
def insurance_coverage_percent : ℝ := 0.75
def cat_visit_cost : ℝ := 120
def pet_insurance_coverage : ℝ := 60

-- Define the amounts covered by insurance 
def insurance_coverage_amount : ℝ := doc_visit_cost * insurance_coverage_percent
def tim_payment_for_doc_visit : ℝ := doc_visit_cost - insurance_coverage_amount
def tim_payment_for_cat_visit : ℝ := cat_visit_cost - pet_insurance_coverage

-- Define the total payment Tim needs to make
def tim_total_payment : ℝ := tim_payment_for_doc_visit + tim_payment_for_cat_visit

-- State the main theorem
theorem tim_total_payment_correct : tim_total_payment = 135 := by
  sorry

end tim_total_payment_correct_l271_271222


namespace mandy_bike_time_l271_271487

-- Definitions of the ratios and time spent on yoga
def ratio_gym_bike : ℕ × ℕ := (2, 3)
def ratio_yoga_exercise : ℕ × ℕ := (2, 3)
def time_yoga : ℕ := 20

-- Theorem stating that Mandy will spend 18 minutes riding her bike
theorem mandy_bike_time (r_gb : ℕ × ℕ) (r_ye : ℕ × ℕ) (t_y : ℕ) 
  (h_rgb : r_gb = (2, 3)) (h_rye : r_ye = (2, 3)) (h_ty : t_y = 20) : 
  let t_e := (r_ye.snd * t_y) / r_ye.fst
  let t_part := t_e / (r_gb.fst + r_gb.snd)
  t_part * r_gb.snd = 18 := sorry

end mandy_bike_time_l271_271487


namespace prove_tan_C_prove_area_ABC_l271_271842

variables (A B C : ℝ)
variables (a b c : ℝ)
variables (angle_A angle_B angle_C : ℝ)
variables (S_ABC : ℝ)

-- Conditions
axiom triangle_sides : ∀ (a b c : ℝ), (a^2 + b^2 - c^2)/(2 * a * b) > 0 -- acute angle C
axiom side_relation : b = 2 * a
axiom sin_law : sqrt(15) * a * sin angle_A = b * sin angle_B * sin angle_C
axiom side_sum : a + c = 6

-- Problem 1: Prove tan C = sqrt(15)
theorem prove_tan_C : tan (angle_C) = sqrt(15) :=
sorry

-- Problem 2: Prove area of triangle ABC is sqrt(15)
theorem prove_area_ABC : S_ABC = sqrt(15) :=
sorry

end prove_tan_C_prove_area_ABC_l271_271842


namespace jasmine_swims_laps_l271_271103

theorem jasmine_swims_laps (laps_per_day : ℕ) (days_per_week : ℕ) (weeks : ℕ) : 
  (laps_per_day = 12) → 
  (days_per_week = 5) → 
  (weeks = 5) → 
  laps_per_day * days_per_week * weeks = 300 := 
by 
  intros h1 h2 h3
  rw [h1, h2, h3]
  exact rfl

end jasmine_swims_laps_l271_271103


namespace isosceles_trapezoid_dot_product_l271_271855

theorem isosceles_trapezoid_dot_product
  (A B C D : ℝ) 
  (AB AD DC : ℝ)
  (iso_trap : is_isosceles_trapezoid A B C D)
  (h_AB : AB = 4)
  (h_AD : AD = 2)
  (h_DC : DC = 4)
  (angle_DAB : ∠DAB = 1 / 6 * π) :
  (\overrightarrow{AD} \cdot \overrightarrow{AB} = 4) := 
  sorry

end isosceles_trapezoid_dot_product_l271_271855


namespace incorrect_glassware_statement_l271_271318

theorem incorrect_glassware_statement :
    let initial_mass := 50
    let initial_fraction := 0.2
    let desired_fraction := 0.1
    let added_mass := 50
    let correct_glassware := ["beaker", "measuring cylinder", "dropper with rubber bulb", "glass rod"]
    ∃ (mass_solute : ℕ) (mass_solution : ℕ), mass_solute = initial_mass * initial_fraction ∧
    desired_fraction = mass_solute / (initial_mass + added_mass) ∧
    ["beaker", "measuring cylinder", "glass rod"] ≠ correct_glassware :=
by
  sorry

end incorrect_glassware_statement_l271_271318


namespace find_upper_base_length_l271_271751

variables {A B C D M : Type}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space M]
variables (AB: line_segment A B) (CD: line_segment C D)
variables (AD : ℝ) (d : ℝ)

noncomputable def upper_base_length (A B C D M : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space M]
  (ABCD : bool)
  (DM_perp_AB : line_segment D M ⊥ line_segment A B)
  (MC_eq_CD : line_segment M C = line_segment C D)
  (AD_length : line_segment A D)
  (d_value : AD_length = d)
: Prop :=
BC = d / 2

theorem find_upper_base_length :
∀ (A B C D M : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space M]
  (ABCD : bool)
  (DM_perp_AB : line_segment D M ⊥ line_segment A B)
  (MC_eq_CD : line_segment M C = line_segment C D)
  (AD_length : line_segment A D)
  (d_value : AD_length = d),
  upper_base_length A B C D M ABCD DM_perp_AB MC_eq_CD AD_length d_value := sorry

end find_upper_base_length_l271_271751


namespace length_of_second_train_is_approximately_159_98_l271_271231

noncomputable def length_of_second_train : ℝ :=
  let length_first_train := 110 -- meters
  let speed_first_train := 60 -- km/hr
  let speed_second_train := 40 -- km/hr
  let time_to_cross := 9.719222462203025 -- seconds
  let km_per_hr_to_m_per_s := 5 / 18 -- conversion factor from km/hr to m/s
  let relative_speed := (speed_first_train + speed_second_train) * km_per_hr_to_m_per_s -- relative speed in m/s
  let total_distance := relative_speed * time_to_cross -- total distance covered
  total_distance - length_first_train -- length of the second train

theorem length_of_second_train_is_approximately_159_98 :
  abs (length_of_second_train - 159.98) < 0.01 := 
by
  sorry -- Placeholder for the actual proof

end length_of_second_train_is_approximately_159_98_l271_271231


namespace pair_students_l271_271432

theorem pair_students :
  (∃ (S5 S4 S3 : Finset ℕ), S5.card = 6 ∧ S4.card = 7 ∧ S3.card = 1 ∧
  (∃ (pairs : Finset (ℕ × ℕ)), 
    (∀ (p : ℕ × ℕ), p ∈ pairs → (p.1 ∈ S5 ∧ p.2 ∈ S4) ∨ (p.1 ∈ S4 ∧ p.2 ∈ S3)) ∧ 
    pairs.card = 7) ∧ 
  ∃ (ways : ℕ), ways = 7 * 720 * 1) :=
exists.intro (Finset.range 6)
(exists.intro (Finset.range 7)
(exists.intro (Finset.singleton 3)
(and.intro (Finset.card_range 6)
(and.intro (Finset.card_range 7)
(and.intro (Finset.card_singleton 3)
(exists.intro 
  ((Finset.range 7).product (Finset.range 6) ∪ (Finset.singleton 3).product (Finset.range 1))
(and.intro 
  (λ p h,
  ((Finset.mem_product.1 (Finset.mem_union.1 h)).2.1 →
    or.inl ⟨Finset.range 6, Finset.range 7⟩) 
  // Similarly for the other combination
  ) ∧ sorry) ∧ sorry)))) sorry.

end pair_students_l271_271432


namespace total_grains_in_grey_areas_l271_271441

theorem total_grains_in_grey_areas (total_grains_circle1 : ℕ) (white_grains_circle1 : ℕ) (total_grains_circle2 : ℕ) (white_grains_circle2 : ℕ) :
  total_grains_circle1 = 87 → white_grains_circle1 = 68 →
  total_grains_circle2 = 110 → white_grains_circle2 = 68 →
  (total_grains_circle2 - white_grains_circle2) + (total_grains_circle1 - white_grains_circle1) = 61 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end total_grains_in_grey_areas_l271_271441


namespace polynomial_root_property_l271_271495

noncomputable def f (n : ℕ) (x : ℝ) : ℝ :=
∑ k in Finset.range (n + 1), x ^ k / k.factorial

theorem polynomial_root_property (n : ℕ) :
  (Even n → ∀ x : ℝ, f n x ≠ 0) ∧ (Odd n → ∃! x : ℝ, f n x = 0) :=
by sorry

end polynomial_root_property_l271_271495


namespace min_value_fraction_l271_271736

theorem min_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2 * y = 1) : 
  ( (x + 1) * (y + 1) / (x * y) ) >= 8 + 4 * Real.sqrt 3 :=
sorry

end min_value_fraction_l271_271736


namespace find_upper_base_length_l271_271753

variables {A B C D M : Type}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space M]
variables (AB: line_segment A B) (CD: line_segment C D)
variables (AD : ℝ) (d : ℝ)

noncomputable def upper_base_length (A B C D M : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space M]
  (ABCD : bool)
  (DM_perp_AB : line_segment D M ⊥ line_segment A B)
  (MC_eq_CD : line_segment M C = line_segment C D)
  (AD_length : line_segment A D)
  (d_value : AD_length = d)
: Prop :=
BC = d / 2

theorem find_upper_base_length :
∀ (A B C D M : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space M]
  (ABCD : bool)
  (DM_perp_AB : line_segment D M ⊥ line_segment A B)
  (MC_eq_CD : line_segment M C = line_segment C D)
  (AD_length : line_segment A D)
  (d_value : AD_length = d),
  upper_base_length A B C D M ABCD DM_perp_AB MC_eq_CD AD_length d_value := sorry

end find_upper_base_length_l271_271753


namespace find_B_min_fraction_of_squares_l271_271910

-- Lean 4 statement for part (1)
theorem find_B (A B C a b c : ℝ) (h_triangle : A + B + C = π) 
(h_sides : a > 0 ∧ b > 0 ∧ c > 0) 
(h_identity : (cos A) / (1 + sin A) = (sin (2 * B)) / (1 + cos (2 * B))) 
(h_C : C = 2 * π / 3) : 
B = π / 6 := sorry

-- Lean 4 statement for part (2)
theorem min_fraction_of_squares (A B C a b c : ℝ) 
(h_triangle : A + B + C = π) 
(h_sides : a > 0 ∧ b > 0 ∧ c > 0) 
(h_identity : (cos A) / (1 + sin A) = (sin (2 * B)) / (1 + cos (2 * B))) 
(h_C : C = 2 * π / 3) : 
∀ a b c, ∃ m, m = 4 * sqrt 2 - 5 ∧ (a^2 + b^2) / c^2 = m := sorry

end find_B_min_fraction_of_squares_l271_271910


namespace water_usage_difference_l271_271184

theorem water_usage_difference (C X : ℕ)
    (h1 : C = 111000)
    (h2 : C = 3 * X)
    (days : ℕ) (h3 : days = 365) :
    (C * days - X * days) = 26910000 := by
  sorry

end water_usage_difference_l271_271184


namespace a_five_is_twenty_five_l271_271396

def a : ℕ → ℕ
| 0       := 0  -- We don't use a₀, so providing default value
| 1       := 1
| (n + 1) := a n + 2 * n + 1

theorem a_five_is_twenty_five : a 5 = 25 := 
by sorry

end a_five_is_twenty_five_l271_271396


namespace inscribed_circle_radius_range_l271_271530

noncomputable def r_range (AD DB : ℝ) (angle_A : ℝ) : Set ℝ :=
  { r | 0 < r ∧ r < (-3 + Real.sqrt 33) / 2 }

theorem inscribed_circle_radius_range (AD DB : ℝ) (angle_A : ℝ) (h1 : AD = 2 * Real.sqrt 3) 
    (h2 : DB = Real.sqrt 3) (h3 : angle_A > 60) : 
    r_range AD DB angle_A = { r | 0 < r ∧ r < (-3 + Real.sqrt 33) / 2 } :=
by
  sorry

end inscribed_circle_radius_range_l271_271530


namespace find_train_speed_l271_271598

variable (L V : ℝ)

-- Conditions
def condition1 := V = L / 10
def condition2 := V = (L + 600) / 30

-- Theorem statement
theorem find_train_speed (h1 : condition1 L V) (h2 : condition2 L V) : V = 30 :=
by
  sorry

end find_train_speed_l271_271598


namespace angle_DEC_eq_angle_AEF_iff_l271_271095

variables {A B C D E F : Type} [IsTriangle A B C] 
variables {BC CA : LineSegment} (D_on_BC : OnSegment D BC) (E_on_CA : OnSegment E CA) 
variables (F_on_extension_BA : OnExtension F BA)

theorem angle_DEC_eq_angle_AEF_iff :
  ∠DEC = ∠AEF ↔ (AF / AE * BD / BF * CE / CD = 1) :=
by sorry

end angle_DEC_eq_angle_AEF_iff_l271_271095


namespace second_smallest_is_seven_l271_271245

theorem second_smallest_is_seven : ∀ (a b c d : ℕ), (a = 5) → (b = 8) → (c = 9) → (d = 7) → ∃ (x : ℕ), x ∈ {a, b, c, d} ∧ x = 7 ∧ (∀ y ∈ {a, b, c, d}, y < x → y = 5) :=
by
  intros a b c d ha hb hc hd
  use d
  split
  · simp [ha, hb, hc, hd]
  split
  · assumption
  · intros y hy hlt
    repeat { cases hy; simp [ha, hb, hc, hd] at *; contradiction <|> assumption <|> skip }
  sorry

end second_smallest_is_seven_l271_271245


namespace proof_problem_l271_271881

noncomputable def f (a b : ℝ) (x : ℝ) := a * Real.sin (2 * x) + b * Real.cos (2 * x)

theorem proof_problem (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : ∀ x : ℝ, f a b x ≤ |f a b (π / 6)|) : 
  (f a b (11 * π / 12) = 0) ∧
  (|f a b (7 * π / 12)| < |f a b (π / 5)|) ∧
  (¬ (∀ x : ℝ, f a b x = f a b (-x)) ∧ ¬ (∀ x : ℝ, f a b x = -f a b (-x))) := 
sorry

end proof_problem_l271_271881


namespace factorization_1_factorization_2_l271_271684

variable (x y : ℝ)

-- Problem 1: Prove -x^2 + 12xy - 36y^2 = -(x - 6y)^2
theorem factorization_1 : -x^2 + 12xy - 36y^2 = -(x - 6y)^2 := 
sorry

-- Problem 2: Prove x^4 - 9x^2 = x^2(x + 3)(x - 3)
theorem factorization_2 : x^4 - 9x^2 = x^2 * (x + 3) * (x - 3) :=
sorry

end factorization_1_factorization_2_l271_271684


namespace jackson_tray_pieces_l271_271629

theorem jackson_tray_pieces :
  ∀ (tray_length tray_width piece_length piece_width : ℕ),
  tray_length = 24 →
  tray_width = 20 →
  piece_length = 3 →
  piece_width = 2 →
  (tray_length * tray_width) / (piece_length * piece_width) = 80 :=
by
  intros tray_length tray_width piece_length piece_width h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end jackson_tray_pieces_l271_271629


namespace intersection_A_complement_B_range_of_a_l271_271726

-- Define sets A and B with their respective conditions
def U : Set ℝ := Set.univ
def A (a : ℝ) : Set ℝ := {x | (x - 2) * (x - (3 * a + 1)) < 0}
def B (a : ℝ) : Set ℝ := {x | (x - 2 * a) / (x - (a^2 + 1)) < 0}

-- Question 1: Prove the intersection when a = 2
theorem intersection_A_complement_B (a : ℝ) (h : a = 2) : 
  A a ∩ (U \ B a) = {x | 2 < x ∧ x ≤ 4} ∪ {x | 5 ≤ x ∧ x < 7} :=
by sorry

-- Question 2: Find the range of a such that A ∪ B = A given a ≠ 1
theorem range_of_a (a : ℝ) (h : a ≠ 1) : 
  (A a ∪ B a = A a) ↔ (1 < a ∧ a ≤ 3 ∨ a = -1) :=
by sorry

end intersection_A_complement_B_range_of_a_l271_271726


namespace equal_red_blue_segments_l271_271956

-- Define the setup of the grid
variables {k n : ℕ}

-- Define a property to check the coloring conditions
def colored_correctly (grid : matrix (fin(2*k)) (fin(2*n)) (bool)) : Prop :=
  (∀ i : fin (2*k), (∑ j, ite (grid i j = tt) 1 0 = k) ∧ (∑ j, ite (grid i j = ff) 1 0 = k)) ∧
  (∀ j : fin (2*n), (∑ i, ite (grid i j = tt) 1 0 = n) ∧ (∑ i, ite (grid i j = ff) 1 0 = n))

-- Define the property of segments colored correctly
def segments_colored_correctly (grid : matrix (fin(2*k)) (fin(2*n)) (bool)) : Prop :=
  ∀ i : fin (2*k - 1), ∀ j : fin (2*n - 1), 
  (grid i j = grid i.succ j) ∨ (grid i j = grid i j.succ)

-- Main statement to prove
theorem equal_red_blue_segments (grid : matrix (fin(2*k)) (fin(2*n)) (bool)) 
    (hc : colored_correctly grid) (hs : segments_colored_correctly grid) :
    (∑ i : fin (2*k - 1), ∑ j : fin (2*n - 1), if (grid i j = tt ∧ grid i.succ j = tt) ∨ (grid i j = tt ∧ grid i j.succ = tt) then 1 else 0) =
    (∑ i : fin (2*k - 1), ∑ j : fin (2*n - 1), if (grid i j = ff ∧ grid i.succ j = ff) ∨ (grid i j = ff ∧ grid i j.succ = ff) then 1 else 0) :=
begin
sorry -- Proof to be provided
end

end equal_red_blue_segments_l271_271956


namespace total_perimeter_l271_271628

/-- 
A rectangular plot where the long sides are three times the length of the short sides. 
One short side is 80 feet. Prove the total perimeter is 640 feet.
-/
theorem total_perimeter (s : ℕ) (h : s = 80) : 8 * s = 640 :=
  by sorry

end total_perimeter_l271_271628


namespace find_B_min_fraction_of_squares_l271_271911

-- Lean 4 statement for part (1)
theorem find_B (A B C a b c : ℝ) (h_triangle : A + B + C = π) 
(h_sides : a > 0 ∧ b > 0 ∧ c > 0) 
(h_identity : (cos A) / (1 + sin A) = (sin (2 * B)) / (1 + cos (2 * B))) 
(h_C : C = 2 * π / 3) : 
B = π / 6 := sorry

-- Lean 4 statement for part (2)
theorem min_fraction_of_squares (A B C a b c : ℝ) 
(h_triangle : A + B + C = π) 
(h_sides : a > 0 ∧ b > 0 ∧ c > 0) 
(h_identity : (cos A) / (1 + sin A) = (sin (2 * B)) / (1 + cos (2 * B))) 
(h_C : C = 2 * π / 3) : 
∀ a b c, ∃ m, m = 4 * sqrt 2 - 5 ∧ (a^2 + b^2) / c^2 = m := sorry

end find_B_min_fraction_of_squares_l271_271911


namespace prove_g_l271_271138

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log2 x else g x

axiom odd_f : ∀ x : ℝ, f (-x) = -f x

theorem prove_g (g : ℝ → ℝ) (h : f (- (1 / 4)) = 2) : g (- (1/4)) = 2 := by
  sorry

end prove_g_l271_271138


namespace union_of_A_and_B_l271_271046

open Set

def A := {x | (x + 1)*(x - 2) = 0 ∧ 0 < x}  -- 0 < x ensures x ∈ ℕ
def B := {2, 4, 5}

theorem union_of_A_and_B :
  A ∪ B = {2, 4, 5} :=
by
  sorry

end union_of_A_and_B_l271_271046


namespace intersect_polyline_l271_271097

-- Define the unit square and the polyline with the given conditions.
structure Square :=
  (side_length : ℝ)
  (is_unit : side_length = 1)

structure Polyline (square : Square) :=
  (vertices : list (ℝ × ℝ))
  (length : ℝ)
  (length_gt_1000 : length > 1000)
  (no_self_intersections : ∀ x y z w : (ℝ × ℝ), 
        x ≠ y ∧ y ≠ z ∧ z ≠ w ∧ 
        ((x, y) ∈ list.zip vertices (vertices.tail) → (y, z) ∈ list.zip vertices (vertices.tail) 
        → ¬∃ u v : ℝ, (x.1 = u ∧ y.1 = u ∧ z.1 = v ∧ w.1 = v)))

-- The theorem statement
theorem intersect_polyline (square : Square) (polyline : Polyline square) :
  ∃ line : ℝ → ℝ,
    (∃ k : ℝ, ∀ x : ℝ, line x = k ∨ line x = x + 1 ∨ line x = x - 1) ∧
    (∃ n : ℕ, ∃ points : list (ℝ × ℝ), list.length points ≥ 501 ∧ 
      ∀ p ∈ points, ∃ segment : ℝ × ℝ, 
        segment ∈ list.zip polyline.vertices (polyline.vertices.tail) ∧ 
        segment.1.2 = p.2 ∧ p.1 = line segment.1.1) :=
sorry

end intersect_polyline_l271_271097


namespace parabola_focus_l271_271967

theorem parabola_focus (p : ℝ) (hp : p > 0) :
    ∀ (x y : ℝ), (x = 2 * p * y^2) ↔ (x, y) = (1 / (8 * p), 0) :=
by 
  sorry

end parabola_focus_l271_271967


namespace part1_part2_l271_271843

noncomputable def triangle_condition
(a b c A B C : ℝ) : Prop :=
A < π / 2 ∧
(b * Real.sin A * Real.cos C + c * Real.sin A * Real.cos B = (Real.sqrt 3 / 2) * a)

noncomputable def f (A ω x : ℝ) : ℝ :=
Real.tan A * Real.sin (ω * x) * Real.cos (ω * x) - (1 / 2) * Real.cos (2 * ω * x)

noncomputable def g (x : ℝ) : ℝ :=
Real.sin (2 * x + π / 3)

theorem part1 (a b c A B C : ℝ)
  (h : triangle_condition a b c A B C) :
  A = π / 3 :=
sorry

theorem part2 :
  (∀ x, - π / 24 ≤ x ∧ x ≤ π / 4 → (1 / 2) ≤ g x ∧ g x ≤ 1) :=
sorry

end part1_part2_l271_271843


namespace f_neg_a_l271_271358

noncomputable def f (x : ℝ) : ℝ :=
if x < -2 then 2^(-x)
else -Real.logBase (1/2) (x + 12)

theorem f_neg_a (a : ℝ) (h : f a = 4) : f (-a) = 16 :=
sorry

end f_neg_a_l271_271358


namespace sum_first_and_third_angle_l271_271981

-- Define the conditions
variable (A : ℕ)
axiom C1 : A + 2 * A + (A - 40) = 180

-- State the theorem to be proven
theorem sum_first_and_third_angle : A + (A - 40) = 70 :=
by
  sorry

end sum_first_and_third_angle_l271_271981


namespace max_value_product_focal_distances_l271_271159

theorem max_value_product_focal_distances {a b c : ℝ} 
  (h1 : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1) 
  (h2 : ∀ x : ℝ, -a ≤ x ∧ x ≤ a) 
  (e : ℝ) :
  (∀ x : ℝ, (a - e * x) * (a + e * x) ≤ a^2) :=
sorry

end max_value_product_focal_distances_l271_271159


namespace sticks_form_equilateral_triangle_l271_271705

theorem sticks_form_equilateral_triangle (n : ℕ) :
  n ≥ 5 → (n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5) → 
  ∃ k : ℕ, ∑ i in finset.range (n + 1), i = 3 * k :=
by {
  sorry
}

end sticks_form_equilateral_triangle_l271_271705


namespace volume_of_pyramid_with_spheres_l271_271539

noncomputable theory

-- Define the variables for the problem
variables (a : ℝ)

-- Define the conditions based on the problem statement
def pyramid_base_side_length : ℝ := a
def pyramid_height : ℝ := -a / 2
def sphere_radius : ℝ := a / 3

-- Define the volume of the solid based on the derived correct answer
def volume_of_solid := (81 - 16 * Real.pi) / 486 * a^3

-- State the theorem to be proved
theorem volume_of_pyramid_with_spheres :
  volume_of_solid a = ((81 - 16 * Real.pi) / 486) * a^3 :=
by
  sorry

end volume_of_pyramid_with_spheres_l271_271539


namespace boat_distance_along_stream_l271_271856

theorem boat_distance_along_stream
  (distance_against_stream : ℝ)
  (speed_still_water : ℝ)
  (travel_time : ℝ) 
  (effective_speed_against_stream : speed_still_water - 2 = distance_against_stream / travel_time) :
  distance_against_stream = 5 →
  speed_still_water = 7 →
  travel_time = 1 →
  let v_s := speed_still_water - distance_against_stream / travel_time in
  let effective_speed_along_stream := speed_still_water + v_s in
  effective_speed_along_stream * travel_time = 9 :=
begin
  intros h1 h2 h3,
  simp only [h1, h2, h3],
  let v_s := 7 - 5 / 1, -- calculates the stream speed
  have h_v_s : v_s = 2, by norm_num,
  let effective_speed_along_stream := 7 + v_s,
  suffices : effective_speed_along_stream * 1 = 9,
  { exact this },
  simp only [h_v_s],
  unfold effective_speed_along_stream,
  norm_num,
sorry -- proof is represented by sorry as directed
end

end boat_distance_along_stream_l271_271856


namespace seats_needed_l271_271206

def flute_players : ℕ := 5
def trumpet_players : ℕ := 3 * flute_players
def trombone_players : ℕ := trumpet_players - 8
def drummers : ℕ := trombone_players + 11
def clarinet_players : ℕ := 2 * flute_players
def french_horn_players : ℕ := trombone_players + 3
def total_seats_needed : ℕ := flute_players + trumpet_players + trombone_players + drummers + clarinet_players + french_horn_players

theorem seats_needed (s : ℕ) (h : s = 65) : total_seats_needed = s :=
by {
  have h_flutes : flute_players = 5 := rfl,
  have h_trumpets : trumpet_players = 3 * flute_players := rfl,
  have h_trombones : trombone_players = trumpet_players - 8 := rfl,
  have h_drums : drummers = trombone_players + 11 := rfl,
  have h_clarinets : clarinet_players = 2 * flute_players := rfl,
  have h_french_horns : french_horn_players = trombone_players + 3 := rfl,
  have h_total : total_seats_needed = flute_players + trumpet_players + trombone_players + drummers + clarinet_players + french_horn_players := rfl,
  rw [h_flutes, h_trumpets, h_trombones, h_drums, h_clarinets, h_french_horns] at h_total,
  simp only [flute_players, trumpet_players, trombone_players, drummers, clarinet_players, french_horn_players] at h_total,
  norm_num at h_total,
  exact h,
}

end seats_needed_l271_271206


namespace circles_are_separated_l271_271673

def circle_relation (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) :=
  let d := Real.sqrt ((c2.1 - c1.1) ^ 2 + (c2.2 - c1.2) ^ 2) in
  if d > r1 + r2 then "separated"
  else if d = r1 + r2 then "externally tangent"
  else if d < r1 + r2 ∧ d > Real.abs (r2 - r1) then "intersect"
  else if d = Real.abs (r2 - r1) then "internally tangent"
  else "contained"

theorem circles_are_separated :
  circle_relation (-1, -3) (3, -1) 1 3 = "separated" :=
by
  sorry

end circles_are_separated_l271_271673


namespace find_quadratic_with_axis_l271_271650

def quadratic_symmetry_axis (f : ℝ → ℝ) := ∃ h k : ℝ, ∀ x : ℝ, f x = (x + h)^2 + k

theorem find_quadratic_with_axis 
  (A B C D : ℝ → ℝ)
  (hA : quadratic_symmetry_axis A)
  (hB : quadratic_symmetry_axis B)
  (hC : quadratic_symmetry_axis C)
  (hD : quadratic_symmetry_axis D)
  (axis_A : ∃ h, h = -1 ∧ ∀ x, A x = (x + h)^2)
  (axis_B : ∀ h, ¬ (h = -1 ∧ ∀ x, B x = (x + h)^2))
  (axis_C : ∀ h, ¬ (h = -1 ∧ ∀ x, C x = (x + h)^2))
  (axis_D : ∀ h, ¬ (h = -1 ∧ ∀ x, D x = (x + h)^2)) :
  A = λ x, (x + 1)^2 :=
by sorry

end find_quadratic_with_axis_l271_271650


namespace find_trigonometric_expression_l271_271791

theorem find_trigonometric_expression (α : Real) (m : Real) (h1 : m < 0) (h2 : terminal_side_passes_through α (-3 * m) (4 * m)) :
  2 * Real.sin α + Real.cos α = -1 :=
begin
  sorry
end

end find_trigonometric_expression_l271_271791


namespace find_sum_a_b_l271_271220

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem find_sum_a_b :
  ∃ a b : ℕ, 
    (∀(r : ℝ), r > 0 → r = 16 →
    (∀ (P1 P2 P3 : Point), 
      (distance P1 P2 = 32 ∧ distance P2 P3 = 32 ∧ distance P3 P1 = 32) ∧
      (∀ (O1 O2 O3 : Circle), 
        radius O1 = 16 ∧ radius O2 = 16 ∧ radius O3 = 16 ∧ 
        externally_tangent O1 O2 ∧ externally_tangent O2 O3 ∧ externally_tangent O3 O1) → 
      ∀(T : Triangle), 
        T.area = sqrt (a : ℝ) + sqrt (b : ℝ))) → 
    a + b = 1769472 :=
sorry

end find_sum_a_b_l271_271220


namespace paint_shop_cost_l271_271293

def cuboid (length width height : ℝ) : Prop :=
  (length > 0) ∧ (width > 0) ∧ (height > 0)

def paint_coverage (red blue yellow : ℝ × ℝ) : Prop :=
  red = (36.50, 16) ∧ blue = (32.25, 17) ∧ yellow = (33.75, 18)

def paint_plan (floor ceiling walls : ℝ × ℝ × ℝ) : Prop :=
  floor = (blue, 8, 10) ∧
  ceiling = (yellow, 8, 10) ∧
  walls = (red, (8, 12), (10, 12))

def total_cost (red_cost blue_cost yellow_cost : ℝ) : Prop :=
  red_cost = 985.50 ∧ blue_cost = 161.25 ∧ yellow_cost = 168.75

theorem paint_shop_cost
  (length width height : ℝ)
  (red blue yellow : ℝ × ℝ)
  (floor ceiling walls : ℝ × ℝ × ℝ)
  (red_cost blue_cost yellow_cost : ℝ) :
  cuboid length width height →
  paint_coverage red blue yellow →
  paint_plan floor ceiling walls →
  total_cost red_cost blue_cost yellow_cost →
  red_cost + blue_cost + yellow_cost = 1315.50 :=
by
  intros _ _ _ _
  sorry

end paint_shop_cost_l271_271293


namespace number_of_ways_to_distribute_balls_l271_271492

-- Define the variables representing the number of balls and boxes
variables (numBalls numBoxes : ℕ)
variables (labelBox1 labelBox2 labelBox3 : ℕ)

-- Non-computable definition to find the number of ways to distribute the balls
noncomputable def distributeBalls : ℕ :=
  let remainingBalls := numBalls - labelBox1 - labelBox2 - labelBox3 in
  if remainingBalls ≥ 0 then
    Nat.choose (remainingBalls + 2) 2
  else
    0

-- The main theorem which states the number of ways to distribute 9 balls into 3 boxes
theorem number_of_ways_to_distribute_balls : distributeBalls 9 3 0 1 2 = 10 :=
by
  -- Use a simplification tactic or similar to state the theorem. The proof is omitted.
  sorry

end number_of_ways_to_distribute_balls_l271_271492


namespace largest_possible_triangle_area_l271_271512

theorem largest_possible_triangle_area :
  ∃ (a b c : ℕ), 
  let P := a + b + c in
  let s := P / 2 in
  s * (s - a) * (s - b) * (s - c) = 4 * (a + b + c)^2 ∧
  (√(s * (s - a) * (s - b) * (s - c)) = a + b + c) ∧
  (a * b * c = 60) :=
sorry

end largest_possible_triangle_area_l271_271512


namespace find_B_min_of_sum_of_squares_l271_271920

-- Given conditions in a)
variables {A B C a b c : ℝ}
hypothesis (h1 : C = 2 * Real.pi / 3)
hypothesis (h2 : (Real.cos A) / (1 + Real.sin A) = Real.sin (2 * B) / (1 + Real.cos (2 * B)))

-- Part (1) prove B = π / 6
theorem find_B (h1 : C = 2 * Real.pi / 3) (h2 : (Real.cos A) / (1 + Real.sin A) = Real.sin (2 * B) / (1 + Real.cos (2 * B))) : B = Real.pi / 6 :=
by sorry

-- Part (2) find the minimum value of (a^2 + b^2) / (c^2)
theorem min_of_sum_of_squares (h1 : C = 2 * Real.pi / 3) (h2 : (Real.cos A) / (1 + Real.sin A) = Real.sin (2 * B) / (1 + Real.cos (2 * B))) : ∃ (m : ℝ), m = 4 * Real.sqrt 2 - 5 ∧ ∀ a b c, a > 0 ∧ b > 0 ∧ c > 0 → (a^2 + b^2) / c^2 ≥ m :=
by sorry

end find_B_min_of_sum_of_squares_l271_271920


namespace poly_at_2_eq_0_l271_271300

def poly (x : ℝ) : ℝ := x^6 - 12 * x^5 + 60 * x^4 - 160 * x^3 + 240 * x^2 - 192 * x + 64

theorem poly_at_2_eq_0 : poly 2 = 0 := by
  sorry

end poly_at_2_eq_0_l271_271300


namespace prime_div_factorial_l271_271876

theorem prime_div_factorial (p n : ℕ) (hp : p.Prime) (hn : 0 < n)
  (h : p^p ∣ n.fact) : p^(p+1) ∣ n.fact := 
sorry

end prime_div_factorial_l271_271876


namespace maximum_knights_and_courtiers_l271_271455

theorem maximum_knights_and_courtiers :
  ∃ (a b : ℕ), 12 ≤ a ∧ a ≤ 18 ∧ 10 ≤ b ∧ b ≤ 20 ∧ (1 / a + 1 / b = 1 / 7) ∧
               (∀ (a' b' : ℕ), 12 ≤ a' ∧ a' ≤ 18 ∧ 10 ≤ b' ∧ b' ≤ 20 ∧ (1 / a' + 1 / b' = 1 / 7) → b ≤ b') → 
  a = 14 ∧ b = 14 :=
by
  use 14, 14
  split
  repeat { sorry }

end maximum_knights_and_courtiers_l271_271455


namespace cos_func_shift_l271_271554

theorem cos_func_shift :
  ∀ x : ℝ, 
  ∃ k : ℝ, 
    k = 2 ∧ 
    y = cos (k * x - π/3) →
    y_shifted = cos (k * (x + π/3)) →
    y_result = -cos (k * x) := 
by 
  intros 
  sorry

end cos_func_shift_l271_271554


namespace johns_remaining_funds_l271_271109

-- Definitions for the given conditions
def savings_octal : ℕ := 5372
def ticket_cost_decimal : ℕ := 1200

-- Helper function to convert octal to decimal
def octal_to_decimal (n : ℕ) : ℕ :=
  let s := n.digits 8
  s.enum_from 0 |>.sum (λ (p : ℕ × ℕ), p.fst * 8 ^ p.snd)

def savings_decimal := octal_to_decimal savings_octal
def remaining_funds := savings_decimal - ticket_cost_decimal

-- The proof statement
theorem johns_remaining_funds : remaining_funds = 1610 :=
by
  -- Skipping the proof
  sorry

end johns_remaining_funds_l271_271109


namespace value_of_m_l271_271426

theorem value_of_m (m : ℝ) : (3 = 2 * m + 1) → m = 1 :=
by
  intro h
  -- skipped proof due to requirement
  sorry

end value_of_m_l271_271426


namespace length_of_segments_equal_d_l271_271839

noncomputable def d_eq (AB BC AC : ℝ) (h : AB = 550 ∧ BC = 580 ∧ AC = 620) : ℝ :=
  if h_eq : AB = 550 ∧ BC = 580 ∧ AC = 620 then 342 else 0

theorem length_of_segments_equal_d (AB BC AC : ℝ) (h : AB = 550 ∧ BC = 580 ∧ AC = 620) :
  d_eq AB BC AC h = 342 :=
by
  sorry

end length_of_segments_equal_d_l271_271839


namespace congruent_regular_pentagons_triangle_x_angle_l271_271520

theorem congruent_regular_pentagons_triangle_x_angle :
  ∃ x : ℝ, (∀ (P Q R : Type) 
  (angle_PQR angle_PRQ angle_QPR : ℝ), 
  (angle_PQR = x) ∧ (angle_PRQ = x) ∧ 
  (angle_QPR = (180 - 2 * x)) ∧ 
  (angle_QPR + 108 + x + 108 = 360) ∧
  (x = 36)) :=
begin
  use 36,
  intros P Q R angle_PQR angle_PRQ angle_QPR,
  split,
  assumption, -- x == angle_PQR
  split,
  assumption, -- x == angle_PRQ
  split,
  assumption, -- angle_QPR == 180 - 2 * x
  split,
  assumption, -- angle_QPR + 108 + x + 108 = 360
  exact rfl -- x == 36
end

end congruent_regular_pentagons_triangle_x_angle_l271_271520


namespace total_amount_spent_l271_271351

variables (D B : ℝ)

-- Conditions
def condition1 : Prop := B = 1.5 * D
def condition2 : Prop := D = B - 15

-- Question: Prove that the total amount they spent together is 75.00
theorem total_amount_spent (h1 : condition1 D B) (h2 : condition2 D B) : B + D = 75 :=
sorry

end total_amount_spent_l271_271351


namespace total_rooms_to_paint_l271_271622

-- Definitions based on conditions
def hours_per_room : ℕ := 8
def rooms_already_painted : ℕ := 8
def hours_to_paint_rest : ℕ := 16

-- Theorem statement
theorem total_rooms_to_paint :
  rooms_already_painted + hours_to_paint_rest / hours_per_room = 10 :=
  sorry

end total_rooms_to_paint_l271_271622


namespace min_value_xn_1012_l271_271524

noncomputable def f (x : ℝ) : ℝ :=
if h : x ∈ Icc (-3) 0 then 2^(-x)
else if h : x ∈ Icc 0 3 then 2^(-x)  /* Extend as per periodic property if needed */
else if x < -3 then f (x + 6)
else f (x - 6)

theorem min_value_xn_1012
  (x : ℕ → ℝ)
  (n : ℕ)
  (h1 : 0 ≤ x 1)
  (h2 : ∀ i j, i < j → x i < x j)
  (h3 : ∑ i in finset.range (n - 1), (f (x i) - f (x (i + 1))).abs = 2019) :
  x n = 1012 :=
sorry

end min_value_xn_1012_l271_271524


namespace total_games_played_l271_271214

-- Definitions based on given conditions
def num_players := 50
def players_per_game := 2

-- Theorem to prove the number of total games played
theorem total_games_played : (nat.choose num_players players_per_game) = 1225 :=
by
  sorry

end total_games_played_l271_271214


namespace perpendicular_bisects_third_side_l271_271537

open EuclideanGeometry

variable {ABC : Triangle}
variable {A B C : Point}
variable {D E M : Point}

-- Conditions
variable (hAD : altitude A BC)
variable (hBE : altitude B AC)
variable (hD : foot_of_altitude A BC D)
variable (hE : foot_of_altitude B AC E)
variable (hM : midpoint D E M)

-- Question: Prove that the perpendicular drawn from M bisects the side AB
theorem perpendicular_bisects_third_side :
  ∀ {A B C : Point} {D E M : Point},
  altitude A BC →
  altitude B AC →
  midpoint D E M →
  is_midpoint_of_perpendicular_intersection M A B :=
by
  sorry

end perpendicular_bisects_third_side_l271_271537


namespace area_of_region_l271_271515

noncomputable def integral_sqrt_2x (a b : ℝ) : ℝ :=
∫ x in a..b, Real.sqrt (2 * x)

noncomputable def integral_sqrt_2x_minus_x_plus_4 (a b : ℝ) : ℝ :=
∫ x in a..b, (Real.sqrt (2 * x) - x + 4)

theorem area_of_region : 
  let parabola_eqn : (ℝ × ℝ) → Prop := λ p, p.2 ^ 2 = 2 * p.1
  let line_eqn : (ℝ × ℝ) → Prop := λ p, p.2 = p.1 - 4 
in 
  integral_sqrt_2x 0 2 + integral_sqrt_2x_minus_x_plus_4 2 8 = 18 :=
by
  sorry

end area_of_region_l271_271515


namespace find_constant_k_l271_271090

theorem find_constant_k (S : ℕ → ℝ) (a : ℕ → ℝ) (k : ℝ)
  (h₁ : ∀ n, S n = 3 * 2^n + k)
  (h₂ : ∀ n, 1 ≤ n → a n = S n - S (n - 1))
  (h₃ : ∃ q, ∀ n, 1 ≤ n → a (n + 1) = a n * q ) :
  k = -3 := 
sorry

end find_constant_k_l271_271090


namespace card_movement_limit_l271_271943

-- Definitions for the problem conditions
def can_move_left (n : ℕ) (pos : ℕ) (rectangle : ℕ → option ℕ) : Prop :=
  rectangle pos = some n ∧ rectangle (pos + 1) = none

def num_moves (number_of_cards : ℕ) : ℕ :=
  ∑ n in finset.range number_of_cards, n

-- Problem statement in Lean
theorem card_movement_limit (number_of_cards : ℕ) (rectangle : ℕ → option ℕ) : 
  number_of_cards ≤ 1000 → 
  (∀ n pos, can_move_left n pos rectangle → n < number_of_cards) → 
  num_moves number_of_cards < 500000 :=
by
  intros h1 h2,
  sorry

end card_movement_limit_l271_271943


namespace sin_cos_sixth_l271_271122

theorem sin_cos_sixth (θ : ℝ) (h : Real.sin (2 * θ) = 1 / 3) :
  Real.sin θ ^ 6 + Real.cos θ ^ 6 = 11 / 12 :=
sorry

end sin_cos_sixth_l271_271122


namespace volume_pyramid_TACD_l271_271185

-- Definitions for the problem
variables (T A B C D : Type) [LinearOrder T]
variables (S r1 r2 : ℝ)

def is_parallel (BC AD : Type) : Prop := ∃ k : ℝ, ∀ b : BC, ∀ a : AD, b = k * a

def volume_of_pyramid
  (is_trapezoid_base : is_parallel BC AD)
  (distance_A_to_TCD : r1)
  (distance_B_to_TCD : r2)
  (area_TCD : S)
: ℝ :=
1/3 * S * (r1 + r2)

theorem volume_pyramid_TACD (is_trapezoid_base : is_parallel BC AD)
  (distance_A_to_TCD : r1)
  (distance_B_to_TCD : r2)
  (area_TCD : S)
: volume_of_pyramid is_trapezoid_base distance_A_to_TCD distance_B_to_TCD area_TCD = (S * (r1 + r2)) / 3 := sorry

end volume_pyramid_TACD_l271_271185


namespace sticks_form_equilateral_triangle_l271_271703

theorem sticks_form_equilateral_triangle {n : ℕ} (h1 : n ≥ 5) (h2 : n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5) :
  (∃ a b c, a = b ∧ b = c ∧ a + b + c = (n * (n + 1)) / 2) :=
by
  sorry

end sticks_form_equilateral_triangle_l271_271703


namespace part1_solution_set_part2_values_of_a_l271_271037

-- Parameters and definitions for the function and conditions
def f (x a : ℝ) := |x - a^2| + |x - (2*a + 1)|

-- Part (1) When a = 2, the inequality's solution set
theorem part1_solution_set (x : ℝ) : f x 2 ≥ 4 ↔ (x ≤ 3/2 ∨ x ≥ 11/2) := 
sorry

-- Part (2) Range of values for a given f(x, a) ≥ 4
theorem part2_values_of_a (a : ℝ) : 
∀ x, f x a ≥ 4 → (a ≤ -1 ∨ a ≥ 3) :=
sorry

end part1_solution_set_part2_values_of_a_l271_271037


namespace limit_fraction_simplification_l271_271299

theorem limit_fraction_simplification :
  (tendsto (fun x => (x^2 - 1) / (2 * x^2 - x - 1)) (𝓝 1) (𝓝 (2 / 3))) :=
begin
  -- we will provide the proof steps later
  sorry
end

end limit_fraction_simplification_l271_271299


namespace not_possible_draw_2005_vectors_with_zero_sum_l271_271098

-- Definitions and variables
variables {R : Type*} [field R] [decidable_eq R] (v : fin 2005 → (R × R))

-- Statement of the theorem
theorem not_possible_draw_2005_vectors_with_zero_sum (h : ∀ s : finset (fin 2005), s.card = 10 → ∃ t : finset (fin 2005), t ⊆ s ∧ t.card = 3 ∧ finset.sum t v = (0, 0)) :
  false :=
begin
  sorry
end

end not_possible_draw_2005_vectors_with_zero_sum_l271_271098


namespace regression_eq_decrease_l271_271742

theorem regression_eq_decrease (x : ℝ) (h : ∀ x, y = 2 - 3 * x) : 
  (y.eval (x + 1)) = y.eval x - 3 :=
  sorry

end regression_eq_decrease_l271_271742


namespace sticks_form_equilateral_triangle_l271_271713

theorem sticks_form_equilateral_triangle (n : ℕ) :
  (∃ (s : set ℕ), s = {1, 2, ..., n} ∧ ∃ t : ℕ → ℕ, t ∈ s ∧ t s ∩ { a, b, c} = ∅ 
   and t t  = t and a + b + {
    sum s = nat . multiplicity = S
}) <=> n

example: ℕ in {
  n ≥ 5 ∧ n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5 } equilateral triangle (sorry)
   sorry


Sorry  = fiancé in the propre case, {1,2, ..., n}

==> n.g.e. : natal sets {1, 2, ... n }  + elastical


end sticks_form_equilateral_triangle_l271_271713


namespace intersection_of_open_sets_unctbl_l271_271639

-- Definitions for Lean code from the conditions
def is_open (U : set ℝ) : Prop :=
  ∀ x ∈ U, ∃ a b : ℝ, x ∈ set.Ioo a b ∧ set.Ioo a b ⊆ U

def intersects (U V : set ℝ) : Prop :=
  ∃ x, x ∈ U ∧ x ∈ V

def intersects_01 (S : set ℝ) : Prop :=
  ∀ U, is_open U → intersects U (set.Ioo 0 1) → intersects U S

-- Lean theorem to prove the conclusion
theorem intersection_of_open_sets_unctbl (S : set ℝ) (T : ℕ → set ℝ)
  (h1 : intersects_01 S)
  (h2 : ∀ n, is_open (T n))
  (h3 : ∀ n, S ⊆ T n) :
  ¬ set.countable (⋂ n, T n) :=
sorry

end intersection_of_open_sets_unctbl_l271_271639


namespace rowing_speed_in_still_water_l271_271251

theorem rowing_speed_in_still_water (d t1 t2 : ℝ) 
  (h1 : d = 750) (h2 : t1 = 675) (h3 : t2 = 450) : 
  (d / t1 + (d / t2 - d / t1) / 2) = 1.389 := 
by
  sorry

end rowing_speed_in_still_water_l271_271251


namespace enclosed_area_is_correct_l271_271514

noncomputable def enclosed_area : ℝ :=
  ∫ x in 0..1, (Real.exp x - Real.exp (-x)) + ∫ x in 0..1, Real.exp (-x)

theorem enclosed_area_is_correct : enclosed_area = Real.exp 1 + Real.exp (-1) - 2 :=
by
  sorry

end enclosed_area_is_correct_l271_271514


namespace integral_even_function_l271_271369

variable (f : ℝ → ℝ)

def even_fun (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

theorem integral_even_function :
  even_fun f →
  ∫ x in 0..6, f x = 8 →
  ∫ x in -6..6, f x = 16 :=
by
  intro h_even h_int
  sorry

end integral_even_function_l271_271369


namespace ratio_of_amounts_l271_271641

variable (x y : ℕ) (total : ℕ := 5000) (x_amount : ℕ := 1000)

-- Conditions
def condition_one : total = x + y := rfl
def condition_two : x_amount = x := rfl

-- Theorem
theorem ratio_of_amounts (h1 : total = 5000) (h2 : x_amount = 1000) (hx : x = x_amount) (hy : y = total - x_amount) :
  x_amount * 4 = y :=
sorry

end ratio_of_amounts_l271_271641


namespace part1_solution_set_part2_range_of_a_l271_271026

def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1_solution_set (x : ℝ) : 
  (f x 2 ≥ 4) ↔ (x ≤ 3 / 2 ∨ x ≥ 11 / 2) :=
sorry

theorem part2_range_of_a (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 4) → (a ≤ -1 ∨ a ≥ 3) :=
sorry

end part1_solution_set_part2_range_of_a_l271_271026


namespace division_powers_5_half_division_powers_6_3_division_powers_formula_division_powers_combination_l271_271591

def f (n : ℕ) (a : ℚ) : ℚ := a ^ (2 - n)

theorem division_powers_5_half : f 5 (1/2) = 8 := by
  -- skip the proof
  sorry

theorem division_powers_6_3 : f 6 3 = 1/81 := by
  -- skip the proof
  sorry

theorem division_powers_formula (n : ℕ) (a : ℚ) (h : n > 0) : f n a = a^(2 - n) := by
  -- skip the proof
  sorry

theorem division_powers_combination : f 5 (1/3) * f 4 3 * f 5 (1/2) + f 5 (-1/4) / f 6 (-1/2) = 20 := by
  -- skip the proof
  sorry

end division_powers_5_half_division_powers_6_3_division_powers_formula_division_powers_combination_l271_271591


namespace integral_result_l271_271212

noncomputable def integral_value := ∫ x in -1..1, (x^2 * tan x + x^3 + 1)

theorem integral_result : integral_value = 2 :=
sorry

end integral_result_l271_271212


namespace division_remainder_l271_271722

theorem division_remainder (f g : Polynomial ℝ) (x : ℝ) :
  f = x^4 + 1 ∧ g = x^2 - 4x + 7 →
  Polynomial.modByMonic f g = 8*x - 62 :=
by
  sorry

end division_remainder_l271_271722


namespace fill_blanks_l271_271493

/-
Given the following conditions:
1. 20 * (x1 - 8) = 20
2. x2 / 2 + 17 = 20
3. 3 * x3 - 4 = 20
4. (x4 + 8) / 12 = y4
5. 4 * x5 = 20
6. 20 * (x6 - y6) = 100

Prove that:
1. x1 = 9
2. x2 = 6
3. x3 = 8
4. x4 = 4 and y4 = 1
5. x5 = 5
6. x6 = 7 and y6 = 2
-/
theorem fill_blanks (x1 x2 x3 x4 y4 x5 x6 y6 : ℕ) :
  20 * (x1 - 8) = 20 →
  x2 / 2 + 17 = 20 →
  3 * x3 - 4 = 20 →
  (x4 + 8) / 12 = y4 →
  4 * x5 = 20 →
  20 * (x6 - y6) = 100 →
  x1 = 9 ∧
  x2 = 6 ∧
  x3 = 8 ∧
  x4 = 4 ∧
  y4 = 1 ∧
  x5 = 5 ∧
  x6 = 7 ∧
  y6 = 2 :=
by
  sorry

end fill_blanks_l271_271493


namespace south_side_crazy_street_20th_house_l271_271529

def contains_digit_3 (n : ℕ) : Prop :=
  (n.to_string.contains '3')

def is_valid_house_number (n : ℕ) : Prop :=
  (n % 2 = 1) ∧ ¬ contains_digit_3 n

def valid_house_numbers (n : ℕ) : ℕ :=
  (List.range (2 * n).succ).filter is_valid_house_number ![n - 1]

theorem south_side_crazy_street_20th_house : valid_house_numbers 20 = 59 := 
  sorry

end south_side_crazy_street_20th_house_l271_271529


namespace find_a_for_d_eq_a_squared_l271_271681

def move_last_digit_to_first (a : ℕ) : ℕ :=
  let str_a := a.to_digits.reverse
  let len := str_a.length
  if len = 0 then 0
  else match str_a with
  | [] => 0
  | (last :: rest) => (last.to_nat * 10^(len - 1)) + rest.reverse.foldl (λ acc x, acc * 10 + x.to_nat) 0

def move_first_digit_to_end (c : ℕ) : ℕ :=
  let str_c := c.to_digits
  let len := str_c.length
  if len = 0 then 0
  else match str_c with
  | [] => 0
  | (first :: rest) => rest.foldl (λ acc x, acc * 10 + x.to_nat) 0 * 10 + first.to_nat

def d (a : ℕ) : ℕ :=
  move_first_digit_to_end ((move_last_digit_to_first a) ^ 2)

theorem find_a_for_d_eq_a_squared :
  {a : ℕ // d a = a ^ 2 } = {1, 2, 3, 222 .... 21}
:= sorry

end find_a_for_d_eq_a_squared_l271_271681


namespace find_m_min_distance_l271_271860

noncomputable def C1_parametric_x (t : ℝ) : ℝ := (1/2) * t
noncomputable def C1_parametric_y (t : ℝ) (m : ℝ) : ℝ := m + (real.sqrt 3 / 2) * t

def C2_cartesian (x y : ℝ) : Prop :=
  (x - real.sqrt 3)^2 + (y - 1)^2 = 4

theorem find_m_min_distance (m : ℝ) :
  ∀ t : ℝ, ∃ x y : ℝ, (y = m + real.sqrt 3 * x) → 
    (C2_cartesian x y) → 
    (abs (sqrt ((x - x)^2 + (y - y)^2)) = 1) → 
    m = 2 ∨ m = -6 :=
sorry

end find_m_min_distance_l271_271860


namespace number_of_zeros_in_expansion_of_square_l271_271409

theorem number_of_zeros_in_expansion_of_square (h : 999999999995 = 10^12 - 5) :
  let n := 999999999995 in 
  let z := 11 in 
  ∃ m : ℕ, m = (n^2) ∧ (∃ k : ℕ, k ≤ m ∧ ((10^13 - 10^3) * 10^n == m ∧ k == z)) := 
sorry

end number_of_zeros_in_expansion_of_square_l271_271409


namespace area_centroid_triangle_l271_271117

noncomputable def area_of_centroid_triangle (ABC : Triangle) (P : Point) 
  (hP_in_ABC : P ∈ ABC) (h_area_ABC : ABC.area = 24) : ℝ :=
let G1 := centroid (Triangle.mk P ABC.B ABC.C) in
let G2 := centroid (Triangle.mk P ABC.C ABC.A) in
let G3 := centroid (Triangle.mk P ABC.A ABC.B) in
(Triangle.mk G1 G2 G3).area

theorem area_centroid_triangle
  {ABC : Triangle}
  {P : Point}
  (hP_in_ABC : P ∈ ABC)
  (h_area_ABC : ABC.area = 24) :
  area_of_centroid_triangle ABC P hP_in_ABC h_area_ABC = 8/3 :=
sorry

end area_centroid_triangle_l271_271117


namespace problem_a_problem_b_l271_271252

-- Define the conditions for problem (a):
variable (x y z : ℝ)
variable (h_xyz : x * y * z = 1)

theorem problem_a (hx : x ≠ 1) (hy : y ≠ 1) (hz : z ≠ 1) :
  (x^2 / (x - 1)^2) + (y^2 / (y - 1)^2) + (z^2 / (z - 1)^2) ≥ 1 :=
sorry

-- Define the conditions for problem (b):
variable (a b c : ℚ)

theorem problem_b (h_abc : a * b * c = 1) :
  ∃ (x y z : ℚ), x ≠ 1 ∧ y ≠ 1 ∧ z ≠ 1 ∧ (x * y * z = 1) ∧ 
  (x^2 / (x - 1)^2 + y^2 / (y - 1)^2 + z^2 / (z - 1)^2 = 1) :=
sorry

end problem_a_problem_b_l271_271252


namespace min_value_of_parabola_in_interval_l271_271391

theorem min_value_of_parabola_in_interval :
  ∀ x : ℝ, -10 ≤ x ∧ x ≤ 0 → (x^2 + 12 * x + 35) ≥ -1 := by
  sorry

end min_value_of_parabola_in_interval_l271_271391


namespace magnitude_of_complex_l271_271971

def complex_number := Complex.mk 2 3 -- Define the complex number 2+3i

theorem magnitude_of_complex : Complex.abs complex_number = Real.sqrt 13 := by
  sorry

end magnitude_of_complex_l271_271971


namespace total_votes_l271_271851

variable (V : ℕ)
variable (ha : 45 * V / 100 - 25 * V / 100 = 800)

theorem total_votes (h : ha) : V = 4000 := sorry

end total_votes_l271_271851


namespace f_increasing_interval_k_values_l271_271177

noncomputable def f (x : ℝ) : ℝ := Real.log x - 0.5 * (x - 1)^2

theorem f_increasing_interval (x : ℝ) (h : x > 1) : x ∈ Ioo 1 (Real.sqrt 2) ↔ (∃ δ > 0, ∀ y ∈ Ioo x (x + δ), f y > f x) :=
sorry

theorem k_values (k : ℝ) : k < 1 ↔ ∀ x > 1, f x > k * (x - 1) :=
sorry

end f_increasing_interval_k_values_l271_271177


namespace categorizeNumbers_correct_l271_271686

variables {realNumbers : List ℝ} 
variable (positiveNumbers : Set ℝ)
variable (integers : Set ℤ)
variable (negativeFractions : Set ℚ)
variable (irrationals : Set ℝ)

noncomputable def isPositive (x : ℝ) : Prop := x > 0
noncomputable def isInteger (x : ℝ) : Prop := ∃ (z : ℤ), x = z
noncomputable def isNegativeFraction (x : ℝ) : Prop := ∃ (q : ℚ), x < 0 ∧ x = q
noncomputable def isIrrational (x : ℝ) : Prop := ¬ ∃ (q : ℚ), x = q

theorem categorizeNumbers_correct :
  realNumbers = [ -2.5, 0, 25, (1 / 9 : ℝ), -1.2121121112, -3 / 4, real.pi] →
  positiveNumbers = { x | isPositive x } →
  integers = { x | ∃ (z : ℤ), ↑z = x } →
  negativeFractions = { x | ∃ (q : ℚ), x < 0 ∧ ↑q = x } →
  irrationals = { x | isIrrational x } →
  positiveNumbers = { 25, (1 / 9), real.pi } ∧ 
  integers = { 0, 25 } ∧ 
  negativeFractions = { -2.5, -3 / 4} ∧ 
  irrationals = { -1.2121121112..., real.pi } := by
  sorry

end categorizeNumbers_correct_l271_271686


namespace instantaneous_velocity_at_2_l271_271198

noncomputable def s (t : ℝ) : ℝ := 3 + t^2

theorem instantaneous_velocity_at_2 : ∀ Δt, lim (Δt → 0) ((s (2 + Δt) - s 2) / Δt) = 4 := by
  sorry

end instantaneous_velocity_at_2_l271_271198


namespace tile_difference_l271_271460

theorem tile_difference :
  let initial_blue_tiles := 20
  let initial_green_tiles := 15
  let first_border_tiles := 18
  let second_border_tiles := 18
  let total_green_tiles := initial_green_tiles + first_border_tiles + second_border_tiles
  let total_blue_tiles := initial_blue_tiles
  total_green_tiles - total_blue_tiles = 31 := 
by
  sorry

end tile_difference_l271_271460


namespace find_value_l271_271084

-- Definitions of the parameters according to the conditions
def 同 := 1
def 心 := 0
def remaining_digits := [2, 3, 4, 5, 6, 7, 8, 9]

def 振 : ℕ := sorry
def 兴 : ℕ := sorry
def 中 : ℕ := sorry
def 华 : ℕ := sorry

def 两 : ℕ := sorry
def 岸 : ℕ := sorry
def 四 : ℕ := sorry
def 地 : ℕ := sorry

-- Define the sums
def x := 振 + 兴 + 中 + 华
def y := 两 + 岸 + 四 + 地

-- The theorem we will prove
theorem find_value : x = 27 :=
by
  have h1 : x + y = 44 := sorry
  have h2 : x = y + 10 := sorry
  have h3 := calc 
    y = 17 := sorry
    x = 27 := sorry
  sorry

end find_value_l271_271084


namespace max_knights_and_courtiers_l271_271450

theorem max_knights_and_courtiers (a b : ℕ) (ha : 12 ≤ a ∧ a ≤ 18) (hb : 10 ≤ b ∧ b ≤ 20) :
  (1 / a : ℚ) + (1 / b) = (1 / 7) → a = 14 ∧ b = 14 :=
begin
  -- Proof would go here.
  sorry
end

end max_knights_and_courtiers_l271_271450


namespace cauchy_schwarz_l271_271886

theorem cauchy_schwarz {n : ℕ} (x y : Fin n → ℝ) :
  abs (∑ i, x i * y i) ≤ sqrt (∑ i, (x i)^2) * sqrt (∑ i, (y i)^2) ∧ 
  (abs (∑ i, x i * y i) = sqrt (∑ i, (x i)^2) * sqrt (∑ i, (y i)^2) ↔ 
    ∃ λ : ℝ, ∀ i, x i = λ * y i) := 
sorry  -- Proof to be constructed later

end cauchy_schwarz_l271_271886


namespace equilateral_triangle_side_length_l271_271610

theorem equilateral_triangle_side_length
  (O A B C : Type)
  (area_circle : real)
  (OA_length : real)
  (equilateral : Prop)
  (chord_BC : Prop)
  (point_O_outside : Prop)
  (O_center: O)
  (A_point: O)
  (B_point: O)
  (C_point: O)
  (r : real)
  (s: real) :
  area_circle = 324 * real.pi ∧
  OA_length = 6 * real.sqrt 3 ∧
  equilateral ∧
  chord_BC ∧
  point_O_outside →
  r = 18 →
  s = 6 * real.sqrt 3 :=
by sorry

end equilateral_triangle_side_length_l271_271610


namespace tangent_line_at_e_intervals_and_extremes_of_g_range_of_a_l271_271382

noncomputable def f (x : ℝ) : ℝ := Real.log x - x
noncomputable def g (x : ℝ) : ℝ := f x + 2*x - 4*Real.log x - 2/x

-- (1) Equation of the tangent line to y = f(x) at x = e
theorem tangent_line_at_e :
∃ y : ℝ → ℝ, (y = λ x, (1/Real.exp(1) - 1) * x) :=
sorry

-- (2) Intervals of monotonicity and extreme values of g(x)
theorem intervals_and_extremes_of_g :
  (∀ x : ℝ, 0 < x ∧ x < 1 → g' x > 0) ∧
  (∀ x : ℝ, 1 < x ∧ x < 2 → g' x < 0) ∧
  (∀ x : ℝ, 2 < x → g' x > 0) ∧
  (g 1 = -1) ∧
  (g 2 = -3 * Real.log 2 + 1) :=
sorry

-- (3) Range of real numbers for a
theorem range_of_a :
(∀ x : ℝ, 0 < x → f x ≤ ((a - 1) * x + 1)) → (a ≥ 1/Real.exp(2)) :=
sorry

end tangent_line_at_e_intervals_and_extremes_of_g_range_of_a_l271_271382


namespace part1_part2_l271_271905

-- Define the condition for the first part
def condition (A B C : ℝ) : Prop :=
  (C = 2 * Real.pi / 3) ∧ (1 + Real.sin A ≠ 0) ∧ (2 * Real.cos B ≠ 0) ∧ 
  (Real.cos A / (1 + Real.sin A) = Real.sin (2 * B) / (1 + Real.cos (2 * B)))

-- Theorem for the first part: If \( C = \frac{2\pi}{3} \), then \( B = \frac{\pi}{6} \)
theorem part1 (A B C : ℝ) (h : condition A B C) : B = Real.pi / 6 :=
  sorry

-- Define the condition for the second part as the side ratios expression
def ratio_expression (a b c : ℝ) : ℝ :=
  (a^2 + b^2) / (c^2)

-- Theorem for the second part: Minimum value of \(\frac{a^2 + b^2}{c^2}\)
theorem part2 (a b c : ℝ) (A B C : ℝ) 
  (h : condition A B C) : ratio_expression a b c = 4 * Real.sqrt 2 - 5 :=
  sorry

end part1_part2_l271_271905


namespace hexagon_angle_l271_271183

theorem hexagon_angle (h : Sum [2, 3, 3, 4, 5, 6] = 720) : 
  6 * (720 / 23) = 4320 / 23 :=
by sorry

end hexagon_angle_l271_271183


namespace elevation_representation_l271_271321

-- Define constants for elevations
def mountEverest_elevation := 8844 -- meters
def marianaTrench_elevation := -11034 -- meters

-- Define conditions for elevations
axiom aboveSeaLevel_positive : ∀ x, x > 0 -> x isPositiveNumber
axiom belowSeaLevel_negative : ∀ x, x < 0 -> x isNegativeNumber

-- Main theorem to prove
theorem elevation_representation :
  (mountEverest_elevation > 0 ∧ mountEverest_elevation = 8844) ∧
  (marianaTrench_elevation < 0 ∧ marianaTrench_elevation = -11034) :=
sorry

end elevation_representation_l271_271321


namespace smallest_value_of_a_for_polynomial_l271_271976

theorem smallest_value_of_a_for_polynomial (r1 r2 r3 : ℕ) (h_prod : r1 * r2 * r3 = 30030) :
  (r1 + r2 + r3 = 54) ∧ (r1 * r2 * r3 = 30030) → 
  (∀ a, a = r1 + r2 + r3 → a ≥ 54) :=
by
  sorry

end smallest_value_of_a_for_polynomial_l271_271976


namespace max_knights_l271_271680

-- Define properties for a knight (truth-teller) and a liar (false-teller)
structure Person (i : ℕ) :=
  (number : ℝ)
  (is_knight : Prop)
  (first_statement_true : Prop := number > i)
  (second_statement_true : Prop := number < i)

def conditions : Prop :=
  ∃ (persons : Fin 10 → Person),
  (∀ n : Fin 10, persons n).is_knight → 
    ((persons n).first_statement_true ∧ ∃ m : Fin 10, (persons m).second_statement_true)

theorem max_knights (persons : Fin 10 → Person) : ∀ (n : Fin 10), persons n).is_knight → (∃ k ≤ 9, k = (persons.filter Person.is_knight).card) :=
  sorry

end max_knights_l271_271680


namespace floor_function_example_l271_271326

theorem floor_function_example : 
  let a := 15.3 in 
  (Int.floor (a * a)) - (Int.floor a) * (Int.floor a) = 9 :=
by 
  -- Introducing the "let" bindings in Lean scope
  let a := 15.3
  sorry

end floor_function_example_l271_271326


namespace trigonometric_identity_l271_271354

theorem trigonometric_identity (θ : ℝ) (h : sin θ + 2 * cos θ = 0) :
  (1 + sin (2 * θ)) / (cos θ ^ 2) = 1 :=
by 
  sorry

end trigonometric_identity_l271_271354


namespace intersection_of_A_and_B_l271_271112

-- Define the sets A and B
def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {x | ∃ a ∈ A, x = 2 * a - 1}

-- The main statement to prove
theorem intersection_of_A_and_B : A ∩ B = {1, 3} :=
by {
  sorry
}

end intersection_of_A_and_B_l271_271112


namespace distance_from_M_to_left_directrix_l271_271784

-- Define the ellipse equation
def ellipse_eq (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 3) = 1

-- Define being 2 units away from the right focus
def distance_to_right_focus (x y : ℝ) : Prop :=
  let f2 := (1, 0) in  -- right focus
  (x - f2.1)^2 + y^2 = 4  -- distance^2 = 2^2 = 4

-- Define the directrix condition
def distance_to_left_directrix (x y : ℝ) : Prop :=
  x / (1 / 2) = 4  -- taking y coordinate = 0, hence x = 4 when e = 1/2

-- The statement to be proven
theorem distance_from_M_to_left_directrix :
  ∃ (x y : ℝ), ellipse_eq x y ∧ distance_to_right_focus x y → distance_to_left_directrix x y :=
sorry -- Proof is omitted

end distance_from_M_to_left_directrix_l271_271784


namespace fraction_sum_of_decimal_l271_271199

theorem fraction_sum_of_decimal (a b : ℕ) (h₁ : 0.478125 = a / b) (h₂ : Nat.gcd a b = 1) :
  a = 153 ∧ b = 320 ∧ a + b = 473 :=
by
  sorry

end fraction_sum_of_decimal_l271_271199


namespace distribute_rabbits_l271_271980

-- Definitions of the problem conditions
def rabbit_names : List String := ["Peter", "Pauline", "Flopsie", "Mopsie", "Cotton-tail", "Thumper"]
def parent_rabbits : List String := ["Peter", "Pauline"]
def child_rabbits : List String := ["Flopsie", "Mopsie", "Cotton-tail", "Thumper"]

-- Main theorem statement
theorem distribute_rabbits :
  let stores := 5
  let parents := parent_rabbits
  let children := child_rabbits
  (∀ s, s ⊆ rabbit_names → s ∈ Finset (Fin 5)) → -- Every subset of rabbit names maps to a valid store number
  -- No store gets both a parent and a child
  (∀ s : Fin 5, ¬(∃ p ∈ parents, ∃ c ∈ children, p ∈ s ∧ c ∈ s)) →
  -- No two siblings end up in the same store
  (∀ s : Fin 5, ¬(∃ x ∈ children, ∃ y ∈ children, x ≠ y ∧ x ∈ s ∧ y ∈ s)) →
  -- The total number of ways to distribute the rabbits is 200.
  ∃ total_ways : Nat, total_ways = 200 :=
sorry

end distribute_rabbits_l271_271980


namespace outfit_choices_l271_271057

theorem outfit_choices (shirts pants hats colors : ℕ) 
  (h_shirts : shirts = 8) 
  (h_pants : pants = 8) 
  (h_hats : hats = 8) 
  (h_colors : colors = 8) 
  (h_not_same_color : shirts = pants ∧ ∀ (s p : ℕ), s ≠ p) : 
  ∃ (acceptable_outfits : ℕ), acceptable_outfits = 448 := 
by 
  use 448
  sorry

end outfit_choices_l271_271057


namespace range_of_k_l271_271042

theorem range_of_k (k : ℝ) (h₁ : k > 0) 
  (h₂ : ∀ (A B : ℝ × ℝ), 
         ((A.1 + A.2 - k = 0) ∧ (B.1 + B.2 - k = 0)) ∧ (A.1^2 + A.2^2 = 4) ∧ (B.1^2 + B.2^2 = 4) ∧ 
         (|((0, 0) - A) + ((0, 0) - B)| ≥ (sqrt 3) / 3 * |A - B|)) :
  k ∈ set.Ico (sqrt 2) (2 * sqrt 2) :=
sorry

end range_of_k_l271_271042


namespace ratio_black_white_extended_pattern_l271_271328

def originalBlackTiles : ℕ := 8
def originalWhiteTiles : ℕ := 17
def originalSquareSide : ℕ := 5
def extendedSquareSide : ℕ := 7
def newBlackTiles : ℕ := (extendedSquareSide * extendedSquareSide) - (originalSquareSide * originalSquareSide)
def totalBlackTiles : ℕ := originalBlackTiles + newBlackTiles
def totalWhiteTiles : ℕ := originalWhiteTiles

theorem ratio_black_white_extended_pattern : totalBlackTiles / totalWhiteTiles = 32 / 17 := sorry

end ratio_black_white_extended_pattern_l271_271328


namespace product_defect_rate_correct_l271_271964

-- Definitions for the defect rates of the stages
def defect_rate_stage1 : ℝ := 0.10
def defect_rate_stage2 : ℝ := 0.03

-- Definitions for the probability of passing each stage without defects
def pass_rate_stage1 : ℝ := 1 - defect_rate_stage1
def pass_rate_stage2 : ℝ := 1 - defect_rate_stage2

-- Definition for the overall probability of a product not being defective
def pass_rate_overall : ℝ := pass_rate_stage1 * pass_rate_stage2

-- Definition for the overall defect rate based on the above probabilities
def defect_rate_product : ℝ := 1 - pass_rate_overall

-- The theorem statement to be proved
theorem product_defect_rate_correct : defect_rate_product = 0.127 :=
by
  -- Proof here
  sorry

end product_defect_rate_correct_l271_271964


namespace base6_addition_l271_271234

theorem base6_addition :
  (2343:ℕ)₆ + (15325:ℕ)₆ = (22112:ℕ)₆ := sorry

end base6_addition_l271_271234


namespace lily_pad_half_lake_l271_271435

theorem lily_pad_half_lake
  (P : ℕ → ℝ) -- Define a function P(n) which represents the size of the patch on day n.
  (h1 : ∀ n, P n = P (n - 1) * 2) -- Every day, the patch doubles in size.
  (h2 : P 58 = 1) -- It takes 58 days for the patch to cover the entire lake (normalized to 1).
  : P 57 = 1 / 2 :=
by
  sorry

end lily_pad_half_lake_l271_271435


namespace david_tips_l271_271313

noncomputable def avg_tips_resort (tips_other_months : ℝ) (months : ℕ) := tips_other_months / months

theorem david_tips 
  (tips_march_to_july_september : ℝ)
  (tips_august_resort : ℝ)
  (total_tips_delivery_driver : ℝ)
  (total_tips_resort : ℝ)
  (total_tips : ℝ)
  (fraction_august : ℝ)
  (avg_tips := avg_tips_resort tips_march_to_july_september 6):
  tips_august_resort = 4 * avg_tips →
  total_tips_delivery_driver = 2 * avg_tips →
  total_tips_resort = tips_march_to_july_september + tips_august_resort →
  total_tips = total_tips_resort + total_tips_delivery_driver →
  fraction_august = tips_august_resort / total_tips →
  fraction_august = 1 / 2 :=
by
  sorry

end david_tips_l271_271313


namespace area_centroid_triangle_l271_271116

noncomputable def area_of_centroid_triangle (ABC : Triangle) (P : Point) 
  (hP_in_ABC : P ∈ ABC) (h_area_ABC : ABC.area = 24) : ℝ :=
let G1 := centroid (Triangle.mk P ABC.B ABC.C) in
let G2 := centroid (Triangle.mk P ABC.C ABC.A) in
let G3 := centroid (Triangle.mk P ABC.A ABC.B) in
(Triangle.mk G1 G2 G3).area

theorem area_centroid_triangle
  {ABC : Triangle}
  {P : Point}
  (hP_in_ABC : P ∈ ABC)
  (h_area_ABC : ABC.area = 24) :
  area_of_centroid_triangle ABC P hP_in_ABC h_area_ABC = 8/3 :=
sorry

end area_centroid_triangle_l271_271116


namespace area_of_quadrilateral_FDBG_l271_271993

theorem area_of_quadrilateral_FDBG
  (A B C D E F G : Type)
  (AB AC : ℝ)
  (area_ABC : ℝ)
  (mid_AB : D)
  (mid_AC : E)
  (angle_bisector_inter_de : F)
  (angle_bisector_inter_bc : G)
  (hAB : AB = 40)
  (hAC : AC = 20)
  (h_area_ABC : area_ABC = 160)
  (h_mid_AB : D = midpoint A B)
  (h_mid_AC : E = midpoint A C)
  (h_inter_de : F = angle_bisector A B C ∩ DE)
  (h_inter_bc : G = angle_bisector A B C ∩ BC) :
  area_of_quadrilateral F D B G = 256 / 3 := 
sorry

end area_of_quadrilateral_FDBG_l271_271993


namespace sum_of_reciprocal_gp_l271_271304

variable (n : ℕ) (r S : ℝ)

theorem sum_of_reciprocal_gp (hS : S = (1 - (2 * r)^(2 * n)) / (1 - 2 * r)) :
  let S_rec := (2 * r * (1 - 1 / (4^n * r^(2 * n)))) / (2 * r - 1)
  in S_rec = S / (2^n * r^(2 * n - 1)) :=
sorry

end sum_of_reciprocal_gp_l271_271304


namespace part1_solution_set_part2_range_of_a_l271_271002

-- Define the function f for a general a
def f (x a : ℝ) : ℝ := |x - a^2| + |x - (2 * a) + 1|

-- Part 1: When a = 2
theorem part1_solution_set (x : ℝ) (h : f x 2 ≥ 4) : x ≤ 3/2 ∨ x ≥ 11/2 :=
  sorry

-- Part 2: Range of values for a
theorem part2_range_of_a (a : ℝ) (h : ∀ x, f x a ≥ 4) : a ≤ -1 ∨ a ≥ 3 :=
  sorry

end part1_solution_set_part2_range_of_a_l271_271002


namespace line_plane_relationship_l271_271827

-- Define line l, line m and plane α
variable (l m : Affine.Line ℝ) (α : Affine.Plane ℝ)

-- Conditions:
-- 1. m is in plane α
-- 2. l is parallel to m
-- We'll use the Lean's predicate to check the conditions

-- Definition: Line m is in plane α
def line_in_plane (m : Affine.Line ℝ) (α : Affine.Plane ℝ) : Prop :=
  ∃ p : Affine.Point ℝ, p ∈ m ∧ p ∈ α

-- Definition: Line l is parallel to line m
def lines_parallel (l m : Affine.Line ℝ) : Prop := 
  ∀ p₁ p₂ : Affine.Point ℝ, p₁ ∈ l ∧ p₂ ∈ m → ∃ v : Affine.Vector ℝ, (p₂ -ᵥ p₁) ∥ v

-- Theorem: Relationship between line l and plane α given conditions
theorem line_plane_relationship 
  (h₁ : line_in_plane m α)
  (h₂ : lines_parallel l m) :
  (∀ p₁ : Affine.Point ℝ, p₁ ∈ l → p₁ ∈ α) ∨ (∀ q : Affine.Point ℝ, q ∈ l → ∃ v : Affine.Vector ℝ, (q -ᵥ some_point_in α) ∥ v) := 
sorry

end line_plane_relationship_l271_271827


namespace part1_part2_l271_271919

namespace MathProof

-- defining the basic setup of the triangle and given constraints
variables (A B C : ℝ) (a b c : ℝ)

-- condition 1: the equation relating cosines and sines
def condition1 : Prop := (cos A) / (1 + sin A) = (sin (2 * B)) / (1 + cos (2 * B))

-- condition 2: specific value of angle C
def condition2 : Prop := C = 2 * π / 3

-- question 1: finding angle B
def findB : Prop := B = π / 6

-- The minimum value of (a^2 + b^2) / c^2
def min_value_expr : ℝ := (a^2 + b^2) / c^2
def min_value : ℝ := 4 * real.sqrt 2 - 5

-- Proving both parts
theorem part1 (h1 : condition1) (h2 : condition2) : findB := sorry

theorem part2 (h1 : condition1) (h2 : condition2) : min_value_expr = min_value := sorry

end MathProof

end part1_part2_l271_271919


namespace no_valid_volume_l271_271627

theorem no_valid_volume (V : ℕ) (hV : V ∈ [240, 300, 400, 500, 600]) :
  ∀ (x : ℕ), 10 * x ^ 3 ≠ V := by
  intro x
  cases hV with
  | inl h₁ =>
    have : x ^ 3 ≠ 24 := sorry
    have : 10 * x ^ 3 ≠ 240 := sorry
  | inr h₂ =>
    cases h₂ with
    | inl h₃ =>
      have : x ^ 3 ≠ 30 := sorry
      have : 10 * x ^ 3 ≠ 300 := sorry
    | inr h₄ =>
      cases h₄ with
      | inl h₅ =>
        have : x ^ 3 ≠ 40 := sorry
        have : 10 * x ^ 3 ≠ 400 := sorry
      | inr h₆ =>
         cases h₆ with
        | inl h₇ =>
          have : x ^ 3 ≠ 50 := sorry
          have : 10 * x ^ 3 ≠ 500 := sorry
        | inr h₈ =>
          have : x ^ 3 ≠ 60 := sorry
          have : 10 * x ^ 3 ≠ 600 := sorry

end no_valid_volume_l271_271627


namespace brokerage_percentage_calculation_l271_271565

theorem brokerage_percentage_calculation
  (face_value : ℝ)
  (discount_percentage : ℝ)
  (cost_price : ℝ)
  (h_face_value : face_value = 100)
  (h_discount_percentage : discount_percentage = 6)
  (h_cost_price : cost_price = 94.2) :
  ((cost_price - (face_value - (discount_percentage / 100 * face_value))) / cost_price * 100) = 0.2124 := 
by
  sorry

end brokerage_percentage_calculation_l271_271565


namespace no_adjacent_AB_l271_271346

theorem no_adjacent_AB (s1 s2 s3 s4 s5 : String) (A B : String) :
  (s1 ≠ s2) ∧ (s2 ≠ s3) ∧ (s3 ≠ s4) ∧ (s4 ≠ s5) ∧ (s1 ≠ s3) ∧ (s2 ≠ s4) ∧ (s3 ≠ s5) ∧ (s1 ≠ s4) ∧ (s2 ≠ s5) ∧ (s1 ≠ s5) ∧ 
  (A ∈ {s1, s2, s3, s4, s5}) ∧
  (B ∈ {s1, s2, s3, s4, s5}) ∧
  ¬((s1 = A ∧ s2 = B) ∨ (s2 = A ∧ s3 = B) ∨ (s3 = A ∧ s4 = B) ∨ (s4 = A ∧ s5 = B) ∨
    (s1 = B ∧ s2 = A) ∨ (s2 = B ∧ s3 = A) ∨ (s3 = B ∧ s4 = A) ∨ (s4 = B ∧ s5 = A)) 
  → finset.card {s1, s2, s3, s4, s5} = 72 :=
by
  sorry

end no_adjacent_AB_l271_271346


namespace annual_increase_rate_l271_271683

theorem annual_increase_rate (r : ℝ) (h : 70400 * (1 + r)^2 = 89100) : r = 0.125 :=
sorry

end annual_increase_rate_l271_271683


namespace find_p_plus_q_l271_271975

noncomputable def p_plus_q : ℕ :=
  let ABC := 170 in
  let BAC := 90 in
  let radius := 17 in
  let O := sorry in -- Don't define O explicitly, leave it for the proof
  let OB := sorry in -- OB should be defined within the proof context
  if relatively_prime (numerator OB) (denominator OB) then
    (numerator OB) + (denominator OB)
  else
    sorry

theorem find_p_plus_q 
  (ABC_perimeter : ∀ (A B C : ℝ), A + B + C = 170)
  (angle_BAC_right : ∀ (B A C : ℝ), A^2 + B^2 = C^2)
  (circle_tangent : ∃ (O : ℝ), ∀ (A B C : ℝ), dist O A = 17 ∧ dist O B = 17) :
  p_plus_q = 142 := sorry

end find_p_plus_q_l271_271975


namespace curve_M_cartesian_eq_proof_curve_N_cartesian_eq_proof_min_distance_between_AB_l271_271043

noncomputable def curve_M_parametric_eq (α : ℝ) : ℝ × ℝ :=
  (2 * Real.cos α, 2 + 2 * Real.sin α)

noncomputable def curve_M_cartesian_eq (x y : ℝ) : Prop :=
  x^2 + (y - 2)^2 = 4

noncomputable def curve_N_polar_eq (ρ θ : ℝ) : Prop :=
  ρ * Real.sin (θ + Real.pi / 3) = 8

noncomputable def curve_N_cartesian_eq (x y : ℝ) : Prop :=
  sqrt 3 * x + y - 16 = 0

theorem curve_M_cartesian_eq_proof (α : ℝ) :
  ∃ (x y : ℝ), curve_M_parametric_eq α = (x, y) ∧ curve_M_cartesian_eq x y :=
sorry

theorem curve_N_cartesian_eq_proof (ρ θ : ℝ) :
  ∃ (x y : ℝ), curve_N_polar_eq ρ θ ∧ curve_N_cartesian_eq x y :=
sorry

noncomputable def distance_from_center_to_line (x₀ y₀ : ℝ) (a b c : ℝ) : ℝ :=
  Real.abs (a * x₀ + b * y₀ + c) / Real.sqrt (a^2 + b^2)

noncomputable def radius_M : ℝ := 2

noncomputable def center_M : ℝ × ℝ := (0, 2)

theorem min_distance_between_AB : 
  let (x₀, y₀) := center_M in 
  let d := distance_from_center_to_line x₀ y₀ (sqrt 3) 1 (-16) in
  d - radius_M = 5 :=
sorry

end curve_M_cartesian_eq_proof_curve_N_cartesian_eq_proof_min_distance_between_AB_l271_271043


namespace corrected_mean_l271_271604

theorem corrected_mean (mean : ℝ) (n : ℕ) (wrong_obs correct_obs : ℝ) 
    (initial_mean : mean = 36) (num_observations : n = 50) 
    (wrong_observation : wrong_obs = 21) (correct_observation : correct_obs = 48) : 
    (initial_corrected_mean : ((mean * n + (correct_obs - wrong_obs)) / n = 36.54)) :=
begin
  sorry
end

end corrected_mean_l271_271604


namespace joe_lowest_score_dropped_l271_271254

theorem joe_lowest_score_dropped (A B C D : ℕ) 
  (h1 : A + B + C + D = 160)
  (h2 : A + B + C = 135) 
  (h3 : D ≤ A ∧ D ≤ B ∧ D ≤ C) :
  D = 25 :=
sorry

end joe_lowest_score_dropped_l271_271254


namespace _l271_271746

@[simp] theorem upper_base_length (ABCD is_trapezoid: Boolean)
  (point_M: Boolean)
  (perpendicular_DM_AB: Boolean)
  (MC_eq_CD: Boolean)
  (AD_eq_d: ℝ)
  : BC = d / 2 := sorry

end _l271_271746


namespace derivative_of_f_l271_271415

noncomputable def f (x : ℝ) : ℝ := Real.exp (-x) * (Real.cos x + Real.sin x)

theorem derivative_of_f (x : ℝ) : deriv f x = -2 * Real.exp (-x) * Real.sin x :=
by sorry

end derivative_of_f_l271_271415


namespace max_value_of_function_l271_271592

/-- The maximum value of the function y = x + cos(2x) in the interval (0, π/4) is π/12 + √3/2. --/
theorem max_value_of_function : 
  ∃ x ∈ set.Ioo 0 (π / 4), 
  is_local_max (λ x, x + Real.cos (2 * x)) x ∧ (x + Real.cos (2 * x) = (π / 12 + Real.sqrt 3 / 2)) :=
sorry

end max_value_of_function_l271_271592


namespace find_B_min_of_sum_of_squares_l271_271922

-- Given conditions in a)
variables {A B C a b c : ℝ}
hypothesis (h1 : C = 2 * Real.pi / 3)
hypothesis (h2 : (Real.cos A) / (1 + Real.sin A) = Real.sin (2 * B) / (1 + Real.cos (2 * B)))

-- Part (1) prove B = π / 6
theorem find_B (h1 : C = 2 * Real.pi / 3) (h2 : (Real.cos A) / (1 + Real.sin A) = Real.sin (2 * B) / (1 + Real.cos (2 * B))) : B = Real.pi / 6 :=
by sorry

-- Part (2) find the minimum value of (a^2 + b^2) / (c^2)
theorem min_of_sum_of_squares (h1 : C = 2 * Real.pi / 3) (h2 : (Real.cos A) / (1 + Real.sin A) = Real.sin (2 * B) / (1 + Real.cos (2 * B))) : ∃ (m : ℝ), m = 4 * Real.sqrt 2 - 5 ∧ ∀ a b c, a > 0 ∧ b > 0 ∧ c > 0 → (a^2 + b^2) / c^2 ≥ m :=
by sorry

end find_B_min_of_sum_of_squares_l271_271922


namespace symmetric_line_eq_l271_271367

variable (a b : ℝ)

def P : ℝ × ℝ := (a, b)
def Q : ℝ × ℝ := (b - 1, a + 1)

theorem symmetric_line_eq :
  symmetric P Q l → equation_of_line l = (x - y + 1 = 0) := 
  sorry

end symmetric_line_eq_l271_271367


namespace particle_horizontal_distance_l271_271279

theorem particle_horizontal_distance :
  ∀ (x y : ℝ),
  (y = x^3 - 3*x^2 - x + 3) →
  ((∃ xP, y = 5 ∧ xP^3 - 3*xP^2 - xP + 3 = 5 ∧ (xP = 1 ∨ xP = 1 + sqrt 3 ∨ xP = 1 - sqrt 3)) →
   (∃ xQ, y = -2 ∧ xQ^3 - 3*xQ^2 - xQ + 3 = -2 ∧ (xQ = 1 ∨ xQ = 1 + sqrt 6 ∨ xQ = 1 - sqrt 6)) →
   ∃ d, d = abs (sqrt 6 - sqrt 3)) sorry

end particle_horizontal_distance_l271_271279


namespace correct_pair_l271_271248

-- Define the equation and the check for the specific pair
def equation (x y : ℕ) : Prop := 5 * x + 4 * y = 14

theorem correct_pair : equation 2 1 := 
by {
  have h : 5 * 2 + 4 * 1 = 14 := by decide,
  exact h
}

end correct_pair_l271_271248


namespace g_2010_l271_271882

noncomputable def g : ℝ → ℝ := sorry

axiom g_defined_for_all_pos (x : ℝ) (hx : 0 < x) : 0 < g(x)

axiom functional_equation (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : y < x) :
  g(x - y) = (g(x * y) + 1)^(1/3)

theorem g_2010 : g 2010 = 1 := 
by
  sorry

end g_2010_l271_271882


namespace integral_sqrt_quarter_circle_l271_271660

noncomputable def integral_result : ℝ :=
  ∫ x in 0..1, real.sqrt (1 - x^2)

theorem integral_sqrt_quarter_circle :
  integral_result = real.pi / 4 := 
sorry

end integral_sqrt_quarter_circle_l271_271660


namespace find_quadratic_polynomial_l271_271720

theorem find_quadratic_polynomial :
  ∃ p : ℝ → ℝ, (∀ x, p x = 2 * x^2 - 3 * x - 1) ∧
  (p (-2) = 13) ∧ (p 1 = -2) ∧ (p 3 = 8) := 
by
  let p := λ x : ℝ, 2 * x^2 - 3 * x - 1
  use p
  split
  · intros x
    refl
  split
  · show p (-2) = 13
    sorry
  split
  · show p 1 = -2
    sorry
  · show p 3 = 8
    sorry

end find_quadratic_polynomial_l271_271720


namespace find_f_of_2_l271_271733

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 5

theorem find_f_of_2 : f 2 = 5 := by
  sorry

end find_f_of_2_l271_271733


namespace sam_pens_count_l271_271845

-- Lean 4 statement
theorem sam_pens_count :
  ∃ (black_pens blue_pens pencils red_pens : ℕ),
    (black_pens = blue_pens + 10) ∧
    (blue_pens = 2 * pencils) ∧
    (pencils = 8) ∧
    (red_pens = pencils - 2) ∧
    (black_pens + blue_pens + red_pens = 48) :=
by {
  sorry
}

end sam_pens_count_l271_271845


namespace find_ab_l271_271826

theorem find_ab (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 39) : a * b = 15 :=
by
  sorry

end find_ab_l271_271826


namespace simplify_and_evaluate_expression_l271_271174

theorem simplify_and_evaluate_expression (m : ℝ) (h : m = 2):
  ( ( (2 * m + 1) / m - 1 ) / ( (m^2 - 1) / m ) ) = 1 :=
by
  rw [h] -- Replace m by 2
  sorry

end simplify_and_evaluate_expression_l271_271174


namespace number_of_people_l271_271994

theorem number_of_people (x y z : ℕ) 
  (h1 : x + y + z = 12) 
  (h2 : 2 * x + y / 2 + z / 4 = 12) : 
  x = 5 ∧ y = 1 ∧ z = 6 := 
by
  sorry

end number_of_people_l271_271994


namespace range_of_a_l271_271389

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  real.exp x - a * (1/2 * x^2 - x)

theorem range_of_a (a : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 
    ∃ f' : ℝ → ℝ, 
      (∀ x, f' x = real.exp x - a * (x - 1)) ∧ 
      (f' x1 = 0 ∧ f' x2 = 0)) ↔ a ∈ set.Ioi (real.exp 2) :=
sorry

end range_of_a_l271_271389


namespace formation_enthalpy_benzene_l271_271257

/-- Define the enthalpy changes based on given conditions --/
def ΔH_acetylene : ℝ := 226.7 -- kJ/mol for C₂H₂
def ΔH_benzene_formation : ℝ := 631.1 -- kJ for reactions forming C₆H₆
def ΔH_benzene_phase_change : ℝ := -33.9 -- kJ for phase change of C₆H₆

/-- Define the enthalpy change of formation for benzene --/
def ΔH_formation_benzene : ℝ := 3 * ΔH_acetylene + ΔH_benzene_formation + ΔH_benzene_phase_change

/-- Theorem stating the heat change in the reaction equals the calculated value --/
theorem formation_enthalpy_benzene :
  ΔH_formation_benzene = -82.9 :=
by
  sorry

end formation_enthalpy_benzene_l271_271257


namespace general_term_l271_271890

noncomputable def sequence (n : ℕ) : ℝ :=
match n with
| 0 => 1
| k + 1 => (sequence k - 1) / (2 * sequence k + 4)

theorem general_term : ∀ n : ℕ, 
  sequence (n + 1) = (2^(n) - 3^(n)) / (2 * 3^(n) - 2^(n)) :=
by
  sorry

end general_term_l271_271890


namespace integral_of_semicircle_eq_49pi_over_2_l271_271424

theorem integral_of_semicircle_eq_49pi_over_2 
  (n : ℕ)
  (h : ∃ r : ℕ, 3 * n = (7 / 2 : ℚ) * r)
  (hn_min : ∀ m : ℕ, (∃ r : ℕ, 3 * m = (7 / 2 : ℚ) * r) → n ≤ m) :
  ∫ x in -n..n, sqrt (n^2 - x^2) = 49 * π / 2 :=
sorry

end integral_of_semicircle_eq_49pi_over_2_l271_271424


namespace min_num_cubes_to_construct_l271_271310

noncomputable def minimum_cubes (share_faces : (Nat → Nat → Bool)) (front_view : List (Nat × Nat))
  (side_view : List (Nat × Nat)) : Nat :=
  if (all_cubes_share_faces share_faces) ∧ (matches_front_view front_view) ∧ (matches_side_view side_view)
  then 3
  else sorry

axiom all_cubes_share_faces : (Nat → Nat → Bool) → Prop

axiom matches_front_view : List (Nat × Nat) → Prop

axiom matches_side_view : List (Nat × Nat) → Prop

theorem min_num_cubes_to_construct (share_faces: (Nat → Nat → Bool))
  (front_view : List (Nat × Nat)) (side_view : List (Nat × Nat))
  (h1 : all_cubes_share_faces share_faces)
  (h2 : matches_front_view front_view)
  (h3 : matches_side_view side_view) :
  minimum_cubes share_faces front_view side_view = 3 :=
by 
  sorry

end min_num_cubes_to_construct_l271_271310


namespace tax_increase_proof_l271_271615

variables (old_tax_rate new_tax_rate : ℝ) (old_income new_income : ℝ)

def old_taxes_paid (old_tax_rate old_income : ℝ) : ℝ := old_tax_rate * old_income

def new_taxes_paid (new_tax_rate new_income : ℝ) : ℝ := new_tax_rate * new_income

def increase_in_taxes (old_tax_rate new_tax_rate old_income new_income : ℝ) : ℝ :=
  new_taxes_paid new_tax_rate new_income - old_taxes_paid old_tax_rate old_income

theorem tax_increase_proof :
  increase_in_taxes 0.20 0.30 1000000 1500000 = 250000 := by
  sorry

end tax_increase_proof_l271_271615


namespace correct_proposition_among_abcd_l271_271292

theorem correct_proposition_among_abcd :
  (∀ x ∈ Ioo 0 (π / 4), sin x > cos x ↔ ¬ ∃ x₀ ∈ Ioo 0 (π / 4), sin x₀ ≤ cos x₀) ∧
  (∀ x, sin x + cos x ≤ sqrt 2) ∧
  (∀ a b : ℝ, a + b = 0 ↔ a = b ∧ a = 0) ∧
  (¬ (∀ x : ℝ, 2 * cos (x - π / 4) ^ 2 - 1 = sin 2 * x) ∧ ¬ (∀ x : ℝ, 2 * cos (x - π / 4) ^ 2 - 1 = - sin 2 * x)) →
  (∀ x, sin x + cos x ≤ sqrt 2) :=
begin
  sorry
end

end correct_proposition_among_abcd_l271_271292


namespace sticks_form_equilateral_triangle_l271_271710

theorem sticks_form_equilateral_triangle (n : ℕ) :
  (∃ (s : set ℕ), s = {1, 2, ..., n} ∧ ∃ t : ℕ → ℕ, t ∈ s ∧ t s ∩ { a, b, c} = ∅ 
   and t t  = t and a + b + {
    sum s = nat . multiplicity = S
}) <=> n

example: ℕ in {
  n ≥ 5 ∧ n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5 } equilateral triangle (sorry)
   sorry


Sorry  = fiancé in the propre case, {1,2, ..., n}

==> n.g.e. : natal sets {1, 2, ... n }  + elastical


end sticks_form_equilateral_triangle_l271_271710


namespace sticks_form_equilateral_triangle_l271_271704

theorem sticks_form_equilateral_triangle (n : ℕ) :
  n ≥ 5 → (n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5) → 
  ∃ k : ℕ, ∑ i in finset.range (n + 1), i = 3 * k :=
by {
  sorry
}

end sticks_form_equilateral_triangle_l271_271704


namespace part1_problem_l271_271931

theorem part1_problem
  (A B C : Real.Angle)
  (a b c : ℝ)
  (cosA : Real.cos A)
  (sinA : Real.sin A)
  (sin2B : Real.sin (2 * B))
  (cos2B : Real.cos (2 * B))
  (hC : C = 2 * π / 3)
  (h : cosA / (1 + sinA) = sin2B / (1 + cos2B))
  : B = π / 6 := by
  sorry

end part1_problem_l271_271931


namespace modulus_of_z_l271_271472

theorem modulus_of_z (z : ℂ) (i : ℂ) (h1 : i = complex.I) (h2 : (1 - i) * z = 2 * i) : complex.abs z = real.sqrt 2 := 
by sorry

end modulus_of_z_l271_271472


namespace parallel_planes_x_plus_y_l271_271788

def planes_parallel (x y : ℝ) : Prop :=
  ∃ k : ℝ, (x = -k) ∧ (1 = k * y) ∧ (-2 = (1 / 2) * k)

theorem parallel_planes_x_plus_y (x y : ℝ) (h : planes_parallel x y) : x + y = 15 / 4 :=
sorry

end parallel_planes_x_plus_y_l271_271788


namespace translation_a_b_equals_neg8_l271_271772

theorem translation_a_b_equals_neg8
    (a b : ℝ)
    (A : ℝ × ℝ) (B : ℝ × ℝ)
    (A1 : ℝ × ℝ) (B1 : ℝ × ℝ)
    (transA : A = (1, -3))
    (transB : B = (2, 1))
    (transA1 : A1 = (a, 2))
    (transB1 : B1 = (-1, b))
    (valid_translation : (A1.1 - A1.0 = A.0 - A.0) ∧ (A1.2 - A.2 = A.1 - A.1) ∧ 
                         (B1.1 - B1.0 = B.0 - B.0) ∧ (B1.2 - B.2 = B.1 - B.1)) :
    a - b = -8 :=
begin
  sorry
end

end translation_a_b_equals_neg8_l271_271772


namespace median_of_multiples_of_10_from_10_to_5000_mode_of_digits_in_multiples_of_10_from_10_to_5000_l271_271239

theorem median_of_multiples_of_10_from_10_to_5000 :
  ∃ (median : ℕ), median = 2510 ∧
  (∀ (n : ℕ), (n ≥ 10 ∧ n ≤ 5000 ∧ n % 10 = 0) → 
    List.sorted_insert n multiples_of_10 = List.range 10 501) :=
sorry

theorem mode_of_digits_in_multiples_of_10_from_10_to_5000 :
  ∃ (mode : ℕ), mode = 0 ∧
  Multiset.mode (Multiset.bind (Finset.Icc 10 5000) (λ n, (n % 10)::𝓐)) = 0 :=
sorry

end median_of_multiples_of_10_from_10_to_5000_mode_of_digits_in_multiples_of_10_from_10_to_5000_l271_271239


namespace min_max_cos_inequality_proof_l271_271467

noncomputable def min_max_cos_inequality 
  (n : ℕ) 
  (h : n > 0) 
  (x : Fin n.succ → ℝ) 
  (h_pos : ∀ (i : Fin n.succ), 0 < x i) 
  : Prop := 
  min (List.cons (x 0) ((List.finRange n).map (λ i, (1 / x i.castSucc.succ) + x i.succ))) 
      ≤ 2 * Real.cos (π / (n + 2)) ∧ 
  2 * Real.cos (π / (n + 2)) ≤ 
  max (List.cons (x 0) ((List.finRange n).map (λ i, (1 / x i.castSucc.succ) + x i.succ)))

theorem min_max_cos_inequality_proof 
  (n : ℕ) 
  (h : n > 0) 
  (x : Fin n.succ → ℝ) 
  (h_pos : ∀ (i : Fin n.succ), 0 < x i) 
  : min_max_cos_inequality n h x h_pos := 
sorry

end min_max_cos_inequality_proof_l271_271467


namespace cube_sum_identity_l271_271582

theorem cube_sum_identity (a b : ℝ) (h₁ : a + b = 12) (h₂ : a * b = 20) : a^3 + b^3 = 1008 := 
by 
 sorry

end cube_sum_identity_l271_271582


namespace ethan_book_page_count_l271_271324

-- Define the conditions
def read_on_saturday_morning := 40
def read_on_saturday_night := 10
def read_next_day := 2 * (read_on_saturday_morning + read_on_saturday_night)
def pages_left_to_read := 210

-- Define the goal (question)
def total_pages_in_book := read_on_saturday_morning + read_on_saturday_night + read_next_day + pages_left_to_read

-- The theorem to prove
theorem ethan_book_page_count :
  total_pages_in_book = 360 :=
begin
  sorry
end

end ethan_book_page_count_l271_271324


namespace smallest_multiple_17_mod_71_l271_271571

theorem smallest_multiple_17_mod_71 (a : ℕ) : 
  (17 * a ≡ 3 [MOD 71]) → 17 * a = 1139 :=
by
  intro h
  sorry

end smallest_multiple_17_mod_71_l271_271571


namespace proof_problem_l271_271141

open Set

variable {U : Set ℕ} {A : Set ℕ} {B : Set ℕ}

def problem_statement (U A B : Set ℕ) : Prop :=
  ((U \ A) ∪ B) = {2, 3}

theorem proof_problem :
  problem_statement {0, 1, 2, 3} {0, 1, 2} {2, 3} :=
by
  unfold problem_statement
  simp
  sorry

end proof_problem_l271_271141


namespace pentagon_triangles_l271_271407

theorem pentagon_triangles (vertices : ℕ) (adjacent_edges : ℕ) (non_adjacent_edges: ℕ) (diagonals: ℕ) :
  vertices = 5 ∧ adjacent_edges = 2 ∧ non_adjacent_edges = 3 ∧ diagonals = 2 → 
  ∃ triangles : ℕ, triangles = 3 :=
by
  intro h,
  cases h with h_vertices h_adjacent,
  cases h_adjacent with h_adjacent_edges h_non_adjacent,
  cases h_non_adjacent with h_non_adjacent_edges h_diagonals,
  use 3,
  sorry

end pentagon_triangles_l271_271407


namespace max_triangle_area_l271_271868

theorem max_triangle_area (AB BC AC : ℝ) (h1 : AB = 16) (h2 : ∃ x, BC = 3 * x ∧ AC = 4 * x ∧ x > 16 / 7 ∧ x < 16) :
  ∃ (x : ℝ), let BC := 3 * x
               let AC := 4 * x
               let s := (AB + BC + AC) / 2
               let area := s * (s - AB) * (s - BC) * (s - AC)
               area ≤ 128^2 :=
by
  sorry

end max_triangle_area_l271_271868


namespace jasmine_bottle_counts_l271_271871

def small_bottle_volume : ℕ := 25
def medium_bottle_volume : ℕ := 75
def large_bottle_volume : ℕ := 600

theorem jasmine_bottle_counts :
  let num_medium_bottles := large_bottle_volume / medium_bottle_volume in
  let total_bottles := num_medium_bottles in
  total_bottles = 8 :=
by 
  sorry

end jasmine_bottle_counts_l271_271871


namespace player_a_winning_strategy_l271_271631

theorem player_a_winning_strategy (P : ℝ) : 
  (∃ m n : ℕ, P = m / (2 ^ n) ∧ m < 2 ^ n)
  ∨ P = 0
  ∨ P = 1 ↔
  (∀ d : ℝ, ∃ d_direction : ℤ, 
    (P + (d * d_direction) = 0) ∨ (P + (d * d_direction) = 1)) :=
sorry

end player_a_winning_strategy_l271_271631


namespace find_upper_base_length_l271_271754

variables {A B C D M : Type}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space M]
variables (AB: line_segment A B) (CD: line_segment C D)
variables (AD : ℝ) (d : ℝ)

noncomputable def upper_base_length (A B C D M : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space M]
  (ABCD : bool)
  (DM_perp_AB : line_segment D M ⊥ line_segment A B)
  (MC_eq_CD : line_segment M C = line_segment C D)
  (AD_length : line_segment A D)
  (d_value : AD_length = d)
: Prop :=
BC = d / 2

theorem find_upper_base_length :
∀ (A B C D M : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space M]
  (ABCD : bool)
  (DM_perp_AB : line_segment D M ⊥ line_segment A B)
  (MC_eq_CD : line_segment M C = line_segment C D)
  (AD_length : line_segment A D)
  (d_value : AD_length = d),
  upper_base_length A B C D M ABCD DM_perp_AB MC_eq_CD AD_length d_value := sorry

end find_upper_base_length_l271_271754


namespace problem_statement_l271_271478

theorem problem_statement (g : ℝ → ℝ) :
  (∀ x y : ℝ, g (g x + y) = g (x + y) + x * g y - 2 * x * y - x + 2) →
  (∃ m t : ℝ, m = 1 ∧ t = 3 ∧ m * t = 3) :=
sorry

end problem_statement_l271_271478


namespace angle_sum_impossible_l271_271413

theorem angle_sum_impossible (A1 A2 A3 : ℝ) (h : A1 + A2 + A3 = 180) :
  ¬ ((A1 > 90 ∧ A2 > 90 ∧ A3 < 90) ∨ (A1 > 90 ∧ A3 > 90 ∧ A2 < 90) ∨ (A2 > 90 ∧ A3 > 90 ∧ A1 < 90)) :=
sorry

end angle_sum_impossible_l271_271413


namespace households_using_both_brands_l271_271275

def total : ℕ := 260
def neither : ℕ := 80
def onlyA : ℕ := 60
def onlyB (both : ℕ) : ℕ := 3 * both

theorem households_using_both_brands (both : ℕ) : 80 + 60 + both + onlyB both = 260 → both = 30 :=
by
  intro h
  sorry

end households_using_both_brands_l271_271275


namespace infinitely_many_lines_through_P_perpendicular_to_l_within_alpha_l271_271740

-- Definitions based on the conditions
variables {Point Line Plane : Type}
variable α : Plane
variable l : Line
variable P : Point
variable perpendicular : Line → Plane → Prop
variable on_plane : Point → Plane → Prop
variable line_through : Point → Line → Line → Prop

-- Given conditions
axiom l_perpendicular_to_alpha : perpendicular l α
axiom P_on_alpha : on_plane P α

-- The statement to be proven
theorem infinitely_many_lines_through_P_perpendicular_to_l_within_alpha :
  ∃ (L : Line), (line_through P L l) ∧ (∀ L', line_through P L' l → on_plane P α) :=
sorry

end infinitely_many_lines_through_P_perpendicular_to_l_within_alpha_l271_271740


namespace transformed_ln_function_correct_l271_271224

-- Define the original function
def original_function (x : ℝ) : ℝ := log (x + 1)

-- Define the stretch transformation
def stretch_x (f : ℝ → ℝ) (factor : ℝ) (x : ℝ) : ℝ := f (x / factor)

-- Define the shift transformation
def shift_x (f : ℝ → ℝ) (shift : ℝ) (x : ℝ) : ℝ := f (x - shift)

-- Define the transformed function
def transformed_function (x : ℝ) : ℝ := log ((x + 2) / 3)

-- Statement to prove
theorem transformed_ln_function_correct :
  stretch_x (shift_x original_function 1) 3 = transformed_function :=
sorry

end transformed_ln_function_correct_l271_271224


namespace find_upper_base_length_l271_271752

variables {A B C D M : Type}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space M]
variables (AB: line_segment A B) (CD: line_segment C D)
variables (AD : ℝ) (d : ℝ)

noncomputable def upper_base_length (A B C D M : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space M]
  (ABCD : bool)
  (DM_perp_AB : line_segment D M ⊥ line_segment A B)
  (MC_eq_CD : line_segment M C = line_segment C D)
  (AD_length : line_segment A D)
  (d_value : AD_length = d)
: Prop :=
BC = d / 2

theorem find_upper_base_length :
∀ (A B C D M : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space M]
  (ABCD : bool)
  (DM_perp_AB : line_segment D M ⊥ line_segment A B)
  (MC_eq_CD : line_segment M C = line_segment C D)
  (AD_length : line_segment A D)
  (d_value : AD_length = d),
  upper_base_length A B C D M ABCD DM_perp_AB MC_eq_CD AD_length d_value := sorry

end find_upper_base_length_l271_271752


namespace julia_played_tag_with_20_kids_l271_271110

theorem julia_played_tag_with_20_kids 
  (kids_monday : ℕ) (kids_tuesday : ℕ)
  (h_monday : kids_monday = 7)
  (h_tuesday : kids_tuesday = 13) : 
  kids_monday + kids_tuesday = 20 := 
by
  rw [h_monday, h_tuesday]
  exact rfl

end julia_played_tag_with_20_kids_l271_271110


namespace part_a_part_b_l271_271475

-- Definition based on conditions
def S (n k : ℕ) : ℕ :=
  -- Placeholder: Actual definition would count the coefficients
  -- of (x+1)^n that are not divisible by k.
  sorry

-- Part (a) proof statement
theorem part_a : S 2012 3 = 324 :=
by sorry

-- Part (b) proof statement
theorem part_b : 2012 ∣ S (2012^2011) 2011 :=
by sorry

end part_a_part_b_l271_271475


namespace edge_count_upper_bound_l271_271136

-- Definitions of graph, vertex count, edge count, and subgraph absence
variables {V : Type} [DecidableEq V] (G : SimpleGraph V)

-- Condition: G does not contain K3 as a subgraph
def not_contains_K3 (G : SimpleGraph V) : Prop :=
  ¬(∃ (u v w : V), u ≠ v ∧ v ≠ w ∧ u ≠ w ∧ G.Adj u v ∧ G.Adj v w ∧ G.Adj u w)

-- Statement of the theorem
theorem edge_count_upper_bound (n : ℕ) (hV : Fintype.card V = n) (hK3 : not_contains_K3 G) :
  Fintype.card (G.edgeSet) ≤ n * n / 4 :=
sorry

end edge_count_upper_bound_l271_271136


namespace odd_square_iff_odd_l271_271889

open Nat

theorem odd_square_iff_odd (n : ℕ) : odd (n ^ 2) → odd n :=
by
  sorry

end odd_square_iff_odd_l271_271889


namespace ad_gt_bc_l271_271731

theorem ad_gt_bc (a b c d : ℝ) (h1 : a + d = b + c) (h2 : |a - d| < |b - c|) : ad > bc := 
sorry

end ad_gt_bc_l271_271731


namespace range_of_k_l271_271066

-- Definitions for the line and curve
def line (k : ℝ) (x : ℝ) : ℝ := k * x - 1
def curve (x : ℝ) : ℝ := x - 1 + 1 / (Real.exp x)

-- Our goal is to prove the range of k for which the line and curve have no common points
theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, line k x ≠ curve x) ↔ (1 - Real.exp 1 < k ∧ k < 1) :=
sorry

end range_of_k_l271_271066


namespace sum_of_solutions_l271_271578

theorem sum_of_solutions : 
  let solutions := {x | 0 < x ∧ x ≤ 30 ∧ 17 * (5 * x - 3) % 10 = 34 % 10}
  in (∑ x in solutions, x) = 225 := by
  sorry

end sum_of_solutions_l271_271578


namespace sticks_forming_equilateral_triangle_l271_271694

theorem sticks_forming_equilateral_triangle (n : ℕ) (h1 : n ≥ 5) :
  (∃ l : list ℕ, (∀ x ∈ l, x ∈ finset.range (n + 1) ∧ x > 0) ∧ l.sum % 3 = 0) ↔ 
  (n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5) := 
sorry

end sticks_forming_equilateral_triangle_l271_271694


namespace remainder_polynomial_l271_271721

theorem remainder_polynomial (p r : ℚ[X]) (q : ℚ[X]) (p1 : p.eval 1 = 4) (p2 : p.eval 2 = 3) :
  (∃ q : ℚ[X], p = q * ((X - 1)^2 * (X - 2)) + r) ∧ degree r < 3 → 
  r = - (5 / 2) * X^2 + (1 / 2) * X + 6 :=
sorry

end remainder_polynomial_l271_271721


namespace find_sin_theta_l271_271120

variables (a b c d : EuclideanSpace ℝ (Fin 3))

-- Given conditions
axiom h_a_norm : ‖a‖ = 2
axiom h_b_norm : ‖b‖ = 6
axiom h_c_norm : ‖c‖ = 4
axiom h_d_norm : ‖d‖ = 3
axiom h_cross : a × (a × b) = c
axiom h_cos_phi : ∃ φ, cos φ = 1/2 ∧ angle b d = φ

-- Required to prove
theorem find_sin_theta : ∃ θ, sin θ = 1/3 ∧ angle a b = θ :=
sorry

end find_sin_theta_l271_271120


namespace arithmetic_sequence_and_sum_l271_271139

-- Arithmetic sequence definitions
def arithmetic_sequence (a d : ℕ → ℤ) := ∀ n : ℕ, a (n + 1) = a n + d n

-- Conditions
def Σ (a : ℕ → ℤ) (n : ℕ) := ∑ i in finset.range (n+1), a i
def condition_1 (a : ℕ → ℤ) (d : ℕ → ℤ) : Prop := Σ a 4 = 4 * Σ a 2
def condition_2 (a : ℕ → ℤ) (d : ℕ → ℤ) : Prop := 2 * a 0 + 1 = a 1

-- General formula for the sequence
def general_formula (a : ℕ → ℤ) : Prop := ∀ n : ℕ, a n = 2 * n - 1

-- Sequence b_n definition
def b_n (a : ℕ → ℤ) (n : ℕ) : ℚ := 1 / (a n * a (n + 1) : ℚ)

-- Sum of the first n terms of {b_n}
def T (b : ℕ → ℚ) (n : ℕ) := ∑ i in finset.range n, b i

-- The required sum formula
def sum_formula (T : (ℕ → ℚ) → ℕ → ℚ) (a : ℕ → ℤ) (n : ℕ) : Prop := T (b_n a) n = n / (2 * n + 1 : ℚ)

-- Statement
theorem arithmetic_sequence_and_sum :
  ∃ a d : ℕ → ℤ,
  condition_1 a d ∧ condition_2 a d ∧
  (general_formula a) ∧
  (∀ n : ℕ, sum_formula T a n) :=
by
  sorry

end arithmetic_sequence_and_sum_l271_271139


namespace trapezoid_upper_base_BC_l271_271766

theorem trapezoid_upper_base_BC (A B C D M : Point) (d : ℝ)
  (h1 : Trapezoid A B C D)
  (h2 : OnLine M A B)
  (h3 : Perpendicular D M A B)
  (h4 : Distance M C = Distance C D)
  (h5 : Distance A D = d) : Distance B C = d / 2 := 
sorry

end trapezoid_upper_base_BC_l271_271766


namespace find_a_l271_271061

theorem find_a (a b c d : ℕ) (h1 : a + b = d) (h2 : b + c = 6) (h3 : c + d = 7) : a = 1 :=
by
  sorry

end find_a_l271_271061


namespace shaded_area_l271_271519

/-- Given a scenario with two concentric circles where the chord AB is 100 units and tangent to the smaller circle,
    and given radii of the circles are 40 and 60 units respectively, prove that the area of the 
    shaded region between the circles is 2000π square units. -/
theorem shaded_area (r1 r2 : ℝ) (AB : ℝ) (h1 : AB = 100) (h2 : r1 = 40) (h3 : r2 = 60) : 
  ∃ (A : ℝ), A = π * (r2^2 - r1^2) ∧ A = 2000 * π := 
by 
  let R := r2
  let r := r1
  have hR : R = 60 := h3
  have hr : r = 40 := h2
  have hab : AB = 100 := h1
  have area : π * (R^2 - r^2) = 2000 * π :=
    by 
      calc
      π * (R^2 - r^2) 
          = π * (60^2 - 40^2) : by {rw [hR, hr]}
      ... = π * (3600 - 1600) : by norm_num
      ... = π * 2000 : by norm_num
      ... = 2000 * π : by ring
  existsi (2000 * π)
  exact ⟨π * (R^2 - r^2), area⟩

end shaded_area_l271_271519


namespace area_of_shaded_region_l271_271087

theorem area_of_shaded_region :
  ∀ (r : ℝ) (A B C D E F : Topology.Point),
  (∀ p : Topology.Point, Topology.is_midpoint p D E → Topology.is_semicircle p D E) →
  (∀ p : Topology.Point, Topology.is_midpoint p E F → Topology.is_semicircle p E F) →
  Topology.is_semicircle D F E → 
  r = 2 →
  Topology.distance A C = 4 * r →
  Topology.distance B F = 4 * r →
  4 * r * 4 * r = 16 :=
by
sorry

end area_of_shaded_region_l271_271087


namespace cos_plus_sin_l271_271777

open Real

theorem cos_plus_sin (α : ℝ) (h₁ : tan α = -2) (h₂ : (π / 2) < α ∧ α < π) : 
  cos α + sin α = (sqrt 5) / 5 :=
sorry

end cos_plus_sin_l271_271777


namespace units_digit_of_8_pow_120_l271_271244

theorem units_digit_of_8_pow_120 : (8 ^ 120) % 10 = 6 := 
by
  sorry

end units_digit_of_8_pow_120_l271_271244


namespace sum_geq_n_l271_271888

theorem sum_geq_n (f : ℕ+ → ℕ+) (hf : Function.Injective f) (n : ℕ) (hn : n > 0) :
  (∑ k in Finset.range n, f (k + 1) / (k + 1) : ℝ) ≥ n :=
  sorry

end sum_geq_n_l271_271888


namespace tangent_equality_intersects_inequality_outside_inequality_l271_271274

-- Definitions for the problem context
def is_tangent (l : Line) (O : Circle) : Prop := 
  ∃ M, l ⊥ O.center_line ∧ l ∩ O.center_line = {M} ∧ l meets_circle_tangentially O

def intersects (l : Line) (O : Circle) : Prop := 
  ∃ P Q, l ⊥ O.center_line ∧ l ∩ O.center_line = {M} ∧ l intersects_circle O

def is_outside (l : Line) (O : Circle) : Prop := 
  ∃ M, l ⊥ O.center_line ∧ l ∩ O.center_line = {M} ∧ ¬l intersects_circle O

-- Tangent lengths
def length_AP (A M : Point) (r : ℝ) (P : Point) : ℝ := 
  let a := distance A M in sqrt (a ^ 2 + (distance M O.center) ^ 2 - r ^ 2)

def length_BQ (B M : Point) (r : ℝ) (Q : Point) : ℝ := 
  let b := distance B M in sqrt (b ^ 2 + (distance M O.center) ^ 2 - r ^ 2)

def length_CR (C M : Point) (r : ℝ) (R : Point) : ℝ := 
  let c := distance C M in sqrt (c ^ 2 + (distance M O.center) ^ 2 - r ^ 2)

-- Proving the necessary conditions
theorem tangent_equality
(h1 : is_tangent l O)
(h2 : distance A M > distance B M ∧ distance B M > distance C M)
(h3 : ∀ A B C : Point, ∃ AP BQ CR : ℝ, AP = length_AP A M r P ∧ 
BQ = length_BQ B M r Q ∧ CR = length_CR C M r R)
: AB * CR + BC * AP = AC * BQ :=
sorry

theorem intersects_inequality
(h1 : intersects l O)
(h2 : distance A M > distance B M ∧ distance B M > distance C M)
(h3 : ∀ A B C : Point, ∃ AP BQ CR : ℝ, AP = length_AP A M r P ∧ 
BQ = length_BQ B M r Q ∧ CR = length_CR C M r R)
: AB * CR + BC * AP < AC * BQ :=
sorry

theorem outside_inequality
(h1 : is_outside l O)
(h2 : distance A M > distance B M ∧ distance B M > distance C M)
(h3 : ∀ A B C : Point, ∃ AP BQ CR : ℝ, AP = length_AP A M r P ∧ 
BQ = length_BQ B M r Q ∧ CR = length_CR C M r R)
: AB * CR + BC * AP > AC * BQ :=
sorry

end tangent_equality_intersects_inequality_outside_inequality_l271_271274


namespace coefficient_x_degree_one_l271_271427

noncomputable def calculate_a (a : ℤ) : Prop :=
  let expression_sum := (1 + 1)^3 * (2*1 - 1 + a)^5 in
  expression_sum = 256

theorem coefficient_x_degree_one (a : ℤ) (h1 : calculate_a a) : 
  (let coeff_x := 3 * (binom 5 1 * 2 + binom 5 2 * 2 * (-1) + binom 5 3 * 2 * 1 + binom 5 4 * 2 * (-1) + binom 5 5 * 2) in
  coeff_x = 0) :=
sorry

end coefficient_x_degree_one_l271_271427


namespace max_partitions_l271_271873

theorem max_partitions (p : ℕ) (n : ℕ) (S : Finset (Fin (p^n)))
    (h_prime : Nat.Prime p)
    (h_card : S.card = p^n)
    (P : Finset (Finset (Finset (Fin (p^n)))))
    (h_part : ∀ {part}, part ∈ P → ∀ s ∈ part, s.nonempty ∧ s.card % p = 0)
    (h_inter : ∀ {part1 part2}, part1 ∈ P → part2 ∈ P → ∀ s1 ∈ part1, ∀ s2 ∈ part2, s1 ≠ s2 → Finset.card (s1 ∩ s2) ≤ 1) :
    P.card ≤ (p^n - 1) / (p - 1) := sorry

end max_partitions_l271_271873


namespace incorrect_expression_l271_271063

theorem incorrect_expression (x y : ℝ) (h : x > y) : ¬ (1 - 3*x > 1 - 3*y) :=
sorry

end incorrect_expression_l271_271063


namespace find_a_b_solve_ineq2_l271_271041

noncomputable theory

-- Definitions based on the problem conditions
def solution_set_ineq1 (a : ℝ) := { x : ℝ | -2/5 < x ∧ x < 1 }

theorem find_a_b (a b : ℝ) (h : solution_set_ineq1 a = { x : ℝ | b < x ∧ x < 1 }) : 
  a = -5 ∧ b = -2/5 :=
sorry -- Proof goes here

def solution_set_ineq2 (a : ℝ) : set ℝ :=
if a = 3 then { x : ℝ | x ≠ 1 }
else if a = 0 then { x : ℝ | x < 1 }
else if a > 3 then { x : ℝ | x < 3/a ∨ x > 1 }
else if 0 < a ∧ a < 3 then { x : ℝ | x < 1 ∨ x > 3/a }
else { x : ℝ | 3/a < x ∧ x < 1 }

theorem solve_ineq2 (a : ℝ) : solution_set_ineq2 a = 
  if a = 3 then { x : ℝ | x ≠ 1 }
  else if a = 0 then { x : ℝ | x < 1 }
  else if a > 3 then { x : ℝ | x < 3/a ∨ x > 1 }
  else if 0 < a ∧ a < 3 then { x : ℝ | x < 1 ∨ x > 3/a }
  else { x : ℝ | 3/a < x ∧ x < 1 } :=
sorry -- Proof goes here

end find_a_b_solve_ineq2_l271_271041


namespace probability_diana_greater_apollo_sum_at_least_10_l271_271676

theorem probability_diana_greater_apollo_sum_at_least_10 :
  (∑ d in Finset.range 9, ∑ a in Finset.range d, if d + a >= 10 then 1 else 0) / 64 = (3/8 : ℚ) :=
by
  sorry

end probability_diana_greater_apollo_sum_at_least_10_l271_271676


namespace triangle_side_lengths_l271_271052

theorem triangle_side_lengths {x : ℤ} (h₁ : x + 4 > 10) (h₂ : x + 10 > 4) (h₃ : 10 + 4 > x) :
  ∃ (n : ℕ), n = 7 :=
by
  sorry

end triangle_side_lengths_l271_271052


namespace part1_solution_set_part2_range_of_a_l271_271023

def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1_solution_set (x : ℝ) : 
  (f x 2 ≥ 4) ↔ (x ≤ 3 / 2 ∨ x ≥ 11 / 2) :=
sorry

theorem part2_range_of_a (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 4) → (a ≤ -1 ∨ a ≥ 3) :=
sorry

end part1_solution_set_part2_range_of_a_l271_271023


namespace estimate_number_of_trees_l271_271857

-- Definitions derived from the conditions
def forest_length : ℝ := 100
def forest_width : ℝ := 0.5
def plot_length : ℝ := 1
def plot_width : ℝ := 0.5
def tree_counts : List ℕ := [65110, 63200, 64600, 64700, 67300, 63300, 65100, 66600, 62800, 65500]

-- The main theorem stating the problem
theorem estimate_number_of_trees :
  let avg_trees_per_plot := tree_counts.sum / tree_counts.length
  let total_plots := (forest_length * forest_width) / (plot_length * plot_width)
  avg_trees_per_plot * total_plots = 6482100 :=
by
  sorry

end estimate_number_of_trees_l271_271857


namespace find_upper_base_length_l271_271758

-- Define the trapezoid and its properties.
variables (d : ℝ)
variables (A D : ℝ × ℝ) (M : ℝ × ℝ) (C : ℝ × ℝ)

-- Define the conditions of the problem.
def is_trapezoid (A B C D : ℝ × ℝ) : Prop := 
  ∃ M, M.1 = (A.1 + B.1) / 2 ∧ M.1 = (C.1 + D.1) / 2
  
def perpendicular (P Q : ℝ × ℝ) : Prop :=
  P.1 = Q.1

def equal_distance (P Q R : ℝ × ℝ) : Prop :=
  dist P Q = dist Q R

-- Setting the exact locations of points
def coordinates : Prop := 
  D = (0, 0) ∧ A = (d, 0) ∧ perpendicular (M) (D, A)

-- Required proof
theorem find_upper_base_length :
  coordinates d A D M ∧ equal_distance M C D → 
  dist (A, C) = d / 2 :=
by sorry

end find_upper_base_length_l271_271758


namespace max_count_of_selected_numbers_l271_271170

theorem max_count_of_selected_numbers :
  ∃ (S : Finset ℕ), S ⊆ (Finset.range 21 \ {0}) ∧
                    (∀ a b ∈ S, lcm a b ∈ S) ∧
                    S.card = 6 :=
sorry

end max_count_of_selected_numbers_l271_271170


namespace percentage_of_money_spent_l271_271997

theorem percentage_of_money_spent (initial_amount remaining_amount : ℝ) (h_initial : initial_amount = 500) (h_remaining : remaining_amount = 350) :
  (((initial_amount - remaining_amount) / initial_amount) * 100) = 30 :=
by
  -- Start the proof
  sorry

end percentage_of_money_spent_l271_271997


namespace initial_people_in_castle_l271_271989

theorem initial_people_in_castle (P : ℕ) (provisions : ℕ → ℕ → ℕ) :
  (provisions P 90) - (provisions P 30) = provisions (P - 100) 90 ↔ P = 300 :=
by
  sorry

end initial_people_in_castle_l271_271989


namespace circumcenter_dot_product_l271_271113

variables {A B C O : Type} [inner_product_space ℝ A] -- Define the points and the space

-- Define the sides opposite to angles A, B, C
variables (a b c : ℝ)

-- Circumcenter O and side lengths
variables (O : A) (h_b : b = 3) (h_c : c = 5) (h_O : is_circumcenter O A B C)

-- The main proof statement
theorem circumcenter_dot_product (h_b : b = 3) (h_c : c = 5) :
  let OA := vector_between O A,
      OB := vector_between O B,
      OC := vector_between O C,
      AB := vector_between A B,
      AC := vector_between A C,
      BC := vector_between B C
  in 
  OA • BC = 8 :=
  sorry

end circumcenter_dot_product_l271_271113


namespace sara_marble_count_l271_271169

variable (x : ℝ) (y : ℝ)

theorem sara_marble_count (hx : x = 792.0) (hy : y = 233.0) : x + y = 1025.0 :=
by {
  rw [hx, hy],
  exact rfl,
}

end sara_marble_count_l271_271169


namespace markov_equation_pairwise_coprime_l271_271496

theorem markov_equation_pairwise_coprime (m n p: ℤ) (h : m^2 + n^2 + p^2 = 3 * m * n * p) :
    Nat.coprime m n ∧ Nat.coprime n p ∧ Nat.coprime m p :=
sorry

end markov_equation_pairwise_coprime_l271_271496


namespace range_of_m_l271_271069

variable (a : ℝ) (x : ℝ) (m : ℝ)
noncomputable def f (x : ℝ) := log a (3 + 3^x + 4^x - m)

theorem range_of_m {a : ℝ} {f : ℝ → ℝ} :
  (∀ x : ℝ, ∃ y : ℝ, f y = x) → (m ≥ 3) := 
by {
  assume h : ∀ x : ℝ, ∃ y : ℝ, f y = x,
  sorry,
}

end range_of_m_l271_271069


namespace chessboard_not_l_shaped_decomposable_l271_271256

def is_l_shaped (cells : Finset (ℕ × ℕ)) : Prop :=
  cells.card = 4 ∧
  -- L-shape conditions: it can be mapped to {(0,0), (1,0), (2,0), (2,1)} or any rotation/reflection thereof
  ((cells = {(i, j), (i + 1, j), (i + 2, j), (i + 2, j + 1)} ∨ 
    cells = {(i, j), (i, j + 1), (i, j + 2), (i + 1, j + 2)} ∨
    cells = {(i, j), (i - 1, j), (i - 2, j), (i - 2, j - 1)} ∨
    cells = {(i, j), (i, j - 1), (i, j - 2), (i - 1, j - 2)} ∨
    cells = {(i, j), (i + 1, j), (i + 2, j), (i + 2, j - 1)} ∨
    cells = {(i, j), (i, j - 1), (i, j - 2), (i + 1, j - 2)} ∨
    cells = {(i, j), (i - 1, j), (i - 2, j), (i - 2, j + 1)} ∨
    cells = {(i, j), (i, j + 1), (i, j + 2), (i - 1, j + 2)}))
  )

def board_l_shaped_decomposable (remaining_cells : Finset (ℕ × ℕ)) : Prop :=
  ∃ (partition : Finset (Finset (ℕ × ℕ))), 
    (∀ part ∈ partition, is_l_shaped part) ∧
    (remaining_cells = partition.bUnion id)

noncomputable 
def central2x2 : Finset (ℕ × ℕ) := {(3,3), (3,4), (4,3), (4,4)}

def remaining_cells : Finset (ℕ × ℕ) := (Finset.range 8).product (Finset.range 8) \ central2x2

theorem chessboard_not_l_shaped_decomposable : ¬ board_l_shaped_decomposable remaining_cells :=
sorry  

end chessboard_not_l_shaped_decomposable_l271_271256


namespace count_leap_years_l271_271652

def is_leap_year (y : ℕ) : Prop :=
  (y % 4 = 0 ∧ y % 100 ≠ 0) ∨ (y % 400 = 0)

theorem count_leap_years {ys : list ℕ} (h1 : ys = [1964, 1978, 1995, 1996, 2001, 2100]) :
  (ys.filter is_leap_year).length = 2 :=
by
  sorry

end count_leap_years_l271_271652


namespace general_formula_a_n_sum_first_n_terms_b_n_l271_271854

-- Given problem conditions
variable {d : ℤ}
variable (a : ℕ → ℤ)
variable (b : ℕ → ℤ)

axiom arith_seq : ∀ n : ℕ, a (n + 1) - a n = d
axiom a4_eq_10 : a 4 = 10
axiom geom_seq : (a 3 < a 6) -> (a 6 < a 10)

-- Problem Ⅰ: General formula for a_n
theorem general_formula_a_n :
  (∀ n : ℕ, a n = n + 6) :=
sorry

-- Problem Ⅱ: Formula for the sum of the first n terms of {b_n}
theorem sum_first_n_terms_b_n :
  (∀ n : ℕ, b n = 2 ^ a n → (b 1 = 128 ∧ ∀ n : ℕ, b (n + 1) / b n = 2)) →
  (∀ n : ℕ, ∑ i in range n, b i = 2 ^ (n + 7) - 128) :=
sorry

end general_formula_a_n_sum_first_n_terms_b_n_l271_271854


namespace find_value_of_2_a_sub_b_l271_271131

theorem find_value_of_2_a_sub_b (a b : ℝ) (h1 : a > b) (h2 : 2^a + 2^b = 75) (h3 : 2^(-a) + 2^(-b) = 1 / 12) : 2^(a - b) = 4 := 
  sorry

end find_value_of_2_a_sub_b_l271_271131


namespace lines_parallel_lines_coincident_lines_perpendicular_l271_271399

-- Definitions of the lines l1 and l2 with variable m
def line1 (m : ℝ) : ℝ × ℝ → Prop := λ p, (m + 3) * p.1 + 4 * p.2 = 5 - 3 * m
def line2 (m : ℝ) : ℝ × ℝ → Prop := λ p, 2 * p.1 + (m + 5) * p.2 = 8

-- Parallel lines proof
theorem lines_parallel {m : ℝ} : 
  (∀ p q : ℝ × ℝ, line1 m p ∧ line2 m p ∧ line1 m q ∧ line2 m q → (m + 3) * (m + 5) = 8) ↔ m = -7 :=
by sorry

-- Coincident lines proof
theorem lines_coincident {m : ℝ} : 
  (∀ p : ℝ × ℝ, line1 m p → line2 m p) ↔ m = -1 :=
by sorry

-- Perpendicular lines proof
theorem lines_perpendicular {m : ℝ} : 
  (∀ p q : ℝ × ℝ, line1 m p ∧ line2 m q → 2 * (m + 3) + 4 * (m + 5) = 0) ↔ m = - 13 / 3 :=
by sorry

end lines_parallel_lines_coincident_lines_perpendicular_l271_271399


namespace largest_number_of_minerals_per_shelf_l271_271486

theorem largest_number_of_minerals_per_shelf (d : ℕ) :
  d ∣ 924 ∧ d ∣ 1386 ∧ d ∣ 462 ↔ d = 462 :=
by
  sorry

end largest_number_of_minerals_per_shelf_l271_271486


namespace truth_tellers_count_l271_271547

variables (P1 P2 P3 P4 : Prop)

-- Conditions
variables (C1 : P1 ↔ (∃ n, n ∈ {1, 3}))
variables (C2 : P2 ↔ (∃ n, n ∈ {0, 2, 4}))
variables (C3 : P3 ↔ (∃ n, n ∈ {2, 3}))
variables (C4 : P4 ↔ (∃ n, n ∈ {1, 4}))

theorem truth_tellers_count (hP1 : P1 → 1 ∈ {1, 3})
                           (hP2 : P2 → 0 ∈ {0, 2, 4})
                           (hP3 : P3 → 2 ∈ {2, 3})
                           (hP4 : P4 → 1 ∈ {1, 4})
                           (hNotBothOddAndEven : ¬(P1 ∧ P2))
                           (hNotBothPrimeAndSquare : ¬(P3 ∧ P4)) :
  ∃ n, n = 2 ∧ (n ∈ {1, 2, 3, 4}) :=
sorry

end truth_tellers_count_l271_271547


namespace inequality_with_means_l271_271725

theorem inequality_with_means 
  (n : ℕ) (a : Fin n → ℝ) 
  (h_sorted : ∀ i j : Fin n, i ≤ j → a i ≤ a j) 
  (M1 := (1 / n) * (∑ i, a i)) 
  (M2 := (2 / (n * (n - 1))) * ∑ i j, if i < j then a i * a j else 0) 
  (Q := Real.sqrt (M1 ^ 2 - M2)) :
  ((a 0) ≤ M1 - Q) ∧ ((M1 - Q) ≤ (M1 + Q)) ∧ ((M1 + Q) ≤ (a (n - 1))) ∧ 
  ((a 0) = a 1 ∧ (a 1) = a 2 ∧ ... ∧ (a (n - 1)) = a (n - 1)) ↔ (∀ i j : Fin n, a i = a j) :=
by 
  sorry

end inequality_with_means_l271_271725


namespace ashley_percentage_secured_l271_271440

noncomputable def marks_secured : ℕ := 332
noncomputable def max_marks : ℕ := 400
noncomputable def percentage_secured : ℕ := (marks_secured * 100) / max_marks

theorem ashley_percentage_secured 
    (h₁ : marks_secured = 332)
    (h₂ : max_marks = 400) :
    percentage_secured = 83 := by
  -- Proof goes here
  sorry

end ashley_percentage_secured_l271_271440


namespace solve_problem_l271_271879

def nested_sqrt (b : ℝ) : ℝ := real.sqrt (b + real.sqrt (b + real.sqrt (b + ...)))

def odot (a b : ℝ) : ℝ := a^2 + nested_sqrt b

def problem (g : ℝ) : Prop :=
  odot 4 g = 20 → g = 12

theorem solve_problem (g : ℝ) : problem g :=
  sorry

end solve_problem_l271_271879


namespace complement_of_M_with_respect_to_U_l271_271727

noncomputable def U : Set ℕ := {1, 2, 3, 4}
noncomputable def M : Set ℕ := {1, 2, 3}

theorem complement_of_M_with_respect_to_U :
  (U \ M) = {4} :=
by
  sorry

end complement_of_M_with_respect_to_U_l271_271727


namespace seats_needed_on_bus_l271_271203

variable (f t tr dr c h : ℕ)

def flute_players := 5
def trumpet_players := 3 * flute_players
def trombone_players := trumpet_players - 8
def drummers := trombone_players + 11
def clarinet_players := 2 * flute_players
def french_horn_players := trombone_players + 3

theorem seats_needed_on_bus :
  f = 5 →
  t = 3 * f →
  tr = t - 8 →
  dr = tr + 11 →
  c = 2 * f →
  h = tr + 3 →
  f + t + tr + dr + c + h = 65 :=
by
  sorry

end seats_needed_on_bus_l271_271203


namespace problem1_proof_problem2_proof_l271_271662

-- Define the calculations and their results for Problem 1
def problem1_expr : ℝ := (-2)^2 - (3 - 5) - real.sqrt 4 + 2 * 3
def problem1_ans : ℝ := 10

-- Prove Problem 1
theorem problem1_proof : problem1_expr = problem1_ans := by
  sorry

-- Define the calculations and their results for Problem 2
def problem2_expr : ℝ := real.cbrt 64 - real.sqrt 9 + abs (real.sqrt 2 - 1) + (-1)^3
def problem2_ans : ℝ := real.sqrt 2 - 1

-- Prove Problem 2
theorem problem2_proof : problem2_expr = problem2_ans := by
  sorry

end problem1_proof_problem2_proof_l271_271662


namespace calculate_b_l271_271196

-- Given the coordinates and conditions
def pointP (b : ℝ) : ℝ × ℝ := (0, b)
def pointS (b : ℝ) : ℝ × ℝ := (5, b - 5)
def pointQ (b : ℝ) : ℝ × ℝ := (b, 0)
def pointR : ℝ × ℝ := (5, 0)

-- The base and height of triangles
def baseQOP (b : ℝ) : ℝ := b
def heightQOP (b : ℝ) : ℝ := b

def baseQRS (b : ℝ) : ℝ := 5 - b
def heightQRS (b : ℝ) : ℝ := b - 5

-- The ratio of the areas of triangles QRS and QOP
def area_ratio (b : ℝ) := (baseQRS b * heightQRS b) / (baseQOP b * heightQOP b)

-- The condition of the problem
axiom ratio_condition : ∀ b : ℝ, 1 < b ∧ b < 5 → area_ratio b = (4 / 9)

-- The proof problem
theorem calculate_b : ∃ b : ℝ, 1 < b ∧ b < 5 ∧ b = 3.6 :=
by {
  sorry
}

end calculate_b_l271_271196


namespace part1_problem_l271_271926

theorem part1_problem
  (A B C : Real.Angle)
  (a b c : ℝ)
  (cosA : Real.cos A)
  (sinA : Real.sin A)
  (sin2B : Real.sin (2 * B))
  (cos2B : Real.cos (2 * B))
  (hC : C = 2 * π / 3)
  (h : cosA / (1 + sinA) = sin2B / (1 + cos2B))
  : B = π / 6 := by
  sorry

end part1_problem_l271_271926


namespace moles_of_KCl_formed_l271_271341

-- Define 1 mole of HCl and 1 mole of KOH
def moles_HCl := 1
def moles_KOH := 1

-- Define the reaction product
def reaction_product (HCl KOH : Nat) : Nat :=
  if (HCl = 1) ∧ (KOH = 1) then 1 else 0

-- Prove the number of moles of KCl formed
theorem moles_of_KCl_formed : reaction_product moles_HCl moles_KOH = 1 :=
by
  -- Use the condition definition directly here
  exact rfl

-- Add proof here, currently as placeholder
sorry

end moles_of_KCl_formed_l271_271341


namespace sum_of_solutions_eq_225_l271_271572

theorem sum_of_solutions_eq_225 :
  ∑ x in Finset.filter (λ x, 0 < x ∧ x ≤ 30 ∧ (17 * (5 * x - 3)) % 10 = 4) (Finset.range 31), x = 225 :=
by sorry

end sum_of_solutions_eq_225_l271_271572


namespace sticks_forming_equilateral_triangle_l271_271696

theorem sticks_forming_equilateral_triangle (n : ℕ) (h1 : n ≥ 5) :
  (∃ l : list ℕ, (∀ x ∈ l, x ∈ finset.range (n + 1) ∧ x > 0) ∧ l.sum % 3 = 0) ↔ 
  (n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5) := 
sorry

end sticks_forming_equilateral_triangle_l271_271696


namespace transylvanian_human_truth_transylvanian_vampire_lie_l271_271510

-- Definitions of predicates for human and vampire behavior
def is_human (A : Type) : Prop := ∀ (X : Prop), (A → X) → X
def is_vampire (A : Type) : Prop := ∀ (X : Prop), (A → X) → ¬X

-- Lean definitions for the problem
theorem transylvanian_human_truth (A : Type) (X : Prop) (h_human : is_human A) (h_says_true : A → X) :
  X :=
by sorry

theorem transylvanian_vampire_lie (A : Type) (X : Prop) (h_vampire : is_vampire A) (h_says_true : A → X) :
  ¬X :=
by sorry

end transylvanian_human_truth_transylvanian_vampire_lie_l271_271510


namespace part1_part2_l271_271903

-- Define the condition for the first part
def condition (A B C : ℝ) : Prop :=
  (C = 2 * Real.pi / 3) ∧ (1 + Real.sin A ≠ 0) ∧ (2 * Real.cos B ≠ 0) ∧ 
  (Real.cos A / (1 + Real.sin A) = Real.sin (2 * B) / (1 + Real.cos (2 * B)))

-- Theorem for the first part: If \( C = \frac{2\pi}{3} \), then \( B = \frac{\pi}{6} \)
theorem part1 (A B C : ℝ) (h : condition A B C) : B = Real.pi / 6 :=
  sorry

-- Define the condition for the second part as the side ratios expression
def ratio_expression (a b c : ℝ) : ℝ :=
  (a^2 + b^2) / (c^2)

-- Theorem for the second part: Minimum value of \(\frac{a^2 + b^2}{c^2}\)
theorem part2 (a b c : ℝ) (A B C : ℝ) 
  (h : condition A B C) : ratio_expression a b c = 4 * Real.sqrt 2 - 5 :=
  sorry

end part1_part2_l271_271903


namespace polynomial_has_root_l271_271332

noncomputable def a : ℝ := real.root 5 (2 + real.sqrt 3)
noncomputable def b : ℝ := real.root 5 (2 - real.sqrt 3)

theorem polynomial_has_root :
  a * b = 1 →
  (b^5 - 5 * b^3 + 5 * b - 4 = 0) :=
by 
  sorry

end polynomial_has_root_l271_271332


namespace general_term_a_l271_271541

noncomputable def S (n : ℕ) : ℤ := 3^n - 2

noncomputable def a (n : ℕ) : ℤ :=
  if n = 1 then 1 else 2 * 3^(n - 1)

theorem general_term_a (n : ℕ) (hn : n > 0) : a n = if n = 1 then 1 else 2 * 3^(n - 1) := by
  -- Proof goes here
  sorry

end general_term_a_l271_271541


namespace find_abs_lower_bound_l271_271678

noncomputable theory

-- Define the polynomial and conditions
def poly (n : ℕ) (a : ℕ → ℝ) :=
  monic (λ x, x^n + a (n-1) * x^(n-1) + a (n-2) * x^(n-2))

theorem find_abs_lower_bound (n : ℕ) (a : ℕ → ℝ) (h_monic : poly n a)
  (h_eq : a (n-1) = -a (n-2)) : 
  ∃ lb, lb = 1 ∧ ∀ a_n2 : ℝ, ((a_n2 ^ 2 - 2 * a_n2) ≥ a_n2 ^ 2 - 2 * a_n2) 
    ∧ |lb| = 1 :=
sorry

end find_abs_lower_bound_l271_271678


namespace equivalent_math_problem_l271_271349

noncomputable def fractional_part (x : ℝ) : ℝ := x - x.floor

noncomputable def binom (n : ℕ) (k : ℕ) : ℕ :=
  if h : k ≤ n then nat.choose n k else 0

noncomputable def f (n : ℕ) (x : ℝ) : ℝ :=
  (1 - fractional_part x) * (binom n (x.floor.to_nat)) +
  fractional_part x * (binom n (x.floor.to_nat + 1))

theorem equivalent_math_problem :
  ∀ (m n : ℕ), 2 ≤ m → 2 ≤ n →
    (∑ i in finset.range (m * n), f m i / n) = 123 →
    (∑ i in finset.range (m * n), f n i / m) = 74 :=
begin
  intros m n hm2 hn2 hsum,
  sorry,
end

end equivalent_math_problem_l271_271349


namespace cousin_room_distribution_cousin_room_distribution_l271_271938

-- Define the problem conditions as constants
constant numberOfCousins : ℕ := 5
constant numberOfRooms : ℕ := 5

-- State the theorem equivalent to the original math problem
theorem cousin_room_distribution : 
  ∃ (ways : ℕ), ways = 67 ∧ 
  (ways = (∑ i in (finset.powerset (finset.range numberOfCousins)).filter (λ s, s.card ≤ numberOfRooms), 
    card partitions_of_phone_distinct(number_of_cousins:5,set: nat)):= 67)
proof
construct the proof with combination sum index and total partitions of all distinguish 
number_of_cousins 5 in 5 identifical rooms 
  mention lean function here 
qed


# Translate the mathematical problem into Lean statement
theorem cousin_room_distribution :
  (∑ s in (finset.powerset (finset.range numberOfCousins)) ∩ {s | s.card ≤ numberOfRooms}, 
     1) = 67 := sorry

end cousin_room_distribution_cousin_room_distribution_l271_271938


namespace part1_part2_l271_271902

-- Define the condition for the first part
def condition (A B C : ℝ) : Prop :=
  (C = 2 * Real.pi / 3) ∧ (1 + Real.sin A ≠ 0) ∧ (2 * Real.cos B ≠ 0) ∧ 
  (Real.cos A / (1 + Real.sin A) = Real.sin (2 * B) / (1 + Real.cos (2 * B)))

-- Theorem for the first part: If \( C = \frac{2\pi}{3} \), then \( B = \frac{\pi}{6} \)
theorem part1 (A B C : ℝ) (h : condition A B C) : B = Real.pi / 6 :=
  sorry

-- Define the condition for the second part as the side ratios expression
def ratio_expression (a b c : ℝ) : ℝ :=
  (a^2 + b^2) / (c^2)

-- Theorem for the second part: Minimum value of \(\frac{a^2 + b^2}{c^2}\)
theorem part2 (a b c : ℝ) (A B C : ℝ) 
  (h : condition A B C) : ratio_expression a b c = 4 * Real.sqrt 2 - 5 :=
  sorry

end part1_part2_l271_271902


namespace quadrilateral_covered_by_circles_l271_271534

theorem quadrilateral_covered_by_circles
  {A B C D : Type}
  (distance : A → A → ℝ)
  (convex_quadrilateral : A → A → A → A → Prop)
  (side_length_at_most_7 : ∀ (X Y : A), distance X Y ≤ 7)
  (r : ℝ) : 
  r = 5 →
  convex_quadrilateral A B C D → 
  (∀ O : A, 
    (distance O A ≤ r ∨ distance O B ≤ r ∨ distance O C ≤ r ∨ distance O D ≤ r)) :=
by
  intros hr hcvq O
  sorry

end quadrilateral_covered_by_circles_l271_271534


namespace locus_of_projections_is_circle_l271_271619

-- Definitions based on conditions
variables (V : Type*) [inner_product_space ℝ V]
variables (l : submodule ℝ V) (P Q : V)
hypothesis (h_line_nonzero : l ≠ ⊥)
hypothesis (h_P_notin_l : P ∉ l)

-- The mathematically equivalent proof problem
theorem locus_of_projections_is_circle {V : Type*} [inner_product_space ℝ V]
  (l : submodule ℝ V) (P Q : V)
  (h_line_nonzero : l ≠ ⊥) (h_P_notin_l : P ∉ l) :
  ∃ (α : affine_subspace ℝ V), 
  (α.direction = l ⊥) ∧
  (Q ∈ α) ∧ 
  (l = affine_span ℝ {Q}) ∧
  (∃ (circle : set V), (is_circle_with_diameter P Q α.direction circle)) :=
sorry

end locus_of_projections_is_circle_l271_271619


namespace sum_of_solutions_eq_225_l271_271573

theorem sum_of_solutions_eq_225 :
  ∑ x in Finset.filter (λ x, 0 < x ∧ x ≤ 30 ∧ (17 * (5 * x - 3)) % 10 = 4) (Finset.range 31), x = 225 :=
by sorry

end sum_of_solutions_eq_225_l271_271573


namespace not_equivalent_l271_271247

noncomputable def A := 3.25 * 10^(-6)
noncomputable def B := 3.25
noncomputable def C := 325 * 10^(-8)
noncomputable def D := (3.25 / 10) * 10^(-5)
noncomputable def E := 1 / 308000000

theorem not_equivalent : E ≠ A := sorry

end not_equivalent_l271_271247


namespace min_value_2x_plus_4y_l271_271374

theorem min_value_2x_plus_4y
  (x y : ℝ)
  (h : x + 2 * y = 3) :
  2 ^ x + 4 ^ y ≥ 4 * real.sqrt 2 :=
sorry

end min_value_2x_plus_4y_l271_271374


namespace part1_solution_set_part2_values_of_a_l271_271030

-- Parameters and definitions for the function and conditions
def f (x a : ℝ) := |x - a^2| + |x - (2*a + 1)|

-- Part (1) When a = 2, the inequality's solution set
theorem part1_solution_set (x : ℝ) : f x 2 ≥ 4 ↔ (x ≤ 3/2 ∨ x ≥ 11/2) := 
sorry

-- Part (2) Range of values for a given f(x, a) ≥ 4
theorem part2_values_of_a (a : ℝ) : 
∀ x, f x a ≥ 4 → (a ≤ -1 ∨ a ≥ 3) :=
sorry

end part1_solution_set_part2_values_of_a_l271_271030


namespace debelyn_gave_dolls_to_andrena_l271_271670

theorem debelyn_gave_dolls_to_andrena :
  ∀ (D : ℕ),
  ∃ (DebelynInitial ChristelInitial AndrenaChristelDiff AndrenaDebelynDiff ChristelToAndrena remainingAndrena remainingDebelyn remainingChristel : ℕ),
  DebelynInitial = 20 ∧
  ChristelInitial = 24 ∧
  ChristelToAndrena = 5 ∧
  AndrenaChristelDiff = 2 ∧
  AndrenaDebelynDiff = 3 ∧
  remainingChristel = ChristelInitial - ChristelToAndrena ∧
  remainingAndrena = remainingChristel + AndrenaChristelDiff ∧
  remainingDebelyn = DebelynInitial - D ∧
  remainingAndrena = remainingDebelyn + AndrenaDebelynDiff ∧
  D = 2 :=
by
  intros D
  use 20, 24, 2, 3, 5, 21, 18, 19
  -- Adding conditions and values
  tauto

end debelyn_gave_dolls_to_andrena_l271_271670


namespace varphi_not_k_pi_l271_271528

theorem varphi_not_k_pi (k : ℤ) :
  ¬ (∃ (k : ℤ), ∀ x : ℝ, cos (3 * x + k * π) = cos (3 * (-x) + k * π)) := 
sorry

end varphi_not_k_pi_l271_271528


namespace area_of_triangle_proof_l271_271715

noncomputable def vec := ℕ → ℝ

def v1 : vec := λ i, [2, 1, 0][i]
def v2 : vec := λ i, [3, 3, 2][i]
def v3 : vec := λ i, [5, 8, 1][i]

def sub (a b: vec) : vec := λ i, a i - b i

def cross (u v: vec) : vec :=
  λ i, match i with
       | 0 => u 1 * v 2 - u 2 * v 1
       | 1 => u 2 * v 0 - u 0 * v 2
       | 2 => u 0 * v 1 - u 1 * v 0
       | _ => 0 

def magnitude (v: vec) : ℝ :=
  real.sqrt (v 0 ^ 2 + v 1 ^ 2 + v 2 ^ 2)

def area_of_triangle (a b c: vec) : ℝ :=
  1/2 * magnitude (cross (sub b a) (sub c a))

theorem area_of_triangle_proof :
  area_of_triangle v1 v2 v3 = real.sqrt 170 / 2 := 
sorry

end area_of_triangle_proof_l271_271715


namespace study_group_books_l271_271264

theorem study_group_books (x n : ℕ) (h1 : n = 5 * x - 2) (h2 : n = 4 * x + 3) : x = 5 ∧ n = 23 := by
  sorry

end study_group_books_l271_271264


namespace problem_statement_l271_271477

theorem problem_statement (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) :
  (a + 1 / a) ^ 2 + (b + 1 / b) ^ 2 ≥ 25 / 2 := 
by
  sorry

end problem_statement_l271_271477


namespace greatest_integer_not_exceed_1000x_l271_271645

noncomputable def cube_shadow_problem (x : ℝ) : ℝ :=
  if h : (2 * x - 1) = 23 then 1000 * x 
  else 0

theorem greatest_integer_not_exceed_1000x :
  ∃ x : ℝ, 2 * x - 1 = 23 ∧ int.floor (cube_shadow_problem x) = 12000 :=
by
  sorry

end greatest_integer_not_exceed_1000x_l271_271645


namespace range_of_a_l271_271774

variables (a x : ℝ)
def p : Prop := (x - 3) * (x + 1) < 0
def q : Prop := (x - 2) / (x - 4) < 0
def r : Prop := a < x ∧ x < 2 * a
def Sufficient (p q r : Prop) : Prop := (p ∧ q) → r

theorem range_of_a (h : a > 0) (hs : Sufficient p q r) : 3/2 ≤ a ∧ a ≤ 2 := by
  sorry

end range_of_a_l271_271774


namespace power_function_value_l271_271425

theorem power_function_value (α : ℝ) (f : ℝ → ℝ) (h₁ : f x = x ^ α) (h₂ : f (1 / 2) = 4) : f 8 = 1 / 64 := by
  sorry

end power_function_value_l271_271425


namespace part1_part2_l271_271011

def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1 (x : ℝ) : 
  ∀ x, f x 2 ≥ 4 ↔ (x ≤ 3 / 2 ∨ x ≥ 11 / 2) :=
by sorry

theorem part2 (x a : ℝ) :
  f x a ≥ 4 → (a ≤ -1 ∨ a ≥ 3) :=
by sorry

end part1_part2_l271_271011


namespace find_upper_base_length_l271_271755

-- Define the trapezoid and its properties.
variables (d : ℝ)
variables (A D : ℝ × ℝ) (M : ℝ × ℝ) (C : ℝ × ℝ)

-- Define the conditions of the problem.
def is_trapezoid (A B C D : ℝ × ℝ) : Prop := 
  ∃ M, M.1 = (A.1 + B.1) / 2 ∧ M.1 = (C.1 + D.1) / 2
  
def perpendicular (P Q : ℝ × ℝ) : Prop :=
  P.1 = Q.1

def equal_distance (P Q R : ℝ × ℝ) : Prop :=
  dist P Q = dist Q R

-- Setting the exact locations of points
def coordinates : Prop := 
  D = (0, 0) ∧ A = (d, 0) ∧ perpendicular (M) (D, A)

-- Required proof
theorem find_upper_base_length :
  coordinates d A D M ∧ equal_distance M C D → 
  dist (A, C) = d / 2 :=
by sorry

end find_upper_base_length_l271_271755


namespace cos_B_and_area_of_triangle_l271_271870

theorem cos_B_and_area_of_triangle (A B C : ℝ) (a b c : ℝ)
  (h_sin_A : Real.sin A = Real.sin (2 * B))
  (h_a : a = 4) (h_b : b = 6) :
  Real.cos B = 1 / 3 ∧ ∃ (area : ℝ), area = 8 * Real.sqrt 2 :=
by
  sorry  -- Proof goes here

end cos_B_and_area_of_triangle_l271_271870


namespace part_a_part_b_l271_271525

-- Define the convex quadrilateral ABCD and calculate its extensions.
variables (A B C D O M N P Q : Point)
-- Definitions of the midpoints.
def midpoint (X Y : Point) : Point := ⟨(X.x + Y.x) / 2, (X.y + Y.y) / 2⟩

-- Let the points M, N, P, and Q be the midpoints of their respective segments.
def M := midpoint A B
def N := midpoint C D
def P := midpoint A C
def Q := midpoint B D

-- Areas of specific regions
noncomputable def area (points : List Point) : ℝ := sorry 

-- Conditions: Define the areas based on the points
def S_ABD := area [A, B, D]
def S_ACD := area [A, C, D]
def S_ABCD := S_ABD + S_ACD

-- Statements to prove
theorem part_a : area [P, M, Q, N] = |S_ABD - S_ACD| / 2 := sorry

theorem part_b : area [O, P, Q] = S_ABCD / 4 := sorry

end part_a_part_b_l271_271525


namespace part1_part2_l271_271016

def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1 (x : ℝ) : 
  ∀ x, f x 2 ≥ 4 ↔ (x ≤ 3 / 2 ∨ x ≥ 11 / 2) :=
by sorry

theorem part2 (x a : ℝ) :
  f x a ≥ 4 → (a ≤ -1 ∨ a ≥ 3) :=
by sorry

end part1_part2_l271_271016


namespace even_function_max_min_values_monotonic_function_b_range_l271_271383

noncomputable def f (x : ℝ) (b : ℝ) (c : ℝ) : ℝ := x^2 + b*x + c

theorem even_function_max_min_values (b c : ℝ) (h_even : ∀ x : ℝ, f(-x) b c = f x b c) (h_f1 : f 1 b c = 0) :
  b = 0 ∧ c = -1 ∧ ∀ x ∈ set.Icc (-1 : ℝ) (3 : ℝ), f x 0 (-1) ≤ 8 ∧ f x 0 (-1) ≥ -1 := sorry

theorem monotonic_function_b_range (b c : ℝ) :
  (b ≥ 2 ∨ b ≤ -6) ↔ ∀ x y ∈ set.Icc (-1 : ℝ) (3 : ℝ), (x < y → f x b c ≤ f y b c ∨ f x b c ≥ f y b c) := sorry

end even_function_max_min_values_monotonic_function_b_range_l271_271383


namespace round_5614_to_nearest_hundredth_l271_271499

theorem round_5614_to_nearest_hundredth : Real.round_to_hundredth 5.614 = 5.61 :=
by
  sorry

end round_5614_to_nearest_hundredth_l271_271499


namespace best_k_k_l271_271337

theorem best_k_k' (v w x y z : ℝ) (hv : 0 < v) (hw : 0 < w) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  1 < (v / (v + w) + w / (w + x) + x / (x + y) + y / (y + z) + z / (z + v)) ∧ 
  (v / (v + w) + w / (w + x) + x / (x + y) + y / (y + z) + z / (z + v)) < 4 :=
sorry

end best_k_k_l271_271337


namespace derivative_f_l271_271794

variable x : ℝ

noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x - 1)

theorem derivative_f : deriv f x = 2 / (2 * x - 1) :=
by
  sorry

end derivative_f_l271_271794


namespace friend_jogging_time_l271_271058

theorem friend_jogging_time (D : ℝ) (my_time : ℝ) (friend_speed : ℝ) :
  my_time = 3 * 60 →
  friend_speed = 2 * (D / my_time) →
  (D / friend_speed) = 90 :=
by
  sorry

end friend_jogging_time_l271_271058


namespace washing_time_per_dish_l271_271666

theorem washing_time_per_dish (time_sweeping_per_room time_laundry_per_load num_rooms_sweeping num_loads_laundry num_dishes_billy : ℕ) (H₁ : time_sweeping_per_room = 3) (H₂ : time_laundry_per_load = 9) (H₃ : num_rooms_sweeping = 10) (H₄ : num_loads_laundry = 2) (H₅ : num_dishes_billy = 6) : 
  (time_sweeping_per_room * num_rooms_sweeping) = (time_laundry_per_load * num_loads_laundry + 12) → 
  let t := 12 / num_dishes_billy in t = 2 :=
by {
  intros H₆,
  have total_time_anna : time_sweeping_per_room * num_rooms_sweeping = 30,
  { rw [H₁, H₃], norm_num },
  have total_time_billy : time_laundry_per_load * num_loads_laundry = 18,
  { rw [H₂, H₄], norm_num },
  have required_time_billy := total_time_anna - total_time_billy,
  { rw [total_time_anna, total_time_billy], norm_num },
  have t_def : t = required_time_billy / num_dishes_billy,
  { rw [required_time_billy, H₅], norm_num },
  exact eq.symm t_def,
  sorry
}

end washing_time_per_dish_l271_271666


namespace part1_problem_l271_271929

theorem part1_problem
  (A B C : Real.Angle)
  (a b c : ℝ)
  (cosA : Real.cos A)
  (sinA : Real.sin A)
  (sin2B : Real.sin (2 * B))
  (cos2B : Real.cos (2 * B))
  (hC : C = 2 * π / 3)
  (h : cosA / (1 + sinA) = sin2B / (1 + cos2B))
  : B = π / 6 := by
  sorry

end part1_problem_l271_271929


namespace train_length_l271_271642

variable (T1 T2 P : ℕ) (L V : ℝ)

theorem train_length (h1 : T1 = 39) (h2 : T2 = 9) (h3 : P = 1000) 
  (h4 : L = V * T2) (h5 : L + P = V * T1) : L = 300 :=
by
  have h6: V = L / 9, from sorry,
  have h7: L + 1000 = (L / 9) * 39, from sorry,
  have h8: L + 1000 = 39 * L / 9, from sorry,
  have h9: 9 * L + 9 * 1000 = 39 * L, from sorry,
  have h10: 9L + 9000 = 39L, from sorry,
  have h11: 9000 = 30L, from sorry,
  have h12: L = 9000 / 30, from sorry,
  exact sorry

end train_length_l271_271642


namespace part1_solution_set_part2_range_of_a_l271_271005

-- Define the function f for a general a
def f (x a : ℝ) : ℝ := |x - a^2| + |x - (2 * a) + 1|

-- Part 1: When a = 2
theorem part1_solution_set (x : ℝ) (h : f x 2 ≥ 4) : x ≤ 3/2 ∨ x ≥ 11/2 :=
  sorry

-- Part 2: Range of values for a
theorem part2_range_of_a (a : ℝ) (h : ∀ x, f x a ≥ 4) : a ≤ -1 ∨ a ≥ 3 :=
  sorry

end part1_solution_set_part2_range_of_a_l271_271005


namespace john_vegetables_used_l271_271107

noncomputable def pounds_of_beef_bought : ℕ := 4
noncomputable def pounds_of_beef_used : ℕ := pounds_of_beef_bought - 1
noncomputable def pounds_of_vegetables_used : ℕ := 2 * pounds_of_beef_used

theorem john_vegetables_used : pounds_of_vegetables_used = 6 :=
by
  -- the proof can be provided here later
  sorry

end john_vegetables_used_l271_271107


namespace sum_of_first_90_terms_l271_271837

theorem sum_of_first_90_terms :
  (∃ a d : ℝ, (∑ i in finset.range 15, (a + i * d)) = 150 ∧ (∑ i in finset.range 75, (a + i * d)) = 75) →
  (∑ i in finset.range 90, (a + i * d)) = -112.5 :=
sorry

end sum_of_first_90_terms_l271_271837


namespace decreasing_interval_of_f_on_0_to_pi_l271_271833

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := 2 * real.sin (2 * x + φ)

theorem decreasing_interval_of_f_on_0_to_pi :
  (∃ φ : ℝ, 0 < φ ∧ φ < (π / 2) ∧ f 0 φ = sqrt 3) →
  ∃ a b : ℝ, (a = π / 12) ∧ (b = 7 * π / 12) ∧ 
  (∀ x : ℝ, a ≤ x ∧ x ≤ b → ∀ ε > 0, f(x-ε) > f(x) ∨ f(x+ε) < f(x)) :=
by
  sorry

end decreasing_interval_of_f_on_0_to_pi_l271_271833


namespace angle_AP7P8_l271_271086

variable {α β γ δ ε ζ η θ ι : Type}

def P1 (A B : Type) := sorry
def P2 (B C : Type) := sorry
def P3 (A B : Type) := sorry
def P4 (B C : Type) := sorry
def P5 (A B : Type) := sorry
def P6 (B C : Type) := sorry
def P7 (A B : Type) := sorry
def P8 (B C : Type) := sorry

theorem angle_AP7P8 
  (A B C : Type)
  (P1_Points : ∀ P1 P3 P5 P7 : A, PointOnLine P1 (BA A) ∧ PointOnLine P3 (BA A) ∧ PointOnLine P5 (BA A) ∧ PointOnLine P7 (BA A))
  (P2_Points : ∀ P2 P4 P6 P8 : B, PointOnLine P2 (BC B) ∧ PointOnLine P4 (BC B) ∧ PointOnLine P6 (BC B) ∧ PointOnLine P8 (BC B))
  (EqualSegments : BP1 = P1P2 ∧ P1P2 = P2P3 ∧ P2P3 = P3P4 ∧ P3P4 = P4P5 ∧ P4P5 = P5P6 ∧ P5P6 = P6P7 ∧ P6P7 = P7P8)
  (Angle_ABC_5Deg : ∠ ABC = 5) : ∠ AP7P8 = 40 :=
by
  sorry

end angle_AP7P8_l271_271086


namespace possible_values_x_l271_271261

variable (a b x : ℕ)

theorem possible_values_x (h1 : a + b = 20)
                          (h2 : a * x + b * 3 = 109) :
    x = 10 ∨ x = 52 :=
sorry

end possible_values_x_l271_271261


namespace units_digit_of_quotient_l271_271663

theorem units_digit_of_quotient : 
  (7 ^ 2023 + 4 ^ 2023) % 9 = 2 → 
  (7 ^ 2023 + 4 ^ 2023) / 9 % 10 = 0 :=
by
  -- condition: calculation of modulo result
  have h1 : (7 ^ 2023 + 4 ^ 2023) % 9 = 2 := sorry

  -- we have the target statement here
  exact sorry

end units_digit_of_quotient_l271_271663


namespace harold_kept_20_marbles_l271_271812

theorem harold_kept_20_marbles
  (initial_marbles : ℕ)
  (friends : ℕ)
  (marbles_per_friend : ℕ)
  (initial_marbles_eq : initial_marbles = 100)
  (friends_eq : friends = 5)
  (marbles_per_friend_eq : marbles_per_friend = 16) :
  initial_marbles - friends * marbles_per_friend = 20 :=
by
  rw [initial_marbles_eq, friends_eq, marbles_per_friend_eq]
  exact calc
    100 - 5 * 16 = 100 - 80 : by rfl
            ... = 20 : by rfl

end harold_kept_20_marbles_l271_271812


namespace fixed_point_l271_271094

-- Given declarations
variables {A B C D I J E F T : Type}

-- Triangle vertices
def in_triangle (A B C : Type) := true -- simplistic placeholder for triangle configuration

-- Definitions of obtained points and properties
def point_on_BC (D : Type) (BC : Type) := true
def incenter (I : Type) (ABD : Type) := true
def A_excenter (J : Type) (ADC : Type) := true
def perp_foot_I (E : Type) (BC : Type) := true
def perp_foot_J (F : Type) (BC : Type) := true
def midpoint (T : Type) (EF : Type) := true
def perp_line_l (T : Type) (IJ : Type) := true

-- The fixed point theorem to be proven
theorem fixed_point (A B C D I J E F T : Type)
    (h1 : in_triangle A B C)
    (h2 : point_on_BC D (B, C))
    (h3 : incenter I (A, B, D))
    (h4 : A_excenter J (A, D, C))
    (h5 : perp_foot_I E B C )
    (h6 : perp_foot_J F B C )
    (h7 : midpoint T (E, F))
    (h8 : perp_line_l T (I, J)) :
    ∃ S : Type, perp_line_l T (I, J) ∧ (some_fixed_point_satisfies S) := 
sorry

end fixed_point_l271_271094


namespace xyz_eq_7cubed_l271_271420

theorem xyz_eq_7cubed (x y z : ℤ) (h1 : x^2 * y * z^3 = 7^4) (h2 : x * y^2 = 7^5) : x * y * z = 7^3 := 
by 
  sorry

end xyz_eq_7cubed_l271_271420


namespace part1_solution_set_part2_range_of_a_l271_271025

def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1_solution_set (x : ℝ) : 
  (f x 2 ≥ 4) ↔ (x ≤ 3 / 2 ∨ x ≥ 11 / 2) :=
sorry

theorem part2_range_of_a (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 4) → (a ≤ -1 ∨ a ≥ 3) :=
sorry

end part1_solution_set_part2_range_of_a_l271_271025


namespace cat_food_percentage_is_approximately_2_33_l271_271297

noncomputable def food_percentage_per_cat 
  (food_per_dog : ℝ)
  (num_dogs : ℕ)
  (num_cats : ℕ)
  (num_hamsters : ℕ)
  (total_food_for_cats : ℝ)
  (total_food_for_hamsters : ℝ) : ℝ :=
  let total_food := num_dogs * food_per_dog + total_food_for_cats + total_food_for_hamsters in
  (total_food_for_cats / num_cats) / total_food * 100

theorem cat_food_percentage_is_approximately_2_33 
  (D : ℝ)   -- food amount per dog
  (h₁ : ∀ d₁ d₂ : ℝ, d₁ = D ∧ d₂ = D)  -- all dogs receive D food
  (h₂ : ∀ c₁ c₂ : ℝ, c₁ = (1.5 * D) / 6 ∧ c₂ = (1.5 * D) / 6)  -- all cats receive equal food
  (h₃ : ∀ h₁ h₂ : ℝ, h₁ = (0.25 * D) / 10 ∧ h₂ = (0.25 * D) / 10)  -- all hamsters receive equal food
  (h₄ : 6 * (1.5 * D) / 6 = 1.5 * D)  -- total food for all cats
  (h₅ : 10 * (0.25 * D) / 10 = 0.25 * D)  -- total food for all hamsters
  (approx : food_percentage_per_cat D 9 6 10 (1.5 * D) (0.25 * D) ≈ 2.33) : 
  true :=
begin
  sorry
end

end cat_food_percentage_is_approximately_2_33_l271_271297


namespace cube_vertices_shapes_l271_271490

theorem cube_vertices_shapes :
  ∀ (V : set (fin 8)), (∃ (s : ℕ), s ∈ {1, 3, 4, 5} ∧
    (s = 1 ∨ s = 2 ∨ s = 3 ∨ s = 4 ∨ s = 5)) :=
begin
  intros V,
  -- Define that selecting 4 vertices of a cube forms shapes corresponding to the sequence numbers {1, 3, 4, 5}
  sorry
end

end cube_vertices_shapes_l271_271490


namespace sin_840_eq_sqrt3_over_2_l271_271259

theorem sin_840_eq_sqrt3_over_2 : sin (840 * real.pi / 180) = sqrt 3 / 2 := by
  -- We translate degree measure to radians
  have h₁ : 840 * real.pi / 180 = 14 * real.pi / 3, by sorry
  -- Use the reduction formula for the sine function
  have h₂ : sin (14 * real.pi / 3) = sin (2 * real.pi + 2 * real.pi / 3), by sorry
  -- Known identity: sin (2π + θ) = sin θ
  have h₃ : sin (2 * real.pi + 2 * real.pi / 3) = sin (2 * real.pi / 3), by sorry
  -- Use the identity for the sine function within 2π
  have h⁴ : sin (2 * real.pi / 3) = sin (real.pi - real.pi / 3), by sorry
  -- Use the identity for π - α
  have h⁵ : sin (real.pi - real.pi / 3) = sin (real.pi / 3), by sorry
  -- Finally apply the known value
  have h⁶ : sin (real.pi / 3) = sqrt 3 / 2, by sorry
  -- Combine all to get the desired result
  rw [h₁, h₂, h₃, h⁴, h⁵, h⁶]

end sin_840_eq_sqrt3_over_2_l271_271259


namespace square_area_of_equilateral_triangle_on_hyperbola_l271_271213

noncomputable def equilateral_triangle_on_hyperbola (vertices : List (ℝ × ℝ)) : Prop :=
  vertices.length = 3 ∧ 
  ∀ (v : ℝ × ℝ), v ∈ vertices → v.1 * v.2 = 4

def centroid (A B C : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

def square_of_area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  let area := (1/2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))
  area^2

theorem square_area_of_equilateral_triangle_on_hyperbola :
  ∃ (A B C : ℝ × ℝ),
    (centroid A B C = (2, 2)) ∧ 
    (equilateral_triangle_on_hyperbola [A, B, C]) ∧ 
    (square_of_area_of_triangle A B C = 108) :=
begin
  sorry
end

end square_area_of_equilateral_triangle_on_hyperbola_l271_271213


namespace problem_real_numbers_inequality_l271_271953

open Real

theorem problem_real_numbers_inequality 
  (a1 b1 a2 b2 : ℝ) :
  a1 * b1 + a2 * b2 ≤ sqrt (a1^2 + a2^2) * sqrt (b1^2 + b2^2) :=
by 
  sorry

end problem_real_numbers_inequality_l271_271953


namespace crayons_left_in_drawer_l271_271864

theorem crayons_left_in_drawer :
  let initial_crayons := 7.5 in
  let mary_took := 3.2 in
  let mark_took := 0.5 in
  let jane_took := 1.3 in
  let mary_returned := 0.7 in
  let sarah_added := 3.5 in
  let tom_added := 2.8 in
  let alice_took := 1.5 in
  initial_crayons - mary_took - mark_took - jane_took + mary_returned + sarah_added + tom_added - alice_took = 8.0 :=
by
  sorry

end crayons_left_in_drawer_l271_271864


namespace rationalize_denominator_l271_271954

theorem rationalize_denominator :
  (∃ (a b c d e f g h i j k l : Real), 
    (a = Real.cbrt 9 ∧ b = Real.sqrt 5) ∧
    (f = Real.cbrt 2 ∧ g = Real.sqrt 5) ∧
    (i = (Real.sqrt 5) * (Real.cbrt 2) ∧ j = (Real.sqrt 5) * (Real.cbrt 3) ∧ 
     k = (Real.cbrt 4) ∧ 
     l = (i - j - 5) / (k - 5))) →
  Real.cbrt 9 + Real.sqrt 5 = 
  (Real.sqrt 5 * Real.cbrt 2 - 6 * Real.cbrt 3 - 5) / (Real.cbrt 4 - 5) :=
by
  sorry

end rationalize_denominator_l271_271954


namespace kramer_pack_cases_l271_271466

theorem kramer_pack_cases (boxes_per_minute : ℕ) (cases_per_2_hours : ℕ) (hours : ℕ) :
  boxes_per_minute = 10 → cases_per_2_hours = 240 → ∀ x : ℕ, 120 * x = (cases_per_2_hours / 2) * x := by
  intros h1 h2 x
  rw [h2]
  simp
  sorry

end kramer_pack_cases_l271_271466


namespace locus_of_circumcenters_is_circle_l271_271806

open Real EuclideanGeometry

-- Given the setup of two intersecting circles and points P, Q, C, A, B as described
variable {C : Point}
variable {P Q : Point}
variable {circle1 circle2 : Circle}

-- Points P and Q lie on both circles
axiom P_on_circles : point_on_circle P circle1 ∧ point_on_circle P circle2
axiom Q_on_circles : point_on_circle Q circle1 ∧ point_on_circle Q circle2

-- Point C is an arbitrary point distinct from P and Q on the first circle
axiom C_on_circle1 : point_on_circle C circle1
axiom C_ne_P : C ≠ P
axiom C_ne_Q : C ≠ Q

-- Points A and B are the second intersections of lines CP and CQ with the second circle
axiom A_on_circle2 : point_on_circle (line_intersection (line_through C P) circle2) circle2
axiom B_on_circle2 : point_on_circle (line_intersection (line_through C Q) circle2) circle2

-- Prove that the locus of the circumcenters of ΔABC is a circle
theorem locus_of_circumcenters_is_circle :
  ∃ (circ_center : Circle), 
  ∀ (C : Point), 
  C ≠ P → C ≠ Q → point_on_circle C circle1 →
  let A := line_intersection (line_through C P) circle2 in
  let B := line_intersection (line_through C Q) circle2 in
  circumcenter A B C ∈ circ_center :=
sorry

end locus_of_circumcenters_is_circle_l271_271806


namespace find_the_number_l271_271418

theorem find_the_number (n : ℤ) 
    (h : 45 - (28 - (n - (15 - 18))) = 57) :
    n = 37 := 
sorry

end find_the_number_l271_271418


namespace num_elements_in_set_l271_271201

theorem num_elements_in_set :
  {p : ℝ × ℝ | ∃ x y, p = ⟨x, y⟩ ∧ log (x^3 + (1/3) * y^3 + (1/9)) = log x + log y}.card = 1 :=
sorry

end num_elements_in_set_l271_271201


namespace empty_strange_sequences_iff_nilpotent_l271_271875

def strange_sequence (S : Finset ℕ) (A : Matrix (Fin S.card) (Fin S.card) ℕ) (x : ℕ → Fin S.card) :=
  (∀ (i : ℕ), x i ∈ S) ∧ (∀ (k : ℕ), A (x k) (x (k + 1)) = 1)

noncomputable def is_nilpotent (A : Matrix (Fin S.card) (Fin S.card) ℕ) :=
  ∃ m : ℕ, A^m = 0

theorem empty_strange_sequences_iff_nilpotent
  (S : Finset ℕ) (hS : ∃ n : ℕ, S = Finset.range n)
  (A : Matrix (Fin S.card) (Fin S.card) ℕ)
  (hA_entries : ∀ i j, A i j = 0 ∨ A i j = 1) :
  (¬∃ x : ℕ → Fin S.card, strange_sequence S A x) ↔ is_nilpotent A := 
sorry

end empty_strange_sequences_iff_nilpotent_l271_271875


namespace find_integers_between_squares_l271_271406

-- Defining the core problem for the Lean proof statement
def num_satisfying_integers (a b : ℝ) : ℕ :=
  { n : ℕ | a < Real.sqrt (n * (n - 1)) ∧ Real.sqrt (n * (n - 1)) < b }.toFinset.card

theorem find_integers_between_squares : num_satisfying_integers 2000 2005 = 5 := by
  sorry

end find_integers_between_squares_l271_271406


namespace good_apples_count_l271_271682

def total_apples : ℕ := 14
def unripe_apples : ℕ := 6

theorem good_apples_count : total_apples - unripe_apples = 8 :=
by
  unfold total_apples unripe_apples
  sorry

end good_apples_count_l271_271682


namespace part1_solution_set_part2_range_of_a_l271_271008

-- Define the function f for a general a
def f (x a : ℝ) : ℝ := |x - a^2| + |x - (2 * a) + 1|

-- Part 1: When a = 2
theorem part1_solution_set (x : ℝ) (h : f x 2 ≥ 4) : x ≤ 3/2 ∨ x ≥ 11/2 :=
  sorry

-- Part 2: Range of values for a
theorem part2_range_of_a (a : ℝ) (h : ∀ x, f x a ≥ 4) : a ≤ -1 ∨ a ≥ 3 :=
  sorry

end part1_solution_set_part2_range_of_a_l271_271008


namespace find_B_min_fraction_of_squares_l271_271908

-- Lean 4 statement for part (1)
theorem find_B (A B C a b c : ℝ) (h_triangle : A + B + C = π) 
(h_sides : a > 0 ∧ b > 0 ∧ c > 0) 
(h_identity : (cos A) / (1 + sin A) = (sin (2 * B)) / (1 + cos (2 * B))) 
(h_C : C = 2 * π / 3) : 
B = π / 6 := sorry

-- Lean 4 statement for part (2)
theorem min_fraction_of_squares (A B C a b c : ℝ) 
(h_triangle : A + B + C = π) 
(h_sides : a > 0 ∧ b > 0 ∧ c > 0) 
(h_identity : (cos A) / (1 + sin A) = (sin (2 * B)) / (1 + cos (2 * B))) 
(h_C : C = 2 * π / 3) : 
∀ a b c, ∃ m, m = 4 * sqrt 2 - 5 ∧ (a^2 + b^2) / c^2 = m := sorry

end find_B_min_fraction_of_squares_l271_271908


namespace Jill_mow_time_approx_l271_271105

noncomputable def mow_time (length width : ℕ) (swath_width overlap : ℕ) (speed : ℕ) : ℚ :=
let effective_swath_width := (swath_width - overlap) / 12
let num_strips := width / effective_swath_width
let total_distance := num_strips * length
let time := total_distance / speed in
time

theorem Jill_mow_time_approx : 
  mow_time 100 160 30 6 4500 ≈ 1.8 := sorry

end Jill_mow_time_approx_l271_271105


namespace x_varies_z_pow_l271_271824

variable (k j : ℝ)
variable (y z : ℝ)

-- Given conditions
def x_varies_y_squared (x : ℝ) := x = k * y^2
def y_varies_z_cuberoot_squared := y = j * z^(2/3)

-- To prove: 
theorem x_varies_z_pow (x : ℝ) (h1 : x_varies_y_squared k y x) (h2 : y_varies_z_cuberoot_squared j z y) : ∃ m : ℝ, x = m * z^(4/3) :=
by
  sorry

end x_varies_z_pow_l271_271824


namespace correct_answer_l271_271893

variable {x y : ℝ}

def p : Prop :=
  ∀ x y : ℝ, x = y → y ≠ 0 → x / y = 1

def q : Prop :=
  ∀ x1 x2 : ℝ, x1 ≠ x2 → (real.exp x1 - real.exp x2) / (x1 - x2) > 0

theorem correct_answer :
  (p ∨ q) ∧ (¬p ∨ q) :=
by
  -- Proof of the theorem is omitted
  sorry

end correct_answer_l271_271893


namespace general_formula_arithmetic_sum_of_sequence_l271_271363

def arithmetic_sequence (n : ℕ) : ℕ → ℤ 
| n := -n + 2

def sequence_sum (n : ℕ) : ℤ := (n : ℤ) / (1 - 2 * (n : ℤ))

theorem general_formula_arithmetic (S_3 S_5 : ℤ) (h1 : S_3 = 0) (h2 : S_5 = -5) : 
  ∀ n : ℕ, arithmetic_sequence n = -n + 2 := 
by
  sorry

theorem sum_of_sequence (n : ℕ) (h : ∀ n : ℕ, arithmetic_sequence n = -n + 2) : 
  sequence_sum n = n / (1 - 2 * n) := 
by
  sorry

end general_formula_arithmetic_sum_of_sequence_l271_271363


namespace ball_bounces_height_l271_271608

theorem ball_bounces_height (initial_height : ℝ) (decay_factor : ℝ) (threshold : ℝ) (n : ℕ) :
  initial_height = 20 →
  decay_factor = 3/4 →
  threshold = 2 →
  n = 9 →
  initial_height * (decay_factor ^ n) < threshold :=
by
  intros
  sorry

end ball_bounces_height_l271_271608


namespace distance_apart_after_2_hours_l271_271290

theorem distance_apart_after_2_hours 
  (Jay_speed : ℝ) (Paul_speed : ℝ) (time_hours : ℝ) 
  (Jay_speed_cond : Jay_speed = 0.8 / 15)
  (Paul_speed_cond : Paul_speed = 3 / 30)
  (time_hours_cond : time_hours = 2):
  let time_minutes := time_hours * 60 in
  let Jay_distance := Jay_speed * time_minutes in
  let Paul_distance := Paul_speed * time_minutes in
  Jay_distance + Paul_distance = 18.4 := 
by 
  { -- proof placeholder
    sorry }


end distance_apart_after_2_hours_l271_271290


namespace observation_confidence_l271_271072

-- Define the conditions
def confidence_level := 0.95
def probability_condition (k : ℝ) := P (k > 3.841) = 0.05

-- Define the theorem to be proven
theorem observation_confidence (k : ℝ) (P : ℝ → Prop) :
  probability_condition k → confidence_level = 0.95 → k^2 > 3.841 :=
by sorry

end observation_confidence_l271_271072


namespace hexagon_vector_expression_count_l271_271743

variable (V : Type) [AddCommGroup V] [Module ℝ V]

noncomputable def is_hexagon (A B C D E F : V) : Prop :=
  -- Conditions defining a regular hexagon
  (A - B) = (B - C) ∧ (B - C) = (C - D) ∧ (C - D) = (D - E) ∧
  (D - E) = (E - F) ∧ (E - F) = (F - A)

theorem hexagon_vector_expression_count
  (A B C D E F : V) (h : is_hexagon A B C D E F) :
  (
    ((B - C) + (C - D) + (E - C) = A - C) ∧
    ((2 • (B - C) + (1 : ℝ) • C = A - C)) ∧
    ((F - E) + (E - D) = A - C) ∧
    ((B - C) - (B - A) = A - C)
  ) → (num : ℕ) := {
  sorry  
}

end hexagon_vector_expression_count_l271_271743


namespace intersection_domain_range_l271_271786

def f (x : ℝ) := 1 / sqrt (1 - x)
def g (x : ℝ) := Real.log x

theorem intersection_domain_range : 
  let A := {x : ℝ | 1 - x > 0}
  let B := {y : ℝ | ∃ x : ℝ, y = Real.log x}
  A ∩ B = {x : ℝ | x < 1} :=
by
  sorry

end intersection_domain_range_l271_271786


namespace monday_occurs_five_times_l271_271178

theorem monday_occurs_five_times (N : ℕ) :
  (∃ (days : list ℕ), 
    days.length = 30 ∧
    ∃ friday_indexes, 
      list.fridays days friday_indexes ∧
      friday_indexes.length = 5
  ) →
  (∃ (days : list ℕ),
    days.length = 31 ∧
    ∀ (day_indexes : list ℕ),
      list.mondays days day_indexes →
      day_indexes.length = 5
  ) :=
sorry

end monday_occurs_five_times_l271_271178


namespace sum_floor_expression_l271_271661

-- Defining the floor function
def floor (x : ℝ) : ℤ := Int.floor x

-- Defining the expression inside the sum
def expr (k : ℕ) : ℤ := floor ((k + Real.sqrt k : ℝ) / k)

-- The main statement to prove
theorem sum_floor_expression : (∑ k in Finset.range 1989 \ Finset.range 1, expr (k + 2)) = 1989 := by
  sorry

end sum_floor_expression_l271_271661


namespace ellipse_equation_and_min_area_of_triangle_l271_271379

theorem ellipse_equation_and_min_area_of_triangle
  (a b : ℝ)
  (h1 : a = sqrt 3 * b)
  (h2 : (x : ℝ) → Set (x : ℝ, y : ℝ) := { p | x^2 / a^2 + p.snd^2 / b^2 = 1 })
  (hdirectrix : x = -3 * sqrt 2 / 2 = -a^2 / c)
  (l1 l2 : ℝ → ℝ)
  (h3 : l1 = (λ x, sqrt 3 / 3 * x))
  (h4 : l2 = (λ x, -sqrt 3 / 3 * x))
  (A B : ℝ × ℝ)
  (h5 : ∃ x1, A = (x1, sqrt 3 / 3 * x1))
  (h6 : ∃ x2, B = (x2, -sqrt 3 / 3 * x2))
  (P : ℝ × ℝ)
  (h7 : ∃ λ : ℝ, λ > 0 ∧ (P = (1 / (1 + λ) • A + λ / (1 + λ) • B))
  (h8 : P ∈ { p | p.fst^2 / a^2 - p.snd^2 / b^2 = 1 }) :

  -- Part 1: The equation of the ellipse
  (a^2 = 3 ∧ b^2 = 1) ∧

  -- Part 2: The minimum value of the area of ΔOAB
  (∀ (λ : ℝ), λ > 0 → (1 / 2 * (abs (λ) + 1 / abs (λ) + 2) * sqrt 3 ≥ 2 * sqrt 3) :=
sorry

end ellipse_equation_and_min_area_of_triangle_l271_271379


namespace hiker_miles_l271_271814

-- Defining the conditions as a def
def total_steps (flips : ℕ) (additional_steps : ℕ) : ℕ := flips * 100000 + additional_steps

def steps_per_mile : ℕ := 1500

-- The target theorem to prove the number of miles walked
theorem hiker_miles (flips : ℕ) (additional_steps : ℕ) (s_per_mile : ℕ) 
  (h_flips : flips = 72) (h_additional_steps : additional_steps = 25370) 
  (h_s_per_mile : s_per_mile = 1500) : 
  (total_steps flips additional_steps) / s_per_mile = 4817 :=
by
  -- sorry is used to skip the actual proof
  sorry

end hiker_miles_l271_271814


namespace tax_increase_proof_l271_271614

variables (old_tax_rate new_tax_rate : ℝ) (old_income new_income : ℝ)

def old_taxes_paid (old_tax_rate old_income : ℝ) : ℝ := old_tax_rate * old_income

def new_taxes_paid (new_tax_rate new_income : ℝ) : ℝ := new_tax_rate * new_income

def increase_in_taxes (old_tax_rate new_tax_rate old_income new_income : ℝ) : ℝ :=
  new_taxes_paid new_tax_rate new_income - old_taxes_paid old_tax_rate old_income

theorem tax_increase_proof :
  increase_in_taxes 0.20 0.30 1000000 1500000 = 250000 := by
  sorry

end tax_increase_proof_l271_271614


namespace multiples_count_in_range_200_600_l271_271815

theorem multiples_count_in_range_200_600 : 
  let n := 11,
      a := 200,
      b := 600,
      m := Nat.lcm 12 9 in
  m = 36 → (λ (x : ℕ), (x >= a ∧ x <= b ∧ x % m = 0)) = n :=
by {
  n_def : decide_eq (n = 11)
  range_def : decide_eq (range a b = (a, b)),
  lcm_def : decide_eq (m = 36), sorry
}

end multiples_count_in_range_200_600_l271_271815


namespace reuleaux_triangle_area_eq_l271_271180

noncomputable def area_of_reuleaux_triangle (side_length : ℝ) : ℝ :=
  let h := (side_length * real.sqrt 3) / 2
  let equilateral_area := (side_length * h) / 2
  let sector_area := (π * side_length ^ 2) / 6
  let calota_area := sector_area - equilateral_area
  3 * calota_area + equilateral_area

theorem reuleaux_triangle_area_eq :
  area_of_reuleaux_triangle 1 = (π / 2) - (real.sqrt 3 / 2) :=
  sorry

end reuleaux_triangle_area_eq_l271_271180


namespace greatest_positive_integer_x_l271_271315

theorem greatest_positive_integer_x : ∃ (x : ℕ), (x > 0) ∧ (∀ y : ℕ, y > 0 → (y^3 < 20 * y → y ≤ 4)) ∧ (x^3 < 20 * x) ∧ ∀ z : ℕ, (z > 0) → (z^3 < 20 * z → x ≥ z)  :=
sorry

end greatest_positive_integer_x_l271_271315


namespace exists_infinite_n_with_prime_divisors_l271_271350

-- Definition: greatest prime divisor of a positive integer.
def p (n : ℕ) : ℕ := sorry -- Placeholder for the actual definition.

-- Theorem: There are infinitely many positive integers \( n \) such that \( p(n) < p(n+1) < p(n+2) \).
theorem exists_infinite_n_with_prime_divisors (h : ∀ n : ℕ, n > 1 → p(n) > 0) :
  ∃^∞ n : ℕ, n > 1 ∧ p(n) < p(n + 1) ∧ p(n + 2) > p(n + 1) := sorry

end exists_infinite_n_with_prime_divisors_l271_271350


namespace round_5614_to_nearest_hundredth_l271_271498

theorem round_5614_to_nearest_hundredth : Real.round_to_hundredth 5.614 = 5.61 :=
by
  sorry

end round_5614_to_nearest_hundredth_l271_271498


namespace vertex_locus_is_parabola_l271_271135

noncomputable def vertex_locus_curved (a c : ℝ) (h1 : 0 < a) (h2 : 0 < c) :
  set (ℝ × ℝ) :=
  {p : (ℝ × ℝ) | ∃ t : ℝ, p = (-(t/(2*a)), c - (t^2/(4*a)))}

theorem vertex_locus_is_parabola (a c : ℝ) (h1 : 0 < a) (h2 : 0 < c) :
  ∀ p ∈ vertex_locus_curved a c h1 h2, ∃ x : ℝ, p = (x, -a*x^2 + c) :=
by
  sorry

end vertex_locus_is_parabola_l271_271135


namespace students_with_dogs_l271_271635

theorem students_with_dogs (total_students : ℕ) (half_students : total_students = 100)
                           (girls_percentage : ℕ) (boys_percentage : ℕ)
                           (girls_dog_percentage : ℕ) (boys_dog_percentage : ℕ)
                           (P1 : half_students / 2 = 50)
                           (P2 : girls_dog_percentage = 20)
                           (P3 : boys_dog_percentage = 10) :
                           (50 * girls_dog_percentage / 100 + 
                            50 * boys_dog_percentage / 100) = 15 :=
by sorry

end students_with_dogs_l271_271635


namespace possible_values_of_n_are_1_prime_or_prime_squared_l271_271438

/-- A function that determines if an n x n grid with n marked squares satisfies the condition
    that every rectangle of exactly n grid squares contains at least one marked square. -/
def satisfies_conditions (n : ℕ) (marked_squares : List (ℕ × ℕ)) : Prop :=
  n.succ.succ ≤ marked_squares.length ∧ ∀ (a b : ℕ), a * b = n → ∃ x y, (x, y) ∈ marked_squares ∧ x < n ∧ y < n

/-- The main theorem stating the possible values of n. -/
theorem possible_values_of_n_are_1_prime_or_prime_squared :
  ∀ (n : ℕ), (∃ p : ℕ, Prime p ∧ (n = 1 ∨ n = p ∨ n = p^2)) ↔ satisfies_conditions n marked_squares :=
by
  sorry

end possible_values_of_n_are_1_prime_or_prime_squared_l271_271438


namespace seats_needed_on_bus_l271_271204

variable (f t tr dr c h : ℕ)

def flute_players := 5
def trumpet_players := 3 * flute_players
def trombone_players := trumpet_players - 8
def drummers := trombone_players + 11
def clarinet_players := 2 * flute_players
def french_horn_players := trombone_players + 3

theorem seats_needed_on_bus :
  f = 5 →
  t = 3 * f →
  tr = t - 8 →
  dr = tr + 11 →
  c = 2 * f →
  h = tr + 3 →
  f + t + tr + dr + c + h = 65 :=
by
  sorry

end seats_needed_on_bus_l271_271204


namespace area_of_triangle_min_value_of_a_l271_271844

namespace TriangleProblem

variables {A B C : ℝ}
variables {a b c : ℝ} 

-- First proof question
theorem area_of_triangle (h1 : sqrt 3 * c * cos A = a * sin C) (h2 : 4 * sin C = c^2 * sin B) :
  (1/2 * b * c * sin A) = sqrt 3 := 
sorry

-- Second proof question
theorem min_value_of_a (h1 : sqrt 3 * c * cos A = a * sin C) (h2 : innerProduct (vecAB) (vecAC) = 4) :
  a = 2 * sqrt 2 :=
sorry

end TriangleProblem

end area_of_triangle_min_value_of_a_l271_271844


namespace sum_of_lengths_constant_l271_271874

noncomputable section

-- Definitions for an equilateral triangle and the lengths
variables {ABC : Type*} [MetricSpace ABC] [EquilateralTriangle ABC]
variables (A B C P : ABC)
variables (AC1 BA1 CB1 : Length)

-- Given conditions
def isParallelLine (p q r s : ABC) : Prop := LineSegment pq.isParallel LineSegment rs

-- Problem statement in Lean 4
theorem sum_of_lengths_constant (P : Point (inside ABC)) :
  ∃ AC1 BA1 CB1 : Length, 
    isParallelLine A_1'A_1 A B ∧ 
    isParallelLine B_1'B_1 B C ∧ 
    isParallelLine C_1'C_1 C A ∧
    AC1 + BA1 + CB1 = AB := 
begin
  sorry
end

end sum_of_lengths_constant_l271_271874


namespace chess_tournament_points_distribution_l271_271431

theorem chess_tournament_points_distribution (n : ℕ) (P : ℕ → ℝ) :
  n = 20 ∧
  ∀ i, 1 ≤ i ∧ i ≤ 20 → ∑ j in (finset.range 20).erase i, P j ≤ 19 ∧
  ∑ i in finset.range 20, P i = 190 ∧
  ∃ P19, P19 = P 19 ∧ P19 = 9.5 →
  ∃ P20, P20 = P 20 ∧ (P20 = 0 ∨ P20 = 0.5) →
  ∃ P1, P1 = P 1 ∧ (P1 = 10.5 ∨ P1 = 10) := by
  sorry

end chess_tournament_points_distribution_l271_271431


namespace true_statement_l271_271044

-- Definitions for the propositions
def p : Prop := ∀ x > 0, sin x < x

def l1 (a : ℝ) : ℝ → ℝ → Prop := λ x y, a * x + 2 * y + 1 = 0
def l2 (a : ℝ) : ℝ → ℝ → Prop := λ x y, x + (a - 1) * y - 1 = 0
def lines_parallel (a : ℝ) : Prop := ∀ x y, l1 a x y ↔ l2 a x y
def q : Prop := (lines_parallel 2) ∨ (lines_parallel (-1))

-- Main theorem to prove
theorem true_statement : p ∨ q :=
by
  sorry

end true_statement_l271_271044


namespace find_sum_of_abc_l271_271522

theorem find_sum_of_abc
  (h : ∀ x : ℝ, sin x ^ 2 + sin (2 * x) ^ 2 + sin (3 * x) ^ 2 + sin (4 * x) ^ 2 = 2) :
  let a := 1
  let b := 2
  let c := 5
  (cos a x * cos b x * cos c x = 0) ∧ (a + b + c = 8) :=
by
  sorry

end find_sum_of_abc_l271_271522


namespace find_B_min_fraction_of_squares_l271_271909

-- Lean 4 statement for part (1)
theorem find_B (A B C a b c : ℝ) (h_triangle : A + B + C = π) 
(h_sides : a > 0 ∧ b > 0 ∧ c > 0) 
(h_identity : (cos A) / (1 + sin A) = (sin (2 * B)) / (1 + cos (2 * B))) 
(h_C : C = 2 * π / 3) : 
B = π / 6 := sorry

-- Lean 4 statement for part (2)
theorem min_fraction_of_squares (A B C a b c : ℝ) 
(h_triangle : A + B + C = π) 
(h_sides : a > 0 ∧ b > 0 ∧ c > 0) 
(h_identity : (cos A) / (1 + sin A) = (sin (2 * B)) / (1 + cos (2 * B))) 
(h_C : C = 2 * π / 3) : 
∀ a b c, ∃ m, m = 4 * sqrt 2 - 5 ∧ (a^2 + b^2) / c^2 = m := sorry

end find_B_min_fraction_of_squares_l271_271909


namespace distance_between_lines_is_correct_l271_271190

-- Define the first line equation in standard form
def line1 (x y : ℝ) : Prop := 6 * x + 8 * y - 6 = 0

-- Define the second line equation in standard form
def line2 (x y : ℝ) : Prop := 6 * x + 8 * y + 5 = 0

-- Define the distance function between two parallel lines
noncomputable def distance_between_parallel_lines (A B C₁ C₂ : ℝ) : ℝ :=
  |C₂ - C₁| / Real.sqrt (A^2 + B^2)

-- Define a constant for the computed distance
def computed_distance : ℝ := distance_between_parallel_lines 6 8 (-6) 5

-- Prove that the distance is 1.1
theorem distance_between_lines_is_correct : computed_distance = 1.1 :=
by
  sorry

end distance_between_lines_is_correct_l271_271190


namespace find_B_min_value_a2_b2_c2_l271_271900

theorem find_B (A B C : ℝ) (h1 : C = 2 * π / 3) 
  (h2 : (cos A) / (1 + sin A) = (sin (2 * B)) / (1 + cos (2 * B))) : B = π / 6 :=
sorry

theorem min_value_a2_b2_c2 (A B C a b c : ℝ) (h1 : C = 2 * π / 3)
  (h2 : (cos A) / (1 + sin A) = (sin (2 * B)) / (1 + cos (2 * B))) 
  (h3 : triangles_side_identity a b c A B C AE BA CE) :
  (a^2 + b^2) / c^2 = 4 * Real.sqrt 2 - 5 :=
sorry

end find_B_min_value_a2_b2_c2_l271_271900


namespace find_a_l271_271376

variable (a : ℝ)

-- Define the conditions
def line1_perpendicular (a : ℝ) : Prop :=
  let slope1 := 3 / (3 - 3^a)
  let slope2 := 2
  slope1 * slope2 = -1

-- The theorem we need to prove
theorem find_a (h : line1_perpendicular a) : a = 2 := 
sorry

end find_a_l271_271376


namespace example_problem_l271_271229

-- Definitions and conditions derived from the original problem statement
def smallest_integer_with_two_divisors (m : ℕ) : Prop := m = 2
def second_largest_integer_with_three_divisors_less_than_100 (n : ℕ) : Prop := n = 25

theorem example_problem (m n : ℕ) 
    (h1 : smallest_integer_with_two_divisors m) 
    (h2 : second_largest_integer_with_three_divisors_less_than_100 n) : 
    m + n = 27 :=
by sorry

end example_problem_l271_271229


namespace find_B_min_value_a2_b2_c2_l271_271901

theorem find_B (A B C : ℝ) (h1 : C = 2 * π / 3) 
  (h2 : (cos A) / (1 + sin A) = (sin (2 * B)) / (1 + cos (2 * B))) : B = π / 6 :=
sorry

theorem min_value_a2_b2_c2 (A B C a b c : ℝ) (h1 : C = 2 * π / 3)
  (h2 : (cos A) / (1 + sin A) = (sin (2 * B)) / (1 + cos (2 * B))) 
  (h3 : triangles_side_identity a b c A B C AE BA CE) :
  (a^2 + b^2) / c^2 = 4 * Real.sqrt 2 - 5 :=
sorry

end find_B_min_value_a2_b2_c2_l271_271901


namespace exponent_zero_value_of_neg_3_raised_to_zero_l271_271987

theorem exponent_zero (x : ℤ) (hx : x ≠ 0) : x ^ 0 = 1 :=
by
  -- Proof goes here
  sorry

theorem value_of_neg_3_raised_to_zero : (-3 : ℤ) ^ 0 = 1 :=
by
  exact exponent_zero (-3) (by norm_num)

end exponent_zero_value_of_neg_3_raised_to_zero_l271_271987


namespace fibonacci_divisible_by_2014_l271_271129

-- Define the Fibonacci sequence
def Fibonacci : ℕ → ℕ
| 0 := 0
| 1 := 1
| 2 := 1
| (n + 2) := Fibonacci (n + 1) + Fibonacci n

-- Theorem statement for the proof problem
theorem fibonacci_divisible_by_2014 :
  ∃ n ≥ 1, 2014 ∣ Fibonacci n :=
sorry

end fibonacci_divisible_by_2014_l271_271129


namespace likelihood_ratio_sqrt2_l271_271147

namespace ProbabilityProblem

open Nat

theorem likelihood_ratio_sqrt2 (n : ℕ) :
  let P (k : ℕ) := (Nat.choose (2 * n) k / 2 ^ (2 * n) : ℝ)
  let PA := P n
  let PB := (Nat.choose (2 * n) n / 2 ^ (2 * n)) ^ 2 / (Nat.choose (4 * n) (2 * n) / 2 ^ (4 * n))
  PA ≠ 0 ∧ PB ≠ 0 → PB / PA = Real.sqrt 2 :=
by
  sorry

end ProbabilityProblem

end likelihood_ratio_sqrt2_l271_271147


namespace part1_solution_set_part2_range_of_a_l271_271027

def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1_solution_set (x : ℝ) : 
  (f x 2 ≥ 4) ↔ (x ≤ 3 / 2 ∨ x ≥ 11 / 2) :=
sorry

theorem part2_range_of_a (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 4) → (a ≤ -1 ∨ a ≥ 3) :=
sorry

end part1_solution_set_part2_range_of_a_l271_271027


namespace find_a_and_b_l271_271356

theorem find_a_and_b (a b : ℝ) 
  (h1 : ∀ x ∈ set.Icc (-(real.pi / 3)) (2 * real.pi / 3), 
        y = a * real.sin (2 * x - real.pi / 3) + b → 
        -2 ≤ y ∧ y ≤ 4) :
  (a = 3 ∧ b = 1) ∨ (a = -3 ∧ b = 1) :=
by {
  sorry
}

end find_a_and_b_l271_271356


namespace dmitriev_is_older_l271_271617

variables (Alekseev Borisov Vasilyev Grigoryev Dima Dmitriev : ℤ)

def Lesha := Alekseev + 1
def Borya := Borisov + 2
def Vasya := Vasilyev + 3
def Grisha := Grigoryev + 4

theorem dmitriev_is_older :
  Dima + 10 = Dmitriev :=
sorry

end dmitriev_is_older_l271_271617


namespace price_per_maple_tree_l271_271051

theorem price_per_maple_tree 
  (cabin_price : ℕ) (initial_cash : ℕ) (remaining_cash : ℕ)
  (num_cypress : ℕ) (price_cypress : ℕ)
  (num_pine : ℕ) (price_pine : ℕ)
  (num_maple : ℕ) 
  (total_raised_from_trees : ℕ) :
  cabin_price = 129000 ∧ 
  initial_cash = 150 ∧ 
  remaining_cash = 350 ∧ 
  num_cypress = 20 ∧ 
  price_cypress = 100 ∧ 
  num_pine = 600 ∧ 
  price_pine = 200 ∧ 
  num_maple = 24 ∧ 
  total_raised_from_trees = 129350 - initial_cash → 
  (price_maple : ℕ) = 300 :=
by 
  sorry

end price_per_maple_tree_l271_271051


namespace maximum_value_of_sum_l271_271065

variables (x y : ℝ)

def s : ℝ := x + y

theorem maximum_value_of_sum (h : s ≤ 9) : s = 9 :=
sorry

end maximum_value_of_sum_l271_271065


namespace cuberoot_eq_3_implies_cube_eq_19683_l271_271822

theorem cuberoot_eq_3_implies_cube_eq_19683 (x : ℝ) (h : (x + 6)^(1/3) = 3) : (x + 6)^3 = 19683 := by
  sorry

end cuberoot_eq_3_implies_cube_eq_19683_l271_271822


namespace steps_left_to_climb_l271_271646

-- Define the conditions
def total_stairs : ℕ := 96
def climbed_stairs : ℕ := 74

-- The problem: Prove that the number of stairs left to climb is 22
theorem steps_left_to_climb : (total_stairs - climbed_stairs) = 22 :=
by 
  sorry

end steps_left_to_climb_l271_271646


namespace max_perimeter_isosceles_triangle_division_l271_271506

theorem max_perimeter_isosceles_triangle_division :
  ∀ (base height : ℝ) (pieces : ℕ),
    base = 8 ∧ height = 10 ∧ pieces = 8 →
    let perimeters := λ k, 1 + Real.sqrt (height^2 + (k : ℝ)^2) + Real.sqrt (height^2 + (k + 1 : ℝ)^2) in
    (Finset.range pieces).sup perimeters = 22.21 :=
begin
  -- Base, height and pieces given
  intros base height pieces h,
  -- Extract conditions from h
  rcases h with ⟨hb, hh, hp⟩,
  sorry
end

end max_perimeter_isosceles_triangle_division_l271_271506


namespace average_cost_per_individual_l271_271594

-- Given conditions
variable (n : ℕ) (total_bill : ℝ) (gratuity_rate : ℝ) (total_with_gratuity : ℝ)

-- Specific values for our problem
def group_size := 7
def total_bill_incl_gratuity := 840
def gratuity_percentage := 0.20

-- Theorem statement
theorem average_cost_per_individual : 
  total_with_gratuity = total_bill_incl_gratuity ∧ 
  n = group_size ∧ 
  gratuity_rate = gratuity_percentage →
  total_bill / n = 100 :=
by
  sorry

end average_cost_per_individual_l271_271594


namespace find_B_min_value_a2_b2_c2_l271_271899

theorem find_B (A B C : ℝ) (h1 : C = 2 * π / 3) 
  (h2 : (cos A) / (1 + sin A) = (sin (2 * B)) / (1 + cos (2 * B))) : B = π / 6 :=
sorry

theorem min_value_a2_b2_c2 (A B C a b c : ℝ) (h1 : C = 2 * π / 3)
  (h2 : (cos A) / (1 + sin A) = (sin (2 * B)) / (1 + cos (2 * B))) 
  (h3 : triangles_side_identity a b c A B C AE BA CE) :
  (a^2 + b^2) / c^2 = 4 * Real.sqrt 2 - 5 :=
sorry

end find_B_min_value_a2_b2_c2_l271_271899


namespace integral_f_l271_271877

def f (x : ℝ) : ℝ :=
  if x ∈ Set.Icc 0 1 then x^2 else
  if x ∈ Set.Ioi 1 ∩ Set.Iic (Real.exp 1) then 1 / x else 0

theorem integral_f : ∫ x in 0..Real.exp 1, f x = 4 / 3 :=
  sorry

end integral_f_l271_271877


namespace Darwin_fraction_spent_on_food_l271_271312

variable (money : ℕ) (final_remaining : ℕ)
variable (gas_expense remaining_after_gas money_spent_on_food : ℕ)

def fraction_spent_on_food_after_gas (money : ℕ) (final_remaining : ℕ) : Prop :=
  let gas_expense := money / 3
  let remaining_after_gas := money - gas_expense
  let money_spent_on_food := remaining_after_gas - final_remaining
  money_spent_on_food / remaining_after_gas = 1 / 4

theorem Darwin_fraction_spent_on_food (h₁ : money = 600) (h₂ : final_remaining = 300) :
    fraction_spent_on_food_after_gas 600 300 :=
by
  let gas_expense := money / 3
  let remaining_after_gas := money - gas_expense
  let money_spent_on_food := remaining_after_gas - final_remaining
  have h : money_spent_on_food / remaining_after_gas = 1 / 4 := sorry
  exact h

end Darwin_fraction_spent_on_food_l271_271312


namespace count_multiples_of_7_not_14_300_l271_271405

open Finset

def count_multiples_of_7_not_14 (n : ℕ) : ℕ :=
  (Icc 1 n).filter (λ k, k % 7 = 0 ∧ k % 14 ≠ 0).card

theorem count_multiples_of_7_not_14_300 : count_multiples_of_7_not_14 300 = 21 :=
  by sorry

end count_multiples_of_7_not_14_300_l271_271405


namespace percentage_of_pear_juice_in_blend_l271_271488

theorem percentage_of_pear_juice_in_blend 
  (pear_juice_per_pear : ℝ) 
  (orange_juice_per_orange : ℝ) 
  (equal_volume : ℝ) :
  (9 / 18 : ℝ) = 0.50 :=
by
  -- Assume pear_juice_per_pear and orange_juice_per_orange values from conditions
  have h1 : pear_juice_per_pear = 9 / 4, by sorry
  have h2 : orange_juice_per_orange = 10 / 3, by sorry
  -- The equal volume is used to form the juice blend
  have h3 : equal_volume = 9, by sorry
  -- Calculation to find the percentage of pear juice in the blend
  have h4 : (equal_volume / (equal_volume * 2)) = (9 / 18), by sorry
  -- Thus, the percentage of pear juice in the blend is 50%
  exact rfl

#eval percentage_of_pear_juice_in_blend (9 / 4) (10 / 3) 9

end percentage_of_pear_juice_in_blend_l271_271488


namespace find_B_min_value_a2_b2_c2_l271_271898

theorem find_B (A B C : ℝ) (h1 : C = 2 * π / 3) 
  (h2 : (cos A) / (1 + sin A) = (sin (2 * B)) / (1 + cos (2 * B))) : B = π / 6 :=
sorry

theorem min_value_a2_b2_c2 (A B C a b c : ℝ) (h1 : C = 2 * π / 3)
  (h2 : (cos A) / (1 + sin A) = (sin (2 * B)) / (1 + cos (2 * B))) 
  (h3 : triangles_side_identity a b c A B C AE BA CE) :
  (a^2 + b^2) / c^2 = 4 * Real.sqrt 2 - 5 :=
sorry

end find_B_min_value_a2_b2_c2_l271_271898


namespace frequency_hundred_times_greater_l271_271947

theorem frequency_hundred_times_greater (x : ℕ) (h₁ : x - 1 ∈ ℕ) (h₂ : 10^2 = 100) : 
    x = 3 → (x - 2 = 1) := 
by
  intros
  apply eq_of_sub_eq
  sorry

end frequency_hundred_times_greater_l271_271947


namespace part1_solution_set_part2_range_of_a_l271_271003

-- Define the function f for a general a
def f (x a : ℝ) : ℝ := |x - a^2| + |x - (2 * a) + 1|

-- Part 1: When a = 2
theorem part1_solution_set (x : ℝ) (h : f x 2 ≥ 4) : x ≤ 3/2 ∨ x ≥ 11/2 :=
  sorry

-- Part 2: Range of values for a
theorem part2_range_of_a (a : ℝ) (h : ∀ x, f x a ≥ 4) : a ≤ -1 ∨ a ≥ 3 :=
  sorry

end part1_solution_set_part2_range_of_a_l271_271003


namespace complex_modulus_multiplication_vector_dot_product_zero_l271_271246

theorem complex_modulus_multiplication (z1 z2 : ℂ) : 
  abs (z1 * z2) = abs z1 * abs z2 := 
sorry

theorem vector_dot_product_zero {V : Type*} [inner_product_space ℝ V] 
  {a b : V} (ha : a ≠ 0) (hb : b ≠ 0) :
  norm (a + b) = norm (a - b) → 
  ⟪a, b⟫ = 0 := 
sorry

end complex_modulus_multiplication_vector_dot_product_zero_l271_271246


namespace count_arrangements_l271_271056

theorem count_arrangements : 
  let letters : List Char := ['B', 'A₁', 'N₁', 'A₂', 'N₂', 'A₃'] in
  (letters.perm.length = 720) := sorry

end count_arrangements_l271_271056


namespace divide_triangle_equal_area_l271_271365

theorem divide_triangle_equal_area
  (A B C P Q : Point)
  (hABC : equilateral_triangle A B C)
  (hP : on_side P A B)
  (hQ : on_side Q B C) :
  ∃ (L1 L2 : Line), divides_triangle_eq_area ABC L1 L2 P Q :=
by
  sorry

end divide_triangle_equal_area_l271_271365


namespace min_value_a_plus_b_l271_271394

theorem min_value_a_plus_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : (2 / a) + (2 / b) = 1) :
  a + b >= 8 :=
sorry

end min_value_a_plus_b_l271_271394


namespace product_sequence_10_l271_271317

theorem product_sequence_10 : (∏ k in Finset.range 9, (1 + (1 / (k + 1)))) = 10 := 
by
  sorry

end product_sequence_10_l271_271317


namespace part1_part2_l271_271907

-- Define the condition for the first part
def condition (A B C : ℝ) : Prop :=
  (C = 2 * Real.pi / 3) ∧ (1 + Real.sin A ≠ 0) ∧ (2 * Real.cos B ≠ 0) ∧ 
  (Real.cos A / (1 + Real.sin A) = Real.sin (2 * B) / (1 + Real.cos (2 * B)))

-- Theorem for the first part: If \( C = \frac{2\pi}{3} \), then \( B = \frac{\pi}{6} \)
theorem part1 (A B C : ℝ) (h : condition A B C) : B = Real.pi / 6 :=
  sorry

-- Define the condition for the second part as the side ratios expression
def ratio_expression (a b c : ℝ) : ℝ :=
  (a^2 + b^2) / (c^2)

-- Theorem for the second part: Minimum value of \(\frac{a^2 + b^2}{c^2}\)
theorem part2 (a b c : ℝ) (A B C : ℝ) 
  (h : condition A B C) : ratio_expression a b c = 4 * Real.sqrt 2 - 5 :=
  sorry

end part1_part2_l271_271907


namespace part1_solution_set_part2_values_of_a_l271_271033

-- Parameters and definitions for the function and conditions
def f (x a : ℝ) := |x - a^2| + |x - (2*a + 1)|

-- Part (1) When a = 2, the inequality's solution set
theorem part1_solution_set (x : ℝ) : f x 2 ≥ 4 ↔ (x ≤ 3/2 ∨ x ≥ 11/2) := 
sorry

-- Part (2) Range of values for a given f(x, a) ≥ 4
theorem part2_values_of_a (a : ℝ) : 
∀ x, f x a ≥ 4 → (a ≤ -1 ∨ a ≥ 3) :=
sorry

end part1_solution_set_part2_values_of_a_l271_271033


namespace hypotenuse_unique_l271_271309

theorem hypotenuse_unique (a b : ℝ) (h: ∃ x : ℝ, x^2 = a^2 + b^2 ∧ x > 0) : 
  ∃! c : ℝ, c^2 = a^2 + b^2 :=
sorry

end hypotenuse_unique_l271_271309


namespace sticks_form_equilateral_triangle_l271_271701

theorem sticks_form_equilateral_triangle {n : ℕ} (h1 : n ≥ 5) (h2 : n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5) :
  (∃ a b c, a = b ∧ b = c ∧ a + b + c = (n * (n + 1)) / 2) :=
by
  sorry

end sticks_form_equilateral_triangle_l271_271701


namespace cousin_room_distribution_cousin_room_distribution_l271_271939

-- Define the problem conditions as constants
constant numberOfCousins : ℕ := 5
constant numberOfRooms : ℕ := 5

-- State the theorem equivalent to the original math problem
theorem cousin_room_distribution : 
  ∃ (ways : ℕ), ways = 67 ∧ 
  (ways = (∑ i in (finset.powerset (finset.range numberOfCousins)).filter (λ s, s.card ≤ numberOfRooms), 
    card partitions_of_phone_distinct(number_of_cousins:5,set: nat)):= 67)
proof
construct the proof with combination sum index and total partitions of all distinguish 
number_of_cousins 5 in 5 identifical rooms 
  mention lean function here 
qed


# Translate the mathematical problem into Lean statement
theorem cousin_room_distribution :
  (∑ s in (finset.powerset (finset.range numberOfCousins)) ∩ {s | s.card ≤ numberOfRooms}, 
     1) = 67 := sorry

end cousin_room_distribution_cousin_room_distribution_l271_271939


namespace euler_formula_proof_l271_271162

noncomputable
def euler_formula (O1 O2 : Point) (r R : ℝ) (triangle : Triangle) : Prop :=
distance O1 O2 ^ 2 = R ^ 2 - 2 * r * R

theorem euler_formula_proof (O1 O2 : Point) (r R : ℝ) (triangle : Triangle) 
  (incircle : is_incircle O1 r triangle) (circumcircle : is_circumcircle O2 R triangle) : 
  euler_formula O1 O2 r R triangle :=
by
  sorry

end euler_formula_proof_l271_271162


namespace amy_tickets_l271_271659

theorem amy_tickets (initial_tickets : ℕ) (total_tickets : ℕ) (bought_tickets : ℕ) 
  (h1 : initial_tickets = 33) (h2 : total_tickets = 54): bought_tickets = 21 :=
by 
  rw [h1, h2]
  have buy_tickets: 54 - 33 = 21 := sorry
  rw buy_tickets
  sorry

end amy_tickets_l271_271659


namespace small_forward_time_l271_271491

theorem small_forward_time 
  (pg sg pf c : ℕ)
  (average_time_per_player : ℕ) 
  (num_players : ℕ)
  (total_time : ℕ)
  (total_pg : ℕ)
  (total_sg : ℕ)
  (total_pf : ℕ)
  (total_c : ℕ)
  (remaining_time : ℕ) :
  total_pg = 130 →
  total_sg = 145 →
  total_pf = 60 →
  total_c = 180 →
  average_time_per_player = 120 →
  num_players = 5 →
  total_time = average_time_per_player * num_players →
  remaining_time = total_time - (total_pg + total_sg + total_pf + total_c) →
  remaining_time = 85 :=
by {
  intros,
  sorry
}

end small_forward_time_l271_271491


namespace seats_needed_l271_271205

def flute_players : ℕ := 5
def trumpet_players : ℕ := 3 * flute_players
def trombone_players : ℕ := trumpet_players - 8
def drummers : ℕ := trombone_players + 11
def clarinet_players : ℕ := 2 * flute_players
def french_horn_players : ℕ := trombone_players + 3
def total_seats_needed : ℕ := flute_players + trumpet_players + trombone_players + drummers + clarinet_players + french_horn_players

theorem seats_needed (s : ℕ) (h : s = 65) : total_seats_needed = s :=
by {
  have h_flutes : flute_players = 5 := rfl,
  have h_trumpets : trumpet_players = 3 * flute_players := rfl,
  have h_trombones : trombone_players = trumpet_players - 8 := rfl,
  have h_drums : drummers = trombone_players + 11 := rfl,
  have h_clarinets : clarinet_players = 2 * flute_players := rfl,
  have h_french_horns : french_horn_players = trombone_players + 3 := rfl,
  have h_total : total_seats_needed = flute_players + trumpet_players + trombone_players + drummers + clarinet_players + french_horn_players := rfl,
  rw [h_flutes, h_trumpets, h_trombones, h_drums, h_clarinets, h_french_horns] at h_total,
  simp only [flute_players, trumpet_players, trombone_players, drummers, clarinet_players, french_horn_players] at h_total,
  norm_num at h_total,
  exact h,
}

end seats_needed_l271_271205


namespace pascal_fifth_number_l271_271866

theorem pascal_fifth_number {n : ℕ} (h : n = 10) : binom n 4 = 210 :=
by
  rw [h]
  sorry

end pascal_fifth_number_l271_271866


namespace radius_range_of_circle_l271_271792

theorem radius_range_of_circle (r : ℝ) :
  (∀ (x y : ℝ), (x - 3)^2 + (y + 5)^2 = r^2 → 
  (abs (4*x - 3*y - 2) = 1)) →
  4 < r ∧ r < 6 :=
by
  sorry

end radius_range_of_circle_l271_271792


namespace ratio_M_N_l271_271891

variables {M Q P N R : ℝ}

-- Conditions
def condition1 : M = 0.40 * Q := sorry
def condition2 : Q = 0.25 * P := sorry
def condition3 : N = 0.75 * R := sorry
def condition4 : R = 0.60 * P := sorry

-- Theorem to prove
theorem ratio_M_N : M / N = 2 / 9 := sorry

end ratio_M_N_l271_271891


namespace prob_m_eq_n_prob_m_gt_n_l271_271616

-- The probability space when rolling a fair die twice consists of 36 outcomes
-- for simplicity we consider the die rolls to be from 1 to 6 as usual and independent.

def sample_space : Finset (ℕ × ℕ) := 
  (Finset.fin_range 7).product (Finset.fin_range 7)

-- Event A: outcomes where the first die equals the second die
def event_A : Finset (ℕ × ℕ) := 
  Finset.filter (fun p => p.1 = p.2) sample_space

-- Event B: outcomes where the first die is greater than the second die
def event_B : Finset (ℕ × ℕ) := 
  Finset.filter (fun p => p.1 > p.2) sample_space

-- Probability of an event is the ratio of its cardinality to the total sample space cardinality
def probability (event : Finset (ℕ × ℕ)) : ℚ :=
  event.card / sample_space.card

-- Problem 1: probability that the first die is equal to the second die is 1/6
theorem prob_m_eq_n : probability event_A = 1 / 6 := sorry

-- Problem 2: probability that the first die is greater than the second die is 5/12
theorem prob_m_gt_n : probability event_B = 5 / 12 := sorry

end prob_m_eq_n_prob_m_gt_n_l271_271616


namespace smallest_number_to_divide_3600_is_15_l271_271242

-- Conditions
def is_perfect_cube (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = k^3

def prime_factors (n : ℕ) (p : ℕ) : ℕ :=
  if n % p = 0 then 1 + prime_factors (n / p) p else 0

def smallest_divisor_to_perfect_cube (n : ℕ) (d : ℕ) : Prop :=
  is_perfect_cube (n / d) ∧ ∀ m, m < d → ¬is_perfect_cube (n / m)

-- Problem statement
theorem smallest_number_to_divide_3600_is_15 : 
  smallest_divisor_to_perfect_cube 3600 15 :=
sorry

end smallest_number_to_divide_3600_is_15_l271_271242


namespace angle_S_is_36_l271_271955

-- Definitions used in the conditions.
variable (P Q R S T : Type) -- Replace these with appropriate types
variable (angle P angle Q angle R angle S : ℝ)
variable (PT QT TR RS : ℝ)

-- Conditions
axiom intersect_at_T : ∃ T, (PT = QT) ∧ (QT = TR) ∧ (TR = RS)
axiom angle_rel : angle R = 3 * angle P

-- Problem: Prove the degree measure of angle S is 36 degrees.
theorem angle_S_is_36 : angle S = 36 :=
by sorry

end angle_S_is_36_l271_271955


namespace range_of_a_l271_271423

noncomputable def f (x : ℝ) : ℝ := x^2 - 4 * x

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Icc (-4 : ℝ) a, -4 ≤ f x ∧ f x ≤ 32) ->
  2 ≤ a ∧ a ≤ 8 :=
sorry

end range_of_a_l271_271423


namespace prop3_prop4_l271_271771

variables (α β : Type) -- as planes
variable (m n : Type) -- as lines

-- Definitions for parallel and perpendicular relationships (we assume these are already defined)
class parallel (A B : Type) : Prop :=
(is_parallel : ∀ {A B : Type}, // Define the relationship here)
class perpendicular (A B : Type) : Prop :=
(is_perpendicular : ∀ {A B : Type}, // Define the relationship here)

-- Proposition 3
theorem prop3 (m α n β : Type) [perpendicular m α] [perpendicular n β] [perpendicular m n] :
  perpendicular α β := sorry

-- Proposition 4
theorem prop4 (α β : Type) [perpendicular α β] (m n : Type) [perpendicular m α] [perpendicular n β] :
  perpendicular m n := sorry

end prop3_prop4_l271_271771


namespace minimized_angle_line_eqn_l271_271970

theorem minimized_angle_line_eqn (M : Point := ⟨2, 1⟩) (C_center : Point := ⟨1, 0⟩) (C_radius : ℝ := 2) :
  ∀ (l : Line), (l.passes_through M) ∧ (l.intersects_circle (circle C_center C_radius)) ∧ minimized_angle (C_center, M, l) → l.equation = "x + y - 3 = 0" :=
by
  sorry

end minimized_angle_line_eqn_l271_271970


namespace polar_coordinates_of_curve_range_of_ratios_l271_271444

theorem polar_coordinates_of_curve (x y : ℝ) (α β : ℝ)
  (h1 : x = sqrt 2 + cos α)
  (h2 : y = sin α) :
  ∃ (ρ θ : ℝ), ρ^2 - 2 * sqrt 2 * ρ * cos θ + 1 = 0 := 
sorry

theorem range_of_ratios (β : ℝ) 
  (hβ : -π / 4 < β ∧ β < 0) :
  ∃ r : ℝ, r ∈ Ioo (sqrt 2 / 2) sqrt 2 ∧
    r = (2 * sqrt 2 * cos (β + π / 4)) / (2 * sqrt 2 * cos β) :=
sorry

end polar_coordinates_of_curve_range_of_ratios_l271_271444


namespace part1_problem_l271_271930

theorem part1_problem
  (A B C : Real.Angle)
  (a b c : ℝ)
  (cosA : Real.cos A)
  (sinA : Real.sin A)
  (sin2B : Real.sin (2 * B))
  (cos2B : Real.cos (2 * B))
  (hC : C = 2 * π / 3)
  (h : cosA / (1 + sinA) = sin2B / (1 + cos2B))
  : B = π / 6 := by
  sorry

end part1_problem_l271_271930


namespace work_fraction_together_l271_271347

theorem work_fraction_together (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (1/x) + (1/y) = (1/x) + (1/y) :=
begin
  sorry
end

end work_fraction_together_l271_271347


namespace odd_handshakes_even_count_l271_271320

theorem odd_handshakes_even_count
  (N : ℕ) -- total number of people
  (m : ℕ) -- total number of handshakes
  (n : Fin N → ℕ) -- number of handshakes made by each person
  (H_sum : ∑ i, n i = 2 * m) : 
  ∃ k : Fin N, 2 ∣ k ∧ ∃ f : Fin N → ℕ, ∀ i, n i % 2 = 1 → f i % 2 = 1 :=
sorry

end odd_handshakes_even_count_l271_271320


namespace slope_of_line_l271_271241

theorem slope_of_line (x1 y1 x2 y2 : ℝ) (hx1 : x1 = 1) (hy1 : y1 = 3) (hx2 : x2 = 4) (hy2 : y2 = -3) : 
  (y2 - y1) / (x2 - x1) = -2 :=
by
  rw [hx1, hy1, hx2, hy2]
  norm_num
  sorry

end slope_of_line_l271_271241


namespace find_b_l271_271979

theorem find_b (a b : ℤ) 
  (h1 : a * b = 2 * (a + b) + 14) 
  (h2 : b - a = 3) : 
  b = 8 :=
sorry

end find_b_l271_271979


namespace inequality_solution_set_minimum_abc_l271_271790

theorem inequality_solution_set : 
  { x : Real | abs (x + 2) - abs (2 * x - 2) > 2 } = { x : Real | (2 / 3 : Real) < x ∧ x < 2 } := sorry

theorem minimum_abc (a b c : Real) (h : 1 < a ∧ 1 < b ∧ 1 < c) : 
  (a - 1) * (b - 1) * (c - 1) = 1 → min abc = 8 :=
sorry

end inequality_solution_set_minimum_abc_l271_271790


namespace count_ordered_triples_l271_271470

def T := {i | 1 ≤ i ∧ i ≤ 20}

def succ (a b : ℕ) : Prop :=
  (0 < a - b ∧ a - b ≤ 10) ∨ (b - a > 10)

theorem count_ordered_triples : 
  let triples := { (x, y, z) | x ∈ T ∧ y ∈ T ∧ z ∈ T ∧ succ x y ∧ succ y z ∧ succ z x }
  in finset.card triples = 1260 :=
sorry

end count_ordered_triples_l271_271470


namespace grogg_expected_value_l271_271811

theorem grogg_expected_value (n : ℕ) (p : ℝ) (h_n : 2 ≤ n) (h_p : 0 < p ∧ p < 1) :
  (p + n * p^n * (1 - p) = 1) ↔ (p = 1 / n^(1/n:ℝ)) :=
sorry

end grogg_expected_value_l271_271811


namespace roots_quadratic_equation_l271_271360

theorem roots_quadratic_equation (x1 x2 : ℝ) (h1 : x1^2 - x1 - 1 = 0) (h2 : x2^2 - x2 - 1 = 0) :
  (x2 / x1) + (x1 / x2) = -3 :=
by
  sorry

end roots_quadratic_equation_l271_271360


namespace distance_between_midpoints_l271_271511

/-- Given points in the plane and certain distances between them,
    prove the specified distance between midpoints of certain segments. -/
theorem distance_between_midpoints 
  (A B C D Q R S T U V : Type → Prop)
  (midpoint_AB : Q)
  (midpoint_AC : R)
  (midpoint_AD : S)
  (midpoint_BC : T)
  (midpoint_BD : U)
  (midpoint_CD : V)
  (QR_eq : dist Q R = 2001)
  (SU_eq : dist S U = 2002)
  (TV_eq : dist T V = 2003) :
  dist (midpoint (Q,U)) (midpoint (R,V)) = 2001 := 
sorry

end distance_between_midpoints_l271_271511


namespace min_cos_C_l271_271840

theorem min_cos_C (A B C : ℝ) (h : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧ A + B + C = π)
  (h1 : (1 / Real.sin A) + (2 / Real.sin B) = 3 * ((1 / Real.tan A) + (1 / Real.tan B))) :
  Real.cos C ≥ (2 * Real.sqrt 10 - 2) / 9 := 
sorry

end min_cos_C_l271_271840


namespace cooking_time_eq_80_l271_271996

-- Define the conditions
def hushpuppies_per_guest : Nat := 5
def number_of_guests : Nat := 20
def hushpuppies_per_batch : Nat := 10
def time_per_batch : Nat := 8

-- Calculate total number of hushpuppies needed
def total_hushpuppies : Nat := hushpuppies_per_guest * number_of_guests

-- Calculate number of batches needed
def number_of_batches : Nat := total_hushpuppies / hushpuppies_per_batch

-- Calculate total time needed
def total_time_needed : Nat := number_of_batches * time_per_batch

-- Statement to prove the correctness
theorem cooking_time_eq_80 : total_time_needed = 80 := by
  sorry

end cooking_time_eq_80_l271_271996


namespace range_of_function_l271_271569

-- Define the original function
def f (x : ℝ) := log 2 (sqrt (sin x))

-- Define the domain
def domain (x : ℝ) := 0 < x ∧ x < π

-- State the theorem
theorem range_of_function : ∀ y, (∃ x, domain x ∧ f x = y) ↔ y ∈ Iic 0 := sorry

end range_of_function_l271_271569


namespace rooks_rearrangement_possible_l271_271949

/-- Place eight rooks on an 8x8 chessboard so no two rooks can attack each other. 
   Paint 27 of the remaining squares red and prove the possibility to re-arrange the rooks
   based on the given conditions. -/
theorem rooks_rearrangement_possible :
  ∀ (rooks : Finset (Fin 8 × Fin 8)) (red_squares : Finset (Fin 8 × Fin 8)),
    rooks.card = 8 ∧ (∀ (p1 p2 : Fin 8 × Fin 8), p1 ≠ p2 → (p1.1 ≠ p2.1 ∧ p1.2 ≠ p2.2)) ∧
    red_squares.card = 27 →
    ∃ (new_rooks : Finset (Fin 8 × Fin 8)),
      new_rooks.card = 8 ∧
      (∀ (p1 p2 : Fin 8 × Fin 8), p1 ≠ p2 → (p1.1 ≠ p2.1 ∧ p1.2 ≠ p2.2)) ∧
      (∀ (rook : Fin 8 × Fin 8), rook ∈ new_rooks → rook ∉ red_squares) ∧
      ∃ (rook : Fin 8 × Fin 8), rook ∉ rooks ∧ rook ∈ new_rooks
:= sorry

end rooks_rearrangement_possible_l271_271949


namespace equation_one_solution_equation_two_solution_l271_271175

theorem equation_one_solution (x : ℝ) : (6 * x - 7 = 4 * x - 5) ↔ (x = 1) := by
  sorry

theorem equation_two_solution (x : ℝ) : ((x + 1) / 2 - 1 = 2 + (2 - x) / 4) ↔ (x = 4) := by
  sorry

end equation_one_solution_equation_two_solution_l271_271175


namespace total_surface_area_of_square_pyramid_is_correct_l271_271867

-- Define the base side length and height from conditions
def a : ℝ := 3
def PD : ℝ := 4

-- Conditions
def square_pyramid : Prop :=
  let AD := a
  let PA := Real.sqrt (PD^2 - a^2)
  let Area_PAD := (1 / 2) * AD * PA
  let Area_PCD := Area_PAD
  let Area_base := a * a
  let Total_surface_area := Area_base + 2 * Area_PAD + 2 * Area_PCD
  Total_surface_area = 9 + 6 * Real.sqrt 7

-- Theorem statement
theorem total_surface_area_of_square_pyramid_is_correct : square_pyramid := sorry

end total_surface_area_of_square_pyramid_is_correct_l271_271867


namespace yogurt_strawberry_probability_l271_271322

theorem yogurt_strawberry_probability :
  let p_first_days := 1/2
      p_last_days := 3/4 
      case_1a_prob := (Real.binom 3 2) * (Real.binom 3 2) * (p_first_days^2) * (p_last_days^2)
      case_1b_prob := (Real.binom 3 3) * (Real.binom 3 1) * (p_first_days^3) * (p_last_days)
      total_prob_per_comb := case_1a_prob + case_1b_prob
      num_ways := Real.binom 6 4 
      final_prob := num_ways * total_prob_per_comb
  in final_prob = 1485 / 64 := by
  sorry

end yogurt_strawberry_probability_l271_271322


namespace sum_of_solutions_eq_225_l271_271574

theorem sum_of_solutions_eq_225 :
  ∑ x in Finset.filter (λ x, 0 < x ∧ x ≤ 30 ∧ (17 * (5 * x - 3)) % 10 = 4) (Finset.range 31), x = 225 :=
by sorry

end sum_of_solutions_eq_225_l271_271574


namespace angle_B_possible_values_l271_271878

variables {A B C O H : Type} -- variables representing points
variables [inner_product_space ℝ A] [inner_product_space ℝ B] 
variables [inner_product_space ℝ C] [inner_product_space ℝ O] 
variables [inner_product_space ℝ H]

-- Representations and conditions for the problem
def is_circumcenter (O : A) (ABC : Type) [has_circumcircle ABC] : Prop := sorry
def is_orthocenter (H : A) (ABC : Type) [has_orthocenter ABC] : Prop := sorry
def BO_equals_BH (O H B : A) [dist O B = dist B H] : Prop := sorry

noncomputable def angle_B (ABC : Type) [has_angles ABC] : ℝ := sorry

-- Lean theorem statement for the given problem
theorem angle_B_possible_values (O H B : Type) (ABC : Type)
  [is_circumcenter O ABC] [is_orthocenter H ABC] [BO_equals_BH O H B] 
  : angle_B ABC = 60 ∨ angle_B ABC = 120 :=
sorry

end angle_B_possible_values_l271_271878


namespace sticks_form_equilateral_triangle_l271_271699

theorem sticks_form_equilateral_triangle {n : ℕ} (h1 : n ≥ 5) (h2 : n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5) :
  (∃ a b c, a = b ∧ b = c ∧ a + b + c = (n * (n + 1)) / 2) :=
by
  sorry

end sticks_form_equilateral_triangle_l271_271699


namespace partition_into_three_summands_l271_271286

theorem partition_into_three_summands (n : ℕ) (h : n ≥ 3) : 
  (∑ k in Finset.range (n - 2), (n - 1 - k)) = (n - 2) * (n - 1) / 2 :=
by
  sorry

end partition_into_three_summands_l271_271286


namespace digit_one_bases_count_l271_271724

theorem digit_one_bases_count :
  let count_bases_with_final_digit_one := 
    (Finset.filter (λ b : ℕ, 3 ≤ b ∧ b ≤ 8 ∧ 783 % b = 0)
      (Finset.range (8 + 1))).card
  count_bases_with_final_digit_one = 1 :=
by 
  have h : Finset.filter (λ b : ℕ, 3 ≤ b ∧ b ≤ 8 ∧ 783 % b = 0)
    (Finset.range (8 + 1)) = {3} := 
    by {
      apply Finset.ext,
      intro b,
      simp only [Finset.mem_filter, Finset.mem_range, Nat.succ_le_succ_iff, lt_add_iff_pos_right, Nat.mem_range, Finset.mem_singleton],
      split_ifs,
      any_goals { simp at h_1 ⊢, assumption },
      all_goals { simp, norm_num at *, 
        repeat { rintro ⟨⟩ | ⟨_ | _⟩ },
        exact ⟨le_refl _, rfl⟩,
        all_goals { try {exfalso, exact not_le_of_lt‹_›anch, }
      monotone at specific cases. }
      detail use a casesver reach ⟨ naturally },
      literally,
     prove it to exacts point ⟩,
-- mock step verbose for clarification, feel free to customize
-- cases ⟩  },
 end⟩

      exact {
       admitted sorry 
}
 
end digit_one_bases_count_l271_271724


namespace sum_of_numerator_and_denominator_l271_271200

theorem sum_of_numerator_and_denominator : 
  let x := 3.71717171 in
  (∃ (a b : ℤ), x * (b:ℝ) = a ∧ b ≠ 0 ∧ Int.gcd a b = 1 ∧ a + b = 467) :=
by sorry

end sum_of_numerator_and_denominator_l271_271200


namespace min_sum_of_arith_seq_l271_271081

theorem min_sum_of_arith_seq (a : ℕ → ℤ) (d : ℤ) (h1 : abs (a 5) = abs (a 11)) (h2 : d > 0) :
  ∃ n : ℕ, n = 7 ∧ (∀ m : ℕ, m ≠ 7 → S_n a m > S_n a 7) :=
begin
  sorry
end

-- Definition of the sum of the first n terms
def S_n (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (finset.range n).sum a

end min_sum_of_arith_seq_l271_271081


namespace solution_set_l271_271137

def f (x : ℝ) : ℝ := x^2 - 4 * x + 3
def g (x : ℝ) : ℝ := 3^x - 2

theorem solution_set (x : ℝ) : f (g x) > 0 ↔ x ∈ set.Iio 1 ∪ set.Ioi (real.log 5 / real.log 3) := 
by
  sorry

end solution_set_l271_271137


namespace perpendicular_circumcircle_l271_271991

variables {A1 B1 C1 A B C : Type} [metric_space A1] [metric_space B1] [metric_space C1]

-- Assuming the circles centered at A1, B1, C1 touch each other pairwise at points A, B, and C.
def touches_pairwise (p q r : Type) [metric_space p] [metric_space q] [metric_space r] : Prop :=
  ∃ (A B C : Type), 
    (dist A B = dist A C) ∧ 
    (dist B A = dist B C) ∧ 
    (dist C A = dist C B)

-- Prove that the circumcircle of triangle formed by touching points is perpendicular to each original circle.
theorem perpendicular_circumcircle
  (h : touches_pairwise A1 B1 C1) :
  ∃ P : Type, (circumcircle P A B C) ∧ ∀ (X ∈ {A1, B1, C1}), perpendicular_at (circle X) (circumcircle P) :=
sorry

end perpendicular_circumcircle_l271_271991


namespace part1_solution_set_part2_range_of_a_l271_271019

def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1_solution_set (x : ℝ) : 
  (f x 2 ≥ 4) ↔ (x ≤ 3 / 2 ∨ x ≥ 11 / 2) :=
sorry

theorem part2_range_of_a (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 4) → (a ≤ -1 ∨ a ≥ 3) :=
sorry

end part1_solution_set_part2_range_of_a_l271_271019


namespace Q_contribution_l271_271156

def P_contribution : ℕ := 4000
def P_months : ℕ := 12
def Q_months : ℕ := 8
def profit_ratio_PQ : ℚ := 2 / 3

theorem Q_contribution :
  ∃ X : ℕ, (P_contribution * P_months) / (X * Q_months) = profit_ratio_PQ → X = 9000 := 
by sorry

end Q_contribution_l271_271156


namespace traveling_zoo_l271_271080

theorem traveling_zoo (x y : ℕ) (h1 : x + y = 36) (h2 : 4 * x + 6 * y = 100) : x = 14 ∧ y = 22 :=
by {
  sorry
}

end traveling_zoo_l271_271080


namespace BD_over_BO_value_l271_271076

-- Given conditions as definitions
def centered_at_circle (O : Point) (A : Point) (circle : Circle) : Prop := circle.center = O ∧ A ∈ circle
def tangent_to_circle (BA : Line) (A : Point) (circle : Circle) : Prop := BA.tangent_at A circle
def right_triangle_45_degrees (ABC : Triangle) (A B : Point) : Prop := ABC.right_angle_at A ∧ ABC.angle B = 45
def circle_intersects_BO (circle : Circle) (BO : Line) (D : Point) : Prop := circle.intersect_line BO = {D}
def chord_extends_BC_to_E (BC : Segment) (E : Point) (circle : Circle) : Prop := ∃ (line_BE : Line), line_BE.contains BC ∧ line_BE ∩ circle = {BC.start, E}

-- Main statement to prove
theorem BD_over_BO_value
  (circle : Circle) (O A B C D E : Point)
  (BO : Line) (BC : Segment)
  (h1 : centered_at_circle O A circle)
  (h2 : tangent_to_circle (line B A) A circle)
  (h3 : right_triangle_45_degrees (triangle A B C) A B)
  (h4 : circle_intersects_BO circle BO D)
  (h5 : chord_extends_BC_to_E (segment B C) E circle) :
  BD / BO = (2 - sqrt 2) / 2 := sorry

end BD_over_BO_value_l271_271076


namespace probability_sum_less_than_or_equal_16_l271_271992

def is_fair_six_sided_die (n : ℕ) : Prop := n ∈ finset.range 7

theorem probability_sum_less_than_or_equal_16 :
  let S := (finset.range 7).product (finset.range 7).product (finset.range 7)
  (S.filter (λ xyz, xyz.1 + xyz.2.1 + xyz.2.2 ≤ 16)).card.to_rat / S.card.to_rat = 53 / 54 := sorry

end probability_sum_less_than_or_equal_16_l271_271992


namespace b6_b8_value_l271_271741

def arithmetic_seq (a : ℕ → ℕ) := ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d
def nonzero_sequence (a : ℕ → ℕ) := ∀ n : ℕ, a n ≠ 0
def geometric_seq (b : ℕ → ℕ) := ∃ r : ℕ, ∀ n : ℕ, b (n + 1) = b n * r

theorem b6_b8_value (a b : ℕ → ℕ) (d : ℕ) 
  (h_arith : arithmetic_seq a) 
  (h_nonzero : nonzero_sequence a) 
  (h_cond1 : 2 * a 3 = a 1^2) 
  (h_cond2 : a 1 = d)
  (h_geo : geometric_seq b)
  (h_b13 : b 13 = a 2)
  (h_b1 : b 1 = a 1) :
  b 6 * b 8 = 72 := 
sorry

end b6_b8_value_l271_271741


namespace train_braking_time_and_distance_l271_271644

theorem train_braking_time_and_distance 
  (v0 : ℝ) (a : ℝ) 
  (h_v0 : v0 = 30) 
  (h_a : a = -0.5) :
  let t := 60 in 
  let s := 900 in 
  (v0 + a * t = 0) ∧ (s = ∫ x in 0..t, (v0 + a * x)) :=
by 
  sorry

end train_braking_time_and_distance_l271_271644


namespace volume_pyramid_proof_l271_271436

noncomputable def volume_pyramid (AB BC CG : ℝ) (M : ℝ × ℝ × ℝ) : ℝ :=
  let EB := real.sqrt (AB^2 + BC^2 + (CG - 1)^2)
  let base_area := BC * EB
  let height := (M.2.2 : ℝ) -- Third coordinate of M
  (1 / 3) * base_area * height

theorem volume_pyramid_proof :
  volume_pyramid 4 1 2 (0,1 / 3,4 / 3) = 4 * real.sqrt 2 / 3 :=
by
  sorry

end volume_pyramid_proof_l271_271436


namespace value_of_v1_using_Horner_method_l271_271395

theorem value_of_v1_using_Horner_method :
  ∀ (x : ℝ), 
  ∀ (f : ℝ → ℝ),
  (f x = 4 * x^5 + 2 * x^4 + 3.5 * x^3 - 2.6 * x^2 + 1.7 * x - 0.8) →
  (f x = (((((4 * x + 2) * x + 3.5) * x - 2.6) * x + 1.7) * x - 0.8)) →
  ∃ (v0 v1 : ℝ), v0 = 4 ∧ v1 = v0 * 5 + 2 ∧ v1 = 22 :=
begin
  intro x,
  intro f,
  intros hf1 hf2,
  use [4, 4 * 5 + 2],
  split,
  { refl },
  split,
  { refl },
  { sorry },
end

end value_of_v1_using_Horner_method_l271_271395


namespace arithmetic_geometric_seq_range_l271_271836

theorem arithmetic_geometric_seq_range (x a1 a2 y b1 b2 : ℝ) 
  (h1 : a1 + a2 = x + y) (h2 : x * y = b1 * b2) :
  ∃ z : ℝ, (z = (x + y)^2 / (x * y)) ∧ (z ∈ Icc 4 (+∞) ∨ z ∈ Icc (-∞) 0) :=
sorry

end arithmetic_geometric_seq_range_l271_271836


namespace proof_problem1_proof_problem2_l271_271665

noncomputable def problem1 : Prop :=
  (2^(-1 / 2) + ( -4 )^0 / sqrt 2 + 1 / ( sqrt 2 - 1 ) - sqrt ( 1 - sqrt 5 )^0) = 2 * sqrt 2

noncomputable def problem2 : Prop :=
  (log 2 25) * (log 3 (1 / 16)) * (log 5 (1 / 9)) = 16

theorem proof_problem1 : problem1 := by
  sorry

theorem proof_problem2 : problem2 := by
  sorry

end proof_problem1_proof_problem2_l271_271665


namespace sum_of_even_conditions_l271_271176

theorem sum_of_even_conditions (m n : ℤ) :
  ((∃ k l : ℤ, m = 2 * k ∧ n = 2 * l) → ∃ p : ℤ, m + n = 2 * p) ∧
  (∃ q : ℤ, m + n = 2 * q → (∃ k l : ℤ, m = 2 * k ∧ n = 2 * l) → False) :=
by
  sorry

end sum_of_even_conditions_l271_271176


namespace sum_of_cubes_equality_l271_271714

theorem sum_of_cubes_equality (a b p n : ℕ) (hp : Nat.Prime p) (ha : a > 0) (hb : b > 0) (hn : n > 0) :
  (a^3 + b^3 = p^n) ↔ 
  (∃ k : ℕ, a = 2^k ∧ b = 2^k ∧ p = 2 ∧ n = 3*k + 1) ∨
  (∃ k : ℕ, a = 3^k ∧ b = 2 * 3^k ∧ p = 3 ∧ n = 3*k + 2) ∨
  (∃ k : ℕ, a = 2 * 3^k ∧ b = 3^k ∧ p = 3 ∧ n = 3*k + 2) := sorry

end sum_of_cubes_equality_l271_271714


namespace probability_reach_2C_l271_271273

noncomputable def f (x C : ℝ) : ℝ :=
  x / (2 * C)

theorem probability_reach_2C (x C : ℝ) (hC : 0 < C) (hx : 0 < x ∧ x < 2 * C) :
  f x C = x / (2 * C) := 
by
  sorry

end probability_reach_2C_l271_271273


namespace acute_triangle_inequality_l271_271294

noncomputable def inequality_proof {A B C D E F H X Y Z : Type*}
  (triangle_ABC : ∀ {A B C : Type}, Prop)
  (altitudes_meet_H : ∀ {A B C D E F H : Type}, Prop)
  (circle_meets_AD_BE_CF : ∀ {D E F X Y Z : Type}, Prop)
  (AH DX BH EY CH FZ : ℝ) : Prop :=
  triangle_ABC ∧ altitudes_meet_H ∧ circle_meets_AD_BE_CF → 
  ∀ (AH DX BH EY CH FZ : ℝ), 
    AH / DX + BH / EY + CH / FZ ≥ 3

theorem acute_triangle_inequality 
  (triangle_ABC : ∀ {A B C : Type}, Prop)
  (altitudes_meet_H : ∀ {A B C D E F H : Type}, Prop)
  (circle_meets_AD_BE_CF : ∀ {D E F X Y Z : Type}, Prop)
  (AH DX BH EY CH FZ : ℝ) : 
  triangle_ABC ∧ altitudes_meet_H ∧ circle_meets_AD_BE_CF → 
  ∀ (AH DX BH EY CH FZ : ℝ), 
    AH / DX + BH / EY + CH / FZ ≥ 3 :=
by
  intro h
  sorry

end acute_triangle_inequality_l271_271294


namespace part1_solution_set_part2_values_of_a_l271_271034

-- Parameters and definitions for the function and conditions
def f (x a : ℝ) := |x - a^2| + |x - (2*a + 1)|

-- Part (1) When a = 2, the inequality's solution set
theorem part1_solution_set (x : ℝ) : f x 2 ≥ 4 ↔ (x ≤ 3/2 ∨ x ≥ 11/2) := 
sorry

-- Part (2) Range of values for a given f(x, a) ≥ 4
theorem part2_values_of_a (a : ℝ) : 
∀ x, f x a ≥ 4 → (a ≤ -1 ∨ a ≥ 3) :=
sorry

end part1_solution_set_part2_values_of_a_l271_271034


namespace translation_A_to_A_l271_271861

-- Define the initial point A
def A : ℝ × ℝ := (-2, 3)

-- Define the translated point A' after specified transformations
def A' : ℝ × ℝ :=
  let A1_y := snd A - 3 // Y-coordinate after moving down by 3
  let A2_x := fst A + 4 // X-coordinate after moving right by 4
  (A2_x, A1_y)

-- The theorem to prove
theorem translation_A_to_A' : A' = (2, 0) := by
  sorry

end translation_A_to_A_l271_271861


namespace jaclyn_constant_term_l271_271143

variable {R : Type*} [CommRing R] (P Q : Polynomial R)

theorem jaclyn_constant_term (hP : P.leadingCoeff = 1) (hQ : Q.leadingCoeff = 1)
  (deg_P : P.degree = 4) (deg_Q : Q.degree = 4)
  (constant_terms_eq : P.coeff 0 = Q.coeff 0)
  (coeff_z_eq : P.coeff 1 = Q.coeff 1)
  (product_eq : P * Q = Polynomial.C 1 * 
    Polynomial.C 1 * Polynomial.C 1 * Polynomial.C (-1) *
    Polynomial.C 1) :
  Jaclyn's_constant_term = 3 :=
sorry

end jaclyn_constant_term_l271_271143


namespace distance_from_A_to_origin_l271_271443

def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem distance_from_A_to_origin :
  distance (-6) 8 0 0 = 10 :=
by {
  -- The proof starts here, but we leave it as 'sorry' as per the instruction
  sorry 
}

end distance_from_A_to_origin_l271_271443


namespace hyperbola_eccentricity_eq_sqrt_5_l271_271040

theorem hyperbola_eccentricity_eq_sqrt_5
  (a b : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (h : ∀ a b : ℝ, (|b * c| / sqrt(a^2 + b^2) = 2 * a)) :
  sqrt(1 + (b / a)^2) = sqrt(5) := 
by sorry

end hyperbola_eccentricity_eq_sqrt_5_l271_271040


namespace inequality_proof_l271_271124

noncomputable def inequality (a b c : ℝ) (ha: a > 1) (hb: b > 1) (hc: c > 1) : Prop :=
  (a * b) / (c - 1) + (b * c) / (a - 1) + (c * a) / (b - 1) >= 12

theorem inequality_proof (a b c : ℝ) (ha: a > 1) (hb: b > 1) (hc: c > 1) : inequality a b c ha hb hc :=
by
  sorry

end inequality_proof_l271_271124
